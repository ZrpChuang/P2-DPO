"""P2-DPO losses aligned with the paper objective.

The functions in this file operate on sequence log-probabilities. The trainer is
responsible for running the policy/reference models under the required visual
contexts and passing the resulting log-probabilities here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class P2DPOLossConfig:
    """Hyperparameters from the P2-DPO objective."""

    beta: float = 0.1
    lambda_calib: float = 0.3
    dynamic_loss_weighting: bool = True
    w_base: float = 0.5
    alpha_max: float = 0.2
    tau: float = 0.1
    eps: float = 1e-8


@dataclass
class P2DPOLossInputs:
    """Sequence log-probabilities needed by the paper losses.

    Each tensor should be shaped ``(batch,)`` and should already be reduced over
    the answer tokens, either by summation or by the same averaging convention
    used elsewhere in training.
    """

    focus_policy_win: Optional[Tensor] = None
    focus_policy_lose: Optional[Tensor] = None
    focus_ref_win: Optional[Tensor] = None
    focus_ref_lose: Optional[Tensor] = None
    calib_policy_win_aug: Optional[Tensor] = None
    calib_policy_win_deg: Optional[Tensor] = None
    calib_ref_lose_aug: Optional[Tensor] = None
    calib_ref_lose_deg: Optional[Tensor] = None
    robust_policy_win: Optional[Tensor] = None
    robust_policy_lose: Optional[Tensor] = None
    robust_ref_win: Optional[Tensor] = None
    robust_ref_lose: Optional[Tensor] = None
    clip_crop_score: Optional[Tensor] = None
    clip_image_score: Optional[Tensor] = None
    focus_weight: Optional[Tensor] = None
    robust_weight: Optional[Tensor] = None


@dataclass
class P2DPOLossOutput:
    loss: Tensor
    per_sample_loss: Tensor
    focus_dpo_loss: Optional[Tensor] = None
    calibration_loss: Optional[Tensor] = None
    focus_loss: Optional[Tensor] = None
    robust_loss: Optional[Tensor] = None
    focus_weight: Optional[Tensor] = None
    robust_weight: Optional[Tensor] = None
    focus_logits: Optional[Tensor] = None
    calibration_logits: Optional[Tensor] = None
    robust_logits: Optional[Tensor] = None


def _require(name: str, value: Optional[Tensor]) -> Tensor:
    if value is None:
        raise ValueError(f"Missing required tensor: {name}")
    return value


def _same_shape(*values: Tensor) -> None:
    shapes = {tuple(value.shape) for value in values}
    if len(shapes) != 1:
        raise ValueError(f"Expected matching tensor shapes, got {sorted(shapes)}")


def dpo_loss(
    policy_win_logps: Tensor,
    policy_lose_logps: Tensor,
    ref_win_logps: Tensor,
    ref_lose_logps: Tensor,
    beta: float,
) -> Tuple[Tensor, Tensor]:
    """Standard DPO loss used by both focus and robustness pairs.

    Implements ``-log sigmoid(beta * ((log pi_theta^w - log pi_ref^w)
    - (log pi_theta^l - log pi_ref^l)))``.
    """

    _same_shape(policy_win_logps, policy_lose_logps, ref_win_logps, ref_lose_logps)
    logits = beta * (
        (policy_win_logps - ref_win_logps)
        - (policy_lose_logps - ref_lose_logps)
    )
    return -F.logsigmoid(logits), logits


def calibration_loss(
    policy_win_aug_logps: Tensor,
    policy_win_deg_logps: Tensor,
    ref_lose_aug_logps: Tensor,
    ref_lose_deg_logps: Tensor,
    beta: float,
) -> Tuple[Tensor, Tensor]:
    """Calibration loss over perceptual confidence gains.

    Implements the paper term
    ``-log sigmoid(beta * ((log pi_theta(y_w|I,I_crop,P)
    - log pi_theta(y_w|I_deg,P)) - (log pi_ref(y_l|I,I_crop,P)
    - log pi_ref(y_l|I_deg,P))))``.
    """

    _same_shape(
        policy_win_aug_logps,
        policy_win_deg_logps,
        ref_lose_aug_logps,
        ref_lose_deg_logps,
    )
    policy_gain = policy_win_aug_logps - policy_win_deg_logps
    ref_gain = ref_lose_aug_logps - ref_lose_deg_logps
    logits = beta * (policy_gain - ref_gain)
    return -F.logsigmoid(logits), logits


def dynamic_deficit_weights(
    clip_crop_score: Tensor,
    clip_image_score: Tensor,
    *,
    w_base: float = 0.5,
    alpha_max: float = 0.2,
    tau: float = 0.1,
    eps: float = 1e-8,
) -> Tuple[Tensor, Tensor]:
    """Compute DDW weights from the CLIPScore ratio in the paper."""

    _same_shape(clip_crop_score, clip_image_score)
    ratio = clip_crop_score / clip_image_score.clamp_min(eps)
    alpha = alpha_max * torch.tanh((ratio - 1.0) / tau)
    focus_weight = w_base + alpha
    robust_weight = w_base - alpha
    return focus_weight, robust_weight


def p2_dpo_loss(
    inputs: P2DPOLossInputs,
    config: P2DPOLossConfig = P2DPOLossConfig(),
) -> P2DPOLossOutput:
    """Compute the unified P2-DPO objective.

    Supports full P2-DPO batches containing both focus and robustness pairs, and
    also supports ablation batches containing only one pair type.
    """

    focus_dpo_losses = None
    focus_logits = None
    if inputs.focus_policy_win is not None:
        focus_dpo_losses, focus_logits = dpo_loss(
            _require("focus_policy_win", inputs.focus_policy_win),
            _require("focus_policy_lose", inputs.focus_policy_lose),
            _require("focus_ref_win", inputs.focus_ref_win),
            _require("focus_ref_lose", inputs.focus_ref_lose),
            beta=config.beta,
        )

    calib_losses = None
    calib_logits = None
    if inputs.calib_policy_win_aug is not None:
        calib_losses, calib_logits = calibration_loss(
            _require("calib_policy_win_aug", inputs.calib_policy_win_aug),
            _require("calib_policy_win_deg", inputs.calib_policy_win_deg),
            _require("calib_ref_lose_aug", inputs.calib_ref_lose_aug),
            _require("calib_ref_lose_deg", inputs.calib_ref_lose_deg),
            beta=config.beta,
        )

    if focus_dpo_losses is not None:
        focus_losses = focus_dpo_losses
        if calib_losses is not None:
            focus_losses = focus_losses + config.lambda_calib * calib_losses
    else:
        focus_losses = None

    robust_losses = None
    robust_logits = None
    if inputs.robust_policy_win is not None:
        robust_losses, robust_logits = dpo_loss(
            _require("robust_policy_win", inputs.robust_policy_win),
            _require("robust_policy_lose", inputs.robust_policy_lose),
            _require("robust_ref_win", inputs.robust_ref_win),
            _require("robust_ref_lose", inputs.robust_ref_lose),
            beta=config.beta,
        )

    if focus_losses is None and robust_losses is None:
        raise ValueError("At least one of focus or robustness losses is required.")

    if focus_losses is not None and robust_losses is not None:
        if inputs.focus_weight is not None or inputs.robust_weight is not None:
            focus_weight = _require("focus_weight", inputs.focus_weight)
            robust_weight = _require("robust_weight", inputs.robust_weight)
        elif config.dynamic_loss_weighting and inputs.clip_crop_score is not None:
            focus_weight, robust_weight = dynamic_deficit_weights(
                _require("clip_crop_score", inputs.clip_crop_score),
                _require("clip_image_score", inputs.clip_image_score),
                w_base=config.w_base,
                alpha_max=config.alpha_max,
                tau=config.tau,
                eps=config.eps,
            )
        else:
            focus_weight = torch.full_like(focus_losses, config.w_base)
            robust_weight = torch.full_like(robust_losses, config.w_base)
        _same_shape(focus_losses, robust_losses, focus_weight, robust_weight)
        per_sample_loss = focus_weight * focus_losses + robust_weight * robust_losses
    elif focus_losses is not None:
        focus_weight = torch.ones_like(focus_losses)
        robust_weight = None
        per_sample_loss = focus_losses
    else:
        focus_weight = None
        robust_weight = torch.ones_like(_require("robust_losses", robust_losses))
        per_sample_loss = robust_losses

    return P2DPOLossOutput(
        loss=per_sample_loss.mean(),
        per_sample_loss=per_sample_loss,
        focus_dpo_loss=focus_dpo_losses,
        calibration_loss=calib_losses,
        focus_loss=focus_losses,
        robust_loss=robust_losses,
        focus_weight=focus_weight,
        robust_weight=robust_weight,
        focus_logits=focus_logits,
        calibration_logits=calib_logits,
        robust_logits=robust_logits,
    )
