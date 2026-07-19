"""P2-DPO training entry helpers.

This module keeps the paper-aligned loss import path used by the shell scripts:
``llava_dpo.train.p2_dpo_train``. The full model/data training loop should call
``compute_p2_dpo_loss`` after collecting the required sequence log-probabilities.
"""

from __future__ import annotations

import argparse
from typing import Optional

from .p2_dpo_loss import P2DPOLossConfig, P2DPOLossInputs, p2_dpo_loss


def add_p2_dpo_loss_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--scale_coeff", type=float, default=0.1)
    parser.add_argument("--lambda_cal", type=float, default=0.3)
    parser.add_argument("--dynamic_loss_weighting", type=str, default="True")
    parser.add_argument("--ddw_base_weight", type=float, default=0.5)
    parser.add_argument("--ddw_alpha_max", type=float, default=0.2)
    parser.add_argument("--ddw_tau", type=float, default=0.1)
    return parser


def str_to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


def config_from_args(args: argparse.Namespace) -> P2DPOLossConfig:
    return P2DPOLossConfig(
        beta=float(getattr(args, "scale_coeff", 0.1)),
        lambda_calib=float(getattr(args, "lambda_cal", 0.3)),
        dynamic_loss_weighting=str_to_bool(
            getattr(args, "dynamic_loss_weighting", True)
        ),
        w_base=float(getattr(args, "ddw_base_weight", 0.5)),
        alpha_max=float(getattr(args, "ddw_alpha_max", 0.2)),
        tau=float(getattr(args, "ddw_tau", 0.1)),
    )


def compute_p2_dpo_loss(
    loss_inputs: P2DPOLossInputs,
    args: Optional[argparse.Namespace] = None,
):
    config = config_from_args(args or argparse.Namespace())
    return p2_dpo_loss(loss_inputs, config)


def main() -> None:
    raise SystemExit(
        "This repository exposes the paper-aligned P2-DPO loss. "
        "Wire compute_p2_dpo_loss into the local LLaVA training loop before "
        "launching distributed training."
    )


if __name__ == "__main__":
    main()
