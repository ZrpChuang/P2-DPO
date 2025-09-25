import copy  #深拷贝，浅拷贝
import inspect
import warnings #警告信息
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn

from transformers.generation.logits_process import (
    LogitsProcessorList,
)#处理概率分布的

from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
import transformers
from transformers.generation.utils import SampleOutput



def sample(
    self,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    logits_warper: Optional[LogitsProcessorList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> Union[SampleOutput, torch.LongTensor]:
    # --- [初始化部分] ---
    # 这部分代码与原始版本基本一致，主要是设置各种默认值
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id

    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None

    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )
    
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
    this_peer_finished = False

    # 为CD逻辑准备一个独立的kwargs副本，这很重要，因为它有自己的past_key_values
    model_kwargs_cd = None

    # --- [自回归生成循环] ---
    while True:
        if synced_gpus:
            # synced_gpus logic (unchanged)
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            if this_peer_finished_flag.item() == 0.0:
                break

        # 准备主模型的输入
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # 主模型前向传播
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if synced_gpus and this_peer_finished:
            continue

        next_token_logits = outputs.logits[:, -1, :]

        # --- [核心修改] VCD / Noisy / Normal 生成逻辑 ---
        
        # 从 self.generation_config 获取参数，这是解决ValueError的关键
        generation_mode = getattr(self.generation_config, "generation_mode", "normal")
        images_cd = model_kwargs.get("images_cd", None)
        
        # 只有在需要 'noisy' 或 'contrastive' 模式，并且传入了 images_cd 时，才进行第二次前向传播
        if generation_mode in ['noisy', 'contrastive'] and images_cd is not None:
            # 第一次进入时，为CD创建独立的 model_kwargs 副本
            if model_kwargs_cd is None:
                model_kwargs_cd = self._prepare_model_kwargs_for_cd(model_kwargs)

            # 假设模型类有 prepare_inputs_for_generation_cd 方法
            # 如果没有，可能需要用 self.prepare_inputs_for_generation
            if hasattr(self, 'prepare_inputs_for_generation_cd'):
                 model_inputs_cd = self.prepare_inputs_for_generation_cd(input_ids, **model_kwargs_cd)
            else:
                 # Fallback if cd-specific method does not exist
                 model_inputs_cd = self.prepare_inputs_for_generation(input_ids, **model_kwargs_cd)

            outputs_cd = self(
                **model_inputs_cd,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )
            next_token_logits_cd = outputs_cd.logits[:, -1, :]

            if generation_mode == 'noisy':
                # 模式1: 纯噪声图片回答
                next_token_scores = logits_processor(input_ids, next_token_logits_cd)
                next_token_scores = logits_warper(input_ids, next_token_scores)
            
            elif generation_mode == 'contrastive':
                # 模式2: 对比解码回答
                cd_alpha = getattr(self.generation_config, "cd_alpha", 1.0)
                cd_beta = getattr(self.generation_config, "cd_beta", 0.1)

                cutoff = torch.log(torch.tensor(cd_beta, device=next_token_logits.device)) + next_token_logits.max(dim=-1, keepdim=True).values
                diffs = (1 + cd_alpha) * next_token_logits - cd_alpha * next_token_logits_cd
                cd_logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))

                next_token_scores = logits_processor(input_ids, cd_logits)
                next_token_scores = logits_warper(input_ids, cd_logits)

        else:
            # 模式3: 正常回答 (默认行为)
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)
        
        # --- [采样和后续处理] ---
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

        # 当需要时，存储 scores, attentions, hidden_states (unchanged)
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)
            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # 对已完成的序列使用 pad_token (unchanged)
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # 更新 input_ids 和 model_kwargs (关键修改)
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())

        # 更新主模型的 past_key_values
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )
        
        # 如果使用了CD，也必须更新CD模型的 past_key_values
        if generation_mode in ['noisy', 'contrastive'] and images_cd is not None:
            model_kwargs_cd = self._update_model_kwargs_for_generation(
                outputs_cd, model_kwargs_cd, is_encoder_decoder=self.config.is_encoder_decoder
            )

        # 检查停止条件 (unchanged)
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )
            if unfinished_sequences.max() == 0:
                this_peer_finished = True

        if stopping_criteria(input_ids, scores):
            this_peer_finished = True

        if this_peer_finished and not synced_gpus:
            break

    # --- [返回结果] ---
    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return SampleEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return SampleDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
    else:
        return input_ids

# 还需要一个辅助函数来正确处理 kwargs 的复制
# 建议在模型类中定义，或者在这里定义然后动态添加到模型实例上
# 这里我们直接在 sample 函数外部定义它，并假设可以访问到 self
def _prepare_model_kwargs_for_cd(self, model_kwargs):
    """Creates a deep copy of model_kwargs for contrastive decoding, ensuring tensors are copied."""
    # This is a simplified version. A robust implementation should handle different types.
    kwargs_cd = {}
    for key, value in model_kwargs.items():
        if isinstance(value, torch.Tensor):
            kwargs_cd[key] = value.clone()
        elif isinstance(value, (list, dict)):
            kwargs_cd[key] = copy.deepcopy(value)
        else:
            kwargs_cd[key] = value
    return kwargs_cd

# 在猴子补丁时，也把这个辅助函数加上
def evolve_vcd_sampling():
    transformers.generation.utils.GenerationMixin.sample = sample
    transformers.generation.utils.GenerationMixin._sample = sample
    # 动态地将辅助函数添加到Mixin类中，这样在sample内部就能用 self._prepare_model_kwargs_for_cd
    transformers.generation.utils.GenerationMixin._prepare_model_kwargs_for_cd = _prepare_model_kwargs_for_cd
    
    # 你的 LLaVA 模型可能没有 prepare_inputs_for_generation_cd, 我们需要确认
    # 如果没有，在 sample 函数中已经有了 fallback 逻辑，所以问题不大