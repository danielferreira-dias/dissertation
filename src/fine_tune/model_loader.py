"""Load base VLM + tokenizer + apply LoRA, all via Unsloth's FastVisionModel.

Replaces the previous AutoModelForImageTextToText + manual peft.get_peft_model
flow with Unsloth's batched-aware path. Key differences:

  - FastVisionModel.from_pretrained must be called BEFORE any other CUDA op
    (it patches CUDA kernels in-place).
  - Vision-tower freezing is a flag on get_peft_model, not a manual loop.
  - Unsloth bundles Triton kernels for fla / chunk_gated_delta_rule, so
    Qwen 3.5's slow-path fp32 fallback no longer applies.
"""
from __future__ import annotations

from typing import Any

import torch

from fine_tune.config import LoraConfig, ModelConfig

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _dtype_from_str(s: str) -> torch.dtype:
    try:
        return _DTYPE_MAP[s]
    except KeyError as e:
        raise ValueError(f"unsupported torch_dtype: {s!r}") from e


def load_model_and_processor(model_cfg: ModelConfig, lora_cfg: LoraConfig):
    """Load base VLM + tokenizer/processor, apply Unsloth LoRA, return both.

    Returns `(model, tokenizer)`. For VLMs, `tokenizer` is actually a processor
    that quacks like a tokenizer (has .tokenizer, .image_processor, etc).
    """
    # IMPORTANT: import + call FastVisionModel before any other CUDA op.
    from unsloth import FastVisionModel

    dtype = _dtype_from_str(model_cfg.torch_dtype)

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=model_cfg.id,
        max_seq_length=getattr(model_cfg, "max_seq_length", 4096),
        load_in_4bit=getattr(model_cfg, "load_in_4bit", False),
        dtype=dtype,
        trust_remote_code=model_cfg.trust_remote_code,
        # Unsloth selects flash-attn / SDPA internally per model; don't override
        # unless we have a known reason. Honour the YAML if user pinned one.
        # attn_implementation=model_cfg.attn_implementation,
    )

    # target_modules accepts a list of suffixes, "all-linear", or a regex.
    target_modules: Any = lora_cfg.target_modules
    if isinstance(target_modules, list) and len(target_modules) == 0:
        target_modules = "all-linear"

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=not lora_cfg.freeze_vision_tower,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=lora_cfg.r,
        lora_alpha=lora_cfg.alpha,
        lora_dropout=lora_cfg.dropout,
        bias=lora_cfg.bias,
        target_modules=target_modules,
        modules_to_save=lora_cfg.modules_to_save or None,
        use_rslora=False,
        loftq_config=None,
        random_state=42,
    )

    # Print a concise sanity line: trainable %, vision-frozen, rank.
    model.print_trainable_parameters()
    print(
        f"[unsloth] r={lora_cfg.r} alpha={lora_cfg.alpha} "
        f"freeze_vision_tower={lora_cfg.freeze_vision_tower} "
        f"target_modules={target_modules!r}"
    )
    return model, tokenizer
