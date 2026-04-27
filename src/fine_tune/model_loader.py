"""Load base VLM + processor + apply LoRA."""
from __future__ import annotations

from typing import Any

import torch

from fine_tune.config import LoraConfig, ModelConfig


_VISION_PREFIXES = ("vision_tower", "vision_model", "visual")


def freeze_vision_tower(model: Any) -> int:
    """Set requires_grad=False on params whose names contain a vision prefix.

    Returns the number of parameters frozen. Idempotent.
    """
    count = 0
    for name, p in model.named_parameters():
        if any(prefix in name for prefix in _VISION_PREFIXES):
            if p.requires_grad:
                count += 1
            p.requires_grad = False
    return count


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
    """Load base model + processor, optionally freeze vision, wrap with PEFT LoRA.

    HF libs are imported lazily so the `freeze_vision_tower` unit tests can run
    without transformers/peft installed.
    """
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoProcessor,
    )
    from peft import LoraConfig as PeftLoraConfig, get_peft_model

    torch_dtype = _dtype_from_str(model_cfg.torch_dtype)

    config = AutoConfig.from_pretrained(
        model_cfg.id, trust_remote_code=model_cfg.trust_remote_code,
    )
    arch = (config.architectures or ["AutoModelForCausalLM"])[0]

    # transformers 5.x consolidates Vision2Seq + ImageTextToText into the latter.
    multimodal_markers = (
        "Vision2Seq", "ImageTextToText",
        "Gemma3", "Gemma4", "MedGemma",
    )
    is_qwen_vl = "Qwen" in arch and "VL" in arch
    if is_qwen_vl or any(m in arch for m in multimodal_markers):
        loader = AutoModelForImageTextToText
    else:
        loader = AutoModelForCausalLM

    model = loader.from_pretrained(
        model_cfg.id,
        torch_dtype=torch_dtype,
        attn_implementation=model_cfg.attn_implementation,
        trust_remote_code=model_cfg.trust_remote_code,
    )
    processor = AutoProcessor.from_pretrained(
        model_cfg.id, trust_remote_code=model_cfg.trust_remote_code,
    )

    if lora_cfg.freeze_vision_tower:
        n_frozen = freeze_vision_tower(model)
        print(f"Froze {n_frozen} vision-tower parameters.")

    peft_cfg = PeftLoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.alpha,
        lora_dropout=lora_cfg.dropout,
        bias=lora_cfg.bias,
        target_modules=lora_cfg.target_modules,
        modules_to_save=lora_cfg.modules_to_save or None,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    return model, processor
