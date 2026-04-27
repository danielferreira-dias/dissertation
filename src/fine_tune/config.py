"""Typed config loader. YAML -> RunConfig dataclass."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    id: str
    trust_remote_code: bool = True
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"


@dataclass
class LoraConfig:
    r: int
    alpha: int
    dropout: float
    bias: str
    target_modules: list[str] | str
    modules_to_save: list[str] = field(default_factory=list)
    freeze_vision_tower: bool = True


@dataclass
class TrainingConfig:
    output_dir: str
    num_train_epochs: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_ratio: float
    lr_scheduler_type: str
    optim: str
    weight_decay: float
    bf16: bool
    gradient_checkpointing: bool
    logging_steps: int
    eval_strategy: str
    eval_steps: int
    save_strategy: str
    save_total_limit: int
    load_best_model_at_end: bool
    metric_for_best_model: str
    greater_is_better: bool
    seed: int
    report_to: list[str]


@dataclass
class DataConfig:
    train_file: str
    val_file: str
    max_seq_length: int
    image_max_pixels: int


@dataclass
class HubConfig:
    repo_id: str
    push_strategy: str = "end"
    private: bool = True


@dataclass
class RunConfig:
    model: ModelConfig
    lora: LoraConfig
    training: TrainingConfig
    data: DataConfig
    hub: HubConfig
    raw: dict[str, Any] = field(default_factory=dict)


def load_config(path: Path) -> RunConfig:
    """Load a YAML config file and return a typed RunConfig."""
    raw = yaml.safe_load(Path(path).read_text())
    return RunConfig(
        model=ModelConfig(**raw["model"]),
        lora=LoraConfig(**raw["lora"]),
        training=TrainingConfig(**raw["training"]),
        data=DataConfig(**raw["data"]),
        hub=HubConfig(**raw["hub"]),
        raw=raw,
    )
