"""Tests for src/fine_tune/config.py."""
from pathlib import Path

import pytest

from fine_tune.config import RunConfig, load_config

CONFIG_PATH = Path("src/fine_tune/configs/qwen3.5-9b.yaml")


def test_load_config_returns_run_config():
    cfg = load_config(CONFIG_PATH)
    assert isinstance(cfg, RunConfig)


def test_load_config_parses_model_section():
    cfg = load_config(CONFIG_PATH)
    assert cfg.model.id == "Qwen/Qwen3.5-9B"
    assert cfg.model.torch_dtype == "bfloat16"
    assert cfg.model.trust_remote_code is True


def test_load_config_parses_lora_section():
    cfg = load_config(CONFIG_PATH)
    assert cfg.lora.r == 64
    assert cfg.lora.alpha == 128
    assert "q_proj" in cfg.lora.target_modules
    assert cfg.lora.freeze_vision_tower is True


def test_load_config_parses_training_section():
    cfg = load_config(CONFIG_PATH)
    assert cfg.training.num_train_epochs == 5
    assert cfg.training.per_device_train_batch_size == 1
    assert cfg.training.gradient_accumulation_steps == 16
    assert cfg.training.optim == "paged_adamw_8bit"
    assert cfg.training.learning_rate == pytest.approx(1e-4)


def test_load_config_parses_data_section():
    cfg = load_config(CONFIG_PATH)
    assert cfg.data.train_file.endswith("train.full_reasoning.jsonl")
    assert cfg.data.val_file.endswith("val.full_reasoning.jsonl")
    assert cfg.data.max_seq_length == 4096


def test_load_config_parses_hub_section():
    cfg = load_config(CONFIG_PATH)
    assert cfg.hub.repo_id == "danielfdias98/qwen3.5-9b-derm-reasoning"
    assert cfg.hub.private is True


def test_load_all_four_model_configs():
    """All four checked-in model YAMLs parse into RunConfig."""
    for name in ("medgemma-1.5-4b", "gemma-4-e4b", "qwen3.5-4b", "qwen3.5-9b"):
        cfg = load_config(Path(f"src/fine_tune/configs/{name}.yaml"))
        assert cfg.model.id
        assert cfg.hub.repo_id.endswith("-derm-reasoning")
        assert cfg.training.num_train_epochs == 5


def test_raw_dict_preserved():
    """The raw YAML dict is available on .raw for Trainer kwargs passthrough."""
    cfg = load_config(CONFIG_PATH)
    assert isinstance(cfg.raw, dict)
    assert "model" in cfg.raw
    assert "training" in cfg.raw
