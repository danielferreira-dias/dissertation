"""Fine-tune a VLM using TRL SFTTrainer + PEFT LoRA.

Usage:
    # Full reasoning (default) — matches paths in the YAML
    python -m fine_tune.train --config src/fine_tune/configs/qwen3.5-9b.yaml

    # Label-only ablation — swaps data files + suffixes output_dir / hub repo
    python -m fine_tune.train --config src/fine_tune/configs/qwen3.5-9b.yaml --format label_only

    # Resume from latest checkpoint in the run's output_dir
    python -m fine_tune.train --config ... --resume

    # One forward/backward step to detect OOM / wiring errors
    python -m fine_tune.train --config ... --dry-run
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import TrainingArguments
from transformers.integrations import TensorBoardCallback
from trl import SFTTrainer

from fine_tune.callbacks import CSVLoggerCallback, HFHubPushCallback
from fine_tune.config import RunConfig, load_config
from fine_tune.data import MultimodalCollator, load_chat_dataset
from fine_tune.model_loader import load_model_and_processor


# Replace the "full_reasoning" substring in these fields when --format label_only is passed.
# Everything else in the config is unchanged.
_FORMAT_ALIASES = {
    "full_reasoning": {
        "train_file": "train.full_reasoning.jsonl",
        "val_file": "val.full_reasoning.jsonl",
        "output_suffix": "-full_reasoning",
        "hub_suffix": "-full-reasoning",
    },
    "label_only": {
        "train_file": "train.label_only.jsonl",
        "val_file": "val.label_only.jsonl",
        "output_suffix": "-label_only",
        "hub_suffix": "-label-only",
    },
}


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _set_seeds(seed: int) -> None:
    import random
    import numpy as np
    import transformers
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    transformers.set_seed(seed)


def _apply_format(cfg: RunConfig, fmt: str) -> RunConfig:
    """Swap data file paths and suffix output_dir / hub.repo_id based on fmt.

    The YAML defaults to full_reasoning. If fmt == "full_reasoning", we still
    suffix for consistency (so both runs' artifacts live in distinct dirs/repos).
    """
    if fmt not in _FORMAT_ALIASES:
        raise ValueError(f"unknown format: {fmt!r}")
    alias = _FORMAT_ALIASES[fmt]

    # Rewrite data file paths to the requested format
    train_parent = Path(cfg.data.train_file).parent
    val_parent = Path(cfg.data.val_file).parent
    cfg.data.train_file = str(train_parent / alias["train_file"])
    cfg.data.val_file = str(val_parent / alias["val_file"])

    # Suffix output_dir and hub.repo_id so the two ablation runs don't clash
    cfg.training.output_dir = cfg.training.output_dir + alias["output_suffix"]
    cfg.hub.repo_id = cfg.hub.repo_id + alias["hub_suffix"]
    return cfg


def _copy_config_into_output_dir(cfg_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / "config.yaml"
    shutil.copy(cfg_path, dest)
    return dest


def _build_training_args(cfg: RunConfig) -> TrainingArguments:
    t = cfg.training
    return TrainingArguments(
        output_dir=t.output_dir,
        num_train_epochs=t.num_train_epochs,
        per_device_train_batch_size=t.per_device_train_batch_size,
        per_device_eval_batch_size=t.per_device_train_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        learning_rate=t.learning_rate,
        warmup_ratio=t.warmup_ratio,
        lr_scheduler_type=t.lr_scheduler_type,
        optim=t.optim,
        weight_decay=t.weight_decay,
        bf16=t.bf16,
        gradient_checkpointing=t.gradient_checkpointing,
        logging_steps=t.logging_steps,
        eval_strategy=t.eval_strategy,
        eval_steps=t.eval_steps,
        save_strategy=t.save_strategy,
        save_total_limit=t.save_total_limit,
        load_best_model_at_end=t.load_best_model_at_end,
        metric_for_best_model=t.metric_for_best_model,
        greater_is_better=t.greater_is_better,
        seed=t.seed,
        report_to=t.report_to,
        remove_unused_columns=False,
    )


def _append_manifest_row(
    cfg: RunConfig,
    fmt: str,
    cfg_sha: str,
    wall_clock_hours: float,
    best_loss: float | None,
) -> None:
    manifest = Path("/workspace/runs/manifest.csv")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    is_new = not manifest.exists()
    with manifest.open("a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow([
                "timestamp", "model", "format", "config_sha",
                "best_eval_loss", "wall_clock_hours", "hub_repo",
            ])
        w.writerow([
            datetime.utcnow().isoformat(timespec="seconds"),
            cfg.model.id,
            fmt,
            cfg_sha,
            f"{best_loss:.4f}" if best_loss is not None else "n/a",
            f"{wall_clock_hours:.2f}",
            cfg.hub.repo_id,
        ])


def _dry_run(trainer: SFTTrainer) -> None:
    """One fwd+bwd step to detect OOM / wiring errors before a full epoch."""
    print("[dry-run] Fetching one batch + executing one training step...")
    loader = trainer.get_train_dataloader()
    batch = next(iter(loader))
    trainer.model.train()
    device = trainer.model.device
    inputs = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
    outputs = trainer.model(**inputs)
    loss = outputs.loss
    loss.backward()
    print(f"[dry-run] OK. loss={loss.item():.4f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--format", choices=list(_FORMAT_ALIASES.keys()),
                   default="full_reasoning",
                   help="Which ablation variant to train on")
    p.add_argument("--resume", action="store_true",
                   help="Resume from latest checkpoint in output_dir")
    p.add_argument("--dry-run", action="store_true",
                   help="One fwd+bwd step, no real training")
    args = p.parse_args()

    cfg = load_config(args.config)
    cfg = _apply_format(cfg, args.format)

    _set_seeds(cfg.training.seed)

    output_dir = Path(cfg.training.output_dir)
    saved_cfg_path = _copy_config_into_output_dir(args.config, output_dir)

    print(f"Loading base model + processor: {cfg.model.id}")
    model, processor = load_model_and_processor(cfg.model, cfg.lora)

    print(f"Loading data: train={cfg.data.train_file} val={cfg.data.val_file}")
    train_ds = load_chat_dataset(Path(cfg.data.train_file))
    val_ds = load_chat_dataset(Path(cfg.data.val_file))

    collator = MultimodalCollator(processor=processor, max_seq_length=cfg.data.max_seq_length)

    training_args = _build_training_args(cfg)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=getattr(processor, "tokenizer", processor),
    )

    trainer.add_callback(CSVLoggerCallback(output_path=output_dir / "metrics.csv"))
    trainer.add_callback(HFHubPushCallback(
        repo_id=cfg.hub.repo_id,
        private=cfg.hub.private,
        config_yaml_path=saved_cfg_path,
    ))
    trainer.add_callback(TensorBoardCallback())

    if args.dry_run:
        _dry_run(trainer)
        return

    start = time.time()
    trainer.train(resume_from_checkpoint=args.resume)
    wall_clock_hours = (time.time() - start) / 3600

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))

    best_loss = trainer.state.best_metric
    _append_manifest_row(cfg, args.format, _git_sha(), wall_clock_hours, best_loss)

    print(f"Training complete. Best eval_loss={best_loss}, wall={wall_clock_hours:.2f}h")
    print(f"Final adapter: {final_dir}")
    print(f"HF Hub repo:   https://huggingface.co/{cfg.hub.repo_id}")


if __name__ == "__main__":
    main()
