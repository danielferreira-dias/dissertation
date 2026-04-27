"""TrainerCallbacks for CSV logging and HF Hub push."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from transformers import TrainerCallback


class CSVLoggerCallback(TrainerCallback):
    """Writes step-level metrics to a CSV on the volume.

    Duplicates TensorBoard logs in plain-text form for dissertation tables.
    Inherits the no-op default lifecycle hooks (on_train_begin, on_step_end,
    etc.) so transformers 5.x's full callback dispatch doesn't AttributeError.
    """

    HEADER = ["step", "epoch", "loss", "eval_loss", "learning_rate"]

    def __init__(self, output_path: Path):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._written_header = self.output_path.exists()

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        if not logs:
            return
        epoch = getattr(state, "epoch", None)
        row = [
            str(state.global_step),
            f"{epoch:.4f}" if epoch is not None else "",
            str(logs.get("loss", "")),
            str(logs.get("eval_loss", "")),
            str(logs.get("learning_rate", "")),
        ]
        new_file = not self._written_header
        with self.output_path.open("a", newline="") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(self.HEADER)
                self._written_header = True
            w.writerow(row)


class HFHubPushCallback(TrainerCallback):
    """At on_train_end, push the (best, already-loaded) PEFT adapter to HF Hub.

    Uploads only the adapter (~100-400 MB), not base weights. Also uploads
    the training config.yaml as a repo file for reproducibility.
    """

    def __init__(self, repo_id: str, private: bool, config_yaml_path: Path):
        self.repo_id = repo_id
        self.private = private
        self.config_yaml_path = Path(config_yaml_path)

    def on_train_end(self, args: Any, state: Any, control: Any, model=None, **kwargs) -> None:
        if model is None:
            return
        best_epoch = _best_epoch_from_state(state)
        best_loss = state.best_metric if getattr(state, "best_metric", None) is not None else "n/a"
        commit_msg = f"Fine-tuned on structured reasoning — best epoch {best_epoch}, eval_loss {best_loss}"
        model.push_to_hub(self.repo_id, private=self.private, commit_message=commit_msg)
        self._upload_config(commit_msg)

    def _upload_config(self, commit_msg: str) -> None:
        from huggingface_hub import upload_file
        if not self.config_yaml_path.exists():
            return
        upload_file(
            path_or_fileobj=str(self.config_yaml_path),
            path_in_repo="config.yaml",
            repo_id=self.repo_id,
            commit_message=f"Add training config — {commit_msg}",
        )


def _best_epoch_from_state(state: Any) -> str:
    history = getattr(state, "log_history", []) or []
    best: tuple[Any, float] | None = None
    for row in history:
        if "eval_loss" in row:
            if best is None or row["eval_loss"] < best[1]:
                best = (row.get("epoch", "?"), row["eval_loss"])
    return str(best[0]) if best else "?"
