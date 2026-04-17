"""Tests for CSVLoggerCallback (HF Hub callback is verified on pod)."""
import csv
from pathlib import Path
from unittest.mock import MagicMock

from fine_tune.callbacks import CSVLoggerCallback


def test_csv_logger_writes_header_and_row(tmp_path: Path):
    out = tmp_path / "metrics.csv"
    cb = CSVLoggerCallback(output_path=out)
    args = MagicMock()
    state = MagicMock(global_step=10, epoch=0.5)
    control = MagicMock()
    cb.on_log(args, state, control, logs={"loss": 1.23, "learning_rate": 1e-4})
    rows = list(csv.reader(out.read_text().splitlines()))
    assert rows[0] == ["step", "epoch", "loss", "eval_loss", "learning_rate"]
    assert rows[1][0] == "10"
    assert rows[1][2] == "1.23"


def test_csv_logger_appends_row_when_file_exists(tmp_path: Path):
    out = tmp_path / "metrics.csv"
    cb = CSVLoggerCallback(output_path=out)
    args = MagicMock()
    cb.on_log(args, MagicMock(global_step=10, epoch=0.5), MagicMock(),
              logs={"loss": 1.0})
    cb.on_log(args, MagicMock(global_step=20, epoch=1.0), MagicMock(),
              logs={"loss": 0.5})
    lines = out.read_text().splitlines()
    assert len(lines) == 3  # header + 2 rows


def test_csv_logger_noop_on_empty_logs(tmp_path: Path):
    out = tmp_path / "metrics.csv"
    cb = CSVLoggerCallback(output_path=out)
    cb.on_log(MagicMock(), MagicMock(global_step=0, epoch=0), MagicMock(), logs=None)
    cb.on_log(MagicMock(), MagicMock(global_step=0, epoch=0), MagicMock(), logs={})
    # File should not exist — nothing was logged
    assert not out.exists()


def test_csv_logger_records_eval_loss_when_present(tmp_path: Path):
    out = tmp_path / "metrics.csv"
    cb = CSVLoggerCallback(output_path=out)
    cb.on_log(MagicMock(), MagicMock(global_step=200, epoch=1.0), MagicMock(),
              logs={"eval_loss": 0.8})
    rows = list(csv.reader(out.read_text().splitlines()))
    assert rows[0] == ["step", "epoch", "loss", "eval_loss", "learning_rate"]
    # eval_loss is column index 3
    assert rows[1][3] == "0.8"
