"""Tests for src/fine_tune/prepare_data.py."""
import json
from pathlib import Path

import pytest

from fine_tune.prepare_data import (
    build_full_reasoning_payload,
    build_label_only_payload,
    entry_to_chat,
    prepare_splits,
)

FIXTURE = Path(__file__).parent / "fixtures" / "reasoning_sample.jsonl"


def load_fixture() -> list[dict]:
    return [json.loads(l) for l in FIXTURE.read_text().splitlines() if l.strip()]


class TestBuildLabelOnlyPayload:
    def test_has_only_two_keys(self):
        entry = load_fixture()[0]
        payload = build_label_only_payload(entry)
        assert set(payload.keys()) == {"diagnosis", "category"}

    def test_diagnosis_is_prettified(self):
        entry = load_fixture()[8]  # basal_cell_carcinoma
        payload = build_label_only_payload(entry)
        assert payload["diagnosis"] == "Basal Cell Carcinoma"

    def test_category_passed_through(self):
        entry = load_fixture()[0]
        payload = build_label_only_payload(entry)
        assert payload["category"] == "malignant"


class TestBuildFullReasoningPayload:
    def test_emits_unified_schema_keys(self):
        entry = load_fixture()[0]
        payload = build_full_reasoning_payload(entry)
        assert set(payload.keys()) >= {
            "diagnosis", "top_n", "confidence", "category",
            "observation", "reasoning", "differentials",
        }

    def test_diagnosis_is_prettified_ground_truth(self):
        entry = load_fixture()[0]
        payload = build_full_reasoning_payload(entry)
        assert payload["diagnosis"] == "Melanoma"

    def test_top_n_starts_with_diagnosis(self):
        entry = load_fixture()[0]  # 2 differentials
        payload = build_full_reasoning_payload(entry)
        assert payload["top_n"][0] == "Melanoma"
        assert len(payload["top_n"]) == 3  # diagnosis + 2 differentials

    def test_top_n_handles_zero_differentials(self):
        # Builder is format-agnostic; even flagged entries (which prepare_splits
        # would route to audit) are handled safely when the differentials list is empty.
        entry = load_fixture()[3]
        payload = build_full_reasoning_payload(entry)
        assert payload["top_n"] == ["Psoriasis"]

    def test_observation_is_nested_dict(self):
        entry = load_fixture()[0]
        payload = build_full_reasoning_payload(entry)
        assert isinstance(payload["observation"], dict)
        assert "morphology" in payload["observation"]


class TestEntryToChat:
    def test_label_only_format_produces_minimal_assistant_json(self):
        entry = load_fixture()[0]
        chat = entry_to_chat(entry, image_root=Path("/workspace/data/images"), format_name="label_only")
        assistant = json.loads(chat["messages"][1]["content"])
        assert set(assistant.keys()) == {"diagnosis", "category"}

    def test_full_reasoning_format_produces_unified_schema(self):
        entry = load_fixture()[0]
        chat = entry_to_chat(entry, image_root=Path("/workspace/data/images"), format_name="full_reasoning")
        assistant = json.loads(chat["messages"][1]["content"])
        assert "top_n" in assistant
        assert "observation" in assistant
        assert "reasoning" in assistant

    def test_image_path_is_absolute(self):
        entry = load_fixture()[0]
        chat = entry_to_chat(entry, image_root=Path("/workspace/data/images"), format_name="full_reasoning")
        img_content = next(c for c in chat["messages"][0]["content"] if c["type"] == "image")
        assert img_content["image"].startswith("/workspace/data/images/")
        assert img_content["image"].endswith("final/train/melanoma/mel-01.jpg")

    def test_label_only_and_full_reasoning_share_same_user_turn(self):
        """Both formats use the same image and the same user prompt — they differ only in assistant payload."""
        entry = load_fixture()[0]
        chat_lo = entry_to_chat(entry, image_root=Path("/workspace/data/images"), format_name="label_only")
        chat_fr = entry_to_chat(entry, image_root=Path("/workspace/data/images"), format_name="full_reasoning")
        assert chat_lo["messages"][0] == chat_fr["messages"][0]

    def test_unknown_format_raises(self):
        entry = load_fixture()[0]
        with pytest.raises(ValueError):
            entry_to_chat(entry, image_root=Path("/workspace/data/images"), format_name="bogus")


class TestPrepareSplits:
    def test_emits_four_format_variant_files(self, tmp_path):
        stats = prepare_splits(
            source=FIXTURE,
            out_dir=tmp_path,
            val_fraction=0.2,
            image_root=Path("/workspace/data/images"),
            seed=42,
        )
        for fname in (
            "train.label_only.jsonl",
            "val.label_only.jsonl",
            "train.full_reasoning.jsonl",
            "val.full_reasoning.jsonl",
            "audit.jsonl",
        ):
            assert (tmp_path / fname).exists(), f"missing {fname}"

    def test_separates_flagged_entries(self, tmp_path):
        prepare_splits(
            source=FIXTURE,
            out_dir=tmp_path,
            val_fraction=0.2,
            image_root=Path("/workspace/data/images"),
            seed=42,
        )
        audit_lines = (tmp_path / "audit.jsonl").read_text().splitlines()
        assert len(audit_lines) == 2  # 2 flagged in fixture

    def test_both_formats_have_identical_row_count(self, tmp_path):
        prepare_splits(
            source=FIXTURE,
            out_dir=tmp_path,
            val_fraction=0.2,
            image_root=Path("/workspace/data/images"),
            seed=42,
        )
        train_lo = len((tmp_path / "train.label_only.jsonl").read_text().splitlines())
        train_fr = len((tmp_path / "train.full_reasoning.jsonl").read_text().splitlines())
        val_lo = len((tmp_path / "val.label_only.jsonl").read_text().splitlines())
        val_fr = len((tmp_path / "val.full_reasoning.jsonl").read_text().splitlines())
        assert train_lo == train_fr
        assert val_lo == val_fr
        # 10 total, 2 flagged -> 8 total training rows per format
        assert train_lo + val_lo == 8

    def test_deterministic_with_fixed_seed(self, tmp_path):
        out_a = tmp_path / "a"; out_a.mkdir()
        out_b = tmp_path / "b"; out_b.mkdir()
        for out_dir in (out_a, out_b):
            prepare_splits(
                source=FIXTURE,
                out_dir=out_dir,
                val_fraction=0.2,
                image_root=Path("/workspace/data/images"),
                seed=42,
            )
        for fname in ("train.label_only.jsonl", "train.full_reasoning.jsonl",
                      "val.label_only.jsonl", "val.full_reasoning.jsonl"):
            assert (out_a / fname).read_bytes() == (out_b / fname).read_bytes()

    def test_stratified_split_keeps_each_class_in_train(self, tmp_path):
        prepare_splits(
            source=FIXTURE,
            out_dir=tmp_path,
            val_fraction=0.2,
            image_root=Path("/workspace/data/images"),
            seed=42,
        )
        train_classes = set()
        for line in (tmp_path / "train.label_only.jsonl").read_text().splitlines():
            payload = json.loads(json.loads(line)["messages"][1]["content"])
            train_classes.add(payload["diagnosis"])
        assert "Melanoma" in train_classes
        assert "Psoriasis" in train_classes
        assert "Eczema" in train_classes
