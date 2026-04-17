"""Tests for src/fine_tune/data.py."""
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from fine_tune.data import MultimodalCollator, load_chat_dataset


def make_chat_row(image_path: str, answer_json: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": "Diagnose this skin condition with structured reasoning."},
            ]},
            {"role": "assistant", "content": answer_json},
        ]
    }


def test_load_chat_dataset_reads_jsonl(tmp_path: Path):
    jsonl = tmp_path / "tiny.jsonl"
    rows = [make_chat_row("/fake/a.jpg", '{"diagnosis":"A"}'),
            make_chat_row("/fake/b.jpg", '{"diagnosis":"B"}')]
    jsonl.write_text("\n".join(json.dumps(r) for r in rows))
    ds = load_chat_dataset(jsonl)
    assert len(ds) == 2
    assert ds[0]["messages"][0]["role"] == "user"


def test_load_chat_dataset_skips_blank_lines(tmp_path: Path):
    jsonl = tmp_path / "with_blanks.jsonl"
    row = make_chat_row("/fake/a.jpg", '{"diagnosis":"A"}')
    jsonl.write_text(json.dumps(row) + "\n\n" + json.dumps(row) + "\n")
    ds = load_chat_dataset(jsonl)
    assert len(ds) == 2


class FakeProcessor:
    """Minimal processor double: user turn → ids [1,2,3], assistant turn → ids [4,5]."""
    def __init__(self):
        self.tokenizer = MagicMock()
        self.tokenizer.pad_token_id = 0

    def apply_chat_template(self, conversation, tokenize=True, return_dict=True,
                             add_generation_prompt=False, **kwargs):
        user_turns = [m for m in conversation if m["role"] == "user"]
        assistant_turns = [m for m in conversation if m["role"] == "assistant"]
        ids = []
        for _ in user_turns:
            ids.extend([1, 2, 3])
        # When add_generation_prompt is True, no assistant ids (we're tokenizing
        # the prefix before the assistant response); otherwise include assistant.
        if not add_generation_prompt:
            for _ in assistant_turns:
                ids.extend([4, 5])
        input_ids = torch.tensor([ids])
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "pixel_values": torch.zeros(1, 3, 8, 8),
        }


def test_collator_masks_user_tokens_in_labels(monkeypatch):
    from fine_tune import data as data_mod
    monkeypatch.setattr(data_mod, "_open_image", lambda p: MagicMock(name=f"img:{p}"))

    processor = FakeProcessor()
    collator = MultimodalCollator(processor=processor, max_seq_length=2048)

    row = make_chat_row("/fake/a.jpg", '{"diagnosis":"Melanoma"}')
    batch = collator([row])

    assert batch["input_ids"].tolist() == [[1, 2, 3, 4, 5]]
    assert batch["labels"].tolist() == [[-100, -100, -100, 4, 5]]


def test_collator_pads_batch_to_longest(monkeypatch):
    from fine_tune import data as data_mod
    monkeypatch.setattr(data_mod, "_open_image", lambda p: MagicMock())

    class VarProcessor(FakeProcessor):
        def __init__(self):
            super().__init__()
            self._call = 0

        def apply_chat_template(self, conversation, add_generation_prompt=False, **kwargs):
            self._call += 1
            # Return different lengths for first vs second full sequence,
            # but correct shorter user-only prefix for add_generation_prompt=True.
            user_turns = [m for m in conversation if m["role"] == "user"]
            if add_generation_prompt:
                ids = [1, 2, 3] if user_turns else []
            else:
                ids = [1, 2, 3, 4, 5] if self._call <= 2 else [1, 2, 3, 4, 5, 6, 7, 8]
            input_ids = torch.tensor([ids])
            return {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "pixel_values": torch.zeros(1, 3, 8, 8),
            }

    collator = MultimodalCollator(processor=VarProcessor(), max_seq_length=2048)
    rows = [make_chat_row("/fake/a.jpg", "{}"), make_chat_row("/fake/b.jpg", "{}")]
    batch = collator(rows)
    assert batch["input_ids"].shape[0] == 2  # batch dim
    assert batch["input_ids"].shape[1] >= 5  # padded to longest


def test_collator_respects_max_seq_length(monkeypatch):
    from fine_tune import data as data_mod
    monkeypatch.setattr(data_mod, "_open_image", lambda p: MagicMock())

    class LongProcessor(FakeProcessor):
        def apply_chat_template(self, conversation, add_generation_prompt=False, **kwargs):
            if add_generation_prompt:
                ids = [1, 2, 3]
            else:
                ids = list(range(1, 21))  # 20 tokens total
            input_ids = torch.tensor([ids])
            return {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "pixel_values": torch.zeros(1, 3, 8, 8),
            }

    collator = MultimodalCollator(processor=LongProcessor(), max_seq_length=10)
    row = make_chat_row("/fake/a.jpg", "{}")
    batch = collator([row])
    assert batch["input_ids"].shape[-1] == 10  # truncated
