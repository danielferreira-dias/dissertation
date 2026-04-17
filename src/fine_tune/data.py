"""Dataset + collator for VLM SFT.

The collator:
  1. Loads each row's image from disk
  2. Applies the processor's chat template to produce input_ids + pixel_values
  3. Masks user-turn tokens in `labels` (-100) so loss is computed only on assistant turn
  4. Pads the batch to the longest sample (respecting max_seq_length)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image


def _open_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_chat_dataset(path: Path) -> list[dict[str, Any]]:
    """Eagerly load a chat-format jsonl into a list of dicts.

    28k rows × small dict fits in RAM easily; images stay on disk and
    are loaded per-batch inside the collator.
    """
    rows: list[dict[str, Any]] = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class MultimodalCollator:
    """Collate chat-format rows into a padded batch with masked labels.

    Label masking: everything up to (and including) the user-only prefix is
    set to -100 so the loss is computed only over the assistant response.
    """

    def __init__(self, processor: Any, max_seq_length: int):
        self.processor = processor
        self.max_seq_length = max_seq_length
        tokenizer = getattr(processor, "tokenizer", processor)
        self.pad_token_id = getattr(tokenizer, "pad_token_id", None) or 0

    def __call__(self, rows: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids_list: list[torch.Tensor] = []
        label_list: list[torch.Tensor] = []
        pixel_list: list[torch.Tensor] = []

        for row in rows:
            conversation, images = self._split_row(row)

            full = self.processor.apply_chat_template(
                conversation,
                tokenize=True,
                return_dict=True,
                add_generation_prompt=False,
                images=images if images else None,
            )
            full_ids = full["input_ids"].squeeze(0)

            user_only = [m for m in conversation if m["role"] == "user"]
            prefix = self.processor.apply_chat_template(
                user_only,
                tokenize=True,
                return_dict=True,
                add_generation_prompt=True,
                images=images if images else None,
            )
            prefix_len = prefix["input_ids"].shape[-1]

            labels = full_ids.clone()
            labels[:prefix_len] = -100

            input_ids_list.append(full_ids)
            label_list.append(labels)
            if "pixel_values" in full:
                pv = full["pixel_values"]
                pixel_list.append(pv.squeeze(0) if pv.dim() == 4 and pv.shape[0] == 1 else pv)

        input_ids, attention_mask, labels = self._pad(input_ids_list, label_list)
        batch: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if pixel_list:
            if pixel_list[0].dim() == 3:
                batch["pixel_values"] = torch.stack(pixel_list, dim=0)
            else:
                batch["pixel_values"] = torch.cat(pixel_list, dim=0)
        return batch

    def _split_row(self, row: dict[str, Any]) -> tuple[list[dict], list[Image.Image]]:
        messages = row["messages"]
        images: list[Image.Image] = []
        clean_messages: list[dict] = []
        for m in messages:
            if isinstance(m["content"], list):
                parts: list[dict] = []
                for c in m["content"]:
                    if c.get("type") == "image":
                        images.append(_open_image(c["image"]))
                        parts.append({"type": "image"})
                    else:
                        parts.append(c)
                clean_messages.append({"role": m["role"], "content": parts})
            else:
                clean_messages.append(m)
        return clean_messages, images

    def _pad(
        self,
        ids: list[torch.Tensor],
        labels: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_len = min(self.max_seq_length, max(x.shape[-1] for x in ids))
        batch_size = len(ids)
        out_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=ids[0].dtype)
        out_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        out_labels = torch.full((batch_size, max_len), -100, dtype=labels[0].dtype)
        for i, (x, y) in enumerate(zip(ids, labels)):
            n = min(x.shape[-1], max_len)
            out_ids[i, :n] = x[:n]
            out_mask[i, :n] = 1
            out_labels[i, :n] = y[:n]
        return out_ids, out_mask, out_labels
