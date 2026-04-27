"""Dataset + collator for VLM SFT via Unsloth.

The chat-format JSONL stores image references as filesystem paths (cheap to
keep on disk; loading 25k PIL objects up front is ~30GB+ RAM). Per-batch we
swap each `{"type": "image", "image": "<path>"}` → `{"type": "image",
"image": <PIL>}` and hand the rows to UnslothVisionDataCollator.

Label masking is delegated to `train_on_responses_only` inside the collator.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from PIL import Image


def _open_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_chat_dataset(path: Path):
    """Eagerly load a chat-format jsonl into a `datasets.Dataset`.

    Image paths stay as strings — the per-batch collator wrapper resolves them
    to PIL just-in-time, so we never hold 25k images in memory.
    """
    from datasets import Dataset

    rows: list[dict[str, Any]] = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rows.append(_normalise_row(d))
    return Dataset.from_list(rows)


def _normalise_row(d: dict[str, Any]) -> dict[str, Any]:
    """Ensure every message has a parts-list `content` (Unsloth requires uniform shape)."""
    out_messages = []
    for m in d.get("messages", []):
        content = m.get("content")
        if isinstance(content, list):
            out_messages.append({"role": m["role"], "content": content})
        else:
            out_messages.append({
                "role": m["role"],
                "content": [{"type": "text", "text": str(content)}],
            })
    return {"messages": out_messages}


class PathToImageCollator:
    """Wraps UnslothVisionDataCollator to load PIL images from path just-in-time.

    Unsloth's collator expects rows where image-typed parts already contain
    PIL objects. Our dataset stores filesystem paths to keep RAM usage flat
    across 25k+ samples, so we resolve them per-batch (≤16 images per call).
    """

    def __init__(self, model: Any, tokenizer: Any, max_seq_length: int | None = None):
        from unsloth.trainer import UnslothVisionDataCollator

        kwargs = {}
        if max_seq_length is not None:
            kwargs["max_seq_length"] = max_seq_length
        self._inner = UnslothVisionDataCollator(model, tokenizer, **kwargs)

    def __call__(self, rows: Sequence[dict[str, Any]]) -> Any:
        materialised = [self._materialise_pil(row) for row in rows]
        return self._inner(materialised)

    @staticmethod
    def _materialise_pil(row: dict[str, Any]) -> dict[str, Any]:
        out_messages = []
        for m in row.get("messages", []):
            new_content = []
            for c in m.get("content", []):
                if c.get("type") == "image" and isinstance(c.get("image"), str):
                    new_content.append({"type": "image", "image": _open_image(c["image"])})
                else:
                    new_content.append(c)
            out_messages.append({"role": m["role"], "content": new_content})
        return {**row, "messages": out_messages}
