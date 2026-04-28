"""Dataset + collator for VLM SFT via Unsloth.

Two supported input shapes — both handled transparently by PathToImageCollator:

  1. Local JSONL (cheap dev loop on a pod with images on disk):
       {"messages": [..., {"type": "image", "image": "<rel/path>"}, ...]}
     Per-batch the collator does Image.open(path) just-in-time. We never hold
     25k PIL images in memory.

  2. Hub-loaded dataset with image bytes embedded via datasets.Image() feature
     (training on a fresh pod without local files):
       row["messages"] = [..., {"type": "image", "image": "<rel/path>"}, ...]
       row["image"]    = <PIL.Image (auto-decoded)>
     The collator detects the top-level PIL and substitutes it into the
     image-typed content part — so messages remain in the format Unsloth's
     UnslothVisionDataCollator expects (PIL objects inline) without any
     upfront .map() over the dataset.

Label masking is delegated to `train_on_responses_only` inside the collator.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from PIL import Image


def _open_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_chat_dataset(path_or_repo: str | Path, *, config_name: str | None = None,
                      split: str = "train"):
    """Load a chat-format dataset for SFT.

    Path-vs-repo dispatch:
      - Path-like (str/Path that exists on disk OR ends in .jsonl): load JSONL.
      - Anything else: treat as a HuggingFace Hub repo_id; uses datasets.load_dataset.

    Hub configs are expected to be the image-embedded variants
    (`<fmt>_with_images`) — those carry a top-level `image` Image() feature so
    the collator can reach pixels without local files.
    """
    from datasets import Dataset, load_dataset

    p = Path(path_or_repo) if isinstance(path_or_repo, (str, Path)) else None
    if p is not None and (p.exists() or str(p).endswith(".jsonl")):
        return _load_local_jsonl(p)

    # Hub repo
    return load_dataset(str(path_or_repo), name=config_name, split=split)


def _load_local_jsonl(path: Path):
    from datasets import Dataset
    rows: list[dict[str, Any]] = []
    with path.open() as f:
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
