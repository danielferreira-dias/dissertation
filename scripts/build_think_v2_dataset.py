"""Build and optionally upload the cleaned visual-thinking dermatology dataset.

This script creates ``danielfdias98/derm-reasoning-think-v2`` from the local
``data/fine_tune/full_reasoning`` JSONL files. It is intentionally deterministic:
no model/API rewriting is used. The generated ``<think>`` block is rebuilt from
visible structured fields instead of copying the original free-form reasoning.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = REPO_ROOT / "data" / "fine_tune" / "full_reasoning"
DEFAULT_OUT_DIR = REPO_ROOT / "dataset_export" / "derm-reasoning-think-v2"
DEFAULT_REPO_ID = "danielfdias98/derm-reasoning-think-v2"

USER_PROMPT_FALLBACK = "Diagnose this skin condition with structured reasoning."

SOURCE_PRIORITY = {
    "pad_ufes": 0,
    "scin": 1,
    "skincap": 2,
    "dermnet_nz": 3,
    "kaggle_dermnet": 4,
    "unknown": 99,
}

HARD_DERMOSCOPY_RE = re.compile(
    r"\b(dermoscopy|dermoscopic|under magnification|reticular pigment network|pigment network)\b",
    re.IGNORECASE,
)

HARD_NONCLINICAL_IMAGE_RE = re.compile(
    r"\b(laboratory|microscopic|culture|agar|petri|koh)\b",
    re.IGNORECASE,
)

HARD_NONCLINICAL_MEDIA_RE = re.compile(
    r"\b(agar|petri\s+dish|culture\s+medium|culture\s+plate|biopsy\s+slide|histolog(?:y|ic|ical)|microscopic\s+slide)\b",
    re.IGNORECASE,
)

SANITIZE_REPLACEMENTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(?:intensely\s+|highly\s+)?pruritic\b", re.I), "visibly excoriated"),
    (re.compile(r"\bpruritus\b", re.I), "visible excoriation"),
    (re.compile(r"\bitch(?:y|ing)?\b", re.I), "excoriated-appearing"),
    (re.compile(r"\b(tender(?:ness)?|painful|pain)\b", re.I), ""),
    (re.compile(r"\b(?:sudden|acute)\s+onset\b", re.I), ""),
    (re.compile(r"\bacute/subacute\b|\bsubacute\b", re.I), ""),
    (re.compile(r"\bhistory of [^.;,]*", re.I), ""),
    (re.compile(r"\bpatient reports [^.;,]*", re.I), ""),
    (re.compile(r"\b(?:frequently\s+)?exposed to water\b", re.I), "on exposed skin"),
    (re.compile(r"\bwater exposure\b", re.I), "exposure context"),
    (re.compile(r"\bswimwear\b", re.I), "covered areas"),
    (re.compile(r"\bjellyfish\b|\banemone larvae\b", re.I), "marine irritants"),
    (re.compile(r"\brecurrent\b", re.I), ""),
    (re.compile(r"\b(?:new\s+)?medication(?: exposure| use)?\b", re.I), "clinical exposure"),
    (re.compile(r"\b(fever|prodrome|duration)\b", re.I), ""),
    (re.compile(r"\basymptomatic\b", re.I), ""),
    (re.compile(r"\bchronic\b", re.I), ""),
    (re.compile(r"\bpalpation\b", re.I), "visual inspection"),
    (re.compile(r"\bpalpable\b", re.I), "raised"),
    (re.compile(r"\bfirm(?:er|ness)?\b", re.I), "raised"),
    (re.compile(r"\bindurat(?:ed|ion)\b", re.I), "thickened-appearing"),
    (re.compile(r"\bgritty\b", re.I), "scaly"),
    (re.compile(r"\bsandpaper(?:-like)?\b", re.I), "rough"),
    (re.compile(r"\bnon[- ]?blanching\b|\bblanching\b|\bblanch\b", re.I), "purpuric-appearing"),
    (re.compile(r"\bwood'?s lamp\b|\bwoods lamp\b|\bwood lamp\b", re.I), "additional light-based testing"),
    (re.compile(r"\bauspitz(?:-like)? sign\b", re.I), ""),
    (re.compile(r"\bbiopsy(?:-proven)?\b|\blab(?:oratory)? confirmation\b|\bculture\b|\bpcr\b", re.I), "confirmatory testing"),
    (re.compile(r"\bserolog(?:y|ic(?:al)?)\b", re.I), "confirmatory testing"),
    (re.compile(r"\bdermoscop(?:y|ic)\b", re.I), "visual"),
    (re.compile(r"\bunder magnification\b", re.I), "in the close-up view"),
    (re.compile(r"\breticular pigment network\b|\bpigment network\b", re.I), "pigment pattern"),
    (re.compile(r"\bevolv(?:e|es|ing|ed)?\b", re.I), "varies"),
    (re.compile(r"\bdue to clotted blood\b", re.I), "from a dark vascular-appearing focus"),
    (re.compile(r"\bdeeper?\s+subcutaneous\b|\bsubcutaneous\b", re.I), "larger nodular"),
    (re.compile(r"\bhyperalgesia\b|\ballodynia\b|\btrophic changes?\b", re.I), ""),
    (re.compile(r"\bhistolog(?:y|ic|ical)\b|\bmicroscop(?:y|ic)\b", re.I), "visual"),
    (re.compile(r"\bsubepidermal\b", re.I), "deeper-appearing"),
    (re.compile(r"\bepiderm(?:is|al)\b", re.I), "surface"),
    (re.compile(r"\bcollagen\b", re.I), "fibrous-appearing tissue"),
    (re.compile(r"\bfibroblasts?\b", re.I), "cellular detail"),
    (re.compile(r"\binfiltrate\b", re.I), "inflammatory-appearing change"),
]

FRAGMENT_REPLACEMENTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bhas a more\s*(?:and may be)?", re.I), ""),
    (re.compile(r"\bhave a more\s*(?:and may be)?", re.I), ""),
    (re.compile(r"\bwith a more\s*(?:and may be)?", re.I), ""),
    (re.compile(r"\bis usually\s*(?:and may be)?", re.I), ""),
    (re.compile(r"\bare usually\s*(?:and may be)?", re.I), ""),
    (re.compile(r"\bmay be\s*(?:,|;|\.)", re.I), ""),
    (re.compile(r"\band may be\s*(?:,|;|\.)", re.I), ""),
]

DROP_HIDDEN_SENTENCE_RE = re.compile(
    r"\b(sudden onset|acute onset|history of|patient reports|new medication|medication exposure|"
    r"tender|tenderness|painful|pain|asymptomatic|fever|prodrome|duration|"
    r"wood'?s lamp|woods lamp|wood lamp|auspitz(?:-like)? sign|biopsy|lab confirmation|"
    r"laboratory confirmation|culture|pcr|serology|serologic|serological|"
    r"systemic signs?|systemic symptoms?|systemic process|systemic causes?|"
    r"water exposure|exposed to water|swimwear|jellyfish|anemone larvae|"
    r"hyperalgesia|allodynia|trophic changes?|"
    r"histolog(?:y|ic|ical)|microscop(?:y|ic)|epidermis|subepidermal|collagen|fibroblast|infiltrate|"
    r"evolves?|evolving|evolved)\b",
    re.IGNORECASE,
)

DROP_DIFFERENTIAL_CONDITION_RE = re.compile(
    r"\b(normal urine under wood'?s lamp|normal urine under woods lamp)\b",
    re.IGNORECASE,
)

CONDITION_REPLACEMENTS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"Excoriations secondary to pruritus", re.IGNORECASE),
        "Excoriations from scratching",
    ),
    (
        re.compile(r"Scarlet fever", re.IGNORECASE),
        "Scarlatiniform eruption",
    ),
    (
        re.compile(r"Chronic Sun Damage", re.IGNORECASE),
        "Sun Damage",
    ),
    (
        re.compile(r"Complex regional pain syndrome", re.IGNORECASE),
        "Complex regional syndrome",
    ),
    (
        re.compile(
            r"Polymorphic eruption of pregnancy \(PEP\)\s*/\s*Pruritic urticarial papules and plaques of pregnancy \(PUPPP\)",
            re.IGNORECASE,
        ),
        "Polymorphic eruption of pregnancy (PEP) / PUPPP",
    ),
    (
        re.compile(r"Pruritic urticarial papules and plaques of pregnancy \(PUPPP\)", re.IGNORECASE),
        "PUPPP",
    ),
    (
        re.compile(r"Asymptomatic papules of the penis \(Tyson glands\) or Pearly penile papules", re.IGNORECASE),
        "Pearly penile papules / Tyson glands",
    ),
    (
        re.compile(r"Asymptomatic papules of childhood \(Gianotti-Crosti syndrome\)", re.IGNORECASE),
        "Gianotti-Crosti syndrome",
    ),
    (
        re.compile(r"Asymptomatic papules of the penis \(if located on genitalia\)", re.IGNORECASE),
        "Pearly penile papules",
    ),
    (
        re.compile(r"Asymptomatic papular lesions of other viral etiologies \(e\.g\., HPV\)", re.IGNORECASE),
        "Other papular viral lesions",
    ),
    (
        re.compile(r"Asymptomatic papular lesions of primary HIV infection", re.IGNORECASE),
        "Papular eruption of primary HIV infection",
    ),
    (
        re.compile(r"Excoriations from other pruritic conditions", re.IGNORECASE),
        "Excoriations from other dermatoses",
    ),
]


@dataclass
class Example:
    split: str
    row_idx: int
    image: str
    image_abs: Path
    instruction: str
    answer: dict[str, Any]
    class_label: str
    source: str
    image_id: str
    content_hash: str | None
    quality_flags: list[str] = field(default_factory=list)
    quarantine_reasons: list[str] = field(default_factory=list)


def prettify_label(label: str) -> str:
    return label.replace("_", " ").replace("-", " ").title()


def normalize_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def fallback_source(image_path: str) -> str:
    base = Path(image_path).stem
    if base.startswith("PAT_"):
        return "pad_ufes"
    if base.lstrip("-").isdigit():
        return "dermnet_nz"
    return "kaggle_dermnet"


def build_source_map(repo_root: Path = REPO_ROOT) -> dict[str, str]:
    source_map: dict[str, str] = {}
    for source in ("scin", "pad_ufes", "skincap", "kaggle_dermnet", "dermnet_nz"):
        source_dir = repo_root / "data" / "dataset" / source
        if not source_dir.is_dir():
            continue
        for image in source_dir.rglob("*"):
            if image.is_file() and image.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                source_map.setdefault(image.stem, source)
    return source_map


def attribute_source(image_path: str, source_map: dict[str, str]) -> str:
    return source_map.get(Path(image_path).stem) or fallback_source(image_path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_reason(example: Example, reason: str) -> None:
    if reason not in example.quarantine_reasons:
        example.quarantine_reasons.append(reason)


def add_flag(example: Example, flag: str) -> None:
    if flag not in example.quality_flags:
        example.quality_flags.append(flag)


def extract_user_turn(messages: list[dict[str, Any]]) -> tuple[str, str | None]:
    instruction = USER_PROMPT_FALLBACK
    image_path: str | None = None
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get("type") == "image" and image_path is None:
                image_path = part.get("image")
            elif part.get("type") == "text":
                instruction = part.get("text") or instruction
    return instruction, image_path


def extract_assistant_content(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") != "assistant":
            continue
        content = message.get("content", "")
        if isinstance(content, list):
            return "".join(str(part.get("text", "")) for part in content if part.get("type") == "text")
        return str(content)
    return ""


def resolve_image_path(repo_root: Path, image_path: str) -> Path:
    p = Path(image_path)
    return p if p.is_absolute() else repo_root / p


def load_examples(
    source_dir: Path,
    repo_root: Path = REPO_ROOT,
    *,
    limit: int | None = None,
    source_map: dict[str, str] | None = None,
) -> list[Example]:
    source_map = source_map if source_map is not None else build_source_map(repo_root)
    examples: list[Example] = []

    for split in ("train", "val"):
        path = source_dir / f"{split}.jsonl"
        with path.open() as f:
            for idx, line in enumerate(f):
                if limit is not None and idx >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                messages = payload.get("messages", [])
                instruction, image_path = extract_user_turn(messages)
                assistant_raw = extract_assistant_content(messages)
                if not image_path:
                    image_path = ""
                image_abs = resolve_image_path(repo_root, image_path) if image_path else repo_root / "__missing__"
                try:
                    answer = json.loads(assistant_raw)
                except json.JSONDecodeError:
                    answer = {}
                content_hash = sha256_file(image_abs) if image_abs.exists() and image_abs.is_file() else None
                class_label = Path(image_path).parent.name if image_path else ""
                example = Example(
                    split=split,
                    row_idx=idx,
                    image=image_path,
                    image_abs=image_abs,
                    instruction=instruction,
                    answer=answer,
                    class_label=class_label,
                    source=attribute_source(image_path, source_map) if image_path else "unknown",
                    image_id=Path(image_path).stem if image_path else "",
                    content_hash=content_hash,
                )
                if not answer:
                    add_reason(example, "invalid_response_json")
                if not image_abs.exists() or not image_abs.is_file():
                    add_reason(example, "missing_image")
                examples.append(example)
    return examples


def is_hard_dermoscopy(example: Example) -> bool:
    payload = json.dumps(example.answer, ensure_ascii=False)
    return bool(HARD_DERMOSCOPY_RE.search(f"{example.image}\n{payload}"))


def is_nonclinical_diagnostic_media(example: Example) -> bool:
    payload = json.dumps(example.answer, ensure_ascii=False)
    return bool(HARD_NONCLINICAL_IMAGE_RE.search(example.image) or HARD_NONCLINICAL_MEDIA_RE.search(payload))


def canonical_key(example: Example) -> tuple[int, int, int]:
    confidence = str(example.answer.get("confidence", "")).lower()
    confidence_rank = {"high": 0, "medium": 1, "": 2, "low": 3}.get(confidence, 2)
    source_rank = SOURCE_PRIORITY.get(example.source, SOURCE_PRIORITY["unknown"])
    return confidence_rank, source_rank, example.row_idx


def mark_quality_and_duplicates(examples: list[Example]) -> list[dict[str, Any]]:
    for example in examples:
        if str(example.answer.get("confidence", "")).lower() == "low":
            add_reason(example, "low_confidence")
        if is_hard_dermoscopy(example):
            add_reason(example, "hard_dermoscopy")
        if is_nonclinical_diagnostic_media(example):
            add_reason(example, "hard_nonclinical_diagnostic_media")

    duplicate_groups: list[dict[str, Any]] = []
    by_hash: dict[str, list[Example]] = collections.defaultdict(list)
    for example in examples:
        if example.content_hash:
            by_hash[example.content_hash].append(example)

    for content_hash, group in sorted(by_hash.items()):
        if len(group) <= 1:
            continue
        duplicate_groups.append({
            "content_hash": content_hash,
            "classes": sorted({g.class_label for g in group}),
            "splits": sorted({g.split for g in group}),
            "rows": [audit_stub(g) for g in sorted(group, key=lambda r: (r.split, r.row_idx))],
        })

        if len({g.class_label for g in group}) > 1:
            for example in group:
                add_reason(example, "duplicate_conflicting_class")
            continue

        active = [g for g in group if not g.quarantine_reasons]
        if len(active) <= 1:
            continue

        val_candidates = [g for g in active if g.split == "val"]
        if val_candidates:
            keep = min(val_candidates, key=canonical_key)
            for example in active:
                if example is keep:
                    continue
                if example.split == "train":
                    add_reason(example, "duplicate_cross_split_train_leak")
                else:
                    add_reason(example, "duplicate_same_split_noncanonical")
        else:
            keep = min(active, key=canonical_key)
            for example in active:
                if example is not keep:
                    add_reason(example, "duplicate_same_split_noncanonical")

    return duplicate_groups


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def clean_text(text: Any, *, drop_hidden_sentences: bool = False) -> tuple[str, bool]:
    if text is None:
        return "", False
    original = str(text)
    if drop_hidden_sentences:
        sentences = [s for s in split_sentences(original) if not DROP_HIDDEN_SENTENCE_RE.search(s)]
        cleaned = " ".join(sentences)
    else:
        cleaned = original
    for _ in range(3):
        before_pass = cleaned
        for pattern, replacement in SANITIZE_REPLACEMENTS:
            cleaned = pattern.sub(replacement, cleaned)
        for pattern, replacement in FRAGMENT_REPLACEMENTS:
            cleaned = pattern.sub(replacement, cleaned)
        if cleaned == before_pass:
            break
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    cleaned = re.sub(r"\(\s+", "(", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+/\s+", "/", cleaned)
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    cleaned = re.sub(r"\s+\.", ".", cleaned)
    cleaned = re.sub(r"\s*,\s*", ", ", cleaned)
    cleaned = re.sub(r"\s*;\s*", "; ", cleaned)
    cleaned = re.sub(r"\s*:\s*", ": ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = cleaned.strip(" ,;")
    return cleaned, cleaned != original


def ensure_sentence(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    return text if text.endswith((".", "!", "?")) else f"{text}."


def fallback_why_not(condition: str, diagnosis: str) -> str:
    return (
        "The visible morphology, color, texture, border, and distribution are "
        f"more consistent with {diagnosis} than {condition}."
    )


def cleaned_field(example: Example, key: str) -> tuple[str, bool]:
    return clean_text(example.answer.get(key, ""))


def clean_differentials(example: Example) -> tuple[list[dict[str, str]], bool]:
    changed = False
    out: list[dict[str, str]] = []
    differentials = example.answer.get("differentials") or []
    if not isinstance(differentials, list):
        return out, True
    for item in differentials:
        if not isinstance(item, dict):
            changed = True
            continue
        condition = str(item.get("condition", "")).strip()
        original_condition = condition
        for pattern, replacement in CONDITION_REPLACEMENTS:
            condition = pattern.sub(replacement, condition)
        condition = re.sub(r"\s*\((?:chronic)\)\s*", " ", condition, flags=re.IGNORECASE)
        condition = re.sub(r"\b(?:acute|chronic)\b", "", condition, flags=re.IGNORECASE)
        condition = re.sub(r"\bpruritus\b|\bitch\b", "scratching", condition, flags=re.IGNORECASE)
        condition = re.sub(r"\bsubcutaneous\b", "nodular", condition, flags=re.IGNORECASE)
        condition = re.sub(r"\s{2,}", " ", condition).strip(" -")
        condition = re.sub(r"\(\s*(?:or)?\s*\)", "", condition).strip()
        if condition and condition[0].islower() and original_condition[:1].isupper():
            condition = condition[:1].upper() + condition[1:]
        if DROP_DIFFERENTIAL_CONDITION_RE.search(condition):
            changed = True
            continue
        if condition != original_condition:
            changed = True
        why_not, did_change = clean_text(item.get("why_not", ""), drop_hidden_sentences=True)
        changed = changed or did_change
        if condition:
            out.append({"condition": condition, "why_not": ensure_sentence(why_not)})
    return out, changed


def build_answer(example: Example) -> tuple[dict[str, Any], bool]:
    changed = False
    fields: dict[str, str] = {}
    for key in ("observation", "morphology", "color", "texture", "border", "distribution"):
        fields[key], did_change = cleaned_field(example, key)
        changed = changed or did_change
    differentials, did_change = clean_differentials(example)
    changed = changed or did_change
    diagnosis = prettify_label(example.class_label)
    for differential in differentials:
        if not differential.get("why_not"):
            differential["why_not"] = fallback_why_not(differential["condition"], diagnosis)
            add_flag(example, "infilled_empty_differential_rationale")
            changed = True
    answer = {
        "diagnosis": diagnosis,
        "category": str(example.answer.get("category", "")).strip(),
        "confidence": str(example.answer.get("confidence", "medium")).strip() or "medium",
        "observation": fields["observation"],
        "morphology": fields["morphology"],
        "color": fields["color"],
        "texture": fields["texture"],
        "border": fields["border"],
        "distribution": fields["distribution"],
        "differentials": differentials,
    }
    original_diagnosis = str(example.answer.get("diagnosis", "")).strip()
    if original_diagnosis and normalize_label(original_diagnosis) != normalize_label(example.class_label):
        add_flag(example, "diagnosis_reanchored_to_class")
        changed = True
    if changed:
        add_flag(example, "sanitized_text")
    return answer, changed


def nonempty_bullet(label: str, value: str) -> str | None:
    value = value.strip()
    if not value:
        return None
    return f"- {label}: {value}"


def compact_text(text: str, max_chars: int = 280) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:")
    return f"{cut}."


def build_think_block(answer: dict[str, Any]) -> str:
    bullets = [
        nonempty_bullet("Morphology", answer.get("morphology", "")),
        nonempty_bullet("Color", answer.get("color", "")),
        nonempty_bullet("Texture", answer.get("texture", "")),
        nonempty_bullet("Border", answer.get("border", "")),
        nonempty_bullet("Distribution", answer.get("distribution", "")),
    ]
    visible = [b for b in bullets if b]
    if not visible and answer.get("observation"):
        visible = [f"- Observation: {answer['observation']}"]

    differential_lines: list[str] = []
    for differential in answer.get("differentials", [])[:3]:
        if not isinstance(differential, dict):
            continue
        condition = differential.get("condition", "").strip()
        why_not = compact_text(differential.get("why_not", ""))
        if condition and why_not:
            differential_lines.append(f"- {condition}: {why_not}")
        elif condition:
            differential_lines.append(
                f"- {condition}: considered, but the visible pattern above better supports {answer['diagnosis']}."
            )
    if not differential_lines:
        differential_lines = [f"- The visible pattern above best supports {answer['diagnosis']}."]

    limitations = (
        "- Symptoms, duration, tenderness, palpation findings, lab confirmation, "
        "dermoscopy, and patient history are not inferable from the image alone "
        "unless explicitly visible or provided."
    )

    return "\n".join([
        "Visible evidence:",
        *(visible or ["- No structured visible evidence was available."]),
        "",
        "Differential reasoning:",
        *differential_lines,
        "",
        "Limitations:",
        limitations,
    ])


def build_response(example: Example) -> str:
    answer, _ = build_answer(example)
    answer_json = json.dumps(answer, ensure_ascii=False)
    return f"<think>\n{build_think_block(answer)}\n</think>\n<answer>\n{answer_json}\n</answer>"


def audit_stub(example: Example) -> dict[str, Any]:
    return {
        "split": example.split,
        "row_idx": example.row_idx,
        "image": example.image,
        "image_id": example.image_id,
        "class": example.class_label,
        "diagnosis": example.answer.get("diagnosis", ""),
        "confidence": example.answer.get("confidence", ""),
        "source": example.source,
        "content_hash": example.content_hash,
        "quality_flags": list(example.quality_flags),
        "quarantine_reasons": list(example.quarantine_reasons),
    }


def output_row(example: Example) -> dict[str, Any]:
    return {
        "image": str(example.image_abs),
        "instruction": example.instruction,
        "response": build_response(example),
        "image_id": example.image_id,
        "class": example.class_label,
        "source": example.source,
        "content_hash": example.content_hash or "",
        "quality_flags": example.quality_flags,
    }


def build_dataset_rows(examples: list[Example]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    duplicate_groups = mark_quality_and_duplicates(examples)
    kept_examples = [e for e in examples if not e.quarantine_reasons]
    rows = {
        "train": [output_row(e) for e in kept_examples if e.split == "train"],
        "val": [output_row(e) for e in kept_examples if e.split == "val"],
    }
    quarantine = [audit_stub(e) for e in examples if e.quarantine_reasons]
    summary = summarize(examples, rows, quarantine, duplicate_groups)
    return rows, summary


def summarize(
    examples: list[Example],
    rows: dict[str, list[dict[str, Any]]],
    quarantine: list[dict[str, Any]],
    duplicate_groups: list[dict[str, Any]],
) -> dict[str, Any]:
    reasons = collections.Counter(
        reason for example in examples for reason in example.quarantine_reasons
    )
    sources = collections.Counter(example.source for example in examples)
    classes = collections.Counter(example.class_label for example in examples)
    kept_hash_splits: dict[str, set[str]] = collections.defaultdict(set)
    for split, split_rows in rows.items():
        for row in split_rows:
            kept_hash_splits[row["content_hash"]].add(split)
    cross_split_hashes = {
        content_hash: sorted(splits)
        for content_hash, splits in kept_hash_splits.items()
        if content_hash and len(splits) > 1
    }
    return {
        "source": str(DEFAULT_SOURCE_DIR),
        "total_rows": len(examples),
        "input_split_rows": dict(collections.Counter(e.split for e in examples)),
        "kept_split_rows": {split: len(split_rows) for split, split_rows in rows.items()},
        "quarantined_rows": len(quarantine),
        "quarantine_reasons": dict(sorted(reasons.items())),
        "duplicate_group_count": len(duplicate_groups),
        "sources": dict(sorted(sources.items())),
        "num_classes": len(classes),
        "top_classes": classes.most_common(30),
        "post_clean_cross_split_hash_count": len(cross_split_hashes),
        "post_clean_cross_split_hashes": cross_split_hashes,
    }


def write_audit(out_dir: Path, examples: list[Example], duplicate_groups: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    audit_dir = out_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    quarantine_path = audit_dir / "quarantine.jsonl"
    with quarantine_path.open("w") as f:
        for example in examples:
            if example.quarantine_reasons:
                f.write(json.dumps(audit_stub(example), ensure_ascii=False) + "\n")
    with (audit_dir / "duplicate_groups.jsonl").open("w") as f:
        for group in duplicate_groups:
            f.write(json.dumps(group, ensure_ascii=False) + "\n")
    with (audit_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def write_jsonl_export(out_dir: Path, rows: dict[str, list[dict[str, Any]]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, split_rows in rows.items():
        with (out_dir / f"{split}.jsonl").open("w") as f:
            for row in split_rows:
                lightweight = dict(row)
                lightweight["image"] = str(Path(row["image"]).relative_to(REPO_ROOT)) if Path(row["image"]).is_relative_to(REPO_ROOT) else row["image"]
                f.write(json.dumps(lightweight, ensure_ascii=False) + "\n")


def print_samples(examples: list[Example], n: int = 20) -> None:
    shown = 0
    for example in examples:
        if example.quarantine_reasons:
            continue
        shown += 1
        print(f"\n{'=' * 80}")
        print(f"SAMPLE {shown}: {example.split}/{example.row_idx} {example.class_label} {example.image}")
        print(build_response(example)[:2500])
        if shown >= n:
            break


def build_readme(repo_id: str, summary: dict[str, Any]) -> str:
    kept = summary["kept_split_rows"]
    reasons = summary["quarantine_reasons"]
    reasons_md = "\n".join(f"| `{k}` | {v} |" for k, v in reasons.items()) or "| none | 0 |"
    return f"""---
language: [en]
license: cc-by-nc-sa-4.0
task_categories: [image-text-to-text, visual-question-answering]
size_categories: [10K<n<100K]
tags: [medical, dermatology, vision-language, vlm-finetune, fairness, chain-of-thought, dataset-cleaning]
pretty_name: Dermatology Reasoning Dataset — Visible Thinking v2
---

# Dermatology Reasoning Dataset — Visible Thinking v2

This is a cleaned visible-thinking successor to
[`danielfdias98/derm-reasoning-full-reasoning`](https://huggingface.co/datasets/danielfdias98/derm-reasoning-full-reasoning).
The original dataset and the earlier CoT variant are not overwritten.

Every assistant response is formatted as:

```text
<think>
Visible evidence:
- ...

Differential reasoning:
- ...

Limitations:
- Symptoms, duration, tenderness, palpation findings, lab confirmation, dermoscopy, and patient history are not inferable from the image alone unless explicitly visible or provided.
</think>
<answer>
{{"diagnosis": "...", "category": "...", "confidence": "...", ...}}
</answer>
```

The `<think>` block is rebuilt deterministically from visible structured fields
(`morphology`, `color`, `texture`, `border`, `distribution`, and `observation`).
The original free-form `reasoning` field is **not copied** into the thinking
block.

## Cleaning Summary

| Metric | Count |
|---|---:|
| Source rows | {summary['total_rows']} |
| Train rows kept | {kept.get('train', 0)} |
| Validation rows kept | {kept.get('val', 0)} |
| Rows quarantined | {summary['quarantined_rows']} |
| Duplicate hash groups detected | {summary['duplicate_group_count']} |
| Post-clean train/val duplicate hashes | {summary['post_clean_cross_split_hash_count']} |

## Quarantine Reasons

| Reason | Rows |
|---|---:|
{reasons_md}

Rows were quarantined, not silently discarded during auditing. The build script
preserves `audit/quarantine.jsonl`, `audit/duplicate_groups.jsonl`, and
`audit/summary.json` locally for reproducibility.

## Schema

- `image`: embedded image
- `instruction`: user instruction
- `response`: `<think>/<answer>` formatted assistant response
- `image_id`: filename stem
- `class`: original class folder label
- `source`: attributed source dataset
- `content_hash`: SHA-256 of image bytes
- `quality_flags`: deterministic cleanup flags

## Quick Load

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}")
print(ds)
print(ds["train"][0]["response"])
```

## License

CC-BY-NC-SA 4.0, inheriting the most restrictive component of the source data.
Non-commercial research use only.

## Citation

```bibtex
@misc{{dias2026derm-reasoning-think-v2,
  author = {{Ferreira Dias, Daniel}},
  title  = {{Dermatology Reasoning Dataset — Visible Thinking v2}},
  year   = {{2026}},
  howpublished = {{\\url{{https://huggingface.co/datasets/{repo_id}}}}},
}}
```
"""


def make_dataset_dict(rows: dict[str, list[dict[str, Any]]]):
    from datasets import Dataset, DatasetDict, Features, Image, Sequence, Value

    features = Features({
        "image": Image(),
        "instruction": Value("string"),
        "response": Value("string"),
        "image_id": Value("string"),
        "class": Value("string"),
        "source": Value("string"),
        "content_hash": Value("string"),
        "quality_flags": Sequence(Value("string")),
    })
    return DatasetDict({
        split: Dataset.from_list(split_rows, features=features)
        for split, split_rows in rows.items()
    })


def resolve_hf_token() -> str | None:
    from dotenv import load_dotenv
    from huggingface_hub import HfApi

    load_dotenv(REPO_ROOT / ".env")
    token = os.environ.get("HF_TOKEN")
    if token:
        try:
            HfApi(token=token).whoami()
            return token
        except Exception as exc:
            print(
                f"WARNING: HF_TOKEN from .env/environment was rejected by Hugging Face "
                f"({type(exc).__name__}); falling back to cached HF login.",
                file=sys.stderr,
            )
            os.environ.pop("HF_TOKEN", None)
    try:
        HfApi().whoami()
    except Exception as exc:
        raise RuntimeError("No valid HF_TOKEN and no valid cached Hugging Face login") from exc
    return None


def push_to_hub(rows: dict[str, list[dict[str, Any]]], summary: dict[str, Any], repo_id: str, audit_dir: Path | None = None) -> None:
    from huggingface_hub import HfApi

    token = resolve_hf_token()

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
    dataset = make_dataset_dict(rows)
    dataset.push_to_hub(repo_id, token=token)
    api.upload_file(
        path_or_fileobj=build_readme(repo_id, summary).encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add visible-thinking v2 dataset card",
    )
    if audit_dir and audit_dir.is_dir():
        for name in ("quarantine.jsonl", "duplicate_groups.jsonl", "summary.json"):
            path = audit_dir / name
            if path.exists():
                api.upload_file(
                    path_or_fileobj=str(path),
                    path_in_repo=f"audit/{name}",
                    repo_id=repo_id,
                    repo_type="dataset",
                    commit_message=f"Add audit/{name}",
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--limit", type=int, help="Limit rows per split for sample validation")
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()

    if args.push and args.limit is not None:
        sys.exit("Refusing to push with --limit. Re-run without --limit for the full dataset.")

    examples = load_examples(args.source_dir, REPO_ROOT, limit=args.limit)
    duplicate_groups = mark_quality_and_duplicates(examples)
    kept_examples = [e for e in examples if not e.quarantine_reasons]
    rows = {
        "train": [output_row(e) for e in kept_examples if e.split == "train"],
        "val": [output_row(e) for e in kept_examples if e.split == "val"],
    }
    quarantine = [audit_stub(e) for e in examples if e.quarantine_reasons]
    summary = summarize(examples, rows, quarantine, duplicate_groups)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.dry_run:
        print_samples(examples, args.samples)
        return

    write_jsonl_export(args.out_dir, rows)
    write_audit(args.out_dir, examples, duplicate_groups, summary)
    (args.out_dir / "README.md").write_text(build_readme(args.repo_id, summary))
    print(f"Wrote local export and audit to {args.out_dir}")

    if args.push:
        push_to_hub(rows, summary, args.repo_id, args.out_dir / "audit")
        print(f"Pushed dataset to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
