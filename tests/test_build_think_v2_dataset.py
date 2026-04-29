import importlib.util
import json
import sys
from pathlib import Path

from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_think_v2_dataset.py"
spec = importlib.util.spec_from_file_location("build_think_v2_dataset", SCRIPT)
think_v2 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = think_v2
spec.loader.exec_module(think_v2)


def write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path)


def chat_line(image: str, label: str, *, confidence: str = "medium", extra: dict | None = None) -> str:
    answer = {
        "diagnosis": label.replace("_", " ").title(),
        "category": "inflammatory",
        "confidence": confidence,
        "observation": "Painful, pruritic papules with a chronic course.",
        "morphology": "Firm palpable papules",
        "color": "Pink to red",
        "texture": "Gritty sandpaper-like surface",
        "border": "Well defined",
        "distribution": "Localized to the arm",
        "reasoning": "RAW REASONING SHOULD NOT BE COPIED",
        "differentials": [
            {
                "condition": "Contact dermatitis",
                "why_not": "Typically itchy and related to a history of new medication exposure.",
            }
        ],
    }
    if extra:
        answer.update(extra)
    row = {
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Diagnose this skin condition with structured reasoning."},
            ]},
            {"role": "assistant", "content": json.dumps(answer)},
        ]
    }
    return json.dumps(row)


def write_split(source_dir: Path, split: str, lines: list[str]) -> None:
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / f"{split}.jsonl").write_text("\n".join(lines) + "\n")


def parse_response(response: str) -> dict:
    assert response.count("<think>") == 1
    assert response.count("</think>") == 1
    assert response.count("<answer>") == 1
    assert response.count("</answer>") == 1
    answer_text = response.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
    return json.loads(answer_text)


def response_without_limitations(response: str) -> str:
    think = response.split("<think>", 1)[1].split("</think>", 1)[0]
    return think.split("Limitations:", 1)[0] + response.split("<answer>", 1)[1]


def test_response_uses_think_answer_and_sanitizes_hidden_claims(tmp_path: Path):
    img = tmp_path / "final/train/eczema/a.png"
    write_image(img, (255, 0, 0))
    source_dir = tmp_path / "source"
    write_split(source_dir, "train", [chat_line("final/train/eczema/a.png", "eczema")])
    write_split(source_dir, "val", [])

    examples = think_v2.load_examples(source_dir, tmp_path, source_map={})
    rows, summary = think_v2.build_dataset_rows(examples)
    assert summary["kept_split_rows"]["train"] == 1

    response = rows["train"][0]["response"]
    answer = parse_response(response)
    assert answer["diagnosis"] == "Eczema"
    assert "RAW REASONING SHOULD NOT BE COPIED" not in response
    assert "Contact dermatitis:" in response
    assert answer["differentials"][0]["why_not"]
    assert "more consistent with Eczema than Contact dermatitis" in answer["differentials"][0]["why_not"]

    cleaned = response_without_limitations(response).lower()
    for banned in ("painful", "pruritic", "pruritus", "chronic", "palpable", "firm", "gritty", "sandpaper", "history of", "new medication"):
        assert banned not in cleaned


def test_empty_or_hidden_differential_rationale_is_repaired(tmp_path: Path):
    img = tmp_path / "final/train/tinea/a.png"
    write_image(img, (10, 120, 10))
    source_dir = tmp_path / "source"
    line = chat_line(
        "final/train/tinea/a.png",
        "tinea",
        extra={
            "differentials": [
                {"condition": "Eczema", "why_not": ""},
                {"condition": "Psoriasis", "why_not": "Serology and biopsy would be required."},
                {"condition": "Chronic paronychia", "why_not": ""},
            ]
        },
    )
    write_split(source_dir, "train", [line])
    write_split(source_dir, "val", [])

    examples = think_v2.load_examples(source_dir, tmp_path, source_map={})
    rows, _ = think_v2.build_dataset_rows(examples)
    answer = parse_response(rows["train"][0]["response"])

    why_nots = [item["why_not"] for item in answer["differentials"]]
    assert all(why_nots)
    assert [item["condition"] for item in answer["differentials"]] == ["Eczema", "Psoriasis", "Paronychia"]
    assert why_nots[0] == "The visible morphology, color, texture, border, and distribution are more consistent with Tinea than Eczema."
    assert why_nots[1] == "The visible morphology, color, texture, border, and distribution are more consistent with Tinea than Psoriasis."
    assert why_nots[2] == "The visible morphology, color, texture, border, and distribution are more consistent with Tinea than Paronychia."


def test_low_confidence_rows_are_quarantined(tmp_path: Path):
    img = tmp_path / "final/train/eczema/low.png"
    write_image(img, (255, 0, 0))
    source_dir = tmp_path / "source"
    write_split(source_dir, "train", [chat_line("final/train/eczema/low.png", "eczema", confidence="low")])
    write_split(source_dir, "val", [])

    examples = think_v2.load_examples(source_dir, tmp_path, source_map={})
    rows, summary = think_v2.build_dataset_rows(examples)
    assert rows["train"] == []
    assert summary["quarantine_reasons"]["low_confidence"] == 1


def test_same_hash_cross_split_keeps_val_and_quarantines_train(tmp_path: Path):
    train_img = tmp_path / "final/train/eczema/dup_train.png"
    val_img = tmp_path / "final/train/eczema/dup_val.png"
    write_image(train_img, (10, 20, 30))
    write_image(val_img, (10, 20, 30))
    source_dir = tmp_path / "source"
    write_split(source_dir, "train", [chat_line("final/train/eczema/dup_train.png", "eczema")])
    write_split(source_dir, "val", [chat_line("final/train/eczema/dup_val.png", "eczema")])

    examples = think_v2.load_examples(source_dir, tmp_path, source_map={})
    rows, summary = think_v2.build_dataset_rows(examples)
    assert rows["train"] == []
    assert len(rows["val"]) == 1
    assert summary["post_clean_cross_split_hash_count"] == 0
    assert summary["quarantine_reasons"]["duplicate_cross_split_train_leak"] == 1


def test_conflicting_duplicate_hash_group_is_quarantined(tmp_path: Path):
    img_a = tmp_path / "final/train/eczema/conflict_a.png"
    img_b = tmp_path / "final/train/psoriasis/conflict_b.png"
    write_image(img_a, (1, 2, 3))
    write_image(img_b, (1, 2, 3))
    source_dir = tmp_path / "source"
    write_split(source_dir, "train", [
        chat_line("final/train/eczema/conflict_a.png", "eczema"),
        chat_line("final/train/psoriasis/conflict_b.png", "psoriasis"),
    ])
    write_split(source_dir, "val", [])

    examples = think_v2.load_examples(source_dir, tmp_path, source_map={})
    rows, summary = think_v2.build_dataset_rows(examples)
    assert rows["train"] == []
    assert summary["quarantine_reasons"]["duplicate_conflicting_class"] == 2


def test_hard_dermoscopy_rows_are_quarantined(tmp_path: Path):
    img = tmp_path / "final/train/lentigo/derm.png"
    write_image(img, (100, 100, 20))
    source_dir = tmp_path / "source"
    write_split(source_dir, "train", [
        chat_line(
            "final/train/lentigo/derm.png",
            "lentigo",
            extra={"observation": "Reticular pigment network under magnification."},
        )
    ])
    write_split(source_dir, "val", [])

    examples = think_v2.load_examples(source_dir, tmp_path, source_map={})
    rows, summary = think_v2.build_dataset_rows(examples)
    assert rows["train"] == []
    assert summary["quarantine_reasons"]["hard_dermoscopy"] == 1


def test_nonclinical_culture_media_rows_are_quarantined(tmp_path: Path):
    img = tmp_path / "final/train/tinea/culture.png"
    write_image(img, (220, 220, 200))
    source_dir = tmp_path / "source"
    write_split(source_dir, "train", [
        chat_line(
            "final/train/tinea/culture.png",
            "tinea",
            extra={"distribution": "Localized to the surface of an agar medium in a petri dish."},
        )
    ])
    write_split(source_dir, "val", [])

    examples = think_v2.load_examples(source_dir, tmp_path, source_map={})
    rows, summary = think_v2.build_dataset_rows(examples)
    assert rows["train"] == []
    assert summary["quarantine_reasons"]["hard_nonclinical_diagnostic_media"] == 1


def test_build_is_deterministic(tmp_path: Path):
    img = tmp_path / "final/train/eczema/a.png"
    write_image(img, (255, 0, 0))
    source_dir = tmp_path / "source"
    write_split(source_dir, "train", [chat_line("final/train/eczema/a.png", "eczema")])
    write_split(source_dir, "val", [])

    first_examples = think_v2.load_examples(source_dir, tmp_path, source_map={})
    second_examples = think_v2.load_examples(source_dir, tmp_path, source_map={})
    first_rows, first_summary = think_v2.build_dataset_rows(first_examples)
    second_rows, second_summary = think_v2.build_dataset_rows(second_examples)
    assert first_rows == second_rows
    assert first_summary == second_summary
