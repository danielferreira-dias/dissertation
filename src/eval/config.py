"""
Evaluation pipeline configuration.
Defines models, benchmarks, and evaluation settings.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ── Models ──────────────────────────────────────────────────────────────────
MODELS = {
    "medgemma-4b": {
        "hf_id": "google/medgemma-1.5-4b-it",
        "type": "medical",
        "size": "4B",
    },
    "gemma4-e4b": {
        "hf_id": "google/gemma-4-E4B-it",
        "type": "general",
        "size": "4B",
    },
    "qwen3.5-4b": {
        "hf_id": "Qwen/Qwen3.5-4B",
        "type": "general",
        "size": "4B",
    },
    "qwen3.5-9b": {
        "hf_id": "Qwen/Qwen3.5-9B",
        "type": "general",
        "size": "9B",
    },
}

# ── Benchmarks ──────────────────────────────────────────────────────────────
BENCHMARKS = {
    "fitzpatrick17k": {
        "path": PROJECT_ROOT / "final" / "benchmarks" / "fitzpatrick17k_1000",
        "metadata": PROJECT_ROOT / "final" / "benchmarks" / "fitzpatrick17k_1000" / "benchmark_metadata.csv",
        "type": "classification",
        "description": "1,000 Fitzpatrick17k images with FST labels. Top-1/Top-6 accuracy. Comparable to SkinFlow (29.19% Top-1).",
    },
    "mm_skin_vqa": {
        "path": PROJECT_ROOT / "final" / "benchmarks" / "mm_skin" / "MM-SkinQA",
        "metadata": PROJECT_ROOT / "final" / "benchmarks" / "mm_skin" / "vqa" / "MM-Skin_test.csv",
        "type": "vqa",
        "description": "5,452 VQA pairs from MM-Skin. Open-ended dermatology reasoning.",
    },
    "confusion_triads": {
        "path": PROJECT_ROOT / "final" / "test",
        "type": "classification",
        "description": "820 images across 6 classes (2 confusion triads). Custom evaluation set.",
    },
}

# ── Evaluation Settings ─────────────────────────────────────────────────────
RESULTS_DIR = PROJECT_ROOT / "final" / "results"

# ── Prompts ────────────────────────────────────────────────────────────────
# Prompts include the expected JSON format (improves output quality per vLLM docs).
# vLLM guided_json enforces the schema structurally.

CLASSIFICATION_PROMPT = """Diagnose this skin condition. Respond with a JSON object:
{
  "diagnosis": "most likely condition",
  "top_6": ["1st most likely", "2nd most likely", "3rd", "4th", "5th", "6th"],
  "confidence": "low | medium | high",
  "reasoning": "brief explanation of visual features"
}
Each entry in top_6 must be a different skin condition name."""

TRIAD_CLASSES = [
    "seborrheic_dermatitis",
    "psoriasis",
    "eczema",
    "seborrheic_keratosis",
    "basal_cell_carcinoma",
    "melanoma",
]

TRIAD_PROMPT = """Which of these 6 conditions does this image most likely show?
1. seborrheic_dermatitis
2. psoriasis
3. eczema
4. seborrheic_keratosis
5. basal_cell_carcinoma
6. melanoma

Respond with a JSON object:
{
  "diagnosis": "one of the 6 conditions above",
  "confidence": "low | medium | high",
  "reasoning": "brief explanation of visual features"
}"""

VQA_PROMPT_SUFFIX = " Answer concisely in one or two sentences. Respond with JSON: {\"answer\": \"your answer\"}"

# ── JSON Schemas for vLLM guided decoding ──────────────────────────────────

CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "diagnosis": {"type": "string"},
        "top_6": {
            "type": "array",
            "items": {"type": "string", "minLength": 3},
            "minItems": 6,
            "maxItems": 6,
        },
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "reasoning": {"type": "string"},
    },
    "required": ["diagnosis", "top_6", "confidence", "reasoning"],
}

TRIAD_SCHEMA = {
    "type": "object",
    "properties": {
        "diagnosis": {"type": "string", "enum": TRIAD_CLASSES},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "reasoning": {"type": "string"},
    },
    "required": ["diagnosis", "confidence", "reasoning"],
}

VQA_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
    },
    "required": ["answer"],
}
