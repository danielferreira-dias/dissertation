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

# Classification prompt for Fitzpatrick17k and confusion triads
CLASSIFICATION_PROMPT = """Look at this dermatological image and provide your diagnosis.

Respond with ONLY a JSON object:
{
  "diagnosis": "the most likely skin condition",
  "top_6": ["most likely", "2nd most likely", "3rd", "4th", "5th", "6th"],
  "confidence": "low | medium | high",
  "reasoning": "brief explanation of visual features supporting your diagnosis"
}"""

# For confusion triad evaluation, constrain to 6 classes
TRIAD_PROMPT = """Look at this dermatological image. Based on the visual features, which of the following conditions does this image most likely show?

1. Seborrheic Dermatitis
2. Psoriasis
3. Eczema
4. Seborrheic Keratosis
5. Basal Cell Carcinoma
6. Melanoma

Respond with ONLY a JSON object:
{
  "diagnosis": "one of the 6 conditions above",
  "confidence": "low | medium | high",
  "reasoning": "brief explanation of visual features"
}"""
