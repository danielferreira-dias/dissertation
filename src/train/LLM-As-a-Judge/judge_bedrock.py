"""
LLM-as-Judge for Fitzpatrick17k predictions via AWS Bedrock Converse API.

Uses Claude Opus 4.5 or Sonnet 4.5 on AWS Bedrock to clinically judge
vision-model predictions against ground-truth diagnoses using the SkinFlow
rubric (correct / subclass / safety-critical / wrong).

Quick start:
  # Single batch file, Sonnet (cheaper, faster)
  python3 LLM-As-a-Judge/judge_bedrock.py \
    --model sonnet \
    --input data/judge_batches/judge_batch_medgemma-4b_1_c1.jsonl \
    --output data/judge_verdicts/verdicts_medgemma-4b_1_c1.json

  # Process ALL batch files in parallel (8 workers)
  python3 LLM-As-a-Judge/judge_bedrock.py --model sonnet --all --workers 8

  # Use Opus for higher quality
  python3 LLM-As-a-Judge/judge_bedrock.py --model opus --all --workers 4

Prerequisites:
  - AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY in .env
  - AWS_REGION_NAME in .env (must be a region with Bedrock + Claude access;
    eu-west-1, eu-central-1, us-east-1, or us-west-2 are typical choices)
  - Bedrock model access enabled for Claude Opus/Sonnet in your AWS account
    (visit AWS Bedrock → Model access, request access to anthropic.claude-*)
  - pip install boto3 python-dotenv tqdm
"""

import argparse
import concurrent.futures
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Bedrock model IDs ────────────────────────────────────────────────────────
# Bedrock requires inference profile IDs for cross-region routing.
# Use "us." prefix for US profiles, "eu." for EU profiles.
# Adjust these IDs to match what's available in your AWS region.
MODEL_IDS = {
    "opus": {
        "us": "us.anthropic.claude-opus-4-5-20250929-v1:0",
        "eu": "eu.anthropic.claude-opus-4-5-20250929-v1:0",
        "global": "anthropic.claude-opus-4-5-20250929-v1:0",
    },
    "sonnet": {
        "us": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "eu": "eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "global": "anthropic.claude-sonnet-4-5-20250929-v1:0",
    },
    # Fallbacks for older Claude versions if 4.5 is not yet enabled:
    "opus-3": {
        "us": "us.anthropic.claude-3-opus-20240229-v1:0",
        "eu": "eu.anthropic.claude-3-opus-20240229-v1:0",
        "global": "anthropic.claude-3-opus-20240229-v1:0",
    },
    "sonnet-3-5": {
        "us": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "eu": "eu.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "global": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    },
}


def resolve_model_id(short_name: str, region: str) -> str:
    """Pick the right model ID based on region prefix."""
    profiles = MODEL_IDS.get(short_name)
    if not profiles:
        raise ValueError(f"Unknown model '{short_name}'. Options: {list(MODEL_IDS)}")
    if region.startswith("us-"):
        return profiles["us"]
    if region.startswith("eu-"):
        return profiles["eu"]
    return profiles["global"]


# ── Rubric ──────────────────────────────────────────────────────────────────
GT_CLASSES_114 = (
    "acanthosis nigricans, acne, acne vulgaris, acquired autoimmune bullous disease/herpes gestationis, "
    "acrodermatitis enteropathica, actinic keratosis, allergic contact dermatitis, aplasia cutis, "
    "basal cell carcinoma, basal cell carcinoma morpheiform, becker nevus, behcets disease, "
    "calcinosis cutis, cheilitis, congenital nevus, dariers disease, dermatofibroma, dermatomyositis, "
    "disseminated actinic porokeratosis, drug eruption, drug induced pigmentary changes, "
    "dyshidrotic eczema, eczema, ehlers danlos syndrome, epidermal nevus, epidermolysis bullosa, "
    "erythema annulare centrifigum, erythema elevatum diutinum, erythema multiforme, erythema nodosum, "
    "factitial dermatitis, fixed eruptions, folliculitis, fordyce spots, granuloma annulare, "
    "granuloma pyogenic, hailey hailey disease, halo nevus, hidradenitis, ichthyosis vulgaris, "
    "incontinentia pigmenti, juvenile xanthogranuloma, kaposi sarcoma, keloid, keratosis pilaris, "
    "langerhans cell histiocytosis, lentigo maligna, lichen amyloidosis, lichen planus, lichen simplex, "
    "livedo reticularis, lupus erythematosus, lupus subacute, lyme disease, lymphangioma, "
    "malignant melanoma, melanoma, milia, mucinosis, mucous cyst, mycosis fungoides, myiasis, "
    "naevus comedonicus, necrobiosis lipoidica, nematode infection, neurodermatitis, neurofibromatosis, "
    "neurotic excoriations, neutrophilic dermatoses, nevocytic nevus, nevus sebaceous of jadassohn, "
    "papilomatosis confluentes and reticulate, paronychia, pediculosis lids, perioral dermatitis, "
    "photodermatoses, pilar cyst, pilomatricoma, pityriasis lichenoides chronica, pityriasis rosea, "
    "pityriasis rubra pilaris, porokeratosis actinic, porokeratosis of mibelli, porphyria, "
    "port wine stain, prurigo nodularis, psoriasis, pustular psoriasis, pyogenic granuloma, "
    "rhinophyma, rosacea, sarcoidosis, scabies, scleroderma, scleromyxedema, seborrheic dermatitis, "
    "seborrheic keratosis, solid cystic basal cell carcinoma, squamous cell carcinoma, stasis edema, "
    "stevens johnson syndrome, striae, sun damaged skin, superficial spreading melanoma ssm, "
    "syringoma, telangiectases, tick bite, tuberous sclerosis, tungiasis, urticaria, "
    "urticaria pigmentosa, vitiligo, xanthomas, xeroderma pigmentosum"
)

SYSTEM_PROMPT = f"""You are an expert dermatology clinical judge evaluating vision model predictions on the Fitzpatrick17k benchmark. Pure text reasoning — no images.

For each entry you receive, judge the model's primary diagnosis `d` against the ground truth `gt` using this rubric:

- "c" (correct): `d` exactly matches `gt`, OR is a medically accepted synonym/alias/abbreviation (e.g. "SCC"="squamous cell carcinoma", "atopic dermatitis"="eczema", "morphea"="scleroderma", "hives"="urticaria", "BCC"="basal cell carcinoma")
- "s" (subclass): `d` is a clinically valid subclass/subtype/variant of `gt`, or `gt` is a subclass of `d` (e.g. "plaque psoriasis" for "psoriasis", "nodular BCC" for "basal cell carcinoma", "discoid lupus erythematosus" for "lupus erythematosus", "acral lentiginous melanoma" for "melanoma")
- "sc" (safety-critical false): `d` crosses a critical clinical boundary — benign↔malignant misclassification, or infectious↔non-infectious misclassification that could harm patient management
- "w" (wrong): `d` is clinically unrelated to `gt`, empty, invalid, or gibberish

For TOP-6 scoring, find the first position (1-6) in `t6` where any entry is correct/subclass of `gt`. Set `p` to that position. If `d` itself is correct/subclass, set p=1. If no match anywhere, set p=0.

The 114 canonical Fitzpatrick17k class names (for synonym/subclass reference):
{GT_CLASSES_114}

OUTPUT FORMAT: Return ONLY a valid JSON array (no markdown fences, no commentary), one object per input entry in the SAME ORDER as input:
[{{"i":0,"v":"c","p":1}},{{"i":1,"v":"w","p":0}},...]
"""


# ── Bedrock client ──────────────────────────────────────────────────────────


def make_client(region: str) -> Any:
    """Create a Bedrock runtime client with sensible timeouts and retries."""
    config = Config(
        region_name=region,
        retries={"max_attempts": 5, "mode": "adaptive"},
        read_timeout=300,
        connect_timeout=30,
    )
    return boto3.client("bedrock-runtime", config=config)


def converse_judge(client: Any, model_id: str, entries: list[dict]) -> list[dict]:
    """Send a batch of entries to Bedrock Converse API and parse verdicts.

    Returns a list of {"i": ..., "v": ..., "p": ...} dicts.
    """
    user_content = (
        "Judge these entries. Return ONLY the JSON array.\n\n"
        + json.dumps(entries, ensure_ascii=False)
    )

    response = client.converse(
        modelId=model_id,
        system=[{"text": SYSTEM_PROMPT}],
        messages=[
            {
                "role": "user",
                "content": [{"text": user_content}],
            }
        ],
        inferenceConfig={
            "maxTokens": 8192,
            "temperature": 0.0,
        },
    )

    # Extract text from the response
    output_message = response["output"]["message"]
    text_parts = [block.get("text", "") for block in output_message.get("content", [])]
    raw_text = "\n".join(text_parts).strip()

    # Strip markdown fences if the model added them despite instructions
    if raw_text.startswith("```"):
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
        raw_text = re.sub(r"\s*```\s*$", "", raw_text)

    # Find the JSON array (model may include leading prose)
    match = re.search(r"\[\s*\{.*\}\s*\]", raw_text, re.DOTALL)
    if match:
        raw_text = match.group(0)

    try:
        verdicts = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse JSON verdict array: {e}\nRaw: {raw_text[:500]}")

    if not isinstance(verdicts, list):
        raise ValueError(f"Expected JSON array, got {type(verdicts).__name__}")

    return verdicts


# ── Batch processing ────────────────────────────────────────────────────────


def load_batch_jsonl(path: Path) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def process_batch_file(
    client: Any,
    model_id: str,
    batch_path: Path,
    output_path: Path,
    sub_batch_size: int = 50,
    max_retries: int = 3,
) -> dict:
    """Process a single batch JSONL file, writing verdicts to output_path."""
    if output_path.exists():
        return {"path": str(output_path), "status": "skipped (exists)", "count": 0}

    entries = load_batch_jsonl(batch_path)
    verdicts: list[dict] = []

    # Chunk entries into sub-batches to keep each Converse call small
    for start in range(0, len(entries), sub_batch_size):
        sub = entries[start : start + sub_batch_size]
        for attempt in range(max_retries):
            try:
                sub_verdicts = converse_judge(client, model_id, sub)
                if len(sub_verdicts) != len(sub):
                    raise ValueError(
                        f"Verdict count mismatch: got {len(sub_verdicts)}, expected {len(sub)}"
                    )
                verdicts.extend(sub_verdicts)
                break
            except (ClientError, ValueError) as e:
                if attempt == max_retries - 1:
                    raise
                wait = 2 ** (attempt + 1)
                print(f"  ! {batch_path.name} sub-batch {start}: {e} — retrying in {wait}s", file=sys.stderr)
                time.sleep(wait)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(verdicts))
    return {"path": str(output_path), "status": "ok", "count": len(verdicts)}


# ── CLI ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Judge vision-model dermatology predictions via AWS Bedrock Converse API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_IDS),
        default="sonnet",
        help="Short model name (default: sonnet). Use 'opus' for highest quality.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Override full Bedrock model ID (e.g. 'us.anthropic.claude-opus-...'). "
        "Takes precedence over --model.",
    )
    parser.add_argument(
        "--region",
        default=os.environ.get("AWS_REGION_NAME", "us-east-1"),
        help="AWS region for Bedrock (default: $AWS_REGION_NAME or us-east-1). "
        "Must be a region where Claude is available (e.g. us-east-1, us-west-2, eu-west-1).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to a single batch JSONL file to judge.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the verdicts JSON (required with --input).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process every JSONL file under data/judge_batches/ and write to data/judge_verdicts/.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel workers when using --all (default: 4).",
    )
    parser.add_argument(
        "--sub-batch-size",
        type=int,
        default=50,
        help="Entries per Bedrock Converse call (default: 50).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be called without hitting Bedrock.",
    )
    args = parser.parse_args()

    # Resolve model
    model_id = args.model_id or resolve_model_id(args.model, args.region)

    print(f"Region:       {args.region}")
    print(f"Model short:  {args.model}")
    print(f"Model ID:     {model_id}")
    print(f"Sub-batch:    {args.sub_batch_size}")
    print()

    if args.dry_run:
        print("--dry-run set, exiting before making any Bedrock calls.")
        return

    if not os.environ.get("AWS_ACCESS_KEY_ID"):
        sys.exit("AWS_ACCESS_KEY_ID not set. Put it in .env or export it.")

    client = make_client(args.region)

    # ── Single file mode ────────────────────────────────────────────────────
    if args.input is not None:
        if args.output is None:
            sys.exit("--input requires --output")
        result = process_batch_file(
            client, model_id, args.input, args.output, args.sub_batch_size
        )
        print(f"{result['path']}: {result['status']} ({result['count']} verdicts)")
        return

    # ── Bulk mode ───────────────────────────────────────────────────────────
    if args.all:
        in_dir = PROJECT_ROOT / "data" / "judge_batches"
        out_dir = PROJECT_ROOT / "data" / "judge_verdicts"
        batches = sorted(in_dir.glob("judge_batch_*.jsonl"))
        if not batches:
            sys.exit(f"No batch files found in {in_dir}")

        def verdict_path_for(batch: Path) -> Path:
            # judge_batch_medgemma-4b_1_c1.jsonl → verdicts_medgemma-4b_1_c1.json
            stem = batch.stem.replace("judge_batch_", "verdicts_")
            return out_dir / f"{stem}.json"

        targets = [(b, verdict_path_for(b)) for b in batches]
        todo = [(b, o) for b, o in targets if not o.exists()]
        print(f"Total batches: {len(targets)}  Already done: {len(targets) - len(todo)}  To run: {len(todo)}")

        if not todo:
            print("Nothing to do.")
            return

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    process_batch_file, client, model_id, b, o, args.sub_batch_size
                ): b.name
                for b, o in todo
            }
            with tqdm(total=len(futures), desc="Judging batches") as bar:
                for fut in concurrent.futures.as_completed(futures):
                    name = futures[fut]
                    try:
                        result = fut.result()
                        tqdm.write(f"  ✓ {name}: {result['status']} ({result['count']})")
                    except Exception as e:
                        tqdm.write(f"  ✗ {name}: {e}")
                    bar.update(1)
        return

    parser.print_help()
    sys.exit("Pass --input+--output for a single file, or --all for bulk mode.")


if __name__ == "__main__":
    main()
