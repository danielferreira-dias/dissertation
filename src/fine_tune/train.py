"""
Fine-tune a VLM with LoRA or QLoRA on dermatology training data.

Supports all 4 student models and both data formats (label-only, full-reasoning).
Uses Hugging Face transformers + PEFT for adapter training.

Usage:
  # LoRA on Qwen 3.5 4B with label-only data
  python src/fine_tune/train.py --model qwen3.5-4b --data-format label_only --method lora

  # QLoRA on Qwen 3.5 9B with full reasoning
  python src/fine_tune/train.py --model qwen3.5-9b --data-format full_reasoning --method qlora

  # All Phase 1 experiments (Qwen 3.5 4B, 4 runs)
  python src/fine_tune/train.py --model qwen3.5-4b --data-format label_only --method lora
  python src/fine_tune/train.py --model qwen3.5-4b --data-format full_reasoning --method lora
  python src/fine_tune/train.py --model qwen3.5-4b --data-format label_only --method qlora
  python src/fine_tune/train.py --model qwen3.5-4b --data-format full_reasoning --method qlora
"""

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from PIL import Image
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

# ── Model Registry ──────────────────────────────────────────────────────────

MODELS = {
    "qwen3.5-4b": {
        "hf_id": "Qwen/Qwen3.5-4B",
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    },
    "qwen3.5-9b": {
        "hf_id": "Qwen/Qwen3.5-9B",
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    },
    "medgemma-4b": {
        "hf_id": "google/medgemma-1.5-4b-it",
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    },
    "gemma4-e4b": {
        "hf_id": "google/gemma-4-E4B-it",
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    },
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ── Data Loading ────────────────────────────────────────────────────────────

def load_training_data(data_dir: Path, split: str = "train") -> list[dict]:
    """Load JSONL training data."""
    path = data_dir / f"{split}.jsonl"
    entries = [json.loads(line) for line in open(path)]
    print(f"Loaded {len(entries)} {split} examples from {path}")
    return entries


class VLMDataCollator:
    """Collate function that processes images and text for VLM training.

    Handles the image loading and processor application at collation time
    to avoid storing all processed tensors in memory.
    """

    def __init__(self, processor, max_length: int = 2048):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, batch: list[dict]) -> dict:
        texts = []
        images = []

        for example in batch:
            messages = example["messages"]
            user_msg = messages[0]
            assistant_msg = messages[1]

            # Extract image path from user message
            image_path = None
            user_text = ""
            for content in user_msg["content"]:
                if content["type"] == "image":
                    image_path = str(PROJECT_ROOT / content["image"])
                elif content["type"] == "text":
                    user_text = content["text"]

            # Load image
            img = Image.open(image_path).convert("RGB")
            images.append(img)

            # Build chat-template text
            conversation = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]},
                {"role": "assistant", "content": [{"type": "text", "text": assistant_msg["content"]}]},
            ]
            text = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
            texts.append(text)

        # Process batch through the model's processor
        inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        # For causal LM training, labels = input_ids (shifted internally by the model)
        labels = inputs["input_ids"].clone()

        # Mask user/system tokens so loss is only on assistant response
        for i, text in enumerate(texts):
            # Find where the assistant response starts in tokens
            # Tokenize just the part up to the assistant response
            user_part = text.split(assistant_msg["content"])[0] if assistant_msg["content"] in text else text
            user_tokens = self.processor.tokenizer(user_part, return_tensors="pt", add_special_tokens=False)
            user_len = user_tokens["input_ids"].shape[1]
            labels[i, :user_len] = -100

        # Mask padding tokens
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels

        return inputs


# ── Model Setup ─────────────────────────────────────────────────────────────

def load_model_and_processor(model_key: str, method: str):
    """Load model with optional quantization and LoRA adapters."""
    model_cfg = MODELS[model_key]
    hf_id = model_cfg["hf_id"]
    hf_token = os.environ.get("HF_TOKEN")

    print(f"Loading {hf_id} with method={method}...")

    # Processor (tokenizer + image processor)
    processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True, token=hf_token)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # Quantization config for QLoRA
    bnb_config = None
    if method == "qlora":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    # Load base model
    model = AutoModelForVision2Seq.from_pretrained(
        hf_id,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=hf_token,
        attn_implementation="flash_attention_2",
    )

    # Prepare for k-bit training (QLoRA)
    if method == "qlora":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # Freeze vision encoder — train only the language model adapters
    if hasattr(model, "visual"):
        for param in model.visual.parameters():
            param.requires_grad = False
    elif hasattr(model, "vision_tower"):
        for param in model.vision_tower.parameters():
            param.requires_grad = False
    elif hasattr(model, "vision_model"):
        for param in model.vision_model.parameters():
            param.requires_grad = False

    # LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=model_cfg["lora_target_modules"],
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, processor


# ── Training ────────────────────────────────────────────────────────────────

def run_training(
    model,
    processor,
    train_data: list[dict],
    val_data: list[dict],
    output_dir: Path,
    args: argparse.Namespace,
):
    """Run the training loop."""
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    collator = VLMDataCollator(processor, max_length=args.max_length)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        bf16=True,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    print(f"\nStarting training: {len(train_data)} train, {len(val_data)} val")
    print(f"Output: {output_dir}")
    trainer.train()

    # Save final adapter
    final_dir = output_dir / "final_adapter"
    model.save_pretrained(str(final_dir))
    processor.save_pretrained(str(final_dir))
    print(f"\nAdapter saved to {final_dir}")

    return trainer


# ── Merge & Export ──────────────────────────────────────────────────────────

def merge_and_export(model_key: str, adapter_dir: Path, output_dir: Path):
    """Merge LoRA adapter back into base model and export full bf16 weights."""
    from peft import PeftModel

    model_cfg = MODELS[model_key]
    hf_id = model_cfg["hf_id"]
    hf_token = os.environ.get("HF_TOKEN")

    print(f"Loading base model {hf_id} for merge...")
    base_model = AutoModelForVision2Seq.from_pretrained(
        hf_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        token=hf_token,
    )

    print(f"Loading adapter from {adapter_dir}...")
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))

    print("Merging adapter into base model...")
    merged = model.merge_and_unload()

    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(output_dir))

    processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True, token=hf_token)
    processor.save_pretrained(str(output_dir))

    print(f"Merged model saved to {output_dir}")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fine-tune VLM with LoRA/QLoRA")
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()),
                        help="Model to fine-tune")
    parser.add_argument("--data-format", required=True, choices=["label_only", "full_reasoning"],
                        help="Training data format")
    parser.add_argument("--method", required=True, choices=["lora", "qlora"],
                        help="Training method")
    parser.add_argument("--data-dir", type=Path, default=Path("data/fine_tune"),
                        help="Base directory for training data")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: models/<model>-<format>-<method>)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8,
                        help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=2048,
                        help="Max sequence length for tokenization")
    parser.add_argument("--lora-r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--merge", action="store_true",
                        help="Merge adapter into base model after training")
    args = parser.parse_args()

    # Resolve output directory
    if args.output_dir is None:
        args.output_dir = PROJECT_ROOT / "models" / f"{args.model}-{args.data_format}-{args.method}"

    # Load data
    data_path = args.data_dir / args.data_format
    train_data = load_training_data(data_path, "train")
    val_data = load_training_data(data_path, "val")

    # Load model
    model, processor = load_model_and_processor(args.model, args.method)

    # Train
    trainer = run_training(model, processor, train_data, val_data, args.output_dir, args)

    # Optionally merge
    if args.merge:
        adapter_dir = args.output_dir / "final_adapter"
        merged_dir = args.output_dir / "merged"
        merge_and_export(args.model, adapter_dir, merged_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
