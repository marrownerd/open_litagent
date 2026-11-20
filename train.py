"""Memory-conscious QLoRA (CUDA) / LoRA (CPU) SFT of search + extraction JSON.

Training uses synthetic JSONL for a smoke test. Fine-tuning for research quality
requires a larger, verified corpus of expert-labeled abstracts and search traces.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune the academic search/extraction agent")
    parser.add_argument("--dataset", default="data/train.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output-dir", default="adapters/academic-qlora")
    parser.add_argument("--mode", choices=["auto", "qlora", "lora-cpu"], default="auto")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    args = parser.parse_args()
    if args.max_length < 128 or args.gradient_accumulation < 1:
        parser.error("max-length must be >= 128 and gradient-accumulation >= 1")
    if not Path(args.dataset).exists():
        parser.error(f"Missing {args.dataset}; run make_dataset.py first")

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    mode = ("qlora" if torch.cuda.is_available() else "lora-cpu") if args.mode == "auto" else args.mode
    if mode == "qlora" and not torch.cuda.is_available():
        parser.error("QLoRA 4-bit training requires a CUDA GPU; use --mode lora-cpu")
    if mode == "lora-cpu" and torch.cuda.is_available():
        print("CPU LoRA explicitly selected; CUDA will not be used.")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if mode == "qlora":
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("Run pip install -r requirements-gpu.txt for CUDA QLoRA") from exc
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=dtype,
                                 bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, device_map={"": 0}, quantization_config=cfg,
            torch_dtype=dtype, low_cpu_mem_usage=True)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        # Full-precision frozen 1.5B weights ~6 GB; LoRA trainable matrices are small.
        # CPU mode is much slower than CUDA QLoRA, but avoids GPU requirements.
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float32, low_cpu_mem_usage=True,
            device_map="cpu")
    model.config.use_cache = False
    lora = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    dataset = load_dataset("json", data_files=args.dataset, split="train")
    if "prompt" not in dataset.column_names or "completion" not in dataset.column_names:
        raise ValueError("Training data must contain conversational prompt and completion columns")
    dataset = dataset.remove_columns([column for column in dataset.column_names
                                      if column not in {"prompt", "completion"}])

    config = SFTConfig(
        output_dir=args.output_dir,
        max_length=args.max_length,
        completion_only_loss=True,
        eos_token="<|im_end|>",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        optim="adamw_torch",
        fp16=mode == "qlora" and dtype == torch.float16,
        bf16=mode == "qlora" and dtype == torch.bfloat16,
        use_cpu=mode == "lora-cpu",
        packing=False,
        save_strategy="no",
        logging_steps=1,
        report_to="none",
        seed=42,
    )
    trainer = SFTTrainer(
        model=model, args=config, train_dataset=dataset,
        processing_class=tokenizer, peft_config=lora,
    )
    trainer.train()
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved PEFT adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
