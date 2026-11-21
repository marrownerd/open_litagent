"""Memory-conscious QLoRA (CUDA) / LoRA (CPU) SFT of search + extraction JSON.

Training uses synthetic JSONL for a smoke test. Fine-tuning for research quality
requires a larger, verified corpus of expert-labeled abstracts and search traces.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from training_eval import evaluation_example, score_responses, split_for_validation

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune the academic search/extraction agent")
    parser.add_argument("--dataset", default="data/train.jsonl")
    parser.add_argument("--eval-dataset", help="Separate held-out JSONL with the same format")
    parser.add_argument("--eval-fraction", type=float, default=0.2,
                        help="Fraction of source groups held out per task without --eval-dataset")
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
    if not 0 < args.eval_fraction < 1:
        parser.error("eval-fraction must be between 0 and 1")
    if args.eval_dataset:
        eval_path = Path(args.eval_dataset)
        if not eval_path.is_file():
            parser.error(f"Missing {eval_path}")
        if eval_path.samefile(args.dataset):
            parser.error("eval-dataset must be a separate held-out file")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(output_dir / "train.log", encoding="utf-8")],
    )
    logger.info("Training configuration %s", vars(args))

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
    from transformers.trainer_callback import PrinterCallback
    from trl import SFTConfig, SFTTrainer

    mode = ("qlora" if torch.cuda.is_available() else "lora-cpu") if args.mode == "auto" else args.mode
    if mode == "qlora" and not torch.cuda.is_available():
        parser.error("QLoRA 4-bit training requires a CUDA GPU; use --mode lora-cpu")
    logger.info("Resolved training mode %s", mode)

    def load_examples(path):
        examples = load_dataset("json", data_files=path, split="train")
        if not len(examples) or not {"prompt", "completion"} <= set(examples.column_names):
            raise ValueError(f"{path} must contain conversational prompt and completion examples")
        return examples

    dataset = load_examples(args.dataset)
    if args.eval_dataset:
        eval_dataset = load_examples(args.eval_dataset)
    else:
        dataset, eval_dataset = split_for_validation(dataset, args.eval_fraction, seed=42)
    logger.info("Dataset sizes: train=%d, validation=%d", len(dataset), len(eval_dataset))
    dataset = dataset.select_columns(["prompt", "completion"])

    logger.info("Loading tokenizer and model %s", args.model)
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
        logging_first_step=True,
        disable_tqdm=True,
        report_to="none",
        seed=42,
    )

    class TrainingLogger(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if state.is_world_process_zero and logs:
                logger.info("Step %d/%d %s", state.global_step, state.max_steps, logs)

    trainer = SFTTrainer(
        model=model, args=config, train_dataset=dataset,
        processing_class=tokenizer, peft_config=lora,
        callbacks=[TrainingLogger()],
    )
    trainer.remove_callback(PrinterCallback)
    train_result = trainer.train()
    trainer.save_metrics("train", train_result.metrics)
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Saved PEFT adapter to %s", args.output_dir)

    logger.info("Evaluating generated responses on %d examples", len(eval_dataset))
    trainer.model.eval()
    trainer.model.gradient_checkpointing_disable()

    def generate_responses():
        with torch.inference_mode():
            for index, row in enumerate(eval_dataset, start=1):
                task, _, _ = evaluation_example(row)
                inputs = tokenizer.apply_chat_template(
                    row["prompt"], tokenize=True, add_generation_prompt=True,
                    return_tensors="pt", return_dict=True,
                ).to(trainer.model.device)
                generated = trainer.model.generate(
                    **inputs, do_sample=False, use_cache=True,
                    max_new_tokens=160 if task == "search" else 430,
                    pad_token_id=tokenizer.pad_token_id,
                )
                prediction = tokenizer.decode(
                    generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
                )
                logger.info("Validation example %d/%d (%s)", index, len(eval_dataset), task)
                yield row, prediction

    metrics = score_responses(generate_responses())
    trainer.save_metrics("eval", metrics)
    logger.info("Validation metrics %s", metrics)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Training or evaluation failed")
        raise SystemExit(1)
