"""Local Transformers (CUDA 4-bit or CPU FP32) and llama.cpp GGUF inference."""
from __future__ import annotations

from typing import Protocol


DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


class ChatModel(Protocol):
    def generate(self, messages: list[dict[str, str]], max_new_tokens: int = 350) -> str: ...


class TransformersModel:
    def __init__(self, model_name: str = DEFAULT_MODEL, adapter: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            try:
                import bitsandbytes  # noqa: F401
            except ImportError as exc:
                raise RuntimeError("CUDA 4-bit inference requires requirements-gpu.txt") from exc
            config = BitsAndBytesConfig(load_in_4bit=True,
                                        bnb_4bit_quant_type="nf4",
                                        bnb_4bit_compute_dtype=dtype,
                                        bnb_4bit_use_double_quant=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map={"": 0}, quantization_config=config,
                torch_dtype=dtype, low_cpu_mem_usage=True)
        else:
            # 1.5B * 4 bytes ≈ 6GB, fits under 16GB RAM for single-user inference.
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map="cpu", torch_dtype=torch.float32,
                low_cpu_mem_usage=True)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()

    def generate(self, messages: list[dict[str, str]], max_new_tokens: int = 350) -> str:
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True)
        inputs = {name: t.to(self.model.device) for name, t in inputs.items()}
        with self.torch.inference_mode():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id)
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


class GGUFModel:
    def __init__(self, path: str, context_size: int = 4096):
        from llama_cpp import Llama
        self.model = Llama(model_path=path, n_ctx=context_size, n_gpu_layers=0,
                           chat_format="chatml", verbose=False)

    def generate(self, messages: list[dict[str, str]], max_new_tokens: int = 350) -> str:
        result = self.model.create_chat_completion(
            messages=messages, max_tokens=max_new_tokens, temperature=0.0)
        return str(result["choices"][0]["message"]["content"] or "").strip()
