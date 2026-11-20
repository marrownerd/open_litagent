# Prerequisites

Linux x86_64, Python 3.10 or 3.11, 16 GB RAM. For QLoRA, an NVIDIA CUDA GPU with up to 16 GB VRAM and a matching PyTorch/CUDA installation. Network access for Hugging Face model downloads and academic APIs.

# Installation

```bash
cd open_lit_agent
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For NVIDIA 4-bit inference and QLoRA:

```bash
pip install -r requirements-gpu.txt
```

For CPU GGUF inference, install the llama.cpp binding (C++ toolchain required):

```bash
pip install -r requirements-gguf.txt
huggingface-cli download bartowski/Qwen2.5-1.5B-Instruct-GGUF \
  Qwen2.5-1.5B-Instruct-Q4_K_M.gguf --local-dir models
```

OpenAlex basic searches do not require a key. For higher API limits, optionally set:

```bash
export OPENALEX_API_KEY='your_free_openalex_api_key'
```

# Fine-tuning

Create synthetic prompt/completion training examples:

```bash
python make_dataset.py --output data/train.jsonl --n 120
```

Run 4-bit QLoRA on NVIDIA CUDA:

```bash
python train.py --dataset data/train.jsonl --mode qlora \
  --output-dir adapters/academic-qlora --max-length 512 --max-steps 30
```

Alternatively, run full-precision LoRA on a CPU-only 16 GB RAM computer (substantially slower):

```bash
python train.py --dataset data/train.jsonl --mode lora-cpu \
  --output-dir adapters/academic-cpu --max-length 256 --max-steps 5
```

# Run the agent

Run with the base Transformers model on CPU or CUDA:

```bash
python main.py 'Do graph neural networks improve molecular property prediction?' \
  --backend transformers --provider both --max-papers 5
```

Run with a locally fine-tuned PEFT adapter:

```bash
python main.py 'Do graph neural networks improve molecular property prediction?' \
  --backend transformers --adapter adapters/academic-qlora
```

Run on CPU with GGUF (base model only; PEFT adapters are not directly compatible with GGUF):

```bash
python main.py 'Do graph neural networks improve molecular property prediction?' \
  --backend gguf --gguf-path models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf
```

Read `outputs/review.md` and `outputs/review.json`. To run offline integration/unit tests (no model download):

```bash
pytest -q
```
