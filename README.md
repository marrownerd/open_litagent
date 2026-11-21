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

Training logs the configuration, progress, loss, learning rate, and gradient norm
to the console and `train.log` in the output directory. The same directory contains
`train_results.json` with the final training metrics and `eval_results.json` with
validation metrics computed from greedy generations.

By default, validation holds out roughly 20% of the source groups per task using
a fixed seed. Variants of the same research question or synthetic study stay
together. Set the fraction with `--eval-fraction`, or pass an independently
held-out JSONL in the same format with `--eval-dataset`.

- **Both tasks** report the fraction of outputs that are valid JSON objects.
- **Search** reports schema validity, action accuracy, provider accuracy on
  examples whose reference calls for a search, and exact JSON object match.
- **Extraction** reports schema validity, exact matches for the full object and
  individual fields, and the fraction of emitted non-null evidence values that
  appear in the abstract after whitespace normalization. Field matching includes
  expected nulls. Grounding measures quote presence.

Rates range from 0 to 1. Each rate has a companion `_count` with its denominator
and is omitted if there are zero eligible examples. The bundled synthetic
dataset measures pipeline behavior. Use expert-labeled held-out examples to
assess research quality.

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
