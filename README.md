# NanoGPT

Minimal, from-scratch reproduction of GPT-2 (decoder-only Transformer), trained on the FineWeb-Edu 10B token dataset and evaluated on HellaSwag. Built for readability while staying fast enough to train a real model.

## Highlights

- Clean GPT-2 architecture implementation (`train_gpt2.py`): multi-head causal self-attention, MLP blocks, LayerNorm, learned positional embeddings, weight-tied output head.
- Runs on **single-GPU** and **multi-GPU** setups via PyTorch `DistributedDataParallel` (DDP).
- Mixed-precision training with gradient accumulation.
- Cosine LR schedule with linear warmup.
- HellaSwag validation for downstream sanity-check evaluation.
- Experiment tracking via Weights & Biases (`wandb`).
- Dependency management via `uv` (`pyproject.toml` + `uv.lock`).

## A note on batch size

Original GPT-2 used a total batch size of **2^19 (≈ 524,288) tokens** per optimizer step. Due to compute limitations, this project reduces the total batch size to **2^15 (≈ 32,768) tokens** per step. Gradient accumulation makes up the difference where needed, and the LR schedule is kept consistent with the smaller batch. Everything else in the recipe (architecture, optimizer, schedule shape) follows the original.

## Multi-GPU support

Training scales across multiple GPUs on one node via DDP (see `train_gpt2.py`):

```bash
# Single GPU
python src/nanogpt/train_gpt2.py

# Multi-GPU (single node, e.g. 4 GPUs)
torchrun --standalone --nproc_per_node=4 src/nanogpt/train_gpt2.py

# Multi-node (example: 2 nodes, 4 GPUs each)
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
    --master_addr=<MASTER_IP> --master_port=29500 src/nanogpt/train_gpt2.py
```

Each process is pinned to one GPU. Gradients sync automatically across ranks after each backward pass, and the effective batch scales with GPU count — adjust the gradient-accumulation steps to keep the total token batch at 2^15 (or higher, if more compute is available).

## Project structure

```
.
├── pyproject.toml
├── uv.lock
└── src
    ├── checkpoints
    │   └── latest.pt                  # saved model + optimizer state
    ├── data
    │   └── input.txt                  # raw sample text
    ├── log
    │   └── log.txt                    # training/eval logs
    ├── notebooks
    │   └── play.ipynb                 # scratch/experimentation notebook
    └── nanogpt
        ├── train_gpt2.py              # model definition + training loop (single & multi-GPU)
        ├── fineweb.py                 # downloads & tokenizes FineWeb-Edu into .npy shards
        ├── hellaswag.py               # HellaSwag eval harness
        ├── test.py                    # tests
        ├── hellaswag/
        │   └── hellaswag_val.jsonl    # HellaSwag validation set
        └── data/
            └── edu_fineweb10B/        # tokenized training/val shards (.npy, uint16 token ids)
                ├── edufineweb_train_000001.npy ... 000099.npy
                └── edufineweb_val_000000.npy
```

## How it works — step by step

### 1. Data preparation (`fineweb.py`)
Downloads the FineWeb-Edu 10B-token dataset (via `datasets`), tokenizes it with the GPT-2 BPE tokenizer (`tiktoken`), and writes it out as flat `.npy` shards of `uint16` token IDs under `src/nanogpt/data/edu_fineweb10B/`. One shard is held out as the validation split (`edufineweb_val_000000.npy`); the rest are training shards. Shards are memory-mapped at train time for fast, low-overhead reads.

```bash
uv run python src/nanogpt/fineweb.py
```

### 2. Model definition (`train_gpt2.py`)
Follows the GPT-2 architecture exactly:
- Token embedding + learned positional embedding, summed.
- Stack of Transformer decoder blocks, each:
  - LayerNorm → multi-head causal self-attention → residual add
  - LayerNorm → MLP (4x expansion, GELU) → residual add
- Final LayerNorm → linear head projecting to vocab size.
- Output head weights tied to input token embedding (weight tying), as in the original GPT-2.

### 3. Training loop
- Batches of token sequences loaded from the `.npy` shards.
- Forward pass computes next-token prediction logits; loss is cross-entropy against the shifted sequence.
- Mixed precision (`bfloat16`) speeds up training and cuts memory use.
- Gradients accumulated over micro-batches to reach the target total batch size of 2^15 tokens per optimizer step.
- Global-norm gradient clipping applied before each optimizer step.
- Optimizer: AdamW, with weight decay applied only to 2D+ parameters (matrix weights), not biases or LayerNorm gains.
- LR schedule: linear warmup, then cosine decay to a minimum value.
- Metrics (loss, LR, grad norm, tokens/sec) logged to `src/log/log.txt` and to Weights & Biases.

### 4. Distributed training (multi-GPU)
- `torchrun` spawns one process per GPU.
- Each process holds a full model replica wrapped in `DistributedDataParallel`.
- Each process reads a different slice of the training shards so no two GPUs see the same batch.
- After backward, DDP all-reduces gradients across GPUs before the optimizer step.
- Only rank 0 handles logging and checkpoint saving to avoid duplicated writes.

### 5. Evaluation
- **Validation loss**: periodically computed on the held-out `edufineweb_val_000000.npy` shard, no gradient updates.
- **HellaSwag** (`hellaswag.py`): downstream commonsense-reasoning eval run against `hellaswag/hellaswag_val.jsonl`, giving accuracy as a sanity check that the model is actually learning useful representations, not just minimizing loss.

```bash
uv run python src/nanogpt/hellaswag.py
```

### 6. Checkpointing
Model weights, optimizer state, and step count are periodically saved to `src/checkpoints/latest.pt` so training can resume and the final model can be used for inference/generation.

### 7. Experimentation
`notebooks/play.ipynb` is a scratch notebook for poking at the model, tokenizer, or data outside the main training script.

## Setup

This project uses [`uv`](https://github.com/astral-sh/uv) for dependency management.

```bash
uv sync
```

### Dependencies (`pyproject.toml`)

```
datasets>=5.0.1
matplotlib>=3.11.1
tiktoken>=0.14.0
torch>=2.14.0
torchvision>=0.29.0
tqdm>=4.70.0
transformers>=5.16.1
wandb>=0.29.0
```

## Acknowledgements

Architecture and training recipe based on OpenAI's GPT-2 paper ("Language Models are Unsupervised Multitask Learners"), trained on the FineWeb-Edu dataset and evaluated with HellaSwag.