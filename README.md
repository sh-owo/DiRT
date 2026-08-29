# DiRT (Difference Review Transformer)

Decoder-only LLM trained from scratch in **JAX + Flax** (pre-train, SFT, inference, analysis).

## Key idea

Each layer = **ProposeBlock** + **ReviewBlock** (`dirt/models/blocks/`, `dirt/models/layers.py`):

```
z_L ──▶ ProposeBlock ──▶ new ──┐
                               │
        δv = new − z_L          ▼
        ┌─────────────────▶ ReviewBlock ──▶ out = new + review
z_L ────┘
```

ProposeBlock is a standard attention + SwiGLU FFN sublayer producing `new`. ReviewBlock inspects the difference δv = new − z_L (cross-attends δv against z_L) and applies `review = direction · magnitude`, where `direction` is a normalized SwiGLU projection and `magnitude` a learned scalar gate. `out = new + review`.

## Model configs

`configs/model/` (vocab 32,128 · seq_len 2,048 · bf16).

| Config | d_model | Layers | Heads | d_ffn |
|--------|--------:|-------:|------:|------:|
| dirt_200m | 1024 | 6  | 16 | 2816 |
| dirt_350m | 1024 | 12 | 16 | 2816 |
| dirt_700m | 1024 | 24 | 16 | 2816 |
| dirt_1B   | 1536 | 24 | 24 | 4096 |
| dirt_3B   | 1536 | 48 | 24 | 4096 |

## Usage

```bash
pip install -e .

# prepare tokenizer (T5 SentencePiece) + optional local data shards
python scripts/prepare_tokenizer.py
python scripts/prepare_fineweb.py --tokenizer-model data/tokenizer --seq-len 2048 \
  --tokens-target 500000000 --output-dir data/tokenized --shard-seqs 8192

# pre-train (defaults: dirt_700m + tpu_v4_32 + fineweb_edu)
python scripts/train.py                                  # or: model=dirt_1B data=fineweb_edu

# SFT (defaults: ultrachat_200k + sft_tpu_v4_32)
python scripts/sft.py model=dirt_700m train.pretrained_path=/path/to/ckpt

# export checkpoint to safetensors
python -m dirt.train.export --ckpt-dir checkpoints/dirt/<step> --model dirt_700m
```

**Inference** — top-p / top-k / temperature / repetition-penalty sampling (`dirt/inference/`):

```bash
python scripts/inference.py --config-path configs/model/dirt_700m.yaml \
  --model-path model.safetensors --tokenizer-path data/tokenizer \
  --prompt "Hello!" --max-new-tokens 256 --temperature 0.8 --top-p 0.95
```

**Analysis** — per-layer metrics (delta_v, magnitude, entropy, NLL) on eval data:

```bash
python scripts/analyze.py --config-path configs/model/dirt_700m.yaml \
  --model-path model.safetensors --tokenizer-path data/tokenizer \
  --n-batches 10 --batch-size 128
```

## Data

- Pre-train: **FineWeb-Edu** (streaming, optional local shards)
- SFT: **UltraChat-200k**
- Tokenizer: **T5** SentencePiece

Dependencies in `pyproject.toml` (JAX, Flax, Orbax, Hydra, datasets, sentencepiece).