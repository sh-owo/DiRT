from __future__ import annotations

import argparse
from pathlib import Path

import jax
import yaml

from dirt.inference import run_generation
from dirt.inference.common import load_model
from dirt.models.config import ModelConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=Path, required=True, help="Path to model config YAML")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to model.safetensors")
    parser.add_argument("--tokenizer-path", type=Path, required=True, help="Path to SentencePiece tokenizer model")
    parser.add_argument("--prompt", type=str, default="Brain-inspired language model is", help="Input prompt")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling parameter (0 = disabled)")
    args = parser.parse_args()

    with open(args.config_path) as f:
        m = yaml.safe_load(f)
    model_cfg = ModelConfig(
        name=m["name"], vocab_size=m["vocab_size"], d_model=m["d_model"],
        n_blocks=m["n_blocks"], n_heads=m["n_heads"], head_dim=m["head_dim"],
        d_ffn=m["d_ffn"], max_seq_len=m["max_seq_len"], rope_base=m["rope_base"],
        rms_norm_eps=m["rms_norm_eps"], attn_dropout=m.get("attn_dropout", 0.0),
        dtype=m["dtype"],
    )

    infer_cfg = {
        "model_path": str(args.model_path),
        "tokenizer_model": str(args.tokenizer_path),
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
    }

    result = run_generation(model_cfg, infer_cfg)
    if jax.process_index() == 0:
        print(result)


if __name__ == "__main__":
    main()
