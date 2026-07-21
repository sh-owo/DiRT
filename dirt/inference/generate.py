from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig, OmegaConf

from dirt.inference.common import load_model
from dirt.inference.sampler import sample_next_token
from dirt.models.config import ModelConfig


def _to_dict(cfg: DictConfig) -> dict:
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)
    return dict(cfg)


def _load_tokenizer(tokenizer_model: str):
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise ImportError("sentencepiece is required for inference") from exc
    return spm.SentencePieceProcessor(model_file=tokenizer_model)


def run_generation(cfg: DictConfig) -> str:
    model_cfg_dict = _to_dict(cfg.model)
    infer_cfg = _to_dict(cfg.inference)

    model_cfg = ModelConfig(
        name=model_cfg_dict["name"],
        vocab_size=model_cfg_dict["vocab_size"],
        d_model=model_cfg_dict["d_model"],
        n_blocks=model_cfg_dict["n_blocks"],
        n_heads=model_cfg_dict["n_heads"],
        head_dim=model_cfg_dict["head_dim"],
        d_ffn=model_cfg_dict["d_ffn"],
        max_seq_len=model_cfg_dict["max_seq_len"],
        rope_base=model_cfg_dict["rope_base"],
        rms_norm_eps=model_cfg_dict["rms_norm_eps"],
        attn_dropout=model_cfg_dict.get("attn_dropout", 0.0),
        dtype=model_cfg_dict["dtype"],
    )

    model, params = load_model(Path(infer_cfg["model_path"]), model_cfg)
    tokenizer = _load_tokenizer(infer_cfg["tokenizer_model"])

    def predict_fn(input_ids, attention_mask):
        return model.apply(
            {"params": params}, input_ids, train=False, attention_mask=attention_mask
        )[0]

    prompt = infer_cfg["prompt"]
    max_new_tokens = infer_cfg.get("max_new_tokens", 256)
    temperature = infer_cfg.get("temperature", 0.8)
    top_p = infer_cfg.get("top_p", 0.95)
    top_k = infer_cfg.get("top_k", 0)

    prompt_ids = tokenizer.encode(prompt, out_type=int)
    max_len = model_cfg.max_seq_len
    cur_len = len(prompt_ids)

    if cur_len >= max_len:
        raise ValueError(
            f"Prompt length ({cur_len}) exceeds max_seq_len ({max_len})"
        )

    buffer = np.zeros((1, max_len), dtype=np.int32)
    buffer[0, :cur_len] = prompt_ids

    attn_mask = np.zeros((1, max_len), dtype=np.int32)
    attn_mask[0, :cur_len] = 1

    rng = jax.random.PRNGKey(0)

    for step in range(max_new_tokens):
        inp = jnp.array(buffer)
        mask_arr = jnp.array(attn_mask)
        logits = predict_fn(inp, mask_arr)
        next_logit = logits[0, cur_len - 1, :]

        rng, sample_rng = jax.random.split(rng)
        next_token = sample_next_token(sample_rng, next_logit, temperature, top_p, top_k)
        next_token = int(next_token)

        buffer[0, cur_len] = next_token
        attn_mask[0, cur_len] = 1
        cur_len += 1

    generated = buffer[0, len(prompt_ids):cur_len].tolist()
    output = tokenizer.decode(generated, out_type=str)
    return output
