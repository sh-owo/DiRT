from typing import Any

import jax
import jax.numpy as jnp
from omegaconf import DictConfig, OmegaConf

from dirt.inference.sampler import sample_next_token
from dirt.models import DiRTModel, ModelConfig
from dirt.train.checkpointing import restore_train_state
from dirt.train.schedules import compute_total_steps
from dirt.train.state import DiRTTrainState
from dirt.train.trainer import _make_optimizer


def _to_dict(cfg_node: Any) -> dict[str, Any]:
    if isinstance(cfg_node, DictConfig):
        return OmegaConf.to_container(cfg_node, resolve=True)
    return dict(cfg_node)


def _load_tokenizer(tokenizer_model: str):
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise ImportError("sentencepiece is required for generation") from exc
    return spm.SentencePieceProcessor(model_file=tokenizer_model)


def run_generation(cfg: DictConfig) -> str:
    model_cfg_dict = _to_dict(cfg.model)
    train_cfg = _to_dict(cfg.train)
    data_cfg = _to_dict(cfg.data)
    infer_cfg = _to_dict(cfg.infer)

    model_cfg = ModelConfig(**model_cfg_dict)
    model = DiRTModel(model_cfg)

    total_steps = compute_total_steps(train_cfg, data_cfg)
    tx, _ = _make_optimizer(train_cfg, total_steps)

    rng = jax.random.PRNGKey(int(cfg.seed))
    dummy_ids = jnp.zeros((1, model_cfg.max_seq_len), dtype=jnp.int32)
    variables = model.init({"params": rng}, dummy_ids, train=False)
    state = DiRTTrainState.create(apply_fn=model.apply, params=variables["params"], tx=tx)

    ckpt_dir = str(cfg.checkpoint_path or train_cfg["checkpoint_dir"])
    state = restore_train_state(state, ckpt_dir)

    tokenizer = _load_tokenizer(str(data_cfg["tokenizer_model"]))
    prompt = str(infer_cfg.get("prompt", ""))
    prompt_ids = tokenizer.encode(prompt, out_type=int)
    if not prompt_ids:
        prompt_ids = [1]

    max_new_tokens = int(infer_cfg["max_new_tokens"])
    if len(prompt_ids) + max_new_tokens > model_cfg.max_seq_len:
        raise ValueError("prompt + max_new_tokens exceeds max_seq_len")

    generated = list(prompt_ids)

    temperature = float(infer_cfg["temperature"])
    top_p = float(infer_cfg["top_p"])
    top_k = int(infer_cfg["top_k"])

    for _ in range(max_new_tokens):
        ids = jnp.asarray([generated], dtype=jnp.int32)
        logits, _ = model.apply(
            {"params": state.params},
            ids,
            train=False,
        )
        next_logits = logits[0, -1, :]
        rng, token_rng = jax.random.split(rng)
        next_token = int(sample_next_token(token_rng, next_logits, temperature, top_p, top_k))
        generated.append(next_token)

    text = tokenizer.decode(generated)
    print(text)
    return text
