from __future__ import annotations

import os
from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np

from dirt.models.config import ModelConfig


def _flatten(d, prefix=""):
    result = {}
    for k, v in d.items():
        path = f"{prefix}/{k}" if prefix else k
        if isinstance(v, Mapping):
            result.update(_flatten(v, path))
        else:
            result[path] = np.asarray(v)
    return result


def save_safetensors(ckpt_dir: str, params, model_cfg: ModelConfig, step: int) -> None:
    from jax.experimental.multihost_utils import process_allgather

    params_full = process_allgather(params)

    if jax.process_index() != 0:
        return

    path = os.path.join(ckpt_dir, "model.safetensors")
    flat = _flatten(params_full)

    if path.startswith("gs://"):
        import gcsfs
        from safetensors.flax import save

        fs = gcsfs.GCSFileSystem()
        with fs.open(path, "wb") as f:
            f.write(save(flat))
    else:
        from safetensors.flax import save_file

        save_file(flat, path)

    print(f"Model exported to {path} (step {step})")


if __name__ == "__main__":
    import argparse

    from jax.experimental.multihost_utils import process_allgather

    parser = argparse.ArgumentParser(
        description="Export an Orbax checkpoint step to safetensors"
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        required=True,
        help="Checkpoint step directory (local or gs://..., e.g. .../dirt_700m/57000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output safetensors path (default: <ckpt-dir>/model.safetensors)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="dirt_700m",
        help="Model config name (e.g. dirt_700m, dirt_1B, dirt_3B)",
    )
    args = parser.parse_args()

    from hydra import compose, initialize_config_dir
    from orbax.checkpoint import PyTreeCheckpointer

    from dirt.models.model import DiRTModel
    from dirt.models.config import ModelConfig

    config_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "configs")
    )
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose("config", overrides=[f"model={args.model}"])
    m = cfg.model
    model_cfg = ModelConfig(
        name=m.name, vocab_size=m.vocab_size, d_model=m.d_model,
        n_blocks=m.n_blocks, n_heads=m.n_heads, head_dim=m.head_dim,
        d_ffn=m.d_ffn, max_seq_len=m.max_seq_len, rope_base=m.rope_base,
        rms_norm_eps=m.rms_norm_eps, attn_dropout=m.get("attn_dropout", 0.0),
        dtype=m.dtype,
    )

    model = DiRTModel(cfg=model_cfg)
    dummy = jnp.ones((1, model_cfg.max_seq_len), dtype=jnp.int32)
    variables = model.init(jax.random.PRNGKey(0), dummy, train=True)
    treedef = jax.tree_util.tree_structure(variables["params"])

    ckptr = PyTreeCheckpointer()
    restored = ckptr.restore(args.ckpt_dir)

    params = jax.tree_util.tree_unflatten(treedef, restored[0])

    step = int(os.path.basename(args.ckpt_dir.rstrip("/")))
    output_dir = args.output or args.ckpt_dir.rstrip("/").rsplit("/", 1)[0]

    jax.distributed.initialize()
    save_safetensors(output_dir, params, model_cfg, step)
