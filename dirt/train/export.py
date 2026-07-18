from __future__ import annotations

import os
from collections.abc import Mapping

import jax
import numpy as np

from dirt.models.config import ModelConfig


def _flatten(d, prefix=""):
    result = {}
    for k, v in d.items():
        path = f"{prefix}/{k}" if prefix else k
        if isinstance(v, Mapping):
            result.update(_flatten(v, path))
        else:
            arr = jax.device_get(v) if isinstance(v, jax.Array) else v
            result[path] = np.asarray(arr)
    return result


def save_safetensors(ckpt_dir: str, params, model_cfg: ModelConfig, step: int) -> None:
    path = os.path.join(ckpt_dir, "model.safetensors")
    flat = _flatten(params)

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
