from __future__ import annotations

from pathlib import Path

from safetensors.flax import load_file

from dirt.models.config import ModelConfig
from dirt.models.model import DiRTModel


def load_model(model_path: Path, model_cfg: ModelConfig) -> tuple[DiRTModel, dict]:
    flat = load_file(str(model_path))
    params = _unflatten(flat)
    model = DiRTModel(cfg=model_cfg)
    return model, params


def _unflatten(flat: dict) -> dict:
    result = {}
    for key, value in flat.items():
        parts = key.split("/")
        d = result
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value
    return result
