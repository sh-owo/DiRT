from __future__ import annotations

import math
from typing import Any

import optax


def compute_total_steps(train_cfg: dict[str, Any], data_cfg: dict[str, Any]) -> int:
    if train_cfg.get("max_steps") is not None:
        return int(train_cfg["max_steps"])
    tokens_per_step = int(train_cfg["global_batch_size"]) * int(data_cfg["seq_len"])
    target_tokens = int(train_cfg["target_train_tokens"])
    return int(math.ceil(target_tokens / tokens_per_step))


def create_learning_rate_schedule(train_cfg: dict[str, Any], total_steps: int) -> optax.Schedule:
    warmup_steps = int(train_cfg["warmup_steps"])
    peak = float(train_cfg["lr_peak"])
    end = float(train_cfg["lr_end"])
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=peak,
        warmup_steps=warmup_steps,
        decay_steps=max(total_steps - warmup_steps, 1),
        end_value=end,
    )