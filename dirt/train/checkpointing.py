from __future__ import annotations

from typing import Any

from flax.training import checkpoints


def save_train_state(state: Any, checkpoint_dir: str, step: int, keep: int) -> None:
    checkpoints.save_checkpoint(
        ckpt_dir=checkpoint_dir,
        target=state,
        step=step,
        overwrite=False,
        keep=keep,
    )


def restore_train_state(target: Any, checkpoint_dir: str) -> Any:
    return checkpoints.restore_checkpoint(ckpt_dir=checkpoint_dir, target=target)
