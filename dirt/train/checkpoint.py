from __future__ import annotations

from typing import Optional, Tuple, Callable

import jax
import orbax.checkpoint as ocp

from dirt.train.sharding import reshard, get_replicated_sharding

Array = jax.Array
jtu = jax.tree_util


def _is_leaf(x):
    return isinstance(x, Array)


def _replicate_scalar(x, mesh):
    if isinstance(x, Array) and x.ndim == 0:
        return reshard(x, get_replicated_sharding(mesh))
    return x


def create_checkpoint_manager(
    ckpt_dir: str,
    max_to_keep: int = 5,
    save_interval_steps: int = 1000,
) -> ocp.CheckpointManager:
    options = ocp.CheckpointManagerOptions(
        max_to_keep=max_to_keep,
        save_interval_steps=save_interval_steps,
    )
    return ocp.CheckpointManager(
        ckpt_dir,
        ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler()),
        options=options,
    )


def save_checkpoint(
    mngr: ocp.CheckpointManager,
    step: int,
    params,
    opt_state,
) -> None:
    mngr.save(step, (jtu.tree_leaves(params), jtu.tree_leaves(opt_state)))


def restore_checkpoint(
    mngr: ocp.CheckpointManager,
    params,
    opt_state,
) -> Tuple[Optional[int], ...]:
    if mngr.latest_step() is None:
        return params, opt_state, 0

    restored = mngr.restore(mngr.latest_step())
    params = jtu.tree_unflatten(jtu.tree_structure(params), restored[0])
    opt_state = jtu.tree_unflatten(jtu.tree_structure(opt_state), restored[1])
    return params, opt_state, mngr.latest_step() + 1


def replicate_opt_state_scalars(opt_state, mesh):
    return jtu.tree_map(
        lambda x: _replicate_scalar(x, mesh),
        opt_state,
        is_leaf=_is_leaf,
    )
