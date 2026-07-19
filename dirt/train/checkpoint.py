from __future__ import annotations

import os
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
    from jax.experimental.multihost_utils import process_allgather

    params_full = process_allgather(params)
    opt_full = process_allgather(opt_state)

    if jax.process_index() == 0:
        mngr.save(step, (jtu.tree_leaves(params_full), jtu.tree_leaves(opt_full)))
    else:
        jax.lax.barrier()


def restore_checkpoint(
    mngr: ocp.CheckpointManager,
    params,
    opt_state,
) -> Tuple[Optional[int], ...]:
    from jax.experimental.multihost_utils import broadcast_one_to_all

    if mngr.latest_step() is None:
        return params, opt_state, 0

    if jax.process_index() == 0:
        restored = mngr.restore(mngr.latest_step())
        p_full = jtu.tree_unflatten(jtu.tree_structure(params), restored[0])
        o_full = jtu.tree_unflatten(jtu.tree_structure(opt_state), restored[1])
    else:
        p_full = None
        o_full = None

    p_full = broadcast_one_to_all(p_full)
    o_full = broadcast_one_to_all(o_full)
    return p_full, o_full, mngr.latest_step() + 1


def replicate_opt_state_scalars(opt_state, mesh):
    return jtu.tree_map(
        lambda x: _replicate_scalar(x, mesh),
        opt_state,
        is_leaf=_is_leaf,
    )


def init_checkpoint(checkpoint_path, checkpoint_dir, model_name, keep, save_interval, params, opt_state):
    base = checkpoint_path or ""
    remote = os.path.join(base, checkpoint_dir, model_name)

    if remote.startswith("gs://"):
        local_base = os.path.abspath(f"/tmp/dirt_ckpt/{checkpoint_dir}")
        ckpt_dir = os.path.join(local_base, model_name)
        gcs_target = remote
    else:
        ckpt_dir = os.path.abspath(remote)
        gcs_target = None

    os.makedirs(ckpt_dir, exist_ok=True)
    mngr = create_checkpoint_manager(ckpt_dir, keep, save_interval)
    return mngr, params, opt_state, 0, ckpt_dir, gcs_target


def sync_checkpoint(mngr, step, params, opt_state, gcs_target):
    save_checkpoint(mngr, step, params, opt_state)
    if gcs_target:
        import gcsfs
        import shutil

        fs = gcsfs.GCSFileSystem()
        fs.put(mngr.directory, gcs_target, recursive=True)
        shutil.rmtree(mngr.directory)
