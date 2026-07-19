from __future__ import annotations

import math
import os
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from jax.experimental.multihost_utils import sync_global_devices
from omegaconf import DictConfig
from tqdm import trange

from dirt.models.config import ModelConfig
from dirt.models.model import DiRTModel
from dirt.train.checkpoint import (
    init_checkpoint,
    replicate_opt_state_scalars,
    sync_checkpoint,
)
from dirt.train.data import create_data_iter
from dirt.train.sharding import (
    create_mesh,
    shard_params,
)

jax.config.update("jax_threefry_partitionable", True)

Array = jax.Array
jtu = jax.tree_util


def _is_leaf(x):
    return isinstance(x, Array)


def build_model_config(cfg: DictConfig) -> ModelConfig:
    m = cfg.model
    return ModelConfig(
        name=m.name,
        vocab_size=m.vocab_size,
        d_model=m.d_model,
        n_blocks=m.n_blocks,
        n_heads=m.n_heads,
        head_dim=m.head_dim,
        d_ffn=m.d_ffn,
        max_seq_len=m.max_seq_len,
        rope_base=m.rope_base,
        rms_norm_eps=m.rms_norm_eps,
        attn_dropout=m.get("attn_dropout", 0.0),
        dtype=m.dtype,
    )


def compute_total_steps(train_cfg: DictConfig, model_cfg: ModelConfig) -> int:
    tokens_per_step = (
        train_cfg.global_batch_size
        * model_cfg.max_seq_len
        * train_cfg.get("grad_accum_steps", 1)
    )
    total_tokens = train_cfg.target_train_tokens
    total = math.ceil(total_tokens / tokens_per_step)
    if train_cfg.max_steps is not None and train_cfg.max_steps > 0:
        total = min(total, train_cfg.max_steps)
    return total


def create_mesh_from_config(train_cfg: DictConfig):
    devices = jax.devices()
    n = len(devices)
    fsdp_size = train_cfg.get("fsdp_size", 8)
    if fsdp_size > n:
        fsdp_size = n
    while n % fsdp_size != 0:
        fsdp_size //= 2
    replica_size = n // fsdp_size
    return create_mesh((replica_size, fsdp_size), ("replica", "data"))


def build_optimizer(
    train_cfg: DictConfig, total_steps: int
) -> Tuple[optax.GradientTransformation, optax.Schedule]:
    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=train_cfg.lr_peak,
        warmup_steps=train_cfg.warmup_steps,
        decay_steps=total_steps,
        end_value=train_cfg.lr_end,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(train_cfg.grad_clip),
        optax.scale_by_adam(
            b1=train_cfg.adam_beta1,
            b2=train_cfg.adam_beta2,
            eps=train_cfg.adam_eps,
        ),
        optax.add_decayed_weights(train_cfg.weight_decay),
        optax.scale_by_schedule(scheduler),
        optax.scale(-1),
    )
    return optimizer, scheduler


def cast_pytree(pytree, dtype: jnp.dtype):
    def _cast(x):
        if isinstance(x, Array):
            return x.astype(dtype)
        return x
    return jtu.tree_map(_cast, pytree)


def count_params(params) -> int:
    return sum(x.size for x in jtu.tree_leaves(params) if isinstance(x, Array))


def run_training(cfg: DictConfig) -> None:
    proc_idx = jax.process_index()
    is_main = proc_idx == 0

    if is_main:
        print(f"processes={jax.process_count()}, devices={jax.device_count()}")
        print(f"cfg={cfg}")

    model_cfg = build_model_config(cfg)
    train_cfg = cfg.train
    data_cfg = cfg.data
    shard_fsdp = train_cfg.get("fsdp", True)
    fsdp_threshold = train_cfg.get("fsdp_threshold", 2**18)

    total_steps = compute_total_steps(train_cfg, model_cfg)
    if is_main:
        print(f"total_steps={total_steps}")

    mesh = create_mesh_from_config(train_cfg)
    if is_main:
        print(f"mesh={mesh}")
    sync_global_devices("mesh_created")

    model = DiRTModel(cfg=model_cfg)

    key = jax.random.PRNGKey(cfg.seed)
    key, init_key = jax.random.split(key)

    dummy = jnp.ones((1, model_cfg.max_seq_len), dtype=jnp.int32)
    variables = model.init(init_key, dummy, train=True)
    params = variables["params"]

    params = cast_pytree(params, jnp.dtype(cfg.model.dtype))

    params = shard_params(params, mesh, shard_fsdp=shard_fsdp, threshold=fsdp_threshold)
    if is_main:
        print(f"params={count_params(params):,}")

    def _sharding_of(x):
        return x.sharding if isinstance(x, Array) else None

    param_shardings = jtu.tree_map(_sharding_of, params, is_leaf=_is_leaf)

    optimizer, lr_schedule = build_optimizer(train_cfg, total_steps)
    opt_state = optimizer.init(params)

    opt_state = replicate_opt_state_scalars(opt_state, mesh)
    opt_state_shardings = jtu.tree_map(_sharding_of, opt_state, is_leaf=_is_leaf)

    mngr, params, opt_state, first_step, ckpt_dir, gcs_target = init_checkpoint(
        cfg.checkpoint_path, train_cfg.checkpoint_dir, model_cfg.name,
        train_cfg.keep_checkpoints, train_cfg.save_every, params, opt_state,
    )

    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(("replica", "data"), None)
    )

    def _constrain_tree(tree, shardings):
        return jtu.tree_map(
            lambda t, s: jax.lax.with_sharding_constraint(t, s)
            if isinstance(t, Array) and s is not None
            else t,
            tree, shardings,
        )

    @partial(jax.jit, donate_argnums=(0, 1))
    def train_step(params, opt_state, x, y, key):
        params = _constrain_tree(params, param_shardings)
        opt_state = _constrain_tree(opt_state, opt_state_shardings)
        x = jax.lax.with_sharding_constraint(x, data_sharding)
        y = jax.lax.with_sharding_constraint(y, data_sharding)

        def loss_fn(p):
            logits, all_metrics = model.apply({"params": p}, x, train=True)
            logits = logits.astype(jnp.float32)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
            per_block = {
                f"block_{i}/{k}": jnp.mean(m[k])
                for i, m in enumerate(all_metrics[:-1])
                for k in ["delta_v", "gate", "review"]
            }
            return loss, {**per_block, **all_metrics[-1]}

        (loss, metrics_agg), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        grads = _constrain_tree(grads, param_shardings)
        grad_norm = optax.global_norm(grads)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        params = cast_pytree(params, jnp.dtype(cfg.model.dtype))
        return params, opt_state, loss, metrics_agg, grad_norm

    @partial(jax.jit, donate_argnums=(0,))
    def eval_step(params, x, y):
        params = _constrain_tree(params, param_shardings)
        x = jax.lax.with_sharding_constraint(x, data_sharding)
        y = jax.lax.with_sharding_constraint(y, data_sharding)

        logits, all_metrics = model.apply({"params": params}, x, train=False)
        logits = logits.astype(jnp.float32)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
        per_block = {
            f"block_{i}/{k}": jnp.mean(m[k])
            for i, m in enumerate(all_metrics[:-1])
            for k in ["delta_v", "gate", "review"]
        }
        return loss, {**per_block, **all_metrics[-1]}

    train_iter = create_data_iter(
        "train", data_cfg, model_cfg.max_seq_len,
        train_cfg.global_batch_size, mesh,
    )
    eval_iter = create_data_iter(
        "val", data_cfg, model_cfg.max_seq_len,
        train_cfg.global_batch_size, mesh,
    )

    if is_main:
        wandb.init(
            project="dirt",
            config={
                "model": dict(model_cfg.__dict__),
                "train": dict(train_cfg),
                "data": dict(data_cfg),
                "total_steps": total_steps,
                "n_params": count_params(params),
                "shard_fsdp": shard_fsdp,
                "fsdp_threshold": fsdp_threshold,
            },
        )

    postfix = {}
    pbar = trange(
        first_step,
        total_steps,
        initial=first_step,
        total=total_steps,
        disable=not is_main,
    )

    for step in pbar:
        if step > 0 and step % train_cfg.eval_every == 0:
            eval_losses = []
            eval_metrics = []
            num_eval = train_cfg.get("eval_batches", 64)
            for _ in range(num_eval):
                x_eval, y_eval = next(eval_iter)
                loss_val, eval_agg = eval_step(params, x_eval, y_eval)
                eval_losses.append(loss_val.item())
                eval_metrics.append({k: v.item() for k, v in eval_agg.items()})
            avg_val = float(np.mean(eval_losses))
            avg_metrics = {k: float(np.mean([m[k] for m in eval_metrics])) for k in eval_metrics[0]}
            postfix["val_loss"] = avg_val
            if is_main:
                wandb.log({"loss/val": avg_val, **{f"metrics/val_{k}": v for k, v in avg_metrics.items()}}, step=step)

        key, step_key = jax.random.split(key)
        x, y = next(train_iter)
        params, opt_state, loss, agg_metrics, grad_norm = train_step(params, opt_state, x, y, step_key)
        loss_val = loss.item()

        if step % train_cfg.save_every == 0:
            sync_checkpoint(mngr, step, params, opt_state, gcs_target)

        if step % train_cfg.log_every == 0:
            lr_val = lr_schedule(step)
            postfix["loss"] = loss_val
            postfix["lr"] = f"{lr_val:.2e}"
            if is_main:
                wandb.log({
                    "loss/train": loss_val,
                    "lr": lr_val,
                    "grad_norm": grad_norm.item(),
                    **{f"metrics/{k}": v.item() for k, v in agg_metrics.items()},
                }, step=step)

        pbar.set_postfix(**postfix)

    pbar.close()
    mngr.wait_until_finished()

    from dirt.train.export import save_safetensors
    save_safetensors(ckpt_dir, params, model_cfg, step)

    if is_main:
        wandb.finish()
