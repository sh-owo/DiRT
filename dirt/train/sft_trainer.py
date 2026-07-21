from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from jax.experimental.multihost_utils import sync_global_devices
from omegaconf import DictConfig
from tqdm import trange

from dirt.models.model import DiRTModel
from dirt.train.checkpoint import (
    init_checkpoint,
    replicate_opt_state_scalars,
    sync_checkpoint,
    load_safetensors_checkpoint,
)
from dirt.train.sft_data import create_sft_data_iter
from dirt.train.sharding import shard_params
from dirt.train.trainer import (
    Array,
    build_model_config,
    compute_total_steps,
    create_mesh_from_config,
    build_optimizer,
    cast_pytree,
    count_params,
)
from dirt.train.trainer import _is_leaf as _is_array

jax.config.update("jax_threefry_partitionable", True)

jtu = jax.tree_util


def run_sft_training(cfg: DictConfig) -> None:
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

    pretrained_path = cfg.get("pretrained_path", None)
    if pretrained_path is None:
        raise ValueError(
            "pretrained_path must be provided for SFT training. "
            "Usage: pretrained_path=/path/to/model.safetensors"
        )
    params = load_safetensors_checkpoint(pretrained_path, model_cfg, mesh)
    if is_main:
        print(f"loaded pretrained params from {pretrained_path}")

    if is_main:
        print(f"params={count_params(params):,}")

    def _sharding_of(x):
        return x.sharding if isinstance(x, jax.Array) else None

    param_shardings = jtu.tree_map(_sharding_of, params, is_leaf=_is_array)

    optimizer, lr_schedule = build_optimizer(train_cfg, total_steps)
    opt_state = optimizer.init(params)

    opt_state = replicate_opt_state_scalars(opt_state, mesh)
    opt_state_shardings = jtu.tree_map(_sharding_of, opt_state, is_leaf=_is_array)

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
            if isinstance(t, jax.Array) and s is not None
            else t,
            tree, shardings,
        )

    @partial(jax.jit, donate_argnums=(0, 1))
    def train_step(params, opt_state, x, y, attn_mask, loss_mask, key):
        params = _constrain_tree(params, param_shardings)
        opt_state = _constrain_tree(opt_state, opt_state_shardings)
        x = jax.lax.with_sharding_constraint(x, data_sharding)
        y = jax.lax.with_sharding_constraint(y, data_sharding)
        attn_mask = jax.lax.with_sharding_constraint(attn_mask, data_sharding)
        loss_mask = jax.lax.with_sharding_constraint(loss_mask, data_sharding)

        def loss_fn(p):
            logits, all_metrics = model.apply({"params": p}, x, train=True, attention_mask=attn_mask)
            logits = logits.astype(jnp.float32)
            per_token = optax.softmax_cross_entropy_with_integer_labels(logits, y)
            masked = per_token * loss_mask
            loss = masked.sum() / jnp.maximum(loss_mask.sum(), 1.0)
            per_block = {
                f"block_{i}/{k}": jnp.mean(m[k])
                for i, m in enumerate(all_metrics[:-1])
                for k in ["delta_v", "imp_review", "gate", "review", "out"]
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
    def eval_step(params, x, y, attn_mask, loss_mask):
        params = _constrain_tree(params, param_shardings)
        x = jax.lax.with_sharding_constraint(x, data_sharding)
        y = jax.lax.with_sharding_constraint(y, data_sharding)
        attn_mask = jax.lax.with_sharding_constraint(attn_mask, data_sharding)
        loss_mask = jax.lax.with_sharding_constraint(loss_mask, data_sharding)

        logits, all_metrics = model.apply({"params": params}, x, train=False, attention_mask=attn_mask)
        logits = logits.astype(jnp.float32)
        per_token = optax.softmax_cross_entropy_with_integer_labels(logits, y)
        masked = per_token * loss_mask
        loss = masked.sum() / jnp.maximum(loss_mask.sum(), 1.0)
        per_block = {
            f"block_{i}/{k}": jnp.mean(m[k])
            for i, m in enumerate(all_metrics[:-1])
            for k in ["delta_v", "imp_review", "gate", "review", "out"]
        }
        return loss, {**per_block, **all_metrics[-1]}

    train_iter = create_sft_data_iter(
        "train", data_cfg, model_cfg.max_seq_len,
        train_cfg.global_batch_size, mesh,
    )
    eval_iter = create_sft_data_iter(
        "eval", data_cfg, model_cfg.max_seq_len,
        train_cfg.global_batch_size, mesh,
    )

    if is_main:
        wandb.init(
            project="dirt-sft",
            config={
                "model": dict(model_cfg.__dict__),
                "train": dict(train_cfg),
                "data": dict(data_cfg),
                "total_steps": total_steps,
                "n_params": count_params(params),
                "shard_fsdp": shard_fsdp,
                "fsdp_threshold": fsdp_threshold,
                "pretrained_path": pretrained_path,
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
            num_eval = train_cfg.get("eval_batches", 32)
            for _ in range(num_eval):
                x_eval, y_eval, attn_eval, lm_eval = next(eval_iter)
                loss_val, eval_agg = eval_step(params, x_eval, y_eval, attn_eval, lm_eval)
                eval_losses.append(loss_val.item())
                eval_metrics.append({k: v.item() for k, v in eval_agg.items()})
            avg_val = float(np.mean(eval_losses))
            avg_metrics = {k: float(np.mean([m[k] for m in eval_metrics])) for k in eval_metrics[0]}
            postfix["val_loss"] = avg_val
            if is_main:
                wandb.log({"loss/val": avg_val, **{f"metrics/val_{k}": v for k, v in avg_metrics.items()}}, step=step)

        key = jax.random.PRNGKey(step)
        x, y, attn_batch, lm_batch = next(train_iter)
        params, opt_state, loss, agg_metrics, grad_norm = train_step(
            params, opt_state, x, y, attn_batch, lm_batch, key,
        )
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
