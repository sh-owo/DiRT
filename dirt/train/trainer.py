import json
import os
import time
from pathlib import Path
from typing import Any

import flax.jax_utils as jax_utils
import jax
import jax.numpy as jnp
import numpy as np
import optax
from omegaconf import DictConfig, OmegaConf

from dirt.models import ModelConfig, DiRTModel
from dirt.train.checkpointing import restore_train_state, save_train_state
from dirt.train.data import build_batch_iterator, shard_batch_for_devices
from dirt.train.losses import autoregressive_nll, perplexity_from_nll
from dirt.train.schedules import compute_total_steps, create_learning_rate_schedule
from dirt.train.state import DiRTTrainState


def _to_dict(cfg_node: Any) -> dict[str, Any]:
    if isinstance(cfg_node, DictConfig):
        return OmegaConf.to_container(cfg_node, resolve=True)
    return dict(cfg_node)


def _make_optimizer(train_cfg: dict[str, Any], total_steps: int) -> tuple[optax.GradientTransformation, optax.Schedule]:
    lr_schedule = create_learning_rate_schedule(train_cfg, total_steps)
    tx = optax.chain(
        optax.clip_by_global_norm(float(train_cfg["grad_clip"])),
        optax.adamw(
            learning_rate=lr_schedule,
            b1=float(train_cfg["adam_beta1"]),
            b2=float(train_cfg["adam_beta2"]),
            eps=float(train_cfg["adam_eps"]),
            weight_decay=float(train_cfg["weight_decay"]),
        ),
    )
    return tx, lr_schedule


def _init_or_restore_state(
    model: DiRTModel,
    tx: optax.GradientTransformation,
    model_cfg: ModelConfig,
    ckpt_dir: str,
    seed: int,
) -> DiRTTrainState:
    rng = jax.random.PRNGKey(seed)
    dummy_ids = jnp.zeros((1, model_cfg.max_seq_len), dtype=jnp.int32)
    variables = model.init({"params": rng}, dummy_ids, train=False)
    state = DiRTTrainState.create(apply_fn=model.apply, params=variables["params"], tx=tx)
    restored = restore_train_state(state, ckpt_dir)
    return restored


def _make_train_step(model: DiRTModel):
    def train_step(
        state: DiRTTrainState,
        batch: dict[str, jnp.ndarray],
    ) -> tuple[DiRTTrainState, dict[str, jnp.ndarray]]:
        def loss_fn(params: Any) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
            logits, _ = model.apply(
                {"params": params},
                batch["input_ids"],
                train=True,
            )
            nll = autoregressive_nll(logits, batch["input_ids"])
            return nll, {"loss": nll, "nll": nll}

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        grads = jax.lax.pmean(grads, axis_name="data")
        metrics = jax.lax.pmean(metrics, axis_name="data")
        state = state.apply_gradients(grads=grads)
        return state, metrics

    return jax.pmap(train_step, axis_name="data", donate_argnums=(0,))


def _make_eval_step(model: DiRTModel):
    def eval_step(
        state: DiRTTrainState,
        batch: dict[str, jnp.ndarray],
    ) -> dict[str, jnp.ndarray]:
        logits, _ = model.apply(
            {"params": state.params},
            batch["input_ids"],
            train=False,
        )
        nll = autoregressive_nll(logits, batch["input_ids"])
        return jax.lax.pmean({"nll": nll}, axis_name="data")

    return jax.pmap(eval_step, axis_name="data")


def _as_python_metric(value: Any) -> Any:
    arr = np.asarray(value)
    if arr.ndim == 0:
        return float(arr)
    return arr.tolist()


def _run_eval_loop(
    state_repl: DiRTTrainState,
    eval_iter: Any,
    eval_batches: int,
    local_device_count: int,
    eval_step_fn: Any,
) -> dict[str, Any]:
    metric_sums: dict[str, np.ndarray] = {}
    for _ in range(eval_batches):
        host_batch = next(eval_iter)["input_ids"]
        sharded = shard_batch_for_devices(host_batch, local_device_count)
        batch = {"input_ids": jnp.asarray(sharded)}
        metrics = eval_step_fn(state_repl, batch)
        metrics_host = jax.tree_util.tree_map(lambda x: np.asarray(jax_utils.unreplicate(x)), metrics)
        for k, v in metrics_host.items():
            if k not in metric_sums:
                metric_sums[k] = np.array(v, copy=True)
            else:
                metric_sums[k] += np.array(v)

    reduced = {k: v / float(eval_batches) for k, v in metric_sums.items()}
    reduced["ppl"] = perplexity_from_nll(float(reduced["nll"]))
    return {k: _as_python_metric(v) for k, v in reduced.items()}


def run_training(cfg: DictConfig) -> None:
    model_cfg_dict = _to_dict(cfg.model)
    train_cfg = _to_dict(cfg.train)
    data_cfg = _to_dict(cfg.data)

    model_cfg = ModelConfig(**model_cfg_dict)
    model = DiRTModel(model_cfg)

    total_steps = compute_total_steps(train_cfg, data_cfg)
    tx, lr_schedule = _make_optimizer(train_cfg, total_steps)

    ckpt_dir = str(train_cfg["checkpoint_dir"])
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    process_index = jax.process_index()
    process_count = jax.process_count()
    local_device_count = jax.local_device_count()

    global_batch_size = int(train_cfg["global_batch_size"])
    if global_batch_size % process_count != 0:
        raise ValueError("global_batch_size must be divisible by process_count")
    local_batch_size = global_batch_size // process_count
    if local_batch_size % local_device_count != 0:
        raise ValueError("local batch size must be divisible by local_device_count")

    seed = int(cfg.seed)
    state = _init_or_restore_state(model, tx, model_cfg, ckpt_dir, seed)
    state_repl = jax_utils.replicate(state)

    start_step = int(jax.device_get(jax_utils.unreplicate(state_repl.step)))

    train_iter = build_batch_iterator(
        data_cfg=data_cfg,
        split="train",
        batch_size=local_batch_size,
        process_index=process_index,
        process_count=process_count,
        seed=seed + process_index,
    )

    eval_iter = build_batch_iterator(
        data_cfg=data_cfg,
        split="eval",
        batch_size=local_batch_size,
        process_index=process_index,
        process_count=process_count,
        seed=seed + 1000 + process_index,
    )

    train_step_fn = _make_train_step(model)
    eval_step_fn = _make_eval_step(model)

    metrics_file = Path("train_metrics.jsonl")
    if process_index == 0 and not metrics_file.exists():
        metrics_file.write_text("", encoding="utf-8")

    t0 = time.time()
    for step in range(start_step, total_steps):
        host_batch = next(train_iter)["input_ids"]
        sharded = shard_batch_for_devices(host_batch, local_device_count)
        batch = {"input_ids": jnp.asarray(sharded)}

        state_repl, metrics = train_step_fn(state_repl, batch)

        if step % int(train_cfg["log_every"]) == 0:
            metrics_host = jax.tree_util.tree_map(lambda x: np.asarray(jax_utils.unreplicate(x)), metrics)
            nll_val = float(metrics_host["nll"])
            ppl_val = perplexity_from_nll(nll_val)
            lr_val = float(lr_schedule(step))
            elapsed = time.time() - t0

            if process_index == 0:
                payload = {
                    "step": step,
                    "lr": lr_val,
                    "loss": float(metrics_host["loss"]),
                    "nll": nll_val,
                    "ppl": ppl_val,
                    "elapsed_sec": elapsed,
                }
                with metrics_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(payload, ensure_ascii=True) + "\n")
                print(
                    f"step={step} lr={lr_val:.6e} loss={payload['loss']:.4f} "
                    f"nll={payload['nll']:.4f} ppl={payload['ppl']:.3f}"
                )

        if step > 0 and step % int(train_cfg["eval_every"]) == 0:
            eval_metrics = _run_eval_loop(
                state_repl=state_repl,
                eval_iter=eval_iter,
                eval_batches=int(train_cfg["eval_batches"]),
                local_device_count=local_device_count,
                eval_step_fn=eval_step_fn,
            )
            if process_index == 0:
                print(
                    f"eval step={step} nll={float(eval_metrics['nll']):.4f} ppl={float(eval_metrics['ppl']):.3f}"
                )

        if step > 0 and step % int(train_cfg["save_every"]) == 0:
            if process_index == 0:
                unrep = jax_utils.unreplicate(state_repl)
                save_train_state(
                    state=unrep,
                    checkpoint_dir=ckpt_dir,
                    step=step,
                    keep=int(train_cfg["keep_checkpoints"]),
                )

    if process_index == 0:
        unrep = jax_utils.unreplicate(state_repl)
        save_train_state(
            state=unrep,
            checkpoint_dir=ckpt_dir,
            step=total_steps,
            keep=int(train_cfg["keep_checkpoints"]),
        )


def run_evaluation(cfg: DictConfig) -> dict[str, Any]:
    model_cfg_dict = _to_dict(cfg.model)
    train_cfg = _to_dict(cfg.train)
    data_cfg = _to_dict(cfg.data)

    model_cfg = ModelConfig(**model_cfg_dict)
    model = DiRTModel(model_cfg)

    total_steps = compute_total_steps(train_cfg, data_cfg)
    tx, _ = _make_optimizer(train_cfg, total_steps)

    ckpt_dir = str(cfg.checkpoint_path or train_cfg["checkpoint_dir"])
    state = _init_or_restore_state(model, tx, model_cfg, ckpt_dir, int(cfg.seed))
    state_repl = jax_utils.replicate(state)

    process_index = jax.process_index()
    process_count = jax.process_count()
    local_device_count = jax.local_device_count()

    global_batch_size = int(train_cfg["global_batch_size"])
    if global_batch_size % process_count != 0:
        raise ValueError("global_batch_size must be divisible by process_count")
    local_batch_size = global_batch_size // process_count

    eval_iter = build_batch_iterator(
        data_cfg=data_cfg,
        split="eval",
        batch_size=local_batch_size,
        process_index=process_index,
        process_count=process_count,
        seed=int(cfg.seed) + 2000 + process_index,
    )

    eval_step_fn = _make_eval_step(model)

    metrics = _run_eval_loop(
        state_repl=state_repl,
        eval_iter=eval_iter,
        eval_batches=int(train_cfg["eval_batches"]),
        local_device_count=local_device_count,
        eval_step_fn=eval_step_fn,
    )

    if process_index == 0:
        print(json.dumps(metrics, ensure_ascii=True, indent=2))
    return metrics
