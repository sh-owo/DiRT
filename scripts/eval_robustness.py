from __future__ import annotations

import argparse
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from jax.experimental.multihost_utils import process_allgather

from dirt.inference.generate import _load_tokenizer
from dirt.models.config import ModelConfig
from dirt.models.model import DiRTModel
from dirt.models.base_model import BaseModel
from dirt.train.checkpoint import load_safetensors_checkpoint
from dirt.train.sharding import create_mesh, get_data_shard_fn


def corrupt(clean, p, vocab_size, rng):
    mask = jax.random.bernoulli(rng, p, clean.shape)
    random_tokens = jax.random.randint(rng, clean.shape, 0, vocab_size)
    corrupted = jnp.where(mask, random_tokens, clean)
    return corrupted, mask


def load_model_config(config_path: str) -> ModelConfig:
    with open(config_path) as f:
        m = yaml.safe_load(f)
    return ModelConfig(
        name=m["name"], vocab_size=m["vocab_size"], d_model=m["d_model"],
        n_blocks=m["n_blocks"], n_heads=m["n_heads"], head_dim=m["head_dim"],
        d_ffn=m["d_ffn"], max_seq_len=m["max_seq_len"], rope_base=m["rope_base"],
        rms_norm_eps=m.get("rms_norm_eps", 1e-6),
        attn_dropout=m.get("attn_dropout", 0.0),
        dtype=m["dtype"],
    )


def tokenize_stream(stream, tokenizer, seq_len, eos_id, batch_size, n_procs, proc_idx):
    """Yield batches of (clean_ids,) from streaming dataset."""
    buffer = []
    B_local = batch_size // n_procs
    for sample in stream:
        text = sample.get("text") or sample.get("page") or str(sample)
        ids = tokenizer.encode(text, out_type=int)
        ids = ids + [eos_id]
        buffer.extend(ids)
        while len(buffer) >= B_local * (seq_len + 1):
            chunk = np.array(buffer[: B_local * (seq_len + 1)], dtype=np.int32)
            buffer = buffer[B_local * (seq_len + 1):]
            clean = chunk.reshape(B_local, seq_len + 1)
            yield clean


def main():
    jax.distributed.initialize()

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--model-type", choices=["dirt", "base"], required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--n-batches", type=int, default=50)
    parser.add_argument("--corrupt-probs", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2, 0.3])
    parser.add_argument("--ablate", action="store_true", help="also run force_gate_zero ablation")
    parser.add_argument("--data-path", type=str, default="wikitext")
    parser.add_argument("--data-name", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--data-split", type=str, default="test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    is_main = jax.process_index() == 0
    n_procs = jax.process_count()
    proc_idx = jax.process_index()

    model_cfg = load_model_config(str(args.model_config))
    n_layers = model_cfg.n_blocks
    seq_len = model_cfg.max_seq_len
    vocab_size = model_cfg.vocab_size

    devices = jax.devices()
    n_devices = len(devices)
    mesh = create_mesh((1, n_devices), ("replica", "data"))

    if args.model_type == "dirt":
        model = DiRTModel(cfg=model_cfg)
    else:
        model = BaseModel(cfg=model_cfg)

    if is_main:
        print(f"devices={n_devices}, mesh={mesh}")
        print(f"model={args.model_type}, n_layers={n_layers}, vocab={vocab_size}")
        print(f"checkpoint={args.checkpoint}")

    params = load_safetensors_checkpoint(str(args.checkpoint), model_cfg, mesh)

    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(("replica", "data"), None)
    )
    shard_fn = get_data_shard_fn(mesh, data_sharding)

    tokenizer = _load_tokenizer(str(args.tokenizer))
    eos_id = tokenizer.eos_id() if hasattr(tokenizer, "eos_id") else 1

    from datasets import load_dataset
    ds = load_dataset(args.data_path, args.data_name, split=args.data_split, streaming=True)
    ds_iter = iter(ds)
    data_gen = tokenize_stream(ds_iter, tokenizer, seq_len, eos_id, args.batch_size, n_procs, proc_idx)

    @jax.jit
    def eval_fn(params, corrupted, force_zero):
        logits, all_metrics = model.apply(
            {"params": params}, corrupted, train=False, force_gate_zero=force_zero,
        )
        logits_f32 = logits.astype(jnp.float32)
        return logits_f32, all_metrics

    base_rng = jax.random.PRNGKey(args.seed)

    for p in args.corrupt_probs:
        acc_all = []
        ppl_all = []
        gate_error_all = []
        gate_clean_all = []
        ablate_acc_all = []

        if is_main:
            print(f"\n{'=' * 60}")
            print(f"  p = {p:.2f}")
            print(f"{'=' * 60}")

        for batch_id in range(args.n_batches):
            try:
                clean = next(data_gen)
            except StopIteration:
                break

            B_local, T_full = clean.shape
            clean_in = clean[:, :seq_len]
            clean_target = clean[:, 1:seq_len + 1]

            rng = jax.random.fold_in(base_rng, batch_id)
            corrupted_input, mask = corrupt(clean_in, p, vocab_size, rng)

            corrupted_sharded = shard_fn(corrupted_input)

            logits_f32, all_metrics = eval_fn(params, corrupted_sharded, False)

            loss = optax.softmax_cross_entropy_with_integer_labels(logits_f32, clean_target)
            ppl_batch = jnp.exp(loss.mean())

            preds = jnp.argmax(logits_f32, axis=-1)
            correct = (preds == clean_target) & mask
            acc_batch = correct.sum() / (mask.sum() + 1e-8)

            logits_host = np.array(process_allgather(logits_f32))
            loss_host = np.array(process_allgather(loss))
            mask_host = np.array(process_allgather(mask))
            clean_target_host = np.array(process_allgather(clean_target))

            loss_mean = float(np.mean(loss_host))
            ppl_all.append(float(np.exp(loss_mean)))

            preds_host = np.argmax(logits_host, axis=-1)
            correct_host = (preds_host == clean_target_host) & mask_host
            acc_val = float(correct_host.sum() / max(mask_host.sum(), 1))
            acc_all.append(acc_val)

            if args.model_type == "dirt":
                layer_gates = [
                    np.array(process_allgather(m["magnitude_mean"]))
                    for m in all_metrics[:-1]
                ]
                gate_mean = np.mean(layer_gates, axis=0)
                g_err = float(gate_mean[mask_host].mean()) if mask_host.sum() > 0 else 0.0
                g_clean = float(gate_mean[~mask_host].mean())
                gate_error_all.append(g_err)
                gate_clean_all.append(g_clean)

                if args.ablate:
                    logits_ablate, _ = eval_fn(params, corrupted_sharded, True)
                    logits_ablate_host = np.array(process_allgather(logits_ablate))
                    preds_ablate = np.argmax(logits_ablate_host, axis=-1)
                    correct_ablate = (preds_ablate == clean_target_host) & mask_host
                    acc_abl = float(correct_ablate.sum() / max(mask_host.sum(), 1))
                    ablate_acc_all.append(acc_abl)

            if is_main and batch_id % 10 == 9:
                print(f"  batch {batch_id + 1}/{args.n_batches}  acc={acc_val:.4f}  ppl={float(np.exp(loss_mean)):.2f}")

        if not acc_all:
            if is_main:
                print("  No data processed.")
            continue

        mean_acc = float(np.mean(acc_all))
        mean_ppl = float(np.mean(ppl_all))

        if is_main:
            print(f"  >> acc(corrupted)={mean_acc:.4f}  ppl={mean_ppl:.2f}")

        if args.model_type == "dirt" and gate_error_all:
            mean_g_err = float(np.mean(gate_error_all))
            mean_g_clean = float(np.mean(gate_clean_all))
            if is_main:
                print(f"  >> gate@error={mean_g_err:.4f}  gate@clean={mean_g_clean:.4f}  "
                      f"diff={mean_g_err - mean_g_clean:.4f}")

        if args.model_type == "dirt" and args.ablate and ablate_acc_all:
            mean_abl_acc = float(np.mean(ablate_acc_all))
            if is_main:
                print(f"  >> ablated(gate=0) acc={mean_abl_acc:.4f}  "
                      f"gap={mean_acc - mean_abl_acc:.4f}")


if __name__ == "__main__":
    main()
