from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.multihost_utils import process_allgather

from dirt.analyze.base import run_inference, compute_per_token_loss, create_data_sharding
from dirt.models.model import DiRTModel
from dirt.models.base_model import BaseModel


def gather_across_devices(x):
    return np.array(process_allgather(x))


def run(
    dirt_params,
    base_params,
    dirt_model: DiRTModel,
    base_model: BaseModel,
    mesh,
    tokenizer,
    pad_id: int,
    seq_len: int,
    n_batches: int,
    batch_size: int,
    output_dir: Path,
    seed: int,
    is_main: bool,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    ds = load_dataset("cimec/lambada", split="test", streaming=True)
    data_iter = iter(ds)

    n_procs = jax.process_count()
    proc_idx = jax.process_index()
    B_per_proc = batch_size // n_procs
    max_samples = n_batches * batch_size

    data_sharding = create_data_sharding(mesh)
    shard_fn = lambda x: jax.lax.with_sharding_constraint(jax.device_put(x), data_sharding)

    dirt_target_nlls = []
    base_target_nlls = []

    total_samples = 0
    for batch_id in range(n_batches):
        batch_texts = []
        batch_ids = []
        batch_target_masks = []

        for _ in range(B_per_proc):
            try:
                sample = next(data_iter)
            except StopIteration:
                break

            text = sample["text"]
            last_space = text.rfind(" ")
            if last_space < 0:
                continue
            context = text[:last_space]

            full_ids = tokenizer.encode(text, out_type=int, add_bos=False, add_eos=False)
            ctx_ids = tokenizer.encode(context, out_type=int, add_bos=False, add_eos=False)

            if len(full_ids) > seq_len:
                full_ids = full_ids[:seq_len]

            target_start = min(len(ctx_ids), len(full_ids))
            if target_start >= len(full_ids):
                continue

            padded = full_ids + [int(pad_id)] * (seq_len - len(full_ids))
            mask = [0] * target_start + [1] * (len(full_ids) - target_start) + [0] * (seq_len - len(full_ids))

            batch_texts.append(text)
            batch_ids.append(padded)
            batch_target_masks.append(mask)

        if not batch_ids:
            break

        x = np.array(batch_ids, dtype=np.int32)
        full_mask = np.array(batch_target_masks, dtype=jnp.float32)

        input_sharded = shard_fn(x)
        loss_mask_sharded = shard_fn(full_mask[:, 1:])

        dirt_logits, _ = run_inference(dirt_model, dirt_params, input_sharded)
        base_logits, _ = run_inference(base_model, base_params, input_sharded)

        dirt_nll = compute_per_token_loss(dirt_logits[:, :-1, :], input_sharded[:, 1:])
        base_nll = compute_per_token_loss(base_logits[:, :-1, :], input_sharded[:, 1:])

        dirt_nll_full = gather_across_devices(dirt_nll)
        base_nll_full = gather_across_devices(base_nll)
        loss_mask_full = gather_across_devices(loss_mask_sharded)

        for i in range(dirt_nll_full.shape[0]):
            mask_i = loss_mask_full[i].astype(bool)
            c = int(mask_i.sum())
            if c == 0:
                continue
            dirt_tgt = float(np.mean(dirt_nll_full[i][mask_i]))
            base_tgt = float(np.mean(base_nll_full[i][mask_i]))
            dirt_target_nlls.append(dirt_tgt)
            base_target_nlls.append(base_tgt)

        total_samples += dirt_nll_full.shape[0]

        if is_main:
            print(f"  batch {batch_id + 1}/{n_batches} — {total_samples} samples")

    if not is_main:
        return {}

    dirt_target_nlls = np.array(dirt_target_nlls)
    base_target_nlls = np.array(base_target_nlls)
    diffs = base_target_nlls - dirt_target_nlls

    mean_dirt = float(np.mean(dirt_target_nlls))
    mean_base = float(np.mean(base_target_nlls))
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    win_count = int(np.sum(diffs > 0))
    n = len(dirt_target_nlls)

    table_path = output_dir / f"lambada_results_seed{seed}.txt"
    with open(table_path, "w") as f:
        f.write(f"LAMBADA Last-Word NLL (seed {seed}, n={n})\n")
        f.write(f"{'sample':>6} | {'dirt_nll':>10} | {'base_nll':>10} | {'diff':>10}\n")
        f.write("-" * 44 + "\n")
        for i in range(min(n, 50)):
            f.write(f"{i:6d} | {dirt_target_nlls[i]:10.6f} | {base_target_nlls[i]:10.6f} | {diffs[i]:+10.6f}\n")
        if n > 50:
            f.write(f"  ... ({n - 50} more samples)\n")
        f.write("-" * 44 + "\n")
        f.write(f"{'mean':>6} | {mean_dirt:10.6f} | {mean_base:10.6f} | {mean_diff:+10.6f}\n")
        f.write(f"{'std':>6} | {float(np.std(dirt_target_nlls, ddof=1)):10.6f} | {float(np.std(base_target_nlls, ddof=1)):10.6f} | {std_diff:10.6f}\n")
        f.write(f"DiRT wins: {win_count}/{n} ({win_count/max(n,1):.1%})\n")

    print(f"\n=== LAMBADA Last-Word NLL (seed {seed}) ===")
    print(f"  n={n}")
    print(f"  DiRT: {mean_dirt:.6f} ± {float(np.std(dirt_target_nlls, ddof=1)):.6f}")
    print(f"  Base: {mean_base:.6f} ± {float(np.std(base_target_nlls, ddof=1)):.6f}")
    print(f"  diff: {mean_diff:+.6f} ± {std_diff:.6f}")
    print(f"  DiRT wins: {win_count}/{n} ({win_count/max(n,1):.1%})")
    print(f"  Table: {table_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig1, ax1 = plt.subplots(figsize=(8, 5))
        bins = np.linspace(min(dirt_target_nlls.min(), base_target_nlls.min()),
                          max(dirt_target_nlls.max(), base_target_nlls.max()), 50)
        ax1.hist(dirt_target_nlls, bins=bins, alpha=0.5, label=f"DiRT (mean={mean_dirt:.4f})", color="#2196F3")
        ax1.hist(base_target_nlls, bins=bins, alpha=0.5, label=f"Base (mean={mean_base:.4f})", color="#FF5722")
        ax1.set_xlabel("Target NLL")
        ax1.set_ylabel("Count")
        ax1.set_title(f"LAMBADA Last-Word NLL Distribution (seed {seed})")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        fig1.tight_layout()
        fig1.savefig(output_dir / f"lambada_nll_histogram_seed{seed}.png", dpi=150)
        plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=(7, 7))
        lim = max(base_target_nlls.max(), dirt_target_nlls.max()) * 1.05
        ax2.scatter(base_target_nlls, dirt_target_nlls, alpha=0.3, s=5, c="steelblue")
        ax2.plot([0, lim], [0, lim], "r--", alpha=0.5, label="y=x")
        ax2.plot([0, lim], [0, lim * 0.98], "g--", alpha=0.3, label="DiRT -2%")
        ax2.set_xlabel("Base NLL")
        ax2.set_ylabel("DiRT NLL")
        ax2.set_title(f"LAMBADA Last-Word NLL: DiRT vs Base (seed {seed})")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, lim)
        ax2.set_ylim(0, lim)
        fig2.tight_layout()
        fig2.savefig(output_dir / f"lambada_nll_scatter_seed{seed}.png", dpi=150)
        plt.close(fig2)

    except ImportError:
        print("  matplotlib not available, skipping plots")

    return {
        "mean_nll_dirt": mean_dirt,
        "mean_nll_base": mean_base,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "win_count": win_count,
        "n_samples": n,
        "nll_dirt_all": dirt_target_nlls.tolist(),
        "nll_base_all": base_target_nlls.tolist(),
    }
