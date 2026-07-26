from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from scipy.stats import spearmanr
from jax.experimental.multihost_utils import process_allgather

from dirt.inference.generate import _load_tokenizer
from dirt.models.config import ModelConfig, dtype_from_name
from dirt.models.model import DiRTModel
from dirt.train.checkpoint import load_safetensors_checkpoint
from dirt.train.sharding import create_mesh, get_data_shard_fn


def main():
    jax.distributed.initialize()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--n-batches", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    is_main = jax.process_index() == 0

    with open(args.config_path) as f:
        m = yaml.safe_load(f)
    model_cfg = ModelConfig(
        name=m["name"], vocab_size=m["vocab_size"], d_model=m["d_model"],
        n_blocks=m["n_blocks"], n_heads=m["n_heads"], head_dim=m["head_dim"],
        d_ffn=m["d_ffn"], max_seq_len=m["max_seq_len"], rope_base=m["rope_base"],
        rms_norm_eps=m.get("rms_norm_eps", 1e-6),
        attn_dropout=m.get("attn_dropout", 0.0),
        dtype=m["dtype"],
    )
    n_layers = model_cfg.n_blocks
    seq_len = model_cfg.max_seq_len

    devices = jax.devices()
    n_devices = len(devices)
    mesh = create_mesh((1, n_devices), ("replica", "data"))

    params = load_safetensors_checkpoint(str(args.model_path), model_cfg, mesh)
    model = DiRTModel(cfg=model_cfg)

    if is_main:
        print(f"devices={n_devices}, mesh={mesh}")

    tokenizer = _load_tokenizer(str(args.tokenizer_path))
    pad_id = tokenizer.pad_id()

    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft", streaming=True)
    ds_iter = iter(ds)

    n_procs = jax.process_count()
    proc_idx = jax.process_index()
    B_per_proc = args.batch_size // n_procs
    max_positions = args.n_batches * args.batch_size * seq_len

    gate_all = np.zeros((n_layers, max_positions), dtype=np.float32)
    ent_all = np.zeros(max_positions, dtype=np.float32)
    nll_all = np.zeros(max_positions, dtype=np.float32)
    pos_ids = np.zeros((max_positions, 3), dtype=np.int32)
    raw_input_ids = {}
    total = 0

    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(("replica", "data"), None)
    )
    shard_fn = get_data_shard_fn(mesh, data_sharding)

    for batch_id in range(args.n_batches):
        input_ids_list = []
        for _ in range(B_per_proc):
            try:
                sample = next(ds_iter)
            except StopIteration:
                break
            text = ""
            for msg in sample["messages"]:
                text += f"\n{msg['role']}: {msg['content']}"
            ids = tokenizer.encode(text, out_type=int)
            ids = ids[:seq_len]
            pad_len = seq_len - len(ids)
            ids += [int(pad_id)] * pad_len
            input_ids_list.append(ids)

        if not input_ids_list:
            break

        input_ids = np.array(input_ids_list, dtype=np.int32)
        B_local, T = input_ids.shape

        input_sharded = shard_fn(input_ids)
        logits, all_metrics = model.apply(
            {"params": params}, input_sharded, train=False,
        )

        logits_float = logits.astype(jnp.float32)
        probs = jax.nn.softmax(logits_float, axis=-1)
        log_probs = jnp.log(probs + 1e-8)

        entropy_local = -jnp.sum(probs * log_probs, axis=-1)
        nll_local = -log_probs[jnp.arange(input_sharded.shape[0])[:, None], jnp.arange(T)[None, :], input_sharded]

        input_full = np.array(process_allgather(input_sharded))
        entropy_full = np.array(process_allgather(entropy_local))
        nll_full = np.array(process_allgather(nll_local))

        B_full, T = input_full.shape

        if is_main:
            raw_input_ids[batch_id] = input_full

        for L in range(n_layers):
            gate_full = np.array(process_allgather(all_metrics[L]["magnitude_mean"]))
            if is_main:
                gate_all[L, total:total + B_full * T] = gate_full.ravel()

        if is_main:
            ent_all[total:total + B_full * T] = entropy_full.ravel()
            nll_all[total:total + B_full * T] = nll_full.ravel()
            pos_ids[total:total + B_full * T] = np.column_stack([
                np.full(B_full * T, batch_id, dtype=np.int32),
                np.repeat(np.arange(B_full, dtype=np.int32), T),
                np.tile(np.arange(T, dtype=np.int32), B_full),
            ])
            total += B_full * T
            print(f"  batch {batch_id + 1}/{args.n_batches} — {total:,} positions")

    gate_all = gate_all[:, :total]
    ent_all = ent_all[:total]
    nll_all = nll_all[:total]
    pos_ids = pos_ids[:total]

    if not is_main:
        return

    print(f"\nTotal positions analyzed: {total:,}")
    print()

    print("layer | std    | spearman(ent) | spearman(nll) | mean")
    for L in range(n_layers):
        g = gate_all[L]
        s = float(np.std(g))
        if s > 1e-6:
            r_e, _ = spearmanr(g, ent_all)
            r_n, _ = spearmanr(g, nll_all)
            r_e_str = f"{r_e:.4f}"
            r_n_str = f"{r_n:.4f}"
        else:
            r_e = r_n = None
            r_e_str = "  constant "
            r_n_str = "  constant "
        print(f"{L:5d} | {s:.4f} | {r_e_str:>13} | {r_n_str:>13} | {float(np.mean(g)):.4f}")

    print("\n--- binned: entropy decile vs mean gate ---")
    deciles = np.percentile(ent_all, np.linspace(0, 100, 11))
    header = "layer | " + " ".join(f"d{i:2d}" for i in range(10))
    print(header)
    for L in range(n_layers):
        row = [f"L{L:2d}  | "]
        for i in range(10):
            mask = (ent_all >= deciles[i]) & (ent_all < deciles[i + 1])
            val = float(np.mean(gate_all[L][mask])) if mask.sum() > 0 else 0.0
            row.append(f"{val:.3f}")
        print(" ".join(row))

    for L in range(n_layers):
        g = gate_all[L]
        if float(np.std(g)) < 1e-6:
            print(f"\n=== Layer {L} — constant, skip ===")
            continue

        top_idx = np.argsort(g)[-20:]
        bot_idx = np.argsort(g)[:20]

        print(f"\n=== Layer {L} gate 상위 20 ===")
        for idx_entry in top_idx:
            b_id, b, t = pos_ids[idx_entry]
            ids = raw_input_ids[b_id][b]
            ctx_start = max(0, t - 16)
            ctx = tokenizer.decode(ids[ctx_start:t + 1].tolist())
            tok = tokenizer.decode([ids[t]])
            print(f"  g={g[idx_entry]:.4f} | ent={ent_all[idx_entry]:.2f} | ...{ctx} \u27f6 [{tok}]")

        print(f"\n=== Layer {L} gate 하위 20 ===")
        for idx_entry in bot_idx:
            b_id, b, t = pos_ids[idx_entry]
            ids = raw_input_ids[b_id][b]
            ctx_start = max(0, t - 16)
            ctx = tokenizer.decode(ids[ctx_start:t + 1].tolist())
            tok = tokenizer.decode([ids[t]])
            print(f"  g={g[idx_entry]:.4f} | ent={ent_all[idx_entry]:.2f} | ...{ctx} \u27f6 [{tok}]")


if __name__ == "__main__":
    main()
