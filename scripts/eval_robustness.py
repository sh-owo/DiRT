from __future__ import annotations

import argparse
from itertools import islice
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
    rng_mask, rng_tok = jax.random.split(rng)
    mask = jax.random.bernoulli(rng_mask, p, clean.shape)
    random_tokens = jax.random.randint(rng_tok, clean.shape, 0, vocab_size)
    corrupted = jnp.where(mask, random_tokens, clean)
    return corrupted, mask


def get_rng(base_rng, batch_id, p):
    return jax.random.fold_in(jax.random.fold_in(base_rng, batch_id), int(p * 100 + 0.5))


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


def create_data_gen(data_path, data_name, data_split, tokenizer, seq_len, eos_id, batch_size, n_procs, proc_idx):
    from datasets import load_dataset
    ds = load_dataset(data_path, data_name, split=data_split, streaming=True)
    stream = islice(iter(ds), proc_idx, None, n_procs)
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
            yield chunk.reshape(B_local, seq_len + 1)


def eval_robustness(model, params, make_gen, shard_fn, eval_fn, probs, vocab_size, base_rng, is_main, n_batches, *, force_gate_zero=False):
    results = {}
    for p in probs:
        acc_corrupt_sum, acc_clean_sum = 0.0, 0.0
        ppl_list, batch_count = [], 0
        n_corrupt = 0
        gen = make_gen()
        for batch_id, clean in enumerate(gen):
            if batch_id >= n_batches:
                break
            batch_count += 1

            rng = get_rng(base_rng, batch_id, p)
            corrupted, mask = corrupt(clean, p, vocab_size, rng)

            corrupted_sharded = shard_fn(corrupted)
            clean_sharded = shard_fn(clean)
            logits_f32, _ = eval_fn(params, corrupted_sharded, force_gate_zero)

            pred = jnp.argmax(logits_f32[:, :-1], axis=-1)
            target = clean_sharded[:, 1:]
            mask_in = shard_fn(mask)[:, :-1]
            loss = optax.softmax_cross_entropy_with_integer_labels(logits_f32[:, :-1], target)

            pred_host = np.array(process_allgather(pred))
            target_host = np.array(process_allgather(target))
            mask_in_host = np.array(process_allgather(mask_in))
            loss_host = np.array(process_allgather(loss))

            correct = (pred_host == target_host)
            eps = 1e-8

            acc_corrupt_sum += float((correct & mask_in_host).sum() / max(mask_in_host.sum(), 1))
            acc_clean_sum   += float((correct & ~mask_in_host).sum() / max((~mask_in_host).sum(), 1))
            ppl_list.append(float(np.exp(np.mean(loss_host))))
            n_corrupt += int(mask_in_host.sum())

        if batch_count > 0:
            results[p] = {
                "acc_corrupt": acc_corrupt_sum / batch_count,
                "acc_clean":   acc_clean_sum / batch_count,
                "ppl":         float(np.mean(ppl_list)),
                "n":           n_corrupt,
            }
            if is_main:
                print(f"  p={p:.2f}  acc_corrupt={results[p]['acc_corrupt']:.4f}  "
                      f"acc_clean={results[p]['acc_clean']:.4f}  ppl={results[p]['ppl']:.2f}  n={results[p]['n']}")
    return results


def eval_gate_on_errors(model, params, make_gen, shard_fn, eval_fn, vocab_size, base_rng, is_main, n_batches):
    p = 0.15
    gate_err_all, gate_clean_all = [], []
    gen = make_gen()
    for batch_id, clean in enumerate(gen):
        if batch_id >= n_batches:
            break
        rng = get_rng(base_rng, batch_id, p)
        corrupted, mask = corrupt(clean, p, vocab_size, rng)

        corrupted_sharded = shard_fn(corrupted)
        logits_f32, all_metrics = eval_fn(params, corrupted_sharded, False)

        mask_host = np.array(process_allgather(shard_fn(mask)))

        if all_metrics:
            layer_gates = [
                np.array(process_allgather(m["magnitude_mean"]))
                for m in all_metrics[:-1]
            ]
            if layer_gates:
                gate_mean = np.mean(layer_gates, axis=0)
                g_err = float(gate_mean[mask_host].mean()) if mask_host.sum() > 0 else 0.0
                g_clean = float(gate_mean[~mask_host].mean())
                gate_err_all.append(g_err)
                gate_clean_all.append(g_clean)

    if gate_err_all:
        mean_g_err = float(np.mean(gate_err_all))
        mean_g_clean = float(np.mean(gate_clean_all))
        if is_main:
            print(f"  >> gate at error={mean_g_err:.4f}  gate at clean={mean_g_clean:.4f}  "
                  f"diff={mean_g_err - mean_g_clean:.4f}")
    return {"gate_error": mean_g_err, "gate_clean": mean_g_clean}


def main():
    jax.distributed.initialize()

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--model-type", choices=["dirt", "base"], required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-batches", type=int, default=50)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--corrupt-probs", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2, 0.3])
    parser.add_argument("--gate-p", type=float, default=0.15)
    parser.add_argument("--ablate", action="store_true")
    parser.add_argument("--data-path", type=str, default="wikitext")
    parser.add_argument("--data-name", type=str, default="wikitext-103-raw-v1")
    parser.add_argument("--data-split", type=str, default="test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    is_main = jax.process_index() == 0
    n_procs = jax.process_count()
    proc_idx = jax.process_index()

    model_cfg = load_model_config(str(args.model_config))
    vocab_size = model_cfg.vocab_size
    seq_len = args.seq_len if args.seq_len <= model_cfg.max_seq_len else model_cfg.max_seq_len

    devices = jax.devices()
    n_devices = len(devices)
    mesh = create_mesh((1, n_devices), ("replica", "data"))

    if args.model_type == "dirt":
        model = DiRTModel(cfg=model_cfg)
    else:
        model = BaseModel(cfg=model_cfg)

    if is_main:
        print(f"devices={n_devices}, mesh={mesh}")
        print(f"model={args.model_type}, vocab={vocab_size}")
        print(f"checkpoint={args.checkpoint}")

    params = load_safetensors_checkpoint(str(args.checkpoint), model_cfg, mesh)

    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(("replica", "data"), None)
    )
    shard_fn = get_data_shard_fn(mesh, data_sharding)

    tokenizer = _load_tokenizer(str(args.tokenizer))
    eos_id = tokenizer.eos_id() if hasattr(tokenizer, "eos_id") else 1

    def make_gen():
        return create_data_gen(
            args.data_path, args.data_name, args.data_split,
            tokenizer, seq_len, eos_id, args.batch_size, n_procs, proc_idx,
        )

    @jax.jit
    def eval_fn(params, corrupted, force_zero):
        logits, all_metrics = model.apply(
            {"params": params}, corrupted, train=False, force_gate_zero=force_zero,
        )
        return logits.astype(jnp.float32), all_metrics

    base_rng = jax.random.PRNGKey(args.seed)

    # ============================================================
    # 검증 1: Robustness — 손상 prefix에서 다음 예측
    # ============================================================
    print("\n" + "=" * 60)
    print("  [검증 1] Robustness — corrupt prefix next-token accuracy")
    print("=" * 60)
    results = eval_robustness(
        model, params, make_gen, shard_fn, eval_fn,
        args.corrupt_probs, vocab_size, base_rng, is_main, args.n_batches,
    )

    if is_main:
        print("\n  Summary — p  | acc_corrupt  acc_clean   ppl       n")
        for p, v in results.items():
            print(f"    p={p:.2f}   | {v['acc_corrupt']:.4f}      {v['acc_clean']:.4f}    {v['ppl']:.2f}   {v['n']}")

    # ============================================================
    # 검증 2: Gate 반응 (DiRT 전용)
    # ============================================================
    if args.model_type == "dirt":
        print("\n" + "=" * 60)
        print(f"  [검증 2] Gate at error vs clean (p={args.gate_p:.2f})")
        print("=" * 60)
        eval_gate_on_errors(
            model, params, make_gen, shard_fn, eval_fn,
            vocab_size, base_rng, is_main, args.n_batches,
        )

    # ============================================================
    # 검증 3: Ablation — gate=0 vs normal (DiRT 전용)
    # ============================================================
    if args.model_type == "dirt" and args.ablate:
        print("\n" + "=" * 60)
        print("  [검증 3] Gate Ablation (gate=0)")
        print("=" * 60)
        ablated = eval_robustness(
            model, params, make_gen, shard_fn, eval_fn,
            args.corrupt_probs, vocab_size, base_rng, is_main, args.n_batches,
            force_gate_zero=True,
        )

        if is_main:
            print("\n  Normal vs Ablated (gate=0):")
            print("  p    | normal_ppl  ablated_ppl  gap_ppl   normal_acc  ablated_acc  gap_acc")
            for p in args.corrupt_probs:
                if p in results and p in ablated:
                    nr = results[p]
                    ar = ablated[p]
                    gap_ppl = nr["ppl"] - ar["ppl"]
                    gap_acc = nr["acc_corrupt"] - ar["acc_corrupt"]
                    print(f"  {p:.2f} | {nr['ppl']:.2f}       {ar['ppl']:.2f}        {gap_ppl:+.2f}     "
                          f"{nr['acc_corrupt']:.4f}       {ar['acc_corrupt']:.4f}       {gap_acc:+.4f}")


if __name__ == "__main__":
    main()
