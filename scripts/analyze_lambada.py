from __future__ import annotations

import argparse
from pathlib import Path

import jax
import numpy as np

from dirt.inference.generate import _load_tokenizer
from dirt.train.sharding import create_mesh
from dirt.models.model import DiRTModel
from dirt.models.base_model import BaseModel

from dirt.analyze.base import (
    load_analysis_config,
    load_and_shard_model,
    build_model_from_config,
    save_csv,
)


def count_params(params) -> int:
    import jax.tree_util as jtu
    return sum(x.size for x in jtu.tree_leaves(params) if isinstance(x, jax.Array))


def main():
    jax.distributed.initialize()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to analysis YAML config")
    parser.add_argument("--output-dir", type=str, default="outputs/lambada_eval", help="Output directory")
    args = parser.parse_args()

    is_main = jax.process_index() == 0
    output_dir = Path(args.output_dir)

    config = load_analysis_config(args.config)

    devices = jax.devices()
    n_devices = len(devices)
    mesh = create_mesh((1, n_devices), ("replica", "data"))

    if is_main:
        print(f"devices={n_devices}, mesh={mesh}")
        print(f"config: {config.model_size}, seeds={config.seeds}")
        output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = _load_tokenizer(config.tokenizer_path)
    pad_id = tokenizer.pad_id()

    all_results = {}
    results_by_seed = {}
    first_seed_dirt_params = None
    first_seed_base_params = None
    dirt_model_instance = None
    base_model_instance = None

    for seed_idx, seed in enumerate(config.seeds):
        if is_main:
            print(f"\n{'='*60}")
            print(f"Seed {seed} ({seed_idx + 1}/{len(config.seeds)})")
            print(f"{'='*60}")

        ckpt_dirt = config.checkpoints["dirt"][seed]
        ckpt_base = config.checkpoints["base"][seed]

        if is_main:
            print(f"  Loading DiRT model from {ckpt_dirt}")
        dirt_params = load_and_shard_model(
            config.dirt_config, ckpt_dirt, "dirt", mesh,
            seed=seed, cache_dir=config.cache_dir,
        )
        if dirt_model_instance is None:
            dirt_model_instance, _ = build_model_from_config(config.dirt_config, "dirt")

        if is_main:
            print(f"  Loading Base model from {ckpt_base}")
        base_params = load_and_shard_model(
            config.base_config, ckpt_base, "base", mesh,
            seed=seed, cache_dir=config.cache_dir,
        )
        if base_model_instance is None:
            base_model_instance, _ = build_model_from_config(config.base_config, "base")

        if is_main:
            n_dirt = count_params(dirt_params)
            n_base = count_params(base_params)
            print(f"  DiRT params: {n_dirt:,}  |  Base params: {n_base:,}")

        if seed_idx == 0 and is_main:
            print(f"  Running inference (first call = JIT compile, may take a few min)...")

        from dirt.analyze import lambada_eval
        result = lambada_eval.run(
            dirt_params=dirt_params,
            base_params=base_params,
            dirt_model=dirt_model_instance,
            base_model=base_model_instance,
            mesh=mesh,
            tokenizer=tokenizer,
            pad_id=pad_id,
            seq_len=config.seq_len,
            n_batches=config.n_batches,
            batch_size=config.batch_size,
            output_dir=output_dir,
            seed=seed,
            is_main=is_main,
        )

        if not result:
            continue

        result["_env_n_layers_dirt"] = dirt_model_instance.cfg.n_blocks
        result["_env_n_layers_base"] = base_model_instance.cfg.n_blocks
        result["_env_d_model"] = dirt_model_instance.cfg.d_model
        result["_env_n_batches"] = config.n_batches
        result["_env_batch_size"] = config.batch_size
        result["_env_seq_len"] = config.seq_len
        result["_env_dataset"] = "cimec/lambada"

        results_by_seed[seed] = {
            "mean_nll_dirt": result["mean_nll_dirt"],
            "mean_nll_base": result["mean_nll_base"],
            "mean_diff": result["mean_diff"],
        }

        all_results[(seed, "lambada_eval")] = result

        if first_seed_dirt_params is None:
            first_seed_dirt_params = dirt_params
            first_seed_base_params = base_params

    if not is_main:
        return

    csv_path = output_dir / "analysis_results.csv"
    save_csv(all_results, csv_path)

    print(f"\n{'='*60}")
    print("LAMBADA evaluation complete!")
    print(f"Output: {output_dir}")
    print(f"CSV:    {csv_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
