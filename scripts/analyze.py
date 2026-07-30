from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from dirt.analyze import gain_magnitude, layer_comparison, magnitude_cv, magnitude_difficulty, position_ppl, propose_review_cosine, seed_verification
from dirt.inference.generate import _load_tokenizer
from dirt.train.sharding import create_mesh
from dirt.models.model import DiRTModel
from dirt.models.base_model import BaseModel

from dirt.analyze.base import (
    load_analysis_config,
    load_and_shard_model,
    build_model_from_config,
    collect_analysis_data,
    build_chunked_batches,
    save_csv,
)

from dirt.analyze import (
    trajectory_pca,
)


def count_params(params) -> int:
    import jax.tree_util as jtu
    return sum(x.size for x in jtu.tree_leaves(params) if isinstance(x, jax.Array))


def _is_leaf(x):
    return isinstance(x, jax.Array)


def main():
    jax.distributed.initialize()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to analysis YAML config")
    parser.add_argument("--output-dir", type=str, default="outputs/analysis", help="Output directory")
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

    from datasets import load_dataset
    ds = load_dataset(
        config.dataset_hf_name,
        config.dataset_hf_config,
        split=config.dataset_split,
        streaming=True,
    )

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
            base_model_instance, base_cfg = build_model_from_config(config.base_config, "base")

        if is_main:
            n_dirt = count_params(dirt_params)
            n_base = count_params(base_params)
            print(f"  DiRT params: {n_dirt:,}  |  Base params: {n_base:,}")
        if seed_idx == 0 and is_main:
            print(f"  Running inference (first call = JIT compile, may take a few min)...")

        ds_iter = iter(ds)
        rng = np.random.default_rng(42 + seed_idx)

        all_batches = build_chunked_batches(
            ds_iter, tokenizer,
            seq_len=config.seq_len,
            batch_size=config.batch_size,
            n_batches=config.n_batches,
            text_key=config.dataset_text_key,
            eos_id=config.dataset_eos_id,
        )

        data = collect_analysis_data(
            dirt_params=dirt_params,
            base_params=base_params,
            dirt_model=dirt_model_instance,
            base_model=base_model_instance,
            mesh=mesh,
            all_batches=all_batches,
            tokenizer=tokenizer,
            pad_id=pad_id,
            seq_len=config.seq_len,
            n_batches=config.n_batches,
            batch_size=config.batch_size,
            subsample_size=config.subsample_size,
            is_main=is_main,
            main_rng=rng,
        )

        if first_seed_dirt_params is None:
            first_seed_dirt_params = dirt_params
            first_seed_base_params = base_params

        if not is_main or data is None:
            continue

        seed_output = output_dir / f"seed_{seed}"
        seed_output.mkdir(parents=True, exist_ok=True)

        mean_dirt_loss = float(np.mean(data.dirt_loss))
        mean_base_loss = float(np.mean(data.base_loss))
        results_by_seed[seed] = {
            "val_loss_dirt": mean_dirt_loss,
            "val_loss_base": mean_base_loss,
        }

        per_seed_tests = [
            ("position_ppl", position_ppl),
            ("magnitude_cv", magnitude_cv),
            ("magnitude_difficulty", magnitude_difficulty),
            ("propose_review_cosine", propose_review_cosine),
            ("gain_magnitude", gain_magnitude),
            ("trajectory_pca", trajectory_pca),
        ]

        for test_name, mod in per_seed_tests:
            if is_main:
                print(f"  Running {test_name}...")

            kwargs = {}
            if test_name == "position_ppl":
                kwargs = dict(dirt_loss=data.dirt_loss, base_loss=data.base_loss,
                              pos_ids=data.pos_ids, output_dir=seed_output, seed=seed)
            elif test_name == "magnitude_cv":
                kwargs = dict(magnitudes=data.magnitudes, n_layers=data.n_layers_dirt,
                              output_dir=seed_output, seed=seed)
            elif test_name == "magnitude_difficulty":
                kwargs = dict(magnitudes=data.magnitudes, dirt_loss=data.dirt_loss,
                              pos_ids=data.pos_ids, token_ids=data.token_ids,
                              n_layers=data.n_layers_dirt, output_dir=seed_output,
                              seed=seed, tokenizer=tokenizer)
            elif test_name == "propose_review_cosine":
                kwargs = dict(delta_v=data.delta_v, review=data.review,
                              direction=data.direction, n_layers=data.n_layers_dirt,
                              output_dir=seed_output, seed=seed,
                              hidden_base=data.hidden_base,
                              n_layers_base=data.n_layers_base)
            elif test_name == "gain_magnitude":
                kwargs = dict(dirt_loss=data.dirt_loss, base_loss=data.base_loss,
                              magnitudes=data.magnitudes, n_layers=data.n_layers_dirt,
                              output_dir=seed_output, seed=seed)
            elif test_name == "trajectory_pca":
                kwargs = dict(hidden_dirt=data.hidden_dirt, hidden_base=data.hidden_base,
                              n_layers_dirt=data.n_layers_dirt,
                              n_layers_base=data.n_layers_base,
                              output_dir=seed_output, seed=seed,
                              delta_v=data.delta_v, review=data.review,
                              sentence_hidden_dirt=data.sentence_hidden_dirt,
                              sentence_hidden_base=data.sentence_hidden_base,
                              sentence_texts=data.sentence_texts,
                              tokenizer=tokenizer,
                              sent_full_hidden_dirt=data.sent_full_hidden_dirt,
                              sent_full_hidden_base=data.sent_full_hidden_base,
                              sent_token_ids=data.sent_token_ids,
                              sent_text=data.sent_text,
                              sent_dv_raw=data.sent_dv_raw,
                              sent_rv_raw=data.sent_rv_raw)

            result = mod.run(**kwargs)

            result["_env_n_tokens"] = data.n_tokens
            result["_env_n_subsample"] = data.n_subsample
            result["_env_n_layers_dirt"] = data.n_layers_dirt
            result["_env_n_layers_base"] = data.n_layers_base
            result["_env_d_model"] = data.d_model
            result["_env_n_batches"] = config.n_batches
            result["_env_batch_size"] = config.batch_size
            result["_env_seq_len"] = config.seq_len
            result["_env_dataset"] = f"{config.dataset_hf_name}/{config.dataset_hf_config}"
            if test_name == "position_ppl":
                result["_env_n_deciles"] = 10
            if test_name == "trajectory_pca":
                result["_env_n_pc"] = 5
            if test_name == "gain_magnitude":
                result["_env_gain_pct"] = "top/bottom 10%"
            if test_name == "magnitude_difficulty":
                result["_env_position_bands"] = "pos=0,pos1-10,pos10-50,pos50+"

            all_results[(seed, test_name)] = result

        all_results[(seed, "overall")] = {
            "mean_dirt_loss": mean_dirt_loss,
            "mean_base_loss": mean_base_loss,
            "mean_gain": mean_base_loss - mean_dirt_loss,
        }

    if not is_main or not results_by_seed:
        return

    seed_results_for_aggregation = {}
    for s in config.seeds:
        if s in results_by_seed:
            seed_results_for_aggregation[s] = results_by_seed[s]

    agg_dir = output_dir / "aggregated"
    agg_dir.mkdir(parents=True, exist_ok=True)

    result_seed = seed_verification.run_aggregate(seed_results_for_aggregation, agg_dir)
    result_seed["_env_n_seeds"] = len(seed_results_for_aggregation)
    result_seed["_env_seeds"] = ",".join(str(s) for s in sorted(seed_results_for_aggregation.keys()))
    all_results[("all", "seed_verification")] = result_seed

    dirt_n = count_params(first_seed_dirt_params)
    base_n = count_params(first_seed_base_params)
    first_seed_mean_dirt = results_by_seed.get(config.seeds[0], {}).get("val_loss_dirt", 0.0)
    first_seed_mean_base = results_by_seed.get(config.seeds[0], {}).get("val_loss_base", 0.0)
    result_layer = layer_comparison.run(
        dirt_val_loss=first_seed_mean_dirt,
        base_val_loss=first_seed_mean_base,
        dirt_n_params=dirt_n,
        base_n_params=base_n,
        output_dir=agg_dir,
    )
    all_results[("all", "layer_comparison")] = result_layer
    result_layer["_env_dirt_config"] = str(config.dirt_config)
    result_layer["_env_base_config"] = str(config.base_config)

    csv_path = output_dir / "analysis_results.csv"
    save_csv(all_results, csv_path)

    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"Output: {output_dir}")
    print(f"CSV:    {csv_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
