from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from jax.experimental.multihost_utils import process_allgather

from dirt.models.config import ModelConfig
from dirt.models.model import DiRTModel
from dirt.models.base_model import BaseModel
from dirt.train.checkpoint import load_safetensors_checkpoint
from dirt.train.sharding import create_mesh, get_data_shard_fn


@dataclass
class AnalysisConfig:
    model_size: str
    n_batches: int
    batch_size: int
    seq_len: int
    subsample_size: int
    dirt_config: str
    base_config: str
    tokenizer_path: str
    seeds: list[int]
    checkpoints: dict[str, dict[int, str]]
    cache_dir: str | None = None


def load_analysis_config(path: str) -> AnalysisConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    ckpt_raw = raw["checkpoints"]
    checkpoints = {
        "dirt": {int(k): v for k, v in ckpt_raw["dirt"].items()},
        "base": {int(k): v for k, v in ckpt_raw["base"].items()},
    }
    return AnalysisConfig(
        model_size=raw["model_size"],
        n_batches=raw["n_batches"],
        batch_size=raw["batch_size"],
        seq_len=raw["seq_len"],
        subsample_size=raw.get("subsample_size", 10000),
        dirt_config=raw["dirt_config"],
        base_config=raw["base_config"],
        tokenizer_path=raw["tokenizer_path"],
        seeds=raw["seeds"],
        checkpoints=checkpoints,
        cache_dir=raw.get("cache_dir", None),
    )


def build_model_from_config(config_path: str, model_type: str):
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    cfg = ModelConfig(
        name=raw["name"],
        vocab_size=raw["vocab_size"],
        d_model=raw["d_model"],
        n_blocks=raw["n_blocks"],
        n_heads=raw["n_heads"],
        head_dim=raw["head_dim"],
        d_ffn=raw["d_ffn"],
        max_seq_len=raw["max_seq_len"],
        rope_base=raw["rope_base"],
        rms_norm_eps=raw.get("rms_norm_eps", 1e-6),
        attn_dropout=raw.get("attn_dropout", 0.0),
        dtype=raw["dtype"],
        model_type=model_type,
    )
    if model_type == "base":
        model = BaseModel(cfg=cfg)
    else:
        model = DiRTModel(cfg=cfg)
    return model, cfg


def load_and_shard_model(
    config_path: str,
    ckpt_path: str,
    model_type: str,
    mesh,
    seed: int | None = None,
    cache_dir: str | None = None,
):
    import os
    import tempfile

    model_cfg = _load_model_config(config_path, model_type)
    local_path = str(ckpt_path)

    if local_path.startswith("gs://"):
        import gcsfs
        fs = gcsfs.GCSFileSystem()

        if cache_dir is not None and seed is not None:
            os.makedirs(cache_dir, exist_ok=True)
            fname = f"{model_type}_seed{seed}.safetensors"
            cached = os.path.join(cache_dir, fname)
            if os.path.exists(cached):
                print(f"  Cache hit: {cached}")
            else:
                print(f"  Downloading from GCS → {cached} ...")
                fs.get(local_path, cached)
                print(f"  Download complete.")
            local_path = cached
        else:
            tmp = tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False)
            tmp_path = tmp.name
            tmp.close()
            print(f"  Downloading from GCS to temp file...")
            fs.get(local_path, tmp_path)
            print(f"  Download complete.")
            local_path = tmp_path

    print(f"  Loading safetensors & sharding...")
    params = load_safetensors_checkpoint(local_path, model_cfg, mesh)
    print(f"  Model loaded.")

    if cache_dir is None and local_path != str(ckpt_path):
        os.unlink(local_path)

    return params


def _load_model_config(config_path: str, model_type: str) -> ModelConfig:
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return ModelConfig(
        name=raw["name"],
        vocab_size=raw["vocab_size"],
        d_model=raw["d_model"],
        n_blocks=raw["n_blocks"],
        n_heads=raw["n_heads"],
        head_dim=raw["head_dim"],
        d_ffn=raw["d_ffn"],
        max_seq_len=raw["max_seq_len"],
        rope_base=raw["rope_base"],
        rms_norm_eps=raw.get("rms_norm_eps", 1e-6),
        attn_dropout=raw.get("attn_dropout", 0.0),
        dtype=raw["dtype"],
        model_type=model_type,
    )


@dataclass
class AnalysisData:
    dirt_loss: np.ndarray
    base_loss: np.ndarray
    pos_ids: np.ndarray
    token_ids: dict[int, np.ndarray]

    magnitudes: list[np.ndarray]

    delta_v: list[np.ndarray]
    review: list[np.ndarray]
    direction: list[np.ndarray]
    hidden_dirt: list[np.ndarray]
    hidden_base: list[np.ndarray]

    sentence_hidden_dirt: list[np.ndarray] | None = None
    sentence_hidden_base: list[np.ndarray] | None = None
    sentence_texts: list[str] | None = None

    sent_full_hidden_dirt: list[np.ndarray] | None = None
    sent_full_hidden_base: list[np.ndarray] | None = None
    sent_token_ids: np.ndarray | None = None
    sent_text: str | None = None
    sent_dv_raw: list[np.ndarray] | None = None
    sent_rv_raw: list[np.ndarray] | None = None

    n_tokens: int = 0
    n_subsample: int = 0
    n_layers_dirt: int = 0
    n_layers_base: int = 0
    d_model: int = 0


def create_data_sharding(mesh):
    return jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(("replica", "data"), None)
    )


def build_chunked_batches(
    data_iter, tokenizer, seq_len: int, batch_size: int, n_batches: int
) -> np.ndarray:
    buffer = []
    for sample in data_iter:
        ids = tokenizer.encode(sample["text"], out_type=int)
        buffer.extend(ids)
        if len(buffer) >= n_batches * batch_size * seq_len:
            break
    buffer = np.array(buffer[:n_batches * batch_size * seq_len], dtype=np.int32)
    return buffer.reshape(n_batches, batch_size, seq_len)


def run_inference(model, params, input_sharded, analysis_mode: bool = False):
    return model.apply(
        {"params": params},
        input_sharded,
        train=False,
        analysis_mode=analysis_mode,
    )


def compute_per_token_loss(logits, targets):
    logits_float = logits.astype(jnp.float32)
    log_probs = jax.nn.log_softmax(logits_float, axis=-1)
    nll = -log_probs[
        jnp.arange(targets.shape[0])[:, None],
        jnp.arange(targets.shape[1])[None, :],
        targets,
    ]
    return nll


def gather_across_devices(x):
    return np.array(process_allgather(x))


def subsample_vectors(local_arr, local_mask, B_local: int, T: int, d_model: int):
    flat = local_arr.reshape(-1, d_model)
    n_local = flat.shape[0]
    mask_sum = int(np.sum(local_mask))
    if mask_sum == 0:
        return np.zeros((0, d_model), dtype=np.float32)
    selected = flat[local_mask]
    return selected


def collect_analysis_data(
    dirt_params,
    base_params,
    dirt_model: DiRTModel,
    base_model: BaseModel,
    mesh,
    all_batches: np.ndarray,
    tokenizer,
    pad_id: int,
    seq_len: int,
    n_batches: int,
    batch_size: int,
    subsample_size: int,
    is_main: bool,
    main_rng: np.random.Generator,
) -> Optional[AnalysisData]:
    n_procs = jax.process_count()
    proc_idx = jax.process_index()
    B_per_proc = batch_size // n_procs
    max_positions = n_batches * batch_size * (seq_len - 1)
    n_layers_dirt = dirt_model.cfg.n_blocks
    n_layers_base = base_model.cfg.n_blocks
    d_model = dirt_model.cfg.d_model

    dirt_loss_all = np.zeros(max_positions, dtype=np.float32)
    base_loss_all = np.zeros(max_positions, dtype=np.float32)
    pos_ids_all = np.zeros((max_positions, 3), dtype=np.int32)
    raw_token_ids = {}

    mag_all = [np.zeros(max_positions, dtype=np.float32) for _ in range(n_layers_dirt)]

    subsample_per_batch = max(subsample_size // n_batches, 1)

    delta_v_list = [[] for _ in range(n_layers_dirt)]
    review_list = [[] for _ in range(n_layers_dirt)]
    direction_list = [[] for _ in range(n_layers_dirt)]
    hidden_dirt_list = [[] for _ in range(n_layers_dirt + 1)]
    hidden_base_list = [[] for _ in range(n_layers_base + 1)]

    total = 0
    total_subsample = 0
    sentence_hidden_dirt_arr = None
    sentence_hidden_base_arr = None
    sentence_texts_arr = None
    sent_full_hidden_dirt_arr = None
    sent_full_hidden_base_arr = None
    sent_token_ids_arr = None
    sent_text_str = None

    data_sharding = create_data_sharding(mesh)
    shard_fn = get_data_shard_fn(mesh, data_sharding)

    print(f"  Processing {n_batches} batches ({batch_size} x {seq_len})...")
    print(f"  (First batch includes JIT compilation — may take a few minutes)")

    for batch_id in range(n_batches):
        batch_data = all_batches[batch_id]
        local_input = batch_data[proc_idx * B_per_proc : (proc_idx + 1) * B_per_proc]
        input_sharded = shard_fn(local_input)

        dirt_logits, dirt_metrics = run_inference(
            dirt_model, dirt_params, input_sharded, analysis_mode=True
        )
        base_logits, base_metrics = run_inference(
            base_model, base_params, input_sharded, analysis_mode=True
        )

        dirt_nll = compute_per_token_loss(dirt_logits[:, :-1, :], input_sharded[:, 1:])
        base_nll = compute_per_token_loss(base_logits[:, :-1, :], input_sharded[:, 1:])

        loss_mask = (input_sharded[:, 1:] != pad_id).astype(jnp.float32)

        input_full = gather_across_devices(input_sharded)
        dirt_nll_full = gather_across_devices(dirt_nll)
        base_nll_full = gather_across_devices(base_nll)
        loss_mask_full = gather_across_devices(loss_mask)

        B_full, T = input_full.shape
        n_new_total = B_full * (T - 1)
        valid_flat = loss_mask_full.ravel().astype(bool)
        n_new = int(valid_flat.sum())

        mag_full_list = []
        for L in range(n_layers_dirt):
            mag_full_list.append(
                gather_across_devices(dirt_metrics[L]["magnitude_raw"])[:, :-1, :]
            )

        dv_gathered = []
        rv_gathered = []
        dr_gathered = []
        for L in range(n_layers_dirt):
            dv_gathered.append(gather_across_devices(dirt_metrics[L]["delta_v_raw"])[:, :-1, :].reshape(-1, d_model))
            rv_gathered.append(gather_across_devices(dirt_metrics[L]["review_raw"])[:, :-1, :].reshape(-1, d_model))
            dr_gathered.append(gather_across_devices(dirt_metrics[L]["direction_raw"])[:, :-1, :].reshape(-1, d_model))

        hd_gathered_raw = []
        for L in range(n_layers_dirt):
            hd_gathered_raw.append(gather_across_devices(dirt_metrics[L]["z_L_hidden"]))
        hd_gathered_raw.append(gather_across_devices(dirt_metrics[n_layers_dirt - 1]["x_before_norm_hidden"]))
        hd_gathered = [h[:, :-1, :].reshape(-1, d_model) for h in hd_gathered_raw]

        hb_gathered_raw = []
        for L in range(n_layers_base):
            hb_gathered_raw.append(gather_across_devices(base_metrics[L]["z_L"]))
        hb_gathered_raw.append(gather_across_devices(base_metrics[n_layers_base - 1]["x"]))
        hb_gathered = [h[:, :-1, :].reshape(-1, d_model) for h in hb_gathered_raw]

        if is_main:
            raw_token_ids[batch_id] = input_full
            dirt_loss_all[total:total + n_new] = dirt_nll_full.ravel()[valid_flat]
            base_loss_all[total:total + n_new] = base_nll_full.ravel()[valid_flat]
            pos_flat_full = np.column_stack([
                np.full(n_new_total, batch_id, dtype=np.int32),
                np.repeat(np.arange(B_full, dtype=np.int32), T - 1),
                np.tile(np.arange(1, T, dtype=np.int32), B_full),
            ])
            pos_ids_all[total:total + n_new] = pos_flat_full[valid_flat]

            for L in range(n_layers_dirt):
                mag_all[L][total:total + n_new] = np.abs(mag_full_list[L].ravel())[valid_flat]

            rng_batch = np.random.default_rng(42 + batch_id)
            valid_indices = np.where(valid_flat)[0]
            k = min(subsample_per_batch, n_new)
            sel = rng_batch.choice(n_new, size=k, replace=False)
            batch_indices = valid_indices[sel]
            for L in range(n_layers_dirt):
                delta_v_list[L].append(dv_gathered[L][batch_indices])
                review_list[L].append(rv_gathered[L][batch_indices])
                direction_list[L].append(dr_gathered[L][batch_indices])

            for L in range(n_layers_dirt + 1):
                hidden_dirt_list[L].append(hd_gathered[L][batch_indices])
            for L in range(n_layers_base + 1):
                hidden_base_list[L].append(hb_gathered[L][batch_indices])

            if batch_id == 0:
                sentence_hidden_dirt_arr = [h[0, :3] for h in hd_gathered_raw]
                sentence_hidden_base_arr = [h[0, :3] for h in hb_gathered_raw]
                sentence_texts_arr = []
                for idx in range(3):
                    ids = input_full[idx]
                    text = tokenizer.decode(ids[ids != int(pad_id)].tolist())
                    sentence_texts_arr.append(text)

                sent_full_hidden_dirt_arr = [h[0] for h in hd_gathered_raw]
                sent_full_hidden_base_arr = [h[0] for h in hb_gathered_raw]
                sent_token_ids_arr = input_full[0]
                sent_text_str = tokenizer.decode(
                    sent_token_ids_arr[sent_token_ids_arr != int(pad_id)].tolist()
                )

                sent_dv_raw_arr = [
                    gather_across_devices(dirt_metrics[L]["delta_v_raw"])[0]
                    for L in range(n_layers_dirt)
                ]
                sent_rv_raw_arr = [
                    gather_across_devices(dirt_metrics[L]["review_raw"])[0]
                    for L in range(n_layers_dirt)
                ]

            total_subsample += len(batch_indices)
            total += n_new
            mean_dirt = float(np.mean(dirt_nll_full.ravel()[valid_flat]))
            print(f"  batch {batch_id + 1}/{n_batches} — {total:,} pos, {total_subsample:,} subsampled  (masked mean={mean_dirt:.4f})")

    if not is_main:
        return None

    dirt_loss_all = dirt_loss_all[:total]
    base_loss_all = base_loss_all[:total]
    pos_ids_all = pos_ids_all[:total]
    mag_all = [m[:total] for m in mag_all]

    delta_v = [np.concatenate(arrs)[:subsample_size] for arrs in delta_v_list]
    review = [np.concatenate(arrs)[:subsample_size] for arrs in review_list]
    direction = [np.concatenate(arrs)[:subsample_size] for arrs in direction_list]

    hidden_dirt = []
    for L in range(n_layers_dirt + 1):
        arrs = [a for a in hidden_dirt_list[L] if a.size > 0]
        hidden_dirt.append(np.concatenate(arrs)[:subsample_size])

    hidden_base = []
    for L in range(n_layers_base + 1):
        arrs = [a for a in hidden_base_list[L] if a.size > 0]
        hidden_base.append(np.concatenate(arrs)[:subsample_size])

    token_ids_map = {}
    for b_id, arr in raw_token_ids.items():
        token_ids_map[b_id] = np.array(arr)

    print(f"\nTotal: {total:,} positions, {total_subsample:,} subsampled for vectors")

    return AnalysisData(
            dirt_loss=dirt_loss_all,
            base_loss=base_loss_all,
            pos_ids=pos_ids_all,
            token_ids=token_ids_map,
            magnitudes=mag_all,
            delta_v=delta_v,
            review=review,
            direction=direction,
            hidden_dirt=hidden_dirt,
            hidden_base=hidden_base,
            sentence_hidden_dirt=sentence_hidden_dirt_arr,
            sentence_hidden_base=sentence_hidden_base_arr,
            sentence_texts=sentence_texts_arr,
            sent_full_hidden_dirt=sent_full_hidden_dirt_arr if total > 0 else None,
            sent_full_hidden_base=sent_full_hidden_base_arr if total > 0 else None,
            sent_token_ids=sent_token_ids_arr if total > 0 else None,
            sent_text=sent_text_str if total > 0 else None,
            sent_dv_raw=sent_dv_raw_arr if total > 0 else None,
            sent_rv_raw=sent_rv_raw_arr if total > 0 else None,
            n_tokens=total,
            n_subsample=total_subsample,
            n_layers_dirt=n_layers_dirt,
            n_layers_base=n_layers_base,
            d_model=d_model,
        )


def save_csv(results: dict, path: Path):
    import csv
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["test", "seed", "metric", "value"])
        def _sort_key(item):
            k = item[0]
            return (str(k[0]), str(k[1]))
        for (seed, test_name), metrics in sorted(results.items(), key=_sort_key):
            for metric_name, value in metrics.items():
                is_env = metric_name.startswith("_env_")
                out_test = "env" if is_env else test_name
                out_metric = metric_name[5:] if is_env else metric_name
                if isinstance(value, (int, float, np.integer, np.floating)):
                    writer.writerow([out_test, str(seed), out_metric, float(value)])
                elif is_env and isinstance(value, str):
                    writer.writerow([out_test, str(seed), out_metric, value])
    print(f"CSV saved to {path}")
