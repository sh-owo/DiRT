from __future__ import annotations

import glob as glob_module
import os
from typing import Iterator, Tuple

from hydra.utils import to_absolute_path

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig

from dirt.train.sharding import get_data_shard_fn

Array = jax.Array
NamedSharding = jax.sharding.NamedSharding
P = jax.sharding.PartitionSpec


def _load_tokenizer(path: str):
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise ImportError("sentencepiece is required for streaming mode") from exc
    return spm.SentencePieceProcessor(model_file=path)


def _load_stream(name: str, config: str, split: str, shuffle_buffer: int, seed: int):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("datasets is required for streaming mode") from exc
    ds = load_dataset(path=name, name=config, split=split, streaming=True)
    if shuffle_buffer > 0:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
    return iter(ds)


def create_data_iter(
    split: str,
    data_cfg: DictConfig,
    seq_len: int,
    global_batch_size: int,
    mesh: jax.sharding.Mesh,
) -> Iterator[Tuple[Array, Array]]:
    n_procs = jax.process_count()
    proc_idx = jax.process_index()
    B_per_proc = global_batch_size // n_procs

    data_sharding = NamedSharding(mesh, P(("replica", "data"), None))
    shard_fn = get_data_shard_fn(mesh, data_sharding)

    if data_cfg.get("use_local_shards", False):
        pattern = (
            data_cfg.local_train_pattern
            if split == "train"
            else data_cfg.local_eval_pattern
        )
        shard_dir = data_cfg.local_shard_dir
        if not os.path.isabs(shard_dir):
            shard_dir = to_absolute_path(shard_dir)
        full_pattern = os.path.join(shard_dir, pattern)
        shard_paths = sorted(glob_module.glob(full_pattern))
        if not shard_paths:
            raise FileNotFoundError(f"No shards found: {full_pattern}")
        data = np.concatenate([np.load(p).ravel() for p in shard_paths], axis=0)
        n_total = len(data)
        per_proc = n_total // n_procs
        data = data[proc_idx * per_proc : (proc_idx + 1) * per_proc]

        while True:
            ix = np.random.randint(0, len(data) - seq_len - 1, size=(B_per_proc,))
            x = np.take(data, np.arange(seq_len) + ix[:, None], axis=0).astype(np.int32)
            y = np.take(data, np.arange(1, seq_len + 1) + ix[:, None], axis=0).astype(np.int32)
            yield shard_fn(x), shard_fn(y)

    elif data_cfg.get("backend") == "hf_stream":
        tokenizer_path = data_cfg.tokenizer_model
        if not os.path.isabs(tokenizer_path):
            tokenizer_path = to_absolute_path(tokenizer_path)
        tokenizer = _load_tokenizer(tokenizer_path)
        stream = _load_stream(
            data_cfg.hf_name,
            data_cfg.hf_config,
            data_cfg.train_split if split == "train" else data_cfg.eval_split,
            data_cfg.shuffle_buffer,
            0,
        )
        eos_id = data_cfg.eos_id
        text_key = data_cfg.text_key

        need_tokens = B_per_proc * (seq_len + 1)
        buffer = []
        while True:
            while len(buffer) < need_tokens:
                try:
                    sample = next(stream)
                except StopIteration:
                    stream = _load_stream(
                        data_cfg.hf_name, data_cfg.hf_config,
                        data_cfg.train_split if split == "train" else data_cfg.eval_split,
                        data_cfg.shuffle_buffer, 0,
                    )
                    sample = next(stream)
                ids = tokenizer.encode(sample[text_key], out_type=int)
                ids.append(eos_id)
                buffer.extend(ids)

            tokens = np.asarray(buffer[:need_tokens], dtype=np.int32)
            buffer = buffer[need_tokens:]
            batch = tokens.reshape(B_per_proc, seq_len + 1)
            x = batch[:, :-1]
            y = batch[:, 1:]
            yield shard_fn(x), shard_fn(y)

    else:
        raise NotImplementedError(
            f"Unknown data backend: {data_cfg.get('backend')}. "
            "Use 'hf_stream' or set use_local_shards=true"
        )
