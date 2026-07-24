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


def _finalize_example(
    ids: list[int], mask: list[int], eos_id: int, seq_len: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = ids + [eos_id]
    mask = mask + [1]

    pad_len = (seq_len + 1) - len(ids)
    if pad_len > 0:
        ids += [0] * pad_len
        mask += [0] * pad_len

    tokens = np.array(ids, dtype=np.int32)
    m = np.array(mask, dtype=np.float32)

    x = tokens[:-1]
    y = tokens[1:]
    loss_mask = m[1:]

    return x, y, loss_mask


def _process_conversation(
    conv: dict, tokenizer, eos_id: int, seq_len: int
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    prompt_ids = tokenizer.encode("Instruction: " + conv["prompt"] + "\n\n", out_type=int)
    if len(prompt_ids) >= seq_len + 1:
        return

    current_ids = list(prompt_ids)
    current_mask = [0] * len(prompt_ids)

    messages = conv["messages"]
    for i in range(0, len(messages) - 1, 2):
        user_msg = messages[i]["content"]
        asst_msg = messages[i + 1]["content"]

        user_prefix = "\nUser: " + user_msg + "\nAssistant: "
        asst_part = asst_msg

        user_prefix_ids = tokenizer.encode(user_prefix, out_type=int)
        asst_ids = tokenizer.encode(asst_part, out_type=int)
        turn_ids = user_prefix_ids + asst_ids

        if len(current_ids) + len(turn_ids) + 1 > seq_len + 1:
            if len(current_ids) > len(prompt_ids):
                yield _finalize_example(current_ids, current_mask, eos_id, seq_len)
            break

        current_ids += turn_ids
        current_mask += [0] * len(user_prefix_ids) + [1] * len(asst_ids)

    if len(current_ids) > len(prompt_ids):
        yield _finalize_example(current_ids, current_mask, eos_id, seq_len)


def create_data_iter(
    split: str,
    data_cfg: DictConfig,
    seq_len: int,
    global_batch_size: int,
    mesh: jax.sharding.Mesh,
) -> Iterator[Tuple[jax.Array, jax.Array, jax.Array]]:
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
            loss_mask = np.ones_like(y, dtype=np.float32)
            yield shard_fn(x), shard_fn(y), shard_fn(loss_mask)

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

        while True:
            batch_x, batch_y, batch_mask = [], [], []

            while len(batch_x) < B_per_proc:
                try:
                    sample = next(stream)
                except StopIteration:
                    stream = _load_stream(
                        data_cfg.hf_name, data_cfg.hf_config,
                        data_cfg.train_split if split == "train" else data_cfg.eval_split,
                        data_cfg.shuffle_buffer, 0,
                    )
                    sample = next(stream)

                for x, y, m in _process_conversation(sample, tokenizer, eos_id, seq_len):
                    batch_x.append(x)
                    batch_y.append(y)
                    batch_mask.append(m)
                    if len(batch_x) >= B_per_proc:
                        break

            x = np.stack(batch_x[:B_per_proc], axis=0)
            y = np.stack(batch_y[:B_per_proc], axis=0)
            loss_mask = np.stack(batch_mask[:B_per_proc], axis=0)

            yield shard_fn(x), shard_fn(y), shard_fn(loss_mask)

    else:
        raise NotImplementedError(
            f"Unknown data backend: {data_cfg.get('backend')}. "
            "Use 'hf_stream' or set use_local_shards=true"
        )
