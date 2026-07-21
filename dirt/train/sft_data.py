from __future__ import annotations

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
        raise ImportError("sentencepiece is required for SFT data") from exc
    return spm.SentencePieceProcessor(model_file=path)


def _load_stream(name: str, split: str, shuffle_buffer: int, seed: int, streaming: bool = True):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("datasets is required for SFT data") from exc
    ds = load_dataset(name, split=split, streaming=streaming)
    if shuffle_buffer > 0:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
    return iter(ds)


def _format_text(sample: dict, data_cfg: DictConfig) -> str:
    template = data_cfg.get("template", None)
    if template is not None:
        instruction = sample.get("instruction", "")
        output = sample.get("output", "")
        inp = sample.get("input", "")
        text = template.replace("{instruction}", instruction).replace("{output}", output)
        text = text.replace("{input}", inp)
        if not inp.strip():
            text = "\n".join(
                line for line in text.split("\n")
                if "### Input:" not in line
            )
            while "\n\n\n" in text:
                text = text.replace("\n\n\n", "\n\n")
            text = text.strip()
        return text
    return sample.get(data_cfg.get("text_key", "text"), "")


def _count_prompt_tokens(text: str, marker: str, tokenizer) -> int:
    marker_pos = text.rfind(marker)
    if marker_pos == -1:
        return 0
    prompt_part = text[:marker_pos + len(marker)]
    return len(tokenizer.encode(prompt_part, out_type=int))


def create_sft_data_iter(
    split: str,
    data_cfg: DictConfig,
    seq_len: int,
    global_batch_size: int,
    mesh: jax.sharding.Mesh,
) -> Iterator[Tuple[Array, Array, Array, Array]]:
    n_procs = jax.process_count()
    proc_idx = jax.process_index()
    B_per_proc = global_batch_size // n_procs

    data_sharding = NamedSharding(mesh, P(("replica", "data"), None))
    shard_fn = get_data_shard_fn(mesh, data_sharding)

    tokenizer_path = data_cfg.tokenizer_model
    if not os.path.isabs(tokenizer_path):
        tokenizer_path = to_absolute_path(tokenizer_path)
    tokenizer = _load_tokenizer(tokenizer_path)

    eos_id = data_cfg.eos_id
    response_marker = data_cfg.get("response_marker", "### Response:\n")
    eval_percent = data_cfg.get("eval_percent", 5)

    hf_name = data_cfg.hf_name
    hf_split = data_cfg.train_split if split == "train" else data_cfg.eval_split

    if hf_split == data_cfg.eval_split and hf_split == data_cfg.train_split:
        from datasets import load_dataset
        dataset = load_dataset(hf_name, split=hf_split, streaming=False)
        n_total = len(dataset)
        n_eval = int(n_total * eval_percent / 100.0)
        if split == "eval":
            stream = iter(dataset.take(n_eval))
        else:
            train_ds = dataset.skip(n_eval)
            if data_cfg.shuffle_buffer > 0:
                train_ds = train_ds.shuffle(seed=0)
            stream = iter(train_ds)
    else:
        shuffle = data_cfg.shuffle_buffer if split == "train" else 0
        stream = _load_stream(hf_name, hf_split, shuffle, 0)

    while True:
        inp_ids = np.zeros((B_per_proc, seq_len), dtype=np.int32)
        labels = np.zeros((B_per_proc, seq_len), dtype=np.int32)
        attn_mask = np.zeros((B_per_proc, seq_len), dtype=np.int32)
        loss_mask = np.zeros((B_per_proc, seq_len), dtype=np.int32)

        for i in range(B_per_proc):
            try:
                sample = next(stream)
            except StopIteration:
                stream = _load_stream(hf_name, hf_split, data_cfg.shuffle_buffer, 0)
                sample = next(stream)

            text = _format_text(sample, data_cfg)
            full_ids = tokenizer.encode(text, out_type=int) + [eos_id]
            prompt_len = _count_prompt_tokens(text, response_marker, tokenizer)

            N = len(full_ids)
            if N > seq_len + 1:
                excess = N - (seq_len + 1)
                trim_prompt = min(excess, prompt_len)
                full_ids = full_ids[trim_prompt:]
                prompt_len -= trim_prompt
                if len(full_ids) > seq_len + 1:
                    full_ids = full_ids[:seq_len + 1]

            N = len(full_ids)
            n_prompt_targets = max(0, prompt_len - 1)

            x = full_ids[:-1] if N > 1 else full_ids
            y = full_ids[1:] if N > 1 else full_ids

            x_arr = np.array(x + [0] * max(0, seq_len - len(x)), dtype=np.int32)[:seq_len]
            y_arr = np.array(y + [0] * max(0, seq_len - len(y)), dtype=np.int32)[:seq_len]
            attn_arr = np.array([1] * min(N - 1, seq_len) + [0] * max(0, seq_len - (N - 1)), dtype=np.int32)[:seq_len]
            lm_arr = np.array(
                [0] * min(n_prompt_targets, seq_len)
                + [1] * min(N - 1 - n_prompt_targets, seq_len - n_prompt_targets)
                + [0] * max(0, seq_len - (N - 1)),
                dtype=np.int32,
            )[:seq_len]

            inp_ids[i] = x_arr
            labels[i] = y_arr
            attn_mask[i] = attn_arr
            loss_mask[i] = lm_arr

        yield shard_fn(inp_ids), shard_fn(labels), shard_fn(attn_mask), shard_fn(loss_mask)
