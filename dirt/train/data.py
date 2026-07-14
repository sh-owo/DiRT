from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np


def _iter_local_sequences(
    shard_dir: Path,
    pattern: str,
    seq_len: int,
    seed: int,
    repeat: bool,
    process_index: int,
    process_count: int,
) -> Iterator[np.ndarray]:
    rng = np.random.default_rng(seed)
    files = sorted(shard_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No local shards matched pattern '{pattern}' in {shard_dir}")

    global_idx = 0
    while True:
        order = np.arange(len(files))
        rng.shuffle(order)
        for file_idx in order:
            arr = np.load(files[file_idx], mmap_mode="r")
            if arr.ndim == 1:
                usable = (arr.shape[0] // seq_len) * seq_len
                arr2 = np.asarray(arr[:usable], dtype=np.int32).reshape(-1, seq_len)
            elif arr.ndim == 2 and arr.shape[1] == seq_len:
                arr2 = np.asarray(arr, dtype=np.int32)
            else:
                flat = np.asarray(arr, dtype=np.int32).reshape(-1)
                usable = (flat.shape[0] // seq_len) * seq_len
                arr2 = flat[:usable].reshape(-1, seq_len)

            row_order = np.arange(arr2.shape[0])
            rng.shuffle(row_order)
            for row in row_order:
                if global_idx % process_count == process_index:
                    yield np.asarray(arr2[row], dtype=np.int32)
                global_idx += 1

        if not repeat:
            return


def _iter_hf_sequences(
    data_cfg: dict[str, Any],
    split: str,
    seq_len: int,
    seed: int,
    process_index: int,
    process_count: int,
) -> Iterator[np.ndarray]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("datasets package is required for HF streaming backend") from exc

    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise ImportError("sentencepiece package is required for text tokenization") from exc

    tokenizer_model = data_cfg["tokenizer_model"]
    sp = spm.SentencePieceProcessor(model_file=tokenizer_model)

    ds = load_dataset(
        path=data_cfg["hf_name"],
        name=data_cfg.get("hf_config"),
        split=split,
        streaming=True,
    )

    buffer_size = int(data_cfg.get("shuffle_buffer", 0))
    if buffer_size > 0:
        ds = ds.shuffle(seed=seed, buffer_size=buffer_size)

    eos_id = int(data_cfg["eos_id"])
    text_key = str(data_cfg["text_key"])

    token_buffer: list[int] = []
    global_idx = 0
    for sample in ds:
        text = sample[text_key]
        ids = sp.encode(text, out_type=int)
        token_buffer.extend(ids)
        token_buffer.append(eos_id)

        while len(token_buffer) >= seq_len:
            seq = np.asarray(token_buffer[:seq_len], dtype=np.int32)
            del token_buffer[:seq_len]
            if global_idx % process_count == process_index:
                yield seq
            global_idx += 1


def _batch_iterator(seq_iter: Iterator[np.ndarray], batch_size: int) -> Iterator[np.ndarray]:
    batch: list[np.ndarray] = []
    for seq in seq_iter:
        batch.append(seq)
        if len(batch) == batch_size:
            yield np.stack(batch, axis=0)
            batch = []


def build_batch_iterator(
    data_cfg: dict[str, Any],
    split: str,
    batch_size: int,
    process_index: int,
    process_count: int,
    seed: int,
) -> Iterator[dict[str, np.ndarray]]:
    seq_len = int(data_cfg["seq_len"])
    use_local = bool(data_cfg.get("use_local_shards", False))

    if use_local:
        shard_dir = Path(str(data_cfg["local_shard_dir"]))
        pattern = str(data_cfg["local_train_pattern"] if split == "train" else data_cfg["local_eval_pattern"])
        repeat = split == "train"
        seq_iter = _iter_local_sequences(
            shard_dir=shard_dir,
            pattern=pattern,
            seq_len=seq_len,
            seed=seed,
            repeat=repeat,
            process_index=process_index,
            process_count=process_count,
        )
    else:
        split_name = str(data_cfg["train_split"] if split == "train" else data_cfg["eval_split"])
        seq_iter = _iter_hf_sequences(
            data_cfg=data_cfg,
            split=split_name,
            seq_len=seq_len,
            seed=seed,
            process_index=process_index,
            process_count=process_count,
        )

    for batch in _batch_iterator(seq_iter, batch_size=batch_size):
        yield {"input_ids": batch}


def shard_batch_for_devices(batch: np.ndarray, local_device_count: int) -> np.ndarray:
    if batch.shape[0] % local_device_count != 0:
        raise ValueError("Local batch size must be divisible by local device count")
    per_device = batch.shape[0] // local_device_count
    return batch.reshape(local_device_count, per_device, batch.shape[1])
