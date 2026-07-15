from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _load_tokenizer(path: str):
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise ImportError("sentencepiece is required") from exc
    return spm.SentencePieceProcessor(model_file=path)


def _load_stream(name: str, config: str, split: str, shuffle_buffer: int, seed: int):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("datasets is required") from exc

    ds = load_dataset(path=name, name=config, split=split, streaming=True)
    if shuffle_buffer > 0:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
    return ds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-name", type=str, default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--hf-config", type=str, default="sample-10BT")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--text-key", type=str, default="text")
    parser.add_argument("--tokenizer-model", type=str, required=True)
    parser.add_argument("--eos-id", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--tokens-target", type=int, default=500000000)
    parser.add_argument("--shard-seqs", type=int, default=8192)
    parser.add_argument("--output-dir", type=str, default="data/tokenized")
    parser.add_argument("--prefix", type=str, default="train")
    parser.add_argument("--shuffle-buffer", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    tokenizer = _load_tokenizer(args.tokenizer_model)
    stream = _load_stream(args.hf_name, args.hf_config, args.split, args.shuffle_buffer, args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    buffer: list[int] = []
    shard: list[np.ndarray] = []
    total_tokens = 0
    shard_idx = 0

    for sample in stream:
        text = sample[args.text_key]
        ids = tokenizer.encode(text, out_type=int)
        ids.append(args.eos_id)
        buffer.extend(ids)

        while len(buffer) >= args.seq_len and total_tokens < args.tokens_target:
            seq = np.asarray(buffer[: args.seq_len], dtype=np.int32)
            del buffer[: args.seq_len]
            shard.append(seq)
            total_tokens += int(args.seq_len)

            if len(shard) >= args.shard_seqs:
                arr = np.stack(shard, axis=0)
                np.save(out_dir / f"{args.prefix}-{shard_idx:05d}.npy", arr)
                shard = []
                shard_idx += 1

        if total_tokens >= args.tokens_target:
            break

    if shard:
        arr = np.stack(shard, axis=0)
        np.save(out_dir / f"{args.prefix}-{shard_idx:05d}.npy", arr)

    print(f"saved {total_tokens} tokens to {out_dir}")


if __name__ == "__main__":
    main()
