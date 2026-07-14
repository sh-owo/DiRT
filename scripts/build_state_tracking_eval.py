from __future__ import annotations

import argparse
import json
from pathlib import Path

from dirt.train.state_tracking import generate_state_tracking_samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/state_tracking_eval.jsonl")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    samples = generate_state_tracking_samples(n_samples=args.samples, seed=args.seed)

    with out_path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps({"prompt": s.prompt, "answer": s.answer}, ensure_ascii=True) + "\n")

    print(f"wrote {len(samples)} samples to {out_path}")


if __name__ == "__main__":
    main()
