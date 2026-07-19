from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def mmlu_pro(predict_fn, tokenizer_model: str | Path, output_dir: Path | None = None):
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets is required for MMLU-Pro evaluation")

    try:
        import sentencepiece as spm
    except ImportError:
        raise ImportError("sentencepiece is required for MMLU-Pro evaluation")

    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_model))

    if is_main := jax.process_index() == 0:
        print("Loading MMLU-Pro dataset...")
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test", streaming=True)

    correct = 0
    total = 0

    for sample in ds:
        question = sample["question"]
        options = sample["options"]
        answer_idx = sample["answer"]

        scores = []
        for opt in options:
            text = question + " " + opt
            tokens = tokenizer.encode(text, out_type=int)
            input_ids = jnp.array([tokens], dtype=jnp.int32)

            logits = predict_fn(input_ids)
            logits = logits.astype(jnp.float32)
            log_probs = jax.nn.log_softmax(logits, axis=-1)

            opt_start = len(tokenizer.encode(question, out_type=int)) + 1
            opt_end = len(tokens)
            token_log_probs = []
            for pos in range(opt_start, opt_end):
                token_log_probs.append(float(log_probs[pos - 1, tokens[pos]]))

            score = float(np.mean(token_log_probs)) if token_log_probs else -float("inf")
            scores.append(score)

        if int(np.argmax(scores)) == int(answer_idx):
            correct += 1
        total += 1

        if is_main and total % 100 == 0:
            print(f"  [{total}] Acc: {correct/total:.4f} ({correct}/{total})")

    accuracy = correct / total
    if is_main:
        print(f"\nMMLU-Pro Results:")
        print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "mmlu_pro_results.txt", "w") as f:
            f.write(f"Accuracy: {accuracy:.4f} ({correct}/{total})\n")

    return {"accuracy": accuracy, "correct": correct, "total": total}
