from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def run(
    magnitudes: list[np.ndarray],
    dirt_loss: np.ndarray,
    pos_ids: np.ndarray,
    token_ids: dict[int, np.ndarray],
    n_layers: int,
    output_dir: Path,
    seed: int,
    tokenizer=None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for L in range(n_layers):
        mag = magnitudes[L]
        loss = dirt_loss
        valid = mag > -1e10
        if np.sum(valid) < 10:
            continue
        rho, p = spearmanr(mag[valid], loss[valid])
        results[f"spearman_L{L}"] = float(rho)
        results[f"spearman_p_L{L}"] = float(p)

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 5))
            s = 0.3 if len(mag) > 10000 else 1.0
            ax.scatter(loss[valid], mag[valid], alpha=0.3, s=s, c="steelblue")
            ax.set_xlabel("Token NLL")
            ax.set_ylabel("|magnitude|")
            ax.set_title(f"L{L}: magnitude vs NLL (ρ={rho:.4f}, seed {seed})")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            fig.savefig(output_dir / f"2b_magnitude_vs_loss_L{L}.png", dpi=150)
            plt.close(fig)

        except ImportError:
            pass

        n_top = 20
        top_idx = np.argsort(mag)[-n_top:]
        bot_idx = np.argsort(mag)[:n_top]

        ctx_path = output_dir / f"2b_top_bottom_L{L}_seed{seed}.txt"
        with open(ctx_path, "w") as f:
            f.write(f"=== Layer {L} |magnitude| 상위 {n_top} ===\n")
            for idx in reversed(top_idx):
                b_id, b, t = pos_ids[idx]
                ids = token_ids.get(b_id, np.array([[0]]))
                if b < ids.shape[0] and t < ids.shape[1]:
                    tok_id = int(ids[b, t])
                    tok_str = tokenizer.decode([tok_id]) if tokenizer else f"[id={tok_id}]"
                    f.write(f"  mag={mag[idx]:.6f} | loss={loss[idx]:.4f} | pos={b_id}:{b}:{t} → {tok_str}\n")

            f.write(f"\n=== Layer {L} |magnitude| 하위 {n_top} ===\n")
            for idx in bot_idx:
                b_id, b, t = pos_ids[idx]
                ids = token_ids.get(b_id, np.array([[0]]))
                if b < ids.shape[0] and t < ids.shape[1]:
                    tok_id = int(ids[b, t])
                    tok_str = tokenizer.decode([tok_id]) if tokenizer else f"[id={tok_id}]"
                    f.write(f"  mag={mag[idx]:.6f} | loss={loss[idx]:.4f} | pos={b_id}:{b}:{t} → {tok_str}\n")

    print(f"\n=== magnitude_difficulty (seed {seed}) ===")
    for L in range(n_layers):
        k = f"spearman_L{L}"
        if k in results:
            print(f"  L{L}: ρ={results[k]:.4f} (p={results.get(f'spearman_p_L{L}', 1):.4e})")
    print(f"  상하위 토큰: {output_dir / f'2b_top_bottom_L*_seed{seed}.txt'}")

    return results
