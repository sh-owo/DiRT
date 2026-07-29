from __future__ import annotations

from pathlib import Path

import numpy as np


def run(
    dirt_loss: np.ndarray,
    base_loss: np.ndarray,
    pos_ids: np.ndarray,
    output_dir: Path,
    seed: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    T = int(pos_ids[:, 2].max()) + 1 if len(pos_ids) > 0 else 2048

    pos_ppl_dirt = np.zeros(T)
    pos_ppl_base = np.zeros(T)
    pos_count = np.zeros(T)
    for t in range(T):
        mask = pos_ids[:, 2] == t
        c = int(mask.sum())
        pos_count[t] = c
        if c > 0:
            pos_ppl_dirt[t] = float(np.mean(dirt_loss[mask]))
            pos_ppl_base[t] = float(np.mean(base_loss[mask]))

    decile_edges = np.percentile(base_loss, np.linspace(0, 100, 11))
    decile_labels = [f"d{i}" for i in range(10)]
    decile_mean_loss = []
    decile_mean_delta = []
    for i in range(10):
        lo, hi = decile_edges[i], decile_edges[i + 1]
        if i == 9:
            mask = (base_loss >= lo) & (base_loss <= hi)
        else:
            mask = (base_loss >= lo) & (base_loss < hi)
        c = int(mask.sum())
        if c > 0:
            decile_mean_loss.append(float(np.mean(base_loss[mask])))
            decile_mean_delta.append(float(np.mean(dirt_loss[mask] - base_loss[mask])))
        else:
            decile_mean_loss.append(0.0)
            decile_mean_delta.append(0.0)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        valid = pos_count > 0
        ax1.plot(np.arange(T)[valid], pos_ppl_base[valid], label="Base", alpha=0.8)
        ax1.plot(np.arange(T)[valid], pos_ppl_dirt[valid], label="DiRT", alpha=0.8)
        ax1.set_xlabel("Position in sequence")
        ax1.set_ylabel("Mean NLL")
        ax1.set_title(f"Per-position NLL (seed {seed})")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        x_pos = np.arange(10)
        colors = ["green" if d < 0 else "red" for d in decile_mean_delta]
        ax2.bar(x_pos, decile_mean_delta, color=colors, alpha=0.7)
        ax2.axhline(0, color="black", linewidth=0.5)
        ax2.set_xlabel("Base NLL decile")
        ax2.set_ylabel("DiRT NLL - Base NLL")
        ax2.set_title("ΔNLL by difficulty decile")
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f"{i*10}-{(i+1)*10}%" for i in range(10)], rotation=45, fontsize=8)

        plt.tight_layout()
        fig.savefig(output_dir / "1b_pos_ppl.png", dpi=150)
        plt.close(fig)

    except ImportError:
        print("  [test_1b] matplotlib not available, skipping plots")

    results = {}
    for i in range(10):
        results[f"decile_{i}_delta"] = decile_mean_delta[i]
        results[f"decile_{i}_base_mean"] = decile_mean_loss[i]

    print(f"\n=== position_ppl (seed {seed}) ===")
    for i in range(10):
        print(f"  decile {i*10}-{(i+1)*10}%: Δ={decile_mean_delta[i]:+.6f}  base_mean={decile_mean_loss[i]:.6f}")

    return results
