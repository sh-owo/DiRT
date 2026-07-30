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
    pos_std_dirt = np.zeros(T)
    pos_min_dirt = np.zeros(T)
    pos_max_dirt = np.zeros(T)
    pos_ppl_base = np.zeros(T)
    pos_std_base = np.zeros(T)
    pos_min_base = np.zeros(T)
    pos_max_base = np.zeros(T)
    pos_count = np.zeros(T)
    for t in range(T):
        mask = pos_ids[:, 2] == t
        c = int(mask.sum())
        pos_count[t] = c
        if c > 0:
            vals_dirt = dirt_loss[mask]
            vals_base = base_loss[mask]
            pos_ppl_dirt[t] = float(np.mean(vals_dirt))
            pos_std_dirt[t] = float(np.std(vals_dirt))
            pos_min_dirt[t] = float(np.min(vals_dirt))
            pos_max_dirt[t] = float(np.max(vals_dirt))
            pos_ppl_base[t] = float(np.mean(vals_base))
            pos_std_base[t] = float(np.std(vals_base))
            pos_min_base[t] = float(np.min(vals_base))
            pos_max_base[t] = float(np.max(vals_base))

    decile_edges = np.percentile(base_loss, np.linspace(0, 100, 11))
    decile_mean_loss = []
    decile_mean_delta = []
    decile_std_delta = []
    decile_min_delta = []
    decile_max_delta = []
    for i in range(10):
        lo, hi = decile_edges[i], decile_edges[i + 1]
        if i == 9:
            mask = (base_loss >= lo) & (base_loss <= hi)
        else:
            mask = (base_loss >= lo) & (base_loss < hi)
        c = int(mask.sum())
        if c > 0:
            decile_mean_loss.append(float(np.mean(base_loss[mask])))
            delta = dirt_loss[mask] - base_loss[mask]
            decile_mean_delta.append(float(np.mean(delta)))
            decile_std_delta.append(float(np.std(delta)))
            decile_min_delta.append(float(np.min(delta)))
            decile_max_delta.append(float(np.max(delta)))
        else:
            decile_mean_loss.append(0.0)
            decile_mean_delta.append(0.0)
            decile_std_delta.append(0.0)
            decile_min_delta.append(0.0)
            decile_max_delta.append(0.0)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig1, ax1 = plt.subplots(figsize=(8, 5))
        valid = pos_count > 0
        ax1.plot(np.arange(T)[valid], pos_ppl_base[valid], label="Base", alpha=0.8)
        ax1.plot(np.arange(T)[valid], pos_ppl_dirt[valid], label="DiRT", alpha=0.8)
        ax1.set_xlabel("Position in sequence")
        ax1.set_ylabel("Mean NLL")
        ax1.set_title(f"Per-position NLL (seed {seed})")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        fig1.tight_layout()
        fig1.savefig(output_dir / "1b_pos_ppl.png", dpi=150)
        plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        x_pos = np.arange(10)
        colors = ["green" if d < 0 else "red" for d in decile_mean_delta]
        ax2.bar(x_pos, decile_mean_delta, color=colors, alpha=0.7)
        ax2.errorbar(x_pos, decile_mean_delta, yerr=decile_std_delta,
                     fmt="none", capsize=3, color="black", alpha=0.5)
        ax2.axhline(0, color="black", linewidth=0.5)
        ax2.set_xlabel("Base NLL decile")
        ax2.set_ylabel("DiRT NLL - Base NLL")
        ax2.set_title("ΔNLL by difficulty decile")
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f"{i*10}-{(i+1)*10}%" for i in range(10)], rotation=45, fontsize=8)
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(output_dir / "1b_delta_nll_decile.png", dpi=150)
        plt.close(fig2)

    except ImportError:
        print("  [test_1b] matplotlib not available, skipping plots")

    results = {}
    for i in range(10):
        results[f"decile_{i}_delta_mean"] = decile_mean_delta[i]
        results[f"decile_{i}_delta_std"] = decile_std_delta[i]
        results[f"decile_{i}_delta_min"] = decile_min_delta[i]
        results[f"decile_{i}_delta_max"] = decile_max_delta[i]
        results[f"decile_{i}_base_mean"] = decile_mean_loss[i]

    print(f"\n=== position_ppl (seed {seed}) ===")
    for i in range(10):
        print(f"  decile {i*10}-{(i+1)*10}%: Δ={decile_mean_delta[i]:+.4f}±{decile_std_delta[i]:.4f}  base_mean={decile_mean_loss[i]:.4f}")

    return results
