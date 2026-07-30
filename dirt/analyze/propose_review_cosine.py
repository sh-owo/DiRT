from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def run(
    delta_v: list[np.ndarray],
    review: list[np.ndarray],
    direction: list[np.ndarray],
    n_layers: int,
    output_dir: Path,
    seed: int,
    hidden_base: list[np.ndarray] | None = None,
    n_layers_base: int = 0,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    cos_results = []
    for L in range(n_layers):
        dv = delta_v[L]
        rv = review[L]
        if dv.shape[0] == 0 or rv.shape[0] == 0:
            cos_results.append([])
            continue

        dv_norm = dv / (np.linalg.norm(dv, axis=-1, keepdims=True) + 1e-12)
        rv_norm = rv / (np.linalg.norm(rv, axis=-1, keepdims=True) + 1e-12)
        cos = np.sum(dv_norm * rv_norm, axis=-1)
        cos = np.clip(cos, -1.0, 1.0)
        cos_results.append(cos)

    mean_cos = [float(np.mean(c)) if len(c) > 0 else 0.0 for c in cos_results]
    std_cos = [float(np.std(c)) if len(c) > 0 else 0.0 for c in cos_results]
    neg_frac = [float(np.mean(c < 0)) if len(c) > 0 else 0.0 for c in cos_results]
    pos_frac = [float(np.mean(c > 0)) if len(c) > 0 else 0.0 for c in cos_results]

    base_cos = None
    base_mean_cos = None
    if hidden_base is not None and n_layers_base > 1:
        base_cos_list = []
        for L in range(1, n_layers_base):
            prev = hidden_base[L - 1]
            curr = hidden_base[L]
            d_prev = prev - (hidden_base[L - 2] if L >= 2 else np.zeros_like(prev))
            d_curr = curr - prev
            p_norm = d_prev / (np.linalg.norm(d_prev, axis=-1, keepdims=True) + 1e-12)
            c_norm = d_curr / (np.linalg.norm(d_curr, axis=-1, keepdims=True) + 1e-12)
            cos_b = np.sum(p_norm * c_norm, axis=-1)
            cos_b = np.clip(cos_b, -1.0, 1.0)
            base_cos_list.append(cos_b)
        base_cos = base_cos_list
        base_mean_cos = [float(np.mean(c)) if len(c) > 0 else 0.0 for c in base_cos_list]
        base_neg_frac = [float(np.mean(c < 0)) if len(c) > 0 else 0.0 for c in base_cos_list]
        base_pos_frac = [float(np.mean(c > 0)) if len(c) > 0 else 0.0 for c in base_cos_list]

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, max(1, (n_layers + 1) // 2), figsize=(4 * min(3, n_layers), 6))
        axes = axes.flatten() if n_layers > 1 else [axes]
        for L in range(n_layers):
            ax = axes[L]
            c = cos_results[L]
            if len(c) == 0:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
                continue
            parts = ax.violinplot(c, positions=[0], showmeans=True, showmedians=True)
            ax.set_title(f"L{L} (mean={mean_cos[L]:.3f})")
            ax.set_ylabel("cos(propose, review)")
            ax.set_xticks([])
            ax.axhline(0, color="red", linestyle="--", alpha=0.4)
            ax.grid(True, axis="y", alpha=0.3)

        for i in range(len(cos_results), len(axes)):
            axes[i].set_visible(False)

        plt.suptitle(f"Propose-Review Direction Cosine (seed {seed})", fontsize=14)
        plt.tight_layout()
        fig.savefig(output_dir / "2c_cosine_violin.png", dpi=150)
        plt.close(fig)

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        x = np.arange(n_layers)
        w = 0.2
        ax2.bar(x - w * 1.5, pos_frac, w, label="DiRT cos>0 (reinforce)", color="green", alpha=0.8)
        ax2.bar(x - w * 0.5, neg_frac, w, label="DiRT cos<0 (correct)", color="red", alpha=0.8)
        if base_mean_cos is not None:
            base_x = np.arange(1, n_layers_base)
            ax2.bar(base_x + w * 0.5, base_pos_frac, w, label="Base same-dir", color="darkorange", alpha=0.6)
            ax2.bar(base_x + w * 1.5, base_neg_frac, w, label="Base turn", color="brown", alpha=0.6)
        ax2.set_xlabel("Layer")
        ax2.set_ylabel("Fraction")
        ax2.set_title(f"Reinforce vs Correct per Layer (seed {seed})")
        ax2.set_xticks(np.arange(max(n_layers, n_layers_base)))
        ax2.legend(fontsize=8)
        ax2.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        fig2.savefig(output_dir / "2c_reinforce_vs_correct.png", dpi=150)
        plt.close(fig2)

    except ImportError:
        print("  [test_2c] matplotlib not available, skipping plots")

    results = {}
    for L in range(n_layers):
        results[f"cos_mean_L{L}"] = mean_cos[L]
        results[f"cos_std_L{L}"] = std_cos[L]
        results[f"neg_frac_L{L}"] = neg_frac[L]
        results[f"pos_frac_L{L}"] = pos_frac[L]
    if base_mean_cos is not None:
        for L in range(1, n_layers_base):
            idx = L - 1
            results[f"base_cos_mean_L{L}"] = base_mean_cos[idx]
            results[f"base_neg_frac_L{L}"] = base_neg_frac[idx]
            results[f"base_pos_frac_L{L}"] = base_pos_frac[idx]

    print(f"\n=== propose_review_cosine (seed {seed}) ===")
    for L in range(n_layers):
        label = "REINFORCE" if mean_cos[L] > 0 else "CORRECT"
        print(f"  L{L}: mean_cos={mean_cos[L]:+.4f}  std={std_cos[L]:.4f}  "
              f"neg={neg_frac[L]:.2%}  pos={pos_frac[L]:.2%}  → {label}")
    if base_mean_cos is not None:
        for L in range(1, n_layers_base):
            idx = L - 1
            label = "STRAIGHT" if base_mean_cos[idx] > 0 else "TURN"
            print(f"  Base L{L}: mean_cos={base_mean_cos[idx]:+.4f}  "
                  f"neg={base_neg_frac[idx]:.2%}  pos={base_pos_frac[idx]:.2%}  → {label}")

    return results
