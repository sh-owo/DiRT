from __future__ import annotations

from pathlib import Path

import numpy as np


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
    cos_bins = np.linspace(-1.0, 1.0, 21)

    cos_results = []
    for L in range(n_layers):
        dv = delta_v[L]
        rv = review[L]
        if dv.shape[0] == 0 or rv.shape[0] == 0:
            cos_results.append(np.array([]))
            continue
        dv_norm = dv / (np.linalg.norm(dv, axis=-1, keepdims=True) + 1e-12)
        rv_norm = rv / (np.linalg.norm(rv, axis=-1, keepdims=True) + 1e-12)
        cos = np.sum(dv_norm * rv_norm, axis=-1)
        cos = np.clip(cos, -1.0, 1.0)
        cos_results.append(cos)

    mean_cos = [float(np.mean(c)) if len(c) > 0 else 0.0 for c in cos_results]
    std_cos = [float(np.std(c)) if len(c) > 0 else 0.0 for c in cos_results]
    min_cos = [float(np.min(c)) if len(c) > 0 else 0.0 for c in cos_results]
    max_cos = [float(np.max(c)) if len(c) > 0 else 0.0 for c in cos_results]
    median_cos = [float(np.median(c)) if len(c) > 0 else 0.0 for c in cos_results]
    neg_frac = [float(np.mean(c < 0)) if len(c) > 0 else 0.0 for c in cos_results]
    pos_frac = [float(np.mean(c > 0)) if len(c) > 0 else 0.0 for c in cos_results]

    base_cos = None
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
        base_std_cos = [float(np.std(c)) if len(c) > 0 else 0.0 for c in base_cos_list]
        base_min_cos = [float(np.min(c)) if len(c) > 0 else 0.0 for c in base_cos_list]
        base_max_cos = [float(np.max(c)) if len(c) > 0 else 0.0 for c in base_cos_list]
        base_median_cos = [float(np.median(c)) if len(c) > 0 else 0.0 for c in base_cos_list]
        base_neg_frac = [float(np.mean(c < 0)) if len(c) > 0 else 0.0 for c in base_cos_list]
        base_pos_frac = [float(np.mean(c > 0)) if len(c) > 0 else 0.0 for c in base_cos_list]

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def _histogram(ax, cos_vals_list, labels, colors, title):
            bin_centers = (cos_bins[:-1] + cos_bins[1:]) / 2
            for cos_vals, label, color in zip(cos_vals_list, labels, colors):
                counts, _ = np.histogram(cos_vals, bins=cos_bins)
                frac = counts / max(len(cos_vals), 1)
                ax.bar(bin_centers, frac, width=0.09, alpha=0.7, label=label, color=color)
            ax.set_xlabel("cosine")
            ax.set_ylabel("fraction")
            ax.set_title(title)
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, alpha=0.3)

        # dirt violin
        fig, axes = plt.subplots(2, (n_layers + 1) // 2, figsize=(4 * min(3, n_layers), 6))
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
        fig.suptitle(f"Propose-Review Direction Cosine (seed {seed})", fontsize=14)
        fig.tight_layout()
        fig.savefig(output_dir / "2c_cosine_violin.png", dpi=150)
        plt.close(fig)

        # base violin
        if base_cos is not None:
            n_base_plots = len(base_cos)
            fig_b, axes_b = plt.subplots(3, (n_base_plots + 2) // 3, figsize=(12, 8))
            axes_b = axes_b.flatten()
            for L in range(n_base_plots):
                ax = axes_b[L]
                c = base_cos[L]
                if len(c) == 0:
                    ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
                    continue
                parts = ax.violinplot(c, positions=[0], showmeans=True, showmedians=True)
                ax.set_title(f"Base L{L + 1} (mean={base_mean_cos[L]:.3f})")
                ax.set_ylabel("cos(prev_dir, curr_dir)")
                ax.set_xticks([])
                ax.axhline(0, color="red", linestyle="--", alpha=0.4)
                ax.grid(True, axis="y", alpha=0.3)
            for i in range(n_base_plots, len(axes_b)):
                axes_b[i].set_visible(False)
            fig_b.suptitle(f"Base Layer-to-Layer Direction Cosine (seed {seed})", fontsize=14)
            fig_b.tight_layout()
            fig_b.savefig(output_dir / "2c_base_cosine_violin.png", dpi=150)
            plt.close(fig_b)

        # dirt combined reinforce histogram
        dirt_colors = plt.cm.Blues(np.linspace(0.4, 0.9, n_layers))
        fig_dc, ax_dc = plt.subplots(figsize=(8, 5))
        _histogram(ax_dc, cos_results,
                   [f"L{L}" for L in range(n_layers)],
                   dirt_colors,
                   f"DiRT Cosine Distribution by Layer (seed {seed})")
        fig_dc.tight_layout()
        fig_dc.savefig(output_dir / "2c_dirt_reinforce_vs_correct.png", dpi=150)
        plt.close(fig_dc)

        # dirt per-layer histograms
        for L in range(n_layers):
            fig_dl, ax_dl = plt.subplots(figsize=(6, 4))
            _histogram(ax_dl, [cos_results[L]], [f"L{L}"], ["#2196F3"],
                       f"DiRT L{L} Cosine Distribution (seed {seed})")
            fig_dl.tight_layout()
            fig_dl.savefig(output_dir / f"2c_dirt_L{L}_reinforce_vs_correct.png", dpi=150)
            plt.close(fig_dl)

        # base combined reinforce histogram
        if base_cos is not None:
            base_colors = plt.cm.Oranges(np.linspace(0.4, 0.9, len(base_cos)))
            fig_bc, ax_bc = plt.subplots(figsize=(8, 5))
            _histogram(ax_bc, base_cos,
                       [f"L{L + 1}" for L in range(len(base_cos))],
                       base_colors,
                       f"Base Cosine Distribution by Layer (seed {seed})")
            fig_bc.tight_layout()
            fig_bc.savefig(output_dir / "2c_base_reinforce_vs_correct.png", dpi=150)
            plt.close(fig_bc)

            # base per-layer histograms
            for L in range(len(base_cos)):
                fig_bl, ax_bl = plt.subplots(figsize=(6, 4))
                _histogram(ax_bl, [base_cos[L]], [f"L{L + 1}"], ["#FF5722"],
                           f"Base L{L + 1} Cosine Distribution (seed {seed})")
                fig_bl.tight_layout()
                fig_bl.savefig(output_dir / f"2c_base_L{L + 1}_reinforce_vs_correct.png", dpi=150)
                plt.close(fig_bl)

        # reinforce vs correct bar chart (existing)
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        x = np.arange(n_layers)
        w = 0.2
        ax2.bar(x - w * 1.5, pos_frac, w, label="DiRT cos>0 (reinforce)", color="green", alpha=0.8)
        ax2.bar(x - w * 0.5, neg_frac, w, label="DiRT cos<0 (correct)", color="red", alpha=0.8)
        if base_cos is not None:
            base_x = np.arange(1, n_layers_base)
            ax2.bar(base_x + w * 0.5, base_pos_frac, w, label="Base same-dir", color="darkorange", alpha=0.6)
            ax2.bar(base_x + w * 1.5, base_neg_frac, w, label="Base turn", color="brown", alpha=0.6)
        ax2.set_xlabel("Layer")
        ax2.set_ylabel("Fraction")
        ax2.set_title(f"Reinforce vs Correct per Layer (seed {seed})")
        ax2.set_xticks(np.arange(max(n_layers, n_layers_base)))
        ax2.legend(fontsize=8)
        ax2.grid(True, axis="y", alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(output_dir / "2c_reinforce_vs_correct.png", dpi=150)
        plt.close(fig2)

    except ImportError:
        print("  [test_2c] matplotlib not available, skipping plots")

    results = {}
    for L in range(n_layers):
        c = cos_results[L]
        results[f"cos_mean_L{L}"] = mean_cos[L]
        results[f"cos_std_L{L}"] = std_cos[L]
        results[f"cos_min_L{L}"] = min_cos[L]
        results[f"cos_max_L{L}"] = max_cos[L]
        results[f"cos_median_L{L}"] = median_cos[L]
        results[f"neg_frac_L{L}"] = neg_frac[L]
        results[f"pos_frac_L{L}"] = pos_frac[L]
        if len(c) > 0:
            counts, _ = np.histogram(c, bins=cos_bins)
            for bi in range(len(cos_bins) - 1):
                results[f"hist_L{L}_bin{bi}"] = int(counts[bi])

    if base_cos is not None:
        for L in range(1, n_layers_base):
            idx = L - 1
            c = base_cos[idx]
            results[f"base_cos_mean_L{L}"] = base_mean_cos[idx]
            results[f"base_cos_std_L{L}"] = base_std_cos[idx]
            results[f"base_cos_min_L{L}"] = base_min_cos[idx]
            results[f"base_cos_max_L{L}"] = base_max_cos[idx]
            results[f"base_cos_median_L{L}"] = base_median_cos[idx]
            results[f"base_neg_frac_L{L}"] = base_neg_frac[idx]
            results[f"base_pos_frac_L{L}"] = base_pos_frac[idx]
            if len(c) > 0:
                counts, _ = np.histogram(c, bins=cos_bins)
                for bi in range(len(cos_bins) - 1):
                    results[f"base_hist_L{L}_bin{bi}"] = int(counts[bi])

    print(f"\n=== propose_review_cosine (seed {seed}) ===")
    for L in range(n_layers):
        label = "REINFORCE" if mean_cos[L] > 0 else "CORRECT"
        print(f"  L{L}: mean_cos={mean_cos[L]:+.4f}  std={std_cos[L]:.4f}  "
              f"neg={neg_frac[L]:.2%}  pos={pos_frac[L]:.2%}  → {label}")
    if base_cos is not None:
        for L in range(1, n_layers_base):
            idx = L - 1
            label = "STRAIGHT" if base_mean_cos[idx] > 0 else "TURN"
            print(f"  Base L{L}: mean_cos={base_mean_cos[idx]:+.4f}  "
                  f"neg={base_neg_frac[idx]:.2%}  pos={base_pos_frac[idx]:.2%}  → {label}")

    return results
