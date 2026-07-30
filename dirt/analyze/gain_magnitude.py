from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def run(
    dirt_loss: np.ndarray,
    base_loss: np.ndarray,
    magnitudes: list[np.ndarray],
    n_layers: int,
    output_dir: Path,
    seed: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    gain = base_loss - dirt_loss
    mag_total = np.sum(magnitudes, axis=0)

    valid = np.isfinite(gain) & np.isfinite(mag_total)
    gain = gain[valid]
    mag_total = mag_total[valid]

    if len(gain) < 10:
        print(f"  [test_2d] insufficient valid tokens: {len(gain)}")
        return {}

    rho, p = spearmanr(gain, mag_total)

    p10 = int(len(gain) * 0.1)
    top_gain_idx = np.argsort(gain)[-p10:]
    bot_gain_idx = np.argsort(gain)[:p10]
    mag_top = mag_total[top_gain_idx]
    mag_bot = mag_total[bot_gain_idx]
    mean_mag_top = float(np.mean(mag_top))
    mean_mag_bot = float(np.mean(mag_bot))
    std_mag_top = float(np.std(mag_top))
    std_mag_bot = float(np.std(mag_bot))
    min_mag_top = float(np.min(mag_top))
    max_mag_top = float(np.max(mag_top))
    min_mag_bot = float(np.min(mag_bot))
    max_mag_bot = float(np.max(mag_bot))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 5))
        s = 0.3 if len(gain) > 10000 else 1.0
        ax.scatter(gain, mag_total, alpha=0.3, s=s, c="darkorange")
        ax.set_xlabel("Gain (base_loss - dirt_loss)")
        ax.set_ylabel("Σ_l |magnitude_l|")
        ax.set_title(f"Gain vs Total Magnitude (ρ={rho:.4f}, p={p:.2e}, seed {seed})")
        ax.grid(True, alpha=0.3)

        xlim = ax.get_xlim()
        ax.axvline(0, color="gray", linestyle="--", alpha=0.5)

        plt.tight_layout()
        fig.savefig(output_dir / "2d_gain_vs_magnitude.png", dpi=150)
        plt.close(fig)

        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.bar(["top 10% gain", "bottom 10% gain"], [mean_mag_top, mean_mag_bot],
                color=["green", "red"], alpha=0.7)
        ax2.set_ylabel("Mean Σ|magnitude|")
        ax2.set_title("Magnitude by Gain Extremes")
        ax2.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        fig2.savefig(output_dir / "2d_gain_extremes.png", dpi=150)
        plt.close(fig2)

    except ImportError:
        print("  [test_2d] matplotlib not available, skipping plots")

    print(f"\n=== gain_magnitude (seed {seed}) ===")
    print(f"  spearman ρ={rho:.4f}  p={p:.2e}")
    print(f"  top 10% gain: mean_mag={mean_mag_top:.6f}")
    print(f"  top 10% gain: mean_mag={mean_mag_top:.4f} std={std_mag_top:.4f} min={min_mag_top:.4f} max={max_mag_top:.4f}")
    print(f"  bottom 10% gain: mean_mag={mean_mag_bot:.4f} std={std_mag_bot:.4f} min={min_mag_bot:.4f} max={max_mag_bot:.4f}")
    print(f"  → {'Positive correlation ✓' if rho > 0 and p < 0.05 else 'Weak/no correlation'}")

    return {
        "spearman_rho": float(rho),
        "spearman_p": float(p),
        "mean_mag_top_gain": mean_mag_top,
        "std_mag_top_gain": std_mag_top,
        "min_mag_top_gain": min_mag_top,
        "max_mag_top_gain": max_mag_top,
        "mean_mag_bot_gain": mean_mag_bot,
        "std_mag_bot_gain": std_mag_bot,
        "min_mag_bot_gain": min_mag_bot,
        "max_mag_bot_gain": max_mag_bot,
        "n_valid_tokens": int(len(gain)),
    }
