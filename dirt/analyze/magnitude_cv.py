from __future__ import annotations

from pathlib import Path

import numpy as np


def run(
    magnitudes: list[np.ndarray],
    n_layers: int,
    output_dir: Path,
    seed: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    cv_values = []
    for L in range(n_layers):
        mag = magnitudes[L]
        mean_mag = float(np.mean(mag))
        std_mag = float(np.std(mag))
        cv = std_mag / mean_mag if mean_mag > 1e-12 else 0.0
        cv_values.append(cv)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(n_layers), cv_values, color="purple", alpha=0.7)
        ax.set_xlabel("Layer")
        ax.set_ylabel("CV (|magnitude|)")
        ax.set_title(f"Token-wise |magnitude| Coefficient of Variation (seed {seed})")
        ax.set_xticks(range(n_layers))
        ax.axhline(np.mean(cv_values), color="red", linestyle="--", alpha=0.5, label=f"mean CV={np.mean(cv_values):.3f}")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        fig.savefig(output_dir / "2a_cv_per_layer.png", dpi=150)
        plt.close(fig)

    except ImportError:
        print("  [test_2a] matplotlib not available, skipping plots")

    results = {}
    for L in range(n_layers):
        mag = magnitudes[L]
        results[f"cv_layer_{L}"] = cv_values[L]
        results[f"mean_mag_layer_{L}"] = float(np.mean(mag))
        results[f"std_mag_layer_{L}"] = float(np.std(mag))
        results[f"min_mag_layer_{L}"] = float(np.min(mag))
        results[f"max_mag_layer_{L}"] = float(np.max(mag))
        results[f"median_mag_layer_{L}"] = float(np.median(mag))

    print(f"\n=== magnitude_cv (seed {seed}) ===")
    for L in range(n_layers):
        print(f"  L{L}: CV={cv_values[L]:.4f}  mean={float(np.mean(magnitudes[L])):.6f}  std={float(np.std(magnitudes[L])):.6f}")

    return results
