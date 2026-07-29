from __future__ import annotations

from pathlib import Path

import numpy as np


def run(
    dirt_val_loss: float,
    base_val_loss: float,
    dirt_n_params: int,
    base_n_params: int,
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    gpt6l_loss = None

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = ["DiRT 6L", "GPT 12L"]
        losses = [dirt_val_loss, base_val_loss]
        colors = ["#2196F3", "#FF5722"]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(labels, losses, color=colors, alpha=0.8, width=0.5)
        for bar, val in zip(bars, losses):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=11)
        ax.set_ylabel("Validation NLL")
        ax.set_title("DiRT 6L vs GPT 12L")
        ax.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        fig.savefig(output_dir / "1f_comparison.png", dpi=150)
        plt.close(fig)

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        metrics_labels = ["Val NLL", "Params (M)"]
        dirt_vals = [dirt_val_loss, dirt_n_params / 1e6]
        base_vals = [base_val_loss, base_n_params / 1e6]
        x = np.arange(len(metrics_labels))
        w = 0.35
        ax2.bar(x - w / 2, dirt_vals, w, label="DiRT 6L", color="#2196F3", alpha=0.8)
        ax2.bar(x + w / 2, base_vals, w, label="GPT 12L", color="#FF5722", alpha=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics_labels)
        ax2.legend()
        ax2.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        fig2.savefig(output_dir / "1f_metrics_comparison.png", dpi=150)
        plt.close(fig2)

    except ImportError:
        print("  [test_1f] matplotlib not available, skipping plots")

    better = "DiRT 6L" if dirt_val_loss < base_val_loss else "GPT 12L" if base_val_loss < dirt_val_loss else "tie"

    print(f"\n=== layer_comparison ===")
    print(f"  DiRT 6L  val_loss={dirt_val_loss:.6f}  params={dirt_n_params:,}")
    print(f"  GPT 12L  val_loss={base_val_loss:.6f}  params={base_n_params:,}")
    print(f"  Better: {better}")

    return {
        "dirt_val_loss": dirt_val_loss,
        "base_val_loss": base_val_loss,
        "dirt_n_params": dirt_n_params,
        "base_n_params": base_n_params,
        "better": better,
    }
