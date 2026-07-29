from __future__ import annotations

from pathlib import Path

import numpy as np


def _traj_len(traj):
    return float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=-1)))


def run(
    hidden_dirt: list[np.ndarray],
    hidden_base: list[np.ndarray],
    n_layers_dirt: int,
    n_layers_base: int,
    output_dir: Path,
    seed: int,
    delta_v: list[np.ndarray] | None = None,
    review: list[np.ndarray] | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    n_subsample = hidden_dirt[0].shape[0] if len(hidden_dirt) > 0 and hidden_dirt[0].shape[0] > 0 else 0
    if n_subsample == 0:
        print("  [trajectory_pca] no subsampled data, skipping")
        return {}

    B, D = hidden_dirt[0].shape
    n_pc = 5

    dirt_points = [hidden_dirt[0]]
    for L in range(n_layers_dirt):
        z_L = hidden_dirt[L]
        new_L = z_L + delta_v[L]
        dirt_points.append(new_L)
        dirt_points.append(hidden_dirt[L + 1])
    n_dirt_pts = len(dirt_points)

    base_points = list(hidden_base)
    n_base_pts = len(base_points)

    all_stacked = np.concatenate(
        [np.stack(base_points, axis=1).reshape(-1, D),
         np.stack(dirt_points, axis=1).reshape(-1, D)],
        axis=0,
    )

    mean = np.mean(all_stacked, axis=0, keepdims=True)
    centered = all_stacked - mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    pca_5d = centered @ Vt[:n_pc].T

    base_flat = pca_5d[:n_base_pts * B].reshape(B, n_base_pts, n_pc)
    dirt_flat = pca_5d[n_base_pts * B:].reshape(B, n_dirt_pts, n_pc)

    mean_base = np.mean(base_flat, axis=0)
    mean_dirt = np.mean(dirt_flat, axis=0)

    traj_len_base = _traj_len(mean_base)
    traj_len_dirt = _traj_len(mean_dirt)

    results = {
        "traj_len_base": traj_len_base,
        "traj_len_dirt": traj_len_dirt,
        "n_subsample_tokens": n_subsample,
    }

    dv_norms = [float(np.mean(np.linalg.norm(delta_v[L], axis=-1))) for L in range(n_layers_dirt)]
    rv_norms = [float(np.mean(np.linalg.norm(review[L], axis=-1))) for L in range(n_layers_dirt)]
    for L in range(n_layers_dirt):
        results[f"propose_norm_L{L}"] = dv_norms[L]
        results[f"review_norm_L{L}"] = rv_norms[L]

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("  [trajectory_pca] matplotlib not available, skipping plots")
        return results

    base_color = "#A0A0A0"
    base_label = "Base 12L"
    dirt_z_color = "#2196F3"
    dirt_new_color = "#E67E22"
    dirt_review_color = "#2196F3"
    start_color = "#2ECC71"
    end_color = "#E74C3C"

    def draw_traj(ax, data_base, data_dirt, x_i, y_i, show_legend: bool):
        ax.plot(data_base[:, x_i], data_base[:, y_i], "o-",
                color=base_color, label=base_label, alpha=0.7, markersize=2.5, linewidth=1)
        for i in range(len(data_base)):
            ax.annotate(str(i), (data_base[i, x_i], data_base[i, y_i]),
                       fontsize=5.5, color=base_color, alpha=0.6)

        pt = 0
        ax.scatter(data_dirt[pt, x_i], data_dirt[pt, y_i],
                  c=[start_color], s=60, marker="*", zorder=5, label="start")
        for L in range(n_layers_dirt):
            z_L = data_dirt[pt]
            new_L = data_dirt[pt + 1]
            z_next = data_dirt[pt + 2]
            ax.annotate(str(L), (z_L[x_i], z_L[y_i]),
                       fontsize=6.5, color=dirt_z_color, fontweight="bold")
            ax.plot([z_L[x_i], new_L[x_i]], [z_L[y_i], new_L[y_i]],
                    "--", color=dirt_new_color, alpha=0.9, linewidth=1.5,
                    label="propose" if show_legend and L == 0 else "")
            ax.plot([new_L[x_i], z_next[x_i]], [new_L[y_i], z_next[y_i]],
                    "-", color=dirt_review_color, alpha=0.9, linewidth=2,
                    label="review" if show_legend and L == 0 else "")
            pt += 3

        ax.scatter(data_dirt[-1, x_i], data_dirt[-1, y_i],
                  c=[end_color], s=60, marker="s", zorder=5, label="end")
        ax.set_xlabel(f"PC{x_i + 1}")
        ax.set_ylabel(f"PC{y_i + 1}")
        ax.grid(True, alpha=0.3)
        if show_legend:
            ax.legend(fontsize=7, loc="best")

    title = f"Hidden State Trajectory (PCA, seed {seed})"

    pc_pairs = [(i, j) for i in range(n_pc) for j in range(i + 1, n_pc)]
    n_pairs = len(pc_pairs)
    n_cols = 2
    n_rows = (n_pairs + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols + 1, 3 * n_rows + 1))
    axes = axes.flatten()
    for idx, (i, j) in enumerate(pc_pairs):
        draw_traj(axes[idx], mean_base, mean_dirt, i, j, show_legend=(idx == 0))
        axes[idx].set_title(f"PC{i + 1}-PC{j + 1}", fontsize=9)
    for idx in range(n_pairs, len(axes)):
        axes[idx].set_visible(False)
    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    fig.savefig(output_dir / "trajectory_pca_2d_grid.png", dpi=150)
    plt.close(fig)

    fig2 = plt.figure(figsize=(10, 8))
    ax3d = fig2.add_subplot(111, projection="3d")
    ax3d.plot(mean_base[:, 0], mean_base[:, 1], mean_base[:, 2],
              "o-", color=base_color, label=base_label, alpha=0.7, markersize=3, linewidth=1)
    for i in range(len(mean_base)):
        ax3d.text(mean_base[i, 0], mean_base[i, 1], mean_base[i, 2],
                str(i), fontsize=6, color=base_color, alpha=0.6)

    pt = 0
    ax3d.scatter(*mean_dirt[pt], c=start_color, s=120, marker="*", zorder=5, label="start")
    for L in range(n_layers_dirt):
        z_L = mean_dirt[pt]
        new_L = mean_dirt[pt + 1]
        z_next = mean_dirt[pt + 2]
        ax3d.text(*z_L, str(L), fontsize=8, color=dirt_z_color, fontweight="bold")
        ax3d.plot(*np.column_stack([z_L, new_L]), "--",
                color=dirt_new_color, alpha=0.9, linewidth=2, label="propose" if L == 0 else "")
        ax3d.plot(*np.column_stack([new_L, z_next]), "-",
                color=dirt_review_color, alpha=0.9, linewidth=3, label="review" if L == 0 else "")
        pt += 3

    ax3d.scatter(*mean_dirt[-1], c=end_color, s=120, marker="s", zorder=5, label="end")
    ax3d.set_xlabel("PC1")
    ax3d.set_ylabel("PC2")
    ax3d.set_zlabel("PC3")
    ax3d.set_title(title)
    ax3d.legend(fontsize=8, loc="best")
    ax3d.grid(True, alpha=0.3)

    for angle in [30, 120]:
        ax3d.view_init(elev=20, azim=angle)
        angle_str = f"elev20_azim{angle}"
        fig2.savefig(output_dir / f"trajectory_pca_3d_{angle_str}.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    fig3, ax_bar = plt.subplots(figsize=(8, 4))
    x = np.arange(n_layers_dirt)
    w = 0.35
    ax_bar.bar(x - w / 2, dv_norms, w, label="propose (||delta_v||)", color=dirt_new_color, alpha=0.8)
    ax_bar.bar(x + w / 2, rv_norms, w, label="review (||review||)", color=dirt_review_color, alpha=0.8)
    ax_bar.set_xlabel("Layer")
    ax_bar.set_ylabel("Mean norm")
    ax_bar.set_title("Propose vs Review Step Size per Layer")
    ax_bar.set_xticks(x)
    ax_bar.legend()
    ax_bar.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig3.savefig(output_dir / "trajectory_pca_propose_vs_review.png", dpi=150)
    plt.close(fig3)

    print(f"\n=== trajectory_pca (seed {seed}) ===")
    print(f"  Trajectory length: Base={traj_len_base:.4f}  DiRT={traj_len_dirt:.4f}")
    print(f"  Subsample: {n_subsample} tokens")
    for L in range(n_layers_dirt):
        ratio = rv_norms[L] / dv_norms[L] if dv_norms[L] > 1e-12 else float("inf")
        print(f"  L{L}: ||propose||={dv_norms[L]:.6f}  ||review||={rv_norms[L]:.6f}  ratio={ratio:.2f}")

    return results
