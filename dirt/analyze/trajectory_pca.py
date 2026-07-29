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
    sentence_hidden_dirt: list[np.ndarray] | None = None,
    sentence_hidden_base: list[np.ndarray] | None = None,
    sentence_texts: list[str] | None = None,
    tokenizer=None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    n_subsample = hidden_dirt[0].shape[0] if len(hidden_dirt) > 0 and hidden_dirt[0].shape[0] > 0 else 0
    if n_subsample == 0:
        print("  [trajectory_pca] no subsampled data, skipping")
        return {}

    B, D = hidden_dirt[0].shape

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("  [trajectory_pca] matplotlib not available, skipping plots")
        return {}

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
    mean_all = np.mean(all_stacked, axis=0, keepdims=True)
    centered_all = all_stacked - mean_all
    U, S, Vt = np.linalg.svd(centered_all, full_matrices=False)
    pca_5d = centered_all @ Vt[:5].T

    base_flat = pca_5d[:n_base_pts * B].reshape(B, n_base_pts, 5)
    dirt_flat = pca_5d[n_base_pts * B:].reshape(B, n_dirt_pts, 5)
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

    title = f"Hidden State Trajectory (PCA, seed {seed})"

    pc_pairs = [(i, j) for i in range(5) for j in range(i + 1, 5)]
    n_pairs = len(pc_pairs)
    n_cols = 2
    n_rows = (n_pairs + n_cols - 1) // n_cols
    fig_grid, axes_grid = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols + 1, 3 * n_rows + 1))
    axes_grid = axes_grid.flatten()

    base_color_line = "#A0A0A0"
    dirt_new_color = "#E67E22"
    dirt_review_color = "#2196F3"
    start_color = "#2ECC71"
    end_color = "#E74C3C"

    def draw_traj(ax, data_base, data_dirt, x_i, y_i, show_legend: bool):
        ax.plot(data_base[:, x_i], data_base[:, y_i], "o-",
                color=base_color_line, label="Base 12L", alpha=0.7, markersize=2.5, linewidth=1)
        for i in range(len(data_base)):
            ax.annotate(str(i), (data_base[i, x_i], data_base[i, y_i]),
                       fontsize=5.5, color=base_color_line, alpha=0.6)

        pt = 0
        ax.scatter(data_dirt[pt, x_i], data_dirt[pt, y_i],
                  c=[start_color], s=60, marker="*", zorder=5, label="start")
        for L in range(n_layers_dirt):
            z_L = data_dirt[pt]
            new_L = data_dirt[pt + 1]
            z_next = data_dirt[pt + 2]
            ax.annotate(str(L), (z_L[x_i], z_L[y_i]),
                       fontsize=6.5, color=dirt_review_color, fontweight="bold")
            ax.plot([z_L[x_i], new_L[x_i]], [z_L[y_i], new_L[y_i]],
                    "--", color=dirt_new_color, alpha=0.9, linewidth=1.5,
                    label="propose" if show_legend and L == 0 else "")
            ax.plot([new_L[x_i], z_next[x_i]], [new_L[y_i], z_next[y_i]],
                    "-", color=dirt_review_color, alpha=0.9, linewidth=2,
                    label="review" if show_legend and L == 0 else "")
            pt += 2

        ax.scatter(data_dirt[-1, x_i], data_dirt[-1, y_i],
                  c=[end_color], s=60, marker="s", zorder=5, label="end")
        ax.set_xlabel(f"PC{x_i + 1}")
        ax.set_ylabel(f"PC{y_i + 1}")
        ax.grid(True, alpha=0.3)
        if show_legend:
            ax.legend(fontsize=7, loc="best")

    for idx, (i, j) in enumerate(pc_pairs):
        draw_traj(axes_grid[idx], mean_base, mean_dirt, i, j, show_legend=(idx == 0))
        axes_grid[idx].set_title(f"PC{i + 1}-PC{j + 1}", fontsize=9)
    for idx in range(n_pairs, len(axes_grid)):
        axes_grid[idx].set_visible(False)
    fig_grid.suptitle(title, fontsize=12)
    fig_grid.tight_layout()
    fig_grid.savefig(output_dir / "trajectory_pca_2d_grid.png", dpi=150)
    plt.close(fig_grid)

    # --- 3D ---
    fig3d = plt.figure(figsize=(10, 8))
    ax3d = fig3d.add_subplot(111, projection="3d")
    ax3d.plot(mean_base[:, 0], mean_base[:, 1], mean_base[:, 2],
              "o-", color=base_color_line, label="Base 12L", alpha=0.7, markersize=3, linewidth=1)
    for i in range(len(mean_base)):
        ax3d.text(mean_base[i, 0], mean_base[i, 1], mean_base[i, 2],
                str(i), fontsize=6, color=base_color_line, alpha=0.6)
    pt = 0
    ax3d.scatter(*mean_dirt[0, :3], c=start_color, s=120, marker="*", zorder=5, label="start")
    for L in range(n_layers_dirt):
        z_L = mean_dirt[pt]
        new_L = mean_dirt[pt + 1]
        z_next = mean_dirt[pt + 2]
        ax3d.text(*z_L[:3], str(L), fontsize=8, color=dirt_review_color, fontweight="bold")
        ax3d.plot(*np.column_stack([z_L[:3], new_L[:3]]), "--",
                color=dirt_new_color, alpha=0.9, linewidth=2, label="propose" if L == 0 else "")
        ax3d.plot(*np.column_stack([new_L[:3], z_next[:3]]), "-",
                color=dirt_review_color, alpha=0.9, linewidth=3, label="review" if L == 0 else "")
        pt += 2
    ax3d.scatter(*mean_dirt[-1, :3], c=end_color, s=120, marker="s", zorder=5, label="end")
    ax3d.set_xlabel("PC1"); ax3d.set_ylabel("PC2"); ax3d.set_zlabel("PC3")
    ax3d.set_title(title)
    ax3d.legend(fontsize=8, loc="best")
    ax3d.grid(True, alpha=0.3)
    for angle in [30, 120]:
        ax3d.view_init(elev=20, azim=angle)
        fig3d.savefig(output_dir / f"trajectory_pca_3d_elev20_azim{angle}.png", dpi=150, bbox_inches="tight")
    plt.close(fig3d)

    # --- 3×3 sentence PCA grid ---
    if sentence_hidden_dirt is not None and sentence_hidden_base is not None:
        n_sent = 3
        n_sent_layers_dirt = len(sentence_hidden_dirt) - 1
        n_sent_layers_base = len(sentence_hidden_base) - 1
        sent_dirt_points = [sentence_hidden_dirt[0]]
        for L in range(n_sent_layers_dirt):
            z_L = sentence_hidden_dirt[L]
            new_L = z_L + delta_v[L][:n_sent]
            sent_dirt_points.append(new_L)
            sent_dirt_points.append(sentence_hidden_dirt[L + 1])
        n_sent_dirt_pts = len(sent_dirt_points)

        base_cmap_name = "OrRd"
        dirt_cmap_name = "PuBu"

        pc_pairs_sent = [(0, 1), (2, 3), (4, 5)]
        pc_targets = [2, 4, 6]

        titles_short = []
        for t in sentence_texts:
            short = t.strip().replace("\n", " ")[-40:].strip()
            titles_short.append(f"...{short}" if len(t) > 40 else short)

        fig_sent, axes_sent = plt.subplots(3, 3, figsize=(12, 12))
        for row in range(n_sent):
            stacked = np.concatenate(
                [np.stack(sent_dirt_points, axis=1)[row].reshape(-1, D),
                 np.stack(sentence_hidden_base, axis=1)[row].reshape(-1, D)],
                axis=0,
            )
            mean_s = np.mean(stacked, axis=0, keepdims=True)
            centered_s = stacked - mean_s
            U_s, S_s, Vt_s = np.linalg.svd(centered_s, full_matrices=False)

            for col, (pc_i, pc_j) in enumerate(pc_pairs_sent):
                ax = axes_sent[row][col]
                n_comp = pc_targets[col]
                proj = centered_s @ Vt_s[:n_comp].T
                n_base_pts_s = n_sent_layers_base + 1
                base_proj = proj[:n_base_pts_s]
                dirt_proj = proj[n_base_pts_s:]

                base_cmap = plt.get_cmap(base_cmap_name)
                dirt_cmap = plt.get_cmap(dirt_cmap_name)

                n_base_pts = len(base_proj)
                for i in range(n_base_pts - 1):
                    alpha = 0.3 + 0.7 * i / max(n_base_pts - 2, 1)
                    ax.plot(base_proj[i:i+2, pc_i], base_proj[i:i+2, pc_j], "-o",
                            color=base_cmap(alpha), alpha=0.9, markersize=4, linewidth=1.5)
                ax.scatter(base_proj[-1, pc_i], base_proj[-1, pc_j],
                          c=[base_cmap(1.0)], s=40, marker="s", zorder=4)

                n_dirt_pts = len(dirt_proj)
                pt = 0
                for L in range(n_sent_layers_dirt):
                    alpha = 0.3 + 0.7 * L / max(n_sent_layers_dirt - 1, 1)
                    c = dirt_cmap(alpha)
                    z_L = dirt_proj[pt]
                    new_L = dirt_proj[pt + 1]
                    z_next = dirt_proj[pt + 2]
                    ax.plot([z_L[pc_i], new_L[pc_i]], [z_L[pc_j], new_L[pc_j]],
                            "--", color=c, alpha=0.9, linewidth=1.5)
                    ax.plot([new_L[pc_i], z_next[pc_i]], [new_L[pc_j], z_next[pc_j]],
                            "-", color=c, alpha=0.9, linewidth=2)
                    ax.annotate(str(L), (z_L[pc_i], z_L[pc_j]),
                               fontsize=7, color=c, fontweight="bold")
                    pt += 2

                ax.scatter(base_proj[0, pc_i], base_proj[0, pc_j],
                          c=[base_cmap(0.3)], s=50, marker="*", zorder=5)
                ax.scatter(base_proj[-1, pc_i], base_proj[-1, pc_j],
                          c=[base_cmap(1.0)], s=50, marker="s", zorder=5)
                ax.scatter(dirt_proj[0, pc_i], dirt_proj[0, pc_j],
                          c=[dirt_cmap(0.3)], s=50, marker="*", zorder=5)
                ax.scatter(dirt_proj[-1, pc_i], dirt_proj[-1, pc_j],
                          c=[dirt_cmap(1.0)], s=50, marker="s", zorder=5)

                ax.set_xlabel(f"PC{pc_i + 1}")
                ax.set_ylabel(f"PC{pc_j + 1}")
                ax.grid(True, alpha=0.3)
                if row == 0:
                    ax.set_title(f"PC{pc_i + 1}-PC{pc_j + 1}", fontsize=10)

            axes_sent[row][0].set_ylabel(titles_short[row], fontsize=8)

        fig_sent.suptitle(f"Sentence Trajectory (seed {seed})", fontsize=14)
        fig_sent.tight_layout()
        fig_sent.savefig(output_dir / "trajectory_pca_sentence_grid.png", dpi=150)
        plt.close(fig_sent)

    # --- Propose vs Review bar ---
    fig_bar, ax_bar = plt.subplots(figsize=(8, 4))
    x = np.arange(n_layers_dirt)
    w = 0.35
    ax_bar.bar(x - w / 2, dv_norms, w, label="propose (||delta_v||)", color=dirt_new_color, alpha=0.8)
    ax_bar.bar(x + w / 2, rv_norms, w, label="review (||review||)", color=dirt_review_color, alpha=0.8)
    ax_bar.set_xlabel("Layer"); ax_bar.set_ylabel("Mean norm")
    ax_bar.set_title("Propose vs Review Step Size per Layer")
    ax_bar.set_xticks(x); ax_bar.legend(); ax_bar.grid(True, axis="y", alpha=0.3)
    fig_bar.tight_layout()
    fig_bar.savefig(output_dir / "trajectory_pca_propose_vs_review.png", dpi=150)
    plt.close(fig_bar)

    print(f"\n=== trajectory_pca (seed {seed}) ===")
    print(f"  Trajectory length: Base={traj_len_base:.4f}  DiRT={traj_len_dirt:.4f}")
    print(f"  Subsample: {n_subsample} tokens")
    for L in range(n_layers_dirt):
        ratio = rv_norms[L] / dv_norms[L] if dv_norms[L] > 1e-12 else float("inf")
        print(f"  L{L}: ||propose||={dv_norms[L]:.6f}  ||review||={rv_norms[L]:.6f}  ratio={ratio:.2f}")

    return results
