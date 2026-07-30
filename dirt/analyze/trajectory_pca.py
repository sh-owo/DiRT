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
    sent_full_hidden_dirt: list[np.ndarray] | None = None,
    sent_full_hidden_base: list[np.ndarray] | None = None,
    sent_token_ids: np.ndarray | None = None,
    sent_text: str | None = None,
    sent_dv_raw: list[np.ndarray] | None = None,
    sent_rv_raw: list[np.ndarray] | None = None,
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
    ev_ratio = S**2 / np.sum(S**2)
    print(f"  Explained variance: PC1={ev_ratio[0]:.1%} PC2={ev_ratio[1]:.1%} PC3={ev_ratio[2]:.1%} "
          f"PC4={ev_ratio[3]:.1%} PC5={ev_ratio[4]:.1%}  "
          f"PC1~PC3={np.sum(ev_ratio[:3]):.1%}")
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
        "ev_pc1": float(ev_ratio[0]),
        "ev_pc2": float(ev_ratio[1]),
        "ev_pc3": float(ev_ratio[2]),
        "ev_pc4": float(ev_ratio[3]),
        "ev_pc5": float(ev_ratio[4]),
        "ev_pc1_3": float(np.sum(ev_ratio[:3])),
    }

    dv_norms = [float(np.mean(np.linalg.norm(delta_v[L], axis=-1))) for L in range(n_layers_dirt)]
    rv_norms = [float(np.mean(np.linalg.norm(review[L], axis=-1))) for L in range(n_layers_dirt)]
    dv_norms_std = [float(np.std(np.linalg.norm(delta_v[L], axis=-1))) for L in range(n_layers_dirt)]
    rv_norms_std = [float(np.std(np.linalg.norm(review[L], axis=-1))) for L in range(n_layers_dirt)]
    dv_norms_min = [float(np.min(np.linalg.norm(delta_v[L], axis=-1))) for L in range(n_layers_dirt)]
    rv_norms_min = [float(np.min(np.linalg.norm(review[L], axis=-1))) for L in range(n_layers_dirt)]
    dv_norms_max = [float(np.max(np.linalg.norm(delta_v[L], axis=-1))) for L in range(n_layers_dirt)]
    rv_norms_max = [float(np.max(np.linalg.norm(review[L], axis=-1))) for L in range(n_layers_dirt)]
    for L in range(n_layers_dirt):
        results[f"propose_norm_L{L}"] = dv_norms[L]
        results[f"propose_std_L{L}"] = dv_norms_std[L]
        results[f"propose_min_L{L}"] = dv_norms_min[L]
        results[f"propose_max_L{L}"] = dv_norms_max[L]
        results[f"review_norm_L{L}"] = rv_norms[L]
        results[f"review_std_L{L}"] = rv_norms_std[L]
        results[f"review_min_L{L}"] = rv_norms_min[L]
        results[f"review_max_L{L}"] = rv_norms_max[L]

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

    # --- Single-token sentence trajectory (2D grid + 3D) ---
    if (sent_full_hidden_dirt is not None and sent_full_hidden_base is not None
            and sent_dv_raw is not None and sent_rv_raw is not None):
        T_full = sent_token_ids.shape[0]

        # Find anchor: first period in decoded text + offset
        text_preview = tokenizer.decode(sent_token_ids.tolist()) if tokenizer else ""
        first_period = text_preview.find(". ")
        if first_period == -1:
            first_period = text_preview.find(".")
        if first_period > 0:
            char_count = 0
            anchor = min(50, T_full // 2)
            for i in range(min(T_full, 500)):
                tok_str = tokenizer.decode([int(sent_token_ids[i])]) if tokenizer else str(int(sent_token_ids[i]))
                char_count += len(tok_str)
                if char_count > first_period + 2:
                    anchor = min(i + 3, T_full - 10)
                    break
        else:
            anchor = min(100, T_full // 2)

        n_win = 5
        token_positions = [anchor + i for i in range(-2, 3)]
        token_positions = [max(0, min(p, T_full - 1)) for p in token_positions]

        def _build_dirt_traj(hidden, dv, n_l, pos):
            pts = [hidden[0][pos]]
            for L in range(n_l):
                pts.append(pts[-1] + dv[L][pos])
                pts.append(hidden[L + 1][pos])
            return np.stack(pts, axis=0)

        def _build_base_traj(hidden, n_l, pos):
            return np.stack([hidden[L][pos] for L in range(n_l + 1)], axis=0)

        def _token_str(ids, pos, tok):
            tid = int(ids[pos])
            s = tok.decode([tid]) if tok else str(tid)
            return s if s else f"[id={tid}]"

        pc_pairs_s = [(0, 1), (2, 3), (4, 5)]
        pc_targets_s = [2, 4, 6]

        # DiRT 5x3 grid
        fig_5d, axes_5d = plt.subplots(5, 3, figsize=(12, 18))
        for row, pos in enumerate(token_positions):
            traj_d = _build_dirt_traj(sent_full_hidden_dirt, sent_dv_raw, n_layers_dirt, pos)
            tk = _token_str(sent_token_ids, pos, tokenizer)
            for col, (p_i, p_j) in enumerate(pc_pairs_s):
                n_comp = pc_targets_s[col]
                centered = traj_d - traj_d.mean(axis=0, keepdims=True)
                U, S, Vt = np.linalg.svd(centered, full_matrices=False)
                proj = centered @ Vt[:n_comp].T
                ax = axes_5d[row][col]
                cmap_d = plt.cm.viridis
                pt = 0
                for L in range(n_layers_dirt):
                    alpha = 0.3 + 0.7 * L / max(n_layers_dirt - 1, 1)
                    c = cmap_d(alpha)
                    z_L = proj[pt]
                    new_L = proj[pt + 1]
                    z_n = proj[pt + 2]
                    ax.plot([z_L[p_i], new_L[p_i]], [z_L[p_j], new_L[p_j]],
                            "--", color=c, alpha=0.9, linewidth=1.5)
                    ax.plot([new_L[p_i], z_n[p_i]], [new_L[p_j], z_n[p_j]],
                            "-", color=c, alpha=0.9, linewidth=2)
                    ax.annotate(str(L), (z_L[p_i], z_L[p_j]),
                               fontsize=7, color=c, fontweight="bold")
                    pt += 2
                ax.scatter(proj[0, p_i], proj[0, p_j],
                          c=[cmap_d(0.3)], s=80, marker="*", zorder=5)
                ax.scatter(proj[-1, p_i], proj[-1, p_j],
                          c=[cmap_d(1.0)], s=80, marker="X", zorder=5)
                ax.set_xlabel(f"PC{p_i + 1}")
                ax.set_ylabel(f"PC{p_j + 1}")
                ax.grid(True, alpha=0.3)
            axes_5d[row][0].set_ylabel(f"Token: \"{tk}\"", fontsize=8)
        fig_5d.suptitle(f"DiRT Single-Token Trajectories (seed {seed})", fontsize=12)
        fig_5d.tight_layout()
        fig_5d.savefig(output_dir / "trajectory_pca_2d_grid_dirt.png", dpi=150)
        plt.close(fig_5d)

        # Base 5x3 grid
        fig_5b, axes_5b = plt.subplots(5, 3, figsize=(12, 18))
        for row, pos in enumerate(token_positions):
            traj_b = _build_base_traj(sent_full_hidden_base, n_layers_base, pos)
            tk = _token_str(sent_token_ids, pos, tokenizer)
            n_l_base = traj_b.shape[0] - 1
            for col, (p_i, p_j) in enumerate(pc_pairs_s):
                n_comp = pc_targets_s[col]
                centered = traj_b - traj_b.mean(axis=0, keepdims=True)
                U, S, Vt = np.linalg.svd(centered, full_matrices=False)
                proj = centered @ Vt[:n_comp].T
                ax = axes_5b[row][col]
                cmap_b = plt.cm.viridis
                for i in range(n_l_base):
                    alpha = 0.3 + 0.7 * i / max(n_l_base - 1, 1)
                    c = cmap_b(alpha)
                    ax.plot(proj[i:i+2, p_i], proj[i:i+2, p_j], "-o",
                            color=c, alpha=0.9, markersize=4, linewidth=1.5)
                ax.scatter(proj[0, p_i], proj[0, p_j],
                          c=[cmap_b(0.3)], s=80, marker="*", zorder=5)
                ax.scatter(proj[-1, p_i], proj[-1, p_j],
                          c=[cmap_b(1.0)], s=80, marker="X", zorder=5)
                ax.set_xlabel(f"PC{p_i + 1}")
                ax.set_ylabel(f"PC{p_j + 1}")
                ax.grid(True, alpha=0.3)
            axes_5b[row][0].set_ylabel(f"Token: \"{tk}\"", fontsize=8)
        fig_5b.suptitle(f"Base Single-Token Trajectories (seed {seed})", fontsize=12)
        fig_5b.tight_layout()
        fig_5b.savefig(output_dir / "trajectory_pca_2d_grid_base.png", dpi=150)
        plt.close(fig_5b)

        # DiRT 3D (anchor token)
        pos_3d = token_positions[2]
        traj_d3 = _build_dirt_traj(sent_full_hidden_dirt, sent_dv_raw, n_layers_dirt, pos_3d)
        tk_3d = _token_str(sent_token_ids, pos_3d, tokenizer)
        centered_d3 = traj_d3 - traj_d3.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(centered_d3, full_matrices=False)
        proj_d3 = centered_d3 @ Vt[:3].T

        fig_3d = plt.figure(figsize=(10, 8))
        ax_3d = fig_3d.add_subplot(111, projection="3d")
        cmap_d3 = plt.cm.viridis
        pt = 0
        for L in range(n_layers_dirt):
            alpha = 0.3 + 0.7 * L / max(n_layers_dirt - 1, 1)
            c = cmap_d3(alpha)
            z_L = proj_d3[pt]
            new_L = proj_d3[pt + 1]
            z_n = proj_d3[pt + 2]
            ax_3d.plot(*np.column_stack([z_L, new_L]), "--", color=c, alpha=0.9, linewidth=2)
            ax_3d.plot(*np.column_stack([new_L, z_n]), "-", color=c, alpha=0.9, linewidth=3)
            ax_3d.text(*z_L, str(L), fontsize=8, color=c, fontweight="bold")
            pt += 2
        ax_3d.scatter(*proj_d3[0], c=[cmap_d3(0.3)], s=120, marker="*", zorder=5, label="start")
        ax_3d.scatter(*proj_d3[-1], c=[cmap_d3(1.0)], s=120, marker="X", zorder=5, label="end")
        ax_3d.set_xlabel("PC1"); ax_3d.set_ylabel("PC2"); ax_3d.set_zlabel("PC3")
        ax_3d.set_title(f"DiRT Token: \"{tk_3d}\"\nPC1-PC2-PC3 (seed {seed})")
        ax_3d.legend(); ax_3d.grid(True, alpha=0.3)
        fig_3d.tight_layout()
        fig_3d.savefig(output_dir / "trajectory_pca_3d_dirt.png", dpi=150)
        plt.close(fig_3d)

        # Base 3D (same token)
        traj_b3 = _build_base_traj(sent_full_hidden_base, n_layers_base, pos_3d)
        centered_b3 = traj_b3 - traj_b3.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(centered_b3, full_matrices=False)
        proj_b3 = centered_b3 @ Vt[:3].T
        n_l_base = traj_b3.shape[0] - 1

        fig_3b = plt.figure(figsize=(10, 8))
        ax_3b = fig_3b.add_subplot(111, projection="3d")
        cmap_b3 = plt.cm.viridis
        for i in range(n_l_base):
            alpha = 0.3 + 0.7 * i / max(n_l_base - 1, 1)
            c = cmap_b3(alpha)
            ax_3b.plot(*np.column_stack([proj_b3[i], proj_b3[i+1]]), "-o",
                      color=c, alpha=0.9, linewidth=2, markersize=4)
            ax_3b.text(*proj_b3[i], str(i), fontsize=7, color=c, alpha=0.8)
        ax_3b.scatter(*proj_b3[0], c=[cmap_b3(0.3)], s=120, marker="*", zorder=5, label="start")
        ax_3b.scatter(*proj_b3[-1], c=[cmap_b3(1.0)], s=120, marker="X", zorder=5, label="end")
        ax_3b.set_xlabel("PC1"); ax_3b.set_ylabel("PC2"); ax_3b.set_zlabel("PC3")
        ax_3b.set_title(f"Base Token: \"{tk_3d}\"\nPC1-PC2-PC3 (seed {seed})")
        ax_3b.legend(); ax_3b.grid(True, alpha=0.3)
        fig_3b.tight_layout()
        fig_3b.savefig(output_dir / "trajectory_pca_3d_base.png", dpi=150)
        plt.close(fig_3b)

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
