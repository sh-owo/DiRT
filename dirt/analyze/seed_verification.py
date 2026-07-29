from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats


def run_aggregate(results_by_seed: dict[int, dict], output_dir: Path) -> dict:
    seeds = sorted(results_by_seed.keys())
    dirt_losses = []
    base_losses = []
    for s in seeds:
        r = results_by_seed[s]
        dl = r.get("val_loss_dirt")
        bl = r.get("val_loss_base")
        if dl is not None and bl is not None:
            dirt_losses.append(dl)
            base_losses.append(bl)

    dirt_losses = np.array(dirt_losses)
    base_losses = np.array(base_losses)
    diffs = dirt_losses - base_losses

    n = len(diffs)
    mean_diff = float(np.mean(diffs)) if n > 0 else 0.0
    std_diff = float(np.std(diffs, ddof=1)) if n > 1 else 0.0

    if n > 1 and std_diff > 1e-12:
        t_stat = mean_diff / (std_diff / np.sqrt(n))
        p_val = 2 * scipy_stats.t.sf(abs(t_stat), df=n - 1)
    else:
        t_stat = 0.0
        p_val = 1.0

    neg_count = int(np.sum(diffs < 0))

    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "1a_seed_table.txt"
    with open(table_path, "w") as f:
        f.write(f"{'seed':>6} | {'dirt_loss':>10} | {'base_loss':>10} | {'diff':>10}\n")
        f.write("-" * 44 + "\n")
        for i, s in enumerate(seeds):
            dl = dirt_losses[i] if i < len(dirt_losses) else float("nan")
            bl = base_losses[i] if i < len(base_losses) else float("nan")
            d = diffs[i] if i < len(diffs) else float("nan")
            f.write(f"{s:6d} | {dl:10.6f} | {bl:10.6f} | {d:10.6f}\n")
        f.write("-" * 44 + "\n")
        f.write(f"{'mean':>6} | {float(np.mean(dirt_losses)):10.6f} | {float(np.mean(base_losses)):10.6f} | {mean_diff:10.6f}\n")
        f.write(f"{'std':>6} | {float(np.std(dirt_losses, ddof=1)):10.6f} | {float(np.std(base_losses, ddof=1)):10.6f} | {std_diff:10.6f}\n\n")
        f.write(f"t-statistic: {t_stat:.6f}\n")
        f.write(f"p-value:     {p_val:.6f}\n")
        f.write(f"neg count:   {neg_count}/{n}\n")
        if n >= 5:
            sig = "significant" if p_val < 0.05 else "not significant"
            f.write(f"Result: {sig} (p={'<' if p_val < 0.05 else '>'} 0.05)\n")

    print(f"\n=== seed_verification ===")
    print(f"  {table_path}")
    for i, s in enumerate(seeds):
        print(f"  seed {s}: dirt={dirt_losses[i]:.6f}  base={base_losses[i]:.6f}  diff={diffs[i]:.6f}")
    print(f"  mean_diff={mean_diff:.6f}  std_diff={std_diff:.6f}")
    print(f"  t={t_stat:.4f}  p={p_val:.4e}  neg={neg_count}/{n}")

    return {
        "mean_dirt_loss": float(np.mean(dirt_losses)) if len(dirt_losses) > 0 else 0.0,
        "mean_base_loss": float(np.mean(base_losses)) if len(base_losses) > 0 else 0.0,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "t_statistic": t_stat,
        "p_value": p_val,
        "neg_count": neg_count,
        "n_seeds": n,
    }
