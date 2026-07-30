from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def _decode(tokenizer, ids):
    if tokenizer is None:
        return f"[id={ids[0]}]"
    try:
        return tokenizer.decode(ids, skip_special_tokens=True)
    except TypeError:
        return tokenizer.decode(ids)


def run(
    magnitudes: list[np.ndarray],
    dirt_loss: np.ndarray,
    pos_ids: np.ndarray,
    token_ids: dict[int, np.ndarray],
    n_layers: int,
    output_dir: Path,
    seed: int,
    tokenizer=None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    position = pos_ids[:, 2]
    bands = [(0, 1), (1, 10), (10, 50), (50, position.max() + 1)]
    band_labels = ["pos=0", "pos1-10", "pos10-50", "pos50+"]

    results = {}
    for L in range(n_layers):
        mag = magnitudes[L]
        loss = dirt_loss
        valid = mag > -1e10
        if np.sum(valid) < 10:
            continue

        rho_all, p_all = spearmanr(mag[valid], loss[valid])
        results[f"spearman_L{L}"] = float(rho_all)
        results[f"spearman_p_L{L}"] = float(p_all)

        mask_clean = valid & (position > 0)
        c = int(np.sum(mask_clean))
        if c > 10:
            rho_clean, p_clean = spearmanr(mag[mask_clean], loss[mask_clean])
            results[f"spearman_clean_L{L}"] = float(rho_clean)
            results[f"spearman_clean_p_L{L}"] = float(p_clean)
        else:
            results[f"spearman_clean_L{L}"] = 0.0
            results[f"spearman_clean_p_L{L}"] = 1.0

        for bi, (lo, hi) in enumerate(bands):
            mask_band = valid & (position >= lo) & (position < hi)
            cb = int(np.sum(mask_band))
            if cb > 10:
                rho_b, p_b = spearmanr(mag[mask_band], loss[mask_band])
            else:
                rho_b, p_b = 0.0, 1.0
            results[f"strat_rho_L{L}_{band_labels[bi]}"] = float(rho_b)
            results[f"strat_p_L{L}_{band_labels[bi]}"] = float(p_b)

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 5))
            s = 0.3 if len(mag) > 10000 else 1.0
            ax.scatter(loss[valid], mag[valid], alpha=0.3, s=s, c="steelblue")
            ax.set_xlabel("Token NLL")
            ax.set_ylabel("|magnitude|")
            ax.set_title(f"L{L}: ρ_all={rho_all:.4f} ρ_clean={results[f'spearman_clean_L{L}']:.4f} (seed {seed})")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            fig.savefig(output_dir / f"2b_magnitude_vs_loss_L{L}.png", dpi=150)
            plt.close(fig)

        except ImportError:
            pass

        n_top = 20
        pos0 = position == 0
        mask_pos = valid & (position > 0)

        ctx_path = output_dir / f"2b_top_bottom_L{L}_seed{seed}.txt"
        with open(ctx_path, "w") as f:
            f.write(f"=== Layer {L} — position=0 only ===\n")
            mag_pos0 = mag[pos0]
            if len(mag_pos0) > 0:
                top0 = np.argsort(mag_pos0)[-min(n_top, len(mag_pos0)):]
                for idx in np.array(np.where(pos0)[0])[top0]:
                    b_id, b, t = pos_ids[idx]
                    ids_arr = token_ids.get(b_id, np.array([[0]]))
                    if b < ids_arr.shape[0] and t < ids_arr.shape[1]:
                        ctx_start = max(0, t - 20)
                        ctx_ids = ids_arr[b, ctx_start:t + 1].tolist()
                        ctx_str = _decode(tokenizer, ctx_ids)
                        tok_str = _decode(tokenizer, [int(ids_arr[b, t])])
                        f.write(f"  loss={loss[idx]:.4f} | mag={mag[idx]:.6f} | ctx[-20:] → [{tok_str}]\n")
                        f.write(f"    {ctx_str}\n")

            f.write(f"\n=== Layer {L} — position>0, dedup by token ===\n")
            idx_pos = np.where(mask_pos)[0]
            if len(idx_pos) > 0:
                mag_pos = mag[idx_pos]
                seen_tokens = {}
                order = np.argsort(mag_pos)
                for idx in reversed(order):
                    b_id, b, t = pos_ids[idx_pos[idx]]
                    ids_arr = token_ids.get(b_id, np.array([[0]]))
                    if b < ids_arr.shape[0] and t < ids_arr.shape[1]:
                        tok_id = int(ids_arr[b, t])
                        tok_str = _decode(tokenizer, [tok_id])
                        if tok_str not in seen_tokens:
                            ctx_start = max(0, t - 20)
                            ctx_ids = ids_arr[b, ctx_start:t + 1].tolist()
                            ctx_str = _decode(tokenizer, ctx_ids)
                            seen_tokens[tok_str] = (mag[idx_pos[idx]], loss[idx_pos[idx]], ctx_str)
                        if len(seen_tokens) >= n_top:
                            break
                sorted_tokens = sorted(seen_tokens.items(), key=lambda x: -x[1][0])[:n_top]
                for tok_str, (m_val, l_val, ctx_str) in sorted_tokens:
                    f.write(f"  loss={l_val:.4f} | mag={m_val:.6f} | ctx[-20:] → [{tok_str}]\n")
                    f.write(f"    {ctx_str}\n")

            f.write(f"\n=== Layer {L} — |magnitude| 하위 {n_top} (position>0) ===\n")
            if len(idx_pos) > 0:
                mag_pos = mag[idx_pos]
                seen_low = {}
                order = np.argsort(mag_pos)
                for idx in order:
                    b_id, b, t = pos_ids[idx_pos[idx]]
                    ids_arr = token_ids.get(b_id, np.array([[0]]))
                    if b < ids_arr.shape[0] and t < ids_arr.shape[1]:
                        tok_id = int(ids_arr[b, t])
                        tok_str = _decode(tokenizer, [tok_id])
                        if tok_str not in seen_low:
                            ctx_start = max(0, t - 20)
                            ctx_ids = ids_arr[b, ctx_start:t + 1].tolist()
                            ctx_str = _decode(tokenizer, ctx_ids)
                            seen_low[tok_str] = (mag[idx_pos[idx]], loss[idx_pos[idx]], ctx_str)
                        if len(seen_low) >= n_top:
                            break
                sorted_low = sorted(seen_low.items(), key=lambda x: x[1][0])[:n_top]
                for tok_str, (m_val, l_val, ctx_str) in sorted_low:
                    f.write(f"  loss={l_val:.4f} | mag={m_val:.6f} | ctx[-20:] → [{tok_str}]\n")
                    f.write(f"    {ctx_str}\n")

    print(f"\n=== magnitude_difficulty (seed {seed}) ===")
    for L in range(n_layers):
        k = f"spearman_L{L}"
        if k in results:
            clean = results.get(f"spearman_clean_L{L}", 0)
            print(f"  L{L}: ρ_all={results[k]:+.4f}  ρ_pos>0={clean:+.4f}")
    print(f"\n  Stratified by position:")
    for L in range(n_layers):
        parts = [f"  L{L}:"]
        for bl in band_labels:
            v = results.get(f"strat_rho_L{L}_{bl}", 0)
            parts.append(f"{bl}={v:+.4f}")
        print(" ".join(parts))

    return results
