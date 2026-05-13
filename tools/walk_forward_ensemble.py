"""
Walk-forward validation with SEED ENSEMBLE.

Per fold:
  1. Train N models (different seeds)
  2. At test time, each model votes on action (majority vote)
  3. One env executes the consensus action
  -> Variance from single-seed PPO is averaged out

This is the production-grade answer to "which seed did I draw?"

Usage:
    python -m tools.walk_forward_ensemble                   # default N=5 seeds
    python -m tools.walk_forward_ensemble --n-seeds 3       # lighter
    python -m tools.walk_forward_ensemble --quick           # 50k steps (smoke test)
"""
from __future__ import annotations

import argparse
import io
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from stable_baselines3 import PPO

from src.backtest import (
    annualization_factor, max_drawdown, profit_factor, sharpe_ratio,
)
from src.data import load_data
from src.env import GoldTradingEnv
from tools.walk_forward import (
    buy_and_hold_equity_return, make_env_fn, train_fold,
)


# ---------------------------------------------------------------------------
# Ensemble helpers
# ---------------------------------------------------------------------------

def train_ensemble(train_df, cfg, timesteps, base_seed, n_seeds):
    """Train N models with different seeds. Return list of (model, vn) tuples."""
    models = []
    for i in range(n_seeds):
        seed = base_seed + i * 7  # spread seeds to avoid correlation
        print(f"   seed {i+1}/{n_seeds} (seed={seed})...", end=" ", flush=True)
        t0 = time.time()
        model, vn = train_fold(train_df, cfg, timesteps, seed)
        print(f"({(time.time()-t0)/60:.1f}m)")
        models.append((model, vn))
    return models


def ensemble_backtest(test_df, cfg, models, method="prob_avg"):
    """
    Backtest with ensemble action selection.

    method="prob_avg" (default):
        Average action probability distributions across all models, then argmax.
        Preserves partial confidence (e.g., 60% Buy from 3 models + 100% Hold from 2
        -> averaged probs may still favor Buy). Much less conservative than majority vote.

    method="majority_vote" (legacy):
        Each model casts a discrete vote. Most-voted action wins.
        Ties broken by lowest action index (Hold-biased). Known to be too conservative.

    All models see the same env state (same obs).
    """
    import torch as th

    env_kwargs = cfg["env"]
    raw_env = GoldTradingEnv(test_df, **env_kwargs)

    obs, _ = raw_env.reset()
    action_counts = Counter()  # track consensus actions
    n_actions = raw_env.action_space.n  # 4

    while True:
        if method == "prob_avg":
            # ---- Probability averaging: average policy distributions ----
            probs_list = []
            for model, vn in models:
                if vn is not None:
                    norm_obs = vn.normalize_obs(obs[None, :])[0]
                else:
                    norm_obs = obs
                # Convert to tensor and get action distribution
                obs_tensor, _ = model.policy.obs_to_tensor(norm_obs)
                with th.no_grad():
                    dist = model.policy.get_distribution(obs_tensor)
                    probs = dist.distribution.probs.cpu().numpy().flatten()
                probs_list.append(probs)

            # Average probabilities across all seeds
            avg_probs = np.mean(probs_list, axis=0)
            consensus = int(np.argmax(avg_probs))

        else:
            # ---- Legacy majority vote ----
            votes = []
            for model, vn in models:
                if vn is not None:
                    norm_obs = vn.normalize_obs(obs[None, :])[0]
                else:
                    norm_obs = obs
                action, _ = model.predict(norm_obs, deterministic=True)
                action = int(np.asarray(action).flatten()[0])
                votes.append(action)

            vote_counter = Counter(votes)
            max_count = max(vote_counter.values())
            candidates = [a for a, c in vote_counter.items() if c == max_count]
            consensus = min(candidates)

        action_counts[consensus] += 1

        obs, _, term, trunc, _ = raw_env.step(consensus)
        if term or trunc:
            break

    equity = np.array(raw_env.equity_curve, dtype=np.float64)
    ann = annualization_factor(cfg["data"]["interval"])
    bot_return = (equity[-1] / equity[0] - 1) * 100
    bnh_return = buy_and_hold_equity_return(test_df, env_kwargs)
    bnh_raw = (test_df["close"].iloc[-1] / test_df["close"].iloc[0] - 1) * 100

    return {
        "total_return_pct": bot_return,
        "total_trades": raw_env.total_trades,
        "win_rate_pct": raw_env.winning_trades / max(raw_env.total_trades, 1) * 100,
        "max_dd_pct": max_drawdown(equity) * 100,
        "sharpe": sharpe_ratio(equity, ann),
        "profit_factor": profit_factor(raw_env.trade_returns),
        "buy_and_hold_pct": bnh_return,
        "buy_and_hold_raw_pct": bnh_raw,
        "alpha_pct": bot_return - bnh_return,
        "equity_curve": equity,
        "action_distribution": dict(action_counts),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--train-size", type=int, default=1500)
    parser.add_argument("--test-size", type=int, default=400)
    parser.add_argument("--step-size", type=int, default=400)
    parser.add_argument("--timesteps", type=int, default=200000)
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 50k steps/seed (~1 min/seed)")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--n-seeds", type=int, default=5,
                        help="Number of seeds per fold (default 5)")
    parser.add_argument("--method", default="prob_avg",
                        choices=["prob_avg", "majority_vote"],
                        help="Ensemble method: prob_avg (default) or majority_vote")
    args = parser.parse_args()

    if args.quick:
        args.timesteps = 50_000

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print("=" * 70)
    print(" WALK-FORWARD ENSEMBLE VALIDATION")
    print("=" * 70)
    print(f"  train window:  {args.train_size:,} rows")
    print(f"  test window:   {args.test_size:,} rows")
    print(f"  step size:     {args.step_size:,} rows")
    print(f"  timesteps/seed: {args.timesteps:,}")
    print(f"  seeds/fold:    {args.n_seeds}")
    print(f"  mode: {'QUICK (smoke test)' if args.quick else 'FULL'}")
    print(f"  ensemble method: {args.method}")
    print(f"  reward: {cfg['env']['reward_type']}")

    print("\nLoading data...")
    df = load_data(cfg)
    print(f"  total rows: {len(df):,}")

    # Build folds
    folds = []
    start = 0
    while start + args.train_size + args.test_size <= len(df):
        train_df = df.iloc[start:start + args.train_size].reset_index(drop=True)
        test_df = df.iloc[
            start + args.train_size : start + args.train_size + args.test_size
        ].reset_index(drop=True)
        folds.append((train_df, test_df, start))
        start += args.step_size

    n_folds = len(folds)
    print(f"  folds: {n_folds}")

    if n_folds == 0:
        print("\nERROR: ข้อมูลไม่พอ")
        return

    # Estimate time
    est_fps = 280
    est_secs = args.timesteps / est_fps * args.n_seeds * n_folds
    print(f"  est. time: ~{est_secs/60:.0f} min")

    # Run
    results = []
    t_start = time.time()

    for i, (train_df, test_df, start_idx) in enumerate(folds):
        elapsed = (time.time() - t_start) / 60
        eta = elapsed / max(i, 1) * (n_folds - i) if i > 0 else est_secs / 60

        print(f"\n{'=' * 70}")
        print(f" Fold {i+1}/{n_folds} | rows {start_idx}-"
              f"{start_idx + args.train_size + args.test_size}"
              f" | elapsed {elapsed:.1f}m | ETA {eta:.0f}m")
        print(f"{'=' * 70}")

        print(f" [train] {args.n_seeds} seeds x {args.timesteps:,} steps...")
        models = train_ensemble(
            train_df, cfg, args.timesteps, args.base_seed + i * 100, args.n_seeds
        )

        print(f" [test] ensemble {args.method} backtest...")
        m = ensemble_backtest(test_df, cfg, models, method=args.method)
        m["fold"] = i + 1
        m["start_row"] = start_idx
        results.append(m)

        # Action distribution
        ad = m.get("action_distribution", {})
        total_steps = sum(ad.values()) or 1
        ad_str = " ".join(
            f"{['Hold','Buy','Sell','Close'][k]}={v}({v/total_steps*100:.0f}%)"
            for k, v in sorted(ad.items())
        )

        print(f"   -> return={m['total_return_pct']:+6.2f}%  "
              f"sharpe={m['sharpe']:5.2f}  "
              f"DD={m['max_dd_pct']:5.2f}%  "
              f"trades={m['total_trades']:4d}  "
              f"alpha={m['alpha_pct']:+6.2f}%")
        print(f"      actions: {ad_str}")

        # Free memory
        del models

    # ---------- Summary ----------
    print("\n" + "=" * 70)
    print(" ENSEMBLE WALK-FORWARD RESULTS")
    print("=" * 70)
    print(f"  {'Fold':<5} {'Return%':>9} {'Sharpe':>7} {'DD%':>7} {'WinR%':>7} "
          f"{'Trades':>7} {'BnH%':>8} {'Alpha%':>8}")
    print(f"  {'-'*5} {'-'*9} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*8}")
    for r in results:
        print(f"  {r['fold']:<5} {r['total_return_pct']:>+9.2f} "
              f"{r['sharpe']:>7.2f} {r['max_dd_pct']:>7.2f} "
              f"{r['win_rate_pct']:>7.2f} {r['total_trades']:>7} "
              f"{r['buy_and_hold_pct']:>+8.2f} {r['alpha_pct']:>+8.2f}")
    print(f"  {'-'*5} {'-'*9} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*8}")

    returns = np.array([r["total_return_pct"] for r in results])
    sharpes = np.array([r["sharpe"] for r in results])
    dds = np.array([r["max_dd_pct"] for r in results])
    alphas = np.array([r["alpha_pct"] for r in results])

    profit_folds = int((returns > 0).sum())
    alpha_folds = int((alphas > 0).sum())

    print(f"\n  Mean return:   {returns.mean():+7.2f}%   (std {returns.std():5.2f}, "
          f"min {returns.min():+.2f}, max {returns.max():+.2f})")
    print(f"  Mean sharpe:   {sharpes.mean():7.2f}    (std {sharpes.std():5.2f})")
    print(f"  Mean DD:       {dds.mean():7.2f}%   (worst: {dds.max():.2f}%)")
    print(f"  Mean alpha:    {alphas.mean():+7.2f}%")
    print(f"  Profitable folds:        {profit_folds}/{n_folds}  ({profit_folds/n_folds:.0%})")
    print(f"  Folds beating buy-hold:  {alpha_folds}/{n_folds}  ({alpha_folds/n_folds:.0%})")

    # ---------- Diagnosis ----------
    print("\n" + "=" * 70)
    print(" DIAGNOSIS")
    print("=" * 70)

    consistency = profit_folds / n_folds
    alpha_score = alpha_folds / n_folds
    sharpe_consistency = (sharpes.mean() / (sharpes.std() + 1e-9)) if len(sharpes) > 1 else 0

    if consistency >= 0.75 and alpha_score >= 0.6 and sharpes.mean() > 1.0:
        verdict = "ROBUST"
        emoji = "G"
        msg = "model ทำงานได้ดี consistent ในหลาย period"
        next_step = "ทำ Stress test (period ตลาดยาก) เป็น step ถัดไป"
    elif consistency >= 0.5 and sharpes.mean() > 0.5:
        verdict = "PARTIALLY ROBUST"
        emoji = "Y"
        msg = "บอทกำไรครึ่ง — ขึ้นกับสภาพตลาด"
        next_step = "ดู fold ที่ขาดทุน -> ระบุว่าตลาดแบบไหนที่บอทแย่"
    else:
        verdict = "NOT ROBUST"
        emoji = "R"
        msg = "กำไรไม่ consistent — ผลก่อนหน้าออก lucky"
        next_step = "ทบทวน reward function / features / regularization"

    print(f"  [{emoji}] {verdict}")
    print(f"  {msg}")
    print(f"  Sharpe consistency (mean/std): {sharpe_consistency:.2f}  (>1.5 = stable)")
    print(f"\n  Next step: {next_step}")

    # ---------- Save ----------
    out_dir = Path("./walk_forward_output")
    out_dir.mkdir(exist_ok=True)

    df_results = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ("equity_curve", "action_distribution")}
        for r in results
    ])
    csv_name = f"fold_results_ensemble_n{args.n_seeds}_{args.method}.csv"
    df_results.to_csv(out_dir / csv_name, index=False)

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(13, 8),
                                 gridspec_kw={"height_ratios": [3, 1]})
        ax1, ax2 = axes

        cmap = plt.get_cmap("viridis")
        for idx, r in enumerate(results):
            eq = r["equity_curve"]
            normalized = eq / eq[0] * 100
            color = cmap(idx / max(n_folds - 1, 1))
            ax1.plot(normalized, alpha=0.85, color=color,
                     label=f"Fold {r['fold']} ({r['total_return_pct']:+.1f}%)")

        ax1.axhline(100, color="gray", linestyle="--", alpha=0.5, label="Initial $100")
        ax1.set_xlabel("Step within test window")
        ax1.set_ylabel("Equity (normalized)")
        ax1.set_title(f"Ensemble Walk-Forward ({args.n_seeds} seeds) | "
                      f"Mean return: {returns.mean():+.2f}% | "
                      f"Profit rate: {profit_folds}/{n_folds}")
        ax1.legend(loc="best", fontsize=9)
        ax1.grid(True, alpha=0.3)

        x = np.arange(1, n_folds + 1)
        colors = ["green" if r > 0 else "red" for r in returns]
        ax2.bar(x, returns, color=colors, alpha=0.7)
        bnh = np.array([r["buy_and_hold_pct"] for r in results])
        ax2.plot(x, bnh, "ko-", label="Buy-and-Hold (0.01 lot)", markersize=6)
        ax2.axhline(0, color="black", linewidth=0.5)
        ax2.set_xlabel("Fold")
        ax2.set_ylabel("Return (%)")
        ax2.set_xticks(x)
        ax2.legend(loc="best", fontsize=9)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_dir / f"walk_forward_equity_ensemble_n{args.n_seeds}_{args.method}.png", dpi=120)
        print(f"\n  Saved: {out_dir / f'walk_forward_equity_ensemble_n{args.n_seeds}.png'}")
    except ImportError:
        pass

    print(f"  Saved: {out_dir / csv_name}")
    print(f"\n  Total time: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
