"""
Walk-forward validation for PPO gold trading bot.

แบ่งข้อมูลเป็น rolling windows:
  Fold 1: train [0:6000]   test [6000:7500]
  Fold 2: train [1500:7500] test [7500:9000]
  ...

แต่ละ fold train fresh model จาก scratch แล้ว backtest บน test window
สรุปผลข้าม folds เพื่อดูว่า bot consistent หรือเก่งแค่บางช่วง

Usage:
    python -m tools.walk_forward                          # default 200k steps/fold
    python -m tools.walk_forward --quick                  # 50k steps/fold (test pipeline)
    python -m tools.walk_forward --timesteps 300000       # custom
    python -m tools.walk_forward --train-size 8000 --test-size 1500
"""
from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Force UTF-8 for stdout (Windows Thai cp874 ไม่รองรับ unicode เช่น ±, →)
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.backtest import (
    annualization_factor, max_drawdown, profit_factor, sharpe_ratio,
)
from src.data import load_data
from src.env import GoldTradingEnv

# LSTM support (optional)
try:
    from sb3_contrib import RecurrentPPO
    HAS_RECURRENT = True
except ImportError:
    HAS_RECURRENT = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_env_fn(df, env_kwargs, seed, rank=0):
    def _init():
        env = GoldTradingEnv(df, **env_kwargs)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def train_fold(train_df, cfg, timesteps, seed):
    """Train fresh PPO/RecurrentPPO on train_df. Return (model, vec_normalize, use_lstm)"""
    env_kwargs = cfg["env"]
    n_envs = cfg["train"]["n_envs"]
    use_vn = cfg["train"]["use_vec_normalize"]
    use_lstm = env_kwargs.get("use_lstm", False)

    train_env = DummyVecEnv([
        make_env_fn(train_df, env_kwargs, seed, i) for i in range(n_envs)
    ])
    if use_vn:
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    ppo_cfg = cfg["ppo"]

    if use_lstm:
        # RecurrentPPO with LSTM
        if not HAS_RECURRENT:
            raise ImportError("sb3-contrib required for LSTM. pip install sb3-contrib")

        lstm_hidden = ppo_cfg.get("lstm_hidden_size", 128)
        policy_kwargs = dict(
            net_arch=dict(pi=ppo_cfg["net_arch"]["pi"], vf=ppo_cfg["net_arch"]["vf"]),
            lstm_hidden_size=lstm_hidden,
            n_lstm_layers=1,
        )

        model = RecurrentPPO(
            policy=ppo_cfg["policy"],  # "MlpLstmPolicy"
            env=train_env,
            learning_rate=ppo_cfg["learning_rate"],
            n_steps=ppo_cfg["n_steps"],
            batch_size=ppo_cfg["batch_size"],
            n_epochs=ppo_cfg["n_epochs"],
            gamma=ppo_cfg["gamma"],
            gae_lambda=ppo_cfg["gae_lambda"],
            clip_range=ppo_cfg["clip_range"],
            ent_coef=ppo_cfg["ent_coef"],
            vf_coef=ppo_cfg["vf_coef"],
            max_grad_norm=ppo_cfg["max_grad_norm"],
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=seed,
            device="cpu",  # RecurrentPPO better on CPU for MLP+LSTM
        )
    else:
        # Standard PPO with MLP
        policy_kwargs = dict(
            net_arch=dict(pi=ppo_cfg["net_arch"]["pi"], vf=ppo_cfg["net_arch"]["vf"]),
        )

        model = PPO(
            policy=ppo_cfg["policy"],
            env=train_env,
            learning_rate=ppo_cfg["learning_rate"],
            n_steps=ppo_cfg["n_steps"],
            batch_size=ppo_cfg["batch_size"],
            n_epochs=ppo_cfg["n_epochs"],
            gamma=ppo_cfg["gamma"],
            gae_lambda=ppo_cfg["gae_lambda"],
            clip_range=ppo_cfg["clip_range"],
            ent_coef=ppo_cfg["ent_coef"],
            vf_coef=ppo_cfg["vf_coef"],
            max_grad_norm=ppo_cfg["max_grad_norm"],
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=seed,
            device="auto",
        )

    model.learn(total_timesteps=timesteps, progress_bar=True)

    return model, (train_env if use_vn else None), use_lstm


def buy_and_hold_equity_return(test_df, env_kwargs):
    """
    APPLES-TO-APPLES B&H: ซื้อ lot_size เดียวกับบอท ที่ candle แรก ถือถึงสุดท้าย
    คืน % equity return บนบัญชี (เทียบกับบอทได้ตรง ๆ)

    เดิมใน walk_forward เก่าใช้ raw price % change (price_end/price_start - 1)*100
    ซึ่งสมมติ leverage 100% notional — ไม่ใช่สิ่งที่บอททำ
    บอทเทรด 0.01 lot × 100 contract = $1 ต่อ $1 price move = ~0.24x leverage บน $10k
    """
    initial = env_kwargs["initial_balance"]
    lot = env_kwargs["lot_size"]
    contract = env_kwargs["contract_size"]
    spread = env_kwargs["spread"]
    commission = env_kwargs["commission"]

    entry_price = float(test_df["close"].iloc[0]) + spread / 2.0
    exit_price = float(test_df["close"].iloc[-1]) - spread / 2.0
    pnl = (exit_price - entry_price) * lot * contract
    final_equity = initial - commission + pnl  # commission ตอน open (mark-to-market end)
    return (final_equity / initial - 1) * 100


def backtest_fold(test_df, cfg, model, vn, use_lstm=False):
    """Run trained model on test_df. Return metrics dict + equity curve."""
    env_kwargs = cfg["env"]
    raw_env = GoldTradingEnv(test_df, **env_kwargs)

    obs, _ = raw_env.reset()

    # LSTM state tracking
    lstm_state = None
    episode_start = np.ones((1,), dtype=bool)

    while True:
        if vn is not None:
            norm_obs = vn.normalize_obs(obs[None, :])[0]
        else:
            norm_obs = obs

        if use_lstm:
            action, lstm_state = model.predict(
                norm_obs, state=lstm_state,
                episode_start=episode_start, deterministic=True
            )
            episode_start = np.zeros((1,), dtype=bool)
        else:
            action, _ = model.predict(norm_obs, deterministic=True)

        action = int(np.asarray(action).flatten()[0])
        obs, _, term, trunc, _ = raw_env.step(action)
        if term or trunc:
            break

    equity = np.array(raw_env.equity_curve, dtype=np.float64)
    ann = annualization_factor(cfg["data"]["interval"])
    bot_return = (equity[-1] / equity[0] - 1) * 100

    # Equity-based B&H (apples-to-apples with bot's account return)
    bnh_return = buy_and_hold_equity_return(test_df, env_kwargs)
    # Old raw-price metric retained for context (NOT comparable to bot return)
    bnh_raw_return = (test_df["close"].iloc[-1] / test_df["close"].iloc[0] - 1) * 100

    return {
        "total_return_pct": bot_return,
        "total_trades": raw_env.total_trades,
        "win_rate_pct": raw_env.winning_trades / max(raw_env.total_trades, 1) * 100,
        "max_dd_pct": max_drawdown(equity) * 100,
        "sharpe": sharpe_ratio(equity, ann),
        "profit_factor": profit_factor(raw_env.trade_returns),
        "buy_and_hold_pct": bnh_return,           # apples-to-apples
        "buy_and_hold_raw_pct": bnh_raw_return,   # raw price % (legacy, for ref)
        "alpha_pct": bot_return - bnh_return,
        "equity_curve": equity,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--train-size", type=int, default=6000)
    parser.add_argument("--test-size", type=int, default=1500)
    parser.add_argument("--step-size", type=int, default=1500)
    parser.add_argument("--timesteps", type=int, default=200000)
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 50k steps/fold (~3 นาที/fold)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.quick:
        args.timesteps = 50_000

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print("=" * 70)
    print(" WALK-FORWARD VALIDATION")
    print("=" * 70)
    print(f"  train window: {args.train_size:,} rows")
    print(f"  test window:  {args.test_size:,} rows")
    print(f"  step size:    {args.step_size:,} rows")
    print(f"  timesteps/fold: {args.timesteps:,}")
    print(f"  mode: {'QUICK (smoke test)' if args.quick else 'FULL'}")

    print("\nLoading data...")
    df = load_data(cfg)
    print(f"  total rows: {len(df):,}")

    # ---------- Build folds ----------
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
    print(f"  number of folds: {n_folds}")

    if n_folds == 0:
        print("\nERROR: ข้อมูลไม่พอสำหรับ walk-forward")
        print("       ลด --train-size หรือ --test-size หรือเพิ่ม period ใน config")
        return

    if n_folds < 3:
        print(f"\nWARNING: มีแค่ {n_folds} folds — น้อยไปสำหรับ statistical confidence")
        print("         พิจารณาลด --step-size ลงครึ่ง")

    # Estimate time
    est_fps = 280
    est_secs_per_fold = args.timesteps / est_fps
    est_total_min = (est_secs_per_fold * n_folds) / 60
    print(f"  est. time: ~{est_total_min:.0f} นาที (อาจเร็ว/ช้ากว่านี้ ±20%)")

    # ---------- Run all folds ----------
    results = []
    t_start = time.time()

    for i, (train_df, test_df, start_idx) in enumerate(folds):
        elapsed = (time.time() - t_start) / 60
        eta = elapsed / max(i, 1) * (n_folds - i) if i > 0 else est_total_min

        print(f"\n{'─' * 70}")
        print(f" Fold {i+1}/{n_folds} | rows {start_idx}-{start_idx + args.train_size + args.test_size}"
              f" | elapsed {elapsed:.1f}m | ETA {eta:.0f}m")
        print(f"{'─' * 70}")

        print(f" [train] fold {i+1}...")
        model, vn, use_lstm = train_fold(train_df, cfg, args.timesteps, args.seed + i * 100)

        print(f" [test]  fold {i+1}...")
        m = backtest_fold(test_df, cfg, model, vn, use_lstm=use_lstm)
        m["fold"] = i + 1
        m["start_row"] = start_idx
        results.append(m)

        print(f"   → return={m['total_return_pct']:+6.2f}%  "
              f"sharpe={m['sharpe']:5.2f}  "
              f"DD={m['max_dd_pct']:5.2f}%  "
              f"trades={m['total_trades']:4d}  "
              f"alpha={m['alpha_pct']:+6.2f}%")

        del model, vn  # free memory

    # ---------- Summary ----------
    print("\n" + "=" * 70)
    print(" WALK-FORWARD RESULTS")
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
    print(f"  Mean DD:       {dds.mean():7.2f}%   (worst across folds: {dds.max():.2f}%)")
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
        verdict = "🟢 ROBUST"
        msg = "model ทำงานได้ดี consistent ในหลาย period — สัญญาณน่าสนใจ"
        next_step = "ทำ Stress test (period ตลาดยาก ๆ) เป็น step ถัดไป"
    elif consistency >= 0.5 and sharpes.mean() > 0.5:
        verdict = "🟡 PARTIALLY ROBUST"
        msg = "บอทกำไรครึ่ง ๆ — ขึ้นกับสภาพตลาด ไม่ universal"
        next_step = "ดู fold ที่ขาดทุน → ระบุว่าตลาดแบบไหนที่บอทแย่"
    else:
        verdict = "🔴 NOT ROBUST"
        msg = "กำไรไม่ consistent — ผลก่อนหน้านี้ออก lucky"
        next_step = "ทบทวน reward function / features / regularization ก่อนไปต่อ"

    print(f"  {verdict}")
    print(f"  {msg}")
    print(f"  Sharpe consistency (mean/std): {sharpe_consistency:.2f}  (>1.5 = stable)")
    print(f"\n  Next step: {next_step}")

    # ---------- Save ----------
    out_dir = Path("./walk_forward_output")
    out_dir.mkdir(exist_ok=True)

    df_results = pd.DataFrame([
        {k: v for k, v in r.items() if k != "equity_curve"}
        for r in results
    ])
    df_results.to_csv(out_dir / "fold_results.csv", index=False)

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(13, 8),
                                 gridspec_kw={"height_ratios": [3, 1]})
        ax1, ax2 = axes

        cmap = plt.get_cmap("viridis")
        for i, r in enumerate(results):
            eq = r["equity_curve"]
            normalized = eq / eq[0] * 100
            color = cmap(i / max(n_folds - 1, 1))
            ax1.plot(normalized, alpha=0.85, color=color,
                     label=f"Fold {r['fold']} ({r['total_return_pct']:+.1f}%)")

        ax1.axhline(100, color="gray", linestyle="--", alpha=0.5, label="Initial $100")
        ax1.set_xlabel("Step within test window")
        ax1.set_ylabel("Equity (normalized)")
        ax1.set_title(f"Walk-Forward Equity Curves — {n_folds} folds | "
                      f"Mean return: {returns.mean():+.2f}% | "
                      f"Profit rate: {profit_folds}/{n_folds}")
        ax1.legend(loc="best", fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Bar chart: per-fold returns
        x = np.arange(1, n_folds + 1)
        colors = ["green" if r > 0 else "red" for r in returns]
        ax2.bar(x, returns, color=colors, alpha=0.7)
        bnh = np.array([r["buy_and_hold_pct"] for r in results])
        ax2.plot(x, bnh, "ko-", label="Buy-and-Hold", markersize=6)
        ax2.axhline(0, color="black", linewidth=0.5)
        ax2.set_xlabel("Fold")
        ax2.set_ylabel("Return (%)")
        ax2.set_xticks(x)
        ax2.legend(loc="best", fontsize=9)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_dir / "walk_forward_equity.png", dpi=120)
        print(f"\n  Saved: {out_dir / 'walk_forward_equity.png'}")
    except ImportError:
        pass

    print(f"  Saved: {out_dir / 'fold_results.csv'}")
    print(f"\n  Total time: {(time.time() - t_start) / 60:.1f} นาที")


if __name__ == "__main__":
    main()
