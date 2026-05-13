"""
Bot behavior inspector — เปิดกล่องดำดูว่าบอทจริง ๆ ทำอะไรในแต่ละ fold

ตอบคำถามสำคัญ:
  1. Long-bias check: บอทใช้เวลา long/short/flat กี่ %?
  2. Direction-vs-market: บอทเปิด long ตอนตลาดขึ้นหรือลง? (short into rallies = bad)
  3. Trade outcome by direction: long วินกี่ %? short วินกี่ %?
  4. Hold duration: winners ถูกตัดเร็วเกิน? losers ถือนานเกิน?
  5. Critical losing trades: trades ที่ขาดทุนหนัก เกิดตอนตลาดไปทางไหน?

Usage:
    python -m tools.inspect_actions
    python -m tools.inspect_actions --fold 1            # รันแค่ fold เดียว
    python -m tools.inspect_actions --quick             # 50k steps (smoke test)
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

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from src.data import load_data
from src.env import GoldTradingEnv
from tools.walk_forward import train_fold


def backtest_with_log(test_df, cfg, model, vn):
    """Backtest + log every step. คืน step_log (DataFrame) + trade_log (DataFrame)."""
    env_kwargs = cfg["env"]
    raw_env = GoldTradingEnv(test_df, **env_kwargs)

    closes = test_df["close"].values
    step_records = []
    trade_records = []

    obs, _ = raw_env.reset()
    open_trade: dict | None = None  # track currently-open trade

    while True:
        step_idx = raw_env.current_step
        pos_before = raw_env.position
        price = float(closes[step_idx])

        if vn is not None:
            norm_obs = vn.normalize_obs(obs[None, :])[0]
        else:
            norm_obs = obs
        action, _ = model.predict(norm_obs, deterministic=True)
        action = int(np.asarray(action).flatten()[0])

        # log BEFORE step (state at decision time)
        step_records.append({
            "step": step_idx,
            "price": price,
            "position_before": pos_before,
            "action": action,
        })

        obs, _, term, trunc, _ = raw_env.step(action)

        # detect trade open
        pos_after = raw_env.position
        if pos_before == 0 and pos_after != 0:
            open_trade = {
                "entry_step": step_idx,
                "entry_price": raw_env.entry_price,
                "direction": pos_after,  # +1 long, -1 short
            }

        # detect trade close
        if pos_before != 0 and pos_after == 0 and open_trade is not None:
            # close exit price (from env logic)
            if open_trade["direction"] == 1:
                exit_price = price - cfg["env"]["spread"] / 2.0
            else:
                exit_price = price + cfg["env"]["spread"] / 2.0
            duration = step_idx - open_trade["entry_step"]
            pnl_pct = (exit_price - open_trade["entry_price"]) / open_trade["entry_price"] \
                      * open_trade["direction"]
            trade_records.append({
                "entry_step": open_trade["entry_step"],
                "exit_step": step_idx,
                "duration": duration,
                "direction": open_trade["direction"],
                "entry_price": open_trade["entry_price"],
                "exit_price": exit_price,
                "pnl_pct": pnl_pct * 100,
                "win": pnl_pct > 0,
            })
            open_trade = None

        if term or trunc:
            break

    step_log = pd.DataFrame(step_records)
    trade_log = pd.DataFrame(trade_records)
    return step_log, trade_log, raw_env.equity_curve


def market_direction_at(test_df, step, lookback=20):
    """% change ใน lookback bars ที่ผ่านมา. + = ขึ้น, - = ลง"""
    if step - lookback < 0:
        lookback = step
    if lookback < 1:
        return 0.0
    p_now = float(test_df["close"].iloc[step])
    p_then = float(test_df["close"].iloc[step - lookback])
    return (p_now - p_then) / p_then * 100


def market_direction_forward(test_df, step, lookahead=20):
    """% change ใน lookahead bars ข้างหน้า (ดูว่า trade เปิดถูกทางไหม)"""
    end_step = min(step + lookahead, len(test_df) - 1)
    p_now = float(test_df["close"].iloc[step])
    p_future = float(test_df["close"].iloc[end_step])
    return (p_future - p_now) / p_now * 100


def analyze_fold(fold_idx, train_df, test_df, cfg, timesteps, seed):
    """Train + inspect one fold. Print analysis."""
    print(f"\n{'═' * 70}")
    print(f" FOLD {fold_idx} ANALYSIS")
    print(f"{'═' * 70}")

    print(f" [train] {timesteps:,} steps...")
    t0 = time.time()
    model, vn = train_fold(train_df, cfg, timesteps, seed)
    print(f"   training done ({(time.time()-t0)/60:.1f} min)")

    print(f" [test+log] backtesting...")
    step_log, trade_log, equity = backtest_with_log(test_df, cfg, model, vn)

    # ---- High-level fold metrics ----
    bot_ret = (equity[-1] / equity[0] - 1) * 100
    bnh_ret = (test_df["close"].iloc[-1] / test_df["close"].iloc[0] - 1) * 100
    print(f"\n  Bot return: {bot_ret:+.2f}%   B&H: {bnh_ret:+.2f}%   "
          f"Alpha: {bot_ret - bnh_ret:+.2f}%   Trades: {len(trade_log)}")

    # ---- Q1: Long/Short/Flat time distribution ----
    n_steps = len(step_log)
    pos_counts = step_log["position_before"].value_counts().to_dict()
    pct_long = pos_counts.get(1, 0) / n_steps * 100
    pct_short = pos_counts.get(-1, 0) / n_steps * 100
    pct_flat = pos_counts.get(0, 0) / n_steps * 100

    print(f"\n  ── Q1: TIME DISTRIBUTION ──")
    print(f"    Long:  {pct_long:5.1f}%    Short: {pct_short:5.1f}%    Flat:  {pct_flat:5.1f}%")
    if pct_long + pct_short < 30:
        print(f"    ⚠ บอทอยู่ flat {pct_flat:.0f}% — เทรดน้อยเกินไป")
    if pct_short > pct_long * 1.5:
        print(f"    ⚠ SHORT-BIASED (short {pct_short:.0f}% vs long {pct_long:.0f}%)")
    elif pct_long > pct_short * 1.5:
        print(f"    ⚠ LONG-BIASED (long {pct_long:.0f}% vs short {pct_short:.0f}%)")
    else:
        print(f"    บอท long/short สมดุล")

    # ---- Q2: Trade direction vs market direction ----
    if len(trade_log) > 0:
        trade_log["market_20bar_before"] = trade_log["entry_step"].apply(
            lambda s: market_direction_at(test_df, s, 20)
        )
        trade_log["market_20bar_forward"] = trade_log["entry_step"].apply(
            lambda s: market_direction_forward(test_df, s, 20)
        )

        n_long = (trade_log["direction"] == 1).sum()
        n_short = (trade_log["direction"] == -1).sum()
        print(f"\n  ── Q2: TRADE COUNT BY DIRECTION ──")
        print(f"    Long entries:  {n_long}    Short entries: {n_short}")

        # short entries during uptrending markets = bad
        short_into_up = trade_log[
            (trade_log["direction"] == -1) & (trade_log["market_20bar_before"] > 1.0)
        ]
        long_into_down = trade_log[
            (trade_log["direction"] == 1) & (trade_log["market_20bar_before"] < -1.0)
        ]
        print(f"    Short into uptrend (last20bar +>1%):  {len(short_into_up)}/{n_short}")
        print(f"    Long  into downtrend (last20bar -<1%): {len(long_into_down)}/{n_long}")

        # ---- Q3: Win rate by direction ----
        print(f"\n  ── Q3: WIN RATE & AVG PnL BY DIRECTION ──")
        for dir_val, dir_name in [(1, "Long "), (-1, "Short")]:
            sub = trade_log[trade_log["direction"] == dir_val]
            if len(sub) > 0:
                wr = sub["win"].mean() * 100
                avg_win = sub[sub["win"]]["pnl_pct"].mean() if sub["win"].any() else 0
                avg_loss = sub[~sub["win"]]["pnl_pct"].mean() if (~sub["win"]).any() else 0
                avg_dur = sub["duration"].mean()
                print(f"    {dir_name}: n={len(sub):3d}  WR={wr:5.1f}%  "
                      f"avgWin={avg_win:+5.2f}%  avgLoss={avg_loss:+5.2f}%  "
                      f"avgDur={avg_dur:.0f} bars")

        # ---- Q4: Duration: winners vs losers ----
        print(f"\n  ── Q4: HOLD DURATION (winners vs losers) ──")
        winners = trade_log[trade_log["win"]]
        losers = trade_log[~trade_log["win"]]
        if len(winners) > 0 and len(losers) > 0:
            print(f"    Winners: avg={winners['duration'].mean():.1f} bars  "
                  f"median={winners['duration'].median():.0f}  "
                  f"max={winners['duration'].max()}")
            print(f"    Losers:  avg={losers['duration'].mean():.1f} bars  "
                  f"median={losers['duration'].median():.0f}  "
                  f"max={losers['duration'].max()}")
            if winners["duration"].mean() < losers["duration"].mean() * 0.7:
                print(f"    ⚠ CUTTING WINNERS SHORT (winners avg < 70% of losers avg)")
            if losers["duration"].mean() > 30 and losers["duration"].mean() > winners["duration"].mean() * 1.5:
                print(f"    ⚠ HOLDING LOSERS TOO LONG (bag-holding pattern)")

        # ---- Q5: Worst trades ----
        print(f"\n  ── Q5: WORST 5 LOSING TRADES ──")
        worst = trade_log.nsmallest(5, "pnl_pct")
        print(f"    {'Dir':<5} {'Entry':<6} {'Exit':<6} {'Dur':<5} "
              f"{'PnL%':<8} {'Mkt20before':<12} {'Mkt20after':<12}")
        for _, t in worst.iterrows():
            d = "Long" if t["direction"] == 1 else "Short"
            print(f"    {d:<5} {int(t['entry_step']):<6} {int(t['exit_step']):<6} "
                  f"{int(t['duration']):<5} {t['pnl_pct']:<+7.2f}% "
                  f"{t['market_20bar_before']:<+11.2f}% {t['market_20bar_forward']:<+11.2f}%")

    # ---- Save artifacts ----
    out_dir = Path("./inspect_output")
    out_dir.mkdir(exist_ok=True)
    step_log.to_csv(out_dir / f"fold{fold_idx}_steps.csv", index=False)
    if len(trade_log) > 0:
        trade_log.to_csv(out_dir / f"fold{fold_idx}_trades.csv", index=False)
    print(f"\n  Saved: inspect_output/fold{fold_idx}_steps.csv, fold{fold_idx}_trades.csv")

    return {"bot_ret": bot_ret, "bnh_ret": bnh_ret, "n_trades": len(trade_log)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--train-size", type=int, default=1500)
    parser.add_argument("--test-size", type=int, default=400)
    parser.add_argument("--step-size", type=int, default=400)
    parser.add_argument("--timesteps", type=int, default=200000)
    parser.add_argument("--quick", action="store_true",
                        help="Quick: 50k steps (smoke test, ~3 min/fold)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0,
                        help="0=all folds, 1/2/3=only that fold")
    args = parser.parse_args()

    if args.quick:
        args.timesteps = 50_000

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print("=" * 70)
    print(" BOT BEHAVIOR INSPECTOR")
    print("=" * 70)
    print(f"  config: {args.config}  reward={cfg['env']['reward_type']}  "
          f"window={cfg['env']['window_size']}  multi_tf={cfg['data']['use_multi_timeframe']}")
    print(f"  fold params: train={args.train_size} test={args.test_size} step={args.step_size}")
    print(f"  timesteps/fold: {args.timesteps:,}")

    print("\nLoading data...")
    df = load_data(cfg)
    print(f"  total rows: {len(df):,}")

    folds = []
    s = 0
    while s + args.train_size + args.test_size <= len(df):
        folds.append((s, s + args.train_size, s + args.train_size + args.test_size))
        s += args.step_size
    print(f"  folds: {len(folds)}")

    target_folds = [args.fold - 1] if args.fold > 0 else list(range(len(folds)))

    t_start = time.time()
    for i in target_folds:
        if i >= len(folds):
            print(f"\nFold {i+1} ไม่มี (มีแค่ {len(folds)} folds)")
            continue
        a, b, c = folds[i]
        train_df = df.iloc[a:b].reset_index(drop=True)
        test_df = df.iloc[b:c].reset_index(drop=True)
        analyze_fold(i + 1, train_df, test_df, cfg, args.timesteps, args.seed + i * 100)

    print(f"\n{'=' * 70}")
    print(f" Total time: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
