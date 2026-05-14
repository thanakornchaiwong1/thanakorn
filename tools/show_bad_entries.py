"""
Show exactly where the bot enters wrong — trade-by-trade analysis.
Simulates every valid setup bar (action mask = True) and tracks what happens.
Outputs CSV + prints worst trades so you can look them up on TradingView.
"""
import sys, os, io
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
import yaml
from src.data import load_data
from src.env import GoldTradingEnv

with open("config_scalp.yaml") as f:
    cfg = yaml.safe_load(f)

df = load_data(cfg)
atr = (df["atr_ratio"] * df["close"]).values
highs = df["high"].values
lows  = df["low"].values
closes = df["close"].values

SL_M = cfg["env"]["sl_atr_mult"]
TP_M = cfg["env"]["tp_atr_mult"]
SPREAD = cfg["env"]["spread"]

# ── Analyze these folds ──────────────────────────────────────────────
FOLDS = {
    "Fold2_BAD":  (70000,  80000),
    "Fold7_BEST": (170000, 180000),
}

def simulate_trade(entry_idx, direction, max_bars=500):
    """Simulate one trade from entry_idx. Returns trade dict."""
    ep = closes[entry_idx] + direction * SPREAD / 2
    a  = atr[entry_idx]
    if a <= 0 or np.isnan(a):
        return None

    sl = ep - direction * SL_M * a
    tp = ep + direction * TP_M * a

    mfe = mae = 0.0
    result = "TIMEOUT"
    exit_bar = entry_idx
    first_bar_ok = None

    for j in range(entry_idx + 1, min(entry_idx + max_bars, len(df) - 1)):
        h, l = highs[j], lows[j]

        fav = (h - ep) * direction if direction == 1 else (ep - l)
        adv = (ep - l) * direction if direction == 1 else (h - ep)
        mfe = max(mfe, fav)
        mae = max(mae, adv)

        if first_bar_ok is None:
            mid = (h + l) / 2
            first_bar_ok = (mid > ep) if direction == 1 else (mid < ep)

        hit_tp = (h >= tp) if direction == 1 else (l <= tp)
        hit_sl = (l <= sl) if direction == 1 else (h >= sl)

        if hit_tp or hit_sl:
            if hit_tp and hit_sl:
                result = "SL"        # conservative: assume SL hit first
            elif hit_tp:
                result = "TP"
            else:
                result = "SL"
            exit_bar = j
            break

    bars_held = exit_bar - entry_idx
    tp_pct = mfe / (TP_M * atr[entry_idx]) * 100 if atr[entry_idx] > 0 else 0

    return {
        "entry_row":   entry_idx,
        "direction":   "LONG" if direction == 1 else "SHORT",
        "entry_price": round(ep, 2),
        "sl_price":    round(sl, 2),
        "tp_price":    round(tp, 2),
        "atr":         round(a, 2),
        "result":      result,
        "bars_held":   bars_held,
        "mfe_atr":     round(mfe / a, 2) if a > 0 else 0,
        "mae_atr":     round(mae / a, 2) if a > 0 else 0,
        "tp_progress%": round(tp_pct, 1),
        "first_bar_ok": first_bar_ok,
        "noise_stop":  result == "SL" and bars_held <= 5,
    }


for fold_name, (start, end) in FOLDS.items():
    print(f"\n{'='*70}")
    print(f"  {fold_name} | rows {start}-{end}")
    print(f"{'='*70}")

    # Build env to get action masks
    seg = df.iloc[start:end].reset_index(drop=True)
    env = GoldTradingEnv(seg, **cfg["env"])

    trades = []
    for i in range(env.window_size, len(seg) - 501):
        env.current_step = i
        env.position = 0
        mask = env.action_masks()

        if not mask[1]:
            continue   # no valid entry here

        abs_i = start + i
        trend_dir = env._get_trend_direction()
        actual_dir = trend_dir if trend_dir != 0 else 1

        t = simulate_trade(abs_i, actual_dir)
        if t:
            trades.append(t)

    if not trades:
        print("  No valid setup bars found.")
        continue

    tdf = pd.DataFrame(trades)
    n = len(tdf)
    tp = tdf[tdf["result"] == "TP"]
    sl = tdf[tdf["result"] == "SL"]
    wr = len(tp) / n * 100

    print(f"  Total setups:  {n}")
    print(f"  TP / SL:       {len(tp)} / {len(sl)}")
    print(f"  Win Rate:      {wr:.1f}%  (breakeven = 25%)")
    print(f"  Avg bars/TP:   {tp['bars_held'].mean():.0f}" if len(tp) else "  No TP trades")
    print(f"  Avg bars/SL:   {sl['bars_held'].mean():.0f}" if len(sl) else "")

    # SL analysis
    if len(sl):
        noise = sl[sl["noise_stop"]]
        wrong_dir = sl[sl["first_bar_ok"] == False]
        almost_tp = sl[sl["tp_progress%"] > 50]
        print(f"\n  --- Why SL trades lost ---")
        print(f"  Noise stops (SL <5 bars):  {len(noise)}/{len(sl)} ({len(noise)/len(sl)*100:.0f}%)")
        print(f"  Wrong direction bar 1:     {len(wrong_dir)}/{len(sl)} ({len(wrong_dir)/len(sl)*100:.0f}%)")
        print(f"  Got >50% to TP then died:  {len(almost_tp)}/{len(sl)} ({len(almost_tp)/len(sl)*100:.0f}%)")
        print(f"  Avg MAE (ATR):             {sl['mae_atr'].mean():.2f}  (SL at {SL_M})")
        print(f"  Avg MFE before SL (ATR):   {sl['mfe_atr'].mean():.2f}")

    # ── Print 10 worst SL trades ─────────────────────────────────────
    print(f"\n  --- 10 Worst SL trades (look these up on TradingView) ---")
    print(f"  {'Row':>7}  {'Dir':>6}  {'Entry':>10}  {'SL':>10}  {'TP':>10}  "
          f"{'Bars':>5}  {'MFE':>6}  {'MAE':>6}  {'TP%':>6}  Noise? WrongDir?")

    worst = sl.nsmallest(10, "tp_progress%") if len(sl) >= 10 else sl
    for _, r in worst.iterrows():
        noise_flag = "YES" if r["noise_stop"] else ""
        dir_flag   = "YES" if r["first_bar_ok"] == False else ""
        print(f"  {int(r['entry_row']):>7}  {r['direction']:>6}  "
              f"{r['entry_price']:>10.2f}  {r['sl_price']:>10.2f}  {r['tp_price']:>10.2f}  "
              f"{int(r['bars_held']):>5}  {r['mfe_atr']:>6.2f}  {r['mae_atr']:>6.2f}  "
              f"{r['tp_progress%']:>5.0f}%  {noise_flag:>5}  {dir_flag}")

    # ── Print 5 best TP trades for comparison ────────────────────────
    if len(tp):
        print(f"\n  --- 5 Best TP trades (what good entries look like) ---")
        print(f"  {'Row':>7}  {'Dir':>6}  {'Entry':>10}  {'Bars':>5}  {'MFE':>6}  {'MAE':>6}")
        best = tp.nsmallest(5, "mae_atr")
        for _, r in best.iterrows():
            print(f"  {int(r['entry_row']):>7}  {r['direction']:>6}  "
                  f"{r['entry_price']:>10.2f}  {int(r['bars_held']):>5}  "
                  f"{r['mfe_atr']:>6.2f}  {r['mae_atr']:>6.2f}")

    # Save CSV
    out = f"walk_forward_output/{fold_name}_trades.csv"
    tdf.to_csv(out, index=False)
    print(f"\n  Saved: {out} ({n} rows)")

print(f"\n{'='*70}")
print("  HOW TO USE:")
print("  1. Open TradingView -> Gold/XAUUSD M15")
print("  2. Look up entry_row number in the CSV")
print("     Row 70000 = approx: (70000 / 96) = 729 trading days from 27 Sep 2017")
print("     = roughly Jul 2020")
print("  3. Find that date/price and see if the entry makes sense visually")
print(f"{'='*70}")
