"""
Deep Trade-Level Root Cause Analysis
=====================================
Simulates the EXACT TP/SL logic from env.py on raw M15 data to understand:
1. Direction accuracy — did price go the right way after entry?
2. MFE/MAE — how far did each trade get toward TP vs SL?
3. SL tightness — are stops being clipped by noise before the move?
4. Fold-level behavior — does gold behavior change over time?
5. LONG vs SHORT — any directional bias?
6. Time-of-day effects — which sessions produce good trades?

Uses a SYSTEMATIC strategy (enter every N bars based on trend) to isolate
market behavior from agent quality — this shows what's achievable.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from src.env import prepare_features, compute_daily_features, merge_daily_into_primary

# ── Config ──────────────────────────────────────────────────────────
CSV_M15 = "./data/xauusd_m15.csv"
CSV_D1  = "./data/xauusd_d1.csv"

SL_ATR_MULT = 1.5
TP_ATR_MULT = 4.5
SPREAD = 0.16
COMMISSION = 0.07
LOT = 0.01
CONTRACT = 100.0

# Walk-forward folds (same as walk_forward.py)
TRAIN_WINDOW = 50000
TEST_WINDOW  = 10000
STEP_SIZE    = 20000


def load_data():
    """Load and prepare data exactly as env.py does."""
    df = pd.read_csv(CSV_M15, parse_dates=["time"])
    df = df.set_index("time")
    df_prepared = prepare_features(df, reset_index=False)

    df_daily = pd.read_csv(CSV_D1, parse_dates=["time"])
    df_daily = df_daily.set_index("time")
    df_daily_feat = compute_daily_features(df_daily)

    merged = merge_daily_into_primary(df_prepared, df_daily_feat)
    merged = merged.reset_index(drop=True)

    # Recompute ATR in USD (same as env.py)
    atr_usd = (merged["atr_ratio"] * merged["close"]).values
    return merged, atr_usd


def simulate_trade(df, atr, idx, direction):
    """
    Simulate a single trade from bar `idx` with given direction (+1 long, -1 short).
    Returns dict with trade lifecycle data.
    """
    entry_price = df["close"].iloc[idx]
    if direction == 1:
        entry_price += SPREAD / 2.0
    else:
        entry_price -= SPREAD / 2.0

    cur_atr = atr[idx]
    if cur_atr <= 0 or np.isnan(cur_atr):
        return None

    if direction == 1:
        sl_price = entry_price - SL_ATR_MULT * cur_atr
        tp_price = entry_price + TP_ATR_MULT * cur_atr
    else:
        sl_price = entry_price + SL_ATR_MULT * cur_atr
        tp_price = entry_price - TP_ATR_MULT * cur_atr

    sl_dist = abs(entry_price - sl_price)
    tp_dist = abs(entry_price - tp_price)

    # Track MFE (max favorable) and MAE (max adverse)
    max_favorable = 0.0
    max_adverse = 0.0
    result = None
    exit_bar = None
    bars_held = 0

    # Was the first move in our direction?
    first_move_favorable = None

    for j in range(idx + 1, min(idx + 500, len(df))):  # max 500 bars (~5 days)
        h = df["high"].iloc[j]
        l = df["low"].iloc[j]
        c = df["close"].iloc[j]

        if direction == 1:
            favorable = h - entry_price
            adverse = entry_price - l
        else:
            favorable = entry_price - l
            adverse = h - entry_price

        max_favorable = max(max_favorable, favorable)
        max_adverse = max(max_adverse, adverse)

        if first_move_favorable is None and j == idx + 1:
            mid = (h + l) / 2.0
            if direction == 1:
                first_move_favorable = mid > entry_price
            else:
                first_move_favorable = mid < entry_price

        # Check TP/SL
        hit_tp = False
        hit_sl = False
        if direction == 1:
            if h >= tp_price:
                hit_tp = True
            if l <= sl_price:
                hit_sl = True
        else:
            if l <= tp_price:
                hit_tp = True
            if h >= sl_price:
                hit_sl = True

        if hit_tp or hit_sl:
            # If both hit in same bar, check which was hit first using open direction
            if hit_tp and hit_sl:
                # Conservative: assume SL was hit (worst case)
                result = "SL"
            elif hit_tp:
                result = "TP"
            else:
                result = "SL"
            exit_bar = j
            bars_held = j - idx
            break
    else:
        # Timed out (500 bars, no TP/SL hit)
        result = "TIMEOUT"
        exit_bar = min(idx + 499, len(df) - 1)
        bars_held = exit_bar - idx

    # PnL
    if result == "TP":
        pnl = tp_dist * LOT * CONTRACT - COMMISSION * 2
    elif result == "SL":
        pnl = -sl_dist * LOT * CONTRACT - COMMISSION * 2
    else:
        final_close = df["close"].iloc[exit_bar]
        pnl = (final_close - entry_price) * direction * LOT * CONTRACT - COMMISSION * 2

    # TP progress: how far toward TP did the trade get? (0% = never moved, 100% = hit TP)
    tp_progress_pct = (max_favorable / tp_dist * 100) if tp_dist > 0 else 0

    return {
        "entry_bar": idx,
        "direction": direction,
        "entry_price": entry_price,
        "atr": cur_atr,
        "sl_dist": sl_dist,
        "tp_dist": tp_dist,
        "result": result,
        "bars_held": bars_held,
        "mfe": max_favorable,
        "mae": max_adverse,
        "mfe_atr": max_favorable / cur_atr if cur_atr > 0 else 0,
        "mae_atr": max_adverse / cur_atr if cur_atr > 0 else 0,
        "tp_progress_pct": tp_progress_pct,
        "pnl": pnl,
        "first_move_ok": first_move_favorable,
        "sl_hit_within_5bars": result == "SL" and bars_held <= 5,
        "sl_hit_within_10bars": result == "SL" and bars_held <= 10,
    }


def analyze_systematic(df, atr, start, end, label=""):
    """
    Run systematic trades on a data segment.
    Strategy: Enter every 20 bars, alternating long/short, PLUS
    a trend-aligned strategy using d_trend_strength.
    """
    trades_random = []
    trades_trend = []

    segment = df.iloc[start:end]
    # Ensure we have enough lookahead
    max_entry = min(end, len(df) - 501)

    for i in range(start, max_entry, 20):
        # Random direction (alternating) — baseline
        direction = 1 if ((i - start) // 20) % 2 == 0 else -1
        t = simulate_trade(df, atr, i, direction)
        if t:
            trades_random.append(t)

        # Trend-aligned: use daily trend
        if "d_trend_strength" in df.columns:
            trend = df["d_trend_strength"].iloc[i]
            if trend > 0.1:
                t2 = simulate_trade(df, atr, i, 1)  # long with uptrend
                if t2:
                    trades_trend.append(t2)
            elif trend < -0.1:
                t2 = simulate_trade(df, atr, i, -1)  # short with downtrend
                if t2:
                    trades_trend.append(t2)

    return trades_random, trades_trend


def print_trade_stats(trades, label):
    """Print comprehensive statistics for a set of trades."""
    if not trades:
        print(f"  {label}: No trades")
        return

    tdf = pd.DataFrame(trades)
    n = len(tdf)
    tp_trades = tdf[tdf["result"] == "TP"]
    sl_trades = tdf[tdf["result"] == "SL"]
    timeout_trades = tdf[tdf["result"] == "TIMEOUT"]

    wr = len(tp_trades) / n * 100
    total_pnl = tdf["pnl"].sum()
    avg_pnl = tdf["pnl"].mean()

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  Total trades:       {n}")
    print(f"  TP / SL / Timeout:  {len(tp_trades)} / {len(sl_trades)} / {len(timeout_trades)}")
    print(f"  Win Rate:           {wr:.1f}%")
    print(f"  Total PnL:          ${total_pnl:.2f}")
    print(f"  Avg PnL/trade:      ${avg_pnl:.4f}")
    print()

    # MFE / MAE analysis
    print(f"  --- MFE/MAE Analysis ---")
    print(f"  Avg MFE (ATR):      {tdf['mfe_atr'].mean():.2f}  (TP is at {TP_ATR_MULT})")
    print(f"  Avg MAE (ATR):      {tdf['mae_atr'].mean():.2f}  (SL is at {SL_ATR_MULT})")

    if len(sl_trades) > 0:
        print(f"\n  --- SL Trades Deep Dive ---")
        print(f"  Avg TP progress:    {sl_trades['tp_progress_pct'].mean():.1f}%")
        print(f"  Median TP progress: {sl_trades['tp_progress_pct'].median():.1f}%")

        # How many SL trades got >50% to TP?
        sl_almost = sl_trades[sl_trades["tp_progress_pct"] > 50]
        print(f"  SL trades >50% to TP: {len(sl_almost)}/{len(sl_trades)} ({len(sl_almost)/len(sl_trades)*100:.1f}%)")

        # How many SL trades got >75% to TP?
        sl_almost75 = sl_trades[sl_trades["tp_progress_pct"] > 75]
        print(f"  SL trades >75% to TP: {len(sl_almost75)}/{len(sl_trades)} ({len(sl_almost75)/len(sl_trades)*100:.1f}%)")

        # SL speed: how quickly did trades hit SL?
        sl_fast5 = sl_trades[sl_trades["sl_hit_within_5bars"]]
        sl_fast10 = sl_trades[sl_trades["sl_hit_within_10bars"]]
        print(f"  SL within 5 bars:   {len(sl_fast5)}/{len(sl_trades)} ({len(sl_fast5)/len(sl_trades)*100:.1f}%) — noise stops")
        print(f"  SL within 10 bars:  {len(sl_fast10)}/{len(sl_trades)} ({len(sl_fast10)/len(sl_trades)*100:.1f}%)")
        print(f"  Avg bars to SL:     {sl_trades['bars_held'].mean():.1f}")

    if len(tp_trades) > 0:
        print(f"\n  --- TP Trades ---")
        print(f"  Avg bars to TP:     {tp_trades['bars_held'].mean():.1f}")
        print(f"  Avg MAE (ATR):      {tp_trades['mae_atr'].mean():.2f}  (drawdown before winning)")

    # First move analysis
    first_move_ok = tdf[tdf["first_move_ok"] == True]
    first_move_bad = tdf[tdf["first_move_ok"] == False]
    if len(first_move_ok) > 0:
        wr_ok = len(first_move_ok[first_move_ok["result"] == "TP"]) / len(first_move_ok) * 100
        print(f"\n  --- First Bar Direction ---")
        print(f"  First bar favorable:    {len(first_move_ok)}/{n} ({len(first_move_ok)/n*100:.1f}%)")
        print(f"  WR when first bar OK:   {wr_ok:.1f}%")
    if len(first_move_bad) > 0:
        wr_bad = len(first_move_bad[first_move_bad["result"] == "TP"]) / len(first_move_bad) * 100
        print(f"  WR when first bar BAD:  {wr_bad:.1f}%")

    # Direction analysis
    longs = tdf[tdf["direction"] == 1]
    shorts = tdf[tdf["direction"] == -1]
    if len(longs) > 0:
        wr_long = len(longs[longs["result"] == "TP"]) / len(longs) * 100
        print(f"\n  --- LONG vs SHORT ---")
        print(f"  LONG:  {len(longs)} trades, WR={wr_long:.1f}%, PnL=${longs['pnl'].sum():.2f}")
    if len(shorts) > 0:
        wr_short = len(shorts[shorts["result"] == "TP"]) / len(shorts) * 100
        print(f"  SHORT: {len(shorts)} trades, WR={wr_short:.1f}%, PnL=${shorts['pnl'].sum():.2f}")


def main():
    print("Loading data...")
    df, atr = load_data()
    print(f"  Loaded {len(df)} rows")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}" if hasattr(df.index, 'date') else "")
    print(f"  Avg ATR: ${np.nanmean(atr):.2f}")
    print(f"  SL = {SL_ATR_MULT} x ATR = ~${SL_ATR_MULT * np.nanmean(atr):.2f}")
    print(f"  TP = {TP_ATR_MULT} x ATR = ~${TP_ATR_MULT * np.nanmean(atr):.2f}")
    print(f"  RR = 1:{TP_ATR_MULT/SL_ATR_MULT:.0f}, Breakeven WR = {100/(TP_ATR_MULT/SL_ATR_MULT+1):.1f}%")

    # ── 1. Full-dataset analysis ──
    print("\n" + "="*70)
    print("  PART 1: FULL DATASET ANALYSIS")
    print("="*70)

    all_random, all_trend = analyze_systematic(df, atr, 0, len(df) - 501, "Full")
    print_trade_stats(all_random, "RANDOM DIRECTION (baseline — what's the base rate?)")
    print_trade_stats(all_trend, "TREND-ALIGNED (daily trend > 0.1 -> long, < -0.1 -> short)")

    # ── 2. Per-fold analysis (walk-forward test windows) ──
    print("\n\n" + "="*70)
    print("  PART 2: PER-FOLD TEST WINDOW ANALYSIS (Trend-Aligned)")
    print("="*70)

    n_rows = len(df)
    fold = 0
    start = 0
    fold_results = []
    while True:
        test_start = start + TRAIN_WINDOW
        test_end = test_start + TEST_WINDOW
        if test_end > n_rows:
            break
        fold += 1
        _, trend_trades = analyze_systematic(df, atr, test_start, test_end, f"Fold {fold}")
        if trend_trades:
            tdf = pd.DataFrame(trend_trades)
            wr = len(tdf[tdf["result"] == "TP"]) / len(tdf) * 100
            pnl = tdf["pnl"].sum()
            avg_atr = atr[test_start:test_end].mean()
            fold_results.append({
                "fold": fold,
                "rows": f"{test_start}-{test_end}",
                "trades": len(tdf),
                "wr": wr,
                "pnl": pnl,
                "avg_atr": avg_atr,
                "avg_tp_progress_sl": tdf[tdf["result"] == "SL"]["tp_progress_pct"].mean() if len(tdf[tdf["result"] == "SL"]) > 0 else 0,
            })
            print(f"  Fold {fold} [{test_start}-{test_end}]: {len(tdf)} trades, WR={wr:.1f}%, PnL=${pnl:.2f}, ATR=${avg_atr:.2f}")
        start += STEP_SIZE

    # ── 3. SL tightness sensitivity ──
    print("\n\n" + "="*70)
    print("  PART 3: SL TIGHTNESS SENSITIVITY TEST")
    print("  (What if SL were wider? Same entries, different SL/TP)")
    print("="*70)

    test_segment_start = TRAIN_WINDOW
    test_segment_end = TRAIN_WINDOW + TEST_WINDOW * 3  # Use 3 folds of data

    configs = [
        (1.0, 3.0, "SL=1.0 TP=3.0 (tight, RR 1:3)"),
        (1.5, 4.5, "SL=1.5 TP=4.5 (current, RR 1:3)"),
        (2.0, 6.0, "SL=2.0 TP=6.0 (wider, RR 1:3)"),
        (2.5, 7.5, "SL=2.5 TP=7.5 (very wide, RR 1:3)"),
        (1.5, 3.0, "SL=1.5 TP=3.0 (same SL, RR 1:2)"),
        (1.5, 6.0, "SL=1.5 TP=6.0 (same SL, RR 1:4)"),
        (2.0, 4.0, "SL=2.0 TP=4.0 (wider SL, RR 1:2)"),
    ]

    # Save original
    orig_sl = SL_ATR_MULT
    orig_tp = TP_ATR_MULT

    for sl_m, tp_m, desc in configs:
        # Monkey-patch globals for simulate_trade
        globals()["SL_ATR_MULT"] = sl_m
        globals()["TP_ATR_MULT"] = tp_m

        trades = []
        for i in range(test_segment_start, min(test_segment_end, len(df) - 501), 20):
            if "d_trend_strength" in df.columns:
                trend = df["d_trend_strength"].iloc[i]
                if trend > 0.1:
                    t = simulate_trade(df, atr, i, 1)
                    if t:
                        trades.append(t)
                elif trend < -0.1:
                    t = simulate_trade(df, atr, i, -1)
                    if t:
                        trades.append(t)

        if trades:
            tdf = pd.DataFrame(trades)
            wr = len(tdf[tdf["result"] == "TP"]) / len(tdf) * 100
            pnl = tdf["pnl"].sum()
            fast_sl = len(tdf[(tdf["result"] == "SL") & (tdf["sl_hit_within_5bars"])]) if len(tdf[tdf["result"] == "SL"]) > 0 else 0
            sl_count = len(tdf[tdf["result"] == "SL"])
            print(f"  {desc}")
            print(f"    -> {len(trades)} trades, WR={wr:.1f}%, PnL=${pnl:.2f}, SL<5bars={fast_sl}/{sl_count}")

    # Restore
    globals()["SL_ATR_MULT"] = orig_sl
    globals()["TP_ATR_MULT"] = orig_tp

    # ── 4. ATR regime analysis ──
    print("\n\n" + "="*70)
    print("  PART 4: ATR REGIME ANALYSIS")
    print("  (Does gold volatility affect trade outcomes?)")
    print("="*70)

    if all_trend:
        tdf = pd.DataFrame(all_trend)
        # Split by ATR percentile
        atr_median = tdf["atr"].median()
        atr_p25 = tdf["atr"].quantile(0.25)
        atr_p75 = tdf["atr"].quantile(0.75)

        low_vol = tdf[tdf["atr"] < atr_p25]
        mid_vol = tdf[(tdf["atr"] >= atr_p25) & (tdf["atr"] <= atr_p75)]
        high_vol = tdf[tdf["atr"] > atr_p75]

        for subset, name in [(low_vol, "Low vol (ATR < P25)"), (mid_vol, "Mid vol (P25-P75)"), (high_vol, "High vol (ATR > P75)")]:
            if len(subset) > 0:
                wr = len(subset[subset["result"] == "TP"]) / len(subset) * 100
                pnl = subset["pnl"].sum()
                print(f"  {name}: {len(subset)} trades, WR={wr:.1f}%, PnL=${pnl:.2f}, Avg ATR=${subset['atr'].mean():.2f}")

    # ── 5. Sample individual trades for qualitative review ──
    print("\n\n" + "="*70)
    print("  PART 5: SAMPLE TRADES (10 TP + 10 SL from trend-aligned)")
    print("="*70)

    if all_trend:
        tdf = pd.DataFrame(all_trend)
        tp_samples = tdf[tdf["result"] == "TP"].head(10)
        sl_samples = tdf[tdf["result"] == "SL"].head(10)

        print("\n  --- Winning Trades (TP Hit) ---")
        print(f"  {'Bar':>7}  {'Dir':>5}  {'Entry':>10}  {'ATR':>7}  {'Bars':>5}  {'MFE_ATR':>8}  {'MAE_ATR':>8}  {'PnL':>8}")
        for _, t in tp_samples.iterrows():
            d_str = "LONG" if t["direction"] == 1 else "SHORT"
            print(f"  {int(t['entry_bar']):>7}  {d_str:>5}  {t['entry_price']:>10.2f}  {t['atr']:>7.2f}  {int(t['bars_held']):>5}  {t['mfe_atr']:>8.2f}  {t['mae_atr']:>8.2f}  {t['pnl']:>+8.4f}")

        print("\n  --- Losing Trades (SL Hit) ---")
        print(f"  {'Bar':>7}  {'Dir':>5}  {'Entry':>10}  {'ATR':>7}  {'Bars':>5}  {'MFE_ATR':>8}  {'MAE_ATR':>8}  {'TP_Prog%':>8}  {'Fast?':>5}")
        for _, t in sl_samples.iterrows():
            d_str = "LONG" if t["direction"] == 1 else "SHORT"
            fast = "YES" if t["sl_hit_within_5bars"] else ""
            print(f"  {int(t['entry_bar']):>7}  {d_str:>5}  {t['entry_price']:>10.2f}  {t['atr']:>7.2f}  {int(t['bars_held']):>5}  {t['mfe_atr']:>8.2f}  {t['mae_atr']:>8.2f}  {t['tp_progress_pct']:>7.1f}%  {fast:>5}")

        # ── 5b: "Almost made it" trades ──
        print("\n  --- 'Almost Made It' SL Trades (>60% to TP before dying) ---")
        almost = tdf[(tdf["result"] == "SL") & (tdf["tp_progress_pct"] > 60)]
        if len(almost) > 0:
            print(f"  Found {len(almost)}/{len(tdf[tdf['result']=='SL'])} SL trades that got >60% to TP")
            print(f"  {'Bar':>7}  {'Dir':>5}  {'Entry':>10}  {'ATR':>7}  {'Bars':>5}  {'MFE_ATR':>8}  {'MAE_ATR':>8}  {'TP_Prog%':>8}")
            for _, t in almost.head(10).iterrows():
                d_str = "LONG" if t["direction"] == 1 else "SHORT"
                print(f"  {int(t['entry_bar']):>7}  {d_str:>5}  {t['entry_price']:>10.2f}  {t['atr']:>7.2f}  {int(t['bars_held']):>5}  {t['mfe_atr']:>8.2f}  {t['mae_atr']:>8.2f}  {t['tp_progress_pct']:>7.1f}%")
        else:
            print("  None found.")

    # ── 6. The key question: Is 1:3 RR achievable? ──
    print("\n\n" + "="*70)
    print("  PART 6: KEY CONCLUSIONS")
    print("="*70)

    if all_random and all_trend:
        rdf = pd.DataFrame(all_random)
        tdf = pd.DataFrame(all_trend)

        rwr = len(rdf[rdf["result"] == "TP"]) / len(rdf) * 100
        twr = len(tdf[tdf["result"] == "TP"]) / len(tdf) * 100

        print(f"\n  Random baseline WR:     {rwr:.1f}% (breakeven = 25%)")
        print(f"  Trend-aligned WR:       {twr:.1f}%")
        print(f"  Trend edge:             {twr - rwr:+.1f}% WR improvement")
        print()

        sl_trades = tdf[tdf["result"] == "SL"]
        if len(sl_trades) > 0:
            fast_sl_pct = len(sl_trades[sl_trades["sl_hit_within_5bars"]]) / len(sl_trades) * 100
            avg_tp_prog = sl_trades["tp_progress_pct"].mean()
            print(f"  SL trades hit within 5 bars: {fast_sl_pct:.1f}% — {'HIGH -> SL is too tight (noise)' if fast_sl_pct > 40 else 'OK — SL gives room'}")
            print(f"  Avg TP progress of SL trades: {avg_tp_prog:.1f}% — {'many get close -> SL timing issue' if avg_tp_prog > 30 else 'trades die early -> direction issue'}")

            almost_pct = len(sl_trades[sl_trades["tp_progress_pct"] > 50]) / len(sl_trades) * 100
            print(f"  SL trades >50% to TP: {almost_pct:.1f}% — {'significant -> wider SL or trailing could help' if almost_pct > 15 else 'few -> direction is the bottleneck'}")

        print()
        if twr >= 28:
            print("  [OK] Trend-aligned systematic achieves profitable WR (>=28%)")
            print("     The agent needs to learn to replicate this — not find a new strategy")
        elif twr >= 25:
            print("  [WARN]  Trend-aligned barely breaks even (25-28%)")
            print("     May need additional edge (entry timing, volatility filter)")
        else:
            print("  [BAD] Even trend-aligned can't break even (<25%)")
            print("     The 1:3 RR with current ATR mults may be fundamentally too wide")
            print("     Consider: lower TP (1:2 RR), or wider SL, or different timeframe")


if __name__ == "__main__":
    main()
