"""
Oracle test: Supply Zone SHORT edge
====================================
Tests specifically: H1 bear + supply zone → SHORT WR
Compared against random SHORT baseline (~23.8%)

Entry direction: SHORT (d = -1)
SL = 1.5 ATR above entry, TP = 4.5 ATR below entry
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import yaml

def load_data():
    with open("config_scalp.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    from src.data import load_data as _ld
    df = _ld(cfg)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


def sim_short(df: pd.DataFrame, mask: pd.Series, label: str,
              sl_atr=1.5, tp_atr=4.5, cooldown=4, max_hold=200, spread=0.16):
    """Simulate SHORT entries where mask is True."""
    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    atrs   = (df["atr_ratio"] * df["close"]).values
    mask_v = mask.values

    trades = []
    last = -cooldown - 1
    n = len(df)

    for i in range(50, n - max_hold - 2):
        if not mask_v[i]:
            continue
        if i - last < cooldown:
            continue
        atr = atrs[i]
        if np.isnan(atr) or atr <= 0:
            continue

        entry = closes[i] - spread / 2.0   # SHORT: sell at bid
        tp    = entry - tp_atr * atr
        sl    = entry + sl_atr * atr

        outcome = None
        ep = None
        for j in range(i + 1, min(i + max_hold + 1, n)):
            if lows[j] <= tp:
                outcome = "win"; ep = tp; break
            elif highs[j] >= sl:
                outcome = "loss"; ep = sl; break

        if outcome is None:
            ep = closes[min(i + max_hold, n - 1)]
            outcome = "win" if closes[i] > ep else "loss"

        trades.append({"win": 1 if outcome == "win" else 0,
                       "pnl": (entry - ep)})
        last = i

    if not trades:
        print(f"  {label:50s} | NO TRADES")
        return 0.0

    tdf = pd.DataFrame(trades)
    wr = tdf["win"].mean() * 100
    n_tr = len(tdf)
    active = mask_v.mean() * 100

    baseline_wr = 23.8  # known random SHORT baseline
    vs_base = wr - baseline_wr
    breakeven = 1 / (1 + tp_atr / sl_atr) * 100  # 18.2%
    vs_be = wr - breakeven

    tag = " [EDGE]" if vs_base >= 2.0 else (" [ok]" if vs_base >= 0 else " [BAD]")
    sign = "+" if vs_base >= 0 else ""
    print(f"  {label:50s} | T={n_tr:4d} | Act={active:4.1f}% | "
          f"WR={wr:5.1f}% | vs_baseline={sign}{vs_base:.1f}%{tag}")
    return wr


def main():
    print("=" * 70)
    print("  SUPPLY ZONE SHORT — ORACLE TEST")
    print("  SL=1.5 ATR, TP=4.5 ATR, RR 1:3, breakeven WR=18.2%")
    print("  Random SHORT baseline: ~23.8%")
    print("=" * 70)

    df = load_data()

    # ── Derived masks ──────────────────────────────────────────────────────
    h1_bear    = df["h1_trend_strength"] < -0.05 if "h1_trend_strength" in df.columns \
                 else pd.Series(False, index=df.index)
    supply_act = df["supply_active"] > 0.5 if "supply_active" in df.columns \
                 else pd.Series(False, index=df.index)
    bear_body  = (df["close"] < df["open"]) & \
                 (df["candle_body_ratio"] > 0.4) if "candle_body_ratio" in df.columns \
                 else (df["close"] < df["open"])
    # ds_distance: signed distance to zone / ATR
    # negative = price below zone top = inside or approaching from below (supply reversal = price came from below and now AT zone)
    # We want price to be IN or AT the supply zone
    if "ds_distance" in df.columns:
        near_supply = supply_act & (df["ds_distance"].abs() < 1.0)  # within 1 ATR of zone
    else:
        near_supply = supply_act

    choch_bear = df["choch_bearish"] > 0.1 if "choch_bearish" in df.columns \
                 else pd.Series(False, index=df.index)

    print("\n  [Baseline checks]")
    # H1 bear activity
    print(f"  H1 bear active:    {h1_bear.mean()*100:.1f}% of bars")
    print(f"  Supply active:     {supply_act.mean()*100:.1f}% of bars")
    print(f"  Near supply:       {near_supply.mean()*100:.1f}% of bars")
    print(f"  Bear body:         {bear_body.mean()*100:.1f}% of bars")
    print(f"  CHOCH bear:        {choch_bear.mean()*100:.1f}% of bars")

    print("\n" + "-" * 70)
    print("  SHORT CONDITIONS vs WR")
    print("-" * 70)

    # 1. Random SHORT baseline (sanity check)
    rng = np.random.default_rng(42)
    rand_mask = pd.Series(rng.integers(0, 2, len(df)).astype(bool), index=df.index)
    sim_short(df, rand_mask, "Random SHORT (sanity check)")

    # 2. Pure supply zone (no trend filter)
    sim_short(df, supply_act, "Supply zone only")

    # 3. H1 bear only
    sim_short(df, h1_bear, "H1 bear only")

    # 4. H1 bear + supply
    sim_short(df, h1_bear & supply_act, "H1 bear + supply_active")

    # 5. H1 bear + supply + near zone
    sim_short(df, h1_bear & near_supply, "H1 bear + supply near zone (<1 ATR)")

    # 6. H1 bear + supply + bearish body
    sim_short(df, h1_bear & supply_act & bear_body, "H1 bear + supply + bearish body")

    # 7. H1 bear + supply + near zone + bearish body  ← TARGET
    sim_short(df, h1_bear & near_supply & bear_body,
              "H1 bear + supply near + bearish body  [TARGET]")

    # 8. Current mask: H1 bear + CHOCH bear + bearish body
    sim_short(df, h1_bear & choch_bear & bear_body,
              "H1 bear + CHOCH bear + bearish body (current)")

    # 9. Combined: either supply OR choch
    sim_short(df, h1_bear & (supply_act | choch_bear) & bear_body,
              "H1 bear + (supply OR choch) + body")

    # 10. Both paths together (supply AND choch)
    sim_short(df, h1_bear & supply_act & choch_bear & bear_body,
              "H1 bear + supply AND choch + body")

    # ── SRF Bear (Chart 3 setup: breakout-retest) ────────────────────────
    print("\n" + "-" * 70)
    print("  SRF BEAR SHORT (Breakout-Retest = Chart 3 setup)")
    print("-" * 70)
    if "srf_bull_dist" in df.columns:
        # srf_bear = former support now resistance
        # We don't have srf_bear_dist directly, but we can proxy:
        # Price near a level that was previously broken = ds_distance ≈ 0 with supply
        # Better: compute wick ratio + supply for rejection test
        pass

    # ── Engulfing detection ───────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("  BEARISH ENGULFING (Chart 1,2,4 setup)")
    print("-" * 70)
    closes_arr = df["close"].values
    opens_arr  = df["open"].values
    engulf = np.zeros(len(df), dtype=bool)
    for ii in range(1, len(df)):
        prev_bull = closes_arr[ii-1] > opens_arr[ii-1]  # prev candle bullish
        curr_bear = closes_arr[ii] < opens_arr[ii]       # curr candle bearish
        curr_open_above_prev_close = opens_arr[ii] > closes_arr[ii-1]
        curr_close_below_prev_open = closes_arr[ii] < opens_arr[ii-1]
        if prev_bull and curr_bear and curr_open_above_prev_close and curr_close_below_prev_open:
            engulf[ii] = True
    engulf_s = pd.Series(engulf, index=df.index)
    print(f"  Engulfing active: {engulf_s.mean()*100:.1f}% of bars")

    sim_short(df, engulf_s, "Bearish Engulfing only")
    sim_short(df, h1_bear & engulf_s, "H1 bear + Engulfing")
    sim_short(df, h1_bear & supply_act & engulf_s, "H1 bear + supply + Engulfing")
    sim_short(df, h1_bear & choch_bear & engulf_s, "H1 bear + CHOCH + Engulfing")

    # ── Upper Wick Rejection ──────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("  UPPER WICK REJECTION (rejection at supply)")
    print("-" * 70)
    highs_arr = df["high"].values
    atrs_arr  = (df["atr_ratio"] * df["close"]).values
    wick_ratio = np.zeros(len(df), dtype=np.float32)
    for ii in range(len(df)):
        atr_v = atrs_arr[ii]
        if atr_v > 0:
            upper_wick = highs_arr[ii] - max(closes_arr[ii], opens_arr[ii])
            wick_ratio[ii] = upper_wick / atr_v
    wick_s = pd.Series(wick_ratio, index=df.index)
    big_wick = wick_s > 0.5  # upper wick > 0.5 ATR = rejection
    print(f"  Big upper wick (>0.5 ATR): {big_wick.mean()*100:.1f}% of bars")

    sim_short(df, big_wick, "Big upper wick only")
    sim_short(df, h1_bear & supply_act & big_wick, "H1 bear + supply + big wick")
    sim_short(df, h1_bear & (choch_bear | (supply_act & big_wick)) & bear_body,
              "H1 bear + (CHOCH OR supply+wick) + body")

    # ── Pullback context: M15 rising into supply ──────────────────────────
    print("\n" + "-" * 70)
    print("  M15 PULLBACK INTO SUPPLY (price came from below)")
    print("-" * 70)
    ret3 = df["close"].pct_change(3)  # last 3 bars return
    pullback_up = ret3 > 0.0005  # price rose 0.05%+ in last 3 bars (was going up)
    print(f"  M15 pullback up (3-bar positive): {pullback_up.mean()*100:.1f}% of bars")

    sim_short(df, h1_bear & supply_act & pullback_up, "H1 bear + supply + pullback up")
    sim_short(df, h1_bear & supply_act & pullback_up & bear_body,
              "H1 bear + supply + pullback + body")
    sim_short(df, h1_bear & supply_act & pullback_up & engulf_s,
              "H1 bear + supply + pullback + Engulfing")

    print("\n" + "-" * 70)
    print("  INTERPRETATION:")
    print("  Edge = WR vs random SHORT baseline (+2% = add to mask)")
    print("  Breakeven WR at RR 1:3 = 18.2% (any WR above this = EV+)")
    print("-" * 70)


if __name__ == "__main__":
    main()
