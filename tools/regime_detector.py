"""
Regime Detector — แยกตลาด trending vs choppy ด้วย ADX จาก daily data

Hypothesis (Plan B): บอท Iter 2 (trend follower) เก่งใน trending,
บอท Iter 4 (mean reverter w/ multi-tf) เก่งใน choppy

ADX bands:
  ADX > 25  -> TREND  (ใช้ Iter 2)
  ADX < 20  -> CHOP   (ใช้ Iter 4)
  20-25     -> NEUTRAL (hysteresis: คงรุ่นเดิมเพื่อกัน flip-flop)

Usage (offline analysis):
    python -m tools.regime_detector
"""
from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Literal

# Force UTF-8 for stdout (Windows Thai cp874 ไม่รองรับ unicode)
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


Regime = Literal["TREND", "CHOP", "NEUTRAL"]


def compute_adx(df_daily: pd.DataFrame, window: int = 14) -> pd.Series:
    """ADX จาก daily OHLC. คืน Series indexed by daily date."""
    from ta.trend import ADXIndicator

    df = df_daily.copy()
    df.columns = [c.lower() for c in df.columns]
    adx = ADXIndicator(df["high"], df["low"], df["close"], window=window).adx()
    return adx.dropna()


def classify_regime(adx_value: float, prev_regime: Regime = "CHOP",
                    upper: float = 25.0, lower: float = 20.0) -> Regime:
    """
    Hysteresis classifier:
      ADX > upper        -> TREND
      ADX < lower        -> CHOP
      lower <= ADX <= upper -> stay with prev (avoid flip-flop)
    """
    if adx_value > upper:
        return "TREND"
    if adx_value < lower:
        return "CHOP"
    return prev_regime


def regime_series(df_daily: pd.DataFrame, window: int = 14,
                  upper: float = 25.0, lower: float = 20.0) -> pd.Series:
    """Apply hysteresis classifier across full daily series."""
    adx = compute_adx(df_daily, window=window)
    regimes = []
    prev: Regime = "CHOP"
    for v in adx.values:
        r = classify_regime(float(v), prev, upper, lower)
        regimes.append(r)
        prev = r
    return pd.Series(regimes, index=adx.index, name="regime")


def align_regime_to_primary(regime_daily: pd.Series,
                            primary_index: pd.DatetimeIndex) -> pd.Series:
    """
    Map daily regime -> primary (4h) timeline ด้วย shift(1)+ffill
    เหมือน merge_daily_into_primary ใน env.py — กัน look-ahead
    """
    if regime_daily.index.tz is not None:
        regime_daily = regime_daily.copy()
        regime_daily.index = regime_daily.index.tz_localize(None)
    if primary_index.tz is not None:
        primary_index = primary_index.tz_localize(None)
    safe = regime_daily.shift(1).dropna()
    aligned = safe.reindex(primary_index, method="ffill")
    return aligned


# ---------------------------------------------------------------------------
# Offline analysis: validate hypothesis on existing walk-forward folds
# ---------------------------------------------------------------------------

def main():
    """
    Load same data ที่ walk_forward ใช้, compute regime, ดูว่า
    แต่ละ fold (เดิม train 1500, test 400, step 400) อยู่ใน regime ไหนเป็นหลัก
    """
    import yaml
    from src.data import load_yfinance, load_data

    with open("config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print("=" * 70)
    print(" REGIME ANALYSIS — validate Plan B hypothesis")
    print("=" * 70)

    # ดึง primary (4h) แบบเดียวกับ walk_forward ใช้
    print("\nLoading primary (4h) + daily data...")
    df_primary = load_data(cfg)  # แล้วมี datetime index หาย (reset_index ใน load_data)
    # โหลด daily แยกสำหรับ ADX
    df_daily_raw = load_yfinance(cfg["data"]["symbol"], "1d", cfg["data"]["period"])

    print(f"  primary rows: {len(df_primary):,}")
    print(f"  daily rows:   {len(df_daily_raw):,}")

    # ADX + regime
    print("\nComputing ADX(14) + regime classification...")
    adx = compute_adx(df_daily_raw, window=14)
    regimes = regime_series(df_daily_raw, window=14, upper=25.0, lower=20.0)

    print(f"  ADX range: {adx.min():.1f} - {adx.max():.1f}")
    print(f"  ADX mean:  {adx.mean():.1f}")
    print(f"  Regime distribution (daily):")
    counts = regimes.value_counts()
    total = len(regimes)
    for r in ["TREND", "CHOP", "NEUTRAL"]:
        n = counts.get(r, 0)
        print(f"    {r:8s}: {n:4d} days ({n/total*100:5.1f}%)")

    # ----- Fold-by-fold analysis (matching walk_forward args ที่ user รัน) -----
    # walk_forward.py สร้าง folds จาก df ที่ load_data คืน (reset_index แล้ว) ไม่มี datetime
    # เราต้อง re-load โดยเก็บ datetime เพื่อ map regime
    print("\n" + "=" * 70)
    print(" FOLD-BY-FOLD REGIME (train_size=1500 test_size=400 step=400)")
    print("=" * 70)

    # rebuild merged df แต่เก็บ datetime index
    from src.env import prepare_features, compute_daily_features, merge_daily_into_primary
    df_pri_raw = load_yfinance(cfg["data"]["symbol"], cfg["data"]["interval"], cfg["data"]["period"])
    df_pri_feat = prepare_features(df_pri_raw, reset_index=False)
    df_d_feat = compute_daily_features(df_daily_raw)
    df_merged = merge_daily_into_primary(df_pri_feat, df_d_feat)
    primary_idx = df_merged.index

    # align regime to primary timeline
    regime_4h = align_regime_to_primary(regimes, primary_idx)

    # walk_forward params ที่ user ใช้
    train_size, test_size, step_size = 1500, 400, 400
    n_rows = len(df_merged)
    folds = []
    s = 0
    while s + train_size + test_size <= n_rows:
        folds.append((s, s + train_size, s + train_size + test_size))
        s += step_size

    iter4_alphas = [-19.87, -13.27, +0.78]  # จาก fold_results.csv ที่รันไปแล้ว
    iter4_returns = [+1.76, +0.10, +7.00]
    bnh_returns = [+21.63, +13.37, +6.23]

    print(f"\n  {'Fold':<5} {'TestStart':<20} {'TestEnd':<20} "
          f"{'TREND%':>7} {'CHOP%':>7} {'NEUT%':>7} "
          f"{'Iter4_a':>7} {'BnH':>7}  Hypothesis")
    print(f"  {'-'*5} {'-'*20} {'-'*20} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}  {'-'*30}")

    for i, (a, b, c) in enumerate(folds):
        test_start_ts = primary_idx[b]
        test_end_ts = primary_idx[c-1]
        test_regimes = regime_4h.iloc[b:c]
        rc = test_regimes.value_counts()
        n = len(test_regimes)
        trend_pct = rc.get("TREND", 0) / n * 100
        chop_pct = rc.get("CHOP", 0) / n * 100
        neut_pct = rc.get("NEUTRAL", 0) / n * 100

        # Hypothesis check
        if i < len(iter4_alphas):
            a4 = iter4_alphas[i]
            bnh = bnh_returns[i]
            # Hypothesis: trending fold → Iter 4 should LOSE alpha (mean reverter ตามเทรนด์ไม่ทัน)
            # Choppy fold → Iter 4 should WIN alpha
            if trend_pct > 50:
                hyp = "TRENDING → Iter4 should lose"
                ok = "✓" if a4 < -3 else "✗"
            elif chop_pct > 50:
                hyp = "CHOPPY → Iter4 should win"
                ok = "✓" if a4 > -3 else "✗"
            else:
                hyp = "MIXED → unclear"
                ok = "?"
            print(f"  {i+1:<5} {str(test_start_ts)[:19]:<20} {str(test_end_ts)[:19]:<20} "
                  f"{trend_pct:>6.1f}% {chop_pct:>6.1f}% {neut_pct:>6.1f}% "
                  f"{a4:>+6.1f}% {bnh:>+6.1f}%  {ok} {hyp}")
        else:
            print(f"  {i+1:<5} {str(test_start_ts)[:19]:<20} {str(test_end_ts)[:19]:<20} "
                  f"{trend_pct:>6.1f}% {chop_pct:>6.1f}% {neut_pct:>6.1f}%")

    # Verdict
    print("\n" + "=" * 70)
    print(" VERDICT")
    print("=" * 70)
    print("  ถ้า hypothesis ถูก (trending folds = Iter4 แย่, choppy folds = Iter4 ดี)")
    print("  -> Plan B ensemble คุ้มค่าที่จะ build ต่อ")
    print("  ถ้าไม่ -> ออกแบบ regime detector ใหม่หรือใช้ feature อื่น (volatility, BB width)")


if __name__ == "__main__":
    main()
