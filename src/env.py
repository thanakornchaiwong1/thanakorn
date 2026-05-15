"""
Gymnasium environment สำหรับ XAUUSD (Gold) trading.

Features ที่ implement:
  - Discrete(4) action: 0=Hold, 1=Buy, 2=Sell, 3=Close
  - Observation: window of normalized features + position info (3 ค่า)
  - Spread + commission ถูกหักจริง -> ป้องกัน overtrade
  - Max drawdown stop (default 50%)
  - Invalid-action handling (buy ตอนถืออยู่, close ตอน flat) -> ส่ง flag ให้ reward fn
  - Reward function เลือกผ่าน reward_type (delegate ไป rewards.py)
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

# Robust import: รองรับทั้งรันเป็น module (`from src.env`) และรันจากใน src/
try:
    from .rewards import get_reward_fn
except ImportError:
    from rewards import get_reward_fn  # type: ignore


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    # Technical momentum/trend
    "returns",
    "rsi",
    "sma_ratio",
    "macd_diff",
    "bb_position",
    "atr_ratio",
    # Price action: Demand/Supply zones (base-consolidation detection)
    "demand_active",
    "supply_active",
    "ds_distance",
    "ds_strength",
    "ds_freshness",
    # Price action: Fair Value Gap
    "fvg_bull_active",
    "fvg_bear_active",
    "fvg_distance",
    "fvg_strength",
    # Market context
    "is_active_session",
    "candle_body_ratio",
    # Price action: Candle patterns (vectorized, no loop overhead)
    # engulfing_bear: bullish→bearish full engulf = strong reversal at supply
    # engulfing_bull: bearish→bullish full engulf = strong reversal at demand
    # m15_pullback: 3-bar price change / ATR (+ve=price went up, -ve=went down)
    #   Used in mask Path 2: supply reversal requires pullback_val > 0.3 ATR
    "engulfing_bear",
    "engulfing_bull",
    "m15_pullback",
    # CHOCH re-added to observation after fixing:
    # OLD: 5-bar pivot → 34.8% active (too noisy to learn from)
    # NEW: 20-bar pivot, 8-bar memory → 5.9% active = real events
    # Oracle v2: LONG+CHOCH=34.6% WR (+9.7%), SHORT+CHOCH=27.4% WR (+2.5%)
    "choch_bullish",  # decaying score: 1.0=just happened, 0=expired (2h window)
    "choch_bearish",  # same for bearish CHOCH
    # SRF: oracle test shows +3.7% WR edge → included in observation
    "srf_bull_dist",  # distance to nearest bullish SRF
]

DAILY_FEATURE_COLUMNS = [
    "d_rsi",
    "d_macd_diff",
    "d_returns_1d",
    "d_returns_5d",
    "d_sma_20_ratio",
    "d_sma_50_ratio",
    "d_atr_ratio",
    "d_bb_position",
    "d_trend_strength",   # composite trend score
]

# H1 features (resampled from M15) — 11 features
# H1 gives timelier trend signal than D1 for M15 scalping
# "D1 มันใหญ่เกินไป สำหรับ M15" — user feedback
H1_FEATURE_COLUMNS = [
    "h1_rsi",
    "h1_macd_diff",
    "h1_returns_1h",
    "h1_returns_4h",         # 4H momentum (4 x H1 bars)
    "h1_sma_20_ratio",       # 20 H1 bars ≈ 1 day
    "h1_sma_50_ratio",       # 50 H1 bars ≈ 2 days
    "h1_atr_ratio",
    "h1_bb_position",
    "h1_trend_strength",     # composite: sma_align + macd (key mask feature)
    # Consistency features (rolling 8 H1 bars = 8h window)
    # Diagnosis: bad folds have H1 neutral 72-80% → CHOCH fires on noise
    # Fix: require 60%+ of last 8 H1 bars in same direction = real trend
    "h1_bull_consistency",   # fraction of last 8 H1 bars that are bullish (>0.05)
    "h1_bear_consistency",   # fraction of last 8 H1 bars that are bearish (<-0.05)
]


def _compute_structure_features(df: pd.DataFrame, atr_values: pd.Series) -> pd.DataFrame:
    """
    SMC Market Structure: CHOCH + Swing HH/HL/LH/LL detection.

    Based on trader chart analysis:
      - Track swing highs and lows (local extremes over SWING_WINDOW bars each side)
      - CHOCH (Change of Character): price breaks ABOVE last swing high after downtrend
        → signals potential reversal to bullish structure
      - BOS  (Break of Structure): price breaks ABOVE last swing high in uptrend
        → confirms bullish structure continues
      - struct_bullish = 1 when market is making HH + HL pattern
      - The DEMAND zone that traders use = the HL in bullish structure after CHOCH

    Features:
      choch_bullish  : 1.0 if bullish CHOCH occurred in last CHOCH_MEMORY candles
      choch_bearish  : 1.0 if bearish CHOCH occurred in last CHOCH_MEMORY candles
      struct_bullish : 1.0 if currently in confirmed bullish structure (HH > prev HH)
      struct_bearish : 1.0 if currently in confirmed bearish structure (LL < prev LL)
      dist_to_swing_low : signed distance from close to last swing low / ATR
                          (positive = above swing low = good for long)
    """
    # FIXED: Increased from 5→20 bars = significant swing ~5h on M15 (not micro-pivot)
    # OLD: 5-bar pivot fired on every tiny swing → CHOCH active 34.8% of bars (noise)
    # NEW: 20-bar pivot = real structural swing highs/lows humans draw on chart
    SWING_WINDOW = 20    # bars each side to confirm swing high/low
    CHOCH_MEMORY = 8     # FIXED: 40→8 bars = 2h active (not 10h "always on")
    MIN_CHOCH_BODY = 0.6 # ADDED: CHOCH candle must be strong (body > 0.6x ATR)

    n = len(df)
    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    opens  = df["open"].values
    atr    = atr_values.values

    choch_bull  = np.zeros(n, dtype=np.float32)
    choch_bear  = np.zeros(n, dtype=np.float32)
    struct_bull = np.zeros(n, dtype=np.float32)
    struct_bear = np.zeros(n, dtype=np.float32)
    dist_swl    = np.zeros(n, dtype=np.float32)

    # Rolling swing points
    swing_highs: list = []  # (bar_idx, price)
    swing_lows:  list = []

    last_choch_bull_bar = -CHOCH_MEMORY - 1
    last_choch_bear_bar = -CHOCH_MEMORY - 1

    for i in range(SWING_WINDOW, n - SWING_WINDOW):
        cur_atr = atr[i] if not np.isnan(atr[i]) and atr[i] > 0 else 1e-9

        # --- Detect swing high at bar i-SWING_WINDOW (confirmed now) ---
        pivot = i - SWING_WINDOW
        if pivot >= SWING_WINDOW:
            is_sh = all(highs[pivot] >= highs[pivot - k] for k in range(1, SWING_WINDOW + 1)) and \
                    all(highs[pivot] >= highs[pivot + k] for k in range(1, SWING_WINDOW + 1))
            is_sl = all(lows[pivot]  <= lows[pivot  - k] for k in range(1, SWING_WINDOW + 1)) and \
                    all(lows[pivot]  <= lows[pivot  + k] for k in range(1, SWING_WINDOW + 1))

            if is_sh:
                swing_highs.append((pivot, highs[pivot]))
                if len(swing_highs) > 6:
                    swing_highs.pop(0)

            if is_sl:
                swing_lows.append((pivot, lows[pivot]))
                if len(swing_lows) > 6:
                    swing_lows.pop(0)

        # --- CHOCH detection + candle quality filter ---
        # Candle body must be strong to count as a real CHOCH break (not a spike)
        candle_body = abs(closes[i] - opens[i])
        strong_candle = candle_body >= MIN_CHOCH_BODY * cur_atr

        if len(swing_highs) >= 2 and len(swing_lows) >= 2 and strong_candle:
            prev_sh = swing_highs[-2][1]
            prev_sl = swing_lows[-2][1]
            last_sh = swing_highs[-1][1]
            last_sl = swing_lows[-1][1]

            # Bullish CHOCH: strong candle closes ABOVE last swing high
            # AND last swing high was lower than previous (= downtrend structure)
            # AND the break is meaningful (> 0.2 ATR above the swing high)
            if (closes[i] > last_sh + 0.2 * cur_atr
                    and last_sh < prev_sh
                    and closes[i] > opens[i]):  # must be bullish candle
                last_choch_bull_bar = i

            # Bearish CHOCH: strong candle closes BELOW last swing low
            # AND last swing low was higher than previous (= uptrend structure breaking)
            if (closes[i] < last_sl - 0.2 * cur_atr
                    and last_sl > prev_sl
                    and closes[i] < opens[i]):  # must be bearish candle
                last_choch_bear_bar = i

        # --- Structure determination ---
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            if swing_highs[-1][1] > swing_highs[-2][1] and swing_lows[-1][1] > swing_lows[-2][1]:
                struct_bull[i] = 1.0
            elif swing_lows[-1][1] < swing_lows[-2][1] and swing_highs[-1][1] < swing_highs[-2][1]:
                struct_bear[i] = 1.0

        # --- CHOCH memory (decays linearly) ---
        if i - last_choch_bull_bar <= CHOCH_MEMORY:
            choch_bull[i] = 1.0 - (i - last_choch_bull_bar) / CHOCH_MEMORY
        if i - last_choch_bear_bar <= CHOCH_MEMORY:
            choch_bear[i] = 1.0 - (i - last_choch_bear_bar) / CHOCH_MEMORY

        # --- Distance to nearest swing low ---
        if swing_lows:
            nearest_sl = swing_lows[-1][1]
            raw = closes[i] - nearest_sl
            dist_swl[i] = np.clip(raw / cur_atr, -3.0, 3.0)

    df["choch_bullish"]     = choch_bull
    df["choch_bearish"]     = choch_bear
    df["struct_bullish"]    = struct_bull
    df["struct_bearish"]    = struct_bear
    df["dist_to_swing_low"] = dist_swl

    return df


def _compute_rbr_srf_features(df: pd.DataFrame, atr_values: pd.Series) -> pd.DataFrame:
    """
    RBR (Rally-Base-Rally) / DBD (Drop-Base-Drop) and SRF (Support-Resistance Flip).

    RBR Demand zone (from trader charts):
      Rally (strong bullish move) → Base (tight consolidation) → Rally again
      The BASE is the demand zone — strongest type of demand
      "พอปิดแท่งราคาย้อนมาเข้า BUY เลย" = entry when price returns to base

    DBD Supply zone:
      Drop → Base → Drop = supply zone at the base

    SRF (Support-Resistance Flip):
      Former support level that gets broken → becomes resistance
      Former resistance that gets broken → becomes support
      "แนวรับเปลี่ยนเป็นแนวต้าน" = when price returns to broken support = SHORT

    Features:
      rbr_active    : 1.0 if there's an active RBR demand zone
      dbd_active    : 1.0 if there's an active DBD supply zone
      rbr_distance  : signed distance from close to nearest RBR zone / ATR
      srf_bull_dist : distance to nearest bullish SRF (former resistance now support)
      srf_bear_dist : distance to nearest bearish SRF (former support now resistance)
    """
    STRONG_MOVE = 1.0   # rally/drop = body > 1.0 * ATR
    BASE_CANDLES = 5    # max base width (candles)
    BASE_RANGE   = 0.5  # base height < 0.5 ATR (tight consolidation)
    MAX_AGE      = 80   # zone valid for 80 candles
    SWING_LOOKBACK = 10 # bars to find swing high/low for SRF

    n = len(df)
    opens  = df["open"].values
    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    atr    = atr_values.values

    rbr_act    = np.zeros(n, dtype=np.float32)
    dbd_act    = np.zeros(n, dtype=np.float32)
    rbr_dist   = np.zeros(n, dtype=np.float32)
    srf_b_dist = np.zeros(n, dtype=np.float32)
    srf_s_dist = np.zeros(n, dtype=np.float32)

    active_rbr: list = []  # (zl, zh, birth_idx)
    active_dbd: list = []
    active_srf_bull: list = []  # former resistance now support
    active_srf_bear: list = []  # former support now resistance

    # Track recent swing highs/lows for SRF
    recent_swing_highs: list = []
    recent_swing_lows:  list = []

    for i in range(max(BASE_CANDLES + 2, SWING_LOOKBACK), n):
        cur_atr = atr[i] if not np.isnan(atr[i]) and atr[i] > 0 else 1e-9

        # ─── Detect swing levels for SRF ───────────────────────────────
        piv = i - SWING_LOOKBACK // 2
        if piv >= SWING_LOOKBACK:
            half = SWING_LOOKBACK // 2
            is_sh = highs[piv] == max(highs[piv-half:piv+half+1])
            is_sl = lows[piv]  == min(lows[piv-half:piv+half+1])
            if is_sh:
                recent_swing_highs.append((piv, highs[piv]))
                if len(recent_swing_highs) > 8:
                    recent_swing_highs.pop(0)
            if is_sl:
                recent_swing_lows.append((piv, lows[piv]))
                if len(recent_swing_lows) > 8:
                    recent_swing_lows.pop(0)

        # ─── Detect SRF: broken support becomes resistance ──────────────
        # Bearish SRF: price closes below an old swing low (breaks support)
        if recent_swing_lows:
            for idx, (sl_bar, sl_price) in enumerate(recent_swing_lows):
                if closes[i] < sl_price and i - sl_bar > 3:
                    # Support broken → sl_price becomes resistance (bearish SRF)
                    active_srf_bear.append((sl_price - 0.5*cur_atr, sl_price + 0.3*cur_atr, i))
                    recent_swing_lows.pop(idx)
                    break
        # Bullish SRF: price closes above an old swing high (breaks resistance)
        if recent_swing_highs:
            for idx, (sh_bar, sh_price) in enumerate(recent_swing_highs):
                if closes[i] > sh_price and i - sh_bar > 3:
                    active_srf_bull.append((sh_price - 0.3*cur_atr, sh_price + 0.5*cur_atr, i))
                    recent_swing_highs.pop(idx)
                    break

        # ─── Detect RBR: Rally → Base → Rally ──────────────────────────
        body_i = closes[i] - opens[i]
        if body_i > STRONG_MOVE * cur_atr:  # Current strong bullish candle
            # Look back for base (tight range candles) before this rally
            # Then check there was a rally BEFORE the base too
            base_end = i - 1
            base_candles_idx = []
            for j in range(base_end, max(i - BASE_CANDLES - 1, 0), -1):
                range_j = highs[j] - lows[j]
                body_j = abs(closes[j] - opens[j])
                if range_j < BASE_RANGE * cur_atr:  # tight candle = base
                    base_candles_idx.append(j)
                else:
                    break

            if len(base_candles_idx) >= 1:
                # Check there was a rally before the base
                pre_base = base_candles_idx[-1] - 1
                if pre_base >= 0:
                    pre_body = closes[pre_base] - opens[pre_base]
                    if pre_body > 0.5 * cur_atr:  # bullish before base = RBR!
                        zl = min(lows[j] for j in base_candles_idx)
                        zh = max(highs[j] for j in base_candles_idx)
                        if zh - zl < 1.5 * cur_atr:  # tight zone only
                            active_rbr.append((zl, zh, i))

        # Detect DBD: Drop → Base → Drop
        if -body_i > STRONG_MOVE * cur_atr:
            base_candles_idx = []
            for j in range(i-1, max(i-BASE_CANDLES-1, 0), -1):
                range_j = highs[j] - lows[j]
                if range_j < BASE_RANGE * cur_atr:
                    base_candles_idx.append(j)
                else:
                    break
            if len(base_candles_idx) >= 1:
                pre_base = base_candles_idx[-1] - 1
                if pre_base >= 0:
                    pre_body = opens[pre_base] - closes[pre_base]
                    if pre_body > 0.5 * cur_atr:
                        zl = min(lows[j] for j in base_candles_idx)
                        zh = max(highs[j] for j in base_candles_idx)
                        if zh - zl < 1.5 * cur_atr:
                            active_dbd.append((zl, zh, i))

        # ─── Update zones (expire + fill check) ────────────────────────
        def update_zones(zones):
            surviving = []
            for zl, zh, birth in zones:
                if i - birth > MAX_AGE:
                    continue
                if zl <= closes[i] <= zh:  # filled
                    continue
                surviving.append((zl, zh, birth))
            return surviving

        active_rbr = update_zones(active_rbr)
        active_dbd = update_zones(active_dbd)
        active_srf_bear = [(zl, zh, b) for zl, zh, b in active_srf_bear if i-b <= MAX_AGE]
        active_srf_bull = [(zl, zh, b) for zl, zh, b in active_srf_bull if i-b <= MAX_AGE]

        # ─── Compute features ──────────────────────────────────────────
        if active_rbr:
            rbr_act[i] = 1.0
            nearest = min(active_rbr, key=lambda z: abs(closes[i] - (z[0]+z[1])/2))
            zl, zh, _ = nearest
            raw = closes[i] - zh if closes[i] > zh else (closes[i] - zl if closes[i] < zl else 0.0)
            rbr_dist[i] = np.clip(raw / cur_atr, -3.0, 3.0)

        if active_dbd:
            dbd_act[i] = 1.0

        # SRF distances (signed: near = agent should trade)
        if active_srf_bull:
            nearest_b = min(active_srf_bull, key=lambda z: abs(closes[i] - (z[0]+z[1])/2))
            raw = closes[i] - nearest_b[1] if closes[i] > nearest_b[1] else (closes[i] - nearest_b[0] if closes[i] < nearest_b[0] else 0.0)
            srf_b_dist[i] = np.clip(raw / cur_atr, -3.0, 3.0)
        if active_srf_bear:
            nearest_s = min(active_srf_bear, key=lambda z: abs(closes[i] - (z[0]+z[1])/2))
            raw = closes[i] - nearest_s[1] if closes[i] > nearest_s[1] else (closes[i] - nearest_s[0] if closes[i] < nearest_s[0] else 0.0)
            srf_s_dist[i] = np.clip(raw / cur_atr, -3.0, 3.0)

    df["rbr_active"]    = rbr_act
    df["dbd_active"]    = dbd_act
    df["rbr_distance"]  = rbr_dist
    df["srf_bull_dist"] = srf_b_dist
    df["srf_bear_dist"] = srf_s_dist

    return df


def _compute_fvg_features(df: pd.DataFrame, atr_values: pd.Series) -> pd.DataFrame:
    """
    Compute Fair Value Gap (FVG) features as observation signals for the RL agent.

    FVG definition (Smart Money Concepts):
      - Bullish FVG: Low[n-2] > High[n]  (gap up — price skipped a zone)
      - Bearish FVG: High[n-2] < Low[n]  (gap down)

    An FVG remains "active" until price fills it (close enters the gap zone).
    We track the nearest active FVG and compute:
      - fvg_bull_active: 1.0 if there's an unfilled bullish FVG nearby, else 0
      - fvg_bear_active: 1.0 if there's an unfilled bearish FVG nearby, else 0
      - fvg_distance:    signed distance from close to nearest FVG zone / ATR
                         (+ = above zone, - = below zone, ~0 = inside zone)
      - fvg_strength:    gap size / ATR (bigger gap = more significant)

    Max lookback for active FVGs: 20 candles (older ones are stale).
    """
    n = len(df)
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    atr = atr_values.values

    fvg_bull = np.zeros(n, dtype=np.float32)
    fvg_bear = np.zeros(n, dtype=np.float32)
    fvg_dist = np.zeros(n, dtype=np.float32)
    fvg_str = np.zeros(n, dtype=np.float32)

    # Track active FVG zones: list of (zone_low, zone_high, direction, birth_idx)
    active_zones: list = []
    MAX_FVG_AGE = 20  # candles before FVG expires

    for i in range(2, n):
        cur_atr = atr[i] if not np.isnan(atr[i]) and atr[i] > 0 else 1e-9

        # --- Detect new FVGs at candle i ---
        # Bullish FVG: candle[i-2].low > candle[i].high (gap up)
        if lows[i - 2] > highs[i]:
            zone_low = highs[i]
            zone_high = lows[i - 2]
            active_zones.append((zone_low, zone_high, 1, i))  # 1 = bullish

        # Bearish FVG: candle[i-2].high < candle[i].low (gap down)
        if highs[i - 2] < lows[i]:
            zone_low = highs[i - 2]
            zone_high = lows[i]
            active_zones.append((zone_low, zone_high, -1, i))  # -1 = bearish

        # --- Expire old zones & remove filled zones ---
        surviving = []
        for zl, zh, direction, birth in active_zones:
            age = i - birth
            if age > MAX_FVG_AGE:
                continue  # expired
            # Filled: close has entered the zone
            if zl <= closes[i] <= zh:
                continue  # filled by current candle
            surviving.append((zl, zh, direction, birth))
        active_zones = surviving

        # --- Compute features from nearest active FVG ---
        if not active_zones:
            # No active FVG
            fvg_bull[i] = 0.0
            fvg_bear[i] = 0.0
            fvg_dist[i] = 0.0
            fvg_str[i] = 0.0
        else:
            # Find nearest zone by distance from close to zone midpoint
            best_dist = float("inf")
            best_zone = active_zones[0]
            for zone in active_zones:
                zl, zh, direction, birth = zone
                mid = (zl + zh) / 2.0
                d = abs(closes[i] - mid)
                if d < best_dist:
                    best_dist = d
                    best_zone = zone

            zl, zh, direction, birth = best_zone
            gap_size = zh - zl

            # Has any active bullish / bearish FVG?
            has_bull = any(d == 1 for _, _, d, _ in active_zones)
            has_bear = any(d == -1 for _, _, d, _ in active_zones)
            fvg_bull[i] = 1.0 if has_bull else 0.0
            fvg_bear[i] = 1.0 if has_bear else 0.0

            # Distance: positive = price above zone, negative = below
            if closes[i] > zh:
                raw_dist = closes[i] - zh
            elif closes[i] < zl:
                raw_dist = closes[i] - zl  # negative
            else:
                raw_dist = 0.0  # inside zone
            fvg_dist[i] = np.clip(raw_dist / cur_atr, -3.0, 3.0)

            # Strength: gap size relative to ATR
            fvg_str[i] = np.clip(gap_size / cur_atr, 0.0, 3.0)

    df["fvg_bull_active"] = fvg_bull
    df["fvg_bear_active"] = fvg_bear
    df["fvg_distance"] = fvg_dist
    df["fvg_strength"] = fvg_str

    return df


def _find_base_zone(opens, highs, lows, closes, impulse_idx, direction, atr_val,
                    max_base_candles=6, min_impulse_atr=1.2):
    """
    Find the consolidation BASE before an impulse candle.
    This matches how traders draw D/S zones: capture the full base width.

    For DEMAND (direction=1, bullish impulse at impulse_idx):
      - Look back up to max_base_candles before impulse
      - Base = small candles (body < 0.5 ATR) that consolidate before the rally
      - Zone = min(lows of base) to max(highs of base)

    For SUPPLY (direction=-1, bearish impulse):
      - Same but for drops
    """
    base_start = impulse_idx - 1
    base_candles = []

    for j in range(impulse_idx - 1, max(impulse_idx - max_base_candles - 1, -1), -1):
        if j < 0:
            break
        body = abs(closes[j] - opens[j])
        # A base candle = small body (consolidation, not a strong directional move)
        if body < 0.6 * atr_val:
            base_candles.append(j)
        else:
            # Stop at any strong opposite candle
            if direction == 1 and closes[j] < opens[j] and body > 0.8 * atr_val:
                base_candles.append(j)
                break
            elif direction == -1 and closes[j] > opens[j] and body > 0.8 * atr_val:
                base_candles.append(j)
                break
            else:
                break  # strong same-direction = stop

    if not base_candles:
        # Fallback: use just the candle immediately before impulse
        j = impulse_idx - 1
        if j >= 0:
            base_candles = [j]
        else:
            return None, None

    # Zone = full range of base candles
    zone_low  = min(lows[j]  for j in base_candles)
    zone_high = max(highs[j] for j in base_candles)

    # Ensure zone has minimum width (at least 0.1 ATR)
    if zone_high - zone_low < 0.1 * atr_val:
        zone_high = zone_low + 0.2 * atr_val

    return zone_low, zone_high


def _compute_demand_supply_features(df: pd.DataFrame, atr_values: pd.Series) -> pd.DataFrame:
    """
    Compute Demand/Supply zone features — Base-consolidation method.

    Based on real trader chart analysis:
      - Zone = the FULL consolidation base (3-8 candles) before a strong impulse
        NOT just one candle — traders draw wide zones over the consolidation area
      - Demand zone: base before strong bullish impulse (DBR pattern)
      - Supply zone: base before strong bearish impulse (RBD pattern)
      - Entry signal: when price RETURNS and enters the zone (not just nearby)

    Features:
      - demand_active:  1.0 if bullish D/S zone exists
      - supply_active:  1.0 if bearish D/S zone exists
      - ds_distance:    signed distance from close to nearest zone edge / ATR
                        0 = inside zone (= entry zone!), positive = above, negative = below
      - ds_strength:    impulse move strength (normalized 0-1)
      - ds_freshness:   1.0 = fresh (first test), decays with retests
    """
    # FIXED: 1.2→2.0 ATR = only strong institutional moves create real zones
    # OLD: body > 1.2 ATR → average candle qualifies → 42% of bars had supply active
    # NEW: body > 2.0 ATR → only significant impulse moves → zone is meaningful
    MOVE_THRESHOLD = 2.0   # impulse = body > 2.0 * ATR (strong institutional move)
    MAX_ZONE_AGE   = 40    # FIXED: 80→40 bars = 10h (zones expire faster, stay fresh)
    MAX_TESTS      = 1     # FIXED: 2→1 = fresh zone only (first touch is strongest)

    n = len(df)
    opens  = df["open"].values
    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    atr    = atr_values.values

    demand_act = np.zeros(n, dtype=np.float32)
    supply_act = np.zeros(n, dtype=np.float32)
    ds_dist    = np.zeros(n, dtype=np.float32)
    ds_str     = np.zeros(n, dtype=np.float32)
    ds_fresh   = np.zeros(n, dtype=np.float32)

    # Active zones: (zone_low, zone_high, direction, birth_idx, test_count, strength)
    active_zones: list = []

    for i in range(5, n):
        cur_atr = atr[i] if not np.isnan(atr[i]) and atr[i] > 0 else 1e-9
        body = closes[i] - opens[i]
        body_size = abs(body)

        # --- Detect new zone when impulse candle found ---
        if body_size > MOVE_THRESHOLD * cur_atr:
            if body > 0:
                # Bullish impulse → find base below = DEMAND zone
                zl, zh = _find_base_zone(opens, highs, lows, closes, i, 1, cur_atr)
                if zl is not None:
                    strength = min(body_size / cur_atr / 3.0, 1.0)
                    active_zones.append((zl, zh, 1, i, 0, strength))
            else:
                # Bearish impulse → find base above = SUPPLY zone
                zl, zh = _find_base_zone(opens, highs, lows, closes, i, -1, cur_atr)
                if zl is not None:
                    strength = min(body_size / cur_atr / 3.0, 1.0)
                    active_zones.append((zl, zh, -1, i, 0, strength))

        # --- Update zones: expiry, tests, broken ---
        surviving = []
        for zl, zh, direction, birth, tests, strength in active_zones:
            age = i - birth
            if age > MAX_ZONE_AGE:
                continue  # expired

            # Check if price entered the zone this candle
            if lows[i] <= zh and highs[i] >= zl:
                tests += 1
                if tests > MAX_TESTS:
                    continue  # depleted

            # Check if zone is broken (close through zone)
            if direction == 1 and closes[i] < zl:  # demand broken
                continue
            if direction == -1 and closes[i] > zh:  # supply broken
                continue

            surviving.append((zl, zh, direction, birth, tests, strength))
        active_zones = surviving

        # --- Compute features from nearest zone ---
        if not active_zones:
            continue

        # Find nearest zone by distance from close to zone midpoint
        best_dist = float("inf")
        best_zone = active_zones[0]
        for zone in active_zones:
            zl, zh, d, b, t, s = zone
            mid = (zl + zh) / 2.0
            dist = abs(closes[i] - mid)
            if dist < best_dist:
                best_dist = dist
                best_zone = zone

        zl, zh, direction, birth, tests, strength = best_zone

        has_demand = any(d == 1 for _, _, d, _, _, _ in active_zones)
        has_supply = any(d == -1 for _, _, d, _, _, _ in active_zones)
        demand_act[i] = 1.0 if has_demand else 0.0
        supply_act[i] = 1.0 if has_supply else 0.0

        # Distance: positive = price above zone, negative = below
        if closes[i] > zh:
            raw_dist = closes[i] - zh
        elif closes[i] < zl:
            raw_dist = closes[i] - zl
        else:
            raw_dist = 0.0  # inside zone
        ds_dist[i] = np.clip(raw_dist / cur_atr, -3.0, 3.0)

        # Strength: how strong was the creating move (normalized 0-1)
        ds_str[i] = np.clip(strength / 5.0, 0.0, 1.0)

        # Freshness: 1.0 = untested, decays by 0.33 per test
        ds_fresh[i] = max(0.0, 1.0 - tests * 0.33)

    df["demand_active"] = demand_act
    df["supply_active"] = supply_act
    df["ds_distance"] = ds_dist
    df["ds_strength"] = ds_str
    df["ds_freshness"] = ds_fresh

    return df


def prepare_features(df: pd.DataFrame, reset_index: bool = True) -> pd.DataFrame:
    """
    เตรียม technical features จาก OHLC[V] DataFrame
    คาดหวัง columns: open, high, low, close (volume optional)
    คืน DataFrame ที่ dropna แล้วและมี FEATURE_COLUMNS ครบ

    reset_index=False: ใช้สำหรับ multi-timeframe merge (ต้องเก็บ datetime index ไว้)
    """
    from ta.trend import SMAIndicator, MACD
    from ta.momentum import RSIIndicator
    from ta.volatility import BollingerBands, AverageTrueRange

    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns.str.lower())
    if missing:
        raise ValueError(f"DataFrame missing OHLC columns: {missing}")

    df = df.copy()
    df.columns = df.columns.str.lower()

    # Returns (normalized)
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

    # Momentum
    df["rsi"] = RSIIndicator(df["close"], window=14).rsi() / 100.0  # 0-1

    # Trend
    df["sma_20"] = SMAIndicator(df["close"], window=20).sma_indicator()
    df["sma_50"] = SMAIndicator(df["close"], window=50).sma_indicator()
    df["sma_ratio"] = df["sma_20"] / df["sma_50"]

    macd = MACD(df["close"])
    # normalize MACD diff ด้วย close price (scale-invariant)
    df["macd_diff"] = macd.macd_diff() / df["close"]

    # Volatility
    bb = BollingerBands(df["close"])
    bb_high = bb.bollinger_hband()
    bb_low = bb.bollinger_lband()
    df["bb_position"] = (df["close"] - bb_low) / (bb_high - bb_low + 1e-9)

    atr = AverageTrueRange(df["high"], df["low"], df["close"])
    atr_values = atr.average_true_range()
    df["atr_ratio"] = atr_values / df["close"]

    # --- Fair Value Gap (FVG) features ---
    df = _compute_fvg_features(df, atr_values)

    # --- Demand/Supply zone features (institutional order flow) ---
    df = _compute_demand_supply_features(df, atr_values)

    # --- SMC Market Structure: CHOCH + Swing HH/HL/LH/LL ---
    df = _compute_structure_features(df, atr_values)

    # --- RBR/DBD zones + SRF (Support-Resistance Flip) ---
    df = _compute_rbr_srf_features(df, atr_values)

    # --- Session feature ---
    # Gold sessions: Asian 00-08 UTC (low vol), London 08-13 (breakout),
    # NY overlap 13-17 (highest vol), NY late 17-22 (quiet)
    # Binary: 1 = active session (London + NY overlap), 0 = quiet (Asian + late)
    if hasattr(df.index, 'hour'):
        hour = df.index.hour
    elif "time" in df.columns:
        hour = pd.to_datetime(df["time"]).dt.hour
    else:
        hour = pd.Series(np.zeros(len(df), dtype=int), index=df.index)

    df["is_active_session"] = ((hour >= 8) & (hour < 17)).astype(np.float32)

    # --- Candle body ratio (momentum indicator) ---
    # 1.0 = full body marubozu (strong conviction)
    # 0.0 = doji (complete indecision)
    candle_range = df["high"] - df["low"]
    candle_body = (df["close"] - df["open"]).abs()
    df["candle_body_ratio"] = (candle_body / (candle_range + 1e-9)).clip(0.0, 1.0).astype(np.float32)

    # --- Engulfing pattern detection (vectorized) ---
    # Bearish engulfing: prev candle bullish → curr candle bearish AND body fully engulfs prev body
    #   Classic reversal at supply: "ชนโซนแล้วเกิด Engulfing เข้า sell"
    #   Oracle: H1 bear + supply + pullback + Engulfing → WR 34.6% (+10.8% vs baseline)
    # Bullish engulfing: symmetric for demand zone reversal entries
    prev_close = df["close"].shift(1)
    prev_open  = df["open"].shift(1)

    prev_bullish = prev_close > prev_open
    curr_bearish = df["close"] < df["open"]
    bear_engulfs = (df["open"] >= prev_close) & (df["close"] <= prev_open)
    df["engulfing_bear"] = (prev_bullish & curr_bearish & bear_engulfs).astype(np.float32)

    prev_bearish = prev_close < prev_open
    curr_bullish = df["close"] > df["open"]
    bull_engulfs = (df["open"] <= prev_close) & (df["close"] >= prev_open)
    df["engulfing_bull"] = (prev_bearish & curr_bullish & bull_engulfs).astype(np.float32)

    # --- M15 Pullback Strength (3-bar normalized price change) ---
    # Measures: how many ATR units has price moved over the last 3 bars (45 min on M15)
    # +1.0 = price rose 1 ATR   → strong pullback UP into supply zone
    # -1.0 = price fell 1 ATR   → strong pullback DOWN into demand zone
    #  0.0 = price flat (sideways, no directional pullback)
    # Used in mask Path 2: supply reversal needs pullback_val > 0.3 ATR (price came from below)
    price_change_3 = (df["close"] - df["close"].shift(3)).fillna(0.0)
    atr_price = (df["atr_ratio"] * df["close"]).replace(0.0, 1e-9)
    df["m15_pullback"] = (price_change_3 / atr_price).clip(-3.0, 3.0).astype(np.float32)

    df = df.dropna()
    if reset_index:
        df = df.reset_index(drop=True)
    return df


def compute_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    คำนวณ daily-timeframe features จาก daily OHLC[V] DataFrame
    คืน DataFrame ที่มีเฉพาะ DAILY_FEATURE_COLUMNS + datetime index
    """
    from ta.trend import SMAIndicator, MACD
    from ta.momentum import RSIIndicator
    from ta.volatility import BollingerBands, AverageTrueRange

    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    df["d_returns_1d"] = df["close"].pct_change()
    df["d_returns_5d"] = df["close"].pct_change(5)  # 5-day momentum
    df["d_rsi"] = RSIIndicator(df["close"], window=14).rsi() / 100.0

    sma_20 = SMAIndicator(df["close"], window=20).sma_indicator()
    sma_50 = SMAIndicator(df["close"], window=50).sma_indicator()
    df["d_sma_20_ratio"] = df["close"] / sma_20
    df["d_sma_50_ratio"] = df["close"] / sma_50

    macd = MACD(df["close"])
    df["d_macd_diff"] = macd.macd_diff() / df["close"]

    bb = BollingerBands(df["close"])
    bb_high = bb.bollinger_hband()
    bb_low = bb.bollinger_lband()
    df["d_bb_position"] = (df["close"] - bb_low) / (bb_high - bb_low + 1e-9)

    atr = AverageTrueRange(df["high"], df["low"], df["close"])
    df["d_atr_ratio"] = atr.average_true_range() / df["close"]

    # Composite trend strength: 0 = no trend, +/-1 = strong trend (up/down)
    sma_alignment = (sma_20 - sma_50) / df["close"]  # ปกติอยู่ในช่วง +/-0.05
    macd_normalized = df["d_macd_diff"]               # ปกติอยู่ในช่วง +/-0.02
    df["d_trend_strength"] = (sma_alignment * 10 + macd_normalized * 25).clip(-1.0, 1.0)

    return df[DAILY_FEATURE_COLUMNS].dropna()


def merge_daily_into_primary(
    df_primary: pd.DataFrame,
    df_daily_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge daily features เข้า primary (4h) DataFrame โดย:
    - shift(1) ของ daily — ใช้ daily bar ของ 'เมื่อวาน' ที่ปิดสมบูรณ์แล้ว -> ไม่ look-ahead
    - reindex + forward-fill เข้า primary timeline -> ทุก 4h bar รู้ macro state ปัจจุบัน

    Both DataFrames ต้องมี datetime index
    """
    # Strip timezone ถ้ามี mismatch
    if df_primary.index.tz is not None:
        df_primary = df_primary.copy()
        df_primary.index = df_primary.index.tz_localize(None)
    if df_daily_features.index.tz is not None:
        df_daily_features = df_daily_features.copy()
        df_daily_features.index = df_daily_features.index.tz_localize(None)

    # Safety shift: daily bar 'เมื่อวาน' ที่ปิดสมบูรณ์แล้วเท่านั้น
    daily_safe = df_daily_features.shift(1).dropna()

    # Forward-fill เข้า primary timeline
    daily_aligned = daily_safe.reindex(df_primary.index, method="ffill")

    # Concat columns
    merged = pd.concat([df_primary, daily_aligned], axis=1)
    merged = merged.dropna()
    return merged


def compute_h1_features(df_m15: pd.DataFrame) -> pd.DataFrame:
    """
    คำนวณ H1 features โดย resample จาก M15 DataFrame.

    M15 df ต้องมี datetime index และ columns: open, high, low, close

    H1 ให้ trend signal ที่ timely กว่า D1 สำหรับ M15 scalping:
      - 20 H1 bars = ~1 day (vs D1 SMA20 = 20 วัน)
      - H1 trend สะท้อน intraday structure ที่ agent เทรดอยู่
      - ไม่ slow เกิน (D1) ไม่ noisy เกิน (M15)

    คืน DataFrame ที่มีเฉพาะ H1_FEATURE_COLUMNS + datetime index
    """
    from ta.trend import SMAIndicator, MACD
    from ta.momentum import RSIIndicator
    from ta.volatility import BollingerBands, AverageTrueRange

    # Resample M15 → H1
    df_m15 = df_m15.copy()
    df_m15.columns = [c.lower() for c in df_m15.columns]

    # Keep only OHLC for resampling (volume optional)
    ohlc_cols = {c: "first" if c == "open" else
                    "max"   if c == "high" else
                    "min"   if c == "low"  else
                    "last"  if c == "close" else
                    "sum"   for c in df_m15.columns if c in ("open","high","low","close","volume")}

    df_h1 = df_m15.resample("1h").agg(ohlc_cols).dropna()
    df_h1.columns = [c.lower() for c in df_h1.columns]

    if len(df_h1) < 60:
        raise ValueError(f"H1 data too short ({len(df_h1)} bars) — need ≥60 for indicators")

    # Returns
    df_h1["h1_returns_1h"] = df_h1["close"].pct_change()
    df_h1["h1_returns_4h"] = df_h1["close"].pct_change(4)   # 4H momentum

    # RSI(14) on H1
    df_h1["h1_rsi"] = RSIIndicator(df_h1["close"], window=14).rsi() / 100.0

    # SMA: 20 H1 ≈ 1 day, 50 H1 ≈ 2 days
    sma_20 = SMAIndicator(df_h1["close"], window=20).sma_indicator()
    sma_50 = SMAIndicator(df_h1["close"], window=50).sma_indicator()
    df_h1["h1_sma_20_ratio"] = df_h1["close"] / sma_20
    df_h1["h1_sma_50_ratio"] = df_h1["close"] / sma_50

    # MACD
    macd = MACD(df_h1["close"])
    df_h1["h1_macd_diff"] = macd.macd_diff() / df_h1["close"]

    # Bollinger Bands
    bb = BollingerBands(df_h1["close"])
    bb_high = bb.bollinger_hband()
    bb_low  = bb.bollinger_lband()
    df_h1["h1_bb_position"] = (df_h1["close"] - bb_low) / (bb_high - bb_low + 1e-9)

    # ATR
    atr = AverageTrueRange(df_h1["high"], df_h1["low"], df_h1["close"])
    df_h1["h1_atr_ratio"] = atr.average_true_range() / df_h1["close"]

    # Composite trend: same formula as daily but on H1
    sma_align = (sma_20 - sma_50) / df_h1["close"]
    macd_norm  = df_h1["h1_macd_diff"]
    df_h1["h1_trend_strength"] = (sma_align * 10 + macd_norm * 25).clip(-1.0, 1.0)

    # ── H1 Trend Consistency (rolling 8 H1 bars = 8h window) ──────────────
    # Root cause fix: bad folds have H1 neutral 72-80% → CHOCH fires on noise
    #   Fold3 (WR=10%): H1 bull only 10.1% → can't get 60% consecutive
    #   Fold7 (WR=39%): H1 bull 23.3% → extended bull periods → 60% easy
    # Using mask_trend_threshold=0.05 (default, matches config)
    _CONS_THRESHOLD = 0.05
    _CONS_WINDOW    = 8     # 8 H1 bars = 8 hours
    df_h1["h1_bull_consistency"] = (
        (df_h1["h1_trend_strength"] > _CONS_THRESHOLD)
        .rolling(_CONS_WINDOW, min_periods=4)
        .mean()
        .fillna(0.0)
        .astype(np.float32)
    )
    df_h1["h1_bear_consistency"] = (
        (df_h1["h1_trend_strength"] < -_CONS_THRESHOLD)
        .rolling(_CONS_WINDOW, min_periods=4)
        .mean()
        .fillna(0.0)
        .astype(np.float32)
    )

    return df_h1[H1_FEATURE_COLUMNS].dropna()


def merge_h1_into_primary(
    df_primary: pd.DataFrame,
    df_h1_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge H1 features เข้า M15 DataFrame โดย:
    - shift(1) ของ H1 — ใช้ H1 bar ที่ปิดสมบูรณ์แล้ว -> ไม่ look-ahead
      (H1 bar 09:00-10:00 ใช้ใน M15 bars ของ 10:00 เป็นต้นไป)
    - reindex + forward-fill เข้า M15 timeline
      (ทุก M15 bar ในชั่วโมงเดียวกัน ใช้ H1 bar ก่อนหน้า)

    Both DataFrames ต้องมี datetime index
    """
    # Strip timezone
    if df_primary.index.tz is not None:
        df_primary = df_primary.copy()
        df_primary.index = df_primary.index.tz_localize(None)
    if df_h1_features.index.tz is not None:
        df_h1_features = df_h1_features.copy()
        df_h1_features.index = df_h1_features.index.tz_localize(None)

    # shift(1): use previous completed H1 bar (no look-ahead)
    h1_safe = df_h1_features.shift(1).dropna()

    # Forward-fill into M15 timeline
    h1_aligned = h1_safe.reindex(df_primary.index, method="ffill")

    # Merge
    merged = pd.concat([df_primary, h1_aligned], axis=1)
    merged = merged.dropna()
    return merged


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class GoldTradingEnv(gym.Env):
    """
    XAUUSD Trading Environment — Sniper Scalping Mode

    Actions (Discrete(3)):
        0 = Hold   (do nothing)
        1 = Buy    (open long)  — invalid if already in position
        2 = Sell   (open short) — invalid if already in position

    NO manual Close — trades are auto-closed by TP/SL:
        SL = sl_atr_mult × ATR  (default 1.5)
        TP = tp_atr_mult × ATR  (default 4.5)  → RR 1:3

    The agent learns WHEN and WHICH DIRECTION to enter.
    The env handles exits via structure (TP/SL).

    Observation (Box):
        [window_size * n_features] flatten + [position, unrealized_pnl_pct, duration_norm]

    PnL formula:
        pnl_usd = (exit_price - entry_price) * direction * lot_size * contract_size
        XAUUSD: contract_size = 100 oz/lot, mini lot 0.01 -> $1 per $1 price move
    """
    metadata = {"render_modes": ["human"]}

    # Action constants
    HOLD, BUY, SELL = 0, 1, 2

    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 10_000.0,
        lot_size: float = 0.01,
        spread: float = 0.30,
        commission: float = 0.07,
        contract_size: float = 100.0,
        max_position: int = 1,
        window_size: int = 50,
        reward_type: str = "scalp_sniper",
        max_drawdown_pct: float = 0.5,
        feature_columns: Optional[list] = None,
        trade_cooldown: int = 0,
        sl_atr_mult: float = 1.5,
        tp_atr_mult: float = 4.5,
        use_lstm: bool = False,
        use_action_mask: bool = False,      # True = hard rule-based entry filter
        mask_trend_threshold: float = 0.3,  # min |d_trend_strength| for valid entry
        mask_zone_threshold: float = 1.5,   # max |ds_distance| for valid entry
        mask_require_session: bool = True,  # require active session for entry
    ):
        super().__init__()

        # Auto-detect feature columns (M15 + H1 + D1 if available)
        if feature_columns is None:
            all_known = FEATURE_COLUMNS + H1_FEATURE_COLUMNS + DAILY_FEATURE_COLUMNS
            feature_columns = [c for c in all_known if c in df.columns]
            if not feature_columns:
                raise ValueError(
                    "DataFrame ไม่มี feature columns ที่รู้จักเลย "
                    "เรียก prepare_features(df) ก่อน"
                )

        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame missing feature columns: {missing}")
        min_rows = window_size + 10 if not use_lstm else 100
        if len(df) <= min_rows:
            raise ValueError(
                f"DataFrame สั้นเกินไป (len={len(df)}) "
                f"ต้องมากกว่า {min_rows}"
            )

        self.df = df.reset_index(drop=True)
        self.feature_columns = feature_columns

        # config
        self.initial_balance = float(initial_balance)
        self.lot_size = float(lot_size)
        self.spread = float(spread)
        self.commission = float(commission)
        self.contract_size = float(contract_size)
        self.max_position = int(max_position)
        self.window_size = int(window_size)
        self.reward_type = reward_type
        self.max_drawdown_pct = float(max_drawdown_pct)
        self.trade_cooldown = int(trade_cooldown)
        self.use_lstm = bool(use_lstm)

        # TP/SL config (ATR-based, fixed exit only)
        self.sl_atr_mult = float(sl_atr_mult)
        self.tp_atr_mult = float(tp_atr_mult)

        # Action masking config
        self.use_action_mask = bool(use_action_mask)
        self.mask_trend_threshold = float(mask_trend_threshold)
        self.mask_zone_threshold = float(mask_zone_threshold)
        self.mask_require_session = bool(mask_require_session)

        # Reward fn
        self._reward_fn = get_reward_fn(reward_type)

        # Spaces: 3 actions (Hold, Buy, Sell) — no manual Close
        n_features = len(feature_columns)
        if self.use_lstm:
            # LSTM mode: single-step features + position info (no window)
            obs_dim = n_features + 3
        else:
            # MLP mode: flattened window + position info
            obs_dim = self.window_size * n_features + 3
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Pre-extract data arrays for speed
        self._features = self.df[feature_columns].to_numpy(dtype=np.float32)
        self._closes = self.df["close"].to_numpy(dtype=np.float64)
        self._highs = self.df["high"].to_numpy(dtype=np.float64)
        self._lows = self.df["low"].to_numpy(dtype=np.float64)

        # Pre-compute ATR values for TP/SL (use atr_ratio * close to get ATR in USD)
        if "atr_ratio" in self.df.columns:
            self._atr = (self.df["atr_ratio"] * self.df["close"]).to_numpy(dtype=np.float64)
        else:
            # fallback: simple ATR from high-low
            self._atr = (self.df["high"] - self.df["low"]).rolling(14).mean().to_numpy(dtype=np.float64)

        # init state
        self.reset()

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0          # -1 short / 0 flat / 1 long
        self.entry_price = 0.0
        self.entry_step = 0
        self._tp_price = 0.0       # take profit level
        self._sl_price = 0.0       # stop loss level
        self.total_trades = 0
        self.winning_trades = 0
        self.trade_returns: list[float] = []
        self.equity_curve: list[float] = [self.initial_balance]

        # peak for drawdown stop
        self._peak_equity = self.initial_balance

        # cooldown
        self._cooldown_remaining = 0

        # reset reward state
        self._reward_state: dict = {}

        return self._get_observation(), self._get_info()

    def step(self, action: int):
        action = int(action)
        current_price = float(self._closes[self.current_step])
        current_high = float(self._highs[self.current_step])
        current_low = float(self._lows[self.current_step])

        invalid_action = False
        trade_closed = False

        # ---- Cooldown tick ----
        in_cooldown = self._cooldown_remaining > 0
        if in_cooldown:
            self._cooldown_remaining -= 1

        # ---- Check TP/SL FIRST (before new actions) ----
        if self.position != 0:
            hit_tp = False
            hit_sl = False

            if self.position == 1:  # Long
                if current_high >= self._tp_price:
                    hit_tp = True
                    exit_price = self._tp_price
                elif current_low <= self._sl_price:
                    hit_sl = True
                    exit_price = self._sl_price
            else:  # Short
                if current_low <= self._tp_price:
                    hit_tp = True
                    exit_price = self._tp_price
                elif current_high >= self._sl_price:
                    hit_sl = True
                    exit_price = self._sl_price

            if hit_tp or hit_sl:
                pnl = (exit_price - self.entry_price) * self.position \
                      * self.lot_size * self.contract_size
                self.balance += pnl - self.commission
                self.trade_returns.append(pnl / self.initial_balance)

                self.total_trades += 1
                if pnl > 0:
                    self.winning_trades += 1

                trade_closed = True
                self.position = 0
                self.entry_price = 0.0
                self._tp_price = 0.0
                self._sl_price = 0.0

        # ---- Execute action (only if flat) ----
        # Direction-invariant: BUY = "trade with trend"
        # In bearish trend, BUY actually opens a SHORT position
        trend_dir = self._get_trend_direction()
        if action == self.BUY and trend_dir == -1:
            action = self.SELL  # remap BUY → SHORT when bearish

        if action == self.BUY:
            if self.position == 0 and not in_cooldown:
                atr = float(self._atr[self.current_step])
                if atr > 0:
                    self.position = 1
                    self.entry_price = current_price + self.spread / 2.0
                    self.entry_step = self.current_step
                    self.balance -= self.commission

                    # Set TP/SL levels (fixed)
                    self._sl_price = self.entry_price - self.sl_atr_mult * atr
                    self._tp_price = self.entry_price + self.tp_atr_mult * atr
            else:
                invalid_action = True

        elif action == self.SELL:
            if self.position == 0 and not in_cooldown:
                atr = float(self._atr[self.current_step])
                if atr > 0:
                    self.position = -1
                    self.entry_price = current_price - self.spread / 2.0
                    self.entry_step = self.current_step
                    self.balance -= self.commission

                    # Set TP/SL levels (fixed, reversed for short)
                    self._sl_price = self.entry_price + self.sl_atr_mult * atr
                    self._tp_price = self.entry_price - self.tp_atr_mult * atr
            else:
                invalid_action = True

        # ---- Update equity curve ----
        equity = self.balance
        if self.position != 0:
            if self.position == 1:
                unrealized = (current_price - self.spread / 2.0 - self.entry_price) \
                             * self.lot_size * self.contract_size
            else:
                unrealized = (self.entry_price - current_price - self.spread / 2.0) \
                             * self.lot_size * self.contract_size
            equity += unrealized
        self.equity_curve.append(equity)
        if equity > self._peak_equity:
            self._peak_equity = equity

        # ---- Reward ----
        reward = float(self._reward_fn(
            self,
            {"trade_closed": trade_closed, "action": action, "invalid_action": invalid_action},
        ))

        # ---- Advance ----
        self.current_step += 1

        # ---- Termination ----
        end_of_data = self.current_step >= len(self.df) - 1
        bankrupt = equity <= self.initial_balance * (1.0 - self.max_drawdown_pct)
        terminated = bool(end_of_data or bankrupt)
        truncated = False

        # Close position at end (mark-to-market)
        if terminated and self.position != 0:
            self.balance = equity
            self.position = 0

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    # ------------------------------------------------------------------
    # Action Masking (for MaskablePPO)
    # ------------------------------------------------------------------
    def action_masks(self) -> np.ndarray:
        """
        Hard rule-based entry filter — called by MaskablePPO each step.

        Returns bool array [Hold_ok, Buy_ok, Sell_ok].

        BIDIRECTIONAL: LONG when H1 bull+demand, SHORT when H1 bear+supply.
        Same quality conditions applied symmetrically to both directions.

        Previous SHORT WR was poor (23.1%) because features were wrong:
          - supply_active: fired 42% of bars (MOVE_THRESHOLD too low = 1.2 ATR)
          - CHOCH: fired 34.8% of bars (5-bar micro pivot, 10h memory)
        Fixed in this version:
          - supply_active: MOVE_THRESHOLD 1.2→2.0, MAX_TESTS 2→1
          - CHOCH: SWING_WINDOW 5→20, MEMORY 40→8, + candle quality

          Conditions (ALL required):
            1. H1 trend direction (|h1_trend| > threshold → defines LONG or SHORT)
            2. Zone aligned with trend (demand for LONG, supply for SHORT)
            3. Momentum candle in trend direction (body_ratio > 0.4)

          Agent observation includes H1 features, SRF, CHOCH, SMA → agent learns
          WHICH demand zone + H1 trend setups to actually enter (further filtering).

        If already in position → only Hold
        If action masking disabled → all actions valid
        """
        if not self.use_action_mask:
            return np.array([True, True, True], dtype=bool)

        # In position → only hold (no pyramiding)
        if self.position != 0:
            return np.array([True, False, False], dtype=bool)

        step = self.current_step

        # --- Condition 1: H1 trend direction — CONSISTENCY check (not snapshot) ---
        # FIX: snapshot (single bar) caused trading in sideways H1
        #   Bad folds: H1 neutral 72-80% → brief bull spikes → false CHOCH signal
        #   Fix: require 60%+ of last 8 H1 bars (8h) in same direction
        #   Bad fold H1 bull 10% → can't sustain 60% → blocked
        #   Good fold H1 bull 23% → extended bull runs → passes easily
        trend_col = "h1_trend_strength" if "h1_trend_strength" in self.df.columns \
                    else "d_trend_strength"
        if trend_col not in self.df.columns:
            return np.array([True, False, False], dtype=bool)

        if "h1_bull_consistency" in self.df.columns:
            # Precomputed rolling 8-H1-bar consistency (fast O(1) lookup)
            bull_cons = float(self.df["h1_bull_consistency"].iloc[step])
            bear_cons = float(self.df["h1_bear_consistency"].iloc[step])
            trend_bullish = bull_cons >= 0.60   # 6/8 H1 bars = real bull trend
            trend_bearish = bear_cons >= 0.60   # 6/8 H1 bars = real bear trend
        else:
            # Fallback: snapshot (old behaviour)
            h1_trend = float(self.df[trend_col].iloc[step])
            trend_bullish = h1_trend >  self.mask_trend_threshold
            trend_bearish = h1_trend < -self.mask_trend_threshold

        if not (trend_bullish or trend_bearish):
            return np.array([True, False, False], dtype=bool)

        # --- Candle body / direction (used by both paths) ---
        can_long  = False
        can_short = False

        if "candle_body_ratio" in self.df.columns:
            body_ratio  = float(self.df["candle_body_ratio"].iloc[step])
            close_val   = float(self.df["close"].iloc[step])
            open_val    = float(self.df["open"].iloc[step])
            bullish_bar = (close_val > open_val) and (body_ratio > 0.4)
            bearish_bar = (close_val < open_val) and (body_ratio > 0.4)
        else:
            bullish_bar = True
            bearish_bar = True

        # ─────────────────────────────────────────────────────────────────
        # PATH 1: CHOCH Momentum (oracle-proven)
        #   LONG:  H1 bull + CHOCH bull + bullish body → WR 28.3% (+3.4%)
        #   SHORT: H1 bear + CHOCH bear + bearish body → WR 29.3% (+5.5%)
        #   "CHOCH = ราคาทะลุ swing high/low = momentum break"
        #   Agent learns from demand/supply/SRF in obs → further filters internally
        # ─────────────────────────────────────────────────────────────────
        if "choch_bullish" in self.df.columns:
            choch_bull = float(self.df["choch_bullish"].iloc[step])
            choch_bear = float(self.df["choch_bearish"].iloc[step])

            if trend_bullish and (choch_bull > 0.1) and bullish_bar:
                can_long = True   # LONG: momentum breakout confirmed
            if trend_bearish and (choch_bear > 0.1) and bearish_bar:
                can_short = True  # SHORT: momentum breakdown confirmed

        # ─────────────────────────────────────────────────────────────────
        # PATH 2: Supply Reversal SHORT only (oracle-proven)
        #
        #   SHORT: H1 bear + supply_active + pullback_up + bearish body
        #     Oracle (snapshot): WR 28.3% (+4.5%) — 138 trades
        #     Oracle (cons60%):  WR 24.2% (+0.4%) — 95 trades
        #     Using snapshot H1 for this path (consistency hurts supply reversal
        #     because best entry is at START of bear trend, not after 8h consistent)
        #     "ราคาขึ้นมาชน supply แล้ว reverse ลง" (trader charts 1,2,4)
        #
        #   LONG demand reversal: REMOVED
        #     Oracle: WR 18.6% (-6.3%) = WORSE than random [BAD]
        #     Reason: "price drops to demand in uptrend" often breaks through
        #     demand entirely → trade enters, price continues down → SL hit
        #     Agent still sees demand_active in observation → learns contextually
        # ─────────────────────────────────────────────────────────────────
        if "m15_pullback" in self.df.columns:
            pullback_val = float(self.df["m15_pullback"].iloc[step])

            # SHORT supply reversal: use snapshot H1 (not consistency — see above)
            # Check raw h1_trend_strength for this path only
            if (bearish_bar and "supply_active" in self.df.columns):
                h1_raw = float(self.df[trend_col].iloc[step])
                supply_val = float(self.df["supply_active"].iloc[step])
                if h1_raw < -self.mask_trend_threshold and supply_val > 0.5 and pullback_val > 0.3:
                    can_short = True  # SHORT: pullback into supply → reject

        if not (can_long or can_short):
            return np.array([True, False, False], dtype=bool)

        # --- Condition 4: Active session (optional) ---
        if self.mask_require_session and "is_active_session" in self.df.columns:
            if float(self.df["is_active_session"].iloc[step]) < 0.5:
                return np.array([True, False, False], dtype=bool)

        # --- All conditions met ---
        # BUY = direction-invariant entry:
        #   can_long  → _get_trend_direction()=+1 → BUY stays BUY → LONG
        #   can_short → _get_trend_direction()=-1 → BUY remapped to SELL → SHORT
        can_buy = can_long or can_short  # type: ignore[possibly-undefined]
        return np.array([True, can_buy, False], dtype=bool)

    # ------------------------------------------------------------------
    # Direction-Invariant Helpers
    # ------------------------------------------------------------------
    def _get_trend_direction(self) -> int:
        """
        Returns +1 (execute BUY as LONG) or -1 (execute BUY as SHORT).

        BIDIRECTIONAL: H1 trend determines direction for BOTH LONG and SHORT.
        - H1 bullish (> threshold) → BUY = LONG  → observation NOT flipped
        - H1 bearish (< -threshold) → BUY remapped to SELL = SHORT → observation FLIPPED
        - Direction-invariant: agent always 'sees' a bullish setup regardless of direction
        """
        trend_col = "h1_trend_strength" if "h1_trend_strength" in self.df.columns \
                    else "d_trend_strength"
        if trend_col in self.df.columns:
            t = float(self.df[trend_col].iloc[self.current_step])
            if t > self.mask_trend_threshold:
                return 1   # H1 bullish → LONG
            if t < -self.mask_trend_threshold:
                return -1  # H1 bearish → SHORT
        return 0  # no clear trend → hold (mask blocks entry)

    def _flip_features_bearish(self, feat_window: np.ndarray) -> np.ndarray:
        """
        Flip feature window so bearish market looks identical to bullish.
        Agent always sees the same 'long setup' pattern — no direction confusion.

        feat_window shape: (window_size, n_features)
        n_features includes BOTH M15 and daily features (they're merged into each row).
        """
        fc = self.feature_columns
        out = feat_window.copy()

        def idx(name):
            return fc.index(name) if name in fc else None

        # --- M15 directional features ---
        for col in ["returns", "macd_diff"]:
            i = idx(col)
            if i is not None:
                out[:, i] *= -1

        for col in ["rsi", "bb_position"]:
            i = idx(col)
            if i is not None:
                out[:, i] = 1.0 - out[:, i]

        # Swap demand ↔ supply
        id_d, id_s = idx("demand_active"), idx("supply_active")
        if id_d is not None and id_s is not None:
            out[:, id_d], out[:, id_s] = feat_window[:, id_s].copy(), feat_window[:, id_d].copy()

        # Swap fvg_bull ↔ fvg_bear
        id_fb, id_fs = idx("fvg_bull_active"), idx("fvg_bear_active")
        if id_fb is not None and id_fs is not None:
            out[:, id_fb], out[:, id_fs] = feat_window[:, id_fs].copy(), feat_window[:, id_fb].copy()

        # Negate signed distances
        for col in ["ds_distance", "fvg_distance"]:
            i = idx(col)
            if i is not None:
                out[:, i] *= -1

        # --- Daily features (already merged into each row of window) ---
        for col in ["d_macd_diff", "d_returns_1d", "d_returns_5d", "d_trend_strength"]:
            i = idx(col)
            if i is not None:
                out[:, i] *= -1

        for col in ["d_rsi", "d_bb_position"]:
            i = idx(col)
            if i is not None:
                out[:, i] = 1.0 - out[:, i]

        # --- H1 features (same flip logic as daily) ---
        for col in ["h1_macd_diff", "h1_returns_1h", "h1_returns_4h", "h1_trend_strength"]:
            i = idx(col)
            if i is not None:
                out[:, i] *= -1

        for col in ["h1_rsi", "h1_bb_position"]:
            i = idx(col)
            if i is not None:
                out[:, i] = 1.0 - out[:, i]

        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_observation(self) -> np.ndarray:
        trend_dir = self._get_trend_direction()

        if self.use_lstm:
            features = self._features[self.current_step:self.current_step + 1]  # (1, n_feat)
        else:
            start = self.current_step - self.window_size
            end = self.current_step
            features = self._features[start:end]  # (window, n_feat)

        # Direction-invariant: flip features for bearish trend
        if trend_dir == -1:
            features = self._flip_features_bearish(features)

        features_flat = features.flatten()

        if self.position != 0:
            current_price = float(self._closes[self.current_step])
            unrealized_pnl_pct = (current_price - self.entry_price) / self.entry_price * self.position
            duration = (self.current_step - self.entry_step) / 100.0
        else:
            unrealized_pnl_pct = 0.0
            duration = 0.0

        # Flip position sign when bearish (agent always sees itself as "long")
        effective_position = float(self.position) * trend_dir if trend_dir != 0 else float(self.position)

        position_info = np.array(
            [effective_position, float(unrealized_pnl_pct), float(duration)],
            dtype=np.float32,
        )

        obs = np.concatenate([features_flat.astype(np.float32), position_info])
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def _get_info(self) -> dict:
        win_rate = self.winning_trades / max(self.total_trades, 1)
        equity = self.equity_curve[-1] if self.equity_curve else self.balance
        return {
            "balance": self.balance,
            "equity": equity,
            "total_trades": self.total_trades,
            "win_rate": win_rate,
            "position": self.position,
            "step": self.current_step,
        }

    def render(self):
        info = self._get_info()
        print(
            f"step={info['step']:5d} | pos={info['position']:+d} | "
            f"bal={info['balance']:.2f} | eq={info['equity']:.2f} | "
            f"trades={info['total_trades']} | wr={info['win_rate']:.2%}"
        )
