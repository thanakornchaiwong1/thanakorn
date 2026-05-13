"""
Reward functions for XAUUSD trading environment.

Registry pattern: env เลือก reward ตามชื่อจาก config (`reward_type`)
- ทุก reward function รับ (env, ctx) คืน float
- env._reward_state เก็บ running stats ของ reward ที่ stateful (เช่น differential Sharpe)
  -> ต้อง reset ใน env.reset()
"""
from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def pnl_reward(env, ctx: Dict[str, Any]) -> float:
    """
    Reward = equity change ต่อ step / initial_balance
    ข้อดี: dense, เข้าใจง่าย
    ข้อเสีย: ไม่ penalize volatility -> agent อาจเรียนรู้แบบ gambler
    """
    if len(env.equity_curve) < 2:
        return 0.0
    return float((env.equity_curve[-1] - env.equity_curve[-2]) / env.initial_balance)


def sharpe_reward(env, ctx: Dict[str, Any]) -> float:
    """
    Rolling Sharpe จาก trade_returns 20 รายการล่าสุด
    Sparse: reward เปลี่ยนเฉพาะตอนปิด trade -> ผสมกับ equity change เพื่อความ dense
    """
    # dense baseline ก่อนมี trade พอ
    if len(env.trade_returns) < 2:
        if len(env.equity_curve) >= 2:
            return float((env.equity_curve[-1] - env.equity_curve[-2]) / env.initial_balance * 0.1)
        return 0.0

    returns = np.array(env.trade_returns[-20:])
    if returns.std() == 0:
        return 0.0
    sharpe = returns.mean() / returns.std()
    return float(np.clip(sharpe * 0.01, -1.0, 1.0))


def differential_sharpe_reward(env, ctx: Dict[str, Any]) -> float:
    """
    Differential Sharpe Ratio (Moody & Saffell, 2001)
    -> dense, online, penalize volatility โดยอัตโนมัติ
    State: running mean (A) และ running mean of squares (B) ของ step returns
    """
    state = env._reward_state
    if "ds_A" not in state:
        state["ds_A"] = 0.0
        state["ds_B"] = 0.0
        state["ds_eta"] = 0.01

    if len(env.equity_curve) < 2:
        return 0.0

    eq_prev = env.equity_curve[-2]
    if eq_prev <= 0:
        return 0.0
    Rt = (env.equity_curve[-1] - eq_prev) / eq_prev

    A = state["ds_A"]
    B = state["ds_B"]
    eta = state["ds_eta"]
    delta_A = Rt - A
    delta_B = Rt**2 - B

    var = B - A**2
    if var > 1e-9:
        Dt = (B * delta_A - 0.5 * A * delta_B) / (var ** 1.5)
    else:
        Dt = 0.0

    state["ds_A"] = A + eta * delta_A
    state["ds_B"] = B + eta * delta_B

    return float(np.clip(Dt, -1.0, 1.0))


def multi_objective_reward(env, ctx: Dict[str, Any]) -> float:
    """
    Reward แบบหลาย component — แนะนำสำหรับ production
    Components:
      1) equity_return   (dense primary signal)
      2) trade_quality   (bonus winning / mild penalty losing)
      3) drawdown_penalty (penalize เมื่อ rolling DD > 10%)
      4) hold_too_long   (mild penalty ถือนานเกิน 100 candles)
      5) overtrade       (penalty ถ้า avg trade duration < 5 candles)
      6) invalid_action  (penalty หนัก: buy/sell ตอนถือ position แล้ว, close ตอนไม่ถือ)
    """
    trade_closed: bool = ctx.get("trade_closed", False)
    action: int = ctx.get("action", 0)
    invalid: bool = ctx.get("invalid_action", False)

    reward = 0.0

    # 1) Equity change — primary dense signal
    if len(env.equity_curve) >= 2:
        equity_return = (env.equity_curve[-1] - env.equity_curve[-2]) / env.initial_balance
        reward += equity_return * 1.0

    # 2) Trade quality bonus/penalty (asymmetric: bonus > penalty -> ไม่ทำให้กลัวเทรด)
    if trade_closed and len(env.trade_returns) > 0:
        last = env.trade_returns[-1]
        reward += 0.02 if last > 0 else -0.01

    # 3) Drawdown penalty (rolling 20-step)
    if len(env.equity_curve) > 20:
        recent = np.asarray(env.equity_curve[-20:], dtype=np.float64)
        peak = recent.max()
        if peak > 0:
            dd = 1.0 - recent[-1] / peak
            if dd > 0.10:
                reward -= dd * 0.1

    # 4) Hold-too-long penalty
    if env.position != 0:
        hold_duration = env.current_step - env.entry_step
        if hold_duration > 100:
            reward -= 0.001

    # 5) Overtrade penalty (เฉพาะตอน action เป็น trade action)
    if action in (1, 2, 3) and env.total_trades > 0:
        avg_duration = env.current_step / env.total_trades
        if avg_duration < 5:
            reward -= 0.005

    # 6) Invalid-action penalty (ลงโทษหนักให้ agent เลิกทำ)
    if invalid:
        reward -= 0.01

    return float(np.clip(reward, -1.0, 1.0))


def trend_aware_reward(env, ctx: Dict[str, Any]) -> float:
    """
    Reward function (Iter 4) — Differential Sharpe + trend awareness ผ่าน
    pre-computed d_trend_strength feature

    Components:
      1. Differential Sharpe (base)         - weight +/-1.0
      2. Trend ride bonus (per-step)        - holding in trend direction
      3. Counter-trend entry penalty        - threshold 0.5
      4. Invalid action penalty
    """
    state = env._reward_state
    if "ds_A" not in state:
        state["ds_A"] = 0.0
        state["ds_B"] = 0.0
        state["ds_eta"] = 0.01

    action: int = ctx.get("action", 0)
    invalid: bool = ctx.get("invalid_action", False)

    reward = 0.0

    # ============================================================
    # Component 1: Differential Sharpe (base, dense, risk-adjusted)
    # ============================================================
    if len(env.equity_curve) >= 2:
        eq_prev = env.equity_curve[-2]
        if eq_prev > 0:
            Rt = (env.equity_curve[-1] - eq_prev) / eq_prev
            A, B, eta = state["ds_A"], state["ds_B"], state["ds_eta"]
            delta_A = Rt - A
            delta_B = Rt**2 - B
            var = B - A**2
            if var > 1e-9:
                Dt = (B * delta_A - 0.5 * A * delta_B) / (var ** 1.5)
            else:
                Dt = 0.0
            state["ds_A"] = A + eta * delta_A
            state["ds_B"] = B + eta * delta_B
            reward += float(np.clip(Dt, -1.0, 1.0)) * 1.0

    # ============================================================
    # Component 2: Trend Ride Bonus (per-step ขณะถือ position)
    # ============================================================
    if env.position != 0 and "d_trend_strength" in env.df.columns:
        trend = float(env.df["d_trend_strength"].iloc[env.current_step])
        # บวกเมื่อถือไปทางเดียวกับเทรนด์
        aligned = env.position * trend
        if aligned > 0:
            reward += 0.005 * min(aligned, 1.0)

    # ============================================================
    # Component 3: Counter-Trend Entry Penalty (threshold 0.5)
    # ============================================================
    if action in (1, 2) and not invalid:  # buy or sell, success
        new_position = 1 if action == 1 else -1
        if "d_trend_strength" in env.df.columns:
            trend = float(env.df["d_trend_strength"].iloc[env.current_step])
            counter_trend = -new_position * trend  # บวกเมื่อสวนเทรนด์
            if counter_trend > 0.5:
                reward -= 0.01

    # ============================================================
    # Component: Invalid Action Penalty
    # ============================================================
    if invalid:
        reward -= 0.01

    return float(np.clip(reward, -1.0, 1.0))


def scalp_sniper_reward(env, ctx: Dict[str, Any]) -> float:
    """
    Reward function for M15 sniper scalping.

    Philosophy: No cooldown, no time gates. Instead, reward QUALITY entries
    and punish random entries. The agent learns to wait for structure (FVG)
    + trend alignment before entering. No forced patience — natural discipline.

    Components:
      1. Trade PnL — primary signal (amplified, asymmetric)
      2. Entry quality score — bonus for FVG-aligned entries, penalty for random
      3. Trend alignment — per-step bonus for riding with macro trend
      4. Counter-trend entry penalty
      5. Invalid action penalty
    """
    trade_closed: bool = ctx.get("trade_closed", False)
    action: int = ctx.get("action", 0)
    invalid: bool = ctx.get("invalid_action", False)

    reward = 0.0

    # ============================================================
    # Component 1: Trade PnL (sparse, large magnitude)
    # ============================================================
    if trade_closed and len(env.trade_returns) > 0:
        pnl = env.trade_returns[-1]
        if pnl > 0:
            reward += min(pnl * 50.0, 0.5)
        else:
            reward += max(pnl * 30.0, -0.3)

    # ============================================================
    # Component 2: Entry Quality Score (FVG-based)
    # Reward entries near FVG zones, penalize entries without structure
    # This teaches the agent to wait for setup — not via time, but via quality
    # ============================================================
    if action in (1, 2) and not invalid:
        new_pos = 1 if action == 1 else -1
        quality = 0.0

        # (a) FVG proximity: is there a relevant FVG zone nearby?
        if "fvg_distance" in env.df.columns:
            fvg_dist = abs(float(env.df["fvg_distance"].iloc[env.current_step]))
            fvg_str = float(env.df["fvg_strength"].iloc[env.current_step])

            # Check matching direction: bullish FVG for Buy, bearish for Sell
            has_bull = float(env.df["fvg_bull_active"].iloc[env.current_step]) > 0.5
            has_bear = float(env.df["fvg_bear_active"].iloc[env.current_step]) > 0.5
            fvg_aligned = (new_pos == 1 and has_bull) or (new_pos == -1 and has_bear)

            if fvg_aligned and fvg_dist < 1.5:
                # Near an aligned FVG zone — quality entry!
                quality += 0.01 * (1.5 - fvg_dist)  # closer = better
                if fvg_str > 0.5:
                    quality += 0.005  # strong gap = extra bonus
            elif not fvg_aligned and not has_bull and not has_bear:
                # No FVG at all — no structure, random entry
                quality -= 0.005

        # (b) Trend alignment bonus at entry
        if "d_trend_strength" in env.df.columns:
            trend = float(env.df["d_trend_strength"].iloc[env.current_step])
            aligned = new_pos * trend
            if aligned > 0.3:
                quality += 0.005  # entering with strong trend
            elif aligned > 0:
                quality += 0.002  # entering with mild trend

        reward += quality

    # ============================================================
    # Component 3: Trend ride bonus (per-step while holding)
    # ============================================================
    if env.position != 0 and "d_trend_strength" in env.df.columns:
        trend = float(env.df["d_trend_strength"].iloc[env.current_step])
        aligned = env.position * trend
        if aligned > 0:
            reward += 0.002 * min(aligned, 1.0)
        elif aligned < -0.3:
            reward -= 0.001

    # ============================================================
    # Component 4: Counter-trend entry penalty
    # ============================================================
    if action in (1, 2) and not invalid and "d_trend_strength" in env.df.columns:
        new_pos = 1 if action == 1 else -1
        trend = float(env.df["d_trend_strength"].iloc[env.current_step])
        counter = -new_pos * trend
        if counter > 0.3:
            reward -= 0.008
        if counter > 0.6:
            reward -= 0.015

    # ============================================================
    # Component 5: Invalid action penalty
    # ============================================================
    if invalid:
        reward -= 0.005

    return float(np.clip(reward, -1.0, 1.0))


def scalp_sniper_v2_reward(env, ctx: Dict[str, Any]) -> float:
    """
    Reward v2 — designed for TP/SL auto-close architecture (Discrete(3)).

    Key improvements over v1:
      - Fixed TP/SL rewards (ATR-independent) → consistent signal across price levels
      - Amplified entry quality signals (3× stronger FVG/trend)
      - Balanced breakeven: TP=+0.12, SL=-0.04 → breakeven at 25% WR (matches 1:3 RR)

    Components:
      1. Fixed TP/SL outcome    (sparse, dominant) → +0.12 TP / -0.04 SL
      2. Entry quality score    (sparse, at entry) → FVG alignment + trend direction
      3. Trend ride bonus       (dense, per-step)  → aligned position with daily trend
      4. Invalid action penalty (sparse)           → -0.01 per invalid action
    """
    trade_closed: bool = ctx.get("trade_closed", False)
    action: int = ctx.get("action", 0)
    invalid: bool = ctx.get("invalid_action", False)

    reward = 0.0

    # ============================================================
    # Component 1: Fixed TP/SL Outcome (ATR-independent)
    #   TP hit → +0.12   SL hit → -0.04
    #   Ratio 3:1 matches the RR structure → breakeven at 25% WR
    #   Agent needs WR > 25% for positive expected reward
    # ============================================================
    if trade_closed and len(env.trade_returns) > 0:
        pnl = env.trade_returns[-1]
        if pnl > 0:   # TP hit
            reward += 0.12
        else:          # SL hit
            reward -= 0.04

    # ============================================================
    # Component 2: Entry Quality Score (at entry only)
    #   FVG aligned + close to zone → up to +0.045
    #   No structure at all → -0.015
    #   Counter-trend → -0.02
    #   Best possible entry: +0.055 / Worst: -0.035
    # ============================================================
    if action in (1, 2) and not invalid:
        new_pos = 1 if action == 1 else -1
        quality = 0.0

        # (a) FVG alignment check
        if "fvg_distance" in env.df.columns:
            fvg_dist = abs(float(env.df["fvg_distance"].iloc[env.current_step]))
            fvg_str = float(env.df["fvg_strength"].iloc[env.current_step])

            has_bull = float(env.df["fvg_bull_active"].iloc[env.current_step]) > 0.5
            has_bear = float(env.df["fvg_bear_active"].iloc[env.current_step]) > 0.5
            fvg_aligned = (new_pos == 1 and has_bull) or (new_pos == -1 and has_bear)

            if fvg_aligned and fvg_dist < 1.5:
                # Near aligned FVG zone → quality entry
                quality += 0.03 * (1.5 - fvg_dist) / 1.5   # 0 to +0.03
                if fvg_str > 0.5:
                    quality += 0.015  # strong gap = bonus
            elif not has_bull and not has_bear:
                # No FVG structure at all → random entry penalty
                quality -= 0.015

        # (b) Trend alignment at entry
        if "d_trend_strength" in env.df.columns:
            trend = float(env.df["d_trend_strength"].iloc[env.current_step])
            aligned = new_pos * trend
            if aligned > 0.3:
                quality += 0.01     # entering with strong trend
            elif aligned < -0.3:
                quality -= 0.02     # counter-trend entry

        reward += quality

    # ============================================================
    # Component 3: Trend Ride Bonus (dense, per-step while holding)
    #   Reward for holding aligned with daily trend
    #   Typical trade ~30-60 steps → accumulates to +0.03-0.06
    #   Sized to be less than TP reward (0.12) → outcome dominates
    # ============================================================
    if env.position != 0 and "d_trend_strength" in env.df.columns:
        trend = float(env.df["d_trend_strength"].iloc[env.current_step])
        aligned = env.position * trend
        if aligned > 0:
            reward += 0.001 * min(aligned, 1.0)
        elif aligned < -0.3:
            reward -= 0.0005

    # ============================================================
    # Component 4: Invalid Action Penalty
    #   Strong penalty: agent must learn Hold while in position
    # ============================================================
    if invalid:
        reward -= 0.01

    return float(np.clip(reward, -1.0, 1.0))


def scalp_sniper_v3_reward(env, ctx: Dict[str, Any]) -> float:
    """
    Reward v3 — Trend-focused, FVG removed from reward.

    Oracle test proved: FVG has ZERO predictive power for TP/SL on M15
    (25.4% WR with FVG vs 25.6% random). Daily trend IS the edge
    (28.3% WR with FVG+trend vs 25.6% random).

    Strategy: reward purely based on outcomes + daily trend alignment.
    FVG stays in observation (agent can learn to use or ignore),
    but reward doesn't bias agent toward FVG entries.

    Components:
      1. Fixed TP/SL outcome  (+0.12 / -0.04)   — dominant signal
      2. Trend alignment at entry (+0.015 / -0.025) — the real edge
      3. Trend ride bonus     (+0.001/step)      — dense signal
      4. Invalid action penalty (-0.01)
    """
    trade_closed: bool = ctx.get("trade_closed", False)
    action: int = ctx.get("action", 0)
    invalid: bool = ctx.get("invalid_action", False)

    reward = 0.0

    # ============================================================
    # Component 1: Fixed TP/SL Outcome
    # ============================================================
    if trade_closed and len(env.trade_returns) > 0:
        pnl = env.trade_returns[-1]
        if pnl > 0:   # TP hit
            reward += 0.12
        else:          # SL hit
            reward -= 0.04

    # ============================================================
    # Component 2: Trend Alignment at Entry (amplified)
    #   This is THE edge: oracle shows 28.3% WR with trend vs 25.6% random
    #   Strong trend aligned: +0.015
    #   Mild trend aligned:   +0.005
    #   Counter-trend:        -0.025
    # ============================================================
    if action in (1, 2) and not invalid:
        new_pos = 1 if action == 1 else -1

        if "d_trend_strength" in env.df.columns:
            trend = float(env.df["d_trend_strength"].iloc[env.current_step])
            aligned = new_pos * trend

            if aligned > 0.5:
                reward += 0.015     # strong trend alignment
            elif aligned > 0.2:
                reward += 0.005     # mild trend alignment
            elif aligned < -0.3:
                reward -= 0.015     # counter-trend entry
            elif aligned < -0.5:
                reward -= 0.025     # strong counter-trend

    # ============================================================
    # Component 3: Trend Ride Bonus (dense, per-step)
    # ============================================================
    if env.position != 0 and "d_trend_strength" in env.df.columns:
        trend = float(env.df["d_trend_strength"].iloc[env.current_step])
        aligned = env.position * trend
        if aligned > 0:
            reward += 0.001 * min(aligned, 1.0)
        elif aligned < -0.3:
            reward -= 0.0005

    # ============================================================
    # Component 4: Invalid Action Penalty
    # ============================================================
    if invalid:
        reward -= 0.01

    return float(np.clip(reward, -1.0, 1.0))


def scalp_sniper_v4_reward(env, ctx: Dict[str, Any]) -> float:
    """
    Reward v4 — RR-adaptive + volatility-aware.

    Key improvements from data-driven analysis:
      1. TP/SL reward automatically scales with actual RR ratio from env config
         → works correctly for RR 1:2, 1:3, 1:5 etc.
      2. Volatility awareness: bonus for high-ATR entries (data shows high vol = better WR)
      3. Kept FVG + trend components (v2 vs v3 test proved FVG bonus helps)
      4. Wider SL compatible (designed for SL 2.0 ATR which reduces noise stops from 38% to 23%)

    Reward math:
      SL penalty fixed at -0.04
      TP reward = 0.04 * RR (e.g., 0.08 for 1:2, 0.12 for 1:3, 0.20 for 1:5)
      Breakeven WR in reward space always matches the actual breakeven WR
    """
    trade_closed: bool = ctx.get("trade_closed", False)
    action: int = ctx.get("action", 0)
    invalid: bool = ctx.get("invalid_action", False)

    reward = 0.0

    # ============================================================
    # Component 1: RR-Adaptive TP/SL Outcome
    #   Dynamically reads RR from env config
    #   SL penalty = -0.04 (fixed)
    #   TP reward  = +0.04 * RR → breakeven WR matches actual
    # ============================================================
    if trade_closed and len(env.trade_returns) > 0:
        pnl = env.trade_returns[-1]
        rr = env.tp_atr_mult / max(env.sl_atr_mult, 0.01)  # e.g. 2.0, 3.0, 5.0
        if pnl > 0:   # TP hit
            reward += 0.04 * rr  # +0.08 for 1:2, +0.12 for 1:3, +0.20 for 1:5
        else:          # SL hit
            reward -= 0.04

    # ============================================================
    # Component 2: Entry Quality Score
    #   (a) FVG alignment (proven useful: v2 > v3)
    #   (b) Trend alignment
    #   (c) Volatility bonus (data: high ATR = 28.7% WR vs low ATR = 25.7%)
    # ============================================================
    if action in (1, 2) and not invalid:
        new_pos = 1 if action == 1 else -1
        quality = 0.0

        # (a) FVG alignment
        if "fvg_distance" in env.df.columns:
            fvg_dist = abs(float(env.df["fvg_distance"].iloc[env.current_step]))
            fvg_str = float(env.df["fvg_strength"].iloc[env.current_step])

            has_bull = float(env.df["fvg_bull_active"].iloc[env.current_step]) > 0.5
            has_bear = float(env.df["fvg_bear_active"].iloc[env.current_step]) > 0.5
            fvg_aligned = (new_pos == 1 and has_bull) or (new_pos == -1 and has_bear)

            if fvg_aligned and fvg_dist < 1.5:
                quality += 0.03 * (1.5 - fvg_dist) / 1.5
                if fvg_str > 0.5:
                    quality += 0.015
            elif not has_bull and not has_bear:
                quality -= 0.015

        # (b) Trend alignment at entry
        if "d_trend_strength" in env.df.columns:
            trend = float(env.df["d_trend_strength"].iloc[env.current_step])
            aligned = new_pos * trend
            if aligned > 0.3:
                quality += 0.01
            elif aligned < -0.3:
                quality -= 0.02

        # (c) Volatility bonus: reward entries during high-vol periods
        if "atr_ratio" in env.df.columns:
            atr_ratio = float(env.df["atr_ratio"].iloc[env.current_step])
            # atr_ratio typical range: 0.001 - 0.01
            # High vol (> 75th percentile ~0.003) → bonus
            if atr_ratio > 0.003:
                quality += 0.005
            elif atr_ratio < 0.001:
                quality -= 0.005  # low vol → discourage entry

        reward += quality

    # ============================================================
    # Component 3: Trend Ride Bonus (dense, per-step)
    # ============================================================
    if env.position != 0 and "d_trend_strength" in env.df.columns:
        trend = float(env.df["d_trend_strength"].iloc[env.current_step])
        aligned = env.position * trend
        if aligned > 0:
            reward += 0.001 * min(aligned, 1.0)
        elif aligned < -0.3:
            reward -= 0.0005

    # ============================================================
    # Component 4: Invalid Action Penalty
    # ============================================================
    if invalid:
        reward -= 0.01

    return float(np.clip(reward, -1.0, 1.0))


def scalp_sniper_v5_reward(env, ctx: Dict[str, Any]) -> float:
    """
    Reward v5 — Demand/Supply zone overhaul.

    Key changes from v2:
      - FVG replaced with Demand/Supply zone entry quality
      - Session awareness: bonus for entering during active sessions (London/NY)
      - Freshness: untested D/S zones get higher reward
      - Fixed TP/SL rewards (same as v2 which was best performer)

    Components:
      1. Fixed TP/SL outcome    (+0.12 / -0.04)   — proven best in v2
      2. D/S zone entry quality (sparse, at entry)  — replaces FVG
      3. Trend alignment        (sparse + dense)    — daily trend edge
      4. Session bonus          (sparse, at entry)  — active session = better
      5. Invalid action penalty
    """
    trade_closed: bool = ctx.get("trade_closed", False)
    action: int = ctx.get("action", 0)
    invalid: bool = ctx.get("invalid_action", False)

    reward = 0.0

    # ============================================================
    # Component 1: Fixed TP/SL Outcome (same as v2 — proven best)
    # ============================================================
    if trade_closed and len(env.trade_returns) > 0:
        pnl = env.trade_returns[-1]
        if pnl > 0:   # TP hit
            reward += 0.12
        else:          # SL hit
            reward -= 0.04

    # ============================================================
    # Component 2: D/S Zone Entry Quality (replaces FVG)
    #   Buy near demand zone = institutional buy area = quality entry
    #   Sell near supply zone = institutional sell area = quality entry
    #   Freshness matters: untested zones are stronger signals
    # ============================================================
    if action in (1, 2) and not invalid:
        new_pos = 1 if action == 1 else -1
        quality = 0.0

        if "ds_distance" in env.df.columns:
            ds_dist = abs(float(env.df["ds_distance"].iloc[env.current_step]))
            ds_str = float(env.df["ds_strength"].iloc[env.current_step])
            ds_fresh = float(env.df["ds_freshness"].iloc[env.current_step])

            has_demand = float(env.df["demand_active"].iloc[env.current_step]) > 0.5
            has_supply = float(env.df["supply_active"].iloc[env.current_step]) > 0.5

            # Aligned: buy near demand, sell near supply
            ds_aligned = (new_pos == 1 and has_demand) or (new_pos == -1 and has_supply)

            if ds_aligned and ds_dist < 1.5:
                # Near aligned D/S zone — institutional order area
                proximity_bonus = 0.03 * (1.5 - ds_dist) / 1.5   # 0 to +0.03
                strength_bonus = 0.015 * ds_str if ds_str > 0.3 else 0.0
                freshness_bonus = 0.01 * ds_fresh   # untested = +0.01, tested = less
                quality += proximity_bonus + strength_bonus + freshness_bonus
            elif not has_demand and not has_supply:
                # No D/S structure at all — random entry
                quality -= 0.015

        # Trend alignment at entry
        if "d_trend_strength" in env.df.columns:
            trend = float(env.df["d_trend_strength"].iloc[env.current_step])
            aligned = new_pos * trend
            if aligned > 0.3:
                quality += 0.01
            elif aligned < -0.3:
                quality -= 0.02

        # Session bonus: entering during active session (London/NY)
        if "is_active_session" in env.df.columns:
            is_active = float(env.df["is_active_session"].iloc[env.current_step])
            if is_active > 0.5:
                quality += 0.005  # active session = higher vol = better moves

        reward += quality

    # ============================================================
    # Component 3: Trend Ride Bonus (dense, per-step)
    # ============================================================
    if env.position != 0 and "d_trend_strength" in env.df.columns:
        trend = float(env.df["d_trend_strength"].iloc[env.current_step])
        aligned = env.position * trend
        if aligned > 0:
            reward += 0.001 * min(aligned, 1.0)
        elif aligned < -0.3:
            reward -= 0.0005

    # ============================================================
    # Component 4: Invalid Action Penalty
    # ============================================================
    if invalid:
        reward -= 0.01

    return float(np.clip(reward, -1.0, 1.0))


def scalp_sniper_v6_reward(env, ctx: Dict[str, Any]) -> float:
    """
    Reward v6 — Anti-overtrade: v2 base + entry cost + D/S awareness.

    Root cause from analysis:
      - v2 (best: +0.50%) had avg 297 trades/fold
      - v5 (worst: -1.55%) had avg 374 trades/fold (+26% overtrade)
      - More trades = more commission drag = worse performance
      - The agent needs DISINCENTIVE to enter unless confident

    Design:
      1. Fixed TP/SL outcome: +0.12 / -0.04 (proven in v2)
      2. Entry COST: -0.003 per entry (makes agent think twice)
      3. D/S zone bonus at entry (SMALLER than v5 — max +0.025 vs +0.055)
      4. FVG bonus at entry (from v2 — proven helpful)
      5. Trend alignment (from v2)
      6. Trend ride bonus (from v2)

    Expected behavior: agent enters ~200-250 trades/fold (vs 300-400 before)
    Only enters when D/S + FVG + trend all confirm = high-quality setup
    """
    trade_closed: bool = ctx.get("trade_closed", False)
    action: int = ctx.get("action", 0)
    invalid: bool = ctx.get("invalid_action", False)

    reward = 0.0

    # ============================================================
    # Component 1: Fixed TP/SL Outcome (proven best in v2)
    # ============================================================
    if trade_closed and len(env.trade_returns) > 0:
        pnl = env.trade_returns[-1]
        if pnl > 0:   # TP hit
            reward += 0.12
        else:          # SL hit
            reward -= 0.04

    # ============================================================
    # Component 2: Entry Quality + Entry Cost
    #   Entry cost: -0.003 per entry REGARDLESS of quality
    #   This forces agent to only enter when expected reward > 0.003
    #   Quality bonuses can offset cost for good setups
    # ============================================================
    if action in (1, 2) and not invalid:
        new_pos = 1 if action == 1 else -1

        # Entry cost — THE key anti-overtrade mechanism
        reward -= 0.003

        # (a) D/S zone bonus (HALF the magnitude of v5)
        if "ds_distance" in env.df.columns:
            ds_dist = abs(float(env.df["ds_distance"].iloc[env.current_step]))
            ds_str = float(env.df["ds_strength"].iloc[env.current_step])
            ds_fresh = float(env.df["ds_freshness"].iloc[env.current_step])

            has_demand = float(env.df["demand_active"].iloc[env.current_step]) > 0.5
            has_supply = float(env.df["supply_active"].iloc[env.current_step]) > 0.5
            ds_aligned = (new_pos == 1 and has_demand) or (new_pos == -1 and has_supply)

            if ds_aligned and ds_dist < 1.5:
                reward += 0.015 * (1.5 - ds_dist) / 1.5  # max +0.015
                if ds_fresh > 0.5:
                    reward += 0.005  # fresh zone bonus

        # (b) FVG bonus (from v2 — proven to help)
        if "fvg_distance" in env.df.columns:
            fvg_dist = abs(float(env.df["fvg_distance"].iloc[env.current_step]))
            has_bull = float(env.df["fvg_bull_active"].iloc[env.current_step]) > 0.5
            has_bear = float(env.df["fvg_bear_active"].iloc[env.current_step]) > 0.5
            fvg_aligned = (new_pos == 1 and has_bull) or (new_pos == -1 and has_bear)

            if fvg_aligned and fvg_dist < 1.5:
                reward += 0.01 * (1.5 - fvg_dist) / 1.5  # max +0.01

        # (c) Trend alignment at entry
        if "d_trend_strength" in env.df.columns:
            trend = float(env.df["d_trend_strength"].iloc[env.current_step])
            aligned = new_pos * trend
            if aligned > 0.3:
                reward += 0.008
            elif aligned < -0.3:
                reward -= 0.015  # counter-trend penalty

    # ============================================================
    # Component 3: Trend Ride Bonus (dense, from v2)
    # ============================================================
    if env.position != 0 and "d_trend_strength" in env.df.columns:
        trend = float(env.df["d_trend_strength"].iloc[env.current_step])
        aligned = env.position * trend
        if aligned > 0:
            reward += 0.001 * min(aligned, 1.0)
        elif aligned < -0.3:
            reward -= 0.0005

    # ============================================================
    # Component 4: Invalid Action Penalty
    # ============================================================
    if invalid:
        reward -= 0.01

    return float(np.clip(reward, -1.0, 1.0))


def style_sniper_reward(env, ctx: Dict[str, Any]) -> float:
    """
    Style A: SNIPER — few trades, high RR (1:3 to 1:5)

    Philosophy: Wait for triple confluence (trend + structure + momentum),
    enter 1 trade, win big. Heavy entry cost forces extreme selectivity.
    Target: 50-100 trades per fold.

    Reward:  TP=+0.15, SL=-0.05  |  Entry cost=-0.01  |  Confluence bonus
    """
    trade_closed: bool = ctx.get("trade_closed", False)
    action: int = ctx.get("action", 0)
    invalid: bool = ctx.get("invalid_action", False)
    reward = 0.0

    # 1. Fixed TP/SL outcome
    if trade_closed and len(env.trade_returns) > 0:
        pnl = env.trade_returns[-1]
        if pnl > 0:
            reward += 0.15   # big win
        else:
            reward -= 0.05   # controlled loss

    # 2. Entry: heavy cost + triple confluence bonus
    if action in (1, 2) and not invalid:
        new_pos = 1 if action == 1 else -1
        reward -= 0.01  # heavy entry cost — forces selectivity

        confluence = 0
        # (a) Trend alignment
        if "d_trend_strength" in env.df.columns:
            trend = float(env.df["d_trend_strength"].iloc[env.current_step])
            if new_pos * trend > 0.3:
                confluence += 1
                reward += 0.005
            elif new_pos * trend < -0.3:
                reward -= 0.02  # counter-trend = heavy penalty

        # (b) D/S zone proximity
        if "ds_distance" in env.df.columns:
            ds_dist = abs(float(env.df["ds_distance"].iloc[env.current_step]))
            has_demand = float(env.df["demand_active"].iloc[env.current_step]) > 0.5
            has_supply = float(env.df["supply_active"].iloc[env.current_step]) > 0.5
            ds_aligned = (new_pos == 1 and has_demand) or (new_pos == -1 and has_supply)
            if ds_aligned and ds_dist < 1.0:
                confluence += 1
                reward += 0.008

        # (c) Strong candle confirmation
        if "candle_body_ratio" in env.df.columns:
            body = float(env.df["candle_body_ratio"].iloc[env.current_step])
            if body > 0.6:  # strong directional candle
                confluence += 1
                reward += 0.005

        # Triple confluence mega-bonus
        if confluence >= 3:
            reward += 0.01  # all three aligned = perfect sniper setup

    # 3. Trend ride bonus
    if env.position != 0 and "d_trend_strength" in env.df.columns:
        trend = float(env.df["d_trend_strength"].iloc[env.current_step])
        aligned = env.position * trend
        if aligned > 0:
            reward += 0.001 * min(aligned, 1.0)

    # 4. Invalid
    if invalid:
        reward -= 0.01

    return float(np.clip(reward, -1.0, 1.0))


def style_scalp_reward(env, ctx: Dict[str, Any]) -> float:
    """
    Style B: SCALPING — frequent trades, low RR (1:1 to 1:2), high WR needed

    Philosophy: Quick in, quick out. Grab small moves.
    Breakeven WR: 50% (1:1) or 33% (1:2). Need WR > 55%.
    Target: 300-500 trades per fold.

    Reward:  TP=+0.06, SL=-0.04  |  Momentum bonus  |  Session bonus
    """
    trade_closed: bool = ctx.get("trade_closed", False)
    action: int = ctx.get("action", 0)
    invalid: bool = ctx.get("invalid_action", False)
    reward = 0.0

    # 1. TP/SL outcome — tighter ratio (reflects 1:1-1:2 RR)
    if trade_closed and len(env.trade_returns) > 0:
        pnl = env.trade_returns[-1]
        if pnl > 0:
            reward += 0.06   # smaller win
        else:
            reward -= 0.04   # smaller loss

    # 2. Entry: light cost + momentum focus
    if action in (1, 2) and not invalid:
        new_pos = 1 if action == 1 else -1
        reward -= 0.001  # light entry cost — allow frequent trading

        # (a) Momentum: strong candle = good for scalping
        if "candle_body_ratio" in env.df.columns:
            body = float(env.df["candle_body_ratio"].iloc[env.current_step])
            if body > 0.5:
                reward += 0.003

        # (b) Session: scalping works best during active sessions
        if "is_active_session" in env.df.columns:
            is_active = float(env.df["is_active_session"].iloc[env.current_step])
            if is_active > 0.5:
                reward += 0.003
            else:
                reward -= 0.003  # scalping during Asian = bad

        # (c) Trend alignment (mild)
        if "d_trend_strength" in env.df.columns:
            trend = float(env.df["d_trend_strength"].iloc[env.current_step])
            if new_pos * trend > 0.2:
                reward += 0.003
            elif new_pos * trend < -0.3:
                reward -= 0.005

    # 3. No trend ride (scalp = quick exit, not hold)

    # 4. Invalid
    if invalid:
        reward -= 0.01

    return float(np.clip(reward, -1.0, 1.0))


def style_ds_trailing_reward(env, ctx: Dict[str, Any]) -> float:
    """
    Style C: DEMAND/SUPPLY + TRAILING STOP — enter at zones, let profits run

    Philosophy: Enter at institutional zones, use trailing stop to ride trends.
    No fixed TP — trailing stop captures big moves and cuts small ones.
    Target: 150-250 trades per fold.

    Reward: PnL-proportional (variable TP) + D/S zone bonus + trend ride
    """
    trade_closed: bool = ctx.get("trade_closed", False)
    action: int = ctx.get("action", 0)
    invalid: bool = ctx.get("invalid_action", False)
    reward = 0.0

    # 1. Trade outcome — PnL-proportional (trailing = variable profit)
    if trade_closed and len(env.trade_returns) > 0:
        pnl = env.trade_returns[-1]
        if pnl > 0:
            # Scale reward with profit size (trailing can produce big wins)
            reward += min(pnl * 40.0, 0.30)   # cap at 0.30 for huge wins
        else:
            reward -= 0.04   # fixed loss penalty (SL is fixed)

    # 2. Entry: D/S zone focused
    if action in (1, 2) and not invalid:
        new_pos = 1 if action == 1 else -1
        reward -= 0.005  # moderate entry cost

        # (a) D/S zone alignment — THE key entry signal
        if "ds_distance" in env.df.columns:
            ds_dist = abs(float(env.df["ds_distance"].iloc[env.current_step]))
            ds_fresh = float(env.df["ds_freshness"].iloc[env.current_step])
            has_demand = float(env.df["demand_active"].iloc[env.current_step]) > 0.5
            has_supply = float(env.df["supply_active"].iloc[env.current_step]) > 0.5
            ds_aligned = (new_pos == 1 and has_demand) or (new_pos == -1 and has_supply)

            if ds_aligned and ds_dist < 1.5:
                reward += 0.02 * (1.5 - ds_dist) / 1.5  # max +0.02
                if ds_fresh > 0.5:
                    reward += 0.01  # fresh zone = strong signal
            elif not has_demand and not has_supply:
                reward -= 0.01  # no zone = no structure

        # (b) Trend alignment
        if "d_trend_strength" in env.df.columns:
            trend = float(env.df["d_trend_strength"].iloc[env.current_step])
            if new_pos * trend > 0.3:
                reward += 0.008
            elif new_pos * trend < -0.3:
                reward -= 0.015

    # 3. Trend ride bonus (STRONGER than other styles — trailing = ride trend)
    if env.position != 0 and "d_trend_strength" in env.df.columns:
        trend = float(env.df["d_trend_strength"].iloc[env.current_step])
        aligned = env.position * trend
        if aligned > 0:
            reward += 0.002 * min(aligned, 1.0)  # 2x stronger ride bonus
        elif aligned < -0.3:
            reward -= 0.001

    # 4. Invalid
    if invalid:
        reward -= 0.01

    return float(np.clip(reward, -1.0, 1.0))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REWARD_REGISTRY: Dict[str, Callable] = {
    "pnl": pnl_reward,
    "sharpe": sharpe_reward,
    "differential_sharpe": differential_sharpe_reward,
    "multi_objective": multi_objective_reward,
    "trend_aware": trend_aware_reward,
    "scalp_sniper": scalp_sniper_reward,
    "scalp_sniper_v2": scalp_sniper_v2_reward,
    "scalp_sniper_v3": scalp_sniper_v3_reward,
    "scalp_sniper_v4": scalp_sniper_v4_reward,
    "scalp_sniper_v5": scalp_sniper_v5_reward,
    "scalp_sniper_v6": scalp_sniper_v6_reward,
    "style_sniper": style_sniper_reward,
    "style_scalp": style_scalp_reward,
    "style_ds_trailing": style_ds_trailing_reward,
}


def get_reward_fn(name: str) -> Callable:
    """ดึง reward function จาก registry ตามชื่อ"""
    if name not in REWARD_REGISTRY:
        raise ValueError(
            f"Unknown reward_type='{name}'. "
            f"Available: {sorted(REWARD_REGISTRY.keys())}"
        )
    return REWARD_REGISTRY[name]
