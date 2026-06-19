"""
Unified Specialist Bot — 5 rule-based strategies for XAUUSD M15

Strategies (all oracle-proven):
1. S/R LONG       (multi-touch support bounce) — OOS +84R
2. S/R SHORT      (multi-touch resistance reject) — IS +142R
3. Demand LONG    (demand zone rejection) — OOS 39% WR
4. Bull Engulfing (strict engulfing pattern) — OOS +231R 🏆
5. Inside Bar Break LONG — OOS +106R

Runs parallel to:
- RL Bot (Conf10 ensemble) — different magic number
- Momentum Bot (FIBO + Breakout + Momentum)

All decisions logged to live_logs/specialist_decisions_*.csv with column "strategy"
"""
from __future__ import annotations
import sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import csv
import datetime as dt
import time
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

import MetaTrader5 as mt5
from src.mt5_bridge import MT5Bridge
from src.structure_engine import StructureEngine
try:
    from tools import auto_tuner          # self-learning จากผลเทรด (2026-06-04)
except Exception:
    try:
        import auto_tuner
    except Exception:
        auto_tuner = None

# Chart generation (lazy import to avoid blocking startup)
CHARTS_DIR = Path("live_logs/charts")
TRADES_LOG  = Path("live_logs/trade_results.csv")

# Magic numbers per strategy (for order tracking)
MAGIC_MAP = {
    "bull_engulf": 2001,
    "inside_bar_break": 2002,
    "sr_long": 2003,
    "sr_short": 2004,
    "demand_long": 2005,
    "supply_zone": 2006,
    "dsf_short": 2007,    # Demand-to-Supply Flip
    "fvg_short": 2008,    # Fair Value Gap SHORT
    "fvg_long": 2009,     # Fair Value Gap LONG
    "dsf_long": 2010,     # Supply-to-Demand Flip LONG
    "bear_engulf": 2011,  # Bearish Engulfing SHORT
    "inside_bar_break_short": 2012,  # Inside Bar Break SHORT
    "breakout_long": 2013,           # Breakout LONG (level break + retest)
    "breakout_short": 2014,          # Breakout SHORT
    "momentum_long": 2015,           # Momentum Continuation LONG
    "momentum_short": 2016,          # Momentum Continuation SHORT
    "reentry_short": 2017,           # Smart Re-entry SHORT (หลัง Smart Exit)
    "reentry_long": 2018,            # Smart Re-entry LONG (หลัง Smart Exit)
    "choch_long":  2019,             # CHOCH LONG (Change of Character bullish)
    "choch_short": 2020,             # CHOCH SHORT (Change of Character bearish)
    "zone_retest_long": 2021,        # Zone Retest LONG (SMC Demand zone retest)
    "zone_retest_short": 2022,       # Zone Retest SHORT (SMC Supply zone retest)
    "srf_long": 2023,               # SRF LONG (Resistance→Support flip, breakout retest)
    "srf_short": 2024,              # SRF SHORT (Support→Resistance flip, breakdown retest)
    "zone_first_touch_long": 2025,  # Zone First Touch LONG (fresh Demand zone contact)
    "zone_first_touch_short": 2026, # Zone First Touch SHORT (fresh Supply zone contact)
    "srf_bounce_long": 2027,        # SRF Bounce LONG (horizontal multi-touch support + bounce)
    "srf_bounce_short": 2028,       # SRF Bounce SHORT (horizontal multi-touch resistance + reject)
    "pinbar_long": 2029,            # Pin Bar/Hammer LONG @ Demand
    "pinbar_short": 2030,           # Pin Bar/Shooting Star SHORT @ Supply
    "star_long": 2031,              # Morning Star LONG @ Demand
    "star_short": 2032,             # Evening Star SHORT @ Supply
    "htf_retest_long": 2033,        # HTF Retest LONG (bounce ที่ H1/H4 Demand)
    "htf_retest_short": 2034,       # HTF Retest SHORT (reject ที่ H1/H4 Supply)
    "break_retest_long": 2035,      # Break-Retest LONG (breakout→retest support→buy) — playbook
    "break_retest_short": 2036,     # Break-Retest SHORT (breakdown→retest resist→sell) — playbook
    "oversold_bounce_long": 2037,   # Oversold Bounce LONG (catch ก้น reversal ที่ oversold)
    "oversold_bounce_short": 2038,  # Overbought Bounce SHORT (catch ยอด reversal ที่ overbought)
    "trendline_long": 2039,         # Ascending trendline bounce (3+ higher lows) — user 06-06
    "trendline_short": 2040,        # Descending trendline reject (3+ lower highs) — user 06-06
    "hl_retest_long": 2041,         # Higher-Low Retest LONG (BASE→BREAKOUT→PULLBACK) — user 06-06
    "hl_retest_short": 2042,        # Lower-High Retest SHORT mirror — user 06-06
    "liq_sweep_long": 2043,         # Liquidity Sweep LONG (stop hunt + reverse) — user 06-09
    "liq_sweep_short": 2044,        # Liquidity Sweep SHORT mirror — user 06-09
    "imb_continuation_long": 2045,  # IMB Continuation LONG (impulse breakout) — user 06-09
    "imb_continuation_short": 2046, # IMB Continuation SHORT mirror — user 06-09
    "supply_react_short": 2047,     # Supply Reaction SHORT (light, user request 06-09)
    "demand_react_long": 2048,      # Demand Reaction LONG (light mirror, user 06-09)
    "momentum_breakout_long": 2049, # Momentum Breakout LONG (vertical impulse, user 06-12)
    "momentum_breakout_short": 2050, # Momentum Breakdown SHORT mirror (user 06-12)
    "consolidation_breakout_long": 2051,  # CONS-BREAK LONG (user 06-18: 5+bars tight + close pierce)
    "consolidation_breakout_short": 2052, # CONS-BREAK SHORT mirror
    "supply_rejection_short": 2053,       # SUPPLY-REJ SHORT (user 06-18: wicks 3+ ที่ supply + red strong = flip)
    "demand_rejection_long": 2054,        # DEMAND-REJ LONG mirror
}

# Smart Exit Memory — จำว่าปิดที่ zone ไหน เพื่อ re-enter ที่ zone ตรงข้าม
SMART_EXIT_MEMORY = []

# Signal Cooldown — ป้องกัน strategy เดิมยิง signal ซ้ำถี่เกิน
# key = (strategy, direction)  value = datetime ของ signal ล่าสุด
_SIGNAL_LAST_FIRE: dict = {}

# Anti-Whipsaw — จำ order ล่าสุดที่ fill (กัน flip ทิศที่ราคาใกล้เดิมใน chop)
_LAST_FILLED: dict = {"dir": None, "price": 0.0, "time": None}

# ── Anti-Churn Cooldown (2026-06-03) ──────────────────────────────────────
# หลักฐาน: 61 ไม้/9h, 66% ถือ <3 นาที, gross win≈loss, spread กิน -$92 → balance -$58
# โรค: auto-exit ปิดเร็ว → detector ยิงซ้ำ → re-enter ทันที (3 path: main/flip/re-entry)
# แก้: หลัง position ปิด ห้ามเข้าใหม่ทุก path เป็นเวลา ENTRY_COOLDOWN_SEC
_LAST_CLOSE_TIME = None
_PREV_OPEN_COUNT = 0
_LAST_TUNER_UPDATE = None   # auto-tuner: เรียนจากผลเทรดทุก ~30 นาที
_M5_TREND_STATE = {"v": None}  # 2026-06-12: hysteresis ของ M5 fast-trend (คง trend ตอน retest เล็ก)
# 2026-06-17: add 2 logic-fix ที่ดี (จาก session Jun17, cross-val) บน Jun12 base — user "add เฉพาะตัวดี"
_FIX_ZONE_RANGE_CONFLICT = True  # zone-veto เคารพ range_pos: short ที่ยอด(>65%)/buy ที่ก้น(<35%) ไม่บล็อก. cross-val +$563/+46/−60/+28
_FIX_ZONE_FLIP = True            # demand ทะลุลง=flip supply(sell) / supply ทะลุขึ้น=flip support(buy). cross-val +$281/+24ไม้
_AUTO_TUNER_OFF = True           # 2026-06-17 user: ปิด auto-tuner blocking (จะคัด strategy เองจาก live stats). record_entry ยังทำงาน=เก็บสถิติ
_FIX_NO_BUY_TOP = True           # 2026-06-17 user chart (กรอบแดง buy swing high): weak-long ห้าม buy ส่วนบน M5 range (>65%) = ซื้อยอด
_FIX_BREAKOUT_EXEMPT = True       # 2026-06-17 user: srf_long breakout-retest buy สูงได้หลังทะลุ — แต่เฉพาะ context_bias≠BEAR
#   regime/m5_momentum โดนเด้งสั้นในขาลงหลอก (อ่าน trend_bull/UP) → srf_long buy breakout ปลอม แพ้ −128~−186
#   context_bias (M5 swing 2h: LH+LL) ไม่โดนเด้งหลอก → ใช้มันกรอง downtrend แทน
# NOTE: ลอง block weak-long ทั้งหมดตอน context_bias=BEAR (Jun17) → FAIL (แพ้ 4/5 window, ไม่แก้ Jun17 เพราะ grind ช้า=NEUTRAL). อย่าทำ
# NOTE: ลองถอด Gate 1b (sell ชน support) ตามที่ user ขอ → FAIL (Jun9-13 −353: ขายชน support แล้วเด้ง แพ้, Jun17 +0 ไม่แก้). Gate 1b ปกป้องอยู่ อย่าถอด
_FIX_AUTO_RANGE = True            # 2026-06-17 user (Jun17 inversion: buy ยอด/sell ก้น): auto-range 30-bar M5 (2.5h = range ที่ user ตี) → ห้าม buy บน 1/3 / ห้าม sell ล่าง 1/3 (ยกเว้น momentum_breakout). บังคับ sell ยอด/buy ก้น ทุก regime
_AUTO_RANGE_TOP = 0.66           # buy ห้ามถ้า pos > นี้
_AUTO_RANGE_BOT = 0.34           # sell ห้ามถ้า pos < นี้
_AR_TF = "M5"                    # TF คำนวณ range: "M5" หรือ "M1"
_AR_BARS = 12                    # M5×12 = 1h. A/B 1h-vs-2.5h: +$920 (Jun9-13 +500, Jun15 −160→+207, Jun16 +58, Jun17 −5). window สั้นปรับตามตลาดเร็วกว่า
_FIX_IMB_QUALITY = False          # 2026-06-17 ลองแล้ว FAIL −$245 (fvg/imb 3/13W). detector fvg_long ใช้ rule หยาบ (gap 0.3ATR) ≠ "IMB ที่ user เลือกด้วยตา" → ไม่ใช่คุณภาพจริง
_FIX_SRF_BREAKOUT = True          # 2026-06-18 user สั่ง "ไม่ต้องบล็อค ให้ยิงเก็บ live data": exempt SRF/break_retest/breakout/cons_breakout จาก auto-range+Gate1b+fresh-peak เมื่อไม่สวนเทรนชัด (context_bias bear→block long / bull→block short เท่านั้น)
_FIX_VBOUNCE = True               # 2026-06-18 user (01:36 V-bounce 4275→4321 พลาด): exempt oversold_bounce จาก Gate 2.7 Stage 2 climax block — detector เองมี m5_ext>1.8+reversal candle = scope แคบ ปลอดภัย
_AR_TREND_EXEMPT = False         # 2026-06-17 ลองแล้ว FAIL −$892 (แม้ Jun15 ขาขึ้น −250): ผ่อน auto-range ให้ trend buy/sell = ไม้ trend-continuation ที่ block ไว้ = net loser. ครบ 4 วิธี trend-follow FAIL หมด (−685/−914/−892). auto-range mean-reversion (+$791) = edge. อย่าเปิด
_FIX_TREND_FOLLOW = True         # 2026-06-18 user "buy ตามเทรนไปเรื่อยๆ จนกว่าจะเปลี่ยน": exempt 5 strat user (srf/supply/engulf/star/break_retest) ใน confirmed trend + ADD weak counter-trend block (ส่วนใหม่ที่ขาด)
_BLOCK_WEAK_COUNTER = True       # 2026-06-18 user: trend BULL → block weak SHORT (supply_react/srf/break_retest/dsf/engulf/htf_retest) เพื่อไม่ให้ sell pullback. mirror for BEAR
ENTRY_COOLDOWN_SEC = 0    # 2026-06-17 user: ถอด cooldown — มี signal เข้าเลย (เก็บ live data). throttle เหลือ MAX_POS+stacking+1 decision/M1 bar


def _in_entry_cooldown() -> bool:
    """True = เพิ่งปิด position ไม่ถึง cooldown → ห้ามเข้าใหม่ (กัน churn)."""
    if _LAST_CLOSE_TIME is None:
        return False
    return (dt.datetime.now() - _LAST_CLOSE_TIME).total_seconds() < ENTRY_COOLDOWN_SEC


def _whipsaw_circuit_breaker(symbol):
    """CIRCUIT BREAKER (2026-06-12 user: "SL รัวๆ"): หยุดเข้าใหม่ถ้า SL รัวใน whipsaw.
    เคส Jun12 14:48-15:16: chop→trend transition แกว่ง 4175↔4202 → 5 SL ติด -$142
    กฎ: ≥3 ไม้แพ้ใน 15 นาทีล่าสุด AND net < -$30 → pause entries (รอให้ storm ผ่าน)
    ไม้แพ้เก่าจะ age out จาก window → resume เอง. ปกติ (8W/1L เช้า) ไม่ trigger
    """
    try:
        deals = mt5.history_deals_get(
            dt.datetime.now() - dt.timedelta(minutes=15),
            dt.datetime.now() + dt.timedelta(minutes=1))
        if not deals:
            return False
        outs = [d for d in deals if d.entry == 1 and d.symbol == symbol]
        losses = [d.profit for d in outs if d.profit < 0]
        net = sum(d.profit for d in outs)
        return len(losses) >= 3 and net < -30.0
    except Exception:
        return False

# cooldown (วินาที) ต่อ strategy group
_SIGNAL_COOLDOWN_SECS = {
    # Pattern strategies — fire ได้ใหม่หลัง 15 นาที
    "bull_engulf": 900, "bear_engulf": 900,
    "fvg_long": 900,    "fvg_short": 900,
    "momentum_long": 900, "momentum_short": 900,
    "breakout_long": 900, "breakout_short": 900,
    "inside_bar_break": 900, "inside_bar_break_short": 900,
    "sr_long": 900, "sr_short": 900,
    # Zone strategies — fire ได้ใหม่หลัง 20 นาที (ลดการยิงซ้ำที่ zone เดิม)
    "zone_retest_long": 1200, "zone_retest_short": 1200,
    "zone_first_touch_long": 1200, "zone_first_touch_short": 1200,
    "demand_long": 1200, "supply_zone": 1200,
    "srf_long": 1200, "srf_short": 1200,
    "srf_bounce_long": 1200, "srf_bounce_short": 1200,
    "pinbar_long": 1200, "pinbar_short": 1200,
    "star_long": 1200, "star_short": 1200,
    "htf_retest_long": 1200, "htf_retest_short": 1200,
    # Structural — fire ได้ใหม่หลัง 10 นาที (structure change เร็วกว่า)
    "choch_long": 600, "choch_short": 600,
    "dsf_long": 600,   "dsf_short": 600,
    # 2026-06-08: ตัวที่ผม add เมื่อวาน — ขาดทุนหนักเพราะ spam
    "hl_retest_long": 1500, "hl_retest_short": 1500,    # 25 min — pattern เปลี่ยนช้า
    "trendline_long": 1500, "trendline_short": 1500,    # 25 min
    "break_retest_long": 600, "break_retest_short": 600,  # 10 min — best performer
    # 2026-06-09: ขาดทุน -$183 จาก spam (no cooldown)
    "oversold_bounce_long": 1500, "oversold_bounce_short": 1500,    # 25 min
    "reentry_long": 900, "reentry_short": 900,                       # 15 min
    # NEW: Liquidity Sweep (stop hunt)
    "liq_sweep_long": 900, "liq_sweep_short": 900,                  # 15 min
    # NEW: IMB Continuation (impulse breakout) — body strict ทำให้ fire ไม่บ่อย
    "imb_continuation_long": 600, "imb_continuation_short": 600,    # 10 min
    # NEW: Supply/Demand React (light, user 06-09)
    "supply_react_short": 900, "demand_react_long": 900,            # 15 min
}

LIVE_LOGS_DIR = Path("live_logs")
REGIME_CLASSIFIER_PATH = "walk_forward_output/agi_regime_classifier.pkl"
SKIP_LONG_REGIMES = {3, 5, 6}
SKIP_SHORT_REGIMES = {3, 6}
ATR_WINDOW = 14
MT_LOOKBACK = 200
MT_TOLERANCE = 0.3  # ATR multiplier
STRUCTURE_ENGINE = StructureEngine(swing_window=5)


def detect_zone_flip(m15_bars_np, cur_price: float,
                     h1_low: float, h1_high: float, direction: str) -> tuple[bool, str]:
    """
    ตรวจ Zone Flip ตามหลัก SMC (ใช้ M15 candle — ตรง TF กับการวิเคราะห์ SMC structure):
      Supply → Demand: ราคาทะลุ supply zone อย่างแรง (IMB) แล้วกลับมา retest
      Demand → Supply: ราคาทะลุ demand zone อย่างแรง (IMB) แล้วกลับมา retest

    หลักเกณฑ์:
      1. M15 candle มี body > 1.3x ATR (strong IMB)
      2. candle เปิดต่ำกว่า zone แล้วปิดสูงกว่า (Supply flip) หรือกลับกัน
      3. ราคาปัจจุบัน retest กลับมาที่ zone level (ATR-based tolerance)
      4. ราคาต้องไม่ trending against flip direction (higher-lows = ห้าม SHORT flip)
    Returns: (flipped, reason)
    """
    if m15_bars_np is None or len(m15_bars_np) < 20 or h1_high <= h1_low:
        return False, ""
    rng = h1_high - h1_low
    if rng <= 0: return False, ""

    # M15 ATR จาก 14 bars ล่าสุด
    hl_list = [m15_bars_np[i][2] - m15_bars_np[i][3]
               for i in range(max(0, len(m15_bars_np)-14), len(m15_bars_np))]
    atr_m15 = sum(hl_list) / len(hl_list) if hl_list else 1.0

    # ── Anti-trend check: ถ้าราคากำลัง trending สวน flip direction → ไม่ flip ──
    # ดู 4 bars ล่าสุด: ถ้า 3 higher-lows ติดกัน → ห้าม SHORT flip
    #                    ถ้า 3 lower-highs ติดกัน → ห้าม LONG flip
    if len(m15_bars_np) >= 5:
        recent_lows  = [float(m15_bars_np[-(i+1)][3]) for i in range(4)][::-1]  # oldest→newest
        recent_highs = [float(m15_bars_np[-(i+1)][2]) for i in range(4)][::-1]
        if direction == "short":
            # 3 consecutive higher lows = uptrend → ห้าม SHORT flip
            hl_count = sum(1 for i in range(1, len(recent_lows))
                          if recent_lows[i] > recent_lows[i-1])
            if hl_count >= 3:
                return False, ""
        elif direction == "long":
            # 3 consecutive lower highs = downtrend → ห้าม LONG flip
            lh_count = sum(1 for i in range(1, len(recent_highs))
                          if recent_highs[i] < recent_highs[i-1])
            if lh_count >= 3:
                return False, ""

    # Retest tolerance = ATR-based (แทน % ที่กว้างเกินไป)
    retest_tol = 1.5 * atr_m15   # ~$10-12 ที่ ATR ~$7

    if direction == "long":
        # Supply zone เดิม อยู่ที่ 55%+ ของ range
        zone_lvl = h1_low + rng * 0.55
        # ตรวจ 30 bars M15 ล่าสุด (~7.5 ชม.) ว่ามี bullish IMB ทะลุ supply ไหม
        for i in range(max(0, len(m15_bars_np)-30), len(m15_bars_np)-1):
            b    = m15_bars_np[i]
            o, c = float(b[1]), float(b[4])
            body = abs(c - o)
            # Bullish IMB: เปิดต่ำกว่า supply, ปิดสูงกว่า supply อย่างมาก, body แรง
            if (o < zone_lvl
                    and c > zone_lvl + atr_m15 * 0.3
                    and body > atr_m15 * 1.3
                    and c > o):
                # Retest: ราคาปัจจุบันกลับลงมาใกล้ zone level (ATR-based)
                if zone_lvl - retest_tol <= cur_price <= zone_lvl + retest_tol:
                    return True, (f"Supply({zone_lvl:.1f})→Demand FLIP "
                                  f"[M15 IMB {body/atr_m15:.1f}x ATR, retest]")

    elif direction == "short":
        # Demand zone เดิม อยู่ที่ 45%- ของ range
        zone_lvl = h1_low + rng * 0.45
        for i in range(max(0, len(m15_bars_np)-30), len(m15_bars_np)-1):
            b    = m15_bars_np[i]
            o, c = float(b[1]), float(b[4])
            body = abs(c - o)
            # Bearish IMB: เปิดสูงกว่า demand, ปิดต่ำกว่า demand อย่างมาก, body แรง
            if (o > zone_lvl
                    and c < zone_lvl - atr_m15 * 0.3
                    and body > atr_m15 * 1.3
                    and c < o):
                if zone_lvl - retest_tol <= cur_price <= zone_lvl + retest_tol:
                    return True, (f"Demand({zone_lvl:.1f})→Supply FLIP "
                                  f"[M15 IMB {body/atr_m15:.1f}x ATR, retest]")

    return False, ""


def compute_atr(df, window=14):
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    return pd.Series(tr).rolling(window, min_periods=1).mean().values


def compute_mt_counts(df, atr, lookback=MT_LOOKBACK, tol_mult=MT_TOLERANCE):
    """Multi-touch counts for current bar (vs historical)."""
    n = len(df)
    if n < lookback + 1:
        return 0, 0
    high = df["high"].values
    low = df["low"].values
    atr_now = atr[-1]
    if atr_now < 0.01:
        return 0, 0
    tol = tol_mult * atr_now
    win_lows = low[-lookback-1:-1]
    win_highs = high[-lookback-1:-1]
    cur_low = low[-1]
    cur_high = high[-1]
    mt_long = int(((win_lows >= cur_low - tol) & (win_lows <= cur_low + tol)).sum())
    mt_short = int(((win_highs >= cur_high - tol) & (win_highs <= cur_high + tol)).sum())
    return mt_long, mt_short


# ============ STRATEGY DETECTORS ============

def detect_sr_long(df, atr, mt_long, min_touches=3, wick_min=0.4, body_min=0.3):
    """S/R Specialist LONG (oracle: +84R OOS, 29.7% WR). 2026-06-03 คลาย 4→3 touch ให้ยิงบ่อยขึ้น."""
    if mt_long < min_touches:
        return None
    close = df["close"].values; open_ = df["open"].values
    high = df["high"].values; low = df["low"].values
    atr_now = atr[-1]
    if close[-1] <= open_[-1]: return None
    body = abs(close[-1] - open_[-1]) / atr_now
    if body < body_min: return None
    lower_wick = (min(close[-1], open_[-1]) - low[-1]) / atr_now
    if lower_wick < wick_min: return None
    # Zone = current candle rejection area (wick zone ที่ reject)
    # zone_lo = low ของแท่งปัจจุบัน (จุด reject)
    # zone_hi = body bottom (open/close ที่ต่ำกว่า)
    return {"direction": "long", "strategy": "sr_long",
            "zone_hi": float(min(close[-1], open_[-1])), "zone_lo": float(low[-1]),
            "reason": f"SR LONG mt={mt_long} wick={lower_wick:.2f}ATR body={body:.2f}"}


def detect_sr_short(df, atr, mt_short, min_touches=3, wick_min=0.4, body_min=0.3):
    """S/R Specialist SHORT (oracle: +142R IS, 27% WR). 2026-06-03 คลาย 4→3 touch."""
    if mt_short < min_touches:
        return None
    close = df["close"].values; open_ = df["open"].values
    high = df["high"].values
    atr_now = atr[-1]
    if close[-1] >= open_[-1]: return None
    body = abs(close[-1] - open_[-1]) / atr_now
    if body < body_min: return None
    upper_wick = (high[-1] - max(close[-1], open_[-1])) / atr_now
    if upper_wick < wick_min: return None
    # Zone = current candle rejection area (wick zone ที่ reject)
    # zone_hi = high ของแท่งปัจจุบัน (จุด reject)
    # zone_lo = body top (open/close ที่สูงกว่า)
    return {"direction": "short", "strategy": "sr_short",
            "zone_hi": float(high[-1]), "zone_lo": float(max(close[-1], open_[-1])),
            "reason": f"SR SHORT mt={mt_short} wick={upper_wick:.2f}ATR body={body:.2f}"}


# ════════════════════════════════════════════════════════════════════════
# BREAK-RETEST (playbook user 2026-06-04: "breakdown→retest→sell")
# = หลังราคาทะลุระดับ → เด้งกลับมา retest ระดับนั้น (flip) → reject → เข้าตามทิศ break
# ใช้ find_sr_levels (multi-touch = กำแพงจริง). regime gate คุมทิศ (trend_bear→short ผ่าน)
# ════════════════════════════════════════════════════════════════════════
def detect_break_retest_short(df, atr, sr_levels, retest_tol=0.6, wick_min=0.3):
    """Breakdown→Retest→SELL: rally ขึ้น retest แนวต้าน (broken support→resist) + reject ลง."""
    if df is None or len(df) < 8 or not sr_levels:
        return None
    high = df["high"].values; low = df["low"].values
    close = df["close"].values; open_ = df["open"].values
    a = float(atr[-1]) if len(atr) else 1.0
    if a <= 0:
        return None
    cur = float(close[-1])
    recent_high = float(max(high[-3:]))            # ยอดที่เพิ่ง rally ขึ้นไปแตะ
    for lbl, lvl in sr_levels:
        if lbl != "RESIST":
            continue
        if not (lvl - retest_tol * a <= recent_high <= lvl + 1.2 * a):
            continue                                # ยอด rally แตะ resist (wick spike เหนือได้ = liquidity grab)
        if min(low[-6:-1]) >= lvl - 0.4 * a:
            continue                                # ต้องเพิ่ง rally มาจากล่าง (retest จริง)
        upper_wick = (recent_high - max(open_[-1], close[-1])) / a
        if cur < lvl - 0.05 * a and (upper_wick >= wick_min or close[-1] < open_[-1]):
            return {"direction": "short", "strategy": "break_retest_short",
                    "zone_hi": float(lvl + 0.4 * a), "zone_lo": float(cur - 0.6 * a),
                    "reason": f"BREAK-RETEST SHORT: rally retest RESIST {lvl:.1f} reject wick={upper_wick:.2f}ATR (playbook)"}
    return None


def detect_break_retest_long(df, atr, sr_levels, retest_tol=0.6, wick_min=0.3):
    """Breakout→Retest→BUY (mirror): dip ลง retest แนวรับ (broken resist→support) + bounce ขึ้น."""
    if df is None or len(df) < 8 or not sr_levels:
        return None
    high = df["high"].values; low = df["low"].values
    close = df["close"].values; open_ = df["open"].values
    a = float(atr[-1]) if len(atr) else 1.0
    if a <= 0:
        return None
    cur = float(close[-1])
    recent_low = float(min(low[-3:]))
    for lbl, lvl in sr_levels:
        if lbl != "SUPPORT":
            continue
        if not (lvl - 1.2 * a <= recent_low <= lvl + retest_tol * a):
            continue                                # dip แตะ support (wick spike ใต้ได้ = liquidity grab)
        if max(high[-6:-1]) <= lvl + 0.4 * a:
            continue                                # ต้องเพิ่ง dip มาจากบน
        lower_wick = (min(open_[-1], close[-1]) - recent_low) / a
        if cur > lvl + 0.05 * a and (lower_wick >= wick_min or close[-1] > open_[-1]):
            return {"direction": "long", "strategy": "break_retest_long",
                    "zone_hi": float(cur + 0.6 * a), "zone_lo": float(lvl - 0.4 * a),
                    "reason": f"BREAK-RETEST LONG: dip retest SUPPORT {lvl:.1f} bounce wick={lower_wick:.2f}ATR (playbook)"}
    return None


# ════════════════════════════════════════════════════════════════════════
# OVERSOLD/OVERBOUGHT BOUNCE (catch ก้น/ยอด reversal — user 2026-06-04: "ไม่ buy ก้นที่กลับตัว")
# ราคาดิ่ง/พุ่งเกิน 2 ATR จาก M5 EMA21 (extreme) + แท่งกลับตัว → counter-trend reversal ที่ถูกจังหวะ
# (ต่างจาก pinbar กลางทาง: ต้อง oversold/overbought จริงเท่านั้น = ก้น/ยอด ไม่ใช่ rally เข้า supply)
# ════════════════════════════════════════════════════════════════════════
def _m5_ema21(c):
    k = 2 / 22; e = float(c[-26])
    for v in c[-26:]:
        e = v * k + e * (1 - k)
    return e


def detect_oversold_bounce_long(df, atr, ext_min=1.8):
    """Oversold Bounce LONG: ราคาดิ่งใต้ EMA มาก (oversold) + แท่งกลับตัว bullish = catch ก้น."""
    if df is None or len(df) < 26:
        return None
    c = df["close"].values; o = df["open"].values; l = df["low"].values
    a = float(atr[-1]) if len(atr) else 1.0
    if a <= 0:
        return None
    ext = (_m5_ema21(c) - c[-1]) / a          # +ve = ราคาใต้ EMA = oversold
    if ext < ext_min:
        return None
    # 2026-06-05: -$245 จาก catching falling knife 5 SLs — ต้องการ SIGNIFICANT reversal:
    #   close > prev close + 0.5×ATR (ไม่ใช่แค่กระตุก) OR hammer ที่เด่นชัด
    green_strong = c[-1] > o[-1] and (c[-1] - c[-2]) > 0.5 * a
    lower_wick = min(o[-1], c[-1]) - l[-1]
    body_abs = abs(c[-1] - o[-1])
    hammer = lower_wick > 1.5 * body_abs and lower_wick > 0.4 * a and c[-1] >= o[-1]
    if green_strong or hammer:
        return {"direction": "long", "strategy": "oversold_bounce_long",
                "zone_hi": float(c[-1] + 1.0 * a), "zone_lo": float(min(l[-3:]) - 0.3 * a),
                "reason": f"OVERSOLD BOUNCE LONG: {ext:.1f}ATR ใต้ EMA + strong bullish (catch ก้น oversold)"}
    return None


def detect_oversold_bounce_short(df, atr, ext_min=1.8):
    """Overbought Bounce SHORT: ราคาพุ่งเหนือ EMA มาก (overbought) + แท่งกลับตัว bearish = catch ยอด."""
    if df is None or len(df) < 26:
        return None
    c = df["close"].values; o = df["open"].values; h = df["high"].values
    a = float(atr[-1]) if len(atr) else 1.0
    if a <= 0:
        return None
    ext = (c[-1] - _m5_ema21(c)) / a          # +ve = ราคาเหนือ EMA = overbought
    if ext < ext_min:
        return None
    # 2026-06-05 mirror: significant red close (ลง > 0.5 ATR) ไม่ใช่แค่กระตุก
    red_strong = c[-1] < o[-1] and (c[-2] - c[-1]) > 0.5 * a
    upper_wick = h[-1] - max(o[-1], c[-1])
    body_abs = abs(c[-1] - o[-1])
    star = upper_wick > 1.5 * body_abs and upper_wick > 0.4 * a and c[-1] <= o[-1]
    if red_strong or star:
        return {"direction": "short", "strategy": "oversold_bounce_short",
                "zone_hi": float(max(h[-3:]) + 0.3 * a), "zone_lo": float(c[-1] - 1.0 * a),
                "reason": f"OVERBOUGHT BOUNCE SHORT: {ext:.1f}ATR เหนือ EMA + bearish reversal (catch ยอด overbought)"}
    return None


# ════════════════════════════════════════════════════════════════════════
# TRENDLINE detector (2026-06-06 user: "เพิ่มเทรนไลน์ด้วยครับ")
# หา swing highs/lows ใน 30 bars + connect → เทรนไลน์
# entry: ราคา test เทรนไลน์ + ไม่ทะลุ (touch & reject)
# ════════════════════════════════════════════════════════════════════════
def _find_swings(values, kind, lookback=30, edge=2):
    """หา swing highs (kind='high') หรือ swing lows (kind='low') ใน lookback บาร์ล่าสุด."""
    swings = []
    n = len(values)
    start = max(edge, n - lookback)
    for i in range(start, n - edge):
        if kind == "high":
            if all(values[i] >= values[i-j] for j in range(1, edge+1)) and \
               all(values[i] >= values[i+j] for j in range(1, edge+1)):
                swings.append((i, float(values[i])))
        else:  # low
            if all(values[i] <= values[i-j] for j in range(1, edge+1)) and \
               all(values[i] <= values[i+j] for j in range(1, edge+1)):
                swings.append((i, float(values[i])))
    return swings


def detect_trendline_short(df, atr, lookback=30, min_touches=3, tol_atr=0.4):
    """Descending trendline (3+ lower highs) — sell ที่ rally retest trendline + reject."""
    if df is None or len(df) < lookback + 3:
        return None
    h = df["high"].values; c = df["close"].values; o = df["open"].values
    a = float(atr[-1]) if len(atr) else 1.0
    if a <= 0:
        return None
    swings = _find_swings(h, "high", lookback=lookback)
    if len(swings) < min_touches:
        return None
    recent = swings[-min_touches:]
    # ต้อง descending (lower highs)
    if not all(recent[i][1] < recent[i-1][1] - 0.1 * a for i in range(1, len(recent))):
        return None
    # extrapolate trendline ถึง current bar
    x1, y1 = recent[0]; x2, y2 = recent[-1]
    slope = (y2 - y1) / max(1, (x2 - x1))
    cur_idx = len(df) - 1
    tline = y2 + slope * (cur_idx - x2)
    cur_h = float(h[-1]); cur_c = float(c[-1]); cur_o = float(o[-1])
    # ราคาแตะ trendline + reject (close ต่ำกว่า trendline + bearish)
    if not (cur_h >= tline - tol_atr * a):
        return None
    if cur_c > tline:  # ไม่ reject = breakout
        return None
    if cur_c >= cur_o:  # ต้อง bearish candle
        return None
    return {"direction": "short", "strategy": "trendline_short",
            "zone_hi": float(max(h[-3:]) + 0.3 * a), "zone_lo": float(cur_c - 1.0 * a),
            "reason": f"TRENDLINE SHORT: descending trendline @{tline:.1f} ({len(recent)} touches, slope {slope:.2f})"}


def detect_trendline_long(df, atr, lookback=30, min_touches=3, tol_atr=0.4):
    """Ascending trendline (3+ higher lows) — buy ที่ dip retest trendline + bounce."""
    if df is None or len(df) < lookback + 3:
        return None
    l = df["low"].values; c = df["close"].values; o = df["open"].values
    a = float(atr[-1]) if len(atr) else 1.0
    if a <= 0:
        return None
    swings = _find_swings(l, "low", lookback=lookback)
    if len(swings) < min_touches:
        return None
    recent = swings[-min_touches:]
    # ต้อง ascending (higher lows)
    if not all(recent[i][1] > recent[i-1][1] + 0.1 * a for i in range(1, len(recent))):
        return None
    x1, y1 = recent[0]; x2, y2 = recent[-1]
    slope = (y2 - y1) / max(1, (x2 - x1))
    cur_idx = len(df) - 1
    tline = y2 + slope * (cur_idx - x2)
    cur_l = float(l[-1]); cur_c = float(c[-1]); cur_o = float(o[-1])
    if not (cur_l <= tline + tol_atr * a):
        return None
    if cur_c < tline:  # ทะลุลง
        return None
    if cur_c <= cur_o:  # ต้อง bullish
        return None
    return {"direction": "long", "strategy": "trendline_long",
            "zone_hi": float(cur_c + 1.0 * a), "zone_lo": float(min(l[-3:]) - 0.3 * a),
            "reason": f"TRENDLINE LONG: ascending trendline @{tline:.1f} ({len(recent)} touches, slope {slope:.2f})"}


# ════════════════════════════════════════════════════════════════════════
# HIGHER-LOW RETEST (user 2026-06-06: "ออกแบบการเข้าให้ครับ")
# Pattern: BASE → BREAKOUT (rally >= 2 ATR) → PULLBACK ที่ยังเป็น HL → BOUNCE
# = playbook BASE→BREAKOUT→RETEST ของ user เป๊ะ
# ════════════════════════════════════════════════════════════════════════
def detect_hl_retest_long(df, atr, lookback=25, peak_min_atr=1.8, pullback_min_atr=0.4):
    """Higher-Low Retest LONG: pullback bounce หลัง real breakout rally.

    1. หา swing low L1 ใน lookback ก่อน
    2. หา peak H1 หลัง L1 ที่ rally >= 1.8 ATR (real breakout)
    3. ปัจจุบัน pulled back >= 0.4 ATR จาก H1 แต่ยัง > L1 (HL จริง)
    4. แท่งล่าสุด bullish + ปิดสูงกว่าแท่งก่อน (turn up)
    """
    if df is None or len(df) < lookback + 3:
        return None
    h = df["high"].values; l = df["low"].values
    c = df["close"].values; o = df["open"].values
    a = float(atr[-1]) if len(atr) else 1.0
    if a <= 0:
        return None
    n = len(df)
    # Swing low L1 in last lookback bars (excluding current 2)
    L1_idx, L1 = None, None
    for i in range(n-3, max(2, n-3-lookback), -1):
        if (l[i] <= l[i-1] and l[i] <= l[i-2] and
            l[i] <= l[i+1] and l[i] <= l[i+2]):
            L1_idx = i; L1 = float(l[i])
            break
    if L1_idx is None:
        return None
    # Peak H1 after L1
    H1 = float(max(h[L1_idx:n-1]))
    if (H1 - L1) < peak_min_atr * a:
        return None
    # Current state: pulled back from H1, still above L1
    cur = float(c[-1])
    if cur >= H1 - pullback_min_atr * a:
        return None
    if cur <= L1:    # ทะลุ L1 = ไม่ใช่ HL
        return None
    # Bullish confirmation
    if c[-1] <= o[-1]:
        return None
    if c[-1] <= c[-2]:
        return None
    # ── DOUBLE BOTTOM bonus: หา L0 ก่อน L1 (within 1 ATR) → neckline projection ──
    is_db = False; L0 = None
    for j in range(L1_idx-3, max(2, L1_idx-lookback), -1):
        if (l[j] <= l[j-1] and l[j] <= l[j-2] and
            l[j] <= l[j+1] and l[j] <= l[j+2]):
            if abs(l[j] - L1) <= 1.0 * a:
                is_db = True; L0 = float(l[j])
            break
    if is_db:
        tp_hi = H1 + (H1 - L1)   # neckline projection
        rsn = f"DOUBLE BOTTOM + HL-RETEST LONG: L0={L0:.1f}≈L1={L1:.1f}, pullback @{cur:.1f}, TP→{tp_hi:.1f}"
    else:
        tp_hi = cur + 1.0 * a
        rsn = f"HL-RETEST LONG: pullback @{cur:.1f} from H1={H1:.1f}, HL above L1={L1:.1f}"
    return {"direction": "long", "strategy": "hl_retest_long",
            "zone_hi": float(tp_hi), "zone_lo": float(L1 - 0.3 * a),
            "reason": rsn}


def detect_hl_retest_short(df, atr, lookback=25, peak_min_atr=1.8, pullback_min_atr=0.4):
    """Lower-High Retest SHORT: pullback reject หลัง real breakdown drop. Mirror ของ long."""
    if df is None or len(df) < lookback + 3:
        return None
    h = df["high"].values; l = df["low"].values
    c = df["close"].values; o = df["open"].values
    a = float(atr[-1]) if len(atr) else 1.0
    if a <= 0:
        return None
    n = len(df)
    H1_idx, H1 = None, None
    for i in range(n-3, max(2, n-3-lookback), -1):
        if (h[i] >= h[i-1] and h[i] >= h[i-2] and
            h[i] >= h[i+1] and h[i] >= h[i+2]):
            H1_idx = i; H1 = float(h[i])
            break
    if H1_idx is None:
        return None
    L1 = float(min(l[H1_idx:n-1]))
    if (H1 - L1) < peak_min_atr * a:
        return None
    cur = float(c[-1])
    if cur <= L1 + pullback_min_atr * a:
        return None
    if cur >= H1:
        return None
    if c[-1] >= o[-1]:
        return None
    if c[-1] >= c[-2]:
        return None
    # ── DOUBLE TOP bonus: หา H0 ก่อน H1 (within 1 ATR) → neckline projection ──
    is_dt = False; H0 = None
    for j in range(H1_idx-3, max(2, H1_idx-lookback), -1):
        if (h[j] >= h[j-1] and h[j] >= h[j-2] and
            h[j] >= h[j+1] and h[j] >= h[j+2]):
            if abs(h[j] - H1) <= 1.0 * a:
                is_dt = True; H0 = float(h[j])
            break
    if is_dt:
        tp_lo = L1 - (H1 - L1)   # neckline projection ลง
        rsn = f"DOUBLE TOP + LH-RETEST SHORT: H0={H0:.1f}≈H1={H1:.1f}, pullback @{cur:.1f}, TP→{tp_lo:.1f}"
    else:
        tp_lo = cur - 1.0 * a
        rsn = f"LH-RETEST SHORT: pullback @{cur:.1f} from L1={L1:.1f}, LH below H1={H1:.1f}"
    return {"direction": "short", "strategy": "hl_retest_short",
            "zone_hi": float(H1 + 0.3 * a), "zone_lo": float(tp_lo),
            "reason": rsn}


# ════════════════════════════════════════════════════════════════════════
# LIQUIDITY SWEEP / STOP HUNT (2026-06-09 — add per user playbook)
# Pattern: ราคา wick ทะลุก้นเดิม (stops hit) → กลับขึ้นเหนือก้น = trapped sellers
# Strong reversal signal — มักเกิดที่ key swing point
# ════════════════════════════════════════════════════════════════════════
def detect_liq_sweep_long(df, atr, lookback=15, min_wick_atr=0.4):
    """Liquidity Sweep LONG: wick below recent low + bullish reversal."""
    if df is None or len(df) < lookback + 3:
        return None
    h = df["high"].values; l = df["low"].values
    c = df["close"].values; o = df["open"].values
    a = float(atr[-1]) if len(atr) else 1.0
    if a <= 0:
        return None
    # หา recent low จาก bars before recent 3 bars
    recent_low = float(min(l[-lookback:-3]))
    # หา bar ที่ sweep ใน last 3 bars
    sweep_idx = None
    for i in [-3, -2, -1]:
        if l[i] < recent_low - 0.15 * a and c[i] > recent_low:
            sweep_idx = i
            break
    if sweep_idx is None:
        return None
    # current bar ต้อง bullish + close > recent_low
    if c[-1] <= o[-1]:
        return None
    if c[-1] <= recent_low + 0.1 * a:
        return None
    # wick size ของ sweep bar
    wick = recent_low - l[sweep_idx]
    if wick < min_wick_atr * a:
        return None
    return {"direction": "long", "strategy": "liq_sweep_long",
            "zone_hi": float(c[-1] + 1.0 * a),
            "zone_lo": float(l[sweep_idx] - 0.3 * a),
            "reason": f"LIQ SWEEP LONG: wick to {l[sweep_idx]:.1f} below low {recent_low:.1f} reversed up"}


def detect_liq_sweep_short(df, atr, lookback=15, min_wick_atr=0.4):
    """Liquidity Sweep SHORT: wick above recent high + bearish reversal."""
    if df is None or len(df) < lookback + 3:
        return None
    h = df["high"].values; l = df["low"].values
    c = df["close"].values; o = df["open"].values
    a = float(atr[-1]) if len(atr) else 1.0
    if a <= 0:
        return None
    recent_high = float(max(h[-lookback:-3]))
    sweep_idx = None
    for i in [-3, -2, -1]:
        if h[i] > recent_high + 0.15 * a and c[i] < recent_high:
            sweep_idx = i
            break
    if sweep_idx is None:
        return None
    if c[-1] >= o[-1]:
        return None
    if c[-1] >= recent_high - 0.1 * a:
        return None
    wick = h[sweep_idx] - recent_high
    if wick < min_wick_atr * a:
        return None
    return {"direction": "short", "strategy": "liq_sweep_short",
            "zone_hi": float(h[sweep_idx] + 0.3 * a),
            "zone_lo": float(c[-1] - 1.0 * a),
            "reason": f"LIQ SWEEP SHORT: wick to {h[sweep_idx]:.1f} above high {recent_high:.1f} reversed down"}


# ════════════════════════════════════════════════════════════════════════
# IMB CONTINUATION (2026-06-09 user: "เพิ่ม IMB continuation")
# IMB candle = body > 1.5× avg ของ 3 bar ก่อน → impulse breakout
# Entry: เข้าตามทิศ IMB ทันที + need close near high/low (strong close)
# ════════════════════════════════════════════════════════════════════════
def detect_imb_continuation_long(df, atr, body_mult=1.5, close_pos=0.7):
    """IMB Continuation LONG: large bullish body + close near high = impulse breakout."""
    if df is None or len(df) < 6:
        return None
    h = df["high"].values; l = df["low"].values
    o = df["open"].values; c = df["close"].values
    a = float(atr[-1]) if len(atr) else 1.0
    if a <= 0:
        return None
    # current bar bullish + body big
    body = c[-1] - o[-1]
    if body <= 0:
        return None
    prev_ranges = (h[-4:-1] - l[-4:-1])
    avg_range = float(prev_ranges.mean()) if len(prev_ranges) else 0
    if avg_range <= 0:
        return None
    if body < body_mult * avg_range:
        return None
    # close near high (strong close — top X% of bar range)
    bar_range = h[-1] - l[-1]
    if bar_range <= 0:
        return None
    close_pct = (c[-1] - l[-1]) / bar_range
    if close_pct < close_pos:
        return None
    # close > previous 3 bars high (breakout of recent consolidation)
    if c[-1] <= float(h[-4:-1].max()):
        return None
    return {"direction": "long", "strategy": "imb_continuation_long",
            "zone_hi": float(c[-1] + 1.5 * a),
            "zone_lo": float(l[-1] - 0.3 * a),
            "reason": f"IMB CONTINUATION LONG: body {body:.1f} > {body_mult}× avg_range {avg_range:.1f}, close@{close_pct:.0%}"}


def detect_imb_continuation_short(df, atr, body_mult=1.5, close_pos=0.7):
    """IMB Continuation SHORT mirror: large bearish body + close near low."""
    if df is None or len(df) < 6:
        return None
    h = df["high"].values; l = df["low"].values
    o = df["open"].values; c = df["close"].values
    a = float(atr[-1]) if len(atr) else 1.0
    if a <= 0:
        return None
    body = o[-1] - c[-1]
    if body <= 0:
        return None
    prev_ranges = (h[-4:-1] - l[-4:-1])
    avg_range = float(prev_ranges.mean()) if len(prev_ranges) else 0
    if avg_range <= 0:
        return None
    if body < body_mult * avg_range:
        return None
    bar_range = h[-1] - l[-1]
    if bar_range <= 0:
        return None
    close_pct = (h[-1] - c[-1]) / bar_range
    if close_pct < close_pos:
        return None
    if c[-1] >= float(l[-4:-1].min()):
        return None
    return {"direction": "short", "strategy": "imb_continuation_short",
            "zone_hi": float(h[-1] + 0.3 * a),
            "zone_lo": float(c[-1] - 1.5 * a),
            "reason": f"IMB CONTINUATION SHORT: body {body:.1f} > {body_mult}× avg_range {avg_range:.1f}, close@{close_pct:.0%}"}


# ════════════════════════════════════════════════════════════════════════
# SUPPLY REACT SHORT (2026-06-09 user request — light version แยก, SHORT only)
# Pattern: ราคาแตะ Supply zone + แท่ง bearish + ไส้บนยืนยัน reject
# Lighter than zone_retest_short (wick 0.3→0.15, ไม่ต้อง strict body)
# NO bull mirror (user explicit request)
# ════════════════════════════════════════════════════════════════════════
def detect_supply_react_short(df, atr, zones, min_wick_atr=0.15, zone_tol_atr=0.3):
    """Supply Reaction SHORT (light): zone touch + bearish/shooting-star bar + upper wick.

    1a. Current bar bearish (close < open) + wick ≥ 0.15 ATR — OR —
    1b. Shooting star: upper wick ≥ 0.4 ATR AND wick > 2× body (ยอมรับ close ≥ open เล็กน้อย)
    2. high ของแท่ง touch supply zone area (zone ± 0.3 ATR)
    3. close ไม่หลุดบนโซน (ไม่ใช่ breakout up)

    2026-06-11 Fix B mirror: ยอมรับ shooting star ที่ supply zone
    """
    if df is None or len(df) < 5 or not zones:
        return None
    h = df["high"].values; l = df["low"].values
    o = df["open"].values; c = df["close"].values
    a = float(atr[-1]) if len(atr) else 1.0
    if a <= 0:
        return None
    upper_wick = (h[-1] - max(o[-1], c[-1])) / a
    body = abs(c[-1] - o[-1]) / a
    is_bearish = c[-1] < o[-1]
    is_shooting_star = upper_wick >= 0.4 and (body < 0.01 or upper_wick > 2.0 * body)
    if is_bearish:
        if upper_wick < min_wick_atr:
            return None
    elif is_shooting_star:
        pass  # accept shooting star at zone
    else:
        return None
    # find supply zone that high touches
    for z in zones:
        if z.get("type") != "SUPPLY":
            continue
        zlo = z["lo"]; zhi = z["hi"]
        # high in zone area (with tolerance)
        if h[-1] < zlo - zone_tol_atr * a or h[-1] > zhi + zone_tol_atr * a:
            continue
        # close ไม่ทะลุเหนือโซน (= not a breakout up)
        if c[-1] > zhi + 0.3 * a:
            continue
        return {
            "direction": "short",
            "strategy": "supply_react_short",
            "zone_hi": float(zhi + 0.3 * a),
            "zone_lo": float(c[-1] - 1.0 * a),
            "reason": f"SUPPLY REACT SHORT (light): Supply {zlo:.0f}-{zhi:.0f} reject wick={upper_wick:.2f}ATR"
        }
    return None


def detect_demand_react_long(df, atr, zones, min_wick_atr=0.15, zone_tol_atr=0.3):
    """Demand Reaction LONG (light, mirror): zone touch + bullish/hammer bar + lower wick.

    1a. Current bar bullish (close > open) + wick ≥ 0.15 ATR — OR —
    1b. Hammer: lower wick ≥ 0.4 ATR AND wick > 2× body (ยอมรับ close ≤ open เล็กน้อย)
    2. low ของแท่ง touch demand zone area (zone ± 0.3 ATR)
    3. close ไม่หลุดใต้โซน (ไม่ใช่ breakdown)

    2026-06-11 Fix B: ยอมรับ hammer ที่ zone — ราคาแท่งไส้ล่างยาว reject demand = bounce signal
    แม้ close < open เล็กน้อย (doji/hammer shape ที่ zone = strength)
    """
    if df is None or len(df) < 5 or not zones:
        return None
    h = df["high"].values; l = df["low"].values
    o = df["open"].values; c = df["close"].values
    a = float(atr[-1]) if len(atr) else 1.0
    if a <= 0:
        return None
    lower_wick = (min(o[-1], c[-1]) - l[-1]) / a
    body = abs(c[-1] - o[-1]) / a
    is_bullish = c[-1] > o[-1]
    is_hammer = lower_wick >= 0.4 and (body < 0.01 or lower_wick > 2.0 * body)
    if is_bullish:
        if lower_wick < min_wick_atr:
            return None
    elif is_hammer:
        pass  # accept hammer at zone
    else:
        return None
    for z in zones:
        if z.get("type") != "DEMAND":
            continue
        zlo = z["lo"]; zhi = z["hi"]
        if l[-1] > zhi + zone_tol_atr * a or l[-1] < zlo - zone_tol_atr * a:
            continue
        if c[-1] < zlo - 0.3 * a:
            continue
        return {
            "direction": "long",
            "strategy": "demand_react_long",
            "zone_hi": float(c[-1] + 1.0 * a),
            "zone_lo": float(zlo - 0.3 * a),
            "reason": f"DEMAND REACT LONG (light): Demand {zlo:.0f}-{zhi:.0f} bounce wick={lower_wick:.2f}ATR"
        }
    return None


def compute_swing_bias(df, lookback=25, edge=2):
    """Context bias จาก swing structure: BULL (HH+HL) / BEAR (LH+LL) / NEUTRAL."""
    if df is None or len(df) < lookback + 5:
        return "NEUTRAL"
    h = df["high"].values; l = df["low"].values
    n = len(h)
    swings_h, swings_l = [], []
    for i in range(max(edge, n-lookback), n-edge):
        if (h[i] >= h[i-1] and h[i] >= h[i-2] and
            h[i] >= h[i+1] and h[i] >= h[i+2]):
            swings_h.append(float(h[i]))
        if (l[i] <= l[i-1] and l[i] <= l[i-2] and
            l[i] <= l[i+1] and l[i] <= l[i+2]):
            swings_l.append(float(l[i]))
    if len(swings_h) < 2 or len(swings_l) < 2:
        return "NEUTRAL"
    # HH+HL = bull
    if swings_h[-1] > swings_h[-2] and swings_l[-1] > swings_l[-2]:
        return "BULL"
    # LH+LL = bear
    if swings_h[-1] < swings_h[-2] and swings_l[-1] < swings_l[-2]:
        return "BEAR"
    return "NEUTRAL"


def compute_m15_structure_bias(m15_bars):
    """
    ตรวจ M15 market structure bias จาก swing high/low pattern
    ────────────────────────────────────────────────────────────
    HH + HL = BULLISH structure → Bias = LONG
    LH + LL = BEARISH structure → Bias = SHORT
    Mixed   = NEUTRAL

    Bias เปลี่ยนเมื่อ CHOCH ยืนยัน:
    - CHOCH LONG (break above LH) → Bias flip LONG
    - CHOCH SHORT (break below HL) → Bias flip SHORT

    ตามภาพ user:
    ┌ Left oval:  After CHOCH → Bias BUY  → เข้า LONG signals
    └ Right oval: After CHOCH → Bias SELL → เข้า SHORT signals เท่านั้น
                  "Momentum ขึ้น Bias sell" = bounce = ไม่เข้า LONG
    """
    if m15_bars is None or len(m15_bars) < 25:
        return "NEUTRAL"

    highs = [float(b[2]) for b in m15_bars[-30:]]
    lows  = [float(b[3]) for b in m15_bars[-30:]]
    n = len(highs)

    # Swing points (3 bars each side — M15 scale)
    swing_highs, swing_lows = [], []
    for i in range(3, n - 3):
        if (highs[i] >= highs[i-1] and highs[i] >= highs[i-2] and highs[i] >= highs[i-3]
                and highs[i] >= highs[i+1] and highs[i] >= highs[i+2] and highs[i] >= highs[i+3]):
            swing_highs.append(highs[i])
        if (lows[i] <= lows[i-1] and lows[i] <= lows[i-2] and lows[i] <= lows[i-3]
                and lows[i] <= lows[i+1] and lows[i] <= lows[i+2] and lows[i] <= lows[i+3]):
            swing_lows.append(lows[i])

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return "NEUTRAL"

    hh = swing_highs[-1] > swing_highs[-2]   # Higher High
    hl = swing_lows[-1]  > swing_lows[-2]    # Higher Low
    lh = swing_highs[-1] < swing_highs[-2]   # Lower High
    ll = swing_lows[-1]  < swing_lows[-2]    # Lower Low

    if hh and hl:
        return "LONG"
    elif lh and ll:
        return "SHORT"
    return "NEUTRAL"


def detect_choch_long(df, atr, lookback=25):
    """CHOCH LONG — Change of Character: BEARISH → BULLISH structure flip

    ปัญหาที่แก้: M15 EMA ยัง lag หลัง CHOCH → bot block LONG ตลอด
    Logic:
      1. หา swing high (LH) ล่าสุดในขาลง — จุดที่ราคา reject ลง >= 1.5 ATR
      2. ราคา CLOSE เหนือ LH นั้นใน 15 bars ล่าสุด → CHOCH ยืนยัน
      3. ราคา pullback กลับมาที่ CHOCH level (S2F zone) ± 0.5 ATR
      4. Bullish candle ยืนยัน → LONG
    Bypass: M15 trend filter + Supply zone VETO (structure เปลี่ยนแล้ว)
    """
    n = len(df)
    if n < lookback + 5:
        return None
    close = df["close"].values
    open_ = df["open"].values
    high  = df["high"].values
    low   = df["low"].values
    atr_now = atr[-1]
    if atr_now <= 0:
        return None

    # หา swing high ล่าสุด (LH ในขาลง) ที่ reject ลง >= 2.0 ATR
    lh_levels = []
    for i in range(n - 5, max(n - lookback, 3), -1):
        if (high[i] >= high[i-1] and high[i] >= high[i-2]
                and high[i] >= high[i+1] and high[i] >= high[i+2]):
            future_drop = high[i] - low[i:min(i + 8, n)].min()
            # threshold 2.0 ATR — กัน LH เล็กๆ ใน pullback noise
            if future_drop >= 2.0 * atr_now:
                lh_levels.append((i, high[i]))

    if not lh_levels:
        return None

    # ตรวจ CHOCH: ราคา close เหนือ LH + 0.1 ATR (ไม่ใช่แค่ scratch)
    choch_level = None
    for idx, lvl in lh_levels:
        if any(close[j] > lvl + 0.1 * atr_now for j in range(max(idx + 1, n - 15), n - 1)):
            choch_level = lvl
            break

    if choch_level is None:
        return None

    # Entry: pullback กลับมาที่ CHOCH level (S2F zone)
    cur_close = close[-1]
    cur_open  = open_[-1]
    zone_lo = choch_level - 0.5 * atr_now
    zone_hi = choch_level + 0.5 * atr_now
    if not (zone_lo <= cur_close <= zone_hi):
        return None

    # Bullish confirmation
    if cur_close <= cur_open:
        return None
    body = abs(cur_close - cur_open) / atr_now
    if body < 0.25:
        return None

    return {"direction": "long", "strategy": "choch_long",
            "zone_hi": zone_hi, "zone_lo": zone_lo,
            "reason": f"CHOCH LONG: LH {choch_level:.0f} broke→S2F pullback body={body:.2f}ATR"}


def detect_choch_short(df, atr, lookback=25):
    """CHOCH SHORT — Change of Character: BULLISH → BEARISH structure flip

    Mirror ของ choch_long:
      1. หา swing low (HL) ล่าสุดในขาขึ้น — จุดที่ราคา bounce >= 1.5 ATR
      2. ราคา CLOSE ต่ำกว่า HL นั้นใน 15 bars ล่าสุด → CHOCH ยืนยัน
      3. ราคา pullback กลับมาที่ CHOCH level (R2S zone) ± 0.5 ATR
      4. Bearish candle ยืนยัน → SHORT
    """
    n = len(df)
    if n < lookback + 5:
        return None
    close = df["close"].values
    open_ = df["open"].values
    high  = df["high"].values
    low   = df["low"].values
    atr_now = atr[-1]
    if atr_now <= 0:
        return None

    hl_levels = []
    for i in range(n - 5, max(n - lookback, 3), -1):
        if (low[i] <= low[i-1] and low[i] <= low[i-2]
                and low[i] <= low[i+1] and low[i] <= low[i+2]):
            future_rise = high[i:min(i + 8, n)].max() - low[i]
            # เพิ่ม threshold 2.0 ATR (เดิม 1.5) — กัน HL เล็กๆ ใน pullback
            # HL ที่ valid ต้องมี bounce ที่แรงพอ ไม่ใช่แค่ noise
            if future_rise >= 2.0 * atr_now:
                hl_levels.append((i, low[i]))

    if not hl_levels:
        return None

    choch_level = None
    for idx, lvl in hl_levels:
        # เพิ่มตรวจ: break ต้องชัดเจน (close < lvl - 0.1*ATR) ไม่ใช่แค่ scratch
        if any(close[j] < lvl - 0.1 * atr_now for j in range(max(idx + 1, n - 15), n - 1)):
            choch_level = lvl
            break

    if choch_level is None:
        return None

    cur_close = close[-1]
    cur_open  = open_[-1]
    zone_lo = choch_level - 0.5 * atr_now
    zone_hi = choch_level + 0.5 * atr_now
    if not (zone_lo <= cur_close <= zone_hi):
        return None

    if cur_close >= cur_open:
        return None
    body = abs(cur_close - cur_open) / atr_now
    if body < 0.25:
        return None

    return {"direction": "short", "strategy": "choch_short",
            "zone_hi": zone_hi, "zone_lo": zone_lo,
            "reason": f"CHOCH SHORT: HL {choch_level:.0f} broke→R2S pullback body={body:.2f}ATR"}


def detect_dsf_short(df, atr, lookback=60, tol_atr=0.35, min_swing_usd=0.0):
    """DSF SHORT v2 — Demand→Supply Flip (zone quality filters)

    v1 → v2 เปลี่ยนแปลง:
    ┌────────────────────────┬──────────────────────────┬───────────────────────────────┐
    │ เงื่อนไข               │ v1 (เก่า)                │ v2 (ใหม่)                     │
    ├────────────────────────┼──────────────────────────┼───────────────────────────────┤
    │ Zone width             │ 0.6 ATR (ไม่มี min)      │ max(0.6 ATR, $3.0) — กัน M1  │
    │ Formation impulse      │ ไม่เช็ค                  │ body >= 0.8 ATR ใน 3 bars     │
    │ Clean exit             │ ไม่เช็ค                  │ close > dhi ใน 4 bars ≤1 osc │
    │ Break quality          │ แค่ recent_min < dlo     │ + break candle body >= 0.35   │
    │ Zone contamination     │ ไม่เช็ค                  │ wick penetrations <= 5        │
    │ Retest quality         │ ไม่เช็ค                  │ opposing candle ≤1 ใน zone   │
    │ Tolerance (tol_atr)    │ 0.5 ATR                  │ 0.35 ATR (tighter retest)     │
    │ min_swing_usd          │ —                        │ M1=$5, M5=$0 (กัน M1 noise)  │
    └────────────────────────┴──────────────────────────┴───────────────────────────────┘

    ดูตัวอย่าง GOOD: ภาพ 1 (retest clean ไม่มีแท่งแดงใน zone ตอนขึ้น)
                      ภาพ 3 (IMB+clean), ภาพ 5 (impulse แรง)
    ดูตัวอย่าง BAD:  ภาพ 4 (มีไส้เยอะ, ไม่แข็งแรง → clean_exit fails)
    """
    n = len(df)
    if n < lookback + 5: return None
    close = df["close"].values
    open_ = df["open"].values
    high  = df["high"].values
    low   = df["low"].values
    atr_now = atr[-1]
    if atr_now <= 0: return None

    demand_zones = []
    for i in range(n - 20, max(n - lookback, 5), -1):
        # ── 1. swing low: local minimum ────────────────────────────────────────
        if not (low[i] <= low[i-1] and low[i] <= low[i+1]):
            continue
        future_high = high[i:min(i+10, n)].max()
        swing_size = future_high - low[i]
        if swing_size / atr_now < 2.0:
            continue
        # min_swing_usd: กัน M1 noise — swing $3.34 (2*M1 ATR) ไม่พอ ต้อง >= $5
        if min_swing_usd > 0 and swing_size < min_swing_usd:
            continue

        # ── 2. Zone boundaries — minimum $3 ป้องกัน M1 zone จิ๋ว ─────────────
        dlo = low[i]
        dhi = dlo + max(0.6 * atr_now, 3.0)

        # ── 3. Formation impulse: bounce ออกจาก zone ต้อง STRONG ──────────────
        # อย่างน้อย 1 แท่งเขียวใน 3 bars แรก body >= 0.8 ATR
        has_impulse = False
        for j in range(i + 1, min(i + 4, n)):
            b = (close[j] - open_[j])
            if b > 0 and b / atr_now >= 0.8:
                has_impulse = True
                break
        if not has_impulse:
            continue

        # ── 4. Clean exit: ราคา close เหนือ dhi ภายใน 4 bars ≤1 oscillation ──
        # zone ที่ดี = ราคาออกจาก zone เร็วและ clean ไม่ oscillate นาน
        # (ภาพ 4 ล้มเหลงตรงนี้ — ราคาวน dhi↕dlo หลายครั้ง)
        clean_exit = False
        osc_count = 0
        for j in range(i + 1, min(i + 5, n)):
            if close[j] > dhi:
                clean_exit = True
                break
            elif close[j] >= dlo:
                osc_count += 1
        if not clean_exit or osc_count > 1:
            continue

        # ── 5. Zone contamination: ผ่อนคลาย > 5 (ไม่ punish consolidation zone) ──
        # ตรวจแค่ว่า zone ไม่ถูก "บด" จนหมดคุณค่า
        # (การตรวจ retest quality จะจัดการ dirty approach แทน)
        wick_pens = sum(
            1 for j in range(i + 1, min(i + 30, n - 5))
            if low[j] < dhi and close[j] > dlo
        )
        if wick_pens > 5:
            continue

        demand_zones.append((i, dlo, dhi))

    if not demand_zones:
        return None

    cur_close = close[-1]
    cur_open  = open_[-1]

    for zi, dlo, dhi in demand_zones:
        # ── 6. Break: ราคาหลุด dlo ไปแล้ว ──────────────────────────────────────
        recent_min = low[-15:-1].min()
        if recent_min > dlo:
            continue

        # ── 7. Break quality: break candle ต้องเป็น bearish body ≥ 0.35 ATR ───
        # กันกรณีที่ราคา spike ลงมาแล้วกลับ (ไม่ใช่ break จริง)
        break_ok = False
        for j in range(max(zi + 1, n - 20), n - 1):
            if low[j] < dlo:
                brk_body = open_[j] - close[j]  # bearish = open > close
                if brk_body > 0 and brk_body / atr_now >= 0.35:
                    break_ok = True
                    break
        if not break_ok:
            continue

        # ── 8. Retest: ราคา return กลับมาที่ zone เดิม (กลายเป็น Supply) ──────
        zone_lo = dlo - tol_atr * atr_now
        zone_hi = dhi + tol_atr * atr_now
        if not (zone_lo <= cur_close <= zone_hi):
            continue

        # ── 8b. RETEST QUALITY: ตอนราคา approach zone จากบนลงมา ─────────────
        # ไม่ควรมีแท่งเขียว (bullish) อยู่ใน zone ≥ 2 แท่ง
        # (ภาพ 1 = ขึ้นมา zone ไม่มีแท่งแดง → ภาพนี้คือ DSF LONG ดังนั้น short
        #  ต้องเช็คว่าลงมา zone ไม่มีแท่งเขียว = sellers ยังควบคุมอยู่)
        opposing_in_zone = sum(
            1 for j in range(max(0, n - 6), n - 1)
            if (low[j] < dhi and high[j] > dlo)            # candle ล้ำเข้า original zone (ไม่ใช่ expanded)
            and close[j] > open_[j]                        # bullish (ย้อนทิศ SHORT)
        )
        if opposing_in_zone > 1:
            continue  # retest dirty — buyers ยังต้าน

        # ── 9. Bearish rejection candle ──────────────────────────────────────
        if cur_close >= cur_open:
            continue
        body = abs(cur_close - cur_open) / atr_now
        if body < 0.25:
            continue

        return {"direction": "short", "strategy": "dsf_short",
                "zone_hi": zone_hi, "zone_lo": dlo,
                "reason": f"DSF SHORT: Demand {dlo:.0f}-{dhi:.0f} broke→Supply body={body:.2f}ATR"}

    return None


def detect_dsf_long(df, atr, lookback=60, tol_atr=0.35, min_swing_usd=0.0):
    """DSF LONG v2 — Supply→Demand Flip (zone quality filters)

    v1 → v2 เปลี่ยนแปลง: (mirror ของ dsf_short v2)
    ┌────────────────────────┬──────────────────────────┬───────────────────────────────┐
    │ เงื่อนไข               │ v1 (เก่า)                │ v2 (ใหม่)                     │
    ├────────────────────────┼──────────────────────────┼───────────────────────────────┤
    │ Zone width             │ 0.6 ATR (ไม่มี min)      │ max(0.6 ATR, $3.0) — กัน M1  │
    │ Formation impulse      │ ไม่เช็ค                  │ body >= 0.8 ATR ใน 3 bars     │
    │ Clean exit             │ ไม่เช็ค                  │ close < slo ใน 4 bars ≤1 osc │
    │ Break quality          │ แค่ recent_max > shi     │ + break candle body >= 0.35   │
    │ Zone contamination     │ ไม่เช็ค                  │ wick penetrations <= 5        │
    │ Retest quality         │ ไม่เช็ค                  │ opposing candle ≤1 ใน zone   │
    │ Tolerance (tol_atr)    │ 0.5 ATR                  │ 0.35 ATR (tighter retest)     │
    │ min_swing_usd          │ —                        │ M1=$5, M5=$0 (กัน M1 noise)  │
    └────────────────────────┴──────────────────────────┴───────────────────────────────┘

    ดูตัวอย่าง GOOD: ภาพ 1 (ขึ้นมา zone ไม่มีแท่งแดงเลย = retest quality สูง)
                      ภาพ 2 (CHOCH+clean), ภาพ 3 (IMB+clean), ภาพ 5 (impulse แรง)
    ดูตัวอย่าง BAD:  ภาพ 4 (มีไส้เยอะ, ไม่แข็งแรง → clean_exit fails)
    """
    n = len(df)
    if n < lookback + 5: return None
    close = df["close"].values
    open_ = df["open"].values
    high  = df["high"].values
    low   = df["low"].values
    atr_now = atr[-1]
    if atr_now <= 0: return None

    supply_zones = []
    for i in range(n - 20, max(n - lookback, 5), -1):
        # ── 1. swing high: local maximum ────────────────────────────────────────
        if not (high[i] >= high[i-1] and high[i] >= high[i+1]):
            continue
        future_low = low[i:min(i+10, n)].min()
        swing_size = high[i] - future_low
        if swing_size / atr_now < 2.0:
            continue
        # min_swing_usd: กัน M1 noise — swing $3.34 (2*M1 ATR) ไม่พอ ต้อง >= $5
        if min_swing_usd > 0 and swing_size < min_swing_usd:
            continue

        # ── 2. Zone boundaries — minimum $3 ─────────────────────────────────────
        shi = high[i]
        slo = shi - max(0.6 * atr_now, 3.0)

        # ── 3. Formation impulse: reject ออกจาก zone ต้อง STRONG ──────────────
        # อย่างน้อย 1 แท่งแดงใน 3 bars แรก body >= 0.8 ATR
        has_impulse = False
        for j in range(i + 1, min(i + 4, n)):
            b = (open_[j] - close[j])  # bearish candle
            if b > 0 and b / atr_now >= 0.8:
                has_impulse = True
                break
        if not has_impulse:
            continue

        # ── 4. Clean exit: ราคา close ต่ำกว่า slo ภายใน 4 bars ≤1 oscillation ─
        clean_exit = False
        osc_count = 0
        for j in range(i + 1, min(i + 5, n)):
            if close[j] < slo:
                clean_exit = True
                break
            elif close[j] <= shi:
                osc_count += 1
        if not clean_exit or osc_count > 1:
            continue

        # ── 5. Zone contamination: ผ่อนคลาย > 5 ─────────────────────────────────
        wick_pens = sum(
            1 for j in range(i + 1, min(i + 30, n - 5))
            if high[j] > slo and close[j] < shi
        )
        if wick_pens > 5:
            continue

        supply_zones.append((i, slo, shi))

    if not supply_zones:
        return None

    cur_close = close[-1]
    cur_open  = open_[-1]

    for zi, slo, shi in supply_zones:
        # ── 6. Break: ราคาทะลุ shi ขึ้นไปแล้ว ───────────────────────────────────
        recent_max = high[-15:-1].max()
        if recent_max < shi:
            continue

        # ── 7. Break quality: break candle ต้องเป็น bullish body ≥ 0.35 ATR ───
        break_ok = False
        for j in range(max(zi + 1, n - 20), n - 1):
            if high[j] > shi:
                brk_body = close[j] - open_[j]  # bullish = close > open
                if brk_body > 0 and brk_body / atr_now >= 0.35:
                    break_ok = True
                    break
        if not break_ok:
            continue

        # ── 8. Retest: ราคา pullback กลับมาที่ zone เดิม (กลายเป็น Demand) ────
        zone_lo = slo - tol_atr * atr_now
        zone_hi = shi + tol_atr * atr_now
        if not (zone_lo <= cur_close <= zone_hi):
            continue

        # ── 8b. RETEST QUALITY: ตอนราคา approach zone จากล่างขึ้นบน ──────────
        # ไม่ควรมีแท่งแดง (bearish) อยู่ใน zone ≥ 2 แท่ง
        # (ภาพ 1: "ขึ้นมาถึง zone ไม่มีแท่งแดงเลย" = zone แข็งแรง = LONG quality สูง)
        opposing_in_zone = sum(
            1 for j in range(max(0, n - 6), n - 1)
            if (low[j] < shi and high[j] > slo)            # candle ล้ำเข้า original zone (ไม่ใช่ expanded)
            and close[j] < open_[j]                        # bearish (ย้อนทิศ LONG)
        )
        if opposing_in_zone > 1:
            continue  # retest dirty — sellers ยังต้าน

        # ── 9. Bullish rejection candle ──────────────────────────────────────
        if cur_close <= cur_open:
            continue
        body = abs(cur_close - cur_open) / atr_now
        if body < 0.25:
            continue

        return {"direction": "long", "strategy": "dsf_long",
                "zone_hi": shi, "zone_lo": zone_lo,
                "reason": f"DSF LONG: Supply {slo:.0f}-{shi:.0f} broke→Demand body={body:.2f}ATR"}

    return None


def detect_fvg_short(df, atr, min_gap_atr=0.3, lookback=40):
    """FVG (Fair Value Gap) SHORT — SMC 3-candle imbalance จริงๆ

    Pattern ถูกต้อง:
      C1 [i-1]: แท่งใดก็ได้ — C1.low คือขอบล่าง
      C2 [i  ]: แท่ง bearish impulse ขนาดใหญ่ (กระโดดข้ามราคา)
      C3 [i+1]: แท่งหลัง — C3.high คือขอบบน

    FVG zone จริง: C3.high < C1.low (มีช่องว่างที่ C2 ข้ามผ่านไปโดยไม่ถูก trade)
      gap_top = C1.low
      gap_bot = C3.high

    เมื่อราคา return ขึ้นมาที่ gap → SHORT (ราคาเติม imbalance แล้วเด้งลง)

    SMC rule: FVG SHORT ต้องอยู่ในโซน SUPPLY (บน structure) เท่านั้น
    """
    n = len(df)
    if n < lookback + 5: return None
    close = df["close"].values
    open_ = df["open"].values
    high  = df["high"].values
    low   = df["low"].values
    atr_now = atr[-1]
    if atr_now <= 0: return None

    # SMC context: FVG SHORT only valid in UPPER structure
    range_high = high[-30:-1].max()
    range_low  = low[-30:-1].min()
    range_mid  = (range_high + range_low) / 2

    cur_close = close[-1]
    cur_open  = open_[-1]

    # ราคาต้องอยู่ในโซน Supply (บน) ไม่ใช่ Demand (ล่าง)
    if cur_close < range_mid: return None

    for i in range(n - 6, max(n - lookback, 5), -1):
        if i + 1 >= n: continue

        # C2 ต้องเป็น bearish impulse (ลงแรง)
        if close[i] >= open_[i]: continue
        body_imb = (open_[i] - close[i]) / atr_now
        if body_imb < 1.5: continue

        # FVG จริง: C1.low > C3.high (ช่องว่างระหว่าง C1 กับ C3 ที่ C2 ข้ามผ่าน)
        c1_low  = low[i - 1]
        c3_high = high[i + 1]
        if c1_low <= c3_high: continue  # ไม่มีช่องว่างจริง = ไม่ใช่ FVG
        gap_top = c1_low
        gap_bot = c3_high
        gap_size = (gap_top - gap_bot) / atr_now
        if gap_size < min_gap_atr: continue

        # FVG ต้องอยู่ในโซน Supply (บน midpoint)
        if gap_bot < range_mid: continue

        # ราคาปัจจุบัน return ขึ้นมาเติม FVG
        if not (gap_bot <= cur_close <= gap_top): continue

        # Bearish rejection ที่ FVG
        if cur_close >= cur_open: continue
        body = abs(cur_close - cur_open) / atr_now
        if body < 0.2: continue

        return {"direction": "short", "strategy": "fvg_short",
                "zone_hi": gap_top, "zone_lo": gap_bot,
                "reason": f"FVG SHORT (SMC): C1low={gap_top:.0f} C3high={gap_bot:.0f} gap={gap_size:.1f}ATR body={body:.2f}ATR"}

    return None


def detect_fvg_long(df, atr, min_gap_atr=0.3, lookback=40):
    """FVG LONG — SMC 3-candle imbalance จริงๆ

    Pattern:
      C1 [i-1]: แท่งใดก็ได้ — C1.high คือขอบบน
      C2 [i  ]: bullish impulse ขนาดใหญ่
      C3 [i+1]: C3.low คือขอบล่าง

    FVG zone: C3.low > C1.high (ช่องว่างที่ C2 ข้ามขึ้นไปโดยไม่ถูก trade)
      gap_bot = C1.high
      gap_top = C3.low

    เมื่อราคา pullback ลงมาที่ gap → LONG (เติม imbalance แล้วดีดขึ้น)
    """
    n = len(df)
    if n < lookback + 5: return None
    close = df["close"].values
    open_ = df["open"].values
    high  = df["high"].values
    low   = df["low"].values
    atr_now = atr[-1]
    if atr_now <= 0: return None

    cur_close = close[-1]
    cur_open  = open_[-1]

    for i in range(n - 6, max(n - lookback, 5), -1):
        if i + 1 >= n: continue
        # C2 ต้องเป็น bullish impulse
        if close[i] <= open_[i]: continue
        body_imb = (close[i] - open_[i]) / atr_now
        if body_imb < 1.5: continue

        # FVG จริง: C3.low > C1.high
        c1_high = high[i - 1]
        c3_low  = low[i + 1]
        if c3_low <= c1_high: continue  # ไม่มีช่องว่าง = ไม่ใช่ FVG
        gap_bot = c1_high
        gap_top = c3_low
        gap_size = (gap_top - gap_bot) / atr_now
        if gap_size < min_gap_atr: continue

        # ราคา return เข้า FVG
        if not (gap_bot <= cur_close <= gap_top): continue

        # Bullish rejection
        if cur_close <= cur_open: continue
        body = abs(cur_close - cur_open) / atr_now
        if body < 0.2: continue

        return {"direction": "long", "strategy": "fvg_long",
                "zone_hi": gap_top, "zone_lo": gap_bot,
                "reason": f"FVG LONG: gap {gap_bot:.0f}-{gap_top:.0f} ({gap_size:.1f}ATR) body={body:.2f}ATR"}

    return None


def detect_supply_zone_short(df, atr, drop_atr=2.0, lookback=40, tol_atr=0.4):
    """Supply Zone SHORT — price returns to origin of a sharp bearish move. (2026-06-03 drop 3.0→2.0 ให้ยิงบ่อยขึ้น)

    Logic: find the most recent bar where a strong drop (≥drop_atr ATR) started,
    mark that bar's HIGH as the supply zone top. When price returns to zone → SHORT.
    Unlike S/R SHORT (requires multi-touch), supply zones form from ONE strong move.

    SMC rule: supply zones are at the TOP of structure. Only fire when price is in
    the upper 40% of the recent 30-bar range (not at demand/bottom).
    """
    n = len(df)
    if n < lookback + 5: return None
    close = df["close"].values; open_ = df["open"].values
    high = df["high"].values; low = df["low"].values
    atr_now = atr[-1]
    if atr_now <= 0: return None

    # SMC context: supply zones only valid in UPPER structure
    range_high = high[-30:-1].max()
    range_low  = low[-30:-1].min()
    range_mid  = (range_high + range_low) / 2
    if close[-1] < range_mid:  # price below midpoint = demand territory, not supply
        return None

    # Find the origin bar of the most recent sharp drop
    zone_hi = None
    for i in range(n - 3, n - lookback, -1):
        # Sharp drop: high[i] to min(low[i:i+4]) >= drop_atr ATR
        future_low = low[i:min(i + 5, n)].min()
        drop = (high[i] - future_low) / atr_now
        if drop >= drop_atr:
            zone_hi = high[i]
            zone_lo = zone_hi - tol_atr * atr_now
            break

    if zone_hi is None: return None

    # Supply zone itself must also be in upper structure (above midpoint)
    if zone_lo < range_mid: return None

    # Current bar must be inside or just below zone
    cur_high = high[-1]
    if not (zone_lo <= cur_high <= zone_hi + 0.3 * atr_now):
        return None

    # Bearish rejection: current bar bearish with meaningful body
    if close[-1] >= open_[-1]: return None
    body = abs(close[-1] - open_[-1]) / atr_now
    if body < 0.3: return None

    return {"direction": "short", "strategy": "supply_zone",
            "zone_hi": zone_hi, "zone_lo": zone_lo,
            "reason": f"Supply Zone SHORT zone={zone_lo:.0f}-{zone_hi:.0f} body={body:.2f}ATR"}


def detect_demand_long(df, atr, wick_min=0.4, body_min=0.2):
    """Demand zone LONG — uses simple proxy (last 10-bar low + bullish rejection). (2026-06-03 wick 0.5→0.4)"""
    n = len(df)
    if n < 30: return None
    close = df["close"].values; open_ = df["open"].values
    high = df["high"].values; low = df["low"].values
    atr_now = atr[-1]
    # Proxy demand zone: current low within 0.3 ATR of last 10-bar low
    recent_low = low[-10:-1].min()
    if low[-1] > recent_low + 0.3 * atr_now: return None
    if close[-1] <= open_[-1]: return None
    body = abs(close[-1] - open_[-1]) / atr_now
    if body < body_min: return None
    lower_wick = (min(close[-1], open_[-1]) - low[-1]) / atr_now
    if lower_wick < wick_min: return None
    return {"direction": "long", "strategy": "demand_long",
            "zone_hi": recent_low + 0.3 * atr_now, "zone_lo": recent_low,
            "reason": f"Demand LONG body={body:.2f} wick={lower_wick:.2f}ATR"}


def detect_bull_engulfing(df, atr, min_body_atr=0.6):  # PROVEN: oracle +231R OOS
    """Strict bullish engulfing — oracle: +231R OOS 33.6% WR (BIGGEST!).
    ต้องเกิดใกล้ Demand zone (ล่าง 40% ของ range) ไม่ใช่กลางทาง
    """
    n = len(df)
    if n < 30: return None
    close = df["close"].values; open_ = df["open"].values
    high = df["high"].values; low = df["low"].values
    atr_now = atr[-1]
    if close[-1] <= open_[-1]: return None  # current must be bullish
    if close[-2] >= open_[-2]: return None  # previous must be bearish
    if open_[-1] > close[-2]: return None
    if close[-1] < open_[-2]: return None
    body = abs(close[-1] - open_[-1]) / atr_now
    if body < min_body_atr: return None
    # ── Zone proximity: ต้องเกิดใกล้ Demand (ล่าง 40% ของ 30-bar range) ──
    range_high = high[-30:].max()
    range_low  = low[-30:].min()
    range_size = range_high - range_low
    if range_size > 0:
        position = (close[-1] - range_low) / range_size
        if position > 0.40:  # อยู่เหนือ 40% = ไม่ใช่ Demand zone = ไม่เข้า
            return None
    # zone = mother candle range (SL ต่ำกว่า low ของ mother candle)
    mother_low = min(low[-1], low[-2])
    mother_high = max(high[-1], high[-2])
    return {"direction": "long", "strategy": "bull_engulf",
            "zone_hi": mother_high, "zone_lo": mother_low,
            "reason": f"Bull Engulf strict body={body:.2f}ATR pos={position:.0%}"}


def detect_bear_engulfing(df, atr, min_body_atr=0.6):
    """Strict bearish engulfing — mirror ของ bull_engulf.
    ต้องเกิดใกล้ Supply zone (บน 60% ของ range) ไม่ใช่กลางทาง
    """
    n = len(df)
    if n < 30: return None
    close = df["close"].values; open_ = df["open"].values
    high = df["high"].values; low = df["low"].values
    atr_now = atr[-1]
    if close[-1] >= open_[-1]: return None   # current must be bearish
    if close[-2] <= open_[-2]: return None   # previous must be bullish
    if open_[-1] < close[-2]: return None    # open below prev close = no engulf
    if close[-1] > open_[-2]: return None    # close above prev open = no engulf
    body = abs(close[-1] - open_[-1]) / atr_now
    if body < min_body_atr: return None
    # ── Zone proximity: ต้องเกิดใกล้ Supply (บน 60% ของ 30-bar range) ──
    range_high = high[-30:].max()
    range_low  = low[-30:].min()
    range_size = range_high - range_low
    if range_size > 0:
        position = (close[-1] - range_low) / range_size
        if position < 0.60:  # อยู่ต่ำกว่า 60% = ไม่ใช่ Supply zone = ไม่เข้า
            return None
    mother_low = min(low[-1], low[-2])
    mother_high = max(high[-1], high[-2])
    return {"direction": "short", "strategy": "bear_engulf",
            "zone_hi": mother_high, "zone_lo": mother_low,
            "reason": f"Bear Engulf strict body={body:.2f}ATR pos={position:.0%}"}


def detect_inside_bar_break_long(df, atr):
    """Inside bar then break above — oracle: +106R OOS."""
    n = len(df)
    if n < 4: return None
    close = df["close"].values; open_ = df["open"].values
    high = df["high"].values; low = df["low"].values
    # Bar -2 is inside bar -3
    if not (high[-2] <= high[-3] and low[-2] >= low[-3]): return None
    # Current bar breaks above mother bar high
    if close[-1] <= high[-3]: return None
    if close[-1] <= open_[-1]: return None
    return {"direction": "long", "strategy": "inside_bar_break",
            "zone_hi": high[-3], "zone_lo": low[-3],
            "reason": f"Inside Bar break above {high[-3]:.2f}"}


def detect_inside_bar_break_short(df, atr):
    """Inside bar then break below — mirror ของ inside_bar_break_long."""
    n = len(df)
    if n < 4: return None
    close = df["close"].values; open_ = df["open"].values
    high = df["high"].values; low = df["low"].values
    # Bar -2 is inside bar -3
    if not (high[-2] <= high[-3] and low[-2] >= low[-3]): return None
    # Current bar breaks below mother bar low
    if close[-1] >= low[-3]: return None
    if close[-1] >= open_[-1]: return None   # must be bearish
    return {"direction": "short", "strategy": "inside_bar_break_short",
            "zone_hi": high[-3], "zone_lo": low[-3],
            "reason": f"Inside Bar break below {low[-3]:.2f}"}


# ═══════════════════════════════════════════════════════════════════════
# Pin Bar / Hammer + Star patterns AT ZONE (candlestick + zone confirmation)
# XAU เกิด liquidity grab/stop hunt บ่อยที่ S/D zone → pin bar = rejection จริง
# ═══════════════════════════════════════════════════════════════════════

def detect_pinbar_long(df, atr, zones, min_wick_ratio=1.8):
    """Pin Bar / Hammer LONG — ไส้ล่างยาว (ปฏิเสธราคาต่ำ) ที่ Demand zone → BUY

    Hammer: ไส้ล่าง >= 2× body + >= ครึ่งแท่ง, ไส้บนสั้น → stop hunt แล้วเด้ง
    ต้องอยู่ที่ Demand zone (rejection จริง ไม่ใช่ noise กลางอากาศ)
    """
    n = len(df)
    if n < 5 or not zones:
        return None
    close = df["close"].values; open_ = df["open"].values
    high = df["high"].values; low = df["low"].values
    atr_now = float(atr[-1])
    if atr_now <= 0:
        return None
    o, c, h, l = open_[-1], close[-1], high[-1], low[-1]
    body = abs(c - o)
    rng = h - l
    if rng <= 0:
        return None
    lower_wick = min(o, c) - l
    upper_wick = h - max(o, c)
    # Hammer shape
    if lower_wick < min_wick_ratio * max(body, 0.01 * atr_now):
        return None
    if lower_wick < 0.5 * rng:        # ไส้ล่าง >= ครึ่งแท่ง
        return None
    if upper_wick > body + 0.1 * atr_now:   # ไส้บนต้องสั้น
        return None
    if rng < 0.3 * atr_now:           # แท่งเล็กเกิน = noise
        return None
    # ต้องอยู่ที่ Demand zone (low แตะ/แทงเข้า zone)
    tol = atr_now * 0.3
    for z in zones:
        if z["type"] != "DEMAND":
            continue
        zlo, zhi = z["lo"], z["hi"]
        if (zlo - atr_now) <= l <= (zhi + tol):
            return {
                "direction": "long",
                "strategy": "pinbar_long",
                "zone_hi": float(zhi),
                "zone_lo": float(min(zlo, l)),   # SL ใต้ไส้ pin
                "reason": (f"Pin Bar (Hammer) LONG @ Demand {zlo:.0f}-{zhi:.0f} "
                           f"wick={lower_wick/atr_now:.2f}ATR body={body/atr_now:.2f}ATR")
            }
    return None


def detect_pinbar_short(df, atr, zones, min_wick_ratio=1.8):
    """Pin Bar / Shooting Star SHORT — ไส้บนยาว (ปฏิเสธราคาสูง) ที่ Supply zone → SELL"""
    n = len(df)
    if n < 5 or not zones:
        return None
    close = df["close"].values; open_ = df["open"].values
    high = df["high"].values; low = df["low"].values
    atr_now = float(atr[-1])
    if atr_now <= 0:
        return None
    o, c, h, l = open_[-1], close[-1], high[-1], low[-1]
    body = abs(c - o)
    rng = h - l
    if rng <= 0:
        return None
    lower_wick = min(o, c) - l
    upper_wick = h - max(o, c)
    if upper_wick < min_wick_ratio * max(body, 0.01 * atr_now):
        return None
    if upper_wick < 0.5 * rng:
        return None
    if lower_wick > body + 0.1 * atr_now:
        return None
    if rng < 0.3 * atr_now:
        return None
    tol = atr_now * 0.3
    for z in zones:
        if z["type"] != "SUPPLY":
            continue
        zlo, zhi = z["lo"], z["hi"]
        if (zlo - tol) <= h <= (zhi + atr_now):
            return {
                "direction": "short",
                "strategy": "pinbar_short",
                "zone_hi": float(max(zhi, h)),   # SL เหนือไส้ pin
                "zone_lo": float(zlo),
                "reason": (f"Pin Bar (Shooting Star) SHORT @ Supply {zlo:.0f}-{zhi:.0f} "
                           f"wick={upper_wick/atr_now:.2f}ATR body={body/atr_now:.2f}ATR")
            }
    return None


def detect_star_long(df, atr, zones):
    """Morning Star LONG — 3-candle reversal ที่ Demand zone → BUY
    C1 bearish (down) → C2 small body (ลังเล) → C3 bullish ปิดเข้า body C1
    """
    n = len(df)
    if n < 5 or not zones:
        return None
    close = df["close"].values; open_ = df["open"].values
    high = df["high"].values; low = df["low"].values
    atr_now = float(atr[-1])
    if atr_now <= 0:
        return None
    c1o, c1c = open_[-3], close[-3]
    c2o, c2c = open_[-2], close[-2]
    c3o, c3c = open_[-1], close[-1]
    c1_body = c1c - c1o            # < 0 = bearish
    c2_body = abs(c2c - c2o)
    c3_body = c3c - c3o            # > 0 = bullish
    # C1 bearish meaningful
    if c1_body >= 0 or abs(c1_body) < 0.3 * atr_now:
        return None
    # C2 small body (indecision/doji)
    if c2_body > 0.35 * atr_now:
        return None
    # C3 bullish meaningful + ปิดเหนือ midpoint C1 (recovery)
    if c3_body <= 0 or c3_body < 0.3 * atr_now:
        return None
    if c3c < (c1o + c1c) / 2:
        return None
    pattern_low = min(low[-3], low[-2], low[-1])
    tol = atr_now * 0.3
    for z in zones:
        if z["type"] != "DEMAND":
            continue
        zlo, zhi = z["lo"], z["hi"]
        if (zlo - atr_now) <= pattern_low <= (zhi + tol):
            return {
                "direction": "long",
                "strategy": "star_long",
                "zone_hi": float(zhi),
                "zone_lo": float(min(zlo, pattern_low)),
                "reason": (f"Morning Star LONG @ Demand {zlo:.0f}-{zhi:.0f} "
                           f"C1={c1_body/atr_now:.1f} C3={c3_body/atr_now:.1f}ATR")
            }
    return None


def detect_star_short(df, atr, zones):
    """Evening Star SHORT — 3-candle reversal ที่ Supply zone → SELL
    C1 bullish (up) → C2 small body → C3 bearish ปิดเข้า body C1
    """
    n = len(df)
    if n < 5 or not zones:
        return None
    close = df["close"].values; open_ = df["open"].values
    high = df["high"].values; low = df["low"].values
    atr_now = float(atr[-1])
    if atr_now <= 0:
        return None
    c1o, c1c = open_[-3], close[-3]
    c2o, c2c = open_[-2], close[-2]
    c3o, c3c = open_[-1], close[-1]
    c1_body = c1c - c1o            # > 0 = bullish
    c2_body = abs(c2c - c2o)
    c3_body = c3c - c3o            # < 0 = bearish
    if c1_body <= 0 or c1_body < 0.3 * atr_now:
        return None
    if c2_body > 0.35 * atr_now:
        return None
    if c3_body >= 0 or abs(c3_body) < 0.3 * atr_now:
        return None
    if c3c > (c1o + c1c) / 2:
        return None
    pattern_high = max(high[-3], high[-2], high[-1])
    tol = atr_now * 0.3
    for z in zones:
        if z["type"] != "SUPPLY":
            continue
        zlo, zhi = z["lo"], z["hi"]
        if (zlo - tol) <= pattern_high <= (zhi + atr_now):
            return {
                "direction": "short",
                "strategy": "star_short",
                "zone_hi": float(max(zhi, pattern_high)),
                "zone_lo": float(zlo),
                "reason": (f"Evening Star SHORT @ Supply {zlo:.0f}-{zhi:.0f} "
                           f"C1={c1_body/atr_now:.1f} C3={c3_body/atr_now:.1f}ATR")
            }
    return None


def detect_htf_retest_short(df, atr, htf_zones, min_body_atr=0.2):
    """HTF Retest SHORT — ราคา reject ที่ H1/H4 Supply → SELL

    user feedback (6/2 01:08): bot detect H1 supply ไว้ veto long เท่านั้น
    ไม่ได้ใช้เป็น trigger short → พลาด sell ที่ resistance
    แก้: ใช้ HTF supply เป็น trigger — high แตะ supply + bearish reject candle
    """
    n = len(df)
    if n < 5 or not htf_zones:
        return None
    close = df["close"].values; open_ = df["open"].values
    high = df["high"].values; low = df["low"].values
    atr_now = float(atr[-1])
    if atr_now <= 0:
        return None
    # bearish reject candle (ปิดแดง + ไส้บน = ปฏิเสธราคาสูง)
    if close[-1] >= open_[-1]:
        return None
    body = (open_[-1] - close[-1]) / atr_now
    upper_wick = (high[-1] - max(open_[-1], close[-1])) / atr_now
    if body < min_body_atr and upper_wick < 0.3:
        return None
    tol = atr_now * 0.3
    for z in htf_zones:
        if z.get("type") != "SUPPLY":
            continue
        zlo, zhi = z["lo"], z["hi"]
        # high แตะเข้า supply (reject ที่ขอบล่าง-กลาง zone)
        if (zlo - tol) <= high[-1] <= (zhi + atr_now) and close[-1] < zhi:
            return {
                "direction": "short",
                "strategy": "htf_retest_short",
                "zone_hi": float(zhi),
                "zone_lo": float(zlo),
                "reason": (f"HTF Retest SHORT @ {z.get('tf','H?')} Supply {zlo:.0f}-{zhi:.0f} "
                           f"body={body:.2f}ATR wick={upper_wick:.2f}ATR (reject ที่ HTF resistance)")
            }
    return None


def detect_htf_retest_long(df, atr, htf_zones, min_body_atr=0.2):
    """HTF Retest LONG — ราคา bounce ที่ H1/H4 Demand → BUY (mirror)"""
    n = len(df)
    if n < 5 or not htf_zones:
        return None
    close = df["close"].values; open_ = df["open"].values
    high = df["high"].values; low = df["low"].values
    atr_now = float(atr[-1])
    if atr_now <= 0:
        return None
    if close[-1] <= open_[-1]:
        return None
    body = (close[-1] - open_[-1]) / atr_now
    lower_wick = (min(open_[-1], close[-1]) - low[-1]) / atr_now
    if body < min_body_atr and lower_wick < 0.3:
        return None
    tol = atr_now * 0.3
    for z in htf_zones:
        if z.get("type") != "DEMAND":
            continue
        zlo, zhi = z["lo"], z["hi"]
        if (zlo - atr_now) <= low[-1] <= (zhi + tol) and close[-1] > zlo:
            return {
                "direction": "long",
                "strategy": "htf_retest_long",
                "zone_hi": float(zhi),
                "zone_lo": float(zlo),
                "reason": (f"HTF Retest LONG @ {z.get('tf','H?')} Demand {zlo:.0f}-{zhi:.0f} "
                           f"body={body:.2f}ATR wick={lower_wick:.2f}ATR (bounce ที่ HTF support)")
            }
    return None


# ════════════════════════════════════════════════════════════════════════
# 2026-06-18 user playbook: "กระจุก ≥5 แท่ง แล้วมี IMB ยิงออกไป = demand/supply"
# คนละ pattern กับ momentum_breakout (vertical impulse) — อันนี้ต้อง consolidate ก่อน
# entry = close ของแท่ง breakout เลย (ไม่ retest), SL ใต้/เหนือ consolidation
# ════════════════════════════════════════════════════════════════════════
def _cons_flatness(high, low, n):
    """user 06-18 (ภาพ 4b): ครึ่งแรก vs ครึ่งหลังของ consolidation ต้อง flat (ไม่ wedge/sloped)
    คืนค่า drift ratio (0 = flat, 1 = ครึ่งหลังเลื่อน 100% ของ range). ใช้ ≤ 0.35 = flat OK."""
    half = n // 2
    h1 = max(high[-n:-half]) if half > 0 else high[-1]
    l1 = min(low[-n:-half]) if half > 0 else low[-1]
    h2 = max(high[-half:]) if half > 0 else high[-1]
    l2 = min(low[-half:]) if half > 0 else low[-1]
    mid1 = (h1 + l1) / 2.0; mid2 = (h2 + l2) / 2.0
    full_h = max(h1, h2); full_l = min(l1, l2)
    rng = full_h - full_l
    return abs(mid2 - mid1) / rng if rng > 0 else 0.0


def detect_supply_rejection_short(df, atr, zones, min_wicks=2, min_wick_atr=0.3, min_body_atr=0.3):
    """user 06-18 playbook: wicks 3+ ที่ supply + red strong candle = SHORT + flip LONG
    pattern: bot buy ที่ supply → wicks ก่อตัว → IMB red drop → flip sell ทันที"""
    n = len(df)
    if n < 7 or not zones: return None
    h = df["high"].values; l = df["low"].values
    c = df["close"].values; o = df["open"].values
    a = float(atr[-1]) if len(atr) else 1.0
    if a <= 0: return None
    cur_c = float(c[-1]); cur_o = float(o[-1])
    # current bar red strong (IMB drop confirms)
    if cur_c >= cur_o: return None
    body = cur_o - cur_c
    if body < min_body_atr * a: return None
    # หา supply zone ที่ wicks 5 bars แตะอยู่
    peak5 = float(max(h[-6:-1]))
    in_sup = None
    for z in (zones or []):
        if z.get("type") == "SUPPLY":
            if z["lo"] - 0.3*a <= peak5 <= z["hi"] + 0.5*a:
                in_sup = z; break
    if not in_sup: return None
    # นับ upper wicks ที่แตะ supply (last 5 closed bars)
    wc = 0
    for i in range(-6, -1):
        body_top = max(o[i], c[i])
        uw = h[i] - body_top
        if uw > min_wick_atr * a and h[i] >= in_sup["lo"] - 0.3*a:
            wc += 1
    if wc < min_wicks: return None
    return {"direction": "short", "strategy": "supply_rejection_short",
            "zone_hi": in_sup["hi"] + 0.5*a, "zone_lo": cur_c - 1.0*a,
            "reason": f"SUPPLY-REJ SHORT: {wc} wicks @ supply {in_sup['lo']:.1f}-{in_sup['hi']:.1f} + red body {body/a:.1f}ATR (user playbook)"}


def detect_demand_rejection_long(df, atr, zones, min_wicks=2, min_wick_atr=0.3, min_body_atr=0.3):
    """Mirror: wicks 3+ ที่ demand + green strong = LONG + flip SHORT"""
    n = len(df)
    if n < 7 or not zones: return None
    h = df["high"].values; l = df["low"].values
    c = df["close"].values; o = df["open"].values
    a = float(atr[-1]) if len(atr) else 1.0
    if a <= 0: return None
    cur_c = float(c[-1]); cur_o = float(o[-1])
    if cur_c <= cur_o: return None
    body = cur_c - cur_o
    if body < min_body_atr * a: return None
    dip5 = float(min(l[-6:-1]))
    in_dem = None
    for z in (zones or []):
        if z.get("type") == "DEMAND":
            if z["lo"] - 0.5*a <= dip5 <= z["hi"] + 0.3*a:
                in_dem = z; break
    if not in_dem: return None
    wc = 0
    for i in range(-6, -1):
        body_bot = min(o[i], c[i])
        lw = body_bot - l[i]
        if lw > min_wick_atr * a and l[i] <= in_dem["hi"] + 0.3*a:
            wc += 1
    if wc < min_wicks: return None
    return {"direction": "long", "strategy": "demand_rejection_long",
            "zone_hi": cur_c + 1.0*a, "zone_lo": in_dem["lo"] - 0.5*a,
            "reason": f"DEMAND-REJ LONG: {wc} wicks @ demand {in_dem['lo']:.1f}-{in_dem['hi']:.1f} + green body {body/a:.1f}ATR (user playbook)"}


def detect_consolidation_breakout_long(df, atr, cons_bars=8, max_range_atr=2.5,
                                       min_range_atr=0.3, max_drift=0.35, break_min_atr=0.05):
    """user 06-18 playbook: ≥8 แท่งกระจุก flat (drift≤35%) → แท่งล่าสุดปิดเหนือ range = LONG.
    Tuned จาก 5 ภาพ user: range 2-5 ATR ดี, sloped (drift>35%) ไม่ดี (ภาพ 4b)."""
    n = len(df)
    if n < cons_bars + 2:
        return None
    high = df["high"].values; low = df["low"].values
    close = df["close"].values; open_ = df["open"].values
    a = float(atr[-1]) if len(atr) else 1.0
    if a <= 0:
        return None
    cons_h = float(max(high[-cons_bars-1:-1]))
    cons_l = float(min(low[-cons_bars-1:-1]))
    rng = cons_h - cons_l
    if rng <= 0 or rng / a > max_range_atr or rng / a < min_range_atr:
        return None
    # flatness: กัน wedge/sloped (ภาพ 4b user ว่าไม่ดี)
    drift = _cons_flatness(high[-cons_bars-1:-1], low[-cons_bars-1:-1], cons_bars)
    if drift > max_drift:
        return None
    cur_c = float(close[-1]); cur_o = float(open_[-1])
    # close ทะลุ + body bullish (จบแท่ง)
    if cur_c <= cons_h + break_min_atr * a:
        return None
    if cur_c <= cur_o:
        return None
    return {"direction": "long", "strategy": "consolidation_breakout_long",
            "zone_hi": cur_c + 0.5 * a, "zone_lo": cons_l,
            "reason": f"CONS-BREAK LONG: {cons_bars}+bars flat {cons_l:.1f}-{cons_h:.1f} (rng {rng/a:.1f}ATR, drift {drift*100:.0f}%), close@{cur_c:.1f}"}


def detect_consolidation_breakout_short(df, atr, cons_bars=8, max_range_atr=2.5,
                                        min_range_atr=0.3, max_drift=0.35, break_min_atr=0.05):
    """Mirror: ≥8 แท่งกระจุก flat → close ใต้ range = SHORT."""
    n = len(df)
    if n < cons_bars + 2:
        return None
    high = df["high"].values; low = df["low"].values
    close = df["close"].values; open_ = df["open"].values
    a = float(atr[-1]) if len(atr) else 1.0
    if a <= 0:
        return None
    cons_h = float(max(high[-cons_bars-1:-1]))
    cons_l = float(min(low[-cons_bars-1:-1]))
    rng = cons_h - cons_l
    if rng <= 0 or rng / a > max_range_atr or rng / a < min_range_atr:
        return None
    drift = _cons_flatness(high[-cons_bars-1:-1], low[-cons_bars-1:-1], cons_bars)
    if drift > max_drift:
        return None
    cur_c = float(close[-1]); cur_o = float(open_[-1])
    if cur_c >= cons_l - break_min_atr * a:
        return None
    if cur_c >= cur_o:
        return None
    return {"direction": "short", "strategy": "consolidation_breakout_short",
            "zone_hi": cons_h, "zone_lo": cur_c - 0.5 * a,
            "reason": f"CONS-BREAK SHORT: {cons_bars}+bars flat {cons_l:.1f}-{cons_h:.1f} (rng {rng/a:.1f}ATR, drift {drift*100:.0f}%), close@{cur_c:.1f}"}


def detect_momentum_breakout_long(df, atr, base_lookback=12, min_impulse_atr=1.5,
                                  max_base_atr=3.0, min_break_atr=0.2):
    """MOMENTUM BREAKOUT LONG — impulse แรงทะลุ consolidation base (vertical breakout, ไม่ต้อง retest)

    2026-06-12 user (chart 2 ครั้ง): "การขึ้นแบบนี้บอทต้อง buy ตาม momentum"
    rally 4080→4164 พุ่งตรงไม่ retest → breakout_long/momentum_long (ต้องการ retest/pullback) ไม่ยิง
    → บอทได้แต่ไล่ยอด srf_long @4164. detector นี้จับ "ระเบิดขึ้น" เข้าเร็วตอนต้น move

    Pattern:
    1. มี consolidation base ก่อนหน้า (base_lookback bars, range ≤ max_base_atr × ATR = แคบ)
    2. แท่งล่าสุด bullish + body > min_impulse_atr × ATR (impulse แรง = range expansion)
    3. close ทะลุเหนือ base high (+ min_break_atr × ATR ยืนยัน break)
    → LONG ทันที (is_mb → ข้าม zone/room/climax; spike-guard $40 กันไล่ยอด)
    """
    n = len(df)
    if n < base_lookback + 3:
        return None
    close = df["close"].values; open_ = df["open"].values
    high = df["high"].values; low = df["low"].values
    a = float(atr[-1])
    if a <= 0:
        return None
    # base = bars ก่อนแท่งปัจจุบัน (ไม่รวม [-1])
    base_hi = float(high[-base_lookback-1:-1].max())
    base_lo = float(low[-base_lookback-1:-1].min())
    if (base_hi - base_lo) > max_base_atr * a:
        return None  # ไม่ใช่ base แคบ (ตลาด trending อยู่แล้ว ไม่ใช่ breakout จาก consolidation)
    # แท่งล่าสุด = impulse เขียวแรง
    if close[-1] <= open_[-1]:
        return None
    body = (close[-1] - open_[-1]) / a
    if body < min_impulse_atr:
        return None
    # close ต้องทะลุเหนือ base high (clear break)
    if close[-1] < base_hi + min_break_atr * a:
        return None
    return {
        "direction": "long",
        "strategy": "momentum_breakout_long",
        "zone_hi": float(close[-1] + 2.0 * a),
        "zone_lo": float(base_hi - 0.3 * a),
        "reason": f"MOMENTUM BREAKOUT LONG: impulse ทะลุ base {base_hi:.1f} body={body:.2f}ATR (vertical breakout)"
    }


def detect_momentum_breakout_short(df, atr, base_lookback=12, min_impulse_atr=1.5,
                                   max_base_atr=3.0, min_break_atr=0.2):
    """MOMENTUM BREAKDOWN SHORT — impulse แดงแรงทะลุ base ลง (vertical breakdown, ไม่ต้อง retest)

    Mirror ของ detect_momentum_breakout_long (user 2026-06-12: "ทำ short ทุกรอบที่เพิ่ม strategy")
    chart Jun12 00:48-01:05: ยอด spike 4170 → momentum sell 4170→4135 แต่ break_retest_short โดน veto
    (regime=trend_bull lag) → บอทไม่ sell. detector นี้จับ "ระเบิดลง" เข้าเร็ว + ข้าม regime block (is_mb)

    Pattern (mirror):
    1. มี consolidation base ก่อนหน้า (range ≤ max_base_atr × ATR = แคบ)
    2. แท่งล่าสุด bearish + body > min_impulse_atr × ATR (impulse แรง)
    3. close ทะลุใต้ base low (− min_break_atr × ATR ยืนยัน break)
    → SHORT ทันที (is_mb → ข้าม zone/room/climax/regime; spike-guard $40 กันไล่ก้น)
    """
    n = len(df)
    if n < base_lookback + 3:
        return None
    close = df["close"].values; open_ = df["open"].values
    high = df["high"].values; low = df["low"].values
    a = float(atr[-1])
    if a <= 0:
        return None
    base_hi = float(high[-base_lookback-1:-1].max())
    base_lo = float(low[-base_lookback-1:-1].min())
    if (base_hi - base_lo) > max_base_atr * a:
        return None  # ไม่ใช่ base แคบ
    if close[-1] >= open_[-1]:
        return None  # ต้อง bearish
    body = (open_[-1] - close[-1]) / a
    if body < min_impulse_atr:
        return None
    if close[-1] > base_lo - min_break_atr * a:
        return None  # ต้องปิดใต้ base low
    return {
        "direction": "short",
        "strategy": "momentum_breakout_short",
        "zone_hi": float(base_lo + 0.3 * a),
        "zone_lo": float(close[-1] - 2.0 * a),
        "reason": f"MOMENTUM BREAKDOWN SHORT: impulse ทะลุ base {base_lo:.1f} body={body:.2f}ATR (vertical breakdown)"
    }


def detect_breakout_long(df, atr, lookback=40, consolidation_bars=10,
                          break_atr=0.5, retest_tol_atr=0.4):
    """BREAKOUT LONG — ราคา break ผ่าน resistance level แล้ว retest กลับมา → LONG

    Pattern (ตาม chart user):
    1. หา resistance level ที่ราคา reject 2+ ครั้งใน lookback bars (consolidation)
    2. ราคา break ขึ้นเหนือ level (close > level + break_atr ATR)
    3. ราคา pullback กลับมา test level (เข้าใกล้ level ± retest_tol_atr ATR)
    4. มี bullish rejection ที่ level → LONG
    """
    n = len(df)
    if n < lookback + 5: return None
    close = df["close"].values; open_ = df["open"].values
    high = df["high"].values; low = df["low"].values
    atr_now = atr[-1]
    if atr_now <= 0: return None

    # หา resistance level ใน lookback (highest swing high ที่ถูก reject)
    swing_highs = []
    for i in range(n - 5, n - lookback, -1):
        if high[i] >= high[i-1] and high[i] >= high[i+1]:
            swing_highs.append((i, high[i]))
    if len(swing_highs) < 2: return None

    # เลือก level ที่ถูกแตะ 2+ ครั้ง (consolidation level)
    resistance = None
    for idx, hi in swing_highs:
        # นับว่าใน lookback มีกี่ครั้งที่แตะ level นี้ (±0.5 ATR)
        # ขยายจาก 0.3→0.5 เพราะ M1 swing highs อาจห่างกัน $0.5-1 แต่เป็น level เดียวกัน
        touches = sum(1 for _, h in swing_highs if abs(h - hi) <= 0.5 * atr_now)
        if touches >= 2:
            resistance = hi
            break
    if resistance is None: return None

    # ต้องมี break ที่ผ่าน level (มี close ใน 5-20 bars ที่ผ่านมาสูงกว่า resistance + break_atr ATR)
    # ขยาย 15→20 bars: breakout อาจเกิดก่อน retest 15-20 M1 bars
    recent_max_close = close[-20:-1].max() if n >= 21 else close[:-1].max()
    if recent_max_close < resistance + break_atr * atr_now: return None  # ไม่มี break ชัด

    # ตอนนี้ราคา pullback กลับมา test level (close[-1] ใกล้ resistance)
    if not (resistance - retest_tol_atr * atr_now <= low[-1] <= resistance + retest_tol_atr * atr_now):
        return None

    # Bullish rejection candle (close > open + body > 0.25 ATR)
    if close[-1] <= open_[-1]: return None
    body = abs(close[-1] - open_[-1]) / atr_now
    if body < 0.25: return None

    return {"direction": "long", "strategy": "breakout_long",
            "zone_hi": float(resistance + retest_tol_atr * atr_now),
            "zone_lo": float(resistance - retest_tol_atr * atr_now),
            "reason": f"BREAKOUT LONG: resistance {resistance:.2f} broken→retest body={body:.2f}ATR"}


def detect_breakout_short(df, atr, lookback=40, consolidation_bars=10,
                           break_atr=0.5, retest_tol_atr=0.4):
    """BREAKOUT SHORT — ราคา break ผ่าน support level แล้ว retest กลับมา → SHORT
    Mirror ของ breakout_long
    """
    n = len(df)
    if n < lookback + 5: return None
    close = df["close"].values; open_ = df["open"].values
    high = df["high"].values; low = df["low"].values
    atr_now = atr[-1]
    if atr_now <= 0: return None

    # หา support level (swing lows)
    swing_lows = []
    for i in range(n - 5, n - lookback, -1):
        if low[i] <= low[i-1] and low[i] <= low[i+1]:
            swing_lows.append((i, low[i]))
    if len(swing_lows) < 2: return None

    support = None
    for idx, lo in swing_lows:
        touches = sum(1 for _, l in swing_lows if abs(l - lo) <= 0.5 * atr_now)
        if touches >= 2:
            support = lo
            break
    if support is None: return None

    # ต้องมี break ลง (close ใน 5-20 bars ที่ผ่านมาต่ำกว่า support - break_atr ATR)
    recent_min_close = close[-20:-1].min() if n >= 21 else close[:-1].min()
    if recent_min_close > support - break_atr * atr_now: return None

    # ราคา pullback กลับมา test level
    if not (support - retest_tol_atr * atr_now <= high[-1] <= support + retest_tol_atr * atr_now):
        return None

    # Bearish rejection candle
    if close[-1] >= open_[-1]: return None
    body = abs(close[-1] - open_[-1]) / atr_now
    if body < 0.25: return None

    return {"direction": "short", "strategy": "breakout_short",
            "zone_hi": float(support + retest_tol_atr * atr_now),
            "zone_lo": float(support - retest_tol_atr * atr_now),
            "reason": f"BREAKOUT SHORT: support {support:.2f} broken→retest body={body:.2f}ATR"}


# ═══════════════════════════════════════════════════════════════════════
# Zone Retest Strategy (SMC Zone-First)
# ปรัชญา: User SMC framework — หา zone ก่อน แล้วรอราคา pullback มา retest
#
# หลักการ: Demand zone = swing low ที่ bounce ≥ 2 ATR
#          เมื่อราคา pullback กลับมาแตะ zone + มี bullish reaction → BUY
#          Supply zone = swing high ที่ reject ≥ 2 ATR
#          เมื่อราคา pullback กลับมาแตะ zone + มี bearish reaction → SELL
#
# ต่างจาก DSF: DSF ต้อง break + flip zone ก่อน (เข้มกว่า)
# Zone Retest: เข้าที่ zone เดิมที่ยังไม่ถูก break (เร็วกว่า)
# ═══════════════════════════════════════════════════════════════════════

def detect_zone_retest_long(df, atr, zones, min_body_atr=0.2):
    """Zone Retest LONG — ราคา pullback ลงมาแตะ Demand zone + bullish reaction

    เงื่อนไข:
    1. ราคาอยู่ใน/ใกล้ Demand zone (zone_lo - tol ≤ low ≤ zone_hi + tol)
    2. Low ของแท่งปัจจุบันแตะเข้า zone (wick test)
    3. Close อยู่เหนือ zone_lo (ไม่ break ลง)
    4. แท่งปัจจุบัน bullish (close > open) = reaction
    5. ไม่ใช่แท่งเล็กเกินไป (body ≥ min_body_atr)
    """
    n = len(df)
    if n < 10 or not zones:
        return None
    close = df["close"].values
    open_ = df["open"].values
    high = df["high"].values
    low = df["low"].values
    atr_now = atr[-1]
    if atr_now <= 0:
        return None

    # Current candle must be bullish
    if close[-1] <= open_[-1]:
        return None
    body = (close[-1] - open_[-1]) / atr_now
    if body < min_body_atr:
        return None

    # Lower wick should show rejection (optional but improves quality)
    lower_wick = (min(close[-1], open_[-1]) - low[-1]) / atr_now

    for z in zones:
        if z["type"] != "DEMAND":
            continue
        zlo = z["lo"]
        zhi = z["hi"]
        # 2026-06-06 strict — mirror short fix
        tol = atr_now * 0.3

        # ราคาแท่งปัจจุบันต้องแตะโซนจริง (low ใน zone area)
        if low[-1] > zhi + tol or low[-1] < zlo - atr_now * 0.5:
            continue  # ห่างเกินหรือทะลุไปแล้ว
        if close[-1] < zlo:
            continue  # closed below zone = zone broken
        # close ต้องไม่ลอยไปไกลโซน
        if close[-1] > zhi + atr_now * 0.5:
            continue
        # ต้องมี reject confirmation: ไส้ล่างชัด AND body bullish > 0.3 ATR
        if lower_wick < 0.3 or body < 0.3:
            continue

        return {
            "direction": "long",
            "strategy": "zone_retest_long",
            "zone_hi": zhi,
            "zone_lo": zlo,
            "reason": f"Zone Retest LONG: Demand {zlo:.0f}-{zhi:.0f} "
                      f"body={body:.2f}ATR wick={lower_wick:.2f}ATR"
        }

    return None


def detect_zone_retest_short(df, atr, zones, min_body_atr=0.2):
    """Zone Retest SHORT — ราคา pullback ขึ้นมาแตะ Supply zone + bearish reaction

    Mirror ของ zone_retest_long
    """
    n = len(df)
    if n < 10 or not zones:
        return None
    close = df["close"].values
    open_ = df["open"].values
    high = df["high"].values
    low = df["low"].values
    atr_now = atr[-1]
    if atr_now <= 0:
        return None

    # Current candle must be bearish
    if close[-1] >= open_[-1]:
        return None
    body = (open_[-1] - close[-1]) / atr_now
    if body < min_body_atr:
        return None

    # Upper wick should show rejection
    upper_wick = (high[-1] - max(close[-1], open_[-1])) / atr_now

    for z in zones:
        if z["type"] != "SUPPLY":
            continue
        zlo = z["lo"]
        zhi = z["hi"]
        # 2026-06-06 user: "บอทเข้ามั่ว" — verified 12 trades 8% WR
        # ราก: tol 1.0 ATR + body_below = ยิงตอนราคาทะลุโซนไปไกลแล้ว = chase ไม่ใช่ retest
        # แก้ STRICT: ราคาต้องอยู่ "ใน/ใกล้ขอบ" โซน + ต้องมี reject candle ชัด
        tol = atr_now * 0.3

        # ราคาแท่งปัจจุบันต้องแตะโซนจริง (high ใน zone area)
        if high[-1] < zlo - tol or high[-1] > zhi + atr_now * 0.5:
            continue  # ห่างเกินหรือทะลุไปแล้ว
        if close[-1] > zhi:
            continue  # closed above zone = zone broken (ไม่ใช่ retest)
        # close ต้องไม่หล่นลงไกลโซน — ถ้าหล่น = chase, ไม่ใช่ retest
        if close[-1] < zlo - atr_now * 0.5:
            continue
        # ต้องมี reject confirmation: ไส้บนชัด AND body bearish > 0.3 ATR
        if upper_wick < 0.3 or body < 0.3:
            continue

        return {
            "direction": "short",
            "strategy": "zone_retest_short",
            "zone_hi": zhi,
            "zone_lo": zlo,
            "reason": f"Zone Retest SHORT: Supply {zlo:.0f}-{zhi:.0f} "
                      f"body={body:.2f}ATR wick={upper_wick:.2f}ATR"
        }

    return None


# ═══════════════════════════════════════════════════════════════════════
# Zone First-Touch Strategy
# ปรัชญา: ราคาเข้าแตะ Supply/Demand zone เป็นครั้งแรก (หรือหลัง gap)
#          → เข้าตามทิศทาง zone ทันทีที่มี rejection candle
#
# ต่างจาก zone_retest: retest = ราคาเคย test ไปแล้ว แล้วกลับมา
#                       first_touch = ราคาเพิ่งเข้า zone ครั้งแรก
#
# SMC: "First touch of Supply zone = high probability SELL"
#      เพราะ smart money ยังไม่ได้ consume orders ทั้งหมดใน zone
# ═══════════════════════════════════════════════════════════════════════


def detect_zone_first_touch_short(df, atr, zones, min_body_atr=0.15, lookback_fresh=20):
    """Zone First-Touch SHORT — ราคาแตะ Supply zone ครั้งแรก (หรือหลัง gap) + bearish reaction → SELL

    เงื่อนไข:
    1. ราคาเพิ่งเข้า Supply zone (high[-1] ≥ zone_lo และ candle ก่อนๆ อยู่ต่ำกว่า zone)
    2. Candle ปัจจุบัน bearish (close < open)
    3. ไม่มี bar ใน lookback_fresh bars ก่อนหน้าที่เข้า zone นี้แล้ว (zone fresh)
    4. Close ยังไม่ทะลุเหนือ zone_hi (ยังอยู่ใน zone หรือ reject)
    """
    n = len(df)
    if n < lookback_fresh + 5 or not zones:
        return None

    close = df["close"].values
    open_ = df["open"].values
    high  = df["high"].values
    low   = df["low"].values
    atr_now = float(atr[-1])
    if atr_now <= 0:
        return None

    # Current candle must be bearish
    if close[-1] >= open_[-1]:
        return None
    body = (open_[-1] - close[-1]) / atr_now
    if body < min_body_atr:
        return None

    upper_wick = (high[-1] - max(open_[-1], close[-1])) / atr_now

    for z in zones:
        if z["type"] != "SUPPLY":
            continue
        zlo, zhi = z["lo"], z["hi"]
        tol = atr_now * 0.3

        # Current candle high must enter zone
        if high[-1] < zlo - tol:
            continue
        if close[-1] > zhi + tol:
            continue  # pushed too far above zone

        # Zone freshness: ตรวจว่าในช่วง lookback_fresh bars ก่อนหน้า
        # ราคาอยู่ต่ำกว่า zone หรือเปล่า (= zone ยัง fresh ไม่ถูก test มาก)
        recent_touches = sum(
            1 for i in range(max(0, n - lookback_fresh - 1), n - 1)
            if high[i] >= zlo - tol
        )
        # ถ้า touch น้อย (≤ 2) = first/fresh touch
        # ถ้า touch มาก = zone ถูก tested หลายครั้งแล้ว → ใช้ zone_retest แทน
        if recent_touches > 3:
            continue  # zone worn out ใช้ zone_retest_short แทน

        return {
            "direction": "short",
            "strategy": "zone_first_touch_short",
            "zone_hi": float(zhi),
            "zone_lo": float(zlo),
            "reason": (f"Zone First Touch SHORT: Supply {zlo:.0f}-{zhi:.0f} "
                       f"fresh_touches={recent_touches} "
                       f"body={body:.2f}ATR wick={upper_wick:.2f}ATR")
        }

    return None


def detect_zone_first_touch_long(df, atr, zones, min_body_atr=0.15, lookback_fresh=20):
    """Zone First-Touch LONG — ราคาแตะ Demand zone ครั้งแรก + bullish reaction → BUY

    Mirror ของ detect_zone_first_touch_short
    """
    n = len(df)
    if n < lookback_fresh + 5 or not zones:
        return None

    close = df["close"].values
    open_ = df["open"].values
    high  = df["high"].values
    low   = df["low"].values
    atr_now = float(atr[-1])
    if atr_now <= 0:
        return None

    # Current candle must be bullish
    if close[-1] <= open_[-1]:
        return None
    body = (close[-1] - open_[-1]) / atr_now
    if body < min_body_atr:
        return None

    lower_wick = (min(open_[-1], close[-1]) - low[-1]) / atr_now

    for z in zones:
        if z["type"] != "DEMAND":
            continue
        zlo, zhi = z["lo"], z["hi"]
        tol = atr_now * 0.3

        # Current candle low must enter/touch zone
        if low[-1] > zhi + tol:
            continue
        if close[-1] < zlo - tol:
            continue  # pushed too far below zone

        # Zone freshness check
        recent_touches = sum(
            1 for i in range(max(0, n - lookback_fresh - 1), n - 1)
            if low[i] <= zhi + tol
        )
        if recent_touches > 3:
            continue  # use zone_retest_long instead

        return {
            "direction": "long",
            "strategy": "zone_first_touch_long",
            "zone_hi": float(zhi),
            "zone_lo": float(zlo),
            "reason": (f"Zone First Touch LONG: Demand {zlo:.0f}-{zhi:.0f} "
                       f"fresh_touches={recent_touches} "
                       f"body={body:.2f}ATR wick={lower_wick:.2f}ATR")
        }

    return None


def detect_srf_bounce_long(df, atr, lookback=60, min_touches=3, min_body_atr=0.25):
    """SRF Bounce LONG — ราคาแตะ horizontal support แข็ง (3+ touches) + bullish bounce → BUY

    ต่างจาก srf_long: ไม่ต้องการ breakout ก่อน — แค่ support แข็งที่ทดสอบหลายครั้ง + เด้ง
    ต่างจาก zone_first_touch: ใช้ horizontal multi-touch level (จับ SRF ที่ zone detection ไม่เห็น)
    (user framework: ราคาแตะ SRF support แข็ง + M5 bounce → buy bounce)
    """
    n = len(df)
    if n < lookback + 5:
        return None
    close = df["close"].values
    open_ = df["open"].values
    high  = df["high"].values
    low   = df["low"].values
    atr_now = float(atr[-1])
    if atr_now <= 0:
        return None

    # Current candle = bullish bounce (body พอ + ไส้ล่าง = rejection)
    if close[-1] <= open_[-1]:
        return None
    body = (close[-1] - open_[-1]) / atr_now
    lower_wick = (min(open_[-1], close[-1]) - low[-1]) / atr_now
    if body < min_body_atr and lower_wick < 0.3:
        return None

    tol = atr_now * 0.3
    scan_start = max(0, n - lookback)
    scan_end = n - 2

    # หา swing lows ใน window
    swing_lows = []
    for i in range(scan_start + 2, scan_end - 1):
        if (low[i] <= low[i-1] and low[i] <= low[i-2]
                and low[i] <= low[i+1] and low[i] <= low[i+2]):
            swing_lows.append(low[i])
    if len(swing_lows) < min_touches:
        return None

    # cluster swing lows → หา level ที่มี touches >= min_touches
    for lvl in swing_lows:
        cluster = [s for s in swing_lows if abs(s - lvl) <= tol]
        if len(cluster) < min_touches:
            continue
        support = min(cluster)   # ใช้ low สุดของ cluster เป็น support
        # current low ต้องแตะ/ใกล้ support (retest) + ไม่ทะลุลง
        if low[-1] > support + atr_now * 0.6:
            continue
        if close[-1] < support - tol:
            continue
        return {
            "direction": "long",
            "strategy": "srf_bounce_long",
            "zone_hi": float(support + 0.4 * atr_now),
            "zone_lo": float(support),
            "reason": (f"SRF Bounce LONG: support {support:.0f} touches={len(cluster)} "
                       f"body={body:.2f}ATR wick={lower_wick:.2f}ATR (เด้งจาก support แข็ง)")
        }
    return None


def detect_srf_bounce_short(df, atr, lookback=60, min_touches=3, min_body_atr=0.25):
    """SRF Bounce SHORT — ราคาแตะ horizontal resistance แข็ง (3+ touches) + bearish reject → SELL

    Mirror ของ detect_srf_bounce_long (balance rule)
    """
    n = len(df)
    if n < lookback + 5:
        return None
    close = df["close"].values
    open_ = df["open"].values
    high  = df["high"].values
    low   = df["low"].values
    atr_now = float(atr[-1])
    if atr_now <= 0:
        return None

    # Current candle = bearish reject (body พอ + ไส้บน = rejection)
    if close[-1] >= open_[-1]:
        return None
    body = (open_[-1] - close[-1]) / atr_now
    upper_wick = (high[-1] - max(open_[-1], close[-1])) / atr_now
    if body < min_body_atr and upper_wick < 0.3:
        return None

    tol = atr_now * 0.3
    scan_start = max(0, n - lookback)
    scan_end = n - 2

    swing_highs = []
    for i in range(scan_start + 2, scan_end - 1):
        if (high[i] >= high[i-1] and high[i] >= high[i-2]
                and high[i] >= high[i+1] and high[i] >= high[i+2]):
            swing_highs.append(high[i])
    if len(swing_highs) < min_touches:
        return None

    for lvl in swing_highs:
        cluster = [s for s in swing_highs if abs(s - lvl) <= tol]
        if len(cluster) < min_touches:
            continue
        resist = max(cluster)
        if high[-1] < resist - atr_now * 0.6:
            continue
        if close[-1] > resist + tol:
            continue
        return {
            "direction": "short",
            "strategy": "srf_bounce_short",
            "zone_hi": float(resist),
            "zone_lo": float(resist - 0.4 * atr_now),
            "reason": (f"SRF Bounce SHORT: resistance {resist:.0f} touches={len(cluster)} "
                       f"body={body:.2f}ATR wick={upper_wick:.2f}ATR (reject จาก resistance แข็ง)")
        }
    return None


# ═══════════════════════════════════════════════════════════════════════
# SRF (Support-Resistance Flip) Strategy
# ปรัชญา: เมื่อราคา breakout ทะลุ resistance → resistance กลายเป็น support
#          เมื่อ pullback กลับมาแตะจุดเดิม + มี bullish reaction → BUY
#          (Mirror: breakdown support → support กลายเป็น resistance → SELL)
#
# หลักการ SMC: "Flip Zone" — Smart money สร้างแนว S/R
#   เมื่อ breakout แสดงว่า S/R ถูก consume → role flip
#   Pullback retest = institutional re-entry point
# ═══════════════════════════════════════════════════════════════════════


def detect_srf_long(df, atr, lookback=60, min_body_atr=0.2, min_touches=2):
    """SRF LONG — Resistance flipped to Support, breakout → pullback → BUY

    Logic:
    1. หา swing high (resistance) ใน lookback window
       - ต้องมี >= min_touches ที่ราคา high เข้าใกล้ระดับนี้ (±0.3 ATR)
    2. ราคา breakout ทะลุขึ้นไปเหนือ resistance (close > resistance)
    3. ราคา pullback กลับลงมาแตะ resistance เดิม (ตอนนี้เป็น support)
       - low ของ candle ปัจจุบัน ≤ resistance + 0.5 ATR
    4. Candle ปัจจุบัน = bullish reaction (close > open)
    5. Close อยู่เหนือ resistance (ไม่ break กลับลง)
    """
    n = len(df)
    if n < lookback + 10:
        return None

    close = df["close"].values
    open_ = df["open"].values
    high = df["high"].values
    low = df["low"].values
    atr_now = float(atr[-1])
    if atr_now <= 0:
        return None

    # Current candle must be bullish
    if close[-1] <= open_[-1]:
        return None
    body = (close[-1] - open_[-1]) / atr_now
    if body < min_body_atr:
        return None

    # Step 1: Find resistance levels in lookback window
    # Look for swing highs (local maxima within ±3 bars)
    scan_start = max(0, n - lookback)
    scan_end = n - 5  # exclude last 5 bars (these are the breakout/retest area)

    resistance_levels = []
    for i in range(scan_start + 3, scan_end):
        # Is this a local swing high? (higher than ±3 neighbors)
        if (high[i] >= max(high[i-3:i]) and
                high[i] >= max(high[i+1:i+4])):
            resistance_levels.append((i, high[i]))

    if not resistance_levels:
        return None

    # Step 2: For each resistance, check for SRF pattern
    tol = atr_now * 0.3  # tolerance for "touching" a level

    for res_idx, res_level in resistance_levels:
        # Count touches at this level (highs within tolerance)
        touches = 0
        for i in range(scan_start, scan_end):
            if abs(high[i] - res_level) <= tol:
                touches += 1
        if touches < min_touches:
            continue

        # Step 2: Check breakout — at least one candle after resistance
        # had close > resistance level
        breakout_found = False
        breakout_idx = None
        for i in range(res_idx + 1, n - 1):
            if close[i] > res_level + tol:
                breakout_found = True
                breakout_idx = i
                break

        if not breakout_found or breakout_idx is None:
            continue

        # Step 3: After breakout, price must pull back to near resistance
        # Current candle low should be near/at the old resistance (now support)
        if low[-1] > res_level + atr_now * 0.8:
            continue  # too far above — not a retest
        if close[-1] < res_level - tol:
            continue  # closed below resistance — level broken back down

        # Step 4: Make sure there's some gap between breakout and retest
        # (not just a straight break-through)
        bars_since_breakout = (n - 1) - breakout_idx
        if bars_since_breakout < 2:
            continue  # need at least 2 bars between breakout and retest

        # Step 5: Check that price actually went higher after breakout
        # (confirms the breakout was real)
        max_after_breakout = max(high[breakout_idx:n])
        if max_after_breakout < res_level + atr_now * 0.5:
            continue  # breakout wasn't convincing

        # Lower wick quality (rejection from support)
        lower_wick = (min(close[-1], open_[-1]) - low[-1]) / atr_now

        return {
            "direction": "long",
            "strategy": "srf_long",
            "zone_hi": float(res_level + tol),
            "zone_lo": float(res_level - tol),
            "reason": (f"SRF LONG: R→S flip @ {res_level:.5g} "
                       f"touches={touches} breakout_bars={bars_since_breakout} "
                       f"body={body:.2f}ATR wick={lower_wick:.2f}ATR")
        }

    return None


def detect_srf_short(df, atr, lookback=60, min_body_atr=0.2, min_touches=2):
    """SRF SHORT — Support flipped to Resistance, breakdown → pullback → SELL

    Mirror ของ srf_long:
    1. หา swing low (support) ที่ราคา low แตะหลายครั้ง
    2. ราคา breakdown ทะลุลงใต้ support
    3. Pullback กลับขึ้นมาแตะ support เดิม (ตอนนี้เป็น resistance)
    4. Bearish reaction candle → SELL
    """
    n = len(df)
    if n < lookback + 10:
        return None

    close = df["close"].values
    open_ = df["open"].values
    high = df["high"].values
    low = df["low"].values
    atr_now = float(atr[-1])
    if atr_now <= 0:
        return None

    # Current candle must be bearish
    if close[-1] >= open_[-1]:
        return None
    body = (open_[-1] - close[-1]) / atr_now
    if body < min_body_atr:
        return None

    # Step 1: Find support levels (swing lows)
    scan_start = max(0, n - lookback)
    scan_end = n - 5

    support_levels = []
    for i in range(scan_start + 3, scan_end):
        if (low[i] <= min(low[i-3:i]) and
                low[i] <= min(low[i+1:i+4])):
            support_levels.append((i, low[i]))

    if not support_levels:
        return None

    tol = atr_now * 0.3

    for sup_idx, sup_level in support_levels:
        # Count touches
        touches = 0
        for i in range(scan_start, scan_end):
            if abs(low[i] - sup_level) <= tol:
                touches += 1
        if touches < min_touches:
            continue

        # Check breakdown
        breakdown_found = False
        breakdown_idx = None
        for i in range(sup_idx + 1, n - 1):
            if close[i] < sup_level - tol:
                breakdown_found = True
                breakdown_idx = i
                break

        if not breakdown_found or breakdown_idx is None:
            continue

        # Pullback: high of current candle near old support (now resistance)
        if high[-1] < sup_level - atr_now * 0.8:
            continue  # too far below
        if close[-1] > sup_level + tol:
            continue  # closed above support — level reclaimed

        # Gap between breakdown and retest
        bars_since_breakdown = (n - 1) - breakdown_idx
        if bars_since_breakdown < 2:
            continue

        # Confirm breakdown was real
        min_after_breakdown = min(low[breakdown_idx:n])
        if min_after_breakdown > sup_level - atr_now * 0.5:
            continue

        upper_wick = (high[-1] - max(close[-1], open_[-1])) / atr_now

        return {
            "direction": "short",
            "strategy": "srf_short",
            "zone_hi": float(sup_level + tol),
            "zone_lo": float(sup_level - tol),
            "reason": (f"SRF SHORT: S→R flip @ {sup_level:.5g} "
                       f"touches={touches} breakdown_bars={bars_since_breakdown} "
                       f"body={body:.2f}ATR wick={upper_wick:.2f}ATR")
        }

    return None


# ═══════════════════════════════════════════════════════════════════════
# Momentum Continuation Strategy
# ปรัชญา: เมื่อ M15 trend ชัดเจน (EMA8 > EMA21) + ราคา pullback มา EMA
#          + แล้วมี rejection candle → เข้าตาม momentum ที่ pullback
#
# หลักการ SMC: "Order Flow" — เมื่อ smart money ดันราคาขึ้น
#   ราคาจะทำ Wave ขึ้น → pullback เล็กๆ → Wave ขึ้นต่อ
#   จุดเข้าที่ดี = ตอน pullback ลงมาใกล้ EMA + มี rejection bounce
# ═══════════════════════════════════════════════════════════════════════

def detect_momentum_long(df, atr, ema_fast=8, ema_slow=21, min_pullback_atr=0.3):
    """Momentum Continuation LONG — ราคา pullback ใน uptrend แล้ว bounce

    เงื่อนไข:
    1. M15 EMA(8) > EMA(21) → uptrend ชัดเจน
    2. EMA(8) slope > 0 (ยังเป็นขาขึ้น)
    3. ราคา low ลงมาใกล้ EMA(8) หรือ EMA(21) (pullback)
    4. 3 จาก 5 bars สุดท้ายทำ higher-low (momentum structure)
    5. Bar สุดท้ายเป็น bullish rejection (close > open, lower wick)
    """
    n = len(df)
    if n < ema_slow + 10:
        return None

    close = df["close"].values
    open_ = df["open"].values
    high  = df["high"].values
    low   = df["low"].values
    atr_now = float(atr[-1])
    if atr_now <= 0:
        return None

    # EMA calculation
    def _ema_arr(vals, period):
        k = 2.0 / (period + 1)
        out = [vals[0]]
        for v in vals[1:]:
            out.append(v * k + out[-1] * (1 - k))
        return out

    ema_f = _ema_arr(close[-(ema_slow+5):].tolist(), ema_fast)
    ema_s = _ema_arr(close[-(ema_slow+5):].tolist(), ema_slow)

    # 1) EMA(8) > EMA(21) — uptrend
    if ema_f[-1] <= ema_s[-1]:
        return None

    # 2) EMA(8) slope positive (ยังขึ้นอยู่)
    if ema_f[-1] <= ema_f[-3]:
        return None

    # 3) Pullback: bar ล่าสุด low ลงมาใกล้ EMA (within 1.5 ATR of EMA8)
    pullback_dist = low[-1] - ema_f[-1]
    if pullback_dist > 1.0 * atr_now:  # ยังห่าง EMA มาก → ไม่ pullback
        return None
    if pullback_dist < -2.0 * atr_now:  # ลงต่ำกว่า EMA มากเกินไป → trend อาจ break
        return None

    # 4) Higher-low structure: 3 จาก 5 bars สุดท้ายทำ higher low
    if n >= 6:
        recent_lows = low[-5:]
        hl_count = sum(1 for i in range(1, len(recent_lows))
                       if recent_lows[i] > recent_lows[i-1])
        if hl_count < 3:
            return None
    else:
        return None

    # 5) Bullish rejection candle: close > open + lower wick > body
    if close[-1] <= open_[-1]:
        return None
    body = close[-1] - open_[-1]
    lower_wick = min(open_[-1], close[-1]) - low[-1]
    if body < min_pullback_atr * atr_now:
        return None

    # SL = just below EMA8 (fast line = pullback support)
    # ถ้าราคา close ต่ำกว่า EMA8 → setup invalid → SL ที่ EMA8 สมเหตุสมผล
    # ไม่ใช้ EMA21 เป็น zone_lo เพราะ EMA21 ไกลเกิน → SL กว้างโดยไม่จำเป็น
    zone_lo = float(ema_f[-1])           # SL just below EMA8
    zone_hi = float(max(ema_f[-1], close[-1]))   # TP reference: above EMA8

    return {
        "direction": "long",
        "strategy": "momentum_long",
        "zone_hi": zone_hi,
        "zone_lo": zone_lo,
        "reason": (f"Momentum LONG: EMA8={ema_f[-1]:.0f}>EMA21={ema_s[-1]:.0f} "
                   f"pullback={pullback_dist/atr_now:.1f}ATR "
                   f"body={body/atr_now:.2f}ATR HL={hl_count}/4"),
    }


def detect_momentum_short(df, atr, ema_fast=8, ema_slow=21, min_pullback_atr=0.3):
    """Momentum Continuation SHORT — ราคา pullback ใน downtrend แล้ว reject

    Mirror ของ momentum_long:
    1. EMA(8) < EMA(21) → downtrend
    2. EMA(8) slope < 0
    3. ราคา high ขึ้นมาใกล้ EMA(8) (pullback)
    4. 3 จาก 5 bars ทำ lower-high
    5. Bar สุดท้ายเป็น bearish rejection (close < open, upper wick)
    """
    n = len(df)
    if n < ema_slow + 10:
        return None

    close = df["close"].values
    open_ = df["open"].values
    high  = df["high"].values
    low   = df["low"].values
    atr_now = float(atr[-1])
    if atr_now <= 0:
        return None

    def _ema_arr(vals, period):
        k = 2.0 / (period + 1)
        out = [vals[0]]
        for v in vals[1:]:
            out.append(v * k + out[-1] * (1 - k))
        return out

    ema_f = _ema_arr(close[-(ema_slow+5):].tolist(), ema_fast)
    ema_s = _ema_arr(close[-(ema_slow+5):].tolist(), ema_slow)

    # 1) EMA(8) < EMA(21) — downtrend
    if ema_f[-1] >= ema_s[-1]:
        return None

    # 2) EMA(8) slope negative
    if ema_f[-1] >= ema_f[-3]:
        return None

    # 3) Pullback: bar ล่าสุด high ขึ้นมาใกล้ EMA (within 1.5 ATR of EMA8)
    pullback_dist = ema_f[-1] - high[-1]
    if pullback_dist > 1.0 * atr_now:
        return None
    if pullback_dist < -2.0 * atr_now:
        return None

    # 4) Lower-high structure: 3 จาก 5 bars ทำ lower high
    if n >= 6:
        recent_highs = high[-5:]
        lh_count = sum(1 for i in range(1, len(recent_highs))
                       if recent_highs[i] < recent_highs[i-1])
        if lh_count < 3:
            return None
    else:
        return None

    # 5) Bearish rejection candle
    if close[-1] >= open_[-1]:
        return None
    body = open_[-1] - close[-1]
    upper_wick = high[-1] - max(open_[-1], close[-1])
    if body < min_pullback_atr * atr_now:
        return None

    # SL = just above EMA8 (fast line = pullback resistance)
    # ถ้าราคา close สูงกว่า EMA8 → setup invalid
    zone_hi = float(ema_f[-1])           # SL just above EMA8
    zone_lo = float(min(ema_f[-1], close[-1]))   # TP reference: below EMA8

    return {
        "direction": "short",
        "strategy": "momentum_short",
        "zone_hi": zone_hi,
        "zone_lo": zone_lo,
        "reason": (f"Momentum SHORT: EMA8={ema_f[-1]:.0f}<EMA21={ema_s[-1]:.0f} "
                   f"pullback={pullback_dist/atr_now:.1f}ATR "
                   f"body={body/atr_now:.2f}ATR LH={lh_count}/4"),
    }


def load_regime_classifier(path=REGIME_CLASSIFIER_PATH):
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def classify_regime(df, classifier):
    if classifier is None or len(df) < 100:
        return -1
    close = df["close"].values
    returns = np.diff(close, prepend=close[0]) / np.where(close != 0, close, 1.0)
    last_50 = pd.Series(returns[-50:])
    skew = last_50.skew() if not np.isnan(last_50.skew()) else 0
    kurt = last_50.kurt() if not np.isnan(last_50.kurt()) else 0
    ac1 = last_50.autocorr(lag=1)
    if np.isnan(ac1): ac1 = 0.0
    chunk = close[-100:]
    lags = list(range(2, 20))
    tau_vals = []
    for lag in lags:
        sd = np.std(chunk[lag:] - chunk[:-lag])
        if sd > 0:
            tau_vals.append((lag, sd))
    if len(tau_vals) >= 5:
        ls = np.array([v[0] for v in tau_vals])
        ts = np.array([v[1] for v in tau_vals])
        hurst = np.polyfit(np.log(ls), np.log(ts), 1)[0] * 2.0
    else:
        hurst = 0.5

    feat = np.array([hurst, ac1, skew, kurt, 0, 0, 0.002, 1.0])
    feat = np.nan_to_num(feat, nan=0, posinf=1e6, neginf=-1e6)
    feat = np.clip(feat, -1e4, 1e4)
    scaled = classifier["scaler"].transform(feat.reshape(1, -1))
    return int(classifier["gmm"].predict(scaled)[0])


# ═══════════════════════════════════════════════════════════════════════
# CLEAN DECISION — rule เดียวที่สอดคล้องกัน (แทน 26 guards ที่ขัดกัน)
# user logic จากภาพ ~30 รูป สรุปเป็น 3 gate:
#   1. ทิศตรงโซน: LONG ที่ demand, SHORT ที่ supply (ห้ามซื้อชนเพดาน/ขายชนพื้น)
#   2. อย่าจับมีดตก: reversal entry สวน M5 momentum แรง → block
#   3. มี room: opposing zone ห่าง ≥ $2 (กัน buy top / sell bottom)
# momentum/breakout = ยกเว้น gate 1,2 (เทรดตามทิศได้)
# reversal pattern (pinbar/star/srf_bounce/htf_retest) = ยกเว้น gate 2 (จับ reversal)
# ═══════════════════════════════════════════════════════════════════════

_MOMENTUM_BREAKOUT = {
    "momentum_long", "momentum_short", "breakout_long", "breakout_short",
    "dsf_long", "dsf_short", "inside_bar_break", "inside_bar_break_short",
    "momentum_breakout_long", "momentum_breakout_short",  # 2026-06-12: vertical impulse breakout (is_mb)
}
_REVERSAL_PATTERN = {
    "pinbar_long", "pinbar_short", "star_long", "star_short",
    "srf_bounce_long", "srf_bounce_short", "htf_retest_long", "htf_retest_short",
    "srf_long", "srf_short",
}
# STRONG reversal = candle ยืนยันแรงพอจะสวนเทรนได้ (pinbar/star เท่านั้น)
# htf_retest/srf_bounce/srf/zone = ต้องตามเทรน (ไม่ buy dip ในขาลง / ไม่ sell rally ในขาขึ้น)
# user (ภาพ 06-03): ขาลง → SELL ที่ rally/supply, อย่า buy demand ที่ล้มซ้ำ ("DSF")
_STRONG_REVERSAL = {
    "pinbar_long", "pinbar_short", "star_long", "star_short",
}
# SELF-CONFIRM = ยืนยันทิศด้วย candle/structure เอง (engulfing reject / SRF flip / pinbar) →
# ยกเว้น Gate 1 (zone-label) เพราะ candle/flip override label ที่อาจผิด (SRF เคย support→label demand)
# user (chart 06-03): engulfing ที่ SRF (เคย support→resistance) = SELL ถูก แต่ Gate 1 เห็น demand เลยบล็อก
# Gate 1b (sr_levels) + RR-reject ยังกัน "ชน wall" อยู่ (ไม่ใช้ demand/supply label ที่ flip ได้)
_SELF_CONFIRM = {
    "bear_engulf", "bull_engulf",
    "srf_long", "srf_short", "srf_bounce_long", "srf_bounce_short",
    "pinbar_long", "pinbar_short", "star_long", "star_short",
    "choch_long", "choch_short",
}


def _clean_entry_decision(setup, direction, price, ltf_zones, htf_zones, m5_momentum, atr_ref,
                          m15_bullish=None, range_pos=0.5, sr_levels=None, m5_ext=0.0,
                          m15_regime="chop", m5_recent_high5=None, m5_recent_low5=None,
                          m5_last2_dir=None, m1_momentum_str=None,
                          m5_imb_dir=None, m5_struct_bias=None, context_bias=None,
                          macro_range_pos=0.5,
                          pullback_in_bull=False, pullback_in_bear=False):
    """ตัดสินใจ entry ด้วย 3 gate ที่สอดคล้องกัน. คืน (ok, reason, opposing_tp).

    ltf_zones (M5/M15) = ใช้ตัดสินทิศ (Gate 1) — เทรด M1 ตีโซน M1/M5/M15
    htf_zones (H1/H4) = ใช้แค่หา TP target (Gate 3) ไม่ block ทิศ (กว้างเกิน)
    กรองโซนกว้างเกิน $8 ออกจาก Gate 1 (zone ที่ดีต้อง precise)
    """
    strat = setup.get("strategy", "")
    is_mb = strat in _MOMENTUM_BREAKOUT
    is_rev = strat in _REVERSAL_PATTERN
    # 2026-06-17 user "เลือกโซนคุณภาพ เช่น มี imb": fvg/imb = โซนที่มี imbalance จริง (คุณภาพ) → exempt mean-reversion guards
    _IMB_QUALITY = {"fvg_long", "fvg_short", "imb_continuation_long", "imb_continuation_short"}
    _imb_q = _FIX_IMB_QUALITY and strat in _IMB_QUALITY
    # 2026-06-18 user "ไม่ต้องบล็อค ให้ยิงเก็บ live data": exempt SRF/break_retest/breakout/cons_breakout
    #   เงื่อนไข: trend-aligned (context_bias หรือ NEUTRAL อนุญาต — user "เก็บข้อมูล")
    _SRF_BREAKOUT_STRATS = {"srf_long", "srf_short", "break_retest_long", "break_retest_short",
                            "breakout_long", "breakout_short",
                            "consolidation_breakout_long", "consolidation_breakout_short"}
    # NEW: ไม่เข้มงวด trend-aligned (NEUTRAL ก็ปล่อย) — user สั่ง "ไม่บล็อค"
    _srf_brk = (_FIX_SRF_BREAKOUT and strat in _SRF_BREAKOUT_STRATS and not (
        (context_bias == "BEAR" and direction == "long") or
        (context_bias == "BULL" and direction == "short")))   # บล็อกแค่สวนเทรนชัด
    tol = 0.3 * atr_ref
    # โซนสำหรับ Gate 1: M5/M15 + กรองกว้างเกิน $8 ออก (H4 $20 = หยาบเกิน)
    _g1_zones = [z for z in ltf_zones if (z["hi"] - z["lo"]) <= 8.0]

    # ── 2026-06-17 TREND-FOLLOW (user "ทำไมบอทไม่ sell ตามเทรน") ──
    # ใน confirmed trend (context_bias = M5 swing structure robust ไม่ lag เหมือน EMA regime):
    #   ขาลง(BEAR)=ขาย rally ตามเทรน / ขาขึ้น(BULL)=ซื้อ dip → bypass mean-reversion gates
    #   (1b sell-ชน-support, chop range_pos, auto-range, Gate3 room, climax) ที่มองทุกระดับว่า "ถูกเกินไป"
    # user playbook ขาลง: SRF retest / SUPPLY zone / bear engulf / shooting star / break_retest (= แนวรับเบรกเป็นแนวต้าน)
    #   ขายจนกว่าโครงสร้างเปลี่ยน (context_bias เลิกเป็น BEAR = CHOCH). mirror ขาขึ้น. เฉพาะ strategy พวกนี้ (ไม่ใช่ทุก short)
    _TREND_SELL_STRATS = {"srf_short", "supply_react_short", "supply_zone", "bear_engulf",
                          "star_short", "break_retest_short"}
    _TREND_BUY_STRATS = {"srf_long", "demand_react_long", "demand_long", "bull_engulf",
                         "star_long", "break_retest_long"}
    _trend_follow = _FIX_TREND_FOLLOW and (
        (context_bias == "BEAR" and direction == "short" and strat in _TREND_SELL_STRATS) or
        (context_bias == "BULL" and direction == "long" and strat in _TREND_BUY_STRATS))

    # 2026-06-18 user "buy ตามเทรนจนเปลี่ยน + IMB/cons/engulf = early reversal เร็วกว่า CHOCH":
    # NARROW LIST: บล็อกเฉพาะ "pullback retest patterns" ที่ fire ทุก swing ของ pullback
    # ไม่บล็อก: engulf/star/pinbar/momentum_breakout/cons_breakout/fvg/imb = confirmation/reversal signals
    _WEAK_COUNTER_SHORT = {"supply_react_short", "srf_short", "break_retest_short",
                           "dsf_short", "htf_retest_short", "trendline_short",
                           "zone_retest_short", "supply_zone"}
    _WEAK_COUNTER_LONG = {"demand_react_long", "srf_long", "break_retest_long",
                          "dsf_long", "htf_retest_long", "trendline_long",
                          "zone_retest_long", "demand_long"}

    # 2026-06-18: pullback detection (HH+HL / LH+LL pattern M5 12 bars) — รับจาก make_decision
    # = เร็วกว่า context_bias (ที่ใช้ 25 bars), จับ "rally ค่อยๆ ขึ้น" ที่ context_bias ไม่ flip ทัน

    # strong reversal exempt: pinbar/star/momentum_breakout/oversold_bounce/choch/engulf/cons/fvg/imb
    # block เฉพาะ weak (pullback retest) — ใน context_bias หรือ M5 pullback ก็ตาม
    if _BLOCK_WEAK_COUNTER:
        _bull_active = (context_bias == "BULL") or pullback_in_bull
        _bear_active = (context_bias == "BEAR") or pullback_in_bear
        if _bull_active and direction == "short" and strat in _WEAK_COUNTER_SHORT:
            _src = "ctx=BULL" if context_bias == "BULL" else "M5 HH+HL pullback"
            return False, f"trend BULL ({_src}) — block weak SHORT [{strat}]: รอ reversal", None
        if _bear_active and direction == "long" and strat in _WEAK_COUNTER_LONG:
            _src = "ctx=BEAR" if context_bias == "BEAR" else "M5 LH+LL bounce"
            return False, f"trend BEAR ({_src}) — block weak LONG [{strat}]: รอ reversal", None

    # ── AUTO-RANGE anti-inversion (user chart Jun17: บอท buy ยอด/sell ก้น = กลับหัว) ──
    # macro range (M5×_AR_BARS) → บังคับ "sell ยอด/buy ก้น". ยกเว้น momentum_breakout + trend-follow
    # + trend-aligned (ขาขึ้น buy ตามเทรนได้แม้ยอด / ขาลง sell ตามเทรนได้แม้ก้น) — climax/RR ยังคุมไม้แย่
    _trend_aligned = _AR_TREND_EXEMPT and (
        (context_bias == "BULL" and direction == "long") or
        (context_bias == "BEAR" and direction == "short"))
    if _FIX_AUTO_RANGE and not is_mb and not _trend_follow and not _trend_aligned and not _imb_q and not _srf_brk:
        if direction == "long" and macro_range_pos > _AUTO_RANGE_TOP:
            return False, f"AUTO-RANGE: buy ส่วนบน {macro_range_pos*100:.0f}% ของ range — ห้ามซื้อยอด รอ dip", None
        if direction == "short" and macro_range_pos < _AUTO_RANGE_BOT:
            return False, f"AUTO-RANGE: sell ส่วนล่าง {macro_range_pos*100:.0f}% ของ range — ห้ามขายก้น รอ rally", None

    # ── Gate 1: REMOVED 2026-06-04 (user: "พลาดหลาย move") ──
    # เดิมบล็อก "LONG ครึ่งบน supply / SHORT ครึ่งล่าง demand" → บล็อก 50% ของ signal
    # ซ้ำซ้อนกับ trend-lock: ตอนนี้ทิศคุมด้วย M15 trend แล้ว → LONG ที่ supply ในขาขึ้น
    # = เทรนทะลุ resistance = ควรเข้า (Gate 1 บล็อกผิด). ใช้ zone label ที่ flip ได้ = mislabel
    # การกัน "ชน wall" จริง = Gate 1b (sr_levels multi-touch) + Gate 3 (room) แทน
    is_self_confirm = strat in _SELF_CONFIRM  # (ยังใช้คำนวณ context อื่น)

    # ── Gate 1b: อย่าเข้า "ชน" แนวรับแนวต้านสำคัญ (user 2026-06-03: "buy ชน supply / sell ชน demand") ──
    # buy ชน resistance = ซื้อชนเพดาน เด้งลง | sell ชน support = ขายชนพื้น เด้งขึ้น
    # ใช้ find_sr_levels (multi-touch S/R = กำแพงจริง) ไม่ใช่ micro-zone (98% over-block).
    # "ชน" = ราคาห่างกำแพงตรงข้าม ≤ $1.5. ยกเว้น momentum/breakout (ทะลุกำแพงได้)
    if not is_mb and sr_levels and not _trend_follow and not _imb_q and not _srf_brk:
        _CHON = 1.5
        if direction == "long":
            for lbl, lvl in sr_levels:
                if lbl == "RESIST" and 0 <= (lvl - price) <= _CHON:
                    return False, f"buy ชน Resistance {lvl:.1f} (ซื้อชนเพดาน ห่าง ${lvl-price:.1f} เด้งลง)", None
        else:
            for lbl, lvl in sr_levels:
                if lbl == "SUPPORT" and 0 <= (price - lvl) <= _CHON:
                    return False, f"sell ชน Support {lvl:.1f} (ขายชนพื้น ห่าง ${price-lvl:.1f} เด้งขึ้น)", None

    # ── Gate 1c: ราคา WITHIN zone ตรงข้าม = ห้ามเข้า (2026-06-09 user: "บอท buy ที่ supply") ──
    # bot bought @4335.81 IN supply 4334-4337 (break_retest_long ตีความ "support" ที่ใน supply zone)
    # ใช้ _g1_zones (M5/M15 < $8 wide) — ไม่ exempt REVERSAL_AT_WALL (zone-correct ต้องสำคัญสุด)
    #
    # FIX #2B (2026-06-10): DSF ZONE FLIP — exempt directional strategies จาก "in zone" check
    # ปัญหา: demand ~4173-4179 ถูก break ไปแล้ว (DSF) แต่ bot ยังบล็อก SHORT ทุกตัว
    #   TV 16:33 zone_first_touch_short @4174 VETOED "sell in DEMAND" → พลาดขาลง $18
    #   TV 14:44-14:58 srf/momentum/imb/zone_retest ถูกบล็อก 16 ครั้ง (44% ของ veto ทั้งหมด)
    # แก้: ถ้า regime ไม่ใช่ trend ตรงข้าม → exempt strategies ที่มี directional confirmation ในตัว
    #   SHORT ใน chop/trend_bear: dsf/imb_continuation/srf/zone_retest/choch exempt "in DEMAND"
    #   LONG ใน chop/trend_bull: dsf/imb_continuation/srf/zone_retest/choch exempt "in SUPPLY"
    # เหตุผล: demand ที่ถูก break = DSF = sell zone จริง, strategies เหล่านี้มี confirmation แล้ว
    # ยังบล็อก: trend_bull + SHORT ใน demand (ซื้อ dip ใน trend ขึ้น = demand ยังแข็ง)
    #           break_retest/zone_first_touch ที่ไม่มี strong confirmation
    _DSF_EXEMPT_STRATS = {
        "dsf_short", "dsf_long",
        "imb_continuation_short", "imb_continuation_long",
        "srf_short", "srf_long", "srf_bounce_short", "srf_bounce_long",
        "zone_retest_short", "zone_retest_long",
        "choch_short", "choch_long",
        # 2026-06-12: ถอด supply_react_short/demand_react_long ออก (light detector buy ที่ supply/sell ที่ demand)
        #   เคส chart Jun12: demand_react_long @4226 ใน SUPPLY (trend_bull DSF-exempt) = buy ยอด → ราคาลง
        #   DSF จริงราคาจะอยู่ "เหนือ" supply ที่ break แล้ว (ไม่อยู่ใน zone) → Gate 1c ไม่ block อยู่แล้ว
        # FIX #7B (2026-06-10): fvg exempt DSF — rally ทะลุ supply เก่า (DSF) แต่ fvg_long ยังโดน block
        # เคส TV 20:36: fvg_long @4152 VETOED "buy in SUPPLY 4148-4156" ขณะ rally $50
        # supply นั้นถูก break ไปแล้ว = DSF = fvg continuation เข้าได้
        "fvg_long", "fvg_short",
    }
    _dsf_exempt = strat in _DSF_EXEMPT_STRATS
    # 2026-06-12: momentum_breakout เคารพ Gate 1c ด้วย (อย่าขายเข้า demand / buy เข้า supply — รอทะลุพ้นโซน)
    if not is_mb or strat in ("momentum_breakout_long", "momentum_breakout_short"):
        # ── Fix A (2026-06-11): CONTESTED RANGE — ราคาอยู่ใน DEMAND+SUPPLY ซ้อนกัน ──
        # ใน sideway มี swing high (สร้าง SUPPLY) + swing low (สร้าง DEMAND) ที่ราคาใกล้กัน
        # → zone classification ก้ำกึ่ง → Gate เจอ zone ฝั่งตรงข้ามก่อน เลย block ทั้งที่ detector ยิงถูกฝั่ง
        # หลักฐาน: Chart 2 (11:00-11:08) break_retest_long @4072-4076 ที่ SRF/demand จริง
        #   โดน block "buy ในโซน SUPPLY 4071-4076" (มี supply zone ซ้อน) → พลาด zone → เข้าสูงไปแพ้
        #   12:09 เข้า @4078 ใกล้ zone = ชนะ +$104 (พิสูจน์ว่าเข้าที่ zone = ถูก)
        # แก้: ถ้าราคาอยู่ใน "ทั้ง 2 ฝั่ง" (contested) → เชื่อ detector ที่ยิง ไม่ block
        #   ยังเคารพเทรน: long block เฉพาะ trend_bear, short block เฉพาะ trend_bull (ไม่ buy supply ขาลง)
        _has_demand_here = any(z.get("type") == "DEMAND" and z["lo"] <= price <= z["hi"] for z in _g1_zones)
        _has_supply_here = any(z.get("type") == "SUPPLY" and z["lo"] <= price <= z["hi"] for z in _g1_zones)
        _contested = _has_demand_here and _has_supply_here
        # 2026-06-12: momentum_breakout ไม่ใช้ contested exemption (เคส short ขายเข้า demand @4204 = sell ก้น)
        #   ต้องทะลุโซนจริง (ราคาพ้น zone) ก่อน ไม่ใช่เข้าไปในโซนฝั่งตรงข้าม
        if strat in ("momentum_breakout_long", "momentum_breakout_short"):
            _contested = False
        for z in _g1_zones:
            if direction == "long" and z.get("type") == "SUPPLY":
                if z["lo"] <= price <= z["hi"]:
                    # FIX #2B: exempt ถ้า trend ไม่สวน (chop/trend_bull + LONG = OK)
                    if _dsf_exempt and m15_regime != "trend_bear":
                        pass  # supply อาจถูก break แล้ว (DSF) → allow LONG ที่มี confirmation
                    elif (_FIX_ZONE_FLIP and m5_recent_high5 is not None
                          and m5_recent_high5 > z["hi"] + 0.3 * atr_ref):
                        pass  # ZONE-FLIP: supply ทะลุขึ้น = flip เป็น support → breakout retest = buy
                    elif _contested and m15_regime == "chop":
                        pass  # Fix A: contested = range trading เท่านั้น (chop). ในเทรนเคารพโซน (2026-06-12)
                    else:
                        return False, f"buy ในโซน SUPPLY {z['lo']:.0f}-{z['hi']:.0f} (ราคาใน zone)", None
            elif direction == "short" and z.get("type") == "DEMAND":
                if z["lo"] <= price <= z["hi"]:
                    # FIX #2B: exempt ถ้า trend ไม่สวน (chop/trend_bear + SHORT = OK)
                    if _dsf_exempt and m15_regime != "trend_bull":
                        pass  # demand อาจถูก break แล้ว (DSF) → allow SHORT ที่มี confirmation
                    elif (_FIX_ZONE_FLIP and m5_recent_low5 is not None
                          and m5_recent_low5 < z["lo"] - 0.3 * atr_ref):
                        pass  # ZONE-FLIP: demand ทะลุลง = flip เป็น supply → breakdown retest = sell
                    elif _contested and m15_regime == "chop":
                        pass  # Fix A: contested = range trading เท่านั้น (chop). ในเทรนเคารพโซน (2026-06-12)
                    else:
                        return False, f"sell ในโซน DEMAND {z['lo']:.0f}-{z['hi']:.0f} (ราคาใน zone)", None

    # ── REVERSAL_AT_WALL set: ใช้ใน Gate 2 + 2.7 + 2.8 + 2.9 + 2.10 + 2.11 + Gate 3 ──
    # 2026-06-10 FIX: ถอด break_retest ออก — เป็น structural retest ไม่ใช่ wall rejection
    # break_retest_long ใน trend_bear = buy ที่ demand ซึ่งกำลังจะ break (จับมีดตก -$150.65)
    # break_retest_short ใน trend_bull = sell ที่ supply ซึ่งกำลังจะ break (mirror)
    # ต้องเคารพ trend alignment, fresh peak/dip, M5 bounce, IMB/structure, context bias
    _REVERSAL_AT_WALL = {"pinbar_long", "pinbar_short", "star_long", "star_short",
                         "bull_engulf", "bear_engulf",
                         "srf_long", "srf_short", "srf_bounce_long", "srf_bounce_short",
                         "trendline_long", "trendline_short",
                         "liq_sweep_long", "liq_sweep_short",
                         "supply_react_short", "demand_react_long"}

    # ── Fix C (2026-06-11): ZONE-DETECTOR set — detector ที่ระบุ S/R/zone level เฉพาะของตัวเอง ──
    # range_pos (ก้น/ยอด range) เป็น proxy หยาบ — ไม่รู้จัก "แนวรับ/ต้านกลาง range"
    # เคส chart Jun11 22:00: break_retest_long @4073 ที่ demand support (เด้ง 3 ครั้ง) โดน block
    #   "LONG ต้องที่ก้น range (51%)" เพราะ range 18 บาร์รวมก้นดิ่ง 4054 → 4073 อ่านเป็นกลาง
    # zone-detector มี level ของตัวเองแล้ว → ยกเว้น range_pos floor (chop NEUTRAL) แต่ยังตรวจ knife/climax
    _ZONE_DETECTOR = {"break_retest_long", "break_retest_short",
                      "zone_retest_long", "zone_retest_short",
                      "zone_first_touch_long", "zone_first_touch_short",
                      "sr_long", "sr_short", "demand_long", "supply_zone",
                      "dsf_long", "dsf_short", "fvg_long", "fvg_short"}

    # ── Gate 2: REGIME-AWARE (regime จาก slope+EMA, แก้ user 2026-06-04: EMA lag เห็นขาลงเป็น chop) ──
    # trend_bear/bull: trend-follow (sell rally / buy dip) | chop: mean-reversion (ขายบนซื้อล่าง)
    if m15_regime == "chop":
        # 2026-06-09 v2: REVERSAL_AT_WALL exempt chop check (wall reject = trade ที่ extreme ของ chop)
        if strat in _REVERSAL_AT_WALL:
            pass   # wall rejection ผ่านได้ทั้ง direction ใน chop
        elif m5_momentum == "DOWN":
            if direction == "long":
                return False, "chop: M5 momentum DOWN ห้าม LONG (catching falling knife)", None
        elif m5_momentum == "UP":
            if direction == "short":
                return False, "chop: M5 momentum UP ห้าม SHORT (chase rally)", None
        elif strat in ("momentum_breakout_long", "momentum_breakout_short"):
            pass   # vertical breakout = new high/low (range_pos สุดขั้วปกติ) → ยกเว้น range_pos (spike-guard คุมแทน)
        # Fix C REVERTED (2026-06-12): zone-detector กลับมาเคารพ range_pos — กัน sell ก้น/buy ยอด (วินัยเก่า)
        else:
            # M5 NEUTRAL → range_pos rule (ขายบน ซื้อล่าง) — ยกเว้น trend-follow (ขาย rally ขาลง/ซื้อ dip ขาขึ้น)
            if direction == "short" and range_pos < 0.60 and not _trend_follow:
                return False, f"chop: SHORT ต้องที่ยอด range ({range_pos*100:.0f}%)", None
            if direction == "long" and range_pos > 0.40 and not _trend_follow:
                return False, f"chop: LONG ต้องที่ก้น range ({range_pos*100:.0f}%)", None
    elif m15_regime == "trend_bear":
        # ขาลง → SELL rally. 2026-06-09: REVERSAL_AT_WALL ผ่านได้ทันที (candle/structure = proof)
        if direction == "long":
            _rev_ok = False
            if strat == "oversold_bounce_long" and m5_ext > 1.8:
                _rev_ok = True
            elif strat in _REVERSAL_AT_WALL:
                _rev_ok = True   # wall rejection มี proof แล้ว ไม่ต้อง m5_ext
            elif strat == "momentum_breakout_long":
                _rev_ok = True   # 2026-06-12: vertical breakout UP = momentum reversal จริง → ข้าม regime lag
            elif strat in {"zone_retest_long", "hl_retest_long"} and m5_ext > 1.5:
                _rev_ok = True
            # 2026-06-18 user (rally $52 4218→4270 บอทไม่เข้า): exempt breakout/cons_break/srf/fvg จาก regime lag
            #   M15 regime lag 30-60 นาที หลัง dump → block recovery rally. detector พวกนี้ confirm structure ใหม่แล้ว
            elif strat in {"consolidation_breakout_long", "srf_long", "break_retest_long", "fvg_long"}:
                _rev_ok = True
            if not _rev_ok:
                return False, f"เทรนขาลง — LONG ไม่ได้ (ยกเว้น REVERSAL_AT_WALL หรือ zone_retest m5_ext>1.5: ตอนนี้ {m5_ext:+.1f})", None
    elif m15_regime == "trend_bull":
        # ขาขึ้น → BUY dip. REVERSAL_AT_WALL ผ่านได้ทันที (mirror)
        if direction == "short":
            _rev_ok = False
            if strat == "oversold_bounce_short" and m5_ext < -1.8:
                _rev_ok = True
            elif strat in _REVERSAL_AT_WALL:
                _rev_ok = True
            elif strat == "momentum_breakout_short":
                _rev_ok = True   # 2026-06-12: vertical breakdown DOWN = momentum reversal จริง → ข้าม regime lag (เคส Jun12 momentum sell ที่ยอด spike)
            elif strat in {"zone_retest_short", "hl_retest_short"} and m5_ext < -1.5:
                _rev_ok = True
            # 2026-06-18 mirror: exempt breakdown patterns จาก regime_bull lag
            elif strat in {"consolidation_breakout_short", "srf_short", "break_retest_short", "fvg_short"}:
                _rev_ok = True
            if not _rev_ok:
                return False, f"เทรนขาขึ้น — SHORT ไม่ได้ (ยกเว้น REVERSAL_AT_WALL หรือ zone_retest m5_ext<-1.5: ตอนนี้ {m5_ext:+.1f})", None

    # ── Gate 2.6: OVER-EXTENSION — กัน fade climax (counter-trend) + ผ่อน trend-follow ──
    # 2026-06-05 v1: exempt trend-follow ทั้งหมด → bot SHORT ที่ bounce 21:51 SL -$48 (4361 hit)
    # 2026-06-05 v2: trend-follow ผ่อน 2.5→3.5 threshold (allow trend continuation แต่กัน climax extreme)
    if not is_mb:
        _is_trend_follow = _trend_follow or (
            (direction == "short" and m15_regime == "trend_bear") or
            (direction == "long" and m15_regime == "trend_bull")
        )
        # trend-follow: 3.5 (ยืดธรรมดาในเทรนต่อเนื่อง OK) | counter-trend: 2.5 (กัน fade)
        thresh = 3.5 if _is_trend_follow else 2.5
        if direction == "short" and m5_ext > thresh:
            return False, f"short ยืดลง {m5_ext:.1f} ATR ใต้ EMA (climax/oversold) ยอมที่ > {thresh}", None
        if direction == "long" and m5_ext < -thresh:
            return False, f"buy ยืดขึ้น {abs(m5_ext):.1f} ATR เหนือ EMA (climax/overbought) ยอมที่ > {thresh}", None

    # ── Gate 2.6b (2026-06-12): ABSOLUTE SPIKE GUARD — กัน buy ยอด spike (user: "buy ข้างบน") ──
    # ปัญหา: Gate 2.6 ใช้ ATR-normalized → spike ทำ ATR พอง → guard พลาด
    #   chart Jun12 00:42: srf_long @4164 หลัง rally $84 (4080→4164 ใน 15min) ATR พุ่ง $19 → ext แค่ 2.8 ATR → ผ่าน → SL -$30
    # แก้: ABSOLUTE distance จากก้น M5 (recent_low5 ~25min) — ไม่ขึ้นกับ ATR
    #   rally > $40 จากก้น = spike → block LONG ทุก strategy (ไล่ซื้อยอด)
    #   เข้าได้ใหม่เมื่อ pullback / เวลาผ่าน (recent_low5 ขยับขึ้น) = ซื้อจุดดีกว่า ไม่ใช่ยอด
    # user 2026-06-12: "แก้เฉพาะตอน buy ข้างบน" → ทำเฉพาะ LONG (SHORT mirror ไม่ทำ ตามที่ขอ)
    _SPIKE_ABS = 40.0
    if direction == "long" and m5_recent_low5 is not None:
        _rally_abs = price - float(m5_recent_low5)
        if _rally_abs > _SPIKE_ABS:
            return False, f"buy หลัง spike ${_rally_abs:.0f} จากก้น {float(m5_recent_low5):.0f} (>${_SPIKE_ABS:.0f} abs) — ไล่ยอด spike", None
    # 2026-06-12: SHORT mirror — กัน sell ก้น spike crash (user: "ทำ short ทุกรอบ") → คุม momentum_breakout_short ไม่ไล่ก้น
    if direction == "short" and m5_recent_high5 is not None:
        _drop_abs = float(m5_recent_high5) - price
        if _drop_abs > _SPIKE_ABS:
            return False, f"sell หลัง spike ${_drop_abs:.0f} จากยอด {float(m5_recent_high5):.0f} (>${_SPIKE_ABS:.0f} abs) — ไล่ก้น spike", None

    # ── Gate 2.7: FRESH PEAK/DIP guard — อย่า buy ที่ยอดเพิ่งปั่น / sell ที่ก้นเพิ่งดิ่ง ──
    # user 2026-06-05 (chart): bot buy @4481.57 ขณะ M5 peak เพิ่งแตะ 4483.34 (ห่างแค่ $1.77)
    # = supply กำลังก่อแต่ยังไม่ถูก label → Gate 1b/Gate 3 ไม่ทัน → bot ซื้อยอด rally แล้ว SL
    # user 2026-06-05 (chart 2): bot buy @4481.74 [choch_long] ห่าง peak 4482.39 = $0.65 ก็ SL อีก
    # → ตอนนี้ exempt เฉพาะ "reversal candle ที่ wall" (pinbar/star/engulf) เท่านั้น
    # choch/srf/zone_retest = continuation/structure → ต้องเคารพ wall ใหม่ที่กำลังก่อ
    # แก้: ใช้ราคา action (M5 high/low 5 แท่ง = 25 นาที) เป็น wall ที่ "กำลังก่อ"
    # REVERSAL_AT_WALL = strats ที่ "ปฏิเสธที่กำแพง" — exempt Gate 2.7 (กำแพง คือ entry)
    # 2026-06-05 user: "sell ที่ SRF retest" — SRF flip level = wall จริง → exempt climax block
    # (REVERSAL_AT_WALL defined above ก่อน Gate 2)
    # 2026-06-05 user: ขาลง $113 บล็อก SHORT 38/51 — Stage 2 climax ก็ exempt trend-follow ด้วย
    _is_trend_follow_27 = (
        (direction == "short" and m15_regime == "trend_bear") or
        (direction == "long" and m15_regime == "trend_bull")
    )
    if not is_mb and strat not in _REVERSAL_AT_WALL and not _imb_q and not _srf_brk:
        _FRESH_PROX = 2.0
        _CLIMAX_ATR = 3.0       # ATR M5 > $3 = climax volatility (sharp recent move)
        _CLIMAX_MULT = 1.5      # in climax, proximity threshold scales with ATR
        if direction == "long" and m5_recent_high5 is not None:
            dh = float(m5_recent_high5) - price
            # Stage 1 (normal): block if within $2 of fresh peak
            if 0 <= dh <= _FRESH_PROX:
                return False, f"buy ใกล้ M5 peak สด {m5_recent_high5:.1f} (ห่าง ${dh:.1f} ≤ $2 — supply กำลังก่อ)", None
            # Stage 2 (climax): exempt trend-follow (LONG ใน trend_bull = rally continuation)
            # 2026-06-18: exempt oversold_bounce_long ด้วย — detector require m5_ext>1.8+reversal candle = V-bounce pattern, peak ก่อน crash ไม่ valid
            _v_bounce_exempt = _FIX_VBOUNCE and strat == "oversold_bounce_long"
            if not _is_trend_follow_27 and not _v_bounce_exempt and atr_ref > _CLIMAX_ATR and 0 <= dh <= _CLIMAX_MULT * atr_ref:
                return False, f"buy หลัง climax rally ATR=${atr_ref:.1f} (ห่าง peak ${dh:.1f} ≤ {_CLIMAX_MULT:.1f}×ATR — เด้งกลับเสี่ยง)", None
        if direction == "short" and m5_recent_low5 is not None:
            dl = price - float(m5_recent_low5)
            if 0 <= dl <= _FRESH_PROX:
                return False, f"sell ใกล้ M5 dip สด {m5_recent_low5:.1f} (ห่าง ${dl:.1f} ≤ $2 — demand กำลังก่อ)", None
            # Stage 2 (climax): exempt trend-follow (SHORT ใน trend_bear = drop continuation)
            # 2026-06-18 mirror: exempt oversold_bounce_short (V-rejection หลัง rally)
            _v_bounce_exempt = _FIX_VBOUNCE and strat == "oversold_bounce_short"
            if not _is_trend_follow_27 and not _v_bounce_exempt and atr_ref > _CLIMAX_ATR and 0 <= dl <= _CLIMAX_MULT * atr_ref:
                return False, f"sell หลัง climax drop ATR=${atr_ref:.1f} (ห่าง dip ${dl:.1f} ≤ {_CLIMAX_MULT:.1f}×ATR — เด้งกลับเสี่ยง)", None

    # ── 2026-06-17 (user chart: กรอบแดง bot buy ที่ swing high) — weak-long ห้ามซื้อส่วนบน range ──
    # buy ที่ "ยอด local" (>65% ของ M5 5-bar range) = ซื้อยอด rally เล็กๆ ที่กำลังจะกลับ. รอ dip
    # เฉพาะ weak-long (buy zone/dip) — momentum_breakout/reversal-candle/CHOCH ไม่กระทบ
    # 2026-06-17: srf_long = breakout-retest → buy สูงได้หลังทะลุ S/R (user สอน) แต่เฉพาะบริบทเทรนไม่ลง
    #   context_bias (M5 swing 2h: LH+LL=BEAR) robust กว่า regime/m5_momentum (ไม่โดนเด้งสั้นในขาลงหลอก)
    #   → BEAR (เด้งขาลง) ยังบล็อก srf_long กัน buy breakout ปลอม | NEUTRAL/BULL (range/up) ปล่อย
    _WEAK_LONG_NO_TOP = {"demand_react_long", "srf_long", "sr_long", "fvg_long",
                         "zone_first_touch_long", "trendline_long", "hl_retest_long", "zone_retest_long"}
    if _FIX_BREAKOUT_EXEMPT and context_bias != "BEAR":
        _WEAK_LONG_NO_TOP = _WEAK_LONG_NO_TOP - {"srf_long"}
    if (_FIX_NO_BUY_TOP and direction == "long" and strat in _WEAK_LONG_NO_TOP
            and m5_recent_high5 is not None and m5_recent_low5 is not None):
        _rng_nt = float(m5_recent_high5) - float(m5_recent_low5)
        if _rng_nt > 0 and (price - float(m5_recent_low5)) / _rng_nt > 0.65:
            return False, f"buy ส่วนบน range ({(price-float(m5_recent_low5))/_rng_nt*100:.0f}% ของ M5 range) — ซื้อยอด รอ dip", None

    # ── Gate 2.9: M1 MOMENTUM CHECK — STRICT ── (2026-06-08 user: "บอท sell ขณะ momentum UP ไม่หยุด")
    # ก่อนหน้า exempt REVERSAL_AT_WALL ทั้งหมด → hl_retest spam ขายตอน M1 UP = -$503
    # ใหม่: เฉพาะ candle pattern จริง (pinbar/star/engulf) ที่ candle = visible reject เท่านั้น exempt
    # hl_retest/srf/trendline = pullback pattern ไม่ใช่ visible reject → ต้องเคารพ M1 momentum
    _STRICT_REVERSAL = {"pinbar_long", "pinbar_short", "star_long", "star_short",
                        "bull_engulf", "bear_engulf"}
    # 2026-06-12: _STRICT_REVERSAL exempt M1-momentum เฉพาะ CHOP — ใน TREND (counter-trend) ต้องเคารพ M1
    #   เคส Jun12 01:16-03:05: pinbar_short/bear_engulf สวน uptrend ตอน M1 UP (ราคากำลังขึ้น) → -$106
    #   winners (09:22/14:31 pinbar_short) sell ตอน M1 DOWN (ราคาดิ่ง) → ไม่กระทบ
    #   ตัวแยกแพ้/ชนะ = M1 momentum ตอนเข้า ไม่ใช่ regime (winners ก็ trend_bull เหมือนกัน)
    _m1mom_exempt = (strat in _STRICT_REVERSAL) and (m15_regime == "chop")
    if not is_mb and not _m1mom_exempt and m1_momentum_str is not None:
        # m1_momentum_str: "UP_STRONG" | "DOWN_STRONG" | "NEUTRAL"
        if direction == "short" and m1_momentum_str == "UP_STRONG":
            return False, "M1 momentum UP แรง (4+ green/higher closes) — รอ M1 reject ก่อน SHORT", None
        if direction == "long" and m1_momentum_str == "DOWN_STRONG":
            return False, "M1 momentum DOWN แรง (4+ red/lower closes) — รอ M1 bounce ก่อน LONG", None

    # ── Gate 2.8: M5 STRONG BOUNCE = bot จะ SHORT ตอน trend แทบกลับตัว (2026-06-05 21:51 SL -$48) ──
    # ในขาลง 21:50 close +6.66, 21:55 close +6.68 = 2 candle เขียว body > 1×ATR ติดกัน → bounce แรง
    # bot SHORT ที่ 4351 → bounce ต่อไป 4371 = -$48. แก้: ใน trend_bear ห้าม SHORT ถ้า 2 M5 ติดกันเขียวแรง
    # (mirror: trend_bull ห้าม LONG ถ้า 2 M5 ติดกันแดงแรง = top forming)
    if not is_mb and strat not in _REVERSAL_AT_WALL and m5_last2_dir is not None:
        # m5_last2_dir = (bar1_body_atr, bar2_body_atr) sign+magnitude ของ 2 M5 bar ล่าสุด
        b1, b2 = m5_last2_dir
        if direction == "short" and m15_regime == "trend_bear":
            if b1 > 0.5 and b2 > 0.5:   # 2 bars เขียวแรง > 0.5 ATR each
                return False, f"M5 bounce แรง (2 candle เขียว body {b1:.1f}+{b2:.1f} ATR) ใน trend_bear → trend อาจกลับ รอ rally exhaust", None
        if direction == "long" and m15_regime == "trend_bull":
            if b1 < -0.5 and b2 < -0.5:  # 2 bars แดงแรง
                return False, f"M5 pullback แรง (2 candle แดง body {abs(b1):.1f}+{abs(b2):.1f} ATR) ใน trend_bull → trend อาจกลับ รอ pullback exhaust", None

    # ── Gate 2.10: IMB + STRUCTURE CHECK (2026-06-09 user: "สังเกตุแท่ง IMB + โครงสร้างราคา") ──
    # m5_imb_dir: "UP" = IMB impulse ขึ้น (large bullish body), "DOWN" = IMB ลง, None
    # m5_struct_bias: "BULL" (M5 EMA8>EMA21) / "BEAR" / "FLAT"
    # ห้าม SHORT ตอน IMB up หรือ M5 bull structure (สวน momentum/structure)
    # ห้าม LONG ตอน IMB down หรือ M5 bear structure (catching knife)
    # ยกเว้น REVERSAL_AT_WALL (pinbar/star/engulf/srf at extreme zone — top/bottom catcher)
    if not is_mb and strat not in _REVERSAL_AT_WALL:
        if direction == "short":
            if m5_imb_dir == "UP":
                return False, "M5 IMB impulse UP — ห้าม SHORT สวน impulse", None
            if m5_struct_bias == "BULL" and m5_ext < 1.0:
                # M5 EMA bull structure + ราคาไม่ได้ overbought → counter-trend = block
                return False, "M5 structure BULL (EMA8>EMA21) — รอ rally ขึ้นก่อน SHORT", None
        elif direction == "long":
            if m5_imb_dir == "DOWN":
                return False, "M5 IMB impulse DOWN — ห้าม LONG สวน impulse (catching knife)", None
            if m5_struct_bias == "BEAR" and m5_ext > -1.0:
                return False, "M5 structure BEAR (EMA8<EMA21) — รอ dip ลงก่อน LONG", None

    # ── Gate 2.11: CONTEXT BIAS (2026-06-09 user: "bias ขาลง bot ต้องเน้น sell") ──
    # context_bias: BULL (HH+HL ใน lookback) / BEAR (LH+LL) / NEUTRAL
    # block counter-bias trades unless REVERSAL_AT_WALL (top/bottom catchers)
    if not is_mb and strat not in _REVERSAL_AT_WALL:
        if context_bias == "BEAR" and direction == "long":
            return False, "context bias BEAR (LH+LL) — เน้น SELL, ห้าม LONG counter-bias", None
        if context_bias == "BULL" and direction == "short":
            return False, "context bias BULL (HH+HL) — เน้น BUY, ห้าม SHORT counter-bias", None

    # ── Gate 2.11b (Fix B', 2026-06-11): WEAK-REVERSAL impulse guard ──
    # demand_react/supply_react/liq_sweep = detector "light" (zone-touch บาร์เดียว) อยู่ใน REVERSAL_AT_WALL
    # → bypass knife gates ทั้งหมด (2.8/2.10/2.11/2.12). แต่ buy demand บาร์เดียวขณะ M5 impulse ลงแรง = จับมีดตก
    # หลักฐาน Chart1 (Thai 15:06-15:46): demand_react_long @4106/4103 + liq_sweep @4091 buy สวน momentum sell
    #   = -$62 + เต็มช่อง MAX_POS → short ที่ PASS (4107-4094) เปิดไม่ได้ → บอท "ไม่ได้ sell" ทั้งที่ signal ผ่าน
    # แก้: subject weak-reversal เข้า IMB impulse check (M5 แท่งล่าสุด body>1.5× = impulse สวนทิศ)
    #   ยกเว้นถ้า oversold/overbought extreme (m5_ext ≥ 1.8) = ก้น/ยอดจริง (oversold_bounce territory) → ผ่าน
    #   demand_react ยิงบน M1 (แท่ง M1 เขียว) แต่แท่ง M5 ยังแดง → m5_imb_dir=DOWN จับได้
    # pinbar/star/engulf (candle confirmation ชัด) ไม่กระทบ — ยัง exempt เต็ม (ไม่อยู่ใน _WEAK_REVERSAL)
    _WEAK_REVERSAL = {"demand_react_long", "supply_react_short",
                      "liq_sweep_long", "liq_sweep_short"}
    if not is_mb and strat in _WEAK_REVERSAL:
        if direction == "long" and m5_imb_dir == "DOWN" and m5_ext < 1.8:
            return False, f"Fix B': weak-reversal LONG ขณะ M5 IMB impulse DOWN (ext {m5_ext:+.1f}<1.8 ยังไม่ oversold) — จับมีดตก รอ pinbar/star", None
        if direction == "short" and m5_imb_dir == "UP" and m5_ext > -1.8:
            return False, f"Fix B': weak-reversal SHORT ขณะ M5 IMB impulse UP (ext {m5_ext:+.1f}>-1.8 ยังไม่ overbought) — chase rally รอ pinbar/star", None

    # ── Gate 2.12: RANGE-POS FLOOR — อย่า short ก้น range / long ยอด range ──
    # 2026-06-10 log analysis: 07:34 SHORT@4315 range_pos=19% ใน trend_bear consolidation
    # = sell ก้น local range 3 ครั้ง → SL -$65. ก้นไม่มีที่ลง ควรรอ rally ขึ้นยอดก่อน
    # mirror: trend_bull LONG ที่ range_pos > 80% = ซื้อยอด ไม่มีที่ขึ้น รอ dip
    # exempt: momentum/breakout (ทะลุ range ได้) + REVERSAL_AT_WALL (reject ที่ extreme = valid)
    if not is_mb and strat not in _REVERSAL_AT_WALL:
        if direction == "short" and m15_regime == "trend_bear" and range_pos < 0.20:
            return False, f"trend_bear แต่ range_pos {range_pos*100:.0f}% = ก้น range — อย่า short ก้น รอ rally", None
        if direction == "long" and m15_regime == "trend_bull" and range_pos > 0.80:
            return False, f"trend_bull แต่ range_pos {range_pos*100:.0f}% = ยอด range — อย่า long ยอด รอ dip", None
    # FIX #7A (2026-06-10): EXTREME FLOOR — แม้ momentum/breakout ก็ห้ามที่ก้นสุด/ยอดสุด
    # เคส TV 20:20: inside_bar_break_short @4134 range_pos=2% ที่ demand bottom
    # → ผ่าน (exempt Gate 2.12 เพราะ is_mb) → SL เมื่อราคา rally กลับ $50
    # แก้: range_pos < 10% SHORT / > 90% LONG → block แม้ momentum/breakout
    # ที่ extreme สุดไม่มี room ลงต่อ/ขึ้นต่อ ยกเว้น REVERSAL_AT_WALL (reject ที่ wall)
    # 2026-06-12: ยกเว้น momentum_breakout_long ด้วย — breakout ทำ new high (range_pos ~100%) เป็นปกติ
    #   spike-guard ($40 abs) คุมไม่ให้ไล่ยอดไกลเกินแทน
    if is_mb and not _trend_follow and strat not in _REVERSAL_AT_WALL and strat not in ("momentum_breakout_long", "momentum_breakout_short") and range_pos is not None:
        if direction == "short" and range_pos < 0.10:
            return False, f"momentum/breakout SHORT ที่ก้นสุด range_pos {range_pos*100:.0f}% — ไม่มี room ลงต่อ", None
        if direction == "long" and range_pos > 0.90:
            return False, f"momentum/breakout LONG ที่ยอดสุด range_pos {range_pos*100:.0f}% — ไม่มี room ขึ้นต่อ", None

    # ── Gate 2.13: HTF_RETEST ห้าม buy ที่ยอด rally / sell ที่ก้น drop ──
    # 2026-06-11 chart: htf_retest_long @4120 หลัง rally $19 จากก้น 4101
    # = buy ที่ยอด bounce ไม่ใช่ support จริง. M15 zone เก่ากว้าง $9 ทำให้ fire ผิด
    # แก้: ถ้า rally > 1.5×ATR จาก recent M5 low → block htf_retest_long (mirror short)
    if strat.startswith("htf_retest"):
        if direction == "long" and m5_recent_low5 is not None:
            rally = price - float(m5_recent_low5)
            if rally > 1.5 * atr_ref:
                return False, f"htf_retest LONG หลัง rally ${rally:.1f} (>{1.5*atr_ref:.1f}) จากก้น {m5_recent_low5:.1f} — buy ยอด bounce", None
        if direction == "short" and m5_recent_high5 is not None:
            drop = float(m5_recent_high5) - price
            if drop > 1.5 * atr_ref:
                return False, f"htf_retest SHORT หลัง drop ${drop:.1f} (>{1.5*atr_ref:.1f}) จากยอด {m5_recent_high5:.1f} — sell ก้น drop", None

    # ── Gate 3: มี room ถึง opposing zone (≥ $2) + คืน opposing เป็น TP target ──
    # REVERSAL_AT_WALL (SRF/pinbar/star/engulf) = threshold $1 (wall reject ไม่ต้อง $2 room)
    # 2026-06-05 user: SRF retest ที่ H1 demand $1.6 away ก็ขายได้ ราคาทะลุได้
    _all_z = list(ltf_zones) + list(htf_zones)
    opp = []
    for z in _all_z:
        if direction == "long" and z.get("type") == "SUPPLY" and z["lo"] > price + 0.1:
            opp.append(z["lo"])
        elif direction == "short" and z.get("type") == "DEMAND" and z["hi"] < price - 0.1:
            opp.append(z["hi"])
    opposing_tp = None
    _room_min = 1.0 if strat in _REVERSAL_AT_WALL else 2.0
    if opp:
        nearest = min(opp) if direction == "long" else max(opp)
        dist = abs(nearest - price)
        if dist < _room_min:
            # 2026-06-12: momentum_breakout ทะลุ wall ได้ (vertical breakout) → ไม่ block, ใช้ RR-based TP
            #   เคส 00:30 srf_long @4107 โดน block "opposing zone 4108 $0.6 no room" แต่ราคาทะลุไป 4164
            if strat not in ("momentum_breakout_long", "momentum_breakout_short") and not _trend_follow:
                return False, f"opposing zone {nearest:.0f} ห่าง ${dist:.1f} < ${_room_min:.0f} (ไม่มี room)", None
        else:
            opposing_tp = nearest

    return True, f"ทิศตรงโซน + momentum OK + room ✓", opposing_tp


def make_decision(bridge, classifier):
    # M5 = primary (3x faster detection than M15, matches user's M1/M5 analysis style)
    df_m5 = bridge.fetch_bars(timeframe="M5", n_bars=500)   # ~41h of data
    df_m5 = df_m5.reset_index(drop=True)
    # M1 for fine CHOCH / engulfing structure
    df_m1 = bridge.fetch_bars(timeframe="M1", n_bars=300)   # last 5h on M1
    df_m1 = df_m1.reset_index(drop=True)
    # M15 kept for context / S/R multi-touch (needs longer history)
    df = bridge.fetch_bars(timeframe="M15", n_bars=250)
    df = df.reset_index(drop=True)

    tick = bridge.current_tick()
    atr    = compute_atr(df, ATR_WINDOW)        # M15 ATR for S/R zones
    atr_m5 = compute_atr(df_m5, ATR_WINDOW)    # M5 ATR for patterns
    atr_m1 = compute_atr(df_m1, ATR_WINDOW)    # M1 ATR for fine patterns
    mt_long, mt_short = compute_mt_counts(df, atr)  # S/R still on M15

    # ── Weekend / Off-Market Session Filter ─────────────────────────────
    # ตลาด XAUUSD ปิด: Friday ~22:00 UTC → Sunday ~22:00 UTC
    # ช่วงนี้ data = noise (spread กว้าง, liquidity ต่ำ, M1 zone = fake)
    # ห้ามเปิดออเดอร์ใหม่ แต่ยัง log decision ปกติ
    _now_utc = dt.datetime.utcnow()
    _wd = _now_utc.weekday()  # 0=Mon, 4=Fri, 5=Sat, 6=Sun
    _is_weekend = (
        (_wd == 4 and _now_utc.hour >= 22) or  # Friday >= 22:00
        _wd == 5 or                              # Saturday
        (_wd == 6 and _now_utc.hour < 22)        # Sunday < 22:00
    )

    # ── Pre-compute zones ทุก TF ─────────────────────────────────────────
    # M5, M15 zones สำหรับ zone_retest/first_touch/pinbar/star แต่ละ TF
    _pre_m1_zones = []
    _pre_m5_zones = []
    _pre_m15_zones = []
    try:
        _pre_m1_zones, _ = find_demand_supply_zones(bridge.symbol, mt5.TIMEFRAME_M1, n_bars=120)
    except Exception:
        pass
    try:
        _pre_m5_zones, _ = find_demand_supply_zones(bridge.symbol, mt5.TIMEFRAME_M5, n_bars=200)
    except Exception:
        pass
    try:
        _pre_m15_zones, _ = find_demand_supply_zones(bridge.symbol, mt5.TIMEFRAME_M15, n_bars=200)
    except Exception:
        pass
    # S/R levels (multi-touch) สำหรับ break-retest detector + Gate 1b (คำนวณครั้งเดียว ใช้ซ้ำ)
    _sr_levels = []
    try:
        _sr_levels = find_sr_levels(bridge.symbol)
    except Exception:
        pass

    # ── Pre-compute Multi-TF zones for retest detectors ─────────────────
    # เทรด M1 → เน้น M5/M15 (สเกลใกล้ M1) + H1/H4 เป็น context/significant level
    # M5 มี zone_retest ดูแลแล้ว → ที่นี่ใช้ M15 + H1 + H4 (กันพลาดโซนที่ M5 ยังไม่ก่อตัว)
    _pre_htf_zones = []
    try:
        _m15z, _ = find_demand_supply_zones(bridge.symbol, mt5.TIMEFRAME_M15, n_bars=80)
        _h1z, _ = find_demand_supply_zones(bridge.symbol, mt5.TIMEFRAME_H1, n_bars=60)
        _h4z, _ = find_demand_supply_zones(bridge.symbol, mt5.TIMEFRAME_H4, n_bars=60)
        for _z in _m15z:
            _z["tf"] = "M15"
        for _z in _h1z:
            _z["tf"] = "H1"
        for _z in _h4z:
            _z["tf"] = "H4"
        _pre_htf_zones = _m15z + _h1z + _h4z   # M15 ก่อน (ใกล้ M1 สุด) → priority
    except Exception:
        pass

    # ── Wrapper: inject _det_atr so SL uses the CORRECT timeframe ATR ──
    def _wrap(fn, det_atr):
        """Wrap detector to inject _det_atr (timeframe-specific ATR for SL)"""
        def _inner(d, a):
            r = fn(d, a)
            if r:
                r["_det_atr"] = float(det_atr[-1]) if hasattr(det_atr, '__len__') else float(det_atr)
            return r
        return _inner

    _atr_m1_val = float(atr_m1[-1])
    _atr_m5_val = float(atr_m5[-1])

    # ── M5 Momentum Gate: Price Action + EMA fallback ──────────────────
    # v1 (EMA): ช้า ~15-20 นาที กว่า EMA cross → พลาดจุดเข้า
    # v2 (Price Action): ใช้ higher-low / lower-high → detect ภายใน 3 bars (15 นาที)
    #   UP:   3 higher-lows ติดกัน + range >= 0.5 ATR → momentum ขึ้น
    #   DOWN: 3 lower-highs ติดกัน + range >= 0.5 ATR → momentum ลง
    _srf_long_ok = True
    _srf_short_ok = True
    _m5_momentum = "NONE"
    _m1_engulf_at_zone = False  # M1 engulfing override flag
    _m5_reversal_entry = False  # True = entry ผ่าน M5-REVERSAL → ใช้ RR เต็ม (ไม่ใช่ counter-trend)
    try:
        _m5_lows = df_m5["low"].values
        _m5_highs = df_m5["high"].values
        _m5_atr_last = float(atr_m5[-1]) if len(atr_m5) > 0 else 1.0

        if len(_m5_lows) >= 5:
            # ── Price Action: higher-lows / lower-highs (FAST) ──
            _hl = sum(1 for i in range(-3, 0) if _m5_lows[i] > _m5_lows[i-1])
            _lh = sum(1 for i in range(-3, 0) if _m5_highs[i] < _m5_highs[i-1])
            _hl_range = _m5_lows[-1] - _m5_lows[-4]   # total range of HL move
            _lh_range = _m5_highs[-4] - _m5_highs[-1]  # total range of LH move

            if _hl >= 3 and _hl_range > 0.5 * _m5_atr_last:
                _m5_momentum = "UP"
            elif _lh >= 3 and _lh_range > 0.5 * _m5_atr_last:
                _m5_momentum = "DOWN"

        # ── EMA Fallback: ถ้า price action ไม่ชัด → ใช้ EMA ──
        if _m5_momentum == "NONE":
            _m5_closes = df_m5["close"].values
            if len(_m5_closes) >= 26:
                def _quick_ema(vals, period):
                    k = 2 / (period + 1)
                    e = vals[0]
                    for v in vals[1:]:
                        e = v * k + e * (1 - k)
                    return e
                _m5_e8 = _quick_ema(_m5_closes[-15:].tolist(), 8)
                _m5_e21 = _quick_ema(_m5_closes[-26:].tolist(), 21)
                _m5_spread = abs(_m5_e8 - _m5_e21) / _m5_atr_last if _m5_atr_last > 0 else 0
                if _m5_e8 < _m5_e21 and _m5_spread > 0.15:
                    _m5_momentum = "DOWN"
                elif _m5_e8 > _m5_e21 and _m5_spread > 0.15:
                    _m5_momentum = "UP"

        # SRF gate (keep original behavior)
        if _m5_momentum == "DOWN":
            _srf_long_ok = False
        elif _m5_momentum == "UP":
            _srf_short_ok = False

        # ── M1 Engulfing Override: ถ้ามี engulfing บน zone → override momentum guard ──
        # user feedback: M1 engulfing ที่ Demand = สัญญาณ reversal ชัดเจน
        _m1_o = df_m1["open"].values
        _m1_c = df_m1["close"].values
        _m1_h = df_m1["high"].values
        _m1_l = df_m1["low"].values
        if len(_m1_c) >= 3:
            for _ei in range(-3, 0):
                _e_body = _m1_c[_ei] - _m1_o[_ei]
                _e_prev_body = _m1_c[_ei-1] - _m1_o[_ei-1]
                _e_range = _m1_h[_ei] - _m1_l[_ei]
                # Bullish engulfing: current green > prev red, body > 0.3 ATR
                if (_e_body > 0 and _e_prev_body < 0 and
                    abs(_e_body) > abs(_e_prev_body) * 0.8 and
                    _e_range > 0 and abs(_e_body) > 0.3 * _m5_atr_last):
                    _m1_engulf_at_zone = True  # bullish engulfing found
                    break
                # Bearish engulfing: current red > prev green
                if (_e_body < 0 and _e_prev_body > 0 and
                    abs(_e_body) > abs(_e_prev_body) * 0.8 and
                    _e_range > 0 and abs(_e_body) > 0.3 * _m5_atr_last):
                    _m1_engulf_at_zone = True  # bearish engulfing found
                    break
    except Exception:
        pass

    # Run all detectors: M1 finest → M5 medium → M15 context
    detectors = [
        # ════ 2026-06-18 user: SUPPLY/DEMAND REJECTION (TOP — flip-trigger) ════
        # wicks 3+ ที่ supply/demand + red/green strong = reversal entry + flip opposite pos
        _wrap(lambda d, a: detect_supply_rejection_short(df_m1, atr_m1, _pre_m5_zones), atr_m1),
        _wrap(lambda d, a: detect_demand_rejection_long(df_m1, atr_m1, _pre_m5_zones), atr_m1),
        _wrap(lambda d, a: detect_supply_rejection_short(df_m5, atr_m5, _pre_m5_zones), atr_m5),
        _wrap(lambda d, a: detect_demand_rejection_long(df_m5, atr_m5, _pre_m5_zones), atr_m5),
        # ════ 2026-06-18 user: CONSOLIDATION BREAKOUT (≥5 แท่งกระจุก + IMB ปิดทะลุ) ════
        # เข้าที่ close แท่ง breakout เลย (ไม่ retest). ตรง playbook user ที่ส่ง 5 ภาพ
        _wrap(lambda d, a: detect_consolidation_breakout_long(df_m1, atr_m1), atr_m1),
        _wrap(lambda d, a: detect_consolidation_breakout_short(df_m1, atr_m1), atr_m1),
        _wrap(lambda d, a: detect_consolidation_breakout_long(df_m5, atr_m5), atr_m5),
        _wrap(lambda d, a: detect_consolidation_breakout_short(df_m5, atr_m5), atr_m5),
        # ════ MOMENTUM BREAKOUT (user 2026-06-12: "ขึ้นแบบนี้บอทต้อง buy ตาม momentum") ════
        # vertical impulse ทะลุ base (ไม่ต้อง retest) — จับ rally ที่พุ่งตรงตั้งแต่ต้น move
        # is_mb + ยกเว้น range_pos extreme; spike-guard ($40) กันไล่ยอด. M5 (ชัด) + M1 (เร็ว)
        _wrap(lambda d, a: detect_momentum_breakout_long(df_m5, atr_m5), atr_m5),
        _wrap(lambda d, a: detect_momentum_breakout_short(df_m5, atr_m5), atr_m5),
        _wrap(lambda d, a: detect_momentum_breakout_long(df_m1, atr_m1), atr_m1),
        _wrap(lambda d, a: detect_momentum_breakout_short(df_m1, atr_m1), atr_m1),
        # ════ TOP PRIORITY: BREAK-RETEST (playbook user 2026-06-04) ════
        # breakdown→retest→sell / breakout→retest→buy ที่ S/R จริง (multi-touch) + reject candle
        # เช็ค M5 (โครงสร้างชัด) แล้ว M1 (จังหวะเข้า). regime gate คุมทิศ
        _wrap(lambda d, a: detect_break_retest_short(df_m5, atr_m5, _sr_levels), atr_m5),
        _wrap(lambda d, a: detect_break_retest_long(df_m5, atr_m5, _sr_levels), atr_m5),
        _wrap(lambda d, a: detect_break_retest_short(df_m1, atr_m1, _sr_levels), atr_m1),
        _wrap(lambda d, a: detect_break_retest_long(df_m1, atr_m1, _sr_levels), atr_m1),
        # ── OVERSOLD/OVERBOUGHT BOUNCE: catch ก้น/ยอด reversal (counter-trend ที่ extreme) ──
        _wrap(lambda d, a: detect_oversold_bounce_long(df_m5, atr_m5), atr_m5),
        _wrap(lambda d, a: detect_oversold_bounce_short(df_m5, atr_m5), atr_m5),
        # ── TRENDLINE: 3+ swing touches → trade reject/bounce (user 2026-06-06) ──
        _wrap(lambda d, a: detect_trendline_short(df_m5, atr_m5), atr_m5),
        _wrap(lambda d, a: detect_trendline_long(df_m5, atr_m5), atr_m5),
        # ── HL-RETEST DISABLED 2026-06-09: ขาดทุน -$503 ──
        # ── LIQUIDITY SWEEP: stop hunt + reverse (new 2026-06-09) ──
        _wrap(lambda d, a: detect_liq_sweep_long(df_m5, atr_m5), atr_m5),
        _wrap(lambda d, a: detect_liq_sweep_short(df_m5, atr_m5), atr_m5),
        _wrap(lambda d, a: detect_liq_sweep_long(df_m1, atr_m1), atr_m1),
        _wrap(lambda d, a: detect_liq_sweep_short(df_m1, atr_m1), atr_m1),
        # ── IMB CONTINUATION: impulse breakout (new 2026-06-09 user request) ──
        _wrap(lambda d, a: detect_imb_continuation_long(df_m5, atr_m5), atr_m5),
        _wrap(lambda d, a: detect_imb_continuation_short(df_m5, atr_m5), atr_m5),
        _wrap(lambda d, a: detect_imb_continuation_long(df_m1, atr_m1), atr_m1),
        _wrap(lambda d, a: detect_imb_continuation_short(df_m1, atr_m1), atr_m1),
        # ── SUPPLY/DEMAND REACT (light version, user 2026-06-09) ──
        _wrap(lambda d, a: detect_supply_react_short(df_m5, atr_m5, _pre_m5_zones), atr_m5),
        _wrap(lambda d, a: detect_demand_react_long(df_m5, atr_m5, _pre_m5_zones), atr_m5),
        _wrap(lambda d, a: detect_supply_react_short(df_m1, atr_m1, _pre_m5_zones), atr_m1),
        _wrap(lambda d, a: detect_demand_react_long(df_m1, atr_m1, _pre_m5_zones), atr_m1),
        # ════ PRIORITY TIER: strategy หลักของ user (S/R + Demand/Supply ก่อน) ════
        # (2026-06-03 user: บอทต้องเทรด S/R/Demand/Supply ตรงๆ ก่อน ไม่ใช่ htf_retest M1 noise)
        # ── S/R แนวรับแนวต้าน (M15 multi-touch + rejection candle) ──
        lambda d, a: detect_sr_long(d, a, mt_long),
        lambda d, a: detect_sr_short(d, a, mt_short),
        # ── M1: CHOCH (ก่อนสุด — structure change สำคัญที่สุด) ──
        _wrap(lambda d, a: detect_choch_long(df_m1, atr_m1), atr_m1),
        _wrap(lambda d, a: detect_choch_short(df_m1, atr_m1), atr_m1),
        # ── M1 finest: DSF + FVG + Engulfing ──
        # XAU M1 ATR > $0.5 → ใช้ $5 (ค่าเดิม), EUR/Forex → 2.5×ATR
        _wrap(lambda d, a: detect_dsf_short(df_m1, atr_m1,
              min_swing_usd=5.0 if float(atr_m1[-1]) > 0.5 else float(atr_m1[-1]) * 2.5), atr_m1),
        _wrap(lambda d, a: detect_dsf_long(df_m1, atr_m1,
              min_swing_usd=5.0 if float(atr_m1[-1]) > 0.5 else float(atr_m1[-1]) * 2.5), atr_m1),
        _wrap(lambda d, a: detect_fvg_short(df_m1, atr_m1, min_gap_atr=0.3), atr_m1),
        _wrap(lambda d, a: detect_fvg_long(df_m1, atr_m1, min_gap_atr=0.3), atr_m1),
        _wrap(lambda d, a: detect_bull_engulfing(df_m1, atr_m1, min_body_atr=0.6), atr_m1),
        _wrap(lambda d, a: detect_bear_engulfing(df_m1, atr_m1, min_body_atr=0.6), atr_m1),
        # ── M1 Zone First Touch (highest priority — fresh zone contact) ──
        _wrap(lambda d, a: detect_zone_first_touch_long(df_m1, atr_m1, _pre_m5_zones), atr_m1),
        _wrap(lambda d, a: detect_zone_first_touch_short(df_m1, atr_m1, _pre_m5_zones), atr_m1),
        # ── M5 SRF Bounce (horizontal multi-touch support/resistance + bounce) ──
        # จับ SRF level ที่ zone detection ไม่เห็น (user framework: buy bounce ที่ support แข็ง)
        _wrap(lambda d, a: detect_srf_bounce_long(df_m5, atr_m5) if _srf_long_ok else None, atr_m5),
        _wrap(lambda d, a: detect_srf_bounce_short(df_m5, atr_m5) if _srf_short_ok else None, atr_m5),
        # ── M1 Pin Bar / Hammer + Star ที่ zone (candlestick + zone confirmation) ──
        # XAU liquidity grab/stop hunt ที่ S/D → pin bar rejection (เกิดบ่อย แม่นบนทอง)
        _wrap(lambda d, a: detect_pinbar_long(df_m1, atr_m1, _pre_m5_zones), atr_m1),
        _wrap(lambda d, a: detect_pinbar_short(df_m1, atr_m1, _pre_m5_zones), atr_m1),
        _wrap(lambda d, a: detect_star_long(df_m5, atr_m5, _pre_m5_zones), atr_m5),
        _wrap(lambda d, a: detect_star_short(df_m5, atr_m5, _pre_m5_zones), atr_m5),
        # ── M1 Zone Retest (SMC Zone-First): retest actual demand/supply zones ──
        _wrap(lambda d, a: detect_zone_retest_long(df_m1, atr_m1, _pre_m5_zones), atr_m1),
        _wrap(lambda d, a: detect_zone_retest_short(df_m1, atr_m1, _pre_m5_zones), atr_m1),
        # ── M1 SRF (Support-Resistance Flip): breakout → retest → entry ──
        # Gated by M5 EMA trend: ห้าม SRF_LONG ตอน M5 downtrend ชัด (ให้ momentum fire แทน)
        _wrap(lambda d, a: detect_srf_long(df_m1, atr_m1, lookback=60) if _srf_long_ok else None, atr_m1),
        _wrap(lambda d, a: detect_srf_short(df_m1, atr_m1, lookback=60) if _srf_short_ok else None, atr_m1),
        _wrap(lambda d, a: detect_demand_long(df_m1, atr_m1, wick_min=0.4, body_min=0.2), atr_m1),
        _wrap(lambda d, a: detect_supply_zone_short(df_m1, atr_m1, drop_atr=1.5, lookback=60), atr_m1),
        _wrap(lambda d, a: detect_breakout_long(df_m1, atr_m1), atr_m1),
        _wrap(lambda d, a: detect_breakout_short(df_m1, atr_m1), atr_m1),
        # ── M5: CHOCH + DSF + FVG + patterns ──
        _wrap(lambda d, a: detect_choch_long(df_m5, atr_m5), atr_m5),
        _wrap(lambda d, a: detect_choch_short(df_m5, atr_m5), atr_m5),
        _wrap(lambda d, a: detect_dsf_short(df_m5, atr_m5), atr_m5),
        _wrap(lambda d, a: detect_dsf_long(df_m5, atr_m5), atr_m5),
        _wrap(lambda d, a: detect_fvg_short(df_m5, atr_m5), atr_m5),
        _wrap(lambda d, a: detect_fvg_long(df_m5, atr_m5), atr_m5),
        _wrap(lambda d, a: detect_bull_engulfing(df_m5, atr_m5, min_body_atr=0.6), atr_m5),
        _wrap(lambda d, a: detect_bear_engulfing(df_m5, atr_m5, min_body_atr=0.6), atr_m5),
        # ── M5 Zone First Touch ──
        _wrap(lambda d, a: detect_zone_first_touch_long(df_m5, atr_m5, _pre_m5_zones), atr_m5),
        _wrap(lambda d, a: detect_zone_first_touch_short(df_m5, atr_m5, _pre_m5_zones), atr_m5),
        # ── M5 Zone Retest ──
        _wrap(lambda d, a: detect_zone_retest_long(df_m5, atr_m5, _pre_m5_zones), atr_m5),
        _wrap(lambda d, a: detect_zone_retest_short(df_m5, atr_m5, _pre_m5_zones), atr_m5),
        # ── M5 SRF (Support-Resistance Flip) — gated by M5 EMA trend ──
        _wrap(lambda d, a: detect_srf_long(df_m5, atr_m5, lookback=60) if _srf_long_ok else None, atr_m5),
        _wrap(lambda d, a: detect_srf_short(df_m5, atr_m5, lookback=60) if _srf_short_ok else None, atr_m5),
        _wrap(lambda d, a: detect_demand_long(df_m5, atr_m5, wick_min=0.4, body_min=0.2), atr_m5),
        _wrap(lambda d, a: detect_supply_zone_short(df_m5, atr_m5, drop_atr=2.0), atr_m5),
        _wrap(lambda d, a: detect_breakout_long(df_m5, atr_m5), atr_m5),
        _wrap(lambda d, a: detect_breakout_short(df_m5, atr_m5), atr_m5),
        # ── M5 Momentum Continuation (ใช้ M5 เพราะจับ pullback wave ได้เร็ว) ──
        _wrap(lambda d, a: detect_momentum_long(df_m5, atr_m5), atr_m5),
        _wrap(lambda d, a: detect_momentum_short(df_m5, atr_m5), atr_m5),
        # ── M15: same strategies as M1/M5 — TP อิง M15 ATR (ใหญ่กว่า) ──
        # user framework: เทรด M15 TP ตาม M15 (ไกลกว่า M1/M5) — เช่น $20-40
        detect_choch_long,
        detect_choch_short,
        detect_dsf_short,
        detect_dsf_long,
        detect_fvg_short,
        detect_fvg_long,
        detect_bull_engulfing,
        detect_bear_engulfing,
        detect_inside_bar_break_long,
        detect_inside_bar_break_short,
        detect_breakout_long,
        detect_breakout_short,
        detect_demand_long,
        detect_supply_zone_short,
        # M15 Zone strategies (zones จาก M15 → TP อิง M15 structure)
        _wrap(lambda d, a: detect_zone_retest_long(df, atr, _pre_m15_zones), atr),
        _wrap(lambda d, a: detect_zone_retest_short(df, atr, _pre_m15_zones), atr),
        _wrap(lambda d, a: detect_zone_first_touch_long(df, atr, _pre_m15_zones), atr),
        _wrap(lambda d, a: detect_zone_first_touch_short(df, atr, _pre_m15_zones), atr),
        _wrap(lambda d, a: detect_pinbar_long(df, atr, _pre_m15_zones), atr),
        _wrap(lambda d, a: detect_pinbar_short(df, atr, _pre_m15_zones), atr),
        _wrap(lambda d, a: detect_star_long(df, atr, _pre_m15_zones), atr),
        _wrap(lambda d, a: detect_star_short(df, atr, _pre_m15_zones), atr),
        _wrap(lambda d, a: detect_srf_long(df, atr, lookback=60) if _srf_long_ok else None, atr),
        _wrap(lambda d, a: detect_srf_short(df, atr, lookback=60) if _srf_short_ok else None, atr),
        _wrap(lambda d, a: detect_srf_bounce_long(df, atr), atr),
        _wrap(lambda d, a: detect_srf_bounce_short(df, atr), atr),
        # ── HTF Retest (FALLBACK — เดิม #1 ครอง 24 ไม้/วัน): H1/H4 retest เฉพาะเมื่อไม่มี setup อื่น ──
        _wrap(lambda d, a: detect_htf_retest_short(df_m1, atr_m1, _pre_htf_zones), atr_m1),
        _wrap(lambda d, a: detect_htf_retest_long(df_m1, atr_m1, _pre_htf_zones), atr_m1),
        # ── M15 Momentum Continuation ──
        detect_momentum_long,
        detect_momentum_short,
    ]

    setup = None
    for det in detectors:
        try:
            result = det(df, atr)
        except Exception as e:
            result = None
        if result and result.get("direction"):
            setup = result
            break

    # AGI overlay
    regime = classify_regime(df, classifier)
    vetoed = False
    final_direction = setup["direction"] if setup else None
    _orig_direction = setup["direction"] if setup else None   # เก็บไว้ก่อน veto chain (สำหรับ Clean Decision)
    if final_direction == "long" and regime in SKIP_LONG_REGIMES:
        vetoed = True
        final_direction = None
    elif final_direction == "short" and regime in SKIP_SHORT_REGIMES:
        vetoed = True
        final_direction = None

    # ── Weekend Session Veto ────────────────────────────────────────────
    if _is_weekend and final_direction:
        vetoed = True
        if setup:
            setup["reason"] += " [VETO: Weekend/off-market — no new entries]"
        final_direction = None

    # Multi-TF Structure Analysis (oracle-validated quality detector)
    h1_proxy = df.set_index(pd.date_range(end=dt.datetime.now(), periods=len(df), freq="15min"))
    h1_proxy = h1_proxy[["open","high","low","close"]].resample("1h").agg(
        {"open":"first","high":"max","low":"min","close":"last"}).dropna()
    try:
        structure = STRUCTURE_ENGINE.analyze(df, h1_df=h1_proxy, check_session=True)
    except Exception:
        structure = {"long_score": 0, "short_score": 0, "verdict": "ERROR"}

    # ── H1 Range Position Filter ─────────────────────────────────────────
    # SMC rule: LONG เข้าที่ Demand (ล่าง), SHORT เข้าที่ Supply (บน)
    # ตรวจ position ของราคาใน H1 72h range ก่อน execute
    # ถ้าขัดกับ zone → veto signal นั้น
    # ─────────────────────────────────────────────────────────────────────
    _h1_low = 0.0    # default (ป้องกัน NameError ใน Zone Flip Override)
    _h1_high = 0.0
    h1_range_pos = 0.5   # default neutral
    try:
        if len(h1_proxy) >= 12:
            h1_bars = min(len(h1_proxy), 72)
            _h1_low  = float(h1_proxy["low"].tail(h1_bars).min())
            _h1_high = float(h1_proxy["high"].tail(h1_bars).max())
            _h1_rng  = _h1_high - _h1_low
            if _h1_rng > 0:
                _cur = tick.get("bid", float(h1_proxy["close"].iloc[-1]))
                h1_range_pos = (_cur - _h1_low) / _h1_rng
    except Exception:
        pass

    # ── อ่าน dynamic config จาก auto_analyzer (live update ไม่ต้อง restart) ──
    _dyn_long_veto  = 0.55   # default
    _dyn_short_veto = 0.45
    try:
        _dyn_cfg_path = Path("live_logs/dynamic_config.json")
        if _dyn_cfg_path.exists():
            import json as _json
            _dyn = _json.loads(_dyn_cfg_path.read_text(encoding="utf-8"))
            _dyn_long_veto  = float(_dyn.get("xau", {}).get("h1_long_veto_pct",  0.55))
            _dyn_short_veto = float(_dyn.get("xau", {}).get("h1_short_veto_pct", 0.45))
    except Exception:
        pass

    # ── Structure-confirmed bypass: strategies ที่ zone/trend VETO ไม่ควรบล็อก ──
    # DSF: zone flip ยืนยันแล้ว → bypass zone VETO
    # CHOCH: structure change ยืนยันแล้ว → bypass M15 lag + zone VETO
    #        "CHOCH = ราคา break swing high/low → M15 EMA ยัง lag แต่ structure เปลี่ยนแล้ว"
    _breakout_strats = {"dsf_short", "dsf_long", "choch_long", "choch_short"}
    _is_breakout = setup and setup.get("strategy") in _breakout_strats

    # ── Cache M15 + M5 zones (จับ SRF/Demand ละเอียดกว่า M15 อย่างเดียว) ──
    # ปัญหาเดิม: M15 zones อาจไม่เห็น M1/M5 SRF zone (เช่น 4441-4443)
    # Fix: รวม M5 zones เข้ามาด้วย → จับ zone ที่ user เห็นบน M1 chart ได้
    _cached_m15_zones = []
    try:
        _cached_m15_zones, _ = find_demand_supply_zones(bridge.symbol, mt5.TIMEFRAME_M15, n_bars=48)
    except Exception:
        pass

    _cached_m5_zones = []
    try:
        _cached_m5_zones, _ = find_demand_supply_zones(bridge.symbol, mt5.TIMEFRAME_M5, n_bars=120)
    except Exception:
        pass

    # ── H1 Zone Detection (larger structure zones) ──────────────────────
    # H1 zones = zones ที่ใช้เวลา form นานกว่า (stronger) = context สำคัญ
    # ใช้ n_bars=100 = 100 ชั่วโมง ย้อนหลัง
    _cached_h1_zones = []
    try:
        _cached_h1_zones, _ = find_demand_supply_zones(bridge.symbol, mt5.TIMEFRAME_H1, n_bars=100)
    except Exception:
        pass

    # ── H4 Zone Detection (major structural zones) ───────────────────────
    # H4 zones = zones ขนาดใหญ่มาก = strong rejection/support areas
    # ใช้ n_bars=60 = 240 ชั่วโมง (10 วัน) ย้อนหลัง
    _cached_h4_zones = []
    try:
        _cached_h4_zones, _ = find_demand_supply_zones(bridge.symbol, mt5.TIMEFRAME_H4, n_bars=60)
    except Exception:
        pass

    # รวม M15 + M5 zones สำหรับ entry detection (deduplicate)
    _all_zones = list(_cached_m15_zones)
    _m15_atr = float(atr[-1]) if len(atr) > 0 else 8.0
    for z5 in _cached_m5_zones:
        overlaps = any(
            abs(z5["lo"] - z15["lo"]) < _m15_atr * 0.5 and z5["type"] == z15["type"]
            for z15 in _cached_m15_zones
        )
        if not overlaps:
            _all_zones.append(z5)

    # _htf_zones = H1+H4 zones สำหรับ context/warning (ไม่ใช่ entry signal)
    # เก็บแยกเพื่อใช้: 1) TP cap  2) entry warning  3) Zone Lock HTF
    _htf_zones = []
    _htf_atr = float(atr[-1]) * 2.0  # H1/H4 zones กว้างกว่า tolerance ใหญ่ขึ้น
    for zh1 in _cached_h1_zones:
        zh1["tf"] = "H1"
        _htf_zones.append(zh1)
    for zh4 in _cached_h4_zones:
        # dedup กับ H1 zones
        overlaps_h1 = any(
            abs(zh4["lo"] - zh1["lo"]) < _htf_atr and zh4["type"] == zh1["type"]
            for zh1 in _cached_h1_zones
        )
        if not overlaps_h1:
            zh4["tf"] = "H4"
            _htf_zones.append(zh4)

    # ── M1 CHOCH Bias (เร็วกว่า M15 Structure Bias) ─────────────────────
    # ปัญหา: M15 bias ใช้ swing points 3 bars each side (ช้า ~45 min lag)
    # Fix: ตรวจ M1 CHOCH โดยตรง → ถ้าเจอ CHOCH ล่าสุดภายใน 30 bars
    #      ใช้ bias จาก CHOCH นั้นแทน (เร็วกว่า M15 bias 10-30 นาที)
    # ผลลัพธ์: หลัง CHOCH BULL → block bear_engulfing/fvg_short ทันที
    #          ไม่ต้องรอ M15 swing points form
    _m1_choch_bias = None  # None = ไม่เจอ CHOCH
    try:
        _choch_long_test = detect_choch_long(df_m1, atr_m1, lookback=30)
        _choch_short_test = detect_choch_short(df_m1, atr_m1, lookback=30)
        if _choch_long_test:
            _m1_choch_bias = "LONG"
        elif _choch_short_test:
            _m1_choch_bias = "SHORT"
    except Exception:
        pass
    # M5 CHOCH as fallback
    if _m1_choch_bias is None:
        try:
            _choch_long_m5 = detect_choch_long(df_m5, atr_m5, lookback=20)
            _choch_short_m5 = detect_choch_short(df_m5, atr_m5, lookback=20)
            if _choch_long_m5:
                _m1_choch_bias = "LONG"
            elif _choch_short_m5:
                _m1_choch_bias = "SHORT"
        except Exception:
            pass

    # ── Zone Detection (v2): ใช้ actual zones แทน H1 range % ─────────────
    # ใช้ _all_zones (M15+M5 รวม) แทน M15 อย่างเดียว
    # FIX: ใช้ M5 ATR แทน M15 ATR สำหรับ tolerance
    # เพราะ M15 ATR ช่วงกลางคืนยังสูง (จาก London/NY) → ขยาย zone ซ้อนกัน
    # M5 ATR ปรับตัวเร็วกว่า → tolerance เหมาะกับ volatility จริง
    _actual_in_supply = h1_range_pos >= _dyn_long_veto   # fallback
    _actual_in_demand = h1_range_pos <= _dyn_short_veto  # fallback
    try:
        _mid_hv = tick.get("bid", 0.0)
        _zone_tol_atr = float(atr_m5[-1]) if len(atr_m5) > 0 else float(atr[-1])
        _actual_in_supply = any(
            z["type"] == "SUPPLY" and z["lo"] - _zone_tol_atr * 0.3 <= _mid_hv <= z["hi"] + _zone_tol_atr * 0.3
            for z in _all_zones
        )
        _actual_in_demand = any(
            z["type"] == "DEMAND" and z["lo"] - _zone_tol_atr * 0.3 <= _mid_hv <= z["hi"] + _zone_tol_atr * 0.3
            for z in _all_zones
        )
    except Exception:
        pass  # ใช้ fallback h1_range_pos

    _zone_strat = setup.get("strategy", "") if setup else ""
    # Zone strategies เข้าที่ zone boundary โดยตรง → ไม่ควรถูก zone position veto บล็อก
    # (strategy ตรวจ zone เองแล้ว เช่น zone_retest_short = ที่ Supply จริง)
    _is_zone_strategy = _zone_strat in (
        "zone_retest_long", "zone_retest_short",
        "zone_first_touch_long", "zone_first_touch_short",
        "srf_bounce_long", "srf_bounce_short",
        "pinbar_long", "pinbar_short",
        "star_long", "star_short",
        "htf_retest_long", "htf_retest_short",
    )

    if final_direction and not vetoed:
        # LONG ห้ามเข้าถ้าอยู่ใน Supply zone จริง (ยกเว้น DSF flip + zone strategies + ZONE-RANGE: buy ที่ก้น)
        if final_direction == "long" and _actual_in_supply and not _is_breakout and not _is_zone_strategy and not (_FIX_ZONE_RANGE_CONFLICT and h1_range_pos <= 0.35):
            vetoed = True
            if setup:
                setup["reason"] += f" [VETO: BUY@Supply-zone ({h1_range_pos*100:.0f}%)]"
        # SHORT ห้ามเข้าถ้าอยู่ใน Demand zone จริง (ยกเว้น DSF flip + zone strategies + ZONE-RANGE: short ที่ยอด)
        elif final_direction == "short" and _actual_in_demand and not _is_breakout and not _is_zone_strategy and not (_FIX_ZONE_RANGE_CONFLICT and h1_range_pos >= 0.65):
            vetoed = True
            if setup:
                setup["reason"] += f" [VETO: SHORT@Demand-zone ({h1_range_pos*100:.0f}%)]"
        # supply_zone strategy ต้องอยู่ใน SUPPLY จริง (≥60%) ไม่ใช่ MID zone
        elif (final_direction == "short"
              and setup and setup.get("strategy") == "supply_zone"
              and not _actual_in_supply):
            vetoed = True
            if setup:
                setup["reason"] += f" [VETO: supply_zone ต้องอยู่ใน Supply zone จริง ({h1_range_pos*100:.0f}%)]"
        if vetoed:
            final_direction = None

    # ── Zone Flip Override: zone ที่ถูก break แรง (IMB) จะ flip ทิศทาง ──────
    # ย้ายมาอยู่หลัง H1 Range Position Filter → BEFORE M15/H4/H1 Strong veto
    # เพื่อให้ un-veto แล้วยังต้องผ่าน trend filters อีกชั้น
    # "BUY บน supply ได้ถ้า supply ถูก break แรงๆ แล้วราคา retest กลับมา"
    if vetoed and setup and setup.get("direction"):
        raw_dir = setup["direction"]
        zone_vetoed = (
            (raw_dir == "long"  and "Supply" in setup.get("reason","")) or
            (raw_dir == "short" and "Demand" in setup.get("reason",""))
        )
        if zone_vetoed:
            try:
                m15_bars_np = mt5.copy_rates_from_pos(bridge.symbol,
                                                       mt5.TIMEFRAME_M15, 0, 50)
                cur_price  = tick.get("bid", 0.0)
                flipped, flip_desc = detect_zone_flip(
                    m15_bars_np, cur_price, _h1_low, _h1_high, raw_dir)
                if flipped:
                    vetoed = False
                    final_direction = raw_dir
                    setup["reason"] += f" [{flip_desc}]"
            except Exception:
                pass

    # ── M1/M5 CHOCH Bias Filter (Zone-First จาก User SMC Framework) ──────
    # ปัญหา: M15 Structure Bias ช้า (ต้องรอ swing form 3 bars = ~45 นาที)
    #         ระหว่างนั้น pattern สวนทิศ CHOCH ยังเข้าได้
    # Fix: ถ้า M1/M5 CHOCH ยืนยัน bias → block pattern สวนทิศทันที
    #
    # User SMC insight: หลัง CHOCH BULL → ทุก pullback = BUY opportunity
    #   Bear Engulfing ที่ Demand zone หลัง CHOCH BULL ≠ SELL signal
    #   มันคือ "pullback ลงมา zone เพื่อ BUY" ไม่ใช่ "reversal SHORT"
    #
    # เฉพาะ pattern strategies (engulfing, FVG, momentum, breakout, inside_bar)
    # ทุก strategy ต้อง respect CHOCH bias
    # CHOCH ของตัวเอง (M5/M15) ก็ต้อง respect CHOCH ของ M1 เพราะ M1 เร็วกว่า
    _pattern_strats = {
        "bull_engulf", "bear_engulf",
        "fvg_short", "fvg_long",
        "momentum_long", "momentum_short",
        "breakout_long", "breakout_short",
        "inside_bar_break", "inside_bar_break_short",
        "sr_long", "sr_short",
        "zone_retest_long", "zone_retest_short",
        "zone_first_touch_long", "zone_first_touch_short",
        "srf_long", "srf_short",
        "choch_long", "choch_short",
        "dsf_long", "dsf_short",
    }
    if (final_direction and not vetoed and _m1_choch_bias is not None
            and setup and setup.get("strategy") in _pattern_strats):
        _strat_name = setup.get("strategy", "")
        if _m1_choch_bias == "LONG" and final_direction == "short":
            # CHOCH BULL confirmed → block SHORT patterns
            vetoed = True
            setup["reason"] += f" [VETO: CHOCH BULL confirmed → {_strat_name} SHORT blocked]"
            final_direction = None
        elif _m1_choch_bias == "SHORT" and final_direction == "long":
            # CHOCH SHORT confirmed → block LONG patterns
            vetoed = True
            setup["reason"] += f" [VETO: CHOCH SHORT confirmed → {_strat_name} LONG blocked]"
            final_direction = None

    # ── HTF Zone Awareness (H1/H4 context) — TP TARGET เท่านั้น ──────────
    # user feedback (6/2): เทรด M1 ตีโซนจาก M1/M5 — H1/H4 ไม่ควร BLOCK entry
    #   H1/H4 ใช้สำหรับ: TP ceiling/floor (เก็บยาวถึงโซนใหญ่) เท่านั้น
    #   in-zone block ลบออก: zone_retest/srf/htf_retest self-validate อยู่แล้ว
    # ผล: bot จะเข้า setup M1 ได้ตามที่ user ชี้ + TP วิ่งถึงโซน H1/H4 ข้างหน้า
    if final_direction and not vetoed and setup and _htf_zones:
        _htf_price = float(tick.get("bid", 0) if isinstance(tick, dict) else
                           (tick.bid if tick else 0))
        # Tolerance 0.3 ATR (ลดจาก 0.8) — 0.8 กว้างเกิน ทำให้ H1 Demand
        # ขยายจาก 4506 ไปถึง 4512 → block SHORT ทั้งหมด = DEADLOCK
        _htf_tol = float(atr[-1]) * 0.3

        # LONG block: ทั้ง supply zone (resistance ทั้งโซน — ไม่ซื้อชนเพดาน)
        # SHORT block: เฉพาะครึ่งล่างของ demand (ใกล้ support floor จริง)
        #   → กัน demand กว้าง/ซ้อน over-block: 4490 ครึ่งบน demand → ยอม short (sell ที่ resistance)
        #     4486 ครึ่งล่าง demand (ใกล้ floor) → block short (bounce แน่)
        _in_htf_supply = any(
            z["type"] == "SUPPLY" and
            (z["lo"] - _htf_tol) <= _htf_price <= (z["hi"] + _htf_tol)
            for z in _htf_zones
        )
        _in_htf_demand = any(
            z["type"] == "DEMAND" and
            (z["lo"] - _htf_tol) <= _htf_price <= (z["lo"] + 0.55 * (z["hi"] - z["lo"]))
            for z in _htf_zones
        )

        # หา H1/H4 zone ที่ใกล้ที่สุดในทิศตรงข้าม (สำหรับ TP cap)
        _htf_opposing_zone = None
        if final_direction == "long":
            # หา H1/H4 Supply zone ที่อยู่เหนือราคา (= TP ceiling)
            supply_above = [z for z in _htf_zones
                           if z["type"] == "SUPPLY" and z["lo"] > _htf_price]
            if supply_above:
                _htf_opposing_zone = min(supply_above, key=lambda z: z["lo"])
        elif final_direction == "short":
            # หา H1/H4 Demand zone ที่อยู่ใต้ราคา (= TP floor)
            demand_below = [z for z in _htf_zones
                           if z["type"] == "DEMAND" and z["hi"] < _htf_price]
            if demand_below:
                _htf_opposing_zone = max(demand_below, key=lambda z: z["hi"])

        _strat_htf = setup.get("strategy", "")
        # Zone-based strategies bypass HTF veto:
        # ถ้า zone_retest_short fire ที่ Supply zone → H1 Demand zone ข้างล่างไม่ควร block
        # เพราะ signal มาจาก zone ที่ confirm แล้ว (ไม่ใช่ random pattern)
        _htf_exceptions = {
            "srf_long", "srf_short",
            "srf_bounce_long", "srf_bounce_short",
            "zone_first_touch_long", "zone_first_touch_short",
            "zone_retest_long", "zone_retest_short",
            "pinbar_long", "pinbar_short",
            "star_long", "star_short",
        }

        # ── ลบ in-zone block ออกทั้งหมด ──────────────────────────────────
        # เทรด M1: ตีโซนจาก M1 — H1/H4 ไม่ block entry
        # Wall Guard (M5/M15) จัดการ "ห้ามชนกำแพงใกล้" อยู่แล้ว
        # HTF ที่นี่ใช้แค่ TP target เท่านั้น
        # (เดิม block 121 ครั้งใน 1 วัน → miss ทุก entry ที่ user ชี้)
        if False and _in_htf_demand and final_direction == "short":
            # ← disabled: ลบ H1 in-zone block ออก
            _htf_zone_info = next((z for z in _htf_zones if z["type"] == "DEMAND" and
                                   (z["lo"] - _htf_tol) <= _htf_price <= (z["hi"] + _htf_tol)), {})
            vetoed = True
            setup["reason"] += (f" [VETO: HTF Demand {_htf_zone_info.get('tf','H?')} "
                                 f"{_htf_zone_info.get('lo',0):.0f}-{_htf_zone_info.get('hi',0):.0f}"
                                 f" → SHORT blocked]")
            final_direction = None
            print(f"   [HTF-BLOCK] {_strat_htf} SHORT blocked — inside H1/H4 Demand zone "
                  f"{_htf_zone_info.get('lo',0):.0f}-{_htf_zone_info.get('hi',0):.0f}", flush=True)

        elif _htf_opposing_zone and not vetoed:
            # H1/H4 zone ขวาง TP → cap TP ที่ zone boundary (เก็บกำไรก่อน H1 zone)
            # ลบ no-room veto ออก: ไม่ block entry เพราะ H1 ใกล้ (เทรด M1 ตีโซน M1)
            # เหลือแค่ WARN + TP cap เท่านั้น (H1 = TP target ไม่ใช่ entry block)
            _zone_dist = abs(_htf_opposing_zone["lo"] - _htf_price) if final_direction == "long" \
                         else abs(_htf_price - _htf_opposing_zone["hi"])
            _zone_atr_dist = _zone_dist / max(float(atr[-1]), 1)
            if _zone_atr_dist < 5.0:
                # H1/H4 zone อยู่ใกล้ → warn ว่า TP จะถูก cap
                setup["reason"] += (f" [HTF-WARN: {_htf_opposing_zone.get('tf','H?')} "
                                     f"{_htf_opposing_zone['type']} "
                                     f"{_htf_opposing_zone['lo']:.0f}-{_htf_opposing_zone['hi']:.0f}"
                                     f" {_zone_dist:.1f}$ away → TP capped]")

    # ── Zone Direction Lock ──────────────────────────────────────────────
    # SMC Rule: ราคาอยู่ใน zone → เข้าได้เฉพาะทิศที่ zone กำหนด
    #
    # Priority: HTF (H1/H4) > LTF (M5/M15)
    # ถ้า H1/H4 บอก Supply → SHORT allowed ไม่ว่า M5 จะบอกอะไร
    # ถ้า H1/H4 บอก Demand → LONG allowed ไม่ว่า M5 จะบอกอะไร
    # ถ้าไม่มี HTF zone conflict → ใช้ M5/M15 zone
    if final_direction and not vetoed and setup:
        _cur_price = float(tick.get("bid", 0) if isinstance(tick, dict) else
                           (tick.bid if tick else 0))
        _zone_tol  = float(atr[-1]) * 0.5
        _htf_tol2  = float(atr[-1]) * 0.8

        # HTF zone status (priority)
        _htf_in_supply = any(
            z["type"] == "SUPPLY" and
            (z["lo"] - _htf_tol2) <= _cur_price <= (z["hi"] + _htf_tol2)
            for z in _htf_zones
        )
        _htf_in_demand = any(
            z["type"] == "DEMAND" and
            (z["lo"] - _htf_tol2) <= _cur_price <= (z["hi"] + _htf_tol2)
            for z in _htf_zones
        )

        # M5/M15 zone status (lower priority — override by HTF)
        _ltf_in_demand = any(
            z["type"] == "DEMAND" and
            (z["lo"] - _zone_tol) <= _cur_price <= (z["hi"] + _zone_tol)
            for z in _all_zones
        )
        _ltf_in_supply = any(
            z["type"] == "SUPPLY" and
            (z["lo"] - _zone_tol) <= _cur_price <= (z["hi"] + _zone_tol)
            for z in _all_zones
        )

        # ถ้า HTF กำหนดทิศแล้ว → M5 zone ที่ขัดกันจะถูก ignore
        # (H1/H4 Supply → SHORT is fine even if M5 says Demand here)
        _in_demand = _ltf_in_demand and not _htf_in_supply  # M5 Demand แต่ไม่ใช่ HTF Supply
        _in_supply = _ltf_in_supply and not _htf_in_demand  # M5 Supply แต่ไม่ใช่ HTF Demand

        _zone_lock_exceptions = {"srf_long", "srf_short", "choch_long", "choch_short",
                                  "zone_first_touch_long", "zone_first_touch_short"}
        _strat_now = setup.get("strategy", "")

        if _in_demand and final_direction == "short" and _strat_now not in _zone_lock_exceptions:
            vetoed = True
            setup["reason"] += f" [VETO: ZoneLock M5/M15 DEMAND → {_strat_now} SHORT blocked]"
            final_direction = None
        elif _in_supply and final_direction == "long" and _strat_now not in _zone_lock_exceptions:
            vetoed = True
            setup["reason"] += f" [VETO: ZoneLock M5/M15 SUPPLY → {_strat_now} LONG blocked]"
            final_direction = None

    # ── Wait for Zone Touch ──────────────────────────────────────────────
    # SMC Rule: Pattern strategies (momentum/engulf/fvg/breakout) ควรเข้าเฉพาะ
    #           เมื่อ price อยู่ใกล้ Demand (สำหรับ LONG) หรือ Supply (สำหรับ SHORT)
    #           ไม่ควรเข้าขณะ price อยู่กลางๆ ห่างจาก zone (= no man's land)
    #
    # ภาพที่ user ส่งมา: bot buy/sell ตอน price ห่างจาก zone = พลาดชัวร์
    # Fix: require price ≤ ZONE_TOUCH_MAX_DIST ATR จาก nearest relevant zone
    _wait_zone_strats = {
        "momentum_long", "momentum_short",
        "bull_engulf", "bear_engulf",
        "fvg_long", "fvg_short",
        "breakout_long", "breakout_short",
        "inside_bar_break", "inside_bar_break_short",
        "sr_long", "sr_short",
    }
    ZONE_TOUCH_MAX_DIST = 2.0   # ATR — ถ้า price ห่างจาก zone เกินนี้ → block

    if final_direction and not vetoed and setup:
        _wz_strat = setup.get("strategy", "")
        if _wz_strat in _wait_zone_strats:
            _wz_price = float(tick.get("bid", 0) if isinstance(tick, dict) else
                              (tick.bid if tick else 0))
            _wz_atr = float(atr[-1])
            _max_dist = ZONE_TOUCH_MAX_DIST * _wz_atr

            if final_direction == "long":
                # หา Demand zone ใกล้สุด
                _best_dist = 9999.0
                for z in _all_zones:
                    if z["type"] == "DEMAND":
                        # dist = 0 ถ้าอยู่ใน zone, dist > 0 ถ้าอยู่เหนือ zone
                        dist = max(0.0, _wz_price - z["hi"])
                        _best_dist = min(_best_dist, dist)
                if _best_dist > _max_dist:
                    vetoed = True
                    setup["reason"] += (f" [VETO: WaitZone — LONG but nearest Demand"
                                        f" ${_best_dist:.1f} away (max ${_max_dist:.1f})]")
                    final_direction = None

            elif final_direction == "short":
                # หา Supply zone ใกล้สุด
                _best_dist = 9999.0
                for z in _all_zones:
                    if z["type"] == "SUPPLY":
                        # dist = 0 ถ้าอยู่ใน zone, dist > 0 ถ้าอยู่ต่ำกว่า zone
                        dist = max(0.0, z["lo"] - _wz_price)
                        _best_dist = min(_best_dist, dist)
                if _best_dist > _max_dist:
                    vetoed = True
                    setup["reason"] += (f" [VETO: WaitZone — SHORT but nearest Supply"
                                        f" ${_best_dist:.1f} away (max ${_max_dist:.1f})]")
                    final_direction = None

    # ── H4 Trend Filter: ป้องกันเข้าสวนเทรนด์ใน MID zone ───────────────
    # ถ้าอยู่ใน MID zone (45-55%) และ signal ขัดกับ H4 trend → VETO
    # เพราะ MID = ไม่มี structural edge, H4 ต้องยืนยันทิศทาง
    if final_direction and not vetoed:
        try:
            h4_rates = mt5.copy_rates_from_pos(bridge.symbol, mt5.TIMEFRAME_H4, 0, 5)
            if h4_rates is not None and len(h4_rates) >= 4:
                # [-2] = last completed H4 bar, [-3] = previous completed bar
                h4_last_close = float(h4_rates[-2][4])
                h4_prev_close = float(h4_rates[-3][4])
                h4_bullish = h4_last_close > h4_prev_close
                h4_label = "BULL" if h4_bullish else "BEAR"
                # ใน MID zone เท่านั้น: signal ต้องสอดคล้องกับ H4
                if 0.45 < h1_range_pos < 0.55:
                    if final_direction == "short" and h4_bullish:
                        vetoed = True
                        if setup: setup["reason"] += f" [VETO: SHORT@MID+H4{h4_label}]"
                        final_direction = None
                    elif final_direction == "long" and not h4_bullish:
                        vetoed = True
                        if setup: setup["reason"] += f" [VETO: LONG@MID+H4{h4_label}]"
                        final_direction = None
        except Exception:
            pass

    # ── M15 Trend: คำนวณ label ก่อน (ใช้ทั้ง filter + log) ──────────────
    _m15_trend_label = ""
    _m15_bullish = None
    _m15_ema_sep = 0.0   # EMA separation (ATR) — 0 = chop default
    _m15_slope8 = 0.0    # ราคาเปลี่ยนกี่ ATR ใน 8 M15 bars (2h) — จับเทรนแบบ "ตา" เห็น (LH+LL)
    _m15_regime = "chop"
    try:
        m15_rates = mt5.copy_rates_from_pos(bridge.symbol, mt5.TIMEFRAME_M15, 0, 25)
        if m15_rates is not None and len(m15_rates) >= 22:
            closes_m15 = [float(b[4]) for b in m15_rates[-22:]]
            def _ema(vals, period):
                k = 2 / (period + 1)
                e = vals[0]
                for v in vals[1:]:
                    e = v * k + e * (1 - k)
                return e
            ema8  = _ema(closes_m15[-15:], 8)
            ema21 = _ema(closes_m15, 21)
            _m15_bullish = ema8 > ema21
            _m15_trend_label = "BULL" if _m15_bullish else "BEAR"
            # EMA separation (ATR units) — < 0.4 = chop (EMA flat), >= 0.4 = trend ชัด
            _m15_atr = float(atr[-1]) if len(atr) > 0 and atr[-1] > 0 else 1.0
            _m15_ema_sep = (ema8 - ema21) / _m15_atr
            # slope8 = ราคาเปลี่ยนกี่ ATR ใน 8 M15 bars (2h) — robust กว่า EMA ในเทรนค่อยเป็นค่อยไป
            if len(closes_m15) >= 9:
                _m15_slope8 = (closes_m15[-1] - closes_m15[-9]) / _m15_atr
    except Exception:
        pass

    # ── REGIME: ใช้ slope (price ลง/ขึ้นต่อเนื่อง = "ตา" เห็น) + EMA แยกชัด ──
    # (user 2026-06-04: บอทเห็น chop เพราะ EMA lag ในขาลงค่อยเป็นค่อยไป → ซื้อก้นในขาลง)
    # slope จับเทรนที่ EMA ตามไม่ทัน (verified: จุด user วง slope8=-1.2~-1.7=ขาลงชัด แต่ EMA=chop)
    # 2026-06-05 user (chart 16:51 chop $17 range): OR→AND. EMA lag เอาเอง trend
    # แต่ slope=+0.51 (chop) → trend_bull = ผิด. ต้องการทั้ง slope AND EMA agree
    if _m15_slope8 < -0.8 and _m15_bullish is False:
        _m15_regime = "trend_bear"
    elif _m15_slope8 > 0.8 and _m15_bullish is True:
        _m15_regime = "trend_bull"
    else:
        _m15_regime = "chop"

    # ── FIX #1 (2026-06-10): REGIME LAG OVERRIDE — price-action ตรวจเร็วกว่า EMA ──
    # ปัญหา: TV 12:52-14:28 trend_bull ค้าง 96 นาทีหลัง peak → BUY สวนขาลง 3 ไม้ -$83
    # แก้: ถ้า price drop > 2.0 ATR จาก M15 recent high → override trend_bull → chop
    #      ถ้า price rally > 2.0 ATR จาก M15 recent low → override trend_bear → chop
    # ใช้ M15 high/low 12 bars (3 ชม.) = จับ peak/dip ที่เพิ่ง establish
    _REGIME_OVERRIDE_ATR = 2.0
    try:
        if m15_rates is not None and len(m15_rates) >= 13 and _m15_atr > 0:
            _m15_recent_high12 = max(float(b[2]) for b in m15_rates[-13:-1])  # high of 12 closed bars
            _m15_recent_low12 = min(float(b[3]) for b in m15_rates[-13:-1])   # low of 12 closed bars
            _cur_price = closes_m15[-1]
            if _m15_regime == "trend_bull" and (_m15_recent_high12 - _cur_price) > _REGIME_OVERRIDE_ATR * _m15_atr:
                _m15_regime = "chop"
                print(f"   [REGIME-OVERRIDE] trend_bull→chop: price {_cur_price:.0f} dropped ${_m15_recent_high12 - _cur_price:.1f} from M15 high {_m15_recent_high12:.0f} (>{_REGIME_OVERRIDE_ATR}×ATR${_m15_atr:.1f})", flush=True)
            elif _m15_regime == "trend_bear" and (_cur_price - _m15_recent_low12) > _REGIME_OVERRIDE_ATR * _m15_atr:
                _m15_regime = "chop"
                print(f"   [REGIME-OVERRIDE] trend_bear→chop: price {_cur_price:.0f} rallied ${_cur_price - _m15_recent_low12:.1f} from M15 low {_m15_recent_low12:.0f} (>{_REGIME_OVERRIDE_ATR}×ATR${_m15_atr:.1f})", flush=True)
    except Exception:
        pass

    # ── FIX #2 REVERTED (2026-06-12): M5 FAST-TREND ถอดออก ──
    # เหตุ: ทำให้ regime flip trend_bear ตอน chop drop → บอท trend-follow ไล่ขายถึงก้น (sell ก้น = เข้าผิดจุด)
    #   หลักฐาน: Jun10 chop (โค้ดเก่า M15 regime) +$169 vs Jun12 chop (M5 fast-trend) -$278 = swing $447
    #   short วันนี้เข้าที่ก้น range (0%) ซ้ำๆ → SL. คืน M15 regime เดิม = ขายที่ยอด range (วินัยเก่า)
    # momentum_breakout detector ยังจับ spike จริงได้ (is_mb) — ไม่ต้องพึ่ง regime flip

    # ── M15 Trend Filter: ห้ามเข้าสวนเทรนด์ M15 ──────────────────────────────
    # M15 BULLISH = ห้าม SHORT สวน, M15 BEARISH = ห้าม LONG สวน
    #
    # extreme zone bypass (v2): ใช้ actual DSF zones แทน H1 range %
    # เหตุผล: H1 range 72h กว้างเกิน (ราคาสูงสุด 3 วันก่อน) ทำให้ 4495 ดูเหมือน
    #         "extreme demand" (18% ของ range) ทั้งที่จริงอยู่ใน Supply zone
    # → extreme_long_zone = True เฉพาะถ้าอยู่ใน Demand zone จริงๆ (DSF detect)
    # → extreme_short_zone = True เฉพาะถ้าอยู่ใน Supply zone จริงๆ
    _is_choch = setup and setup.get("strategy") in {"choch_long", "choch_short"}

    # ── M15 Structure Bias ──────────────────────────────────────────────────────
    # HH+HL = LONG bias, LH+LL = SHORT bias
    # Bias เปลี่ยนเมื่อ CHOCH ยืนยัน (ไม่ใช่ EMA lag)
    # ─────────────────────────────────────────────────────────────────────────────
    _m15_struct_bias = "NEUTRAL"
    try:
        _m15_bars_b = mt5.copy_rates_from_pos(bridge.symbol, mt5.TIMEFRAME_M15, 0, 50)
        _m15_struct_bias = compute_m15_structure_bias(_m15_bars_b)
    except Exception:
        pass

    # CHOCH override: ถ้า signal คือ CHOCH → bias flip ทันที
    if _is_choch and setup:
        _m15_struct_bias = "LONG" if setup.get("strategy") == "choch_long" else "SHORT"

    if final_direction and not vetoed and _m15_bullish is not None and not _is_choch:
        # CHOCH ยืนยัน structure change → skip M15 EMA filter (EMA lag หลัง CHOCH)
        # Non-CHOCH strategies ยังคง filter ตามปกติ
        extreme_long_zone  = h1_range_pos <= 0.20  # fallback: เข้มขึ้น 35%→20%
        extreme_short_zone = h1_range_pos >= 0.80  # fallback: เข้มขึ้น 65%→80%
        try:
            _z_all = _all_zones  # M15+M5 zones combined
            _mid = tick.get("bid", 0.0)
            extreme_long_zone = any(
                z["type"] == "DEMAND" and z["lo"] <= _mid <= z["hi"]
                for z in _z_all
            )
            extreme_short_zone = any(
                z["type"] == "SUPPLY" and z["lo"] <= _mid <= z["hi"]
                for z in _z_all
            )
        except Exception:
            pass  # fallback ใช้ H1 range %

        # ── Bias Override: ถ้า M15 Structure Bias ตรงกับทิศ signal → EMA ไม่ veto ──
        # เหตุผล: หลัง CHOCH, swing structure flip ก่อน EMA cross → Bias เร็วกว่า EMA
        # ป้องกัน DEADLOCK: EMA=BULL blocks SHORT + Bias=SHORT blocks LONG = ทุกทิศถูก block
        _bias_agrees = (
            (_m15_struct_bias == "SHORT" and final_direction == "short") or
            (_m15_struct_bias == "LONG"  and final_direction == "long")
        )

        # ── M5 Momentum Reversal Override ────────────────────────────────
        # จาก user feedback (5/29 top @ 4592): momentum กลับตัวลง แต่ M15=BULL (lag)
        # → block ทุก SHORT signal (breakout/srf/choch/zone ที่ Supply)
        # Fix: ถ้า M5 momentum ยืนยันทิศ signal → bypass M15 EMA veto (จับ reversal)
        #   counter-trend TP cap (1.5x) ยังทำงาน → ความเสี่ยงควบคุมได้
        _m5_momentum_agrees = (
            (final_direction == "short" and _m5_momentum == "DOWN") or
            (final_direction == "long"  and _m5_momentum == "UP")
        )

        if final_direction == "short" and _m15_bullish and not extreme_short_zone and not _bias_agrees and not _m5_momentum_agrees:
            vetoed = True
            if setup:
                setup["reason"] += f" [VETO: M15={_m15_trend_label} แต่เข้า SHORT สวนเทรนด์ (zone {h1_range_pos*100:.0f}%)]"
            final_direction = None
        elif final_direction == "long" and not _m15_bullish and not extreme_long_zone and not _bias_agrees and not _m5_momentum_agrees:
            vetoed = True
            if setup:
                setup["reason"] += f" [VETO: M15={_m15_trend_label} แต่เข้า LONG สวนเทรนด์ (zone {h1_range_pos*100:.0f}%)]"
            final_direction = None
        elif _m5_momentum_agrees and ((final_direction == "short" and _m15_bullish) or (final_direction == "long" and not _m15_bullish)):
            _m5_reversal_entry = True   # confirmed reversal → ใช้ RR เต็ม (ไม่ cap counter-trend)
            if setup:
                setup["reason"] += f" [M5-REVERSAL-OK: M5 momentum={_m5_momentum} ยืนยัน {final_direction.upper()} → bypass M15 lag, RR เต็ม]"

    # ── S/R Wall Guard: อย่า short ชน support / อย่า buy ชน resistance ──────
    # user framework: ราคาที่ support/SRF แข็ง → มักเด้ง → ห้าม SHORT (รอ buy bounce)
    #                 ราคาที่ resistance แข็ง → มักถูก reject → ห้าม BUY
    # ใช้กับ momentum/breakout ด้วย (เทรดกลางทางได้ แต่ห้ามชนโซนตรงข้ามที่ใกล้)
    # ยกเว้น breakout-retest: จัดการด้วย _broke check ข้างใน (ทะลุ level แล้ว = ผ่าน)
    #   ไม่ใช่ exempt breakout ทั้งก้อน (00:53 breakout_short ชน demand = ต้องโดนกัน)
    # wall = M5/M15 zones + recent M5 swing pivots (จับ SRF level ที่ zone detection ไม่เห็น)
    if final_direction and not vetoed:
        try:
            _wall_atr = float(atr_m5[-1]) if len(atr_m5) > 0 else float(atr[-1])
            _price_now = tick.get("bid", 0.0) if isinstance(tick, dict) else float(tick.bid)
            # proximity: ราคาต้อง "ติด" level จริงๆ (กำลังจะเด้ง) ถึงบล็อก
            # ไม่ใช่ block ทุก scalp ที่มี level อยู่ห่างๆ (short เก็บ $4 ถึง support = OK)
            # XAU $2-3.5, forex 1.0×ATR
            _PROX = (1.0 * _wall_atr) if _is_forex else min(max(1.5 * _wall_atr, 2.0), 3.5)
            _m5_h = df_m5["high"].values
            _m5_l = df_m5["low"].values

            # รวม support/resist: zones (M5/M15) + recent M5 swing pivots (lookback 40)
            _supports = [z["lo"] for z in _all_zones if z["type"] == "DEMAND"]
            _resists  = [z["hi"] for z in _all_zones if z["type"] == "SUPPLY"]
            _lb = min(40, len(_m5_l) - 3)
            for _i in range(len(_m5_l) - 3, len(_m5_l) - _lb, -1):
                if _i < 2:
                    break
                if (_m5_l[_i] <= _m5_l[_i-1] and _m5_l[_i] <= _m5_l[_i-2]
                        and _m5_l[_i] <= _m5_l[_i+1] and _m5_l[_i] <= _m5_l[_i+2]):
                    _supports.append(float(_m5_l[_i]))
                if (_m5_h[_i] >= _m5_h[_i-1] and _m5_h[_i] >= _m5_h[_i-2]
                        and _m5_h[_i] >= _m5_h[_i+1] and _m5_h[_i] >= _m5_h[_i+2]):
                    _resists.append(float(_m5_h[_i]))

            _recent_low = float(min(_m5_l[-10:])) if len(_m5_l) >= 10 else _price_now
            _recent_high = float(max(_m5_h[-10:])) if len(_m5_h) >= 10 else _price_now

            if final_direction == "short":
                # support ที่อยู่ใกล้ "ใต้/ที่" ราคา (ราคากำลังจะชน) ภายใน PROX
                _near = [s for s in _supports if (_price_now - _PROX) <= s <= (_price_now + 0.2 * _wall_atr)]
                if _near:
                    _sup = max(_near)   # support ใกล้ราคาที่สุด
                    _broke = _recent_low < _sup - 0.5 * _wall_atr   # ทะลุลงใต้ support แล้ว?
                    if not _broke:
                        vetoed = True
                        if setup:
                            setup["reason"] += f" [VETO: SHORT ชน support {_sup:.0f} (${_price_now - _sup:.1f} ใต้ราคา) → รอ bounce/ทะลุ]"
                        final_direction = None
            elif final_direction == "long":
                # resistance ที่อยู่ใกล้ "เหนือ/ที่" ราคา ภายใน PROX
                _near = [r for r in _resists if (_price_now - 0.2 * _wall_atr) <= r <= (_price_now + _PROX)]
                if _near:
                    _res = min(_near)   # resistance ใกล้ราคาที่สุด
                    _broke = _recent_high > _res + 0.5 * _wall_atr   # ทะลุขึ้นเหนือ resistance แล้ว?
                    if not _broke:
                        vetoed = True
                        if setup:
                            setup["reason"] += f" [VETO: LONG ชน resistance {_res:.0f} (${_res - _price_now:.1f} เหนือราคา) → รอ reject/ทะลุ]"
                        final_direction = None
        except Exception:
            pass

    # ── Trade-from-Zone Gate: LOOSENED (เอา hard gate ออก) ─────────────────
    # เหตุผล: srf = breakout-retest → flip level เป็น NEW structure (R→S/S→R)
    #   ไม่ใช่ swing-based zone → บังคับ _all_zones = บล็อก branch ③④ (over-filter)
    # srf self-validate structure แล้ว (flip+touches+retest) + Wall Guard กันชนกำแพง
    # ถ้า srf เข้าจุดอ่อน → แก้คุณภาพที่ min_touches ของ detector (soft) ไม่ใช่ hard veto
    #   (ตามหลัก user: ลดความผิดพลาดชัดๆ ไม่ over-filter valid setup)

    # ── Pullback@Supply VETO: "โมเมนตั้มขาขึ้นในขาลง = Pullback ไม่ใช่ Reversal" ─
    # Logic (ตาม user framework):
    #   1. M15 BEARISH = overall downtrend
    #   2. ราคาอยู่ใน Supply zone = pullback ขึ้นมาถึง Supply
    #   3. → บล็อก LONG ทุก strategy ยกเว้น dsf_long (zone flip ยืนยันแล้ว)
    #
    # กรณีที่บล็อก: momentum_long, breakout_long, fvg_long, bull_engulf ฯลฯ
    #   ที่ Supply zone ในขาลง = pullback เข้า Supply → ควร SHORT ไม่ใช่ LONG
    if final_direction == "long" and not vetoed:
        _strat = setup.get("strategy", "") if setup else ""
        if _strat not in {"dsf_long", "demand_long"}:  # demand_long = at Demand = OK
            try:
                _pb_zones = _all_zones  # M15+M5 zones combined
                _mid = tick.get("bid", 0.0)
                _at_supply = any(
                    z["type"] == "SUPPLY"
                    and z["lo"] - atr[-1] * 0.3 <= _mid <= z["hi"] + atr[-1] * 0.3
                    for z in _pb_zones
                )
                if _at_supply and _m15_bullish is False:
                    # M15 BEARISH + ราคาอยู่ใน Supply = Pullback ไม่ใช่ Reversal
                    vetoed = True
                    setup["reason"] += " [VETO: Pullback@Supply+M15BEAR → SHORT opp not LONG]"
                    final_direction = None
            except Exception:
                pass

    # ── M15 Structure Bias Filter ──────────────────────────────────────────────
    # Bias = M15 swing structure (HH+HL = LONG, LH+LL = SHORT)
    # เปลี่ยนเมื่อ CHOCH ยืนยัน — ไม่ใช่ EMA lag
    #
    # ตามภาพ user:
    #   Left oval:  Bias=BUY  → เฉพาะ LONG signals
    #   Right oval: Bias=SELL → เฉพาะ SHORT signals
    #   "Momentum ขึ้น Bias sell" = bounce → block LONG ไม่สนใจ
    #
    # Bypass: CHOCH + DSF = structural level trades (ไม่ขึ้นกับ Bias)
    # Bias bypass: structural zone entries ไม่ขึ้นกับ Bias
    # demand_long = เข้าที่ Demand zone = valid LONG ไม่ว่า Bias จะเป็นอะไร
    # supply_zone = เข้าที่ Supply zone = valid SHORT ไม่ว่า Bias จะเป็นอะไร
    _bias_bypass_strats = {
        "choch_long", "choch_short",   # structure change ยืนยัน
        "dsf_long", "dsf_short",        # zone flip ยืนยัน
        "demand_long",                  # at Demand zone = structural LONG (bypass Bias SHORT)
        "supply_zone",                  # at Supply zone = structural SHORT (bypass Bias LONG)
        "zone_retest_long", "zone_retest_short",
        "zone_first_touch_long", "zone_first_touch_short",  # first touch = structural
        "srf_long", "srf_short",
    }
    if final_direction and not vetoed and _m15_struct_bias != "NEUTRAL":
        _strat_bias = setup.get("strategy", "") if setup else ""
        if _strat_bias not in _bias_bypass_strats:
            if _m15_struct_bias == "SHORT" and final_direction == "long":
                vetoed = True
                if setup:
                    setup["reason"] += f" [VETO: M15-Struct=SHORT → block LONG (Bias sell)]"
                final_direction = None
            elif _m15_struct_bias == "LONG" and final_direction == "short":
                vetoed = True
                if setup:
                    setup["reason"] += f" [VETO: M15-Struct=LONG → block SHORT (Bias buy)]"
                final_direction = None

    # ── H1 Strong Candle Veto: ห้ามสวน H1 candle body ใหญ่ ─────────────────
    # ถ้า H1 bar ล่าสุดที่ปิดแล้ว มี body ≥ 1.5x H1 ATR → ราคากำลัง momentum
    # ห้ามเข้าสวน แม้อยู่ใน extreme zone เพราะราคาอาจวิ่งต่อ
    # ─────────────────────────────────────────────────────────────────────
    if final_direction and not vetoed:
        try:
            if len(h1_proxy) >= 15:
                h1_closes = h1_proxy["close"].values[-15:]
                h1_highs  = h1_proxy["high"].values[-15:]
                h1_lows   = h1_proxy["low"].values[-15:]
                # ATR(14) on H1
                h1_tr = []
                for i in range(1, len(h1_closes)):
                    tr = max(
                        float(h1_highs[i]) - float(h1_lows[i]),
                        abs(float(h1_highs[i]) - float(h1_closes[i-1])),
                        abs(float(h1_lows[i])  - float(h1_closes[i-1]))
                    )
                    h1_tr.append(tr)
                h1_atr = sum(h1_tr) / len(h1_tr) if h1_tr else 1.0
                # Last completed H1 bar body
                last_h1_open  = float(h1_proxy["open"].iloc[-2])
                last_h1_close = float(h1_proxy["close"].iloc[-2])
                h1_body = last_h1_close - last_h1_open  # positive=bullish, negative=bearish
                h1_body_abs = abs(h1_body)
                if h1_body_abs >= 1.5 * h1_atr:
                    h1_is_bearish = h1_body < 0
                    h1_is_bullish = h1_body > 0
                    if final_direction == "long" and h1_is_bearish:
                        vetoed = True
                        if setup:
                            setup["reason"] += f" [VETO: H1 Strong Bear body={h1_body:.1f} vs ATR={h1_atr:.1f}]"
                        final_direction = None
                    elif final_direction == "short" and h1_is_bullish:
                        vetoed = True
                        if setup:
                            setup["reason"] += f" [VETO: H1 Strong Bull body={h1_body:.1f} vs ATR={h1_atr:.1f}]"
                        final_direction = None
        except Exception:
            pass

    # (Zone Flip Override ย้ายไปอยู่หลัง H1 Range Position Filter แล้ว — line ~909)

    # ── Anti-Conflict Guard: ห้ามเปิด position ขัดกันในสัญลักษณ์เดียว ──
    # bull_engulf BUY + dsf_short SELL พร้อมกัน = Hedge ตัวเอง = ขาดทุนสองทาง
    # ป้องกัน: ถ้ามี SELL อยู่แล้ว → ห้าม BUY, ถ้ามี BUY อยู่แล้ว → ห้าม SELL
    # Zone strategies ที่ผ่าน zone position veto → อนุญาต flip ที่ main loop
    # + Reversal strategies (choch/breakout/srf) → flip ได้ถ้า M5 momentum ยืนยัน
    #   (จาก user feedback: top @ 4592 choch_short ถูกบล็อกเพราะมี BUY 2 ตัว)
    _m5_confirms_flip = (
        (final_direction == "short" and _m5_momentum == "DOWN") or
        (final_direction == "long"  and _m5_momentum == "UP")
    )
    _reversal_flip_eligible = _m5_confirms_flip and _zone_strat in (
        "choch_long", "choch_short",
        "breakout_long", "breakout_short",
        "srf_long", "srf_short",
    )
    _is_flip_eligible = _reversal_flip_eligible or _zone_strat in (
        "zone_retest_long", "zone_retest_short",
        "zone_first_touch_long", "zone_first_touch_short",
    )
    if final_direction and not vetoed:
        try:
            existing_pos = mt5.positions_get(symbol=bridge.symbol)
            if existing_pos:
                conflict_type = 1 if final_direction == "long" else 0
                conflicting = [p for p in existing_pos if p.type == conflict_type]
                if conflicting:
                    if _is_flip_eligible:
                        # Zone strategy → ปล่อยผ่าน ให้ main loop flip
                        if setup:
                            setup["reason"] += f" [FLIP-ELIGIBLE: มี {len(conflicting)} pos ตรงข้าม → main loop จะ flip]"
                    else:
                        opp = "SELL" if final_direction == "long" else "BUY"
                        vetoed = True
                        if setup:
                            setup["reason"] += f" [VETO: มี {opp} {len(conflicting)} pos เปิดอยู่]"
                        final_direction = None
        except Exception:
            pass

    # ── M5 Momentum Guard: ห้าม weak reversal entry สวน M5 momentum ──────
    # จาก user feedback (6/1 20:39-40): bot BUY ที่ SRF ในขาลง = จับมีดตก
    #   เดิม M1-ENGULF-OVERRIDE ปล่อย engulfing เดี่ยว → knife! (เอาออก)
    #   + demand_long/srf ไม่อยู่ใน list → bypass guard (เพิ่มเข้า)
    # block "weak" reversal (แค่อยู่ที่โซน) สวน momentum:
    #   LONG (M5 DOWN): zone_retest/first_touch/demand/srf/srf_bounce
    #   SHORT (M5 UP):  เช่นเดียวกัน ฝั่งตรงข้าม
    # ยกเว้น: pinbar/star = strong reversal pattern (ไส้ปฏิเสธ/3-แท่ง) → จับ reversal จริงได้
    _weak_rev_long = {"zone_first_touch_long", "zone_retest_long", "demand_long",
                      "srf_long", "srf_bounce_long"}
    _weak_rev_short = {"zone_first_touch_short", "zone_retest_short", "supply_zone",
                       "srf_short", "srf_bounce_short"}
    if final_direction and not vetoed and _m5_momentum != "NONE":
        _is_zone_entry = setup.get("strategy", "") if setup else ""
        _zone_counter_momentum = (
            (final_direction == "long" and _m5_momentum == "DOWN" and
             _is_zone_entry in _weak_rev_long) or
            (final_direction == "short" and _m5_momentum == "UP" and
             _is_zone_entry in _weak_rev_short)
        )
        if _zone_counter_momentum:
            vetoed = True
            if setup:
                setup["reason"] += f" [VETO: M5 momentum={_m5_momentum} สวน {final_direction.upper()} (จับมีดตก) — รอ pinbar/star ยืนยัน]"
            final_direction = None

    # ── H1 CHOCH / Bias Detection ──────────────────────────────────────────
    # ตรวจ "ตำแหน่งราคาใน range ของ H1" — approach นี้ robust ที่สุด
    # ไม่ต้องการ swing detection ที่ซับซ้อน
    #
    # หลักการ:
    #   ถ้าราคาอยู่ในครึ่งบนของ range 72h  → market อยู่ใน BULLISH territory
    #   ถ้าราคาอยู่ในครึ่งล่างของ range 72h → market อยู่ใน BEARISH territory
    #
    # ตัวอย่างจาก user's chart (May 22 01:42):
    #   Low 72h = ~4453 (bottom May 20)
    #   High 72h = ~4589 (peak May 19)
    #   Range = 136 pts | Current = 4529
    #   Position = (4529-4453)/136 = 0.56 = upper 56% → BULLISH BIAS
    #
    # ผล: SHORT ถูกลด lot ลง 50% เพราะขัดกับ market structure
    # ─────────────────────────────────────────────────────────────────────
    h1_choch_bullish = False
    h1_choch_bearish = False
    h1_range_position = 0.5   # default = midpoint (neutral)
    try:
        h1_bars = min(len(h1_proxy), 72)   # ใช้ 72 H1 bars = 3 วัน
        if h1_bars >= 12:
            h1_low   = float(h1_proxy["low"].tail(h1_bars).min())
            h1_high  = float(h1_proxy["high"].tail(h1_bars).max())
            h1_range = h1_high - h1_low
            if h1_range > 0:
                cur_price = tick.get("bid", float(h1_proxy["close"].iloc[-1]))
                h1_range_position = (cur_price - h1_low) / h1_range

                # Upper 55%+ of range = bullish territory (recovered from bottom)
                if h1_range_position >= 0.55:
                    h1_choch_bullish = True
                # Lower 45%- of range = bearish territory (sold off from top)
                elif h1_range_position <= 0.45:
                    h1_choch_bearish = True
    except Exception:
        pass

    # Lot sizing: Regime × Structure (tiered)
    lot = 0.0
    structure_score = 0
    if final_direction:
        structure_score = structure["long_score"] if final_direction == "long" else structure["short_score"]
        base = 0.02 if regime == 1 else 0.01  # AGI regime tier
        # Structure multiplier (oracle: 50+ = clear edge, 70+ = strong)
        if structure_score < 30:    struct_mult = 1.0
        elif structure_score < 50:  struct_mult = 1.5
        elif structure_score < 70:  struct_mult = 2.0
        else:                       struct_mult = 3.0

        # ── Zone Width multiplier — narrow zone = SL แน่น = risk น้อย → lot ใหญ่ได้ ──
        # ผู้ใช้: "หากมั่นใจโซนแคบเราสามารถบีบโซนให้เล็กแล้วออก lot ใหญ่ได้"
        zone_mult = 1.0
        if setup:
            zh = setup.get("zone_hi"); zl = setup.get("zone_lo")
            atr_now = float(atr[-1]) if atr is not None and len(atr) > 0 else 0
            if zh is not None and zl is not None and atr_now > 0:
                zone_width_atr = abs(zh - zl) / atr_now
                if zone_width_atr < 0.3:    zone_mult = 2.5  # zone แคบมาก (มั่นใจสูง)
                elif zone_width_atr < 0.5:  zone_mult = 2.0  # zone แคบ
                elif zone_width_atr < 1.0:  zone_mult = 1.5  # zone ปานกลาง
                else:                       zone_mult = 1.0  # zone กว้าง

        lot = round(round(base * struct_mult * zone_mult / 0.01) * 0.01, 2)
        # Safety cap (กัน lot ใหญ่เกินไป)
        lot = min(lot, 0.05)

        # ── Lot scaling:
        # base × struct_mult (1.0-3.0) × zone_mult (1.0-2.5)
        # narrow zone + high struct → up to 0.05 lot
        # wide zone + low struct → 0.01 lot

    # ── Zone-aware SL override ───────────────────────────────────────────
    # ถ้า strategy เป็น momentum (EMA-based zone) แต่ราคาอยู่ใกล้ Demand/Supply zone
    # ให้ใช้ zone boundary เป็น SL แทน EMA
    # SMC: "SL ควรอยู่ใต้ Demand zone ไม่ใช่แค่ใต้ EMA"
    _out_zone_hi = setup.get("zone_hi") if setup else None
    _out_zone_lo = setup.get("zone_lo") if setup else None

    if setup and final_direction and not vetoed:
        _momentum_strats = {"momentum_long", "momentum_short", "bull_engulf", "bear_engulf",
                            "fvg_long", "fvg_short", "breakout_long", "breakout_short"}
        _strat_out = setup.get("strategy", "")
        _cur_bid = float(tick.get("bid", 0) if isinstance(tick, dict) else
                         (tick.bid if tick else 0))
        _zone_override_tol = float(atr[-1]) * 1.5  # ถ้า zone ห่างไม่เกิน 1.5 ATR

        if _strat_out in _momentum_strats:
            if final_direction == "long":
                # หา Demand zone ใกล้สุดที่อยู่ใต้ราคา → ใช้ zone_lo เป็น SL
                best_demand = None
                for z in _all_zones:
                    if z["type"] == "DEMAND" and z["hi"] <= _cur_bid + _zone_override_tol:
                        if best_demand is None or z["hi"] > best_demand["hi"]:
                            best_demand = z
                if best_demand is not None:
                    _out_zone_lo = best_demand["lo"]   # SL = below demand zone_lo
                    _out_zone_hi = best_demand["hi"]
                    if setup:
                        setup["reason"] += f" [SL→DemandZone {best_demand['lo']:.0f}-{best_demand['hi']:.0f}]"
            elif final_direction == "short":
                # หา Supply zone ใกล้สุดที่อยู่เหนือราคา → ใช้ zone_hi เป็น SL
                best_supply = None
                for z in _all_zones:
                    if z["type"] == "SUPPLY" and z["lo"] >= _cur_bid - _zone_override_tol:
                        if best_supply is None or z["lo"] < best_supply["lo"]:
                            best_supply = z
                if best_supply is not None:
                    _out_zone_hi = best_supply["hi"]   # SL = above supply zone_hi
                    _out_zone_lo = best_supply["lo"]
                    if setup:
                        setup["reason"] += f" [SL→SupplyZone {best_supply['lo']:.0f}-{best_supply['hi']:.0f}]"

    # ═══ CLEAN DECISION OVERRIDE: rule เดียวที่สอดคล้อง (แทน 26 guards) ═══
    # user request: "rule-based ที่ไม่ขัดกัน ไม่ใช่อะไรก็ veto หมด"
    # ใช้ผลของ Clean Decision เป็นตัวตัดสินสุดท้าย (override veto chain ทั้งหมด)
    # ยกเว้น weekend (hard safety — ไม่ override)
    if setup and _orig_direction and not _is_weekend:
        try:
            _price_now = float(tick.get("bid", 0) if isinstance(tick, dict) else (tick.bid if tick else 0))
            _atr_ref = float(setup.get("_det_atr", atr[-1]))
            # ตำแหน่งราคาใน local M5 range (1.5h): 0=ก้น range, 1=ยอด range
            # 2026-06-06 user: chart 4340 ใกล้ supply 4346 — แต่ 36-bar range รวม high 4356 → 4340=36% ผิด
            # ลด 36→18 bars (1.5h) ให้ตอบสนอง local edges ถูกต้อง
            _m5_range_pos = 0.5
            try:
                _rng_n = 18
                _rh = float(df_m5["high"].iloc[-_rng_n:].max())
                _rl = float(df_m5["low"].iloc[-_rng_n:].min())
                if _rh > _rl and _price_now > 0:
                    _m5_range_pos = max(0.0, min(1.0, (_price_now - _rl) / (_rh - _rl)))
            except Exception:
                pass
            # 2026-06-17: macro range 30 M5 bars (2.5h = range ที่ user ตี) — robust กว่า 18-bar (ไม่โดน spike หลอก)
            _macro_rng_pos = 0.5
            try:
                _ar_src = df_m1 if _AR_TF == "M1" else df_m5
                _mrh = float(_ar_src["high"].iloc[-_AR_BARS:].max())
                _mrl = float(_ar_src["low"].iloc[-_AR_BARS:].min())
                if _mrh > _mrl and _price_now > 0:
                    _macro_rng_pos = max(0.0, min(1.0, (_price_now - _mrl) / (_mrh - _mrl)))
            except Exception:
                pass
            # Over-extension: ราคาห่าง M5 EMA21 กี่ ATR (+ = ใต้ EMA = ยืดลง, - = เหนือ = ยืดขึ้น)
            # ยืดเกิน 2 ATR = climax/oversold-overbought → เด้งกลับ → อย่าไล่เข้า
            _m5_ext = 0.0
            try:
                _cl = df_m5["close"].values
                if len(_cl) >= 26 and _atr_ref > 0:
                    _k = 2 / (21 + 1); _e = float(_cl[-26])
                    for _v in _cl[-26:]:
                        _e = float(_v) * _k + _e * (1 - _k)
                    _m5_ext = (_e - _price_now) / _atr_ref
            except Exception:
                pass
            # LTF (M5/M15) = ตัดสินทิศ, HTF (H1/H4) = TP target
            _sr_levels_ck = _sr_levels   # ใช้ที่คำนวณไว้แล้ว (ไม่ซ้ำ)
            # M5 fresh peak/dip (5 closed bars ≈ 25 min) — กัน buy ยอดเพิ่งปั่น / sell ก้นเพิ่งดิ่ง
            _m5_rh5 = None; _m5_rl5 = None
            _m5_last2_dir = None
            try:
                if len(df_m5) >= 6:
                    _m5_rh5 = float(df_m5["high"].iloc[-6:-1].max())
                    _m5_rl5 = float(df_m5["low"].iloc[-6:-1].min())
                # 2 M5 candles ล่าสุด (closed) — body in ATR units (+green / -red)
                if len(df_m5) >= 3 and _atr_ref > 0:
                    _b1 = (float(df_m5["close"].iloc[-3]) - float(df_m5["open"].iloc[-3])) / _atr_ref
                    _b2 = (float(df_m5["close"].iloc[-2]) - float(df_m5["open"].iloc[-2])) / _atr_ref
                    _m5_last2_dir = (_b1, _b2)
            except Exception:
                pass
            # M1 momentum: นับ higher/lower closes + green/red bars ใน 5 closed M1 bars
            _m1_mom_str = None
            try:
                if len(df_m1) >= 6:
                    _m1_c = df_m1["close"].values[-6:-1]   # 5 closed
                    _m1_o = df_m1["open"].values[-6:-1]
                    _higher = sum(1 for i in range(1, 5) if _m1_c[i] > _m1_c[i-1])
                    _green = sum(1 for i in range(5) if _m1_c[i] > _m1_o[i])
                    _red = sum(1 for i in range(5) if _m1_c[i] < _m1_o[i])
                    _lower = sum(1 for i in range(1, 5) if _m1_c[i] < _m1_c[i-1])
                    if _green >= 4 and _higher >= 3:
                        _m1_mom_str = "UP_STRONG"
                    elif _red >= 4 and _lower >= 3:
                        _m1_mom_str = "DOWN_STRONG"
                    else:
                        _m1_mom_str = "NEUTRAL"
            except Exception:
                pass
            # IMB direction: ตรวจแท่ง M5 ล่าสุดว่าเป็น impulse (body > 1.5 × avg prev 3 ranges)
            _m5_imb_dir = None
            try:
                if len(df_m5) >= 5:
                    _o = float(df_m5["open"].iloc[-1]); _c = float(df_m5["close"].iloc[-1])
                    _body = abs(_c - _o)
                    _prev_ranges = (df_m5["high"].iloc[-4:-1].values - df_m5["low"].iloc[-4:-1].values)
                    _avg_range = float(_prev_ranges.mean()) if len(_prev_ranges) else 0
                    if _avg_range > 0 and _body > 1.5 * _avg_range:
                        _m5_imb_dir = "UP" if _c > _o else "DOWN"
            except Exception:
                pass
            # M5 EMA structure bias (EMA8 vs EMA21)
            _m5_struct = None
            try:
                _cl = df_m5["close"].values
                if len(_cl) >= 22:
                    def _ema(vals, n):
                        k=2/(n+1); e=float(vals[-n*2])
                        for v in vals[-n*2:]: e=v*k+e*(1-k)
                        return e
                    _e8 = _ema(_cl, 8); _e21 = _ema(_cl, 21)
                    _sep = (_e8 - _e21) / _atr_ref if _atr_ref > 0 else 0
                    if _sep > 0.3: _m5_struct = "BULL"
                    elif _sep < -0.3: _m5_struct = "BEAR"
                    else: _m5_struct = "FLAT"
            except Exception:
                pass
            # Context bias from swing structure (HH+HL = BULL, LH+LL = BEAR)
            _ctx_bias = "NEUTRAL"
            try:
                _ctx_bias = compute_swing_bias(df_m5, lookback=25)
            except Exception:
                pass
            # 2026-06-18 user "ตรงยอดสุดแล้วราคาลง = trend bear / ราคาผ่านยอดเก่า = trend bull ต่อ"
            # CHOCH-style flip: ราคาเปลี่ยนแรง > 2 ATR จาก 12-bar high/low → FLIP context_bias ตรงทิศ
            #   = ไม่ cancel เป็น NEUTRAL แต่ FLIP เป็น BEAR/BULL → block ฝั่งสวนเทรนใหม่ + allow ฝั่งตามเทรนใหม่
            try:
                _rh12 = float(df_m5["high"].iloc[-13:-1].max())
                _rl12 = float(df_m5["low"].iloc[-13:-1].min())
                _drop_from_high = _rh12 - _price_now
                _rally_from_low = _price_now - _rl12
                # ใช้ตัวที่แรงกว่า (ราคาเพิ่งทำอะไร)
                if _drop_from_high > 2.0 * _atr_ref and _drop_from_high > _rally_from_low:
                    _ctx_bias = "BEAR"   # CHOCH bear: ยอด + drop = bear trend ใหม่
                elif _rally_from_low > 2.0 * _atr_ref and _rally_from_low > _drop_from_high:
                    _ctx_bias = "BULL"   # CHOCH bull: ก้น + rally = bull trend ใหม่
            except Exception:
                pass
            # 2026-06-18 user "Morning Star pattern": ถ้ามี bullish/bearish reversal pattern ก่อตัวที่ก้น/ยอด
            # → flip context_bias = NEUTRAL ทันที (ไม่รอ rally $13) → block weak counter เร็วขึ้น
            try:
                # Morning Star: 3 แท่ง M5 — red strong, doji/small body, green strong
                if len(df_m5) >= 4:
                    _o = df_m5["open"].values; _c = df_m5["close"].values
                    _h = df_m5["high"].values; _l = df_m5["low"].values
                    _b1 = _c[-3] - _o[-3]; _b2 = _c[-2] - _o[-2]; _b3 = _c[-1] - _o[-1]
                    _range2 = _h[-2] - _l[-2]
                    _atr_chk = _atr_ref if _atr_ref > 0 else 1.0
                    # Morning Star (bullish bottom): red big + small body + green big
                    _ms_bull = (_b1 < -0.8 * _atr_chk and abs(_b2) < 0.3 * _atr_chk and _range2 > 0
                                and _b3 > 0.8 * _atr_chk and _ctx_bias == "BEAR")
                    # Evening Star (bearish top): green big + small body + red big
                    _es_bear = (_b1 > 0.8 * _atr_chk and abs(_b2) < 0.3 * _atr_chk and _range2 > 0
                                and _b3 < -0.8 * _atr_chk and _ctx_bias == "BULL")
                    if _ms_bull:
                        _ctx_bias = "NEUTRAL"   # Morning Star → ไม่ trust BEAR อีก
                    elif _es_bear:
                        _ctx_bias = "NEUTRAL"   # Evening Star → ไม่ trust BULL อีก
                # Liq Sweep reversal: doji + long wick > 1.5 ATR + close > open (bullish) at low
                if len(df_m5) >= 2:
                    _o1 = float(df_m5["open"].iloc[-1]); _c1 = float(df_m5["close"].iloc[-1])
                    _h1 = float(df_m5["high"].iloc[-1]); _l1 = float(df_m5["low"].iloc[-1])
                    _lower_wick = min(_o1, _c1) - _l1
                    _upper_wick = _h1 - max(_o1, _c1)
                    _atr_chk = _atr_ref if _atr_ref > 0 else 1.0
                    if _lower_wick > 1.5 * _atr_chk and _ctx_bias == "BEAR":
                        _ctx_bias = "NEUTRAL"   # liq sweep bullish at low → ไม่ trust BEAR
                    elif _upper_wick > 1.5 * _atr_chk and _ctx_bias == "BULL":
                        _ctx_bias = "NEUTRAL"   # liq sweep bearish at high → ไม่ trust BULL
            except Exception:
                pass
            # 2026-06-18 pullback detection (M5 12 bars HH+HL / LH+LL) — เร็วกว่า context_bias
            _pullback_bull = False; _pullback_bear = False
            try:
                _h12 = df_m5["high"].iloc[-13:-1].values
                _l12 = df_m5["low"].iloc[-13:-1].values
                _sh = []; _sl = []
                for _i in range(2, len(_h12)-2):
                    if _h12[_i] >= max(_h12[_i-2:_i+3]): _sh.append(float(_h12[_i]))
                    if _l12[_i] <= min(_l12[_i-2:_i+3]): _sl.append(float(_l12[_i]))
                if len(_sh) >= 2 and len(_sl) >= 2:
                    _hh = _sh[-1] > _sh[-2]; _hl = _sl[-1] > _sl[-2]
                    _lh = _sh[-1] < _sh[-2]; _ll = _sl[-1] < _sl[-2]
                    _rh13 = float(df_m5["high"].iloc[-13:].max())
                    _rl13 = float(df_m5["low"].iloc[-13:].min())
                    if _hh and _hl and (_rh13 - _price_now) > 0.3 * _atr_ref:
                        _pullback_bull = True
                    if _lh and _ll and (_price_now - _rl13) > 0.3 * _atr_ref:
                        _pullback_bear = True
            except Exception:
                pass
            _ck_ok, _ck_reason, _ck_tp = _clean_entry_decision(
                setup, _orig_direction, _price_now, list(_all_zones), list(_pre_htf_zones),
                _m5_momentum, _atr_ref, _m15_bullish, _m5_range_pos, _sr_levels_ck, _m5_ext,
                _m15_regime, _m5_rh5, _m5_rl5, _m5_last2_dir, _m1_mom_str,
                _m5_imb_dir, _m5_struct, _ctx_bias, _macro_rng_pos,
                _pullback_bull, _pullback_bear)
            if _ck_ok:
                final_direction = _orig_direction
                vetoed = False
                setup["reason"] += f" [✅ CLEAN-PASS: {_ck_reason}]"
                # opposing zone = TP target (เก็บยาวถึงโซนตรงข้าม)
                if _ck_tp is not None:
                    htf_tp_cap = _ck_tp
            else:
                final_direction = None
                vetoed = True
                setup["reason"] += f" [🚫 CLEAN-BLOCK: {_ck_reason}]"
        except Exception as _e:
            print(f"   [CLEAN-DECISION-ERR] {_e}", flush=True)

    return {
        "decision_time": dt.datetime.now().isoformat(timespec="seconds"),
        "live_bid": tick.get("bid", 0) if isinstance(tick, dict) else (tick.bid if tick else 0),
        "live_ask": tick.get("ask", 0) if isinstance(tick, dict) else (tick.ask if tick else 0),
        "strategy": setup["strategy"] if setup else "none",
        "raw_direction": setup["direction"] if setup else "NONE",
        "agi_regime": regime,
        "vetoed": vetoed,
        "final_action": (final_direction.upper() if final_direction else "HOLD"),
        "lot": lot,
        # regime สำหรับ auto-tuner + Gate 2: chop / trend_bull / trend_bear (slope + EMA)
        "regime": _m15_regime,
        "atr": (setup.get("_det_atr", float(atr[-1])) if setup else float(atr[-1])),
        "mt_long": mt_long,
        "mt_short": mt_short,
        "structure_long": structure["long_score"],
        "structure_short": structure["short_score"],
        "structure_verdict": structure["verdict"],
        "reason": setup["reason"] if setup else "no setup",
        "h1_choch": ("BULLISH" if h1_choch_bullish else "BEARISH" if h1_choch_bearish else "NONE"),
        "m15_trend": _m15_trend_label,
        "m5_reversal": _m5_reversal_entry,
        "zone_hi": _out_zone_hi,
        "zone_lo": _out_zone_lo,
        "htf_tp_cap": (float(_htf_opposing_zone["lo"]) if final_direction == "long"
                       and '_htf_opposing_zone' in dir() and _htf_opposing_zone
                       else (float(_htf_opposing_zone["hi"]) if final_direction == "short"
                             and '_htf_opposing_zone' in dir() and _htf_opposing_zone
                             else None)),
        "htf_zone_info": _htf_opposing_zone if '_htf_opposing_zone' in dir() else None,
    }


def has_open_position(symbol, magic):
    p = mt5.positions_get(symbol=symbol)
    return p is not None and any(x.magic == magic for x in p)


def has_conflicting_position(symbol, direction):
    """ห้ามเปิด LONG ถ้ามี SHORT อยู่แล้ว และกลับกัน — ป้องกัน position ขัดกัน"""
    p = mt5.positions_get(symbol=symbol)
    if not p:
        return False
    # direction="long" → ต้องการ BUY (type=0) → ห้ามถ้ามี SELL (type=1)
    # direction="short" → ต้องการ SELL (type=1) → ห้ามถ้ามี BUY (type=0)
    conflict_type = 1 if direction == "long" else 0
    return any(x.type == conflict_type for x in p)


def flip_losing_positions(symbol, new_direction, dec):
    """Smart Conflict Resolution: ปิด position ฝั่งตรงข้ามเมื่อสัญญาณใหม่แข็งแรง

    เงื่อนไข FLIP:
    Mode A (ปกติ): struct >= 50 + H1 CHOCH match → ปิดเฉพาะ losing
    Mode B (zone strategy): zone_retest/zone_first_touch → ปิดทั้งหมด (take profit + flip)
      - ไม่ต้องการ CHOCH match เพราะ zone signal มีคุณภาพสูงอยู่แล้ว
      - ปิดทั้ง winning (= take profit ที่ zone) และ losing (= cut loss)

    Return: True ถ้าปิดได้หมดแล้ว (ไม่มี conflict เหลือ), False ถ้ายัง conflict อยู่
    """
    strategy = dec.get("strategy", "")
    # zone + reversal strategies → flip mode (ปิดทั้งหมด)
    # reversal (choch/breakout/srf) ผ่าน anti-conflict guard มาแล้ว = M5 momentum ยืนยัน
    # 2026-06-06 user: BUY ที่ retest หลัง breakout บล็อกเพราะ conflict — ขยาย flip set
    _is_zone_flip = strategy in (
        "zone_retest_long", "zone_retest_short",
        "zone_first_touch_long", "zone_first_touch_short",
        "choch_long", "choch_short",
        "breakout_long", "breakout_short",
        "srf_long", "srf_short",
        # REVERSAL_AT_WALL (strong candle confirmation) → flip ได้
        "pinbar_long", "pinbar_short", "star_long", "star_short",
        "bull_engulf", "bear_engulf",
        # 2026-06-18 user: SUPPLY/DEMAND REJECTION = strongest reversal signal → flip ทันที
        "supply_rejection_short", "demand_rejection_long",
        "srf_bounce_long", "srf_bounce_short",
        "trendline_long", "trendline_short",
        "hl_retest_long", "hl_retest_short",
    )
    # REVERSAL_AT_WALL = strong pattern, ลด struct requirement (เดิม 40)
    # 2026-06-09: ลบ trendline (loser, ไม่ใช่ candle pattern จริง)
    _is_strong_reversal = strategy in (
        "pinbar_long", "pinbar_short", "star_long", "star_short",
        "bull_engulf", "bear_engulf",
    )

    if new_direction == "long":
        struct_score = dec.get("structure_long", 0)
    else:
        struct_score = dec.get("structure_short", 0)

    h1_choch = dec.get("h1_choch", "NONE")

    if _is_zone_flip:
        # ── Mode B: Zone Strategy Flip ──
        # REVERSAL_AT_WALL (strong pattern): struct >= 20, อื่นๆ: 40
        _min_struct = 20 if _is_strong_reversal else 40
        if struct_score < _min_struct:
            return False
    else:
        # ── Mode A: ปกติ ──
        if struct_score < 50:
            return False
        if new_direction == "long" and h1_choch != "BULLISH":
            return False
        if new_direction == "short" and h1_choch != "BEARISH":
            return False

    # ── หา position ฝั่งตรงข้าม ──
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return True

    conflict_type = 1 if new_direction == "long" else 0  # SELL=1, BUY=0
    conflict_positions = [p for p in positions if p.type == conflict_type]

    if not conflict_positions:
        return True

    if _is_zone_flip:
        # Zone strategy: ปิดทุก position ตรงข้าม (winning = take profit ที่ zone)
        to_close = conflict_positions
    else:
        # ปกติ: ปิดเฉพาะ losing
        losing = [p for p in conflict_positions if p.profit < 0]
        winning = [p for p in conflict_positions if p.profit >= 0]
        if not losing:
            return len(winning) == 0
        to_close = losing

    # ── ปิด positions ──
    closed_count = 0
    for pos in to_close:
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            continue

        # close SELL = BUY, close BUY = SELL
        if pos.type == 1:  # SELL → close ด้วย BUY
            close_type = mt5.ORDER_TYPE_BUY
            close_price = tick.ask
        else:  # BUY → close ด้วย SELL
            close_type = mt5.ORDER_TYPE_SELL
            close_price = tick.bid

        close_req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": pos.volume,
            "type": close_type,
            "position": pos.ticket,
            "price": close_price,
            "deviation": 30,
            "magic": pos.magic,
            "comment": f"FLIP:{dec.get('strategy','?')}_{new_direction}"[:29],  # MT5 max 29 ตัว
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        result = mt5.order_send(close_req)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            opp_dir = "SELL" if pos.type == 1 else "BUY"
            print(f"[FLIP] Closed losing {opp_dir} #{pos.ticket} ({pos.profit:+.2f}$) "
                  f"→ new {new_direction.upper()} signal (struct={struct_score}, CHOCH={h1_choch})",
                  flush=True)
            closed_count += 1
        else:
            err = result.retcode if result else "None"
            print(f"[FLIP-FAIL] #{pos.ticket}: retcode={err}", flush=True)

    # เช็คว่ายังเหลือ conflict ไหม
    remaining = mt5.positions_get(symbol=symbol)
    if remaining:
        still_conflict = [p for p in remaining if p.type == conflict_type]
        return len(still_conflict) == 0
    return True


def submit_order(symbol, direction, lot, atr, magic, comment="",
                 sl_atr=1.5, tp_atr=4.5,
                 zone_hi=None, zone_lo=None, target_rr=None,
                 max_lot=0.05, counter_trend=False,
                 htf_tp_cap=None):
    """ส่งออเดอร์ — Zone-based SL + Structure-aware TP:

    SL: zone boundary + buffer (หรือ ATR fallback)
    TP: คำนวณจาก RR ratio แล้ว clamp ด้วย S/R levels + H1 range
      1) คำนวณ TP ตาม Dynamic RR (zone width)
      2) Counter-trend → cap TP ที่ 1.5x SL (เก็บสั้น ไม่เก็บยาว)
      3) หาแนว S/R ที่อยู่ระหว่าง entry→TP → ดึง TP เข้ามาก่อนแนว S/R ($1 buffer)
      4) TP ห้ามไกลเกิน H1 72h range boundary (High/Low)
      5) TP ขั้นต่ำ = 1.5x SL dist (RR 1:1.5) — ถ้า S/R บีบจน TP < 1.5x → ใช้ 1.5x
    """
    tick = mt5.symbol_info_tick(symbol)
    info = mt5.symbol_info(symbol)
    if tick is None or info is None:
        return {"ok": False, "error": "no tick"}

    # ── Symbol-type detection ────────────────────────────────────────────
    # Metal/Index (XAU, XAG, oil): price > 10 → ใช้ค่า original ที่ tune แล้ว
    # Forex (EUR/USD, GBP/USD...): price ≤ 10 → ใช้ ATR-relative ทุกค่า
    _cur_bid = float(mt5.symbol_info_tick(symbol).bid) if mt5.symbol_info_tick(symbol) else 0
    _is_forex = _cur_bid < 10.0   # EUR=1.16, GBP=1.26 vs XAU=4400, Oil=75

    SL_BUFFER = 0.3 * atr   # buffer (ATR-relative ทำงานดีทั้งสอง)
    MIN_SL = 0.5 * atr
    # ── MIN_SL floor อิง "noise จริงของตลาด" (M5 ATR) ไม่ใช่แค่ entry-TF ──
    # user (00:14): entry M1 ATR ต่ำ → SL จิ๋ว $1.5 → โดน M5 noise ($4.2 ATR) เด้งปกติใน 2 นาที
    # → floor = 0.8× M5 ATR (กัน noise) + absolute $1.5 (กัน spread)
    _m5_atr_noise = 0.0
    try:
        _m5b = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 16)
        if _m5b is not None and len(_m5b) >= 15:
            _trs = [max(float(_m5b[i]['high'])-float(_m5b[i]['low']),
                        abs(float(_m5b[i]['high'])-float(_m5b[i-1]['close'])),
                        abs(float(_m5b[i]['low'])-float(_m5b[i-1]['close']))) for i in range(1, 15)]
            _m5_atr_noise = 0.8 * (sum(_trs) / len(_trs))
    except Exception:
        pass
    if not _is_forex:
        MIN_SL = max(MIN_SL, 1.5, _m5_atr_noise)
    # MAX_SL: Symbol-aware cap (2026-06-05 user: "SL กว้างไป" — tighten 2.5→2.0, $15→$10)
    #   XAU/Metals (price > 10): 2.0 ATR — โซนใกล้กว่า ไม่ต้อง $15 room
    #   Forex (EUR/USD etc, price ≤ 10): 1.5 ATR
    MAX_SL = 1.5 * atr if _is_forex else 2.0 * atr

    # ── ATR Spike Protection: absolute SL ceiling ────────────────────────
    # 2026-06-05 user: "ออเดอร์ล่าสุด sl กว้างไป" — ลด $15→$10 (climax ATR $7 ให้ $10 cap)
    MAX_SL_ABS = 10.0   # USD, XAU only — was $15 before climax tightening
    if not _is_forex:
        MAX_SL = min(MAX_SL, MAX_SL_ABS)

    # ── Night/Quiet Session Scaling ──────────────────────────────
    # ช่วงกลางคืน (01:00-08:00 ICT) ทองไม่ค่อยวิ่ง range แคบ
    # → TP/SL เก็บสั้น เพื่อเก็บ TP ให้ได้จริงในช่วง low volatility
    # (จากข้อมูล: 5/29 overnight range $13, แต่ TP $3-8 → 5 BUY SL ติด)
    _now_h = dt.datetime.now().hour
    _is_night = (1 <= _now_h < 8) and not _is_forex
    if _is_night:
        # night tighter แต่มี floor $3 (กัน low-ATR บีบ SL จนจิ๋วกว่า MIN_SL)
        MAX_SL = min(MAX_SL, max(1.5 * atr, 3.0))

    # ── TF-aware RR: TP อิง ATR ของ TF ที่เข้า ─────────────────────────────
    # user framework: เข้า TF ไหน TP ตาม TF นั้น
    #   M1 ATR ~$1-2 → RR 2-3 → TP $3-8
    #   M5 ATR ~$2-5 → RR 2-3 → TP $6-15
    #   M15 ATR ~$5-10 → RR 2-3 → TP $15-40 (ภาพ M15 ที่ user ส่ง)
    # TP = 3× ATR ของ TF นั้น (เหมาะสมกับการวิ่งของแต่ละ TF)
    if target_rr is None:
        # คำนวณ TP จาก 3× ATR ของ TF ที่เข้า แทน RR × SL_dist
        # (atr คือ _det_atr ที่ส่งมาจาก TF จริง)
        _tf_tp_target = 3.0 * atr   # TP = 3 ATR ของ TF นั้น
        target_rr = 3.0            # fallback RR (จะถูก override โดย Zone Extension)
    # 2026-06-18 user "01:00 SELL @4367 TP เร็ว เก็บ $4 แต่ราคาวิ่งต่อ $80": ขยาย TP สำหรับ vertical move patterns
    #   momentum_breakout + consolidation_breakout = pattern ที่ราคาวิ่งไกล → trailing SL ตามล็อก ride trend ได้
    _vertical = (comment or "").replace("Spec_", "").lower()
    if any(_vertical.startswith(p) for p in ("momentum_breakout", "consolidation_breakout")):
        target_rr = max(target_rr, 7.0)   # TP กว้าง 7× SL (~$30+) ให้ trailing SL ทำงาน ride trend dump/rally

    # Night session: cap RR 2.0 (ก่อนหน้า 1.3 ทำให้ TP จิ๋วเกิน — user framework min 1:2)
    # 2026-06-18: vertical patterns (momentum/cons_breakout) ไม่ cap night → ride trend dump/rally
    _vertical_pat = any(_vertical.startswith(p) for p in ("momentum_breakout", "consolidation_breakout"))
    if _is_night and not _vertical_pat:
        target_rr = min(target_rr, 2.0)
        print(f"   [NIGHT-SESSION] h={_now_h}:xx → RR capped 2.0, MIN_SL $1.5, MAX_SL floor $3", flush=True)

    if direction == "long":
        otype = mt5.ORDER_TYPE_BUY
        price = tick.ask
        # Sanity check zone_lo — ถ้าไกลกว่า 10×MAX_SL → ใช้ ATR fallback
        _zone_lo_ok = (zone_lo is not None and
                       abs(price - zone_lo) <= MAX_SL * 10 and
                       zone_lo > price * 0.5)  # ต้องไม่ต่ำกว่าครึ่งราคา
        if _zone_lo_ok:
            sl = zone_lo - SL_BUFFER
        else:
            if zone_lo is not None:
                print(f"   [SL-SANITY] zone_lo={zone_lo:.5f} too far from price={price:.5f} → ATR fallback", flush=True)
            sl = price - sl_atr * atr
        sl_dist = price - sl
        sl_dist = max(sl_dist, MIN_SL)
        sl_dist = min(sl_dist, MAX_SL)
        sl_dist = min(sl_dist, max(2.0 * atr, MIN_SL))  # 2026-06-05 บีบ 2.5→2.0×ATR (user: SL กว้างไป)
        sl = price - sl_dist
        # TF-aware TP: ใช้ 3×ATR ของ TF ที่เข้า (หรือ RR×SL ถ้าใหญ่กว่า)
        _tp_by_tf = price + 3.0 * atr
        _tp_by_rr = price + sl_dist * target_rr
        raw_tp = max(_tp_by_tf, _tp_by_rr)   # เอาที่ไกลกว่า (แต่จะถูก cap โดย Zone Extension)
    else:
        otype = mt5.ORDER_TYPE_SELL
        price = tick.bid
        # Sanity check zone_hi
        _zone_hi_ok = (zone_hi is not None and
                       abs(zone_hi - price) <= MAX_SL * 10 and
                       zone_hi < price * 2.0)
        if _zone_hi_ok:
            sl = zone_hi + SL_BUFFER
        else:
            if zone_hi is not None:
                print(f"   [SL-SANITY] zone_hi={zone_hi:.5f} too far from price={price:.5f} → ATR fallback", flush=True)
            sl = price + sl_atr * atr
        sl_dist = sl - price
        sl_dist = max(sl_dist, MIN_SL)
        sl_dist = min(sl_dist, MAX_SL)
        sl_dist = min(sl_dist, max(2.0 * atr, MIN_SL))  # 2026-06-05 บีบ 2.5→2.0×ATR (user: SL กว้างไป)
        sl = price + sl_dist
        _tp_by_tf = price - 3.0 * atr
        _tp_by_rr = price - sl_dist * target_rr
        raw_tp = min(_tp_by_tf, _tp_by_rr)   # เอาที่ไกลกว่า (เล็กกว่า สำหรับ short)

    # ── Volatility Lot Reduction: SL กว้าง → ลด max_lot ──────────────
    # 2026-06-10 log analysis: ATR พุ่ง $15-20 → SL $10 cap → risk $50/ไม้ (0.05 lot)
    # = 1 SL กลืน winner 2-3 ไม้. แก้: SL ≥$8 → ลด max_lot → risk ~$30/ไม้
    #   SL $10 @0.03 = -$30 | SL $8 @0.03 = -$24 (vs เดิม -$50/-$40)
    # winner ก็ลดตาม แต่ loss/win ratio ดีขึ้น (1 SL ≈ 1 winner แทน 2-3)
    _VOL_SL_THRESH = 8.0    # SL distance threshold ($ for XAU)
    _VOL_LOT_CAP = 0.03     # ลด lot จาก 0.05 → 0.03
    if not _is_forex and sl_dist >= _VOL_SL_THRESH:
        max_lot = min(max_lot, _VOL_LOT_CAP)
        print(f"   [VOL-LOT] SL ${sl_dist:.1f} ≥ ${_VOL_SL_THRESH} → max_lot capped {_VOL_LOT_CAP}", flush=True)

    # ── Risk-Based Position Sizing ────────────────────────────────────
    # user เลือก: lot ปรับอัตโนมัติให้เสี่ยงเงินคงที่ทุกไม้ (ไม่ว่า SL กว้างแค่ไหน)
    #   lot = RISK_PER_TRADE / (sl_dist × value_per_price_per_lot)
    #   SL $15 → lot ~0.033 | SL $10 → lot 0.05 (cap) | SL $5 → lot 0.05 (cap, risk<$50)
    # value_per_price = ค่าเงินต่อ $1 ราคา ต่อ 1 lot (XAU=100, EUR=100k×price)
    RISK_PER_TRADE = 50.0   # USD ต่อไม้ (user setting)
    _risk_based_lot = None
    try:
        _tick_val = float(info.trade_tick_value)
        _tick_sz  = float(info.trade_tick_size)
        _val_per_price = (_tick_val / _tick_sz) if _tick_sz > 0 else float(info.trade_contract_size)
        if _val_per_price > 0 and sl_dist > 0:
            _risk_based_lot = RISK_PER_TRADE / (sl_dist * _val_per_price)
            lot = min(_risk_based_lot, max_lot)   # cap ที่ max_lot (risk จะ < $50 ถ้า SL แคบ)
            _actual_risk = lot * sl_dist * _val_per_price
            print(f"   [RISK-SIZE] ${RISK_PER_TRADE}/trade ÷ (SL ${sl_dist:.2f} × {_val_per_price:.0f}) → lot {_risk_based_lot:.3f} → ใช้ {lot:.3f} (risk จริง ${_actual_risk:.0f})", flush=True)
    except Exception as _e:
        print(f"   [RISK-SIZE-ERR] {_e} → ใช้ lot เดิม {lot}", flush=True)

    # ── Counter-trend TP Cap: สวนเทรนเก็บสั้น RR 1:1.5 ──────────────
    # จากข้อมูล: counter-trend WR=24%, avg TP dist=$26 → ไม่ถึง TP เลย
    # แก้: cap RR ที่ 1.5 (lot ไม่เพิ่มแล้ว เพราะ risk-based sizing คุม risk อยู่)
    COUNTER_RR = 1.5
    if counter_trend and target_rr > COUNTER_RR:
        original_rr_for_counter = target_rr
        target_rr = COUNTER_RR
        if direction == "long":
            raw_tp = price + sl_dist * COUNTER_RR
        else:
            raw_tp = price - sl_dist * COUNTER_RR
        print(f"   [COUNTER-TP] สวนเทรน → cap RR {original_rr_for_counter:.1f}→{COUNTER_RR} "
              f"| TP dist ${sl_dist*COUNTER_RR:.1f} (เดิม ${sl_dist*original_rr_for_counter:.1f})",
              flush=True)

    # ── Structure-aware TP: ดึง TP เข้าถ้ามี S/R ขวาง ──────────────────
    tp = raw_tp
    tp_reason = ""
    # XAU ใช้ค่าเดิม ($1.0 ที่ tune ไว้แล้ว), EUR/Forex ใช้ ATR-relative
    SR_TP_BUFFER = 1.0 if not _is_forex else 0.2 * atr
    MIN_RR = 1.5

    try:
        sr_levels = find_sr_levels(symbol)
        if sr_levels:
            if direction == "long":
                # หา RESIST ที่อยู่ระหว่าง entry→TP (กั้นทางขึ้น)
                _sr_gap = 0.3 * atr if _is_forex else 2.0   # XAU: $2 gap เดิม
                blocking = [lvl for lbl, lvl in sr_levels
                            if lbl == "RESIST" and price + _sr_gap < lvl < raw_tp]
                if blocking:
                    nearest_resist = min(blocking)  # แนวต้านใกล้สุด
                    capped_tp = nearest_resist - SR_TP_BUFFER
                    # เช็คว่ายัง RR >= MIN_RR
                    if (capped_tp - price) >= MIN_RR * sl_dist:
                        tp = capped_tp
                        tp_reason = f" TP capped@RESIST {nearest_resist:.0f}"
            else:
                # หา SUPPORT ที่อยู่ระหว่าง entry→TP (กั้นทางลง)
                blocking = [lvl for lbl, lvl in sr_levels
                            if lbl == "SUPPORT" and raw_tp < lvl < price - _sr_gap]
                if blocking:
                    nearest_support = max(blocking)  # แนวรับใกล้สุด
                    capped_tp = nearest_support + SR_TP_BUFFER
                    if (price - capped_tp) >= MIN_RR * sl_dist:
                        tp = capped_tp
                        tp_reason = f" TP capped@SUPPORT {nearest_support:.0f}"
    except Exception:
        pass

    # ── H1 Range Boundary: TP ห้ามไกลเกินขอบ H1 72h range ──────────
    try:
        h1_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 72)
        if h1_rates is not None and len(h1_rates) >= 12:
            h1_high = max(float(b[2]) for b in h1_rates)
            h1_low  = min(float(b[3]) for b in h1_rates)
            if direction == "long" and tp > h1_high:
                capped_tp = h1_high - SR_TP_BUFFER
                if (capped_tp - price) >= MIN_RR * sl_dist:
                    tp = capped_tp
                    tp_reason = f" TP capped@H1_HIGH {h1_high:.0f}"
            elif direction == "short" and tp < h1_low:
                capped_tp = h1_low + SR_TP_BUFFER
                if (price - capped_tp) >= MIN_RR * sl_dist:
                    tp = capped_tp
                    tp_reason = f" TP capped@H1_LOW {h1_low:.0f}"
    except Exception:
        pass

    # ── HTF Zone TP Cap (H1/H4 zone เป็น ceiling/floor) ─────────────────
    # ถ้า H1/H4 Supply/Demand zone ขวางอยู่ก่อน TP → ดึง TP เข้ามา
    # ป้องกันการตั้ง TP เกิน H1/H4 zone (ซึ่ง price มักจะ reject ที่นั่น)
    # 2026-06-03 FIX: opposing zone ใกล้กว่า MIN_RR×SL → RR แย่ = ปฏิเสธไม้
    # (เดิม cap ถูกปฏิเสธ → TP ค้างไกลข้ามเพดาน เช่น buy 4471 ใต้ supply 4477 แต่ TP 4505 $168!)
    # 2026-06-09 user: "ปิดเร็วเกินไป +$17 ออเดอร์ถูกทางแต่ TP แค่ $3.51"
    # ราก: htf_tp_cap ใช้ HTF zone ใกล้ๆ บีบ TP จนเล็ก → กำไรเล็ก
    # แก้: ถ้า htf_tp_cap ใกล้กว่า 2.5×SL → IGNORE cap, ใช้ raw_tp (RR×SL) แทน
    MIN_RR_HTF = 2.5   # min TP = 2.5×SL (เดิม 1.2)
    # 2026-06-05 user: "bot ไม่ออก order เลย" — deadlock เพราะ pinbar/star ผ่าน Clean Decision
    # แต่ HTF-cap (supply $1 ใกล้) → RR-REJECT. REVERSAL_AT_WALL ออกแบบมาเพื่อ break wall = exempt
    # ใช้ comment match แบบ exact (comment = "Spec_{strategy}", truncated to 29 ตัว)
    # ── Universal RR Floor: opposing HTF zone < SL → reject (RR<1) ──────────
    # 2026-06-10 log analysis: 21:48 SHORT@4282 ห่าง H1 Demand $2.6 แต่ SL $9.91
    # → RR=0.26 (เสี่ยง $10 ได้ $2.6) = ขาดทุนแน่. TP อาจตั้งไกลกว่า zone แต่ price จะ
    # reject ที่ zone → ไม่มีทางถึง TP. บังคับทุก strategy รวม REVERSAL_AT_WALL
    if htf_tp_cap is not None:
        _htf_room = abs(htf_tp_cap - price)
        if _htf_room < sl_dist:
            print(f"   [RR-FLOOR] opposing zone {htf_tp_cap:.0f} ห่าง ${_htf_room:.1f} < SL ${sl_dist:.1f} → RR<1 ไม่คุ้ม", flush=True)
            return {"ok": False, "error": f"opposing zone {htf_tp_cap:.0f} (${_htf_room:.1f}) < SL (${sl_dist:.1f}) → RR<1"}

    _REV_PREFIXES = ("pinbar_", "star_", "bull_eng", "bear_eng", "srf_", "trendline_", "liq_sweep", "break_retest", "supply_react", "demand_react")
    _strat_raw = (comment or "").replace("Spec_", "").lower()
    _skip_htf_cap = any(_strat_raw.startswith(p) for p in _REV_PREFIXES)
    # 2026-06-09: ถ้า cap ใกล้กว่า 2.5×SL → ignore เลย, ใช้ raw_tp (กว้างกว่า)
    if htf_tp_cap is not None and not _skip_htf_cap:
        _cap_dist = abs(htf_tp_cap - price)
        if _cap_dist < MIN_RR_HTF * sl_dist:
            # cap ใกล้เกิน → ignore (ปล่อยให้ TP กว้างตาม RR)
            print(f"   [HTF-CAP-SKIP] cap @{htf_tp_cap:.1f} ห่าง ${_cap_dist:.1f} < {MIN_RR_HTF}×SL ${sl_dist:.1f} → ignore cap, ใช้ raw_tp", flush=True)
            htf_tp_cap = None
    if htf_tp_cap is not None and not _skip_htf_cap:
        if direction == "long" and htf_tp_cap < tp:
            capped = htf_tp_cap - SR_TP_BUFFER  # ATR-based buffer ก่อนถึง zone
            if (capped - price) >= MIN_RR_HTF * sl_dist:
                tp = capped
                tp_reason += f" [HTF-TP-CAP @ {htf_tp_cap:.0f}]"
                print(f"   [HTF-TP-CAP] LONG TP capped at H1/H4 Supply {htf_tp_cap:.2f} "
                      f"(raw_tp={raw_tp:.2f} → tp={tp:.2f})", flush=True)
            else:
                _room = htf_tp_cap - price
                print(f"   [RR-REJECT] LONG: supply {htf_tp_cap:.0f} ห่างแค่ ${_room:.1f} < {MIN_RR_HTF}×SL ${sl_dist:.1f} → ชนเพดาน ไม่คุ้ม", flush=True)
                return {"ok": False, "error": f"buy ใต้ supply {htf_tp_cap:.0f} (room ${_room:.1f} < {MIN_RR_HTF}xSL ${sl_dist:.1f}) RR ต่ำ"}
        elif direction == "short" and htf_tp_cap > tp:
            capped = htf_tp_cap + SR_TP_BUFFER  # ATR-based buffer
            if (price - capped) >= MIN_RR_HTF * sl_dist:
                tp = capped
                tp_reason += f" [HTF-TP-CAP @ {htf_tp_cap:.0f}]"
                print(f"   [HTF-TP-CAP] SHORT TP capped at H1/H4 Demand {htf_tp_cap:.2f} "
                      f"(raw_tp={raw_tp:.2f} → tp={tp:.2f})", flush=True)
            else:
                _room = price - htf_tp_cap
                print(f"   [RR-REJECT] SHORT: demand {htf_tp_cap:.0f} ห่างแค่ ${_room:.1f} < {MIN_RR_HTF}×SL ${sl_dist:.1f} → ไม่คุ้ม", flush=True)
                return {"ok": False, "error": f"short เหนือ demand {htf_tp_cap:.0f} (room ${_room:.1f} < {MIN_RR_HTF}xSL ${sl_dist:.1f}) RR ต่ำ"}

    # ── Zone Extension "เก็บยาว": ขยาย TP ไปถึง opposing zone จริง ──────────
    # user feedback (20:22/21:16): SL เล็ก → TP เล็กตาม (RR×SL) = เก็บสั้นเกิน
    # แก้: TP อิง "โซนเป้าหมายจริง" (absolute) ไม่ใช่ RR×SL
    #   → 2 ไม้ที่ level เดียวกัน เป้าเดียวกัน = TP เท่ากัน (เหมือน 02:28)
    # extend เฉพาะถ้า zone ไกลกว่า TP ปัจจุบัน + RR<=6 + dist<=$50 (กันไกลเกิน)
    if not counter_trend and not _is_forex:
        try:
            _ext_cand = []
            for _tf in (mt5.TIMEFRAME_M5, mt5.TIMEFRAME_M15, mt5.TIMEFRAME_H1):
                _zs, _ = find_demand_supply_zones(symbol, _tf, 60)
                for z in _zs:
                    if direction == "long" and z["type"] == "SUPPLY" and z["lo"] > price + 1.0:
                        _ext_cand.append(float(z["lo"]))
                    elif direction == "short" and z["type"] == "DEMAND" and z["hi"] < price - 1.0:
                        _ext_cand.append(float(z["hi"]))
            if _ext_cand:
                _ztgt = min(_ext_cand) if direction == "long" else max(_ext_cand)
                _max_ext = min(6.0 * sl_dist, 50.0)   # RR<=6 + abs<=$50
                if direction == "long":
                    _ztp = _ztgt - SR_TP_BUFFER
                    if _ztp > tp and (_ztp - price) <= _max_ext:
                        tp = _ztp
                        tp_reason += f" [ZONE-TP→{_ztgt:.0f} เก็บยาว RR{(_ztp-price)/sl_dist:.1f}]"
                else:
                    _ztp = _ztgt + SR_TP_BUFFER
                    if _ztp < tp and (price - _ztp) <= _max_ext:
                        tp = _ztp
                        tp_reason += f" [ZONE-TP→{_ztgt:.0f} เก็บยาว RR{(price-_ztp)/sl_dist:.1f}]"
        except Exception:
            pass

    # ── Absolute TP Ceiling: กัน "TP ไกลเกินไป" ──────────────────────────
    # user framework: TP อิงโซน RR 1:2-1:5 แต่ไม่ไกลเกิน ~$50
    # (เช่น SL $15 TP $45 OK, แต่ TP $70+ = ไกลเกินจริง price ไม่ถึง)
    MAX_TP_ABS = 50.0   # USD, XAU only
    if not _is_forex:
        if direction == "long" and (tp - price) > MAX_TP_ABS:
            tp = price + MAX_TP_ABS
            tp_reason += f" [TP-ABS-CAP $50]"
            print(f"   [TP-ABS-CAP] LONG TP capped at +$50 (price+50={tp:.2f})", flush=True)
        elif direction == "short" and (price - tp) > MAX_TP_ABS:
            tp = price - MAX_TP_ABS
            tp_reason += f" [TP-ABS-CAP $50]"
            print(f"   [TP-ABS-CAP] SHORT TP capped at -$50 (price-50={tp:.2f})", flush=True)

    # ── Lot Scaling: ปิดแล้ว — risk-based sizing คุม lot จาก SL dist โดยตรง ──
    # (เดิม: TP สั้น → เพิ่ม lot → แต่ขัดกับ fixed-risk เพราะเพิ่มความเสี่ยง)
    actual_rr = abs(tp - price) / sl_dist if sl_dist > 0 else target_rr

    if tp_reason:
        print(f"   [TP-ADJ]{tp_reason} | RR {target_rr:.1f}→{actual_rr:.1f} | raw_tp={raw_tp:.2f}→tp={tp:.2f}", flush=True)

    # ── MIN RR check: ห้ามส่ง order ที่ RR < 1.2 (จาก log: star_long TP $2.2 < SL $3.1)
    # TP ถูก cap จน RR น้อยเกิน → ยกเลิกดีกว่าเข้าเสี่ยงแล้วไม่คุ้ม
    _final_rr = abs(tp - price) / sl_dist if sl_dist > 0 else 0
    if not _is_forex and _final_rr < 1.2:
        print(f"   [MIN-RR-BLOCK] RR={_final_rr:.2f} < 1.2 → ยกเลิก (TP={tp:.2f} SL={sl:.2f})", flush=True)
        return {"ok": False, "error": f"RR too low: {_final_rr:.2f} (TP={tp:.2f} SL={sl:.2f})"}

    d = info.digits
    step = info.volume_step
    lot = round(round(lot / step) * step, 8)
    lot = max(lot, info.volume_min)
    lot = min(lot, info.volume_max)
    req = {
        "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol,
        "volume": lot, "type": otype,
        "price": round(price, d), "sl": round(sl, d), "tp": round(tp, d),
        "deviation": 20, "magic": magic, "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_FOK,
    }
    r = mt5.order_send(req)
    # 2026-06-17 resilience: r is None = IPC send failed (order ไม่ถึง server) → reconnect + retry
    #   กัน "decision ผ่านแต่ order หล่นเงียบ" ตอน MT5 IPC กระตุก (user: บอทไม่เปิด order)
    _try = 0
    while r is None and _try < 2:
        _try += 1
        try:
            mt5.initialize()
        except Exception:
            pass
        time.sleep(1)
        try:
            _tk = mt5.symbol_info_tick(symbol)
            if _tk:
                req["price"] = round(_tk.ask if direction == "long" else _tk.bid, d)
        except Exception:
            pass
        print(f"   [SUBMIT-RETRY {_try}] IPC send failed → reconnect+retry ({comment})", flush=True)
        r = mt5.order_send(req)
    if r is None or r.retcode != mt5.TRADE_RETCODE_DONE:
        return {"ok": False, "error": f"{r.retcode if r else 'None'} {r.comment if r else mt5.last_error()}"}
    return {"ok": True, "order_id": r.order, "entry": r.price, "lot": r.volume, "sl": sl, "tp": tp}


def log_decision(log_path, decision):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    first_write = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(decision.keys()))
        if first_write:
            w.writeheader()
        w.writerow(decision)


def save_entry_chart(df_m5, entry_price, direction, strategy, order_id, atr_val, symbol="XAUUSD"):
    """บันทึก chart ทุกครั้งที่เข้าออเดอร์ — DEMAND=แดง SUPPLY=น้ำเงิน"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import mplfinance as mpf

        CHARTS_DIR.mkdir(parents=True, exist_ok=True)
        n_plot = min(80, len(df_m5))
        plot_df = df_m5.tail(n_plot).copy()
        plot_df.index = pd.date_range(end=dt.datetime.now(), periods=n_plot, freq="5min")
        ohlc = plot_df[["open", "high", "low", "close", "volume"]].copy()

        range_high = ohlc["high"].iloc[-30:].max()
        range_low  = ohlc["low"].iloc[-30:].min()
        range_mid  = (range_high + range_low) / 2

        fig, axes = mpf.plot(
            ohlc, type="candle", volume=False, returnfig=True,
            figsize=(16, 8), style="nightclouds",
            title=f"{symbol} M5  {direction.upper()} @ {entry_price:.5g}  [{strategy}]  #{order_id}"
        )
        ax = axes[0]

        # DEMAND = แดง (ล่าง — โซนซื้อ)
        ax.axhspan(range_low - 0.05 * atr_val, range_low + 0.5 * atr_val,
                   alpha=0.25, color="red")
        ax.text(2, range_low + 0.1 * atr_val, "DEMAND", color="red",
                fontsize=8, fontweight="bold")

        # SUPPLY = น้ำเงิน (บน — โซนขาย)
        ax.axhspan(range_high - 0.5 * atr_val, range_high + 0.05 * atr_val,
                   alpha=0.25, color="deepskyblue")
        ax.text(2, range_high - 0.4 * atr_val, "SUPPLY", color="deepskyblue",
                fontsize=8, fontweight="bold")

        # Midpoint (structural divider)
        ax.axhline(y=range_mid, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.text(n_plot - 4, range_mid + 0.1, "MID", color="gray", fontsize=7)

        # Entry line + circle
        color = "lime" if direction == "long" else "yellow"
        ax.axhline(y=entry_price, color=color, linewidth=1.5, linestyle="--", alpha=0.9)
        y_range = range_high - range_low
        circle = plt.Circle((n_plot - 1, entry_price), y_range * 0.02,
                             color=color, fill=False, linewidth=3, zorder=10)
        ax.add_patch(circle)

        arrow = "▲" if direction == "long" else "▼"
        ax.annotate(
            f"{arrow} {direction.upper()} @ {entry_price:.2f}\n[{strategy}]",
            xy=(n_plot - 1, entry_price),
            xytext=(n_plot - 12, entry_price + (2 if direction == "long" else -3)),
            fontsize=9, color=color, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=color, lw=2),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#0d0d1a", edgecolor=color, alpha=0.9)
        )

        # SL / TP lines
        sl = entry_price + (1.5 * atr_val if direction == "short" else -1.5 * atr_val)
        tp = entry_price - (4.5 * atr_val if direction == "short" else -4.5 * atr_val)
        ax.axhline(y=sl, color="orange", linewidth=1, linestyle=":", alpha=0.8)
        ax.text(n_plot - 6, sl + 0.2, f"SL {sl:.0f}", color="orange", fontsize=7)
        ax.axhline(y=tp, color="cyan", linewidth=1, linestyle=":", alpha=0.8)
        ax.text(n_plot - 6, tp - 1.2, f"TP {tp:.0f}", color="cyan", fontsize=7)

        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = CHARTS_DIR / f"{ts}_{strategy}_{direction}_{entry_price:.0f}.png"
        fig.savefig(fname, dpi=120, bbox_inches="tight", facecolor="#131722")
        plt.close(fig)
        return str(fname)
    except Exception as e:
        return f"chart_error:{e}"


def log_trade_result(order_id, strategy, direction, entry_price, lot,
                     sl_price, tp_price, chart_path, entry_time):
    """บันทึก trade entry ลง trade_results.csv เพื่อ research."""
    TRADES_LOG.parent.mkdir(parents=True, exist_ok=True)
    first = not TRADES_LOG.exists()
    row = {
        "order_id": order_id, "entry_time": entry_time,
        "strategy": strategy, "direction": direction,
        "entry_price": entry_price, "lot": lot,
        "sl_price": round(sl_price, 2), "tp_price": round(tp_price, 2),
        "exit_time": "", "exit_price": "", "profit_usd": "",
        "result": "OPEN", "chart_path": chart_path,
    }
    with open(TRADES_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if first:
            w.writeheader()
        w.writerow(row)


def update_closed_trades():
    """ดึง deals ที่ปิดแล้วและอัปเดต result ใน trade_results.csv"""
    if not TRADES_LOG.exists():
        return
    try:
        df = pd.read_csv(TRADES_LOG, dtype=str)
        open_mask = df["result"] == "OPEN"
        if not open_mask.any():
            return
        # ดึง deal history 48h
        since = dt.datetime.now() - dt.timedelta(hours=48)
        deals = mt5.history_deals_get(since, dt.datetime.now()) or []
        close_deals = {d.order: d for d in deals if d.entry == 1}  # entry=1 = close deal
        updated = False
        for idx in df[open_mask].index:
            oid = str(df.at[idx, "order_id"])
            matched = [d for d in close_deals.values() if str(d.position_id) == oid]
            if matched:
                d = matched[0]
                exit_time = dt.datetime.fromtimestamp(d.time).isoformat()
                df.at[idx, "exit_time"]  = exit_time
                df.at[idx, "exit_price"] = str(d.price)
                df.at[idx, "profit_usd"] = str(round(d.profit, 2))
                df.at[idx, "result"]     = "WIN" if d.profit > 0 else "LOSS"
                updated = True
        if updated:
            df.to_csv(TRADES_LOG, index=False)
    except Exception as e:
        print(f"[trade_result_updater] {e}")


# ──────────────────────────────────────────────────────────────────
# find_sr_levels — compatibility wrapper (TP capping ใน submit_order ยังใช้)
# ──────────────────────────────────────────────────────────────────
def find_sr_levels(symbol, timeframe=None, n_bars=72):
    """Compatibility wrapper: แปลง zones → S/R levels สำหรับ TP capping."""
    if timeframe is None:
        timeframe = mt5.TIMEFRAME_H1
    zones, _ = find_demand_supply_zones(symbol, timeframe, n_bars)
    levels = []
    for z in zones:
        if z["type"] == "DEMAND":
            levels.append(("SUPPORT", z["lo"]))
        else:
            levels.append(("RESIST", z["hi"]))
    return levels


# ──────────────────────────────────────────────────────────────────
# Smart Exit v2 — ใช้ Demand/Supply zones เดียวกับ Entry system
# เข้า M1 → ออก M1   ไม่ใช่อิง H1 swing level
# ──────────────────────────────────────────────────────────────────

def find_demand_supply_zones(symbol, timeframe, n_bars=60):
    """หา Demand/Supply zones ด้วย logic เดียวกับ DSF entry system.

    Demand zone = swing low ที่ bounce ขึ้น >= 2 ATR (เหมือน detect_dsf_short)
    Supply zone = swing high ที่ reject ลง >= 2 ATR (เหมือน detect_dsf_long)

    Returns: (zones_list, atr_value)
        zones_list = [{"type": "DEMAND"|"SUPPLY", "lo": float, "hi": float}, ...]
    """
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
    if rates is None or len(rates) < 20:
        return [], 0

    high  = [float(b[2]) for b in rates]
    low   = [float(b[3]) for b in rates]
    close = [float(b[4]) for b in rates]

    # Calculate ATR (same way as entry system)
    atr_vals = []
    for i in range(1, len(rates)):
        tr = max(high[i] - low[i],
                 abs(high[i] - close[i-1]),
                 abs(low[i] - close[i-1]))
        atr_vals.append(tr)
    atr_val = sum(atr_vals[-14:]) / 14 if len(atr_vals) >= 14 else (
        sum(atr_vals) / len(atr_vals) if atr_vals else 1.0)

    zones = []

    # Demand zones = swing lows with bounce >= 2 ATR (same logic as DSF)
    for i in range(2, len(rates) - 2):
        if low[i] <= low[i-1] and low[i] <= low[i+1]:
            future_high = max(high[i:min(i + 10, len(rates))])
            if (future_high - low[i]) / atr_val >= 2.0:
                zones.append({
                    "type": "DEMAND",
                    "lo": low[i],
                    "hi": low[i] + 0.6 * atr_val,
                })

    # Supply zones = swing highs with drop >= 2 ATR (same logic as DSF)
    for i in range(2, len(rates) - 2):
        if high[i] >= high[i-1] and high[i] >= high[i+1]:
            future_low = min(low[i:min(i + 10, len(rates))])
            if (high[i] - future_low) / atr_val >= 2.0:
                zones.append({
                    "type": "SUPPLY",
                    "lo": high[i] - 0.6 * atr_val,
                    "hi": high[i],
                })

    # ── Wick Cluster Zones: ไส้เทียนแตะระดับเดียวกัน 3+ ครั้ง ────────────
    # จาก user feedback: แนวไส้เทียนเยอะๆ = Supply/Demand zone ที่ swing high ไม่จับ
    # เช่น ceiling resistance (ไส้บนแตะ 4503-4504 หลายครั้ง) = Supply zone
    # หรือ floor support (ไส้ล่างแตะ 4490-4491 หลายครั้ง) = Demand zone
    _open = [float(b[1]) for b in rates]
    for i in range(8, len(rates) - 2):
        # ── Supply: upper wick cluster (ceiling rejection) ──
        _ref_h = high[i]
        _wick_count = 0
        for j in range(max(0, i - 15), min(i + 3, len(rates))):
            if abs(high[j] - _ref_h) <= 0.3 * atr_val:
                _body_top = max(close[j], _open[j])
                _upper_wick = high[j] - _body_top
                _candle_range = high[j] - low[j]
                if _candle_range > 0 and _upper_wick > 0.15 * _candle_range:
                    _wick_count += 1
        if _wick_count >= 3:
            future_low = min(low[i:min(i + 10, len(rates))])
            if (_ref_h - future_low) / atr_val >= 1.5:
                zones.append({
                    "type": "SUPPLY",
                    "lo": _ref_h - 0.6 * atr_val,
                    "hi": _ref_h,
                })

        # ── Demand: lower wick cluster (floor support) ──
        _ref_l = low[i]
        _wick_count = 0
        for j in range(max(0, i - 15), min(i + 3, len(rates))):
            if abs(low[j] - _ref_l) <= 0.3 * atr_val:
                _body_bot = min(close[j], _open[j])
                _lower_wick = _body_bot - low[j]
                _candle_range = high[j] - low[j]
                if _candle_range > 0 and _lower_wick > 0.15 * _candle_range:
                    _wick_count += 1
        if _wick_count >= 3:
            future_high = max(high[i:min(i + 10, len(rates))])
            if (future_high - _ref_l) / atr_val >= 1.5:
                zones.append({
                    "type": "DEMAND",
                    "lo": _ref_l,
                    "hi": _ref_l + 0.6 * atr_val,
                })

    # Deduplicate zones within 0.5 ATR (keep first of each type)
    clean = []
    for z in sorted(zones, key=lambda x: x["lo"]):
        if not clean or abs(z["lo"] - clean[-1]["lo"]) > 0.5 * atr_val or z["type"] != clean[-1]["type"]:
            clean.append(z)

    # ── Cross-type dedup: Demand ซ้อน Supply ถ้า overlap > 80% → ลบตัวเล็กกว่า ──
    # ปัญหาเดิม: ลบทุก overlap → Demand zone ที่ user เห็นหายไป
    # Fix: ปล่อย mild overlap ไว้ (Demand 4445 + Supply 4443 = ไม่ซ้อนมาก = keep ทั้งคู่)
    # ลบเฉพาะ significant overlap (>80% ของ zone เล็กกว่า = basically same zone)
    final = []
    for z in clean:
        significant_overlap = False
        for existing in final:
            if existing["type"] != z["type"]:
                overlap = min(z["hi"], existing["hi"]) - max(z["lo"], existing["lo"])
                smaller_size = min(z["hi"] - z["lo"], existing["hi"] - existing["lo"])
                if smaller_size > 0 and overlap / smaller_size > 0.8:
                    significant_overlap = True
                    break
        if not significant_overlap:
            final.append(z)

    return final, atr_val


def detect_zone_bounce(symbol, zone_lo, zone_hi, direction, atr_val):
    """ตรวจว่าราคา bounce จาก zone หรือไม่ — ดูทั้ง M5 + M1.

    เข้า M1 → ต้องเช็ค M1 ด้วย ไม่ใช่แค่ M5

    Bounce patterns (3 แบบ):
    1. Rejection candle (hammer / shooting star)
    2. Engulfing candle
    3. Structural bounce: higher-lows (M1) = zone hold ไม่ทะลุ เด้งกลับ
    """
    m1 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 12)
    m5 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 6)

    if direction == "short":
        # SELL → กลัว bounce ขึ้นจาก Demand zone
        proximity = 1.0 * atr_val  # กว้างกว่าเดิม (v1 ใช้ 0.5)

        # === Pattern 1 & 2: Rejection / Engulfing บน M5 ===
        if m5 is not None and len(m5) >= 4:
            for c in m5[-4:-1]:  # 3 แท่ง M5 ที่ปิดแล้ว
                c_low = float(c[3]); c_close = float(c[4])
                c_open = float(c[1]); c_high = float(c[2])

                if c_low > zone_hi + proximity:
                    continue

                body = c_close - c_open
                lower_wick = min(c_open, c_close) - c_low
                candle_range = c_high - c_low
                if candle_range == 0:
                    continue

                # Hammer
                if lower_wick > 0.5 * candle_range and body > 0:
                    return True, f"M5 Hammer bounce จาก Demand {zone_lo:.0f}-{zone_hi:.0f}"
                # Bullish engulfing
                if body > 0 and body > 0.25 * atr_val:
                    return True, f"M5 Engulf bounce จาก Demand {zone_lo:.0f}-{zone_hi:.0f}"

        # === Pattern 1 & 2: Rejection / Engulfing บน M1 ===
        if m1 is not None and len(m1) >= 4:
            for c in m1[-4:-1]:
                c_low = float(c[3]); c_close = float(c[4])
                c_open = float(c[1]); c_high = float(c[2])

                if c_low > zone_hi + proximity:
                    continue

                body = c_close - c_open
                lower_wick = min(c_open, c_close) - c_low
                candle_range = c_high - c_low
                if candle_range == 0:
                    continue

                # Hammer M1
                if lower_wick > 0.6 * candle_range and body > 0:
                    return True, f"M1 Hammer bounce จาก Demand {zone_lo:.0f}-{zone_hi:.0f}"
                # Engulfing M1
                if body > 0 and body > 0.10 * atr_val:
                    return True, f"M1 Engulf bounce จาก Demand {zone_lo:.0f}-{zone_hi:.0f}"

        # === Pattern 3: Structural bounce — higher-lows จาก zone (M1) ===
        # "ไม่ทะลุโซนเด้งกลับ" — zone hold + 3 HL + close เหนือ zone
        if m1 is not None and len(m1) >= 8:
            recent = m1[-8:-1]  # 7 แท่ง M1 ล่าสุดที่ปิด
            lows   = [float(c[3]) for c in recent]
            closes = [float(c[4]) for c in recent]

            # zone hold: low ทุกแท่งอยู่เหนือ zone_lo (ไม่ทะลุ)
            zone_held = all(l >= zone_lo - 0.3 * atr_val for l in lows)

            # อย่างน้อย 1 แท่งแตะ zone (low อยู่ใน zone_lo ~ zone_hi+proximity)
            touched_zone = any(l <= zone_hi + 0.5 * atr_val for l in lows)

            # higher-lows: 3+ จาก 6 คู่
            hl_count = sum(1 for i in range(1, len(lows)) if lows[i] > lows[i-1])

            # close ล่าสุดเด้งเหนือ zone
            last_close = closes[-1]

            if zone_held and touched_zone and hl_count >= 3 and last_close > zone_hi:
                return True, f"M1 zone bounce: {hl_count}HL จาก Demand {zone_lo:.0f}-{zone_hi:.0f}"

    elif direction == "long":
        # BUY → กลัว bounce ลงจาก Supply zone
        proximity = 1.0 * atr_val

        # === Pattern 1 & 2: M5 ===
        if m5 is not None and len(m5) >= 4:
            for c in m5[-4:-1]:
                c_high = float(c[2]); c_close = float(c[4])
                c_open = float(c[1]); c_low = float(c[3])

                if c_high < zone_lo - proximity:
                    continue

                body = c_close - c_open
                upper_wick = c_high - max(c_open, c_close)
                candle_range = c_high - c_low
                if candle_range == 0:
                    continue

                if upper_wick > 0.5 * candle_range and body < 0:
                    return True, f"M5 Shooting star จาก Supply {zone_lo:.0f}-{zone_hi:.0f}"
                if body < 0 and abs(body) > 0.25 * atr_val:
                    return True, f"M5 Engulf reject จาก Supply {zone_lo:.0f}-{zone_hi:.0f}"

        # === Pattern 1 & 2: M1 ===
        if m1 is not None and len(m1) >= 4:
            for c in m1[-4:-1]:
                c_high = float(c[2]); c_close = float(c[4])
                c_open = float(c[1]); c_low = float(c[3])

                if c_high < zone_lo - proximity:
                    continue

                body = c_close - c_open
                upper_wick = c_high - max(c_open, c_close)
                candle_range = c_high - c_low
                if candle_range == 0:
                    continue

                if upper_wick > 0.6 * candle_range and body < 0:
                    return True, f"M1 Shooting star จาก Supply {zone_lo:.0f}-{zone_hi:.0f}"
                if body < 0 and abs(body) > 0.10 * atr_val:
                    return True, f"M1 Engulf reject จาก Supply {zone_lo:.0f}-{zone_hi:.0f}"

        # === Pattern 3: Structural bounce — lower-highs จาก zone (M1) ===
        if m1 is not None and len(m1) >= 8:
            recent = m1[-8:-1]
            highs  = [float(c[2]) for c in recent]
            closes = [float(c[4]) for c in recent]

            zone_held = all(h <= zone_hi + 0.3 * atr_val for h in highs)
            touched_zone = any(h >= zone_lo - 0.5 * atr_val for h in highs)
            lh_count = sum(1 for i in range(1, len(highs)) if highs[i] < highs[i-1])
            last_close = closes[-1]

            if zone_held and touched_zone and lh_count >= 3 and last_close < zone_lo:
                return True, f"M1 zone reject: {lh_count}LH จาก Supply {zone_lo:.0f}-{zone_hi:.0f}"

    return False, ""


def breakeven_check(symbol):
    """Trailing Breakeven แบบ Step — ล็อคกำไรเป็นขั้นบันได

    ขั้นที่ 1: ราคาวิ่ง >= 1.0x SL_dist → ย้าย SL มา entry + 0.5  (BE)
    ขั้นที่ 2: ราคาวิ่ง >= 2.0x SL_dist → ย้าย SL มา entry + 1.0x SL_dist  (ล็อค 50%)
    ขั้นที่ 3: ราคาวิ่ง >= 3.0x SL_dist → ย้าย SL มา entry + 2.0x SL_dist  (ล็อค 67%)

    ตัวอย่าง: entry=4500, SL=4490 (SL_dist=10)
      ราคาถึง 4510 → SL ย้ายมา 4500.5   (BE, ถ้าโดน SL = +$0.5)
      ราคาถึง 4520 → SL ย้ายมา 4510     (ถ้าโดน SL = +$10)
      ราคาถึง 4530 → SL ย้ายมา 4520     (ถ้าโดน SL = +$20)
    """
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return
    info = mt5.symbol_info(symbol)
    if info is None:
        return
    d = info.digits
    BE_BUFFER = 0.5  # ขั้นแรก: entry +/- 0.5 USD

    # ── M5 ATR สำหรับ Tight Trailing (จับ peak เมื่อกำไรเยอะ) ──
    _atr_m5 = 0.0
    try:
        _m5 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 16)
        if _m5 is not None and len(_m5) >= 15:
            _trs = [max(float(_m5[i][2]) - float(_m5[i][3]),
                        abs(float(_m5[i][2]) - float(_m5[i-1][4])),
                        abs(float(_m5[i][3]) - float(_m5[i-1][4]))) for i in range(1, 15)]
            _atr_m5 = sum(_trs) / len(_trs)
    except Exception:
        pass
    TIGHT_PROFIT_USD = 20.0   # กำไร > $20 → trail 1.8 ATR (ล็อก peak แต่ให้เทรนวิ่งต่อ)
    TIGHT_ATR_MULT = 1.8      # 2026-06-03 คลาย 1.2→1.8 (user: ปิดเร็วตอนตามเทรน)

    # Trailing steps: (trigger_multiplier, sl_offset_multiplier)
    # trigger = entry +/- trigger_mult * sl_dist
    # new_sl  = entry +/- sl_offset_mult * sl_dist
    # 2026-06-18 user: "SL ที่ BE กำไร 0 ยังไม่ปลอดภัย — ล็อกกำไรเร็วขึ้น"
    # เพิ่ม step 1.5x และ 2.5x และ 4x — ล็อกกำไรขั้นบันไดละเอียดขึ้น (ไม่กระทบ ride trend ใหญ่)
    TRAIL_STEPS = [
        (1.0, 0.0),   # Step 1: กำไร 1x → BE
        (1.5, 0.5),   # Step 1.5 (NEW): กำไร 1.5x → ล็อก 0.5x SL_dist
        (2.0, 1.0),   # Step 2: กำไร 2x → ล็อก 1x
        (2.5, 1.5),   # Step 2.5 (NEW): กำไร 2.5x → ล็อก 1.5x
        (3.0, 2.0),   # Step 3: กำไร 3x → ล็อก 2x
        (4.0, 3.0),   # Step 4 (NEW): กำไร 4x → ล็อก 3x (gain เพิ่ม trailing)
    ]

    for pos in positions:
        entry = pos.price_open
        cur_sl = pos.sl
        cur_tp = pos.tp

        if pos.type == 0:  # BUY
            # คำนวณ original SL distance (ใช้ SL เดิม ถ้ายังไม่เคยย้าย)
            # ถ้า SL ถูกย้ายแล้ว ใช้ TP distance / RR ratio ประมาณ SL_dist เดิม
            if cur_sl < entry:
                orig_sl_dist = entry - cur_sl
            else:
                # SL ถูกย้ายแล้ว (BE+) — ประมาณ orig SL จาก TP
                tp_dist = cur_tp - entry if cur_tp > entry else 0
                orig_sl_dist = tp_dist / 3.0 if tp_dist > 0 else 5.0  # fallback

            if orig_sl_dist <= 0:
                continue

            cur_profit_dist = tick.bid - entry

            # หา step สูงสุดที่ trigger ได้
            best_new_sl = None
            best_step = None
            for trigger_mult, offset_mult in TRAIL_STEPS:
                if cur_profit_dist >= trigger_mult * orig_sl_dist:
                    if offset_mult == 0.0:
                        candidate_sl = round(entry + BE_BUFFER, d)
                    else:
                        candidate_sl = round(entry + offset_mult * orig_sl_dist, d)
                    # SL ใหม่ต้องดีกว่า (สูงกว่า) SL เดิม
                    if candidate_sl > cur_sl:
                        best_new_sl = candidate_sl
                        best_step = trigger_mult

            # ── Tight Trailing: กำไร > $20 → SL ตาม 1.2 ATR ใต้ราคา (จับ peak) ──
            if pos.profit > TIGHT_PROFIT_USD and _atr_m5 > 0:
                _tight = round(tick.bid - TIGHT_ATR_MULT * _atr_m5, d)
                if _tight > cur_sl and (best_new_sl is None or _tight > best_new_sl):
                    best_new_sl = _tight
                    best_step = -1   # marker = tight

            if best_new_sl is not None:
                req = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "symbol": symbol,
                    "position": pos.ticket,
                    "sl": best_new_sl,
                    "tp": round(cur_tp, d),
                }
                r = mt5.order_send(req)
                if r and r.retcode == mt5.TRADE_RETCODE_DONE:
                    locked = best_new_sl - entry
                    _tag = "TIGHT" if best_step == -1 else f"step {best_step:.0f}x"
                    print(f"[TRAIL] BUY #{pos.ticket} SL {cur_sl:.2f} -> {best_new_sl:.2f} ({_tag}, lock +${locked:.1f}, bid={tick.bid:.2f})", flush=True)

        elif pos.type == 1:  # SELL
            if cur_sl > entry:
                orig_sl_dist = cur_sl - entry
            else:
                tp_dist = entry - cur_tp if cur_tp < entry else 0
                orig_sl_dist = tp_dist / 3.0 if tp_dist > 0 else 5.0

            if orig_sl_dist <= 0:
                continue

            cur_profit_dist = entry - tick.ask

            best_new_sl = None
            best_step = None
            for trigger_mult, offset_mult in TRAIL_STEPS:
                if cur_profit_dist >= trigger_mult * orig_sl_dist:
                    if offset_mult == 0.0:
                        candidate_sl = round(entry - BE_BUFFER, d)
                    else:
                        candidate_sl = round(entry - offset_mult * orig_sl_dist, d)
                    if candidate_sl < cur_sl:
                        best_new_sl = candidate_sl
                        best_step = trigger_mult

            # ── Tight Trailing: กำไร > $20 → SL ตาม 1.2 ATR เหนือราคา (จับ peak) ──
            if pos.profit > TIGHT_PROFIT_USD and _atr_m5 > 0:
                _tight = round(tick.ask + TIGHT_ATR_MULT * _atr_m5, d)
                if _tight < cur_sl and (best_new_sl is None or _tight < best_new_sl):
                    best_new_sl = _tight
                    best_step = -1

            if best_new_sl is not None:
                req = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "symbol": symbol,
                    "position": pos.ticket,
                    "sl": best_new_sl,
                    "tp": round(cur_tp, d),
                }
                r = mt5.order_send(req)
                if r and r.retcode == mt5.TRADE_RETCODE_DONE:
                    locked = entry - best_new_sl
                    _tag = "TIGHT" if best_step == -1 else f"step {best_step:.0f}x"
                    print(f"[TRAIL] SELL #{pos.ticket} SL {cur_sl:.2f} -> {best_new_sl:.2f} ({_tag}, lock +${locked:.1f}, ask={tick.ask:.2f})", flush=True)


def momentum_exit_check(symbol):
    """Smart Exit v3: ประเมินตลาด + ปิดเอง 4 trigger:
    1. M5 lower-highs ≥ 2 + กำไร ≥ $5 (BUY) → momentum อ่อน
    2. M1 reject ที่ resistance ใกล้ + กำไร ≥ $3 (BUY) → reject ชัด
    3. CHOCH ตรงข้าม (M5) + กำไร > $1 → structure break
    4. กำไร ≥ $15 + M1 bar ตรงข้าม body > 0.5×ATR → take profit
    (mirror สำหรับ SHORT)
    """
    global _LAST_CLOSE_TIME
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return

    m5 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 20)
    if m5 is None or len(m5) < 6:
        return
    m1 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 15)
    if m1 is None or len(m1) < 5:
        return

    m5_highs = [float(b[2]) for b in m5]
    m5_lows = [float(b[3]) for b in m5]
    m5_closes = [float(b[4]) for b in m5]
    m5_opens = [float(b[1]) for b in m5]
    m1_o = [float(b[1]) for b in m1]
    m1_c = [float(b[4]) for b in m1]
    m1_h = [float(b[2]) for b in m1]
    m1_l = [float(b[3]) for b in m1]

    # M1 ATR
    m1_atr = sum(max(m1_h[i]-m1_l[i], abs(m1_h[i]-m1_c[i-1]), abs(m1_l[i]-m1_c[i-1]))
                 for i in range(1, len(m1_c))) / (len(m1_c)-1)

    # M5 momentum (loosened: ≥ 2 ครั้ง แทน 3)
    _lh_count = sum(1 for i in range(-3, 0) if m5_highs[i] < m5_highs[i-1])
    _hl_count = sum(1 for i in range(-3, 0) if m5_lows[i] > m5_lows[i-1])

    # CHOCH M5
    _choch_bear = False; _choch_bull = False
    if len(m5_lows) >= 6:
        if m5_highs[-3] > max(m5_highs[-6:-3]) and m5_lows[-1] < min(m5_lows[-3:-1]):
            _choch_bear = True
        if m5_lows[-3] < min(m5_lows[-6:-3]) and m5_highs[-1] > max(m5_highs[-3:-1]):
            _choch_bull = True

    # Nearest M5 resistance/support
    cur_price = tick.bid
    _resists_above = [h for h in m5_highs[-15:-1] if h > cur_price + 0.3]
    _supports_below = [l for l in m5_lows[-15:-1] if l < cur_price - 0.3]
    _nearest_resist = min(_resists_above) if _resists_above else None
    _nearest_support = max(_supports_below) if _supports_below else None

    # M1 last bar bearish/bullish strong?
    last_body_m1 = m1_c[-1] - m1_o[-1]
    last_body_pct = abs(last_body_m1) / m1_atr if m1_atr > 0 else 0
    last_bar_bearish_strong = last_body_m1 < 0 and last_body_pct > 0.5
    last_bar_bullish_strong = last_body_m1 > 0 and last_body_pct > 0.5
    # 2026-06-18 user: M1 higher-low (short reversal) / lower-high (long reversal) — เร็วกว่า M5
    _m1_higher_low = len(m1_l) >= 3 and m1_l[-1] > m1_l[-2] > m1_l[-3]   # 3 bars HL ติด = uptrend reversal
    _m1_lower_high = len(m1_h) >= 3 and m1_h[-1] < m1_h[-2] < m1_h[-3]   # 3 bars LH ติด = downtrend reversal

    # 2026-06-18 user "Morning Star ที่ M1 ออกเลย": detect M1 reversal patterns สำหรับ exit เร็ว
    _m1_morning_star = False; _m1_evening_star = False
    _m1_liq_sweep_bull = False; _m1_liq_sweep_bear = False
    if len(m1_c) >= 3 and m1_atr > 0:
        _b1m1 = m1_c[-3] - m1_o[-3]; _b2m1 = m1_c[-2] - m1_o[-2]; _b3m1 = m1_c[-1] - m1_o[-1]
        _rng2m1 = m1_h[-2] - m1_l[-2]
        # Morning Star (bullish reversal): red strong + doji/small + green strong
        _m1_morning_star = (_b1m1 < -0.7 * m1_atr and abs(_b2m1) < 0.3 * m1_atr
                            and _rng2m1 > 0 and _b3m1 > 0.7 * m1_atr)
        # Evening Star (bearish reversal): green strong + doji/small + red strong
        _m1_evening_star = (_b1m1 > 0.7 * m1_atr and abs(_b2m1) < 0.3 * m1_atr
                            and _rng2m1 > 0 and _b3m1 < -0.7 * m1_atr)
        # Liq sweep on M1 last bar: long lower wick (bullish sweep) / long upper wick (bearish sweep)
        _lw_m1 = min(m1_o[-1], m1_c[-1]) - m1_l[-1]
        _uw_m1 = m1_h[-1] - max(m1_o[-1], m1_c[-1])
        _m1_liq_sweep_bull = _lw_m1 > 1.5 * m1_atr and _b3m1 > 0
        _m1_liq_sweep_bear = _uw_m1 > 1.5 * m1_atr and _b3m1 < 0

    info = mt5.symbol_info(symbol)
    d = info.digits if info else 2

    # ── IMB EXIT: 2 strong opposing M5 candles + M1 ยืนยัน → ปิด (2026-06-09 ปรับ strict ขึ้น) ──
    # ก่อนหน้า IMB ปิดเร็วเกิน (-$15.75 1 เคส) — ต้องการ M1 momentum agree ด้วย
    _m5_atr_calc = 0.0
    if len(m5_closes) >= 15:
        _trs = [max(m5_highs[i]-m5_lows[i], abs(m5_highs[i]-m5_closes[i-1]),
                    abs(m5_lows[i]-m5_closes[i-1])) for i in range(1, 15)]
        _m5_atr_calc = sum(_trs) / len(_trs)
    _b1_m5 = m5_closes[-2] - m5_opens[-2] if len(m5_closes) >= 2 else 0
    _b2_m5 = m5_closes[-1] - m5_opens[-1] if len(m5_closes) >= 1 else 0
    # ใหม่: ต้อง body แรงกว่า 0.7 ATR each (เดิม 0.5) + ต้องสอดคล้องกับ M1
    _green_strong_2 = _m5_atr_calc > 0 and _b1_m5 > 0.7 * _m5_atr_calc and _b2_m5 > 0.7 * _m5_atr_calc
    _red_strong_2 = _m5_atr_calc > 0 and _b1_m5 < -0.7 * _m5_atr_calc and _b2_m5 < -0.7 * _m5_atr_calc
    # M1 confirmation: last 3 M1 closed bars ต้อง agree ทิศเดียวกับ IMB
    _m1_green3 = sum(1 for i in range(-3,0) if m1_c[i] > m1_o[i]) >= 2 if len(m1_c) >= 3 else False
    _m1_red3   = sum(1 for i in range(-3,0) if m1_c[i] < m1_o[i]) >= 2 if len(m1_c) >= 3 else False
    _green_strong_2 = _green_strong_2 and _m1_green3
    _red_strong_2   = _red_strong_2 and _m1_red3

    for pos in positions:
        pnl = pos.profit
        direction = "long" if pos.type == 0 else "short"
        entry = pos.price_open
        exit_reason = None

        # IMB exit ก่อน (works even when losing)
        if direction == "short" and _green_strong_2:
            exit_reason = f"IMB BULL breakout ({pnl:+.1f})"
        elif direction == "long" and _red_strong_2:
            exit_reason = f"IMB BEAR breakdown ({pnl:+.1f})"
        # 2026-06-18 user "Morning Star M1 ออกเลย — ขาดทุนได้": M1 reversal pattern ปิดทันทีไม่ดู pnl
        elif direction == "short" and (_m1_morning_star or _m1_liq_sweep_bull):
            _pat = "Morning Star" if _m1_morning_star else "Liq Sweep Bull"
            exit_reason = f"M1 {_pat} ${pnl:+.1f} (reversal — exit regardless)"
        elif direction == "long" and (_m1_evening_star or _m1_liq_sweep_bear):
            _pat = "Evening Star" if _m1_evening_star else "Liq Sweep Bear"
            exit_reason = f"M1 {_pat} ${pnl:+.1f} (reversal — exit regardless)"
        if exit_reason:
            pass  # ไปต่อ close logic ข้างล่าง
        elif pnl <= 0.5:
            continue   # ขาดทุนเล็ก — ไม่ปิด ให้ SL/TP จัดการ (สำหรับ trigger อื่นๆ)

        # ── PATIENT EXIT — strict thresholds (2026-06-09 user: "ปิดเร็วเกิน +$17 ถูกทางอยู่") ──
        # CHOCH ปิดได้เฉพาะ pnl ≥ $10 (เคยเก็บ +$1, +$4 = sub-optimal)
        # 2026-06-12 user: "take big profit ride ต่อในเทรนแรง — ออกเมื่อ momentum/structure เปลี่ยน"
        #   เคส short @4205 ปิด +$30 ที่ 4195 (M1 เด้ง) แต่ราคาลงต่อ 4188 = ออกเร็วไป
        #   แก้: TP เฉพาะตอน M5 เริ่มอ่อน (short: higher-low / long: lower-high) ไม่ใช่แค่ M1 เด้งเดียว
        #   เทรนแรง (M5 ยังทำ new extreme) → ride ต่อ ให้ trailing SL + CHOCH/IMB คุมตอนกลับตัว
        _m5_weak_for_long  = len(m5_highs) >= 2 and m5_highs[-1] < m5_highs[-2]   # uptrend ทำ lower-high = อ่อน
        _m5_weak_for_short = len(m5_lows) >= 2 and m5_lows[-1] > m5_lows[-2]      # downtrend ทำ higher-low = อ่อน
        if direction == "long":
            if _choch_bear and pnl >= 10.0:
                exit_reason = f"M5 CHOCH BEAR +${pnl:.1f}"
            elif pnl >= 25.0 and last_bar_bearish_strong and _m5_weak_for_long:
                exit_reason = f"take big profit +${pnl:.1f} (M5 lower-high)"
            # 2026-06-18 user: BIG WINNER LONG ($50+) → ปิดเร็วบน M1 lower-high
            # 2026-06-18 user "Morning/Evening Star ที่ M1 ออกเลย — ขาดทุนได้เลย": close ทันทีบน signal ไม่ดู pnl
            elif _m1_evening_star or _m1_liq_sweep_bear:
                _pat = "Evening Star" if _m1_evening_star else "Liq Sweep Bear"
                exit_reason = f"M1 {_pat} ${pnl:+.1f} (reversal signal — exit regardless)"
            elif pnl >= 50.0 and last_bar_bearish_strong and _m1_lower_high:
                exit_reason = f"BIG WINNER take profit +${pnl:.1f} (M1 lower-high)"
        elif direction == "short":
            if _choch_bull and pnl >= 10.0:
                exit_reason = f"M5 CHOCH BULL +${pnl:.1f}"
            elif pnl >= 25.0 and last_bar_bullish_strong and _m5_weak_for_short:
                exit_reason = f"take big profit +${pnl:.1f} (M5 higher-low)"
            # 2026-06-18 user: BIG WINNER ($50+) → ปิดเร็วบน M1 reversal (ไม่ต้องรอ M5)
            # 2026-06-18 user mirror: close SHORT บน Morning Star/Liq Sweep bullish — exit regardless of pnl
            elif _m1_morning_star or _m1_liq_sweep_bull:
                _pat = "Morning Star" if _m1_morning_star else "Liq Sweep Bull"
                exit_reason = f"M1 {_pat} ${pnl:+.1f} (reversal signal — exit regardless)"
            elif pnl >= 50.0 and last_bar_bullish_strong and _m1_higher_low:
                exit_reason = f"BIG WINNER take profit +${pnl:.1f} (M1 higher-low)"

        if exit_reason:
            close_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
            close_price = tick.bid if pos.type == 0 else tick.ask
            close_req = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": pos.volume,
                "type": close_type,
                "position": pos.ticket,
                "price": close_price,
                "deviation": 30,
                "magic": pos.magic,
                "comment": f"MX:{exit_reason}"[:29],  # MT5 comment max 29 ตัว (เกิน=order_send คืน None)
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }
            result = mt5.order_send(close_req)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"📉 [MOMENTUM-EXIT] {direction.upper()} #{pos.ticket} +${pnl:.2f} — {exit_reason}", flush=True)
                _LAST_CLOSE_TIME = dt.datetime.now()  # anti-churn: ตั้ง cooldown ทันที
            else:
                err = result.retcode if result else "None"
                print(f"[MOM-EXIT-FAIL] #{pos.ticket}: {err}", flush=True)


def smart_exit_check(symbol):
    """v2: ตรวจทุก open position → ถ้าราคาชน Demand/Supply zone แล้ว bounce → ปิด

    ใช้ zone เดียวกับ Entry system (DSF logic) ไม่ใช่ H1 swing
    เช็คทั้ง M15 + M5 zones, bounce detection บน M5 + M1

    กฎ:
    1. หา Demand/Supply zones จาก M15 + M5 (same logic as DSF entry)
    2. SELL + ราคาชน Demand zone + bounce (M5/M1) → ปิด
    3. BUY + ราคาชน Supply zone + bounce (M5/M1) → ปิด
    4. ปิดเฉพาะเมื่อ position กำไร > 0 (ไม่ปิดขาดทุน — ให้ SL จัดการ)
    """
    global _LAST_CLOSE_TIME
    # ── DISABLED 2026-06-03 (user: "ปิดเร็วตอนตามเทรน") ──────────────────
    # smart_exit ปิด short ตามเทรนที่ demand bounce เล็กๆ (+$14-18 ถือ 5min) ในขาลง = ตัดเทรนทิ้ง
    # ตอนนี้ trend-lock เข้ม = ทุกไม้ตามเทรน → ปล่อยให้วิ่งถึง TP/CHOCH/trailing ไม่ปิดที่ bounce เล็ก
    # exit ที่เหลือ: TP(opposing zone) + momentum_exit(CHOCH+big profit≥$25) + trailing SL
    # ถ้าอยากเปิดกลับ: ลบ return บรรทัดล่าง
    return

    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return

    # หา zones จากทั้ง M15 และ M5 — เหมือนที่ Entry system ใช้
    zones_m15, atr_m15 = find_demand_supply_zones(symbol, mt5.TIMEFRAME_M15, 60)
    zones_m5, atr_m5   = find_demand_supply_zones(symbol, mt5.TIMEFRAME_M5, 60)

    all_zones = zones_m15 + zones_m5
    if not all_zones:
        return

    # ใช้ ATR M15 เป็น reference (เสถียรกว่า M5)
    atr_ref = atr_m15 if atr_m15 > 0 else atr_m5
    if atr_ref <= 0:
        return

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return

    for pos in positions:
        direction = "long" if pos.type == 0 else "short"
        pnl = pos.profit

        # PATIENT (user 2026-06-03): take profit ที่โซนเป้าเฉพาะกำไรมีนัย ≥$12
        # (เดิม pnl>0 → ปิด +$0.76 ที่ engulf เดี่ยว = เร็วเกิน). โซนเป้า = TP เก็บกำไรจริง
        if pnl < 12.0:
            continue

        cur_price = tick.bid if direction == "long" else tick.ask

        for zone in all_zones:
            # SELL → กลัว Demand zone ข้างล่าง
            if direction == "short" and zone["type"] == "DEMAND" and zone["hi"] < pos.price_open:
                # ราคาอยู่ใกล้ zone? (proximity 1.5 ATR เหนือ zone top)
                if cur_price <= zone["hi"] + 1.5 * atr_ref and cur_price >= zone["lo"] - 0.5 * atr_ref:
                    bounced, reason = detect_zone_bounce(
                        symbol, zone["lo"], zone["hi"], "short", atr_ref)
                    if bounced:
                        close_req = {
                            "action": mt5.TRADE_ACTION_DEAL,
                            "symbol": symbol,
                            "volume": pos.volume,
                            "type": mt5.ORDER_TYPE_BUY,
                            "position": pos.ticket,
                            "price": tick.ask,
                            "deviation": 30,
                            "magic": pos.magic,
                            "comment": f"SX:{reason}"[:29],  # MT5 comment max 29 ตัว
                            "type_time": mt5.ORDER_TIME_GTC,
                            "type_filling": mt5.ORDER_FILLING_FOK,
                        }
                        result = mt5.order_send(close_req)
                        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                            print(f"\U0001f514 [SMART-EXIT] SELL #{pos.ticket} +${pnl:.2f} -- {reason}", flush=True)
                            _LAST_CLOSE_TIME = dt.datetime.now()  # anti-churn
                            # จำไว้เพื่อ Re-entry: ปิด SHORT ที่ Demand → รอ SHORT ใหม่ที่ Supply
                            SMART_EXIT_MEMORY.append({
                                "direction": "short",
                                "exit_price": tick.ask,
                                "entry_price": pos.price_open,
                                "exit_time": dt.datetime.now(),
                                "profit": pnl,
                                "bounce_zone": {"lo": zone["lo"], "hi": zone["hi"]},
                                "used": False,
                            })
                            print(f"   [RE-MEM] จำ SHORT exit @ Demand {zone['lo']:.0f}-{zone['hi']:.0f} → รอ re-enter ที่ Supply", flush=True)
                        else:
                            err = result.retcode if result else "None"
                            print(f"❌ [SE-FAIL] #{pos.ticket}: retcode={err}", flush=True)
                        break  # ปิดแค่ 1 position ต่อ 1 check

            # BUY → กลัว Supply zone ข้างบน
            elif direction == "long" and zone["type"] == "SUPPLY" and zone["lo"] > pos.price_open:
                if cur_price >= zone["lo"] - 1.5 * atr_ref and cur_price <= zone["hi"] + 0.5 * atr_ref:
                    bounced, reason = detect_zone_bounce(
                        symbol, zone["lo"], zone["hi"], "long", atr_ref)
                    if bounced:
                        close_req = {
                            "action": mt5.TRADE_ACTION_DEAL,
                            "symbol": symbol,
                            "volume": pos.volume,
                            "type": mt5.ORDER_TYPE_SELL,
                            "position": pos.ticket,
                            "price": tick.bid,
                            "deviation": 30,
                            "magic": pos.magic,
                            "comment": f"SX:{reason}"[:29],  # MT5 comment max 29 ตัว
                            "type_time": mt5.ORDER_TIME_GTC,
                            "type_filling": mt5.ORDER_FILLING_FOK,
                        }
                        result = mt5.order_send(close_req)
                        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                            print(f"\U0001f514 [SMART-EXIT] BUY #{pos.ticket} +${pnl:.2f} -- {reason}", flush=True)
                            _LAST_CLOSE_TIME = dt.datetime.now()  # anti-churn
                            # จำไว้เพื่อ Re-entry: ปิด LONG ที่ Supply → รอ LONG ใหม่ที่ Demand
                            SMART_EXIT_MEMORY.append({
                                "direction": "long",
                                "exit_price": tick.bid,
                                "entry_price": pos.price_open,
                                "exit_time": dt.datetime.now(),
                                "profit": pnl,
                                "bounce_zone": {"lo": zone["lo"], "hi": zone["hi"]},
                                "used": False,
                            })
                            print(f"   [RE-MEM] จำ LONG exit @ Supply {zone['lo']:.0f}-{zone['hi']:.0f} → รอ re-enter ที่ Demand", flush=True)
                        else:
                            err = result.retcode if result else "None"
                            print(f"❌ [SE-FAIL] #{pos.ticket}: retcode={err}", flush=True)
                        break


# ──────────────────────────────────────────────────────────────────
# Smart Re-entry — ปิดที่ zone bounce → รอเข้าใหม่ที่ zone ตรงข้าม
#  ตามชาร์ต: Smart Exit ปิด SHORT ที่ Demand bounce (+$33)
#  → ราคาขึ้นกลับไป Supply → Sell รอบ 2 (SL แคบ $2.5, RR 12x)
#  → ราคาทะลุ Demand → TP ที่ CHOCH Demand ฝั่งซ้าย (+$31)
#  = รวม $64 > TP เดิม $33  (ด้วย risk เท่ากันได้ 4.5x)
# ──────────────────────────────────────────────────────────────────

def smart_reentry_check(symbol, max_lot=0.05):
    """Smart Re-entry: หลัง Smart Exit ปิดตอน zone bounce → รอราคากลับ zone ตรงข้าม → เข้าใหม่

    Logic (จากชาร์ต user):
    1. Smart Exit ปิด SHORT ที่ Demand bounce → จำไว้ใน SMART_EXIT_MEMORY
    2. รอราคากลับขึ้นไปที่ Supply zone → เจอ rejection candle
    3. SHORT ใหม่: SL แคบ (เหนือ Supply $1-2) + lot ใหญ่ขึ้น (same risk)
    4. TP = Demand zone ถัดลงไป (structure target = CHOCH level)
    5. RR 5-12x เพราะ SL แคบมาก

    Mirror สำหรับ LONG:
    1. Smart Exit ปิด LONG ที่ Supply bounce
    2. รอราคาลงมา Demand zone → rejection
    3. LONG ใหม่: SL ใต้ Demand + lot ใหญ่ขึ้น
    4. TP = Supply zone ถัดขึ้นไป

    Returns: dict with re-entry signal or None
    """
    global SMART_EXIT_MEMORY

    # ── DISABLED 2026-06-03 (user: ลด churn) ──────────────────────────────
    # smart_reentry ยิง 25 ไม้/วัน = มากสุด แต่ไม่ใช่ strategy ของ user + เลี่ยง guard = churn
    # detector ปกติ (zone_retest/sr/htf_retest) จับ re-entry setup เดียวกันได้อยู่แล้ว (ผ่าน guard ครบ)
    # ถ้าอยากเปิดกลับ: ลบ return None บรรทัดล่าง
    return None

    if not SMART_EXIT_MEMORY:
        return None

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return None

    # ลบ memory เก่ากว่า 4 ชม. (structure อาจเปลี่ยนแล้ว)
    now = dt.datetime.now()
    SMART_EXIT_MEMORY = [m for m in SMART_EXIT_MEMORY
                         if (now - m["exit_time"]).total_seconds() < 4 * 3600]

    if not SMART_EXIT_MEMORY:
        return None

    # หา zones ปัจจุบัน (เดียวกับ Entry + Smart Exit)
    zones_m15, atr_m15 = find_demand_supply_zones(symbol, mt5.TIMEFRAME_M15, 60)
    zones_m5, atr_m5   = find_demand_supply_zones(symbol, mt5.TIMEFRAME_M5, 60)
    all_zones = zones_m15 + zones_m5
    atr_ref = atr_m15 if atr_m15 > 0 else atr_m5

    if not all_zones or atr_ref <= 0:
        return None

    # ดึง M1 candles สำหรับ rejection check
    m1 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 5)
    if m1 is None or len(m1) < 3:
        return None

    for mem in SMART_EXIT_MEMORY:
        if mem.get("used"):
            continue

        # ━━━ SHORT Re-entry: ปิด SHORT ที่ Demand → รอ SHORT ใหม่ที่ Supply ━━━
        if mem["direction"] == "short":
            cur_price = tick.bid

            for zone in all_zones:
                if zone["type"] != "SUPPLY":
                    continue

                # ราคาอยู่ใน Supply zone? (±0.3 ATR tolerance)
                if not (zone["lo"] - 0.3 * atr_ref <= cur_price <= zone["hi"] + 0.5 * atr_ref):
                    continue

                # ต้องมี bearish rejection บน M1 (แท่งล่าสุดที่ปิดแล้ว)
                last = m1[-2]
                c_close = float(last[4]); c_open = float(last[1])
                c_high = float(last[2]); c_low = float(last[3])
                body = c_close - c_open
                candle_range = c_high - c_low

                # แท่งแดง (bearish)
                if body >= 0 or candle_range == 0:
                    continue

                # upper wick reject จาก Supply zone
                upper_wick = c_high - max(c_open, c_close)
                has_rejection = upper_wick > 0.3 * candle_range
                has_body = abs(body) > 0.08 * atr_ref  # body มีขนาดพอ

                if not (has_rejection or has_body):
                    continue

                # SL = เหนือ Supply zone + buffer $1
                sl = zone["hi"] + 1.0
                sl_dist = sl - cur_price
                if sl_dist <= 0.5:  # SL แคบเกินไป
                    continue

                # TP = nearest Demand zone ข้างล่าง (structure target)
                demand_below = sorted(
                    [z for z in all_zones if z["type"] == "DEMAND" and z["hi"] < cur_price - 2.0],
                    key=lambda z: z["hi"], reverse=True)

                if not demand_below:
                    continue

                tp = demand_below[0]["lo"]  # Demand zone bottom
                tp_dist = cur_price - tp

                # RR ต้อง >= 3 (re-entry ต้อง high-quality)
                if tp_dist < 3.0 * sl_dist:
                    continue

                rr = tp_dist / sl_dist

                # Lot: risk-adjusted — SL แคบ → lot ใหญ่ขึ้น (same $ risk)
                # Base risk = 1 ATR × 0.01 lot × contract_size
                # contract_size: XAU=100, EUR=100000 → use MT5 info for accuracy
                _sym_info = mt5.symbol_info(symbol)
                _contract = float(_sym_info.trade_contract_size) if _sym_info else 100.0
                base_risk_usd = atr_ref * 0.01 * _contract  # risk ของ 0.01 lot ที่ 1 ATR
                target_risk = base_risk_usd * 3.0  # ยอมเสี่ยง ~3x base risk
                lot = target_risk / (sl_dist * _contract)
                lot = max(0.01, min(round(lot, 2), max_lot))

                mem["used"] = True
                return {
                    "direction": "short",
                    "strategy": "reentry_short",
                    "sl": sl, "tp": tp, "lot": lot, "rr": rr,
                    "zone_hi": zone["hi"], "zone_lo": zone["lo"],
                    "atr": atr_ref,
                    "reason": (f"Smart Re-entry SHORT @ Supply {zone['lo']:.0f}-{zone['hi']:.0f} "
                               f"SL={sl:.1f}(${sl_dist:.1f}) TP={tp:.1f}(${tp_dist:.1f}) "
                               f"RR={rr:.1f} lot={lot}")
                }

        # ━━━ LONG Re-entry: ปิด LONG ที่ Supply → รอ LONG ใหม่ที่ Demand ━━━
        elif mem["direction"] == "long":
            cur_price = tick.ask

            for zone in all_zones:
                if zone["type"] != "DEMAND":
                    continue

                if not (zone["lo"] - 0.5 * atr_ref <= cur_price <= zone["hi"] + 0.3 * atr_ref):
                    continue

                last = m1[-2]
                c_close = float(last[4]); c_open = float(last[1])
                c_high = float(last[2]); c_low = float(last[3])
                body = c_close - c_open
                candle_range = c_high - c_low

                # แท่งเขียว (bullish)
                if body <= 0 or candle_range == 0:
                    continue

                lower_wick = min(c_open, c_close) - c_low
                has_rejection = lower_wick > 0.3 * candle_range
                has_body = body > 0.08 * atr_ref

                if not (has_rejection or has_body):
                    continue

                # SL = ใต้ Demand zone - buffer $1
                sl = zone["lo"] - 1.0
                sl_dist = cur_price - sl
                if sl_dist <= 0.5:
                    continue

                # TP = nearest Supply zone ข้างบน
                supply_above = sorted(
                    [z for z in all_zones if z["type"] == "SUPPLY" and z["lo"] > cur_price + 2.0],
                    key=lambda z: z["lo"])

                if not supply_above:
                    continue

                tp = supply_above[0]["hi"]  # Supply zone top
                tp_dist = tp - cur_price

                if tp_dist < 3.0 * sl_dist:
                    continue

                rr = tp_dist / sl_dist

                base_risk_usd = atr_ref * 0.01 * 100
                target_risk = base_risk_usd * 3.0
                lot = target_risk / (sl_dist * 100)
                lot = max(0.01, min(round(lot, 2), max_lot))

                mem["used"] = True
                return {
                    "direction": "long",
                    "strategy": "reentry_long",
                    "sl": sl, "tp": tp, "lot": lot, "rr": rr,
                    "zone_lo": zone["lo"], "zone_hi": zone["hi"],
                    "atr": atr_ref,
                    "reason": (f"Smart Re-entry LONG @ Demand {zone['lo']:.0f}-{zone['hi']:.0f} "
                               f"SL={sl:.1f}(${sl_dist:.1f}) TP={tp:.1f}(${tp_dist:.1f}) "
                               f"RR={rr:.1f} lot={lot}")
                }

    return None


def submit_reentry_order(symbol, reentry, max_lot=0.05):
    """ส่ง Re-entry order — ใช้ SL/TP/lot ที่คำนวณมาแล้ว (ไม่ผ่าน submit_order ปกติ)"""
    tick = mt5.symbol_info_tick(symbol)
    info = mt5.symbol_info(symbol)
    if tick is None or info is None:
        return {"ok": False, "error": "no tick"}

    d = info.digits
    step = info.volume_step
    direction = reentry["direction"]
    strategy = reentry["strategy"]
    magic = MAGIC_MAP.get(strategy, 2017)

    lot = reentry["lot"]
    lot = round(round(lot / step) * step, 8)
    lot = max(lot, info.volume_min)
    lot = min(lot, min(info.volume_max, max_lot))

    if direction == "short":
        otype = mt5.ORDER_TYPE_SELL
        price = tick.bid
    else:
        otype = mt5.ORDER_TYPE_BUY
        price = tick.ask

    req = {
        "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol,
        "volume": lot, "type": otype,
        "price": round(price, d),
        "sl": round(reentry["sl"], d),
        "tp": round(reentry["tp"], d),
        "deviation": 20, "magic": magic,
        "comment": f"Spec_{strategy}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }
    r = mt5.order_send(req)
    if r is None or r.retcode != mt5.TRADE_RETCODE_DONE:
        return {"ok": False, "error": f"{r.retcode if r else 'None'} {r.comment if r else mt5.last_error()}"}
    return {"ok": True, "order_id": r.order, "entry": r.price, "lot": lot,
            "sl": reentry["sl"], "tp": reentry["tp"]}


def main():
    global _LAST_CLOSE_TIME, _PREV_OPEN_COUNT, _LAST_TUNER_UPDATE  # anti-churn + auto-tuner
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--live", action="store_true", help="Submit real orders to MT5")
    parser.add_argument("--symbol", default="XAUUSD.iux")
    parser.add_argument("--max-lot", type=float, default=0.05)
    args = parser.parse_args()

    print("=" * 70)
    print(" 🎯 SPECIALIST BOT — 5 rule-based strategies for XAU M15")
    print("=" * 70)
    print(" Strategies: Bull Engulf, Inside Bar Break, S/R LONG, S/R SHORT, Demand LONG")
    print(" AGI Overlay: SKIP long in R{3,5,6}, SKIP short in R{3,6}")
    print(" Kelly: 2x boost in R1 golden regime")
    print("=" * 70)

    log_path = LIVE_LOGS_DIR / f"specialist_decisions_{dt.date.today().isoformat()}.csv"
    classifier = load_regime_classifier()
    if classifier:
        print(f"Loaded AGI classifier ({classifier['k']} regimes)")

    with MT5Bridge() as bridge:
        ai = bridge.account_info()
        print(f"Account: {ai['login']} on {ai['server']} (${ai['balance']:.2f})")
        print(f"Logging to: {log_path}\n")

        if args.once:
            dec = make_decision(bridge, classifier)
            print(f"[{dec['decision_time']}] live ${dec['live_bid']:.2f}")
            print(f"  Strategy: {dec['strategy']}")
            print(f"  MT touches: L={dec['mt_long']} S={dec['mt_short']}")
            print(f"  AGI Regime: R{dec['agi_regime']}  Vetoed: {dec['vetoed']}")
            print(f"  Reason: {dec['reason']}")
            print(f"  >>> {dec['final_action']} {dec['lot']:.3f} lot <<<")
            log_decision(log_path, dec)
        elif args.loop:
            print(f"LOOP mode — PRIMARY=M1, symbol={args.symbol}, max_lot={args.max_lot}, live={args.live}\n", flush=True)
            last_bar = None
            while True:
                # 2026-06-17: resilience — กัน MT5 IPC error crash บอท (re-apply หลัง revert Jun12)
                try:
                    has_new, latest = bridge.is_new_bar("M1", last_bar)  # check every M1 bar (1 min)
                except Exception as _ipc_err:
                    print(f"⚠️ [LOOP-ERR is_new_bar] {type(_ipc_err).__name__}: {_ipc_err} — reconnect+retry 15s", flush=True)
                    try:
                        mt5.initialize()
                    except Exception:
                        pass
                    time.sleep(15)
                    continue
                if has_new:
                    last_bar = latest
                    try:
                        dec = make_decision(bridge, classifier)
                    except Exception as _md_err:
                        print(f"⚠️ [LOOP-ERR make_decision] {type(_md_err).__name__}: {_md_err} — reconnect+retry 15s", flush=True)
                        try:
                            mt5.initialize()
                        except Exception:
                            pass
                        time.sleep(15)
                        continue

                    # ── Anti-Churn: ตรวจ position count ลดลง = เพิ่งมีปิด → ตั้ง cooldown ──
                    try:
                        _now_open = mt5.positions_get(symbol=args.symbol)
                        _now_n = len(_now_open) if _now_open else 0
                        if _now_n < _PREV_OPEN_COUNT:
                            _LAST_CLOSE_TIME = dt.datetime.now()
                        _PREV_OPEN_COUNT = _now_n
                    except Exception:
                        pass

                    # ── CIRCUIT BREAKER: SL รัวๆ ใน whipsaw → หยุดเข้าชั่วคราว (ยังปิด position ได้) (2026-06-12) ──
                    _cb_active = (args.live and dec['final_action'] in ("LONG", "SHORT") and not dec['vetoed']
                                  and _whipsaw_circuit_breaker(args.symbol))
                    if _cb_active:
                        print(f"🛑 [{dec['decision_time']}] CIRCUIT-BREAKER: SL รัว ≥3 ไม้/15min — หยุดเข้าใหม่ ({dec['strategy']} {dec['final_action']}) รอ storm ผ่าน", flush=True)

                    # Submit order if --live and have signal
                    if args.live and dec['final_action'] in ("LONG", "SHORT") and not dec['vetoed'] and not _cb_active:
                        strategy = dec['strategy']
                        magic = MAGIC_MAP.get(strategy, 2099)
                        direction = dec['final_action'].lower()
                        # ── Position Management: max 3 + smart conflict resolution ──
                        xau_positions = mt5.positions_get(symbol=args.symbol)
                        n_pos = len(xau_positions) if xau_positions else 0

                        # Step 1: Max position limit
                        # 2026-06-11 user: "ออเดอร์ไหนที่เปิดอยู่หากเกิด strategy ทิศทางเดียวกันเข้าซ้ำได้เลย"
                        # ถ้า position เดิมทิศเดียวกัน → allow stack สูงสุด 3
                        # ถ้ามี position ตรงข้ามอยู่ → cap 2 เหมือนเดิม (flip_losing_positions จัดการ)
                        _all_same_dir = all(
                            (p.type == 0 and direction == "long") or
                            (p.type == 1 and direction == "short")
                            for p in (xau_positions or [])
                        ) if n_pos > 0 else True
                        _max_cap = 3 if _all_same_dir else 2
                        if n_pos >= _max_cap:
                            print(f"[MAX-POS] {n_pos} positions open (cap {_max_cap}{'=same-dir' if _all_same_dir else ''}), skip ({strategy} {direction})", flush=True)

                        else:
                            # Step 2: Stacking — 2026-06-09 user: "ถ้ามี strategy ทิศทางเดียวกันก็เปิดซ้ำได้"
                            # 2026-06-18 user: "ซ้ำหรือใกล้ที่เดิมก็ได้ถ้ามั่นใจ แต่ไม่เกิน 3" → ลด $2 → $0 (= ไม่มี min distance)
                            #   MAX_POS=3 (same dir) คุมเพดานแล้ว + signal cooldown ต่อ strategy คุม spam
                            _stack_min = 0.0
                            _cur_px = float(dec.get('live_bid', 0) or 0)
                            _same_dir_near = [
                                p for p in (xau_positions or [])
                                if ((p.type == 0 and direction == "long") or
                                    (p.type == 1 and direction == "short"))
                                and abs(p.price_open - _cur_px) < _stack_min
                            ]
                            if _same_dir_near:
                                _exist_px = _same_dir_near[0].price_open
                                print(f"⏳ [{dec['decision_time']}] skip stack — {direction} pos @{_exist_px:.1f} ห่าง ${abs(_exist_px-_cur_px):.1f} <${_stack_min:.0f} (รอราคาขยับ)", flush=True)
                            else:
                                # Step 3: Smart Conflict Resolution
                                conflict = has_conflicting_position(args.symbol, direction)
                                can_enter = True
                                if conflict:
                                    flipped = flip_losing_positions(args.symbol, direction, dec)
                                    if flipped:
                                        can_enter = True
                                    else:
                                        can_enter = False
                                        opp = "SELL" if direction == "long" else "BUY"
                                        struct_key = "structure_long" if direction == "long" else "structure_short"
                                        struct_val = dec.get(struct_key, 0)
                                        h1_choch = dec.get("h1_choch", "NONE")
                                        print(f"🚫 [{dec['decision_time']}] Signal {dec['final_action']} BLOCKED — "
                                              f"conflict {opp} (struct={struct_val}, CHOCH={h1_choch})", flush=True)

                                # ── Anti-Whipsaw: ห้าม flip ทิศที่ราคาใกล้เดิมภายใน 15 นาที ──
                                # จากข้อมูล 6/2: buy@4520 → sell@4514 ใน 4 นาที = whipsaw เสียทั้งคู่
                                if can_enter:
                                    _lf = _LAST_FILLED
                                    if (_lf['dir'] is not None and _lf['dir'] != direction
                                            and _lf['time'] is not None
                                            and (dt.datetime.now() - _lf['time']).total_seconds() < 900
                                            and abs(dec['live_bid'] - _lf['price']) < 5.0):
                                        print(f"🔄 [{dec['decision_time']}] ANTI-WHIPSAW: เพิ่ง {_lf['dir'].upper()} @{_lf['price']:.1f} → ห้าม {direction.upper()} ใกล้กัน (chop)", flush=True)
                                        can_enter = False

                                # ── Anti-Churn Cooldown: บังคับใช้ทุก strategy (VIP bypass = spam disaster) ──
                                # 2026-06-08 user: "บอทเข้ามั่ว" — verified hl_retest spam 9 ครั้งใน 9 นาที
                                if can_enter and _in_entry_cooldown():
                                    _wait = ENTRY_COOLDOWN_SEC - (dt.datetime.now() - _LAST_CLOSE_TIME).total_seconds()
                                    print(f"⏸️ [{dec['decision_time']}] COOLDOWN: เพิ่งปิด position รออีก {_wait:.0f}s (กัน churn) — skip {direction.upper()}", flush=True)
                                    can_enter = False

                                # ── Auto-Tuner: combo (strategy×regime) ขาดทุนซ้ำ → skip (เรียนจากผลเทรด) ──
                                if can_enter and auto_tuner is not None and not _AUTO_TUNER_OFF:
                                    _reg = dec.get('regime', '')
                                    # Auto-tuner: ทุก strategy ผูก (VIP bypass = ไม่เรียนรู้ = ขาดทุนซ้ำ)
                                    # ยกเว้น break_retest (proven winner historical +$155 WR 77%) — อย่าให้ noise blocking
                                    _AT_EXEMPT = {"break_retest_long", "break_retest_short"}
                                    try:
                                        if strategy not in _AT_EXEMPT and auto_tuner.is_blocked(strategy, _reg):
                                            print(f"🧠 [{dec['decision_time']}] AUTO-TUNER skip: {strategy} [{_reg}] ขาดทุนซ้ำ (บอทเรียนเอง)", flush=True)
                                            can_enter = False
                                    except Exception:
                                        pass

                                if can_enter:
                                    atr_val = dec.get('atr', 0)
                                    lot_to_use = min(dec['lot'], args.max_lot)
                                    # ── Counter-trend detection ──
                                    m15t = dec.get('m15_trend', '')
                                    is_counter = (
                                        (direction == "long" and m15t == "BEAR") or
                                        (direction == "short" and m15t == "BULL")
                                    )
                                    # M5-reversal = confirmed reversal (M5 momentum + structure break)
                                    # → ไม่ใช่ blind counter-trend → ใช้ RR เต็ม (TP ใหญ่ขึ้น)
                                    if dec.get('m5_reversal'):
                                        is_counter = False
                                    # ── Signal Cooldown: strategy เดียวกันต้องรอ cooldown ก่อน fire ใหม่ ──
                                    # 2026-06-08 user: "บอทเข้ามั่ว" — hl_retest ยิง 9 ครั้งใน 9 นาที!
                                    # bug: _SIGNAL_COOLDOWN_SECS define แต่ไม่ enforce → ตรวจตรงนี้
                                    _cd_key_chk = (strategy, direction)
                                    _cd_secs = _SIGNAL_COOLDOWN_SECS.get(strategy, 600)  # default 10 min
                                    _last_fire = _SIGNAL_LAST_FIRE.get(_cd_key_chk)
                                    if _last_fire and (dt.datetime.now() - _last_fire).total_seconds() < _cd_secs:
                                        _remain = _cd_secs - (dt.datetime.now() - _last_fire).total_seconds()
                                        print(f"⏱️ [{dec['decision_time']}] SIG-COOLDOWN: {strategy} {direction} เพิ่งยิงไป รออีก {_remain:.0f}s", flush=True)
                                        continue
                                    # HTF TP cap: ถ้ามี H1/H4 zone ขวาง → cap TP ที่ zone boundary
                                    _htf_cap = dec.get('htf_tp_cap')
                                    res = submit_order(args.symbol, direction, lot_to_use, atr_val,
                                                       magic, comment=f"Spec_{strategy}",
                                                       zone_hi=dec.get('zone_hi'),
                                                       zone_lo=dec.get('zone_lo'),
                                                       max_lot=args.max_lot,
                                                       counter_trend=is_counter,
                                                       htf_tp_cap=_htf_cap)
                                    if res['ok']:
                                        _cd_key = (strategy, direction)
                                        _SIGNAL_LAST_FIRE[_cd_key] = dt.datetime.now()
                                        _LAST_FILLED['dir'] = direction
                                        _LAST_FILLED['price'] = res['entry']
                                        _LAST_FILLED['time'] = dt.datetime.now()
                                        # Auto-Tuner: บันทึก ticket→(strategy,regime) เพื่อเรียนจาก P/L ภายหลัง
                                        if auto_tuner is not None:
                                            try:
                                                auto_tuner.record_entry(res['order_id'], strategy, dec.get('regime', ''))
                                            except Exception:
                                                pass
                                        print(f"✅ [{dec['decision_time']}] ORDER: {direction.upper()} {res['lot']} lot @ {res['entry']:.5f} (magic {magic}, strategy {strategy})", flush=True)
                                        df_m5 = bridge.fetch_bars(timeframe="M5", n_bars=500)
                                        chart_path = save_entry_chart(
                                            df_m5.reset_index(drop=True), res['entry'],
                                            direction, strategy, res['order_id'], atr_val,
                                            symbol=args.symbol
                                        )
                                        print(f"   📸 chart: {chart_path}", flush=True)
                                        log_trade_result(
                                            order_id=res['order_id'], strategy=strategy,
                                            direction=direction, entry_price=res['entry'],
                                            lot=res['lot'], sl_price=res['sl'], tp_price=res['tp'],
                                            chart_path=chart_path, entry_time=dec['decision_time']
                                        )
                                        update_closed_trades()
                                    else:
                                        print(f"❌ [{dec['decision_time']}] ORDER FAILED: {res['error']}", flush=True)
                        update_closed_trades()  # check ทุก bar ว่ามี position ปิดแล้วไหม

                    marker = "🎯" if dec['final_action'] != "HOLD" else "·"
                    print(f"{marker} [{dec['decision_time']}] ${dec['live_bid']:.5f} "
                          f"[{dec['strategy']}] → {dec['final_action']}", flush=True)
                    log_decision(log_path, dec)

                    # ── Breakeven + Smart Exit + Momentum Exit: เช็คทุก M1 bar ──
                    if args.live:
                        try:
                            breakeven_check(args.symbol)
                        except Exception as e:
                            print(f"[BE-ERR] breakeven error: {e}", flush=True)
                        try:
                            momentum_exit_check(args.symbol)
                        except Exception as e:
                            print(f"[ME-ERR] momentum_exit error: {e}", flush=True)
                        try:
                            smart_exit_check(args.symbol)
                        except Exception as e:
                            print(f"[SE-ERR] smart_exit error: {e}", flush=True)
                        # ── Smart Re-entry: เช็คหลัง Smart Exit ปิด position ──
                        try:
                            reentry = smart_reentry_check(args.symbol, args.max_lot)
                            if reentry and _in_entry_cooldown():
                                print(f"   [RE-ENTRY] skip — COOLDOWN กัน churn (เพิ่งปิด position)", flush=True)
                                reentry = None
                            if reentry:
                                res = submit_reentry_order(args.symbol, reentry, args.max_lot)
                                if res["ok"]:
                                    print(f"   [RE-ENTRY] {reentry['reason']}", flush=True)
                                    print(f"   [RE-ENTRY] order={res['order_id']} entry={res['entry']} "
                                          f"lot={res['lot']} SL={res['sl']} TP={res['tp']}", flush=True)
                                else:
                                    print(f"   [RE-ENTRY FAIL] {reentry['reason']} → {res['error']}", flush=True)
                        except Exception as e:
                            print(f"[RE-ERR] reentry error: {e}", flush=True)

                        # ── Auto-Tuner: เรียนจากผลเทรดเองทุก ~30 นาที (self-learning) ──
                        if auto_tuner is not None and not _AUTO_TUNER_OFF:
                            try:
                                _now_t = dt.datetime.now()
                                if (_LAST_TUNER_UPDATE is None or
                                        (_now_t - _LAST_TUNER_UPDATE).total_seconds() > 1800):
                                    _bl = auto_tuner.update_blocklist(args.symbol)
                                    _LAST_TUNER_UPDATE = _now_t
                                    if _bl:
                                        print(f"🧠 AUTO-TUNER เรียนรู้: ปิด {[(b['strategy'], b['regime']) for b in _bl]} (ขาดทุนซ้ำ)", flush=True)
                            except Exception as _e:
                                print(f"[TUNER-ERR] {_e}", flush=True)
                else:
                    time.sleep(10)   # M1 polling — เช็คทุก 10s ให้ทัน M1 bar ใหม่
        else:
            print("Use --once or --loop")


if __name__ == "__main__":
    main()
