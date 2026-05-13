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
    "returns",
    "log_returns",
    "rsi",
    "sma_ratio",
    "macd_diff",
    "bb_position",
    "atr_ratio",
    # Price action: Fair Value Gap
    "fvg_bull_active",
    "fvg_bear_active",
    "fvg_distance",
    "fvg_strength",
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

    # --- Time-of-day features (cyclical encoding) ---
    # Gold has session-dependent behavior: Asian (low vol), London (high vol), NY (high vol)
    if hasattr(df.index, 'hour'):
        hour = df.index.hour + df.index.minute / 60.0
    elif "time" in df.columns:
        hour = pd.to_datetime(df["time"]).dt.hour + pd.to_datetime(df["time"]).dt.minute / 60.0
    else:
        hour = pd.Series(np.zeros(len(df)), index=df.index)

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0).astype(np.float32)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0).astype(np.float32)

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
    ):
        super().__init__()

        # Auto-detect feature columns
        if feature_columns is None:
            all_known = FEATURE_COLUMNS + DAILY_FEATURE_COLUMNS
            feature_columns = [c for c in all_known if c in df.columns]
            if not feature_columns:
                raise ValueError(
                    "DataFrame ไม่มี feature columns ที่รู้จักเลย "
                    "เรียก prepare_features(df) ก่อน"
                )

        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame missing feature columns: {missing}")
        if len(df) <= window_size + 10:
            raise ValueError(
                f"DataFrame สั้นเกินไป (len={len(df)}) "
                f"ต้องมากกว่า window_size+10={window_size+10}"
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

        # TP/SL config (ATR-based)
        self.sl_atr_mult = float(sl_atr_mult)
        self.tp_atr_mult = float(tp_atr_mult)

        # Reward fn
        self._reward_fn = get_reward_fn(reward_type)

        # Spaces: 3 actions (Hold, Buy, Sell) — no manual Close
        n_features = len(feature_columns)
        obs_dim = self.window_size * n_features + 3  # +3 = position info
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
        if action == self.BUY:
            if self.position == 0 and not in_cooldown:
                atr = float(self._atr[self.current_step])
                if atr > 0:
                    self.position = 1
                    self.entry_price = current_price + self.spread / 2.0
                    self.entry_step = self.current_step
                    self.balance -= self.commission

                    # Set TP/SL levels
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

                    # Set TP/SL levels (reversed for short)
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
    # Helpers
    # ------------------------------------------------------------------
    def _get_observation(self) -> np.ndarray:
        start = self.current_step - self.window_size
        end = self.current_step
        window = self._features[start:end].flatten()

        if self.position != 0:
            current_price = float(self._closes[self.current_step])
            unrealized_pnl_pct = (current_price - self.entry_price) / self.entry_price * self.position
            duration = (self.current_step - self.entry_step) / 100.0
        else:
            unrealized_pnl_pct = 0.0
            duration = 0.0

        position_info = np.array(
            [float(self.position), float(unrealized_pnl_pct), float(duration)],
            dtype=np.float32,
        )

        obs = np.concatenate([window.astype(np.float32), position_info])
        # safety: NaN/Inf guard
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
