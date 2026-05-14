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
    "rsi",
    "sma_ratio",
    "macd_diff",
    "bb_position",
    "atr_ratio",
    # Price action: Demand/Supply zones (institutional order flow)
    "demand_active",
    "supply_active",
    "ds_distance",
    "ds_strength",
    "ds_freshness",
    # Price action: Fair Value Gap (kept — v2 proved it helps as context)
    "fvg_bull_active",
    "fvg_bear_active",
    "fvg_distance",
    "fvg_strength",
    # Market context
    "is_active_session",
    "candle_body_ratio",
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


def _compute_demand_supply_features(df: pd.DataFrame, atr_values: pd.Series) -> pd.DataFrame:
    """
    Compute Demand/Supply zone features for the RL agent.

    Demand/Supply zones (Smart Money Concepts / Institutional Order Flow):
      - Demand zone: the last BEARISH candle before a strong BULLISH move
        (= institutional buy orders were filled here)
      - Supply zone: the last BULLISH candle before a strong BEARISH move
        (= institutional sell orders were filled here)

    A "strong move" = candle body > MOVE_THRESHOLD * ATR

    Features:
      - demand_active:  1.0 if there's an active demand zone nearby
      - supply_active:  1.0 if there's an active supply zone nearby
      - ds_distance:    signed distance from close to nearest zone / ATR
      - ds_strength:    strength of the move that created the zone (normalized 0-1)
      - ds_freshness:   1.0 = untested, decays with each test (fresh zones = stronger)

    Key differences from FVG:
      - D/S zones represent real institutional order flow, not just wick gaps
      - D/S zones are LESS frequent = more selective signal
      - Freshness tracking = zone quality degrades with each touch
      - Longer max age (50 candles vs 20) = institutional memory persists
    """
    MOVE_THRESHOLD = 1.0   # strong candle = body > 1.0 * ATR
    MAX_ZONE_AGE = 50      # zones expire after 50 candles (~12.5h on M15)
    MAX_TESTS = 3           # zone depleted after 3 touches

    n = len(df)
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    atr = atr_values.values

    demand_act = np.zeros(n, dtype=np.float32)
    supply_act = np.zeros(n, dtype=np.float32)
    ds_dist = np.zeros(n, dtype=np.float32)
    ds_str = np.zeros(n, dtype=np.float32)
    ds_fresh = np.zeros(n, dtype=np.float32)

    # Active zones: (zone_low, zone_high, direction, birth_idx, test_count, strength)
    active_zones: list = []

    for i in range(2, n):
        cur_atr = atr[i] if not np.isnan(atr[i]) and atr[i] > 0 else 1e-9

        # --- Detect new zones at candle i ---
        body = closes[i] - opens[i]  # positive = bullish
        body_size = abs(body)

        if body_size > MOVE_THRESHOLD * cur_atr:
            if body > 0:
                # Strong bullish candle -> demand zone from preceding bearish candle
                for j in range(i - 1, max(i - 5, -1), -1):
                    if j < 0:
                        break
                    if closes[j] < opens[j]:  # bearish candle = demand zone base
                        zone_low = lows[j]
                        zone_high = max(opens[j], closes[j])
                        strength = body_size / cur_atr
                        active_zones.append((zone_low, zone_high, 1, i, 0, strength))
                        break
            else:
                # Strong bearish candle -> supply zone from preceding bullish candle
                for j in range(i - 1, max(i - 5, -1), -1):
                    if j < 0:
                        break
                    if closes[j] > opens[j]:  # bullish candle = supply zone base
                        zone_low = min(opens[j], closes[j])
                        zone_high = highs[j]
                        strength = body_size / cur_atr
                        active_zones.append((zone_low, zone_high, -1, i, 0, strength))
                        break

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
        use_lstm: bool = False,
        use_action_mask: bool = False,      # True = hard rule-based entry filter
        mask_trend_threshold: float = 0.3,  # min |d_trend_strength| for valid entry
        mask_zone_threshold: float = 1.5,   # max |ds_distance| for valid entry
        mask_require_session: bool = True,  # require active session for entry
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

        Setup conditions (ALL must be met for entry to be allowed):
          1. Strong trend: |d_trend_strength| > mask_trend_threshold
          2. D/S zone nearby: (demand/supply active) AND |ds_distance| < mask_zone_threshold
          3. Active session: is_active_session = 1  (if mask_require_session=True)

        Direction aligned:
          - Bullish trend → only Buy allowed
          - Bearish trend → only Sell allowed

        If already in position → only Hold (no pyramiding)
        If action masking disabled → all actions valid
        """
        if not self.use_action_mask:
            return np.array([True, True, True], dtype=bool)

        # In position → only hold
        if self.position != 0:
            return np.array([True, False, False], dtype=bool)

        step = self.current_step

        # --- Condition 1: Trend (multi-bar confirmation) ---
        # Use average of last 4 bars to avoid single-bar d_trend flips
        # This fixes fold 2: gold rallying but 1-bar MACD dip was triggering SHORT
        trend = 0.0
        if "d_trend_strength" in self.df.columns:
            lookback = min(4, step)
            trend = float(self.df["d_trend_strength"].iloc[max(0, step-lookback):step+1].mean())
        trend_bullish = trend > self.mask_trend_threshold
        trend_bearish = trend < -self.mask_trend_threshold
        if not (trend_bullish or trend_bearish):
            return np.array([True, False, False], dtype=bool)  # no clear trend

        # Extra: M15 SMA cross check — must agree with daily trend
        # If M15 SMA20 > SMA50 but daily says bearish → skip (conflicting signal)
        if "sma_ratio" in self.df.columns:
            sma_ratio = float(self.df["sma_ratio"].iloc[step])
            m15_bullish = sma_ratio > 1.0   # M15 SMA20 > SMA50
            m15_bearish = sma_ratio < 1.0
            # Block if M15 and daily trend strongly disagree
            if trend_bullish and m15_bearish and sma_ratio < 0.998:
                return np.array([True, False, False], dtype=bool)
            if trend_bearish and m15_bullish and sma_ratio > 1.002:
                return np.array([True, False, False], dtype=bool)

        # --- Condition 2: D/S zone nearby & aligned ---
        zone_buy = False
        zone_sell = False
        if "demand_active" in self.df.columns:
            demand = float(self.df["demand_active"].iloc[step])
            supply = float(self.df["supply_active"].iloc[step])
            ds_dist = abs(float(self.df["ds_distance"].iloc[step]))
            near = ds_dist < self.mask_zone_threshold
            zone_buy = (demand > 0.5) and near
            zone_sell = (supply > 0.5) and near

        if not (zone_buy or zone_sell):
            return np.array([True, False, False], dtype=bool)  # no zone

        # --- Condition 3: Active session ---
        if self.mask_require_session and "is_active_session" in self.df.columns:
            is_active = float(self.df["is_active_session"].iloc[step]) > 0.5
            if not is_active:
                return np.array([True, False, False], dtype=bool)  # quiet session

        # --- Condition 4: Confirmation candle ---
        # Price must be moving IN the direction of the trade right now
        # Prevents entering when bar is running the opposite way (59% of bad trades)
        confirmed = True
        if "candle_body_ratio" in self.df.columns:
            body_ratio = float(self.df["candle_body_ratio"].iloc[step])
            close_val  = float(self.df["close"].iloc[step])
            open_val   = float(self.df["open"].iloc[step])
            bullish_bar = (close_val > open_val) and (body_ratio > 0.4)
            bearish_bar = (close_val < open_val) and (body_ratio > 0.4)

            if trend_bullish and not bullish_bar:
                confirmed = False   # trying to go long on a bearish/doji bar
            if trend_bearish and not bearish_bar:
                confirmed = False   # trying to go short on a bullish/doji bar

        if not confirmed:
            return np.array([True, False, False], dtype=bool)

        # --- All conditions met: BUY = "trade with trend" (direction-invariant) ---
        can_buy = (trend_bullish and zone_buy) or (trend_bearish and zone_sell)
        can_sell = False

        return np.array([True, can_buy, can_sell], dtype=bool)

    # ------------------------------------------------------------------
    # Direction-Invariant Helpers
    # ------------------------------------------------------------------
    def _get_trend_direction(self) -> int:
        """Returns +1 if bullish, -1 if bearish, 0 if neutral."""
        if "d_trend_strength" in self.df.columns:
            t = float(self.df["d_trend_strength"].iloc[self.current_step])
            if t > self.mask_trend_threshold:
                return 1
            if t < -self.mask_trend_threshold:
                return -1
        return 0

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
