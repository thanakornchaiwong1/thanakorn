"""
Data loading + train/test split for XAUUSD.

Sources:
  - yfinance: ฟรี ใช้ symbol "GC=F" (gold futures continuous)
  - csv: ไฟล์ที่มี columns open, high, low, close [, volume]

NOTE: train/test split ตาม "เวลา" เท่านั้น — ห้าม random shuffle
       เพราะจะทำให้ test set leak future info เข้า train set
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

try:
    from .env import (
        prepare_features,
        compute_daily_features,
        merge_daily_into_primary,
        compute_h1_features,
        merge_h1_into_primary,
    )
except ImportError:
    from env import (  # type: ignore
        prepare_features,
        compute_daily_features,
        merge_daily_into_primary,
        compute_h1_features,
        merge_h1_into_primary,
    )


def load_yfinance(symbol: str, interval: str, period: str) -> pd.DataFrame:
    """
    ดึง OHLCV จาก yfinance (handle MultiIndex columns + custom intervals via resample)

    yfinance รองรับแค่ native intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
    สำหรับ 2h/3h/4h/6h/8h/12h เรา download 1h แล้ว resample ให้เอง
    """
    import yfinance as yf

    NATIVE_INTERVALS = {
        "1m", "2m", "5m", "15m", "30m", "60m", "1h", "90m",
        "1d", "5d", "1wk", "1mo", "3mo",
    }
    # custom intervals -> download "1h" แล้ว resample
    RESAMPLE_MAP = {
        "2h": "1h", "3h": "1h", "4h": "1h",
        "6h": "1h", "8h": "1h", "12h": "1h",
    }

    if interval in NATIVE_INTERVALS:
        download_interval = interval
        resample_to = None
    elif interval in RESAMPLE_MAP:
        download_interval = RESAMPLE_MAP[interval]
        resample_to = interval
        print(f"  ({interval} ไม่ใช่ native interval ของ yfinance "
              f"-> download {download_interval} แล้ว resample เป็น {interval})")
    else:
        raise ValueError(
            f"Unsupported interval: '{interval}'. "
            f"Native: {sorted(NATIVE_INTERVALS)}, custom: {sorted(RESAMPLE_MAP.keys())}"
        )

    df = yf.download(
        symbol, period=period, interval=download_interval,
        auto_adjust=False, progress=False,
    )
    if df is None or df.empty:
        raise RuntimeError(
            f"yfinance returned empty data for symbol={symbol} "
            f"interval={download_interval} period={period}"
        )

    # yfinance >= 0.2.40 คืน MultiIndex (Price, Ticker) — flatten
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [str(c).lower() for c in df.columns]
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep].copy().dropna()

    # Resample 1h -> target interval (e.g. 4h)
    if resample_to is not None:
        agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
        if "volume" in df.columns:
            agg["volume"] = "sum"
        df = df.resample(resample_to).agg(agg).dropna()

    return df


def load_csv(path: str, parse_time: bool = False) -> pd.DataFrame:
    """
    Load OHLCV from CSV file.
    parse_time=True: parse 'time' column as datetime and set as index
                     (needed for multi-timeframe merge with daily data)
    """
    df = pd.read_csv(path)
    df.columns = [str(c).lower() for c in df.columns]
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    if parse_time and "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")
    return df


def load_data(cfg: dict) -> pd.DataFrame:
    """
    Load + prepare features ตาม config
    คืน DataFrame ที่พร้อมส่งเข้า GoldTradingEnv

    ถ้า data.use_multi_timeframe=true:
        - download primary (e.g. 4h via 1h resample)
        - download secondary (1d) -> compute daily features
        - merge daily features เข้า primary timeline ด้วย shift(1) + ffill
        -> df มีทั้ง 4h features (7 ตัว) + daily features (8 ตัว) = 15 ตัว
    มิฉะนั้น: 4h features อย่างเดียว (7 ตัว)
    """
    dcfg = cfg["data"]
    cache_path = Path(cfg["paths"]["data_cache"])

    source = dcfg["source"]
    use_mtf = dcfg.get("use_multi_timeframe", False)

    if source == "yfinance":
        print(f"  fetching {dcfg['symbol']} from yfinance "
              f"(interval={dcfg['interval']}, period={dcfg['period']})")
        df_primary_raw = load_yfinance(dcfg["symbol"], dcfg["interval"], dcfg["period"])
    elif source == "csv":
        if not dcfg.get("csv_path"):
            raise ValueError("data.csv_path required when data.source=csv")
        # parse_time=True for multi-tf merge (need datetime index)
        df_primary_raw = load_csv(dcfg["csv_path"], parse_time=use_mtf)
        print(f"  loaded CSV: {dcfg['csv_path']} ({len(df_primary_raw):,} rows)")
    else:
        raise ValueError(f"Unknown data.source: {source}")

    # cache raw OHLCV (primary)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df_primary_raw.to_csv(cache_path, index=True)

    if not use_mtf:
        # Single timeframe — existing behavior
        df = prepare_features(df_primary_raw)
        return df

    # ----- Multi-timeframe path -----
    use_h1 = dcfg.get("use_h1_from_m15", True)   # default ON: H1 from M15 resample
    use_d1 = bool(dcfg.get("csv_daily_path"))     # D1 only if csv provided

    print(f"  multi-timeframe enabled: primary={dcfg['interval']}"
          f"{', +H1 (resampled)' if use_h1 else ''}"
          f"{', +D1' if use_d1 else ''}")

    # 1) Compute primary features (keep datetime index for merge)
    df_primary_features = prepare_features(df_primary_raw, reset_index=False)
    df_merged = df_primary_features.copy()

    # 2) H1 features (resampled from M15 — timelier than D1 for scalping)
    if use_h1:
        try:
            df_h1_features = compute_h1_features(df_primary_raw)
            df_merged = merge_h1_into_primary(df_merged, df_h1_features)
            print(f"  H1 rows: {len(df_h1_features)}, merged rows after H1: {len(df_merged)}")
        except Exception as e:
            print(f"  ⚠️  H1 compute failed ({e}), skipping H1 features")

    # 3) D1 features (optional — only if csv_daily_path provided)
    if use_d1:
        daily_csv = dcfg["csv_daily_path"]
        print(f"  loading daily from CSV: {daily_csv}")
        df_daily_raw = load_csv(daily_csv, parse_time=True)
        df_daily_features = compute_daily_features(df_daily_raw)
        df_merged = merge_daily_into_primary(df_merged, df_daily_features)
        print(f"  D1 rows: {len(df_daily_features)}, merged rows after D1: {len(df_merged)}")
    elif not use_h1 and not use_d1:
        # Fallback: fetch D1 from yfinance
        print(f"  fetching daily data for macro features...")
        df_daily_raw = load_yfinance(dcfg["symbol"], "1d", dcfg["period"])
        df_daily_features = compute_daily_features(df_daily_raw)
        df_merged = merge_daily_into_primary(df_merged, df_daily_features)

    print(f"  primary rows: {len(df_primary_features)}, "
          f"total rows: {len(df_merged)}")

    return df_merged.reset_index(drop=True)


def split_train_test(df: pd.DataFrame, train_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Time-based split — train = ช่วงแรก, test = ช่วงหลัง"""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
    n = len(df)
    split = int(n * train_ratio)
    train = df.iloc[:split].reset_index(drop=True)
    test = df.iloc[split:].reset_index(drop=True)
    return train, test
