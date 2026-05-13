"""
Export XAUUSD M15 + Daily data from MT5 for scalping bot training.

Usage:
    python -m tools.mt5_export
    python -m tools.mt5_export --timeframe M5
    python -m tools.mt5_export --bars 100000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd


SYMBOL = "XAUUSD.iux"

TF_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}


def export_mt5(symbol: str, timeframe: str, max_bars: int, out_dir: Path):
    """Download OHLCV from MT5 and save as CSV."""
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

    info = mt5.symbol_info(symbol)
    if info is None:
        mt5.shutdown()
        raise ValueError(f"Symbol '{symbol}' not found in MT5")

    tf_val = TF_MAP[timeframe]
    print(f"Downloading {symbol} {timeframe} (max {max_bars:,} bars)...")

    rates = mt5.copy_rates_from_pos(symbol, tf_val, 0, max_bars)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No data returned for {symbol} {timeframe}")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.rename(columns={"tick_volume": "volume"})
    df = df[["time", "open", "high", "low", "close", "volume", "spread"]]

    first = df["time"].iloc[0]
    last = df["time"].iloc[-1]
    days = (last - first).days

    print(f"  Rows: {len(df):,}")
    print(f"  From: {first}")
    print(f"  To:   {last}")
    print(f"  Span: {days} days (~{days/30:.0f} months)")

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"xauusd_{timeframe.lower()}.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    return csv_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--timeframe", default="M15", choices=list(TF_MAP.keys()))
    parser.add_argument("--bars", type=int, default=200000)
    parser.add_argument("--out-dir", default="./data")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    # Export primary timeframe
    export_mt5(args.symbol, args.timeframe, args.bars, out_dir)

    # Always also export Daily for macro features
    if args.timeframe != "D1":
        print()
        export_mt5(args.symbol, "D1", 5000, out_dir)


if __name__ == "__main__":
    main()
