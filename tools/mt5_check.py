"""Quick MT5 connection check and data availability scan."""
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

if not mt5.initialize():
    print(f"MT5 initialize FAILED: {mt5.last_error()}")
    exit(1)

print("MT5 connected!")
print(f"  Company: {mt5.terminal_info().company}")

acc = mt5.account_info()
if acc:
    print(f"  Account: {acc.login}")
    print(f"  Currency: {acc.currency}")

# Find gold symbols
symbols = mt5.symbols_get()
gold_syms = [s.name for s in symbols if "XAU" in s.name.upper() or "GOLD" in s.name.upper()]
print(f"\nGold symbols: {gold_syms[:15]}")

# Try common XAUUSD names
found_sym = None
for sym_name in ["XAUUSD", "XAUUSDm", "GOLD", "GOLDm", "XAUUSD.", "XAUUSD.i",
                 "XAUUSD.a", "XAUUSD#", "XAUUSDc"]:
    info = mt5.symbol_info(sym_name)
    if info:
        found_sym = sym_name
        print(f"\nFound symbol: {sym_name}")
        print(f"  Bid: {info.bid}")
        print(f"  Ask: {info.ask}")
        print(f"  Spread: {info.spread} points")
        print(f"  Point: {info.point}")
        print(f"  Contract size: {info.trade_contract_size}")
        break

if found_sym is None and gold_syms:
    found_sym = gold_syms[0]
    info = mt5.symbol_info(found_sym)
    print(f"\nUsing first gold symbol: {found_sym}")
    if info:
        print(f"  Bid: {info.bid}, Ask: {info.ask}")

if found_sym:
    # Check data availability across timeframes
    for tf_name, tf_val in [("M1", mt5.TIMEFRAME_M1), ("M5", mt5.TIMEFRAME_M5),
                             ("M15", mt5.TIMEFRAME_M15), ("H1", mt5.TIMEFRAME_H1),
                             ("H4", mt5.TIMEFRAME_H4), ("D1", mt5.TIMEFRAME_D1)]:
        rates = mt5.copy_rates_from_pos(found_sym, tf_val, 0, 200000)
        if rates is not None and len(rates) > 0:
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            first = df["time"].iloc[0]
            last = df["time"].iloc[-1]
            days = (last - first).days
            print(f"  {tf_name:>3}: {len(rates):>8,} bars | {first.date()} to {last.date()} ({days} days, ~{days/30:.0f} months)")
        else:
            print(f"  {tf_name:>3}: NO DATA")

mt5.shutdown()
print("\nDone!")
