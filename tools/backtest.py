"""Backtest harness — replay specialist_bot.make_decision on historical data.

WHY: หยุดเดาจากภาพเดี่ยว → วัดผลรวมจริงก่อน deploy (user 2026-06-13).
HOW: pull historical bars จาก MT5 → mock mt5.* ให้คืนข้อมูล "as of" เวลา sim (NO lookahead)
     → step ทีละ M1 bar → call make_decision (logic จริง) → จำลอง entry + SL/TP → สรุปผล

VALIDATION: รันบนวันที่มี trade_results จริง → เทียบ signal/ทิศ/เวลา ว่าใกล้ของจริงไหม
LIMITATION v1: exit = SL/TP เท่านั้น (ไม่รวม momentum_exit/trailing); auto_tuner/circuit_breaker ปิด
               → เหมาะสำหรับ "relative comparison" (เทียบ baseline vs change) ไม่ใช่ทำนาย P&L เป๊ะ

USAGE: python tools/backtest.py --start 2026-06-11 --end 2026-06-13 [--symbol XAUUSD.iux]
"""
import argparse
import datetime as dt
import sys
import io
import numpy as np

import MetaTrader5 as mt5

# MT5 rates dtype (ตรงกับที่ copy_rates_from_pos คืน)
_RATES_DTYPE = np.dtype([
    ("time", "<i8"), ("open", "<f8"), ("high", "<f8"), ("low", "<f8"),
    ("close", "<f8"), ("tick_volume", "<u8"), ("spread", "<i4"), ("real_volume", "<u8"),
])

_TF_NAME = {mt5.TIMEFRAME_M1: "M1", mt5.TIMEFRAME_M5: "M5", mt5.TIMEFRAME_M15: "M15",
            mt5.TIMEFRAME_H1: "H1", mt5.TIMEFRAME_H4: "H4"}
_TF_SECS = {mt5.TIMEFRAME_M1: 60, mt5.TIMEFRAME_M5: 300, mt5.TIMEFRAME_M15: 900,
            mt5.TIMEFRAME_H1: 3600, mt5.TIMEFRAME_H4: 14400}

# ── Sim state (global, ใช้ใน mock) ─────────────────────────────────────────
_SIM = {
    "raw": {},        # tf -> structured array (full history, sorted by time)
    "now": 0,         # current sim epoch (วินาที) = เวลาปิด M1 bar ปัจจุบัน
    "positions": [],  # list ของ dict: {ticket,type,price_open,sl,tp,magic,comment,volume,time,strategy}
    "closed": [],     # trades ที่ momentum/smart exit ปิด (ผ่าน mock order_send)
    "regime_map": [], # {strategy,regime,pnl,exit_ep} ของไม้ปิด → ใช้ sim auto_tuner
    "next_ticket": 1000,
    "symbol": "XAUUSD.iux",
    "digits": 2,
    "point": 0.01,
}


class _Pos:
    """mock position object (มี attribute เหมือน mt5 position)."""
    __slots__ = ("ticket", "type", "price_open", "sl", "tp", "magic", "comment",
                 "volume", "time", "profit", "strategy")
    def __init__(self, **k):
        for key, v in k.items():
            setattr(self, key, v)


class _Tick:
    __slots__ = ("time", "bid", "ask", "last", "volume")
    def __init__(self, t, bid, ask):
        self.time = t; self.bid = bid; self.ask = ask; self.last = bid; self.volume = 0


class _SymInfo:
    digits = 2; point = 0.01; spread = 2
    volume_min = 0.01; volume_max = 50.0; volume_step = 0.01
    trade_contract_size = 100.0; currency_profit = "USD"


# ── M1 → reconstruct forming bar ของ TF อื่น (NO lookahead) ────────────────
def _bars_as_of(tf, n, now_epoch):
    """คืน structured array ของ tf, n bars ล่าสุด ที่ "เห็นได้" ณ now_epoch.
    forming bar (bar ที่ยังไม่ปิด) สร้างจาก M1 จนถึง now → ไม่ใช้ข้อมูลอนาคต.
    """
    raw = _SIM["raw"].get(tf)
    if raw is None or len(raw) == 0:
        return None
    secs = _TF_SECS[tf]
    # closed bars: bar ที่ปิดแล้ว (start + secs <= now)
    closed_mask = (raw["time"] + secs) <= now_epoch
    closed = raw[closed_mask]
    # forming bar: bar ปัจจุบัน (start <= now < start+secs) — สร้างจาก M1
    cur_start = (now_epoch // secs) * secs
    out = closed
    if tf != mt5.TIMEFRAME_M1:
        m1 = _SIM["raw"].get(mt5.TIMEFRAME_M1)
        if m1 is not None:
            seg = m1[(m1["time"] >= cur_start) & (m1["time"] <= now_epoch)]
            if len(seg) > 0:
                forming = np.zeros(1, dtype=_RATES_DTYPE)
                forming["time"] = cur_start
                forming["open"] = seg["open"][0]
                forming["high"] = seg["high"].max()
                forming["low"] = seg["low"].min()
                forming["close"] = seg["close"][-1]
                forming["tick_volume"] = seg["tick_volume"].sum()
                out = np.concatenate([closed, forming]) if len(closed) else forming
    else:
        # M1: bar ที่ปิด ณ now (time <= now ที่ time+60<=now ... จริงๆ ใช้ time<=now)
        out = raw[raw["time"] <= now_epoch]
    if len(out) == 0:
        return None
    return out[-n:].copy()


# ── Mocks ───────────────────────────────────────────────────────────────────
def _mock_copy_rates_from_pos(symbol, tf, pos, n):
    return _bars_as_of(tf, n, _SIM["now"])

def _mock_copy_rates_from(symbol, tf, end_time, n):
    ep = int(end_time.timestamp()) if isinstance(end_time, dt.datetime) else int(end_time)
    return _bars_as_of(tf, n, min(ep, _SIM["now"]))

def _cur_m1_close():
    m1 = _SIM["raw"][mt5.TIMEFRAME_M1]
    sub = m1[m1["time"] <= _SIM["now"]]
    return float(sub["close"][-1]) if len(sub) else 0.0

def _mock_symbol_info_tick(symbol):
    px = _cur_m1_close()
    if px <= 0:
        return None
    sp = _SIM["point"] * _SymInfo.spread
    return _Tick(_SIM["now"], bid=px, ask=px + sp)

def _mock_positions_get(symbol=None):
    px = _cur_m1_close()
    out = []
    for p in _SIM["positions"]:
        # update floating profit
        diff = (px - p["price_open"]) if p["type"] == 0 else (p["price_open"] - px)
        prof = diff * p["volume"] * _SymInfo.trade_contract_size
        out.append(_Pos(ticket=p["ticket"], type=p["type"], price_open=p["price_open"],
                        sl=p["sl"], tp=p["tp"], magic=p["magic"], comment=p["comment"],
                        volume=p["volume"], time=p["time"], profit=prof, strategy=p["strategy"]))
    return tuple(out)

def _mock_symbol_info(symbol):
    return _SymInfo()

def _mock_order_send(req):
    r = type("R", (), {})()
    r.retcode = mt5.TRADE_RETCODE_DONE
    r.volume = req.get("volume", 0.05)
    r.comment = "sim"
    # CLOSE order (momentum/smart exit มี "position": ticket) → ปิด sim position + บันทึก
    pos_tk = req.get("position")
    if pos_tk:
        price = req.get("price", _cur_m1_close())
        for p in list(_SIM["positions"]):
            if p["ticket"] == pos_tk:
                diff = (price - p["price_open"]) if p["type"] == 0 else (p["price_open"] - price)
                pnl = diff * p["volume"] * _SymInfo.trade_contract_size
                _SIM["closed"].append({**p, "exit": price, "exit_ep": _SIM["now"], "pnl": pnl,
                                       "result": "WIN" if pnl > 0 else "LOSS"})
                _SIM["positions"].remove(p)
                _SIM["last_close_ep"] = _SIM["now"]
                break
        r.order = pos_tk
        return r
    # ENTRY probe (submit_order ใช้เพื่อคืน SL/TP) — ไม่เปิด position เอง (run loop เปิด)
    r.order = _SIM["next_ticket"]; _SIM["next_ticket"] += 1
    r.price = req.get("price", _cur_m1_close())
    return r

def _mock_history_deals_get(*a, **k):
    return ()   # ปิด auto_tuner/circuit_breaker ใน v1

def _noop(*a, **k):
    return True


# ── auto_tuner sim (ตรงกับ tools/auto_tuner.py: MIN_TRADES=4, cut=-3, window=4d) ──
_AT_MIN_TRADES = 4
_AT_EXPECTANCY_CUT = -3.0
_AT_WINDOW_SEC = 4 * 86400

def _sim_is_blocked(strategy, regime, now_ep):
    rel = [t for t in _SIM["regime_map"]
           if t["strategy"] == strategy and t["regime"] == regime
           and (now_ep - t["exit_ep"]) <= _AT_WINDOW_SEC]
    if len(rel) < _AT_MIN_TRADES:
        return False
    return (sum(t["pnl"] for t in rel) / len(rel)) < _AT_EXPECTANCY_CUT

def _record_regime(p, pnl):
    _SIM["regime_map"].append({"strategy": p["strategy"], "regime": p.get("entry_regime", "chop"),
                               "pnl": pnl, "exit_ep": _SIM["now"]})


# ── Pull data ────────────────────────────────────────────────────────────────
def pull_data(symbol, start, end, warmup_days=3):
    if not mt5.initialize():
        print("MT5 init failed"); sys.exit(1)
    s = start - dt.timedelta(days=warmup_days)
    for tf in (mt5.TIMEFRAME_M1, mt5.TIMEFRAME_M5, mt5.TIMEFRAME_M15,
               mt5.TIMEFRAME_H1, mt5.TIMEFRAME_H4):
        rates = mt5.copy_rates_range(symbol, tf, s, end)
        if rates is None or len(rates) == 0:
            print(f"  WARN no data for {_TF_NAME[tf]}"); _SIM["raw"][tf] = np.zeros(0, _RATES_DTYPE); continue
        arr = np.zeros(len(rates), dtype=_RATES_DTYPE)
        for f in ("time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"):
            if f in rates.dtype.names:
                arr[f] = rates[f]
        arr = np.sort(arr, order="time")
        _SIM["raw"][tf] = arr
        print(f"  {_TF_NAME[tf]}: {len(arr)} bars  {dt.datetime.fromtimestamp(int(arr['time'][0]))} → {dt.datetime.fromtimestamp(int(arr['time'][-1]))}")
    info = mt5.symbol_info(symbol)
    if info:
        _SIM["digits"] = info.digits; _SIM["point"] = info.point
        _SymInfo.digits = info.digits; _SymInfo.point = info.point
        _SymInfo.trade_contract_size = getattr(info, "trade_contract_size", 100.0)
    mt5.shutdown()


def install_mocks(sb):
    """patch mt5 functions (sb.mt5 = MetaTrader5 module เดียวกัน)."""
    for mod in (mt5,):
        mod.copy_rates_from_pos = _mock_copy_rates_from_pos
        mod.copy_rates_from = _mock_copy_rates_from
        mod.symbol_info_tick = _mock_symbol_info_tick
        mod.positions_get = _mock_positions_get
        mod.symbol_info = _mock_symbol_info
        mod.order_send = _mock_order_send
        mod.history_deals_get = _mock_history_deals_get
        mod.initialize = lambda *a, **k: True
        mod.shutdown = lambda *a, **k: True
        mod.symbol_select = lambda *a, **k: True
        mod.last_error = lambda: (0, "ok")
    # auto_tuner (ถ้า import) — ปิด
    try:
        sb.auto_tuner.is_blocked = lambda *a, **k: False
        sb.auto_tuner.record_entry = lambda *a, **k: None
        sb.auto_tuner.update_blocklist = lambda *a, **k: None
    except Exception:
        pass


# ── Run ───────────────────────────────────────────────────────────────────────
def run(symbol, start, end, max_pos=2, cooldown_sec=150):
    import importlib
    sb = importlib.import_module("tools.specialist_bot") if "tools" in sys.modules else None
    if sb is None:
        sys.path.insert(0, ".")
        import tools.specialist_bot as sb
    install_mocks(sb)
    classifier = None  # AGI overlay off ใน backtest (consistent)

    bridge = sb.MT5Bridge.__new__(sb.MT5Bridge)
    bridge.symbol = symbol

    m1 = _SIM["raw"][mt5.TIMEFRAME_M1]
    start_ep = int(start.timestamp()); end_ep = int(end.timestamp())
    steps = m1[(m1["time"] >= start_ep) & (m1["time"] <= end_ep)]
    print(f"\nReplaying {len(steps)} M1 bars ({start} → {end})...\n")

    trades = []   # closed trades
    _SIM["last_close_ep"] = 0

    for i, bar in enumerate(steps):
        _SIM["now"] = int(bar["time"]) + 60   # เวลา "ปิด" M1 bar นี้
        bar_hi, bar_lo, bar_cl = float(bar["high"]), float(bar["low"]), float(bar["close"])

        # 1) check SL/TP ของ position ที่เปิดอยู่ (ใช้ high/low ของ M1 bar นี้)
        still_open = []
        for p in _SIM["positions"]:
            hit = None; exitpx = None
            if p["type"] == 0:  # long
                if bar_lo <= p["sl"]: hit, exitpx = "SL", p["sl"]
                elif bar_hi >= p["tp"]: hit, exitpx = "TP", p["tp"]
            else:               # short
                if bar_hi >= p["sl"]: hit, exitpx = "SL", p["sl"]
                elif bar_lo <= p["tp"]: hit, exitpx = "TP", p["tp"]
            if hit:
                diff = (exitpx - p["price_open"]) if p["type"] == 0 else (p["price_open"] - exitpx)
                pnl = diff * p["volume"] * _SymInfo.trade_contract_size
                trades.append({**p, "exit": exitpx, "exit_ep": _SIM["now"], "pnl": pnl, "result": "WIN" if pnl > 0 else "LOSS"})
                _record_regime(p, pnl)
                _SIM["last_close_ep"] = _SIM["now"]
            else:
                still_open.append(p)
        _SIM["positions"] = still_open

        # 1b) v2 smart-exit DISABLED — ต้อง sim auto_tuner ด้วย ไม่งั้น over-state (future work)
        #     sb.momentum_exit_check(symbol); sb.smart_exit_check(symbol)
        last_close_ep = _SIM["last_close_ep"]
        # 2) decision
        try:
            dec = sb.make_decision(bridge, classifier)
        except Exception as e:
            continue
        if dec.get("vetoed") or dec.get("final_action") not in ("LONG", "SHORT"):
            continue

        direction = dec["final_action"].lower()
        strat = dec["strategy"]
        cur_regime = dec.get("regime", "chop")
        # auto_tuner sim: block combo ที่ขาดทุนซ้ำ (เหมือน live)
        if _sim_is_blocked(strat, cur_regime, _SIM["now"]):
            continue
        # position mgmt (เบื้องต้น): cooldown + MAX_POS (same-dir cap 3) + stacking $2
        if (_SIM["now"] - last_close_ep) < cooldown_sec:
            continue
        same_dir = sum(1 for p in _SIM["positions"]
                       if (p["type"] == 0) == (direction == "long"))
        opp = sum(1 for p in _SIM["positions"] if (p["type"] == 0) != (direction == "long"))
        cap = 3 if opp == 0 else 2
        if len(_SIM["positions"]) >= cap:
            continue
        # stacking distance $2 same dir
        near = any(abs(p["price_open"] - bar_cl) < 2.0 for p in _SIM["positions"]
                   if (p["type"] == 0) == (direction == "long"))
        if near:
            continue

        # 3) SL/TP via submit_order จริง (mock order_send)
        magic = sb.MAGIC_MAP.get(strat, 2099)
        atr_val = float(dec.get("atr", 0) or 0)
        res = sb.submit_order(symbol, direction, 0.05, atr_val, magic, comment=strat,
                              zone_hi=dec.get("zone_hi"), zone_lo=dec.get("zone_lo"),
                              htf_tp_cap=dec.get("htf_tp_cap"))
        if not res.get("ok"):
            continue
        _SIM["positions"].append({
            "ticket": res.get("order_id", _SIM["next_ticket"]), "type": 0 if direction == "long" else 1,
            "price_open": res["entry"], "sl": res["sl"], "tp": res["tp"], "magic": magic,
            "comment": strat, "volume": 0.05, "time": _SIM["now"], "strategy": strat,
            "entry_regime": cur_regime,
        })

    # close remaining at last price (mark-to-market)
    px = _cur_m1_close()
    for p in _SIM["positions"]:
        diff = (px - p["price_open"]) if p["type"] == 0 else (p["price_open"] - px)
        pnl = diff * p["volume"] * _SymInfo.trade_contract_size
        trades.append({**p, "exit": px, "exit_ep": _SIM["now"], "pnl": pnl, "result": "OPEN-MTM"})

    return trades


def report(trades):
    from collections import defaultdict
    closed = [t for t in trades if t["result"] in ("WIN", "LOSS")]
    print("=" * 60)
    if not closed:
        print("ไม่มีไม้ปิด"); return
    w = sum(1 for t in closed if t["result"] == "WIN")
    tot = sum(t["pnl"] for t in closed)
    print(f"TOTAL: {w}W/{len(closed)-w}L  WR {w/len(closed)*100:.0f}%  P/L ${tot:+.2f}  ({len(closed)} trades)")
    print("-" * 60)
    s = defaultdict(lambda: [0, 0, 0.0])
    for t in closed:
        st = t["strategy"]
        s[st][0 if t["result"] == "WIN" else 1] += 1
        s[st][2] += t["pnl"]
    print("Per strategy:")
    for st in sorted(s, key=lambda x: s[x][2], reverse=True):
        a = s[st]
        print(f"  {st:24s} {a[0]:2d}W/{a[1]:2d}L  ${a[2]:+8.2f}")
    print("-" * 60)
    print("By day:")
    byday = defaultdict(lambda: [0, 0, 0.0])
    for t in closed:
        d = dt.datetime.fromtimestamp(t["time"]).strftime("%Y-%m-%d")
        byday[d][0 if t["result"] == "WIN" else 1] += 1
        byday[d][2] += t["pnl"]
    for d in sorted(byday):
        a = byday[d]
        print(f"  {d}  {a[0]:2d}W/{a[1]:2d}L  ${a[2]:+8.2f}")


if __name__ == "__main__":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="XAUUSD.iux")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    args = ap.parse_args()
    start = dt.datetime.strptime(args.start, "%Y-%m-%d")
    end = dt.datetime.strptime(args.end, "%Y-%m-%d") + dt.timedelta(days=1)
    _SIM["symbol"] = args.symbol
    print(f"Pulling data {args.symbol} {start} → {end}...")
    pull_data(args.symbol, start, end)
    trades = run(args.symbol, start, end)
    report(trades)
