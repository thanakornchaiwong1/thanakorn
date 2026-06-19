"""
auto_tuner.py — บอทเรียนจากผลเทรดเอง (self-learning จาก P/L)

หลักการ: track expectancy per (strategy × regime) จากผลเทรดจริง (MT5 deals)
→ ถ้า combo ไหนขาดทุนซ้ำ (≥ MIN_TRADES ไม้ + expectancy < cut) → auto-block ใน regime นั้น
→ combo ที่กำไร → ปล่อยเทรดต่อ
→ rolling window (ลืมผลเก่า) ให้ strategy ฟื้นได้ถ้ากลับมากำไร

ไม่แตะ entry logic เดิม — แค่เพิ่มชั้น "เรียนรู้" บนผลลัพธ์
user 2026-06-04: "อยากให้บอทเรียนเองจากผลเทรด ไม่ต้องสอนทีละเคส"
"""
from __future__ import annotations
import os
import csv
import json
import datetime as dt
from collections import defaultdict

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "live_logs")
_MAP = os.path.join(_DIR, "trade_regime_map.csv")        # ticket → strategy, regime (ตอนเข้า)
_BLOCKLIST = os.path.join(_DIR, "auto_tuner_blocklist.json")
_LOG = os.path.join(_DIR, "auto_tuner.log")

# ── พารามิเตอร์ (ปรับได้) ──────────────────────────────────────────────
MIN_TRADES = 4          # 2026-06-05: ลด 8→4 (รัน react เร็ว — เคส -$245 ต้องรอ 8 ไม้ช้าไป)
EXPECTANCY_CUT = -3.0   # expectancy < -$3/ไม้ → block (ขาดทุนเฉลี่ยซ้ำ)
WINDOW_DAYS = 4         # rolling window — ลืมผลเก่ากว่านี้ (ให้ strategy ฟื้นได้)


# ════════════════════════════════════════════════════════════════════════
# 1. บันทึกตอนเข้า: ticket → (strategy, regime)
# ════════════════════════════════════════════════════════════════════════
def record_entry(ticket, strategy: str, regime: str):
    """เรียกตอนเข้าออเดอร์สำเร็จ — เก็บ ticket คู่กับ strategy+regime เพื่อ join P/L ภายหลัง."""
    try:
        new = not os.path.exists(_MAP)
        with open(_MAP, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["ticket", "strategy", "regime", "entry_time"])
            w.writerow([ticket, strategy, regime,
                        dt.datetime.now().isoformat(timespec="seconds")])
    except Exception:
        pass


def _load_map() -> dict:
    out = {}
    if not os.path.exists(_MAP):
        return out
    try:
        with open(_MAP, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                out[str(row["ticket"])] = (row["strategy"], row["regime"])
    except Exception:
        pass
    return out


# ════════════════════════════════════════════════════════════════════════
# 2. เรียนรู้: คิด expectancy per (strategy,regime) → เขียน blocklist + log
# ════════════════════════════════════════════════════════════════════════
def update_blocklist(symbol: str = "XAUUSD.iux") -> list:
    """อ่านผลเทรดจริง (MT5 deals) → group ตาม (strategy,regime) → block ตัวที่ขาดทุนซ้ำ.

    คืน list ของ combo ที่ block. เขียน blocklist.json + auto_tuner.log
    """
    if mt5 is None:
        return []
    tmap = _load_map()
    if not tmap:
        return []
    # ── P/L จริงต่อ position (MT5 OUT deals) ในช่วง window ──
    frm = dt.datetime.now() - dt.timedelta(days=WINDOW_DAYS)
    try:
        deals = mt5.history_deals_get(frm, dt.datetime.now() + dt.timedelta(hours=1)) or []
    except Exception:
        return []
    pl = {}
    for d in deals:
        if getattr(d, "symbol", "") == symbol and d.entry == 1:   # OUT = realized
            pl[str(d.position_id)] = pl.get(str(d.position_id), 0.0) + d.profit + d.swap + d.commission

    # ── join ticket→(strat,regime) กับ P/L → group ──
    grp = defaultdict(lambda: [0, 0.0, 0])   # [count, total_pl, wins]
    for tk, (strat, reg) in tmap.items():
        if tk in pl:
            g = grp[(strat, reg)]
            g[0] += 1
            g[1] += pl[tk]
            if pl[tk] > 0:
                g[2] += 1

    blocklist = []
    report = []
    for (strat, reg), (n, tot, wins) in sorted(grp.items()):
        exp = tot / n if n else 0.0
        wr = wins / n * 100 if n else 0
        status = "ok"
        if n >= MIN_TRADES and exp < EXPECTANCY_CUT:
            blocklist.append({"strategy": strat, "regime": reg})
            status = "BLOCKED"
        report.append(f"  {strat:22s} [{reg:10s}] {n:2d}ไม้ WR{wr:3.0f}% exp=${exp:+5.1f} tot=${tot:+6.0f} {status}")

    # ── safety: ถ้าจะ block จนเหลือ combo ที่กำไรน้อยเกินไป — ยอม block ได้ (บอท HOLD = ปลอดภัย) ──
    try:
        json.dump(blocklist, open(_BLOCKLIST, "w", encoding="utf-8"))
    except Exception:
        pass
    try:
        with open(_LOG, "a", encoding="utf-8") as f:
            f.write(f"\n[{dt.datetime.now():%Y-%m-%d %H:%M}] AUTO-TUNER "
                    f"(window {WINDOW_DAYS}d, min {MIN_TRADES} ไม้, cut ${EXPECTANCY_CUT}/ไม้):\n")
            if report:
                f.write("\n".join(report) + "\n")
            else:
                f.write("  (ยังไม่มีข้อมูลพอ — รอสะสมไม้)\n")
            if blocklist:
                f.write(f"  >>> AUTO-BLOCKED: {[(b['strategy'], b['regime']) for b in blocklist]}\n")
    except Exception:
        pass
    return blocklist


# ════════════════════════════════════════════════════════════════════════
# 3. เช็คตอนตัดสินใจ: (strategy,regime) ถูก auto-block ไหม
# ════════════════════════════════════════════════════════════════════════
_cache = None
_cache_mtime = 0.0


def is_blocked(strategy: str, regime: str) -> bool:
    """True = combo นี้ขาดทุนซ้ำ auto-tuner สั่งปิด (อ่านจาก blocklist.json, cache by mtime)."""
    global _cache, _cache_mtime
    if not os.path.exists(_BLOCKLIST):
        return False
    try:
        m = os.path.getmtime(_BLOCKLIST)
        if _cache is None or m != _cache_mtime:
            _cache = json.load(open(_BLOCKLIST, encoding="utf-8"))
            _cache_mtime = m
        return any(b.get("strategy") == strategy and b.get("regime") == regime for b in (_cache or []))
    except Exception:
        return False
