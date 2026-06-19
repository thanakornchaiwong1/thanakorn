"""Batch experiment runner — ทดสอบ system-level changes vs baseline (autonomous)."""
import sys, io, os, datetime as dt
from collections import defaultdict
sys.path.insert(0, os.getcwd())
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import tools.backtest as bt

START = dt.datetime(2026, 6, 9)
END = dt.datetime(2026, 6, 14)
SYM = "XAUUSD.iux"
bt._SIM["symbol"] = SYM
print(f"Pulling {SYM} {START}..{END}")
bt.pull_data(SYM, START, END)

RESULTS = []

def reset():
    bt._SIM["positions"] = []; bt._SIM["closed"] = []; bt._SIM["regime_map"] = []
    bt._SIM["next_ticket"] = 1000; bt._SIM["skip"] = set()
    for k in ("f_skip_chop", "f_no_counter", "f_chop_no_counter"):
        bt._SIM[k] = False

def runexp(label, flags=None, skip=None, cooldown=150):
    reset()
    if flags:
        for k in flags: bt._SIM[k] = True
    if skip:
        bt._SIM["skip"] = set(skip)
    tr = bt.run(SYM, START, END, cooldown_sec=cooldown)
    c = [t for t in tr if t["result"] in ("WIN", "LOSS")]
    byday = defaultdict(float)
    for t in c:
        byday[dt.datetime.fromtimestamp(t["time"]).strftime("%m-%d")] += t["pnl"]
    tot = sum(t["pnl"] for t in c)
    w = sum(1 for t in c if t["result"] == "WIN")
    days = " ".join(f"{d}:{v:+.0f}" for d, v in sorted(byday.items()))
    RESULTS.append((label, tot, w, len(c), days))
    print(f"DONE: {label}: ${tot:+.0f} ({w}W/{len(c)-w}L) | {days}", flush=True)

runexp("00 BASELINE")
runexp("01 skip_chop (ไม่เทรด chop เลย)", flags=["f_skip_chop"])
runexp("02 chop_no_counter (chop เทรดตาม M15 trend)", flags=["f_chop_no_counter"])
runexp("03 no_counter (ไม่ counter-trend ทุก regime)", flags=["f_no_counter"])
runexp("04 cooldown 300 (churn น้อยลง)", cooldown=300)
runexp("05 cooldown 90 (churn มากขึ้น)", cooldown=90)

print("\n" + "=" * 70)
print("SUMMARY (vs baseline):")
base = RESULTS[0][1]
for label, tot, w, n, days in RESULTS:
    delta = tot - base
    mark = "  <-- ดีกว่า baseline" if delta > 0 and label != RESULTS[0][0] else ""
    print(f"  {label:42s} ${tot:+7.0f}  (Δ{delta:+.0f}){mark}")

# write to file
with open("live_logs/_bt_batch_results.txt", "w", encoding="utf-8") as f:
    f.write(f"Backtest batch {START.date()}..{END.date()}  (baseline=${base:+.0f})\n\n")
    for label, tot, w, n, days in RESULTS:
        f.write(f"{label:42s} ${tot:+7.0f} (Δ{tot-base:+.0f}) {w}W/{n-w}L | {days}\n")
print("\nwritten to live_logs/_bt_batch_results.txt")
