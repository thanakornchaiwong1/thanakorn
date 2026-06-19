import sys, io, datetime as dt
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
import MetaTrader5 as mt5
mt5.initialize(); sym='XAUUSD.iux'
# จับคู่ open→close เพื่อดู direction + pnl + duration
since=dt.datetime.now()-dt.timedelta(hours=16)
deals=mt5.history_deals_get(since, dt.datetime.now()+dt.timedelta(minutes=2)) or []
ds=[d for d in deals if d.symbol==sym]
# group by position_id
from collections import defaultdict
pos=defaultdict(list)
for d in ds: pos[d.position_id].append(d)
trades=[]
for pid,dl in pos.items():
    dl.sort(key=lambda x:x.time)
    op=[d for d in dl if d.entry==0]; cl=[d for d in dl if d.entry==1]
    if not op or not cl: continue
    o=op[0]; pnl=sum(d.profit for d in cl)
    trades.append((o.time, 'BUY' if o.type==0 else 'SELL', o.price, o.comment, pnl, dl[-1].time))
trades.sort()
def th(ep): return (dt.datetime.fromtimestamp(ep)+dt.timedelta(hours=6)).strftime("%m-%d %H:%M")  # Thai = MT5+6h
print(f"=== เทรดเมื่อคืน (16h, เวลาไทย) — {len(trades)} ไม้ ===")
nl=sum(p for _,_,_,_,p,_ in trades); w=sum(1 for *_,p,_ in trades if p>0)
buys=[t for t in trades if t[1]=='BUY']; sells=[t for t in trades if t[1]=='SELL']
print(f"รวม ${nl:+.2f} | {w}W/{len(trades)-w}L | BUY {len(buys)} ไม้ (${sum(t[4] for t in buys):+.0f}) | SELL {len(sells)} ไม้ (${sum(t[4] for t in sells):+.0f})")
print(f"\n{'เปิด(ไทย)':12s} {'dir':4s} {'ราคา':>8s} {'pnl':>6s} strategy")
for ot,side,pr,cm,pnl,ct in trades:
    print(f"{th(ot):12s} {side:4s} {pr:>8.2f} {pnl:>+6.1f} {cm}")
mt5.shutdown()
