# Gold Trading Bot — Project Context

> **READ THIS FIRST** — สรุป project ทั้งหมดเพื่อ Claude (ใน Claude Code) เข้าใจ context ที่สั่งสมมาจาก chat session ก่อนหน้า

## 1. เป้าหมายของ Project

- **เจ้าของ:** Thana — developer ที่ self-describe ว่า "มือใหม่ ไม่เก่ง code"
- **เป้าระยะยาว:** เอาบอทไปเทรดเงินจริง (XAUUSD/Gold)
- **เป้าระยะสั้น:** สร้างบอท RL (PPO + Stable-Baselines3) ที่ผ่าน walk-forward validation อย่าง robust ก่อนทดลองเงินจริง
- **Hardware:** Windows + GTX 1060 6GB (CUDA 12.1, ~280-700 fps สำหรับ training PPO)

## 2. Tech Stack ที่ใช้

```
Python 3.11 + venv (.venv)
torch 2.3.1+cu121 (GPU)
stable-baselines3 2.3.2
gymnasium 0.29.1
yfinance (data source — มี limitation: 1h data limit 730 วัน, ไม่รองรับ 4h native)
ta library (technical indicators)
pandas, numpy, matplotlib, tensorboard, pyyaml
```

## 3. Project Structure

```
gold-bot/
├── config.yaml                  # master config — แก้ที่นี่ที่เดียว
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── env.py                   # Gymnasium env + features (multi-tf)
│   ├── rewards.py               # 5 reward functions in registry
│   ├── data.py                  # yfinance/CSV loader + multi-tf merge
│   ├── train.py                 # PPO trainer with VecNormalize + callbacks
│   └── backtest.py              # backtest with metrics + plot
├── tools/
│   ├── check_env.py             # sanity check ก่อน train (ต้องรันก่อน train ทุกครั้ง)
│   ├── walk_forward.py          # rolling-window validation (CORE TOOL)
│   └── overfit_check.py         # เปรียบ in-sample vs out-of-sample
├── data/                        # cached OHLCV csv
├── checkpoints/                 # PPO model + VecNormalize
├── logs/                        # tensorboard
├── walk_forward_output/         # fold metrics + plots
└── backtest_output/             # backtest metrics + plots
```

## 4. ประวัติ Iterations (สำคัญที่สุด — อ่านให้ครบ)

### Initial Run: train 500k บน 1h data, multi_objective reward
- Backtest บน test set: Sharpe 3.75, return +20%, alpha +8.6%
- **มี bug ใน backtest.py** (อ่าน state หลัง VecEnv auto-reset) — ตัวเลขแรกเป็น 0 ทั้งหมด ภายหลังแก้ใน iteration 4 (ใช้ raw env แทน VecEnv ใน inference)

### Walk-Forward Iter 0 (multi_objective, lr 3e-4 → 1e-4, 1h)
| Metric | Value |
|---|---|
| Mean return | -1.13% |
| Mean Sharpe | -0.47 |
| Profitable folds | 0/3 |
| Mean alpha | -15.51% |
- **Diagnosis:** บอท over-trade รุนแรง (262 trades/fold) — multi_objective reward สอนให้ "เก็บกำไรเล็ก ๆ บ่อย ๆ" misaligned incentives

### Walk-Forward Iter 1 (เปลี่ยนเป็น differential_sharpe, 1h)
| Metric | Value |
|---|---|
| Mean return | +0.95% |
| Mean Sharpe | +0.12 |
| Profitable folds | 2/3 |
| Mean alpha | -13.43% |
| Avg trades | 128 (ลดครึ่ง) |
- **Diagnosis:** reward function เปลี่ยนถูกทาง บอทลด over-trade ครึ่งหนึ่ง แต่ยังแพ้ตลาดเพราะ 1h timeframe noise เยอะ

### Walk-Forward Iter 2 (4h timeframe — เปลี่ยน interval)
| Metric | Value |
|---|---|
| Mean return | +7.93% |
| Mean Sharpe | **+3.09** |
| Profitable folds | **3/3 ✅** |
| Mean alpha | -5.87% |
| Avg trades | 56 |
- **ก้าวกระโดดใหญ่** — แต่ alpha ยังลบ
- **Diagnosis:** บอทเก่งระดับ professional fund แต่ "ตามตลาด" ใน Fold 2 (alpha -16%) เพราะไม่มี macro context

### Walk-Forward Iter 3 (Multi-Timeframe: 4h + 1d features)
| Metric | Value |
|---|---|
| Mean return | +5.05% |
| Mean Sharpe | +1.46 |
| Profitable folds | 2/3 |
| **Folds beating B&H** | **1/3 ⭐** (ครั้งแรกที่บอทชนะตลาด) |
| Mean alpha | -8.69% |
| Avg trades | 40 |
- **Fold 3 alpha +8.43%** — บอทชนะตลาด!
- แต่ Fold 1 (ตลาด trending แรง +21%) บอทตามไม่ทัน alpha -20%
- **Diagnosis:** บอทกลายเป็น mean-reverter เก่ง แต่ตามเทรนด์แรง ๆ ไม่ได้

### Walk-Forward Iter 4 (Trend-Aware Reward — เพิ่ม `d_trend_strength` feature + reward bonus)
| Metric | Value |
|---|---|
| Mean return | +6.37% |
| Mean Sharpe | +1.80 |
| Profitable folds | 2/3 |
| Folds beating B&H | 1/3 |
| Mean alpha | -7.37% |
| Worst fold alpha | -22.88% |
- ดีขึ้นเล็กน้อย แต่ Fold 1 ยังพลาด
- **Diagnosis:** reward signal เบาเกินไป + threshold strict เกิน + ใช้ pre-computed feature แทน raw price action

### Walk-Forward Iter 4.1 (REDESIGN reward — รันแล้ว, REGRESSION → REVERTED)
**Changes vs Iter 4:**
- ใช้ raw price action (entry_price → current_price) แทน normalized feature
- Per-step continuous shaping
- Counter-trend threshold: 0.5 → **0.15** (trigger บ่อยขึ้น 3x)
- Bonus magnitude: 0.005/step → **scaled max 0.05** (10x แรง)
- เพิ่ม cut-loss penalty (ถือ losing > 20 bars)
- เพิ่ม scalping penalty (ปิด winner ใน 3 bars)

**Result (vs Iter 4):** mean return +2.96% (vs +6.37%), Sharpe 1.18 (vs 1.80),
*reported* alpha -10.79% (vs -7.37%) — REGRESSION → reverted rewards.py to Iter 4 spec

### 🚨 BENCHMARK BUG DISCOVERED (Iter 4.1 post-mortem)
`tools/walk_forward.py` เก่าใช้ `bnh_return = raw price % change` เป็น benchmark
แต่บอทเทรด 0.01 lot × 100 contract = ~0.24x leverage บน $10k account
=> bot return กับ B&H อยู่คนละ scale ห้ามเอามาลบกันตรง ๆ

**Re-scored ด้วย apples-to-apples B&H (`tools/rescore.py`):**
| Iter | Old reported alpha | NEW corrected alpha |
|---|---|---|
| Iter 4 (estimate) | -7.37% | **+1.20%** ⭐ |
| Iter 4.1 | -10.79% | -2.22% |

Per-fold Iter 4.1 corrected: F1 -5.47%, F2 -5.33%, F3 +4.14%
**Fix:** แก้ `walk_forward.py` ให้คำนวณ B&H equity-based + เพิ่ม `buy_and_hold_raw_pct` ไว้อ้างอิง
**ควรทำ:** ทุก decision criteria + เป้า alpha ต้อง rescale ใหม่ (เก่าโหดเกินไป)

### Walk-Forward Iter 4 RE-RUN (fresh, corrected B&H, seed=42)
| Metric | Value |
|---|---|
| Mean return | +4.74% |
| Mean alpha | -0.43% |
| Profitable | 3/3 |

Per-fold: F1 +1.61% (α-5.62%), F2 +4.82% (α-0.61%), F3 +7.80% (α+4.94%)
**ดูดีในตอนแรก แต่...**

### 🚨 STABILITY FAILURE (seed=7 retest of trend_aware)
| Fold | seed=42 | seed=7 |
|---|---|---|
| 1 | +1.61% (α-5.62) | +2.10% (α-5.13) |
| 2 | +4.82% (α-0.61) | +2.67% (α-2.76) |
| 3 | **+7.80%** (α+4.94) | **-11.84%** (α-14.70) 💥 |
| Mean | +4.74% / α-0.43% | -2.36% / α-7.53% |

Fold 3 swung **19.6pp** between seeds. trend_aware reward = high variance across PPO inits
**Verdict:** trend_aware NOT robust. Stress test premature

### Walk-Forward Iter 5 (differential_sharpe + multi-tf + window=20) — N=3 SEEDS
| Metric | seed=42 | seed=7 | seed=123 |
|---|---|---|---|
| Mean return | +4.02% | +3.69% | +2.15% |
| Mean alpha | -1.15% | -1.49% | -3.03% |
| Profitable | 3/3 | 3/3 | **1/3** |
| Folds beat B&H | 1/3 | 2/3 | 1/3 |
| Diagnosis | 🟡 | 🟢 | **🔴** |

Per-fold spread (high → low):
- F1 returns: +1.35, +0.70, -0.46 (1.81pp spread, mostly stable)
- F2 returns: +8.63, +5.65, +2.49 (6.14pp spread, all positive)
- F3 returns: +8.87, +4.05, **-1.72** (10.6pp spread, crosses zero)

**Verdict:** N=2 ROBUST verdict was FALSE. Third seed exposed regression — esp Fold 3 (-1.72%, DD 15%). DS less catastrophic than trend_aware (worst F3 -1.72 vs -11.84), but still seed-dependent
Across 9 fold-runs: 7/9 profitable, 4/9 beat B&H, mean alpha ~-1.9%
**Single-seed PPO not production-ready** — need ensemble or variance reduction

## 5. Config Settings ปัจจุบัน (สำคัญ — อ่านก่อนแก้!)

```yaml
data:
  source: "yfinance"
  symbol: "GC=F"
  interval: "4h"               # ใช้ 4h (resample จาก 1h ใน data.py)
  period: "2y"                 # 1h data limit 730 days
  use_multi_timeframe: true    # เพิ่ม 1d daily features (9 ตัว)
  train_test_split: 0.8

env:
  initial_balance: 10000.0
  lot_size: 0.01
  spread: 0.30
  commission: 0.07
  contract_size: 100.0
  window_size: 20              # Iter 4: ลดจาก 30 เพื่อกัน overfit
  reward_type: "trend_aware"   # ใช้ trend_aware (Iter 4/4.1)

ppo:
  learning_rate: 1.0e-4        # ลดจาก 3e-4 ใน Iter 1 (training stability)
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  ent_coef: 0.01
  net_arch: {pi: [256, 256], vf: [256, 256]}
```

**Walk-forward params ที่ใช้:**
```
--train-size 1500 --test-size 400 --step-size 400
```
ให้ 3 folds, ~15 นาที/run บน GTX 1060

## 6. Known Issues / Gotchas

### A. Encoding bug (FIXED already)
Windows ภาษาไทย (cp874) อ่าน UTF-8 ไม่ออก — ทุก `open()` ของ yaml ต้องใส่ `encoding="utf-8"`
- `tools/check_env.py`, `src/train.py`, `src/backtest.py` — แก้แล้วทุกตัว

### B. yfinance limitations
- `4h` interval **ไม่ใช่ native** ของ yfinance → download `1h` แล้ว resample เอง (logic ใน `data.py:load_yfinance`)
- `1h` data จำกัด 730 วัน (~2 ปี)
- ถ้าจะใช้ data > 2y ต้องเปลี่ยน source (Stooq, MT5 export, etc.)

### C. Look-Ahead Bias Prevention
ใน `merge_daily_into_primary` ใช้ `shift(1)` ก่อน reindex+ffill — ทุก 4h timestamp เห็นแค่ "daily bar ที่ปิดสมบูรณ์เมื่อวาน" ไม่ใช่ของวันนี้ที่ยังไม่ปิด

### D. VecEnv Auto-Reset Bug (FIXED)
Backtest ที่ใช้ DummyVecEnv → env auto-reset เมื่อ done → อ่าน equity_curve = ค่าเริ่มต้น
**Fix:** ใช้ raw GoldTradingEnv ตรง ๆ + manual normalize obs ผ่าน VecNormalize.normalize_obs

### E. config.yaml ของ user ไม่ตรงกับ default ใน skill
User edit config มาหลายรอบ (Iter 0→4) — **อย่า overwrite ทั้งไฟล์** ให้แก้ทีละบรรทัดด้วย str_replace

## 7. Next Steps (ตามที่คุยกันใน chat)

### A. Iter 4.1 (กำลังจะรัน — replace rewards.py)
**Decision criteria after run:**
- 🟢 Mean alpha > -3% → ไป Stress Test (ใช้บอทเดียว ไม่ต้อง ensemble)
- 🟡 -3% > alpha > -7% → ไป Plan B (Two-Bot Ensemble)
- 🔴 Worse than Iter 4 → revert + go straight to Plan B

### B. Plan B: Two-Bot Ensemble (ถ้าจำเป็น)
Combine **Iter 2 model** (trend follower, no multi-tf) + **Iter 4/4.1 model** (mean reverter, multi-tf):

```
                ┌──────────────────────┐
                │  REGIME DETECTOR     │
                │  ADX-based switch    │
                └──────────┬───────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
       ADX > 25 (trending)      ADX < 20 (choppy)
       Iter 2 model              Iter 4 model
       (4h, single-tf)          (4h, multi-tf)
```

**Files to build for Plan B:**
1. `tools/regime_detector.py` — คำนวณ ADX จาก daily data
2. `tools/ensemble_backtest.py` — load 2 models, switch ตาม regime
3. `tools/walk_forward_ensemble.py` — walk-forward ของ ensemble
4. Retrain Iter 2 (config: interval=4h, multi_tf=false, reward=differential_sharpe, window=30)
5. Save 2 models ในโฟลเดอร์แยก: `checkpoints/iter2_trend/`, `checkpoints/iter4_revert/`

### C. Stress Test (ถัดไปหลังจากผ่าน Iter 4.1 หรือ Plan B)
ทดสอบบอทบน gold bear market 2013-2015 — ต้อง download data ผ่าน source อื่น (yfinance ไม่มี 1h ย้อนเก่าขนาดนั้น)
- ใช้ daily data จาก yfinance (ดึงได้นาน) แล้ว resample ให้เป็น `interval` ที่ต้องการ (แต่จะได้แค่ 1d resolution → ต้อง simulate 4h ด้วย OHLC interpolation, complicated)
- หรือใช้ MT5 export แบบ manual (cleaner)

### D. After Stress Test
1. MT5 broker integration
2. Risk management layer (position sizing, daily loss limit, max DD trigger)
3. Paper trading 2-3 เดือน
4. เริ่มเงินจริง $100-500 (เงินที่เสียได้)

## 8. คำแนะนำสำหรับ Claude (Claude Code)

1. **อ่าน context นี้ก่อนทำอะไรก็ตาม** — อย่า assume ตาม code อย่างเดียว
2. **Check config.yaml ก่อนแก้** — user มีค่าที่ tune มาหลายรอบ
3. **ทุก yaml open ต้องใช้ encoding="utf-8"** — Windows Thai locale
4. **ก่อน train ทุกครั้ง** ให้ลบ checkpoints, logs, walk_forward_output (และ data ถ้าเปลี่ยน data source)
5. **ใช้ walk-forward ไม่ใช่ single backtest** ในการประเมินผล — single backtest unreliable มาก
6. **รัน `python -m tools.check_env`** ก่อน walk-forward ทุกครั้ง — verify obs shape + features
7. **Communication style:** user เป็นมือใหม่ — อธิบายเหตุผลของทุก decision ไม่ใช่แค่ action
8. **User บอก "เป้าเงินจริง"** — ต้องเตือน reality (3-6 เดือน journey, ไม่ใช่ 3-6 วัน) ไม่ให้ false confidence

## 9. คำที่ user ใช้บ่อย (จะได้ตอบได้ตรงทาง)

- "พักก่อน" / "หยุด" — ขอเวลาคิด
- "เดินหน้าต่อ" / "ลุย" / "เริ่มเลย" — go ahead
- "ผมจริงจัง" — wants serious effort, no shortcut
- "เก่ง opus 4.7" — push for quality, don't half-ass

## 10. ผลที่คาดหวัง (จาก iter 0 ถึงเป้าหมายสุดท้าย)

```
Iter 0:  alpha -15.51%  (baseline)
Iter 1:  alpha -13.43%  (-2 improvement)
Iter 2:  alpha  -5.87%  (-7 improvement)
Iter 3:  alpha  -8.69%  (regression แต่ first beating B&H 1/3)
Iter 4:  alpha  -7.37%  (ดีขึ้นเล็กน้อย)
Iter 4.1: เป้า alpha > -3%  (กำลังจะรัน)

ระยะกลาง (Plan B Ensemble): alpha 0% ถึง +5% (consistently)
ระยะไกล (Stress Test ผ่าน): alpha + ใน bear market ด้วย
ระยะสุดท้าย (Real money ready): paper trading 3 เดือน + alpha + ทุกสภาพตลาด
```
