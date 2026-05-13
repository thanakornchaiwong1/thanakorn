"""
Re-score past iterations ด้วย apples-to-apples B&H benchmark

Bug ที่แก้: walk_forward.py เก่าใช้ raw price % change เป็น B&H
ซึ่งสมมติ leverage 100% notional แต่บอทเทรด 0.01 lot = ~0.24x leverage บน $10k
-> "alpha" ที่รายงานในทุก iter ที่ผ่านมาผิด (ดูแย่กว่าความจริง)

Usage:
    python -m tools.rescore                  # rescore Iter 4.1 (จาก fold_results.csv)
    python -m tools.rescore --iter4-mean     # estimate Iter 4 mean ด้วย
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from src.data import load_data
from tools.walk_forward import buy_and_hold_equity_return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--train-size", type=int, default=1500)
    parser.add_argument("--test-size", type=int, default=400)
    parser.add_argument("--step-size", type=int, default=400)
    parser.add_argument("--fold-csv", default="walk_forward_output/fold_results.csv",
                        help="CSV ของ run ที่ต้อง re-score (Iter 4.1)")
    parser.add_argument("--iter4-mean-return", type=float, default=6.37,
                        help="Mean return % ของ Iter 4 (จาก PROJECT_CONTEXT)")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print("=" * 75)
    print(" RE-SCORE WITH CORRECTED B&H BENCHMARK")
    print("=" * 75)
    print(f"  config: {args.config}")
    print(f"  fold params: train={args.train_size} test={args.test_size} step={args.step_size}")

    # ---- Reload same folds เพื่อ compute corrected B&H ----
    print("\nLoading data + rebuilding folds...")
    df = load_data(cfg)
    folds = []
    s = 0
    while s + args.train_size + args.test_size <= len(df):
        folds.append((s, s + args.train_size, s + args.train_size + args.test_size))
        s += args.step_size
    print(f"  total rows: {len(df):,}, folds: {len(folds)}")

    env_kwargs = cfg["env"]

    # ---- Compute corrected B&H per fold ----
    print("\nComputing apples-to-apples B&H per fold...")
    print(f"  (lot={env_kwargs['lot_size']}, contract={env_kwargs['contract_size']}, "
          f"spread={env_kwargs['spread']}, balance=${env_kwargs['initial_balance']:,.0f})")

    bnh_corrected = []
    bnh_raw = []
    for i, (a, b, c) in enumerate(folds):
        test_df = df.iloc[b:c].reset_index(drop=True)
        bnh_eq = buy_and_hold_equity_return(test_df, env_kwargs)
        bnh_pct = (test_df["close"].iloc[-1] / test_df["close"].iloc[0] - 1) * 100
        bnh_corrected.append(bnh_eq)
        bnh_raw.append(bnh_pct)
        print(f"  Fold {i+1}: raw price Δ = {bnh_pct:+6.2f}%   "
              f"-> 0.01-lot equity B&H = {bnh_eq:+6.2f}%   "
              f"({bnh_pct/bnh_eq:.1f}x leverage gap)")

    # ---- Re-score Iter 4.1 from fold_results.csv ----
    csv_path = Path(args.fold_csv)
    if csv_path.exists():
        print(f"\n{'=' * 75}")
        print(f" ITER 4.1 RE-SCORE (from {csv_path})")
        print(f"{'=' * 75}")

        df_results = pd.read_csv(csv_path)
        df_results["bnh_corrected_pct"] = bnh_corrected[:len(df_results)]
        df_results["alpha_corrected_pct"] = (
            df_results["total_return_pct"] - df_results["bnh_corrected_pct"]
        )
        df_results["bnh_raw_pct"] = bnh_raw[:len(df_results)]
        df_results["alpha_raw_pct"] = (
            df_results["total_return_pct"] - df_results["bnh_raw_pct"]
        )

        print(f"\n  {'Fold':<5} {'BotRet%':>8} {'BnH_raw':>8} {'Alpha_OLD':>10} "
              f"{'BnH_eq':>8} {'Alpha_NEW':>10}  Δ Alpha")
        print(f"  {'-'*5} {'-'*8} {'-'*8} {'-'*10} {'-'*8} {'-'*10}  {'-'*8}")
        for _, r in df_results.iterrows():
            d_alpha = r["alpha_corrected_pct"] - r["alpha_raw_pct"]
            print(f"  {int(r['fold']):<5} "
                  f"{r['total_return_pct']:>+7.2f}% "
                  f"{r['bnh_raw_pct']:>+7.2f}% "
                  f"{r['alpha_raw_pct']:>+9.2f}% "
                  f"{r['bnh_corrected_pct']:>+7.2f}% "
                  f"{r['alpha_corrected_pct']:>+9.2f}%  {d_alpha:>+7.2f}%")

        print(f"  {'-'*5} {'-'*8} {'-'*8} {'-'*10} {'-'*8} {'-'*10}  {'-'*8}")
        mean_old = df_results["alpha_raw_pct"].mean()
        mean_new = df_results["alpha_corrected_pct"].mean()
        prof_old = (df_results["alpha_raw_pct"] > 0).sum()
        prof_new = (df_results["alpha_corrected_pct"] > 0).sum()
        n = len(df_results)
        print(f"  Mean alpha:   OLD {mean_old:+.2f}%   NEW {mean_new:+.2f}%   "
              f"Δ {mean_new-mean_old:+.2f}%")
        print(f"  Folds beating B&H:   OLD {prof_old}/{n}   NEW {prof_new}/{n}")

        out_path = Path("walk_forward_output/fold_results_rescored.csv")
        df_results.to_csv(out_path, index=False)
        print(f"\n  Saved: {out_path}")
    else:
        print(f"\n  (fold_results.csv not found at {csv_path}, skip Iter 4.1 rescore)")

    # ---- Estimate Iter 4 mean ----
    print(f"\n{'=' * 75}")
    print(f" ITER 4 MEAN ESTIMATE (per-fold data ไม่มี ใช้ mean จาก PROJECT_CONTEXT)")
    print(f"{'=' * 75}")
    iter4_mean_ret = args.iter4_mean_return
    iter4_mean_alpha_old = -7.37
    bnh_corrected_mean = float(np.mean(bnh_corrected))
    bnh_raw_mean = float(np.mean(bnh_raw))
    iter4_mean_alpha_new_estimate = iter4_mean_ret - bnh_corrected_mean

    print(f"  Iter 4 mean bot return:        {iter4_mean_ret:+.2f}%")
    print(f"  Mean B&H (raw price, OLD):     {bnh_raw_mean:+.2f}%   -> alpha_old {iter4_mean_alpha_old:+.2f}%")
    print(f"  Mean B&H (0.01-lot, NEW):      {bnh_corrected_mean:+.2f}%   -> alpha_new {iter4_mean_alpha_new_estimate:+.2f}%")
    print(f"  Δ alpha estimate:              {iter4_mean_alpha_new_estimate - iter4_mean_alpha_old:+.2f}%")

    # ---- Verdict ----
    print(f"\n{'=' * 75}")
    print(f" VERDICT")
    print(f"{'=' * 75}")
    if csv_path.exists():
        if mean_new > 0:
            v = f"🟢 Iter 4.1 ACTUALLY BEATS apples-to-apples B&H (mean alpha {mean_new:+.2f}%)"
        elif mean_new > -2:
            v = f"🟡 Iter 4.1 close to par (mean alpha {mean_new:+.2f}%)"
        else:
            v = f"🔴 Iter 4.1 still underperforms (mean alpha {mean_new:+.2f}%)"
        print(f"  {v}")
    print(f"\n  ทุก iteration ที่ผ่านมาเคยรายงาน alpha ผิด (เทียบกับ benchmark ที่ใช้ leverage")
    print(f"  ต่างกันคนละ scale) — ควร rescore Iter 0-3 ด้วยถ้ามีข้อมูลเก็บไว้")


if __name__ == "__main__":
    main()
