"""
Overfit check — backtest บน TRAIN set แล้วเทียบกับ TEST set

ทฤษฎี:
  - In-sample (train set):  model "เคยเห็น" data นี้ตอน train
  - Out-of-sample (test):   model "ไม่เคยเห็น"
  - ถ้า in-sample >>> out-of-sample = overfit
  - ปกติ in-sample จะ "ดีกว่านิดหน่อย" (5-30%) เป็นเรื่องธรรมดา
  - ถ้า in-sample ดีกว่า 2 เท่าขึ้นไป = overfit รุนแรง

Usage:
    python -m tools.overfit_check
"""
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.backtest import (
    annualization_factor, max_drawdown, profit_factor, sharpe_ratio,
)
from src.data import load_data, split_train_test
from src.env import GoldTradingEnv


def run_on_dataset(name, df, cfg, model, vn):
    env_kwargs = cfg["env"]
    raw_env = GoldTradingEnv(df, **env_kwargs)

    obs, _ = raw_env.reset()
    while True:
        norm_obs = vn.normalize_obs(obs[None, :])[0] if vn else obs
        action, _ = model.predict(norm_obs, deterministic=True)
        action = int(np.asarray(action).flatten()[0])
        obs, _, term, trunc, _ = raw_env.step(action)
        if term or trunc:
            break

    equity = np.array(raw_env.equity_curve, dtype=np.float64)
    ann = annualization_factor(cfg["data"]["interval"])

    return {
        "name": name,
        "rows": len(df),
        "total_return_pct": (equity[-1] / equity[0] - 1) * 100,
        "total_trades": raw_env.total_trades,
        "win_rate_pct": raw_env.winning_trades / max(raw_env.total_trades, 1) * 100,
        "max_dd_pct": max_drawdown(equity) * 100,
        "sharpe": sharpe_ratio(equity, ann),
        "profit_factor": profit_factor(raw_env.trade_returns),
        "buy_and_hold_pct": (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100,
    }


def main():
    with open("config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print("Loading data...")
    df = load_data(cfg)
    train_df, test_df = split_train_test(df, cfg["data"]["train_test_split"])

    # Load model + VecNormalize
    vec_path = Path(cfg["paths"]["vecnormalize_path"])
    if vec_path.exists():
        dummy = DummyVecEnv([lambda: GoldTradingEnv(train_df, **cfg["env"])])
        vn = VecNormalize.load(str(vec_path), dummy)
        vn.training = False
        vn.norm_reward = False
    else:
        vn = None

    model_path = Path(cfg["paths"]["best_model_dir"]) / "best_model.zip"
    if not model_path.exists():
        model_path = Path(cfg["paths"]["checkpoint_dir"]) / "ppo_gold_final.zip"
    print(f"Model: {model_path}")
    model = PPO.load(str(model_path))

    print("\nRunning train-set backtest (in-sample)...")
    train_res = run_on_dataset("TRAIN", train_df, cfg, model, vn)

    print("Running test-set backtest (out-of-sample)...")
    test_res = run_on_dataset("TEST", test_df, cfg, model, vn)

    # Print comparison
    print("\n" + "=" * 64)
    print(" OVERFIT CHECK — TRAIN vs TEST")
    print("=" * 64)
    print(f"  {'Metric':<22} {'TRAIN (seen)':>14} {'TEST (unseen)':>14} {'Gap':>10}")
    print(f"  {'-'*22} {'-'*14} {'-'*14} {'-'*10}")

    metrics_to_compare = [
        ("Total return %",   "total_return_pct"),
        ("Sharpe ratio",     "sharpe"),
        ("Win rate %",       "win_rate_pct"),
        ("Max DD %",         "max_dd_pct"),
        ("Profit factor",    "profit_factor"),
        ("Total trades",     "total_trades"),
    ]
    for label, key in metrics_to_compare:
        tr, te = train_res[key], test_res[key]
        if isinstance(tr, float):
            gap = (te / tr - 1) * 100 if tr != 0 else 0
            print(f"  {label:<22} {tr:>14.4f} {te:>14.4f} {gap:>+9.1f}%")
        else:
            print(f"  {label:<22} {tr:>14} {te:>14}")

    print()
    print(f"  Buy-and-hold (test):  {test_res['buy_and_hold_pct']:+.2f}%")
    print(f"  Bot return (test):    {test_res['total_return_pct']:+.2f}%")
    alpha = test_res['total_return_pct'] - test_res['buy_and_hold_pct']
    print(f"  Alpha (bot - hold):   {alpha:+.2f}%   {'(bot ดีกว่าซื้อทิ้ง)' if alpha > 0 else '(บอทแพ้ซื้อทิ้ง!)'}")

    # Diagnosis
    print("\n" + "=" * 64)
    print(" DIAGNOSIS")
    print("=" * 64)
    sharpe_gap = train_res["sharpe"] - test_res["sharpe"]
    return_gap = train_res["total_return_pct"] - test_res["total_return_pct"]

    if sharpe_gap > 5 or train_res["sharpe"] > test_res["sharpe"] * 3:
        print("  🔴 OVERFIT รุนแรง — bot จำ pattern ใน train set แต่ไม่ generalize")
        print("     แนะนำ: ลด net_arch ลงเป็น [128, 128], เพิ่ม ent_coef เป็น 0.05")
    elif sharpe_gap > 2:
        print("  🟡 OVERFIT ปานกลาง — gap มีอยู่แต่ไม่อันตราย ใช้งานได้ระวัง ๆ")
    else:
        print("  🟢 OK — model generalize ได้ดี gap น้อย")

    if alpha > 5:
        print("  🟢 บอทเก่งกว่า buy-and-hold มาก = มี skill จริง")
    elif alpha > 0:
        print("  🟡 บอทดีกว่า buy-and-hold เล็กน้อย = พอใช้")
    else:
        print("  🔴 บอทแพ้ buy-and-hold = ไม่คุ้มเสียค่า commission/spread")


if __name__ == "__main__":
    main()
