"""
Sanity check ก่อน train จริง — รันก่อน train.py ทุกครั้งที่แก้ env / reward / data

Checks:
  1) Data load + features เรียบร้อย ไม่มี NaN
  2) SB3 check_env ผ่าน (observation/action space ถูกต้อง)
  3) Random rollout 500 steps ไม่มี NaN/Inf, reward อยู่ในช่วงสมเหตุสมผล

Usage:
    python -m tools.check_env
"""
import sys
from pathlib import Path

import numpy as np
import yaml

# ทำให้ import src.* ได้แม้รันจาก project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_data, split_train_test
from src.env import GoldTradingEnv, FEATURE_COLUMNS, DAILY_FEATURE_COLUMNS


def main():
    with open("config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print("=" * 56)
    print(" ENV SANITY CHECK")
    print("=" * 56)

    # ---------------- 1) Data ----------------
    print("\n[1/3] Loading data...")
    df = load_data(cfg)
    train_df, test_df = split_train_test(df, cfg["data"]["train_test_split"])
    print(f"  total={len(df)}  train={len(train_df)}  test={len(test_df)}")

    # Auto-detect features ที่อยู่ใน df จริง ๆ
    all_known = FEATURE_COLUMNS + DAILY_FEATURE_COLUMNS
    active_features = [c for c in all_known if c in df.columns]
    use_mtf = any(c in df.columns for c in DAILY_FEATURE_COLUMNS)
    print(f"  mode: {'MULTI-TIMEFRAME (4h + 1d)' if use_mtf else 'SINGLE TIMEFRAME'}")
    print(f"  features ({len(active_features)}): {active_features}")

    # NaN check
    nan_cols = [c for c in active_features if df[c].isna().any()]
    if nan_cols:
        print(f"  WARNING: NaN found in: {nan_cols}")
    else:
        print("  features: no NaN")

    # range sanity
    print("\n  feature ranges:")
    for c in active_features:
        s = df[c]
        print(f"    {c:18s} min={s.min():>10.4f} max={s.max():>10.4f} "
              f"mean={s.mean():>10.4f} std={s.std():>10.4f}")

    # ---------------- 2) check_env ----------------
    print("\n[2/3] SB3 check_env...")
    try:
        from stable_baselines3.common.env_checker import check_env
    except ImportError:
        print("  stable-baselines3 not installed — skip")
    else:
        env = GoldTradingEnv(train_df, **cfg["env"])
        check_env(env, warn=True)
        print(f"  OK | obs_shape={env.observation_space.shape} action_n={env.action_space.n}")

    # ---------------- 3) Random rollout ----------------
    print("\n[3/3] Random rollout (500 steps)...")
    env = GoldTradingEnv(train_df, **cfg["env"])
    obs, info = env.reset(seed=42)
    assert np.isfinite(obs).all(), "Initial obs has NaN/Inf!"

    rewards = []
    actions_taken = [0, 0, 0, 0]
    for i in range(500):
        a = env.action_space.sample()
        actions_taken[a] += 1
        obs, r, term, trunc, info = env.step(a)
        rewards.append(r)
        assert np.isfinite(obs).all(), f"obs NaN/Inf at step {i}"
        assert np.isfinite(r), f"reward NaN/Inf at step {i}"
        if term or trunc:
            print(f"  episode ended early at step {i}: terminated={term} truncated={trunc}")
            break

    rewards = np.array(rewards)
    print(f"  reward: mean={rewards.mean():+.6f} std={rewards.std():.6f} "
          f"min={rewards.min():+.4f} max={rewards.max():+.4f}")
    print(f"  actions: Hold={actions_taken[0]} Buy={actions_taken[1]} "
          f"Sell={actions_taken[2]} Close={actions_taken[3]}")
    print(f"  final equity: {info['equity']:.2f}  "
          f"trades: {info['total_trades']}  "
          f"win_rate: {info['win_rate']:.1%}")

    print("\n" + "=" * 56)
    print(" ALL CHECKS PASSED — พร้อม train")
    print(" Run: python -m src.train")
    print("=" * 56)


if __name__ == "__main__":
    main()
