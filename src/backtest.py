"""
Backtest trained PPO model on held-out test set.

Usage:
    python -m src.backtest                                    # ใช้ best_model.zip
    python -m src.backtest --model checkpoints/ppo_gold_final.zip
    python -m src.backtest --config config.yaml --model path/to.zip
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

try:
    from .data import load_data, split_train_test
    from .env import GoldTradingEnv
except ImportError:
    from data import load_data, split_train_test  # type: ignore
    from env import GoldTradingEnv  # type: ignore


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def max_drawdown(equity: np.ndarray) -> float:
    """Max DD เป็นสัดส่วน (0-1)"""
    if len(equity) < 2:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / peak
    return float(dd.max())


def profit_factor(trade_returns) -> float:
    """sum(wins) / sum(|losses|) — > 1.0 = กำไร"""
    if not trade_returns:
        return 0.0
    arr = np.asarray(trade_returns, dtype=np.float64)
    wins = arr[arr > 0].sum()
    losses = -arr[arr < 0].sum()
    if losses <= 0:
        return float("inf") if wins > 0 else 0.0
    return float(wins / losses)


def sharpe_ratio(equity: np.ndarray, periods_per_year: int) -> float:
    """Annualized Sharpe (risk-free rate = 0)"""
    if len(equity) < 2:
        return 0.0
    rets = np.diff(equity) / equity[:-1]
    if rets.std() == 0:
        return 0.0
    return float(rets.mean() / rets.std() * np.sqrt(periods_per_year))


def annualization_factor(interval: str) -> int:
    """ตัวคูณ annualize ตาม timeframe (อิง 252 trading days)"""
    table = {
        "1m": 252 * 24 * 60, "5m": 252 * 24 * 12, "15m": 252 * 24 * 4,
        "30m": 252 * 24 * 2, "1h": 252 * 24, "4h": 252 * 6, "1d": 252,
    }
    return table.get(interval.lower(), 252 * 24)


# ---------------------------------------------------------------------------
# Backtest runner
# ---------------------------------------------------------------------------

def run_backtest(model_path: str, cfg: dict) -> dict:
    print(f"[1/3] Loading data...")
    df = load_data(cfg)
    _, test_df = split_train_test(df, cfg["data"]["train_test_split"])
    print(f"  test rows: {len(test_df)}")

    env_kwargs = cfg["env"]
    vec_path = Path(cfg["paths"]["vecnormalize_path"])
    use_vn = cfg["train"]["use_vec_normalize"] and vec_path.exists()

    print(f"[2/3] Loading model: {model_path}")
    # raw env: ไม่ wrap VecEnv -> ไม่มี auto-reset -> อ่าน state ตอนจบได้
    raw_env = GoldTradingEnv(test_df, **env_kwargs)

    # โหลด VecNormalize stats แยก (ใช้แค่ normalize observation ไม่ได้ใช้ env ของมัน)
    vn = None
    if use_vn:
        dummy = DummyVecEnv([lambda: GoldTradingEnv(test_df, **env_kwargs)])
        vn = VecNormalize.load(str(vec_path), dummy)
        vn.training = False
        vn.norm_reward = False
        print(f"  VecNormalize loaded: {vec_path}")
    elif cfg["train"]["use_vec_normalize"]:
        print(f"  WARNING: use_vec_normalize=true แต่ไม่พบ {vec_path} -> รันแบบ raw obs")

    model = PPO.load(model_path)  # ไม่ pass env -> predict ใช้ obs ที่เรา normalize เอง

    # ---------------- Run episode ----------------
    print(f"[3/3] Running backtest...")
    obs, _ = raw_env.reset()
    actions_log: list[int] = []
    step = 0
    while True:
        # normalize obs ด้วย stats ที่ load มา (ถ้ามี)
        if vn is not None:
            norm_obs = vn.normalize_obs(obs[None, :])[0]
        else:
            norm_obs = obs

        action, _ = model.predict(norm_obs, deterministic=True)
        action = int(np.asarray(action).flatten()[0])
        actions_log.append(action)

        obs, _reward, terminated, truncated, _info = raw_env.step(action)
        step += 1
        if terminated or truncated:
            break

    # อ่าน state จาก raw_env -- ไม่ถูก auto-reset เพราะเราคุมเอง
    equity = np.array(raw_env.equity_curve, dtype=np.float64)
    total_trades = int(raw_env.total_trades)
    winning_trades = int(raw_env.winning_trades)
    trade_returns = list(raw_env.trade_returns)
    final_balance = float(raw_env.balance)

    # ---------------- Metrics ----------------
    ann = annualization_factor(cfg["data"]["interval"])
    metrics = {
        "total_return_pct":     (equity[-1] / equity[0] - 1.0) * 100.0,
        "final_balance":        final_balance,
        "total_trades":         total_trades,
        "win_rate_pct":         (winning_trades / max(total_trades, 1)) * 100.0,
        "max_drawdown_pct":     max_drawdown(equity) * 100.0,
        "sharpe_ratio":         sharpe_ratio(equity, ann),
        "profit_factor":        profit_factor(trade_returns),
        "avg_trade_return_pct": (float(np.mean(trade_returns)) * 100.0) if trade_returns else 0.0,
        "steps":                step,
    }

    # ---------------- Print ----------------
    print("\n" + "=" * 56)
    print(" BACKTEST RESULTS")
    print("=" * 56)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:24s} {v:>12.4f}")
        else:
            print(f"  {k:24s} {v:>12}")

    actions = np.array(actions_log)
    names = ["Hold", "Buy", "Sell", "Close"]
    print("\n  Action distribution:")
    for i, name in enumerate(names):
        cnt = int((actions == i).sum())
        pct = cnt / max(len(actions), 1) * 100
        bar = "#" * int(pct / 2)
        print(f"    {name:6s} {cnt:6d} ({pct:5.1f}%) {bar}")

    # ---------------- Save outputs ----------------
    out_dir = Path("./backtest_output")
    out_dir.mkdir(exist_ok=True)
    pd.DataFrame({"step": np.arange(len(equity)), "equity": equity}).to_csv(
        out_dir / "equity_curve.csv", index=False,
    )
    pd.DataFrame(metrics, index=[0]).to_csv(out_dir / "metrics.csv", index=False)

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1]})
        ax1, ax2 = axes
        ax1.plot(equity, linewidth=1.2, color="#1f77b4")
        ax1.axhline(equity[0], color="gray", linestyle="--", alpha=0.5, label="Initial balance")
        peak = np.maximum.accumulate(equity)
        ax1.plot(peak, linewidth=0.7, color="green", alpha=0.4, label="Peak")
        ax1.fill_between(np.arange(len(equity)), equity, peak,
                         where=(equity < peak), color="red", alpha=0.15, label="Drawdown")
        ax1.set_ylabel("Equity (USD)")
        ax1.set_title(
            f"Equity Curve | Return: {metrics['total_return_pct']:+.2f}%  "
            f"Sharpe: {metrics['sharpe_ratio']:.2f}  "
            f"MaxDD: {metrics['max_drawdown_pct']:.1f}%  "
            f"Trades: {metrics['total_trades']}"
        )
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

        # subplot 2: drawdown%
        dd_series = (peak - equity) / peak * 100
        ax2.fill_between(np.arange(len(equity)), 0, -dd_series, color="red", alpha=0.5)
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_xlabel("Step")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_dir / "equity_curve.png", dpi=120)
        print(f"\n  Plot saved: {out_dir / 'equity_curve.png'}")
    except ImportError:
        print("\n  (pip install matplotlib เพื่อ plot equity curve)")

    print(f"  Files: {out_dir}/equity_curve.csv  metrics.csv\n")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", default=None,
                        help="path ไปที่ .zip (default: ./checkpoints/best/best_model.zip)")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Resolve model path: best -> final
    if args.model:
        model_path = Path(args.model)
    else:
        best = Path(cfg["paths"]["best_model_dir"]) / "best_model.zip"
        final = Path(cfg["paths"]["checkpoint_dir"]) / "ppo_gold_final.zip"
        if best.exists():
            model_path = best
        elif final.exists():
            print(f"best_model.zip ไม่พบ ใช้ final model แทน")
            model_path = final
        else:
            raise FileNotFoundError(
                f"No model found. Train ก่อน: python -m src.train"
            )

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    run_backtest(str(model_path), cfg)


if __name__ == "__main__":
    main()
