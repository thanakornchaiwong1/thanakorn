"""Quick test to verify trade cooldown mechanism."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from src.data import load_data
from src.env import GoldTradingEnv

with open("config_scalp.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
df = load_data(cfg)
env = GoldTradingEnv(df, **cfg["env"])

print(f"Cooldown setting: {env.trade_cooldown} M15 bars")
print(f"obs_dim: {env.observation_space.shape[0]}")

# Test 1: Manual action sequence
obs, _ = env.reset()
ACTION_NAMES = ["Hold", "Buy", "Sell", "Close"]
actions = [1, 0, 0, 3, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
#          Buy Hold Hold Close Buy(X) Buy(X) ... 20 Holds ... Buy(OK)
print("\nTest 1: Manual sequence (Buy -> Close -> cooldown -> Buy)")
for i, a in enumerate(actions):
    obs, r, t, tr, info = env.step(a)
    cd = env._cooldown_remaining
    pos = info["position"]
    trades = info["total_trades"]
    print(f"  step {i:2d}: {ACTION_NAMES[a]:>5} | pos={pos:+d} | trades={trades} | cooldown={cd}")

# Test 2: Rapid-fire trading for 2000 steps
print("\nTest 2: Rapid-fire Buy/Close cycles over 2000 steps")
obs, _ = env.reset()
step_count = 0
while step_count < 2000:
    # Try to Buy
    obs, r, t, tr, info = env.step(1)
    if t: break
    step_count += 1

    if info["position"] == 1:
        # Hold 2 steps then Close
        for _ in range(2):
            obs, r, t, tr, info = env.step(0)
            if t: break
            step_count += 1
        obs, r, t, tr, info = env.step(3)
        if t: break
        step_count += 1
    else:
        # Couldn't buy (cooldown) -> hold
        obs, r, t, tr, info = env.step(0)
        if t: break
        step_count += 1

trades_done = info["total_trades"]
print(f"  Steps: {step_count}")
print(f"  Trades: {trades_done}")
print(f"  Avg steps/trade: {step_count / max(trades_done, 1):.1f}")
print(f"  Expected min gap: {env.trade_cooldown + 3} steps (cooldown + hold)")
print(f"  Max possible trades: ~{2000 // (env.trade_cooldown + 3)}")
print(f"  COOLDOWN WORKING!" if trades_done < 200 else "  WARNING: too many trades!")
