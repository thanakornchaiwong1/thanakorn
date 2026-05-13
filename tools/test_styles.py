"""
Head-to-head comparison: Sniper vs Scalping vs D/S Trailing

A) Sniper:      RR 1:3, SL=1.5, TP=4.5, heavy entry cost, few trades
B) Scalping:    RR 1:2, SL=1.0, TP=2.0, light cost, many trades
C) D/S Trail:   SL=1.5, trailing stop (activate 1.5 ATR, trail 1.0 ATR), ride trends

Usage:
    python -m tools.test_styles
"""
import subprocess
import sys
import time
from pathlib import Path

CONFIGS = [
    {
        "name": "A_Sniper",
        "sl": 1.5, "tp": 4.5,
        "reward": "style_sniper",
        "extra": [],
    },
    {
        "name": "B_Scalping",
        "sl": 1.0, "tp": 2.0,
        "reward": "style_scalp",
        "extra": [],
    },
    {
        "name": "C_DS_Trailing",
        "sl": 1.5, "tp": 4.5,
        "reward": "style_ds_trailing",
        "extra": ["--exit-mode", "trailing",
                  "--trailing-activate", "1.5",
                  "--trailing-dist", "1.0"],
    },
]

COMMON_ARGS = [
    sys.executable, "-m", "tools.walk_forward",
    "--config", "config_scalp.yaml",
    "--train-size", "50000",
    "--test-size", "10000",
    "--step-size", "20000",
    "--timesteps", "500000",
]


def main():
    out_dir = Path("./walk_forward_output")
    out_dir.mkdir(exist_ok=True)

    total_start = time.time()

    for i, cfg in enumerate(CONFIGS):
        print("\n" + "#" * 70)
        print(f"# STYLE {i+1}/{len(CONFIGS)}: {cfg['name']}")
        print(f"#   SL={cfg['sl']}, TP={cfg['tp']}, reward={cfg['reward']}")
        if cfg['extra']:
            print(f"#   extra: {' '.join(cfg['extra'])}")
        print("#" * 70 + "\n")

        args = COMMON_ARGS + [
            "--sl-atr", str(cfg["sl"]),
            "--tp-atr", str(cfg["tp"]),
            "--reward", cfg["reward"],
        ] + cfg["extra"]

        t0 = time.time()
        result = subprocess.run(args, cwd=str(Path(__file__).parent.parent))
        elapsed = (time.time() - t0) / 60

        print(f"\n>>> {cfg['name']} completed in {elapsed:.1f} min (exit code {result.returncode})")

        # Rename outputs
        for f in ["fold_results.csv", "walk_forward_equity.png"]:
            src = out_dir / f
            dst = out_dir / f"{cfg['name']}_{f}"
            if src.exists():
                if dst.exists():
                    dst.unlink()
                src.rename(dst)
                print(f"  Saved: {dst}")

    total_min = (time.time() - total_start) / 60
    print(f"\n{'='*70}")
    print(f"  ALL DONE! Total time: {total_min:.1f} min")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
