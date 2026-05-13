"""
Batch runner: compare RR 1:2, 1:3, 1:5 with walk-forward validation.

All configs use:
  - SL = 2.0 ATR (wider, reduces noise stops from 38% to 23%)
  - MLP architecture (best from testing)
  - scalp_sniper_v4 reward (RR-adaptive)
  - 300k timesteps/fold (balance speed vs quality)

Usage:
    python -m tools.test_rr_configs
"""
import subprocess
import sys
import time
from pathlib import Path

CONFIGS = [
    {"name": "RR_1to2", "sl": 2.0, "tp": 4.0},
    {"name": "RR_1to3", "sl": 2.0, "tp": 6.0},
    {"name": "RR_1to5", "sl": 2.0, "tp": 10.0},
]

COMMON_ARGS = [
    sys.executable, "-m", "tools.walk_forward",
    "--config", "config_scalp.yaml",
    "--train-size", "50000",
    "--test-size", "10000",
    "--step-size", "20000",
    "--timesteps", "300000",
    "--reward", "scalp_sniper_v4",
]


def main():
    out_dir = Path("./walk_forward_output")
    out_dir.mkdir(exist_ok=True)

    total_start = time.time()

    for i, cfg in enumerate(CONFIGS):
        print("\n" + "#" * 70)
        print(f"# CONFIG {i+1}/{len(CONFIGS)}: {cfg['name']}  (SL={cfg['sl']}, TP={cfg['tp']})")
        print("#" * 70 + "\n")

        args = COMMON_ARGS + [
            "--sl-atr", str(cfg["sl"]),
            "--tp-atr", str(cfg["tp"]),
        ]

        t0 = time.time()
        result = subprocess.run(args, cwd=str(Path(__file__).parent.parent))
        elapsed = (time.time() - t0) / 60

        print(f"\n>>> {cfg['name']} completed in {elapsed:.1f} min (exit code {result.returncode})")

        # Rename output files to include config name
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
