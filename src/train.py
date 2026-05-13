"""
Train PPO agent for XAUUSD trading.

Usage:
    python -m src.train                                  # ใช้ config.yaml
    python -m src.train --config custom.yaml
    python -m src.train --timesteps 100000               # override total_timesteps
    python -m src.train --resume checkpoints/ppo_gold_100000_steps.zip
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

try:
    from .data import load_data, split_train_test
    from .env import GoldTradingEnv
except ImportError:
    from data import load_data, split_train_test  # type: ignore
    from env import GoldTradingEnv  # type: ignore


def make_env_fn(df, env_kwargs, seed: int, rank: int = 0):
    """factory สำหรับ DummyVecEnv — แต่ละ env ได้ seed ต่างกัน"""
    def _init():
        env = GoldTradingEnv(df, **env_kwargs)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="override train.total_timesteps")
    parser.add_argument("--resume", type=str, default=None,
                        help="path ของ .zip checkpoint ที่จะ resume training")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ---------------- Data ----------------
    print("[1/4] Loading data...")
    df = load_data(cfg)
    train_df, test_df = split_train_test(df, cfg["data"]["train_test_split"])
    print(f"  rows: total={len(df)} train={len(train_df)} test={len(test_df)}")

    # ---------------- Envs ----------------
    print("[2/4] Building environments...")
    env_kwargs = cfg["env"]
    n_envs = cfg["train"]["n_envs"]
    seed = cfg["train"]["seed"]

    train_env = DummyVecEnv([make_env_fn(train_df, env_kwargs, seed, i) for i in range(n_envs)])
    eval_env = DummyVecEnv([make_env_fn(test_df, env_kwargs, seed + 999, 0)])

    if cfg["train"]["use_vec_normalize"]:
        train_env = VecNormalize(
            train_env, norm_obs=True, norm_reward=True, clip_obs=10.0,
        )
        eval_env = VecNormalize(
            eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0,
            training=False,
        )
        # EvalCallback ของ SB3 จะ sync obs_rms ระหว่าง train/eval ให้อัตโนมัติ
        # (เห็นได้ใน sb3.common.callbacks.EvalCallback._on_step)

    # ---------------- Model ----------------
    print("[3/4] Building PPO...")
    ppo_cfg = cfg["ppo"]
    policy_kwargs = dict(
        net_arch=dict(pi=ppo_cfg["net_arch"]["pi"], vf=ppo_cfg["net_arch"]["vf"]),
    )

    Path(cfg["paths"]["tensorboard_log"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["best_model_dir"]).mkdir(parents=True, exist_ok=True)

    if args.resume:
        print(f"  resuming from: {args.resume}")
        model = PPO.load(args.resume, env=train_env, tensorboard_log=cfg["paths"]["tensorboard_log"])
    else:
        model = PPO(
            policy=ppo_cfg["policy"],
            env=train_env,
            learning_rate=ppo_cfg["learning_rate"],
            n_steps=ppo_cfg["n_steps"],
            batch_size=ppo_cfg["batch_size"],
            n_epochs=ppo_cfg["n_epochs"],
            gamma=ppo_cfg["gamma"],
            gae_lambda=ppo_cfg["gae_lambda"],
            clip_range=ppo_cfg["clip_range"],
            ent_coef=ppo_cfg["ent_coef"],
            vf_coef=ppo_cfg["vf_coef"],
            max_grad_norm=ppo_cfg["max_grad_norm"],
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=seed,
            tensorboard_log=cfg["paths"]["tensorboard_log"],
            device="auto",
        )

    # ---------------- Callbacks ----------------
    # eval_freq / save_freq ที่ส่งให้ callback คือ "per env" -> หารด้วย n_envs
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=cfg["paths"]["best_model_dir"],
        log_path="./logs/eval/",
        eval_freq=max(cfg["train"]["eval_freq"] // n_envs, 1),
        n_eval_episodes=cfg["train"]["n_eval_episodes"],
        deterministic=True,
        render=False,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=max(cfg["train"]["checkpoint_freq"] // n_envs, 1),
        save_path=cfg["paths"]["checkpoint_dir"],
        name_prefix="ppo_gold",
        save_vecnormalize=cfg["train"]["use_vec_normalize"],
    )

    # ---------------- Train ----------------
    print("[4/4] Training...")
    total_timesteps = args.timesteps or cfg["train"]["total_timesteps"]
    print(f"  total_timesteps={total_timesteps:,}  n_envs={n_envs}")
    print(f"  tensorboard: tensorboard --logdir {cfg['paths']['tensorboard_log']}\n")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_cb, ckpt_cb],
            progress_bar=True,
            reset_num_timesteps=(args.resume is None),
        )
    finally:
        # save แม้ Ctrl+C
        final_path = Path(cfg["paths"]["checkpoint_dir"]) / "ppo_gold_final"
        model.save(str(final_path))
        print(f"\nFinal model: {final_path}.zip")

        if cfg["train"]["use_vec_normalize"]:
            train_env.save(cfg["paths"]["vecnormalize_path"])
            print(f"VecNormalize: {cfg['paths']['vecnormalize_path']}")

        print(f"Best model:  {cfg['paths']['best_model_dir']}/best_model.zip")
        print("\nNext step:")
        print("  python -m src.backtest")


if __name__ == "__main__":
    main()
