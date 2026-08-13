"""
Train SAC on ReachEnv. SAC is off-policy and typically more sample-efficient
than PPO on continuous control tasks like this one -- useful to compare
against the PPO baseline and report which does better and why.

Usage:
    python training/train_sac.py --timesteps 100000 --run-name sac_reach_v1
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from environments.reach_env import ReachEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--run-name", type=str, default="sac_reach")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_dir = os.path.join(args.log_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    train_env = Monitor(ReachEnv())
    eval_env = Monitor(ReachEnv())

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=run_dir,
        log_path=run_dir,
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
    )

    model = SAC(
        "MlpPolicy",
        train_env,
        verbose=1,
        seed=args.seed,
        tensorboard_log=run_dir,
        learning_rate=3e-4,
        buffer_size=100_000,
        batch_size=256,
        gamma=0.98,
        train_freq=1,
        gradient_steps=1,
    )

    model.learn(total_timesteps=args.timesteps, callback=eval_callback)
    model.save(os.path.join(run_dir, "final_model"))
    print(f"Training complete. Model + logs saved to {run_dir}")


if __name__ == "__main__":
    main()
