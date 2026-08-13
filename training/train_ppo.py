"""
Train PPO on the ReachEnv using Stable-Baselines3.

Usage:
    python training/train_ppo.py --timesteps 200000 --run-name ppo_reach_v1
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from environments.reach_env import ReachEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--run-name", type=str, default="ppo_reach")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_dir = os.path.join(args.log_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    def env_fn():
        return Monitor(ReachEnv())

    vec_env = make_vec_env(env_fn, n_envs=args.n_envs, seed=args.seed)
    eval_env = Monitor(ReachEnv())

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=run_dir,
        log_path=run_dir,
        eval_freq=max(5000 // args.n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        seed=args.seed,
        tensorboard_log=run_dir,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        gamma=0.98,
    )

    model.learn(total_timesteps=args.timesteps, callback=eval_callback)
    model.save(os.path.join(run_dir, "final_model"))
    print(f"Training complete. Model + logs saved to {run_dir}")


if __name__ == "__main__":
    main()
