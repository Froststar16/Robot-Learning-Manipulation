"""
Evaluate a trained model on ReachEnv and report success rate / reward stats.

Usage:
    python evaluation/evaluate.py --model-path logs/ppo_reach_v1/best_model.zip --episodes 50
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from stable_baselines3 import PPO

from environments.reach_env import ReachEnv


def evaluate(model_path: str, n_episodes: int = 50, seed: int = 100):
    env = ReachEnv()
    model = PPO.load(model_path)

    rewards, successes, ep_lengths = [], [], []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        ep_reward, steps = 0.0, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            steps += 1
            done = terminated or truncated
        rewards.append(ep_reward)
        successes.append(info["is_success"])
        ep_lengths.append(steps)

    print(f"Episodes: {n_episodes}")
    print(f"Success rate: {100 * np.mean(successes):.1f}%")
    print(f"Mean reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    print(f"Mean episode length: {np.mean(ep_lengths):.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()
    evaluate(args.model_path, args.episodes, args.seed)
