"""
Train a pick-and-place policy with SAC (default) or PPO, with an automatic
curriculum over the environment's `difficulty` parameter.

Why SAC by default here, when the reach task used PPO as its headline?
--------------------------------------------------------------------
Reach is a short-horizon, dense-reward, almost-convex problem, and PPO's
sample inefficiency does not hurt much. Pick-and-place has a long horizon, a
staged reward, and a narrow "success funnel" (you must grasp before carrying
matters), so the useful experience is rare. Off-policy learning with a replay
buffer reuses each of those rare grasps many times instead of throwing the
batch away after one gradient step. Being able to state that trade-off is
more interesting than the result itself.

Curriculum
----------
`CurriculumCallback` watches the evaluation success rate and nudges
`difficulty` up when the policy clears a threshold. The env starts with the
box almost always in the same place and the goal a short lift above it, and
ends with both broadly randomised. This is a simple version of automatic
curriculum learning: no separate teacher, just a success-rate trigger.

Usage
-----
    python training/train_pick_place.py --timesteps 400000 --run-name sac_pp_v1
    python training/train_pick_place.py --algo ppo --timesteps 1000000
    python training/train_pick_place.py --no-curriculum --difficulty 1.0
    tensorboard --logdir logs/
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environments.pick_place_env import PickPlaceEnv  # noqa: E402


def make_env(difficulty: float, reward_type: str, seed: int = 0):
    def _init():
        env = PickPlaceEnv(difficulty=difficulty, reward_type=reward_type)
        env.reset(seed=seed)
        return Monitor(env, info_keywords=("is_success", "is_grasped"))
    return _init


class CurriculumCallback(BaseCallback):
    """Raise `difficulty` on both the training and eval envs once the policy
    is reliably succeeding at the current level.

    Reads the success rate out of the Monitor-wrapped episode buffer rather
    than recomputing it, so it costs nothing.
    """

    def __init__(self, train_env, eval_env, threshold=0.6, step=0.2,
                 window=30, check_every=5000, verbose=1):
        super().__init__(verbose)
        self.train_env, self.eval_env = train_env, eval_env
        self.threshold, self.step, self.window = threshold, step, window
        self.check_every = check_every
        self._last_check = 0
        self.difficulty = 0.0

    def _set(self, d: float):
        self.difficulty = float(np.clip(d, 0.0, 1.0))
        for venv in (self.train_env, self.eval_env):
            venv.env_method("set_difficulty", self.difficulty)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_check < self.check_every:
            return True
        self._last_check = self.num_timesteps

        buf = self.model.ep_info_buffer
        if not buf or len(buf) < self.window:
            return True
        recent = list(buf)[-self.window:]
        rate = float(np.mean([ep.get("is_success", 0.0) for ep in recent]))
        self.logger.record("curriculum/success_rate", rate)
        self.logger.record("curriculum/difficulty", self.difficulty)

        if rate >= self.threshold and self.difficulty < 1.0:
            self._set(self.difficulty + self.step)
            if self.verbose:
                print(f"[curriculum] success {rate:.0%} -> "
                      f"difficulty {self.difficulty:.2f} "
                      f"at {self.num_timesteps} steps")
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["sac", "ppo"], default="sac")
    p.add_argument("--timesteps", type=int, default=400_000)
    p.add_argument("--run-name", type=str, default="sac_pick_place")
    p.add_argument("--reward-type", choices=["dense", "sparse"], default="dense")
    p.add_argument("--difficulty", type=float, default=0.0)
    p.add_argument("--no-curriculum", action="store_true")
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    run_dir = os.path.join("logs", args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    n_envs = 1 if args.algo == "sac" else args.n_envs
    train_env = DummyVecEnv([
        make_env(args.difficulty, args.reward_type, args.seed + i)
        for i in range(n_envs)
    ])
    eval_env = DummyVecEnv([make_env(args.difficulty, args.reward_type, 10_000)])

    if args.algo == "sac":
        model = SAC(
            "MlpPolicy", train_env, verbose=1, seed=args.seed,
            learning_rate=3e-4, buffer_size=400_000, batch_size=256,
            tau=0.01, gamma=0.98, learning_starts=3_000,
            train_freq=1, gradient_steps=1,
            policy_kwargs=dict(net_arch=[256, 256]),
            tensorboard_log=run_dir,
        )
    else:
        model = PPO(
            "MlpPolicy", train_env, verbose=1, seed=args.seed,
            learning_rate=3e-4, n_steps=1024, batch_size=256, n_epochs=10,
            gamma=0.99, gae_lambda=0.95, ent_coef=0.005,
            policy_kwargs=dict(net_arch=[256, 256]),
            tensorboard_log=run_dir,
        )

    callbacks = [
        EvalCallback(
            eval_env, best_model_save_path=run_dir, log_path=run_dir,
            eval_freq=max(10_000 // n_envs, 1), n_eval_episodes=20,
            deterministic=True, render=False,
        )
    ]
    if not args.no_curriculum:
        callbacks.append(CurriculumCallback(train_env, eval_env))

    model.learn(total_timesteps=args.timesteps, callback=callbacks,
                progress_bar=False)
    model.save(os.path.join(run_dir, "final_model"))
    print(f"done. model + logs in {run_dir}")


if __name__ == "__main__":
    main()
