"""
Evaluate a trained pick-and-place policy.

Reports more than a success rate, because on a contact-rich task "40% success"
is not a diagnosis. The three numbers that matter are:

    grasp rate    -- did it ever close on the box?      (bottleneck reached)
    lift rate     -- did it get the box off the table?  (grasp was stable)
    success rate  -- did it get the box to the goal?    (the actual task)

The gaps between them tell you what to fix:

    low grasp, low lift, low success   -> exploration never finds the grasp;
                                          lower the curriculum, raise entropy
    high grasp, low lift               -> grasp is unstable; friction, finger
                                          gains, or contact solver settings
    high lift, low success             -> stage-2 reward or horizon too short
    high everything but jittery        -> policy is fine; eval noise

Usage:
    python evaluation/evaluate_pick_place.py --model-path logs/ppo_pp_v1/best_model.zip
    python evaluation/evaluate_pick_place.py --model-path ... --sweep
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from stable_baselines3 import PPO, SAC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environments.pick_place_env import PickPlaceEnv, LIFT_Z  # noqa: E402


def load_model(path: str):
    """SB3 does not record the algorithm in the zip, so try both."""
    for cls in (PPO, SAC):
        try:
            return cls.load(path, device="cpu"), cls.__name__
        except Exception:
            continue
    raise RuntimeError(f"could not load {path} as PPO or SAC")


def evaluate(model, difficulty: float, episodes: int, seed: int = 0,
             deterministic: bool = True):
    env = PickPlaceEnv(difficulty=difficulty)
    n_success = n_grasp = n_lift = 0
    returns, final_dists = [], []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        total = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, r, term, trunc, info = env.step(action)
            total += r
            if term or trunc:
                break
        n_success += int(info["is_success"])
        n_grasp += int(info["ever_grasped"])
        n_lift += int(info["peak_box_height"] > LIFT_Z)
        returns.append(total)
        final_dists.append(info["dist_box_goal"])

    env.close()
    return {
        "difficulty": difficulty,
        "success": n_success / episodes,
        "grasp": n_grasp / episodes,
        "lift": n_lift / episodes,
        "return": float(np.mean(returns)),
        "final_dist": float(np.mean(final_dists)),
    }


def print_row(r: dict):
    print(f"  {r['difficulty']:>4.2f} | {r['grasp']:>6.0%} | {r['lift']:>5.0%} "
          f"| {r['success']:>7.0%} | {r['return']:>7.2f} | {r['final_dist']:.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--difficulty", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stochastic", action="store_true",
                   help="sample actions instead of using the mean")
    p.add_argument("--sweep", action="store_true",
                   help="evaluate across the full curriculum, not one level")
    args = p.parse_args()

    model, algo = load_model(args.model_path)
    print(f"{algo} policy: {args.model_path}")
    print(f"{args.episodes} episodes, "
          f"{'stochastic' if args.stochastic else 'deterministic'} actions\n")
    print("  diff | grasp  | lift  | success | return  | final dist (m)")
    print("  -----+--------+-------+---------+---------+---------------")

    levels = [0.0, 0.25, 0.5, 0.75, 1.0] if args.sweep else [args.difficulty]
    results = [
        evaluate(model, d, args.episodes, args.seed, not args.stochastic)
        for d in levels
    ]
    for r in results:
        print_row(r)

    # Where the policy is losing episodes, in plain terms.
    r = results[-1]
    print()
    if r["grasp"] < 0.5:
        print("  Bottleneck: the policy rarely grasps at all. This is an "
              "exploration problem, not a control problem.")
    elif r["lift"] < r["grasp"] - 0.15:
        print("  Bottleneck: it grasps but drops. Look at finger gains, pad "
              "friction, and solver iterations before touching the RL.")
    elif r["success"] < r["lift"] - 0.15:
        print("  Bottleneck: it lifts but does not deliver. Stage-2 reward "
              "weight or episode length is the place to look.")
    else:
        print("  No single dominant failure mode -- remaining losses look "
              "like ordinary policy noise.")


if __name__ == "__main__":
    main()
