"""
Record a trained (or random/scripted) policy's rollout as an animated GIF.

Requires offscreen rendering support (EGL or OSMesa). If you're on Linux
without a display, set MUJOCO_GL=osmesa before running. On Windows/macOS
with a normal desktop session, this usually works with no extra setup.

Usage:
    # Record a trained PPO policy, picks a successful episode if one exists
    # within --max-attempts tries so the GIF actually shows a reach.
    python evaluation/record_gif.py --model-path logs/ppo_reach_v1/best_model.zip \
        --out results/demo_rollout.gif

    # No model given -> records a scripted (hand-coded) controller instead,
    # useful for a demo GIF before you've trained anything.
    python evaluation/record_gif.py --out results/demo_scripted.gif
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import imageio

from environments.reach_env import ReachEnv


def scripted_action(env, target_q, gain=3.0):
    err = target_q - env.data.qpos
    return np.clip(err * gain, -1, 1)


def target_q_from_ik(target, L1=0.1, L2=0.1):
    x, y = target
    d = min(np.hypot(x, y), L1 + L2 - 1e-6)
    cos_e = np.clip((d**2 - L1**2 - L2**2) / (2 * L1 * L2), -1, 1)
    elbow = np.arccos(cos_e)
    shoulder = np.arctan2(y, x) - np.arctan2(L2 * np.sin(elbow), L1 + L2 * np.cos(elbow))
    return np.array([shoulder, elbow])


def run_episode(env, model, seed):
    obs, info = env.reset(seed=seed)
    frames = [env.render()]
    target_q = target_q_from_ik(env._get_target_pos())
    done = False
    while not done:
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = scripted_action(env, target_q)
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(env.render())
        done = terminated or truncated
    return frames, info["is_success"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None,
                         help="Path to a trained SB3 model .zip. If omitted, uses a scripted controller.")
    parser.add_argument("--out", type=str, default="results/demo_rollout.gif")
    parser.add_argument("--max-attempts", type=int, default=15,
                         help="Try this many seeds looking for a successful episode.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=0)
    args = parser.parse_args()

    env = ReachEnv(render_mode="rgb_array")

    model = None
    if args.model_path:
        from stable_baselines3 import PPO
        model = PPO.load(args.model_path)

    best_frames, found_success = None, False
    for i in range(args.max_attempts):
        seed = args.seed_start + i
        frames, success = run_episode(env, model, seed)
        if best_frames is None:
            best_frames = frames
        if success:
            best_frames = frames
            found_success = True
            print(f"Found a successful episode at seed {seed} ({len(frames)} frames).")
            break

    if not found_success:
        print(f"No success within {args.max_attempts} attempts -- saving the last episode anyway.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    imageio.mimsave(args.out, best_frames, fps=args.fps, loop=0)
    print(f"Saved GIF to {args.out} ({len(best_frames)} frames, {args.fps} fps)")


if __name__ == "__main__":
    main()
