"""
Record a trained pick-and-place policy as an animated GIF.

Same idea as evaluation/record_gif.py for the reach task, with one addition:
it retries seeds until it finds an episode that actually succeeds, and it
prints which seed worked so the GIF is reproducible. A README GIF that shows a
failure is worse than no GIF, and silently cherry-picking without recording
the seed makes the result unreproducible.

Usage:
    python evaluation/record_pick_place_gif.py --model-path logs/ppo_pp_v1/best_model.zip
    python evaluation/record_pick_place_gif.py --model-path ... --camera angled --difficulty 1.0

On headless Linux, prefix with MUJOCO_GL=osmesa (see README).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import imageio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environments.pick_place_env import PickPlaceEnv  # noqa: E402
from evaluation.evaluate_pick_place import load_model  # noqa: E402


def record_episode(env, model, seed: int, stride: int):
    obs, _ = env.reset(seed=seed)
    frames = [env.render()]
    step = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        step += 1
        if step % stride == 0:
            frames.append(env.render())
        if term or trunc:
            frames.append(env.render())
            return frames, info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--out", default="results/pick_place_rollout.gif")
    p.add_argument("--camera", default="angled",
                   choices=["angled", "topdown"])
    p.add_argument("--difficulty", type=float, default=1.0)
    p.add_argument("--max-tries", type=int, default=25)
    p.add_argument("--stride", type=int, default=2,
                   help="record every Nth control step (50 Hz control)")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--hold", type=int, default=12,
                   help="extra frames held on the final state")
    args = p.parse_args()

    model, algo = load_model(args.model_path)
    env = PickPlaceEnv(difficulty=args.difficulty, render_mode="rgb_array",
                       camera=args.camera)

    for seed in range(args.max_tries):
        frames, info = record_episode(env, model, seed, args.stride)
        if info["is_success"]:
            frames += [frames[-1]] * args.hold  # linger on the delivered box
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            imageio.mimsave(args.out, frames, fps=args.fps, loop=0)
            print(f"{algo} policy, seed {seed}: success in {len(frames)} frames")
            print(f"saved {args.out}")
            env.close()
            return
        print(f"seed {seed}: failed "
              f"(grasped={info['ever_grasped']}, "
              f"peak height={info['peak_box_height']:.3f} m) -- retrying")

    env.close()
    raise SystemExit(
        f"no successful episode in {args.max_tries} seeds. Either the policy "
        f"is weaker than the GIF implies, or difficulty={args.difficulty} is "
        f"above what it was trained to. Check with evaluate_pick_place.py "
        f"--sweep before recording."
    )


if __name__ == "__main__":
    main()
