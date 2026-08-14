"""
Plot the pick-and-place training curve, annotated with curriculum steps.

The curriculum annotation is the point of this script. A raw success-rate
curve for a curriculum run is misleading: every time the difficulty increases
the success rate drops, so the curve looks like the policy is getting worse
when it is actually being handed a harder problem. Marking the difficulty
changes turns a confusing sawtooth into a readable story.

Usage:
    python evaluation/plot_pick_place.py --run-dir logs/ppo_pp_v1
    python evaluation/plot_pick_place.py --run-dir logs/ppo_pp_v1 --out results/pick_place_curve.png
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read_curriculum_steps(run_dir: str):
    """Pull (timestep, difficulty) from the tensorboard event file, if present."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        return []

    events = glob.glob(os.path.join(run_dir, "*", "events.out.tfevents.*"))
    if not events:
        return []
    ea = EventAccumulator(max(events, key=os.path.getmtime))
    ea.Reload()
    if "curriculum/difficulty" not in ea.Tags().get("scalars", []):
        return []

    changes, last = [], None
    for s in ea.Scalars("curriculum/difficulty"):
        if last is None or s.value > last + 1e-9:
            changes.append((s.step, s.value))
            last = s.value
    return changes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--title", default="Pick-and-place: success rate over training")
    p.add_argument("--smooth", type=int, default=5,
                   help="moving-average window over eval checkpoints")
    args = p.parse_args()

    # A run resumed with --resume leaves several evaluation files behind, one
    # per segment. Merge them so the curve covers the whole of training rather
    # than just the last chunk.
    files = sorted(glob.glob(os.path.join(args.run_dir, "evaluations*.npz")))
    if not files:
        raise SystemExit(f"no evaluations*.npz in {args.run_dir} -- "
                         "was EvalCallback enabled?")

    steps_all, succ_all = [], []
    for f in files:
        d = np.load(f)
        if "successes" not in d:
            raise SystemExit(f"{f} has no success data -- the env must report "
                             "is_success in info.")
        steps_all.append(d["timesteps"])
        succ_all.append(np.array(d["successes"]).mean(axis=1))

    steps = np.concatenate(steps_all)
    succ = np.concatenate(succ_all)
    order = np.argsort(steps)
    steps, succ = steps[order], succ[order]
    if len(files) > 1:
        print(f"merged {len(files)} evaluation segments")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, succ, color="#9ab", lw=1, alpha=0.8, label="eval success rate")

    if args.smooth > 1 and len(succ) >= args.smooth:
        k = np.ones(args.smooth) / args.smooth
        sm = np.convolve(succ, k, mode="valid")
        ax.plot(steps[args.smooth - 1:], sm, color="#1f6fb4", lw=2.2,
                label=f"{args.smooth}-eval moving average")

    for step, diff in read_curriculum_steps(args.run_dir):
        ax.axvline(step, color="#c0392b", ls="--", lw=1, alpha=0.6)
        ax.text(step, 1.02, f"d={diff:.1f}", rotation=0, fontsize=8,
                color="#c0392b", ha="center", va="bottom")

    ax.set_xlabel("environment steps")
    ax.set_ylabel("success rate")
    ax.set_ylim(-0.03, 1.08)
    ax.set_title(args.title)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.text(0.01, 0.01,
             "dashed lines = curriculum difficulty increases; dips after them "
             "are the task getting harder, not the policy getting worse",
             fontsize=8, color="#555")

    out = args.out or os.path.join("results", "pick_place_curve.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved {out}")
    print(f"final success rate (last 5 evals): {succ[-5:].mean():.0%}")


if __name__ == "__main__":
    main()
