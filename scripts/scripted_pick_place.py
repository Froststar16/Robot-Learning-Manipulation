"""
A hand-written waypoint controller that solves the pick-and-place task
through the *real* actuators and the *real* physics.

Why this file exists
--------------------
On the reach task, PPO trained flat at 0% for 60k steps and it was not
obvious whether the bug was in the RL or in the simulation. What settled it
was deriving analytic IK and driving the arm with a scripted P-controller:
that failed too, which proved the problem was physics (arm links dragging on
the ground plane), not learning.

Pick-and-place has contact, a free-floating object and a staged reward, so
there are far more ways for the simulation to be quietly broken. This script
is the same diagnostic, written *before* any training run:

    if the scripted controller cannot solve the task,
    no amount of PPO/SAC tuning will either.

It is also the demonstration source for Phase 5 (behavior cloning vs RL):
`--save-demos` writes (observation, action) pairs that a BC policy can be
trained on.

Usage
-----
    python scripts/scripted_pick_place.py --episodes 20
    python scripts/scripted_pick_place.py --episodes 100 --difficulty 1.0
    python scripts/scripted_pick_place.py --episodes 200 --save-demos data/demos.npz
    python scripts/scripted_pick_place.py --episodes 1 --gif results/scripted.gif
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environments.pick_place_env import PickPlaceEnv, analytic_ik_3link  # noqa: E402


class ScriptedPickPlace:
    """Finite state machine over IK waypoints.

    States: APPROACH -> DESCEND -> CLOSE -> LIFT -> CARRY -> HOLD

    The controller only ever emits actions in the environment's own action
    space (delta joint targets + gripper command), so it is subject to exactly
    the same actuator limits and rate limits as a learned policy. That is what
    makes it a valid diagnostic.
    """

    APPROACH_HEIGHT = 0.13
    JOINT_TOL = 0.02
    CLOSE_STEPS = 12

    def __init__(self, env: PickPlaceEnv):
        self.env = env
        self.reset()

    def reset(self):
        self.state = "APPROACH"
        self.timer = 0
        self._last_q = self.env.data.qpos[:3].copy()

    # -- IK wrapper -------------------------------------------------------
    def _ik(self, x, z):
        q = analytic_ik_3link(
            x, z,
            l1=self.env.l1, l2=self.env.l2, le=self.env.le,
            base_x=self.env.base_xz[0], base_z=self.env.base_xz[1],
            approach=-np.pi / 2,   # point straight down for a top grasp
            elbow_up=True,
        )
        # Unreachable waypoint: hold the last valid target rather than
        # crashing. In practice this only fires if the sampling ranges in the
        # env are widened past the arm's workspace.
        if q is None:
            return self._last_q
        self._last_q = q
        return q

    def _drive(self, q_target, grip):
        """P-controller in joint space, clipped to the env's action range."""
        # Drive the *servo target* toward the waypoint, not the measured
        # angle. The env integrates action * delta_max into its target, so
        # feeding back measured position here would double-integrate and wind
        # the target up past the waypoint -- a textbook integrator-windup
        # oscillation, and the first bug this controller hit.
        err_cmd = q_target - self.env._ctrl[:3]
        a = np.clip(err_cmd / self.env.delta_max, -1.0, 1.0)
        # Arrival is judged on the *measured* angles, which is what actually
        # matters for lining the gripper up with the box.
        err_meas = float(np.abs(q_target - self.env.data.qpos[:3]).max())
        return np.concatenate([a, [grip]]), err_meas

    # -- policy -----------------------------------------------------------
    def act(self, obs=None):
        env = self.env
        box = env._box_pos()
        goal = env._goal_pos()

        if self.state == "APPROACH":
            q = self._ik(box[0], box[2] + self.APPROACH_HEIGHT)
            a, err = self._drive(q, +1.0)  # gripper open
            if err < self.JOINT_TOL:
                self.state = "DESCEND"
            return a

        if self.state == "DESCEND":
            q = self._ik(box[0], box[2])
            a, err = self._drive(q, +1.0)
            if err < self.JOINT_TOL:
                self.state, self.timer = "CLOSE", 0
            return a

        if self.state == "CLOSE":
            q = self._ik(box[0], box[2])
            a, _ = self._drive(q, -1.0)  # gripper closed
            self.timer += 1
            if self.timer >= self.CLOSE_STEPS:
                self.state = "LIFT"
            return a

        if self.state == "LIFT":
            q = self._ik(box[0], max(goal[2], 0.22))
            a, err = self._drive(q, -1.0)
            if err < self.JOINT_TOL:
                self.state = "CARRY"
            return a

        # CARRY / HOLD
        q = self._ik(goal[0], goal[2])
        a, _ = self._drive(q, -1.0)
        return a


def rollout(env, ctrl, collect=False, frames=None):
    obs, _ = env.reset()
    ctrl.reset()
    obs_buf, act_buf = [], []
    total_r, success = 0.0, False
    while True:
        a = ctrl.act(obs)
        if collect:
            obs_buf.append(obs)
            act_buf.append(a)
        obs, r, term, trunc, info = env.step(a)
        total_r += r
        if frames is not None:
            frames.append(env.render())
        success = success or info["is_success"]
        if term or trunc:
            break
    return success, total_r, obs_buf, act_buf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--difficulty", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-demos", type=str, default=None)
    p.add_argument("--gif", type=str, default=None)
    p.add_argument("--camera", type=str, default="topdown")
    args = p.parse_args()

    render = args.gif is not None
    env = PickPlaceEnv(
        difficulty=args.difficulty,
        render_mode="rgb_array" if render else None,
        camera=args.camera,
    )
    env.reset(seed=args.seed)
    ctrl = ScriptedPickPlace(env)

    successes, returns = 0, []
    all_obs, all_act = [], []
    frames = [] if render else None

    for ep in range(args.episodes):
        ok, ret, ob, ac = rollout(
            env, ctrl, collect=args.save_demos is not None,
            frames=frames if ep == 0 else None,
        )
        successes += int(ok)
        returns.append(ret)
        if ok and args.save_demos:
            all_obs.extend(ob)
            all_act.extend(ac)

    rate = successes / args.episodes
    print(f"scripted success rate: {successes}/{args.episodes} = {rate:.0%}")
    print(f"mean return: {np.mean(returns):.2f}")

    if args.save_demos and all_obs:
        os.makedirs(os.path.dirname(args.save_demos) or ".", exist_ok=True)
        np.savez_compressed(
            args.save_demos,
            observations=np.array(all_obs, dtype=np.float32),
            actions=np.array(all_act, dtype=np.float32),
        )
        print(f"saved {len(all_obs)} demo transitions -> {args.save_demos}")

    if render and frames:
        import imageio
        os.makedirs(os.path.dirname(args.gif) or ".", exist_ok=True)
        imageio.mimsave(args.gif, frames[::2], fps=25)
        print(f"saved gif -> {args.gif}")

    env.close()


if __name__ == "__main__":
    main()
