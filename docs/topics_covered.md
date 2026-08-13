# Topics covered, and how

This page exists mainly for reviewers (recruiters, interviewers, PIs) who
want to quickly map the code in this repo to robot learning concepts,
without reading every file.

## Implemented and tested

| Topic | Where | Notes |
|---|---|---|
| MJCF modeling / forward kinematics | `environments/assets/reacher_arm.xml` | Hand-written 2-link planar arm. See the note below on a physics bug this surfaced. |
| Custom Gymnasium environment design | `environments/reach_env.py` | Direct `mujoco` Python bindings (not `dm_control`), observation/reward/termination design. |
| Model-free RL: on-policy (PPO) | `training/train_ppo.py` | Vectorized envs, `EvalCallback`, tensorboard logging. |
| Model-free RL: off-policy (SAC) | `training/train_sac.py` | For comparing sample efficiency against PPO. |
| Reward shaping | `reach_env.py: step()` | Dense negative-distance reward + control penalty + success bonus; documented rationale below. |
| Domain randomization / sim-to-real basics | `domain_randomization/randomize.py` | Randomizes damping, armature, actuator gear, link mass per episode. |
| Testing RL environments | `tests/test_env.py` | Includes an analytic-IK reachability check and a scripted-controller regression test — see below for why this matters. |

## Planned, scoped but not implemented

| Topic | Where | Notes |
|---|---|---|
| Contact-rich manipulation (grasping) | `environments/pick_place_env.py` | Left as a documented stub rather than a half-working implementation. |
| Imitation learning (BC / DAgger) | — | Needs `pick_place_env.py` first, since reach is close to trivially solvable by RL alone. |
| Sparse reward + HER | — | Natural fit once pick-and-place exists. |

## A debugging note worth reading

While building the reach environment, PPO trained for 60k timesteps without
any improvement in success rate (stuck at 0%). Before assuming it was a
hyperparameter problem, I checked whether the task was physically solvable
at all: I derived the analytic 2-link inverse kinematics for the arm,
confirmed sampled targets were always within reach, then tried a simple
proportional controller through the actual actuators (not by teleporting
joint angles) — and *that* also failed to converge.

That isolated the bug to the physics model, not the learning algorithm: the
arm links were resting exactly at the same height as the ground plane, so
MuJoCo's contact solver was generating friction forces that resisted joint
motion almost entirely. Disabling collision between the arm and the ground
(`contype`/`conaffinity`) fixed it immediately — a constant-torque command
that produced ~0.03 rad/s of motion before the fix produced the physically
expected ~10 rad/s steady-state velocity afterward.

The general lesson, and the reason this is worth including in a portfolio
repo: **when a learning curve is flat, check whether the environment is
solvable before tuning the algorithm.** A scripted controller or analytic
solution is usually the fastest way to separate an environment bug from a
learning problem, and it's much cheaper than a hyperparameter sweep.

## Reward design rationale

The reward is `-distance - 0.001 * ||action||^2`, plus a `+5.0` bonus on
success. Distance-based dense reward was chosen over sparse (reward only on
success) because:

- with only 2 DOF and no contact dynamics, dense reward is well-shaped and
  doesn't risk the reward hacking that's more common in higher-dimensional
  tasks
- it makes the learning curve visible from the first few thousand steps,
  which matters for fast iteration during development
- the small control penalty discourages high-frequency oscillation without
  meaningfully changing the optimal policy

For the planned pick-and-place task, sparse reward + HER is worth comparing
against shaped reward, since staged manipulation tasks are exactly where
reward shaping gets hard to get right and HER was designed to help.
