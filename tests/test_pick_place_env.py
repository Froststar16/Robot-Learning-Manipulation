"""
Tests for PickPlaceEnv.

The philosophy is the same as tests/test_env.py for the reach task: verify
the *simulation* independently of the *learning*, so that a flat training
curve can be attributed to one or the other. The two tests that carry the
most weight here are:

  - test_analytic_ik_matches_forward_kinematics: the IK we use to script
    the task agrees with MuJoCo's own forward kinematics.
  - test_scripted_controller_solves_task: a hand-written controller, using
    only the env's public action space, actually solves the task. If this
    ever fails, the environment is broken and no RL result from it means
    anything.
"""

import os
import sys

import numpy as np
import pytest
import mujoco

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.pick_place_env import (  # noqa: E402
    PickPlaceEnv,
    analytic_ik_3link,
    SUCCESS_RADIUS,
    LIFT_Z,
)
from scripts.scripted_pick_place import ScriptedPickPlace  # noqa: E402


@pytest.fixture(scope="module")
def env():
    e = PickPlaceEnv(difficulty=0.0)
    yield e
    e.close()


def test_spaces_and_reset(env):
    obs, info = env.reset(seed=0)
    assert env.action_space.shape == (4,)
    assert obs.shape == env.observation_space.shape
    assert np.all(np.isfinite(obs))
    assert info["is_grasped"] is False
    assert not info["is_success"]


def test_seeding_is_deterministic():
    a, b = PickPlaceEnv(difficulty=1.0), PickPlaceEnv(difficulty=1.0)
    o1, _ = a.reset(seed=123)
    o2, _ = b.reset(seed=123)
    np.testing.assert_allclose(o1, o2, atol=1e-8)
    o3, _ = a.reset(seed=124)
    assert not np.allclose(o1, o3)
    a.close(); b.close()


def test_box_starts_resting_on_the_floor(env):
    env.reset(seed=7)
    assert env._box_pos()[2] == pytest.approx(0.022, abs=3e-3)
    assert abs(env._box_pos()[1]) < 1e-6  # stays in the x-z plane


def test_analytic_ik_matches_forward_kinematics(env):
    """IK -> qpos -> MuJoCo FK should land the grip site on the target."""
    env.reset(seed=0)
    targets = [(0.35, 0.05), (0.40, 0.02), (0.45, 0.25), (0.30, 0.30)]
    for x, z in targets:
        q = analytic_ik_3link(
            x, z, l1=env.l1, l2=env.l2, le=env.le,
            base_x=env.base_xz[0], base_z=env.base_xz[1],
            approach=-np.pi / 2, elbow_up=True,
        )
        assert q is not None, f"({x}, {z}) should be reachable"
        env.data.qpos[:3] = q
        mujoco.mj_forward(env.model, env.data)
        grip = env._grip_pos()
        assert grip[0] == pytest.approx(x, abs=1e-6)
        assert grip[2] == pytest.approx(z, abs=1e-6)


def test_ik_rejects_unreachable_targets(env):
    assert analytic_ik_3link(
        2.0, 0.0, l1=env.l1, l2=env.l2, le=env.le,
        base_x=env.base_xz[0], base_z=env.base_xz[1],
    ) is None


def test_contact_filtering_excludes_the_arm(env):
    """Regression test for the physics bug found on the reach task.

    There, arm links resting on the ground plane generated friction that
    fought every joint motion. Here contact is re-enabled *selectively*, so
    the invariant to protect is: no contact ever involves an arm link.
    """
    arm_geoms = {
        mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, g)
        for g in ("g_link1", "g_link2", "g_link3", "g_palm", "pillar")
    }
    env.reset(seed=3)
    for _ in range(120):
        env.step(env.action_space.sample())
        for i in range(env.data.ncon):
            c = env.data.contact[i]
            assert c.geom1 not in arm_geoms and c.geom2 not in arm_geoms


def test_gravity_compensation_tracks_targets(env):
    """With gravity compensation the position servos should have ~no
    steady-state error; without it they sag. This is what lets an open-loop
    IK waypoint actually land on a 4.4 cm box."""
    env.reset(seed=0)
    target = np.array([0.3, -0.9, -0.6])
    env.data.ctrl[:3] = target
    env._ctrl[:3] = target
    for _ in range(150):
        env._physics_step()
    assert np.abs(env.data.qpos[:3] - target).max() < 5e-3


def test_sparse_reward_convention():
    e = PickPlaceEnv(difficulty=0.0, reward_type="sparse")
    e.reset(seed=0)
    _, r, _, _, info = e.step(np.zeros(4))
    assert r == (0.0 if info["is_success"] else -1.0)
    e.close()


def test_action_clipping_is_safe(env):
    env.reset(seed=0)
    obs, r, term, trunc, _ = env.step(np.array([50.0, -50.0, 50.0, -50.0]))
    assert np.all(np.isfinite(obs)) and np.isfinite(r)
    lo, hi = env.arm_range[:, 0], env.arm_range[:, 1]
    assert np.all(env._ctrl[:3] >= lo - 1e-9) and np.all(env._ctrl[:3] <= hi + 1e-9)


def test_episode_truncates_at_the_step_limit():
    e = PickPlaceEnv(difficulty=0.0, max_episode_steps=25)
    e.reset(seed=0)
    for i in range(25):
        _, _, term, trunc, _ = e.step(np.zeros(4))
    assert trunc and not term
    e.close()


@pytest.mark.parametrize("difficulty", [0.0, 1.0])
def test_scripted_controller_solves_task(difficulty):
    """The load-bearing test: the task is physically solvable.

    A flat RL curve on an env that passes this test is an RL problem.
    A flat RL curve on an env that fails it is a physics problem.
    """
    e = PickPlaceEnv(difficulty=difficulty)
    ctrl = ScriptedPickPlace(e)
    successes = 0
    for ep in range(5):
        obs, _ = e.reset(seed=100 + ep)
        ctrl.reset()
        while True:
            obs, _, term, trunc, info = e.step(ctrl.act(obs))
            if term or trunc:
                break
        successes += int(info["is_success"])
    assert successes >= 4, f"scripted controller only solved {successes}/5"
    e.close()


def test_success_requires_the_box_not_the_gripper(env):
    """Sanity check on the success condition: it is defined on the box, so
    parking an empty gripper on the goal must not count."""
    env.reset(seed=0)
    goal = env._goal_pos()
    q = analytic_ik_3link(
        goal[0], goal[2], l1=env.l1, l2=env.l2, le=env.le,
        base_x=env.base_xz[0], base_z=env.base_xz[1],
    )
    if q is not None:
        env.data.qpos[:3] = q
        mujoco.mj_forward(env.model, env.data)
        assert np.linalg.norm(env._grip_pos() - goal) < SUCCESS_RADIUS
        assert not env._info(0.0)["is_success"]


# ---------------------------------------------------------------------------
# Sparse reward-mode parity
#
# reward_type="sparse" could previously never report success: the grasp/
# success bookkeeping (_ever_grasped, _grasp_streak, _peak_box_z) lived
# inside the dense branch of _reward(), below the early return for sparse
# mode, so it was never updated when reward_type="sparse". Sparse mode
# returned a constant -1, never terminated, and every info-dict diagnostic
# tied to grasping was permanently dead. The fix pulls that bookkeeping into
# _update_task_state(), which now runs before the reward-mode branch.
#
# The invariant these protect: task state -- what actually happened in the
# episode -- must not depend on which reward function is scoring it.
# ---------------------------------------------------------------------------

# info keys that are physics-derived and must match between reward modes.
TASK_STATE_KEYS = [
    "is_success",
    "at_goal",
    "box_speed",
    "is_grasped",
    "dist_grip_box",
    "dist_box_goal",
    "box_height",
    "ever_grasped",
    "peak_carry_height",
]


def _rollout(reward_type, actions, seed, **kwargs):
    """Step a fresh env through a fixed action sequence, recording task state."""
    e = PickPlaceEnv(reward_type=reward_type, **kwargs)
    _, info = e.reset(seed=seed)
    trace = [{k: info[k] for k in TASK_STATE_KEYS}]
    rewards = []
    for a in actions:
        _, r, terminated, truncated, info = e.step(a)
        trace.append({k: info[k] for k in TASK_STATE_KEYS})
        rewards.append(r)
        if terminated or truncated:
            break
    e.close()
    return trace, rewards


def test_task_state_is_independent_of_reward_type():
    """Dense and sparse envs must agree on every physics-derived quantity.

    grasp_init_prob=1.0 is load-bearing: with random actions from the true
    start state the box is never grasped, so is_grasped/ever_grasped would
    be False in both modes and the bug would slip through undetected.
    Forcing a pre-placed grasp is what makes the two traces diverge under
    the old code.
    """
    rng = np.random.default_rng(0)
    actions = rng.uniform(-1.0, 1.0, size=(60, 4)).astype(np.float32)

    dense, _ = _rollout("dense", actions, seed=0, grasp_init_prob=1.0)
    sparse, _ = _rollout("sparse", actions, seed=0, grasp_init_prob=1.0)

    assert len(dense) == len(sparse), (
        "episodes ended at different steps -- termination must not depend "
        "on the reward mode"
    )
    for step, (d, s) in enumerate(zip(dense, sparse)):
        for key in TASK_STATE_KEYS:
            assert d[key] == pytest.approx(s[key]), (
                f"info[{key!r}] diverged at step {step}: "
                f"dense={d[key]} sparse={s[key]}"
            )


def test_sparse_mode_tracks_grasp_state():
    """Narrow regression: ever_grasped must update in sparse mode.

    Under the old code this stayed False for the entire episode regardless
    of what the arm did, which is the one fact that made sparse success
    unreachable.
    """
    rng = np.random.default_rng(1)
    actions = (0.2 * rng.uniform(-1.0, 1.0, size=(30, 4))).astype(np.float32)

    trace, _ = _rollout("sparse", actions, seed=0, grasp_init_prob=1.0)

    assert any(s["is_grasped"] for s in trace), (
        "no stable grasp registered in sparse mode despite starting "
        "pre-grasped"
    )
    assert trace[-1]["ever_grasped"]
    assert trace[-1]["peak_carry_height"] > LIFT_Z


def test_sparse_reward_is_bounded_and_terminal_is_zero():
    """Sparse reward takes only {-1, 0}, and 0 iff success.

    Guards the SB3/HER convention: a stray shaping term leaking into the
    sparse path would silently break goal relabelling later, so this is
    checked at the source rather than discovered as a confusing HER result.
    """
    rng = np.random.default_rng(2)
    actions = rng.uniform(-1.0, 1.0, size=(40, 4)).astype(np.float32)

    trace, rewards = _rollout("sparse", actions, seed=0, grasp_init_prob=1.0)

    assert set(np.round(rewards, 12)).issubset({-1.0, 0.0})
    for r, s in zip(rewards, trace[1:]):
        assert (r == 0.0) == s["is_success"]


def test_sparse_task_is_solvable_end_to_end():
    """Strong guard: the scripted controller must succeed under sparse
    reward, not just agree with dense mode on paper. Parity proves the two
    modes agree; this proves the thing they agree on is reachable."""
    e = PickPlaceEnv(reward_type="sparse", difficulty=0.0)
    obs, info = e.reset(seed=0)
    ctrl = ScriptedPickPlace(e)

    reward = None
    for _ in range(e.max_episode_steps):
        obs, reward, term, trunc, info = e.step(ctrl.act(obs))
        if term or trunc:
            break

    assert info["is_success"], "scripted controller failed under sparse reward"
    assert reward == 0.0
    assert info["ever_grasped"]
    e.close()
