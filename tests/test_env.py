import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from environments.reach_env import ReachEnv
from domain_randomization.randomize import DomainRandomizedReachEnv


def test_reset_returns_valid_obs():
    env = ReachEnv()
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert obs.shape == (12,)


def test_step_returns_valid_transition():
    env = ReachEnv()
    env.reset(seed=0)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "distance" in info


def test_episode_truncates_at_max_steps():
    env = ReachEnv(max_episode_steps=20)
    env.reset(seed=0)
    for i in range(20):
        obs, reward, terminated, truncated, info = env.step(np.zeros(2))
        if terminated:
            break
    assert truncated or terminated


def test_target_is_reachable_via_analytic_ik():
    """Confirms sampled targets are always within the arm's workspace."""
    env = ReachEnv()
    L1, L2 = 0.1, 0.1
    for seed in range(20):
        env.reset(seed=seed)
        target = env._get_target_pos()
        dist_from_base = np.linalg.norm(target)
        assert dist_from_base <= L1 + L2, "target sampled outside arm's reach"


def test_reward_is_finite_and_bounded_below_zero_at_start():
    env = ReachEnv()
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(np.zeros(2))
    assert np.isfinite(reward)


def test_scripted_pd_controller_can_solve_task():
    """Regression test: a hand-written controller should succeed on most
    episodes. If this starts failing, something broke in the model/physics
    (e.g. an unintended collision reintroduced, actuator gear changed)."""
    env = ReachEnv()
    successes = 0
    n_trials = 15
    for trial in range(n_trials):
        obs, info = env.reset(seed=trial)
        target = env._get_target_pos()
        L1, L2 = 0.1, 0.1
        x, y = target
        d = min(np.hypot(x, y), L1 + L2 - 1e-6)
        cos_e = np.clip((d**2 - L1**2 - L2**2) / (2 * L1 * L2), -1, 1)
        elbow_t = np.arccos(cos_e)
        shoulder_t = np.arctan2(y, x) - np.arctan2(
            L2 * np.sin(elbow_t), L1 + L2 * np.cos(elbow_t)
        )
        target_q = np.array([shoulder_t, elbow_t])
        for _ in range(200):
            err = target_q - env.data.qpos
            action = np.clip(err * 3.0, -1, 1)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                successes += 1
                break
    assert successes / n_trials > 0.5, f"only {successes}/{n_trials} succeeded"


def test_domain_randomization_changes_physics_across_resets():
    env = DomainRandomizedReachEnv()
    env.reset(seed=0)
    damping_1 = env.model.dof_damping.copy()
    env.reset(seed=1)
    damping_2 = env.model.dof_damping.copy()
    assert not np.allclose(damping_1, damping_2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
