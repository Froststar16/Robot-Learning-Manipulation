"""
A 2-link planar arm reaching task built directly on the MuJoCo Python
bindings (not dm_control), wrapped as a Gymnasium environment.

Task: move the end effector to a randomly sampled target position.
Observation: joint angles, joint velocities, end-effector position,
             target position, vector from end effector to target.
Action: torque on shoulder and elbow joints (continuous, [-1, 1]).
Reward: dense, negative distance to target + small control penalty.
"""

import os
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco


ASSET_PATH = os.path.join(os.path.dirname(__file__), "assets", "reacher_arm.xml")


class ReachEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 200,
        target_radius_range: tuple = (0.10, 0.18),
        success_threshold: float = 0.02,
        control_penalty_weight: float = 0.001,
    ):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(ASSET_PATH)
        self.data = mujoco.MjData(self.model)

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.target_radius_range = target_radius_range
        self.success_threshold = success_threshold
        self.control_penalty_weight = control_penalty_weight

        self._step_count = 0
        self._renderer = None

        # obs = [cos(q1), sin(q1), cos(q2), sin(q2), qvel(2), ee_pos(2),
        #        target_pos(2), ee_to_target(2)]  -> 12-dim
        obs_high = np.full(12, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
        )

        self._ee_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "end_effector"
        )
        self._target_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target"
        )

    def _get_ee_pos(self) -> np.ndarray:
        return self.data.xpos[self._ee_body_id][:2].copy()

    def _get_target_pos(self) -> np.ndarray:
        return self.data.xpos[self._target_body_id][:2].copy()

    def _get_obs(self) -> np.ndarray:
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        ee_pos = self._get_ee_pos()
        target_pos = self._get_target_pos()
        return np.concatenate(
            [
                np.cos(qpos),
                np.sin(qpos),
                qvel,
                ee_pos,
                target_pos,
                ee_pos - target_pos,
            ]
        ).astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Randomize initial joint angles slightly around a neutral pose.
        self.data.qpos[:] = self.np_random.uniform(-0.3, 0.3, size=self.model.nq)
        self.data.qvel[:] = 0.0

        # Sample a target at a random radius/angle within reach of the arm
        # (link1 + link2 length = 0.2, so keep targets comfortably inside).
        radius = self.np_random.uniform(*self.target_radius_range)
        angle = self.np_random.uniform(-np.pi, np.pi)
        target_x, target_y = radius * np.cos(angle), radius * np.sin(angle)
        target_body_jntadr = self.model.body("target").jntadr[0] if self.model.body("target").jntadr.size else None
        # The target body has no joint (it's static in the XML), so we move
        # it by editing its position directly via mocap-free body pos override.
        self.model.body_pos[self._target_body_id][:2] = [target_x, target_y]

        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        ee_pos = self._get_ee_pos()
        target_pos = self._get_target_pos()
        dist = float(np.linalg.norm(ee_pos - target_pos))

        control_penalty = self.control_penalty_weight * float(np.square(action).sum())
        reward = -dist - control_penalty

        terminated = dist < self.success_threshold
        if terminated:
            reward += 5.0  # success bonus

        truncated = self._step_count >= self.max_episode_steps

        info = {"distance": dist, "is_success": terminated}
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=240, width=240)
        self._renderer.update_scene(self.data)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def make_env(**kwargs):
    """Factory function so this can be registered with gymnasium.make if desired."""
    return ReachEnv(**kwargs)
