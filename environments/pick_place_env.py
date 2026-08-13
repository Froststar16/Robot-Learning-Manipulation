"""
Pick-and-place: a 3-DOF planar arm with a parallel-jaw gripper must pick up a
box and carry it to a target location.

This is Phase 3 of the roadmap and the contact-rich counterpart to
`reach_env.py`. Everything is built directly on the `mujoco` Python bindings
(no dm_control / robosuite), same as the reach task.

Design decisions worth being able to defend in an interview
-----------------------------------------------------------
1. **Joint-level position control, not raw torque.** The actuators in the
   MJCF are `position` servos. The policy outputs a small *delta* on each
   servo target, so it is doing residual/velocity-style control. Raw torque
   control on a gravity-loaded arm forces the policy to learn gravity
   compensation before it can learn the task, which wastes most of a small
   sample budget. Real manipulators expose a joint position interface for
   exactly this reason.

2. **Explicit gravity compensation.** Each physics substep applies
   `qfrc_applied = qfrc_bias` on the arm joints, which cancels gravity and
   Coriolis terms. This is standard on real arms and it makes the position
   servo track its target essentially exactly (steady-state error drops from
   ~0.023 rad to ~0 rad in this model). Without it, an open-loop IK
   controller misses the box by more than the box is wide.

3. **Selective contact, not global contact.** See the long comment in
   `assets/pick_place_arm.xml`. The reach task disabled *all* contact to fix
   a bug where arm links dragged on the ground; that fix cannot be copied
   here because grasping is contact. Instead, contact bitmasks allow exactly
   box/floor and box/fingertip pairs.

4. **Potential-based shaping.** The dense reward is built from *differences*
   of a potential function (negative distance), not from the raw distance.
   Potential-based shaping (Ng et al., 1999) provably leaves the optimal
   policy unchanged, which is a much safer thing to claim than "I tuned the
   reward until it worked".

5. **Staged reward.** Before a grasp exists, only gripper->box progress is
   rewarded. After it exists, only box->goal progress is. Rewarding both at
   once produces a policy that hovers near the box while nudging it toward
   the goal, because that collects both terms without ever committing to a
   grasp.

6. **Curriculum.** `difficulty` in [0, 1] widens the initial-state and goal
   distributions. At 0 the box is nearly always in the same place and the
   goal is a short lift directly above it; at 1 both are broadly randomised.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

_ASSET = os.path.join(os.path.dirname(__file__), "assets", "pick_place_arm.xml")

# Success threshold: box centre within this distance of the goal centre.
SUCCESS_RADIUS = 0.05
# Box centre height when resting on the floor (half the box edge).
BOX_REST_Z = 0.022
# Considered "lifted" above this height.
LIFT_Z = 0.075


# --------------------------------------------------------------------------
# Analytic inverse kinematics for the 3-link planar arm.
#
# Used by the scripted controller and by the tests, exactly like the 2-link
# IK in the reach task: if a hand-written controller driving the *real*
# actuators cannot solve the task, then a flat RL learning curve is a physics
# bug, not an RL bug. That distinction is what made the reach-task debugging
# tractable, so the same tool is provided here up front.
#
# Convention: joints rotate about +y. A rotation of +theta about +y maps
# +x toward -z, so the planar angle psi measured as atan2(z, x) satisfies
# psi = -theta. The solver works in psi and converts at the end.
# --------------------------------------------------------------------------
def analytic_ik_3link(
    x: float,
    z: float,
    l1: float,
    l2: float,
    le: float,
    base_x: float = 0.0,
    base_z: float = 0.0,
    approach: float = -np.pi / 2,
    elbow_up: bool = True,
) -> Optional[np.ndarray]:
    """Joint angles placing the grip site at world (x, z).

    `approach` is the absolute planar angle of the final link, i.e. the
    direction the gripper points. -pi/2 means pointing straight down, which
    is what a top-down grasp needs.

    Returns an array of 3 joint angles, or None if the pose is unreachable.
    """
    # Back off along the approach direction to find where the wrist must be.
    wx = x - le * np.cos(approach)
    wz = z - le * np.sin(approach)

    dx, dz = wx - base_x, wz - base_z
    r2 = dx * dx + dz * dz
    c2 = (r2 - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    if not -1.0 <= c2 <= 1.0:
        return None  # outside the annulus the two links can span

    psi2 = np.arccos(c2)
    if elbow_up:
        psi2 = -psi2
    psi1 = np.arctan2(dz, dx) - np.arctan2(l2 * np.sin(psi2), l1 + l2 * np.cos(psi2))
    psi3 = approach - psi1 - psi2

    return -np.array([psi1, psi2, psi3])  # psi -> theta


class PickPlaceEnv(gym.Env):
    """Gymnasium environment for the pick-and-place task."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 250,
        frame_skip: int = 10,
        difficulty: float = 0.0,
        reward_type: str = "dense",
        gravity_compensation: bool = True,
        camera: str = "topdown",
        width: int = 640,
        height: int = 480,
    ) -> None:
        super().__init__()
        if reward_type not in ("dense", "sparse"):
            raise ValueError("reward_type must be 'dense' or 'sparse'")

        self.model = mujoco.MjModel.from_xml_path(_ASSET)
        self.data = mujoco.MjData(self.model)

        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self.reward_type = reward_type
        self.gravity_compensation = gravity_compensation
        self.difficulty = float(np.clip(difficulty, 0.0, 1.0))

        self.render_mode = render_mode
        self._camera = camera
        self._width, self._height = width, height
        self._renderer: Optional[mujoco.Renderer] = None

        # ---- cache MuJoCo ids so we never do name lookups in the hot loop
        n = mujoco.mj_name2id
        self._sid_grip = n(self.model, mujoco.mjtObj.mjOBJ_SITE, "grip_site")
        self._sid_goal = n(self.model, mujoco.mjtObj.mjOBJ_SITE, "goal")
        self._bid_box = n(self.model, mujoco.mjtObj.mjOBJ_BODY, "box")
        self._gid_box = n(self.model, mujoco.mjtObj.mjOBJ_GEOM, "g_box")
        self._gid_padl = n(self.model, mujoco.mjtObj.mjOBJ_GEOM, "pad_left")
        self._gid_padr = n(self.model, mujoco.mjtObj.mjOBJ_GEOM, "pad_right")
        self._jid_box = n(self.model, mujoco.mjtObj.mjOBJ_JOINT, "box_free")
        self._box_qadr = self.model.jnt_qposadr[self._jid_box]
        self._box_vadr = self.model.jnt_dofadr[self._jid_box]

        # ---- link lengths read out of the model rather than hardcoded, so
        #      editing the XML cannot silently desync the IK.
        bid = lambda nm: n(self.model, mujoco.mjtObj.mjOBJ_BODY, nm)
        self.base_xz = (
            float(self.model.body_pos[bid("link1")][0]),
            float(self.model.body_pos[bid("link1")][2]),
        )
        self.l1 = float(self.model.body_pos[bid("link2")][0])
        self.l2 = float(self.model.body_pos[bid("link3")][0])
        self.le = float(
            self.model.body_pos[bid("palm")][0]
            + self.model.site_pos[self._sid_grip][0]
        )

        # ---- joint / actuator limits
        self.arm_range = self.model.jnt_range[:3].copy()          # (3, 2)
        self.grip_open = float(self.model.jnt_range[3, 1])        # 0.045
        self.grip_closed = float(self.model.jnt_range[3, 0])      # 0.0
        self.delta_max = 0.08  # rad of servo-target change per control step

        # ---- spaces
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        obs_dim = 33
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._ctrl = np.zeros(self.model.nu)
        self._step_count = 0
        self._prev_reach_pot = 0.0
        self._prev_place_pot = 0.0
        self._had_grasp = False
        self._gave_grasp_bonus = False
        self._gave_lift_bonus = False

    # ------------------------------------------------------------------ core
    def set_difficulty(self, difficulty: float) -> None:
        """Curriculum hook. 0 = easiest, 1 = full randomisation."""
        self.difficulty = float(np.clip(difficulty, 0.0, 1.0))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        d = self.difficulty

        # Home pose: arm folded up and clear of the table, fingers open.
        home = np.array([0.50, -1.30, -0.70])
        self.data.qpos[:3] = home
        self.data.qpos[3:5] = self.grip_open
        self._ctrl[:3] = home
        self._ctrl[3:] = self.grip_open

        # Box: on the floor, x widens with difficulty.
        box_x = self.np_random.uniform(0.40 - 0.03 - 0.07 * d,
                                       0.40 + 0.03 + 0.07 * d)
        self.data.qpos[self._box_qadr + 0] = box_x
        self.data.qpos[self._box_qadr + 1] = 0.0
        self.data.qpos[self._box_qadr + 2] = BOX_REST_Z
        self.data.qpos[self._box_qadr + 3 : self._box_qadr + 7] = [1, 0, 0, 0]

        # Goal: at d=0 a short lift straight up; at d=1 anywhere sensible.
        goal_x = box_x + self.np_random.uniform(-1, 1) * (0.02 + 0.16 * d)
        goal_x = float(np.clip(goal_x, 0.28, 0.54))
        goal_z = float(self.np_random.uniform(0.14 - 0.04 * d, 0.18 + 0.16 * d))
        self.model.site_pos[self._sid_goal] = [goal_x, 0.0, goal_z]

        mujoco.mj_forward(self.model, self.data)
        # Let the box settle onto the floor before the episode starts.
        for _ in range(20):
            self._physics_step()

        self._step_count = 0
        self._had_grasp = False
        self._gave_grasp_bonus = False
        self._gave_lift_bonus = False
        self._prev_reach_pot = -self._dist_grip_box()
        self._prev_place_pot = -self._dist_box_goal()

        return self._get_obs(), self._info(0.0)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

        # Arm: residual position control.
        self._ctrl[:3] = np.clip(
            self._ctrl[:3] + action[:3] * self.delta_max,
            self.arm_range[:, 0],
            self.arm_range[:, 1],
        )
        # Gripper: absolute command, so the policy can commit to closing.
        grip_cmd = self.grip_closed + (action[3] + 1.0) * 0.5 * (
            self.grip_open - self.grip_closed
        )
        self._ctrl[3:] = grip_cmd

        self.data.ctrl[:] = self._ctrl
        for _ in range(self.frame_skip):
            self._physics_step()

        self._step_count += 1
        reward, success = self._reward(action)
        terminated = bool(success)
        truncated = self._step_count >= self.max_episode_steps
        return self._get_obs(), reward, terminated, truncated, self._info(reward)

    def _physics_step(self) -> None:
        if self.gravity_compensation:
            # Cancel gravity + Coriolis on the arm joints only (dofs 0:5 are
            # the 3 arm joints and 2 fingers; the box's 6 free dofs must keep
            # their gravity or it would float).
            self.data.qfrc_applied[:5] = self.data.qfrc_bias[:5]
        mujoco.mj_step(self.model, self.data)

    # -------------------------------------------------------------- observation
    def _grip_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._sid_grip].copy()

    def _goal_pos(self) -> np.ndarray:
        return self.model.site_pos[self._sid_goal].copy()

    def _box_pos(self) -> np.ndarray:
        return self.data.xpos[self._bid_box].copy()

    def _dist_grip_box(self) -> float:
        return float(np.linalg.norm(self._grip_pos() - self._box_pos()))

    def _dist_box_goal(self) -> float:
        return float(np.linalg.norm(self._box_pos() - self._goal_pos()))

    def is_grasped(self) -> bool:
        """True iff *both* fingertip pads are currently in contact with the box.

        Reading `data.contact` directly is the honest way to do this: it asks
        the physics engine what is actually touching, rather than inferring a
        grasp from finger positions (which would report success while the box
        sits on the table between open fingers).
        """
        left = right = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = (c.geom1, c.geom2)
            if self._gid_box in pair:
                other = pair[0] if pair[1] == self._gid_box else pair[1]
                if other == self._gid_padl:
                    left = True
                elif other == self._gid_padr:
                    right = True
        return left and right

    def _get_obs(self) -> np.ndarray:
        qa = int(self._box_qadr)
        va = int(self._box_vadr)
        box_pos = self._box_pos()
        grip = self._grip_pos()
        goal = self._goal_pos()
        obs = np.concatenate(
            [
                self.data.qpos[:3],                       # arm joint angles   3
                self.data.qvel[:3],                       # arm joint vels     3
                self.data.qpos[3:5],                      # finger positions   2
                self.data.qvel[3:5],                      # finger velocities  2
                grip,                                     # gripper xyz        3
                box_pos,                                  # box xyz            3
                self.data.qpos[qa + 3 : qa + 7],          # box orientation    4
                self.data.qvel[va : va + 3],              # box linear vel     3
                goal,                                     # goal xyz           3
                box_pos - grip,                           # gripper->box       3
                goal - box_pos,                           # box->goal          3
                [1.0 if self.is_grasped() else 0.0],      # grasp flag         1
            ]
        )
        return obs.astype(np.float32)

    # ------------------------------------------------------------------ reward
    def _reward(self, action: np.ndarray):
        d_place = self._dist_box_goal()
        success = d_place < SUCCESS_RADIUS

        if self.reward_type == "sparse":
            # Goal-conditioned convention that HER expects: 0 on success,
            # -1 otherwise. Kept as an explicit comparison point against the
            # shaped reward below.
            return (0.0 if success else -1.0), success

        grasped = self.is_grasped()
        box_z = self._box_pos()[2]
        reach_pot = -self._dist_grip_box()
        place_pot = -d_place

        r = 0.0
        if grasped:
            # Stage 2: only carry progress counts.
            r += 20.0 * (place_pot - self._prev_place_pot)
            if not self._gave_grasp_bonus:
                r += 2.0
                self._gave_grasp_bonus = True
            if not self._gave_lift_bonus and box_z > LIFT_Z:
                r += 3.0
                self._gave_lift_bonus = True
        else:
            # Stage 1: only approach progress counts.
            r += 10.0 * (reach_pot - self._prev_reach_pot)
            if self._had_grasp and box_z < LIFT_Z:
                r -= 1.0  # dropped it

        r -= 0.005 * float(np.sum(np.square(action)))
        if success:
            r += 10.0

        self._prev_reach_pot = reach_pot
        self._prev_place_pot = place_pot
        self._had_grasp = grasped
        return float(r), success

    def _info(self, reward: float) -> dict[str, Any]:
        return {
            "is_success": self._dist_box_goal() < SUCCESS_RADIUS,
            "is_grasped": self.is_grasped(),
            "dist_grip_box": self._dist_grip_box(),
            "dist_box_goal": self._dist_box_goal(),
            "box_height": float(self._box_pos()[2]),
            "difficulty": self.difficulty,
        }

    # ------------------------------------------------------------------ render
    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, self._height, self._width)
        self._renderer.update_scene(self.data, camera=self._camera)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
