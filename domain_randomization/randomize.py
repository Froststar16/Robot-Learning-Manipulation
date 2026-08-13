"""
Domain-randomized version of ReachEnv: on every reset, physical parameters
(joint damping, armature, actuator gear, link mass) are resampled within a
range. Train on this and evaluate zero-shot on the vanilla ReachEnv (or a
held-out randomization range) to measure sim-to-real-style generalization.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from environments.reach_env import ReachEnv


class DomainRandomizedReachEnv(ReachEnv):
    def __init__(
        self,
        damping_range: tuple = (0.5, 2.0),
        armature_range: tuple = (0.02, 0.10),
        gear_range: tuple = (7.0, 13.0),
        mass_scale_range: tuple = (0.7, 1.3),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.damping_range = damping_range
        self.armature_range = armature_range
        self.gear_range = gear_range
        self.mass_scale_range = mass_scale_range

        # Cache nominal values so randomization is always relative to the
        # original model, not compounded across resets.
        self._nominal_damping = self.model.dof_damping.copy()
        self._nominal_armature = self.model.dof_armature.copy()
        self._nominal_gear = self.model.actuator_gear.copy()
        self._nominal_mass = self.model.body_mass.copy()

    def _randomize_physics(self):
        n_dof = self.model.dof_damping.shape[0]
        self.model.dof_damping[:] = self.np_random.uniform(*self.damping_range, size=n_dof)
        self.model.dof_armature[:] = self.np_random.uniform(*self.armature_range, size=n_dof)

        gear = self.model.actuator_gear.copy()
        gear[:, 0] = self.np_random.uniform(*self.gear_range, size=gear.shape[0])
        self.model.actuator_gear[:] = gear

        mass_scale = self.np_random.uniform(*self.mass_scale_range, size=self._nominal_mass.shape)
        self.model.body_mass[:] = self._nominal_mass * mass_scale

    def reset(self, *, seed=None, options=None):
        # Seed the RNG first (super().reset does this), then randomize
        # physics using that same RNG for reproducibility.
        obs, info = super().reset(seed=seed, options=options)
        self._randomize_physics()
        return obs, info


def make_env(**kwargs):
    return DomainRandomizedReachEnv(**kwargs)
