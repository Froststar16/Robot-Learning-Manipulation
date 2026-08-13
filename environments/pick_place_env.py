"""
STATUS: scaffolded, not yet implemented -- this is the Phase 3 extension
(see docs/topics_covered.md for the full roadmap).

Planned task: a 3-DOF arm with a simple parallel-jaw gripper must pick up a
free box object and place it at a target location. This introduces contact-
rich dynamics (grasping), which ReachEnv deliberately avoids.

Implementation plan:
    1. Extend reacher_arm.xml (or a new pick_place_arm.xml) with:
         - a 3rd joint for a wrist, and a 2-DOF gripper (two fingers)
         - a free-floating box body with a joint type="free"
         - re-enable contype/conaffinity for the gripper fingers and the
           box (they were disabled in reacher_arm.xml -- see note in that
           file about why contact was the source of an earlier physics bug)
    2. Observation: joint angles/velocities, gripper state, box pose,
       box-to-target vector, end-effector-to-box vector.
    3. Reward: staged/shaped -- reach toward box, close gripper when near,
       lift, move to target, release. Consider a sparse-reward variant
       with HER (Hindsight Experience Replay, supported by SB3's SAC) as
       a comparison point against the shaped-reward version -- this is a
       good place to demonstrate reward-design tradeoffs in a writeup.
    4. Curriculum: start with the box close to the gripper and the target
       close to the box, then progressively widen the initial-state
       distribution as success rate improves.

This file is intentionally left as a scoped placeholder rather than a
half-working implementation -- see the project README for why.
"""

raise NotImplementedError(
    "PickPlaceEnv is planned but not yet implemented. See the module "
    "docstring in this file for the implementation plan, and "
    "docs/topics_covered.md for how this fits into the project roadmap."
)
