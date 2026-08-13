# Pick-and-place: what's going on, and why

Companion to `docs/topics_covered.md`, covering Phase 3. This is written so
that every design choice in `environments/pick_place_env.py` and
`environments/assets/pick_place_arm.xml` can be defended out loud.

---

## 1. Why pick-and-place is a genuinely harder problem than reach

Reach is a *smooth* problem: the reward is a monotone function of one
distance, the optimal policy is close to "move down the gradient", and there
is no state where the arm has to commit to something irreversible.

Pick-and-place breaks all three of those.

| Property | Reach | Pick-and-place |
|---|---|---|
| Dynamics | Smooth, contact-free | Discontinuous at every contact event |
| Reward landscape | Single basin | Two stages separated by a bottleneck (the grasp) |
| Horizon | ~50 steps | ~150-250 steps |
| State | Arm only | Arm + a 6-DOF object with its own dynamics |
| Failure modes | Slow convergence | Dropping, knocking the object away, squeezing it out |

The grasp is a **bottleneck state**: almost no trajectory reaches the goal
without passing through it, and random exploration finds it rarely. That is
the single fact that drives most of the design below.

---

## 2. Contact modelling, and why the reach-task fix could not be reused

On the reach task, the arm trained flat at 0% because the links rested on
the ground plane and the contact solver generated friction that fought all
joint motion. The fix was `contype="0" conaffinity="0"` — disabling contact
globally.

That fix is *unavailable* here, because grasping is contact. So the model
uses MuJoCo's contact bitmask filtering instead. Two geoms collide iff:

```
(A.contype & B.conaffinity) != 0   or   (B.contype & A.conaffinity) != 0
```

with the assignments:

| Geom | contype | conaffinity |
|---|---|---|
| floor | 1 | 1 |
| box | 4 | 1 |
| fingertip pads | 2 | 4 |
| arm links, palm, pillar | 0 | 0 |

which produces exactly two live contact pairs — box/floor and box/pad — and
no others. `tests/test_pick_place_env.py::test_contact_filtering_excludes_the_arm`
asserts this invariant over 120 random-action steps, so the original bug
cannot silently come back.

**Talking point:** contact filtering is not a hack to work around a bug; it
is how you express "these two parts of my robot are not supposed to be
collision-checked against each other" in any physics engine. Real robot
models do the same thing for adjacent links.

---

## 3. Control interface: position servos + gravity compensation

The actuators are MuJoCo `position` actuators — a PD servo per joint — and
the policy's action is a bounded *delta* on each servo target:

```
ctrl[i] <- clip(ctrl[i] + a[i] * delta_max, joint_lo, joint_hi)
```

with `delta_max = 0.08 rad` per 50 Hz control step (≈ 4 rad/s at full
deflection).

Three reasons, in order of importance:

1. **It matches real hardware.** Almost every commercial manipulator exposes
   a joint position or joint velocity interface with a servo underneath. A
   policy trained on raw torques is trained against an interface most robots
   do not offer.
2. **It removes gravity from the learning problem.** Under torque control
   the policy must first learn to hold the arm up before it can learn
   anything about the task, which eats most of a small sample budget.
3. **Delta actions give smooth trajectories.** Absolute position targets let
   a policy teleport its setpoint across the workspace between steps, which
   produces violent motion and bad contact behaviour.

On top of that, each physics substep applies

```python
data.qfrc_applied[:5] = data.qfrc_bias[:5]
```

which cancels gravity and Coriolis terms on the arm and finger DOFs (not on
the box's free joint — that must keep falling). Measured effect in this
model: steady-state joint tracking error drops from ~0.023 rad to ~0 rad.
0.023 rad at the end of a 0.75 m arm is roughly 1.7 cm of end-effector
error, and the box is only 4.4 cm wide, so without this an open-loop IK
waypoint misses the grasp about as often as it makes it.

This is `test_gravity_compensation_tracks_targets`.

**Talking point:** gravity compensation is standard on real arms (it is what
makes a collaborative robot back-driveable by hand). Doing it in simulation
is not cheating, it is modelling the controller the hardware already has.

---

## 4. Reward design

### Potential-based shaping

The dense reward is built from *differences* of a potential, not raw
distances:

```
r_shaping = w * (phi(s') - phi(s)),   phi(s) = -distance(s)
```

Ng, Harada & Russell (1999) proved that shaping of exactly this form leaves
the optimal policy unchanged. A raw `-distance` term does not have that
guarantee: it silently adds a time cost, which biases the policy toward
ending episodes rather than solving them. Being able to say "this shaping is
potential-based, therefore policy-invariant" is a stronger claim than "I
tuned it until it worked".

### Staged, not summed

```
if grasped:  reward progress of box  -> goal
else:        reward progress of grip -> box
```

The tempting alternative — reward both at once — produces a well-known
degenerate policy: the arm hovers next to the box nudging it toward the goal
along the floor, because that collects both terms without ever paying the
cost of committing to a grasp. Gating on the grasp makes the two stages
mutually exclusive, so the only route to stage-2 reward is through the
bottleneck.

### Event bonuses

| Event | Value | Why |
|---|---|---|
| first grasp | +2 | Marks the bottleneck explicitly |
| first lift above 7.5 cm | +3 | Distinguishes "touching" from "carrying" |
| box within 5 cm of goal | +10, terminate | The actual objective |
| dropped after grasping | −1 | Otherwise grab-and-drop cycles are free |
| control cost | −0.005·‖a‖² | Discourages bang-bang chatter |

Each bonus fires **once per episode** (`_gave_grasp_bonus` etc.). A repeatable
bonus is the most common reward-hacking hole in manipulation tasks: a policy
will learn to open and close the gripper forever rather than move the box.

### The sparse variant

`reward_type="sparse"` returns 0 on success and −1 otherwise, which is the
convention SB3's HER expects. It exists so the repo can show a
shaped-vs-sparse+HER comparison rather than asserting that shaping was
necessary.

---

## 5. Grasp detection from contacts, not from finger angles

```python
for i in range(data.ncon):
    c = data.contact[i]
    ...  # is this contact box <-> pad_left / pad_right?
grasped = left and right
```

The lazy alternative is "fingers are closed and the box is nearby", which
reports a grasp when the box is simply sitting on the table between two
closed fingers. Asking the solver what is *actually* in contact is both
correct and a good demonstration of understanding MuJoCo's contact
structures (`data.ncon`, `data.contact[i].geom1/geom2`).

---

## 6. Curriculum

`difficulty ∈ [0, 1]` controls the initial-state and goal distributions:

| | difficulty 0 | difficulty 1 |
|---|---|---|
| box x | 0.37 – 0.43 | 0.30 – 0.50 |
| goal x | box x ± 0.02 | box x ± 0.18 |
| goal z | 0.14 – 0.18 | 0.10 – 0.34 |

So at 0 the task is "lift the box straight up a little", and at 1 it is
"pick it up from anywhere and put it anywhere".

`CurriculumCallback` in `training/train_pick_place.py` raises the difficulty
by 0.2 whenever the trailing-30-episode success rate clears 60%. This is the
simplest form of automatic curriculum learning: a success-rate trigger, no
separate teacher policy or learning-progress estimator. Worth naming the
alternatives when discussing it — Teacher-Student curricula, ALP-GMM,
reverse curriculum generation, and goal relabelling (HER) as the
curriculum-free alternative.

---

## 7. The scripted controller, and why it was written before any training

`scripts/scripted_pick_place.py` is a finite state machine over analytic-IK
waypoints:

```
APPROACH -> DESCEND -> CLOSE -> LIFT -> CARRY
```

It emits actions in the environment's own action space, so it is subject to
the same rate limits and actuator limits a learned policy would be. It solves
100% of episodes at every difficulty level.

This is the direct descendant of the debugging technique that resolved the
reach-task bug: *if a hand-written controller cannot solve the task through
the real actuators, the problem is physics, not RL.* Writing it before the
first training run means a flat learning curve is immediately attributable.

It found one real bug on its own. The first version fed back the *measured*
joint angles to compute its action, but the environment already integrates
actions into a servo target — so the controller was closing a loop around an
integrator and winding the target far past the waypoint, producing an
oscillation that never converged. Driving the *commanded* target instead and
judging arrival on the measured angles fixed it. That failure mode
(double integration through a residual action space) is worth knowing,
because a learned policy in the same action space can exhibit the same
oscillation if `delta_max` is set too large.

The controller doubles as the demonstration source for Phase 5:
`--save-demos` writes `(observation, action)` pairs for behavior cloning.

---

## 8. Analytic inverse kinematics for a 3-link planar arm

With three joints in a plane the arm is redundant: two DOF position the
gripper, the third is free, so we pin it by fixing the **approach angle**
`phi` (the absolute direction the gripper points). For a top-down grasp,
`phi = -pi/2`.

```
wrist = target - le * (cos phi, sin phi)          # back off along approach
c2    = (|wrist - base|^2 - l1^2 - l2^2) / (2 l1 l2)
psi2  = ±acos(c2)                                  # ± selects elbow up/down
psi1  = atan2(dz, dx) - atan2(l2 sin psi2, l1 + l2 cos psi2)
psi3  = phi - psi1 - psi2
```

Sign convention worth understanding: the joints rotate about **+y**, and a
rotation of +θ about +y maps +x toward −z. So the planar angle measured as
`atan2(z, x)` is `ψ = −θ`. The solver works in ψ and negates at the end. Sign
errors of exactly this kind are the most common bug in hand-derived
kinematics, which is why `test_analytic_ik_matches_forward_kinematics`
checks the solution against MuJoCo's own forward kinematics to 1e-6 rather
than trusting the derivation.

The link lengths are read out of the loaded model
(`model.body_pos`, `model.site_pos`) rather than hardcoded, so editing the
MJCF cannot silently desync the IK from the geometry.

Unreachability is returned as `None` rather than clamped, because a clamped
IK solution is a *wrong* answer that looks like a right one.

---

## 9. Where this sits relative to real manipulation research

Honest framing, useful in interviews:

- **Simplification: planar.** A real arm is 6-7 DOF in SE(3). Planar keeps
  the IK analytic and the state space small enough to debug. The observation
  already carries the box's full quaternion, so the extension to 3D is a
  model change plus a bigger network, not a redesign.
- **Simplification: state-based observations.** Real systems observe pixels
  or point clouds. The env exposes object pose directly. The obvious next
  step is a vision variant using MuJoCo's offscreen renderer with a frame
  stack or a small CNN encoder.
- **Simplification: parallel-jaw, top-down grasps only.** No grasp-pose
  selection, no dexterous manipulation.
- **Not simplified:** the contact dynamics are real, the grasp is real
  friction rather than a weld constraint, and the object can be dropped.
  A common shortcut in tutorial pick-and-place environments is to attach the
  object to the gripper with an equality constraint when the fingers are
  close enough. That is not done here, and it is worth saying so.
