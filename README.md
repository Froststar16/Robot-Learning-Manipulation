# robot-learning-manipulation

A little robot arm learns to point at things. Then it learns to pick them up.
Both in a physics engine, both from scratch, and both taking considerably
longer than I expected — mostly for reasons that had nothing to do with the
learning algorithm.

This is a portfolio project that turned into a running lesson in
**"don't trust the algorithm before you trust the environment."** Two separate
times, a flat training curve turned out to be a simulation bug rather than an
RL bug. Both stories are below, because they're the most useful thing in here.

---

## The 30-second version, for people who don't do robotics

A robot arm is just a few motors bolted together. Telling it "put the red block
over there" sounds simple, but the arm has no idea what a block is, where its
own hand is, or which way to turn a motor. Somebody has to teach it.

The old way: a human works out the maths for every motion, by hand, for every
task. Works beautifully until the block moves two centimetres.

The new way — **robot learning** — is to let the arm figure it out by trying.
It flails around, occasionally does something slightly less useless than usual,
gets told "warmer," and slowly stops flailing. Do that a few hundred thousand
times in a simulator (where crashing the robot costs nothing) and you get a
policy: a small neural network that looks at where things are and decides what
the motors should do next.

This repo is that whole loop, built by hand:

1. **Build a robot** — write out the arm's geometry, mass and joints as a
   physics model.
2. **Build a task** — spawn a target somewhere random, define what "success"
   and "warmer" mean.
3. **Let it learn** — an RL algorithm does the flailing and the improving.
4. **Check it isn't lying to you** — tests, hand-derived maths, scripted
   controllers that prove the task is possible in the first place.

Step 4 is the one people skip, and it's where both of my bugs lived.

---

## What's actually in here

### Task 1: Reach — *touch the target*

A 2-link arm has to move its tip to a randomly placed dot. The training
equivalent of a scale exercise: small enough that when something breaks, you
can actually find out why.

PPO and SAC (via Stable-Baselines3) take it from 0% to ~25–40% success in 200k
timesteps on a CPU. Not state-of-the-art, and it doesn't need to be — the point
is a clean, correct, tested pipeline.

### Task 2: Pick-and-place — *actually grab the thing*

A 3-DOF arm with a two-fingered gripper has to pick a box up off the table and
carry it to a target position. This is a genuinely different problem, not just
"reach with extra steps":

|                   | Reach                      | Pick-and-place                                    |
| ----------------- | -------------------------- | ------------------------------------------------- |
| Physics           | Smooth, nothing touches    | Discontinuous — contact forces every timestep      |
| Reward landscape  | One smooth hill to climb   | Two stages with a wall between them                |
| The catch         | —                          | You must **grasp** before anything else counts     |
| Ways to fail      | Be slow                    | Drop it, knock it away, squeeze it out sideways    |

That middle row is the whole problem. The grasp is a **bottleneck**: virtually
no path to success avoids it, and an arm flailing randomly almost never finds
it by accident. Most of the design in `environments/pick_place_env.py` exists
to deal with that one fact — a two-stage reward, a curriculum that starts easy,
and an off-policy algorithm that can replay each rare successful grasp many
times instead of seeing it once and forgetting.

### And the supporting cast

- **Domain randomization** — every episode, the arm's damping, mass, gearing
  and inertia get jittered. A policy that only works in one exact simulated
  world isn't much of a robot learning story.
- **20 passing tests**, including hand-derived inverse kinematics checked
  against MuJoCo's own forward kinematics, and a scripted controller that
  solves pick-and-place 100% of the time. That last one is load-bearing: it
  proves the task is *possible* before I'm allowed to blame the RL for
  anything.
- **Two debugging writeups** (`docs/topics_covered.md`,
  `docs/pick_place_notes.md`). Read these ones.

Everything is built directly on the **MuJoCo Python bindings** — no
`dm_control`, no `robosuite`. More work up front, but nothing in the
physics-to-RL pipeline is a black box I inherited from someone else's defaults.

---

## Results

**A trained policy reaching the target:**

![PPO reaching a target](results/demo_rollout.gif)

*PPO, 200k timesteps, ~25–40% success at convergence on this budget. The blue
end-effector tracks over to the red target and the episode terminates on
contact.*

**Training curve:**

![Evaluation success rate over training](results/eval_success_chart.png)

Success rate climbs from 0% as PPO learns the task, with the usual on-policy
noise between eval checkpoints rather than a smooth line — expected at this
budget on a CPU-only run.

**Pick-and-place:**

![Scripted pick and place](results/scripted_pick.gif)

*<!-- TODO: replace with a trained-policy GIF once SAC has run, and add the
success-rate curve. Until then this is the scripted IK controller, which is
honest as long as it's labelled. -->*

---

## The two bugs worth reading about

### Bug 1: the arm was glued to the floor

Trained PPO for 60k steps. Success rate: 0%, flat as a table.

Before touching a single hyperparameter, I checked whether the task was even
solvable — derived the 2-link inverse kinematics by hand, confirmed every
sampled target was reachable, then drove the joints there with a basic
proportional controller through the actual actuators.

**That failed too.** Which was excellent news, because it meant the bug was in
the physics, not the learning. The arm links were sitting at exactly ground
level, so MuJoCo's contact solver was generating friction that fought nearly
every bit of motion. The same torque that produced ~0.03 rad/s of joint
velocity before the fix produced a physically sensible ~10 rad/s after turning
off arm–ground collisions. One `contype="0" conaffinity="0"` later, PPO started
learning normally.

**Moral:** if the learning curve is flat, check whether the environment is
solvable before you touch the algorithm. A scripted controller is a lot cheaper
than a hyperparameter sweep.

### Bug 2: the fix from Bug 1 nearly broke Task 2

The obvious move for pick-and-place was to copy the fix that worked: turn off
contact. Except **grasping *is* contact** — turn it off and the gripper's
fingers pass through the box like a ghost.

So instead of a global switch, the model uses MuJoCo's contact bitmasks to
allow exactly two collision pairs — box↔floor and box↔fingertips — and nothing
else. The arm can still sweep through the table without the old friction
problem, but the gripper can still pick things up. There's a test that runs 120
random-action steps and asserts no contact ever involves an arm link, so Bug 1
can't quietly come back.

**Moral:** a fix that works is not the same as a fix you understand. I got away
with the blunt version once; the second task made me learn what the setting
actually did.

### Bug 3 (bonus): my own controller fought itself

The scripted controller for pick-and-place oscillated forever and never reached
its waypoint. Cause: it was feeding back the arm's *measured* joint angles, but
the environment already integrates actions into a position target — so the
controller was closing a loop around an integrator and winding the target far
past where it wanted to go. Classic integrator windup, self-inflicted.

Worth knowing because a *learned* policy in the same action space can do
exactly the same thing if the per-step action limit is set too large.

---

## Quickstart

```bash
git clone https://github.com/Froststar16/Robot-Learning-Manipulation.git
cd Robot-Learning-Manipulation
python -m venv venv && source venv/bin/activate    # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt

pytest tests/ -v                                   # 20 tests, ~2 seconds
```

**Reach:**

```bash
python training/train_ppo.py --timesteps 100000 --run-name ppo_reach_v1
python evaluation/evaluate.py --model-path logs/ppo_reach_v1/best_model.zip --episodes 50
```

**Pick-and-place:**

```bash
# Sanity check first: can a hand-written controller solve it? (should print 100%)
python scripts/scripted_pick_place.py --episodes 20 --difficulty 1.0

# Train it. SAC by default -- expect 45-90 min for 150k steps on CPU.
python training/train_pick_place.py --timesteps 150000 --run-name sac_pp_v1
tensorboard --logdir logs/
```

Watch `curriculum/difficulty` in tensorboard. If it never leaves 0.0, the policy
isn't grasping yet — check `rollout/ep_rew_mean` is above ~2 first, which means
it's at least learning to reach the box.

**Making GIFs:**

```bash
python evaluation/record_gif.py --model-path logs/ppo_reach_v1/best_model.zip --out results/demo_rollout.gif
python scripts/scripted_pick_place.py --episodes 1 --gif results/scripted_pick.gif --camera angled
```

Needs offscreen rendering — fine on most desktops. On headless Linux:

```bash
sudo apt-get install libosmesa6 libgl1-mesa-dev libglfw3
MUJOCO_GL=osmesa python evaluation/record_gif.py --out results/demo_rollout.gif
```

---

## A few design decisions, and why

**Position control with gravity compensation, not raw torques.** Under torque
control the policy has to learn to hold the arm up against gravity before it
can learn anything about the task, which eats most of a small sample budget.
Real manipulators expose a joint position interface with a servo underneath for
exactly this reason. Gravity compensation drops the steady-state joint error
from ~0.023 rad to roughly zero — which matters, because 0.023 rad at the end
of a 0.75 m arm is 1.7 cm of error, and the box is only 4.4 cm wide.

**Potential-based reward shaping.** The dense reward comes from *differences*
of negative distance, not raw negative distance. That specific form is provably
policy-invariant (Ng et al., 1999). "This shaping doesn't change the optimal
policy" is a much stronger thing to be able to say than "I tuned it until it
worked."

**A staged reward, not a summed one.** Reward approach-the-box and
move-box-to-goal at the same time and you get a policy that hovers next to the
box nudging it along the floor — collecting both terms without ever committing
to a grasp. Gating stage 2 behind an actual grasp closes that loophole.

**Grasp detection read from the contact solver**, not inferred from finger
positions. "Fingers closed and box nearby" reports a successful grasp when the
box is simply sitting on the table between two closed fingers.

**A tiny task before a hard one.** Reach exists so that correctness is
*checkable* — analytic IK, scripted controllers — before adding contact
dynamics, where bugs get much harder to find. Bug 2 is what that buys you.

---

## Repo structure

```
environments/
  reach_env.py              # Task 1 -- 2-link arm, no contact
  pick_place_env.py         # Task 2 -- 3-DOF arm + gripper, contact-rich
  assets/reacher_arm.xml
  assets/pick_place_arm.xml
training/
  train_ppo.py
  train_sac.py
  train_pick_place.py       # SAC/PPO + automatic curriculum callback
scripts/
  scripted_pick_place.py    # IK waypoint controller; also generates BC demos
evaluation/
  evaluate.py
  record_gif.py
domain_randomization/
  randomize.py
tests/
  test_env.py               # 7 tests -- reach
  test_pick_place_env.py    # 13 tests -- pick-and-place
docs/
  topics_covered.md         # concept map + Bug 1
  pick_place_notes.md       # the design rationale + Bugs 2 and 3
results/
```

---

## Roadmap

| # | Phase | Status |
|---|---|---|
| 1 | Foundations & environment setup | done |
| 2 | RL baseline: reach task | done |
| 3 | Manipulation: pick-and-place | done |
| 4 | Robustness: domain randomization | done (reach) |
| 5 | Imitation learning comparison | next |
| 6 | Evaluation, visualization & docs | done (reach), in progress (pick-place) |

**Phase 5** is the interesting one and it's now unblocked, because the scripted
controller can generate demonstrations (`--save-demos`). The plan is behaviour
cloning vs. SAC vs. BC-pretrained-then-SAC on identical budgets, plus a
sparse-reward + Hindsight Experience Replay variant (already wired up via
`reward_type="sparse"`) to test whether hindsight relabelling can recover what
hand-designed reward staging buys you.

---

## Where this sits relative to real manipulation research

Being straight about the simplifications, since they're all deliberate:

- **Planar, not 3D.** Keeps the IK analytic and the state space small enough to
  debug. The observation already carries the box's full quaternion, so going 3D
  is a model change and a bigger network, not a redesign.
- **State observations, not pixels.** The env hands the policy the box's pose
  directly. Real systems get camera images. MuJoCo's offscreen renderer is
  already wired up, so this is the natural next extension.
- **Top-down parallel-jaw grasps only.** No grasp pose selection, no dexterity.

And what is *not* simplified: the contact dynamics are real, the grasp is real
friction rather than a weld constraint, and the box can genuinely be dropped. A
common shortcut in tutorial pick-and-place environments is to glue the object
to the gripper with an equality constraint once the fingers get close enough.
That isn't done here.

---

## License

MIT — use it, fork it, break it, fix it.
