# robot-learning-manipulation

A little robot arm, learning to point at things, in a physics engine, from scratch.

This started as a portfolio project and turned into a decent lesson in
"don't trust the algorithm before you trust the environment." Details
below, including the part where the arm got stuck to the floor for a
while (not on purpose).

## What's here

- **A custom Gymnasium environment**, built directly on the MuJoCo Python
  bindings — no `dm_control`, no `robosuite`. A 2-link arm learns to reach
  a randomly placed target. Small, but every line of the physics → RL
  pipeline is mine and I can explain all of it.
- **RL baselines that actually train**: PPO and SAC via Stable-Baselines3,
  going from 0% to ~30-40% success in 80k timesteps on a CPU. Not
  state-of-the-art numbers, they don't need to be — the point is a clean,
  correct, well-tested pipeline.
- **Domain randomization**, because a policy that only works in one exact
  simulated world isn't much of a robot learning story.
- **7 passing tests**, including one that solves the task with hand-derived
  inverse kinematics just to make sure the task is *possible* before
  blaming the RL algorithm for anything.
- **A debugging writeup** (`docs/topics_covered.md`) about the one time the
  arm physically could not move because it was glued to the ground by
  contact friction. This is the most interesting file in the repo.

## What's *not* here (yet)

`environments/pick_place_env.py` is a scoped-but-unbuilt stub for a
contact-rich pick-and-place task — grasping, a free-floating object, the
whole harder problem. I'd rather hand you a working reach task with an
honest plan than a manipulation demo held together with hope. The
implementation plan is written out in that file if you want to pick it up.

## Results

**A trained policy actually reaching the target:**

![PPO reaching a target](results/demo_rollout.gif)

*PPO, 200k timesteps, ~25-40% success rate at convergence on this small a
budget. The gif above is a successful episode — the blue end-effector
tracks over to the red target and the episode terminates on contact.*

**Training curve:**

![Evaluation success rate over training](results/eval_success_chart.png)

Success rate climbs from 0% early in training up as PPO learns the reach
task, with the usual on-policy noise between eval checkpoints rather than
a perfectly smooth curve — expected at this training budget on a CPU-only
run. Longer training, a tighter success threshold curriculum, or SAC
(more sample-efficient off-policy) would push this further; see
`docs/topics_covered.md` for the reward-shaping notes behind these numbers.

## Record your own demo GIF

```bash
# Uses a trained model, retries a few seeds to find a successful episode
python evaluation/record_gif.py --model-path logs/ppo_reach_v1/best_model.zip --out results/demo_rollout.gif

# No model yet? Records a scripted (hand-derived IK) controller instead
python evaluation/record_gif.py --out results/demo_scripted.gif
```

Needs offscreen rendering support. Works out of the box on most desktop
setups (Windows/macOS with a GPU, or Linux with a display). On headless
Linux without a display, install OSMesa and set `MUJOCO_GL=osmesa` first:
```bash
# Ubuntu/Debian
sudo apt-get install libosmesa6 libgl1-mesa-dev libglfw3
MUJOCO_GL=osmesa python evaluation/record_gif.py --out results/demo_rollout.gif
```



```bash
git clone https://github.com/<your-username>/robot-learning-manipulation.git
cd robot-learning-manipulation
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

pytest tests/ -v                      # sanity check, should take < 1 second

python training/train_ppo.py --timesteps 100000 --run-name ppo_reach_v1
python evaluation/evaluate.py --model-path logs/ppo_reach_v1/best_model.zip --episodes 50
```

Rendering needs a display or GPU with EGL/OSMesa — training and eval both
work fine headless without it.

## The bug worth reading about

Trained PPO for 60k steps. Success rate: 0%, flat as a table. Before
tuning a single hyperparameter, I checked whether the task was even
solvable — derived the 2-link inverse kinematics by hand, confirmed every
sampled target was in reach, then drove the joints there with a basic
proportional controller through the actual actuators.

That failed too. Which was actually good news — it meant the bug was in
the physics, not the learning algorithm. Turned out the arm links were
sitting at exactly the same height as the ground plane, so MuJoCo's
contact solver was generating friction that fought almost every bit of
motion. Same torque command that produced ~0.03 rad/s of joint velocity
before the fix produced the physically correct ~10 rad/s after turning off
arm-ground collisions. One `contype="0" conaffinity="0"` later, PPO
started learning normally.

Moral, and the reason it's in the README and not just buried in a commit
message: **if the learning curve is flat, check whether the environment is
solvable before you touch the algorithm.** A scripted controller is a lot
cheaper than a hyperparameter sweep.

## Repo structure

```
environments/
  reach_env.py           # the real thing -- implemented and tested
  pick_place_env.py      # scoped stub, not built yet, plan included
  assets/reacher_arm.xml # MJCF model
training/
  train_ppo.py
  train_sac.py
evaluation/
  evaluate.py
  record_gif.py          # rollout -> GIF, for demo purposes
domain_randomization/
  randomize.py
tests/
  test_env.py
docs/
  topics_covered.md      # concept map + the debugging story
results/
  demo_rollout.gif        # trained policy successfully reaching a target
  eval_success_chart.png  # tensorboard success-rate curve over training
```

## Why it's built this way

- **Raw MuJoCo over a framework**: more work up front, but nothing in the
  physics-to-RL pipeline is a black box I inherited from someone else's
  defaults.
- **A tiny 2-DOF task before a hard one**: makes correctness actually
  checkable (analytic IK, scripted controllers) before adding the contact
  dynamics where bugs get much harder to find.

## License

MIT — use it, fork it, break it, fix it.
