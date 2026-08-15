"""
Behaviour cloning baseline for pick-and-place.

Trains a policy by supervised regression on (observation, action) pairs from
the scripted controller (`scripts/scripted_pick_place.py --save-demos`), then
evaluates it by *rolling it out* in the real environment -- not by held-out
regression loss, which does not tell you whether the resulting closed-loop
policy can actually complete the task.

Why this exists (see docs / README for the full framing): it is the cheapest
possible baseline for Phase 5, deliberately run before the sparse+HER
experiment. The predicted result is that BC does well close to the training
distribution and degrades as difficulty rises, because the scripted
controller is deterministic and the demos only cover a thin manifold of
states -- once a BC policy drifts even slightly off that manifold there is no
data telling it how to recover. Compounding error under distribution shift,
not a network-capacity problem.

Network architecture: SB3's default PPO policy is a [64, 64] MLP with Tanh
activations. I don't have this repo's train_pick_place.py to confirm whether
policy_kwargs overrides that default, so this mirrors the SB3 default and
should be reconciled against the real PPO network before the BC-vs-RL
numbers are presented side by side as "matched architecture."

Usage
-----
    python training/train_bc.py --demos data/demos.npz --epochs 100
    python training/train_bc.py --demos data/demos.npz --sweep-only --checkpoint out/bc_policy.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environments.pick_place_env import PickPlaceEnv, LIFT_Z  # noqa: E402


class DemoDataset(Dataset):
    """Wraps the (observation, action) pairs saved by --save-demos."""

    def __init__(self, path: str):
        data = np.load(path)
        self.obs = torch.from_numpy(data["observations"]).float()
        self.act = torch.from_numpy(data["actions"]).float()
        assert self.obs.shape[0] == self.act.shape[0]

    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self, i):
        return self.obs[i], self.act[i]


class BCPolicy(nn.Module):
    """[64, 64] MLP + Tanh head, matching SB3 PPO's default MlpPolicy shape.

    Tanh on the output is not cosmetic: the env's action_space is
    Box(-1, 1, shape=(4,)), so an unbounded linear head could output actions
    the env would silently clip, corrupting the gradient signal near the
    bounds. Squashing analytically avoids that.
    """

    def __init__(self, obs_dim: int = 33, act_dim: int = 4, hidden=(64, 64)):
        super().__init__()
        layers = []
        d = obs_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.Tanh()]
            d = h
        layers += [nn.Linear(d, act_dim), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        return self.net(obs)

    def act(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            a = self.forward(torch.from_numpy(obs).float().unsqueeze(0))
        return a.squeeze(0).numpy()


def train(args):
    full = DemoDataset(args.demos)
    n_val = max(1, int(0.1 * len(full)))
    train_set, val_set = random_split(
        full, [len(full) - n_val, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=512)

    policy = BCPolicy()
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    best_val, best_state, patience = float("inf"), None, 0
    for epoch in range(args.epochs):
        policy.train()
        for obs, act in train_loader:
            opt.zero_grad()
            loss = loss_fn(policy(obs), act)
            loss.backward()
            opt.step()

        policy.eval()
        with torch.no_grad():
            val_loss = np.mean([
                loss_fn(policy(obs), act).item() for obs, act in val_loader
            ])
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:3d}  val_mse {val_loss:.5f}")

        if val_loss < best_val - 1e-5:
            best_val, best_state, patience = val_loss, {
                k: v.clone() for k, v in policy.state_dict().items()
            }, 0
        else:
            patience += 1
            if patience >= args.patience:
                print(f"  early stop at epoch {epoch} (best val_mse {best_val:.5f})")
                break

    policy.load_state_dict(best_state)
    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
    torch.save(policy.state_dict(), args.checkpoint)
    print(f"saved best checkpoint (val_mse {best_val:.5f}) -> {args.checkpoint}")
    return policy


def evaluate_sweep(policy: BCPolicy, difficulties, episodes: int, seed: int):
    """Grasp / lift / success rates and mean return, reported separately per
    difficulty -- same philosophy as evaluate_pick_place.py: the gaps between
    the three numbers localize *where* a policy fails, not just whether it
    does. This is a standalone stand-in; if evaluate_pick_place.py exposes a
    reusable eval_policy(env, act_fn) function, swap this out for it so BC
    and PPO numbers come from literally the same code path.
    """
    results = {}
    for d in difficulties:
        env = PickPlaceEnv(difficulty=d)
        grasped = lifted = succeeded = 0
        returns = []
        for ep in range(episodes):
            obs, info = env.reset(seed=seed + ep)
            ep_return, ep_grasped, ep_lifted, ep_success = 0.0, False, False, False
            for _ in range(env.max_episode_steps):
                obs, r, term, trunc, info = env.step(policy.act(obs))
                ep_return += r
                ep_grasped = ep_grasped or info["is_grasped"]
                ep_lifted = ep_lifted or info["peak_carry_height"] > LIFT_Z
                ep_success = ep_success or info["is_success"]
                if term or trunc:
                    break
            grasped += int(ep_grasped)
            lifted += int(ep_lifted)
            succeeded += int(ep_success)
            returns.append(ep_return)
        env.close()
        results[d] = dict(
            grasp_rate=grasped / episodes,
            lift_rate=lifted / episodes,
            success_rate=succeeded / episodes,
            mean_return=float(np.mean(returns)),
        )
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--demos", type=str, default="data/demos.npz")
    p.add_argument("--checkpoint", type=str, default="out/bc_policy.pt")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-episodes", type=int, default=30)
    p.add_argument("--sweep-only", action="store_true",
                    help="skip training, just evaluate an existing checkpoint")
    args = p.parse_args()

    torch.manual_seed(args.seed)

    if args.sweep_only:
        policy = BCPolicy()
        policy.load_state_dict(torch.load(args.checkpoint))
        policy.eval()
    else:
        policy = train(args)
        policy.eval()

    print("\nsweep evaluation (rollouts, not regression loss):")
    results = evaluate_sweep(policy, [0.0, 0.5, 1.0], args.eval_episodes, args.seed)
    for d, r in results.items():
        print(
            f"  difficulty {d:.1f}: grasp {r['grasp_rate']:.0%}  "
            f"lift {r['lift_rate']:.0%}  success {r['success_rate']:.0%}  "
            f"mean_return {r['mean_return']:.2f}"
        )


if __name__ == "__main__":
    main()
