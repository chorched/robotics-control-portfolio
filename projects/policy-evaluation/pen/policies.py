"""Action-chunking policies for AdroitHandPen-v1, trained from D4RL ``pen-human-v1`` demos.

Two policy families share one network: an MLP that maps (observation, action chunk,
time features) to an 8 x 24 action chunk.

* ``bc``  - behaviour cloning, masked MSE on the demonstrated chunk (time input = 0).
* ``fm``  - conditional flow matching: learn the velocity field that carries Gaussian
  noise x0 to the demonstrated chunk along straight paths x_t = (1 - t) x0 + t a.
  Sampling integrates the field with 8 Euler steps (deterministic at temperature 0).

Chunks near the end of an episode are zero-padded and masked out of the loss. The
policy is executed by re-planning every 8 steps ("success" = the task's own success
flag is raised at any of 100 steps).

This is the model used to produce the 78-candidate checkpoint pool (3 seeds x
{1, 3, 12, 20} demonstrations x checkpoints at 0 ... 10k steps) whose rollout
outcomes are in ``data/``. Requires PyTorch, gymnasium-robotics and the D4RL file.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

OBS_DIM, ACT_DIM, CHUNK, HIDDEN = 45, 24, 8, 256
IN_DIM = OBS_DIM + CHUNK * ACT_DIM + 3
OUT_DIM = CHUNK * ACT_DIM
ODE_STEPS = 8
HORIZON = 100
EPISODE_STEPS = 200


def time_features(t: torch.Tensor) -> torch.Tensor:
    return torch.cat([t, torch.sin(2 * math.pi * t), torch.cos(2 * math.pi * t)], dim=1)


class ChunkMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(IN_DIM, HIDDEN), nn.SiLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.SiLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.SiLU(),
            nn.Linear(HIDDEN, OUT_DIM),
        )

    def forward(self, obs, chunk, t):
        return self.body(torch.cat([obs, chunk.reshape(len(obs), OUT_DIM), time_features(t)], dim=1)).reshape(
            -1, CHUNK, ACT_DIM)


def chunk_dataset(observations: np.ndarray, actions: np.ndarray, episode_starts) -> tuple:
    """One anchor per demo step: (obs, 8-step action chunk zero-padded, validity mask)."""
    obs, chunks, masks = [], [], []
    for s in episode_starts:
        ep = actions[s : s + EPISODE_STEPS]
        for k in range(EPISODE_STEPS):
            valid = min(CHUNK, EPISODE_STEPS - k)
            c = np.zeros((CHUNK, ACT_DIM), np.float32)
            c[:valid] = ep[k : k + valid]
            obs.append(observations[s + k])
            chunks.append(c)
            masks.append(np.r_[np.ones(valid), np.zeros(CHUNK - valid)].astype(np.float32))
    return np.stack(obs), np.stack(chunks), np.stack(masks)


def masked_mse(err: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (mask.unsqueeze(-1) * err.pow(2)).sum() / (ACT_DIM * mask.sum().clamp_min(1e-9))


def loss(model: ChunkMLP, family: str, obs, chunk, mask, gen: torch.Generator) -> torch.Tensor:
    b = len(obs)
    if family == "bc":
        pred = model(obs, torch.zeros_like(chunk), torch.zeros(b, 1))
        return masked_mse(pred - chunk, mask)
    x0 = torch.randn(b, CHUNK, ACT_DIM, generator=gen)
    t = torch.rand(b, 1, generator=gen)
    x_t = (1 - t).unsqueeze(-1) * x0 + t.unsqueeze(-1) * chunk
    return masked_mse(model(obs, x_t, t) - (chunk - x0), mask)  # target velocity a - x0


def train(family: str, data: tuple, seed: int, steps: int, checkpoints=(0, 200, 500, 1000, 2500, 5000, 10000),
          batch: int = 256, lr: float = 1e-3) -> dict:
    """Returns {step: state_dict} at the requested checkpoint steps."""
    torch.manual_seed(seed)
    model = ChunkMLP()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x, y, m = (torch.from_numpy(a) for a in data)
    gen = torch.Generator().manual_seed(seed * 10 + 2)
    pick = torch.Generator().manual_seed(seed * 10 + 1)
    saved = {}
    for step in range(steps + 1):
        if step in checkpoints:
            saved[step] = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if step == steps:
            break
        idx = torch.randint(0, len(x), (batch,), generator=pick)
        opt.zero_grad(set_to_none=True)
        loss(model, family, x[idx], y[idx], m[idx], gen).backward()
        opt.step()
    return saved


@torch.no_grad()
def act(model: ChunkMLP, family: str, obs: np.ndarray, temperature: float = 0.0,
        gen: torch.Generator | None = None) -> np.ndarray:
    o = torch.from_numpy(obs.astype(np.float32))[None]
    if family == "bc":
        return model(o, torch.zeros(1, CHUNK, ACT_DIM), torch.zeros(1, 1))[0].numpy()
    x = torch.randn(1, CHUNK, ACT_DIM, generator=gen) * temperature
    for i in range(ODE_STEPS):  # Euler integration of the learned flow
        x = x + model(o, x, torch.full((1, 1), i / ODE_STEPS)) / ODE_STEPS
    return x[0].numpy()


def rollout(model: ChunkMLP, family: str, env, reset_seed: int, temperature: float = 0.0) -> bool:
    """One episode; True if the task's success flag is raised at any step."""
    obs, _ = env.reset(seed=int(reset_seed))
    gen = torch.Generator().manual_seed(900000 + int(reset_seed))
    steps = 0
    while steps < HORIZON:
        block = np.clip(act(model, family, np.asarray(obs), temperature, gen), -1.0, 1.0)
        for a in block[: HORIZON - steps]:
            obs, _r, terminated, truncated, info = env.step(a)
            steps += 1
            if info.get("success", False):
                return True
            if terminated or truncated:
                return False
    return False


def make_env():
    import gymnasium as gym
    import gymnasium_robotics

    gym.register_envs(gymnasium_robotics)
    return gym.make("AdroitHandPen-v1", max_episode_steps=HORIZON)
