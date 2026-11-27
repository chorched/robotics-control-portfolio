"""Residual PPO on top of an impedance controller, and the plain-PPO comparison.

    tau = tau_impedance(state) + alpha * TAU_MAX * a,     a = policy(obs) in [-1, 1]^7

With ``alpha = 0.1`` the learned part can add at most 10% of each joint's torque
limit, so a fresh, random policy still leaves the impedance controller in charge.
The "direct" arm uses the same PPO with alpha = 1 and no base controller.

Reward per 20 ms step (both arms)::

    r = max(-r_clip, -w1 (e / e0)^2 - w3 |qdot|^2 / (7 qdot0^2) - w2 |a|^2 / 7)

plus a terminal penalty of r_clip for every remaining step if the episode is cut
short (divergence, joint speed, MuJoCo warning). Training uses condition C1.

    python -m residual_rl.residual --method residual --seed 0 --steps 1000000 --out runs/res_s0
"""

from __future__ import annotations

import argparse
import json
import math
from functools import partial
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .controllers import Impedance
from .env import E0, EPISODE_STEPS, QDOT0, TAU_MAX, FetchTorqueEnv, condition_spec

# Frozen configuration of the reported runs (pilot_a, chosen before the main runs).
CONFIG = {
    "reward": {"w1": 1.0, "w2": 0.01, "w3": 0.001, "r_clip": 10.0},
    "alpha": 0.1,
    "impedance": {"K": 800.0, "D": 80.0},  # best of the impedance grid search
    "joint_pd": {"kp": 800.0, "kd": 30.0},  # both baseline optima sit on the edge of the searched grid
    "disturbance_scale": 2.0,
    "approach": "data_eq_active",  # how the mocap weld is released
    "log_std_init": {"residual": -1.0, "direct": 0.0},
    "ppo": {"n_envs": 8, "n_steps": 1024, "batch_size": 256, "n_epochs": 10, "learning_rate": 3e-4,
            "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.0, "vf_coef": 0.5,
            "max_grad_norm": 0.5, "net_arch": [256, 256]},
}


class ResidualEnv(gym.Wrapper):
    def __init__(self, env, base, alpha, reward):
        super().__init__(env)
        self.base, self.alpha, self.reward = base, alpha, reward
        self.action_space = spaces.Box(-1.0, 1.0, shape=(len(TAU_MAX),), dtype=np.float32)
        self._obs = None

    def reset(self, **kwargs):
        if self.base is not None:
            self.base.reset()
        self._obs, info = self.env.reset(**kwargs)
        return self._obs, info

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=float).reshape(-1), -1.0, 1.0)
        core = self.env.unwrapped
        tau = self.alpha * np.array(TAU_MAX) * a
        if self.base is not None:
            tau = tau + self.base(self._obs, core)
        self._obs, _, terminated, truncated, info = self.env.step(tau)
        rw, e = self.reward, info["e_norm"]
        if math.isfinite(e):
            qdot = core.data.qvel[core.dadr]
            r_env = -rw["w1"] * (e / E0) ** 2 - rw["w3"] * float(np.sum(qdot**2)) / (7.0 * QDOT0**2)
        else:
            r_env = 0.0
        reward = max(-rw["r_clip"], r_env - rw["w2"] * float(np.sum(a**2)) / 7.0)
        if terminated:
            reward -= rw["r_clip"] * info["remaining_steps"]
        return self._obs, reward, terminated, truncated, info


def build_env(method: str, condition: str = "C1", trajectory: str = "T1", cfg=CONFIG):
    residual = method == "residual"
    base = Impedance(cfg["impedance"]["K"], cfg["impedance"]["D"]) if residual else None
    env = FetchTorqueEnv(cfg["approach"], max_steps=EPISODE_STEPS, q_noise=0.0, trajectory=trajectory,
                         condition=condition_spec(condition, cfg["disturbance_scale"]))
    return ResidualEnv(env, base, cfg["alpha"] if residual else 1.0, cfg["reward"])


def train(method: str, seed: int, steps: int, out: Path, cfg=CONFIG) -> dict:
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    p = cfg["ppo"]
    out.mkdir(parents=True, exist_ok=True)
    venv = make_vec_env(partial(build_env, method), n_envs=p["n_envs"], vec_env_cls=SubprocVecEnv, seed=seed)
    vn = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=p["gamma"])
    lr0 = p["learning_rate"]
    model = PPO("MlpPolicy", vn, n_steps=p["n_steps"], batch_size=p["batch_size"], n_epochs=p["n_epochs"],
                learning_rate=lambda progress: lr0 * progress, gamma=p["gamma"], gae_lambda=p["gae_lambda"],
                clip_range=p["clip_range"], ent_coef=p["ent_coef"], vf_coef=p["vf_coef"],
                max_grad_norm=p["max_grad_norm"], seed=seed, device="cpu", verbose=0,
                policy_kwargs={"net_arch": {"pi": p["net_arch"], "vf": p["net_arch"]},
                               "activation_fn": torch.nn.Tanh, "log_std_init": cfg["log_std_init"][method]})
    try:
        model.learn(total_timesteps=steps)
        model.save(str(out / "model.zip"))
        vn.save(str(out / "vecnormalize.pkl"))
    finally:
        venv.close()
    return {"method": method, "seed": seed, "steps": int(model.num_timesteps)}


def evaluate_episode(policy, method: str, condition: str, trajectory: str, seed: int, cfg=CONFIG) -> dict:
    """``policy(obs) -> action`` in [-1, 1]^7; ``None`` runs the base controller alone."""
    env = build_env(method, condition, trajectory, cfg)
    obs, _ = env.reset(seed=seed)
    errs, done, term, k = [], False, False, 0
    while not done:
        act = np.zeros(len(TAU_MAX)) if policy is None else policy(obs)
        obs, _, term, trunc, info = env.step(act)
        k += 1
        if math.isfinite(info["e_norm"]):
            errs.append((k, info["e_norm"]))
        done = term or trunc
    e = np.array([v for _, v in errs])
    ss = np.array([v for kk, v in errs if kk >= 50])  # steady state: t >= 1 s
    return {"e_rms": float(np.sqrt(np.mean(e**2))), "e_rms_ss": float(np.sqrt(np.mean((ss if len(ss) else e) ** 2))),
            "e_max": float(e.max()), "terminated": bool(term), "steps": k}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["residual", "direct"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(train(args.method, args.seed, args.steps, args.out)))


if __name__ == "__main__":
    main()
