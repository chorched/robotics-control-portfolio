"""Run SmolVLA on LIBERO-Object with the executors in ``realtime_chunking``.

Runs inside the container in ``docker/`` (LeRobot v0.4.4, LIBERO 8f1084e,
MuJoCo, torch 2.8 + cu128, SmolVLA weights pinned to a Hub revision):

    docker build -t smolvla-libero docker/
    docker run --gpus all -v $PWD:/work smolvla-libero \
        python examples/libero_eval.py --mode async --drop elapsed --tasks 0,1,2 --init-states 40-47

Timing protocols
  paused  inference stops the simulator (the usual offline benchmark protocol)
  sync    blocking query; the simulator keeps running, the arm holds its target
  async   background query, chunk queue, expired-prefix drop (nominal | elapsed)

Latency L is measured from "observation in hand" to "action chunk back"
(pre-processing + forward + un-normalisation). ``--inject-delay-ms`` adds a
sleep inside the query, so it is part of L. One JSON line per episode is
printed; aggregate with ``realtime_chunking.stats``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from realtime_chunking import (  # noqa: E402
    AsyncChunkExecutor,
    SyncChunkExecutor,
    ThreadedRequester,
    zero_hold,
)

_torch_load = torch.load
torch.load = lambda *a, **k: _torch_load(*a, **{"weights_only": False, **k})  # LIBERO init-state pickles

MODEL_DIR = "/opt/models/smolvla_libero"
SUITE = "libero_object"
DUMMY = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
MAX_STEPS = 280


# ------------------------------------------------------------------- policy side


def load_policy(num_steps: int | None, device: str = "cuda"):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(MODEL_DIR)
    cfg.pretrained_path = MODEL_DIR
    cfg.device = device
    if num_steps:
        cfg.num_steps = num_steps  # flow-matching denoising steps used by sample_actions
    policy = get_policy_class(cfg.type).from_pretrained(MODEL_DIR, config=cfg, strict=False)
    policy.config.num_steps = cfg.num_steps
    policy.eval().to(device)
    pre, post = make_pre_post_processors(cfg, pretrained_path=MODEL_DIR)
    return policy, pre, post


def quat2axisangle(q: np.ndarray) -> np.ndarray:
    w = float(np.clip(q[3], -1.0, 1.0))
    den = math.sqrt(max(1.0 - w * w, 0.0))
    if den < 1e-8:
        return np.zeros(3, dtype=np.float32)
    return ((q[:3] / den) * (2.0 * math.acos(w))).astype(np.float32)


def format_obs(raw: dict, lang: str) -> dict:
    """LIBERO observation -> single-sample LeRobot observation."""
    out: dict = {}
    for key, cam in (("observation.images.image", "agentview_image"),
                     ("observation.images.image2", "robot0_eye_in_hand_image")):
        t = torch.from_numpy(np.ascontiguousarray(raw[cam])).permute(2, 0, 1).float().div_(255.0)
        out[key] = torch.flip(t, dims=[1, 2]).unsqueeze(0)
    state = np.concatenate([
        np.asarray(raw["robot0_eef_pos"], np.float32),
        quat2axisangle(np.asarray(raw["robot0_eef_quat"], np.float32)),
        np.asarray(raw["robot0_gripper_qpos"], np.float32),
    ])
    out["observation.state"] = torch.from_numpy(state).unsqueeze(0)
    out["task"] = lang
    return out


def make_chunk_fn(policy, pre, post, lang: str, inject_ms: float):
    def chunk_fn(obs) -> np.ndarray:
        raw, _t = obs
        batch = pre(format_obs(raw, lang))
        with torch.inference_mode():
            chunk = policy.predict_action_chunk(batch)  # (1, T, A)
        torch.cuda.synchronize()
        acts = np.stack([post(chunk[:, i, :]).detach().cpu().float().numpy().reshape(-1)[:7]
                         for i in range(chunk.shape[1])])
        if inject_ms > 0:
            time.sleep(inject_ms / 1000.0)
        return np.clip(acts.astype(np.float32), -1.0, 1.0)

    return chunk_fn


# ---------------------------------------------------------------------- env side


def build_suite():
    cfg = Path.home() / ".libero" / "config.yaml"
    if not cfg.exists():  # LIBERO blocks on stdin without it
        b = "/opt/LIBERO/libero/libero"
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text(f"benchmark_root: {b}\nbddl_files: {b}/bddl_files\ninit_states: {b}/init_files\n"
                       f"datasets: /opt/LIBERO/datasets\nassets: {b}/assets\n")
    from libero.libero import benchmark
    return benchmark.get_benchmark_dict()[SUITE]()


def build_env(suite, task_id: int):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task = suite.get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(0)
    return env, task.language


def run_episode(env, init_state, executor) -> dict:
    env.reset()
    raw = env.set_init_state(init_state)
    for _ in range(10):  # let objects settle
        raw, _, _, _ = env.step(DUMMY)
    executor.reset()
    success, t = False, 0
    t0 = time.perf_counter()
    for t in range(MAX_STEPS):
        act = executor.step((raw, t), t)
        raw, _r, done, _info = env.step(act)
        if env.check_success():
            success = True
            break
        if done:
            break
    if hasattr(executor.req, "join"):
        executor.req.join()
    s = executor.stats
    return {
        "success": success, "steps": t + 1, "queries": s.queries,
        "deadlineMiss": round(s.deadline_miss_rate, 4),
        "stepsElapsedPerQuery": round(s.mean_elapsed_per_query, 2),
        "dropMinusElapsed": round(s.mean_drop_minus_elapsed, 2),
        "heldPerQuery": round(s.held_per_query, 2),
        "seconds": round(time.perf_counter() - t0, 1),
    }


def parse_range(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b) + 1) if b else [int(a)])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["paused", "sync", "async"], required=True)
    ap.add_argument("--drop", choices=["nominal", "elapsed"], default="elapsed")
    ap.add_argument("--replan-every", type=int, default=8, help="sync/paused: actions used per query")
    ap.add_argument("--chunk-use-limit", type=int, default=None, help="async: L_c, fresh actions kept per chunk")
    ap.add_argument("--h-trig", type=int, default=10, help="async: fire next query at this queue length")
    ap.add_argument("--inject-delay-ms", type=float, default=0.0)
    ap.add_argument("--num-steps", type=int, default=None, help="SmolVLA denoising steps")
    ap.add_argument("--tasks", default="0,1,2")
    ap.add_argument("--init-states", default="40-47")
    args = ap.parse_args()

    policy, pre, post = load_policy(args.num_steps)
    suite = build_suite()
    for task_id in parse_range(args.tasks):
        env, lang = build_env(suite, task_id)
        dt = float(env.env.control_timestep)  # 0.05 s in LIBERO
        req = ThreadedRequester(make_chunk_fn(policy, pre, post, lang, args.inject_delay_ms))
        if args.mode == "async":
            ex = AsyncChunkExecutor(req, dt, args.h_trig, args.drop, args.chunk_use_limit,
                                    hold_fn=zero_hold, initial_action=DUMMY)
        else:
            ex = SyncChunkExecutor(req, dt, args.replan_every, charge_latency=args.mode == "sync",
                                   hold_fn=zero_hold, initial_action=DUMMY)
        inits = np.asarray(suite.get_task_init_states(task_id))
        for i in parse_range(args.init_states):
            rec = run_episode(env, inits[i], ex)
            print(json.dumps({"task": task_id, "init": i, **vars(args), **rec}), flush=True)
        env.close()


if __name__ == "__main__":
    main()
