"""Identify an SO-101 servo model in MuJoCo from real-robot trajectories.

Data: the public NeuralActuator dataset (NAD), SO-101 0 g trajectories of
``go_up_and_stay_still`` and ``pick_and_place`` (16 train / 2 validation / 2 test
files, ~62.5 Hz). Model: ``so101_new_calib.xml`` from the SO-ARM100 repository.
Neither is redistributed here - see the project README for download links.

Each trajectory is replayed open loop: ``qpos`` starts at the first measured frame,
the recorded ``goal_pos`` is held zero-order over each sample interval and the
model integrates at its own 2 ms step. Error metric::

    NRMSE = mean over the 5 arm joints of RMSE(q_sim - q_meas) / std(q_meas)

Fitted: global position gain ``kp``, velocity gain ``kv`` and joint ``frictionloss``
(damping and armature kept at the model defaults). Step 1 is a 3x3x3 log grid, step
2 Nelder-Mead (12 evaluations) from the best grid point, both on two training
trajectories - the original study had a fixed budget of 200 replays. The validation
split is untouched by the fit and the test split is reported.

    python -m servo_compliance.calibration --nad <NAD>/data/so101 --xml <SO101>/so101_new_calib.xml
    python -m servo_compliance.calibration ... --fit        # re-run the identification
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ARM = [0, 1, 2, 3, 4]
TASKS = ["go_up_and_stay_still", "pick_and_place"]
FIT_FILES = ["pick_and_place/empty/train/008.csv", "go_up_and_stay_still/empty/train/007.csv"]

PUBLIC_NEW = dict(kp=998.22, kv=2.731, frictionloss=0.052)  # so101_new_calib.xml
PUBLIC_OLD = dict(kp=17.8, kv=0.0, frictionloss=0.052)  # so101_old_calib.xml gains
FITTED = dict(kp=14.0887, kv=0.4701, frictionloss=0.2844)  # result of --fit (run Bc1bt)


@dataclass
class Trajectory:
    name: str
    t: np.ndarray
    q: np.ndarray  # measured position (n, 6), rad
    goal: np.ndarray  # commanded position (n, 6), rad
    load: np.ndarray  # Present_Load counts (n, 6)


def load_split(nad_dir: str, split: str) -> list[Trajectory]:
    out = []
    for task in TASKS:
        for path in sorted(glob.glob(os.path.join(nad_dir, task, "empty", split, "*.csv"))):
            a = np.genfromtxt(path, delimiter=",", names=True, encoding="utf-8")
            col = lambda p: np.column_stack([a[f"{p}{i}"] for i in range(1, 7)]).astype(float)  # noqa: E731
            name = f"{task}/empty/{split}/{os.path.basename(path)}"
            out.append(Trajectory(name, a["timestamp"].astype(float), col("pos"), col("goal_pos"), col("load")))
    return out


class ServoArm:
    """MuJoCo SO-101 with global, settable position-servo parameters."""

    def __init__(self, xml: str, timestep: float = 0.002) -> None:
        import mujoco

        self.mj = mujoco
        self.m = mujoco.MjModel.from_xml_path(xml)
        self.d = mujoco.MjData(self.m)
        self.m.opt.timestep = timestep
        self.aid = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in JOINTS]
        jid = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINTS]
        self.qadr = np.array([self.m.jnt_qposadr[j] for j in jid])
        self.dadr = np.array([self.m.jnt_dofadr[j] for j in jid])

    def set_params(self, kp: float, kv: float, frictionloss: float) -> None:
        # MuJoCo position actuator: force = kp * ctrl - kp * q - kv * qdot
        for a in self.aid:
            self.m.actuator_gainprm[a, 0] = kp
            self.m.actuator_biasprm[a, :3] = (0.0, -kp, -kv)
        for dof in self.dadr:
            self.m.dof_frictionloss[dof] = frictionloss

    def replay(self, tr: Trajectory, record_force: bool = False):
        m, d, mj = self.m, self.d, self.mj
        mj.mj_resetData(m, d)
        d.qpos[self.qadr] = tr.q[0]
        mj.mj_forward(m, d)
        n = len(tr.t)
        qsim = np.empty((n, 6))
        force = np.empty((n, 6)) if record_force else None
        qsim[0] = d.qpos[self.qadr]
        if record_force:
            force[0] = d.actuator_force[self.aid]
        for i in range(1, n):
            d.ctrl[:] = tr.goal[i - 1]  # zero-order hold of the recorded command
            for _ in range(max(1, int(round((tr.t[i] - tr.t[i - 1]) / m.opt.timestep)))):
                mj.mj_step(m, d)
            qsim[i] = d.qpos[self.qadr]
            if record_force:
                force[i] = d.actuator_force[self.aid]
        return qsim, force


def nrmse(arm: ServoArm, trajs: list[Trajectory]) -> dict:
    res = [[] for _ in range(6)]
    meas = [[] for _ in range(6)]
    for tr in trajs:
        qs, _ = arm.replay(tr)
        for j in range(6):
            res[j].append(qs[:, j] - tr.q[:, j])
            meas[j].append(tr.q[:, j])
    per = {}
    for j in range(6):
        r, q = np.concatenate(res[j]), np.concatenate(meas[j])
        sd = np.std(q)
        per[JOINTS[j]] = float(np.sqrt(np.mean(r**2)) / sd) if sd > 0 else float("nan")
    return {"arm": float(np.mean([per[JOINTS[j]] for j in ARM])), "perJoint": per}


def fit(arm: ServoArm, fit_trajs: list[Trajectory]) -> dict:
    """Grid + Nelder-Mead on log10(kp), log10(kv + 1e-3), log10(frictionloss).

    Objective: mean over the fit trajectories of each trajectory's own arm NRMSE.
    """
    from scipy.optimize import minimize

    def unpack(x):
        return (float(np.clip(10 ** x[0], 3.0, 2e4)), float(np.clip(10 ** x[1] - 1e-3, 0.0, 200.0)),
                float(np.clip(10 ** x[2], 1e-4, 5.0)))

    def objective(x) -> float:
        arm.set_params(*unpack(x))
        val = float(np.mean([nrmse(arm, [tr])["arm"] for tr in fit_trajs]))
        return val if np.isfinite(val) and val < 50 else 50.0

    grid = [np.log10([kp, kv + 1e-3, fl]) for kp, kv, fl in
            itertools.product([30.0, 300.0, 2000.0], [0.01, 0.5, 5.0], [0.005, 0.052, 0.3])]
    best_val, best_x = min(((objective(x), x) for x in grid), key=lambda r: r[0])
    nm = minimize(objective, best_x, method="Nelder-Mead", options=dict(maxfev=12, xatol=1e-3, fatol=1e-4))
    x = nm.x if nm.fun < best_val else best_x
    kp, kv, fl = unpack(x)
    return {"kp": kp, "kv": kv, "frictionloss": fl, "objective": float(min(nm.fun, best_val)), "gridBest": best_val}


def torque_proxy_fit(arm: ServoArm, trajs: list[Trajectory]) -> dict:
    """Regress simulated actuator torque on Present_Load counts, per joint (slope Nm/count, R^2)."""
    tau, load = [], []
    for tr in trajs:
        _, f = arm.replay(tr, record_force=True)
        tau.append(f)
        load.append(tr.load)
    tau, load = np.concatenate(tau), np.concatenate(load)
    out = {}
    for j, name in enumerate(JOINTS):
        A = np.column_stack([load[:, j], np.ones(len(load))])
        coef, *_ = np.linalg.lstsq(A, tau[:, j], rcond=None)
        pred = A @ coef
        r2 = 1 - np.sum((tau[:, j] - pred) ** 2) / np.sum((tau[:, j] - tau[:, j].mean()) ** 2)
        out[name] = {"slopeNmPerCount": float(coef[0]), "interceptNm": float(coef[1]), "r2": float(r2)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nad", required=True, help="NAD data/so101 directory")
    ap.add_argument("--xml", required=True, help="so101_new_calib.xml")
    ap.add_argument("--fit", action="store_true", help="re-identify kp, kv, frictionloss")
    ap.add_argument("--out", type=Path, default=Path("results/calibration.json"))
    ap.add_argument("--trace-out", type=Path, default=Path("results/calibration_trace.csv"))
    ap.add_argument("--trace-joint", default="elbow_flex", choices=JOINTS)
    args = ap.parse_args()

    arm = ServoArm(args.xml)
    splits = {s: load_split(args.nad, s) for s in ("train", "validation", "test")}
    params = dict(FITTED)
    if args.fit:
        by_name = {t.name: t for t in splits["train"]}
        params = fit(arm, [by_name[n] for n in FIT_FILES])
        print("fitted:", params)

    result = {"params": {"fitted": params, "publicNew": PUBLIC_NEW, "publicOldGains": PUBLIC_OLD}, "nrmse": {}}
    for label, p in [("fitted", params), ("publicNew", PUBLIC_NEW)]:
        arm.set_params(p["kp"], p["kv"], p["frictionloss"])
        result["nrmse"][label] = {s: nrmse(arm, t) for s, t in splits.items()}
        print(f"{label:10s}", "  ".join(f"{s} {v['arm']:.4f}" for s, v in result["nrmse"][label].items()))

    arm.set_params(params["kp"], params["kv"], params["frictionloss"])
    result["torqueProxy"] = torque_proxy_fit(arm, splits["test"])

    # one test trajectory, measured vs simulated, for the figure
    tr, j = splits["test"][-1], JOINTS.index(args.trace_joint)
    rows = [tr.t - tr.t[0], tr.q[:, j], tr.goal[:, j]]
    for p in (params, PUBLIC_NEW):
        arm.set_params(p["kp"], p["kv"], p["frictionloss"])
        rows.append(arm.replay(tr)[0][:, j])
    args.trace_out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.trace_out, np.column_stack(rows)[::2], delimiter=",", fmt="%.5f",
               header=f"t_s,q_meas,q_goal,q_sim_fitted,q_sim_publicNew  # {args.trace_joint}, {tr.name}", comments="")
    args.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
