"""Closed-loop force control on the calibrated SO-101 in MuJoCo.

Pipeline (one 50 Hz control period)::

    actuator torque --quantise--> Present_Load counts (+-1 count noise)
      --delay d periods--> torque = slope * counts - gravity(q)
      --(E1: minus Coulomb friction when moving)--> f_hat = -(J J^T + eps I)^-1 J tau_ext
      --admittance--> p_ref_z += (dt / D_a) * S_delta(f_hat_z - f_d)
      --IK (damped least squares)--> servo position command

The tool presses a plate mounted on a vertical spring of stiffness ``k_e``. Because
the servo arm itself is compliant (kp = 14.1 gives ~680 N/m vertically), the loop
sees k_eff = (1/k_arm + 1/k_e)^-1, which is measured with a small-step test and used
to set D_a for a requested g = k_eff * dt / D_a.

    python -m servo_compliance.press --xml <SO101>/so101_new_calib.xml --d 1 --g 0.5
    python -m servo_compliance.press --xml ... --boundary      # 5 seeds x (d, g) grid
"""

from __future__ import annotations

import argparse
import json
import tempfile
import zlib
from collections import deque
from pathlib import Path

import numpy as np

from .admittance import dead_zone
from .calibration import FITTED

NARM = 5
DT = 0.02
DT_PHYS = 0.00025
NSUB = int(round(DT / DT_PHYS))
EE_RADIUS = 0.008
PLATE_MASS = 0.05
Q_NOM = np.array([0.0, 0.80, -1.54, -1.55, 0.0])  # well-conditioned, limit-safe press pose
F_D, DELTA, EPS, VEL_GUARD = 2.0, 0.1, 1e-3, 0.01
T_RUN, T_SS = 8.0, 3.0
TRAVEL_UP, TRAVEL_DOWN = 0.008, 0.012  # reference envelope around the touch point (m)
APPROACH_V, F_TOUCH = 0.008, 1.0
# Present_Load -> torque slopes (N m / count), arm joints, from calibration.torque_proxy_fit
LOAD_SLOPES = np.array([-0.00546, -0.00351, -0.00331, -0.00448, -0.00321])


def build_xml(base_xml: str, ke: float | None, plate_xy=None, plate_top_z=None, params=FITTED) -> str:
    base = Path(base_xml)
    txt = base.read_text(encoding="utf-8")

    def sub(old: str, new: str) -> None:
        nonlocal txt
        assert txt.count(old) == 1, f"pattern not unique in {base.name}: {old[:60]}"
        txt = txt.replace(old, new)

    sub('meshdir="assets"', f'meshdir="{(base.parent / "assets").as_posix()}"')
    sub('<mujoco model="so101_new_calib">',
        f'<mujoco model="so101_new_calib">\n  <option timestep="{DT_PHYS}" integrator="implicitfast"/>')
    sub('<joint damping="0.60" frictionloss="0.052" armature="0.028"/>',
        f'<joint damping="0.60" frictionloss="{params["frictionloss"]:.17g}" armature="0.028"/>')
    sub('<position kp="998.22" kv="2.731" forcerange="-2.94 2.94"/>',
        f'<position kp="{params["kp"]:.17g}" kv="{params["kv"]:.17g}" forcerange="-2.94 2.94"/>')
    site = ('<site group="3" name="gripperframe" pos="-0.0079 -0.000218121 -0.0981274" '
            'quat="0.707107 -0 0.707107 -2.37788e-17"/>')
    sub(site, site + f'\n<geom name="ee_tip" type="sphere" size="{EE_RADIUS}" pos="-0.0079 -0.000218121 -0.0981274" '
                     'contype="2" conaffinity="4" group="3" solref="0.002 1" friction="1 0.005 0.0001"/>')
    if ke is not None:
        damp = 0.9 * 2.0 * np.sqrt(ke * PLATE_MASS)
        sub("  </worldbody>", f"""    <body name="plate" pos="{plate_xy[0]:.6f} {plate_xy[1]:.6f} {plate_top_z - 0.006:.6f}">
      <joint name="plate_slide" type="slide" axis="0 0 1" stiffness="{ke:.6f}" damping="{damp:.6f}"
             range="-0.06 0" limited="true" springref="0"/>
      <geom type="box" size="0.06 0.06 0.006" mass="{PLATE_MASS}" contype="4" conaffinity="4" group="3"
            friction="1 0.005 0.0001" solref="0.002 1"/>
    </body>
  </worldbody>""")
    out = Path(tempfile.mkdtemp()) / f"so101_press_{'free' if ke is None else int(ke)}.xml"
    out.write_text(txt, encoding="utf-8")
    return str(out)


class PressSim:
    def __init__(self, xml_path: str) -> None:
        import mujoco

        self.mj = mujoco
        self.m = mujoco.MjModel.from_xml_path(xml_path)
        self.d = mujoco.MjData(self.m)
        self.dk = mujoco.MjData(self.m)  # scratch data for kinematics
        self.site = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self.has_plate = self.m.nq > 6
        self.ke = float(self.m.jnt_stiffness[6]) if self.has_plate else 0.0
        self.lo, self.hi = self.m.jnt_range[:NARM, 0].copy(), self.m.jnt_range[:NARM, 1].copy()
        self.c_lo, self.c_hi = self.m.actuator_ctrlrange[:NARM, 0].copy(), self.m.actuator_ctrlrange[:NARM, 1].copy()
        self.g = abs(float(self.m.opt.gravity[2]))
        self.d.qpos[:NARM] = Q_NOM
        self.q_ref = Q_NOM.copy()
        self.d.ctrl[:NARM] = Q_NOM
        mujoco.mj_forward(self.m, self.d)

    def settle(self, t: float) -> None:
        self.d.ctrl[:NARM] = np.clip(self.q_ref, self.c_lo, self.c_hi)
        for _ in range(int(round(t / self.m.opt.timestep))):
            self.mj.mj_step(self.m, self.d)

    def snapshot(self):
        return self.d.qpos.copy(), self.d.qvel.copy(), self.d.ctrl.copy(), self.d.time, self.q_ref.copy()

    def restore(self, snap) -> None:
        self.d.qpos[:], self.d.qvel[:], self.d.ctrl[:], self.d.time, q_ref = snap
        self.q_ref = q_ref.copy()
        self.d.qfrc_applied[:] = 0.0
        self.mj.mj_forward(self.m, self.d)

    def ee(self) -> np.ndarray:
        return self.d.site_xpos[self.site].copy()

    def _kin(self, q):
        dk = self.dk
        dk.qpos[:] = self.d.qpos
        dk.qpos[:NARM] = q
        dk.qvel[:] = 0.0
        dk.qacc[:] = 0.0
        self.mj.mj_forward(self.m, dk)
        return dk

    def jac_grav(self, q):
        dk = self._kin(q)
        jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        self.mj.mj_jacSite(self.m, dk, jp, jr, self.site)
        dk.qacc[:] = 0.0  # rne with qvel = qacc = 0 gives the gravity torque
        dk.qvel[:] = 0.0
        g = np.zeros(self.m.nv)
        self.mj.mj_rne(self.m, dk, 1, g)
        return jp[:, :NARM].copy(), g[:NARM].copy()

    def ik(self, p_target, seed, iters: int = 12, lam: float = 0.01):
        q = np.array(seed, float)
        for _ in range(iters):
            dk = self._kin(q)
            e = np.asarray(p_target) - dk.site_xpos[self.site]
            if np.linalg.norm(e) < 1e-6:
                break
            jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
            self.mj.mj_jacSite(self.m, dk, jp, jr, self.site)
            J = jp[:, :NARM]
            dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(3), e)
            if np.linalg.norm(dq) > 0.15:
                dq *= 0.15 / np.linalg.norm(dq)
            q = np.clip(q + dq, self.lo, self.hi)
        return q

    def f_true(self) -> float:
        return self.ke * (-self.d.qpos[6]) - PLATE_MASS * self.g if self.has_plate else 0.0

    def step(self, q_cmd) -> None:
        self.d.ctrl[:NARM] = np.clip(q_cmd, self.c_lo, self.c_hi)
        for _ in range(NSUB):
            self.mj.mj_step(self.m, self.d)


class PressRig:
    """Arm + spring plate, settled once and restored before each run."""

    def __init__(self, base_xml: str, ke: float, params=FITTED) -> None:
        free = PressSim(build_xml(base_xml, None, params=params))
        free.settle(10.0)
        self.p0 = free.ee()
        self.ke = ke
        top = float(self.p0[2] - EE_RADIUS - 0.0002)
        self.sim = PressSim(build_xml(base_xml, ke, self.p0[:2], top, params=params))
        self.sim.settle(10.0)
        self.snap = self.sim.snapshot()
        self.friction = params["frictionloss"]

    def measure_keff(self) -> float:
        """Command 1-6 mm downward steps and regress the true contact force on them."""
        s = self.sim
        s.restore(self.snap)
        q = s.q_ref.copy()
        offs, fs = [], []
        for k in range(1, 7):
            p = self.p0.copy()
            p[2] -= 0.001 * k
            q = s.ik(p, q)
            s.q_ref = q.copy()
            s.settle(1.5)
            offs.append(0.001 * k)
            fs.append(s.f_true())
        return float(np.polyfit(offs, fs, 1)[0])

    def run(self, D_a: float, delay: int, estimator: str = "E1", seed: int = 0) -> dict:
        s = self.sim
        s.restore(self.snap)
        rng = np.random.default_rng(seed)
        buf: deque = deque(maxlen=delay + 1)
        q_cmd = s.q_ref.copy()
        p_ref = self.p0.copy()
        z_lo, z_hi = self.p0[2] - TRAVEL_DOWN, self.p0[2] + TRAVEL_UP
        t_run, n_pin, n_steps = None, 0, 0
        trace = []
        while True:
            q, qd = s.d.qpos[:NARM].copy(), s.d.qvel[:NARM].copy()
            tau_act = s.d.qfrc_actuator[:NARM].copy()
            counts = np.round(tau_act / LOAD_SLOPES) + rng.integers(-1, 2, NARM)
            J, grav = s.jac_grav(q)
            buf.append((counts, qd, J, grav))
            counts_d, qd_d, J_d, g_d = buf[0]  # what the controller sees, d periods old
            tau_ext = LOAD_SLOPES * counts_d - g_d
            if estimator == "E1":  # remove Coulomb friction where the joint is clearly moving
                moving = np.abs(qd_d) >= VEL_GUARD
                tau_ext = tau_ext - np.where(moving, self.friction * np.sign(qd_d), 0.0)
            f_hat = -np.linalg.solve(J_d @ J_d.T + EPS * np.eye(3), J_d @ tau_ext)
            if t_run is None:  # approach until first touch
                p_ref[2] -= APPROACH_V * DT
                if s.f_true() >= F_TOUCH:
                    t_run = 0.0
            else:
                p_ref[2] += (DT / D_a) * dead_zone(f_hat[2] - F_D, DELTA)
            p_ref[2] = min(max(p_ref[2], z_lo), z_hi)
            n_pin += int(p_ref[2] <= z_lo + 1e-9 or p_ref[2] >= z_hi - 1e-9)
            q_cmd = s.ik(p_ref, q_cmd)
            s.step(q_cmd)
            n_steps += 1
            if t_run is not None:
                trace.append((t_run, s.f_true(), float(f_hat[2]), float(p_ref[2] - self.p0[2])))
                t_run += DT
                if t_run >= T_RUN - 1e-9:
                    break
            if n_steps > (T_RUN + 3.0) / DT:
                break
        tr = np.array(trace)
        win = tr[:, 0] >= T_RUN - T_SS
        f_ss = tr[win, 1]
        ripple = float(np.std(f_ss))
        sat = n_pin / max(1, n_steps)
        return {
            "ssErr": float(abs(f_ss.mean() - F_D)),
            "ripple": ripple,
            "eSS": float(np.mean(tr[win, 1] - tr[win, 2])),
            "saturatedFrac": float(sat),
            "stable": bool(ripple <= 0.1 * F_D and sat < 0.5 and np.max(np.abs(tr[:, 1])) < 20.0),
            "trace": tr[::2].round(5).tolist(),
        }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xml", required=True)
    ap.add_argument("--ke", type=float, default=300.0, help="plate spring stiffness, N/m")
    ap.add_argument("--d", type=int, default=1, help="sensing delay in control periods")
    ap.add_argument("--g", type=float, default=0.5, help="loop gain k_eff*dt/D_a")
    ap.add_argument("--estimator", choices=["E0", "E1"], default="E1")
    ap.add_argument("--boundary", action="store_true", help="replicate the stability boundary, 5 seeds per point")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    rig = PressRig(args.xml, args.ke)
    k_eff = rig.measure_keff()
    print(f"k_e = {args.ke:.0f} N/m  ->  measured k_eff = {k_eff:.1f} N/m")
    if not args.boundary:
        r = rig.run(k_eff * DT / args.g, args.d, args.estimator, seed=1)
        print({k: v for k, v in r.items() if k != "trace"})
        if args.out:
            args.out.write_text(json.dumps({"kEff": k_eff, "d": args.d, "g": args.g, **r}))
        return
    rows = []
    for d, gs in ((1, (0.6, 0.7, 0.8, 0.9, 1.0, 1.2)), (0, (1.4, 1.6, 1.8, 2.0, 2.2, 2.5))):
        for g in gs:
            runs = [rig.run(k_eff * DT / g, d, "E1", seed=zlib.crc32(f"{d}-{g}".encode()) % 9973 + 7919 * i)
                    for i in range(5)]
            n_st = sum(r["stable"] for r in runs)
            rows.append({"d": d, "g": g, "stable": n_st, "n": 5})
            print(f"d={d} g={g:.1f}: stable {n_st}/5")
    if args.out:
        args.out.write_text(json.dumps({"kEff": k_eff, "boundary": rows}, indent=1))


if __name__ == "__main__":
    main()
