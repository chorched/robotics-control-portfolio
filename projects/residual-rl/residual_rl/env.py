"""Torque-level end-effector tracking on the Fetch arm (MuJoCo, gymnasium-robotics model).

The stock Fetch model is position-controlled through a mocap weld. Here the weld is
released and seven joint-torque actuators are injected, so a controller commands raw
joint torques (limits ``TAU_MAX``). Task: follow a 0.1 m circle (T1) or figure-eight
(T2) with the gripper for 8 s at 50 Hz.

Disturbances are sampled per episode in the *simulated* plant; the controller's own
model (Jacobian, gravity, IK) stays nominal:

* C0 nominal
* C1 training distribution: extra viscous + Coulomb joint friction, link masses x0.8-1.3,
  random 0.2 s force pulses on the end effector, joint-position noise at reset
* C2 out of distribution: 1.5x friction / pulse scale, masses x1.3-1.5
* C3 out of distribution: masses x1.2 only

Episodes terminate on divergence (> 0.15 m), joint speed > 10 rad/s, a non-finite
state or a MuJoCo warning.
"""
import math
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
import mujoco
import numpy as np
from gymnasium import spaces

TIMESTEP = 0.002
DT = 0.02
FRAME_SKIP = int(round(DT / TIMESTEP))
Q_RESET = (0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0)
TAU_MAX = (5.0, 224.0, 11.0, 101.0, 5.0, 44.0, 5.0)
IK_LAMBDA = 1e-2
IK_STEP_CLIP = 0.05
Q_NOISE = 0.02
PERIOD = 8.0
OMEGA = 1.0
RADIUS = 0.10
REACH_TOL = 1e-3
JOINT_MARGIN_DEG = 5.0
EPISODE_STEPS = int(round(PERIOD / DT))
HOLD_STEPS = int(round(2.0 / DT))
HOLD_DRIFT_LIMIT = 0.01
DROP_THRESHOLD = 0.005
W1, E0, W2, W3, QDOT0, R_CLIP = 1.0, 0.01, 0.01, 0.001, 1.0, 10.0
E_DIVERGE = 0.15
QDOT_MAX = 10.0
OBS_DIM = 32
B_MAX = 2.0
TAU_C_MAX = 1.0
MASS_RANGE = (0.8, 1.3)
PULSE_RATE = 0.5
PULSE_DUR = 0.2
PULSE_MAG = 10.0
APPROACHES = ("data_eq_active", "model_eq_active0")
ARM_JOINTS = ("robot0:shoulder_pan_joint", "robot0:shoulder_lift_joint",
              "robot0:upperarm_roll_joint", "robot0:elbow_flex_joint",
              "robot0:forearm_roll_joint", "robot0:wrist_flex_joint",
              "robot0:wrist_roll_joint")
ARM_ACT = "arm_tau_"
EE_SITE = "robot0:grip"
MOCAP_BODY = "robot0:mocap"


def find_fetch_xml():
    root = Path(gymnasium_robotics.__file__).resolve().parent
    hits = sorted(p for p in root.rglob("reach.xml") if p.parent.name == "fetch")
    if not hits:
        raise FileNotFoundError("no fetch/reach.xml under " + str(root))
    return hits[0]


def native_timestep():
    return float(mujoco.MjSpec.from_file(str(find_fetch_xml())).compile().opt.timestep)


def name_id(model, kind, name):
    idx = mujoco.mj_name2id(model, kind, name)
    if idx < 0:
        raise KeyError("MuJoCo name not found: " + name)
    return idx


def compile_model(inject=True):
    spec = mujoco.MjSpec.from_file(str(find_fetch_xml()))
    if inject:
        for i, joint in enumerate(ARM_JOINTS):
            spec.add_actuator(name=ARM_ACT + str(i), target=joint,
                              trntype=mujoco.mjtTrn.mjTRN_JOINT,
                              ctrllimited=True, ctrlrange=[-TAU_MAX[i], TAU_MAX[i]])
    model = spec.compile()
    model.opt.timestep = TIMESTEP
    return model


def weld_ids(model):
    mocap = name_id(model, mujoco.mjtObj.mjOBJ_BODY, MOCAP_BODY)
    return np.array([e for e in range(model.neq)
                     if model.eq_type[e] == mujoco.mjtEq.mjEQ_WELD
                     and mocap in (model.eq_obj1id[e], model.eq_obj2id[e])], dtype=int)


def arm_qadr(model):
    return np.array([model.jnt_qposadr[name_id(model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in ARM_JOINTS])


def reset_state(model, data, approach, welds):
    mujoco.mj_resetData(model, data)
    data.qpos[arm_qadr(model)] = Q_RESET
    if approach == "data_eq_active":
        data.eq_active[welds] = 0
    mujoco.mj_forward(model, data)


def build_model(approach):
    model = compile_model()
    welds = weld_ids(model)
    if approach == "model_eq_active0":
        model.eq_active0[welds] = 0
    elif approach != "data_eq_active":
        raise ValueError("unknown release approach: " + str(approach))
    data = mujoco.MjData(model)
    reset_state(model, data, approach, welds)
    return model, data, welds


def ref_pos_vel(t, x0, kind="T1"):
    a = OMEGA * t
    c, s = math.cos(a), math.sin(a)
    if kind == "T1":
        pos = x0 + RADIUS * np.array([c - 1.0, s, 0.0])
        vel = RADIUS * OMEGA * np.array([-s, c, 0.0])
    elif kind == "T2":
        pos = x0 + RADIUS * np.array([s, 0.5 * math.sin(2 * a), 0.0])
        vel = RADIUS * OMEGA * np.array([c, math.cos(2 * a), 0.0])
    else:
        raise ValueError("unknown trajectory: " + str(kind))
    return pos, vel


def tracking_reward(err, qdot):
    return -W1 * (err / E0) ** 2 - W3 * float(np.sum(np.asarray(qdot) ** 2)) / (7.0 * QDOT0 ** 2)


def shaped_reward(r_env, a):
    return max(-R_CLIP, r_env - W2 * float(np.sum(np.asarray(a, dtype=float) ** 2)) / 7.0)


def sample_pulse_schedule(rng, n, scale):
    if n > 100_000:
        raise ValueError("pulse schedule too long: " + str(n))
    sched = np.zeros((n, 3))
    events, left, force = [], 0, np.zeros(3)
    dur = int(round(PULSE_DUR / DT))
    p_start = PULSE_RATE * DT
    for k in range(n):
        if left == 0 and rng.uniform() < p_start:
            mag = float(rng.uniform(0.0, PULSE_MAG * scale))
            d = rng.normal(size=3)
            d = d / np.linalg.norm(d)
            force = mag * d
            left = dur
            events.append({"step": k, "mag_N": mag, "dir": d.tolist()})
        if left > 0:
            sched[k] = force
            left -= 1
    return sched, events


def condition_spec(name, scale=1.0):
    specs = {
        "C0": None,
        "C1": {"scale": scale, "friction": True, "mass_range": MASS_RANGE, "pulses": True, "q_noise": Q_NOISE},
        "C2": {"scale": 1.5 * scale, "friction": True, "mass_range": (1.3, 1.5), "pulses": True,
               "q_noise": 1.5 * Q_NOISE},
        "C3": {"scale": 0.0, "friction": False, "mass_range": (1.2, 1.2), "pulses": False, "q_noise": 0.0},
    }
    if name not in specs:
        raise ValueError("unknown condition: " + str(name))
    spec = specs[name]
    return None if spec is None else dict(spec, name=name)


class FetchTorqueEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, approach, max_steps=EPISODE_STEPS, q_noise=Q_NOISE, trajectory="T1", condition=None):
        super().__init__()
        self.approach = approach
        self.condition = condition
        self.model, self.data, self.welds = build_model(approach)
        self.nom_model = build_model(approach)[0]
        m = self.model
        joints = [name_id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in ARM_JOINTS]
        self.qadr = np.array([m.jnt_qposadr[j] for j in joints])
        self.dadr = np.array([m.jnt_dofadr[j] for j in joints])
        rng = m.jnt_range[joints]
        limited = rng[:, 1] > rng[:, 0]
        self.qlo = np.where(limited, rng[:, 0], -np.inf)
        self.qhi = np.where(limited, rng[:, 1], np.inf)
        self.act = np.array([name_id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, ARM_ACT + str(i))
                             for i in range(len(ARM_JOINTS))])
        self.site = name_id(m, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
        self.scratch = mujoco.MjData(self.nom_model)
        self.max_steps = max_steps
        self.q_noise = condition["q_noise"] if condition else q_noise
        self.links = np.array(m.jnt_bodyid[joints])
        self.ee_body = m.site_bodyid[self.site]
        self.nom_mass = m.body_mass[self.links].copy()
        self.nom_inertia = m.body_inertia[self.links].copy()
        self.nom_damp = m.dof_damping[self.dadr].copy()
        self.nom_fric = m.dof_frictionloss[self.dadr].copy()
        self.ep_params = {}
        self.pulse_sched = None
        self.trajectory = trajectory
        self.q0 = self.data.qpos[self.qadr].copy()
        self.x0 = self.ee_pos()
        self.action_space = spaces.Box(-np.array(TAU_MAX, dtype=np.float32), np.array(TAU_MAX, dtype=np.float32),
                                       shape=(len(ARM_JOINTS),), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.t = 0.0
        self.steps = 0
        self.warn_count = 0

    def reference(self, t):
        return ref_pos_vel(t, self.x0, self.trajectory)

    def ee_pos(self):
        return self.data.site_xpos[self.site].copy()

    def _sync_nominal(self):
        s = self.scratch
        s.qpos[:] = self.data.qpos
        s.qvel[:] = 0.0
        s.eq_active[:] = self.data.eq_active
        mujoco.mj_forward(self.nom_model, s)
        return s

    def _arm_jac(self, d):
        jac = np.zeros((3, self.nom_model.nv))
        rot = np.zeros((3, self.nom_model.nv))
        mujoco.mj_jacSite(self.nom_model, d, jac, rot, self.site)
        return jac[:, self.dadr]

    def jacobian(self):
        return self._arm_jac(self._sync_nominal())

    def gravity_torque(self):
        return self._sync_nominal().qfrc_bias[self.dadr].copy()

    def ik(self, target, q_init, iters=300, tol=1e-4):
        s = self.scratch
        s.qpos[:] = self.data.qpos
        s.qvel[:] = 0.0
        q = np.clip(np.asarray(q_init, dtype=float), self.qlo, self.qhi)
        for _ in range(iters):
            s.qpos[self.qadr] = q
            mujoco.mj_kinematics(self.nom_model, s)
            mujoco.mj_comPos(self.nom_model, s)
            e = target - s.site_xpos[self.site]
            if np.linalg.norm(e) < tol:
                break
            jac = self._arm_jac(s)
            dq = jac.T @ np.linalg.solve(jac @ jac.T + IK_LAMBDA * IK_LAMBDA * np.eye(3), e)
            peak = float(np.max(np.abs(dq)))
            if peak > IK_STEP_CLIP:
                dq *= IK_STEP_CLIP / peak
            q = np.clip(q + dq, self.qlo, self.qhi)
        s.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.nom_model, s)
        err =float(np.linalg.norm(target - s.site_xpos[self.site]))
        return q, err

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        reset_state(self.model, self.data, self.approach, self.welds)
        self.ep_params = self._sample_disturbance()
        noise =self.np_random.uniform(-self.q_noise, self.q_noise, size=len(self.qadr))
        self.data.qpos[self.qadr] = self.q0 + noise
        mujoco.mj_forward(self.model, self.data)
        self.x0 = self.ee_pos()
        self.t = 0.0
        self.steps = 0
        self.warn_count = 0
        return self._obs(), {"ee_start": self.x0.tolist()}

    def _sample_disturbance(self):
        m = self.model
        m.body_mass[self.links] = self.nom_mass
        m.body_inertia[self.links] = self.nom_inertia
        m.dof_damping[self.dadr] = self.nom_damp
        m.dof_frictionloss[self.dadr] = self.nom_fric
        params = {}
        c = self.condition
        if c is not None:
            n = len(self.dadr)
            if c["friction"]:
                b = self.np_random.uniform(0.0, B_MAX * c["scale"], size=n)
                tc = self.np_random.uniform(0.0, TAU_C_MAX * c["scale"], size=n)
                m.dof_damping[self.dadr] = self.nom_damp + b
                m.dof_frictionloss[self.dadr] = tc
                params["damping"] = b.tolist()
                params["coulomb"] = tc.tolist()
            if c["mass_range"] is not None:
                lo, hi = c["mass_range"]
                sm = self.np_random.uniform(lo, hi, size=len(self.links))
                m.body_mass[self.links] = self.nom_mass * sm
                m.body_inertia[self.links] = self.nom_inertia * sm[:, None]
                params["mass_scale"] = sm.tolist()
        mujoco.mj_setConst(m, self.data)
        self.pulse_sched = None
        if c is not None and c["pulses"]:
            self.pulse_sched, events = sample_pulse_schedule(self.np_random, self.max_steps, c["scale"])
            params["pulse_events"] = events
        return params

    def _warnings(self):
        return int(sum(int(self.data.warning[k].number) for k in range(len(self.data.warning))))

    def step(self, action):
        tau = np.clip(np.asarray(action, dtype=float).reshape(-1), -np.array(TAU_MAX), np.array(TAU_MAX))
        self.data.ctrl[self.act] = tau
        active = self.pulse_sched is not None and self.steps < len(self.pulse_sched)
        self.data.xfrc_applied[self.ee_body, :3] = self.pulse_sched[self.steps] if active else 0.0
        warn_before = self._warnings()
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        self.t = self.steps * DT
        warn_after = self._warnings()
        self.warn_count += warn_after - warn_before
        finite = bool(np.all(np.isfinite(self.data.qpos)) and np.all(np.isfinite(self.data.qvel)))
        if finite:
            qdot = self.data.qvel[self.dadr]
            err = float(np.linalg.norm(self.reference(self.t)[0] - self.ee_pos()))
            r_env = tracking_reward(err, qdot)
            diverged = err > E_DIVERGE
            fast = float(np.max(np.abs(qdot))) > QDOT_MAX
        else:
            err = float("nan")
            r_env = 0.0
            diverged = fast = False
        warned = warn_after > warn_before
        terminated = (not finite) or diverged or fast or warned
        truncated = (not terminated) and self.steps >= self.max_steps
        reason = ("nonfinite" if not finite else "diverged" if diverged else "qdot" if fast
                  else "mj_warning" if warned else "")
        info = {"e_norm": err, "t": self.t, "r_env": r_env, "remaining_steps": self.max_steps - self.steps,
                "term_reason": reason}
        return self._obs(), r_env, terminated, truncated, info

    def _obs(self):
        ref, ref_vel = self.reference(self.t)
        ee = self.ee_pos()
        ee_vel = self.jacobian() @ self.data.qvel[self.dadr]
        phase = OMEGA * self.t
        obs = np.concatenate([self.data.qpos[self.qadr], self.data.qvel[self.dadr],
                              ee - self.x0, ee_vel, ref - self.x0, ref_vel, ref - ee,
                              [math.sin(phase), math.cos(phase), self.t / PERIOD]])
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


