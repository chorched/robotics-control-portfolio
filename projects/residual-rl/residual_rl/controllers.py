"""Classical baselines on FetchTorqueEnv (both use the controller's nominal model)."""

import numpy as np

from .env import DT


class JointPD:
    """Joint-space PD around IK targets of the reference, plus gravity compensation."""

    def __init__(self, kp: float, kd: float):
        self.kp, self.kd = kp, kd
        self.reset()

    def reset(self):
        self.q_warm = None
        self.q_des_prev = None

    def __call__(self, obs, env):
        target, _ = env.reference(env.t)
        q_des, _ = env.ik(target, env.q0 if self.q_warm is None else self.q_warm)
        self.q_warm = q_des
        qd_des = np.zeros_like(q_des) if self.q_des_prev is None else (q_des - self.q_des_prev) / DT
        self.q_des_prev = q_des
        q, qd = env.data.qpos[env.qadr], env.data.qvel[env.dadr]
        return self.kp * (q_des - q) + self.kd * (qd_des - qd) + env.gravity_torque()


class Impedance:
    """Cartesian spring-damper on the gripper, tau = J^T (K e + D e_dot) + g(q)."""

    def __init__(self, k: float, d: float):
        self.k, self.d = k, d

    def reset(self):
        pass

    def __call__(self, obs, env):
        target, vel_d = env.reference(env.t)
        jac = env.jacobian()
        v = jac @ env.data.qvel[env.dadr]
        return jac.T @ (self.k * (target - env.ee_pos()) + self.d * (vel_d - v)) + env.gravity_torque()
