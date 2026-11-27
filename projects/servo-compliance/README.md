# Compliant control of a low-cost servo arm (SO-101)

**Problem.** Low-cost hobby servos (Feetech STS3215) expose no torque sensing. Can an SO-101 arm detect contact and run admittance control well enough for gentle interaction?

**What I did**
- Audited the servo signals: `Present_Load` is a PWM duty cycle, not torque; `Present_Current` is sign–magnitude encoded and is read as unsigned by LeRobot.
- Calibrated a MuJoCo actuator model on public real-robot trajectories.
- Derived the stability condition of damping-only admittance with a one-step delay and checked it in closed-loop simulation.
- Benchmarked learned load detection on a public actuator dataset (NAD) with task-held-out splits.

**Key results**
- Actuator model test NRMSE **0.164 → 0.085** (fitted $k_p = 14.1$; the public model uses 998).
- Stability: with $g = k_{\mathrm{eff}}\Delta t / D_a$, stable iff $0 < g < 1$ with one-step delay, $0 < g < 2$ without. Sustained instability appeared exactly at $g = 1$ and $g = 2$ in simulation.
- Task-held-out load detection: a short-trained 1D-CNN reaches AUROC 0.85–1.00, above a physics-residual baseline in 3 of 4 directions. Decision thresholds do **not** transfer across tasks (negative result, kept).

**Limits.** Simulation and public data only; the hardware stage (SO-101 + load cell) has not started.

*Code and figures: in progress.*
