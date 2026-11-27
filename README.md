# Robotics & Control Portfolio

**Allen Li** · MSc Systems and Control, TU Delft

Personal projects in robot control, robot learning and real-time policy execution.
Each project states the problem, what I built, and the measured result, with its limits.
All work here is my own; it contains no code or data from any employer.

| Project | Problem | Key result | Status |
| --- | --- | --- | --- |
| [Compliant control of a low-cost servo arm](projects/servo-compliance/) | Contact sensing and admittance control on an SO-101 arm with STS3215 servos | Actuator model calibrated in MuJoCo: test NRMSE **0.164 → 0.085**; admittance stability bound confirmed in simulation | Simulation done, hardware stage pending |
| [Real-time execution of a VLA policy](projects/vla-realtime/) | A 389 ms SmolVLA inference spans ~8 control periods; how to execute action chunks on time | Aligning stale actions by elapsed steps: success **0/24 → 15/24**; 3 vs 10 denoising steps on held-out tasks: **51/56 vs 17/56** | Simulation (LIBERO) |
| [Sample-efficient evaluation of robot policies](projects/policy-evaluation/) | Choosing the best checkpoint among 78 candidates with as few rollouts as possible | Paired sequential selection uses **~33%** of the rollouts of exhaustive evaluation, with a certified 5-pt guarantee | Done |
| [Residual RL for torque-level tracking](projects/residual-rl/) | Fetch 7-DoF arm, end-effector tracking under friction, mass and force disturbances | Residual PPO beats the impedance baseline in **7 of 8** pre-registered condition × trajectory cells | Done |

> Code, figures and per-project write-ups are being added project by project.

## How results are reported

- Hypotheses and decision rules are written down before the main runs.
- Every number in a project README points to the run that produced it.
- Negative results and limitations are kept, not dropped.

## Contact

[LinkedIn](https://www.linkedin.com/in/allen-li-767913163)
