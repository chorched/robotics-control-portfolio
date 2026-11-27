# Robotics & Control Portfolio

[![tests](https://github.com/chorched/robotics-control-portfolio/actions/workflows/tests.yml/badge.svg)](https://github.com/chorched/robotics-control-portfolio/actions/workflows/tests.yml)

**Allen Li** · MSc Systems and Control, TU Delft · [LinkedIn](https://www.linkedin.com/in/allen-li-767913163)

Personal projects in robot control, estimation and robot learning. Each one starts from a concrete
engineering problem, builds the tool or controller, and measures it with pre-stated decision rules.
Negative results are reported, not dropped. All work here is my own and contains no code or data from
any employer.

## Projects

| | Project | Key result |
| --- | --- | --- |
| <img src="projects/vla-realtime/figures/timeline.png" width="260"> | **[Real-time execution of a VLA policy](projects/vla-realtime/)**<br>SmolVLA on LIBERO with a slow GPU policy and a 50 Hz loop. Built the chunk executors, found the stale-action bug, explained what governs real-time success | Success **0/24 → 15/24** after fixing the stale-action rule; 3 vs 10 denoising steps on held-out tasks: **51/56 vs 17/56** |
| <img src="projects/servo-compliance/figures/block_diagram.png" width="260"> | **[Compliant control of a low-cost servo arm](projects/servo-compliance/)**<br>Force control on an SO-101 that only accepts position commands: servo audit, model identification, admittance stability, learned load detection | Servo model NRMSE **0.164 → 0.085**; stability limit $g<2\sin\frac{\pi}{2(2d+1)}$ derived, then confirmed in closed-loop MuJoCo |
| <img src="projects/policy-evaluation/figures/result_budget.png" width="260"> | **[Sample-efficient evaluation of robot policies](projects/policy-evaluation/)**<br>Picking the best of 78 checkpoints with anytime-valid paired confidence sequences; numpy-only tool with a CLI | **32.7%** of the rollouts of exhaustive evaluation; offline loss picks BC checkpoints that are **26 points** worse |
| <img src="projects/residual-rl/figures/overview.png" width="260"> | **[Residual RL for torque-level tracking](projects/residual-rl/)**<br>Residual PPO on an impedance controller for a torque-actuated Fetch arm under friction, mass and force disturbances | Beats the tuned impedance controller in **7 of 8** pre-registered cells; plain PPO terminates in 60–80% of episodes |
| <img src="projects/toa-ekf-localization/figures/overview.png" width="260"> | **[Acoustic TOA calibration and EKF localization](projects/toa-ekf-localization/)**<br>Sensor-bias calibration, weighted NLS, Kalman and extended Kalman filtering with NIS/NEES consistency checks | Calibration cuts RMSE **165 → 17 mm**; EKF **13 mm**, consistent at the 95% level |
| <img src="projects/slip-safety-paper/figures/concept.png" width="260"> | **[Latency-aware slip safety for chunked policies](projects/slip-safety-paper/)** · *paper in progress*<br>A friction-cone safety layer that previews the blind steps of an action-chunked policy | Near the friction limit, drops **197 → 23**; latency-aware preview is faster than a fixed worst-case margin |

## How the work is done

- **Problem first.** Each project states what fails and why, before the method.
- **Pre-registered decisions.** Hypotheses, thresholds and analysis code were fixed (and hashed)
  before the main runs, and comparison sentences come from a stated rule.
- **Reproducible.** Every figure is regenerated from data in the repository by `figures/make_figures.py`.
  Shipped result tables let the statistics be re-derived bit for bit, and the tests check that they are.
- **Honest limits.** Each README ends with what the results do *not* show.

## Skills shown

Control: admittance and impedance control, discrete-time stability with delay, control barrier
functions, actuator identification · Estimation: Kalman / extended Kalman filters, nonlinear least
squares, sensor calibration, consistency testing · Robot learning: VLA deployment, action chunking,
flow matching, behaviour cloning, PPO / residual RL · Statistics: anytime-valid confidence sequences,
bootstrap, paired designs · Tools: Python, NumPy, PyTorch, MuJoCo, LeRobot, Stable-Baselines3,
Docker, MATLAB.

```bash
git clone https://github.com/chorched/robotics-control-portfolio
cd robotics-control-portfolio/projects/<project>
pip install -r requirements.txt && python -m pytest tests -q
```
