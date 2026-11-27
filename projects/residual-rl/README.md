# Residual RL for torque-level tracking (Fetch arm)

**Problem.** Track end-effector trajectories with joint torques on a 7-DoF Fetch arm in MuJoCo when friction, mass errors and external forces differ from the controller's model.

**What I did**
- Residual PPO on top of an impedance controller, compared with plain PPO and two tuned baselines (joint PD, impedance). 10 runs of ~1M steps each, 8 parallel CPU environments (~3,600 steps/s).
- Pre-registered a paired-bootstrap claim rule; comparison sentences are generated from the rule, not written by hand. Frozen code files are locked by SHA-256 and the run ledger is append-only.

**Key results**
- Residual PPO has lower steady-state error than the impedance baseline in **7 of 8** condition × trajectory cells, including out-of-distribution conditions; the 8th is "no decidable difference".
- Plain PPO terminates early in 60–80% of episodes.
- Honest boundary: joint PD is still better on trajectory T2 (e.g. 0.0071 m vs 0.0169 m), and the baselines got 43 tuning configs vs 3 for RL.

*Code and figures: in progress.*
