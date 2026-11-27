# Sample-efficient evaluation of robot learning policies

**Problem.** Picking the best checkpoint of a robot policy needs many rollouts, and the cheap signal (offline validation loss) picks the wrong one.

**What I did**
- Trained behavior-cloning and flow-matching policies on AdroitHandPen-v1 (D4RL `pen-human-v1` demos): 24 fits, 168 checkpoints.
- Implemented paired sequential selection with betting confidence sequences (Waudby-Smith & Ramdas) on shared initial states, with a guarantee of selecting a candidate within 5 points of the best with 95% probability.
- Re-sampled public real-robot evaluation logs (RoboArena) to test the method off-simulator.

**Key results**
- Selecting by task success instead of offline loss: **+26 points** (44% vs 18%); replicated over 10 seeds at **+27.9 points**.
- Paired selection on a 78-candidate pool stops after a median **32.7%** of the rollouts of exhaustive fixed-sample evaluation (20/20 runs stop; the marginal test hits the cap 18/20 times). Fresh initial states: 35.3%, 10/10.
- On RoboArena logs, pairing needs ~70% fewer sessions (5,112 vs 16,896).
- Cheap proxies (perturbed friction, training steps, offline loss) can prune the tail but cannot find the best (Kendall τ ≤ 0.03 within the top 20) — negative result, kept.

*Code and figures: in progress.*
