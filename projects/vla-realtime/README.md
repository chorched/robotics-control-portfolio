# Real-time execution of a VLA policy (SmolVLA on LIBERO)

**Problem.** SmolVLA inference takes a median of ~389 ms, about 8 control periods of 50 ms. When the simulator keeps running during inference, synchronous execution misses 88% of control deadlines. How should action chunks be executed so the policy still works in real time?

**What I did**
- Built a GPU container (LeRobot v0.4.4, LIBERO, MuJoCo 3.3.7, torch 2.8) with two timing protocols: simulation paused during inference, or advancing in real time.
- Found that the async executor dropped stale actions by *nominal* latency, discarding ~9.7 still-valid actions per request; changed it to drop by *elapsed control steps*.
- Ran pre-registered latency dose–response and chunk-length experiments.

**Key results**
- Elapsed-step alignment: success **0/24 → 15/24** under the same load and initial states (Newcombe 95% CI for the difference +0.38 to +0.79).
- What matters is $E$, the number of control steps that elapse per request, not latency in ms. Async needs usable chunk length $L_c > E$: at +300 ms ($E \approx 11.7$), $L_c = 14$ gives 10/24 vs $L_c = 8$ 1/24.
- Fewer denoising steps win in real time: on held-out tasks 3–9, 3 steps vs 10 steps succeed **51/56 vs 17/56** (exact McNemar $p = 5.4\times10^{-9}$).

**Limits.** Simulation only; 3 quasi-static LIBERO-Object tasks for the main grid; latency injected with sleep.

*Code and figures: in progress.*
