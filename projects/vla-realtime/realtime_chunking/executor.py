"""Action-chunk executors for a slow policy driving a fast control loop.

A chunking policy (SmolVLA, ACT, pi0, ...) looks at the observation captured at
control step ``t0`` and returns actions meant for steps ``t0, t0+1, ...``.
If the query takes ``E`` control steps to come back, the first ``E`` actions of
the chunk are already in the past. What the executor does with them decides
whether the policy works in real time.

Three executors are implemented, all driven one control step at a time through
``step(obs, t) -> action``:

* ``SyncChunkExecutor``  - blocking query; the world keeps moving while it
  thinks (the arm holds its last target), then the chunk is played from its
  first action.  ``charge_latency=False`` gives the "paused simulator"
  protocol, where inference costs no control steps.
* ``AsyncChunkExecutor`` - the query runs in the background; the next query is
  fired once the action queue drops to ``h_trig`` entries; when a chunk
  arrives its expired prefix is discarded and the rest is appended.

The expired prefix is the one detail that matters:

``drop_mode="nominal"``  d = ceil(L / dt), the latency measured in wall time.
``drop_mode="elapsed"``  d = min(ceil(L / dt), E), E = control steps that
                         really elapsed while the query was in flight.

They coincide only when the control loop runs exactly at real time. When the
loop slows down (CPU contention, a heavy renderer) the nominal rule throws
away actions that are still in the future; in the LIBERO experiments this was
~9.7 actions per request and took success from 15/24 to 0/24.
"""

from __future__ import annotations

import collections
import math
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, Protocol

import numpy as np

ChunkFn = Callable[[object], np.ndarray]  # obs -> (H, act_dim) actions starting at the obs step
HoldFn = Callable[[Optional[np.ndarray]], np.ndarray]


@dataclass
class ChunkResult:
    actions: np.ndarray
    latency_s: float  # measured wall-clock latency of the query
    step_at_dispatch: int


class Requester(Protocol):
    """Runs one policy query at a time."""

    def busy(self) -> bool: ...

    def dispatch(self, obs, step: int) -> None: ...

    def poll(self, step: int) -> Optional[ChunkResult]: ...

    def query_blocking(self, obs, step: int) -> ChunkResult: ...


# --------------------------------------------------------------------------- rules


def expired_prefix(latency_s: float, dt: float, steps_elapsed: int, mode: str) -> int:
    """Number of leading chunk actions to discard as already in the past."""
    d = int(math.ceil(latency_s / dt)) if dt > 0 else 0
    if mode == "elapsed":
        d = min(d, steps_elapsed)
    elif mode != "nominal":
        raise ValueError(f"unknown drop mode {mode!r}")
    return max(0, d)


def zero_hold(last: Optional[np.ndarray]) -> np.ndarray:
    """Default hold for delta-action policies: no motion, keep the last gripper value."""
    if last is None:
        raise ValueError("zero_hold needs an action shape; pass hold_fn explicitly")
    out = np.zeros_like(last)
    if out.shape[-1] == 7:  # LIBERO convention: 6-DoF delta + gripper
        out[-1] = last[-1]
    return out


# ------------------------------------------------------------------------ counters


@dataclass
class ExecStats:
    steps: int = 0
    fresh: int = 0  # action applied exactly at the step it was scheduled for
    late: int = 0  # action applied after its scheduled step
    held: int = 0  # queue empty, hold applied
    queries: int = 0  # queries dispatched
    harvested: int = 0  # queries whose result has been used
    expired: int = 0  # actions dropped as stale
    drop_minus_elapsed: float = 0.0  # sum over queries of (d - E); > 0 means future actions were lost
    elapsed_sum: int = 0

    @property
    def deadline_miss_rate(self) -> float:
        return (self.late + self.held) / max(1, self.steps)

    @property
    def mean_elapsed_per_query(self) -> float:
        return self.elapsed_sum / max(1, self.harvested)

    @property
    def mean_drop_minus_elapsed(self) -> float:
        return self.drop_minus_elapsed / max(1, self.harvested)

    @property
    def held_per_query(self) -> float:
        return self.held / max(1, self.queries)


# ----------------------------------------------------------------------- executors


class AsyncChunkExecutor:
    def __init__(
        self,
        requester: Requester,
        dt: float,
        h_trig: int,
        drop_mode: str = "elapsed",
        chunk_use_limit: Optional[int] = None,
        hold_fn: HoldFn = zero_hold,
        initial_action: Optional[np.ndarray] = None,
    ) -> None:
        self.req = requester
        self.dt = dt
        self.h_trig = h_trig
        self.drop_mode = drop_mode
        self.limit = chunk_use_limit
        self.hold_fn = hold_fn
        self.initial_action = initial_action
        self.reset()

    def reset(self) -> None:
        self.queue: collections.deque = collections.deque()  # (scheduled_step, action)
        self.last = None if self.initial_action is None else np.asarray(self.initial_action).copy()
        self.stats = ExecStats()

    def _harvest(self, t: int) -> None:
        res = self.req.poll(t)
        if res is None:
            return
        n_elapsed = t - res.step_at_dispatch
        d = expired_prefix(res.latency_s, self.dt, n_elapsed, self.drop_mode)
        d = min(d, len(res.actions))
        self.stats.expired += d
        self.stats.harvested += 1
        self.stats.drop_minus_elapsed += d - n_elapsed
        self.stats.elapsed_sum += n_elapsed
        # Keep the schedule contiguous: never re-use a step that is already queued.
        base = max(t, self.queue[-1][0] + 1 if self.queue else t)
        block = res.actions[d:] if self.limit is None else res.actions[d : d + self.limit]
        for i, a in enumerate(block):
            self.queue.append((base + i, np.asarray(a)))

    def step(self, obs, t: int) -> np.ndarray:
        self._harvest(t)
        if not self.req.busy() and len(self.queue) <= self.h_trig:
            self.req.dispatch(obs, t)
            self.stats.queries += 1
        self.stats.steps += 1
        if self.queue:
            scheduled, act = self.queue.popleft()
            if scheduled == t:
                self.stats.fresh += 1
            else:
                self.stats.late += 1
            self.last = act
            return act
        self.stats.held += 1
        return self.hold_fn(self.last)


class SyncChunkExecutor:
    """Blocking query, then play up to ``replan_every`` actions from the chunk start."""

    def __init__(
        self,
        requester: Requester,
        dt: float,
        replan_every: int,
        charge_latency: bool = True,
        hold_fn: HoldFn = zero_hold,
        initial_action: Optional[np.ndarray] = None,
    ) -> None:
        self.req = requester
        self.dt = dt
        self.k = replan_every
        self.charge = charge_latency
        self.hold_fn = hold_fn
        self.initial_action = initial_action
        self.reset()

    def reset(self) -> None:
        self.block: Optional[np.ndarray] = None
        self.idx = 0
        self.pending_holds = 0
        self.last = None if self.initial_action is None else np.asarray(self.initial_action).copy()
        self.stats = ExecStats()

    def step(self, obs, t: int) -> np.ndarray:
        self.stats.steps += 1
        if self.pending_holds > 0:  # the world moved on while the policy was thinking
            self.pending_holds -= 1
            self.stats.held += 1
            return self.hold_fn(self.last)
        if self.block is None or self.idx >= min(self.k, len(self.block)):
            res = self.req.query_blocking(obs, t)
            self.stats.queries += 1
            self.stats.harvested += 1
            self.block, self.idx = res.actions, 0
            if self.charge:
                # The first command after the query goes out ceil(L/dt) periods late;
                # the periods in between are spent holding the last target.
                lag = max(0, int(math.ceil(res.latency_s / self.dt)) - 1)
                self.stats.elapsed_sum += lag + 1
                if lag > 0:
                    self.pending_holds = lag - 1
                    self.stats.held += 1
                    return self.hold_fn(self.last)
        act = np.asarray(self.block[self.idx])
        self.idx += 1
        self.stats.fresh += 1
        self.last = act
        return act


# ----------------------------------------------------------------------- requesters


class SimulatedRequester:
    """Latency counted in control steps - for simulation studies without a GPU.

    ``latency_steps(rng)`` draws E, the control steps a query occupies.
    ``slowdown`` is how much slower than real time the control loop runs: the
    wall-clock latency the executor *measures* is ``E * dt * slowdown``. With
    ``slowdown > 1`` the nominal drop rule over-estimates the elapsed steps.
    """

    def __init__(
        self,
        chunk_fn: ChunkFn,
        dt: float,
        latency_steps: Callable[[np.random.Generator], int],
        slowdown: float = 1.0,
        seed: int = 0,
    ) -> None:
        self.chunk_fn = chunk_fn
        self.dt = dt
        self.latency_steps = latency_steps
        self.slowdown = slowdown
        self.rng = np.random.default_rng(seed)
        self._pending: Optional[tuple[int, int, np.ndarray]] = None  # (dispatch, ready, actions)

    def busy(self) -> bool:
        return self._pending is not None

    def dispatch(self, obs, step: int) -> None:
        e = max(1, int(self.latency_steps(self.rng)))
        self._pending = (step, step + e, self.chunk_fn(obs))

    def poll(self, step: int) -> Optional[ChunkResult]:
        if self._pending is None or step < self._pending[1]:
            return None
        t0, t1, acts = self._pending
        self._pending = None
        return ChunkResult(acts, (t1 - t0) * self.dt * self.slowdown, t0)

    def query_blocking(self, obs, step: int) -> ChunkResult:
        e = max(1, int(self.latency_steps(self.rng)))
        return ChunkResult(self.chunk_fn(obs), e * self.dt, step)


class ThreadedRequester:
    """Real wall-clock queries on a worker thread (used with a GPU policy)."""

    def __init__(self, chunk_fn: ChunkFn) -> None:
        self.chunk_fn = chunk_fn
        self._thread: Optional[threading.Thread] = None
        self._result: Optional[ChunkResult] = None
        self._error: Optional[BaseException] = None

    def busy(self) -> bool:
        return self._thread is not None

    def dispatch(self, obs, step: int) -> None:
        self._result, self._error = None, None

        def run() -> None:
            t0 = time.perf_counter()
            try:
                acts = self.chunk_fn(obs)
                self._result = ChunkResult(acts, time.perf_counter() - t0, step)
            except BaseException as exc:  # surfaced on the next poll
                self._error = exc

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def poll(self, step: int) -> Optional[ChunkResult]:
        if self._thread is None or self._thread.is_alive():
            return None
        self._thread.join()
        self._thread = None
        if self._error is not None:
            raise RuntimeError("policy query failed") from self._error
        return self._result

    def query_blocking(self, obs, step: int) -> ChunkResult:
        t0 = time.perf_counter()
        acts = self.chunk_fn(obs)
        return ChunkResult(acts, time.perf_counter() - t0, step)

    def join(self, timeout: float = 60.0) -> None:
        if self._thread is not None:
            self._thread.join(timeout)
