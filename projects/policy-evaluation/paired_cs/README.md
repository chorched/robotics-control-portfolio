# paired-cs

Sequential checkpoint selection with hedged-betting confidence sequences, plus
paired (same-initial-state) certification of a selection that has already been
made.  Extracted from my original analysis scripts (runs A2t / A4p / A5p) so the same
statistics can be reused without copying them.

Dependency: **numpy only** (Python >= 3.9).

```
pip install -e .
```

## The problem

You have `K` candidates (checkpoints, hyper-parameter settings, policies) and a
distribution of initial states.  One *rollout* = evaluate one candidate on one
initial state and observe success/failure.  Rollouts are expensive, so:

1. **select** -- spend as few rollouts as possible to name a candidate that is
   provably within `delta` of the best candidate.
2. **certify** -- given outcomes that already exist for a fixed set of initial
   states, bound how much success rate the selected candidate gave up, and turn
   that bound into a three-valued verdict.

## Input format: the candidate x initial-state table

An `.npz` file (written with `numpy.savez`) holding

| key | shape | meaning |
| --- | --- | --- |
| `Y` | `(K, n)` | outcome per candidate per initial state, values in `[0, 1]` (success indicators).  Required. |
| `seeds` | `(n,)` int | label of each initial state (must be unique).  Optional, defaults to `0..n-1`. |
| `evaluated` | `(K, n)` 0/1 | `1` = this cell was really observed, `0` = missing.  Optional, defaults to all ones. |
| `candidate` or `candidate_label` | `(K,)` str | candidate ids.  Optional, defaults to `c0..c{K-1}`. |

```python
import numpy as np
np.savez("table.npz", Y=Y, seeds=seeds, evaluated=mask, candidate=labels)
```

`Y[:, i]` is one draw of the pseudo-population: **every candidate is scored on
the same initial state `i`.**  Missing cells (different seed sets per candidate)
are handled by `evaluated`; paired statistics use only the states observed for
both candidates of a pair.

## Command 1 -- `select`

```bash
python -m paired_cs select --table table.npz --key Y \
    --delta 0.05 --alpha 0.05 --method pair --seed 0 --cap 200000
```

* `--method pair` runs `S2_pair` (pairwise confidence sequences, recommend
  when rollouts are not the binding constraint but confidence is);
  `--method marg` runs `S2_marg` (one marginal sequence per candidate).
* `--seed` seeds the with-replacement sampler over the table columns, so a
  `(seed, method, table)` triple is a reproducible replication.
* `--cap` is a hard cap on total rollouts; hitting it yields
  `"stopReason": "cap"` and `"selected": null`.
* `--key` names the npz array holding the outcome matrix (default `Y`).

Output (stdout, JSON):

```json
{
  "command": "select", "method": "S2_pair", "seed": 0, "delta": 0.05,
  "alpha": 0.05, "cap": 200000, "nCandidates": 78, "nPool": 3000,
  "stopped": true, "stopReason": "rule",
  "selected": 38, "selectedLabel": "fm_mask_N20_s43/step5000",
  "totalRollouts": 152928, "rounds": 2120, "nLive": 72,
  "chosenLower": -0.0475000000000001, "chosenUpper": null,
  "csEmpty": 0, "csDegenerate": 0, "runtimeSec": 179.4
}
```

`stopReason` is `rule` (stopping rule fired), `single` (a single live candidate
remained), or `cap`.

## Command 2 -- `certify`

```bash
python -m paired_cs certify --reference table.npz --selected 38 \
    --delta 0.05 --alpha 0.05 --n-comparisons 231
```

* `--selected` accepts a candidate index (`38`) or a label from the npz.
* `--n-comparisons m` is the number of comparisons that `--alpha` has to cover;
  every pair runs at `alpha / m`.  Use `n_selected * (K - 1)`
  (`231 = 3 * 77` in the A5p batch) when several candidates are certified on the
  same table.  Do not pass `1` unless you really have a single comparison.

Output (stdout, JSON):

```json
{
  "command": "certify", "selected": "fm_mask_N20_s43/step5000",
  "selectedIndex": 38, "delta": 0.05, "alpha": 0.05,
  "nComparisons": 231, "alphaPerComparison": 0.00021645021645021645,
  "regretLower": 0.0, "regretUpper": 0.0275, "verdict": "correct",
  "decisiveRival": "fm_mask_N20_s29/step10000", "nDecisive": 10000,
  "nCandidates": 78, "missingRivals": [], "nMissingRivals": 0
}
```

`[regretLower, regretUpper]` bounds `max_j p_j - p_selected`.  The verdict is

* `correct` -- `regretUpper <= delta`, i.e. the selection lost at most `delta`;
* `wrong` -- `regretLower > delta`, i.e. it provably lost more than `delta`;
* `undecidable` -- the interval straddles `delta`.

`missingRivals` lists the rivals that have **no initial state in common** with
the selected candidate (`n = 0`), as `{"rival": label, "rivalIndex": index}`
entries.  Such a rival still enters the regret: its difference is only known to
lie in `[-1, 1]`, so it forces `regretUpper = 1` and the verdict `undecidable`
instead of being silently dropped.  `nMissingRivals` is the count.  Only when
*every* rival has `n = 0` does `certify` raise `ValueError` (there is nothing to
certify against).

Bounds are rounded to 6 decimals by default (`--decimals`, or `--no-round` for
full precision).

## Guarantees and the assumptions they rest on

The confidence sequences are **anytime-valid**: they hold simultaneously at
every sample size, which is why no fixed horizon has to be fixed before the
experiment, and why the same machinery serves both the sequential selection and
the post-hoc certification.

* Initial states are i.i.d. draws from the evaluation distribution, and every
  candidate is scored on the *same* draw (paired).

  If candidates are scored on independent draws the paired interval is still
  valid (the differences are then simply independent) but it loses its
  sharpness -- use two independent intervals instead.
* `select`: the table columns must be exchangeable draws from that
  distribution; the with-replacement sampler inside `select` assumes exactly
  that.
* `certify`: the `n` shared initial states are i.i.d. and were *not* chosen
  because of the observed outcomes (no optional stopping on the certification
  data, no seed shopping).
* Joint level: `select` spends `alpha / K` (marg) or `alpha / (K(K-1)/2)`
  (pair) per sequence, `certify` spends `alpha / n_comparisons`.  The overall
  statement is only at level `alpha` if `n_comparisons` counts every comparison
  you make on that data.
* `certify` reports the regret against the *observed* candidate set.  It says
  nothing about candidates that were never evaluated.
* **Every rival must share initial states with the selected candidate.**
  A rival with no shared initial state contributes no paired data, so its
  difference `p_j - p_selected` is only bounded by the trivial interval
  `[-1, 1]`; that rival therefore makes the certification **undecidable**
  (`regretUpper = 1 > delta`).  Such rivals are reported in `missingRivals`
  and are *not* silently excluded from the bound.  If all rivals are in that
  state, `certify` raises `ValueError`.  To get a decidable verdict, give the
  selected candidate and the rivals at least a few hundred common initial
  states (the A5p / A9t reference tables have >= 500 per pair).

## Library use

```python
from paired_cs import load_table, run_selection, run_replications, certify, certify_many

tab = load_table("table.npz")
res = run_selection(tab.Y, tab.K, delta=0.05, method="S2_pair", seed=0, cap=200_000)
# 20 replications, optionally across processes (call from a script guarded by
# `if __name__ == "__main__":` -- the multiprocessing "spawn" start method
# re-imports the main module):
reps = run_replications(tab.Y, tab.K, delta=0.05, method="S2_pair",
                        seeds=range(20), cap=200_000, n_jobs=8)
out = certify(tab, "fm_mask_N20_s43/step5000", delta=0.05, alpha=0.05, n_comparisons=231)
```

`BettingCS` is exposed for building new statistics on top of the same
confidence sequence (for example the anytime coverage check in `tests/`).

## Tests

```bash
python -m pytest tests -q
```

The suite contains a sequential-coverage check of the confidence sequence
(T1), a bit-exact regression against the A4p selection runs (T2), a bit-exact
regression against the A5p paired certification (T3), a CLI smoke test (T4), and
the no-shared-initial-state check (T5).  T2/T3 read the tables shipped in `../data/` (`selection_pool.npz`,
`reference.npz`) and the recorded values in `../data/expected.json`; the full
20-replication T2 check runs with `RUN_SLOW=1`.  T5 certifies a synthetic 4-candidate table in which one
rival shares no initial state with the selected candidate: the verdict must be
`undecidable` with `regretUpper == 1.0` and that rival must appear in
`missingRivals`; after giving the pair shared states the verdict becomes
decidable again.
