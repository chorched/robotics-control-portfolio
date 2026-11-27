"""Command line interface: ``python -m paired_cs {select,certify} ...``.

Both subcommands print a single JSON object to stdout and exit 0 on success.
Errors print a JSON object with an ``error`` key to stderr and exit 2.
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np

from . import __version__
from .certify import certify
from .select import load_table, run_selection


def _sanitize(obj):
    """Replace non-finite floats by ``None`` so the payload is valid JSON."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if (math.isnan(x) or math.isinf(x)) else x
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    return obj


def _emit(payload) -> None:
    sys.stdout.write(json.dumps(_sanitize(payload), indent=2, sort_keys=True) + "\n")
    sys.stdout.flush()


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--delta", type=float, default=0.05,
                   help="tolerance in success-rate units (default 0.05 = 5 pp)")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="overall miscoverage level (default 0.05)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m paired_cs",
        description="Sequential paired selection with confidence sequences, and "
                    "paired certification of an already-made selection.",
    )
    ap.add_argument("--version", action="version", version=f"paired-cs {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    ps = sub.add_parser("select", help="run one sequential selection replication")
    ps.add_argument("--table", required=True, help="reference npz (candidate x initial state)")
    ps.add_argument("--key", default="Y", help="npz array holding the outcome matrix (default Y)")
    _add_common(ps)
    ps.add_argument("--method", choices=["pair", "marg", "S2_pair", "S2_marg"],
                    default="pair",
                    help="pair -> S2_pair (default), marg -> S2_marg")
    ps.add_argument("--seed", type=int, default=0,
                    help="seed of the with-replacement initial-state sampler")
    ps.add_argument("--cap", type=int, default=200000,
                    help="hard cap on total rollouts (default 200000)")

    pc = sub.add_parser("certify", help="paired regret certification of a selection")
    pc.add_argument("--reference", required=True, help="reference npz (candidate x initial state)")
    pc.add_argument("--key", default="Y", help="npz array holding the outcome matrix (default Y)")
    pc.add_argument("--selected", required=True,
                    help="selected candidate: index or label from the npz")
    _add_common(pc)
    pc.add_argument("--n-comparisons", type=int, required=True,
                    help="number of comparisons covered by alpha (m); each pair uses alpha/m")
    pc.add_argument("--decimals", type=int, default=6,
                    help="rounding of the reported bounds (default 6)")
    pc.add_argument("--no-round", action="store_true", help="report full precision bounds")
    return ap


def _cmd_select(args) -> dict:
    table = load_table(args.table, key=args.key)
    method = {"pair": "S2_pair", "marg": "S2_marg",
              "S2_pair": "S2_pair", "S2_marg": "S2_marg"}[args.method]
    res = run_selection(table.Y, table.K, args.delta, method, args.seed, args.cap,
                        alpha=args.alpha)
    chosen = res["chosen"]
    return {
        "command": "select",
        "table": str(args.table),
        "method": method,
        "methodAlias": args.method,
        "seed": int(args.seed),
        "delta": float(args.delta),
        "alpha": float(args.alpha),
        "cap": int(args.cap),
        "nCandidates": int(table.K),
        "nPool": int(table.n),
        "stopped": bool(res["stopped"]),
        "stopReason": str(res["stopWhy"]),
        "selected": None if chosen is None else int(chosen),
        "selectedLabel": None if chosen is None else table.labels[int(chosen)],
        "totalRollouts": int(res["totalRollouts"]),
        "rounds": int(res["rounds"]),
        "nLive": int(res["nLive"]),
        "chosenLower": res["chosenL"] if math.isfinite(res["chosenL"]) else None,
        "chosenUpper": res["chosenU"] if math.isfinite(res["chosenU"]) else None,
        "csEmpty": int(res["csEmpty"]),
        "csDegenerate": int(res["csDegenerate"]),
        "runtimeSec": float(res["runtimeSec"]),
    }


def _cmd_certify(args) -> dict:
    table = load_table(args.reference, key=args.key)
    decimals = None if args.no_round else int(args.decimals)
    res = certify(table, args.selected, delta=args.delta, alpha=args.alpha,
                  n_comparisons=args.n_comparisons, decimals=decimals)
    res.pop("pairs", None)
    out = {
        "command": "certify",
        "reference": str(args.reference),
        "delta": float(args.delta),
        "alpha": float(args.alpha),
        "nComparisons": int(args.n_comparisons),
        "alphaPerComparison": float(args.alpha) / int(args.n_comparisons),
        "selected": res["selected"],
        "selectedIndex": res["selectedIndex"],
        "regretLower": res["regretLower"],
        "regretUpper": res["regretUpper"],
        "verdict": res["verdict"],
        "decisiveRival": res["decisiveRival"],
        "decisiveRivalIndex": res["decisiveRivalIndex"],
        "nDecisive": res["nDecisive"],
        "nCandidates": res["nCandidates"],
        "missingRivals": res["missingRivals"],
        "nMissingRivals": res["nMissingRivals"],
    }
    return out


def main(argv=None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    try:
        if args.command == "select":
            payload = _cmd_select(args)
        else:
            payload = _cmd_certify(args)
    except (OSError, KeyError, ValueError) as exc:
        sys.stderr.write(json.dumps({"error": type(exc).__name__, "detail": str(exc)}) + "\n")
        return 2
    _emit(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
