#!/usr/bin/env python3
"""The dynamic-support audit: how much mass sits outside the language, per rule.

This is the historical S2 instrument audited in `docs/state.md`, and the bridge
between two results that do not meet on their own. `dm/eval/attribution.py`
established that **the static field
grid is already learned** -- 0.0008-0.0085 bits/drawing on symbols that cannot
occur at a position class. The free-running reports establish that generated
programs still fault. Neither says whether *dynamic* state -- an illegal closer,
a halt inside an open scope, an opener past its depth bound -- carries measurable
probability **before** exposure bias begins.

This measures exactly that, teacher-forced, on existing checkpoints, with no
training. It reports `q` and `-log2(1-q)` decomposed by the mutually exclusive
rules of `docs/state-freeze.md` §5 and stratified by state, at both mask levels.

Read it as a gate input and not as a claim: teacher-forced illegal mass is what a
*decoder* could recover. It is one input to the fail-closed S4 gate, together
with the three-cell sampler report and a complete Direction 2 C3 instrument.

    # the three venues the freeze names: L0 flat control, L1 repeat, structured
    python3 scripts/state_support.py \
        runs/synthetic_c2flat24000_byte_square_s0.pt \
        runs/synthetic_converged_byte_square_s0.pt \
        runs/composed_sconv24000_byte_square_s0.pt

CPU by default, no training, minutes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dm.eval.provenance import environment, identity, sha256_file, verify_corpus
from dm.eval.records import codec_for, is_planner, load
from dm.eval.reports import json_safe, staged_path
from dm.eval.state_support import (
    DYNAMIC_RULES,
    POLICY_RULES,
    STATIC_RULES,
    dynamic_support,
)
from dm.isa.state import CANONICAL, MASK_RULES, VM_SAFE
from dm.train import RUNS

#: Bumped when a change makes new support reports incomparable with old ones.
SCHEMA = 3


def read(path: Path, limit: int | None, device: str, batch_size: int,
         protocol_path: Path) -> dict:
    """One checkpoint's audit, on the val split it was actually scored on."""
    model, record = load(path)
    if is_planner(record):
        raise SystemExit(
            f"{path} is a planner record. Its stroke decoder is prompted with a "
            "six-byte summary that is not a program, so a language mask over its "
            "output needs the planner's own prefix contract first "
            "(dm/eval/redundancy.py). Out of scope for Direction 1."
        )
    # The record's own width, never today's table: a pre-v2 `token` checkpoint is
    # 269 symbols wide and `CODECS["token"]` is 274, and a wrong-width reshape does
    # not raise.
    codec = codec_for(record)
    verified = verify_corpus(record)
    config, val, policy = verified.config, verified.val, verified.policy
    programs = val[:limit] if limit else val
    reading = dynamic_support(model.to(device), programs, codec, policy,
                              device=device, max_len=config.max_len,
                              batch_size=batch_size)
    reading |= {
        "report_schema": SCHEMA,
        "name": record.get("name", path.stem),
        "checkpoint": str(path),
        "corpus": verified.fingerprint,
        "corpus_verified": True,
        "train_opcodes": list(verified.train_opcodes),
        "val_opcodes": list(verified.val_opcodes),
        "policy_source": "train+val, equal opcode sets asserted",
        "device": device,
        "limit": limit,
        "steps_done": record.get("steps_done"),
        "schema": record.get("schema"),
        "record_bits_per_drawing": (record.get("final") or {}).get("bits_per_drawing"),
    }
    source_paths = [
        Path(__file__), Path("dm/eval/provenance.py"),
        Path("dm/eval/state_support.py"), Path("dm/isa/state.py"),
        Path("dm/isa/codec.py"), Path("dm/models/transformer.py"),
        protocol_path,
    ]
    protocol_digest = sha256_file(protocol_path)
    reading["protocol_sha256"] = protocol_digest
    reading["provenance"] = environment(argv=sys.argv, device=device,
                                        source_paths=source_paths)
    reading["identity"] = identity(
        record_path=path.with_suffix(".json"), checkpoint_path=path,
        corpus_fingerprint=verified.fingerprint, policy_digest=policy.digest,
        cells=["teacher_forced"], seeds=[], n=len(programs),
        cap=config.max_len,
        sampler={"mode": "teacher_forced", "batch_size": batch_size,
                 "limit": limit}, protocol_digest=protocol_digest,
        source_paths=source_paths,
    )
    return reading


def table(readings: list[dict], level: str) -> None:
    names = [r["name"][-26:] for r in readings]
    print(f"\n### {level} mask\n")
    print(f"  {'':<30}" + "".join(f"{n:>28}" for n in names))

    def row(label: str, values, fmt: str = "{:>12.6f}") -> None:
        print(f"  {label:<30}" + "".join(f"{fmt.format(v):>28}" for v in values))

    cells = [r["levels"][level] for r in readings]
    row("bits/drawing (raw)", [r["bits_per_drawing"] for r in readings], "{:>12.2f}")
    row("recoverable -log2(1-q)", [c["renorm_bits_per_drawing"] for c in cells])
    row("  position-class rules", [c["static_bits_per_drawing"] for c in cells])
    row("  corpus-allowlist rule", [c["policy_bits_per_drawing"] for c in cells])
    row("  dynamic-state rules", [c["dynamic_bits_per_drawing"] for c in cells])
    row("q mean", [c["q_mean"] for c in cells])
    row("q p99", [c["q_p99"] for c in cells])
    row("q max", [c["q_max"] for c in cells])
    row("positions", [c["positions"] for c in cells], "{:>12d}")
    row("union residual", [c["union_residual_per_position"] for c in cells],
        "{:>12.2e}")
    print()
    for reason in MASK_RULES:
        tag = ("static " if reason in STATIC_RULES
               else "policy " if reason in POLICY_RULES else "dynamic")
        row(f"{tag} {reason.value}",
            [c["by_rule"][reason.value]["mass_per_position"] for c in cells])


def strata_table(readings: list[dict], level: str) -> None:
    print(f"\n### {level}: recoverable bits/drawing by state stratum\n")
    names = [r["name"][-26:] for r in readings]
    print(f"  {'':<30}" + "".join(f"{n:>28}" for n in names))
    strata = sorted({s for r in readings for s in r["levels"][level]["by_stratum"]})
    for stratum in strata:
        values = []
        for reading in readings:
            entry = reading["levels"][level]["by_stratum"].get(stratum)
            values.append(entry["renorm_bits_per_drawing"] if entry else float("nan"))
        print(f"  {stratum:<30}"
              + "".join(f"{v:>28.6f}" if not math.isnan(v) else f"{'--':>28}"
                      for v in values))
    print(f"\n  {'positions':<30}"
          + "".join(f"{r['levels'][level]['positions']:>28d}" for r in readings))
    print(f"\n### {level}: where the mass concentrates — worst positions\n")
    for reading in readings:
        print(f"  {reading['name']}")
        for row in reading["levels"][level]["worst_positions"][:6]:
            print(f"      {row['renorm_bits']:9.4f} bits  q={row['q']:.4f}  "
                  f"program {row['program']:>4} byte {row['byte']:>5}  "
                  f"{row['stratum']:<20} {row['dominant_rule']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", type=Path, nargs="+")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--protocol", type=Path,
                    default=Path("docs/state-protocol-v2.json"))
    ap.add_argument("--limit", type=int, default=None,
                    help="score only the first N val programs; for a smoke test, "
                         "never for a reading -- it changes the denominator")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing report for this exact setting")
    args = ap.parse_args()

    if not args.protocol.exists():
        raise SystemExit(f"missing protocol: {args.protocol}")
    protocol_digest = sha256_file(args.protocol)
    stem = "_".join(sorted(p.stem for p in args.checkpoints))[:110]
    suffix = f"n{args.limit}" if args.limit else "full"
    out = args.out or RUNS / (
        f"state_support_{stem}_{suffix}_pr{protocol_digest[:12]}.json"
    )
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} already exists; pass --overwrite to replace it")

    partial = {
        "report_schema": SCHEMA,
        "status": "running",
        "requested_checkpoints": [str(path) for path in args.checkpoints],
        "limit": args.limit,
        "device": args.device,
        "batch_size": args.batch_size,
        "readings": [],
    }
    write_report(out, partial)
    try:
        readings = []
        for path in args.checkpoints:
            readings.append(read(
                path, args.limit, args.device, args.batch_size, args.protocol
            ))
            write_report(out, {**partial, "readings": readings})
    except Exception as exc:
        write_report(out, {
            **partial,
            "status": "incomplete",
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "readings": readings,
        })
        raise

    print("\n## dynamic support audit — teacher-forced, no training\n")
    for reading in readings:
        cell = reading["levels"][CANONICAL]
        print(f"  {reading['name']:<44} {reading['codec']:<12} "
              f"n={reading['n']:<5} canonical corpus={reading['canonical_corpus']:.3f} "
              f"policy={reading['policy_digest']}")
        print(f"      {' '.join(reading['policy']['opcodes'])}")
        if cell["empty_support_positions"]:
            print(f"      NOTE {cell['empty_support_positions']} position(s) had no "
                  "legal symbol at all; excluded from the renormalisation total")

    digests = {r["corpus"].get("val") for r in readings}
    if len(digests) > 1:
        print("\n  val digests DIFFER — these columns describe different corpora "
              "and must not be differenced.")

    for level in (CANONICAL, VM_SAFE):
        table(readings, level)
    strata_table(readings, CANONICAL)

    print("\n  `q` is the mass on symbols the state forbids, at temperature 1,")
    print("  before `forbid` and before `top_k`. `-log2(1-q)` is what")
    print("  renormalising onto the legal set recovers; `-log2(q)` is never")
    print("  reported, because it grows as illegal mass shrinks.")
    print()
    print("  Three buckets. `static` is what a position class already implies and")
    print("  what `dm/eval/attribution.py` priced at 0.0008–0.0085 bits/drawing --")
    print("  the operand-domain rule belongs there, because 'this position is an XF")
    print("  operand' is the instruction's own layout. `policy` is the corpus's")
    print("  opcode allowlist, still position-class only and still invisible to")
    print("  attribution, which read the ISA's grid rather than the corpus's.")
    print("  `dynamic` is the scope stack: closers, depth and halt legality, and")
    print("  it is the only bucket direction 1 exists to price. Recoverable bits")
    print("  use exact stagewise renormalisation costs; the mass-share columns are")
    print("  descriptive only. The stage costs telescope to the union cost.")

    out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "report_schema": SCHEMA,
        "status": "complete",
        "shared_val_digest": len(digests) == 1,
        "static_rules": [r.value for r in STATIC_RULES],
        "policy_rules": [r.value for r in POLICY_RULES],
        "dynamic_rules": [r.value for r in DYNAMIC_RULES],
        "readings": readings,
    }
    write_report(out, report)
    print(f"\n  -> {out}")
    return 0


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = staged_path(path)
    staged.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False))
    staged.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
