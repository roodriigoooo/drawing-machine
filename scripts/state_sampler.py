#!/usr/bin/env python3
"""Three decoder cells on one checkpoint: raw, VM-safe, canonical. No training.

This is the historical S3 instrument audited in `docs/state.md`. Its question is
deliberately narrow: **given the
same weights, what does constraining the decoder to the language do to
free-running reliability?** Nothing here is evidence that a model *learned*
state -- that is the G/C/E table of S5, gated behind this report and Direction 2's
C3, and the freeze says the two must not be collapsed into one "state helps"
number.

The three cells are the same sampler with one operation added:

    raw        the existing sampler, unchanged, PAD/BOS forbidden and the halt
               monitor deciding termination. Not "no rules" -- it still reports
               where the mask *would* have bitten.
    vm_safe    symbols whose execution would fault from this prefix are removed.
               Prefix-local by construction: a loop replays its body, so this cell
               still faults, and that residue is a result (freeze §1.2).
    canonical  the paper's language: the checkpoint's own opcode allowlist, exact
               operand domains, properly nested scopes, matching closers, the
               depth bounds, and no halt inside an open scope.

**Paired by construction.** All three cells of one draw read the *same*
pre-generated uniforms, one per row per step, consumed whether or not a row has
stopped. Seeding alone is not enough: a mask changes the distribution, so it
changes how `multinomial` walks its RNG stream, and two identically seeded cells
would diverge after the first masked position for a reason that is not the mask.

**A mask does not buy termination.** Forbidding `HALT` inside an open scope turns
an invalid early halt into a longer program and sometimes into a cap hit. A capped
row is *incomplete*, never upgraded to valid, and `close_cost` says how many bytes
it was from closing. A force-close decoder would be a fourth cell with its own
identity, and this script does not implement one.

    python3 scripts/state_sampler.py runs/composed_sconv24000_byte_square_s0.pt \
        --seeds 5 --n 256

CPU by default, no training.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dm.data.dataset import ProgramDataset
from dm.eval.metrics import length_stats
from dm.eval.provenance import environment, identity, sha256_file, verify_corpus
from dm.eval.quality import CLOUD_POINTS, clouds_of, quality_of, subsample
from dm.eval.records import codec_for, is_planner, load
from dm.eval.reports import json_safe, staged_path
from dm.eval.sampling import SamplingConfig, parse_temperature, parse_top_k
from dm.eval.state_analysis import paired_summary
from dm.eval.state_support import GenerationMass
from dm.isa.codec import BOS, PAD
from dm.isa.state import LEVELS, MASK_RULES, RAW, LanguagePolicy, LegacyStateMonitor, StateMonitor
from dm.train import RUNS, build_data
from dm.vm.interp import VM, FaultKind

#: Bumped when a change makes new state-sampler reports incomparable with old ones.
SCHEMA = 3

#: Fuel is a resource bound, not a structural fault: a perfectly canonical program
#: with nested counts can exhaust it (`docs/state-freeze.md` §1.2).
RESOURCE_FAULTS = {FaultKind.OUT_OF_FUEL.value}

#: Which real drawings stand as the geometry reference. Fixed and separate from the
#: draw seeds, so no draw of the model can move it -- `scripts/resample.py`'s rule
#: and the same constant, because the two reports have to be readable together.
REFERENCE_SEED = 1_000

#: The columns a paired difference is taken on. `valid_halt_rate` is the frozen
#: primary endpoint; the rest explain the mechanism and the trade-off, and are not
#: corrected for multiplicity (`docs/state-freeze.md` §3.5).
PAIRED = ("valid_halt_rate", "invalid_halt_rate", "cap_rate", "fuel_rate",
          "structurally_valid", "canonical_rate", "empty", "len_p50",
          "length_emd", "coverage", "mmd", "nna")


def corpus_of(config, cache: dict) -> tuple[list[bytes], list[bytes]]:
    """`build_data`, memoised across the checkpoints of one invocation.

    The composed corpus is 200,000 scenes and takes ~63 s to rebuild, which is
    longer than every draw of every cell put together. Two seeds of one arm share
    a corpus exactly -- `data_seed` is independent of the model seed by design --
    so rebuilding it per checkpoint would double the wall time of a paired report
    to produce the identical programs.
    """
    key = json.dumps({**asdict(config), "tier": int(config.tier)}, sort_keys=True,
                     default=str)
    if key not in cache:
        cache[key] = build_data(config)
    return cache[key]


def variates_for(rows: int, width: int, seed: int) -> torch.Tensor:
    """One uniform per row per step, from a generator of its own.

    Its own generator, not the global one: the cells must share these draws
    exactly, and anything else in the process that consumes global RNG would
    otherwise shift them.
    """
    return torch.rand(rows, width, generator=torch.Generator().manual_seed(seed))


def geometry_empty_policy(protocol: dict) -> str:
    """How a deterministic empty geometry draw affects report completion."""
    value = (protocol.get("sampler") or {}).get("empty_geometry_policy")
    if value is None and protocol.get("protocol_id") == "direction1-state-v2":
        return "incomplete"
    if value not in {"incomplete", "floor_failure"}:
        raise ValueError(
            "protocol sampler.empty_geometry_policy must be 'incomplete' or "
            "'floor_failure'"
        )
    return value


def one_cell(model, codec, policy: LanguagePolicy, mask_mode: str, *,
             rows: int, cap: int, variates: torch.Tensor, sampler: SamplingConfig,
             device: str, reference: list[bytes], classes: torch.Tensor | None,
             cloud_reference, cloud_points: int) -> dict:
    """One decoder cell, one draw."""
    monitor = (LegacyStateMonitor(codec, rows, policy)
               if mask_mode == RAW
               else StateMonitor(codec, rows, policy, mask_mode=mask_mode))
    mass = GenerationMass(monitor)
    ids = model.generate(
        rows, max_new=cap, temperature=sampler.temperature, top_k=sampler.top_k,
        device=device, monitor=monitor, forbid=(PAD, BOS), classes=classes,
        variates=variates, on_logits=mass,
    )
    programs = [codec.decode(row.tolist()) for row in ids.cpu()]
    verdicts = monitor.verdicts()
    vm = VM()
    traces = [vm.run(program) for program in programs]

    faults: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    first_exclusions: Counter[str] = Counter()
    symbol_first_exclusions: Counter[str] = Counter()
    row_records: list[dict] = []
    for trace, verdict in zip(traces, verdicts):
        faults.update(f.kind.value for f in trace.faults)
        reasons.update(v.reason.value for v in verdict.violations)

    for verdict in verdicts:
        first_exclusions.update(
            reason.value for reason in verdict.first_exclusions
            if reason is not None
        )
    for row in monitor.first_exclusions:
        symbol_first_exclusions.update(
            reason.value for reason in row if reason is not None
        )

    for index, (program, trace, verdict, cap_hit) in enumerate(
        zip(programs, traces, verdicts, ~monitor.done)
    ):
        causes = Counter(
            reason.value for reason in verdict.first_exclusions
            if reason is not None
        )
        row_records.append({
            "row": index,
            "program_hex": program.hex(),
            "length_bytes": len(program),
            "halted": bool(verdict.halted),
            "canonical": bool(verdict.canonical),
            "vm_valid": bool(trace.valid),
            "capped": bool(cap_hit),
            "faults": sorted({fault.kind.value for fault in trace.faults}),
            "violations": sorted({violation.reason.value
                                   for violation in verdict.violations}),
            "first_exclusions": dict(causes),
        })

    structural = [
        {f.kind.value for f in trace.faults} - RESOURCE_FAULTS for trace in traces
    ]
    halted = [v.halted for v in verdicts]
    canonical = [v.canonical for v in verdicts]
    # The frozen primary endpoint: halted at a boundary HALT, canonical under this
    # checkpoint's own policy, and structurally clean. All three, conjoined, over
    # the *requested* sample with no row dropped.
    valid_halt = [h and c and not s for h, c, s in zip(halted, canonical, structural)]
    capped = [not bool(done) for done in monitor.done]
    empty = [trace.is_empty for trace in traces]
    close_cost = [state.close_cost() for state, cap_hit in zip(monitor.states, capped)
                  if cap_hit]

    stats: dict = {
        "mask_mode": mask_mode,
        "n": rows,
        "valid_halt_rate": float(np.mean(valid_halt)),
        "invalid_halt_rate": float(np.mean([h and not v for h, v in zip(halted, valid_halt)])),
        "halted_rate": float(np.mean(halted)),
        "canonical_rate": float(np.mean(canonical)),
        "cap_rate": float(np.mean(capped)),
        "structurally_valid": float(np.mean([not s for s in structural])),
        "vm_valid": float(np.mean([t.valid for t in traces])),
        "fuel_rate": float(np.mean([FaultKind.OUT_OF_FUEL.value
                                    in {f.kind.value for f in t.faults}
                                    for t in traces])),
        "nonempty": float(np.mean([not value for value in empty])),
        # Structural and termination summaries remain complete even when the
        # optional geometry estimator is disabled or cannot score empty rows.
        "empty": float(np.mean(empty)),
        "faults": dict(faults.most_common()),
        "violations": dict(reasons.most_common()),
        "first_exclusions": dict(first_exclusions.most_common()),
        "symbol_first_exclusions": dict(symbol_first_exclusions.most_common()),
        "rows": row_records,
        "close_cost_mean": float(np.mean(close_cost)) if close_cost else 0.0,
        "close_cost_max": float(np.max(close_cost)) if close_cost else 0.0,
        "pressure": monitor.pressure(),
        "mass": mass.as_dict(),
    }
    stats |= length_stats(programs, reference, capped)
    if cloud_reference is not None:
        geometry = quality_of(programs, cloud_reference, cloud_points)
        # Empty decodes shrink one side of the comparison, so the estimator would
        # change with the sample. Reported unavailable rather than scored on the
        # survivors (`docs/traps.md`), and *only* the geometry half is withheld --
        # the structural columns above are the primary endpoint and stand.
        stats["geometry_status"] = ("complete" if not geometry["empty"]
                                   else "unavailable_empty")
        stats |= geometry
    if cloud_reference is not None and stats.get("geometry_status") != "complete":
        stats["geometry_complete"] = False
    elif cloud_reference is not None:
        stats["geometry_complete"] = True
    return stats


def summarise(draws: list[dict], columns) -> dict:
    out: dict = {}
    for column in columns:
        values = [d[column] for d in draws
                  if isinstance(d.get(column), (int, float))
                  and not (isinstance(d[column], float) and math.isnan(d[column]))]
        if not values:
            continue
        out[f"{column}_mean"] = st.mean(values)
        out[f"{column}_sd"] = st.stdev(values) if len(values) > 1 else float("nan")
    return out


def paired(cells: dict[str, list[dict]], treatment: str, control: str = RAW) -> dict:
    """Per-draw differences between two cells that shared their random variates.

    Draw seeds measure Monte Carlo noise on one checkpoint and nothing else. This
    interval is therefore **checkpoint-conditional**: it does not licence a claim
    across trained models, which needs the five paired model seeds of S5
    (`docs/state-freeze.md` §3.4).
    """
    out: dict = {}
    for column in PAIRED:
        pairs = [
            (t[column], c[column])
            for t, c in zip(cells[treatment], cells[control])
            if isinstance(t.get(column), (int, float))
            and isinstance(c.get(column), (int, float))
            and not (math.isnan(float(t[column])) or math.isnan(float(c[column])))
        ]
        if not pairs:
            continue
        deltas = [t - c for t, c in pairs]
        inference = paired_summary(deltas)
        out[column] = {
            **inference,
            "delta": inference["mean"],
            "draws": len(deltas),
        }
    return out


def report_path(name: str, n: int, seeds: int, quality: bool, points: int,
                device: str, sampler: SamplingConfig, policy: LanguagePolicy,
                cells: list[str], protocol_digest: str | None = None) -> Path:
    quality_key = f"p{points}_r{REFERENCE_SEED}" if quality else "noquality"
    device_key = str(device).replace(":", "-").replace("/", "-")
    base = SamplingConfig(sampler.top_k, sampler.temperature)
    cell_key = "-".join(cell.replace("_", "") for cell in sorted(set(cells)))
    protocol_key = f"_pr{protocol_digest[:12]}" if protocol_digest else ""
    return RUNS / (f"state_sampler_{name}_n{n}x{seeds}_{quality_key}_d{device_key}_"
                   f"{base.slug}_cells-{cell_key}_pol{policy.digest}"
                   f"{protocol_key}.json")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", type=Path, nargs="+")
    ap.add_argument("--seeds", type=int, default=5, help="draws per cell")
    ap.add_argument("--n", type=int, default=256, help="samples per draw")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--top-k", type=parse_top_k, default=40, metavar="K|none")
    ap.add_argument("--temperature", type=parse_temperature, default=1.0)
    ap.add_argument("--cells", nargs="+", default=list(LEVELS), choices=list(LEVELS))
    ap.add_argument("--no-quality", dest="quality", action="store_false",
                    help="skip the geometry half, which is the slow one")
    ap.add_argument("--cloud-points", type=int, default=CLOUD_POINTS)
    ap.add_argument("--protocol", type=Path,
                    default=Path("docs/state-protocol-v2.json"))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.seeds < 1 or args.n < 1:
        ap.error("--seeds and --n must both be positive")
    if RAW not in args.cells:
        ap.error("the raw cell is the control every difference is taken against")
    if not args.protocol.exists():
        raise SystemExit(f"missing protocol: {args.protocol}")
    sampler = SamplingConfig(args.top_k, args.temperature)
    protocol_payload = json.loads(args.protocol.read_text())
    empty_policy = geometry_empty_policy(protocol_payload)
    protocol_digest = sha256_file(args.protocol)
    floors: dict[tuple, list[dict]] = {}

    for path in args.checkpoints:
        model, record = load(path)
        if is_planner(record):
            raise SystemExit(
                f"{path} is a planner record; its stroke decoder is prompted with a "
                "six-byte summary that is not a program. Out of scope for Direction 1."
            )
        codec = codec_for(record)
        verified = verify_corpus(record)
        config, train, val, policy = (verified.config, verified.train,
                                      verified.val, verified.policy)
        out = report_path(record["name"], args.n, args.seeds, args.quality,
                          args.cloud_points, args.device, sampler, policy,
                          args.cells, protocol_digest)
        if out.exists() and not args.overwrite:
            raise SystemExit(f"{out} already exists; pass --overwrite to replace it")

        protocol_path = args.protocol
        source_paths = [
            Path(__file__),
            Path("dm/eval/provenance.py"), Path("dm/eval/state_support.py"),
            Path("dm/eval/state_analysis.py"), Path("dm/isa/state.py"),
            Path("dm/isa/codec.py"), Path("dm/models/transformer.py"),
            protocol_path,
        ]
        model = model.to(args.device)
        # The cap is the trainer's own rule. A different cap is a different
        # experiment, and `cap_rate` is meaningless without it.
        p99 = ProgramDataset(val, codec, config.max_len).length_stats()["p99"]
        cap = min(config.max_len, int(config.gen_cap * p99))
        n_classes = (record.get("model") or {}).get("n_classes", 0)
        classes = (torch.arange(args.n, device=args.device) % n_classes
                   if n_classes else None)

        cloud_reference = None
        if args.quality:
            cloud_reference, _ = clouds_of(
                subsample(val, args.n, REFERENCE_SEED), args.cloud_points)

        cells: dict[str, list[dict]] = {cell: [] for cell in args.cells}
        floor: list[dict] = []
        identity_payload = identity(
            record_path=path.with_suffix(".json"), checkpoint_path=path,
            corpus_fingerprint=verified.fingerprint, policy_digest=policy.digest,
            cells=args.cells, seeds=list(range(args.seeds)), n=args.n, cap=cap,
            sampler=sampler.as_dict(), protocol_digest=protocol_digest,
            source_paths=source_paths,
        )
        provenance = environment(argv=sys.argv, device=args.device,
                                source_paths=source_paths)

        def snapshot(
            status: str,
            error: dict | None = None,
            *,
            record=record,
            path=path,
            codec=codec,
            verified=verified,
            policy=policy,
            cap=cap,
            identity_payload=identity_payload,
            protocol_digest=protocol_digest,
            provenance=provenance,
            cells=cells,
            floor=floor,
        ) -> dict:
            return {
                "report_schema": SCHEMA,
                "status": status,
                "error": error,
                "name": record["name"],
                "checkpoint": str(path),
                "codec": codec.name,
                "corpus": verified.fingerprint,
                "corpus_verified": True,
                "policy": policy.as_dict(),
                "policy_digest": policy.digest,
                "policy_source": "train+val, equal opcode sets asserted",
                "train_opcodes": list(verified.train_opcodes),
                "val_opcodes": list(verified.val_opcodes),
                "sampler": sampler.as_dict(),
                "cell_samplers": {
                    cell: SamplingConfig(args.top_k, args.temperature, cell,
                                         policy.digest).as_dict()
                    for cell in args.cells
                },
                "device": args.device,
                "draw_seeds": list(range(args.seeds)),
                "seeds": args.seeds,
                "n": args.n,
                "cap": cap,
                "cap_rule": "min(max_len, gen_cap * val p99), from the record",
                "cloud_points": args.cloud_points if args.quality else None,
                "empty_geometry_policy": empty_policy,
                "variates": "pre-generated uniforms, one per row per step, shared by "
                            "every cell of a draw",
                "record_bits_per_drawing": (
                    record.get("final") or {}).get("bits_per_drawing"),
                "teacher_forced_note": "identical across cells: a mask is a decoder "
                                       "intervention and changes no weight",
                "identity": identity_payload,
                "protocol_sha256": protocol_digest,
                "provenance": provenance,
                "cells": {cell: {"draws": draws,
                                 **summarise(draws, PAIRED + ("halted_rate", "vm_valid"))}
                          for cell, draws in cells.items()},
                "paired_vs_raw": ({
                    cell: paired(cells, cell) for cell in args.cells
                    if cell != RAW
                } if all(cells[cell] for cell in args.cells) else {}),
                "floor": floor,
                "completion": {
                    "requested_draws": args.seeds * len(args.cells),
                    "completed_draws": sum(len(draws) for draws in cells.values()),
                    "geometry_missing": [
                        f"{cell}:{index}"
                        for cell, draws in cells.items()
                        for index, draw in enumerate(draws)
                        if args.quality and not draw.get("geometry_complete", False)
                    ],
                },
            }

        write(out, snapshot("running"))
        try:
            for seed in range(args.seeds):
                variates = variates_for(args.n, cap, seed)
                for cell in args.cells:
                    print(f"{record['name']}: draw {seed + 1}/{args.seeds}  cell {cell}",
                          flush=True)
                    cells[cell].append(one_cell(
                        model, codec, policy, cell, rows=args.n, cap=cap,
                        variates=variates, sampler=sampler, device=args.device,
                        reference=val, classes=classes,
                        cloud_reference=cloud_reference,
                        cloud_points=args.cloud_points,
                    ))
                    write(out, snapshot("running"))
        except Exception as exc:
            write(out, snapshot("incomplete", {
                "type": type(exc).__name__, "message": str(exc),
            }))
            raise

        if cloud_reference is not None:
            # Keyed by both corpus splits: the reference comes from validation,
            # while the replicate floor is sampled from training. Measuring
            # it again for the second seed would spend the same minutes to print
            # the same number -- or a different one, and invite the reader to
            # difference two estimates of one quantity.
            key = (
                verified.fingerprint.get("train"),
                verified.fingerprint.get("val"),
                args.n,
                args.cloud_points,
            )
            cached = floors.get(key) if all(key[:2]) else None
            computed = cached if cached is not None else [
                quality_of(
                    subsample(train, args.n, seed),
                    cloud_reference,
                    args.cloud_points,
                )
                for seed in range(args.seeds)
            ]
            floor[:] = computed
            if all(key[:2]):
                floors[key] = list(floor)

        summary = snapshot("complete")
        geometry_missing = summary["completion"]["geometry_missing"]
        summary["completion"]["geometry_floor_failed"] = bool(
            geometry_missing and empty_policy == "floor_failure"
        )
        summary["status"] = (
            "complete"
            if not geometry_missing or empty_policy == "floor_failure"
            else "incomplete"
        )
        write(out, summary)
        show(summary, cells, floor)
        print(f"\n  -> {out}")
    return 0


def write(path: Path, report: dict) -> None:
    staged = staged_path(path)
    staged.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False))
    staged.replace(path)


def show(summary: dict, cells: dict[str, list[dict]], floor: list[dict]) -> None:
    print(f"\n## {summary['name']}  ({summary['codec']}, policy "
          f"{summary['policy_digest']}: {' '.join(summary['policy']['opcodes'])})")
    print(f"{summary['seeds']} draws x {summary['n']} samples, cap {summary['cap']}, "
          f"top_k={summary['sampler']['top_k']}, "
          f"temperature={summary['sampler']['temperature']:g}, PAD/BOS forbidden\n")
    order = [c for c in LEVELS if c in cells]
    print(f"  {'column':<22}" + "".join(f"{c:>22}" for c in order) + f"{'floor':>20}")
    for column in PAIRED:
        row = f"  {column:<22}"
        for cell in order:
            values = [d[column] for d in cells[cell]
                      if isinstance(d.get(column), (int, float))
                      and not math.isnan(float(d[column]))]
            row += (f"{st.mean(values):>13.4f}"
                    f" ±{(st.stdev(values) if len(values) > 1 else 0.0):<7.4f}"
                    if values else f"{'--':>22}")
        real = [f[column] for f in floor if column in f
                and not math.isnan(float(f[column]))]
        row += (f"{st.mean(real):>13.4f} ±{st.stdev(real):<6.4f}"
                if len(real) > 1 else f"{'--':>20}")
        print(row)

    print("\n  paired differences against the raw cell (same variates per draw):")
    for cell, table in summary["paired_vs_raw"].items():
        print(f"    {cell}")
        for column in ("valid_halt_rate", "invalid_halt_rate", "cap_rate",
                       "length_emd", "coverage", "nna"):
            entry = table.get(column)
            if entry:
                print(f"      {column:<20} {entry['delta']:+.4f} "
                      f"± {entry['ci95_half_width']:.4f}  "
                      f"(p_t={entry['p_value']:.4f}, n={entry['draws']})")

    print("\n  mask pressure and illegal mass, scored against VM-safe and canonical support")
    print("  — on the prefixes each cell itself produced, which diverge after the")
    print("  first masked position, so this is not one quantity measured three ways:")
    for cell in order:
        first = cells[cell][0]
        mass_levels = first["mass"].get("levels")
        if mass_levels is None:
            level = first["mass"].get("scored_against", "canonical")
            mass_levels = {level: first["mass"]}
        pressure = first["pressure"]
        for level, mass in mass_levels.items():
            print(f"    {cell:<10} {level:<9} q mean {mass['q_mean']:.6f}  "
                  f"q p99 {mass['q_p99']:.6f}  q max {mass['q_max']:.6f}  "
                  f"recoverable {mass['renorm_bits_per_drawing']:.4f} bits/row")
        rules = {r.value: pressure["rule_positions"][r.value] for r in MASK_RULES
                 if pressure["rule_positions"][r.value]}
        print(f"               positions {pressure['positions_masked']}, "
              f"absorbed rows {pressure['absorbed_rows']}, "
              f"empty support {pressure['empty_support_events']}, rules {rules}")

    print("\n  A capped row is incomplete, never valid: `close_cost` is how many")
    print("  bytes it was from closing its scopes and halting. A mask does not buy")
    print("  termination, and this table is a decoder result -- not evidence that")
    print("  any model learned state.")
    for cell in order:
        costs = [d["close_cost_mean"] for d in cells[cell]]
        worst = [d["close_cost_max"] for d in cells[cell]]
        print(f"    {cell:<10} close_cost mean {st.mean(costs):6.2f}  "
              f"max {max(worst):6.0f}")


if __name__ == "__main__":
    raise SystemExit(main())
