#!/usr/bin/env python3
"""Build, inspect or score Direction 2's frozen natural-twin manifests.

Two modes, deliberately separated by a protocol freeze:

```bash
# C0b: build and inspect without any model weights
PYTHONPATH=. .venv/bin/python scripts/context_c3.py \
  runs/synthetic_c2flat24000_byte_square_s0.pt \
  --manifest runs/context_c3_synthetic_flat_v3.json --build-manifest-only

# C3: score, but only under a protocol that already names this manifest hash
PYTHONPATH=. .venv/bin/python scripts/context_c3.py \
  runs/synthetic_c2flat24000_byte_square_s0.pt \
  runs/synthetic_c2flat24000_byte_square_s1.pt \
  --manifest runs/context_c3_synthetic_flat_v3.json \
  --protocol docs/context-protocol-v1.json
```

The venue family follows the checkpoint's own corpus: a `synthetic` record
builds the in-support x-step twins **and** their exact quarter-turn rotation,
and a `composed` record builds the copy-2 shape twins.  The two co-primary
venues therefore live in two manifests over two corpora, and
`scripts/context_gate.py` is what applies the frozen multiplicity procedure
across them.  Construction never loads weights; scoring refuses to construct.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dm.eval.context import (
    BUILDER_SEED,
    CO_PRIMARY_VENUES,
    DEFAULT_CASES,
    MANIFEST_SCHEMA,
    cases_from_manifest,
    first_token_divergence,
    load_manifest,
    manifest_dict,
    score_cases,
    summarise_by_venue,
    validate_cases,
    write_manifest,
)
from dm.eval.context_cases import (
    CorpusStats,
    build_shape_cases,
    build_step_cases,
    census,
    rotate_cases,
)
from dm.eval.provenance import (
    environment,
    sha256_file,
    verify_corpus,
    verify_frozen_sources,
)
from dm.eval.records import codec_for, composed_val_scenes, load
from dm.eval.reports import json_safe, staged_path
from dm.isa.codec import ByteCodec

SCHEMA = 3
SOURCE_PATHS = (
    Path("scripts/context_c3.py"), Path("dm/eval/context.py"),
    Path("dm/eval/context_cases.py"), Path("dm/eval/recovery.py"),
    Path("dm/eval/provenance.py"), Path("dm/data/composed.py"),
    Path("dm/data/augment.py"), Path("dm/isa/state.py"), Path("dm/isa/codec.py"),
    Path("dm/models/transformer.py"),
)


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = staged_path(path)
    staged.write_text(json.dumps(json_safe(value), indent=2, sort_keys=True,
                                 allow_nan=False))
    staged.replace(path)


def _record(path: Path) -> dict:
    return json.loads(path.with_suffix(".json").read_text())


def _build(record: dict, verified, *, seed: int, max_cases: int) -> dict:
    """Every case this corpus supports, plus the census that explains the yield."""
    codec = ByteCodec()
    data = verified.config.data
    if data == "synthetic":
        stats = CorpusStats.from_programs(
            verified.train, label=f"{data}:{record.get('name', '?')}",
            split="train",
        )
        held_out = not (set(verified.val) & set(verified.train))
        cases, rejected = build_step_cases(
            verified.val, verified.policy, codec, stats, seed=seed,
            max_cases=max_cases, held_out=held_out,
        )
        rotated, rotation_rejected = rotate_cases(cases, verified.policy, codec,
                                                  stats)
        # Two axis diagnostics, because they fail differently. The exact
        # rotation preserves the relation byte for byte and moves the whole
        # program out of support; the in-place y venue moves only the two copies
        # the relation is about. Only the second can distinguish "the relation
        # does not transfer" from "the model has no distribution here".
        y_axis, y_rejected = build_step_cases(
            verified.val, verified.policy, codec, stats, seed=seed,
            max_cases=max_cases, held_out=held_out, axis="y",
        )
        for reason, count in rotation_rejected.items():
            rejected[reason] += count
        # Namespaced, because the y venue walks the same candidates and its
        # reason names would otherwise be summed into the primary's and make
        # both columns unreadable.
        for reason, count in y_rejected.items():
            rejected[f"yaxis:{reason}"] += count
        cases = cases + rotated + y_axis
    elif data == "composed":
        # The step support is meaningless here -- a composed orbit is a D4
        # image, not a translation -- so it is not counted rather than counted
        # into an empty table that could later be read as "no steps found".
        stats = CorpusStats.from_programs(
            verified.train, label=f"{data}:{record.get('name', '?')}",
            split="train", step_programs=(), step_split="none",
        )
        scenes = composed_val_scenes(record, verified.val)
        cases, rejected = build_shape_cases(
            scenes, verified.policy, codec, stats, seed=seed,
            max_cases=max_cases,
            held_out=not (set(verified.val) & set(verified.train)),
        )
    else:
        raise SystemExit(
            f"no Direction 2 venue is defined for a {data!r} corpus; the two "
            "co-primary venues are the flat synthetic step twins and the "
            "composed copy-2 shape twins"
        )
    validate_cases(cases, verified.val, verified.policy)
    report = census(cases, rejected, stats=stats, seed=seed,
                    requested=max_cases, sources=len(verified.val),
                    extra={"corpus_split_overlap":
                           len(set(verified.val) & set(verified.train))})
    return manifest_dict(cases=cases, census=report, corpus=verified.fingerprint,
                         policy=verified.policy, corpus_stats=stats.as_dict(),
                         seed=seed)


def _brief(value, limit: int = 10) -> str:
    """A distribution as a readable line: the head, then how much was elided."""
    if not isinstance(value, dict) or len(value) <= limit:
        return str(value)
    head = dict(list(value.items())[:limit])
    return f"{head} (+{len(value) - limit} more keys)"


def _print_balance(manifest: dict) -> None:
    report = manifest["census"]
    print(f"venues={report['venue_case_counts']} "
          f"cases={report['included_cases']}/{report['requested_cases']} "
          f"sources={report['source_programs']}")
    print(f"  rejected: {report['rejected']}")
    identity = report["identity"]
    print(f"  identity: held out={identity['held_out_from_train']} "
          f"pooled components={identity['components']} "
          f"(cross-venue overlap {identity['primary_donor_overlap']}, "
          f"biases nothing -- venues are scored apart)")
    for venue, entry in identity["by_venue"].items():
        print(f"    {venue}: n={entry['cases']} "
              f"components={entry['components']} "
              f"largest={entry['largest_component']} "
              f"primary/donor overlap={entry['primary_donor_overlap']} "
              f"max donor reuse={entry['max_cases_per_control_donor']}")
    print(f"  controls: targets={identity['nondegenerate_controls']} "
          f"blocks={identity['nondegenerate_block_controls']}")
    for name, value in report["matching"].items():
        print(f"  match {name}: {_brief(value)}")
    for name, value in report["balance"].items():
        print(f"  balance {name}: {_brief(value)}")


def _instrument_checks(manifest: dict, manifest_path: str, cases,
                       reports: list[dict], protocol: dict) -> dict:
    identity = manifest["census"]["identity"]
    venues = set(manifest["census"]["venues"])
    minimum = protocol.get("context_c3", {}).get("min_venue_yield", {})
    per_venue = {venue: sum(case.venue == venue for case in cases)
                 for venue in venues}
    return {
        "nondegenerate_control_pairs": (
            identity["nondegenerate_controls"] == len(cases)
            and all(report["summary_all"]["nondegenerate_control_rate"] == 1.0
                    for report in reports)
        ),
        "nondegenerate_block_controls": (
            identity["nondegenerate_block_controls"] == len(cases)
        ),
        "donor_dependence_accounted_for": all(
            summary["bootstrap_D_minus_control"].get("cluster_unit")
            == "connected_source_donor_component"
            for report in reports for summary in report["venues"].values()
        ),
        "held_out_cases": bool(identity["held_out_from_train"]),
        # Per venue, because that is the unit each bootstrap resamples. Two
        # venues sharing a source program biases neither of them.
        "source_donor_partition": all(
            entry["primary_donor_overlap"] == 0
            for entry in identity["by_venue"].values()
        ),
        # §4.4's marginal/support columns exist for every row and travelled
        # into the report, so the gate no longer has to fail closed on them.
        "support_and_frequency_controls": all(
            case.features.legal_first_bytes > 0 for case in cases
        ) and all(
            "coordinate_marginal_distance" in row["support"]
            for report in reports for row in report["rows"]
        ),
        "venue_yield_met": all(
            per_venue.get(venue, 0) >= count
            for venue, count in minimum.items() if venue in venues
        ),
        "declared_venues_present": venues == set(
            (protocol.get("context_c3", {}).get("manifests") or {})
            .get(manifest_path, {}).get("venues", venues)
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoints", type=Path, nargs="+")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--protocol", type=Path,
                    default=Path("docs/context-protocol-v1.json"))
    ap.add_argument(
        "--build-manifest-only", action="store_true",
        help="build cases without loading model weights; inspect the balance "
             "report and freeze the resulting hash in a protocol before "
             "running the scorer",
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--cases", type=int, default=DEFAULT_CASES)
    ap.add_argument("--seed", type=int, default=BUILDER_SEED)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--bootstrap-reps", type=int, default=2000)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.cases < 1 or args.batch_size < 1:
        ap.error("--cases and --batch-size must be positive")

    # Read records and rebuild the corpus before touching any model state, so
    # the case table cannot be a function of logits and the manifest records the
    # split fingerprint that was actually verified.
    first_record = _record(args.checkpoints[0])
    verified = verify_corpus(first_record)
    if verified.config.codec != "byte":
        raise SystemExit("Direction 2 is frozen to the absolute byte codec")

    if args.build_manifest_only:
        if args.manifest.exists() and not args.overwrite:
            raise SystemExit(
                f"{args.manifest} already exists; pass --overwrite to replace it"
            )
        manifest = _build(first_record, verified, seed=args.seed,
                          max_cases=args.cases)
        write_manifest(args.manifest, manifest)
        _print_balance(manifest)
        print(f"-> {args.manifest}")
        print(f"sha256={manifest['manifest_sha256']}")
        return 0

    if not args.protocol.exists():
        raise SystemExit(f"missing protocol: {args.protocol}")
    if not args.manifest.exists():
        raise SystemExit(
            f"missing prebuilt context manifest: {args.manifest}; run "
            "--build-manifest-only, inspect the balance report, then freeze "
            "its hash in the protocol before scoring"
        )
    manifest = load_manifest(args.manifest)
    if manifest.get("corpus") != verified.fingerprint:
        raise SystemExit("context manifest corpus does not match checkpoint record")
    if manifest.get("policy_digest") != verified.policy.digest:
        raise SystemExit("context manifest policy does not match checkpoint record")

    protocol = json.loads(args.protocol.read_text())
    if protocol.get("status") != "frozen":
        raise SystemExit("selected context protocol is not frozen")
    verify_frozen_sources(protocol, SOURCE_PATHS)
    frozen = protocol.get("context_c3", {})
    entry = (frozen.get("manifests") or {}).get(str(args.manifest))
    if not entry or entry.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise SystemExit(
            "context manifest path/hash is not frozen in the selected protocol"
        )
    expected = {Path(name).resolve() for name in entry.get("checkpoints", [])}
    provided = {path.resolve() for path in args.checkpoints}
    if not expected or provided != expected:
        raise SystemExit(
            "C3 checkpoints do not exactly match the protocol's frozen set"
        )
    cases = cases_from_manifest(manifest)
    validate_cases(cases, verified.val, verified.policy)

    out = args.out or Path(
        f"runs/context_c3_{verified.config.data}_byte_"
        f"{manifest['manifest_sha256'][:12]}.json"
    )
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} already exists; pass --overwrite to replace it")

    base = {
        "report_schema": SCHEMA,
        "status": "running",
        "venues": manifest["census"]["venues"],
        "co_primary_venues": [venue for venue in manifest["census"]["venues"]
                              if venue in CO_PRIMARY_VENUES],
        "manifest": str(args.manifest),
        "manifest_schema": MANIFEST_SCHEMA,
        "manifest_sha256": manifest["manifest_sha256"],
        "corpus": verified.fingerprint,
        "corpus_stats": manifest["corpus_stats"],
        "policy": verified.policy.as_dict(),
        "policy_digest": verified.policy.digest,
        "checkpoint_count": len(args.checkpoints),
        "device": args.device,
        "batch_size": args.batch_size,
        "bootstrap_reps": args.bootstrap_reps,
        "protocol": str(args.protocol),
        "protocol_sha256": sha256_file(args.protocol),
        "provenance": environment(argv=sys.argv, device=args.device,
                                  source_paths=[*SOURCE_PATHS, args.protocol,
                                                args.manifest]),
        "census": manifest["census"],
        "reports": [],
    }
    write(out, base)
    reports: list[dict] = []
    try:
        for checkpoint in args.checkpoints:
            record = _record(checkpoint)
            check = verify_corpus(record)
            if check.fingerprint != verified.fingerprint:
                raise ValueError(f"C3 checkpoint corpus differs: {checkpoint}")
            if check.policy.digest != verified.policy.digest:
                raise ValueError(f"C3 checkpoint policy differs: {checkpoint}")
            codec = codec_for(record)
            if not isinstance(codec, ByteCodec) and codec.name != "byte":
                raise ValueError(f"C3 checkpoint is not absolute byte: {checkpoint}")
            model, loaded = load(checkpoint)
            rows = score_cases(model, cases, codec, device=args.device,
                               max_len=check.config.max_len,
                               batch_size=args.batch_size)
            venues = summarise_by_venue(rows, seed=args.seed,
                                        reps=args.bootstrap_reps)
            reports.append({
                "name": loaded.get("name", checkpoint.stem),
                "model_seed": loaded.get("config", {}).get("seed"),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "record_sha256": sha256_file(checkpoint.with_suffix(".json")),
                "venues": venues,
                "summary_all": _pooled(venues, rows, args.seed,
                                       args.bootstrap_reps),
                "first_token": first_token_divergence(model, cases, codec,
                                                      device=args.device),
                "rows": rows,
            })
            write(out, {**base, "reports": reports})
    except Exception as exc:
        write(out, {**base, "status": "incomplete",
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                    "reports": reports})
        raise

    checks = _instrument_checks(manifest, str(args.manifest), cases, reports,
                                protocol)
    result = {
        **base,
        "status": "complete",
        "reports": reports,
        "gate_input": {
            "instrument_checks": checks,
            "instrument_complete": all(checks.values()),
            "note": (
                "Per-venue point estimates and component intervals only. The "
                "frozen SESOI, sign agreement across model seeds and the "
                "two-venue multiplicity correction are applied by "
                "scripts/context_gate.py over both co-primary reports."
            ),
        },
    }
    write(out, result)
    print(f"C3 cases={len(cases)} checkpoints={len(reports)}")
    for report in reports:
        for venue, summary in report["venues"].items():
            print(f"  {report['name']} {venue}: "
                  f"n={summary['n_cases']} "
                  f"D={summary['D_bits_per_byte_mean']:.6f} "
                  f"Delta={summary['D_minus_control_bits_per_byte_mean']:.6f} "
                  f"CI={summary['bootstrap_D_minus_control']['ci95']}")
    print(f"  -> {out}")
    return 0


def _pooled(venues: dict, rows: list[dict], seed: int, reps: int) -> dict:
    """A descriptive across-venue line, never a co-primary result.

    Kept because a reader wants one number for the run; labelled here so it can
    never be mistaken for the frozen endpoint, which is per venue.
    """
    from dm.eval.context import summarise_scores

    summary = summarise_scores(rows, seed=seed, reps=reps)
    summary["descriptive_only"] = True
    summary["venue_case_counts"] = {venue: value["n_cases"]
                                    for venue, value in venues.items()}
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
