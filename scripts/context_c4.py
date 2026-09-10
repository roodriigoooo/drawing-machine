#!/usr/bin/env python3
"""Run Direction 2's C4 paired free-running completion diagnostic.

C3 measured the conditional distribution with the truth in hand. C4 asks the
same question of the model's own outputs: prompt at the instruction boundary
immediately before the target, decode both worlds under one shared block of
pre-generated uniforms, and ask whether each world's completions lean toward
its own compatible continuation.

```bash
PYTHONPATH=. .venv/bin/python scripts/context_c4.py \
  runs/synthetic_c2flat24000_byte_square_s0.pt \
  runs/synthetic_c2flat24000_byte_square_s1.pt \
  --manifest runs/context_c3_synthetic_flat_v4.json \
  --protocol docs/context-protocol-v4.json
```

The decode settings are read from the protocol, never from the command line:
a sampler chosen at the prompt is a sampler chosen after seeing the data.
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
    cases_from_manifest,
    load_manifest,
    validate_cases,
)
from dm.eval.context_c4 import (
    CompletionConfig,
    complete_cases,
    per_case,
    summarise_by_venue,
)
from dm.eval.provenance import (
    environment,
    sha256_file,
    verify_corpus,
    verify_frozen_sources,
)
from dm.eval.records import codec_for, load
from dm.eval.reports import json_safe, staged_path
from dm.isa.codec import ByteCodec

#: 5 adds the two generation controls and their contrasts. A schema-4 report has
#: no control column at all, and `scripts/context_gate.py` reports that as
#: `not_measured` rather than letting the absence read as a pass.
SCHEMA = 5
SOURCE_PATHS = (
    Path("scripts/context_c4.py"), Path("dm/eval/context.py"),
    Path("dm/eval/context_c4.py"), Path("dm/eval/context_cases.py"),
    Path("dm/eval/provenance.py"), Path("dm/isa/codec.py"),
    Path("dm/isa/state.py"), Path("dm/models/transformer.py"),
)


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = staged_path(path)
    staged.write_text(json.dumps(json_safe(value), indent=2, sort_keys=True,
                                 allow_nan=False))
    staged.replace(path)


def _record(path: Path) -> dict:
    return json.loads(path.with_suffix(".json").read_text())


def config_from(protocol: dict) -> CompletionConfig:
    """The frozen decode settings, or a refusal.

    Every field is required. A protocol that leaves one out has not frozen the
    sampler, and a sampler completed from this script's defaults would be a
    choice made after the manifest -- which is the one thing the freeze exists
    to prevent.
    """
    spec = (protocol.get("context_c3") or {}).get("c4") or {}
    sampler = spec.get("sampler")
    diagnostic = spec.get("diagnostic")
    if not sampler or not diagnostic:
        raise SystemExit(
            "the protocol does not freeze a C4 sampler and diagnostic; C4 "
            "cannot choose its own decode settings"
        )
    missing = [key for key in ("top_k", "temperature", "variate_seed")
               if key not in sampler]
    missing += [key for key in ("draws_per_world", "cap_symbols")
                if key not in diagnostic]
    if missing:
        raise SystemExit(f"protocol c4 block is missing {missing}")
    if diagnostic.get("structural_mask") != "off":
        raise SystemExit(
            "C4's primary result decodes raw; a masked cell is a named "
            "intervention and needs its own protocol entry"
        )
    # `draws_per_world` is either one number or a per-venue table with a
    # `default`. The table is how the frozen promotion rule expands the
    # co-primary venues without spending the same budget on the diagnostics.
    draws = diagnostic["draws_per_world"]
    by_venue: tuple[tuple[str, int], ...] = ()
    if isinstance(draws, dict):
        table = dict(draws)
        if "default" not in table:
            raise SystemExit(
                "a per-venue draws_per_world table must name a 'default', or a "
                "venue added later would decode at no declared depth"
            )
        default = int(table.pop("default"))
        by_venue = tuple(sorted((venue, int(value))
                                for venue, value in table.items()))
    else:
        default = int(draws)
    return CompletionConfig(
        draws=default,
        draws_by_venue=by_venue,
        cap=int(diagnostic["cap_symbols"]),
        top_k=sampler["top_k"],
        temperature=float(sampler["temperature"]),
        seed=int(sampler["variate_seed"]),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoints", type=Path, nargs="+")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--protocol", type=Path,
                    default=Path("docs/context-protocol-v5.json"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--bootstrap-reps", type=int, default=2000)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if not args.protocol.exists():
        raise SystemExit(f"missing protocol: {args.protocol}")
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("status") != "frozen":
        raise SystemExit("selected context protocol is not frozen")
    verify_frozen_sources(protocol, SOURCE_PATHS)
    config = config_from(protocol)

    first_record = _record(args.checkpoints[0])
    verified = verify_corpus(first_record)
    if verified.config.codec != "byte":
        raise SystemExit("Direction 2 is frozen to the absolute byte codec")

    manifest = load_manifest(args.manifest)
    frozen = protocol["context_c3"]
    entry = (frozen.get("manifests") or {}).get(str(args.manifest))
    if not entry or entry.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise SystemExit(
            "context manifest path/hash is not frozen in the selected protocol"
        )
    expected = {Path(name).resolve() for name in entry.get("checkpoints", [])}
    if {path.resolve() for path in args.checkpoints} != expected:
        raise SystemExit(
            "C4 checkpoints do not exactly match the protocol's frozen set"
        )
    if manifest.get("corpus") != verified.fingerprint:
        raise SystemExit("context manifest corpus does not match checkpoint record")
    cases = cases_from_manifest(manifest)
    validate_cases(cases, verified.val, verified.policy)

    # The protocol tag is in the filename because the manifest is *not* what
    # distinguishes two C4 runs. v5 scores v4's manifests unchanged -- same
    # cases, same hash -- and differs only in what it decodes from them. Keyed
    # on the manifest alone, a v5 run lands on the v4 report's path and
    # `--overwrite` silently destroys the superseded record that `context.md`
    # cites and `context_gate_v4.json` names by content hash.
    tag = str(protocol.get("protocol", args.protocol.stem)).rsplit("-", 1)[-1]
    out = args.out or Path(
        f"runs/context_c4_{verified.config.data}_byte_"
        f"{manifest['manifest_sha256'][:12]}_{tag}.json"
    )
    if out.exists() and not args.overwrite:
        raise SystemExit(
            f"{out} already exists; pass --overwrite to replace it. If this is a "
            "report from an earlier freeze, write a new one instead -- a scored "
            "report is a record and is never overwritten in place."
        )

    base = {
        "report_schema": SCHEMA,
        "status": "running",
        "stage": "C4",
        "venues": manifest["census"]["venues"],
        "co_primary_venues": [venue for venue in manifest["census"]["venues"]
                              if venue in CO_PRIMARY_VENUES],
        "manifest": str(args.manifest),
        "manifest_sha256": manifest["manifest_sha256"],
        "corpus": verified.fingerprint,
        "policy": verified.policy.as_dict(),
        "policy_digest": verified.policy.digest,
        "completion_config": config.as_dict(),
        "promotion_rule": (frozen.get("c4") or {}).get("promotion_rule"),
        "device": args.device,
        "bootstrap_reps": args.bootstrap_reps,
        "protocol": str(args.protocol),
        "protocol_sha256": sha256_file(args.protocol),
        "provenance": environment(argv=sys.argv, device=args.device,
                                  source_paths=[*SOURCE_PATHS, args.protocol,
                                                args.manifest]),
        "reports": [],
    }
    write(out, base)
    reports: list[dict] = []
    try:
        for checkpoint in args.checkpoints:
            record = _record(checkpoint)
            check = verify_corpus(record)
            if check.fingerprint != verified.fingerprint:
                raise ValueError(f"C4 checkpoint corpus differs: {checkpoint}")
            if check.policy.digest != verified.policy.digest:
                raise ValueError(f"C4 checkpoint policy differs: {checkpoint}")
            codec = codec_for(record)
            if not isinstance(codec, ByteCodec) and codec.name != "byte":
                raise ValueError(f"C4 checkpoint is not absolute byte: {checkpoint}")
            model, loaded = load(checkpoint)
            draws = complete_cases(model, cases, codec, verified.policy,
                                   config=config, device=args.device)
            aggregated = per_case(draws, cases)
            reports.append({
                "name": loaded.get("name", checkpoint.stem),
                "model_seed": loaded.get("config", {}).get("seed"),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "record_sha256": sha256_file(checkpoint.with_suffix(".json")),
                "venues": summarise_by_venue(aggregated, seed=BUILDER_SEED,
                                             reps=args.bootstrap_reps),
                "cases": aggregated,
                "draws": draws,
            })
            write(out, {**base, "reports": reports})
    except Exception as exc:
        write(out, {**base, "status": "incomplete",
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                    "reports": reports})
        raise

    write(out, {**base, "status": "complete", "reports": reports})
    print(f"C4 cases={len(cases)} checkpoints={len(reports)} "
          f"draws/world={config.as_dict()['draws_per_world']} cap={config.cap}")
    for report in reports:
        for venue, summary in report["venues"].items():
            print(f"  {report['name']} {venue}: n={summary['n_cases']} "
                  f"D_gen={summary['D_gen_mean']:+.4f} "
                  f"CI={[round(v, 4) for v in summary['bootstrap_D_gen']['ci95']]} "
                  f"half={summary['half_width']:.4f}")
            block = summary["bootstrap_D_gen_minus_block_control"]["ci95"]
            print(f"      control target={summary['D_gen_target_control_mean']:+.4f} "
                  f"block={summary['D_gen_block_control_mean']:+.4f} "
                  f"| Delta_gen={summary['D_gen_minus_block_control_mean']:+.4f} "
                  f"CI={[round(v, 4) for v in block]}")
            print(f"      reach={summary['reach_rate']:.3f} "
                  f"hit_own={summary['hit_own_rate']:.3f} "
                  f"hit_other={summary['hit_other_rate']:.3f} "
                  f"relation={summary['relation_consistent_rate']:.3f} "
                  f"canonical={summary['canonical_rate']:.3f} "
                  f"halt/fault/cap="
                  f"{summary['halt_rate']:.2f}/{summary['fault_rate']:.2f}/"
                  f"{summary['cap_rate']:.2f}")
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
