"""Qualification Adapter: run one development cell, or decide the package.

Two subcommands, because the two halves cost very different things and must not
be able to run each other by accident.

    run     one full-budget development cell: train, reload strict, apply the
            generic guards, run the repaired stability probe, reconcile every
            identity, and write a qualification cell report.

    decide  read the package's cell reports, apply the frozen gate, and name the
            schedule that freezes a training protocol -- or stop.

**Two packages, one Adapter.** `--package f5b` is the schedule qualification that
ran and stopped at `unstable_feedback`; `--package f5c` is the single predeclared
correction that stop is allowed to buy -- one lever (the shared input norm's gain
calibration), one schedule, fresh seeds and a fresh corpus, three cells and no
retry. Which seeds, which corpus and which training-protocol condition are right
comes from `dm.eval.feedback_contract.PACKAGES`, so this file never decides it.

**What a qualification cell deliberately does not do.** It never builds a
Direction 2 case table, never scores a contrast and never computes `G`. That is
not an omission for cost: a schedule or a calibration chosen after looking at
relation outcomes would make the estimation pilot its own configuration-selection
data (`docs/directions.md` §7 invariant 13), and
`dm.eval.feedback_qualification.qualify_cell` refuses a cell carrying such a
field at *any* level rather than ignoring it.

Thin by contract (§7 invariant 15): every rule, threshold and comparison lives in
`dm.eval.feedback_qualification`, `dm.eval.feedback_evidence` and
`dm.eval.feedback_contract`. This file reads arguments, trains, measures and
formats output.

    PYTHONPATH=. .venv/bin/python scripts/feedback_qualify.py run \\
        --package f5c --seed 200 --replicate 1 --device mps \\
        --corpus runs/feedback_corpus_f5c_audited.json
    PYTHONPATH=. .venv/bin/python scripts/feedback_qualify.py decide \\
        --package f5c runs/feedback_f5c_*_cell.json \\
        --out runs/feedback_correction.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

import dm.train
from dm.data.dataset import ProgramDataset, loader
from dm.eval import feedback_contract as contract
from dm.eval.feedback import (
    DEVELOPMENT,
    generic_guards,
    stability,
    stability_subset,
)
from dm.eval.feedback_evidence import (
    assemble_cell,
    deterministic_execution,
    execution_environment,
    reconstruct,
    source_identity,
    verify_report,
)
from dm.eval.feedback_qualification import (
    qualify_cell,
    qualify_correction,
    qualify_schedules,
)
from dm.eval.provenance import canonical_digest, sha256_file
from dm.eval.reports import json_safe, staged_path
from dm.isa.codec import CODECS
from dm.train import TrainConfig, train


def _cell_settings(args: argparse.Namespace) -> tuple[dict, str]:
    """The frozen training configuration for this cell, and its tag.

    Both come from the contract. The Adapter's only contribution is which cell of
    which package was asked for, which is the whole of the choice a command line
    is allowed to make here.
    """
    if args.package == "f5c":
        settings = contract.training_config_correction(
            args.seed, args.replicate, codec=args.codec)
    else:
        if args.replicate != 1:
            raise SystemExit(
                "the F5b package runs each development seed once; a replicate "
                "other than 1 is not part of it"
            )
        settings = contract.training_config_development(
            args.schedule, args.seed, codec=args.codec)
    return settings, settings["tag"]


def _run(args: argparse.Namespace) -> int:
    package = contract.package(args.package)
    if args.seed not in package.seeds:
        print(f"seed {args.seed} is not a {args.package} seed "
              f"{list(package.seeds)}: refused", file=sys.stderr)
        return 1
    codec = CODECS[args.codec]
    started = time.time()
    determinism = (deterministic_execution() if args.deterministic
                   else {"requested": False,
                         "deterministic_algorithms":
                             bool(torch.are_deterministic_algorithms_enabled()),
                         "cudnn_deterministic": bool(
                             torch.backends.cudnn.deterministic),
                         "cudnn_benchmark": bool(torch.backends.cudnn.benchmark)})

    settings, tag = _cell_settings(args)
    tag = args.tag or tag
    settings["tag"] = tag
    schedule = settings["pass_schedule"]
    out = args.out or Path(f"runs/{tag}_cell.json")
    record_path = dm.train.RUNS / f"{tag}.json"
    if out.resolve() == record_path.resolve():
        raise ValueError(
            "cell report path aliases the training record; the record carries "
            "the whole trajectory and must not be overwritten"
        )

    cfg = TrainConfig(**settings, device=args.device,
                      artifact_provenance=DEVELOPMENT)
    _, val_programs = dm.train.build_data(cfg)
    record = train(cfg, verbose=True)
    checkpoint_path = dm.train.RUNS / f"{tag}.pt"
    if not record_path.exists() or not checkpoint_path.exists():
        raise RuntimeError(
            "training completed without both its record and checkpoint: incomplete"
        )

    model, reload_stage = reconstruct(checkpoint_path, device=args.device)

    encoded = sorted(len(codec.encode(program)) for program in val_programs)
    generation_max_new = max(1, min(
        cfg.max_len, int(cfg.gen_cap * encoded[int(0.99 * (len(encoded) - 1))])
    ))
    variates = torch.rand(
        len(val_programs), generation_max_new,
        generator=torch.Generator().manual_seed(contract.seed_for("variates")),
    )
    guards = generic_guards(
        model, val_programs, codec, device=args.device, max_len=cfg.max_len,
        batch_size=cfg.batch_size, generation_n=len(val_programs),
        generation_max_new=generation_max_new, variates=variates,
    )

    subset = stability_subset(
        val_programs, codec, size=contract.STABILITY_SUBSET_SIZE,
        deepest_pass=max(contract.STABILITY_PASSES),
    )
    rows = [val_programs[index] for index in subset["indices"]]
    _, inputs, targets = next(iter(loader(
        ProgramDataset(rows, codec, cfg.max_len),
        batch_size=len(rows), shuffle=False)))
    report = stability(
        model, inputs, targets, device=args.device,
        valid_halt_loss=guards["values"]["valid_halt_loss"],
        symbols_per_byte=codec.stride, subset=subset,
    )

    protocol_body = json.loads(contract.PROTOCOL_PATH.read_text())
    cell = assemble_cell(
        package_name=package.name,
        schedule=schedule,
        seed=args.seed,
        replicate=args.replicate,
        record=record,
        record_path=record_path,
        checkpoint_path=checkpoint_path,
        manifest_path=args.corpus,
        protocol_body=protocol_body,
        protocol_digest=contract.digest_of(protocol_body),
        reload=reload_stage,
        generic_guards=guards,
        stability_report=report,
        environment={**execution_environment(device=args.device),
                     "determinism": determinism},
    )
    verdict = qualify_cell(cell, package)

    body = {
        "schema": contract.QUALIFICATION_SCHEMA,
        "protocol": contract.PROTOCOL,
        "provenance": DEVELOPMENT,
        "decision_value": "schedule_qualification_only",
        "package": package.as_dict(),
        "cell": json_safe(cell),
        "verdict": json_safe(verdict),
        "elapsed_s": round(time.time() - started, 1),
    }
    # Over everything but itself, because a document cannot contain its own hash.
    # `decide` recomputes this from the payload it reads, which is the only
    # version of the check with any content.
    body["report_sha256"] = canonical_digest(body, "report_sha256")
    staged = staged_path(out)
    staged.write_text(json.dumps(body, indent=1, sort_keys=True) + "\n")
    staged.replace(out)
    print(json.dumps({"package": package.name, "schedule": schedule,
                      "seed": args.seed, "replicate": args.replicate,
                      "passed": verdict["passed"],
                      "failures": sorted(verdict["failures"]),
                      "checkpoint_sha256": verdict["checkpoint_sha256"],
                      "report": str(out)}, indent=1))
    return 0 if verdict["passed"] else 1


def _decide(args: argparse.Namespace) -> int:
    package = contract.package(args.package)
    # The source digest of the tree *this* process is running, recomputed once.
    # Cells trained under different code are not comparable, and a package is
    # exactly long enough for the code to change underneath it.
    current = source_identity()["combined"]
    cells: list[dict] = []
    envelopes: list[dict] = []
    for path in args.cells:
        try:
            verified = verify_report(path)
        except (KeyError, ValueError) as exc:
            print(f"{exc}", file=sys.stderr)
            return 1
        cell = verified["cell"]
        if cell.get("package") != package.name:
            print(f"{path} belongs to package {cell.get('package')!r}, not "
                  f"{package.name!r}: refused", file=sys.stderr)
            return 1
        cell["hashes"]["source"]["recomputed"] = current
        cells.append(cell)
        envelopes.append(verified["envelope"])

    if package.name == contract.CORRECTION_PACKAGE:
        result = qualify_correction(cells, package)
    else:
        by_schedule: dict[str, list[dict]] = {}
        for cell in cells:
            by_schedule.setdefault(cell["schedule"], []).append(cell)
        result = qualify_schedules(by_schedule, package)
    result["cell_reports"] = envelopes

    summary = {
        "package": package.name,
        "order": result["order"],
        "selection_rule": result["selection_rule"],
        # Per cell, and by clause name: "cells failed" is not a diagnosis, and a
        # reader deciding what to do next needs the clause rather than the count.
        "schedules": {
            name: {"eligible": entry["eligible"], "passed": entry["passed"],
                   "failures": sorted(entry["failures"]),
                   "cells": {f"s{cell['seed']}r{cell['replicate']}":
                             sorted(cell["failures"])
                             for cell in entry["cells"]}}
            for name, entry in result["schedules"].items()},
        "frozen_schedule": result["frozen_schedule"],
        "label": result["label"],
    }
    if "reproducibility" in result:
        summary["reproducibility"] = result["reproducibility"]
    print(json.dumps(summary, indent=1))
    if args.out is not None:
        staged = staged_path(args.out)
        staged.write_text(json.dumps(json_safe(result), indent=1,
                                     sort_keys=True) + "\n")
        staged.replace(args.out)
        print(f"\nqualification {args.out}  sha256 {sha256_file(args.out)}")
    if result["frozen_schedule"] is None:
        print(f"\nSTOP: {result['label']}", file=sys.stderr)
        if package.name == contract.CORRECTION_PACKAGE:
            print(contract.CORRECTION_GATE["terminal_rule"], file=sys.stderr)
        return 1
    print(f"\nfreeze the training protocol on {result['frozen_schedule']} "
          f"({result['selection_rule']})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="one full-budget development cell")
    run.add_argument("--package", default="f5b", choices=sorted(contract.PACKAGES))
    run.add_argument("--schedule", default=contract.DEFAULT_PASS_SCHEDULE,
                     choices=sorted(contract.PASS_SCHEDULES),
                     help="F5b only; the F5c package declares one schedule and "
                          "compares none")
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--replicate", type=int, default=1,
                     help="which run of this seed, 1-based. Only the F5c "
                          "package's repeated seed has a replicate above 1")
    run.add_argument("--codec", default="byte", choices=["byte", "bit"])
    run.add_argument("--device", default="mps")
    run.add_argument("--tag", default=None)
    run.add_argument("--corpus", type=Path, required=True,
                     help="the audited corpus manifest this cell must reconcile "
                          "against; the record's own fingerprint has to match it "
                          "exactly, which is the check F5b did not have")
    run.add_argument("--no-deterministic", dest="deterministic",
                     action="store_false",
                     help="do not ask torch for deterministic kernels. The F5c "
                          "package requires them and its environment clause "
                          "fails without them")
    run.add_argument("--out", type=Path, default=None)
    run.set_defaults(func=_run, deterministic=True)

    decide = sub.add_parser("decide", help="apply the frozen gate to the cells")
    decide.add_argument("--package", default="f5b",
                        choices=sorted(contract.PACKAGES))
    decide.add_argument("cells", nargs="+", type=Path)
    decide.add_argument("--out", type=Path, default=None)
    decide.set_defaults(func=_decide)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
