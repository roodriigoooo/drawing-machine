"""Isolated CPU post-update scan cost; not total safety overhead or R5 qualification."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import dm.train_relation as training_module
from dm.eval.relation_evidence import optimizer_digest, rng_digest, state_digest
from scripts.relation_diagnostic_benchmark import _fixture, _summary
from scripts.relation_resource_benchmark import _training_config


def trajectory(config, training, start_rng, *, scan):
    training_module.restore_rng_state(start_rng)
    trainer = training_module.RelationTrainer(config, training)
    original = training_module._validate_updated_numerics
    scan_seconds = []
    step_seconds = []

    def measured(model, optimizer):
        start = time.perf_counter()
        original(model, optimizer)
        scan_seconds.append(time.perf_counter() - start)

    # Harness-only substitution. No production configuration can bypass safety.
    training_module._validate_updated_numerics = measured if scan else lambda model, optimizer: None
    try:
        for _ in range(config.steps):
            start = time.perf_counter()
            trainer.step()
            step_seconds.append(time.perf_counter() - start)
    finally:
        training_module._validate_updated_numerics = original
    # Both finite trajectories must pass the real validator outside timing too.
    original(trainer.model, trainer.optimizer)
    evidence = {
        "model": state_digest(trainer.model.state_dict()),
        "optimizer": optimizer_digest(trainer.optimizer, trainer.model),
        "rng": rng_digest(config.device, trainer.cursor.generator),
        "history": trainer.history,
        "accounting": asdict(trainer.accounting),
        "diagnostics": trainer.diagnostics(),
    }
    return {"step_seconds": step_seconds, "scan_seconds": scan_seconds}, evidence


def benchmark():
    training = _fixture()
    start_rng = training_module.capture_rng_state()
    cells = {}
    for arm in ("none", "span_affine_v1"):
        for budget in (24_000_000, 1):
            config = _training_config(arm, budget)
            pairs = []
            for repeat in range(6):
                order = (True, False) if repeat % 2 == 0 else (False, True)
                runs, identities = {}, {}
                for scan in order:
                    runs[str(scan)], identities[str(scan)] = trajectory(
                        config, training, start_rng, scan=scan)
                if identities["True"] != identities["False"]:
                    raise AssertionError("scan changed exact trajectory evidence")
                pairs.append({"repeat": repeat, "timing_warmup": repeat == 0,
                              "order": list(order), "runs": runs,
                              "exact_equality": True, "identities": identities["True"]})
            retained = pairs[1:]
            on = [sum(p["runs"]["True"]["step_seconds"]) for p in retained]
            off = [sum(p["runs"]["False"]["step_seconds"]) for p in retained]
            scans = [sum(p["runs"]["True"]["scan_seconds"]) for p in retained]
            cells[f"{arm}_{budget}"] = {
                "config": asdict(config), "pairs": pairs,
                "scan_seconds_per_seven_updates": _summary(scans),
                "paired_whole_update_delta_seconds": _summary([a - b for a, b in zip(on, off, strict=True)]),
                "paired_whole_update_ratio": _summary([a / b for a, b in zip(on, off, strict=True)]),
                "environment": training_module._environment(config),
            }
    paths = {*training_module.AUTHORITATIVE_R4_SOURCES,
             "scripts/relation_numerical_guard_benchmark.py",
             "scripts/relation_resource_benchmark.py", "scripts/relation_diagnostic_benchmark.py"}
    return {
        "schema": 1, "kind": "r4_numerical_guard_cost", "resource_qualification": False,
        "estimand": "post_update_scan_only_with_objective_and_gradient_guards_common",
        "device": "cpu", "telemetry": "off", "retained_pairs": 5,
        "discarded_paired_timing_warmups": 1, "lr_warmup_updates": 2,
        "training_program_fingerprint": training.program_fingerprint,
        "supervision_identity": training.supervision_identity,
        "case_ids": [case.case_id for case in training.cases], "cells": cells,
        "source_hashes": {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sorted(paths, key=str)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Reserve exclusive output before fixture/model work, retaining failures.
    with args.output.open("x") as handle:
        try:
            result = benchmark()
        except BaseException as exc:
            json.dump({"status": "failed", "error": repr(exc)}, handle)
            raise
        json.dump(result, handle, sort_keys=True, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
