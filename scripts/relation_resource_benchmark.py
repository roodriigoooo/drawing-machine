"""Fresh-process paired resource evidence for R4 Point 5; never a qualification.

Answers the question the first instrumentation slice left open: what does
measuring cost, and what can a measurement actually witness?

Fresh child processes run each declared cell under paired probe arms from a
cold start, because a warm parent cannot witness a cold peak and an in-process
A/B inherits the other arm's allocator state. Every child reports a specifically
named work window, endpoint and interior resident size, and process-lifetime
high-water over its explicitly broader memory window.
`summarize` pairs those readings by cell and repeat and refuses an incomplete,
duplicated, non-finite or identity-inconsistent grid rather than averaging it.

The parent additionally records per-call reader costs, empty-scope costs, one
memory attribution point, and an in-process matched trainer comparison whose
purpose is the transparency proof a cross-process pairing cannot give: the same
trajectory, digests and history with telemetry off, on, and on with sampling.

Every number here is absolute seconds or bytes on this snapshot. None isolates
allocation ownership, is a resource qualification, sets a threshold, or is a
speed estimate for another shape.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import mmap
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

from dm.relation import resources
from dm.relation.wiring_spec import training_config
from dm.train_relation import (
    AUTHORITATIVE_R4_SOURCES,
    RelationTrainConfig,
    RelationTrainer,
    TrainingCorpus,
    _environment,
    capture_rng_state,
    restore_rng_state,
)
from scripts.relation_diagnostic_benchmark import (
    MODEL_CONFIG,
    _fixture,
    _summary,
    _verify_numerical_transparency,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()

#: Declared workloads. Their definitions are published verbatim in every report;
#: labels do not imply that timings may be subtracted from one another.
CELLS = ("allocation", "metadata", "train_update")
#: Ordered baseline-first probe arms of the primary paired grid.
PROBES = ("off", "endpoints")
#: The sampler arm is paired against instrumented endpoints, not against `off`.
SAMPLED_PROBES = ("endpoints", "sampled")
ALL_PROBES = ("off", "endpoints", "sampled")

ALLOCATION_BYTES = 32 * 2**20
SAMPLE_INTERVAL_SECONDS = 0.001
STEPS = 7
#: Scopes the trainer emits per update, innermost first.
PHASES = ("metadata", "train_update", "step")
CELL_DEFINITIONS = {
    "allocation": {
        "work": "allocate, touch, interior-read and release one anonymous 32 MiB mmap",
        "work_count": 1,
        "setup_outside_work": "none",
    },
    "metadata": {
        "work": "seven RelationBatchPlanner.batch calls using the trainer cursor",
        "work_count": STEPS,
        "setup_outside_work": "engineering fixture and RelationTrainer construction",
    },
    "train_update": {
        "work": "seven complete RelationTrainer.step calls, including metadata and train_update child scopes",
        "work_count": STEPS,
        "setup_outside_work": "engineering fixture and RelationTrainer construction",
    },
}

FIXTURE_IDENTITY = {
    "venue": "venue1",
    "seed": 41,
    "motif_pool_seed": 40,
    "cases": 8,
    "negative_row_index": 1,
    "steps": STEPS,
    "batch_size": 3,
    "max_len": 256,
    "allocation_bytes": ALLOCATION_BYTES,
    "sample_interval_seconds": SAMPLE_INTERVAL_SECONDS,
    "model_config": MODEL_CONFIG,
    "trainer_config": {
        "arm": "span_affine_v1",
        "lr": 3e-3,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "adam_betas": [0.9, 0.95],
        "adam_eps": 1e-8,
        "attention_budget": 24_000_000,
        "corpus_provenance": "engineering",
        "seed_namespace": "engineering",
    },
}


# ---------------------------------------------------------------------------
# Paired summary over fresh-process readings
# ---------------------------------------------------------------------------


def _positive_finite(value: object) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0)


def summarize(records, probes=PROBES, *, expected_repeats: int | None = None) -> dict[str, object]:
    """Pair raw child readings by cell and repeat, or refuse the whole grid.

    The grid must be complete, duplicate-free, positive, finite and drawn from
    one fixture identity. A partial grid is refused rather than averaged: an
    overhead computed from unpaired readings measures process spread, not
    instrumentation.
    """
    records = list(records)
    if not records:
        raise ValueError("no controlled resource readings to summarize")
    baseline, arm = probes
    environments: dict[str, dict[str, object]] = {}
    readings: dict[tuple[str, int, str], float] = {}
    repeats: set[int] = set()
    for record in records:
        if record["identity"] != FIXTURE_IDENTITY:
            raise ValueError("controlled readings disagree on declared fixture identity")
        cell = record["cell"]
        if cell not in CELLS or record.get("cell_definition") != CELL_DEFINITIONS[cell]:
            raise ValueError("controlled readings disagree on declared work definition")
        if record.get("wall_window") != "whole_instrumented_call_v3" or \
                record.get("memory_window") != "endpoint_probes_around_declared_work_body_v2":
            raise ValueError("controlled readings disagree on declared observation windows")
        environment = record.get("child_environment")
        template = _child_environment()
        if not isinstance(environment, dict) or set(environment) != set(template) or any(
                type(environment[key]) is not type(value) for key, value in template.items()):
            raise ValueError("controlled child environment is malformed")
        if any(environment[key] < 1 for key in ("torch_num_threads", "torch_num_interop_threads")):
            raise ValueError("controlled child thread counts are invalid")
        if environment["deterministic_algorithms_enabled"] != (cell != "allocation") or \
                environment["deterministic_algorithms_warn_only"] or \
                environment["torch_default_dtype"] != "torch.float32":
            raise ValueError("controlled child numerical policy disagrees with declared cell")
        if environment != environments.setdefault(cell, environment):
            raise ValueError("controlled child environments disagree within a cell")
        if record["probe"] not in probes:
            raise ValueError(f"unexpected probe arm {record['probe']!r}")
        observations = record.get("phase_observations")
        expected_phases = [] if record["probe"] == "off" else (
            list(PHASES) * STEPS if cell == "train_update" else [cell])
        if not isinstance(observations, list) or [row.get("phase") for row in observations] != expected_phases:
            raise ValueError("controlled phase observations disagree with declared work")
        ledger = resources.PhaseLedger({"step": ("metadata", "train_update")})
        for row in observations:
            observation = resources.PhaseResources(**row)
            observation.validate(require_scope=True)
            expected_interval = SAMPLE_INTERVAL_SECONDS if record["probe"] == "sampled" else None
            if not observation.completed or observation.sample_interval_seconds != expected_interval or (
                    expected_interval is None and observation.samples != 0):
                raise ValueError("controlled phase observation disagrees with declared probe")
            if cell == "train_update":
                ledger(observation)
            elif observation.parent_scope_id is not None:
                raise ValueError("standalone controlled observation requires a root scope")
        if cell == "train_update" and observations:
            ledger.unattributed_seconds("step", ("metadata", "train_update"))
            if ledger.totals()["step"].completed_records != STEPS:
                raise ValueError("controlled training trace requires seven complete steps")
        for name, expected in (("phase_records", len(observations)),
                               ("phase_samples", sum(row["samples"] for row in observations))):
            if name in record and (type(record[name]) is not int or record[name] != expected):
                raise ValueError("controlled phase counts contradict observations")
        if not _positive_finite(record["wall_seconds"]):
            raise ValueError(f"non-finite or non-positive wall time in cell {record['cell']!r}")
        if record["cell"] not in CELLS or type(record["repeat"]) is not int or \
                record["repeat"] < 0:
            raise ValueError("controlled record cell/repeat is malformed")
        key = (record["cell"], record["repeat"], record["probe"])
        if key in readings:
            raise ValueError(f"duplicate controlled reading for {key}")
        readings[key] = float(record["wall_seconds"])
        repeats.add(record["repeat"])
    cells = sorted({cell for cell, _, _ in readings})
    if set(cells) != set(CELLS):
        raise ValueError(f"controlled cells {cells} are not the declared {list(CELLS)}")
    if expected_repeats is not None:
        if type(expected_repeats) is not int or expected_repeats < 1:
            raise ValueError("expected repeat count must be a positive integer")
        if repeats != set(range(expected_repeats)):
            raise ValueError(
                f"controlled repeats {sorted(repeats)} are not the declared range "
                f"0..{expected_repeats - 1}")
    for cell in cells:
        for repeat in sorted(repeats):
            for probe in probes:
                if (cell, repeat, probe) not in readings:
                    raise ValueError(
                        f"incomplete paired grid: {cell!r} repeat {repeat} probe {probe!r}")

    summary: dict[str, object] = {}
    for cell in cells:
        order = sorted(repeats)
        off = [readings[(cell, repeat, baseline)] for repeat in order]
        on = [readings[(cell, repeat, arm)] for repeat in order]
        summary[cell] = {
            "estimand": "whole_instrumented_call_latency_v3",
            "baseline_probe": baseline,
            "arm_probe": arm,
            "repeats": len(order),
            "baseline_seconds": _summary(off),
            "arm_seconds": _summary(on),
            "paired_overhead_seconds": _summary(
                [a - b for a, b in zip(on, off, strict=True)]),
            "paired_on_off_ratio": _summary(
                [a / b for a, b in zip(on, off, strict=True)]),
            "raw_baseline_seconds": off,
            "raw_arm_seconds": on,
        }
    return summary


# ---------------------------------------------------------------------------
# One cell in one fresh process
# ---------------------------------------------------------------------------


def _training_config(arm: str, budget: int) -> RelationTrainConfig:
    return training_config(arm, budget, steps=STEPS, warmup=2)


def _child_environment() -> dict[str, object]:
    """Observed child settings; requested config is recorded separately."""
    return _environment(_training_config("span_affine_v1", 24_000_000))["observed"]


def _run_cell(cell: str, probe: str) -> dict[str, object]:
    """Execute one cell in this process and return its physical record."""
    records: list[resources.PhaseResources] = []
    observer = None if probe == "off" else records.append
    interval = SAMPLE_INTERVAL_SECONDS if probe == "sampled" else None
    requested, rss_during = 0, None

    if cell == "allocation":
        requested = ALLOCATION_BYTES
        rss_before, high_before = resources._memory_bytes()

        def workload() -> None:
            nonlocal rss_during
            block = mmap.mmap(-1, ALLOCATION_BYTES)
            for offset in range(0, ALLOCATION_BYTES, 4096):
                block[offset] = 1
            # Explicit interior observation, separate from endpoint probes.
            rss_during = resources._rss_bytes()
            block.close()

        start = time.perf_counter()
        with resources.resource_phase(observer, "allocation",
                                      sample_interval_seconds=interval):
            workload()
        wall = time.perf_counter() - start
    else:
        training = _fixture()
        trainer = RelationTrainer(_training_config("span_affine_v1", 24_000_000), training,
                                  resource_observer=observer,
                                  resource_sample_interval=interval)
        # Fixture/model construction is declared setup, outside the work window.
        rss_before, high_before = resources._memory_bytes()
        start = time.perf_counter()
        if cell == "metadata":
            # Seven planner calls; this is not a trainer-step phase reading.
            with resources.resource_phase(observer, "metadata",
                                          sample_interval_seconds=interval):
                for _ in range(STEPS):
                    trainer.planner.batch(trainer.cursor.next())
        else:
            for _ in range(STEPS):
                trainer.step()
        wall = time.perf_counter() - start
        rss_during = resources._rss_bytes()

    rss_after, high_after = resources._memory_bytes()
    high_water_rises = [record.process_high_water_rise_bytes for record in records
                        if record.process_high_water_rise_bytes is not None]
    return {
        "cell": cell,
        "probe": probe,
        "wall_seconds": wall,
        "identity": FIXTURE_IDENTITY,
        "child_environment": _child_environment(),
        "cell_definition": CELL_DEFINITIONS[cell],
        "wall_window": "whole_instrumented_call_v3",
        "memory_window": "endpoint_probes_around_declared_work_body_v2",
        "requested_bytes": requested,
        "rss_before_bytes": rss_before,
        "rss_during_bytes": rss_during,
        "rss_after_bytes": rss_after,
        "process_high_water_before_bytes": high_before,
        "process_high_water_after_bytes": high_after,
        "phase_records": len(records),
        "phase_samples": sum(record.samples for record in records),
        "process_high_water_rise_bytes": max(high_water_rises) if high_water_rises else None,
        "phase_observations": [record.__dict__ for record in records],
        "interior_exceeds_endpoints": bool(
            rss_during > max(rss_before, rss_after)),
        "high_water_rose_in_outer_window": bool(high_after > high_before),
        "attribution": resources.attribute_memory().as_dict(),
    }


def child(cell: str, probe: str, repeat: int = 0) -> dict[str, object]:
    """Run one cell in a fresh process; a warm parent cannot witness a cold peak."""
    if cell not in CELLS:
        raise ValueError(f"unknown resource cell {cell!r}")
    if probe not in ALL_PROBES:
        raise ValueError(f"unknown probe arm {probe!r}")
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--child", cell, "--probe", probe],
        capture_output=True, text=True, check=True, cwd=str(ROOT),
        env={**os.environ, "PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1"},
    )
    record = json.loads(completed.stdout)
    record["repeat"] = repeat
    return record


# ---------------------------------------------------------------------------
# In-process probe costs and the transparency proof
# ---------------------------------------------------------------------------


def _per_call_seconds(fn, calls: int) -> dict[str, object]:
    fn()  # one discarded warmup; a first read can fault in its own pages
    start = time.perf_counter()
    for _ in range(calls):
        fn()
    elapsed = time.perf_counter() - start
    return {"calls": calls, "total_seconds": elapsed, "seconds_per_call": elapsed / calls}


def _reader_costs(reader_calls: int, phase_calls: int) -> dict[str, object]:
    sink: list[resources.PhaseResources] = []

    def empty(observer, interval):
        with resources.resource_phase(observer, "probe", sample_interval_seconds=interval):
            pass

    costs: dict[str, object] = {
        "rss_bytes": _per_call_seconds(resources._rss_bytes, reader_calls),
        "memory_bytes": _per_call_seconds(resources._memory_bytes, reader_calls),
        "empty_phase_disabled": _per_call_seconds(lambda: empty(None, None), phase_calls),
        "empty_phase_observer": _per_call_seconds(lambda: empty(sink.append, None), phase_calls),
        "empty_phase_sampled": _per_call_seconds(
            lambda: empty(sink.append, SAMPLE_INTERVAL_SECONDS), max(1, phase_calls // 10)),
    }
    if sys.platform == "darwin":
        # Recorded to justify replacing the forking reader, not as a live path.
        costs["ps_reference_reader"] = _per_call_seconds(
            resources._ps_rss_bytes, max(1, reader_calls // 200))
    # A whole-heap walk is why attribution is a declared point, not a sampler.
    costs["attribute_memory"] = _per_call_seconds(resources.attribute_memory, 3)
    costs["records_collected"] = len(sink)
    return costs


def _trajectory(config: RelationTrainConfig, training: TrainingCorpus, probe: str,
                start_rng: dict[str, object]):
    """One trajectory under one telemetry arm, from an identical RNG state."""
    restore_rng_state(start_rng)
    ledger = None if probe == "off" else resources.PhaseLedger(
        {"step": ("metadata", "train_update")}
    )
    records: list[resources.PhaseResources] = []

    def observer(record: resources.PhaseResources) -> None:
        ledger(record)
        records.append(record)

    trainer = RelationTrainer(
        config, training,
        resource_observer=None if probe == "off" else observer,
        resource_sample_interval=SAMPLE_INTERVAL_SECONDS if probe == "sampled" else None,
    )
    step_times: list[float] = []
    for _ in range(STEPS):
        start = time.perf_counter()
        trainer.step()
        step_times.append(time.perf_counter() - start)
    trainer._endpoint_rng = capture_rng_state(trainer.cursor.generator)
    return step_times, trainer, ledger, records


def _phase_evidence(ledger: resources.PhaseLedger,
                    records: list[resources.PhaseResources]) -> dict[str, object]:
    totals = ledger.totals()
    # A ledger that disagrees with its own raw records is not evidence.
    for name, entry in totals.items():
        raw = [record for record in records if record.phase == name]
        assert entry.records == len(raw)
        assert entry.wall_seconds_total == sum(record.wall_seconds for record in raw)
        assert entry.peak_lower_bound_bytes == max(record.peak_lower_bound_bytes
                                                   for record in raw)
        assert entry.peak_lower_bound_bytes <= entry.peak_upper_bound_bytes
        assert entry.isolated_peak_bytes is None or \
            entry.isolated_peak_bytes <= entry.peak_lower_bound_bytes
    return {
        "emitted_phase_order": [record.phase for record in records[:len(PHASES)]],
        "records": len(records),
        "totals": ledger.as_dict(),
        "unattributed_seconds": ledger.unattributed_seconds("step", ["metadata", "train_update"]),
        "process_high_water_rise_records": {name: entry.isolated_peak_records
                                             for name, entry in totals.items()},
        "samples": {name: entry.samples for name, entry in totals.items()},
    }


def _in_process_transparency(repeats: int) -> dict[str, object]:
    """Matched arms proving telemetry changes no number the trainer produces."""
    training = _fixture()
    conditions = [
        {"arm": "none", "geometry": "unsplit", "budget": 24_000_000},
        {"arm": "none", "geometry": "split", "budget": 1},
        {"arm": "span_affine_v1", "geometry": "unsplit", "budget": 24_000_000},
        {"arm": "span_affine_v1", "geometry": "split", "budget": 1},
    ]
    results: dict[str, object] = {}
    for condition in conditions:
        arm, geometry, budget = condition["arm"], condition["geometry"], condition["budget"]
        config = _training_config(arm, budget)
        start_rng = capture_rng_state()
        reference = {name: _trajectory(config, training, name, start_rng)
                     for name in ALL_PROBES}
        transparency = {
            name: _verify_numerical_transparency(reference[name][1], reference["off"][1])
            for name in ("endpoints", "sampled")
        }
        for name, verdict in transparency.items():
            assert verdict["history_exact"], name
            assert verdict["parameters_exact"], name
            assert verdict["optimizer_groups_exact"], name
            assert verdict["optimizer_moments_exact"], name
            assert verdict["rng_exact"], name
        phases = {name: _phase_evidence(reference[name][2], reference[name][3])
                  for name in ("endpoints", "sampled")}
        for evidence in phases.values():
            assert evidence["emitted_phase_order"] == list(PHASES)
            assert evidence["records"] == STEPS * len(PHASES)

        raw: dict[str, list[list[float]]] = {name: [[] for _ in range(STEPS)]
                                             for name in ALL_PROBES}
        for repeat in range(repeats + 1):  # one discarded warmup, `repeats` measured
            order = ALL_PROBES if repeat % 2 == 0 else tuple(reversed(ALL_PROBES))
            for name in order:
                step_times, *_ = _trajectory(config, training, name, capture_rng_state())
                if repeat > 0:
                    for index in range(STEPS):
                        raw[name][index].append(step_times[index])

        steps = []
        for index in range(STEPS):
            entry: dict[str, object] = {"step": index + 1}
            for name in ALL_PROBES:
                entry[f"{name}_step_seconds"] = _summary(raw[name][index])
            for name in ("endpoints", "sampled"):
                paired = [on - off for on, off in
                          zip(raw[name][index], raw["off"][index], strict=True)]
                entry[f"{name}_overhead_seconds"] = _summary(paired)
                entry[f"{name}_overhead_ratio"] = (
                    statistics.median(raw[name][index]) / statistics.median(raw["off"][index]))
            steps.append(entry)

        results[f"{arm}_{geometry}"] = {
            "arm": arm,
            "geometry": geometry,
            "attention_budget": budget,
            "transparency": transparency,
            "phases": phases,
            "records_per_step": len(PHASES),
            "endpoint_probes_per_step": 2 * len(PHASES),
            "steps": steps,
            "raw_measurements": {f"{name}_step_seconds": raw[name] for name in ALL_PROBES},
        }
    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _source_hashes() -> dict[str, str]:
    paths = (
        "dm/relation/resources.py",
        "dm/relation/wiring_spec.py",
        "dm/train_relation.py",
        "scripts/relation_diagnostic_benchmark.py",
        "scripts/relation_resource_benchmark.py",
    )
    names = sorted(set(paths) | {str(path.relative_to(ROOT)) for path in AUTHORITATIVE_R4_SOURCES})
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}


def benchmark(args) -> dict[str, object]:
    if type(args.repeats) is not int or args.repeats < 1:
        raise ValueError("benchmark repeats must be a positive integer")
    # Counterbalance both cell and arm order. Fresh processes isolate raw
    # readings; counterbalancing prevents a fixed host-order effect becoming an arm.
    controlled = []
    for repeat in range(args.repeats):
        cells = CELLS if repeat % 2 == 0 else tuple(reversed(CELLS))
        probes = ALL_PROBES if repeat % 2 == 0 else tuple(reversed(ALL_PROBES))
        for cell in cells:
            for probe in probes:
                controlled.append(child(cell, probe, repeat))
    paired = [record for record in controlled if record["probe"] in PROBES]
    sampled = [record for record in controlled if record["probe"] in SAMPLED_PROBES]

    allocation = [record for record in controlled if record["cell"] == "allocation"]
    for record in allocation:
        # Keep raw, potentially non-comparable readings visible. Neither
        # endpoint nor a process high-water rise proves allocation ownership.
        record["interior_exceeds_endpoints"] = bool(
            record["rss_during_bytes"] > max(record["rss_before_bytes"],
                                              record["rss_after_bytes"]))
        record["high_water_rose_in_outer_window"] = bool(
            record["process_high_water_after_bytes"] >
            record["process_high_water_before_bytes"])

    return {
        "schema": 3,
        "kind": "r4_resource_probe_benchmark",
        "record_semantics": "controlled_whole_call_windows_v3",
        "resource_qualification": False,
        "cells": list(CELLS),
        "cell_definitions": CELL_DEFINITIONS,
        "probes": list(ALL_PROBES),
        "phases": list(PHASES),
        "steps_per_trajectory": STEPS,
        "sample_interval_seconds": SAMPLE_INTERVAL_SECONDS,
        "fixture_identity": FIXTURE_IDENTITY,
        "repeats": args.repeats,
        "controlled": {
            "endpoint_probe_overhead": summarize(paired, PROBES,
                                                  expected_repeats=args.repeats),
            "sampler_overhead": summarize(sampled, SAMPLED_PROBES,
                                           expected_repeats=args.repeats),
            "raw_records": controlled,
            "counterbalanced_order": "even=declared, odd=reversed",
        },
        "readers": _reader_costs(args.reader_calls, args.phase_calls),
        "in_process": _in_process_transparency(args.repeats),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "threads": torch.get_num_threads(),
            "mps_available": torch.backends.mps.is_available(),
            "cuda_available": torch.cuda.is_available(),
            "resident_size_reader": "libproc_proc_pidinfo" if sys.platform == "darwin"
            else "proc_self_statm",
        },
        "source_hashes": _source_hashes(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", choices=CELLS,
                        help="internal: run one cell in this process and print its record")
    parser.add_argument("--probe", choices=ALL_PROBES, default="off",
                        help="internal: probe arm for --child")
    parser.add_argument("--output", type=Path,
                        help="destination engineering JSON; refuses overwrite")
    parser.add_argument("--repeats", type=int, default=5,
                        help="controlled repeats per cell and arm (default: 5)")
    parser.add_argument("--reader-calls", type=int, default=20_000,
                        help="per-call reader timing calls (default: 20000)")
    parser.add_argument("--phase-calls", type=int, default=2_000,
                        help="empty-scope timing calls (default: 2000)")
    args = parser.parse_args()
    if args.child is not None:
        print(json.dumps(_run_cell(args.child, args.probe), allow_nan=False))
        return
    # Refuse the destination before any work: a benchmark that runs and then
    # discards its readings has spent the measurement it cannot repeat.
    if args.output is not None and args.output.exists():
        parser.error(f"output already exists: {args.output}")

    report = benchmark(args)
    encoded = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as f:
            f.write(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
