"""Engineering benchmark measuring shared-trunk component diagnostic overhead.

Fixes tiny CPU fixtures, seven-step schedules, and identical initial states to
compare component diagnostics on (events {1, 2, 3, 7}) versus off (events ()).
Distinguishes setup, metadata planning, and train-step execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from dm.data import relation
from dm.relation.wiring_spec import MODEL_CONFIG
from dm.relation.wiring_spec import fixture as _fixture
from dm.train_relation import (
    RelationTrainConfig,
    RelationTrainer,
    TrainingCorpus,
    capture_rng_state,
    restore_rng_state,
)


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def _run_single_trajectory(
    config: RelationTrainConfig,
    training: TrainingCorpus,
    start_rng: dict[str, object] | None = None,
) -> tuple[float, list[float], list[float], RelationTrainer]:
    """Run one full trajectory, returning setup time, per-step plan times, per-step step times, and trainer."""
    if start_rng is not None:
        restore_rng_state(start_rng)

    t0 = time.perf_counter()
    trainer = RelationTrainer(config, training)
    t_setup = time.perf_counter() - t0

    plan_times: list[float] = []
    step_times: list[float] = []

    orig_batch = trainer.planner.batch

    def timed_batch(indices):
        tp0 = time.perf_counter()
        plan = orig_batch(indices)
        plan_times.append(time.perf_counter() - tp0)
        return plan

    trainer.planner.batch = timed_batch

    for _ in range(config.steps):
        ts0 = time.perf_counter()
        trainer.step()
        ts1 = time.perf_counter()
        step_times.append(ts1 - ts0)

    trainer._endpoint_rng = capture_rng_state(trainer.cursor.generator)

    return t_setup, plan_times, step_times, trainer


def _verify_numerical_transparency(
    on_trainer: RelationTrainer,
    off_trainer: RelationTrainer,
) -> dict[str, object]:
    """Prove numerical transparency: parameters, optimizer, and RNG states are identical."""
    assert on_trainer.history == off_trainer.history
    assert on_trainer.accounting == off_trainer.accounting

    max_p_diff = 0.0
    for p_on, p_off in zip(on_trainer.model.parameters(), off_trainer.model.parameters(), strict=True):
        diff = float((p_on.detach() - p_off.detach()).abs().max())
        max_p_diff = max(max_p_diff, diff)
        assert torch.allclose(p_on, p_off, atol=1e-6, rtol=1e-6)

    opt_on = on_trainer.optimizer.state_dict()
    opt_off = off_trainer.optimizer.state_dict()
    assert opt_on["param_groups"] == opt_off["param_groups"]
    assert set(opt_on["state"]) == set(opt_off["state"])

    max_moment_diff = 0.0
    moments_exact = True
    for k in opt_on["state"]:
        s_on, s_off = opt_on["state"][k], opt_off["state"][k]
        if set(s_on) != set(s_off):
            moments_exact = False
            break
        for sub_k in s_on:
            v_on, v_off = s_on[sub_k], s_off[sub_k]
            if isinstance(v_on, torch.Tensor):
                diff = float((v_on.detach() - v_off.detach()).abs().max())
                max_moment_diff = max(max_moment_diff, diff)
                if not torch.equal(v_on, v_off):
                    moments_exact = False
            elif v_on != v_off:
                moments_exact = False

    rng_on = getattr(on_trainer, "_endpoint_rng", None) or capture_rng_state(on_trainer.cursor.generator)
    rng_off = getattr(off_trainer, "_endpoint_rng", None) or capture_rng_state(off_trainer.cursor.generator)
    rng_exact = (
        rng_on["python"] == rng_off["python"]
        and rng_on["numpy"][0] == rng_off["numpy"][0]
        and np.array_equal(rng_on["numpy"][1], rng_off["numpy"][1])
        and rng_on["numpy"][2:] == rng_off["numpy"][2:]
        and torch.equal(rng_on["torch_cpu"], rng_off["torch_cpu"])
        and (
            rng_on["data_generator"] is None
            if rng_off["data_generator"] is None
            else (
                rng_off["data_generator"] is not None
                and torch.equal(rng_on["data_generator"], rng_off["data_generator"])
            )
        )
    )

    return {
        "history_exact": bool(on_trainer.history == off_trainer.history),
        "parameters_exact": bool(max_p_diff == 0.0),
        "max_parameter_discrepancy": max_p_diff,
        "optimizer_groups_exact": bool(opt_on["param_groups"] == opt_off["param_groups"]),
        "optimizer_moments_exact": bool(moments_exact),
        "max_moment_discrepancy": max_moment_diff,
        "rng_exact": bool(rng_exact),
    }


def benchmark(repeats: int = 9) -> dict[str, object]:
    training = _fixture()
    conditions = [
        {"arm": "none", "geometry": "unsplit", "budget": 24_000_000},
        {"arm": "none", "geometry": "split", "budget": 1},
        {"arm": "span_affine_v1", "geometry": "unsplit", "budget": 24_000_000},
        {"arm": "span_affine_v1", "geometry": "split", "budget": 1},
    ]

    results: dict[str, object] = {}

    for cond in conditions:
        arm = cond["arm"]
        geom = cond["geometry"]
        budget = cond["budget"]
        label = f"{arm}_{geom}"

        on_cfg = RelationTrainConfig(
            arm,
            {**MODEL_CONFIG, "relation_schema": arm},
            seed=31,
            batch_size=3,
            max_len=256,
            steps=7,
            warmup=2,
            attention_budget=budget,
            corpus_provenance="engineering",
            seed_namespace="engineering",
        )
        off_cfg = RelationTrainConfig(
            arm,
            {**MODEL_CONFIG, "relation_schema": arm},
            seed=31,
            batch_size=3,
            max_len=256,
            steps=7,
            warmup=2,
            attention_budget=budget,
            component_gradient_events=(),
            corpus_provenance="engineering",
            seed_namespace="engineering",
        )

        # Transparency check
        start_rng = capture_rng_state()
        _, _, _, ref_on = _run_single_trajectory(on_cfg, training, start_rng=start_rng)
        _, _, _, ref_off = _run_single_trajectory(off_cfg, training, start_rng=start_rng)
        transparency = _verify_numerical_transparency(ref_on, ref_off)
        assert transparency["history_exact"]
        assert transparency["parameters_exact"]
        assert transparency["optimizer_groups_exact"]
        assert transparency["optimizer_moments_exact"]
        assert transparency["rng_exact"]

        # Inspect step geometry, extra backward calls, buffer bytes from ref_on
        steps_info = []
        for step_idx in range(7):
            h_row = ref_on.history[step_idx]
            batch_ids = h_row["batch_ids"]
            event_rec = next(
                (r for r in ref_on.component_records if r["step"] == step_idx + 1),
                None,
            )
            t_width = max(len(training.cases[c].flat) for c in batch_ids)
            rows = max(1, min(len(batch_ids), budget // max(1, t_width**2)))
            m_chunks = (len(batch_ids) + rows - 1) // rows
            pos_chunks = 0
            for m in range(m_chunks):
                start = m * rows
                stop = min(start + rows, len(batch_ids))
                if sum(
                    sum(
                        1
                        for b in relation.boundaries(training.cases[c].flat)
                        if b < len(training.cases[c].flat)
                        and b not in relation.covered_boundaries(training.cases[c].actions)
                        and b in {a.boundary: a for a in training.cases[c].actions}
                    )
                    for c in batch_ids[start:stop]
                ) > 0:
                    pos_chunks += 1

            steps_info.append({
                "step": step_idx + 1,
                "is_event": event_rec is not None,
                "batch_ids": batch_ids,
                "batch_width": t_width,
                "rows_per_chunk": rows,
                "chunks": m_chunks,
                "positive_chunks": pos_chunks,
                "extra_backward_calls": 0 if event_rec is None else event_rec["extra_backward_calls"],
                "buffer_bytes": 0 if event_rec is None else event_rec["buffer_bytes"],
            })

        on_setups: list[float] = []
        off_setups: list[float] = []
        on_plan_times: list[list[float]] = [[] for _ in range(7)]
        off_plan_times: list[list[float]] = [[] for _ in range(7)]
        on_step_times: list[list[float]] = [[] for _ in range(7)]
        off_step_times: list[list[float]] = [[] for _ in range(7)]

        for repeat in range(repeats + 1):  # 1 discarded warmup, 9 measured
            order = ("on", "off") if repeat % 2 == 0 else ("off", "on")
            for mode in order:
                start_rng = capture_rng_state()
                cfg = on_cfg if mode == "on" else off_cfg
                t_setup, plan_t, step_t, _ = _run_single_trajectory(cfg, training, start_rng=start_rng)
                if repeat > 0:
                    if mode == "on":
                        on_setups.append(t_setup)
                        for s in range(7):
                            on_plan_times[s].append(plan_t[s])
                            on_step_times[s].append(step_t[s])
                    else:
                        off_setups.append(t_setup)
                        for s in range(7):
                            off_plan_times[s].append(plan_t[s])
                            off_step_times[s].append(step_t[s])

        # Step summaries
        per_step_summary = []
        for s in range(7):
            on_summary = _summary(on_step_times[s])
            off_summary = _summary(off_step_times[s])
            diffs = [on - off for on, off in zip(on_step_times[s], off_step_times[s], strict=True)]
            per_step_summary.append({
                "step_info": steps_info[s],
                "on_step_seconds": on_summary,
                "off_step_seconds": off_summary,
                "difference_seconds": _summary(diffs),
                "on_plan_seconds": _summary(on_plan_times[s]),
                "off_plan_seconds": _summary(off_plan_times[s]),
            })

        results[label] = {
            "arm": arm,
            "geometry": geom,
            "attention_budget": budget,
            "transparency": transparency,
            "setup_seconds": {
                "on": _summary(on_setups),
                "off": _summary(off_setups),
            },
            "steps": per_step_summary,
            "raw_measurements": {
                "on_setup_seconds": on_setups,
                "off_setup_seconds": off_setups,
                "on_plan_seconds": on_plan_times,
                "off_plan_seconds": off_plan_times,
                "on_step_seconds": on_step_times,
                "off_step_seconds": off_step_times,
            },
        }

    root = Path(__file__).resolve().parents[1]
    paths = (
        "dm/train_relation.py",
        "dm/relation/wiring_spec.py",
        "dm/models/relation.py",
        "dm/eval/relation_training_evidence.py",
        "scripts/relation_diagnostic_benchmark.py",
    )
    source_hashes = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in paths
    }

    return {
        "schema": 2,
        "kind": "r4_diagnostic_overhead_benchmark",
        "model_constructed": True,
        "resource_qualification": False,
        "fixture_identity": {
            "venue": "venue1",
            "seed": 41,
            "motif_pool_seed": 40,
            "cases": 8,
            "negative_row_index": 1,
            "training_program_fingerprint": training.program_fingerprint,
        },
        "repeats": repeats,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "threads": torch.get_num_threads(),
            "mps_available": torch.backends.mps.is_available(),
            "cuda_available": torch.cuda.is_available(),
        },
        "source_hashes": source_hashes,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="destination engineering JSON; refuses overwrite",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=9,
        help="number of timed repeats after one warmup (default: 9)",
    )
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error(f"output already exists: {args.output}")

    report = benchmark(repeats=args.repeats)
    encoded = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as f:
            f.write(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
