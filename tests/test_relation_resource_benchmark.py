"""Fresh-process resource evidence must preserve pairing and fail closed."""

import argparse
import copy
import json
import sys

import pytest

from scripts import relation_resource_benchmark as benchmark


def observations(cell, probe):
    from test_relation_resources import _record
    if probe == "off":
        return []
    if cell != "train_update":
        return [_record(cell, 1.0, scope_id=1, start=0).__dict__]
    result = []
    for index in range(benchmark.STEPS):
        parent, start = 3 * index + 1, 4 * index
        result.extend([
            _record("metadata", 1.0, scope_id=parent + 1, parent_scope_id=parent, start=start).__dict__,
            _record("train_update", 1.0, scope_id=parent + 2, parent_scope_id=parent, start=start + 1).__dict__,
            _record("step", 3.0, scope_id=parent, start=start).__dict__,
        ])
    return result


def rows():
    return [{"cell": cell, "repeat": repeat, "probe": probe,
             "wall_seconds": 2.0 if probe == "off" else 3.0,
             "identity": copy.deepcopy(benchmark.FIXTURE_IDENTITY),
             "child_environment": {**benchmark._child_environment(),
                                   "deterministic_algorithms_enabled": cell != "allocation",
                                   "deterministic_algorithms_warn_only": False},
             "cell_definition": copy.deepcopy(benchmark.CELL_DEFINITIONS[cell]),
             "phase_observations": observations(cell, probe),
             "wall_window": "whole_instrumented_call_v3",
             "memory_window": "endpoint_probes_around_declared_work_body_v2"}
            for cell in benchmark.CELLS for repeat in range(2)
            for probe in ("off", "endpoints")]


def test_summary_uses_paired_raw_readings():
    result = benchmark.summarize(rows())
    for summary in result.values():
        assert summary["paired_overhead_seconds"] == {"median": 1, "min": 1, "max": 1}
        assert summary["paired_on_off_ratio"] == {"median": 1.5, "min": 1.5, "max": 1.5}


@pytest.mark.parametrize("mutation", ["identity", "missing", "duplicate", "zero", "nan", "whole_repeat"])
def test_summary_refuses_bad_controlled_records(mutation):
    record = copy.deepcopy(rows())
    if mutation == "identity":
        record[0]["identity"] = {"changed": 1}
    elif mutation == "missing":
        record.pop()
    elif mutation == "duplicate":
        record[-1]["repeat"] = 0
    elif mutation == "zero":
        record[0]["wall_seconds"] = 0
    elif mutation == "nan":
        record[0]["wall_seconds"] = float("nan")
    else:
        record = [row for row in record if row["repeat"] == 0]
    with pytest.raises(ValueError):
        benchmark.summarize(record, expected_repeats=2)


def test_transient_allocation_has_interior_witness():
    record = benchmark.child("allocation", "off")
    assert record["requested_bytes"] == 32 * 2**20
    assert record["process_high_water_after_bytes"] >= record["process_high_water_before_bytes"]
    assert record["wall_window"] == "whole_instrumented_call_v3"
    assert record["memory_window"] == "endpoint_probes_around_declared_work_body_v2"
    assert record["cell_definition"] == benchmark.CELL_DEFINITIONS["allocation"]


def test_publication_refuses_existing_path_before_work(tmp_path, monkeypatch):
    destination = tmp_path / "existing.json"
    destination.write_text("preserve")
    monkeypatch.setattr(benchmark.sys, "argv", ["benchmark", "--output", str(destination)])
    monkeypatch.setattr(benchmark, "benchmark", lambda _: pytest.fail("started benchmark"))
    with pytest.raises(SystemExit):
        benchmark.main()
    assert destination.read_text() == "preserve"


# ---------------------------------------------------------------------------
# Point 5 -- reader cost, cell records and the whole report
# ---------------------------------------------------------------------------


def test_reader_costs_place_every_probe_in_its_own_order():
    costs = benchmark._reader_costs(reader_calls=500, phase_calls=50)
    for entry in costs.values():
        if isinstance(entry, dict) and "seconds_per_call" in entry:
            assert entry["seconds_per_call"] > 0
    if sys.platform == "darwin":
        assert costs["ps_reference_reader"]["seconds_per_call"] > 0


@pytest.mark.parametrize("probe", ["off", "endpoints", "sampled"])
def test_every_cell_runs_in_a_fresh_process(probe):
    for cell in benchmark.CELLS:
        record = benchmark.child(cell, probe, repeat=3)
        assert (record["cell"], record["probe"], record["repeat"]) == (cell, probe, 3)
        assert record["wall_seconds"] > 0
        assert record["identity"] == benchmark.FIXTURE_IDENTITY
        assert record["attribution"]["unreadable_tensors"] == 0
        assert (record["phase_records"] > 0) is (probe != "off")
        assert len(record["phase_observations"]) == record["phase_records"]
        # No sampler exists off the sampled arm, so zero samples is exact there.
        # On it, only a phase far longer than the period is guaranteed a tick:
        # the ~2 ms allocation cell can legitimately finish before the first one.
        if probe != "sampled":
            assert record["phase_samples"] == 0
        elif cell == "train_update":
            assert record["phase_samples"] > 0


def test_unknown_cell_or_probe_is_refused():
    with pytest.raises(ValueError, match="unknown resource cell"):
        benchmark.child("nowhere", "off")
    with pytest.raises(ValueError, match="unknown probe arm"):
        benchmark.child("allocation", "guess")


def test_report_pairs_both_grids_and_keeps_its_raw_records():
    args = argparse.Namespace(repeats=1, reader_calls=200, phase_calls=20)
    report = benchmark.benchmark(args)
    assert report["schema"] == 3
    assert report["kind"] == "r4_resource_probe_benchmark"
    assert report["resource_qualification"] is False
    assert report["record_semantics"] == "controlled_whole_call_windows_v3"
    assert report["cell_definitions"] == benchmark.CELL_DEFINITIONS
    controlled = report["controlled"]
    assert set(controlled["endpoint_probe_overhead"]) == set(benchmark.CELLS)
    assert set(controlled["sampler_overhead"]) == set(benchmark.CELLS)
    assert len(controlled["raw_records"]) == len(benchmark.CELLS) * len(benchmark.ALL_PROBES)
    assert set(report["in_process"]) == {
        "none_unsplit", "none_split", "span_affine_v1_unsplit", "span_affine_v1_split",
    }
    for result in report["in_process"].values():
        for verdict in result["transparency"].values():
            assert verdict["history_exact"] is True
            assert verdict["parameters_exact"] is True
            assert verdict["optimizer_moments_exact"] is True
            assert verdict["rng_exact"] is True
        assert result["endpoint_probes_per_step"] == 2 * len(benchmark.PHASES)
        assert result["phases"]["sampled"]["samples"]["train_update"] > 0
        assert result["phases"]["endpoints"]["samples"]["train_update"] == 0
        assert len(result["steps"]) == report["steps_per_trajectory"]
    assert set(report["source_hashes"]) == {
        str(path.relative_to(benchmark.ROOT)) for path in benchmark.AUTHORITATIVE_R4_SOURCES
    } | {
        "dm/relation/resources.py", "dm/train_relation.py", "dm/relation/wiring_spec.py",
        "scripts/relation_diagnostic_benchmark.py",
        "scripts/relation_resource_benchmark.py",
    }
    json.dumps(report, allow_nan=False)


def test_allocation_record_retains_raw_window_observations():
    """A raw child record never turns process observations into owned memory."""
    record = benchmark.child("allocation", "endpoints")
    assert record["phase_samples"] == 0
    assert record["process_high_water_rise_bytes"] is None or \
        record["process_high_water_rise_bytes"] >= record["rss_during_bytes"]
    assert isinstance(record["interior_exceeds_endpoints"], bool)
    assert isinstance(record["high_water_rose_in_outer_window"], bool)
