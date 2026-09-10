"""CR-5 controlled metadata refusal and deterministic timer witness."""
import copy
from contextlib import contextmanager

import pytest
from test_relation_resource_benchmark import rows

from scripts import relation_resource_benchmark as benchmark


@pytest.mark.parametrize("field", ["threads", "recipe", "work", "wall", "memory"])
def test_summary_refuses_control_drift(field):
    records = copy.deepcopy(rows())
    record = records[0]
    if field == "threads":
        record["child_environment"]["torch_num_threads"] += 1
    elif field == "recipe":
        for row in records:
            row["identity"]["trainer_config"]["adam_eps"] *= 2
    elif field == "work":
        record["cell_definition"]["work_count"] += 1
    elif field == "wall":
        record["wall_window"] = "wrong"
    else:
        record["memory_window"] = "wrong"
    with pytest.raises(ValueError):
        benchmark.summarize(records, expected_repeats=2)


def test_allocation_timer_encloses_probe_setup_and_teardown(monkeypatch):
    clock = [0.0]

    @contextmanager
    def phase(*args, **kwargs):
        clock[0] += 2
        yield
        clock[0] += 3

    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(benchmark.resources, "resource_phase", phase)
    monkeypatch.setattr(benchmark.resources, "_memory_bytes", lambda: (100, 100))
    monkeypatch.setattr(benchmark.resources, "_rss_bytes", lambda: 100)
    record = benchmark._run_cell("allocation", "endpoints")
    assert record["wall_seconds"] == 5.0
