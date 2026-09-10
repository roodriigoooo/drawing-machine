"""F-2 stored traces use the live ledger protocol."""
import pytest
from test_relation_resource_benchmark import rows

from scripts import relation_resource_benchmark as benchmark


@pytest.mark.parametrize("fault", ["foreign", "replay", "overlap", "outside", "incomplete", "count", "samples", "root"])
def test_stored_topology_refuses(fault):
    records = rows()
    record = next(r for r in records if r["cell"] == "train_update" and r["probe"] == "endpoints")
    trace = record["phase_observations"]
    if fault == "foreign":
        trace[0]["parent_scope_id"] = 999999
    elif fault == "replay":
        trace[1]["scope_id"] = trace[0]["scope_id"]
    elif fault == "overlap":
        trace[1]["work_start_seconds"] = 0.5
        trace[1]["work_end_seconds"] = 1.5
    elif fault == "outside":
        trace[0]["work_start_seconds"] = -1.0
        trace[0]["work_end_seconds"] = 0.0
    elif fault == "incomplete":
        trace.pop()
    elif fault == "count":
        record["phase_records"] = 0
    elif fault == "samples":
        record["phase_samples"] = 1
    else:
        record = next(r for r in records if r["cell"] == "metadata" and r["probe"] == "endpoints")
        record["phase_observations"][0]["parent_scope_id"] = 999999
    with pytest.raises(ValueError):
        benchmark.summarize(records)
