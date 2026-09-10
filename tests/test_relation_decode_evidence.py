"""Strict non-model-bearing Point 4 engineering report acceptance."""
import copy
import json

import pytest

from dm.eval.relation_decode_evidence import (
    DecodeEvidenceRefused,
    build_report,
    publish_report,
    validate_report,
)


@pytest.fixture(scope="module")
def report():
    value = json.loads(json.dumps(build_report(), allow_nan=False))
    validate_report(value)
    return value


def test_report_roundtrip(report):
    validate_report(json.loads(json.dumps(report)))


def test_unmodified_json_control(report):
    validate_report(copy.deepcopy(report))


@pytest.mark.parametrize("mutation,reason", [
    (lambda r: r.update(weights=[]), "report: key set"),
    (lambda r: r.pop("seed"), "report: key set"),
    (lambda r: r["fixtures"][0].update(cache_max_abs=float("nan")), "nonfinite"),
    (lambda r: r["fixtures"][0]["observed_tokens"][0].__setitem__(0, 2), "raw output"),
    (lambda r: r["fixtures"][0]["events"][-1].update(returned_width=99), "completion shape"),
    (lambda r: r["fixtures"][0]["events"][0].update(mode="oracle_copy"), "request identity"),
    (lambda r: r["fixtures"][0].update(uniform_digest="0" * 64), "uniform field"),
    (lambda r: r["fixtures"][0]["events"][-1]["per_row"][0][0].__setitem__(1, 999),
     "does not reconcile"),
])
def test_report_refuses_mutation(report, mutation, reason):
    forged = copy.deepcopy(report)
    validate_report(forged)
    mutation(forged)
    with pytest.raises(DecodeEvidenceRefused, match=reason):
        validate_report(forged)


def _false_stop_status(report):
    fixture = report["fixtures"][0]
    for event in fixture["events"]:
        if event["kind"] == "row_stopped":
            event.update(cause="halt", byte_offset=999, at_horizon=False)
    fixture["events"][-1]["statuses"][0].update(
        cause="halt", byte_offset=999, at_horizon=False)


def _false_score_coordinates_support(report):
    score = next(
        event for fixture in report["fixtures"] for event in fixture["events"]
        if event["kind"] == "score_detail"
    )
    score.update(
        row=999,
        byte_offset=-1,
        endpoints=[[900, 901]],
        supports=[],
        normalization="invented",
    )


def _false_batch_calls(report):
    totals = report["fixtures"][0]["events"][-1]["totals"]
    for pair in totals:
        if pair[0] in ("sampler_calls", "head_batches"):
            pair[1] = 999


def _out_of_range_action(report):
    fixture = report["fixtures"][0]
    fixture["events"].insert(-1, {
        "row": 999,
        "byte_offset": -1,
        "action_source": "predicted",
        "candidate_count": 0,
        "action_key": None,
        "decision_score": 0.0,
        "executor_attempts": 0,
        "faults": [],
        "exit_reason": "invented",
        "admission_bytes": 0,
        "kind": "action_decision",
    })


@pytest.mark.parametrize("mutation", [
    _false_stop_status,
    _false_score_coordinates_support,
    _false_batch_calls,
    _out_of_range_action,
])
def test_report_refuses_coordinated_semantic_forgery(report, mutation):
    forged = json.loads(json.dumps(report))
    mutation(forged)
    with pytest.raises(DecodeEvidenceRefused):
        validate_report(forged)


def _fixture(report, name="predicted_copy_count2"):
    return next(f for f in report["fixtures"] if f["name"] == name)


def _work(fixture, values, calls=None):
    completed = fixture["events"][-1]
    for pairs in (completed["per_row"][0], completed["totals"]):
        for pair in pairs:
            if pair[0] in values:
                pair[1] = values[pair[0]]
    for name in ("expected_work", "observed_work"):
        fixture[name]["per_row"][0].update(values)
        fixture[name]["totals"].update(values | (calls or {}))
    for pair in completed["totals"]:
        if pair[0] in (calls or {}):
            pair[1] = calls[pair[0]]


def _fabricate_attempts(report, attempts=1_000_000):
    f = _fixture(report)
    action = next(e for e in f["events"] if e["kind"] == "action_decision")
    action.update(executor_attempts=attempts, faults=[["canvas", attempts - 1]])
    _work(f, {"executor_attempts": attempts, "predicted_search_attempts": attempts})


def _erase_oracle(report):
    f = _fixture(report, "oracle_copy_count2")
    f["events"] = [e for e in f["events"] if e["kind"] != "action_decision"]
    _work(f, {"literal_decisions": 7, "sampler_rows": 7, "uniform_coordinates_read": 7,
              "executor_attempts": 0, "oracle_executor_calls": 0, "copy_admissions": 0,
              "copy_admitted_bytes": 0, "copy_delivered_bytes": 0,
              "fed_literal_bytes": 6, "fed_copy_bytes": 0}, {"sampler_calls": 7})


def _reverse_queries(report):
    f = _fixture(report, "predicted_copy_literal")
    middle = f["events"][1:-2]
    pairs = [middle[i:i + 2] for i in range(0, len(middle), 2)]
    f["events"][1:-2] = [e for pair in reversed(pairs) for e in pair]


@pytest.mark.parametrize("mutate", [
    _fabricate_attempts,
    lambda r: _fabricate_attempts(r, 2),
    _erase_oracle,
    _reverse_queries,
    lambda r: next(e for e in _fixture(r)["events"] if e["kind"] == "action_decision")
        .update(decision_score=123456.0),
    lambda r: next(e for e in _fixture(r)["events"] if e["kind"] == "score_detail")
        .update(gate=[1000.0, -1000.0]),
    lambda r: r["fixtures"][0]["events"][0].update(kind=[]),
    lambda r: next(e for e in _fixture(r)["events"] if e["kind"] == "action_decision")
        .update(faults=[[[], 1]]),
    lambda r: r["fixtures"][0].update(horizon=True),
])
def test_correction_forgeries_refuse_valid_json(report, mutate):
    forged = copy.deepcopy(report)
    validate_report(forged)
    mutate(forged)
    with pytest.raises(DecodeEvidenceRefused):
        validate_report(forged)


@pytest.mark.parametrize("mutate", [
    lambda r: r["determinism"].update(position_keyed_uniforms=1),
    lambda r: r["fixtures"][0]["observer_comparison"].update(rng_equal=1),
    lambda r: r["fixtures"][0]["events"][0].update(rows=True),
    lambda r: r["fixtures"][0]["expected_statuses"][0].update(at_horizon=1),
    lambda r: r["fixtures"][0]["observed_statuses"][0].update(at_horizon=1),
    lambda r: r["fixtures"][0]["expected_work"]["totals"].update(sampler_calls=True),
    lambda r: r["fixtures"][0]["observed_work"]["totals"].update(sampler_calls=True),
    lambda r: next(e for e in _fixture(r)["events"] if e["kind"] == "score_detail")
        .update(supports=[]),
    lambda r: next(e for e in _fixture(r)["events"] if e["kind"] == "action_decision")
        .update(exit_reason=[]),
    lambda r: _fixture(r)["events"].insert(1, _fixture(r)["events"].pop(-2)),
])
def test_nested_domains_and_stop_placement_refuse(report, mutate):
    forged = copy.deepcopy(report)
    validate_report(forged)
    mutate(forged)
    with pytest.raises(DecodeEvidenceRefused):
        validate_report(forged)


@pytest.mark.parametrize("helper", ["candidate_spans", "execute_copy", "prefix_progress"])
def test_judge_implementation_failure_propagates(report, monkeypatch, helper):
    sentinel = RuntimeError(helper)

    def fail(*args, **kwargs):
        raise sentinel

    monkeypatch.setattr("dm.eval.relation_decode_evidence." + helper, fail)
    with pytest.raises(RuntimeError) as caught:
        validate_report(report)
    assert caught.value is sentinel


def test_publication_success_roundtrips(tmp_path):
    path = tmp_path / "report.json"
    publish_report(path)
    validate_report(json.loads(path.read_text()))


def test_failed_build_leaves_no_success_artifact(tmp_path, monkeypatch):
    path = tmp_path / "report.json"
    failure = RuntimeError("fixture failed")

    def fail():
        raise failure

    monkeypatch.setattr("dm.eval.relation_decode_evidence.build_report", fail)
    with pytest.raises(RuntimeError) as caught:
        publish_report(path)
    assert caught.value is failure
    assert not path.exists()


def test_publication_refuses_existing_before_running(tmp_path, monkeypatch):
    path = tmp_path / "report.json"
    path.write_text("untouched")
    monkeypatch.setattr("dm.eval.relation_decode_evidence.build_report",
                        lambda: pytest.fail("existing output reached fixtures"))
    with pytest.raises(FileExistsError):
        publish_report(path)
    assert path.read_text() == "untouched"
