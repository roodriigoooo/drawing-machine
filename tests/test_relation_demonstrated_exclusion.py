"""Persistent provenance refusal precedes any source-exclusion claim."""
import json
from types import SimpleNamespace

import pytest

from dm.relation import demonstrated_exclusion as exclusion


def test_first_difference_normalized_fields():
    assert exclusion.first_difference({"a": [1]}, {"a": [2]}) == {
        "path": "$.a[0]", "recorded": 1, "rebuilt": 2}
    assert exclusion.first_difference({"a": []}, {"a": []}) is None
    assert exclusion.first_difference([], [1])["path"] == "$.length"
    assert exclusion.first_difference({}, {"a": 1})["recorded_present"] is False


def test_rebuild_mismatch_retains_witness_without_scan(tmp_path, monkeypatch):
    body = {"corpus_sha256": "old", "cases": [{"case_id": "x", "hash": "old"}],
            "acceptance": {"config": {}, "count": 1}}
    rebuilt_body = {"corpus_sha256": "new", "cases": [{"case_id": "x", "hash": "new"}],
                    "acceptance": {"config": {}, "count": 2}}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(body))
    calls = []
    monkeypatch.setattr(exclusion.relation, "load_manifest", lambda p: body)
    monkeypatch.setattr(exclusion.relation, "manifest_config", lambda b: {})
    monkeypatch.setattr(exclusion.relation, "build", lambda c: calls.append(c) or SimpleNamespace())
    monkeypatch.setattr(exclusion.relation, "manifest", lambda c: rebuilt_body)
    result = exclusion.audit_scientific_sources(path)
    assert calls == [{}]
    assert result["status"] == "scientific_manifest_rebuild_mismatch"
    assert result["source_scan_performed"] is False
    assert result["first_case_difference"]["path"] == "$[0].hash"
    assert result["first_acceptance_difference"]["path"] == "$.count"
    assert result["model_calls"] == result["optimizer_updates"] == 0


def test_successful_rebuild_scans_unannotated_spans(tmp_path, monkeypatch):
    from dm.relation.demonstrated_fixture import build_fit_fixture

    source = build_fit_fixture().sources[0]
    body = {"corpus_sha256": "same", "acceptance": {"config": {}}, "cases": []}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(body))
    monkeypatch.setattr(exclusion.relation, "load_manifest", lambda p: body)
    monkeypatch.setattr(exclusion.relation, "manifest_config", lambda b: {})
    monkeypatch.setattr(exclusion.relation, "build", lambda c: SimpleNamespace(
        cases=[SimpleNamespace(flat=source + b"\x00")]))
    monkeypatch.setattr(exclusion.relation, "manifest", lambda c: body)
    monkeypatch.setattr(exclusion, "build_fit_fixture", lambda: SimpleNamespace(sources=(source,)))
    monkeypatch.setattr(exclusion, "build_pilot_fixture", lambda: SimpleNamespace(sources=()))
    result = exclusion.audit_scientific_sources(path)
    assert result["source_scan_performed"]
    assert result["status"] == "task_invalid"
    assert result["overlaps"] == [source.hex()]
    assert result["unique_three_instruction_source_spans"] == 1


def test_cli_persists_refusal_and_preserves_existing_output(tmp_path, monkeypatch):
    from scripts import relation_demonstrated_exclusion as cli

    output = tmp_path / "failure.json"
    result = {"status": "scientific_manifest_rebuild_mismatch", "overlaps": [],
              "model_calls": 0, "optimizer_updates": 0}
    monkeypatch.setattr(cli, "audit_scientific_sources", lambda p: dict(result))
    monkeypatch.setattr(cli.sys, "argv", ["exclusion", "manifest.json", str(output)])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 1
    saved = output.read_bytes()
    assert json.loads(saved)["status"] == result["status"]
    assert json.loads(saved)["source_sha256"]
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert output.read_bytes() == saved
