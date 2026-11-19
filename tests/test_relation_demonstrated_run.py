"""Real initialization/checkpoint/readout checks plus fail-closed card tests."""
import json
from copy import deepcopy
from dataclasses import asdict

import pytest
import torch

from dm.relation import demonstrated_run as run
from dm.relation.demonstrated_card import CELLS, config, recipe
from dm.relation.demonstrated_fixture import build_fit_fixture
from dm.relation.demonstrated_readout import evaluate, fit_result
from dm.relation.wiring import identity
from dm.train_relation import RelationTrainer, TrainingCorpus


@pytest.fixture(scope="module")
def training():
    return TrainingCorpus.engineering_cases(build_fit_fixture().cases)


@pytest.mark.parametrize("name,stage,arm,seed", CELLS)
def test_json_cell_config_roundtrip(name, stage, arm, seed):
    original = config(arm, seed)
    cell = json.loads(json.dumps({"config": asdict(original)}))
    assert run.cell_config(cell) == original


def test_serialized_execution_reaches_step_zero_checkpoint(tmp_path, monkeypatch):
    cell = recipe()["cells"][0]
    card = {"recipe": {"cells": [cell]}}
    run.write(tmp_path / "run-card.json", card)
    monkeypatch.setattr(run, "validate_card", lambda c: None)

    def stop(trainer, *args):
        assert trainer.completed_step == 0
        assert (tmp_path / "fit-131-0.pt").exists()
        raise RuntimeError("integration stops before primary updates")

    monkeypatch.setattr(run, "evaluate", stop)
    with pytest.raises(RuntimeError, match="integration stops"):
        run.execute(tmp_path, run.digest(card))
    terminal = json.loads((tmp_path / "terminal.json").read_text())
    assert terminal["completed_updates"] == {}
    assert terminal["active_step"] == 0


def test_real_paired_initialization_independent_state(training):
    baseline = RelationTrainer(config("none", 131), training)
    relation = RelationTrainer(config("span_affine_v1", 131), training)
    left = baseline.model.state_dict()
    right = relation.model.state_dict()
    assert left.keys() <= right.keys()
    assert all(torch.equal(value, right[key]) for key, value in left.items())
    assert all(value.data_ptr() != right[key].data_ptr() for key, value in left.items())
    assert baseline.cursor.next() == relation.cursor.next()
    before = identity(baseline)
    relation.step()  # isolated integration update, never a primary experiment cell
    assert identity(baseline) == before
    rows = evaluate(baseline, training.cases[:1], "fit", lambda row: None, "baseline-test")
    assert rows[0]["queries"] == []
    assert rows[0]["decodes"]["oracle_copy"]["target_exact"]
    assert "predicted_copy" not in rows[0]["decodes"]
    assert identity(baseline) == before


def test_actual_checkpoint_roundtrip_and_evaluation(tmp_path, training):
    cfg = config("span_affine_v1", 131)
    trainer = RelationTrainer(cfg, training)
    trainer.step()
    cell = {"cell_id": "integration-only", "config": asdict(cfg)}
    card = {"integration_only": True}
    path = tmp_path / "checkpoint.pt"
    saved = run.save_checkpoint(path, trainer, card, cell)
    restored = run.load_checkpoint(path, card, cell, training, 1)
    assert identity(restored) == saved
    with pytest.raises(FileExistsError):
        run.save_checkpoint(path, trainer, card, cell)
    with pytest.raises(ValueError, match="binding mismatch"):
        run.load_checkpoint(path, {"wrong": True}, cell, training, 1)
    before = identity(restored)
    cases = training.cases[:1] + training.cases[16:17]
    rows = evaluate(restored, cases, "fit", lambda row: None, "test-checkpoint")
    assert identity(restored) == before
    assert len(rows) == 2
    assert rows[0]["decodes"]["oracle_copy"]["target_exact"]
    assert "oracle_copy" not in rows[1]["decodes"]
    for row in rows:
        assert sum(q["prompt"] for q in row["queries"]) == 1
        assert row["all_byte_nll"]["count"] == len(cases[0].flat)
        for decoded in row["decodes"].values():
            decisions = [e for e in decoded["events"] if e["kind"] == "action_decision"]
            assert decoded["copy_invocations"] == sum(e["admission_bytes"] > 0 for e in decisions)
            assert any(e["kind"] == "request_completed" for e in decoded["events"])
    restored.step()
    trainer.step()
    assert identity(restored) == identity(trainer)


def test_card_rejects_swapped_or_modified_audit(tmp_path):
    (tmp_path / "fit-audit.json").write_text('{"status": "candidate_pilot_audit_only"}')
    with pytest.raises(ValueError, match="unrecognized audit identity"):
        run.make_card(tmp_path)


def test_validate_reconstructs_entire_card(monkeypatch):
    card = {"audits": {"fit-audit.json": {"path": "/tmp/audits/fit-audit.json"}},
            "recipe": {"x": 1}, "environment": {"cpu": True}}
    monkeypatch.setattr(run, "make_card", lambda path: card)
    run.validate_card(card)
    for key in ("recipe", "environment"):
        altered = deepcopy(card)
        altered[key] = {}
        with pytest.raises(ValueError, match="drift"):
            run.validate_card(altered)


def test_exclusive_launch_refuses_before_any_fixture(tmp_path, monkeypatch):
    card = {"test": True}
    run.write(tmp_path / "run-card.json", card)
    run.write(tmp_path / "launch.json", {})
    monkeypatch.setattr(run, "validate_card", lambda c: None)
    monkeypatch.setattr(run, "build_fit_fixture", lambda: pytest.fail("fixture must not run"))
    with pytest.raises(FileExistsError):
        run.execute(tmp_path, run.digest(card))


def test_setup_failure_preserves_terminal(tmp_path, monkeypatch):
    card = {"test": True}
    run.write(tmp_path / "run-card.json", card)
    monkeypatch.setattr(run, "validate_card", lambda c: None)

    def fail():
        assert run.signal.getitimer(run.signal.ITIMER_REAL)[0] > 0
        raise TimeoutError("setup cap")

    monkeypatch.setattr(run, "build_fit_fixture", fail)
    with pytest.raises(TimeoutError):
        run.execute(tmp_path, run.digest(card))
    terminal = json.loads((tmp_path / "terminal.json").read_text())
    assert terminal["status"] == "budget_exhausted"
    assert terminal["completed_updates"] == {}


def stored_fit(exact=15, negative=0, halt=False):
    return [{"case_id": str(i), "class": "positive" if i < 16 else "broken",
             "queries": [{"prompt": True, "gate_nll": 1, "margin": None,
                          "highest_valid_full_target_correct": i < exact}],
             "decodes": {"predicted_copy": {"exact_with_copy": i < exact,
                         "copy_invocations": int(i < 16 or i < 16 + negative),
                         "valid_halt": halt}}} for i in range(32)]


def test_fit_thresholds_halt_and_missing_margin_denominators():
    result = fit_result(stored_fit())
    assert result["pass"] and result["weak_halt"]
    assert result["positive_margin"]["count"] == 0
    assert not fit_result(stored_fit(exact=14))["pass"]
    assert not fit_result(stored_fit(negative=1))["pass"]
    with pytest.raises(ValueError, match="denominator"):
        fit_result(stored_fit()[:-1])
