"""R5 fixed-fixture and failure-first lifecycle witnesses; no smoke launch."""
import json
from dataclasses import asdict, replace

import pytest
import torch

from dm.relation import wiring
from dm.train_relation import RelationTrainer
from scripts.relation_resource_benchmark import _training_config


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(wiring, "ROOT", tmp_path)
    monkeypatch.setattr(wiring, "sources", lambda: {"fixture": "a" * 64})
    (tmp_path / "artifacts/relation").mkdir(parents=True)
    return wiring.prepare("boundaries")[0]


def test_fixture_and_recipe_frozen():
    training = wiring._fixture()
    assert training.program_fingerprint == "ad12c0418d55e67dc1f74ebd1c4e5905ba48bf6b13be9a2ef3a29e53c679b8c0"
    assert training.supervision_identity == "97e69a64d5d5a468c50694515855669049ab82b6cea068d382ca9b041b6e1cd7"
    assert [(c.target_start, c.target_stop) for c in training.cases] == [
        (18, 72), (10, 30), (36, 144), (10, 40), (10, 40), (40, 50), (18, 54), (18, 36)]
    for arm in wiring.ARMS:
        assert asdict(wiring.config(arm)) == asdict(replace(
            _training_config(arm, 24_000_000), steps=56, warmup=7))


@pytest.mark.parametrize("mutation", ["missing", "digest", "status", "extra", "malformed"])
def test_preparation_evidence_refuses(prepared, mutation):
    path = prepared / "prepared.json"
    record = json.loads(path.read_text())
    if mutation == "missing":
        path.unlink()
    elif mutation == "malformed":
        path.write_text("{")
    else:
        record[{"digest": "card_sha256", "status": "status", "extra": "extra"}[mutation]] = "bad"
        path.write_text(json.dumps(record))
    with pytest.raises(wiring.WiringRefused, match="preparation"):
        wiring.validate(prepared)


def test_terminal_restart_refuses_before_loading(prepared, monkeypatch):
    card = wiring.validate(prepared)
    wiring.write_json(prepared / "launch.json", {"status": "attempted", "card_sha256": wiring.digest(card)})
    wiring.write_json(prepared / "terminal.json", {"status": "failed"})
    monkeypatch.setattr(wiring, "load_checkpoint", lambda *a: pytest.fail("checkpoint loading"))
    with pytest.raises(wiring.WiringRefused, match="terminal"):
        wiring.restart_worker(prepared, "none")


def test_child_failed_load_reserves_attempt(prepared, monkeypatch):
    card = wiring.validate(prepared)
    wiring.write_json(prepared / "launch.json", {"status": "attempted", "card_sha256": wiring.digest(card)})
    error = RuntimeError("injected load failure")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(wiring, "load_checkpoint", fail)
    with pytest.raises(RuntimeError) as caught:
        wiring.restart_worker(prepared, "none")
    assert caught.value is error
    assert (prepared / "none-restart-attempt.json").is_file()
    with pytest.raises(wiring.WiringRefused, match="retry refused"):
        wiring.restart_worker(prepared, "none")


@pytest.mark.parametrize("field,value", [
    ("arm", "span_affine_v1"), ("role", "final"), ("role", "unknown"),
    ("completed_step", True), ("completed_step", 1), ("completed_step", -1),
])
def test_checkpoint_envelope_role_arm_step(tmp_path, field, value):
    card, training = wiring.run_card("roles"), wiring._fixture()
    trainer = RelationTrainer(wiring.config("none"), training)
    path = tmp_path / "state.pt"
    wiring.save_checkpoint(path, trainer, card)
    payload = torch.load(path, weights_only=False)
    payload[field] = value
    torch.save(payload, path)
    before = torch.get_rng_state().clone()
    with pytest.raises(wiring.WiringRefused, match="mismatch"):
        wiring.load_checkpoint(path, card, training)
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("schema", [True, False, 1.0, "1", None])
def test_checkpoint_schema_exact_integer(tmp_path, schema):
    card, training = wiring.run_card("schema"), wiring._fixture()
    trainer = RelationTrainer(wiring.config("none"), training)
    path = tmp_path / "state.pt"
    wiring.save_checkpoint(path, trainer, card)
    payload = torch.load(path, weights_only=False)
    payload["schema"] = schema
    torch.save(payload, path)
    before = torch.get_rng_state().clone()
    with pytest.raises(wiring.WiringRefused, match="envelope"):
        wiring.load_checkpoint(path, card, training)
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("mutation", ["failed", "empty", "digest", "schema"])
def test_r0_matching_environment_is_not_certificate(tmp_path, monkeypatch, mutation):
    report = json.loads(wiring.R0_PATH.read_text())
    if mutation == "failed":
        report["decision"]["qualified"] = False
    elif mutation == "empty":
        report["repeatability"]["readings"] = []
    elif mutation == "schema":
        report["repeatability"]["schema"] = True
    else:
        report["repeatability"]["readings"][0]["model_state"] = "bad"
    path = tmp_path / "r0.json"
    path.write_text(json.dumps(report))
    monkeypatch.setattr(wiring, "R0_PATH", path)
    with pytest.raises(wiring.WiringRefused, match="R0"):
        wiring.validate_r0()


def test_frozen_r0_raw_evidence_passes():
    assert wiring.validate_r0()["route"] == "exact_state_reproduction"


@pytest.mark.parametrize("failure", ["serialize", "commit"])
def test_checkpoint_failure_never_exposes_final(tmp_path, monkeypatch, failure):
    card = wiring.run_card("publication")
    trainer = RelationTrainer(wiring.config("none"), wiring._fixture())
    path = tmp_path / "state.pt"
    original = torch.save
    error = OSError("injected publication failure")

    def save(value, destination, *args, **kwargs):
        if isinstance(value, dict) and value.get("namespace") == wiring.NAMESPACE:
            destination.write(b"partial")
            raise error
        return original(value, destination, *args, **kwargs)

    def link(*args):
        raise error

    if failure == "serialize":
        monkeypatch.setattr(torch, "save", save)
    else:
        monkeypatch.setattr(wiring.os, "link", link)
    with pytest.raises(OSError) as caught:
        wiring.save_checkpoint(path, trainer, card)
    assert caught.value is error
    assert not path.exists()
    assert path.with_name("state.pt.incomplete").exists()
    with pytest.raises(FileExistsError):
        wiring.save_checkpoint(path, trainer, card)
