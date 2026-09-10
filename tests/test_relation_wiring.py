"""R5 preparation/refusal and continuation adapter witnesses; no smoke launch."""
import io
import json
import os
import subprocess
import sys
from dataclasses import replace

import pytest
import torch

from dm.relation import wiring
from dm.train_relation import AUTHORITATIVE_R4_SOURCES, RelationTrainer, RelationTrainingRefused


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(wiring, "ROOT", tmp_path)
    (tmp_path / "artifacts/relation").mkdir(parents=True)
    monkeypatch.setattr(wiring, "sources", lambda: {"fixed": "a" * 64})
    return wiring.prepare("test")


def test_card_pins_scope_without_model_work(monkeypatch):
    monkeypatch.setattr(wiring, "RelationTrainer", lambda *a: pytest.fail("model work"))
    card = wiring.run_card("review")
    assert card["budget"] == {"primary_updates": 112, "restart_updates": 56, "optimizer_calls": 168}
    assert card["evaluation_steps"] == (0, 28, 56)
    assert len(card["case_order"]) == 8
    assert card["case_order"][1]["action_targets"] == 0
    assert card["configs"]["span_affine_v1"]["warmup"] == 7
    assert card["configs"]["none"]["steps"] == 56
    assert card["program_fingerprint"] == wiring._fixture().program_fingerprint
    assert wiring.encoded(card) == wiring.encoded(wiring.run_card("review"))


@pytest.mark.parametrize("run_id", ["../escape", "", "/tmp/test", "a/b", "a" * 65])
def test_bad_run_id_refuses(run_id):
    with pytest.raises(wiring.WiringRefused):
        wiring.run_card(run_id)


def test_prepare_reserves_and_refuses_overwrite_before_fixture(prepared, monkeypatch):
    path, digest = prepared
    assert wiring.digest(wiring.validate(path)) == digest
    monkeypatch.setattr(wiring, "_fixture", lambda: pytest.fail("fixture work"))
    with pytest.raises(FileExistsError):
        wiring.prepare("test")
    assert not (path / "launch.json").exists()


def test_wrong_approval_refuses_model_work(prepared, monkeypatch):
    path, _ = prepared
    monkeypatch.setattr(wiring, "RelationTrainer", lambda *a: pytest.fail("model work"))
    with pytest.raises(wiring.WiringRefused, match="approved"):
        wiring.execute(path, "wrong")
    assert not (path / "launch.json").exists()


@pytest.mark.parametrize("field", ["source_hashes", "environment", "case_order", "budget", "mode_order"])
def test_card_drift_refuses(prepared, field):
    path, _ = prepared
    card = json.loads((path / "run-card.json").read_text())
    card[field] = {}
    (path / "run-card.json").write_text(json.dumps(card))
    with pytest.raises(wiring.WiringRefused, match="drift"):
        wiring.validate(path)


@pytest.mark.parametrize("arm", wiring.ARMS)
@pytest.mark.parametrize("cut", [0, 2])
def test_r5_checkpoint_roundtrip_and_r4_refusal(tmp_path, arm, cut):
    card = wiring.run_card("test")
    training = wiring._fixture()
    trainer = RelationTrainer(wiring.config(arm), training)
    for _ in range(cut):
        trainer.step()
    before = wiring.identity(trainer)
    path = tmp_path / "r5.pt"
    wiring.save_checkpoint(path, trainer, card)
    restored = wiring.load_checkpoint(path, card, training)
    assert wiring.identity(restored) == before
    trainer.step()
    restored.step()
    assert wiring.identity(trainer) == wiring.identity(restored)
    with pytest.raises(RelationTrainingRefused):
        RelationTrainer.load(path, training=training, sources=AUTHORITATIVE_R4_SOURCES)
    with pytest.raises(FileExistsError):
        wiring.save_checkpoint(path, trainer, card)
    with pytest.raises(RelationTrainingRefused, match="temporary test"):
        replace(trainer.config, artifact_output_policy="exclusive_r5_engineering_v1")


def test_checkpoint_forged_identity_restores_caller_rng(tmp_path):
    card, training = wiring.run_card("test"), wiring._fixture()
    trainer = RelationTrainer(wiring.config("none"), training)
    path = tmp_path / "r5.pt"
    wiring.save_checkpoint(path, trainer, card)
    payload = torch.load(path, weights_only=False)
    payload["identity"]["model"] = "bad"
    torch.save(payload, path)
    torch.manual_seed(191)
    before = torch.get_rng_state().clone()
    with pytest.raises(wiring.WiringRefused, match="identity mismatch"):
        wiring.load_checkpoint(path, card, training)
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("arm", wiring.ARMS)
def test_decode_evidence_is_rng_neutral_and_negative_has_no_oracle(arm):
    card, training = wiring.run_card("test"), wiring._fixture()
    trainer = RelationTrainer(wiring.config(arm), training)
    before = wiring.identity(trainer)
    output = io.StringIO()
    wiring.evaluate(trainer, card, output)
    rows = [json.loads(line) for line in output.getvalue().splitlines()]
    assert len(rows) == 8 * len(wiring.MODES[arm])
    assert wiring.identity(trainer) == before
    negative = next(row for row in rows if row["row"] == 1 and row["mode"] == "oracle_copy")
    assert negative["action_targets"] == 0
    assert all(event.get("admission_bytes", 0) == 0 for event in negative["events"])
    assert all(row["target_bytes"] > 0 for row in rows)


def test_route_budget_order_and_restart_accounting(prepared, monkeypatch):
    path, approved = prepared
    root = path.parents[2]
    wiring.write_json(root / "artifacts/relation/r4-r0-cpu-20260906.json", {
        "repeatability": {"environment": wiring.execution_environment(device="cpu")}})
    calls, evaluations = [], []

    class FakeTrainer:
        def __init__(self, config, training):
            self.config, self.completed_step, self.history = config, 0, []

        def step(self):
            self.completed_step += 1
            self.history.append({"step": self.completed_step})
            calls.append(self.config.arm)

    monkeypatch.setattr(wiring, "RelationTrainer", FakeTrainer)
    monkeypatch.setattr(wiring, "evaluate", lambda t, c, h: evaluations.append((t.config.arm, t.completed_step)))
    monkeypatch.setattr(wiring, "save_checkpoint", lambda p, t, c, **kw: p.touch(exist_ok=False))
    monkeypatch.setattr(wiring, "identity", lambda t: {"step": t.completed_step})

    def child(command, **kwargs):
        arm = command[-1]
        calls.extend([arm] * 28)
        wiring.write_json(path / f"{arm}-restart-result.json", {
            "identity": {"step": 56}, "replayed_updates": 28})
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(wiring.subprocess, "run", child)
    wiring.execute(path, approved)
    assert len(calls) == 168
    assert evaluations == [(arm, step) for arm in wiring.ARMS for step in (0, 28, 56)]
    assert json.loads((path / "terminal.json").read_text())["status"] == "complete"
    with pytest.raises(wiring.WiringRefused, match="retry refused"):
        wiring.execute(path, approved)


def test_r5_codec_fresh_process_continuation(tmp_path):
    card, training = wiring.run_card("test"), wiring._fixture()
    trainer = RelationTrainer(wiring.config("span_affine_v1"), training)
    trainer.step()
    path = tmp_path / "state.pt"
    wiring.save_checkpoint(path, trainer, card)
    trainer.step()
    expected = wiring.identity(trainer)
    code = '''
import json, sys
from dm.relation import wiring
trainer = wiring.load_checkpoint(sys.argv[1], wiring.run_card("test"), wiring._fixture())
trainer.step()
print(json.dumps(wiring.identity(trainer), sort_keys=True))
'''
    result = subprocess.run([sys.executable, "-c", code, str(path)], check=True,
                            text=True, capture_output=True,
                            env={**os.environ, "PYTHONPATH": ".", "PYTHONDONTWRITEBYTECODE": "1"})
    assert json.loads(result.stdout) == expected


def test_attempted_failure_is_terminal_and_not_retried(prepared, monkeypatch):
    path, approved = prepared
    wiring.write_json(wiring.ROOT / "artifacts/relation/r4-r0-cpu-20260906.json", {
        "repeatability": {"environment": wiring.execution_environment(device="cpu")}})
    error = KeyboardInterrupt("stop here")

    def fail(*args):
        raise error

    monkeypatch.setattr(wiring, "RelationTrainer", fail)
    with pytest.raises(KeyboardInterrupt) as caught:
        wiring.execute(path, approved)
    assert caught.value is error
    assert json.loads((path / "terminal.json").read_text())["status"] == "failed"
    with pytest.raises(wiring.WiringRefused, match="retry refused"):
        wiring.execute(path, approved)


def test_decode_unexpected_rng_use_refuses_and_restores(monkeypatch):
    trainer = RelationTrainer(wiring.config("none"), wiring._fixture())
    before = wiring.identity(trainer)
    original = trainer.model.generate_relation

    def consume(*args, **kwargs):
        torch.rand(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(trainer.model, "generate_relation", consume)
    with pytest.raises(wiring.WiringRefused, match="consumed training RNG"):
        wiring.evaluate(trainer, wiring.run_card("test"), io.StringIO())
    assert wiring.identity(trainer) == before
