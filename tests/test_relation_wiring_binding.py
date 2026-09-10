"""RB-1/RB-2: card-to-state binding before publication or restoration."""
from copy import copy
from dataclasses import asdict, replace

import pytest
import torch

from dm.relation import wiring
from dm.train_relation import AUTHORITATIVE_R4_SOURCES, RelationTrainer, TrainingCorpus


@pytest.mark.parametrize("arm", wiring.ARMS)
@pytest.mark.parametrize("route", ["save", "load"])
def test_live_provenance_drift_refuses(tmp_path, monkeypatch, arm, route):
    card, training = wiring.run_card("binding"), wiring._fixture()
    trainer = RelationTrainer(wiring.config(arm), training)
    path = tmp_path / "state.pt"
    if route == "load":
        wiring.save_checkpoint(path, trainer, card)
    changed = copy(training)
    # Adversarial in-process drift, not a valid manifest-backed constructor.
    object.__setattr__(changed, "provenance", "scientific")
    if route == "save":
        trainer.training = changed
        monkeypatch.setattr(trainer, "save", lambda *a, **kw: pytest.fail("temporary codec work"))
        with pytest.raises(wiring.WiringRefused, match="corpus"):
            wiring.save_checkpoint(path, trainer, card)
        assert list(tmp_path.iterdir()) == []
    else:
        monkeypatch.setattr(RelationTrainer, "load", lambda *a, **kw: pytest.fail("R4 restoration"))
        with pytest.raises(wiring.WiringRefused, match="corpus"):
            wiring.load_checkpoint(path, card, changed)


@pytest.mark.parametrize("arm", wiring.ARMS)
def test_rehashed_foreign_card_refuses_before_restore(tmp_path, monkeypatch, arm):
    card, training = wiring.run_card("binding"), wiring._fixture()
    trainer = RelationTrainer(wiring.config(arm), training)
    path = tmp_path / "state.pt"
    wiring.save_checkpoint(path, trainer, card)
    payload = torch.load(path, weights_only=False)
    card["configs"][arm]["steps"] = 7
    card["configs"][arm]["warmup"] = 2
    payload["execution_state"]["config"] = card["configs"][arm]
    payload["card_sha256"] = wiring.digest(card)
    torch.save(payload, path)
    monkeypatch.setattr(RelationTrainer, "load", lambda *a, **kw: pytest.fail("R4 restoration"))
    with pytest.raises(wiring.WiringRefused, match="card"):
        wiring.load_checkpoint(path, card, training)


def changed_corpus(training, change):
    cases = list(training.cases)
    if change == "supervision":
        cases[7] = replace(cases[7], actions=())
    elif change == "order":
        cases[6], cases[7] = cases[7], cases[6]
    elif change == "program":
        cases[7] = replace(cases[7], flat=cases[7].flat + b"\x00")
    elif change == "case_id":
        cases[7] = replace(cases[7], case_id="changed-future-row")
    return TrainingCorpus.engineering_cases(cases)


@pytest.mark.parametrize("arm", wiring.ARMS)
@pytest.mark.parametrize("change", ["supervision", "order", "program", "case_id"])
def test_changed_corpus_refuses_before_publication(tmp_path, monkeypatch, arm, change):
    card = wiring.run_card("binding")
    training = changed_corpus(wiring._fixture(), change)
    trainer = RelationTrainer(wiring.config(arm), training)
    monkeypatch.setattr(trainer, "save", lambda *a, **kw: pytest.fail("temporary codec work"))
    with pytest.raises(wiring.WiringRefused, match="corpus"):
        wiring.save_checkpoint(tmp_path / "state.pt", trainer, card)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("arm", wiring.ARMS)
@pytest.mark.parametrize("change", ["schedule", "warmup", "optimizer", "model", "seed"])
def test_wrong_recipe_refuses_before_publication(tmp_path, monkeypatch, arm, change):
    card = wiring.run_card("binding")
    config = wiring.config(arm)
    edits = {
        "schedule": {"steps": 7, "warmup": 2},
        "warmup": {"warmup": 2},
        "optimizer": {"lr": 0.002},
        "model": {"model": {**config.model, "d_model": 32}},
        "seed": {"seed": 32},
    }
    trainer = RelationTrainer(replace(config, **edits[change]), wiring._fixture())
    monkeypatch.setattr(trainer, "save", lambda *a, **kw: pytest.fail("temporary codec work"))
    with pytest.raises(wiring.WiringRefused, match="config"):
        wiring.save_checkpoint(tmp_path / "state.pt", trainer, card)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("arm", wiring.ARMS)
@pytest.mark.parametrize("change", ["supervision", "order", "program", "case_id"])
def test_valid_r4_bundle_and_caller_cannot_override_card(tmp_path, monkeypatch, arm, change):
    card = wiring.run_card("binding")
    training = changed_corpus(wiring._fixture(), change)
    trainer = RelationTrainer(wiring.config(arm), training)
    # Deliberately bypass R5 writer, reproducing pre-repair envelopes with
    # internally consistent R4 state and a caller corpus agreeing with it.
    state = trainer.save(tmp_path / "inner.pt", sources=AUTHORITATIVE_R4_SOURCES)
    payload = {"schema": 1, "namespace": wiring.NAMESPACE, "card_sha256": wiring.digest(card),
               "execution_state": {k: v for k, v in state.items() if k not in ("schema", "namespace")},
               "identity": wiring.identity(trainer), "arm": arm,
               "role": "continuation", "completed_step": 0}
    path = tmp_path / "forged.pt"
    torch.save(payload, path)
    monkeypatch.setattr(RelationTrainer, "load", lambda *a, **kw: pytest.fail("R4 restoration"))
    before = wiring.rng_digest("cpu")
    before_policy = (torch.are_deterministic_algorithms_enabled(),
                     torch.is_deterministic_algorithms_warn_only_enabled())
    with pytest.raises(wiring.WiringRefused, match="corpus"):
        wiring.load_checkpoint(path, card, training)
    assert before == wiring.rng_digest("cpu")
    assert before_policy == (torch.are_deterministic_algorithms_enabled(),
                            torch.is_deterministic_algorithms_warn_only_enabled())


@pytest.mark.parametrize("arm", wiring.ARMS)
@pytest.mark.parametrize("field,value", [
    ("supervision_sha256", "a" * 64), ("training_program_fingerprint", "a" * 64),
    ("provenance", "scientific"), ("manifest_path", "foreign.json"),
    ("supervision_identity_version", "unknown"),
])
def test_bundle_corpus_refuses_before_restore(tmp_path, monkeypatch, arm, field, value):
    card, training = wiring.run_card("binding"), wiring._fixture()
    trainer = RelationTrainer(wiring.config(arm), training)
    path = tmp_path / "state.pt"
    wiring.save_checkpoint(path, trainer, card)
    payload = torch.load(path, weights_only=False)
    payload["execution_state"]["corpus"][field] = value
    torch.save(payload, path)
    monkeypatch.setattr(RelationTrainer, "load", lambda *a, **kw: pytest.fail("R4 restoration"))
    with pytest.raises(wiring.WiringRefused, match="corpus"):
        wiring.load_checkpoint(path, card, training)


@pytest.mark.parametrize("arm", wiring.ARMS)
def test_noncanonical_card_refuses_publication(tmp_path, monkeypatch, arm):
    card, training = wiring.run_card("binding"), wiring._fixture()
    config = replace(wiring.config(arm), steps=7, warmup=2)
    card["configs"][arm] = asdict(config)
    trainer = RelationTrainer(config, training)
    monkeypatch.setattr(trainer, "save", lambda *a, **kw: pytest.fail("temporary codec work"))
    with pytest.raises(wiring.WiringRefused, match="card"):
        wiring.save_checkpoint(tmp_path / "state.pt", trainer, card)
    assert list(tmp_path.iterdir()) == []
