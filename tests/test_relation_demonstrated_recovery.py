"""The named repair must preserve saved state and refuse unrelated migrations."""
from copy import deepcopy

import pytest

from dm.relation import demonstrated_recovery as recovery
from dm.relation.demonstrated_fixture import build_fit_fixture
from dm.relation.demonstrated_run import load_checkpoint, make_card, save_checkpoint
from dm.relation.wiring import identity
from dm.train_relation import TrainingCorpus


@pytest.fixture(scope="module")
def card():
    audits = recovery.ROOT / "artifacts/relation/demonstrated-affine-i0-03"
    if not audits.exists() or not recovery.PREVIOUS.exists():
        pytest.skip("named recovery evidence is not available in this checkout")
    return make_card(audits, recover=True)


def test_recovery_preserves_all_state_and_saves_under_successor_card(card, tmp_path):
    cell = card["recipe"]["cells"][0]
    training = TrainingCorpus.engineering_cases(build_fit_fixture().cases)
    trainer, histories = recovery.restore(card, cell, training)
    assert trainer.completed_step == 256
    assert trainer.history == [r["history"] for r in histories[:256]]
    assert len(histories) == 1024
    before = identity(trainer)
    path = tmp_path / "recovered.pt"
    assert save_checkpoint(path, trainer, card, cell) == before
    restored = load_checkpoint(path, card, cell, training, 256)
    assert identity(restored) == before


@pytest.mark.parametrize("mutation", ("recipe", "training_source", "environment"))
def test_recovery_rejects_unrelated_changes(card, mutation):
    changed = deepcopy(card)
    if mutation == "recipe":
        changed["recipe"]["cells"][0]["config"]["lr"] *= 2
    elif mutation == "training_source":
        changed["source_sha256"]["dm/train_relation.py"] = "0" * 64
    else:
        changed["environment"] = {}
    with pytest.raises(ValueError, match="changed|source change"):
        recovery.specification(changed)


def test_recovery_rejects_modified_input(card, monkeypatch):
    monkeypatch.setitem(recovery.INPUTS, "fit-131-256.pt", "0" * 64)
    with pytest.raises(ValueError, match="recovery input changed"):
        recovery.specification(card)
