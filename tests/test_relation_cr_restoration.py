"""CR-1 transaction regressions at caller-owned restoration boundaries."""
from copy import deepcopy

import pytest
import torch
from test_relation_r4_readiness import _trainer

from dm.eval.relation_evidence import optimizer_digest, rng_digest, state_digest
from dm.train_relation import (
    AUTHORITATIVE_R4_SOURCES,
    RelationTrainer,
    RelationTrainingRefused,
    load_training_bundle,
)


@pytest.mark.parametrize("digest", ["model", "optimizer", "rng"])
def test_wrong_digest_rolls_back_all_caller_state(tmp_path, digest):
    saved = _trainer()
    saved.step()
    path = tmp_path / "state.pt"
    payload = saved.save(path, sources=AUTHORITATIVE_R4_SOURCES)
    payload["content_digests"][digest] = "0" * 64
    torch.save(payload, path)
    caller = _trainer()
    parameters = tuple(caller.model.parameters())
    before = (state_digest(caller.model.state_dict()), optimizer_digest(caller.optimizer, caller.model),
              rng_digest("cpu", caller.cursor.generator))
    with pytest.raises(RelationTrainingRefused, match="digest"):
        load_training_bundle(path, model=caller.model, optimizer=caller.optimizer,
                             data_generator=caller.cursor.generator, training=caller.training,
                             sources=AUTHORITATIVE_R4_SOURCES)
    assert before == (state_digest(caller.model.state_dict()), optimizer_digest(caller.optimizer, caller.model),
                      rng_digest("cpu", caller.cursor.generator))
    assert all(a is b for a, b in zip(parameters, caller.model.parameters(), strict=True))
    assert all(a is b for a, b in zip(parameters, caller.optimizer.param_groups[0]["params"], strict=True))


@pytest.mark.parametrize("kind", ["class", "order", "interrupt"])
def test_restore_preflight_and_baseexception(tmp_path, monkeypatch, kind):
    saved = _trainer()
    saved.step()
    path = tmp_path / "state.pt"
    saved.save(path, sources=AUTHORITATIVE_R4_SOURCES)
    caller = _trainer()
    if kind == "class":
        caller.optimizer = torch.optim.SGD(caller.model.parameters(), lr=0.01)
    elif kind == "order":
        caller.optimizer.param_groups[0]["params"].reverse()
    else:
        original = caller.optimizer.load_state_dict
        calls = 0

        def interrupted(state):
            nonlocal calls
            calls += 1
            original(state)
            if calls == 1:
                raise KeyboardInterrupt("injected after optimizer restoration")

        monkeypatch.setattr(caller.optimizer, "load_state_dict", interrupted)
    before = (state_digest(caller.model.state_dict()), deepcopy(caller.optimizer.state_dict()),
              rng_digest("cpu", caller.cursor.generator))
    expected = KeyboardInterrupt if kind == "interrupt" else RelationTrainingRefused
    with pytest.raises(expected):
        load_training_bundle(path, model=caller.model, optimizer=caller.optimizer,
                             data_generator=caller.cursor.generator, training=caller.training,
                             sources=AUTHORITATIVE_R4_SOURCES)
    assert state_digest(caller.model.state_dict()) == before[0]
    assert caller.optimizer.state_dict() == before[1]
    assert rng_digest("cpu", caller.cursor.generator) == before[2]


def test_high_level_rejection_restores_deterministic_policy(tmp_path):
    saved = _trainer()
    path = tmp_path / "state.pt"
    payload = saved.save(path, sources=AUTHORITATIVE_R4_SOURCES)
    payload["content_digests"]["model"] = "0" * 64
    torch.save(payload, path)
    original = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())
    try:
        torch.use_deterministic_algorithms(False, warn_only=True)
        before = rng_digest("cpu")
        with pytest.raises(RelationTrainingRefused):
            RelationTrainer.load(path, training=saved.training, sources=AUTHORITATIVE_R4_SOURCES)
        assert not torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
        assert rng_digest("cpu") == before
    finally:
        torch.use_deterministic_algorithms(original[0], warn_only=original[1])
