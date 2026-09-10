"""CR-2 sparse Adam and failed-update witnesses."""
import os
import subprocess
import sys
from dataclasses import replace

import pytest
import torch
from test_relation_r4_readiness import _trainer, _training

from dm.eval.relation_evidence import optimizer_digest, state_digest
from dm.train_relation import (
    AUTHORITATIVE_R4_SOURCES,
    RelationBucketCursor,
    RelationTrainer,
    RelationTrainingRefused,
    TrainingCorpus,
)


def test_negative_only_adam_round_trip(tmp_path):
    training = TrainingCorpus.engineering_cases(tuple(replace(c, actions=()) for c in _training().cases))
    trainer = _trainer(training)
    for step in range(4):
        path = tmp_path / f"cut-{step}.pt"
        trainer.save(path, sources=AUTHORITATIVE_R4_SOURCES)
        resumed = RelationTrainer.load(path, training=training, sources=AUTHORITATIVE_R4_SOURCES)
        assert optimizer_digest(trainer.optimizer, trainer.model) == optimizer_digest(resumed.optimizer, resumed.model)
        trainer.step()
        resumed.step()
        assert state_digest(trainer.model.state_dict()) == state_digest(resumed.model.state_dict())
        for name, parameter in trainer.model.named_parameters():
            action = name.startswith("relation.") and name not in ("relation.norm.weight", "relation.gate.weight")
            assert (parameter not in trainer.optimizer.state) == action


def _mixed_trainer():
    original = _training()
    cursor = RelationBucketCursor.seeded([len(c.flat) + 1 for c in original.cases], 1, 31)
    negatives = {cursor.next()[0], cursor.next()[0]}
    training = TrainingCorpus.engineering_cases(tuple(
        replace(case, actions=()) if index in negatives else case
        for index, case in enumerate(original.cases)))
    template = _trainer(training)
    return RelationTrainer(replace(template.config, batch_size=1), training)


@pytest.mark.parametrize("cut", [0, 2, 3, 7])
def test_sparse_mixed_fresh_process_continuation(tmp_path, cut):
    trainer = _mixed_trainer()
    for _ in range(cut):
        trainer.step()
    path = tmp_path / "cut.pt"
    trainer.save(path, sources=AUTHORITATIVE_R4_SOURCES)
    while trainer.completed_step < trainer.config.steps:
        trainer.step()
    expected = (state_digest(trainer.model.state_dict()), optimizer_digest(trainer.optimizer, trainer.model))
    code = '''
import json, sys
from test_relation_cr_numerics import _mixed_trainer
from dm.train_relation import RelationTrainer, AUTHORITATIVE_R4_SOURCES
from dm.eval.relation_evidence import state_digest, optimizer_digest
from pathlib import Path
training = _mixed_trainer().training
trainer = RelationTrainer.load(Path(sys.argv[1]), training=training, sources=AUTHORITATIVE_R4_SOURCES)
while trainer.completed_step < trainer.config.steps:
    trainer.step()
print(json.dumps([state_digest(trainer.model.state_dict()), optimizer_digest(trainer.optimizer, trainer.model)]))
'''
    import json
    result = subprocess.run([sys.executable, "-c", code, str(path)], check=True, capture_output=True,
                            text=True, env={**os.environ, "PYTHONPATH": ".:tests", "PYTHONDONTWRITEBYTECODE": "1"})
    assert tuple(json.loads(result.stdout)) == expected
    assert [row["positive_boundaries"] > 0 for row in trainer.history[:3]] == [False, False, True]


@pytest.mark.parametrize("mutation", ["omission", "counter", "dtype"])
def test_sparse_mixed_impossible_adam_state_refuses(tmp_path, mutation):
    trainer = _mixed_trainer()
    for _ in range(3):
        trainer.step()
    path = tmp_path / "invalid.pt"
    payload = trainer.save(path, sources=AUTHORITATIVE_R4_SOURCES)
    index = next(index for index, (name, _) in enumerate(trainer.model.named_parameters())
                 if name == "relation.query.weight")
    item = payload["optimizer"]["state"][index]
    assert float(item["step"]) == 1
    if mutation == "omission":
        del payload["optimizer"]["state"][index]
    elif mutation == "counter":
        item["step"].fill_(3)
    else:
        item["step"] = item["step"].double()
    torch.save(payload, path)
    with pytest.raises(RelationTrainingRefused, match="optimizer"):
        RelationTrainer.load(path, training=trainer.training, sources=AUTHORITATIVE_R4_SOURCES)


def test_negative_second_moment_refuses_publication(tmp_path):
    trainer = _trainer()
    trainer.step()
    next(iter(trainer.optimizer.state.values()))["exp_avg_sq"].flatten()[0] = -1
    with pytest.raises(RelationTrainingRefused, match="second moment"):
        trainer.save(tmp_path / "invalid.pt", sources=AUTHORITATIVE_R4_SOURCES)


@pytest.mark.parametrize("when", ["objective", "gradient", "model", "exp_avg", "exp_avg_sq"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_update_fail_stops_without_success(tmp_path, monkeypatch, when, value):
    trainer = _trainer()
    if when == "objective":
        original_forward = trainer.model.forward

        def corrupt_forward(*args, **kwargs):
            logits, states = original_forward(*args, **kwargs)
            return logits * value, states

        monkeypatch.setattr(trainer.model, "forward", corrupt_forward)
        monkeypatch.setattr(trainer.optimizer, "step", lambda: pytest.fail("invalid objective reached Adam"))
    elif when == "gradient":
        next(trainer.model.parameters()).register_hook(lambda grad: torch.full_like(grad, value))
        monkeypatch.setattr(trainer.optimizer, "step", lambda: pytest.fail("invalid gradient reached Adam"))
    else:
        original = trainer.optimizer.step

        def corrupt():
            original()
            with torch.no_grad():
                tensor = next(trainer.model.parameters()) if when == "model" else next(iter(trainer.optimizer.state.values()))[when]
                tensor.flatten()[0] = value

        monkeypatch.setattr(trainer.optimizer, "step", corrupt)
    with pytest.raises(RelationTrainingRefused, match="nonfinite"):
        trainer.step()
    assert trainer._failed
    assert trainer.completed_step == 0
    assert not trainer.history
    with pytest.raises(RelationTrainingRefused, match="failed an earlier update"):
        trainer.step()
    with pytest.raises(RelationTrainingRefused, match="failed trainer"):
        trainer.save(tmp_path / "failed.pt", sources=AUTHORITATIVE_R4_SOURCES)
