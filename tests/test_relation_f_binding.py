"""F-1 public ingress must reject mutable supervision before model work."""
from dataclasses import replace

import pytest
from test_relation_r4_readiness import _trainer, _training

from dm.train_relation import RelationTrainingRefused, TrainingCorpus


@pytest.mark.parametrize("field,value", [
    ("actions", lambda c: list(c.actions)),
    ("flat", lambda c: bytearray(c.flat)),
    ("structured", lambda c: bytearray(c.structured)),
    ("groups", lambda c: list(c.groups)),
    ("plan", lambda c: (list(c.plan[0]),) if c.plan else ([],)),
    ("source_ids", lambda c: list(c.source_ids)),
    ("donor_groups", lambda c: []),
    ("donor_source_ids", lambda c: []),
    ("actions", lambda c: (None,)),
])
@pytest.mark.parametrize("index", [0, -1])
def test_mutable_or_malformed_future_supervision_refuses(field, value, index):
    cases = list(_training().cases)
    cases[index] = replace(cases[index], **{field: value(cases[index])})
    with pytest.raises(RelationTrainingRefused, match="immutable canonical"):
        TrainingCorpus.engineering_cases(cases)


def test_outer_sequence_is_snapshot():
    cases = list(_training().cases)
    training = TrainingCorpus.engineering_cases(cases)
    identity = training.supervision_identity
    cases.clear()
    assert training.cases
    assert training.supervision_identity == identity


def test_warm_cache_list_mutation_refuses_before_trainer_work(monkeypatch):
    import dm.train_relation as module

    original = _training().cases[0]
    actions = list(original.actions)
    assert actions
    case = replace(original, actions=actions)
    monkeypatch.setattr(module, "RelationTrainer", lambda *a, **k: pytest.fail("model work"))
    with pytest.raises(RelationTrainingRefused, match="immutable canonical"):
        training = TrainingCorpus.engineering_cases([case])
        trainer = module.RelationTrainer(None, training)
        trainer.planner.batch([0])
        actions.clear()
        trainer.step()


def test_outer_list_mutation_preserves_cached_targets():
    cases = list(_training().cases)
    training = TrainingCorpus.engineering_cases(cases)
    trainer = _trainer(training)
    before = trainer.planner.batch([0])
    cases.clear()
    after = trainer.planner.batch([0])
    assert after.queries == before.queries
    assert after.targets.equal(before.targets)
    assert after.inputs.equal(before.inputs)
    trainer.step()
    assert trainer.completed_step == 1
