"""CR-3 live corpus and publication binding regressions."""
from pathlib import Path

import pytest
from test_relation_r4_readiness import _trainer, _training

import dm.train_relation as module
from dm.train_relation import AUTHORITATIVE_R4_SOURCES, RelationTrainingRefused, TrainingCorpus


@pytest.mark.parametrize("owner", ["trainer", "planner"])
@pytest.mark.parametrize("operation", ["step", "save"])
def test_live_corpus_rebinding_refuses_before_work(tmp_path, owner, operation):
    trainer = _trainer()
    target = trainer if owner == "trainer" else trainer.planner
    target.training = _training()
    with pytest.raises(RelationTrainingRefused, match="corpus binding"):
        if operation == "step":
            trainer.step()
        else:
            trainer.save(tmp_path / "rebound.pt", sources=AUTHORITATIVE_R4_SOURCES)
    assert trainer.cursor.consumed_batches == 0
    assert trainer.completed_step == 0
    assert not (tmp_path / "rebound.pt").exists()


def test_low_level_writer_requires_prework_binding(tmp_path, monkeypatch):
    trainer = _trainer()
    kwargs = {"config": trainer.config, "completed_step": 0, "model": trainer.model,
              "optimizer": trainer.optimizer, "sampler": trainer.cursor.state_dict(),
              "accounting": trainer.accounting, "history": trainer.history, "training": trainer.training,
              "data_generator": trainer.cursor.generator, "sources": AUTHORITATIVE_R4_SOURCES,
              "diagnostics": trainer.diagnostics()}
    path = tmp_path / "unbound.pt"
    with pytest.raises(TypeError, match="process_environment"):
        module.save_training_bundle(path, **kwargs)
    drifted = dict(trainer._process_source_hashes)
    drifted[next(iter(drifted))] = "0" * 64
    monkeypatch.setattr(module, "source_hashes", lambda: drifted)
    with pytest.raises(RelationTrainingRefused, match="source contents drifted"):
        module.save_training_bundle(path, **kwargs, process_environment=trainer._process_environment,
                                    process_source_hashes=trainer._process_source_hashes)
    assert not path.exists()


def test_fabricated_manifest_constructor_refuses():
    with pytest.raises(RelationTrainingRefused, match="manifest-backed"):
        TrainingCorpus(_training().cases, Path("missing.json"), "0" * 64, "0" * 64, "0" * 64, "scientific")
