"""Direction 4 readiness repairs: physical, corpus and checkpoint binding."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

import dm.train_relation as train_module
from dm.data import relation
from dm.eval.relation_evidence import optimizer_digest, rng_digest, state_digest
from dm.isa.codec import ByteCodec
from dm.models.transformer import Config, DrawingLM
from dm.train_relation import (
    AUTHORITATIVE_R4_SOURCES,
    RelationTrainer,
    RelationTrainingRefused,
    TrainingCorpus,
    load_training_bundle,
    load_training_corpus,
)

MODEL = {
    "vocab_size": ByteCodec.vocab_size,
    "d_model": 16,
    "n_layers": 1,
    "n_heads": 2,
    "max_len": 256,
    "relation_schema": "span_affine_v1",
}


def _training() -> TrainingCorpus:
    built = relation.build_venue1(
        8, seed=41, motifs=relation.motif_pool(16, seed=40), tuples=relation.venue1_tuples()
    )
    return TrainingCorpus.engineering_cases(built.cases)


def _trainer(training: TrainingCorpus | None = None) -> RelationTrainer:
    return RelationTrainer(
        train_module.RelationTrainConfig(
            "span_affine_v1", MODEL, seed=31, batch_size=3, max_len=256,
            steps=7, warmup=2, corpus_provenance="engineering",
            seed_namespace="engineering",
        ),
        _training() if training is None else training,
    )


def test_manifest_snapshot_translates_io_and_json_root_refusals(tmp_path):
    with pytest.raises(RelationTrainingRefused, match="not readable"):
        load_training_corpus(tmp_path / "absent.json")
    root = tmp_path / "list.json"
    root.write_text("[]")
    with pytest.raises(RelationTrainingRefused, match="root must be an object"):
        load_training_corpus(root)


def test_manifest_snapshot_is_derived_once_even_if_the_path_is_replaced(tmp_path, monkeypatch):
    source = Path("runs/relation_corpus_v1.json")
    path = tmp_path / "manifest.json"
    original = source.read_bytes()
    path.write_bytes(original)
    original_snapshot = relation.read_manifest_snapshot(path)
    fixture = _training()

    class Rebuilt:
        def of(self, stratum):
            assert stratum == relation.STRATUM_TRAIN
            return fixture.cases

    def replacement_build(config):
        path.write_text("[]")
        return Rebuilt()

    monkeypatch.setattr(train_module.corpus, "build", replacement_build)
    monkeypatch.setattr(train_module.corpus, "manifest", lambda _: original_snapshot.body)
    loaded = load_training_corpus(path)
    assert path.read_text() == "[]"
    assert loaded.file_sha256 == original_snapshot.file_sha256
    assert loaded.canonical_payload_sha256 == original_snapshot.canonical_payload_sha256
    assert loaded.manifest_path == path


def test_supervision_identity_binds_an_unconsumed_action_only_change(tmp_path):
    training = _training()
    trainer = _trainer(training)
    path = tmp_path / "step-zero.pt"
    trainer.save(path, sources=AUTHORITATIVE_R4_SOURCES)
    changed = TrainingCorpus.engineering_cases(
        (replace(training.cases[0], actions=()), *training.cases[1:])
    )
    assert changed.program_fingerprint == training.program_fingerprint
    assert changed.supervision_identity != training.supervision_identity
    with pytest.raises(RelationTrainingRefused, match="corpus identity"):
        RelationTrainer.load(path, training=changed, sources=AUTHORITATIVE_R4_SOURCES)


def test_checkpoint_refuses_runtime_drift_nonfinite_state_and_recipe_drift(tmp_path):
    training = _training()
    runtime = _trainer(training)
    path = tmp_path / "rejected.pt"
    torch.use_deterministic_algorithms(False)
    try:
        with pytest.raises(RelationTrainingRefused, match="runtime"):
            runtime.save(path, sources=AUTHORITATIVE_R4_SOURCES)
    finally:
        torch.use_deterministic_algorithms(True)

    nonfinite = _trainer(training)
    with torch.no_grad():
        next(nonfinite.model.parameters()).flatten()[0] = float("nan")
    with pytest.raises(RelationTrainingRefused, match="NaN or infinity"):
        nonfinite.save(path, sources=AUTHORITATIVE_R4_SOURCES)

    recipe = _trainer(training)
    recipe.optimizer.param_groups[0]["weight_decay"] = 0.75
    with pytest.raises(RelationTrainingRefused, match="recipe"):
        recipe.save(path, sources=AUTHORITATIVE_R4_SOURCES)


def test_checkpoint_binds_process_source_snapshot_and_decoder_dependency(tmp_path, monkeypatch):
    trainer = _trainer()
    assert any(path.name == "decoding.py" for path in AUTHORITATIVE_R4_SOURCES)
    changed = dict(trainer._process_source_hashes)
    changed[next(iter(changed))] = "0" * 64
    monkeypatch.setattr(train_module, "source_hashes", lambda: changed)
    with pytest.raises(RelationTrainingRefused, match="source contents changed"):
        trainer.save(tmp_path / "drift.pt", sources=AUTHORITATIVE_R4_SOURCES)


@pytest.mark.parametrize("mutation", ["model_nan", "weight_decay", "parameter_remap", "moment_nan"])
def test_rehashed_invalid_checkpoint_state_refuses_before_restoration(tmp_path, mutation):
    training = _training()
    trainer = _trainer(training)
    trainer.step()
    path = tmp_path / "tampered.pt"
    trainer.save(path, sources=AUTHORITATIVE_R4_SOURCES)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if mutation == "model_nan":
        payload["model"][next(iter(payload["model"]))].flatten()[0] = float("nan")
        payload["content_digests"]["model"] = state_digest(payload["model"])
    elif mutation == "weight_decay":
        payload["optimizer"]["param_groups"][0]["weight_decay"] = 0.75
    elif mutation == "parameter_remap":
        payload["optimizer"]["param_groups"][0]["params"].reverse()
    else:
        payload["optimizer"]["state"][0]["exp_avg"].flatten()[0] = float("nan")
    torch.save(payload, path)

    torch.manual_seed(987)
    target = DrawingLM(Config(**MODEL))
    target_optimizer = torch.optim.AdamW(
        target.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.01
    )
    before = (
        state_digest(target.state_dict()),
        optimizer_digest(target_optimizer, target),
        rng_digest("cpu"),
    )
    with pytest.raises(RelationTrainingRefused):
        load_training_bundle(path, model=target, optimizer=target_optimizer,
                             data_generator=None, training=training,
                             sources=AUTHORITATIVE_R4_SOURCES)
    after = (
        state_digest(target.state_dict()),
        optimizer_digest(target_optimizer, target),
        rng_digest("cpu"),
    )
    assert after == before


def test_failed_checkpoint_publication_cleans_staging_and_preserves_destination(tmp_path, monkeypatch):
    trainer = _trainer()
    path = tmp_path / "existing.pt"
    path.write_bytes(b"keep this valid destination")

    def fail_replace(source, destination):
        raise OSError("injected replacement failure")

    monkeypatch.setattr(train_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replacement failure"):
        trainer.save(path, sources=AUTHORITATIVE_R4_SOURCES)
    assert path.read_bytes() == b"keep this valid destination"
    assert not list(tmp_path.glob(".existing.pt.*.tmp"))


def test_schema_two_bundle_is_refused_with_its_missing_binding_reason(tmp_path):
    trainer = _trainer()
    path = tmp_path / "old-schema.pt"
    trainer.save(path, sources=AUTHORITATIVE_R4_SOURCES)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["schema"] = 2
    torch.save(payload, path)
    with pytest.raises(RelationTrainingRefused, match="schema 2.*corpus/process binding"):
        RelationTrainer.load(path, training=trainer.training, sources=AUTHORITATIVE_R4_SOURCES)
