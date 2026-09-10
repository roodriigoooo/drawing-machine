"""R4.2/R4.3 adapter, packed ownership and objective tests."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace

import pytest
import torch

from dm.data import relation
from dm.isa.codec import PAD, ByteCodec
from dm.models.transformer import Config, DrawingLM
from dm.train import lr_at
from dm.train_relation import (
    AUTHORITATIVE_R4_SOURCES,
    RelationAccounting,
    RelationBucketCursor,
    RelationTrainConfig,
    RelationTrainer,
    RelationTrainingRefused,
    TrainingCorpus,
    TrunkIdentity,
    build_batch_plan,
    load_training_bundle,
    pack_candidates,
    relation_loss,
    relation_train_step,
    save_training_bundle,
)


def _training() -> TrainingCorpus:
    build = relation.build_venue1(
        8, seed=41, motifs=relation.motif_pool(16, seed=40),
        tuples=relation.venue1_tuples(),
    )
    return TrainingCorpus.engineering_cases(build.cases)


def test_batch_plan_keeps_indices_bound_to_rows_and_targets_only_trainable_boundaries():
    training = _training()
    plan = build_batch_plan(training, [3, 1], max_len=256)
    assert plan.case_indices.tolist() == [3, 1]
    for query in plan.queries:
        case = training.cases[query.case_index]
        assert query.boundary < len(case.flat)  # post-HALT is never queried
        assert query.boundary not in relation.covered_boundaries(case.actions)
        if query.gate_target:
            assert query.equivalents
            assert all((item.source_start, item.source_stop) in query.spans
                       for item in query.equivalents)
        else:
            assert not query.equivalents


def test_packed_relation_rows_are_ragged_query_local_and_differentiable():
    training = _training()
    plan = build_batch_plan(training, [0, 2], max_len=256)
    torch.manual_seed(3)
    model = DrawingLM(Config(vocab_size=ByteCodec.vocab_size, d_model=16, n_layers=1,
                             n_heads=2, max_len=256, relation_schema="span_affine_v1"))
    logits, states = model(plan.inputs, return_state=True)
    packed, equivalents, gates = pack_candidates(plan, states)
    scores = model.score_relation(packed)
    loss = relation_loss(logits, plan, scores=scores, equivalents=equivalents,
                         gate_targets=gates)
    loss.total.backward()
    assert loss.content_bytes == int((plan.targets != PAD).sum())
    assert loss.reachable_boundaries == len(plan.queries)
    assert loss.positive_boundaries == len(equivalents)
    assert packed.row_splits[-1] == packed.spans
    assert model.relation is not None
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.relation.parameters())


def test_none_arm_is_exactly_byte_only_and_padding_has_no_loss_term():
    training = _training()
    plan = build_batch_plan(training, [0, 1], max_len=256)
    torch.manual_seed(7)
    model = DrawingLM(Config(vocab_size=ByteCodec.vocab_size, d_model=16, n_layers=1,
                             n_heads=2, max_len=256))
    logits = model(plan.inputs)
    loss = relation_loss(logits, plan)
    assert loss.gate_sum.item() == loss.joint_sum.item() == 0.0
    assert torch.equal(loss.total, loss.byte)
    changed = logits.detach().clone()
    changed[plan.targets == PAD] = torch.randn_like(changed[plan.targets == PAD])
    assert torch.equal(relation_loss(changed, plan).byte_sum, loss.byte_sum)


def test_microbatch_accumulation_preserves_fixed_denominator_losses_and_updates():
    training = _training()
    plan = build_batch_plan(training, [0, 1, 2, 3], max_len=256)
    args = {"vocab_size": ByteCodec.vocab_size, "d_model": 16, "n_layers": 1,
            "n_heads": 2, "max_len": 256, "relation_schema": "span_affine_v1"}
    torch.manual_seed(88)
    whole = DrawingLM(Config(**args))
    split = DrawingLM(Config(**args))
    split.load_state_dict(whole.state_dict())
    whole_opt = torch.optim.AdamW(whole.parameters(), lr=1e-3)
    split_opt = torch.optim.AdamW(split.parameters(), lr=1e-3)
    whole_loss = relation_train_step(whole, whole_opt, plan, grad_clip=1.0)
    split_loss = relation_train_step(split, split_opt, plan, grad_clip=1.0,
                                     attention_budget=plan.inputs.shape[1] ** 2)
    assert torch.allclose(whole_loss.total, split_loss.total, atol=1e-6, rtol=1e-6)
    assert all(torch.allclose(left, right, atol=1e-6, rtol=1e-6)
               for left, right in zip(whole.state_dict().values(), split.state_dict().values(), strict=True))


def test_temporary_bundle_restores_model_optimizer_rng_and_accounting(tmp_path):
    training = _training()
    torch.use_deterministic_algorithms(True)
    model_args = {"vocab_size": ByteCodec.vocab_size, "d_model": 16, "n_layers": 1,
                  "n_heads": 2, "max_len": 256, "relation_schema": "span_affine_v1"}
    config = RelationTrainConfig("span_affine_v1", model_args, seed=9, batch_size=2,
                                 max_len=256, steps=7, warmup=2,
                                 corpus_provenance="engineering", seed_namespace="engineering")
    from dm.train_relation import _environment, source_hashes
    process_environment, process_source_hashes = _environment(config), source_hashes()
    sampler = RelationBucketCursor.seeded(
        [len(case.flat) + 1 for case in training.cases], batch_size=config.batch_size,
        base_seed=config.seed,
    )
    indices = sampler.next()
    plan = build_batch_plan(training, indices, max_len=256)
    torch.manual_seed(9)
    model = DrawingLM(Config(**model_args))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_at(0, config),
                                 weight_decay=config.weight_decay,
                                 betas=config.adam_betas, eps=config.adam_eps,
                                 amsgrad=config.adam_amsgrad, foreach=config.adam_foreach,
                                 fused=config.adam_fused, capturable=config.adam_capturable,
                                 differentiable=config.adam_differentiable,
                                 maximize=config.adam_maximize)
    trunk = TrunkIdentity.of(model)
    loss = relation_train_step(model, optimizer, plan, grad_clip=config.grad_clip)
    accounting = RelationAccounting()
    assert loss.unclipped_grad_norm is not None and loss.clipped_grad_norm is not None
    accounting.observe(plan, loss, unclipped_norm=loss.unclipped_grad_norm,
                       clipped_norm=loss.clipped_grad_norm)
    assert loss.factors is not None
    history = [{
        "step": 1,
        "batch_ids": indices,
        "lr": lr_at(0, config),
        "byte_sum": float(loss.byte_sum.detach()),
        "gate_sum": float(loss.gate_sum.detach()),
        "joint_sum": float(loss.joint_sum.detach()),
        "content_bytes": loss.content_bytes,
        "reachable_boundaries": loss.reachable_boundaries,
        "positive_boundaries": loss.positive_boundaries,
        "factors": loss.factors.as_dict(),
    }]
    path = tmp_path / "r4-test.pt"
    diagnostics = {"component_policy": "explicit_override", "component_events": [],
                   "trunk": trunk.as_dict(), "component_records": []}
    config = replace(config, component_gradient_events=())
    save_training_bundle(path, config=config, completed_step=1, model=model,
                         optimizer=optimizer, sampler=sampler.state_dict(),
                         accounting=accounting, history=history, training=training,
                         data_generator=sampler.generator, sources=AUTHORITATIVE_R4_SOURCES,
                         diagnostics=diagnostics, process_environment=process_environment,
                         process_source_hashes=process_source_hashes)
    torch.manual_seed(111)
    restored_model = DrawingLM(Config(**model_args))
    restored_optimizer = torch.optim.AdamW(
        restored_model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
        betas=config.adam_betas, eps=config.adam_eps, amsgrad=config.adam_amsgrad,
        foreach=config.adam_foreach, fused=config.adam_fused,
        capturable=config.adam_capturable, differentiable=config.adam_differentiable,
        maximize=config.adam_maximize)
    restored_generator = torch.Generator().manual_seed(222)
    bundle = load_training_bundle(path, model=restored_model, optimizer=restored_optimizer,
                                  data_generator=restored_generator, training=training,
                                  sources=AUTHORITATIVE_R4_SOURCES)
    assert bundle["completed_step"] == 1
    assert all(torch.equal(left, right) for left, right in zip(
        model.state_dict().values(), restored_model.state_dict().values(), strict=True))


def test_non_epoch_bundle_resume_replays_the_exact_next_batches(tmp_path):
    training = _training()
    model_args = {"vocab_size": ByteCodec.vocab_size, "d_model": 16, "n_layers": 1,
                  "n_heads": 2, "max_len": 256, "relation_schema": "span_affine_v1"}
    config = RelationTrainConfig("span_affine_v1", model_args, seed=31, batch_size=3,
                                 max_len=256, steps=7, warmup=2,
                                 corpus_provenance="engineering", seed_namespace="engineering")
    uninterrupted = RelationTrainer(config, training)
    for _ in range(3):
        uninterrupted.step()
    interrupted = RelationTrainer(config, training)
    interrupted.step()  # deliberately in the middle of a three-batch epoch
    bundle = tmp_path / "resume.pt"
    interrupted.save(bundle, sources=AUTHORITATIVE_R4_SOURCES)
    resumed = RelationTrainer.load(bundle, training=training, sources=AUTHORITATIVE_R4_SOURCES)
    for _ in range(2):
        resumed.step()
    assert resumed.history == uninterrupted.history
    assert resumed.accounting == uninterrupted.accounting
    assert all(torch.equal(left, right) for left, right in zip(
        resumed.model.state_dict().values(), uninterrupted.model.state_dict().values(), strict=True))


def test_resumed_bucket_cursor_reconciles_exactly_with_the_configured_corpus():
    cursor = RelationBucketCursor.seeded([7, 9, 11, 13], batch_size=3, base_seed=5)
    cursor.next()
    state = cursor.state_dict()
    state["batch_size"] = 999
    state["lengths"] = [1, 1, 1, 1]
    state["batches"] = [[0]]
    with pytest.raises(ValueError, match="disagrees"):
        RelationBucketCursor.from_state(state, lengths=[7, 9, 11, 13], batch_size=3, base_seed=5)


def test_bundle_load_refuses_a_tampered_sampler_before_it_can_replay(tmp_path):
    training = _training()
    model_args = {"vocab_size": ByteCodec.vocab_size, "d_model": 16, "n_layers": 1,
                  "n_heads": 2, "max_len": 256, "relation_schema": "span_affine_v1"}
    config = RelationTrainConfig("span_affine_v1", model_args, seed=31, batch_size=3,
                                 max_len=256, steps=7, warmup=2,
                                 corpus_provenance="engineering", seed_namespace="engineering")
    trainer = RelationTrainer(config, training)
    trainer.step()
    path = tmp_path / "tampered-sampler.pt"
    trainer.save(path, sources=AUTHORITATIVE_R4_SOURCES)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["sampler"]["batch_size"] = 999
    payload["sampler"]["lengths"] = [1] * len(training.cases)
    payload["sampler"]["batches"] = [[0]]
    torch.save(payload, path)
    with pytest.raises(RelationTrainingRefused, match="sampler"):
        RelationTrainer.load(path, training=training, sources=AUTHORITATIVE_R4_SOURCES)


def test_authoritative_source_list_covers_runtime_action_and_byte_semantics():
    root = AUTHORITATIVE_R4_SOURCES[0].parents[1]
    names = {path.relative_to(root).as_posix() for path in AUTHORITATIVE_R4_SOURCES}
    assert {"dm/relation/spec.py", "dm/relation/transducer.py", "dm/isa/codec.py"} <= names


def test_fresh_process_non_epoch_resume_reproduces_final_bundle_exactly(tmp_path):
    training = _training()
    model_args = {"vocab_size": ByteCodec.vocab_size, "d_model": 16, "n_layers": 1,
                  "n_heads": 2, "max_len": 256, "relation_schema": "span_affine_v1"}
    config = RelationTrainConfig("span_affine_v1", model_args, seed=31, batch_size=3,
                                 max_len=256, steps=7, warmup=2,
                                 corpus_provenance="engineering", seed_namespace="engineering")
    uninterrupted = RelationTrainer(config, training)
    for _ in range(3):
        uninterrupted.step()
    expected = tmp_path / "expected.pt"
    uninterrupted.save(expected, sources=AUTHORITATIVE_R4_SOURCES)

    interrupted = RelationTrainer(config, training)
    interrupted.step()
    checkpoint = tmp_path / "checkpoint.pt"
    resumed = tmp_path / "resumed.pt"
    interrupted.save(checkpoint, sources=AUTHORITATIVE_R4_SOURCES)
    script = """
from pathlib import Path
import sys
from dm.data import relation
from dm.isa.codec import ByteCodec
from dm.train_relation import AUTHORITATIVE_R4_SOURCES, RelationTrainConfig, RelationTrainer, TrainingCorpus

build = relation.build_venue1(8, seed=41, motifs=relation.motif_pool(16, seed=40), tuples=relation.venue1_tuples())
training = TrainingCorpus.engineering_cases(build.cases)
model = {\"vocab_size\": ByteCodec.vocab_size, \"d_model\": 16, \"n_layers\": 1, \"n_heads\": 2, \"max_len\": 256, \"relation_schema\": \"span_affine_v1\"}
config = RelationTrainConfig(\"span_affine_v1\", model, seed=31, batch_size=3, max_len=256, steps=7, warmup=2, corpus_provenance=\"engineering\", seed_namespace=\"engineering\")
trainer = RelationTrainer.load(Path(sys.argv[1]), training=training, sources=AUTHORITATIVE_R4_SOURCES)
for _ in range(2):
    trainer.step()
trainer.save(Path(sys.argv[2]), sources=AUTHORITATIVE_R4_SOURCES)
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = str(AUTHORITATIVE_R4_SOURCES[0].parents[1])
    result = subprocess.run([sys.executable, "-c", script, str(checkpoint), str(resumed)],
                            text=True, capture_output=True, check=False, env=env)
    assert result.returncode == 0, result.stderr
    left = torch.load(expected, map_location="cpu", weights_only=False)
    right = torch.load(resumed, map_location="cpu", weights_only=False)
    for key in ("accounting", "history", "content_digests"):
        assert left[key] == right[key]
    assert {key: value for key, value in left["sampler"].items() if key != "generator"} == \
           {key: value for key, value in right["sampler"].items() if key != "generator"}
    assert torch.equal(left["sampler"]["generator"], right["sampler"]["generator"])


def test_relation_train_config_refuses_dropout_or_non_byte_model_regimes():
    base = {"vocab_size": ByteCodec.vocab_size, "d_model": 16, "n_layers": 1,
            "n_heads": 2, "max_len": 256, "relation_schema": "none"}
    with pytest.raises(ValueError, match="dropout-free"):
        RelationTrainConfig("none", {**base, "dropout": 0.2}, seed=1, batch_size=2,
                            max_len=256, steps=4, warmup=1,
                            corpus_provenance="engineering", seed_namespace="engineering")
    with pytest.raises(ValueError, match="max_len"):
        RelationTrainConfig("none", base, seed=1, batch_size=2, max_len=128, steps=4, warmup=1,
                            corpus_provenance="engineering", seed_namespace="engineering")
