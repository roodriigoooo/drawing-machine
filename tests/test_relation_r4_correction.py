"""Adversarial R4 boundaries and independent arithmetic/caching witnesses."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from dm.data import relation
from dm.eval.relation_evidence import optimizer_digest
from dm.isa.codec import PAD, ByteCodec
from dm.models.transformer import Config, DrawingLM
from dm.train import lr_at
from dm.train_relation import (
    AUTHORITATIVE_R4_SOURCES,
    RelationBatchPlanner,
    RelationTrainConfig,
    RelationTrainer,
    RelationTrainingRefused,
    TrainingCorpus,
    build_batch_plan,
    pack_candidates,
    relation_loss,
    relation_train_step,
)


@pytest.fixture
def training():
    built = relation.build_venue1(
        8, seed=41, motifs=relation.motif_pool(16, seed=40), tuples=relation.venue1_tuples())
    return TrainingCorpus.engineering_cases(built.cases)


def _config(arm="span_affine_v1", **changes):
    model = {"vocab_size": ByteCodec.vocab_size, "d_model": 16, "n_layers": 1,
             "n_heads": 2, "max_len": 256, "relation_schema": arm}
    return RelationTrainConfig(arm, model, seed=31, batch_size=3, max_len=256,
                               **{"steps": 7, "warmup": 2, "corpus_provenance": "engineering",
                                  "seed_namespace": "engineering", **changes})


@pytest.mark.parametrize("mutation", ["future", "operand", "duplicate_span", "duplicate_query",
                                      "negative_row", "duplicate_action", "bad_action"])
def test_plan_refuses_semantic_corruption_before_gather(training, mutation):
    plan = build_batch_plan(training, [0, 1], max_len=256)
    queries = list(plan.queries)
    index = next(i for i, query in enumerate(queries) if query.equivalents)
    query = queries[index]
    if mutation == "future":
        queries[index] = replace(query, spans=((0, query.boundary + 3),))
    elif mutation == "operand":
        queries[index] = replace(query, spans=((1, query.boundary),))
    elif mutation == "duplicate_span":
        queries[index] = replace(query, spans=query.spans + query.spans[:1])
    elif mutation == "duplicate_query":
        queries.append(query)
    elif mutation == "negative_row":
        queries[index] = replace(query, row=-1)
    elif mutation == "duplicate_action":
        queries[index] = replace(query, equivalents=query.equivalents + query.equivalents[:1])
    else:
        queries[index] = replace(query, equivalents=(replace(query.equivalents[0], d4=True),))
    malformed = replace(plan, queries=tuple(queries))
    with pytest.raises(RelationTrainingRefused):
        pack_candidates(malformed, torch.zeros(*plan.inputs.shape, 16))


def test_loss_refuses_reassigned_gates_or_positive_queries(training):
    plan = build_batch_plan(training, [0, 1], max_len=256)
    model = DrawingLM(Config(**_config().model))
    logits, states = model(plan.inputs, return_state=True)
    packed, equivalents, gates = pack_candidates(plan, states)
    scores = model.score_relation(packed)
    with pytest.raises(RelationTrainingRefused):
        relation_loss(logits, plan, scores=scores, equivalents=equivalents,
                      gate_targets=1 - gates)
    assert len(equivalents) >= 2
    with pytest.raises(RelationTrainingRefused):
        relation_loss(logits, plan, scores=scores,
                      equivalents=(equivalents[0],) * len(equivalents), gate_targets=gates)


def test_no_positive_batch_has_byte_and_gate_loss_and_no_joint_denominator(training):
    negative = replace(training.cases[0], actions=())
    plan = build_batch_plan(TrainingCorpus.engineering_cases([negative]), [0], max_len=256)
    model = DrawingLM(Config(**_config().model))
    logits, states = model(plan.inputs, return_state=True)
    packed, equivalents, gates = pack_candidates(plan, states)
    loss = relation_loss(logits, plan, scores=model.score_relation(packed),
                         equivalents=equivalents, gate_targets=gates)
    assert loss.positive_boundaries == 0
    assert loss.joint_sum.item() == loss.action_joint.item() == 0
    assert torch.isfinite(loss.total)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    result = relation_train_step(model, optimizer, plan, grad_clip=1)
    assert torch.isfinite(result.total)


def test_packed_gather_has_constant_graph_size_and_exact_endpoint_gradients(training):
    plan = build_batch_plan(training, [0, 1], max_len=256)
    states = torch.randn(*plan.inputs.shape, 16, requires_grad=True)
    packed, _, _ = pack_candidates(plan, states)
    expected = torch.zeros_like(states)
    for query in plan.queries:
        expected[query.row, query.boundary] += 1
        for start, stop in query.spans:
            expected[query.row, start] += 2
            expected[query.row, stop] += 3
    total = packed.query_states.sum() + 2 * packed.start_states.sum() + 3 * packed.stop_states.sum()
    pending, visited = [total.grad_fn], set()
    while pending:
        node = pending.pop()
        if node is not None and node not in visited:
            visited.add(node)
            pending.extend(child for child, _ in node.next_functions)
    assert len(visited) <= 16  # a bounded gather graph, independent of span count
    total.backward()
    assert torch.equal(states.grad, expected)


def test_trainer_reuses_case_derivations_after_first_epoch(training, monkeypatch):
    trainer = RelationTrainer(_config(), training)
    for _ in range(3):
        trainer.step()
    def unexpected(*args, **kwargs):
        pytest.fail("an immutable case's derivations were rebuilt in a later epoch")
    monkeypatch.setattr(relation, "derivations", unexpected)
    for _ in range(3):
        trainer.step()


def test_metadata_retention_is_bounded_and_eviction_preserves_exact_batches(training):
    planner = RelationBatchPlanner(training, max_len=256, max_cached_cases=1, max_cached_spans=400)
    for indices in ([0, 1, 2], [3, 2, 0], [0, 3, 1]):
        expected = build_batch_plan(training, indices, max_len=256)
        actual = planner.batch(indices)
        assert actual.queries == expected.queries
        assert torch.equal(actual.inputs, expected.inputs)
        assert torch.equal(actual.targets, expected.targets)
        assert planner.cached_spans <= 400 and len(planner._cases) <= 1


def test_none_step_matches_historical_byte_body_exactly(training):
    plan = build_batch_plan(training, [0, 1, 2], max_len=256)
    trainer = RelationTrainer(_config("none"), training)
    legacy = DrawingLM(Config(**trainer.config.model))
    legacy.load_state_dict(trainer.model.state_dict())
    # Same LR as the trainer's first update: the historical body sets
    # `group["lr"] = lr_at(step, cfg)` before every step.
    old_opt = torch.optim.AdamW(legacy.parameters(), lr=lr_at(0, trainer.config),
                               weight_decay=trainer.config.weight_decay, betas=(0.9, 0.95))
    old_opt.zero_grad(set_to_none=True)
    logits = legacy(plan.inputs)
    nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), plan.targets.reshape(-1),
                          ignore_index=PAD, reduction="sum")
    (nll / (plan.targets != PAD).sum().clamp(min=1)).backward()
    torch.nn.utils.clip_grad_norm_(legacy.parameters(), trainer.config.grad_clip)
    old_opt.step()
    result = relation_train_step(trainer.model, trainer.optimizer, plan,
                                 grad_clip=trainer.config.grad_clip)
    assert torch.equal(result.byte_sum, nll.detach())
    for old, new in zip(legacy.parameters(), trainer.model.parameters(), strict=True):
        assert torch.equal(old.grad, new.grad)
        assert torch.equal(old, new)
    assert optimizer_digest(old_opt, legacy) == optimizer_digest(trainer.optimizer, trainer.model)


@pytest.mark.parametrize("mutation", ["sampler_rng", "history_sum", "accounting_sum", "nan",
                                      "step_type", "rng_missing", "cursor"])
def test_bundle_reconciles_continuation_evidence(training, tmp_path, mutation):
    trainer = RelationTrainer(_config(), training)
    trainer.step()
    path = tmp_path / "fixture.pt"
    trainer.save(path, sources=AUTHORITATIVE_R4_SOURCES)
    payload = torch.load(path, weights_only=False)
    if mutation == "sampler_rng":
        payload["sampler"]["generator"] = torch.Generator().manual_seed(99).get_state()
    elif mutation == "history_sum":
        payload["history"][0]["byte_sum"] += 10
    elif mutation == "accounting_sum":
        payload["accounting"]["semantic_bytes"] += 1
    elif mutation == "nan":
        payload["accounting"]["joint_sum"] = float("nan")
    elif mutation == "step_type":
        payload["history"][0]["step"] = True
    elif mutation == "rng_missing":
        payload["rng"]["data_generator"] = None
    else:
        payload["sampler"]["cursor"] = 2
        payload["completed_step"] = 2
    torch.save(payload, path)
    with pytest.raises(RelationTrainingRefused):
        RelationTrainer.load(path, training=training, sources=AUTHORITATIVE_R4_SOURCES)


@pytest.mark.parametrize("field,value", [("lr", float("nan")), ("lr", -1),
                                         ("weight_decay", float("inf")),
                                         ("grad_clip", 0), ("deterministic", False)])
def test_configuration_rejects_invalid_numeric_or_determinism_policy(field, value):
    with pytest.raises(RelationTrainingRefused):
        _config(**{field: value})


def test_builtin_evidence_validator_defect_is_not_an_expected_refusal(training, tmp_path, monkeypatch):
    import dm.train_relation as module
    trainer = RelationTrainer(_config(), training)
    trainer.step()
    path = tmp_path / "fixture.pt"
    trainer.save(path, sources=AUTHORITATIVE_R4_SOURCES)
    def defect(*args):
        raise ValueError("injected implementation defect")
    monkeypatch.setattr(module, "validate_bundle_shape", defect)
    with pytest.raises(ValueError, match="injected implementation defect") as raised:
        RelationTrainer.load(path, training=training, sources=AUTHORITATIVE_R4_SOURCES)
    assert type(raised.value) is ValueError
