"""R4 remaining points 1-3: charged retention, paired schedule/order/continuation, diagnostics.

Test IDs follow docs/copy-relation-r4-training-plan.md (P1-Txx, P2-Txx, P3-Txx).
Every witness here is an engineering result about the trainer; none is
evidence that a model learns anything.
"""

from __future__ import annotations

import copy
import gc
import json
import math
import os
import random
import subprocess
import sys
import threading
import weakref
from dataclasses import replace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

import dm.train_relation as module
from dm.data import relation
from dm.data.dataset import (
    BUCKET_EPOCH_ALGORITHM,
    BucketedBatchSampler,
    ProgramDataset,
    bucket_epoch_plan,
    loader,
)
from dm.eval.relation_contract import seed_for
from dm.eval.relation_evidence import optimizer_digest, rng_digest, state_digest
from dm.eval.relation_training_evidence import (
    COMPONENT_NAMES,
    FACTOR_ALLOWANCE_PER_QUERY,
    RelationTrainingEvidenceRefused,
    default_component_events,
    validate_bundle_shape,
)
from dm.isa.codec import PAD, ByteCodec
from dm.models.relation import (
    COUNT_SUPPORT,
    FACTOR_NAMES,
    EquivalentAction,
    PackedCandidates,
    RelationLayoutError,
    RelationScores,
    factor_marginal_nll,
    joint_valid_nll,
    relation_parameter_count,
)
from dm.models.transformer import Config, DrawingLM
from dm.relation import resources
from dm.train import lr_at, micro_batches
from dm.train_relation import (
    AUTHORITATIVE_R4_SOURCES,
    ActionTarget,
    CaseMetadata,
    CaseQuery,
    RelationBatchPlanner,
    RelationBucketCursor,
    RelationTrainConfig,
    RelationTrainer,
    RelationTrainingComplete,
    RelationTrainingRefused,
    TrainingCorpus,
    TrunkIdentity,
    _ComponentAccumulator,
    _microbatch_rows,
    build_batch_plan,
    capture_rng_state,
    pack_candidates,
    relation_loss,
    relation_train_step,
    restore_rng_state,
    retained_charge,
)

TOL = {"atol": 1e-6, "rtol": 1e-6}
MODEL = {"vocab_size": ByteCodec.vocab_size, "d_model": 16, "n_layers": 1, "n_heads": 2,
         "max_len": 256}


def _cases(seed: int = 41):
    return relation.build_venue1(8, seed=seed, motifs=relation.motif_pool(16, seed=seed - 1),
                                 tuples=relation.venue1_tuples()).cases


@pytest.fixture(scope="module")
def training() -> TrainingCorpus:
    return TrainingCorpus.engineering_cases(_cases())


def _config(arm: str = "span_affine_v1", **changes) -> RelationTrainConfig:
    fields = {"seed": 31, "batch_size": 3, "max_len": 256, "steps": 7, "warmup": 2,
              "corpus_provenance": "engineering", "seed_namespace": "engineering"}
    fields.update(changes)
    return RelationTrainConfig(arm, {**MODEL, "relation_schema": arm}, **fields)


def _plan_equal(left, right) -> None:
    assert left.queries == right.queries
    assert torch.equal(left.case_indices, right.case_indices)
    assert torch.equal(left.inputs, right.inputs)
    assert torch.equal(left.targets, right.targets)
    assert (left.content_bytes, left.reachable_boundaries, left.positive_boundaries) == \
           (right.content_bytes, right.reachable_boundaries, right.positive_boundaries)


def _digests(trainer: RelationTrainer) -> tuple[str, str, str]:
    return (state_digest(trainer.model.state_dict()),
            optimizer_digest(trainer.optimizer, trainer.model),
            rng_digest("cpu", trainer.cursor.generator))


def _assert_optimizers_equal(left: torch.optim.Optimizer, right: torch.optim.Optimizer) -> None:
    s1, s2 = left.state_dict(), right.state_dict()
    assert s1["param_groups"] == s2["param_groups"]
    assert set(s1["state"]) == set(s2["state"])
    for k in s1["state"]:
        for item_key in s1["state"][k]:
            v1, v2 = s1["state"][k][item_key], s2["state"][k][item_key]
            if isinstance(v1, torch.Tensor):
                assert torch.equal(v1, v2)
            else:
                assert v1 == v2


def _run(config: RelationTrainConfig, training: TrainingCorpus, steps: int) -> RelationTrainer:
    trainer = RelationTrainer(config, training)
    for _ in range(steps):
        trainer.step()
    return trainer


@pytest.mark.parametrize("arm", ["none", "span_affine_v1"])
def test_resource_observer_preserves_exact_training(training, arm, tmp_path):
    config = _config(arm, steps=2, warmup=0)
    initial_rng = capture_rng_state()
    plain = _run(config, training, 2)
    plain_digests = _digests(plain)
    restore_rng_state(initial_rng)
    records = []
    measured = RelationTrainer(config, training, resource_observer=records.append)
    for _ in range(2):
        measured.step()
    assert _digests(measured) == plain_digests
    assert measured.history == plain.history
    assert measured.accounting == plain.accounting
    assert measured.diagnostics() == plain.diagnostics()
    assert [record.phase for record in records] == ["metadata", "train_update", "step"] * 2
    assert all(record.completed and record.wall_seconds >= 0 for record in records)
    assert all(record.samples == 0 and record.sample_interval_seconds is None
               for record in records)
    # The enclosing scope covers both inner scopes and their probe overhead.
    for base in (0, 3):
        metadata, update, whole = records[base:base + 3]
        assert whole.wall_seconds >= metadata.wall_seconds + update.wall_seconds
    path = tmp_path / "physical-not-resumed.pt"
    measured.save(path, sources=AUTHORITATIVE_R4_SOURCES)
    restored = RelationTrainer.load(path, training=training, sources=AUTHORITATIVE_R4_SOURCES)
    assert restored.resource_observer is None
    assert restored.resource_sample_interval is None
    assert restored.history == measured.history


@pytest.mark.parametrize("arm", ["none", "span_affine_v1"])
def test_resource_sampling_preserves_exact_training(training, arm):
    """The in-phase peak sampler is a thread, not a change to the trajectory."""
    config = _config(arm, steps=2, warmup=0)
    initial_rng = capture_rng_state()
    plain = _run(config, training, 2)
    plain_digests = _digests(plain)
    restore_rng_state(initial_rng)
    threads_before = threading.active_count()
    ledger = resources.PhaseLedger({"step": ("metadata", "train_update")})
    measured = RelationTrainer(config, training, resource_observer=ledger,
                               resource_sample_interval=0.001)
    for _ in range(2):
        measured.step()
    assert _digests(measured) == plain_digests
    assert measured.history == plain.history
    assert measured.accounting == plain.accounting
    assert measured.diagnostics() == plain.diagnostics()
    assert threading.active_count() == threads_before
    totals = ledger.totals()
    assert set(totals) == {"metadata", "train_update", "step"}
    assert all(entry.records == 2 and entry.completed_records == 2
               for entry in totals.values())
    assert totals["step"].samples > 0
    assert ledger.unattributed_seconds("step", ["metadata", "train_update"]) >= 0
    for entry in totals.values():
        assert entry.peak_lower_bound_bytes <= entry.peak_upper_bound_bytes


def test_resource_probe_count_per_step_is_exact(training, monkeypatch):
    """Three scopes, two endpoint probes each, and nothing else per update."""
    real = resources._memory_bytes
    probes = []
    monkeypatch.setattr(resources, "_memory_bytes",
                        lambda: (probes.append(1), real())[1])
    records = []
    trainer = RelationTrainer(_config(steps=2, warmup=0), training,
                              resource_observer=records.append)
    trainer.step()
    assert len(probes) == 6
    trainer.step()
    assert len(probes) == 12
    assert len(records) == 6


def test_resource_sampling_without_a_collector_is_refused(training):
    with pytest.raises(RelationTrainingRefused, match="requires a collector"):
        RelationTrainer(_config(), training, resource_sample_interval=0.001)


@pytest.mark.parametrize("interval", [0, -0.5, True, "0.01", float("nan")])
def test_resource_sample_interval_domain_is_refused(training, interval):
    with pytest.raises(ValueError, match="sample interval"):
        RelationTrainer(_config(), training, resource_observer=[].append,
                        resource_sample_interval=interval)


def test_resource_collector_failure_poison_trainer(training):
    error = RuntimeError("resource collector failed")

    def observer(record):
        if record.phase == "train_update":
            raise error

    trainer = RelationTrainer(_config(), training, resource_observer=observer)
    with pytest.raises(RuntimeError) as caught:
        trainer.step()
    assert caught.value is error
    with pytest.raises(RelationTrainingRefused, match="unusable"):
        trainer.step()


def _objective_history(trainer: RelationTrainer) -> list[dict]:
    return [{key: value for key, value in row.items() if key != "factors"}
            for row in trainer.history]


# ---------------------------------------------------------------------------
# Point 1 -- bounded metadata retention
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("limit", ["max_cached_cases", "max_cached_spans", "max_cached_bytes"])
@pytest.mark.parametrize("value", [-1, True, 1.5, "8", None])
def test_p1_t01_invalid_limits_are_refused_by_planner_and_config(training, limit, value):
    with pytest.raises(RelationTrainingRefused):
        RelationBatchPlanner(training, max_len=256, **{limit: value})
    with pytest.raises(RelationTrainingRefused):
        _config(**{limit: value})


@pytest.mark.parametrize("limit", ["max_cached_cases", "max_cached_spans", "max_cached_bytes"])
def test_p1_t01_each_zero_limit_disables_retention_and_still_builds_the_batch(training, limit):
    planner = RelationBatchPlanner(training, max_len=256, **{limit: 0})
    assert not planner.enabled
    for indices in ([0, 1, 2], [3, 4], [0, 1, 2]):
        _plan_equal(planner.batch(indices), build_batch_plan(training, indices, max_len=256))
    snapshot = planner.snapshot()
    assert snapshot.resident_cases == snapshot.resident_spans == snapshot.retained_charge_bytes == 0
    assert snapshot.requested_cases == snapshot.misses == snapshot.disabled_bypasses == 8
    assert snapshot.hits == snapshot.admissions == snapshot.evictions == 0


def _independent_inventory(root: object) -> int:
    """A gc-based traversal that shares no code with `retained_charge`."""
    seen: set[int] = set()
    total = 0
    pending = [root]
    while pending:
        obj = pending.pop()
        if id(obj) in seen or isinstance(obj, (type, str)):
            continue
        seen.add(id(obj))
        total += sys.getsizeof(obj)
        pending.extend(gc.get_referents(obj))
    return total


def test_p1_t02_charge_bounds_an_independent_inventory_and_charges_equivalences(training):
    target = ActionTarget(0, 6, 1, 32, 0, 3)
    shared_spans = ((0, 6), (0, 12), (6, 12))
    many = tuple(ActionTarget(0, 6, d4, dx, dy, count) for d4 in range(8) for dx in (-32, 0, 32)
                 for dy in (-32, 0, 32) for count in COUNT_SUPPORT)
    queries = (CaseQuery(12, 1, shared_spans, (target,)), CaseQuery(18, 1, shared_spans, (target,)),
               CaseQuery(24, 1, ((0, 6),), many))
    entry = CaseMetadata(tuple(range(40)), queries)
    charge = retained_charge(entry)
    assert charge >= _independent_inventory(entry)
    lean = CaseMetadata(tuple(range(40)), (CaseQuery(24, 0, ((0, 6),), ()),))
    assert entry.spans == 7 and lean.spans == 1
    assert charge > retained_charge(lean) + sum(sys.getsizeof(item) for item in many)
    for bad in ([1, 2], torch.zeros(2), CaseMetadata((1,), (CaseQuery(1, 0, ([0, 1],), ()),))):
        with pytest.raises(RuntimeError, match="unchargeable"):
            retained_charge(bad)


def test_p1_t03_exact_admission_charge_admits_and_one_byte_less_bypasses(training):
    probe = RelationBatchPlanner(training, max_len=256)
    probe.batch([0])
    exact = probe.snapshot().retained_charge_bytes
    assert exact == probe.snapshot().high_water_charge_bytes > probe.snapshot().index_charge_bytes > 0
    admitting = RelationBatchPlanner(training, max_len=256, max_cached_bytes=exact)
    _plan_equal(admitting.batch([0]), build_batch_plan(training, [0], max_len=256))
    assert admitting.snapshot().resident_cases == 1
    assert admitting.snapshot().retained_charge_bytes == exact
    bypassing = RelationBatchPlanner(training, max_len=256, max_cached_bytes=exact - 1)
    _plan_equal(bypassing.batch([0]), build_batch_plan(training, [0], max_len=256))
    snapshot = bypassing.snapshot()
    assert snapshot.resident_cases == 0 and snapshot.oversize_bypasses == 1
    assert snapshot.retained_charge_bytes <= exact - 1


def _entry_charges(training) -> dict[int, int]:
    charges = {}
    for index in range(len(training.cases)):
        planner = RelationBatchPlanner(training, max_len=256)
        planner.batch([index])
        charges[index] = planner.snapshot().retained_charge_bytes - planner.snapshot().index_charge_bytes
    return charges


def test_p1_t04_each_guard_alone_can_evict_and_all_hold_after_every_lookup(training):
    spans = {index: sum(len(q.spans) for q in build_batch_plan(training, [index], max_len=256).queries)
             for index in range(8)}
    charges = _entry_charges(training)
    planners = {
        "cases": RelationBatchPlanner(training, max_len=256, max_cached_cases=2),
        "spans": RelationBatchPlanner(training, max_len=256, max_cached_spans=max(spans.values()) + 1),
        "bytes": RelationBatchPlanner(training, max_len=256,
                                      max_cached_bytes=max(charges.values()) + min(charges.values()) + 2**12),
    }
    for name, planner in planners.items():
        for indices in ([0, 1, 2], [3, 4, 5], [6, 7, 0], [1, 2, 3]):
            _plan_equal(planner.batch(indices), build_batch_plan(training, indices, max_len=256))
            snapshot = planner.snapshot()  # snapshot asserts all three bounds
            assert snapshot.resident_cases <= snapshot.max_cached_cases
            assert snapshot.resident_spans <= snapshot.max_cached_spans
            assert snapshot.retained_charge_bytes <= snapshot.max_cached_bytes
        assert planner.snapshot().evictions > 0, name


def test_p1_t05_lru_trace_recency_eviction_oversize_and_snapshot_isolation(training):
    spans = {index: sum(len(q.spans) for q in build_batch_plan(training, [index], max_len=256).queries)
             for index in range(8)}
    oversize = max(spans, key=spans.__getitem__)
    small = sorted((index for index in spans if index != oversize), key=spans.__getitem__)[:3]
    a, b, c = small
    planner = RelationBatchPlanner(training, max_len=256, max_cached_cases=2,
                                   max_cached_spans=spans[oversize] - 1)
    for index in (a, b, a, c):
        planner.batch([index])
    assert list(planner._cases) == [a, c]
    before = planner.snapshot()
    assert (before.requested_cases, before.hits, before.misses, before.admissions,
            before.evictions, before.oversize_bypasses) == (4, 1, 3, 3, 1, 0)
    planner.batch([oversize])
    assert list(planner._cases) == [a, c]
    after = planner.snapshot()
    assert (after.requested_cases, after.hits, after.misses, after.admissions, after.evictions,
            after.oversize_bypasses) == (5, 1, 4, 3, 1, 1)
    assert after.resident_spans == spans[a] + spans[c]
    assert planner.snapshot() == after and list(planner._cases) == [a, c]  # no recency mutation
    mutable = after.as_dict()
    mutable["hits"] = 99
    assert planner.snapshot().hits == 1


def _policies(training):
    warm = RelationBatchPlanner(training, max_len=256)
    for indices in ([0, 1, 2], [3, 4, 5], [6, 7]):
        warm.batch(indices)
    spans = [sum(len(q.spans) for q in build_batch_plan(training, [i], max_len=256).queries)
             for i in range(8)]
    return {
        "disabled": RelationBatchPlanner(training, max_len=256, max_cached_cases=0),
        "cold": RelationBatchPlanner(training, max_len=256),
        "warm": warm,
        "evicting": RelationBatchPlanner(training, max_len=256, max_cached_cases=1),
        "oversize": RelationBatchPlanner(training, max_len=256, max_cached_spans=sorted(spans)[3]),
    }


def test_p1_t06_every_cache_policy_yields_the_identical_batch(training):
    trace = [[0, 1, 2], [3, 4, 5], [6, 7], [7, 3, 1], [0, 6, 2], [4, 5]]
    for planner in _policies(training).values():
        for indices in trace:
            _plan_equal(planner.batch(indices), build_batch_plan(training, indices, max_len=256))
        planner.snapshot()
    assert _policies(training)["oversize"].batch([0]) is not None


def test_p1_t07_rows_rebase_corpora_do_not_share_and_warm_lookups_derive_nothing(training, monkeypatch):
    planner = RelationBatchPlanner(training, max_len=256)
    first = planner.batch([0, 1, 2])
    second = planner.batch([2, 0, 1])
    by_case = {index: [q for q in first.queries if q.case_index == index] for index in (0, 1, 2)}
    for row, index in enumerate([2, 0, 1]):
        owned = [q for q in second.queries if q.row == row]
        assert all(q.case_index == index for q in owned)
        assert [(q.boundary, q.gate_target, q.spans, q.equivalents) for q in owned] == \
               [(q.boundary, q.gate_target, q.spans, q.equivalents) for q in by_case[index]]
    other = TrainingCorpus.engineering_cases(_cases(seed=53))
    assert not torch.equal(RelationBatchPlanner(other, max_len=256).batch([0]).targets,
                           planner.batch([0]).targets)
    with pytest.raises(RelationTrainingRefused):
        RelationBatchPlanner(other.cases, max_len=256)  # type: ignore[arg-type]
    for indices in ([3, 4, 5], [6, 7]):
        planner.batch(indices)
    warm_trace = ([7, 0, 3], [1, 2, 4], [5, 6])
    references = [build_batch_plan(training, indices, max_len=256) for indices in warm_trace]

    def fail(*args, **kwargs):
        pytest.fail("a fitting warm workload called a derivation/enumeration again")
    monkeypatch.setattr(module.corpus, "derivations", fail)
    monkeypatch.setattr(module, "candidate_spans", fail)
    for indices, reference in zip(warm_trace, references, strict=True):
        _plan_equal(planner.batch(indices), reference)
    assert planner.snapshot().hits == 3 + 1 + 8 and planner.snapshot().misses == 8


@pytest.mark.parametrize("arm", ["none", "span_affine_v1"])
def test_p1_t08_cache_policy_is_numerically_invisible_across_epochs(training, arm):
    policies = {"disabled": {"max_cached_cases": 0}, "default": {},
                "evicting": {"max_cached_cases": 1}, "oversize": {"max_cached_spans": 200}}
    runs = {}
    fwd_counts = {}
    opt_counts = {}
    policy_grads = {name: [] for name in policies}
    for name, changes in policies.items():
        trainer = RelationTrainer(_config(arm, **changes), training)
        fwd_calls = 0
        opt_calls = 0
        orig_fwd = trainer.model.forward
        orig_opt = trainer.optimizer.step

        def fwd_hook(*a, fn=orig_fwd, **kw):
            nonlocal fwd_calls
            fwd_calls += 1
            return fn(*a, **kw)

        def opt_hook(*a, fn=orig_opt, t=trainer, pol=name, **kw):
            nonlocal opt_calls
            opt_calls += 1
            step_grads = {p_name: p.grad.detach().clone() for p_name, p in t.model.named_parameters() if p.grad is not None}
            policy_grads[pol].append(step_grads)
            return fn(*a, **kw)

        trainer.model.forward = fwd_hook
        trainer.optimizer.step = opt_hook
        for _ in range(7):
            trainer.step()
        runs[name] = trainer
        fwd_counts[name] = fwd_calls
        opt_counts[name] = opt_calls

    reference = runs["default"]
    assert reference.cursor.epoch == 3 and len(reference.history[-1]["batch_ids"]) in (2, 3)
    probe_plan = reference.planner.batch([0, 1])
    with torch.no_grad():
        ref_logits = reference.model(probe_plan.inputs)
    for name, trainer in runs.items():
        assert trainer.history == reference.history, name
        assert _digests(trainer) == _digests(reference), name
        assert trainer.component_records == reference.component_records, name
        assert fwd_counts[name] == fwd_counts["default"] == 7, name
        assert opt_counts[name] == opt_counts["default"] == 7, name
        # Directly check step-by-step parameter gradients across cache policies
        assert len(policy_grads[name]) == 7, name
        for k in range(7):
            assert set(policy_grads[name][k]) == set(policy_grads["default"][k]), f"{name} step {k}"
            for p_name, grad in policy_grads[name][k].items():
                assert torch.equal(grad, policy_grads["default"][k][p_name]), f"{name} step {k} param {p_name}"
        # Directly check model parameters, optimizer, RNG state, forward/update counts and logits
        for p_ref, p_run in zip(reference.model.parameters(), trainer.model.parameters(), strict=True):
            assert torch.equal(p_ref, p_run), name
        _assert_optimizers_equal(trainer.optimizer, reference.optimizer)
        assert torch.equal(trainer.cursor.generator.get_state(), reference.cursor.generator.get_state()), name
        assert trainer.completed_step == reference.completed_step == 7, name
        assert len(trainer.history) == len(reference.history) == 7, name
        with torch.no_grad():
            assert torch.equal(trainer.model(probe_plan.inputs), ref_logits), name
    assert runs["evicting"].planner.snapshot().evictions > 0
    assert runs["disabled"].planner.snapshot().resident_cases == 0


def test_p1_t09_live_plan_survives_eviction_and_nothing_tensor_bearing_is_retained(training):
    planner = RelationBatchPlanner(training, max_len=256, max_cached_cases=1)
    plan = planner.batch([0])
    entry_ref = weakref.ref(planner._cases[0][0])
    planner.batch([1])  # evicts case 0
    assert 0 not in planner._cases
    gc.collect()
    assert entry_ref() is None  # the cache dropped ownership; the plan holds spans only
    plan.validate()
    _plan_equal(plan, build_batch_plan(training, [0], max_len=256))
    del plan
    trainer = _run(_config(), training, 3)
    for entry, _ in trainer.planner._cases.values():
        assert retained_charge(entry) > 0  # closed-type walk: no tensor/graph inside
    assert not any(isinstance(value, torch.Tensor) for row in trainer.history for value in row.values())
    assert not any(isinstance(value, torch.Tensor)
                   for record in trainer.component_records for value in record.values())


def test_p1_t10_failed_miss_leaves_no_partial_entry_or_corrupted_ledger(training, monkeypatch):
    lengths = sorted(len(case.flat) + 1 for case in training.cases)
    planner = RelationBatchPlanner(training, max_len=lengths[-1] - 1)
    longest = max(range(8), key=lambda i: len(training.cases[i].flat))
    shortest = min(range(8), key=lambda i: len(training.cases[i].flat))
    with pytest.raises(RelationTrainingRefused, match="above max_len"):
        planner.batch([longest])
    assert planner.snapshot().requested_cases == 0 and not planner._cases
    planner.batch([shortest])
    assert planner.snapshot().requested_cases == planner.snapshot().misses == 1

    def defect(*args, **kwargs):
        raise RuntimeError("injected enumerator defect")
    monkeypatch.setattr(module, "candidate_spans", defect)
    other = next(i for i in range(8) if i not in (shortest, longest))
    with pytest.raises(RuntimeError, match="injected enumerator defect"):
        planner.batch([other])
    assert planner.snapshot().requested_cases == 1 and list(planner._cases) == [shortest]


@pytest.mark.parametrize("policy_name,planner_kwargs", [
    ("fitting", {}),
    ("disabled", {"max_cached_cases": 0}),
    ("oversize", {"max_cached_spans": 2}),
    ("evicting", {"max_cached_cases": 2}),
])
@pytest.mark.parametrize("defect_kind", [
    "invalid_span_zero",
    "duplicate_span",
    "out_of_support_action",
    "unsupported_retained_type",
    "empty_queries",
])
def test_p1_t10_semantic_defect_before_admission_recovers_cleanly(training, monkeypatch, policy_name, planner_kwargs, defect_kind):
    planner = RelationBatchPlanner(training, max_len=256, **planner_kwargs)
    p1 = planner.batch([1])
    assert p1.case_indices.tolist() == [1]
    if policy_name in ("fitting", "evicting"):
        assert 1 in planner._cases
    before = planner.snapshot()
    cases_before = dict(planner._cases)

    orig_spans = module.candidate_spans
    orig_derive = module._derive_case_metadata

    if defect_kind == "invalid_span_zero":
        monkeypatch.setattr(module, "candidate_spans",
                            lambda prefix, boundary: ((0, 1),) if boundary == 0 else orig_spans(prefix, boundary))
        expected_exc = RelationTrainingRefused
    elif defect_kind == "duplicate_span":
        def patched_derive(p, idx):
            meta = orig_derive(p, idx)
            if idx == 0 and meta.queries:
                q0 = meta.queries[0]
                bad_spans = q0.spans + q0.spans[:1] if q0.spans else ((0, 0), (0, 0))
                bad_q = replace(q0, spans=bad_spans)
                return replace(meta, queries=(bad_q, *meta.queries[1:]))
            return meta
        monkeypatch.setattr(module, "_derive_case_metadata", patched_derive)
        expected_exc = RelationTrainingRefused
    elif defect_kind == "out_of_support_action":
        def patched_derive(p, idx):
            meta = orig_derive(p, idx)
            if idx == 0:
                for i, q in enumerate(meta.queries):
                    if q.equivalents:
                        bad_act = replace(q.equivalents[0], d4=999)
                        bad_q = replace(q, equivalents=(bad_act, *q.equivalents[1:]))
                        return replace(meta, queries=(*meta.queries[:i], bad_q, *meta.queries[i+1:]))
            return meta
        monkeypatch.setattr(module, "_derive_case_metadata", patched_derive)
        expected_exc = RelationTrainingRefused
    elif defect_kind == "unsupported_retained_type":
        def patched_derive(p, idx):
            meta = orig_derive(p, idx)
            if idx == 0:
                bad_queries = list(meta.queries)
                if bad_queries:
                    bad_q = replace(bad_queries[0], spans=[(0, 0)])
                    return replace(meta, queries=(bad_q, *bad_queries[1:]))
                return replace(meta, sequence=list(meta.sequence))
            return meta
        monkeypatch.setattr(module, "_derive_case_metadata", patched_derive)
        expected_exc = (RelationTrainingRefused, RuntimeError)
    elif defect_kind == "empty_queries":
        monkeypatch.setattr(module, "_derive_case_metadata",
                            lambda p, idx: replace(orig_derive(p, idx), queries=()) if idx == 0 else orig_derive(p, idx))
        expected_exc = RelationTrainingRefused

    with pytest.raises(expected_exc):
        planner.batch([0])

    assert 0 not in planner._cases
    assert dict(planner._cases) == cases_before
    after_fail = planner.snapshot()
    assert after_fail.admissions == before.admissions
    assert after_fail.resident_cases == before.resident_cases
    assert after_fail.hits == before.hits
    assert after_fail.misses == before.misses
    assert after_fail.requested_cases == before.requested_cases
    assert after_fail.evictions == before.evictions
    if policy_name in ("fitting", "evicting"):
        assert 1 in planner._cases

    monkeypatch.undo()
    recovered = planner.batch([0])
    assert recovered.case_indices.tolist() == [0]
    recovered_snap = planner.snapshot()
    assert recovered_snap.requested_cases == before.requested_cases + 1
    assert recovered_snap.misses == before.misses + 1


def test_p1_t11_endpoint_gradients_are_exact_with_a_bounded_gather_graph(training):
    widest = max(range(8), key=lambda i: sum(len(q.spans) for q in
                                            build_batch_plan(training, [i], max_len=256).queries))
    for indices in ([0], [widest], [0, 1, 2, 3, 4, 5, 6, 7]):
        plan = build_batch_plan(training, indices, max_len=256)
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
        assert len(visited) <= 16
        total.backward()
        assert torch.equal(states.grad, expected)
        assert packed.start_states.shape[0] == sum(len(q.spans) for q in plan.queries)


def test_p1_t12_restart_is_cold_and_numerically_exact(training, tmp_path):
    reference = _run(_config(max_cached_cases=1), training, 7)
    interrupted = _run(_config(max_cached_cases=1), training, 4)
    assert interrupted.planner.snapshot().evictions > 0
    path = tmp_path / "warm.pt"
    interrupted.save(path, sources=AUTHORITATIVE_R4_SOURCES)
    resumed = RelationTrainer.load(path, training=training, sources=AUTHORITATIVE_R4_SOURCES)
    assert resumed.planner.snapshot().requested_cases == 0
    for _ in range(3):
        resumed.step()
    assert resumed.history == reference.history and _digests(resumed) == _digests(reference)
    assert resumed.planner.snapshot().requested_cases != reference.planner.snapshot().requested_cases


# ---------------------------------------------------------------------------
# Point 2 -- paired schedule, historical ordering and exact continuation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("changes", [
    {"steps": 0}, {"steps": True}, {"warmup": 7}, {"warmup": -1}, {"warmup": 1.5},
    {"lr": float("nan")}, {"lr": float("inf")}, {"lr": 0}, {"schedule": "linear"},
    {"seed_namespace": "model_a"}, {"seed_namespace": "corpus", "seed": seed_for("corpus")},
    {"seed_namespace": "unknown"}, {"component_gradient_events": (3, 1)},
    {"component_gradient_events": (1, 1)}, {"component_gradient_events": (8,)},
    {"component_gradient_events": [1]}, {"component_gradient_events": (True,)},
])
def test_p2_t01_invalid_schedule_or_seed_identity_fails_before_a_trainer_exists(changes):
    with pytest.raises(RelationTrainingRefused):
        _config(**changes)


def test_p2_t01_named_model_seed_must_carry_its_frozen_value():
    config = _config(corpus_provenance="development", seed_namespace="development_a", seed=seed_for("development_a"))
    assert config.seed == seed_for("development_a")
    with pytest.raises(RelationTrainingRefused, match="does not match|frozen value"):
        _config(corpus_provenance="development", seed_namespace="development_a", seed=seed_for("development_a") + 1)
    sci_config = _config(corpus_provenance="scientific", seed_namespace="model_a", seed=seed_for("model_a"))
    assert sci_config.seed == seed_for("model_a")
    with pytest.raises(RelationTrainingRefused, match="does not match|frozen value"):
        _config(corpus_provenance="scientific", seed_namespace="model_a", seed=seed_for("model_a") + 1)


@pytest.mark.parametrize("prov,ns,seed_val,valid", [
    ("engineering", "engineering", 31, True),
    ("development", "development_a", seed_for("development_a"), True),
    ("development", "development_b", seed_for("development_b"), True),
    ("scientific", "model_a", seed_for("model_a"), True),
    ("scientific", "model_b", seed_for("model_b"), True),
    ("scientific", "development_a", seed_for("development_a"), False),
    ("scientific", "engineering", 31, False),
    ("development", "model_a", seed_for("model_a"), False),
    ("development", "engineering", 31, False),
    ("engineering", "model_a", seed_for("model_a"), False),
    ("engineering", "development_a", seed_for("development_a"), False),
    ("unknown", "engineering", 31, False),
    ("scientific", "unknown", seed_for("model_a"), False),
    (123, "engineering", 31, False),
    ("engineering", 123, 31, False),
    ("engineering", "engineering", -1, False),
    ("engineering", "engineering", 2**63, False),
    ("engineering", "engineering", 1.5, False),
    ("engineering", "engineering", True, False),
    ("development", "development_a", seed_for("development_a") + 1, False),
    ("scientific", "model_a", seed_for("model_a") + 1, False),
])
def test_p2_t01_provenance_matrix_config_refusals(prov, ns, seed_val, valid):
    if valid:
        cfg = _config(corpus_provenance=prov, seed_namespace=ns, seed=seed_val)
        assert cfg.corpus_provenance == prov and cfg.seed_namespace == ns and cfg.seed == seed_val
    else:
        with pytest.raises(RelationTrainingRefused):
            _config(corpus_provenance=prov, seed_namespace=ns, seed=seed_val)


def test_p2_t01_provenance_trainer_refusal_precedes_construction_and_rng(training):
    # Engineering corpus refuses scientific/development configs before RNG mutation
    sci_config = _config(corpus_provenance="scientific", seed_namespace="model_a", seed=seed_for("model_a"))
    dev_config = _config(corpus_provenance="development", seed_namespace="development_a", seed=seed_for("development_a"))
    for mismatched in (sci_config, dev_config):
        initial_rng = torch.random.get_rng_state()
        with pytest.raises(RelationTrainingRefused, match="disagrees"):
            RelationTrainer(mismatched, training)
        assert torch.equal(torch.random.get_rng_state(), initial_rng)

    # Relabeled corpus also refused
    initial_rng = torch.random.get_rng_state()
    with pytest.raises(RelationTrainingRefused, match="manifest-backed"):
        TrainingCorpus(training.cases, None, None, None, None, "scientific")
    assert torch.equal(torch.random.get_rng_state(), initial_rng)


def test_p2_t02_learning_rates_match_hand_pinned_values_and_lr_at(training):
    lr = 3e-3
    trainer = _run(_config(lr=lr), training, 7)
    used = [row["lr"] for row in trainer.history]
    assert used[0] == lr / 2                       # first warmup update: lr / warmup
    assert used[1] == lr                           # last warmup update
    assert used[2] == lr * 0.5 * (1 + math.cos(0))  # first decay update, progress 0
    assert used[6] == lr * 0.5 * (1 + math.cos(math.pi * 4 / 5))
    assert used[6] > 0                             # final update is not forced to zero
    assert used == [lr_at(k, trainer.config) for k in range(7)]
    flat = _run(_config("none", warmup=0, steps=3), training, 1)
    assert flat.history[0]["lr"] == lr


@pytest.mark.parametrize("arm", ["none", "span_affine_v1"])
def test_p2_t03_group_history_and_config_learning_rates_reconcile(training, arm, tmp_path):
    trainer = RelationTrainer(_config(arm), training)
    assert all(group["lr"] == lr_at(0, trainer.config) for group in trainer.optimizer.param_groups)
    for k in range(7):
        trainer.step()
        assert all(group["lr"] == lr_at(k, trainer.config) == trainer.history[-1]["lr"]
                   for group in trainer.optimizer.param_groups)
    path = tmp_path / "steps.pt"
    trainer.save(path, sources=AUTHORITATIVE_R4_SOURCES)
    for mutation in ("completed_plus", "completed_minus", "group_lr"):
        payload = torch.load(path, weights_only=False)
        if mutation == "completed_plus":
            payload["completed_step"] += 1
        elif mutation == "completed_minus":
            payload["completed_step"] -= 1
        else:
            payload["optimizer"]["param_groups"][0]["lr"] = lr_at(5, trainer.config)
        with pytest.raises(RelationTrainingEvidenceRefused):
            validate_bundle_shape(payload)


def test_p2_t04_completion_is_a_typed_refusal_that_changes_nothing(training, tmp_path):
    trainer = _run(_config(), training, 7)
    before = (trainer.cursor.state_dict(), list(trainer.history), _digests(trainer),
              trainer.planner.snapshot(), len(trainer.component_records))
    for _ in range(2):
        with pytest.raises(RelationTrainingComplete):
            trainer.step()
    after = (trainer.cursor.state_dict(), list(trainer.history), _digests(trainer),
             trainer.planner.snapshot(), len(trainer.component_records))
    assert {k: v for k, v in before[0].items() if k != "generator"} == \
           {k: v for k, v in after[0].items() if k != "generator"}
    assert torch.equal(before[0]["generator"], after[0]["generator"])
    assert before[1:] == after[1:]
    path = tmp_path / "complete.pt"
    trainer.save(path, sources=AUTHORITATIVE_R4_SOURCES)
    loaded = RelationTrainer.load(path, training=training, sources=AUTHORITATIVE_R4_SOURCES)
    assert loaded.completed_step == 7 and _digests(loaded) == _digests(trainer)
    with pytest.raises(RelationTrainingComplete):
        loaded.step()


@pytest.mark.parametrize("lengths,batch_size", [
    ([5] * 7, 3), ([7, 3, 99, 3, 41, 5, 60, 12, 12, 1], 3), ([4, 9, 2], 8), ([11], 1),
    ([(i * 37) % 23 + 1 for i in range(400)], 2),
])
def test_p2_t05_cursor_reproduces_the_historical_sampler_for_three_epochs(lengths, batch_size):
    sampler = BucketedBatchSampler(lengths, batch_size, seed=17)
    cursor = RelationBucketCursor.seeded(lengths, batch_size, base_seed=17)
    for epoch in range(3):
        expected = list(sampler)
        assert len(expected) == cursor.batches_per_epoch
        got = [cursor.next() for _ in range(len(expected))]
        assert got == expected
        assert cursor.epoch == epoch + 1 and cursor.cursor == len(expected)
        plan, state = bucket_epoch_plan(lengths, batch_size, epoch=epoch, seed=17)
        assert plan == expected and torch.equal(cursor.generator.get_state(), state)


def test_p2_t06_global_rng_and_cache_warmth_cannot_move_batch_order(training):
    control = RelationBucketCursor.seeded([len(c.flat) + 1 for c in training.cases], 3, base_seed=31)
    expected = [control.next() for _ in range(9)]
    cursor = RelationBucketCursor.seeded([len(c.flat) + 1 for c in training.cases], 3, base_seed=31)
    planner = RelationBatchPlanner(training, max_len=256, max_cached_cases=2)
    got = []
    for step in range(9):
        torch.rand(1000)
        np.random.rand(10)
        random.random()
        py, npy, tch = random.getstate(), np.random.get_state(), torch.get_rng_state()
        indices = cursor.next()
        planner.batch(indices)
        assert random.getstate() == py and torch.equal(torch.get_rng_state(), tch)
        assert all(np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b
                   for a, b in zip(np.random.get_state(), npy, strict=True))
        got.append(indices)
    assert got == expected


def test_p2_t07_historical_loader_and_r4_planner_agree_on_ids_targets_and_supervised_updates(training):
    programs = [case.flat for case in training.cases]
    dataset = ProgramDataset(programs, ByteCodec(), 256)
    historical = loader(dataset, 3, seed=31)
    cursor = RelationBucketCursor.seeded([len(s) for s in dataset.seqs], 3, base_seed=31)
    planner = RelationBatchPlanner(training, max_len=256)
    tails_differ = 0
    torch.manual_seed(5)
    model = DrawingLM(Config(**MODEL))
    for _ in range(2):
        for index, inputs, targets in historical:
            plan = planner.batch(cursor.next())
            assert plan.case_indices.tolist() == index.tolist()
            assert torch.equal(plan.targets, targets)
            content = targets != PAD
            assert torch.equal(plan.inputs[content], inputs[content])
            tail = ~content
            assert bool((plan.inputs[tail] == PAD).all())
            tails_differ += int((inputs[tail] != PAD).sum())
            # The inactive tail cannot reach any supervised causal position.
            for rectangle in (inputs, plan.inputs):
                model.zero_grad(set_to_none=True)
                logits = model(rectangle)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                                       ignore_index=PAD, reduction="sum")
                loss.backward()
                grads = [p.grad.clone() for p in model.parameters()]
                if rectangle is inputs:
                    reference = (logits.detach()[content], loss.detach(), grads)
                else:
                    assert torch.allclose(logits.detach()[content], reference[0], **TOL)
                    assert torch.allclose(loss.detach(), reference[1], **TOL)
                    assert all(torch.allclose(a, b, **TOL) for a, b in zip(grads, reference[2], strict=True))
    assert tails_differ > 0  # the explicit, expected historical difference


def test_p2_t08_paired_arms_share_init_bytes_indices_denominators_and_geometry(training):
    none = RelationTrainer(_config("none"), training)
    head = RelationTrainer(_config("span_affine_v1"), training)
    shared = dict(none.model.named_parameters())
    for name, parameter in head.model.named_parameters():
        if not name.startswith("relation."):
            assert torch.equal(parameter, shared[name]), name
    for _ in range(7):
        none.step()
        head.step()
        left, right = none.history[-1], head.history[-1]
        assert left["batch_ids"] == right["batch_ids"] and left["lr"] == right["lr"]
        assert left["content_bytes"] == right["content_bytes"]
        assert left["reachable_boundaries"] == right["reachable_boundaries"]
        plan_left = build_batch_plan(training, left["batch_ids"], max_len=256)
        plan_right = build_batch_plan(training, right["batch_ids"], max_len=256)
        assert torch.equal(plan_left.inputs, plan_right.inputs)
        assert _microbatch_rows(plan_left, none.config.attention_budget) == \
               _microbatch_rows(plan_right, head.config.attention_budget)
    assert any(not torch.equal(p, shared[n]) for n, p in head.model.named_parameters()
               if not n.startswith("relation."))  # arms are allowed to diverge after updates


@pytest.mark.parametrize("split", [False, True])
def test_p2_t09_none_arm_matches_the_historical_multi_step_byte_body(training, split):
    budget = 1 if split else 24_000_000  # budget 1 forces one row per chunk
    trainer = RelationTrainer(_config("none", attention_budget=budget), training)
    legacy = DrawingLM(Config(**trainer.config.model))
    legacy.load_state_dict(trainer.model.state_dict())
    cfg = trainer.config
    optimizer = torch.optim.AdamW(legacy.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                                  betas=(0.9, 0.95))
    cursor = RelationBucketCursor.seeded([len(c.flat) + 1 for c in training.cases], 3, base_seed=31)
    assert _microbatch_rows(build_batch_plan(training, [0, 1, 2], max_len=256), budget) == (1 if split else 3)
    for step in range(7):
        plan = build_batch_plan(training, cursor.next(), max_len=256)
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, cfg)
        n_tokens = (plan.targets != PAD).sum().clamp(min=1)
        optimizer.zero_grad(set_to_none=True)
        total = torch.zeros(())
        for chunk_in, chunk_tgt, _ in micro_batches(plan.inputs, plan.targets, budget):
            logits = legacy(chunk_in)
            nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), chunk_tgt.reshape(-1),
                                  ignore_index=PAD, reduction="sum")
            (nll / n_tokens).backward()
            total += nll.detach()
        torch.nn.utils.clip_grad_norm_(legacy.parameters(), cfg.grad_clip)
        optimizer.step()
        result = trainer.step()
        assert trainer.history[-1]["batch_ids"] == plan.case_indices.tolist()
        assert trainer.history[-1]["lr"] == lr_at(step, cfg)
        assert torch.equal(result.byte_sum, total)
        for old, new in zip(legacy.parameters(), trainer.model.parameters(), strict=True):
            assert torch.equal(old.grad, new.grad)  # post-clip gradients of this step
            assert torch.equal(old, new)
        assert optimizer_digest(optimizer, legacy) == optimizer_digest(trainer.optimizer, trainer.model)
    if split:
        unsplit = _run(_config("none"), training, 7)
        assert all(torch.allclose(a, b, **TOL) for a, b in
                   zip(unsplit.model.parameters(), trainer.model.parameters(), strict=True))


@pytest.mark.parametrize("cut", [0, 1, 3, 7])
def test_p2_t10_in_process_save_load_at_every_cursor_state(training, tmp_path, cut):
    reference = _run(_config(), training, 7)
    trainer = _run(_config(), training, cut)
    expected_state = {0: (0, 0), 1: (1, 1), 3: (1, 3), 7: (3, 1)}[cut]
    assert (trainer.cursor.epoch, trainer.cursor.cursor) == expected_state
    path = tmp_path / f"cut{cut}.pt"
    trainer.save(path, sources=AUTHORITATIVE_R4_SOURCES)
    resumed = RelationTrainer.load(path, training=training, sources=AUTHORITATIVE_R4_SOURCES)
    assert _digests(resumed) == _digests(trainer)
    if cut == 7:
        with pytest.raises(RelationTrainingComplete):
            resumed.step()
        return
    resumed.step()
    assert resumed.history[cut] == reference.history[cut]
    for _ in range(6 - cut):
        resumed.step()
    assert resumed.history == reference.history
    assert resumed.component_records == reference.component_records
    assert _digests(resumed) == _digests(reference)


_RESUME_SCRIPT = """
from pathlib import Path
import sys
from dm.data import relation
from dm.train_relation import AUTHORITATIVE_R4_SOURCES, RelationTrainer, RelationTrainingComplete, TrainingCorpus

build = relation.build_venue1(8, seed=41, motifs=relation.motif_pool(16, seed=40), tuples=relation.venue1_tuples())
training = TrainingCorpus.engineering_cases(build.cases)
trainer = RelationTrainer.load(Path(sys.argv[1]), training=training, sources=AUTHORITATIVE_R4_SOURCES)
while True:
    try:
        trainer.step()
    except RelationTrainingComplete:
        break
trainer.save(Path(sys.argv[2]), sources=AUTHORITATIVE_R4_SOURCES)
"""


@pytest.mark.parametrize("arm,cut", [("span_affine_v1", 0), ("span_affine_v1", 1),
                                     ("span_affine_v1", 2), ("span_affine_v1", 3),
                                     ("span_affine_v1", 6), ("span_affine_v1", 7),
                                     ("none", 0), ("none", 2), ("none", 6)])
def test_p2_t11_p3_t12_fresh_process_resume_matches_uninterrupted_run(training, tmp_path, arm, cut):
    """Cuts cover step zero, warmup, the epoch boundary, both sides of each diagnostic event
    (events are {1, 2, 3, 7}) and completion."""
    expected = tmp_path / "expected.pt"
    _run(_config(arm), training, 7).save(expected, sources=AUTHORITATIVE_R4_SOURCES)
    checkpoint, resumed = tmp_path / "checkpoint.pt", tmp_path / "resumed.pt"
    _run(_config(arm), training, cut).save(checkpoint, sources=AUTHORITATIVE_R4_SOURCES)
    env = {**os.environ, "PYTHONPATH": str(AUTHORITATIVE_R4_SOURCES[0].parents[1]),
           "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run([sys.executable, "-c", _RESUME_SCRIPT, str(checkpoint), str(resumed)],
                            text=True, capture_output=True, check=False, env=env)
    assert result.returncode == 0, result.stderr
    left = torch.load(expected, map_location="cpu", weights_only=False)
    right = torch.load(resumed, map_location="cpu", weights_only=False)
    for key in ("completed_step", "accounting", "history", "content_digests", "diagnostics", "config"):
        assert left[key] == right[key], key
    assert {k: v for k, v in left["sampler"].items() if k != "generator"} == \
           {k: v for k, v in right["sampler"].items() if k != "generator"}
    assert torch.equal(left["sampler"]["generator"], right["sampler"]["generator"])
    assert [r["step"] for r in right["diagnostics"]["component_records"]] == [1, 2, 3, 7]


@pytest.mark.parametrize("mutation", [
    "steps", "warmup", "namespace", "seed", "epoch_order", "cursor", "group_lr", "history_lr",
    "sampler_rng", "rng_copy", "schema1", "algorithm", "base_seed",
    "prov_off_diagonal", "prov_unknown", "ns_off_diagonal",
])
def test_p2_t12_tampered_schedule_seed_or_order_state_is_refused(training, tmp_path, mutation):
    trainer = _run(_config(), training, 4)
    path = tmp_path / "tamper.pt"
    trainer.save(path, sources=AUTHORITATIVE_R4_SOURCES)
    payload = torch.load(path, weights_only=False)
    if mutation == "steps":
        payload["config"]["steps"] = 9
    elif mutation == "warmup":
        payload["config"]["warmup"] = 1
    elif mutation == "namespace":
        payload["config"]["seed_namespace"] = "model_a"
    elif mutation == "prov_off_diagonal":
        payload["config"]["corpus_provenance"] = "scientific"
    elif mutation == "prov_unknown":
        payload["config"]["corpus_provenance"] = "unrecognized"
    elif mutation == "ns_off_diagonal":
        payload["config"]["seed_namespace"] = "development_a"
        payload["config"]["seed"] = seed_for("development_a")
    elif mutation == "seed":
        payload["config"]["seed"] = 32
    elif mutation == "epoch_order":
        batches = payload["sampler"]["batches"]
        batches[1], batches[2] = batches[2], batches[1]  # still covers the corpus once
    elif mutation == "cursor":
        payload["sampler"]["cursor"] = 2
    elif mutation == "group_lr":
        payload["optimizer"]["param_groups"][0]["lr"] = 0.5
    elif mutation == "history_lr":
        payload["history"][1]["lr"] = payload["history"][0]["lr"]
    elif mutation == "sampler_rng":
        payload["sampler"]["generator"] = torch.Generator().manual_seed(1).get_state()
        payload["rng"]["data_generator"] = payload["sampler"]["generator"]
    elif mutation == "rng_copy":
        payload["rng"]["data_generator"] = torch.Generator().manual_seed(1).get_state()
    elif mutation == "schema1":
        payload["schema"] = 1
    elif mutation == "algorithm":
        payload["sampler"]["algorithm"] = "continuous_v0"
    else:
        payload["sampler"]["base_seed"] = 30
    if mutation not in ("schema1",):
        with pytest.raises(RelationTrainingEvidenceRefused):
            validate_bundle_shape(payload)
    torch.save(payload, path)
    with pytest.raises(RelationTrainingRefused) as raised:
        RelationTrainer.load(path, training=training, sources=AUTHORITATIVE_R4_SOURCES)
    if mutation == "schema1":
        assert "schema 1" in str(raised.value) and "cannot be resumed" in str(raised.value)


def test_p2_t13_historical_sampler_regressions_survive_the_shared_helper():
    lengths = [7, 3, 99, 3, 41, 5, 60, 12, 12, 1]
    ordered = BucketedBatchSampler(lengths, 3, shuffle=False, pool_batches=2)
    assert list(ordered) == [[1, 3, 5], [0, 4, 2], [9, 7, 8], [6]]  # stable ties, partial batch
    assert list(ordered) == list(ordered)
    shuffled = BucketedBatchSampler(lengths, 3, pool_batches=2, seed=17)
    before = torch.get_rng_state()
    first, second = list(shuffled), list(shuffled)
    assert torch.equal(torch.get_rng_state(), before)
    assert shuffled._epoch == 2 and first != second
    assert sorted(i for b in first for i in b) == list(range(10)) == sorted(i for b in second for i in b)
    assert first == bucket_epoch_plan(lengths, 3, epoch=0, seed=17, pool_batches=2)[0]
    assert second == bucket_epoch_plan(lengths, 3, epoch=1, seed=17, pool_batches=2)[0]
    assert BUCKET_EPOCH_ALGORITHM == "historical_bucket_epoch_v1"


@pytest.mark.parametrize("failure_site", ["planning", "backward", "optimizer", "interrupt"])
def test_p2_t14_fail_stop_lifecycle_at_all_failure_sites(training, tmp_path, monkeypatch, failure_site):
    trainer = _run(_config(), training, 1)
    valid_bundle = tmp_path / "valid_step1.pt"
    trainer.save(valid_bundle, sources=AUTHORITATIVE_R4_SOURCES)
    valid_bytes = valid_bundle.read_bytes()

    if failure_site == "planning":
        monkeypatch.setattr(trainer.planner, "batch",
                            lambda *a, **kw: (_ for _ in ()).throw(RelationTrainingRefused("injected planning refusal")))
        with pytest.raises(RelationTrainingRefused, match="injected planning refusal"):
            trainer.step()
    elif failure_site == "backward":
        def failing_train_step(*a, **kw):
            raise RuntimeError("injected backward error")
        monkeypatch.setattr(module, "relation_train_step", failing_train_step)
        with pytest.raises(RuntimeError, match="injected backward error"):
            trainer.step()
    elif failure_site == "optimizer":
        def failing_opt_step(*a, **kw):
            with torch.no_grad():
                next(trainer.model.parameters()).add_(1.0)
            raise RuntimeError("injected optimizer failure")
        monkeypatch.setattr(trainer.optimizer, "step", failing_opt_step)
        with pytest.raises(RuntimeError, match="injected optimizer failure"):
            trainer.step()
    elif failure_site == "interrupt":
        monkeypatch.setattr(trainer.planner, "batch",
                            lambda *a, **kw: (_ for _ in ()).throw(KeyboardInterrupt("injected interrupt")))
        with pytest.raises(KeyboardInterrupt, match="injected interrupt"):
            trainer.step()

    # Original exception stayed loud; trainer is now marked failed.
    # Subsequent step() must refuse without touching state.
    monkeypatch.undo()
    with pytest.raises(RelationTrainingRefused, match="failed"):
        trainer.step()

    # Subsequent save() must refuse.
    absent_path = tmp_path / "should_not_exist.pt"
    with pytest.raises(RelationTrainingRefused, match="failed"):
        trainer.save(absent_path, sources=AUTHORITATIVE_R4_SOURCES)
    assert not absent_path.exists()

    # Existing valid bundle stays intact.
    with pytest.raises(RelationTrainingRefused, match="failed"):
        trainer.save(valid_bundle, sources=AUTHORITATIVE_R4_SOURCES)
    assert valid_bundle.read_bytes() == valid_bytes

    # Fresh recovery from valid step 1 bundle matches uninterrupted continuation
    resumed = RelationTrainer.load(valid_bundle, training=training, sources=AUTHORITATIVE_R4_SOURCES)
    for _ in range(3):
        resumed.step()
    uninterrupted = _run(_config(), training, 4)
    assert resumed.completed_step == uninterrupted.completed_step == 4
    assert resumed.history == uninterrupted.history
    assert _digests(resumed) == _digests(uninterrupted)


# ---------------------------------------------------------------------------
# Point 3 -- factor losses and component-gradient diagnostics
# ---------------------------------------------------------------------------


def _scores(span_logits, gate, d4, dx, dy, count, splits) -> RelationScores:
    spans = len(span_logits)
    packed = PackedCandidates(
        query_states=torch.zeros(len(splits) - 1, 4), start_states=torch.zeros(spans, 4),
        stop_states=torch.zeros(spans, 4), length_bins=torch.zeros(spans, dtype=torch.long),
        source_start=torch.arange(spans, dtype=torch.long) * 6,
        source_stop=torch.arange(spans, dtype=torch.long) * 6 + 6,
        row_splits=torch.tensor(splits, dtype=torch.long))
    return RelationScores(packed, torch.tensor(span_logits, dtype=torch.float32), torch.tensor(gate),
                          torch.tensor(d4), torch.tensor(dx), torch.tensor(dy), torch.tensor(count))


def test_p3_t01_factor_marginals_follow_probability_algebra_while_joint_penalizes_cross_products():
    # The head prefers span 0, d4=1, dx=32 and count 3: the *invalid* cross
    # product of the two valid correlated tuples below.  Every projected valid
    # value still carries almost all of its factor's mass.
    scores = _scores([4.0, 0.0, -9.0], [[0.0, 0.0]], [[0.0, 4.0] + [-9.0] * 6], [[-9.0, 0.0, 4.0]],
                     [[-9.0, 4.0, -9.0]], [[0.0, 4.0, -9.0]], [0, 3])
    valid = (EquivalentAction(0, 0, 0, 0, 0, 2), EquivalentAction(0, 1, 1, 32, 0, 3))
    cross = (EquivalentAction(0, 0, 1, 32, 0, 3), EquivalentAction(0, 1, 0, 0, 0, 2))
    marginals = factor_marginal_nll(scores, [valid])
    p_span = F.softmax(scores.span_logits, 0)
    p_d4 = F.softmax(scores.d4_logits[0], 0)
    p_dx = F.softmax(scores.dx_logits[0], 0)
    p_dy = F.softmax(scores.dy_logits[0], 0)
    p_count = F.softmax(scores.count_logits[0], 0)
    expected = {"span": -math.log(p_span[0] + p_span[1]), "d4": -math.log(p_d4[0] + p_d4[1]),
                "dx": -math.log(p_dx[1] + p_dx[2]), "dy": -math.log(p_dy[1]),
                "count": -math.log(p_count[0] + p_count[1])}
    for name in FACTOR_NAMES:
        assert torch.allclose(marginals[name], torch.tensor([expected[name]]), **TOL), name
        assert float(marginals[name]) < 1e-3  # every projected value has (almost) all the mass
    cross_marginals = factor_marginal_nll(scores, [cross])
    assert all(torch.allclose(marginals[n], cross_marginals[n], **TOL) for n in FACTOR_NAMES)
    joint_valid = float(joint_valid_nll(scores, [valid]))
    joint_cross = float(joint_valid_nll(scores, [cross]))
    assert joint_valid > 1.0 > joint_cross  # only the joint objective tells them apart
    assert joint_valid > float(sum(marginals[name] for name in FACTOR_NAMES))


def test_p3_t02_projected_values_count_once_duplicates_refused_and_order_is_irrelevant():
    scores = _scores([0.5, 0.1, -0.4, 0.9], [[0.0, 0.0]], [[0.0] * 8], [[0.0] * 3], [[0.0] * 3],
                     [[0.0] * 3], [0, 4])
    actions = (EquivalentAction(0, 0, 3, 0, 32, 2), EquivalentAction(0, 0, 3, 32, 32, 3),
               EquivalentAction(0, 2, 3, 0, -32, 2), EquivalentAction(0, 3, 5, 0, 32, 4))
    marginals = factor_marginal_nll(scores, [actions])
    p_span = F.softmax(scores.span_logits, 0)
    assert torch.allclose(marginals["span"], -torch.log(p_span[0] + p_span[2] + p_span[3]), **TOL)
    assert torch.allclose(marginals["d4"], torch.tensor([-math.log(2 / 8)]), **TOL)
    assert torch.allclose(marginals["dx"], torch.tensor([-math.log(2 / 3)]), **TOL)
    assert torch.allclose(marginals["count"], torch.tensor([0.0]), atol=1e-6)
    permuted = factor_marginal_nll(scores, [actions[::-1]])
    assert all(torch.allclose(marginals[n], permuted[n], **TOL) for n in FACTOR_NAMES)
    with pytest.raises(RelationLayoutError, match="duplicate"):
        factor_marginal_nll(scores, [actions + actions[:1]])
    with pytest.raises(RelationLayoutError):
        factor_marginal_nll(scores, [()])


def test_p3_t03_query_local_denominators_signed_supports_and_no_count_five():
    extreme = [0.1, 0.2, 1e4, -1e4, 5.0]
    scores = _scores(extreme, [[0.0, 0.0]] * 2, [[0.0] * 8] * 2, [[1.0, 2.0, 3.0]] * 2,
                     [[3.0, 2.0, 1.0]] * 2, [[0.0, 1.0, 2.0]] * 2, [0, 2, 5])
    only_first = _scores(extreme[:2], [[0.0, 0.0]], [[0.0] * 8], [[1.0, 2.0, 3.0]], [[3.0, 2.0, 1.0]],
                         [[0.0, 1.0, 2.0]], [0, 2])
    action = (EquivalentAction(0, 0, 0, -32, 32, 4),)
    both = factor_marginal_nll(scores, [action])
    alone = factor_marginal_nll(only_first, [action])
    for name in FACTOR_NAMES:
        assert torch.allclose(both[name], alone[name], **TOL), name
    assert torch.allclose(both["dx"], -F.log_softmax(torch.tensor([1.0, 2.0, 3.0]), 0)[0:1], **TOL)
    assert torch.allclose(both["dy"], -F.log_softmax(torch.tensor([3.0, 2.0, 1.0]), 0)[2:3], **TOL)
    assert torch.allclose(both["count"], -F.log_softmax(torch.tensor([0.0, 1.0, 2.0]), 0)[2:3], **TOL)
    for bad in (EquivalentAction(0, 0, 0, 0, 0, 5), EquivalentAction(0, 0, 0, 16, 0, 2),
                EquivalentAction(0, 2, 0, 0, 0, 2), EquivalentAction(1, 0, 0, 0, 0, 2)):
        with pytest.raises(RelationLayoutError):
            factor_marginal_nll(scores, [(bad,)])


def _negative_fixture(training) -> TrainingCorpus:
    cases = list(training.cases)
    cases[1] = replace(cases[1], actions=())
    return TrainingCorpus.engineering_cases(cases)


@pytest.mark.parametrize("rows", [4, 3, 2, 1])
def test_p3_t04_chunking_preserves_raw_counts_losses_gradients_and_component_norms(training, rows):
    fixture = _negative_fixture(training)
    plan = build_batch_plan(fixture, [0, 1, 2, 3], max_len=256)
    assert any(q.case_index == 1 for q in plan.queries) and plan.positive_boundaries > 0
    torch.manual_seed(3)
    whole = DrawingLM(Config(**MODEL, relation_schema="span_affine_v1"))
    split = DrawingLM(Config(**MODEL, relation_schema="span_affine_v1"))
    split.load_state_dict(whole.state_dict())
    trunk = TrunkIdentity.of(whole)
    optimizers = [torch.optim.AdamW(m.parameters(), lr=1e-3) for m in (whole, split)]
    width = plan.inputs.shape[1]

    # Capture pre/post-clip gradients for every parameter
    whole_pre_clip, whole_post_clip = {}, {}
    split_pre_clip, split_post_clip = {}, {}
    orig_clip = torch.nn.utils.clip_grad_norm_

    def clip_hook(parameters, max_norm, pre_dict, post_dict, model):
        for name, p in model.named_parameters():
            if p.grad is not None:
                pre_dict[name] = p.grad.detach().clone()
        res = orig_clip(parameters, max_norm)
        for name, p in model.named_parameters():
            if p.grad is not None:
                post_dict[name] = p.grad.detach().clone()
        return res

    from unittest import mock

    with mock.patch("torch.nn.utils.clip_grad_norm_",
                    side_effect=lambda params, mn: clip_hook(params, mn, whole_pre_clip, whole_post_clip, whole)):
        reference = relation_train_step(whole, optimizers[0], plan, grad_clip=1.0, trunk=trunk)

    with mock.patch("torch.nn.utils.clip_grad_norm_",
                    side_effect=lambda params, mn: clip_hook(params, mn, split_pre_clip, split_post_clip, split)):
        chunked = relation_train_step(split, optimizers[1], plan, grad_clip=1.0,
                                      attention_budget=rows * width ** 2, trunk=trunk)

    whole_updated = {name: p.detach().clone() for name, p in whole.named_parameters()}
    split_updated = {name: p.detach().clone() for name, p in split.named_parameters()}

    assert _microbatch_rows(plan, rows * width ** 2) == rows
    assert reference.factors.denominator == chunked.factors.denominator == plan.positive_boundaries
    for name in FACTOR_NAMES:
        assert math.isclose(reference.factors.sums[name], chunked.factors.sums[name],
                            rel_tol=1e-6, abs_tol=1e-6), name
    for name in COMPONENT_NAMES:
        left, right = reference.components.components[name], chunked.components.components[name]
        assert left.denominator == right.denominator and left.status == right.status == "measured"
        assert math.isclose(left.norm, right.norm, rel_tol=1e-6, abs_tol=1e-6), name
    assert torch.allclose(reference.total, chunked.total, **TOL)

    # Directly check all parameter pre-clip, post-clip and updated tensors with maximum discrepancy tracking
    max_pre_diff = 0.0
    max_post_diff = 0.0
    max_up_diff = 0.0
    all_param_names = [name for name, _ in whole.named_parameters()]
    assert set(whole_pre_clip) == set(split_pre_clip) == set(all_param_names)

    for name in all_param_names:
        pre_d = float((whole_pre_clip[name] - split_pre_clip[name]).abs().max())
        max_pre_diff = max(max_pre_diff, pre_d)
        assert torch.allclose(whole_pre_clip[name], split_pre_clip[name], atol=1e-6, rtol=1e-6), f"pre_clip {name}"

        post_d = float((whole_post_clip[name] - split_post_clip[name]).abs().max())
        max_post_diff = max(max_post_diff, post_d)
        assert torch.allclose(whole_post_clip[name], split_post_clip[name], atol=1e-6, rtol=1e-6), f"post_clip {name}"

        up_d = float((whole_updated[name] - split_updated[name]).abs().max())
        max_up_diff = max(max_up_diff, up_d)
        assert torch.allclose(whole_updated[name], split_updated[name], atol=1e-6, rtol=1e-6), f"updated {name}"

    assert max_pre_diff <= 1e-6
    assert max_post_diff <= 1e-6
    assert max_up_diff <= 1e-6


def test_p3_t05_negative_batch_and_none_arm_distinguish_zero_denominator_from_not_applicable(training):
    negative = TrainingCorpus.engineering_cases([replace(training.cases[0], actions=())])
    plan = build_batch_plan(negative, [0], max_len=256)
    model = DrawingLM(Config(**MODEL, relation_schema="span_affine_v1"))
    loss = relation_train_step(model, torch.optim.AdamW(model.parameters(), lr=1e-3), plan,
                               grad_clip=1.0, trunk=TrunkIdentity.of(model))
    assert loss.factors.as_dict() == {"status": "no_positive_boundaries", "unit": "nats",
                                      "denominator": 0, "sums": {n: 0.0 for n in FACTOR_NAMES}}
    joint = loss.components.components["action_joint"]
    assert (joint.status, joint.norm, joint.denominator) == ("no_positive_boundaries", None, 0)
    for name in ("byte", "gate"):
        item = loss.components.components[name]
        assert item.status == "measured" and math.isfinite(item.norm) and item.norm > 0
    assert torch.isfinite(loss.total) and loss.components.extra_backward_calls == 2
    plain = DrawingLM(Config(**MODEL))
    none_loss = relation_train_step(plain, torch.optim.AdamW(plain.parameters(), lr=1e-3), plan,
                                    grad_clip=1.0, trunk=TrunkIdentity.of(plain))
    assert none_loss.factors.as_dict() == {"status": "not_applicable", "unit": "nats",
                                           "denominator": None, "sums": None}
    for name in ("gate", "action_joint"):
        item = none_loss.components.components[name]
        assert (item.status, item.norm, item.denominator) == ("not_applicable", None, None)
    assert none_loss.components.extra_backward_calls == 1


def test_p3_t06_component_norms_match_independent_gradients_and_the_surface_is_unchanged(training):
    plan = build_batch_plan(training, [0, 1, 2], max_len=256)
    torch.manual_seed(11)
    model = DrawingLM(Config(**MODEL, relation_schema="span_affine_v1"))
    trunk = TrunkIdentity.of(model)
    params = trunk.parameters_of(model)
    independent = {}
    for name in COMPONENT_NAMES:
        model.zero_grad(set_to_none=True)
        logits, states = model(plan.inputs, return_state=True)
        packed, equivalents, gates = pack_candidates(plan, states)
        loss = relation_loss(logits, plan, scores=model.score_relation(packed),
                             equivalents=equivalents, gate_targets=gates)
        term = {"byte": loss.byte, "gate": loss.gate, "action_joint": loss.action_joint}[name]
        term.backward()
        independent[name] = math.sqrt(sum(float(p.grad.double().square().sum()) for p in params
                                          if p.grad is not None))
    reference = DrawingLM(Config(**MODEL, relation_schema="span_affine_v1"))
    reference.load_state_dict(model.state_dict())
    measured = relation_train_step(reference, torch.optim.AdamW(reference.parameters(), lr=1e-3),
                                   plan, grad_clip=1.0, trunk=trunk).components
    for name in COMPONENT_NAMES:
        assert math.isclose(measured.components[name].norm, independent[name], rel_tol=1e-6, abs_tol=1e-6)
        assert measured.components[name].norm > 0 and math.isfinite(measured.components[name].norm)
    assert len(list(model.relation.parameters())) == 10
    assert model.n_params() == reference.n_params() == \
           DrawingLM(Config(**MODEL)).n_params() + relation_parameter_count(16)


def test_p3_t07_aggregate_norm_reflects_cancellation_unlike_a_sum_of_chunk_norms():
    weight = nn.Parameter(torch.tensor([3.0, -4.0]))
    accumulator = _ComponentAccumulator([weight])
    accumulator.observe("byte", (weight * torch.tensor([1.0, 1.0])).sum())
    accumulator.observe("byte", (weight * torch.tensor([-1.0, -1.0])).sum())
    assert accumulator.norm("byte") == 0.0
    assert accumulator.extra_backward_calls == 2 and accumulator.buffer_bytes == 8
    naive = _ComponentAccumulator([weight])
    naive.observe("byte", (weight * torch.tensor([1.0, 1.0])).sum())
    assert naive.norm("byte") == pytest.approx(math.sqrt(2))
    assert weight.grad is None  # diagnostic differentiation never populates .grad


@pytest.mark.parametrize("arm", ["none", "span_affine_v1"])
@pytest.mark.parametrize("budget", [24_000_000, 1])
def test_p3_t08_diagnostics_on_and_off_are_numerically_identical(training, arm, budget):
    trainer_en = RelationTrainer(_config(arm, attention_budget=budget), training)
    trainer_dis = RelationTrainer(_config(arm, attention_budget=budget, component_gradient_events=()), training)

    fwd_en, fwd_dis = 0, 0
    opt_en, opt_dis = 0, 0
    orig_fwd_en = trainer_en.model.forward
    orig_fwd_dis = trainer_dis.model.forward
    orig_opt_en = trainer_en.optimizer.step
    orig_opt_dis = trainer_dis.optimizer.step

    def fwd_en_hook(*a, **kw):
        nonlocal fwd_en
        fwd_en += 1
        return orig_fwd_en(*a, **kw)

    def fwd_dis_hook(*a, **kw):
        nonlocal fwd_dis
        fwd_dis += 1
        return orig_fwd_dis(*a, **kw)

    last_en_grads = {}
    last_dis_grads = {}

    def opt_en_hook(*a, **kw):
        nonlocal opt_en
        opt_en += 1
        for name, p in trainer_en.model.named_parameters():
            if p.grad is not None:
                last_en_grads[name] = p.grad.detach().clone()
        return orig_opt_en(*a, **kw)

    def opt_dis_hook(*a, **kw):
        nonlocal opt_dis
        opt_dis += 1
        for name, p in trainer_dis.model.named_parameters():
            if p.grad is not None:
                last_dis_grads[name] = p.grad.detach().clone()
        return orig_opt_dis(*a, **kw)

    trainer_en.model.forward = fwd_en_hook
    trainer_dis.model.forward = fwd_dis_hook
    trainer_en.optimizer.step = opt_en_hook
    trainer_dis.optimizer.step = opt_dis_hook

    for _ in range(7):
        trainer_en.step()
        trainer_dis.step()
        # Direct on/off gradient comparison at every update step (event and off-event steps)
        assert set(last_en_grads) == set(last_dis_grads)
        for name, grad in last_en_grads.items():
            assert torch.equal(grad, last_dis_grads[name]), f"gradient discrepancy on {name}"

    enabled = trainer_en
    disabled = trainer_dis
    assert enabled.component_records and not disabled.component_records
    assert _objective_history(enabled) == _objective_history(disabled)
    assert enabled.history == disabled.history  # factors are observation only
    assert _digests(enabled) == _digests(disabled)
    assert enabled.accounting == disabled.accounting
    assert enabled.config.component_policy == "schedule_boundaries_v1"
    assert disabled.config.component_policy == "explicit_override"

    # Forward and update call count assertions
    expected_fwd = sum((len(row["batch_ids"]) if budget == 1 else 1) for row in enabled.history)
    assert fwd_en == fwd_dis == expected_fwd
    assert opt_en == opt_dis == 7

    # Directly check model parameters, optimizer, RNG state
    for p_en, p_dis in zip(enabled.model.parameters(), disabled.model.parameters(), strict=True):
        assert torch.equal(p_en, p_dis)
    _assert_optimizers_equal(enabled.optimizer, disabled.optimizer)
    assert torch.equal(enabled.cursor.generator.get_state(), disabled.cursor.generator.get_state())
    assert enabled.completed_step == disabled.completed_step == 7
    assert len(enabled.history) == len(disabled.history) == 7

    # Directly check matched-geometry logits
    probe_plan = enabled.planner.batch([0, 1])
    with torch.no_grad():
        en_logits = enabled.model(probe_plan.inputs)
        dis_logits = disabled.model(probe_plan.inputs)
        assert torch.equal(en_logits, dis_logits)


def test_p3_t09_cadence_is_precomputed_and_off_event_steps_run_no_component_backward(training, monkeypatch):
    assert default_component_events(7, 2) == (1, 2, 3, 7)
    assert default_component_events(3, 0) == (1, 3)
    assert default_component_events(1, 0) == (1,)
    assert default_component_events(2, 1) == (1, 2)
    assert _config(warmup=0, steps=3).component_events == (1, 3)
    assert _config(warmup=0, steps=1).component_events == (1,)
    trainer = RelationTrainer(_config(component_gradient_events=(1, 3, 7)), training)
    calls = []
    real = torch.autograd.grad
    monkeypatch.setattr(torch.autograd, "grad", lambda *a, **k: calls.append(1) or real(*a, **k))
    for step in range(1, 8):
        before = len(calls)
        loss = trainer.step()
        if step in (1, 3, 7):
            assert loss.components is not None and len(calls) > before
        else:
            assert loss.components is None and len(calls) == before
    assert [record["step"] for record in trainer.component_records] == [1, 3, 7]
    assert trainer.diagnostics()["component_events"] == [1, 3, 7]


def test_p3_t10_trunk_identity_counts_tied_weights_once_and_refuses_unexpected_parameters():
    model = DrawingLM(Config(**MODEL, relation_schema="span_affine_v1"))
    trunk = TrunkIdentity.of(model)
    assert "head.weight" not in trunk.names and "embed.weight" in trunk.names
    assert not any(name.startswith("relation.") for name in trunk.names)
    assert trunk.parameters == model.n_params() - relation_parameter_count(16)
    plain = DrawingLM(Config(**MODEL))
    assert trunk == TrunkIdentity.of(plain)
    # Check optimizer parameter ordering and R3 counts are untouched
    opt_params = list(model.parameters())
    assert len(opt_params) == len(trunk.names) + 10  # 10 relation parameters outside trunk

    # Untied head refusal
    untied = DrawingLM(Config(**MODEL))
    untied.head.weight = nn.Parameter(untied.embed.weight.detach().clone())
    with pytest.raises(RelationTrainingRefused, match="tied"):
        TrunkIdentity.of(untied)
    with pytest.raises(RelationTrainingRefused, match="tied"):
        trunk.parameters_of(untied)


@pytest.mark.parametrize("mutation_site", [
    "root_extra",
    "block_extra",
    "attn_extra",
    "ff_extra",
    "removed_param",
    "reshaped_param",
    "retyped_param",
    "aliased_param",
    "extra_alias",
    "overlapping_view",
])
def test_p3_t10_extended_trunk_mutations_refused_at_construction_and_reuse(mutation_site):
    model = DrawingLM(Config(**MODEL, relation_schema="span_affine_v1"))
    identity = TrunkIdentity.of(model)

    if mutation_site == "root_extra":
        model.extra = nn.Parameter(torch.zeros(3))
    elif mutation_site == "block_extra":
        model.blocks[0].extra = nn.Parameter(torch.zeros(3))
    elif mutation_site == "attn_extra":
        model.blocks[0].attn.extra = nn.Parameter(torch.zeros(3))
    elif mutation_site == "ff_extra":
        model.blocks[0].ff.extra = nn.Parameter(torch.zeros(3))
    elif mutation_site == "removed_param":
        del model.blocks[0].attn.proj.weight
    elif mutation_site == "reshaped_param":
        model.blocks[0].norm_attn.weight = nn.Parameter(torch.zeros(32))
    elif mutation_site == "retyped_param":
        model.blocks[0].norm_attn.weight = nn.Parameter(torch.zeros(16, dtype=torch.bfloat16))
    elif mutation_site == "aliased_param":
        model.blocks[0].norm_ff.weight = model.blocks[0].norm_attn.weight
    elif mutation_site == "extra_alias":
        model.blocks[0].attn.qkv.extra = model.blocks[0].attn.qkv.weight
    elif mutation_site == "overlapping_view":
        model.blocks[0].norm_ff.weight = nn.Parameter(model.blocks[0].attn.qkv.weight.flatten()[1:17])

    with pytest.raises(RelationTrainingRefused):
        TrunkIdentity.of(model)
    with pytest.raises(RelationTrainingRefused):
        identity.parameters_of(model)


def test_p3_t11_event_buffers_are_bounded_and_not_retained(training):
    trainer = _run(_config(component_gradient_events=(1, 2, 3)), training, 3)
    limit = 3 * trainer.trunk.parameters * 4
    for record in trainer.component_records:
        assert 0 < record["buffer_bytes"] <= limit
        assert record["extra_backward_calls"] in (2, 3)
        assert not any(isinstance(v, torch.Tensor) for v in record.values())
    assert all(not isinstance(v, torch.Tensor) for row in trainer.history for v in row.values())
    assert not hasattr(trainer, "_accumulator")
    for entry, _ in trainer.planner._cases.values():
        retained_charge(entry)


@pytest.mark.parametrize("mutation", [
    "event_step", "extra_event", "missing_event", "batch_ids", "factor_name", "factor_denominator",
    "factor_status", "factor_nan", "factor_negative", "trunk_digest", "component_nan",
    "component_weight", "component_status_zero_to_na", "policy",
    "buffer_bytes_too_high", "buffer_bytes_negative", "calls_too_low", "calls_too_high",
    "coordinated_218_to_219", "forged_reachable", "forged_positive",
])
def test_p3_t13_tampered_diagnostic_records_are_refused_by_reconciliation(training, tmp_path, mutation):
    trainer = _run(_config(), training, 4)
    path = tmp_path / "diag.pt"
    trainer.save(path, sources=AUTHORITATIVE_R4_SOURCES)
    payload = torch.load(path, weights_only=False)
    records = payload["diagnostics"]["component_records"]
    positive_row = next(r for r in payload["history"] if r["positive_boundaries"] > 0)
    if mutation == "event_step":
        records[0]["step"] = 4
    elif mutation == "extra_event":
        records.append({**records[-1], "step": 4, "batch_ids": payload["history"][3]["batch_ids"]})
    elif mutation == "missing_event":
        records.pop()
    elif mutation == "batch_ids":
        records[0]["batch_ids"] = list(reversed(records[0]["batch_ids"]))
    elif mutation == "factor_name":
        positive_row["factors"]["sums"]["gate"] = positive_row["factors"]["sums"].pop("span")
    elif mutation == "factor_denominator":
        positive_row["factors"]["denominator"] += 1
    elif mutation == "factor_status":
        positive_row["factors"]["status"] = "not_applicable"
    elif mutation == "factor_nan":
        positive_row["factors"]["sums"]["d4"] = float("nan")
    elif mutation == "factor_negative":
        positive_row["factors"]["sums"]["d4"] = -2 * FACTOR_ALLOWANCE_PER_QUERY * positive_row["positive_boundaries"]
    elif mutation == "trunk_digest":
        payload["diagnostics"]["trunk"]["digest"] = "0" * 64
    elif mutation == "component_nan":
        records[0]["components"]["byte"]["norm"] = float("nan")
    elif mutation == "component_weight":
        records[0]["components"]["gate"]["weight"] = 1.0
    elif mutation == "component_status_zero_to_na":
        records[0]["components"]["byte"] = {"status": "not_applicable", "norm": None,
                                            "denominator": None, "weight": 1.0}
    elif mutation == "buffer_bytes_too_high":
        records[0]["buffer_bytes"] = 10**12
    elif mutation == "buffer_bytes_negative":
        records[0]["buffer_bytes"] = -1
    elif mutation == "calls_too_low":
        records[0]["extra_backward_calls"] = 0
    elif mutation == "calls_too_high":
        records[0]["extra_backward_calls"] = 100
    elif mutation == "coordinated_218_to_219":
        row = payload["history"][0]
        row["content_bytes"] += 1
        for key in ("semantic_bytes", "content_symbols", "byte_denominator"):
            payload["accounting"][key] += 1
        payload["diagnostics"]["component_records"][0]["components"]["byte"]["denominator"] += 1
    elif mutation == "forged_reachable":
        row = payload["history"][0]
        row["reachable_boundaries"] += 1
        payload["accounting"]["gate_denominator"] += 1
        payload["accounting"]["reachable_queries"] += 1
        payload["diagnostics"]["component_records"][0]["components"]["gate"]["denominator"] += 1
    elif mutation == "forged_positive":
        row = payload["history"][0]
        row["positive_boundaries"] -= 1
        row["factors"]["denominator"] -= 1
        payload["accounting"]["positive_denominator"] -= 1
        payload["diagnostics"]["component_records"][0]["components"]["action_joint"]["denominator"] -= 1
    else:
        payload["diagnostics"]["component_policy"] = "explicit_override"

    if mutation in ("buffer_bytes_too_high", "buffer_bytes_negative", "calls_too_low",
                    "calls_too_high", "coordinated_218_to_219"):
        with pytest.raises(RelationTrainingEvidenceRefused):
            validate_bundle_shape(payload)

    torch.save(payload, path)
    with pytest.raises(RelationTrainingRefused):
        RelationTrainer.load(path, training=training, sources=AUTHORITATIVE_R4_SOURCES)


def test_p3_t14_finite_zero_is_recorded_and_validator_defects_stay_loud(training, tmp_path, monkeypatch):
    trainer = _run(_config(), training, 1)
    path = tmp_path / "zero.pt"
    trainer.save(path, sources=AUTHORITATIVE_R4_SOURCES)
    payload = torch.load(path, weights_only=False)
    payload["diagnostics"]["component_records"][0]["components"]["byte"]["norm"] = 0.0
    validate_bundle_shape(payload)  # an applicable finite zero is a zero, not N/A
    import dm.eval.relation_training_evidence as evidence

    def defect(*args, **kwargs):
        raise ValueError("injected factor validator defect")
    monkeypatch.setattr(evidence, "_validate_factor_record", defect)
    with pytest.raises(ValueError, match="injected factor validator defect") as raised:
        validate_bundle_shape(payload)
    assert type(raised.value) is ValueError


def test_p3_t15_full_support_projection_and_extreme_logits_are_stable(training, tmp_path):
    every = tuple(EquivalentAction(0, 0, d4, dx, dy, count) for d4 in range(8)
                  for dx in (-32, 0, 32) for dy in (-32, 0, 32) for count in COUNT_SUPPORT)
    huge = _scores([3e38], [[0.0, 0.0]], [[3e38, -3e38] + [0.0] * 6], [[1e30, 0.0, -1e30]],
                   [[0.0] * 3], [[-3e38, 3e38, 0.0]], [0, 1])
    marginals = factor_marginal_nll(huge, [every])
    for name in FACTOR_NAMES:
        value = float(marginals[name])
        assert math.isfinite(value) and -FACTOR_ALLOWANCE_PER_QUERY <= value <= 1e-5, name
    nonfinite = _scores([float("nan")], [[0.0, 0.0]], [[0.0] * 8], [[0.0] * 3], [[0.0] * 3],
                        [[0.0] * 3], [0, 1])
    assert not math.isfinite(float(factor_marginal_nll(nonfinite, [every[:1]])["span"]))
    trainer = _run(_config(), training, 1)
    path = tmp_path / "allowance.pt"
    trainer.save(path, sources=AUTHORITATIVE_R4_SOURCES)
    payload = torch.load(path, weights_only=False)
    row = payload["history"][0]
    assert row["positive_boundaries"] > 0
    row["factors"]["sums"]["d4"] = -FACTOR_ALLOWANCE_PER_QUERY * row["positive_boundaries"]
    validate_bundle_shape(payload)
    row["factors"]["sums"]["d4"] = -FACTOR_ALLOWANCE_PER_QUERY * row["positive_boundaries"] * 1.01
    with pytest.raises(RelationTrainingEvidenceRefused, match="allowance"):
        validate_bundle_shape(payload)


@pytest.mark.parametrize("domain_mutation", [
    "empty_lengths",
    "zero_pool",
    "negative_pool",
    "bool_pool",
    "fractional_seed",
    "huge_denominator",
    "coordinated_huge_seed",
    "coordinated_huge_counts",
    "negative_batch_size",
    "bool_batch_size",
    "unhashable_batch_ids",
    "huge_warmup_steps",
    "forged_padding",
])
def test_p3_t13_malformed_domains_refused_at_both_boundaries(training, tmp_path, domain_mutation):
    trainer = _run(_config(), training, 1)
    path = tmp_path / "domain_tamper.pt"
    trainer.save(path, sources=AUTHORITATIVE_R4_SOURCES)
    payload = torch.load(path, weights_only=False)

    if domain_mutation == "empty_lengths":
        payload["sampler"]["lengths"] = []
    elif domain_mutation == "zero_pool":
        payload["sampler"]["pool_batches"] = 0
    elif domain_mutation == "negative_pool":
        payload["sampler"]["pool_batches"] = -1
    elif domain_mutation == "bool_pool":
        payload["sampler"]["pool_batches"] = True
    elif domain_mutation == "fractional_seed":
        payload["sampler"]["base_seed"] = 1.5
    elif domain_mutation == "huge_denominator":
        payload["history"][0]["positive_boundaries"] = 10**400
        payload["history"][0]["factors"]["denominator"] = 10**400
    elif domain_mutation == "coordinated_huge_seed":
        payload["config"]["seed"] = 10**400
        payload["sampler"]["base_seed"] = 10**400
    elif domain_mutation == "coordinated_huge_counts":
        payload["history"][0]["content_bytes"] = 10**400
        payload["history"][0]["reachable_boundaries"] = 10**400
        payload["history"][0]["positive_boundaries"] = 10**400
    elif domain_mutation == "negative_batch_size":
        payload["sampler"]["batch_size"] = -1
    elif domain_mutation == "bool_batch_size":
        payload["sampler"]["batch_size"] = True
    elif domain_mutation == "unhashable_batch_ids":
        payload["history"][0]["batch_ids"] = [[0], 2]
    elif domain_mutation == "huge_warmup_steps":
        payload["config"]["steps"] = 10**400 + 1
        payload["config"]["warmup"] = 10**400
        payload["diagnostics"]["component_events"] = [1, 10**400, 10**400 + 1]
    elif domain_mutation == "forged_padding":
        payload["accounting"]["padded_positions"] += 1

    with pytest.raises(RelationTrainingEvidenceRefused) as direct_exc:
        validate_bundle_shape(payload)
    assert not isinstance(direct_exc.value, (OverflowError, TypeError))

    torch.save(payload, path)
    with pytest.raises(RelationTrainingRefused) as load_exc:
        RelationTrainer.load(path, training=training, sources=AUTHORITATIVE_R4_SOURCES)
    assert not isinstance(load_exc.value, (OverflowError, TypeError, IndexError))


def test_p3_t13_coordinated_sampler_longer_than_corpus_refused_without_index_error(training, tmp_path):
    trainer_8 = _run(_config(), training, 1)
    base_path = tmp_path / "base.pt"
    trainer_8.save(base_path, sources=AUTHORITATIVE_R4_SOURCES)
    base_corpus = torch.load(base_path, weights_only=False)["corpus"]

    nine = TrainingCorpus.engineering_cases([*training.cases, training.cases[0]])
    ntrainer = RelationTrainer(_config(), nine)
    for _ in range(3):
        ntrainer.step()
    path = tmp_path / "nine_row_bundle.pt"
    ntrainer.save(path, sources=AUTHORITATIVE_R4_SOURCES)
    payload = torch.load(path, weights_only=False)
    payload["corpus"] = base_corpus
    torch.save(payload, path)
    # The payload was generated against 9 cases and is internally consistent
    validate_bundle_shape(payload)
    # Loading it against an 8-case corpus must refuse on sampler lengths before row indexing
    with pytest.raises(RelationTrainingRefused, match="sampler lengths disagree") as load_exc:
        RelationTrainer.load(path, training=training, sources=AUTHORITATIVE_R4_SOURCES)
    assert not isinstance(load_exc.value, (IndexError,))


def test_p3_t13_valid_step_zero_and_completed_bundles_load(training, tmp_path):
    initial = RelationTrainer(_config(), training)
    zero_path = tmp_path / "zero.pt"
    initial.save(zero_path, sources=AUTHORITATIVE_R4_SOURCES)
    loaded_zero = RelationTrainer.load(zero_path, training=training, sources=AUTHORITATIVE_R4_SOURCES)
    assert loaded_zero.completed_step == 0
    assert loaded_zero.history == []

    finished = _run(_config(), training, 7)
    done_path = tmp_path / "done.pt"
    finished.save(done_path, sources=AUTHORITATIVE_R4_SOURCES)
    loaded_done = RelationTrainer.load(done_path, training=training, sources=AUTHORITATIVE_R4_SOURCES)
    assert loaded_done.completed_step == 7
    assert len(loaded_done.history) == 7


def test_p2_t14_save_refuses_forged_history_before_publication(training, tmp_path):
    trainer = _run(_config(), training, 1)
    trainer.history[0]["reachable_boundaries"] += 1
    trainer.accounting.gate_denominator += 1
    trainer.accounting.reachable_queries += 1
    trainer.component_records[0]["components"]["gate"]["denominator"] += 1

    path = tmp_path / "forged_save.pt"
    with pytest.raises(RelationTrainingRefused, match="disagrees with verified corpus"):
        trainer.save(path, sources=AUTHORITATIVE_R4_SOURCES)
    assert not path.exists()


def test_p2_t14_save_refuses_forged_trunk_identity_and_buffer_bound(training, tmp_path):
    trainer = _run(_config(), training, 1)
    diagnostics = copy.deepcopy(trainer.diagnostics())
    diagnostics["trunk"]["shapes"][0][0] *= 2
    diagnostics["trunk"]["parameters"] += trainer.trunk.shapes[0][0] * trainer.trunk.shapes[0][1]
    diagnostics["component_records"][0]["buffer_bytes"] = 12 * diagnostics["trunk"]["parameters"]
    path = tmp_path / "forged_trunk_save.pt"
    with pytest.raises(RelationTrainingRefused, match="trunk identity differs"):
        module.save_training_bundle(
            path, config=trainer.config, completed_step=trainer.completed_step,
            model=trainer.model, optimizer=trainer.optimizer, sampler=trainer.cursor.state_dict(),
            accounting=trainer.accounting, history=trainer.history, training=training,
            data_generator=trainer.cursor.generator, sources=AUTHORITATIVE_R4_SOURCES,
            diagnostics=diagnostics,
            process_environment=trainer._process_environment,
            process_source_hashes=trainer._process_source_hashes,
        )
    assert not path.exists()


def test_p3_t08_diagnostic_benchmark_executes_and_refuses_overwrite(tmp_path):
    from scripts.relation_diagnostic_benchmark import benchmark

    report = benchmark(repeats=1)
    assert report["schema"] == 2
    assert report["kind"] == "r4_diagnostic_overhead_benchmark"
    assert report["model_constructed"] is True
    assert set(report["results"]) == {
        "none_unsplit",
        "none_split",
        "span_affine_v1_unsplit",
        "span_affine_v1_split",
    }
    for res in report["results"].values():
        assert res["transparency"]["history_exact"] is True
        assert res["transparency"]["parameters_exact"] is True
        assert res["transparency"]["rng_exact"] is True
        assert len(res["steps"]) == 7

    out_file = tmp_path / "diagnostic_bench.json"
    out_file.write_text(json.dumps(report, indent=2))
    assert out_file.exists()

    env = {**os.environ, "PYTHONPATH": ".", "PYTHONDONTWRITEBYTECODE": "1"}
    proc = subprocess.run(
        [sys.executable, "scripts/relation_diagnostic_benchmark.py", "--output", str(out_file), "--repeats", "1"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert proc.returncode != 0
    assert "already exists" in proc.stderr


def test_p3_t08_diagnostic_benchmark_detects_rng_mutation(training):
    from scripts.relation_diagnostic_benchmark import (
        _run_single_trajectory,
        _verify_numerical_transparency,
    )
    orig_step = module.RelationTrainer.step

    def rng_mutating_step(self):
        res = orig_step(self)
        if self.config.component_gradient_events is not None:
            random.random()
        return res

    baseline = module.capture_rng_state()
    from unittest import mock
    with mock.patch.object(module.RelationTrainer, "step", rng_mutating_step):
        module.restore_rng_state(baseline)
        _, _, _, left = _run_single_trajectory(_config(), training, start_rng=baseline)
        module.restore_rng_state(baseline)
        _, _, _, right = _run_single_trajectory(_config(component_gradient_events=()), training, start_rng=baseline)
        res = _verify_numerical_transparency(left, right)
        assert res["rng_exact"] is False, "Must detect injected RNG difference"

    module.restore_rng_state(baseline)
    _, _, _, left_clean = _run_single_trajectory(_config(), training, start_rng=baseline)
    module.restore_rng_state(baseline)
    _, _, _, right_clean = _run_single_trajectory(_config(component_gradient_events=()), training, start_rng=baseline)
    res_clean = _verify_numerical_transparency(left_clean, right_clean)
    assert res_clean["rng_exact"] is True, "Clean trajectories must have rng_exact=True"
