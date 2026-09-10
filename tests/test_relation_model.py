"""R3: packed relation scoring, joint equivalence loss, and bounded search."""

from __future__ import annotations

import ast
import inspect
import time
from dataclasses import fields
from importlib.util import resolve_name
from itertools import product

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from dm.data import relation as corpus
from dm.eval import relation_contract as contract
from dm.isa.asm import assemble
from dm.models.relation import (
    COPY,
    EMIT,
    RELATION_KBEST_ACTIONS,
    RELATION_LENGTH_BIN_EDGES,
    RELATION_LENGTH_BINS,
    RELATION_MAX_CANDIDATE_SPANS,
    RELATION_RANK,
    EquivalentAction,
    PackedCandidates,
    RelationHead,
    RelationLayoutError,
    RelationScores,
    boundary_states,
    choose_action,
    joint_valid_nll,
    relation_parameter_count,
)
from dm.models.transformer import Config, DrawingLM
from dm.relation import CopyExecution, FaultCode, TransducerFault


def _packed(*, d_model: int = 8, queries: int = 2) -> PackedCandidates:
    torch.manual_seed(12)
    counts = [2, 1][:queries]
    splits = torch.tensor([0, *torch.tensor(counts).cumsum(0).tolist()])
    spans = int(splits[-1])
    return PackedCandidates(
        query_states=torch.randn(queries, d_model, requires_grad=True),
        start_states=torch.randn(spans, d_model, requires_grad=True),
        stop_states=torch.randn(spans, d_model, requires_grad=True),
        length_bins=torch.arange(spans, dtype=torch.long),
        source_start=torch.arange(spans, dtype=torch.long) * 32,
        source_stop=torch.arange(spans, dtype=torch.long) * 32 + torch.tensor([7, 13, 33])[:spans],
        row_splits=splits,
    )


def _valid_prefix() -> bytes:
    return assemble("MOVE 112 120\nLINE 128 136")


def test_schema_is_optional_append_only_and_exactly_owns_the_frozen_surface():
    old = {"vocab_size": 258, "d_model": 128, "n_layers": 4, "n_heads": 4}
    assert Config(**old).relation_schema == "none"
    model = DrawingLM(Config(**old, relation_schema="span_affine_v1"))
    names = [name.removeprefix("relation.") for name, _ in model.named_parameters()
             if name.startswith("relation.")]
    assert names == [
        "norm.weight", "query.weight", "key_start.weight", "key_stop.weight",
        "length.weight", "gate.weight", "d4.weight", "dx.weight", "dy.weight",
        "count.weight",
    ]
    assert model.n_params() == model.cfg.n_params() == 839_808
    assert relation_parameter_count(128) == 15_104
    # The pre-R3 checkpoint state remains strict-loadable into its default arm.
    old_model = DrawingLM(Config(**old))
    DrawingLM(Config(**old)).load_state_dict(old_model.state_dict(), strict=True)


def test_model_constants_are_bound_to_the_frozen_r1_contract():
    assert RELATION_RANK == contract.RELATION_RANK
    assert RELATION_LENGTH_BINS == contract.RELATION_LENGTH_BINS
    assert RELATION_LENGTH_BIN_EDGES == contract.RELATION_LENGTH_BIN_EDGES
    assert RELATION_MAX_CANDIDATE_SPANS == contract.RELATION_MAX_CANDIDATE_SPANS
    assert RELATION_KBEST_ACTIONS == contract.RELATION_KBEST_ACTIONS
    assert relation_parameter_count(contract.PILOT_D_MODEL) == contract.relation_overhead()
    assert Config(vocab_size=258, relation_schema="span_affine_v1").n_params() == \
        contract.parameter_table()["span_affine_v1"]["total"]
    assert set(corpus.D4_SUPPORT) == set(range(8))


def test_unknown_or_feedback_combined_relation_schema_fails_before_model_construction():
    with pytest.raises(ValueError, match="relation_schema"):
        Config(vocab_size=2, relation_schema="span_v2")
    with pytest.raises(ValueError, match="may not coexist"):
        Config(vocab_size=2, feedback_schema="glu_v1", relation_schema="span_affine_v1")


def test_same_seed_keeps_shared_model_tensors_and_ordinary_logits_exact():
    shape = {"vocab_size": 31, "d_model": 16, "n_layers": 2, "n_heads": 2, "max_len": 32}
    torch.manual_seed(33)
    plain = DrawingLM(Config(**shape))
    torch.manual_seed(33)
    relation = DrawingLM(Config(**shape, relation_schema="span_affine_v1"))
    torch.manual_seed(33)
    relation_repeat = DrawingLM(Config(**shape, relation_schema="span_affine_v1"))
    torch.manual_seed(34)
    relation_other_seed = DrawingLM(Config(**shape, relation_schema="span_affine_v1"))
    shared = set(plain.state_dict()) & set(relation.state_dict())
    assert all(torch.equal(plain.state_dict()[name], relation.state_dict()[name])
               for name in shared)
    ids = torch.tensor([[1, 2, 3]])
    assert torch.equal(plain(ids), relation(ids))
    assert not any(name.startswith("relation.") for name in plain.state_dict())
    relation_names = [name for name in relation.state_dict() if name.startswith("relation.")]
    assert all(torch.equal(relation.state_dict()[name], relation_repeat.state_dict()[name])
               for name in relation_names)
    assert any(not torch.equal(relation.state_dict()[name], relation_other_seed.state_dict()[name])
               for name in relation_names)
    plain.share_non_embedding_init(9)
    relation.share_non_embedding_init(9)
    shared = set(plain.state_dict()) & set(relation.state_dict())
    assert all(torch.equal(plain.state_dict()[name], relation.state_dict()[name])
               for name in shared)


def test_packed_scores_are_ragged_and_no_other_query_enters_a_span_denominator():
    head = RelationHead(8)
    packed = _packed()
    scores = head(packed)
    assert scores.span_log_probs(0).shape == (2,)
    assert scores.span_log_probs(1).shape == (1,)
    changed = PackedCandidates(
        query_states=packed.query_states,
        start_states=torch.cat((packed.start_states[:2], packed.start_states[2:] + 1000)),
        stop_states=packed.stop_states, length_bins=packed.length_bins,
        source_start=packed.source_start, source_stop=packed.source_stop,
        row_splits=packed.row_splits,
    )
    changed_scores = head(changed)
    assert torch.allclose(scores.span_log_probs(0), changed_scores.span_log_probs(0))


def test_head_equation_matches_its_independent_packed_algebra():
    head = RelationHead(8)
    packed = _packed(queries=1)
    scores = head(packed)
    query = head.norm(packed.query_states)
    keys = (head.key_start(head.norm(packed.start_states))
            + head.key_stop(head.norm(packed.stop_states)) + head.length(packed.length_bins))
    expected = (head.query(query).repeat_interleave(torch.tensor([2]), dim=0) * keys).sum(-1) / 32**0.5
    assert torch.allclose(scores.span_logits, expected)


def test_bos_and_byte_boundary_indexing_is_explicit_and_off_by_one_safe():
    # State 0 is BOS/boundary offset 0; byte boundary b is position b, after
    # byte b - 1.  Distinct values make a one-position shift observable.
    states = torch.arange(5 * 3, dtype=torch.float32).reshape(5, 3)
    selected = boundary_states(states, torch.tensor([0, 1, 4]))
    assert torch.equal(selected, torch.stack((states[0], states[1], states[4])))
    with pytest.raises(RelationLayoutError, match="outside"):
        boundary_states(states, torch.tensor([5]))
    with pytest.raises(RelationLayoutError, match="int64"):
        boundary_states(states, torch.tensor([0.0]))


@pytest.mark.parametrize("query_dtype,start_dtype,stop_dtype", [
    (torch.float64, torch.float64, torch.float64),
    (torch.float32, torch.float64, torch.float32),
])
def test_state_dtype_faults_are_typed_and_happen_before_projection(query_dtype, start_dtype, stop_dtype):
    packed = _packed(queries=1)
    bad = PackedCandidates(
        query_states=packed.query_states.detach().to(query_dtype),
        start_states=packed.start_states.detach().to(start_dtype),
        stop_states=packed.stop_states.detach().to(stop_dtype),
        length_bins=packed.length_bins, source_start=packed.source_start,
        source_stop=packed.source_stop, row_splits=packed.row_splits,
    )
    with pytest.raises(RelationLayoutError, match="dtype"):
        RelationHead(8)(bad)


def test_joint_loss_is_joint_not_factor_marginal_and_has_finite_gradients_everywhere():
    head = RelationHead(8)
    packed = _packed(queries=1)
    scores = head(packed)
    equivalents = [[
        EquivalentAction(0, 0, 0, -32, -32, 2),
        EquivalentAction(0, 1, 1, 0, 0, 3),
    ]]
    loss = joint_valid_nll(scores, equivalents)
    assert loss.shape == (1,) and torch.isfinite(loss).all()
    total = loss.sum() - scores.gate_log_probs(0)[COPY] - scores.gate_log_probs(0)[EMIT]
    total.backward()
    for state in (packed.query_states, packed.start_states, packed.stop_states):
        assert state.grad is not None and torch.isfinite(state.grad).all()
    for _, parameter in head.named_parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0


def test_joint_loss_matches_direct_bruteforce_logsumexp_and_reaches_the_trunk():
    model = DrawingLM(Config(vocab_size=17, d_model=8, n_layers=1, n_heads=2,
                             max_len=16, relation_schema="span_affine_v1"))
    _, states = model(torch.tensor([[1, 2, 3, 4]]), return_state=True)
    packed = PackedCandidates(
        query_states=states[0, 3:4], start_states=states[0, :2],
        stop_states=states[0, 1:3], length_bins=torch.tensor([0, 0]),
        source_start=torch.tensor([0, 4]), source_stop=torch.tensor([4, 8]),
        row_splits=torch.tensor([0, 2]),
    )
    scores = model.score_relation(packed)
    actions = [
        EquivalentAction(0, 0, 0, -32, -32, 2),
        EquivalentAction(0, 1, 1, 0, 0, 3),
    ]
    loss = joint_valid_nll(scores, [actions])
    direct = []
    span = scores.span_log_probs(0)
    factors = scores.factor_log_probs(0)
    for action in actions:
        direct.append(span[action.span] + factors[0][action.d4] + factors[1][(-32, 0, 32).index(action.dx)]
                      + factors[2][(-32, 0, 32).index(action.dy)] + factors[3][(2, 3, 4).index(action.total_count)])
    assert torch.allclose(loss, -torch.logsumexp(torch.stack(direct), 0), atol=1e-7)
    loss.sum().backward()
    assert model.blocks[0].attn.qkv.weight.grad is not None
    assert model.blocks[0].attn.qkv.weight.grad.abs().sum() > 0


def test_joint_loss_rejects_duplicate_cross_query_and_non_candidate_equivalences():
    scores = RelationHead(8)(_packed())
    action = EquivalentAction(0, 0, 0, -32, -32, 2)
    with pytest.raises(RelationLayoutError, match="duplicate"):
        joint_valid_nll(scores, [[action, action]])
    with pytest.raises(RelationLayoutError, match="crosses"):
        joint_valid_nll(scores, [[action, EquivalentAction(1, 2, 0, -32, -32, 2)]])
    with pytest.raises(RelationLayoutError, match="not a candidate"):
        joint_valid_nll(scores, [[EquivalentAction(0, 2, 0, -32, -32, 2)]])
    malformed = EquivalentAction(0, 0, [], -32, -32, 2)  # type: ignore[arg-type]
    with pytest.raises(RelationLayoutError, match="d4"):
        joint_valid_nll(scores, [[malformed]])


def test_joint_loss_is_permutation_invariant_and_improves_with_valid_mass():
    scores = RelationHead(8)(_packed(queries=1))
    actions = [EquivalentAction(0, 0, 0, -32, -32, 2),
               EquivalentAction(0, 1, 1, 0, 0, 3)]
    before = joint_valid_nll(scores, [actions])
    after = joint_valid_nll(scores, [list(reversed(actions))])
    assert torch.allclose(before, after)
    # One candidate, so raising the selected D4 logit only raises valid mass.
    one = RelationScores(scores.packed, scores.span_logits, scores.gate_logits,
                         scores.d4_logits.clone(), scores.dx_logits, scores.dy_logits,
                         scores.count_logits)
    action = EquivalentAction(0, 0, 0, -32, -32, 2)
    single_packed = PackedCandidates(
        query_states=one.packed.query_states, start_states=one.packed.start_states[:1],
        stop_states=one.packed.stop_states[:1], length_bins=one.packed.length_bins[:1],
        source_start=one.packed.source_start[:1], source_stop=one.packed.source_stop[:1],
        row_splits=torch.tensor([0, 1]),
    )
    single = RelationHead(8)(single_packed)
    boosted = RelationScores(single.packed, single.span_logits, single.gate_logits,
                             single.d4_logits + F.one_hot(torch.tensor([0]), 8) * 4,
                             single.dx_logits, single.dy_logits, single.count_logits)
    assert joint_valid_nll(boosted, [[action]]) < joint_valid_nll(single, [[action]])


def test_malformed_packed_layout_fails_before_projection():
    packed = _packed()
    bad = PackedCandidates(
        query_states=torch.randn(1, 8), start_states=torch.randn(769, 8),
        stop_states=torch.randn(769, 8), length_bins=torch.zeros(769, dtype=torch.long),
        source_start=torch.arange(769, dtype=torch.long) * 8,
        source_stop=torch.arange(769, dtype=torch.long) * 8 + 7,
        row_splits=torch.tensor([0, 769]),
    )
    with pytest.raises(RelationLayoutError, match="candidate-span cap"):
        RelationHead(8)(bad)
    with pytest.raises(ValueError, match="no relation parameters"):
        DrawingLM(Config(vocab_size=8)).score_relation(packed)
    wrong_bin = PackedCandidates(
        query_states=packed.query_states, start_states=packed.start_states,
        stop_states=packed.stop_states, length_bins=torch.ones(3, dtype=torch.long),
        source_start=packed.source_start, source_stop=packed.source_stop,
        row_splits=packed.row_splits,
    )
    with pytest.raises(RelationLayoutError, match="length_bins"):
        RelationHead(8)(wrong_bin)


def test_search_uses_canonical_ties_and_never_exceeds_the_frozen_cap():
    packed = _packed(queries=1)
    scores = RelationHead(8)(packed)
    scores = RelationScores(scores.packed, torch.zeros(2), torch.tensor([[0.0, 10.0]]),
                            torch.zeros(1, 8), torch.zeros(1, 3), torch.zeros(1, 3),
                            torch.zeros(1, 3))
    calls = []

    def executor(prefix, action, *, policy, max_len):
        calls.append(action)
        if len(calls) == 2:
            return CopyExecution(action, b"ok")
        raise TransducerFault(FaultCode.CANVAS, "test refusal")

    result = choose_action(scores, 0, _valid_prefix(), max_len=128, executor=executor)
    assert not result.emit and result.expansions == len(calls) == 2
    assert calls[0].source_start == 0
    assert result.action == calls[-1]
    assert result.expansions <= RELATION_KBEST_ACTIONS
    assert result.faults == (("canvas", 1),)
    assert result.exit_reason == "accepted_copy"


def _search_scores(gate: Tensor, span: Tensor, d4: Tensor, dx: Tensor, dy: Tensor,
                   count: Tensor) -> RelationScores:
    packed = _packed(queries=1)
    return RelationScores(packed, span, gate, d4, dx, dy, count)


def test_search_rethrows_structural_r2_faults_but_converts_action_faults_only():
    scores = _search_scores(torch.tensor([[0.0, 10.0]]), torch.zeros(2),
                            torch.zeros(1, 8), torch.zeros(1, 3), torch.zeros(1, 3),
                            torch.zeros(1, 3))

    with pytest.raises(TransducerFault) as structural:
        choose_action(scores, 0, b"\x01\x02", max_len=128)
    assert structural.value.code is FaultCode.MALFORMED_PREFIX

    with pytest.raises(TransducerFault) as structural:
        choose_action(scores, 0, assemble("REPEAT 2 0 0\nMOVE 1 1\nENDREP"),
                      max_len=128)
    assert structural.value.code is FaultCode.NON_FLAT_PREFIX

    def action_specific(*_args, **_kwargs):
        raise TransducerFault(FaultCode.CANVAS, "candidate leaves canvas")

    result = choose_action(scores, 0, _valid_prefix(), max_len=128, executor=action_specific)
    assert result.emit and result.expansions == RELATION_KBEST_ACTIONS
    assert result.faults == (("canvas", RELATION_KBEST_ACTIONS),)
    assert result.exit_reason == "attempt_cap"

    for code in (FaultCode.ACTION_TYPE, FaultCode.UNSUPPORTED_D4):
        def impossible(*_args, code=code, **_kwargs):
            raise TransducerFault(code, "must surface")

        with pytest.raises(TransducerFault) as refusal:
            choose_action(scores, 0, _valid_prefix(), max_len=128, executor=impossible)
        assert refusal.value.code is code


def test_prefix_is_preflighted_before_emit_and_empty_candidate_returns():
    emit_scores = _search_scores(torch.tensor([[10.0, 0.0]]), torch.zeros(2),
                                 torch.zeros(1, 8), torch.zeros(1, 3),
                                 torch.zeros(1, 3), torch.zeros(1, 3))
    with pytest.raises(TransducerFault) as malformed:
        choose_action(emit_scores, 0, b"\x01\x02", max_len=128)
    assert malformed.value.code is FaultCode.MALFORMED_PREFIX
    empty = PackedCandidates(
        query_states=torch.zeros(1, 8), start_states=torch.empty(0, 8),
        stop_states=torch.empty(0, 8), length_bins=torch.empty(0, dtype=torch.long),
        source_start=torch.empty(0, dtype=torch.long), source_stop=torch.empty(0, dtype=torch.long),
        row_splits=torch.tensor([0, 0]),
    )
    empty_scores = RelationScores(empty, torch.empty(0), torch.tensor([[0.0, 0.0]]),
                                  torch.zeros(1, 8), torch.zeros(1, 3),
                                  torch.zeros(1, 3), torch.zeros(1, 3))
    with pytest.raises(TransducerFault) as malformed:
        choose_action(empty_scores, 0, b"\x01\x02", max_len=128)
    assert malformed.value.code is FaultCode.MALFORMED_PREFIX


def test_search_pins_emit_copy_all_invalid_and_cap_exhaustion_paths():
    factors = (torch.zeros(1, 8), torch.zeros(1, 3), torch.zeros(1, 3), torch.zeros(1, 3))
    calls = []

    def valid(prefix, action, *, policy, max_len):
        calls.append(action)
        return CopyExecution(action, b"ok")

    emit_scores = _search_scores(torch.tensor([[10.0, 0.0]]), torch.zeros(2), *factors)
    emit = choose_action(emit_scores, 0, _valid_prefix(), max_len=128, executor=valid)
    assert emit.emit and emit.expansions == 0 and emit.faults == ()
    assert emit.exit_reason == "emit_dominates" and calls == []
    copy_scores = _search_scores(torch.tensor([[0.0, 10.0]]), torch.zeros(2), *factors)
    copy = choose_action(copy_scores, 0, _valid_prefix(), max_len=128, executor=valid)
    assert not copy.emit and copy.expansions == 1 and len(calls) == 1
    assert copy.faults == () and copy.exit_reason == "accepted_copy"

    def invalid(*_args, **_kwargs):
        raise TransducerFault(FaultCode.CANVAS, "invalid")

    exhausted = choose_action(copy_scores, 0, _valid_prefix(), max_len=128, executor=invalid)
    assert exhausted.emit and exhausted.expansions == RELATION_KBEST_ACTIONS
    assert exhausted.faults == (("canvas", RELATION_KBEST_ACTIONS),)
    assert exhausted.exit_reason == "attempt_cap"

    packed = PackedCandidates(
        query_states=torch.zeros(1, 8), start_states=torch.empty(0, 8),
        stop_states=torch.empty(0, 8), length_bins=torch.empty(0, dtype=torch.long),
        source_start=torch.empty(0, dtype=torch.long), source_stop=torch.empty(0, dtype=torch.long),
        row_splits=torch.tensor([0, 0]),
    )
    no_candidates = RelationScores(
        packed, torch.empty(0), torch.tensor([[0.0, 10.0]]),
        torch.zeros(1, 8), torch.zeros(1, 3), torch.zeros(1, 3), torch.zeros(1, 3),
    )
    empty = choose_action(no_candidates, 0, _valid_prefix(), max_len=128, executor=valid)
    assert empty.emit and empty.expansions == 0 and empty.faults == ()
    assert empty.exit_reason == "no_candidates"


def test_search_support_exhaustion_metadata_is_defensive_and_exact(monkeypatch):
    # One span has 8*3*3*3 = 216 actions, above frozen cap 32. Raise cap only
    # to reach defensive exhaustion branch without changing frozen supports.
    base = _packed(queries=1)
    packed = PackedCandidates(
        query_states=base.query_states,
        start_states=base.start_states[:1],
        stop_states=base.stop_states[:1],
        length_bins=base.length_bins[:1],
        source_start=base.source_start[:1],
        source_stop=base.source_stop[:1],
        row_splits=torch.tensor([0, 1]),
    )
    scores = RelationScores(
        packed, torch.zeros(1), torch.tensor([[0.0, 10.0]]),
        torch.zeros(1, 8), torch.zeros(1, 3), torch.zeros(1, 3), torch.zeros(1, 3),
    )

    def invalid(*args, **kwargs):
        raise TransducerFault(FaultCode.CANVAS, "invalid")

    monkeypatch.setattr("dm.models.relation.RELATION_KBEST_ACTIONS", 217)
    result = choose_action(scores, 0, _valid_prefix(), max_len=128, executor=invalid)
    assert result.emit and result.expansions == 216
    assert result.faults == (("canvas", 216),)
    assert result.exit_reason == "support_exhausted"


def test_search_matches_bruteforce_ordering_and_executor_calls_on_randomized_logits():
    generator = torch.Generator().manual_seed(71)
    for _ in range(100):
        scores = _search_scores(
            torch.randn(1, 2, generator=generator), torch.randn(2, generator=generator),
            torch.randn(1, 8, generator=generator), torch.randn(1, 3, generator=generator),
            torch.randn(1, 3, generator=generator), torch.randn(1, 3, generator=generator),
        )
        gate, span = scores.gate_log_probs(0), scores.span_log_probs(0)
        d4, dx, dy, count = scores.factor_log_probs(0)
        ordered = []
        for indexes in product(range(2), range(8), range(3), range(3), range(3)):
            score = float(gate[COPY] + span[indexes[0]] + d4[indexes[1]] + dx[indexes[2]]
                          + dy[indexes[3]] + count[indexes[4]])
            ordered.append((score, indexes))
        ordered.sort(key=lambda item: (-item[0], item[1]))
        expected = [item for item in ordered if item[0] >= float(gate[EMIT])][:RELATION_KBEST_ACTIONS]
        calls = []

        def executor(prefix, action, *, policy, max_len, calls=calls):
            calls.append(action)
            valid = (action.source_start // 32 + action.step.d4.code
                     + action.step.dx // 32 + action.step.dy // 32 + action.total_count) % 7 == 0
            if valid:
                return CopyExecution(action, b"ok")
            raise TransducerFault(FaultCode.CANVAS, "invalid candidate")

        result = choose_action(scores, 0, _valid_prefix(), max_len=128, executor=executor)
        first_valid = next((item for item in expected if (
            item[1][0] + item[1][1] + (-1 + item[1][2]) + (-1 + item[1][3])
            + (2 + item[1][4])) % 7 == 0), None)
        assert result.emit is (first_valid is None)
        assert result.expansions == (len(expected) if first_valid is None
                                     else expected.index(first_valid) + 1)
        assert len(calls) == result.expansions
        expected_calls = expected if first_valid is None else expected[:expected.index(first_valid) + 1]
        expected_keys = [
            (item[1][0] * 32, item[1][1], -32 + item[1][2] * 32,
             -32 + item[1][3] * 32, 2 + item[1][4])
            for item in expected_calls
        ]
        assert [(action.source_start, action.step.d4.code, action.step.dx,
                 action.step.dy, action.total_count) for action in calls] == expected_keys
        if first_valid is not None:
            assert result.action is not None
            assert (result.action.source_start, result.action.step.d4.code,
                    result.action.step.dx, result.action.step.dy,
                    result.action.total_count) == expected_keys[-1]


def test_compensating_tie_uses_canonical_full_action_order():
    # Span 0/D4 0 and span 1/D4 1 have equal total score; the complete canonical
    # tuple must decide their executor order rather than the factors' rank slots.
    scores = _search_scores(torch.tensor([[0.0, 10.0]]), torch.tensor([1.0, 0.0]),
                            torch.tensor([[0.0, 1.0, -10, -10, -10, -10, -10, -10]]),
                            torch.tensor([[0.0, -10.0, -20.0]]),
                            torch.tensor([[0.0, -10.0, -20.0]]),
                            torch.tensor([[0.0, -10.0, -20.0]]))
    calls = []

    def invalid(prefix, action, *, policy, max_len):
        calls.append(action)
        raise TransducerFault(FaultCode.CANVAS, "exercise ordering")

    choose_action(scores, 0, _valid_prefix(), max_len=128, executor=invalid)
    assert [(action.source_start // 32, action.step.d4.code) for action in calls[1:3]] == [(0, 0), (1, 1)]


def test_packed_width_benchmark_at_contract_and_cap_widths():
    head = RelationHead(128).eval()
    for width in (320, 768):
        packed = PackedCandidates(
            query_states=torch.randn(1, 128), start_states=torch.randn(width, 128),
            stop_states=torch.randn(width, 128), length_bins=torch.zeros(width, dtype=torch.long),
            source_start=torch.arange(width, dtype=torch.long) * 8,
            source_stop=torch.arange(width, dtype=torch.long) * 8 + 7,
            row_splits=torch.tensor([0, width]),
        )
        started = time.perf_counter()
        scores = head(packed)
        elapsed = time.perf_counter() - started
        assert scores.span_logits.shape == (width,) and torch.isfinite(scores.span_logits).all()
        assert elapsed >= 0.0  # Recorded by the test runner; deliberately non-gating.


def _import_names(source, package):
    names = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = resolve_name("." * node.level + (node.module or ""), package)
            names.append(base)
            names.extend(base + "." + alias.name for alias in node.names)
    return names


@pytest.mark.parametrize("source", [
    "from ..eval import evidence", "import dm.eval.evidence",
    "from .. import eval", "from dm.data import relation",
])
def test_runtime_import_audit_detects_forbidden_forms(source):
    names = _import_names(source, "dm.models")
    assert any(name.startswith(("dm.eval", "dm.data.relation")) for name in names)


def test_runtime_modules_have_no_corpus_or_training_evaluation_dependency():
    for module_name in (
        "dm.models.relation",
        "dm.models.transformer",
        "dm.relation.candidates",
        "dm.relation.decoding",
        "dm.relation.queue",
        "dm.relation.spec",
    ):
        source = inspect.getsource(__import__(module_name, fromlist=["*"]))
        imports = _import_names(source, module_name.rpartition(".")[0])
        assert not any(name.startswith(("dm.data.relation", "dm.train", "dm.eval"))
                       for name in imports), module_name


def test_decode_callable_and_event_payloads_are_target_free():
    from dm.relation.decoding import (
        ActionDecision,
        RequestCompleted,
        RequestStarted,
        RowStopped,
        ScoreDetail,
    )

    forbidden = ("target", "case", "corpus", "equivalence", "oracle")
    parameters = inspect.signature(DrawingLM.generate_relation).parameters
    assert not any(any(word in name for word in forbidden[:-1]) for name in parameters)
    assert "oracle_actions" in parameters  # Explicitly isolated diagnostic input.
    for record in (RequestStarted, ActionDecision, ScoreDetail, RowStopped, RequestCompleted):
        names = {field.name for field in fields(record)}
        assert not any(any(word in name for word in forbidden) for name in names)

    class TargetBearingSentinel:
        def get(self, *args, **kwargs):
            pytest.fail("predicted path inspected oracle/target-bearing input")

    model = DrawingLM(Config(vocab_size=258, d_model=16, n_layers=1, n_heads=2,
                             relation_schema="span_affine_v1"))
    for mode in ("standard", "predicted_copy"):
        with pytest.raises(ValueError, match="oracle"):
            model.generate_relation(
                [b""], 0, mode=mode, literal_policy="content_bytes_v1",
                oracle_actions=TargetBearingSentinel(),
            )
