"""Bounded, model-local scoring for Direction 4 relation actions.

This module deliberately depends on the target-free runtime only.  Candidate
construction belongs to callers (and, for corpus work, ``dm.data.relation``);
the model receives an already bounded packed representation and never imports a
corpus builder, trainer, or evaluator.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..isa.transform import D4, Transform
from ..relation import (
    COUNT_SUPPORT,
    D4_SUPPORT,
    PREDICTED_SUPPORT,
    TRANSLATION_SUPPORT,
    CopyActionKey,
    CopyExecution,
    FaultCode,
    TransducerFault,
    execute_copy,
    validate_prefix,
)

# These values are frozen by the R1 protocol.  They are intentionally local to
# model code: importing their corpus owner would make ordinary model imports
# construct a data dependency.  The R3 contract tests bind these values to the
# protocol's public constants.
RELATION_RANK = 32
RELATION_LENGTH_BINS = 8
RELATION_MAX_CANDIDATE_SPANS = 768
RELATION_KBEST_ACTIONS = 32
RELATION_LENGTH_BIN_EDGES = (12, 21, 33, 48, 69, 99, 141, 192)
RELATION_SCHEMAS = ("none", "span_affine_v1")

EMIT, COPY = 0, 1

# Only these codes can describe an otherwise-valid prefix paired with a bad
# candidate.  The action key is constructed internally, so ACTION_TYPE and all
# unlisted/future faults are impossible at this seam and must stay loud.
_CANDIDATE_LOCAL_FAULTS = frozenset({
    FaultCode.BOUNDARY,
    FaultCode.SOURCE_RANGE,
    FaultCode.SOURCE_EMPTY,
    FaultCode.SOURCE_BYTES,
    FaultCode.SOURCE_INSTRUCTIONS,
    FaultCode.SOURCE_GAP,
    FaultCode.SOURCE_COORDINATES,
    FaultCode.HALT_SOURCE,
    FaultCode.CANVAS,
    FaultCode.MAX_LENGTH,
})


def relation_parameter_count(d_model: int) -> int:
    """Exact bias-free parameter count of the fixed R3 head."""
    return (d_model + 3 * d_model * RELATION_RANK
            + RELATION_LENGTH_BINS * RELATION_RANK
            + d_model * (2 + len(D4_SUPPORT) + 2 * len(TRANSLATION_SUPPORT)
                         + len(COUNT_SUPPORT)))


def length_bin(span_bytes: int) -> int:
    """Return the one frozen length bin which contains ``span_bytes``."""
    for index, edge in enumerate(RELATION_LENGTH_BIN_EDGES):
        if span_bytes <= edge:
            return index
    raise RelationLayoutError("a source span exceeds the frozen byte cap")


def boundary_states(top_states: Tensor, byte_offsets: Tensor) -> Tensor:
    """Select normalized states using the frozen BOS/byte boundary convention.

    Position zero is BOS and therefore boundary byte offset zero; boundary
    offset ``b > 0`` is model position ``b`` (after byte ``b - 1``).  Candidate
    construction remains caller-owned, but this small checked selection avoids
    each future adapter spelling that off-by-one rule independently.
    """
    if top_states.ndim != 2 or not top_states.is_floating_point():
        raise RelationLayoutError("top_states must be a floating [positions, D] tensor")
    if byte_offsets.ndim != 1 or byte_offsets.dtype != torch.long:
        raise RelationLayoutError("byte_offsets must be an int64 [N] tensor")
    if byte_offsets.device != top_states.device:
        raise RelationLayoutError("byte_offsets must share the state device")
    if byte_offsets.numel() and (int(byte_offsets.min()) < 0
                                 or int(byte_offsets.max()) >= top_states.shape[0]):
        raise RelationLayoutError("a byte boundary is outside available BOS/byte states")
    return top_states[byte_offsets]


class RelationLayoutError(ValueError):
    """A caller supplied a malformed packed layout or action equivalence."""


class RMSNorm(nn.Module):
    """Local affine RMSNorm, kept independent of the transformer module."""

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, states: Tensor) -> Tensor:
        return self.weight * states * torch.rsqrt(
            states.float().square().mean(-1, keepdim=True).to(states.dtype) + self.eps
        )


@dataclass(frozen=True)
class PackedCandidates:
    """A ragged, bounded collection of source spans for boundary queries.

    ``row_splits`` is the only ownership map: candidates in
    ``row_splits[q]:row_splits[q + 1]`` belong to query ``q``.  No padding row
    exists, so a candidate that is not real cannot enter a span denominator.
    """

    query_states: Tensor
    start_states: Tensor
    stop_states: Tensor
    length_bins: Tensor
    source_start: Tensor
    source_stop: Tensor
    row_splits: Tensor

    def validate(self, d_model: int, head_dtype: torch.dtype) -> None:
        q, start, stop = self.query_states, self.start_states, self.stop_states
        if q.ndim != 2 or q.shape[1] != d_model:
            raise RelationLayoutError(f"query_states must have shape [Q, {d_model}]")
        if start.ndim != 2 or start.shape != stop.shape or start.shape[1] != d_model:
            raise RelationLayoutError(f"start_states and stop_states must have shape [S, {d_model}]")
        if not q.is_floating_point() or not start.is_floating_point() or not stop.is_floating_point():
            raise RelationLayoutError("all state tensors must be floating point")
        if q.dtype != start.dtype or q.dtype != stop.dtype:
            raise RelationLayoutError("all state tensors must have one identical dtype")
        if q.dtype != head_dtype:
            raise RelationLayoutError(
                f"state dtype {q.dtype} does not match relation-head dtype {head_dtype}")
        if len({q.device, start.device, stop.device}) != 1:
            raise RelationLayoutError("state tensors must share one device")
        spans = start.shape[0]
        for name, value, size in (
            ("length_bins", self.length_bins, spans),
            ("source_start", self.source_start, spans),
            ("source_stop", self.source_stop, spans),
        ):
            if value.ndim != 1 or value.shape[0] != size or value.dtype != torch.long:
                raise RelationLayoutError(f"{name} must be an int64 [S] tensor")
            if value.device != q.device:
                raise RelationLayoutError(f"{name} must share the state device")
        splits = self.row_splits
        if splits.ndim != 1 or splits.shape[0] != q.shape[0] + 1 or splits.dtype != torch.long:
            raise RelationLayoutError("row_splits must be an int64 [Q + 1] tensor")
        if splits.device != q.device:
            raise RelationLayoutError("row_splits must share the state device")
        # Scalar extraction here is after all bounded-shape checks and does not
        # allocate a dense candidate representation.
        values = splits.detach().cpu().tolist()
        if not values or values[0] != 0 or values[-1] != spans or any(
                right < left for left, right in pairwise(values)):
            raise RelationLayoutError("row_splits must monotonically partition exactly S spans")
        if any(right - left > RELATION_MAX_CANDIDATE_SPANS
               for left, right in pairwise(values)):
            raise RelationLayoutError("a query exceeds the frozen candidate-span cap")
        if self.length_bins.numel() and (
                int(self.length_bins.min()) < 0 or int(self.length_bins.max()) >= RELATION_LENGTH_BINS):
            raise RelationLayoutError("length_bins contains an out-of-range index")
        if self.source_start.numel() and not bool(torch.all(
                (0 <= self.source_start) & (self.source_start < self.source_stop))):
            raise RelationLayoutError(
                "each source span must have non-negative source_start < source_stop")
        if self.source_start.numel():
            widths = self.source_stop - self.source_start
            if int(widths.max()) > RELATION_LENGTH_BIN_EDGES[-1]:
                raise RelationLayoutError("a source span exceeds the frozen byte cap")
            expected_bins = torch.tensor(
                [length_bin(int(width)) for width in widths.detach().cpu().tolist()],
                dtype=torch.long, device=self.length_bins.device,
            )
            if not torch.equal(self.length_bins, expected_bins):
                raise RelationLayoutError("length_bins does not match source span byte lengths")

    @property
    def queries(self) -> int:
        return self.query_states.shape[0]

    @property
    def spans(self) -> int:
        return self.start_states.shape[0]


@dataclass(frozen=True)
class RelationScores:
    """Raw factor logits for a validated :class:`PackedCandidates` layout."""

    packed: PackedCandidates
    span_logits: Tensor
    gate_logits: Tensor
    d4_logits: Tensor
    dx_logits: Tensor
    dy_logits: Tensor
    count_logits: Tensor

    def _factor_log_probs(self, logits: Tensor, query: int) -> Tensor:
        if not 0 <= query < self.packed.queries:
            raise RelationLayoutError(f"query index {query} is outside [0, {self.packed.queries})")
        return F.log_softmax(logits[query].float(), dim=-1)

    def span_log_probs(self, query: int) -> Tensor:
        if not 0 <= query < self.packed.queries:
            raise RelationLayoutError(f"query index {query} is outside [0, {self.packed.queries})")
        begin, end = self.packed.row_splits[query:query + 2].detach().cpu().tolist()
        if begin == end:
            return self.span_logits.new_empty((0,), dtype=torch.float32)
        return F.log_softmax(self.span_logits[begin:end].float(), dim=0)

    def gate_log_probs(self, query: int) -> Tensor:
        return self._factor_log_probs(self.gate_logits, query)

    def factor_log_probs(self, query: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return (self._factor_log_probs(self.d4_logits, query),
                self._factor_log_probs(self.dx_logits, query),
                self._factor_log_probs(self.dy_logits, query),
                self._factor_log_probs(self.count_logits, query))


@dataclass(frozen=True)
class EquivalentAction:
    """One valid joint action, represented using canonical factor *values*."""

    query: int
    span: int
    d4: int
    dx: int
    dy: int
    total_count: int

    def canonical_key(self) -> tuple[int, int, int, int, int, int]:
        return (self.query, self.span, self.d4, self.dx, self.dy, self.total_count)


def _support_index(values: tuple[int, ...], value: int, name: str) -> int:
    if type(value) is not int or value not in values:
        raise RelationLayoutError(f"{name}={value!r} is outside frozen support {values}")
    return values.index(value)


def _validated_action(scores: RelationScores, action: EquivalentAction
                      ) -> tuple[int, int, int, int, int, int]:
    """Validate every field before it can enter a set key or index a tensor."""
    if not isinstance(action, EquivalentAction):
        raise RelationLayoutError("an equivalence entry is not an EquivalentAction")
    query = action.query
    if type(query) is not int or not 0 <= query < scores.packed.queries:
        raise RelationLayoutError("equivalence query is outside the packed layout")
    begin, end = scores.packed.row_splits[query:query + 2].detach().cpu().tolist()
    if type(action.span) is not int or not begin <= action.span < end:
        raise RelationLayoutError("equivalence span is not a candidate owned by its query")
    d4 = _support_index(D4_SUPPORT, action.d4, "d4")
    dx = _support_index(TRANSLATION_SUPPORT, action.dx, "dx")
    dy = _support_index(TRANSLATION_SUPPORT, action.dy, "dy")
    count = _support_index(COUNT_SUPPORT, action.total_count, "total_count")
    return query, begin, d4, dx, dy, count


def _action_log_probability(scores: RelationScores, action: EquivalentAction) -> Tensor:
    query, begin, d4, dx, dy, count = _validated_action(scores, action)
    span = scores.span_log_probs(query)[action.span - begin]
    d4_lp, dx_lp, dy_lp, count_lp = scores.factor_log_probs(query)
    return span + d4_lp[d4] + dx_lp[dx] + dy_lp[dy] + count_lp[count]


def joint_valid_nll(scores: RelationScores,
                    equivalences: Sequence[Sequence[EquivalentAction]]) -> Tensor:
    """Unreduced ``-log sum p(action)`` for each positive boundary.

    Each inner set must contain alternatives for exactly one query.  Duplicate
    keys are rejected rather than silently manufacturing probability mass.
    All normalisation and aggregation remains float32 even when the trunk ran in
    a reduced dtype.
    """
    losses: list[Tensor] = []
    for alternatives in equivalences:
        if not alternatives:
            raise RelationLayoutError("a positive boundary has no valid equivalent action")
        seen: set[tuple[int, int, int, int, int, int]] = set()
        query: int | None = None
        values: list[Tensor] = []
        for action in alternatives:
            action_query, _, _, _, _, _ = _validated_action(scores, action)
            if query is None:
                query = action_query
            elif action_query != query:
                raise RelationLayoutError("one equivalence set crosses packed queries")
            # `_validated_action` makes this key exclusively fixed-width ints;
            # malformed public fields cannot leak an unhashable-TypeError here.
            key = action.canonical_key()
            if key in seen:
                raise RelationLayoutError("duplicate equivalent action key")
            seen.add(key)
            values.append(_action_log_probability(scores, action))
        losses.append(-torch.logsumexp(torch.stack(values).float(), dim=0))
    if not losses:
        return scores.span_logits.new_empty((0,), dtype=torch.float32)
    return torch.stack(losses)


#: The five conditional factors of one COPY action, in reporting order.  The
#: gate is deliberately absent: it is a separately normalized objective term.
FACTOR_NAMES: tuple[str, ...] = ("span", "d4", "dx", "dy", "count")


def factor_marginal_nll(scores: RelationScores,
                        equivalences: Sequence[Sequence[EquivalentAction]]
                        ) -> dict[str, Tensor]:
    """Diagnostic per-factor valid-value NLL, one ``[P]`` tensor per factor.

    For a positive query with valid joint set ``A``, factor ``f`` reports
    ``-logsumexp(log p_f(v) for v in unique_f(A))``: the mass the factor head
    gives to the *projected* valid values.  Distinct valid tuples that share a
    span or translation count that value once; duplicate complete action keys
    remain an error, exactly as in :func:`joint_valid_nll`.

    This is **not** an objective.  The joint loss is what is optimized, because
    a sum of these marginals is minimized by invalid cross-products of valid
    factor values.  The computation runs detached and non-recording on the same
    validated scores, so calling it cannot change the recorded graph, ``.grad``
    or any RNG stream.  Span probabilities are normalized over the query's real
    candidate segment; the other factors over their frozen categorical support.
    """
    columns: dict[str, list[Tensor]] = {name: [] for name in FACTOR_NAMES}
    with torch.no_grad():
        for alternatives in equivalences:
            if not alternatives:
                raise RelationLayoutError("a positive boundary has no valid equivalent action")
            seen: set[tuple[int, int, int, int, int, int]] = set()
            query: int | None = None
            values: dict[str, list[int]] = {name: [] for name in FACTOR_NAMES}
            for action in alternatives:
                action_query, begin, d4, dx, dy, count = _validated_action(scores, action)
                if query is None:
                    query = action_query
                elif action_query != query:
                    raise RelationLayoutError("one equivalence set crosses packed queries")
                key = action.canonical_key()
                if key in seen:
                    raise RelationLayoutError("duplicate equivalent action key")
                seen.add(key)
                for name, index in zip(FACTOR_NAMES, (action.span - begin, d4, dx, dy, count),
                                       strict=True):
                    if index not in values[name]:
                        values[name].append(index)
            assert query is not None
            log_probs = dict(zip(FACTOR_NAMES,
                                 (scores.span_log_probs(query).detach(),
                                  *(lp.detach() for lp in scores.factor_log_probs(query))),
                                 strict=True))
            for name in FACTOR_NAMES:
                picked = log_probs[name][torch.tensor(values[name], dtype=torch.long,
                                                      device=log_probs[name].device)]
                columns[name].append(-torch.logsumexp(picked.float(), dim=0))
    empty = scores.span_logits.new_empty((0,), dtype=torch.float32)
    return {name: (torch.stack(column) if column else empty.clone())
            for name, column in columns.items()}


class RelationHead(nn.Module):
    """The exact bias-free ``span_affine_v1`` parameter surface."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.norm = RMSNorm(d_model)
        self.query = nn.Linear(d_model, RELATION_RANK, bias=False)
        self.key_start = nn.Linear(d_model, RELATION_RANK, bias=False)
        self.key_stop = nn.Linear(d_model, RELATION_RANK, bias=False)
        self.length = nn.Embedding(RELATION_LENGTH_BINS, RELATION_RANK)
        self.gate = nn.Linear(d_model, 2, bias=False)
        self.d4 = nn.Linear(d_model, len(D4_SUPPORT), bias=False)
        self.dx = nn.Linear(d_model, len(TRANSLATION_SUPPORT), bias=False)
        self.dy = nn.Linear(d_model, len(TRANSLATION_SUPPORT), bias=False)
        self.count = nn.Linear(d_model, len(COUNT_SUPPORT), bias=False)

    def forward(self, packed: PackedCandidates) -> RelationScores:
        packed.validate(self.d_model, self.query.weight.dtype)
        q = self.norm(packed.query_states)
        starts = self.norm(packed.start_states)
        stops = self.norm(packed.stop_states)
        q_rank = self.query(q)
        keys = self.key_start(starts) + self.key_stop(stops) + self.length(packed.length_bins)
        spans = (q_rank.new_zeros((packed.spans,)) if packed.spans == 0 else
                 (q_rank.repeat_interleave(
                     packed.row_splits[1:] - packed.row_splits[:-1], dim=0) * keys
                  ).sum(-1) / math.sqrt(RELATION_RANK))
        return RelationScores(
            packed=packed, span_logits=spans, gate_logits=self.gate(q),
            d4_logits=self.d4(q), dx_logits=self.dx(q), dy_logits=self.dy(q),
            count_logits=self.count(q),
        )


@dataclass(frozen=True)
class RelationDecision:
    """One deterministic R3 choice; bytes are never queued or emitted here."""

    emit: bool
    score: float
    action: CopyActionKey | None = None
    execution: CopyExecution | None = None
    expansions: int = 0
    faults: tuple[tuple[str, int], ...] = ()
    exit_reason: str = "no_candidates"


def score_snapshot(scores: RelationScores, query: int, row: int, offset: int):
    """Detach bounded raw scores into owned immutable primitives, never states."""
    from ..relation.decoding import ScoreDetail

    begin, end = scores.packed.row_splits[query:query + 2].tolist()
    def values(tensor):
        return tuple(float(value) for value in tensor.detach().cpu().tolist())
    return ScoreDetail(
        row, offset,
        tuple(zip(scores.packed.source_start[begin:end].tolist(),
                  scores.packed.source_stop[begin:end].tolist(), strict=True)),
        values(scores.gate_logits[query]), values(scores.span_logits[begin:end]),
        values(scores.d4_logits[query]), values(scores.dx_logits[query]),
        values(scores.dy_logits[query]), values(scores.count_logits[query]),
        (("gate", (EMIT, COPY)), ("d4", D4_SUPPORT), ("dx", TRANSLATION_SUPPORT),
         ("dy", TRANSLATION_SUPPORT), ("count", COUNT_SUPPORT)))


def choose_action(scores: RelationScores, query: int, prefix: bytes, *, max_len: int,
                  executor: Callable[..., CopyExecution] = execute_copy) -> RelationDecision:
    """Compare EMIT with the best executor-valid COPY within the frozen cap."""
    # This precedes gate comparison and the empty-candidate return. A malformed
    # prefix is not a legitimate situation in which to choose EMIT.
    validate_prefix(prefix)
    if not 0 <= query < scores.packed.queries:
        raise RelationLayoutError("query index is outside the packed layout")
    begin, end = scores.packed.row_splits[query:query + 2].detach().cpu().tolist()
    for logits in (scores.span_logits[begin:end], scores.gate_logits[query],
                   scores.d4_logits[query], scores.dx_logits[query],
                   scores.dy_logits[query], scores.count_logits[query]):
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("relation head produced nonfinite raw scores")
    gate = scores.gate_log_probs(query)
    emit_score = float(gate[EMIT])
    if begin == end:
        return RelationDecision(True, emit_score)
    span_lp = scores.span_log_probs(query)
    factors = scores.factor_log_probs(query)
    # Each ordering has canonical original-index tie breaks.  Heap entries use
    # positions in these orderings, so the Cartesian lattice is never materialised.
    orders = [sorted(range(end - begin), key=lambda i: (-float(span_lp[i]), i))]
    orders += [sorted(range(logp.numel()), key=lambda i: (-float(logp[i]), i))
               for logp in factors]

    def score_at(indices: tuple[int, int, int, int, int]) -> float:
        return float(gate[COPY] + span_lp[orders[0][indices[0]]]
                     + factors[0][orders[1][indices[1]]]
                     + factors[1][orders[2][indices[2]]]
                     + factors[2][orders[3][indices[3]]]
                     + factors[3][orders[4][indices[4]]])

    start = (0, 0, 0, 0, 0)
    def canonical(indices: tuple[int, int, int, int, int]) -> tuple[int, int, int, int, int]:
        return tuple(orders[dimension][rank] for dimension, rank in enumerate(indices))

    # Score then canonical action indices, not rank positions: factor rankings
    # can permute canonical indices, and ties must match brute-force ordering.
    heap: list[tuple[float, tuple[int, int, int, int, int], tuple[int, int, int, int, int]]] = [
        (-score_at(start), canonical(start), start)
    ]
    seen = {start}
    expansions = 0
    faults: dict[str, int] = {}
    exit_reason = "support_exhausted"
    while heap and expansions < RELATION_KBEST_ACTIONS:
        neg_score, _, indices = heapq.heappop(heap)
        copy_score = -neg_score
        if copy_score < emit_score:
            exit_reason = "emit_dominates"
            break
        span = begin + orders[0][indices[0]]
        key = CopyActionKey(
            boundary=len(prefix), source_start=int(scores.packed.source_start[span]),
            source_stop=int(scores.packed.source_stop[span]),
            step=Transform(D4.of(D4_SUPPORT[orders[1][indices[1]]]),
                           TRANSLATION_SUPPORT[orders[2][indices[2]]],
                           TRANSLATION_SUPPORT[orders[3][indices[3]]]),
            total_count=COUNT_SUPPORT[orders[4][indices[4]]],
        )
        expansions += 1
        try:
            execution = executor(prefix, key, policy=PREDICTED_SUPPORT, max_len=max_len)
        except TransducerFault as fault:
            if fault.code not in _CANDIDATE_LOCAL_FAULTS:
                raise
            faults[fault.code.value] = faults.get(fault.code.value, 0) + 1
            execution = None
        if execution is not None:
            return RelationDecision(False, copy_score, key, execution, expansions,
                                    tuple(sorted(faults.items())), "accepted_copy")
        for dimension, order in enumerate(orders):
            next_indices = list(indices)
            next_indices[dimension] += 1
            candidate = tuple(next_indices)
            if next_indices[dimension] < len(order) and candidate not in seen:
                seen.add(candidate)
                heapq.heappush(heap, (-score_at(candidate), canonical(candidate), candidate))
    if expansions == RELATION_KBEST_ACTIONS:
        exit_reason = "attempt_cap"
    return RelationDecision(True, emit_score, expansions=expansions,
                            faults=tuple(sorted(faults.items())), exit_reason=exit_reason)
