"""Model-blind controls for the separate demonstrated-affine task.

This module is not an inference adapter. The reference solver receives bytes
only; suffixes and annotations enter the audit only after its prediction.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product

from dm.data.augment import Affine, apply
from dm.data.relation import RelationCase, copy_actions_at, derivations
from dm.isa.spec import Op
from dm.isa.transform import Transform
from dm.relation.candidates import candidate_spans
from dm.relation.spec import (
    COUNT_SUPPORT,
    D4_SUPPORT,
    TRANSLATION_SUPPORT,
    CopyActionKey,
)
from dm.relation.transducer import execute_copy, prefix_index

NAMESPACE = "direction4-demonstrated-affine-v1"
TRANSFORMS = tuple(Transform.of(*values) for values in product(
    D4_SUPPORT, TRANSLATION_SUPPORT, TRANSLATION_SUPPORT))


class TaskInvalid(ValueError):
    """A concrete task witness must be resolved before learning."""


@dataclass(frozen=True)
class ReferencePrediction:
    task_class: str
    transforms: tuple[Transform, ...]
    target: bytes | None
    query_start: int | None


def motifs(prefix: bytes) -> tuple[bytes, ...]:
    """Parse MOVE-delimited, coordinate-only motifs without future boundaries.

    Each motif contains MOVE and at least two LINEs. The final LINE count is
    fixed at two in this initial grammar, so a truncated final motif is not
    mistaken for a complete query. Other L0 programs are generic controls.
    """
    marks, _, ops = prefix_index(prefix)
    if len(ops) != 9 or ops != (Op.MOVE, Op.LINE, Op.LINE) * 3:
        return ()
    return tuple(prefix[marks[i]:marks[i + 3]] for i in (0, 3, 6))


def solve_prefix(prefix: bytes) -> ReferencePrediction:
    """Derive the compatible affine set from ordinary A, image(A), B bytes."""
    parts = motifs(prefix)
    if not parts:
        return ReferencePrediction("generic", (), None, None)
    source, demonstration, query = parts
    compatible = tuple(t for t in TRANSFORMS
                       if apply(source, Affine.of(t)) == demonstration)
    if not compatible:
        return ReferencePrediction("broken", (), None, len(source) + len(demonstration))
    images = {apply(query, Affine.of(t)) for t in compatible}
    if None in images or len(images) != 1:
        raise TaskInvalid("demonstration lacks a unique feasible target")
    return ReferencePrediction("positive", compatible, images.pop(),
                               len(source) + len(demonstration))


def orbit_key(motif: bytes) -> bytes:
    """Canonicalize coordinate geometry under D4 and arbitrary translation.

    Unlike an absolute-byte identity this also excludes transformed source
    equivalents from held-out families. Opcode skeleton remains part of key.
    """
    _, coords, _ = prefix_index(motif)
    if not coords:
        raise TaskInvalid("source family has no coordinates")
    keys = []
    for code in D4_SUPPORT:
        transformed = apply(motif, Affine.of(Transform.of(code, 0, 0)))
        if transformed is None:
            raise TaskInvalid("D4 source transformation left canvas")
        xs, ys = transformed[coords[0]], transformed[coords[0] + 1]
        # Signed integers avoid clipping a normalized geometry at canvas edges.
        key = []
        coordinate_bytes = {at for p in coords for at in (p, p + 1)}
        for at, value in enumerate(transformed):
            key.append(value - (xs if at in coords else ys)
                       if at in coordinate_bytes else value)
        keys.append(tuple(key))
    return repr(min(keys)).encode("ascii")


def full_target_keys(prefix: bytes, target: bytes) -> tuple[CopyActionKey, ...]:
    """Retain every one-action full-target key, including alternate spans."""
    flat = prefix + target
    return tuple(CopyActionKey(a.boundary, a.source_start, a.source_stop,
                              a.step, a.total_count)
                 for a in copy_actions_at(flat, len(prefix), len(flat))
                 if a.target_stop == len(flat))


def audit_negative(prefix: bytes, target: bytes) -> dict:
    """Reject even a partial COPY at any literal-reachable continuation point.

    This is stronger than rejecting complete COPY-only derivations. Any mixed
    literal/COPY derivation must have a first COPY; exhaustively ruling out that
    first step also rules out alternate spans and multi-action continuations.
    HALT is audited but cannot legally be part of a COPY source.
    """
    if solve_prefix(prefix).task_class == "positive":
        raise TaskInvalid("negative prompt has a compatible demonstration")
    flat = prefix + target + bytes([Op.HALT])
    marks, _, _ = prefix_index(flat)
    checked = 0
    for boundary in marks:
        if len(prefix) <= boundary < len(flat):
            checked += 1
            actions = copy_actions_at(flat, boundary, len(flat))
            if actions:
                raise TaskInvalid(f"negative COPY witness at {boundary}: {actions[0].key()}")
    return {"continuation_boundaries": checked, "correct_copy_steps": 0}


def audit_positive(prefix: bytes, target: bytes) -> dict:
    prediction = solve_prefix(prefix)
    if prediction.task_class != "positive" or prediction.target != target:
        raise TaskInvalid("prefix-only prediction differs from stored positive target")
    keys = full_target_keys(prefix, target)
    if not keys:
        raise TaskInvalid("positive has no full-target production action")
    for key in keys:
        if key.total_count not in COUNT_SUPPORT:
            raise TaskInvalid("correct key exceeds production support")
        if execute_copy(prefix, key, max_len=255).appended != target:
            raise TaskInvalid("positive oracle differs from target")
    # Walk every COPY-reachable successor, not only annotated first decisions.
    flat = prefix + target
    pending, visited, edges = [len(prefix)], set(), []
    while pending:
        boundary = pending.pop()
        if boundary in visited or boundary == len(flat):
            continue
        visited.add(boundary)
        actions = derivations(flat, boundary, boundary, len(flat))
        if not actions:
            raise TaskInvalid(f"positive derivation dead end at {boundary}")
        for action in actions:
            key = CopyActionKey(action.boundary, action.source_start,
                                action.source_stop, action.step, action.total_count)
            appended = execute_copy(flat[:boundary], key, max_len=255).appended
            if appended != flat[boundary:action.target_stop]:
                raise TaskInvalid("derivation executor differs from stored continuation")
            edges.append(list(action.key()))
            pending.append(action.target_stop)
    return {"compatible_transforms": [[t.d4.code, t.dx, t.dy]
                                      for t in prediction.transforms],
            "full_target_keys": [list(k.key()) for k in keys],
            "derivation_edges": edges, "reachable_copy_queries": len(visited)}


def audit_causal_queries(cases: Sequence[RelationCase]) -> dict:
    """Audit trainer scheduling plus every correct COPY successor across splits.

    Demo EMIT positions are scheduling labels, never non-derivability evidence.
    Shared positive inputs must agree on full remaining target, not just on an
    intersecting first-step set. Fail closed on a gate conflict at any successor.
    """
    from dm.train_relation import RelationBatchPlanner, TrainingCorpus

    planner = RelationBatchPlanner(TrainingCorpus.engineering_cases(cases), max_len=256)
    seen: dict[bytes, tuple[int, frozenset, bytes | None, str]] = {}
    labels = {"emit": 0, "copy": 0}

    def record(prefix, gate, keys, target, case_id):
        labels["copy" if gate else "emit"] += 1
        value = (gate, frozenset(keys), target, case_id)
        old = seen.get(prefix)
        if old is not None:
            if old[0] != gate:
                raise TaskInvalid(f"equal-prefix gate conflict: {old[3]} / {case_id}")
            if gate and (old[2] != target or not old[1].intersection(value[1])):
                raise TaskInvalid(f"equal-prefix action conflict: {old[3]} / {case_id}")
            if gate:
                value = (gate, old[1].intersection(value[1]), target, case_id)
        seen[prefix] = value

    for index, case in enumerate(cases):
        # Public planner route guarantees this audit sees actual training queries.
        plan = planner.batch([index])
        for query in plan.queries:
            boundary = query.boundary
            positive = next((a for a in case.actions if a.boundary == boundary), None)
            end = positive.target_stop if positive else boundary
            keys = [a.key() for a in derivations(case.flat, boundary, boundary, end)]
            record(case.flat[:boundary], int(positive is not None), keys,
                   case.flat[boundary:end] if positive else None, case.case_id)
        for action in case.actions:
            pending, visited = [action.boundary], set()
            while pending:
                boundary = pending.pop()
                if boundary in visited or boundary == action.target_stop:
                    continue
                visited.add(boundary)
                steps = derivations(case.flat, boundary, boundary, action.target_stop)
                record(case.flat[:boundary], 1, [a.key() for a in steps],
                       case.flat[boundary:action.target_stop], case.case_id)
                pending.extend(a.target_stop for a in steps)
    return {"unique_prefixes": len(seen), "presentations_including_successor_audit": labels}


def counterfactual_first_copy(prefix: bytes, source_start: int, source_stop: int) -> dict:
    """Model-blind feasible-plan panel for a fixed old-task source/distractor prefix."""
    if (source_start, source_stop) not in candidate_spans(prefix, len(prefix)):
        raise TaskInvalid("counterfactual source is not a production candidate")
    from dm.relation.transducer import TransducerFault

    targets: dict[bytes, list] = {}
    for transform in TRANSFORMS:
        key = CopyActionKey(len(prefix), source_start, source_stop, transform, 2)
        try:
            target = execute_copy(prefix, key, max_len=255).appended
        except TransducerFault:
            continue
        targets.setdefault(target, []).append(list(key.key()))
    return {"prefix_hex": prefix.hex(), "distinct_compatible_suffixes": len(targets),
            "targets": [{"target_hex": target.hex(), "keys": keys}
                        for target, keys in sorted(targets.items())]}
