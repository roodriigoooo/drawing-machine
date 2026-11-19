"""Adversarial model-blind controls; no learned model is constructed."""
from dataclasses import replace

import pytest

from dm.data.augment import Affine, apply
from dm.data.relation import CopyAction, RelationCase
from dm.isa.transform import Transform
from dm.relation.demonstrated import (
    TaskInvalid,
    audit_causal_queries,
    audit_negative,
    audit_positive,
    counterfactual_first_copy,
    full_target_keys,
    orbit_key,
    solve_prefix,
)

A = bytes([1, 80, 90, 2, 107, 96, 2, 99, 123])
B = bytes([1, 117, 100, 2, 130, 119, 2, 142, 108])
T = Transform.of(2, 32, 0)


def image(source, transform=T):
    result = apply(source, Affine.of(transform))
    assert result is not None
    return result


def case(transform=T, *, case_id="positive"):
    prefix = A + image(A, transform) + B
    target = image(B, transform)
    action = CopyAction(27, 18, 27, transform, 2, 27, 36)
    flat = prefix + target + b"\x00"
    return RelationCase(case_id, "demonstrated", "positive", flat, flat,
                        (), (), (action,), 27, 36)


def test_prefix_only_prediction_and_counterfactual_changes():
    first, second = case(), case(Transform.of(1, 0, -32))
    assert first.prompt[-9:] == second.prompt[-9:]
    assert first.target != second.target
    assert solve_prefix(first.prompt).target == first.target
    assert solve_prefix(second.prompt).target == second.target
    audit_positive(first.prompt, first.target)
    with pytest.raises(TaskInvalid, match="stored positive target"):
        audit_positive(first.prompt, second.target)


def test_no_future_or_case_metadata_solver_arguments():
    import inspect
    assert tuple(inspect.signature(solve_prefix).parameters) == ("prefix",)
    original = case()
    changed = replace(original, case_id="future-id", groups=("secret",),
                      flat=original.prompt + b"\x00")
    assert solve_prefix(changed.prompt) == solve_prefix(original.prompt)


def test_generic_and_broken_are_prefix_decidable():
    assert solve_prefix(A).task_class == "generic"
    broken = A + B + B
    assert solve_prefix(broken).task_class == "broken"
    # Unrelated target has no correct COPY even after literal decisions.
    target = bytes([1, 77, 101, 2, 112, 141, 2, 151, 124])
    assert audit_negative(broken, target)["correct_copy_steps"] == 0


def test_annotation_removal_and_accidental_negatives_refused():
    positive = case()
    with pytest.raises(TaskInvalid, match="compatible demonstration"):
        audit_negative(positive.prompt, positive.target)
    with pytest.raises(TaskInvalid, match="negative COPY witness"):
        audit_negative(A + B + B, image(B))


def test_alternate_spans_retained():
    prefix = A + A + A
    keys = full_target_keys(prefix, A)
    assert {(k.source_start, k.source_stop) for k in keys} >= {(9, 18), (18, 27)}
    assert len(audit_positive(prefix, A)["full_target_keys"]) == len(keys)


def test_deliberate_equal_prefix_gate_conflict():
    positive = case()
    negative = replace(positive, case_id="conflict", actions=())
    with pytest.raises(TaskInvalid, match="equal-prefix gate conflict"):
        audit_causal_queries((positive, negative))


def test_deliberate_equal_prefix_action_conflict():
    positive = case()
    other = case(Transform.of(1, 0, -32), case_id="conflict")
    other = replace(other, flat=positive.prompt + other.target + b"\x00")
    with pytest.raises(TaskInvalid, match="equal-prefix action conflict"):
        audit_causal_queries((positive, other))


def test_valid_causal_panel_and_termination_query():
    report = audit_causal_queries((case(),))
    assert report["presentations_including_successor_audit"]["copy"] == 2
    assert report["presentations_including_successor_audit"]["emit"] == 10


def test_orbit_key_excludes_transformed_source_equivalents():
    assert orbit_key(A) == orbit_key(image(A))
    assert orbit_key(A) != orbit_key(B)


def test_fixed_fit_fixture_is_balanced_and_reproducible():
    from dm.relation.demonstrated_fixture import audit_fit_fixture, build_fit_fixture

    fixture = build_fit_fixture()
    assert fixture == build_fit_fixture()
    report = audit_fit_fixture(fixture)
    assert len(report["rows"]) == 32
    assert report["matched_bins"] == {
        "positive": {"27:30": 16}, "negative": {"27:30": 16}}
    assert report["model_calls"] == report["optimizer_updates"] == 0
    assert report["training_authorized_by_this_artifact"] is False
    assert len(fixture.donor_pairs) == 8
    for recipient_id, donor_id in fixture.donor_pairs:
        recipient = next(c for c in fixture.cases if c.case_id == recipient_id)
        donor = next(c for c in fixture.cases if c.case_id == donor_id)
        negative = next(c for c in fixture.cases if c.source_case_id == recipient_id)
        assert negative.donor_source_ids == donor.source_ids
        assert negative.prompt[:9] == recipient.prompt[:9]
        assert negative.prompt[9:18] == donor.prompt[9:18]
        assert negative.target == donor.target


def test_negative_later_copy_is_refused():
    # First literal motif is unrelated; following target can copy that newly
    # emitted motif. A first-boundary-only negative audit would miss this.
    target = bytes([1, 77, 101, 2, 112, 141, 2, 151, 124])
    with pytest.raises(TaskInvalid, match="negative COPY witness at 36"):
        audit_negative(A + B + B, target + target)


def test_truncated_grammar_is_not_a_positive():
    from dm.relation.transducer import TransducerFault
    assert solve_prefix(case().prompt[:-3]).task_class == "generic"
    with pytest.raises(TransducerFault):
        solve_prefix(case().prompt[:-1])


def test_old_first_copy_has_multiple_feasible_suffixes():
    report = counterfactual_first_copy(A, 0, len(A))
    assert report["distinct_compatible_suffixes"] == 72
    assert report["prefix_hex"] == A.hex()
