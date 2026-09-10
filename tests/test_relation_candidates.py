"""R4.1: target-free runtime candidate enumeration."""

from __future__ import annotations

import pytest

from dm.data import relation
from dm.eval.relation_r4_verification import verify_candidates
from dm.isa.asm import assemble
from dm.relation.candidates import candidate_spans
from dm.relation.transducer import FaultCode, TransducerFault


def _legacy(flat: bytes, boundary: int) -> tuple[tuple[int, int], ...]:
    """The R1 implementation retained only as an extraction oracle in this test."""
    marks = relation.boundaries(flat)
    return tuple(relation._eligible_spans(flat, marks, boundary,
                                          relation._halt_offsets(flat)))


def test_runtime_enumeration_is_exactly_the_former_corpus_enumeration():
    build = relation.build_venue1(
        24, seed=19, motifs=relation.motif_pool(32, seed=17),
        tuples=relation.venue1_tuples(),
    )
    for case in build.cases:
        for boundary in relation.boundaries(case.flat):
            assert candidate_spans(case.flat, boundary) == _legacy(case.flat, boundary)
            assert relation.candidate_spans(case.flat, boundary) == _legacy(case.flat, boundary)


def test_runtime_parser_is_the_executor_parser_and_fails_loudly():
    malformed = b"\x01\x0a"
    with pytest.raises(TransducerFault) as refusal:
        candidate_spans(malformed, 2)
    assert refusal.value.code is FaultCode.MALFORMED_PREFIX
    # The corpus facade's historical compatibility contract is intentionally
    # narrower: corpus detection represents an unparseable candidate set as empty.
    assert relation.candidate_spans(malformed, 2) == ()


def test_halt_and_non_boundary_cases_are_target_free_and_deterministic():
    flat = assemble("MOVE 10 10\nLINE 20 20\nHALT")
    assert candidate_spans(flat, len(flat) - 2) == ()
    spans = candidate_spans(flat, len(flat))
    assert spans
    assert all(stop < len(flat) for _, stop in spans)


def test_extraction_verifier_visits_every_boundary_and_refuses_drift(monkeypatch):
    built = relation.build_venue1(4, seed=19, motifs=relation.motif_pool(32, seed=17),
                                 tuples=relation.venue1_tuples())
    report = verify_candidates(built.cases)
    assert report["boundaries"] == sum(len(relation.boundaries(case.flat)) for case in built.cases)
    assert report["mismatches"] == 0 and report["candidate_spans"] > 0
    monkeypatch.setattr("dm.eval.relation_r4_verification.candidate_spans", lambda *args: ())
    with pytest.raises(relation.ManifestRefused, match="candidate extraction mismatch"):
        verify_candidates(built.cases)
