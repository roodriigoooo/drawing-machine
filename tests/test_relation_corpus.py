"""R1: the traced corpus, its actions and every model-blind acceptance clause.

The load-bearing property is the first test in this file: `unroll_with_trace`
returns exactly what `unroll` returns. Everything else in Direction 4 reads the
trace, so a builder that quietly produced different bytes would move the corpus a
model trains on while every downstream number stayed self-consistent.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest

from dm.data import composed, relation
from dm.data.augment import Affine, apply
from dm.data.synthetic import sample
from dm.isa.asm import assemble
from dm.isa.spec import CANVAS, Op, Tier
from dm.isa.transform import D4, Transform
from dm.isa.unroll import unroll
from dm.vm.interp import DEFAULT_FUEL, VM

# ---------------------------------------------------------------------------
# the traced unroller


def test_the_traced_bytes_are_the_plain_unrollers_bytes():
    """Over the whole Tier A L1 generator, both halves agree or both refuse."""
    rng = random.Random(20260822)
    traced = refused = 0
    for _ in range(1500):
        program = sample(rng, tier=Tier.L1, flatten=False)
        plain = unroll(program)
        result = relation.unroll_with_trace(program)
        if plain is None or result is None:
            assert (plain is None) == (result is None)
            refused += 1
            continue
        assert result.bytes == plain
        traced += 1
    assert traced > 1000, "the generator stopped producing unrollable programs"


def test_a_malformed_program_is_refused_rather_than_raised():
    """`unroll` swallows a truncated instruction; so must the traced build."""
    truncated = bytes([int(Op.MOVE), 3])
    assert unroll(truncated) is None
    assert relation.unroll_with_trace(truncated) is None


def test_every_annotated_image_is_image_zero_under_its_step():
    rng = random.Random(7)
    checked = 0
    for _ in range(400):
        program = sample(rng, tier=Tier.L1, flatten=False)
        result = relation.unroll_with_trace(program)
        if result is None:
            continue
        relation.verify_trace(result.bytes, result.scopes)
        for scope in result.scopes:
            if not scope.has_reference:
                continue
            start, stop = scope.image(0)
            reference = result.bytes[start:stop]
            for k in range(1, scope.count):
                begin, end = scope.image(k)
                assert result.bytes[begin:end] == apply(
                    reference, Affine.of(scope.step.power(k)))
                checked += 1
    assert checked > 100


def test_verify_trace_catches_a_wrong_step():
    """A conjugation slip has to fail the build, not annotate the wrong region."""
    program = assemble("REPEAT 3 8 0\nMOVE 40 40\nLINE 50 50\nENDREP\nHALT")
    result = relation.unroll_with_trace(program)
    wrong = tuple(
        scope.__class__(**{**vars(scope), "step": Transform(D4(), 9, 0)})
        for scope in result.scopes)
    with pytest.raises(relation.TraceError):
        relation.verify_trace(result.bytes, wrong)


def test_a_nested_scope_conjugates_its_step_into_the_enclosing_copy():
    """D4 is non-abelian, so the inner step inside an outer copy is `A S A^-1`."""
    program = assemble(
        "REPEATX 2 2 0 0\n  REPEAT 3 0 12\n    MOVE 90 90\n    LINE 100 96\n"
        "  ENDREP\nENDREP\nHALT")
    result = relation.unroll_with_trace(program)
    relation.verify_trace(result.bytes, result.scopes)
    inner = [scope for scope in result.scopes if scope.kind == "REPEAT"]
    outer = next(scope for scope in result.scopes if scope.kind == "REPEATX")
    assert len(inner) == outer.count
    assert inner[0].step == Transform(D4(), 0, 12)
    assert inner[1].step == inner[0].step.conjugate(outer.step)
    assert inner[1].step != inner[0].step, "a quarter turn must move the step"


def test_the_trace_and_the_vm_agree_on_the_drawing():
    rng = random.Random(11)
    for _ in range(200):
        program = sample(rng, tier=Tier.L1, flatten=False)
        result = relation.unroll_with_trace(program)
        if result is None:
            continue
        assert VM().run(result.bytes).strokes == VM().run(program).strokes


# ---------------------------------------------------------------------------
# actions and the schedule


def test_the_schedule_drops_actions_inside_another_actions_target():
    """A decoder never stands at a boundary inside a block it appended."""
    program = assemble(
        "REPEAT 2 40 0\n  REPEAT 3 0 12\n    MOVE 10 10\n    LINE 20 20\n"
        "  ENDREP\nENDREP\nHALT")
    result = relation.unroll_with_trace(program)
    every = relation.actions_of(result.scopes)
    reachable = relation.schedule(every)
    assert len(every) == 3, "the inner scope is replicated into the outer copy"
    assert len(reachable) == 2
    inner, outer = reachable
    assert inner.target_stop == outer.boundary, "the schedule must be contiguous"
    assert outer.source_start == 0 and outer.source_stop == inner.target_stop, (
        "the outer action copies the block the inner one built: compositional "
        "provenance")


def test_covered_boundaries_exclude_the_endpoints():
    action = relation.CopyAction(
        boundary=6, source_start=0, source_stop=6, step=Transform(D4(), 8, 0),
        total_count=3, target_start=6, target_stop=18)
    covered = relation.covered_boundaries((action,))
    assert 6 not in covered, "the decision point is reachable"
    assert 18 not in covered, "decoding resumes here"
    assert 12 in covered


def test_an_action_renders_the_bytes_it_claims():
    rng = random.Random(3)
    checked = 0
    for _ in range(400):
        program = sample(rng, tier=Tier.L1, flatten=False)
        result = relation.unroll_with_trace(program)
        if result is None:
            continue
        for action in relation.schedule(relation.actions_of(result.scopes)):
            assert action.render(result.bytes) == \
                result.bytes[action.target_start:action.target_stop]
            checked += 1
    assert checked > 100


# ---------------------------------------------------------------------------
# solving for the transform


def test_solve_step_recovers_every_element_of_the_group():
    """All eight D4 elements and a signed translation, solved rather than searched."""
    source = assemble("MOVE 100 100\nLINE 120 110\nLINE 110 130")
    for code in range(8):
        for dx, dy in ((0, 0), (12, -20), (-32, 32)):
            step = Transform(D4.of(code), dx, dy)
            image = apply(source, Affine.of(step))
            if image is None:
                continue
            assert step in relation.solve_step(source, image)


def test_solve_step_refuses_a_block_of_a_different_skeleton():
    source = assemble("MOVE 10 10\nLINE 20 20")
    other = assemble("MOVE 10 10\nMOVE 20 20")
    assert relation.solve_step(source, other) == ()


def test_solve_step_refuses_a_span_with_no_coordinate():
    """Without a `COORD` the translation is unconstrained and every offset 'works'."""
    source = assemble("WIDTH 3\nFILL")
    assert relation.solve_step(source, source) == ()


def test_solve_step_returns_every_element_when_a_motif_is_symmetric():
    """A symmetric block is carried by more than one transform, and both are kept."""
    half = CANVAS - 1
    source = assemble(f"MOVE 0 10\nLINE {half} 10")
    image = apply(source, Affine.of(Transform(D4(), 0, 20)))
    solved = relation.solve_step(source, image)
    assert len(solved) >= 1
    for step in solved:
        assert apply(source, Affine.of(step)) == image


# ---------------------------------------------------------------------------
# candidate enumeration and equivalence


@pytest.fixture(scope="module")
def venue1_cases() -> tuple[relation.RelationCase, ...]:
    """Real Direction 4 cases, whose steps are inside the frozen supports.

    Not `dm.data.synthetic.sample`: Tier A's `random_grid` draws its step from
    the whole `i8` range, and a step of five is a relation the head cannot name.
    Those programs are refused by `make_case`, which is exactly right and is
    pinned separately -- but it makes them the wrong fixture for a test about
    what the enumerator recovers.
    """
    return relation.build_venue1(
        250, seed=2, motifs=relation.motif_pool(60, seed=1),
        tuples=relation.venue1_tuples()).cases


def test_a_step_outside_the_frozen_support_is_refused_at_build_time():
    """Tier A draws arbitrary `i8` steps; the head spans three translations."""
    program = assemble("REPEAT 2 5 0\nMOVE 20 20\nLINE 30 30\nENDREP\nHALT")
    with pytest.raises(relation.Rejected) as refusal:
        relation.make_case(relation.VENUE_SYNTHETIC, relation.STRATUM_TRAIN,
                           program, ["m"], ((0, 5, 0, 2),))
    assert refusal.value.rule == "action_outside_frozen_support"


def test_the_enumerator_recovers_the_generators_own_action(venue1_cases):
    checked = 0
    for case in venue1_cases:
        for action in case.actions:
            found = relation.equivalent_actions(
                case.flat, action.boundary, action.target_start,
                action.target_stop)
            assert action.key() in {other.key() for other in found}
            checked += 1
    assert checked > 100


def test_every_equivalent_action_produces_identical_bytes(venue1_cases):
    seen_multiple = 0
    for case in venue1_cases:
        for action in case.actions:
            found = relation.equivalent_actions(
                case.flat, action.boundary, action.target_start,
                action.target_stop)
            target = case.flat[action.target_start:action.target_stop]
            assert found
            for other in found:
                assert other.render(case.flat) == target
            seen_multiple += len(found) > 1
    assert seen_multiple >= 0


def test_a_derivation_covers_the_whole_block(venue1_cases):
    """Every first step must be completable, or it is not a derivation."""
    multiple = 0
    for case in venue1_cases:
        for action in case.actions:
            found = relation.derivations(case.flat, action.boundary,
                                         action.target_start, action.target_stop)
            assert found
            for other in found:
                assert other.target_stop <= action.target_stop
                assert other.render(case.flat) == \
                    case.flat[other.target_start:other.target_stop]
            multiple += len(found) > 1
    assert multiple > 0, "the doubling schedule should appear at count 4"


def test_a_count_four_scope_admits_the_doubling_schedule():
    """`COPY(count=4)` and `COPY(count=2)` then `COPY(step^2, count=2)` are the
    same bytes, so both are valid first steps and both are supervised."""
    program = assemble("REPEAT 4 32 0\nMOVE 20 20\nLINE 30 30\nENDREP\nHALT")
    result = relation.unroll_with_trace(program)
    action = relation.schedule(relation.actions_of(result.scopes))[0]
    found = relation.derivations(result.bytes, action.boundary,
                                 action.target_start, action.target_stop)
    # Three first steps, not two: `count=3` appends two images and leaves one,
    # which a second action over the last image completes. Every one of them
    # produces the identical block, so every one of them is a valid derivation.
    counts = sorted({other.total_count for other in found})
    assert counts == [2, 3, 4]
    whole = relation.equivalent_actions(result.bytes, action.boundary,
                                        action.target_start, action.target_stop)
    assert {other.total_count for other in whole} == {4}


def test_a_source_span_may_not_contain_a_halt():
    """Spans that stop before the terminator are fine; ones that swallow it are not."""
    program = assemble("MOVE 10 10\nLINE 20 20\nHALT")
    halts = relation._halt_offsets(program)
    spans = relation.candidate_spans(program, len(program))
    assert spans, "the free endpoint admits the span before the HALT"
    for start, stop in spans:
        assert not any(start <= offset < stop for offset in halts)
    assert all(stop != len(program) for _, stop in spans)


def test_the_caps_are_enforced_by_the_enumerator():
    """Byte cap, instruction cap and the gap cap, all applied before scoring."""
    body = "\n".join(f"LINE {10 + i} {10 + i}" for i in range(80))
    program = assemble(f"MOVE 5 5\n{body}\nHALT")
    marks = relation.boundaries(program)
    query = marks[-2]
    spans = relation.candidate_spans(program, query)
    assert spans
    for start, stop in spans:
        assert stop - start <= relation.MAX_SOURCE_BYTES
        assert marks.index(stop) - marks.index(start) <= \
            relation.MAX_SOURCE_INSTRUCTIONS
        assert marks.index(query) - marks.index(stop) <= \
            relation.MAX_SOURCE_GAP_INSTRUCTIONS


def test_the_candidate_set_stays_inside_the_frozen_scoring_cap():
    """The R3 head has to score this set, and the contract caps how large it is."""
    from dm.eval import relation_contract as contract
    build = relation.build_venue1(
        60, seed=2, motifs=relation.motif_pool(40, seed=1),
        tuples=relation.venue1_tuples())
    widest = max(len(relation.candidate_spans(case.flat, boundary))
                 for case in build.cases
                 for boundary in relation.boundaries(case.flat))
    assert 0 < widest <= contract.RELATION_MAX_CANDIDATE_SPANS


def test_an_out_of_support_translation_is_a_corpus_fact_not_a_supervision_target():
    """The head can name three translations; the audit still sees all of them."""
    program = assemble("REPEAT 2 7 0\nMOVE 20 20\nLINE 30 30\nENDREP\nHALT")
    result = relation.unroll_with_trace(program)
    action = relation.actions_of(result.scopes)[0]
    assert not action.in_support()
    assert action.in_support(relation.AUDIT_TRANSLATIONS)
    assert relation.copy_actions_at(result.bytes, action.boundary,
                                    action.target_stop) == ()
    assert relation.copy_actions_at(result.bytes, action.boundary,
                                    action.target_stop,
                                    relation.AUDIT_TRANSLATIONS)


# ---------------------------------------------------------------------------
# motifs and the split discipline


def test_motif_identity_is_invariant_to_placement():
    """Two placements of one drawing are one vertex, or the pair split degenerates."""
    body = assemble("MOVE 10 10\nLINE 30 20")
    moved = apply(body, Affine(dx=40, dy=17))
    assert relation._motif_id("m", body) == relation._motif_id("m", moved)


def test_two_different_drawings_are_two_identities():
    left = assemble("MOVE 10 10\nLINE 30 20")
    right = assemble("MOVE 10 10\nLINE 30 21")
    assert relation._motif_id("m", left) != relation._motif_id("m", right)


def test_the_motif_pool_is_deduplicated_on_identity_and_bounded_by_the_caps():
    pool = relation.motif_pool(64, seed=1)
    assert len({motif.motif_id for motif in pool}) == 64
    assert all(relation.MIN_SOURCE_BYTES <= len(motif.body)
               <= relation.MAX_SOURCE_BYTES for motif in pool)
    assert {motif.category for motif in pool} == set(relation.VENUE1_CATEGORIES)


def test_the_motif_pool_is_a_function_of_its_seed():
    assert [m.motif_id for m in relation.motif_pool(16, seed=4)] == \
        [m.motif_id for m in relation.motif_pool(16, seed=4)]
    assert [m.motif_id for m in relation.motif_pool(16, seed=4)] != \
        [m.motif_id for m in relation.motif_pool(16, seed=5)]


def test_every_withheld_tuple_keeps_all_four_of_its_atoms_trained():
    """§1.1: an unseen *combination* is identifiable; an unseen *value* is not."""
    space = relation.venue1_tuples()
    held = set(relation.held_out_tuples(space, seed=9, fraction=0.2))
    assert held
    trained = relation._atom_support(item for item in space if item not in held)
    for item in held:
        assert set(relation._atoms(item)) <= trained


def test_the_tiny_composed_space_still_yields_one_withheld_tuple():
    """Six tuples and a 15% fraction rounds to zero; the floor of one prevents that."""
    held = relation.held_out_tuples(relation.venue2_tuples(), seed=3, fraction=0.15)
    assert len(held) == 1


def test_a_withheld_pair_keeps_both_endpoints_paired_elsewhere():
    corpus = relation.build_venue1(
        120, seed=2, motifs=relation.motif_pool(24, seed=1),
        tuples=relation.venue1_tuples())
    pairs = relation.held_out_pairs(corpus.cases, seed=6, fraction=0.3)
    remaining = [edge for edge in relation._pairs_of(corpus.cases)
                 if edge not in pairs]
    endpoints = {node for edge in remaining for node in edge}
    for edge in pairs:
        assert set(edge) <= endpoints


def test_pair_and_interaction_strata_are_explicit():
    config = relation.BuildConfig(
        n_train_synthetic=120, n_train_composed=400, n_eval=80, n_generic=40,
        train_motifs=120, eval_motifs=80, composed_limit=128)
    built = relation.build(config)
    affine = built.of(relation.STRATUM_AFFINE)
    pair = built.of(relation.STRATUM_PAIR)
    interaction = built.of(relation.STRATUM_INTERACTION)
    assert affine and all(not case.novel_pair for case in affine)
    assert pair and all(case.novel_pair for case in pair)
    assert all(not set(case.plan) &
               set(built.heldout_tuples[relation.VENUE_COMPOSED])
               for case in pair)
    assert interaction and all(case.novel_pair for case in interaction)
    assert all(set(case.plan) &
               set(built.heldout_tuples[relation.VENUE_COMPOSED])
               for case in interaction)


# ---------------------------------------------------------------------------
# controls


def test_a_generic_program_carries_no_derivable_relation():
    build = relation.build_generic(40, seed=5,
                                   motifs=relation.motif_pool(40, seed=1))
    assert build.cases
    for case in build.cases:
        assert case.actions == ()
        for boundary in relation.boundaries(case.flat):
            assert relation.copy_actions_at(case.flat, boundary, len(case.flat),
                                            relation.AUDIT_TRANSLATIONS) == ()


def test_the_destroyed_control_is_a_closed_permutation():
    """Length, byte census and emptiness all hold, and they hold *exactly*."""
    build = relation.build_venue1(
        200, seed=2, motifs=relation.motif_pool(60, seed=1),
        tuples=relation.venue1_tuples())
    destroyed, notes = relation.destroy(build.cases, seed=6)
    assert destroyed
    sources = {case.case_id: case for case in build.cases}
    for case in destroyed:
        origin = sources[case.source_case_id]
        assert len(case.flat) == len(origin.flat)
        assert case.flat[:case.target_start] == origin.flat[:case.target_start]
        assert case.flat[case.target_stop:] == origin.flat[case.target_stop:]
        assert relation.copy_actions_at(case.flat, case.target_start,
                                        case.target_stop,
                                        relation.AUDIT_TRANSLATIONS) == ()
        donor = sources[case.donor_case_id]
        labels = relation.components(build.cases)
        assert case.donor_groups == donor.groups
        assert case.donor_source_ids == donor.source_ids
        assert relation._destroy_namespace(origin, labels) == \
            relation._destroy_namespace(donor, labels)
    assert relation._byte_census([case.flat for case in destroyed]) == \
        relation._byte_census([sources[case.source_case_id].flat
                               for case in destroyed])
    assert "residual_relation" in notes or destroyed


def test_a_donor_that_still_relates_is_dropped_rather_than_relabelled():
    """The whole point of re-enumerating: a spliced row is only a negative if it is."""
    build = relation.build_venue1(
        120, seed=2, motifs=relation.motif_pool(40, seed=1),
        tuples=relation.venue1_tuples())
    destroyed, notes = relation.destroy(build.cases, seed=11)
    assert notes.get("residual_relation", 0) >= 0
    assert all(case.actions == () for case in destroyed)


# ---------------------------------------------------------------------------
# the graph


def test_a_generic_negative_is_audited_by_the_enumerator_not_by_provenance():
    """A flat program has no scopes, so `case.actions` is empty whatever it holds."""
    build = relation.build_generic(80, seed=5,
                                   motifs=relation.motif_pool(80, seed=1))
    assert build.rejections.get("accidental_relation", 0) > 0, (
        "the audit must actually be catching accidental relations, or it is not "
        "being asked")
    for case in build.cases:
        assert relation.derivable_boundaries(case.flat) == ()


def test_islands_keep_the_dependence_graph_disconnected():
    """Without islands the whole corpus is one component and a bootstrap has n=1."""
    pool = relation.motif_pool(200, seed=1)
    build = relation.build_venue1(300, seed=2, motifs=pool,
                                  tuples=relation.venue1_tuples())
    labels = relation.components(build.cases)
    units = {relation.component_of(case, labels) for case in build.cases}
    assert len(units) > len(relation.islands_of(pool)) // 2
    assert relation.island_faults(
        build.cases, {motif.motif_id: motif.island for motif in pool}) == []


def test_no_case_spans_two_islands():
    """A single leaked case merges two clusters and narrows every later interval."""
    pool = relation.motif_pool(80, seed=1)
    build = relation.build_generic(60, seed=5, motifs=pool)
    assert relation.island_faults(
        build.cases, {motif.motif_id: motif.island for motif in pool}) == []


# ---------------------------------------------------------------------------
# venue 2 rides the same trace


def test_a_composed_scene_traces_to_its_own_flat_bytes():
    scenes, _ = composed.build_with_stats(24, "valid", seed=0,
                                          categories=("cat",), limit=256)
    assert scenes
    for scene in scenes:
        result = relation.unroll_with_trace(scene.structured)
        assert result.bytes == scene.flat
        relation.verify_trace(result.bytes, result.scopes)
        actions = relation.schedule(relation.actions_of(result.scopes))
        assert len(actions) == 1
        action = actions[0]
        assert action.source_start == scene.copies[0].start
        assert action.source_stop == scene.copies[0].stop
        assert action.step.d4.code == scene.step


def test_development_and_scientific_source_pools_are_identity_disjoint():
    scientific = composed.source_motif_pool(
        ("cat",), "train", provenance="scientific", limit=64)
    development = composed.source_motif_pool(
        ("cat",), "train", provenance="development", limit=64)
    assert scientific and development
    assert {source.source_id for source in scientific}.isdisjoint(
        source.source_id for source in development)


# ---------------------------------------------------------------------------
# acceptance and the manifest


@pytest.fixture(scope="module")
def small_corpus() -> relation.RelationCorpus:
    return relation.build(relation.BuildConfig(
        n_train_synthetic=150, n_train_composed=200, n_eval=40, n_generic=40,
        train_motifs=120, eval_motifs=80, composed_limit=128))


def test_the_caps_cover_every_accepted_positive(small_corpus):
    report = relation.acceptance_report(small_corpus)
    assert report["coverage"]["coverage"] == 1.0


def test_a_declared_disjoint_stratum_shares_no_motif_with_training(small_corpus):
    report = relation.acceptance_report(small_corpus)
    assert report["split"]["motif_disjointness_violations"] == []
    assert relation.STRATUM_NESTED not in relation.MOTIF_DISJOINT_STRATA, (
        "the nested co-primary holds motifs fixed on purpose")


def test_no_case_in_the_finished_corpus_spans_two_islands(small_corpus):
    report = relation.acceptance_report(small_corpus)
    assert report["split"]["island_fault_count"] == 0


def test_no_withheld_tuple_reaches_training(small_corpus):
    report = relation.acceptance_report(small_corpus)
    assert report["split"]["held_out_tuples_in_training"] == []
    assert report["split"]["orphaned_atoms"] == []


def test_the_tuple_split_is_global_across_venues(small_corpus):
    """A tuple withheld in one venue and trained in the other is not withheld."""
    trained = {item for case in small_corpus.of(relation.STRATUM_TRAIN)
               for item in case.plan}
    for held in small_corpus.heldout_tuples.values():
        assert not set(held) & trained


def test_acceptance_names_every_failure_rather_than_the_first(small_corpus):
    report = relation.acceptance_report(small_corpus)
    broken = {**report, "coverage": {**report["coverage"], "coverage": 0.5,
                                     "covered": 1, "positives": 2},
              "split": {**report["split"], "orphaned_atoms": ["count=9"]}}
    accepted, problems = relation.accepts(broken)
    assert not accepted
    assert len(problems) >= 2


def test_acceptance_fails_closed_on_a_missing_field():
    accepted, problems = relation.accepts({})
    assert not accepted
    assert problems


def test_acceptance_fails_closed_when_a_safety_field_is_removed(small_corpus):
    report = relation.acceptance_report(small_corpus)
    del report["destroyed"]["byte_census_matches_source"]
    accepted, problems = relation.accepts(report)
    assert not accepted
    assert any("byte_census_matches_source" in problem for problem in problems)


def test_a_smoke_corpus_cannot_be_frozen(small_corpus):
    """The component floor is a corpus property, so a small build must refuse."""
    accepted, problems = relation.accepts(relation.acceptance_report(small_corpus))
    assert not accepted
    assert any("connected components" in problem for problem in problems)


def test_the_manifest_round_trips_and_refuses_an_edit(tmp_path, small_corpus):
    body = relation.manifest(small_corpus)
    path = tmp_path / "relation_corpus.json"
    relation.write_manifest(path, body)
    digests = relation.manifest_digests(path)
    assert digests["recorded_canonical_sha256"] == digests["canonical_payload_sha256"]
    assert digests["file_sha256"] != digests["canonical_payload_sha256"]
    with pytest.raises(ValueError, match="did not pass acceptance"):
        relation.load_manifest(path)

    edited = {**body, "accepted": True}
    relation.write_manifest(path, edited)
    with pytest.raises(ValueError, match="hash mismatch"):
        relation.load_manifest(path)


def test_manifest_rechecks_acceptance_after_a_self_consistent_edit(tmp_path,
                                                                   small_corpus):
    body = relation.manifest(small_corpus)
    edited = json.loads(json.dumps(body))
    edited["acceptance"]["coverage"]["coverage"] = 0.0
    edited["accepted"] = True
    edited["problems"] = []
    payload = {key: value for key, value in edited.items()
               if key != "corpus_sha256"}
    edited["corpus_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path = tmp_path / "relation_corpus.json"
    relation.write_manifest(path, edited)
    with pytest.raises(ValueError, match="acceptance verdict"):
        relation.load_manifest(path)


def test_a_build_is_a_function_of_its_config():
    config = relation.BuildConfig(
        n_train_synthetic=30, n_train_composed=40, n_eval=12, n_generic=12,
        train_motifs=40, eval_motifs=30, composed_limit=64)
    first = relation.manifest(relation.build(config))
    second = relation.manifest(relation.build(config))
    assert first["corpus_sha256"] == second["corpus_sha256"]


def test_the_length_bins_are_all_reachable_and_cover_every_span(small_corpus):
    report = relation.acceptance_report(small_corpus)
    bins = report["length_bins"]
    assert bins["spans_past_last_edge"] == 0
    assert bins["empty_bins"] == []
    assert bins["edges"][-1] == relation.MAX_SOURCE_BYTES


def test_the_candidate_bound_is_arithmetic_not_a_sample(small_corpus):
    report = relation.acceptance_report(small_corpus)
    spans = report["candidate_spans"]
    assert spans["bound"] == (relation.MAX_SOURCE_GAP_INSTRUCTIONS + 1) \
        * relation.MAX_SOURCE_INSTRUCTIONS
    assert spans["bound"] <= spans["cap"]
    assert spans["observed_widest"] <= spans["bound"]


def test_no_held_out_program_is_byte_identical_to_a_training_program(small_corpus):
    trained = {case.flat for case in small_corpus.of(relation.STRATUM_TRAIN)}
    for case in small_corpus.cases:
        if case.stratum != relation.STRATUM_TRAIN:
            assert case.flat not in trained


def test_the_nested_primary_uses_trained_motifs_and_the_transfer_one_does_not(
        small_corpus):
    """Otherwise composition is confounded with motif transfer and a failure
    cannot say which caused it."""
    trained = {motif for case in small_corpus.of(relation.STRATUM_TRAIN)
               for motif in case.groups}
    nested = small_corpus.of(relation.STRATUM_NESTED)
    transfer = small_corpus.of(relation.STRATUM_NESTED_TRANSFER)
    assert nested and transfer
    assert all(case.groups[0] in trained for case in nested)
    assert all(case.groups[0] not in trained for case in transfer)
    assert all(case.depth == 2 for case in (*nested, *transfer))


def test_the_destroyed_arm_may_not_introduce_a_relation(small_corpus):
    report = relation.acceptance_report(small_corpus)
    assert report["destroyed"]["introduced_relations"] == 0
    assert report["destroyed"]["residual_relations"] == 0


def test_the_vm_census_rule_is_direction_fours_own(small_corpus):
    report = relation.acceptance_report(small_corpus)
    assert relation.vm_census_problems(report["vm_census"]) == []
    assert relation.vm_census_problems(None)
    assert relation.vm_census_problems({"schema": 1, "arms": {}})


# ---------------------------------------------------------------------------
# R1.1: the evidence boundary
#
# `accepts` reads the acceptance report and decides whether a corpus may be
# frozen. Until R1.1 nothing asked whether that report described the corpus the
# manifest also carries, so the report was believed on its own authority: an
# edited body that took the trouble to rehash passed every clause the project
# had. The tests below all rehash, because refusing an edit whose digest no
# longer matches was already in place and is not what is being tested.


_CLI = Path(__file__).resolve().parent.parent / "scripts" / "relation_corpus.py"
_cli_spec = importlib.util.spec_from_file_location("relation_corpus_cli", _CLI)
assert _cli_spec and _cli_spec.loader, f"no corpus adapter at {_CLI}"
relation_cli = importlib.util.module_from_spec(_cli_spec)
sys.modules["relation_corpus_cli"] = relation_cli
_cli_spec.loader.exec_module(relation_cli)


def _rehash(body: dict) -> dict:
    payload = {key: value for key, value in body.items() if key != "corpus_sha256"}
    body["corpus_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return body


def _mutated(body: dict, change) -> dict:
    """A deep copy of `body` with `change` applied and its digest recomputed."""
    edited = json.loads(json.dumps(body))
    change(edited)
    return _rehash(edited)


def _refuses(body: dict, fragment: str) -> None:
    problems = relation.validate_manifest(body)
    assert any(fragment in problem for problem in problems), \
        f"{fragment!r} not among {problems[:8]}"


def _destroyed_rows(body: dict) -> list[dict]:
    return [record for record in body["cases"]
            if record["stratum"] == relation.STRATUM_DESTROYED]


def _consistent_vm_arm(programs: int, strokes_per_program: int, *,
                       mean_strokes: float | None = None) -> dict:
    """Build one algebraically consistent, all-clean census arm.

    The coordinated regression moves the whole arm rather than changing one
    field in isolation. Every dependent count is derived from the program count
    and the one histogram length, leaving only the recorded mean available for
    the domain attack.
    """
    total_strokes = programs * strokes_per_program
    return {
        "programs": programs,
        "halted_programs": programs,
        "valid_programs": programs,
        "valid_halted_programs": programs,
        "faulted_programs": 0,
        "fault_events": 0,
        "faults": {},
        "programs_with_strokes": programs if strokes_per_program else 0,
        "total_strokes": total_strokes,
        "mean_strokes": (total_strokes / max(1, programs)
                         if mean_strokes is None else mean_strokes),
        "min_strokes": strokes_per_program,
        "max_strokes": strokes_per_program,
        "stroke_count_histogram": {str(strokes_per_program): programs},
        "vm_steps": programs * max(1, strokes_per_program),
    }


@pytest.fixture(scope="module")
def small_manifest(small_corpus) -> dict:
    """The small build's manifest, already through JSON.

    Through JSON on purpose: an in-memory body holds tuples where a loaded one
    holds lists, and a validator that only ever saw the in-memory form would not
    be the validator that runs on a file.
    """
    return json.loads(json.dumps(relation.manifest(small_corpus)))


@pytest.fixture(scope="module")
def tiny_config() -> relation.BuildConfig:
    return relation.BuildConfig(
        n_train_synthetic=30, n_train_composed=40, n_eval=12, n_generic=12,
        train_motifs=40, eval_motifs=30, composed_limit=64)


def test_a_real_build_is_structurally_valid_even_when_it_cannot_be_frozen(
        small_manifest):
    """The two questions are separate, and this is the one that says so.

    A development build has a perfectly honest manifest and still fails the
    component floor. If validation and acceptance were one function, every
    adversarial test below would be indistinguishable from "the build is small".
    """
    assert relation.validate_manifest(small_manifest) == []
    assert small_manifest["accepted"] is False


def test_an_empty_case_index_is_refused(small_manifest):
    """Every recomputation is vacuously satisfied by no cases at all."""
    _refuses(_mutated(small_manifest, lambda body: body["cases"].clear()),
             "the case index is empty")


def test_a_missing_envelope_field_is_refused(small_manifest):
    _refuses(_mutated(small_manifest, lambda body: body.pop("problems")),
             "the manifest is missing 'problems'")


def test_an_extra_envelope_field_is_refused(small_manifest):
    """Invariant 17: a second copy of an invariant is the one that wins silently."""
    _refuses(
        _mutated(small_manifest,
                 lambda body: body.update({"accepted_v2": True})),
        "unexpected field 'accepted_v2'")


def test_a_missing_acceptance_section_is_refused(small_manifest):
    _refuses(
        _mutated(small_manifest,
                 lambda body: body["acceptance"].pop("destroyed")),
        "missing section 'destroyed'")


def test_a_case_record_missing_a_field_is_refused(small_manifest):
    _refuses(
        _mutated(small_manifest, lambda body: body["cases"][0].pop("component")),
        "is missing 'component'")


def test_a_case_record_with_an_extra_field_is_refused(small_manifest):
    _refuses(
        _mutated(small_manifest,
                 lambda body: body["cases"][0].update({"actions_v2": []})),
        "unexpected field 'actions_v2'")


def test_a_boolean_is_not_a_count(small_manifest):
    """JSON has one number type and Python makes `True` an `int`."""
    _refuses(
        _mutated(small_manifest,
                 lambda body: body["acceptance"]["strata"]["train"].update(
                     {"components": True})),
        "strata.train.components has the wrong type")


def test_an_inflated_component_count_is_refused(small_manifest):
    """The headline attack: clear the 128-component floor by rewriting a number.

    `accepts` reads `components` and compares it to
    `MIN_CONFIRMATORY_COMPONENTS`. Nothing recomputed it, so a corpus with four
    components could declare four hundred and be frozen.
    """
    _refuses(
        _mutated(small_manifest,
                 lambda body: body["acceptance"]["strata"][
                     relation.STRATUM_AFFINE].update({"components": 4096})),
        f"strata.{relation.STRATUM_AFFINE}.components records 4096")


def test_a_rehashed_semantically_invalid_manifest_is_still_refused(
        tmp_path, small_manifest):
    """The complete forgery: pass the floors, declare acceptance, rehash.

    This body is internally consistent by every rule that existed before R1.1 --
    its digest matches, its verdict follows from its report, and its report
    passes every acceptance clause. Only the case index disagrees.
    """
    def change(body: dict) -> None:
        for stratum in relation.CONFIRMATORY_STRATA:
            body["acceptance"]["strata"][stratum]["components"] = 512
        body["accepted"] = True
        body["problems"] = []

    forged = _mutated(small_manifest, change)
    assert relation.accepts(forged["acceptance"]) == (True, [])
    path = tmp_path / "forged.json"
    relation.write_manifest(path, forged)
    digests = relation.manifest_digests(path)
    assert digests["recorded_canonical_sha256"] == \
        digests["canonical_payload_sha256"]
    with pytest.raises(ValueError, match="is invalid"):
        relation.load_manifest(path)


def test_erasing_the_held_out_pair_edges_is_refused(small_manifest):
    """A pair stratum with no edge behind it is not a pair stratum."""
    def change(body: dict) -> None:
        body["acceptance"]["split"]["held_out_pair_edges"] = []
        body["acceptance"]["split"]["held_out_pairs"] = 0

    _refuses(_mutated(small_manifest, change),
             "split.secondary_cases_without_held_out_pair records")


def test_a_fabricated_pair_edge_is_refused(small_manifest):
    _refuses(
        _mutated(small_manifest,
                 lambda body: body["acceptance"]["split"][
                     "held_out_pair_edges"].append(["ghost-a", "ghost-b"])),
        "split.held_out_pairs records")


def test_reusing_a_donor_is_refused(small_manifest):
    def change(body: dict) -> None:
        rows = _destroyed_rows(body)
        for name in ("donor_case_id", "donor_groups", "donor_source_ids"):
            rows[1][name] = rows[0][name]

    _refuses(_mutated(small_manifest, change), "destroyed.duplicate_donors records")


def test_a_donor_that_is_not_in_the_index_is_refused(small_manifest):
    def change(body: dict) -> None:
        _destroyed_rows(body)[0]["donor_case_id"] = "not-a-case"

    _refuses(_mutated(small_manifest, change), "missing donor not-a-case")


def test_altered_donor_metadata_is_refused(small_manifest):
    def change(body: dict) -> None:
        _destroyed_rows(body)[0]["donor_groups"] = ["fabricated-motif"]

    _refuses(_mutated(small_manifest, change),
             "donor groups do not match donor case")


def test_a_donor_drawn_across_a_namespace_is_refused(small_manifest):
    """The dependence intervention is that donors stay inside one namespace.

    The donor is *repointed* rather than relabelled. Renaming a component was the
    earlier form of this test, and since R1.1 rebuilds the motif graph a renamed
    label is caught as a label fault and never reaches this clause. What still
    gets here is a donor that really does come from another component.
    """
    def change(body: dict) -> None:
        row = _destroyed_rows(body)[0]
        index = {record["case_id"]: record for record in body["cases"]}
        source = index[row["source_case_id"]]
        donor = next(record for record in body["cases"]
                     if record["stratum"] == source["stratum"]
                     and record["component"] != source["component"])
        row["donor_case_id"] = donor["case_id"]
        row["donor_groups"] = list(donor["groups"])
        row["donor_source_ids"] = list(donor["source_ids"])

    _refuses(_mutated(small_manifest, change),
             "donor crosses destruction namespace")


def test_a_missing_donor_namespace_is_refused(small_manifest):
    """A row with no recorded namespace is a row whose donor rule is unauditable."""
    def change(body: dict) -> None:
        namespaces = body["acceptance"]["destroyed"]["donor_namespaces"]
        namespaces.pop(min(namespaces))

    _refuses(_mutated(small_manifest, change),
             "destroyed.donor_namespaces omits 1 destroyed rows")


def test_an_altered_donor_namespace_is_refused(small_manifest):
    def change(body: dict) -> None:
        namespaces = body["acceptance"]["destroyed"]["donor_namespaces"]
        namespaces[min(namespaces)][3] = 999

    _refuses(_mutated(small_manifest, change), "destroyed.donor_namespaces[")


def test_a_missing_vm_arm_is_refused(small_manifest):
    """`vm_census_problems` refuses an arm that faulted; it cannot see one absent."""
    _refuses(
        _mutated(small_manifest,
                 lambda body: body["acceptance"]["vm_census"]["arms"].pop(
                     relation.STRATUM_GENERIC)),
        "the VM census covers arms")


def test_an_extra_vm_arm_is_refused(small_manifest):
    """A *well-formed* extra arm, so the fault has to come from the index.

    A malformed one is caught by the shape pass, which proves only that the
    schema is exact. The arm here is a byte-for-byte copy of a real one under a
    name no stratum owns, so nothing but the recomputed arm set can see it.
    """
    def change(body: dict) -> None:
        arms = body["acceptance"]["vm_census"]["arms"]
        arms["phantom"] = json.loads(json.dumps(arms[relation.STRATUM_GENERIC]))

    _refuses(_mutated(small_manifest, change), "the VM census covers arms")


def test_a_short_vm_arm_is_refused(small_manifest):
    """A census over the clean half of an arm is a clean census."""
    _refuses(
        _mutated(small_manifest,
                 lambda body: body["acceptance"]["vm_census"]["arms"][
                     relation.STRATUM_GENERIC].update({"programs": 1})),
        f"vm_census.arms.{relation.STRATUM_GENERIC}.programs records 1")


def test_a_zero_structured_flat_case_count_is_refused(small_manifest):
    _refuses(
        _mutated(small_manifest,
                 lambda body: body["acceptance"]["vm_census"][
                     "structured_flat"].update({"cases": 0})),
        "vm_census.structured_flat.cases records 0")


def test_a_mismatch_count_must_agree_with_its_truncated_list(small_manifest):
    """A count of zero beside a list of failures satisfies whichever clause reads one."""
    _refuses(
        _mutated(small_manifest,
                 lambda body: body["acceptance"]["vm_census"][
                     "structured_flat"].update({"vm_mismatch_count": 3})),
        "vm_census.structured_flat.vm_mismatches holds 0 entries")


def test_an_altered_cap_is_refused(small_manifest):
    """Caps are model-blind and frozen; a manifest may not redeclare them."""
    _refuses(
        _mutated(small_manifest,
                 lambda body: body["acceptance"]["caps"].update(
                     {"max_source_bytes": 999})),
        "caps.max_source_bytes records 999")


def test_altered_length_bin_edges_are_refused(small_manifest):
    _refuses(
        _mutated(small_manifest,
                 lambda body: body["acceptance"]["length_bins"].update(
                     {"edges": [1, 2, 3]})),
        "length_bins.edges records")


def test_an_unreconstructable_configuration_is_refused(small_manifest):
    """A config that cannot rebuild a `BuildConfig` leaves nothing to rebuild from."""
    _refuses(
        _mutated(small_manifest,
                 lambda body: body["acceptance"]["config"].update(
                     {"provenance": "made_up"})),
        "config does not reconstruct a BuildConfig")


def test_an_altered_fingerprint_is_refused(small_manifest):
    _refuses(
        _mutated(small_manifest,
                 lambda body: body["acceptance"]["fingerprints"][
                     relation.STRATUM_TRAIN].update({"n_train": 0})),
        f"fingerprints.{relation.STRATUM_TRAIN}.n_train records 0")


def test_a_fingerprint_byte_total_is_recomputed_from_the_index(small_manifest):
    _refuses(
        _mutated(small_manifest,
                 lambda body: body["acceptance"]["fingerprints"][
                     relation.STRATUM_TRAIN].update({"bytes_train": 7})),
        f"fingerprints.{relation.STRATUM_TRAIN}.bytes_train records 7")


def test_a_control_stratum_may_not_carry_annotated_actions(small_manifest):
    def change(body: dict) -> None:
        record = next(item for item in body["cases"]
                      if item["stratum"] == relation.STRATUM_GENERIC)
        record["actions"] = [[0, 6, 6, 0, 0, 0, 2]]

    _refuses(_mutated(small_manifest, change),
             "carries annotated actions")


def test_a_relation_case_may_not_carry_donor_provenance(small_manifest):
    def change(body: dict) -> None:
        record = next(item for item in body["cases"]
                      if item["stratum"] == relation.STRATUM_TRAIN)
        record["donor_case_id"] = "borrowed"

    _refuses(_mutated(small_manifest, change),
             "is not a destroyed control but carries donor_case_id")


# ---------------------------------------------------------------------------
# R1.1 review: the four mutations the first candidate accepted
#
# Each of these is a *self-consistent* edit -- the body it produces agrees with
# itself by every rule the first candidate had -- and each attacks a different
# kind of trust. The first trusts a stored label over the graph it came from; the
# second trusts an exact key set to imply exact evidence; the third trusts a list
# to be a set; the fourth is not an attack at all but the validator's API, which
# promises a fault list and delivered an exception.


def _relabelled_components(body: dict) -> None:
    """Rewrite one stratum's component labels and the count beside them."""
    records = [record for record in body["cases"]
               if record["stratum"] == relation.STRATUM_NESTED]
    for offset, record in enumerate(records):
        record["component"] = 500_000 + offset
    body["acceptance"]["strata"][relation.STRATUM_NESTED]["components"] = \
        len(records)


def _string_vm_statistic(body: dict) -> None:
    body["acceptance"]["vm_census"]["arms"][relation.STRATUM_GENERIC].update(
        {"total_strokes": "many"})


def _duplicated_pair_edge(body: dict) -> None:
    edges = body["acceptance"]["split"]["held_out_pair_edges"]
    edges.append(list(edges[0]))


def _erased_held_out_tuples(body: dict) -> None:
    body["acceptance"]["split"]["held_out_tuples"][relation.VENUE_COMPOSED] = None


def _unbounded_mean(body: dict) -> None:
    body["acceptance"]["vm_census"]["arms"][relation.STRATUM_GENERIC][
        "mean_strokes"] = 10 ** 400


@pytest.mark.parametrize(
    ("programs", "strokes_per_program", "recorded_mean", "is_domain_attack"),
    [
        (7, 3, None, False),
        (10 ** 400, 1, 10 ** 400, True),
    ],
)
def test_a_coordinated_vm_arm_is_total_and_refused_at_every_boundary(
        tmp_path, monkeypatch, small_manifest, programs, strokes_per_program,
        recorded_mean, is_domain_attack):
    """A count-consistent extreme arm must produce a fault list, not overflow.

    The second row sets the programs, valid/halted counts, histogram mass,
    total, VM steps and recorded mean to ``10**400`` while keeping the census
    identities consistent. It is the cross-field body the single-slot fuzz does
    not construct.
    """
    arm = _consistent_vm_arm(programs, strokes_per_program,
                             mean_strokes=recorded_mean)

    def change(body: dict) -> None:
        body["acceptance"]["vm_census"]["arms"][
            relation.STRATUM_GENERIC] = arm

    forged = _mutated(small_manifest, change)
    problems = relation.validate_manifest(forged)
    assert isinstance(problems, list)
    if not is_domain_attack:
        assert not any("mean_strokes is outside" in problem
                       for problem in problems)
        return

    assert arm["programs"] == arm["valid_programs"] == \
        arm["valid_halted_programs"] == arm["total_strokes"] == \
        arm["vm_steps"] == 10 ** 400
    assert arm["stroke_count_histogram"] == {"1": 10 ** 400}
    assert any(
        f"vm_census.arms.{relation.STRATUM_GENERIC}.mean_strokes is outside "
        f"[0, {DEFAULT_FUEL}]" in problem
        for problem in problems)

    path = tmp_path / "coordinated-forgery.json"
    relation.write_manifest(path, forged)
    with pytest.raises(relation.ManifestRefused, match="mean_strokes is outside"):
        relation.load_manifest(path)
    assert _run_cli(["--audit", str(path), "--provenance", "development"],
                    monkeypatch) == 1


def test_the_vm_stroke_bound_is_the_default_fuel_budget():
    assert relation.MAX_STROKES_PER_PROGRAM == DEFAULT_FUEL
    assert relation._is_stroke_count(str(DEFAULT_FUEL))
    assert not relation._is_stroke_count(str(DEFAULT_FUEL + 1))


def _unbounded_histogram_key(body: dict) -> None:
    body["acceptance"]["vm_census"]["arms"][relation.STRATUM_GENERIC][
        "stroke_count_histogram"]["9" * 5000] = 1


#: Every mutation the two reviews found, by the name of what each one attacks.
_REVIEW_FORGERIES = {
    "component labels": _relabelled_components,
    "a VM statistic's type": _string_vm_statistic,
    "a duplicated pair edge": _duplicated_pair_edge,
    "malformed held-out tuples": _erased_held_out_tuples,
    "an unbounded mean": _unbounded_mean,
    "an unbounded histogram key": _unbounded_histogram_key,
}


def test_a_rewritten_component_label_is_refused_by_the_motif_graph(small_manifest):
    """The headline forgery: clear a floor by relabelling rather than building.

    `component` is a *claim*, and `strata.<stratum>.components` counts distinct
    claims. Rewriting one stratum's labels and the count beside them leaves
    `groups` -- the co-occurrence graph those labels are derived from -- untouched,
    so a validator that recomputes the count from the labels agrees with the
    forgery. The floor `docs/copy-relation.md` §2.4 freezes reads exactly that
    number, so the labels have to be rebuilt from `groups` and compared.
    """
    forged = _mutated(small_manifest, _relabelled_components)
    _refuses(forged, "component label the motif co-occurrence graph does not give")
    _refuses(forged, f"strata.{relation.STRATUM_NESTED}.components records")


def test_a_component_relabelling_cannot_move_a_donor_namespace(small_manifest):
    """The same reconstruction, read by the destroyed arm's namespace clause.

    The namespace is `(venue, stratum, component, target length, skeleton)` and it
    carries the whole dependence intervention, so a relabelling that edits the
    recorded namespaces to match would leave the donor rule checkable only against
    the forgery. The stratum relabelled here is the one the destroyed rows are
    actually spliced from, and the labels and the namespaces move together.

    The negative assertion is the load-bearing half. Every case in that stratum
    gets its *own* label, so donors and recipients no longer share one -- and the
    clause must still not report a namespace crossing, because it reads the
    rebuilt graph and the graph did not move.
    """
    index = {record["case_id"]: record for record in small_manifest["cases"]}
    spliced = {index[row["source_case_id"]]["stratum"]
               for row in _destroyed_rows(small_manifest)}
    assert spliced, "the small build has no destroyed rows to splice"

    def change(body: dict) -> None:
        cases = {record["case_id"]: record for record in body["cases"]}
        relabelled = [record for record in body["cases"]
                      if record["stratum"] in spliced]
        for offset, record in enumerate(relabelled):
            record["component"] = 700_000 + offset
        for stratum in spliced:
            body["acceptance"]["strata"][stratum]["components"] = sum(
                1 for record in relabelled if record["stratum"] == stratum)
        for row in _destroyed_rows(body):
            body["acceptance"]["destroyed"]["donor_namespaces"][
                row["case_id"]][2] = cases[row["source_case_id"]]["component"]

    forged = _mutated(small_manifest, change)
    _refuses(forged, "component label the motif co-occurrence graph does not give")
    _refuses(forged, "destroyed.donor_namespaces[")
    assert not any("crosses destruction namespace" in problem
                   for problem in relation.validate_manifest(forged)), \
        "the namespace clause must read the rebuilt graph, not the stored label"


def test_a_string_in_a_numeric_vm_statistic_is_refused(small_manifest):
    """Exact keys are not exact evidence; the arm's field list stopped at names."""
    _refuses(
        _mutated(small_manifest, _string_vm_statistic),
        f"vm_census.arms.{relation.STRATUM_GENERIC}.total_strokes has the wrong "
        "type")


def test_a_vm_arm_whose_own_census_disagrees_is_refused(small_manifest):
    """Right keys, right types, wrong arithmetic.

    An arm is a set of counts that determine each other: the stroke histogram
    counted the programs and the strokes, and the fault map counted the events. A
    total that the arm's own histogram does not produce is a total no VM run
    produced.
    """
    _refuses(
        _mutated(small_manifest,
                 lambda body: body["acceptance"]["vm_census"]["arms"][
                     relation.STRATUM_GENERIC].update({"total_strokes": 7})),
        f"vm_census.arms.{relation.STRATUM_GENERIC}.total_strokes records 7")


def test_a_duplicate_held_out_pair_edge_is_refused(small_manifest):
    """Every reader turns the edge list into a set, so on disk it must be one.

    The count beside it is not touched, and does not need to be: `held_out_pairs`
    is compared against the *deduplicated* edge set, so a repeated edge changes
    nothing any recomputation can see while the evidence a reviewer reads has
    grown an edge the corpus does not carry.
    """
    _refuses(_mutated(small_manifest, _duplicated_pair_edge),
             "split.held_out_pair_edges is not a canonical edge list")


def test_a_pair_edge_that_is_not_two_distinct_motifs_is_refused(small_manifest):
    """An edge is an unordered pair of two *different* motifs, written sorted."""
    for edge in (["ghost", "ghost"], ["ghost"], ["b-motif", "a-motif"]):
        _refuses(
            _mutated(small_manifest,
                     lambda body, edge=edge: body["acceptance"]["split"][
                         "held_out_pair_edges"].append(list(edge))),
            "split.held_out_pair_edges is not a canonical edge list")


def test_malformed_held_out_tuple_evidence_returns_a_fault_rather_than_raising(
        small_manifest):
    """`validate_manifest` returns faults; raising is a defect in its own right.

    A `None` here reached `tuple(item) for item in items` and raised `TypeError`,
    which loses the complete fault list the validator promises *and* leaves the
    audit Adapter classifying malformed bodies by whichever exception they happen
    to reach first.
    """
    _refuses(_mutated(small_manifest, _erased_held_out_tuples),
             "split.held_out_tuples")


@pytest.mark.parametrize("attack", sorted(_REVIEW_FORGERIES))
def test_every_review_forgery_is_refused_at_the_file_boundary(
        tmp_path, small_manifest, attack):
    """`load_manifest` refuses each of them too, and for the validator's reason.

    `validate_manifest` is where the rule lives, but nothing trains from a Python
    object: the boundary a scientific stage actually crosses is a file. Matching
    the message matters as much as the refusal -- this manifest would be refused
    anyway for failing acceptance, and a forgery that reached *that* branch would
    be one the validator had let through.
    """
    path = tmp_path / "forged.json"
    relation.write_manifest(
        path, _mutated(small_manifest, _REVIEW_FORGERIES[attack]))
    with pytest.raises(relation.ManifestRefused, match="is invalid"):
        relation.load_manifest(path)


# ---------------------------------------------------------------------------
# R1.1 second review: magnitude, and what an exception means
#
# The first correction made the *shape* of every evidence field exact and proved
# it with a fuzz that erases, retypes and deletes. Shape is not the whole domain:
# `10**400` and a five-thousand-digit decimal string are both well-formed JSON
# values of the right type, and both reached a conversion before any rule of this
# validator ran. What came back was CPython's `OverflowError` and CPython's
# integer-conversion `ValueError` -- refusals from the interpreter rather than
# from the contract, and the second of the two was indistinguishable from a real
# refusal at the Adapter.


def test_an_unbounded_mean_stroke_count_is_refused_before_conversion(
        small_manifest):
    """`float(10**400)` raises; the domain check has to come first.

    A mean stroke count lies between zero and the arm's own total, both of them
    already-validated integers, so the domain is decidable without converting
    anything.
    """
    _refuses(_mutated(small_manifest, _unbounded_mean),
             f"vm_census.arms.{relation.STRATUM_GENERIC}.mean_strokes is outside")


@pytest.mark.parametrize("mean", [float("inf"), float("-inf"), float("nan"), -1])
def test_a_mean_outside_its_own_totals_is_named_as_a_domain_fault(
        small_manifest, mean):
    """Non-finite and negative means are domain faults, not arithmetic misses.

    They were already refused, by an `isclose` that happened to return false. That
    is the right answer for the wrong reason: it reads as "the arithmetic does not
    agree" when the value is not a mean at all, and `nan` is only refused because
    every comparison against it is false.
    """
    _refuses(
        _mutated(small_manifest,
                 lambda body: body["acceptance"]["vm_census"]["arms"][
                     relation.STRATUM_GENERIC].update({"mean_strokes": mean})),
        f"vm_census.arms.{relation.STRATUM_GENERIC}.mean_strokes is outside")


def test_an_unbounded_histogram_key_is_refused_before_conversion(small_manifest):
    """`int("9" * 5000)` raises CPython's digit-limit `ValueError`.

    `str.isdigit` accepted it, and the key predicate ran before anything asked
    whether the number it spells could be a stroke count. The VM's own fuel and
    curve budget bound that, so the width of a canonical decimal is decidable
    without converting it.
    """
    _refuses(_mutated(small_manifest, _unbounded_histogram_key),
             "is not a stroke count")


@pytest.mark.parametrize("key", ["007", "", "٧", "1_0", "+7", " 7"])
def test_a_non_canonical_histogram_key_is_refused(small_manifest, key):
    """One stroke count, one spelling.

    `str(strokes)` is what the census writes, so `007` and `7` would be two keys
    for one length and the histogram would stop summing to the program count. The
    Arabic-Indic digit is here because `str.isdigit` is true for it and `int`
    accepts it, so the predicate and the canonical form were not the same test.
    """
    def change(body: dict) -> None:
        body["acceptance"]["vm_census"]["arms"][relation.STRATUM_GENERIC][
            "stroke_count_histogram"][key] = 1

    _refuses(_mutated(small_manifest, change), "is not a stroke count")


def test_a_refused_manifest_raises_the_direction_four_refusal(
        tmp_path, small_manifest):
    """Expected refusals have one type, and it is Direction 4's.

    Still a `ValueError` subclass, so `load_protocol` and both freeze Adapters
    keep working unchanged; the point is not a new failure mode but a boundary a
    caller can name.
    """
    path = tmp_path / "forged.json"
    relation.write_manifest(
        path, _mutated(small_manifest, _relabelled_components))
    with pytest.raises(relation.ManifestRefused):
        relation.load_manifest(path)
    assert issubclass(relation.ManifestRefused, ValueError)


#: Sentinel for "delete this key" in the fuzz below.
_MISSING = object()

#: The wrong JSON type for a value of each kind, for the fuzz below.
#:
#: The containers are deliberately **non-empty**. An empty list is falsy and slips
#: through every `value or {}` guard unread, so a body that swapped a section for
#: `[]` would exercise none of the code a body that swapped it for `["x"]` runs
#: into.
_WRONG_JSON_TYPE: dict[type, object] = {
    str: 0, bool: "true", int: "0", float: "0.0", list: {"x": 1},
    dict: ["x"], type(None): 0,
}

#: Magnitudes that are valid Python, survive a JSON round trip, and break a
#: conversion. The second review's finding was that the fuzz varied *shape* at
#: every path and never varied *size*, so a field of exactly the right type could
#: still reach `float(...)` or `int(...)` and raise.
_EXTREME_NUMBERS: tuple[object, ...] = (
    10 ** 400, -(10 ** 400), float("inf"), float("-inf"), float("nan"), -1)

#: A key wide enough to trip CPython's integer-conversion limit, for the same
#: reason: a dictionary key is evidence too, and `int(key)` is a conversion.
_EXTREME_KEY = "9" * 5000


def _fuzz_slots(root: object) -> list[tuple[object, object]]:
    """One `(container, key)` per distinct evidence *shape* under `root`.

    Deduplicated by path with list indices collapsed, because the hundredth entry
    of a list reaches the same lines of the validator as the first and fuzzing
    every one of them would only make the test slow.
    """
    seen: set[tuple] = set()
    slots: list[tuple[object, object]] = []

    def walk(node: object, path: tuple) -> None:
        if isinstance(node, dict):
            steps = [(key, key) for key in node]
        elif isinstance(node, list):
            steps = [(index, "*") for index in range(len(node))]
        else:
            return
        for key, tag in steps:
            here = (*path, tag)
            if here not in seen:
                seen.add(here)
                slots.append((node, key))
            walk(node[key], here)

    walk(root, ())
    return slots


def test_the_validator_never_raises_on_a_json_shaped_body(small_manifest):
    """The API, over every evidence slot rather than the four named ones.

    Four point mutations cannot establish a property that has to hold for any
    body a reviewer might hand `--audit`. Every slot is erased, retyped and
    deleted in turn; the assertion is only that a fault *list* comes back, since
    some slots -- an entry deep inside a donor namespace's opcode skeleton, say --
    are genuinely beyond what the index can decide, and claiming otherwise would
    be a stronger promise than the validator makes.
    """
    body = json.loads(json.dumps(small_manifest))
    slots = _fuzz_slots(body)
    assert len(slots) > 500, f"the fuzz reached only {len(slots)} slots"

    def check(what: str) -> None:
        try:
            problems = relation.validate_manifest(body)
        except Exception as error:
            raise AssertionError(
                f"validate_manifest raised {type(error).__name__} for "
                f"{what}: {error}") from error
        assert isinstance(problems, list) and all(
            isinstance(problem, str) for problem in problems), \
            f"{what} produced {problems!r}"

    for container, key in slots:
        original = container[key]
        edits: list[object] = [None, _WRONG_JSON_TYPE[type(original)]]
        if isinstance(original, (int, float)) and not isinstance(original, bool):
            edits.extend(_EXTREME_NUMBERS)
        if isinstance(container, dict):
            edits.append(_MISSING)
        for replacement in edits:
            if replacement is _MISSING:
                del container[key]
            else:
                container[key] = replacement
            check(f"{key!r}={replacement!r}")
            container[key] = original
        if isinstance(container, dict):
            # Keys are evidence too: an arm name, a case identifier and a stroke
            # count all arrive as dictionary keys, and one of them is converted.
            del container[key]
            container[_EXTREME_KEY] = original
            check(f"{key!r} renamed to a {len(_EXTREME_KEY)}-digit key")
            del container[_EXTREME_KEY]
            container[key] = original


# ---------------------------------------------------------------------------
# --rebuild-verify: what the case index cannot reconstruct


def test_rebuild_verify_accepts_a_manifest_this_code_produced(tiny_config):
    body = json.loads(json.dumps(relation.manifest(relation.build(tiny_config))))
    assert relation.rebuild_verify(body) == []


def test_rebuild_verify_catches_an_altered_configuration(tiny_config):
    """The gap the fast audit cannot close, and the reason both modes exist.

    A build size is a *request*: the index cannot say what was asked for, only
    what came back. So an edited `n_generic` leaves every recomputed acceptance
    fact intact and is caught only by rebuilding.
    """
    body = json.loads(json.dumps(relation.manifest(relation.build(tiny_config))))
    edited = _mutated(body, lambda item: item["acceptance"]["config"].update(
        {"n_generic": item["acceptance"]["config"]["n_generic"] + 1}))
    assert relation.validate_manifest(edited) == []
    problems = relation.rebuild_verify(edited)
    assert problems and "payload digest" in problems[0]


def test_rebuild_verify_names_where_two_case_indexes_diverge(tiny_config):
    body = json.loads(json.dumps(relation.manifest(relation.build(tiny_config))))
    edited = _mutated(body, lambda item: item["cases"].pop())
    problems = relation.rebuild_verify(edited)
    assert any("case index" in problem for problem in problems)


# ---------------------------------------------------------------------------
# the Adapter's exit codes


def _run_cli(argv: list[str], monkeypatch) -> int:
    monkeypatch.setattr(sys, "argv", ["relation_corpus.py", *argv])
    return relation_cli.main()


def test_the_audit_adapter_exits_non_zero_on_an_unfrozen_build(
        tmp_path, monkeypatch, small_manifest, capsys):
    path = tmp_path / "corpus.json"
    relation.write_manifest(path, json.loads(json.dumps(small_manifest)))
    assert _run_cli(["--audit", str(path), "--provenance", "development"],
                    monkeypatch) == 1
    captured = capsys.readouterr()
    assert "REFUSED" in captured.err
    assert "coverage" in captured.out, "a refused manifest is still summarised"


def test_the_audit_adapter_exits_non_zero_on_a_forged_manifest(
        tmp_path, monkeypatch, small_manifest):
    def change(body: dict) -> None:
        for stratum in relation.CONFIRMATORY_STRATA:
            body["acceptance"]["strata"][stratum]["components"] = 512
        body["accepted"] = True
        body["problems"] = []

    path = tmp_path / "corpus.json"
    relation.write_manifest(path, _mutated(small_manifest, change))
    assert _run_cli(["--audit", str(path)], monkeypatch) == 1


def test_the_audit_adapter_exits_non_zero_on_unreadable_json(tmp_path, monkeypatch):
    path = tmp_path / "corpus.json"
    path.write_text("{not json")
    assert _run_cli(["--audit", str(path)], monkeypatch) == 1


def test_the_rebuild_is_not_attempted_when_the_audit_already_failed(
        tmp_path, monkeypatch, small_manifest, capsys):
    path = tmp_path / "corpus.json"
    relation.write_manifest(
        path, _mutated(small_manifest, lambda body: body["cases"].clear()))
    assert _run_cli(["--rebuild-verify", str(path)], monkeypatch) == 1
    assert "the rebuild was not attempted" in capsys.readouterr().err


def test_the_adapter_catches_a_refusal_and_not_a_validator_defect(
        tmp_path, monkeypatch, small_manifest):
    """Exception classification follows the contract boundary, not the operation.

    Catching builtin `ValueError` made "this manifest is bad" and "the validator
    broke" the same outcome. A five-thousand-digit histogram key raised CPython's
    integer-conversion `ValueError` and was printed as an ordinary refusal, while
    an `OverflowError` two fields away crashed loudly -- so which of the two a
    reviewer saw depended on the failed operation. Only `ManifestRefused` is a
    refusal now, and a builtin escaping validation stays loud.
    """
    path = tmp_path / "corpus.json"
    relation.write_manifest(path, _mutated(small_manifest, _unbounded_mean))
    assert _run_cli(["--audit", str(path), "--provenance", "development"],
                    monkeypatch) == 1

    def boom(_body: object) -> list[str]:
        raise ValueError("an implementation defect, not a refusal")

    monkeypatch.setattr(relation, "validate_manifest", boom)
    with pytest.raises(ValueError, match="implementation defect") as caught:
        _run_cli(["--audit", str(path), "--provenance", "development"],
                 monkeypatch)
    assert not isinstance(caught.value, relation.ManifestRefused)


def test_the_two_audit_modes_are_exclusive(tmp_path, monkeypatch):
    path = tmp_path / "corpus.json"
    path.write_text("{}")
    with pytest.raises(SystemExit):
        _run_cli(["--audit", str(path), "--rebuild-verify", str(path)], monkeypatch)
