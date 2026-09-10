"""Direction 2's contract: model-blind twins, matched controls, exact scoring.

`docs/directions.md` §4.7 lists eight verifications that precede the C3 run.
Each one has a test here, and the builder invariants of §4.3-§4.4 have theirs
beside them, because a manifest is the only thing that travels from the
model-blind build to the scoring run.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from dm.data import composed, synthetic
from dm.data.augment import Affine, apply
from dm.eval.context import (
    ContextCase,
    cases_from_manifest,
    components,
    field_map,
    first_token_divergence,
    legal_first_bytes,
    load_manifest,
    manifest_dict,
    prefix_is_live,
    score_cases,
    score_spans,
    signature,
    summarise_by_venue,
    summarise_scores,
    validate_cases,
    write_manifest,
)
from dm.eval.context_cases import (
    CorpusStats,
    build_shape_cases,
    build_step_cases,
    census,
    recover_transform,
    rotate_cases,
    translation,
)
from dm.isa.asm import assemble
from dm.isa.codec import ByteCodec
from dm.isa.state import LanguagePolicy
from dm.models.transformer import Config, DrawingLM


class ConstantLM(torch.nn.Module):
    """Logits that ignore the prefix entirely.

    §4.7 rule 5's null model, and rule 4's as well: a preference that depends on
    the target string alone must cancel in the symmetric contrast.
    """

    def __init__(self, vocab: int, favour: dict[int, float] | None = None) -> None:
        super().__init__()
        table = torch.zeros(vocab)
        for symbol, boost in (favour or {}).items():
            table[symbol] = boost
        self.table = table

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.table.view(1, 1, -1).expand(x.shape[0], x.shape[1], -1)


def _analytic_bits(table: torch.Tensor, program: bytes, codec: ByteCodec,
                   start: int) -> float:
    """Target bits from the logit table alone, without torch's loss path."""
    values = [float(value) for value in table]
    peak = max(values)
    total = math.log(sum(math.exp(value - peak) for value in values)) + peak
    symbols = codec.with_bos(program)[1:]
    return sum((total - values[symbol]) / math.log(2)
               for symbol in symbols[start * codec.stride:])


def _reference_bits(model, program: bytes, codec: ByteCodec) -> list[float]:
    """Per-symbol bits computed straight from the model, for alignment checks."""
    seq = torch.tensor([codec.with_bos(program)], dtype=torch.long)
    logits = model(seq[:, :-1])
    nll = F.cross_entropy(logits.reshape(-1, codec.vocab_size),
                          seq[:, 1:].reshape(-1), reduction="none")
    return [float(value) / math.log(2) for value in nll]


def _corpus(n_train: int = 3000, n_val: int = 300):
    train, val = synthetic.split(n_train, n_val, seed=0, tier=1, flatten=True)
    policy = LanguagePolicy.from_programs(train + val)
    stats = CorpusStats.from_programs(train, label="synthetic:test", split="train")
    return train, val, policy, stats


def _cases(max_cases: int = 6):
    _, val, policy, stats = _corpus()
    cases, rejected = build_step_cases(val, policy, ByteCodec(), stats,
                                       max_cases=max_cases)
    return cases, rejected, val, policy, stats


def _swap_worlds(case: ContextCase) -> ContextCase:
    return replace(case, prefix_a=case.prefix_b, prefix_b=case.prefix_a,
                   target_a=case.target_b, target_b=case.target_a)


def _relabel(case: ContextCase) -> ContextCase:
    """A complete A/B relabelling: the control pair follows the worlds."""
    return replace(_swap_worlds(case),
                   control_target_a=case.control_target_b,
                   control_target_b=case.control_target_a)


def _swap_targets(case: ContextCase) -> ContextCase:
    return replace(case, target_a=case.target_b, target_b=case.target_a)


# ---------------------------------------------------------------------------
# builder contract


def test_step_twins_carry_both_matched_controls():
    cases, rejected, val, policy, _ = _cases()
    assert len(cases) == 6
    validate_cases(cases, val, policy)
    for case in cases:
        assert case.venue == "synthetic_flat_step"
        assert case.donor_kind == "self_translation"
        assert case.control_donor_index not in (case.source_index,
                                                case.donor_index)
        # The relation-bearing block abuts its continuation, and the unrelated
        # block takes the *identical* displacement.
        assert case.relevant.distance_to_target == 0
        assert case.relevant.stop == len(case.prefix_a)
        assert (case.relevant.dx, case.relevant.dy) == (case.unrelated.dx,
                                                        case.unrelated.dy)
        assert case.relevant.length == case.unrelated.length
        assert case.relevant.byte_distance == case.unrelated.byte_distance
        # The block control differs from world A in one span and nowhere else.
        assert case.block_prefix_a == case.prefix_a
        start, stop = case.unrelated.start, case.unrelated.stop
        assert case.block_prefix_a[:start] == case.block_prefix_b[:start]
        assert case.block_prefix_a[stop:] == case.block_prefix_b[stop:]
        assert case.block_prefix_a[start:stop] != case.block_prefix_b[start:stop]
        assert stop <= case.relevant.start
    assert rejected["candidate_rows"] >= len(cases)


def test_case_construction_is_deterministic_in_its_seed():
    _, val, policy, stats = _corpus()
    first, _ = build_step_cases(val, policy, ByteCodec(), stats, max_cases=5)
    second, _ = build_step_cases(val, policy, ByteCodec(), stats, max_cases=5)
    assert first == second
    other, _ = build_step_cases(val, policy, ByteCodec(), stats, max_cases=5,
                                seed=7)
    assert other != first


def test_the_counterfactual_step_comes_from_the_frozen_corpus_support():
    _, val, policy, stats = _corpus()
    cases, _ = build_step_cases(val, policy, ByteCodec(), stats, max_cases=6)
    pooled = {value for _, values in stats.step_support for value in values}
    for case in cases:
        shift = translation(case.target_a, case.target_b)
        assert shift is not None
        # Copy 3 sits at `2 d`, so the two worlds' targets differ by twice the
        # change in step -- and the counterfactual step itself has to be one the
        # generator actually produced, not an arbitrary i8.
        assert (shift[0] % 2, shift[1] % 2) == (0, 0)
        observed = (case.transform[0] // 2, case.transform[1] // 2)
        counterfactual = (observed[0] + case.relevant.dx,
                          observed[1] + case.relevant.dy)
        assert counterfactual != observed
        assert abs(counterfactual[0] or counterfactual[1]) in pooled
        assert (shift[0], shift[1]) == (2 * case.relevant.dx,
                                        2 * case.relevant.dy)


def test_the_rotation_venue_moves_the_step_from_x_to_y():
    cases, _, val, policy, stats = _cases()
    rotated, rejected = rotate_cases(cases, policy, ByteCodec(), stats)
    assert not rejected
    assert len(rotated) == len(cases)
    validate_cases(rotated, val, policy)
    turn = Affine(quarter_turns=1)
    for base, moved in zip(cases, rotated):
        assert base.relevant.dy == 0 and base.relevant.dx != 0
        assert moved.relevant.dx == 0 and moved.relevant.dy != 0
        assert "exact_d4_rotation" in moved.features.strata
        # Exact, not approximate: the rotated case *is* the rotated bytes.
        for name in ("prefix_a", "prefix_b", "target_a", "target_b",
                     "control_target_a", "control_target_b"):
            assert apply(getattr(base, name), turn) == getattr(moved, name)
        assert moved.venue == "synthetic_flat_step_d4r1"


def test_recover_transform_verifies_the_whole_affine():
    motif = assemble("MOVE 10 20\nLINE 30 40\nLINE 12 60")
    for candidate in (Affine(quarter_turns=1), Affine(mirror=True),
                      Affine(dx=5, dy=-7), Affine(dx=3, dy=4, mirror=True,
                                                  quarter_turns=3)):
        image = apply(motif, candidate)
        assert image is not None
        assert recover_transform(motif, image) == candidate
    # A target that is not an image of the source under any element is refused
    # rather than approximated.
    assert recover_transform(motif, assemble("MOVE 10 20\nLINE 30 41\nLINE 12 60")) is None


def test_shape_twins_derive_their_target_from_the_verified_transform():
    scenes = composed.build(300, "valid", seed=composed.val_seed(0),
                            orbit_sizes=(2,))
    flat = [scene.flat for scene in scenes]
    policy = LanguagePolicy.from_programs(flat)
    stats = CorpusStats.from_programs(flat, label="composed:test", split="val",
                                      step_programs=(), step_split="none")
    cases, rejected = build_shape_cases(scenes, policy, ByteCodec(), stats,
                                        max_cases=4)
    assert cases, f"the shape venue produced nothing: {dict(rejected)}"
    validate_cases(cases, flat, policy)
    for case in cases:
        assert case.venue == "composed_shape_copy2"
        assert case.donor_kind == "substituted_block"
        assert case.target_ordinal == 2
        transform = Affine(dx=case.transform[0], dy=case.transform[1],
                           mirror=bool(case.transform[2]),
                           quarter_turns=case.transform[3])
        start, stop = case.relevant.start, case.relevant.stop
        assert apply(case.prefix_a[start:stop], transform) == case.target_a
        assert apply(case.prefix_b[start:stop], transform) == case.target_b
        assert signature(case.target_a) == signature(case.target_b)


def test_distractor_spans_are_recorded_and_never_touch_the_orbit():
    for scene in composed.build(40, "valid", seed=composed.val_seed(0),
                                orbit_sizes=(2,)):
        assert len(scene.distractor_spans) == scene.distractors
        for span in scene.distractor_spans:
            assert all(span.stop <= copy.start or copy.stop <= span.start
                       for copy in scene.copies)


# ---------------------------------------------------------------------------
# manifest


def test_context_manifest_is_frozen_and_content_addressed(tmp_path):
    cases, rejected, val, policy, stats = _cases(4)
    rotated, rotation_rejected = rotate_cases(cases, policy, ByteCodec(), stats)
    every = cases + rotated
    report = census(every, rejected + rotation_rejected, stats=stats, seed=0,
                    requested=4, sources=len(val))
    assert all(entry["primary_donor_overlap"] == 0
               for entry in report["identity"]["by_venue"].values())
    assert report["identity"]["nondegenerate_controls"] == len(every)
    assert report["identity"]["nondegenerate_block_controls"] == len(every)
    assert report["venue_case_counts"] == {"synthetic_flat_step": 4,
                                           "synthetic_flat_step_d4r1": 4}
    manifest = manifest_dict(cases=every, census=report,
                             corpus={"train": "train", "val": "val"},
                             policy=policy, corpus_stats=stats.as_dict())
    path = tmp_path / "context.json"
    write_manifest(path, manifest)
    restored = load_manifest(path)
    assert cases_from_manifest(restored) == every
    assert cases_from_manifest(restored, "synthetic_flat_step") == cases
    assert restored["manifest_sha256"]

    broken = json.loads(json.dumps(manifest))
    broken["cases"][0]["control_target_b"] = broken["cases"][0][
        "control_target_a"
    ]
    body = {key: value for key, value in broken.items()
            if key != "manifest_sha256"}
    broken["manifest_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    write_manifest(path, broken)
    with pytest.raises(ValueError, match="control targets must differ"):
        load_manifest(path)


def test_schema_refuses_degenerate_and_tangled_cases():
    cases, _, _, _, _ = _cases(2)
    case = cases[0]
    with pytest.raises(ValueError, match="control targets must differ"):
        replace(case, control_target_b=case.control_target_a)
    with pytest.raises(ValueError, match="unrelated-block control worlds are equal"):
        replace(case, block_prefix_b=case.block_prefix_a)
    with pytest.raises(ValueError, match="primary targets must differ"):
        replace(case, target_b=case.target_a)
    with pytest.raises(ValueError, match="third identity"):
        replace(case, control_donor_index=case.source_index)
    with pytest.raises(ValueError, match="disagrees with the recorded donor"):
        replace(case, donor_kind="substituted_block")


def test_validation_catches_an_edit_outside_the_recorded_span():
    cases, _, val, policy, _ = _cases(2)
    case = cases[0]
    validate_cases([case], val, policy)
    # Move a byte the manifest says is shared between the two worlds.
    tampered = bytearray(case.prefix_b)
    tampered[0] = (tampered[0] + 1) % 256
    with pytest.raises(ValueError, match="differ outside the recorded span"):
        validate_cases([replace(case, prefix_b=bytes(tampered))], val, policy)
    with pytest.raises(ValueError, match="source digest"):
        validate_cases([replace(case, source_digest="drifted")], val, policy)


# ---------------------------------------------------------------------------
# scorer verification (§4.7)


def test_four_way_nlls_match_an_analytic_reference():
    cases, _, _, _, _ = _cases(2)
    codec = ByteCodec()
    model = ConstantLM(codec.vocab_size, favour={0x01: 2.0, 0x02: 1.0})
    rows = score_cases(model, cases, codec, max_len=512, batch_size=4)
    for case, row in zip(cases, rows):
        for name, target, prefix in (
            ("a_given_a", case.target_a, case.prefix_a),
            ("b_given_a", case.target_b, case.prefix_a),
            ("a_given_b", case.target_a, case.prefix_b),
            ("b_given_b", case.target_b, case.prefix_b),
        ):
            expected = _analytic_bits(model.table, prefix + target, codec,
                                      len(prefix))
            assert row["primary"]["nll_bits"][name] == pytest.approx(expected,
                                                                    abs=1e-4)


def test_a_prefix_independent_model_gives_zero_contrast():
    cases, _, _, _, _ = _cases(3)
    codec = ByteCodec()
    # The favoured bytes make one target string genuinely cheaper than the
    # other, which is exactly the unconditional preference the symmetric
    # contrast has to cancel.
    model = ConstantLM(codec.vocab_size, favour={0x01: 3.0, 0x40: 1.5})
    rows = score_cases(model, cases, codec, max_len=512, batch_size=4)
    for row in rows:
        assert row["D"]["bits"] == pytest.approx(0.0, abs=1e-6)
        assert row["D_target_control"]["bits"] == pytest.approx(0.0, abs=1e-6)
        assert row["D_block_control"]["bits"] == pytest.approx(0.0, abs=1e-6)
        assert row["D_minus_control"]["bits"] == pytest.approx(0.0, abs=1e-6)
    divergence = first_token_divergence(model, cases, codec)
    assert all(entry["kl_bits_a_to_b"] == pytest.approx(0.0, abs=1e-9)
               for entry in divergence)


def test_world_swaps_preserve_D_and_target_swaps_reverse_it():
    cases, _, _, _, _ = _cases(3)
    codec = ByteCodec()
    torch.manual_seed(0)
    model = DrawingLM(Config(vocab_size=codec.vocab_size, d_model=32,
                             n_layers=2, n_heads=2, max_len=512))
    base = score_cases(model, cases, codec, max_len=512, batch_size=4)
    swapped = score_cases(model, [_swap_worlds(case) for case in cases], codec,
                          max_len=512, batch_size=4)
    reversed_targets = score_cases(model, [_swap_targets(case) for case in cases],
                                   codec, max_len=512, batch_size=4)
    relabelled = score_cases(model, [_relabel(case) for case in cases], codec,
                             max_len=512, batch_size=4)
    for row, other, flipped, full in zip(base, swapped, reversed_targets,
                                         relabelled):
        assert other["D"]["bits"] == pytest.approx(row["D"]["bits"], abs=1e-4)
        assert flipped["D"]["bits"] == pytest.approx(-row["D"]["bits"], abs=1e-4)
        # Swapping the worlds without relabelling the control pair reverses the
        # control contrast, because the control's "matched" world is decided by
        # the prefix it is paired with -- a sign that has to move exactly where
        # the estimand says and nowhere else (§4.7 rule 2).
        assert other["D_target_control"]["bits"] == pytest.approx(
            -row["D_target_control"]["bits"], abs=1e-4)
        assert full["D_target_control"]["bits"] == pytest.approx(
            row["D_target_control"]["bits"], abs=1e-4)
        assert full["D_minus_control"]["bits"] == pytest.approx(
            row["D_minus_control"]["bits"], abs=1e-4)


def test_target_spans_align_with_an_independent_scoring_path():
    cases, _, _, _, _ = _cases(2)
    codec = ByteCodec()
    torch.manual_seed(1)
    model = DrawingLM(Config(vocab_size=codec.vocab_size, d_model=32,
                             n_layers=2, n_heads=2, max_len=512))
    model.eval()
    requests = [(case.prefix_a, case.target_a) for case in cases]
    with torch.no_grad():
        scored = score_spans(model, requests, codec, max_len=512, batch_size=2)
        for (prefix, target), row in zip(requests, scored):
            reference = _reference_bits(model, prefix + target, codec)
            lo = len(prefix) * codec.stride
            assert row["bits"] == pytest.approx(sum(reference[lo:]), abs=1e-4)
            assert row["prefix_bits"] == pytest.approx(sum(reference[:lo]),
                                                       abs=1e-4)
            assert len(row["per_byte"]) == len(target)


def test_field_contributions_partition_the_contrast():
    cases, _, _, _, _ = _cases(3)
    codec = ByteCodec()
    torch.manual_seed(2)
    model = DrawingLM(Config(vocab_size=codec.vocab_size, d_model=32,
                             n_layers=2, n_heads=2, max_len=512))
    rows = score_cases(model, cases, codec, max_len=512, batch_size=4)
    for case, row in zip(cases, rows):
        fields = {name: value for name, value in row["fields"].items()
                  if name != "first_instruction"}
        assert set(fields) == set(field_map(case.target_a))
        assert sum(fields.values()) == pytest.approx(row["D"]["bits"], abs=1e-4)
    # The venue is x-anisotropic by construction, so the y half of a target
    # cannot be where a step effect lives; the column exists to show that.
    assert all("coord_x" in row["fields"] for row in rows)


def test_bootstrap_clusters_connected_sources_and_donors():
    def row(source: int, donor: int, control: int) -> dict:
        contrast = {"bits": 1.0, "bits_per_byte": 1.0}
        return {
            "source_index": source, "donor_index": donor,
            "control_donor_index": control, "venue": "synthetic_flat_step",
            "control_targets_distinct": True, "strata": [],
            "fields": {"coord_x": 1.0}, "prefix_bytes": 40, "target_bytes": 15,
            "primary": {"nll_bits": {"a_given_a": 3.0}},
            "prefix_nll_bits": {"a": 1.0, "b": 1.0, "block_b": 1.0},
            "D": contrast, "D_target_control": contrast,
            "D_block_control": contrast, "D_minus_control": contrast,
            "D_minus_block_control": contrast,
        }

    # Rows one and two share control donor 10 and are therefore one dependency
    # component; the third is independent. Two clusters, not three sources.
    rows = [row(1, 1, 10), row(2, 2, 10), row(3, 3, 30)]
    assert len(components(rows)) == 2
    summary = summarise_scores(rows, seed=7, reps=50)
    assert summary["bootstrap_D_minus_control"]["clusters"] == 2
    assert summary["bootstrap_D_minus_control"]["cluster_unit"] == (
        "connected_source_donor_component"
    )
    assert summary["bootstrap_D_minus_control"]["largest_cluster"] == 2


def test_summaries_are_reported_per_venue_and_never_pooled():
    cases, _, _, policy, stats = _cases(3)
    rotated, _ = rotate_cases(cases, policy, ByteCodec(), stats)
    codec = ByteCodec()
    torch.manual_seed(3)
    model = DrawingLM(Config(vocab_size=codec.vocab_size, d_model=32,
                             n_layers=2, n_heads=2, max_len=512))
    rows = score_cases(model, cases + rotated, codec, max_len=512, batch_size=4)
    summaries = summarise_by_venue(rows, seed=7, reps=50)
    assert set(summaries) == {"synthetic_flat_step", "synthetic_flat_step_d4r1"}
    assert all(summary["n_cases"] == 3 for summary in summaries.values())
    assert summaries["synthetic_flat_step_d4r1"]["strata"][
        "exact_d4_rotation"] == 3


def test_scoring_refuses_a_truncated_target():
    cases, _, _, _, _ = _cases(1)
    codec = ByteCodec()
    model = ConstantLM(codec.vocab_size)
    with pytest.raises(ValueError, match="truncated at max_len"):
        score_spans(model, [(cases[0].prefix_a, cases[0].target_a)], codec,
                    max_len=8, batch_size=1)


def test_model_blind_support_is_read_off_the_policy():
    cases, _, _, policy, _ = _cases(1)
    case = cases[0]
    assert prefix_is_live(case.prefix_a, policy)
    support = legal_first_bytes(case.prefix_a, policy, ByteCodec())
    assert support == case.features.legal_first_bytes
    assert 0 < support <= len(policy.opcodes)


def test_driver_instrument_checks_fail_closed_on_a_missing_instrument():
    from scripts.context_c3 import _instrument_checks

    cases, rejected, val, _, stats = _cases(4)
    manifest = {"census": census(cases, rejected, stats=stats, seed=0,
                                 requested=4, sources=len(val))}
    protocol = {"context_c3": {
        "min_venue_yield": {"synthetic_flat_step": 4},
        "manifests": {"m.json": {"venues": ["synthetic_flat_step"]}},
    }}

    def reports(*, rate: float = 1.0,
                unit: str = "connected_source_donor_component") -> list[dict]:
        return [{
            "summary_all": {"nondegenerate_control_rate": rate},
            "venues": {"synthetic_flat_step": {
                "bootstrap_D_minus_control": {"cluster_unit": unit}}},
            "rows": [{"support": {"coordinate_marginal_distance": 0.0}}
                     for _ in cases],
        }]

    assert all(_instrument_checks(manifest, "m.json", cases, reports(),
                                  protocol).values())
    degenerate = _instrument_checks(manifest, "m.json", cases,
                                    reports(rate=0.5), protocol)
    assert not degenerate["nondegenerate_control_pairs"]
    # Resampling primary sources alone would count donor-linked rows twice.
    naive = _instrument_checks(manifest, "m.json", cases,
                               reports(unit="source_index"), protocol)
    assert not naive["donor_dependence_accounted_for"]
    short = dict(protocol)
    short["context_c3"] = dict(protocol["context_c3"],
                               min_venue_yield={"synthetic_flat_step": 99})
    assert not _instrument_checks(manifest, "m.json", cases, reports(),
                                  short)["venue_yield_met"]


@pytest.mark.parametrize("name", ["runs/context_c3_synthetic_flat_v3.json",
                                  "runs/context_c3_composed_shape_v3.json"])
def test_the_frozen_manifests_agree_with_the_committed_protocol(name):
    from pathlib import Path

    path = Path(name)
    if not path.exists():
        pytest.skip(f"{name} is a local artifact; rebuild it with "
                    "scripts/context_c3.py --build-manifest-only")
    protocol = json.loads(Path("docs/context-protocol-v1.json").read_text())
    entry = protocol["context_c3"]["manifests"][name]
    manifest = load_manifest(path)
    assert manifest["manifest_sha256"] == entry["manifest_sha256"]
    assert manifest["census"]["venues"] == entry["venues"]
    assert manifest["corpus"] == entry["corpus"]
    assert manifest["policy_digest"] == entry["policy_digest"]
    cases = cases_from_manifest(manifest)
    for venue, count in entry["venue_case_counts"].items():
        assert sum(case.venue == venue for case in cases) == count
        assert count >= protocol["context_c3"]["min_venue_yield"][venue]


def test_the_y_axis_venue_relays_the_step_without_moving_the_program():
    _, val, policy, stats = _corpus()
    base, _ = build_step_cases(val, policy, ByteCodec(), stats, max_cases=6)
    cases, rejected = build_step_cases(val, policy, ByteCodec(), stats,
                                       max_cases=6, axis="y")
    assert cases
    validate_cases(cases, val, policy)
    for case in cases:
        assert case.venue == "synthetic_flat_step_yaxis"
        assert "forced_y_axis" in case.features.strata
        # The relation is now vertical, and it is still an exact translation.
        assert case.relevant.dx == 0 and case.relevant.dy != 0
        assert case.transform[0] == 0 and case.transform[1] != 0
        shift = translation(case.target_a, case.target_b)
        assert shift == (2 * case.relevant.dx, 2 * case.relevant.dy)
        # Copy 1 and everything before it are untouched corpus bytes: only the
        # two copies the relation is about move, which is what keeps the
        # diagnostic inside the model's support.
        body = case.prefix_a[case.relevant.start:case.relevant.stop]
        assert apply(body, Affine(dy=-(case.transform[1] // 2))) is not None
    assert rejected["axis_off_canvas"] > 0, "the canvas bound should bite"
    # A forced x axis reproduces the corpus's own branch, so it is a no-op here.
    same, _ = build_step_cases(val, policy, ByteCodec(), stats, max_cases=6,
                               axis="x")
    assert [c.prefix_a for c in same] == [c.prefix_a for c in base]


def test_the_census_counts_components_inside_each_venue():
    cases, rejected, val, policy, stats = _cases(4)
    rotated, rotation_rejected = rotate_cases(cases, policy, ByteCodec(), stats)
    report = census(cases + rotated, rejected + rotation_rejected, stats=stats,
                    seed=0, requested=4, sources=len(val))
    by_venue = report["identity"]["by_venue"]
    assert set(by_venue) == {"synthetic_flat_step", "synthetic_flat_step_d4r1"}
    # The rotated venue reuses its base's sources, so pooling would report one
    # set of components where each venue in fact resamples its own four.
    for entry in by_venue.values():
        assert entry["cases"] == 4
        assert entry["components"] == 4
        assert entry["primary_donor_overlap"] == 0
    assert report["identity"]["components"] == 4
