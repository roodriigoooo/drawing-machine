"""The constructed compositional corpus: exact by construction, or worthless.

A corpus whose ceiling is *estimated* buys nothing this project could not
already get from QuickDraw. The whole value is that the generator knows the
answer, so these tests check the three things that would quietly destroy that:

- **the two forms must draw the same picture**, or the ceiling is the price of a
  different drawing;
- **each copy must be an exact image of the first**, or the corpus contains
  near-repeats -- the SVG-Icons8 failure, reproduced by our own hand;
- **the saving must be the measured difference between the forms**, not
  arithmetic that could drift away from what the emitter actually wrote.

And one test earns the provenance API on its own: the existing translational
matcher finds **nothing** on a transformed corpus, because a mirrored copy
shares no bytes with its original. A detector-driven reading would report a
recovery of NaN and look like a null result.
"""

import numpy as np
import pytest

from dm.data import composed
from dm.data.augment import Affine, apply
from dm.eval.recovery import repeat_spans
from dm.isa.asm import parse
from dm.isa.codec import CODECS
from dm.isa.spec import Op
from dm.isa.transform import D4, Transform
from dm.vm.interp import VM

VM_ = VM()
#: Small, and drawn from `valid` so the tests never touch the 100k train split.
N = 60


def corpus(**kwargs) -> list[composed.Scene]:
    return composed.build(N, "valid", seed=0, limit=600, **kwargs)


def test_the_two_forms_draw_the_same_picture():
    """The ceiling is the difference between two spellings of one drawing. If
    they draw different pictures it is the price of a different drawing."""
    for scene in corpus():
        assert VM_.run(scene.flat).valid, scene.flat.hex()
        assert VM_.run(scene.flat).strokes == VM_.run(scene.structured).strokes


def test_the_saving_is_the_measured_difference_between_the_forms():
    """Read off the bytes, not computed twice. `(n-1)·|M| - 6` is the intended
    arithmetic and the emitter is what actually decides it."""
    for scene in corpus():
        assert scene.saved == len(scene.flat) - len(scene.structured)
        assert scene.saved > 0, "a scene with nothing to fold is not a scene"


def test_every_copy_is_an_exact_image_of_the_first():
    """Not similar, not within a tolerance -- equal. A copy that had been
    rounded apart is a near-repeat, and a corpus of near-repeats is the thing
    `docs/tier-c.md` records as a failure."""
    for scene in corpus():
        first = scene.flat[scene.copies[0]]
        step = Transform(D4.of(scene.step), 0, 0)
        for k, span in enumerate(scene.copies):
            want = first if k == 0 else apply(first, Affine.of(step.power(k)))
            assert want is not None
            assert scene.flat[span] == want, f"copy {k} is not an exact image"


def test_the_copies_do_not_overlap_on_the_canvas():
    """Two copies drawn on top of each other are one shape claiming the
    redundancy of two."""
    for scene in corpus():
        boxes = [composed.bounds(scene.flat[span]) for span in scene.copies]
        assert all(b is not None for b in boxes)
        for i, a in enumerate(boxes):
            for b in boxes[i + 1:]:
                assert (a[2] < b[0] or b[2] < a[0]
                        or a[3] < b[1] or b[3] < a[1]), "copies overlap"


def test_the_flat_form_carries_no_structure_opcode():
    """The trace is what a model trains on, and it must not contain the answer.
    Handing it `REPEATX` is the L1 mistake one tier up."""
    for scene in corpus():
        assert {i.mnemonic for i in parse(scene.flat)} <= {"MOVE", "LINE", "HALT"}
        assert int(Op.REPEATX) in scene.structured


def test_the_translational_matcher_finds_the_control_and_misses_the_transform():
    """**The test that earns the provenance API.**

    `dm.eval.repeats` matches bytes, so it finds a translated copy and cannot
    find a mirrored one -- `x -> 255 - x` changes every coordinate byte. Driving
    `recovery` off the detector would therefore report *no repeats at all* on
    the corpus this project built to contain them, and a NaN would read as a
    null result rather than as a broken instrument.
    """
    control = composed.build(N, "valid", seed=0, limit=600, control=True)
    found = sum(bool(repeat_spans(s.flat, max_body=64)) for s in control)
    assert found > N // 2, "the matcher should find plain translated repeats"

    turned = [s for s in corpus() if s.step != D4().code]
    assert turned, "the transformed corpus must contain non-identity steps"
    missed = sum(not repeat_spans(s.flat, max_body=64) for s in turned)
    assert missed == len(turned), "a byte matcher cannot see a mirrored copy"


def test_provenance_lines_up_with_what_the_metrics_expect():
    """`recovery` wants (first, rest) and `recovery_by_copy` wants every copy.
    Both are derived from the same spans, so they cannot disagree."""
    scenes = corpus()
    spans = composed.spans_of(scenes)
    copies = composed.copies_of(scenes)
    assert len(copies) == len(scenes)
    for scene, (group,) in zip(scenes, copies):
        assert group == list(scene.copies)
    for scene, (pair,) in zip([s for s in scenes if len(s.copies) > 1], spans):
        head, rest = pair
        assert head == scene.copies[0]
        assert rest.start == scene.copies[1].start
        assert rest.stop == scene.copies[-1].stop


def test_the_corpus_is_deterministic_in_its_seed():
    """A corpus that moved between the run and the measurement would reproduce
    the run-4 fault, where a result was read against a split it never saw."""
    assert [s.flat for s in corpus()] == [s.flat for s in corpus()]
    other = composed.build(N, "valid", seed=1, limit=600)
    assert [s.flat for s in other] != [s.flat for s in corpus()]


def test_train_and_val_share_no_scene():
    """QuickDraw's own splits make the motif pools disjoint, so a val scene
    cannot be a memorised train motif wearing a new transform."""
    train, val = composed.split(40, 20, seed=0, limit=600)
    assert not set(train) & set(val)
    assert len(set(train)) == len(train)


def test_the_foldable_fraction_lands_where_the_design_says():
    """Not a property of the code so much as of the policy, and it is the number
    that decides whether the corpus is a real task: near 1.0 the program is
    nothing but copies, near 0 there is no signal to find."""
    fractions = np.array([s.foldable for s in corpus()])
    assert 0.15 < fractions.mean() < 0.60
    assert fractions.min() > 0.0


def test_the_control_and_the_transform_arms_are_matched_at_one_orbit_size():
    """The comparison the claim rests on. Same generator, same motif pool, same
    distractor policy -- only the group differs, so a difference in `recovery`
    cannot be a difference in program length or in how much there was to find."""
    turned = composed.build(N, "valid", seed=0, limit=600, orbit_sizes=(2,))
    control = composed.build(N, "valid", seed=0, limit=600, control=True)

    lengths = [np.mean([len(s.flat) for s in arm]) for arm in (turned, control)]
    folds = [np.mean([s.foldable for s in arm]) for arm in (turned, control)]
    assert abs(lengths[0] - lengths[1]) / lengths[1] < 0.15
    assert abs(folds[0] - folds[1]) < 0.05
    assert all(len(s.copies) == 2 for s in turned + control)
    assert {s.step for s in control} == {D4().code}
    assert D4().code not in {s.step for s in turned}


def test_a_control_orbit_of_four_is_refused_by_the_canvas():
    """Not a limitation of the generator: four disjoint *translated* copies of a
    quarter-canvas motif need a step smaller than the motif, so they would
    overlap. The rotations can do it because they use the canvas's own symmetry,
    and that asymmetry is a fact about the group rather than about the code."""
    with pytest.raises(ValueError):
        composed.build(5, "valid", seed=0, limit=600, control=True,
                       orbit_sizes=(4,))


def test_scenes_fit_the_stride_one_codecs():
    """`max_len` is where corpora die -- Tier D was 58% over. The bit codec is
    8x and is out of scope by design (the fusion axis this corpus revives is
    `token` vs `token_typed`, both stride 1), but the stride-1 arms must fit
    with room to spare."""
    lengths = np.array([len(s.flat) for s in corpus()])
    assert lengths.max() < 2048
    assert np.percentile(lengths, 99) < 1024


def test_provenance_has_one_entry_per_scene_even_when_it_is_empty():
    """`recovery` indexes `spans[i]` by program, so a comprehension that
    *filtered* would slide every later scene's copies onto the wrong program and
    still report a number. No orbit of one exists today, which is exactly why
    the alignment has to be structural rather than true by luck."""
    scenes = corpus()
    assert len(composed.spans_of(scenes)) == len(scenes)
    lone = composed.Scene(flat=b"", structured=b"", copies=(slice(0, 1),),
                          step=D4().code, saved=0, distractors=0)
    assert composed.spans_of([lone]) == [[]]


def test_the_constructed_ceiling_is_read_off_the_two_forms():
    """The denominator `recovery` is a fraction of. It is the difference between
    the flat and structured spellings and nothing else -- an oracle *searching*
    for the same structure can only ever report a lower bound, which is the
    whole reason this corpus exists."""
    scenes = corpus()
    ceiling = composed.ceiling(scenes)
    assert ceiling["saved"] == sum(len(s.flat) - len(s.structured) for s in scenes)
    assert ceiling["bytes"] == sum(len(s.flat) for s in scenes)
    assert ceiling["programs_with_repeat"] == 1.0
    assert 0.15 < ceiling["saved_frac"] < 0.60
    assert ceiling["source"] == "construction"


def test_the_val_seed_offset_has_exactly_one_definition():
    """A metric rebuilds the val half alone to recover the copy spans, so the
    offset cannot also live as a literal in whatever script needs it: a corpus
    free to drift from the one the trainer used reports real bits about bytes
    that are not copies."""
    _, val = composed.split_scenes(5, N, seed=3, limit=600)
    direct = composed.build(N, "valid", seed=composed.val_seed(3), limit=600)
    assert [s.flat for s in val] == [s.flat for s in direct]
    assert composed.val_seed(0) != 0


def test_the_structured_spelling_is_what_makes_the_fusion_axis_exist():
    """The reason `split` takes `structured` at all, and it is a measurement.

    A typed alphabet splits operand values by `Kind`, and a flat trace is
    `MOVE`/`LINE`/`HALT` -- one `Kind`, so `token` and `token_typed` spell it
    *byte-identically* and the axis is not merely small but absent. `REPEATX`
    brings `Kind.XF`, `Kind.COUNT` and `Kind.DELTA`, and the two codecs part.
    """
    scenes = corpus()
    token, typed = CODECS["token"], CODECS["token_typed"]
    assert all(token.encode(s.flat) == typed.encode(s.flat) for s in scenes)
    assert all(token.encode(s.structured) != typed.encode(s.structured)
               for s in scenes)


def test_split_returns_whichever_spelling_was_asked_for():
    """Two experiments on one corpus: flat is claim 2's setup, structured is the
    ISA v2 form. They must be the same scenes -- same seed, same draws -- or the
    two halves of P5 are measured on two different corpora."""
    flat_train, flat_val = composed.split(20, N, seed=0, limit=600)
    stru_train, stru_val = composed.split(20, N, seed=0, structured=True, limit=600)
    train, val = composed.split_scenes(20, N, seed=0, limit=600)
    assert flat_val == [s.flat for s in val]
    assert stru_val == [s.structured for s in val]
    assert flat_train == [s.flat for s in train]
    assert stru_train == [s.structured for s in train]
    assert all(len(a) < len(b) for a, b in zip(stru_val, flat_val))


def test_the_policy_caption_asks_the_generator_and_not_the_config():
    """The fault this function exists to prevent, and it shipped once: P5's
    control reports were captioned `n ∈ {2,4}` because `orbit_sizes` was absent
    from the config, while `orbits(control=True)` offers exactly one count. A
    caption is the only thing a reader has to tell the two arms of the comparison
    apart, so an unfalsifiable one is worse than none."""
    assert composed.policy_label({"control": True}) == (
        "translation only (control), n ∈ {2}, flat trace")
    assert composed.policy_label({}) == "D4 orbits, n ∈ {2, 4}, flat trace"
    assert composed.policy_label({"orbit_sizes": (2,)}) == (
        "D4 orbits, n ∈ {2}, flat trace")
    assert composed.policy_label({"structured": True}).endswith("REPEATX form")
    # A control corpus can never be captioned with a four-copy orbit, whatever
    # the config asked for -- the canvas is what refuses, not the flag.
    assert "4" not in composed.policy_label({"control": True, "orbit_sizes": (2, 4)})
    assert all(scene.copies and len(scene.copies) == 2
               for scene in corpus(control=True, orbit_sizes=(2, 4)))
