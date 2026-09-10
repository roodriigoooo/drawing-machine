"""Re-spelling a corpus, and the operation whose damage is under test.

Two rules from `docs/traps.md` are enforced here rather than trusted. **A
transform refuses rather than clamps**, because clamping is a deformation and a
deformation of a repeated body is not a repeat. And **operands move by `Kind`,
never by position**: a radius is a length, a `REPEAT` step is a vector, a count
is neither, and treating every byte as a coordinate corrupts all three.

The third rule is new and was found by this experiment: **the sub-pixel rounding
phase is a draw.** The identical corpus under the identical ×1.07 retains a
different fraction of its ceiling depending on where the scale is centred, so a
retention read at one phase is one sample of a distribution.
"""

import numpy as np
import pytest

from dm.data.canonical import POLICIES
from dm.data.refit import (
    jitter_program,
    respell,
    respell_corpus,
    scale_program,
)
from dm.isa.asm import assemble, parse
from dm.isa.spec import CANVAS
from dm.vm.interp import VM


def _box(x: int = 40, y: int = 40, w: int = 60) -> bytes:
    return assemble("\n".join([
        f"MOVE {x} {y}", f"LINE {x + w} {y}", f"LINE {x + w} {y + w}",
        f"LINE {x} {y + w}", f"LINE {x} {y}", "HALT",
    ]))


def test_a_scale_that_leaves_the_canvas_is_refused_not_clamped():
    assert scale_program(_box(10, 10, 200), 2.0) is None
    assert scale_program(_box(40, 40, 60), 1.07) is not None


def test_a_scale_moves_operands_by_kind():
    """A `CIRCLE` radius and a `REPEAT` step are not coordinates, and a scale
    that treated them as such would draw a different picture at every factor."""
    program = assemble("MOVE 60 60\nCIRCLE 10\nREPEAT 3 8 0\nLINE 70 70\nENDREP\nHALT")
    scaled = scale_program(program, 2.0)
    assert scaled is not None
    by_mnemonic = {i.mnemonic: i.args for i in parse(scaled)}
    assert by_mnemonic["CIRCLE"] == (20,)               # a length, scaled
    assert by_mnemonic["REPEAT"][0] == 3                # a count, invariant
    assert by_mnemonic["REPEAT"][1:] == (16, 0)         # a vector, scaled


def test_an_integer_factor_is_exact_and_leaves_a_repeat_intact():
    """`round(k * x) == k * x`, so an integer scale is in the transform group by
    arithmetic. What keeps it out of the ISA is the canvas, not the rounding."""
    from dm.eval.repeats import compress

    # Centred on the canvas and small, so doubling about its own centre still
    # fits: an integer scale is exact, and the canvas is a *separate* limit.
    lines: list[str] = []
    for k in range(3):
        lines += [f"MOVE {110 + 12 * k} 124", f"LINE {118 + 12 * k} 124",
                  f"LINE {118 + 12 * k} 132"]
    program = assemble("\n".join([*lines, "HALT"]))
    before = compress(program)[0]
    assert before > 0
    scaled = scale_program(program, 2.0)
    assert scaled is not None
    assert compress(scaled)[0] == before


def test_the_rounding_phase_changes_the_result():
    """The finding that makes a single-centre retention a draw rather than a
    number: a sub-pixel offset decides which side each coordinate falls on."""
    program = _box(41, 41, 61)
    outputs = {scale_program(program, 1.07, (p / 8.0, p / 8.0)) for p in range(8)}
    assert len(outputs) > 1


def test_the_scale_centre_is_a_convention_and_both_are_available():
    program = _box(41, 41, 61)
    assert scale_program(program, 1.07, about="bbox") != \
        scale_program(program, 1.07, about="origin")
    with pytest.raises(ValueError):
        scale_program(program, 1.07, about="nowhere")


def test_a_jitter_that_leaves_the_canvas_is_refused():
    rng = np.random.default_rng(0)
    edge = assemble(f"MOVE 0 0\nLINE {CANVAS - 1} {CANVAS - 1}\nHALT")
    assert any(jitter_program(edge, 1, rng) is None for _ in range(20))


def test_a_respelled_program_executes_and_keeps_its_stroke_count():
    program = _box()
    for allow_curve in (False, True):
        result = respell(program, 1.0, allow_curve, POLICIES["none"])
        assert result is not None
        respelled, error = result
        trace = VM().run(respelled)
        assert trace.valid and trace.halted
        assert len(trace.strokes) == len(VM().run(program).strokes)
        assert error <= 1.0 + 1e-9


def test_a_drawing_the_fitter_cannot_spell_is_refused_and_counted():
    """A disc is geometry a stroke fitter cannot carry, and dropping it silently
    would change the drawing while leaving the corpus looking complete."""
    disc = assemble("MOVE 60 60\nCIRCLE 20\nHALT")
    assert respell(disc, 1.0, True, POLICIES["none"]) is None
    _, stats = respell_corpus([disc, _box()], 1.0, True)
    assert stats["refused"] == 1 and stats["with_regions"] == 1


def test_the_source_index_travels_with_the_respelled_program():
    """Two spellings refuse different drawings, and a comparison between them has
    to be paired on the source rather than on position in a shrunken list."""
    disc = assemble("MOVE 60 60\nCIRCLE 20\nHALT")
    programs = [_box(20, 20, 40), disc, _box(60, 60, 50)]
    out, stats = respell_corpus(programs, 1.0, True)
    assert stats["keep"] == [0, 2] and len(out) == 2


def test_the_two_arms_report_the_error_they_were_asked_for():
    """The byte column only means something at matched fidelity, so both arms
    have to honour the same tolerance -- it is the one knob they share."""
    programs = [_box(30, 30, 80), _box(50, 20, 60)]
    for tol in (1.0, 2.0, 4.0):
        for allow_curve in (False, True):
            _, stats = respell_corpus(programs, tol, allow_curve)
            assert stats["error_max"] <= tol + 1e-9
