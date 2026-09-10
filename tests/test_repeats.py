"""The repeat oracle: it must find planted structure and refuse noise."""

import random

import pytest

from dm.data import quickdraw
from dm.data.augment import Affine, apply
from dm.eval.repeats import (
    REPEAT_OVERHEAD,
    best_repeat,
    best_symmetry_repeat,
    compress,
    corpus_stats,
    symmetry_stats,
)
from dm.isa.asm import assemble, parse
from dm.isa.spec import Op
from dm.isa.transform import D4, Transform


def _grid(n: int, dx: int, dy: int) -> bytes:
    """`n` copies of a square, each offset (dx, dy) from the last. Flat L0."""
    lines = []
    for i in range(n):
        x, y = 10 + dx * i, 10 + dy * i
        lines += [f"MOVE {x} {y}", f"LINE {x + 20} {y}", f"LINE {x + 20} {y + 20}",
                  f"LINE {x} {y + 20}", f"LINE {x} {y}"]
    lines.append("HALT")
    return assemble("\n".join(lines))


def test_finds_a_planted_translational_repeat():
    repeat = best_repeat(parse(_grid(3, 40, 0)))
    assert repeat is not None
    assert (repeat.body_instrs, repeat.count, repeat.dx, repeat.dy) == (5, 3, 40, 0)
    # Two redundant copies of a 15-byte body, less the REPEAT/ENDREP overhead.
    assert repeat.saved == 2 * 15 - REPEAT_OVERHEAD


def test_savings_scale_with_the_number_of_copies():
    saved3, _ = compress(_grid(3, 40, 0))
    saved5, _ = compress(_grid(5, 40, 0))
    assert saved5 - saved3 == 2 * 15


def test_diagonal_repeats_are_found_too():
    repeat = best_repeat(parse(_grid(4, 12, 7)))
    assert repeat is not None and (repeat.dx, repeat.dy) == (12, 7)


def test_offsets_outside_i8_are_refused():
    """REPEAT's dx/dy are i8; a 200-unit step is not encodable."""
    assert best_repeat(parse(_grid(2, 200, 0))) is None


def test_a_single_copy_is_not_a_repeat():
    assert best_repeat(parse(_grid(1, 40, 0))) is None


def test_a_repeat_that_does_not_pay_for_itself_is_refused():
    """One redundant MOVE is 3 bytes against 5 bytes of overhead."""
    program = assemble("MOVE 10 10\nMOVE 20 10\nHALT")
    assert best_repeat(parse(program)) is None


def test_random_coordinates_yield_almost_nothing():
    rng = random.Random(0)
    programs = []
    for _ in range(40):
        lines = [f"MOVE {rng.randint(0, 255)} {rng.randint(0, 255)}"]
        lines += [f"LINE {rng.randint(0, 255)} {rng.randint(0, 255)}"
                  for _ in range(20)]
        lines.append("HALT")
        programs.append(assemble("\n".join(lines)))
    assert corpus_stats(programs)["saved_frac"] < 0.01


def test_tolerance_separates_absent_structure_from_destroyed_structure():
    """A repeat knocked one unit off is invisible at tol=0 and found at tol=1.

    This is the SVG-Icons8 diagnosis in miniature: 0.62% of bytes compressible
    exactly against 3.17% at +-1 px meant the pipeline had rounded the repeats
    apart, not that the icons lacked them.
    """
    lines = []
    for i in range(3):
        x, y = 10 + 40 * i, 10 + (1 if i == 2 else 0)   # third copy is 1 off
        lines += [f"MOVE {x} {y}", f"LINE {x + 20} {y}", f"LINE {x + 20} {y + 20}",
                  f"LINE {x} {y + 20}", f"LINE {x} {y}"]
    lines.append("HALT")
    program = assemble("\n".join(lines))
    assert compress(program, tol=0)[0] < compress(program, tol=1)[0]


def test_corpus_stats_reports_a_ratio_above_one_only_when_structure_exists():
    stats = corpus_stats([_grid(4, 30, 0)] * 5)
    assert stats["ratio"] > 1.0
    assert stats["programs_with_repeat"] == 1.0



# --------------------------------------------------------------------------
# ISA v2: orbits under D4


def orbit_program(code: int, n: int = 2, dx: int = 0, dy: int = 0) -> bytes:
    """A motif and its images, written flat -- what `unroll` would emit."""
    body = assemble("MOVE 20 30\nLINE 60 35\nLINE 55 70")
    step = Transform(D4.of(code), dx, dy)
    out = b""
    for k in range(n):
        copy = body if k == 0 else apply(body, Affine.of(step.power(k)))
        assert copy is not None
        out += copy
    return out + bytes([int(Op.HALT)])


def test_a_mirrored_orbit_is_invisible_to_the_translational_oracle():
    """`x -> 255 - x` changes every coordinate byte, so byte matching finds
    nothing. This is the measurement `REPEATX` exists to make possible, stated
    as the failure of the instrument that came before it."""
    program = orbit_program(D4(0, True).code)
    assert corpus_stats([program])["saved"] == 0
    found = best_symmetry_repeat(program)
    assert found is not None
    assert found.count == 2
    assert found.step.d4 == D4(0, True)


@pytest.mark.parametrize("code", [1, 2, 3, 4, 7])
def test_every_non_identity_element_is_found_with_its_own_element(code):
    """A rotation reported as a mirror would put the wrong opcode in the
    structured form and the ceiling would be for a drawing nobody asked for."""
    found = best_symmetry_repeat(orbit_program(code))
    assert found is not None
    assert found.step.d4.code == code


def test_a_translated_orbit_is_found_by_both_and_costs_one_byte_more():
    """`REPEATX` carries its element as well as its step, so it is one byte
    wider than `REPEAT`. Both oracles see a pure translation, and the D4 one
    must report exactly that byte less -- which is the check that the two
    overheads are not silently the same number."""
    program = orbit_program(D4().code, n=3, dx=60, dy=0)
    flat = corpus_stats([program])["saved"]
    found = best_symmetry_repeat(program)
    assert found is not None and found.step.d4 == D4()
    assert found.saved == flat - 1


def test_natural_quickdraw_has_no_transformed_reuse():
    """The denominator. A constructed ceiling means nothing if the corpus it is
    built from already had one -- and QuickDraw does not: 0 of 300 programs
    carry a D4 orbit at all, so every fold the composed corpus offers is
    structure the generator put there."""
    programs = quickdraw.load(("cat", "bus"), "valid", limit=120)
    stats = symmetry_stats(programs, max_body=64)
    assert stats["with_orbit"] == 0
    assert stats["fraction"] == 0.0
