"""The between-drawing oracle: it must find planted sharing and price it honestly.

This is the denominator `Op.CALL` would need, and `docs/traps.md` carries the
rule that paid for `REPEATX`: *before adding an opcode to remove redundancy,
measure the redundancy it would remove.* An oracle that overstated the saving
would license an opcode the corpus cannot pay for.

Two things therefore have to hold. The arithmetic has to charge the **whole**
call site -- `CALL` takes a `Kind.ID` and no placement, so a body has to be
positioned by `XFORM`/`ENDX` and pretending otherwise prices a different opcode.
And the library has to be addressable: 256 bodies, because the operand is a byte.
"""

import pytest

from dm.data.augment import Affine, apply
from dm.eval.library import CALL_SITE_BYTES, MAX_LIBRARY, library_stats
from dm.isa.asm import assemble
from dm.isa.spec import SPECS, Op


def _blob(x: int, y: int) -> list[str]:
    return [f"MOVE {x} {y}", f"LINE {x + 20} {y}", f"LINE {x + 20} {y + 14}",
            f"LINE {x} {y + 14}", f"LINE {x} {y}"]


def _drawing(*origins: tuple[int, int]) -> bytes:
    lines: list[str] = []
    for x, y in origins:
        lines += _blob(x, y)
    lines.append("HALT")
    return assemble("\n".join(lines))


def test_the_call_site_cost_comes_from_the_isa_table():
    assert CALL_SITE_BYTES == (SPECS[Op.XFORM].size + SPECS[Op.CALL].size
                               + SPECS[Op.ENDX].size)
    assert MAX_LIBRARY == 256           # `Kind.ID` is one byte


def test_it_finds_a_body_shared_across_drawings_and_prices_it_exactly():
    # One 15-byte body in four drawings: 4 * 15 spelled out, 15 + 4 * 7 as a
    # library entry plus four call sites.
    programs = [_drawing((10, 10)), _drawing((90, 40)),
                _drawing((30, 120)), _drawing((150, 60))]
    stats = library_stats(programs)
    assert stats["library"] == 1
    assert stats["call_sites"] == 4
    assert stats["saved"] == 3 * 15 - CALL_SITE_BYTES * 4


def test_a_body_too_small_to_pay_for_its_call_site_is_not_counted():
    """`(k - 1) * b - 7k > 0` fails for a short body however often it recurs, and
    an oracle that counted it anyway would report a saving a real ISA cannot
    collect."""
    tiny = [assemble("MOVE 10 10\nLINE 12 12\nHALT") for _ in range(50)]
    assert library_stats(tiny)["saved"] == 0


def test_position_is_not_part_of_a_body_s_identity():
    """A `CALL` is placed by the transform tier, so the same shape at another
    position is the same body. Matching raw bytes would find nothing."""
    single = library_stats([_drawing((10, 10))] * 3)
    moved = library_stats([_drawing((10, 10)), _drawing((70, 30)), _drawing((5, 200))])
    assert moved["saved"] == single["saved"]


def test_the_d4_column_is_invariant_under_a_global_symmetry():
    """The reading that makes canonicalisation's null checkable from the other
    side: the oracle has quotiented by D4, so re-framing by a D4 element cannot
    move it."""
    programs = [_drawing((10, 10), (60, 90)), _drawing((30, 30)),
                _drawing((80, 120), (10, 10))]
    turned = [apply(p, Affine(0, 0, True, 1)) for p in programs]
    assert all(t is not None for t in turned)
    assert library_stats(turned, d4=True)["saved"] == library_stats(programs, d4=True)["saved"]


def test_a_mirrored_body_is_only_shared_under_the_d4_normalisation():
    upright = _drawing((10, 10))
    lines = ["MOVE 10 10", "LINE 10 30", "LINE 24 30", "LINE 24 10", "LINE 10 10",
             "HALT"]
    rotated = assemble("\n".join(lines))
    pair = [upright, rotated] * 3
    assert library_stats(pair, d4=True)["saved"] > library_stats(pair)["saved"]


def test_a_coarse_grid_is_flagged_as_not_a_compression_number():
    programs = [_drawing((10, 10)), _drawing((11, 11))]
    assert library_stats(programs, grid=1)["exact"]
    assert not library_stats(programs, grid=4)["exact"]


@pytest.mark.parametrize("grid", [1, 2, 8])
def test_a_corpus_that_shares_nothing_reports_nothing(grid):
    """QuickDraw reads 0.00% here at every grid, and a detector that could not
    report a clean zero would make that null unreadable."""
    programs = [_drawing((3 * i, 5 * i + 7)) for i in range(1, 6)]
    for i, _ in enumerate(programs):                      # make each shape unique
        programs[i] = assemble("\n".join(
            [f"MOVE {10 + i} 10", f"LINE {40 + 3 * i} {20 + i}",
             f"LINE {20 + i} {70 + 2 * i}", "HALT"]))
    assert library_stats(programs, grid=grid)["saved"] == 0
