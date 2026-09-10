"""The Tier C ingest: geometry in, and a split that cannot leak."""

from pathlib import Path

import pytest

from dm.data.tabler import SCALE, family, split, svg_to_program
from dm.isa.asm import parse
from dm.isa.spec import CANVAS

ICONS = Path("data/tabler/icons")
needs_corpus = pytest.mark.skipif(
    not (ICONS / "outline").exists(),
    reason="Tabler not cloned; see docs/tier-c.md",
)

MENU = """<svg viewBox="0 0 24 24">
  <path d="M4 6l16 0" /><path d="M4 12l16 0" /><path d="M4 18l16 0" />
</svg>"""


def test_integer_grid_scaling_is_exact():
    """24 x 10 = 240 <= 255. The bars land on exact multiples, which is the
    whole reason Tabler replaced SVG-Icons8 (docs/tier-c.md)."""
    instrs = parse(svg_to_program(MENU))
    ys = [i.args[1] for i in instrs if i.mnemonic in ("MOVE", "LINE")]
    assert ys == [60, 60, 120, 120, 180, 180]
    assert max(ys) < CANVAS


def test_a_translated_repeat_survives_quantisation():
    """Rounding commutes with integer translation, so the oracle still sees it."""
    from dm.eval.repeats import best_repeat

    repeat = best_repeat(parse(svg_to_program(MENU)))
    assert repeat is not None and repeat.count == 3 and (repeat.dx, repeat.dy) == (0, 60)


def test_family_strips_variant_suffixes():
    assert family("battery-4") == "battery"
    assert family("arrow-up") == "arrow"
    assert family("bell-off") == "bell"
    assert family("star-filled") == "star"
    assert family("wifi") == "wifi"


@needs_corpus
def test_split_is_disjoint_and_deterministic():
    train, val = split(ICONS)
    assert train and val
    assert not (set(train) & set(val)), "an identical program is in both splits"
    assert (train, val) == split(ICONS), "split is not reproducible for a fixed seed"


@needs_corpus
def test_split_val_fraction_is_approximately_honoured():
    train, val = split(ICONS, val_frac=0.1)
    frac = len(val) / (len(train) + len(val))
    assert 0.08 < frac < 0.14, frac


@needs_corpus
def test_every_program_parses_and_scales_into_the_canvas():
    train, val = split(ICONS)
    for program in (train + val)[:500]:
        for instr in parse(program):
            assert all(0 <= a < CANVAS for a in instr.args)
    assert SCALE * 24 <= CANVAS - 1
