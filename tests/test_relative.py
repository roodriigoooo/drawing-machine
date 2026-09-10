"""The relative view: exact, information-preserving, and cheap under translation.

The third property is the one the axis exists for. Tier C's x18 augmentation
was refuted because absolute coordinates make a translated icon a *new* icon --
so the claim that a relative view fixes that is pinned here as a byte count,
before any training run is spent on it.
"""

from pathlib import Path

import pytest

from dm.data import synthetic, tabler
from dm.data.augment import Affine, apply
from dm.isa.asm import assemble, parse
from dm.isa.codec import CODECS, RelativeCodec
from dm.isa.relative import to_absolute, to_relative
from dm.isa.spec import Kind, SPECS, Op
from dm.vm.interp import VM

CORPUS = synthetic.dataset(200, seed=0)
DELTA_CODECS = [c for c in CODECS.values() if isinstance(c, RelativeCodec)]

ICONS = Path("data/tabler/icons")
needs_tabler = pytest.mark.skipif(
    not (ICONS / "outline").exists(), reason="Tabler not cloned; see docs/tier-c.md"
)


def test_view_roundtrips_on_every_program():
    for program in CORPUS:
        assert to_absolute(to_relative(program)) == program


def test_view_is_bijective_in_both_directions():
    """Also from the delta side, which is the direction generation runs in: a
    sampled stream is delta-domain and has to map onto exactly one program."""
    for program in CORPUS[:50]:
        relative = to_relative(program)
        assert to_relative(to_absolute(relative)) == relative


@pytest.mark.parametrize("codec", DELTA_CODECS, ids=lambda c: c.name)
def test_delta_arm_is_the_same_length_and_vocabulary_as_its_absolute_partner(codec):
    """Relativity has to be orthogonal to the other three axes or it is not an
    axis. Identical length and vocabulary is what makes a delta/absolute
    difference attributable to the view and to nothing else."""
    inner = codec.inner
    assert codec.vocab_size == inner.vocab_size
    assert codec.stride == inner.stride
    for program in CORPUS[:50]:
        assert len(codec.encode(program)) == len(inner.encode(program))


def test_opcodes_are_untouched_so_the_parse_is_identical():
    """The halt monitor walks instruction boundaries. It is correct in the
    delta domain only because the rewrite moves operand values and never an
    opcode byte, so both views parse into the same instruction sequence."""
    for program in CORPUS[:50]:
        relative = to_relative(program)
        assert [i.mnemonic for i in parse(relative)] == [i.mnemonic for i in parse(program)]


def test_non_coordinate_operands_are_untouched():
    """A CIRCLE radius and a WIDTH are lengths and a REPEAT's dx/dy is already a
    displacement. Rewriting any of them against the pen would corrupt it, and
    the by-Kind rule is what prevents that."""
    program = assemble(
        "WIDTH 3\nMOVE 10 20\nCIRCLE 7\nREPEAT 3 8 -4\nLINE 30 40\nENDREP\nHALT"
    )
    relative = to_relative(program)
    for original, rewritten in zip(parse(program), parse(relative)):
        spec = SPECS[Op[original.mnemonic]]
        for kind, before, after in zip(spec.operands, original.args, rewritten.args):
            if kind is not Kind.COORD:
                assert before == after, (original.mnemonic, kind)


def test_translation_costs_exactly_one_coordinate_pair():
    """The measurement the axis is owed.

    Under absolute coordinates a translated program shares no coordinate byte
    with its original -- which is why x18 augmentation multiplied Tier C's task
    as fast as its corpus (`PLAN.md` section 7). Under the relative view only
    the first coordinate pair moves, whatever the program's length.
    """
    shift = Affine(dx=7, dy=-5)
    checked, absolute_cost = 0, []
    for program in CORPUS:
        moved = apply(program, shift)
        if moved is None:  # left the canvas; not this test's subject
            continue
        checked += 1
        absolute_diff = sum(a != b for a, b in zip(program, moved))
        relative_diff = sum(
            a != b for a, b in zip(to_relative(program), to_relative(moved))
        )
        assert relative_diff <= 2, (relative_diff, absolute_diff)
        absolute_cost.append(absolute_diff)
    assert checked > 100
    # And the absolute view really is expensive, or the bound above proves
    # nothing: it moves *every* coordinate byte, so its cost grows with the
    # program while the relative view's stays at one pair. The floor is 2 --
    # a MOVE/CIRCLE program has only one pair to move -- so the claim is about
    # the distribution, not the minimum.
    assert sum(absolute_cost) / checked > 20
    assert max(absolute_cost) > 40


def test_mirrors_and_turns_stay_full_price():
    """The other half of the same statement, and the one a policy could get
    wrong: the linear part acts on the deltas themselves, so a mirror is still
    a genuinely new sequence. A relative view licenses translation-heavy
    augmentation and nothing else."""
    pairs = (
        (p, apply(p, Affine(mirror=True))) for p in CORPUS if len(p) > 20
    )
    program, mirrored = next((p, m) for p, m in pairs if m is not None)
    differing = sum(a != b for a, b in zip(to_relative(program), to_relative(mirrored)))
    assert differing > 2


def test_decode_is_permissive_on_a_malformed_stream():
    """Validity is measured in the VM for all eight codecs. A codec that raised
    here would move one arm's failures out of that measurement."""
    garbage = bytes([0xFE, 0x01, 0x02, 0x03])
    assert to_absolute(garbage) == garbage  # unknown opcode: copied verbatim
    truncated = assemble("MOVE 10 20") + bytes([int(Op.LINE), 5])  # LINE lacks its y
    assert to_absolute(to_relative(truncated)) == truncated


@pytest.mark.parametrize("codec", DELTA_CODECS, ids=lambda c: c.name)
def test_delta_codecs_render_the_same_drawing(codec):
    """End to end: encode under the delta view, decode, execute. The geometry
    has to be the absolute program's, or the view is not a view."""
    vm = VM()
    for program in CORPUS[:30]:
        decoded = codec.decode(codec.encode(program))
        assert vm.run(decoded).strokes == vm.run(program).strokes


@needs_tabler
def test_tabler_roundtrips_under_the_relative_view():
    """Tier C is the corpus the axis was built for, and its programs contain the
    repeats that make claim 2 measurable. Checked on the real corpus rather than
    on synthetic alone."""
    for program in tabler.load(ICONS, limit=200):
        assert to_absolute(to_relative(program)) == program
