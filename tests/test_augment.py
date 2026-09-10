"""Integer-affine augmentation: it must add samples and lose no structure."""

from itertools import product

import pytest

from dm.data.augment import Affine, apply, expand, policies
from dm.eval.repeats import corpus_stats
from dm.isa.asm import assemble, parse
from dm.isa.spec import CANVAS
from dm.isa.strokes import join, split
from dm.isa.transform import D4
from dm.vm.interp import VM


def grid(n: int = 3, dx: int = 40, dy: int = 0, x0: int = 10, y0: int = 10) -> bytes:
    lines = []
    for i in range(n):
        x, y = x0 + dx * i, y0 + dy * i
        lines += [f"MOVE {x} {y}", f"LINE {x + 20} {y}", f"LINE {x + 20} {y + 20}"]
    lines.append("HALT")
    return assemble("\n".join(lines))


def test_translation_preserves_the_repeat_exactly():
    """The property the whole module exists for: rounding commutes with an
    integer translation, so a REPEAT stays emittable (docs/tier-c.md)."""
    before = corpus_stats([grid()])
    after = corpus_stats([apply(grid(), Affine(dx=13, dy=7))])
    assert after["saved"] == before["saved"] > 0


@pytest.mark.parametrize("transform", [
    Affine(dx=5, dy=-5), Affine(mirror=True), Affine(quarter_turns=1),
    Affine(quarter_turns=2), Affine(quarter_turns=3, mirror=True, dx=4),
])
def test_every_generator_preserves_the_repeat(transform):
    assert corpus_stats([apply(grid(), transform)])["saved"] == corpus_stats([grid()])["saved"]


def test_out_of_canvas_is_rejected_not_clamped():
    """Clamping is a deformation, which is what this module avoids."""
    program = grid(x0=200, dx=0, n=1)          # valid, but hard against the edge
    assert apply(program, Affine(dx=1)) is not None
    assert apply(program, Affine(dx=100)) is None


def test_scalar_operands_are_lengths_and_do_not_move():
    """A CIRCLE's radius and a WIDTH are invariant under this group; treating
    them positionally as coordinates would corrupt both."""
    program = assemble("WIDTH 3\nMOVE 10 10\nCIRCLE 7\nHALT")
    got = parse(apply(program, Affine(dx=5, dy=5, mirror=True, quarter_turns=1)))
    assert [i.args for i in got if i.mnemonic == "WIDTH"] == [(3,)]
    assert [i.args for i in got if i.mnemonic == "CIRCLE"] == [(7,)]


def test_repeat_delta_takes_the_linear_part_only():
    """REPEAT's dx/dy are a vector: rotated and mirrored, never translated."""
    program = assemble("REPEAT 3 10 0\nMOVE 20 20\nLINE 40 20\nENDREP\nHALT")
    turned = parse(apply(program, Affine(quarter_turns=1)))
    assert turned[0].args == (3, 0, -10)          # (10, 0) rotated a quarter turn
    mirrored = parse(apply(program, Affine(mirror=True)))
    assert mirrored[0].args == (3, -10, 0)


def test_expand_multiplies_the_corpus_and_keeps_the_original_first():
    programs = [grid()]
    out = expand(programs, policies(shifts=(-8, 8), mirror=True))
    assert out[0] == programs[0], "the identity must come first"
    assert len(out) > 8


def test_expand_preserves_the_oracle_ceiling_per_byte():
    """More bytes and proportionally the same structure -- the acceptance test
    any augmentation policy has to pass before it is adopted."""
    programs = [grid(), grid(n=4, dx=0, dy=30, x0=60)]
    base = corpus_stats(programs)
    aug = corpus_stats(expand(programs, policies()))
    assert aug["bytes"] > 5 * base["bytes"]
    assert abs(aug["saved_frac"] - base["saved_frac"]) < 1e-9


def test_a_deforming_policy_would_fail_that_acceptance_test():
    """Documents why the group is restricted: a non-integer scale destroys most
    of the ceiling, which is the SVG-Icons8 failure in miniature."""
    scaled = []
    for program in [grid()]:
        lines = []
        for instr in parse(program):
            args = [round(a * 1.07) for a in instr.args]
            assert all(a < CANVAS for a in args)
            lines.append(" ".join([instr.mnemonic, *map(str, args)]))
        scaled.append(assemble("\n".join(lines)))
    assert corpus_stats(scaled)["saved"] < corpus_stats([grid()])["saved"]


# --------------------------------------------------------------------------
# ISA v2: the transform tier


def transformed(mnemonic: str = "XFORM") -> bytes:
    """A program whose scopes have to be re-expressed, not just re-addressed."""
    if mnemonic == "XFORM":
        return assemble(f"XFORM {D4(1, False).code} 10 5\nMOVE 30 40\nLINE 60 45\n"
                        "ENDX\nMOVE 20 20\nLINE 25 25\nHALT")
    return assemble(f"REPEATX 3 {D4(1, False).code} 8 0\nMOVE 60 70\nLINE 80 75\n"
                    "ENDREP\nHALT")


@pytest.mark.parametrize("source", ["XFORM", "REPEATX"])
def test_augmenting_a_scoped_program_commutes_with_running_it(source):
    """**The property the transform tier has to have and does not get for
    free.** Transforming a program by `A` must draw exactly `A` applied to what
    the program drew -- so a scope's own transform has to be conjugated into the
    new frame (`Transform.conjugate`), not carried across unchanged.

    D4 is non-abelian and the failure is invisible on the identity, on pure
    translations of unscoped programs, and on a mirror composed with itself. It
    shows up on a rotation meeting an offset, which is every interesting
    program, so this sweeps all eight elements against three shifts.
    """
    program = transformed(source)
    vm = VM()
    assert vm.run(program).valid
    before = [s.points for s in vm.run(program).strokes]
    assert before, "the fixture has to draw something"

    checked = 0
    for turns, mirror, dx, dy in product(range(4), (False, True), (0, 7, -5), (0, -3)):
        transform = Affine(dx, dy, mirror, turns)
        moved = apply(program, transform)
        if moved is None:
            continue                      # left the canvas: rejected, not clamped
        want = [tuple(tuple(float(v) for v in transform.point(int(x), int(y)))
                      for x, y in stroke) for stroke in before]
        assert [s.points for s in vm.run(moved).strokes] == want, transform
        checked += 1
    assert checked >= 40, "the sweep has to actually run"


def test_a_repeat_step_is_conjugated_and_not_merely_carried():
    """A quarter turn seen through a mirror is the *opposite* quarter turn. An
    implementation that copied the operand across would pass every identity and
    translation test and draw the wrong picture under a reflection."""
    program = transformed("REPEATX")
    mirrored = apply(program, Affine(mirror=True))
    assert mirrored is not None
    step = next(i.args[1] for i in parse(mirrored) if i.mnemonic == "REPEATX")
    assert step == D4.of(D4(1, False).code).conjugate(D4(0, True)).code
    assert step != D4(1, False).code, "a mirror must not leave a rotation alone"


def test_a_scope_that_cannot_be_respelled_in_i8_is_rejected():
    """Conjugation moves an `XFORM`'s shift by the outer translation, and the
    operand is i8. Rejected rather than wrapped, for the same reason a drawing
    that leaves the canvas is rejected rather than clamped."""
    program = assemble(f"XFORM {D4(2, False).code} 120 120\nMOVE 30 40\nLINE 60 45\n"
                       "ENDX\nHALT")
    assert apply(program, Affine(dx=0, dy=0)) is not None
    assert apply(program, Affine(dx=60, dy=60)) is None


def test_stroke_splitting_survives_a_mirror_byte_for_byte():
    """`docs/direction.md` §2.7 flags that mirroring reverses stroke order under
    some conventions. This one rewrites operands in place and never reorders
    instructions, so `join(split(p)) == p` still holds -- and it has to, because
    claim 3's parameter-matching argument rests on it."""
    for transform in (Affine(mirror=True), Affine(quarter_turns=1),
                      Affine(quarter_turns=3, mirror=True, dx=4)):
        moved = apply(grid(), transform)
        assert moved is not None
        assert join(split(moved)) == moved
        assert [i.mnemonic for i in parse(moved)] == [i.mnemonic for i in parse(grid())]
