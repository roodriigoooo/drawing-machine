"""The transform tier: the group, the VM's semantics, and the ceiling.

Three things have to hold before any `XFORM` measurement means anything.

**The group has to be a group**, checked by its *action* on the canvas rather
than against a hand-written multiplication table -- D4 is non-abelian and a
transcribed table is a silent transposition waiting to happen.

**Its convention has to be `dm/data/augment.py:Affine`'s, on all eight
elements.** The augmenter and the ISA describe the same eight symmetries, and
two conventions agreeing on seven of them would be a fault nothing downstream
could see: an augmented L2 program and an `XFORM` program would simply mean
different drawings.

**And the composition order with `REPEAT` has to be the one written down.** A
nested program means two different drawings under the other order, so it is
pinned here rather than recovered from whichever run happens first.
"""

from itertools import product

import pytest

from dm.data.augment import Affine
from dm.isa.asm import assemble
from dm.isa.spec import CANVAS, MAX_REPEAT_DEPTH, ISAError
from dm.isa.transform import D4, D4_ORDER, IDENTITY, Transform, compose_all
from dm.vm.interp import VM, FaultKind

ALL = [D4.of(code) for code in range(D4_ORDER)]
#: Corners, edges and interior. Enough that a wrong element cannot agree by luck.
PROBES = [(x, y) for x in (0, 1, 7, 128, 254, CANVAS - 1)
          for y in (0, 3, 99, 200, CANVAS - 1)]


def test_d4_is_closed_associative_and_invertible_by_its_action():
    for a, b in product(ALL, ALL):
        composed = a.then(b)
        assert all(composed.point(*p) == b.point(*a.point(*p)) for p in PROBES), (a, b)
    for a in ALL:
        assert a.then(a.inverse()) == D4()
        assert a.inverse().then(a) == D4()
    for a, b, c in product(ALL, ALL, ALL):
        assert a.then(b).then(c) == a.then(b.then(c))


def test_the_eight_elements_are_distinct_and_the_codes_round_trip():
    """Eight, not four and not sixteen: a mirror composed with a mirror is the
    identity, so a parameterisation that double-counted would show up here."""
    assert len({tuple(d.point(1, 2) + d.point(200, 3)) for d in ALL}) == D4_ORDER
    assert [D4.of(d.code) for d in ALL] == ALL
    with pytest.raises(ISAError):
        D4.of(8)


def test_the_convention_is_the_augmenters_on_every_element():
    """`dm/data/augment.py` and the ISA have to describe the same group action,
    or an augmented corpus and an `XFORM` corpus disagree about what a mirror
    is -- and both would look internally consistent."""
    for turns, mirror in product(range(4), (False, True)):
        d4, affine = D4(turns, mirror), Affine(0, 0, mirror, turns)
        assert all(d4.point(*p) == affine.point(*p) for p in PROBES)
        assert all(d4.linear(*p) == affine.linear(*p) for p in PROBES)


def test_a_transform_maps_the_canvas_onto_itself_before_translation():
    """Integer arithmetic and no half-pixel centre, which is what keeps a
    transformed repeat *exact*. Only the translation can leave the canvas, and
    that is the corpus generator's business to reject."""
    for d4 in ALL:
        for x, y in PROBES:
            u, v = d4.point(x, y)
            assert 0 <= u < CANVAS and 0 <= v < CANVAS


def test_translation_composes_through_the_linear_part():
    """`B(A p + t_A) + t_B = (BA) p + (B t_A + t_B)`. Getting this wrong is
    invisible on the identity and on pure translations, and wrong everywhere a
    rotation meets an offset -- which is every interesting program."""
    a = Transform.of(D4(1, False).code, 5, -3)
    b = Transform.of(D4(0, True).code, -7, 11)
    composed = a.then(b)
    for x, y in PROBES:
        assert composed.point(x, y) == b.point(*a.point(x, y))


def test_powers_are_repeated_composition_and_a_mirror_is_an_involution():
    t = Transform.of(D4(1, False).code, 3, -4)
    assert t.power(0) == IDENTITY and t.power(1) == t
    for k in range(2, 6):
        assert t.power(k) == t.power(k - 1).then(t)
    assert Transform.of(D4(0, True).code, 0, 0).power(2).is_identity
    # Four quarter turns is the identity on points, translation included.
    assert Transform.of(D4(1, False).code, 0, 0).power(4).is_identity


def test_compose_all_lets_the_innermost_scope_act_first():
    """`stack[0]` is the outermost open scope and acts *last* -- SVG's rule, and
    the only fold under which "an `XFORM` transforms everything inside it" holds
    for nested scopes. The other order is a different drawing, not a different
    spelling, because D4 is non-abelian: this test fails under it."""
    outer = Transform.of(D4(0, True).code, 0, 5)
    inner = Transform.of(D4(1, False).code, 2, 0)
    for x, y in PROBES:
        assert compose_all([outer, inner]).point(x, y) == outer.point(*inner.point(x, y))
    assert compose_all([outer, inner]) != compose_all([inner, outer])


# --------------------------------------------------------------------------
# the VM


def geometry(source: str):
    trace = VM().run(assemble(source))
    return trace, [s.points for s in trace.strokes]


def test_an_identity_xform_changes_nothing():
    """The tier has to be free when it is not used, or every pre-v2 trace moves
    and every recorded result with it."""
    plain, want = geometry("MOVE 10 20\nLINE 40 20\nHALT")
    scoped, got = geometry("XFORM 0 0 0\nMOVE 10 20\nLINE 40 20\nENDX\nHALT")
    assert got == want
    assert scoped.valid and plain.valid


def test_a_scoped_transform_applies_inside_and_not_outside():
    mirror = D4(0, True).code
    _, got = geometry(f"XFORM {mirror} 0 0\nMOVE 10 20\nLINE 40 20\nENDX\n"
                      "MOVE 10 20\nLINE 40 20\nHALT")
    assert got[0] == ((245.0, 20.0), (215.0, 20.0))
    assert got[1] == ((10.0, 20.0), (40.0, 20.0)), "ENDX must restore the frame"


def test_repeatx_draws_the_body_under_the_step_applied_k_times():
    """The compression primitive. Iteration k runs under `T^k`, so `REPEATX 4`
    with a quarter turn is a four-fold rotationally symmetric figure from one
    body."""
    quarter = D4(1, False).code
    _, got = geometry(f"REPEATX 4 {quarter} 0 0\nMOVE 10 20\nLINE 40 20\nENDREP\nHALT")
    assert len(got) == 4
    step = Transform.of(quarter, 0, 0)
    for k, stroke in enumerate(got):
        assert stroke[0] == tuple(float(v) for v in step.power(k).point(10, 20))


def test_repeatx_is_repeat_when_its_transform_is_the_identity():
    """One loop implementation for both, so the iteration accounting cannot
    drift between them -- and that accounting is what claim 2 measures."""
    _, plain = geometry("REPEAT 3 30 0\nMOVE 10 10\nLINE 20 10\nENDREP\nHALT")
    _, same = geometry("REPEATX 3 0 30 0\nMOVE 10 10\nLINE 20 10\nENDREP\nHALT")
    assert same == plain


def test_the_repeat_offset_is_untransformed_and_the_transform_maps_the_result():
    """PLAN.md, direction item 2, decision 4. The other order is a different
    drawing, so this is the assertion that the specification was implemented
    rather than reverse-engineered from a run."""
    mirror = D4(0, True).code
    _, got = geometry(f"XFORM {mirror} 0 0\nREPEAT 2 20 0\n"
                      "MOVE 10 20\nLINE 30 20\nENDREP\nENDX\nHALT")
    # Offsets 0 and 20 applied first, then the mirror: 10 -> 245, 30 -> 225.
    assert got[0] == ((245.0, 20.0), (225.0, 20.0))
    assert got[1] == ((225.0, 20.0), (205.0, 20.0))


def test_nested_scopes_apply_the_inner_one_first():
    """The enclosing scope transforms what the inner scope produced, which is
    what "wraps" means. Asserted against the composition *and* against the
    hand-applied order, so a change to `compose_all` cannot quietly take the
    VM with it."""
    quarter, mirror = D4(1, False).code, D4(0, True).code
    _, got = geometry(f"XFORM {quarter} 0 0\nXFORM {mirror} 0 0\n"
                      "MOVE 10 20\nLINE 40 20\nENDX\nENDX\nHALT")
    outer, inner = Transform.of(quarter, 0, 0), Transform.of(mirror, 0, 0)
    assert got[0][0] == tuple(float(v) for v in outer.point(*inner.point(10, 20)))
    assert got[0][0] == tuple(float(v) for v in compose_all([outer, inner]).point(10, 20))


@pytest.mark.parametrize("source,kind", [
    ("ENDX\nHALT", FaultKind.UNMATCHED_ENDREP),
    ("XFORM 1 0 0\nMOVE 1 1\nLINE 2 2\nHALT", FaultKind.UNTERMINATED_REPEAT),
    ("XFORM 1 0 0\n" * (MAX_REPEAT_DEPTH + 1) + "MOVE 1 1\nLINE 2 2\nHALT",
     FaultKind.DEPTH_OVERFLOW),
    ("REPEATX 0 1 1 1\nLINE 5 5\nENDREP\nHALT", FaultKind.ZERO_REPEAT),
])
def test_structural_faults_are_reported_not_raised(source, kind):
    """Shared with the repeat machinery on purpose: an unclosed scope and a
    stray closer are the same structural failures, and every fault name is
    mirrored in `port/include/dm_isa.h`, which does not carry this tier yet."""
    assert kind in {f.kind for f in VM().run(assemble(source)).faults}


def test_an_out_of_range_transform_code_cannot_be_assembled():
    """Authoring is strict where execution is permissive. `VM.run` masks a
    transform byte to three bits so a generated program stays total; an
    assembler that did the same would let a corpus generator write `XFORM 9`
    and silently mean `XFORM 1`."""
    with pytest.raises(ISAError):
        assemble("XFORM 9 0 0\nENDX\nHALT")
    assert assemble("XFORM 7 0 0\nENDX\nHALT")


# --------------------------------------------------------------------------
# the ceiling


def test_the_saving_is_exact_and_known_by_construction():
    """`REPEATX` costs 5 bytes plus `ENDREP`, so `n` copies of a `b`-byte body
    cost `b + 6` against `n·b` flat. **That is the ceiling every claim-2-shaped
    result needs on day one** -- the compression available is arithmetic, not an
    estimate, which is the property `docs/direction.md` §6 selects proposals on.
    """
    quarter = D4(1, False).code
    body = "MOVE 10 20\nLINE 40 20\nLINE 40 60"          # 9 bytes
    structured = assemble(f"REPEATX 4 {quarter} 0 0\n{body}\nENDREP\nHALT")
    step = Transform.of(quarter, 0, 0)

    lines = []
    for k in range(4):
        pts = [step.power(k).point(x, y)
               for x, y in ((10, 20), (40, 20), (40, 60))]
        lines.append("MOVE {} {}\nLINE {} {}\nLINE {} {}".format(
            *(v for p in pts for v in p)))
    flat = assemble("\n".join(lines) + "\nHALT")

    assert len(structured) == 9 + 6 + 1          # body + REPEATX + ENDREP + HALT
    assert len(flat) == 4 * 9 + 1
    assert len(flat) - len(structured) == 21
    # And the two draw the same picture, which is what makes it a *saving*.
    assert [s.points for s in VM().run(structured).strokes] == [
        s.points for s in VM().run(flat).strokes]
