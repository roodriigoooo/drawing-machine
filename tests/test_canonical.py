"""Canonicalisation, and the null that has to fire exactly.

`dm/data/canonical.py` argues that an **exact** re-framing cannot change either
within-program ceiling, because it carries the set of foldable repeats across
bijectively. That is a theorem, so the test for it is not "the number moved a
little" -- it is byte equality on a real corpus. If it ever fails, the policy is
not in D4 ⋉ integer translation and no other number measured under it is
readable.

The other half is that `d4` is a genuine choice function on the orbit: two
drawings related by a symmetry have to land on the *same* representative, or it
re-frames without canonicalising anything.
"""

import numpy as np
import pytest

from dm.data.augment import Affine, apply
from dm.data.canonical import (
    POLICIES,
    d4_representative,
    principal_frame,
    reframe,
)
from dm.eval.library import library_stats
from dm.eval.repeats import corpus_stats, symmetry_stats
from dm.isa.asm import assemble
from dm.isa.spec import CANVAS


def _icons(limit: int = 60) -> list[bytes]:
    tabler = pytest.importorskip("dm.data.tabler")
    programs = tabler.split()[0][:limit]
    if not programs:
        pytest.skip("Tabler icons are not on this machine")
    return programs


def _rosette() -> bytes:
    """Four arms around a centre: a shape whose D4 orbit is non-trivial."""
    lines = ["MOVE 128 128", "LINE 128 40", "LINE 140 52", "MOVE 128 128",
             "LINE 216 128", "LINE 204 140", "HALT"]
    return assemble("\n".join(lines))


def test_the_representative_is_the_same_for_every_member_of_the_orbit():
    """The property that makes it a canonicalisation rather than a rotation."""
    program = _rosette()
    target = d4_representative(program)[0]
    for code in range(8):
        image = apply(program, Affine(0, 0, bool(code & 1), (code >> 1) & 3))
        assert image is not None
        assert d4_representative(image)[0] == target


def test_the_representative_is_idempotent():
    program = _rosette()
    once = d4_representative(program)[0]
    assert d4_representative(once)[0] == once


def test_an_exact_policy_leaves_both_within_program_ceilings_byte_identical():
    """The provable null. Not "close": equal.

    `docs/direction.md` §7.1 proposes measuring canonicalisation by asking
    whether the repeat ceiling went up. It cannot: a single element of
    D4 ⋉ integer translation maps a repeat at step `t` to a repeat at step
    `L(t)`, and `L` is a signed axis permutation, so the fold survives and stays
    inside `i8`. The measurement is therefore a theorem check, and its value is
    that a policy which is *not* exact fails it loudly.
    """
    programs = _icons()
    canon = [d4_representative(p)[0] for p in programs]
    assert canon != programs                        # the policy did something

    assert corpus_stats(canon)["saved"] == corpus_stats(programs)["saved"]
    assert symmetry_stats(canon)["saved"] == symmetry_stats(programs)["saved"]


@pytest.mark.parametrize("limit", [60, 120, 300])
def test_only_the_translation_normalised_library_column_can_move(limit):
    """The oracle has already quotiented by the group the policy applies, so the
    `d4` column cannot move -- and a policy that moved *it* would have moved an
    artifact. The translation-normalised column is the only place a re-framing
    can show up at all.

    **Its sign is deliberately not asserted, because it is not monotone.** The
    representative is chosen from the *whole* drawing, so two icons that shared
    a body can be sent to different frames and stop sharing it: at 300 icons the
    column rises (17.92% -> 19.58%) and at 120 it falls. A canonicalisation
    aligns drawings that are globally alike and separates ones that are alike
    only in a part, and a test that pinned a direction would be pinning which of
    those a sample happened to contain.
    """
    programs = _icons(limit)
    canon = [d4_representative(p)[0] for p in programs]
    assert library_stats(canon, d4=True)["saved"] == library_stats(programs, d4=True)["saved"]
    assert library_stats(canon)["saved"] != library_stats(programs)["saved"]


def test_the_principal_frame_is_a_rotation_and_not_a_reflection():
    """A reflection here would fold mirror pairs together silently, which is
    `d4`'s job and is exact; hiding it inside a float rotation would put an
    exact canonicalisation inside an inexact one and confuse the two results."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        strokes = [rng.uniform(0.0, 255.0, size=(12, 2)) for _ in range(3)]
        basis = principal_frame(strokes)
        assert np.isclose(np.linalg.det(basis), 1.0)
        assert np.allclose(basis @ basis.T, np.eye(2), atol=1e-9)


def test_the_principal_frame_undoes_a_rotation_of_the_same_drawing():
    """Two tilts of one drawing have to reach the same frame, or canonicalising
    orientation does nothing for the similarity it exists to create."""
    rng = np.random.default_rng(1)
    base = [rng.uniform(-60.0, 60.0, size=(30, 2)) for _ in range(2)]
    angle = 0.7
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    tilted = [s @ rot.T for s in base]
    a = reframe(base, POLICIES["rot"])
    b = reframe(tilted, POLICIES["rot"])
    assert np.allclose(np.concatenate(a), np.concatenate(b), atol=1e-6)


def test_every_frame_policy_keeps_the_drawing_on_the_canvas():
    rng = np.random.default_rng(2)
    strokes = [rng.uniform(0.0, 255.0, size=(20, 2)) for _ in range(4)]
    for name in ("rot", "aspect", "rot_aspect"):
        out = np.concatenate(reframe(strokes, POLICIES[name]))
        assert out.min() >= -1e-9 and out.max() <= CANVAS - 1 + 1e-9


def test_the_policy_table_says_which_policies_are_exact():
    """The flag is what decides whether a number under a policy is a compression
    result or a description of rounding damage, so it is asserted rather than
    left to the reader."""
    assert POLICIES["none"].exact and POLICIES["d4"].exact
    assert not any(POLICIES[n].exact for n in ("rot", "aspect", "rot_aspect"))
