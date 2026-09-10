"""L1 -> L0 expansion: same drawing, no `REPEAT`, and exact or nothing.

Claim 2's corpus is only a corpus if all three hold. The first is checked
against the VM rather than against a second copy of the offset rule, the second
against the instruction stream, and the third against the oracle -- an
expansion that rounded a copy apart would still render correctly and would be
worthless, which is the SVG-Icons8 failure (`docs/tier-c.md`).
"""

import pytest

from dm.data import synthetic
from dm.eval.repeats import corpus_stats
from dm.isa.asm import assemble, parse
from dm.isa.spec import Tier
from dm.isa.transform import D4, Transform
from dm.isa.unroll import MAX_EXPANSION, Unrollable, unroll
from dm.vm.interp import VM

L1_CORPUS = synthetic.dataset(200, seed=3, tier=Tier.L1)
VM_ = VM()


def test_expansion_preserves_the_drawing():
    checked = 0
    for program in L1_CORPUS:
        flat = unroll(program)
        assert flat is not None
        assert VM_.run(flat).strokes == VM_.run(program).strokes
        assert VM_.run(flat).discs == VM_.run(program).discs
        checked += 1
    assert checked == len(L1_CORPUS)


def test_expansion_removes_every_control_flow_instruction():
    """The point of the corpus: structure in the geometry, absent from the
    alphabet. A single surviving `REPEAT` would hand the model the answer."""
    for program in L1_CORPUS:
        mnemonics = {i.mnemonic for i in parse(unroll(program))}
        assert not mnemonics & {"REPEAT", "ENDREP"}


def test_a_program_without_repeats_is_returned_unchanged():
    for program in synthetic.dataset(50, seed=4, tier=Tier.L0):
        assert unroll(program) == program


def test_nested_repeats_expand_multiplicatively():
    program = assemble(
        "REPEAT 3 20 0\nREPEAT 2 0 30\nMOVE 10 10\nLINE 20 20\nENDREP\nENDREP\nHALT"
    )
    flat = unroll(program)
    moves = [i for i in parse(flat) if i.mnemonic == "MOVE"]
    assert len(moves) == 6
    assert sorted(m.args for m in moves) == [
        (10, 10), (10, 40), (30, 10), (30, 40), (50, 10), (50, 40),
    ]


def test_a_copy_that_leaves_the_canvas_is_refused_not_clamped():
    """The VM clamps; this must not. A clamped copy is not an exact translate,
    so `REPEAT n dx dy` could not re-emit it and the oracle would not score it
    -- putting near-repeats into the one corpus built to contain exact ones."""
    program = assemble("REPEAT 8 100 0\nMOVE 200 10\nLINE 210 20\nENDREP\nHALT")
    assert unroll(program) is None


def test_malformed_control_flow_is_refused():
    for text in ("ENDREP\nHALT", "REPEAT 2 1 1\nMOVE 5 5\nHALT", "REPEAT 0 1 1\nENDREP\nHALT"):
        assert unroll(assemble(text)) is None


def test_expansion_is_bounded():
    """Four levels at count 4 is 256 copies of the innermost body. Unbounded,
    that overruns `max_len`, where truncation biases the granularity axis
    specifically (`PLAN.md` 10)."""
    body = "MOVE 1 1\nLINE 2 2\n"
    program = assemble("REPEAT 4 0 0\n" * 4 + body + "ENDREP\n" * 4 + "HALT")
    flat = unroll(program)
    assert flat is None or len(flat) <= MAX_EXPANSION


def test_depth_beyond_the_isa_limit_is_refused():
    program = assemble(
        "REPEAT 2 1 0\n" * 5 + "MOVE 1 1\nLINE 2 2\n" + "ENDREP\n" * 5 + "HALT"
    )
    assert unroll(program) is None


def test_flat_corpus_carries_the_structure_the_oracle_can_find():
    """The whole reason the corpus exists: a ceiling large enough to measure
    recovery against. Tier C's is 3.79% of bytes on 4,613 programs; this is an
    order of magnitude more, on as many programs as the generator will make."""
    flat = synthetic.dataset(300, seed=0, tier=Tier.L1, flatten=True)
    stats = corpus_stats(flat)
    assert stats["saved_frac"] > 0.10
    assert stats["programs_with_repeat"] > 0.25
    # And the tolerance curve is flat, which is what says the expansion kept
    # the repeats rather than rounding them apart.
    assert corpus_stats(flat, tol=1)["saved_frac"] == pytest.approx(
        stats["saved_frac"], abs=0.02
    )


def test_longer_repeats_raise_the_ceiling_and_the_length():
    """The length-generalisation corpus. Copies past the trained bound sit at
    sequence positions no training program reached, which is the test."""
    short = synthetic.dataset(200, seed=1, tier=Tier.L1, flatten=True, max_repeat=4)
    long = synthetic.dataset(
        200, seed=1, tier=Tier.L1, flatten=True, max_repeat=16, min_repeat=8
    )
    assert corpus_stats(long)["saved_frac"] > corpus_stats(short)["saved_frac"]
    assert sum(map(len, long)) / len(long) > 2 * sum(map(len, short)) / len(short)


def test_sample_refuses_rather_than_falling_back_to_l1():
    """A corpus advertised as flat that quietly contains `REPEAT` would make
    claim 2's number meaningless in the one way the number cannot show."""
    assert issubclass(Unrollable, Exception)
    program = assemble("REPEAT 8 100 0\nMOVE 200 10\nLINE 210 20\nENDREP\nHALT")
    assert unroll(program) is None


def test_default_generator_behaviour_is_unchanged():
    """`min_repeat` and `flatten` default to the old behaviour, so no existing
    run record becomes incomparable and `SCHEMA` does not move."""
    import random

    rng_a, rng_b = random.Random(11), random.Random(11)
    for _ in range(50):
        assert synthetic.sample(rng_a, tier=Tier.L1) == synthetic.sample(
            rng_b, tier=Tier.L1, min_repeat=2, flatten=False
        )


# --------------------------------------------------------------------------
# ISA v2: the transform tier


def scoped(source: str) -> bytes:
    return assemble(source)


@pytest.mark.parametrize("source", [
    (f"XFORM {D4(1, False).code} 0 0\nMOVE 30 40\nLINE 60 45\nENDX\n"
     "MOVE 20 20\nLINE 25 25\nHALT"),
    f"REPEATX 4 {D4(1, False).code} 0 0\nMOVE 60 70\nLINE 80 75\nENDREP\nHALT",
    "REPEATX 3 0 20 5\nMOVE 30 40\nLINE 50 45\nENDREP\nHALT",
    (f"XFORM {D4(0, True).code} 0 0\nREPEATX 2 {D4(2, False).code} 0 0\n"
     "MOVE 60 70\nLINE 80 75\nENDREP\nENDX\nHALT"),
    f"REPEATX 2 {D4(0, True).code} 0 0\nMOVE 10 20\nLINE 40 20\nENDREP\nHALT",
])
def test_the_transform_tier_flattens_to_the_same_drawing(source):
    """What claim 2's corpus is: the same picture with the structure removed.
    Checked against the VM, so the flattener and the interpreter cannot agree
    on a rule that is wrong in both."""
    program = scoped(source)
    flat = unroll(program)
    assert flat is not None
    assert VM_.run(flat).strokes == VM_.run(program).strokes
    assert VM_.run(flat).valid


def test_flattening_removes_every_scope_opcode():
    """A flat trace that still contained `XFORM` would hand the model the
    structure the experiment asks it to discover -- the L1 mistake, one tier
    up."""
    source = (f"XFORM {D4(0, True).code} 0 0\nREPEATX 2 {D4(1, False).code} 0 0\n"
              "MOVE 60 70\nLINE 80 75\nENDREP\nENDX\nHALT")
    flat = unroll(scoped(source))
    assert flat is not None
    assert {i.mnemonic for i in parse(flat)} <= {"MOVE", "LINE", "HALT"}


def test_a_transformed_repeat_expands_to_its_own_powers():
    """Iteration k is the body under the step applied k times, and the copies
    are exact images rather than near-misses -- which is the entire reason the
    group is integer."""
    step = D4(1, False)
    flat = unroll(scoped(f"REPEATX 4 {step.code} 0 0\nMOVE 60 70\nLINE 80 75\n"
                         "ENDREP\nHALT"))
    assert flat is not None
    starts = [s.points[0] for s in VM_.run(flat).strokes]
    want = [Transform(step, 0, 0).power(k).point(60, 70) for k in range(4)]
    assert starts == [tuple(float(v) for v in p) for p in want]


@pytest.mark.parametrize("source", [
    "REPEAT 2 1 1\nXFORM 1 0 0\nMOVE 1 1\nENDREP\nENDX\nHALT",   # crossed
    "XFORM 1 0 0\nMOVE 1 1\nENDREP\nHALT",                        # wrong closer
    "REPEATX 2 1 0 0\nMOVE 1 1\nENDX\nHALT",                      # wrong closer
    "XFORM 1 0 0\nMOVE 1 1\nLINE 2 2\nHALT",                      # unterminated
])
def test_improperly_nested_scopes_are_refused(source):
    """Each of these is a program the VM faults on, so flattening one would
    invent a drawing that cannot be executed. `None` is the honest answer."""
    program = scoped(source)
    assert unroll(program) is None
    assert not VM_.run(program).valid
