import random

import numpy as np
import pytest

from dm.data import synthetic
from dm.isa.asm import assemble, disassemble, parse
from dm.isa.codec import (
    CODECS,
    N_SPECIAL,
    BitCodec,
    ByteCodec,
    TokenCodec,
    opcode_mask,
)
from dm.isa.spec import (
    CANVAS,
    ISAError,
    N_OPCODES,
    OPCODE_SLOTS,
    Kind,
    Op,
    Tier,
    decode_operand,
    encode_operand,
)
from dm.vm.interp import VM, FaultKind
from dm.vm.render import rasterize, to_svg

CORPUS = synthetic.dataset(200, seed=0)


def test_delta_operands_roundtrip():
    for v in (-128, -1, 0, 1, 127):
        assert decode_operand(Kind.DELTA, encode_operand(Kind.DELTA, v)) == v


def test_assemble_disassemble_roundtrip():
    for program in CORPUS:
        assert assemble(disassemble(program)) == program


@pytest.mark.parametrize("codec", list(CODECS.values()), ids=lambda c: c.name)
def test_codec_roundtrip(codec):
    for program in CORPUS:
        assert codec.decode(codec.encode(program)) == program


def test_sequence_lengths_scale_as_expected():
    program = CORPUS[0]
    n = len(program)
    assert len(ByteCodec().encode(program)) == n
    assert len(TokenCodec().encode(program)) == n
    assert len(BitCodec().encode(program)) == 8 * n


def test_vocab_sizes():
    """ISA v2, 2026-08-11. `token` moved 269 -> 274 when the opcode region was
    reserved at 16 slots, and `token_typed` 1293 -> 1554 because `Kind.XF` adds
    a whole 256-value alphabet. `byte` and `bit` do not move at any table size,
    which is why no claim-2, claim-3 or claim-4 number did either."""
    assert ByteCodec().vocab_size == 258
    assert TokenCodec().vocab_size == 274
    assert TokenCodec(typed_operands=True).vocab_size == 1554
    assert BitCodec().vocab_size == 4


def test_the_pre_v2_token_layout_can_still_be_reconstructed():
    """37 token-arm checkpoints predate the reserved region, and a checkpoint is
    rebuilt from the vocabulary in its blob while the codec is rebuilt from
    today's table. Without an exact reconstruction those checkpoints would decode
    their own generated symbols to *different opcodes*, silently."""
    legacy, typed = TokenCodec(opcode_slots=11), TokenCodec(True, opcode_slots=11,
                                                            n_kinds=5)
    assert (legacy.vocab_size, typed.vocab_size) == (269, 1293)
    for program in CORPUS:
        assert legacy.decode(legacy.encode(program)) == program
        assert typed.decode(typed.encode(program)) == program


def test_a_legacy_codec_refuses_an_opcode_added_after_it():
    """Refused rather than wrapped. Aliasing a v2 opcode onto a v1 symbol is
    exactly the failure the reserved region exists to prevent, and it would be
    invisible -- the stream would decode to a valid, different program."""
    with pytest.raises(ISAError):
        TokenCodec(opcode_slots=11).encode(assemble("XFORM 3 10 10\nENDX\nHALT"))


def test_the_reserved_slots_are_not_bytecode_bytes():
    """A model can emit a symbol in a slot no opcode occupies yet. It has to
    mean "nothing", the way PAD does, and not the last opcode in the table."""
    codec = TokenCodec()
    spare = N_SPECIAL + N_OPCODES          # the first unused slot
    assert spare < N_SPECIAL + OPCODE_SLOTS
    assert codec.decode([spare]) == b""
    chunk = np.array([[spare]], dtype=np.int16)
    assert codec.symbols_to_bytes(chunk)[0] == -1


def test_opcode_mask_matches_instruction_starts():
    for program in CORPUS:
        mask = opcode_mask(program)
        expected, pc = [False] * len(program), 0
        for instr in parse(program):
            expected[pc] = True
            pc += 1 + len(instr.args)
        assert mask == expected


def test_synthetic_corpus_is_valid_and_nonempty():
    vm = VM()
    for program in CORPUS:
        trace = vm.run(program)
        assert trace.valid, trace.faults
        assert not trace.is_empty


def test_repeat_expands_and_restores_offset():
    body = "MOVE 10 10\nLINE 20 10\nENDREP\nMOVE 10 10\nLINE 20 10\nHALT"
    looped = VM().run(assemble("REPEAT 3 30 0\n" + body))
    assert looped.valid
    # three offset copies plus the trailing unoffset stroke
    assert len(looped.strokes) == 4
    assert [s.points[0][0] for s in looped.strokes] == [10, 40, 70, 10]


def test_nested_repeat_composes_offsets():
    trace = VM().run(
        assemble(
            "REPEAT 2 40 0\nREPEAT 2 0 40\nMOVE 10 10\nLINE 20 10\nENDREP\nENDREP\nHALT"
        )
    )
    assert trace.valid
    origins = [s.points[0] for s in trace.strokes]
    assert origins == [(10, 10), (10, 50), (50, 10), (50, 50)]


@pytest.mark.parametrize(
    "src,kind",
    [
        ("ENDREP\nHALT", FaultKind.UNMATCHED_ENDREP),
        ("REPEAT 2 1 1\nLINE 5 5\nHALT", FaultKind.UNTERMINATED_REPEAT),
        ("REPEAT 0 1 1\nLINE 5 5\nENDREP\nHALT", FaultKind.ZERO_REPEAT),
        ("MOVE 1 1\nLINE 2 2", FaultKind.NO_HALT),
    ],
)
def test_structural_faults_are_reported_not_raised(src, kind):
    trace = VM().run(assemble(src))
    assert kind in {f.kind for f in trace.faults}


def test_unknown_opcode_and_truncation():
    assert VM().run(bytes([0xEE])).faults[0].kind is FaultKind.UNKNOWN_OPCODE
    assert VM().run(bytes([int(Op.LINE), 5])).faults[0].kind is FaultKind.TRUNCATED


def test_repeat_depth_overflow_is_bounded():
    src = "REPEAT 2 1 1\n" * 8 + "LINE 5 5\n" + "ENDREP\n" * 8 + "HALT"
    trace = VM().run(assemble(src))
    assert FaultKind.DEPTH_OVERFLOW in {f.kind for f in trace.faults}


def test_random_bytes_never_crash_the_vm():
    """Generated streams will be garbage early in training; the VM is the one
    place that has to survive it."""
    rng, vm = random.Random(1), VM(fuel=5000)
    for _ in range(500):
        program = bytes(rng.randrange(256) for _ in range(rng.randint(0, 64)))
        trace = vm.run(program)
        assert isinstance(trace.valid, bool)


def test_l0_tier_emits_no_control_flow():
    corpus = synthetic.dataset(100, seed=3, tier=Tier.L0)
    control = {int(Op.REPEAT), int(Op.ENDREP)}
    for program in corpus:
        starts = opcode_mask(program)
        assert not any(b in control for b, is_op in zip(program, starts) if is_op)


def test_splits_are_deduplicated_and_disjoint():
    """Tier A once leaked 7.0% of val into train and repeated 1,167 train
    samples, which biased bits/drawing -- the headline metric -- by an unknown
    amount in the optimistic direction."""
    train, val = synthetic.split(2_000, 200, seed=0)
    assert len(set(train)) == len(train)
    assert len(set(val)) == len(val)
    assert not set(val) & set(train)


def test_random_grid_does_not_collapse_onto_its_repeat_count():
    """Every field used to be derived from `count`, leaving three distinct
    programs for a fixed box -- the source of the leak above."""
    rng = random.Random(0)
    grids = {synthetic.random_grid(rng, (16, 16, 239, 239)) for _ in range(1_000)}
    assert len(grids) > 900


def test_repeat_delta_stays_inside_the_i8_operand():
    """REPEAT's dx/dy are i8. `step` used to stay under 127 only as a side effect
    of the canvas size and the minimum repeat count, with nothing pinning it."""
    rng = random.Random(0)
    for _ in range(2_000):
        x1, y1 = rng.randint(8, CANVAS - 1), rng.randint(8, CANVAS - 1)
        head = synthetic.random_grid(rng, (0, 0, x1, y1)).splitlines()[0].split()
        assert all(-128 <= int(v) <= 127 for v in head[2:4]), head


def test_render_produces_ink():
    trace = VM().run(assemble("WIDTH 3\nMOVE 20 20\nLINE 200 200\nHALT"))
    ink = rasterize(trace)
    assert ink.shape == (256, 256)
    assert 0.0 < ink.mean() < 1.0
    assert to_svg(trace).startswith("<svg")


def test_svg_layers_compose_without_hiding_each_other():
    """The contract a provenance sheet rests on: several traces of one program
    drawn in one frame, so a reader can see which of its bytes are the planted
    orbit (`scripts/figures.py`). A layer that painted its own background would
    hide every layer under it, and one that ignored `ink` would make every layer
    the same colour -- both fail silently as a picture that looks fine."""
    trace = VM().run(assemble("MOVE 20 20\nLINE 200 200\nMOVE 128 128\nCIRCLE 40\nHALT"))
    default = to_svg(trace)
    assert 'fill="white"' in default and 'stroke="black"' in default

    layer = to_svg(trace, ink="#c8654a", background=None)
    assert 'fill="white"' not in layer
    assert layer.count("#c8654a") >= 1 and 'stroke="black"' not in layer
    # Same geometry either way: `ink` is a colour, never a transform.
    assert layer.count("<polyline") == default.count("<polyline")
    assert layer.count("<circle") == default.count("<circle")
