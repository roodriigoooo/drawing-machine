"""The state layer, pinned against the VM and against the frozen truth table.

`docs/state-freeze.md` says its §1 truth table and its §2/§4 coverage tables are
*measured* and reproducible here rather than asserted in prose. This file is that
promise: if the VM's scope bookkeeping moves, or a corpus generator's opcode mix
changes, these fail instead of the language quietly meaning something else.

It also holds the **conformance fixture** of the freeze's §4.1 -- exhaustive short
programs over the whole opcode table. It is a *test* object: it is never used as
training data, as an evaluation split, or to tune a mask.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch

from dm.data import composed, synthetic
from dm.isa import state as S
from dm.isa.asm import assemble
from dm.isa.codec import BOS, CODECS, PAD
from dm.isa.spec import ISAError, Kind, Op
from dm.isa.unroll import unroll
from dm.models.transformer import Config, DrawingLM
from dm.vm.interp import VM, FaultKind

#: Every opcode the canonical language can contain: the ISA minus `CALL`, which
#: `VM.run` refuses, and minus reserved `COLOR`.
FULL = S.LanguagePolicy.of(
    ["HALT", "MOVE", "LINE", "CURVE", "CIRCLE", "WIDTH", "FILL",
     "REPEAT", "ENDREP", "XFORM", "ENDX", "REPEATX"],
    label="conformance",
)

#: Faults that are *not* structural. Fuel is a resource bound: a perfectly
#: canonical program with nested counts can exhaust it, which is
#: `docs/state-freeze.md` §1.2's first amendment.
RESOURCE_FAULTS = {FaultKind.OUT_OF_FUEL}


def structural(trace) -> set[FaultKind]:
    return {f.kind for f in trace.faults} - RESOURCE_FAULTS


# ---------------------------------------------------------------------------
# the frozen truth table


#: One row per line of `docs/state-freeze.md` §1: the program, whether the
#: canonical predicate admits it, and which structural faults the VM reports.
TRUTH_TABLE: tuple[tuple[str, bytes, bool, set[FaultKind]], ...] = (
    ("standalone xform", assemble("XFORM 3 10 10\nMOVE 10 10\nLINE 20 20\nENDX\nHALT"),
     True, set()),
    ("repeatx", bytes([0x0D, 2, 2, 0, 0, 0x01, 10, 10, 0x02, 20, 20, 0x08, 0x00]),
     True, set()),
    ("xf=8 on xform", bytes([0x0B, 8, 0, 0, 0x01, 10, 10, 0x02, 20, 20, 0x0C, 0x00]),
     False, set()),
    ("xf=8 on repeatx",
     bytes([0x0D, 2, 8, 0, 0, 0x01, 10, 10, 0x02, 20, 20, 0x08, 0x00]), False, set()),
    ("reserved color", bytes([0x0A, 3, 0x01, 10, 10, 0x02, 20, 20, 0x00]), False, set()),
    ("crossing at count one",
     bytes([0x07, 1, 0, 0, 0x0B, 2, 0, 0, 0x01, 10, 10, 0x02, 20, 20, 0x08, 0x0C, 0x00]),
     False, set()),
    ("crossing at count two",
     bytes([0x07, 2, 4, 0, 0x0B, 2, 0, 0, 0x01, 10, 10, 0x02, 20, 20, 0x08, 0x0C, 0x00]),
     False, {FaultKind.UNTERMINATED_REPEAT}),
    ("crossing under the floor",
     bytes([0x0B, 2, 0, 0, 0x07, 1, 0, 0, 0x01, 10, 10, 0x02, 20, 20, 0x0C, 0x08, 0x00]),
     False, set()),
    ("call", bytes([0x09, 3, 0x00]), False, {FaultKind.CALL_UNSUPPORTED}),
    ("endrep closes xform",
     bytes([0x0B, 2, 0, 0, 0x01, 10, 10, 0x02, 20, 20, 0x08, 0x00]),
     False, {FaultKind.UNMATCHED_ENDREP}),
    ("endx closes repeat",
     bytes([0x07, 2, 4, 0, 0x01, 10, 10, 0x02, 20, 20, 0x0C, 0x00]),
     False, {FaultKind.UNMATCHED_ENDREP}),
    ("endx closes repeatx's own slot",
     bytes([0x0D, 1, 2, 0, 0, 0x01, 10, 10, 0x02, 20, 20, 0x0C, 0x08, 0x00]),
     False, {FaultKind.UNMATCHED_ENDREP}),
    ("halt inside a repeat", bytes([0x07, 2, 4, 0, 0x01, 10, 10, 0x00]),
     False, {FaultKind.UNTERMINATED_REPEAT}),
    ("halt inside a transform", bytes([0x0B, 2, 0, 0, 0x01, 10, 10, 0x00]),
     False, {FaultKind.UNTERMINATED_REPEAT}),
    ("zero count", bytes([0x07, 0, 4, 0, 0x01, 10, 10, 0x02, 20, 20, 0x08, 0x00]),
     False, {FaultKind.ZERO_REPEAT}),
    ("zero count on repeatx",
     bytes([0x0D, 0, 2, 0, 0, 0x01, 10, 10, 0x02, 20, 20, 0x08, 0x00]),
     False, {FaultKind.ZERO_REPEAT}),
    ("five nested repeats",
     bytes([0x07, 2, 1, 0] * 5 + [0x02, 5, 5] + [0x08] * 5 + [0x00]),
     False, {FaultKind.DEPTH_OVERFLOW}),
    ("five nested transforms",
     bytes([0x0B, 2, 0, 0] * 5 + [0x01, 5, 5, 0x02, 9, 9] + [0x0C] * 5 + [0x00]),
     False, {FaultKind.DEPTH_OVERFLOW}),
    ("repeatx x4 plus a transform",
     bytes([0x0D, 2, 2, 0, 0] * 4 + [0x0B, 2, 0, 0] + [0x01, 5, 5, 0x02, 9, 9]
           + [0x0C] + [0x08] * 4 + [0x00]),
     False, {FaultKind.DEPTH_OVERFLOW}),
    ("width zero", bytes([0x05, 0, 0x01, 10, 10, 0x02, 20, 20, 0x00]), True, set()),
    ("not an opcode", bytes([0x40, 0x00]), False, {FaultKind.UNKNOWN_OPCODE}),
    ("truncated", bytes([0x01, 10]), False, {FaultKind.TRUNCATED}),
    ("no halt", bytes([0x01, 10, 10, 0x02, 20, 20]), False, {FaultKind.NO_HALT}),
    ("trailing bytes after halt",
     bytes([0x01, 10, 10, 0x02, 20, 20, 0x00, 0x01, 5, 5]), False, set()),
    ("dangling xform in a repeatx body",
     bytes([0x0D, 1, 2, 0, 0, 0x0B, 2, 0, 0, 0x01, 10, 10, 0x02, 20, 20, 0x08, 0x00]),
     False, set()),
    ("dangling xform in a repeat body",
     bytes([0x07, 1, 0, 0, 0x0B, 2, 0, 0, 0x01, 10, 10, 0x02, 20, 20, 0x08, 0x00]),
     False, {FaultKind.UNTERMINATED_REPEAT}),
)


@pytest.mark.parametrize("name,program,canonical,faults",
                         TRUTH_TABLE, ids=[row[0] for row in TRUTH_TABLE])
def test_the_frozen_truth_table_holds(name, program, canonical, faults):
    """`docs/state-freeze.md` §1, row by row, against both predicates at once.

    The rows where the two disagree are the point of the table: the VM accepts a
    raw `XF = 8`, a reserved `COLOR`, a count-one crossing and its mirror image,
    and the paper still calls all four non-canonical. A change that made either
    column agree with the other would silently redefine what the mask constrains.
    """
    verdict = S.trace(program, FULL)
    assert verdict.canonical is canonical, (name, verdict.violations)
    assert structural(VM().run(program)) == faults, name


def test_the_two_unroll_defects_are_still_there():
    """`docs/state-freeze.md` §1.1: `unroll` raises on `XF > 7` where it
    contracts to return None, and accepts `CALL` where the VM refuses it.

    Pinned rather than fixed, so that a repair is a repair and not a silent
    redefinition of analysis acceptance. Both are unreachable from authoring and
    reachable from generated bytes.
    """
    with pytest.raises(ISAError):
        unroll(bytes([0x0B, 8, 0, 0, 0x01, 10, 10, 0x0C, 0x00]))
    assert unroll(bytes([0x09, 3, 0x00])) is not None
    assert not VM().run(bytes([0x09, 3, 0x00])).valid


# ---------------------------------------------------------------------------
# the exhaustive conformance fixture


#: Instructions the fixture composes, chosen to cover every ISA opcode and every
#: branch of the state machine: all L0 instructions, both loop openers, both
#: closers, a standalone transform, a zero count, an out-of-domain `XF`, a
#: reserved opcode and `CALL`.
PROBE_INSTRUCTIONS: tuple[bytes, ...] = (
    bytes([0x00]),                  # HALT
    bytes([0x01, 1, 1]),            # MOVE
    bytes([0x02, 2, 2]),            # LINE
    bytes([0x03, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]),  # CURVE
    bytes([0x04, 3]),               # CIRCLE
    bytes([0x05, 2]),               # WIDTH
    bytes([0x06]),                  # FILL
    bytes([0x07, 2, 1, 1]),         # REPEAT 2
    bytes([0x07, 0, 1, 1]),         # REPEAT 0
    bytes([0x08]),                  # ENDREP
    bytes([0x0B, 2, 0, 0]),         # XFORM 2
    bytes([0x0B, 8, 0, 0]),         # XFORM 8 -- outside D4
    bytes([0x0C]),                  # ENDX
    bytes([0x0D, 2, 2, 0, 0]),      # REPEATX 2
    bytes([0x0A, 0]),               # COLOR -- reserved
    bytes([0x09, 0]),               # CALL -- unsupported
)


def probe_programs(max_instructions: int = 4):
    for length in range(1, max_instructions + 1):
        for combo in itertools.product(PROBE_INSTRUCTIONS, repeat=length):
            yield b"".join(combo)


PROBE = tuple(probe_programs())


def test_the_probe_fixture_is_the_size_the_freeze_says():
    """16 instructions, every sequence up to length four. Stated so a change to
    the fixture is a decision rather than a silent narrowing of coverage."""
    assert len(PROBE) == sum(16 ** k for k in range(1, 5)) == 69_904


def test_the_probe_fixture_contains_every_isa_opcode():
    seen = {int(program[0]) for program in PROBE_INSTRUCTIONS}
    assert seen == {int(op) for op in Op}


def test_canonical_membership_implies_the_vm_accepts():
    """The containment the paper's two outcome columns rest on.

    Exhaustive over the fixture: a program the canonical predicate admits has no
    structural VM fault. The converse is deliberately false -- that is what the
    truth table's disagreements are -- and fuel is excluded because it is a
    resource bound and not a structural one.
    """
    vm = VM()
    for program in PROBE:
        if S.trace(program, FULL).canonical:
            assert not structural(vm.run(program)), program.hex()


def test_a_terminal_violation_is_where_the_vm_stops():
    """A `terminal` violation means the parse stops, and it stops exactly where
    `VM.run` breaks: unknown opcode, truncation, a closer with nothing to close,
    a depth overflow, `CALL`."""
    vm = VM()
    for program in PROBE:
        verdict = S.trace(program, FULL)
        terminal = [v for v in verdict.violations if v.terminal]
        if terminal:
            assert structural(vm.run(program)), program.hex()


def test_state_labels_are_a_function_of_the_prefix_alone():
    """The prefix-only state-label invariant retained in `docs/state.md`.

    The snapshot at position `i` must be what stepping `program[:i]` produces --
    no target symbol may be consulted while producing its own label, or a
    state-conditioned model is being handed the answer.
    """
    for program in PROBE[:2_000]:
        verdict = S.trace(program, FULL)
        for i, snapshot in enumerate(verdict.snapshots):
            partial = S.GrammarState(FULL)
            for byte in program[:i]:
                partial.step(byte)
            assert partial.snapshot() == snapshot, (program.hex(), i)


def test_a_non_canonical_prefix_reaches_the_declared_absorbing_state():
    """The absorbing contract: once a prefix has left the canonical language it
    stays out, and its `status` says so in one field for every arm to read."""
    state = S.GrammarState(FULL)
    for byte in bytes([0x0B, 8, 0, 0]):        # XFORM with XF = 8
        state.step(byte)
    assert not state.canonical
    assert state.status is S.Status.CANONICAL_FAULT
    for byte in bytes([0x01, 10, 10]):         # a perfectly good MOVE
        state.step(byte)
    assert not state.canonical
    assert state.status is S.Status.CANONICAL_FAULT


# ---------------------------------------------------------------------------
# closer, halt and depth legality, checked against the executor


#: Whole instructions to try as a continuation, so the VM's verdict is about the
#: instruction and never about a missing operand. Every operand is inside its
#: canonical domain, so the only thing that can fault is the *structure*: a
#: closer with nothing to close, a halt inside a scope, an opener at its bound.
CANDIDATES: tuple[bytes, ...] = (
    bytes([int(Op.HALT)]),
    bytes([int(Op.ENDREP)]),
    bytes([int(Op.ENDX)]),
    bytes([int(Op.FILL)]),
    bytes([int(Op.MOVE), 5, 5]),
    bytes([int(Op.WIDTH), 2]),
    bytes([int(Op.REPEAT), 2, 1, 0]),
    bytes([int(Op.XFORM), 2, 0, 0]),
    bytes([int(Op.REPEATX), 2, 2, 0, 0]),
)

#: Programs whose prefixes visit the interesting scope states, including the two
#: shapes where the VM's two stacks disagree with a naive counter.
SCOPE_PROBES: tuple[bytes, ...] = (
    bytes([0x07, 2, 1, 0, 0x0B, 2, 0, 0, 0x01, 5, 5, 0x0C, 0x08, 0x00]),
    bytes([0x0B, 2, 0, 0, 0x07, 2, 1, 0, 0x01, 5, 5, 0x0C, 0x08, 0x00]),
    bytes([0x0D, 2, 2, 0, 0, 0x0B, 2, 0, 0, 0x01, 5, 5, 0x0C, 0x08, 0x00]),
    bytes([0x0D, 2, 2, 0, 0, 0x0D, 2, 2, 0, 0, 0x01, 5, 5, 0x08, 0x08, 0x00]),
    bytes([0x07, 2, 1, 0] * 4 + [0x01, 5, 5] + [0x08] * 4 + [0x00]),
    bytes([0x0B, 2, 0, 0] * 4 + [0x01, 5, 5] + [0x0C] * 4 + [0x00]),
)


def test_the_vm_safe_mask_is_exactly_the_instructions_that_do_not_fault_here():
    """The VM-safe predicate, checked against the executor.

    For every **canonical live prefix** ending at an instruction boundary, the
    mask admits an instruction's opcode if and only if the VM, run on
    `prefix + instruction`, reports no structural fault beyond the `NO_HALT` the
    unfinished prefix already had. That is the definition -- "does emitting this
    here make a fault inevitable" -- turned into a measurement, and it covers the
    closer, halt and depth rules in one pass.

    Canonical prefixes only, and deliberately: from a *non-canonical* prefix a
    loop can replay a body that faults at some earlier pc, which no
    source-prefix predicate can see. That limit is measured by
    `test_a_crossing_body_faults_on_replay`, not hidden here.
    """
    vm = VM()
    for program in SCOPE_PROBES:
        state = S.GrammarState(FULL)
        for offset, byte in enumerate(program):
            if state.phase is S.Phase.OPCODE and not state.done and state.canonical:
                legal = state.legal_bytes(S.VM_SAFE)
                for candidate in CANDIDATES:
                    faults = structural(vm.run(program[:offset] + candidate))
                    faulted = bool(faults - {FaultKind.NO_HALT})
                    assert bool(candidate[0] in legal) is not faulted, (
                        f"{program.hex()} @{offset} + {candidate.hex()}: {faults}"
                    )
            state.step(byte)


def test_a_crossing_body_faults_on_replay():
    """`docs/state-freeze.md` §1.2, amendment 2, as a measurement.

    The VM-safe mask is prefix-local. A crossing `ENDX` inside a `REPEAT` body is
    admitted -- at count one it really does not fault -- and at count two the
    *second* iteration re-executes it with nothing open and faults. Seeing that
    from the prefix needs the suspendable executor of S4, so the residual faults
    of the VM-safe cell are a reported result rather than an instrument failure.
    """
    crossing = bytes([0x0B, 2, 0, 0, 0x07, 2, 1, 0, 0x01, 5, 5, 0x0C, 0x08, 0x00])
    state = S.GrammarState(FULL)
    for byte in crossing[:11]:                 # up to the crossing ENDX
        state.step(byte)
    assert int(Op.ENDX) in state.legal_bytes(S.VM_SAFE)
    assert int(Op.ENDX) not in state.legal_bytes(S.CANONICAL)
    assert FaultKind.UNMATCHED_ENDREP in structural(VM().run(crossing))
    # At count one the same shape executes cleanly, which is why the VM-safe mask
    # cannot refuse it without calling a VM-accepted program a fault.
    at_one = bytes([0x0B, 2, 0, 0, 0x07, 1, 0, 0, 0x01, 5, 5, 0x0C, 0x08, 0x00])
    assert not structural(VM().run(at_one))


def test_endx_cannot_close_the_transform_a_repeatx_owns():
    """`ENDX` is legal only when the innermost open scope is a standalone
    `XFORM`. A `REPEATX` owns its slot -- `_Frame.xform_floor` in the VM -- so a
    body that tried to close it would leave the frame pointing past the stack."""
    state = S.GrammarState(FULL)
    for byte in bytes([0x0D, 2, 2, 0, 0]):
        state.step(byte)
    assert not state.endx_legal and not state.endx_legal_vm
    assert state.endrep_legal
    for byte in bytes([0x0B, 2, 0, 0]):        # a standalone XFORM inside the body
        state.step(byte)
    assert state.endx_legal and state.endx_legal_vm
    assert not state.endrep_legal              # the transform must close first
    assert state.endrep_legal_vm               # but the VM would pop the frame


def test_a_crossing_is_vm_safe_and_not_canonical():
    """The freeze's chosen policy, in one assertion: the VM-safe mask lets a
    crossing through because whether it faults depends on a repeat count, and the
    canonical mask refuses it because the language is properly nested."""
    state = S.GrammarState(FULL)
    for byte in bytes([0x07, 1, 0, 0, 0x0B, 2, 0, 0]):
        state.step(byte)
    assert int(Op.ENDREP) in state.legal_bytes(S.VM_SAFE)
    assert int(Op.ENDREP) not in state.legal_bytes(S.CANONICAL)


def test_operand_domains_are_exact():
    """`XF` is 0-7 canonically and any byte to the VM; `COUNT` is 1-255 to both,
    because zero is `ZERO_REPEAT`. Nothing else is narrowed: a coordinate a
    transform may clamp and a `WIDTH 0` the VM raises to one are legal."""
    state = S.GrammarState(FULL)
    state.step(int(Op.XFORM))
    assert state.expected_kind is Kind.XF
    assert set(state.legal_bytes(S.CANONICAL).tolist()) == set(range(8))
    assert len(state.legal_bytes(S.VM_SAFE)) == 256

    state = S.GrammarState(FULL)
    state.step(int(Op.REPEAT))
    assert state.expected_kind is Kind.COUNT
    for level in (S.CANONICAL, S.VM_SAFE):
        assert set(state.legal_bytes(level).tolist()) == set(range(1, 256))

    state = S.GrammarState(FULL)
    state.step(int(Op.WIDTH))
    assert 0 in state.legal_bytes(S.CANONICAL)


def test_a_zero_operand_is_not_a_halt():
    """`MOVE 0 0` contains the HALT byte twice, at operand positions. The mask
    must admit it and the monitor must not stop -- the same fault the parse-state
    halt monitor was built for, reintroduced one layer up."""
    state = S.GrammarState(FULL)
    state.step(int(Op.MOVE))
    assert state.phase is S.Phase.OPERAND
    assert 0x00 in state.legal_bytes(S.CANONICAL)
    monitor = S.StateMonitor(CODECS["byte"], 1, FULL)
    monitor.prime(CODECS["byte"].encode(assemble("MOVE 0 0")))
    assert not monitor.done[0]
    assert monitor.pending[0] == 0


def test_a_prompt_that_ends_mid_instruction_keeps_its_operand_state():
    """A partial instruction has grammar state and symbol state and has not
    executed. The mask at that point is the expected operand's domain, not the
    opcode set."""
    codec = CODECS["byte"]
    monitor = S.StateMonitor(codec, 1, FULL)
    monitor.prime(codec.encode(bytes([int(Op.REPEAT), 3])))   # count given, deltas owed
    assert monitor.pending[0] == 2
    support = monitor.support(0)
    assert len(support.legal) == 256          # a DELTA is every encoded i8
    assert not monitor.done[0]


def test_an_empty_legal_set_is_an_instrument_error():
    """A mask with nothing to offer must fail loudly. Falling back to the raw
    distribution would turn a structural claim into a best-effort sampler."""
    policy = S.LanguagePolicy.of(["HALT", "XFORM"], label="deliberately broken")
    monitor = S.StateMonitor(CODECS["byte"], 1, policy)
    monitor.prime(CODECS["byte"].encode(bytes([int(Op.XFORM), 0, 0, 0] * 4)))
    with pytest.raises(S.StateError, match="no legal canonical symbol"):
        monitor.allowed()


def test_a_policy_refuses_a_language_the_vm_cannot_execute():
    """`CALL` under `call_policy='unsupported'`, reserved `COLOR`, a missing
    `HALT` and a depth past the VM's own bound are all refused at construction:
    a policy is the definition of the language, so a contradiction in it is not
    a runtime surprise."""
    for opcodes in (["HALT", "CALL"], ["HALT", "COLOR"], ["MOVE", "LINE"]):
        with pytest.raises(S.StateError):
            S.LanguagePolicy.of(opcodes)
    with pytest.raises(S.StateError):
        S.LanguagePolicy.of(["HALT", "MOVE"], max_repeat_depth=9)


# ---------------------------------------------------------------------------
# every alphabet spells the same language


@pytest.mark.parametrize("name", sorted(CODECS))
def test_symbol_consumption_agrees_with_the_codec(name):
    """`_Alphabet.consume` and `Codec.decode` must not drift: the monitor's parse
    state comes from the first and every reported program from the second.

    Against `stream_bytes`, not the program, because a relative codec's symbols
    spell the delta view -- the same domain distinction `HaltMonitor` makes.
    """
    codec = CODECS[name]
    alphabet = S.SupportTable(codec, FULL).alphabet
    for program in synthetic.dataset(8, seed=11):
        cursor = S.SymbolCursor(codec.stride)
        out = bytearray()
        for symbol in codec.encode(program):
            byte = alphabet.consume(symbol, cursor)
            if byte is not None:
                out.append(byte)
        assert bytes(out) == codec.stream_bytes(program)


@pytest.mark.parametrize("name", sorted(CODECS))
@pytest.mark.parametrize("level", [S.CANONICAL, S.VM_SAFE])
def test_every_alphabet_admits_exactly_the_legal_bytes(name, level):
    """The alphabets differ in how they spell a byte and not in which bytes are
    legal. So the *bytes* reachable through a codec's symbol mask must equal the
    byte-level legal set -- including the `bit` alphabet, where a symbol is a bit
    and the mask is the set of prefixes of legal complete bytes.
    """
    codec = CODECS[name]
    table = S.SupportTable(codec, FULL, level)
    program = bytes([0x0D, 2, 2, 0, 0, 0x01, 10, 10, 0x0B, 2, 0, 0])
    state = S.GrammarState(FULL)
    for byte in program:
        state.step(byte)
        if state.done:
            break
        want = set(state.legal_bytes(level).tolist())
        reachable = set()
        for candidate in range(256):
            cursor = S.SymbolCursor(codec.stride)
            admitted = True
            for symbol in _spelling(codec, state, candidate):
                if not table.of(state, cursor).row[symbol]:
                    admitted = False
                    break
                spelled = table.alphabet.consume(symbol, cursor)
                assert spelled is None or spelled == candidate
            if admitted:
                reachable.add(candidate)
        assert reachable == want, (name, level, state.snapshot().stratum)


def _spelling(codec, state: S.GrammarState, byte: int) -> list[int]:
    """How this alphabet spells `byte` at this position, canonically.

    The alphabets differ here and nowhere else: `byte` has one symbol per byte,
    `bit` has eight, and the token alphabets keep opcodes in a disjoint region and
    -- when typed -- one value alphabet per `Kind`. Spelling a coordinate with the
    `XF` alphabet is a *representation* fault, so a test that always spelled
    operands as coordinates would be testing its own spelling.
    """
    from dm.isa.codec import N_SPECIAL, BitCodec, TokenCodec

    inner = getattr(codec, "inner", codec)
    if isinstance(inner, BitCodec):
        return [N_SPECIAL + ((byte >> shift) & 1) for shift in range(7, -1, -1)]
    if not isinstance(inner, TokenCodec):
        return [N_SPECIAL + byte]
    if state.phase is S.Phase.OPCODE:
        if byte in S.OPCODE_BYTES:
            return [inner._op_base + S.OPCODE_BYTES.index(byte)]
        # Not an opcode at all: the only spelling left is a value token, which the
        # canonical mask refuses for its region and the VM-safe mask refuses
        # through the static grid.
        return [inner._value_token(Kind.COORD, byte)]
    kind = state.expected_kind
    assert kind is not None
    return [inner._value_token(kind, byte)]


# ---------------------------------------------------------------------------
# the sampler


def _tiny(codec) -> DrawingLM:
    torch.manual_seed(0)
    return DrawingLM(Config(vocab_size=codec.vocab_size, d_model=32, n_layers=2,
                            n_heads=2, max_len=512))


@pytest.mark.parametrize("name", ["byte", "token", "token_typed", "bit"])
def test_a_canonically_masked_row_is_canonical_or_capped(name):
    """Invariant 3: a masked row that reaches `HALT` is canonical and carries no
    structural fault, and a row that hits the cap is *incomplete* rather than
    upgraded to valid. Run on an untrained model on purpose -- the mask is a
    property of the decoder, so it must hold at the worst possible weights.
    """
    codec = CODECS[name]
    policy = S.LanguagePolicy.of(["HALT", "MOVE", "LINE", "REPEAT", "ENDREP"])
    rows, cap = 8, 240
    monitor = S.StateMonitor(codec, rows, policy)
    ids = _tiny(codec).generate(rows, max_new=cap, monitor=monitor, forbid=(PAD, BOS))
    vm = VM()
    for row, verdict in zip(ids.cpu(), monitor.verdicts()):
        program = codec.decode(row.tolist())
        if verdict.halted:
            assert verdict.canonical, verdict.violations
            assert not structural(vm.run(program))
        else:
            assert not verdict.canonical      # capped, so incomplete
    assert monitor.pressure()["empty_support_events"] == 0


def test_the_mask_is_applied_before_top_k():
    """Truncation must see the surviving distribution. Applied the other way, a
    small `top_k` can be filled entirely with symbols the mask then removes, and
    the row samples from whatever is left of the tail."""
    codec = CODECS["byte"]
    policy = S.LanguagePolicy.of(["HALT", "MOVE", "LINE", "REPEAT", "ENDREP"])
    monitor = S.StateMonitor(codec, 4, policy)
    ids = _tiny(codec).generate(4, max_new=64, monitor=monitor, top_k=1,
                                forbid=(PAD, BOS))
    for row, verdict in zip(ids.cpu(), monitor.verdicts()):
        program = codec.decode(row.tolist())
        assert verdict.halted or len(program) >= 1
        if verdict.halted:
            assert verdict.canonical


def test_pre_generated_variates_make_two_decodes_identical():
    """Common random numbers, which is what pairs the three decoder cells.

    Two runs of the *same* cell under one variate block must be byte-identical,
    and the block is consumed one uniform per row per step whatever the mask
    does -- so a cell that stops earlier stays aligned by absolute step index.
    """
    codec = CODECS["byte"]
    policy = S.LanguagePolicy.of(["HALT", "MOVE", "LINE", "REPEAT", "ENDREP"])
    model = _tiny(codec)
    variates = torch.rand(4, 96, generator=torch.Generator().manual_seed(3))
    runs = [
        model.generate(4, max_new=96, monitor=S.StateMonitor(codec, 4, policy),
                       forbid=(PAD, BOS), variates=variates)
        for _ in range(2)
    ]
    assert torch.equal(runs[0], runs[1])
    with pytest.raises(ValueError, match="variates are"):
        model.generate(4, max_new=96, variates=variates[:, :10])


def test_inverse_cdf_never_emits_a_masked_zero_probability_tail():
    """A near-one variate must land on the last positive masked bucket."""
    codec = CODECS["byte"]
    policy = S.LanguagePolicy.of(["HALT", "MOVE", "LINE"])
    rows, width = 3, 4
    below_one = torch.nextafter(torch.tensor(1.0), torch.tensor(0.0)).item()
    variates = torch.full((rows, width), below_one)
    monitor = S.StateMonitor(codec, rows, policy)
    ids = _tiny(codec).generate(
        rows, max_new=width, monitor=monitor, forbid=(PAD, BOS),
        variates=variates,
    )
    for row, verdict in zip(ids.cpu(), monitor.verdicts()):
        program = codec.decode(row.tolist())
        assert FaultKind.UNKNOWN_OPCODE not in structural(VM().run(program))
        if verdict.halted:
            assert verdict.canonical


@pytest.mark.parametrize("value", [-0.1, 1.0, float("nan"), float("inf")])
def test_inverse_cdf_refuses_values_outside_the_uniform_contract(value):
    codec = CODECS["byte"]
    with pytest.raises(ValueError, match="variates"):
        _tiny(codec).generate(1, max_new=2, variates=torch.full((1, 2), value))


def test_the_raw_cell_measures_pressure_without_masking():
    """The raw cell is the existing sampler unchanged. It still reports where the
    mask *would* have removed support, which is what makes the three-cell table a
    comparison rather than three unrelated reports."""
    codec = CODECS["byte"]
    policy = S.LanguagePolicy.of(["HALT", "MOVE", "LINE", "REPEAT", "ENDREP"])
    raw = S.StateMonitor(codec, 4, policy, mask_mode=S.RAW)
    _tiny(codec).generate(4, max_new=64, monitor=raw, forbid=(PAD, BOS))
    pressure = raw.pressure()
    assert pressure["mask_mode"] == S.RAW
    assert pressure["positions_masked"] > 0
    assert pressure["rule_positions"]["static_grid"] > 0
    # And it really did not mask: an untrained model off the allowlist cannot
    # produce only canonical rows by luck.
    assert not all(v.canonical for v in raw.verdicts())


def test_legacy_raw_termination_does_not_absorb_grammar_faults():
    """S3 raw uses the old boundary/unknown-opcode halt monitor.

    A grammar observer may become terminal on an unmatched closer, CALL, or
    policy-depth overflow; none of those is allowed to truncate the raw
    sampler before the legacy monitor sees the next boundary HALT.
    """
    codec = CODECS["byte"]
    cases = (
        bytes([int(Op.ENDREP), int(Op.HALT)]),
        bytes([int(Op.CALL), 3, int(Op.HALT)]),
        bytes([int(Op.REPEAT), 1, 0, 0] * 5 + [int(Op.HALT)]),
    )
    for program in cases:
        monitor = S.LegacyStateMonitor(codec, 1, FULL)
        for symbol in codec.encode(program):
            monitor.step(np.array([[symbol]], dtype=np.int64))
        assert monitor.done[0], program.hex()
        assert monitor.verdicts()[0].halted is False
        assert not monitor.diagnostic_done[0] or monitor.done[0]

    grammar = S.StateMonitor(codec, 1, FULL, mask_mode=S.RAW)
    grammar.step(np.array([[codec.encode(bytes([int(Op.ENDREP)]))[0]]]))
    assert grammar.done[0]


def test_prompt_after_halt_is_trailing_not_canonical():
    """Prompt priming must observe symbols after a terminal HALT."""
    codec = CODECS["byte"]
    monitor = S.StateMonitor(codec, 1, FULL)
    monitor.prime(codec.encode(bytes([int(Op.HALT), int(Op.MOVE), 1, 1])))
    verdict = monitor.verdicts()[0]
    assert monitor.done[0]
    assert not verdict.canonical
    assert S.Reason.TRAILING in verdict.reasons()
    assert verdict.final.offset == 4


def test_legacy_monitor_separates_sampler_and_diagnostic_completion():
    """A terminal grammar trace stops accounting, not the frozen raw sampler."""
    codec = CODECS["byte"]
    monitor = S.LegacyStateMonitor(codec, 1, FULL)
    symbol = codec.encode(bytes([int(Op.ENDREP)]))[0]
    monitor.step(np.array([[symbol]], dtype=np.int64))
    assert not monitor.done[0]
    assert monitor.diagnostic_done[0]


def test_vm_safe_uses_vm_depth_when_policy_is_narrower():
    """A policy depth is canonical support; VM-safe support uses VM depth."""
    policy = S.LanguagePolicy.of(
        ["HALT", "REPEAT", "ENDREP"], max_repeat_depth=1,
    )
    state = S.GrammarState(policy)
    for byte in bytes([int(Op.REPEAT), 1, 0, 0]):
        state.step(byte)
    assert int(Op.REPEAT) not in state.legal_bytes(S.CANONICAL)
    assert int(Op.REPEAT) in state.legal_bytes(S.VM_SAFE)


def test_call_policy_is_a_closed_vocabulary():
    with pytest.raises(S.StateError, match="call_policy"):
        S.LanguagePolicy.of(["HALT"], call_policy="execute")


def test_close_cost_is_zero_at_a_clean_top_level_boundary():
    state = S.GrammarState(FULL)
    assert state.close_cost() == 0


def test_canonical_fault_remains_absorbing_after_halt():
    state = S.GrammarState(S.LanguagePolicy.of(
        ["HALT", "REPEAT", "ENDREP"], max_repeat_depth=1
    ))
    for byte in bytes([int(Op.REPEAT), 1, 0, 0] * 2):
        state.step(byte)
    assert state.status is S.Status.CANONICAL_FAULT
    state.step(int(Op.ENDREP))
    state.step(int(Op.HALT))
    assert state.status is S.Status.CANONICAL_FAULT


def test_token_wrong_region_is_a_representation_fault():
    codec = CODECS["token"]
    monitor = S.StateMonitor(codec, 1, FULL)
    wrong_opcode_region = codec.value_symbol(Kind.COORD, int(Op.MOVE))
    value_one = codec.value_symbol(Kind.COORD, 1)
    halt = codec.opcode_symbol(int(Op.HALT))
    assert wrong_opcode_region is not None and value_one is not None and halt is not None
    symbols = [wrong_opcode_region, value_one, value_one, halt]
    for symbol in symbols:
        monitor.step(np.array([[symbol]], dtype=np.int64))
    verdict = monitor.verdicts()[0]
    assert S.Reason.REPRESENTATION in verdict.reasons()
    assert not verdict.canonical


def test_the_halt_monitor_contract_still_holds_for_old_callers():
    """`StateMonitor` is a drop-in for `HaltMonitor`: same `stride`/`step`
    signature, same `done`/`pending` fields. `stride` is 1 for every alphabet
    because a mask decides per symbol while a parse decides per byte."""
    codec = CODECS["bit"]
    monitor = S.StateMonitor(codec, 2, FULL, mask_mode=S.RAW)
    assert monitor.stride == 1
    program = assemble("MOVE 1 1\nHALT")
    symbols = np.array([codec.encode(program), codec.encode(program)]).T
    for column in symbols:
        monitor.step(column.reshape(1, -1))
    assert monitor.done.all()
    assert [v.canonical for v in monitor.verdicts()] == [True, True]


# ---------------------------------------------------------------------------
# the frozen coverage tables


#: `docs/state-freeze.md` §2 and §4, exactly. The corpora are rebuilt the way
#: their checkpoints were scored -- the same `n_train` -- because a val split
#: filtered against a different train split is a different corpus.
VENUES: dict[str, dict] = {
    "synthetic L1": {
        "opcodes": {"MOVE", "LINE", "CURVE", "CIRCLE", "WIDTH", "REPEAT", "ENDREP",
                    "HALT"},
        "max_depth": 1,
        "strata": {"opcode_halt_legal": 11_376, "opcode_in_scope": 3_132},
    },
    "synthetic L0 flat": {
        "opcodes": {"MOVE", "LINE", "CURVE", "CIRCLE", "WIDTH", "HALT"},
        "max_depth": 0,
        "strata": {"opcode_halt_legal": 18_629, "opcode_in_scope": 0},
    },
    "composed structured": {
        "opcodes": {"MOVE", "LINE", "REPEATX", "ENDREP", "HALT"},
        "max_depth": 1,
        "strata": {"opcode_halt_legal": 20_285, "opcode_in_scope": 23_973},
    },
    "composed flat": {
        "opcodes": {"MOVE", "LINE", "HALT"},
        "max_depth": 0,
        "strata": {"opcode_halt_legal": 74_471, "opcode_in_scope": 0},
    },
}


def _venue_programs() -> dict[str, list[bytes]]:
    _, l1 = synthetic.split(100_000, 1_000, seed=0)
    _, l0 = synthetic.split(100_000, 1_000, seed=0, flatten=True)
    scenes = composed.build(1_000, "valid", seed=composed.val_seed(0),
                            categories=("cat", "dog", "bus", "car", "tree"))
    return {
        "synthetic L1": l1,
        "synthetic L0 flat": l0,
        "composed structured": [s.structured for s in scenes],
        "composed flat": [s.flat for s in scenes],
    }


@pytest.mark.slow
def test_the_frozen_coverage_table_is_what_the_corpora_contain():
    """§2's allowlists and §4's coverage counts, measured.

    Read the failures as prohibitions rather than as bookkeeping: no natural
    venue reaches scope depth 2 and none contains a standalone `XFORM … ENDX`, so
    a depth-2 or `ENDX` result is an unseen-embedding result until a supplement
    supplies the state in every arm identically.
    """
    for name, programs in _venue_programs().items():
        expected = VENUES[name]
        policy = S.LanguagePolicy.from_programs(programs, label=name)
        assert {op.name for op in policy.opcodes} == expected["opcodes"], name

        strata: dict[str, int] = {}
        depth = 0
        admitted = True
        for program in programs:
            verdict = S.trace(program, policy)
            assert verdict.canonical, (name, verdict.violations)
            state = S.GrammarState(policy)
            for byte in program:
                snapshot = state.snapshot()
                strata[snapshot.stratum] = strata.get(snapshot.stratum, 0) + 1
                # Open *source* scopes. A `REPEATX` is one scope even though it
                # occupies an entry in each of the VM's two stacks, so
                # `loop_depth + xform_depth` reads 2 for the composed corpus's
                # single loop and would make its coverage look deeper than it is.
                depth = max(depth, len(state.scope_stack))
                admitted &= bool(byte in state.legal_bytes(S.CANONICAL))
                state.step(byte)
        assert admitted, f"{name}: a corpus target was outside the canonical mask"
        assert depth == expected["max_depth"], name
        for stratum, count in expected["strata"].items():
            assert strata.get(stratum, 0) == count, (name, stratum)
        assert "operand_none" not in strata, name


def test_the_state_probe_reaches_states_no_corpus_does():
    """The reason the fixture exists. `endx_legal`, scope depth 2 and up, and the
    standalone transform branch are unreachable from every natural venue, so a
    parser tested only on corpora would be untested exactly where the paper's
    generalisation questions live."""
    seen = {"endx": False, "depth2": False, "xform_top": False}
    for program in PROBE:
        state = S.GrammarState(FULL)
        for byte in program:
            if state.done:
                break
            seen["endx"] |= state.endx_legal
            seen["depth2"] |= (state.loop_depth + state.xform_depth) >= 2
            seen["xform_top"] |= state.top_scope is Op.XFORM
            state.step(byte)
    assert all(seen.values()), seen
