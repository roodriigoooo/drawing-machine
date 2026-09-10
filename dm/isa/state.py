"""Prefix state: which language a mask constrains, and what is legal *here*.

`HaltMonitor` (`dm/isa/codec.py`) can answer "how many operand bytes remain?"
and nothing else. It cannot answer "which operand *kind* is expected?", "which
closer matches the open scope?" or "is a top-level halt legal at this point?" —
the three questions Direction 1 turned into an experiment. This module answers
them in the layers audited by `docs/state.md`, and it is the only
definition of the canonical language.

**Three predicates, deliberately kept apart** (the truth table is frozen in
[`docs/state-freeze.md`](../../docs/state-freeze.md) §1 and pinned in
`tests/test_state.py`):

- **canonical** — program membership for the paper: an explicit opcode
  allowlist, exact operand domains, properly nested scopes, matching closers,
  the depth bounds and a terminal top-level `HALT`.
- **VM acceptance** — `dm.vm.interp.VM.run` reporting no fault. Deliberately
  more permissive: the executor is total, so it masks `XF = 8` to three bits,
  treats reserved `COLOR` as a no-op and accepts a count-one crossed scope.
- **VM-safe** — the per-symbol predicate this module offers next to canonical:
  *emitting this symbol here does not make a VM fault inevitable from this
  prefix*. It is weaker than canonical and is **not** VM acceptance. A crossing
  whose fault depends on a repeat count is VM-safe and non-canonical at once, and
  `docs/state-freeze.md` §5 freezes that: a mask may not relabel a VM-accepted
  but non-canonical program as a VM fault.

**VM-safe is prefix-local, and measurement says that is a real limit.** A loop
replays its body, so bytes already emitted execute again: a `REPEAT` body holding
a dangling `XFORM` pushes another transform scope on every iteration and faults
`DEPTH_OVERFLOW` at some later one, and a body holding a crossing `ENDX` faults
`UNMATCHED_ENDREP` on the second. Neither is visible from the source prefix --
seeing it needs the suspendable executor of S4 -- so VM-safe-masked rows *do*
still fault, which is one of the reasons the canonical mask is the primary one:
proper nesting makes every iteration's scope bookkeeping identical, so a
canonically masked row cannot manufacture a structural fault by replay. The one
fault a canonical row can still take is `OUT_OF_FUEL`, which is a resource bound
rather than a structural one and is reported in its own column.

**Why this is not the VM, and cannot be.** A source-prefix parser and a runtime
executor are different machines. When `ENDREP` arrives the VM jumps *backwards*
and executes already-generated body bytes again, so a tracker that applied one
transition per source instruction would report the wrong cursor and transform
after any loop. This layer therefore carries parsing and control-scope facts
only — no geometry, no cursor, no transform. Exact execution snapshots need a
suspendable executor and were the historical S4 design. `docs/state.md` records
why that work is now retired.

**What it does mirror exactly is the VM's scope bookkeeping**, because that is
what decides what a closer may close. `VM.run` keeps a repeat-frame stack *and*
a transform-scope list with a floor, and the two interact in ways no single
counter reproduces: `ENDREP` on a `REPEATX` frame deletes every transform scope
from its own slot upward, `ENDREP` on a plain `REPEAT` frame deletes none (so a
dangling `XFORM` in its body survives the loop and faults at `HALT`), and an
`ENDX` may close a scope that sits *below* an open repeat frame. All three are
measured, and `tests/test_state.py` pins this module against the VM rather than
against a second reading of the rule.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum, IntEnum
from functools import lru_cache
from typing import NamedTuple

import numpy as np

from .codec import (
    N_SPECIAL,
    BitCodec,
    ByteCodec,
    Codec,
    HaltMonitor,
    RelativeCodec,
    TokenCodec,
)
from .spec import MAX_REPEAT_DEPTH, SPECS, ISAError, Kind, Op, spec_for

#: Every byte the ISA reads as an opcode. The static grid, and the only set a
#: `byte`-alphabet model has to discover for itself.
OPCODE_BYTES: tuple[int, ...] = tuple(sorted(int(op) for op in SPECS))

_ALL_BYTES = np.arange(256, dtype=np.int64)
_OPCODE_BYTE_ARRAY = np.array(OPCODE_BYTES, dtype=np.int64)
#: Position of each opcode byte in the token alphabets' reserved region. The same
#: table `dm.isa.codec` builds; read here so a mask and a decode agree by
#: construction rather than by two matching literals.
_OPCODE_SLOT = {byte: index for index, byte in enumerate(OPCODE_BYTES)}

#: Opcodes that open a scope, and the closer each one takes. `REPEATX` is closed
#: by `ENDREP` -- it *is* `REPEAT` with a transform, and the composed corpus
#: spells it that way (`dm/data/composed.py`).
CLOSER_OF: dict[Op, Op] = {Op.REPEAT: Op.ENDREP, Op.REPEATX: Op.ENDREP, Op.XFORM: Op.ENDX}

#: Operand values each `Kind` may canonically hold, from `encode_operand`'s own
#: refusals plus the ISA's stated ranges. Narrow on purpose and no narrower:
#: a coordinate a transform may later clamp is legal, and so is `WIDTH 0`, which
#: the VM clamps to one. Those are semantic choices, not structural faults
#: (`docs/state-freeze.md` §1).
CANONICAL_VALUES: dict[Kind, np.ndarray] = {
    Kind.COORD: _ALL_BYTES,
    Kind.DELTA: _ALL_BYTES,          # every encoded i8
    Kind.COUNT: np.arange(1, 256, dtype=np.int64),
    Kind.SCALAR: _ALL_BYTES,
    Kind.ID: _ALL_BYTES,
    Kind.XF: np.arange(8, dtype=np.int64),
}

#: The subset of the above a *VM* fault is inevitable outside of. Only
#: `COUNT = 0` qualifies: it is `ZERO_REPEAT`. `XF = 8` is masked to `0` and
#: executes, so it is canonical-only.
VM_SAFE_VALUES: dict[Kind, np.ndarray] = {
    **{kind: _ALL_BYTES for kind in Kind},
    Kind.COUNT: CANONICAL_VALUES[Kind.COUNT],
}


class Phase(IntEnum):
    """Where in an instruction a byte position sits."""

    OPCODE = 0
    OPERAND = 1


class Status(IntEnum):
    """A prefix's one-field summary; numeric values are category IDs.

    `LIVE` and `HALTED` describe the parse; `CANONICAL_FAULT` says the prefix has
    left the canonical language and `VM_FAULT` that the executor would have
    stopped. Terminal faults are absorbing: precedence is
    `VM_FAULT > CANONICAL_FAULT > HALTED > LIVE`, so a halted prefix cannot erase
    an earlier violation. This is the declared **absorbing** input for an invalid
    unmasked prefix: every arm of a state-input comparison maps such a prefix to
    the same value, or the treatment has no defined next input after the first
    wrong closer (`docs/state-freeze.md` §1).

    Do not implement that precedence with ``max(Status)``: the stable enum values
    are report/model category IDs, not an ordering relation.
    """

    LIVE = 0
    CANONICAL_FAULT = 1
    HALTED = 2
    VM_FAULT = 3


class Reason(str, Enum):
    """Why a symbol is excluded, or why a program is not canonical.

    The first eight are the **frozen, mutually exclusive** mask rules of
    `docs/state-freeze.md` §5, in precedence order: a symbol removed by several
    is credited to the first. The last three are program-level verdicts no mask
    can prevent -- they are properties of where the stream *ended*.
    """

    CONTROL = "control"
    STATIC_GRID = "static_grid"
    REPRESENTATION = "representation"
    OPCODE_POLICY = "opcode_policy"
    OPERAND_DOMAIN = "operand_domain"
    SCOPE = "scope"
    DEPTH = "depth"
    HALT = "halt"
    TRUNCATED = "truncated"
    NO_HALT = "no_halt"
    TRAILING = "trailing"


#: The frozen precedence. Reported in this order, never summed across rules
#: without also reporting the union.
MASK_RULES: tuple[Reason, ...] = (
    Reason.CONTROL,
    Reason.STATIC_GRID,
    Reason.REPRESENTATION,
    Reason.OPCODE_POLICY,
    Reason.OPERAND_DOMAIN,
    Reason.SCOPE,
    Reason.DEPTH,
    Reason.HALT,
)

#: Rules the VM-safe level applies. `REPRESENTATION` is absent because a
#: wrong-region symbol still decodes to a byte the VM runs; the policy rule
#: survives only for `CALL`, the operand rule only for `COUNT = 0`, and the
#: scope rule only where the closer has nothing at all to close.
VM_SAFE_RULES: frozenset[Reason] = frozenset(
    {Reason.CONTROL, Reason.STATIC_GRID, Reason.OPCODE_POLICY,
     Reason.OPERAND_DOMAIN, Reason.SCOPE, Reason.DEPTH, Reason.HALT}
)

CANONICAL = "canonical"
VM_SAFE = "vm_safe"
RAW = "raw"
LEVELS: tuple[str, ...] = (RAW, VM_SAFE, CANONICAL)

# There is no callable/library executor in the current VM. Keeping this as an
# explicit closed vocabulary prevents a future report from spelling an
# unimplemented policy as if it were executable.
CALL_POLICIES: frozenset[str] = frozenset({"unsupported"})


class StateError(ISAError):
    """The instrument was asked something it must refuse rather than guess."""


# ---------------------------------------------------------------------------
# layer 1: which language


@dataclass(frozen=True)
class LanguagePolicy:
    """The explicit opcode allowlist and the canonical rules over it.

    **`Tier` is not a policy key.** `Tier.L2` contains `CALL`, which `VM.run`
    refuses outright, and `TrainConfig.tier` is ignored by the QuickDraw and
    composed loaders entirely -- so a tier inferred after the fact would name a
    language no checkpoint was trained on. Every report in this branch carries an
    allowlist reconstructed from the corpus the checkpoint was actually scored
    on (`from_programs`), and `docs/state-freeze.md` §2 records the four measured
    ones.

    `call_policy` is `"unsupported"` and frozen there: `CALL` is allocated in the
    ISA, `VM.run` reports `CALL_UNSUPPORTED`, and an `ID` state means nothing
    until a library address space exists.
    """

    opcodes: frozenset[Op]
    max_repeat_depth: int = MAX_REPEAT_DEPTH
    max_xform_depth: int = MAX_REPEAT_DEPTH
    call_policy: str = "unsupported"
    label: str = ""

    def __post_init__(self) -> None:
        if self.call_policy not in CALL_POLICIES:
            raise StateError(
                f"call_policy must be one of {sorted(CALL_POLICIES)}, got "
                f"{self.call_policy!r}; no other CALL executor is implemented"
            )
        if Op.HALT not in self.opcodes:
            raise StateError(
                "a policy without HALT describes a language with no terminating "
                "program; the canonical predicate would reject every corpus"
            )
        if Op.CALL in self.opcodes:
            raise StateError(
                "CALL is in the allowlist but call_policy='unsupported'; "
                "VM.run reports CALL_UNSUPPORTED, so this policy admits programs "
                "the executor refuses"
            )
        if Op.COLOR in self.opcodes:
            raise StateError(
                "COLOR is Tier.RESERVED -- the VM treats it as a no-op and no "
                "corpus contains it, so admitting it would put an unspecified "
                "instruction inside the canonical language"
            )
        for name, depth in (("repeat", self.max_repeat_depth),
                            ("xform", self.max_xform_depth)):
            if not 1 <= depth <= MAX_REPEAT_DEPTH:
                raise StateError(
                    f"max_{name}_depth={depth} is outside the VM's own bound of "
                    f"{MAX_REPEAT_DEPTH}; a mask cannot admit a program the "
                    "executor faults on by construction"
                )

    @classmethod
    def of(cls, opcodes: Iterable[Op | str], **kwargs) -> LanguagePolicy:
        """A policy from mnemonics or `Op`s, for tests and probe fixtures."""
        resolved = frozenset(
            op if isinstance(op, Op) else Op[str(op).upper()] for op in opcodes
        )
        return cls(resolved, **kwargs)

    @classmethod
    def from_programs(cls, programs: Sequence[bytes], label: str = "",
                      **kwargs) -> LanguagePolicy:
        """The allowlist a corpus actually uses, by walking it with the ISA.

        Raises on a program the ISA cannot parse. A corpus is the authority on
        which language its checkpoint was trained on, and a corpus that does not
        parse is not one -- reading past the break would silently narrow the
        allowlist to whatever the prefix happened to contain.
        """
        used: set[Op] = set()
        for index, program in enumerate(programs):
            pc = 0
            while pc < len(program):
                spec = spec_for(program[pc])  # raises UnknownOpcode, deliberately
                if pc + spec.size > len(program):
                    raise StateError(
                        f"program {index} is truncated at byte {pc} "
                        f"({spec.mnemonic}); an allowlist read off a broken "
                        "corpus is not the language the model was trained on"
                    )
                used.add(spec.op)
                pc += spec.size
        return cls(frozenset(used), label=label, **kwargs)

    def allows(self, op: Op) -> bool:
        return op in self.opcodes

    @property
    def opcode_bytes(self) -> np.ndarray:
        return np.array(sorted(int(op) for op in self.opcodes), dtype=np.int64)

    def as_dict(self) -> dict:
        """The policy as a report field. Part of every state report's identity."""
        return {
            "opcodes": [Op(op).name for op in sorted(self.opcodes)],
            "max_repeat_depth": self.max_repeat_depth,
            "max_xform_depth": self.max_xform_depth,
            "call_policy": self.call_policy,
            "label": self.label,
        }

    @property
    def digest(self) -> str:
        """Stable short key over everything except the human label."""
        body = {k: v for k, v in self.as_dict().items() if k != "label"}
        return hashlib.sha256(
            json.dumps(body, sort_keys=True).encode()
        ).hexdigest()[:12]


#: Every opcode the ISA can execute without a guaranteed fault, which is the
#: allowlist the VM-safe level uses in place of a policy's.
VM_SAFE_OPCODES: frozenset[Op] = frozenset(SPECS) - {Op.CALL}


# ---------------------------------------------------------------------------
# layer 2: grammar and control-scope state


def _name_of(op: Op | None) -> str:
    """A scope's name for a diagnostic, without asserting one is open."""
    return "no" if op is None else op.name


class Violation(NamedTuple):
    """One departure from the canonical language, at a byte offset.

    `terminal` says the *parse* stops here, which it does exactly where
    `VM.run` breaks: an unknown opcode, a truncated instruction, a closer with
    nothing to close, a depth overflow and `CALL`. A non-terminal violation
    leaves the state live and is carried beside it, because a program can leave
    the canonical language and go on to halt.
    """

    reason: Reason
    offset: int
    detail: str
    terminal: bool


class _Frame(NamedTuple):
    """An open repeat frame, mirroring `dm.vm.interp._Frame`'s bookkeeping.

    `repeat_count` rather than `count`: a `NamedTuple` field called `count` would
    shadow `tuple.count`, and this one is read by diagnostics.
    """

    opener: Op
    seq: int                 # source instruction index, so scopes can be ordered
    xform_slot: int | None   # the transform scope a REPEATX owns, if any
    xform_floor: int         # the floor in force before this frame opened
    repeat_count: int


class _Scope(NamedTuple):
    """An open transform scope: one entry of the VM's `xforms` list."""

    owner: Op                # XFORM, or REPEATX for the slot a loop owns
    seq: int


class StateKey(NamedTuple):
    """Everything about a prefix that decides its legal-symbol set.

    Small and hashable so a support table memoises on it: a corpus of 220,000
    byte positions visits a handful of distinct keys, and materialising one
    `(positions, vocabulary)` mask instead would be larger than the logits
    (`dm/eval/attribution.py` makes the same trade).
    """

    phase: Phase
    kind: int            # `Kind` value at an operand position, else -1
    halt_legal: bool
    endrep_canonical: bool
    endrep_vm: bool
    endx_canonical: bool
    endx_vm: bool
    repeat_full: bool
    xform_full: bool
    repeat_vm_full: bool
    xform_vm_full: bool
    done: bool

    @property
    def stratum(self) -> str:
        """The reporting stratum of `docs/state-freeze.md` §5.

        Opcode boundaries are split by whether a halt is legal there, because that
        is the one dynamic fact the flat venues never vary and the structured
        venue varies at 54% of its boundaries. Operand positions are split by
        `Kind`, which is what a typed alphabet already gives away.
        """
        if self.phase is Phase.OPERAND:
            # `>= 0`, not truthiness: `Kind.COORD` is 0, and the falsy reading
            # filed every coordinate operand -- most of the corpus -- under a
            # stratum meaning "no kind expected".
            return (f"operand_{Kind(self.kind).name.lower()}" if self.kind >= 0
                    else "operand_none")
        return "opcode_halt_legal" if self.halt_legal else "opcode_in_scope"


class StateSnapshot(NamedTuple):
    """The prefix state a report stratifies by and a model would be shown.

    Grammar and control scope only. Cursor, transform and path cardinality are
    execution state and arrive with the suspendable executor (S4).
    """

    key: StateKey
    pending: int
    loop_depth: int
    xform_depth: int
    xform_floor: int
    top_scope: int       # `Op` value of the innermost open scope, else -1
    status: Status
    offset: int

    @property
    def phase(self) -> Phase:
        return self.key.phase

    @property
    def expected_kind(self) -> Kind | None:
        return None if self.key.kind < 0 else Kind(self.key.kind)

    @property
    def halt_legal(self) -> bool:
        return self.key.halt_legal

    @property
    def stratum(self) -> str:
        """`StateKey.stratum`, which is where the definition lives."""
        return self.key.stratum


class GrammarState:
    """Parsing and control-scope state over a source byte prefix.

    Advanced only by *committed* bytes: a partial instruction has grammar state
    and has not executed. One instance per sampler row; a teacher-forced dataset
    computes one `trace` and slices it.
    """

    __slots__ = ("_first_exclusions", "_floor", "_frames", "_halted", "_offset",
                 "_op", "_pending", "_seq", "_terminal", "_violations", "_xforms",
                 "policy")

    def __init__(self, policy: LanguagePolicy) -> None:
        self.policy = policy
        self._pending: list[Kind] = []
        self._op: Op | None = None
        self._frames: list[_Frame] = []
        self._xforms: list[_Scope] = []
        self._floor = 0
        self._seq = 0
        self._offset = 0
        self._halted = False
        self._terminal: Violation | None = None
        self._violations: list[Violation] = []
        # One byte-level first-exclusion cause per committed byte. `None` means
        # the byte survived the canonical cascade. Reports use this sequence,
        # never the full diagnostic violation list, for exclusive accounting.
        self._first_exclusions: list[Reason | None] = []

    # -- read-only view -----------------------------------------------------

    @property
    def phase(self) -> Phase:
        return Phase.OPERAND if self._pending else Phase.OPCODE

    @property
    def expected_kind(self) -> Kind | None:
        return self._pending[0] if self._pending else None

    @property
    def pending(self) -> int:
        return len(self._pending)

    @property
    def loop_depth(self) -> int:
        return len(self._frames)

    @property
    def xform_depth(self) -> int:
        return len(self._xforms)

    @property
    def xform_floor(self) -> int:
        return self._floor

    @property
    def offset(self) -> int:
        return self._offset

    @property
    def scope_stack(self) -> tuple[Op, ...]:
        """Open scopes, outermost first, as the source opened them.

        A merged view of the two stacks the VM keeps, ordered by source
        position. `REPEATX` appears once: its frame and the transform scope it
        owns are one instruction.
        """
        merged = [(f.seq, f.opener) for f in self._frames]
        merged += [(s.seq, s.owner) for s in self._xforms if s.owner is not Op.REPEATX]
        return tuple(op for _, op in sorted(merged))

    @property
    def top_scope(self) -> Op | None:
        """The innermost open scope -- the one a closer must match."""
        best: tuple[int, Op] | None = None
        for seq, op in ([(f.seq, f.opener) for f in self._frames]
                        + [(s.seq, s.owner) for s in self._xforms]):
            if best is None or seq > best[0]:
                best = (seq, op)
        return None if best is None else best[1]

    @property
    def done(self) -> bool:
        """No further byte of this row is executed: halted, or a hard fault."""
        return self._halted or self._terminal is not None

    @property
    def halted(self) -> bool:
        return self._halted

    @property
    def terminal(self) -> Violation | None:
        return self._terminal

    @property
    def violations(self) -> tuple[Violation, ...]:
        return tuple(self._violations)

    @property
    def canonical(self) -> bool:
        """No canonical violation *so far*. Program membership also needs a
        terminal top-level `HALT`, which only `StateTrace` can see."""
        return not self._violations

    @property
    def status(self) -> Status:
        if self._terminal is not None:
            return Status.VM_FAULT if self._terminal.terminal else Status.CANONICAL_FAULT
        # A canonical departure is absorbing even if the VM later reaches a
        # HALT. The old precedence let CANONICAL_FAULT become HALTED, which made
        # the declared absorbing input depend on how much trailing program was
        # observed.
        if self._violations:
            return Status.CANONICAL_FAULT
        if self._halted:
            return Status.HALTED
        return Status.LIVE

    # -- legality -----------------------------------------------------------

    @property
    def halt_legal(self) -> bool:
        """`HALT` is legal only at an opcode boundary with nothing open.

        Canonical and VM agree here: `VM.run` faults `UNTERMINATED_REPEAT` when
        either stack is non-empty, and then halts anyway -- so an invalid halt is
        still a halt, and §3 of the freeze counts it apart from a valid one.
        """
        return (self.phase is Phase.OPCODE and not self._frames and not self._xforms)

    @property
    def endrep_legal(self) -> bool:
        """Canonical: the innermost open scope is a `REPEAT` or a `REPEATX`."""
        return bool(self._frames) and self.top_scope in (Op.REPEAT, Op.REPEATX)

    @property
    def endrep_legal_vm(self) -> bool:
        """VM: any open frame. It pops the innermost one whatever sits above it,
        which is how the count-one crossing executes without a fault."""
        return bool(self._frames)

    @property
    def endx_legal(self) -> bool:
        """Canonical: the innermost open scope is a standalone `XFORM`.

        It cannot close the transform a `REPEATX` owns -- the loop owns that
        slot -- and it cannot reach past an open frame.
        """
        return bool(self._xforms) and self.top_scope is Op.XFORM

    @property
    def endx_legal_vm(self) -> bool:
        """VM: `len(xforms) > xform_floor`, exactly. The floor rises only for a
        `REPEATX`, so an `ENDX` inside a plain `REPEAT` body may close a scope
        opened *outside* the loop -- a crossing, and no fault at count one."""
        return len(self._xforms) > self._floor

    def key(self) -> StateKey:
        return StateKey(
            phase=self.phase,
            kind=-1 if self.expected_kind is None else int(self.expected_kind),
            halt_legal=self.halt_legal,
            endrep_canonical=self.endrep_legal,
            endrep_vm=self.endrep_legal_vm,
            endx_canonical=self.endx_legal,
            endx_vm=self.endx_legal_vm,
            repeat_full=len(self._frames) >= self.policy.max_repeat_depth,
            xform_full=len(self._xforms) >= self.policy.max_xform_depth,
            repeat_vm_full=len(self._frames) >= MAX_REPEAT_DEPTH,
            xform_vm_full=len(self._xforms) >= MAX_REPEAT_DEPTH,
            done=self.done,
        )

    def snapshot(self) -> StateSnapshot:
        top = self.top_scope
        return StateSnapshot(
            key=self.key(), pending=self.pending, loop_depth=self.loop_depth,
            xform_depth=self.xform_depth, xform_floor=self._floor,
            top_scope=-1 if top is None else int(top), status=self.status,
            offset=self._offset,
        )

    def legal_bytes(self, level: str = CANONICAL) -> np.ndarray:
        """Bytecode byte values this prefix may be followed by, at one level.

        The cascade of `docs/state-freeze.md` §5 over *bytes*; the codec adapter
        below maps the result into an alphabet's symbols. Empty for a finished
        row, which is an instrument error at the sampler and not a fallback.
        """
        return _byte_cascade(self.key(), self.policy, level)[-1][1]

    def close_cost(self) -> int:
        """Bytecode bytes that would take this prefix to a canonical `HALT`.

        The pending operands of the instruction in flight, one closer for every
        open scope, and the `HALT` itself. A row that hits the decode cap is
        *incomplete*, and this says by how much -- which is the difference between
        "the mask pushed a program just past the budget" and "the mask left it
        arbitrarily far from ever terminating" (`docs/state-freeze.md` §3.2).

        Zero once halted, or at a live top-level opcode boundary. The latter is
        already closed; it does not need to pay for the terminating HALT. A
        partial instruction still needs its owed operands, and an open scope
        still needs its closer(s) plus a HALT.
        """
        if self._halted or (self.phase is Phase.OPCODE and not self.scope_stack):
            return 0
        return self.pending + len(self.scope_stack) + 1

    def copy(self) -> GrammarState:
        clone = GrammarState(self.policy)
        clone._pending = list(self._pending)
        clone._op = self._op
        clone._frames = list(self._frames)
        clone._xforms = list(self._xforms)
        clone._floor = self._floor
        clone._seq = self._seq
        clone._offset = self._offset
        clone._halted = self._halted
        clone._terminal = self._terminal
        clone._violations = list(self._violations)
        clone._first_exclusions = list(self._first_exclusions)
        return clone

    # -- advancing ----------------------------------------------------------

    def _flag(self, reason: Reason, detail: str, terminal: bool = False) -> None:
        violation = Violation(reason, self._offset, detail, terminal)
        self._violations.append(violation)
        if terminal:
            self._terminal = violation

    @property
    def first_exclusions(self) -> tuple[Reason | None, ...]:
        """The exclusive byte-level cause recorded for each committed byte."""
        return tuple(self._first_exclusions)

    def _first_exclusion(self, byte: int) -> Reason | None:
        """Return the first frozen rule that excludes one byte at this prefix."""
        if self.done:
            return Reason.TRAILING
        alive = True
        for reason, surviving in _byte_cascade(self.key(), self.policy, CANONICAL):
            excluded = alive and byte not in surviving
            if excluded:
                return reason
            alive = byte in surviving
            if not alive:
                return reason
        return None

    def record_symbol_violation(self, reason: Reason, detail: str) -> None:
        """Record a codec-level violation without pretending it was a byte fault."""
        self._flag(reason, detail)

    def step(self, byte: int) -> None:
        """Commit one bytecode byte.

        A byte offered to a finished row is *trailing*: the VM stopped, every
        later byte is dead, and the program is not canonical. Recorded once so
        the count is visible, then ignored -- raising here would make a sloppy
        caller's bug look like a corpus property.
        """
        self._first_exclusions.append(self._first_exclusion(byte))
        if self.done:
            if not any(v.reason is Reason.TRAILING for v in self._violations):
                self._flag(Reason.TRAILING,
                           f"byte 0x{byte:02x} after the stream stopped")
            self._offset += 1
            return

        if self._pending:
            self._operand(byte)
        else:
            self._opcode(byte)
        self._offset += 1

    def _operand(self, byte: int) -> None:
        kind = self._pending.pop(0)
        if byte not in CANONICAL_VALUES[kind]:
            self._flag(
                Reason.OPERAND_DOMAIN,
                f"{kind.name} operand {byte} outside its domain",
            )
        if self._op in (Op.REPEAT, Op.REPEATX) and kind is Kind.COUNT:
            # The frame is pushed by the opcode, before its count is known, so
            # the count is filled in here. It is read only by diagnostics: the
            # VM-safe level does not use it, because whether a *crossing* faults
            # depends on the count and the freeze fixes crossings as VM-safe.
            frame = self._frames[-1]
            self._frames[-1] = frame._replace(repeat_count=byte)

    def _opcode(self, byte: int) -> None:
        try:
            spec = spec_for(byte)
        except ISAError:
            self._flag(Reason.STATIC_GRID, f"0x{byte:02x} is not an opcode",
                       terminal=True)
            return
        op, self._op, self._seq = spec.op, spec.op, self._seq + 1
        if not self.policy.allows(op):
            self._flag(
                Reason.OPCODE_POLICY,
                f"{spec.mnemonic} is outside this policy's allowlist",
                terminal=op is Op.CALL,
            )
            if op is Op.CALL:
                return

        if op is Op.HALT:
            if not self.halt_legal:
                self._flag(Reason.HALT,
                           f"HALT with {len(self._frames)} frame(s) and "
                           f"{len(self._xforms)} transform scope(s) open")
            self._halted = True
            return

        if op in (Op.REPEAT, Op.REPEATX):
            self._open_loop(op, spec.mnemonic)
        elif op is Op.XFORM:
            self._open_xform(spec.mnemonic)
        elif op is Op.ENDREP:
            self._close_loop()
        elif op is Op.ENDX:
            self._close_xform()

        if self._terminal is None:
            self._pending = list(spec.operands)

    def _open_loop(self, op: Op, mnemonic: str) -> None:
        if len(self._frames) >= MAX_REPEAT_DEPTH:
            self._flag(Reason.DEPTH, f"{mnemonic} at repeat depth "
                                     f"{len(self._frames)}", terminal=True)
            return
        if len(self._frames) >= self.policy.max_repeat_depth:
            # This is outside the checkpoint's canonical language but still
            # executable by the VM. Keep tracking it so the VM-safe mask uses
            # the VM's depth fact rather than the policy's narrower fact.
            self._flag(Reason.DEPTH, f"{mnemonic} at policy repeat depth "
                                     f"{len(self._frames)}")
        slot = None
        if op is Op.REPEATX:
            if len(self._xforms) >= MAX_REPEAT_DEPTH:
                self._flag(Reason.DEPTH, f"{mnemonic} at transform depth "
                                         f"{len(self._xforms)}", terminal=True)
                return
            if len(self._xforms) >= self.policy.max_xform_depth:
                self._flag(Reason.DEPTH, f"{mnemonic} at policy transform depth "
                                         f"{len(self._xforms)}")
            slot = len(self._xforms)
            self._xforms.append(_Scope(Op.REPEATX, self._seq))
        self._frames.append(_Frame(op, self._seq, slot, self._floor, repeat_count=0))
        if slot is not None:
            self._floor = slot + 1

    def _open_xform(self, mnemonic: str) -> None:
        if len(self._xforms) >= MAX_REPEAT_DEPTH:
            self._flag(Reason.DEPTH,
                       f"{mnemonic} at transform depth {len(self._xforms)}",
                       terminal=True)
            return
        if len(self._xforms) >= self.policy.max_xform_depth:
            self._flag(Reason.DEPTH,
                       f"{mnemonic} at policy transform depth {len(self._xforms)}")
        self._xforms.append(_Scope(Op.XFORM, self._seq))

    def _close_loop(self) -> None:
        if not self._frames:
            self._flag(Reason.SCOPE, "ENDREP with no open repeat frame",
                       terminal=True)
            return
        if not self.endrep_legal:
            # A crossing: an `XFORM` opened inside this body is still open.
            # The VM pops the frame anyway and faults later, or not at all when
            # the count is one, so this is non-terminal by measurement.
            self._flag(Reason.SCOPE,
                       f"ENDREP closes a {self._frames[-1].opener.name} frame "
                       f"under an open {_name_of(self.top_scope)} scope")
        frame = self._frames.pop()
        if frame.xform_slot is not None:
            # `VM.run` does `del xforms[slot:]`, so a `REPEATX` drops every
            # transform scope its body left open. A plain `REPEAT` drops none --
            # which is why a dangling `XFORM` in a `REPEAT` body survives the
            # loop and faults at `HALT`.
            del self._xforms[frame.xform_slot:]
        self._floor = frame.xform_floor

    def _close_xform(self) -> None:
        if not self.endx_legal_vm:
            self._flag(Reason.SCOPE,
                       "ENDX with no closable transform scope "
                       f"(depth {len(self._xforms)}, floor {self._floor})",
                       terminal=True)
            return
        if not self.endx_legal:
            self._flag(Reason.SCOPE,
                       "ENDX closes a transform scope opened outside the "
                       f"innermost {_name_of(self.top_scope)} scope")
        self._xforms.pop()


# ---------------------------------------------------------------------------
# the byte-level cascade, shared by masks and by the exclusion accounting


@lru_cache(maxsize=4096)
def _byte_cascade(key: StateKey, policy: LanguagePolicy,
                  level: str) -> tuple[tuple[Reason, np.ndarray], ...]:
    """Nested legal-byte sets, one per rule, in the frozen precedence order.

    Returns `((rule, the bytes surviving it), ...)`, starting from every byte.
    The difference between consecutive entries is what that rule removed, which
    is how the report attributes `q` to exactly one rule per symbol without ever
    summing overlapping marginals.

    `CONTROL` and `REPRESENTATION` are absent: both are symbol-level rules about
    how an alphabet *spells* a byte, and they are applied by the adapter below.

    A pure function of `(StateKey, policy, level)`, which is what makes the
    memoisation exact: those are precisely the facts the rules read, and a corpus
    of a quarter of a million byte positions visits a few dozen distinct keys.
    The returned arrays are shared and must not be mutated.
    """
    if level not in (CANONICAL, VM_SAFE):
        raise StateError(f"no legal-byte cascade for level {level!r}")
    canonical = level == CANONICAL
    rules = MASK_RULES if canonical else VM_SAFE_RULES
    out: list[tuple[Reason, np.ndarray]] = []

    def stage(reason: Reason, keep: np.ndarray) -> None:
        """Narrow the surviving byte set, or carry it through unchanged.

        A rule this level does not apply still gets an entry, so the byte
        cascade has one stage per frozen rule at every level and the reported
        decomposition lines up column for column between the two masks.
        """
        surviving = out[-1][1] if out else _ALL_BYTES
        if reason in rules:
            surviving = np.intersect1d(surviving, keep, assume_unique=False)
        out.append((reason, surviving))

    if key.done:
        for reason in MASK_RULES:
            out.append((reason, np.empty(0, dtype=np.int64)))
        return tuple(out)

    if key.phase is Phase.OPERAND:
        kind = Kind(key.kind)
        domain = CANONICAL_VALUES[kind] if canonical else VM_SAFE_VALUES[kind]
        stage(Reason.STATIC_GRID, _ALL_BYTES)   # every byte is a legal operand
        stage(Reason.OPCODE_POLICY, _ALL_BYTES)
        stage(Reason.OPERAND_DOMAIN, domain)
        stage(Reason.SCOPE, _ALL_BYTES)
        stage(Reason.DEPTH, _ALL_BYTES)
        stage(Reason.HALT, _ALL_BYTES)
        return tuple(out)

    stage(Reason.STATIC_GRID, _OPCODE_BYTE_ARRAY)
    allowed = policy.opcodes if canonical else VM_SAFE_OPCODES
    stage(Reason.OPCODE_POLICY,
          np.array(sorted(int(op) for op in allowed), dtype=np.int64))
    stage(Reason.OPERAND_DOMAIN, _ALL_BYTES)

    closers: list[int] = []
    if not (key.endrep_canonical if canonical else key.endrep_vm):
        closers.append(int(Op.ENDREP))
    if not (key.endx_canonical if canonical else key.endx_vm):
        closers.append(int(Op.ENDX))
    stage(Reason.SCOPE, np.setdiff1d(_ALL_BYTES, np.array(closers, dtype=np.int64)))

    openers: list[int] = []
    repeat_full = key.repeat_full if canonical else key.repeat_vm_full
    xform_full = key.xform_full if canonical else key.xform_vm_full
    if repeat_full:
        openers += [int(Op.REPEAT), int(Op.REPEATX)]
    if xform_full:
        openers += [int(Op.XFORM), int(Op.REPEATX)]
    stage(Reason.DEPTH, np.setdiff1d(_ALL_BYTES, np.array(openers, dtype=np.int64)))

    stage(Reason.HALT,
          _ALL_BYTES if key.halt_legal
          else np.setdiff1d(_ALL_BYTES, np.array([int(Op.HALT)], dtype=np.int64)))
    return tuple(out)


# ---------------------------------------------------------------------------
# program-level verdict


@dataclass(frozen=True)
class StateTrace:
    """One program's state at every byte, plus its canonical verdict.

    `snapshots[i]` is the state **before** byte `i`, which is the state a model
    predicting that byte would be conditioned on -- a function of the prefix
    alone, never of the target. `final` is the state after the last byte.
    """

    policy: LanguagePolicy
    length: int
    snapshots: tuple[StateSnapshot, ...]
    final: StateSnapshot
    violations: tuple[Violation, ...]
    halted: bool
    first_exclusions: tuple[Reason | None, ...] = ()

    @property
    def canonical(self) -> bool:
        """Canonical membership: no violation anywhere, and it halted."""
        return not self.violations and self.halted

    @property
    def headline(self) -> Violation | None:
        return self.violations[0] if self.violations else None

    def reasons(self) -> tuple[Reason, ...]:
        return tuple(dict.fromkeys(v.reason for v in self.violations))


def trace(program: bytes, policy: LanguagePolicy) -> StateTrace:
    """Walk a whole program once; keep the state before every byte.

    Two verdicts a mask cannot produce are added at the end: a stream that ran
    out mid-instruction is `TRUNCATED`, and one that never reached a `HALT` is
    `NO_HALT`. Both are `VM.run`'s own outcomes for the same streams.
    """
    state = GrammarState(policy)
    snapshots = []
    for byte in program:
        snapshots.append(state.snapshot())
        state.step(byte)
    final = state.snapshot()
    violations = list(state.violations)
    if state.pending and state.terminal is None:
        violations.append(Violation(Reason.TRUNCATED, len(program),
                                    f"{state.pending} operand byte(s) owed", True))
    elif not state.done:
        violations.append(Violation(Reason.NO_HALT, len(program),
                                    "the stream ended while live", False))
    return StateTrace(policy=policy, length=len(program),
                      snapshots=tuple(snapshots), final=final,
                      violations=tuple(violations), halted=state.halted,
                      first_exclusions=state.first_exclusions)


# ---------------------------------------------------------------------------
# layer 3: the codec adapter -- symbols, not bytes


class SymbolCursor:
    """Where inside one bytecode byte an alphabet currently sits.

    **Representation state, not ISA state.** `dm/isa/state.py`'s first two
    layers know nothing about it, and the split matters: the `bit` alphabet's
    partial byte and the `token` alphabets' opcode/value regions are facts about
    the spelling, while `scope_stack` and `halt_legal` are facts about the
    program. Conflating them would put a codec's own structure inside the
    language's definition.
    """

    __slots__ = ("filled", "prefix", "width")

    def __init__(self, width: int) -> None:
        self.width = width      # symbols per bytecode byte
        self.filled = 0         # symbols of the current byte already committed
        self.prefix = 0         # their value, MSB first (the `bit` alphabet)

    @property
    def partial(self) -> bool:
        return self.filled > 0

    def reset(self) -> None:
        self.filled = self.prefix = 0

    def key(self) -> tuple[int, int]:
        return (self.filled, self.prefix)


class Support(NamedTuple):
    """Legal symbols at one position, and what each rule removed.

    `removed` is in the frozen precedence order and is *mutually exclusive*: a
    symbol excluded by several rules is credited to the first, so the parts sum
    to the union exactly once.
    """

    legal: np.ndarray                              # symbol ids, ascending
    removed: tuple[tuple[Reason, np.ndarray], ...]
    row: np.ndarray                                # (vocab,) bool, for the sampler

    @property
    def empty(self) -> bool:
        return not len(self.legal)


class _Alphabet:
    """How one codec spells a byte, and which symbols can spell a byte set.

    Two questions, and they are the whole of the codec-facing contract: *which
    byte does this symbol commit* (`consume`), and *which symbols could spell a
    byte from this set here* (`support`). `strict` separates the two masks --
    canonical support admits only the spelling the alphabet reserves for this
    position, VM-safe support admits any symbol that decodes into the set.

    A `RelativeCodec` delegates to its inner alphabet unchanged. The relative
    rewrite touches `Kind.COORD` *values* only, leaves every opcode byte and
    every instruction boundary alone, and every `COORD` domain is the full byte
    -- so the grammar-level legal set is identical in both views. Execution state
    is a different matter and needs the pen accumulator (S4).
    """

    def __init__(self, codec: Codec) -> None:
        inner = codec.inner if isinstance(codec, RelativeCodec) else codec
        if not isinstance(inner, (ByteCodec, TokenCodec, BitCodec)):
            raise StateError(  # pragma: no cover - a new codec states its own rule
                f"no legality rule for {type(inner).__name__}"
            )
        self.codec, self.inner = codec, inner
        self.vocab = codec.vocab_size
        self.width = codec.stride

    def support(self, byte_values: np.ndarray, key: StateKey,
                cursor: SymbolCursor, strict: bool) -> np.ndarray:
        """`(vocab,)` bool: this symbol can spell a byte in `byte_values` here."""
        inner = self.inner
        allowed = np.zeros(self.vocab, dtype=bool)
        if not len(byte_values):
            return allowed

        if isinstance(inner, ByteCodec):
            # Untyped and stride 1: one symbol per byte, and no spelling choice
            # to be canonical or otherwise about.
            allowed[N_SPECIAL + byte_values] = True
            return allowed

        if isinstance(inner, BitCodec):
            # A bit is legal iff at least one legal complete byte carries this
            # bit prefix. The partial byte lives in the cursor, never in the ISA.
            if cursor.filled:
                byte_values = byte_values[
                    (byte_values >> (8 - cursor.filled)) == cursor.prefix
                ]
            shift = 7 - cursor.filled
            for bit in (0, 1):
                if np.any(((byte_values >> shift) & 1) == bit):
                    allowed[N_SPECIAL + bit] = True
            return allowed

        at_opcode = key.phase is Phase.OPCODE
        if not strict or at_opcode:
            # The opcode region spells an opcode byte. At an *operand* position it
            # also decodes to that byte value (`TokenCodec.decode`), which is a
            # legal operand byte spelled in the wrong region -- admitted VM-safe,
            # refused canonically.
            for byte in byte_values.tolist():
                symbol = inner.opcode_symbol(byte)
                if symbol is not None:
                    allowed[symbol] = True
        if not strict or not at_opcode:
            kinds = range(inner.n_kinds) if inner.typed_operands else (0,)
            if inner.typed_operands and strict and key.kind >= 0:
                if key.kind >= inner.n_kinds:
                    raise StateError(
                        f"{inner.name} has {inner.n_kinds} operand alphabets and "
                        f"this position expects {Kind(key.kind).name}; a "
                        "checkpoint of that width cannot spell this corpus"
                    )
                kinds = (key.kind,)
            for kind in kinds:
                for byte in byte_values.tolist():
                    symbol = inner.value_symbol(Kind(kind), byte)
                    if symbol is not None:
                        allowed[symbol] = True
        return allowed

    def consume(self, symbol: int, cursor: SymbolCursor) -> int | None:
        """Commit one symbol; return the completed bytecode byte, or None.

        `None` means "no byte yet": either the alphabet is mid-byte (`bit`) or the
        symbol carries no bytecode at all (`PAD`, `BOS`, an opcode slot no opcode
        occupies), which is the answer `decode` and `HaltMonitor` already give it.
        """
        inner = self.inner
        if isinstance(inner, (ByteCodec, TokenCodec)):
            return inner.symbol_to_byte(symbol)
        if isinstance(inner, BitCodec):
            if symbol < N_SPECIAL:
                # A control symbol inside a byte is not a bit. The partial byte is
                # abandoned, exactly as `decode` drops it.
                cursor.reset()
                return None
            cursor.prefix = (cursor.prefix << 1) | (symbol - N_SPECIAL)
            cursor.filled += 1
            if cursor.filled < 8:
                return None
            byte = cursor.prefix
            cursor.reset()
            return byte
        return inner.symbol_to_byte(symbol)


class SupportTable:
    """Legal-symbol support for one codec, one policy and one mask level.

    Memoised on `(StateKey, cursor)`, because the corpora visit a handful of
    distinct states and every consumer -- the sampler, the teacher-forced audit
    and the tests -- wants the identical set.
    """

    def __init__(self, codec: Codec, policy: LanguagePolicy,
                 level: str = CANONICAL) -> None:
        if level not in (CANONICAL, VM_SAFE):
            raise StateError(
                f"level {level!r} has no support set; {RAW!r} means 'do not mask'"
            )
        self.alphabet = _Alphabet(codec)
        self.codec, self.policy, self.level = codec, policy, level
        self._cache: dict[tuple, Support] = {}
        #: Scratch space for consumers that want a derived form of the same
        #: support -- `dm.eval.state_support` keeps its selection matrices here so
        #: they are memoised on exactly the key the support is memoised on, rather
        #: than in a parallel cache that could fall out of step with this one.
        self.matrices: dict = {}

    @property
    def vocab_size(self) -> int:
        return self.alphabet.vocab

    def of(self, state: GrammarState, cursor: SymbolCursor) -> Support:
        return self.of_key(state.key(), cursor)

    def of_key(self, state_key: StateKey, cursor: SymbolCursor) -> Support:
        """Support for a state named by its key, which is all the rules read."""
        key = (state_key, cursor.key())
        hit = self._cache.get(key)
        if hit is None:
            hit = self._build(state_key, cursor)
            self._cache[key] = hit
        return hit

    def _build(self, state_key: StateKey, cursor: SymbolCursor) -> Support:
        """One `Support`, with every removal credited to exactly one rule.

        The frozen precedence is applied by *order of application*: each rule
        takes the symbols still alive that it excludes, and the ones already gone
        stay credited where they were. So the parts partition the union and can
        be summed exactly once (`docs/state-freeze.md` §5).
        """
        alphabet = self.alphabet
        if state_key.done:
            # A finished row has no next symbol. `StateMonitor.allowed` never
            # masks such a row -- it would be all `-inf` -- so this exists for
            # reports that ask a stopped row what it could have emitted: nothing.
            return Support(legal=np.empty(0, dtype=np.int64), removed=(),
                           row=np.zeros(alphabet.vocab, dtype=bool))

        stages = _byte_cascade(state_key, self.policy, self.level)
        strict = self.level == CANONICAL
        alive = np.ones(alphabet.vocab, dtype=bool)
        removed: list[tuple[Reason, np.ndarray]] = []

        def credit(reason: Reason, keep: np.ndarray) -> None:
            nonlocal alive
            kill = alive & ~keep
            if kill.any():
                removed.append((reason, np.flatnonzero(kill).astype(np.int64)))
                alive = alive & ~kill

        # Rule 1, `control`: symbols that spell no bytecode byte in this alphabet.
        credit(Reason.CONTROL,
               alphabet.support(_ALL_BYTES, state_key, cursor, strict=False))
        for reason, byte_values in stages:
            # Byte-level rules are credited against *any* spelling, so that a
            # symbol excluded both by its byte and by its region is credited to
            # the byte rule -- which is what the frozen precedence says.
            credit(reason, alphabet.support(byte_values, state_key, cursor,
                                            strict=False))
            if reason is Reason.STATIC_GRID and strict:
                # Rule 3, `representation`, sits between the static grid and the
                # policy: a symbol that spells a legal byte in the wrong region of
                # the alphabet is a representation fault and not a VM fault, so it
                # gets its own row and is never counted as a fault the VM reports.
                credit(Reason.REPRESENTATION,
                       alphabet.support(byte_values, state_key, cursor, strict=True))
        return Support(legal=np.flatnonzero(alive).astype(np.int64),
                       removed=tuple(removed), row=alive)


class StateMonitor:
    """Row-wise parse state, halting and legal-output support for `generate`.

    A drop-in for `dm.isa.codec.HaltMonitor` -- same `stride`/`step` contract and
    the same `done`/`pending` fields -- that additionally answers `allowed()`
    with a per-row mask over the vocabulary.

    **`stride` is 1 for every alphabet, deliberately.** `HaltMonitor` decides
    once per bytecode byte because that is the smallest unit at which a stream
    can be *parsed*; a mask has to be applied to every symbol, including each of
    the eight bits of a byte. So this monitor consumes one symbol column per
    call, carries the partial byte in a `SymbolCursor`, and advances the grammar
    only when a byte completes. A prompt that ends mid-instruction, or mid-byte,
    is therefore not a hazard here: it has grammar state and symbol state and has
    not executed.

    `mask_mode="raw"` builds the same states and reports the same pressure while
    masking nothing, which is what makes the three decoder cells differ in one
    operation.
    """

    __slots__ = ("_absorbed", "_empty", "_positions", "_pressure", "_symbol_exclusions",
                 "_tables", "codec", "cursors", "done", "mask_mode", "pending",
                 "policy", "states", "stride")

    def __init__(self, codec, rows: int, policy: LanguagePolicy,
                 mask_mode: str = CANONICAL) -> None:
        if mask_mode not in LEVELS:
            raise StateError(f"mask_mode must be one of {LEVELS}, got {mask_mode!r}")
        self.codec, self.stride, self.policy, self.mask_mode = codec, 1, policy, mask_mode
        self.states = [GrammarState(policy) for _ in range(rows)]
        self.cursors = [SymbolCursor(codec.stride) for _ in range(rows)]
        self.done = np.zeros(rows, dtype=bool)
        self.pending = np.zeros(rows, dtype=np.int16)
        self._tables = {
            level: SupportTable(codec, policy, level) for level in (VM_SAFE, CANONICAL)
        }
        self._pressure = {reason: 0 for reason in MASK_RULES}
        self._absorbed = np.zeros(rows, dtype=bool)
        self._empty = 0
        self._positions = 0
        self._symbol_exclusions: list[list[Reason | None]] = [[] for _ in range(rows)]

    # -- the HaltMonitor contract ------------------------------------------

    def step(self, chunk: np.ndarray) -> np.ndarray:
        """Consume one symbol per row; return the rows that have stopped.

        `chunk` is `(1, rows)` of symbol ids, matching `HaltProtocol` at
        `stride = 1`.
        """
        column = np.asarray(chunk).reshape(-1)
        if len(column) != len(self.states):
            raise StateError(
                f"{len(column)} symbols for {len(self.states)} rows; the monitor "
                "keeps one parse state per row and cannot realign them"
            )
        for row, symbol in enumerate(column.tolist()):
            state = self.states[row]
            symbol = int(symbol)
            reason = self._symbol_first_exclusion(row, symbol)
            self._symbol_exclusions[row].append(reason)
            if reason is Reason.REPRESENTATION:
                state.record_symbol_violation(
                    reason,
                    f"symbol {symbol} spells a legal byte in a non-canonical "
                    "alphabet region",
                )
            byte = self.alphabet.consume(int(symbol), self.cursors[row])
            if byte is None:
                continue
            state.step(byte)
            self.pending[row] = state.pending
            self.done[row] = state.done
        return self.done

    @property
    def alphabet(self) -> _Alphabet:
        return self._tables[CANONICAL].alphabet

    # -- the new half -------------------------------------------------------

    def support(self, row: int) -> Support:
        level = CANONICAL if self.mask_mode == RAW else self.mask_mode
        return self._tables[level].of(self.states[row], self.cursors[row])

    def _symbol_first_exclusion(self, row: int, symbol: int) -> Reason | None:
        state = self.states[row]
        if state.done:
            return None
        table = self._tables[CANONICAL]
        if symbol < 0 or symbol >= table.vocab_size:
            return Reason.CONTROL
        support = table.of(state, self.cursors[row])
        if support.row[symbol]:
            return None
        for reason, symbols in support.removed:
            if symbol in symbols:
                return reason
        return Reason.CONTROL

    def allowed(self) -> np.ndarray:
        """`(rows, vocab)` bool -- which symbols each row may emit next.

        Rows that have stopped are all-True. They are overwritten with `PAD` by
        the sampler anyway, and a row of `-inf` logits would make the softmax
        NaN and the multinomial backend-defined (`docs/traps.md`).

        A **live** row with an empty legal set is an instrument error and is
        raised as one: silently falling back to the raw distribution would turn a
        structural claim into a best-effort sampler (`docs/state.md`).
        """
        vocab = self._tables[CANONICAL].vocab_size
        out = np.ones((len(self.states), vocab), dtype=bool)
        masking = self.mask_mode != RAW
        for row, state in enumerate(self.states):
            if state.done:
                continue
            self._positions += 1
            # Computed in the raw cell too, and thrown away. It is memoised, so
            # the cost is a dictionary lookup, and it buys the raw cell the same
            # pressure columns as the masked ones -- which is what makes the
            # three-cell table a comparison rather than three reports.
            support = self.support(row)
            if support.empty:
                self._empty += 1
                if masking:
                    raise StateError(
                        f"row {row} has no legal {self.mask_mode} symbol at byte "
                        f"offset {state.offset} (status {state.status.name}); an "
                        "empty legal set is an instrument error, not a licence to "
                        "unmask"
                    )
            for reason, symbols in support.removed:
                if len(symbols):
                    self._pressure[reason] += 1
            if not state.canonical:
                self._absorbed[row] = True
            if masking and not support.empty:
                out[row] = support.row
        return out

    def prime(self, symbols: Sequence[int] | np.ndarray, row: int = 0) -> None:
        """Feed an existing prefix through one row's state.

        For a caller that builds a prompt itself. `generate` primes through
        `step`, which is the same path.
        """
        for symbol in np.asarray(symbols).reshape(-1).tolist():
            symbol = int(symbol)
            reason = self._symbol_first_exclusion(row, symbol)
            self._symbol_exclusions[row].append(reason)
            if reason is Reason.REPRESENTATION:
                self.states[row].record_symbol_violation(
                    reason,
                    f"symbol {symbol} spells a legal byte in a non-canonical "
                    "alphabet region",
                )
            byte = self.alphabet.consume(symbol, self.cursors[row])
            if byte is not None:
                self.states[row].step(byte)
        self.pending[row] = self.states[row].pending
        self.done[row] = self.states[row].done

    def pressure(self) -> dict:
        """Mask pressure: how often each rule removed support, and from where.

        Positions rather than probability mass -- mass needs the logits and is
        `dm/eval/state_support.py`'s job. `absorbed` counts rows whose prefix had
        already left the canonical language when the mask was applied, which is
        the declared absorbing case and must be published beside every masked
        cell.
        """
        return {
            "mask_mode": self.mask_mode,
            "positions_masked": self._positions,
            "rule_positions": {r.value: self._pressure[r] for r in MASK_RULES},
            "absorbed_rows": int(self._absorbed.sum()),
            "empty_support_events": self._empty,
        }

    @property
    def diagnostic_done(self) -> np.ndarray:
        """Rows whose grammar observer has no further prefix to observe."""
        return np.array([state.done for state in self.states], dtype=bool)

    @property
    def first_exclusions(self) -> tuple[tuple[Reason | None, ...], ...]:
        return tuple(tuple(row) for row in self._symbol_exclusions)

    def verdicts(self) -> list[StateTrace]:
        """One per row: the canonical verdict of what that row actually emitted."""
        return [
            StateTrace(policy=self.policy, length=state.offset,
                       snapshots=(), final=state.snapshot(),
                       violations=state.violations, halted=state.halted,
                       first_exclusions=state.first_exclusions)
            for state in self.states
        ]


class LegacyStateMonitor:
    """The frozen raw sampler plus a grammar-only diagnostic observer.

    `HaltMonitor` owns termination and is deliberately called on the exact
    legacy stride. `StateMonitor` observes every emitted symbol for pressure,
    representation faults and exclusive causes, but its terminal grammar state
    cannot stop the row. This is the seam that keeps raw generation faithful to
    the old sampler while still making the audit visible.
    """

    __slots__ = ("_legacy", "_observer", "codec", "cursors", "done", "pending",
                 "policy", "states", "stride")

    def __init__(self, codec: Codec, rows: int, policy: LanguagePolicy) -> None:
        self.codec = codec
        self.policy = policy
        self._legacy = HaltMonitor(codec, rows)
        self._observer = StateMonitor(codec, rows, policy, mask_mode=RAW)
        self.stride = self._legacy.stride
        self.done = self._legacy.done
        self.pending = self._legacy.pending
        self.states = self._observer.states
        self.cursors = self._observer.cursors

    def step(self, chunk: np.ndarray) -> np.ndarray:
        chunk = np.asarray(chunk)
        if chunk.shape[0] != self.stride:
            raise StateError(
                f"legacy monitor expected {self.stride} symbols per byte, got "
                f"{chunk.shape[0]}"
            )
        # Observe before termination is applied to the next model step. The
        # legacy monitor still receives the original chunk unchanged.
        for symbol_row in chunk:
            self._observer.step(symbol_row.reshape(1, -1))
        self._legacy.step(chunk)
        return self.done

    def allowed(self) -> np.ndarray:
        # Compute diagnostic pressure, but return an all-true row: raw means the
        # original sampler's distribution, with no legality mask applied.
        self._observer.allowed()
        return np.ones((len(self.states), self.codec.vocab_size), dtype=bool)

    def support(self, row: int) -> Support:
        return self._observer.support(row)

    def pressure(self) -> dict:
        result = self._observer.pressure()
        result["termination"] = "legacy_halt_monitor"
        return result

    @property
    def diagnostic_done(self) -> np.ndarray:
        """Rows whose diagnostic grammar trace is terminal.

        Raw generation still terminates only through the legacy monitor's
        ``done`` array.  Support accounting, however, has no meaningful next
        grammar state after the observer reaches a terminal fault; exposing the
        observer's state keeps those discarded logits out of generated-prefix
        legality mass without changing a single raw sampling decision.
        """
        return self._observer.diagnostic_done

    @property
    def first_exclusions(self) -> tuple[tuple[Reason | None, ...], ...]:
        return self._observer.first_exclusions

    def verdicts(self) -> list[StateTrace]:
        return self._observer.verdicts()


__all__ = [
    "CALL_POLICIES",
    "CANONICAL",
    "CANONICAL_VALUES",
    "CLOSER_OF",
    "LEVELS",
    "MASK_RULES",
    "OPCODE_BYTES",
    "RAW",
    "VM_SAFE",
    "VM_SAFE_OPCODES",
    "GrammarState",
    "LanguagePolicy",
    "LegacyStateMonitor",
    "Phase",
    "Reason",
    "StateError",
    "StateKey",
    "StateMonitor",
    "StateSnapshot",
    "StateTrace",
    "Status",
    "Support",
    "SupportTable",
    "SymbolCursor",
    "Violation",
    "trace",
]
