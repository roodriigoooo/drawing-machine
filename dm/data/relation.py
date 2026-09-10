"""Direction 4's traced corpus: flat bytecode beside the copy relations inside it.

`dm.isa.unroll.unroll` answers "what does this structured program draw" and
throws away *why*. Direction 4 needs both halves at once: the flat L0 bytes a
model trains on, and the exact `COPY(source span, affine step, total count)`
actions that would reproduce the relation-bearing part of them. So this Module
adds `unroll_with_trace`, whose first return value is byte-for-byte the existing
unroller's and whose second is the provenance that unroller discards.

**Provenance, never detection.** `dm.eval.repeats` folds a flat program to
*infer* repeats and `dm.eval.recovery` scores that inference. Both are good
instruments and both are inferences. Here the structured source is in hand, so
every span and every transform is known by construction -- and then checked
against the flat bytes it claims to describe (`verify_trace`), so an arithmetic
slip fails the build instead of annotating the wrong region.

**Model-blind, and that is a rule** (`docs/copy-relation.md` §7 invariant 17).
Nothing here loads a checkpoint, reads a logit or consults a score. Candidate
caps, value supports and rejection rules are constants below; a case outside them
fails *acceptance*, and no case is ever removed after a model has scored it.

**Equivalence is retained, not collapsed.** Several distinct actions can append
byte-identical continuations -- a motif symmetric under a mirror admits two D4
elements, and a body that is itself repetitive admits two source spans. The
generator's own provenance label is one arbitrary member of that set. Supervising
it alone would turn an annotation choice into a model error, so
`equivalent_actions` re-derives the whole set from the bytes by *solving* for the
transform rather than by searching a Cartesian product of every signed byte
(`docs/copy-relation.md` §3.2).

The three layers are deliberately separate:

    unroll_with_trace   structured program -> flat bytes + scope provenance
    actions_of          scope provenance   -> the canonical left-to-right schedule
    equivalent_actions  flat bytes         -> every action that produces one target

The first is generator truth, the second is a schedule over it, and the third is
a fact about the bytes that does not know the generator exists. Acceptance
requires all three to agree.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from bisect import bisect_left
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from functools import lru_cache
from itertools import pairwise
from pathlib import Path

from ..isa.asm import Instr, assemble
from ..isa.spec import CANVAS, MAX_REPEAT_DEPTH, ISAError, Kind, Op, spec_for
from ..isa.transform import D4, D4_ORDER, IDENTITY, Transform
from ..isa.unroll import MAX_EXPANSION, _chunks, _matching, unroll
from ..relation.candidates import candidate_spans as _runtime_candidate_spans
from ..relation.spec import (
    COUNT_SUPPORT,
    D4_SUPPORT,
    RUNTIME_BOUNDS,
    TRANSLATION_SUPPORT,
    CopyActionKey,
)
from ..relation.transducer import TransducerFault
from ..vm.interp import DEFAULT_FUEL, VM
from ..vm.render import to_svg
from . import composed, synthetic
from .augment import Affine, apply
from .feedback_audit import _ARM_FIELDS as _VM_ARM_FIELDS
from .feedback_audit import AUDIT_SCHEMA, vm_census
from .fingerprint import fingerprint

# ---------------------------------------------------------------------------
# model-blind caps and supports
#
# Every constant here is frozen *before* any model exists and is serialised into
# `docs/copy-relation-protocol-v0.json` by `dm.eval.relation_contract`. They are
# corpus-shape facts rather than thresholds, which is why they live beside the
# builder that has to honour them: a cap the enumerator does not apply is a cap
# that only exists in prose.

#: Scope opcodes a traced program may contain. `CALL` is excluded because the
#: learned-library tier has no flat expansion, and an unknown opener would be
#: silently copied through as a plain instruction.
SCOPE_OPCODES = frozenset({"REPEAT", "REPEATX", "XFORM"})
CLOSER_OPCODES = frozenset({"ENDREP", "ENDX"})

#: How many instructions may stand between a candidate source span's end and the
#: boundary the action is decided at.
#:
#: **Not zero.** An earlier revision froze `source_stop == boundary`, which is
#: cheap and covers every canonical action -- but it turns the mechanism
#: `docs/copy-relation.md` §3.1 asks for, a pointer at an arbitrary earlier
#: whole-instruction span, into a suffix copier. It also makes the R3 head's
#: `key_stop` projection constant across every candidate at a boundary, so 4,096
#: parameters would buy no discrimination at all. The endpoint is free again,
#: bounded by this cap so the candidate set stays `O(gap * cap)` per boundary
#: rather than `O(T * cap)`.
#:
#: A span that stops early and skips intervening bytes is admitted on its merits:
#: the estimand is exact equality of the target block, so an action that renders
#: it *is* correct however unusual the span it points at. Refusing such an action
#: would score a decoder wrong for being right.
MAX_SOURCE_GAP_INSTRUCTIONS = RUNTIME_BOUNDS.max_source_gap_instructions

#: Largest source span an action may name, in whole instructions and in bytes.
#:
#: Two caps rather than one because they bound different costs. The instruction
#: cap bounds how many `(start, stop)` candidate keys the R3 head has to score;
#: the byte cap bounds how much a single accepted action may append, which is
#: what keeps a copied block from consuming the whole `max_len` budget.
#:
#: **Set from a model-blind coverage audit, with headroom, and then enforced.**
#: The observed canonical sources at the frozen venue settings run to 5
#: instructions / 15 bytes on `synthetic_nested_repeat` and 54 instructions / 162
#: bytes on `composed_motif_relation` (`coverage_report`). The caps sit above
#: both so that acceptance rejects nothing on size at these settings, and
#: `coverage_report` re-checks 100% coverage on every build rather than trusting
#: that. A case above the cap fails acceptance; it is never dropped after a model
#: has scored it (`docs/copy-relation.md` §7 invariant 17).
#:
#: The byte cap is also the last span-length bin edge the contract freezes, so
#: every accepted span falls inside a bin by construction.
MAX_SOURCE_INSTRUCTIONS = RUNTIME_BOUNDS.max_source_instructions
MAX_SOURCE_BYTES = RUNTIME_BOUNDS.max_source_bytes

#: How many candidate source spans one boundary may present to the R3 head.
#:
#: `docs/copy-relation.md` §3.3 declares that the head's candidate cap is frozen
#: at R1 and recorded, so it is a constant here rather than a number the scorer
#: picks later. Set above the widest set the frozen corpus actually presents
#: (`CANDIDATE_SPAN_BOUND`), which is arithmetic about the caps rather than a
#: corpus that presented more would silently truncate the set the annotated
#: action lives in.
MAX_CANDIDATE_SPANS = 768

#: Upper edges of the span-length bins the R3 key embedding indexes, in bytes.
#:
#: **The edges, not just a count.** Declaring "eight bins" fixes a parameter
#: count and nothing else: without boundaries there is no way to check that a bin
#: is reachable, that every accepted source falls in one, or that two builds agree
#: on which bin a span lands in. They live beside the caps rather than in the
#: contract because the last edge *is* `MAX_SOURCE_BYTES` -- one number with one
#: owner -- and `length_bin_report` audits both on every build.
LENGTH_BIN_EDGES: tuple[int, ...] = (12, 21, 33, 48, 69, 99, 141, 192)

#: Smallest source span an action may name. An action over one two-byte
#: instruction replaces fewer sampling decisions than the action itself costs to
#: describe, so admitting it would inflate the copy-decision denominator with
#: cases no decoder should ever want. Frozen, not tuned.
MIN_SOURCE_INSTRUCTIONS = RUNTIME_BOUNDS.min_source_instructions
MIN_SOURCE_BYTES = RUNTIME_BOUNDS.min_source_bytes

#: A source span must carry at least one `COORD` operand.
#:
#: Not a convenience: `solve_step` recovers `(dx, dy)` from where a coordinate
#: lands, and a span of pure `WIDTH`/`FILL`/`CIRCLE` instructions constrains the
#: translation not at all -- every `(dx, dy)` in the support would "produce" the
#: target and the equivalence set would be the whole support. Such a span is
#: excluded from the candidate set, so the ambiguity never reaches supervision.
MIN_SOURCE_COORDS = RUNTIME_BOUNDS.min_source_coords

#: The frozen value supports the R3 factor heads range over.
#:
#: Factorised on purpose (`docs/copy-relation.md` §3.3): an unseen *combination*
#: of supported atoms stays representable while no unseen atomic value is ever
#: claimed. `COUNT_SUPPORT` starts at 2 because a count of 1 appends nothing and
#: is spelled `EMIT`.
# Re-exported from dm.relation.spec for R1 compatibility.  The pure runtime owns
# these execution values so inference need not import this corpus builder.

#: Translations an action may name, as an explicit sorted tuple rather than a
#: range, so the protocol serialises the exact set the enumerator tests
#: membership against.
#:
#: **Three values, not 256, and the head is why.** A factor head over every
#: signed byte would cost `2 * d_model * 256 = 65,536` parameters at `D=128` --
#: twice the entire 32,768 extension budget `docs/copy-relation.md` §2.4 allows,
#: before a single span key exists. The venues emit exactly these three
#: translations, so this is the support the corpus has and the support the head
#: can represent, and the two being the same is what makes "the action is
#: representable" a structural fact rather than a hope.

#: The widest translation the *audit* considers, which is every `i8`.
#:
#: The destroyed-relation control has to prove that no copy relation survives in
#: the bytes -- not merely that none survives that the head could have named. A
#: donor block related to its prefix by a translation of seven is still a
#: relation, and filing that row as a negative would be exactly the fault §4.3
#: forbids. So the audit widens the support and supervision does not.
AUDIT_TRANSLATIONS: tuple[int, ...] = tuple(range(-128, 128))


class TraceError(ValueError):
    """The structured program and its flat trace disagree.

    Raised rather than returned: a trace that does not describe its own bytes is
    a builder defect, and every caller here would have to turn a `None` back into
    exactly this exception.
    """


# ---------------------------------------------------------------------------
# the traced unroller


@dataclass(frozen=True)
class TracedScope:
    """One expanded scope, and where its images land in the flat trace.

    `start` is the byte offset of image 0. Image `k` occupies
    `[start + k*body_bytes, start + (k+1)*body_bytes)` -- exact because
    `dm.data.augment.apply` rewrites instructions in place and so preserves every
    instruction's size, and therefore the whole body's length.

    `has_reference` is what separates a relation the flat bytes *contain* from one
    they merely came from. `REPEAT`/`REPEATX` emit image 0 unchanged, so the
    reference span is in the trace and a decoder can point at it. `XFORM` emits
    only the transformed body, so its source is not in the trace at all and no
    `COPY` action can name it. Recorded either way, because the audit needs to
    know the scope existed.
    """

    kind: str
    depth: int
    step: Transform
    count: int
    start: int
    body_bytes: int
    has_reference: bool

    @property
    def stop(self) -> int:
        """One past the last byte of the last image."""
        return self.start + self.count * self.body_bytes

    def image(self, k: int) -> tuple[int, int]:
        """Byte range of the 0-based `k`-th image."""
        if not 0 <= k < self.count:
            raise ValueError(f"image {k} outside 0..{self.count - 1}")
        begin = self.start + k * self.body_bytes
        return begin, begin + self.body_bytes

    def rebased(self, offset: int, outer: Transform) -> TracedScope:
        """This scope as it appears inside an enclosing copy transformed by `outer`.

        Two changes and no others. The offset shifts, because `apply` preserves
        instruction sizes and therefore relative byte positions. The step
        **conjugates**: the copy holds `outer(body)`, whose images are
        `outer(step^k(body)) = (outer step outer^-1)^k (outer(body))`, so the
        relation a decoder would have to name inside the copy is
        `outer ∘ step ∘ outer^-1`. D4 is non-abelian, so leaving the step alone
        annotates a transform that does not relate the bytes it points at --
        which `verify_trace` catches, loudly, rather than shipping.
        """
        return replace(self, start=self.start + offset,
                       step=self.step.conjugate(outer))


@dataclass(frozen=True)
class Traced:
    """The flat trace and the scopes that produced it."""

    bytes: bytes
    scopes: tuple[TracedScope, ...]


def _steps_for(instr: Instr) -> tuple[Transform, ...] | None:
    """The per-image transforms a scope opener implies, or None if it faults.

    Lifted verbatim from `dm.isa.unroll._expand` so the two cannot drift: a
    second reading of the same rule is a second place for it to be wrong, and the
    byte-identity requirement makes any drift a silent corpus defect.
    """
    if instr.mnemonic == "XFORM":
        return (Transform.of(*instr.args),)
    count = instr.args[0]
    if count < 1:
        return None  # ZERO_REPEAT: a fault, not a drawing
    step = (Transform(IDENTITY.d4, *instr.args[1:])
            if instr.mnemonic == "REPEAT" else Transform.of(*instr.args[1:]))
    return tuple(step.power(k) for k in range(count))


def _expand_traced(chunks: list[tuple[Instr, bytes]],
                   depth: int) -> tuple[bytes, list[TracedScope]] | None:
    """`dm.isa.unroll._expand`, with the provenance it discards kept.

    A faithful mirror rather than a reimplementation: the rejection order, the
    depth bound and the `MAX_EXPANSION` check are in the same places, because the
    first return value has to equal `unroll`'s byte for byte and a bound applied
    one copy later would produce a longer program on exactly the inputs the bound
    exists for.
    """
    out = bytearray()
    scopes: list[TracedScope] = []
    index = 0
    while index < len(chunks):
        instr, raw = chunks[index]
        if instr.mnemonic in CLOSER_OPCODES:
            return None  # unmatched: the VM faults, and so does this
        if instr.mnemonic not in SCOPE_OPCODES:
            out += raw
            index += 1
            continue
        if depth >= MAX_REPEAT_DEPTH:
            return None  # the VM's DEPTH_OVERFLOW, refused for the same reason

        end = _matching(chunks, index)
        if end is None:
            return None  # unterminated, crossed, or closed by the wrong opcode
        expanded = _expand_traced(chunks[index + 1 : end - 1], depth + 1)
        if expanded is None:
            return None
        body, body_scopes = expanded

        steps = _steps_for(instr)
        if steps is None:
            return None

        base = len(out)
        for step in steps:
            copy = body if step.is_identity else apply(body, Affine.of(step))
            if copy is None:
                return None  # left the canvas: this copy is not an exact image
            offset = len(out)
            out += copy
            # The enclosing copy carries the whole inner structure with it, moved
            # and conjugated. Emitted before this scope's own record so the list
            # stays in flat-offset order for every nesting shape.
            scopes.extend(inner.rebased(offset, step) for inner in body_scopes)
            if len(out) > MAX_EXPANSION:
                return None

        scopes.append(TracedScope(
            kind=instr.mnemonic,
            depth=depth,
            # `steps[1]` is the per-iteration step for `REPEAT`/`REPEATX` and the
            # whole map for `XFORM`, which has exactly one image; taking
            # `steps[-1]` would read `step^(count-1)` on a repeat.
            step=steps[1] if len(steps) > 1 else steps[0],
            count=len(steps),
            start=base,
            body_bytes=len(body),
            has_reference=instr.mnemonic != "XFORM",
        ))
        index = end
    return bytes(out), scopes


def unroll_with_trace(program: bytes) -> Traced | None:
    """The flat L0 trace of `program` and its scope provenance, or None.

    `unroll_with_trace(p).bytes == unroll(p)` for every program either accepts,
    and both are `None` for every program either refuses. That equality is an
    acceptance clause (`accept_case`) and a test, not a comment: the existing
    unroller and the VM stay independent checks on this builder, which is the
    only reason a trace derived here can be trusted to describe the bytes a model
    actually trains on.

    Scopes come back in flat-offset order, innermost first within a copy.
    """
    try:
        chunks = _chunks(program)
    except ISAError:
        # `AsmError` on a truncated instruction, `UnknownOpcode` on an unknown
        # one; `unroll` swallows the same base class and returns None, and a
        # traced build that raised where the plain one returned would silently
        # build a different corpus.
        return None
    expanded = _expand_traced(chunks, depth=0)
    if expanded is None:
        return None
    flat, scopes = expanded
    return Traced(bytes=flat, scopes=tuple(scopes))


def verify_trace(flat: bytes, scopes: tuple[TracedScope, ...]) -> None:
    """Assert every annotated image really is image 0 under the annotated step.

    The whole direction rests on cutting at exactly the right byte offsets under
    exactly the right transform. This turns an off-by-one or a missed conjugation
    from a silent corpus defect into a build failure -- the same job
    `dm.data.feedback.verify_scopes` does for Direction 3, extended from a pure
    translation to the full group.

    `XFORM` scopes are checked for containment only: their reference is not in
    the trace, so there is nothing to compare image 0 against.
    """
    for scope in scopes:
        if scope.stop > len(flat):
            raise TraceError(
                f"{scope.kind} at {scope.start} claims bytes up to {scope.stop}, "
                f"past the {len(flat)}-byte trace"
            )
        if not scope.has_reference:
            continue
        start, stop = scope.image(0)
        reference = flat[start:stop]
        for k in range(1, scope.count):
            begin, end = scope.image(k)
            want = apply(reference, Affine.of(scope.step.power(k)))
            if want is None or flat[begin:end] != want:
                raise TraceError(
                    f"image {k} of the {scope.kind} at {scope.start} "
                    f"({begin}:{end}) is not image 0 under {scope.step!r}"
                )


# ---------------------------------------------------------------------------
# instruction boundaries


def boundaries(program: bytes) -> tuple[int, ...]:
    """Every canonical instruction boundary, `0` and `len(program)` included.

    An action may only name whole instructions and may only be queried where the
    decoder is between them; every span endpoint in this Module is an element of
    this tuple. Raises on a malformed program rather than returning a partial
    answer, because a boundary list that silently stops early would place a span
    endpoint mid-instruction.
    """
    return _prefix_index(program)[0]


@lru_cache(maxsize=8192)
def _prefix_index(program: bytes) -> tuple[tuple[int, ...], tuple[int, ...],
                                           tuple[int, ...], tuple[int, ...]]:
    """Boundaries, `HALT` offsets, `COORD` offsets and the opcode sequence, once.

    The free source endpoint multiplied the candidate set by the gap cap, and
    three per-span predicates -- "contains no `HALT`", "carries a coordinate" and
    "has the target block's skeleton" -- were each a walk over the span.
    Precomputing the first two as sorted offset lists turns them into a bisect,
    and precomputing the opcode sequence turns the third into a tuple-slice
    comparison in mark-index space. That is what keeps a corpus build at the
    frozen size in minutes rather than an hour. Cached on the program bytes
    because every caller here works one program at a time and asks several
    questions about it.
    """
    marks = [0]
    halts: list[int] = []
    coords: list[int] = []
    opcodes: list[int] = []
    pc = 0
    while pc < len(program):
        spec = spec_for(program[pc])
        opcodes.append(int(spec.op))
        if spec.op is Op.HALT:
            halts.append(pc)
        index = 0
        while index < len(spec.operands):
            if spec.operands[index] is Kind.COORD:
                # Pair starts only: a `COORD` operand is half of a point, and
                # counting both halves would make `MIN_SOURCE_COORDS` mean
                # something different from what its name says.
                coords.append(pc + 1 + index)
                index += 2
            else:
                index += 1
        pc += spec.size
        if pc > len(program):
            raise TraceError(f"truncated {spec.mnemonic} at byte {pc - spec.size}")
        marks.append(pc)
    return tuple(marks), tuple(halts), tuple(coords), tuple(opcodes)


def _halt_offsets(program: bytes) -> frozenset[int]:
    """Byte offsets of every `HALT`. A source span may not contain one."""
    return frozenset(_prefix_index(program)[1])


def _skeleton(program: bytes) -> tuple[int, ...]:
    """The opcode sequence. `apply` never changes it, so two blocks with
    different skeletons cannot be images of each other -- the cheapest possible
    rejection, and it runs before any group arithmetic."""
    return _prefix_index(program)[3]


# ---------------------------------------------------------------------------
# actions


@dataclass(frozen=True)
class CopyAction:
    """`COPY(source span, affine step, total count)` against one flat program.

    `total_count` includes the reference, so a count of 4 appends `step^1`,
    `step^2` and `step^3` -- the spelling `docs/copy-relation.md` §3.1 freezes.
    The target block is therefore `count - 1` images long and starts exactly at
    `boundary`, which is also `source_stop` for a canonical action and need not
    be for an equivalent one.
    """

    boundary: int
    source_start: int
    source_stop: int
    step: Transform
    total_count: int
    target_start: int
    target_stop: int

    @property
    def source_bytes(self) -> int:
        return self.source_stop - self.source_start

    @property
    def target_bytes(self) -> int:
        return self.target_stop - self.target_start

    def key(self) -> tuple:
        """A hashable identity for equivalence-set membership and reporting."""
        return (self.boundary, self.source_start, self.source_stop,
                self.step.d4.code, self.step.dx, self.step.dy, self.total_count)

    def runtime_key(self) -> CopyActionKey:
        """Drop corpus-only target/provenance fields for the R2 executor."""
        return CopyActionKey(
            boundary=self.boundary,
            source_start=self.source_start,
            source_stop=self.source_stop,
            step=self.step,
            total_count=self.total_count,
        )

    def render(self, flat: bytes) -> bytes | None:
        """The bytes this action appends, or None if any image leaves the canvas.

        Independent of the trace by construction: it reads the source out of the
        flat program and re-applies the group, so comparing it against the
        annotated target is a real check rather than a restatement.
        """
        source = flat[self.source_start:self.source_stop]
        out = bytearray()
        for k in range(1, self.total_count):
            image = apply(source, Affine.of(self.step.power(k)))
            if image is None:
                return None
            out += image
        return bytes(out)

    def in_support(self, translations: Sequence[int] = TRANSLATION_SUPPORT) -> bool:
        """Whether every factor lies in the supports the R3 heads span.

        `translations` widens for the audit and never for supervision; see
        `AUDIT_TRANSLATIONS`.
        """
        return (self.step.d4.code in D4_SUPPORT
                and self.step.dx in translations
                and self.step.dy in translations
                and self.total_count in COUNT_SUPPORT)


def actions_of(scopes: tuple[TracedScope, ...]) -> tuple[CopyAction, ...]:
    """The generator's own action per relation-bearing scope, in flat order.

    One action per scope, naming image 0 as the source and every later image as
    the target. `XFORM` scopes and single-image scopes contribute nothing: there
    is no reference in the trace to point at, and a count of 1 appends no bytes.

    This is provenance, not a schedule -- it includes actions whose boundary sits
    inside another action's target, which a decoder never reaches. `schedule`
    applies that filter, and keeps this function's output as the audit trail.
    """
    out: list[CopyAction] = []
    for scope in scopes:
        if not scope.has_reference or scope.count < 2:
            continue
        source_start, source_stop = scope.image(0)
        out.append(CopyAction(
            boundary=source_stop,
            source_start=source_start,
            source_stop=source_stop,
            step=scope.step,
            total_count=scope.count,
            target_start=source_stop,
            target_stop=scope.stop,
        ))
    return tuple(sorted(out, key=lambda action: (action.target_start,
                                                 action.target_stop)))


def schedule(actions: tuple[CopyAction, ...]) -> tuple[CopyAction, ...]:
    """The left-to-right, non-overlapping actions a `predicted_copy` decode reaches.

    **An action whose boundary lies strictly inside another action's target block
    is unreachable and is dropped.** After a `COPY`, the decoder appends the whole
    target deterministically and resumes at its end, so it never stands at a
    boundary inside it and never makes a decision there. Those relations are real
    -- a nested inner repeat inside an outer copy is copied along with everything
    else -- and they stay in `actions_of` for the audit; what they are not is a
    position where a gate has a target.

    Outermost-first is the consequence, not a preference: an enclosing scope's
    reference *contains* every inner scope, so an inner action's boundary is
    below the enclosing action's target start and survives, while the inner
    actions replicated into copies 1..n-1 do not.
    """
    out: list[CopyAction] = []
    covered_to = -1
    for action in sorted(actions, key=lambda a: (a.target_start, -a.target_bytes)):
        if action.boundary < covered_to:
            continue
        if action.target_start < covered_to:
            continue
        out.append(action)
        covered_to = action.target_stop
    return tuple(out)


def covered_boundaries(actions: tuple[CopyAction, ...]) -> frozenset[int]:
    """Boundaries a scheduled action appends past, which a decode never queries.

    The endpoints are *not* covered: the target's first byte is the boundary the
    action itself is decided at, and its last byte is where decoding resumes.
    Only the boundaries strictly between them are unreachable, and they are the
    ones `L_gate` must not average over.
    """
    out: set[int] = set()
    for action in actions:
        out.update(range(action.target_start + 1, action.target_stop))
    return frozenset(out)


# ---------------------------------------------------------------------------
# equivalence: every action that produces one target block


def _coord_pairs(program: bytes, limit: int = 3) -> list[tuple[int, int, int]]:
    """Up to `limit` `(offset, x, y)` points, by `Kind` and never positionally.

    The same rule `dm.data.augment.apply` and `dm.data.composed.bounds` follow: a
    `CIRCLE`'s radius sits where a `MOVE`'s `x` sits, and solving a transform
    from it would produce a map that relates nothing.
    """
    _, _, coords, _ = _prefix_index(program)
    return [(at, program[at], program[at + 1]) for at in coords[:limit]]


def _first_coord(program: bytes) -> tuple[int, int, int] | None:
    """`(byte offset of x, x, y)` of the first `COORD` pair, or None."""
    found = _coord_pairs(program, limit=1)
    return found[0] if found else None


def _candidate_d4(source: bytes, image: bytes) -> tuple[int, ...]:
    """Which D4 codes could carry `source` onto `image`, from one difference vector.

    **A translation cancels in a difference**, so `D.linear(p1 - p0)` has to equal
    `q1 - q0` for the two blocks' corresponding points. Usually one or two of the
    eight elements survive that, which is what turns `solve_step` from eight full
    `apply` calls into one or two. When no pair of points has a non-zero
    difference -- a block whose coordinates all coincide -- nothing is learned and
    all eight are returned, so the answer is never narrowed by an accident.
    """
    left = _coord_pairs(source)
    right = _coord_pairs(image)
    if len(left) != len(right) or len(left) < 2:
        return D4_SUPPORT
    for (first, second), (image_first, image_second) in zip(pairwise(left),
                                                            pairwise(right)):
        delta = (second[1] - first[1], second[2] - first[2])
        if delta == (0, 0):
            continue
        want = (image_second[1] - image_first[1], image_second[2] - image_first[2])
        return tuple(code for code in D4_SUPPORT
                     if D4.of(code).linear(*delta) == want)
    return D4_SUPPORT


def solve_step(source: bytes, image: bytes) -> tuple[Transform, ...]:
    """Every element of D4 ⋉ Z² carrying `source` onto `image`, exactly.

    **Solved, not searched.** The group has 8 · 256 · 256 elements and §3.2
    forbids materialising that product. The linear part is read off a difference
    of two points, which a translation cannot affect; the translation then follows
    by subtraction from one point; and the candidate is verified by re-applying
    the whole map to the whole span. So the answer is exact, and the cost is one
    or two `apply` calls rather than half a million.

    Several elements can succeed, and that is the point: a motif symmetric under
    a mirror is carried onto the same bytes by two different transforms, and
    calling one of them "the" answer would invent a model error out of an
    annotation choice.
    """
    if len(source) != len(image) or _skeleton(source) != _skeleton(image):
        return ()
    anchor = _first_coord(source)
    if anchor is None:
        return ()
    at, x, y = anchor
    target_x, target_y = image[at], image[at + 1]
    out: list[Transform] = []
    for code in _candidate_d4(source, image):
        d4 = D4.of(code)
        px, py = d4.point(x, y)
        candidate = Transform(d4, target_x - px, target_y - py)
        if apply(source, Affine.of(candidate)) == image:
            out.append(candidate)
    return tuple(out)


def _eligible_spans(flat: bytes, marks: tuple[int, ...], boundary: int,
                    halts: frozenset[int]) -> list[tuple[int, int]]:
    """Candidate source spans in the prefix, under the frozen caps.

    Bounded five ways -- the gap cap, instruction count, byte length, `HALT`
    exclusion and the coordinate rule -- and every bound is applied here rather
    than after scoring, so a case the caps exclude is a case the corpus rejects
    (`docs/copy-relation.md` §3.2). The inner walk breaks on the first span past
    either size cap because both grow monotonically as the start moves left.

    Ordered so that spans ending at the boundary come first. That is not a
    preference the enumerator acts on -- every candidate is verified against the
    bytes and the result is sorted -- but it keeps the common case at the front
    for a reader stepping through it.
    """
    if boundary not in marks:
        return []
    _, halt_offsets, coord_offsets, _ = _prefix_index(flat)
    query_index = marks.index(boundary)
    out: list[tuple[int, int]] = []
    for gap in range(MAX_SOURCE_GAP_INSTRUCTIONS + 1):
        stop_index = query_index - gap
        if stop_index < MIN_SOURCE_INSTRUCTIONS:
            break
        stop = marks[stop_index]
        for start_index in range(stop_index - MIN_SOURCE_INSTRUCTIONS, -1, -1):
            if stop_index - start_index > MAX_SOURCE_INSTRUCTIONS:
                break
            start = marks[start_index]
            if stop - start > MAX_SOURCE_BYTES:
                break
            if stop - start < MIN_SOURCE_BYTES:
                continue
            if bisect_left(halt_offsets, stop) > bisect_left(halt_offsets, start):
                continue
            if MIN_SOURCE_COORDS and bisect_left(coord_offsets, stop) - \
                    bisect_left(coord_offsets, start) < MIN_SOURCE_COORDS:
                continue
            out.append((start, stop))
    return out


def candidate_spans(flat: bytes, boundary: int) -> tuple[tuple[int, int], ...]:
    """The candidate source spans a decision at `boundary` may choose between.

    Public because the R3 head has to score exactly this set and the contract
    caps how large it may be; `candidate_span_report` audits the widest one the
    frozen corpus presents against `RELATION_MAX_CANDIDATE_SPANS`.
    """
    # The runtime primitive is the single owner of prefix parsing and span
    # bounds.  This compatibility boundary intentionally preserves the corpus
    # module's historical malformed-input result; corpus acceptance subsequently
    # validates well-formed programs independently.
    try:
        return _runtime_candidate_spans(flat, boundary)
    except TransducerFault:
        return ()

def _images(source: bytes, step: Transform, limit: int) -> list[bytes]:
    """`step^1 .. step^k` of `source`, stopping at `limit` bytes or the canvas edge.

    Built once and reused for every count in the support, because the images of a
    given `(source, step)` are a prefix chain: a count of 4 appends exactly what a
    count of 3 appends plus one more image. Recomputing them per count would run
    the group law `|COUNT_SUPPORT|` times over the same bytes.
    """
    out: list[bytes] = []
    current = Transform()
    while len(out) * len(source) < limit:
        current = current.then(step)
        image = apply(source, Affine.of(current))
        if image is None:
            break
        out.append(image)
    return out


def copy_actions_at(flat: bytes, boundary: int, limit: int,
                    translations: Sequence[int] = TRANSLATION_SUPPORT
                    ) -> tuple[CopyAction, ...]:
    """Every in-support action at `boundary` whose images match the bytes ahead.

    One step, not a derivation: an action qualifies when its own appended bytes
    are exactly what stands between `boundary` and where it ends, and it may end
    anywhere at or before `limit`. That is the widest honest statement of "this
    block still contains a copy relation", which is what the relation-destroyed
    control has to be able to *refuse* -- a donor block whose first half is a
    valid image of the prefix is not a negative just because its second half is
    not (`docs/copy-relation.md` §4.3).

    Enumeration is over source spans only. The count follows from arithmetic, the
    transform from `solve_step`, and both are then verified against the bytes. No
    Cartesian product over signed bytes and counts is ever built
    (`docs/copy-relation.md` §3.2).
    """
    try:
        marks = boundaries(flat)
    except TraceError:
        return ()
    if boundary not in marks or limit > len(flat):
        return ()
    return tuple(sorted(
        _steps_at(flat, marks, _halt_offsets(flat), boundary, limit, translations),
        key=CopyAction.key))


def _steps_at(flat: bytes, marks: tuple[int, ...], halts: frozenset[int],
              position: int, limit: int,
              translations: Sequence[int] = TRANSLATION_SUPPORT) -> list[CopyAction]:
    out: list[CopyAction] = []
    remaining = limit - position
    _, _, _, opcodes = _prefix_index(flat)
    index_of = {mark: index for index, mark in enumerate(marks)}
    here = index_of[position]
    for start, stop in _eligible_spans(flat, marks, position, halts):
        size = stop - start
        if size > remaining:
            continue
        # Skeleton first, in mark-index space. `apply` never changes an opcode,
        # so two blocks with different opcode sequences cannot be images of each
        # other -- and this rejects almost every candidate before any group
        # arithmetic runs.
        width = index_of[stop] - index_of[start]
        if opcodes[index_of[start]:index_of[stop]] != opcodes[here:here + width]:
            continue
        if position + size not in index_of:
            continue
        source = flat[start:stop]
        for step in solve_step(source, flat[position:position + size]):
            block = bytearray()
            for index, image in enumerate(_images(source, step, remaining)):
                block += image
                total_count = index + 2
                if total_count > max(COUNT_SUPPORT):
                    break
                end = position + len(block)
                if flat[position:end] != bytes(block):
                    break
                if total_count not in COUNT_SUPPORT or end not in marks:
                    continue
                action = CopyAction(
                    boundary=position, source_start=start, source_stop=stop,
                    step=step, total_count=total_count,
                    target_start=position, target_stop=end,
                )
                if action.in_support(translations):
                    out.append(action)
    return out


def derivations(flat: bytes, boundary: int, target_start: int,
                target_stop: int) -> tuple[CopyAction, ...]:
    """Every in-support action at `boundary` that *begins* a copy-only derivation
    of the annotated target block.

    This is the supervision target `docs/copy-relation.md` §4.4 calls "the set of
    valid `COPY` derivations", and it is wider than "the single action that
    appends the whole block" for a reason the group law forces. A count-4 scope
    can be produced by one action of count 4, or by an action of count 2 followed
    by a second whose source is the two images now standing in the prefix and
    whose step is `step^2`. Both schedules append byte-identical bytes, so both
    score identically under the exact-block estimand -- and supervising only the
    maximal one would train the head to call a correct decoder wrong.

    A first step qualifies when it is a valid action here (`copy_actions_at`)
    *and* the remainder is itself derivable. That recursion is bounded by the
    target's own instruction boundaries and memoised, so the whole set costs a
    handful of `apply` calls per case.
    """
    if target_stop <= target_start or target_start != boundary:
        return ()
    try:
        marks = boundaries(flat)
    except TraceError:
        return ()
    if boundary not in marks or target_stop not in marks:
        return ()
    halts = _halt_offsets(flat)
    derivable: dict[int, bool] = {target_stop: True}

    def complete(position: int) -> bool:
        cached = derivable.get(position)
        if cached is None:
            # Seeded False before recursing. Every step advances by at least
            # `MIN_SOURCE_BYTES`, so a cycle is impossible today; the seed keeps
            # the memo correct if that bound is ever relaxed.
            derivable[position] = False
            derivable[position] = any(
                complete(action.target_stop)
                for action in _steps_at(flat, marks, halts, position, target_stop))
        return derivable[position]

    return tuple(sorted(
        (action for action in _steps_at(flat, marks, halts, boundary, target_stop)
         if complete(action.target_stop)),
        key=CopyAction.key))


def derivable_boundaries(flat: bytes,
                         translations: Sequence[int] = AUDIT_TRANSLATIONS
                         ) -> tuple[int, ...]:
    """Every boundary in `flat` at which some copy action is valid.

    The question a negative control has to answer, asked of the *bytes* rather
    than of the generator's provenance. A flat program has no scopes, so
    `actions_of` returns nothing for it whatever it contains -- which made the
    obvious check (`if case.actions`) a tautology that let true relations into a
    false-copy denominator. This walks every boundary and asks the enumerator.

    Defaults to the audit translation range, not the head's support: a relation
    the head could not name is still a relation the bytes contain, and a
    "negative" that holds one is not a negative (`docs/copy-relation.md` §4.3).
    """
    try:
        marks = boundaries(flat)
    except TraceError:
        return ()
    return tuple(
        boundary for boundary in marks
        if boundary < len(flat)
        and copy_actions_at(flat, boundary, len(flat), translations))


def equivalent_actions(flat: bytes, boundary: int, target_start: int,
                       target_stop: int) -> tuple[CopyAction, ...]:
    """The single-action members of `derivations`: one `COPY` for the whole block.

    Kept as its own name because two acceptance clauses ask different questions.
    "Candidate enumeration recovers at least one equivalent valid action" is
    about *this* set containing the generator's own action; "the destroyed
    control contains no annotated positive" is about `derivations` being empty,
    which is the stronger statement.
    """
    return tuple(action for action in derivations(flat, boundary, target_start,
                                                  target_stop)
                 if action.target_stop == target_stop)


# ---------------------------------------------------------------------------
# venues, strata and the frozen tuple spaces

VENUE_SYNTHETIC = "synthetic_nested_repeat"
VENUE_COMPOSED = "composed_motif_relation"
VENUES: tuple[str, ...] = (VENUE_SYNTHETIC, VENUE_COMPOSED)

#: Where a case is allowed to live. `train` is what a checkpoint sees; the two
#: `heldout_*` strata are §2.2's co-primary estimands.  Tuple novelty and pair
#: novelty are explicit, disjoint labels; their interaction is secondary rather
#: than being smuggled into the affine co-primary. `generic` and `destroyed` are
#: the controls §5.3 requires.
STRATUM_TRAIN = "train"
STRATUM_AFFINE = "heldout_affine_tuple"
STRATUM_NESTED = "heldout_nested_composition"
STRATUM_PAIR = "heldout_motif_pair"
STRATUM_INTERACTION = "affine_pair_interaction"
# Compatibility name for callers of the first R1 draft.  It is deliberately an
# alias, not a fourth stratum, so old code cannot recreate a mixed intervention.
STRATUM_SEEN = STRATUM_PAIR
STRATUM_GENERIC = "generic"
STRATUM_DESTROYED = "destroyed"
STRATA: tuple[str, ...] = (STRATUM_TRAIN, STRATUM_AFFINE, STRATUM_NESTED,
                           STRATUM_PAIR, STRATUM_INTERACTION, STRATUM_GENERIC,
                           STRATUM_DESTROYED)
#: The two strata a positive label is decided on. Everything else is a control,
#: a diagnostic or training data, and none of them may move a threshold.
CONFIRMATORY_STRATA: tuple[str, ...] = (STRATUM_AFFINE, STRATUM_NESTED)

#: `synthetic_nested_repeat`'s affine tuple space, as atoms.
#:
#: Deliberately a *product* of small supports rather than a hand-listed set of
#: steps: `heldout_affine_tuple` needs tuples whose every atom is trained and
#: whose combination is not, and only a product space has enough tuples to hold
#: some out while covering every atom in what remains (`held_out_tuples`).
VENUE1_D4: tuple[int, ...] = tuple(range(D4_ORDER))
VENUE1_TRANSLATIONS: tuple[int, ...] = (-32, 0, 32)
VENUE1_COUNTS: tuple[int, ...] = (2, 3, 4)

#: Motif categories `synthetic_nested_repeat` draws leaf bodies from.
#:
#: `random_disc` is excluded: `MOVE`+`CIRCLE` is five bytes, below
#: `MIN_SOURCE_BYTES`, so every action over it would be refused by the candidate
#: rule and the category would contribute rejections only.
VENUE1_CATEGORIES: tuple[str, ...] = ("polyline", "curve", "burst")

#: The box a `synthetic_nested_repeat` motif is drawn inside, and the inset that
#: keeps its images on the canvas under a step with a translation. Small enough
#: that eight D4 elements times nine translations times three counts leaves a
#: usable acceptance rate; the rate itself is measured and reported rather than
#: assumed (`build_venue1`).
VENUE1_BOX: tuple[int, int, int, int] = (72, 72, 136, 136)

#: How many non-copy blocks a `synthetic_nested_repeat` program carries.
#:
#: Not zero: without them every boundary in the program is either inside a copy
#: or the one place a copy starts, and "when not to copy" would have no training
#: signal at all. They also give the false-copy control its material.
VENUE1_DISTRACTORS: tuple[int, ...] = (0, 1, 2)


#: How many motifs share an island.
#:
#: **Islands are what make the dependence graph disconnected.** A case draws its
#: source motif *and* its distractors from one island, so two cases from
#: different islands share no motif and the co-occurrence graph splits into one
#: component per island. Built from a single shared pool instead, both venues
#: collapse to a single component -- a chain of scenes links every motif to every
#: other -- and a connected-component bootstrap over one cluster is not an
#: interval (`docs/copy-relation.md` §12).
#:
#: Four rather than two because a held-out motif *pairing* needs both endpoints
#: to keep co-occurring with something else inside their own island
#: (`held_out_pairs`), which a two-motif island cannot offer.
ISLAND_SIZE = 4


@dataclass(frozen=True)
class Motif:
    """One leaf body, its identity, and the island it may be paired inside.

    `motif_id` is the graph vertex. It is content-addressed rather than an index
    so that two builds of the same pool agree on which motifs are held out even
    if the draw order changes, and so a case record names a motif that can be
    checked rather than a position in a list nobody kept.
    """

    motif_id: str
    category: str
    body: bytes
    island: int = 0
    #: Stable source provenance.  Synthetic motifs use their content identity;
    #: composed motifs use the QuickDraw category/row identity before placement.
    source_id: str = ""


def _canonical_body(body: bytes) -> bytes:
    """`body` translated so its bounding box starts at the origin.

    **Motif identity has to be placement-invariant or the split discipline is
    vacuous.** `dm.data.composed.place` jitters a motif inside its cell, so the
    same QuickDraw drawing produces different bytes in every scene it appears in;
    hashing those bytes would give one vertex per *placement*, every co-occurrence
    edge would be a bridge between two degree-one vertices, and both the pair
    split and the connected-component bootstrap would silently degenerate to a
    per-case resample. Normalising by translation collapses the placements and
    nothing else -- two different drawings cannot normalise to the same bytes.
    """
    extent = composed.bounds(body)
    if extent is None:
        return body
    moved = apply(body, Affine(dx=-extent[0], dy=-extent[1]))
    return body if moved is None else moved


def _motif_id(category: str, body: bytes) -> str:
    digest = hashlib.blake2b(_canonical_body(body), digest_size=8).hexdigest()
    return f"{category}:{digest}"


def islands_of(motifs: Sequence[Motif]) -> list[list[Motif]]:
    """The pool grouped into its islands, in island order."""
    out: dict[int, list[Motif]] = {}
    for motif in motifs:
        out.setdefault(motif.island, []).append(motif)
    return [out[key] for key in sorted(out)]


def motif_pool(n: int, *, seed: int,
               categories: tuple[str, ...] = VENUE1_CATEGORIES,
               box: tuple[int, int, int, int] = VENUE1_BOX) -> list[Motif]:
    """`n` distinct leaf bodies, balanced over categories, in islands of four.

    Drawn *before* any transform and before any pairing, which is the order
    `docs/copy-relation.md` §4.2 requires: split identities first, then apply the
    group, or a validation motif's mirror image is sitting in the training set.

    Islands are assigned here, in draw order, so a pool is a partition and not a
    list a later stage has to remember to cut up.
    """
    builders = {"polyline": synthetic.random_polyline,
                "curve": synthetic.random_curve,
                "burst": synthetic.random_burst}
    missing = [name for name in categories if name not in builders]
    if missing:
        raise ValueError(f"unknown motif categories {missing}")
    rng = random.Random(seed)
    seen: set[str] = set()
    out: list[Motif] = []
    for index in range(200 * n + 1000):
        category = categories[index % len(categories)]
        body = assemble(builders[category](rng, box))
        if not MIN_SOURCE_BYTES <= len(body) <= MAX_SOURCE_BYTES:
            continue
        # Deduplicated on the *identity*, not the bytes: two draws that differ
        # only by a translation are one vertex of the dependence graph, and a
        # pool holding both would report more independent motifs than it has.
        motif_id = _motif_id(category, body)
        if motif_id in seen:
            continue
        seen.add(motif_id)
        out.append(Motif(motif_id, category, body, island=len(out) // ISLAND_SIZE,
                         source_id=f"synthetic:{motif_id}"))
        if len(out) == n:
            return out
    raise TraceError(
        f"only {len(out)}/{n} distinct motifs at seed {seed}; widen the box or "
        "lower the request rather than retrying the seed"
    )


def _boxes_for(body: bytes, transforms: Sequence[Transform]) -> list[tuple[int, ...]] | None:
    """Bounding box of `body` under each transform, or None if two overlap.

    The same containment-and-disjointness test `dm.data.composed.orbit_boxes`
    applies to an orbit, generalised to an arbitrary list of composed transforms
    so it also covers a depth-2 program's `outer^j ∘ inner^i` grid. Overlap is
    refused rather than tolerated: two images on top of each other draw one shape
    while the corpus claims the redundancy of two.
    """
    extent = composed.bounds(body)
    if extent is None:
        return None
    x0, y0, x1, y1 = extent
    out: list[tuple[int, ...]] = []
    for transform in transforms:
        corners = [transform.point(x, y)
                   for x, y in ((x0, y0), (x1, y0), (x0, y1), (x1, y1))]
        xs = [point[0] for point in corners]
        ys = [point[1] for point in corners]
        box = (min(xs), min(ys), max(xs), max(ys))
        if not (0 <= box[0] and 0 <= box[1] and box[2] < CANVAS and box[3] < CANVAS):
            return None
        out.append(box)
    for index, first in enumerate(out):
        for second in out[index + 1:]:
            if not (first[2] < second[0] or second[2] < first[0]
                    or first[3] < second[1] or second[3] < first[1]):
                return None
    return out


#: One `(d4 code, dx, dy, count)` step, as the split discipline enumerates it.
AffineTuple = tuple[int, int, int, int]


def _transform_of(tuple_: AffineTuple) -> Transform:
    code, dx, dy, _count = tuple_
    return Transform(D4.of(code), dx, dy)


def venue1_tuples() -> tuple[AffineTuple, ...]:
    """The whole `synthetic_nested_repeat` step space, in a frozen order."""
    return tuple((code, dx, dy, count)
                 for code in VENUE1_D4
                 for dx in VENUE1_TRANSLATIONS
                 for dy in VENUE1_TRANSLATIONS
                 for count in VENUE1_COUNTS)


def venue2_tuples() -> tuple[AffineTuple, ...]:
    """`composed_motif_relation`'s step space: `dm.data.composed.orbits`, verbatim.

    Read from the audited Direction 2 primitive rather than restated. The orbit
    set is small and its counts are bounded by each element's own order, which is
    a fact about D4 and not a knob -- so this venue can hold out at most the one
    tuple whose atoms both survive elsewhere, and `held_out_tuples` finds which
    rather than being told.
    """
    return tuple((step.d4.code, step.dx, step.dy, count)
                 for step, count in composed.orbits(control=False))


def _atoms(tuple_: AffineTuple) -> tuple[tuple[str, int], ...]:
    code, dx, dy, count = tuple_
    return (("d4", code), ("dx", dx), ("dy", dy), ("count", count))


def held_out_tuples(space: Sequence[AffineTuple], *, seed: int,
                    fraction: float) -> tuple[AffineTuple, ...]:
    """Which complete tuples to withhold, keeping every atom trained.

    **This is the whole of §1.1 made mechanical.** A confirmatory case may test an
    unseen *combination* and may never test an unseen *value*: a flat prefix
    carries no cue for a count the generator sampled, so a numerically unseen
    count is not identifiable and its failure would be a statement about the
    prior, not about the executor. So a tuple is withheld only while every one of
    its four atoms still occurs in at least one tuple that remains.

    Greedy over one frozen shuffle, model-blind and deterministic. The order is a
    function of the seed and the space, never of a score, and a tuple that would
    orphan an atom is skipped rather than the seed retried.
    """
    order = list(space)
    random.Random(seed).shuffle(order)
    remaining = {item: 1 for item in space}
    support: dict[tuple[str, int], int] = {}
    for item in space:
        for atom in _atoms(item):
            support[atom] = support.get(atom, 0) + 1
    # At least one, where one is holdable at all. `composed_motif_relation`'s
    # space is six tuples and its counts are bounded by each element's own order
    # in D4, so a plain fraction would round it to zero and silently leave that
    # venue with no confirmatory contribution -- a fact about the group being
    # read as a design decision.
    want = max(1, int(len(order) * fraction))
    out: list[AffineTuple] = []
    for item in order:
        if len(out) >= want:
            break
        if any(support[atom] <= 1 for atom in _atoms(item)):
            continue
        for atom in _atoms(item):
            support[atom] -= 1
        del remaining[item]
        out.append(item)
    return tuple(sorted(out))


def _scene_bytes(motif: bytes, plan: Sequence[AffineTuple],
                 distractors: Sequence[bytes], cut: int) -> bytes | None:
    """A structured program: nested scopes around `motif`, distractors either side.

    `plan` is outermost first, mirroring `dm.isa.transform.compose_all`'s stack
    convention, so `plan[-1]` is the scope immediately around the motif. Every
    scope is spelled `REPEATX` even when its D4 part is the identity: the flat
    trace is identical either way, and one spelling means the structured half of
    the corpus does not encode the tuple's D4 value in its *opcode* as well as in
    its operand.
    """
    body = bytearray(motif)
    for code, dx, dy, count in reversed(plan):
        body = bytearray(
            bytes([int(Op.REPEATX), count, code, dx & 0xFF, dy & 0xFF])
            + bytes(body) + bytes([int(Op.ENDREP)])
        )
    head = b"".join(distractors[:cut])
    tail = b"".join(distractors[cut:])
    return head + bytes(body) + tail + bytes([int(Op.HALT)])


def _plan_transforms(plan: Sequence[AffineTuple]) -> list[Transform]:
    """Every image's transform, in flat emission order, for the whole nesting.

    Outer image `j` of a scope whose body already expands to `[T^i(B)]` holds
    `A^j T^i (B)`, so the emission order is lexicographic outermost-first and the
    transform is the composition in that order. Derived here and checked against
    the bytes by `verify_trace`, never trusted.
    """
    out = [Transform()]
    for tuple_ in plan:
        step = _transform_of(tuple_)
        count = tuple_[3]
        # `x.then(y)` is `y ∘ x`, so `step^k.then(outer)` is `outer ∘ step^k` --
        # the enclosing scope acting last, which is `compose_all`'s convention
        # and the interpreter's.
        out = [step.power(k).then(outer)
               for outer in out for k in range(count)]
    return out


# ---------------------------------------------------------------------------
# cases


@dataclass(frozen=True)
class RelationCase:
    """One frozen program, its provenance and the block a mode is scored on.

    `target_start`/`target_stop` are the *relation-bearing block*: everything the
    scheduled actions append, from the first boundary a decision is made at to
    the last byte a copy produces. For a depth-2 case that spans both actions on
    purpose -- the compositional estimand is whether the whole nested block comes
    out exactly, not whether one of its two halves does.

    A case with no action at all (`generic`, `destroyed`) carries an empty action
    tuple and an empty target. It is scored for what it must *not* do.
    """

    case_id: str
    venue: str
    stratum: str
    flat: bytes
    structured: bytes
    groups: tuple[str, ...]
    plan: tuple[AffineTuple, ...]
    actions: tuple[CopyAction, ...]
    target_start: int
    target_stop: int
    #: Source identities parallel to `groups`.  These are provenance records,
    #: not placed scene bytes, so development/scientific source disjointness can
    #: be audited after the generator has jittered a motif into a cell.
    source_ids: tuple[str, ...] = ()
    novel_pair: bool = False
    #: Which relation case a destroyed row was spliced from. Empty everywhere
    #: else. Carried so the control's length and byte-census equality can be
    #: checked *against its own source* rather than against itself, which is the
    #: tautology `docs/directions.md` §7 invariant 11 exists to forbid.
    source_case_id: str = ""
    #: The donor used by a destroyed control.  Keeping this separate from the
    #: recipient's `groups` prevents the dependence graph from forgetting which
    #: source supplied the spliced block.
    donor_case_id: str = ""
    donor_groups: tuple[str, ...] = ()
    donor_source_ids: tuple[str, ...] = ()

    @property
    def depth(self) -> int:
        return len(self.plan)

    @property
    def sources(self) -> tuple[str, ...]:
        """Motif identities a relation in this case is *about*.

        One today in both venues. Separate from `groups`, which also holds the
        distractors a program merely contains, because the two graphs answer
        different questions: this one is the resampling unit and `groups` is the
        co-occurrence graph the held-out pair split is drawn on.
        """
        return self.groups[:1]

    @property
    def prompt(self) -> bytes:
        """The prefix a paired free-running comparison conditions on."""
        return self.flat[:self.target_start]

    @property
    def target(self) -> bytes:
        """The bytes both modes are scored against, exactly."""
        return self.flat[self.target_start:self.target_stop]


def _case_id(venue: str, stratum: str, flat: bytes) -> str:
    digest = hashlib.blake2b(digest_size=10)
    digest.update(venue.encode())
    digest.update(b"\0")
    digest.update(stratum.encode())
    digest.update(b"\0")
    digest.update(flat)
    return digest.hexdigest()


class Rejected(ValueError):
    """A candidate case fails a model-blind acceptance rule.

    Carries the rule's name so `build_*` can count *why* a venue refused a draw.
    A generator whose rejections are invisible cannot be audited: "the geometry
    said no twice in three draws" is a fact about what the policy can express and
    belongs in the manifest.
    """

    def __init__(self, rule: str) -> None:
        super().__init__(rule)
        self.rule = rule


def make_case(venue: str, stratum: str, structured: bytes,
              groups: Sequence[str], plan: Sequence[AffineTuple], *,
              novel_pair: bool = False,
              source_ids: Sequence[str] | None = None) -> RelationCase:
    """Trace, verify and score-locate one candidate program, or raise `Rejected`.

    Every clause `docs/copy-relation.md` §4.3 lists that can be settled from one
    program is settled here, at build time, so a defect is a build failure rather
    than a number nobody can reproduce:

    - the traced bytes equal the plain unroller's;
    - every annotated image is image 0 under the annotated step (`verify_trace`);
    - the scheduled actions are contiguous, so the scored block is one range;
    - the generator's own action is inside the enumerator's equivalence set;
    - every action lies inside the frozen caps and supports.
    """
    traced = unroll_with_trace(structured)
    if traced is None:
        raise Rejected("no_exact_flat_trace")
    if traced.bytes != unroll(structured):
        raise Rejected("trace_disagrees_with_unroll")
    verify_trace(traced.bytes, traced.scopes)
    flat = traced.bytes
    # By instruction offset, never by byte value: a coordinate of zero is the
    # byte `0x00` and is not a `HALT`, and rejecting on the byte would quietly
    # remove every drawing that touches the canvas edge.
    if _halt_offsets(flat) != frozenset({len(flat) - 1}):
        raise Rejected("not_one_terminating_halt")

    actions = schedule(actions_of(traced.scopes))
    if plan and len(actions) != len(plan):
        raise Rejected("schedule_does_not_match_plan")
    for earlier, later in pairwise(actions):
        if earlier.target_stop != later.boundary:
            raise Rejected("scheduled_actions_are_not_contiguous")
    for action in actions:
        if action.source_bytes > MAX_SOURCE_BYTES:
            raise Rejected("source_span_over_byte_cap")
        if len(boundaries(flat[action.source_start:action.source_stop])) - 1 \
                > MAX_SOURCE_INSTRUCTIONS:
            raise Rejected("source_span_over_instruction_cap")
        if not action.in_support():
            raise Rejected("action_outside_frozen_support")
        found = equivalent_actions(flat, action.boundary, action.target_start,
                                   action.target_stop)
        if action.key() not in {other.key() for other in found}:
            raise Rejected("candidate_enumeration_misses_the_annotation")
        for other in found:
            if other.render(flat) != flat[other.target_start:other.target_stop]:
                raise Rejected("equivalent_action_renders_different_bytes")

    start = actions[0].boundary if actions else len(flat)
    stop = actions[-1].target_stop if actions else len(flat)
    return RelationCase(
        case_id=_case_id(venue, stratum, flat),
        venue=venue, stratum=stratum, flat=flat, structured=structured,
        groups=tuple(groups), plan=tuple(plan), actions=actions,
        target_start=start, target_stop=stop,
        source_ids=tuple(source_ids if source_ids is not None else groups),
        novel_pair=novel_pair,
    )


@dataclass(frozen=True)
class VenueBuild:
    """What one venue produced, and everything its policy refused on the way."""

    cases: tuple[RelationCase, ...]
    rejections: dict[str, int]
    drawn: int
    #: Motif identity to island, for venues that build their own pool. Empty for
    #: the ones handed a pool that already carries islands.
    islands: dict[str, int] = field(default_factory=dict)


def _distractor_boxes(orbit: Sequence[tuple[int, ...]]) -> list[tuple[int, int, int, int]]:
    """Grid cells the orbit does not touch, so a distractor is never a copy."""
    return [box for box in composed.cells(composed.GRID)
            if all(box[2] < placed[0] or placed[2] < box[0]
                   or box[3] < placed[1] or placed[3] < box[1]
                   for placed in orbit)]


def build_venue1(n: int, *, seed: int, motifs: Sequence[Motif],
                 tuples: Sequence[AffineTuple], depth: int = 1,
                 stratum: str = STRATUM_TRAIN,
                 inner_tuples: Sequence[AffineTuple] = ()) -> VenueBuild:
    """`n` distinct `synthetic_nested_repeat` cases from a fixed motif and tuple pool.

    The pools are arguments rather than draws, because that is what makes the
    split discipline enforceable from outside: a caller hands training motifs and
    training tuples for the training split, and evaluation motifs with withheld
    tuples for a confirmatory stratum. Nothing here can reach a motif or a tuple
    it was not given, so a leak has to be visible at the call site.

    `depth=2` nests `inner_tuples` inside `tuples`; both halves come from the
    *training* tuple space, so the novelty under test is the ordered composition
    and nothing else (`docs/copy-relation.md` §2.2 stratum 2).

    Returns rather than retries on a refusal. Every rejection is a real
    constraint doing its job -- an orbit that leaves the canvas, images that
    overlap, a body too short for the candidate rule -- and a generator that
    silently redrew would hide the rate at which the geometry says no.
    """
    if depth not in (1, 2):
        raise ValueError(f"depth {depth} is not 1 or 2")
    if depth == 2 and not inner_tuples:
        raise ValueError("a depth-2 venue needs an inner tuple pool")
    rng = random.Random(seed)
    rejections: dict[str, int] = {}
    seen: set[bytes] = set()
    out: list[RelationCase] = []
    drawn = 0
    islands = islands_of(motifs)
    source_by_id = {motif.motif_id: motif.source_id or motif.motif_id
                    for motif in motifs}
    for _ in range(200 * n + 2000):
        if len(out) == n:
            break
        drawn += 1
        # Source and distractors come from one island, so two cases from
        # different islands share no motif and the dependence graph is a
        # partition rather than one blob (`ISLAND_SIZE`).
        island = rng.choice(islands)
        motif = rng.choice(island)
        plan: tuple[AffineTuple, ...] = (rng.choice(list(tuples)),)
        if depth == 2:
            plan = (plan[0], rng.choice(list(inner_tuples)))
        placed = _boxes_for(motif.body, _plan_transforms(plan))
        if placed is None:
            rejections["orbit_off_canvas_or_overlapping"] = \
                rejections.get("orbit_off_canvas_or_overlapping", 0) + 1
            continue
        extras: list[bytes] = []
        groups = [motif.motif_id]
        want = rng.choice(VENUE1_DISTRACTORS)
        for box in _distractor_boxes(placed):
            if len(extras) >= want:
                break
            other = rng.choice(island)
            body = composed.place(other.body, box, rng)
            if body is None:
                continue
            extras.append(body)
            groups.append(other.motif_id)
        structured = _scene_bytes(motif.body, plan, extras,
                                  rng.randint(0, len(extras)))
        if structured is None:
            rejections["unrepresentable_scene"] = \
                rejections.get("unrepresentable_scene", 0) + 1
            continue
        try:
            case = make_case(
                VENUE_SYNTHETIC, stratum, structured, groups, plan,
                source_ids=[source_by_id[group] for group in groups])
        except Rejected as refusal:
            rejections[refusal.rule] = rejections.get(refusal.rule, 0) + 1
            continue
        if case.flat in seen:
            rejections["duplicate"] = rejections.get("duplicate", 0) + 1
            continue
        seen.add(case.flat)
        out.append(case)
    return VenueBuild(tuple(out), rejections, drawn)


def composed_islands(split: str, categories: tuple[str, ...],
                     limit: int | None,
                     *, provenance: str = "scientific") -> list[list[Motif]]:
    """The composed source pool, hash-partitioned before limits and islands.

    `source_motif_pool` loads the complete ordered QuickDraw pool, assigns each
    source row to exactly one provenance bucket, then applies `limit`.  Only
    after that cut are motifs grouped into dependence islands.  A different
    scene seed or a different limit therefore cannot turn development rows into
    scientific rows.
    """
    sources = composed.source_motif_pool(
        categories, split, provenance=provenance, limit=limit)
    pool = []
    for index, source in enumerate(sources):
        body = composed.body_of(source.body)
        pool.append(Motif(_motif_id("composed", body), source.category, body,
                          island=index // ISLAND_SIZE, source_id=source.source_id))
    return [pool[at:at + ISLAND_SIZE]
            for at in range(0, len(pool), ISLAND_SIZE)]


def build_venue2(n: int, *, split: str, seed: int,
                 classify: Callable[[AffineTuple, frozenset[frozenset[str]]],
                                    tuple[str, bool] | None],
                 categories: tuple[str, ...] = ("cat", "dog", "bus", "car", "tree"),
                 limit: int | None = None,
                 provenance: str = "scientific") -> VenueBuild:
    """`composed_motif_relation` cases, **partitioned** onto the frozen splits.

    `dm.data.composed` is the generator, unchanged: its motif pool is QuickDraw's
    own train/valid split, and its scenes already carry checked copy spans and
    checked distractor spans. What this adds is the Direction 4 split discipline
    and one filter the candidate rule forces -- a motif longer than
    `MAX_SOURCE_BYTES` cannot be named by any action, so it is refused before
    training rather than after scoring.

    **Scenes are drawn island by island.** `composed.scene` picks its orbit motif
    and its distractors from whatever pool it is handed, so handing it one island
    at a time is what keeps the co-occurrence graph disconnected; handed the whole
    pool it links every motif to every other and the bootstrap loses its clusters
    (`ISLAND_SIZE`). The pool injection is the one thing Direction 4 added to that
    Module, and it defaults to the pre-Direction-4 path.

    Partitioned rather than filtered, and that is the point. Which stratum a scene
    belongs to is decided by `classify` from the scene's own affine tuple and its
    motif-pair edges -- returning the stratum and whether the pairing is a
    withheld one -- so one pass over one generator produces the training split and
    the held-out splits together and no scene can land in two. A held-out *pair*
    needs both endpoints trained, which only a partition of one pool can deliver.
    """
    rejections: dict[str, int] = {}
    out: list[RelationCase] = []
    seen: set[bytes] = set()
    islands = composed_islands(split, categories, limit, provenance=provenance)
    assignment = {motif.motif_id: index
                  for index, island in enumerate(islands) for motif in island}
    per_island = max(1, -(-n // max(1, len(islands))))
    drawn = 0
    for index, island in enumerate(islands):
        if len(out) >= n:
            break
        if len(island) < 2:
            rejections["island_too_small"] = \
                rejections.get("island_too_small", 0) + 1
            continue
        try:
            scenes, stats = composed.build_with_stats(
                per_island, split, seed=seed + index,
                pool=[motif.body for motif in island])
        except ValueError:
            # The geometry refused this island. Counted, never retried with
            # another seed: a policy that redraws until an island cooperates is
            # selecting islands on an outcome.
            rejections["island_geometry"] = \
                rejections.get("island_geometry", 0) + 1
            continue
        drawn += stats["drawn"]
        rejections["composed_generator"] = \
            rejections.get("composed_generator", 0) + stats["rejected"]
        for scene in scenes:
            traced = unroll_with_trace(scene.structured)
            if traced is None:
                rejections["no_exact_flat_trace"] = \
                    rejections.get("no_exact_flat_trace", 0) + 1
                continue
            actions = schedule(actions_of(traced.scopes))
            if len(actions) != 1:
                rejections["not_one_orbit"] = \
                    rejections.get("not_one_orbit", 0) + 1
                continue
            action = actions[0]
            signature: AffineTuple = (action.step.d4.code, action.step.dx,
                                      action.step.dy, action.total_count)
            orbit = _motif_id("composed", scene.flat[scene.copies[0]])
            others = [_motif_id("composed", scene.flat[span])
                      for span in scene.distractor_spans]
            edges = frozenset(frozenset({orbit, other}) for other in others
                              if other != orbit)
            verdict = classify(signature, edges)
            if verdict is None:
                rejections["outside_every_stratum"] = \
                    rejections.get("outside_every_stratum", 0) + 1
                continue
            stratum, novel_pair = verdict
            try:
                by_id = {motif.motif_id: motif for motif in island}
                case = make_case(VENUE_COMPOSED, stratum, scene.structured,
                                 [orbit, *others], (signature,),
                                 novel_pair=novel_pair,
                                 source_ids=[by_id[motif].source_id
                                             for motif in (orbit, *others)])
            except Rejected as refusal:
                rejections[refusal.rule] = rejections.get(refusal.rule, 0) + 1
                continue
            if case.flat in seen:
                rejections["duplicate"] = rejections.get("duplicate", 0) + 1
                continue
            seen.add(case.flat)
            out.append(case)
    return VenueBuild(tuple(out), rejections, drawn, assignment)


# ---------------------------------------------------------------------------
# controls


def build_generic(n: int, *, seed: int, motifs: Sequence[Motif]) -> VenueBuild:
    """Flat programs with no scope at all: the false-copy denominator.

    Motifs placed in distinct grid cells and nothing else. Every boundary in one
    of these programs is a boundary at which the only correct action is `EMIT`,
    so the rate at which a relation-capable checkpoint invokes `COPY` here is
    measurable without any annotation to argue about
    (`docs/copy-relation.md` §5.3).
    """
    rng = random.Random(seed)
    rejections: dict[str, int] = {}
    seen: set[bytes] = set()
    out: list[RelationCase] = []
    drawn = 0
    boxes = composed.cells(composed.GRID)
    islands = islands_of(motifs)
    source_by_id = {motif.motif_id: motif.source_id or motif.motif_id
                    for motif in motifs}
    for _ in range(200 * n + 2000):
        if len(out) == n:
            break
        drawn += 1
        island = rng.choice(islands)
        blocks: list[bytes] = []
        groups: list[str] = []
        for box in rng.sample(boxes, rng.randint(2, len(boxes))):
            motif = rng.choice(island)
            body = composed.place(motif.body, box, rng)
            if body is None:
                continue
            blocks.append(body)
            groups.append(motif.motif_id)
        if len(blocks) < 2:
            rejections["too_few_blocks"] = rejections.get("too_few_blocks", 0) + 1
            continue
        program = b"".join(blocks) + bytes([int(Op.HALT)])
        try:
            case = make_case(
                VENUE_SYNTHETIC, STRATUM_GENERIC, program, groups, (),
                source_ids=[source_by_id[group] for group in groups])
        except Rejected as refusal:
            rejections[refusal.rule] = rejections.get(refusal.rule, 0) + 1
            continue
        # **Asked of the enumerator, not of the trace.** `make_case` fills
        # `actions` from scope provenance, and a program with no scope has none by
        # construction -- so testing `case.actions` here was a tautology that
        # accepted whatever the generator produced. The question a false-copy
        # denominator actually needs is whether any boundary in the program admits
        # a copy action at all, over the *audit* translation range, and that is
        # what `derivable_boundaries` asks.
        if derivable_boundaries(case.flat):
            rejections["accidental_relation"] = \
                rejections.get("accidental_relation", 0) + 1
            continue
        if case.flat in seen:
            rejections["duplicate"] = rejections.get("duplicate", 0) + 1
            continue
        seen.add(case.flat)
        out.append(case)
    return VenueBuild(tuple(out), rejections, drawn)


def _target_stratum(case: RelationCase) -> tuple[int, tuple[int, ...]]:
    """What a destroyed-control donor must match: block length and skeleton.

    Exactly Direction 3's rule, restated for a whole relation block instead of a
    single continuation copy. Matching both is what lets the intervention be a
    *permutation*: the corpus-wide byte multiset and every program's length are
    preserved by construction, so the arms cannot differ in length, opcode census
    or byte census, only in whether the block continues the relation.
    """
    return (case.target_stop - case.target_start, _skeleton(case.target))


#: How many times `destroy` may re-derange a stratum after dropping rows whose
#: donor block still admitted a copy action. Bounded so a pathological stratum
#: terminates; the count that survived and the count that did not are both
#: reported, so a stratum that needed every attempt is visible.
MAX_DERANGEMENT_ATTEMPTS = 8


def _derange(members: list[int], rng: random.Random) -> dict[int, int]:
    """A fixed-point-free permutation of `members`: shuffle once, cycle by one.

    Direction 3's construction, reused because its balance argument is the one
    that survived audit. One cycle, so every member donates exactly once and
    receives exactly once -- which is what makes the arm's byte multiset equal to
    its sources' *exactly* rather than approximately.
    """
    order = list(members)
    rng.shuffle(order)
    return {index: order[(position + 1) % len(order)]
            for position, index in enumerate(order)}


def _destroy_namespace(case: RelationCase, labels: Mapping[str, int]) -> tuple:
    """The frozen donor namespace used by both construction and audit."""
    return (case.venue, case.stratum, component_of(case, dict(labels)),
            *_target_stratum(case))


def destroy(cases: Sequence[RelationCase], *, seed: int) -> tuple[
        tuple[RelationCase, ...], dict[str, int]]:
    """Swap relation blocks between cases, then prove no relation survived.

    Within a `(venue, source stratum, co-occurrence component, length, skeleton)`
    namespace the blocks are deranged, which preserves every program's length
    and the namespace's byte multiset by construction.  A donor block can
    nevertheless be an accidental image of its recipient's own prefix -- two
    motifs of the same skeleton, related by some element of the group -- and
    calling such a row negative would be exactly the fault
    `docs/copy-relation.md` §4.3 forbids.

    **So a failing row is dropped and the stratum is re-deranged, rather than the
    row being dropped from a permutation that then no longer closes.** A
    permutation with holes in it is not a permutation: its donors and its
    recipients are different multisets, and the arm's byte census stops matching
    its sources' for a reason that has nothing to do with the intervention. The
    loop keeps the arm exactly balanced at the cost of some rows, and reports how
    many rows and how many attempts each stratum needed.
    """
    labels = components(cases)
    buckets: dict[tuple, list[int]] = {}
    for index, case in enumerate(cases):
        if case.actions:
            namespace = _destroy_namespace(case, labels)
            buckets.setdefault(namespace, []).append(index)
    rng = random.Random(seed)
    notes: dict[str, int] = {}
    out: list[RelationCase] = []

    def bump(key: str, amount: int = 1) -> None:
        notes[key] = notes.get(key, 0) + amount

    # Hoisted out of the retry loop: a source program does not change between
    # attempts, and re-deriving its relation set once per attempt was most of the
    # build's cost.
    clean = {index: frozenset(derivable_boundaries(cases[index].flat))
             for members in buckets.values() for index in members}

    for _, members in sorted(buckets.items()):
        alive = list(members)
        for attempt in range(MAX_DERANGEMENT_ATTEMPTS):
            if len(alive) < 2:
                bump("stratum_too_small", len(alive))
                alive = []
                break
            donors = _derange(alive, rng)
            spliced: dict[int, bytes] = {}
            residual: list[int] = []
            for index in alive:
                case = cases[index]
                block = cases[donors[index]].target
                candidate = (case.flat[:case.target_start] + block
                             + case.flat[case.target_stop:])
                if len(candidate) != len(case.flat):
                    raise TraceError("a destroyed splice changed a program's length")
                if copy_actions_at(candidate, case.target_start,
                                   case.target_stop, AUDIT_TRANSLATIONS):
                    residual.append(index)
                    bump("residual_at_the_spliced_boundary")
                elif not frozenset(derivable_boundaries(candidate)) <= clean[index]:
                    residual.append(index)
                    bump("splice_introduced_a_relation")
                else:
                    spliced[index] = candidate
            if not residual:
                for index in alive:
                    case = cases[index]
                    donor = cases[donors[index]]
                    out.append(RelationCase(
                        case_id=_case_id(case.venue, STRATUM_DESTROYED,
                                         spliced[index]),
                        venue=case.venue, stratum=STRATUM_DESTROYED,
                        flat=spliced[index], structured=spliced[index],
                        groups=case.groups, plan=(), actions=(),
                        target_start=case.target_start,
                        target_stop=case.target_stop,
                        source_case_id=case.case_id,
                        donor_case_id=donor.case_id,
                        donor_groups=donor.groups,
                        donor_source_ids=donor.source_ids,
                    ))
                bump("attempts", attempt + 1)
                alive = []
                break
            bump("dropped", len(residual))
            alive = [index for index in alive if index not in set(residual)]
        else:
            bump("unbalanced_after_max_attempts", len(alive))
    return tuple(out), notes


# ---------------------------------------------------------------------------
# the dependence graph


def _connected(vertices: Iterable[str],
               edges: Iterable[tuple[str, str]]) -> dict[str, int]:
    """Union-find over a sorted vertex list, labelled by smallest member.

    Sorted and root-minimised so the labelling is a function of the graph rather
    than of dictionary insertion order, and two builds of the same corpus agree
    on component numbers.
    """
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for node in vertices:
        find(node)
    for left, right in edges:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    roots = sorted({find(node) for node in parent})
    index = {root: number for number, root in enumerate(roots)}
    return {node: index[find(node)] for node in sorted(parent)}


def components(cases: Sequence[RelationCase]) -> dict[str, int]:
    """The resampling unit: connected components of the motif co-occurrence graph.

    Vertices are motif identities and an edge joins two motifs that appear in the
    same program -- `docs/copy-relation.md` §4.2, read literally. Two cases that
    share a motif are not independent draws, so it is components and never cases
    that a bootstrap resamples.

    **This is only a usable unit because the pool is partitioned into islands.**
    Built from one shared motif pool, a chain of scenes links every motif to every
    other and the whole corpus is a single component: an interval over one cluster
    is not an interval. An earlier revision worked around that by resampling
    *source* motifs alone and reporting this graph beside it, which left the
    dependence through shared distractors unmodelled. `ISLAND_SIZE` removes the
    dependence instead of renaming the unit -- a case draws every motif it
    contains from one island, so the graph is disconnected by construction and
    the literal reading is the one in force.
    """
    vertices = [motif for case in cases for motif in case.groups]
    edges = [(case.groups[0], other) for case in cases for other in case.groups[1:]]
    return _connected(vertices, edges)


def island_faults(cases: Sequence[RelationCase],
                  motifs: Mapping[str, int]) -> list[str]:
    """Cases whose motifs span more than one island, which would rejoin the graph.

    One line of defence behind `ISLAND_SIZE`: the builders draw from one island by
    construction, and this checks the corpus rather than trusting them, because a
    single leaked case is enough to merge two clusters and quietly narrow every
    interval drawn afterwards.
    """
    faults: list[str] = []
    for case in cases:
        seen = {motifs[motif] for motif in case.groups if motif in motifs}
        if len(seen) > 1:
            faults.append(f"{case.case_id} spans islands {sorted(seen)}")
    return faults


def component_of(case: RelationCase, labels: dict[str, int]) -> int:
    """The resampling unit a case belongs to, via its source motif.

    A case is attributed to one component rather than to all of its motifs, or a
    bootstrap draw would have to decide what a case shared between two drawn
    components is worth. Inside an island every motif of a case lands in the same
    component anyway, so the choice of `groups[0]` is a tie-break and not a
    modelling decision.
    """
    return labels[case.groups[0]]


# ---------------------------------------------------------------------------
# the whole corpus


#: Bumped whenever a change makes a new traced corpus incomparable with an old
#: one. Separate from the protocol schema: the corpus freezes at R1 and the
#: training protocol at R6, and a shared counter would force a spurious bump on
#: whichever froze first.
CORPUS_SCHEMA = 1
MANIFEST_SCHEMA = 1
CORPUS_PROVENANCES: tuple[str, ...] = ("scientific", "development")

#: Which direction owns an artifact written here.
#:
#: `docs/copy-relation.md` §7 invariant 2 keeps Direction 4's artifact namespace
#: disjoint from Direction 3's, and a manifest that carries the number is one a
#: path mistake cannot smuggle across the boundary. Declared beside the schemas
#: rather than in the contract Module because the *manifest* is this file's
#: artifact; `dm.eval.relation_contract.DIRECTION` reads it from here so the
#: number has one owner (`docs/directions.md` §7 invariant 17).
DIRECTION = 4


@dataclass(frozen=True)
class BuildConfig:
    """Every knob a traced-corpus build reads, in one serialisable record.

    A build is a pure function of this and nothing else, which is the property
    that makes a manifest a description of the corpus rather than a description
    of a command line. Sizes are requests: what a venue actually produced, and
    what its geometry refused, come back in the build report.
    """

    n_train_synthetic: int = 20_000
    n_train_composed: int = 20_000
    n_eval: int = 1_000
    n_generic: int = 1_000
    train_motifs: int = 2_048
    eval_motifs: int = 512
    heldout_tuple_fraction: float = 0.15
    heldout_pair_fraction: float = 0.20
    composed_categories: tuple[str, ...] = ("cat", "dog", "bus", "car", "tree")
    #: Size of the QuickDraw motif pool `composed_motif_relation` draws from.
    #:
    #: Bounded on purpose. A held-out motif *pairing* needs both endpoints to
    #: co-occur with something else in training, and over an unbounded pool every
    #: motif appears once, every edge is a bridge, and `held_out_pairs` can
    #: withhold nothing without orphaning a vertex. Capping the pool is what makes
    #: the edge split expressible; the cap is model-blind and frozen here.
    composed_limit: int | None = 2_048
    #: One root seed. Every stream below is derived from it by name, so adding a
    #: stream cannot shift an existing one and a rename is caught by the
    #: manifest rather than silently producing a different corpus.
    data_seed: int = 0
    #: Which stable QuickDraw source partition this build owns.  The scientific
    #: and development partitions are disjoint before limits and islands.
    provenance: str = "scientific"

    def __post_init__(self) -> None:
        if self.provenance not in CORPUS_PROVENANCES:
            raise ValueError(
                f"unknown corpus provenance {self.provenance!r}; expected one of "
                f"{list(CORPUS_PROVENANCES)}")

    def seed_for(self, stream: str) -> int:
        digest = hashlib.sha256(f"direction4-relation:{self.data_seed}:{stream}".encode())
        return int(digest.hexdigest()[:8], 16)


@dataclass(frozen=True)
class RelationCorpus:
    """The frozen traced corpus: cases by stratum, plus the resampling graph."""

    config: BuildConfig
    cases: tuple[RelationCase, ...]
    labels: dict[str, int]
    islands: dict[str, int]
    reports: dict[str, dict]
    heldout_tuples: dict[str, tuple[AffineTuple, ...]]
    heldout_pairs: frozenset[frozenset[str]]

    def of(self, stratum: str) -> tuple[RelationCase, ...]:
        return tuple(case for case in self.cases if case.stratum == stratum)

    def programs(self, stratum: str) -> list[bytes]:
        return [case.flat for case in self.of(stratum)]


def _pairs_of(cases: Sequence[RelationCase]) -> list[frozenset[str]]:
    """Every motif pair that co-occurs in a program, deduplicated and ordered."""
    seen: dict[frozenset[str], None] = {}
    for case in cases:
        for other in case.groups[1:]:
            if other != case.groups[0]:
                seen.setdefault(frozenset({case.groups[0], other}), None)
    return sorted(seen, key=lambda pair: sorted(pair))


def held_out_pairs(cases: Sequence[RelationCase], *, seed: int,
                   fraction: float) -> frozenset[frozenset[str]]:
    """Which co-occurrence *edges* to withhold, keeping both endpoints trained.

    "Motifs never seen together" is an edge-disjoint split, not a vertex split:
    both endpoints must still appear in an unaffected co-occurrence case or the
    stratum would be testing an unseen motif and calling it an unseen pairing
    (`docs/copy-relation.md` §4.2). Because a whole multi-distractor scene gets
    one label, the check simulates removing every case touched by the proposed
    edge; a plain edge-degree test is too weak.
    """
    edges = _pairs_of(cases)
    case_edges = [set(_pairs_of((case,))) for case in cases]
    order = list(edges)
    random.Random(seed).shuffle(order)
    want = int(len(order) * fraction)
    out: list[frozenset[str]] = []
    for edge in order:
        if len(out) >= want:
            break
        trial = [*out, edge]
        # A whole scene is assigned to the secondary stratum when it contains
        # one held-out edge, so edge degree is not enough: a triple scene can
        # remove two other edges at the same time.  Test endpoint support after
        # removing the affected *cases*, which is the split actually applied.
        remaining_nodes = {
            node for edges_for_case in case_edges
            if not edges_for_case & set(trial)
            for edge_for_case in edges_for_case for node in edge_for_case
        }
        if not set(edge) <= remaining_nodes:
            continue
        out.append(edge)
    return frozenset(out)


#: Motifs whose bodies are QuickDraw's validation split. Completely novel
#: motifs, so this is a *transfer* stratum and never a confirmatory one --
#: `docs/copy-relation.md` §4.2 separates them precisely because a novel motif
#: tests something the co-primary strata deliberately hold fixed.
STRATUM_TRANSFER = "novel_motif_transfer"


def build(config: BuildConfig | None = None) -> RelationCorpus:
    """The whole traced corpus, split before any transform and audited on the way.

    The order of operations is the split discipline, executed:

    1. two disjoint motif pools are drawn **before** any transform, so no D4 or
       translation image of an evaluation motif can enter training;
    2. the affine tuple space of each venue is split so that every withheld tuple
       keeps all four of its atoms trained (`held_out_tuples`);
    3. `synthetic_nested_repeat` builds training cases from training motifs and
       training tuples only, and each held-out stratum from the evaluation pool;
    4. `composed_motif_relation` builds one pool and *partitions* it, so a
       withheld motif pairing has both endpoints trained;
    5. the relation-destroyed control is deranged out of the evaluation cases and
       every spliced row is re-enumerated to prove no relation survived;
    6. the co-occurrence graph is built over the finished corpus and its
       connected components become the resampling unit.

    Nothing here consults a model, and nothing chooses a seed twice. A stratum
    that comes out short comes out short, and `reports` says by how much.
    """
    config = config or BuildConfig()
    train_motifs = motif_pool(config.train_motifs, seed=config.seed_for("train_motifs"))
    eval_motifs = motif_pool(config.eval_motifs, seed=config.seed_for("eval_motifs"))
    overlap = {m.motif_id for m in train_motifs} & {m.motif_id for m in eval_motifs}
    if overlap:
        raise TraceError(
            f"{len(overlap)} motif identities are in both pools; the split has to "
            "happen before any transform, so this is a build defect"
        )

    # **One split over the union of both venues' tuple spaces.** The two spaces
    # overlap -- every `composed_motif_relation` orbit is a `(d4, 0, 0, count)`
    # that `synthetic_nested_repeat` can also express -- and a checkpoint trains
    # on both venues at once. Splitting per venue would leave a tuple withheld in
    # one venue and trained in the other, so its *combination* would have been
    # seen after all and the confirmatory stratum would be measuring motif
    # novelty under a familiar action.
    space = tuple(dict.fromkeys((*venue1_tuples(), *venue2_tuples())))
    withheld = frozenset(held_out_tuples(space, seed=config.seed_for("tuples"),
                                         fraction=config.heldout_tuple_fraction))
    train1 = tuple(item for item in venue1_tuples() if item not in withheld)
    held1 = frozenset(item for item in venue1_tuples() if item in withheld)
    held2 = frozenset(item for item in venue2_tuples() if item in withheld)

    builds: dict[str, VenueBuild] = {}
    builds["v1_train"] = build_venue1(
        config.n_train_synthetic, seed=config.seed_for("v1_train"),
        motifs=train_motifs, tuples=train1, stratum=STRATUM_TRAIN)
    builds["v1_affine"] = build_venue1(
        config.n_eval, seed=config.seed_for("v1_affine"),
        motifs=eval_motifs, tuples=tuple(sorted(held1)), stratum=STRATUM_AFFINE)
    # **Trained motifs, unseen ordered composition.** Both halves of the plan are
    # training tuples and every motif is one a training case actually contains,
    # so the only thing this stratum withholds is the *composition* -- which is
    # what §2.2 stratum 2 is about. Drawing it from the evaluation pool instead
    # confounded compositional generalisation with motif transfer, and a failure
    # could not have said which it was. `v1_nested_transfer` keeps the harder
    # version as a secondary.
    #
    # Filtered to motifs that *occur* in accepted training cases, not merely to
    # the training pool: the venue refuses most draws on geometry, so a pool
    # motif can easily have reached no training program at all, and calling it
    # trained would quietly reintroduce the confound this split exists to remove.
    seen_in_training = {motif for case in builds["v1_train"].cases
                        for motif in case.groups}
    trained_motifs = [motif for motif in train_motifs
                      if motif.motif_id in seen_in_training]
    if not trained_motifs:
        raise TraceError(
            "no training motif reached an accepted training case; the nested "
            "co-primary would have no trained motif to compose")
    builds["v1_nested"] = build_venue1(
        config.n_eval, seed=config.seed_for("v1_nested"),
        motifs=trained_motifs, tuples=train1, depth=2, inner_tuples=train1,
        stratum=STRATUM_NESTED)
    builds["v1_nested_transfer"] = build_venue1(
        config.n_eval, seed=config.seed_for("v1_nested_transfer"),
        motifs=eval_motifs, tuples=train1, depth=2, inner_tuples=train1,
        stratum=STRATUM_NESTED_TRANSFER)
    builds["generic"] = build_generic(
        config.n_generic, seed=config.seed_for("generic"), motifs=eval_motifs)

    # Venue 2 needs its own pair split, and a pair split needs the edges to
    # exist first. So the pool is built once with every scene provisionally
    # training, the edges are read off it, and the same pool is partitioned on
    # the second pass. Both passes use one seed, so the scenes are identical.
    probe = build_venue2(config.n_train_composed, split="train",
                         seed=config.seed_for("v2"),
                         classify=lambda tuple_, edges: (STRATUM_TRAIN, False),
                         categories=config.composed_categories,
                         limit=config.composed_limit,
                         provenance=config.provenance)
    # Pair novelty is only identifiable on rows whose affine tuple remains in
    # training.  Choosing edges from the all-train probe allowed an endpoint to
    # occur only in tuple-held-out rows, so the later pair split could orphan it
    # even though `held_out_pairs`' local degree check passed.
    pair_probe = tuple(case for case in probe.cases
                       if case.plan and case.plan[0] not in held2)
    pairs = held_out_pairs(pair_probe, seed=config.seed_for("v2_pairs"),
                           fraction=config.heldout_pair_fraction)

    def classify(tuple_: AffineTuple,
                 edges: frozenset[frozenset[str]]) -> tuple[str, bool] | None:
        novel = bool(edges & pairs)
        if tuple_ in held2:
            if novel:
                # Both interventions are present.  Keep this secondary so a
                # failed affine gate cannot be attributed to tuple novelty
                # alone.
                return STRATUM_INTERACTION, True
            return STRATUM_AFFINE, False
        if novel:
            # A withheld pairing under a trained tuple is a secondary diagnostic
            # that varies motif co-occurrence while holding the affine tuple
            # fixed.
            return STRATUM_PAIR, True
        return STRATUM_TRAIN, False

    builds["v2"] = build_venue2(config.n_train_composed, split="train",
                                seed=config.seed_for("v2"), classify=classify,
                                categories=config.composed_categories,
                                limit=config.composed_limit,
                                provenance=config.provenance)
    builds["v2_transfer"] = build_venue2(
        config.n_eval, split="valid", seed=composed.val_seed(config.seed_for("v2")),
        classify=lambda tuple_, edges: (STRATUM_TRANSFER, False),
        categories=config.composed_categories, limit=config.composed_limit,
        provenance=config.provenance)

    cases: list[RelationCase] = []
    for build_ in builds.values():
        cases.extend(build_.cases)

    # A held-out case whose flat bytes are also a training program is memorised,
    # not generalised. Refused here rather than reported, and refused by *bytes*
    # so a case that arrives at the same program by a different plan is caught
    # too.
    trained = {case.flat for case in cases if case.stratum == STRATUM_TRAIN}
    leaked = sum(1 for case in cases
                 if case.stratum != STRATUM_TRAIN and case.flat in trained)
    cases = [case for case in cases
             if case.stratum == STRATUM_TRAIN or case.flat not in trained]

    evaluable = [case for case in cases
                 if case.stratum in (STRATUM_AFFINE, STRATUM_NESTED,
                                     STRATUM_NESTED_TRANSFER, STRATUM_PAIR,
                                     STRATUM_INTERACTION)]
    destroyed, destroy_notes = destroy(evaluable, seed=config.seed_for("destroy"))
    cases.extend(destroyed)

    islands = {motif.motif_id: motif.island
               for motif in (*train_motifs, *eval_motifs)}
    for name in ("v2", "v2_transfer"):
        islands.update(builds[name].islands)

    corpus = RelationCorpus(
        config=config,
        cases=tuple(cases),
        labels=components(cases),
        islands=islands,
        reports={
            "venues": {name: {"accepted": len(item.cases), "drawn": item.drawn,
                              "rejections": dict(sorted(item.rejections.items()))}
                       for name, item in builds.items()},
            "destroyed": {"accepted": len(destroyed), "offered": len(evaluable),
                          "notes": dict(sorted(destroy_notes.items()))},
            "held_out_pairs": len(pairs),
            "provenance": config.provenance,
            "leaked_into_training": leaked,
        },
        heldout_tuples={VENUE_SYNTHETIC: tuple(sorted(held1)),
                        VENUE_COMPOSED: tuple(sorted(held2))},
        heldout_pairs=pairs,
    )
    return corpus


# ---------------------------------------------------------------------------
# acceptance


def coverage_report(cases: Sequence[RelationCase]) -> dict:
    """Whether the frozen caps and supports cover every accepted positive.

    `docs/copy-relation.md` §3.2 requires a valid derivation for **100%** of
    accepted positive cases, which is a statement about the caps and not about a
    model. Recomputed on every build rather than trusted to the audit that chose
    the caps, because the caps are constants and the corpus is not.
    """
    positives = [case for case in cases if case.actions]
    covered = 0
    single = 0
    multi = 0
    widest_bytes = 0
    widest_instructions = 0
    for case in positives:
        found = derivations(case.flat, case.target_start, case.target_start,
                            case.target_stop)
        if found:
            covered += 1
        if len(found) == 1:
            single += 1
        elif len(found) > 1:
            multi += 1
        for action in case.actions:
            widest_bytes = max(widest_bytes, action.source_bytes)
            widest_instructions = max(
                widest_instructions,
                len(boundaries(case.flat[action.source_start:action.source_stop])) - 1)
    return {
        "positives": len(positives),
        "covered": covered,
        "coverage": covered / max(1, len(positives)),
        "single_derivation": single,
        "multiple_derivations": multi,
        "widest_source_bytes": widest_bytes,
        "widest_source_instructions": widest_instructions,
        "max_source_bytes": MAX_SOURCE_BYTES,
        "max_source_instructions": MAX_SOURCE_INSTRUCTIONS,
    }


def length_bin_report(cases: Sequence[RelationCase],
                      edges: Sequence[int]) -> dict:
    """How the accepted source spans fall into the frozen length bins.

    The contract freezes bin *edges*, not just a bin count, so this can be a
    check rather than an assertion: every accepted span must land inside the last
    edge, and every bin must be reachable. A bin no span ever falls in is a row of
    the key embedding that never receives a gradient, which is a parameter budget
    spent on nothing and a silent asymmetry between the venues.
    """
    counts = [0] * len(edges)
    widest = 0
    outside = 0
    for case in cases:
        for action in case.actions:
            widest = max(widest, action.source_bytes)
            for index, edge in enumerate(edges):
                if action.source_bytes <= edge:
                    counts[index] += 1
                    break
            else:
                outside += 1
    return {
        "edges": list(edges),
        "counts": counts,
        "empty_bins": [index for index, count in enumerate(counts) if not count],
        "spans_past_last_edge": outside,
        "widest_source_bytes": widest,
    }


#: The most candidate spans any boundary can present, from the caps alone.
#:
#: `_eligible_spans` walks at most `MAX_SOURCE_GAP_INSTRUCTIONS + 1` stop
#: positions and, from each, at most `MAX_SOURCE_INSTRUCTIONS` starts. So this
#: bound holds for **every** program, not merely for the ones a build happened to
#: produce -- which is what a cap on the head's scoring set has to be. A sampled
#: maximum would have been evidence about one corpus; this is arithmetic about
#: the rule.
CANDIDATE_SPAN_BOUND = (MAX_SOURCE_GAP_INSTRUCTIONS + 1) * MAX_SOURCE_INSTRUCTIONS


def candidate_span_report(cases: Sequence[RelationCase], *,
                          sample: int = 200) -> dict:
    """What the R3 head has to score at one decision: the bound, and what was seen.

    The bound is exhaustive and free (`CANDIDATE_SPAN_BOUND`); the observed
    figures come from the first `sample` cases and are descriptive only, because
    materialising every candidate set over a frozen-size corpus is half a million
    boundaries of work to re-derive a number the caps already imply.
    """
    widest = 0
    total = 0
    seen = 0
    for case in cases[:sample]:
        for boundary in boundaries(case.flat):
            if boundary >= len(case.flat):
                continue
            found = len(candidate_spans(case.flat, boundary))
            widest = max(widest, found)
            total += found
            seen += 1
    return {
        "bound": CANDIDATE_SPAN_BOUND,
        "cap": MAX_CANDIDATE_SPANS,
        "observed_widest": widest,
        "observed_mean": total / max(1, seen),
        "sampled_cases": min(sample, len(cases)),
        "sampled_boundaries": seen,
    }


def _byte_census(programs: Sequence[bytes]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for program in programs:
        for byte in program:
            key = str(byte)
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def _atom_support(tuples: Iterable[AffineTuple]) -> set[tuple[str, int]]:
    out: set[tuple[str, int]] = set()
    for item in tuples:
        out.update(_atoms(item))
    return out


def _edge_list(edges: Iterable[frozenset[str]]) -> list[list[str]]:
    """Stable JSON form for motif-pair edges."""
    return [sorted(edge) for edge in sorted(edges, key=lambda edge: sorted(edge))]


def _pair_audit(corpus: RelationCorpus,
                by_stratum: Mapping[str, Sequence[RelationCase]]) -> dict:
    """Recompute the held-out edge split from case provenance.

    A count of withheld pairs is not an audit: it says neither which edges were
    withheld nor whether those edges entered training.  This report carries the
    exact edge set and checks endpoints, training leakage, secondary labels and
    that every declared edge was actually represented by an accepted case.
    """
    heldout = set(corpus.heldout_pairs)
    training_edges = set(_pairs_of(by_stratum[STRATUM_TRAIN]))
    training_nodes = {motif for case in by_stratum[STRATUM_TRAIN]
                      for motif in case.groups}
    endpoint_orphans = sorted(
        f"{node} not in training"
        for edge in heldout for node in edge if node not in training_nodes)
    pair_cases = (*by_stratum[STRATUM_PAIR], *by_stratum[STRATUM_INTERACTION])
    pair_case_edges = {
        edge for case in pair_cases for edge in _pairs_of((case,))
        if edge in heldout
    }
    missing_in_cases = heldout - pair_case_edges
    heldout_in_training = heldout & training_edges
    mixed_affine = sorted(
        case.case_id for case in by_stratum[STRATUM_AFFINE]
        if set(_pairs_of((case,))) & heldout)
    unmarked_secondary = sorted(
        case.case_id for case in pair_cases
        if not (set(_pairs_of((case,))) & heldout))
    wrong_interaction = sorted(
        case.case_id for case in by_stratum[STRATUM_INTERACTION]
        if not set(case.plan) & set(corpus.heldout_tuples.get(VENUE_COMPOSED, ()))
    )
    wrong_pair = sorted(
        case.case_id for case in by_stratum[STRATUM_PAIR]
        if set(case.plan) & set(corpus.heldout_tuples.get(VENUE_COMPOSED, ()))
    )
    return {
        "held_out_pair_edges": _edge_list(heldout),
        "held_out_pairs_in_training": _edge_list(heldout_in_training),
        "held_out_pair_endpoint_orphans": endpoint_orphans,
        "held_out_pairs_missing_from_secondary": _edge_list(missing_in_cases),
        "secondary_cases_without_held_out_pair": unmarked_secondary,
        "pure_affine_cases_with_held_out_pair": mixed_affine,
        "interaction_cases_without_held_out_tuple": wrong_interaction,
        "pair_cases_with_held_out_tuple": wrong_pair,
        "pair_cases": len(pair_cases),
    }


def _structured_flat_audit(cases: Sequence[RelationCase]) -> dict:
    """Compare structured and flat execution and rendering for every case."""
    vm_mismatches: list[str] = []
    render_mismatches: list[str] = []
    vm = VM()
    for case in cases:
        structured = vm.run(case.structured)
        flat = vm.run(case.flat)
        same_vm = (
            structured.valid == flat.valid and
            structured.halted == flat.halted and
            structured.strokes == flat.strokes and
            structured.discs == flat.discs and
            structured.regions == flat.regions)
        if not same_vm:
            vm_mismatches.append(case.case_id)
        if to_svg(structured) != to_svg(flat):
            render_mismatches.append(case.case_id)
    return {
        "cases": len(cases),
        "vm_mismatches": vm_mismatches[:8],
        "vm_mismatch_count": len(vm_mismatches),
        "render_mismatches": render_mismatches[:8],
        "render_mismatch_count": len(render_mismatches),
    }


def acceptance_report(corpus: RelationCorpus) -> dict:
    """Every §4.3 clause that can be decided from the corpus alone, as evidence.

    Returns findings, never a verdict: `accepts` applies the rules. That split is
    the same one `dm.eval.feedback_evidence` makes and for the same reason -- a
    function that computes a number and immediately approves it has proved
    nothing (`docs/directions.md` §7 invariant 11).
    """
    by_stratum = {stratum: corpus.of(stratum) for stratum in ALL_STRATA}
    motif_islands = corpus.islands
    train_ids = {motif for case in by_stratum[STRATUM_TRAIN] for motif in case.groups}
    train_source_ids = {
        source for case in by_stratum[STRATUM_TRAIN] for source in case.source_ids}
    train_tuples = {item for case in by_stratum[STRATUM_TRAIN] for item in case.plan}
    shared = {stratum: sorted({motif for case in cases for motif in case.groups}
                              & train_ids)
              for stratum, cases in by_stratum.items()}
    source_shared = {
        stratum: sorted({source for case in cases for source in case.source_ids}
                        & train_source_ids)
        for stratum, cases in by_stratum.items()
    }

    strata: dict[str, dict] = {}
    for stratum, cases in by_stratum.items():
        units = {component_of(case, corpus.labels) for case in cases}
        strata[stratum] = {
            "cases": len(cases),
            "components": len(units),
            "islands": len({motif_islands[motif] for case in cases
                            for motif in case.groups if motif in motif_islands}),
            "motifs_shared_with_training": len(shared[stratum]),
            "source_identities_shared_with_training": len(source_shared[stratum]),
            "venues": dict(sorted(
                (venue, sum(1 for case in cases if case.venue == venue))
                for venue in VENUES)),
            "positives": sum(1 for case in cases if case.actions),
            "novel_pairs": sum(1 for case in cases if case.novel_pair),
            "target_bytes": sum(case.target_stop - case.target_start
                                for case in cases),
        }

    # A held-out tuple may only be an unseen *combination*. Checked against the
    # atoms training actually contains, not against the atoms the space declares:
    # a venue whose geometry refused every case carrying some atom would leave it
    # unsupported however the split was drawn (`docs/copy-relation.md` §1.1).
    supported = _atom_support(train_tuples)
    orphaned = sorted(
        f"{name}={value}"
        for venue, held in corpus.heldout_tuples.items()
        for name, value in _atom_support(held) - supported)

    source_of = {case.case_id: case for case in corpus.cases}
    destroyed = by_stratum[STRATUM_DESTROYED]
    residual = sum(1 for case in destroyed
                   if copy_actions_at(case.flat, case.target_start,
                                      case.target_stop, AUDIT_TRANSLATIONS))
    # Asked of the enumerator over the *whole* program, not of the spliced
    # boundary alone and never of scope provenance, which is empty for a flat
    # program however much relation it contains.
    generic_positives = sum(1 for case in by_stratum[STRATUM_GENERIC]
                            if derivable_boundaries(case.flat))
    # Reported, never a rejection: a destroyed program inherits whatever
    # coincidental relations its *source* already carried, and filtering on that
    # would select the arm on its content. What the splice may not do is add one,
    # which `destroy` enforces row by row.
    destroyed_elsewhere = sum(1 for case in destroyed
                              if derivable_boundaries(case.flat))
    introduced = sum(
        1 for case in destroyed
        if not set(derivable_boundaries(case.flat)) <= set(
            derivable_boundaries(source_of[case.source_case_id].flat)))
    donor_faults: list[str] = []
    donor_ids: list[str] = []
    donor_namespaces: dict[str, tuple] = {}
    labels = corpus.labels
    for case in destroyed:
        source = source_of.get(case.source_case_id)
        donor = source_of.get(case.donor_case_id)
        if source is None:
            donor_faults.append(f"{case.case_id}: missing recipient {case.source_case_id}")
            continue
        if donor is None:
            donor_faults.append(f"{case.case_id}: missing donor {case.donor_case_id}")
            continue
        donor_ids.append(case.donor_case_id)
        donor_namespaces[case.case_id] = _destroy_namespace(source, labels)
        if case.donor_case_id == case.source_case_id:
            donor_faults.append(f"{case.case_id}: donor is recipient")
        if case.donor_groups != donor.groups:
            donor_faults.append(f"{case.case_id}: donor groups do not match donor case")
        if case.donor_source_ids != donor.source_ids:
            donor_faults.append(
                f"{case.case_id}: donor source identities do not match donor case")
        if _destroy_namespace(source, labels) != _destroy_namespace(donor, labels):
            donor_faults.append(f"{case.case_id}: donor crosses destruction namespace")
    duplicate_donors = len(donor_ids) - len(set(donor_ids))
    source_ids_for_donors = set(donor_ids)
    recipient_ids_for_donors = {case.source_case_id for case in destroyed}
    if source_ids_for_donors != recipient_ids_for_donors:
        donor_faults.append("destroyed donor map is not a closed permutation")
    pair_audit = _pair_audit(corpus, by_stratum)
    structured_flat = _structured_flat_audit(corpus.cases)
    flat_arms = {stratum: corpus.programs(stratum) for stratum in ALL_STRATA
                 if corpus.programs(stratum)}
    structured_arms = {
        f"{stratum}.structured": [case.structured for case in by_stratum[stratum]]
        for stratum in ALL_STRATA if by_stratum[stratum]
    }
    census = vm_census({**flat_arms, **structured_arms})
    census["structured_flat"] = structured_flat
    return {
        "schema": CORPUS_SCHEMA,
        "config": {
            **{field: getattr(corpus.config, field)
               for field in ("n_train_synthetic", "n_train_composed", "n_eval",
                             "n_generic", "train_motifs", "eval_motifs",
                             "heldout_tuple_fraction", "heldout_pair_fraction",
                             "composed_limit", "data_seed", "provenance")},
            "composed_categories": list(corpus.config.composed_categories),
        },
        "caps": {
            "max_source_gap_instructions": MAX_SOURCE_GAP_INSTRUCTIONS,
            "max_source_instructions": MAX_SOURCE_INSTRUCTIONS,
            "max_source_bytes": MAX_SOURCE_BYTES,
            "min_source_instructions": MIN_SOURCE_INSTRUCTIONS,
            "min_source_bytes": MIN_SOURCE_BYTES,
            "min_source_coords": MIN_SOURCE_COORDS,
            "d4_support": list(D4_SUPPORT),
            "count_support": list(COUNT_SUPPORT),
            "translation_support": list(TRANSLATION_SUPPORT),
            "audit_translations": [min(AUDIT_TRANSLATIONS), max(AUDIT_TRANSLATIONS)],
        },
        "strata": strata,
        "coverage": coverage_report(corpus.cases),
        "length_bins": length_bin_report(corpus.cases, LENGTH_BIN_EDGES),
        "candidate_spans": candidate_span_report(
            [case for case in corpus.cases if case.actions]),
        "generic": {
            "cases": len(by_stratum[STRATUM_GENERIC]),
            "with_derivable_relation": generic_positives,
        },
        "split": {
            "motif_identities_shared": {stratum: len(found)
                                        for stratum, found in shared.items()},
            "motif_disjoint_strata": list(MOTIF_DISJOINT_STRATA),
            "motif_disjointness_violations": sorted(
                stratum for stratum in MOTIF_DISJOINT_STRATA if shared[stratum]),
            "source_identity_disjointness_violations": sorted(
                stratum for stratum in MOTIF_DISJOINT_STRATA if source_shared[stratum]),
            "untrained_motifs_in_nested_primary": sorted(
                {case.groups[0] for case in by_stratum[STRATUM_NESTED]}
                - train_ids)[:8],
            "held_out_tuples": {venue: [list(item) for item in held]
                                for venue, held in corpus.heldout_tuples.items()},
            "held_out_tuples_in_training": sorted(
                f"{venue}:{item}" for venue, held in corpus.heldout_tuples.items()
                for item in held if item in train_tuples),
            "orphaned_atoms": orphaned,
            "duplicate_case_ids": len(corpus.cases) - len(source_of),
            "held_out_pairs": corpus.reports["held_out_pairs"],
            **pair_audit,
            "leaked_into_training": corpus.reports.get("leaked_into_training", 0),
            "island_faults": island_faults(corpus.cases, motif_islands)[:8],
            "island_fault_count": len(island_faults(corpus.cases, motif_islands)),
        },
        "destroyed": {
            **corpus.reports["destroyed"],
            "residual_relations": residual,
            # Both equalities compare the control against the *sources it was
            # spliced from*, program by program and byte by byte. A permutation
            # of whole blocks between length-and-skeleton-matched rows preserves
            # each program's length and the arm's byte multiset exactly, so
            # anything other than equality is a splice defect.
            "length_mismatches": sum(
                1 for case in destroyed
                if len(case.flat) != len(source_of[case.source_case_id].flat)),
            "byte_census_matches_source": (
                _byte_census([case.flat for case in destroyed])
                == _byte_census([source_of[case.source_case_id].flat
                                 for case in destroyed])),
            "sources": len({case.source_case_id for case in destroyed}),
            "with_derivable_relation_anywhere": destroyed_elsewhere,
            "introduced_relations": introduced,
            "donor_provenance_faults": donor_faults,
            "duplicate_donors": duplicate_donors,
            "donor_namespaces": {
                case_id: list(namespace)
                for case_id, namespace in sorted(donor_namespaces.items())
            },
        },
        "build": corpus.reports["venues"],
        "fingerprints": {stratum: fingerprint(corpus.programs(stratum), [])
                         for stratum in ALL_STRATA},
        "vm_census": census,
    }


#: The minimum accepted connected components a co-primary stratum must carry.
#: `docs/copy-relation.md` §2.4 freezes it before any model exists, and it is a
#: property of the corpus rather than of a result, so acceptance owns it.
MIN_CONFIRMATORY_COMPONENTS = 128


def accepts(report: dict) -> tuple[bool, list[str]]:
    """Whether a traced corpus clears every model-blind acceptance clause.

    Returns the verdict *and* every failure, because a build that fails three
    clauses and reports one sends its author round the loop three times. Fails
    closed on a missing field: absent is not the same as satisfied
    (`docs/copy-relation.md` §7 invariant 1).
    """
    problems: list[str] = []
    if not isinstance(report, dict):
        return False, ["the acceptance report is missing or malformed"]
    if report.get("schema") != CORPUS_SCHEMA:
        problems.append(
            f"the acceptance report has schema {report.get('schema')!r}; expected "
            f"{CORPUS_SCHEMA}")

    def section(name: str) -> dict | None:
        value = report.get(name)
        if not isinstance(value, dict):
            problems.append(f"acceptance report is missing section {name!r}")
            return None
        return value

    def required(body: dict | None, name: str, kind: type | tuple[type, ...],
                 section_name: str) -> object:
        if body is None or name not in body:
            problems.append(f"acceptance report is missing {section_name}.{name}")
            return None
        value = body[name]
        int_kind = kind is int or (isinstance(kind, tuple) and int in kind)
        wrong_int = int_kind and isinstance(value, bool)
        if wrong_int or not isinstance(value, kind):
            problems.append(
                f"acceptance report field {section_name}.{name} has the wrong type")
            return None
        return value

    coverage = section("coverage")
    coverage_value = required(coverage, "coverage", (int, float), "coverage")
    covered = required(coverage, "covered", int, "coverage")
    positives = required(coverage, "positives", int, "coverage")
    split = section("split")
    split_lists = (
        "motif_disjointness_violations", "source_identity_disjointness_violations",
        "held_out_tuples_in_training", "orphaned_atoms", "island_faults",
        "untrained_motifs_in_nested_primary", "held_out_pair_edges",
        "held_out_pairs_in_training", "held_out_pair_endpoint_orphans",
        "held_out_pairs_missing_from_secondary",
        "secondary_cases_without_held_out_pair",
        "pure_affine_cases_with_held_out_pair",
        "interaction_cases_without_held_out_tuple",
        "pair_cases_with_held_out_tuple")
    for name in split_lists:
        required(split, name, list, "split")
    for name in ("duplicate_case_ids", "island_fault_count", "leaked_into_training",
                 "held_out_pairs", "pair_cases"):
        required(split, name, int, "split")
    generic = section("generic")
    generic_cases = required(generic, "cases", int, "generic")
    generic_relations = required(generic, "with_derivable_relation", int, "generic")
    spans = section("candidate_spans")
    cap = required(spans, "cap", int, "candidate_spans")
    bound = required(spans, "bound", int, "candidate_spans")
    bins = section("length_bins")
    past_edge = required(bins, "spans_past_last_edge", int, "length_bins")
    empty_bins = required(bins, "empty_bins", list, "length_bins")
    destroyed = section("destroyed")
    for name in ("residual_relations", "length_mismatches", "introduced_relations",
                 "duplicate_donors"):
        required(destroyed, name, int, "destroyed")
    census_match = required(destroyed, "byte_census_matches_source", bool, "destroyed")
    required(destroyed, "donor_provenance_faults", list, "destroyed")
    strata = section("strata")
    if strata is not None:
        for name in ALL_STRATA:
            item = strata.get(name)
            if not isinstance(item, dict):
                problems.append(f"acceptance report is missing stratum {name!r}")
                continue
            for field_name, field_type in (
                    ("cases", int), ("components", int), ("islands", int),
                    ("motifs_shared_with_training", int),
                    ("source_identities_shared_with_training", int),
                    ("venues", dict), ("positives", int),
                    ("novel_pairs", int), ("target_bytes", int)):
                required(item, field_name, field_type, f"strata.{name}")
    config = section("config")
    for name, kind in (
            ("n_train_synthetic", int), ("n_train_composed", int),
            ("n_eval", int), ("n_generic", int), ("train_motifs", int),
            ("eval_motifs", int), ("heldout_tuple_fraction", (int, float)),
            ("heldout_pair_fraction", (int, float)), ("composed_limit", (int, type(None))),
            ("data_seed", int), ("provenance", str),
            ("composed_categories", list)):
        required(config, name, kind, "config")
    for name in ("caps", "build", "fingerprints"):
        section(name)
    # These comparisons are deliberately guarded by type checks above: malformed
    # reports collect structural failures rather than raising while being audited.
    if isinstance(covered, int) and isinstance(positives, int):
        if covered < 0 or positives < 0 or covered > positives:
            problems.append(
                f"coverage counts are inconsistent: {covered}/{positives}")
        expected_coverage = covered / max(1, positives)
        if isinstance(coverage_value, (int, float)) and \
                coverage_value != expected_coverage:
            problems.append(
                f"reported coverage {coverage_value!r} does not match raw counts "
                f"{covered}/{positives}")
    if isinstance(coverage_value, (int, float)) and coverage_value != 1.0:
        problems.append(
            f"candidate caps cover {covered}/{positives} accepted positives, not all "
            "of them")
    if isinstance(split, dict):
        if split.get("motif_disjointness_violations"):
            problems.append(
                f"strata {split['motif_disjointness_violations']} share motif "
                "identities with training and are declared disjoint")
        if split.get("source_identity_disjointness_violations"):
            problems.append(
                f"strata {split['source_identity_disjointness_violations']} share "
                "source identities with training and are declared disjoint")
        if split.get("held_out_tuples_in_training"):
            problems.append(
                f"withheld tuples {split['held_out_tuples_in_training']} occur in training")
        if split.get("orphaned_atoms"):
            problems.append(
                f"withheld tuples orphan atoms {split['orphaned_atoms']}; a numerically "
                "unseen value is not identifiable from a flat prefix")
        if split.get("duplicate_case_ids"):
            problems.append(f"{split['duplicate_case_ids']} duplicate case identifiers")
        if split.get("untrained_motifs_in_nested_primary"):
            problems.append(
                f"the nested co-primary uses motifs training never saw "
                f"({split['untrained_motifs_in_nested_primary']}), confounding "
                "composition with motif transfer")
        if split.get("island_fault_count"):
            problems.append(
                f"{split['island_fault_count']} cases span more than one motif island, "
                f"rejoining the dependence graph: {split.get('island_faults')}")
        if split.get("leaked_into_training"):
            problems.append(
                f"{split['leaked_into_training']} held-out programs are byte-identical "
                "to a training program")
        if split.get("held_out_pairs_in_training"):
            problems.append("a declared held-out motif pair occurs in training")
        if split.get("held_out_pair_endpoint_orphans"):
            problems.append("a declared held-out motif pair has an untrained endpoint")
        if split.get("held_out_pairs_missing_from_secondary"):
            problems.append("a declared held-out motif pair has no accepted secondary case")
        if split.get("secondary_cases_without_held_out_pair"):
            problems.append("a pair/interaction case is not backed by a held-out pair")
        if split.get("pure_affine_cases_with_held_out_pair"):
            problems.append("the affine co-primary contains pair-novel cases")
        if split.get("interaction_cases_without_held_out_tuple"):
            problems.append("an interaction case is missing tuple novelty")
        if split.get("pair_cases_with_held_out_tuple"):
            problems.append("a pure pair case contains tuple novelty")
    if isinstance(generic_relations, int) and generic_relations:
        problems.append(
            f"{generic_relations} generic programs contain a derivable copy relation; "
            "a false-copy denominator holding true copies is not a false-copy denominator")
    if isinstance(generic_cases, int) and not generic_cases:
        problems.append("the generic control is empty")
    if not isinstance(cap, int) or isinstance(cap, bool) or \
            not isinstance(bound, int) or isinstance(bound, bool) or bound > cap:
        problems.append(
            f"the caps allow {bound!r} candidate spans at a boundary, past the frozen "
            f"scoring cap {cap!r}; the annotated action could be truncated out of the "
            "set the head scores")
    if isinstance(past_edge, int) and past_edge:
        problems.append(
            f"{past_edge} accepted spans fall past the last length-bin edge and index "
            "no bin")
    if isinstance(empty_bins, list) and empty_bins:
        problems.append(
            f"length bins {empty_bins} are never reached; those key embedding rows "
            "would receive no gradient")
    if isinstance(destroyed, dict):
        if destroyed.get("residual_relations"):
            problems.append(
                f"{destroyed['residual_relations']} destroyed rows still admit a copy "
                "action; they are not negatives")
        if destroyed.get("length_mismatches"):
            problems.append(f"{destroyed['length_mismatches']} destroyed rows changed length")
        if census_match is not True:
            problems.append("the destroyed arm's byte census differs from its sources'")
        if destroyed.get("introduced_relations"):
            problems.append(
                f"{destroyed['introduced_relations']} destroyed programs admit a copy "
                "action at a boundary their source was clean at; the splice added a "
                "relation instead of removing one")
        if destroyed.get("duplicate_donors"):
            problems.append("the destroyed donor map reuses a donor case")
        if destroyed.get("donor_provenance_faults"):
            problems.append("the destroyed arm has invalid donor provenance")
    if isinstance(strata, dict):
        for stratum in CONFIRMATORY_STRATA:
            found = (strata.get(stratum) or {}).get("components")
            if not isinstance(found, int) or isinstance(found, bool) \
                    or found < MIN_CONFIRMATORY_COMPONENTS:
                problems.append(
                    f"{stratum} carries {found} connected components, below the frozen "
                    f"{MIN_CONFIRMATORY_COMPONENTS}")
    problems.extend(vm_census_problems(report.get("vm_census")))
    return not problems, problems


def vm_census_problems(census: object) -> list[str]:
    """Direction 4's own reading of a VM census, in its own namespace.

    `dm.data.feedback_audit.audit_accepts` is Direction 3's: it insists the arms
    be named `relational`, `destroyed` and `validation`, which are that
    direction's arms and not these. The census *builder* is shared because it
    computes a fact about programs; the acceptance rule is not, because a rule is
    a contract and Direction 4's contract is its own
    (`docs/copy-relation.md` §7 invariant 2).

    Every arm must be all-valid, all-halted and zero-fault. An empty arm is a
    missing arm, not a clean one.
    """
    if not isinstance(census, dict) or census.get("schema") != AUDIT_SCHEMA:
        return ["the VM census is missing or of an unknown schema"]
    arms = census.get("arms")
    if not isinstance(arms, dict) or not arms:
        return ["the VM census names no program arms"]
    problems: list[str] = []
    for name, arm in sorted(arms.items()):
        if not isinstance(arm, dict):
            problems.append(f"VM census arm {name!r} is malformed")
            continue
        total = arm.get("programs")
        if not isinstance(total, int) or total <= 0:
            problems.append(f"VM census arm {name!r} is empty")
            continue
        if arm.get("valid_halted_programs") != total:
            problems.append(
                f"VM census arm {name!r} has {total - arm.get('valid_halted_programs', 0)} "
                "programs that are not valid and halted")
        if arm.get("fault_events"):
            problems.append(
                f"VM census arm {name!r} raised {arm['fault_events']} VM faults")
    comparison = census.get("structured_flat")
    if not isinstance(comparison, dict):
        problems.append("the VM census is missing structured-versus-flat equality")
    else:
        for name in ("cases", "vm_mismatch_count", "render_mismatch_count"):
            value = comparison.get(name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                problems.append(
                    f"structured-versus-flat VM census field {name!r} is missing")
        if comparison.get("vm_mismatch_count"):
            problems.append("structured and flat programs do not have equal VM traces")
        if comparison.get("render_mismatch_count"):
            problems.append("structured and flat programs do not render equally")
    return problems


# ---------------------------------------------------------------------------
# the manifest


def manifest(corpus: RelationCorpus) -> dict:
    """The frozen record of a build, content-addressed like every other artifact.

    Carries the acceptance report, the split in full and the case index, so an
    auditor can replay the split without rebuilding the corpus and can rebuild
    the corpus without trusting the index. The two digests it grows on write are
    kept apart by name for the reason `PLAN.md` names as a trap: the canonical
    payload digest identifies the *body* and the file digest identifies the
    *bytes*, and they are never equal.
    """
    report = acceptance_report(corpus)
    verdict, problems = accepts(report)
    body = {
        "schema": MANIFEST_SCHEMA,
        "status": "frozen",
        "direction": DIRECTION,
        "acceptance": report,
        "accepted": verdict,
        "problems": problems,
        "cases": [
            {
                "case_id": case.case_id,
                "venue": case.venue,
                "stratum": case.stratum,
                "component": component_of(case, corpus.labels),
                "groups": list(case.groups),
                "source_ids": list(case.source_ids),
                "plan": [list(item) for item in case.plan],
                "bytes": len(case.flat),
                "target": [case.target_start, case.target_stop],
                "actions": [list(action.key()) for action in case.actions],
                "novel_pair": case.novel_pair,
                "source_case_id": case.source_case_id,
                "donor_case_id": case.donor_case_id,
                "donor_groups": list(case.donor_groups),
                "donor_source_ids": list(case.donor_source_ids),
            }
            for case in corpus.cases
        ],
    }
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    body["corpus_sha256"] = hashlib.sha256(encoded).hexdigest()
    return body


def write_manifest(path: Path, body: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f".{path.name}.tmp")
    staged.write_text(json.dumps(body, indent=1, sort_keys=True) + "\n")
    staged.replace(path)


def manifest_digests(path: Path) -> dict:
    """Both hashes derived from one immutable manifest-byte snapshot."""
    snapshot = read_manifest_snapshot(path)
    return {
        "path": snapshot.path,
        "canonical_payload_sha256": snapshot.canonical_payload_sha256,
        "recorded_canonical_sha256": snapshot.recorded_canonical_sha256,
        "file_sha256": snapshot.file_sha256,
    }


# ---------------------------------------------------------------------------
# manifest validation
#
# `accepts` decides whether a *corpus* is good enough to freeze. Nothing decided
# whether the *manifest* describing it is even internally honest, and the two are
# different questions: every acceptance clause reads the recorded acceptance
# report, so a body whose report says one thing and whose case index says another
# passed every clause the project had. Direction 3 paid exactly this cost when its
# F5b corpus clause compared a manifest against itself. The rule here is that no
# number a later stage reads may be believed on the manifest's own authority when
# the case index can produce it independently (`docs/directions.md` §7
# invariant 11).


class ManifestRefused(ValueError):
    """A manifest was read and refused. An *expected* outcome, not a defect.

    Derived from `ValueError` so every existing caller keeps working unchanged --
    `relation_contract.load_protocol` and both freeze Adapters already treat a bad
    corpus as a `ValueError` -- and named so that a caller can tell a refusal from
    a validator that broke.

    Catching the builtin could not tell them apart, because validation code raises
    that class too: a five-thousand-digit histogram key reached `int(key)` and
    surfaced as CPython's integer-conversion `ValueError`, which the audit Adapter
    printed as an ordinary refusal, while an `OverflowError` two fields away
    crashed loudly. Which of the two a reviewer saw depended on the failed
    operation rather than on the contract boundary. Only this type crosses that
    boundary; anything else escaping validation is a defect and stays loud.
    """


#: Exact envelope of a manifest body: every key, and no key that is not here.
#:
#: Exhaustive in both directions on purpose. A missing field is caught by every
#: reader that needs it, but an *extra* field is how a second, weaker copy of an
#: invariant arrives -- and invariant 17 says the weaker copy then wins silently.
MANIFEST_ENVELOPE: dict[str, type | tuple[type, ...]] = {
    "schema": int,
    "status": str,
    "direction": int,
    "acceptance": dict,
    "accepted": bool,
    "problems": list,
    "cases": list,
    "corpus_sha256": str,
}

#: Exact shape of one case-index record.
CASE_RECORD_FIELDS: dict[str, type | tuple[type, ...]] = {
    "case_id": str,
    "venue": str,
    "stratum": str,
    "component": int,
    "groups": list,
    "source_ids": list,
    "plan": list,
    "bytes": int,
    "target": list,
    "actions": list,
    "novel_pair": bool,
    "source_case_id": str,
    "donor_case_id": str,
    "donor_groups": list,
    "donor_source_ids": list,
}

#: The venue builds `build` runs, in the order it runs them.
#:
#: A frozen list rather than "whatever the build reported", so a build that
#: silently stopped running one of its venues -- the transfer stratum, say --
#: produces a manifest that fails validation instead of one that merely has an
#: empty stratum.
VENUE_BUILDS: tuple[str, ...] = ("v1_train", "v1_affine", "v1_nested",
                                 "v1_nested_transfer", "generic", "v2",
                                 "v2_transfer")

#: Exact keys of every acceptance section, and of the records inside them.
_ACCEPTANCE_SECTIONS: tuple[str, ...] = (
    "schema", "config", "caps", "strata", "coverage", "length_bins",
    "candidate_spans", "generic", "split", "destroyed", "build", "fingerprints",
    "vm_census")

_CAPS_FIELDS: dict[str, type | tuple[type, ...]] = {
    "max_source_gap_instructions": int, "max_source_instructions": int,
    "max_source_bytes": int, "min_source_instructions": int,
    "min_source_bytes": int, "min_source_coords": int,
    "d4_support": list, "count_support": list, "translation_support": list,
    "audit_translations": list,
}

_STRATUM_FIELDS: dict[str, type | tuple[type, ...]] = {
    "cases": int, "components": int, "islands": int,
    "motifs_shared_with_training": int,
    "source_identities_shared_with_training": int,
    "venues": dict, "positives": int, "novel_pairs": int, "target_bytes": int,
}

_COVERAGE_FIELDS: dict[str, type | tuple[type, ...]] = {
    "positives": int, "covered": int, "coverage": (int, float),
    "single_derivation": int, "multiple_derivations": int,
    "widest_source_bytes": int, "widest_source_instructions": int,
    "max_source_bytes": int, "max_source_instructions": int,
}

_LENGTH_BIN_FIELDS: dict[str, type | tuple[type, ...]] = {
    "edges": list, "counts": list, "empty_bins": list,
    "spans_past_last_edge": int, "widest_source_bytes": int,
}

_CANDIDATE_SPAN_FIELDS: dict[str, type | tuple[type, ...]] = {
    "bound": int, "cap": int, "observed_widest": int,
    "observed_mean": (int, float), "sampled_cases": int,
    "sampled_boundaries": int,
}

_GENERIC_FIELDS: dict[str, type | tuple[type, ...]] = {
    "cases": int, "with_derivable_relation": int,
}

_SPLIT_FIELDS: dict[str, type | tuple[type, ...]] = {
    "motif_identities_shared": dict, "motif_disjoint_strata": list,
    "motif_disjointness_violations": list,
    "source_identity_disjointness_violations": list,
    "untrained_motifs_in_nested_primary": list, "held_out_tuples": dict,
    "held_out_tuples_in_training": list, "orphaned_atoms": list,
    "duplicate_case_ids": int, "held_out_pairs": int,
    "held_out_pair_edges": list, "held_out_pairs_in_training": list,
    "held_out_pair_endpoint_orphans": list,
    "held_out_pairs_missing_from_secondary": list,
    "secondary_cases_without_held_out_pair": list,
    "pure_affine_cases_with_held_out_pair": list,
    "interaction_cases_without_held_out_tuple": list,
    "pair_cases_with_held_out_tuple": list, "pair_cases": int,
    "leaked_into_training": int, "island_faults": list, "island_fault_count": int,
}

_DESTROYED_FIELDS: dict[str, type | tuple[type, ...]] = {
    "accepted": int, "offered": int, "notes": dict, "residual_relations": int,
    "length_mismatches": int, "byte_census_matches_source": bool,
    "sources": int, "with_derivable_relation_anywhere": int,
    "introduced_relations": int, "donor_provenance_faults": list,
    "duplicate_donors": int, "donor_namespaces": dict,
}

_BUILD_FIELDS: dict[str, type | tuple[type, ...]] = {
    "accepted": int, "drawn": int, "rejections": dict,
}

_FINGERPRINT_FIELDS: dict[str, type | tuple[type, ...]] = {
    "train": str, "val": str, "n_train": int, "n_val": int,
    "bytes_train": int, "bytes_val": int,
}

_STRUCTURED_FLAT_FIELDS: dict[str, type | tuple[type, ...]] = {
    "cases": int, "vm_mismatches": list, "vm_mismatch_count": int,
    "render_mismatches": list, "render_mismatch_count": int,
}

#: How long a truncated evidence list may be. `acceptance_report` slices its
#: mismatch and fault lists to this, so a count and a list agree only in a way
#: that has to be stated: `len(list) == min(count, 8)`. Naming the number here
#: means a change to the slice moves the check with it.
EVIDENCE_LIST_LIMIT = 8

#: The VM census fields one arm carries. Read from the census builder rather than
#: restated, because Direction 3 owns that shape and a second copy of it here
#: would be the weaker one (`docs/directions.md` §7 invariant 17). The *rule*
#: applied to those fields stays Direction 4's, in `vm_census_problems`.
VM_ARM_FIELDS: tuple[str, ...] = _VM_ARM_FIELDS


def _wrong_type(value: object, kind: type | tuple[type, ...]) -> bool:
    """Whether `value` fails `kind`, counting `bool` as not an `int`.

    JSON has one number type and Python makes `True` an `int`, so a field that
    should hold a count will accept a boolean silently. Every count in this
    manifest is evidence, and `True` is not a count.
    """
    int_kind = kind is int or (isinstance(kind, tuple) and int in kind)
    if int_kind and isinstance(value, bool) and kind is not bool:
        return True
    return not isinstance(value, kind)


def _shape_problems(body: object, fields: Mapping[str, type | tuple[type, ...]],
                    where: str) -> list[str]:
    """Exact-key, exact-type checking of one record, in both directions."""
    if not isinstance(body, dict):
        return [f"{where} is missing or is not an object"]
    problems = [f"{where} is missing {name!r}" for name in fields
                if name not in body]
    problems.extend(f"{where} carries unexpected field {name!r}"
                    for name in sorted(set(body) - set(fields)))
    problems.extend(
        f"{where}.{name} has the wrong type"
        for name, kind in fields.items()
        if name in body and _wrong_type(body[name], kind))
    return sorted(problems)


def _is_hex_digest(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _record_pairs(record: Mapping) -> set[frozenset[str]]:
    """`_pairs_of` for one case-index record rather than one `RelationCase`."""
    groups = record["groups"]
    head = groups[0]
    return {frozenset({head, other}) for other in groups[1:] if other != head}


def _canonical_components(cases: Sequence[Mapping]) -> dict[str, int]:
    """`components` and `component_of` for the case index, rebuilt from `groups`.

    **The stored `component` is a claim, not a measurement.** It is a label over
    the motif co-occurrence graph, the graph is `groups`, and the manifest carries
    both -- so a body that rewrites the labels and the count beside them is
    self-consistent everywhere except against the graph it never touched. The
    128-component floor `docs/copy-relation.md` §2.4 freezes reads exactly that
    count, which makes this the one recomputation a forgery most wants skipped.

    Deliberately the same union-find `components` runs, over the same vertices and
    the same `groups[0]` edges, so agreement with the build is exact rather than
    approximate and a divergence is a real difference rather than two spellings of
    one graph.
    """
    vertices = [motif for record in cases for motif in record["groups"]]
    edges = [(record["groups"][0], other)
             for record in cases for other in record["groups"][1:]]
    labels = _connected(vertices, edges)
    return {record["case_id"]: labels[record["groups"][0]] for record in cases}


def _envelope_problems(body: Mapping) -> list[str]:
    problems = _shape_problems(body, MANIFEST_ENVELOPE, "the manifest")
    if body.get("schema") != MANIFEST_SCHEMA:
        problems.append(
            f"the manifest has schema {body.get('schema')!r}; expected "
            f"{MANIFEST_SCHEMA}")
    if body.get("direction") != DIRECTION:
        problems.append(
            f"the manifest declares direction {body.get('direction')!r}; this is "
            f"Direction {DIRECTION}")
    if body.get("status") != "frozen":
        problems.append(f"the manifest is not frozen: status "
                        f"{body.get('status')!r}")
    if not _is_hex_digest(body.get("corpus_sha256")):
        problems.append("the manifest's corpus_sha256 is not a sha256 digest")
    recorded = body.get("problems")
    if isinstance(recorded, list) and not all(isinstance(item, str)
                                              for item in recorded):
        problems.append("the manifest's acceptance problem list is not strings")
    return problems


def _case_index_problems(cases: Sequence) -> list[str]:
    """Whether the case index is a well-formed index of well-formed cases.

    Structure only: whether the corpus is *good* is `accepts`, and whether the
    index agrees with the acceptance report is `_index_agreement_problems`. An
    empty index is a fault here rather than a vacuous pass -- every recomputation
    below is trivially satisfied by no cases at all, which is precisely the
    manifest a rewrite would produce.
    """
    if not cases:
        return [("the case index is empty; every recomputed acceptance fact "
                 "would then be vacuously satisfied")]
    problems: list[str] = []
    seen: set[str] = set()
    duplicates: set[str] = set()
    for position, record in enumerate(cases):
        where = f"cases[{position}]"
        found = _shape_problems(record, CASE_RECORD_FIELDS, where)
        if found:
            problems.extend(found)
            continue
        where = f"case {record['case_id']!r}"
        if not record["case_id"]:
            problems.append(f"cases[{position}] has an empty case identifier")
        elif record["case_id"] in seen:
            duplicates.add(record["case_id"])
        seen.add(record["case_id"])
        if record["venue"] not in VENUES:
            problems.append(f"{where} names unknown venue {record['venue']!r}")
        if record["stratum"] not in ALL_STRATA:
            problems.append(f"{where} names unknown stratum "
                            f"{record['stratum']!r}")
        if record["component"] < 0:
            problems.append(f"{where} has a negative component label")
        if record["bytes"] <= 0:
            problems.append(f"{where} has no bytes")
        problems.extend(
            f"{where} has a {name} entry that is not a motif identity"
            for name in ("groups", "source_ids", "donor_groups",
                         "donor_source_ids")
            if not all(isinstance(item, str) and item for item in record[name]))
        groups = record["groups"]
        if not groups or not all(isinstance(item, str) and item
                                 for item in groups):
            problems.append(f"{where} has no usable motif identities")
        elif record["source_ids"] and \
                len(record["source_ids"]) != len(groups):
            # Parallel *or* absent. `synthetic_nested_repeat` motifs are their own
            # provenance and the destroyed splice keeps its recipient's motifs
            # without its source identities, so an empty list is a legitimate
            # "no separate provenance"; a list of the wrong length is not.
            problems.append(
                f"{where} carries {len(record['source_ids'])} source identities "
                f"for {len(groups)} motifs; they are parallel when present")
        problems.extend(_plan_problems(record, where))
        problems.extend(_target_problems(record, where))
        problems.extend(_donor_shape_problems(record, where))
    problems.extend(f"case identifier {case_id!r} appears more than once; the "
                    "index cannot be read by identifier"
                    for case_id in sorted(duplicates))
    return problems


def _plan_problems(record: Mapping, where: str) -> list[str]:
    problems: list[str] = []
    for item in record["plan"]:
        if not isinstance(item, list) or len(item) != 4 or \
                any(_wrong_type(value, int) for value in item):
            problems.append(f"{where} has a plan entry that is not an affine "
                            "tuple of four integers")
            break
    for action in record["actions"]:
        if not isinstance(action, list) or not action or \
                any(_wrong_type(value, int) for value in action):
            problems.append(f"{where} has an action key that is not integers")
            break
    if record["actions"] and record["stratum"] in (STRATUM_GENERIC,
                                                   STRATUM_DESTROYED):
        problems.append(
            f"{where} is in the {record['stratum']} control and carries annotated "
            "actions; that stratum is scored for what it must not do")
    return problems


def _target_problems(record: Mapping, where: str) -> list[str]:
    target = record["target"]
    if len(target) != 2 or any(_wrong_type(value, int) for value in target):
        return [f"{where} has a target that is not a pair of integers"]
    start, stop = target
    problems: list[str] = []
    if not 0 <= start <= stop <= record["bytes"]:
        problems.append(
            f"{where} has target [{start}, {stop}) outside its "
            f"{record['bytes']} bytes")
    if record["actions"] and stop <= start:
        problems.append(f"{where} carries actions over an empty target block")
    # The two controls differ here on purpose and the difference is load-bearing.
    # `generic` is a program no action was ever scheduled on, so it has no block;
    # a `destroyed` row is a relation case whose block was *replaced*, and that
    # block is what its length, byte-census and donor-namespace comparisons are
    # made over. A destroyed row with an empty target is a row whose intervention
    # left no trace to audit.
    if not record["actions"] and record["stratum"] != STRATUM_DESTROYED \
            and stop != start:
        problems.append(f"{where} has no action but a non-empty target block")
    if record["stratum"] == STRATUM_DESTROYED and stop <= start:
        problems.append(
            f"{where} is a destroyed control with an empty spliced block")
    return problems


def _donor_shape_problems(record: Mapping, where: str) -> list[str]:
    """Donor provenance is present exactly on the destroyed control.

    Both directions matter. A destroyed row without a donor cannot have its
    length and byte census compared against the source it was spliced from, which
    is the tautology invariant 11 exists to forbid; a *relation* row carrying
    donor fields is a row whose provenance says it was spliced when it was not.
    """
    donor_fields = ("source_case_id", "donor_case_id", "donor_groups",
                    "donor_source_ids")
    #: `donor_source_ids` is parallel-or-absent for the same reason a case's own
    #: `source_ids` are: a synthetic donor is its own provenance.
    required = ("source_case_id", "donor_case_id", "donor_groups")
    if record["stratum"] == STRATUM_DESTROYED:
        missing = [name for name in required if not record[name]]
        if missing:
            return [f"{where} is a destroyed control with no {name}"
                    for name in missing]
        if record["donor_source_ids"] and \
                len(record["donor_source_ids"]) != len(record["donor_groups"]):
            return [(f"{where} has donor source identities that do not run "
                     "parallel to its donor motifs")]
        return []
    return [f"{where} is not a destroyed control but carries {name}"
            for name in donor_fields if record[name]]


def _acceptance_shape_problems(report: Mapping) -> list[str]:
    """Exact schemas for every acceptance section and the evidence inside them."""
    problems = [f"the acceptance report is missing section {name!r}"
                for name in _ACCEPTANCE_SECTIONS if name not in report]
    problems.extend(f"the acceptance report carries unexpected section {name!r}"
                    for name in sorted(set(report) - set(_ACCEPTANCE_SECTIONS)))
    if report.get("schema") != CORPUS_SCHEMA:
        problems.append(
            f"the acceptance report has schema {report.get('schema')!r}; expected "
            f"{CORPUS_SCHEMA}")
    problems.extend(_config_problems(report.get("config")))
    problems.extend(_caps_problems(report.get("caps")))
    problems.extend(_strata_shape_problems(report.get("strata")))
    problems.extend(_shape_problems(report.get("coverage"), _COVERAGE_FIELDS,
                                    "coverage"))
    problems.extend(_bounds_problems(report))
    problems.extend(_shape_problems(report.get("generic"), _GENERIC_FIELDS,
                                    "generic"))
    problems.extend(_split_shape_problems(report.get("split")))
    problems.extend(_destroyed_shape_problems(report.get("destroyed")))
    problems.extend(_build_shape_problems(report.get("build")))
    problems.extend(_fingerprint_shape_problems(report.get("fingerprints")))
    problems.extend(_census_shape_problems(report.get("vm_census")))
    problems.extend(_truncation_problems(report))
    return problems


def _string_list_problems(body: object, where: str) -> list[str]:
    """A list of names, used where a later clause builds a set out of it.

    Item types matter here and not merely in principle: `set(...)` over a list
    whose entries are themselves lists raises, and an exception is not the
    complete fault list this validator promises.
    """
    if not isinstance(body, list):
        return [f"{where} is not a list"]
    if not all(isinstance(item, str) and item for item in body):
        return [f"{where} holds an entry that is not a name"]
    return []


def _held_out_tuple_problems(held: object) -> list[str]:
    """The withheld affine tuples, per venue, exactly as `held_out_tuples` wrote.

    `_atom_support` reads these to decide whether a withheld tuple orphans an
    atom -- the clause §1.1 rests on -- so their shape is load-bearing rather than
    cosmetic, and a malformed entry has to be named here instead of raising
    several frames further in.
    """
    where = "split.held_out_tuples"
    if not isinstance(held, dict):
        return [f"{where} is not a per-venue object"]
    if set(held) != set(VENUES):
        return [(f"{where} names {sorted(held)}; the contract holds "
                 f"{sorted(VENUES)}")]
    problems: list[str] = []
    for venue in sorted(held):
        items = held[venue]
        if not isinstance(items, list) or any(
                not isinstance(item, list) or len(item) != 4
                or any(_wrong_type(value, int) for value in item)
                for item in items):
            problems.append(
                f"{where}[{venue!r}] is not a list of four-integer affine tuples")
            continue
        if items != sorted(items) or len({tuple(item) for item in items}) != \
                len(items):
            problems.append(
                f"{where}[{venue!r}] is not sorted and duplicate-free")
    return problems


def _pair_edge_problems(edges: object) -> list[str]:
    """`held_out_pair_edges` is the withheld edge *set*, written canonically.

    Every reader turns it into a set of frozensets, which is why a duplicated or
    reordered entry changed nothing a recomputation could see: `held_out_pairs` is
    compared against the deduplicated set, so the count still agreed while the
    evidence a reviewer reads carried an edge the corpus does not. The canonical
    form is what `_edge_list` writes -- each edge two distinct non-empty motif
    identities in sorted order, the edges themselves sorted and duplicate-free --
    so requiring it makes the list and the set the same object.
    """
    def canonical(edge: object) -> bool:
        return (isinstance(edge, list) and len(edge) == 2
                and all(isinstance(node, str) and node for node in edge)
                and edge[0] < edge[1])

    where = "split.held_out_pair_edges"
    if not isinstance(edges, list):
        return [f"{where} is not a list"]
    malformed = [edge for edge in edges if not canonical(edge)]
    if malformed:
        return [(f"{where} is not a canonical edge list: {malformed[0]!r} is "
                 "not two distinct motif identities in sorted order")]
    if edges != sorted(edges) or \
            len({tuple(edge) for edge in edges}) != len(edges):
        return [(f"{where} is not a canonical edge list: it is not sorted and "
                 "duplicate-free")]
    return []


def _split_shape_problems(split: object) -> list[str]:
    """The split section, including the nested tuple and pair evidence.

    The nested evidence is checked here rather than where it is read, because
    `_index_agreement_problems` runs only on sections whose shape is already
    sound: a cross-check against a malformed body reports the malformation under a
    less useful name, or does not get the chance to report it at all.
    """
    problems = _shape_problems(split, _SPLIT_FIELDS, "split")
    if problems or not isinstance(split, dict):
        return problems
    problems.extend(_held_out_tuple_problems(split["held_out_tuples"]))
    problems.extend(_pair_edge_problems(split["held_out_pair_edges"]))
    return problems


def _destroyed_shape_problems(destroyed: object) -> list[str]:
    """The destroyed section, including the fault list a later clause sets."""
    problems = _shape_problems(destroyed, _DESTROYED_FIELDS, "destroyed")
    if problems or not isinstance(destroyed, dict):
        return problems
    problems.extend(_string_list_problems(destroyed["donor_provenance_faults"],
                                          "destroyed.donor_provenance_faults"))
    return problems


def _config_problems(config: object) -> list[str]:
    """The build configuration is exactly a serialised `BuildConfig`.

    Checked by *constructing* one. A manifest whose config cannot be turned back
    into the dataclass that built it cannot be rebuilt from, so `rebuild_verify`
    would have nothing to run and the freeze would rest on the index alone.
    """
    names = {item.name for item in fields(BuildConfig)}
    schema = {item.name: (list if item.name == "composed_categories"
                          else (int, type(None)) if item.name == "composed_limit"
                          else str if item.name == "provenance"
                          else (int, float) if "fraction" in item.name
                          else int)
              for item in fields(BuildConfig)}
    problems = _shape_problems(config, schema, "config")
    if problems or not isinstance(config, dict):
        return problems
    if set(config) != names:
        return problems
    try:
        BuildConfig(**{**config,
                       "composed_categories": tuple(config["composed_categories"])})
    except (TypeError, ValueError) as error:
        problems.append(f"config does not reconstruct a BuildConfig: {error}")
    return problems


def _caps_problems(caps: object) -> list[str]:
    """The recorded caps are the caps the enumerator actually applies.

    `docs/copy-relation.md` §7 invariant 17 makes the caps model-blind and frozen
    before training. A manifest recording caps other than the module's describes
    a corpus built by code that is no longer here, and every coverage number in
    it was computed under rules a later stage would not reproduce.
    """
    problems = _shape_problems(caps, _CAPS_FIELDS, "caps")
    if problems or not isinstance(caps, dict):
        return problems
    expected = {
        "max_source_gap_instructions": MAX_SOURCE_GAP_INSTRUCTIONS,
        "max_source_instructions": MAX_SOURCE_INSTRUCTIONS,
        "max_source_bytes": MAX_SOURCE_BYTES,
        "min_source_instructions": MIN_SOURCE_INSTRUCTIONS,
        "min_source_bytes": MIN_SOURCE_BYTES,
        "min_source_coords": MIN_SOURCE_COORDS,
        "d4_support": list(D4_SUPPORT),
        "count_support": list(COUNT_SUPPORT),
        "translation_support": list(TRANSLATION_SUPPORT),
        "audit_translations": [min(AUDIT_TRANSLATIONS), max(AUDIT_TRANSLATIONS)],
    }
    return [f"caps.{name} records {caps[name]!r}; the executable contract holds "
            f"{want!r}"
            for name, want in expected.items() if caps[name] != want]


def _strata_shape_problems(strata: object) -> list[str]:
    if not isinstance(strata, dict):
        return ["the acceptance report's strata section is missing"]
    problems = [f"the acceptance report is missing stratum {name!r}"
                for name in ALL_STRATA if name not in strata]
    problems.extend(f"the acceptance report carries unknown stratum {name!r}"
                    for name in sorted(set(strata) - set(ALL_STRATA)))
    for name in ALL_STRATA:
        if name not in strata:
            continue
        found = strata[name]
        problems.extend(_shape_problems(found, _STRATUM_FIELDS,
                                        f"strata.{name}"))
        if not isinstance(found, dict):
            continue
        venues = found.get("venues")
        if isinstance(venues, dict) and set(venues) != set(VENUES):
            problems.append(f"strata.{name}.venues does not name every venue")
    return problems


def _bounds_problems(report: Mapping) -> list[str]:
    """Bin edges and candidate caps are the module's, not the manifest's."""
    problems = _shape_problems(report.get("length_bins"), _LENGTH_BIN_FIELDS,
                               "length_bins")
    problems.extend(_shape_problems(report.get("candidate_spans"),
                                    _CANDIDATE_SPAN_FIELDS, "candidate_spans"))
    bins = report.get("length_bins")
    if isinstance(bins, dict) and not problems:
        if bins["edges"] != list(LENGTH_BIN_EDGES):
            problems.append(
                f"length_bins.edges records {bins['edges']!r}; the executable "
                f"contract holds {list(LENGTH_BIN_EDGES)!r}")
        if len(bins["counts"]) != len(LENGTH_BIN_EDGES):
            problems.append("length_bins.counts does not have one entry per bin")
        elif bins["empty_bins"] != [index for index, count
                                    in enumerate(bins["counts"]) if not count]:
            problems.append(
                "length_bins.empty_bins disagrees with length_bins.counts")
    spans = report.get("candidate_spans")
    if isinstance(spans, dict) and "bound" in spans and "cap" in spans:
        if spans["bound"] != CANDIDATE_SPAN_BOUND:
            problems.append(
                f"candidate_spans.bound records {spans['bound']!r}; the caps give "
                f"{CANDIDATE_SPAN_BOUND}")
        if spans["cap"] != MAX_CANDIDATE_SPANS:
            problems.append(
                f"candidate_spans.cap records {spans['cap']!r}; the executable "
                f"contract holds {MAX_CANDIDATE_SPANS}")
    coverage = report.get("coverage")
    if isinstance(coverage, dict):
        for name, want in (("max_source_bytes", MAX_SOURCE_BYTES),
                           ("max_source_instructions", MAX_SOURCE_INSTRUCTIONS)):
            if name in coverage and coverage[name] != want:
                problems.append(f"coverage.{name} records {coverage[name]!r}; the "
                                f"executable contract holds {want}")
    split = report.get("split")
    if isinstance(split, dict) and "motif_disjoint_strata" in split and \
            split["motif_disjoint_strata"] != list(MOTIF_DISJOINT_STRATA):
        problems.append(
            "split.motif_disjoint_strata does not name the strata the contract "
            "requires to be motif-disjoint")
    return problems


def _build_shape_problems(build_report: object) -> list[str]:
    if not isinstance(build_report, dict):
        return ["the acceptance report's build section is missing"]
    problems = [f"the build report is missing venue build {name!r}"
                for name in VENUE_BUILDS if name not in build_report]
    problems.extend(f"the build report carries unknown venue build {name!r}"
                    for name in sorted(set(build_report) - set(VENUE_BUILDS)))
    for name in VENUE_BUILDS:
        if name in build_report:
            problems.extend(_shape_problems(build_report[name], _BUILD_FIELDS,
                                            f"build.{name}"))
    return problems


def _fingerprint_shape_problems(prints: object) -> list[str]:
    if not isinstance(prints, dict):
        return ["the acceptance report's fingerprints section is missing"]
    problems = [f"the fingerprints section is missing stratum {name!r}"
                for name in ALL_STRATA if name not in prints]
    problems.extend(f"the fingerprints section carries unknown stratum {name!r}"
                    for name in sorted(set(prints) - set(ALL_STRATA)))
    for name in ALL_STRATA:
        if name in prints:
            problems.extend(_shape_problems(prints[name], _FINGERPRINT_FIELDS,
                                            f"fingerprints.{name}"))
    return problems


#: Exact value types for one census arm, derived from the shared field list.
#:
#: Derived rather than restated so `VM_ARM_FIELDS` stays the one authoritative
#: list of what an arm carries (`docs/directions.md` §7 invariant 17). The names
#: are Direction 3's builder's; the types, and the arithmetic in
#: `_vm_arm_problems`, are Direction 4's rule -- the same split `vm_census_problems`
#: already makes, and the reason a shared rule is not written into
#: `dm/data/feedback_audit.py`: that file is a hashed F5b/F5c qualification source
#: and Direction 4 may not move it.
_VM_ARM_SCHEMA: dict[str, type | tuple[type, ...]] = {
    name: (dict if name in {"faults", "stroke_count_histogram"}
           else (int, float) if name == "mean_strokes"
           else int)
    for name in VM_ARM_FIELDS
}


#: The most strokes one program can emit in the default census.
#:
#: `Trace.strokes` grows only through `flush()`. There can be at most one flush
#: per executed instruction plus finalisation, and a successful final flush
#: requires the last instruction not to have flushed, so successful flushes
#: cannot exceed the VM's default fuel budget. `CURVE_STEPS` bounds points added
#: by `CURVE`; it is not a strokes-per-step bound. This is what makes a
#: histogram key's domain decidable, and its decimal width is what makes the key
#: safe to convert.
MAX_STROKES_PER_PROGRAM = DEFAULT_FUEL
MAX_STROKE_DIGITS = len(str(MAX_STROKES_PER_PROGRAM))


def _short(value: object, limit: int = 24) -> str:
    """A bounded `repr`, so one absurd field cannot produce an unreadable fault."""
    text = repr(value)
    return text if len(text) <= limit else f"{text[:limit]}... ({len(text)} chars)"


def _is_stroke_count(key: str) -> bool:
    """One stroke length, one spelling, narrow enough to convert safely.

    `str.isdigit` is not this test, and the difference is the whole finding.
    It is true for `'\u0667'` and for `'007'`, `int` accepts both, and the census
    writes `str(strokes)` -- so a body could carry two keys for one stroke length
    and the histogram would quietly stop summing to the program count. The width
    bound is what makes the conversion itself safe: `int` on an unbounded decimal
    raises CPython's own `ValueError` before any rule here runs, and a refusal
    that comes from the interpreter rather than from the contract is one the
    Adapter cannot tell from a real fault.
    """
    if not (key.isascii() and key.isdigit()) or len(key) > MAX_STROKE_DIGITS:
        return False
    if key != "0" and key.startswith("0"):
        return False
    return int(key) <= MAX_STROKES_PER_PROGRAM


def _count_map_problems(body: Mapping, where: str, *,
                        digit_keys: bool = False) -> list[str]:
    """A JSON object used as a counter: named keys, non-negative integer values.

    When the keys are themselves numbers they are validated **completely before
    anything converts them**. `int` on a five-thousand-digit string raises before
    this function can say a word, so a domain check that runs after the
    conversion is not a domain check.
    """
    problems: list[str] = []
    for key, value in body.items():
        if not isinstance(key, str):
            problems.append(
                f"{where} has a key that is not a string: {_short(key)}")
        elif digit_keys and not _is_stroke_count(key):
            problems.append(
                f"{where} has a key that is not a stroke count in "
                f"[0, {MAX_STROKES_PER_PROGRAM}]: {_short(key)}")
        elif not digit_keys and not key:
            problems.append(
                f"{where} has a key that is not a name: {_short(key)}")
        if _wrong_type(value, int) or value < 0:
            problems.append(f"{where}[{_short(key)}] is not a count")
    return problems


def _vm_arm_problems(name: str, arm: object) -> list[str]:
    """One census arm's value types and its own internal arithmetic.

    Exact keys were checked and were not enough. An arm is a set of counts that
    *determine each other*: the stroke histogram counted the programs and the
    strokes, the fault map counted the events, and the mean is the quotient of two
    of them. So a body could name every field, put a string where a count belongs,
    or carry a total its own histogram contradicts, and pass. A census is evidence
    about what the VM did to a whole arm; a total no run produced is not that.
    """
    where = f"vm_census.arms.{name}"
    problems = _shape_problems(arm, _VM_ARM_SCHEMA, where)
    if problems or not isinstance(arm, dict):
        return problems
    problems.extend(f"{where}.{field} is negative" for field, kind
                    in _VM_ARM_SCHEMA.items() if kind is int and arm[field] < 0)
    problems.extend(_count_map_problems(arm["faults"], f"{where}.faults"))
    problems.extend(_count_map_problems(arm["stroke_count_histogram"],
                                        f"{where}.stroke_count_histogram",
                                        digit_keys=True))
    if problems:
        return sorted(problems)
    histogram = {int(key): value
                 for key, value in arm["stroke_count_histogram"].items()}
    problems.extend(
        f"{where}.{field} records {arm[field]!r}; its own census gives {want!r}"
        for field, want in (
            ("programs", sum(histogram.values())),
            ("total_strokes", sum(key * value for key, value in histogram.items())),
            ("fault_events", sum(arm["faults"].values())),
            ("faulted_programs", arm["programs"] - arm["valid_programs"]),
            ("programs_with_strokes",
             sum(value for key, value in histogram.items() if key)),
            ("min_strokes", min(histogram, default=0)),
            ("max_strokes", max(histogram, default=0)),
        ) if arm[field] != want)
    if arm["valid_halted_programs"] > min(arm["valid_programs"],
                                          arm["halted_programs"]):
        problems.append(f"{where} counts more valid-and-halted programs than "
                        "valid or halted ones")
    if max(arm["halted_programs"], arm["valid_programs"]) > arm["programs"]:
        problems.append(f"{where} counts more halted or valid programs than "
                        "programs")
    if problems:
        # The mean is a quotient of two of the counts above, so it is only
        # decidable once those counts are. Dividing numbers this function has just
        # declared wrong is how a forged `total_strokes` of `10 ** 400` reached a
        # float division and raised, and the quotient would say nothing a reader
        # of the faults above does not already know.
        return sorted(problems)
    # Domain before conversion, for the same reason the histogram keys have one:
    # `float(10 ** 400)` raises `OverflowError`. A mean lies between zero and the
    # arm's own total, but a forged total can be just as large as the recorded
    # mean. The interpreter's per-program bound closes that coordinated arm
    # before conversion as well. The same comparison disposes of the two floats
    # that are not means, since `inf` exceeds the bound and `nan` is not greater
    # than or equal to anything.
    total, recorded = arm["total_strokes"], arm["mean_strokes"]
    upper = min(total, MAX_STROKES_PER_PROGRAM)
    if not 0 <= recorded <= upper:
        return [(f"{where}.mean_strokes is outside [0, {upper}]: "
                 f"{_short(recorded)}")]
    mean = total / max(1, arm["programs"])
    if not math.isclose(float(recorded), mean, rel_tol=0.0, abs_tol=1e-12):
        problems.append(f"{where}.mean_strokes records {recorded!r}; "
                        f"its own totals give {mean!r}")
    return sorted(problems)


def _census_shape_problems(census: object) -> list[str]:
    if not isinstance(census, dict):
        return ["the VM census is missing"]
    expected = {"schema", "arms", "paired_relational_destroyed",
                "structured_flat"}
    problems = [f"the VM census is missing {name!r}"
                for name in sorted(expected - set(census))]
    problems.extend(f"the VM census carries unexpected field {name!r}"
                    for name in sorted(set(census) - expected))
    arms = census.get("arms")
    if not isinstance(arms, dict) or not arms:
        problems.append("the VM census names no program arms")
    else:
        for name in sorted(arms):
            problems.extend(_vm_arm_problems(name, arms[name]))
    paired = census.get("paired_relational_destroyed")
    if paired is not None and not isinstance(paired, dict):
        problems.append("vm_census.paired_relational_destroyed is malformed")
    comparison = census.get("structured_flat")
    problems.extend(_shape_problems(comparison, _STRUCTURED_FLAT_FIELDS,
                                    "vm_census.structured_flat"))
    if isinstance(comparison, dict):
        problems.extend(
            problem for field in ("vm_mismatches", "render_mismatches")
            if isinstance(comparison.get(field), list)
            for problem in _string_list_problems(
                comparison[field], f"vm_census.structured_flat.{field}"))
    return problems


def _truncation_problems(report: Mapping) -> list[str]:
    """Every truncated evidence list agrees with the count beside it.

    `acceptance_report` slices its mismatch and fault lists to
    `EVIDENCE_LIST_LIMIT`, so a list and its count agree when the list holds
    `min(count, limit)` entries. Without this rule a body can carry a count of
    zero beside a list of failures, or an empty list beside a non-zero count, and
    every clause that reads only one of the two is satisfied.
    """
    problems: list[str] = []
    pairs = (("vm_census.structured_flat", ("vm_mismatches", "vm_mismatch_count")),
             ("vm_census.structured_flat", ("render_mismatches",
                                            "render_mismatch_count")),
             ("split", ("island_faults", "island_fault_count")))
    census = report.get("vm_census")
    bodies = {
        "vm_census.structured_flat": (census or {}).get("structured_flat")
        if isinstance(census, dict) else None,
        "split": report.get("split"),
    }
    for where, (list_name, count_name) in pairs:
        body = bodies[where]
        if not isinstance(body, dict):
            continue
        found, count = body.get(list_name), body.get(count_name)
        if not isinstance(found, list) or _wrong_type(count, int):
            continue
        if len(found) != min(count, EVIDENCE_LIST_LIMIT):
            problems.append(
                f"{where}.{list_name} holds {len(found)} entries beside a count of "
                f"{count}; a truncated list carries min(count, "
                f"{EVIDENCE_LIST_LIMIT})")
    return problems


def _index_agreement_problems(cases: Sequence[Mapping],
                              report: Mapping) -> list[str]:
    """Every acceptance fact the case index can produce, recomputed and compared.

    This is the clause that makes the manifest self-checking. The acceptance
    report is a set of assertions about a corpus; the case index is a description
    of the same corpus; and until now nothing required them to agree. Anything
    derivable from the index is derived here and compared, so a report edited to
    pass `accepts` has to be accompanied by an index edited to match -- at which
    point `rebuild_verify` is the remaining defence, and the payload digest is
    the one after that.

    **The index's own claims are not exempt.** `component` is a label over the
    motif co-occurrence graph and `groups` is that graph, so the labels are
    rebuilt from the graph before any number derived from them -- the strata
    counts the 128-component floor reads, and the destroyed arm's donor
    namespace -- is believed.

    What is *not* recomputable is stated rather than skipped silently: coverage
    of the candidate caps, residual and introduced relations, the island map,
    donor block skeletons, the leak count and every fingerprint digest need the
    program bytes, which the index deliberately does not carry. The islands are
    the near miss worth naming: a case's motifs all come from one island, so the
    *graph* is reconstructible from `groups`, but which island a component sits
    in is a property of the pool partition and only `--rebuild-verify` sees it.
    """
    problems: list[str] = []

    def expect(path: str, found: object, want: object) -> None:
        if found != want:
            problems.append(
                f"{path} records {found!r}; the case index gives {want!r}")

    by_stratum: dict[str, list[Mapping]] = {name: [] for name in ALL_STRATA}
    for record in cases:
        by_stratum[record["stratum"]].append(record)
    index = {record["case_id"]: record for record in cases}
    # The graph, rebuilt, before any number derived from it is believed. Every
    # clause below that reads a component -- the strata counts the 128-component
    # floor is applied to, and the destroyed arm's donor namespace -- reads this
    # reconstruction rather than the stored label.
    canonical = _canonical_components(cases)
    mislabelled = sorted(record["case_id"] for record in cases
                         if record["component"] != canonical[record["case_id"]])
    if mislabelled:
        problems.append(
            f"{len(mislabelled)} cases carry a component label the motif "
            f"co-occurrence graph does not give: "
            f"{mislabelled[:EVIDENCE_LIST_LIMIT]}")
    train = by_stratum[STRATUM_TRAIN]
    train_motifs = {motif for record in train for motif in record["groups"]}
    train_sources = {source for record in train
                     for source in record["source_ids"]}
    train_tuples = {tuple(item) for record in train for item in record["plan"]}

    strata = report["strata"]
    expect("the strata table", sum(item["cases"] for item in strata.values()),
           len(cases))
    for stratum, records in by_stratum.items():
        found = strata[stratum]
        motifs = {motif for record in records for motif in record["groups"]}
        sources = {source for record in records
                   for source in record["source_ids"]}
        expect(f"strata.{stratum}.cases", found["cases"], len(records))
        expect(f"strata.{stratum}.components", found["components"],
               len({canonical[record["case_id"]] for record in records}))
        expect(f"strata.{stratum}.positives", found["positives"],
               sum(1 for record in records if record["actions"]))
        expect(f"strata.{stratum}.novel_pairs", found["novel_pairs"],
               sum(1 for record in records if record["novel_pair"]))
        expect(f"strata.{stratum}.target_bytes", found["target_bytes"],
               sum(record["target"][1] - record["target"][0]
                   for record in records))
        expect(f"strata.{stratum}.venues", found["venues"],
               {venue: sum(1 for record in records if record["venue"] == venue)
                for venue in VENUES})
        expect(f"strata.{stratum}.motifs_shared_with_training",
               found["motifs_shared_with_training"], len(motifs & train_motifs))
        expect(f"strata.{stratum}.source_identities_shared_with_training",
               found["source_identities_shared_with_training"],
               len(sources & train_sources))

    expect("coverage.positives", report["coverage"]["positives"],
           sum(1 for record in cases if record["actions"]))
    expect("generic.cases", report["generic"]["cases"],
           len(by_stratum[STRATUM_GENERIC]))
    problems.extend(_split_agreement_problems(by_stratum, report["split"],
                                              train_motifs, train_sources,
                                              train_tuples))
    problems.extend(_destroyed_agreement_problems(by_stratum, index, canonical,
                                                  report["destroyed"]))
    problems.extend(_census_agreement_problems(by_stratum, cases,
                                               report["vm_census"]))
    for stratum, records in by_stratum.items():
        found = report["fingerprints"][stratum]
        expect(f"fingerprints.{stratum}.n_train", found["n_train"], len(records))
        expect(f"fingerprints.{stratum}.bytes_train", found["bytes_train"],
               sum(record["bytes"] for record in records))
        expect(f"fingerprints.{stratum}.n_val", found["n_val"], 0)
        expect(f"fingerprints.{stratum}.bytes_val", found["bytes_val"], 0)
    return problems


def _split_agreement_problems(by_stratum: Mapping[str, Sequence[Mapping]],
                              split: Mapping, train_motifs: set[str],
                              train_sources: set[str],
                              train_tuples: set[tuple]) -> list[str]:
    """The whole split -- tuples, atoms, pair edges and labels -- from the index."""
    problems: list[str] = []

    def expect(path: str, found: object, want: object) -> None:
        if found != want:
            problems.append(
                f"{path} records {found!r}; the case index gives {want!r}")

    cases = [record for records in by_stratum.values() for record in records]
    expect("split.duplicate_case_ids", split["duplicate_case_ids"],
           len(cases) - len({record["case_id"] for record in cases}))
    expect("split.motif_identities_shared", split["motif_identities_shared"],
           {stratum: len({motif for record in records
                          for motif in record["groups"]} & train_motifs)
            for stratum, records in by_stratum.items()})
    expect("split.motif_disjointness_violations",
           split["motif_disjointness_violations"],
           sorted(stratum for stratum in MOTIF_DISJOINT_STRATA
                  if {motif for record in by_stratum[stratum]
                      for motif in record["groups"]} & train_motifs))
    expect("split.source_identity_disjointness_violations",
           split["source_identity_disjointness_violations"],
           sorted(stratum for stratum in MOTIF_DISJOINT_STRATA
                  if {source for record in by_stratum[stratum]
                      for source in record["source_ids"]} & train_sources))
    expect("split.untrained_motifs_in_nested_primary",
           split["untrained_motifs_in_nested_primary"],
           sorted({record["groups"][0]
                   for record in by_stratum[STRATUM_NESTED]}
                  - train_motifs)[:EVIDENCE_LIST_LIMIT])

    held = {venue: {tuple(item) for item in items}
            for venue, items in split["held_out_tuples"].items()}
    expect("split.held_out_tuples_in_training",
           split["held_out_tuples_in_training"],
           sorted(f"{venue}:{item}" for venue, items in held.items()
                  for item in items if item in train_tuples))
    supported = _atom_support(train_tuples)
    expect("split.orphaned_atoms", split["orphaned_atoms"],
           sorted(f"{name}={value}" for items in held.values()
                  for name, value in _atom_support(items) - supported))

    edges = {frozenset(edge) for edge in split["held_out_pair_edges"]}
    expect("split.held_out_pairs", split["held_out_pairs"], len(edges))
    training_edges = {edge for record in by_stratum[STRATUM_TRAIN]
                      for edge in _record_pairs(record)}
    expect("split.held_out_pairs_in_training", split["held_out_pairs_in_training"],
           _edge_list(edges & training_edges))
    expect("split.held_out_pair_endpoint_orphans",
           split["held_out_pair_endpoint_orphans"],
           sorted(f"{node} not in training" for edge in edges for node in edge
                  if node not in train_motifs))
    pair_records = (*by_stratum[STRATUM_PAIR], *by_stratum[STRATUM_INTERACTION])
    represented = {edge for record in pair_records
                   for edge in _record_pairs(record) if edge in edges}
    expect("split.held_out_pairs_missing_from_secondary",
           split["held_out_pairs_missing_from_secondary"],
           _edge_list(edges - represented))
    expect("split.secondary_cases_without_held_out_pair",
           split["secondary_cases_without_held_out_pair"],
           sorted(record["case_id"] for record in pair_records
                  if not _record_pairs(record) & edges))
    expect("split.pure_affine_cases_with_held_out_pair",
           split["pure_affine_cases_with_held_out_pair"],
           sorted(record["case_id"] for record in by_stratum[STRATUM_AFFINE]
                  if _record_pairs(record) & edges))
    composed_held = held.get(VENUE_COMPOSED, set())
    expect("split.interaction_cases_without_held_out_tuple",
           split["interaction_cases_without_held_out_tuple"],
           sorted(record["case_id"]
                  for record in by_stratum[STRATUM_INTERACTION]
                  if not {tuple(item) for item in record["plan"]} & composed_held))
    expect("split.pair_cases_with_held_out_tuple",
           split["pair_cases_with_held_out_tuple"],
           sorted(record["case_id"] for record in by_stratum[STRATUM_PAIR]
                  if {tuple(item) for item in record["plan"]} & composed_held))
    expect("split.pair_cases", split["pair_cases"], len(pair_records))
    return problems


def _destroyed_agreement_problems(by_stratum: Mapping[str, Sequence[Mapping]],
                                  index: Mapping[str, Mapping],
                                  canonical: Mapping[str, int],
                                  destroyed: Mapping) -> list[str]:
    """Donor references, metadata, identity exclusion and closure, from the index.

    The reported fault list is required to *contain* every fault the index can
    decide rather than to equal it, because one clause of the donor namespace --
    the donor block's length and opcode skeleton -- needs program bytes the index
    does not carry. Containment is the sound direction: a report may know more
    than the index, and may never know less.
    """
    problems: list[str] = []
    rows = by_stratum[STRATUM_DESTROYED]
    reported = destroyed["donor_provenance_faults"]
    if destroyed["accepted"] != len(rows):
        problems.append(
            f"destroyed.accepted records {destroyed['accepted']!r}; the case index "
            f"holds {len(rows)} destroyed rows")
    sources = {record["source_case_id"] for record in rows}
    if destroyed["sources"] != len(sources):
        problems.append(
            f"destroyed.sources records {destroyed['sources']!r}; the case index "
            f"gives {len(sources)}")
    donors = [record["donor_case_id"] for record in rows]
    if destroyed["duplicate_donors"] != len(donors) - len(set(donors)):
        problems.append(
            f"destroyed.duplicate_donors records "
            f"{destroyed['duplicate_donors']!r}; the case index gives "
            f"{len(donors) - len(set(donors))}")

    faults: list[str] = []
    for record in rows:
        case_id = record["case_id"]
        source = index.get(record["source_case_id"])
        donor = index.get(record["donor_case_id"])
        if source is None:
            faults.append(f"{case_id}: missing recipient "
                          f"{record['source_case_id']}")
            continue
        if donor is None:
            faults.append(f"{case_id}: missing donor {record['donor_case_id']}")
            continue
        if record["donor_case_id"] == record["source_case_id"]:
            faults.append(f"{case_id}: donor is recipient")
        if record["donor_groups"] != donor["groups"]:
            faults.append(f"{case_id}: donor groups do not match donor case")
        if record["donor_source_ids"] != donor["source_ids"]:
            faults.append(
                f"{case_id}: donor source identities do not match donor case")
        if (donor["venue"], donor["stratum"], canonical[donor["case_id"]]) != \
                (source["venue"], source["stratum"],
                 canonical[source["case_id"]]):
            faults.append(f"{case_id}: donor crosses destruction namespace")
    if set(donors) != sources:
        faults.append("destroyed donor map is not a closed permutation")
    missing = sorted(set(faults) - set(reported))
    if missing:
        problems.append(
            f"destroyed.donor_provenance_faults omits {len(missing)} faults the "
            f"case index shows: {missing[:EVIDENCE_LIST_LIMIT]}")
    problems.extend(_namespace_agreement_problems(
        rows, index, canonical, destroyed["donor_namespaces"]))
    return problems


def _namespace_agreement_problems(rows: Sequence[Mapping],
                                  index: Mapping[str, Mapping],
                                  canonical: Mapping[str, int],
                                  namespaces: Mapping) -> list[str]:
    """Every destroyed row has a persisted donor namespace, and it is its own.

    The namespace is `(venue, stratum, component, target length, skeleton)`. The
    first four come from the index -- the component from the rebuilt graph rather
    than from the stored label, so a relabelled index cannot move a namespace with
    it -- and the skeleton needs bytes. A missing namespace is the interesting
    failure: the destroyed arm's whole dependence intervention is that donors are
    drawn inside one, so a row with no recorded namespace is a row whose
    intervention cannot be checked at all.
    """
    problems: list[str] = []
    wanted = {record["case_id"] for record in rows}
    missing = sorted(wanted - set(namespaces))
    if missing:
        problems.append(
            f"destroyed.donor_namespaces omits {len(missing)} destroyed rows: "
            f"{missing[:EVIDENCE_LIST_LIMIT]}")
    extra = sorted(set(namespaces) - wanted)
    if extra:
        problems.append(
            f"destroyed.donor_namespaces names {len(extra)} cases that are not "
            f"destroyed rows: {extra[:EVIDENCE_LIST_LIMIT]}")
    for record in rows:
        namespace = namespaces.get(record["case_id"])
        source = index.get(record["source_case_id"])
        if namespace is None or source is None:
            continue
        if not isinstance(namespace, list) or len(namespace) != 5:
            problems.append(
                f"destroyed.donor_namespaces[{record['case_id']!r}] is not a "
                "five-part namespace")
            continue
        want = [source["venue"], source["stratum"],
                canonical[source["case_id"]],
                source["target"][1] - source["target"][0]]
        if namespace[:4] != want:
            problems.append(
                f"destroyed.donor_namespaces[{record['case_id']!r}] records "
                f"{namespace[:4]!r}; the case index gives {want!r}")
    return problems


def _census_agreement_problems(by_stratum: Mapping[str, Sequence[Mapping]],
                               cases: Sequence[Mapping],
                               census: Mapping) -> list[str]:
    """The census names one arm per non-empty stratum, flat and structured.

    Both arms, and both counted. `vm_census_problems` already refuses an arm that
    faulted; what it cannot see is an arm that is simply *absent*, or one whose
    program count is smaller than the stratum it claims to cover -- a census over
    the clean half of an arm is a clean census.
    """
    problems: list[str] = []
    arms = census["arms"]
    expected = {}
    for stratum, records in by_stratum.items():
        if records:
            expected[stratum] = len(records)
            expected[f"{stratum}.structured"] = len(records)
    if set(arms) != set(expected):
        problems.append(
            f"the VM census covers arms {sorted(arms)}; the case index requires "
            f"{sorted(expected)}")
    for name, count in sorted(expected.items()):
        arm = arms.get(name)
        if isinstance(arm, dict) and arm.get("programs") != count:
            problems.append(
                f"vm_census.arms.{name}.programs records {arm.get('programs')!r}; "
                f"the case index holds {count} cases")
    comparison = census["structured_flat"]
    if comparison.get("cases") != len(cases):
        problems.append(
            f"vm_census.structured_flat.cases records {comparison.get('cases')!r}; "
            f"the case index holds {len(cases)} cases")
    known = {record["case_id"] for record in cases}
    for name in ("vm_mismatches", "render_mismatches"):
        unknown = sorted(set(comparison.get(name) or []) - known)
        if unknown:
            problems.append(
                f"vm_census.structured_flat.{name} names cases that are not in the "
                f"index: {unknown[:EVIDENCE_LIST_LIMIT]}")
    return problems


def validate_manifest(body: object) -> list[str]:
    """Every structural and cross-field fault in a manifest body, or an empty list.

    Structure and cross-field agreement only. Whether the corpus it describes is
    good enough to freeze stays with `accepts`, so a small development build has a
    perfectly *valid* manifest that is still refused for training -- two questions
    that would be impossible to debug if one function answered both.

    The recomputation pass runs only when the shapes it reads are sound, because a
    cross-check against a malformed section reports the malformation twice under a
    less useful name.
    """
    if not isinstance(body, dict):
        return ["the manifest is missing or malformed"]
    problems = _envelope_problems(body)
    cases, report = body.get("cases"), body.get("acceptance")
    shape: list[str] = []
    if isinstance(cases, list):
        shape.extend(_case_index_problems(cases))
    if isinstance(report, dict):
        shape.extend(_acceptance_shape_problems(report))
    problems.extend(shape)
    if not shape and isinstance(cases, list) and isinstance(report, dict):
        problems.extend(_index_agreement_problems(cases, report))
    return problems


def manifest_config(body: Mapping) -> BuildConfig:
    """The `BuildConfig` a manifest says it was built from."""
    config = dict(body["acceptance"]["config"])
    config["composed_categories"] = tuple(config["composed_categories"])
    return BuildConfig(**config)


def rebuild_verify(body: Mapping) -> list[str]:
    """Rebuild the corpus from the manifest's own config and require equality.

    The exhaustive route, and the only one that covers what the case index cannot
    reconstruct: candidate coverage, residual and introduced relations, the island
    map, the fingerprints and every VM figure all need the program bytes, which
    the index deliberately does not carry. Regenerating them is the check.

    A build is a pure function of its `BuildConfig` -- pinned separately -- so
    canonical payload equality is exact rather than approximate, and any
    difference is either a corpus that was not built by this code or a manifest
    that was edited after it was.
    """
    rebuilt = manifest(build(manifest_config(body)))
    if rebuilt["corpus_sha256"] == body.get("corpus_sha256"):
        return []
    # Only on failure, and through JSON on both sides: an in-memory body holds
    # tuples where a loaded one holds lists, which the digest is blind to and a
    # field-by-field comparison is not.
    normalised = json.loads(json.dumps(rebuilt))
    moved = sorted(name for name in set(normalised) | set(body)
                   if name != "corpus_sha256"
                   and normalised.get(name) != body.get(name))
    problems = [(
        f"the rebuilt corpus has payload digest {rebuilt['corpus_sha256']}, but "
        f"the manifest records {body.get('corpus_sha256')!r}; sections that "
        f"differ: {moved}")]
    if "cases" in moved:
        problems.append(_first_case_difference(normalised["cases"],
                                               body.get("cases") or []))
    return problems


def _first_case_difference(rebuilt: Sequence, recorded: Sequence) -> str:
    """Where two case indexes first disagree, named rather than left to a diff."""
    if len(rebuilt) != len(recorded):
        return (f"the rebuilt case index holds {len(rebuilt)} cases and the "
                f"manifest {len(recorded)}")
    for position, (left, right) in enumerate(zip(rebuilt, recorded, strict=True)):
        if left != right:
            return (f"the case indexes first differ at position {position}: "
                    f"rebuilt {left.get('case_id')!r} against recorded "
                    f"{right.get('case_id')!r}")
    return "the case indexes are equal but the payload digest is not"


@dataclass(frozen=True)
class ManifestSnapshot:
    """Immutable raw bytes and their hashes; ``body`` is a consumer-owned mutable decode.

    This snapshot is not itself acceptance evidence. Consumers must validate the
    decoded body before use and must not mutate it during validation/rebuild.
    """

    path: str
    raw_bytes: bytes
    body: dict
    canonical_payload_sha256: str
    recorded_canonical_sha256: object
    file_sha256: str


def read_manifest_snapshot(path: Path) -> ManifestSnapshot:
    """Read and hash a relation-manifest object without rereading its pathname.

    This is deliberately a byte-snapshot boundary, not the acceptance boundary:
    callers such as audit reporting may need the identities of a forged body in
    order to report why its subsequent validation failed. Use
    :func:`load_manifest` (or ``_validated_manifest_body`` within a single
    snapshot consumer) before trusting the body.
    """
    try:
        raw = Path(path).read_bytes()
    except OSError as error:
        raise ManifestRefused(f"relation corpus manifest {path} is not readable: {error}") from error
    try:
        body = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ManifestRefused(
            f"relation corpus manifest {path} is not readable JSON: {error}") from error
    if not isinstance(body, dict):
        raise ManifestRefused(f"relation corpus manifest {path} root must be an object")
    payload = {key: value for key, value in body.items() if key != "corpus_sha256"}
    return ManifestSnapshot(
        path=str(path),
        raw_bytes=raw,
        body=body,
        canonical_payload_sha256=hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        recorded_canonical_sha256=body.get("corpus_sha256"),
        file_sha256=hashlib.sha256(raw).hexdigest(),
    )


def _validated_manifest_body(snapshot: ManifestSnapshot) -> dict:
    """Validate one already-read manifest body without a second path read.

    Four defences, in the order a forgery has to defeat them. The recorded
    payload digest catches a hand edit. `validate_manifest` catches a rehashed
    body that is structurally wrong or whose acceptance report disagrees with its
    own case index -- the gap that let a self-consistent report be believed on its
    own authority. Re-running `accepts` catches a verdict that does not follow
    from the report. And the verdict itself refuses a corpus that did not pass its
    clauses, because that is not a corpus a scientific cell may train on.

    What none of these can see is the corpus that was never built; that is
    `rebuild_verify`, which is exhaustive and therefore explicit.

    **Every refusal here is `ManifestRefused`, and only refusals are.** A reader
    that catches it is saying "this manifest is not usable"; anything else
    escaping this function is a defect in the reading, and letting it through is
    what keeps the two legible apart.
    """
    path, body = snapshot.path, snapshot.body
    if snapshot.recorded_canonical_sha256 != snapshot.canonical_payload_sha256:
        raise ManifestRefused(
            f"relation corpus manifest hash mismatch for {path}: recorded "
            f"{snapshot.recorded_canonical_sha256!r}, computed "
            f"{snapshot.canonical_payload_sha256!r}")
    invalid = validate_manifest(body)
    if invalid:
        raise ManifestRefused(
            f"relation corpus manifest {path} is invalid: "
            + "; ".join(invalid[:EVIDENCE_LIST_LIMIT])
            + (f" (and {len(invalid) - EVIDENCE_LIST_LIMIT} more)"
               if len(invalid) > EVIDENCE_LIST_LIMIT else ""))
    recomputed, problems = accepts(body.get("acceptance"))
    if body["accepted"] is not recomputed:
        raise ManifestRefused(
            f"relation corpus manifest acceptance verdict disagrees with its report: "
            f"recorded {body['accepted']!r}, recomputed {recomputed!r}")
    if body["problems"] != problems:
        raise ManifestRefused(
            "relation corpus manifest acceptance problems disagree with its report")
    if not recomputed:
        raise ManifestRefused(
            f"relation corpus manifest did not pass acceptance: "
            f"{problems}")
    return body


def load_manifest(path: Path) -> dict:
    """Compatibility accessor for the validated immutable manifest body."""
    return _validated_manifest_body(read_manifest_snapshot(path))


#: A depth-2 stratum built from motifs training never saw.
#:
#: Separate from `heldout_nested_composition` because they are two questions and
#: an earlier revision asked them at once. Building the nested co-primary from a
#: disjoint motif pool confounded compositional generalisation with motif
#: transfer: a failure could not say which had caused it. The co-primary now uses
#: *trained* motifs and withholds only the ordered composition; this stratum
#: keeps the harder transfer version, as a secondary.
STRATUM_NESTED_TRANSFER = "nested_motif_transfer"


#: Strata whose motif identities must be disjoint from training's.
#:
#: **Not every stratum, and the exception is the point.** `heldout_affine_tuple`
#: asks whether an unseen *combination* of trained atoms can be executed, and its
#: cleanest form holds the motif fixed and varies only the tuple -- so
#: `composed_motif_relation` builds it by partitioning one motif pool, and a
#: motif that appears there under a withheld orbit also appears in training under
#: a trained one. Seeing the motif cannot tell a model what the withheld
#: transform does to it. `heldout_nested_composition` holds motifs fixed for the
#: same reason -- its novelty is the ordered composition. The two `*_transfer`
#: strata are the ones that ask about unseen motifs, and only they require
#: disjointness, which is checked.
MOTIF_DISJOINT_STRATA: tuple[str, ...] = (STRATUM_TRANSFER,
                                          STRATUM_NESTED_TRANSFER)

#: Every stratum a build can produce, in report order.
ALL_STRATA: tuple[str, ...] = (*STRATA, STRATUM_TRANSFER, STRATUM_NESTED_TRANSFER)
