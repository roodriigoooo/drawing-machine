"""Reference VM.

Executes ISA bytecode into flat geometry. Faults are *reported*, never raised:
a generated program is expected to be malformed sometimes, and validity rate is
one of the headline metrics. Anything a real device would trap on shows up here
as a Fault record.

This implementation is the specification the eventual C/RTL port must match.
It stays free of rendering dependencies so it can be diffed against the port on
geometry alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..isa.spec import (
    CANVAS,
    MAX_REPEAT_DEPTH,
    Op,
    UnknownOpcode,
    decode_operand,
    spec_for,
)
from ..isa.transform import IDENTITY, Transform, compose_all

DEFAULT_FUEL = 100_000
CURVE_STEPS = 16


class FaultKind(str, Enum):
    UNKNOWN_OPCODE = "unknown_opcode"
    TRUNCATED = "truncated"
    UNMATCHED_ENDREP = "unmatched_endrep"
    UNTERMINATED_REPEAT = "unterminated_repeat"
    DEPTH_OVERFLOW = "depth_overflow"
    ZERO_REPEAT = "zero_repeat"
    OUT_OF_FUEL = "out_of_fuel"
    NO_HALT = "no_halt"
    CALL_UNSUPPORTED = "call_unsupported"


@dataclass(frozen=True)
class Fault:
    kind: FaultKind
    pc: int
    detail: str = ""


@dataclass(frozen=True)
class Stroke:
    """Open polyline. Curves arrive here already flattened."""

    points: tuple[tuple[float, float], ...]
    width: int


@dataclass(frozen=True)
class Disc:
    cx: float
    cy: float
    r: int
    width: int


@dataclass(frozen=True)
class Region:
    """Closed, filled polygon."""

    points: tuple[tuple[float, float], ...]


@dataclass
class Trace:
    strokes: list[Stroke] = field(default_factory=list)
    discs: list[Disc] = field(default_factory=list)
    regions: list[Region] = field(default_factory=list)
    faults: list[Fault] = field(default_factory=list)
    steps: int = 0
    halted: bool = False

    @property
    def valid(self) -> bool:
        return not self.faults

    @property
    def is_empty(self) -> bool:
        return not (self.strokes or self.discs or self.regions)

    def __len__(self) -> int:
        return len(self.strokes) + len(self.discs) + len(self.regions)


@dataclass
class _Frame:
    body: int      # pc of the first instruction inside the loop
    count: int
    iteration: int
    dx: int
    dy: int
    base: tuple[int, int]
    #: Where this frame's transform sits in the transform stack, and the
    #: transform one iteration applies. `None` for plain `REPEAT`, which is a
    #: translation and needs no slot -- so an L0/L1 program pays nothing for the
    #: transform tier existing, which is what keeps every pre-v2 trace
    #: bit-identical.
    xform_slot: int | None = None
    step: Transform = IDENTITY
    #: Transform-stack floor in force before this loop opened, restored when it
    #: closes. A `REPEATX` owns its slot, so an `ENDX` in the body must not be
    #: able to close it -- see `run`.
    xform_floor: int = 0


def flatten_cubic(p0, p1, p2, p3, steps: int) -> list[tuple[float, float]]:
    """The polyline a `CURVE` actually draws: `t = i/steps` for `i = 1..steps`.

    Public because it is the *specification* of the geometry, and anything that
    fits a curve has to be scored against what the device will draw rather than
    against the ideal cubic. `dm/isa/bezier.py` measures its own error through
    this function for exactly that reason -- a second flattener would be a
    second specification, and `port/src/dm_vm.c` matches this one bit for bit
    (`scripts/conformance.py`).
    """
    pts = []
    for i in range(1, steps + 1):
        t = i / steps
        u = 1.0 - t
        a, b, c, d = u * u * u, 3 * u * u * t, 3 * u * t * t, t * t * t
        pts.append(
            (
                a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0],
                a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1],
            )
        )
    return pts


class VM:
    def __init__(self, fuel: int = DEFAULT_FUEL, curve_steps: int = CURVE_STEPS) -> None:
        self.fuel = fuel
        self.curve_steps = curve_steps

    def run(self, program: bytes) -> Trace:
        tr = Trace()
        pc = 0
        cur = (0.0, 0.0)
        width = 1
        offset = (0, 0)
        stack: list[_Frame] = []
        path: list[tuple[float, float]] = []
        # The transform tier (ISA v2). `xforms` is the scope stack and `active`
        # is the whole of it composed into one `(D4, translation)` pair, so
        # `place` costs one table lookup and two adds regardless of depth -- the
        # property that makes the device cost a constant rather than a
        # per-point computation. Recomposed only when the stack changes.
        xforms: list[Transform] = []
        active = IDENTITY
        # Entries below this index belong to an open `REPEATX` and cannot be
        # closed by an `ENDX`. Without it a body that closes the loop's own
        # scope leaves the frame pointing past the end of the stack -- which
        # `dm/isa/unroll.py` already refuses as a crossing, and which the
        # conformance fuzzer found here as a *crash* rather than a fault.
        xform_floor = 0

        def place(x: int, y: int) -> tuple[float, float]:
            # `REPEAT`'s offset is a translation in the *untransformed* frame and
            # the transform maps the result. Written down here because a nested
            # program means two different drawings under the other order, and
            # the choice has to predate the first run rather than be recovered
            # from it later (PLAN.md, direction item 2, decision 4).
            x, y = active.point(x + offset[0], y + offset[1])
            return (
                float(min(CANVAS - 1, max(0, x))),
                float(min(CANVAS - 1, max(0, y))),
            )

        def flush() -> None:
            nonlocal path
            if len(path) >= 2:
                tr.strokes.append(Stroke(tuple(path), width))
            path = []

        while pc < len(program):
            if tr.steps >= self.fuel:
                tr.faults.append(Fault(FaultKind.OUT_OF_FUEL, pc))
                break
            tr.steps += 1

            try:
                spec = spec_for(program[pc])
            except UnknownOpcode as exc:
                tr.faults.append(Fault(FaultKind.UNKNOWN_OPCODE, pc, str(exc)))
                break

            end = pc + spec.size
            if end > len(program):
                tr.faults.append(Fault(FaultKind.TRUNCATED, pc, spec.mnemonic))
                break
            args = [
                decode_operand(kind, byte)
                for kind, byte in zip(spec.operands, program[pc + 1 : end])
            ]
            nxt = end

            if spec.op is Op.HALT:
                flush()
                if stack or xforms:
                    # An unclosed `XFORM` is an unclosed scope, which is what
                    # this fault already means. Halting inside one would leave
                    # the drawing's last strokes transformed by something the
                    # program never closed.
                    tr.faults.append(Fault(FaultKind.UNTERMINATED_REPEAT, pc))
                tr.halted = True
                pc = nxt
                break

            elif spec.op is Op.MOVE:
                flush()
                cur = place(args[0], args[1])
                path = [cur]

            elif spec.op is Op.LINE:
                if not path:
                    path = [cur]
                cur = place(args[0], args[1])
                path.append(cur)

            elif spec.op is Op.CURVE:
                if not path:
                    path = [cur]
                c1 = place(args[0], args[1])
                c2 = place(args[2], args[3])
                end_pt = place(args[4], args[5])
                path.extend(flatten_cubic(cur, c1, c2, end_pt, self.curve_steps))
                cur = end_pt

            elif spec.op is Op.CIRCLE:
                tr.discs.append(Disc(cur[0], cur[1], args[0], width))

            elif spec.op is Op.WIDTH:
                flush()
                width = max(1, args[0])

            elif spec.op is Op.FILL:
                if len(path) >= 3:
                    tr.regions.append(Region(tuple(path)))
                path = []

            elif spec.op in (Op.REPEAT, Op.REPEATX):
                # One path for both, because `REPEATX` *is* `REPEAT` with a
                # transform: two loop implementations would be two places for
                # the iteration accounting to drift, and the iteration
                # accounting is what claim 2 measures.
                if spec.op is Op.REPEAT:
                    count, dx, dy = args
                    step = IDENTITY
                else:
                    # `REPEATX` puts its whole step -- rotation, mirror *and*
                    # shift -- into one transform, and takes nothing through
                    # `REPEAT`'s offset. Splitting them was the first design and
                    # it was wrong in a way only a commutation test finds: with
                    # the shift in the offset and the D4 in the scope, iteration
                    # k is `d4^k(p + k·t)`, which is **not** what translating the
                    # whole program produces, so an augmented copy of the program
                    # drew a different picture. Whole-transform powers conjugate
                    # exactly -- `(A S A^-1)^k = A S^k A^-1` -- so the tier
                    # commutes with the group it is built on.
                    #
                    # `REPEAT` is the special case where the D4 part is the
                    # identity: `(I, t)^k` is a translation by `k·t`, which is
                    # what its own offset already computed.
                    count, code, dx, dy = args
                    step = Transform.of(code & 7, dx, dy)
                    dx = dy = 0
                if count == 0:
                    tr.faults.append(Fault(FaultKind.ZERO_REPEAT, pc))
                    count = 1
                if len(stack) >= MAX_REPEAT_DEPTH:
                    tr.faults.append(Fault(FaultKind.DEPTH_OVERFLOW, pc))
                    break
                slot = None
                if spec.op is Op.REPEATX:
                    # The transform stack has its own bound, and it shares
                    # `DEPTH_OVERFLOW` rather than adding a fault kind: the two
                    # are the same structural failure, and every fault name is
                    # mirrored in `port/include/dm_isa.h`, which does not carry
                    # this tier yet.
                    if len(xforms) >= MAX_REPEAT_DEPTH:
                        tr.faults.append(Fault(FaultKind.DEPTH_OVERFLOW, pc))
                        break
                    slot = len(xforms)
                    xforms.append(IDENTITY)     # iteration 0 is untransformed
                    active = compose_all(xforms)
                stack.append(_Frame(nxt, count, 0, dx, dy, offset, slot, step,
                                    xform_floor))
                if slot is not None:
                    xform_floor = slot + 1

            elif spec.op is Op.ENDREP:
                if not stack:
                    tr.faults.append(Fault(FaultKind.UNMATCHED_ENDREP, pc))
                    break
                frame = stack[-1]
                frame.iteration += 1
                if frame.iteration < frame.count:
                    flush()
                    offset = (
                        frame.base[0] + frame.dx * frame.iteration,
                        frame.base[1] + frame.dy * frame.iteration,
                    )
                    if frame.xform_slot is not None:
                        # Iteration k runs under the step applied k times, by
                        # repeated composition rather than a closed form -- the
                        # translation part is a geometric series and only the
                        # accumulated version is exactly what the C port will
                        # compute.
                        xforms[frame.xform_slot] = frame.step.power(frame.iteration)
                        active = compose_all(xforms)
                    nxt = frame.body
                else:
                    offset = frame.base
                    if frame.xform_slot is not None:
                        del xforms[frame.xform_slot:]
                        active = compose_all(xforms)
                    xform_floor = frame.xform_floor
                    stack.pop()

            elif spec.op is Op.XFORM:
                code, dx, dy = args
                if len(xforms) >= MAX_REPEAT_DEPTH:
                    tr.faults.append(Fault(FaultKind.DEPTH_OVERFLOW, pc))
                    break
                flush()
                xforms.append(Transform.of(code & 7, dx, dy))
                active = compose_all(xforms)

            elif spec.op is Op.ENDX:
                # `<=` rather than `== 0`: an `ENDX` may not close a scope it
                # did not open, and a `REPEATX`'s slot belongs to the loop. Both
                # cases are the same structural error and share the fault.
                if len(xforms) <= xform_floor:
                    # A closer with nothing open, which is `ENDREP`'s failure
                    # exactly. Shared for the same reason `DEPTH_OVERFLOW` is.
                    tr.faults.append(Fault(FaultKind.UNMATCHED_ENDREP, pc))
                    break
                flush()
                xforms.pop()
                active = compose_all(xforms)

            elif spec.op is Op.CALL:
                tr.faults.append(Fault(FaultKind.CALL_UNSUPPORTED, pc))
                break

            pc = nxt

        flush()
        if not tr.halted and not tr.faults:
            tr.faults.append(Fault(FaultKind.NO_HALT, len(program)))
        return tr
