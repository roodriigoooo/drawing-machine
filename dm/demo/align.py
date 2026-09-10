"""Which instruction produced the point that just arrived — and a check that it did.

The page highlights a line of disassembly as the matching geometry appears. The
device does not say this: the UART trace carries points and path terminators,
not program counters, and adding a `pc` marker to the firmware would change the
image whose conformance and cycle numbers are already frozen.

So the alignment is computed on the host, and it is **presentation only**. The
geometry drawn on the canvas is the device's, byte for byte; this file decides
only which text line lights up beside it. The page says so in as many words,
because "the host supplied part of this view" is exactly the kind of thing that
is honest when written down and misleading when not.

**It is also checked rather than trusted.** `align` walks the program with the
reference VM's own emission rules and predicts how many points each stroke will
contain. `verify` compares that prediction against what the device actually
sent. A single disagreement discards the alignment and the page falls back to
an unhighlighted listing: a highlight that drifts one instruction is worse than
no highlight, because it is a claim about the ISA that the viewer cannot check.

Programs using `CALL`, `XFORM`, `ENDX` or `REPEATX` are refused outright. The
class-conditional checkpoint is trained on an L0 corpus and emits none of them,
and a walker that guessed at the transform tier would be a second
implementation of the semantics `port/src/dm_vm.c` is bit-exact against.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..isa.spec import MAX_REPEAT_DEPTH, Op, spec_for
from ..vm.interp import CURVE_STEPS

#: The tier this walker deliberately does not model.
UNSUPPORTED = frozenset({Op.CALL, Op.XFORM, Op.ENDX, Op.REPEATX})


@dataclass(frozen=True)
class Alignment:
    """Where each emitted point came from, in instruction terms.

    `instruction_of_point[i]` is the index into `instructions` of the
    instruction that emitted the i-th point of the whole drawing, counting
    points in wire order. `stroke_points` is how many points each stroke was
    predicted to carry, which is what `verify` checks.
    """

    instruction_of_point: tuple[int, ...]
    stroke_points: tuple[int, ...]
    #: Byte offset of each instruction, so the page can highlight the bytecode
    #: view as well as the disassembly.
    offsets: tuple[int, ...]

    def as_dict(self) -> dict:
        return {
            "instruction_of_point": list(self.instruction_of_point),
            "stroke_points": list(self.stroke_points),
            "offsets": list(self.offsets),
        }


def align(program: bytes) -> Alignment | None:
    """Predict the point-to-instruction map, or None when it cannot be trusted.

    The rules mirror `dm.vm.interp.VM.run` exactly, and the mirroring is the
    risk: two implementations of one specification drift. `verify` is what makes
    that risk visible instead of silent, and the fallback is a listing with no
    highlight rather than a highlight that might be wrong.
    """
    offsets: list[int] = []
    pc = 0
    while pc < len(program):
        try:
            spec = spec_for(program[pc])
        except Exception:
            return None
        if spec.op in UNSUPPORTED:
            return None
        if pc + spec.size > len(program):
            return None
        offsets.append(pc)
        pc += spec.size

    index_of_offset = {offset: i for i, offset in enumerate(offsets)}

    instruction_of_point: list[int] = []
    stroke_points: list[int] = []
    path = 0            # points currently in the open path
    path_owner: list[int] = []   # the instruction index behind each of them
    stack: list[list[int]] = []  # [return offset, count, iteration]
    pc = 0
    have_cur = True     # `cur` starts at (0, 0) and is always defined

    def flush() -> None:
        nonlocal path, path_owner
        if path >= 2:
            stroke_points.append(path)
            instruction_of_point.extend(path_owner)
        path, path_owner = 0, []

    steps = 0
    while pc < len(program):
        steps += 1
        if steps > 1_000_000:
            return None
        spec = spec_for(program[pc])
        index = index_of_offset[pc]
        args = list(program[pc + 1:pc + spec.size])
        nxt = pc + spec.size

        if spec.op is Op.HALT:
            flush()
            break
        elif spec.op is Op.MOVE:
            flush()
            path, path_owner = 1, [index]
        elif spec.op is Op.LINE:
            if path == 0:
                # The VM seeds an empty path with `cur`, which no instruction
                # here emitted. Attributing it to the LINE that revived the path
                # is the only defensible choice and it is what the page shows.
                path, path_owner = 1, [index]
            path += 1
            path_owner.append(index)
        elif spec.op is Op.CURVE:
            if path == 0:
                path, path_owner = 1, [index]
            path += CURVE_STEPS
            path_owner.extend([index] * CURVE_STEPS)
        elif spec.op is Op.CIRCLE:
            pass        # a disc, not a path point
        elif spec.op is Op.WIDTH:
            flush()
        elif spec.op is Op.FILL:
            # A region's points leave the path and are reported by the device as
            # a region rather than a stroke, so they are not in the stroke
            # sequence this alignment indexes. Refused rather than approximated.
            return None
        elif spec.op is Op.REPEAT:
            count = args[0]
            if count == 0 or len(stack) >= MAX_REPEAT_DEPTH:
                return None
            stack.append([nxt, count, 0])
        elif spec.op is Op.ENDREP:
            if not stack:
                return None
            frame = stack[-1]
            frame[2] += 1
            if frame[2] < frame[1]:
                flush()
                nxt = frame[0]
            else:
                stack.pop()
        else:
            return None
        pc = nxt

    _ = have_cur
    return Alignment(tuple(instruction_of_point), tuple(stroke_points), tuple(offsets))


def verify(alignment: Alignment | None, device_strokes: list[dict]) -> Alignment | None:
    """Keep the alignment only if the device drew exactly what it predicted.

    `device_strokes` is the `geometry.strokes` list from a demo record: the
    stroke sequence as it came off the wire. Both the count of strokes and every
    stroke's point count must match. Anything else means the walker and
    `port/src/dm_vm.c` disagree about the ISA, which is a finding worth seeing
    in a test rather than a highlight worth showing on a page.
    """
    if alignment is None:
        return None
    actual = tuple(len(stroke["points"]) for stroke in device_strokes)
    if actual != alignment.stroke_points:
        return None
    return alignment
