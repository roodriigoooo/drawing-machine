"""Re-spelling a corpus's geometry, so two primitives can be compared fairly.

The Bézier question (`docs/direction.md` §7.3) is not "are cubics nice"; it is
whether a control-point body is small enough that a ×1.07 scale stops destroying
the repeat ceiling. Answering it needs two corpora that differ **in the
primitive and in nothing else**, and that is harder than it sounds:

- Tabler's programs already contain `CURVE`, because its source is cubics. So
  "Tabler as authored" is neither arm; it is the mixture, and it is the corpus
  the 23%-retention number on record was measured on -- which makes it the
  *calibration* row and not a comparison row.
- Comparing bytes at two different fidelities is not a comparison. `rdp_eps` and
  a curve-fitting tolerance are different knobs with different units, so a table
  indexed by knob setting says nothing. Both arms here take the same tolerance
  through the same recursion (`dm.isa.bezier.fit_polyline`), differing in one
  boolean, and the *measured* error is reported beside the bytes so the reading
  is made at matched fidelity or not at all.

So the pipeline is: execute the program, take the geometry the VM actually draws,
optionally re-frame it (`dm.data.canonical`), and re-spell it under one
primitive. **The result is a different corpus and gets a different fingerprint**
-- every byte count, ceiling and `bits/drawing` on record belongs to the old
spelling, and `dm/data/fingerprint.py` is what keeps that from being an argument.

One refusal is deliberate. A trace carrying discs or filled regions cannot be
re-spelled by a stroke fitter, and dropping them would quietly change the
drawing. They are counted and the program is refused, so a corpus that contains
any is visible in the report rather than in the residuals.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..isa.asm import AsmError, assemble, parse
from ..isa.bezier import (
    draw,
    fit_polyline,
    hausdorff,
    round_half_up,
    stroke_bytes,
)
from ..isa.spec import BY_MNEMONIC, CANVAS, ISAError, Kind, Op
from ..vm.interp import VM, Trace
from .canonical import POLICIES, Policy, canonicalise_program, reframe


@dataclass
class RespellStats:
    """What the re-spelling did, counted rather than assumed."""

    n: int = 0
    refused: int = 0
    empty: int = 0
    with_regions: int = 0
    bytes: int = 0
    strokes: int = 0
    segments: int = 0
    curves: int = 0
    lines: int = 0
    errors: list[float] = field(default_factory=list)

    def summary(self) -> dict:
        n = max(1, self.n)
        errors = np.asarray(self.errors, dtype=np.float64)
        return {
            "n": self.n,
            "refused": self.refused,
            "empty": self.empty,
            "with_regions": self.with_regions,
            "bytes_per_drawing": self.bytes / n,
            "strokes_per_drawing": self.strokes / n,
            "segments_per_drawing": self.segments / n,
            "curve_fraction": self.curves / max(1, self.segments),
            "numbers_per_segment": (6 * self.curves + 2 * self.lines)
            / max(1, self.segments),
            "error_mean": float(errors.mean()) if len(errors) else float("nan"),
            "error_p95": float(np.percentile(errors, 95)) if len(errors) else float("nan"),
            "error_max": float(errors.max()) if len(errors) else float("nan"),
        }


def geometry(trace: Trace) -> list[np.ndarray]:
    """The stroke polylines a trace draws, as float arrays."""
    return [np.asarray(s.points, dtype=np.float64) for s in trace.strokes]


def respell(program: bytes, tol: float, allow_curve: bool, policy: Policy,
            vm: VM | None = None) -> tuple[bytes, float] | None:
    """One program, re-framed and re-spelled. `None` when it cannot be.

    The returned error is the worst-case distance between the source geometry
    (post-frame, so the two arms are scored against the same thing) and the
    geometry the re-spelled program draws -- measured stroke by stroke and
    maximised, because a mean would let one badly fitted stroke hide behind
    thirty good ones and the tolerance is a *bound*.
    """
    trace = (vm or VM()).run(program)
    if trace.discs or trace.regions:
        return None
    strokes = [s for s in geometry(trace) if len(s) >= 2]
    if not strokes:
        return None
    strokes = reframe(strokes, policy)

    out = bytearray()
    worst = 0.0
    for stroke in strokes:
        start, segments = fit_polyline(stroke, tol, allow_curve=allow_curve)
        if not segments:
            continue
        out += stroke_bytes(start, segments)
        worst = max(worst, hausdorff(stroke, draw(start, segments)))
    if not out:
        return None
    out.append(int(Op.HALT))
    return canonicalise_program(bytes(out), policy), worst


def respell_corpus(programs: list[bytes], tol: float, allow_curve: bool,
                   policy: Policy | str = "none") -> tuple[list[bytes], dict]:
    """A whole corpus re-spelled, with the counts that make it readable."""
    chosen = POLICIES[policy] if isinstance(policy, str) else policy
    vm = VM()
    stats = RespellStats()
    out: list[bytes] = []
    keep: list[int] = []
    for index, program in enumerate(programs):
        result = respell(program, tol, allow_curve, chosen, vm)
        if result is None:
            stats.refused += 1
            trace = vm.run(program)
            stats.with_regions += bool(trace.discs or trace.regions)
            continue
        respelled, error = result
        out.append(respelled)
        # The source index travels with the program. Two spellings of one corpus
        # refuse different drawings, and a comparison between them has to be
        # paired on the *source* -- differencing two arms over two different
        # subsets is the val-set fault this project has already paid for once.
        keep.append(index)
        stats.n += 1
        stats.bytes += len(respelled)
        stats.errors.append(error)
        for instr in parse(respelled):
            if instr.mnemonic == "MOVE":
                stats.strokes += 1
            elif instr.mnemonic == "LINE":
                stats.segments += 1
                stats.lines += 1
            elif instr.mnemonic == "CURVE":
                stats.segments += 1
                stats.curves += 1
    return out, {"tol": tol, "primitive": "bezier" if allow_curve else "polyline",
                 "policy": chosen.name, "exact_policy": chosen.exact,
                 "keep": keep, **stats.summary()}


# ---------------------------------------------------------------------------
# the operation under test


def scale_program(program: bytes, factor: float,
                  phase: tuple[float, float] = (0.0, 0.0),
                  about: str = "bbox") -> bytes | None:
    """Scale a drawing about its own bounding-box centre, rounding to integers.

    **This is the operation `docs/direction.md` §7.3 names, applied to the
    representation rather than to the geometry**, and the distinction is the
    whole experiment: an affine image of a Bézier is the Bézier of the
    transformed control points, so scaling the operands is exact in the reals and
    the *only* loss is that each operand rounds on its own. A repeat survives
    when every operand of its body rounds the same way, which is why the body's
    operand count is the quantity under test.

    **`phase` is not a detail, and leaving it out is how this measurement goes
    wrong.** A sub-pixel offset added before rounding decides which side each
    coordinate falls on, and on a corpus authored on a coarse sublattice --
    Tabler is a 24 grid at ×10, so most coordinates are multiples of ten -- the
    phases resonate: the identical corpus under the identical ×1.07 retains
    **18.7%** of its repeat ceiling scaled about the origin and **34.8%** about
    its own bounding-box centre, which are the same operation at two phases. A
    single centre is therefore one *draw*, not a number, and a retention quoted
    from one is a resolution-floor error of the kind `docs/traps.md` already
    carries for sampling columns. Callers sweep it and report the spread.

    Refused rather than clamped when anything leaves the canvas or an `i8`, for
    `dm/data/augment.py`'s reason: clamping is a deformation, and a deformation
    of a repeated body is not a repeat. Refusals are the caller's to count and
    the arms must be compared on the programs that survive in **both**.

    Operands move by `Kind`, never by position: a radius and a width are lengths
    and take the scale, a `REPEAT` step is a vector and takes it too, a count and
    a transform code are invariant. Treating every byte as a coordinate would
    corrupt all three, which is the same rule `augment.apply` follows.
    """
    try:
        instrs = parse(program)
    except (AsmError, ISAError):
        return None
    xs: list[int] = []
    ys: list[int] = []
    for instr in instrs:
        spec = BY_MNEMONIC[instr.mnemonic]
        index = 0
        while index < len(spec.operands):
            if spec.operands[index] is Kind.COORD:
                xs.append(instr.args[index])
                ys.append(instr.args[index + 1])
                index += 2
            else:
                index += 1
    if not xs:
        return program
    if about == "origin":
        # Every drawing then rounds at the *same* phase, which is what a corpus
        # -wide augmentation policy does and is how the 77%-destroyed number on
        # record was produced. It is a different measurement from the per-drawing
        # centring below -- not a worse one -- and the difference is exactly the
        # phase-averaging that makes one of them low-variance.
        cx = cy = 0.0
    elif about == "bbox":
        cx = (min(xs) + max(xs)) / 2.0
        cy = (min(ys) + max(ys)) / 2.0
    else:
        raise ValueError(f"unknown scale centre {about!r}")

    lines: list[str] = []
    for instr in instrs:
        spec = BY_MNEMONIC[instr.mnemonic]
        args = list(instr.args)
        out: list[int] = []
        index = 0
        while index < len(spec.operands):
            kind = spec.operands[index]
            if kind is Kind.COORD:
                x = round_half_up(factor * (args[index] - cx) + cx + phase[0])
                y = round_half_up(factor * (args[index + 1] - cy) + cy + phase[1])
                if not (0 <= x < CANVAS and 0 <= y < CANVAS):
                    return None
                out += [x, y]
                index += 2
            elif kind is Kind.DELTA:
                dx = round_half_up(factor * args[index])
                dy = round_half_up(factor * args[index + 1])
                if not (-128 <= dx <= 127 and -128 <= dy <= 127):
                    return None
                out += [dx, dy]
                index += 2
            elif kind is Kind.SCALAR:
                value = round_half_up(factor * args[index])
                if not 0 <= value <= 255:
                    return None
                out.append(value)
                index += 1
            else:
                out.append(args[index])
                index += 1
        lines.append(" ".join([instr.mnemonic, *map(str, out)]))
    return assemble("\n".join(lines))


def jitter_program(program: bytes, amplitude: int, rng: np.random.Generator) -> bytes | None:
    """The other inexact operation on record (±1 destroys 65%), for the same table.

    Kept beside `scale_program` because the two share a mechanism -- independent
    per-operand divergence -- and a control-point body should retain more of the
    ceiling under both if the mechanism is what the model says it is. Unlike the
    scale it has no exact-in-the-reals story at all, so it is the harder test.
    """
    try:
        instrs = parse(program)
    except (AsmError, ISAError):
        return None
    lines: list[str] = []
    for instr in instrs:
        spec = BY_MNEMONIC[instr.mnemonic]
        args = list(instr.args)
        out: list[int] = []
        index = 0
        while index < len(spec.operands):
            kind = spec.operands[index]
            if kind is Kind.COORD:
                shift = rng.integers(-amplitude, amplitude + 1, size=2)
                x, y = args[index] + int(shift[0]), args[index + 1] + int(shift[1])
                if not (0 <= x < CANVAS and 0 <= y < CANVAS):
                    return None
                out += [x, y]
                index += 2
            else:
                out.append(args[index])
                index += 1
        lines.append(" ".join([instr.mnemonic, *map(str, out)]))
    return assemble("\n".join(lines))


__all__ = [
    "RespellStats",
    "geometry",
    "jitter_program",
    "respell",
    "respell_corpus",
    "scale_program",
]
