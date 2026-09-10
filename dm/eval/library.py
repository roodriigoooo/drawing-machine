"""Oracle L0 -> L2: how much a corpus repeats itself **between** drawings.

`dm/eval/repeats.py` measures what one program repeats inside itself, which is
what `REPEAT` and `REPEATX` can fold. Nothing in this project had ever looked
between two drawings, and two separate threads need that number:

- **`Op.CALL` is allocated at `Tier.L2` and unbuilt** (`dm/isa/spec.py`), and
  `docs/traps.md` carries the rule that paid for `REPEATX`: *before adding an
  opcode to remove redundancy, measure the redundancy it would remove.* A shared
  body is exactly what a call folds, and its ceiling is countable.
- **Canonicalisation has nowhere else to be scored.** `dm/data/canonical.py`
  proves that an exact re-framing leaves both within-program ceilings *exactly*
  unchanged, so `docs/direction.md` §7.1's proposed test is a provable null. Its
  actual claim -- "two people drawing the same object at different tilts produce
  more similar programs" -- is a statement about pairs of drawings, and this is
  where a pair of drawings can be compared.

**The unit is a stroke, and it is normalised, not matched raw.** A body that
recurs at another position is still a body: `CALL` places it, so position cannot
be part of its identity. Two normalisations are reported and they answer
different questions:

    translation   the same shape, anywhere on the canvas
    d4            the same shape up to one of the eight square symmetries

The second is the one an `XFORM`-placed call could actually reuse, and it is
also the one under which a **global D4 canonicalisation is again a null** -- the
oracle has already quotiented by exactly the group the policy applies. So a
canonicalisation that helps must show up in the *translation* column, and a
policy that only moves the `d4` column has moved an artifact.

**Only `grid = 1` is a compression number.** `dm/eval/repeats.py` carries `tol`
for the same reason: a coarser match separates "this corpus shares no strokes"
from "this pipeline rounded shared strokes apart", and the second is a bug in
the pipeline rather than a property of the data. A saving computed at `grid > 1`
is not lossless and is reported as a detector reading, never as bytes.
"""

from __future__ import annotations

from collections import Counter

from ..isa.asm import AsmError, parse
from ..isa.spec import BY_MNEMONIC, SPECS, ISAError, Kind, Op
from ..isa.strokes import split
from ..isa.transform import D4, D4_ORDER

#: What one call site costs in the ISA as specified: `XFORM code dx dy` to place
#: the body, `CALL id` to invoke it, `ENDX` to close the scope. `CALL` carries no
#: placement operand of its own -- it takes a `Kind.ID` and nothing else -- so a
#: library body has to be positioned by the transform tier, and pretending
#: otherwise would price a different opcode than the one that exists.
CALL_SITE_BYTES = SPECS[Op.XFORM].size + SPECS[Op.CALL].size + SPECS[Op.ENDX].size

#: Library bodies addressable by one `Kind.ID` byte. A ceiling computed over an
#: unbounded library is not a ceiling for this ISA.
MAX_LIBRARY = 256


def _relative(stroke: bytes) -> tuple[tuple[str, ...], list[tuple[int, int]]] | None:
    """A stroke as `(mnemonics, offsets from its own first point)`.

    `HALT` is dropped: it terminates the *program*, not the shape, so leaving it
    in would make the last stroke of every drawing unmatchable against the
    identical shape drawn earlier -- an artifact of where the drawing ended.
    A stroke carrying any non-`COORD` operand is refused rather than keyed on it,
    because a width or a repeat count is not geometry and two bodies that differ
    in one are not the same body.
    """
    try:
        instrs = parse(stroke)
    except (AsmError, ISAError):
        return None
    mnemonics: list[str] = []
    points: list[tuple[int, int]] = []
    for instr in instrs:
        if instr.mnemonic == "HALT":
            continue
        spec = BY_MNEMONIC[instr.mnemonic]
        if any(k is not Kind.COORD for k in spec.operands):
            return None
        mnemonics.append(instr.mnemonic)
        points += [(instr.args[i], instr.args[i + 1]) for i in range(0, len(instr.args), 2)]
    if not points:
        return None
    x0, y0 = points[0]
    return tuple(mnemonics), [(x - x0, y - y0) for x, y in points]


def _key(shape, grid: int, d4: bool):
    """The identity of a body under the requested normalisation."""
    mnemonics, offsets = shape
    if grid > 1:
        offsets = [(round(x / grid), round(y / grid)) for x, y in offsets]
    if not d4:
        return (mnemonics, tuple(offsets))
    best = None
    for code in range(D4_ORDER):
        element = D4.of(code)
        # The *linear* part: the body has already been anchored at its own first
        # point, so what is left is a vector and takes the rotation without the
        # canvas translation -- `dm/isa/transform.py`'s distinction, and applying
        # `point` here instead would fold the anchor back in and make the key
        # depend on position after all.
        turned = tuple(element.linear(x, y) for x, y in offsets)
        if best is None or turned < best:
            best = turned
    return (mnemonics, best)


def library_stats(programs: list[bytes], grid: int = 1, d4: bool = False,
                  min_bytes: int = 0) -> dict:
    """The L2 ceiling of a corpus: bytes a shared-body library could remove.

    A body of `b` bytes occurring `k` times costs `k*b` spelled out and
    `b + CALL_SITE_BYTES*k` as a library entry plus `k` call sites, so it saves
    `(k - 1)*b - CALL_SITE_BYTES*k`. Bodies are admitted in descending order of
    saving until the 256 the `ID` operand can address are used, which is what
    makes this a ceiling for *this* ISA rather than for an idealised one.

    Greedy and therefore an underestimate, deliberately: this is a denominator,
    and the safe direction for a denominator is low.
    """
    shapes = []
    for program in programs:
        for stroke in split(program):
            shape = _relative(stroke)
            if shape is not None and len(stroke) >= min_bytes:
                shapes.append((shape, _body_bytes(shape)))

    counts: Counter = Counter()
    sizes: dict = {}
    for shape, size in shapes:
        key = _key(shape, grid, d4)
        counts[key] += 1
        sizes[key] = size

    savings = []
    for key, k in counts.items():
        b = sizes[key]
        saved = (k - 1) * b - CALL_SITE_BYTES * k
        if saved > 0:
            savings.append((saved, k, b))
    savings.sort(reverse=True)
    chosen = savings[:MAX_LIBRARY]

    total = sum(map(len, programs))
    saved = sum(s for s, _, _ in chosen)
    covered = sum(k for _, k, _ in chosen)
    return {
        "n": len(programs),
        "grid": grid,
        "d4": d4,
        "bytes": total,
        "strokes": len(shapes),
        "distinct": len(counts),
        "reuse_rate": 1.0 - len(counts) / max(1, len(shapes)),
        "library": len(chosen),
        "call_sites": covered,
        "saved": saved,
        "fraction": saved / total if total else 0.0,
        "exact": grid == 1,
        "largest_group": max((k for _, k, _ in chosen), default=0),
    }


def _body_bytes(shape) -> int:
    """The body's own byte length, which is what a call replaces.

    Computed from the ISA table rather than from the source stroke, because the
    body a library stores is re-anchored and a `HALT` has been dropped -- so the
    stroke's own length is not what the call site saves.
    """
    mnemonics, _ = shape
    return sum(SPECS[Op[m]].size for m in mnemonics)


__all__ = ["CALL_SITE_BYTES", "MAX_LIBRARY", "library_stats"]


# `_body_bytes` is defined after `library_stats` for reading order; the module is
# imported before either runs, so the forward reference is resolved by then.
