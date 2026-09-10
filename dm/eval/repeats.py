"""Oracle L0 -> L1 compression: how much `REPEAT` structure a corpus contains.

Claim 2 asks a model to recover `REPEAT` from flat traces and scores it by the
L0 -> L1 compression ratio. That number is meaningless without a denominator:
**how much compressible repetition is in the corpus at all?** This module is
that denominator, computed by an oracle that is allowed to see the whole
program.

It is deliberately strict, because `REPEAT n dx dy` is:

- **translation only** -- no rotation, scale or mirror, so radial symmetry
  (clock ticks, gears, sun rays) is invisible here however obvious it looks;
- **exact** -- a body that recurs one unit off cannot be emitted losslessly, so
  quantisation that rounds two identical elements apart destroys the structure
  rather than blurring it (`docs/tier-c.md`: 0.62% against 3.17% on the same
  icons at +-1 px);
- **i8** -- per-iteration offsets outside [-128, 127] are not encodable.

`tol` exists to separate "this corpus has no repetition" from "this pipeline
destroyed it". A large gap between `tol=0` and `tol=1` is a preprocessing bug,
not a property of the data. Only `tol=0` is a real compression number.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..isa.asm import parse
from ..isa.spec import SPECS, Op
from ..isa.transform import Transform

#: REPEAT (4 bytes) + ENDREP (1). A repeat only pays if the body costs more.
REPEAT_OVERHEAD = SPECS[Op.REPEAT].size + SPECS[Op.ENDREP].size


@dataclass(frozen=True)
class Repeat:
    """`count` copies of `body_instrs` instructions from `start`, step (dx, dy)."""

    start: int
    body_instrs: int
    count: int
    dx: int
    dy: int
    saved: int
    #: Encoded size of one copy of the body. Appended with a default so no
    #: existing positional construction moves, and recorded because
    #: `body_bytes - body_instrs` is the **operand count** of a folded body --
    #: the `n` in the `(1 - p)^n` model `docs/direction.md` §7.3 predicts a
    #: control-point primitive by. A model whose exponent is guessed rather than
    #: measured is an argument; measured, it is a falsifiable prediction.
    body_bytes: int = 0

    @property
    def body_numbers(self) -> int:
        return max(0, self.body_bytes - self.body_instrs)


def _coords(instr) -> list[int]:
    """Operand values, or [] for an instruction with no coordinates."""
    return list(instr.args)


def _offset(a, b, tol: int) -> tuple[int, int] | None:
    """The constant (dx, dy) taking instruction `a` to `b`, if one exists."""
    if a.mnemonic != b.mnemonic:
        return None
    xa, xb = _coords(a), _coords(b)
    if len(xa) != len(xb) or not xa:
        return None if xa != xb else (0, 0)
    deltas = [q - p for p, q in zip(xa, xb)]
    dx, dy = deltas[0], deltas[1] if len(deltas) > 1 else deltas[0]
    for i, d in enumerate(deltas):
        if abs(d - (dx if i % 2 == 0 else dy)) > tol:
            return None
    return dx, dy


def _run_length(instrs, start: int, body: int, dx: int, dy: int, tol: int) -> int:
    """How many consecutive copies of the body follow, including the first."""
    count = 1
    while start + (count + 1) * body <= len(instrs):
        block = instrs[start + count * body : start + (count + 1) * body]
        base = instrs[start : start + body]
        ok = True
        for p, q in zip(base, block):
            if p.mnemonic != q.mnemonic:
                ok = False
                break
            for i, (u, v) in enumerate(zip(_coords(p), _coords(q))):
                if abs((v - u) - count * (dx if i % 2 == 0 else dy)) > tol:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
        count += 1
    return count


def best_repeat(instrs, max_body: int = 16, tol: int = 0) -> Repeat | None:
    """The single most profitable REPEAT in `instrs`, or None."""
    best: Repeat | None = None
    for start in range(len(instrs)):
        for body in range(1, min(max_body, (len(instrs) - start) // 2) + 1):
            step = _offset(instrs[start], instrs[start + body], tol)
            if step is None:
                continue
            dx, dy = step
            if not (-128 <= dx <= 127 and -128 <= dy <= 127):
                continue
            count = _run_length(instrs, start, body, dx, dy, tol)
            if count < 2:
                continue
            body_bytes = sum(SPECS[Op[i.mnemonic]].size for i in instrs[start : start + body])
            saved = (count - 1) * body_bytes - REPEAT_OVERHEAD
            if saved > 0 and (best is None or saved > best.saved):
                best = Repeat(start, body, count, dx, dy, saved, body_bytes)
    return best


def compress(program: bytes, max_body: int = 16, tol: int = 0) -> tuple[int, list[Repeat]]:
    """Greedily fold repeats. Returns (bytes saved, the repeats found).

    Greedy rather than optimal on purpose: this is a denominator, and an
    underestimate of available structure is the safe direction for a claim that
    a model *found* some.
    """
    instrs = parse(program)
    saved, found = 0, []
    while (repeat := best_repeat(instrs, max_body, tol)) is not None:
        saved += repeat.saved
        found.append(repeat)
        end = repeat.start + repeat.count * repeat.body_instrs
        instrs = instrs[: repeat.start + repeat.body_instrs] + instrs[end:]
    return saved, found


def corpus_stats(programs: list[bytes], max_body: int = 16, tol: int = 0) -> dict:
    """Oracle compression over a corpus. `ratio` is the number claim 2 needs."""
    total = saved = with_repeat = 0
    for program in programs:
        total += len(program)
        got, found = compress(program, max_body, tol)
        saved += got
        with_repeat += bool(found)
    n = max(1, len(programs))
    return {
        "n": len(programs),
        "tol": tol,
        "bytes": total,
        "saved": saved,
        "saved_frac": saved / max(1, total),
        "programs_with_repeat": with_repeat / n,
        "ratio": total / max(1, total - saved),
    }


# --------------------------------------------------------------------------
# ISA v2: repeats up to a symmetry


@dataclass(frozen=True)
class SymRepeat:
    """`count` copies from byte `start`, each the previous one under `step`.

    The transformed counterpart of `Repeat`, and the reason it is a separate
    type is that its `step` is a *group element plus* a translation where
    `Repeat`'s is only a translation. Collapsing them would make a mirrored
    orbit and a translated one indistinguishable in a table, which is the
    comparison this corpus exists to draw.
    """

    start: int
    body_bytes: int
    count: int
    step: Transform
    saved: int
    #: Instruction count of one copy, appended with a default for the same
    #: reason `Repeat.body_bytes` was: `body_bytes - body_instrs` is the operand
    #: count of the folded body, and that exponent is what decides whether a
    #: control-point primitive can put *scale* in the transform group.
    body_instrs: int = 0

    @property
    def body_numbers(self) -> int:
        return max(0, self.body_bytes - self.body_instrs)


def _sizes(instrs) -> list[int]:
    return [SPECS[Op[i.mnemonic]].size for i in instrs]


def best_symmetry_repeat(program: bytes, max_body: int = 64, tol: int = 0):
    """The most profitable orbit under D4 ⋉ translation, or None.

    **`REPEAT` cannot express this and `REPEATX` can**, which is the entire
    point: the translational oracle above is blind to a mirrored copy, because
    `x -> 255 - x` changes every coordinate byte and byte matching finds
    nothing. Measured on the constructed corpus: the translational matcher finds
    a repeat in 60 of 60 control scenes and **0 of 60** transformed ones.

    The search costs eight `apply` calls per program and no new transform code.
    For each element `g`, the whole program is transformed once; then copy `k+1`
    is a copy of copy `k` under `(g, t)` exactly when it is a plain *translate*
    of block `k` of the transformed program -- so the existing per-instruction
    offset test does the work, on a pre-transformed operand. Re-deriving the
    by-`Kind` transform here to go faster would be a second copy of the rule in
    `dm/data/augment.py`, and the two would eventually disagree.
    """
    from ..data.augment import Affine, apply
    from ..isa.transform import D4, D4_ORDER

    instrs = list(parse(program))
    if len(instrs) < 2:
        return None
    offsets = [0]
    for size in _sizes(instrs):
        offsets.append(offsets[-1] + size)

    best = None
    for code in range(D4_ORDER):
        d4 = D4.of(code)
        turned_bytes = apply(program, Affine(0, 0, d4.mirror, d4.turns))
        if turned_bytes is None:
            continue
        turned = list(parse(turned_bytes))
        for start in range(len(instrs)):
            for body in range(1, min(max_body, (len(instrs) - start) // 2) + 1):
                step = _offset(turned[start], instrs[start + body], tol)
                if step is None:
                    continue
                dx, dy = step
                if not (-128 <= dx <= 127 and -128 <= dy <= 127):
                    continue
                count = 1
                while start + (count + 1) * body <= len(instrs):
                    lo = start + (count - 1) * body
                    ok = all(
                        _offset(turned[lo + j], instrs[lo + body + j], tol) == (dx, dy)
                        for j in range(body)
                    )
                    if not ok:
                        break
                    count += 1
                if count < 2:
                    continue
                body_bytes = offsets[start + body] - offsets[start]
                # `REPEATX` is one byte wider than `REPEAT`: it carries the
                # element as well as the step.
                saved = (count - 1) * body_bytes - (SPECS[Op.REPEATX].size
                                                    + SPECS[Op.ENDREP].size)
                if saved > 0 and (best is None or saved > best.saved):
                    best = SymRepeat(offsets[start], body_bytes, count,
                                     Transform(d4, dx, dy), saved, body)
    return best


def symmetry_stats(programs: list[bytes], max_body: int = 64, tol: int = 0) -> dict:
    """The transformed ceiling of a corpus, as `corpus_stats` reports the flat one.

    One orbit per program, which is what the constructed corpus contains by
    construction and therefore what a validation against its provenance can
    check. On a natural corpus it is a *lower* bound on the available structure
    and is reported as such -- a drawing with two independent symmetric parts
    has more than this finds.
    """
    total = saved = 0
    found = 0
    elements: dict[int, int] = {}
    for program in programs:
        total += len(program)
        best = best_symmetry_repeat(program, max_body=max_body, tol=tol)
        if best is None:
            continue
        found += 1
        saved += best.saved
        code = best.step.d4.code
        elements[code] = elements.get(code, 0) + 1
    return {
        "n": len(programs),
        "with_orbit": found,
        "bytes": total,
        "saved": saved,
        "fraction": saved / total if total else 0.0,
        "elements": dict(sorted(elements.items())),
    }


# --------------------------------------------------------------------------
# how large a folded body is, which is the exponent of the divergence model


def body_numbers(programs: list[bytes], max_body: int = 16, tol: int = 0,
                 symmetry: bool = False) -> dict:
    """Operand counts of the bodies each oracle actually folds.

    `docs/direction.md` §7.3 models a scale's damage as an independent
    per-operand divergence probability over an `n`-operand body, fits `p` from
    one representation's measured retention, and predicts another's at a smaller
    `n`. **`n` is written there as an estimate ("`n ≈ 40`") and it does not have
    to be one** -- the oracle knows exactly which bodies it folded, so the
    exponent is a measurement like everything else it reports. Quoting the model
    at a guessed `n` would make its prediction unfalsifiable in the direction
    that matters, because a miss could always be blamed on the guess.

    Weighted by bytes saved rather than by occurrence: a body that carries most
    of a corpus's saving is most of what a scale can destroy, and an unweighted
    mean would be dominated by the many tiny folds that save two bytes each.
    """
    numbers: list[int] = []
    weights: list[int] = []
    for program in programs:
        if symmetry:
            best = best_symmetry_repeat(program, max_body=max_body, tol=tol)
            found = [best] if best else []
        else:
            found = compress(program, max_body, tol)[1]
        for repeat in found:
            numbers.append(repeat.body_numbers)
            weights.append(max(1, repeat.saved))
    if not numbers:
        return {"bodies": 0, "mean": float("nan"), "weighted_mean": float("nan")}
    total = sum(weights)
    return {
        "bodies": len(numbers),
        "mean": sum(numbers) / len(numbers),
        "weighted_mean": sum(n * w for n, w in zip(numbers, weights)) / total,
        "min": min(numbers),
        "max": max(numbers),
    }
