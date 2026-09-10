"""L1 -> L0: the program's flat trace, as bytecode.

Claim 2 asks whether a model trained on **flat traces** recovers `REPEAT`, and
that phrasing is Ellis et al.'s program-vs-trace distinction: the trace is what
the drawing looks like, the program is the structure behind it, and the claim is
that the second is recoverable from the first. So claim 2 needs a corpus that
*contains* translational repetition and *does not* contain the `REPEAT` opcode.
Tier A had neither half at once -- at `Tier.L0` the `random_grid` motif is
excluded outright, and at `Tier.L1` the repeat is handed to the model as an
opcode -- so until now claim 2's only corpus was Tier C, which has 4,613
programs and no budget at which both convergence guards pass (`PLAN.md` 7).

Unrolling gives Tier A the missing half at 100k programs, with two properties
Tier C cannot offer:

- **The ceiling is known by construction, not estimated.** The generator chose
  the count and the step, so `dm.eval.repeats` can be checked against ground
  truth rather than trusted.
- **The count is a knob.** Train on `max_repeat=4` and evaluate on 16, and the
  evaluation sequences are ~4x longer than anything seen in training, which is
  the length-generalisation test RoPE was chosen for
  (`dm/models/transformer.py`).

**Refuses rather than clamps.** The VM clamps a placed coordinate to the canvas;
an unrolled copy that needed clamping is no longer an exact translate of its
body, so `REPEAT n dx dy` could not re-emit it and the oracle would not score it.
Emitting the clamped bytes anyway would put near-repeats into the one corpus
built to contain exact ones -- the SVG-Icons8 failure, reproduced by our own
hand (`docs/tier-c.md`). `None` is the honest answer, and the caller decides.
"""

from __future__ import annotations

from .asm import Instr, parse
from .spec import MAX_REPEAT_DEPTH, SPECS, ISAError, Op
from .transform import IDENTITY, Transform

#: Bound on an expansion, in bytecode bytes. Nested repeats multiply -- four
#: levels at count 4 is 256 copies of the innermost body -- and an unbounded
#: expansion would silently produce programs past any `max_len` the sweep uses,
#: where truncation biases the granularity axis specifically (`PLAN.md` 10).
MAX_EXPANSION = 4096


class Unrollable(ISAError):
    """The program has no exact flat trace.

    Raised only by `sample`-level callers that need a program or nothing;
    `unroll` itself returns None so the caller can count rejections.
    """


def _chunks(program: bytes) -> list[tuple[Instr, bytes]]:
    """Each instruction beside its own bytes, so a body can be copied verbatim.

    Re-assembling from `Instr` would work and is what the first version did; it
    also re-encodes every operand through `encode_operand`, which is a second
    path to the same bytes and therefore a second place for them to disagree.
    """
    out, pc = [], 0
    for instr in parse(program):
        size = SPECS[Op[instr.mnemonic]].size
        out.append((instr, program[pc : pc + size]))
        pc += size
    return out


#: Which closer each scope-opening instruction requires. One table rather than a
#: counter per bracket type, because the two kinds of scope can nest inside each
#: other and a counter cannot tell proper nesting from a crossing -- and a
#: crossing is a program the VM faults on, so flattening one would invent a
#: drawing that cannot be executed.
_CLOSES = {"REPEAT": "ENDREP", "REPEATX": "ENDREP", "XFORM": "ENDX"}
_CLOSERS = frozenset(_CLOSES.values())


def _matching(chunks: list[tuple[Instr, bytes]], start: int) -> int | None:
    """Index just past the closer matching the opener at `start`, or None.

    None on a crossing (`REPEAT … XFORM … ENDREP … ENDX`), on an unterminated
    scope and on a closer of the wrong kind -- the three shapes the VM reports
    as `UNTERMINATED_REPEAT` or `UNMATCHED_ENDREP`.
    """
    want: list[str] = []
    for index in range(start, len(chunks)):
        mnemonic = chunks[index][0].mnemonic
        if mnemonic in _CLOSES:
            want.append(_CLOSES[mnemonic])
        elif mnemonic in _CLOSERS:
            if not want or want.pop() != mnemonic:
                return None
            if not want:
                return index + 1
    return None


def _expand(chunks: list[tuple[Instr, bytes]], depth: int) -> bytes | None:
    # Imported here rather than at module scope: `dm.data.augment` is the
    # by-Kind transform rule, and it imports the assembler, which imports this
    # module's siblings. Keeping the edge inside the function keeps `dm.isa`
    # free of a dependency on `dm.data`.
    from ..data.augment import Affine, apply

    out = bytearray()
    index = 0
    while index < len(chunks):
        instr, raw = chunks[index]
        if instr.mnemonic in _CLOSERS:
            return None  # unmatched: the VM faults, and so does this
        if instr.mnemonic not in _CLOSES:
            out += raw
            index += 1
            continue
        if depth >= MAX_REPEAT_DEPTH:
            return None  # the VM's DEPTH_OVERFLOW, refused for the same reason

        end = _matching(chunks, index)
        if end is None:
            return None  # unterminated, crossed, or closed by the wrong opcode
        body = _expand(chunks[index + 1 : end - 1], depth + 1)
        if body is None:
            return None

        # One transform per iteration, and `REPEAT` is the rung of this ladder
        # where the transform is a pure translation. The VM computes the same
        # sequence from `_Frame.step`, so the two agree by construction rather
        # than by a second reading of the rule.
        if instr.mnemonic == "XFORM":
            steps = [Transform.of(*instr.args)]
        else:
            count = instr.args[0]
            if count < 1:
                return None  # ZERO_REPEAT: a fault, not a drawing
            step = (Transform(IDENTITY.d4, *instr.args[1:])
                    if instr.mnemonic == "REPEAT" else Transform.of(*instr.args[1:]))
            steps = [step.power(k) for k in range(count)]

        for step in steps:
            # `apply` carries the by-`Kind` rule, so a nested `REPEAT`'s own
            # dx/dy is a displacement and correctly does not move, and a nested
            # scope's transform is conjugated into this frame rather than left
            # acting in the old one.
            copy = body if step.is_identity else apply(body, Affine.of(step))
            if copy is None:
                return None  # left the canvas: this copy is not an exact image
            out += copy
            if len(out) > MAX_EXPANSION:
                return None
        index = end
    return bytes(out)


def unroll(program: bytes) -> bytes | None:
    """The flat L0 trace of `program`, or None if it has no exact one.

    Geometry-preserving: `VM.run(unroll(p)).strokes == VM.run(p).strokes` for
    every program this returns bytes for, which `tests/test_unroll.py` pins
    against the VM rather than against a second implementation of the rule.
    """
    try:
        chunks = _chunks(program)
    except ISAError:
        return None
    return _expand(chunks, depth=0)
