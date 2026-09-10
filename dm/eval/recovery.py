"""Claim 2's metric: how much of the oracle's `REPEAT` structure did it find?

`dm.eval.repeats` gives the ceiling -- the bytes an oracle could drop by folding
exact translational repeats. This gives the numerator: **how much of that
redundancy the model has stopped paying for.**

The measurement needs no `REPEAT` decoder and no second training run. A repeated
body occurs two or more times in a flat L0 program. If the model has learned
nothing about repetition, copy *k* costs what copy 1 cost. If it has recovered
the structure, later copies are nearly free, because they are determined by the
first copy and a constant offset.

    recovery = 1 - (bits per byte on copies 2..n) / (bits per byte on copy 1)

0.0 means the model pays full price every time; 1.0 means later copies are free.

**Each body is compared against its own first copy**, not against the rest of
the program. The two sets then hold content fixed and vary only position, so the
number cannot be inflated by repeated bodies happening to be made of cheap
instructions. Rates per byte, never totals: the redundant set is smaller than
the first-copy set by construction, so a ratio of totals would report ~1.0 for a
model that had learned nothing.

Two deliberate limits. Only the exact translational repetition `REPEAT n dx dy`
can express is credited, so the result stays a fraction of the oracle ceiling.
And recovery is never compared across corpora: the ceiling is a property of the
corpus (Tabler 3.79%, QuickDraw 0.01%), so it is read against its own
denominator or not at all.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np
import torch
import torch.nn.functional as F

from ..data.dataset import ProgramDataset, collate
from ..isa.asm import parse
from ..isa.codec import PAD, Codec
from ..isa.spec import SPECS, Op
from .repeats import compress


def repeat_copies(program: bytes, max_body: int = 16, tol: int = 0) -> list[list[slice]]:
    """For each oracle repeat, the byte range of every copy, in order.

    `compress` folds greedily and reports positions in the *remaining*
    instruction list, so the same folding is replayed here to map them back onto
    the original byte offsets.

    Per copy rather than first-versus-rest because the length-generalisation
    test needs the ordinal: trained on `REPEAT n <= 4`, a model that learnt the
    *rule* pays the same for copy 12 as for copy 3, and one that learnt a table
    of short repeats does not. Collapsing copies 2..n into one span, which is
    all `recovery` needs, cannot see that difference at all.
    """
    sizes = [SPECS[Op[i.mnemonic]].size for i in parse(program)]
    offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
    alive = list(range(len(sizes)))
    out: list[list[slice]] = []
    for repeat in compress(program, max_body=max_body, tol=tol)[1]:
        block = alive[repeat.start : repeat.start + repeat.count * repeat.body_instrs]
        copies = [
            block[k * repeat.body_instrs : (k + 1) * repeat.body_instrs]
            for k in range(repeat.count)
        ]
        out.append([slice(offsets[c[0]], offsets[c[-1] + 1]) for c in copies])
        alive = alive[: repeat.start + repeat.body_instrs] + \
            alive[repeat.start + repeat.count * repeat.body_instrs :]
    return out


def repeat_spans(program: bytes, max_body: int = 16, tol: int = 0) -> list[tuple[slice, slice]]:
    """For each oracle repeat, the byte range of copy 1 and of copies 2..n.

    Copies 2..n are contiguous by construction -- a repeat is `count`
    consecutive body blocks -- so the later set is one span rather than a list.
    """
    return [
        (copies[0], slice(copies[1].start, copies[-1].stop))
        for copies in repeat_copies(program, max_body=max_body, tol=tol)
        if len(copies) > 1
    ]


@torch.no_grad()
def _symbol_bits(
    model,
    programs: list[bytes],
    codec: Codec,
    device: str | torch.device,
    max_len: int,
    batch_size: int,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield `(program index, per-target-position bits)` for every program.

    Shared by both measurements below so there is one place where the model is
    run and one place where the BOS shift is reasoned about. `with_bos`
    prepends BOS and `collate` shifts targets left by one, so bytecode byte `b`
    lands on target positions `[b * stride, (b + 1) * stride)` -- the identity
    that lets a byte-domain span be read off a symbol-domain array.
    """
    dataset = ProgramDataset(programs, codec, max_len)
    model.eval()
    for start in range(0, len(dataset), batch_size):
        batch = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
        index, inputs, targets = collate(batch)
        inputs, targets = inputs.to(device), targets.to(device)
        nll = F.cross_entropy(
            model(inputs).reshape(-1, codec.vocab_size),
            targets.reshape(-1),
            ignore_index=PAD,
            reduction="none",
        ).view(targets.shape).cpu().numpy() / math.log(2)
        for row, program_index in enumerate(index.tolist()):
            yield program_index, nll[row]


def _span_bits(bits: np.ndarray, span: slice, stride: int) -> tuple[float, int]:
    """Total bits and symbol count for a byte-domain span, clipped to `bits`.

    Clipping rather than assuming: `max_len` truncates, and it truncates the
    bit arm first, so a span past the cut has to contribute nothing to both the
    numerator and the denominator or the rate is computed over a window the
    model was never scored on.
    """
    lo, hi = span.start * stride, min(span.stop * stride, len(bits))
    if hi <= lo:
        return 0.0, 0
    return float(bits[lo:hi].sum()), hi - lo


def recovery(
    model,
    programs: list[bytes],
    codec: Codec,
    device: str | torch.device = "cpu",
    max_len: int = 2048,
    batch_size: int = 32,
    tol: int = 0,
    spans: list[list[tuple[slice, slice]]] | None = None,
) -> dict:
    """Bits per symbol on redundant copies against their own first copy.

    Codec-agnostic: one bytecode byte is `codec.stride` symbols, so byte spans
    are expanded into the codec's own alphabet before scoring.

    **Not comparable across the relativity axis.** Under a relative codec
    (`dm.isa.relative`) a translational repeat is a literally repeated symbol
    sequence, so later copies are cheap because the alphabet says so and not
    because the model discovered anything. A delta arm's recovery number is a
    contrast, and it belongs beside the absolute arm's or nowhere.

    **`spans` overrides detection, and on a transformed corpus it must.**
    `repeat_spans` finds *exact translational* repeats by matching bytes, and a
    mirrored copy does not match -- `x -> 255 - x` changes every coordinate byte
    -- so a detector-driven reading of `dm/data/composed.py` would find no
    repeats at all and report `NaN`. A constructed corpus knows its own copies
    and passes them in; the detector then becomes something to check *against*
    that ground truth rather than something the result rests on.
    """
    stride = codec.stride
    if spans is None:
        spans = [repeat_spans(p, tol=tol) for p in programs]
    elif len(spans) != len(programs):
        raise ValueError(f"{len(spans)} span lists for {len(programs)} programs")
    later_bits = later_n = first_bits = first_n = 0.0
    n_with_repeat = 0

    for program_index, bits in _symbol_bits(
        model, programs, codec, device, max_len, batch_size
    ):
        if not spans[program_index]:
            continue
        n_with_repeat += 1
        for head, rest in spans[program_index]:
            total, count = _span_bits(bits, head, stride)
            first_bits, first_n = first_bits + total, first_n + count
            total, count = _span_bits(bits, rest, stride)
            later_bits, later_n = later_bits + total, later_n + count

    first_rate = first_bits / first_n if first_n else float("nan")
    later_rate = later_bits / later_n if later_n else float("nan")
    return {
        "n": len(programs),
        "n_with_repeat": n_with_repeat,
        "bits_per_symbol_first": first_rate,
        "bits_per_symbol_later": later_rate,
        "recovery": 1.0 - later_rate / first_rate if first_rate else float("nan"),
    }


def recovery_by_copy(
    model,
    programs: list[bytes],
    codec: Codec,
    device: str | torch.device = "cpu",
    max_len: int = 2048,
    batch_size: int = 32,
    tol: int = 0,
    copies: list[list[list[slice]]] | None = None,
) -> dict:
    """The same rate, resolved by copy ordinal -- the length-generalisation test.

    `recovery` answers "are later copies cheaper?". This answers "does that stay
    true past the copy count the model was trained on?", which is the question
    RoPE was chosen for (`dm/models/transformer.py`): train on `REPEAT n <= 4`,
    evaluate on `n = 16`, and copies 5 onward sit at sequence positions no
    training program ever reached.

    Read the *shape*, not one number. A model that learnt the rule has a curve
    that is flat in the ordinal; one that learnt a table of short repeats has a
    curve that turns up somewhere near the trained bound. The two are
    distinguishable at a glance and are not distinguishable in `recovery`, which
    averages them together.

    **Measured 2026-08-07, and the scalar was not merely weaker -- it was wrong
    in sign.** Trained `n <= 4`, evaluated `n = 8..16`: `recovery` read −0.0502,
    i.e. "this model recovered nothing", while the curve showed copies 2-4 at
    1.00-1.18 bits/symbol against a 2.24 reference (about half the cost gone) and
    copies 5+ climbing to 5.95. The scalar would have retracted a real finding
    and hidden a sharper one.

    **Read it beside the absolute *position* each copy sits at**, which this does
    not report and the caller can compute from `repeat_copies`. The break at copy
    5 happened where only 16.7% of spans were past the trained maximum length, so
    what failed to extrapolate is the **count**, not the position -- a
    distinction the ordinal alone cannot draw, and the one that decides whether
    the fix is the training distribution or the architecture.

    The reported ordinals are 1-indexed to match how a repeat reads: copy 1 is
    the body the others are copies *of*, so its rate is the reference and it is
    included.

    `copies` overrides detection, for the reason `recovery` gives: a transformed
    copy shares no bytes with its original, so the translational matcher finds
    nothing on a corpus built out of mirrors.
    """
    stride = codec.stride
    if copies is None:
        per_program = [repeat_copies(p, tol=tol) for p in programs]
    elif len(copies) != len(programs):
        raise ValueError(f"{len(copies)} copy lists for {len(programs)} programs")
    else:
        per_program = copies
    totals: dict[int, list[float]] = {}

    for program_index, bits in _symbol_bits(
        model, programs, codec, device, max_len, batch_size
    ):
        for group in per_program[program_index]:
            for ordinal, span in enumerate(group, start=1):
                total, count = _span_bits(bits, span, stride)
                if not count:
                    continue
                entry = totals.setdefault(ordinal, [0.0, 0.0])
                entry[0] += total
                entry[1] += count

    ordinals = sorted(totals)
    return {
        "n": len(programs),
        "copies": ordinals,
        "bits_per_symbol": [totals[k][0] / totals[k][1] for k in ordinals],
        "symbols": [int(totals[k][1]) for k in ordinals],
    }
