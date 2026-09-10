"""What the stroke decoder pays for information the summary already gave it.

The planner factorises as `p(s) * p(x | s)` evaluated at `s = f(x)`, and the
summary is a **deterministic function of the stroke** (`dm/isa/strokes.py`) --
that determinism is what makes the bound a bound. So the scheme transmits the
summary's information twice: once explicitly as `p(s)`, and once implicitly
inside `p(x | s)`, because a decoder that has been told the first point, the
extent, the length and whether this stroke ends the drawing should not spend
bits re-specifying them.

**If it does spend them, claim 3's 39-49 bit loss is a coding inefficiency in
this implementation of the factorisation rather than a verdict on
factorisation.** That is the whole question, and either answer is a result: a
decoder that already exploits its conditioning makes the negative result
*stronger* than the one on record.

The measurement is one forward pass over val and no training.

**Every rule here is exact.** A value is called infeasible only when no program
consistent with this summary could have put it there, so the reported saving is
achievable by renormalising the decoder onto the feasible set and by nothing
cleverer. Anything statistical -- "coordinates near the last one are likelier" --
is a different number and does not belong in this file, because it would make the
answer a property of the corpus rather than of the conditioning.

The rules, one attributable name each:

- **`first_point`** -- the stroke's first `COORD` pair *is* `(x0, y0)`, byte for
  byte, since `Summary.of` reads `xs[0], ys[0]`. A singleton feasible set: the
  decoder is being asked to re-transmit a byte it holds.
- **`coord_box`** -- every later coordinate satisfies `|v - x0| <= width`,
  tightened by the prefix's own running extremes to `[max_so_far - width,
  min_so_far + width]`. Exact because `width = max(xs) - min(xs)` with `x0` among
  the `xs`, so a future value outside that window would force the extent past
  what the summary states.
- **`halt`** -- `halts = 0` makes `HALT` impossible at every boundary;
  `halts = 1` with one byte left and no `HALT` yet makes it the only possibility.
- **`fits`** -- `length` pins the byte count and a stroke is a whole number of
  instructions, so a boundary with `r` bytes left admits only opcodes of size
  `<= r`.
- **`isa`** -- unknown opcodes, and every non-byte symbol in the vocabulary.
  **Kept apart from the four above and subtracted before anything is
  attributed to the summary**, because bits reclaimable by knowing the ISA are
  available to a model that was never told the summary, and folding them in
  would answer a different question with a bigger number.

`isa` is a subset relation rather than a separate account: the reported
`summary` saving is `-log2 P(isa and summary rules) + log2 P(isa)`, so the two
compose exactly and neither double-counts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from ..isa.codec import N_SPECIAL, PAD, Codec
from ..isa.spec import CANVAS, SPECS, Kind, Op, UnknownOpcode, spec_for
from ..isa.strokes import Summary, split
from ..models.planner import StrokePlanner, prompt_len, stroke_batch
from ..models.transformer import DrawingLM

#: Rules derived from the summary, in the order the report prints them. `isa` is
#: deliberately not in this tuple: it is the baseline every one of these is read
#: against, not a peer.
SUMMARY_RULES = ("first_point", "coord_box", "halt", "fits")

#: Opcode bytes the ISA defines. Everything else is unparseable, so a decoder
#: that spends mass there is wasting it on a program that cannot exist.
KNOWN_OPCODES = np.zeros(CANVAS, dtype=bool)
KNOWN_OPCODES[[int(op) for op in SPECS]] = True

#: `size_leq[r]` is the opcode set that fits in `r` remaining bytes. Indexed to
#: the longest instruction; past that every opcode fits.
_SIZES = {int(spec.op): spec.size for spec in SPECS.values()}
_MAX_SIZE = max(_SIZES.values())
_SIZE_LEQ = np.array([
    [_SIZES.get(b, _MAX_SIZE + 1) <= r for b in range(CANVAS)]
    for r in range(_MAX_SIZE + 1)
])


class CodecUnsupported(Exception):
    """Raised rather than approximated.

    The masks below are over byte *values*, and only an untyped stride-1
    alphabet maps those onto symbols one-to-one independently of position. Under
    `token` an opcode and an operand with the same byte value are different
    symbols; under `bit` a byte is eight symbols and there is no position where
    a byte-valued distribution exists at all. Returning a number of the right
    shape for those would be the fault this project has found at four levels.
    """


def _check(codec: Codec) -> None:
    if codec.name != "byte":
        raise CodecUnsupported(
            f"{codec.name!r}: the feasible mask is over byte values, which only "
            "the byte alphabet spells one-to-one at every position"
        )


@dataclass
class Positions:
    """One stroke's per-byte feasible sets, plus what each position is.

    `masks[rule]` is `(n_bytes, 256)` and True where *that rule alone* permits
    the value. They are kept apart rather than pre-intersected so the report can
    say which piece of the summary is being ignored, which is the difference
    between "the decoder wastes bits" and "the decoder re-transmits the first
    point".
    """

    masks: dict[str, np.ndarray]
    roles: list[str]                       # "opcode" | "coord" | "operand"
    determined: np.ndarray                 # (n_bytes,) -- feasible set is a singleton
    parsed: int = 0                        # bytes the walk covered
    extra: dict = field(default_factory=dict)


def _box(seen: list[int], extent: int) -> np.ndarray:
    """Values a further coordinate can take given those already emitted.

    `extent` is `max - min` over the *whole* stroke, so a value below
    `max_so_far - extent` or above `min_so_far + extent` would put the final
    extent past what the summary states. Both ends, because the summary bounds
    the spread rather than the position.
    """
    lo = max(0, max(seen) - extent)
    hi = min(CANVAS - 1, min(seen) + extent)
    out = np.zeros(CANVAS, dtype=bool)
    out[lo : hi + 1] = True
    return out


def _one(value: int) -> np.ndarray:
    out = np.zeros(CANVAS, dtype=bool)
    out[value % CANVAS] = True
    return out


def positions(stroke: bytes, summary: Summary) -> Positions:
    """Walk one stroke's parse and record what each byte could have been.

    The walk is the ISA's own (`spec_for`, `spec.size`), so a corpus with
    `REPEAT` bodies or `CURVE` operands lands the coordinate rules on coordinate
    bytes and nowhere else -- `CIRCLE`'s radius is a `SCALAR` and must not be
    read as a position, which is the same distinction `Summary.of` makes when it
    builds the box in the first place.

    Stops where `VM.run` would: an unknown opcode or a truncated final
    instruction ends the walk, and every byte past it carries no constraint. That
    is the conservative direction -- an unparseable tail is scored as free rather
    than as waste.
    """
    n = len(stroke)
    masks = {r: np.ones((n, CANVAS), dtype=bool) for r in (*SUMMARY_RULES, "isa")}
    roles = ["operand"] * n
    determined = np.zeros(n, dtype=bool)
    # What the decoder was *told*. `Summary.length` is u8, so past 255 bytes it
    # cannot state the length at all (`planner.unrepresentable_length`, where
    # the same ceiling breaks the bound's normalisation). When it cannot, the
    # length-dependent rules are **switched off rather than applied to a wrong
    # number**: measured on QuickDraw's val split, 2 strokes of 6,295 are over
    # the ceiling and applying `fits` to their stated 255 excluded the byte that
    # was actually there at 48 positions. A rule that excludes the truth reports
    # infinite savings, so this is the one failure mode that could make the whole
    # measurement lie.
    stated = summary.length
    states_length = stated == n
    seen_x: list[int] = []
    seen_y: list[int] = []
    halted = False
    pc = 0
    while pc < n:
        try:
            spec = spec_for(stroke[pc])
        except UnknownOpcode:
            break
        if pc + spec.size > n:
            break
        roles[pc] = "opcode"
        masks["isa"][pc] = KNOWN_OPCODES
        remaining = stated - pc
        if states_length:
            masks["fits"][pc] = _SIZE_LEQ[min(max(remaining, 0), _MAX_SIZE)]
        if not summary.halts:
            masks["halt"][pc, int(Op.HALT)] = False
        elif states_length and remaining == 1 and not halted:
            # One byte left and the drawing has to end inside this stroke, so
            # there is exactly one instruction that can go here.
            masks["halt"][pc] = _one(int(Op.HALT))
        halted |= spec.op is Op.HALT

        index = 0
        while index < len(spec.operands):
            if spec.operands[index] is Kind.COORD:
                px, py = pc + 1 + index, pc + 2 + index
                roles[px] = roles[py] = "coord"
                if not seen_x:
                    masks["first_point"][px] = _one(summary.x0)
                    masks["first_point"][py] = _one(summary.y0)
                else:
                    masks["coord_box"][px] = _box(seen_x, summary.width)
                    masks["coord_box"][py] = _box(seen_y, summary.height)
                seen_x.append(stroke[px])
                seen_y.append(stroke[py])
                index += 2
            else:
                index += 1
        pc += spec.size

    combined = np.ones((n, CANVAS), dtype=bool)
    for mask in masks.values():
        combined &= mask
    determined = combined.sum(axis=1) == 1
    return Positions(masks=masks, roles=roles, determined=determined, parsed=pc)


def full_vocab(mask: np.ndarray, vocab: int) -> np.ndarray:
    """A byte-value mask as a symbol mask, with PAD and BOS always infeasible.

    The model normalises over its whole vocabulary, so mass it puts on `PAD` or
    `BOS` mid-stroke is waste too. It is folded into `isa` rather than left out,
    because it is exactly the same kind of waste -- probability on a symbol that
    could not have been there -- and it is not something the summary tells you.
    """
    out = np.zeros((mask.shape[0], vocab), dtype=bool)
    out[:, N_SPECIAL : N_SPECIAL + CANVAS] = mask
    return out


def _wasted(logprobs: torch.Tensor, mask: np.ndarray) -> np.ndarray:
    """`-log2 P(feasible)` per position, in bits.

    The bits a coder recovers by renormalising onto the feasible set: the model's
    own conditional, with the impossible values removed and the rest scaled up.
    No retraining and no new parameters, which is what makes this an *achievable*
    saving rather than an estimate of one.
    """
    keep = torch.from_numpy(mask)
    total = torch.logsumexp(logprobs.masked_fill(~keep, -math.inf), dim=-1)
    return (-total / math.log(2)).double().numpy()


@dataclass
class Reading:
    """Per-program totals, in the unit claim 3 is argued in.

    Everything is per program rather than per stroke or per symbol, because the
    number this has to be compared against -- the 39-49 bit gap -- is
    bits/drawing, and a per-symbol saving cannot be differenced against it
    without a stroke count that would itself have to be justified.
    """

    isa: np.ndarray                  # (n_programs,) bits recoverable from the ISA alone
    summary: np.ndarray              # ... and from the summary, on top of the ISA
    per_rule: dict[str, np.ndarray]  # each summary rule's own marginal, over ISA
    determined_bits: np.ndarray      # bits spent at positions with one legal value
    free_bits: np.ndarray            # ... and at every other scored position
    determined_symbols: np.ndarray
    free_symbols: np.ndarray
    strokes: int = 0
    unparsed: int = 0                # strokes whose walk stopped early
    mismatched: int = 0              # strokes whose length the summary cannot state

    def summarise(self, programs: int) -> dict:
        n = max(1, programs)
        out = {
            "isa_bits_per_drawing": float(self.isa.sum()) / n,
            "summary_bits_per_drawing": float(self.summary.sum()) / n,
            "determined_bits_per_symbol": float(
                self.determined_bits.sum() / max(1, self.determined_symbols.sum())),
            "free_bits_per_symbol": float(
                self.free_bits.sum() / max(1, self.free_symbols.sum())),
            "determined_symbols": int(self.determined_symbols.sum()),
            "free_symbols": int(self.free_symbols.sum()),
            "strokes": self.strokes,
            "unparsed_strokes": self.unparsed,
            "length_mismatched_strokes": self.mismatched,
            "n": programs,
        }
        out |= {f"{rule}_bits_per_drawing": float(v.sum()) / n
                for rule, v in self.per_rule.items()}
        # Kept per program so a caller can pair this against another arm the way
        # every bits/drawing comparison in this project is paired.
        out["per_program_summary_bits"] = [round(float(b), 4) for b in self.summary]
        return out


@torch.no_grad()
def planner_rows(planner: StrokePlanner, programs: list[bytes], codec: Codec,
                 device: str | torch.device = "cpu", batch_size: int = 32):
    """`(program, stroke, summary, logprobs)` for every stroke the decoder scores.

    Built on `stroke_batch` rather than beside it, because the alignment between
    a stroke's bytes and the decoder's scored positions is exactly the thing this
    measurement depends on: a row is `BOS`, the six summary symbols, then the
    stroke, and position `given + j` predicts stroke byte `j`. A second copy of
    that layout that drifted would silently attribute one byte's bits to its
    neighbour, which on the `first_point` rule is the difference between the
    answer and its opposite.
    """
    given = prompt_len(codec)
    for start in range(0, len(programs), batch_size):
        chunk = programs[start : start + batch_size]
        batch = stroke_batch(chunk, codec, planner.cfg)
        logits = planner.decoder(batch.inputs.to(device))
        logprobs = F.log_softmax(logits.float(), dim=-1).cpu()
        # The rows are in the same order `stroke_batch` built them, so walking
        # the same split reproduces the pairing without storing it.
        row = 0
        for i, program in enumerate(chunk):
            for stroke in split(program, planner.cfg.max_strokes):
                scored = int((batch.targets[row] != PAD).sum())
                yield (start + i, stroke, Summary.of(stroke),
                       logprobs[row, given : given + scored])
                row += 1


@torch.no_grad()
def flat_rows(model: DrawingLM, programs: list[bytes], codec: Codec,
              max_strokes: int | None = None, max_len: int = 2048,
              device: str | torch.device = "cpu", batch_size: int = 32):
    """The same tuples off a flat AR model, which was never told the summary.

    **This is the control that turns an absolute into a difference.** The
    planner's claim on these bytes is that being handed the summary bought
    something; a model that has to infer the same facts from its own prefix
    gives the number that claim has to beat.

    The asymmetry runs the conservative way: this model sees every *previous*
    stroke of the program and the planner's decoder sees only its own stroke plus
    six bytes. So a planner that does not waste less than this one did not waste
    less despite an advantage, which is the direction that makes a null
    believable.

    A program's row is `BOS` then its bytes, so position `j` predicts byte `j`
    and a stroke that starts at byte `off` occupies `j = off ... off + len - 1`.
    """
    for start in range(0, len(programs), batch_size):
        chunk = programs[start : start + batch_size]
        rows = [codec.with_bos(p)[:max_len] for p in chunk]
        width = max(len(r) for r in rows)
        padded = torch.full((len(rows), width), PAD, dtype=torch.long)
        for i, row in enumerate(rows):
            padded[i, : len(row)] = torch.tensor(row, dtype=torch.long)
        logits = model(padded[:, :-1].to(device))
        logprobs = F.log_softmax(logits.float(), dim=-1).cpu()
        for i, program in enumerate(chunk):
            offset = 0
            for stroke in split(program, max_strokes):
                stop = min(offset + len(stroke), logprobs.shape[1])
                if stop > offset:
                    yield (start + i, stroke[: stop - offset], Summary.of(stroke),
                           logprobs[i, offset:stop])
                offset += len(stroke)


def redundancy(rows, programs: int, vocab: int) -> Reading:
    """Reduce a row source into per-program totals.

    Takes the generator rather than a model so the planner arm and the flat
    control run through **identical** arithmetic. Two copies of this reduction,
    one per arm, is how a comparison ends up measuring its own two
    implementations.
    """
    def zeros() -> np.ndarray:
        return np.zeros(programs, dtype=np.float64)

    reading = Reading(
        isa=zeros(), summary=zeros(),
        per_rule={rule: zeros() for rule in SUMMARY_RULES},
        determined_bits=zeros(), free_bits=zeros(),
        determined_symbols=np.zeros(programs, dtype=np.int64),
        free_symbols=np.zeros(programs, dtype=np.int64),
    )
    for index, stroke, summary, logprobs in rows:
        scored = min(len(stroke), logprobs.shape[0])
        if not scored:
            continue
        reading.strokes += 1
        reading.mismatched += summary.length != len(stroke)
        where = positions(stroke, summary)
        reading.unparsed += where.parsed < len(stroke)

        logprobs = logprobs[:scored]
        isa = full_vocab(where.masks["isa"][:scored], vocab)
        isa_bits = _wasted(logprobs, isa)
        reading.isa[index] += isa_bits.sum()

        both = isa.copy()
        for rule in SUMMARY_RULES:
            rule_mask = isa & full_vocab(where.masks[rule][:scored], vocab)
            reading.per_rule[rule][index] += (_wasted(logprobs, rule_mask)
                                              - isa_bits).sum()
            both &= rule_mask
        # Over the ISA baseline, so nothing the ISA already ruled out is
        # attributed to the conditioning.
        reading.summary[index] += (_wasted(logprobs, both) - isa_bits).sum()

        # The partition reading (`docs/direction.md` 3.2), on the same pass:
        # what the decoder actually spends where it has been left no choice.
        symbols = torch.tensor([N_SPECIAL + b for b in stroke[:scored]])
        spent = -logprobs.gather(1, symbols[:, None]).squeeze(1).double().numpy() / math.log(2)
        pinned = where.determined[:scored]
        reading.determined_bits[index] += spent[pinned].sum()
        reading.free_bits[index] += spent[~pinned].sum()
        reading.determined_symbols[index] += int(pinned.sum())
        reading.free_symbols[index] += int((~pinned).sum())
    return reading


def planner_redundancy(planner: StrokePlanner, programs: list[bytes], codec: Codec,
                       device: str | torch.device = "cpu",
                       batch_size: int = 32) -> dict:
    _check(codec)
    planner.eval()
    rows = planner_rows(planner, programs, codec, device, batch_size)
    return redundancy(rows, len(programs), codec.vocab_size).summarise(len(programs))


def flat_redundancy(model: DrawingLM, programs: list[bytes], codec: Codec,
                    max_strokes: int | None = None, max_len: int = 2048,
                    device: str | torch.device = "cpu",
                    batch_size: int = 32) -> dict:
    _check(codec)
    model.eval()
    rows = flat_rows(model, programs, codec, max_strokes, max_len, device, batch_size)
    return redundancy(rows, len(programs), codec.vocab_size).summarise(len(programs))
