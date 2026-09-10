"""Where an arm's bits actually go, by ISA field — for every arm and codec.

`dm/eval/redundancy.py` asked one arm (the planner's stroke decoder) one
question (is it paying for what its summary already pinned) in one alphabet
(`byte`, because its feasible masks are over byte *values*). `docs/direction.md`
§7.2 asks for the generalisation, and names why: what has produced results in
this project is **instruments that read a model against the ISA's own
structure** — `recovery` against a constructed ceiling, `redundancy` against
exact feasibility rules, `spelling` against position classes, the orbit oracle
against provenance, the library oracle against the `CALL` site. Every one was
hours of work, needed no training, and changed a conclusion.

This is the next member. It reports two things per field, in one forward pass.

**1. Where the budget goes.** Every byte of a program belongs to exactly one
opcode, one operand `Kind`, and one slot within its instruction, so the model's
cost decomposes three ways and each way partitions the whole. Bits per *drawing*
is the headline because that is the unit this project argues in; bits per *byte*
is beside it because a field can be expensive per byte and rare.

**2. What the alphabet has failed to give away, which is the interesting one.**
At an opcode position only 14 of 256 byte values are legal; at an `XF` operand
only 8 are. The bits a model puts on symbols that **cannot occur there** are
bits it is spending to not know the ISA's grid — and they are precisely the
upper bound on what an architectural prior that supplied the grid could
recover. §7.2 proposes a convolutional front end so that "an instruction is 1–4
bytes" is architecture rather than something inferred from position. *This
measures what that would be worth before it is built*, which is
`docs/traps.md`'s rule for opcodes (measure the redundancy first) applied to an
architecture.

**Every codec, and that is the generalisation that mattered.** Feasibility masks
over byte values only make sense for a stride-1 untyped alphabet, which is why
`redundancy.py` refuses the other three. *Attribution* has no such limit: a
bytecode byte is `codec.stride` symbols in any alphabet, and the legal-symbol
set at a position is a property of the codec that every codec can state. So the
same decomposition reads on `byte`, `token`, `token_typed` and `bit` — and the
comparison between them is legal because they encode **identical bytecode**, so
bits/drawing per field holds content fixed and varies only the alphabet.

> **Two units, and only one of them crosses a codec.** Bits per *drawing* and
> per *byte* are comparable across alphabets, because every alphabet spells the
> same bytes. Bits per *symbol* is not: the `bit` codec has eight symbols per
> byte, so its per-symbol rate is a different denominator wearing the same name.
> `docs/traps.md` carries the rule; this module reports per-symbol only inside
> an arm, and never as a headline.

**The reconciliation is what makes it believable.** All three cuts partition
every scored byte, so all three must sum to the same total, and that total must
be the arm's own `bits_per_drawing`. `report` prints the residual; a
decomposition that does not close is a decomposition of something else.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np
import torch
import torch.nn.functional as F

from ..data.dataset import ProgramDataset, collate
from ..isa.codec import (
    N_SPECIAL,
    PAD,
    BitCodec,
    ByteCodec,
    Codec,
    RelativeCodec,
    TokenCodec,
)
from ..isa.spec import SPECS, Kind, UnknownOpcode, spec_for

#: Operand `Kind`s split by axis where the ISA pairs them. `COORD` and `DELTA`
#: arrive two at a time in every instruction that has them (`dm/data/augment.py`
#: relies on the same pairing), and keeping x apart from y is the whole point of
#: the second slot: **y is predicted after x and can use it**. A single `coord`
#: row would average that away, and it is the one number in this file that bears
#: directly on whether within-instruction locality is already being exploited.
FIELDS: tuple[str, ...] = (
    "opcode",
    "coord_x", "coord_y",
    "delta_x", "delta_y",
    "count", "scalar", "id", "xf",
    "unparsed",
)

#: The longest instruction's operand count (`CURVE`), so a slot cut has a fixed
#: width and a report can be printed without discovering its own columns.
MAX_OPERANDS = max(len(spec.operands) for spec in SPECS.values())

SLOTS: tuple[str, ...] = ("op", *(f"operand{i}" for i in range(MAX_OPERANDS)), "unparsed")

OPCODES: tuple[str, ...] = (*(spec.mnemonic for spec in SPECS.values()), "unparsed")

CUTS: tuple[str, ...] = ("field", "slot", "opcode")

#: A fifth cut that exists only where a byte is more than one symbol. Under the
#: `bit` alphabet the legal-symbol reading below is vacuous -- both symbols are
#: always legal, so a bit model cannot waste mass on an impossible one -- and
#: that is exactly the alphabet where "an instruction is 1-4 bytes" is hardest
#: to know. What can be seen instead is *which bit of which field* costs: a
#: coordinate whose position the model has roughly inferred is cheap in its high
#: bits and expensive in its low ones, and a model that has not found the byte
#: grid at all is flat across the eight. `field[k]` with `k` MSB-first, so the
#: curve reads left to right the way the byte does.
BITPLANE = "bitplane"

_KIND_FIELD = {
    Kind.COUNT: "count",
    Kind.SCALAR: "scalar",
    Kind.ID: "id",
    Kind.XF: "xf",
}


# ---------------------------------------------------------------------------
# the walk: every byte gets one label per cut


def labels(program: bytes) -> dict[str, np.ndarray]:
    """One integer label per byte, per cut. Each cut partitions the program.

    The walk is the ISA's own (`spec_for`, `spec.size`), so a `CIRCLE` radius
    lands on `scalar` and a `REPEAT` step on `delta_*` rather than both being
    read as coordinates — the same by-`Kind` discipline `dm/data/augment.py`
    and `dm/eval/redundancy.py` follow, and for the same reason: treating every
    operand as a position corrupts three of them at once.

    Stops where `VM.run` stops. An unknown opcode or a truncated final
    instruction ends the walk and every byte past it is `unparsed` in all three
    cuts — named rather than dropped, because a corpus with any is a corpus this
    decomposition is only partly describing, and the count has to be visible.
    """
    n = len(program)
    field = np.full(n, FIELDS.index("unparsed"), dtype=np.int16)
    slot = np.full(n, SLOTS.index("unparsed"), dtype=np.int16)
    opcode = np.full(n, OPCODES.index("unparsed"), dtype=np.int16)

    pc = 0
    while pc < n:
        try:
            spec = spec_for(program[pc])
        except UnknownOpcode:
            break
        if pc + spec.size > n:
            break
        mnemonic = OPCODES.index(spec.mnemonic)
        opcode[pc : pc + spec.size] = mnemonic
        field[pc] = FIELDS.index("opcode")
        slot[pc] = SLOTS.index("op")

        index = 0
        while index < len(spec.operands):
            kind = spec.operands[index]
            here = pc + 1 + index
            if kind in (Kind.COORD, Kind.DELTA):
                stem = "coord" if kind is Kind.COORD else "delta"
                field[here] = FIELDS.index(f"{stem}_x")
                field[here + 1] = FIELDS.index(f"{stem}_y")
                slot[here] = SLOTS.index(f"operand{index}")
                slot[here + 1] = SLOTS.index(f"operand{index + 1}")
                index += 2
            else:
                field[here] = FIELDS.index(_KIND_FIELD[kind])
                slot[here] = SLOTS.index(f"operand{index}")
                index += 1
        pc += spec.size
    return {"field": field, "slot": slot, "opcode": opcode}


VOCABULARY: dict[str, tuple[str, ...]] = {
    "field": FIELDS, "slot": SLOTS, "opcode": OPCODES,
}


# ---------------------------------------------------------------------------
# what could legally have been emitted here, in this codec's own alphabet


#: Position classes the legal-symbol sets are indexed by. Small and fixed, so
#: the mass on each can be computed once per batch with an index-select rather
#: than by materialising a `(positions, vocabulary)` mask -- which for
#: `token_typed` at `max_len` 3072 would be larger than the logits.
POSITION_CLASSES: tuple[str, ...] = (
    "opcode", "coord", "delta", "count", "scalar", "id", "xf", "free",
)

_CLASS_OF_FIELD = {
    "opcode": "opcode",
    "coord_x": "coord", "coord_y": "coord",
    "delta_x": "delta", "delta_y": "delta",
    "count": "count", "scalar": "scalar", "id": "id", "xf": "xf",
    "unparsed": "free",
}

_CLASS_KIND = {
    "coord": Kind.COORD, "delta": Kind.DELTA, "count": Kind.COUNT,
    "scalar": Kind.SCALAR, "id": Kind.ID, "xf": Kind.XF,
}

#: Values an operand of each `Kind` can actually hold, from `encode_operand`'s
#: own refusals. `XF` is the only one narrower than a byte -- eight elements of
#: D4 -- and it is the field where an alphabet has the most to give away, so
#: getting it from the ISA rather than assuming 256 is not pedantry.
_KIND_VALUES: dict[Kind, np.ndarray] = {
    kind: np.arange(8 if kind is Kind.XF else 256, dtype=np.int64) for kind in Kind
}


def legal_symbols(codec: Codec) -> dict[str, np.ndarray]:
    """Symbol ids that can occur at each position class, under this codec.

    **This is where the alphabets stop being interchangeable, and stating it per
    codec is the point of the module.** `byte` spells an opcode and an operand
    value with the same 256 symbols, so only the ISA's opcode table narrows an
    opcode position. `token` puts opcodes in a disjoint region, so the narrowing
    is free and the model cannot spend anything there. `token_typed` narrows
    every operand position too. `bit` has two symbols and narrows nothing, which
    is exactly the property that makes byte boundaries something it has to
    discover.

    `PAD` and `BOS` are illegal everywhere, in every alphabet: they are model
    controls and never bytecode (`docs/traps.md`), so mass on them is waste of
    the same kind and is counted with it.
    """
    inner = codec.inner if isinstance(codec, RelativeCodec) else codec
    out: dict[str, np.ndarray] = {}

    if isinstance(inner, BitCodec):
        every = np.array([N_SPECIAL, N_SPECIAL + 1], dtype=np.int64)
        return dict.fromkeys(POSITION_CLASSES, every)

    if isinstance(inner, ByteCodec):
        opcodes = np.array(sorted(int(op) for op in SPECS), dtype=np.int64)
        out["opcode"] = N_SPECIAL + opcodes
        for name, kind in _CLASS_KIND.items():
            out[name] = N_SPECIAL + _KIND_VALUES[kind]
        out["free"] = N_SPECIAL + np.arange(256, dtype=np.int64)
        return out

    if isinstance(inner, TokenCodec):
        # Only the slots an opcode actually occupies. A reserved slot is not a
        # bytecode byte -- `decode` drops it -- so a model spending mass there is
        # wasting it exactly as it would on PAD.
        used = min(len(SPECS), inner.opcode_slots)
        out["opcode"] = inner._op_base + np.arange(used, dtype=np.int64)
        for name, kind in _CLASS_KIND.items():
            base = inner._val_base + (256 * int(kind) if inner.typed_operands else 0)
            out[name] = base + _KIND_VALUES[kind]
        out["free"] = np.arange(inner._op_base, inner.vocab_size, dtype=np.int64)
        return out

    raise TypeError(f"no legal-symbol rule for {type(inner).__name__}")


def _class_ids(field: np.ndarray) -> np.ndarray:
    lookup = np.array(
        [POSITION_CLASSES.index(_CLASS_OF_FIELD[name]) for name in FIELDS],
        dtype=np.int16,
    )
    return lookup[field]


# ---------------------------------------------------------------------------
# the pass


@torch.no_grad()
def _score(
    model,
    programs: list[bytes],
    codec: Codec,
    device: str | torch.device,
    max_len: int,
    batch_size: int,
) -> Iterator[tuple[int, bytes, np.ndarray, np.ndarray]]:
    """Yield `(program index, the bytes scored, bits per position, waste per position)`.

    The BOS shift is `dm.eval.recovery._symbol_bits`'s, restated rather than
    imported because this pass also needs the *distribution* and not only the
    target's own cost: `with_bos` prepends BOS and `collate` shifts targets left
    by one, so bytecode byte `b` lands on target positions
    `[b * stride, (b + 1) * stride)`.

    Waste is computed inside the batch, never returned as logits. A
    `(batch, positions, vocabulary)` tensor for `token_typed` at `max_len` 3072
    is 600 MB, and a second copy of it to mask would be the difference between
    this running and this swapping.
    """
    sets = legal_symbols(codec)
    order = [torch.from_numpy(sets[name]) for name in POSITION_CLASSES]
    dataset = ProgramDataset(programs, codec, max_len)
    classes = {
        i: np.repeat(_class_ids(labels(programs[i])["field"]), codec.stride)
        for i in range(len(programs))
    }
    model.eval()
    for start in range(0, len(dataset), batch_size):
        batch = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
        index, inputs, targets = collate(batch)
        inputs, targets = inputs.to(device), targets.to(device)
        logprobs = F.log_softmax(model(inputs).float(), dim=-1)
        nll = F.cross_entropy(
            logprobs.reshape(-1, codec.vocab_size), targets.reshape(-1),
            ignore_index=PAD, reduction="none",
        ).view(targets.shape).cpu().numpy() / math.log(2)
        # Mass on the legal set, one column per position class. Index-select
        # rather than mask: the widest class is 256 symbols where the vocabulary
        # is up to 1,554.
        mass = torch.stack(
            [torch.logsumexp(logprobs[..., ids.to(device)], dim=-1) for ids in order],
            dim=-1,
        ).cpu().numpy() / math.log(2)
        del logprobs
        for row, program_index in enumerate(index.tolist()):
            ids = classes[program_index][: nll.shape[1]]
            waste = -mass[row, : len(ids), :][np.arange(len(ids)), ids]
            yield program_index, programs[program_index], nll[row], waste


@torch.no_grad()
def _score_planner(
    planner,
    programs: list[bytes],
    codec: Codec,
    device: str | torch.device,
    batch_size: int,
) -> Iterator[tuple[int, bytes, np.ndarray, np.ndarray]]:
    """The same rows off the planner's stroke decoder, one row per stroke.

    Reusing `dm.eval.redundancy.planner_rows` rather than re-deriving the
    prompt layout: a row is `BOS`, the six summary symbols, then the stroke, and
    position `given + j` predicts stroke byte `j`. That alignment is exactly what
    a per-field attribution depends on -- one position out and every byte's bits
    are credited to its neighbour, which on the `coord_x`/`coord_y` split is the
    difference between the finding and its mirror image.

    The planner's decoder is scored on *strokes*, so a row's byte string is a
    stroke rather than a program. `labels` walks it the same way: a stroke opens
    with `MOVE` and the last one closes with `HALT`, both of which the ISA walk
    reads without knowing it is looking at a fragment.
    """
    from .redundancy import planner_rows

    sets = legal_symbols(codec)
    order = [torch.from_numpy(sets[name]) for name in POSITION_CLASSES]
    planner.eval()
    for index, stroke, _summary, logprobs in planner_rows(
        planner, programs, codec, device, batch_size
    ):
        scored = logprobs.shape[0]
        if not scored:
            continue
        symbols = codec.encode(stroke)[:scored]
        nll = -logprobs[np.arange(len(symbols)), symbols].double().numpy() / math.log(2)
        mass = torch.stack(
            [torch.logsumexp(logprobs[..., ids], dim=-1) for ids in order], dim=-1
        ).double().numpy() / math.log(2)
        ids = np.repeat(_class_ids(labels(stroke)["field"]), codec.stride)[:scored]
        waste = -mass[: len(ids), :][np.arange(len(ids)), ids]
        yield index, stroke, nll[: len(ids)], waste


def _bitplane_vocabulary(stride: int) -> tuple[str, ...]:
    return tuple(f"{name}[{k}]" for name in FIELDS for k in range(stride))


def reduce_rows(rows, n: int, codec: Codec, kind: str = "flat") -> dict:
    """Fold a row source into per-field totals.

    **Takes the generator rather than a model**, so the flat arm and the
    planner's stroke decoder run through *identical* arithmetic --
    `dm/eval/redundancy.py` makes the same choice for the same reason: two
    copies of a reduction, one per arm, is how a comparison ends up measuring
    its own two implementations.
    """
    stride = codec.stride
    vocabulary = dict(VOCABULARY)
    cuts_here = list(CUTS)
    if stride > 1:
        vocabulary[BITPLANE] = _bitplane_vocabulary(stride)
        cuts_here.append(BITPLANE)
    totals = {
        cut: {
            "bits": np.zeros(len(vocabulary[cut]), dtype=np.float64),
            "waste": np.zeros(len(vocabulary[cut]), dtype=np.float64),
            "symbols": np.zeros(len(vocabulary[cut]), dtype=np.int64),
        }
        for cut in cuts_here
    }
    per_program = np.zeros(n, dtype=np.float64)
    scored_bytes = 0
    truncated = 0

    for index, scored_text, bits, waste in rows:
        cuts = labels(scored_text)
        width = min(len(bits), len(waste))
        per_program[index] += float(bits[:width].sum())
        scored_bytes += width // stride
        truncated += width < len(scored_text) * stride
        for cut in cuts_here:
            if cut == BITPLANE:
                # `field * stride + k`, so `_bitplane_vocabulary` lists the eight
                # planes of one field consecutively and the reader gets a curve
                # rather than a scatter.
                symbol_labels = (np.repeat(cuts["field"], stride) * stride
                                 + np.tile(np.arange(stride, dtype=np.int16),
                                           len(cuts["field"])))[:width]
            else:
                symbol_labels = np.repeat(cuts[cut], stride)[:width]
            size = len(vocabulary[cut])
            totals[cut]["bits"] += np.bincount(symbol_labels, weights=bits[:width],
                                               minlength=size)
            totals[cut]["waste"] += np.bincount(symbol_labels, weights=waste[:width],
                                                minlength=size)
            totals[cut]["symbols"] += np.bincount(symbol_labels, minlength=size)

    out: dict = {
        "n": n,
        "kind": kind,
        "codec": codec.name,
        "stride": stride,
        "bits_per_drawing": float(per_program.sum()) / max(1, n),
        "bytes_per_drawing": scored_bytes / max(1, n),
        "truncated": truncated,
        "per_program_bits": [round(float(b), 4) for b in per_program],
        "cuts": {},
    }
    for cut in cuts_here:
        rows = {}
        for i, name in enumerate(vocabulary[cut]):
            symbols = int(totals[cut]["symbols"][i])
            if not symbols:
                continue
            rows[name] = {
                "bits_per_drawing": totals[cut]["bits"][i] / max(1, n),
                "waste_per_drawing": totals[cut]["waste"][i] / max(1, n),
                "bytes_per_drawing": symbols / stride / max(1, n),
                "bits_per_byte": totals[cut]["bits"][i] / (symbols / stride),
                "waste_per_byte": totals[cut]["waste"][i] / (symbols / stride),
                # Inside an arm only. Eight symbols per byte on the bit codec
                # makes this a different denominator wearing the same name, so
                # it never leaves this dictionary as a cross-codec column.
                "bits_per_symbol": totals[cut]["bits"][i] / symbols,
                "symbols": symbols,
                "share": float(totals[cut]["bits"][i] / max(1e-12,
                                                            totals[cut]["bits"].sum())),
            }
        out["cuts"][cut] = rows
        out[f"{cut}_total_bits_per_drawing"] = float(totals[cut]["bits"].sum()) / max(1, n)
        out[f"{cut}_total_waste_per_drawing"] = float(totals[cut]["waste"].sum()) / max(1, n)
    return out


def attribute(
    model,
    programs: list[bytes],
    codec: Codec,
    device: str | torch.device = "cpu",
    max_len: int = 2048,
    batch_size: int = 16,
) -> dict:
    """A flat AR arm, one forward pass over the split."""
    return reduce_rows(
        _score(model, programs, codec, device, max_len, batch_size),
        len(programs), codec, kind="flat",
    )


def attribute_planner(
    planner,
    programs: list[bytes],
    codec: Codec,
    device: str | torch.device = "cpu",
    batch_size: int = 32,
) -> dict:
    """The planner's stroke decoder, over the same split and the same reduction.

    **The two are not differenced without saying what differs.** This scores only
    the stroke half of the factorisation -- the composition level transmits the
    summary grid and pays for it elsewhere -- so a planner's field totals are not
    an arm's `bits/drawing` and do not reconcile against one. What they are
    comparable with is the *flat* arm's totals on the same bytes, which is what
    `dm/eval/redundancy.py`'s control already established as the legal reading.
    """
    return reduce_rows(
        _score_planner(planner, programs, codec, device, batch_size),
        len(programs), codec, kind="planner",
    )


def reconcile(reading: dict, reference: float | None = None) -> dict:
    """Do the three cuts agree with each other, and with the arm's own record?

    Reported rather than asserted, for `dm/eval/spelling.py`'s reason: a
    residual is how a reader finds out that `max_len` truncated one arm, where a
    silently rebalanced split would simply look correct.
    """
    totals = {cut: reading[f"{cut}_total_bits_per_drawing"] for cut in CUTS}
    spread = max(totals.values()) - min(totals.values())
    out = {"cut_totals": totals, "cut_spread": spread,
           "bits_per_drawing": reading["bits_per_drawing"]}
    if reference is not None:
        out["record_bits_per_drawing"] = reference
        out["residual"] = reading["bits_per_drawing"] - reference
    return out


__all__ = [
    "BITPLANE",
    "CUTS",
    "FIELDS",
    "MAX_OPERANDS",
    "OPCODES",
    "POSITION_CLASSES",
    "SLOTS",
    "VOCABULARY",
    "attribute",
    "attribute_planner",
    "labels",
    "legal_symbols",
    "reconcile",
    "reduce_rows",
]
