"""Where a drawing's bits go, under two spellings of the same program.

`REPEATX` is a compression feature, and P5 measured what it compresses: the
constructed corpus is **40.9% shorter** in the structured spelling than in the
flat one, and the two decode to the identical geometry. The obvious next number
is the difference in `bits/drawing` between a model trained on each spelling --
and taken bare that number is not the fold's, because the two arms differ in
their codec and in their sequence length as well as in whether the copies are
present.

**This splits it.** A scene's two spellings share most of their bytes *exactly*:

    flat        prefix | motif | copies 2..n |          suffix | HALT
    structured  prefix | REPEATX | motif | ENDREP | suffix | HALT

`prefix`, `motif`, `suffix` and `HALT` are byte-identical in both -- the same
bytes in the same order, at shifted positions -- so differencing the model's cost
on them holds content fixed and varies only the spelling's *context*. What is
left is the fold itself, and it has two signs:

- **`copies`** -- bytes only the flat spelling contains. What the flat arm spends
  here is the most the fold could ever save, measured on the arm that pays it
  rather than assumed from the byte count.
- **`header`** -- `REPEATX` and `ENDREP`, bytes only the structured spelling
  contains. What the fold costs.

So `Δ total = copies − header + context`, and only the first two terms belong to
the ISA. `context` is where the codec confound lives, which is why it is reported
as its own line instead of being folded into the answer: a reader who wants the
fold's value reads two rows and a reader who wants the total reads three.

**The reconciliation is the reason any of it can be believed.** The classes
partition every byte of every program, so their bits must add up to the record's
own `bits_per_drawing`. `report` prints both and their residual; a decomposition
that does not close is a decomposition of something else.
"""

from __future__ import annotations

import torch

from ..data.composed import Scene
from ..isa.codec import Codec
from ..isa.spec import SPECS, Op
from .recovery import _span_bits, _symbol_bits

#: Classes in reading order, and which spelling carries each. `both` classes are
#: byte-identical across the two, which is what makes their difference a
#: measurement of context rather than of content.
CLASSES: dict[str, str] = {
    "prefix": "both",
    "header": "structured",
    "motif": "both",
    "copies": "flat",
    "suffix": "both",
    "halt": "both",
}

#: The classes whose difference is the fold, and the sign each carries in it.
#: `copies` is what folding removes and `header` is what it adds, so the fold's
#: value is `copies − header` and nothing else in this file needs to know which
#: way round that goes.
FOLD_SIGN: dict[str, int] = {"copies": +1, "header": -1}

Spans = dict[str, tuple[slice, ...]]


def _halt_at(program: bytes) -> int:
    """Index of the program's single trailing `HALT`.

    Asserted rather than assumed: `dm.data.composed` emits exactly one and the
    suffix span is defined as everything before it, so a program that ended
    otherwise would silently move a byte into the wrong class.
    """
    if not program or program[-1] != int(Op.HALT):
        raise ValueError("a composed scene ends in HALT; this program does not")
    return len(program) - 1


def flat_spans(scene: Scene) -> Spans:
    """Byte classes of the flat trace: the copies are present and unmarked."""
    start = scene.copies[0].start
    body = scene.copies[0].stop - start
    end = scene.copies[-1].stop
    halt = _halt_at(scene.flat)
    return {
        "prefix": (slice(0, start),),
        "motif": (slice(start, start + body),),
        "copies": (slice(start + body, end),),
        "suffix": (slice(end, halt),),
        "halt": (slice(halt, halt + 1),),
    }


def structured_spans(scene: Scene) -> Spans:
    """Byte classes of the `REPEATX` form, from the ISA's own instruction sizes.

    The two header bytes-spans are one class: `REPEATX` opens the block and
    `ENDREP` closes it, they exist only because the fold does, and splitting them
    would invite a reader to price half a bracket.
    """
    start = scene.copies[0].start
    body = scene.copies[0].stop - start
    opener, closer = SPECS[Op.REPEATX].size, SPECS[Op.ENDREP].size
    halt = _halt_at(scene.structured)
    return {
        "prefix": (slice(0, start),),
        "header": (slice(start, start + opener),
                   slice(start + opener + body, start + opener + body + closer)),
        "motif": (slice(start + opener, start + opener + body),),
        "suffix": (slice(start + opener + body + closer, halt),),
        "halt": (slice(halt, halt + 1),),
    }


def check(scene: Scene) -> None:
    """Refuse a scene whose two spellings do not line up, byte for byte.

    Three properties, and every number in this file is wrong if any of them
    fails, which is why they are checked per scene rather than in a test alone:

    1. **Each spelling's classes partition it** -- every byte in exactly one
       class, no gap and no overlap. A gap silently loses bits and the
       reconciliation against `bits_per_drawing` is what would notice, later.
    2. **The `both` classes hold identical bytes.** This is the whole design: the
       difference on them is the cost of the *context*, so if the content moved
       too the number means nothing.
    3. **The fold's arithmetic closes** -- `copies` minus `header` is exactly the
       `saved` the generator recorded, so the byte-level ceiling and the
       bit-level decomposition are describing one fold.
    """
    for text, spans in ((scene.flat, flat_spans(scene)),
                        (scene.structured, structured_spans(scene))):
        covered = sorted((s.start, s.stop) for group in spans.values()
                         for s in group if s.stop > s.start)
        at = 0
        for lo, hi in covered:
            if lo != at:
                raise ValueError(f"classes do not partition the program at {lo}")
            at = hi
        if at != len(text):
            raise ValueError(f"classes cover {at} of {len(text)} bytes")

    flat, structured = flat_spans(scene), structured_spans(scene)
    for name, side in CLASSES.items():
        if side != "both":
            continue
        if scene.flat[flat[name][0]] != scene.structured[structured[name][0]]:
            raise ValueError(f"{name} differs between the two spellings")

    copies = sum(s.stop - s.start for s in flat["copies"])
    header = sum(s.stop - s.start for s in structured["header"])
    if copies - header != scene.saved:
        raise ValueError(f"the fold saves {copies - header} bytes, not {scene.saved}")


def class_bits(
    model,
    programs: list[bytes],
    codec: Codec,
    spans: list[Spans],
    device: str | torch.device = "cpu",
    max_len: int = 2048,
    batch_size: int = 32,
) -> dict:
    """Total bits and symbols per class, over one forward pass of the split.

    One pass, reusing `dm.eval.recovery`'s scorer rather than a second copy of
    the BOS-shift reasoning -- the identity that maps a byte span onto symbol
    positions is subtle enough that two implementations of it would eventually
    disagree, and `recovery` is where it is argued.

    Per drawing rather than per symbol as the headline, because the quantity
    under test is the cost of transmitting one drawing and a per-symbol rate
    would divide by a length the spelling is *changing*. Both are returned.
    """
    if len(spans) != len(programs):
        raise ValueError(f"{len(spans)} span sets for {len(programs)} programs")
    totals: dict[str, list[float]] = {name: [0.0, 0.0] for name in CLASSES}

    for index, bits in _symbol_bits(model, programs, codec, device, max_len, batch_size):
        for name, group in spans[index].items():
            for span in group:
                total, count = _span_bits(bits, span, codec.stride)
                totals[name][0] += total
                totals[name][1] += count

    n = len(programs)
    return {
        "n": n,
        # The codec's stride, so a caller can read the byte count back out of a
        # symbol count without knowing which alphabet this side used -- the whole
        # comparison is between two spellings, and a length in symbols is not a
        # length in bytes for the `bit` codec.
        "stride": codec.stride,
        "classes": {
            name: {
                "bits": total,
                "symbols": int(count),
                "bytes_per_drawing": count / codec.stride / n,
                "per_drawing": total / n,
                "bits_per_symbol": total / count if count else 0.0,
            }
            for name, (total, count) in totals.items()
        },
        "bits_per_drawing": sum(t for t, _ in totals.values()) / n,
        "bytes_per_drawing": sum(c for _, c in totals.values()) / codec.stride / n,
    }


def delta(flat: dict, structured: dict) -> dict:
    """The flat spelling's cost minus the structured one's, split three ways.

    Positive means the flat spelling costs more, i.e. the fold pays. The three
    terms are the point:

    - `fold` -- `copies` (flat only) minus `header` (structured only). **This is
      the ISA's term and the only one it can claim.**
    - `context` -- the shared classes, whose bytes are identical, so this is what
      changed around them: the codec, the sequence length, the position. A
      confound to be *reported*, never subtracted silently.
    - `total` -- their sum, which must equal the difference of the two
      `bits_per_drawing` values, and is computed as that difference so a
      bookkeeping slip cannot hide inside the split.
    """
    def per(side: dict, name: str) -> float:
        return side["classes"][name]["per_drawing"]

    fold = {name: sign * per(flat if CLASSES[name] == "flat" else structured, name)
            for name, sign in FOLD_SIGN.items()}
    context = {name: per(flat, name) - per(structured, name)
               for name, side in CLASSES.items() if side == "both"}
    total = flat["bits_per_drawing"] - structured["bits_per_drawing"]
    return {
        "fold": fold,
        "fold_total": sum(fold.values()),
        "context": context,
        "context_total": sum(context.values()),
        "total": total,
        # Reported rather than asserted: the two sides can differ by truncation
        # if a spelling ran past `max_len`, and a residual is how a reader finds
        # that out instead of reading a silently rebalanced split.
        "residual": total - sum(fold.values()) - sum(context.values()),
    }
