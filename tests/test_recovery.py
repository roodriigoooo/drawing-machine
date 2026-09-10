"""Claim 2's recovery metric, pinned against models whose answer is known."""

import math
from dataclasses import asdict

import numpy as np
import pytest
import torch

from dm.data import composed
from dm.data.dataset import ProgramDataset
from dm.eval.records import composed_val_scenes
from dm.eval.recovery import recovery, recovery_by_copy, repeat_copies, repeat_spans
from dm.eval.repeats import compress
from dm.isa.asm import assemble
from dm.isa.codec import CODECS
from dm.models.transformer import Config, DrawingLM
from dm.train import TrainConfig, build_data


def grid(n: int = 4, dx: int = 30) -> bytes:
    lines = []
    for i in range(n):
        x = 10 + dx * i
        lines += [f"MOVE {x} 10", f"LINE {x + 20} 10", f"LINE {x + 20} 30"]
    lines.append("HALT")
    return assemble("\n".join(lines))


class Oracle(torch.nn.Module):
    """Assigns near-zero loss to positions inside `free`, high loss elsewhere.

    Stands in for a model that has perfectly learned the repeat: it lets the
    metric be checked against a known answer without training anything.
    """

    def __init__(self, vocab: int, free: set[int]) -> None:
        super().__init__()
        self.vocab, self.free = vocab, free

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(*idx.shape, self.vocab)
        for t in range(idx.shape[1]):
            confidence = 20.0 if t in self.free else 0.0
            for b in range(idx.shape[0]):
                nxt = idx[b, t + 1].item() if t + 1 < idx.shape[1] else 0
                logits[b, t, nxt] = confidence
        return logits

    def eval(self):
        return self


def test_spans_cover_the_first_copy_and_the_rest_separately():
    program = grid(n=4)
    spans = repeat_spans(program)
    assert len(spans) == 1
    head, rest = spans[0]
    assert head.stop - head.start == 9              # MOVE + 2 LINE = 9 bytes
    assert rest.stop - rest.start == 27             # three further copies
    assert compress(program)[0] > 0


def test_a_program_without_repeats_has_no_spans():
    assert repeat_spans(assemble("MOVE 10 10\nLINE 90 40\nLINE 12 200\nHALT")) == []


def test_recovery_is_one_when_later_copies_are_free():
    """The ceiling case: a model that pays nothing for the redundant copies."""
    codec = CODECS["byte"]
    program = grid()
    _, rest = repeat_spans(program)[0]
    seqs = ProgramDataset([program], codec, 2048)
    free = set(range(rest.start, min(rest.stop, len(seqs[0][1]))))
    got = recovery(Oracle(codec.vocab_size, free), [program], codec)
    assert got["recovery"] > 0.9
    assert got["n_with_repeat"] == 1


def test_recovery_is_about_zero_when_every_copy_costs_the_same():
    """The null: a model that has learned nothing about repetition."""
    codec = CODECS["byte"]
    program = grid()
    got = recovery(Oracle(codec.vocab_size, set()), [program], codec)
    assert abs(got["recovery"]) < 1e-6


def test_rates_are_per_symbol_not_totals():
    """Totals would report ~0.75 for the null above purely because copies 2..n
    outnumber copy 1 -- the fault this normalisation exists to avoid."""
    codec = CODECS["byte"]
    got = recovery(Oracle(codec.vocab_size, set()), [grid()], codec)
    assert np.isclose(got["bits_per_symbol_first"], got["bits_per_symbol_later"])


def test_it_works_on_the_bit_codec_too():
    """One bytecode byte is eight symbols there; spans must scale with stride."""
    codec = CODECS["bit"]
    program = grid()
    _, rest = repeat_spans(program)[0]
    seqs = ProgramDataset([program], codec, 4096)
    free = set(range(rest.start * 8, min(rest.stop * 8, len(seqs[0][1]))))
    got = recovery(Oracle(codec.vocab_size, free), [program], codec, max_len=4096)
    assert got["recovery"] > 0.9


def test_copies_are_reported_in_order_and_cover_the_program():
    program = grid(n=5)
    copies = repeat_copies(program)
    assert len(copies) == 1
    assert [c.stop - c.start for c in copies[0]] == [9] * 5
    assert [c.start for c in copies[0]] == [0, 9, 18, 27, 36]
    # `repeat_spans` is the same folding, collapsed -- so it cannot drift.
    head, rest = repeat_spans(program)[0]
    assert head == copies[0][0]
    assert (rest.start, rest.stop) == (copies[0][1].start, copies[0][-1].stop)


def test_the_copy_curve_separates_a_rule_from_a_table():
    """The length-generalisation test's whole content.

    Two oracles with the *same* `recovery`: one pays nothing for any later copy
    (the rule), one pays nothing up to copy 4 and full price after (the table
    that ran out). A single number cannot tell them apart and the curve does.
    """
    codec = CODECS["byte"]
    program = grid(n=8, dx=20)
    copies = repeat_copies(program)[0]
    width = len(ProgramDataset([program], codec, 2048)[0][1])

    def free_from(first: int, last: int) -> set[int]:
        return {
            t
            for c in copies[first:last]
            for t in range(c.start, min(c.stop, width))
        }

    rule = recovery_by_copy(
        Oracle(codec.vocab_size, free_from(1, len(copies))), [program], codec
    )
    table = recovery_by_copy(
        Oracle(codec.vocab_size, free_from(1, 4)), [program], codec
    )
    assert rule["copies"] == table["copies"] == list(range(1, 9))
    # The rule stays flat and cheap past the trained bound; the table does not.
    assert max(rule["bits_per_symbol"][1:]) < 0.1
    assert min(table["bits_per_symbol"][4:]) > 1.0
    assert max(table["bits_per_symbol"][1:4]) < 0.1


# --- The transform tier: copies that no byte matcher can find ----------------
#
# `docs/direction.md` section 2, P5. A mirrored copy is `x -> 255 - x`, so it
# shares no coordinate byte with its original and the detector these tests pin
# above finds *nothing* on the corpus built to contain the structure. The
# provenance path is what makes the measurement possible at all, and it is only
# trustworthy if it is scoring the split the run was actually scored on.

COMPOSED_SCENES = 24


def composed_record(**extra) -> tuple[dict, list[bytes]]:
    """A minimal record for a `--data composed` run, and its val split."""
    cfg = TrainConfig(data="composed", n_train=4, n_val=COMPOSED_SCENES,
                      extra={"limit": 400, **extra})
    record = {"name": "composed_test", "config": {**asdict(cfg), "tier": int(cfg.tier)}}
    return record, build_data(cfg)[1]


def test_the_generator_supplies_the_copies_a_byte_matcher_cannot_see():
    """The measurement P5 rests on, end to end: detection reports no repeats at
    all on this corpus, and provenance reports one orbit in every program. The
    difference is not precision -- it is a `NaN` against a number."""
    record, programs = composed_record()
    scenes = composed_val_scenes(record, programs)
    codec = CODECS["byte"]
    model = DrawingLM(Config(vocab_size=codec.vocab_size, max_len=2048,
                             d_model=32, n_layers=1, n_heads=2))

    detected = recovery(model, programs, codec)
    assert detected["n_with_repeat"] == 0
    assert math.isnan(detected["recovery"])

    known = recovery(model, programs, codec, spans=composed.spans_of(scenes))
    assert known["n_with_repeat"] == len(programs)
    assert not math.isnan(known["recovery"])

    curve = recovery_by_copy(model, programs, codec,
                             copies=composed.copies_of(scenes))
    assert curve["copies"][0] == 1
    assert curve["symbols"][0] == sum(s.copies[0].stop - s.copies[0].start
                                      for s in scenes)


def test_provenance_is_refused_rather_than_approximated():
    """Three ways to score real bits against the wrong bytes, all silent, all
    refused. The drift case is the one no reviewer could catch by reading the
    output: every number would be plausible."""
    record, programs = composed_record()
    with pytest.raises(ValueError, match="corpus has drifted"):
        composed_val_scenes(record, programs[::-1])

    other = {"name": "n", "config": {**asdict(TrainConfig(data="quickdraw")),
                                     "tier": int(TrainConfig().tier)}}
    with pytest.raises(ValueError, match="only for the constructed corpus"):
        composed_val_scenes(other, programs)

    folded, folded_programs = composed_record(structured=True)
    with pytest.raises(ValueError, match="structured spelling"):
        composed_val_scenes(folded, folded_programs)
