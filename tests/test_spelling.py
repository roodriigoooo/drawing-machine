"""Where a drawing's bits go under two spellings of the same program.

P5's flat and structured arms differ by ~19 bits/drawing, and **the fold owns
about a third of that**. The rest is the codec and the sequence length. So every
test here is about one of the two ways that split can lie:

- **the classes not lining up** -- a byte in the wrong class, a gap, or a `both`
  class whose content differs between the spellings. Then the difference is not a
  measurement of context and the fold's term is whatever the misalignment made
  it. `check` refuses all three, and it refuses them per scene rather than in a
  test alone, because the corpus is regenerated from a seed and a future
  generator change is exactly what would break the alignment silently.
- **the bits not adding up** -- a decomposition whose parts do not sum to the
  model's own total is a decomposition of something else. Pinned twice: against
  an independent per-position sum, and against arithmetic on a model whose answer
  is known without running it.
"""

import math
from dataclasses import replace

import pytest
import torch

from dm.data import composed
from dm.eval import spelling
from dm.isa.codec import CODECS
from dm.isa.spec import SPECS, Op

N = 24


def corpus(**kwargs) -> list[composed.Scene]:
    return composed.build(N, "valid", seed=0, limit=600, **kwargs)


class Uniform(torch.nn.Module):
    """Equal logits everywhere, so every symbol costs exactly `log2(vocab)`.

    The anchor that needs no training and no tolerance: a class's bits are then
    its symbol count times a constant, and any disagreement is the span
    bookkeeping rather than the model.
    """

    def __init__(self, vocab: int):
        super().__init__()
        self.vocab = vocab

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros(*ids.shape, self.vocab)

    def eval(self):  # `_symbol_bits` calls it; nothing to switch off
        return self


def test_the_classes_partition_both_spellings_of_every_scene():
    """The trust anchor. Every byte of both spellings is in exactly one class,
    the shared classes hold identical bytes, and `copies − header` is the saving
    the generator recorded -- so the byte ceiling and the bit decomposition are
    describing one fold."""
    for scene in corpus():
        spelling.check(scene)
    for scene in corpus(control=True):
        spelling.check(scene)


def test_the_shared_classes_hold_the_same_bytes_in_both_spellings():
    """Stated as its own measurement because the whole design rests on it: the
    difference on these classes is the cost of the *context*, and if the content
    moved as well the number would mean nothing."""
    for scene in corpus():
        flat, structured = spelling.flat_spans(scene), spelling.structured_spans(scene)
        for name, side in spelling.CLASSES.items():
            if side != "both":
                continue
            assert scene.flat[flat[name][0]] == scene.structured[structured[name][0]]
        # And the two exclusive classes are exactly the bytes the other lacks.
        assert sum(s.stop - s.start for s in flat["copies"]) == \
            (len(scene.copies) - 1) * (scene.copies[0].stop - scene.copies[0].start)
        assert sum(s.stop - s.start for s in structured["header"]) == \
            SPECS[Op.REPEATX].size + SPECS[Op.ENDREP].size


@pytest.mark.parametrize("break_it, match", [
    # A copy span reaching past the orbit: the later-copies class then overlaps
    # the suffix, which is the misalignment that would charge the fold for
    # bytes it does not remove.
    (lambda s: replace(s, copies=tuple(slice(c.start, c.stop + 20)
                                       for c in s.copies)),
     "do not partition|classes cover"),
    # A program that does not end in HALT -- `HALT` is opcode 0, so a trailing
    # zero byte would *pass* this guard and be caught one check later by the
    # suffix instead. The suffix is defined as everything before the terminator,
    # so without the guard a byte changes class silently.
    (lambda s: replace(s, structured=s.structured + bytes([int(Op.MOVE)])),
     "ends in HALT"),
    # One operand of one distractor, in one spelling only. Both spellings still
    # parse and still draw -- they just no longer draw the same picture, and the
    # `context` term would carry the difference as if it were the alphabet's.
    (lambda s: replace(s, structured=bytes([s.structured[2] ^ 0xFF])
                       + s.structured[1:]), "differs between the two spellings"),
    # The generator's own saving disagreeing with the two exclusive classes:
    # then the byte ceiling and the bit decomposition describe different folds.
    (lambda s: replace(s, saved=s.saved + 1), "the fold saves"),
])
def test_a_scene_whose_spellings_disagree_is_refused(break_it, match):
    """Four ways to be wrong, each of which otherwise produces a plausible
    number and no exception."""
    scene = next(s for s in corpus() if s.copies[0].start > 0)
    spelling.check(scene)
    with pytest.raises(ValueError, match=match):
        spelling.check(break_it(scene))


def test_a_class_split_sums_to_the_models_own_total():
    """The reconciliation the script prints, pinned here against an independent
    path: bits summed per class must equal bits summed per position."""
    scenes = corpus()
    codec = CODECS["byte"]
    model = Uniform(codec.vocab_size)
    report = spelling.class_bits(model, [s.flat for s in scenes], codec,
                                 [spelling.flat_spans(s) for s in scenes])

    expected = math.log2(codec.vocab_size)
    for entry in report["classes"].values():
        assert entry["bits"] == pytest.approx(entry["symbols"] * expected)
    assert report["bytes_per_drawing"] == pytest.approx(
        sum(len(s.flat) for s in scenes) / len(scenes))
    assert report["bits_per_drawing"] == pytest.approx(
        report["bytes_per_drawing"] * expected)


def test_the_structured_spelling_costs_a_uniform_model_less_because_it_is_shorter():
    """The trivial half of the result, and the reason the interesting half needs
    an instrument: *any* model pays less for a shorter program, so a bare
    difference of totals cannot be the fold's value."""
    scenes = corpus()
    codec = CODECS["byte"]
    model = Uniform(codec.vocab_size)
    flat = spelling.class_bits(model, [s.flat for s in scenes], codec,
                              [spelling.flat_spans(s) for s in scenes])
    structured = spelling.class_bits(model, [s.structured for s in scenes], codec,
                                     [spelling.structured_spans(s) for s in scenes])
    split = spelling.delta(flat, structured)

    # Under a uniform model the context term is exactly zero -- the shared
    # classes hold the same bytes and every byte costs the same -- so the whole
    # difference is the fold, and it is the byte saving times log2(vocab).
    assert split["context_total"] == pytest.approx(0.0, abs=1e-9)
    assert split["residual"] == pytest.approx(0.0, abs=1e-9)
    assert split["fold_total"] == pytest.approx(
        sum(s.saved for s in scenes) / len(scenes) * math.log2(codec.vocab_size))


def test_the_fold_reads_copies_from_the_flat_arm_and_the_header_from_the_other():
    """The one place a sign or a side could be wrong without any test noticing:
    `copies` exists only in the flat spelling and `header` only in the structured
    one, so reading either from the wrong arm would report a zero as a saving."""
    def side(**per_drawing):
        classes = {name: {"bits": 0.0, "symbols": 0, "bytes_per_drawing": 0.0,
                          "per_drawing": per_drawing.get(name, 0.0),
                          "bits_per_symbol": 0.0}
                   for name in spelling.CLASSES}
        return {"n": 1, "stride": 1, "classes": classes,
                "bits_per_drawing": sum(c["per_drawing"] for c in classes.values()),
                "bytes_per_drawing": 0.0}

    flat = side(copies=100.0, motif=10.0, header=999.0)
    structured = side(header=7.0, motif=4.0, copies=999.0)
    split = spelling.delta(flat, structured)

    assert split["fold"] == {"copies": 100.0, "header": -7.0}
    assert split["fold_total"] == pytest.approx(93.0)
    assert split["context_total"] == pytest.approx(6.0)
    # The 999s are the values a side-swap would pick up, and neither appears.
    assert split["total"] == pytest.approx(1109.0 - 1010.0)
    assert split["residual"] == pytest.approx(0.0)


def test_a_stride_eight_alphabet_still_reports_byte_lengths():
    """`bit` spells one byte as eight symbols, so a class's *symbol* count is not
    its byte count. The comparison under test is between two spellings and both
    of its length columns are in bytes."""
    scenes = corpus()[:4]
    codec = CODECS["bit"]
    report = spelling.class_bits(Uniform(codec.vocab_size),
                                 [s.flat for s in scenes], codec,
                                 [spelling.flat_spans(s) for s in scenes],
                                 max_len=8192)
    assert report["stride"] == 8
    assert report["bytes_per_drawing"] == pytest.approx(
        sum(len(s.flat) for s in scenes) / len(scenes))
    assert report["classes"]["motif"]["symbols"] == 8 * sum(
        s.copies[0].stop - s.copies[0].start for s in scenes)
