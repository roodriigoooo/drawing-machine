"""The halting hazard is the diagnosis for a result, so what it measures is
pinned here.

Every arm generates median-24-33-byte programs against a real 41. That is either
the model over-assigning HALT at instruction boundaries -- in which case it is a
property of the model and no sampler change fixes it -- or it is the decode
settings. Telling the two apart is the whole point, so a stub model with known
probabilities is used rather than a trained one: the arithmetic has to be
exactly right before the number means anything.
"""

import numpy as np
import pytest
import torch

from dm.data.synthetic import split
from dm.eval.termination import (
    StrideUnsupported,
    halt_hazard,
    halt_symbol,
    summarise,
)
from dm.isa.asm import assemble
from dm.isa.codec import CODECS, opcode_mask


class ConstantModel:
    """Assigns a fixed probability to the HALT symbol everywhere, the rest
    spread uniformly. The hazard is then known in closed form."""

    def __init__(self, codec, p_halt: float) -> None:
        self.codec, self.p_halt = codec, p_halt

    def eval(self) -> None:
        pass

    def __call__(self, idx: torch.Tensor) -> torch.Tensor:
        vocab = self.codec.vocab_size
        target = halt_symbol(self.codec)
        rest = (1.0 - self.p_halt) / (vocab - 1)
        probs = torch.full((*idx.shape, vocab), rest)
        probs[..., target] = self.p_halt
        return probs.log()


def _programs(n: int = 8) -> list[bytes]:
    """Programs of a known, varied instruction count."""
    out = []
    for k in range(1, n + 1):
        body = "\n".join(f"MOVE {i} {i}" for i in range(k))
        out.append(assemble(body + "\nHALT"))
    return out


def test_the_halt_symbol_comes_from_the_codec_itself():
    """Hardcoding it per codec is how a fourth codec silently measures the wrong
    thing. Encoding a one-instruction program is the codec's own answer."""
    for name in ("byte", "token", "token_typed"):
        codec = CODECS[name]
        assert halt_symbol(codec) == codec.encode(bytes([0x00]))[0]
    assert CODECS["byte"].decode([halt_symbol(CODECS["byte"])]) == b"\x00"


def test_a_constant_hazard_is_recovered_exactly():
    """The arithmetic first: a model that says 0.1 everywhere must read back as
    0.1 at every boundary, and imply the geometric mean length that follows."""
    codec = CODECS["byte"]
    report = halt_hazard(ConstantModel(codec, 0.1), codec, _programs(), batch_size=4)

    assert np.allclose(report["model_hazard"][report["at_risk"] > 0], 0.1)
    # sum of the survival function of a geometric hazard, truncated at the
    # longest program: sum_{i<w} 0.9^i
    width = len(report["boundary"])
    assert report["implied_mean_instructions"] == pytest.approx(
        sum(0.9 ** i for i in range(width))
    )


def test_the_empirical_hazard_is_the_data_not_the_model():
    """It must not depend on the model at all -- it is the reference the model is
    being compared against, and a bug that let the model leak into it would make
    every arm look calibrated."""
    codec = CODECS["byte"]
    programs = _programs()
    one = halt_hazard(ConstantModel(codec, 0.05), codec, programs, batch_size=4)
    two = halt_hazard(ConstantModel(codec, 0.5), codec, programs, batch_size=4)

    assert np.array_equal(one["empirical_hazard"], two["empirical_hazard"])
    assert np.array_equal(one["at_risk"], two["at_risk"])
    # Computed straight from the corpus: of the programs that reach boundary i,
    # the fraction whose last instruction is there.
    counts = [len(opcode_mask(p)) and sum(opcode_mask(p)) for p in programs]
    at_risk = one["at_risk"]
    expected = [
        sum(c == i + 1 for c in counts) / sum(c >= i + 1 for c in counts)
        for i in range(len(at_risk))
    ]
    assert np.allclose(one["empirical_hazard"], expected)
    assert np.allclose(at_risk, [sum(c >= i + 1 for c in counts) for i in range(len(at_risk))])


def test_excess_halts_has_the_sign_that_predicts_undershooting():
    """The number has to point at a fix. Positive means the model fires HALT
    more often than the data does, which is short generation with no sampler
    involved; negative means the opposite."""
    codec = CODECS["byte"]
    programs = _programs()
    eager = halt_hazard(ConstantModel(codec, 0.9), codec, programs, batch_size=4)
    reluctant = halt_hazard(ConstantModel(codec, 0.001), codec, programs, batch_size=4)

    assert eager["excess_halts"] > 0
    assert reluctant["excess_halts"] < 0
    assert eager["implied_mean_instructions"] < reluctant["implied_mean_instructions"]


def test_a_multi_symbol_codec_is_refused_rather_than_approximated():
    """Under the bit codec only the first of eight conditional probabilities is
    on the teacher-forced path. Returning P(first bit is 0) would be a number of
    the right shape and the wrong meaning, and it would be compared across arms
    as if it were the hazard."""
    with pytest.raises(StrideUnsupported):
        halt_hazard(object(), CODECS["bit"], _programs(), batch_size=4)


def test_the_empirical_hazard_reproduces_the_corpus_mean_exactly():
    """The instrument's own check, and the reason its output can be believed.

    A survival function built from the true hazard has to sum to the true mean
    length. If it does not, the boundary alignment or the pooling is wrong, and
    every statement about the model's hazard is measuring that bug instead.
    """
    codec = CODECS["byte"]
    _, val = split(300, 64, seed=0)
    report = halt_hazard(ConstantModel(codec, 0.02), codec, val, batch_size=16)

    assert report["implied_mean_from_empirical"] == pytest.approx(
        report["true_mean_instructions"], abs=1e-9
    )


def test_it_runs_on_real_val_programs_and_the_summary_renders():
    """Boundaries come from a parse, so a program with `REPEAT` bodies must not
    shift the alignment. Real Tier A programs are the case that matters."""
    codec = CODECS["token"]
    _, val = split(200, 32, seed=0)
    report = halt_hazard(ConstantModel(codec, 0.02), codec, val, batch_size=8)

    assert report["true_mean_instructions"] > 1
    assert (report["at_risk"][:-1] >= report["at_risk"][1:]).all(), "at-risk must not grow"
    assert "implied mean" in summarise(report)
