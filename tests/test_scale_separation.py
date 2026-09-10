"""The prompt is the experiment's control surface, so how it is cut is pinned.

`scripts/scale_separation.py` conditions on the first k% of a held-out program
and compares the completion's geometry against the whole truth. Two ways that
silently stops being a paired comparison, both cheap to pin and neither visible
in the output:

* Re-encoding a byte prefix instead of slicing the full encoding. `TokenCodec`
  types operands by walking the stream, and a walk over a prefix that ends
  mid-instruction types the tail differently -- so the model would be prompted
  with symbols the program does not contain.
* Losing the prompt from the completion, which would make the comparison a
  free-running sample dressed up as a continuation.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from dm.data import synthetic
from dm.isa.asm import assemble
from dm.isa.codec import CODECS
from dm.models.transformer import Config, DrawingLM

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "scale_separation.py"
_spec = importlib.util.spec_from_file_location("scale_separation", _PATH)
assert _spec and _spec.loader, f"no driver at {_PATH}"
scale_separation = importlib.util.module_from_spec(_spec)
sys.modules["scale_separation"] = scale_separation
_spec.loader.exec_module(scale_separation)


def _model(codec) -> DrawingLM:
    torch.manual_seed(0)
    return DrawingLM(
        Config(vocab_size=codec.vocab_size, d_model=32, n_layers=2, n_heads=2, max_len=2048)
    )


@pytest.mark.parametrize("name", ["byte", "token", "token_typed", "bit"])
def test_every_completion_begins_with_the_program_it_was_prompted_with(name):
    """The paired comparison exists only if the completion really continues the
    truth. This also catches a prompt re-encoded from a byte prefix: under the
    token codec that produces different symbols, which decode to different
    bytes, and the prefix stops matching."""
    codec = CODECS[name]
    programs = synthetic.dataset(6, seed=3, **synthetic.LADDER["simple"])
    completions, prompts = scale_separation.complete(
        _model(codec), codec, programs, prefix=0.25, device="cpu", cap=140, batch_size=3
    )

    for program, completion, prompt in zip(programs, completions, prompts):
        assert prompt == program[: len(prompt)], "the prompt is not a prefix of the truth"
        assert completion[: len(prompt)] == prompt, "the completion dropped its prompt"
        assert len(prompt) >= 1


def test_the_prompt_is_sliced_from_the_full_encoding_not_re_encoded():
    """The token codec's operand typing depends on a walk over the whole stream.
    A prefix that ends mid-instruction re-encodes differently, and the model
    would be conditioned on symbols the program never contained."""
    codec = CODECS["token"]
    program = assemble("MOVE 10 20\nCURVE 1 2 3 4 5 6\nHALT")
    cut = 5  # MOVE is 3 bytes, so this lands inside CURVE's operands

    assert codec.encode(program[:cut]) != codec.encode(program)[:cut], (
        "this test is vacuous unless re-encoding really differs"
    )
    # +0.5 of a byte so the driver's `int(len * prefix)` cannot land one short
    # through floating point.
    _completions, prompts = scale_separation.complete(
        _model(codec), codec, [program], prefix=(cut + 0.5) / len(program),
        device="cpu", cap=40,
    )
    assert prompts[0] == program[:cut]
    assert codec.encode(program)[:cut] == codec.encode(program)[: len(prompts[0])]


def _score(matched: int, n_truth: int, n_pred: int) -> dict:
    return {"matched": matched, "n_truth": n_truth, "n_prediction": n_pred}


def test_coverage_charges_omitted_and_invented_strokes_alike():
    assert scale_separation.coverage([_score(4, 4, 4)]) == pytest.approx(1.0)
    assert scale_separation.coverage([_score(2, 4, 2)]) == pytest.approx(0.5)  # omitted half
    assert scale_separation.coverage([_score(2, 2, 4)]) == pytest.approx(0.5)  # invented half
    assert scale_separation.coverage([_score(0, 4, 0)]) == pytest.approx(0.0)  # emitted nothing


def test_an_empty_prediction_cannot_beat_an_honest_attempt():
    """The defect the first run exposed, pinned.

    Scoring matched pairs only, one lucky stroke out of forty beats an attempt
    at all forty, and the prompt-only floor -- which emits fewest strokes of
    all -- came out above the model at every rung. Coverage scaling has to
    reverse that ordering.
    """
    lucky_but_empty = [_score(1, 40, 1)]
    honest_attempt = [_score(30, 40, 35)]

    assert scale_separation.coverage(lucky_but_empty) < scale_separation.coverage(honest_attempt)

    # A perfect matched score on 1 of 40 strokes must not out-rank a mediocre
    # one that accounts for most of the drawing.
    perfect_on_one = 1.00 * scale_separation.coverage(lucky_but_empty)
    mediocre_on_most = 0.50 * scale_separation.coverage(honest_attempt)
    assert perfect_on_one < mediocre_on_most


def test_coverage_ignores_programs_with_no_strokes_on_either_side():
    """`max(...)` is zero there, and a zero denominator must not enter the mean."""
    assert scale_separation.coverage([_score(0, 0, 0), _score(2, 2, 2)]) == pytest.approx(1.0)
