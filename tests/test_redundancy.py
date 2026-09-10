"""The redundancy instrument, pinned against masks and models whose answer is
known in closed form.

**One test here matters more than the rest: no rule may ever exclude the byte
that is actually there.** Every number this file produces is `-log2 P(feasible)`,
so a mask that rules out the truth reports an infinite saving on that position
and a merely large one after averaging -- and it would look exactly like a
finding. It caught a real fault on the first run: `Summary.length` is u8, so on
the two val strokes longer than 255 bytes the `fits` rule was applied to a
length the summary cannot state, and excluded the true byte at 48 positions.

The rest is arithmetic. A decoder that puts all its mass on the true byte must
waste nothing under any rule; a uniform one must waste exactly the log of the
feasible fraction, which is computable from the masks with no model at all.
"""

import math

import numpy as np
import pytest
import torch

from dm.data.synthetic import split as synthetic_split
from dm.eval.redundancy import (
    SUMMARY_RULES,
    CodecUnsupported,
    flat_redundancy,
    flat_rows,
    planner_redundancy,
    planner_rows,
    positions,
    redundancy,
)
from dm.isa.asm import assemble
from dm.isa.codec import CODECS, N_SPECIAL
from dm.isa.spec import Op
from dm.isa.strokes import Summary, split
from dm.models.planner import PlannerConfig, StrokePlanner, prompt_len

VOCAB = CODECS["byte"].vocab_size
RULES = (*SUMMARY_RULES, "isa")


def corpus(n: int = 120) -> list[bytes]:
    """Real programs, and Tier A rather than QuickDraw on purpose: it carries
    `REPEAT`, `CIRCLE` and `WIDTH`, so the walk has to tell a `SCALAR` radius
    from a `COORD` position -- the distinction the coordinate rules rest on and
    the one QuickDraw's `MOVE`/`LINE` corpus cannot exercise."""
    train, _ = synthetic_split(n, 8, seed=0)
    return train


def strokes_of(programs: list[bytes], max_strokes: int | None = None):
    for program in programs:
        for stroke in split(program, max_strokes):
            yield stroke, Summary.of(stroke)


def test_no_rule_ever_excludes_the_byte_that_is_actually_there():
    """The one that decides whether any number here can be believed.

    Run over every stroke of a real corpus, every rule, every position. An
    exclusion of the truth is not a conservative error in this instrument -- it
    is an *unbounded* one in the direction of the hypothesis.
    """
    excluded = dict.fromkeys(RULES, 0)
    checked = 0
    for stroke, summary in strokes_of(corpus()):
        where = positions(stroke, summary)
        values = np.frombuffer(stroke, dtype=np.uint8)
        checked += len(stroke)
        for rule in RULES:
            allowed = where.masks[rule][np.arange(len(stroke)), values]
            excluded[rule] += int((~allowed).sum())

    assert checked > 4_000, "the corpus has to be big enough to be a test"
    assert excluded == dict.fromkeys(RULES, 0), excluded


def test_the_length_rules_switch_off_when_the_summary_cannot_state_the_length():
    """`Summary.length` is u8. Past 255 bytes it names a different stroke, and
    applying `fits` to it excluded the true byte 48 times on QuickDraw's val
    split before this was fixed."""
    long = assemble("\n".join(f"MOVE {i % 200} {i % 150}" for i in range(120)))
    summary = Summary.of(long)
    assert len(long) > 255 and summary.length == 255

    where = positions(long, summary)
    assert where.masks["fits"].all(), "a length that cannot be stated constrains nothing"
    values = np.frombuffer(long, dtype=np.uint8)
    for rule in RULES:
        assert where.masks[rule][np.arange(len(long)), values].all()


def test_first_point_is_a_singleton_and_the_box_always_admits_it():
    """`Summary.of` reads `xs[0], ys[0]`, so the stroke's first coordinate pair
    *is* the summary's first two fields -- not approximately, byte for byte."""
    stroke = assemble("MOVE 40 60\nLINE 90 61\nLINE 12 200")
    summary = Summary.of(stroke)
    where = positions(stroke, summary)

    assert where.masks["first_point"][1].sum() == 1
    assert where.masks["first_point"][1, summary.x0]
    assert where.masks["first_point"][2].sum() == 1
    assert where.masks["first_point"][2, summary.y0]
    # The box is defined relative to that point, so it can never exclude it.
    assert where.masks["coord_box"][:, summary.x0].all()
    # And it is a *box*, not a half-line: 12 is below x0 = 40 and legal.
    assert where.masks["coord_box"][4, 12]


def test_halt_is_impossible_when_the_summary_says_the_drawing_continues():
    stroke = assemble("MOVE 10 10\nLINE 20 20")
    where = positions(stroke, Summary.of(stroke))
    assert not where.masks["halt"][0, int(Op.HALT)]

    ending = assemble("MOVE 10 10\nLINE 20 20\nHALT")
    summary = Summary.of(ending)
    assert summary.halts
    forced = positions(ending, summary)
    # One byte left and the drawing has to end inside this stroke.
    assert forced.masks["halt"][len(ending) - 1].sum() == 1
    assert forced.masks["halt"][len(ending) - 1, int(Op.HALT)]


class Confident:
    """A decoder that already knows the answer. Wastes nothing, by definition."""

    def rows(self, programs: list[bytes], max_strokes: int | None = None,
             confidence: float = 30.0):
        for index, program in enumerate(programs):
            for stroke in split(program, max_strokes):
                logits = torch.zeros(len(stroke), VOCAB)
                for j, byte in enumerate(stroke):
                    logits[j, N_SPECIAL + byte] = confidence
                yield index, stroke, Summary.of(stroke), torch.log_softmax(logits, -1)


class Uniform:
    """Uniform over the whole vocabulary. Its waste is arithmetic on the masks."""

    def rows(self, programs: list[bytes], max_strokes: int | None = None):
        for index, program in enumerate(programs):
            for stroke in split(program, max_strokes):
                logits = torch.zeros(len(stroke), VOCAB)
                yield index, stroke, Summary.of(stroke), torch.log_softmax(logits, -1)


def test_a_decoder_that_knows_the_answer_wastes_nothing():
    """The null. Every rule reads ~0, and so does the pinned-position column --
    which is what the planner's stroke decoder turns out to look like."""
    programs = corpus(40)
    got = redundancy(Confident().rows(programs), len(programs), VOCAB).summarise(len(programs))

    assert got["summary_bits_per_drawing"] < 0.01
    assert got["isa_bits_per_drawing"] < 0.01
    for rule in SUMMARY_RULES:
        assert got[f"{rule}_bits_per_drawing"] < 0.01
    assert got["determined_bits_per_symbol"] < 0.01


def test_a_uniform_decoder_wastes_exactly_the_log_of_the_feasible_fraction():
    """The ceiling, and it needs no model: a uniform coder recovers
    `log2(vocab / |feasible|)` at every position, so the instrument can be
    checked against a number computed straight from the masks."""
    programs = corpus(20)
    got = redundancy(Uniform().rows(programs), len(programs), VOCAB).summarise(len(programs))

    isa = summary = 0.0
    for stroke, summ in strokes_of(programs):
        where = positions(stroke, summ)
        allowed = where.masks["isa"].sum(axis=1)
        isa += float(np.log2(VOCAB / allowed).sum())
        combined = where.masks["isa"].copy()
        for rule in SUMMARY_RULES:
            combined &= where.masks[rule]
        summary += float(np.log2(VOCAB / combined.sum(axis=1)).sum()) - float(
            np.log2(VOCAB / allowed).sum())

    assert got["isa_bits_per_drawing"] == pytest.approx(isa / len(programs), rel=1e-6)
    assert got["summary_bits_per_drawing"] == pytest.approx(summary / len(programs), rel=1e-6)
    # And the partition reading: uniform over 258 symbols is log2(258) a symbol,
    # pinned or free, since the model has no idea either way.
    assert got["determined_bits_per_symbol"] == pytest.approx(math.log2(VOCAB), rel=1e-6)


def test_the_summary_saving_is_measured_over_the_isa_and_never_double_counted():
    """`isa` is the baseline, not a peer. A decoder that spends everything on
    unparseable opcodes must show that as ISA waste and *not* as evidence that
    the conditioning is being ignored -- the flat AR control has the same ISA
    knowledge, so folding the two together would answer a different question."""
    programs = corpus(20)
    rows = list(Uniform().rows(programs))
    got = redundancy(iter(rows), len(programs), VOCAB).summarise(len(programs))

    for stroke, summ in strokes_of(programs):
        where = positions(stroke, summ)
        combined = where.masks["isa"].copy()
        for rule in SUMMARY_RULES:
            combined &= where.masks[rule]
        # Nested sets, so the two savings compose exactly rather than overlap.
        assert (combined <= where.masks["isa"]).all()
    assert got["summary_bits_per_drawing"] > 0
    assert got["isa_bits_per_drawing"] > 0


def test_pinned_and_free_positions_are_read_apart():
    """§3.2's reading. A decoder confident only where the summary leaves one
    legal value must separate the two columns by construction."""
    programs = corpus(20)

    def rows():
        for index, program in enumerate(programs):
            for stroke in split(program):
                where = positions(stroke, Summary.of(stroke))
                logits = torch.zeros(len(stroke), VOCAB)
                for j, byte in enumerate(stroke):
                    if where.determined[j]:
                        logits[j, N_SPECIAL + byte] = 30.0
                yield index, stroke, Summary.of(stroke), torch.log_softmax(logits, -1)

    got = redundancy(rows(), len(programs), VOCAB).summarise(len(programs))
    assert got["determined_symbols"] > 0, "the corpus must have pinned positions"
    assert got["determined_bits_per_symbol"] < 0.01
    assert got["free_bits_per_symbol"] == pytest.approx(math.log2(VOCAB), rel=1e-6)


class Positional(torch.nn.Module):
    """Puts all its mass on a symbol that *names the row position it is at*.

    Only useful for one thing, and it is the thing most likely to be silently
    wrong: **alignment**. The planner's rows are `BOS`, six summary symbols,
    then the stroke; the flat arm's are `BOS` then the whole program. Both turn
    row positions into stroke byte positions with an offset that no assertion in
    the arithmetic would catch, and on `first_point` an off-by-one is the
    difference between the answer and its opposite.

    A model whose output *is* its position makes the offset readable directly
    off the returned log-probabilities, which a peeking oracle cannot do -- its
    final column has nothing after it to peek at, so it would fail for a reason
    that has nothing to do with alignment.
    """

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        pos = (torch.arange(idx.shape[1]) % 256 + N_SPECIAL)
        logits = torch.zeros(*idx.shape, VOCAB)
        return logits.scatter(2, pos.expand(idx.shape)[:, :, None], 30.0)


def test_the_planner_rows_start_after_the_summary_prompt():
    """`given` is six summary symbols after `BOS`, and position `given + j`
    predicts stroke byte `j`. Read straight off the slice."""
    programs = corpus(12)
    cfg = PlannerConfig(vocab_size=VOCAB, comp_objective="ar")
    planner = StrokePlanner(cfg)
    planner.decoder = Positional()  # type: ignore[assignment]
    given = prompt_len(CODECS["byte"])
    assert given == 6

    rows = list(planner_rows(planner, programs, CODECS["byte"], batch_size=4))
    assert [stroke for _, stroke, _, _ in rows] == [
        s for p in programs for s in split(p, cfg.max_strokes)]

    for _, stroke, _, logprobs in rows:
        want = (torch.arange(logprobs.shape[0]) + given) % 256 + N_SPECIAL
        assert torch.equal(logprobs.argmax(dim=-1), want)
        assert logprobs.shape[0] == min(len(stroke), cfg.max_stroke_len - given - 1)


def test_the_flat_rows_are_offset_by_where_the_stroke_starts():
    """The control has to be scored on the identical bytes, indexed from the
    program rather than from the stroke: byte `j` of the program is predicted at
    row position `j`, so a stroke starting at `off` occupies `off ...`."""
    programs = corpus(12)
    rows = list(flat_rows(Positional(), programs, CODECS["byte"], batch_size=4))
    assert [stroke for _, stroke, _, _ in rows] == [s for p in programs for s in split(p)]

    offsets: dict[int, int] = {}
    for index, stroke, _, logprobs in rows:
        off = offsets.get(index, 0)
        want = (torch.arange(logprobs.shape[0]) + off) % 256 + N_SPECIAL
        assert torch.equal(logprobs.argmax(dim=-1), want)
        offsets[index] = off + len(stroke)


def test_the_two_arms_are_scored_through_one_reduction():
    """`redundancy` takes a row source rather than a model precisely so the
    planner and the flat control cannot drift apart. Same strokes, same
    summaries, same masks -- only the probabilities differ."""
    programs = corpus(12)
    cfg = PlannerConfig(vocab_size=VOCAB, comp_objective="ar")
    planner = StrokePlanner(cfg)
    planner.decoder = Positional()  # type: ignore[assignment]

    planner_side = [(i, s, m) for i, s, m, _ in
                    planner_rows(planner, programs, CODECS["byte"], batch_size=4)]
    flat_side = [(i, s, m) for i, s, m, _ in
                 flat_rows(Positional(), programs, CODECS["byte"], batch_size=4)]
    assert planner_side == flat_side


def test_a_non_byte_alphabet_is_refused_rather_than_approximated():
    """The mask is over byte values. Under `token` an opcode and an operand with
    the same value are different symbols, and under `bit` no byte-valued
    distribution exists at any position."""
    for name in ("token", "token_typed", "bit"):
        with pytest.raises(CodecUnsupported):
            flat_redundancy(Positional(), corpus(2), CODECS[name])
    with pytest.raises(CodecUnsupported):
        planner_redundancy(StrokePlanner(PlannerConfig(comp_objective="ar")),
                           corpus(2), CODECS["bit"])
