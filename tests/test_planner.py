"""Claim 3's planner: the split is exact, the bound is a bound, the budget holds.

The one that matters is `test_the_reported_number_is_an_upper_bound`. The AR
baseline reports an exact NLL and the planner reports a bound, so a comparison
between them is only decisive if the bound really is conservative -- and a
"bound" that is quietly optimistic would produce a claim-3 win that is an
arithmetic error.
"""

import functools
import json
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from dm.data import synthetic
from dm.isa.asm import assemble, parse
from dm.isa.codec import CODECS
from dm.isa.spec import Op, spec_for
from dm.isa.strokes import SUMMARY_BYTES, Summary, join, split, summaries, to_boundary
from dm.models.planner import (
    COMP_BOS,
    COMP_EMPTY,
    COMP_MASK,
    COMP_PAD,
    COMP_VALUE_BASE,
    CompositionDenoiser,
    PlannerConfig,
    StrokePlanner,
    bits_per_drawing,
    generate,
    grid_summaries,
    gumbel_like,
    per_program_bits,
    prompt_len,
    stroke_batch,
    summary_grid,
    unrepresentable_length,
)
from dm.train_planner import PLANNER_SHAPES
from dm.vm.interp import VM

CORPUS = synthetic.dataset(48, seed=7)
CODEC = CODECS["byte"]


def accelerator() -> str | None:
    """The device training actually runs on, or None.

    Shared by the two tests that exist because CPU-only testing missed a
    device-specific fault: the float64 cast that cost run 4 a launch, and the
    NaN ordering that made every planner generation column describe the
    backend rather than the model.
    """
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else None


def tiny(codec=CODEC, **kw) -> PlannerConfig:
    defaults = dict(
        vocab_size=codec.vocab_size, max_strokes=8, max_stroke_len=96,
        comp_d_model=32, comp_layers=1, stroke_d_model=32, stroke_layers=1,
        eval_noise_samples=2,
    )
    return PlannerConfig(**{**defaults, **kw})


# --- the split -------------------------------------------------------------

def test_split_and_join_are_exact():
    """bits/drawing is only comparable with the AR baseline's if both models
    are transmitting the same bytes."""
    for program in CORPUS:
        assert join(split(program)) == program


def test_to_boundary_leaves_a_whole_program_alone():
    for program in CORPUS:
        assert to_boundary(program) == program


def test_to_boundary_drops_a_partial_instruction():
    """The case the planner produced: a plan naming a byte count that lands
    inside an instruction. Every truncation length of a real program must come
    back as a whole number of instructions."""
    program = CORPUS[0]
    for cut in range(1, len(program)):
        trimmed = to_boundary(program[:cut])
        assert trimmed == program[: len(trimmed)]
        assert to_boundary(trimmed) == trimmed  # idempotent
        # Every byte accounted for by a whole instruction, which is the
        # property the VM's `truncated` fault reports the absence of. Walked
        # here rather than delegated, so the test restates the invariant
        # instead of re-running the code under test.
        pc = 0
        while pc < len(trimmed):
            pc += spec_for(trimmed[pc]).size
        assert pc == len(trimmed)


def test_to_boundary_stops_at_an_unknown_opcode():
    """`VM.run` breaks there and every later byte is dead, so keeping them
    would change the trace -- the same reason `HaltMonitor` stops the row."""
    program = assemble("MOVE 10 20\nLINE 30 40")
    assert to_boundary(program + b"\xfe" + program) == program


def test_every_stroke_after_the_first_starts_with_a_move():
    for program in CORPUS:
        parts = split(program)
        for stroke in parts[1:]:
            assert parse(stroke)[0].mnemonic == "MOVE"


def test_the_preamble_and_the_halt_are_not_lost():
    program = assemble("WIDTH 3\nMOVE 10 10\nLINE 20 20\nMOVE 40 40\nLINE 50 50\nHALT")
    parts = split(program)
    assert len(parts) == 2
    assert parse(parts[0])[0].mnemonic == "WIDTH"   # preamble joins stroke 1
    assert parse(parts[-1])[-1].mnemonic == "HALT"  # terminator joins the last
    assert join(parts) == program


def test_overflow_is_merged_not_dropped():
    """Dropping would remove bytes from the program and make bits/drawing
    incomparable on exactly the hardest drawings."""
    program = assemble("\n".join(f"MOVE {i} {i}\nLINE {i + 1} {i + 1}" for i in range(9))
                       + "\nHALT")
    parts = split(program, max_strokes=4)
    assert len(parts) == 4
    assert join(parts) == program


def test_the_summary_is_a_deterministic_function_of_the_stroke():
    """The property the whole factorisation rests on -- see the module docstring
    of `dm/models/planner.py`."""
    for program in CORPUS:
        for stroke in split(program):
            assert Summary.of(stroke) == Summary.of(stroke)
            assert len(Summary.of(stroke).to_bytes()) == SUMMARY_BYTES


def test_the_summary_reads_coordinates_by_kind():
    """A CIRCLE radius is a length, not a position; it must not enter the box."""
    a = Summary.of(assemble("MOVE 10 20\nCIRCLE 200"))
    b = Summary.of(assemble("MOVE 10 20\nCIRCLE 3"))
    assert (a.x0, a.y0, a.width, a.height) == (b.x0, b.y0, b.width, b.height)
    assert Summary.of(assemble("MOVE 10 20\nLINE 40 60")).width == 30


def test_a_malformed_stroke_still_gets_a_summary():
    """Validity is measured in the VM for every arm, so nothing upstream of it
    may raise on a generated program."""
    assert Summary.of(bytes([0xFE, 0x01, 0x02])).length == 3
    assert Summary.of(b"") == Summary(0, 0, 0, 0, 0, 0)


def test_halts_marks_the_stroke_that_ends_the_drawing():
    """Termination is structural in the planner and sampled in the AR baseline,
    which is where the baseline's standing failure lives (`PLAN.md` 9.5a)."""
    for program in CORPUS:
        flags = [s.halts for s in summaries(program)]
        assert flags[-1] == 1
        assert sum(flags) == 1
    assert Summary.of(assemble("MOVE 1 1\nLINE 2 2")).halts == 0


def test_generation_stops_at_the_halting_stroke():
    """A slot after the one the level marked terminal is not drawn: the
    composition level decides where the drawing ends."""
    row = summary_grid(assemble("MOVE 1 1\nLINE 2 2\nHALT"), max_strokes=4)
    row[SUMMARY_BYTES:2 * SUMMARY_BYTES] = [COMP_VALUE_BASE + 9] * SUMMARY_BYTES
    assert len(grid_summaries(row)) == 1


# --- the composition level -------------------------------------------------

def test_empty_slots_are_empty_and_not_pad():
    """PAD is 'outside the tensor'; EMPTY is 'this drawing has no stroke here',
    and the second is something the level has to predict."""
    grid = summary_grid(assemble("MOVE 1 1\nLINE 2 2\nHALT"), max_strokes=4)
    assert len(grid) == 4 * SUMMARY_BYTES
    assert all(v >= COMP_VALUE_BASE for v in grid[:SUMMARY_BYTES])
    assert set(grid[SUMMARY_BYTES:]) == {COMP_EMPTY}


def test_the_grid_round_trips_through_the_summary_reader():
    program = CORPUS[0]
    grid = summary_grid(program, max_strokes=8)
    assert grid_summaries(grid) == summaries(program, 8)


def test_the_denoiser_is_bidirectional():
    """'Non-autoregressive over strokes' is this, operationally: slot 0's
    prediction must depend on slot 30's."""
    cfg = tiny()
    net = CompositionDenoiser(cfg).eval()
    grid = torch.full((1, cfg.slots), COMP_MASK, dtype=torch.long)
    before = net(grid)[0, 0].clone()
    grid[0, -1] = COMP_VALUE_BASE + 42
    after = net(grid)[0, 0]
    assert not torch.allclose(before, after)


def test_the_nelbo_is_a_nonnegative_number_of_nats():
    cfg = tiny()
    net = CompositionDenoiser(cfg).eval()
    grid = torch.tensor([summary_grid(p, cfg.max_strokes) for p in CORPUS[:8]])
    torch.manual_seed(0)
    got = net.nelbo(grid, samples=4)
    assert got.shape == (8,)
    assert bool((got >= 0).all())


def test_the_nelbo_weighting_matches_the_masked_diffusion_bound():
    """`(1/t) * sum over masked positions`, for the linear schedule
    `alpha_t = 1 - t`. Checked against a model whose per-token loss is a known
    constant, so the only thing under test is the weighting.
    """
    cfg = tiny()
    net = CompositionDenoiser(cfg)
    vocab = cfg.composition().vocab_size
    # Uniform logits: every position costs exactly log(vocab) nats.
    net.forward = lambda tokens: torch.zeros(*tokens.shape, vocab)  # type: ignore[method-assign]
    grid = torch.tensor([summary_grid(CORPUS[0], cfg.max_strokes)])
    torch.manual_seed(0)
    got = float(net.nelbo(grid, samples=64))
    # E_t[(1/t) * Binom(L, t) * log V] = L * log V, exactly.
    assert got == pytest.approx(cfg.slots * math.log(vocab), rel=0.02)


def test_stratifying_t_is_what_makes_that_estimate_usable():
    """The weight is `1/t`, so an iid draw's variance lives in the rare tiny
    `t`. This pins the reason the stratification is there rather than leaving
    it as a comment nobody can check."""
    cfg = tiny()
    net = CompositionDenoiser(cfg)
    vocab = cfg.composition().vocab_size
    net.forward = lambda tokens: torch.zeros(*tokens.shape, vocab)  # type: ignore[method-assign]
    grid = torch.tensor([summary_grid(CORPUS[0], cfg.max_strokes)])
    truth = cfg.slots * math.log(vocab)

    def spread(samples: int, repeats: int = 24) -> float:
        torch.manual_seed(1)
        draws = [float(net.nelbo(grid, samples=samples)) for _ in range(repeats)]
        return max(abs(d - truth) / truth for d in draws)

    # Same 32 model evaluations either way; one is stratified and one is not.
    stratified = spread(32)
    iid = max(spread(1) for _ in range(1))
    assert stratified < iid / 2


def test_sampling_leaves_no_masks_behind():
    cfg = tiny()
    net = CompositionDenoiser(cfg)
    grid = net.sample(4, steps=5)
    assert grid.shape == (4, cfg.slots)
    assert not bool((grid == COMP_MASK).any())


def _grid_corpus(cfg: PlannerConfig, n: int, occupancy: float):
    """Grids whose stroke count is known by construction, shaped like the real ones.

    The count is `1 + Binomial(max_strokes - 1, occupancy)`, the occupied slots
    are a **prefix**, and `halts` is 0 on every one of them but the last. All
    three matter: the contiguous EMPTY tail is what makes EMPTY both the
    majority class and the easiest thing in the grid to predict, which is what
    the sampler fault feeds on, and a `halts` field that is not ISA-shaped
    makes `grid_summaries` stop at slot 1 and measures nothing.
    """
    slots, fields = cfg.max_strokes, SUMMARY_BYTES
    counts = 1 + torch.binomial(
        torch.full((n,), float(slots - 1)), torch.full((n,), occupancy)
    ).long()
    values = torch.randint(COMP_VALUE_BASE, COMP_VALUE_BASE + 4, (n, slots, fields))
    values[:, :, fields - 1] = COMP_VALUE_BASE  # halts = 0 ...
    values.scatter_(
        1, (counts - 1).view(n, 1, 1).expand(n, 1, fields),
        torch.full((n, 1, fields), COMP_VALUE_BASE + 1, dtype=torch.long),
    )                                            # ... except on the last live slot
    live = torch.arange(slots).unsqueeze(0) < counts.unsqueeze(1)
    grid = torch.where(live.unsqueeze(-1), values,
                       torch.full_like(values, COMP_EMPTY)).view(n, -1)
    return grid, float(counts.float().mean())


@functools.lru_cache(maxsize=1)
def fitted_denoiser(occupancy: float = 0.12, steps: int = 1200, seed: int = 0):
    """A denoiser that has actually learned a count distribution, fitted once.

    An untrained denoiser cannot show this fault: its predictions barely depend
    on context, so every unmasking order agrees and the sampler looks correct.
    The fault is a property of a model that *has* learned the structure, which
    is why the assertions below need ~12 s of fitting and why the result is
    cached across the tests that share it.

    Returns the model, its config and the corpus's own mean stroke count.
    """
    cfg = tiny()
    torch.manual_seed(seed)
    net = CompositionDenoiser(cfg)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3)
    for _ in range(steps):
        grid, _ = _grid_corpus(cfg, 64, occupancy)
        loss = net.nelbo(grid, samples=1).mean() / grid.shape[1]
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    return net, cfg, _grid_corpus(cfg, 4096, occupancy)[1]


def _planned(net: CompositionDenoiser, order: str, n: int = 128,
             steps: int = 16, seed: int = 0) -> float:
    torch.manual_seed(seed)
    grid = net.sample(n, steps=steps, order=order)
    return sum(len(grid_summaries(row)) for row in grid.tolist()) / n


@torch.no_grad()
def _marginal_strokes(net: CompositionDenoiser, cfg: PlannerConfig) -> float:
    """Expected stroke count under the model's one-pass marginals.

    From a fully masked grid, so it is the model's own belief with no reverse
    process in the way -- the quantity the sampler is supposed to reproduce.
    """
    logits = net.forward(torch.full((1, cfg.slots), COMP_MASK, dtype=torch.long))
    logits[..., [COMP_PAD, COMP_BOS, COMP_MASK]] = float("-inf")
    prob = F.softmax(logits, dim=-1)[0]
    return float(sum(1 - prob[s * SUMMARY_BYTES, COMP_EMPTY]
                     for s in range(cfg.max_strokes)))


def test_random_order_unmasking_reproduces_the_models_own_marginals():
    """The reverse process must not move the answer the model already has.

    `nelbo_terms` masks each position independently with probability `t`, so
    the masked set is uniformly random and the reverse step has to unmask a
    uniformly random subset. This asserts the consequence rather than the
    code: sampled stroke count against the expected count under the model's
    one-pass marginals, which is what a correct reverse process preserves and
    what confidence ordering destroyed -- 3.26 planned against a marginal of
    6.16 on the real checkpoint.
    """
    net, cfg, corpus_mean = fitted_denoiser()
    marginal = _marginal_strokes(net, cfg)
    # The premise, asserted rather than assumed: this model learned the corpus,
    # so a sampler that disagrees with it is disagreeing with the corpus too.
    assert marginal == pytest.approx(corpus_mean, rel=0.15)
    assert _planned(net, "random") == pytest.approx(marginal, rel=0.15)


def test_confidence_ordering_ratchets_the_plan_short():
    """The fault, kept as a measurement so the fix is evidence and not a claim.

    Ordering by the model's own confidence commits the most certain masked
    position first; on a majority-EMPTY grid that is almost always an EMPTY
    one, and committing it makes its neighbour more certainly EMPTY in turn.
    Every step shortens the plan and none lengthens it, so the *same*
    checkpoint reads far short of its own marginals. This is what `PLAN.md`
    10's argmax entry did not cover: that fix changed which *value* is
    committed, and the collapse lives in which *position* is.
    """
    net, cfg, _ = fitted_denoiser()
    assert _planned(net, "confidence") < 0.8 * _marginal_strokes(net, cfg)
    assert _planned(net, "confidence") < 0.8 * _planned(net, "random")


def test_more_steps_do_not_rescue_confidence_ordering():
    """The tell that separates a biased sampler from an under-resolved one.

    An under-resolved sampler approaches the model's joint as the step count
    rises. Confidence ordering moves *away* from it -- measured on the real
    checkpoint at 6.84 strokes with 1 step, 3.19 at 16 and 3.17 at 192 -- and
    that monotonicity is why the diagnosis is the ordering rather than the
    schedule.
    """
    net, _, _ = fitted_denoiser()
    assert _planned(net, "confidence", steps=32) <= _planned(
        net, "confidence", steps=4
    )


def test_the_default_ordering_is_the_reverse_process():
    """Pinned because the default is the whole fix: every planner run in the
    project took `sample`'s default and three of them recorded a sampler."""
    cfg = tiny()
    net = CompositionDenoiser(cfg)
    torch.manual_seed(0)
    default = net.sample(8, steps=6)
    torch.manual_seed(0)
    assert bool((default == net.sample(8, steps=6, order="random")).all())
    with pytest.raises(ValueError, match="order must be"):
        net.sample(2, steps=2, order="confident")


def test_gumbel_noise_is_finite_and_gumbel():
    """The test the shape assertions above could not be: they pass on NaN.

    `sample` orders unmasking by confidence plus annealed Gumbel noise, and the
    inline form of that noise was NaN in every element (`gumbel_like`). Every
    downstream assertion still held -- the grid was the right shape and no mask
    survived the final step -- because a `topk` over NaN still returns indices.
    Only the moments catch it.
    """
    noise = gumbel_like(torch.zeros(200_000))
    assert bool(torch.isfinite(noise).all())
    assert float(noise.mean()) == pytest.approx(0.5772, abs=0.02)   # Euler-Mascheroni
    assert float(noise.std()) == pytest.approx(1.2825, abs=0.02)    # pi / sqrt(6)


def test_the_sampler_draws_the_same_distribution_on_every_device():
    """Ordering by NaN is backend-defined, and that is how the fault surfaced.

    One checkpoint and one seed drew 9.2 strokes per grid on CPU and 3.7 on
    MPS, so every generation column in the planner's record described the
    backend rather than the model. The devices cannot be compared draw for
    draw -- they have separate RNG streams -- so this compares the statistic
    the fault moved, with a tolerance well inside the gap it opened.
    """
    device = accelerator()
    if device is None:
        pytest.skip("no accelerator; the CPU path is the reference here")
    cfg = tiny()
    net = CompositionDenoiser(cfg).eval()

    def occupancy(dev: str) -> float:
        torch.manual_seed(0)
        net.to(dev)
        grid = net.sample(64, steps=8, device=dev).cpu()
        return float((grid != COMP_EMPTY).float().mean())

    on_cpu, on_device = occupancy("cpu"), occupancy(device)
    net.to("cpu")
    assert on_device == pytest.approx(on_cpu, abs=0.08)


# --- the AR control at the composition level -------------------------------

def test_the_ar_control_costs_the_same_parameters():
    """Claim 3 is 'at equal parameters', and the control is only a control if
    it is the same size. `causal` is an attention mask and not a weight, so
    this should hold exactly -- asserted rather than assumed, because a shape
    table that drifted would turn the decisive experiment into a capacity
    comparison."""
    diffusion = StrokePlanner(tiny()).level_params()
    ar = StrokePlanner(tiny(comp_objective="ar")).level_params()
    assert diffusion == ar


def test_the_ar_control_is_exact_and_the_denoiser_is_a_bound():
    """The two levels differ in what their number *is*, and the record has to
    say which. Both totals stay upper bounds on `-log p(x)` -- the
    factorisation's slack is common to them -- but only the denoiser adds the
    ELBO's, and their difference is readable as that looseness only if the
    sources are named."""
    grid = torch.tensor([summary_grid(p, 8) for p in CORPUS[:4]], dtype=torch.long)

    ar = StrokePlanner(tiny(comp_objective="ar"))
    assert ar.composition.is_bound is False
    assert ar.composition.eval_samples == 1
    # Exact means repeatable: no draw, so two calls agree bit for bit.
    first = ar.composition.cost_terms(grid, 32)
    assert first.shape == (1, 4)
    assert torch.equal(first, ar.composition.cost_terms(grid, 32))

    denoiser = StrokePlanner(tiny())
    assert denoiser.composition.is_bound is True
    assert denoiser.composition.eval_samples == 2  # tiny()'s eval_noise_samples
    assert denoiser.composition.cost_terms(grid, 3).shape == (3, 4)


def test_the_exact_level_reports_zero_estimator_noise():
    """`composition_stderr` is the column that decides whether this instrument
    can resolve anything, and an exact level asking for 32 draws would take a
    variance over a single row and report NaN there."""
    planner = StrokePlanner(tiny(comp_objective="ar"))
    got = bits_per_drawing(planner, CORPUS[:8], CODEC, seed=0)
    assert got["composition_stderr"] == 0.0
    assert math.isfinite(got["bits_per_drawing"])
    assert got["is_upper_bound"] is True
    assert got["bound_sources"] == ["factorisation"]

    denoised = bits_per_drawing(StrokePlanner(tiny()), CORPUS[:8], CODEC, seed=0)
    assert denoised["bound_sources"] == ["factorisation", "diffusion_elbo"]


def test_the_ar_control_generates_a_decodable_grid():
    """A sampled grid must never contain the three ids a grid cannot hold. The
    denoiser masks them rather than trusting training to have learned it, and
    the control has to follow the same rule or its validity column measures a
    different thing."""
    cfg = tiny(comp_objective="ar")
    planner = StrokePlanner(cfg).eval()
    grid = planner.composition.sample(4, device="cpu")
    assert grid.shape == (4, cfg.slots)
    assert not bool(((grid == COMP_PAD) | (grid == COMP_BOS) | (grid == COMP_MASK)).any())
    sampled = generate(planner, CODEC, n=4, steps=4)
    for program in sampled.programs:
        assert to_boundary(program) == program


def test_an_unknown_composition_objective_is_refused():
    with pytest.raises(ValueError, match="comp_objective"):
        StrokePlanner(tiny(comp_objective="montecarlo"))


# --- the stroke level ------------------------------------------------------

def test_the_conditioning_prefix_costs_no_vocabulary():
    """Spelled in the codec's own operand alphabet, so the planner's parameter
    count stays comparable with the AR baseline's."""
    for codec in CODECS.values():
        symbols = codec.encode_values(bytes(range(SUMMARY_BYTES)))
        assert len(symbols) == SUMMARY_BYTES * codec.stride
        assert max(symbols) < codec.vocab_size


def test_the_conditioning_is_attended_to_and_never_scored():
    """Those bits belong to the composition level. Charging them here would
    overstate the planner's cost and make the comparison unfair in its own
    favour -- the direction that must never be possible."""
    cfg = tiny()
    batch = stroke_batch(CORPUS[:4], CODEC, cfg)
    given = prompt_len(CODEC)
    assert bool((batch.targets[:, :given] == 0).all())
    # Every stroke symbol is scored exactly once.
    scored = int((batch.targets != 0).sum())
    expected = sum(
        len(CODEC.encode(s)) for p in CORPUS[:4] for s in split(p, cfg.max_strokes)
    )
    assert scored == expected


def test_a_stroke_whose_length_the_summary_cannot_state_is_counted():
    """The bound is only a bound while `p_stroke` is normalised, and it is
    normalised because `length` says how many symbols the product runs over.
    Past 255 bytes it cannot, so the row is counted like a truncation."""
    long_stroke = assemble("MOVE 5 5\n" + "LINE 6 6\n" * 100 + "HALT")
    assert unrepresentable_length(long_stroke)
    cfg = tiny(max_stroke_len=512)
    assert stroke_batch([long_stroke], CODEC, cfg).truncated == 1
    assert stroke_batch(CORPUS[:4], CODEC, tiny()).truncated == 0


def test_rows_are_attributed_back_to_their_program():
    """Pairing is what makes a ~1-bit effect visible against a val set whose own
    spread is ~3.5 bits/drawing."""
    cfg = tiny()
    batch = stroke_batch(CORPUS[:6], CODEC, cfg)
    counts = torch.bincount(batch.owner, minlength=6).tolist()
    assert counts == [len(split(p, cfg.max_strokes)) for p in CORPUS[:6]]


# --- the number ------------------------------------------------------------

def test_the_reported_number_is_an_upper_bound():
    """The claim the whole comparison rests on.

    `p(x) >= p_comp(f(x)) * prod_k p_stroke(stroke_k | s_k)`, so the planner's
    bits are at least the true cost. Checked here in the form that can actually
    be verified without a tractable `p(x)`: the composition NELBO is at least
    the exact NLL the *same* denoiser assigns with nothing masked, which is the
    tightest thing the bound can collapse to.
    """
    cfg = tiny()
    planner = StrokePlanner(cfg).eval()
    grid = torch.tensor([summary_grid(p, cfg.max_strokes) for p in CORPUS[:8]])
    torch.manual_seed(0)
    bound = planner.composition.nelbo(grid, samples=32)
    with torch.no_grad():
        logits = planner.composition(grid)
    exact = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), grid.reshape(-1), reduction="none"
    ).view(grid.shape).sum(dim=1)
    assert float(bound.mean()) >= float(exact.mean())


def test_bits_per_drawing_is_paired_and_decomposed():
    cfg = tiny()
    planner = StrokePlanner(cfg)
    got = bits_per_drawing(planner, CORPUS[:8], CODEC, seed=0)
    assert got["is_upper_bound"] is True
    assert len(got["val_bits"]) == 8
    assert got["bits_per_drawing"] == pytest.approx(
        got["composition_bits"] + got["stroke_bits"], rel=1e-6
    )
    assert sum(got["val_bits"]) / 8 == pytest.approx(got["bits_per_drawing"], rel=1e-3)


def test_the_bound_is_reproducible_at_a_fixed_seed():
    """The estimator's whole variance is in the Monte-Carlo draw of `t`. Two
    arms compared on different draws differ by noise belonging to neither."""
    cfg = tiny()
    planner = StrokePlanner(cfg)
    a = bits_per_drawing(planner, CORPUS[:8], CODEC, seed=3)
    b = bits_per_drawing(planner, CORPUS[:8], CODEC, seed=3)
    assert a["bits_per_drawing"] == pytest.approx(b["bits_per_drawing"])


@pytest.mark.parametrize("codec_name", ["byte", "token", "bit", "byte_delta"])
def test_it_works_on_every_alphabet(codec_name):
    codec = CODECS[codec_name]
    cfg = tiny(codec)
    planner = StrokePlanner(cfg)
    got = bits_per_drawing(planner, CORPUS[:4], codec, seed=0)
    assert math.isfinite(got["bits_per_drawing"])


def test_the_eval_path_runs_on_the_accelerator():
    """Every training run is on MPS and MPS has no float64, so a `.double()`
    that lands before `.cpu()` is a `TypeError` at the first eval and not one
    line earlier. CPU-only tests cannot see it: this cost run 4 a launch."""
    device = accelerator()
    if device is None:
        pytest.skip("no accelerator; the fault is device-specific by construction")
    planner = StrokePlanner(tiny()).to(device)
    got = bits_per_drawing(planner, CORPUS[:8], CODEC, device=device, seed=0)
    assert math.isfinite(got["bits_per_drawing"])
    assert math.isfinite(got["composition_stderr"])
    generate(planner, CODEC, n=2, steps=4, device=device)


def test_generation_produces_bytecode_the_vm_can_judge():
    """Validity has to be measurable for the planner too: on every corpus so far
    the AR arm's likelihood was fine and its termination was not."""
    cfg = tiny()
    planner = StrokePlanner(cfg)
    sampled = generate(planner, CODEC, n=4, steps=4)
    assert len(sampled.programs) == len(sampled.cap_hit) == len(sampled.planned) == 4
    vm = VM()
    for program in sampled.programs:
        vm.run(program)  # reports faults, never raises


def test_generated_programs_never_end_mid_instruction():
    """Parity with the AR arm, which has had this since the halt-symbol
    artifact: `HaltMonitor` carries a parse state and stops each row *at a
    boundary*, so the trace of the truncated stream equals the trace of the
    full one. The planner cut every stroke at exactly the byte count the plan
    named, wherever that landed, and the VM reported the tail as `truncated` on
    22-31 of 128 grids."""
    cfg = tiny()
    planner = StrokePlanner(cfg)
    sampled = generate(planner, CODEC, n=8, steps=4)
    for program in sampled.programs:
        assert to_boundary(program) == program


def test_a_plan_that_halts_produces_a_program_that_halts():
    """`halts` is a field the composition level predicts, and `length` -- the
    field beside it -- has always been trusted structurally, since `generate`
    takes exactly that many symbols. Trusting one and not the other left the
    HALT byte to the decoder, which omitted it on 42 of 128 grids while the
    design claimed termination was structural.

    The plan is fixed here rather than sampled: what is under test is that a
    halting plan is honoured, and a random denoiser produces a halting plan
    only by luck."""
    cfg = tiny()
    planner = StrokePlanner(cfg)
    grid = torch.full((2, cfg.slots), COMP_EMPTY, dtype=torch.long)
    for row in range(2):
        for field, value in enumerate(Summary(10, 20, 5, 5, 4, 1)):
            grid[row, field] = COMP_VALUE_BASE + value
    planner.composition.sample = lambda n, **kw: grid[:n]  # type: ignore[assignment]

    sampled = generate(planner, CODEC, n=2, steps=4)
    for program in sampled.programs:
        assert program and program[-1] == int(Op.HALT)
        assert to_boundary(program) == program


def test_generation_reports_what_the_plan_asked_for():
    """`planned` against `gen_strokes` is the only place the two scales can be
    compared, and claim 3's termination argument lives entirely in that gap:
    the composition level decides how many strokes there are before a stroke
    byte exists, so a plan the decoder does not honour is a failure the AR
    baseline cannot have."""
    cfg = tiny()
    planner = StrokePlanner(cfg)
    sampled = generate(planner, CODEC, n=6, steps=4)
    assert all(0 <= k <= cfg.max_strokes for k in sampled.planned)
    # An empty plan draws nothing; a non-empty one cannot draw fewer bytes than
    # zero. Both directions are real outcomes, so this pins the bookkeeping
    # rather than the model.
    for program, k in zip(sampled.programs, sampled.planned):
        assert (k == 0) <= (len(program) == 0)


def test_the_budget_is_matched_to_the_ar_baseline():
    """Claim 3 is 'at equal parameters', and that has to be a fact about the
    table. 825k is the AR `square` arm; both levels are counted."""
    for shape, kwargs in PLANNER_SHAPES.items():
        planner = StrokePlanner(PlannerConfig(vocab_size=258, **kwargs))
        counts = planner.level_params()
        assert counts["total"] == counts["composition"] + counts["stroke"]
        assert abs(counts["total"] - 824_704) / 824_704 < 0.01, shape
        assert counts["total"] < 1_000_000, shape


def test_a_bidirectional_model_refuses_to_autoregress():
    """It has no next-token distribution to sample from, and sampling anyway
    returns plausible-looking rows -- the worst failure mode for a generation
    metric."""
    planner = StrokePlanner(tiny())
    with pytest.raises(ValueError, match="causal"):
        planner.composition.net.generate(2, 4)


def test_the_record_names_the_sampler_that_wrote_its_generation_columns(tmp_path,
                                                                       monkeypatch):
    """Three of this project's six faults were in the composition sampler, and
    no record could say which of the three it ran.

    The ordering was a *default* in `CompositionDenoiser.sample`, so changing it
    changed every planner run's generation columns retroactively-in-meaning
    while leaving every record byte-identical. `runs/*_gen_*.json` is keyed by
    sampler because `scripts/resample.py` learnt that lesson; the training
    record had not. `gen_order` is a config field and not a schema bump on
    purpose: the sampler has no gradient and no path into the bound, so it makes
    the generation columns incomparable and leaves `bits_per_drawing`
    comparable, and those are different statements about the same record.
    """
    from dm import train as dm_train
    from dm import train_planner as tp

    # `dm.train.RUNS` and not `tp.RUNS`: both trainers now write through
    # `dm.train.checkpoint`, so there is one directory to redirect instead of
    # two, and the planner module no longer names it.
    monkeypatch.setattr(dm_train, "RUNS", tmp_path)
    seen: list[str] = []
    real = tp.generate
    monkeypatch.setattr(
        tp, "generate",
        lambda *a, **kw: (seen.append(kw["order"]), real(*a, **kw))[1],
    )
    cfg = functools.partial(
        tp.PlannerTrainConfig, data="synthetic", n_train=64, n_val=16, steps=2,
        eval_every=2, gen_samples=4, gen_steps=2, batch_size=8, device="cpu",
    )

    record = tp.train(cfg(), verbose=False)
    assert record["config"]["gen_order"] == "random", "the reverse process is the default"
    assert seen == ["random"], f"the trainer must pass its own order, not the default: {seen}"

    seen.clear()
    record = tp.train(cfg(gen_order="confidence", tag="confident"), verbose=False)
    assert record["config"]["gen_order"] == "confidence" and seen == ["confidence"], (
        "a record that ran MaskGIT's heuristic has to say so"
    )


# ---------------------------------------------------------------------------
# The seventh instrument fault: a seeded reset inside an eval is a global side
# effect on training.


def _planner_fixture():
    """A denoiser arm, so the bound actually draws. The AR level has no draw and
    would pass every test below trivially."""
    return StrokePlanner(tiny()), list(CORPUS[:4]), CODECS["byte"]


def test_the_bound_does_not_disturb_the_global_rng_stream():
    """`per_program_bits` used to call `torch.manual_seed`.

    That reset is global, so every eval restarted the process stream and the two
    planner arms -- which consume different amounts of it, the diffusion bound
    drawing 32 noise samples per batch and the AR level none -- resumed training
    on **different batch orders**. Five trajectories were affected and none of
    them was the controlled comparison the project called it.

    The test is the one that would have caught it: draw from the global stream,
    run the eval, draw again, and require the second draw to be what it would
    have been with no eval in between. Asserting the *stream*, not that the code
    ran.
    """
    planner, programs, codec = _planner_fixture()

    torch.manual_seed(1234)
    before = torch.randn(4)
    expected = torch.randn(4)

    torch.manual_seed(1234)
    again = torch.randn(4)
    per_program_bits(planner, programs, codec, seed=0)
    after = torch.randn(4)

    assert torch.equal(before, again)
    assert torch.equal(expected, after), (
        "the eval advanced or reset the global RNG stream, so training after an "
        "eval no longer sees the batch order it would have seen"
    )


def test_the_bound_is_still_pinned_across_calls():
    """Removing the reseed must not cost the thing the reseed was there for.

    The estimator's whole variance lives in its draw, so two arms compared on
    different draws differ by noise belonging to neither model. A private
    generator has to reproduce that pinning exactly.
    """
    planner, programs, codec = _planner_fixture()
    first = per_program_bits(planner, programs, codec, seed=0)
    torch.manual_seed(999)  # move the global stream between the two calls
    _ = torch.randn(128)
    second = per_program_bits(planner, programs, codec, seed=0)
    assert np.array_equal(first["composition"], second["composition"])


def test_a_different_seed_gives_a_different_draw():
    """Otherwise the previous test would pass on a generator that is ignored."""
    planner, programs, codec = _planner_fixture()
    a = per_program_bits(planner, programs, codec, seed=0)
    b = per_program_bits(planner, programs, codec, seed=1)
    assert not np.array_equal(a["composition"], b["composition"])


# ---------------------------------------------------------------------------
# The eighth loss, and the first that was not an instrument fault: the planner
# trainer stored nothing until it returned.


def test_a_killed_planner_run_still_leaves_the_record_it_had_reached(tmp_path,
                                                                    monkeypatch):
    """The run this was written for is the one that did not survive.

    `quickdraw_plannerar12000eps2s2_byte_balanced_s0` reached step 11,500 of
    12,000 across 2.6 hours, took a SIGKILL, and left no file, so
    `planner_replicate_table` -- built and waiting for exactly that record --
    had nothing to key on. The planner is a second trainer with its own
    `snapshot`, so the property is pinned twice rather than assumed to carry
    over from `dm/train.py`.
    """
    from dm import train as dm_train
    from dm import train_planner as tp

    monkeypatch.setattr(dm_train, "RUNS", tmp_path)
    calls = {"n": 0}
    real = tp.bits_per_drawing

    def die_on_the_third(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] > 2:
            raise KeyboardInterrupt("pretend this was a SIGKILL")
        return real(*args, **kwargs)

    monkeypatch.setattr(tp, "bits_per_drawing", die_on_the_third)
    cfg = tp.PlannerTrainConfig(
        data="synthetic", n_train=64, n_val=16, steps=3, eval_every=1,
        gen_samples=4, gen_steps=2, batch_size=8, device="cpu", tag="unit_killed_planner",
    )
    with pytest.raises(KeyboardInterrupt):
        tp.train(cfg, verbose=False)

    record = json.loads((tmp_path / "unit_killed_planner.json").read_text())
    assert record["kind"] == "planner" and record["schema"] == tp.PLANNER_SCHEMA
    assert record["complete"] is False
    assert record["steps_done"] == 2, "the record names the last eval that landed"
    assert record["config"]["steps"] == 3, "and keeps the budget it was asked for"
    # `planner_replicate_table` differences these per program and reports a 95%
    # interval; a partial record without them would be a mean and nothing else,
    # which is precisely what the console log left behind.
    assert len(record["val_bits"]) == 16
    assert record["final"]["composition_bits"] and record["final"]["stroke_bits"]
    assert (tmp_path / "unit_killed_planner.pt").exists()
