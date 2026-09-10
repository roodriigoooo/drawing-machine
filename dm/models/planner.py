"""Claim 3: non-autoregressive over strokes x autoregressive over bits.

`PLAN.md` section 5 states the argument in four sentences and
`docs/diffusion-strategy.md` at length: **the denoising objective decomposes by
scale and autoregression structurally cannot.** High-noise steps train coarse
layout, low-noise steps train stroke shape; AR over a serialised stream predicts
position 400 exactly the way it predicted position 4. This module is that
argument built.

## The factorisation, and why its number can be believed

A program is a sequence of strokes (`dm.isa.strokes`), and each stroke has a
six-byte summary `s_k = f(stroke_k)` -- start, extent, cost, terminal. Write
`s = f(x)`. Then, for **any** distributions `p_comp` over summaries and `p_stroke` over
stroke bytes,

    p(x)  >=  p_comp(f(x)) * prod_k p_stroke(stroke_k | s_k)

because the right-hand side is one term of the marginalisation over `s` and
every term is non-negative. Taking negative logs, **the planner's bits/drawing
is an upper bound on the cost of transmitting the drawing under this model.**
Layer the composition level's own diffusion bound on top -- an ELBO is an upper
bound on `-log p_comp` -- and the reported number is still an upper bound.

That direction is the whole point. The AR baseline reports an *exact* NLL. A
comparison between an exact number and an optimistic one proves nothing; a
comparison between an exact number and a **conservative** one is decisive in the
direction claim 3 needs. If the planner reads lower, it beat the baseline with a
handicap.

**The bound needs `p_stroke` to be a proper distribution, and this is where it
comes from.** The decoder has no end-of-stroke symbol, so a chain rule over its
symbols would normalise over *prefixes* rather than strings -- which would make
the total smaller than a real model's and turn the "bound" into an optimistic
number, the one direction that must never be possible. It is proper because the
stroke's **length is part of the conditioning**: `s_k.length` fixes how many
symbols the product runs over, so the chain rule normalises over strings of
exactly that length. `length` earns its place in the summary twice over -- once
here and once at generation time.

The two ways that can fail are counted, never assumed away: a stroke longer than
the decoder's context, and a stroke longer than 255 bytes, which `length` cannot
express. Both land in `stroke_truncated`, for the same reason `max_len`
truncation is a column in every AR record.

Two honest costs of this construction, both stated in the record:

- The stroke decoder can spend mass on strokes whose summary is not `s_k`,
  which is exactly the slack the inequality above absorbs. It is not a bug to
  be fixed; it is the reason the bound is a bound.
- Strokes are conditionally independent given their summaries. That is a
  modelling assumption, and a strong one -- it is also precisely the scale
  separation being claimed, so a planner that loses to the AR baseline is
  evidence against the hypothesis rather than against the implementation.

## The composition level

Absorbing-state (masked) discrete diffusion over the summary grid, which is
`max_strokes x SUMMARY_BYTES` u8 tokens plus an EMPTY marker for unused slots --
so the stroke *count* is part of what the level models rather than a separate head.

Forward process: mask each token independently with probability `t ~ U(0, 1]`.
Reverse model: one **bidirectional** pass predicting the clean token at every
masked position, which is what "non-autoregressive over strokes" means
operationally. With the linear schedule `alpha_t = 1 - t`, the continuous-time
NELBO reduces to

    E_{t ~ U(0,1]} [ (1/t) * sum_{i masked} -log p_theta(x_i | x_t) ]

(Austin et al.'s D3PM absorbing kernel; the reduction is the one Sahoo et al.
report for masked diffusion LMs). No time embedding: `x_t` carries its own mask
ratio, and the weight in front is where `t` enters. That is one fewer thing to
tune and, at this budget, thousands of parameters that go into layers instead.

## The stroke level

The AR baseline verbatim -- same `DrawingLM`, same codec, same alphabet -- with
the stroke's summary spelled in the codec's own operand alphabet
(`Codec.encode_values`) as a fixed-length prefix that is conditioned on and
never scored. Conditioning therefore costs **zero** extra vocabulary, so the
planner's parameter count is comparable with the baseline's rather than
comparable-after-an-adjustment, and the granularity axis applies to the prefix
as it does to everything else.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..isa.codec import BOS, PAD, Codec
from ..isa.spec import CANVAS, Op
from ..isa.strokes import SUMMARY_BYTES, Summary, split, summaries, to_boundary
from .transformer import Config, DrawingLM

#: Composition-level vocabulary: PAD, BOS (unused, kept so ids line up with the
#: codec's convention and an off-by-two cannot creep in), the 256 byte values,
#: then EMPTY and MASK.
COMP_PAD, COMP_BOS = PAD, BOS
COMP_VALUE_BASE = 2
COMP_EMPTY = COMP_VALUE_BASE + 256
COMP_MASK = COMP_EMPTY + 1
COMP_VOCAB = COMP_MASK + 1

#: Keeps both logs in `gumbel_like` in range. At 1e-9 the noise is bounded by
#: +-20.7, which is far outside any log-probability the ordering compares.
GUMBEL_EPS = 1e-9


def gumbel_like(x: Tensor) -> Tensor:
    """Standard Gumbel noise shaped like `x`.

    Its own function because the inline form is a precedence trap, and this
    project has now paid for it once. In

        -torch.log(-torch.log(u).clamp_min(eps))

    the unary minus binds *after* the method call, so the clamp lands on
    `log(u)` -- which is non-positive everywhere -- and drives every element to
    `eps`. The negation then hands the outer log a negative number and the
    result is **NaN in every element, on every device**. Nothing raises: the
    NaNs flow into the `topk` that orders unmasking, and ordering by NaN is
    backend-defined, so the composition sampler drew 9.2 strokes per grid on
    CPU and 3.7 on MPS from one checkpoint and one seed (truth: 6.3).

    Clamping the *input* instead keeps both logs in range with no minus sign in
    the way, and `test_gumbel_noise_is_finite_and_gumbel` pins it.
    """
    u = torch.rand_like(x).clamp(GUMBEL_EPS, 1.0 - GUMBEL_EPS)
    return -torch.log(-torch.log(u))


@dataclass
class PlannerConfig:
    """Both levels, and the split of the budget between them.

    Defaults are `dm.train_planner.PLANNER_SHAPES["balanced"]`, which totals
    825k on the byte alphabet -- the AR `square` arm to within 0.1%, so claim
    3's "at equal parameters" is checkable from the record. The split itself is
    a knob and not a finding; `PLANNER_SHAPES` carries the other two rungs.
    """

    vocab_size: int = 258            # the codec's, for the stroke decoder
    max_strokes: int = 32
    max_stroke_len: int = 256        # symbols, the decoder's context

    comp_d_model: int = 80
    comp_layers: int = 5
    comp_heads: int = 4

    stroke_d_model: int = 88
    stroke_layers: int = 4
    stroke_heads: int = 4

    #: Stratified samples of `t` per evaluation pass. Training uses one -- the
    #: gradient only needs an unbiased estimate -- but a *reported* bound needs
    #: a tight one, and the estimator's variance is entirely in this draw.
    #: Measured on an untrained planner over 1,000 programs: 2.94 bits of
    #: standard error at 8 samples against 1.57 at 32, where the corpus
    #: resolution floors this instrument has to clear are ~1-2.5 bits. The
    #: composition net is the small half of the budget and the pass is
    #: evaluation-only, so 32 is cheap; `composition_stderr` is reported either
    #: way, so the choice is checkable rather than trusted.
    eval_noise_samples: int = 32

    #: `"diffusion"` or `"ar"`, and this is claim 3's own hypothesis made into
    #: a switch. The claim is that *non-autoregressive over strokes* beats
    #: autoregression at equal parameters, and the composition level is where
    #: that is claimed -- so the decisive control is the same grid, the same
    #: budget, the same corpus and the same trainer, differing only in whether
    #: the level factorises left-to-right.
    #:
    #: It is also the cheapest control the project can run: 424k parameters
    #: over a fixed 192-token grid, against the stroke decoder's long rows.
    #:
    #: Neither setting changes the parameter count -- `causal` is a mask, not a
    #: weight -- so `PLANNER_SHAPES` is untouched and a diffusion row and an AR
    #: row are comparable without an adjustment.
    comp_objective: str = "diffusion"

    @property
    def slots(self) -> int:
        return self.max_strokes * SUMMARY_BYTES

    def composition(self) -> Config:
        # `abs_pos` on for both. RoPE is relative and this grid is an absolute
        # address -- position 5 is a `halts` flag and position 0 is an x
        # coordinate -- and that is a property of the *grid*, not of the
        # objective, so it does not change with `comp_objective`.
        return Config(
            vocab_size=COMP_VOCAB, d_model=self.comp_d_model,
            n_layers=self.comp_layers, n_heads=self.comp_heads,
            max_len=self.slots, causal=self.comp_objective == "ar", abs_pos=True,
        )

    def stroke(self) -> Config:
        return Config(
            vocab_size=self.vocab_size, d_model=self.stroke_d_model,
            n_layers=self.stroke_layers, n_heads=self.stroke_heads,
            max_len=self.max_stroke_len,
        )


def unrepresentable_length(stroke: bytes) -> bool:
    """True when `Summary.length` cannot state this stroke's true length.

    The summary is u8, so 255 bytes is the ceiling. Past it the decoder's
    product no longer runs over the number of symbols the conditioning names,
    and `p_stroke` stops being normalised -- which would make the reported total
    optimistic rather than conservative. Rare on every corpus in the project
    (Tier A's longest stroke is ~40 bytes), and counted rather than trusted.
    """
    return len(stroke) > CANVAS - 1


def prompt_len(codec: Codec) -> int:
    """Symbols of conditioning in front of every stroke.

    Five bytes in the codec's own alphabet, so it is 5 on the stride-1 arms and
    40 on the bit arm -- the granularity axis reaching the conditioning too,
    which is the point of spelling it in-alphabet rather than in a private one.
    """
    return SUMMARY_BYTES * codec.stride


def summary_grid(program: bytes, max_strokes: int) -> list[int]:
    """One program's summary grid as composition-level token ids.

    Unused slots are EMPTY in all six fields, never PAD: PAD is "outside the
    tensor" and EMPTY is "this drawing has no stroke here", and the second is a
    fact the level has to predict. Conflating them would let the model score
    the stroke count for free.
    """
    grid = [COMP_EMPTY] * (max_strokes * SUMMARY_BYTES)
    for k, summary in enumerate(summaries(program, max_strokes)):
        for j, value in enumerate(summary):
            grid[k * SUMMARY_BYTES + j] = COMP_VALUE_BASE + value
    return grid


class CompositionDenoiser(nn.Module):
    """Masked diffusion over the summary grid. Bidirectional, time-free.

    `DrawingLM` with `causal=False` rather than a second transformer: RMSNorm,
    RoPE and SwiGLU are the parts of this project that must not have two
    implementations, and the tied-embedding budget argument applies here too.
    """

    def __init__(self, cfg: PlannerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.net = DrawingLM(cfg.composition())

    def n_params(self) -> int:
        return self.net.n_params()

    def forward(self, tokens: Tensor) -> Tensor:
        return self.net(tokens)

    #: The composition factor is a *bound* on `-log p_comp`, not the thing
    #: itself. Reported, because the slack in the planner's total has two
    #: sources -- the factorisation and this -- and only one of them is
    #: common to both objectives.
    is_bound = True

    @property
    def eval_samples(self) -> int:
        """Draws `cost_terms` needs for a *reported* number. Training uses one."""
        return max(1, self.cfg.eval_noise_samples)

    def cost_terms(self, grid: Tensor, samples: int = 1,
                   generator: torch.Generator | None = None) -> Tensor:
        """`(samples, B)` estimates of the composition cost in nats.

        The one method `StrokePlanner` calls, so the AR control is a
        substitution rather than a second code path through the trainer and the
        record. For this class it is the NELBO's stratified draws; for
        `CompositionAR` it is one exact number with no draw to average.

        `generator` pins the draw *without* touching the process's stream --
        see `nelbo_terms`.
        """
        return self.nelbo_terms(grid, samples, generator)

    def nelbo(self, grid: Tensor, samples: int = 1,
              generator: torch.Generator | None = None) -> Tensor:
        """Per-drawing NELBO in **nats**, `(B,)`.

        Three variance decisions, and they matter more here than the usual
        amount: this number is *reported*, and the project's own history is a
        list of effects that turned out to be smaller than the noise on the
        instrument measuring them.

        - **One `t` per row**, not one per batch. The estimator averages over
          `t` regardless and per-row draws cost nothing.
        - **Stratified `t`**: sample `j` uses `t = (j + u) / samples`, so the
          draws cover `(0, 1]` evenly instead of clumping. The weight is `1/t`,
          so a naive draw's variance is dominated by the rare tiny `t` -- 64
          iid samples still read ~9% high on a model whose answer is known
          exactly, which is several times any effect claim 3 is looking for.
          At `samples=1` this is identical to an iid draw.
        - **Clamped at 1e-3**, because `t = 0` masks nothing and divides by
          zero. It is a region of measure 1e-3 and contributes nothing.

        Pinned by a `generator` rather than by `torch.manual_seed`. The
        earlier version took a seed and reseeded the process, on the grounds
        that a generator carries a device and MPS had none; MPS generators work
        now, and the reasoning was wrong even then. **A seeded reset inside an
        eval is a global side effect on training** -- it made claim 3's two arms
        resume on different batch orders, which is the project's seventh
        instrument fault. Pinning an estimator's draw is right; pinning it by
        reseeding the process is how the pin reaches the optimiser.
        """
        return self.nelbo_terms(grid, samples, generator).mean(dim=0)

    def nelbo_terms(self, grid: Tensor, samples: int = 1,
                    generator: torch.Generator | None = None) -> Tensor:
        """The individual `(samples, B)` estimates behind `nelbo`.

        Kept separable so the *reported* bound can carry its own Monte-Carlo
        standard error. A bound whose noise is not stated cannot be compared
        against an exact NLL: this project's recurring failure is an effect
        smaller than the instrument measuring it, and an unstated estimator
        variance is that failure with no column to catch it.
        """
        out = grid.new_zeros((samples, grid.shape[0]), dtype=torch.float32)
        for stratum in range(samples):
            u = torch.rand(grid.shape[0], 1, device=grid.device, generator=generator)
            t = ((stratum + u) / samples).clamp_min(1e-3)
            masked = torch.rand(grid.shape, device=grid.device, generator=generator) < t
            # A row with nothing masked contributes zero, which is correct and
            # is also why the weight below cannot be folded into the mean.
            noised = grid.masked_fill(masked, COMP_MASK)
            logits = self.forward(noised)
            nll = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), grid.reshape(-1),
                reduction="none",
            ).view(grid.shape)
            out[stratum] = (nll * masked).sum(dim=1) / t.squeeze(1)
        return out

    @torch.no_grad()
    def sample(self, n: int, steps: int = 16, device: str | torch.device = "cpu",
               temperature: float = 1.0, order: str = "random") -> Tensor:
        """Iterative unmasking: the reverse process of the kernel above.

        Every slot starts masked; each step **samples** a value for every masked
        slot and commits the share the cosine schedule allows.

        **Which slots get committed is decided uniformly at random, because
        that is what the forward process defines.** `nelbo_terms` masks each
        position independently with probability `t`, so the set of positions
        still masked at time `t` is a uniformly random subset and the reverse
        step must unmask a uniformly random subset of it. Ordering by the
        model's own confidence is MaskGIT's heuristic for image tokens; it is
        not this objective's reverse process, and importing it turns the
        sampler into a greedy search over the joint wearing the reverse
        process's clothes.

        **On this grid that heuristic is a ratchet, and it was worth 2.9
        strokes.** 86% of slots are EMPTY, so the most confident masked
        position is almost always one the model wants to fill with EMPTY.
        Committing it raises the (correct) posterior for EMPTY at its
        neighbour, which is then the most confident position, and so on: every
        step is another chance to shorten the plan and none to lengthen it.
        Measured on `quickdraw_plannerdiff24000eps2_byte_balanced_s0`, one
        checkpoint, five seeds, n=256:

        | ordering | strokes planned | `gen_length_emd` |
        |---|---|---|
        | confidence + annealed Gumbel | 3.26 | 25.08 +- 1.80 |
        | confidence + un-annealed Gumbel | 3.29 | ~19-28 |
        | **uniform random** | **6.02** | **4.59 +- 0.98** |

        against a val truth of 6.295 and an AR composition level's 6.31 /
        3.78 +- 0.97. The tell that it is the *ordering* and not the noise:
        more steps makes confidence ordering monotonically worse -- 6.84
        strokes at 1 step, 3.19 at 16, 3.17 at 192 -- because each step is one
        more click of the ratchet, where an under-resolved sampler would
        improve. The model itself was never wrong: its one-pass marginals from
        a fully masked grid sum to 6.16 expected strokes against the corpus's
        6.29.

        **Sampling and not argmax, which is a separate fix and still load
        bearing.** Committing the arg-max of each slot is mode-seeking and on
        this grid the mode is EMPTY everywhere: a planner that produced 93.75%
        valid programs from *true* summaries produced **zero strokes** on all
        32 sampled grids. That fix changed *what* is committed; this one
        changes *which position*, and the collapse lived in the second.

        `order="confidence"` is kept because three planner runs used it and
        their retracted columns are only reproducible through it. It is the
        measured failure, not an alternative.
        """
        if order not in ("random", "confidence"):
            raise ValueError(f"order must be 'random' or 'confidence', got {order!r}")
        grid = torch.full((n, self.cfg.slots), COMP_MASK, dtype=torch.long, device=device)
        length = grid.shape[1]
        for step in range(steps):
            masked = grid == COMP_MASK
            if not bool(masked.any()):
                break
            logits = self.forward(grid)
            # The three ids a grid can never contain: the marker itself, and
            # the two special ids kept only so token numbering matches the
            # codec's. Masking them here rather than trusting training to
            # learn it keeps a sampled grid decodable by construction.
            logits[..., [COMP_PAD, COMP_BOS, COMP_MASK]] = float("-inf")
            scaled = logits / max(temperature, 1e-6)
            choice = torch.multinomial(
                F.softmax(scaled, dim=-1).view(-1, scaled.shape[-1]), 1
            ).view(grid.shape)
            if order == "random":
                score = torch.rand_like(scaled[..., 0])
            else:
                score = F.log_softmax(scaled, dim=-1).gather(
                    -1, choice.unsqueeze(-1)
                ).squeeze(-1)
                # Gumbel noise annealed to zero over the schedule. It does not
                # rescue the ordering -- un-annealed noise plans 3.29 strokes
                # against annealed 3.26 -- because any probability-weighted
                # order walks the same ratchet.
                anneal = 1.0 - (step + 1) / steps
                if anneal > 0:
                    score = score + anneal * gumbel_like(score)

            # Cosine schedule on how many slots stay masked, so the count
            # commits smoothly and the final step leaves nothing behind.
            remaining = int(length * math.cos(math.pi / 2 * (step + 1) / steps))
            keep = max(1, int(masked.sum(dim=1).max().item()) - remaining)
            score = score.masked_fill(~masked, float("-inf"))
            take = score.topk(min(keep, length), dim=1).indices
            commit = torch.zeros_like(masked).scatter_(1, take, True) & masked
            grid = torch.where(commit, choice, grid)
        return grid.masked_fill(grid == COMP_MASK, COMP_EMPTY)


class CompositionAR(nn.Module):
    """The composition level as plain autoregression over the same grid.

    **This is claim 3's hypothesis given something to beat.** Section 5 of
    `PLAN.md` argues that the denoising objective decomposes by scale and
    autoregression structurally cannot, and the composition level is where that
    is claimed. Every other arm in the project tests it indirectly; this tests
    it at the level it is made, on the same 192-token grid, the same corpus and
    the same 424k parameters -- `causal` is a mask, not a weight.

    Three outcomes, all of them worth having:

    - **AR near the denoiser.** The masked-diffusion bound is tight and the
      level is simply hard, so the planner's total is honest and its gap to the
      AR baseline is a modelling gap.
    - **AR well below.** The non-autoregressive factorisation is losing where
      it is supposed to win, and claim 3 is refuted at its cheapest test rather
      than after another 3-hour planner run.
    - **AR well above.** The first direct support the claim has had.

    The number is **exact**, and that changes what the planner's total means
    rather than what it is. `p(x) >= p_comp(f(x)) * prod_k p_stroke(...)` holds
    for any `p_comp`, so the total is still an upper bound -- but its slack is
    now the factorisation alone, where the denoiser adds the ELBO's on top.
    `bound_sources` in the record names which of the two a row carries, because
    "upper bound" without that is two different numbers in one column, the
    fault this project has found at four levels.

    A left-to-right order over a grid of fields is a real modelling choice and
    not a neutral control: the raster order here is slot-major, field-minor, so
    a stroke's six fields are contiguous and a stroke is predicted from every
    earlier stroke in full. That is the order the summaries are laid out in and
    the one a hierarchical decoder would use, which makes it the fair opponent
    rather than a strawman.
    """

    #: The factorisation's slack is the caller's; this level adds none of its
    #: own, so a planner built on it reports a tighter bound for the same model.
    is_bound = False

    @property
    def eval_samples(self) -> int:
        """One. There is no draw to average, and asking for 32 would take a
        variance over a single row -- NaN, in the column that decides whether
        this instrument can resolve anything."""
        return 1

    def __init__(self, cfg: PlannerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.net = DrawingLM(cfg.composition())

    def n_params(self) -> int:
        return self.net.n_params()

    def forward(self, tokens: Tensor) -> Tensor:
        return self.net(tokens)

    def cost_terms(self, grid: Tensor, samples: int = 1,
                   generator: torch.Generator | None = None) -> Tensor:
        """`(1, B)` exact NLL in nats. `samples` and `generator` are ignored.

        Accepted rather than rejected because the caller reports a standard
        error either way, and an estimator with no variance should say so with
        a zero rather than by raising -- `composition_stderr` reading 0.000
        beside the denoiser's 0.496 is the comparison, stated in the column.
        `generator` is accepted for the same reason: the two levels have to be
        substitutable at the call site or the trainer grows a second path.
        """
        del samples, generator
        inputs = torch.cat(
            [torch.full((grid.shape[0], 1), COMP_BOS, dtype=torch.long,
                        device=grid.device),
             grid[:, :-1]],
            dim=1,
        )
        logits = self.forward(inputs)
        nll = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), grid.reshape(-1), reduction="none",
        ).view(grid.shape)
        return nll.sum(dim=1).unsqueeze(0)

    @torch.no_grad()
    def sample(self, n: int, steps: int = 16, device: str | torch.device = "cpu",
               temperature: float = 1.0, order: str = "random") -> Tensor:
        """One left-to-right pass. `steps` and `order` are accepted and ignored.

        There is no masked set to choose from, so there is no ordering to get
        wrong -- which is the whole of why this level's stroke count was right
        from its first eval while the denoiser's never was.

        The grid is fixed at `slots` tokens, so there is nothing to terminate
        and no monitor: the level emits exactly the addresses it has. The three
        ids a grid can never contain are masked for the same reason the
        denoiser masks them -- a sampled grid stays decodable by construction
        rather than by training having learned it.
        """
        grid = self.net.generate(
            n, self.cfg.slots, bos=COMP_BOS, temperature=temperature,
            device=device, forbid=(COMP_PAD, COMP_BOS, COMP_MASK),
        )
        return grid[:, -self.cfg.slots :]


#: The composition level, by name. `StrokePlanner` reads it rather than
#: branching, so adding a third objective is a table entry.
COMPOSITION = {"diffusion": CompositionDenoiser, "ar": CompositionAR}


class StrokePlanner(nn.Module):
    """The two levels, and the bound that makes their sum reportable.

    Deliberately *not* end-to-end: the levels share no gradient, because the
    summary is a deterministic function of the data and each level's target is
    fully observed. That is what lets them be trained on different corpora --
    stroke level on QuickDraw, composition level on Tier D -- which is the
    design and not a workaround (`docs/diffusion-strategy.md` 5.3).
    """

    def __init__(self, cfg: PlannerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.comp_objective not in COMPOSITION:
            raise ValueError(
                f"comp_objective must be one of {sorted(COMPOSITION)}, "
                f"got {cfg.comp_objective!r}"
            )
        self.composition = COMPOSITION[cfg.comp_objective](cfg)
        self.decoder = DrawingLM(cfg.stroke())

    def n_params(self, trainable_only: bool = True) -> int:
        return sum(
            p.numel() for p in self.parameters()
            if p.requires_grad or not trainable_only
        )

    def level_params(self) -> dict[str, int]:
        """Both counts, always reported. A hierarchical model whose parameters
        are quoted for one level only is the StrokeNUWA objection this project
        raises against others (`PLAN.md` section 4)."""
        return {
            "composition": self.composition.n_params(),
            "stroke": self.decoder.n_params(),
            "total": self.n_params(),
        }


def grid_summaries(row: list[int]) -> list[Summary]:
    """A sampled grid row back into summaries, stopping at the first empty slot.

    A slot is empty when all six fields are EMPTY. A *partly* empty slot is
    kept, with the empty fields read as zero: it is a malformed prediction, and
    the project's rule is that malformed output is measured in the VM rather
    than discarded before it gets there (`dm/vm/interp.py`).
    """
    out: list[Summary] = []
    for start in range(0, len(row), SUMMARY_BYTES):
        slot = row[start : start + SUMMARY_BYTES]
        if all(v == COMP_EMPTY for v in slot):
            break
        out.append(Summary(*(max(0, v - COMP_VALUE_BASE) % 256 for v in slot)))
        if out[-1].halts:
            break  # the level said this stroke ends the drawing
    return out


class Sampled(NamedTuple):
    """Programs, and which of them ran into the decoder's context.

    `cap_hit` exists so the planner can report `gen_truncated` in the same
    column as the AR arm, and it is per *program* because that is the unit the
    column is in. The two arms cap different things -- the AR arm caps a whole
    program at `max_new`, the planner caps each stroke at `max_stroke_len` --
    so the number is computed where the cap is known rather than inferred
    afterwards from a length that has already been cut.

    `planned` is how many strokes the composition level committed to before a
    single stroke byte existed. Reported beside the count that came back
    because the two answer different questions: the plan is what the coarse
    scale decided, and the difference between them is how much of that decision
    the decoder threw away.
    """

    programs: list[bytes]
    cap_hit: list[bool]
    planned: list[int]


@torch.no_grad()
def generate(
    planner: StrokePlanner,
    codec: Codec,
    n: int = 128,
    steps: int = 16,
    device: str | torch.device = "cpu",
    temperature: float = 1.0,
    top_k: int | None = 40,
    order: str = "random",
    forbid_specials: bool = False,
) -> Sampled:
    """Sample the layout, then each stroke, then concatenate. Returns bytecode.

    **Stroke length and the end of the drawing both come from the summary**,
    which is why `length` and `halts` are two of the six fields. There is no
    end-of-stroke symbol and no per-stroke halt monitor: the composition level
    decides how many strokes there are, how long each is, and which one ends
    the drawing; the decoder fills them in.

    Termination is therefore *structural* here and *sampled* in the AR
    baseline, and that is a claim-3 prediction rather than a convenience. The
    baseline's standing failure is exactly this axis -- HALT over-assigned
    1.5-2.6x at instruction boundaries, a 22% length undershoot with no
    sampling involved (`PLAN.md` 9.5a) -- and a design that decides length at
    the coarse scale cannot make that particular mistake. If `gen_length_emd`
    does not improve, the scale argument is weaker than section 5 claims.

    **It improves, and the improvement is the factorisation's rather than the
    denoiser's.** One checkpoint per arm, five seeds, n=256: the flat AR
    baseline reads 10.40 +- 4.48, a diffusion composition level 4.59 +- 0.98
    and an AR one 3.78 +- 0.97. Both composition levels beat the flat arm ~2.5x
    on the mean and ~4.6x on the spread, and they do not separate from each
    other -- so what wins is deciding termination once at the coarse scale, not
    the objective that decides it.

    `order` reaches `CompositionDenoiser.sample` and is the fault that hid all
    of this until 2026-08-08; the AR level accepts and ignores it.

    `forbid_specials` removes PAD/BOS from stroke-decoder output in side reports.
    It defaults false so a reporting fix cannot change training-time eval RNG
    consumption and thereby move later optimisation batches.
    """
    given = prompt_len(codec)
    grid = planner.composition.sample(n, steps=steps, device=device,
                                      temperature=temperature, order=order)
    per_row = [grid_summaries(row) for row in grid.cpu().tolist()]

    planned = [len(row) for row in per_row]
    flat = [(i, s) for i, row in enumerate(per_row) for s in row]
    if not flat:
        return Sampled([b""] * n, [False] * n, planned)
    prompts = torch.tensor(
        [codec.encode_values(s.to_bytes()) for _, s in flat], dtype=torch.long
    )
    lengths = [max(1, s.length) for _, s in flat]
    room = planner.cfg.max_stroke_len - given
    symbols = planner.decoder.generate(
        len(flat), min(room, max(lengths) * codec.stride), prompt=prompts,
        temperature=temperature, top_k=top_k, device=device, prime_monitor=False,
        forbid=(PAD, BOS) if forbid_specials else (),
    ).cpu()

    programs: list[bytearray] = [bytearray() for _ in range(n)]
    cap_hit = [False] * n
    for (row, summary), length, out in zip(flat, lengths, symbols):
        want = length * codec.stride
        end = given + min(want, symbols.shape[1] - given)
        # The plan asked for more symbols than the decoder's context holds, so
        # this stroke is shorter than its own summary says. Rare by
        # construction -- `length` is u8 and the context is 250 bytes on the
        # stride-1 arms -- and counted rather than assumed rare.
        cap_hit[row] |= end - given < want
        stroke = to_boundary(codec.decode(out[given:end].tolist()))
        # The plan said this stroke ends the drawing, so the drawing ends. That
        # is what "termination is structural here and sampled in the AR
        # baseline" means operationally, and until now only half of it was
        # built: `halts` decided how many strokes to take and then left the
        # HALT *byte* to the decoder, which omitted it on 42 of 128 grids.
        #
        # This writes a model output rather than inventing one. `halts` is a
        # field the composition level predicted, and `length` -- the field
        # beside it -- has always been trusted structurally, since `generate`
        # takes exactly that many symbols. Trusting one and not the other was
        # the inconsistency.
        #
        # It does mean `no_halt` stops being informative for this arm, by
        # construction rather than by merit. `gen_length_emd` is the column
        # that carries termination here, which is what `PLAN.md` 9.10 states
        # the prediction in.
        if summary.halts and (not stroke or stroke[-1] != int(Op.HALT)):
            stroke += bytes([int(Op.HALT)])
        programs[row] += stroke
    return Sampled([bytes(p) for p in programs], cap_hit, planned)


@dataclass
class StrokeBatch:
    """Every stroke of every program, flattened into rows.

    `owner` is the row's program index, and it is what makes the planner's
    number *paired* with the AR baseline's: bits are scattered back to the
    program they came from, so the two models can be differenced per program
    the way every codec arm already is. Without it the comparison would be
    between two means on a val set whose own spread is ~3.5 bits/drawing --
    several times any effect under test (`PLAN.md` 10).
    """

    inputs: Tensor
    targets: Tensor
    owner: Tensor
    truncated: int


def stroke_batch(programs: list[bytes], codec: Codec, cfg: PlannerConfig) -> StrokeBatch:
    """A row is `BOS`, the summary, then the stroke.

    That order is the sampler's, not a preference: `DrawingLM.generate` puts
    BOS first and treats everything after it as the prompt, so training in this
    layout means the decoder is conditioned at generation time exactly as it was
    at training time. Targets are PAD across the conditioning, which is attended
    to and never scored -- those bits belong to the composition level, and
    charging them twice would overstate the planner's cost.
    """
    given = prompt_len(codec)
    room = cfg.max_stroke_len - given - 1  # -1 for BOS
    rows: list[tuple[int, list[int]]] = []
    truncated = 0
    for index, program in enumerate(programs):
        for stroke in split(program, cfg.max_strokes):
            encoded = codec.encode(stroke)
            summary = Summary.of(stroke)
            # Two ways a stroke stops being described by its own summary, and
            # both break the *properness* the bound rests on -- see
            # `unrepresentable_length` above. Counted together, because the
            # consequence is the same: bits/drawing stops being comparable.
            truncated += len(encoded) > room or unrepresentable_length(stroke)
            rows.append((
                index, [BOS, *codec.encode_values(summary.to_bytes()), *encoded[:room]]
            ))

    width = max(len(r) for _, r in rows)
    inputs = torch.full((len(rows), width), PAD, dtype=torch.long)
    targets = torch.full((len(rows), width), PAD, dtype=torch.long)
    for i, (_, row) in enumerate(rows):
        inputs[i, : len(row)] = torch.tensor(row, dtype=torch.long)
        # Targets are inputs shifted left by one, PAD everywhere the loss must
        # not reach: BOS, the summary, and the final position, which has nothing
        # after it to predict.
        targets[i, given : len(row) - 1] = inputs[i, given + 1 : len(row)]
    return StrokeBatch(
        inputs=inputs[:, :-1],
        targets=targets[:, :-1],
        owner=torch.tensor([i for i, _ in rows], dtype=torch.long),
        truncated=truncated,
    )


@torch.no_grad()
def per_program_bits(
    planner: StrokePlanner,
    programs: list[bytes],
    codec: Codec,
    device: str | torch.device = "cpu",
    batch_size: int = 32,
    seed: int | None = 0,
) -> dict:
    """Bits to transmit each program under the planner -- an **upper bound**.

    Comparable with `dm.eval.metrics.per_program_bits` and with nothing else:
    both are the cost of transmitting the same program under a model, in the one
    unit every codec shares. The two components are kept apart because they
    answer different questions -- `composition` is what the layout costs and is
    where claim 3 expects to win, `stroke` is what the shapes cost and should
    roughly match the AR baseline's spend on the same bytes.

    `seed` fixes the diffusion bound's Monte-Carlo draw. The estimator's whole
    variance lives in that draw, so two arms compared on different draws differ
    by noise that has nothing to do with either model.

    **It is pinned with a private `torch.Generator`, never by reseeding the
    process.** The earlier version called `torch.manual_seed` here, which is a
    global side effect: every eval restarted the process stream and the two
    planner arms then consumed different amounts of it, so they resumed training
    on *different batch orders*. Sharing no gradient is not sharing no RNG
    stream. That fault is why `PLANNER_SCHEMA` is 2 and why no record written
    before it is comparable with one written after.
    """
    planner.eval()
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
    composition = torch.zeros(len(programs), dtype=torch.float64)
    comp_var = torch.zeros(len(programs), dtype=torch.float64)
    stroke = torch.zeros(len(programs), dtype=torch.float64)
    truncated = 0
    # The level says how many draws its own number needs. An exact level asking
    # for 32 would take a variance over one row and report NaN in the column
    # that decides whether this instrument resolves anything.
    samples = planner.composition.eval_samples

    for start in range(0, len(programs), batch_size):
        chunk = programs[start : start + batch_size]
        grid = torch.tensor(
            [summary_grid(p, planner.cfg.max_strokes) for p in chunk],
            dtype=torch.long, device=device,
        )
        # float64 only once off-device: MPS has no float64 (`dm/eval/metrics.py`).
        terms = (planner.composition.cost_terms(grid, samples, generator)
                 .cpu().double() / math.log(2))
        composition[start : start + len(chunk)] = terms.mean(dim=0)
        # Variance of *the mean*, per program. Summed across programs below,
        # because the corpus mean is what gets reported.
        comp_var[start : start + len(chunk)] = (
            terms.var(dim=0, unbiased=True) / samples if samples > 1 else 0.0
        )

        batch = stroke_batch(chunk, codec, planner.cfg)
        truncated += batch.truncated
        inputs, targets = batch.inputs.to(device), batch.targets.to(device)
        logits = planner.decoder(inputs)
        nll = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
            ignore_index=PAD, reduction="none",
        ).view(targets.shape).sum(dim=1).cpu().double() / math.log(2)
        # Several strokes per program, so the rows are summed back onto it.
        stroke[start : start + len(chunk)].index_add_(0, batch.owner, nll)

    return {
        "composition": composition.numpy(),
        "stroke": stroke.numpy(),
        "composition_var": comp_var.numpy(),
        "truncated": truncated,
    }


def bits_per_drawing(
    planner: StrokePlanner,
    programs: list[bytes],
    codec: Codec,
    device: str | torch.device = "cpu",
    batch_size: int = 32,
    seed: int | None = 0,
) -> dict:
    """`per_program_bits` reduced, in the shape `dm.eval.metrics` reports."""
    parts = per_program_bits(planner, programs, codec, device, batch_size, seed)
    total = parts["composition"] + parts["stroke"]
    n = max(1, len(programs))
    return {
        "bits_per_drawing": float(total.sum()) / n,
        "composition_bits": float(parts["composition"].sum()) / n,
        # The bound's own Monte-Carlo noise on the corpus mean. The stroke term
        # is exact, so this is the whole of it. Read it before reading any
        # difference: this project's resolution floors are ~1-2.5 bits per
        # corpus, and a bound noisier than that resolves nothing.
        "composition_stderr": float(parts["composition_var"].sum() ** 0.5) / n,
        "stroke_bits": float(parts["stroke"].sum()) / n,
        # Stated in the record, not inferred by the reader: the AR baseline's
        # number is exact and this one is conservative, so a planner that reads
        # lower has beaten it with a handicap and one that reads higher has not
        # necessarily lost.
        "is_upper_bound": True,
        # *Which* slack, though. Both objectives inherit the factorisation's --
        # `p(x) >= p_comp(f(x)) * prod_k p_stroke(...)` holds for any `p_comp`
        # -- and only the denoiser adds the ELBO's on top. Two rows both
        # labelled "upper bound" with different slack are two different numbers
        # in one column, which is the fault this project has found at four
        # levels; naming the sources is what lets their difference be read as
        # the ELBO's looseness rather than as a modelling result.
        "bound_sources": (
            ["factorisation", "diffusion_elbo"] if planner.composition.is_bound
            else ["factorisation"]
        ),
        "stroke_truncated": parts["truncated"],
        "val_bits": [round(float(b), 4) for b in total],
        "n": len(programs),
    }
