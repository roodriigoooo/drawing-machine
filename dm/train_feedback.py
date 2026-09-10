"""Direction 3's multi-pass training objective and its deterministic pass plan.

**What a pass is.** Pass 1 is ordinary next-token prediction on plain inputs.
Pass `k > 1` takes pass `k-1`'s normalized top-layer states, shifts them right by
one, fuses them with the *original* token embeddings, holds BOS and a sampled
prefix plain, and reruns the whole stack in parallel. The model therefore learns
to read a channel it will meet at decode time, without the training loop ever
decoding.

```text
L_K = L_1 + (1/(K-1)) * sum(L_k, k=2..K)   for K > 1
L_1                                        for K = 1
```

The later passes are **averaged, not summed**. Summing three unnormalised losses
would make the objective's scale a function of the schedule, so a batch that
happened to draw three passes would also get three times the gradient magnitude
-- a learning-rate schedule smuggled in through a data-augmentation knob. Every
pass is normalised by the *same* non-PAD token count, so the passes are on one
scale and a row's contribution does not depend on how many rows shared its batch.

**Nothing is detached.** `L_2`'s gradient reaches the states pass 1 produced,
which is what trains the channel rather than a read-out of a frozen one.

**The plan is drawn before micro-batching and derived from the step.** Micro-batch
count may change peak memory; it may not change a single random number, the
objective or the gradient (`docs/directions.md` §7 invariant 5). Two consequences
shape this Module: the plan is built once per optimiser step for the whole batch
and then *sliced*, and each stream is seeded from `(namespace, step)` rather than
advanced, so the plan for step 17,000 is computable without replaying the 16,999
steps before it -- which is what makes it testable, resumable and identical on
any device.

This Module owns the schedule, the prefix mixin, the jitter and the accounting.
It does **not** own a training loop: `dm.train` keeps the one loop this project
has, and calls in here for the step body when feedback is on.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor

from .data.dataset import PAD
from .eval.feedback_contract import (
    DEFAULT_FEEDBACK_SCHEMA,
    DEFAULT_GAIN_CALIBRATION,
    DEFAULT_PASS_SCHEDULE,
    JITTER_HALF_WIDTH,
    MAX_PASSES,
    PROTOCOL,
    RECORD_SCHEMA,
    expected_passes_per_batch,
    feedback_phase_transition,
    phase_for,
    seed_for,
)
from .eval.feedback_contract import gain_calibration as contract_gain_calibration
from .models.transformer import SUPPORTED_FEEDBACK_SCHEMAS

#: Multiplier mixing a stream's frozen seed with the step. Odd and coprime to any
#: power of two, so consecutive steps cannot land on neighbouring states of the
#: same generator; the point is only that `(namespace, step)` maps injectively
#: onto seeds, not that the map is cryptographic.
_STEP_MIX = 1_000_003


def _generator(namespace: str, step: int) -> torch.Generator:
    """One CPU generator per `(stream, step)`.

    Derived rather than advanced. An advancing generator makes step `n`'s draws a
    function of every step before it, so a resumed run diverges from an
    uninterrupted one, a single step cannot be reproduced in a test, and a
    changed batch count silently reshuffles everything downstream.
    """
    seed = (seed_for(namespace) * _STEP_MIX + step) % (2**63 - 1)
    return torch.Generator().manual_seed(seed)


# ---------------------------------------------------------------------------
# the plan


@dataclass(frozen=True)
class PassPlan:
    """Every random number one optimiser step needs, drawn before it is split.

    `prefix[i]` and `jitter[i]` belong to pass `i + 2` -- pass 1 is plain and has
    neither. Both are indexed by row in the *batch's* order, so slicing them with
    a micro-batch is a slice and never a redraw.
    """

    step: int
    passes: int
    #: `(passes - 1, rows)`: how many leading positions each later pass keeps at
    #: the raw embedding, per row. At least 1, so BOS is always plain.
    prefix: Tensor
    #: `passes - 1` blocks of `(rows, width, d_model)`, added to the carried state
    #: before fusion.
    jitter: tuple[Tensor, ...]

    def __post_init__(self) -> None:
        if not 1 <= self.passes <= MAX_PASSES:
            raise ValueError(f"passes must be 1..{MAX_PASSES}, got {self.passes}")
        if self.prefix.shape[0] != self.passes - 1:
            raise ValueError(
                f"plan has {self.prefix.shape[0]} prefix rows for {self.passes} passes"
            )
        if len(self.jitter) != self.passes - 1:
            raise ValueError(
                f"plan has {len(self.jitter)} jitter blocks for {self.passes} passes"
            )
        if self.passes > 1 and int(self.prefix.min()) < 1:
            raise ValueError("a plain prefix of 0 would fuse BOS, which has no state")
        if any(block.shape[0] != self.prefix.shape[1] for block in self.jitter):
            raise ValueError("jitter and prefix disagree on the row count")

    @property
    def rows(self) -> int:
        # From `prefix`, which is `(passes - 1, rows)` and so keeps its row count
        # even for a one-pass plan where there is no jitter block to ask.
        return int(self.prefix.shape[1])

    def for_rows(self, start: int, stop: int) -> PassPlan:
        """The same plan restricted to a row range. A slice, never a redraw."""
        return PassPlan(
            step=self.step,
            passes=self.passes,
            prefix=self.prefix[:, start:stop],
            jitter=tuple(block[start:stop] for block in self.jitter),
        )


def plan_step(*, step: int, steps: int, lengths: Tensor, width: int,
              d_model: int, schedule: str = DEFAULT_PASS_SCHEDULE) -> PassPlan:
    """Draw the pass count, plain prefixes and jitter for one optimiser step.

    `lengths` is each row's count of non-PAD **input** positions, on CPU. It is a
    parameter rather than something derived from a device tensor here because the
    plan must not depend on where the batch happens to live: an `.item()` on an
    accelerator would make the draw a function of the device's own dispatch, and
    two runs of the same experiment on two machines would stop being the same
    experiment.
    """
    rows = int(lengths.shape[0])
    passes = draw_passes(step, steps, schedule)
    prefix = torch.ones((passes - 1, rows), dtype=torch.long)
    jitter: list[Tensor] = []
    if passes > 1:
        prefix_rng = _generator("prefix", step)
        jitter_rng = _generator("jitter", step)
        # Uniform over 1..L: BOS is always plain, and a draw of L means the row
        # runs entirely plain for that pass -- a legitimate outcome the recipe
        # wants, because the model has to stay able to run without the channel.
        span = lengths.clamp(min=1).to(torch.float64)
        draws = torch.rand((passes - 1, rows), generator=prefix_rng, dtype=torch.float64)
        prefix = (draws * span).floor().to(torch.long) + 1
        for _ in range(passes - 1):
            block = torch.rand((rows, width, d_model), generator=jitter_rng)
            jitter.append((block * 2.0 - 1.0) * JITTER_HALF_WIDTH)
    return PassPlan(step=step, passes=passes, prefix=prefix, jitter=tuple(jitter))


def draw_passes(step: int, steps: int,
                schedule: str = DEFAULT_PASS_SCHEDULE) -> int:
    """How many passes this step's batch gets, under the named schedule.

    One uniform, read against the phase's cumulative weights, so the draw is
    hand-checkable and the frequencies are the schedule's rather than a
    library's. The schedule is a *distribution*: `expected_passes_per_batch()` is
    what it costs in expectation and the run record carries what it actually
    cost.

    **The uniform does not depend on the schedule.** `seed_for("pass_plan")` and
    the step are the whole seed, so F5b's three candidate schedules read the
    *same* draw against different weights: two cells that differ only in
    schedule differ only where the schedule differs, and a step whose draw falls
    outside every candidate's boundary is one-pass in all three.
    """
    weights = phase_for(step, steps, schedule).weights
    draw = float(torch.rand((), generator=_generator("pass_plan", step)))
    cumulative = 0.0
    for index, weight in enumerate(weights, start=1):
        cumulative += weight
        if draw < cumulative:
            return index
    return len(weights)


# ---------------------------------------------------------------------------
# the objective


@dataclass(frozen=True)
class PassOutput:
    """One pass's logits and the normalized states the next pass will carry."""

    logits: Tensor
    states: Tensor


def multi_pass(model, inputs: Tensor, classes: Tensor | None,
               plan: PassPlan) -> list[PassOutput]:
    """Run pass 1 plainly, then each later pass on the previous pass's states.

    The shift is right by one and the caller does it, matching the model seam:
    position `t` carries the state of position `t-1`, and position 0 carries
    nothing because BOS has no predecessor. Getting this direction wrong is the
    single most consequential silent error available here -- a left shift hands
    every position the state of the token *after* it, which trains a model that
    reads the future and validates perfectly well while doing it.
    """
    logits, states = model(inputs, classes, return_state=True)
    out = [PassOutput(logits, states)]
    for index in range(plan.passes - 1):
        carried = torch.zeros_like(states)
        carried[:, 1:] = states[:, :-1]
        carried = carried + plan.jitter[index].to(device=carried.device,
                                                  dtype=carried.dtype)
        logits, states = model(inputs, classes, feedback=carried,
                               plain=plan.prefix[index].to(carried.device),
                               return_state=True)
        out.append(PassOutput(logits, states))
    return out


def pass_nll(logits: Tensor, targets: Tensor) -> Tensor:
    """Summed next-token NLL in nats, PAD excluded.

    Summed rather than averaged so the caller owns the denominator: every pass
    and every micro-batch has to divide by the *same* whole-batch token count, or
    a row's weight in the objective depends on which chunk it landed in.
    """
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
        ignore_index=PAD, reduction="sum",
    )


def objective(outputs: list[PassOutput], targets: Tensor, n_tokens: Tensor
              ) -> tuple[Tensor, Tensor]:
    """`L_K` and the per-pass summed NLLs behind it.

    Both are returned because they answer different questions. `L_K` is what is
    optimised; `sums[0] / n_tokens` is ordinary next-token loss and is the only
    number comparable to a run that never had a feedback channel.
    """
    sums = torch.stack([pass_nll(out.logits, targets) for out in outputs])
    total = sums[0] / n_tokens
    if len(outputs) > 1:
        total = total + (sums[1:] / n_tokens).mean()
    return total, sums


# ---------------------------------------------------------------------------
# accumulation


def chunk_bounds(rows: int, width: int, budget: int, passes: int
                 ) -> Iterator[tuple[int, int]]:
    """Row ranges satisfying `rows * width^2 * passes <= budget`.

    The pass count multiplies the guard because a `K`-pass batch runs the stack
    `K` times over the same width, and every one of those runs holds its own
    activations until the backward. Reusing the single-pass budget would silently
    triple peak memory at the end of training, which is exactly where the
    three-pass batches live and where an out-of-memory kill costs the most.

    This is `dm.train.micro_batches`' rule with that one extra factor, and it is
    stated again here rather than shared because sharing it would mean the
    training loop importing the feedback Module for the *standard* path -- a
    change to code that is closed. At `passes=1` the two agree exactly, which
    `tests/test_train_feedback.py` asserts so the restatement cannot drift.
    """
    per = max(1, min(rows, budget // max(1, width * width * passes)))
    for start in range(0, rows, per):
        yield start, min(start + per, rows)


def accumulate(model, inputs: Tensor, targets: Tensor, classes: Tensor | None,
               *, plan: PassPlan, n_tokens: Tensor, budget: int) -> Tensor:
    """One optimiser step's backward, accumulated over memory-sized chunks.

    Returns the per-pass summed NLLs, detached. The gradient is the whole batch's
    because every chunk divides by the whole batch's token count -- the same
    property `dm.train.micro_batches` relies on, extended to `K` passes.
    """
    if getattr(model.cfg, "feedback_schema", DEFAULT_FEEDBACK_SCHEMA) == "none":
        raise ValueError(
            "a feedback training step needs a feedback-capable model; this one is "
            "feedback_schema='none' and has no channel to train"
        )
    rows, width = inputs.shape
    totals = torch.zeros(plan.passes, device=inputs.device)
    for start, stop in chunk_bounds(rows, width, budget, plan.passes):
        outputs = multi_pass(model, inputs[start:stop],
                             None if classes is None else classes[start:stop],
                             plan.for_rows(start, stop))
        loss, sums = objective(outputs, targets[start:stop], n_tokens)
        loss.backward()
        totals += sums.detach()
    return totals


# ---------------------------------------------------------------------------
# the shared input norm's gain, calibrated once at the phase transition


def input_token_histogram(sequences: Iterable[Tensor], vocab_size: int) -> Tensor:
    """Counts of every token over the training split's **input** positions.

    A row's inputs are its encoded program without the final symbol: `collate`
    hands the model `padded[:, :-1]` and the last symbol is a target only. BOS is
    counted because it is a real input position -- position 0, which every pass
    holds plain. PAD is not counted: it never appears in an encoded program, and
    it is masked here as well so the statement survives a caller that pads first.

    A function of the corpus and the codec alone. No batch composition, no
    bucketing, no device, no model: two runs of the same corpus produce the same
    histogram whatever their batch size, which is what lets the calibration it
    feeds be a *predeclared rule* rather than a draw.
    """
    counts = torch.zeros(vocab_size, dtype=torch.long)
    for sequence in sequences:
        row = torch.as_tensor(sequence, dtype=torch.long).reshape(-1)[:-1]
        if row.numel():
            counts += torch.bincount(row, minlength=vocab_size)[:vocab_size]
    counts[PAD] = 0
    return counts


def embedding_scale(embed: Tensor, counts: Tensor) -> float:
    """`sqrt(sum_v p_v * mean_d E[v, d]^2)`: frequency-weighted embedding RMS.

    The scale the trunk's input positions actually arrive at, as opposed to the
    scale an unweighted mean over the vocabulary would report. The two differ by
    however skewed the corpus is, and this project's are skewed: a handful of
    opcodes and a coordinate alphabet carry most positions.

    Weighted over *rows*, then rooted once, rather than a mean of per-row RMS
    values. The quantity that has to match is a mean square, because that is what
    `RMSNorm` divides by; a mean of square roots is a different number and is not
    the one the norm is about.
    """
    if embed.shape[0] != counts.shape[0]:
        raise ValueError(
            f"the histogram covers {counts.shape[0]} tokens and the embedding "
            f"table has {embed.shape[0]} rows; they must be the same vocabulary"
        )
    total = float(counts.sum())
    if total <= 0:
        raise ValueError(
            "the training split has no counted input positions, so there is no "
            "token distribution to weight the embedding by: incomplete"
        )
    weights = counts.to(embed.dtype).to(embed.device) / total
    mean_square = embed.detach().float().pow(2).mean(dim=-1)
    return float(torch.sqrt((weights.float() * mean_square).sum()))


@torch.no_grad()
def calibrate_gain(model, counts: Tensor, *, rule: str, step: int) -> dict:
    """Apply one named gain calibration, once, and return what it did.

    The whole of the F5c correction package's intervention is this call site.
    Everything else about the run -- the arm, the objective, the schedule, the
    jitter, every threshold -- is what F5b trained, which is what makes the
    package one lever rather than a search (`docs/directions.md` §6, F5c).

    `none` is the default and is a real branch rather than a skipped call: a
    record that says "calibration ran and did nothing" is a different artifact
    from one that says "no calibration was configured", and only the second is
    what every existing checkpoint trained under.
    """
    contract_gain_calibration(rule)
    if model.fuse is None:
        raise ValueError(
            "a gain calibration needs a feedback channel to calibrate; this "
            "model is feedback_schema='none'"
        )
    if rule == "none":
        return {"rule": rule, "applied": False, "step": step}
    value = embedding_scale(model.embed.weight, counts)
    applied = model.fuse.set_shared_gain(value)
    return {
        "rule": rule,
        "applied": True,
        "step": step,
        "formula": contract_gain_calibration(rule),
        "value": value,
        "gain_before": applied["before"],
        "gain_after": applied["after"],
        "feedback_schema": applied["schema"],
        "counted_input_positions": int(counts.sum()),
        "vocabulary": int(counts.shape[0]),
    }


# ---------------------------------------------------------------------------
# accounting


def peak_memory(device: torch.device) -> int | None:
    """Allocator high-water mark, or `None` where there is nothing to read.

    `None` is the honest answer on CPU and is *not* a missing field: the record
    carries the key with a null, so a reader can tell "not measurable here" from
    "nobody recorded it", which the fail-closed rule treats very differently.
    """
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated())
    if device.type == "mps":
        return int(torch.mps.driver_allocated_memory())
    return None


@dataclass
class Accounting:
    """Separate counters for costs that are routinely conflated.

    "Same training tokens" is not "same compute" and not "same data". The bit arm
    sees eight symbols per semantic byte; the pass schedule adds ~13% forward
    passes on top of the step count; padded positions are neither data nor
    signal. A record that collapsed these into one number could not support any
    efficiency statement, and the temptation to make one is exactly what a
    bit-versus-byte comparison invites (`docs/directions.md` §7 invariant 6).
    """

    device: str
    feedback_schema: str
    #: Symbols per bytecode byte, so semantic size can be separated from symbol
    #: count. 1 for byte, 8 for bit.
    stride: int
    #: Which named pass schedule the plan was drawn from. Recorded because the
    #: realized pass histogram alone cannot distinguish a schedule from an
    #: unlucky draw, and F5b compares three of them.
    pass_schedule: str = DEFAULT_PASS_SCHEDULE
    #: Which named gain calibration the run was configured with, and what that
    #: calibration actually did. Both, and separately: the rule is what the cell
    #: was *asked* for and the event is what happened, and a package whose one
    #: lever silently failed to move would otherwise look exactly like one whose
    #: lever moved and changed nothing.
    gain_calibration: str = DEFAULT_GAIN_CALIBRATION
    gain_calibration_event: dict | None = None
    programs_seen: int = 0
    semantic_bytes_seen: int = 0
    content_symbols_seen: int = 0
    padded_positions: int = 0
    content_symbol_forward_passes: int = 0
    padded_forward_positions: int = 0
    pass_histogram: dict[int, int] = field(default_factory=dict)
    peak_memory_bytes: int | None = None
    wall_clock_s: float = 0.0

    def observe(self, plan: PassPlan, *, rows: int, width: int,
                content_symbols: int) -> None:
        self.programs_seen += rows
        self.content_symbols_seen += content_symbols
        # Exact rather than approximate: a row's targets are its whole encoded
        # program after BOS, so the content symbol count is always a whole number
        # of bytecode bytes.
        self.semantic_bytes_seen += content_symbols // self.stride
        self.padded_positions += rows * width
        self.content_symbol_forward_passes += content_symbols * plan.passes
        self.padded_forward_positions += rows * width * plan.passes
        self.pass_histogram[plan.passes] = self.pass_histogram.get(plan.passes, 0) + 1

    @property
    def observed_passes_per_batch(self) -> float:
        batches = sum(self.pass_histogram.values())
        if not batches:
            return float("nan")
        return sum(k * n for k, n in self.pass_histogram.items()) / batches

    def as_dict(self) -> dict:
        return {
            "schema": RECORD_SCHEMA,
            "protocol": PROTOCOL,
            "feedback_schema": self.feedback_schema,
            "pass_schedule": self.pass_schedule,
            "gain_calibration": self.gain_calibration,
            "gain_calibration_event": self.gain_calibration_event,
            "device": self.device,
            "stride": self.stride,
            "programs_seen": self.programs_seen,
            "semantic_bytes_seen": self.semantic_bytes_seen,
            "content_symbols_seen": self.content_symbols_seen,
            "padded_positions": self.padded_positions,
            "content_symbol_forward_passes": self.content_symbol_forward_passes,
            "padded_forward_positions": self.padded_forward_positions,
            # String keys, because JSON has no integer key and a round-tripped
            # record must compare equal to the one that was written.
            "pass_histogram": {str(k): self.pass_histogram[k]
                               for k in sorted(self.pass_histogram)},
            "expected_passes_per_batch": expected_passes_per_batch(self.pass_schedule),
            "observed_passes_per_batch": self.observed_passes_per_batch,
            "peak_memory_bytes": self.peak_memory_bytes,
            "wall_clock_s": self.wall_clock_s,
        }


def validate_schema(schema: str) -> str:
    if schema not in SUPPORTED_FEEDBACK_SCHEMAS:
        raise ValueError(
            f"unknown feedback_schema {schema!r}; expected one of "
            f"{list(SUPPORTED_FEEDBACK_SCHEMAS)}"
        )
    return schema


__all__ = [
    "Accounting",
    "PassOutput",
    "PassPlan",
    "accumulate",
    "calibrate_gain",
    "chunk_bounds",
    "draw_passes",
    "embedding_scale",
    "feedback_phase_transition",
    "input_token_histogram",
    "multi_pass",
    "objective",
    "pass_nll",
    "peak_memory",
    "plan_step",
    "validate_schema",
]
