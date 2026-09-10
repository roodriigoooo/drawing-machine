"""F2: the multi-pass objective and the pass plan behind it.

Three properties carry the stage, and each of them is a way the training loop
could be wrong while producing a perfectly healthy loss curve.

**The shift direction.** A left shift hands every position the state of the token
*after* it. The model then reads the future, trains beautifully, and every
Direction 3 number afterwards is meaningless. Pinned against analytic tensors
rather than against the implementation.

**The denominator and the averaging.** Every pass divides by the same whole-batch
non-PAD token count. Get it wrong and a row's weight in the objective depends on
which micro-batch it landed in -- which makes the gradient a function of the
memory budget.

**Micro-batch invariance.** Splitting a batch may change peak memory and nothing
else: not a random number, not the objective, not the gradient.
"""

from __future__ import annotations

import json
import math
from collections import Counter

import pytest
import torch

import dm.train
from dm.data.dataset import PAD
from dm.eval import feedback_contract as contract
from dm.isa.codec import CODECS
from dm.models.transformer import Config, DrawingLM
from dm.train import TrainConfig, micro_batches, train
from dm.train_feedback import (
    Accounting,
    PassOutput,
    PassPlan,
    accumulate,
    calibrate_gain,
    chunk_bounds,
    draw_passes,
    embedding_scale,
    input_token_histogram,
    multi_pass,
    objective,
    pass_nll,
    plan_step,
)

STEPS = contract.FINAL_STEPS
D_MODEL = 16


def _model(vocab: int = 32, schema: str = "glu_v1", seed: int = 0) -> DrawingLM:
    torch.manual_seed(seed)
    return DrawingLM(Config(vocab_size=vocab, d_model=D_MODEL, n_layers=2,
                            n_heads=2, max_len=64, feedback_schema=schema))


def _batch(rows: int = 4, width: int = 6, vocab: int = 32
           ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inputs, targets and per-row input lengths, with uneven padding.

    Uneven on purpose: a token-count denominator and a per-row prefix both look
    correct on a rectangular batch and are the two things most likely to be wrong
    on a real one.
    """
    generator = torch.Generator().manual_seed(3)
    inputs = torch.randint(2, vocab, (rows, width), generator=generator)
    targets = torch.randint(2, vocab, (rows, width), generator=generator)
    for row, keep in enumerate([width, width - 1, width - 3, 2][:rows]):
        inputs[row, keep:] = PAD
        targets[row, keep:] = PAD
    return inputs, targets, (inputs != PAD).sum(dim=1)


def _plan(step: int, lengths: torch.Tensor, width: int, passes: int) -> PassPlan:
    """A plan with a chosen pass count, drawn from the frozen streams otherwise.

    Tests that need `K = 3` cannot wait for the 3% of steps that draw it, and
    forcing the count is not the same as forging the plan: prefixes and jitter
    still come from `plan_step`'s own generators.
    """
    while True:
        plan = plan_step(step=step, steps=STEPS, lengths=lengths, width=width,
                         d_model=D_MODEL)
        if plan.passes == passes:
            return plan
        step += 1


# ---------------------------------------------------------------------------
# the plan


def test_the_plan_is_a_function_of_the_step_and_nothing_else():
    """Derived, not advanced. An advancing generator makes step n's draws a
    function of every step before it, so a resumed run diverges from an
    uninterrupted one and no single step can be reproduced in a test."""
    _, _, lengths = _batch()
    first = plan_step(step=19_000, steps=STEPS, lengths=lengths, width=6,
                      d_model=D_MODEL)
    second = plan_step(step=19_000, steps=STEPS, lengths=lengths, width=6,
                       d_model=D_MODEL)
    assert first.passes == second.passes
    assert torch.equal(first.prefix, second.prefix)
    assert all(torch.equal(a, b) for a, b in zip(first.jitter, second.jitter))


def test_slicing_the_plan_is_a_slice_and_never_a_redraw():
    """The whole reason the plan exists as an object: micro-batch count may
    change memory, never a random number (`docs/directions.md` §7 invariant 5)."""
    _, _, lengths = _batch(rows=4)
    plan = _plan(18_000, lengths, width=6, passes=2)
    left, right = plan.for_rows(0, 2), plan.for_rows(2, 4)
    assert torch.equal(torch.cat([left.prefix, right.prefix], dim=1), plan.prefix)
    for index, block in enumerate(plan.jitter):
        assert torch.equal(
            torch.cat([left.jitter[index], right.jitter[index]], dim=0), block
        )
    assert left.passes == right.passes == plan.passes


def test_the_first_half_of_training_never_draws_a_second_pass():
    """Exactly, not statistically: the phase weights are `(1.0,)` and the draw is
    a uniform on `[0, 1)`, so there is no tail to be unlucky in."""
    assert all(draw_passes(step, STEPS) == 1 for step in range(0, STEPS // 2, 97))
    assert draw_passes(STEPS // 2 - 1, STEPS) == 1


def test_the_schedule_realises_the_frozen_phase_weights():
    """The schedule is a distribution, so this checks the realisation against it
    rather than pretending the count is fixed. `expected_passes_per_batch` is what
    it costs in expectation and the run record carries what it actually cost."""
    counts = Counter(draw_passes(step, STEPS) for step in range(STEPS))
    assert set(counts) == {1, 2, 3}
    assert counts[1] / STEPS == pytest.approx(0.875, abs=0.01)
    assert counts[2] / STEPS == pytest.approx(0.1175, abs=0.01)
    assert counts[3] / STEPS == pytest.approx(0.0075, abs=0.005)
    observed = sum(k * n for k, n in counts.items()) / STEPS
    assert observed == pytest.approx(contract.expected_passes_per_batch(), abs=0.01)


def test_the_plain_prefix_covers_bos_and_never_runs_past_the_row():
    """`1 <= p <= L`. Below 1 would fuse BOS, which has no predecessor state;
    above `L` would hold padding plain, which is meaningless and would hide a
    real prefix bug behind an inert one."""
    _, _, lengths = _batch(rows=4, width=6)
    for step in range(18_000, 18_400):
        plan = plan_step(step=step, steps=STEPS, lengths=lengths, width=6,
                         d_model=D_MODEL)
        if plan.passes == 1:
            continue
        assert int(plan.prefix.min()) >= 1
        assert bool((plan.prefix <= lengths[None, :]).all())


def test_the_prefix_actually_varies_across_rows_and_passes():
    """A constant prefix would pass every bound check above and train one fixed
    regime instead of the mixture the recipe asks for."""
    _, _, lengths = _batch(rows=4, width=6)
    drawn = {
        tuple(row)
        for step in range(18_000, 18_600)
        for plan in [plan_step(step=step, steps=STEPS, lengths=lengths, width=6,
                               d_model=D_MODEL)]
        if plan.passes > 1
        for row in plan.prefix.tolist()
    }
    assert len(drawn) > 5


def test_jitter_is_bounded_shaped_per_batch_and_independent_per_pass():
    _, _, lengths = _batch(rows=4, width=6)
    plan = _plan(18_000, lengths, width=6, passes=3)
    assert len(plan.jitter) == 2
    for block in plan.jitter:
        assert block.shape == (4, 6, D_MODEL)
        assert float(block.abs().max()) <= contract.JITTER_HALF_WIDTH
    assert not torch.equal(plan.jitter[0], plan.jitter[1])


def test_a_one_pass_plan_carries_no_prefix_and_no_jitter():
    _, _, lengths = _batch()
    plan = _plan(0, lengths, width=6, passes=1)
    assert plan.prefix.shape == (0, lengths.shape[0]) and plan.jitter == ()
    assert plan.rows == lengths.shape[0]


# ---------------------------------------------------------------------------
# the passes


def test_a_later_pass_carries_the_previous_states_shifted_right():
    """The single most consequential silent error available in this stage.

    A left shift hands position `t` the state of `t+1`. The model reads the
    future, the loss curve improves, and every teacher-forced number afterwards
    is measuring a model that cannot exist at decode time. Checked against an
    explicit reconstruction, and against the wrong direction failing.
    """
    model = _model()
    inputs, _, lengths = _batch()
    plan = _plan(18_000, lengths, width=inputs.shape[1], passes=2)

    outputs = multi_pass(model, inputs, None, plan)
    states = outputs[0].states

    right = torch.zeros_like(states)
    right[:, 1:] = states[:, :-1]
    expected = model(inputs, feedback=right + plan.jitter[0], plain=plan.prefix[0])
    assert torch.equal(outputs[1].logits, expected)

    left = torch.zeros_like(states)
    left[:, :-1] = states[:, 1:]
    wrong = model(inputs, feedback=left + plan.jitter[0], plain=plan.prefix[0])
    assert not torch.allclose(outputs[1].logits, wrong, atol=1e-5)


def test_position_zero_carries_nothing_in_any_pass():
    """BOS has no predecessor. The carried slot is zero and the prefix holds it
    plain, so the two guards agree rather than one covering for the other."""
    model = _model()
    inputs, _, lengths = _batch()
    plan = _plan(18_000, lengths, width=inputs.shape[1], passes=2)
    outputs = multi_pass(model, inputs, None, plan)
    # Pass 2 saw a plain BOS, so its first-position logits are pass 1's.
    assert torch.equal(outputs[1].logits[:, 0], outputs[0].logits[:, 0])


@pytest.mark.parametrize("passes", (1, 2, 3))
def test_the_objective_is_pass_one_plus_the_mean_of_the_later_passes(passes):
    """`L_K = L_1 + (1/(K-1)) * sum(L_k)`, against hand-built logits.

    Averaged rather than summed: a sum would make the objective's scale a
    function of the schedule, so a three-pass batch would also get three times
    the gradient -- a learning-rate change smuggled in through a data knob.
    """
    generator = torch.Generator().manual_seed(1)
    targets = torch.tensor([[5, 7, PAD], [2, 3, 4]])
    outputs = [
        PassOutput(logits=torch.randn(2, 3, 9, generator=generator),
                   states=torch.zeros(2, 3, D_MODEL))
        for _ in range(passes)
    ]
    n_tokens = (targets != PAD).sum()
    total, sums = objective(outputs, targets, n_tokens)

    expected = sums[0] / n_tokens
    if passes > 1:
        expected = expected + sum(sums[1:] / n_tokens) / (passes - 1)
    assert torch.allclose(total, expected, atol=0, rtol=0)
    assert sums.shape == (passes,)
    assert torch.allclose(sums[0], pass_nll(outputs[0].logits, targets))


def test_padding_is_excluded_from_every_pass_and_from_the_denominator():
    """Five real targets out of six positions. A loss that scored PAD would be
    optimising a symbol no codec emits and the denominator would be wrong for
    every pass at once."""
    logits = torch.zeros(2, 3, 4)
    targets = torch.tensor([[1, 2, PAD], [3, PAD, PAD]])
    # Uniform logits over four symbols: every scored position costs log 4.
    assert float(pass_nll(logits, targets)) == pytest.approx(3 * torch.tensor(4.0).log())
    assert float(pass_nll(logits, torch.full_like(targets, PAD))) == 0.0


def test_a_later_loss_reaches_the_states_the_earlier_pass_produced():
    """No detach. Without the gradient path, pass 2 trains a read-out of a frozen
    state rather than the channel that produced it, and the mechanism under test
    is never optimised at all."""
    model = _model()
    inputs, targets, lengths = _batch()
    plan = _plan(18_000, lengths, width=inputs.shape[1], passes=2)
    outputs = multi_pass(model, inputs, None, plan)

    grad = torch.autograd.grad(pass_nll(outputs[1].logits, targets),
                               outputs[0].states, retain_graph=True)[0]
    assert float(grad.abs().sum()) > 0


def test_the_fusion_weights_get_a_gradient_only_when_a_later_pass_runs():
    """A one-pass batch touches no fusion parameter, which is what makes the 87%
    of one-pass steps ordinary next-token training rather than a silently
    different objective."""
    model = _model()
    inputs, targets, lengths = _batch()
    n_tokens = (targets != PAD).sum()
    for passes, expected in ((1, False), (2, True)):
        model.zero_grad(set_to_none=True)
        plan = _plan(18_000, lengths, width=inputs.shape[1], passes=passes)
        accumulate(model, inputs, targets, None, plan=plan, n_tokens=n_tokens,
                   budget=10**9)
        gradient = dict(model.named_parameters())["fuse.up.weight"].grad
        assert (gradient is not None and float(gradient.abs().sum()) > 0) is expected


# ---------------------------------------------------------------------------
# micro-batching


def _grads(model, inputs, targets, plan, budget):
    model.zero_grad(set_to_none=True)
    n_tokens = (targets != PAD).sum()
    accumulate(model, inputs, targets, None, plan=plan, n_tokens=n_tokens,
               budget=budget)
    # `None` where a parameter took no gradient at all, which is the honest state
    # of the fusion weights on a one-pass batch. Substituting zeros here would
    # make "never touched" and "touched to zero" the same reading.
    return [None if p.grad is None else p.grad.clone() for p in model.parameters()]


@pytest.mark.parametrize("passes", (1, 2, 3))
def test_splitting_a_batch_leaves_the_gradient_alone(passes):
    """The property `dm.train.micro_batches` already guarantees for one pass,
    extended to `K`. Compared at a tolerance, not bitwise: the chunks sum in a
    different order, and float addition is not associative."""
    model = _model()
    inputs, targets, lengths = _batch(rows=4, width=6)
    plan = _plan(18_000, lengths, width=6, passes=passes)
    width = inputs.shape[1]
    split = 1 * width * width * passes  # one row per chunk

    assert len(list(chunk_bounds(4, width, split, passes))) == 4
    whole = _grads(model, inputs, targets, plan, budget=10**9)
    chunked = _grads(model, inputs, targets, plan, budget=split)
    for left, right in zip(whole, chunked):
        assert (left is None) == (right is None)
        if left is not None:
            assert torch.allclose(left, right, atol=1e-6)


def test_the_memory_guard_counts_the_passes():
    """A `K`-pass batch runs the stack `K` times over the same width and holds
    every one of those activation sets until the backward. Reusing the one-pass
    budget would triple peak memory exactly where the three-pass batches live."""
    assert list(chunk_bounds(8, 4, 8 * 16, 1)) == [(0, 8)]
    assert list(chunk_bounds(8, 4, 8 * 16, 2)) == [(0, 4), (4, 8)]
    assert len(list(chunk_bounds(8, 4, 8 * 16, 3))) == 4


def test_the_one_pass_guard_is_the_trainers_own_rule():
    """`chunk_bounds` restates `dm.train.micro_batches`' sizing with one extra
    factor. Restated rather than shared, because sharing would mean the training
    loop importing the feedback Module for the *standard* path -- so this asserts
    the restatement has not drifted."""
    inputs = torch.zeros(7, 5, dtype=torch.long)
    for budget in (1, 5 * 25, 3 * 5 * 25, 10**9):
        expected = [tuple(chunk.shape[:1]) for chunk, _, _ in
                    micro_batches(inputs, inputs, budget)]
        actual = [(stop - start,) for start, stop in chunk_bounds(7, 5, budget, 1)]
        assert actual == expected, budget


def test_a_feedback_step_refuses_a_model_with_no_channel():
    """`PLAN.md`'s first trap, at the training end: a checkpoint without fusion
    parameters has nothing to train and every pass after the first would be pass
    one again."""
    model = _model(schema="none")
    inputs, targets, lengths = _batch()
    plan = _plan(18_000, lengths, width=inputs.shape[1], passes=2)
    with pytest.raises(ValueError, match="feedback-capable model"):
        accumulate(model, inputs, targets, None, plan=plan,
                   n_tokens=(targets != PAD).sum(), budget=10**9)


# ---------------------------------------------------------------------------
# accounting


def test_the_counters_keep_programs_symbols_and_passes_apart():
    """"Same training tokens" is not "same compute" and not "same data". The bit
    arm sees eight symbols per semantic byte and the schedule adds forward passes
    on top of the step count; a record that collapsed them could not support any
    efficiency statement."""
    accounting = Accounting(device="cpu", feedback_schema="glu_v1", stride=8)
    _, _, lengths = _batch()
    one = _plan(0, lengths, width=6, passes=1)
    two = _plan(18_000, lengths, width=6, passes=2)
    accounting.observe(one, rows=4, width=6, content_symbols=16)
    accounting.observe(two, rows=4, width=6, content_symbols=24)

    assert accounting.programs_seen == 8
    assert accounting.content_symbols_seen == 40
    assert accounting.semantic_bytes_seen == 16 // 8 + 24 // 8
    assert accounting.padded_positions == 2 * 4 * 6
    # 16 symbols once, 24 symbols twice: the gap between data and compute.
    assert accounting.content_symbol_forward_passes == 16 + 24 * 2
    assert accounting.padded_forward_positions == 24 + 24 * 2
    assert accounting.pass_histogram == {1: 1, 2: 1}
    assert accounting.observed_passes_per_batch == 1.5

    published = accounting.as_dict()
    assert published["pass_histogram"] == {"1": 1, "2": 1}
    assert published["expected_passes_per_batch"] == contract.expected_passes_per_batch()
    assert published["protocol"] == contract.PROTOCOL


def test_a_null_peak_memory_is_recorded_rather_than_omitted():
    """There is no allocator high-water mark on CPU. "Not measurable here" has to
    be distinguishable from "nobody recorded it", because the fail-closed rule
    treats the two very differently."""
    published = Accounting(device="cpu", feedback_schema="glu_v1", stride=1).as_dict()
    assert "peak_memory_bytes" in published and published["peak_memory_bytes"] is None
    assert contract.missing_record_fields({**published, "complete": True,
                                           "steps_done": 1, "corpus": {}}) == []


def test_a_record_that_drops_an_accounting_field_is_reported_as_missing():
    published = Accounting(device="cpu", feedback_schema="glu_v1", stride=1).as_dict()
    del published["content_symbol_forward_passes"]
    missing = contract.missing_record_fields(
        {"complete": True, "steps_done": 1, "corpus": {}, "feedback": published}
    )
    assert missing == ["content_symbol_forward_passes"]


# ---------------------------------------------------------------------------
# the trainer


def _tiny(**overrides) -> TrainConfig:
    return TrainConfig(
        **{"codec": "byte", "shape": "square", "n_train": 48, "n_val": 8,
           "steps": 8, "eval_every": 8, "warmup": 1, "batch_size": 8,
           "max_len": 128, "gen_samples": 2, "device": "cpu", **overrides}
    )


def test_a_standard_run_records_exactly_what_it_always_did(tmp_path, monkeypatch):
    """The closed record's guarantee. `feedback_schema` is a new config field with
    a default, and a schema-4 record written today has to stay comparable with one
    written before the field existed."""
    monkeypatch.setattr(dm.train, "RUNS", tmp_path)
    result = train(_tiny(steps=1, eval_every=1, tag="unit_std"), verbose=False)
    assert "feedback" not in result
    assert result["config"]["feedback_schema"] == "none"
    assert result["model"]["feedback_schema"] == "none"


def test_a_feedback_run_trains_checkpoints_and_accounts_for_itself(tmp_path,
                                                                   monkeypatch):
    """End to end through the one training loop, with the accounting the protocol
    requires present in the record it wrote to disk."""
    monkeypatch.setattr(dm.train, "RUNS", tmp_path)
    result = train(_tiny(feedback_schema="glu_v1", tag="unit_feedback"),
                   verbose=False)

    assert result["complete"] is True and result["steps_done"] == 8
    assert contract.missing_record_fields(result) == []
    published = json.loads((tmp_path / "unit_feedback.json").read_text())
    assert published["feedback"] == result["feedback"]

    accounting = result["feedback"]
    assert accounting["feedback_schema"] == "glu_v1"
    assert accounting["stride"] == CODECS["byte"].stride
    assert sum(accounting["pass_histogram"].values()) == 8
    assert accounting["content_symbol_forward_passes"] >= accounting["content_symbols_seen"]
    assert accounting["wall_clock_s"] > 0

    weights = torch.load(tmp_path / "unit_feedback.pt", weights_only=False)
    assert weights["cfg"]["feedback_schema"] == "glu_v1"
    assert all(name in weights["state"] for name in contract.FEEDBACK_PARAMETERS)


def test_the_feedback_checkpoint_reloads_strict_into_its_own_config(tmp_path,
                                                                    monkeypatch):
    """F6 writes eight of these and F7 reloads them before scoring. A checkpoint
    that cannot be rebuilt from its own recorded config is `incomplete`, and
    discovering that after training is the expensive way to find out."""
    monkeypatch.setattr(dm.train, "RUNS", tmp_path)
    train(_tiny(steps=1, eval_every=1, feedback_schema="glu_v1", tag="unit_reload"),
          verbose=False)
    weights = torch.load(tmp_path / "unit_reload.pt", weights_only=False)
    model = DrawingLM(Config(**weights["cfg"]))
    model.load_state_dict(weights["state"], strict=True)
    assert model.n_params() == model.cfg.n_params()


def test_the_training_config_round_trips_through_a_record():
    """`config_from_record` is the only supported way to rebuild a run's corpus,
    and a field it dropped would silently rebuild the wrong experiment."""
    cfg = _tiny(feedback_schema="glu_v1")
    rebuilt = dm.train.config_from_record({"config": {**cfg.__dict__,
                                                      "tier": int(cfg.tier)}})
    assert rebuilt == cfg
    # An old record has no such key and must still rebuild.
    old = {k: v for k, v in cfg.__dict__.items() if k != "feedback_schema"}
    assert dm.train.config_from_record(
        {"config": {**old, "tier": int(cfg.tier)}}
    ).feedback_schema == "none"


def test_an_unknown_feedback_schema_never_reaches_a_training_run():
    with pytest.raises(ValueError, match="unknown feedback_schema"):
        train(_tiny(feedback_schema="glu_v9"), verbose=False)


# ---------------------------------------------------------------------------
# F5c: the shared input norm's gain, calibrated once at the phase transition
#
# The one lever the F5b stop is allowed to buy. Everything else about a
# correction cell -- the arm, the objective, the schedule, the jitter, every
# threshold -- is what F5b trained, so these tests are about the calibration
# firing exactly once, at the step the schedule declares, on a statistic that
# reads no outcome.


def test_the_phase_transition_is_derived_from_the_schedule_not_written_down():
    """A package cannot calibrate at a step its schedule does not switch on at,
    and the rule has to read correctly for a schedule that is multi-pass from
    step 0."""
    for name in contract.PASS_SCHEDULES:
        step = contract.feedback_phase_transition(STEPS, name)
        assert len(contract.phase_for(step, STEPS, name).weights) > 1
        assert len(contract.phase_for(step - 1, STEPS, name).weights) == 1
    # All three project schedules run one pass for the first half.
    assert contract.feedback_phase_transition(STEPS) == STEPS // 2

    always = contract.PASS_SCHEDULES["project_progressive_v1"]
    assert contract.feedback_phase_transition(8, "terminal_mix_v1") == 4
    assert always  # the table is read, not reconstructed

    with pytest.raises(ValueError, match="steps must be positive"):
        contract.feedback_phase_transition(0)


def test_the_calibration_statistic_is_a_frequency_weighted_mean_square():
    """`g = sqrt(sum_v p_v * mean_d E[v, d]^2)`, hand-computed.

    Weighted over rows and rooted once, rather than a mean of per-row RMS values:
    the quantity that has to match is a mean square, because that is what
    `RMSNorm` divides by, and a mean of square roots is a different number.
    """
    counts = torch.tensor([0, 3, 1, 0])
    embed = torch.tensor([[9.0, 9.0], [1.0, 1.0], [3.0, 3.0], [0.0, 0.0]])
    assert embedding_scale(embed, counts) == pytest.approx(
        math.sqrt((3 * 1.0 + 1 * 9.0) / 4), rel=1e-6)

    # The PAD row is the tied head's negative class and is never a content
    # input, so a large PAD embedding cannot move the statistic.
    loud_pad = embed.clone()
    loud_pad[PAD] = 500.0
    assert embedding_scale(loud_pad, counts) == pytest.approx(
        embedding_scale(embed, counts), rel=1e-6)

    with pytest.raises(ValueError, match="same vocabulary"):
        embedding_scale(embed, torch.tensor([1, 1, 1]))
    with pytest.raises(ValueError, match="no counted input positions"):
        embedding_scale(embed, torch.zeros(4, dtype=torch.long))


def test_the_input_histogram_is_a_function_of_the_corpus_alone():
    """A row's inputs are its encoded program without the final symbol, because
    `collate` hands the model `padded[:, :-1]`. No batch composition, no
    bucketing, no device: two runs of one corpus produce one histogram whatever
    their batch size, which is what makes the calibration a predeclared rule."""
    # `[1,2,2,3]` contributes `1,2,2` and `[1,2]` contributes `1`; the final
    # symbol of each program is a target and never an input.
    sequences = [torch.tensor([1, 2, 2, 3]), torch.tensor([1, 2])]
    counts = input_token_histogram(sequences, 5)
    assert counts.tolist() == [0, 2, 2, 0, 0]
    assert int(counts.sum()) == sum(len(s) - 1 for s in sequences)

    # PAD never appears in an encoded program -- it arrives from `collate` -- and
    # it is masked anyway, so the statement survives a caller that pads first.
    assert int(counts[PAD]) == 0
    assert input_token_histogram([torch.tensor([1, 2, PAD, PAD])], 5).tolist() == (
        [0, 1, 1, 0, 0])

    # A one-symbol program has no input position and contributes nothing rather
    # than wrapping to the last element.
    assert input_token_histogram([torch.tensor([1])], 5).tolist() == [0] * 5


def test_the_calibration_sets_every_component_of_the_shared_gain():
    """`stack_input` returns `N(m)`, so a gain filled with `g` hands the trunk
    rows at RMS exactly `g`. That is what makes a fused pass's plain prefix the
    size of the raw-embedding prefill it exists to imitate."""
    model = _model(vocab=32, schema=contract.QUALIFICATION_FEEDBACK_SCHEMA)
    with torch.no_grad():  # a gain that has taken gradient and gone anisotropic
        model.fuse.norm.weight.copy_(
            torch.linspace(0.01, 0.1, model.cfg.d_model))
    counts = torch.ones(32, dtype=torch.long)
    counts[PAD] = 0

    event = calibrate_gain(model, counts, rule="embedding_rms_at_switch_on_v1",
                           step=12_000)
    assert event["applied"] is True
    assert event["step"] == 12_000
    assert event["formula"] == contract.gain_calibration(
        "embedding_rms_at_switch_on_v1")
    assert event["value"] == pytest.approx(
        embedding_scale(model.embed.weight, counts), rel=1e-6)
    gain = model.fuse.norm.weight.detach()
    assert torch.allclose(gain, torch.full_like(gain, event["value"]))
    # The four-number before-summary is what says what was overwritten: by the
    # transition the gain is anisotropic and its mean alone would not.
    assert event["gain_before"]["min"] != event["gain_before"]["max"]

    # And the row scale the trunk then reads is the statistic. Not to the last
    # bit: `RMSNorm` adds `eps` under its reciprocal square root, so the output
    # sits a few parts in ten thousand below the gain. The contract says "the
    # right scale and the right direction, not an equality", and this is where
    # that qualification comes from.
    embed = model.embed.weight.detach()[1:4]
    fused = model.fuse.norm(embed)
    assert torch.allclose(fused.pow(2).mean(-1).sqrt(),
                          torch.full((3,), event["value"]), rtol=5e-3, atol=0.0)


def test_none_is_a_real_branch_and_not_a_skipped_call():
    """A record that says "calibration ran and did nothing" is a different
    artifact from one that says "no calibration was configured", and only the
    second is what every existing checkpoint trained under."""
    model = _model(vocab=32, schema=contract.QUALIFICATION_FEEDBACK_SCHEMA)
    before = model.fuse.norm.weight.detach().clone()
    event = calibrate_gain(model, torch.ones(32, dtype=torch.long), rule="none",
                           step=0)
    assert event == {"rule": "none", "applied": False, "step": 0}
    assert torch.equal(model.fuse.norm.weight.detach(), before)

    with pytest.raises(KeyError, match="unknown gain calibration"):
        calibrate_gain(model, torch.ones(32, dtype=torch.long), rule="v3", step=0)

    plain = _model(vocab=32, schema="none")
    with pytest.raises(ValueError, match="feedback_schema='none'"):
        calibrate_gain(plain, torch.ones(32, dtype=torch.long),
                       rule="embedding_rms_at_switch_on_v1", step=0)


def test_a_gain_a_calibration_could_not_produce_is_refused():
    """The statistic is an RMS and is positive. A zero gain would delete the
    stack's input at every position and a negative one would invert it, and
    neither is a state any calibration rule in the contract can reach."""
    model = _model(vocab=32, schema=contract.QUALIFICATION_FEEDBACK_SCHEMA)
    for value in (0.0, -0.5):
        with pytest.raises(ValueError, match="delete or invert"):
            model.fuse.set_shared_gain(value)
    with pytest.raises(ValueError, match="not a number"):
        model.fuse.set_shared_gain(float("nan"))


def test_a_calibrated_run_fires_once_at_the_transition_and_records_it(
        tmp_path, monkeypatch):
    """End to end, at four steps rather than 24,000: the event is in the record,
    it fired at the step the schedule declares, and the gain the checkpoint
    carries is the value the event names."""
    monkeypatch.setattr(dm.train, "RUNS", tmp_path)
    cfg = _tiny(steps=4, eval_every=4,
                feedback_schema=contract.QUALIFICATION_FEEDBACK_SCHEMA,
                pass_schedule="project_progressive_v1",
                gain_calibration="embedding_rms_at_switch_on_v1",
                tag="unit_calibrated")
    record = train(cfg, verbose=False)
    event = record["feedback"]["gain_calibration_event"]
    assert record["feedback"]["gain_calibration"] == "embedding_rms_at_switch_on_v1"
    assert record["config"]["gain_calibration"] == "embedding_rms_at_switch_on_v1"
    assert event["applied"] is True
    assert event["step"] == contract.feedback_phase_transition(
        cfg.steps, cfg.pass_schedule)
    assert event["feedback_schema"] == contract.QUALIFICATION_FEEDBACK_SCHEMA
    assert event["counted_input_positions"] > 0

    # An uncalibrated run of the same shape records the rule and no event, and
    # keeps the frozen initialisation it started with.
    plain = train(_tiny(steps=4, eval_every=4,
                        feedback_schema=contract.QUALIFICATION_FEEDBACK_SCHEMA,
                        pass_schedule="project_progressive_v1",
                        tag="unit_uncalibrated"), verbose=False)
    assert plain["feedback"]["gain_calibration"] == "none"
    assert plain["feedback"]["gain_calibration_event"] is None


def test_a_budget_that_never_reaches_the_transition_is_refused_before_it_runs(
        tmp_path, monkeypatch):
    """A cell whose declared lever never moved would be indistinguishable from
    one where it moved and changed nothing, so the refusal is at configuration
    time rather than after the run."""
    monkeypatch.setattr(dm.train, "RUNS", tmp_path)
    with pytest.raises(ValueError, match="would never fire"):
        train(_tiny(steps=1, eval_every=1,
                    feedback_schema=contract.QUALIFICATION_FEEDBACK_SCHEMA,
                    pass_schedule="project_progressive_v1",
                    gain_calibration="embedding_rms_at_switch_on_v1",
                    tag="unit_no_transition"), verbose=False)


def test_the_calibration_reads_no_outcome():
    """Model-blind in the sense the contract means: a deterministic function of
    the embedding table and the corpus histogram. There is no parameter through
    which a loss, a validation score, a sample or a case could arrive."""
    import inspect

    signature = inspect.signature(calibrate_gain)
    assert list(signature.parameters) == ["model", "counts", "rule", "step"]
    text = inspect.getsource(calibrate_gain) + inspect.getsource(embedding_scale)
    for forbidden in ("loss", "val_bits", "Delta", "hit_own", "logits", "sample"):
        assert forbidden not in text, forbidden
    assert contract.GAIN_CALIBRATION_RATIONALE["rule"] == (
        contract.CORRECTION_GAIN_CALIBRATION)
    assert contract.GAIN_CALIBRATION_RATIONALE["initial_gain"] == (
        contract.FUSED_NORM_GAIN)
    assert contract.GAIN_CALIBRATION_RATIONALE[
        "jitter_half_width_unchanged"] == contract.JITTER_HALF_WIDTH
