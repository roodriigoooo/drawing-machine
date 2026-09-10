"""F3: the sequential scorer and the paired inference that reads it.

The stage's load-bearing claim is that the standard arm of this scorer is
Direction 2's estimand, computed a different way. Everything else -- the mode
gain, the structure interaction, the shared resamples -- is a difference of two
such numbers, so if that claim fails nothing downstream means anything.

The three scripted models are the specificity half. An instrument that returns a
positive `G` for a model that merely got *generally* better under soft decoding,
or for one that lurches at any prefix edit, cannot support the claim the pilot
exists to make. Each of the three is a null the real thing has to survive.
"""

from __future__ import annotations

import json
import math

import pytest
import torch

from dm.data import synthetic
from dm.eval import feedback_contract as contract
from dm.eval.context import components, score_cases, score_spans
from dm.eval.context_cases import CorpusStats, build_step_cases
from dm.eval.feedback import (
    bootstrap,
    case_requests,
    channel_scales,
    component_draws,
    generic_guards,
    holm,
    mode_gain,
    paired_difference,
    representation_interaction,
    score_cases_sequential,
    score_spans_sequential,
    sequential_matches_full_forward,
    smoke_stability,
    stability,
    stability_subset,
    stability_verdict,
    structure_interaction,
    validate_generation_report,
    validate_score_report,
)
from dm.isa.codec import ByteCodec
from dm.isa.state import LanguagePolicy
from dm.models.transformer import Config, DrawingLM

CODEC = ByteCodec()
BONUS = 4.0


def _cases(max_cases: int = 6):
    train, val = synthetic.split(3000, 300, seed=0, tier=1, flatten=True)
    policy = LanguagePolicy.from_programs(train + val)
    stats = CorpusStats.from_programs(train, label="synthetic:test", split="train")
    cases, _ = build_step_cases(val, policy, CODEC, stats, max_cases=max_cases)
    return cases


CASES = _cases()


def _model(seed: int = 0, schema: str = "glu_v1") -> DrawingLM:
    torch.manual_seed(seed)
    return DrawingLM(Config(vocab_size=CODEC.vocab_size, d_model=16, n_layers=2,
                            n_heads=2, max_len=512,
                            feedback_schema=schema)).eval()


# ---------------------------------------------------------------------------
# scripted models
#
# Each returns uniform logits with one bonus on the gold symbol, so a request's
# bits are a monotone function of its bonus and the three behaviours differ in
# *which* requests get one. That is the only thing under test here: whether the
# instrument can tell them apart.


class Scripted(torch.nn.Module):
    """A model whose per-request gold-symbol bonus is a scripted function."""

    def __init__(self, vocab: int, rule) -> None:
        super().__init__()
        self.vocab = vocab
        self.bonus: dict[tuple, float] = {}
        for case in CASES:
            for prefix, target in case_requests(case):
                key = (tuple(CODEC.encode(prefix)), tuple(CODEC.encode(target)))
                for mode in ("standard", "soft"):
                    self.bonus[(mode, *key)] = rule(case, prefix, target, mode)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # Uniform: the prefix costs the same under every world, so the prefix
        # column contributes nothing to any contrast.
        return torch.zeros(idx.shape[0], idx.shape[1], self.vocab)

    def teacher_forced(self, prompt: torch.Tensor, continuation: torch.Tensor,
                       bos: int = 1, classes=None, device="cpu",
                       mode: str = "standard") -> torch.Tensor:
        rows, length = continuation.shape
        logits = torch.zeros(rows, length, self.vocab)
        for row in range(rows):
            key = (mode, tuple(prompt[row].tolist()),
                   tuple(continuation[row].tolist()))
            boost = self.bonus[key]
            for position in range(length):
                logits[row, position, int(continuation[row, position])] += boost
        return logits


def _relation_follower(case, prefix, target, mode):
    """Prefers the target the prefix's relation implies, and only under soft.

    The relation survives an unrelated block edit, so `block_prefix_b` still
    carries world A's -- which is what makes this model's unrelated-block
    contrast zero.
    """
    if mode != "soft":
        return 0.0
    relation = "b" if prefix == case.prefix_b else "a"
    world = ("a" if target == case.target_a
             else "b" if target == case.target_b else None)
    return BONUS if world == relation else 0.0


def _generic_improver(case, prefix, target, mode):
    """Everything gets cheaper under soft decoding, uniformly."""
    return BONUS if mode == "soft" else 0.0


def _edit_reactor(case, prefix, target, mode):
    """Lurches toward world B's target whenever the prefix was edited at all.

    Relation-blind: it cannot tell a relation-bearing edit from an irrelevant
    one, so its unrelated-block contrast is as large as its primary.
    """
    if mode != "soft":
        return 0.0
    return BONUS if (prefix != case.prefix_a and target == case.target_b) else 0.0


def _scored(rule) -> tuple[list[dict], list[dict]]:
    model = Scripted(CODEC.vocab_size, rule)
    return (
        score_cases_sequential(model, CASES, CODEC, mode="standard"),
        score_cases_sequential(model, CASES, CODEC, mode="soft"),
    )


# ---------------------------------------------------------------------------
# the scorer is Direction 2's estimand


def test_the_request_order_is_direction_2s():
    """Twelve requests, in the order `score_cases` scores them. A permutation
    would silently swap a control for the primary and every number after it would
    be a different estimand with the right name."""
    for case in CASES:
        requests = case_requests(case)
        assert len(requests) == contract.REQUESTS_PER_CASE
        assert requests[0] == (case.prefix_a, case.target_a)
        assert requests[3] == (case.prefix_b, case.target_b)
        assert requests[4] == (case.prefix_a, case.control_target_a)
        assert requests[8] == (case.block_prefix_a, case.target_a)


def test_generic_guards_score_empty_prompt_and_publish_all_three_deltas():
    """The ordinary validation guard uses an empty prompt by design.

    This also pins that validation, valid-halt and truncation quantities are
    emitted together and that the two generation modes receive one uniform
    block rather than separate RNG streams.
    """
    programs = [b"\x00", b"\x01\x02\x00"]
    variates = torch.rand(
        4, 8, generator=torch.Generator().manual_seed(19), dtype=torch.float32
    )
    result = generic_guards(
        _model(), programs, CODEC, max_len=64, generation_n=4,
        generation_max_new=8, variates=variates,
    )
    values = result["values"]
    assert result["validation"]["programs"] == len(programs)
    assert result["variates_shape"] == [4, 8]
    assert set(values) == {
        "validation_cost_delta_bits_per_drawing",
        "valid_halt_delta", "truncation_rate_increase", "valid_halt_loss",
    }
    assert values["valid_halt_loss"] == pytest.approx(
        max(0.0, -values["valid_halt_delta"])
    )


def test_sequential_standard_reproduces_the_batched_scorer():
    """The claim the whole stage rests on.

    Not bit-for-bit and it cannot be: one path attends through a preallocated KV
    cache a position at a time, the other runs one batched forward over the whole
    sequence, and `scaled_dot_product_attention` reduces over different shapes in
    the two. What matters is the size of the disagreement against the size of the
    effect -- the frozen floor is 0.02 bits per target byte and this is four
    orders of magnitude under it.
    """
    model = _model()
    requests = [pair for case in CASES for pair in case_requests(case)]
    report = sequential_matches_full_forward(model, requests, CODEC)
    assert report["matches"], report
    assert report["max_abs_byte_bits_delta"] < 1e-5
    assert report["requests"] == len(CASES) * contract.REQUESTS_PER_CASE


def test_the_sequential_row_reproduces_score_cases_contrast_for_contrast():
    """Same cases, same arithmetic, same numbers. `D`, both controls and the
    per-field breakdown all have to land on the batched scorer's values, because
    Direction 3's `Delta_standard` is quoted against Direction 2's published
    `Delta`."""
    model = _model()
    batched = score_cases(model, CASES, CODEC)
    sequential = score_cases_sequential(model, CASES, CODEC, mode="standard")
    for left, right in zip(batched, sequential):
        assert left["case_id"] == right["case_id"]
        for key in ("D", "D_target_control", "D_block_control"):
            assert abs(left[key]["bits"] - right[key]["bits"]) < 1e-4, key
        assert abs(left["D_minus_control"]["bits_per_byte"]
                   - right["Delta"]["bits_per_byte"]) < 1e-5
        for name, value in left["fields"].items():
            assert abs(value - right["fields"][name]) < 1e-4, name


def test_the_scorer_persists_raw_symbol_nlls():
    """A saved contrast cannot be re-derived into its parts, and every failure
    mode this pilot can have is diagnosed from the parts."""
    model = _model()
    rows = score_cases_sequential(model, CASES, CODEC, mode="soft")
    for row, case in zip(rows, CASES):
        raw = row["raw_symbol_nll_bits"]
        assert set(raw) == {"primary", "target_control", "block_control"}
        for block, requests in raw.items():
            assert set(requests) == {
                "a_given_a", "b_given_a", "a_given_b", "b_given_b"
            }
            for name, symbols in requests.items():
                assert len(symbols) == case.target_bytes * CODEC.stride, name
                # The four-way block totals are the raw NLLs summed, or the
                # contrast is not a function of what was persisted. Relative,
                # because the total is reached by adding ~120 float32 values in
                # two different orders and float addition is not associative.
                assert sum(symbols) == pytest.approx(
                    row[block]["nll_bits"][name], rel=1e-6
                )


def test_all_twelve_raw_symbol_arrays_survive_json_round_trip():
    row = score_cases_sequential(_model(), CASES[:1], CODEC, mode="standard")[0]
    restored = json.loads(json.dumps(row))
    assert restored["raw_symbol_nll_bits"] == row["raw_symbol_nll_bits"]


def test_the_first_target_symbol_is_identical_in_standard_and_soft():
    """The structural zero §2.2 requires be reported rather than averaged away.

    Soft prefills in standard mode, so that symbol's distribution is the same in
    both arms by construction. A pipeline that showed a difference there would be
    measuring its own bookkeeping, and one that hid it inside a total would
    dilute the effect by a factor set only by target length.
    """
    model = _model()
    standard = score_cases_sequential(model, CASES, CODEC, mode="standard")
    soft = score_cases_sequential(model, CASES, CODEC, mode="soft")
    for left, right in zip(standard, soft):
        assert left["first_symbol_bits"] == right["first_symbol_bits"]
    # And the tail does move, or `soft` is not doing anything at all.
    assert any(
        left["raw_symbol_nll_bits"]["primary"]["a_given_a"][1:]
        != right["raw_symbol_nll_bits"]["primary"]["a_given_a"][1:]
        for left, right in zip(standard, soft)
    )


def test_a_codec_that_straddles_the_prefix_seam_is_refused():
    """A codec whose units span the prefix boundary would put the target's first
    symbol partly inside the prompt, and every per-byte column after it would be
    attributed to the wrong byte."""

    class Straddling(ByteCodec):
        name = "straddling"

        def encode(self, program: bytes) -> list[int]:
            return super().encode(program)[:-1] or [2]

    with pytest.raises(ValueError, match="independently"):
        score_spans_sequential(_model(), [(b"\x01\x02\x03", b"\x00")],
                               Straddling())


def test_a_request_past_max_len_is_refused_rather_than_truncated():
    with pytest.raises(ValueError, match="exceeds max_len"):
        score_spans_sequential(_model(), [(CASES[0].prefix_a, CASES[0].target_a)],
                               CODEC, max_len=4)


# ---------------------------------------------------------------------------
# the three scripted behaviours have to come apart


def test_a_relation_follower_shows_a_gain_with_a_null_block_control():
    """The signature the pilot is looking for: soft decoding buys preference for
    the target the prefix's relation implies, and an irrelevant edit buys
    nothing."""
    standard, soft = _scored(_relation_follower)
    gains = mode_gain(standard, soft)
    assert all(row["G"]["bits_per_byte"] > 0 for row in gains)
    for row in soft:
        assert row["D"]["bits"] > 0
        assert abs(row["D_block_control"]["bits"]) < 1e-6
        assert abs(row["D_target_control"]["bits"]) < 1e-6


def test_a_generic_soft_improver_shows_no_gain_at_all():
    """Everything got cheaper, so nothing got *preferred*. The donor-target
    control subtracts a uniform improvement exactly, which is what it is for."""
    standard, soft = _scored(_generic_improver)
    for row in soft:
        assert abs(row["Delta"]["bits"]) < 1e-6
    assert all(abs(row["G"]["bits_per_byte"]) < 1e-6
               for row in mode_gain(standard, soft))


def test_an_edit_reactor_is_caught_by_the_unrelated_block_control():
    """The failure mode `Delta` alone cannot see.

    A model that lurches at *any* prefix edit scores a positive `D` and a
    positive `Delta`, because the donor-target control holds the prefix pair
    fixed. Only the unrelated-block contrast separates it: an irrelevant edit
    moves this model as much as the relation-bearing one does.
    """
    standard, soft = _scored(_edit_reactor)
    gains = mode_gain(standard, soft)
    assert all(row["G"]["bits_per_byte"] > 0 for row in gains), (
        "the naive reading is positive, which is the point"
    )
    for row in soft:
        assert row["D"]["bits"] > 0
        assert abs(row["D_block_control"]["bits"] - row["D"]["bits"]) < 1e-6
        assert abs(row["Delta_block"]["bits"]) < 1e-6


def test_the_block_control_is_what_separates_the_follower_from_the_reactor():
    """Stated as one comparison, because it is the discriminating column and a
    report that dropped it would rank the two models identically."""
    _, follower = _scored(_relation_follower)
    _, reactor = _scored(_edit_reactor)
    for left, right in zip(follower, reactor):
        assert abs(left["D_block_control"]["bits"]) < 1e-6
        assert right["D_block_control"]["bits"] > 1e-6
        assert left["Delta_block"]["bits"] > 1e-6
        assert abs(right["Delta_block"]["bits"]) < 1e-6


# ---------------------------------------------------------------------------
# paired inference


def test_contrasts_are_formed_per_case_before_anything_is_averaged():
    """Averaging first and differencing after is the same number only when every
    cell holds exactly the same cases with the same weights, and one dropped case
    breaks that while still producing a plausible table."""
    standard, soft = _scored(_relation_follower)
    gains = mode_gain(standard, soft)
    assert [row["case_id"] for row in gains] == [c.case_id for c in CASES]
    for row, left, right in zip(gains, soft, standard):
        assert row["G"]["bits"] == pytest.approx(
            left["Delta"]["bits"] - right["Delta"]["bits"]
        )


def test_a_misaligned_pair_is_refused_rather_than_differenced():
    """Every downstream difference is positional and every bootstrap draw is a
    row index, so a silent misalignment would pair case 7's soft score with case
    8's standard one and publish it as an effect."""
    standard, soft = _scored(_relation_follower)
    with pytest.raises(ValueError, match="case order differs"):
        mode_gain(standard, list(reversed(soft)))
    with pytest.raises(ValueError, match="same cases on both sides"):
        mode_gain(standard, soft[:-1])


def test_mode_gain_refuses_rows_scored_in_the_wrong_mode():
    standard, _ = _scored(_relation_follower)
    with pytest.raises(ValueError, match="expected soft rows"):
        mode_gain(standard, standard)


def test_the_structure_interaction_subtracts_the_destroyed_arm():
    """`I_structure = G_relational - G_destroyed`. A relation-destroyed arm that
    produced the same gain would mean the gain is generic recurrence, and this is
    the only quantity that can say so."""
    relational = mode_gain(*_scored(_relation_follower))
    destroyed = mode_gain(*_scored(_generic_improver))
    interaction = structure_interaction(relational, destroyed)
    for row, left, right in zip(interaction, relational, destroyed):
        assert row["I_structure"]["bits_per_byte"] == pytest.approx(
            left["G"]["bits_per_byte"] - right["G"]["bits_per_byte"]
        )
    assert all(row["I_structure"]["bits_per_byte"] > 0 for row in interaction)


def test_the_representation_interaction_is_a_difference_of_interactions():
    relational = mode_gain(*_scored(_relation_follower))
    destroyed = mode_gain(*_scored(_generic_improver))
    interaction = structure_interaction(relational, destroyed)
    exploratory = representation_interaction(interaction, interaction)
    assert all(abs(row["I_representation"]["bits_per_byte"]) < 1e-12
               for row in exploratory)


# ---------------------------------------------------------------------------
# shared resamples


def test_every_cell_resamples_the_same_components_in_the_same_order():
    """Two cells resampled from two independently seeded generators would differ
    by their resampling accident as well as by the thing under test, and a
    difference of two such intervals means nothing."""
    standard, soft = _scored(_relation_follower)
    left = component_draws(standard, reps=64)
    right = component_draws(soft, reps=64)
    assert left.draws == right.draws
    assert left.seed == contract.seed_for("bootstrap")
    assert left.clusters == len(components(standard))


def test_a_resample_built_for_another_table_is_refused():
    standard, _ = _scored(_relation_follower)
    resample = component_draws(standard, reps=16)
    with pytest.raises(ValueError, match="built for"):
        bootstrap(standard[:-1], "D", resample)


def test_the_bootstrap_reports_the_interval_and_whether_it_excludes_zero():
    standard, soft = _scored(_relation_follower)
    gains = mode_gain(standard, soft)
    resample = component_draws(gains, reps=256)
    summary = bootstrap(gains, "G", resample)
    assert summary["mean"] > 0 and summary["excludes_zero"] is True
    assert summary["ci95"][0] <= summary["mean"] <= summary["ci95"][1]
    assert summary["cluster_unit"] == contract.BOOTSTRAP_UNIT
    assert summary["reps"] == 256
    # Floored at one replicate: a finite resample never reports an exact zero.
    assert summary["p_value"] >= 1 / 256


def test_a_null_gain_keeps_zero_inside_its_interval():
    standard, soft = _scored(_generic_improver)
    gains = mode_gain(standard, soft)
    summary = bootstrap(gains, "G", component_draws(gains, reps=256))
    assert summary["excludes_zero"] is False
    assert summary["ci95"][0] <= 0.0 <= summary["ci95"][1]


def test_the_component_bootstrap_matches_direction_2s_procedure():
    """Same graph, same generator, same key order. Direction 3's `G` is a
    difference of two `Delta`s on Direction 2's own cases, so the interval has to
    be built over the same clusters or the two stages cannot be read together."""
    import random

    standard, _ = _scored(_relation_follower)
    groups = components(standard)
    keys = sorted(groups)
    rng = random.Random(contract.seed_for("bootstrap"))
    expected = tuple(
        tuple(index
              for key in [keys[rng.randrange(len(keys))] for _ in keys]
              for index in groups[key])
        for _ in range(32)
    )
    assert component_draws(standard, reps=32).draws == expected


# ---------------------------------------------------------------------------
# multiplicity


def test_holm_steps_down_over_the_two_structure_tests():
    rejected = holm({"I_structure[bit]": 0.001, "I_structure[byte]": 0.02})
    assert rejected["I_structure[bit]"]["rejected"] is True
    assert rejected["I_structure[bit]"]["threshold"] == pytest.approx(0.025)
    assert rejected["I_structure[byte]"]["rejected"] is True
    assert rejected["I_structure[byte]"]["threshold"] == pytest.approx(0.05)


def test_holm_stops_at_the_first_failure():
    """Step-down's whole validity: once a hypothesis fails, every larger p in the
    family fails too, whatever its own threshold would have allowed."""
    rejected = holm({"a": 0.4, "b": 0.03})
    assert rejected["b"]["rejected"] is False and rejected["a"]["rejected"] is False
    assert rejected["b"]["p_adjusted"] == pytest.approx(0.06)


def test_holm_adjusted_p_values_are_monotone():
    rejected = holm({"a": 0.001, "b": 0.002, "c": 0.9})
    adjusted = [rejected[name]["p_adjusted"] for name in ("a", "b", "c")]
    assert adjusted == sorted(adjusted)
    assert all(value <= 1.0 for value in adjusted)


# ---------------------------------------------------------------------------
# stability


def _stability_programs(n: int = 40) -> list[bytes]:
    """A small held-out corpus with a spread of lengths, so the long-row filter
    has something to filter."""
    _, val = synthetic.split(200, n, seed=3, tier=1, flatten=True)
    return val


def _validation_batch(rows: int = 4, width: int = 24):
    generator = torch.Generator().manual_seed(5)
    inputs = torch.randint(2, CODEC.vocab_size, (rows, width), generator=generator)
    targets = torch.randint(2, CODEC.vocab_size, (rows, width), generator=generator)
    targets[-1, width // 2 :] = 0  # PAD, so the quantiles skip padded positions
    return inputs, targets


def test_the_stability_report_covers_every_frozen_pass_depth():
    model = _model()
    inputs, targets = _validation_batch()
    report = stability(model, inputs, targets, passes=(0, 1, 2, 4))
    assert [row["passes"] for row in report["passes"]] == [0, 1, 2, 4]
    assert report["passes"][0]["fused_input_rms_p99"] is None
    assert report["passes"][0]["update_q95"] == 0.0
    assert all(row["finite"] for row in report["passes"])
    assert report["standard_input_rms_p99"] > 0


def test_every_rms_quantile_skips_padded_positions():
    """A padded input position carries the PAD embedding row, which is also the
    tied head's PAD row and is therefore trained hard as a *negative class*
    rather than as an input; its hidden state carries no loss at all. Quantiles
    taken over a padded majority describe the padding. Pinned by making the
    padding differ: if any reported quantile moves, it was reading padding."""
    model = _model()
    inputs, targets = _validation_batch(rows=4, width=24)
    targets[:, 12:] = 0  # every row padded from the same position
    narrow = stability(model, inputs, targets, passes=(0, 1, 2))

    # Same scored region, wildly different padded region.
    padded = inputs.clone()
    padded[:, 12:] = 0
    wide = stability(model, padded, targets, passes=(0, 1, 2))

    assert wide["scored_positions"] == narrow["scored_positions"] == 4 * 12
    assert wide["padded_positions"] == narrow["padded_positions"] == 4 * 12
    for key in ("standard_input_rms_p99", "baseline_hidden_rms_p99",
                "baseline_bits_per_drawing"):
        assert wide[key] == pytest.approx(narrow[key], rel=1e-6), key
    for left, right in zip(narrow["passes"], wide["passes"]):
        assert left["hidden_rms_p99"] == pytest.approx(right["hidden_rms_p99"])
        assert left["update_q95"] == pytest.approx(right["update_q95"])


# ---------------------------------------------------------------------------
# F5b: the repaired stability criteria


def test_finiteness_and_the_logit_bound_skip_padded_positions():
    """The same repair the RMS quantiles already had, on the two clauses that
    still read the whole batch.

    A padded position's logits are the tied head's reading of the PAD row, which
    is trained as a negative class and is most of the batch. `max_abs_logit` over
    it measures how confidently the model rejects PAD; `finite` over it can fail
    on a state no loss ever touched. Both are gated clauses, so both have to be
    scored-position quantities or the gate is about padding.
    """
    model = _model()
    inputs, targets = _validation_batch(rows=4, width=24)
    targets[:, 12:] = 0
    narrow = stability(model, inputs, targets, passes=(0, 1, 2))

    padded = inputs.clone()
    padded[:, 12:] = 0
    wide = stability(model, padded, targets, passes=(0, 1, 2))
    for left, right in zip(narrow["passes"], wide["passes"]):
        assert left["max_abs_logit"] == pytest.approx(right["max_abs_logit"])
        assert left["finite"] == right["finite"]

    # And the masking has to be load-bearing rather than incidental: put a very
    # loud embedding row at the padded positions only, and the reported bound
    # must not move even though the unmasked maximum does. (A non-finite value
    # cannot be confined to the padded region at all -- the head is tied to the
    # embedding table, so poisoning any row poisons that logit column at every
    # position, which is exactly why the *positions* are what get masked.)
    with torch.no_grad():
        model.embed.weight[5] *= 500.0
    loud = inputs.clone()
    loud[:, 12:] = 5
    quiet = inputs.clone()
    quiet[:, 12:] = 0
    assert torch.equal(loud[:, :12], quiet[:, :12])
    with torch.no_grad():
        unmasked_loud = float(model(loud).abs().max())
        unmasked_quiet = float(model(quiet).abs().max())
    assert unmasked_loud > 2 * unmasked_quiet
    for left, right in zip(stability(model, quiet, targets, passes=(0, 1))["passes"],
                           stability(model, loud, targets, passes=(0, 1))["passes"]):
        assert left["max_abs_logit"] == pytest.approx(right["max_abs_logit"])
        assert left["finite"] == right["finite"]


def test_the_stability_subset_is_long_frozen_and_model_blind():
    """The probe's rows are chosen from lengths and a frozen seed, never from
    model output, and every one of them outlives the deepest pass.

    The smoke took the first sixteen validation programs; twelve of them were
    fully converged at pass 32, so the clause read `0.0008` where the same
    checkpoint on the real split fails. A subset that cannot exhibit the failure
    is not a gate.
    """
    programs = _stability_programs()
    subset = stability_subset(programs, CODEC, size=4, deepest_pass=8)
    assert subset["size"] == len(subset["indices"]) == 4
    assert subset["seed"] == contract.seed_for("stability")
    assert subset["deepest_pass"] == 8
    assert all(length > 8 for length in subset["lengths"])
    assert subset == stability_subset(programs, CODEC, size=4, deepest_pass=8)
    # Identities, not just a count: the gate has to be re-runnable on the same
    # rows by someone holding the corpus and the protocol.
    assert subset["digest"] == stability_subset(
        programs, CODEC, size=4, deepest_pass=8)["digest"]
    assert stability_subset(list(reversed(programs)), CODEC, size=4,
                            deepest_pass=8)["digest"] != subset["digest"]

    # Fails closed rather than shrinking: a subset smaller than the frozen size
    # measures something the protocol did not name.
    with pytest.raises(ValueError, match="incomplete"):
        stability_subset(programs, CODEC, size=len(programs), deepest_pass=8)


def test_the_stability_subset_never_reads_the_model():
    """Invariant 1: no model output chooses a case, a row or a threshold. The
    selector takes a corpus and a codec and has no parameter that could carry
    one."""
    import inspect

    signature = inspect.signature(stability_subset)
    assert "model" not in signature.parameters


def test_stability_costs_are_reported_per_semantic_byte():
    """Invariant 6, and the trap `PLAN.md` names by name. The bit arm spends
    eight symbols on one bytecode byte, so a symbol-normalized cost is not
    comparable across representations and calling it `bits_per_byte` makes the
    two arms look like a result."""
    model = _model()
    inputs, targets = _validation_batch(rows=4, width=24)
    per_symbol = stability(model, inputs, targets, passes=(0, 2),
                           symbols_per_byte=1)
    per_byte = stability(model, inputs, targets, passes=(0, 2),
                         symbols_per_byte=8)
    assert per_symbol["symbols_per_byte"] == 1
    assert per_byte["symbols_per_byte"] == 8
    deep_symbol = per_symbol["passes"][-1]
    deep_byte = per_byte["passes"][-1]
    assert deep_byte["converged_bits_per_byte"] == pytest.approx(
        8 * deep_symbol["converged_bits_per_byte"])
    assert deep_byte["standard_converged_bits_per_byte"] == pytest.approx(
        8 * deep_symbol["standard_converged_bits_per_byte"])
    # bits/drawing is a program-level unit and is the same number in both.
    assert deep_byte["bits_per_drawing"] == pytest.approx(
        deep_symbol["bits_per_drawing"])


def test_the_gate_reads_the_wavefront_and_withdraws_the_settling_clause():
    """The deepest-pass `update_q95` blends exact zeros from the converged prefix
    with a live tail, so on a long-enough subset it reads sequence length. The
    gate reads the wavefront instead, and `update_q95_not_settling` is withdrawn
    rather than left failing: no trajectory distinguishes a bad channel from a
    long sequence under it (`docs/feedback-stability.md` §1)."""
    model = _model()
    report = stability(model, *_validation_batch(), passes=(0, 1, 8, 32))
    report["valid_halt_loss"] = 0.0
    rows = {row["passes"]: row for row in report["passes"]}

    # A blended quantile far past the bound, with a settled wavefront: passes.
    for row in rows.values():
        row["update_q95"] = 9.0
        row["update_q95_wavefront"] = 0.0
    verdict = stability_verdict(report)
    assert "update_q95" not in verdict["failures"]
    assert "update_q95_wavefront" not in verdict["failures"]
    assert "update_q95_not_settling" not in verdict["failures"]

    rows[32]["update_q95_wavefront"] = 0.4
    assert "update_q95_wavefront" in stability_verdict(report)["failures"]

    rows[32]["update_q95_wavefront"] = 0.2
    rows[8]["update_q95_wavefront"] = 0.05
    failures = stability_verdict(report)["failures"]
    assert "update_q95_wavefront" not in failures
    assert failures["update_q95_wavefront_not_settling"] == [0.05, 0.2]

    assert "update_q95_not_settling" in contract.WITHDRAWN_CLAUSE_REASONS
    assert (contract.QUALIFICATION_GATE["withdrawn_clauses"]
            == ["update_q95_not_settling"])


def test_the_stability_report_declares_the_repaired_shape():
    """The fields changed meaning -- scored positions, semantic bytes, a
    wavefront column the gate reads -- so an old report and a new one cannot
    carry the same schema number. `REPORT_SCHEMA` is frozen in v0 and describes
    what v0 defined; the repaired shape gets its own counter."""
    report = stability(_model(), *_validation_batch(), passes=(0, 1))
    assert report["schema"] == contract.STABILITY_REPORT_SCHEMA
    assert contract.STABILITY_REPORT_SCHEMA != contract.REPORT_SCHEMA
    assert report["scored_positions_only"] is True


def test_the_fused_prefill_converges_one_position_per_pass():
    """The iteration is triangular, and everything the deepest pass reports has
    to be read through that. Position 0 is always plain, so its state never
    moves; position `t` fuses `previous[t-1]` and attends only to `0..t`, so by
    induction `state_k[t]` is final for every `t < k` -- for *any* weights.

    Two consequences the gate depends on. The recurrence reaches its fixed point
    in exactly `T` passes rather than asymptotically, so 'still moving at 32' is
    a statement about sequence length, not about the channel. And a quantity
    averaged over all scored positions at pass `k` blends a converged prefix with
    a first-visit tail."""
    model = _model()
    inputs, _ = _validation_batch(rows=3, width=20)
    with torch.no_grad():
        _, state = model(inputs, return_state=True)
        history = [state]
        for _ in range(8):
            carried = torch.zeros_like(state)
            carried[:, 1:] = state[:, :-1]
            _, state = model(inputs, feedback=carried, plain=1, return_state=True)
            history.append(state)

    for k in range(1, 9):
        settled = (history[k][:, :k] == history[k - 1][:, :k]).all()
        assert settled, f"pass {k} moved inside its converged prefix"
        if k < inputs.shape[1]:
            moved = (history[k][:, k:] != history[k - 1][:, k:]).any()
            assert moved, f"pass {k} froze past its own wavefront"


def test_the_deepest_pass_separates_the_converged_prefix_from_the_wavefront():
    """`update_q95` mixes exact zeros from the converged prefix with the tail;
    on a subset shorter than the deepest pass it is zero whatever the channel
    does. The report therefore carries the wavefront-restricted number beside
    it, and prices the converged prefix against the standard forward on exactly
    the same positions."""
    model = _model()
    inputs, targets = _validation_batch(rows=4, width=24)
    report = stability(model, inputs, targets, passes=(0, 1, 4, 32))
    rows = {row["passes"]: row for row in report["passes"]}

    # Past the width every position is converged, so the gated number is zero by
    # construction and the wavefront set is empty.
    assert rows[32]["update_q95"] == 0.0
    assert math.isnan(rows[32]["update_q95_wavefront"])
    assert rows[32]["converged_positions"] == report["scored_positions"]
    assert rows[4]["converged_positions"] < report["scored_positions"]
    assert rows[4]["update_q95_wavefront"] > 0.0
    assert not math.isnan(rows[4]["standard_converged_bits_per_byte"])
    assert rows[0]["converged_positions"] == 0
    assert math.isnan(rows[0]["converged_bits_per_byte"])


def test_a_smoke_gates_on_breakage_and_only_reports_the_quality_clauses():
    """§7 invariant 14 cuts both ways: an engineering cell cannot move a
    threshold, and it cannot block a stage on one either. The split is a subset
    of the F7 clause names, never a widening, and nothing is suppressed."""
    verdict = {
        "passed": False,
        "failures": {
            "max_abs_logit": [32],
            "validation_increase_bits_per_drawing": 147.4,
            "update_q95": 0.9,
        },
    }
    split = smoke_stability(verdict)
    assert split["passed"] is False
    assert set(split["failures"]) == {"max_abs_logit"}
    assert set(split["deferred_clauses"]) == {
        "validation_increase_bits_per_drawing", "update_q95"}
    assert split["verdict"] is verdict
    quality_only = smoke_stability({"passed": False, "failures": {
        "validation_increase_bits_per_drawing": 147.4}})
    assert quality_only["passed"] is True and not quality_only["failures"]
    assert quality_only["deferred_clauses"]

    # A strict subset of what `stability_verdict` can emit: the smoke may narrow
    # the gate, never widen it, and a clause name it invents would silently stop
    # gating on anything.
    emitted = {"all_finite", "fused_input_rms_band", "hidden_rms_band",
               "max_abs_logit", "validation_increase_bits_per_drawing",
               "update_q95", "update_q95_not_settling"}
    assert set(split["gated_clauses"]) < emitted


def test_a_freshly_initialised_channel_settles_rather_than_diverging():
    """At initialisation the fused input is deliberately the size of the
    embedding, so iterating the prefill should not walk away. This is the
    engineering sanity check, not the F7 gate -- that one runs on trained
    weights.

    Read it narrowly. Two of the clauses it exercises pass here for structural
    reasons rather than because the channel is healthy: the batch is 24 wide and
    the deepest pass is 32, so every position is converged and both `update_q95`
    clauses are zero by construction
    (`test_the_deepest_pass_separates_the_converged_prefix_from_the_wavefront`);
    and an untrained embedding table still sits at the scale the fused-norm gain
    was initialised to match, so the band is trivially satisfied. What this does
    show is that nothing goes non-finite or loud over 32 iterations."""
    report = stability(_model(), *_validation_batch(),
                       passes=contract.STABILITY_PASSES)
    report["valid_halt_loss"] = 0.0
    verdict = stability_verdict(report)
    assert verdict["passed"], verdict["failures"]
    assert verdict["label"] == "stable"
    assert verdict["deepest_pass"] == 32


def test_the_gate_catches_a_channel_that_blows_up():
    """Reported per criterion, because `unstable_feedback` is a claim label and a
    label without the clause that produced it is not auditable."""
    report = stability(_model(), *_validation_batch(), passes=(0, 1, 8, 32))
    report["valid_halt_loss"] = 0.0
    report["passes"][-1]["hidden_rms_p99"] *= 100
    report["passes"][-1]["max_abs_logit"] = 1e4
    verdict = stability_verdict(report)
    assert verdict["passed"] is False
    assert verdict["label"] == "unstable_feedback"
    assert set(verdict["failures"]) >= {"hidden_rms_band", "max_abs_logit"}


def test_the_gate_catches_a_channel_that_collapses_toward_zero():
    """Decay is as broken a channel as blow-up, and only a band catches both.
    A collapsing state looks completely healthy in a loss curve."""
    report = stability(_model(), *_validation_batch(), passes=(0, 1, 8, 32))
    report["valid_halt_loss"] = 0.0
    report["passes"][-1]["fused_input_rms_p99"] = (
        report["standard_input_rms_p99"] * 0.01
    )
    assert "fused_input_rms_band" in stability_verdict(report)["failures"]


def test_the_gate_catches_a_wavefront_that_is_still_receding():
    """Still moving where movement is possible is not convergence, however small
    the step is.

    The blended `update_q95` twin of this clause is withdrawn: the iteration is
    triangular, so a deepest-pass quantile over *all* scored positions falls as
    the subset's rows get shorter and rises as they get longer, whatever the
    channel does. Restricted to the wavefront the same arithmetic is a statement
    about the operator.
    """
    report = stability(_model(), *_validation_batch(), passes=(0, 1, 8, 32))
    report["valid_halt_loss"] = 0.0
    rows = report["passes"]
    rows[-1]["update_q95_wavefront"] = rows[-2]["update_q95_wavefront"] + 0.01
    failures = stability_verdict(report)["failures"]
    assert "update_q95_wavefront_not_settling" in failures
    assert "update_q95_not_settling" not in failures


def test_the_stability_gate_applies_the_valid_halt_loss_clause():
    report = stability(_model(), *_validation_batch(), passes=(0, 1, 8, 32))
    report["valid_halt_loss"] = contract.STABILITY_GATE["max_valid_halt_loss"] + 0.001
    verdict = stability_verdict(report)
    assert verdict["failures"]["valid_halt_loss"] == report["valid_halt_loss"]


def test_the_stability_gate_refuses_to_pass_without_the_validity_guard():
    report = stability(_model(), *_validation_batch(), passes=(0, 1, 8, 32))
    verdict = stability_verdict(report)
    assert verdict["failures"]["valid_halt_loss"] == "missing"


def test_channel_scales_reads_the_weight_side_scales_and_names_the_gain_role():
    """A weight-side reading, and one that says what the gain *is* in this arm.

    Under `glu_source_v2` the tensor is Listing 3's shared `input_rmsnorm_1`
    gain, applied to the gate input as well as to the post-mixin stack input; the
    two earlier arms carry it as a fused-output gain under the identical
    parameter name. It is not the fused input's RMS in any of them -- `RMSNorm`
    fixes the normalized product's magnitude, not its direction -- and it is not
    the frozen band, which `stability` measures on scored rows.

    The PAD row is reported apart from the content rows: it is the tied head's
    PAD row, trained as a negative class, and is not a scale the stack sees on
    real content.
    """
    model = _model(schema=contract.QUALIFICATION_FEEDBACK_SCHEMA)
    scales = channel_scales(model)
    assert scales["feedback_schema"] == contract.SOURCE_FEEDBACK_SCHEMA_V2
    assert "shared input norm" in scales["gain_role"]
    assert scales["shared_input_norm_gain_mean"] == pytest.approx(
        contract.FUSED_NORM_GAIN, abs=1e-9)
    assert scales["shared_input_norm_gain_std"] == pytest.approx(0.0, abs=1e-9)
    # At initialisation the gain is set to the embedding's own scale, which is
    # what makes a fused pass's plain prefix the size of the raw prefill it
    # imitates -- the right scale and direction, not an equality.
    assert scales["gain_over_embed_rms"] == pytest.approx(1.0, rel=0.1)
    assert scales["pad_row_rms"] > 0

    with torch.no_grad():  # grow the table, leave the gain alone
        model.embed.weight.mul_(4.0)
    grown = channel_scales(model)
    assert (grown["shared_input_norm_gain_mean"]
            == scales["shared_input_norm_gain_mean"])
    assert grown["gain_over_embed_rms"] == pytest.approx(
        scales["gain_over_embed_rms"] / 4.0, rel=1e-5)

    # The legacy arms carry the same tensor doing a different job, and the
    # reading has to say so rather than reporting one number for both.
    for legacy in ("glu_v1", contract.SOURCE_FEEDBACK_SCHEMA):
        assert channel_scales(_model(schema=legacy))["gain_role"] == (
            "fused-output gain"
        )

    with pytest.raises(ValueError, match="feedback_schema='none'"):
        channel_scales(_model(schema="none"))


def test_stability_refuses_a_batch_with_nothing_to_measure():
    """Masked quantiles make an all-PAD batch fail *open* if it is allowed
    through: every reference is `nan`, every band comparison is false, and the
    gate reports `stable` having measured nothing."""
    inputs, targets = _validation_batch()
    with pytest.raises(ValueError, match="incomplete"):
        stability(_model(), inputs, torch.zeros_like(targets))


def test_stability_refuses_a_checkpoint_with_no_channel():
    with pytest.raises(ValueError, match="feedback_schema='none'"):
        stability(_model(schema="none"), *_validation_batch())


# ---------------------------------------------------------------------------
# fail closed


def _score_report(**overrides) -> dict:
    base = {field: 0 for field in contract.REQUIRED_SCORE_FIELDS}
    base |= {
        "schema": contract.REPORT_SCHEMA, "status": "complete",
        "protocol": contract.PROTOCOL, "mode": "soft",
        "sequential_standard_matches_full_forward": True,
        "raw_symbol_nll_bits": {
            block: {
                name: [0.0]
                for name in ("a_given_a", "b_given_a", "a_given_b", "b_given_b")
            }
            for block in ("primary", "target_control", "block_control")
        },
    }
    return base | overrides


def test_a_complete_score_report_passes():
    validate_score_report(_score_report())


def test_a_score_report_missing_one_of_twelve_raw_request_arrays_is_incomplete():
    report = _score_report()
    del report["raw_symbol_nll_bits"]["block_control"]["a_given_a"]
    with pytest.raises(ValueError, match="incomplete"):
        validate_score_report(report)


@pytest.mark.parametrize("missing", ("D_block_control", "raw_symbol_nll_bits",
                                     "Delta", "model_seed"))
def test_a_score_report_missing_a_required_column_is_incomplete(missing):
    report = _score_report()
    del report[missing]
    with pytest.raises(ValueError, match="incomplete"):
        validate_score_report(report)


def test_a_report_that_never_finished_is_incomplete():
    with pytest.raises(ValueError, match="incomplete"):
        validate_score_report(_score_report(status="running"))


def test_an_unchecked_scorer_cannot_publish_a_score():
    """Without the equivalence check the standard arm is not known to be
    Direction 2's estimand, so the report has nothing to compare against."""
    with pytest.raises(ValueError,
                       match="never checked against the full-forward"):
        validate_score_report(
            _score_report(sequential_standard_matches_full_forward=False)
        )


def test_a_report_of_the_wrong_schema_is_incomplete():
    with pytest.raises(ValueError, match="incomplete"):
        validate_score_report(_score_report(schema=contract.REPORT_SCHEMA + 1))


def test_an_engineering_artifact_is_refused_by_the_scientific_gate():
    """One smoke cell has zero scientific decision value, and the marker is in
    the artifact rather than in a filename so the refusal is by schema."""
    from dm.eval.feedback import ENGINEERING, SCIENTIFIC, refuse_engineering

    refuse_engineering({"provenance": SCIENTIFIC}, "a real report")
    with pytest.raises(ValueError, match="incomplete"):
        refuse_engineering({"provenance": ENGINEERING}, "the smoke report")
    # Absent is refused as firmly as engineering: missing provenance is
    # incomplete, never a pass.
    with pytest.raises(ValueError, match="incomplete"):
        refuse_engineering({}, "an unmarked report")


def test_a_generation_report_needs_its_denominators():
    base = {field: 0 for field in contract.REQUIRED_GENERATION_FIELDS}
    base |= {"schema": contract.REPORT_SCHEMA, "status": "complete",
             "protocol": contract.PROTOCOL, "mode": "soft"}
    validate_generation_report(base)
    del base["reach_rate"]
    with pytest.raises(ValueError, match="incomplete"):
        validate_generation_report(base)


def test_nothing_here_pools_model_seeds():
    """Two named checkpoints are a checkpoint-conditional replication, never a
    population. The Module offers no function that takes two seeds' rows and
    returns one interval, and this pins that absence."""
    from dm.eval import feedback as module

    assert not any("pool" in name for name in module.__all__)
    assert contract.PROTOCOL_SCHEMA and not contract.protocol_dict(
        fixture_sha256="x"
    )["inference"]["seeds_pooled"]


def test_the_scorer_never_reads_a_threshold_from_its_own_output():
    """No output chooses a corpus row, donor, case, threshold or checkpoint. The
    scorer takes cases and returns numbers; nothing in it can drop a case."""
    model = _model()
    rows = score_cases_sequential(model, CASES, CODEC, mode="soft")
    assert len(rows) == len(CASES)
    assert [row["case_id"] for row in rows] == [case.case_id for case in CASES]


def test_paired_difference_keeps_the_dependence_identity_columns():
    """The bootstrap resamples connected source/donor components, so a derived
    table that dropped the donor columns could only resample sources -- which
    counts dependent observations as independent and shrinks every interval."""
    standard, soft = _scored(_relation_follower)
    row = paired_difference(soft, standard, "Delta", name="G")[0]
    assert {"source_index", "donor_index", "control_donor_index"} <= set(row)
    assert math.isfinite(row["G"]["bits_per_byte"])


def test_the_full_forward_scorer_is_left_exactly_as_direction_2_wrote_it():
    """Direction 2's module is digest-frozen inside its own protocol. This stage
    imports from it and must not have widened it."""
    model = _model()
    request = [(CASES[0].prefix_a, CASES[0].target_a)]
    assert set(score_spans(model, request, CODEC)[0]) == {
        "prefix_bits", "bits", "per_byte"
    }
