"""C4's contract: paired completions, matched controls, and a denominator.

`docs/directions.md` §4.9 asks whether the teacher-forced selectivity survives
the model's own outputs. That question is answerable only if the instrument is
pinned first, and the C4 run of 2026-08-16 was made before these tests existed
-- the one place Direction 2 inverted its own rule (§4.7, "tests precede the
run"). This file is that debt paid, and it is written the same way §4.7's eight
verifications are: the completions come from a scripted model, so every `D_gen`
below is checked against arithmetic on the case bytes rather than against a
checkpoint.

The scripted models are the argument. Three of them separate hypotheses that a
raw `D_gen` cannot:

- one that emits whatever its own prompt implies scores the full effect;
- one that reacts to *any* prefix edit scores the full effect too, and the
  block control takes all of it back;
- one that ignores the prefix scores exactly zero everywhere.
"""

from __future__ import annotations

import json
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import torch

from dm.data import synthetic
from dm.data.augment import Affine, apply
from dm.eval.context_c4 import (
    CompletionConfig,
    complete_cases,
    per_case,
    summarise_by_venue,
    summarise_completions,
)
from dm.eval.context_cases import CorpusStats, build_step_cases
from dm.isa.codec import ByteCodec
from dm.isa.spec import Op
from dm.isa.state import LanguagePolicy
from dm.models.transformer import PAD_TOKEN, Config, DrawingLM

HALT = bytes([int(Op.HALT)])


# ---------------------------------------------------------------------------
# fixtures


@lru_cache(maxsize=4)
def _corpus():
    train, val = synthetic.split(3000, 300, seed=0, tier=1, flatten=True)
    policy = LanguagePolicy.from_programs(list(train) + list(val))
    stats = CorpusStats.from_programs(train, label="synthetic:test", split="train")
    return tuple(train), tuple(val), policy, stats


@lru_cache(maxsize=4)
def _cases(max_cases: int = 4):
    _, val, policy, stats = _corpus()
    cases, _ = build_step_cases(list(val), policy, ByteCodec(), stats,
                                max_cases=max_cases)
    return tuple(cases), policy


class ScriptedLM(torch.nn.Module):
    """A decoder whose continuation is a lookup on its own prompt.

    It drives `HaltMonitor` exactly the way `DrawingLM.generate` does -- prime
    on the prompt, one `stride`-wide step per position, PAD once a row is done
    -- because C4 reads `monitor.done` for its termination column and a fixture
    that faked that column would test nothing.
    """

    def __init__(self, codec: ByteCodec, table: dict[bytes, bytes],
                 default: bytes = HALT) -> None:
        super().__init__()
        self.codec, self.table, self.default = codec, dict(table), default
        self.calls: list[dict] = []

    def generate(self, n, max_new, *, monitor, prompt, temperature=1.0,
                 top_k=None, device="cpu", forbid=(), variates=None, **kwargs):
        self.calls.append({"rows": n, "max_new": max_new, "variates": variates,
                           "prompt": prompt.clone()})
        given = prompt.shape[1]
        width = given + max_new
        out = torch.full((n, width), PAD_TOKEN, dtype=torch.long)
        out[:, :given] = prompt
        stride = monitor.stride
        for end in range(stride, given + 1, stride):
            monitor.step(out[:, end - stride:end].t().numpy())
        scripts = []
        for row in range(n):
            program = self.codec.decode(prompt[row].tolist())
            scripts.append(self.codec.encode(self.table.get(program,
                                                            self.default)))
        for step in range(given, width):
            offset = step - given
            column = torch.full((n,), PAD_TOKEN, dtype=torch.long)
            for row in range(n):
                if not monitor.done[row] and offset < len(scripts[row]):
                    column[row] = scripts[row][offset]
            out[:, step] = column
            monitor.step(out[:, step:step + 1].t().numpy())
        return out


def _hamming(left: bytes, right: bytes) -> float:
    return sum(a != b for a, b in zip(left, right)) / len(right)


def _run(model, cases, policy, *, draws=1, cap=48):
    config = CompletionConfig(draws=draws, cap=cap, top_k=None, temperature=1.0)
    rows = complete_cases(model, list(cases), ByteCodec(), policy,
                          config=config)
    return rows, per_case(rows, list(cases))


def _own_target_table(cases) -> dict[bytes, bytes]:
    """Emit the continuation this prompt's own relation implies.

    `block_prefix_b` is an *irrelevant* edit, so the relation it displays is
    still world A's and its compatible continuation is still `target_a`. A
    model that gets that right is the one the block control must not penalise.
    """
    table: dict[bytes, bytes] = {}
    for case in cases:
        table[case.prefix_a] = case.target_a + HALT
        table[case.prefix_b] = case.target_b + HALT
        table[case.block_prefix_b] = case.target_a + HALT
    return table


def _edit_reactive_table(cases) -> dict[bytes, bytes]:
    """Emit `target_b` after *any* prefix edit, relevant or not.

    Indistinguishable from the real thing on the primary contrast alone; the
    block control is the only column that separates them.
    """
    table: dict[bytes, bytes] = {}
    for case in cases:
        table[case.prefix_a] = case.target_a + HALT
        table[case.prefix_b] = case.target_b + HALT
        table[case.block_prefix_b] = case.target_b + HALT
    return table


# ---------------------------------------------------------------------------
# the estimand, against arithmetic on the case bytes


def test_a_model_that_follows_its_own_prompt_scores_the_byte_disagreement():
    cases, policy = _cases()
    model = ScriptedLM(ByteCodec(), _own_target_table(cases))
    _, rows = _run(model, cases, policy)
    index = {case.case_id: case for case in cases}
    assert len(rows) == len(cases)
    for row in rows:
        case = index[row["case_id"]]
        # c_A = y_A and c_B = y_B, so f(A) = f(B) = d(y_A, y_B) exactly.
        expected = _hamming(case.target_a, case.target_b)
        assert row["D_gen"]["value"] == pytest.approx(expected)
        assert row["D_gen"]["lean_a"] == pytest.approx(expected)
        assert row["D_gen"]["lean_b"] == pytest.approx(expected)
        assert row["hit_own_rate"] == 1.0
        assert row["hit_other_rate"] == 0.0
        assert row["reach_rate"] == 1.0


def test_a_prefix_independent_model_scores_exactly_zero_on_every_contrast():
    """§4.7 rule 5, in the generation column.

    The symmetric shape cancels an unconditional preference for either target
    string. If it did not, every number C4 reports would carry whichever target
    the model happened to like.
    """
    cases, policy = _cases()
    # One fixed continuation for every prompt: the model cannot be reacting to
    # the prefix, so no contrast may be nonzero.
    model = ScriptedLM(ByteCodec(), {}, default=cases[0].target_a + HALT)
    _, rows = _run(model, cases, policy)
    for row in rows:
        assert row["D_gen"]["value"] == pytest.approx(0.0, abs=1e-12)
        assert row["D_gen_target_control"]["value"] == pytest.approx(0.0, abs=1e-12)
        assert row["D_gen_block_control"]["value"] == pytest.approx(0.0, abs=1e-12)
        assert row["D_gen_minus_control"]["value"] == pytest.approx(0.0, abs=1e-12)
        assert row["D_gen_minus_block_control"]["value"] == pytest.approx(
            0.0, abs=1e-12)


def test_the_block_control_takes_back_an_edit_reactive_model_and_spares_a_real_one():
    """The whole reason C4 grew a control: two models, one `D_gen`.

    A model that follows the relation and a model that lurches at any prefix
    edit produce the *same* primary contrast. `Delta_gen` is what tells them
    apart, and it is the generation analogue of C3's `D_minus_block_control`.
    """
    cases, policy = _cases()
    codec = ByteCodec()
    _, honest = _run(ScriptedLM(codec, _own_target_table(cases)), cases, policy)
    _, reactive = _run(ScriptedLM(codec, _edit_reactive_table(cases)), cases,
                       policy)
    index = {case.case_id: case for case in cases}
    for left, right in zip(honest, reactive):
        case = index[left["case_id"]]
        expected = _hamming(case.target_a, case.target_b)
        # Identical on the primary contrast ...
        assert left["D_gen"]["value"] == pytest.approx(expected)
        assert right["D_gen"]["value"] == pytest.approx(expected)
        # ... and opposite once the irrelevant edit is subtracted.
        assert left["D_gen_block_control"]["value"] == pytest.approx(0.0, abs=1e-12)
        assert left["D_gen_minus_block_control"]["value"] == pytest.approx(expected)
        assert right["D_gen_block_control"]["value"] == pytest.approx(expected)
        assert right["D_gen_minus_block_control"]["value"] == pytest.approx(
            0.0, abs=1e-12)


def test_the_target_control_reuses_the_primary_completions():
    """The donor-target control costs no extra decode, and must not.

    It scores the *same* completions against the control targets, so it is
    computable from rows that were already generated -- which is why it is in
    the report at all.
    """
    cases, policy = _cases()
    model = ScriptedLM(ByteCodec(), _own_target_table(cases))
    raw, rows = _run(model, cases, policy)
    index = {case.case_id: case for case in cases}
    for row in rows:
        case = index[row["case_id"]]
        expected = 0.5 * (
            (_hamming(case.target_a, case.control_target_b)
             - _hamming(case.target_a, case.control_target_a))
            + (_hamming(case.target_b, case.control_target_a)
               - _hamming(case.target_b, case.control_target_b))
        )
        assert row["D_gen_target_control"]["value"] == pytest.approx(expected)
    # No extra generation: the control arms are the primary arms.
    primary = [row for row in raw if row["arm"] == "primary"]
    assert len(primary) == 2 * len(cases)


def test_swapping_the_worlds_preserves_d_gen_and_swapping_targets_reverses_it():
    """§4.7 rule 2, in the generation column."""
    cases, policy = _cases(max_cases=3)
    codec = ByteCodec()
    table = _own_target_table(cases)
    _, base = _run(ScriptedLM(codec, table), cases, policy)

    swapped = tuple(
        replace(case, prefix_a=case.prefix_b, prefix_b=case.prefix_a,
                target_a=case.target_b, target_b=case.target_a,
                block_prefix_a=case.prefix_b,
                control_target_a=case.control_target_b,
                control_target_b=case.control_target_a)
        for case in cases
    )
    # The block control follows the worlds; §4.4's trap is that relabelling one
    # without the other reverses the control contrast silently.
    swapped_table = dict(table)
    for case, moved in zip(cases, swapped):
        swapped_table[moved.block_prefix_b] = case.target_b + HALT
    _, other = _run(ScriptedLM(codec, swapped_table), swapped, policy)
    for left, right in zip(base, other):
        assert right["D_gen"]["value"] == pytest.approx(left["D_gen"]["value"])

    flipped = tuple(replace(case, target_a=case.target_b, target_b=case.target_a)
                    for case in cases)
    _, reversed_rows = _run(ScriptedLM(codec, table), flipped, policy)
    for left, right in zip(base, reversed_rows):
        assert right["D_gen"]["value"] == pytest.approx(-left["D_gen"]["value"])


# ---------------------------------------------------------------------------
# the denominator


def test_an_unreached_target_contributes_zero_and_stays_in_the_denominator():
    """§4.9's survivorship rule, which is the easiest number here to fake.

    A completion shorter than the target has both distances defined as 1, so it
    contributes exactly zero to `f` -- no evidence either way -- and the case
    stays in `n_cases` with `reach_rate` published beside it.
    """
    cases, policy = _cases(max_cases=3)
    # Halt immediately: nothing is emitted before the target width.
    _, rows = _run(ScriptedLM(ByteCodec(), {}, default=HALT), cases, policy)
    for row in rows:
        assert row["reach_rate"] == 0.0
        assert row["D_gen"]["value"] == 0.0
        assert row["hit_own_rate"] == 0.0
    summary = summarise_completions(rows, reps=64)
    assert summary["n_cases"] == len(cases)
    assert summary["reach_rate"] == 0.0
    assert summary["D_gen_mean"] == 0.0


def test_a_partial_reach_is_not_silently_dropped():
    cases, policy = _cases(max_cases=3)
    codec = ByteCodec()
    table = _own_target_table(cases)
    # One case short of its target, the rest complete.
    table[cases[0].prefix_a] = cases[0].target_a[:2] + HALT
    table[cases[0].prefix_b] = cases[0].target_b[:2] + HALT
    _, rows = _run(ScriptedLM(codec, table), cases, policy)
    by_id = {row["case_id"]: row for row in rows}
    short = by_id[cases[0].case_id]
    assert short["reach_rate"] == 0.0
    assert short["D_gen"]["value"] == 0.0
    assert len(rows) == len(cases)


def test_the_draw_is_not_a_resampling_unit():
    """Draws share a prompt and a checkpoint; they average, they do not count.

    The same rule that makes the source/donor component -- and never the token
    -- C3's resampling unit (§4.7 rule 6).
    """
    cases, policy = _cases(max_cases=3)
    model = ScriptedLM(ByteCodec(), _own_target_table(cases))
    raw, rows = _run(model, cases, policy, draws=4)
    assert len(rows) == len(cases)
    # `draws` counts the primary arm only: it is the denominator the reported
    # rates are over, and the block arm is a control, not more evidence.
    assert all(row["draws"] == 8 for row in rows)  # 4 draws x 2 worlds
    # Four (arm, world) combinations are *recorded*; only three are decoded,
    # because the block arm's world A aliases the primary arm's.
    assert len(raw) == len(cases) * 4 * 4


# ---------------------------------------------------------------------------
# behavioural columns


def test_termination_separates_a_fault_from_a_clean_halt():
    """`HaltMonitor.done` fires on HALT *and* on an unknown opcode.

    `VM.run` breaks at both, so reporting them as one category would score a
    dead byte as a clean termination -- and C4's `canonical_rate` is the column
    a reader uses to believe the halt rate.
    """
    cases, policy = _cases(max_cases=2)
    codec = ByteCodec()
    unknown = bytes([max(int(op) for op in Op) + 40])
    halting = ScriptedLM(codec, {}, default=cases[0].target_a + HALT)
    faulting = ScriptedLM(codec, {}, default=unknown)
    _, clean = _run(halting, cases, policy)
    _, broken = _run(faulting, cases, policy)
    assert all(row["halt_rate"] == 1.0 for row in clean)
    assert all(row["fault_rate"] == 0.0 for row in clean)
    assert all(row["fault_rate"] == 1.0 for row in broken)
    assert all(row["halt_rate"] == 0.0 for row in broken)
    assert all(row["canonical_rate"] == 0.0 for row in broken)


def test_a_wrong_offset_copy_is_recovered_but_not_consistent():
    """`relation_recovered` without `relation_consistent` is its own failure.

    A block that is the source translated by the *wrong* step is a different
    thing from an unrelated block, and collapsing them would hide the most
    informative outcome C4 can produce.
    """
    cases, policy = _cases(max_cases=3)
    codec = ByteCodec()
    table: dict[bytes, bytes] = {}
    wanted = []
    for case in cases:
        source = case.prefix_a[case.relevant.start:case.relevant.stop]
        step = (case.transform[0] // 2, case.transform[1] // 2)
        # The relation is a translation by `step`; apply twice as much.
        wrong = apply(source, Affine(dx=2 * step[0], dy=2 * step[1]))
        if wrong is None or wrong == case.target_a:
            continue
        table[case.prefix_a] = wrong + HALT
        wanted.append(case.case_id)
    assert wanted, "no case admitted an on-canvas wrong-offset copy"
    raw, _ = _run(ScriptedLM(codec, table), cases, policy)
    rows = [row for row in raw
            if row["case_id"] in wanted and row["world"] == "a"
            and row["arm"] == "primary"]
    assert rows
    for row in rows:
        assert row["relation_recovered"] is True
        assert row["relation_consistent"] is False
        assert row["hit_own"] is False


def test_the_config_refuses_settings_that_are_not_a_measurement():
    with pytest.raises(ValueError):
        CompletionConfig(draws=0)
    with pytest.raises(ValueError):
        CompletionConfig(cap=0)
    with pytest.raises(ValueError):
        CompletionConfig(top_k=0)
    with pytest.raises(ValueError):
        CompletionConfig(temperature=0.0)
    assert CompletionConfig(top_k=None).as_dict()["structural_mask"] == "off"


def test_per_venue_draws_are_declared_and_defaulted():
    """The promotion rule expands the co-primary venues, not the diagnostics.

    64 draws on a venue whose interval already excludes zero buys nothing; the
    column it does buy is `hit_own_rate`, and that is a co-primary question.
    """
    config = CompletionConfig(draws=8, draws_by_venue=(("synthetic_flat_step", 64),))
    assert config.draws_for("synthetic_flat_step") == 64
    assert config.draws_for("synthetic_flat_step_yaxis") == 8
    assert config.as_dict()["draws_per_world"] == {
        "default": 8, "synthetic_flat_step": 64,
    }


# ---------------------------------------------------------------------------
# the paired decode itself


def _tiny_model(codec: ByteCodec, max_len: int) -> DrawingLM:
    torch.manual_seed(0)
    cfg = Config(vocab_size=codec.vocab_size, d_model=32, n_layers=2, n_heads=2,
                 max_len=max_len)
    return DrawingLM(cfg).eval()


def test_every_arm_of_a_case_consumes_the_identical_variate_block():
    """What makes the pair a pair (§4.9).

    Two decodes seeded identically still diverge, because the prompts differ
    and so does the distribution from the first step. Only a shared variate
    block keeps world A and world B comparable draw for draw.
    """
    cases, policy = _cases(max_cases=2)
    codec = ByteCodec()
    cap = 24
    model = _tiny_model(codec, max(case.prefix_bytes for case in cases) + cap + 2)
    seen: list[torch.Tensor] = []
    original = model.generate

    def spy(*args, **kwargs):
        seen.append(kwargs["variates"])
        return original(*args, **kwargs)

    model.generate = spy  # type: ignore[method-assign]
    complete_cases(model, list(cases), codec, policy,
                   config=CompletionConfig(draws=2, cap=cap, top_k=8))
    assert len(seen) >= 2
    by_shape: dict[tuple, list[torch.Tensor]] = {}
    for block in seen:
        by_shape.setdefault(tuple(block.shape), []).append(block)
    for blocks in by_shape.values():
        for block in blocks[1:]:
            assert torch.equal(block, blocks[0])
    for block in seen:
        assert bool(((block >= 0) & (block < 1)).all())


def test_the_decode_is_reproducible_and_an_identical_prompt_decodes_identically():
    """`block_prefix_a == prefix_a` in both venues, so the block A arm is free.

    Decoding it anyway would burn half the run to reproduce a tensor the
    primary arm already holds -- and if the two ever disagreed, the pairing
    would be broken rather than merely wasteful. This pins both facts.
    """
    cases, policy = _cases(max_cases=2)
    codec = ByteCodec()
    cap = 24
    model = _tiny_model(codec, max(case.prefix_bytes for case in cases) + cap + 2)
    config = CompletionConfig(draws=2, cap=cap, top_k=8)
    first = complete_cases(model, list(cases), codec, policy, config=config)
    second = complete_cases(model, list(cases), codec, policy, config=config)
    assert first == second
    assert all(case.block_prefix_a == case.prefix_a for case in cases)


def test_a_desynchronised_prompt_is_refused_rather_than_measured():
    """If a decoded row does not begin with its own prompt, nothing downstream
    means anything -- the target block would be read at the wrong offset."""
    cases, policy = _cases(max_cases=2)
    codec = ByteCodec()

    class Truncating(ScriptedLM):
        def generate(self, n, max_new, **kwargs):
            out = super().generate(n, max_new, **kwargs)
            out[:, 3] = PAD_TOKEN  # drop a prompt byte from the returned rows
            return out

    with pytest.raises(ValueError, match="does not begin with"):
        _run(Truncating(codec, _own_target_table(cases)), cases, policy)


# ---------------------------------------------------------------------------
# summaries and the promotion metric


def test_the_summary_publishes_reach_beside_every_accuracy_number():
    cases, policy = _cases(max_cases=4)
    model = ScriptedLM(ByteCodec(), _own_target_table(cases))
    _, rows = _run(model, cases, policy, draws=2)
    summary = summarise_completions(rows, reps=128)
    for column in ("reach_rate", "hit_own_rate", "hit_other_rate",
                   "relation_consistent_rate", "relation_recovered_rate",
                   "skeleton_match_rate", "canonical_rate", "halt_rate",
                   "fault_rate", "cap_rate", "longest_compatible_prefix_frac",
                   "D_gen_mean", "D_gen_target_control_mean",
                   "D_gen_block_control_mean", "D_gen_minus_block_control_mean",
                   "half_width", "bootstrap_D_gen",
                   "bootstrap_D_gen_minus_block_control"):
        assert column in summary, column
    assert summary["half_width"] >= 0.0
    by_venue = summarise_by_venue(rows, reps=128)
    assert set(by_venue) == {"synthetic_flat_step"}
    assert by_venue["synthetic_flat_step"]["n_cases"] == len(cases)


def test_an_empty_venue_summarises_without_inventing_a_number():
    summary = summarise_completions([], reps=16)
    assert summary["n_cases"] == 0
    assert np.isnan(summary["D_gen_mean"])
    assert np.isnan(summary["reach_rate"])


# ---------------------------------------------------------------------------
# the freeze


def test_the_c4_protocol_freezes_the_sampler_and_the_estimand():
    """A protocol that leaves a decode setting out has not frozen the sampler.

    `scripts/context_c4.py` refuses one; this checks the committed freezes
    actually carry what it demands, so the refusal is never reached in anger.
    """
    for name in ("context-protocol-v4.json", "context-protocol-v5.json"):
        path = Path("docs") / name
        if not path.exists():
            continue
        block = json.loads(path.read_text())["context_c3"]["c4"]
        sampler, diagnostic = block["sampler"], block["diagnostic"]
        for key in ("top_k", "temperature", "variate_seed"):
            assert key in sampler, (name, key)
        for key in ("draws_per_world", "cap_symbols"):
            assert key in diagnostic, (name, key)
        assert diagnostic["structural_mask"] == "off", name
        assert block["estimand"]["resampling_unit"] == (
            "connected_source_donor_component, as in C3"), name
        assert "promotion_rule" in block, name
