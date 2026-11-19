"""Stored-row arithmetic, paired components and every pilot decision branch."""
from copy import deepcopy

import pytest

from dm.relation.demonstrated_card import recipe
from dm.relation.demonstrated_pilot import build_pilot_fixture
from dm.relation.demonstrated_transfer import component_groups, decide, summarize


@pytest.fixture(scope="module")
def evidence():
    specification = recipe()
    panels = dict(build_pilot_fixture().panels)
    intermediate = set(specification["pilot_intermediate_case_ids"])
    task = {split: [{"case_id": c.case_id, "class": c.stratum,
                     "source_ids": list(c.source_ids), "donor_source_ids": list(c.donor_source_ids)}
                    for c in cases] for split, cases in panels.items()}
    card = {"recipe": specification, "task_rows": {"pilot": task}}
    rows = []
    for seed in (775159550, 728883118):
        for arm in ("none", "span_affine_v1"):
            for split, cases in task.items():
                for case in cases:
                    if split == "train" and case["case_id"] not in intermediate:
                        continue
                    positive = case["class"] == "positive"
                    decoded = {"target_exact": False, "exact_with_copy": False,
                               "copy_invocations": 0, "valid_halt": True}
                    modes = {"standard": dict(decoded)}
                    if arm != "none":
                        modes["predicted_copy"] = {**decoded, "target_exact": positive,
                                                   "exact_with_copy": positive,
                                                   "copy_invocations": int(positive)}
                    if positive:
                        modes["oracle_copy"] = {**decoded, "target_exact": True}
                    rows.append({**case, "cell_seed": seed, "arm": arm, "split": split,
                                 "step": 1024, "checkpoint_digest": f"{seed}-{arm}",
                                 "all_byte_nll": {"sum": 1.0, "count": 37}, "decodes": modes,
                                 "queries": [{"prompt": True, "gate_nll": 0.1, "margin": 1.0,
                                              "highest_valid_full_target_correct": positive}]})
    return card, rows


def test_full_reduction_connected_donors_and_bits(evidence):
    card, rows = evidence
    result = summarize(rows, [{"pass": True}, {"pass": True}], card)
    assert result["diagnosis"] == "replicated_useful_pilot_signal"
    for seed in result["seeds"]:
        assert seed["evaluation_denominator"] == 320
        assert seed["standard_extra_bits_per_program"] == 0
        assert seed["negative_controls"]["broken"]["denominator"] == 64
        assert seed["negative_controls"]["generic"]["denominator"] == 64
        assert seed["training_fit"]["pass"]
        for panel in seed["panels"].values():
            assert panel["components"] == 32
            assert panel["D_copy"] == panel["gap_recovery"] == 1
            assert panel["D_copy_interval"] == [1, 1]


def test_donor_edge_outside_scored_panel():
    a = {"source_ids": ["a"], "donor_source_ids": []}
    b = {"source_ids": ["b"], "donor_source_ids": []}
    donor = {"source_ids": ["a"], "donor_source_ids": ["b"]}
    assert component_groups([a, b], [a, b, donor]) == [[0, 1]]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "checkpoint", "source", "step", "unpaired"])
def test_refuse_bad_join(evidence, mutation):
    card, raw = evidence
    rows = deepcopy(raw)
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(rows[0])
    elif mutation == "checkpoint":
        rows[0]["checkpoint_digest"] = "wrong"
    elif mutation == "source":
        rows[0]["source_ids"] = ["wrong"]
    elif mutation == "step":
        rows[0]["step"] = 512
    else:
        rows[0]["cell_seed"] = 131
    with pytest.raises(ValueError):
        summarize(rows, [{"pass": True}, {"pass": True}], card)


def test_baseline_ceiling_is_na(evidence):
    card, raw = evidence
    rows = deepcopy(raw)
    for row in rows:
        row["decodes"]["standard"]["target_exact"] = True
    result = summarize(rows, [{"pass": True}, {"pass": True}], card)
    assert result["diagnosis"] == "transfer_without_generation_benefit"
    assert all(s["panels"]["affine"]["gap_recovery"] is None for s in result["seeds"])


def test_bits_per_program_not_nats_or_per_byte(evidence):
    import math
    card, raw = evidence
    rows = deepcopy(raw)
    for row in rows:
        if row["arm"] != "none":
            row["all_byte_nll"]["sum"] += 2 * math.log(2)
    result = summarize(rows, [{"pass": True}, {"pass": True}], card)
    assert result["diagnosis"] == "mixed_or_inconclusive"
    assert result["seeds"][0]["standard_extra_bits_per_program"] == pytest.approx(2)


@pytest.mark.parametrize("events, unsafe", [(1, False), (2, True)])
def test_negative_screen_boundary(evidence, events, unsafe):
    card, raw = evidence
    rows = deepcopy(raw)
    selected = [r for r in rows if r["cell_seed"] == 775159550
                and r["arm"] != "none" and r["split"] == "negative"]
    for row in selected[:events]:
        row["decodes"]["predicted_copy"]["copy_invocations"] = 1
    result = summarize(rows, [{"pass": True}, {"pass": True}], card)
    assert (result["diagnosis"] == "unsafe_copying") is unsafe


@pytest.mark.parametrize("failures, unsafe", [(3, False), (4, True)])
def test_halt_screen_boundary(evidence, failures, unsafe):
    card, raw = evidence
    rows = deepcopy(raw)
    selected = [r for r in rows if r["cell_seed"] == 775159550
                and r["arm"] != "none" and r["split"] == "affine"]
    for row in selected[:failures]:
        row["decodes"]["predicted_copy"]["valid_halt"] = False
    result = summarize(rows, [{"pass": True}, {"pass": True}], card)
    assert (result["diagnosis"] == "unsafe_copying") is unsafe


@pytest.mark.parametrize("diagnosis", ["unsafe_copying", "mixed_or_inconclusive",
                                      "selection_bottleneck", "fit_without_transfer",
                                      "transfer_without_generation_benefit",
                                      "replicated_useful_pilot_signal"])
def test_decision_precedence(evidence, diagnosis):
    card, rows = evidence
    seeds = summarize(rows, [{"pass": True}, {"pass": True}], card)["seeds"]
    for seed in seeds:
        panel = seed["panels"]["affine"]
        if diagnosis == "unsafe_copying":
            seed["safety_pass"] = False
        elif diagnosis == "mixed_or_inconclusive":
            seed["training_fit"]["pass"] = False
        elif diagnosis != "replicated_useful_pilot_signal":
            seed["useful_signal"] = False
            if diagnosis == "selection_bottleneck":
                panel["conditional_correct_negative_margin"] = 64
            elif diagnosis == "fit_without_transfer":
                panel["conditional_correct"] = 59
            else:
                panel["D_copy"] = 0
    assert decide(seeds, card["recipe"]["pilot"]) == diagnosis
