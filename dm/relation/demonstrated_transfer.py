"""Frozen, per-seed paired reductions for the demonstrated-affine pilot."""
from __future__ import annotations

import math

import numpy as np

from dm.relation.demonstrated_readout import fit_result


def component_groups(rows, connected_rows=None):
    """Connect every shared source and donor; never resample rows independently."""
    parent = {}

    def root(node):
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for row in rows if connected_rows is None else connected_rows:
        ids = sorted(set(row["source_ids"]) | set(row["donor_source_ids"]))
        if not ids:
            raise ValueError("pilot row has no source component")
        for node in ids[1:]:
            left, right = root(ids[0]), root(node)
            parent[max(left, right)] = min(left, right)
        root(ids[0])
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault(root(row["source_ids"][0]), []).append(index)
    return list(groups.values())


def paired_panel(relation, baseline, settings, connected_rows=None):
    groups = component_groups(relation, connected_rows)
    if len(groups) < 32:
        raise ValueError("positive panel has fewer than 32 source/donor components")
    predicted = np.array([r["decodes"]["predicted_copy"]["target_exact"] for r in relation], dtype=float)
    standard = np.array([r["decodes"]["standard"]["target_exact"] for r in relation], dtype=float)
    oracle = np.array([r["decodes"]["oracle_copy"]["target_exact"] for r in relation], dtype=float)
    none = np.array([r["decodes"]["standard"]["target_exact"] for r in baseline], dtype=float)
    difference = predicted - standard
    gap = float((oracle - standard).mean())
    # Draw component indices once; keep arms, modes and all member rows paired.
    rng = np.random.default_rng(settings["bootstrap_seed"])
    sums = np.array([difference[group].sum() for group in groups])
    sizes = np.array([len(group) for group in groups])
    draws = rng.integers(len(groups), size=(settings["bootstrap_draws"], len(groups)))
    samples = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    gap_sums = np.array([(oracle[group] - standard[group]).sum() for group in groups])
    sampled_gaps = gap_sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    defined = sampled_gaps > 0
    recovery_samples = samples[defined] / sampled_gaps[defined]
    prompts = [next(q for q in r["queries"] if q["prompt"]) for r in relation]
    return {"denominator": len(relation), "components": len(groups),
            "predicted_exact": int(predicted.sum()), "standard_exact": int(standard.sum()),
            "none_exact": int(none.sum()), "oracle_exact": int(oracle.sum()),
            "D_copy": float(difference.mean()),
            "D_copy_vs_none": float((predicted - none).mean()),
            "oracle_gap": gap, "gap_recovery": float(difference.mean()) / gap if gap > 0 else None,
            "gap_recovery_status": "defined" if gap > 0 else "not_applicable_no_positive_oracle_gap",
            "D_copy_interval": np.quantile(samples, [0.025, 0.975]).tolist(),
            "gap_recovery_interval": np.quantile(recovery_samples, [0.025, 0.975]).tolist()
            if len(recovery_samples) and bool(defined.all()) else None,
            "bootstrap_nonpositive_gap_draws": int((~defined).sum()),
            "conditional_correct": sum(q["highest_valid_full_target_correct"] is True for q in prompts),
            "conditional_correct_negative_margin": sum(
                q["highest_valid_full_target_correct"] is True and q["margin"] is not None
                and q["margin"] < 0 for q in prompts),
            "exact_with_copy": sum(r["decodes"]["predicted_copy"]["exact_with_copy"] for r in relation)}


def validate_rows(rows, card):
    expected = card["task_rows"]["pilot"]
    training_ids = set(card["recipe"]["pilot_intermediate_case_ids"])
    lookup = {}
    for row in rows:
        key = (row["cell_seed"], row["arm"], row["split"], row["case_id"])
        if row["step"] != 1024 or key in lookup:
            raise ValueError("pilot duplicate or non-final row")
        lookup[key] = row
    wanted = set()
    for cell in card["recipe"]["cells"]:
        if cell["stage"] != "pilot":
            continue
        for split, cases in expected.items():
            for case in cases:
                if split == "train" and case["case_id"] not in training_ids:
                    continue
                key = (cell["config"]["seed"], cell["config"]["arm"], split, case["case_id"])
                wanted.add(key)
                row = lookup.get(key)
                if row is None:
                    raise ValueError("pilot missing paired final row")
                for field in ("class", "source_ids", "donor_source_ids"):
                    if list(row[field]) != list(case[field]):
                        raise ValueError(f"pilot row identity mismatch: {field}")
        cell_rows = [r for k, r in lookup.items() if k[:2] == (cell["config"]["seed"], cell["config"]["arm"])]
        if len({r["checkpoint_digest"] for r in cell_rows}) != 1:
            raise ValueError("pilot rows join different checkpoints")
    if set(lookup) != wanted:
        raise ValueError("unexpected pilot rows")
    return lookup


def decide(seeds, settings):
    """Precedence frozen before observation; unresolved patterns remain mixed."""
    if any(not s["safety_pass"] for s in seeds):
        return "unsafe_copying"
    if any(not s["standard_noninferiority_pass"] or not s["training_fit"]["pass"] for s in seeds):
        return "mixed_or_inconclusive"
    if all(s["useful_signal"] for s in seeds):
        return "replicated_useful_pilot_signal"
    panels = [s["panels"]["affine"] for s in seeds]
    if all(p["conditional_correct_negative_margin"] == 64 for p in panels):
        return "selection_bottleneck"
    if all(p["conditional_correct"] < settings["conditional_correct_minimum"] for p in panels):
        return "fit_without_transfer"
    if all(p["exact_with_copy"] >= settings["conditional_correct_minimum"] and p["D_copy"] <= 0 for p in panels):
        return "transfer_without_generation_benefit"
    return "mixed_or_inconclusive"


def summarize(rows, fit_results, card):
    if len(fit_results) != 2 or not all(r["pass"] for r in fit_results):
        raise ValueError("pilot reduction requires both fit seeds")
    lookup = validate_rows(rows, card)
    settings = card["recipe"]["pilot"]
    seeds = []
    for seed in (775159550, 728883118):
        paired = {}
        for split in ("train", "iid", "affine", "family", "negative"):
            selected = sorted((k[3], r) for k, r in lookup.items()
                              if k[:3] == (seed, "span_affine_v1", split))
            relation = [r for _, r in selected]
            baseline = [lookup[(seed, "none", split, case_id)] for case_id, _ in selected]
            paired[split] = (relation, baseline)
        connected_rows = [r for pair in paired.values() for r in pair[0]]
        panels = {split: paired_panel(*paired[split], settings, connected_rows)
                  for split in ("iid", "affine", "family")}
        relation = [r for split in ("iid", "affine", "family", "negative") for r in paired[split][0]]
        baseline = [r for split in ("iid", "affine", "family", "negative") for r in paired[split][1]]
        negatives = paired["negative"][0]
        negative_counts = {kind: {"denominator": sum(r["class"] == kind for r in negatives),
                                  "copy_cases": sum(r["class"] == kind and r["decodes"]["predicted_copy"]["copy_invocations"] > 0 for r in negatives)}
                           for kind in sorted({r["class"] for r in negatives})}
        false_copy = sum(item["copy_cases"] for item in negative_counts.values())
        halt_loss = sum(int(r["decodes"]["standard"]["valid_halt"]) - int(r["decodes"]["predicted_copy"]["valid_halt"]) for r in relation) / len(relation)
        standard_halt_loss = sum(int(b["decodes"]["standard"]["valid_halt"]) - int(r["decodes"]["standard"]["valid_halt"]) for r, b in zip(relation, baseline, strict=True)) / len(relation)
        extra_bits = sum(r["all_byte_nll"]["sum"] - b["all_byte_nll"]["sum"] for r, b in zip(relation, baseline, strict=True)) / (len(relation) * math.log(2))
        affine = panels["affine"]
        safety = false_copy <= settings["negative_invocations_maximum"] and halt_loss <= settings["valid_halt_loss_maximum"]
        noninferiority = extra_bits <= settings["standard_extra_bits_per_program_maximum"] and standard_halt_loss <= settings["standard_valid_halt_loss_vs_none_maximum"]
        seeds.append({"seed": seed, "panels": panels, "training_fit": fit_result(paired["train"][0]),
                      "negative_controls": negative_counts, "negative_copy_cases": false_copy,
                      "evaluation_denominator": len(relation), "predicted_valid_halt_loss": halt_loss,
                      "standard_valid_halt_loss_vs_none": standard_halt_loss,
                      "standard_extra_bits_per_program": extra_bits,
                      "safety_pass": safety, "standard_noninferiority_pass": noninferiority,
                      "useful_signal": safety and noninferiority and affine["D_copy"] >= settings["affine_D_copy_minimum"]
                      and affine["gap_recovery"] is not None and affine["gap_recovery"] >= settings["gap_recovery_minimum"]
                      and affine["D_copy_interval"][0] > settings["descriptive_interval_lower_bound_exclusive"]})
    return {"diagnosis": decide(seeds, settings), "fit": fit_results, "seeds": seeds,
            "scope": "descriptive two-seed demonstrated-affine pilot; no scientific qualification"}
