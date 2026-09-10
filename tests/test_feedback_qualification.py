"""F5b: the qualification Module.

Qualification decides whether *a mechanism runs*, not whether it works. It reads
completeness, provenance, resources, convergence, the generic guards and the
repaired stability report, and it has no access to Direction 2's cases, to
`Delta`, to `G` or to any relation outcome -- because a schedule chosen by
looking at those would make the pilot its own selection data
(`docs/directions.md` §7 invariant 13).

The tests below are mostly about *refusal*. A gate that passes a complete cell
is easy; the ones that matter are the cells it must reject -- a scientific
artifact offered as development data, a hash that was recorded rather than
recomputed, a schedule that qualified on one seed, and a ranking by magnitude.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest

from dm.eval import feedback_contract as contract
from dm.eval.feedback import DEVELOPMENT, ENGINEERING, SCIENTIFIC
from dm.eval.feedback_evidence import execution_environment
from dm.eval.feedback_qualification import (
    CELL_FIELDS,
    QUALIFICATION_CLAUSES,
    REQUIRED_HASHES,
    qualify_cell,
    qualify_correction,
    qualify_schedules,
    validation_tail,
)

STEPS = contract.FINAL_STEPS
F5C = contract.F5C_PACKAGE


def _history(final_bits: float = 100.0, slope: float = 0.0) -> list[dict]:
    """Twelve evals over the frozen budget, linear in `slope` bits per 1k steps."""
    evals = [(index + 1) * (STEPS // 12) for index in range(12)]
    return [
        {"step": step, "bits_per_drawing": final_bits + slope * (step - STEPS) / 1000.0,
         "train_loss": 1.0, "gen_validity": 1.0, "elapsed_s": float(step)}
        for step in evals
    ]


def _stability_report(**overrides) -> dict:
    rows = []
    for depth in contract.STABILITY_PASSES:
        rows.append({
            "passes": depth,
            "finite": True,
            "fused_input_rms_p99": None if depth == 0 else 1.0,
            "hidden_rms_p99": 1.0,
            "max_abs_logit": 10.0,
            "update_q95": 0.9,
            "update_q95_wavefront": 0.5 / max(1, depth),
            "wavefront_positions": 100,
            "bits_per_drawing": 100.0 + 0.02 * depth,
            "converged_positions": 10 * depth,
            "converged_semantic_bytes": 10.0 * depth,
            "converged_bits_per_byte": 2.0,
            "standard_converged_bits_per_byte": 2.0,
        })
    report = {
        "schema": contract.STABILITY_REPORT_SCHEMA,
        "programs": contract.STABILITY_SUBSET_SIZE,
        "scored_positions": 4000,
        "padded_positions": 100,
        "scored_positions_only": True,
        "symbols_per_byte": 1,
        "cost_unit": "bits_per_semantic_byte",
        "subset": {
            "seed": contract.seed_for("stability"),
            "deepest_pass": max(contract.STABILITY_PASSES),
            "size": contract.STABILITY_SUBSET_SIZE,
            "eligible": 700,
            "corpus_programs": contract.DEVELOPMENT_N_VAL,
            "indices": list(range(contract.STABILITY_SUBSET_SIZE)),
            "lengths": [200] * contract.STABILITY_SUBSET_SIZE,
            "digest": "d" * 64,
        },
        "standard_input_rms_p99": 1.0,
        "baseline_hidden_rms_p99": 1.0,
        "baseline_bits_per_drawing": 100.0,
        "valid_halt_loss": 0.0,
        "passes": rows,
    }
    report.update(overrides)
    return report


def _calibration_event(package: contract.Package, schedule: str) -> dict | None:
    """What the training loop records when the package declares a calibration."""
    if package.gain_calibration == contract.DEFAULT_GAIN_CALIBRATION:
        return None
    return {
        "rule": package.gain_calibration,
        "applied": True,
        "step": contract.feedback_phase_transition(STEPS, schedule),
        "formula": contract.gain_calibration(package.gain_calibration),
        "value": 0.2371,
        "gain_before": {"mean": 0.0968, "std": 0.03, "min": -0.01, "max": 0.2},
        "gain_after": 0.2371,
        "feedback_schema": contract.QUALIFICATION_FEEDBACK_SCHEMA,
        "counted_input_positions": 5_314_908,
        "vocabulary": 258,
    }


def _environment(**overrides) -> dict:
    body = execution_environment(device="mps")
    body["determinism"] = {"requested": True, "deterministic_algorithms": True,
                           "cudnn_deterministic": True, "cudnn_benchmark": False}
    body["deterministic_algorithms"] = True
    body.update(overrides)
    return body


def _corpus_identity(package: contract.Package, **overrides) -> dict:
    fingerprint = {"train": "a" * 16, "val": "b" * 16, "n_train": package.n_train,
                   "n_val": package.n_val, "bytes_train": 1, "bytes_val": 1}
    body = {
        "manifest": "runs/corpus.json",
        "structure": "relational",
        "canonical_payload_sha256": "c" * 64,
        "recorded_canonical_sha256": "c" * 64,
        "file_sha256": "f" * 64,
        "manifest_fingerprint": dict(fingerprint),
        "record_fingerprint": dict(fingerprint),
        "manifest_data_seed": package.data_seed,
        "record_data_seed": package.data_seed,
        "manifest_derangement_seed": contract.seed_for("derangement"),
        "accepted": True,
        "census_accepted": True,
        "matches": True,
        "differences": [],
    }
    body.update(overrides)
    return body


def _cell(*, schedule: str = "terminal_mix_v1", seed: int = 100,
          package: contract.Package = contract.F5B_PACKAGE, replicate: int = 1,
          **overrides) -> dict:
    history = _history()
    corpus = _corpus_identity(package)
    cell = {
        "package": package.name,
        "schedule": schedule,
        "seed": seed,
        "replicate": replicate,
        "provenance": DEVELOPMENT,
        "environment": _environment(),
        "corpus_identity": corpus,
        "source": {"files": {"dm/train.py": "0" * 64}, "combined": "s" * 64},
        "record": {
            "complete": True,
            "steps_done": STEPS,
            "corpus": dict(corpus["record_fingerprint"]),
            "config": {
                "seed": seed,
                "data_seed": package.data_seed,
                "n_train": package.n_train,
                "n_val": package.n_val,
                "steps": STEPS,
                "codec": "byte",
                "feedback_schema": contract.QUALIFICATION_FEEDBACK_SCHEMA,
                "pass_schedule": schedule,
                "gain_calibration": package.gain_calibration,
            },
            "model": {"params": 857_600},
            "val_lengths": {"truncated": 0.0},
            "train_lengths": {"truncated": 0.0},
            "history": history,
            "final": history[-1],
            "best": history[-1],
            "feedback": {
                "schema": contract.RECORD_SCHEMA,
                "protocol": contract.PROTOCOL,
                "feedback_schema": contract.QUALIFICATION_FEEDBACK_SCHEMA,
                "pass_schedule": schedule,
                "gain_calibration": package.gain_calibration,
                "gain_calibration_event": _calibration_event(package, schedule),
                "device": "mps",
                "stride": 1,
                "programs_seen": 1,
                "semantic_bytes_seen": 1,
                "content_symbols_seen": 1,
                "padded_positions": 1,
                "content_symbol_forward_passes": 1,
                "padded_forward_positions": 1,
                "pass_histogram": {"1": STEPS - 100, "2": 90, "3": 10},
                "expected_passes_per_batch":
                    contract.expected_passes_per_batch(schedule),
                "observed_passes_per_batch": (
                    (STEPS - 100) + 2 * 90 + 3 * 10) / STEPS,
                "peak_memory_bytes": 1,
                "wall_clock_s": 1.0,
            },
        },
        "reload": {"strict": True, "params_match_config": True,
                   "feedback_schema": contract.QUALIFICATION_FEEDBACK_SCHEMA,
                   "provenance": DEVELOPMENT,
                   "checkpoint_sha256": f"{seed:0>64}",
                   "expected_feedback_schema":
                       contract.QUALIFICATION_FEEDBACK_SCHEMA},
        "generic_guards": {
            "passed": True,
            "values": {"validation_cost_delta_bits_per_drawing": 0.2,
                       "valid_halt_delta": 0.0,
                       "truncation_rate_increase": 0.0},
        },
        "stability": {"report": _stability_report()},
        "hashes": {
            name: {"recorded": f"{name}-hash", "recomputed": f"{name}-hash"}
            for name in REQUIRED_HASHES
        },
    }
    cell.update(overrides)
    return cell


def _correction_cell(*, seed: int = 200, replicate: int = 1, **overrides) -> dict:
    """One F5c cell: the correction package's seeds, corpus and one lever."""
    return _cell(schedule=contract.CORRECTION_SCHEDULE, seed=seed,
                 package=F5C, replicate=replicate, **overrides)


def _correction_cells() -> list[dict]:
    """The whole package: seed 200 twice, seed 201 once, and one checkpoint
    digest shared by the repeated seed's two cells."""
    cells = [_correction_cell(seed=seed, replicate=replicate)
             for seed, count in sorted(contract.CORRECTION_REPLICATES.items())
             for replicate in range(1, count + 1)]
    repeated = [cell for cell in cells
                if cell["seed"] == contract.CORRECTION_REPEATED_SEED]
    for cell in repeated:
        cell["reload"]["checkpoint_sha256"] = "c" * 64
    return cells


# ---------------------------------------------------------------------------
# the tail, inherited rather than reinvented


def test_the_validation_tail_is_the_projects_existing_reporting_guard():
    """Same window, same snapping rule, same units as `scripts/sweep.py`. The
    threshold is inherited, so the arithmetic behind it has to be too -- and a
    re-derivation that merely looked like it would make the guard a different
    guard under the same number."""
    from scripts.sweep import tail as sweep_tail

    for slope in (0.0, -0.4, 1.2, -3.0):
        history = _history(slope=slope)
        assert validation_tail(history) == pytest.approx(sweep_tail(history))
    assert math.isnan(validation_tail([]))


# ---------------------------------------------------------------------------
# one cell


def test_a_complete_development_cell_passes_every_clause():
    verdict = qualify_cell(_cell())
    assert verdict["passed"], verdict["failures"]
    assert set(verdict["clauses"]) == set(QUALIFICATION_CLAUSES)
    assert verdict["schedule"] == "terminal_mix_v1"
    assert verdict["seed"] == 100


@pytest.mark.parametrize("provenance", [SCIENTIFIC, ENGINEERING, None, "other"])
def test_only_a_development_cell_can_qualify_a_schedule(provenance):
    """Both directions fail closed. An engineering smoke is one short cell and
    cannot qualify anything; a *scientific* cell is estimation data, and reading
    it here would let the pilot select its own configuration."""
    cell = _cell()
    cell["provenance"] = provenance
    if provenance is None:
        del cell["provenance"]
    verdict = qualify_cell(cell)
    assert not verdict["passed"]
    assert "provenance" in verdict["failures"]


def test_a_recorded_hash_never_stands_in_for_a_recomputed_one():
    """Invariant 11. A boolean the artifact wrote about itself is a claim, not a
    check; the freeze recomputes and compares."""
    cell = _cell()
    cell["hashes"]["checkpoint"]["recomputed"] = "something-else"
    verdict = qualify_cell(cell)
    assert not verdict["passed"]
    assert "hashes" in verdict["failures"]

    missing = _cell()
    del missing["hashes"]["source"]
    assert "hashes" in qualify_cell(missing)["failures"]


def test_a_cell_trained_under_a_different_schedule_than_it_claims_is_refused():
    """The record's own accounting names the schedule the plan was drawn from.
    A cell whose config and accounting disagree trained one thing and is filed
    as another."""
    cell = _cell(schedule="terminal_mix_v1")
    cell["record"]["feedback"]["pass_schedule"] = "project_progressive_v1"
    assert "schedule" in qualify_cell(cell)["failures"]


def test_pilot_seeds_and_pilot_data_cannot_enter_the_qualification():
    """Development uses separate seeds and separate data, so nothing the
    qualification reads can become estimation data."""
    for field, value in (("seed", 0), ("data_seed", 0)):
        cell = _cell()
        cell["record"]["config"][field] = value
        if field == "seed":
            cell["seed"] = value
        assert "development_seeds" in qualify_cell(cell)["failures"], field


def test_resource_and_convergence_clauses_read_the_frozen_thresholds():
    over_budget = _cell()
    over_budget["record"]["model"]["params"] = contract.PARAMETER_BUDGET
    assert "parameters" in qualify_cell(over_budget)["failures"]

    truncated = _cell()
    truncated["record"]["val_lengths"]["truncated"] = 0.01
    assert "truncation" in qualify_cell(truncated)["failures"]

    steep = _cell()
    steep["record"]["history"] = _history(slope=-1.0)
    steep["record"]["final"] = steep["record"]["history"][-1]
    steep["record"]["best"] = steep["record"]["history"][-1]
    assert "validation_tail" in qualify_cell(steep)["failures"]

    drifted = _cell()
    drifted["record"]["best"] = dict(drifted["record"]["final"])
    drifted["record"]["best"]["bits_per_drawing"] -= 2.0
    assert "final_minus_best" in qualify_cell(drifted)["failures"]


def test_a_failed_generic_guard_stops_the_cell():
    cell = _cell()
    cell["generic_guards"]["values"]["validation_cost_delta_bits_per_drawing"] = 3.0
    cell["generic_guards"]["passed"] = False
    assert "generic_guards" in qualify_cell(cell)["failures"]


def test_the_stability_clauses_are_the_repaired_ones():
    """Scored positions, semantic bytes, the frozen long subset and the wavefront
    -- and the withdrawn clause cannot come back through the qualification."""
    unmasked = _cell()
    unmasked["stability"]["report"]["scored_positions_only"] = False
    assert "stability_measurement" in qualify_cell(unmasked)["failures"]

    wrong_unit = _cell()
    wrong_unit["stability"]["report"]["symbols_per_byte"] = 1
    wrong_unit["record"]["config"]["codec"] = "bit"
    wrong_unit["record"]["feedback"]["stride"] = 8
    assert "stability_measurement" in qualify_cell(wrong_unit)["failures"]

    short_subset = _cell()
    short_subset["stability"]["report"]["subset"]["size"] = 4
    assert "stability_subset" in qualify_cell(short_subset)["failures"]

    borrowed_seed = _cell()
    borrowed_seed["stability"]["report"]["subset"]["seed"] += 1
    assert "stability_subset" in qualify_cell(borrowed_seed)["failures"]

    moving = _cell()
    for row in moving["stability"]["report"]["passes"]:
        row["update_q95_wavefront"] = 0.9
    failures = qualify_cell(moving)["failures"]
    assert "stability" in failures
    assert "update_q95_wavefront" in failures["stability"]
    assert "update_q95_not_settling" not in failures["stability"]


def test_a_blended_update_quantile_cannot_fail_a_cell_on_its_own():
    """The clause the repair withdrew. A long subset drives `update_q95` up
    through sequence length alone, and a gate that read it would refuse cells for
    being measured on long programs."""
    cell = _cell()
    for row in cell["stability"]["report"]["passes"]:
        row["update_q95"] = 12.0
    assert qualify_cell(cell)["passed"]


# ---------------------------------------------------------------------------
# schedules and the choice between them


def test_a_schedule_qualifies_only_when_both_seeds_pass():
    good = [_cell(schedule="terminal_mix_v1", seed=seed)
            for seed in contract.DEVELOPMENT_SEEDS]
    mixed = copy.deepcopy(good)
    mixed[1]["record"]["complete"] = False
    result = qualify_schedules({"terminal_mix_v1": mixed})
    assert result["schedules"]["terminal_mix_v1"]["passed"] is False
    assert result["frozen_schedule"] is None
    assert result["label"] == "no_viable_feedback_implementation_at_scale"

    result = qualify_schedules({"terminal_mix_v1": good})
    assert result["schedules"]["terminal_mix_v1"]["passed"] is True
    assert result["frozen_schedule"] == "terminal_mix_v1"
    assert result["label"] == "qualified"


def test_a_missing_seed_is_incomplete_and_never_a_pass():
    one_seed = [_cell(schedule="terminal_mix_v1", seed=100)]
    result = qualify_schedules({"terminal_mix_v1": one_seed})
    assert result["frozen_schedule"] is None
    assert "seeds" in result["schedules"]["terminal_mix_v1"]["failures"]


def test_the_first_predeclared_eligible_schedule_wins_not_the_best_one():
    """Never a magnitude ranking. If both eligible schedules pass, the frozen one
    is the first in the predeclared order even when the second looks better on
    every number the qualification is allowed to see."""
    first = [_cell(schedule="terminal_mix_v1", seed=seed)
             for seed in contract.DEVELOPMENT_SEEDS]
    second = [_cell(schedule="project_progressive_v1", seed=seed)
              for seed in contract.DEVELOPMENT_SEEDS]
    for cell in second:  # strictly better on every visible quantity
        cell["generic_guards"]["values"]["validation_cost_delta_bits_per_drawing"] = 0.0
        for row in cell["stability"]["report"]["passes"]:
            row["update_q95_wavefront"] = 0.0
    result = qualify_schedules({"project_progressive_v1": second,
                                "terminal_mix_v1": first})
    assert result["frozen_schedule"] == "terminal_mix_v1"
    assert result["order"] == list(contract.ELIGIBLE_SCHEDULES)


def test_the_diagnostic_control_can_never_freeze_a_protocol():
    control = [_cell(schedule="two_pass_control_v1", seed=seed)
               for seed in contract.DEVELOPMENT_SEEDS]
    result = qualify_schedules({"two_pass_control_v1": control})
    assert result["schedules"]["two_pass_control_v1"]["eligible"] is False
    assert result["frozen_schedule"] is None
    assert result["label"] == "no_viable_feedback_implementation_at_scale"


def test_an_unstable_schedule_stops_as_unstable_and_never_as_no_gain():
    """`no_feedback_gain` requires complete, stable pilot scores. Qualification
    has none: it never sees `G`, so it cannot produce that label at all."""
    cells = [_cell(schedule="terminal_mix_v1", seed=seed)
             for seed in contract.DEVELOPMENT_SEEDS]
    for cell in cells:
        for row in cell["stability"]["report"]["passes"]:
            row["finite"] = False
    result = qualify_schedules({"terminal_mix_v1": cells})
    assert result["label"] == "unstable_feedback"
    assert result["frozen_schedule"] is None


def test_the_qualification_cannot_read_a_relation_outcome():
    """Structural, not conventional: there is no parameter through which a case,
    a `Delta` or a `G` could arrive, and a cell carrying one is refused rather
    than ignored."""
    import inspect

    for function in (qualify_cell, qualify_schedules):
        text = inspect.getsource(function)
        for forbidden in ("Delta", "delta_soft", '"G"', "hit_own", "context"):
            assert forbidden not in text, (function.__name__, forbidden)

    cell = _cell()
    cell["G"] = 0.5
    with pytest.raises(ValueError, match="incomplete"):
        qualify_cell(cell)


# ---------------------------------------------------------------------------
# the Adapter


def test_a_relation_outcome_cannot_hide_inside_a_nested_block():
    """The F5b gate refused an unexpected *top-level* key and accepted arbitrary
    nesting, so "structurally blind to relation outcomes" described the artifacts
    that happened to exist rather than a guarantee. Every declared level is now
    closed, and a `Delta` two levels down costs an error."""
    for path, key in (
        (("record",), "Delta"),
        (("record", "config"), "G"),
        (("record", "feedback"), "I_structure"),
        (("stability", "report"), "Delta_soft"),
        (("stability", "report", "subset"), "hit_own"),
        (("generic_guards",), "Delta"),
        (("hashes", "record"), "Delta"),
        (("environment",), "G"),
        (("corpus_identity",), "Delta"),
    ):
        cell = _cell()
        node = cell
        for step in path:
            node = node[step]
        node[key] = 0.5
        with pytest.raises(ValueError, match="incomplete"):
            qualify_cell(cell)


def test_the_corpus_clause_compares_the_run_against_the_manifest():
    """The clause F5b did not have, and the one whose absence let six cells
    train on `data_seed=100` while filing a `data_seed=0` manifest. A manifest's
    digests identify the manifest *file*; only the fingerprint identifies the
    programs a checkpoint saw."""
    mismatched = _cell()
    mismatched["corpus_identity"]["matches"] = False
    mismatched["corpus_identity"]["differences"] = ["train: e092df != c4d24e"]
    failures = qualify_cell(mismatched)["failures"]
    assert "corpus_identity" in failures
    assert "fingerprint" in failures["corpus_identity"]

    # A manifest that passes its own digest check but describes another corpus is
    # exactly the F5b artifact, and it must not pass on the digest alone.
    self_consistent = _cell()
    self_consistent["corpus_identity"]["matches"] = False
    self_consistent["corpus_identity"]["differences"] = ["data_seed: 0 != 100"]
    assert "corpus_identity" in qualify_cell(self_consistent)["failures"]

    for field in ("accepted", "census_accepted"):
        rejected = _cell()
        rejected["corpus_identity"][field] = False
        assert "corpus_identity" in qualify_cell(rejected)["failures"], field

    edited = _cell()
    edited["corpus_identity"]["recorded_canonical_sha256"] = "d" * 64
    assert "corpus_identity" in qualify_cell(edited)["failures"]


def test_the_environment_clause_fails_closed_on_a_missing_field():
    for field in contract.REQUIRED_ENVIRONMENT_FIELDS:
        cell = _cell()
        del cell["environment"][field]
        failures = qualify_cell(cell)["failures"]
        assert "environment" in failures, field
        assert field in failures["environment"]["missing_or_unmet"], field


def test_only_a_package_that_requires_determinism_demands_it():
    """F5b did not ask torch for deterministic kernels and could not have known
    to; F5c's reproducibility clause is a statement about exactly that state, so
    the requirement follows the package rather than the calendar."""
    lax = _cell()
    lax["environment"]["deterministic_algorithms"] = False
    lax["environment"]["determinism"]["requested"] = False
    assert qualify_cell(lax)["passed"]

    correction = _correction_cell(seed=200, replicate=1)
    correction["environment"]["deterministic_algorithms"] = False
    correction["environment"]["determinism"]["requested"] = False
    failures = qualify_cell(correction, F5C)["failures"]
    assert "environment" in failures


def test_a_cell_is_judged_by_its_own_package_and_no_other():
    """An F5b cell and an F5c cell are both full-budget development cells under
    one arm. Judging either by the other's rules would check the wrong corpus and
    the wrong training-protocol condition while passing every clause that reads
    the same in both."""
    development = _cell()
    assert "package" in qualify_cell(development, F5C)["failures"]

    correction = _correction_cell(seed=200, replicate=1)
    assert "package" in qualify_cell(correction)["failures"]


def test_the_gain_calibration_is_checked_in_both_halves_and_in_its_event():
    """The package's one lever. A calibration that silently failed to fire looks
    exactly like one that fired and changed nothing, and only the second is a
    result about the mechanism."""
    assert qualify_cell(_correction_cell(), F5C)["passed"]

    missing = _correction_cell()
    missing["record"]["feedback"]["gain_calibration_event"] = None
    assert "gain_calibration" in qualify_cell(missing, F5C)["failures"]

    not_applied = _correction_cell()
    not_applied["record"]["feedback"]["gain_calibration_event"]["applied"] = False
    assert "gain_calibration" in qualify_cell(not_applied, F5C)["failures"]

    wrong_step = _correction_cell()
    wrong_step["record"]["feedback"]["gain_calibration_event"]["step"] = 0
    assert "gain_calibration" in qualify_cell(wrong_step, F5C)["failures"]

    rewritten = _correction_cell()
    rewritten["record"]["feedback"]["gain_calibration_event"]["formula"] = "g = 0.02"
    assert "gain_calibration" in qualify_cell(rewritten, F5C)["failures"]

    uncalibrated = _correction_cell()
    uncalibrated["record"]["config"]["gain_calibration"] = "none"
    assert "gain_calibration" in qualify_cell(uncalibrated, F5C)["failures"]

    # And the other direction: an F5b cell that quietly calibrated is not the
    # cell F5b qualified, whatever its numbers say.
    slipped = _cell()
    slipped["record"]["config"]["gain_calibration"] = F5C.gain_calibration
    assert "gain_calibration" in qualify_cell(slipped)["failures"]


def test_the_correction_package_is_two_seeds_one_repeat_and_a_byte_comparison():
    result = qualify_correction(_correction_cells(), F5C)
    assert result["label"] == "qualified"
    assert result["frozen_schedule"] == contract.CORRECTION_SCHEDULE
    assert result["reproducibility"]["reproduced"] is True
    assert result["package"]["package"] == contract.CORRECTION_PACKAGE


def test_a_repeated_seed_that_does_not_reproduce_is_not_a_mechanism_result():
    """`docs/feedback-stability.md` §3d: one cell's stability verdict flipped
    between two runs of the identical configuration. If that happens again the
    package has measured the machine, and the label has to say so rather than
    filing it against the mechanism."""
    cells = _correction_cells()
    cells[1]["reload"]["checkpoint_sha256"] = "9" * 64
    result = qualify_correction(cells, F5C)
    assert result["label"] == "incomplete_nondeterministic"
    assert result["frozen_schedule"] is None
    assert result["reproducibility"]["reproduced"] is False
    assert "nondeterminism" in result["label_meaning"]


def test_the_correction_package_refuses_a_missing_or_extra_cell():
    for cells in (_correction_cells()[:2],
                  _correction_cells() + [_correction_cell(seed=201, replicate=1)]):
        result = qualify_correction(cells, F5C)
        assert result["frozen_schedule"] is None
        assert result["label"] in ("incomplete", "incomplete_nondeterministic")


def test_a_correction_cell_that_fails_a_stability_clause_stops_as_unstable():
    cells = _correction_cells()
    for row in cells[2]["stability"]["report"]["passes"]:
        row["update_q95_wavefront"] = 0.9
    result = qualify_correction(cells, F5C)
    assert result["label"] == "unstable_feedback"
    assert result["frozen_schedule"] is None


def test_the_correction_gate_moved_no_threshold():
    """One lever. Every number the correction package gates on is the number v0
    or the project's existing reporting guard already carried."""
    assert contract.CORRECTION_GATE["thresholds_moved"] == []
    assert contract.CORRECTION_GATE["jitter_half_width"] == contract.JITTER_HALF_WIDTH
    assert contract.CORRECTION_GATE["feedback_schema"] == (
        contract.QUALIFICATION_FEEDBACK_SCHEMA)
    assert contract.CORRECTION_GATE["retry_allowed"] is False
    assert contract.CORRECTION_GATE["max_cells"] == 3
    assert contract.FUSED_NORM_GAIN == 0.02
    assert contract.JITTER_HALF_WIDTH == 0.02
    # Fresh on both axes: nothing F5b trained can qualify what it motivated.
    assert not set(contract.CORRECTION_SEEDS) & set(contract.DEVELOPMENT_SEEDS)
    assert not set(contract.CORRECTION_SEEDS) & set(contract.MODEL_SEEDS)
    assert contract.CORRECTION_DATA_SEED not in (
        contract.DEVELOPMENT_DATA_SEED, contract.TRAINING_DEFAULTS["data_seed"])


# ---------------------------------------------------------------------------
# the Adapter


def _write_cell_report(directory, cell: dict):
    """One `feedback_qualify.py run` report, with real files behind its hashes."""
    import hashlib
    import json

    from dm.eval.provenance import canonical_digest

    stem = f"{cell['schedule']}_s{cell['seed']}r{cell['replicate']}"
    for name in ("record", "checkpoint"):
        path = directory / f"{stem}_{name}.bin"
        # Separate files, and the checkpoint's *content* depends on the seed
        # alone: two replicates of one seed are two runs that reproduced, which
        # is the state the correction package's reproducibility clause is about.
        # The record differs per replicate, as a real one does.
        content = (f"{name}-{cell['seed']}" if name == "checkpoint"
                   else f"{name}-{cell['seed']}-{cell['replicate']}")
        path.write_bytes(content.encode())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        cell["hashes"][name] = {"recorded": digest, "recomputed": digest,
                                "path": str(path)}
        if name == "checkpoint":
            cell["reload"]["checkpoint_sha256"] = digest

    out = directory / f"{stem}_cell.json"
    body = {
        "schema": contract.QUALIFICATION_SCHEMA,
        "protocol": contract.PROTOCOL,
        "provenance": DEVELOPMENT,
        "decision_value": "schedule_qualification_only",
        "package": contract.package(
            "f5c" if cell["package"] == contract.CORRECTION_PACKAGE else "f5b"
        ).as_dict(),
        "cell": cell,
    }
    body["report_sha256"] = canonical_digest(body, "report_sha256")
    out.write_text(json.dumps(body, indent=1, sort_keys=True) + "\n")
    return out, {name: Path(cell["hashes"][name]["path"])
                 for name in ("record", "checkpoint")}


def _decide_args(cells, out=None, package="f5b"):
    import argparse

    return argparse.Namespace(cells=cells, out=out, package=package)


def test_the_decide_adapter_rehashes_the_artifacts_it_reads(tmp_path, capsys):
    """The Adapter recomputes every hash whose file is still on disk, in this
    process. A digest a program computed and immediately compared against itself
    proves nothing, so the comparison only becomes a check when a *later* process
    does it."""
    import json

    from dm.eval.feedback_evidence import source_identity
    from scripts.feedback_qualify import _decide

    current = source_identity()["combined"]
    paths = []
    artifacts = {}
    for seed in contract.DEVELOPMENT_SEEDS:
        cell = _cell(schedule="terminal_mix_v1", seed=seed)
        cell["hashes"]["source"] = {"recorded": current, "recomputed": current}
        path, files = _write_cell_report(tmp_path, cell)
        paths.append(path)
        artifacts[seed] = files

    out = tmp_path / "qualification.json"
    assert _decide(_decide_args(paths, out)) == 0
    result = json.loads(out.read_text())
    assert result["frozen_schedule"] == "terminal_mix_v1"
    assert result["label"] == "qualified"

    # Change a checkpoint after the fact and the freeze-side recomputation
    # catches it, without any report having to notice.
    artifacts[contract.DEVELOPMENT_SEEDS[0]]["checkpoint"].write_bytes(b"tampered")
    assert _decide(_decide_args(paths)) == 1
    printed = capsys.readouterr()
    assert "hashes" in printed.out


def test_the_decide_adapter_refuses_a_report_that_does_not_match_its_payload(
        tmp_path):
    """A cell report cannot carry a hash of itself, so the digest is taken over
    everything else and checked in a later process. Edit the payload and it moves;
    that is the whole point of keeping it out of `REQUIRED_HASHES`."""
    import json

    from scripts.feedback_qualify import _decide

    path, _ = _write_cell_report(tmp_path, _cell(seed=100))
    body = json.loads(path.read_text())
    body["cell"]["record"]["steps_done"] = 1
    path.write_text(json.dumps(body))
    assert _decide(_decide_args([path])) == 1


def test_the_decide_adapter_catches_cells_trained_under_different_code(tmp_path):
    """A package takes long enough for the tree to change underneath it. The
    source digest is recomputed here, from this tree, and a cell that recorded a
    different one is not comparable with the others."""
    from scripts.feedback_qualify import _decide

    cell = _cell(seed=100)
    cell["hashes"]["source"] = {"recorded": "stale-source",
                                "recomputed": "stale-source"}
    path, _ = _write_cell_report(tmp_path, cell)
    assert _decide(_decide_args([path])) == 1


def test_the_decide_adapter_refuses_an_artifact_that_is_not_development(tmp_path):
    import json

    from scripts.feedback_qualify import _decide

    path, _ = _write_cell_report(tmp_path, _cell(seed=100))
    body = json.loads(path.read_text())
    body["provenance"] = SCIENTIFIC
    path.write_text(json.dumps(body))
    assert _decide(_decide_args([path])) == 1


def test_the_decide_adapter_refuses_a_cell_from_another_package(tmp_path):
    """Two packages now write the same kind of artifact into the same directory,
    and a glob is how one would end up deciding the other."""
    from scripts.feedback_qualify import _decide

    path, _ = _write_cell_report(tmp_path, _cell(seed=100))
    assert _decide(_decide_args([path], package="f5c")) == 1


def test_the_decide_adapter_runs_the_correction_gate_for_the_correction_package(
        tmp_path):
    import json

    from dm.eval.feedback_evidence import source_identity
    from scripts.feedback_qualify import _decide

    current = source_identity()["combined"]
    paths = []
    for cell in _correction_cells():
        cell["hashes"]["source"] = {"recorded": current, "recomputed": current}
        path, _ = _write_cell_report(tmp_path, cell)
        paths.append(path)
    out = tmp_path / "correction.json"
    assert _decide(_decide_args(paths, out, package="f5c")) == 0
    result = json.loads(out.read_text())
    assert result["label"] == "qualified"
    # Two files, one digest: the repeated seed's replicates wrote separate
    # artifacts whose bytes agree, which is what reproducing means and what the
    # neighbouring test breaks to prove the comparison can fail.
    assert result["reproducibility"]["seed"] == contract.CORRECTION_REPEATED_SEED
    assert len(set(result["reproducibility"]["checkpoint_sha256"])) == 1


def test_the_run_adapter_walks_its_whole_body_without_a_full_budget_cell(
        tmp_path, monkeypatch):
    """A development cell costs 24,000 steps, so the path that assembles it is
    the one code path this stage cannot afford to exercise by running it.

    Training and data loading are replaced by fakes; everything downstream is
    real -- a real checkpoint reloaded strict, the real generic guards, the real
    frozen subset, the real stability probe and the real corpus reconciliation --
    so a name that does not resolve or a report field that does not exist fails
    here rather than three hours into the first cell.

    **The faked half is a real hole and is covered elsewhere.** Replacing
    `build_data` hid a `TrainConfig.extra` key the corpus builder rejects, which
    surfaced on the first live invocation;
    `tests/test_feedback_contract.py::test_every_frozen_training_configuration_actually_builds_its_corpus`
    builds every frozen config's corpus for real. Do not read this test as
    covering the corpus path.
    """
    import argparse
    import json

    import torch

    import dm.train
    from dm.data import feedback as corpus_module
    from dm.data import synthetic
    from dm.eval.feedback import DEVELOPMENT as PROVENANCE
    from dm.isa.codec import CODECS
    from dm.models.transformer import Config, DrawingLM
    from scripts import feedback_qualify

    codec = CODECS["byte"]
    deepest = max(contract.STABILITY_PASSES)
    _, val = synthetic.split(200, 400, seed=1, tier=1, flatten=True)
    val = [program for program in val
           if len(codec.with_bos(program)) - 1 > deepest]
    assert len(val) >= contract.STABILITY_SUBSET_SIZE, len(val)

    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(dm.train, "RUNS", runs)
    monkeypatch.setattr(feedback_qualify.dm.train, "RUNS", runs)

    # A real manifest, and a record whose fingerprint really is that manifest's
    # relational arm -- which is the reconciliation F5b never performed.
    paired = corpus_module.build(200, 40, data_seed=0,
                                 seed=contract.seed_for("derangement"))
    manifest_path = tmp_path / "corpus.json"
    corpus_module.write_manifest(manifest_path, corpus_module.manifest(paired))
    fingerprint = corpus_module.balance_report(paired)["corpus"]["relational"]

    cell = _cell(schedule="terminal_mix_v1", seed=100)
    cell["record"]["corpus"] = fingerprint
    cell["record"]["config"]["data_seed"] = 0
    cfg = Config(vocab_size=codec.vocab_size, d_model=16, n_layers=2, n_heads=2,
                 max_len=512,
                 feedback_schema=contract.QUALIFICATION_FEEDBACK_SCHEMA)

    def _fake_build_data(config):
        return val[:16], val

    def _fake_train(config, verbose=False):
        tag = config.tag
        torch.manual_seed(0)
        model = DrawingLM(cfg)
        checkpoint = runs / f"{tag}.pt"
        torch.save({"cfg": {**cfg.__dict__}, "state": model.state_dict(),
                    "provenance": PROVENANCE}, checkpoint)
        record = json.loads(json.dumps(cell["record"]))
        record["checkpoint_sha256"] = feedback_qualify.sha256_file(checkpoint)
        (runs / f"{tag}.json").write_text(json.dumps(record))
        return record

    monkeypatch.setattr(feedback_qualify, "train", _fake_train)
    monkeypatch.setattr(feedback_qualify.dm.train, "build_data", _fake_build_data)

    out = tmp_path / "cell.json"
    args = argparse.Namespace(package="f5b", schedule="terminal_mix_v1", seed=100,
                              replicate=1, codec="byte", device="cpu", tag=None,
                              corpus=manifest_path, deterministic=False, out=out)
    feedback_qualify._run(args)

    body = json.loads(out.read_text())
    assert body["provenance"] == PROVENANCE
    assert body["decision_value"] == "schedule_qualification_only"
    assert body["schema"] == contract.QUALIFICATION_SCHEMA
    assert set(body["cell"]) == set(CELL_FIELDS)
    report = body["cell"]["stability"]["report"]
    assert report["schema"] == contract.STABILITY_REPORT_SCHEMA
    assert report["subset"]["size"] == contract.STABILITY_SUBSET_SIZE
    assert report["subset"]["seed"] == contract.seed_for("stability")
    assert min(report["subset"]["lengths"]) > deepest
    assert report["symbols_per_byte"] == codec.stride
    # The freeze-side pair: the record claimed a checkpoint hash and the file
    # answered, in a different process from the one that wrote it.
    checkpoint = body["cell"]["hashes"]["checkpoint"]
    assert checkpoint["recorded"] == checkpoint["recomputed"]
    assert Path(checkpoint["path"]).name.endswith(".pt")
    # The corpus reconciliation, on a manifest and a record that really agree.
    identity = body["cell"]["corpus_identity"]
    assert identity["matches"] is True
    assert identity["record_fingerprint"] == fingerprint
    digests = corpus_module.manifest_digests(manifest_path)
    assert identity["canonical_payload_sha256"] == digests["canonical_payload_sha256"]
    assert identity["file_sha256"] == digests["file_sha256"]
    assert identity["file_sha256"] != identity["canonical_payload_sha256"]
    # Every field the environment clause requires, recorded rather than declared.
    for field in contract.REQUIRED_ENVIRONMENT_FIELDS:
        assert field in body["cell"]["environment"], field
    # No Direction 2 case, contrast or outcome anywhere in the artifact.
    text = json.dumps(body)
    for forbidden in ("Delta", "hit_own", "block_control", "target_control"):
        assert forbidden not in text, forbidden
