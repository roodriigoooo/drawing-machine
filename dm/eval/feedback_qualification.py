"""F5b: does the mechanism run at full budget and full scale?

**What this Module decides, and what it must never touch.** Qualification asks
whether one named pass schedule produces a checkpoint that is complete, within
its resources, converged, generically harmless and recurrently stable -- twice,
on two development seeds. It answers pass or stop. It never sees Direction 2's
cases, `Delta`, `G`, a relation outcome or a promotion draw, because a schedule
chosen by looking at those would make the estimation pilot its own
configuration-selection data (`docs/directions.md` §7 invariant 13). The
prohibition is structural: `qualify_cell` refuses a cell that carries an
unexpected key rather than ignoring it, so smuggling an outcome in costs an
error instead of a silent read.

**Why the choice between schedules is an order and not a comparison.** Three
schedules are trained; two of them are eligible; the eligible one that freezes
protocol v1 is the *first in the predeclared order* that passes both seeds. Not
the best one. With six cells and a handful of continuous readings, "best" is a
selection over the same trajectories the pilot will later be scored on, and the
project has already paid once for a threshold chosen after the outcome
(`PLAN.md`, the axis claim). The order is frozen in
`dm.eval.feedback_contract.ELIGIBLE_SCHEDULES` before any cell runs.

**Why every criterion here is v0's or the project's.** F5b moves no threshold.
What it changes is which positions and which units four of v0's clauses are
measured on, and it withdraws one clause that has no reading at all under a
triangular recurrence (`docs/feedback-stability.md` §1). Those repairs live in
the qualification layer of `feedback_contract`, with their own schema counter, so
an auditor reading two artifacts can see exactly which statement is v0's and
which is F5b's.

**Fail closed, everywhere.** A missing field, an absent hash, a recorded boolean
where a recomputation was required, one seed instead of two: all `incomplete`,
none of them a pass. The single most expensive failure available to this stage is
a schedule that freezes because a check could not find what it was looking for.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from .feedback import DEVELOPMENT, stability_verdict
from .feedback_contract import (
    CORRECTION_GATE,
    CORRECTION_LABELS,
    CORRECTION_REPEATED_SEED,
    CORRECTION_REPLICATES,
    DEFAULT_GAIN_CALIBRATION,
    DIAGNOSTIC_SCHEDULES,
    ELIGIBLE_SCHEDULES,
    F5B_PACKAGE,
    F5C_PACKAGE,
    FINAL_STEPS,
    GENERIC_GUARDS,
    PARAMETER_BUDGET,
    QUALIFICATION_FEEDBACK_SCHEMA,
    QUALIFICATION_GATE,
    QUALIFICATION_SCHEMA,
    REQUIRED_ENVIRONMENT_FIELDS,
    STABILITY_PASSES,
    STABILITY_REPORT_SCHEMA,
    STABILITY_SUBSET_SIZE,
    Package,
    feedback_phase_transition,
    missing_record_fields,
    pass_schedule,
    seed_for,
)
from .feedback_contract import (
    gain_calibration as calibration_formula,
)
from .feedback_evidence import shape_faults

#: The keys a qualification cell may carry.  A closed set rather than a required
#: subset: an unexpected key is how a relation outcome would arrive, and this
#: stage is defined by what it is not allowed to read.
CELL_FIELDS: tuple[str, ...] = (
    "package", "schedule", "seed", "replicate", "provenance", "record", "reload",
    "generic_guards", "stability", "environment", "corpus_identity", "source",
    "hashes",
)

#: The fields a cell must carry.  `provenance` is checked as a *value* rather
#: than for presence, because its absence is itself the failure the clause
#: reports; everything else missing is `incomplete` before any clause runs.
REQUIRED_CELL_FIELDS: tuple[str, ...] = tuple(
    name for name in CELL_FIELDS if name != "provenance"
)

#: The hashes a freeze recomputes.  Each is `{"recorded": ..., "recomputed": ...}`
#: and the two must agree: a boolean an artifact wrote about itself is a claim,
#: and `docs/directions.md` §7 invariant 11 says a claim never replaces the
#: recomputation (`PLAN.md` invariant 10).
#:
#: **The cell report's own hash is deliberately not here.** It is the envelope the
#: verdict travels in rather than an input the verdict is about, and a hash of an
#: envelope cannot live inside that envelope -- inserting it changes the bytes it
#: was taken over. `scripts/feedback_qualify.py decide` verifies it against the
#: report's canonical payload digest before the gate runs, and refuses on
#: mismatch; the freeze then requires that verification for every cell of the
#: schedule it is about to freeze.
REQUIRED_HASHES: tuple[str, ...] = (
    "record", "checkpoint", "protocol", "corpus", "source",
)

#: Every clause a cell is judged on, in report order.  Named as a tuple so a
#: verdict that silently stopped applying one is a test failure rather than a
#: shorter dictionary nobody counts.
QUALIFICATION_CLAUSES: tuple[str, ...] = (
    "complete",
    "provenance",
    "package",
    "record_fields",
    "architecture",
    "schedule",
    "gain_calibration",
    "development_seeds",
    "budget",
    "parameters",
    "truncation",
    "accounting",
    "corpus_identity",
    "environment",
    "hashes",
    "reload",
    "validation_tail",
    "final_minus_best",
    "generic_guards",
    "stability_measurement",
    "stability_subset",
    "stability",
)

#: Fraction of the schedule the validation tail is measured over.  The project's
#: existing reporting guard, inherited whole: a single eval interval is dominated
#: by eval-to-eval noise in a val mean, and the cadence differs by regime, so an
#: interval version is not even measuring the same part of two schedules.
TAIL_WINDOW = 1 / 3


def validation_tail(history: Sequence[Mapping], window: float = TAIL_WINDOW) -> float:
    """Bits/drawing shed per 1,000 steps over the final `window` of the schedule.

    The same quantity `scripts/sweep.py` reports, including the snapping rule: a
    coarse history widens the window rather than interpolating, which averages in
    the steeper earlier slope and so *overstates* the tail -- the safe direction
    for a guard. The threshold is inherited from that guard, so the arithmetic
    behind it is inherited too, and
    `tests/test_feedback_qualification.py` pins the two against each other rather
    than asserting they agree.

    A tail is necessary evidence of an asymptote, not sufficient: every run here
    anneals its LR to zero, so a small tail is partly a fact about the schedule.
    Nothing may multiply it by a step count and read the product as a forecast.
    """
    if len(history) < 2:
        return float("nan")
    final = history[-1]
    cutoff = final["step"] * (1.0 - window)
    earlier = [entry for entry in history[:-1] if entry["step"] <= cutoff]
    start = (earlier or [history[0]])[-1]
    steps = final["step"] - start["step"]
    if steps <= 0:
        return float("nan")
    return (final["bits_per_drawing"] - start["bits_per_drawing"]) / steps * 1000.0


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(float(value))


def _clause(clauses: dict, name: str, ok: bool, detail) -> None:
    clauses[name] = {"passed": bool(ok), "detail": detail}


def qualify_cell(cell: Mapping, package: Package = F5B_PACKAGE) -> dict:
    """One development cell against every clause. Pass/stop, never a ranking.

    `cell` is assembled by `dm.eval.feedback_evidence`, which owns the file IO
    and every identity comparison; this function owns the arithmetic and the
    refusals, which is the split `docs/directions.md` §7 invariant 15 asks for
    one level deeper than F5b applied it.

    `package` says which seeds, corpus and training-protocol condition are the
    right ones. It is a parameter rather than a constant because two packages now
    exist and the *clauses* are identical between them: an F5c correction cell is
    judged by exactly the instrument that judged F5b, plus the corpus-identity
    and environment clauses F5b was missing.

    **The refusal is over the whole shape, not only the top level.** F5b rejected
    an unexpected top-level key and accepted arbitrary nesting underneath, so its
    blindness to relation outcomes was a property of the artifacts that happened
    to exist rather than a guarantee. `shape_faults` walks every declared level.
    """
    faults = shape_faults(cell, CELL_FIELDS)
    if faults:
        raise ValueError(
            f"qualification cell carries {faults}, which this stage is not "
            f"allowed to read; the declared shape is "
            f"dm.eval.feedback_evidence.CELL_SHAPE: incomplete"
        )
    missing = [name for name in REQUIRED_CELL_FIELDS if name not in cell]
    if missing:
        raise ValueError(
            f"qualification cell is missing {missing}: incomplete"
        )

    record = cell["record"] or {}
    config = record.get("config") or {}
    accounting = record.get("feedback") or {}
    stability_report = (cell["stability"] or {}).get("report") or {}
    subset = stability_report.get("subset") or {}
    guards = cell["generic_guards"] or {}
    guard_values = guards.get("values") or {}
    reload = cell["reload"] or {}
    hashes = cell["hashes"] or {}
    schedule = cell.get("schedule")
    environment = cell["environment"] or {}
    corpus = cell["corpus_identity"] or {}

    clauses: dict[str, dict] = {}

    _clause(clauses, "complete",
            record.get("complete") is True
            and record.get("steps_done") == config.get("steps"),
            {"complete": record.get("complete"),
             "steps_done": record.get("steps_done")})

    _clause(clauses, "provenance", cell.get("provenance") == DEVELOPMENT,
            {"provenance": cell.get("provenance"), "required": DEVELOPMENT})

    # Which package this cell belongs to, and which replicate of it. Checked
    # rather than inferred from the seeds: F5b and F5c both train development
    # cells at full budget under the same arm, and a cell judged by the wrong
    # package's rules would be judged against the wrong corpus and the wrong
    # training-protocol condition while passing every clause that looks the same
    # in both.
    replicate = cell.get("replicate")
    _clause(clauses, "package",
            cell.get("package") == package.name
            and isinstance(replicate, int) and not isinstance(replicate, bool)
            and 1 <= replicate <= package.replicates,
            {"package": cell.get("package"), "required": package.name,
             "replicate": replicate, "replicates_allowed": package.replicates})

    # Over the whole record: `complete`, `steps_done` and `corpus` belong to
    # any run and the pass counters to a feedback run, and
    # `missing_record_fields` reads both halves.
    _clause(clauses, "record_fields", not missing_record_fields(record),
            {"missing": missing_record_fields(record)})

    architecture = (config.get("feedback_schema") == QUALIFICATION_FEEDBACK_SCHEMA
                    and accounting.get("feedback_schema")
                    == QUALIFICATION_FEEDBACK_SCHEMA)
    _clause(clauses, "architecture", architecture,
            {"config": config.get("feedback_schema"),
             "accounting": accounting.get("feedback_schema"),
             "required": QUALIFICATION_FEEDBACK_SCHEMA})

    known = schedule in set(package.schedules)
    _clause(clauses, "schedule",
            known
            and config.get("pass_schedule") == schedule
            # The accounting names the schedule the plan was actually drawn
            # from. A cell whose config and accounting disagree trained one
            # thing and is filed as another, and the realized pass histogram
            # cannot tell them apart from an unlucky draw.
            and accounting.get("pass_schedule") == schedule,
            {"cell": schedule, "config": config.get("pass_schedule"),
             "accounting": accounting.get("pass_schedule"),
             "package_schedules": list(package.schedules)})

    # The one lever, checked in both halves of the record and in the event it
    # produced. A package whose calibration silently failed to fire would
    # otherwise be indistinguishable from one whose calibration fired and
    # changed nothing -- and only the second is a result about the mechanism.
    event = accounting.get("gain_calibration_event")
    calibrated = (config.get("gain_calibration") == package.gain_calibration
                  and accounting.get("gain_calibration") == package.gain_calibration)
    detail = {"config": config.get("gain_calibration"),
              "accounting": accounting.get("gain_calibration"),
              "required": package.gain_calibration}
    if package.gain_calibration == DEFAULT_GAIN_CALIBRATION:
        # `none` means the gain kept its initialisation, so the honest artifact
        # is either no event at all or an event that says it did not apply.
        calibrated = calibrated and (event is None or event.get("applied") is False)
        detail["event"] = event
    else:
        transition = (feedback_phase_transition(config["steps"], schedule)
                      if isinstance(config.get("steps"), int)
                      and schedule in set(package.schedules) else None)
        calibrated = (
            calibrated
            and isinstance(event, Mapping)
            and event.get("rule") == package.gain_calibration
            and event.get("applied") is True
            and event.get("step") == transition
            and event.get("formula") == calibration_formula(package.gain_calibration)
            and _finite(event.get("value")) and float(event["value"]) > 0.0
            and event.get("feedback_schema") == QUALIFICATION_FEEDBACK_SCHEMA
        )
        detail |= {
            "rule": None if not isinstance(event, Mapping) else event.get("rule"),
            "applied": None if not isinstance(event, Mapping) else event.get("applied"),
            "step": None if not isinstance(event, Mapping) else event.get("step"),
            "transition_step": transition,
            "value": None if not isinstance(event, Mapping) else event.get("value"),
        }
    _clause(clauses, "gain_calibration", calibrated, detail)

    _clause(clauses, "development_seeds",
            cell.get("seed") in package.seeds
            and config.get("seed") == cell.get("seed")
            and config.get("data_seed") == package.data_seed,
            {"seed": config.get("seed"), "data_seed": config.get("data_seed"),
             "allowed": list(package.seeds),
             "development_data_seed": package.data_seed})

    _clause(clauses, "budget",
            config.get("steps") == FINAL_STEPS
            and config.get("n_train") == package.n_train
            and config.get("n_val") == package.n_val,
            {"steps": config.get("steps"), "n_train": config.get("n_train"),
             "n_val": config.get("n_val")})

    params = (record.get("model") or {}).get("params")
    _clause(clauses, "parameters",
            isinstance(params, int) and 0 < params < PARAMETER_BUDGET,
            {"params": params, "budget": PARAMETER_BUDGET})

    truncation = [(record.get("val_lengths") or {}).get("truncated"),
                  (record.get("train_lengths") or {}).get("truncated")]
    _clause(clauses, "truncation",
            all(_finite(value) and float(value) == 0.0 for value in truncation),
            {"val": truncation[0], "train": truncation[1]})

    histogram = accounting.get("pass_histogram") or {}
    batches = sum(histogram.values()) if isinstance(histogram, dict) else 0
    observed = accounting.get("observed_passes_per_batch")
    reconciled = (
        isinstance(histogram, dict) and histogram
        and batches == config.get("steps")
        and _finite(observed)
        and math.isclose(
            float(observed),
            sum(int(k) * n for k, n in histogram.items()) / max(1, batches),
            rel_tol=1e-9, abs_tol=1e-9)
    )
    _clause(clauses, "accounting", reconciled,
            {"batches": batches, "steps": config.get("steps"),
             "observed_passes_per_batch": observed})

    # The clause F5b did not have, and the reason it did not notice that all six
    # of its cells trained on a corpus its manifest does not describe. A
    # manifest's digests identify the manifest *file*; a record's fingerprint
    # identifies the *programs*, and only the second is what a checkpoint saw.
    corpus_faults = {}
    if corpus.get("matches") is not True:
        corpus_faults["fingerprint"] = corpus.get("differences") or "missing"
    if corpus.get("accepted") is not True:
        corpus_faults["balance_accepted"] = corpus.get("accepted")
    if corpus.get("census_accepted") is not True:
        corpus_faults["census_accepted"] = corpus.get("census_accepted")
    if corpus.get("canonical_payload_sha256") != corpus.get(
            "recorded_canonical_sha256"):
        corpus_faults["manifest_digest"] = [
            corpus.get("recorded_canonical_sha256"),
            corpus.get("canonical_payload_sha256"),
        ]
    _clause(clauses, "corpus_identity", not corpus_faults,
            corpus_faults or {key: corpus.get(key) for key in
                              ("manifest", "structure", "record_fingerprint",
                               "file_sha256")})

    # Present-and-null is a value; absent is `incomplete`. When the package
    # requires deterministic execution the granted state is a clause too: the
    # reproducibility verdict is a statement about it, and a run that quietly
    # took nondeterministic kernels cannot support one.
    environment_faults = [name for name in REQUIRED_ENVIRONMENT_FIELDS
                          if name not in environment]
    determinism = environment.get("determinism") or {}
    if package.deterministic:
        if environment.get("deterministic_algorithms") is not True:
            environment_faults.append("deterministic_algorithms")
        if determinism.get("requested") is not True:
            environment_faults.append("determinism.requested")
    _clause(clauses, "environment", not environment_faults,
            {"missing_or_unmet": environment_faults} if environment_faults
            else {key: environment.get(key) for key in
                  ("python", "torch", "platform", "device",
                   "deterministic_algorithms")})

    hash_faults = {}
    for name in REQUIRED_HASHES:
        pair = hashes.get(name)
        if not isinstance(pair, Mapping):
            hash_faults[name] = "missing"
            continue
        recorded, recomputed = pair.get("recorded"), pair.get("recomputed")
        if not isinstance(recorded, str) or not isinstance(recomputed, str):
            hash_faults[name] = "not a pair of digests"
        elif recorded != recomputed:
            hash_faults[name] = [recorded, recomputed]
    _clause(clauses, "hashes", not hash_faults, hash_faults or "recomputed")

    _clause(clauses, "reload",
            reload.get("strict") is True
            and reload.get("params_match_config") is True
            and reload.get("feedback_schema") == QUALIFICATION_FEEDBACK_SCHEMA
            and reload.get("provenance") == DEVELOPMENT
            # The digest the reproducibility verdict is taken over, read off the
            # object that was actually reloaded rather than off the record.
            and isinstance(reload.get("checkpoint_sha256"), str),
            dict(reload))

    tail = validation_tail(record.get("history") or [])
    limit = QUALIFICATION_GATE["max_validation_tail_bits_per_drawing_per_1k"]
    _clause(clauses, "validation_tail", _finite(tail) and abs(tail) <= limit,
            {"tail": tail, "limit": limit})

    final_bits = (record.get("final") or {}).get("bits_per_drawing")
    best_bits = (record.get("best") or {}).get("bits_per_drawing")
    drift = (final_bits - best_bits) if _finite(final_bits) and _finite(best_bits) \
        else float("nan")
    drift_limit = QUALIFICATION_GATE["max_final_minus_best_bits_per_drawing"]
    _clause(clauses, "final_minus_best",
            _finite(drift) and drift <= drift_limit,
            {"drift": drift, "limit": drift_limit})

    guard_faults = {}
    if guards.get("passed") is not True:
        guard_faults["passed"] = guards.get("passed")
    for name, threshold, worse in (
        ("validation_cost_delta_bits_per_drawing",
         GENERIC_GUARDS["max_validation_cost_bits_per_drawing"], "above"),
        ("valid_halt_delta", GENERIC_GUARDS["min_valid_halt_delta"], "below"),
        ("truncation_rate_increase",
         GENERIC_GUARDS["max_truncation_rate_increase"], "above"),
    ):
        value = guard_values.get(name)
        if not _finite(value):
            guard_faults[name] = "missing"
        elif (float(value) > threshold) if worse == "above" else (float(value) < threshold):
            guard_faults[name] = value
    _clause(clauses, "generic_guards", not guard_faults,
            guard_faults or dict(guard_values))

    # The repaired *measurement*, checked before its verdict: a stability report
    # taken over padded positions or in symbol units answers a different question
    # from the one the gate asks, and would answer it in the same field names.
    stride = accounting.get("stride")
    measurement = {
        "schema": stability_report.get("schema"),
        "scored_positions_only": stability_report.get("scored_positions_only"),
        "symbols_per_byte": stability_report.get("symbols_per_byte"),
        "stride": stride,
        "cost_unit": stability_report.get("cost_unit"),
        "deepest_pass": subset.get("deepest_pass"),
    }
    _clause(clauses, "stability_measurement",
            stability_report.get("schema") == STABILITY_REPORT_SCHEMA
            and stability_report.get("scored_positions_only") is True
            and stability_report.get("cost_unit") == "bits_per_semantic_byte"
            and stability_report.get("symbols_per_byte") == stride,
            measurement)

    lengths = subset.get("lengths")
    deepest = max(STABILITY_PASSES)
    _clause(clauses, "stability_subset",
            subset.get("size") == STABILITY_SUBSET_SIZE
            and subset.get("seed") == seed_for("stability")
            and subset.get("deepest_pass") == deepest
            and isinstance(subset.get("indices"), list)
            and len(subset["indices"]) == STABILITY_SUBSET_SIZE
            and isinstance(lengths, list)
            and len(lengths) == STABILITY_SUBSET_SIZE
            and all(isinstance(value, int) and value > deepest for value in lengths)
            and isinstance(subset.get("digest"), str),
            {key: subset.get(key) for key in
             ("size", "seed", "deepest_pass", "eligible", "digest")})

    verdict = (stability_verdict(stability_report)
               if stability_report.get("passes") else {"failures": "missing"})
    failures = verdict.get("failures")
    # `fused_input_rms_band` reads a moving reference whose meaning
    # `docs/feedback-stability.md` §3a withdrew; it stays reported and does not
    # gate until a source- or operator-derived reference exists.
    diagnostic = set(QUALIFICATION_GATE["diagnostic_only_clauses"])
    gated = ({name: value for name, value in failures.items()
              if name not in diagnostic}
             if isinstance(failures, dict) else failures)
    _clause(clauses, "stability", isinstance(gated, dict) and not gated,
            gated if gated else "stable")

    ordered = {name: clauses[name] for name in QUALIFICATION_CLAUSES}
    return {
        "schema": QUALIFICATION_SCHEMA,
        "package": cell.get("package"),
        "schedule": schedule,
        "seed": cell.get("seed"),
        "replicate": replicate,
        # The digest the package-level reproducibility clause compares. Carried
        # on the verdict so that comparison never has to re-open a checkpoint.
        "checkpoint_sha256": reload.get("checkpoint_sha256"),
        "eligible": schedule in ELIGIBLE_SCHEDULES,
        "passed": all(entry["passed"] for entry in ordered.values()),
        "clauses": ordered,
        "failures": {name: entry["detail"] for name, entry in ordered.items()
                     if not entry["passed"]},
        "diagnostic_only": sorted(diagnostic),
    }


def qualify_schedules(cells_by_schedule: Mapping[str, Sequence[Mapping]],
                      package: Package = F5B_PACKAGE) -> dict:
    """Both seeds of each schedule, then the frozen choice among them.

    The choice is `ELIGIBLE_SCHEDULES` order and nothing else. A diagnostic
    schedule is evaluated and reported and can never freeze a protocol -- it
    exists to show that the probe can fail, and a control that authorized the
    thing it controls for would not be one.

    Labels are bounded by what this stage can see. `unstable_feedback` when a
    stability clause fails; otherwise `no_viable_feedback_implementation_at_scale`
    when no eligible schedule passes both seeds. Never `no_feedback_gain`: that
    label requires complete, stable pilot *scores*, and qualification has none.
    """
    schedules: dict[str, dict] = {}
    for name in sorted(cells_by_schedule):
        pass_schedule(name)  # refuse an unknown name rather than reporting it
        verdicts = [qualify_cell(cell, package) for cell in cells_by_schedule[name]]
        seeds = [verdict["seed"] for verdict in verdicts]
        failures: dict[str, object] = {}
        if sorted(seeds) != sorted(package.seeds):
            failures["seeds"] = {"present": seeds,
                                 "required": list(package.seeds)}
        failed = {verdict["seed"]: verdict["failures"]
                  for verdict in verdicts if not verdict["passed"]}
        if failed:
            failures["cells"] = failed
        schedules[name] = {
            "eligible": name in ELIGIBLE_SCHEDULES,
            "passed": not failures and name in ELIGIBLE_SCHEDULES,
            "both_seeds": not failures,
            "cells": verdicts,
            "failures": failures,
        }

    frozen = next((name for name in ELIGIBLE_SCHEDULES
                   if schedules.get(name, {}).get("passed")), None)
    unstable = any(
        "stability" in verdict["failures"]
        for entry in schedules.values() for verdict in entry["cells"]
    )
    if frozen is not None:
        label = "qualified"
    elif unstable:
        label = "unstable_feedback"
    else:
        label = "no_viable_feedback_implementation_at_scale"
    return {
        "schema": QUALIFICATION_SCHEMA,
        "package": package.as_dict(),
        "order": list(ELIGIBLE_SCHEDULES),
        "diagnostic": list(DIAGNOSTIC_SCHEDULES),
        "selection_rule": QUALIFICATION_GATE["selection_rule"],
        "schedules": schedules,
        "frozen_schedule": frozen,
        "label": label,
    }


def qualify_correction(cells: Sequence[Mapping],
                       package: Package = F5C_PACKAGE) -> dict:
    """The F5c correction package: three cells, one schedule, a terminal rule.

    Structurally different from `qualify_schedules` in exactly one way, and it is
    the way the package exists for. F5b asked "does *some* eligible schedule pass
    both seeds", so its unit was a schedule and its choice was an order. F5c asks
    "does the one predeclared configuration pass, and does it reproduce", so its
    unit is a seed, one seed runs twice, and the repeat is a *reproducibility*
    control rather than a second sample.

    **Historical reproducibility rule.** The
    repeated seed's two cells must carry one checkpoint SHA-256. If they do not,
    the label is `incomplete_nondeterministic` and it is neither a pass nor a
    statement about the mechanism: two runs of one configuration that disagree
    have measured the machine, and `docs/feedback-stability.md` §3d already
    showed this happening at 24,000 steps with a stability verdict flipping
    between them. That is the observation this clause exists to settle, and it
    settles it in the only direction that was predeclared -- byte equality, or
    stop. A 2026-08-21 post-closure audit found this comparator invalid:
    ``torch.save`` containers depend on filename and these checkpoints carry a
    replicate-specific ``record_name``. This function remains unchanged so the
    immutable F5c artifact is reproducible; never use this branch for a new
    package. ``feedback_evidence.state_dict_sha256`` is the content comparator.

    **The terminal rule is the package.** Any reproducibility failure and any
    gate failure closes Direction 3 at this scale; all three cells passing
    freezes a new training protocol and only then authorizes F6. There is no
    retry, no second calibration and no third schedule, which is what keeps one
    post-hoc correction from becoming the architecture search the bounded pilot
    exists to prevent.
    """
    verdicts = [qualify_cell(cell, package) for cell in cells]
    failures: dict[str, object] = {}

    if len(verdicts) > CORRECTION_GATE["max_cells"]:
        failures["cell_count"] = {
            "present": len(verdicts), "allowed": CORRECTION_GATE["max_cells"]}

    by_seed: dict[int, list[dict]] = {}
    for verdict in verdicts:
        by_seed.setdefault(verdict["seed"], []).append(verdict)
    present = {seed: sorted(v["replicate"] for v in rows)
               for seed, rows in sorted(by_seed.items())}
    required = {seed: list(range(1, count + 1))
                for seed, count in sorted(CORRECTION_REPLICATES.items())}
    if present != required:
        failures["cells"] = {"present": present, "required": required}

    failed = {f"{verdict['seed']}r{verdict['replicate']}": verdict["failures"]
              for verdict in verdicts if not verdict["passed"]}
    if failed:
        failures["clauses"] = failed

    # The reproducibility verdict, computed whether or not the clauses passed:
    # a package that failed a clause *and* failed to reproduce should say both,
    # because the second changes what the first is evidence about.
    repeats = by_seed.get(CORRECTION_REPEATED_SEED) or []
    digests = [verdict.get("checkpoint_sha256") for verdict in repeats]
    reproduced = (
        len(repeats) == CORRECTION_REPLICATES[CORRECTION_REPEATED_SEED]
        and all(isinstance(digest, str) for digest in digests)
        and len(set(digests)) == 1
    )
    reproducibility = {
        "seed": CORRECTION_REPEATED_SEED,
        "replicates": len(repeats),
        "checkpoint_sha256": digests,
        "reproduced": reproduced,
        "rule": CORRECTION_GATE["reproducibility"],
    }

    unstable = any("stability" in verdict["failures"] for verdict in verdicts)
    if not reproduced:
        label = "incomplete_nondeterministic"
    elif failures:
        label = "unstable_feedback" if unstable else (
            "incomplete" if "cells" in failures or "cell_count" in failures
            else "no_viable_feedback_implementation_at_scale")
    else:
        label = "qualified"

    passed = label == "qualified"
    return {
        "schema": QUALIFICATION_SCHEMA,
        "package": package.as_dict(),
        "order": [package.schedules[0]],
        "diagnostic": [],
        "selection_rule": CORRECTION_GATE["selection_rule"],
        "terminal_rule": CORRECTION_GATE["terminal_rule"],
        "gate": dict(CORRECTION_GATE),
        "reproducibility": reproducibility,
        # The same shape `qualify_schedules` returns, so one freeze Adapter and
        # one auditor read both packages: the schedule that would freeze v1, the
        # cells under it, and the label.
        "schedules": {
            package.schedules[0]: {
                "eligible": True,
                "passed": passed,
                "both_seeds": passed,
                "cells": verdicts,
                "failures": failures,
            }
        },
        "frozen_schedule": package.schedules[0] if passed else None,
        "label": label,
        "label_meaning": CORRECTION_LABELS[label],
    }


__all__ = [
    "CELL_FIELDS",
    "QUALIFICATION_CLAUSES",
    "REQUIRED_CELL_FIELDS",
    "REQUIRED_HASHES",
    "TAIL_WINDOW",
    "qualify_cell",
    "qualify_correction",
    "qualify_schedules",
    "validation_tail",
]
