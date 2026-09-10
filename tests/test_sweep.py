"""The paired table is the instrument the codec claims are read off, so what it
is allowed to call resolved is pinned here.

Two independent ways it has been wrong, both of which it now guards:

Pairing on a fixed `data_seed` removes the val-set variance. It does not remove
the *seed* variance, and an earlier version combined only the per-run intervals
in quadrature -- an interval 5-15x too narrow, which marked axes as resolved
whose per-seed deltas did not even agree in sign.

And no interval, however wide, makes a difference between two arms stopped at
different points on their own loss curves into a cost. The schema-4 sweep had
the bit arm shedding 6-15 bits/1k steps against a byte reference flat at 0.03,
with both seeds agreeing in sign on a +42-bit difference: statistically
impeccable, and a measurement of how fast the arms were separating.
"""

import importlib.util
import json
import math
import random
import sys
from pathlib import Path

_SWEEP = Path(__file__).resolve().parent.parent / "scripts" / "sweep.py"
_spec = importlib.util.spec_from_file_location("sweep", _SWEEP)
assert _spec and _spec.loader, f"no sweep driver at {_SWEEP}"
sweep = importlib.util.module_from_spec(_spec)
sys.modules["sweep"] = sweep
_spec.loader.exec_module(sweep)


def _rows(
    deltas: list[float],
    base: list[float] | None = None,
    tail_arm: float = 0.0,
    tail_ref: float = 0.0,
) -> list[dict]:
    """One cell of the typing axis, one seed per entry in `deltas`.

    The per-program difference is constant within a seed, so the within-run
    interval is exactly zero and any interval the table reports has to have come
    from the between-seed term. Both arms default to a flat tail, so a test that
    does not mention convergence is asking only about the interval.
    """
    base = base or [100.0, 110.0, 120.0, 130.0, 140.0]
    rows = []
    for seed, delta in enumerate(deltas):
        common = {
            "regime": "tokmatch", "shape": "square", "seed": f"s{seed}", "steps": 8_844,
        }
        rows.append({**common, "codec": "token", "val_bits": base, "tail": tail_ref})
        rows.append(
            {**common, "codec": "byte", "val_bits": [b + delta for b in base], "tail": tail_arm}
        )
    return rows


def _cells(table: str) -> list[list[str]]:
    return [
        [c.strip() for c in line.split("|")[1:-1]]
        for line in table.splitlines()
        if line.startswith("| typing")
    ]


def test_the_interval_carries_the_between_seed_term():
    """Two seeds that disagree in sign are unresolved however clean each run was."""
    (row,) = _cells(sweep.paired_table(_rows([-0.30, +0.40])))
    axis, arm, ref, shape, regime, steps, k, delta, ci, within, between, tails, spread = row

    assert (axis, arm, ref, shape, regime, k) == (
        "typing", "byte", "token", "square", "tokmatch", "2"
    )
    assert steps == "8,844 / 8,844", "both budgets belong in the row, not just the regime"
    assert "indistinguishable" in delta, "a sign-flipping axis was called resolved"
    assert within == "±0.00", f"within-run interval should be zero here, got {within}"
    # stdev([-0.3, 0.4]) / sqrt(2) * 1.96 = 0.69: entirely the between-seed term.
    assert ci == "±0.69" and between == "±0.69"
    assert spread == "-0.30, +0.40", "per-seed deltas must be printed, not just the mean"
    assert tails == "+0.00 / +0.00"


def test_a_consistent_effect_still_resolves():
    """The wider interval must not be so blunt that nothing can ever be called."""
    (row,) = _cells(sweep.paired_table(_rows([+11.0, +12.0])))
    assert "indistinguishable" not in row[7]
    assert row[7] == "**+11.50**"


def test_a_single_seed_reports_no_between_seed_term():
    """With k=1 there is no between-seed estimate at all, and the table has to
    say so rather than fall back to the within-run interval and look precise."""
    (row,) = _cells(sweep.paired_table(_rows([+0.40])))
    assert row[6] == "1" and row[10] == "n/a"


def test_an_arm_still_descending_cannot_resolve_however_consistent_its_seeds():
    """The schema-4 granularity rows in one test.

    Both seeds agree in sign, the effect is enormous, and the interval clears it
    easily -- and the arm is still shedding 8 bits/1k steps while its reference
    has flattened. There is no interval that makes that a cost, so the guard has
    to sit outside the interval and override it.
    """
    (row,) = _cells(sweep.paired_table(_rows([+40.0, +45.0], tail_arm=-8.0, tail_ref=-0.1)))
    assert "unconverged" in row[7], "a difference against a moving arm was called resolved"
    assert row[11] == "-8.00 / -0.10", "both tails must be shown, not just the flag"
    # And it says *unconverged*, not *indistinguishable*: whether the interval
    # covers zero is not the defect and pretending it is would send the reader
    # after more seeds when what is missing is more steps.
    assert "indistinguishable" not in row[7]


def test_the_guard_reads_the_worst_seed_not_the_mean_of_them():
    """One converged seed does not launder a cell whose other seed is not."""
    rows = _rows([+11.0], tail_arm=-0.05) + [
        {**r, "seed": "s1"} for r in _rows([+12.0], tail_arm=-4.0)
    ]
    (row,) = _cells(sweep.paired_table(rows))
    assert row[6] == "2" and "unconverged" in row[7]
    assert row[11] == "-4.00 / +0.00"


def test_an_unmeasurable_tail_is_not_a_settled_one():
    """A run with a single eval gives no evidence either way, and "we cannot
    tell" must not be reported the same as "it had stopped improving"."""
    rows = [{**r, "tail": float("nan")} for r in _rows([+11.0, +12.0])]
    (row,) = _cells(sweep.paired_table(rows))
    assert "unconverged" in row[7]


def test_a_settled_pair_is_judged_on_its_interval_alone():
    """The guard must not be so blunt that a flat pair can never resolve. A tail
    inside tolerance is a converged run, and the table goes back to the CI."""
    (row,) = _cells(sweep.paired_table(_rows([+11.0, +12.0], tail_arm=-0.4, tail_ref=-0.2)))
    assert row[7] == "**+11.50**", f"a converged pair was flagged: {row[7]}"


def test_an_arm_without_its_reference_is_skipped():
    """Half a pair is not a paired difference. A sweep that died partway through
    leaves exactly this, and it must drop the cell rather than report one arm."""
    rows = [r for r in _rows([+0.40]) if r["codec"] != "token"]
    assert _cells(sweep.paired_table(rows)) == []


def _history(points: list[tuple[int, float]]) -> list[dict]:
    return [{"step": s, "bits_per_drawing": b} for s, b in points]


def test_the_tail_is_a_rate_so_the_eval_interval_does_not_change_it():
    """The two budget regimes evaluate on different intervals -- 500 steps
    step-matched against 1,474 token-matched -- so a tail expressed as a drop
    per eval would not be comparable between the rows being differenced."""
    assert sweep.tail(_history([(0, 200.0), (500, 199.0)])) == -2.0
    assert sweep.tail(_history([(0, 200.0), (1000, 198.0)])) == -2.0


def test_a_run_with_one_eval_has_no_measurable_tail():
    """A rate needs two points. Reporting 0.0 here would read as 'converged',
    which is the one thing a single-eval run gives no evidence for."""
    assert math.isnan(sweep.tail(_history([(500, 200.0)])))
    assert math.isnan(sweep.tail([]))


def _decelerating(steps: int, every: int) -> list[dict]:
    """A run still improving, ever more slowly -- the shape every arm here has."""
    return _history(
        [(s, 160.0 + 40.0 * math.exp(-s / 2000.0)) for s in range(every, steps + 1, every)]
    )


def test_the_tail_window_does_not_depend_on_how_often_the_run_evaluated():
    """The fault this window exists to fix.

    `eval_every` is `steps // 6` under a token budget and a flat 500 otherwise,
    so the converged rows have 24 evals and everything they are differenced
    against has 6. Measured over the final *eval interval*, the same curve then
    reads 2-5x flatter on the denser history -- not because it converged but
    because its last interval sits where the cosine LR has already annealed.
    That is the difference between `(unconverged)` and a reported result.
    """
    sparse = _decelerating(12_000, 2_000)  # 6 evals, the token-matched cadence
    dense = _decelerating(12_000, 500)  # 24 evals, the converged cadence
    assert sweep.tail(sparse) == sweep.tail(dense)


def test_the_tail_is_not_a_single_eval_interval_because_a_val_mean_is_noisy():
    """The other half of the same fault, and the half that actually bit.

    Consecutive evals of one run differ by ~0.11-0.17 bits for reasons that are
    not the model improving. Over a 500-step interval that is a ±0.32 bits/1k
    noise floor against a 0.5 tolerance, so the guard was reading noise: the
    converged token arm reported a *positive* tail on a curve that fell
    monotonically across its whole last third. Averaging the window over 8
    intervals is what makes the estimate smaller than the thing it gates on.
    """
    rng = random.Random(0)
    truth = -0.15 / 1000.0  # bits per step, the measured converged slope
    interval_wrong = window_wrong = 0
    for _ in range(200):
        noisy = _history(
            [
                (s, 161.0 + truth * s + rng.gauss(0.0, 0.113))
                for s in range(500, 12_001, 500)
            ]
        )
        interval_wrong += sweep.tail(noisy, window=500 / 12_000) > 0
        window_wrong += sweep.tail(noisy) > 0
    # A sign error means reporting "this run is getting worse" about a run that
    # is monotonically improving. The interval estimator does it on a fifth of
    # the draws; the window estimator has to be an order of magnitude better.
    assert interval_wrong > 40, "the interval estimator is meant to be the noisy one"
    assert 10 * window_wrong < interval_wrong, (
        f"window {window_wrong}/200 vs interval {interval_wrong}/200 sign errors"
    )


def test_a_history_too_coarse_to_land_on_the_cutoff_widens_the_window():
    """Snapping to the last eval at or before the cutoff averages in the steeper
    earlier slope, so a coarse history overstates its tail. That is the safe
    direction: the failure mode is calling a converged run unconverged, which
    costs steps, not a retraction."""
    fine = _decelerating(12_000, 500)
    coarse = _decelerating(12_000, 2_400)  # cutoff 8,000 falls between evals
    assert abs(sweep.tail(coarse)) > abs(sweep.tail(fine))


def test_a_cell_re_requested_at_a_different_budget_is_not_skipped_as_done():
    """`--converged-steps` exists to be raised when `tail` says 12,000 was not
    enough, and the tag does not carry the budget -- so the resume check saw a
    file with the right name and skipped the run that was asked for, reporting a
    12,000-step number in a row headed 20,000."""
    cfg = sweep.TrainConfig(codec="bit", shape="square", steps=20_000, token_budget=None)
    twelve_k = {"schema": sweep.SCHEMA, "config": {"steps": 12_000, "token_budget": None}}

    assert sweep.resume_verdict(twelve_k, cfg) == "mismatch"
    assert sweep.resume_verdict(None, cfg) == "run"
    assert sweep.resume_verdict({**twelve_k, "schema": sweep.SCHEMA - 1}, cfg) == "run"
    assert sweep.resume_verdict({**twelve_k, "config": {"steps": 20_000,
                                                        "token_budget": None}}, cfg) == "done"


def test_a_token_budgeted_cell_is_matched_on_tokens_not_on_derived_steps():
    """`train` derives `steps` from the token budget, so the driver does not
    know it before the run. Comparing steps would call every token-matched cell
    of every resumed sweep a mismatch."""
    cfg = sweep.TrainConfig(codec="bit", shape="square", token_budget=24_000_000)
    record = {"schema": sweep.SCHEMA,
              "config": {"steps": 1_128, "token_budget": 24_000_000}}

    assert sweep.resume_verdict(record, cfg) == "done"
    assert sweep.resume_verdict(
        {**record, "config": {"steps": 1_128, "token_budget": 48_000_000}}, cfg
    ) == "mismatch"


def _budget_rows(ladder: dict[int, dict[str, float]], codec: str = "byte",
                 corpus: str = "plain") -> list[dict]:
    return [
        {"codec": codec, "shape": "square", "steps": steps, "seed": seed,
         "corpus": corpus, "bits_per_drawing": bits}
        for steps, seeds in ladder.items()
        for seed, bits in seeds.items()
    ]


def _budget_line(rows: list[dict], codec: str = "byte") -> str:
    (line,) = [r for r in sweep.budget_table(rows).splitlines() if r.startswith(f"| {codec}")]
    return line


def test_the_budget_table_reports_what_the_largest_budget_increase_bought():
    """The across-schedule half of convergence. A tail is measured inside one
    cosine-to-zero schedule, where the LR is small by the end whatever the model
    is doing; two schedules of different lengths landing on the same number is
    evidence that does not depend on either of them."""
    line = _budget_line(_budget_rows({8_844: {"s0": 161.89}, 12_000: {"s0": 161.94}}))
    assert "**+0.05**" in line and "1.36x" in line
    assert "8,844→161.9×1" in line and "12,000→161.9×1" in line


def test_the_ladder_is_paired_on_the_seed():
    """Rungs are not always run at every seed, and the run-to-run spread at fixed
    config is the size of the budget effect being measured: byte/square/s0 goes
    161.8 -> 162.6 -> 160.9 over 8,844 -> 12,000 -> 24,000 steps. Differencing a
    one-seed rung against a two-seed mean reports -1.0 where the seed-0 ladder
    says -1.7, which is the difference between 'converged' and 'not'."""
    line = _budget_line(
        _budget_rows({12_000: {"s0": 162.62, "s1": 161.26}, 24_000: {"s0": 160.91}})
    )
    assert "**-1.71**" in line, f"unpaired ladder: {line}"
    assert "| s0 |" in line, "the seeds actually compared have to be named"


def test_rungs_with_no_seed_in_common_report_no_difference():
    """Two rungs at disjoint seeds measure the budget and the seed at once, and
    there is no way to attribute the result to either. Say so rather than print
    a number that looks like the others."""
    line = _budget_line(_budget_rows({12_000: {"s1": 161.26}, 24_000: {"s0": 160.91}}))
    assert "_no shared seed_" in line and "| — |" in line


def test_an_arm_at_a_single_budget_is_not_a_replication():
    """One run cannot replicate itself, and a row implying otherwise would be
    the exact over-claim this table exists to prevent."""
    assert sweep.budget_table(_budget_rows({12_000: {"s0": 161.94}})) == ""


def test_a_ladder_never_mixes_two_corpora():
    """The worst number this project has reported came from here. Tier C ran an
    x18-augmented arm and a plain one at byte/square/6,000/s0, the key was blind
    to the corpus, and the augmented rung silently overwrote the plain one --
    347.7 against 175.7 -- then got differenced against a 4,797-step run as if
    augmentation were a budget. A ladder that mixes corpora is not a ladder."""
    rows = (
        _budget_rows({4_797: {"s0": 332.5}, 6_000: {"s0": 347.7}})
        + _budget_rows({6_000: {"s0": 175.7}}, corpus="aug:shifts=(-8, 8)")
    )
    table = sweep.budget_table(rows)
    assert "| plain |" in table
    # The augmented arm has one rung of its own, so it is not a replication and
    # must not appear at all -- least of all inside the plain arm's ladder.
    assert "175.7" not in table
    assert "**+15.20**" in table  # 347.7 - 332.5, the plain arm's own step


def _written(directory, name: str, **extra) -> None:
    """The smallest record `summarise` will read, plus whatever the test needs."""
    (directory / f"{name}.json").write_text(json.dumps({
        "name": name,
        "schema": sweep.SCHEMA,
        "config": {"codec": "byte", "shape": "square", "steps": 24_000,
                   "categories": ["cat"], "extra": {}},
        "model": {"params": 824_704},
        "val_lengths": {"mean": 111.6},
        "history": [{"step": 12_000, "bits_per_drawing": 423.0},
                    {"step": 24_000, "bits_per_drawing": 422.1}],
        "final": {"bits_per_drawing": 422.1, "gen_validity": 0.99, "gen_nonempty": 1.0,
                  "gen_n": 128, "gen_faults": {}, "gen_truncated": 0.01,
                  "gen_length_emd": 27.7},
        "best": {"step": 24_000, "bits_per_drawing": 422.1},
        **extra,
    }))


def test_two_corpora_never_average_into_one_cell(tmp_path, monkeypatch):
    """Run 4's fault, at the level that would have caught it. Two records with
    the same regime, codec, shape and budget, and different val splits, are two
    measurements and not two seeds of one.

    `budget_table` was keyed by corpus after the Tier C augmentation mix-up and
    this table was not -- it relied on the regime string naming the corpus
    (`m5b24000eps4`), which is a convention, and a convention cannot be checked.
    """
    monkeypatch.setattr(sweep, "RUNS", tmp_path)
    _written(tmp_path, "quickdraw_r24000_byte_square_s0", corpus={"val": "aaaa", "train": "a"})
    _written(tmp_path, "quickdraw_r24000_byte_square_s1", corpus={"val": "bbbb", "train": "b"},
             final={"bits_per_drawing": 999.9, "gen_validity": 0.5, "gen_nonempty": 1.0,
                    "gen_n": 128, "gen_faults": {}, "gen_truncated": 0.0,
                    "gen_length_emd": 1.0},
             best={"step": 24_000, "bits_per_drawing": 999.9})
    table = sweep.summarise("quickdraw")
    body = [ln for ln in table.splitlines() if ln.startswith("| r24000 |")]
    assert len(body) == 2, f"one row per corpus, never a mean of both: {body}"
    assert all("| 1 |" in ln for ln in body), "each row has one seed, not two"


def test_a_paired_difference_never_crosses_a_corpus():
    """The subtraction is per program, so two arms that never scored the same
    programs are not noisily different -- they are different quantities. Run 4
    reported 602.06 against 422.09 as a 180-bit result, and the two numbers came
    from corpora 44% apart in bytes per drawing."""
    rows = _rows([1.5])
    for row in rows:
        row["corpus"] = "val:aaaa" if row["codec"] == "byte" else "val:bbbb"
    assert sweep.paired_table(rows) == "", "no pair survives a corpus mismatch"
    for row in rows:
        row["corpus"] = "val:aaaa"
    assert "**+1.50**" in sweep.paired_table(rows), "the same rows pair once they match"


def test_a_retracted_record_keeps_its_likelihood_and_loses_its_sampling(tmp_path, monkeypatch):
    """Run 4's case. Its composition sampler ordered unmasking by NaN, so every
    generation column described the backend it ran on -- 9.17 strokes per grid
    on CPU against 3.70 on MPS -- while everything from a forward pass was
    checked against the other device and stood.

    Deleting the run would lose a sound 602.06 and averaging it whole would
    publish a void 0.195, so the record names its own bad columns. The failure
    this guards against is subtler than either: `halted` and `wellformed` are
    read off `gen_faults`, and dropping that key makes the `or {}` default
    report a retracted run as `halted` 1.000 -- a fabricated number where the
    honest answer is that nobody knows.
    """
    monkeypatch.setattr(sweep, "RUNS", tmp_path)
    _written(tmp_path, "quickdraw_good24000_byte_square_s0")
    _written(tmp_path, "quickdraw_void24000_byte_square_s0",
             final={"bits_per_drawing": 602.1, "gen_validity": 0.195,
                    "gen_nonempty": 1.0, "gen_n": 128,
                    "gen_faults": {"no_halt": 42}, "gen_truncated": 0.0},
             best={"step": 24_000, "bits_per_drawing": 602.1},
             retracted={"columns": ["gen_validity", "gen_faults"], "reason": "NaN ordering"})
    table = sweep.summarise("quickdraw")

    def cells(regime: str) -> list[str]:
        line = [ln for ln in table.splitlines() if f"| {regime} |" in ln][0]
        return [c.strip() for c in line.split("|")[1:-1]]

    # ... | bits/drawing | drift | tail | best@ | gen valid | halted | wellformed
    # | trunc | len EMD |, so the generation half is the last five.
    void = cells("void24000")
    assert "602.1" in void[8], "the likelihood half survives retraction"
    assert void[-5:] == ["nan"] * 5, f"every generation cell must be nan: {void}"
    assert "quickdraw_void24000_byte_square_s0" in table

    good = cells("good24000")
    assert good[-5:] == ["0.990", "1.000", "1.000", "0.010", "27.70"], (
        f"a sound row is untouched: {good}"
    )


def test_replicates_report_the_resolution_floor_rather_than_overwriting():
    """Two records that differ only in RNG stream are replicates, not rungs.
    Averaging them is right and letting one win is not -- and their spread is
    this corpus's resolution floor, measured for free. Tier C: three draws of
    byte/square/1,000 span 1.52 bits, which is 4x what a two-seed estimate of
    the same floor reported."""
    rows = _budget_rows({1_000: {"s0": 158.51}, 2_000: {"s0": 199.7}})
    rows += _budget_rows({1_000: {"s0": 157.18}})
    line = _budget_line(rows)
    assert "1,000→157.8×1" in line, f"replicates must average: {line}"
    assert "1.33 bits ×1" in line


def _planner_rows(**overrides) -> list[dict]:
    """The two claim-3 arms at two budgets, on one corpus and one seed.

    The numbers are the real ones from `runs/quickdraw_planner*eps2_*.json`,
    because every assertion below is about a fault those records exposed and a
    synthetic ladder would not have exposed any of them.
    """
    base = {
        "codec": "byte", "shape": "balanced", "seed": "s0",
        "corpus": "val:a54035e4", "params": 825_080, "composition_stderr": 0.0,
        "drift": 0.0, "gen_order": "random", "gen_validity": 1.0,
        "length_emd": 4.35, "val_bits": [100.0] * 8, "retracted": [],
        # Every one of these runs finished, and so did every record on disk --
        # the trainer used to write only at exit, so existing was finishing.
        "complete": True,
        # The real records are schema 1: they were trained before the eval
        # stopped reseeding the global RNG stream.
        "schema": 1,
    }
    cells = [
        ("ar", 12_000, 602.16, 181.45, 420.71, -1.75, ["factorisation"]),
        ("ar", 24_000, 596.66, 181.68, 414.98, -0.78, ["factorisation"]),
        ("diffusion", 12_000, 612.38, 190.43, 421.95, -1.83,
         ["factorisation", "diffusion_elbo"]),
        ("diffusion", 24_000, 604.25, 188.11, 416.14, -0.84,
         ["factorisation", "diffusion_elbo"]),
    ]
    return [
        {**base, "name": f"quickdraw_planner{arm}{steps}eps2_byte_balanced_s0",
         "arm": arm, "steps": steps, "bits_per_drawing": bits,
         "composition_bits": comp, "stroke_bits": stroke, "tail": tail_,
         "bound_sources": sources, **overrides}
        for arm, steps, bits, comp, stroke, tail_, sources in cells
    ]


def test_a_planner_ladder_never_differences_one_objective_against_the_other():
    """`comp_objective` is the axis claim 3 is about, so it names the arm.

    Keyed on `(codec, shape)` alone -- the AR sweep's key -- the two planner
    arms are one cell, and the ladder differences a diffusion rung against an AR
    one as if the objective were a budget. That is Tier C's augmentation mix-up
    in a new coordinate, and it is why `budget_table` takes `arm_fields`.
    """
    rows = _planner_rows()
    merged = [ln for ln in sweep.budget_table(rows).splitlines()
              if ln.startswith("| byte |")]
    assert len(merged) == 1 and "**-6.82**" in merged[0], (
        f"the default key must visibly merge the arms, or this test proves nothing: "
        f"{merged}"
    )
    table = sweep.budget_table(rows, arm_fields=("arm", "codec", "shape"),
                               arm_order=lambda key: key)
    ladders = [ln for ln in table.splitlines()
               if ln.startswith(("| ar |", "| diffusion |"))]
    assert len(ladders) == 2, f"one ladder per objective: {ladders}"
    assert "**-5.50**" in ladders[0], f"the AR arm's own rung: {ladders[0]}"
    assert "**-8.13**" in ladders[1], f"the diffusion arm's own rung: {ladders[1]}"


def test_the_rung_says_the_absolute_planner_numbers_are_not_asymptotes():
    """What the 12,000-step rung was run to decide, pinned as the reading.

    `PLAN.md` section 7 named the two branches in advance: within ~2.3 bits of
    24,000 and the absolute totals are quotable; much lower and only the paired
    differences are. Both arms moved 5.5-8.1 bits, so the second branch holds --
    and a regression that silently flattened the ladder would quietly relicense
    quoting 596.66 as a converged cost.
    """
    table = sweep.budget_table(_planner_rows(), arm_fields=("arm", "codec", "shape"),
                               arm_order=lambda key: key)
    deltas = [float(cell.strip("* ")) for line in table.splitlines()
              if line.startswith(("| ar |", "| diffusion |"))
              for cell in [line.split("|")[8]]]
    assert all(abs(d) > 2.3 for d in deltas), (
        f"a rung inside the planner's own ~2.3-bit floor would make the totals "
        f"quotable, and these are not: {deltas}"
    )


def test_a_planner_cell_is_one_record_and_never_a_mean_of_two():
    """Run 4 and its re-run share an arm, a corpus, a budget and a seed, and
    differ by four code fixes. Aggregating the way the AR table aggregates put
    both in one cell labelled `seeds 2` and averaged a retracted record's `nan`
    generation columns into a sound record's. There is one planner seed per arm,
    so the aggregation bought nothing and hid that two records were in it."""
    rows = _planner_rows()
    rerun = dict(rows[3], name="quickdraw_planner24000_byte_balanced_s0",
                 bits_per_drawing=602.06, retracted=["gen_validity"])
    table = sweep.planner_table([*rows, rerun])
    body = [ln for ln in table.splitlines() if ln.startswith("| quickdraw_")]
    assert len(body) == 5, f"one row per record: {body}"
    assert any("604.25" in ln for ln in body) and any("602.06" in ln for ln in body), (
        "neither record may be averaged away"
    )


def test_a_planner_difference_never_crosses_a_corpus():
    """Run 4's fault at the level that would have caught it, and the reason this
    table exists at all: the planner-against-AR difference lived only as
    arithmetic in `PLAN.md`, where nothing could check the corpus."""
    planner = _planner_rows()
    ar = [{"codec": "byte", "shape": "square", "steps": 24_000, "seed": "s0",
           "corpus": "val:different", "params": 824_704, "tail": -0.32,
           "val_bits": [60.0] * 8}]
    assert sweep.planner_vs_ar_table(planner, ar) == "", "a mismatch produces no row"
    ar[0]["corpus"] = "val:a54035e4"
    table = sweep.planner_vs_ar_table(planner, ar)
    assert "**+40.00**" in table, f"the same rows pair once the corpus matches: {table}"


def test_a_planner_difference_is_gated_on_both_tails():
    """The precondition `paired_table` applies, applied here for the same reason
    and with more force: the diffusion arm reads +55.54 against this AR arm at
    12,000 steps and +47.41 at 24,000, so the unqualified number moves 8 bits
    with the budget while the claim it supports does not."""
    ar = [{"codec": "byte", "shape": "square", "steps": 24_000, "seed": "s0",
           "corpus": "val:a54035e4", "params": 824_704, "tail": -0.32,
           "val_bits": [60.0] * 8}]
    def body(rows):
        return [ln for ln in sweep.planner_vs_ar_table(rows, ar).splitlines()
                if ln.startswith(("| ar |", "| diffusion |"))]
    assert all("(unconverged)" in ln for ln in body(_planner_rows()))
    assert not any("(unconverged)" in ln for ln in body(_planner_rows(tail=-0.1)))


def test_a_record_that_cannot_name_its_sampler_says_so(tmp_path, monkeypatch):
    """Three of this project's six faults were in the composition sampler, and
    none of the three records could say which sampler it ran -- the ordering was
    a default in `dm/models/planner.py`, not a field. A record whose generation
    columns cannot name what produced them is not comparable with anything, so
    it is named rather than assumed."""
    monkeypatch.setattr(sweep, "RUNS", tmp_path)
    record = {
        "name": "quickdraw_plannerdiff12000eps2_byte_balanced_s0", "kind": "planner",
        "schema": 1, "corpus": {"val": "a54035e4", "train": "c0ef9464"},
        "config": {"codec": "byte", "shape": "balanced", "steps": 12_000, "seed": 0,
                   "comp_objective": "diffusion", "categories": ["cat"], "extra": {}},
        "model": {"params": 825_080},
        "history": [{"step": 6_000, "bits_per_drawing": 620.0},
                    {"step": 12_000, "bits_per_drawing": 612.38}],
        "final": {"bits_per_drawing": 612.38, "composition_bits": 190.43,
                  "stroke_bits": 421.95, "composition_stderr": 0.51,
                  "bound_sources": ["factorisation", "diffusion_elbo"],
                  "gen_validity": 0.977, "gen_length_emd": 5.73},
        "best": {"step": 12_000, "bits_per_drawing": 612.38},
    }
    assert sweep.planner_row(record)["gen_order"] == "unknown"
    (tmp_path / f"{record['name']}.json").write_text(json.dumps(record))
    assert "predate `gen_order`" in sweep.summarise("quickdraw")

    record["config"]["gen_order"] = "random"
    (tmp_path / f"{record['name']}.json").write_text(json.dumps(record))
    table = sweep.summarise("quickdraw")
    assert "predate `gen_order`" not in table
    assert "| random |" in table


def test_a_planner_record_written_before_the_ar_control_is_still_a_diffusion_arm():
    """`comp_objective` did not exist when run 4 ran, and there was exactly one
    composition objective at the time -- the AR control is what created the
    choice -- so the field's absence *is* its value. `gen_order` is the opposite
    case: two orderings were live before it was recorded, so its absence is
    unknown and stays unknown. Defaulting the determined one and refusing to
    default the undetermined one is one rule, not two."""
    record = {
        "name": "quickdraw_planner24000_byte_balanced_s0", "kind": "planner",
        "corpus": {"val": "a54035e4"},
        "config": {"codec": "byte", "shape": "balanced", "steps": 24_000, "seed": 0},
        "model": {"params": 825_080},
        "history": [{"step": 24_000, "bits_per_drawing": 602.06}],
        "final": {"bits_per_drawing": 602.06, "composition_bits": 188.26,
                  "stroke_bits": 413.80},
        "best": {"step": 24_000, "bits_per_drawing": 602.06},
    }
    row = sweep.planner_row(record)
    assert row["arm"] == "diffusion" and row["gen_order"] == "unknown"
    assert row["bound_sources"] == [], "a missing key is not a claim of no slack"


def test_a_planner_row_below_the_current_schema_is_named_not_dropped():
    """The RNG leak moved batch order, not arithmetic, so a schema-1 planner row
    is still a valid measurement of its own model -- but two of them are not a
    controlled pair. Dropping them would lose five real runs; printing them
    unmarked would repeat the fault the schema bump exists to record. They are
    printed and named."""
    rows = _planner_rows()
    table = sweep.planner_table(rows)
    assert "| 1 ! |" in table, "a stale-schema row must be flagged in its own line"
    assert "seventh instrument fault" in table
    for row in rows:
        assert row["name"] in table

    current = [dict(r, schema=sweep.PLANNER_SCHEMA) for r in rows]
    assert "seventh instrument fault" not in sweep.planner_table(current)


def test_a_partial_record_is_not_a_rung_of_the_budget_it_asked_for(tmp_path, monkeypatch):
    """A killed run leaves a record now, and the record must not lie about it.

    Before `dm.train.checkpoint`, a record existed only if the loop had
    finished, so `config["steps"]` was always the achieved budget and every
    ladder, replicate key and cell in this file keyed on it. The schema-2
    planner re-run is what changed that: 11,500 steps of 12,000, killed, and
    under the old reader it would have entered the table as a 12,000-step run
    sitting in the same cell as a real one. That is the `aug6000`/`ladder6000`
    collision one level down.
    """
    monkeypatch.setattr(sweep, "RUNS", tmp_path)
    _written(tmp_path, "quickdraw_r24000_byte_square_s0",
             final={"bits_per_drawing": 422.1, "step": 24_000, "gen_validity": 0.99,
                    "gen_nonempty": 1.0, "gen_n": 128, "gen_faults": {},
                    "gen_truncated": 0.01, "gen_length_emd": 27.7})
    _written(tmp_path, "quickdraw_r24000_byte_square_s1", complete=False, steps_done=23_000,
             final={"bits_per_drawing": 999.9, "step": 23_000, "gen_validity": 0.5,
                    "gen_nonempty": 1.0, "gen_n": 128, "gen_faults": {},
                    "gen_truncated": 0.0, "gen_length_emd": 1.0})
    table = sweep.summarise("quickdraw")
    body = [ln for ln in table.splitlines() if ln.startswith("| r24000 |")]

    assert len(body) == 2, f"the two runs are different budgets, not two seeds: {body}"
    assert any("| 23000 ! |" in ln for ln in body), (
        f"the killed run must report where it stopped, flagged: {body}"
    )
    assert not any("999.9" in ln and "| 24000 |" in ln for ln in body), (
        "a killed run entered the cell of the budget it was launched with"
    )
    assert "`steps !` marks a run whose trainer never returned" in table
    assert "quickdraw_r24000_byte_square_s1" in table.split("`steps !`")[1]


def test_a_record_without_the_field_is_read_as_complete(tmp_path, monkeypatch):
    """Every record on disk predates `complete`, and every one of them finished.

    The old trainer wrote after the loop, so the existence of a record *was* the
    completion flag. Defaulting to True is what that fact means; defaulting to
    False would flag the entire corpus of past runs as partial.
    """
    monkeypatch.setattr(sweep, "RUNS", tmp_path)
    _written(tmp_path, "quickdraw_r24000_byte_square_s0")
    table = sweep.summarise("quickdraw")
    assert "steps !" not in table
    assert "| 24000 |" in table, "and it keys on the budget its history reached"
