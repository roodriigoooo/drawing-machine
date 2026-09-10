#!/usr/bin/env python3
"""Representation sweep: token vs byte vs bit, at matched budget.

Three regimes, and they answer different questions:

  token-matched  every arm processes the same number of tokens. The bit arm's
                 sequences are ~8x longer, so it gets ~8x fewer optimiser steps.
                 This is the honest equal-compute comparison: what a fixed
                 training budget buys you per representation.

  step-matched   every arm gets the same number of optimiser steps, which hands
                 the bit arm ~8x the compute. The generous-to-bit control.

  converged      every arm gets enough steps to stop improving. Neither budget
                 regime can answer the *representational* question -- "does this
                 alphabet cost bits, or is this arm merely behind on the same
                 curve?" -- because both stop the arms at points they reach at
                 wildly different rates. At the schema-4 sweep the bit arm was
                 still shedding 6-15 bits/1k steps at its token-matched endpoint
                 while its byte reference had flattened to 0.03-0.27, so the
                 +42-bit granularity difference was a rate, not a cost. This
                 regime is the one that compares asymptotes. Square only: the
                 43-token arms are already flat at their token-matched endpoint,
                 so the only arm that needs more steps is bit, and it needs a
                 partner at the same step count.

Results are assembled from runs/*.json, so an interrupted sweep resumes by
re-running with the same arguments.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dm.eval.metrics import paired_delta
from dm.isa.codec import CODECS
from dm.train import RUNS, SCHEMA, SHAPES, TrainConfig, train
from dm.train_planner import PLANNER_SCHEMA

#: Reporting order. Execution order is by cost (see `cells`), so a sweep that
#: dies partway still leaves a complete cheap arm rather than three half arms.
#: The delta arms sort after their absolute partners because relativity is read
#: *against* them: a `_delta` row alone says nothing.
ALPHABETS = ["token", "byte", "bit", "token_typed"]
CODEC_ORDER = [*ALPHABETS, *(f"{c}_delta" for c in ALPHABETS)]

#: Reporting order for regimes: the two budget points first, then the asymptote
#: they bracket. Alphabetical order would put `converged` first and read as if
#: it were the headline; it is the control that says what the headline means.
REGIME_ORDER = ["tokmatch", "stepmatch", "converged"]

#: Symbols per byte, so the driver can run the arms cheapest-first. The relative
#: view rewrites operand values and never the stream's shape, so a delta arm
#: costs exactly what its alphabet costs.
CODEC_COST = {
    name: (8 if name.startswith("bit") else 1) for name in CODEC_ORDER
}

#: The framing is sub-1M parameters, so a row above it has to say so rather than
#: be averaged into a table headed "sub-1M". `token_typed` is over at deep and
#: wide, which is the IconShop objection (README) making itself.
PARAM_BUDGET = 1_000_000

#: Largest |tail| (bits/drawing shed per 1,000 steps at the final eval) at which
#: a run is treated as having reached its asymptote.
#:
#: Not a physical constant -- a reporting guard, sized against the effects under
#: test. The typing and fusion axes are worth 0.5-5 bits, so an arm still losing
#: half a bit every thousand steps is shedding a whole small axis per ~10% of a
#: typical run, and a difference taken against it describes where training
#: stopped. On the schema-4 sweep this admits exactly the token-matched 43-token
#: arms (0.02-0.29) and rejects the step-matched arms (0.55-1.57) and the
#: token-matched bit arm (6.4-14.8), which is the partition the histories
#: support.
TAIL_TOLERANCE = 0.5

#: Fraction of the schedule the tail is measured over.
#:
#: The final eval *interval* is not a fair window, and using it was a fault with
#: two independent causes, both measured on the schema-4/5 records.
#:
#: 1. Noise. The val mean wanders +-0.11-0.17 bits between consecutive evals of
#:    one run (residual sd about a line fitted to the last third). A 500-step
#:    interval turns that into a +-0.32 bits/1k noise floor -- 64% of
#:    `TAIL_TOLERANCE`, on the rows the guard exists to judge. It showed: the
#:    converged token arm reported a *positive* tail, +0.025, on a curve that
#:    fell monotonically over its last third.
#: 2. Cadence. `eval_every` is `steps // 6` under a token budget and a flat 500
#:    otherwise, so the converged rows evaluate 24 times and every row they are
#:    differenced against 6. Their last interval is the last 4% of a
#:    cosine-to-zero schedule where the LR is ~1% of peak; the others' is the
#:    last 17%. Those windows do not ask the same question of a run.
#:
#: A fixed fraction fixes both: it averages 8 intervals instead of 1 (noise
#: floor 0.04 bits/1k) and asks one question of every row. 1/3 divides both
#: cadences exactly -- 2 of 6 evals, 8 of 24 -- so no row interpolates, and the
#: cosine LR is still ~25% of peak where the window opens, i.e. it is a window
#: in which a run that has not converged can still visibly move.
TAIL_WINDOW = 1 / 3


def cells(args) -> list[TrainConfig]:
    out: list[TrainConfig] = []
    for seed in range(args.seeds):
        # `--codecs` and not the whole of `CODEC_ORDER`: the delta arms double
        # the grid, and the summary reads whatever is on disk, so adding an
        # alphabet to a sweep must be a decision rather than a side effect of
        # registering a codec.
        for codec in sorted(args.codecs, key=CODEC_COST.__getitem__):
            # Token-matched: full codec x shape grid.
            for shape in SHAPES:
                out.append(
                    TrainConfig(
                        codec=codec, shape=shape, data=args.data,
                        categories=tuple(args.categories),
                        n_train=args.n_train, n_val=args.n_val,
                        token_budget=args.token_budget, batch_size=args.batch_size,
                        max_len=args.max_len, seed=seed, device=args.device,
                        gen_samples=args.gen_samples,
                        tag=f"{args.data}_tokmatch_{codec}_{shape}_s{seed}",
                    )
                )
            # Step-matched control: one shape only, it is a control not a grid.
            out.append(
                TrainConfig(
                    codec=codec, shape="square", data=args.data,
                    categories=tuple(args.categories),
                    n_train=args.n_train, n_val=args.n_val,
                    steps=args.steps, token_budget=None, batch_size=args.batch_size,
                    max_len=args.max_len, seed=seed, device=args.device,
                    gen_samples=args.gen_samples,
                    tag=f"{args.data}_stepmatch_{codec}_square_s{seed}",
                )
            )
            # Converged control, and it is the expensive one: these 8 cells are
            # ~2.4h against ~1.9h for the other 32, and ~2h of that is the two
            # bit runs. Placed last within each codec block so a sweep that dies
            # partway has already banked both budget regimes, and the codec loop
            # runs cheapest-first so this lands at the end of each seed.
            out.append(
                TrainConfig(
                    codec=codec, shape="square", data=args.data,
                    categories=tuple(args.categories),
                    n_train=args.n_train, n_val=args.n_val,
                    steps=args.converged_steps, token_budget=None,
                    batch_size=args.batch_size,
                    max_len=args.max_len, seed=seed, device=args.device,
                    gen_samples=args.gen_samples,
                    tag=f"{args.data}_converged_{codec}_square_s{seed}",
                )
            )
            # Extra budget rungs for `budget_table`, off by default and last
            # because a rung is normally a multiple of the converged budget. A
            # tail is
            # measured inside one cosine-to-zero schedule and is small by the
            # end of any such run, so the across-schedule evidence -- the same
            # arm at a different total budget -- is the half it cannot supply.
            # The budget goes in the tag because records are keyed by tag and
            # two rungs of one arm have to coexist; that is the whole point.
            for rung in args.budget_rungs:
                if seed >= args.budget_rung_seeds:
                    continue
                out.append(
                    TrainConfig(
                        codec=codec, shape="square", data=args.data,
                        categories=tuple(args.categories),
                        n_train=args.n_train, n_val=args.n_val,
                        steps=rung, token_budget=None, batch_size=args.batch_size,
                        max_len=args.max_len, seed=seed, device=args.device,
                        gen_samples=args.gen_samples,
                        tag=f"{args.data}_budget{rung}_{codec}_square_s{seed}",
                    )
                )
    return out


def tail(history: list[dict], window: float = TAIL_WINDOW) -> float:
    """Bits/drawing shed per 1,000 steps over the final `window` of the schedule.

    `drift` (final - best) catches a run reported *past* its val minimum. This
    catches the other half: a run reported *short* of it. They are the same
    fault -- a number describing where training stopped rather than what the
    representation costs -- and testing only one side misses it.

    Schema 3 failed the first test: the 43-token arms saw 27.7 epochs and were
    reported 0.2-1.2 bits past their minimum. Raising `n_train` to 100k fixed
    that and drift is now +0.00 in all 32 cells -- and every cell failed the
    second test instead, the bit arm by a factor of 50 against its own
    reference. A one-sided gate reported that as clean.

    The window is a fraction of the schedule and not the final eval interval
    (`TAIL_WINDOW`): a single interval is dominated by the eval-to-eval noise in
    a val mean -- a +-0.32 bits/1k floor against a 0.5 tolerance on the 500-step
    cadence -- and the cadence itself differs by regime, so the interval version
    was not even measuring the same part of the schedule on the two arms of a
    difference.

    Snapping to the last eval at or before the cutoff means a coarse history
    widens the window rather than interpolating, which averages in the steeper
    earlier slope and *overstates* the tail -- the safe direction for a guard.

    A tail is necessary evidence of an asymptote and not sufficient evidence:
    every run here anneals its LR to zero, so a small tail is partly a fact
    about the schedule. `budget_table` carries the schedule-independent half --
    the same arm trained under two different budgets.

    **Calibrated 2026-08-08, and it overstates by 2.7-3.8x.** The planner's
    12,000-step rung is the project's first pair where a tail can be checked
    against what doubling the budget actually bought. The diffusion arm's tail
    at 12,000 was -1.83 bits/1k, predicting -22.0 bits over the next 12,000
    steps; it delivered **-8.13**. The AR arm's was -1.75, predicting -21.0; it
    delivered **-5.50**. So a tail read as a rate is wrong by ~3x, in the
    direction the docstring above claims for the window and for the same reason
    -- the slope is still falling inside the window it is measured over.

    That is the safe direction for a *guard* and the wrong direction for an
    *extrapolation*, and this file has only ever used it as the first. Anything
    that multiplies a tail by a step count is reading it as the second.
    """
    if len(history) < 2:
        return float("nan")
    final = history[-1]
    cutoff = final["step"] * (1.0 - window)
    earlier = [e for e in history[:-1] if e["step"] <= cutoff] or [history[0]]
    start = earlier[-1]
    steps = final["step"] - start["step"]
    if steps <= 0:
        return float("nan")
    return (final["bits_per_drawing"] - start["bits_per_drawing"]) / steps * 1000.0


def worst(values: list[float]) -> float:
    """The value furthest from zero, sign kept. NaN if there is nothing to rank.

    A cell is only at its asymptote if *every* seed of it is, so the guard reads
    the worst seed rather than the mean of them.
    """
    ranked = [v for v in values if not math.isnan(v)]
    return max(ranked, key=abs) if ranked else float("nan")


#: The row fields that describe *sampling* rather than likelihood. A record
#: that retracts any generation column retracts all of these together --
#: `retracted()` says why they cannot be blanked one at a time.
GEN_FIELDS = ("gen_validity", "gen_nonempty", "halted", "wellformed",
              "truncated", "length_emd")


def retracted(record: dict) -> list[str]:
    """Columns a record declares void, or an empty list.

    A record can outlive the code that wrote it, and some of what it holds can
    be wrong while the rest is sound. Run 4 is the case this exists for: its
    composition sampler ordered unmasking by NaN, so every `gen_*` column
    describes the backend it ran on, while everything derived from a forward
    pass -- `bits_per_drawing`, its split, `val_bits` -- was checked against the
    other device and stands.

    Deleting the record would lose a sound 602.06; leaving it whole would let a
    void 0.195 be averaged into a table. So the record names its own bad
    columns and every reader honours that, which is the only version of this
    that survives someone reading the JSON directly.
    """
    return list((record.get("retracted") or {}).get("columns") or [])


def achieved_steps(record: dict) -> int:
    """How far the loop actually got, which is not always what it asked for.

    Every ladder, replicate key and budget cell in this file is keyed on
    `steps`, and until `dm.train.checkpoint` existed that could only ever be
    `config["steps"]` -- a record was written after the loop, so a record that
    existed had finished. Now a killed run leaves a record too, and reading its
    requested budget would file an 11,500-step model as a 12,000-step rung.
    That is the `aug6000`/`ladder6000` collision again (`budget_table`), one
    level down: a cell holding a run that is not the run the key names.

    `steps_done` and `final["step"]` are equal by construction -- the loop
    always evaluates at its last step -- and the second is what makes every
    record written before the field readable without a migration. The requested
    budget is the last resort and is reachable only for a record no trainer
    wrote, because both of them stamp `step` on every eval; the case that
    matters, a real run that stopped early, is answered before it.
    """
    final = record.get("final") or {}
    return record.get("steps_done") or final.get("step") or record["config"]["steps"]


def partial_caption(rows: list[dict]) -> str:
    """Names the rows whose trainer never returned, or an empty string.

    Kept in the table rather than filtered out of it, for the same reason
    schema-1 planner rows are kept: a run that stopped at 11,500 of 12,000 steps
    is a valid measurement of an 11,500-step model. What it is not is the
    12,000-step rung it was launched as, and `steps` already says so -- this
    says why the number there is not the number in the command line.
    """
    stopped = sorted(r["name"] for r in rows if not r["complete"])
    if not stopped:
        return ""
    return (
        "\n**`steps !` marks a run whose trainer never returned.** The record is "
        "the last eval that landed, written by `dm.train.checkpoint` rather than "
        "at exit, and `steps` is how far the loop actually got rather than the "
        "budget it was given — so the row is a real measurement of a shorter run "
        "and cannot be paired with a full-length one at the same nominal budget. "
        f"Affected: {', '.join(stopped)}."
    )


def planner_row(record: dict) -> dict:
    """A planner record flattened for `planner_table` and its two companions.

    The arm is read from `config`, not parsed out of the tag. An AR row has to
    go through `split_name` for historical reasons; a planner record carries
    `comp_objective`, `shape`, `codec` and `seed` as fields, and a field that
    exists is always a better key than a naming convention that has to hold --
    `PLAN.md` section 10 is a list of what conventions cost. The tag supplies
    the regime string only, which is a label and never a key.

    `gen_order` is the sampler that wrote this row's generation columns, and it
    is `unknown` on every record written before `dm/train_planner.py` began
    stamping it. Not backfilled: the three records that predate it used
    confidence ordering, and all three already carry a retraction saying so in
    full. Inventing provenance for a column is a smaller version of inventing
    the column.

    **`comp_objective` defaults where `gen_order` does not, and the asymmetry is
    the point.** Run 4's record predates the field, and at that time there was
    exactly one composition objective -- the AR control is what created the
    choice -- so the absence *is* the value. `gen_order` had two live values
    before it was recorded, so its absence is genuinely unknown. Defaulting a
    field whose value was determined and refusing to default one whose value was
    not is the same rule applied twice, not an inconsistency.

    `bound_sources` predates nothing and is simply absent on run 4; an empty
    list prints as `—` and never as "no slack", which is what a missing key
    would mean if it were read as the sources being none.
    """
    config, final = record["config"], record["final"]
    best = record.get("best") or {}
    row = {
        "name": record["name"],
        "schema": record.get("schema", 1),
        "complete": record.get("complete", True),
        "arm": config.get("comp_objective", "diffusion"),
        "codec": config["codec"],
        "shape": config["shape"],
        "steps": achieved_steps(record),
        "seed": f"s{config['seed']}",
        "corpus": corpus_key(record),
        "params": record["model"]["params"],
        "bits_per_drawing": final["bits_per_drawing"],
        "composition_bits": final["composition_bits"],
        "composition_stderr": final.get("composition_stderr", float("nan")),
        "stroke_bits": final["stroke_bits"],
        "bound_sources": final.get("bound_sources") or [],
        "drift": final["bits_per_drawing"] - best.get("bits_per_drawing", float("nan")),
        "tail": tail(record.get("history") or []),
        "gen_order": config.get("gen_order", "unknown"),
        "gen_validity": final.get("gen_validity", float("nan")),
        "length_emd": final.get("gen_length_emd", float("nan")),
        "val_bits": record.get("val_bits") or [],
        "retracted": retracted(record),
    }
    # All-or-nothing across the generation half, exactly as for an AR row: these
    # columns are functions of one another, so blanking one and keeping another
    # publishes a number nobody measured.
    if row["retracted"]:
        row |= dict.fromkeys(GEN_FIELDS, float("nan"))
    return row


def split_name(name: str, data: str) -> tuple[str, str, str, str] | None:
    """`{data}_{regime}_{codec}_{shape}_s{seed}` -> its four fields, or None.

    Splitting on `_` and counting from the left is what this did, and it works
    only while the regime is one token. Tier C's are not -- `c2_800`, `c2_1000`
    -- so every Tabler record parsed as codec `800_bit` and the summary died on
    an unknown codec name. The codec is the field that has to be recovered
    exactly, and it comes from a closed set, so it is matched rather than
    guessed at by position; `shape` and `seed` never contain an underscore, so
    they still come off the right by count.

    Returns None for a tag that does not fit the scheme at all, which the caller
    reports rather than dropping silently -- a record nobody can attribute is a
    reason to fix the tag, not to shrink the table.
    """
    rest = name[len(data) + 1 :]
    head, _, tail = rest.rpartition("_")
    head, _, shape = head.rpartition("_")
    if not head or not shape or not tail.startswith("s"):
        return None
    # Longest first: `token_typed_delta` also ends with `_delta`, and `bit`
    # would match the tail of nothing but must not shadow a longer name.
    for codec in sorted(CODEC_ORDER, key=len, reverse=True):
        if head == codec:
            return "", codec, shape, tail
        if head.endswith(f"_{codec}"):
            return head[: -len(codec) - 1], codec, shape, tail
    return None


def corpus_key(record: dict) -> str:
    """What a run was trained *on*, for grouping ladders and refusing bad pairs.

    A budget ladder differences the same arm at two budgets, so everything else
    has to be held fixed -- and `steps` is the only thing the regimes were meant
    to vary. Anything here that differs makes two records different arms rather
    than two rungs, and mixing them produced the project's worst single reported
    number: an x18-augmented Tier C run at 6,000 steps differenced against a
    plain one as if augmentation were a budget.

    **Prefers the corpus fingerprint**, which is a digest of the programs
    themselves (`dm/data/fingerprint.py`). The config-derived fallback below is
    for records written before fingerprints existed, and it is also the reason
    they exist: it cannot see a *defaulted* value, so `--rdp-eps 2.0` typed and
    omitted name one corpus under two keys, and `n_train` is not in it at all,
    so 100k and 350k programs of the same five categories are one key. Run 4 was
    scored against a baseline differing in categories, `rdp_eps` and `n_train`
    at once, and every config-derived key in the project said they matched.

    `share_init` stays in the fallback. It is not a corpus, but PLAN.md section
    10 is explicit that a shared-init row must never be paired against an
    independent-init one, and a ladder is a pairing.
    """
    corpus = record.get("corpus") or {}
    if corpus.get("val"):
        # `train` is deliberately *not* in the key. Two arms on one val split
        # are comparable per program even when one trained on more data, and a
        # key that refused them would hide a real measurement behind a
        # difference the table can simply state.
        return f"val:{corpus['val']}"
    config = record.get("config") or record
    parts = []
    if config.get("augment"):
        parts.append(
            "aug:" + ",".join(f"{k}={v}" for k, v in sorted(config["augment"].items()))
        )
    parts += [f"{k}={v}" for k, v in sorted((config.get("extra") or {}).items())]
    categories = config.get("categories") or []
    if len(categories) > 1:
        parts.append("+".join(sorted(categories)))
    if config.get("share_init"):
        parts.append("share_init")
    # Marked, so a table can tell a corpus that was *identified* from one that
    # was merely described. Only records predating `dm/data/fingerprint.py`
    # reach here; `scripts/fingerprint.py` backfills them.
    return "cfg:" + (" ".join(parts) or "plain")


def summarise(data: str) -> str:
    rows, stale, unparsed, planner, retractions = [], [], [], [], []
    for path in sorted(RUNS.glob(f"{data}_*.json")):
        r = json.loads(path.read_text())
        final = r.get("final") or {}
        if not final:
            continue
        if r.get("kind") == "planner":
            # A different model with its own schema counter, and a
            # bits/drawing that is an upper bound rather than an exact NLL
            # (`dm/models/planner.py`). Never averaged into this table -- putting
            # a bound and an exact number in one column is the fault this
            # project has now found at four different levels -- but it gets
            # tables of its own below rather than only a name in a list. It had
            # only a name in a list until the 12,000-step rung landed, and the
            # rung was run because `budget_table` needed it, which is a table
            # that structurally could not see it.
            planner.append(planner_row(r))
            continue
        if r.get("schema", 1) != SCHEMA:
            stale.append(r["name"])
            continue
        fields = split_name(r["name"], data)
        if fields is None:
            unparsed.append(r["name"])
            continue
        regime, codec, shape, seed = fields
        # "invalid" collapses two unrelated failures: a program that never
        # terminated, and one that named a malformed instruction. They fail
        # independently, so a single validity number describes neither.
        # `gen_faults` already carries the split.
        faults = final.get("gen_faults") or {}
        n_gen = max(1, final.get("gen_n", 1))
        # A run reported past its own val minimum measures where it landed on
        # its overfitting curve as much as it measures its representation, and
        # under schema 3 that drift was 0.2-1.2 bits -- the size of the typing
        # and fusion axes. It is a column so that it cannot go unnoticed again.
        best = r.get("best") or {}
        best_bits = best.get("bits_per_drawing", float("nan"))
        rows.append(
            {
                # Carried so a caption can name the run behind a cell. The four
                # key fields identify the *cell*, and a cell can hold several
                # records; only `name` identifies the record.
                "name": r["name"],
                "regime": regime, "codec": codec, "shape": shape, "seed": seed,
                "corpus": corpus_key(r),
                "complete": r.get("complete", True),
                "params": r["model"]["params"],
                "steps": achieved_steps(r),
                "seq": r["val_lengths"]["mean"],
                "bits_per_drawing": final["bits_per_drawing"],
                "best_step": best.get("step", float("nan")),
                "drift": final["bits_per_drawing"] - best_bits,
                # The other half of the same question -- see `tail`. Derived
                # from `history`, which every record already carries, so no
                # schema bump was needed to start reporting it.
                "tail": tail(r.get("history") or []),
                "gen_validity": final["gen_validity"],
                "gen_nonempty": final["gen_nonempty"],
                "halted": 1.0 - faults.get("no_halt", 0) / n_gen,
                "wellformed": 1.0
                - sum(v for k, v in faults.items() if k != "no_halt") / n_gen,
                "truncated": final.get("gen_truncated", float("nan")),
                "length_emd": final.get("gen_length_emd", float("nan")),
                "tokens": final.get("tokens_seen", 0),
                "val_bits": r.get("val_bits") or [],
            }
        )
        # Retraction is applied to the *row*, not to the record's columns, and
        # it is all-or-nothing across the generation half. Blanking individual
        # `gen_*` keys does not work: `halted` and `wellformed` are both read
        # off `gen_faults`, so a NaN there becomes a crash or -- worse -- a
        # missing key that the `or {}` default silently reports as `halted`
        # 1.000. The columns are functions of one another, so they are void
        # together. The likelihood half is untouched, which is the whole point
        # of retracting columns rather than deleting the run.
        if retracted(r):
            retractions.append(r["name"])
            rows[-1] |= dict.fromkeys(GEN_FIELDS, float("nan"))

    if not rows:
        # A corpus can hold planner runs and no AR ones -- Tier D will, by
        # design, since FS-COCO cannot enter the flat AR pipeline at all
        # (`PLAN.md` section 3). Returning early on an empty `rows` reported
        # that as "no completed runs", which is a different and false statement.
        if planner:
            return "\n".join([
                f"_no AR runs at schema {SCHEMA} on this corpus ({len(stale)} stale); "
                "the planner tables below stand alone, and the difference against a flat "
                "AR arm that claim 3 rests on cannot be taken here._",
                *planner_sections(planner, rows),
            ])
        return f"no completed runs at schema {SCHEMA} ({len(stale)} stale)"

    # Average over seeds. `steps` is part of the key, not just a printed column:
    # the converged regime is the one you extend when its tail says to, and
    # without this a half-extended cell would average a 12k-step run with a
    # 20k-step one under a single step count.
    # `corpus` is in the key for the same reason `steps` is: a cell averages
    # seeds and nothing else. `budget_table` already keyed by it after the Tier
    # C augmentation mix-up; this table did not, and only the convention that a
    # regime string names its corpus (`m5b24000eps4`) kept two corpora out of
    # one row. Run 4 is what that convention is worth.
    agg: dict[tuple, list[dict]] = {}
    for row in rows:
        agg.setdefault(
            (row["regime"], row["codec"], row["shape"], row["steps"], row["corpus"]), []
        ).append(row)

    order = lambda regime: (
        REGIME_ORDER.index(regime) if regime in REGIME_ORDER else len(REGIME_ORDER)
    )
    lines = [
        "| regime | codec | shape | params | seq | steps | Mtok | seeds | bits/drawing "
        "| drift | tail | best@ | gen valid | halted | wellformed | trunc | len EMD |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    over_budget = False
    for key in sorted(agg, key=lambda k: (order(k[0]), CODEC_ORDER.index(k[1]), k[2], k[3])):
        group = agg[key]
        n = len(group)
        mean = lambda f: sum(g[f] for g in group) / n
        params = group[0]["params"]
        over_budget |= params > PARAM_BUDGET
        cell_tail = worst([g["tail"] for g in group])
        budget = "*" if params > PARAM_BUDGET else ""
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {params:,}{budget} | "
            f"{mean('seq'):.0f} | {key[3]}"
            f"{'' if all(g['complete'] for g in group) else ' !'} | "
            f"{mean('tokens') / 1e6:.1f} | "
            f"{n} | **{mean('bits_per_drawing'):.1f}** | {mean('drift'):+.2f} | "
            f"{cell_tail:+.2f}{'' if abs(cell_tail) <= TAIL_TOLERANCE else ' !'} | "
            f"{mean('best_step'):.0f}/{key[3]} | {mean('gen_validity'):.3f} | "
            f"{mean('halted'):.3f} | {mean('wellformed'):.3f} | "
            f"{mean('truncated'):.3f} | {mean('length_emd'):.2f} |"
        )
    lines.append(f"\n_{len(rows)} runs. `seeds` is per row: a partially complete sweep "
                 "reports what each cell actually has, not the best cell's count._")
    if caption := partial_caption(rows):
        lines.append(caption)
    lines.append(
        "\nbits/drawing is the only cross-codec comparable number: every codec encodes "
        "the same bytecode, so total NLL in bits is the cost of transmitting one drawing. "
        "Per-token loss is NOT comparable across arms."
    )
    lines.append("\nMtok counts real tokens, not padded positions, so it is what the "
                 "token budget actually matched.")
    lines.append(
        "\n`drift` and `tail` are the two sides of one question -- is this number the "
        "representation's cost, or where training happened to stop? `drift` (final - best, "
        "with `best@` naming the argmin eval) is positive when the run was reported *past* "
        "its val minimum; it reached 1.2 bits at schema 3, larger than the typing and fusion "
        f"axes. `tail` is bits/drawing shed per 1,000 steps over the last "
        f"{TAIL_WINDOW:.0%} of the schedule -- a fixed fraction, not the final eval "
        "interval, because eval cadence differs by regime and the interval version "
        "reported a denser history as 2-5x flatter for the same curve. It is flagged `!` "
        f"above ±{TAIL_TOLERANCE:.1f}, i.e. when the run was reported *short* of its "
        "minimum. A row is an asymptote only with both near zero; a difference taken "
        "between one flat arm and one steep one is a rate, not a cost, however tight its "
        "interval looks. Both are within-schedule tests: see the budget-replication table "
        "below for the across-schedule one."
    )
    lines.append(
        "\n`halted` and `wellformed` decompose `gen valid`; `trunc` is the fraction that hit "
        "the length cap, and `len EMD` is the earth-mover distance in bytes between the "
        "generated and real length distributions -- termination described without reference "
        "to a cap."
    )
    lines.append(
        "\n**Every sampling column above is a single draw, taken at the run's final eval, "
        "and they are far noisier than `bits/drawing`.** `planbase24000eps2`'s `len EMD` "
        "reads 2.72 here; the same checkpoint over five seeds reads 10.40 +- 4.48, and that "
        "run's own last 25 evals span 2.72 to 42.45 -- a coefficient of variation of 0.43 "
        "against `bits/drawing`'s ~0.005 on the same corpus. `PLAN.md` section 7 drew a "
        "conclusion about claim 3's venue from the 2.72 and the conclusion was wrong. "
        "**And resampling one checkpoint is not enough either.** On the composed corpus "
        "one config's two training seeds read `len EMD` 191.0 +- 16.2 and 52.3 +- 11.2 over "
        "five draws each: the *between-seed* spread is 4x the between-draw spread, so a "
        "resampled column describes the checkpoint and not the arm. Quote the ordering "
        "across seeds, never the level. "
        "**Rank two arms on a sampling column only from `scripts/resample.py`**, which "
        "reports mean, sd, min and max over `k` seeds; these are for spotting a collapse, "
        "not for ranking."
    )
    if over_budget:
        lines.append(
            f"\n`*` marks a row above the {PARAM_BUDGET:,}-parameter framing. It is stated "
            "rather than averaged in. `token_typed` exceeds the budget at deep and wide "
            "purely through its embedding table, which is the objection to fused "
            "tokenisation making itself; the `reference` shape exceeds it on purpose, to "
            "put a floor under bits/drawing so an effect can be read against the headroom "
            "rather than against the total."
        )
    lines.append(budget_table(rows))
    lines.append(paired_table(rows))
    if stale:
        lines.append(
            f"\n_{len(stale)} run(s) excluded: written before schema {SCHEMA} and trained "
            f"under a different regime. Delete or re-run them._"
        )
    if unparsed:
        # Named, not counted. A record the table cannot attribute is a tag that
        # needs fixing, and a silent drop is how a whole corpus went missing.
        lines.append(
            f"\n_{len(unparsed)} run(s) excluded: the tag does not read as "
            f"`{data}_<regime>_<codec>_<shape>_s<seed>` -- {', '.join(sorted(unparsed))}._"
        )
    if retractions:
        # Named, and the generation columns of those rows read `nan`. A run
        # whose likelihood is sound and whose sampling is not stays in the
        # table for the half that survived; hiding it would lose a good number
        # and averaging it whole would publish a bad one.
        lines.append(
            f"\n_{len(retractions)} run(s) with retracted generation columns, shown as "
            f"`nan` and not averaged: {', '.join(sorted(retractions))}. "
            "The record carries the reason._"
        )
    lines += planner_sections(planner, rows)
    return "\n".join(lines)


def planner_sections(planner: list[dict], ar_rows: list[dict]) -> list[str]:
    """The three claim-3 tables and the two provenance notes, or nothing.

    One function rather than an inline block because a corpus can hold planner
    runs and no AR ones -- Tier D will, by design (`PLAN.md` section 3) -- and
    that path returns from `summarise` before the AR tables are built.
    """
    if not planner:
        return []
    out = [
        planner_table(planner),
        # The planner's own ladder, and the reason `budget_table` takes an
        # `arm_fields` argument. `comp_objective` leads the key because it is
        # the axis claim 3 is about: the two arms are the same model with the
        # composition level's objective swapped, so a ladder keyed on
        # (codec, shape) alone would put them in one cell and difference a
        # diffusion rung against an AR one as if the objective were a budget --
        # Tier C's augmentation mix-up in a new coordinate.
        budget_table(planner, arm_fields=("arm", "codec", "shape"),
                     arm_order=lambda key: key),
        planner_replicate_table(planner),
        planner_vs_ar_table(planner, ar_rows),
    ]
    void = sorted(r["name"] for r in planner if r["retracted"])
    if void:
        out.append(
            f"\n_{len(void)} planner run(s) with retracted generation columns, shown "
            f"as `nan`: {', '.join(void)}. The record carries the reason, and "
            "`scripts/resample.py` carries what replaces them._"
        )
    # Named rather than assumed. A sampler is part of a model's specification
    # (`PLAN.md` section 10) and three of this project's six faults were in one,
    # so a record that cannot say which sampler wrote its generation columns is
    # a record whose generation columns are not comparable with anything.
    unstamped = sorted(r["name"] for r in planner if r["gen_order"] == "unknown")
    if unstamped:
        out.append(
            f"\n_{len(unstamped)} planner run(s) predate `gen_order` and cannot name "
            f"the sampler that wrote their generation columns: {', '.join(unstamped)}._"
        )
    return out


def budget_table(rows: list[dict], arm_fields: tuple[str, ...] = ("codec", "shape"),
                 arm_order=None) -> str:
    """The same arm at two budgets: the half of convergence a tail cannot see.

    Every run here anneals its LR cosine-to-zero over its own `steps`, so a flat
    tail is partly a statement about the *schedule* -- late in any such run the
    LR is small and so is the slope, converged or not. The schedule-independent
    test is replication: train the same arm under a different total budget,
    which is a different schedule end to end, and see whether the number moves.
    Two runs 36% apart in length landing within 0.05 bits is evidence no single
    run can supply, and the sweep already pays for it -- the token-matched and
    converged regimes differ only in `steps`.

    **What names an arm is a parameter, because the planner has a different
    one.** An AR arm is a `(codec, shape)` pair; a planner arm is
    `(comp_objective, codec, shape)`, since `comp_objective` is the axis claim 3
    is about and `shape` names an entry in `PLANNER_SHAPES` rather than in
    `SHAPES`. The ladder itself is the same computation on either -- one arm,
    one corpus, two budgets, paired on the seed -- so it is one function.
    Copying it would have been the third place in this file where a corpus key
    has to be got right.

    Descriptive on purpose. The gate stays in `paired_table`, where the arms
    being differenced are named; this table is the evidence a reader needs to
    decide whether that gate is measuring an asymptote or an annealed LR.

    **The ladder is keyed by corpus as well as by arm, and a cell holds every
    record rather than the last one written.** Both halves were bugs, and Tier C
    found them because it is the only corpus with several regimes at one step
    count. `(codec, shape, steps, seed)` is not unique there: `aug6000` and
    `ladder6000` are both byte/square/6000/s0, and the first silently overwrote
    the second, so a rung of the **x18 augmented corpus** was reported as a rung
    of the plain one -- 347.7 against 175.7, differenced against a 4,797-step run
    as if it were the same arm getting more steps. A ladder that mixes corpora
    is not a budget ladder at all.

    Records that genuinely share an arm, a corpus, a budget and a seed are
    *replicates* -- they differ only in RNG stream -- so they are averaged, and
    `replicates` reports how many, because their spread is this corpus's
    resolution floor measured directly.
    """
    if arm_order is None:
        def arm_order(key):
            return (CODEC_ORDER.index(key[0]), *key[1:])
    ladders: dict[tuple, dict[int, dict[str, list[float]]]] = {}
    for row in rows:
        key = (*(row[f] for f in arm_fields), row["corpus"])
        ladders.setdefault(key, {}).setdefault(row["steps"], {}).setdefault(
            row["seed"], []
        ).append(row["bits_per_drawing"])
    def mean(values):
        return sum(values) / len(values)
    out = ["\n### Budget replication (the same arm, a different schedule)\n",
           f"| {' | '.join(arm_fields)} | corpus | "
           "budget ladder (steps -> bits/drawing, seeds) | "
           "top two | shared seeds | Δ bits | per-seed Δ | ratio | replicates |",
           "|" + "---|" * (len(arm_fields) + 8)]
    seen = False
    for key, ladder in sorted(ladders.items(), key=lambda kv: arm_order(kv[0])):
        *arm, corpus = key
        if len(ladder) < 2:
            continue
        seen = True
        rungs = sorted(ladder)
        curve = ", ".join(
            f"{s:,}→{mean([mean(v) for v in ladder[s].values()]):.1f}×{len(ladder[s])}"
            for s in rungs
        )
        lo, hi = rungs[-2], rungs[-1]
        # Paired on the seed, because a rung is not always run at every seed and
        # the run-to-run spread at fixed config is the size of the budget effect:
        # byte/square/s0 reports 161.8, 162.6 and 160.9 at 8,844, 12,000 and
        # 24,000 steps. Differencing a 1-seed rung against a 2-seed mean would
        # have called that -1.0 when the seed-0 ladder says -1.7, and the sign of
        # the 8,844 -> 12,000 step is seed-dependent outright.
        shared = sorted(set(ladder[lo]) & set(ladder[hi]))
        each = [mean(ladder[hi][seed]) - mean(ladder[lo][seed]) for seed in shared]
        delta = f"**{mean(each):+.2f}**" if each else "_no shared seed_"
        # The spread of records that differ in nothing but their RNG stream.
        # It is this corpus's resolution floor, measured rather than assumed,
        # and it is free wherever a cell happens to have been run twice.
        spreads = [
            max(v) - min(v) for rung in ladder.values() for v in rung.values() if len(v) > 1
        ]
        replicates = f"{max(spreads):.2f} bits ×{len(spreads)}" if spreads else "—"
        out.append(
            f"| {' | '.join(map(str, arm))} | {corpus} | {curve} | {lo:,} → {hi:,} | "
            f"{', '.join(shared) or '—'} | {delta} | "
            f"{', '.join(f'{d:+.2f}' for d in each) or '—'} | {hi / lo:.2f}x | "
            f"{replicates} |"
        )
    if not seen:
        return ""
    out.append(
        "\n`Δ bits` is what the *largest* budget increase on record bought, paired on the "
        "seed — a rung run at one seed cannot be differenced against a two-seed mean, "
        "because the run-to-run spread at fixed config is itself ~1 bit. Near zero means "
        "the arm has an asymptote and this is it, on evidence independent of any one LR "
        "schedule; a large negative value means the arm is still buying bits with steps "
        "and every difference taken against it is a rate. `×n` in the ladder is the seed "
        "count behind each rung. Budgets come from the regimes, which differ in `steps` "
        "and nothing else, so the ladder is free."
    )
    return "\n".join(out)


def planner_table(rows: list[dict]) -> str:
    """Claim 3's arms, their level split, and the two convergence guards.

    Separate from the main table on purpose and for one reason: `bits/drawing`
    is an **upper bound** here and exact there (`dm/models/planner.py`), so the
    two cannot share a column. That separation is why the planner has been read
    by hand out of `runs/*.json` since it was built -- and reading run 4 by hand
    is how it got compared against a corpus differing in categories, `rdp_eps`
    and `n_train` at once. The fix is not to be more careful, it is to give the
    planner tables that carry the corpus key like every other table here.

    `composition` and `stroke` are printed beside the total because the two
    levels share no gradient and answer to different questions, and because the
    12,000-step rung showed they do not converge together: on the AR arm the
    composition level moved **+0.23** bits between 12,000 and 24,000 steps while
    its stroke decoder moved **-5.73**. A total that is still descending can be
    a converged level plus a descending one, and only the split says which.

    **One row per record, and never a mean over records.** The first version of
    this table aggregated by `(arm, codec, shape, steps, corpus)` the way the AR
    table aggregates by regime, and immediately merged run 4 with its re-run --
    two records at one seed that differ by four code fixes -- into a cell
    labelled `seeds 2`. There is one planner seed per arm, so aggregation buys
    nothing here and costs the reader the ability to see that two records are in
    the cell. The replicate spread that aggregation would have hidden is
    reported where it belongs, in `budget_table`.
    """
    if not rows:
        return ""
    out = ["\n### Claim 3 — the planner arms (bits/drawing is an upper bound)\n",
           "| run | arm | codec | shape | corpus | params | steps | bits/drawing | "
           "composition | ±MC | stroke | slack | drift | tail | sampler | gen valid | "
           "len EMD | schema |",
           "|" + "---|" * 18]
    over_budget = False
    for row in sorted(rows, key=lambda r: (r["arm"], CODEC_ORDER.index(r["codec"]),
                                           r["shape"], r["steps"], r["name"])):
        over_budget |= row["params"] > PARAM_BUDGET
        out.append(
            f"| {row['name']} | {row['arm']} | {row['codec']} | {row['shape']} | "
            f"{row['corpus']} | {row['params']:,}"
            f"{'*' if row['params'] > PARAM_BUDGET else ''} | {row['steps']:,}"
            f"{'' if row['complete'] else ' !'} | "
            f"**{row['bits_per_drawing']:.2f}** | {row['composition_bits']:.2f} | "
            f"{row['composition_stderr']:.2f} | {row['stroke_bits']:.2f} | "
            f"{'+'.join(row['bound_sources']) or '—'} | {row['drift']:+.2f} | "
            f"{row['tail']:+.2f}{'' if abs(row['tail']) <= TAIL_TOLERANCE else ' !'} | "
            f"{row['gen_order']} | {row['gen_validity']:.3f} | {row['length_emd']:.2f} | "
            f"{row['schema']}{'' if row['schema'] >= PLANNER_SCHEMA else ' !'} |"
        )
    out.append(
        "\n`slack` is `bound_sources`: what stands between this row and an exact NLL. "
        "`factorisation` alone is an AR composition level, whose own number is exact; "
        "`factorisation+diffusion_elbo` adds the masked-diffusion bound. **Two rows both "
        "labelled 'upper bound' with different slack are two different numbers in one "
        "column**, so their difference reads as the ELBO's looseness and never as a "
        "modelling result. `±MC` is the bound's own Monte-Carlo standard error, 0 by "
        "construction on an AR composition level."
    )
    out.append(
        "\n`gen valid` and `len EMD` are single draws taken at the final eval and they are "
        "here to spot a collapse, not to rank two arms — `scripts/resample.py` is the only "
        "thing entitled to rank them. The gap is not small: the diffusion arm's 12,000-step "
        "record reads 5.73 here against 5.63 ±3.33 over five seeds, and its 24,000-step "
        "record's columns are retracted outright."
    )
    if caption := partial_caption(rows):
        out.append(caption)
    leaked = sorted(r["name"] for r in rows if r["schema"] < PLANNER_SCHEMA)
    if leaked:
        out.append(
            f"\n**`schema !` marks a run trained before `PLANNER_SCHEMA` "
            f"{PLANNER_SCHEMA}, when the eval reseeded the *global* RNG stream** "
            "(`dm/models/planner.py`, the seventh instrument fault). Each row is still a "
            "valid measurement of the model it names — the leak moved batch order, not "
            "arithmetic — but two such rows are **not a controlled pair**: the arms "
            "consumed different amounts of the stream and resumed on different batch "
            "orders. Read differences between them as carrying an unmeasured "
            f"trajectory term. Affected: {', '.join(leaked)}."
        )
    if over_budget:
        out.append(
            f"\n`*` marks a row above the {PARAM_BUDGET:,}-parameter framing. `params` "
            "counts *both* levels: a hierarchical model quoted at one level is the "
            "StrokeNUWA objection this project raises against others."
        )
    return "\n".join(out)


def planner_replicate_table(rows: list[dict]) -> str:
    """Two planner runs at the *same* config, differenced per program.

    The number this project has quoted twice and never had. Every claim-3 effect
    is read against a "same-config floor", and the value used for it -- ~2.3
    bits -- came from three stroke halves that were never a controlled
    comparison, because the eval reseeded the global RNG and the arms resumed on
    different batch orders (`PLAN.md`, the seventh instrument fault). That
    quantity is retracted and this table is what replaces it.

    Keyed on everything that defines the experiment -- arm, codec, shape, steps,
    corpus -- so a pair here differs in **trajectory alone**. Both levels are
    differenced separately, because they answer different questions: an AR
    composition level is exact, so its spread is pure trajectory noise with no
    estimator variance in it at all, and the stroke decoder is the half claim 3
    does not contribute.

    **What the key cannot see is code.** Two runs at one config are a replicate
    pair only if the interpreter between them did not change, and no field in a
    record says that -- `schema` is the only proxy, and it is exactly as good as
    the discipline of bumping it. The first pair this table printed is the
    demonstration: run 4 and its re-run share every key here and are *four code
    fixes* apart, which under `PLAN.md`'s own rule ("bump when a change makes new
    runs incomparable") should have been a schema bump and was not. So the table
    prints the schema of each side and refuses to call any pair a floor on its
    own authority.
    """
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        if row["val_bits"]:
            key = (row["arm"], row["codec"], row["shape"], row["steps"], row["corpus"])
            groups.setdefault(key, []).append(row)
    pairs = {k: v for k, v in groups.items() if len(v) > 1}
    if not pairs:
        return ""

    out = ["\n### Claim 3 — the planner's replicate floor (same config, two runs)\n",
           "| arm | steps | corpus | runs | Δ bits/drawing | 95% CI | Δ composition | "
           "Δ stroke | schema |",
           "|" + "---|" * 9]
    for key in sorted(pairs):
        members = sorted(pairs[key], key=lambda r: r["name"])
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                delta = paired_delta(b["val_bits"], a["val_bits"])
                schemas = f"{a['schema']} vs {b['schema']}"
                out.append(
                    f"| {key[0]} | {key[3]:,} | {key[4]} | {a['name'].split('_')[1]} vs "
                    f"{b['name'].split('_')[1]} | **{abs(delta['delta']):.2f}** | "
                    f"±{delta['ci95']:.2f} | "
                    f"{b['composition_bits'] - a['composition_bits']:+.2f} | "
                    f"{b['stroke_bits'] - a['stroke_bits']:+.2f} | {schemas} |"
                )
    out.append(
        "\n**A pair here is a candidate replicate floor, not automatically one.** The key "
        "covers arm, codec, shape, budget and corpus; it cannot see whether the code "
        "changed between the two runs, and `schema` is the only proxy for that. Read the "
        "schema column before reading the delta."
    )
    out.append(
        "\n- **`1 vs 1` — `planner24000` against `plannerdiff24000eps2` is NOT a "
        "replicate pair.** The two are four code fixes apart at one schema, which under "
        "this project's own rule should have been a bump and was not. Its 2.18 bits mixes "
        "trajectory noise with four changes, and `docs/claim3.md` says so at length. It "
        "is printed because hiding it would be worse: it is the standing evidence that "
        "the schema counter is the guard and has to be used.\n"
        "- **`1 vs 2` is the controlled one.** The RNG fix is what *makes* it controlled "
        "— the schema-1 run's batch order was perturbed by its own evals and the "
        "schema-2 run's is not — so the pair differs in trajectory and nothing else, "
        "which is the definition this project uses (`PLAN.md`: run-to-run noise is the "
        "trajectory, not the initialisation)."
    )
    out.append(
        "\nRead whichever floor applies beside the effect under test: the likelihood gap "
        "is **38.75–48.97 bits** and every candidate floor here is 1–3, which is why no "
        "plausible trajectory term reverses claim 3's verdict."
    )
    return "\n".join(out)


def planner_vs_ar_table(planner_rows: list[dict], ar_rows: list[dict]) -> str:
    """The comparison claim 3 rests on, paired per program and keyed by corpus.

    This existed only as arithmetic in `PLAN.md` until now, and that is exactly
    the shape of run 4's fault: a planner scored on `cat dog bus car tree` at the
    default `rdp_eps` was differenced against an AR arm on
    `cat bus flower sailboat bicycle` at 4.0, by hand, and no key in the project
    could see it. A difference taken per program between two arms that never
    scored the same programs is not noisy, it is a subtraction between two
    different quantities -- so the corpus fingerprint is part of the join and a
    mismatched pair produces no row rather than a wrong one.

    **The direction is asymmetric and it is the whole design.** The planner's
    number is an upper bound and the AR arm's is exact, so a *negative* Δ is
    decisive -- the planner won carrying a handicap -- and a positive one is
    only as decisive as the slack is small. `slack` names what that slack is;
    the AR-composition arm carries `factorisation` alone, which is why it is
    claim 3's tightest number even though it is not claim 3's model.
    """
    if not planner_rows or not ar_rows:
        return ""
    out = ["\n### Claim 3 — planner against the flat AR arm on the same programs\n",
           "| arm | steps | vs | steps | corpus | params | Δ bits/drawing | 95% CI | "
           "lower on | slack | tail planner/AR |",
           "|" + "---|" * 11]
    seen = False
    for planner in sorted(planner_rows, key=lambda r: (r["arm"], r["steps"])):
        for ar in sorted(ar_rows, key=lambda r: (r["shape"], r["steps"])):
            if ar["corpus"] != planner["corpus"] or ar["codec"] != planner["codec"]:
                continue
            if not planner["val_bits"] or not ar["val_bits"]:
                continue
            seen = True
            delta = paired_delta(planner["val_bits"], ar["val_bits"])
            lower = sum(p < a for p, a in zip(planner["val_bits"], ar["val_bits"]))
            tails = (planner["tail"], ar["tail"])
            # The same precondition `paired_table` applies, and for the same
            # reason: two arms stopped at different points on their own curves
            # differ by the rate they were separating at. It bites harder here
            # -- the 12,000-step planner rung reads +55.54 against an AR arm
            # that is already flat, and 24,000 reads +47.41, so the *unqualified*
            # number moves 8 bits with the budget while the claim does not.
            settled = all(
                not math.isnan(t) and abs(t) <= TAIL_TOLERANCE for t in tails
            )
            flag = "" if settled else "  (unconverged)"
            out.append(
                f"| {planner['arm']} | {planner['steps']:,} | AR {ar['shape']} | "
                f"{ar['steps']:,} | {planner['corpus']} | "
                f"{planner['params']:,} / {ar['params']:,} | "
                f"**{delta['delta']:+.2f}**{flag} | ±{delta['ci95']:.2f} | "
                f"{lower}/{len(planner['val_bits'])} | "
                f"{'+'.join(planner['bound_sources']) or '—'} | "
                f"{tails[0]:+.2f} / {tails[1]:+.2f} |"
            )
    if not seen:
        return ""
    out.append(
        "\nPositive means the planner costs more. **A negative number is decisive and a "
        "positive one is not**: the planner's bits/drawing is an upper bound and the AR "
        "arm's is exact, so the planner can only be *shown* to win, never shown to lose "
        "by more than its slack. Read `slack` before the interval."
    )
    out.append(
        "\n`95% CI` is the paired interval over val programs at one seed. It does not carry "
        "a between-seed term, because there is one planner seed per arm; the replicate "
        "spread that stands in for it is **~4.7 bits, provisionally** — one same-config "
        "pair whose second run was killed before it wrote a record "
        "(`docs/history/lost-run.md`), against effects of 39-56. The ~1.2 bits quoted here "
        "until 2026-08-10 came from two runs that were never same-config."
    )
    out.append(
        "\n`(unconverged)` is a precondition and not a statistical statement. Both planner "
        "arms fail it at both budgets, so **the paired differences are readable and the "
        "absolute planner totals are not** — which is what the 12,000-step rung was run to "
        "establish, and it established the second branch."
    )
    return "\n".join(out)


def paired_table(rows: list[dict]) -> str:
    """Each axis as a paired difference against its reference arm.

    Unpaired means cannot resolve these: the axes are worth ~1 bit/drawing and
    the spread between two val sets is ~3.5. Holding `data_seed` fixed means
    every arm scored the *same* programs, so the difference can be taken per
    program, where the shared difficulty cancels.

    Pairing kills the val-set variance and leaves the *seed* variance, and the
    interval has to carry both. An earlier version combined only the per-run
    intervals in quadrature, which answers "how precisely did these two runs
    measure their own difference" -- 5-15x narrower than the question actually
    being asked, and it marked genuinely unresolved axes as resolved. The
    interval below is the wider of the two error terms, and the per-seed deltas
    are printed beside it so the spread is never hidden behind a mean.

    No interval fixes the second failure, because it is not a variance problem.
    A difference between two arms is a *cost* only if both have stopped
    improving; if one is still descending, the difference shrinks with every
    further step and its value is a property of where the sweep stopped. The
    schema-4 token-matched granularity rows are exactly this: bit shedding
    6-15 bits/1k steps against a byte reference flat at 0.03-0.27, reported as
    +16/+42/+72 bits with per-seed signs agreeing, and every one of them would
    have printed as resolved. `tail` is therefore a precondition on the
    comparison, checked before the interval and overriding it.
    """
    # `corpus` is part of the cell key, not a column read afterwards. A paired
    # difference is taken per program, so two arms that never scored the same
    # programs cannot be differenced at all -- the subtraction is not merely
    # noisy, it is between two different quantities. Before this the key was
    # `(regime, codec, shape, seed)` and the corpus travelled in the *regime
    # string* by convention (`m5b24000eps4`); run 4 is what a broken convention
    # costs, and a convention cannot be checked.
    by_cell = {
        (r.get("corpus", ""), r["regime"], r["codec"], r["shape"], r["seed"]): r
        for r in rows
    }
    axes = [
        ("typing", "byte", "token", "one free bit: is this position an opcode?"),
        # The vocabulary sizes are read from the codecs rather than written out,
        # because they move with the ISA and a caption cannot be told that it
        # has: ISA v2 took this pair from 269 -> 1293 to 274 -> 1554 when
        # `Kind.XF` added a whole 256-value alphabet, and the printed note went
        # on claiming the old numbers.
        ("fusion", "token_typed", "token",
         "operand alphabet split per Kind, "
         f"{CODECS['token'].vocab_size} -> {CODECS['token_typed'].vocab_size}"),
        ("granularity", "bit", "byte", "8x length, identical information"),
        # Relativity, once per alphabet. It is orthogonal to the other three by
        # construction -- same vocabulary, same length, same information -- so
        # it is read within an alphabet and never across one, and four rows is
        # what says whether the view interacts with the alphabet or not.
        *(
            (
                "relativity", f"{alphabet}_delta", alphabet,
                "coordinates as deltas from the pen; same alphabet, same length",
            )
            for alphabet in ALPHABETS
        ),
    ]
    out = ["\n### Paired differences (positive = the arm costs more than its reference)\n",
           "| axis | arm | vs | shape | regime | steps arm/ref | seeds | Δ bits/drawing | "
           "95% CI | within-run | between-seed | tail arm/ref | per-seed Δ |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    seen = False
    for axis, arm, ref, note in axes:
        cells: dict[tuple, list[dict]] = {}
        for (corpus, regime, codec, shape, seed), row in by_cell.items():
            other = by_cell.get((corpus, regime, ref, shape, seed))
            if codec != arm or other is None or not row["val_bits"] or not other["val_bits"]:
                continue
            # Keyed by the arm's budget as well as the regime, and sorted by it,
            # so the same axis at several budgets reads as a ladder. That ladder
            # is the across-schedule test *of the difference itself*: an axis
            # whose delta holds while the budget changes is a cost, and one that
            # tracks the budget is a rate. Per-arm asymptotes are `budget_table`;
            # this is the same question asked of the number actually reported.
            cells.setdefault((shape, row["steps"], regime), []).append(
                {
                    **paired_delta(row["val_bits"], other["val_bits"]),
                    "tail_arm": row.get("tail", float("nan")),
                    "tail_ref": other.get("tail", float("nan")),
                    "steps_ref": other["steps"],
                }
            )
        for (shape, steps, regime), pairs in sorted(cells.items()):
            seen = True
            k = len(pairs)
            each = [p["delta"] for p in pairs]
            delta = sum(each) / k
            # Two error terms, and they answer different questions. `within` is
            # how precisely these particular runs measured their own difference;
            # `between` is how far that difference moves when the model seed
            # changes. Only the second generalises, and on this project it is
            # the larger, so the reported interval is the wider of the two.
            within = (sum(p["stderr"] ** 2 for p in pairs) ** 0.5) / k
            between = statistics.stdev(each) / math.sqrt(k) if k > 1 else 0.0
            ci = 1.96 * max(within, between)
            # Checked before the interval and overriding it: an interval on a
            # quantity that is still moving describes the precision of a
            # snapshot, and reporting it as resolved is the exact error the
            # quadrature CI made, in a coordinate no CI can reach.
            tails = [worst([p["tail_arm"] for p in pairs]), worst([p["tail_ref"] for p in pairs])]
            # NaN -- a run with a single eval -- counts as unsettled, not as
            # settled. "We cannot tell whether this arm had stopped improving"
            # and "this arm had stopped improving" are opposite claims, and only
            # one of them licenses reporting the difference as a cost.
            settled = all(not math.isnan(t) and abs(t) <= TAIL_TOLERANCE for t in tails)
            if not settled:
                flag = "  (unconverged)"
            elif abs(delta) <= ci:
                flag = "  (indistinguishable)"
            else:
                flag = ""
            out.append(
                f"| {axis} | {arm} | {ref} | {shape} | {regime} | "
                f"{steps:,} / {pairs[0]['steps_ref']:,} | {k} | "
                f"**{delta:+.2f}**{flag} | ±{ci:.2f} | ±{1.96 * within:.2f} | "
                f"{'±' + format(1.96 * between, '.2f') if k > 1 else 'n/a'} | "
                f"{tails[0]:+.2f} / {tails[1]:+.2f} | "
                f"{', '.join(f'{d:+.2f}' for d in each)} |"
            )
        if cells:
            out.append(f"| | | | | | | | _{note}_ | | | | | |")
    if not seen:
        return ""
    out.append(
        "\n`95% CI` is 1.96x the wider of the two error terms. `within-run` is the "
        "paired interval over val programs for a fixed seed; `between-seed` is the "
        "spread of that difference across seeds, which is the one that generalises "
        "and on this project is 5-15x the other. At k=2 the between-seed term has a "
        "single degree of freedom and 1.96 understates it, so read `per-seed Δ` "
        "directly: an axis whose per-seed deltas differ in sign is unresolved "
        "however narrow the interval looks."
    )
    out.append(
        "\n`steps arm/ref` is each side's training budget, and it differs within a row "
        "under token-matching by construction. Reading one axis and shape down the budget "
        "ladder replicates the *difference* across schedules, which is the test that "
        "separates a cost from a rate: a delta that holds while the budget changes is the "
        "representation's, and one that tracks the budget is where the sweep stopped."
    )
    out.append(
        f"\n`tail arm/ref` is the worst seed's bits/1k steps over the last "
        f"{TAIL_WINDOW:.0%} of the schedule for each "
        f"side of the difference. A row where either exceeds ±{TAIL_TOLERANCE:.1f} is marked "
        "`(unconverged)` whatever its interval: the arms were stopped at different points "
        "on their own curves, so the number is the rate they were separating at, not the "
        "cost of the representation. That marking is a precondition, not a statistical "
        "statement -- no number of seeds removes it, only more steps do."
    )
    return "\n".join(out)


def resume_verdict(record: dict | None, cfg: TrainConfig) -> str:
    """`"run"`, `"done"`, or `"mismatch"` for a cell whose record is on disk.

    Records are keyed by tag and the tag does not carry the budget, so a cell
    re-requested at a *different* budget names the same file. Comparing only the
    existence of that file -- which is what this did -- silently skips the run
    that was asked for and reports the old budget's number under the new one's
    name. The converged regime is exactly where that bites: `--converged-steps`
    exists to be raised when `tail` says 12,000 was not enough, and raising it
    did nothing at all.

    Overwriting instead would be worse. Two budgets of one arm are the
    across-schedule convergence evidence (`budget_table`), so clobbering the
    shorter one destroys the ladder to produce a number the ladder is what
    justifies. Hence a third verdict: say what is on disk, say what was asked
    for, and let the operator decide.

    A token-budgeted cell derives `steps` inside `train`, so its budget is the
    token count; a step-budgeted cell's is the step count. Comparing the one
    that was actually specified keeps this from firing on every token-matched
    cell of every resumed sweep.
    """
    if record is None or record.get("schema", 1) != SCHEMA:
        return "run"
    config = record.get("config") or {}
    if cfg.token_budget is not None:
        return "done" if config.get("token_budget") == cfg.token_budget else "mismatch"
    return "done" if config.get("token_budget") is None and config.get(
        "steps"
    ) == cfg.steps else "mismatch"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    # `composed` is summarise-only, and `main` enforces that. Its cells carry a
    # composition policy -- control, orbit sizes, the flat/structured spelling --
    # that `cells()` has no way to express, so a grid driven from here would
    # train the default policy under tags that claim nothing about which one.
    ap.add_argument("--data", default="synthetic",
                    choices=["synthetic", "quickdraw", "tabler", "composed"])
    ap.add_argument("--categories", nargs="+", default=["cat"])
    ap.add_argument("--seeds", type=int, default=2)
    # The four absolute alphabets are the standing grid. Pass the `_delta`
    # names to add the relativity axis -- it is read within an alphabet, so a
    # delta arm is only worth running beside the absolute partner it is
    # differenced against.
    ap.add_argument("--codecs", nargs="+", default=ALPHABETS, choices=CODEC_ORDER,
                    metavar="CODEC", help=f"default: {' '.join(ALPHABETS)}")
    ap.add_argument("--token-budget", type=int, default=24_000_000)
    ap.add_argument("--steps", type=int, default=3_000, help="for the step-matched control")
    # 12,000 is ~4x where the bit arm still had a 1.14 bits/1k tail and ~1.4x
    # where byte flattened to 0.10, so it is a starting point, not a proof. Read
    # `tail` on the converged rows: if it is still flagged, raise this and re-run
    # those cells. They cannot be resumed -- the LR is cosine-to-zero over
    # `steps`, so a longer run is a different schedule, not a continuation. Move
    # the old records aside so the resume check does not skip them.
    ap.add_argument("--converged-steps", type=int, default=12_000,
                    help="for the converged control; raise it if `tail` is still flagged")
    # A tail is a within-schedule test and every run anneals its LR to zero, so
    # a flat tail is partly a fact about the schedule. The across-schedule test
    # is the same arm at a different budget, and it is what `budget_table`
    # reads. One seed is the default because this rung replicates a difference
    # already measured at k=2 rather than estimating it afresh.
    ap.add_argument("--budget-rungs", type=int, nargs="*", default=[], metavar="STEPS",
                    help="extra square-shape budgets, one cell per codec, for the "
                         "replication ladder in `budget_table`")
    ap.add_argument("--budget-rung-seeds", type=int, default=1)
    # From TrainConfig, not a second copy of the number -- see dm/train.py.
    ap.add_argument("--n-train", type=int, default=TrainConfig.n_train)
    ap.add_argument("--n-val", type=int, default=TrainConfig.n_val)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--gen-samples", type=int, default=128)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--summarise-only", action="store_true")
    args = ap.parse_args()

    if args.data == "composed" and not args.summarise_only:
        raise SystemExit(
            "composed cells are launched from `python3 -m dm.train --data composed`, "
            "not from here: the composition policy (--control, --structured, "
            "--orbit-sizes) is the corpus, and `cells()` cannot express it. Use "
            "`--summarise-only` to read the runs that exist."
        )

    if not args.summarise_only:
        grid = cells(args)
        started, failed = time.time(), []
        for i, cfg in enumerate(grid, 1):
            path = RUNS / f"{cfg.tag}.json"
            record = json.loads(path.read_text()) if path.exists() else None
            verdict = resume_verdict(record, cfg)
            if verdict == "done":
                print(f"[{i}/{len(grid)}] skip {cfg.tag} (done)", flush=True)
                continue
            if verdict == "mismatch":
                on_disk = (record or {}).get("config", {})
                print(
                    f"[{i}/{len(grid)}] SKIP {cfg.tag}: on disk at "
                    f"steps={on_disk.get('steps')} token_budget={on_disk.get('token_budget')}, "
                    f"asked for steps={cfg.steps} token_budget={cfg.token_budget}. Records "
                    "are keyed by tag, so running this would overwrite a budget rung the "
                    "replication ladder is built from. Move the record aside to re-run it.",
                    flush=True,
                )
                continue
            print(f"[{i}/{len(grid)}] {cfg.tag}  ({time.time() - started:.0f}s elapsed)",
                  flush=True)
            try:
                train(cfg)
            except Exception:  # noqa: BLE001 -- one bad cell must not cost the other 31
                # A sweep is hours of compute. Record the cell, keep going, and
                # let a re-run pick up exactly what is missing.
                failed.append(cfg.tag)
                traceback.print_exc()
                print(f"[{i}/{len(grid)}] FAILED {cfg.tag}, continuing", flush=True)
        if failed:
            print(f"\n{len(failed)} cell(s) failed: {', '.join(failed)}", flush=True)

    summary = summarise(args.data)
    print("\n" + summary)
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / f"summary_{args.data}.md").write_text(summary + "\n")


if __name__ == "__main__":
    main()
