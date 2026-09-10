"""Direction 3's sequential scorer, its paired inference and its stability gate.

**Why a new scorer at all.** `dm.eval.context.score_spans` scores a whole
sequence in one batched forward. Soft decoding cannot be expressed that way: the
input at position `t` is fused with the state that produced position `t-1`, and
that state came from a fused input of its own. The recurrence is genuine, so the
scorer has to walk the target one symbol at a time. Computing `Delta_soft` with
the batched scorer is the trap `PLAN.md` names, and it would silently return
`Delta_standard` under another name.

**What stays Direction 2's.** The estimand, the four-way blocks, the two
controls, the contrast arithmetic, the resampling unit, the practical floor and
the case manifest are all inherited unchanged -- that is the whole reason
Direction 3 can compare its `Delta` against Direction 2's published one. The
contrast helpers are imported from `dm.eval.context` rather than re-derived here,
including two underscore-prefixed ones: that module's source digest is frozen
inside `docs/context-protocol-v5.json`, so it must not be edited to widen an
interface, and a re-derivation would risk arithmetic that merely *looks* like the
published estimand.

**What is new.** Three per-case quantities, each formed before any averaging:

```text
g_i(r,s,m) = Delta_i(soft) - Delta_i(standard)          mode gain
i_i(r,m)   = g_i(r,relational,m) - g_i(r,destroyed,m)   structure interaction
b_i(m)     = i_i(bit,m) - i_i(byte,m)                   representation, exploratory
```

Every cell resamples the **same** connected components in the same order, so a
difference of two cells is a difference of two aligned estimates rather than of
two independent resampling accidents. The draws are built once by
`component_draws` and handed to every `bootstrap` call.

**The structural zero is reported, not hidden.** Soft decoding prefills in
standard mode, so the first target symbol's distribution is identical in both
arms by construction. An aggregate that averaged it in would dilute the effect by
a factor that depends only on target length, so `first_symbol_identical` and the
tail are published separately.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ..isa.codec import Codec
from ..vm.interp import VM
from .context import (
    FOUR_WAY,
    ContextCase,
    _block,
    _difference,
    _field_contributions,
    components,
    score_spans,
)
from .feedback_contract import (
    ALPHA,
    BOOTSTRAP_REPS,
    BOOTSTRAP_UNIT,
    EVALUATION_MODES,
    GENERIC_GUARDS,
    REPORT_SCHEMA,
    REQUESTS_PER_CASE,
    REQUIRED_GENERATION_FIELDS,
    REQUIRED_SCORE_FIELDS,
    SOURCE_FEEDBACK_SCHEMA,
    SOURCE_FEEDBACK_SCHEMA_V2,
    STABILITY_GATE,
    STABILITY_PASSES,
    STABILITY_REPORT_SCHEMA,
    STABILITY_RMS_BAND,
    STABILITY_SUBSET_SIZE,
    WITHDRAWN_CLAUSE_REASONS,
    seed_for,
)
from .metrics import sample_programs

#: The three four-way blocks, in the order `dm.eval.context.score_cases` emits
#: them. Named here so a reader never has to count to twelve, and pinned by
#: `tests/test_feedback_eval.py` against that function's own output.
REQUEST_BLOCKS: tuple[str, ...] = ("primary", "target_control", "block_control")

_LOG2 = math.log(2)


def _raw_symbol_nlls(scores: list[dict], *, block: str) -> dict[str, list[float]]:
    """Return one complete four-way raw-symbol block.

    The scorer's twelve requests are laid out as three four-way blocks.  Keeping
    the block name in the persisted shape is deliberate: flattening the control
    names beside the primary names makes it too easy for a report reader to
    mistake one control's ``a_given_a`` for another's.
    """
    if block not in REQUEST_BLOCKS:
        raise ValueError(f"unknown raw-NLL request block {block!r}")
    if len(scores) != len(FOUR_WAY):
        raise ValueError(
            f"{block} has {len(scores)} requests, expected {len(FOUR_WAY)}"
        )
    return {
        name: [float(value) for value in score["per_symbol"]]
        for name, score in zip(FOUR_WAY, scores)
    }


# ---------------------------------------------------------------------------
# sequential scoring


def case_requests(case: ContextCase) -> list[tuple[bytes, bytes]]:
    """The 12 `(prefix, target)` pairs of one case, in the frozen block order.

    Identical to what `dm.eval.context.score_cases` builds. Restated rather than
    imported because that function scores as well as enumerates and its module is
    digest-frozen; `test_the_request_order_is_direction_2s` asserts the two agree
    by comparing the contrasts they produce.
    """
    return [
        # primary: relation-bearing prefix change, compatible targets
        (case.prefix_a, case.target_a), (case.prefix_a, case.target_b),
        (case.prefix_b, case.target_a), (case.prefix_b, case.target_b),
        # unrelated-target control: same prefixes, donor targets
        (case.prefix_a, case.control_target_a),
        (case.prefix_a, case.control_target_b),
        (case.prefix_b, case.control_target_a),
        (case.prefix_b, case.control_target_b),
        # unrelated-block control: same targets, irrelevant prefix edit
        (case.block_prefix_a, case.target_a),
        (case.block_prefix_a, case.target_b),
        (case.block_prefix_b, case.target_a),
        (case.block_prefix_b, case.target_b),
    ]


def _encoded(codec: Codec, prefix: bytes, target: bytes) -> tuple[list[int], list[int]]:
    """Prompt and continuation symbols, refusing a codec that does not split.

    The byte and bit codecs encode byte by byte, so a prefix and its continuation
    encode independently. A codec whose units span the seam would put the
    target's first symbol partly inside the prompt, and every per-byte column
    after it would be attributed to the wrong byte. Checked rather than assumed,
    because the pilot's two codecs both satisfy it and a later one might not.
    """
    prompt, continuation = codec.encode(prefix), codec.encode(target)
    if codec.encode(prefix + target) != prompt + continuation:
        raise ValueError(
            f"codec {codec.name!r} does not encode a prefix and its continuation "
            "independently, so a sequential score cannot be attributed by byte"
        )
    return prompt, continuation


@torch.no_grad()
def score_spans_sequential(
    model, requests: list[tuple[bytes, bytes]], codec: Codec, *,
    mode: str = "standard", device: str | torch.device = "cpu",
    max_len: int = 2048, batch_size: int = 32,
) -> list[dict]:
    """Per-symbol, per-byte and total target bits, teacher-forced one symbol at a time.

    Shape-identical to `dm.eval.context.score_spans`' output, plus `per_symbol`,
    so the two can be differenced request by request.

    Requests are grouped by `(prefix bytes, target bytes)` before batching. Every
    request of one case already shares both -- the case schema forces the two
    prefixes, the two block prefixes and all four targets to one length each --
    so a case is always one group and cases of equal shape merge into larger
    batches. Mixing shapes would need padding inside a recurrence, where a padded
    position still produces a state that the next position would carry.

    `prefix_bits` comes from one batched full forward rather than from this loop.
    It is a support diagnostic and is mode-independent by construction: `standard`
    and `soft` share one standard prefill, so the prefix costs the same bits in
    both, and re-deriving it sequentially would only add float noise to a column
    that carries no contrast.
    """
    if mode not in ("standard", "soft", "fused"):
        raise ValueError(f"unknown scoring mode {mode!r}")
    stride = codec.stride
    out: list[dict | None] = [None] * len(requests)

    groups: dict[tuple[int, int], list[int]] = {}
    for index, (prefix, target) in enumerate(requests):
        groups.setdefault((len(prefix), len(target)), []).append(index)

    for (prefix_bytes, target_bytes), members in sorted(groups.items()):
        if 1 + (prefix_bytes + target_bytes) * stride > max_len:
            raise ValueError(
                f"a request of {prefix_bytes}+{target_bytes} bytes exceeds "
                f"max_len={max_len}: a target scored on fewer symbols than it "
                "costs is not the estimand"
            )
        for start in range(0, len(members), batch_size):
            chunk = members[start : start + batch_size]
            pairs = [_encoded(codec, *requests[i]) for i in chunk]
            prompt = torch.tensor([p for p, _ in pairs], dtype=torch.long)
            continuation = torch.tensor([c for _, c in pairs], dtype=torch.long)
            logits = model.teacher_forced(prompt, continuation, device=device,
                                          mode=mode)
            nll = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                continuation.to(logits.device).reshape(-1),
                reduction="none",
            ).view(continuation.shape).cpu().numpy() / _LOG2
            for row, index in enumerate(chunk):
                per_symbol = nll[row]
                per_byte = per_symbol.reshape(target_bytes, stride).sum(axis=1)
                out[index] = {
                    "bits": float(per_byte.sum()),
                    "per_byte": [float(value) for value in per_byte],
                    "per_symbol": [float(value) for value in per_symbol],
                }

    for index, prefix_bits in _prefix_bits(model, requests, codec, device=device,
                                           max_len=max_len,
                                           batch_size=batch_size).items():
        out[index]["prefix_bits"] = prefix_bits  # type: ignore[index]
    return out  # type: ignore[return-value]


def _prefix_bits(model, requests: list[tuple[bytes, bytes]], codec: Codec, *,
                 device, max_len: int, batch_size: int) -> dict[int, float]:
    """Total prefix bits per request, scored once per distinct prefix.

    A case has three distinct prefixes across its twelve requests -- the two
    worlds and the one edited block, since the block control starts from world
    A's prefix -- so scoring per request would run the same forward four times.
    """
    distinct: dict[bytes, list[int]] = {}
    for index, (prefix, _) in enumerate(requests):
        distinct.setdefault(prefix, []).append(index)
    prefixes = sorted(distinct)
    # An empty prompt has no conditional symbols and therefore costs exactly
    # zero.  Passing it to the batched scorer would construct a `(B, 0)` tensor
    # and fail inside the transformer's reshape; the full validation guard uses
    # empty prompts intentionally, so handle the mathematical identity here.
    nonempty = [prefix for prefix in prefixes if prefix]
    scored = score_spans(
        model, [(b"", prefix) for prefix in nonempty], codec,
        device=device, max_len=max_len, batch_size=batch_size,
    ) if nonempty else []
    by_prefix = {prefix: row["bits"]
                 for prefix, row in zip(nonempty, scored)}
    return {index: float(by_prefix.get(prefix, 0.0))
            for prefix, members in distinct.items()
            for index in members}


def score_cases_sequential(
    model, cases: list[ContextCase], codec: Codec, *, mode: str = "standard",
    device: str | torch.device = "cpu", max_len: int = 2048, batch_size: int = 32,
) -> list[dict]:
    """One row per case: raw NLLs, both controls, the contrasts and `Delta`.

    `Delta` is Direction 2's estimand computed on Direction 2's cases with
    Direction 2's arithmetic; only the scorer underneath it is new. The row also
    carries `first_symbol_bits`, because that symbol's distribution is identical
    in `standard` and `soft` and its contribution to any aggregate is therefore a
    structural zero rather than a measurement.
    """
    requests: list[tuple[bytes, bytes]] = []
    for case in cases:
        requests += case_requests(case)
    scores = score_spans_sequential(model, requests, codec, mode=mode,
                                    device=device, max_len=max_len,
                                    batch_size=batch_size)
    rows = []
    for index, case in enumerate(cases):
        start = REQUESTS_PER_CASE * index
        primary = scores[start : start + 4]
        target_control = scores[start + 4 : start + 8]
        block_control = scores[start + 8 : start + 12]
        primary_block = _block(primary, case.target_bytes)
        target_block = _block(target_control, case.target_bytes)
        block_block = _block(block_control, case.target_bytes)
        rows.append({
            "case_id": case.case_id,
            "venue": case.venue,
            "mode": mode,
            "source_index": case.source_index,
            "donor_index": case.donor_index,
            "control_donor_index": case.control_donor_index,
            "target_bytes": case.target_bytes,
            "prefix_bytes": case.prefix_bytes,
            "strata": list(case.features.strata),
            # Persist all twelve requests, not only the primary four.  A
            # control total without its symbol columns cannot be audited for a
            # seam/length artefact after the report is written.
            "raw_symbol_nll_bits": {
                block: _raw_symbol_nlls(scores, block=block)
                for block, scores in (
                    ("primary", primary),
                    ("target_control", target_control),
                    ("block_control", block_control),
                )
            },
            "nll_bits": primary_block["nll_bits"],
            "primary": primary_block,
            "target_control": target_block,
            "block_control": block_block,
            "D": primary_block["contrast"],
            "D_target_control": target_block["contrast"],
            "D_block_control": block_block["contrast"],
            "Delta": _difference(primary_block["contrast"],
                                 target_block["contrast"]),
            "Delta_block": _difference(primary_block["contrast"],
                                       block_block["contrast"]),
            "fields": _field_contributions(primary, case.target_a),
            # The structural zero, published beside the total it cannot move.
            "first_symbol_bits": {
                name: score["per_symbol"][0]
                for name, score in zip(FOUR_WAY, primary)
            },
            "prefix_nll_bits": {
                "a": primary[0]["prefix_bits"], "b": primary[2]["prefix_bits"],
                "block_b": block_control[2]["prefix_bits"],
            },
        })
    return rows


def validation_costs(model, programs: Sequence[bytes], codec: Codec, *,
                     device: str | torch.device = "cpu",
                     max_len: int = 2048, batch_size: int = 32) -> dict:
    """Score ordinary validation likelihood in the two actual decode modes.

    This is deliberately the sequential teacher-forced estimand, not the
    repeated-prefill stability diagnostic. ``soft`` feeds each gold symbol
    through the same recurrence used by generation; a batched full forward
    would silently measure ``standard`` again. The returned per-program arrays
    make the denominator and the exact paired difference auditable.
    """
    programs = list(programs)
    if not programs:
        raise ValueError("validation cost needs at least one program: incomplete")
    requests = [(b"", program) for program in programs]
    per_mode: dict[str, np.ndarray] = {}
    for mode in EVALUATION_MODES:
        rows = score_spans_sequential(
            model, requests, codec, mode=mode, device=device,
            max_len=max_len, batch_size=batch_size,
        )
        per_mode[mode] = np.asarray([row["bits"] for row in rows],
                                    dtype=np.float64)
    standard = per_mode["standard"]
    soft = per_mode["soft"]
    return {
        "programs": len(programs),
        "standard_bits_per_drawing": float(standard.mean()),
        "soft_bits_per_drawing": float(soft.mean()),
        "validation_cost_delta_bits_per_drawing": float(
            (soft - standard).mean()
        ),
        "per_program_bits": {
            mode: [float(value) for value in values]
            for mode, values in per_mode.items()
        },
    }


def _free_running_mode_stats(model, codec: Codec, *, mode: str, n: int,
                             max_new: int, device: str | torch.device,
                             variates: torch.Tensor) -> dict:
    programs, cap_hit = sample_programs(
        model, codec, n=n, max_new=max_new, device=device,
        top_k=40, variates=variates, mode=mode,
    )
    traces = [VM().run(program) for program in programs]
    halted = np.asarray([trace.halted and not capped
                         for trace, capped in zip(traces, cap_hit)],
                        dtype=bool)
    valid_halt = np.asarray(
        [trace.halted and trace.valid and not capped
         for trace, capped in zip(traces, cap_hit)],
        dtype=bool,
    )
    truncated = np.asarray(cap_hit, dtype=bool)
    return {
        "mode": mode,
        "programs": n,
        "halted_rate": float(halted.mean()),
        "valid_halt_rate": float(valid_halt.mean()),
        "truncation_rate": float(truncated.mean()),
        "valid_halt_count": int(valid_halt.sum()),
        "truncation_count": int(truncated.sum()),
    }


def generic_guards(model, programs: Sequence[bytes], codec: Codec, *,
                   device: str | torch.device = "cpu", max_len: int = 2048,
                   batch_size: int = 32, generation_n: int | None = None,
                   generation_max_new: int | None = None,
                   variates: torch.Tensor | None = None,
                   variates_seed: int | None = None) -> dict:
    """Compute and apply Direction 3's generic soft-vs-standard guards.

    The likelihood guard is exact sequential teacher forcing over the complete
    supplied validation split. The termination guards use the same raw,
    unmasked sampler for both modes and one pre-generated uniform block, so a
    changed stopping path cannot change the random stream or denominator. No
    threshold is inferred from the returned values; all limits come from the
    frozen contract.
    """
    programs = list(programs)
    if not programs:
        raise ValueError("generic guards need a validation split: incomplete")
    validation = validation_costs(
        model, programs, codec, device=device, max_len=max_len,
        batch_size=batch_size,
    )
    n = len(programs) if generation_n is None else int(generation_n)
    if n <= 0:
        raise ValueError("generation guard needs a positive sample count: incomplete")
    width = 512 if generation_max_new is None else int(generation_max_new)
    if width <= 0:
        raise ValueError("generation guard needs a positive decode budget: incomplete")
    if variates is None:
        seed = (seed_for("variates") if variates_seed is None
                else int(variates_seed))
        variates = torch.rand(
            n, width,
            generator=torch.Generator().manual_seed(seed),
            dtype=torch.float32,
        )
    if tuple(variates.shape) != (n, width):
        raise ValueError(
            f"guard variates are {tuple(variates.shape)}, expected {(n, width)}: "
            "standard and soft must consume identical draws"
        )
    generated = {
        mode: _free_running_mode_stats(
            model, codec, mode=mode, n=n, max_new=width, device=device,
            variates=variates,
        )
        for mode in EVALUATION_MODES
    }
    valid_halt_delta = (generated["soft"]["valid_halt_rate"]
                        - generated["standard"]["valid_halt_rate"])
    truncation_delta = (generated["soft"]["truncation_rate"]
                        - generated["standard"]["truncation_rate"])
    values = {
        "validation_cost_delta_bits_per_drawing": (
            validation["validation_cost_delta_bits_per_drawing"]
        ),
        "valid_halt_delta": valid_halt_delta,
        "truncation_rate_increase": truncation_delta,
        # A loss is one-sided because a positive soft gain is not a reason to
        # forgive a negative one elsewhere in the report.
        "valid_halt_loss": max(0.0, -valid_halt_delta),
    }
    failures: dict[str, object] = {}
    nonfinite = [name for name, value in values.items()
                 if not math.isfinite(float(value))]
    if nonfinite:
        # A NaN comparison would otherwise pass every one-sided threshold.  A
        # guard that cannot be evaluated is an incomplete guard, never a pass.
        failures["nonfinite"] = nonfinite
    if values["validation_cost_delta_bits_per_drawing"] > GENERIC_GUARDS[
            "max_validation_cost_bits_per_drawing"]:
        failures["max_validation_cost_bits_per_drawing"] = values[
            "validation_cost_delta_bits_per_drawing"]
    if values["valid_halt_delta"] < GENERIC_GUARDS["min_valid_halt_delta"]:
        failures["min_valid_halt_delta"] = values["valid_halt_delta"]
    if values["truncation_rate_increase"] > GENERIC_GUARDS[
            "max_truncation_rate_increase"]:
        failures["max_truncation_rate_increase"] = values[
            "truncation_rate_increase"]
    return {
        "passed": not failures,
        "failures": failures,
        "limits": dict(GENERIC_GUARDS),
        "values": values,
        "validation": validation,
        "generation": generated,
        "generation_n": n,
        "generation_max_new": width,
        "variates_shape": list(variates.shape),
    }


# ---------------------------------------------------------------------------
# paired per-case differences


def _aligned(left: list[dict], right: list[dict], what: str) -> None:
    """Refuse two row sets that are not the same cases in the same order.

    Every downstream difference is positional and every bootstrap draw is a row
    index, so a silent misalignment would pair case 7's soft score with case 8's
    standard one and report the difference as an effect. Fails closed
    (`docs/directions.md` §7 invariant 11).
    """
    if len(left) != len(right):
        raise ValueError(
            f"{what}: {len(left)} rows against {len(right)}; a paired difference "
            "needs the same cases on both sides"
        )
    mismatched = [(a["case_id"], b["case_id"]) for a, b in zip(left, right)
                  if a["case_id"] != b["case_id"]]
    if mismatched:
        raise ValueError(f"{what}: case order differs, first at {mismatched[0]}")


def paired_difference(left: list[dict], right: list[dict], field: str, *,
                      name: str) -> list[dict]:
    """`left[field] - right[field]`, per case, before anything is averaged.

    Contrasts are formed per case and only then summarised. Averaging first and
    differencing after is the same number only when every cell holds exactly the
    same cases with the same weights, and one dropped case breaks that while
    still producing a plausible table.
    """
    _aligned(left, right, name)
    rows = []
    for a, b in zip(left, right):
        rows.append({
            "case_id": a["case_id"],
            "venue": a["venue"],
            "source_index": a["source_index"],
            "donor_index": a["donor_index"],
            "control_donor_index": a["control_donor_index"],
            "target_bytes": a["target_bytes"],
            name: {
                "bits": a[field]["bits"] - b[field]["bits"],
                "bits_per_byte": (a[field]["bits_per_byte"]
                                  - b[field]["bits_per_byte"]),
            },
        })
    return rows


def mode_gain(standard: list[dict], soft: list[dict]) -> list[dict]:
    """`g_i = Delta_i(soft) - Delta_i(standard)`, at identical weights."""
    for rows, expected in ((standard, "standard"), (soft, "soft")):
        wrong = {row["mode"] for row in rows} - {expected}
        if wrong:
            raise ValueError(f"expected {expected} rows, found modes {sorted(wrong)}")
    return paired_difference(soft, standard, "Delta", name="G")


def structure_interaction(relational: list[dict], destroyed: list[dict]
                          ) -> list[dict]:
    """`i_i = g_i(relational) - g_i(relation_destroyed)`, the primary quantity."""
    return paired_difference(relational, destroyed, "G", name="I_structure")


def representation_interaction(bit: list[dict], byte: list[dict]) -> list[dict]:
    """`b_i = i_i(bit) - i_i(byte)`. Exploratory, and labelled so.

    Bit against byte moves vocabulary, entropy, sequence length, compute and
    feedback-event count together, so a non-zero interval identifies a
    representation *package* and never a vocabulary size.
    """
    return paired_difference(bit, byte, "I_structure", name="I_representation")


# ---------------------------------------------------------------------------
# shared-component inference


@dataclass(frozen=True)
class Resample:
    """One frozen set of connected-component draws, reused by every cell.

    Built once and passed everywhere. Two cells resampled from two independently
    seeded generators would differ by their resampling accident as well as by the
    thing under test, and a difference of two such intervals means nothing.
    """

    draws: tuple[tuple[int, ...], ...]
    clusters: int
    largest_cluster: int
    n_rows: int
    seed: int
    reps: int


def component_draws(rows: list[dict], *, seed: int | None = None,
                    reps: int = BOOTSTRAP_REPS) -> Resample:
    """Resample connected source/donor components, exactly as Direction 2 does.

    The procedure is copied deliberately: same graph, same `random.Random`, same
    key order, same with-replacement draw. Direction 3's `G` is a difference of
    two `Delta`s on Direction 2's own cases, so its interval has to be built over
    the same clusters in the same order or the two stages' numbers cannot be read
    against each other.

    A donor reused by two otherwise distinct sources connects both cases;
    resampling `source_index` alone would count dependent observations as
    independent and shrink every interval.
    """
    seed = seed_for("bootstrap") if seed is None else seed
    groups = components(rows)
    keys = sorted(groups)
    rng = random.Random(seed)
    draws = tuple(
        tuple(index
              for key in [keys[rng.randrange(len(keys))] for _ in keys]
              for index in groups[key])
        for _ in range(reps)
    )
    return Resample(
        draws=draws,
        clusters=len(keys),
        largest_cluster=max(len(members) for members in groups.values()),
        n_rows=len(rows),
        seed=seed,
        reps=reps,
    )


def bootstrap(rows: list[dict], field: str, resample: Resample, *,
              unit: str = "bits_per_byte") -> dict:
    """Component-bootstrap mean, 95% interval and two-sided p, on shared draws."""
    if len(rows) != resample.n_rows:
        raise ValueError(
            f"{len(rows)} rows against a resample built for {resample.n_rows}; "
            "shared draws are row indices and mean nothing on a different table"
        )
    values = [float(row[field][unit]) for row in rows]
    replicates = sorted(
        sum(values[index] for index in draw) / len(draw)
        for draw in resample.draws
    )
    reps = len(replicates)
    below = sum(value <= 0.0 for value in replicates) / reps
    above = sum(value >= 0.0 for value in replicates) / reps
    return {
        "mean": sum(values) / len(values),
        "ci95": [replicates[int(0.025 * (reps - 1))],
                 replicates[int(0.975 * (reps - 1))]],
        # Floored at one replicate, so a finite resample never reports an exact
        # zero it cannot support.
        "p_value": min(1.0, max(1.0 / reps, 2.0 * min(below, above))),
        "clusters": resample.clusters,
        "largest_cluster": resample.largest_cluster,
        "cluster_unit": BOOTSTRAP_UNIT,
        "reps": reps,
        "unit": unit,
        "excludes_zero": (replicates[int(0.025 * (reps - 1))] > 0.0
                          or replicates[int(0.975 * (reps - 1))] < 0.0),
    }


def holm(pvalues: dict[str, float], *, alpha: float = ALPHA) -> dict[str, dict]:
    """Holm step-down over the primary family, with adjusted p values.

    Step-down and not Bonferroni: the family is two tests and the difference in
    power is not academic at `n = 64` components. Once a hypothesis fails, every
    larger p in the family fails too -- that monotonicity is what makes the
    procedure valid, and reporting it as `rejected` per key rather than as a
    single verdict keeps which test carried the family visible.
    """
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    total = len(ordered)
    out: dict[str, dict] = {}
    running = 0.0
    still_rejecting = True
    for rank, (name, p) in enumerate(ordered):
        threshold = alpha / (total - rank)
        running = max(running, min(1.0, (total - rank) * p))
        still_rejecting = still_rejecting and p <= threshold
        out[name] = {
            "p_value": p,
            "threshold": threshold,
            "p_adjusted": running,
            "rejected": still_rejecting,
        }
    return out


# ---------------------------------------------------------------------------
# recurrent stability


def _rms(x: torch.Tensor) -> torch.Tensor:
    """Root mean square over the feature axis, one value per row and position."""
    return x.pow(2).mean(-1).sqrt()


def _quantile(values: torch.Tensor, q: float) -> float:
    flat = values.reshape(-1).float()
    if flat.numel() == 0:
        return float("nan")
    return float(flat.sort().values[min(flat.numel() - 1,
                                        int(q * (flat.numel() - 1)))])


#: What `fuse.norm.weight` does in each arm.  The tensor has one name and three
#: jobs, and a diagnostic that reports the number without the job invites the
#: reading `docs/feedback-stability.md` §3a withdrew.
GAIN_ROLE: dict[str, str] = {
    "glu_v1": "fused-output gain",
    SOURCE_FEEDBACK_SCHEMA: "fused-output gain",
    SOURCE_FEEDBACK_SCHEMA_V2: (
        "shared input norm: the gate input and the post-mixin stack input"
    ),
}


@dataclass(frozen=True)
class _Eligible:
    """One candidate row of the stability subset: its index and its length."""

    index: int
    length: int


def stability_subset(programs: Sequence[bytes], codec: Codec, *,
                     size: int = STABILITY_SUBSET_SIZE,
                     deepest_pass: int = max(STABILITY_PASSES)) -> dict:
    """The frozen, model-blind rows the recurrent probe runs on.

    Eligibility is `scored positions > deepest_pass`. The fused prefill is
    triangular -- `state_k[t]` is final for every `t < k` -- so a row no longer
    than the deepest pass has an empty wavefront and a fully converged prefix,
    and every quantity taken at that pass reads the row's *length* rather than
    the channel. The F5 smoke took the first sixteen validation programs, twelve
    of which were fully converged at pass 32; on the real split, where 70% of
    programs exceed 32 bytes, the same checkpoint fails the same clause.

    Selection is a deterministic sample from `seed_for("stability")` over the
    eligible indices in corpus order. No model, no logits, no lengths measured
    after training: `docs/directions.md` §7 invariant 1 rules out anything the
    model produced choosing which rows it is judged on, and this function has no
    parameter through which one could arrive.

    Returns the identities, not only a count. A gate that cannot be re-run on the
    same rows by someone holding the corpus and the protocol is not auditable,
    so the indices, their lengths and a digest over both go into the report.

    Fails closed when fewer than `size` programs are eligible. Shrinking the
    subset would measure something the protocol did not name, and would do it
    most readily on exactly the corpora that are too short to test the clause.
    """
    eligible = [
        _Eligible(index, len(codec.with_bos(program)) - 1)
        for index, program in enumerate(programs)
    ]
    eligible = [row for row in eligible if row.length > deepest_pass]
    if len(eligible) < size:
        raise ValueError(
            f"only {len(eligible)} of {len(programs)} programs outlive pass "
            f"{deepest_pass}, and the frozen subset needs {size}: incomplete. "
            "A shorter subset converges before the deepest pass and reports "
            "sequence length as convergence"
        )
    chosen = sorted(
        random.Random(seed_for("stability")).sample(range(len(eligible)), size)
    )
    rows = [eligible[position] for position in chosen]
    identity = json.dumps(
        [[row.index, row.length] for row in rows], separators=(",", ":")
    ).encode()
    return {
        "seed": seed_for("stability"),
        "deepest_pass": deepest_pass,
        "size": size,
        "eligible": len(eligible),
        "corpus_programs": len(programs),
        "indices": [row.index for row in rows],
        "lengths": [row.length for row in rows],
        "digest": hashlib.sha256(identity).hexdigest(),
    }


@torch.no_grad()
def channel_scales(model) -> dict:
    """Weight-side scales of the feedback channel. Diagnostic; nothing gates.

    Under `glu_source_v2` the reported gain is the **shared input-norm** gain:
    Listing 3's one `input_rmsnorm_1`, applied both to the gate input and to the
    post-mixin stack input. It is therefore not the "fused-output gain" the two
    earlier arms carry under the same parameter name, and `gain_role` says which
    reading applies rather than leaving it to the caller.

    **Three claims this function does not make.** The gain is not the fused
    input's RMS: `RMSNorm` fixes the normalized product's magnitude but not its
    direction, so the output RMS depends on the gain *components* and on where
    the product points, and a vector gain that has gone anisotropic makes its own
    mean only a weight summary. The gain-over-embedding ratio is not the frozen
    `[0.25, 4]x` band, which is measured on scored rows by `stability`. And under
    `glu_source_v2` the gain additionally scales what `W_G` reads, so it no longer
    describes an output scale alone.

    Both sides still move for unrelated reasons and that is worth reading: the
    embedding table takes gradient at every position of every step, while this
    gain takes it only on the `K>=2` batches the frozen schedule allocates 12.75%
    of -- and, since `glu_source_v2` normalizes the plain prefix too, at every
    position of those batches rather than only past the prefix.

    The PAD row is reported separately and excluded from the content statistics.
    It is the tied head's PAD row as well as the PAD input vector, so it is
    trained as a negative class rather than as an embedding and it is not a
    scale the stack ever sees on real content.
    """
    if model.fuse is None:
        raise ValueError("there is no feedback channel to measure; this "
                         "checkpoint is feedback_schema='none'")
    gain = model.fuse.norm.weight.detach().float()
    rows = _rms(model.embed.weight.detach().float())
    content = rows[1:]
    return {
        "feedback_schema": model.fuse.schema,
        "gain_role": GAIN_ROLE[model.fuse.schema],
        "shared_input_norm_gain_mean": float(gain.mean()),
        "shared_input_norm_gain_std": float(gain.std()),
        "shared_input_norm_gain_min": float(gain.min()),
        "shared_input_norm_gain_max": float(gain.max()),
        "embed_row_rms_mean": float(content.mean()),
        "embed_row_rms_p99": _quantile(content, 0.99),
        "pad_row_rms": float(rows[0]),
        "gain_over_embed_rms": float(gain.mean() / content.mean()),
    }


@torch.no_grad()
def stability(model, inputs: torch.Tensor, targets: torch.Tensor, *,
              passes: tuple[int, ...] = STABILITY_PASSES,
              device: str | torch.device = "cpu",
              valid_halt_loss: float | None = None,
              symbols_per_byte: int = 1,
              subset: dict | None = None) -> dict:
    """Iterate the fused prefill and watch whether the recurrence settles.

    Pass 0 is the plain forward. Pass `k` feeds pass `k-1`'s states back in,
    shifted right by one, exactly as `fused` decoding does once. The pilot only
    ever runs *one* fused prefill, so this is deliberately far past the operating
    point: a channel that diverges at 32 iterations is one whose fixed point does
    not exist, and a single application of it is borrowing against that.

    Both directions matter. A carried state that decays toward zero is as broken
    a channel as one that blows up, and only the RMS *band* catches both -- decay
    looks completely healthy in a loss curve.

    **The iteration is exactly triangular, and the report says where that bites.**
    Position 0 is always plain, so its state never moves; position `t` fuses
    `previous[t-1]` and attends only to `0..t`, so by induction `state_k[t]` is
    final for every `t < k`. The recurrence therefore reaches its fixed point in
    exactly `T` passes for *any* weights, converging one position per pass, and a
    quantity averaged over all scored positions at pass `k` is a blend of a
    converged prefix and a first-visit tail. `update_q95` is the frozen gated
    quantity and keeps its definition; `update_q95_wavefront` restricts it to the
    positions where movement is still possible, and the two `*_converged_*` costs
    price the converged prefix against the standard forward on the very same
    positions. Without them a deepest-pass number reads as convergence when it is
    only the subset's sequence lengths (`docs/directions.md` §F7).

    Every reported quantity is taken over scored positions only -- the RMS
    quantiles, `finite` and `max_abs_logit` alike. A padded position's input
    embedding is the PAD row, which is also the tied head's PAD row and so is
    trained hard as a negative class rather than as an input, and its hidden
    state carries no loss at all. `max_abs_logit` over the padded majority
    measures how confidently the model rejects PAD; `finite` over it can fail on
    a state no loss ever touched. Both are gated clauses, so both are masked.

    `symbols_per_byte` is the codec's stride, and the per-byte costs are
    **semantic** bytes: the bit arm spends eight symbols on one bytecode byte, so
    a symbol-normalized cost is not comparable across representations and calling
    it `bits_per_byte` would make the representation package look like a result
    (`docs/directions.md` §7 invariant 6). `bits_per_drawing` is a program-level
    unit and is the same number under either stride.

    `subset` is the frozen row identity `stability_subset` returned, carried into
    the report so the gate says which rows it judged rather than only how many.
    """
    if model.fuse is None:
        raise ValueError("stability is a property of the feedback channel; this "
                         "checkpoint is feedback_schema='none'")
    model.eval()
    inputs, targets = inputs.to(device), targets.to(device)
    scored = (targets != 0)
    if not bool(scored.any()):
        # Fail closed. Now that the quantiles are masked, an all-PAD batch would
        # make every reference `nan`, every band comparison false, and every
        # clause pass with nothing measured (`docs/directions.md` §7 invariant 11).
        raise ValueError("every target position is PAD, so there is nothing to "
                         "measure the feedback channel on: incomplete")
    # Position 0 is held plain, so its fused value is computed and thrown away.
    fused_positions = scored.clone()
    fused_positions[:, 0] = False
    index = torch.arange(inputs.shape[1], device=inputs.device)[None, :]
    programs = max(1, inputs.shape[0])
    embed = model.embed(inputs)
    standard_input_rms = _quantile(_rms(embed)[scored], 0.99)

    if symbols_per_byte < 1:
        raise ValueError(f"symbols_per_byte must be at least 1, got {symbols_per_byte}")

    baseline_logits, states = model(inputs, return_state=True)
    baseline_hidden_rms = _quantile(_rms(states)[scored], 0.99)
    baseline_cost = _bits_per_drawing(baseline_logits, targets, programs)

    logits = baseline_logits
    rows: list[dict] = []
    depth = 0
    previous = states
    update = torch.zeros_like(_rms(states))
    fused_rms = float("nan")
    for wanted in sorted(passes):
        while depth < wanted:
            carried = torch.zeros_like(previous)
            carried[:, 1:] = previous[:, :-1]
            fused = model.fuse(carried, embed)
            fused_rms = _quantile(_rms(fused)[fused_positions], 0.99)
            logits, current = model(inputs, feedback=carried, plain=1,
                                    return_state=True)
            update = (_rms(current - previous)
                      / _rms(previous).clamp(min=1e-12))
            previous = current
            depth += 1
        converged = scored & (index < depth)
        wavefront = scored & (index >= depth)
        rows.append({
            "passes": depth,
            # Scored positions on both sides. A padded state carries no loss and
            # a padded logit row is the tied head reading PAD as a negative
            # class; neither is a property of the feedback channel.
            "finite": bool(torch.isfinite(previous[scored]).all()
                           and torch.isfinite(logits[scored]).all()),
            "fused_input_rms_p99": None if depth == 0 else fused_rms,
            "hidden_rms_p99": _quantile(_rms(previous)[scored], 0.99),
            "max_abs_logit": float(logits[scored].abs().max()),
            "update_q95": 0.0 if depth == 0 else _quantile(update[scored], 0.95),
            "update_q95_wavefront": (
                0.0 if depth == 0 else _quantile(update[wavefront], 0.95)
            ),
            "wavefront_positions": int(wavefront.sum()),
            "bits_per_drawing": _bits_per_drawing(logits, targets, programs),
            "converged_positions": int(converged.sum()),
            "converged_semantic_bytes": int(converged.sum()) / symbols_per_byte,
            "converged_bits_per_byte": _bits_per_byte(
                logits, targets, converged, symbols_per_byte),
            "standard_converged_bits_per_byte": _bits_per_byte(
                baseline_logits, targets, converged, symbols_per_byte),
        })
    report = {
        "schema": STABILITY_REPORT_SCHEMA,
        "programs": int(inputs.shape[0]),
        "scored_positions": int(scored.sum()),
        "padded_positions": int((~scored).sum()),
        "scored_positions_only": True,
        "symbols_per_byte": int(symbols_per_byte),
        "cost_unit": "bits_per_semantic_byte",
        "subset": None if subset is None else dict(subset),
        "standard_input_rms_p99": standard_input_rms,
        "baseline_hidden_rms_p99": baseline_hidden_rms,
        "baseline_bits_per_drawing": baseline_cost,
        "passes": rows,
    }
    if valid_halt_loss is not None:
        report["valid_halt_loss"] = float(valid_halt_loss)
    return report


def _bits_per_drawing(logits: torch.Tensor, targets: torch.Tensor,
                      programs: int) -> float:
    nll = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
        ignore_index=0, reduction="sum",
    )
    return float(nll) / _LOG2 / programs


def _bits_per_byte(logits: torch.Tensor, targets: torch.Tensor,
                   mask: torch.Tensor, symbols_per_byte: int = 1) -> float:
    """Cost over a chosen set of positions, in bits per **semantic** byte.

    Two passes can therefore be priced on the same positions, and the bit arm's
    eight symbols per bytecode byte are aggregated rather than reported as if a
    symbol were a byte. `nan` for an empty set rather than 0.0, which would read
    as free.
    """
    if not bool(mask.any()):
        return float("nan")
    nll = F.cross_entropy(logits[mask], targets[mask], reduction="sum")
    return float(nll) * symbols_per_byte / _LOG2 / int(mask.sum())


def stability_verdict(report: dict) -> dict:
    """Apply the frozen gate. Any failure stops the pilot before outcome scoring.

    Returned as a per-criterion table rather than a boolean: `unstable_feedback`
    is a claim label, and a label without the clause that produced it is not
    auditable.
    """
    rows = {row["passes"]: row for row in report["passes"]}
    deepest = max(rows)
    low, high = STABILITY_RMS_BAND
    input_reference = report["standard_input_rms_p99"]
    hidden_reference = report["baseline_hidden_rms_p99"]
    failures: dict[str, object] = {}

    if not all(row["finite"] for row in rows.values()):
        failures["all_finite"] = [p for p, row in rows.items() if not row["finite"]]
    out_of_band = [
        p for p, row in rows.items()
        if row["fused_input_rms_p99"] is not None
        and not (low * input_reference <= row["fused_input_rms_p99"]
                 <= high * input_reference)
    ]
    if out_of_band:
        failures["fused_input_rms_band"] = out_of_band
    hidden_out = [
        p for p, row in rows.items()
        if not (low * hidden_reference <= row["hidden_rms_p99"]
                <= high * hidden_reference)
    ]
    if hidden_out:
        failures["hidden_rms_band"] = hidden_out
    loud = [p for p, row in rows.items()
            if row["max_abs_logit"] >= STABILITY_GATE["max_abs_logit"]]
    if loud:
        failures["max_abs_logit"] = loud

    increase = (rows[deepest]["bits_per_drawing"]
                - report["baseline_bits_per_drawing"])
    if increase > STABILITY_GATE["max_pass32_validation_increase_bits_per_drawing"]:
        failures["validation_increase_bits_per_drawing"] = increase
    # **The wavefront, not the blended quantile.** `update_q95` averages exact
    # zeros from the converged prefix against a live tail, so on a subset longer
    # than the deepest pass it reads sequence length: at 32 passes a 16-position
    # row contributes only zeros. The frozen bound and the frozen reference pass
    # are v0's; what F5b repairs is which positions they are measured on.
    #
    # `update_q95_not_settling` is withdrawn rather than left failing. The
    # iteration is triangular, so "still receding at the deepest pass" has no
    # reading that separates a bad channel from a long sequence
    # (`contract.WITHDRAWN_CLAUSE_REASONS`). Its wavefront twin does: the
    # positions that can still move are the ones a contraction has to be
    # contracting.
    wavefront = rows[deepest]["update_q95_wavefront"]
    if wavefront > STABILITY_GATE["max_update_q95"]:
        failures["update_q95_wavefront"] = wavefront
    reference_pass = STABILITY_GATE["update_q95_monotone_from_pass"]
    if reference_pass in rows and deepest != reference_pass and (
            wavefront > rows[reference_pass]["update_q95_wavefront"]):
        failures["update_q95_wavefront_not_settling"] = [
            rows[reference_pass]["update_q95_wavefront"], wavefront
        ]
    if "valid_halt_loss" not in report:
        # The generic guard is an explicit F7 input, not an optional annotation.
        # A caller that omits it has not applied the frozen termination clause.
        failures["valid_halt_loss"] = "missing"
    else:
        loss = report["valid_halt_loss"]
        if (not isinstance(loss, (int, float))
                or not math.isfinite(loss)
                or loss > STABILITY_GATE["max_valid_halt_loss"]):
            failures["valid_halt_loss"] = loss
    return {
        "passed": not failures,
        "label": "stable" if not failures else "unstable_feedback",
        "deepest_pass": deepest,
        "validation_increase_bits_per_drawing": increase,
        # Named in the verdict rather than only in the protocol: a reader has to
        # be able to see, from the artifact alone, which clause was dropped and
        # why the remaining list is shorter than v0's.
        "withdrawn_clauses": dict(WITHDRAWN_CLAUSE_REASONS),
        "failures": failures,
    }


#: The stability clauses an *engineering* run is entitled to gate on: the ones
#: that say something broke, not the ones that say something has not converged.
#:
#: `stability_verdict` is the F7 gate and F7 runs on eight final-step checkpoints;
#: its quality clauses ask a converged model to hold its likelihood under a
#: 32-deep iterate of a channel the objective trains one or two levels of. A
#: 400-step throwaway cannot meet that, and -- by §7 invariant 14 -- it also has
#: no standing to argue the threshold is wrong. So the smoke gates the clauses
#: whose meaning does not depend on the training budget and *reports* the rest
#: verbatim. It never suppresses them: `deferred_clauses` names every clause the
#: smoke declined to judge, so a reader sees what was set aside rather than a
#: shorter list of failures.
SMOKE_STABILITY_CLAUSES: tuple[str, ...] = (
    "all_finite",
    "max_abs_logit",
    "fused_input_rms_band",
    "hidden_rms_band",
)


def smoke_stability(verdict: dict) -> dict:
    """Split the F7 verdict into what an engineering smoke may gate on and what
    it must only report. Never widens the gate: the clause names are a subset."""
    gated = {name: value for name, value in verdict["failures"].items()
             if name in SMOKE_STABILITY_CLAUSES}
    deferred = {name: value for name, value in verdict["failures"].items()
                if name not in SMOKE_STABILITY_CLAUSES}
    return {
        "gated_clauses": list(SMOKE_STABILITY_CLAUSES),
        "passed": not gated,
        "failures": gated,
        "deferred_clauses": deferred,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# fail-closed report checks


#: How an artifact was produced. The F5 smoke exercises every code path end to
#: end and is worth exactly nothing as evidence -- one cell, one seed, a budget
#: chosen to be short. Marking it in the artifact rather than in a filename is
#: what lets the scientific gate reject it by schema instead of by convention.
ENGINEERING = "engineering"
#: An F5b qualification cell: full budget and full scale, separate development
#: seeds and data, and no relation outcome anywhere in its chain.  Distinct from
#: `SCIENTIFIC` in both directions.  A development cell cannot enter an outcome
#: report -- it was never part of the estimation matrix -- and a scientific cell
#: cannot qualify a schedule, because choosing a configuration by looking at
#: pilot cells would make the pilot its own selection data
#: (`docs/directions.md` §7 invariant 13).
DEVELOPMENT = "development"
SCIENTIFIC = "scientific"
PROVENANCE = (ENGINEERING, DEVELOPMENT, SCIENTIFIC)


def refuse_engineering(report: dict, what: str) -> None:
    """Refuse an artifact that is not marked as scientific.

    Composed by the gate rather than folded into `validate_score_report`, because
    protocol v0 froze that function's required-field list and a checker stricter
    than its own freeze is the drift this project keeps paying for. v1 folds
    `provenance` into the list; until then the check is separate and named.

    Absent is refused as firmly as `engineering`: missing provenance is
    `incomplete`, never a pass (`docs/directions.md` §7 invariant 11).
    """
    provenance = report.get("provenance")
    if provenance != SCIENTIFIC:
        raise ValueError(
            f"{what} has provenance {provenance!r}, not {SCIENTIFIC!r}: one "
            "engineering smoke cell has zero scientific decision value: incomplete"
        )


def _require(report: dict, fields: tuple[str, ...], what: str) -> None:
    if not isinstance(report, dict):
        # ValueError, not TypeError: every fail-closed path in this stage raises
        # the same exception carrying the word `incomplete`, because the caller
        # is turning it into a claim label rather than debugging a call site.
        raise ValueError(f"{what} is not a report: incomplete")  # noqa: TRY004
    missing = [field for field in fields if field not in report]
    if missing:
        raise ValueError(f"{what} is missing {missing}: incomplete")
    if report.get("schema") != REPORT_SCHEMA:
        raise ValueError(
            f"{what} is schema {report.get('schema')!r}, expected {REPORT_SCHEMA}: "
            "incomplete"
        )
    if report.get("status") != "complete":
        raise ValueError(
            f"{what} has status {report.get('status')!r}, not 'complete': incomplete"
        )


def validate_score_report(report: dict) -> None:
    """Refuse a teacher-forced report that cannot support the claim it carries.

    A missing control, a missing raw-NLL column or a report that never finished
    yields `incomplete` -- never a zero and never a pass. Direction 2's retracted
    schema-1 report is why that is a rule and not a preference.
    """
    _require(report, REQUIRED_SCORE_FIELDS, "feedback score report")
    if report["mode"] not in EVALUATION_MODES:
        raise ValueError(
            f"score report mode {report['mode']!r} is not one of "
            f"{list(EVALUATION_MODES)}: incomplete"
        )
    raw = report["raw_symbol_nll_bits"]
    if not isinstance(raw, dict) or set(raw) != set(REQUEST_BLOCKS):
        raise ValueError(
            "feedback score report must persist raw symbol NLLs for all three "
            "four-way blocks (all 12 requests): incomplete"
        )
    for block in REQUEST_BLOCKS:
        values = raw[block]
        if not isinstance(values, dict) or set(values) != set(FOUR_WAY):
            raise ValueError(
                f"feedback score report raw NLL block {block!r} is incomplete: "
                "all four requests are required"
            )
        for name in FOUR_WAY:
            symbols = values[name]
            if (not isinstance(symbols, (list, tuple)) or not symbols
                    or any(not math.isfinite(float(value)) for value in symbols)):
                raise ValueError(
                    f"feedback score report raw NLL {block}/{name} is not a "
                    "finite non-empty symbol array: incomplete"
                )
    if not report["sequential_standard_matches_full_forward"]:
        raise ValueError(
            "the sequential scorer was never checked against the full-forward "
            "scorer, so its standard arm is not known to be Direction 2's "
            "estimand: incomplete"
        )


def validate_generation_report(report: dict) -> None:
    _require(report, REQUIRED_GENERATION_FIELDS, "feedback generation report")
    if report["mode"] not in EVALUATION_MODES:
        raise ValueError(
            f"generation report mode {report['mode']!r} is not one of "
            f"{list(EVALUATION_MODES)}: incomplete"
        )


# ---------------------------------------------------------------------------
# equivalence


def sequential_matches_full_forward(
    model, requests: list[tuple[bytes, bytes]], codec: Codec, *,
    device: str | torch.device = "cpu", max_len: int = 2048,
    batch_size: int = 32, tolerance: float = 1e-3,
) -> dict:
    """Check the sequential standard scorer against `score_spans`, symbol by symbol.

    **Not bit-for-bit, and it cannot be.** One path attends through a preallocated
    KV cache a position at a time and the other runs a single batched forward over
    the whole sequence; `scaled_dot_product_attention` reduces over different
    shapes in the two, so the last bits differ. What matters is that the
    difference is orders of magnitude below the effect: the frozen practical floor
    is `0.02` bits per target byte, and the default tolerance here is `1e-3` bits
    on a whole target.

    Returned as a report rather than an assertion, because `validate_score_report`
    refuses a score report that does not carry the answer.
    """
    sequential = score_spans_sequential(model, requests, codec, mode="standard",
                                        device=device, max_len=max_len,
                                        batch_size=batch_size)
    batched = score_spans(model, requests, codec, device=device, max_len=max_len,
                          batch_size=batch_size)
    worst_symbol = 0.0
    worst_total = 0.0
    for left, right in zip(sequential, batched):
        worst_total = max(worst_total, abs(left["bits"] - right["bits"]))
        worst_symbol = max(
            worst_symbol,
            float(np.abs(np.array(left["per_byte"])
                         - np.array(right["per_byte"])).max()),
        )
    return {
        "requests": len(requests),
        "max_abs_target_bits_delta": worst_total,
        "max_abs_byte_bits_delta": worst_symbol,
        "tolerance": tolerance,
        "matches": worst_total <= tolerance and worst_symbol <= tolerance,
    }


__all__ = [
    "ENGINEERING",
    "PROVENANCE",
    "REQUEST_BLOCKS",
    "SCIENTIFIC",
    "Resample",
    "bootstrap",
    "case_requests",
    "component_draws",
    "generic_guards",
    "holm",
    "mode_gain",
    "paired_difference",
    "refuse_engineering",
    "representation_interaction",
    "score_cases_sequential",
    "score_spans_sequential",
    "sequential_matches_full_forward",
    "stability",
    "stability_verdict",
    "structure_interaction",
    "validate_generation_report",
    "validate_score_report",
    "validation_costs",
]
