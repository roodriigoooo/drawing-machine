"""Illegal mass: what a legality mask would remove, before any mask is applied.

`dm/eval/attribution.py` measured the bits an arm spends on symbols that cannot
occur at a position **by position class** -- opcode versus each operand `Kind` --
and found 0.0008-0.0085 bits/drawing over four alphabets and three corpora. That
closed the architectural-prior question: a network told "an instruction is 1-4
bytes" has almost nothing to recover.

It could not answer the next question, because a position class is *static*.
"`ENDREP` is legal here" and "a top-level `HALT` is legal here" are facts about
the open scopes, and no position class carries them. This module is that
measurement: the mass a model puts on symbols the **dynamic** state forbids,
decomposed by the frozen, mutually exclusive rules of
[`docs/state-freeze.md`](../../docs/state-freeze.md) §5 and stratified by state.

Two quantities, and only these two:

- `q` -- the illegal mass `P(symbol not in support)` at temperature 1, before
  `forbid` and before `top_k`; and
- `-log2(1-q)` -- the bits renormalising onto the legal set would recover, which
  is the honest headline because it is additive over positions and reads in the
  unit this project argues in.

**`-log2(q)` is not reported.** It grows as illegal mass shrinks, so it reads
backwards: a model that is nearly perfect scores a large number. The freeze says
so and this module has no code path that computes it.

`-log2(1-q)` is computed as `-log2(legal mass)` rather than from `q`, which is
not a style choice: under the `bit` alphabet a raw prefix can drive `q` to within
float epsilon of 1, where `1 - q` has no significant digits left and the legal
mass still has all of them.

The decomposition closes or the report says so. The per-rule masses partition the
union by construction (`dm.isa.state.Support`), so they must sum to it; `residual`
is printed for the same reason `dm/eval/attribution.py` prints its cut spread --
a decomposition that does not close is a decomposition of something else.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from ..data.dataset import ProgramDataset, collate
from ..isa.codec import PAD, Codec
from ..isa.state import (
    CANONICAL,
    MASK_RULES,
    VM_SAFE,
    GrammarState,
    LanguagePolicy,
    Reason,
    StateKey,
    SupportTable,
    SymbolCursor,
    trace,
)

LOG2 = math.log(2.0)

#: Three buckets, and the split is the whole point of the file.
#:
#: **`STATIC_RULES` are what a *position class* already implies**, which is exactly
#: what `dm/eval/attribution.py` priced at 0.0008-0.0085 bits/drawing.
#: `OPERAND_DOMAIN` belongs here and not with the dynamic rules: "this position is
#: an `XF` operand, so only 0-7 are legal" is a fact about the instruction's own
#: layout, and attribution's `xf` row measures it. Filing it as dynamic overstated
#: the dynamic share by ~2x on the structured venue, where the `XF` operand is the
#: single most expensive position in the corpus.
#:
#: **`POLICY_RULES` need the corpus and not the prefix.** "This checkpoint's
#: language has no `CURVE`" is a property of the allowlist, still position-class
#: only, and still not something attribution could see -- it read the ISA's grid,
#: not the corpus's.
#:
#: **`DYNAMIC_RULES` need the scope stack.** A closer with nothing to close, an
#: opener at its depth bound, a halt inside an open scope: these are the facts no
#: position class carries, and the ones direction 1 exists to price.
STATIC_RULES: tuple[Reason, ...] = (Reason.CONTROL, Reason.STATIC_GRID,
                                    Reason.REPRESENTATION, Reason.OPERAND_DOMAIN)
POLICY_RULES: tuple[Reason, ...] = (Reason.OPCODE_POLICY,)
DYNAMIC_RULES: tuple[Reason, ...] = (Reason.SCOPE, Reason.DEPTH, Reason.HALT)
assert set(STATIC_RULES) | set(POLICY_RULES) | set(DYNAMIC_RULES) == set(MASK_RULES)

#: `(grammar key, codec cursor)` -- everything that decides a position's support.
SupportKey = tuple[StateKey, tuple[int, int]]


def position_keys(program: bytes, policy: LanguagePolicy,
                  stride: int) -> list[SupportKey]:
    """One support key per *symbol* position of `program`, in target order.

    `collate` shifts targets left by one, so target position `t` is the symbol at
    index `t` of the encoded program and is predicted from the state **before**
    it: byte `t // stride`, with `t % stride` symbols of that byte already
    committed. The cursor prefix is the top bits of the byte, which is what makes
    a bit-level mask a set of prefixes of legal complete bytes.

    A function of the prefix alone, asserted by `tests/test_state.py`: no target
    symbol is consulted while producing its own key.
    """
    keys: list[SupportKey] = []
    state = GrammarState(policy)
    for byte in program:
        key = state.key()
        for filled in range(stride):
            prefix = byte >> (8 - filled) if filled else 0
            keys.append((key, (filled, prefix)))
        state.step(byte)
    return keys


#: Selection columns: final legal support, exclusive removed mass by rule, then
#: the support remaining after each rule. The latter makes stagewise
#: renormalisation costs exact rather than a proportional attribution of the
#: union cost.
_N_RULES = len(MASK_RULES)


def _selection(table: SupportTable, key: SupportKey) -> np.ndarray | None:
    """Selection matrices for one support key, memoised.

    One matrix multiply then answers every column at once. The alternative -- an
    `index_select` and a `logsumexp` per rule per position group -- is nine small
    tensor ops and nine device syncs per group, which at one call per decoded
    symbol was the whole cost of a masked draw. `None` where the key has no legal
    symbol at all.
    """
    cache = table.matrices
    hit = cache.get(key)
    if hit is None:
        support = table.of_key(key[0], _cursor(table.alphabet.width, key[1]))
        if support.empty:
            cache[key] = False
            return None
        matrix = np.zeros(
            (table.vocab_size, 1 + 2 * _N_RULES), dtype=np.float64
        )
        matrix[support.legal, 0] = 1.0
        for reason, symbols in support.removed:
            matrix[symbols, 1 + MASK_RULES.index(reason)] = 1.0
        alive = np.ones(table.vocab_size, dtype=bool)
        for index, reason in enumerate(MASK_RULES):
            for removed_reason, symbols in support.removed:
                if removed_reason is reason:
                    alive[symbols] = False
                    break
            matrix[alive, 1 + _N_RULES + index] = 1.0
        cache[key] = matrix
        return matrix
    return None if hit is False else hit


def mass_by_rule(probs: np.ndarray, table: SupportTable,
                 keys: Sequence[SupportKey]) -> dict:
    """Illegal mass for `P` scored positions, decomposed by rule.

    `probs` is a `(P, vocab)` float64 probability array aligned with `keys` --
    probabilities rather than log-probabilities, because every quantity here is a
    *sum* over a set of symbols and a sum of positives has no cancellation to
    protect against. What it does need is float64: the legal mass is a number near
    1 whose distance from 1 is the signal, and in float32 that distance is noise
    below ~1e-7.

    Positions are grouped by support key: the corpora visit a few dozen distinct
    states, so this is a few dozen matrix multiplies per batch rather than one
    reduction per position per rule.
    """
    n = len(keys)
    legal = np.zeros(n, dtype=np.float64)
    per_rule = {reason: np.zeros(n, dtype=np.float64) for reason in MASK_RULES}
    stage_cost = {reason: np.zeros(n, dtype=np.float64) for reason in MASK_RULES}
    empty = np.zeros(n, dtype=bool)

    groups: dict[SupportKey, list[int]] = {}
    for index, key in enumerate(keys):
        groups.setdefault(key, []).append(index)

    for key, rows in groups.items():
        matrix = _selection(table, key)
        if matrix is None:
            # No legal symbol at all. Under a mask this is an instrument error and
            # `StateMonitor` raises; here it is a *reading* -- the position is
            # unreachable in the canonical language, which happens at a prefix that
            # already left it -- so it is counted and excluded from the
            # renormalisation total rather than contributing an infinity.
            empty[rows] = True
            continue
        masses = probs[rows] @ matrix
        legal[rows] = masses[:, 0]
        for column, reason in enumerate(MASK_RULES, start=1):
            per_rule[reason][rows] = masses[:, column]
        stages = masses[:, 1 + _N_RULES :]
        with np.errstate(divide="ignore", invalid="ignore"):
            cumulative = -np.log2(stages)
        previous = np.concatenate(
            [np.zeros((len(rows), 1), dtype=np.float64), cumulative[:, :-1]],
            axis=1,
        )
        costs = np.maximum(0.0, cumulative - previous)
        for column, reason in enumerate(MASK_RULES):
            stage_cost[reason][rows] = costs[:, column]
    with np.errstate(divide="ignore"):
        renorm = -np.log2(legal)
    return {
        "legal_mass": np.where(empty, np.nan, legal),
        # -log2(1-q) from the legal mass itself, never from `1 - q`: under the bit
        # alphabet a raw prefix can drive `q` to within float epsilon of 1, where
        # `1 - q` has no significant digits left and the legal mass still has all
        # of them.
        "renorm_bits": np.where(empty, 0.0, renorm),
        "q": np.where(empty, np.nan, 1.0 - legal),
        "by_rule": per_rule,
        "stage_renorm_bits": stage_cost,
        "empty": empty,
    }


def _cursor(width: int, state: tuple[int, int]) -> SymbolCursor:
    cursor = SymbolCursor(width)
    cursor.filled, cursor.prefix = state
    return cursor


class _Totals:
    """Running sums for one mask level, per rule and per state stratum."""

    def __init__(self, n_programs: int) -> None:
        self.positions = 0
        self.empty = 0
        self.renorm = 0.0
        self.per_program = np.zeros(n_programs, dtype=np.float64)
        self.q: list[float] = []
        self.renorm_values: list[float] = []
        self.q_sum = 0.0
        self.rule_mass = {reason: 0.0 for reason in MASK_RULES}
        self.stage_renorm = {reason: 0.0 for reason in MASK_RULES}
        self.rule_positions = {reason: 0 for reason in MASK_RULES}
        self.strata: dict[str, dict] = {}
        #: The worst individual positions, kept because the aggregate hides the
        #: shape of the answer: measured on the structured venue, `q` averages
        #: 5e-5 and reaches 0.95 somewhere. "A tiny mean" and "a tiny maximum"
        #: are different findings and the gate needs the second.
        self.worst: list[tuple[float, int, int, str, str, float]] = []

    def add_rows(self, rows: np.ndarray, reading: dict, keys: Sequence[SupportKey],
                 strata: Sequence[str], offsets: Sequence[int],
                 keep_worst: int = 12) -> None:
        """One decode step: every live row's position at once.

        The teacher-forced path calls `add` once per program because a program is
        the unit its totals are per; a decode has no programs, so this attributes
        each position to its own row and records the byte offset the row was at.
        """
        bits = reading["renorm_bits"]
        self.positions += len(rows)
        self.empty += int(reading["empty"].sum())
        self.renorm += float(bits.sum())
        np.add.at(self.per_program, rows, bits)
        self.q.extend(float(v) for v in reading["q"] if not math.isnan(v))
        self.renorm_values.extend(float(v) for v in bits if np.isfinite(v))
        self.q_sum += float(np.nansum(reading["q"]))
        for reason in MASK_RULES:
            mass = reading["by_rule"][reason]
            self.rule_mass[reason] += float(mass.sum())
            self.rule_positions[reason] += int((mass > 0).sum())
            self.stage_renorm[reason] += float(
                reading["stage_renorm_bits"][reason].sum()
            )
        for position in np.argsort(bits)[::-1][:keep_worst]:
            if bits[position] <= 0.0:
                break
            dominant = max(MASK_RULES, key=lambda r: reading["by_rule"][r][position])
            self.worst.append((float(bits[position]), int(rows[position]),
                               int(offsets[position]), strata[int(position)],
                               dominant.value, float(reading["q"][position])))
        self.worst = sorted(self.worst, reverse=True)[:keep_worst]
        for stratum in set(strata):
            mask = np.array([s == stratum for s in strata])
            entry = self.strata.setdefault(
                stratum,
                {"positions": 0, "renorm_bits": 0.0,
                 "rule_mass": {r: 0.0 for r in MASK_RULES},
                 "stage_renorm": {r: 0.0 for r in MASK_RULES}},
            )
            entry["positions"] += int(mask.sum())
            entry["renorm_bits"] += float(bits[mask].sum())
            for reason in MASK_RULES:
                entry["rule_mass"][reason] += float(reading["by_rule"][reason][mask].sum())
                entry["stage_renorm"][reason] += float(
                    reading["stage_renorm_bits"][reason][mask].sum()
                )

    def add(self, index: int, reading: dict, keys: Sequence[SupportKey],
            strata: Sequence[str], stride: int = 1, keep_worst: int = 12) -> None:
        self.positions += len(keys)
        self.empty += int(reading["empty"].sum())
        self.renorm += float(reading["renorm_bits"].sum())
        self.per_program[index] += float(reading["renorm_bits"].sum())
        self.q.extend(float(v) for v in reading["q"] if not math.isnan(v))
        self.renorm_values.extend(
            float(v) for v in reading["renorm_bits"] if np.isfinite(v)
        )
        self.q_sum += float(np.nansum(reading["q"]))
        for reason in MASK_RULES:
            mass = reading["by_rule"][reason]
            self.rule_mass[reason] += float(mass.sum())
            self.rule_positions[reason] += int((mass > 0).sum())
            self.stage_renorm[reason] += float(
                reading["stage_renorm_bits"][reason].sum()
            )
        bits = reading["renorm_bits"]
        for position in np.argsort(bits)[::-1][:keep_worst]:
            if bits[position] <= 0.0:
                break
            dominant = max(MASK_RULES,
                           key=lambda r: reading["by_rule"][r][position])
            self.worst.append((float(bits[position]), index,
                               int(position) // max(1, stride),
                               strata[int(position)], dominant.value,
                               float(reading["q"][position])))
        self.worst = sorted(self.worst, reverse=True)[:keep_worst]

        for stratum in set(strata):
            rows = np.array([s == stratum for s in strata])
            entry = self.strata.setdefault(
                stratum,
                {"positions": 0, "renorm_bits": 0.0,
                 "rule_mass": {r: 0.0 for r in MASK_RULES},
                 "stage_renorm": {r: 0.0 for r in MASK_RULES}},
            )
            entry["positions"] += int(rows.sum())
            entry["renorm_bits"] += float(reading["renorm_bits"][rows].sum())
            for reason in MASK_RULES:
                entry["rule_mass"][reason] += float(
                    reading["by_rule"][reason][rows].sum()
                )
                entry["stage_renorm"][reason] += float(
                    reading["stage_renorm_bits"][reason][rows].sum()
                )

    def as_dict(self, n_programs: int, level: str) -> dict:
        n = max(1, n_programs)
        positions = max(1, self.positions)
        union = sum(self.rule_mass.values())
        q_quantiles = (
            np.quantile(self.q, [0.5, 0.9, 0.95, 0.99, 1.0])
            if self.q else [float("nan")] * 5
        )
        bit_quantiles = (
            np.quantile(self.renorm_values, [0.5, 0.9, 0.95, 0.99, 1.0])
            if self.renorm_values else [float("nan")] * 5
        )
        exact_buckets = {
            "static": sum(self.stage_renorm[r] for r in STATIC_RULES) / n,
            "policy": sum(self.stage_renorm[r] for r in POLICY_RULES) / n,
            "dynamic": sum(self.stage_renorm[r] for r in DYNAMIC_RULES) / n,
        }
        return {
            "level": level,
            "positions": self.positions,
            "empty_support_positions": self.empty,
            # The headline: bits a legality mask could recover per drawing, in the
            # unit `dm/eval/attribution.py` reports its 0.0008-0.0085 in.
            "renorm_bits_per_drawing": self.renorm / n,
            # Stage costs telescope exactly to the union cost. These are not
            # proportional mass attributions: the buckets now have an identity.
            "static_bits_per_drawing": exact_buckets["static"],
            "policy_bits_per_drawing": exact_buckets["policy"],
            "dynamic_bits_per_drawing": exact_buckets["dynamic"],
            "stage_renorm_bits_per_drawing": {
                reason.value: self.stage_renorm[reason] / n
                for reason in MASK_RULES
            },
            "static_mass_share": _bucket_share(self, STATIC_RULES),
            "policy_mass_share": _bucket_share(self, POLICY_RULES),
            "dynamic_mass_share": _bucket_share(self, DYNAMIC_RULES),
            "q_mean": float(np.mean(self.q)) if self.q else float("nan"),
            # The protocol asks for sample SD.  One observation has no
            # estimable spread; reporting 0.0 is the explicit finite convention
            # used by the JSON report rather than silently switching to the
            # population denominator.
            "q_std": (float(np.std(self.q, ddof=1))
                      if len(self.q) > 1 else 0.0),
            "q_p50": float(q_quantiles[0]),
            "q_p90": float(q_quantiles[1]),
            "q_p95": float(q_quantiles[2]),
            "q_p99": float(q_quantiles[3]),
            "q_max": float(q_quantiles[4]),
            "renorm_bits_std": (
                float(np.std(self.renorm_values, ddof=1))
                if len(self.renorm_values) > 1 else 0.0
            ),
            "renorm_bits_p50": float(bit_quantiles[0]),
            "renorm_bits_p90": float(bit_quantiles[1]),
            "renorm_bits_p95": float(bit_quantiles[2]),
            "renorm_bits_p99": float(bit_quantiles[3]),
            "renorm_bits_max": float(bit_quantiles[4]),
            "union_mass_per_position": union / positions,
            # The reconciliation, and it is between two different computations:
            # `q` comes from the mass on the *legal* set, the rules sum the mass
            # they each removed. They must agree, and a decomposition that does
            # not close is a decomposition of something else
            # (`dm/eval/attribution.py`'s cut spread, one level down).
            "union_residual_per_position": abs(self.q_sum - union) / positions,
            "by_rule": {
                reason.value: {
                    "mass_per_position": self.rule_mass[reason] / positions,
                    "share_of_union": (self.rule_mass[reason] / union) if union else 0.0,
                    "positions_with_mass": self.rule_positions[reason],
                    "stage_renorm_bits_per_drawing": (
                        self.stage_renorm[reason] / n
                    ),
                }
                for reason in MASK_RULES
            },
            "by_stratum": {
                name: {
                    "positions": entry["positions"],
                    "renorm_bits_per_drawing": entry["renorm_bits"] / n,
                    "mass_per_position": {
                        reason.value: entry["rule_mass"][reason]
                        / max(1, entry["positions"])
                        for reason in MASK_RULES
                    },
                    "stage_renorm_bits_per_drawing": {
                        reason.value: entry["stage_renorm"][reason] / n
                        for reason in MASK_RULES
                    },
                }
                for name, entry in sorted(self.strata.items())
            },
            "per_program_renorm_bits": [float(b) for b in self.per_program],
            "worst_positions": [
                {"renorm_bits": round(bits, 6), "program": program, "byte": offset,
                 "stratum": stratum, "dominant_rule": rule, "q": round(q, 6)}
                for bits, program, offset, stratum, rule, q in self.worst
            ],
        }


def _bucket_share(totals: _Totals, bucket: tuple[Reason, ...]) -> float:
    """The share of the union's illegal *mass* one bucket carries. Exact."""
    union = sum(totals.rule_mass.values())
    if not union:
        return 0.0
    return sum(totals.rule_mass[r] for r in bucket) / union


def _bucket_bits(totals: _Totals, n: int, bucket: tuple[Reason, ...]) -> float:
    """Recoverable bits attributed to one bucket, by mass share.

    **An attribution, not an identity, and it is named so nobody differences it
    against an exact quantity.** `-log2(1-q)` is a function of the *union*: bits
    are not additive over a partition of probability mass, so the eight rules
    cannot each own a slice of them exactly. What is exact is each rule's share of
    the illegal mass (`by_rule.share_of_union`), and this multiplies the
    recoverable bits by the bucket's share of it. The comparison it licenses is an
    order-of-magnitude one against `dm/eval/attribution.py`'s 0.0008-0.0085
    bits/drawing -- not a reconciliation to the last digit.

    The per-*stratum* bits are exact by contrast, because strata partition
    positions rather than mass. That is why the `operand_xf` stratum reconciles
    with attribution's `xf` row to six decimals and this column does not.
    """
    return totals.renorm / max(1, n) * _bucket_share(totals, bucket)


@torch.no_grad()
def dynamic_support(
    model,
    programs: list[bytes],
    codec: Codec,
    policy: LanguagePolicy,
    device: str | torch.device = "cpu",
    max_len: int = 2048,
    batch_size: int = 8,
) -> dict:
    """One teacher-forced pass: raw bits, illegal mass and its decomposition.

    The BOS shift is `dm.eval.recovery._symbol_bits`'s and
    `dm.eval.attribution._score`'s, restated rather than imported because this
    pass needs the whole distribution and not only the target's own cost -- and
    restated *identically*, because one position of drift credits every byte's
    bits to its neighbour.

    Both mask levels come out of one pass. They are nested (`canonical` is a
    subset of `vm_safe`), so scoring them separately would run the model twice to
    read one set of logits two ways.
    """
    tables = {level: SupportTable(codec, policy, level) for level in (VM_SAFE, CANONICAL)}
    totals = {level: _Totals(len(programs)) for level in tables}
    keys_of = {i: position_keys(programs[i], policy, codec.stride)
               for i in range(len(programs))}
    strata_of = {i: [key[0].stratum for key in keys]
                 for i, keys in keys_of.items()}
    verdicts = [trace(programs[i], policy) for i in range(len(programs))]

    dataset = ProgramDataset(programs, codec, max_len)
    raw_bits = np.zeros(len(programs), dtype=np.float64)
    scored_bytes = 0
    model.eval()
    for start in range(0, len(dataset), batch_size):
        batch = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
        index, inputs, targets = collate(batch)
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs).float()
        logprobs = F.log_softmax(logits, dim=-1)
        nll = F.cross_entropy(
            logprobs.reshape(-1, codec.vocab_size), targets.reshape(-1),
            ignore_index=PAD, reduction="none",
        ).view(targets.shape).cpu().numpy() / LOG2
        # Off-device once, then perform softmax in float64.  Casting an already
        # normalised float32 distribution cannot restore illegal mass rounded
        # out when the legal mass was within one float32 epsilon of one.  MPS has
        # no float64 (`docs/traps.md`), hence the host computation.
        probs = torch.softmax(
            logits.detach().cpu().to(torch.float64), dim=-1
        ).numpy()
        del logits, logprobs
        for row, program_index in enumerate(index.tolist()):
            keys = keys_of[program_index]
            width = min(len(keys), targets.shape[1])
            # PAD targets are unscored, exactly as in every other instrument here.
            live = (targets[row, :width] != PAD).cpu().numpy()
            width = int(live.sum())
            if not width:
                continue
            raw_bits[program_index] += float(nll[row, :width].sum())
            scored_bytes += width // codec.stride
            block = probs[row, :width]
            for level, table in tables.items():
                reading = mass_by_rule(block, table, keys[:width])
                totals[level].add(program_index, reading, keys[:width],
                                  strata_of[program_index][:width],
                                  stride=codec.stride)

    n = max(1, len(programs))
    out = {
        "n": len(programs),
        "codec": codec.name,
        "stride": codec.stride,
        "policy": policy.as_dict(),
        "policy_digest": policy.digest,
        "bits_per_drawing": float(raw_bits.sum()) / n,
        "bytes_per_drawing": scored_bytes / n,
        "canonical_corpus": sum(v.canonical for v in verdicts) / n,
        "levels": {level: totals[level].as_dict(len(programs), level)
                   for level in totals},
    }
    for level in totals:
        cell = out["levels"][level]
        # A constrained likelihood is a *diagnostic* and is labelled as one. It
        # must never stand in for raw bits/drawing: renormalising onto a legal set
        # is something a decoder does, and the headline likelihood column is the
        # model's own (`docs/state.md`).
        cell["constrained_bits_per_drawing_diagnostic"] = (
            out["bits_per_drawing"] - cell["renorm_bits_per_drawing"]
        )
    return out


class GenerationMass:
    """The same accounting over a *decode*, fed by `generate`'s raw logits.

    `DrawingLM.generate(on_logits=...)` hands over the logits **before**
    temperature, before the mask and before `top_k`, at the step where the
    monitor still holds the prefix state. So the illegal mass measured here is the
    model's own, on the prefix the model itself produced -- which is the quantity
    the teacher-forced audit cannot reach, because exposure bias is exactly what
    differs between the two.

    Scored against both the VM-safe and canonical supports by default. The
    supports are nested, so both readings come from one logits pass and the
    report can distinguish executable mass from the stricter canonical residue.

    Rows that have stopped are skipped: their logits are discarded by the sampler
    (it forces `PAD`), so counting them would average the model's behaviour with
    the sampler's bookkeeping.
    """

    def __init__(self, monitor,
                 level: str | Sequence[str] = (VM_SAFE, CANONICAL)) -> None:
        self.monitor = monitor
        levels = (level,) if isinstance(level, str) else tuple(level)
        if not levels or any(item not in (VM_SAFE, CANONICAL) for item in levels):
            raise ValueError(f"unsupported generation-mass levels: {levels!r}")
        self.levels = tuple(dict.fromkeys(levels))
        self.tables = {
            item: SupportTable(monitor.codec, monitor.policy, item)
            for item in self.levels
        }
        self.rows = len(monitor.states)
        self.totals = {item: _Totals(self.rows) for item in self.levels}
        self.steps = 0

    def __call__(self, step: int, logits: Tensor) -> None:
        diagnostic_done = getattr(self.monitor, "diagnostic_done", self.monitor.done)
        live = [row for row, done in enumerate(diagnostic_done) if not done]
        if not live:
            return
        self.steps += 1
        keys = [(self.monitor.states[row].key(), self.monitor.cursors[row].key())
                for row in live]
        offsets = [self.monitor.states[row].offset for row in live]
        index = torch.as_tensor(live, dtype=torch.long, device=logits.device)
        # One host transfer per decoded symbol, then softmax in float64.
        # Normalising first in float32 can round a small but real illegal tail
        # out of the quantity this instrument exists to measure.
        probs = torch.softmax(
            logits.index_select(0, index).detach().cpu().to(torch.float64),
            dim=-1,
        ).numpy()
        strata = [key[0].stratum for key in keys]
        for level, table in self.tables.items():
            reading = mass_by_rule(probs, table, keys)
            self.totals[level].add_rows(
                np.array(live), reading, keys, strata, offsets
            )

    def as_dict(self) -> dict:
        cells = {
            level: self.totals[level].as_dict(self.rows, level)
            for level in self.levels
        }
        for cell in cells.values():
            # Per *row*, not per program: a decode has no corpus.
            cell["per_row_renorm_bits"] = cell.pop("per_program_renorm_bits")
            cell["decode_steps"] = self.steps
        if len(cells) == 1:
            out = next(iter(cells.values()))
            out["scored_against"] = self.levels[0]
            return out
        return {
            "levels": cells,
            "decode_steps": self.steps,
            "scored_against": list(self.levels),
        }


__all__ = [
    "DYNAMIC_RULES",
    "POLICY_RULES",
    "STATIC_RULES",
    "GenerationMass",
    "dynamic_support",
    "mass_by_rule",
    "position_keys",
]
