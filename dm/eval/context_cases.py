"""Model-blind construction of Direction 2's natural twins.

Nothing here may see a checkpoint.  The builders take programs, a frozen
`LanguagePolicy`, frozen corpus counts and a seed, and they return cases plus a
census of everything they refused.  `docs/directions.md` §4.3-§4.5 is the
contract; the split from `dm/eval/context.py` is the one §6.2 asks for -- case
construction and causal estimands are different concerns, and only the second
one is ever allowed to touch a model.

Three venues live here:

- **`synthetic_flat_step`** -- copy-3 step twins on the flat synthetic corpus.
  The generator's box is square and `random_grid` takes the horizontal branch,
  so every step in this corpus is an x-step.  That makes the venue the
  *in-support* primary and makes it useless as evidence about the y axis.
- **`synthetic_flat_step_d4r1`** -- the same cases under an exact quarter turn.
  Rotation about the canvas centre is exact and always in-canvas, so the
  relation survives byte-for-byte while the coordinates move to a region the
  corpus never trained on.  An effect that vanishes here is anisotropy, which
  is a finding rather than a failure (§4.1).
- **`composed_shape_copy2`** -- copy-2 shape twins on the constructed
  compositional corpus, where `Scene` knows which of its own bytes are copies
  and which are distractors.

Every venue produces the same three-block case: the relation-bearing twin, an
unrelated-target donor control and an unrelated-block prefix control.  A case
that cannot supply all three is rejected and counted, never emitted with a
missing column (§4.8 turns a missing control into `incomplete`, not a result).
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace

from ..data.augment import Affine, apply
from ..data.composed import Scene, bounds
from ..data.fingerprint import digest
from ..isa.asm import parse
from ..isa.codec import Codec
from ..isa.spec import SPECS, Kind, Op, spec_for
from ..isa.state import LanguagePolicy
from .context import (
    BUILDER_SEED,
    CASE_SCHEMA,
    ContextCase,
    Intervention,
    MatchFeatures,
    byte_distance,
    legal_first_bytes,
    prefix_is_live,
    signature,
    target_is_legal,
)
from .recovery import repeat_copies

#: Displacements offered to the unrelated-block control when the primary
#: intervention is not itself a translation.  Ordered so the search is
#: deterministic; the chosen one is whichever *feasible* candidate best matches
#: the primary edit's byte distance, which is the quantity §4.4 asks to hold
#: fixed between a relevant and an irrelevant block change.
BLOCK_SHIFTS: tuple[tuple[int, int], ...] = tuple(
    (dx * scale, dy * scale)
    for scale in (4, 8, 12, 16, 24, 32)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1),
                   (1, 1), (1, -1), (-1, 1), (-1, -1))
)


# ---------------------------------------------------------------------------
# frozen corpus counts


@dataclass(frozen=True)
class CorpusStats:
    """Marginal counts a matched control is allowed to be built from.

    §4.4: matching features come from *frozen corpus counts*, never from model
    output.  `split` records which half they were counted on, because a
    frequency read off the candidate corpus and one read off the training
    corpus are different quantities and a report that confused them would be
    matching donors against the wrong distribution.
    """

    label: str
    split: str
    programs: int
    x_counts: tuple[int, ...]
    y_counts: tuple[int, ...]
    #: `(copy count, sorted step magnitudes)`.  The counterfactual step is drawn
    #: from the support the generator actually produced at that count, not from
    #: the ISA's whole i8 range.
    step_support: tuple[tuple[int, tuple[int, ...]], ...]
    step_support_split: str

    @classmethod
    def from_programs(cls, programs: Sequence[bytes], *, label: str, split: str,
                      step_programs: Sequence[bytes] | None = None,
                      step_split: str | None = None) -> CorpusStats:
        x_counts = [0] * 256
        y_counts = [0] * 256
        for program in programs:
            for x, y in _coordinates(program):
                x_counts[x] += 1
                y_counts[y] += 1
        source = programs if step_programs is None else step_programs
        support: dict[int, set[int]] = {}
        for program in source:
            for group in repeat_copies(program):
                if len(group) < 2:
                    continue
                delta = translation(program[group[0]], program[group[1]])
                if delta is None or delta == (0, 0):
                    continue
                support.setdefault(len(group), set()).add(
                    abs(delta[0]) or abs(delta[1])
                )
        return cls(
            label=label, split=split, programs=len(programs),
            x_counts=tuple(x_counts), y_counts=tuple(y_counts),
            step_support=tuple((count, tuple(sorted(values)))
                               for count, values in sorted(support.items())),
            step_support_split=step_split or split,
        )

    @property
    def total(self) -> int:
        return sum(self.x_counts)

    def frequency(self, value: int, axis: str) -> float:
        counts = self.x_counts if axis == "x" else self.y_counts
        return counts[value] / max(1, self.total)

    def steps_for(self, count: int) -> tuple[int, ...]:
        table = dict(self.step_support)
        if count in table:
            return table[count]
        pooled: set[int] = set()
        for values in table.values():
            pooled.update(values)
        return tuple(sorted(pooled))

    def has_count(self, count: int) -> bool:
        return count in dict(self.step_support)

    def coordinate_frequency(self, program: bytes) -> float:
        """Mean frozen-corpus frequency of a program's coordinate operands."""
        values = [self.frequency(x, "x") + self.frequency(y, "y")
                  for x, y in _coordinates(program)]
        return sum(values) / (2 * len(values)) if values else 0.0

    def profile(self, programs: Iterable[bytes]) -> tuple[float, ...]:
        """Normalised `x`/`y` histogram of some strings' coordinate operands."""
        bins = [0.0] * 512
        total = 0
        for program in programs:
            for x, y in _coordinates(program):
                bins[x] += 1.0
                bins[256 + y] += 1.0
                total += 2
        return tuple(value / total for value in bins) if total else tuple(bins)

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "split": self.split,
            "programs": self.programs,
            "coordinate_operands": self.total,
            "step_support": {str(count): list(values)
                             for count, values in self.step_support},
            "step_support_split": self.step_support_split,
            "digest": self.digest,
        }

    @property
    def digest(self) -> str:
        """Content hash of the counts themselves, not of the config naming them.

        `dm/data/fingerprint.py`'s rule one level up: a description derived from
        a config cannot see a default, so the frozen matching features are
        identified by the numbers a donor was actually matched against.
        """
        body = json.dumps({
            "label": self.label, "split": self.split, "programs": self.programs,
            "x": list(self.x_counts), "y": list(self.y_counts),
            "steps": [[count, list(values)] for count, values in self.step_support],
            "step_split": self.step_support_split,
        }, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(body).hexdigest()[:16]


def marginal_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    """Total-variation distance between two coordinate profiles."""
    return 0.5 * sum(abs(a - b) for a, b in zip(left, right))


def _coordinates(program: bytes) -> list[tuple[int, int]]:
    """Every `COORD` pair, by `Kind` and never positionally.

    A byte walk rather than `parse`, because the marginals are counted over a
    200,000-program training split and building an `Instr` per instruction to
    read two bytes off it costs about an order of magnitude more than reading
    the two bytes.
    """
    out: list[tuple[int, int]] = []
    pc = 0
    while pc < len(program):
        spec = spec_for(program[pc])
        index = 0
        while index < len(spec.operands):
            if (spec.operands[index] is Kind.COORD
                    and index + 1 < len(spec.operands)
                    and spec.operands[index + 1] is Kind.COORD):
                out.append((program[pc + 1 + index], program[pc + 2 + index]))
                index += 2
            else:
                index += 1
        pc += spec.size
    return out


def translation(left: bytes, right: bytes) -> tuple[int, int] | None:
    """The one constant translation carrying `left` to `right`, or None."""
    first, second = parse(left), parse(right)
    if len(first) != len(second):
        return None
    delta: tuple[int, int] | None = None
    for a, b in zip(first, second):
        if a.mnemonic != b.mnemonic:
            return None
        spec = SPECS[Op[a.mnemonic]]
        index = 0
        while index < len(spec.operands):
            kind = spec.operands[index]
            if (kind is Kind.COORD and index + 1 < len(spec.operands)
                    and spec.operands[index + 1] is Kind.COORD):
                candidate = (b.args[index] - a.args[index],
                             b.args[index + 1] - a.args[index + 1])
                if delta is None:
                    delta = candidate
                elif delta != candidate:
                    return None
                index += 2
            else:
                if a.args[index] != b.args[index]:
                    return None
                index += 1
    return delta


def instruction_spans(program: bytes) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    pc = 0
    for instr in parse(program):
        size = SPECS[Op[instr.mnemonic]].size
        spans.append((pc, pc + size))
        pc += size
    return spans


def recover_transform(source: bytes, image: bytes) -> Affine | None:
    """The complete affine carrying `source` to `image`, verified byte for byte.

    §4.2 forbids inferring the transform from the layout policy: today's
    composed corpus happens to use one quarter turn about the origin with zero
    translation, and a builder that assumed it would silently mis-derive every
    target the day the policy gains a shift.  The D4 part is searched, the
    translation is read off the bounding boxes, and the whole thing is then
    *checked* rather than trusted.
    """
    target_box = bounds(image)
    if target_box is None:
        return None
    for turns in range(4):
        for mirror in (False, True):
            rotated = apply(source, Affine(mirror=mirror, quarter_turns=turns))
            if rotated is None:
                continue
            box = bounds(rotated)
            if box is None:
                continue
            candidate = Affine(dx=target_box[0] - box[0], dy=target_box[1] - box[1],
                               mirror=mirror, quarter_turns=turns)
            if apply(source, candidate) == image:
                return candidate
    return None


def _boxes_overlap(a: tuple[int, ...], b: tuple[int, ...]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


# ---------------------------------------------------------------------------
# shared control construction


def candidate_runs(prefix: bytes, limit: int, want_signature,
                   want_length: int) -> list[tuple[int, int, str]]:
    """Whole-instruction runs of the right size, best match first.

    Searched over `prefix[:limit]`, which is everything before the
    relation-bearing block, so an unrelated control block can never overlap the
    block whose change the primary contrast is about.  Exact skeleton matches
    sort ahead of length-only ones, and a length-only match carries the named
    `relaxed_unrelated_block_skeleton` stratum -- §4.4 allows a relaxation and
    forbids a silent one.

    Among equal matches the latest run wins: the relevant block abuts its
    target, so `distance_to_target` is the gap this venue cannot close and the
    builder should leave it as small as the corpus allows.
    """
    spans = [span for span in instruction_spans(prefix) if span[1] <= limit]
    runs: list[tuple[int, int, str]] = []
    for index, (start, _) in enumerate(spans):
        for stop_index in range(index, len(spans)):
            stop = spans[stop_index][1]
            if stop - start > want_length:
                break
            if stop - start == want_length:
                stratum = ("" if signature(prefix[start:stop]) == want_signature
                           else "relaxed_unrelated_block_skeleton")
                runs.append((start, stop, stratum))
                break
    runs.sort(key=lambda run: (run[2] != "", -run[1]))
    return runs


def _shift_run(prefix: bytes, runs: list[tuple[int, int, str]],
               shift: tuple[int, int], policy: LanguagePolicy
               ) -> tuple[int, int, bytes, str] | None:
    """The first run that survives `shift` on canvas and stays a live prefix."""
    for start, stop, stratum in runs:
        block = prefix[start:stop]
        moved = apply(block, Affine(dx=shift[0], dy=shift[1]))
        if moved is None or moved == block:
            continue
        if not prefix_is_live(prefix[:start] + moved + prefix[stop:], policy):
            continue
        return start, stop, moved, stratum
    return None


def _control_donor(row: dict, pool: Sequence[dict], stats: CorpusStats,
                   used: Counter) -> dict | None:
    """The best matched unrelated-target donor available.

    Matching is on the frozen corpus counts and on the case's own geometry --
    equal target length and skeleton first, then the smallest gap in
    between-target byte distance, coordinate marginals and byte ordinal.  Donor
    reuse is the first key, so the assignment stays as close to a perfect
    matching as the corpus allows (§4.4).
    """
    eligible = [
        other for other in pool
        if other["source_index"] != row["source_index"]
        and other["source_index"] != row["donor_index"]
        and len(other["target_a"]) == len(row["target_a"])
        and other["signature"] == row["signature"]
        and other["target_a"] != other["target_b"]
    ]
    if not eligible:
        return None
    frequency = _pair_frequency(row, stats)
    distance = byte_distance(row["target_a"], row["target_b"])

    def key(other: dict) -> tuple:
        return (
            used[other["source_index"]],
            abs(byte_distance(other["target_a"], other["target_b"]) - distance),
            round(abs(_pair_frequency(other, stats) - frequency), 9),
            abs(other["target_byte_start"] - row["target_byte_start"]),
            other["source_digest"],
            other["group_index"],
        )
    return min(eligible, key=key)


def _pair_frequency(row: dict, stats: CorpusStats) -> float:
    """Mean frozen-corpus frequency of a target pair's coordinate operands.

    This -- not a total-variation distance between the two pairs' own histograms
    -- is what the marginal-frequency control needs.  Ten coordinates drawn from
    a 256-value alphabet make two nearly disjoint empirical distributions
    whatever the donor is, so a TV distance would rank every candidate at ~1 and
    match nothing.
    """
    return 0.5 * (stats.coordinate_frequency(row["target_a"])
                  + stats.coordinate_frequency(row["target_b"]))


def _features(row: dict, donor: dict, stats: CorpusStats,
              policy: LanguagePolicy, codec: Codec,
              strata: Iterable[str]) -> MatchFeatures:
    names = list(strata)
    gap = abs(donor["target_byte_start"] - row["target_byte_start"])
    if gap:
        names.append("unmatched_target_byte_ordinal")
    if row["unrelated"].distance_to_target != row["relevant"].distance_to_target:
        names.append("unmatched_unrelated_block_distance")
    if row["unrelated"].length != row["relevant"].length:
        names.append("unmatched_unrelated_block_length")
    return MatchFeatures(
        target_ordinal_gap=gap,
        prefix_distance_gap=abs(row["unrelated"].distance_to_target
                                - row["relevant"].distance_to_target),
        byte_distance_primary=byte_distance(row["target_a"], row["target_b"]),
        byte_distance_control=byte_distance(donor["target_a"], donor["target_b"]),
        byte_distance_gap=abs(byte_distance(row["target_a"], row["target_b"])
                              - byte_distance(donor["target_a"],
                                              donor["target_b"])),
        coordinate_marginal_distance=abs(_pair_frequency(row, stats)
                                         - _pair_frequency(donor, stats)),
        frequency_a=stats.coordinate_frequency(row["target_a"]),
        frequency_b=stats.coordinate_frequency(row["target_b"]),
        frequency_control_a=stats.coordinate_frequency(donor["target_a"]),
        frequency_control_b=stats.coordinate_frequency(donor["target_b"]),
        legal_first_bytes=legal_first_bytes(row["prefix_a"], policy, codec),
        strata=tuple(sorted(set(names))),
    )


def _assemble(rows: list[dict], *, venue: str, stats: CorpusStats,
              policy: LanguagePolicy, codec: Codec, max_cases: int,
              corpus_split: str, held_out: bool, rejected: Counter,
              control_pool: list[dict] | None = None) -> list[ContextCase]:
    """Match each candidate a control donor, in one deterministic greedy pass.

    A row is admitted only if a donor is available *at that point*, and the two
    identity sets never intersect: a source already spent as somebody's control
    donor cannot become a primary, and a primary cannot become a donor. That is
    §4.4's "partition held-out motifs/donors before building score rows",
    obtained by construction rather than by a random split that would have to be
    checked afterwards.

    Greedy rather than optimal, and deliberately so. A maximum matching would
    depend on the whole candidate set in a way no reader can replay by hand;
    this pass walks the rows in digest order and can be followed row by row. It
    is also why a rejected row does *not* consume its source -- the scarce
    resource here is same-skeleton donors, not candidates.

    `control_pool` exists because the two venues have different supplies of
    matched target pairs.  Step targets all share one skeleton, so another
    case's twin pair is both available and better matched; composed motifs have
    ~864 distinct skeletons over 1,000 scenes, so the only reliable same-key
    pair is a scene's *own* orbit -- a motif and its image, which differ by
    construction.  Either way the donor is a third identity and is excluded
    from the primary set.
    """
    rows = sorted(rows, key=lambda row: (row["source_digest"], row["group_index"]))
    supply = rows if control_pool is None else control_pool
    primaries: set[int] = set()
    donors: set[int] = set()
    used: Counter = Counter()
    cases: list[ContextCase] = []
    for row in rows:
        if len(cases) == max_cases:
            break
        if row["source_index"] in primaries:
            rejected["source_already_primary"] += 1
            continue
        if row["source_index"] in donors:
            # Already spent as somebody's control donor.  The two identity sets
            # stay disjoint, so a case can never be its own control.
            rejected["source_already_control_donor"] += 1
            continue
        pool = [entry for entry in supply
                if entry["source_index"] not in primaries
                and entry["source_index"] != row["source_index"]]
        donor = _control_donor(row, pool, stats, used)
        if donor is None:
            rejected["no_matched_control_donor"] += 1
            continue
        primaries.add(row["source_index"])
        donors.add(donor["source_index"])
        used[donor["source_index"]] += 1
        features = _features(row, donor, stats, policy, codec, row["strata"])
        cases.append(ContextCase(
            case_id=f"{venue}-{len(cases):04d}",
            venue=venue,
            schema=CASE_SCHEMA,
            source_index=row["source_index"],
            source_digest=row["source_digest"],
            source_group="primary",
            donor_index=row["donor_index"],
            donor_digest=row["donor_digest"],
            donor_group=row["donor_group"],
            donor_kind=row["donor_kind"],
            control_donor_index=donor["source_index"],
            control_donor_digest=donor["source_digest"],
            corpus_split=corpus_split,
            held_out_from_train=held_out,
            prefix_a=row["prefix_a"],
            prefix_b=row["prefix_b"],
            target_a=row["target_a"],
            target_b=row["target_b"],
            block_prefix_a=row["block_prefix_a"],
            block_prefix_b=row["block_prefix_b"],
            control_target_a=donor["target_a"],
            control_target_b=donor["target_b"],
            relevant=row["relevant"],
            unrelated=row["unrelated"],
            target_signature=row["signature"],
            target_ordinal=row["target_ordinal"],
            target_byte_start=row["target_byte_start"],
            target_instruction_ordinal=row["target_instruction_ordinal"],
            control_target_byte_start=donor["target_byte_start"],
            transform=row["transform"],
            features=features,
        ))
    return cases


# ---------------------------------------------------------------------------
# venue 1: flat synthetic copy-3 step twins


def _reaxis(body: bytes, delta: tuple[int, int], axis: str, skeleton
            ) -> tuple[tuple[int, int], bytes, bytes] | None:
    """Re-lay the copy chain along `axis`, keeping the step magnitude.

    The exact-rotation venue moves *every* byte of the program and lands the
    model 2.3x outside its own prefix cost, which makes a null there
    uninterpretable -- a model with no usable distribution cannot be asked
    whether its relation use transfers. This moves only the two copies the
    relation is about and leaves the head, the motif's internal shape and the
    coordinate ranges exactly where the corpus put them, so the axis question
    is asked at the smallest distribution shift that can ask it at all.
    """
    magnitude = abs(delta[0]) or abs(delta[1])
    want = (magnitude, 0) if axis == "x" else (0, magnitude)
    second = apply(body, Affine(dx=want[0], dy=want[1]))
    third = apply(body, Affine(dx=2 * want[0], dy=2 * want[1]))
    if second is None or third is None:
        return None
    if any((len(copy), signature(copy)) != (len(body), skeleton)
           for copy in (second, third)):
        return None
    return want, second, third


def _step_rows(programs: Sequence[bytes], policy: LanguagePolicy,
               stats: CorpusStats, seed: int, rejected: Counter,
               axis: str = "observed") -> list[dict]:
    rows: list[dict] = []
    for source_index, program in enumerate(programs):
        source_digest = digest([program])
        for group_index, group in enumerate(repeat_copies(program)):
            if len(group) < 3:
                rejected["fewer_than_three_copies"] += 1
                continue
            body_1 = program[group[0]]
            body_2 = program[group[1]]
            body_3 = program[group[2]]
            delta = translation(body_1, body_2)
            if delta is None or delta == (0, 0):
                rejected["not_constant_translation"] += 1
                continue
            if translation(body_2, body_3) != delta:
                rejected["nonlinear_step"] += 1
                continue
            if apply(body_1, Affine(dx=2 * delta[0], dy=2 * delta[1])) != body_3:
                rejected["third_copy_mismatch"] += 1
                continue
            head = program[:group[0].start]
            skeleton = signature(body_1)
            if axis != "observed":
                moved = _reaxis(body_1, delta, axis, skeleton)
                if moved is None:
                    rejected["axis_off_canvas"] += 1
                    continue
                delta, body_2, body_3 = moved
            prefix_a = head + body_1 + body_2
            if not prefix_is_live(prefix_a, policy):
                rejected["prefix_not_live"] += 1
                continue
            if not target_is_legal(prefix_a, body_3, policy):
                rejected["target_not_legal"] += 1
                continue
            strata: list[str] = []
            if axis != "observed":
                strata.append(f"forced_{axis}_axis")
            if not stats.has_count(len(group)):
                strata.append("pooled_step_support")
            magnitudes = list(stats.steps_for(len(group)))
            # Deterministic per candidate, so the table does not depend on the
            # order programs happen to arrive in.
            random.Random(f"{seed}:{source_digest}:{group_index}").shuffle(magnitudes)
            row = _step_counterfactual(
                program, group, head, body_1, body_2, body_3, delta, magnitudes,
                skeleton, policy, rejected,
            )
            if row is None:
                continue
            row.update({
                "source_index": source_index,
                "source_digest": source_digest,
                "group_index": group_index,
                "donor_index": source_index,
                "donor_digest": source_digest,
                "donor_group": "self",
                "donor_kind": "self_translation",
                "signature": skeleton,
                "target_ordinal": 3,
                # Merged, not replaced: the counterfactual search names its own
                # relaxations and dropping them would hide a named stratum.
                "strata": tuple(sorted(set(row["strata"]) | set(strata))),
            })
            rows.append(row)
    return rows


def _step_counterfactual(program: bytes, group: list[slice], head: bytes,
                         body_1: bytes, body_2: bytes, body_3: bytes,
                         delta: tuple[int, int], magnitudes: list[int],
                         skeleton, policy: LanguagePolicy,
                         rejected: Counter) -> dict | None:
    """World B, plus the unrelated block that takes the identical displacement."""
    sign_x = 1 if delta[0] > 0 else -1
    sign_y = 1 if delta[1] > 0 else -1
    prefix_a = head + body_1 + body_2
    block_start = len(head) + len(body_1)
    runs = candidate_runs(prefix_a, len(head), skeleton, len(body_2))
    if not runs:
        rejected["no_unrelated_block_run"] += 1
        return None
    for magnitude in magnitudes:
        candidate = ((sign_x * magnitude, 0) if delta[0]
                     else (0, sign_y * magnitude))
        if candidate == delta:
            continue
        trial_2 = apply(body_1, Affine(dx=candidate[0], dy=candidate[1]))
        trial_3 = apply(body_1, Affine(dx=2 * candidate[0], dy=2 * candidate[1]))
        if trial_2 is None or trial_3 is None:
            continue
        if any((len(trial), signature(trial)) != (len(body_1), skeleton)
               for trial in (trial_2, trial_3)):
            continue
        if trial_3 == body_3:
            continue
        prefix_b = head + body_1 + trial_2
        if not prefix_is_live(prefix_b, policy):
            continue
        if not target_is_legal(prefix_b, trial_3, policy):
            continue
        if not target_is_legal(prefix_a, trial_3, policy):
            continue
        if not target_is_legal(prefix_b, body_3, policy):
            continue
        shift = (candidate[0] - delta[0], candidate[1] - delta[1])
        found = _shift_run(prefix_a, runs, shift, policy)
        if found is None:
            continue
        start, stop, moved, stratum = found
        block_prefix_b = prefix_a[:start] + moved + prefix_a[stop:]
        if not target_is_legal(block_prefix_b, body_3, policy):
            continue
        if not target_is_legal(block_prefix_b, trial_3, policy):
            continue
        return {
            "prefix_a": prefix_a,
            "prefix_b": prefix_b,
            "target_a": body_3,
            "target_b": trial_3,
            "block_prefix_a": prefix_a,
            "block_prefix_b": block_prefix_b,
            "relevant": Intervention(
                kind="step_twin", start=block_start, stop=len(prefix_a),
                signature=signature(body_2), dx=shift[0], dy=shift[1],
                byte_distance=byte_distance(body_2, trial_2),
                distance_to_target=0,
            ),
            "unrelated": Intervention(
                kind="block_translation", start=start, stop=stop,
                signature=signature(prefix_a[start:stop]),
                dx=shift[0], dy=shift[1],
                byte_distance=byte_distance(prefix_a[start:stop], moved),
                distance_to_target=len(prefix_a) - stop,
            ),
            # `repeat_copies` builds its offsets with numpy, so the ordinals
            # arrive as `np.int64` and would serialise as an unsupported type.
            "target_byte_start": int(group[2].start),
            "target_instruction_ordinal": len(parse(program[:group[2].start])),
            "transform": (2 * delta[0], 2 * delta[1], 0, 0),
            "strata": (stratum,) if stratum else (),
        }
    rejected["no_matched_counterfactual_and_block"] += 1
    return None


#: Which venue each step axis produces.  `observed` is the corpus's own
#: horizontal branch and is the in-support primary; `y` is the axis diagnostic.
STEP_VENUES: dict[str, str] = {
    "observed": "synthetic_flat_step",
    "x": "synthetic_flat_step",
    "y": "synthetic_flat_step_yaxis",
}


def build_step_cases(programs: Sequence[bytes], policy: LanguagePolicy,
                     codec: Codec, stats: CorpusStats, *,
                     seed: int = BUILDER_SEED, max_cases: int = 64,
                     corpus_split: str = "val", held_out: bool = True,
                     axis: str = "observed",
                     ) -> tuple[list[ContextCase], Counter]:
    if axis not in STEP_VENUES:
        raise ValueError(f"axis must be one of {sorted(STEP_VENUES)}, got {axis!r}")
    rejected: Counter = Counter()
    rows = _step_rows(programs, policy, stats, seed, rejected, axis=axis)
    for row in rows:
        row["strata"] = tuple(sorted(set(row["strata"])))
    cases = _assemble(rows, venue=STEP_VENUES[axis], stats=stats,
                      policy=policy, codec=codec, max_cases=max_cases,
                      corpus_split=corpus_split, held_out=held_out,
                      rejected=rejected)
    rejected["candidate_rows"] = len(rows)
    return cases, rejected


# ---------------------------------------------------------------------------
# venue 2: the exact D4 rotation of venue 1


def rotate_cases(cases: Sequence[ContextCase], policy: LanguagePolicy,
                 codec: Codec, stats: CorpusStats, *, quarter_turns: int = 1,
                 mirror: bool = False, venue: str = "synthetic_flat_step_d4r1",
                 ) -> tuple[list[ContextCase], Counter]:
    """Rotate whole cases, exactly, so the x result can be read on the y axis.

    A quarter turn about the canvas centre maps the canvas onto itself, so the
    rotation never leaves it and never rounds; the relation is preserved byte
    for byte while every coordinate moves to a region an x-only corpus never
    trained on.  Coordinate frequencies are recomputed under the same frozen
    counts, so the rotated rows carry their own -- much lower -- support
    columns instead of inheriting the primary venue's.
    """
    transform = Affine(mirror=mirror, quarter_turns=quarter_turns)
    rejected: Counter = Counter()
    out: list[ContextCase] = []
    for case in cases:
        candidates = {name: apply(getattr(case, name), transform)
                      for name in ("prefix_a", "prefix_b", "target_a", "target_b",
                                   "block_prefix_a", "block_prefix_b",
                                   "control_target_a", "control_target_b")}
        if any(value is None for value in candidates.values()):
            rejected["rotation_left_canvas"] += 1
            continue
        moved: dict[str, bytes] = {name: value for name, value in
                                   candidates.items() if value is not None}
        if not all(prefix_is_live(moved[name], policy)
                   for name in ("prefix_a", "prefix_b",
                                "block_prefix_a", "block_prefix_b")):
            rejected["rotated_prefix_not_live"] += 1
            continue
        if not all(target_is_legal(moved[prefix], moved[target], policy)
                   for prefix in ("prefix_a", "prefix_b",
                                  "block_prefix_a", "block_prefix_b")
                   for target in ("target_a", "target_b",
                                  "control_target_a", "control_target_b")):
            rejected["rotated_target_not_legal"] += 1
            continue
        dx, dy = transform.linear(case.relevant.dx, case.relevant.dy)
        relevant = replace(
            case.relevant, dx=dx, dy=dy,
            signature=signature(moved["prefix_a"][case.relevant.start:
                                                  case.relevant.stop]),
            byte_distance=byte_distance(
                moved["prefix_a"][case.relevant.start:case.relevant.stop],
                moved["prefix_b"][case.relevant.start:case.relevant.stop]),
        )
        unrelated = replace(
            case.unrelated, dx=dx, dy=dy,
            signature=signature(moved["block_prefix_a"][case.unrelated.start:
                                                        case.unrelated.stop]),
            byte_distance=byte_distance(
                moved["block_prefix_a"][case.unrelated.start:case.unrelated.stop],
                moved["block_prefix_b"][case.unrelated.start:case.unrelated.stop]),
        )
        primary = 0.5 * (stats.coordinate_frequency(moved["target_a"])
                         + stats.coordinate_frequency(moved["target_b"]))
        control = 0.5 * (stats.coordinate_frequency(moved["control_target_a"])
                         + stats.coordinate_frequency(moved["control_target_b"]))
        # The *relation* is conjugated, not merely rotated. `A T A^-1` is what a
        # scope becomes when its whole frame moves; rotating the relation's
        # translation alone would be right for this venue by accident -- a
        # conjugated pure translation stays a pure translation -- and wrong the
        # moment the relation carries a D4 part, which the shape venue's does.
        relation = Affine(dx=case.transform[0], dy=case.transform[1],
                          mirror=bool(case.transform[2]),
                          quarter_turns=case.transform[3])
        conjugated = Affine.of(
            relation.as_transform().conjugate(transform.as_transform()))
        out.append(replace(
            case,
            case_id=f"{venue}-{len(out):04d}",
            venue=venue,
            relevant=relevant,
            unrelated=unrelated,
            target_signature=signature(moved["target_a"]),
            transform=(conjugated.dx, conjugated.dy, int(conjugated.mirror),
                       conjugated.quarter_turns % 4),
            features=replace(
                case.features,
                coordinate_marginal_distance=abs(primary - control),
                frequency_a=stats.coordinate_frequency(moved["target_a"]),
                frequency_b=stats.coordinate_frequency(moved["target_b"]),
                frequency_control_a=stats.coordinate_frequency(
                    moved["control_target_a"]),
                frequency_control_b=stats.coordinate_frequency(
                    moved["control_target_b"]),
                legal_first_bytes=legal_first_bytes(moved["prefix_a"], policy,
                                                    codec),
                strata=tuple(sorted(set(case.features.strata)
                                    | {"exact_d4_rotation"})),
            ),
            **moved,
        ))
    return out, rejected


# ---------------------------------------------------------------------------
# venue 3: composed copy-2 shape twins


def _scene_blocks(scenes: Sequence[Scene]) -> dict[tuple, list[dict]]:
    """Every whole block in the corpus, keyed by `(length, skeleton)`.

    Donors come from this pool rather than from the orbit alone because the
    distractors are held-out motifs too, and `Scene.distractor_spans` is what
    makes their boundaries knowable at all.
    """
    pool: dict[tuple, list[dict]] = {}
    for index, scene in enumerate(scenes):
        spans = (scene.copies[0],) + tuple(scene.distractor_spans)
        for span in spans:
            block = scene.flat[span]
            pool.setdefault((len(block), signature(block)), []).append(
                {"scene": index, "bytes": block}
            )
    return pool


def _rebase(donor: bytes, anchor: bytes) -> bytes | None:
    """Place a donor motif exactly where the source motif sits."""
    donor_box, anchor_box = bounds(donor), bounds(anchor)
    if donor_box is None or anchor_box is None:
        return None
    return apply(donor, Affine(dx=anchor_box[0] - donor_box[0],
                               dy=anchor_box[1] - donor_box[1]))


def _shape_rows(scenes: Sequence[Scene], policy: LanguagePolicy,
                seed: int, rejected: Counter) -> list[dict]:
    pool = _scene_blocks(scenes)
    rows: list[dict] = []
    for index, scene in enumerate(scenes):
        if len(scene.copies) < 2:
            rejected["fewer_than_two_copies"] += 1
            continue
        motif = scene.flat[scene.copies[0]]
        target_a = scene.flat[scene.copies[1]]
        transform = recover_transform(motif, target_a)
        if transform is None:
            rejected["unverified_orbit_transform"] += 1
            continue
        prefix_a = scene.flat[:scene.copies[1].start]
        if not prefix_is_live(prefix_a, policy):
            rejected["prefix_not_live"] += 1
            continue
        if not target_is_legal(prefix_a, target_a, policy):
            rejected["target_not_legal"] += 1
            continue
        skeleton = signature(motif)
        others = [entry for entry in pool.get((len(motif), skeleton), [])
                  if entry["scene"] != index]
        if not others:
            rejected["no_matched_donor_motif"] += 1
            continue
        head_blocks = [span for span in scene.distractor_spans
                       if span.stop <= scene.copies[0].start]
        if not head_blocks:
            rejected["no_unrelated_head_block"] += 1
            continue
        random.Random(f"{seed}:{digest([scene.flat])}").shuffle(others)
        row = _shape_twin(scene, index, motif, target_a, transform, prefix_a,
                          skeleton, others, head_blocks, policy, rejected)
        if row is None:
            continue
        rows.append(row)
    return rows


def _shape_twin(scene: Scene, index: int, motif: bytes, target_a: bytes,
                transform: Affine, prefix_a: bytes, skeleton,
                others: list[dict], head_blocks: list[slice],
                policy: LanguagePolicy, rejected: Counter) -> dict | None:
    fixed = [bounds(scene.flat[span]) for span in scene.distractor_spans]
    for entry in others:
        donor = _rebase(entry["bytes"], motif)
        if donor is None or donor == motif:
            continue
        target_b = apply(donor, transform)
        if target_b is None or target_b == target_a:
            continue
        if len(target_b) != len(target_a) or signature(target_b) != skeleton:
            continue
        donor_box, image_box = bounds(donor), bounds(target_b)
        if donor_box is None or image_box is None or _boxes_overlap(donor_box,
                                                                   image_box):
            continue
        if any(box is not None and (_boxes_overlap(box, donor_box)
                                    or _boxes_overlap(box, image_box))
               for box in fixed):
            continue
        prefix_b = scene.flat[:scene.copies[0].start] + donor
        if not prefix_is_live(prefix_b, policy):
            continue
        if not all(target_is_legal(prefix, target, policy)
                   for prefix in (prefix_a, prefix_b)
                   for target in (target_a, target_b)):
            continue
        block = _shape_block(scene, prefix_a, head_blocks,
                             byte_distance(motif, donor), policy,
                             (target_a, target_b))
        if block is None:
            continue
        start, stop, shift, block_bytes = block
        return {
            "source_index": index,
            "source_digest": digest([scene.flat]),
            "group_index": 0,
            "donor_index": entry["scene"],
            # Filled in by the caller, which holds every scene's digest.
            "donor_digest": "",
            "donor_group": "donor",
            "donor_kind": "substituted_block",
            "prefix_a": prefix_a,
            "prefix_b": prefix_b,
            "target_a": target_a,
            "target_b": target_b,
            "block_prefix_a": prefix_a,
            "block_prefix_b": block_bytes,
            "signature": signature(target_a),
            "relevant": Intervention(
                kind="shape_twin", start=scene.copies[0].start,
                stop=scene.copies[0].stop, signature=skeleton, dx=0, dy=0,
                byte_distance=byte_distance(motif, donor),
                distance_to_target=0,
            ),
            "unrelated": Intervention(
                kind="block_translation", start=start, stop=stop,
                signature=signature(prefix_a[start:stop]),
                dx=shift[0], dy=shift[1],
                byte_distance=byte_distance(prefix_a[start:stop],
                                            block_bytes[start:stop]),
                distance_to_target=len(prefix_a) - stop,
            ),
            "target_ordinal": 2,
            "target_byte_start": scene.copies[1].start,
            "target_instruction_ordinal": len(
                parse(scene.flat[:scene.copies[1].start])),
            "transform": (transform.dx, transform.dy, int(transform.mirror),
                          transform.quarter_turns % 4),
            "strata": (),
        }
    rejected["no_valid_shape_twin"] += 1
    return None


def _shape_block(scene: Scene, prefix_a: bytes, head_blocks: list[slice],
                 want_distance: int, policy: LanguagePolicy,
                 targets: tuple[bytes, bytes]) -> tuple | None:
    """Translate one whole head distractor, matching the primary byte distance.

    The composed venue's relevant edit replaces a motif, so no displacement is
    "the same" one; what can be matched is block length, instruction type and
    the number of bytes that move.  All three are recorded, and the chosen
    shift is the feasible candidate whose byte distance is closest to the
    primary's.
    """
    fixed = [bounds(scene.flat[scene.copies[0]]),
             bounds(scene.flat[scene.copies[1]])]
    ranked: list[tuple] = []
    for span in head_blocks:
        block = prefix_a[span]
        others = fixed + [bounds(prefix_a[other]) for other in head_blocks
                          if other != span]
        for shift in BLOCK_SHIFTS:
            moved = apply(block, Affine(dx=shift[0], dy=shift[1]))
            if moved is None or moved == block:
                continue
            box = bounds(moved)
            if box is None or any(other is not None and _boxes_overlap(box, other)
                                  for other in others):
                continue
            ranked.append((
                (abs(byte_distance(block, moved) - want_distance), -span.stop,
                 abs(shift[0]), abs(shift[1])),
                span.start, span.stop, shift, moved,
            ))
    ranked.sort(key=lambda entry: entry[0])
    for _, start, stop, shift, moved in ranked:
        candidate = prefix_a[:start] + moved + prefix_a[stop:]
        if not prefix_is_live(candidate, policy):
            continue
        if not all(target_is_legal(candidate, target, policy)
                   for target in targets):
            continue
        return start, stop, shift, candidate
    return None


def build_shape_cases(scenes: Sequence[Scene], policy: LanguagePolicy,
                      codec: Codec, stats: CorpusStats, *,
                      seed: int = BUILDER_SEED, max_cases: int = 64,
                      corpus_split: str = "val", held_out: bool = True,
                      ) -> tuple[list[ContextCase], Counter]:
    rejected: Counter = Counter()
    rows = _shape_rows(scenes, policy, seed, rejected)
    digests = {index: digest([scene.flat]) for index, scene in enumerate(scenes)}
    for row in rows:
        row["donor_digest"] = digests[row["donor_index"]]
    cases = _assemble(rows, venue="composed_shape_copy2", stats=stats,
                      policy=policy, codec=codec, max_cases=max_cases,
                      corpus_split=corpus_split, held_out=held_out,
                      rejected=rejected,
                      control_pool=orbit_control_pool(scenes, digests))
    rejected["candidate_rows"] = len(rows)
    return cases, rejected


def orbit_control_pool(scenes: Sequence[Scene],
                       digests: dict[int, str]) -> list[dict]:
    """Every scene's own orbit as a matched, unrelated target pair.

    A scene's copy 0 and copy 1 always share a skeleton -- the orbit transform
    permutes coordinates and never instructions -- and never share bytes, since
    no offered orbit step is the identity.  That makes them the one same-key
    pair the composed corpus reliably supplies, and it makes the control pair
    the same *kind* of object as the primary pair: a motif and its image.
    """
    return [
        {
            "source_index": index,
            "source_digest": digests[index],
            "group_index": 0,
            "target_a": scene.flat[scene.copies[0]],
            "target_b": scene.flat[scene.copies[1]],
            "target_byte_start": scene.copies[1].start,
            "signature": signature(scene.flat[scene.copies[1]]),
        }
        for index, scene in enumerate(scenes)
        if len(scene.copies) >= 2
        and scene.flat[scene.copies[0]] != scene.flat[scene.copies[1]]
    ]


# ---------------------------------------------------------------------------
# census


def census(cases: Sequence[ContextCase], rejected: Counter, *,
           stats: CorpusStats, seed: int, requested: int,
           sources: int, extra: dict | None = None) -> dict:
    """The model-blind balance report §4.5 requires before any weight loads."""
    from .context import components

    rows = [{"source_index": case.source_index,
             "donor_index": case.donor_index,
             "control_donor_index": case.control_donor_index}
            for case in cases]
    groups = components(rows) if rows else {}
    source_uses = Counter(case.source_index for case in cases)
    donor_uses = Counter(case.control_donor_index for case in cases)
    substitute_uses = Counter(case.donor_index for case in cases
                              if case.donor_kind == "substituted_block")
    return {
        "builder_seed": seed,
        "source_programs": sources,
        "requested_cases": requested,
        "included_cases": len(cases),
        "venues": sorted({case.venue for case in cases}),
        "venue_case_counts": _counts(case.venue for case in cases),
        "rejected": dict(sorted(rejected.items())),
        "corpus_stats": stats.as_dict(),
        "balance": {
            "target_lengths": _counts(case.target_bytes for case in cases),
            "prefix_lengths": _counts(case.prefix_bytes for case in cases),
            "target_ordinals": _counts(case.target_ordinal for case in cases),
            "target_byte_starts": _counts(case.target_byte_start
                                          for case in cases),
            # Hashed, not spelled: a composed motif's skeleton is a hundred
            # `(LINE, (COORD, COORD))` pairs and 28 of them would be most of the
            # manifest.  The key is stable, so two venues can still be compared.
            "target_signatures": _counts(_signature_key(case.target_signature)
                                         for case in cases),
            "relevant_block_lengths": _counts(case.relevant.length
                                              for case in cases),
            "unrelated_block_lengths": _counts(case.unrelated.length
                                               for case in cases),
            "unrelated_block_distance": _counts(
                case.unrelated.distance_to_target for case in cases),
            "intervention_magnitude": _counts(
                (abs(case.relevant.dx), abs(case.relevant.dy))
                for case in cases),
            "primary_byte_distance": _counts(
                case.features.byte_distance_primary for case in cases),
            "control_byte_distance": _counts(
                case.features.byte_distance_control for case in cases),
            "legal_first_bytes": _counts(case.features.legal_first_bytes
                                         for case in cases),
            "strata": _counts(name for case in cases
                              for name in case.features.strata),
        },
        "matching": {
            "target_ordinal_gap": _summary(case.features.target_ordinal_gap
                                           for case in cases),
            "prefix_distance_gap": _summary(case.features.prefix_distance_gap
                                            for case in cases),
            "byte_distance_gap": _summary(case.features.byte_distance_gap
                                          for case in cases),
            "coordinate_marginal_distance": _summary(
                case.features.coordinate_marginal_distance for case in cases),
            # Pooled over the venue, where the histograms are dense enough for
            # a total-variation distance to mean something.  Per case they are
            # ten points in a 256-value alphabet and TV is ~1 by construction.
            "pooled_coordinate_profile_distance": marginal_distance(
                stats.profile([case.target_a for case in cases]
                              + [case.target_b for case in cases]),
                stats.profile([case.control_target_a for case in cases]
                              + [case.control_target_b for case in cases]),
            ) if cases else None,
            "frequency_a": _summary(case.features.frequency_a for case in cases),
            "frequency_control_a": _summary(case.features.frequency_control_a
                                            for case in cases),
        },
        "identity": {
            "unique_primary_sources": len(source_uses),
            "max_cases_per_primary_source": max(source_uses.values(), default=0),
            "unique_control_donors": len(donor_uses),
            "max_cases_per_control_donor": max(donor_uses.values(), default=0),
            "unique_substitute_donors": len(substitute_uses),
            "max_cases_per_substitute_donor": max(substitute_uses.values(),
                                                  default=0),
            "primary_donor_overlap": len(set(source_uses) & set(donor_uses)),
            "components": len(groups),
            "largest_component": max((len(members) for members in groups.values()),
                                     default=0),
            # **Per venue is the number that governs, and the pooled one is
            # reported beside it.** Venues are scored and resampled separately,
            # so a program that is a primary in one and a control donor in
            # another biases nothing -- but pooling them collapses the component
            # count and would fail a partition check that is in fact satisfied
            # everywhere it is defined.
            "by_venue": {
                venue: _venue_identity([case for case in cases
                                        if case.venue == venue])
                for venue in sorted({case.venue for case in cases})
            },
            "held_out_from_train": all(case.held_out_from_train
                                       for case in cases),
            "nondegenerate_controls": sum(
                case.control_target_a != case.control_target_b for case in cases
            ),
            "nondegenerate_block_controls": sum(
                case.block_prefix_a != case.block_prefix_b for case in cases
            ),
        },
        **(extra or {}),
    }


def _venue_identity(cases: Sequence[ContextCase]) -> dict:
    """Dependence and partition facts for the rows one venue actually scores."""
    from .context import components

    groups = components([{"source_index": case.source_index,
                          "donor_index": case.donor_index,
                          "control_donor_index": case.control_donor_index}
                         for case in cases]) if cases else {}
    sources = {case.source_index for case in cases}
    donors = {case.control_donor_index for case in cases}
    reuse = Counter(case.control_donor_index for case in cases)
    return {
        "cases": len(cases),
        "components": len(groups),
        "largest_component": max((len(m) for m in groups.values()), default=0),
        "primary_donor_overlap": len(sources & donors),
        "max_cases_per_control_donor": max(reuse.values(), default=0),
    }


def _signature_key(value) -> str:
    return hashlib.sha256(repr(value).encode()).hexdigest()[:8]


def _counts(values: Iterable) -> dict[str, int]:
    return {str(key): int(value)
            for key, value in sorted(Counter(values).items(),
                                     key=lambda item: str(item[0]))}


def _summary(values: Iterable[float]) -> dict:
    data = [float(value) for value in values]
    if not data:
        return {"n": 0, "mean": None, "min": None, "max": None}
    return {"n": len(data), "mean": sum(data) / len(data),
            "min": min(data), "max": max(data)}


__all__ = [
    "BLOCK_SHIFTS",
    "CorpusStats",
    "build_shape_cases",
    "build_step_cases",
    "candidate_runs",
    "census",
    "instruction_spans",
    "marginal_distance",
    "recover_transform",
    "rotate_cases",
    "translation",
]
