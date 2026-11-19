"""Deterministic candidate fit fixture, not an authorized training card.

The fixed donor permutation is applied before auditing. An accidental relation
refuses the whole build; there is no unmatched donor redraw or seed search.
Pilot allocation and scientific-source exclusion must pass before card freeze.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from dm.data.augment import Affine, apply
from dm.data.relation import CopyAction, RelationCase
from dm.isa.spec import Op
from dm.relation.candidates import candidate_spans
from dm.relation.demonstrated import (
    NAMESPACE,
    TRANSFORMS,
    TaskInvalid,
    audit_causal_queries,
    audit_negative,
    audit_positive,
    counterfactual_first_copy,
    orbit_key,
    solve_prefix,
)
from dm.vm.interp import VM

BUILD_SEED = 20260912


@dataclass(frozen=True)
class FitFixture:
    cases: tuple[RelationCase, ...]
    donor_pairs: tuple[tuple[str, str], ...]
    sources: tuple[bytes, ...]


def _source(rng: random.Random, seen: set[bytes]) -> bytes:
    for _ in range(1000):
        points = tuple((rng.randrange(64, 192), rng.randrange(64, 192))
                       for _ in range(3))
        (x0, y0), (x1, y1), (x2, y2) = points
        if (x1 - x0) * (y2 - y0) == (x2 - x0) * (y1 - y0):
            continue
        body = bytes(v for op, point in zip((Op.MOVE, Op.LINE, Op.LINE), points)
                     for v in (op, *point))
        key = orbit_key(body)
        if key not in seen:
            seen.add(key)
            return body
    raise TaskInvalid("source allocation exhausted fixed 1000-attempt limit")


def _image(body, transform):
    image = apply(body, Affine.of(transform))
    if image is None:
        raise TaskInvalid("allocated source leaves canvas")
    return image


def _case(case_id, task_class, prefix, target, sources, transform=None,
          source_case_id="", donor_case_id="", donor_sources=()):
    flat = prefix + target + bytes([Op.HALT])
    groups = tuple(hashlib.sha256(orbit_key(source)).hexdigest() for source in sources)
    donor_groups = tuple(hashlib.sha256(orbit_key(source)).hexdigest()
                         for source in donor_sources)
    actions = (() if transform is None else
               (CopyAction(len(prefix), len(prefix) - 9, len(prefix), transform,
                           2, len(prefix), len(prefix) + len(target)),))
    return RelationCase(
        case_id=case_id, venue=NAMESPACE, stratum=task_class,
        flat=flat, structured=flat, groups=groups,
        plan=(() if transform is None else
              ((transform.d4.code, transform.dx, transform.dy, 2),)),
        actions=actions, target_start=len(prefix), target_stop=len(prefix) + len(target),
        source_ids=groups, source_case_id=source_case_id, donor_case_id=donor_case_id,
        donor_groups=donor_groups, donor_source_ids=donor_groups,
    )


def build_fit_fixture() -> FitFixture:
    rng = random.Random(BUILD_SEED)
    seen: set[bytes] = set()
    # Allocate independent source pairs before choosing any transform.
    sources = tuple(_source(rng, seen) for _ in range(64))
    transforms = list(TRANSFORMS)
    rng.shuffle(transforms)
    positives = []
    for index in range(16):
        a, b = sources[2 * index:2 * index + 2]
        transform = transforms[index]
        positives.append(_case(f"fit-positive-{index:02d}", "positive",
                               a + _image(a, transform) + b, _image(b, transform),
                               (a, b), transform))
    negatives, pairs = [], []
    # Predeclared adjacent swaps preserve length/opcode skeleton and yield four
    # donor components, rather than turning all controls into one giant cycle.
    for index in range(8):
        recipient, donor = positives[index], positives[index ^ 1]
        a, b = sources[2 * index:2 * index + 2]
        donor_sources = sources[2 * (index ^ 1):2 * (index ^ 1) + 2]
        negatives.append(_case(f"fit-broken-{index:02d}", "broken",
                               a + donor.prompt[9:18] + b, donor.target, (a, b),
                               source_case_id=recipient.case_id,
                               donor_case_id=donor.case_id, donor_sources=donor_sources))
        pairs.append((recipient.case_id, donor.case_id))
    for index in range(8):
        a, b, c, target = sources[32 + 4 * index:36 + 4 * index]
        # Same byte/instruction/candidate counts; two MOVE motifs instead of
        # demonstrated grammar's three. Class is visible in ordinary opcodes.
        prefix = a + bytes([Op.LINE]) + b[1:] + c
        negatives.append(_case(f"fit-generic-{index:02d}", "generic", prefix,
                               target, (a, b, c, target)))
    return FitFixture(tuple(positives + negatives), tuple(pairs), sources)


def audit_fit_fixture(fixture: FitFixture) -> dict:
    if len(fixture.cases) != 32:
        raise TaskInvalid("fit fixture must contain exactly 32 programs")
    rows = []
    bins = {"positive": {}, "negative": {}}
    for case in fixture.cases:
        if len(case.flat) + 1 > 256 or case.flat[case.target_stop:] != b"\x00":
            raise TaskInvalid(f"program capacity/termination mismatch: {case.case_id}")
        trace = VM().run(case.flat)
        if not trace.valid or not trace.halted:
            raise TaskInvalid(f"invalid full program: {case.case_id}")
        task_class = solve_prefix(case.prompt).task_class
        if task_class != case.stratum:
            raise TaskInvalid(f"prefix class differs from stored class: {case.case_id}")
        audit = (audit_positive(case.prompt, case.target) if case.actions else
                 audit_negative(case.prompt, case.target))
        key = f"{len(case.prompt)}:{len(candidate_spans(case.prompt, len(case.prompt)))}"
        arm = "positive" if case.actions else "negative"
        bins[arm][key] = bins[arm].get(key, 0) + 1
        rows.append({"case_id": case.case_id, "class": task_class,
                     "flat_hex": case.flat.hex(), "target_start": case.target_start,
                     "target_stop": case.target_stop, "source_ids": list(case.source_ids),
                     "source_case_id": case.source_case_id,
                     "donor_case_id": case.donor_case_id,
                     "donor_source_ids": list(case.donor_source_ids), **audit})
    if bins["positive"] != bins["negative"]:
        raise TaskInvalid("positive/negative prompt/candidate bins differ")
    # Common query motif, changed visible demonstration, changed target.
    a, b = fixture.sources[:2]
    counterfactuals = []
    for transform in TRANSFORMS[:3]:
        prefix = a + _image(a, transform) + b
        target = _image(b, transform)
        audit_positive(prefix, target)
        counterfactuals.append({"prefix_hex": prefix.hex(), "target_hex": target.hex()})
    if len({row["target_hex"] for row in counterfactuals}) != 3:
        raise TaskInvalid("counterfactual demonstrations did not change target")
    from dm.relation.wiring_spec import fixture as old_fixture

    old_case = old_fixture().cases[0]
    old_action = next(action for action in old_case.actions
                      if action.boundary == old_case.target_start)
    first_copy = counterfactual_first_copy(
        old_case.prompt, old_action.source_start, old_action.source_stop)
    first_copy["case_id"] = old_case.case_id
    first_copy["provenance"] = "unchanged wiring_spec.fixture row 0"
    return {"namespace": NAMESPACE, "build_seed": BUILD_SEED,
            "status": "candidate_fit_audit_only", "model_calls": 0,
            "optimizer_updates": 0, "rows": rows,
            "source_hex": [source.hex() for source in fixture.sources],
            "donor_pairs": list(fixture.donor_pairs), "matched_bins": bins,
            "causal_queries": audit_causal_queries(fixture.cases),
            "counterfactual_demonstrations": counterfactuals,
            "first_copy_control": first_copy,
            "rejection_ledger": [],
            "not_yet_audited": ["pilot allocation and source components",
                                "scientific corpus source exclusion",
                                "all-split causal compatibility"],
            "training_authorized_by_this_artifact": False}
