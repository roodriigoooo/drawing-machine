"""Fixed model-blind pilot allocation for demonstrated-affine I0.

One build seed, adjacent donor swaps, no redraw of failed controls. A refusal
is evidence about this candidate construction, never about learned capacity.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, replace

from dm.data.relation import RelationCase, component_of, components
from dm.relation.candidates import candidate_spans
from dm.relation.demonstrated import (
    TRANSFORMS,
    TaskInvalid,
    audit_causal_queries,
    audit_negative,
    audit_positive,
    orbit_key,
    solve_prefix,
)
from dm.relation.demonstrated_fixture import _case, _image, _source, build_fit_fixture
from dm.vm.interp import VM

PILOT_SEED = 20260913


@dataclass(frozen=True)
class PilotFixture:
    panels: tuple[tuple[str, tuple[RelationCase, ...]], ...]
    sources: tuple[bytes, ...]
    training_transforms: tuple
    heldout_transforms: tuple

    @property
    def cases(self):
        return tuple(case for _, cases in self.panels for case in cases)


def build_pilot_fixture() -> PilotFixture:
    rng = random.Random(PILOT_SEED)
    seen = {orbit_key(source) for source in build_fit_fixture().sources}
    # 64 training pairs, 32 held-out pairs, 320 independent generic quadruples.
    sources = tuple(_source(rng, seen) for _ in range(192 + 320 * 4))
    held = tuple(t for t in TRANSFORMS if t.dx == t.dy == 0)
    trained = tuple(t for t in TRANSFORMS if t not in held)
    # Independent permutations assign tuples to each preallocated family.
    orders = [rng.sample(trained, len(trained)) for _ in range(96)]

    def positive(panel, index, family, transform):
        a, b = sources[2 * family:2 * family + 2]
        return _case(f"{panel}-positive-{index:04d}", "positive",
                     a + _image(a, transform) + b, _image(b, transform), (a, b), transform)

    train = [positive("train", i, i % 64, orders[i % 64][i // 64])
             for i in range(512)]
    iid = [positive("iid", i, i, orders[i][8]) for i in range(64)]
    affine = [positive("affine", i, i, held[i % len(held)]) for i in range(64)]
    family = [positive("family", i, 64 + i % 32, orders[64 + i % 32][i // 32])
              for i in range(64)]

    def broken(panel, positives):
        rows = []
        for index, recipient in enumerate(positives):
            donor = positives[index ^ 1]
            a, b = recipient.prompt[:9], recipient.prompt[18:27]
            rows.append(_case(f"{panel}-broken-{index:04d}", "broken",
                              a + donor.prompt[9:18] + b, donor.target, (a, b),
                              source_case_id=recipient.case_id, donor_case_id=donor.case_id,
                              donor_sources=(donor.prompt[:9], donor.prompt[18:27])))
        return rows

    def generic(panel, start, count):
        rows = []
        for index in range(count):
            offset = 192 + 4 * (start + index)
            a, b, c, target = sources[offset:offset + 4]
            rows.append(_case(f"{panel}-generic-{index:04d}", "generic",
                              a + b"\x02" + b[1:] + c, target, (a, b, c, target)))
        return rows

    return PilotFixture((
        ("train", tuple(train + broken("train", train[:256]) + generic("train", 0, 256))),
        ("iid", tuple(iid)), ("affine", tuple(affine)), ("family", tuple(family)),
        ("negative", tuple(broken("negative", iid) + generic("negative", 256, 64))),
    ), sources, trained, held)


def audit_allocation(fixture: PilotFixture) -> dict:
    panels = dict(fixture.panels)
    if {name: len(rows) for name, rows in panels.items()} != {
            "train": 1024, "iid": 64, "affine": 64, "family": 64, "negative": 128}:
        raise TaskInvalid("pilot panel sizes differ from specification")
    train = panels["train"]
    tuples = {c.plan[0][:3] for c in train if c.actions}
    held = {c.plan[0][:3] for c in panels["affine"]}
    if tuples & held:
        raise TaskInvalid("held-out affine tuple leaked into training")
    if any(not {t[i] for t in held} <= {t[i] for t in tuples} for i in range(3)):
        raise TaskInvalid("held-out affine atom absent from training")
    train_sources = {s for c in train for s in c.source_ids}
    novel_sources = {s for c in panels["family"] for s in c.source_ids}
    if train_sources & novel_sources:
        raise TaskInvalid("held-out source family leaked into training")
    for name in ("iid", "affine"):
        if not {s for c in panels[name] for s in c.source_ids} <= train_sources:
            raise TaskInvalid(f"{name} panel uses untrained source family")
    if any(c.plan[0][:3] not in tuples for c in panels["iid"] + panels["family"]):
        raise TaskInvalid("trained-tuple panel uses untrained tuple")
    if {c.prompt for c in train} & {c.prompt for name, rows in fixture.panels
                                   if name != "train" for c in rows}:
        raise TaskInvalid("exact training prefix leaked into evaluation")
    labels = components(tuple(replace(c, groups=tuple(dict.fromkeys(c.groups + c.donor_groups)))
                              for c in fixture.cases))
    counts = {name: len({component_of(c, labels) for c in panels[name]})
              for name in ("iid", "affine", "family")}
    if min(counts.values()) < 32:
        raise TaskInvalid(f"too few source/donor components: {counts}")
    fit_keys = {orbit_key(source) for source in build_fit_fixture().sources}
    if fit_keys & {orbit_key(source) for source in fixture.sources}:
        raise TaskInvalid("pilot source overlaps fit orbit")
    bins = {}
    for name, rows in fixture.panels:
        bins[name] = {}
        for case in rows:
            key = f"{len(case.prompt)}:{len(candidate_spans(case.prompt, len(case.prompt)))}"
            bins[name][key] = bins[name].get(key, 0) + 1
    return {"panel_sizes": {name: len(rows) for name, rows in fixture.panels},
            "positive_panel_components": counts, "prompt_candidate_bins": bins,
            "training_tuple_count": len(tuples), "heldout_tuple_count": len(held),
            "fit_source_orbit_overlap": 0}


def audit_pilot(fixture: PilotFixture) -> dict:
    allocation = audit_allocation(fixture)
    checked, witnesses = [], []
    # Preserve every failure from this one fixed construction, without replacement.
    for panel, cases in fixture.panels:
        for case in cases:
            try:
                if len(case.flat) + 1 > 256 or case.flat[case.target_stop:] != b"\x00":
                    raise TaskInvalid("program capacity/termination mismatch")
                trace = VM().run(case.flat)
                if not trace.valid or not trace.halted:
                    raise TaskInvalid("invalid full program")
                if solve_prefix(case.prompt).task_class != case.stratum:
                    raise TaskInvalid("prefix-only class differs from stored class")
                if case.actions:
                    audit_positive(case.prompt, case.target)
                else:
                    audit_negative(case.prompt, case.target)
                checked.append(case.case_id)
            except TaskInvalid as exc:
                witnesses.append({"panel": panel, "case_id": case.case_id,
                                  "reason": str(exc), "prompt_hex": case.prompt.hex(),
                                  "target_hex": case.target.hex()})
    try:
        causal = audit_causal_queries(build_fit_fixture().cases + fixture.cases)
    except TaskInvalid as exc:
        causal = {"failure": str(exc)}
        witnesses.append({"check": "all_split_causal_queries", "reason": str(exc)})
    return {"status": "task_invalid" if witnesses else "candidate_pilot_audit_only",
            "allocation": allocation, "passed_rows": len(checked),
            "rejection_ledger": witnesses, "causal_queries": causal,
            "scientific_source_exclusion": "not_yet_audited",
            "model_calls": 0, "optimizer_updates": 0}
