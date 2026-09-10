"""Audits that make a paired feedback corpus's semantic contents inspectable.

The corpus builder owns byte-level construction and exact marginal checks.  VM
execution is a separate audit seam: it is useful evidence about the resulting
programs, but it must not participate in donor selection or make construction
model-aware.  The audit therefore accepts finished program arms and returns a
JSON-safe census without changing either arm.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence

from ..vm.interp import VM

AUDIT_SCHEMA = 1

_ARM_FIELDS = (
    "programs", "halted_programs", "valid_programs", "valid_halted_programs",
    "faulted_programs", "fault_events", "faults", "programs_with_strokes",
    "total_strokes", "mean_strokes", "min_strokes", "max_strokes",
    "stroke_count_histogram", "vm_steps",
)


def _arm_census(programs: Sequence[bytes]) -> dict:
    """Count VM faults and emitted strokes for one complete program arm."""
    vm = VM()
    faults: Counter[str] = Counter()
    stroke_histogram: Counter[str] = Counter()
    halted = 0
    valid = 0
    valid_halted = 0
    programs_with_strokes = 0
    total_strokes = 0
    total_steps = 0

    for program in programs:
        trace = vm.run(program)
        halted += int(trace.halted)
        valid += int(trace.valid)
        valid_halted += int(trace.valid and trace.halted)
        faults.update(fault.kind.value for fault in trace.faults)
        strokes = len(trace.strokes)
        stroke_histogram[str(strokes)] += 1
        programs_with_strokes += int(strokes > 0)
        total_strokes += strokes
        total_steps += trace.steps

    count = len(programs)
    return {
        "programs": count,
        "halted_programs": halted,
        "valid_programs": valid,
        "valid_halted_programs": valid_halted,
        "faulted_programs": count - valid,
        "fault_events": sum(faults.values()),
        "faults": dict(sorted(faults.items())),
        "programs_with_strokes": programs_with_strokes,
        "total_strokes": total_strokes,
        "mean_strokes": total_strokes / max(1, count),
        "min_strokes": min((int(value) for value in stroke_histogram), default=0),
        "max_strokes": max((int(value) for value in stroke_histogram), default=0),
        "stroke_count_histogram": dict(sorted(stroke_histogram.items(),
                                               key=lambda item: int(item[0]))),
        "vm_steps": total_steps,
    }


def vm_census(arms: Mapping[str, Sequence[bytes]]) -> dict:
    """Return the full-corpus VM fault/stroke census for named arms.

    ``relational`` and ``destroyed`` are compared as a paired audit when both
    are present.  Equality here is evidence that the VM-level fault/stroke
    marginals were preserved; it is not a claim that seam bigrams or local
    continuity were preserved, because those are intentionally changed by the
    intervention.
    """
    if not arms:
        raise ValueError("VM census needs at least one program arm: incomplete")
    named = {str(name): _arm_census(programs)
             for name, programs in arms.items()}
    paired = None
    if {"relational", "destroyed"} <= set(named):
        left, right = named["relational"], named["destroyed"]
        paired = {
            "fault_census_equal": left["faults"] == right["faults"],
            "stroke_census_equal": (
                left["stroke_count_histogram"]
                == right["stroke_count_histogram"]
            ),
        }
    return {
        "schema": AUDIT_SCHEMA,
        "arms": named,
        "paired_relational_destroyed": paired,
    }


def audit_accepts(census: dict) -> bool:
    """Whether a persisted census is complete, clean and paired.

    **Three separate questions, and the last two were missing.** Completeness is
    internal arithmetic: the histogram sums to the program count, the fault
    counts sum to the event count, the arms name every field.  Pairing is that
    the intervention left the relational and destroyed VM marginals equal.
    Neither says the corpus is *good*: a control in which every program faulted
    identically satisfies both, and the two arms would then differ in the
    relation and in being broken.

    So the census must also be clean: every arm all-valid, all-halted and
    zero-fault.  That is what the audited Direction 3 build actually is
    (`docs/directions.md` §F4), and making it an acceptance rule rather than a
    reported field is what turns it into a freeze failure instead of a footnote.
    """
    if not isinstance(census, dict) or census.get("schema") != AUDIT_SCHEMA:
        return False
    arms = census.get("arms")
    if not isinstance(arms, dict) or not {
            "relational", "destroyed", "validation"
    } <= set(arms):
        return False
    for arm in arms.values():
        if not isinstance(arm, dict) or not set(_ARM_FIELDS) <= set(arm):
            return False
        integer_fields = {
            field for field in _ARM_FIELDS
            if field not in {"mean_strokes", "faults", "stroke_count_histogram"}
        }
        if any(not isinstance(arm[field], int) or isinstance(arm[field], bool)
               or arm[field] < 0 for field in integer_fields):
            return False
        if not isinstance(arm["mean_strokes"], (int, float)) \
                or not math.isfinite(float(arm["mean_strokes"])):
            return False
        faults = arm["faults"]
        histogram = arm["stroke_count_histogram"]
        if not isinstance(faults, dict) or not isinstance(histogram, dict):
            return False
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0
               for value in faults.values()):
            return False
        if any(not isinstance(key, str) or not key.isdigit()
               or not isinstance(value, int) or isinstance(value, bool) or value < 0
               for key, value in histogram.items()):
            return False
        if sum(histogram.values()) != arm["programs"]:
            return False
        if sum(int(key) * value for key, value in histogram.items()) != arm[
                "total_strokes"]:
            return False
        if sum(faults.values()) != arm["fault_events"]:
            return False
        if arm["valid_programs"] + arm["faulted_programs"] != arm["programs"]:
            return False
        if arm["valid_halted_programs"] > arm["valid_programs"] \
                or arm["halted_programs"] > arm["programs"]:
            return False
        if not math.isclose(
                float(arm["mean_strokes"]),
                arm["total_strokes"] / max(1, arm["programs"]),
                rel_tol=0.0, abs_tol=1e-12):
            return False
        # Clean, not merely self-consistent.
        if arm["faults"] or arm["fault_events"] or arm["faulted_programs"]:
            return False
        if arm["valid_programs"] != arm["programs"]:
            return False
        if arm["halted_programs"] != arm["programs"]:
            return False
        if arm["valid_halted_programs"] != arm["programs"]:
            return False
    paired = census.get("paired_relational_destroyed")
    return paired == {"fault_census_equal": True, "stroke_census_equal": True}


__all__ = ["AUDIT_SCHEMA", "audit_accepts", "vm_census"]
