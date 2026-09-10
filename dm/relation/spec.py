"""Target-free values and frozen runtime bounds for ``COPY``.

The values here are intentionally independent of :mod:`dm.data.relation`.
They describe what an executor may do from an already-generated prefix, not how
the corpus proved that an action is a correct annotation.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..isa.transform import D4_ORDER, Transform

# These values were frozen by R1 and are re-exported by dm.data.relation for
# compatibility.  The runtime owns them so decoding does not import the corpus.
D4_SUPPORT: tuple[int, ...] = tuple(range(D4_ORDER))
TRANSLATION_SUPPORT: tuple[int, ...] = (-32, 0, 32)
COUNT_SUPPORT: tuple[int, ...] = (2, 3, 4)

# Counts 5 and 6 are the explicitly declared numerical extrapolation diagnostic
# from the Direction 4 contract.  This is a finite policy, never an "anything
# larger than training" escape hatch.
ORACLE_COUNT_SUPPORT: tuple[int, ...] = (2, 3, 4, 5, 6)


@dataclass(frozen=True)
class RuntimeBounds:
    """Model-blind structural limits an action must satisfy at execution time."""

    max_source_gap_instructions: int = 4
    max_source_instructions: int = 64
    max_source_bytes: int = 192
    min_source_instructions: int = 2
    min_source_bytes: int = 6
    min_source_coords: int = 1


RUNTIME_BOUNDS = RuntimeBounds()


@dataclass(frozen=True)
class SupportPolicy:
    """The factor support for one caller of the common executor."""

    name: str
    d4_codes: tuple[int, ...]
    translations: tuple[int, ...]
    total_counts: tuple[int, ...]
    bounds: RuntimeBounds = RUNTIME_BOUNDS


PREDICTED_SUPPORT = SupportPolicy(
    name="predicted_v1",
    d4_codes=D4_SUPPORT,
    translations=TRANSLATION_SUPPORT,
    total_counts=COUNT_SUPPORT,
)
ORACLE_SUPPORT = SupportPolicy(
    name="oracle_v1",
    d4_codes=D4_SUPPORT,
    translations=TRANSLATION_SUPPORT,
    total_counts=ORACLE_COUNT_SUPPORT,
)


@dataclass(frozen=True)
class CopyActionKey:
    """A runtime ``COPY`` request, with no target or corpus provenance.

    ``boundary`` is retained to make a key self-describing in logs and to stay
    compatible with the frozen R1 serialized key.  The executor requires it to
    equal ``len(prefix)``; it never receives a target range or target bytes.
    """

    boundary: int
    source_start: int
    source_stop: int
    step: Transform
    total_count: int

    def key(self) -> tuple[int, int, int, int, int, int, int]:
        """The R1 key shape, retained for stable equivalence-set comparison."""
        return (self.boundary, self.source_start, self.source_stop,
                self.step.d4.code, self.step.dx, self.step.dy, self.total_count)
