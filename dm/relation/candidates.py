"""Target-free, prefix-only source-span enumeration for ``COPY``.

This is deliberately the one runtime counterpart of the corpus candidate
enumerator.  It uses :func:`dm.relation.transducer.prefix_index`, so malformed
prefixes and ``Kind.COORD`` interpretation cannot diverge from execution.
"""

from __future__ import annotations

from bisect import bisect_left

from ..isa.spec import Op
from .spec import RUNTIME_BOUNDS, RuntimeBounds
from .transducer import FaultCode, TransducerFault, prefix_index


def candidate_spans(prefix: bytes, boundary: int, *,
                    bounds: RuntimeBounds = RUNTIME_BOUNDS) -> tuple[tuple[int, int], ...]:
    """Enumerate frozen-order eligible source spans for one prefix boundary.

    ``prefix`` is immutable generated L0 bytecode.  Unlike the historical
    corpus wrapper, malformed input is a typed runtime failure: an inference
    caller must not turn a corrupt prefix into an apparently valid ``EMIT``
    decision.  The corpus compatibility wrapper preserves its legacy empty-set
    behavior where appropriate.
    """
    if type(boundary) is not int:
        raise TransducerFault(FaultCode.BOUNDARY, "candidate boundary is not an integer")
    marks, coords, ops = prefix_index(prefix)
    if boundary not in marks:
        return ()
    query_index = marks.index(boundary)
    halt_offsets = tuple(
        marks[index] for index, op in enumerate(ops) if op is Op.HALT
    )
    out: list[tuple[int, int]] = []
    for gap in range(bounds.max_source_gap_instructions + 1):
        stop_index = query_index - gap
        if stop_index < bounds.min_source_instructions:
            break
        stop = marks[stop_index]
        for start_index in range(stop_index - bounds.min_source_instructions, -1, -1):
            if stop_index - start_index > bounds.max_source_instructions:
                break
            start = marks[start_index]
            width = stop - start
            if width > bounds.max_source_bytes:
                break
            if width < bounds.min_source_bytes:
                continue
            if bisect_left(halt_offsets, stop) > bisect_left(halt_offsets, start):
                continue
            if (bounds.min_source_coords and
                    bisect_left(coords, stop) - bisect_left(coords, start)
                    < bounds.min_source_coords):
                continue
            out.append((start, stop))
    return tuple(out)
