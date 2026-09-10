"""Per-row deterministic byte queues for mixed relation decodes."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable


class RowByteQueues:
    """Heterogeneous FIFO queues consumed one synchronized position at a time.

    A row without a queued byte yields ``None`` rather than a padding symbol.
    The decoder can then make its ordinary token/action decision for that row;
    padding never becomes causal context just to align a batch cache.
    """

    def __init__(self, rows: int) -> None:
        if rows < 1:
            raise ValueError("rows must be positive")
        self._queues = [deque() for _ in range(rows)]

    @property
    def rows(self) -> int:
        return len(self._queues)

    def pending(self, row: int) -> int:
        return len(self._queues[self._row(row)])

    def discard(self, row: int) -> int:
        """Remove residual bytes after a stop; return the actual removed count."""
        queue = self._queues[self._row(row)]
        count = len(queue)
        queue.clear()
        return count

    def any_pending(self) -> bool:
        return any(self._queues)

    def enqueue(self, row: int, block: bytes) -> None:
        if not block:
            raise ValueError("a COPY block must contain at least one byte")
        self._queues[self._row(row)].extend(block)

    def pop_position(self) -> tuple[int | None, ...] | None:
        """Consume one queued byte per row, preserving absent rows as ``None``."""
        if not self.any_pending():
            return None
        return tuple(queue.popleft() if queue else None for queue in self._queues)

    def drain(self, consume: Callable[[tuple[int | None, ...]], None]) -> None:
        """Call ``consume`` once for each synchronized copied-output position."""
        while (position := self.pop_position()) is not None:
            consume(position)

    def _row(self, row: int) -> int:
        if not 0 <= row < self.rows:
            raise IndexError(f"row {row} outside 0..{self.rows - 1}")
        return row
