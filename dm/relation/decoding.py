"""Target-free common byte policy and incremental flat-output stop state.

Generated failures are retained outputs, not relaxed runtime prefixes. This
adapter classifies them before the strict candidate/transducer APIs run.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..isa.codec import N_SPECIAL
from ..isa.spec import ISAError, Op, Tier, spec_for

CONTENT_POLICY = "content_bytes_v1"
LEGACY_POLICY = "legacy_codec_v1"
STOPPING_POLICY = "flat_l0_stop_v1"


class DecodeRequestError(ValueError):
    """A common-policy request failed preflight."""


def resolve_policy(mode: str, policy: str | None) -> str:
    if policy is None:
        policy = LEGACY_POLICY if mode == "standard" else CONTENT_POLICY
    if policy not in (LEGACY_POLICY, CONTENT_POLICY):
        raise DecodeRequestError(f"unknown literal policy {policy!r}")
    if policy == LEGACY_POLICY and mode != "standard":
        raise DecodeRequestError("COPY modes refuse legacy_codec_v1")
    return policy


class FlatOutputStop:
    """O(rows) ISA state, shared by delegated standard and COPY decoding.

    External monitors are stop-only, synchronous and row aligned. Their status
    must be monotone even on rows stopped by the built-in rule.
    """

    stride = 1

    def __init__(self, rows: int, prompt_length: int, horizon: int, external=None):
        if external is not None:
            if getattr(external, "stride", None) != 1 or not callable(
                    getattr(external, "step", None)):
                raise DecodeRequestError("common monitor needs stride 1 and step")
            if hasattr(external, "allowed"):
                raise DecodeRequestError("common monitor refuses allowed capability")
        self.external = external
        self.given = prompt_length
        self.width = prompt_length + horizon
        self.position = 0
        self.remaining = [0] * rows
        self.done = np.zeros(rows, dtype=bool)
        self.external_done = np.zeros(rows, dtype=bool)
        self.causes: list[str | None] = [None] * rows
        self.stop_positions: list[int | None] = [None] * rows
        self.useful = [prompt_length] * rows
        self.external_flags = [False] * rows
        self.horizon_flags = [False] * rows
        self.prompt_terminal = [False] * rows

    def step(self, chunk: np.ndarray) -> np.ndarray:
        if chunk.shape != (1, len(self.done)):
            raise ValueError("stop chunk must be one row-aligned byte position")
        live = ~self.done.copy()
        for row in np.flatnonzero(live):
            byte = int(chunk[0, row]) - N_SPECIAL
            if not 0 <= byte <= 255:
                raise ValueError("common output contains a non-content symbol")
            if self.position >= self.given:
                self.useful[row] += 1
            cause = None
            if self.remaining[row]:
                self.remaining[row] -= 1
            else:
                try:
                    spec = spec_for(byte)
                except ISAError:
                    cause = "invalid_opcode"
                else:
                    if spec.op is Op.HALT:
                        cause = "prompt_halt" if self.position < self.given else "halt"
                        self.prompt_terminal[row] = self.position < self.given
                    elif spec.tier is not Tier.L0:
                        cause = "non_flat_opcode"
                    else:
                        self.remaining[row] = spec.size - 1
            if cause is not None:
                self.causes[row] = cause
                self.stop_positions[row] = self.position
                self.done[row] = True
        if self.external is not None:
            status = self.external.step(chunk.copy())
            if not isinstance(status, np.ndarray) or status.dtype != np.bool_ or \
                    status.shape != self.done.shape:
                raise ValueError("external monitor status must be row-aligned boolean ndarray")
            if bool((self.external_done & ~status).any()):
                raise ValueError("external monitor status must be monotone")
            self.external_done = status.copy()
            for row in np.flatnonzero(live & status):
                self.external_flags[row] = True
                if not self.done[row]:
                    self.causes[row] = "external_monitor_stop"
                    self.stop_positions[row] = self.position
                    self.done[row] = True
        if self.position + 1 == self.width and self.position >= self.given:
            for row in np.flatnonzero(live):
                self.horizon_flags[row] = True
                if not self.done[row]:
                    self.causes[row] = ("horizon_partial" if self.remaining[row]
                                        else "horizon_boundary")
                    self.stop_positions[row] = self.position
                    self.done[row] = True
        self.position += 1
        return self.done.copy()


# Record payloads contain only owned immutable primitives. Exact field sets are
# frozen here, independently of any evidence/report schema.
@dataclass(frozen=True, slots=True)
class RequestStarted:
    mode: str
    literal_policy: str
    stopping_policy: str
    sampler_method: str
    rows: int
    prompt_length: int
    horizon: int
    capacity: int
    observation_level: str
    kind: str = "request_started"


@dataclass(frozen=True, slots=True)
class ActionDecision:
    row: int
    byte_offset: int
    action_source: str
    candidate_count: int
    action_key: tuple[int, ...] | None
    decision_score: float | None
    executor_attempts: int
    faults: tuple[tuple[str, int], ...]
    exit_reason: str
    admission_bytes: int
    kind: str = "action_decision"


@dataclass(frozen=True, slots=True)
class ScoreDetail:
    row: int
    byte_offset: int
    endpoints: tuple[tuple[int, int], ...]
    gate: tuple[float, ...]
    span: tuple[float, ...]
    d4: tuple[float, ...]
    dx: tuple[float, ...]
    dy: tuple[float, ...]
    count: tuple[float, ...]
    supports: tuple[tuple[str, tuple[int, ...]], ...]
    normalization: str = "raw_logits;normalization=float32_log_softmax_per_factor_real_spans"
    kind: str = "score_detail"


@dataclass(frozen=True, slots=True)
class RowStopped:
    row: int
    useful_length: int
    generated_bytes: int
    cause: str
    byte_offset: int | None
    pending_before_discard: int
    discarded_bytes: int
    external_monitor_stop: bool
    at_horizon: bool
    prompt_halt: bool
    kind: str = "row_stopped"


ROW_COUNTERS = (
    "generated_bytes", "literal_decisions", "sampler_rows", "uniform_coordinates_read",
    "discarded_sample_rows", "head_queries", "candidate_entries", "executor_attempts",
    "predicted_search_attempts", "oracle_executor_calls", "copy_admissions",
    "copy_admitted_bytes", "copy_delivered_bytes", "copy_pending_bytes", "copy_discarded_bytes",
    "prefill_row_positions", "incremental_row_positions", "fed_literal_bytes", "fed_copy_bytes",
    "fed_pad_symbols", "fed_live_row_positions", "fed_stopped_row_positions",
    "boundary_slots_used",
)
BATCH_COUNTERS = ("sampler_calls", "head_batches", "prefill_calls", "incremental_calls")


@dataclass(frozen=True, slots=True)
class RequestCompleted:
    per_row: tuple[tuple[tuple[str, int], ...], ...]
    totals: tuple[tuple[str, int], ...]
    statuses: tuple[RowStopped, ...]
    returned_width: int
    kv_allocated_tensor_bytes: int
    boundary_value_storage_bytes: int
    boundary_index_entries: int
    kind: str = "request_completed"


DecodeEvent = RequestStarted | ActionDecision | ScoreDetail | RowStopped | RequestCompleted
DecodeObserver = Callable[[DecodeEvent], None]


def reconcile(counters: dict[str, int], explicit_uniforms: bool) -> None:
    """Logical units, not FLOPs, physical memory, or sequential PRNG draws."""
    c = counters
    identities = (
        (c["generated_bytes"], c["literal_decisions"] + c["copy_delivered_bytes"]),
        (c["copy_admitted_bytes"], c["copy_delivered_bytes"] + c["copy_pending_bytes"]
         + c["copy_discarded_bytes"]),
        (c["sampler_rows"], c["literal_decisions"] + c["discarded_sample_rows"]),
        (c["uniform_coordinates_read"], c["sampler_rows"] if explicit_uniforms else 0),
        (c["executor_attempts"], c["predicted_search_attempts"] + c["oracle_executor_calls"]),
        (c["incremental_row_positions"], c["fed_literal_bytes"] + c["fed_copy_bytes"]
         + c["fed_pad_symbols"]),
        (c["incremental_row_positions"], c["fed_live_row_positions"]
         + c["fed_stopped_row_positions"]),
        (c["copy_pending_bytes"], 0),
    )
    if any(left != right for left, right in identities):
        raise RuntimeError("decode work ledger does not reconcile")


class DecodeLedger:
    """Request-local O(rows) counters. No event history, tensors or targets."""

    def __init__(self, rows: int, given: int, explicit: bool,
                 observer: DecodeObserver | None, level: str):
        self.rows = [{name: 0 for name in ROW_COUNTERS} for _ in range(rows)]
        self.calls = dict.fromkeys(BATCH_COUNTERS, 0)
        self.statuses: list[RowStopped | None] = [None] * rows
        self.given = given
        self.explicit = explicit
        self.observer = observer
        self.level = level
        self.kv_bytes = 0
        self.boundary_bytes = 0

    def publish(self, event: DecodeEvent) -> None:
        if self.observer is not None:
            self.observer(event)

    def add(self, row: int, name: str, count: int = 1) -> None:
        self.rows[row][name] += count

    def sampled(self, sampled_rows, live) -> None:
        if not sampled_rows:
            return
        self.calls["sampler_calls"] += 1
        for row in sampled_rows:
            self.add(row, "sampler_rows")
            self.add(row, "uniform_coordinates_read", int(self.explicit))
            self.add(row, "literal_decisions" if live[row] else "discarded_sample_rows")

    def forwarded(self, prefill: bool, origins: tuple[str, ...], done) -> None:
        self.calls["prefill_calls" if prefill else "incremental_calls"] += 1
        for row, origin in enumerate(origins):
            if prefill:
                self.add(row, "prefill_row_positions", self.given + 1)
            else:
                self.add(row, "incremental_row_positions")
                self.add(row, "fed_" + origin + ("_symbols" if origin == "pad" else "_bytes"))
                self.add(row, "fed_stopped_row_positions" if done[row]
                         else "fed_live_row_positions")

    def stopped(self, stop: FlatOutputStop, row: int, discarded: int = 0) -> None:
        if self.statuses[row] is not None:
            return
        cause = stop.causes[row]
        if cause is None:
            raise RuntimeError("stopped row has no classified cause")
        status = RowStopped(row, stop.useful[row], self.rows[row]["generated_bytes"],
                            cause, stop.stop_positions[row], discarded, discarded,
                            stop.external_flags[row], stop.horizon_flags[row],
                            stop.prompt_terminal[row])
        self.statuses[row] = status
        self.publish(status)

    def complete(self, width: int) -> None:
        for counters in self.rows:
            reconcile(counters, self.explicit)
        if any(status is None for status in self.statuses):
            raise RuntimeError("completed request has a live row")
        totals = {name: sum(row[name] for row in self.rows) for name in ROW_COUNTERS}
        totals.update(self.calls)
        reconcile(totals, self.explicit)
        self.publish(RequestCompleted(
            tuple(tuple(row.items()) for row in self.rows), tuple(totals.items()),
            tuple(self.statuses), width, self.kv_bytes, self.boundary_bytes,
            totals["boundary_slots_used"]))
