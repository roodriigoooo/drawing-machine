"""Opt-in CPU physical readings, never deterministic work or resume state.

Three peak statements are distinguished and never conflated:

* ``rss_end_bytes`` is one endpoint sample. A transient allocation that is
  released before the phase ends is invisible to it.
* ``peak_lower_bound_bytes`` is the largest resident size actually observed
  inside the phase (both endpoints plus any in-phase samples). Sampling is
  periodic, so a peak between two samples is missed; this is a lower bound.
* A process high-water rise is observed over the wider memory window, from
  scope setup through sampler teardown. It includes probes, instrumentation and
  unrelated process activity in that window. It is not a workload-owned peak.

Subtracting high-water endpoints is not a phase peak and is never reported.
Neither quantity isolates tensor storage, Python retention or allocator
overhead. Use fresh processes for comparable whole-run high-water measurements.

Collectors are synchronous and RNG-pure, like decoder observers. No history is
retained here. CPU wall time is the work body only. The memory window includes
endpoint probes and sampler lifecycle. A surrounding scope includes all nested
instrumentation overhead in its wall body. Do not use this timer for
asynchronous accelerator work.
"""

from __future__ import annotations

import ctypes
import math
import os
import resource
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from itertools import count
from pathlib import Path

#: ``PROC_PIDTASKINFO`` from ``<sys/proc_info.h>``.
_PROC_PIDTASKINFO = 4
WALL_WINDOW_WORK_BODY_V2 = "work_body_only_v2"
MEMORY_WINDOW_SCOPE_V2 = "scope_setup_to_teardown_v2"
_ACTIVE_SCOPE: ContextVar[int | None] = ContextVar("r4_resource_scope", default=None)
_SCOPE_IDS = count(1)


class PhysicalObservationRefused(ValueError):
    """A physical record contradicts the declared probe semantics."""


class _MacTaskInfo(ctypes.Structure):
    """``struct proc_taskinfo``; declared in full so its size check is real."""

    _fields_ = (
        ("pti_virtual_size", ctypes.c_uint64),
        ("pti_resident_size", ctypes.c_uint64),
        ("pti_total_user", ctypes.c_uint64),
        ("pti_total_system", ctypes.c_uint64),
        ("pti_threads_user", ctypes.c_uint64),
        ("pti_threads_system", ctypes.c_uint64),
        ("pti_policy", ctypes.c_int32),
        ("pti_faults", ctypes.c_int32),
        ("pti_pageins", ctypes.c_int32),
        ("pti_cow_faults", ctypes.c_int32),
        ("pti_messages_sent", ctypes.c_int32),
        ("pti_messages_received", ctypes.c_int32),
        ("pti_syscalls_mach", ctypes.c_int32),
        ("pti_syscalls_unix", ctypes.c_int32),
        ("pti_csw", ctypes.c_int32),
        ("pti_threadnum", ctypes.c_int32),
        ("pti_numrunning", ctypes.c_int32),
        ("pti_priority", ctypes.c_int32),
    )


def _load_proc_pidinfo():
    if sys.platform != "darwin":
        return None
    handle = ctypes.CDLL(None, use_errno=True)
    if not hasattr(handle, "proc_pidinfo"):
        return None
    symbol = handle.proc_pidinfo
    symbol.restype = ctypes.c_int
    symbol.argtypes = (ctypes.c_int, ctypes.c_uint32, ctypes.c_uint64,
                       ctypes.c_void_p, ctypes.c_int)
    return symbol


_PROC_PIDINFO = _load_proc_pidinfo()


@dataclass(frozen=True)
class PhaseResources:
    """One measured CPU scope. Physical, process-local and never resume state."""

    phase: str
    completed: bool
    wall_seconds: float
    rss_start_bytes: int
    rss_end_bytes: int
    process_high_water_start_bytes: int
    process_high_water_end_bytes: int
    #: Largest resident size *observed*, at an endpoint or an in-phase sample.
    #: `peak_lower_bound_bytes` tightens this with an exact isolated peak.
    rss_peak_observed_bytes: int
    #: In-phase samples taken; ``0`` means endpoint readings only.
    samples: int
    #: Requested sampling period, or ``None`` when sampling was disabled.
    sample_interval_seconds: float | None
    #: Identity links nested records without retaining a full event trace.
    scope_id: int = 0
    parent_scope_id: int | None = None
    #: Versioned definitions carried into physical evidence.
    wall_window: str = WALL_WINDOW_WORK_BODY_V2
    memory_window: str = MEMORY_WINDOW_SCOPE_V2
    work_start_seconds: float | None = None
    work_end_seconds: float | None = None

    def validate(self, *, require_scope: bool = False) -> None:
        """Refuse contradictory readings rather than reconciling them by clamp."""
        if type(self.phase) is not str or not self.phase:
            raise PhysicalObservationRefused("resource phase must have a nonempty string name")
        if type(self.completed) is not bool or not math.isfinite(self.wall_seconds) or \
                self.wall_seconds < 0:
            raise PhysicalObservationRefused("resource phase completion/wall reading is invalid")
        if any(type(value) is not int or value < 0 for value in (
                self.rss_start_bytes, self.rss_end_bytes, self.process_high_water_start_bytes,
                self.process_high_water_end_bytes, self.rss_peak_observed_bytes, self.samples)):
            raise PhysicalObservationRefused("resource memory readings must be nonnegative integers")
        if self.process_high_water_end_bytes < self.process_high_water_start_bytes:
            raise PhysicalObservationRefused("process high-water decreased inside one observation window")
        if self.rss_peak_observed_bytes < max(self.rss_start_bytes, self.rss_end_bytes):
            raise PhysicalObservationRefused("observed RSS peak is below an endpoint")
        if self.rss_peak_observed_bytes > self.process_high_water_end_bytes:
            raise PhysicalObservationRefused(
                "observed RSS peak exceeds the process high-water observation")
        if self.samples > 0 or self.sample_interval_seconds is not None:
            validate_sample_interval(self.sample_interval_seconds)
        if self.wall_window != WALL_WINDOW_WORK_BODY_V2 or \
                self.memory_window != MEMORY_WINDOW_SCOPE_V2:
            raise PhysicalObservationRefused("resource record uses an unknown observation-window version")
        if require_scope:
            start, end = self.work_start_seconds, self.work_end_seconds
            if start is None or end is None or not math.isfinite(start) or not math.isfinite(end) or \
                    start < 0 or end < start or not math.isclose(
                        end - start, self.wall_seconds, rel_tol=1e-9, abs_tol=1e-9):
                raise PhysicalObservationRefused("resource work interval disagrees with its duration")
        if require_scope or self.scope_id:
            if type(self.scope_id) is not int or self.scope_id < 1:
                raise PhysicalObservationRefused("resource scope identity is invalid")
            if self.parent_scope_id is not None and (
                    type(self.parent_scope_id) is not int or self.parent_scope_id < 1 or
                    self.parent_scope_id == self.scope_id):
                raise PhysicalObservationRefused("resource parent scope identity is invalid")

    @property
    def peak_lower_bound_bytes(self) -> int:
        """Tightest proven lower bound on this phase's peak resident size.

        Observation alone can miss a peak between two samples. A process
        high-water rise observed in this scope's memory window is also a
        process-level observation and tightens this lower bound; it does not
        establish allocation ownership.
        """
        return max(self.rss_peak_observed_bytes, self.process_high_water_rise_bytes or 0)

    @property
    def peak_upper_bound_bytes(self) -> int:
        """Process high-water at the end of this scope's memory window."""
        return self.process_high_water_end_bytes

    @property
    def process_high_water_rise_bytes(self) -> int | None:
        """A process high-water observed to rise in this record's memory window."""
        end = self.process_high_water_end_bytes
        return end if end > self.process_high_water_start_bytes else None

    @property
    def isolated_peak_bytes(self) -> int | None:
        """Compatibility alias; never interpret as a workload-isolated peak."""
        return self.process_high_water_rise_bytes


ResourceObserver = Callable[[PhaseResources], None]


def _rss_bytes() -> int:
    if sys.platform.startswith("linux"):
        # statm's second field is resident pages, not virtual address space.
        return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    if sys.platform == "darwin":
        # libproc reads this task's resident size in-process, roughly four
        # orders of magnitude cheaper than forking `ps`, so periodic in-phase
        # sampling is affordable. `_ps_rss_bytes` remains the slow cross-check.
        if _PROC_PIDINFO is None:
            raise RuntimeError("macOS resident-size reader proc_pidinfo is unavailable")
        info = _MacTaskInfo()
        written = _PROC_PIDINFO(os.getpid(), _PROC_PIDTASKINFO, 0,
                                ctypes.byref(info), ctypes.sizeof(info))
        if written != ctypes.sizeof(info):
            raise RuntimeError(f"proc_pidinfo returned {written} of {ctypes.sizeof(info)} bytes")
        return int(info.pti_resident_size)
    raise RuntimeError("R4 physical resource readings support Linux and macOS only")


def _ps_rss_bytes() -> int:
    """Independent macOS reference reader; a child process, never a sampler."""
    if sys.platform != "darwin":
        raise RuntimeError("the ps cross-check reader is macOS-only")
    # ps reports this parent's resident size in KiB; the child is not charged
    # to RUSAGE_SELF, but its own resident pages do perturb the machine.
    return int(subprocess.check_output(
        ["ps", "-o", "rss=", "-p", str(os.getpid())], text=True)) * 1024


def _memory_bytes() -> tuple[int, int]:
    rss = _rss_bytes()
    high_water = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss, int(high_water) * (1 if sys.platform == "darwin" else 1024)


def validate_sample_interval(sample_interval_seconds: float | None) -> None:
    """Refuse a non-real, non-finite or non-positive sampling period loudly."""
    if sample_interval_seconds is None:
        return
    if isinstance(sample_interval_seconds, bool) or \
            not isinstance(sample_interval_seconds, (int, float)):
        raise ValueError(  # noqa: TRY004 -- telemetry domains fail closed uniformly
            "resource sample interval must be a real number of seconds")
    if not math.isfinite(sample_interval_seconds) or sample_interval_seconds <= 0:
        raise ValueError("resource sample interval must be finite and positive")


class _PeakSampler:
    """Bounded periodic max-RSS sampler: O(1) state, no history, no RNG use."""

    def __init__(self, interval: float, initial: int) -> None:
        self._interval = interval
        self._stop = threading.Event()
        self._peak = initial
        self._samples = 0
        self._error: BaseException | None = None
        self._started = False
        self._thread = threading.Thread(target=self._run, name="r4-rss-sampler", daemon=True)

    def _run(self) -> None:
        try:
            while not self._stop.wait(self._interval):
                self._peak = max(self._peak, _rss_bytes())
                self._samples += 1
        except BaseException as error:  # noqa: BLE001 -- re-raised by stop(), never swallowed
            self._error = error

    def start(self) -> _PeakSampler:
        self._started = True
        self._thread.start()
        return self

    def stop(self) -> tuple[int, int]:
        """Join the sampler, then surface its peak or its failure."""
        self._stop.set()
        if self._started:
            self._thread.join()
        if self._error is not None:
            raise self._error
        return self._peak, self._samples


@contextmanager
def _measured_phase(observer: ResourceObserver, phase: str,
                    sample_interval_seconds: float | None):
    scope_id = next(_SCOPE_IDS)
    parent_scope_id = _ACTIVE_SCOPE.get()
    token = _ACTIVE_SCOPE.set(scope_id)
    try:
        # The memory window begins before sampler startup and closes after
        # sampler teardown. It intentionally includes instrumentation activity.
        rss_start, high_start = _memory_bytes()
        sampler = None if sample_interval_seconds is None else \
            _PeakSampler(sample_interval_seconds, rss_start).start()
        start = time.perf_counter()
        completed = False
        try:
            yield
            completed = True
        finally:
            end = time.perf_counter()
            elapsed = end - start
            # Never replace a work exception with a telemetry exception. Successful
            # work still fails loudly if its sampler, measurement or collector fails.
            # `stop` joins before it raises, so no sampler outlives its phase.
            try:
                peak, samples = (rss_start, 0) if sampler is None else sampler.stop()
                rss_end, high_end = _memory_bytes()
                record = PhaseResources(
                    phase, completed, elapsed, rss_start, rss_end, high_start, high_end,
                    max(peak, rss_start, rss_end), samples, sample_interval_seconds,
                    scope_id, parent_scope_id,
                    work_start_seconds=start, work_end_seconds=end,
                )
                record.validate(require_scope=True)
                observer(record)
            except BaseException:
                if completed:
                    raise
    finally:
        _ACTIVE_SCOPE.reset(token)


def resource_phase(observer: ResourceObserver | None, phase: str, *,
                   sample_interval_seconds: float | None = None):
    """Measure one CPU scope; disabled path reads no clock or process state.

    ``sample_interval_seconds`` adds a periodic in-phase resident-size sampler,
    which turns ``peak_lower_bound_bytes`` into a real intra-phase witness at
    the cost of one thread and its GIL contention inside the measured window.

    Failed work streams an incomplete record when possible and preserves its
    original exception. A sampler or collector failure after successful work
    propagates; callers must not retry a possibly already committed optimizer
    update.
    """
    validate_sample_interval(sample_interval_seconds)
    if observer is None:
        if sample_interval_seconds is not None:
            raise ValueError("resource sampling requires a collector")
        return nullcontext()
    return _measured_phase(observer, phase, sample_interval_seconds)


@dataclass(frozen=True)
class MemoryAttribution:
    """One instantaneous allocation inventory and arithmetic residual.

    ``allocation_residual_bytes`` is a residual, never an allocator-waste measurement.
    It also holds interpreter and library code, thread stacks, CPython arenas
    that ``tracemalloc`` does not model, and every allocation made outside both
    tracked components. It can be negative, because a counted allocation need
    not be resident. The named inventories may overlap and neither proves
    complete resident ownership; arithmetic is reported as measured and never
    clamped.
    """

    rss_bytes: int
    #: Distinct live CPU storages. A view charges its whole backing allocation.
    torch_cpu_storage_bytes: int
    torch_cpu_storages: int
    #: Tensors whose storage could not be read, so they are charged to nothing.
    unreadable_tensors: int
    #: ``tracemalloc`` current traced bytes, or ``None`` when it is not tracing.
    python_traced_bytes: int | None
    allocation_residual_bytes: int
    inventory_version: str = "allocation_inventory_v2"
    components_may_overlap: bool = True
    native_allocation_coverage_complete: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "rss_bytes": self.rss_bytes,
            "torch_cpu_storage_bytes": self.torch_cpu_storage_bytes,
            "torch_cpu_storages": self.torch_cpu_storages,
            "unreadable_tensors": self.unreadable_tensors,
            "python_traced_bytes": self.python_traced_bytes,
            "allocation_residual_bytes": self.allocation_residual_bytes,
            "inventory_version": self.inventory_version,
            "components_may_overlap": self.components_may_overlap,
            "native_allocation_coverage_complete": self.native_allocation_coverage_complete,
        }

    @property
    def unattributed_bytes(self) -> int:
        """Compatibility alias for the explicitly arithmetic residual."""
        return self.allocation_residual_bytes


def attribute_memory() -> MemoryAttribution:
    """Decompose resident size once at a declared attribution point.

    Deliberately expensive: it walks the whole tracked heap. It is not a
    sampler and must not run inside a measured scope, whose wall time it would
    dominate. Its own cost is measured by the resource benchmark.
    """
    import gc
    import tracemalloc

    import torch  # local: `dm.relation` is otherwise torch-free by design

    storages: dict[int, int] = {}
    unreadable = 0
    for obj in gc.get_objects():
        # `issubclass(type(obj), ...)`, not `isinstance`: the heap holds classes
        # whose metaclass warns on the attribute access `isinstance` performs.
        if not issubclass(type(obj), torch.Tensor):
            continue
        try:
            if obj.device.type != "cpu" or obj.layout is not torch.strided:
                continue
            storage = obj.untyped_storage()
            # A view retains its whole backing allocation; charge it once, by
            # address, rather than charging each view its own element count.
            storages[storage.data_ptr()] = storage.nbytes()
        except (RuntimeError, NotImplementedError, AttributeError):
            unreadable += 1
    traced = tracemalloc.get_traced_memory()[0] if tracemalloc.is_tracing() else None
    rss = _rss_bytes()
    total = sum(storages.values())
    return MemoryAttribution(rss, total, len(storages), unreadable, traced,
                             rss - total - (traced or 0))


@dataclass(frozen=True)
class PhaseTotals:
    """Physical accounting for one phase name; raw readings stay with the caller."""

    phase: str
    records: int
    completed_records: int
    wall_seconds_total: float
    wall_seconds_min: float
    wall_seconds_max: float
    samples: int
    #: Maxima across the phase's records, not sums of unrelated peaks. Each is
    #: taken independently, so they need not originate in the same record.
    peak_lower_bound_bytes: int
    peak_upper_bound_bytes: int
    isolated_peak_records: int
    isolated_peak_bytes: int | None

    def as_dict(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "records": self.records,
            "completed_records": self.completed_records,
            "wall_seconds_total": self.wall_seconds_total,
            "wall_seconds_min": self.wall_seconds_min,
            "wall_seconds_max": self.wall_seconds_max,
            "samples": self.samples,
            "peak_lower_bound_bytes": self.peak_lower_bound_bytes,
            "peak_upper_bound_bytes": self.peak_upper_bound_bytes,
            "isolated_peak_records": self.isolated_peak_records,
            "isolated_peak_bytes": self.isolated_peak_bytes,
        }


class PhaseLedger:
    """Wall time/memory totals plus validated residuals with bounded scope state.

    This is physical accounting only. It is never written into a training
    bundle, never restored on resume and never compared for exact equality.
    Nested phases overlap by construction, so totals are never added across
    phase names. An enclosing residual is available only for relationships
    declared at construction and proven by emitted scope identities.
    """

    def __init__(self, containment: dict[str, Sequence[str]] | None = None) -> None:
        self._totals: dict[str, PhaseTotals] = {}
        containment = {} if containment is None else containment
        self._containment: dict[str, tuple[str, ...]] = {}
        for parent, children in containment.items():
            if type(parent) is not str or not parent or not isinstance(children, Sequence) or \
                    isinstance(children, str) or not children:
                raise ValueError("ledger containment must map a phase to nonempty child names")
            child_names = tuple(children)
            if any(type(child) is not str or not child for child in child_names) or \
                    len(set(child_names)) != len(child_names) or parent in child_names:
                raise ValueError("ledger containment has self or duplicate child phases")
            self._containment[parent] = child_names
        if any(child in self._containment for children in self._containment.values() for child in children):
            raise ValueError("ledger supports sequential two-level scopes only")
        self._retired_scope_id = 0
        self._retired_work_end = 0.0
        self._children_by_parent: dict[int, list[PhaseResources]] = {}
        self._residuals: dict[tuple[str, tuple[str, ...]], tuple[int, float]] = {}
        self._failed = False

    def __call__(self, record: PhaseResources) -> None:
        if self._failed:
            raise PhysicalObservationRefused("resource ledger failed an earlier observation")
        try:
            self._accept(record)
        except BaseException:
            self._failed = True
            raise

    def _accept(self, record: PhaseResources) -> None:
        record.validate(require_scope=bool(self._containment))
        if not self._containment:
            self._observe(record)
            return
        if record.scope_id <= self._retired_scope_id:
            raise PhysicalObservationRefused("duplicate/replayed resource scope")
        if record.work_start_seconds < self._retired_work_end:
            raise PhysicalObservationRefused("resource work overlaps a retired sequential scope")
        if record.parent_scope_id is not None:
            parent = record.parent_scope_id
            if parent <= self._retired_scope_id or parent >= record.scope_id or \
                    (self._children_by_parent and parent not in self._children_by_parent):
                raise PhysicalObservationRefused("foreign/unresolved resource parent")
            children = self._children_by_parent.get(parent, [])
            names = (*[child.phase for child in children], record.phase)
            if not any(names == expected[:len(names)] for expected in self._containment.values()):
                raise PhysicalObservationRefused("duplicate or undeclared resource children")
            if children and (record.scope_id <= children[-1].scope_id or
                             record.work_start_seconds < children[-1].work_end_seconds):
                raise PhysicalObservationRefused("resource child intervals overlap or replay")
            self._children_by_parent[parent] = [*children, record]
            return
        children = self._children_by_parent.get(record.scope_id, [])
        expected = self._containment.get(record.phase)
        if expected is None or tuple(child.phase for child in children) != expected:
            raise PhysicalObservationRefused("resource children do not match declared parent")
        if not record.completed or any(not child.completed for child in children):
            raise PhysicalObservationRefused("incomplete resource scopes cannot produce a residual")
        if any(child.work_start_seconds < record.work_start_seconds or
               child.work_end_seconds > record.work_end_seconds for child in children):
            raise PhysicalObservationRefused("resource child interval outside enclosing work window")
        child_seconds = sum(child.wall_seconds for child in children)
        if child_seconds > record.wall_seconds:
            raise PhysicalObservationRefused("nested resource wall durations exceed enclosing work window")
        # Commit only after the entire sequential scope is proven valid.
        for child in children:
            self._observe(child)
        self._observe(record)
        key = record.phase, expected
        count_, total = self._residuals.get(key, (0, 0.0))
        self._residuals[key] = count_ + 1, total + record.wall_seconds - child_seconds
        self._retired_scope_id = max(record.scope_id, *(child.scope_id for child in children))
        self._retired_work_end = record.work_end_seconds
        self._children_by_parent.clear()

    def _observe(self, record: PhaseResources) -> None:
        previous = self._totals.get(record.phase)
        isolated = record.process_high_water_rise_bytes
        if previous is None:
            self._totals[record.phase] = PhaseTotals(
                phase=record.phase,
                records=1,
                completed_records=int(record.completed),
                wall_seconds_total=record.wall_seconds,
                wall_seconds_min=record.wall_seconds,
                wall_seconds_max=record.wall_seconds,
                samples=record.samples,
                peak_lower_bound_bytes=record.peak_lower_bound_bytes,
                peak_upper_bound_bytes=record.peak_upper_bound_bytes,
                isolated_peak_records=int(isolated is not None),
                isolated_peak_bytes=isolated,
            )
        else:
            self._totals[record.phase] = PhaseTotals(
                phase=record.phase,
                records=previous.records + 1,
                completed_records=previous.completed_records + int(record.completed),
                wall_seconds_total=previous.wall_seconds_total + record.wall_seconds,
                wall_seconds_min=min(previous.wall_seconds_min, record.wall_seconds),
                wall_seconds_max=max(previous.wall_seconds_max, record.wall_seconds),
                samples=previous.samples + record.samples,
                peak_lower_bound_bytes=max(previous.peak_lower_bound_bytes,
                                           record.peak_lower_bound_bytes),
                peak_upper_bound_bytes=max(previous.peak_upper_bound_bytes,
                                           record.peak_upper_bound_bytes),
                isolated_peak_records=previous.isolated_peak_records + int(isolated is not None),
                isolated_peak_bytes=previous.isolated_peak_bytes if isolated is None else
                max(isolated, previous.isolated_peak_bytes or 0),
            )

    def totals(self) -> dict[str, PhaseTotals]:
        return dict(self._totals)

    def unattributed_seconds(self, enclosing: str, nested: Sequence[str]) -> float:
        """Wall time inside ``enclosing`` that its ``nested`` phases do not cover.

        Valid only for an explicitly declared relationship whose each enclosing
        scope emitted exactly the requested, complete direct children. The
        residual includes the nested phases' own probe and collector overhead.
        """
        if self._failed:
            raise PhysicalObservationRefused("resource ledger failed an earlier observation")
        child_names = tuple(nested)
        if enclosing not in self._totals:
            raise KeyError(f"no records for enclosing phase {enclosing!r}")
        if not child_names or len(set(child_names)) != len(child_names) or enclosing in child_names:
            raise ValueError("residual requires distinct non-self nested phases")
        if self._containment.get(enclosing) != child_names:
            raise ValueError(
                f"residual {enclosing!r} <- {child_names!r} was not declared for this ledger")
        if self._children_by_parent:
            raise ValueError("resource scope trace has unresolved/foreign parent records")
        count_, total = self._residuals.get((enclosing, child_names), (0, 0.0))
        if count_ != self._totals[enclosing].records:
            raise ValueError("resource scope trace is incomplete; residual is undefined")
        return total

    def as_dict(self) -> dict[str, object]:
        return {name: totals.as_dict() for name, totals in sorted(self._totals.items())}
