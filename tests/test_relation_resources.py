"""Physical telemetry stays outside deterministic training continuation."""

import sys
import threading
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from dm.relation import resources


def test_phase_units_order_and_lifetime_high_water(monkeypatch):
    calls = []
    snapshots = iter([(100, 900), (80, 900)])
    times = iter([10.0, 12.5])

    def memory():
        calls.append("memory")
        return next(snapshots)

    def clock():
        calls.append("clock")
        return next(times)

    monkeypatch.setattr(resources, "_memory_bytes", memory)
    monkeypatch.setattr(resources.time, "perf_counter", clock)
    records = []
    with resources.resource_phase(records.append, "decode"):
        calls.append("work")
    assert calls == ["memory", "clock", "work", "clock", "memory"]
    assert {key: value for key, value in asdict(records[0]).items()
            if key not in {"scope_id", "parent_scope_id", "wall_window", "memory_window",
                           "work_start_seconds", "work_end_seconds"}} == {
        "phase": "decode", "completed": True, "wall_seconds": 2.5,
        "rss_start_bytes": 100, "rss_end_bytes": 80,
        "process_high_water_start_bytes": 900, "process_high_water_end_bytes": 900,
        "rss_peak_observed_bytes": 100, "samples": 0, "sample_interval_seconds": None,
    }
    assert records[0].peak_lower_bound_bytes == 100
    assert records[0].peak_upper_bound_bytes == 900
    assert records[0].isolated_peak_bytes is None
    assert records[0].wall_window == resources.WALL_WINDOW_WORK_BODY_V2
    assert records[0].memory_window == resources.MEMORY_WINDOW_SCOPE_V2
    assert records[0].scope_id > 0 and records[0].parent_scope_id is None
    with pytest.raises(AttributeError):
        records[0].completed = False


def test_disabled_phase_performs_no_measurement(monkeypatch):
    def forbidden():
        raise AssertionError("disabled telemetry read resources")
    monkeypatch.setattr(resources, "_memory_bytes", forbidden)
    monkeypatch.setattr(resources.time, "perf_counter", forbidden)
    with resources.resource_phase(None, "decode"):
        pass


def test_failed_phase_retains_exception_and_reports_incomplete():
    records = []
    error = RuntimeError("failed work")
    with pytest.raises(RuntimeError) as caught, resources.resource_phase(records.append, "decode"):
        raise error
    assert caught.value is error
    assert records[0].completed is False


def test_work_exception_survives_broken_collector():
    error = RuntimeError("original work failure")

    def broken(record):
        raise ValueError("collector failure")

    with pytest.raises(RuntimeError) as caught, resources.resource_phase(broken, "decode"):
        raise error
    assert caught.value is error


def test_successful_work_does_not_hide_collector_failure():
    error = RuntimeError("collector failure")

    def broken(record):
        raise error

    with pytest.raises(RuntimeError) as caught, resources.resource_phase(broken, "decode"):
        pass
    assert caught.value is error


@pytest.mark.parametrize("mode", ["standard", "predicted_copy", "oracle_copy"])
def test_external_decode_scope_preserves_outputs_work_and_rng(mode):
    import torch

    from dm.eval.relation_evidence import rng_digest, state_digest
    from dm.isa.asm import assemble
    from dm.isa.codec import ByteCodec
    from dm.isa.transform import Transform
    from dm.models.transformer import Config, DrawingLM
    from dm.relation import CopyActionKey
    from dm.train_relation import capture_rng_state, restore_rng_state

    torch.manual_seed(51)
    model = DrawingLM(Config(vocab_size=ByteCodec.vocab_size, d_model=16,
                            n_layers=1, n_heads=2, max_len=128,
                            relation_schema="span_affine_v1")).eval()
    prompt = assemble("MOVE 112 120\nLINE 128 136")
    kwargs = {"mode": mode, "literal_policy": "content_bytes_v1"}
    if mode == "oracle_copy":
        kwargs["oracle_actions"] = {(0, 6): CopyActionKey(6, 0, 6, Transform(), 2)}
    initial = capture_rng_state()
    model_before = state_digest(model.state_dict())
    events = []
    plain = model.generate_relation([prompt], 6, observer=events.append, **kwargs)
    final_rng = rng_digest("cpu")
    restore_rng_state(initial)
    measured_events, physical = [], []
    with resources.resource_phase(physical.append, "decode"):
        measured = model.generate_relation([prompt], 6, observer=measured_events.append, **kwargs)
    assert torch.equal(plain, measured)
    assert events == measured_events
    assert rng_digest("cpu") == final_rng
    assert state_digest(model.state_dict()) == model_before
    assert physical[0].completed


def test_real_cpu_memory_has_explicit_units():
    rss, peak = resources._memory_bytes()
    assert type(rss) is int and rss > 0
    assert type(peak) is int and peak > 0


def test_high_water_platform_units(monkeypatch):
    monkeypatch.setattr(resources.resource, "getrusage", lambda _: SimpleNamespace(ru_maxrss=123))
    monkeypatch.setattr(resources, "_rss_bytes", lambda: 99)
    monkeypatch.setattr(resources.sys, "platform", "darwin")
    assert resources._memory_bytes() == (99, 123)
    monkeypatch.setattr(resources.sys, "platform", "linux")
    assert resources._memory_bytes() == (99, 123 * 1024)


# ---------------------------------------------------------------------------
# Point 5 -- isolated physical-memory witnesses
# ---------------------------------------------------------------------------


def _fixed_memory(monkeypatch, readings):
    stream = iter(readings)
    monkeypatch.setattr(resources, "_memory_bytes", lambda: next(stream))


def test_high_water_rise_is_an_exact_isolated_phase_peak(monkeypatch):
    """ru_maxrss never falls, so a rise was established inside this phase."""
    _fixed_memory(monkeypatch, [(100, 1000), (150, 4000)])
    records = []
    with resources.resource_phase(records.append, "train_update"):
        pass
    record = records[0]
    assert record.isolated_peak_bytes == 4000
    # An exact peak is a tighter lower bound than a missed observation.
    assert record.rss_peak_observed_bytes == 150
    assert record.peak_lower_bound_bytes == 4000
    assert record.peak_upper_bound_bytes == 4000


def test_flat_high_water_reports_no_isolated_peak(monkeypatch):
    """An earlier record can exceed everything this phase used; refuse to guess."""
    _fixed_memory(monkeypatch, [(100, 4000), (150, 4000)])
    records = []
    with resources.resource_phase(records.append, "train_update"):
        pass
    assert records[0].isolated_peak_bytes is None
    assert records[0].rss_peak_observed_bytes == 150
    assert records[0].peak_lower_bound_bytes == 150
    assert records[0].peak_upper_bound_bytes == 4000


def test_sampler_observes_a_transient_both_endpoints_miss(monkeypatch):
    """The witness endpoint RSS cannot produce: a released in-phase allocation."""
    monkeypatch.setattr(resources.resource, "getrusage",
                        lambda _: SimpleNamespace(ru_maxrss=7_000))
    state = {"phase": "quiet"}
    observed = threading.Event()

    def scripted_rss():
        if state["phase"] == "transient":
            observed.set()
            return 900
        return 100

    monkeypatch.setattr(resources, "_rss_bytes", scripted_rss)
    records = []
    with resources.resource_phase(records.append, "metadata", sample_interval_seconds=0.001):
        state["phase"] = "transient"
        assert observed.wait(10)
        state["phase"] = "quiet"
    record = records[0]
    assert (record.rss_start_bytes, record.rss_end_bytes) == (100, 100)
    assert record.peak_lower_bound_bytes == 900
    assert record.samples >= 1
    assert record.sample_interval_seconds == 0.001
    assert record.isolated_peak_bytes is None


def test_real_allocation_gives_an_exact_isolated_peak():
    """Push resident size past the existing process record with real pages."""
    rss_before, high_before = resources._memory_bytes()
    headroom = max(high_before - rss_before, 0) + 32 * 2**20
    records = []
    with resources.resource_phase(records.append, "train_update",
                                  sample_interval_seconds=0.002):
        block = bytearray(headroom)
        for offset in range(0, len(block), 4096):
            block[offset] = 1
        del block
    record = records[0]
    assert record.isolated_peak_bytes is not None
    assert record.isolated_peak_bytes >= rss_before + headroom - 2**20
    assert record.peak_lower_bound_bytes <= record.peak_upper_bound_bytes
    assert record.peak_lower_bound_bytes >= max(record.rss_start_bytes, record.rss_end_bytes)


@pytest.mark.parametrize("interval", [0, -1, -0.5, True, "0.01", float("nan"), float("inf")])
def test_sample_interval_domain_is_refused(interval):
    with pytest.raises(ValueError, match="sample interval"):
        resources.resource_phase([].append, "metadata", sample_interval_seconds=interval)


def test_sampling_without_a_collector_is_refused():
    with pytest.raises(ValueError, match="requires a collector"):
        resources.resource_phase(None, "metadata", sample_interval_seconds=0.01)


def _sampler_reader_failure(monkeypatch):
    attempted = threading.Event()
    failure = RuntimeError("sampler read failed")

    def failing_rss():
        if threading.current_thread().name == "r4-rss-sampler":
            attempted.set()
            raise failure
        return 4096

    monkeypatch.setattr(resources, "_rss_bytes", failing_rss)
    monkeypatch.setattr(resources.resource, "getrusage",
                        lambda _: SimpleNamespace(ru_maxrss=7))
    return attempted, failure


def test_sampler_failure_after_successful_work_propagates(monkeypatch):
    attempted, failure = _sampler_reader_failure(monkeypatch)
    records = []
    with pytest.raises(RuntimeError) as caught, \
            resources.resource_phase(records.append, "metadata", sample_interval_seconds=0.001):
        assert attempted.wait(10)
    assert caught.value is failure
    assert records == []


def test_sampler_failure_never_replaces_a_work_exception(monkeypatch):
    attempted, _ = _sampler_reader_failure(monkeypatch)
    work = RuntimeError("original work failure")
    with pytest.raises(RuntimeError) as caught, \
            resources.resource_phase([].append, "metadata", sample_interval_seconds=0.001):
        assert attempted.wait(10)
        raise work
    assert caught.value is work


@pytest.mark.parametrize("outcome", ["success", "work_failure", "collector_failure"])
def test_no_sampler_thread_outlives_its_phase(outcome):
    def collector(record):
        if outcome == "collector_failure":
            raise ValueError("collector failure")

    before = threading.active_count()
    try:
        with resources.resource_phase(collector, "metadata", sample_interval_seconds=0.001):
            if outcome == "work_failure":
                raise RuntimeError("work failure")
    except (RuntimeError, ValueError):
        pass
    assert threading.active_count() == before
    assert not any(thread.name == "r4-rss-sampler" for thread in threading.enumerate())


# ---------------------------------------------------------------------------
# Point 5 -- platform resident-size readers
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS libproc reader")
def test_macos_libproc_agrees_with_the_ps_reference_reader():
    libproc = resources._rss_bytes()
    reference = resources._ps_rss_bytes()
    assert libproc > 0 and reference > 0
    # ps rounds to KiB and runs later; agreement is bounded, not exact.
    assert abs(libproc - reference) <= max(8 * 2**20, reference // 20)


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS libproc reader")
def test_macos_short_proc_pidinfo_read_is_refused(monkeypatch):
    monkeypatch.setattr(resources, "_PROC_PIDINFO", lambda *args: 0)
    with pytest.raises(RuntimeError, match="proc_pidinfo returned 0 of 96 bytes"):
        resources._rss_bytes()


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS libproc reader")
def test_macos_missing_proc_pidinfo_is_refused(monkeypatch):
    monkeypatch.setattr(resources, "_PROC_PIDINFO", None)
    with pytest.raises(RuntimeError, match="proc_pidinfo is unavailable"):
        resources._rss_bytes()


def test_unsupported_platform_is_refused(monkeypatch):
    monkeypatch.setattr(resources.sys, "platform", "win32")
    with pytest.raises(RuntimeError, match="Linux and macOS only"):
        resources._rss_bytes()


def test_proc_taskinfo_struct_matches_the_kernel_layout():
    assert resources.ctypes.sizeof(resources._MacTaskInfo) == 96
    assert resources._PROC_PIDTASKINFO == 4


# ---------------------------------------------------------------------------
# Point 5 -- per-phase physical accounting
# ---------------------------------------------------------------------------


def _record(phase, wall, *, completed=True, rss=(100, 120), high=(1000, 1000),
            peak=120, samples=0, interval=None, scope_id=0, parent_scope_id=None, start=0.0):
    return resources.PhaseResources(phase, completed, wall, rss[0], rss[1],
                                    high[0], high[1], peak, samples, interval,
                                    scope_id, parent_scope_id,
                                    work_start_seconds=start, work_end_seconds=start + wall)


def test_ledger_accounts_wall_time_and_memory_by_phase():
    ledger = resources.PhaseLedger()
    ledger(_record("metadata", 0.5, high=(1000, 1000), peak=120))
    ledger(_record("train_update", 2.0, high=(1000, 4000), peak=300, samples=7))
    ledger(_record("metadata", 1.5, high=(4000, 4000), peak=200, samples=3))
    totals = ledger.totals()
    assert set(totals) == {"metadata", "train_update"}
    metadata = totals["metadata"]
    assert (metadata.records, metadata.completed_records, metadata.samples) == (2, 2, 3)
    assert metadata.wall_seconds_total == 2.0
    assert (metadata.wall_seconds_min, metadata.wall_seconds_max) == (0.5, 1.5)
    assert metadata.peak_lower_bound_bytes == 200
    assert metadata.isolated_peak_records == 0
    assert metadata.isolated_peak_bytes is None
    update = totals["train_update"]
    assert update.peak_lower_bound_bytes == 4000
    assert update.isolated_peak_records == 1
    assert update.isolated_peak_bytes == 4000
    assert update.peak_upper_bound_bytes == 4000
    assert ledger.as_dict()["train_update"]["isolated_peak_bytes"] == 4000


def test_ledger_counts_incomplete_records_separately():
    ledger = resources.PhaseLedger()
    ledger(_record("step", 1.0))
    ledger(_record("step", 3.0, completed=False))
    assert ledger.totals()["step"].records == 2
    assert ledger.totals()["step"].completed_records == 1


def test_ledger_residual_is_the_unattributed_enclosing_time():
    ledger = resources.PhaseLedger({"step": ("metadata", "train_update")})
    for step in range(2):
        parent = 1 + step * 3
        ledger(_record("metadata", 0.25, scope_id=parent + 1, parent_scope_id=parent, start=step * 2))
        ledger(_record("train_update", 1.0, scope_id=parent + 2, parent_scope_id=parent, start=step * 2 + 0.25))
        ledger(_record("step", 1.5, scope_id=parent, start=step * 2))
    assert ledger.unattributed_seconds("step", ["metadata", "train_update"]) == \
           pytest.approx(0.5)


def test_ledger_refuses_an_undefined_residual():
    with pytest.raises(ValueError, match="self or duplicate"):
        resources.PhaseLedger({"step": ("step",)})
    ledger = resources.PhaseLedger({"step": ("metadata", "train_update")})
    # Equal counts alone cannot prove containment.
    with pytest.raises(resources.PhysicalObservationRefused, match="children"):
        ledger(_record("metadata", 0.25, scope_id=2))

    foreign = resources.PhaseLedger({"step": ("metadata", "train_update")})
    with pytest.raises(resources.PhysicalObservationRefused, match="foreign"):
        foreign(_record("metadata", 0.25, scope_id=2, parent_scope_id=99))

    incomplete = resources.PhaseLedger({"step": ("metadata", "train_update")})
    incomplete(_record("metadata", 0.25, completed=False, scope_id=2, parent_scope_id=1))
    incomplete(_record("train_update", 0.25, scope_id=3, parent_scope_id=1, start=0.25))
    with pytest.raises(resources.PhysicalObservationRefused, match="incomplete"):
        incomplete(_record("step", 1.5, scope_id=1))

    overlapping = resources.PhaseLedger({"step": ("metadata", "train_update")})
    overlapping(_record("metadata", 1.0, scope_id=2, parent_scope_id=1))
    with pytest.raises(resources.PhysicalObservationRefused, match="overlap"):
        overlapping(_record("train_update", 1.0, scope_id=3, parent_scope_id=1))

    with pytest.raises(resources.PhysicalObservationRefused, match="failed"):
        foreign.unattributed_seconds("step", ["step"])
    with pytest.raises(KeyError):
        resources.PhaseLedger().unattributed_seconds("decode", [])


def test_ledger_maxima_keep_the_peak_bound_ordering():
    """Aggregation must not report an exact peak above its own lower bound."""
    ledger = resources.PhaseLedger()
    ledger(_record("step", 1.0, high=(9000, 9000), peak=8000))
    ledger(_record("step", 1.0, high=(9000, 12000), peak=120))
    entry = ledger.totals()["step"]
    assert entry.isolated_peak_bytes == 12000
    assert entry.peak_lower_bound_bytes == 12000
    assert entry.peak_upper_bound_bytes == 12000
    assert entry.isolated_peak_bytes <= entry.peak_lower_bound_bytes \
           <= entry.peak_upper_bound_bytes


def test_contradictory_memory_observation_is_refused_not_clamped():
    record = _record("metadata", 1.0, rss=(100, 100), high=(100, 90), peak=100)
    with pytest.raises(resources.PhysicalObservationRefused, match="high-water decreased"):
        resources.PhaseLedger()(record)

    record = _record("metadata", 1.0, rss=(100, 100), high=(100, 150), peak=200)
    with pytest.raises(resources.PhysicalObservationRefused, match="exceeds"):
        resources.PhaseLedger()(record)


# ---------------------------------------------------------------------------
# Point 5 -- named memory attribution, with an explicitly unnamed residual
# ---------------------------------------------------------------------------


def test_attribution_charges_exact_torch_storage_bytes():
    import torch

    before = resources.attribute_memory()
    tensor = torch.zeros(1024, 1024)
    after = resources.attribute_memory()
    assert after.torch_cpu_storage_bytes - before.torch_cpu_storage_bytes == \
           tensor.untyped_storage().nbytes()
    assert after.torch_cpu_storages - before.torch_cpu_storages == 1
    assert after.unreadable_tensors == 0


def test_attribution_charges_a_view_its_whole_backing_allocation():
    """A tensor view retains its backing allocation and is never charged twice."""
    import torch

    base = torch.zeros(1024, 1024)
    before = resources.attribute_memory()
    view = base[:1]
    alias = base.view(-1)
    after = resources.attribute_memory()
    assert view.numel() * view.element_size() < base.untyped_storage().nbytes()
    assert after.torch_cpu_storage_bytes == before.torch_cpu_storage_bytes
    assert after.torch_cpu_storages == before.torch_cpu_storages
    assert alias.untyped_storage().data_ptr() == base.untyped_storage().data_ptr()


def test_attribution_reconciles_its_own_arithmetic():
    import tracemalloc

    assert resources.attribute_memory().python_traced_bytes is None
    tracemalloc.start()
    try:
        held = bytearray(8 * 2**20)
        record = resources.attribute_memory()
    finally:
        tracemalloc.stop()
    assert record.python_traced_bytes is not None
    assert record.python_traced_bytes >= len(held)
    # The residual is exactly what the named components do not explain.
    assert record.allocation_residual_bytes == (record.rss_bytes
                                                - record.torch_cpu_storage_bytes
                                                - record.python_traced_bytes)
    assert record.as_dict()["allocation_residual_bytes"] == record.allocation_residual_bytes
    assert record.components_may_overlap is True
    assert record.native_allocation_coverage_complete is False
    # A Python bytearray is not torch storage; the components stay separate.
    assert record.torch_cpu_storage_bytes < record.python_traced_bytes


def test_attribution_counts_unreadable_tensors_instead_of_failing():
    import torch

    class Hostile(torch.Tensor):
        @property
        def device(self):
            raise RuntimeError("no device")

    hostile = torch.zeros(4).as_subclass(Hostile)
    record = resources.attribute_memory()
    assert record.unreadable_tensors >= 1
    assert isinstance(hostile, torch.Tensor)
