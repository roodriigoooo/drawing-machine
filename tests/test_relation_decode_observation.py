"""Independent operation spies and immutable observation acceptance."""
import gc
import random
import weakref
from dataclasses import FrozenInstanceError, asdict, replace

import numpy as np
import pytest
import torch
from test_relation_decode_policy import model

from dm.isa.asm import assemble
from dm.isa.codec import BOS, N_SPECIAL
from dm.isa.transform import Transform
from dm.models.relation import PackedCandidates, length_bin, score_snapshot
from dm.models.transformer import LayerCache, RelationBoundaryCache
from dm.relation import CopyActionKey, candidate_spans
from dm.relation.decoding import (
    ActionDecision,
    RequestCompleted,
    RequestStarted,
    RowStopped,
    ScoreDetail,
)

SOURCE = assemble("MOVE 112 120\nLINE 128 136")


def halt(pos, raw):
    raw.fill_(-1000)
    raw[:, 2] = 0


def test_hand_ledger_and_independent_cache_sampler_spies(monkeypatch):
    net = model()
    cache_positions, sampled = [], []
    append, sample = LayerCache.append, torch.multinomial
    def cache_spy(self, k, v):
        cache_positions.append(k.shape[0] * k.shape[2])
        return append(self, k, v)
    def sample_spy(probs, n):
        sampled.append(len(probs))
        return sample(probs, n)
    monkeypatch.setattr(LayerCache, "append", cache_spy)
    monkeypatch.setattr(torch, "multinomial", sample_spy)
    for mode in ("standard", "oracle_copy"):
        events = []
        cache_positions.clear()
        sampled.clear()
        def literal(pos, raw):
            halt(pos, raw)
            if pos < 12:
                raw[:, 2] = -1000
                raw[:, 2 + SOURCE[pos - 6]] = 0
        output = net.generate_relation(
            [SOURCE], 7, mode=mode, literal_policy="content_bytes_v1", observer=events.append,
            observation_level="actions", on_logits=literal,
            **({"oracle_actions": {(0, 6): CopyActionKey(6, 0, 6, Transform(), 2)}}
               if mode == "oracle_copy" else {}))
        assert output.tolist() == [[2 + b for b in SOURCE * 2 + b"\0"]]
        summary = events[-1]
        assert isinstance(summary, RequestCompleted)
        c = dict(summary.totals)
        assert cache_positions == [7, 1, 1, 1, 1, 1, 1]
        assert sum(cache_positions) == c["prefill_row_positions"] + c["incremental_row_positions"]
        assert len(sampled) == c["sampler_calls"]
        assert sum(sampled) == c["sampler_rows"] == c["literal_decisions"]
        assert c["literal_decisions"] == (1 if mode == "oracle_copy" else 7)
        assert c["copy_admitted_bytes"] == c["copy_delivered_bytes"] == (
            6 if mode == "oracle_copy" else 0)
        assert c["copy_pending_bytes"] == c["copy_discarded_bytes"] == 0
        assert c["fed_copy_bytes"] == (6 if mode == "oracle_copy" else 0)
        assert c["prefill_calls"] == 1 and c["incremental_calls"] == 6


@pytest.mark.parametrize("mode", ["standard", "oracle_copy", "predicted_copy"])
def test_stop_failure_and_final_operand_statuses(mode):
    net = model()
    for values, cause in (([255], "invalid_opcode"), ([7], "non_flat_opcode"),
                          ([0], "halt"), ([1, 0], "horizon_partial"),
                          ([1, 0, 0], "horizon_boundary")):
        events = []
        def scripted(pos, raw, values=values):
            raw.fill_(-1000)
            raw[:, values[pos] + 2] = 0
        net.generate_relation([b""], len(values), mode=mode, literal_policy="content_bytes_v1",
                              on_logits=scripted, observer=events.append,
                              variates=torch.zeros(1, len(values)))
        status = events[-1].statuses[0]
        assert status.cause == cause
        assert status.generated_bytes == status.useful_length == len(values)
        assert status.byte_offset == len(values) - 1
        assert status.at_horizon


def test_mid_copy_stop_discards_without_silent_pop(monkeypatch):
    net = model()
    class Stop:
        stride = 1
        def __init__(self):
            self.pos = 0
        def step(self, chunk):
            self.pos += 1
            return np.array([self.pos >= 8, False], dtype=bool)
    events = []
    output = net.generate_relation(
        [SOURCE, SOURCE], 13, mode="oracle_copy", observer=events.append,
        observation_level="actions", on_logits=halt, monitor=Stop(),
        oracle_actions={(r, 6): CopyActionKey(6, 0, 6, Transform(), r + 2) for r in range(2)})
    assert output[0, :8].tolist() == [2 + b for b in SOURCE + SOURCE[:2]]
    assert (output[0, 8:] == 0).all()
    rows = [dict(row) for row in events[-1].per_row]
    assert rows[0]["copy_admitted_bytes"] == 6
    assert rows[0]["copy_delivered_bytes"] == 2
    assert rows[0]["copy_discarded_bytes"] == 4
    assert rows[0]["fed_copy_bytes"] == 2  # stopping operand is fed beside live neighbor
    assert rows[0]["fed_pad_symbols"] == 10
    assert events[-1].statuses[0].cause == "external_monitor_stop"
    assert events[-1].statuses[0].pending_before_discard == 4


def test_standard_stopped_rows_and_uniform_work(monkeypatch):
    net = model()
    events, indexes, positions = [], [], []
    search, append = torch.searchsorted, LayerCache.append
    def inverse(cdf, draw, *args, **kwargs):
        indexes.append(draw.clone())
        return search(cdf, draw, *args, **kwargs)
    def cache(self, k, v):
        positions.append(k.shape[0] * k.shape[2])
        return append(self, k, v)
    monkeypatch.setattr(torch, "searchsorted", inverse)
    monkeypatch.setattr(LayerCache, "append", cache)
    def scripted(pos, raw):
        raw.fill_(-1000)
        raw[0, 2] = 0
        raw[1, 2 + (6 if pos < 2 else 0)] = 0
    uniforms = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    result = net.generate_relation([b"", b""], 3, literal_policy="content_bytes_v1",
                                   variates=uniforms, observer=events.append, on_logits=scripted)
    assert result.tolist() == [[2, 0, 0], [8, 8, 2]]
    c = dict(events[-1].totals)
    assert positions == [2, 2, 2]
    assert c["sampler_calls"] == 3
    assert c["sampler_rows"] == c["uniform_coordinates_read"] == 6
    assert c["literal_decisions"] == 4 and c["discarded_sample_rows"] == 2
    assert c["fed_literal_bytes"] == 3 and c["fed_pad_symbols"] == 1
    assert c["fed_stopped_row_positions"] == 2
    assert all(torch.equal(draw[:, 0], uniforms[:, i]) for i, draw in enumerate(indexes))


@pytest.mark.parametrize("level", ["summary", "actions", "scores"])
def test_observer_owned_transparent_and_failure_aborts(level):
    net = model()
    state = {k: v.clone() for k, v in net.state_dict().items()}
    rng = torch.get_rng_state().clone()
    py_rng = random.getstate()
    np_rng = np.random.get_state()
    kwargs = {"mode": "predicted_copy", "on_logits": halt,
              "variates": torch.zeros(1, 7)}
    plain = net.generate_relation([SOURCE], 1, **kwargs)
    events = []
    observed = net.generate_relation([SOURCE], 1, observer=events.append,
                                     observation_level=level, **kwargs)
    assert torch.equal(plain, observed)
    assert torch.equal(rng, torch.get_rng_state()) and random.getstate() == py_rng
    assert np.array_equal(np_rng[1], np.random.get_state()[1])
    assert all(torch.equal(v, net.state_dict()[k]) for k, v in state.items())
    for event in events:
        with pytest.raises((FrozenInstanceError, AttributeError)):
            event.kind = "forged"
        def primitives(value):
            if isinstance(value, dict):
                return all(primitives(v) for v in value.values())
            if isinstance(value, tuple):
                return all(primitives(v) for v in value)
            return value is None or type(value) in (str, bool, int, float)
        assert primitives(asdict(event))
    assert any(isinstance(e, ScoreDetail) for e in events) == (level == "scores")
    assert any(isinstance(e, ActionDecision) for e in events) == (level != "summary")
    failure = RuntimeError("observer sentinel")
    seen = []
    def raises(event):
        seen.append(event)
        if isinstance(event, RowStopped):
            raise failure
    with pytest.raises(RuntimeError) as caught:
        net.generate_relation([SOURCE], 1, observer=raises, **kwargs)
    assert caught.value is failure
    assert not any(isinstance(e, RequestCompleted) for e in seen)


def test_predicted_mixed_rows_match_independent_full_prefix_states(monkeypatch):
    net = model()
    terminal = assemble("FILL\nFILL\nFILL\nFILL\nFILL\nHALT")
    programs = [SOURCE * 3 + b"\0", SOURCE * 2 + b"\0", SOURCE + b"\0", terminal]
    counts = [3, 2, 0, 0]
    expected_queries = {6: [0, 1, 2], 12: [1], 18: [0]}
    current = -1
    query_errors = []
    boundary_values = []
    original_score = net.score_relation
    original_put = RelationBoundaryCache.put

    def observe_put(cache, row, boundary, state):
        boundary_values.append((row, boundary, state.clone()))
        return original_put(cache, row, boundary, state)

    def scripted_scores(packed):
        scores = original_score(packed)
        rows = expected_queries[current]
        assert packed.queries == len(rows)
        for query, row in enumerate(rows):
            index = torch.tensor([[BOS, *[N_SPECIAL + byte for byte in programs[row][:current]]]])
            with torch.no_grad():
                _, expected = net(index, return_state=True)
            query_errors.append(float((packed.query_states[query] - expected[0, -1]).abs().max()))
        gate = torch.full_like(scores.gate_logits, -1000)
        span = torch.full_like(scores.span_logits, -1000)
        count = torch.full_like(scores.count_logits, -1000)
        for query, row in enumerate(rows):
            copy = current == len(SOURCE) and counts[row] > 1
            gate[query, int(copy)] = 0
            count[query, max(0, counts[row] - 2)] = 0
            begin, end = packed.row_splits[query:query + 2].tolist()
            for candidate in range(begin, end):
                if (int(packed.source_start[candidate]), int(packed.source_stop[candidate])) == (0, 6):
                    span[candidate] = 0
        factors = {}
        for name, selected in (("d4_logits", 0), ("dx_logits", 1), ("dy_logits", 1)):
            logits = torch.full_like(getattr(scores, name), -1000)
            logits[:, selected] = 0
            factors[name] = logits
        return replace(scores, gate_logits=gate, span_logits=span,
                       count_logits=count, **factors)

    def scripted_logits(position, raw):
        nonlocal current
        current = position
        for row, program in enumerate(programs):
            if position < len(program):
                index = torch.tensor([[BOS, *[N_SPECIAL + byte for byte in program[:position]]]])
                with torch.no_grad():
                    expected = net(index)[:, -1]
                torch.testing.assert_close(raw[row], expected[0], atol=2e-7, rtol=2e-6)
        raw.fill_(-1000)
        for row, program in enumerate(programs):
            raw[row, N_SPECIAL + (program[position] if position < len(program) else 6)] = 0

    monkeypatch.setattr(RelationBoundaryCache, "put", observe_put)
    monkeypatch.setattr(net, "score_relation", scripted_scores)
    result = net.generate_relation(
        [SOURCE, SOURCE, SOURCE, terminal], 13, mode="predicted_copy",
        literal_policy="content_bytes_v1", variates=torch.full((4, 19), 0.5),
        on_logits=scripted_logits,
    )
    expected = [[N_SPECIAL + byte for byte in program] for program in programs]
    width = max(map(len, expected))
    expected = [row + [0] * (width - len(row)) for row in expected]
    assert result.tolist() == expected
    assert max(query_errors) <= 2e-6
    for row, boundary, state in boundary_values:
        index = torch.tensor([[BOS, *[N_SPECIAL + byte for byte in programs[row][:boundary]]]])
        with torch.no_grad():
            _, expected_state = net(index, return_state=True)
        torch.testing.assert_close(state, expected_state[0, -1], atol=2e-6, rtol=2e-6)


def test_copy_rows_skip_absolute_uniform_coordinates(monkeypatch):
    programs = [SOURCE * 2 + b"\0", SOURCE * 2 + b"\0"]
    uniforms = (torch.arange(26, dtype=torch.float32).reshape(2, 13) + 1) / 100
    original_search = torch.searchsorted
    for mode in ("predicted_copy", "oracle_copy"):
        net = model()
        current = -1
        draws = []
        original_score = net.score_relation

        def inverse(cdf, draw, *args, draws=draws, **kwargs):
            draws.append(draw.clone())
            return original_search(cdf, draw, *args, **kwargs)

        def logits(position, raw):
            nonlocal current
            current = position
            raw.fill_(-1000)
            for row, program in enumerate(programs):
                raw[row, N_SPECIAL + program[position]] = 0

        def decisions(packed, original_score=original_score):
            scores = original_score(packed)
            gate = torch.full_like(scores.gate_logits, -1000)
            gate[:, 0] = 0
            if current == 6:  # noqa: B023 — callback consumed synchronously within this run.
                gate[0] = torch.tensor([-1000, 0], dtype=gate.dtype)
            fields = {}
            for name, selected in (("d4_logits", 0), ("dx_logits", 1),
                                   ("dy_logits", 1), ("count_logits", 0)):
                values = torch.full_like(getattr(scores, name), -1000)
                values[:, selected] = 0
                fields[name] = values
            spans = torch.full_like(scores.span_logits, -1000)
            for index in range(len(spans)):
                if (int(packed.source_start[index]), int(packed.source_stop[index])) == (0, 6):
                    spans[index] = 0
            return replace(scores, gate_logits=gate, span_logits=spans, **fields)

        monkeypatch.setattr(torch, "searchsorted", inverse)
        if mode == "predicted_copy":
            monkeypatch.setattr(net, "score_relation", decisions)
        events = []
        result = net.generate_relation(
            [SOURCE, SOURCE], 7, mode=mode, literal_policy="content_bytes_v1",
            variates=uniforms, on_logits=logits, observer=events.append,
            **({"oracle_actions": {(0, 6): CopyActionKey(6, 0, 6, Transform(), 2)}}
               if mode == "oracle_copy" else {}),
        )
        assert result.tolist() == [[N_SPECIAL + byte for byte in program] for program in programs]
        expected_draws = [uniforms[1:2, offset, None] for offset in range(6, 12)]
        expected_draws.append(uniforms[:, 12, None])
        assert len(draws) == len(expected_draws)
        assert all(torch.equal(actual, expected) for actual, expected in zip(
            draws, expected_draws, strict=True))
        totals = dict(events[-1].totals)
        assert totals["uniform_coordinates_read"] == totals["sampler_rows"] == 8
        assert totals["copy_delivered_bytes"] == 6


def test_horizon_refuses_copy_admission_and_final_byte_is_unfed(monkeypatch):
    net = model()
    original = net.score_relation

    def force_copy(packed):
        scores = original(packed)
        fields = {}
        gate = torch.full_like(scores.gate_logits, -1000)
        gate[:, 1] = 0
        fields["gate_logits"] = gate
        for name in ("d4_logits", "dx_logits", "dy_logits", "count_logits"):
            fields[name] = torch.zeros_like(getattr(scores, name))
        spans = torch.zeros_like(scores.span_logits)
        return replace(scores, span_logits=spans, **fields)

    monkeypatch.setattr(net, "score_relation", force_copy)
    events = []
    output = net.generate_relation(
        [SOURCE], 1, mode="predicted_copy", literal_policy="content_bytes_v1",
        observer=events.append, observation_level="actions", on_logits=halt,
        variates=torch.zeros(1, 7),
    )
    assert output.tolist() == [[*[N_SPECIAL + byte for byte in SOURCE], N_SPECIAL]]
    action = next(event for event in events if isinstance(event, ActionDecision))
    assert action.action_key is None and action.exit_reason == "attempt_cap"
    assert action.faults == (("max_length", 32),)
    totals = dict(events[-1].totals)
    assert totals["copy_admissions"] == totals["copy_admitted_bytes"] == 0
    assert totals["incremental_calls"] == totals["fed_literal_bytes"] == 0
    with pytest.raises(ValueError, match="max_length"):
        net.generate_relation(
            [SOURCE], 1, mode="oracle_copy",
            oracle_actions={(0, 6): CopyActionKey(6, 0, 6, Transform(), 2)},
        )


def test_observer_preserves_decisions_and_multinomial_rng(monkeypatch):
    import dm.models.relation as relation_model

    net = model()
    original = relation_model.choose_action
    decisions = []

    def choose(*args, **kwargs):
        decision = original(*args, **kwargs)
        decisions.append(decision)
        return decision

    monkeypatch.setattr(relation_model, "choose_action", choose)
    kwargs = {"mode": "predicted_copy", "on_logits": halt,
              "variates": torch.zeros(1, 7)}
    plain = net.generate_relation([SOURCE], 1, **kwargs)
    split = len(decisions)
    events = []
    observed = net.generate_relation(
        [SOURCE], 1, observer=events.append, observation_level="scores", **kwargs,
    )
    assert torch.equal(plain, observed) and decisions[:split] == decisions[split:]

    def literal(position, raw):
        raw.fill_(-1000)
        raw[:, N_SPECIAL + ([6, 6, 0][position])] = 0

    start = torch.get_rng_state().clone()
    plain = net.generate_relation([b""], 3, mode="predicted_copy", on_logits=literal)
    plain_end = torch.get_rng_state().clone()
    torch.set_rng_state(start)
    events = []
    observed = net.generate_relation(
        [b""], 3, mode="predicted_copy", on_logits=literal, observer=events.append,
    )
    assert torch.equal(plain, observed)
    assert torch.equal(plain_end, torch.get_rng_state())


@pytest.mark.parametrize("mode", ["predicted_copy", "oracle_copy"])
def test_actual_copy_observation_preserves_all_cpu_rng_and_owned_scores(monkeypatch, mode):
    from dm.models import relation as relation_model

    net = model()
    original_score = net.score_relation
    original_choose = relation_model.choose_action
    current = 6
    decisions = []

    def score(packed):
        scores = original_score(packed)
        values = {}
        for name, selected in (("gate_logits", int(current == 6)), ("d4_logits", 0),
                               ("dx_logits", 1), ("dy_logits", 1), ("count_logits", 0)):
            logits = torch.full_like(getattr(scores, name), -1000)
            logits[:, selected] = 0
            values[name] = logits
        return replace(scores, **values)

    def choose(*args, **kwargs):
        result = original_choose(*args, **kwargs)
        decisions.append(result)
        return result

    def logits(position, raw):
        nonlocal current
        current = position
        raw.fill_(-1000)
        raw[:, N_SPECIAL] = 0

    monkeypatch.setattr(net, "score_relation", score)
    monkeypatch.setattr(relation_model, "choose_action", choose)
    initial = torch.get_rng_state().clone()
    py_initial = random.getstate()
    np_initial = np.random.get_state()
    weights = {name: value.clone() for name, value in net.state_dict().items()}
    runs = []
    retained = []
    frozen = []
    for level in (None, "summary", "actions", "scores"):
        torch.set_rng_state(initial)
        decisions.clear()
        events = []
        result = net.generate_relation(
            [SOURCE], 7, mode=mode, on_logits=logits,
            observer=None if level is None else events.append,
            observation_level=level or "summary",
            **({"oracle_actions": {(0, 6): CopyActionKey(6, 0, 6, Transform(), 2)}}
               if mode == "oracle_copy" else {}),
        )
        assert result.tolist() == [[*[N_SPECIAL + byte for byte in SOURCE * 2], N_SPECIAL]]
        assert random.getstate() == py_initial
        np_end = np.random.get_state()
        assert np_end[0] == np_initial[0] and np_end[2:] == np_initial[2:]
        assert np.array_equal(np_end[1], np_initial[1])
        assert all(torch.equal(value, net.state_dict()[name]) for name, value in weights.items())
        if events:
            assert dict(events[-1].totals)["copy_admitted_bytes"] == 6
        runs.append((result, torch.get_rng_state().clone(), tuple(decisions)))
        retained.extend(e for e in events if isinstance(e, ScoreDetail))
        frozen = [asdict(e) for e in retained]
    assert all(torch.equal(runs[0][0], output) and torch.equal(runs[0][1], rng)
               and runs[0][2] == actions for output, rng, actions in runs[1:])
    if mode == "predicted_copy":
        assert any(not action.emit for action in runs[0][2]) and retained
    net(torch.tensor([[BOS, N_SPECIAL + 6]]))
    assert [asdict(e) for e in retained] == frozen


@pytest.mark.parametrize("event_type", [RequestStarted, ActionDecision, ScoreDetail, RowStopped])
def test_observer_exception_propagates_at_each_event_seam(event_type):
    sentinel = RuntimeError(event_type.__name__)
    seen = []

    def observer(event):
        seen.append(event)
        if isinstance(event, event_type):
            raise sentinel

    with pytest.raises(RuntimeError) as caught:
        model().generate_relation(
            [SOURCE], 1, mode="predicted_copy", on_logits=halt,
            variates=torch.zeros(1, 7), observer=observer, observation_level="scores",
        )
    assert caught.value is sentinel
    assert not any(isinstance(event, RequestCompleted) for event in seen)


def test_score_payload_is_bounded_at_real_and_synthetic_widths():
    for endpoints in (
        candidate_spans(bytes((1, 0, 0)) * 68, 204),
        tuple((0, 6) for _ in range(768)),
    ):
        width = len(endpoints)
        assert width in (315, 768)
        packed = PackedCandidates(
            query_states=torch.randn(1, 16),
            start_states=torch.randn(width, 16),
            stop_states=torch.randn(width, 16),
            length_bins=torch.tensor([length_bin(stop - start) for start, stop in endpoints]),
            source_start=torch.tensor([start for start, _ in endpoints]),
            source_stop=torch.tensor([stop for _, stop in endpoints]),
            row_splits=torch.tensor([0, width]),
        )
        detail = score_snapshot(model().score_relation(packed), 0, 0, 204)
        assert len(detail.endpoints) == len(detail.span) == width
        assert len(detail.d4) == 8 and len(detail.dx) == len(detail.dy) == len(detail.count) == 3

        def owns_primitives(value):
            if isinstance(value, dict):
                return all(type(key) is str and owns_primitives(item)
                           for key, item in value.items())
            if isinstance(value, tuple):
                return all(owns_primitives(item) for item in value)
            return value is None or type(value) in (str, bool, int, float)

        assert owns_primitives(asdict(detail))


def test_long_request_streams_without_model_event_retention(monkeypatch):
    class TrackedScore(ScoreDetail):
        __slots__ = ("__weakref__",)

    monkeypatch.setattr("dm.relation.decoding.ScoreDetail", TrackedScore)
    references = []
    net = model()
    original = net.score_relation
    continuation = bytes((1, 0, 0)) * 21 + b"\0"

    def emit_only(packed):
        scores = original(packed)
        gate = torch.full_like(scores.gate_logits, -1000)
        gate[:, 0] = 0
        return replace(scores, gate_logits=gate)

    class Sink:
        __slots__ = ("count", "max_endpoints")

        def __init__(self):
            self.count = 0
            self.max_endpoints = 0

        def __call__(self, event):
            self.count += 1
            if isinstance(event, ScoreDetail):
                references.append(weakref.ref(event))
                # Prior score records must not accumulate in request/model ownership.
                assert sum(ref() is not None for ref in references) <= 2
                self.max_endpoints = max(self.max_endpoints, len(event.endpoints))

    def scripted(position, raw):
        raw.fill_(-1000)
        raw[:, N_SPECIAL + continuation[position]] = 0

    monkeypatch.setattr(net, "score_relation", emit_only)
    sink = Sink()
    output = net.generate_relation(
        [b""], len(continuation), mode="predicted_copy", on_logits=scripted,
        variates=torch.zeros(1, len(continuation)), observer=sink, observation_level="scores",
    )
    assert output.tolist() == [[N_SPECIAL + byte for byte in continuation]]
    assert sink.count == 47 and 0 < sink.max_endpoints <= 768
    gc.collect()
    assert references and all(ref() is None for ref in references)


def test_standard_composes_callback_monitor_and_terminal_row_work():
    terminal = assemble("FILL\nFILL\nFILL\nFILL\nFILL\nHALT")

    class Monitor:
        stride = 1

        def __init__(self):
            self.calls = []

        def step(self, chunk):
            self.calls.append(chunk.copy())
            return np.zeros(2, dtype=bool)

    monitor = Monitor()
    callbacks = []
    events = []
    program = SOURCE + b"\x06" * 6 + b"\0"

    def scripted(position, raw):
        callbacks.append(position)
        raw.fill_(-1000)
        raw[0, N_SPECIAL + program[position]] = 0
        raw[1, N_SPECIAL + 6] = 0

    result = model().generate_relation(
        [SOURCE, terminal], 7, literal_policy="content_bytes_v1", monitor=monitor,
        on_logits=scripted, variates=torch.zeros(2, 13), observer=events.append,
    )
    assert result.tolist() == [
        [N_SPECIAL + byte for byte in program],
        [*[N_SPECIAL + byte for byte in terminal], *([0] * 7)],
    ]
    assert callbacks == list(range(6, 13))
    assert len(monitor.calls) == 13
    rows = [dict(row) for row in events[-1].per_row]
    assert rows[0]["prefill_row_positions"] == rows[1]["prefill_row_positions"] == 7
    assert rows[1]["discarded_sample_rows"] == 7
    assert rows[1]["fed_pad_symbols"] == rows[1]["fed_stopped_row_positions"] == 6


def test_standard_failure_between_callback_and_monitor_does_not_complete(monkeypatch):
    sentinel = RuntimeError("sampler failed")
    callbacks = []
    events = []

    class Monitor:
        stride = 1

        def __init__(self):
            self.calls = 0

        def step(self, chunk):
            self.calls += 1
            return np.zeros(1, dtype=bool)

    monitor = Monitor()
    monkeypatch.setattr(torch, "multinomial",
                        lambda *args, **kwargs: (_ for _ in ()).throw(sentinel))
    with pytest.raises(RuntimeError) as caught:
        model().generate_relation(
            [b""], 1, literal_policy="content_bytes_v1", monitor=monitor,
            on_logits=lambda position, raw: callbacks.append(position), observer=events.append,
        )
    assert caught.value is sentinel
    assert callbacks == [0] and monitor.calls == 0
    assert len(events) == 1 and isinstance(events[0], RequestStarted)
