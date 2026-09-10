"""Point 4 policy witnesses; no learned outcomes or corpus inputs."""
import ast
import hashlib
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import torch

from dm.isa.asm import assemble
from dm.isa.codec import N_SPECIAL
from dm.models.transformer import Config, DrawingLM


def model(*, relation=True):
    torch.manual_seed(51)
    return DrawingLM(Config(vocab_size=258, d_model=16, n_layers=1, n_heads=2,
                            max_len=128,
                            relation_schema="span_affine_v1" if relation else "none")).eval()


def script(monkeypatch, net, rows):
    def logits(pos, raw):
        raw.fill_(-1000)
        for row, values in enumerate(rows):
            raw[row, N_SPECIAL + values[min(pos, len(values) - 1)]] = 0
    return logits


def test_historical_generate_source():
    source = Path("dm/models/transformer.py").read_text()
    node = next(n for n in ast.walk(ast.parse(source))
                if isinstance(n, ast.FunctionDef) and n.name == "generate")
    text = "".join(source.splitlines(keepends=True)[node.decorator_list[0].lineno-1:node.end_lineno])
    assert hashlib.sha256(text.encode()).hexdigest() == Path(
        "tests/fixtures/relation_generate_sha256.txt").read_text().strip()


@pytest.mark.parametrize("values", [[255], [7], [0], [1, 0, 7, 255], [1, 0], [6, 0]])
def test_common_forced_emit_parity(monkeypatch, values):
    net = model()
    results = []
    for mode in ("standard", "predicted_copy", "oracle_copy"):
        results.append(net.generate_relation(
            [b""], len(values), mode=mode, literal_policy="content_bytes_v1",
            variates=torch.zeros(1, len(values)), on_logits=script(monkeypatch, net, [values])))
    assert all(torch.equal(results[0], result) for result in results)
    assert results[0][0].tolist() == [N_SPECIAL + b for b in values]


def test_common_forced_emit_mixed_rows_and_none_standard_arm(monkeypatch):
    source = assemble("MOVE 112 120\nLINE 128 136")
    terminal = assemble("FILL\nFILL\nFILL\nFILL\nFILL\nHALT")
    prompts = [source, terminal]
    continuation = [6, 0]
    uniforms = torch.tensor([[0.1] * 8, [0.9] * 8])
    outputs = []
    statuses = []
    for mode in ("standard", "predicted_copy", "oracle_copy"):
        net = model(relation=mode != "standard")
        if mode == "predicted_copy":
            original = net.score_relation

            def force_emit(packed, original=original):
                scores = original(packed)
                gate = torch.full_like(scores.gate_logits, -1000)
                gate[:, 0] = 0
                return replace(scores, gate_logits=gate)

            monkeypatch.setattr(net, "score_relation", force_emit)
        events = []

        def literal(position, raw):
            raw.fill_(-1000)
            raw[0, N_SPECIAL + continuation[position - len(source)]] = 0
            raw[1, N_SPECIAL + 6] = 0

        outputs.append(net.generate_relation(
            prompts, 2, mode=mode, literal_policy="content_bytes_v1",
            variates=uniforms, on_logits=literal, observer=events.append,
        ))
        statuses.append(events[-1].statuses)
    assert all(torch.equal(outputs[0], output) for output in outputs[1:])
    assert outputs[0].tolist() == [
        [*[N_SPECIAL + byte for byte in source], N_SPECIAL + 6, N_SPECIAL],
        [*[N_SPECIAL + byte for byte in terminal], 0, 0],
    ]
    assert [[status.cause for status in group] for group in statuses] == [
        ["halt", "prompt_halt"], ["halt", "prompt_halt"], ["halt", "prompt_halt"],
    ]


@pytest.mark.parametrize("mode", ["standard", "predicted_copy", "oracle_copy"])
def test_common_cdf_boundaries_and_last_positive_bucket(mode):
    net = model()
    predecessor = torch.nextafter(torch.tensor(1.0), torch.tensor(0.0)).item()
    uniforms = torch.tensor([[0.0], [0.5], [predecessor]], dtype=torch.float32)

    def logits(position, raw):
        raw.fill_(-1000)
        raw[:, :N_SPECIAL] = 1000  # Forbidden mass must not affect content-byte CDF.
        raw[0:2, N_SPECIAL + 10] = 0
        raw[0:2, N_SPECIAL + 20] = 0
        raw[2, N_SPECIAL + 255] = 0

    output = net.generate_relation(
        [b"", b"", b""], 1, mode=mode, literal_policy="content_bytes_v1",
        variates=uniforms, on_logits=logits,
    )
    assert output[:, 0].tolist() == [N_SPECIAL + 10, N_SPECIAL + 20, N_SPECIAL + 255]


@pytest.mark.parametrize("mode", ["standard", "predicted_copy", "oracle_copy"])
def test_common_terminal_and_zero_allocate_nothing(monkeypatch, mode):
    net = model()
    monkeypatch.setattr("dm.models.transformer.LayerCache", lambda *a, **k: pytest.fail("allocation"))
    assert net.generate_relation([b"\x00"], 3, mode=mode,
                                 literal_policy="content_bytes_v1").tolist() == [[2]]
    assert net.generate_relation([b""], 0, mode=mode,
                                 literal_policy="content_bytes_v1").shape == (1, 0)


@pytest.mark.parametrize("mode", ["standard", "predicted_copy", "oracle_copy"])
def test_common_uniform_rounding_and_numeric_failure(mode):
    net = model()
    with pytest.raises(ValueError, match="sampler dtype"):
        net.generate_relation([b""], 1, mode=mode, literal_policy="content_bytes_v1",
                              variates=torch.tensor([[1 - 1e-10]], dtype=torch.float64))
    with pytest.raises(ValueError, match="nonfinite"):
        net.generate_relation([b""], 1, mode=mode, literal_policy="content_bytes_v1",
                              on_logits=lambda pos, raw: raw.fill_(float("nan")))


@pytest.mark.parametrize("mode", ["standard", "predicted_copy", "oracle_copy"])
def test_common_preflight_refuses_before_work_or_observation(monkeypatch, mode):
    net = model()
    monkeypatch.setattr(net, "generate_relation", partial(net.generate_relation, mode=mode))
    events = []
    monkeypatch.setattr("dm.models.transformer.LayerCache",
                        lambda *args, **kwargs: pytest.fail("KV allocation"))
    monkeypatch.setattr("dm.models.transformer.RelationBoundaryCache",
                        lambda *args, **kwargs: pytest.fail("boundary allocation"))
    requests = (
        lambda: net.generate_relation([], 1, literal_policy="content_bytes_v1",
                                      observer=events.append),
        lambda: net.generate_relation([b"", b"\x06"], 1, literal_policy="content_bytes_v1",
                                      observer=events.append),
        lambda: net.generate_relation([bytearray()], 1, literal_policy="content_bytes_v1",
                                      observer=events.append),
        lambda: net.generate_relation([b""], 129, literal_policy="content_bytes_v1",
                                      observer=events.append),
        lambda: net.generate_relation([b""], 1, literal_policy="unknown",
                                      observer=events.append),
        lambda: net.generate_relation([b""], 1, mode="predicted_copy",
                                      literal_policy="content_bytes_v1", oracle_actions={},
                                      observer=events.append),
        lambda: net.generate_relation([b""], 1, literal_policy="content_bytes_v1",
                                      variates=torch.tensor([[float("nan")]]),
                                      observer=events.append),
    )
    for request in requests:
        with pytest.raises(ValueError):
            request()
    assert events == []


def test_common_early_return_status_and_monitor_priming():
    class Monitor:
        stride = 1

        def __init__(self):
            self.calls = []

        def step(self, chunk):
            self.calls.append(chunk.copy())
            return np.zeros(chunk.shape[1], dtype=bool)

    for mode in ("standard", "predicted_copy", "oracle_copy"):
        net = model()
        monitor = Monitor()
        events = []
        result = net.generate_relation(
            [b"\x00"], 0, mode=mode, literal_policy="content_bytes_v1",
            monitor=monitor, observer=events.append,
        )
        assert result.tolist() == [[N_SPECIAL]] and monitor.calls == []
        status = events[-1].statuses[0]
        assert status.cause == "zero_horizon" and status.prompt_halt
        assert all(value == 0 for _, value in events[-1].totals)

        monitor = Monitor()
        events = []
        result = net.generate_relation(
            [b"\x06\x00"], 3, mode=mode, literal_policy="content_bytes_v1",
            monitor=monitor, observer=events.append,
        )
        assert result.tolist() == [[N_SPECIAL + 6, N_SPECIAL]]
        assert len(monitor.calls) == 2
        assert events[-1].statuses[0].cause == "prompt_halt"
        assert all(value == 0 for _, value in events[-1].totals)


def test_common_monitor_capabilities_and_monotonicity():
    net = model()

    class Mask:
        stride = 1
        allowed = None

        def step(self, chunk):
            return np.zeros(1, dtype=bool)

    with pytest.raises(ValueError, match="allowed"):
        net.generate_relation([b""], 1, literal_policy="content_bytes_v1", monitor=Mask())

    class BadStride:
        stride = 2

        def step(self, chunk):
            return np.zeros(1, dtype=bool)

    with pytest.raises(ValueError, match="stride 1"):
        net.generate_relation([b""], 1, literal_policy="content_bytes_v1",
                              monitor=BadStride())

    class Nonmonotone:
        stride = 1

        def __init__(self):
            self.calls = 0

        def step(self, chunk):
            self.calls += 1
            return np.array([self.calls == 1, False], dtype=bool)

    def two_steps(position, raw):
        raw.fill_(-1000)
        raw[:, N_SPECIAL + (6 if position == 0 else 0)] = 0

    with pytest.raises(ValueError, match="monotone"):
        net.generate_relation([b"", b""], 2, mode="predicted_copy",
                              literal_policy="content_bytes_v1", monitor=Nonmonotone(),
                              variates=torch.zeros(2, 2), on_logits=two_steps)
    for mode in ("standard", "predicted_copy"):
        with pytest.raises(ValueError, match="oracle"):
            net.generate_relation([b""], 0, mode=mode, literal_policy="content_bytes_v1",
                                  oracle_actions={})


@pytest.mark.parametrize("status", [
    [False],
    np.array([0], dtype=np.int64),
    np.zeros(2, dtype=bool),
])
def test_common_monitor_status_domain_is_loud(status):
    class Invalid:
        stride = 1

        def step(self, chunk):
            return status

    with pytest.raises(ValueError, match="row-aligned boolean ndarray"):
        model().generate_relation([b""], 1, literal_policy="content_bytes_v1",
                                  monitor=Invalid())


@pytest.mark.parametrize("byte,cause", [(0, "halt"), (255, "invalid_opcode"),
                                         (7, "non_flat_opcode")])
def test_common_stop_priority_retains_coincident_flags(byte, cause):
    class Stop:
        stride = 1

        def step(self, chunk):
            return np.ones(1, dtype=bool)

    events = []
    model().generate_relation(
        [b""], 1, mode="predicted_copy", literal_policy="content_bytes_v1",
        monitor=Stop(), observer=events.append, variates=torch.zeros(1, 1),
        on_logits=lambda position, raw: (
            raw.fill_(-1000), raw[:, N_SPECIAL + byte].fill_(0)
        ),
    )
    status = events[-1].statuses[0]
    assert status.cause == cause
    assert status.external_monitor_stop and status.at_horizon
