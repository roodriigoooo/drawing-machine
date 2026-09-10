"""R4.6 relation decode seam: delegation, queues and exact COPY execution."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from dm.isa.asm import assemble
from dm.isa.codec import BOS, ByteCodec
from dm.isa.transform import Transform
from dm.models.transformer import Config, DrawingLM
from dm.relation import CopyActionKey, FaultCode, TransducerFault


def _model(*, relation: bool) -> DrawingLM:
    torch.manual_seed(51)
    return DrawingLM(Config(vocab_size=ByteCodec.vocab_size, d_model=16, n_layers=1,
                            n_heads=2, max_len=128,
                            relation_schema="span_affine_v1" if relation else "none")).eval()


def test_relation_standard_is_an_exact_direct_delegate():
    model = _model(relation=False)
    prompt = b"\x03\x10\x10"  # a complete MOVE instruction
    encoded = torch.tensor([[2 + value for value in prompt]], dtype=torch.long)
    variates = torch.full((1, len(prompt) + 3), 0.5)
    direct = model.generate(1, 3, bos=BOS, prompt=encoded, device="cpu", variates=variates)
    relation = model.generate_relation([prompt], 3, variates=variates)
    assert torch.equal(relation, direct)


def test_oracle_copy_queues_heterogeneous_bytes_without_a_target_argument():
    model = _model(relation=True)
    source = assemble("MOVE 112 120\nLINE 128 136")
    key = CopyActionKey(len(source), 0, len(source), Transform(), 2)
    output = model.generate_relation([source], len(source), mode="oracle_copy",
                                     oracle_actions={(0, len(source)): key})
    assert output[0].tolist() == [2 + value for value in source + source]


def test_predicted_emit_masks_pad_bos_before_position_keyed_inverse_cdf():
    model = _model(relation=True)
    # Zero previously selects PAD's interval and crashed while it was coerced to a byte.
    variates = torch.tensor([[0.0]], dtype=torch.float32)
    predicted = model.generate_relation([b""], 1, mode="predicted_copy", variates=variates)
    assert predicted.item() >= 2


def test_oracle_count_five_needs_no_relation_head_and_uses_oracle_support():
    model = _model(relation=False)
    source = assemble("MOVE 112 120\nLINE 128 136")
    key = CopyActionKey(len(source), 0, len(source), Transform(), 5)
    output = model.generate_relation([source], 4 * len(source), mode="oracle_copy",
                                     oracle_actions={(0, len(source)): key})
    assert output[0].tolist() == [2 + value for value in source * 5]


def test_terminal_prompt_cannot_execute_oracle_copy():
    model = _model(relation=False)
    program = assemble("MOVE 112 120\nLINE 128 136\nHALT")
    source_stop = len(program) - 1
    action = CopyActionKey(len(program), 0, source_stop, Transform(), 2)
    output = model.generate_relation([program], 8, mode="oracle_copy",
                                     oracle_actions={(0, len(program)): action})
    assert output[0, :len(program)].tolist() == [2 + value for value in program]
    assert set(output[0, len(program):].tolist()) <= {0}


def test_dead_tail_after_halt_is_refused_before_oracle_action_selection():
    model = _model(relation=False)
    dead_tail = assemble("MOVE 112 120\nHALT\nMOVE 128 136")
    with pytest.raises(ValueError, match="bytes after HALT"):
        model.generate_relation([dead_tail], 1, mode="oracle_copy")


def test_copied_rows_do_not_call_the_token_sampler(monkeypatch):
    model = _model(relation=False)
    source = assemble("MOVE 112 120\nLINE 128 136")
    action = CopyActionKey(len(source), 0, len(source), Transform(), 2)
    calls: list[tuple[int, ...]] = []
    original = torch.multinomial

    def observed(input, *args, **kwargs):
        calls.append(tuple(input.shape))
        return original(input, *args, **kwargs)

    monkeypatch.setattr(torch, "multinomial", observed)
    model.generate_relation([source, source], len(source), mode="oracle_copy",
                            oracle_actions={(0, len(source)): action,
                                            (1, len(source)): action})
    assert calls == []


def test_relation_decode_refuses_ragged_and_invalid_oracle_requests():
    model = _model(relation=True)
    with pytest.raises(ValueError, match="equal prompt"):
        model.generate_relation([b"", b"\x03\x10\x10"], 1, mode="predicted_copy")
    source = assemble("MOVE 112 120\nLINE 128 136")
    bad = CopyActionKey(0, 0, len(source), Transform(), 2)
    with pytest.raises(TransducerFault) as refusal:
        model.generate_relation([source], 1, mode="oracle_copy",
                                oracle_actions={(0, len(source)): bad})
    assert refusal.value.code is FaultCode.BOUNDARY


@pytest.mark.parametrize("mode,kwargs,prompt", [
    ("predicted_copy", {}, b""),
    ("predicted_copy", {"oracle_actions": {}}, b""),
    ("oracle_copy", {}, b"\xff"),
])
def test_zero_horizon_does_not_bypass_request_validation(mode, kwargs, prompt):
    model = _model(relation=False)
    with pytest.raises(ValueError):
        model.generate_relation([prompt], 0, mode=mode, **kwargs)


def test_malformed_prompt_is_refused_before_cache_allocation(monkeypatch):
    model = _model(relation=False)
    def allocation(*args, **kwargs):
        pytest.fail("invalid request reached KV allocation")
    monkeypatch.setattr("dm.models.transformer.LayerCache", allocation)
    with pytest.raises(ValueError, match="bytes after HALT"):
        model.generate_relation([assemble("HALT\nMOVE 20 20")], 3, mode="oracle_copy")


def test_boundary_state_storage_owns_values_and_is_bounded():
    from dm.models.transformer import RelationBoundaryCache
    cache = RelationBoundaryCache(2, capacity=7, d_model=16, device="cpu", dtype=torch.float32)
    values = torch.randn(2, 5, 16)
    expected = values[0, 2].clone()
    cache.put(0, 6, values[0, 2])
    values.zero_()
    assert torch.equal(cache.get(0, 6), expected)
    assert cache.storage_bytes == 2 * 7 * 16 * 4
    with pytest.raises(ValueError, match="boundary"):
        cache.get(1, 6)


def test_heterogeneous_copy_states_logits_and_sampling_match_full_forwards(monkeypatch):
    import numpy as np

    from dm.isa.codec import PAD
    from dm.models.transformer import RelationBoundaryCache

    model = _model(relation=False)
    source = assemble("MOVE 112 120\nLINE 128 136")
    programs = [source * 2, source * 3]
    actions = {(row, len(source)): CopyActionKey(len(source), 0, len(source), Transform(), row + 2)
               for row in range(2)}
    logits_seen = []
    states_seen = []
    original_put = RelationBoundaryCache.put
    def observe_state(cache, row, boundary, state):
        states_seen.append((row, boundary, state.clone()))
        return original_put(cache, row, boundary, state)
    monkeypatch.setattr(RelationBoundaryCache, "put", observe_state)
    sample_shapes = []
    halt = 2 + assemble("HALT")[0]
    def sampler(probabilities, count):
        sample_shapes.append(tuple(probabilities.shape))
        return torch.full((len(probabilities), 1), halt, dtype=torch.long)
    monkeypatch.setattr(torch, "multinomial", sampler)
    class Stops:
        stride = 1
        def __init__(self):
            self.done = np.zeros(2, dtype=bool)
        def step(self, tokens):
            self.done |= tokens[0] == halt
            return self.done.copy()
    output = model.generate_relation([source, source], len(source) * 2 + 1,
                                     mode="oracle_copy", oracle_actions=actions, monitor=Stops(),
                                     on_logits=lambda pos, value: logits_seen.append((pos, value.clone())))
    assert sample_shapes == [(1, ByteCodec.vocab_size)] * 2
    for row, program in enumerate(programs):
        expected = [2 + value for value in program + assemble("HALT")]
        assert output[row, :len(expected)].tolist() == expected
        assert (output[row, len(expected):] == PAD).all()
    # Independently reconstruct each real row's history. No copied row can
    # accidentally attend to a future byte or to padding used by its neighbor.
    for position, raw in logits_seen:
        for row, program in enumerate(programs):
            if position > len(program):
                continue
            idx = torch.tensor([[BOS, *[2 + value for value in program[:position]]]])
            expected = model(idx)[:, -1]
            torch.testing.assert_close(raw[row], expected[0], atol=2e-7, rtol=2e-6)
    for row, boundary, state in states_seen:
        idx = torch.tensor([[BOS, *[2 + value for value in programs[row][:boundary]]]])
        _, expected = model(idx, return_state=True)
        torch.testing.assert_close(state, expected[0, -1], atol=2e-6, rtol=2e-6)


def test_predicted_copy_executes_the_same_action_and_queries_only_after_queue(monkeypatch):
    from dataclasses import replace

    from dm.models.relation import COPY, EMIT

    model = _model(relation=True)
    source = assemble("MOVE 112 120\nLINE 128 136")
    calls = []
    score = model.score_relation
    def scripted(packed):
        value = score(packed)
        boundary = int(packed.source_stop.max())
        calls.append(boundary)
        gate = torch.full_like(value.gate_logits, -1000)
        gate[:, COPY if boundary == len(source) else EMIT] = 0
        fields = {}
        for name, index in (("d4_logits", 0), ("dx_logits", 1), ("dy_logits", 1),
                            ("count_logits", 0)):
            logits = torch.full_like(getattr(value, name), -1000)
            logits[:, index] = 0
            fields[name] = logits
        spans = torch.full_like(value.span_logits, -1000)
        spans[(packed.source_start == 0) & (packed.source_stop == len(source))] = 0
        return replace(value, gate_logits=gate, span_logits=spans, **fields)
    monkeypatch.setattr(model, "score_relation", scripted)
    monkeypatch.setattr(torch, "multinomial", lambda probs, n: torch.full(
        (len(probs), 1), 2 + assemble("HALT")[0], dtype=torch.long))
    result = model.generate_relation([source], len(source) + 1, mode="predicted_copy")
    assert result[0].tolist() == [2 + value for value in source * 2 + assemble("HALT")]
    assert calls == [len(source), 2 * len(source)]


def test_standard_delegate_preserves_torch_rng_exactly():
    model = _model(relation=True)
    source = assemble("MOVE 112 120\nLINE 128 136")
    prompt = torch.tensor([[2 + byte for byte in source]])
    start = torch.get_rng_state().clone()
    py_start = random.getstate()
    np_start = np.random.get_state()
    direct = model.generate(1, 4, prompt=prompt)
    assert random.getstate() == py_start
    np_direct = np.random.get_state()
    assert np_direct[0] == np_start[0] and np_direct[2:] == np_start[2:]
    assert np.array_equal(np_direct[1], np_start[1])
    direct_end = torch.get_rng_state().clone()
    torch.set_rng_state(start)
    delegated = model.generate_relation([source], 4)
    delegated_end = torch.get_rng_state().clone()
    assert torch.equal(direct, delegated)
    assert torch.equal(direct_end, delegated_end)
    assert random.getstate() == py_start
    np_delegated = np.random.get_state()
    assert np_delegated[0] == np_start[0] and np_delegated[2:] == np_start[2:]
    assert np.array_equal(np_delegated[1], np_start[1])


@pytest.mark.parametrize("mode", ["standard", "predicted_copy", "oracle_copy"])
@pytest.mark.parametrize("failure,cause", [(255, "invalid_opcode"), (7, "non_flat_opcode")])
@pytest.mark.parametrize("at", [0, 1, 2])
def test_generated_failure_is_retained_beside_live_row_without_late_parsing(
        monkeypatch, mode, failure, cause, at):
    from dm.models import transformer

    model = _model(relation=True)
    failed = [6] * at + [failure]
    live = [6, 6, 0]
    parsed = []
    original = transformer.candidate_spans

    def candidate_spy(prefix, boundary):
        parsed.append(prefix)
        return original(prefix, boundary)

    monkeypatch.setattr(transformer, "candidate_spans", candidate_spy)

    def scripted(position, raw):
        raw.fill_(-1000)
        raw[0, 2 + failed[min(position, len(failed) - 1)]] = 0
        raw[1, 2 + live[position]] = 0

    events = []
    result = model.generate_relation(
        [b"", b""], 3, mode=mode, literal_policy="content_bytes_v1",
        variates=torch.zeros(2, 3), on_logits=scripted, observer=events.append,
        observation_level="actions",
    )
    assert result[0].tolist() == [*[2 + byte for byte in failed], *([0] * (2 - at))]
    assert result[1].tolist() == [2 + byte for byte in live]
    assert events[-1].statuses[0].cause == cause
    assert events[-1].statuses[1].cause == "halt"
    assert not any(bytes((failure,)) in prefix for prefix in parsed)
    assert not any(event.row == 0 and event.byte_offset > at for event in events
                   if hasattr(event, "action_source"))


def test_standard_delegate_preserves_callbacks_monitor_and_variates_exactly():
    import numpy as np
    class Monitor:
        stride = 1
        def __init__(self):
            self.readings = []
        def step(self, tokens):
            self.readings.append(tokens.copy())
            return np.zeros(1, dtype=bool)
    model = _model(relation=True)
    source = assemble("MOVE 112 120\nLINE 128 136")
    direct_logits, delegated_logits = [], []
    direct_monitor, delegated_monitor = Monitor(), Monitor()
    variates = torch.linspace(0.01, 0.99, len(source) + 3).unsqueeze(0)
    direct = model.generate(1, 3, prompt=torch.tensor([[2 + b for b in source]]),
                            monitor=direct_monitor, variates=variates,
                            on_logits=lambda pos, raw: direct_logits.append((pos, raw.clone())))
    delegated = model.generate_relation([source], 3, monitor=delegated_monitor, variates=variates,
                                        on_logits=lambda pos, raw: delegated_logits.append((pos, raw.clone())))
    assert torch.equal(direct, delegated)
    for (p1, x1), (p2, x2) in zip(direct_logits, delegated_logits, strict=True):
        assert p1 == p2 and torch.equal(x1, x2)
    assert len(direct_monitor.readings) == len(delegated_monitor.readings)
    assert all(np.array_equal(a, b) for a, b in zip(direct_monitor.readings,
                                                  delegated_monitor.readings, strict=True))
