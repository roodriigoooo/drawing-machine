"""R2: exact target-free COPY execution and synchronized byte queues."""

from __future__ import annotations

import ast
import inspect
from itertools import product

import pytest
import torch

from dm.data import relation
from dm.data.augment import Affine, apply
from dm.isa.asm import assemble
from dm.isa.codec import BOS, ByteCodec
from dm.isa.transform import D4, Transform
from dm.models.transformer import Config, DrawingLM, LayerCache
from dm.relation import (
    ORACLE_COUNT_SUPPORT,
    ORACLE_SUPPORT,
    CopyActionKey,
    FaultCode,
    RowByteQueues,
    TransducerFault,
    execute_copy,
    transducer,
)


def _source() -> bytes:
    """Central coordinates keep every frozen D4/translation image on canvas."""
    return assemble(
        "MOVE 112 120\nCURVE 120 120 128 136 140 128\n"
        "CIRCLE 7\nWIDTH 4\nFILL")


def _key(source: bytes, *, step: Transform | None = None, count: int = 2,
         start: int = 0, stop: int | None = None, boundary: int | None = None) -> CopyActionKey:
    step = Transform() if step is None else step
    stop = len(source) if stop is None else stop
    boundary = len(source) if boundary is None else boundary
    return CopyActionKey(boundary, start, stop, step, count)


@pytest.mark.parametrize("code", range(8))
@pytest.mark.parametrize("dx,dy", tuple(product((-32, 0, 32), repeat=2)))
@pytest.mark.parametrize("count", (2, 3, 4))
def test_predicted_execution_is_exact_for_the_full_frozen_factor_support(code, dx, dy, count):
    source = _source()
    step = Transform(D4.of(code), dx, dy)
    got = execute_copy(source, _key(source, step=step, count=count), max_len=512)
    want = b"".join(apply(source, Affine.of(step.power(power)))
                    for power in range(1, count))
    assert got.appended == want
    # Kind-directed rewriting keeps radii and widths unchanged.
    assert got.appended.count(bytes([7])) >= count - 1
    assert got.appended.count(bytes([4])) >= count - 1


def test_oracle_count_domain_is_explicit_finite_and_shares_the_executor():
    assert ORACLE_COUNT_SUPPORT == (2, 3, 4, 5, 6)
    source = _source()
    key = _key(source, step=Transform(D4(), 0, 0), count=6)
    with pytest.raises(TransducerFault) as predicted:
        execute_copy(source, key, max_len=512)
    assert predicted.value.code is FaultCode.UNSUPPORTED_COUNT
    oracle = execute_copy(source, key, policy=ORACLE_SUPPORT, max_len=512)
    assert oracle.appended == source * 5
    with pytest.raises(TransducerFault) as outside_oracle:
        execute_copy(source, _key(source, count=7), policy=ORACLE_SUPPORT, max_len=512)
    assert outside_oracle.value.code is FaultCode.UNSUPPORTED_COUNT


def test_action_derives_nested_bytes_only_from_its_current_prefix():
    source = assemble("MOVE 110 110\nLINE 130 120")
    inner_key = _key(source, step=Transform(D4(), 32, 0), count=3)
    inner = execute_copy(source, inner_key, max_len=128).appended
    prefix = source + inner
    outer_key = _key(prefix, step=Transform(D4.of(2), 0, 0), count=2)
    outer = execute_copy(prefix, outer_key, max_len=256).appended
    assert outer == apply(prefix, Affine.of(outer_key.step))


@pytest.mark.parametrize(
    "prefix,key,code",
    [
        (bytes([1, 10]), CopyActionKey(2, 0, 2, Transform(), 2), FaultCode.MALFORMED_PREFIX),
        (_source(), CopyActionKey(1, 0, len(_source()), Transform(), 2), FaultCode.BOUNDARY),
        (_source(), CopyActionKey(len(_source()), 1, len(_source()), Transform(), 2), FaultCode.BOUNDARY),
        (_source(), CopyActionKey(len(_source()), 0, len(_source()), Transform(), 1), FaultCode.UNSUPPORTED_COUNT),
        (_source(), CopyActionKey(len(_source()), 0, len(_source()), Transform(D4(), 7, 0), 2), FaultCode.UNSUPPORTED_TRANSLATION),
        (assemble("MOVE 10 10\nLINE 20 20\nHALT"),
         CopyActionKey(len(assemble("MOVE 10 10\nLINE 20 20\nHALT")), 0,
                       len(assemble("MOVE 10 10\nLINE 20 20\nHALT")), Transform(), 2),
         FaultCode.HALT_SOURCE),
        (assemble("REPEAT 2 0 0\nMOVE 10 10\nENDREP"),
         CopyActionKey(len(assemble("REPEAT 2 0 0\nMOVE 10 10\nENDREP")), 0, 4,
                       Transform(), 2), FaultCode.NON_FLAT_PREFIX),
        (assemble("CALL 1\nMOVE 10 10\nLINE 20 20"),
         CopyActionKey(len(assemble("CALL 1\nMOVE 10 10\nLINE 20 20")), 1, 7,
                       Transform(), 2), FaultCode.NON_FLAT_PREFIX),
        (assemble("COLOR 9\nMOVE 10 10\nLINE 20 20"),
         CopyActionKey(len(assemble("COLOR 9\nMOVE 10 10\nLINE 20 20")), 1, 7,
                       Transform(), 2), FaultCode.NON_FLAT_PREFIX),
    ],
)
def test_invalid_prefix_or_key_fails_closed(prefix, key, code):
    with pytest.raises(TransducerFault) as refusal:
        execute_copy(prefix, key, max_len=512)
    assert refusal.value.code is code


def test_canvas_and_length_failures_are_transactional():
    source = assemble("MOVE 240 240\nLINE 245 245")
    canvas_key = _key(source, step=Transform(D4(), 32, 0))
    before = bytes(source)
    with pytest.raises(TransducerFault) as canvas:
        execute_copy(source, canvas_key, max_len=128)
    assert canvas.value.code is FaultCode.CANVAS
    assert source == before

    with pytest.raises(TransducerFault) as length:
        execute_copy(_source(), _key(_source(), count=4), max_len=len(_source()) + 1)
    assert length.value.code is FaultCode.MAX_LENGTH


@pytest.mark.parametrize(
    "action",
    [
        CopyActionKey(0.0, 0, 0, Transform(), 2),
        CopyActionKey(False, 0, 0, Transform(), 2),
        CopyActionKey(0, 0.0, 0, Transform(), 2),
        CopyActionKey(0, 0, 0, "bad", 2),  # type: ignore[arg-type]
        CopyActionKey(0, 0, 0, Transform(D4(), True, 0), 2),
    ],
)
def test_malformed_public_action_values_are_typed_refusals(action):
    with pytest.raises(TransducerFault) as refusal:
        execute_copy(_source(), action, max_len=512)
    assert refusal.value.code is FaultCode.ACTION_TYPE


def test_overlength_preflight_happens_before_any_affine_rewrite(monkeypatch):
    source = _source()
    monkeypatch.setattr(transducer, "apply", lambda *_: pytest.fail("must not transform"))
    with pytest.raises(TransducerFault) as refusal:
        execute_copy(source, _key(source, count=4), max_len=len(source) + 1)
    assert refusal.value.code is FaultCode.MAX_LENGTH


@pytest.mark.parametrize(
    "prefix,key,code",
    [
        (assemble("MOVE 10 10"),
         CopyActionKey(3, 0, 3, Transform(), 2), FaultCode.SOURCE_INSTRUCTIONS),
        (assemble("WIDTH 1\nWIDTH 2\nFILL\nFILL"),
         CopyActionKey(6, 0, 6, Transform(), 2), FaultCode.SOURCE_COORDINATES),
        (assemble("MOVE 10 10\nLINE 20 20\nFILL\nFILL\nFILL\nFILL\nFILL"),
         CopyActionKey(11, 0, 6, Transform(), 2), FaultCode.SOURCE_GAP),
        (assemble("\n".join(f"LINE {i} {i}" for i in range(65))),
         CopyActionKey(195, 0, 195, Transform(), 2), FaultCode.SOURCE_INSTRUCTIONS),
    ],
)
def test_frozen_source_span_bounds_are_enforced(prefix, key, code):
    with pytest.raises(TransducerFault) as refusal:
        execute_copy(prefix, key, max_len=512)
    assert refusal.value.code is code


def test_small_traced_corpus_replays_to_the_exact_flat_vm_and_render_result():
    build = relation.build_venue1(18, seed=31, motifs=relation.motif_pool(24, seed=30),
                                  tuples=relation.venue1_tuples())
    assert build.cases
    for case in build.cases:
        reconstructed = bytearray()
        position = 0
        for action in case.actions:
            reconstructed += case.flat[position:action.boundary]
            block = execute_copy(bytes(reconstructed), action.runtime_key(), max_len=4096).appended
            assert block == case.flat[action.target_start:action.target_stop]
            reconstructed += block
            position = action.target_stop
        reconstructed += case.flat[position:]
        assert bytes(reconstructed) == case.flat
        original = relation.VM().run(case.flat)
        replay = relation.VM().run(bytes(reconstructed))
        assert replay.strokes == original.strokes
        assert relation.to_svg(replay) == relation.to_svg(original)


def test_legacy_action_serialization_is_unchanged_while_runtime_key_drops_targets():
    action = relation.CopyAction(
        boundary=6, source_start=0, source_stop=6, step=Transform(D4(), 32, 0),
        total_count=2, target_start=6, target_stop=12)
    assert action.runtime_key().key() == action.key()
    assert not hasattr(action.runtime_key(), "target_start")


def test_queues_are_heterogeneous_and_never_supply_padding():
    queues = RowByteQueues(3)
    queues.enqueue(0, b"abc")
    queues.enqueue(2, b"Z")
    assert [queues.pop_position(), queues.pop_position(), queues.pop_position()] == [
        (97, None, 90), (98, None, None), (99, None, None)]
    assert queues.pop_position() is None
    with pytest.raises(ValueError):
        queues.enqueue(0, b"")


def test_byte_at_a_time_queue_fill_matches_uncached_state_and_logits_without_sampling_copies():
    """R2 cache fill is one encoded COPY byte per step; queues consume no RNG."""
    torch.manual_seed(4)
    cfg = Config(vocab_size=258, d_model=16, n_layers=1, n_heads=2, max_len=32)
    model = DrawingLM(cfg).eval()
    codec = ByteCodec()
    source = assemble("MOVE 112 120\nLINE 128 136")
    copied = [
        execute_copy(source, _key(source, count=3), max_len=64).appended,
        execute_copy(source, _key(source, count=2), max_len=64).appended,
    ]
    encoded = [codec.encode(block) for block in copied]
    prefix_tokens = [BOS, *codec.encode(source)]
    queues = RowByteQueues(2)
    queues.enqueue(0, copied[0])
    queues.enqueue(1, copied[1])
    sampled_rows: list[int] = []

    def sample(row: int) -> int:
        sampled_rows.append(row)
        return codec.encode(bytes((40 + len(sampled_rows),)))[0]

    suffixes: list[list[int]] = [[], []]
    device = torch.device("cpu")
    caches = [LayerCache(2, cfg.n_heads, 32, cfg.d_head, device, torch.float32)
              for _ in model.blocks]

    with torch.no_grad():
        # Cache BOS plus the real ByteCodec encoding of the source prefix.
        for position, token in enumerate(prefix_tokens):
            idx = torch.full((2, 1), token, dtype=torch.long)
            x = model.embed(idx)
            cos, sin = model._rope(position, 1, device, x.dtype)
            for block, cache in zip(model.blocks, caches):
                x = block(x, cos, sin, cache)
        for position, queued in enumerate(iter(queues.pop_position, None), start=len(prefix_tokens)):
            token = []
            for row, byte in enumerate(queued):
                if byte is None:
                    token.append(sample(row))
                else:
                    token.append(codec.encode(bytes((byte,)))[0])
                    assert token[-1] == encoded[row][len(suffixes[row])]
            for row, symbol in enumerate(token):
                suffixes[row].append(symbol)
            idx = torch.tensor(token, dtype=torch.long).unsqueeze(1)
            x = model.embed(idx)
            cos, sin = model._rope(position, 1, device, x.dtype)
            for block, cache in zip(model.blocks, caches):
                x = block(x, cos, sin, cache)
            cached_state = model.norm(x)
            cached_logits = model.head(cached_state)
            full = torch.tensor([prefix_tokens + suffix for suffix in suffixes], dtype=torch.long)
            uncached_logits, uncached_state = model(full, return_state=True)
            assert torch.allclose(cached_state, uncached_state[:, -1:], atol=1e-5)
            assert torch.allclose(cached_logits, uncached_logits[:, -1:], atol=1e-5)

    assert sampled_rows == [1] * (len(copied[0]) - len(copied[1]))


def test_runtime_package_has_no_corpus_or_neural_dependency():
    for module in ("dm.relation.spec", "dm.relation.transducer", "dm.relation.queue"):
        source = inspect.getsource(__import__(module, fromlist=["*"]))
        imports = [node.module for node in ast.walk(ast.parse(source))
                   if isinstance(node, ast.ImportFrom) and node.module]
        assert not any(name.endswith("data.relation") for name in imports)
        assert not any(name.startswith(("torch", "dm.models"))
                       for name in imports)
