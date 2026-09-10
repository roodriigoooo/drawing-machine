import numpy as np
import pytest
import torch

from dm.data import synthetic
from dm.isa.codec import CODECS
from dm.models.transformer import Config, DrawingLM, LayerCache
from dm.vm.interp import VM

CORPUS = synthetic.dataset(64, seed=5)


@pytest.mark.parametrize("codec", list(CODECS.values()), ids=lambda c: c.name)
def test_symbols_to_bytes_agrees_with_decode(codec):
    """The sampler's vectorised path and the reference decoder must not drift;
    if they do, generation stops somewhere the VM does not.

    Against `stream_bytes`, not against the program: a relative codec's symbols
    spell the delta view, and the halt monitor reads them in that domain. The
    two domains share every opcode byte, so the monitor's verdict is the same
    either way -- which is the property `test_halt_monitor_stops_where_the_vm_
    stops` checks for all eight codecs.
    """
    for program in CORPUS[:16]:
        symbols = np.array(codec.encode(program))
        chunk = symbols.reshape(-1, codec.stride).T  # (stride, n_bytes)
        assert bytes(codec.symbols_to_bytes(chunk).tolist()) == codec.stream_bytes(program)


@pytest.mark.parametrize("codec", list(CODECS.values()), ids=lambda c: c.name)
def test_halt_monitor_stops_where_the_vm_stops(codec):
    """The contract the monitor exists for: truncating a stream at the monitor's
    verdict cannot change the trace. Matching the HALT *symbol* instead breaks
    this under every untyped alphabet -- byte 0x00 is the opcode and the operand
    value zero, and a raw match cuts programs mid-instruction.
    """
    rng, vm = np.random.default_rng(0), VM()
    # Real programs concatenated with garbage: the stream must stop at the first
    # executable HALT and ignore everything the VM would never reach.
    streams = [
        codec.encode(program) + rng.integers(0, codec.vocab_size, 8 * codec.stride).tolist()
        for program in CORPUS[:16]
    ]
    width = max(len(s) for s in streams)
    padded = np.zeros((len(streams), width), dtype=np.int64)  # PAD-filled
    for i, s in enumerate(streams):
        padded[i, : len(s)] = s

    monitor = codec.halt_monitor(len(streams))
    stop = np.full(len(streams), width, dtype=int)
    for end in range(codec.stride, width + 1, codec.stride):
        before = monitor.done.copy()
        done = monitor.step(padded[:, end - codec.stride : end].T)
        stop[done & ~before] = end

    for row, cut in zip(padded, stop):
        full = vm.run(codec.decode(row.tolist()))
        cropped = vm.run(codec.decode(row[:cut].tolist()))
        assert full.strokes == cropped.strokes
        assert [f.kind for f in full.faults] == [f.kind for f in cropped.faults]


def test_halt_monitor_is_not_fooled_by_a_zero_operand():
    """The regression that made the byte arm look like it could not terminate:
    `MOVE 0 0` contains the HALT byte twice, at operand positions."""
    from dm.isa.asm import assemble

    program = assemble("MOVE 0 0\nLINE 9 9\nHALT")
    for codec in CODECS.values():
        symbols = np.array(codec.encode(program))
        monitor = codec.halt_monitor(1)
        stop = None
        for end in range(codec.stride, len(symbols) + 1, codec.stride):
            if monitor.step(symbols[end - codec.stride : end].reshape(-1, 1))[0]:
                stop = end
                break
        assert stop == len(symbols), codec.name  # the final HALT, not a zero operand
        assert codec.decode(symbols[:stop].tolist()) == program, codec.name


def test_kv_cache_matches_full_forward():
    """The cached decode path is a separate code path from training; if it
    silently diverges, every sampling metric is measuring a different model."""
    torch.manual_seed(0)
    cfg = Config(vocab_size=32, d_model=32, n_layers=2, n_heads=2, max_len=64)
    model = DrawingLM(cfg).eval()
    idx = torch.randint(0, 32, (2, 12))

    with torch.no_grad():
        full = model(idx)
        caches = [
            LayerCache(2, cfg.n_heads, idx.shape[1], cfg.d_head, idx.device, torch.float32)
            for _ in model.blocks
        ]
        steps = []
        for pos in range(idx.shape[1]):
            x = model.embed(idx[:, pos : pos + 1])
            cos, sin = model._rope(pos, 1, idx.device, x.dtype)
            for block, cache in zip(model.blocks, caches):
                x = block(x, cos, sin, cache)
            steps.append(model.head(model.norm(x)))
        incremental = torch.cat(steps, dim=1)

    assert torch.allclose(full, incremental, atol=1e-4)


def test_generate_stops_early_on_halt():
    torch.manual_seed(0)
    codec = CODECS["byte"]
    cfg = Config(vocab_size=codec.vocab_size, d_model=32, n_layers=2, n_heads=2, max_len=64)
    out = DrawingLM(cfg).generate(4, max_new=48, monitor=codec.halt_monitor(4))
    assert out.shape[0] == 4 and out.shape[1] <= 48


def test_bit_generation_only_halts_on_a_byte_boundary():
    """A zero byte is HALT only when it starts on a byte boundary. Stopping on
    an unaligned run of eight zero bits would truncate valid programs, and the
    truncation would land in the bit arm's validity score as the codec's fault."""
    torch.manual_seed(0)
    codec = CODECS["bit"]
    cfg = Config(vocab_size=codec.vocab_size, d_model=32, n_layers=2, n_heads=2, max_len=512)
    out = DrawingLM(cfg).generate(4, max_new=256, monitor=codec.halt_monitor(4))
    assert out.shape[1] % 8 == 0


def test_analytic_param_count_matches_reality():
    for shape in (dict(d_model=192, n_layers=2), dict(d_model=128, n_layers=4),
                  dict(d_model=96, n_layers=8)):
        cfg = Config(vocab_size=258, n_heads=4, **shape)
        model = DrawingLM(cfg)
        assert cfg.n_params() == model.n_params()
        assert model.n_params() < 1_000_000, f"{shape} busts the sub-1M budget"


def _tiny(codec, **kw) -> DrawingLM:
    torch.manual_seed(0)
    return DrawingLM(
        Config(vocab_size=codec.vocab_size, d_model=32, n_layers=2, n_heads=2,
               max_len=256, **kw)
    )


def test_a_prompt_is_returned_verbatim_and_generation_continues_after_it():
    """The AR model reconstructs nothing, so a *paired* comparison against a
    held-out program only exists if the model can be made to continue that
    program's prefix. The prefix must come back unaltered or the pair is not a
    pair."""
    codec = CODECS["byte"]
    prompt = torch.tensor([codec.encode(b"\x01\x10\x20")] * 3)
    out = _tiny(codec).generate(3, max_new=6, prompt=prompt)

    assert out.shape == (3, 3 + 6)
    assert torch.equal(out[:, :3], prompt)


def test_a_prompted_decode_matches_an_uncached_forward_pass():
    """The one thing most likely to be wrong: RoPE offsets and cache depth after
    a prefill block. Greedy decoding from a prompt has to reproduce, token for
    token, what recomputing the whole prefix from scratch would give.
    """
    codec = CODECS["byte"]
    model = _tiny(codec)
    prompt = torch.tensor([codec.encode(b"\x01\x10\x20\x02\x30")])

    out = model.generate(1, max_new=5, top_k=1, prompt=prompt, monitor=None)

    # Same thing with no cache at all: full forward pass over the growing
    # sequence, argmax, append.
    sequence = torch.cat([torch.tensor([[1]]), prompt], dim=1)  # BOS + prompt
    for _ in range(5):
        nxt = model(sequence)[:, -1].argmax(dim=-1, keepdim=True)
        sequence = torch.cat([sequence, nxt], dim=1)

    assert torch.equal(out, sequence[:, 1:]), "prefill and recompute disagree"


def test_the_halt_monitor_starts_from_the_prompt_s_parse_state():
    """A prompt that ends mid-instruction leaves operand bytes owed. Starting
    the monitor at an instruction boundary instead would read the next operand
    `0x00` as HALT -- the exact fault the parse-state monitor exists to prevent,
    reintroduced through the prompt."""
    codec = CODECS["byte"]
    monitor = codec.halt_monitor(1)
    # MOVE takes two operands; give the opcode and one of them.
    prompt = torch.tensor([codec.encode(b"\x01\x10")])
    _tiny(codec).generate(1, max_new=1, prompt=prompt, monitor=monitor)

    # One operand byte still owed at the point generation began; the single
    # generated byte then consumes it, so nothing is owed now and no HALT can
    # have been recognised inside the instruction.
    assert monitor.pending.tolist() == [0]
    assert monitor.done.tolist() == [False]


def test_a_prompt_that_is_not_a_whole_number_of_bytecode_bytes_is_refused():
    """Under the bit codec a prompt of 3 symbols leaves the monitor reading
    bytes out of phase for the rest of the run, and every halt verdict after it
    is meaningless. Fail loudly rather than return a plausible block."""
    codec = CODECS["bit"]
    prompt = torch.zeros((2, 3), dtype=torch.long)
    with pytest.raises(ValueError, match="whole number of bytecode bytes"):
        _tiny(codec).generate(2, max_new=8, prompt=prompt, monitor=codec.halt_monitor(2))


def test_the_kv_cache_is_sized_for_the_prompt_as_well_as_the_new_symbols():
    """`LayerCache` is preallocated -- that is what fixed the 40 GiB decode --
    so a prompt that is not counted in its capacity overruns the buffer instead
    of growing it."""
    codec = CODECS["byte"]
    prompt = torch.tensor([codec.encode(bytes(range(1, 41)))] * 2)
    out = _tiny(codec).generate(2, max_new=20, prompt=prompt, monitor=None)
    assert out.shape == (2, 60)


def test_share_non_embedding_init_is_identical_across_vocabularies():
    """The point of the whole thing: two codecs, same layer weights.

    Without this the arms of a paired comparison start from unrelated networks,
    which on Tier B was worth ~2 bits/drawing -- more than any axis under test.
    """
    import torch

    from dm.models.transformer import Config, DrawingLM

    shape = dict(d_model=64, n_layers=2, n_heads=4)
    a = DrawingLM(Config(vocab_size=258, **shape)).share_non_embedding_init(0)
    b = DrawingLM(Config(vocab_size=1293, **shape)).share_non_embedding_init(0)

    named_a, named_b = dict(a.named_parameters()), dict(b.named_parameters())
    assert set(named_a) == set(named_b)
    shared = [n for n in named_a if n != "embed.weight"]
    assert shared, "nothing was shared"
    for name in shared:
        assert torch.equal(named_a[name], named_b[name]), name

    # The embedding is exactly what the codecs differ in, and stays untouched.
    assert named_a["embed.weight"].shape != named_b["embed.weight"].shape


def test_share_non_embedding_init_still_varies_with_seed():
    """Variance reduction, not variance removal: seeds must still differ."""
    import torch

    from dm.models.transformer import Config, DrawingLM

    cfg = Config(vocab_size=258, d_model=64, n_layers=2, n_heads=4)
    a = DrawingLM(cfg).share_non_embedding_init(0)
    b = DrawingLM(cfg).share_non_embedding_init(1)
    assert not torch.equal(
        dict(a.named_parameters())["blocks.0.ff.down.weight"],
        dict(b.named_parameters())["blocks.0.ff.down.weight"],
    )
