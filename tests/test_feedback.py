"""F1: the feedback path.

**Every test here was committed at F0, before the code existed**, as a module of
strict xfails. That ordering is the point: a test written after the
implementation is a description of what the implementation happens to do, and
`docs/directions.md` §6 puts the tests first for exactly that reason. F1 removed
the marker and made them pass; the git history holds both halves.

The interface every test is written against is frozen in
`dm.eval.feedback_contract.MODEL_INTERFACE`; the bytes the `none` path must keep
reproducing are frozen in `tests/fixtures/feedback_baseline_v1.json`.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dm.eval import feedback_contract as contract
from dm.eval import feedback_fixture as fixture
from dm.isa.asm import assemble
from dm.isa.codec import CODECS
from dm.models.transformer import (
    SOURCE_FEEDBACK_SCHEMA,
    SOURCE_FEEDBACK_SCHEMA_V2,
    Config,
    DrawingLM,
)

BASELINE = fixture.load_fixture()
TINY = fixture.TINY
D_MODEL = TINY["d_model"]


def _cell(name: str) -> dict:
    return next(cell for cell in BASELINE["cells"] if cell["name"] == name)


def _config(codec: str = "byte", schema: str = "glu_v1", **kw) -> Config:
    return Config(vocab_size=CODECS[codec].vocab_size, feedback_schema=schema,
                  d_model=TINY["d_model"], n_layers=TINY["n_layers"],
                  n_heads=TINY["n_heads"], max_len=TINY["max_len"], **kw)


def _model(codec: str = "byte", seed: int = 0, schema: str = "glu_v1",
           **kw) -> DrawingLM:
    torch.manual_seed(seed)
    return DrawingLM(_config(codec, schema, **kw)).eval()


def _prompt(codec: str = "byte") -> torch.Tensor:
    rows = [CODECS[codec].encode(assemble(text)[:fixture.PROMPT_BYTES])
            for text in fixture.PROGRAMS]
    return torch.tensor(rows, dtype=torch.long)


def _variates(rows: int, width: int, seed: int = 11) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(rows, width, generator=generator, dtype=torch.float32)


# ---------------------------------------------------------------------------
# compatibility: the old path stays exactly the old path


def test_an_old_config_dictionary_loads_as_the_no_feedback_schema():
    """§3.2 requirement 1. Every checkpoint in `runs/` predates the field, and a
    config that failed to load would take the whole Direction 2 record with it."""
    for cell in BASELINE["cells"]:
        cfg = Config(**cell["config"])
        assert getattr(cfg, contract.CONFIG_FIELD) == contract.DEFAULT_FEEDBACK_SCHEMA


def test_an_old_state_dictionary_loads_strict_and_cannot_be_promoted_silently():
    """§3.2 requirement 2, and the trap on the other side of it.

    Strict both ways. An old checkpoint has to load into a `none` model, and it
    has to *fail* to load into a `glu_v1` one: the feedback tensors are missing,
    and a non-strict load would fill them with a fresh draw and hand back a model
    that decodes through a recurrent input distribution it was never trained on.
    """
    for cell in BASELINE["cells"]:
        if "state" not in cell:
            continue
        spec = fixture.CellSpec(**cell["spec"])
        frozen = {name: fixture.decode_tensor(value)
                  for name, value in cell["state"].items()}

        plain = DrawingLM(_config(spec.codec, "none", n_classes=spec.n_classes,
                                  abs_pos=spec.abs_pos))
        plain.load_state_dict(frozen, strict=True)
        assert not any(name in dict(plain.named_parameters())
                       for name in contract.FEEDBACK_PARAMETERS)

        promoted = DrawingLM(_config(spec.codec, "glu_v1",
                                     n_classes=spec.n_classes,
                                     abs_pos=spec.abs_pos))
        with pytest.raises(RuntimeError, match="[Mm]issing key"):
            promoted.load_state_dict(frozen, strict=True)


def test_the_none_schema_reproduces_the_frozen_baseline_bit_for_bit():
    """§3.2 requirement 3, and the reason the fixture was captured at F0.

    An explicit `feedback_schema="none"` has to be the same function as the code
    that existed before the field did -- not close to it. `verify_fixture`
    compares exact bits; a tolerance here would make every later stage's
    equivalence claim mean "within some tolerance nobody wrote down".
    """
    for cell in BASELINE["cells"]:
        spec = fixture.CellSpec(**cell["spec"])
        model = DrawingLM(_config(spec.codec, "none", n_classes=spec.n_classes,
                                  abs_pos=spec.abs_pos))
        assert getattr(model.cfg, contract.CONFIG_FIELD) == "none"
    verdicts = fixture.verify_fixture(BASELINE)
    assert all(verdict.ok for verdict in verdicts), fixture.describe_failure(
        BASELINE, verdicts
    )


def test_the_none_schema_creates_no_feedback_parameters():
    """§3.2 requirement 2. A parameter that exists but is never read still
    changes the optimiser state, the checkpoint bytes and the count."""
    model = _model(schema="none")
    names = dict(model.named_parameters())
    assert not any(name in names for name in contract.FEEDBACK_PARAMETERS)
    assert model.n_params() == _cell("byte_standard")["n_params"]


def test_an_unknown_feedback_schema_is_refused():
    """A typo that silently disabled feedback would look exactly like a null
    result, which is the one failure mode this pilot cannot afford."""
    with pytest.raises(ValueError, match="feedback_schema"):
        DrawingLM(_config(schema="glu_v2"))


def test_the_model_and_the_frozen_contract_name_the_same_things():
    """The protocol is the freeze and the model is the implementation. They are
    two files, so nothing but a test stops them drifting -- and a drifted
    constant would be discovered at F7, after eight checkpoints exist."""
    from dm.models import transformer

    assert transformer.FEEDBACK_SCHEMAS == contract.FEEDBACK_SCHEMAS
    assert transformer.RUNTIME_MODES == contract.RUNTIME_MODES
    assert transformer.FUSED_NORM_GAIN == contract.FUSED_NORM_GAIN
    assert Config(vocab_size=2).feedback_schema == contract.DEFAULT_FEEDBACK_SCHEMA


@pytest.mark.parametrize("representation", contract.REPRESENTATIONS)
def test_a_pilot_sized_checkpoint_costs_what_the_protocol_says(representation):
    """The counts F6 will actually train, checked against the frozen table rather
    than against the formula that produced it."""
    row = contract.parameter_table()[representation]
    cfg = Config(vocab_size=CODECS[representation].vocab_size,
                 d_model=contract.PILOT_D_MODEL, n_layers=contract.PILOT_N_LAYERS,
                 n_heads=contract.PILOT_N_HEADS, feedback_schema="glu_v1")
    assert cfg.n_params() == row["feedback_capable"] < contract.PARAMETER_BUDGET
    assert DrawingLM(cfg).n_params() == row["feedback_capable"]


def test_a_feedback_mode_is_refused_on_an_ordinary_checkpoint():
    """The trap `PLAN.md` names first: switching feedback on for a checkpoint
    that never trained with it decodes through a recurrent input distribution it
    has never seen, and the output is entirely plausible."""
    plain = _model(schema="none")
    for mode in ("soft", "fused"):
        with pytest.raises(ValueError, match="feedback-capable"):
            plain.generate(1, max_new=2, mode=mode)
    with pytest.raises(ValueError, match="feedback_schema='none'"):
        plain(torch.zeros(1, 3, dtype=torch.long),
              feedback=torch.zeros(1, 3, D_MODEL))


def test_an_unknown_decode_mode_is_refused():
    with pytest.raises(ValueError, match="unknown decode mode"):
        _model().generate(1, max_new=2, mode="softish")


def test_misaligned_carried_state_is_refused_rather_than_broadcast():
    """`feedback` is aligned to `idx` by the caller. A silently broadcast or
    truncated block would fuse every position with the wrong state and still
    produce a loss curve."""
    model = _model()
    idx = torch.zeros(2, 5, dtype=torch.long)
    for bad in (torch.zeros(2, 4, D_MODEL), torch.zeros(1, 5, D_MODEL),
                torch.zeros(2, 5, D_MODEL + 1)):
        with pytest.raises(ValueError, match="feedback is"):
            model(idx, feedback=bad)


# ---------------------------------------------------------------------------
# the fusion itself


def test_glu_v1_owns_exactly_the_three_frozen_tensors():
    """Names are checkpoint contract: eight checkpoints are written at F6 and
    reloaded strict at F7 and F8."""
    model = _model()
    names = dict(model.named_parameters())
    for name in contract.FEEDBACK_PARAMETERS:
        assert name in names, name
    assert names["fuse.up.weight"].shape == (D_MODEL, D_MODEL)
    assert names["fuse.gate.weight"].shape == (D_MODEL, D_MODEL)
    assert names["fuse.norm.weight"].shape == (D_MODEL,)
    # Bias-free, both of them: a bias on the gate would let the channel be
    # switched on without the token embedding saying anything.
    assert not any(name.startswith("fuse.") and name.endswith(".bias")
                   for name in names)


def test_the_overhead_is_two_d_squared_plus_d_in_the_live_module():
    """`2D^2 + D`, and `Config.n_params()` has to agree with the module it
    describes -- which is what caught the missing RMSNorm gain in the first
    place."""
    model = _model()
    assert model.n_params() == model.cfg.n_params()
    assert (model.n_params() - _cell("byte_standard")["n_params"]
            == contract.feedback_overhead(D_MODEL))


def test_the_fused_norm_gain_starts_at_the_embedding_scale():
    """Gain 1 would hand the stack an input whose RMS is ~50x the embedding's at
    step 0, so the opening optimiser steps would be spent undoing the
    initialisation rather than learning the channel."""
    gain = dict(_model().named_parameters())["fuse.norm.weight"]
    assert torch.equal(gain, torch.full_like(gain, contract.FUSED_NORM_GAIN))


def test_the_fusion_is_the_frozen_gated_product():
    """`z_t = RMSNorm(W_U h_(t-1) * sigmoid(W_G e_t))`, computed by hand.

    Hidden state is the value and the token embedding is the gate, in that order.
    Swapping them is a different mechanism that would produce entirely plausible
    training curves.
    """
    model = _model()
    params = dict(model.named_parameters())
    hidden = torch.randn(2, 5, D_MODEL)
    embed = torch.randn(2, 5, D_MODEL)

    product = (hidden @ params["fuse.up.weight"].T) * torch.sigmoid(
        embed @ params["fuse.gate.weight"].T
    )
    expected = params["fuse.norm.weight"] * product * torch.rsqrt(
        product.pow(2).mean(-1, keepdim=True) + 1e-6
    )
    assert torch.allclose(model.fuse(hidden, embed), expected, atol=0, rtol=0)


def test_the_source_arm_normalizes_the_embedding_before_the_gate():
    """The paper's Listing 3 normalizes ``e`` before ``W_G``.

    Keep the old raw-embedding ``glu_v1`` checkpoint path loadable, but make the
    source comparison an explicit architecture arm so an engineering checkpoint
    cannot be relabelled as source-faithful after the fact.
    """
    model = _model(schema=SOURCE_FEEDBACK_SCHEMA)
    assert model.fuse.normalize_gate is True
    params = dict(model.named_parameters())
    hidden = torch.randn(2, 5, D_MODEL)
    embed = torch.randn(2, 5, D_MODEL)
    normalized = embed * torch.rsqrt(
        embed.pow(2).mean(-1, keepdim=True) + 1e-6
    )
    product = (hidden @ params["fuse.up.weight"].T) * torch.sigmoid(
        normalized @ params["fuse.gate.weight"].T
    )
    expected = params["fuse.norm.weight"] * product * torch.rsqrt(
        product.pow(2).mean(-1, keepdim=True) + 1e-6
    )
    assert torch.allclose(model.fuse(hidden, embed), expected, atol=0, rtol=0)
    assert model.n_params() == model.cfg.n_params()
    assert {name for name in params if name.startswith("fuse.")} == set(
        contract.FEEDBACK_PARAMETERS
    )


def test_the_legacy_and_source_arms_are_distinct_explicit_configurations():
    legacy = _model(schema="glu_v1")
    source = _model(schema=SOURCE_FEEDBACK_SCHEMA)
    assert legacy.fuse.normalize_gate is False
    assert source.fuse.normalize_gate is True
    assert contract.SOURCE_FEEDBACK_SCHEMA == SOURCE_FEEDBACK_SCHEMA


# ---------------------------------------------------------------------------
# glu_source_v2: the Listing-3-faithful seam


def _learned_norm(model: DrawingLM, x: torch.Tensor) -> torch.Tensor:
    """`fuse.norm` computed by hand, so a test never asks the module under test
    what the module under test does."""
    gain = dict(model.named_parameters())["fuse.norm.weight"]
    return gain * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


def _affine_free(x: torch.Tensor) -> torch.Tensor:
    """`glu_source_v1`'s gate normalization: RMS with no learned gain."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


def _plain_keep(rows: int, width: int, plain: int) -> torch.Tensor:
    """The `(B, T, 1)` plain-prefix selector the model hands the fusion."""
    return (torch.arange(width)[None, :] < plain)[..., None].expand(rows, width, 1)


def test_the_faithful_arm_uses_one_learned_norm_at_the_gate_and_the_stack_input():
    """Listing 3 lines 9 and 11 name the *same* module, `input_rmsnorm_1`:

    ```text
    r_t = W_U h_(t-1) * sigmoid(W_G N(e_t))
    m_t = prefix_mixin(r_t, e_t)
    z_t = N(m_t)
    ```

    One `N`, learned, in both places. `glu_source_v1` reads the gate through an
    affine-free normalization and the stack input through a separate learned
    gain, which is two normalization operations where the source has one.
    """
    model = _model(schema=SOURCE_FEEDBACK_SCHEMA_V2)
    params = dict(model.named_parameters())
    hidden = torch.randn(2, 5, D_MODEL)
    embed = torch.randn(2, 5, D_MODEL)

    product = (hidden @ params["fuse.up.weight"].T) * torch.sigmoid(
        _learned_norm(model, embed) @ params["fuse.gate.weight"].T
    )
    assert torch.equal(model.fuse(hidden, embed), _learned_norm(model, product))

    # The gain is 0.02, so a learned gate input and an affine-free one differ by
    # ~50x before `W_G` -- this is a different function, not a rescaling of one.
    unshared = (hidden @ params["fuse.up.weight"].T) * torch.sigmoid(
        _affine_free(embed) @ params["fuse.gate.weight"].T
    )
    assert not torch.allclose(model.fuse(hidden, embed),
                              _learned_norm(model, unshared), atol=1e-6)


def test_the_faithful_arm_mixes_the_plain_prefix_in_before_the_shared_norm():
    """Listing 3 line 10 is `prefix_mixin`, line 11 is the norm -- in that order.

    Selecting after the norm, which is what the two earlier arms do, leaves the
    plain prefix at the raw embedding and so lets those positions bypass the
    stack's input scale entirely.
    """
    model = _model(schema=SOURCE_FEEDBACK_SCHEMA_V2)
    hidden = torch.randn(2, 6, D_MODEL)
    embed = torch.randn(2, 6, D_MODEL)
    keep = _plain_keep(2, 6, plain=2)

    fused = model.fuse(hidden, embed)
    mixed = model.fuse.stack_input(hidden, embed, keep)
    assert torch.equal(mixed[:, :2], _learned_norm(model, embed)[:, :2])
    assert torch.equal(mixed[:, 2:], fused[:, 2:])
    assert not torch.allclose(mixed, torch.where(keep, embed, fused), atol=1e-6)


def test_the_faithful_arm_gives_a_fully_plain_pass_the_normalized_embedding():
    """A later training pass whose sampled prefix covers the row still runs
    through Listing 3 line 11, so its stack input is `N(e)` and not `e`.

    The `plain >= width` shortcut that returns the raw embedding is correct for
    the two pre-Listing-3 arms and wrong here, which is why it is a property of
    the arm rather than of the caller.
    """
    idx = fixture.decode_tensor(_cell("byte_standard")["inputs"])
    rows, width = idx.shape
    states = torch.randn(rows, width, D_MODEL)
    for schema, plain_is_raw in (("glu_v1", True),
                                 (SOURCE_FEEDBACK_SCHEMA, True),
                                 (SOURCE_FEEDBACK_SCHEMA_V2, False)):
        model = _model(schema=schema)
        standard = model(idx)
        all_plain = model(idx, feedback=states, plain=width)
        assert torch.equal(all_plain, standard) == plain_is_raw, schema
        # The int shortcut and the per-row tensor path must agree, or a trainer
        # drawing a full-length prefix would compute a different function from
        # one that happened to draw it for every row.
        per_row = model(idx, feedback=states,
                        plain=torch.full((rows,), width, dtype=torch.long))
        assert torch.equal(per_row, all_plain), schema


def test_the_faithful_arm_leaves_pass_one_and_the_standard_prefill_raw():
    """Listing 3 line 5 is `model(e)` and Listing 2 line 1 prefills the same way.

    The asymmetry against line 11 is deliberate: a pass with no carried state has
    nothing to mix in, so there is no mixin to normalize. Structurally, `feedback
    is None` is the only condition that returns the raw embedding.
    """
    model = _model(schema=SOURCE_FEEDBACK_SCHEMA_V2)
    idx = fixture.decode_tensor(_cell("byte_standard")["inputs"])
    prompt = _prompt()
    variates = _variates(prompt.shape[0], prompt.shape[1] + 6)
    before_logits = model(idx)
    before_decode = model.generate(prompt.shape[0], max_new=6, prompt=prompt,
                                   variates=variates, top_k=40, mode="standard")
    with torch.no_grad():
        for name in contract.FEEDBACK_PARAMETERS:
            dict(model.named_parameters())[name].add_(1.5)
    assert torch.equal(model(idx), before_logits)
    assert torch.equal(
        model.generate(prompt.shape[0], max_new=6, prompt=prompt,
                       variates=variates, top_k=40, mode="standard"),
        before_decode,
    )


def test_the_faithful_fused_prefill_normalizes_bos_and_a_soft_prefill_does_not():
    """The two prefills are different calls and Listing 3 applies to one of them.

    `fused` runs a second prefill *through* the mixin, so BOS -- the one position
    with no predecessor state -- reaches the stack as `N(e)`. `soft` keeps the
    standard prefill's raw inputs, which is Listing 2 unchanged. Losing that
    distinction would silently change what a soft decode conditions on.
    """
    model = _model(schema=SOURCE_FEEDBACK_SCHEMA_V2)
    hidden = torch.randn(2, 4, D_MODEL)
    embed = torch.randn(2, 4, D_MODEL)
    bos_only = model.fuse.stack_input(hidden, embed, _plain_keep(2, 4, plain=1))
    assert torch.equal(bos_only[:, 0], _learned_norm(model, embed)[:, 0])
    assert not torch.allclose(bos_only[:, 0], embed[:, 0], atol=1e-6)

    prompt = _prompt()
    rows = prompt.shape[0]
    standard_first, soft_first = [], []
    for mode, sink in (("standard", standard_first), ("soft", soft_first)):
        model.generate(rows, max_new=2, prompt=prompt, monitor=None, top_k=1,
                       mode=mode,
                       on_logits=lambda _s, raw, box=sink: box.append(raw.clone()))
    assert torch.equal(standard_first[0], soft_first[0])


@pytest.mark.parametrize(
    "schema", ("glu_v1", SOURCE_FEEDBACK_SCHEMA, SOURCE_FEEDBACK_SCHEMA_V2)
)
def test_a_fused_decode_agrees_cached_and_uncached(schema):
    """The fused prefill's plain prefix is BOS alone, so the whole decode *is*
    expressible as a growing whole-sequence forward -- and the cached loop has to
    equal it in every arm.

    A soft decode has no such twin under `glu_source_v2`: its prompt carries the
    standard prefill's raw inputs while its generated positions carry `N(m)`, and
    one `plain` count cannot express two input normalizations. That asymmetry is
    the source's, not this implementation's, and
    `test_teacher_forcing_and_generation_are_the_same_decode` pins the soft path
    against a second independent loop instead.
    """
    model = _model(schema=schema)
    prompt = _prompt()
    rows, given = prompt.shape
    max_new = 4
    cached = model.generate(rows, max_new=max_new, prompt=prompt, monitor=None,
                            top_k=1, mode="fused")

    sequence = torch.cat([torch.ones(rows, 1, dtype=torch.long), prompt], dim=1)
    _, plain_states = model(sequence, return_state=True)
    # The prompt block fuses the *standard* prefill's states and keeps them: the
    # fused prefill happens once, and its own states are what the generated
    # positions carry. Re-deriving the prompt's feedback from the fused pass
    # would be a second fused prefill, which §3.3 does not run.
    prefill = torch.zeros_like(plain_states)
    prefill[:, 1:] = plain_states[:, :-1]
    logits, states = model(sequence, feedback=prefill, plain=1, return_state=True)
    for _ in range(max_new):
        nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
        sequence = torch.cat([sequence, nxt], dim=1)
        carried = torch.zeros(rows, sequence.shape[1], D_MODEL)
        carried[:, : given + 1] = prefill
        carried[:, given + 1 :] = states[:, given:]
        logits, states = model(sequence, feedback=carried, plain=1,
                               return_state=True)
    assert torch.equal(cached, sequence[:, 1:])


def test_the_faithful_arm_keeps_the_frozen_tensor_names_and_overhead():
    """The seam moved; the checkpoint contract did not. Same three names, same
    `2D^2 + D`, same 0.02 gain -- so v0's frozen parameter table still describes
    the arm the qualification runs."""
    model = _model(schema=SOURCE_FEEDBACK_SCHEMA_V2)
    names = dict(model.named_parameters())
    assert {name for name in names if name.startswith("fuse.")} == set(
        contract.FEEDBACK_PARAMETERS
    )
    assert names["fuse.up.weight"].shape == (D_MODEL, D_MODEL)
    assert names["fuse.gate.weight"].shape == (D_MODEL, D_MODEL)
    assert names["fuse.norm.weight"].shape == (D_MODEL,)
    assert model.n_params() == model.cfg.n_params()
    assert (model.n_params() - _cell("byte_standard")["n_params"]
            == contract.feedback_overhead(D_MODEL))
    gain = names["fuse.norm.weight"]
    assert torch.equal(gain, torch.full_like(gain, contract.FUSED_NORM_GAIN))


def test_the_two_earlier_arms_keep_their_frozen_seam_exactly():
    """Adding the faithful arm must not move a checkpoint that already exists.

    `glu_v1` gates on the raw embedding, `glu_source_v1` on an affine-free
    normalization of it, both normalize the product with the learned gain, and
    both hand a plain position its raw embedding after that norm.
    """
    hidden = torch.randn(2, 5, D_MODEL)
    embed = torch.randn(2, 5, D_MODEL)
    keep = _plain_keep(2, 5, plain=2)
    for schema, gate_input in (("glu_v1", embed),
                               (SOURCE_FEEDBACK_SCHEMA, _affine_free(embed))):
        model = _model(schema=schema)
        params = dict(model.named_parameters())
        product = (hidden @ params["fuse.up.weight"].T) * torch.sigmoid(
            gate_input @ params["fuse.gate.weight"].T
        )
        expected = _learned_norm(model, product)
        assert torch.equal(model.fuse(hidden, embed), expected), schema
        assert torch.equal(model.fuse.stack_input(hidden, embed, keep),
                           torch.where(keep, embed, expected)), schema


def test_the_three_feedback_arms_are_distinct_named_configurations():
    """Three arms, three names, and the protocol points at exactly one of them.

    `glu_source_v1` stays loadable rather than being corrected in place: this
    repository holds no `runs/` artifact under it, but an external one cannot be
    disproved, and a strict load that succeeds under a *different* computation is
    the one failure this project cannot detect after the fact.
    """
    arms = {schema: _model(schema=schema) for schema in
            ("glu_v1", SOURCE_FEEDBACK_SCHEMA, SOURCE_FEEDBACK_SCHEMA_V2)}
    assert [model.fuse.schema for model in arms.values()] == list(arms)
    assert contract.SOURCE_FEEDBACK_SCHEMA_V2 == SOURCE_FEEDBACK_SCHEMA_V2
    assert contract.TRAINING_DEFAULTS["feedback_schema"] == SOURCE_FEEDBACK_SCHEMA_V2
    hidden = torch.randn(2, 4, D_MODEL)
    # Embedding scale, not unit scale: at RMS 1 the affine-free normalization
    # `glu_source_v1` applies is nearly the identity, and the test would be
    # asking whether two arms differ on inputs neither of them ever sees.
    embed = torch.randn(2, 4, D_MODEL) * contract.FUSED_NORM_GAIN
    outputs = [model.fuse(hidden, embed) for model in arms.values()]
    for left in range(len(outputs)):
        for right in range(left + 1, len(outputs)):
            assert not torch.allclose(outputs[left], outputs[right], atol=1e-6)


def test_the_fusion_is_position_wise():
    """No position may see another one's state through the fusion.

    Attention is where positions mix, and it is causal. A fusion implemented with
    a scan or a convolution would still train, still look sane, and would carry
    information *backwards* across positions before the mask ever applied --
    which at evaluation would let a target byte's input depend on a later byte.
    Checked at a tolerance rather than bitwise: a matmul over a slice blocks
    differently from one over the whole block, and that is arithmetic, not
    mixing.
    """
    model = _model()
    hidden, embed = torch.randn(2, 8, D_MODEL), torch.randn(2, 8, D_MODEL)
    whole = model.fuse(hidden, embed)
    assert torch.allclose(model.fuse(hidden[:, 3:5], embed[:, 3:5]),
                          whole[:, 3:5], atol=1e-6, rtol=0)

    # And changing one position's inputs moves that position and nothing else.
    moved_hidden, moved_embed = hidden.clone(), embed.clone()
    moved_hidden[:, 4] = torch.randn(2, D_MODEL)
    moved_embed[:, 4] = torch.randn(2, D_MODEL)
    moved = model.fuse(moved_hidden, moved_embed)
    assert torch.allclose(moved[:, :4], whole[:, :4], atol=1e-6, rtol=0)
    assert torch.allclose(moved[:, 5:], whole[:, 5:], atol=1e-6, rtol=0)
    assert not torch.allclose(moved[:, 4], whole[:, 4], atol=1e-6, rtol=0)


def test_there_is_no_additive_embedding_shortcut():
    """§3.1. With a shortcut, recurrence-trained weights can ignore the wide
    channel and recover ordinary pretraining loss -- so a null result would be
    unreadable, because the model was never obliged to use what was measured."""
    model = _model()
    embed = torch.randn(2, 3, D_MODEL)
    with torch.no_grad():
        dict(model.named_parameters())["fuse.up.weight"].zero_()
    fused = model.fuse(torch.randn(2, 3, D_MODEL), embed)
    assert torch.equal(fused, torch.zeros_like(fused))


def test_the_carried_state_is_the_tensor_the_head_consumes():
    """Invariant 4: the previous *normalized* top state, `self.norm(x)`, not the
    pre-norm residual. Three tensors in a pre-norm stack answer to "the hidden
    state" and only one of them is the head's input."""
    model = _model(schema="none")
    idx = fixture.decode_tensor(_cell("byte_standard")["inputs"])
    logits, states = model(idx, return_state=True)
    assert torch.equal(logits, model.head(states))


# ---------------------------------------------------------------------------
# runtime modes


def test_standard_mode_never_reads_the_fusion_weights():
    """Invariant 3 in its strongest form: the standard arm of the primary
    contrast has to be the *same function* it was before feedback existed, or the
    contrast measures the implementation instead of the mechanism."""
    model = _model()
    idx = fixture.decode_tensor(_cell("byte_standard")["inputs"])
    prompt = _prompt()
    variates = _variates(prompt.shape[0], prompt.shape[1] + 6)

    before_logits = model(idx)
    before_decode = model.generate(prompt.shape[0], max_new=6, prompt=prompt,
                                   variates=variates, top_k=40, mode="standard")
    with torch.no_grad():
        for name in contract.FEEDBACK_PARAMETERS:
            dict(model.named_parameters())[name].add_(1.5)

    assert torch.equal(model(idx), before_logits)
    assert torch.equal(
        model.generate(prompt.shape[0], max_new=6, prompt=prompt,
                       variates=variates, top_k=40, mode="standard"),
        before_decode,
    )


def test_the_first_position_is_plain_whatever_the_feedback_holds():
    """Invariant 4. BOS has no predecessor, so position 0 cannot carry a state --
    and a feedback tensor whose row 0 was read would make the whole sequence
    depend on whatever the caller happened to leave there."""
    model = _model()
    idx = fixture.decode_tensor(_cell("byte_standard")["inputs"])
    left = torch.randn(idx.shape[0], idx.shape[1], D_MODEL)
    right = left.clone()
    right[:, 0] = torch.randn(idx.shape[0], D_MODEL)
    assert torch.equal(model(idx, feedback=left), model(idx, feedback=right))


def test_plain_positions_take_the_raw_embedding_not_a_zero_state():
    """`plain` is a count, not a mask over the feedback tensor: fusing a zero
    state is `RMSNorm(0) = 0`, which deletes the position's input entirely."""
    model = _model()
    idx = fixture.decode_tensor(_cell("byte_standard")["inputs"])
    states = torch.randn(idx.shape[0], idx.shape[1], D_MODEL)
    standard = model(idx)
    assert torch.equal(model(idx, feedback=states, plain=idx.shape[1]), standard)
    assert not torch.equal(model(idx, feedback=states, plain=1), standard)


def test_plain_may_differ_per_row():
    """F2 draws the plain-prefix length per row, so a scalar-only `plain` would
    force the trainer to run one row at a time or to fake it with padding."""
    model = _model()
    idx = fixture.decode_tensor(_cell("byte_standard")["inputs"])
    states = torch.randn(idx.shape[0], idx.shape[1], D_MODEL)
    rows = torch.tensor([1, idx.shape[1]])
    mixed = model(idx, feedback=states, plain=rows)
    assert torch.equal(mixed[1], model(idx, feedback=states, plain=idx.shape[1])[1])
    assert torch.equal(mixed[0], model(idx, feedback=states, plain=1)[0])


def test_soft_decoding_cached_equals_uncached():
    """The KV cache is a separate code path from the training forward, and a soft
    decode threads a *second* piece of state through it.

    The reference recomputes the whole sequence at every step with the states the
    previous recomputation produced. That is exact rather than approximate: BOS
    and the prompt are plain, so their states never move, and by induction every
    position's input is the same in both runs.
    """
    model = _model()
    prompt = _prompt()
    rows, given = prompt.shape
    max_new = 6
    cached = model.generate(rows, max_new=max_new, prompt=prompt, monitor=None,
                            top_k=1, mode="soft")

    sequence = torch.cat([torch.ones(rows, 1, dtype=torch.long), prompt], dim=1)
    states: torch.Tensor | None = None
    for _ in range(max_new):
        if states is None:
            logits, states = model(sequence, return_state=True)
        else:
            # `states` covers the sequence as it was *before* the last append, so
            # it is exactly one position shorter: positions 1..L-1 of the grown
            # sequence carry states 0..L-2, and position 0 stays plain.
            carried = torch.zeros(rows, sequence.shape[1], D_MODEL)
            carried[:, 1:] = states
            logits, states = model(sequence, feedback=carried, plain=given + 1,
                                   return_state=True)
        nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
        sequence = torch.cat([sequence, nxt], dim=1)

    assert torch.equal(cached, sequence[:, 1:])


def test_a_fused_prefill_replaces_the_standard_cache_rather_than_extending_it():
    """§3.3. The keys and values a standard prefill wrote were computed from
    plain embeddings; attending to them from fused queries is a decode against a
    prefix the model never saw, and it produces perfectly plausible output."""
    model = _model()
    prompt = _prompt()
    rows, given = prompt.shape
    soft = model.generate(rows, max_new=4, prompt=prompt, monitor=None, top_k=1,
                          mode="soft")
    fused = model.generate(rows, max_new=4, prompt=prompt, monitor=None, top_k=1,
                           mode="fused")
    assert not torch.equal(soft, fused)

    sequence = torch.cat([torch.ones(rows, 1, dtype=torch.long), prompt], dim=1)
    _, plain_states = model(sequence, return_state=True)
    carried = torch.zeros(rows, sequence.shape[1], D_MODEL)
    carried[:, 1:] = plain_states[:, :-1]
    logits, _ = model(sequence, feedback=carried, plain=1, return_state=True)
    assert torch.equal(fused[:, given : given + 1],
                       logits[:, -1].argmax(dim=-1, keepdim=True))


def test_the_first_generated_token_is_identical_in_standard_and_soft():
    """The structural zero §2.2 requires be reported rather than averaged away.

    Soft decoding prefills in standard mode, so the first target token's
    distribution is the same in both arms by construction. A pipeline that showed
    a difference there would be measuring its own bookkeeping.
    """
    model = _model()
    prompt = _prompt()
    seen: dict[str, torch.Tensor] = {}
    for mode in contract.EVALUATION_MODES:
        captured: list[torch.Tensor] = []
        model.generate(prompt.shape[0], max_new=3, prompt=prompt, monitor=None,
                       top_k=1, mode=mode,
                       on_logits=lambda _step, raw, sink=captured:
                           sink.append(raw.clone()))
        seen[mode] = captured[0]
    assert torch.equal(seen["standard"], seen["soft"])


def test_teacher_forcing_reproduces_a_full_forward_pass_in_standard_mode():
    """The scoring twin of `generate`, and the path Direction 3's standard arm
    runs on. It has to be the same function the batched scorer is, or
    `Delta_standard` is not the estimand Direction 2 published.

    A tolerance, not equality: one path attends through a preallocated cache a
    position at a time and the other runs one batched forward, so
    `scaled_dot_product_attention` reduces over different shapes.
    """
    model = _model()
    prompt, continuation = _prompt(), _prompt()[:, :4]
    scored = model.teacher_forced(prompt, continuation, mode="standard")

    sequence = torch.cat(
        [torch.ones(prompt.shape[0], 1, dtype=torch.long), prompt, continuation],
        dim=1,
    )
    given = prompt.shape[1]
    reference = model(sequence)[:, given : given + continuation.shape[1]]
    assert scored.shape == reference.shape
    assert torch.allclose(scored, reference, atol=1e-5)


def test_teacher_forcing_and_generation_are_the_same_decode():
    """Feed `generate` its own output back as the continuation and the logits
    have to match, in every mode. The two loops carry the same latent, the same
    cache lifetime and the same plain-BOS rule; binding them here is what stops
    one being fixed without the other."""
    model = _model()
    prompt = _prompt()
    for mode in contract.RUNTIME_MODES:
        captured: list[torch.Tensor] = []
        decoded = model.generate(
            prompt.shape[0], max_new=4, prompt=prompt, monitor=None, top_k=1,
            mode=mode, on_logits=lambda _step, raw, sink=captured:
                sink.append(raw.clone()),
        )
        forced = model.teacher_forced(prompt, decoded[:, prompt.shape[1] :],
                                      mode=mode)
        sampled = torch.stack(captured, dim=1)
        assert torch.equal(forced, sampled), mode


def test_teacher_forcing_scores_the_first_symbol_identically_in_standard_and_soft():
    model = _model()
    prompt, continuation = _prompt(), _prompt()[:, :3]
    standard = model.teacher_forced(prompt, continuation, mode="standard")
    soft = model.teacher_forced(prompt, continuation, mode="soft")
    assert torch.equal(standard[:, 0], soft[:, 0])
    assert not torch.allclose(standard[:, 1:], soft[:, 1:], atol=1e-5)


def test_teacher_forcing_refuses_an_empty_continuation_and_a_dead_channel():
    model = _model()
    with pytest.raises(ValueError, match="continuation is empty"):
        model.teacher_forced(_prompt(), _prompt()[:, :0])
    with pytest.raises(ValueError, match="feedback-capable"):
        _model(schema="none").teacher_forced(_prompt(), _prompt()[:, :2],
                                             mode="soft")


class _AllowOnly:
    """Minimal `SupportProtocol`: a fixed legal set, and it never halts.

    Hand-written rather than `dm.isa.state.StateMonitor` because the model's only
    contract is `stride`, `step` and an optional `allowed` -- and a test that
    reached for the real monitor would be asserting the ISA's behaviour while
    claiming to assert the sampler's.
    """

    stride = 1

    def __init__(self, rows: int, vocab: int, legal: tuple[int, ...]) -> None:
        self.rows, self.vocab, self.legal = rows, vocab, legal
        self.done = np.zeros(rows, dtype=bool)

    def step(self, chunk: np.ndarray) -> np.ndarray:
        return self.done

    def allowed(self) -> np.ndarray:
        mask = np.zeros((self.rows, self.vocab), dtype=bool)
        mask[:, list(self.legal)] = True
        return mask


@pytest.mark.parametrize("mode", contract.EVALUATION_MODES)
def test_the_legality_mask_still_applies_before_the_sampler_in_every_mode(mode):
    """The masked and raw decoder cells must differ in the mask and in nothing
    else, and `on_logits` must still see the distribution *before* it.

    That ordering is what makes the illegal-mass measurement a property of the
    model rather than of the sampler, and a feedback path threaded through the
    same loop is exactly the sort of change that quietly reorders it.
    """
    model = _model()
    legal = (7, 11, 23)
    captured: list[torch.Tensor] = []
    out = model.generate(
        2, max_new=6, mode=mode, top_k=40,
        variates=_variates(2, 6, seed=5),
        monitor=_AllowOnly(2, CODECS["byte"].vocab_size, legal),
        on_logits=lambda _step, raw: captured.append(raw.clone()),
    )
    assert set(out.flatten().tolist()) <= set(legal)
    assert len(captured) == 6
    # Raw means raw: nothing the mask removed has been set to -inf yet.
    assert all(bool(torch.isfinite(row).all()) for row in captured)


def test_standard_and_soft_draw_the_first_symbol_from_one_shared_stream():
    """Pairing, at the level the Direction 3 contrast needs it.

    The two modes prefill identically, so on a shared variate block they must
    also *sample* identically at the first generated position. If they did not,
    the modes would differ in their RNG consumption as well as in the mechanism,
    and every downstream difference would be partly the sampler's.
    """
    model = _model()
    prompt = _prompt()
    variates = _variates(prompt.shape[0], prompt.shape[1] + 5, seed=13)
    decoded = {
        mode: model.generate(prompt.shape[0], max_new=5, prompt=prompt,
                             monitor=None, top_k=40, variates=variates, mode=mode)
        for mode in contract.EVALUATION_MODES
    }
    given = prompt.shape[1]
    assert torch.equal(decoded["standard"][:, given], decoded["soft"][:, given])


def test_a_stopped_row_cannot_change_an_active_row():
    """Latent state is row-aligned, and a row that has halted is still occupying
    a slot in the batch. Leaking its state sideways would make a case's score
    depend on which other cases shared its decode."""
    model = _model()
    codec = CODECS["byte"]
    variates = _variates(2, 6, seed=7)

    monitor = codec.halt_monitor(2)
    monitor.step(np.array([[2, 5]], dtype=np.int64))  # row 0 has already halted
    both = model.generate(2, max_new=6, monitor=monitor, variates=variates,
                          top_k=40, mode="soft")

    alone_monitor = codec.halt_monitor(1)
    alone_monitor.step(np.array([[5]], dtype=np.int64))
    alone = model.generate(1, max_new=6, monitor=alone_monitor,
                           variates=variates[1:2], top_k=40, mode="soft")
    assert torch.equal(both[1:2, : alone.shape[1]], alone)


def test_the_latent_does_not_survive_a_request():
    """§3.3: request-local, cleared at completion. A latent that persisted would
    make the second of two identical calls a different experiment, and every
    paired comparison in the pilot runs two calls."""
    model = _model()
    prompt = _prompt()
    variates = _variates(prompt.shape[0], prompt.shape[1] + 5)
    first = model.generate(prompt.shape[0], max_new=5, prompt=prompt,
                           variates=variates, top_k=40, mode="soft")
    second = model.generate(prompt.shape[0], max_new=5, prompt=prompt,
                            variates=variates, top_k=40, mode="soft")
    assert torch.equal(first, second)


def test_classes_and_absolute_positions_are_added_after_fusion():
    """§3.1: `W_G` sees the token embedding only, and the two optional additive
    tables land on top of the fused input. Gating on a class-shifted embedding
    would make the gate a function of the conditioning signal, which is a
    different model."""
    model = _model(n_classes=3, abs_pos=True)
    idx = fixture.decode_tensor(_cell("byte_standard")["inputs"])
    classes = torch.tensor([0, 2])
    states = torch.randn(idx.shape[0], idx.shape[1], D_MODEL)
    with torch.no_grad():
        model.classes.normal_(0.0, 0.02)
        model.pos.normal_(0.0, 0.02)

    embed = model.embed(idx)
    carried = torch.zeros_like(states)
    carried[:, 1:] = states[:, :-1]
    # Fused over the whole block and then selected, which is what the model does
    # and the only formulation that supports a per-row `plain` boundary at all.
    # Fusing the tail slice instead is the same function and *not* the same
    # bits -- a matmul over 7 positions blocks differently from one over 8 --
    # and this assertion is about where fusion sits in the input pipeline, not
    # about gemm shapes. `test_the_fusion_is_position_wise` carries the claim
    # that the slice is mathematically equivalent.
    fused = model.fuse(carried, embed)
    expected = torch.cat([embed[:, :1], fused[:, 1:]], dim=1)
    expected = expected + model.pos[: idx.shape[1]] + model.classes[classes].unsqueeze(1)

    stack = expected
    cos, sin = model._rope(0, idx.shape[1], idx.device, stack.dtype)
    for block in model.blocks:
        stack = block(stack, cos, sin)
    assert torch.allclose(model(idx, classes, feedback=carried, plain=1),
                          model.head(model.norm(stack)), atol=0, rtol=0)


# ---------------------------------------------------------------------------
# shared initialisation


def test_shared_init_draws_the_feedback_matrices_across_vocabularies():
    """§3.2 requirement 6. Common random numbers is the reason two codecs are
    comparable at all -- measured at ~2 bits/drawing on Tier B, larger than any
    axis under test -- and a feedback matrix left out of it would reintroduce
    exactly that variance in the arm that carries the mechanism."""
    byte_model = _model("byte").share_non_embedding_init(0)
    bit_model = _model("bit").share_non_embedding_init(0)
    byte_params = dict(byte_model.named_parameters())
    bit_params = dict(bit_model.named_parameters())
    for name in ("fuse.up.weight", "fuse.gate.weight"):
        assert torch.equal(byte_params[name], bit_params[name]), name
        assert not torch.equal(byte_params[name],
                               torch.zeros_like(byte_params[name])), name


def test_shared_init_keeps_the_fused_norm_gain_at_the_embedding_scale():
    """The existing rule fills every 1-D parameter with 1.0, which is right for
    the stack's own RMSNorms and wrong for this one. The gain is a scale
    protection, not a draw, so it has to survive the shared-init path."""
    model = _model().share_non_embedding_init(0)
    gain = dict(model.named_parameters())["fuse.norm.weight"]
    assert torch.equal(gain, torch.full_like(gain, contract.FUSED_NORM_GAIN))


def test_shared_init_does_not_move_a_single_pre_feedback_parameter():
    """Drawing the feedback matrices consumes generator draws. Consuming them in
    the wrong place shifts every parameter after it, and the failure is invisible
    in any single arm -- the model still trains, and the two codecs have quietly
    stopped sharing an initialisation."""
    for name in ("byte_shared_init", "bit_shared_init"):
        cell = _cell(name)
        spec = fixture.CellSpec(**cell["spec"])
        frozen, _ = fixture.build_model(spec)
        model = _model(spec.codec).share_non_embedding_init(0)
        for key, tensor in frozen.state_dict().items():
            assert torch.equal(model.state_dict()[key], tensor), f"{name}:{key}"
