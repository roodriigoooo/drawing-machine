"""Sub-million-parameter decoder-only transformer.

Deliberately conventional -- pre-norm, RMSNorm, RoPE, SwiGLU, tied embeddings --
so that any difference the sweep reports is attributable to the *representation*
and not to architecture novelty.

RoPE rather than learned positions is not a style choice: the length
generalisation test (train REPEAT n<=4, test n=16) needs positions the model has
never seen, and learned position tables cannot extrapolate there at all.

Embeddings are tied because at this budget the table is not a rounding error --
it is the difference between the bit arm (vocab 4) and a fused-token arm
(vocab 4096, which would consume the whole budget before a single layer).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..isa.codec import BOS, N_SPECIAL
from ..isa.spec import Op
from ..relation import (
    ORACLE_SUPPORT,
    CopyActionKey,
    RowByteQueues,
    candidate_spans,
    execute_copy,
    prefix_index,
    prefix_progress,
)
from ..relation.decoding import (
    LEGACY_POLICY,
    STOPPING_POLICY,
    ActionDecision,
    DecodeLedger,
    DecodeObserver,
    DecodeRequestError,
    FlatOutputStop,
    RequestStarted,
    resolve_policy,
)
from .relation import (
    RELATION_SCHEMAS,
    PackedCandidates,
    RelationHead,
    RelationLayoutError,
    RelationScores,
    length_bin,
    relation_parameter_count,
)

PAD_TOKEN = 0  # mirrors dm.isa.codec.PAD; kept local so the model has no data dep
BYTE_VOCAB_SIZE = N_SPECIAL + 256

#: Latent-feedback architectures this file implements. `"none"` is the default
#: and *is* the pre-Direction-3 model: it builds no extra parameters, reads no
#: latent state, and reproduces `tests/fixtures/feedback_baseline_v1.json` bit for
#: bit. `"glu_v1"` is the original gated fusion frozen at F0.
#:
#: An unknown value is refused rather than treated as `"none"`. A typo that
#: silently disabled feedback would produce a perfectly healthy training run and
#: a null result, which is the one failure mode the Direction 3 pilot cannot
#: afford to be unable to distinguish.
FEEDBACK_SCHEMAS: tuple[str, ...] = ("none", "glu_v1")

# `glu_v1` is retained as the frozen F0 compatibility arm.  `glu_source_v1` was
# the first correction: it normalizes the token embedding before `W_G`, which is
# the unambiguous half of the source recipe.  `glu_source_v2` is the
# Listing-3-faithful seam -- one *learned* norm shared by the gate input and the
# post-mixin stack input, with the plain prefix mixed in before it.
#
# Three names rather than one corrected arm.  A checkpoint's `feedback_schema`
# is what makes `load_state_dict(strict=True)` mean something: the three arms
# own byte-identical tensor sets, so an artifact written under one of them loads
# perfectly under either of the others and computes a different function in
# silence.  This repository holds no `runs/` artifact under `glu_source_v1`, but
# an external one cannot be disproved, and the cost of keeping the name is one
# branch.
SOURCE_FEEDBACK_SCHEMA = "glu_source_v1"
SOURCE_FEEDBACK_SCHEMA_V2 = "glu_source_v2"

#: The arms that own fusion parameters, in the order they were introduced.
FUSION_SCHEMAS: tuple[str, ...] = (
    "glu_v1", SOURCE_FEEDBACK_SCHEMA, SOURCE_FEEDBACK_SCHEMA_V2
)
SUPPORTED_FEEDBACK_SCHEMAS: tuple[str, ...] = ("none", *FUSION_SCHEMAS)

#: Decode-time behaviours. Mode is an argument to `generate`, never a property of
#: a checkpoint: the primary Direction 3 contrast is standard against soft *in
#: the same weights*, and that is the only comparison with exact parameter
#: equality on both sides.
RUNTIME_MODES: tuple[str, ...] = ("standard", "soft", "fused")

#: Initial gain of the fusion's own RMSNorm, matching the 0.02 the embedding is
#: drawn at. A default gain of 1 would hand the stack a fused input whose RMS is
#: roughly 50x the embedding's before a single step, so early training would be
#: spent undoing the initialisation rather than learning the channel. Pinned in
#: `dm.eval.feedback_contract` and asserted by `tests/test_feedback.py`.
FUSED_NORM_GAIN = 0.02


class HaltProtocol(Protocol):
    """What `generate` needs in order to stop: how many symbols make a decidable
    unit, and a verdict on each row once it has them. Structural, so the model
    keeps no dependency on the ISA."""

    stride: int

    def step(self, chunk: np.ndarray) -> np.ndarray: ...


class SupportProtocol(HaltProtocol, Protocol):
    """A halt monitor that also says which symbols each row may emit next.

    Optional and duck-typed: `generate` calls `allowed()` only when the monitor
    it was given has one, so `dm.isa.codec.HaltMonitor` keeps working unchanged
    and a legality-constrained decode is the *same* sampler with one extra
    operation. That matters for the comparison this exists for -- a raw and a
    masked decoder cell must differ in the mask and in nothing else
    (`docs/state.md`).

    `allowed()` returns `(rows, vocab)` bool for the symbol about to be sampled,
    and is called once per symbol before the temperature-scaled logits are
    truncated. Rows that have already stopped must be all-True: they are
    overwritten with PAD anyway, and an all-`-inf` row makes the softmax NaN and
    the multinomial backend-defined.
    """

    def allowed(self) -> np.ndarray: ...


@dataclass
class Config:
    vocab_size: int
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    max_len: int = 2048
    rope_theta: float = 10_000.0
    dropout: float = 0.0
    #: False makes attention bidirectional, which claim 3's composition level
    #: needs and nothing else does: a denoiser over stroke slots predicts every
    #: slot from every other, and that is what "non-autoregressive over strokes"
    #: means operationally. Defaulted True so no AR run's behaviour, parameter
    #: count or record changes -- the flag is not in `SHAPES` and `Config` is
    #: serialised into every record, so an old checkpoint reloads causal.
    causal: bool = True
    #: Adds a learned absolute position table on top of RoPE.
    #:
    #: Off for the AR arms, and the reason RoPE was chosen states why: the
    #: length-generalisation test (train `REPEAT n <= 4`, test `n = 16`) needs
    #: positions never seen in training, where a learned table cannot
    #: extrapolate at all.
    #:
    #: On for claim 3's composition level, where that argument does not apply
    #: and its opposite does. The summary grid is `max_strokes x 6` and fixed,
    #: so nothing extrapolates; what the level must represent is *absolute*
    #: slot and field identity -- position 5 is a `halts` flag and position 0 is
    #: an x coordinate. Measured without it: the denoiser predicted the same
    #: distribution at every one of 192 positions, namely the corpus-wide EMPTY
    #: rate of 0.861, and the sampler drew zero strokes. RoPE is relative by
    #: construction and a grid of fields is an absolute address.
    abs_pos: bool = False
    #: Number of classes the model is conditioned on; 0 is unconditional.
    #:
    #: **The conditioning is an additive embedding, not a token, and that choice
    #: is what keeps every existing instrument working.** A class *token* would
    #: have to live in the vocabulary, embeddings here are tied to the output
    #: head, so the logits would widen past `codec.vocab_size` -- and
    #: `dm/eval/recovery.py`, `dm/eval/spelling.py` and `dm/eval/redundancy.py`
    #: all reshape logits against the codec's own width. That reshape does not
    #: raise when the width is wrong, it silently reinterprets the tensor. The
    #: additive form changes no width, adds no scored position, and costs
    #: `n_classes x d_model` parameters -- 640 at five classes, 0.08% of the
    #: budget, which is small enough to state rather than control for.
    #:
    #: Zero-initialised for the same reason `pos` is: a fresh conditional model
    #: starts at *exactly* the point its unconditional twin starts at, so the two
    #: arms of the comparison differ by an addition and not by a different draw.
    n_classes: int = 0
    #: Which latent-feedback architecture the checkpoint carries, if any.
    #:
    #: **Last field, and defaulted, because every checkpoint in `runs/` predates
    #: it.** `dm.train` serialises `asdict(model_cfg)` into each record, so an old
    #: record's config dictionary simply omits the key and rebuilds as `"none"`
    #: -- which is the whole of `docs/directions.md` §3.2 requirement 1. A field
    #: without a default, or one inserted before an existing positional argument,
    #: would take the entire Direction 1 and 2 record with it.
    #:
    #: It is checkpoint architecture and not a runtime switch: both feedback
    #: arms own parameters, so a model that has them is a different model. *Which* of the
    #: three runtime modes a decode uses is an argument to `generate`, because
    #: standard and soft have to be comparable at identical weights.
    feedback_schema: str = "none"
    #: Direction 4's optional bounded relation scorer.  Appended and defaulted
    #: for strict old-checkpoint reconstruction; it never changes ``forward`` or
    #: ``generate`` and is mutually exclusive with latent feedback in v1.
    relation_schema: str = "none"

    def __post_init__(self) -> None:
        if self.feedback_schema not in SUPPORTED_FEEDBACK_SCHEMAS:
            raise ValueError(
                f"unknown feedback_schema {self.feedback_schema!r}; expected one "
                f"of {list(SUPPORTED_FEEDBACK_SCHEMAS)}. Refused rather than defaulted: a "
                "typo that silently disabled feedback would look exactly like a "
                "null result"
            )
        if self.relation_schema not in RELATION_SCHEMAS:
            raise ValueError(
                f"unknown relation_schema {self.relation_schema!r}; expected one "
                f"of {list(RELATION_SCHEMAS)}")
        if self.relation_schema != "none" and self.feedback_schema != "none":
            raise ValueError(
                "relation_schema='span_affine_v1' may not coexist with a non-none "
                "feedback_schema in Direction 4 v1")

    @property
    def d_head(self) -> int:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must divide evenly across heads")
        return self.d_model // self.n_heads

    @property
    def d_ff(self) -> int:
        return (8 * self.d_model // 3 + 7) // 8 * 8

    def n_params(self) -> int:
        """Analytic count, tied embeddings, norms and any position table included."""
        per_layer = 4 * self.d_model**2 + 3 * self.d_model * self.d_ff + 2 * self.d_model
        positions = self.max_len * self.d_model if self.abs_pos else 0
        # `2D^2 + D`: two bias-free D x D matrices and the fusion's own RMSNorm
        # gain. The gain is the term an earlier estimate of 2D^2 = 32,768 left
        # out, and 128 parameters is trivial while a count that disagrees with
        # the module is not -- `test_analytic_param_count_matches_reality` is an
        # equality, not a bound.
        feedback = (0 if self.feedback_schema == "none"
                    else 2 * self.d_model**2 + self.d_model)
        relation = (0 if self.relation_schema == "none"
                    else relation_parameter_count(self.d_model))
        return (
            self.vocab_size * self.d_model
            + self.n_layers * per_layer
            + self.d_model
            + positions
            + self.n_classes * self.d_model
            + feedback
            + relation
        )


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def _rope_tables(seq: int, d_head: int, theta: float, device, dtype) -> tuple[Tensor, Tensor]:
    inv = 1.0 / (theta ** (torch.arange(0, d_head, 2, device=device, dtype=torch.float32) / d_head))
    ang = torch.outer(torch.arange(seq, device=device, dtype=torch.float32), inv)
    return ang.cos().to(dtype), ang.sin().to(dtype)


def _apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """x: (B, H, T, Dh). cos/sin are already sliced to this call's positions."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos, sin = cos[None, None], sin[None, None]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class LayerCache:
    """Preallocated key/value store for one attention layer.

    Appending with `torch.cat` allocates a differently-sized pair of tensors at
    every decode step. Caching allocators keep a free block per distinct size and
    never reuse across sizes, so an 800-step decode reserved 40 GiB for a model
    whose weights are 3 MiB -- which is what took the bit arm of the sweep out of
    memory. Writing into a buffer sized once bounds the footprint at its true
    size and returns views, so the hot path allocates nothing at all.
    """

    __slots__ = ("k", "n", "v")

    def __init__(self, batch: int, n_heads: int, capacity: int, d_head: int,
                 device, dtype) -> None:
        self.k = torch.empty(batch, n_heads, capacity, d_head, device=device, dtype=dtype)
        self.v = torch.empty_like(self.k)
        self.n = 0

    def append(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        end = self.n + k.shape[2]
        self.k[:, :, self.n : end] = k
        self.v[:, :, self.n : end] = v
        self.n = end
        return self.k[:, :, :end], self.v[:, :, :end]


class RelationBoundaryCache:
    """Request-owned top-layer boundary values, independent of attention KV.

    Copy states into fixed storage: retaining a slice of a batched forward also
    retains that forward's entire backing tensor. An offset map names only real
    canonical boundaries; unused slots never enter a span distribution.
    """

    def __init__(self, rows: int, *, capacity: int, d_model: int, device, dtype) -> None:
        if any(type(value) is not int or value < 1 for value in (rows, capacity, d_model)):
            raise RelationLayoutError("invalid boundary cache shape")
        self._states = torch.empty((rows, capacity, d_model), device=device, dtype=dtype)
        self._slots: list[dict[int, int]] = [{} for _ in range(rows)]

    @property
    def storage_bytes(self) -> int:
        return self._states.numel() * self._states.element_size()

    def put(self, row: int, boundary: int, state: Tensor) -> None:
        if type(row) is not int or not 0 <= row < len(self._slots) or \
                type(boundary) is not int or boundary < 0:
            raise RelationLayoutError("invalid boundary cache coordinate")
        slots = self._slots[row]
        if boundary in slots:
            raise RelationLayoutError("boundary state was already retained")
        if len(slots) == self._states.shape[1] or state.shape != self._states.shape[2:] or \
                state.device != self._states.device or state.dtype != self._states.dtype:
            raise RelationLayoutError("boundary cache capacity/state mismatch")
        slot = len(slots)
        self._states[row, slot].copy_(state.detach())
        slots[boundary] = slot

    def get(self, row: int, boundary: int) -> Tensor:
        if type(row) is not int or not 0 <= row < len(self._slots) or \
                type(boundary) is not int or boundary not in self._slots[row]:
            raise RelationLayoutError("candidate endpoint lacks a retained boundary state")
        return self._states[row, self._slots[row][boundary]]


class Attention(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.n_heads, self.d_head = cfg.n_heads, cfg.d_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout
        self.is_causal = cfg.causal

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor,
                cache: LayerCache | None = None) -> Tensor:
        b, t, _ = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        causal = self.is_causal
        if cache is not None:
            k, v = cache.append(k, v)
            # A query block spanning the whole cache is self-attention and needs
            # the mask. A shorter one is decoding against a prefix it may attend
            # to in full -- which covers both one-token steps and prompt prefill.
            causal = self.is_causal and cache.n == t
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=causal, dropout_p=self.dropout if self.training else 0.0
        )
        return self.proj(out.transpose(1, 2).reshape(b, t, -1))


class SwiGLU(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Fusion(nn.Module):
    """Gated latent fusion, with an explicit source-faithful gate variant.

    The previous *normalized top-layer* state is the value and the current token
    embedding is the gate. That asymmetry is the mechanism, not a style choice:
    the state is what the model has already computed and the embedding is what it
    is being asked about, so the token selects which parts of a finished
    computation reach the bottom of the stack again.

    **No additive embedding shortcut.** `z_t` replaces the embedding at fused
    positions rather than being added to it. With a shortcut, recurrence-trained
    weights can drive `W_U` to zero, recover ordinary next-token loss through the
    residual embedding, and score a perfectly healthy validation curve while
    never using the wide channel -- and then a null Direction 3 result would say
    nothing about accessibility, because the model was never obliged to use what
    the experiment measures (`docs/directions.md` §3.1).

    Both matrices are bias-free. A bias on the gate would let the channel be
    switched on without the token embedding saying anything, which is the same
    escape route by another route.

    Nothing here is position-aware, so the module is a pointwise function of
    `(hidden, embed)` and the optional absolute-position and class tables are
    added *after* it by `DrawingLM._embed`. `W_G` therefore sees the token
    embedding alone: gating on a class-shifted embedding would make the gate a
    function of the conditioning signal, which is a different model.

    **Three arms, and the only thing that differs is where a normalization
    sits.** `N` is this module's own learned RMSNorm, `fuse.norm`; `rms` is the
    affine-free operation; `mixin(r, e)` puts the raw embedding back at the
    positions the caller holds plain.

    ```text
    glu_v1        z = mixin(N(W_U h * sigmoid(W_G e)),      e)
    glu_source_v1 z = mixin(N(W_U h * sigmoid(W_G rms(e))), e)
    glu_source_v2 r = W_U h * sigmoid(W_G N(e))
                  z = N(mixin(r, e))
    ```

    `glu_source_v2` is the seam the source's Listing 3 actually specifies. Lines
    9 and 11 name the **same** module, `input_rmsnorm_1`, so one learned norm
    reads the gate input and the post-mixin stack input, and the mixin happens
    *before* it -- which means a position held plain reaches the stack as `N(e)`
    instead of bypassing the norm. The earlier two arms use two distinct
    normalization operations and let the plain prefix past the second one.

    Keeping the gain at `0.02` is what makes that placement coherent rather than
    merely faithful: with the norm applied to plain positions too, a gain at the
    embedding's own RMS is exactly what makes a fused pass's plain prefix the
    size of the raw-embedding prefill it exists to imitate. RMS normalization
    rescales each row, so this is the right scale and direction, not an equality.

    Listing 3 line 5 (`h = model(e)`, pass 1) and Listing 2's standard prefill
    stay raw. The asymmetry is the source's: a pass with no carried state has no
    mixin to normalize. Structurally, `feedback is None` is the *only* condition
    under which `DrawingLM._stack_input` returns the untouched embedding.
    """

    def __init__(self, cfg: Config, *, schema: str) -> None:
        super().__init__()
        if schema not in FUSION_SCHEMAS:
            raise ValueError(
                f"{schema!r} is not a fusion arm; expected one of "
                f"{list(FUSION_SCHEMAS)}"
            )
        #: Which of the three placements above this block computes. Read by
        #: `DrawingLM._stack_input`, by the qualification report and by the
        #: architecture tests; it is not a parameter and does not enter the
        #: state dictionary, so the checkpoint contract is unaffected.
        self.schema = schema
        self.up = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.gate = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.norm = RMSNorm(cfg.d_model)
        nn.init.constant_(self.norm.weight, FUSED_NORM_GAIN)

    @property
    def normalize_gate(self) -> bool:
        """Whether `W_G` reads a normalized embedding rather than the raw one.

        True for both source arms and False for `glu_v1`. *How* it is normalized
        is the difference between them and is `_gate_input`'s business.
        """
        return self.schema in (SOURCE_FEEDBACK_SCHEMA, SOURCE_FEEDBACK_SCHEMA_V2)

    @property
    def plain_prefix_is_raw(self) -> bool:
        """Whether a position held plain reaches the stack as its raw embedding.

        False for `glu_source_v2`, whose mixin precedes the shared norm. This is
        a property of the arm rather than of the caller, which is why the
        "every position is plain, so skip the fusion" shortcut has to ask.
        """
        return self.schema != SOURCE_FEEDBACK_SCHEMA_V2

    def _gate_input(self, embed: Tensor) -> Tensor:
        if self.schema == SOURCE_FEEDBACK_SCHEMA_V2:
            # The shared learned norm -- Listing 3 line 9's `input_rmsnorm_1`,
            # the same module line 11 applies after the mixin.
            return self.norm(embed)
        if self.schema == SOURCE_FEEDBACK_SCHEMA:
            # Affine-free, as `glu_source_v1` was trained and frozen. Written out
            # rather than routed through `self.norm` because that module carries
            # a learned gain this arm must not apply here.
            return embed * torch.rsqrt(
                embed.pow(2).mean(-1, keepdim=True) + self.norm.eps
            )
        return embed

    def cross(self, hidden: Tensor, embed: Tensor) -> Tensor:
        """`W_U h_(t-1) * sigmoid(W_G g(e_t))`, before any output normalization.

        Split out because `glu_source_v2` needs the un-normalized product: its
        mixin happens between this and the norm.
        """
        return self.up(hidden) * torch.sigmoid(self.gate(self._gate_input(embed)))

    def stack_input(self, hidden: Tensor, embed: Tensor,
                    keep: Tensor | None = None) -> Tensor:
        """The tensor the stack reads, with the plain prefix mixed in.

        `keep` is a `(B, T, 1)` boolean selecting the positions held plain, or
        `None` for a block in which every position is fused. The mixin sits on
        the arm's declared side of the output norm, which is the whole of the
        difference between `glu_source_v2` and the two arms before it.
        """
        product = self.cross(hidden, embed)
        if self.schema == SOURCE_FEEDBACK_SCHEMA_V2:
            mixed = product if keep is None else torch.where(keep, embed, product)
            return self.norm(mixed)
        fused = self.norm(product)
        return fused if keep is None else torch.where(keep, embed, fused)

    def forward(self, hidden: Tensor, embed: Tensor) -> Tensor:
        """The fully fused stack input: `stack_input` with nothing held plain."""
        return self.stack_input(hidden, embed)

    @torch.no_grad()
    def set_shared_gain(self, value: float) -> dict:
        """Overwrite the shared norm's gain with one scalar, in every component.

        A model method rather than a caller reaching into `fuse.norm.weight` for
        the reason cache behaviour lives in this file: the tensor's *role*
        differs by arm -- a fused-output gain in the two earlier ones, the shared
        input norm in `glu_source_v2` -- so what "the gain" means is the model's
        knowledge and not the trainer's.

        Returns the gain's summary before and after, because a calibration that
        cannot be read back from the record is not a recorded intervention. The
        before-summary is four numbers rather than one: by the time this runs the
        gain has taken gradient and gone anisotropic, so its mean alone would not
        say what was overwritten.
        """
        if not math.isfinite(value):
            raise ValueError(f"a shared-norm gain of {value!r} is not a number")
        if value <= 0.0:
            raise ValueError(
                f"a shared-norm gain of {value!r} would delete or invert the "
                "stack's input; the calibration statistic is an RMS and is positive"
            )
        gain = self.norm.weight
        before = {
            "mean": float(gain.mean()), "std": float(gain.std()),
            "min": float(gain.min()), "max": float(gain.max()),
        }
        gain.fill_(float(value))
        return {"schema": self.schema, "before": before, "after": float(value)}


class Block(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.norm_attn = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.norm_ff = RMSNorm(cfg.d_model)
        self.ff = SwiGLU(cfg)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, cache: dict | None = None) -> Tensor:
        x = x + self.attn(self.norm_attn(x), cos, sin, cache)
        return x + self.ff(self.norm_ff(x))


class DrawingLM(nn.Module):
    #: Declared so the two optional tables have a type outside `__init__`: torch's
    #: `Module.__getattr__` erases everything assigned through it, and both of
    #: these are read from outside the class (`share_non_embedding_init`, the
    #: conditioning tests).
    pos: Tensor | None
    classes: Tensor | None
    #: `None` under `feedback_schema="none"`, which is what makes "standard mode
    #: never reads the fusion weights" structural rather than a convention: there
    #: are no fusion weights to read.
    fuse: Fusion | None
    relation: RelationHead | None

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        # Zero-initialised, so a fresh model starts exactly where it would
        # without the table and the flag cannot change an AR arm by accident.
        self.pos = (
            nn.Parameter(torch.zeros(cfg.max_len, cfg.d_model)) if cfg.abs_pos else None
        )
        # Zero-initialised, so the conditional arm and its unconditional control
        # begin from the identical network and the axis is an addition rather
        # than a second draw. `_init` runs after this and skips it: it only
        # touches `nn.Linear` and `nn.Embedding`.
        self.classes = (
            nn.Parameter(torch.zeros(cfg.n_classes, cfg.d_model))
            if cfg.n_classes else None
        )
        blocks = [Block(cfg) for _ in range(cfg.n_layers)]
        self.blocks = nn.ModuleList(blocks)
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight  # tied
        # Registered last, and constructed off the global stream. Two separate
        # reasons, both aimed at one property: a feedback-capable model and its
        # no-feedback twin, built under the same seed, must be bitwise identical
        # in every parameter they share.
        #
        # Last, because `self.apply` walks children in registration order and
        # every `_init` draw consumes the global RNG -- a fusion block registered
        # earlier would shift every draw after it.
        #
        # Forked, because `nn.Linear.__init__` initialises its weight *at
        # construction*, before `apply` re-initialises anything. Those two draws
        # would land ahead of the whole `apply` pass and shift it wholesale. The
        # fork discards them; `apply` gives the block its real values a moment
        # later, at the tail of the stream where nothing follows.
        #
        # This is the same principle as the zero-initialised `pos` and `classes`
        # tables above: an optional component must not change the model that does
        # not use it.
        if cfg.feedback_schema in FUSION_SCHEMAS:
            with torch.random.fork_rng(devices=[]):
                self.fuse = Fusion(cfg, schema=cfg.feedback_schema)
        else:
            self.fuse = None
        # Unlike fusion, relation construction follows the shared residual
        # re-initialisation below.  Those residual draws are deliberately outside
        # ``self.apply``; registering the head before them would move their RNG
        # stream and change ordinary logits in a relation-capable checkpoint.
        self.relation = None
        self.apply(self._init)
        for block in blocks:  # scaled residual init
            nn.init.normal_(block.attn.proj.weight, std=0.02 / math.sqrt(2 * cfg.n_layers))
            nn.init.normal_(block.ff.down.weight, std=0.02 / math.sqrt(2 * cfg.n_layers))
        if cfg.relation_schema == "span_affine_v1":
            with torch.random.fork_rng(devices=[]):
                self.relation = RelationHead(cfg.d_model)
            # Registered last and initialised only after every shared draw.
            # ``Linear``/``Embedding`` are the only random modules in the head;
            # its RMSNorm gain remains its constructor's exact one-vector.
            self.relation.apply(self._init)
        self._cache: tuple | None = None

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def share_non_embedding_init(self, seed: int) -> DrawingLM:
        """Common random numbers: give every codec the *same* layer weights.

        Two arms of a codec ablation differ in vocabulary, so their embedding
        tables differ in shape, so every draw after the embedding differs too --
        and the arms start from unrelated networks. Measured on Tier B, that
        costs ~2 bits/drawing between two runs of a *provably identical*
        encoding, which is larger than any axis under test
        (`docs/tier-b.md`). Pairing on `data_seed` cancels the val set; nothing
        was cancelling the draw.

        Every non-embedding parameter has the same shape in every arm, because
        `d_model`, `n_layers` and `n_heads` are held fixed across the grid. So
        they can be redrawn from one generator, in name order, independently of
        `vocab_size`. This is variance reduction, not a thumb on the scale: each
        arm is still an unbiased draw from the same initialisation distribution,
        and only the *difference* between arms gets quieter.

        Call before `.to(device)`. The embedding is deliberately untouched --
        it is the thing the codecs genuinely differ in.

        **The class table is untouched for the same reason, and it matters more.**
        It is zero so that a conditional arm and its unconditional control are the
        identical function at step 0; filling it from this generator would both
        destroy that and consume draws the control never consumed, so the two arms'
        *layer* weights would stop matching as well. It is skipped by name rather
        than by shape because it has the same rank as a weight matrix.

        **The fusion block is drawn after the loop rather than inside it**, and
        that is what keeps a feedback-capable arm sharing an initialisation with
        the no-feedback checkpoints this project has already trained. The loop
        walks parameters in name order, so a `fuse.*` entry landing in the middle
        of that order would consume draws early and shift every parameter after
        it -- the arms would still train, and they would have quietly stopped
        sharing anything. Appending is invariant to what the block is called.
        """
        generator = torch.Generator().manual_seed(seed)
        scaled = 0.02 / math.sqrt(2 * self.cfg.n_layers)
        with torch.no_grad():
            for name, param in sorted(self.named_parameters()):
                if name in ("embed.weight", "classes") or name.startswith(("fuse.", "relation.")):
                    # `head.weight` is tied to the embedding and is deduped out of
                    # `named_parameters()`; `classes` must stay at zero; `fuse.*`
                    # is drawn below, after every pre-feedback draw is spent.
                    continue
                if param.dim() == 1:
                    param.fill_(1.0)        # RMSNorm gains: no RNG, no drift
                    continue
                std = scaled if name.endswith(("attn.proj.weight", "ff.down.weight")) else 0.02
                param.copy_(torch.normal(0.0, std, size=param.shape, generator=generator))
            if self.fuse is not None:
                for matrix in (self.fuse.gate.weight, self.fuse.up.weight):
                    matrix.copy_(torch.normal(0.0, 0.02, size=matrix.shape,
                                              generator=generator))
                # Not 1.0, which is what the loop above does to every other
                # one-dimensional parameter. This gain is a scale protection
                # rather than a neutral starting point: it matches the
                # embedding's own RMS so the stack's first fused input is the
                # size of the input it already knows how to read.
                self.fuse.norm.weight.fill_(FUSED_NORM_GAIN)
            if self.relation is not None:
                for name, param in self.relation.named_parameters():
                    if name == "norm.weight":
                        param.fill_(1.0)
                    else:
                        param.copy_(torch.normal(0.0, 0.02, size=param.shape,
                                                 generator=generator))
        return self

    def n_params(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)

    def score_relation(self, packed: PackedCandidates) -> RelationScores:
        """Score already-produced normalized boundary states with the opt-in head."""
        if self.relation is None:
            raise ValueError(
                "score_relation() needs relation_schema='span_affine_v1'; an "
                "ordinary checkpoint owns no relation parameters")
        return self.relation(packed)

    def _rope(self, start: int, length: int, device, dtype) -> tuple[Tensor, Tensor]:
        need = start + length
        if self._cache is None or self._cache[0] < need or self._cache[1] != device:
            seq = max(need, self.cfg.max_len)
            self._cache = (seq, device, *_rope_tables(seq, self.cfg.d_head,
                                                      self.cfg.rope_theta, device, dtype))
        cos, sin = self._cache[2], self._cache[3]
        return cos[start:need], sin[start:need]

    def _stack_input(self, idx: Tensor, feedback: Tensor | None,
                     plain: int | Tensor | None) -> Tensor:
        """Token embeddings, with the selected feedback fusion applied where
        `plain` allows.

        `plain` counts the leading positions **of this call** that keep the raw
        embedding, and it is a count rather than a mask over `feedback` because
        the two are not the same operation: fusing a zero state gives
        `RMSNorm(0) = 0`, which deletes the position's input entirely, while "no
        feedback here" has to mean the embedding the model would otherwise have
        seen. `None` means no position is held plain, which is what a one-token
        decode step wants -- the sequence's own position 0 is BOS and is held
        plain by the prefill, not by every step after it.

        A per-row `plain` is supported because F2 draws the plain-prefix length
        per row; a scalar-only version would force the trainer to run rows one at
        a time or to fake it with padding.

        The fused branch is computed over the whole block and then selected,
        rather than sliced first, so a per-row boundary costs nothing extra and
        the arithmetic does not depend on where the boundary happens to fall.
        `Fusion.stack_input` owns *where* in the arithmetic that selection lands:
        under `glu_source_v2` it precedes the shared norm, so the plain prefix is
        `N(e)` rather than `e`, and this method only says which positions it
        covers.
        """
        e = self.embed(idx)
        if feedback is None:
            return e
        if self.fuse is None:
            raise ValueError(
                "carried state was given to a feedback_schema='none' model. "
                "Switching feedback on for an ordinary checkpoint decodes it "
                "through a recurrent input distribution it was never trained on, "
                "and the output looks entirely plausible"
            )
        width = idx.shape[1]
        if feedback.shape != (idx.shape[0], width, self.cfg.d_model):
            raise ValueError(
                f"feedback is {tuple(feedback.shape)}, expected "
                f"{(idx.shape[0], width, self.cfg.d_model)}: one carried state per "
                "row per position, aligned so feedback[:, t] is fused into t"
            )
        if (isinstance(plain, int) and plain >= width
                and self.fuse.plain_prefix_is_raw):
            # Every position plain, in an arm that hands a plain position its raw
            # embedding: identical to the standard path, and returning `e`
            # directly keeps it *exactly* identical rather than nearly.
            #
            # `glu_source_v2` cannot take this exit. Its mixin precedes the
            # shared norm, so a fully plain block is `N(e)` -- still a fused
            # pass, and still different from the standard forward
            # (`docs/directions.md` §3.1).
            return e
        keep = None
        if plain is not None:
            counts = (plain if isinstance(plain, Tensor)
                      else torch.full((idx.shape[0],), plain, device=idx.device))
            keep = (torch.arange(width, device=idx.device)[None, :]
                    < counts[:, None])[..., None]
        return self.fuse.stack_input(feedback, e, keep)

    def _embed(self, idx: Tensor, start: int = 0, classes: Tensor | None = None,
               feedback: Tensor | None = None,
               plain: int | Tensor | None = None) -> Tensor:
        # Fusion first, then the two optional additive tables: `W_G` gates on the
        # token embedding alone (`docs/directions.md` §3.1), and gating on a
        # class-shifted or position-shifted embedding is a different model.
        x = self._stack_input(idx, feedback, plain)
        if self.pos is not None:
            x = x + self.pos[start : start + idx.shape[1]]
        if self.classes is not None:
            if classes is None:
                raise ValueError(
                    "this model is class-conditional and was called without classes; "
                    "an unconditioned forward would silently score every drawing "
                    "under a zero conditioning vector, which is not any class"
                )
            # Added at *every* position rather than only the first: the signal is
            # a property of the whole drawing, and a prefix-only vector has to
            # survive four layers of attention to reach the tail, which is where
            # a drawing's category is least redundant.
            x = x + self.classes[classes].unsqueeze(1)
        elif classes is not None:
            raise ValueError(
                "classes were given to an unconditional model; it would ignore "
                "them and the run would be filed as conditional"
            )
        return x

    def forward(self, idx: Tensor, classes: Tensor | None = None, *,
                feedback: Tensor | None = None, plain: int | Tensor = 0,
                return_state: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        """Logits for `idx`, optionally fusing carried state into each position.

        `feedback` is `(B, T, D)` carried **normalized top-layer** states, already
        aligned so that `feedback[:, t]` is the state fused into position `t`.
        The caller does the shifting because only the caller knows where the
        states came from: F2's parallel pass shifts the previous pass's states
        right by one, and a decode step carries the state that produced the token
        it is about to feed back.

        `plain` holds the first positions at their raw embeddings -- BOS and F2's
        sampled prefix. **Position 0 is always plain whatever `plain` says**: BOS
        has no predecessor, so there is no state it could carry, and a feedback
        row 0 that was read would make the whole sequence depend on whatever the
        caller happened to leave there (`docs/directions.md` §7 invariant 4).

        `return_state=True` also returns the normalized states this call produced,
        which is the tensor the tied head consumes and therefore exactly the
        tensor the next pass or the next decode step must carry.
        """
        held = max(plain, 1) if isinstance(plain, int) else plain.clamp(min=1)
        x = self._embed(idx, classes=classes, feedback=feedback, plain=held)
        cos, sin = self._rope(0, idx.shape[1], idx.device, x.dtype)
        for block in self.blocks:
            x = block(x, cos, sin)
        state = self.norm(x)
        logits = self.head(state)
        return (logits, state) if return_state else logits

    def _check_mode(self, mode: str) -> None:
        if mode not in RUNTIME_MODES:
            raise ValueError(
                f"unknown decode mode {mode!r}; expected one of {list(RUNTIME_MODES)}"
            )
        if mode != "standard" and self.fuse is None:
            raise ValueError(
                f"mode={mode!r} needs a feedback-capable checkpoint and this one is "
                f"feedback_schema={self.cfg.feedback_schema!r}. Switching feedback "
                "on for an ordinary checkpoint decodes it through a recurrent input "
                "distribution it was never trained on"
            )

    @torch.no_grad()
    def teacher_forced(
        self,
        prompt: Tensor,
        continuation: Tensor,
        bos: int = 1,
        classes: Tensor | None = None,
        device: str | torch.device = "cpu",
        mode: str = "standard",
    ) -> Tensor:
        """`generate`'s scoring twin: the same decode, with the tokens supplied.

        Returns `(n, C, vocab)` logits, where row `i`, position `t` is the
        distribution over `continuation[i, t]` given BOS, the whole prompt and
        `continuation[i, :t]`. Every target is therefore scored under **its own**
        gold history, which is the property Direction 2's four-way blocks rest on.

        **Why this cannot be a batched full-forward pass.** In soft mode the input
        at position `t` is fused with the state that produced position `t-1`, and
        that state was itself computed from a fused input. The recurrence is
        genuine: there is no parallel form, and `dm.eval.context.score_spans`
        -- which scores a whole sequence in one shot -- cannot express it at all.
        Getting `Delta_soft` out of the batched scorer is the trap `PLAN.md`
        names; this method is the alternative it names.

        **Why it lives here and not in the evaluator.** It needs the KV cache, and
        cache behaviour stays local to the transformer (`docs/directions.md` §3.4).
        An evaluator reaching into `LayerCache` would be reaching into private
        state through a seam that only looked like one.

        The three modes mean exactly what they mean in `generate`: `standard`
        reads no fusion weight and no latent, `soft` prefills plainly and fuses
        every scored position after the first, and `fused` discards the standard
        prefill's keys and values and rebuilds the cache from a fused prompt. The
        first scored symbol therefore has identical logits in `standard` and
        `soft` -- the structural zero §2.2 requires be reported rather than
        averaged into a total.
        """
        if not self.cfg.causal:
            raise ValueError(
                "teacher_forced() needs a causal model; a bidirectional one has no "
                "next-token distribution to score against"
            )
        self._check_mode(mode)
        self.eval()
        prompt, continuation = prompt.to(device), continuation.to(device)
        rows, given = prompt.shape
        length = continuation.shape[1]
        if continuation.shape[0] != rows:
            raise ValueError(
                f"continuation has {continuation.shape[0]} rows, prompt has {rows}"
            )
        if length < 1:
            raise ValueError("there is nothing to score: the continuation is empty")
        caches = [
            # BOS, the prompt, and every continuation symbol except the last --
            # the last one is scored, never fed.
            LayerCache(rows, self.cfg.n_heads, given + length, self.cfg.d_head,
                       device, self.embed.weight.dtype)
            for _ in self.blocks
        ]
        cur = torch.cat(
            [torch.full((rows, 1), bos, dtype=torch.long, device=device), prompt],
            dim=1,
        )
        prefill_feedback: Tensor | None = None
        if mode == "fused":
            _, prior = self.forward(cur, classes=classes, return_state=True)
            prefill_feedback = torch.zeros_like(prior)
            prefill_feedback[:, 1:] = prior[:, :-1]
        latent: Tensor | None = None
        scored: list[Tensor] = []
        pos = 0
        for step in range(length):
            if step == 0:
                feedback, plain = prefill_feedback, 1
            else:
                feedback, plain = (None if mode == "standard" else latent), None
            x = self._embed(cur, start=pos, classes=classes, feedback=feedback,
                            plain=plain)
            cos, sin = self._rope(pos, x.shape[1], device, x.dtype)
            for block, cache in zip(self.blocks, caches):
                x = block(x, cos, sin, cache)
            pos += x.shape[1]
            state = self.norm(x)
            latent = state[:, -1:]
            scored.append(self.head(state)[:, -1:])
            cur = continuation[:, step : step + 1]
        return torch.cat(scored, dim=1)

    @torch.no_grad()
    def generate_relation(
        self,
        prompts: Sequence[bytes],
        max_new: int,
        *,
        mode: str = "standard",
        literal_policy: str | None = None,
        observer: DecodeObserver | None = None,
        observation_level: str = "summary",
        bos: int = BOS,
        device: str | torch.device = "cpu",
        monitor: HaltProtocol | None = None,
        oracle_actions: Mapping[tuple[int, int], CopyActionKey] | None = None,
        on_logits: Callable[[int, Tensor], None] | None = None,
        variates: Tensor | None = None,
    ) -> Tensor:
        """Byte-only R4 decode through standard, predicted or oracle COPY paths.

        This is intentionally separate from :meth:`generate`: standard delegates
        directly to that historical method, while the relation modes retain
        normalized top-layer boundary states and advance each copied byte through
        the normal causal cache one position at a time.  No target bytes or
        corpus object is accepted here; oracle inputs are target-free action keys.
        """
        if mode not in ("standard", "predicted_copy", "oracle_copy"):
            raise ValueError("relation decode mode must be standard, predicted_copy, or oracle_copy")
        policy = resolve_policy(mode, literal_policy)
        if observation_level not in ("summary", "actions", "scores"):
            raise DecodeRequestError("unknown observation level")
        if observer is not None and (not callable(observer) or policy == LEGACY_POLICY):
            raise DecodeRequestError("observer requires common policy and a callable")
        if not prompts:
            raise ValueError("relation decode needs at least one immutable byte prompt")
        if type(max_new) is not int or max_new < 0:
            raise ValueError("max_new must be non-negative")
        if bos != BOS or self.cfg.vocab_size != BYTE_VOCAB_SIZE:
            raise ValueError("relation decode is defined only for the standard ByteCodec vocabulary")
        if not self.cfg.causal:
            raise ValueError("relation decode needs a causal model")
        if self.cfg.feedback_schema != "none":
            raise ValueError("relation decode refuses feedback-enabled models")
        if any(not isinstance(prompt, bytes) for prompt in prompts):
            raise ValueError("relation prompts must be immutable bytes")
        lengths = {len(prompt) for prompt in prompts}
        if len(lengths) != 1:
            raise ValueError("relation cache sessions require equal prompt byte lengths")
        given = lengths.pop()
        if given + max_new > self.cfg.max_len:
            raise ValueError("relation prompt plus requested output exceeds model capacity")
        rows = len(prompts)
        prompt_tokens = torch.tensor(
            [[N_SPECIAL + byte for byte in prompt] for prompt in prompts],
            dtype=torch.long, device=device,
        )
        if policy == LEGACY_POLICY:
            # This direct delegation protects all legacy sampling/callback and
            # variate arithmetic.  Relation-aware code never runs on this path.
            return self.generate(rows, max_new, bos=bos, device=device, monitor=monitor,
                                 prompt=prompt_tokens, variates=variates,
                                 on_logits=on_logits)
        if mode == "predicted_copy" and self.relation is None:
            raise ValueError("predicted_copy needs relation_schema='span_affine_v1'")
        if mode != "oracle_copy" and oracle_actions is not None:
            raise ValueError(f"{mode} refuses oracle action data")
        if variates is not None and tuple(variates.shape) != (rows, given + max_new):
            raise ValueError("relation variates need one position-keyed uniform per output row")
        if variates is not None and (not variates.is_floating_point() or
                                     not bool(torch.isfinite(variates).all()) or
                                     bool(((variates < 0) | (variates >= 1)).any())):
            raise ValueError("relation variates must be finite uniforms in [0, 1)")
        if variates is not None:
            sampler_dtype = (torch.get_autocast_dtype(torch.device(device).type)
                             if torch.is_autocast_enabled(torch.device(device).type)
                             else self.embed.weight.dtype)
            converted = variates.to(dtype=sampler_dtype)
            if bool(((converted < 0) | (converted >= 1)).any()):
                raise ValueError("relation variates leave [0, 1) in sampler dtype")
        stop = FlatOutputStop(rows, given, max_new, monitor)
        # Runtime candidate selection must fail loudly on malformed prompt bytes.
        prompt_indexes = [prefix_index(prompt) for prompt in prompts]
        if any(Op.HALT in ops and (ops[-1] is not Op.HALT or ops.count(Op.HALT) != 1)
               for _, _, ops in prompt_indexes):
            raise ValueError("relation prompt has bytes after HALT; its dead tail is not decodable")
        ledger = DecodeLedger(rows, given, variates is not None, observer, observation_level)
        ledger.publish(RequestStarted(mode, policy, STOPPING_POLICY,
                                      "inverse_cdf" if variates is not None else "multinomial",
                                      rows, given, max_new, self.cfg.max_len, observation_level))
        for row, (_, _, ops) in enumerate(prompt_indexes):
            stop.prompt_terminal[row] = bool(ops) and ops[-1] is Op.HALT
        if max_new == 0:
            for row in range(rows):
                stop.causes[row] = "zero_horizon"
                ledger.stopped(stop, row)
            ledger.complete(given)
            return prompt_tokens
        for offset in range(given):
            stop.step(prompt_tokens[:, offset:offset + 1].t().cpu().numpy())
        for row in range(rows):
            if stop.done[row]:
                ledger.stopped(stop, row)
        if bool(stop.done.all()):
            ledger.complete(given)
            return prompt_tokens

        def checked_logits(offset: int, raw: Tensor) -> None:
            if not bool(torch.isfinite(raw).all()):
                raise ValueError("relation byte head produced nonfinite raw scores")
            if on_logits is not None:
                on_logits(offset, raw)
                if not bool(torch.isfinite(raw).all()):
                    raise ValueError("relation byte head produced nonfinite raw scores")

        if mode == "standard":
            # Prime once above; generate must still initialize its local done
            # mask. The adapter replays only its status, never the caller monitor.
            previous_origins = ("literal",) * rows

            def standard_logits(offset: int, raw: Tensor) -> None:
                ledger.forwarded(offset == given, previous_origins, stop.done)
                if offset == given:
                    ledger.kv_bytes = (2 * rows * (given + max_new) * self.cfg.d_model
                                       * len(self.blocks) * self.embed.weight.element_size())
                checked_logits(offset, raw)

            class PrimedStop:
                stride = 1

                def __init__(self):
                    self.replayed = 0

                def step(self, chunk):
                    if self.replayed < given:
                        self.replayed += 1
                        return stop.done.copy()
                    nonlocal previous_origins
                    live = ~stop.done.copy()
                    ledger.sampled(list(range(rows)), live)
                    previous_origins = tuple("literal" if active else "pad" for active in live)
                    for row in range(rows):
                        ledger.add(row, "generated_bytes", int(live[row]))
                    status = stop.step(chunk)
                    for row in range(rows):
                        if status[row]:
                            ledger.stopped(stop, row)
                    return status

            result = self.generate(rows, max_new, bos=bos, device=device, monitor=PrimedStop(),
                                   prompt=prompt_tokens, variates=variates,
                                   on_logits=standard_logits, forbid=(PAD_TOKEN, BOS))
            ledger.complete(result.shape[1])
            return result
        self.eval()
        width = given + max_new
        caches = [LayerCache(rows, self.cfg.n_heads, width, self.cfg.d_head,
                             device, self.embed.weight.dtype) for _ in self.blocks]
        ledger.kv_bytes = sum(t.numel() * t.element_size()
                              for cache in caches for t in (cache.k, cache.v))
        out = torch.full((rows, width), PAD_TOKEN, dtype=torch.long, device=device)
        if given:
            out[:, :given] = prompt_tokens
        done = torch.from_numpy(stop.done.copy()).to(device)
        boundary_state = RelationBoundaryCache(
            rows, capacity=width + 1, d_model=self.cfg.d_model,
            device=device, dtype=self.embed.weight.dtype)
        ledger.boundary_bytes = boundary_state.storage_bytes
        cur = torch.cat([torch.full((rows, 1), bos, dtype=torch.long, device=device),
                         prompt_tokens], dim=1)
        x = self._embed(cur, start=0)
        cos, sin = self._rope(0, x.shape[1], device, x.dtype)
        for block, cache in zip(self.blocks, caches):
            x = block(x, cos, sin, cache)
        state = self.norm(x)
        raw = self.head(state)[:, -1]
        ledger.forwarded(True, ("literal",) * rows, stop.done)
        for row, (marks, _, _) in enumerate(prompt_indexes):
            for boundary in marks:
                boundary_state.put(row, boundary, state[row, boundary])
                ledger.add(row, "boundary_slots_used")
        queues = RowByteQueues(rows)
        generated = [bytearray(prompt) for prompt in prompts]
        pos = given + 1

        for step in range(given, width):
            checked_logits(step, raw)
            eligible: list[int] = []
            query_states: list[Tensor] = []
            starts: list[Tensor] = []
            stops: list[Tensor] = []
            bins: list[int] = []
            start_offsets: list[int] = []
            stop_offsets: list[int] = []
            splits = [0]
            prefixes: list[bytes] = []
            for row, buffer in enumerate(generated):
                if bool(done[row]) or queues.pending(row):
                    continue
                prefix = bytes(buffer)
                progress = prefix_progress(prefix)
                if not progress.complete:
                    # A decoder never stands at an operand boundary. It emits
                    # normally until the shared parser reaches a full L0
                    # instruction, then the next iteration may query COPY.
                    continue
                # `candidate_spans` shares the executor parser and therefore
                # proves both canonicality and target-free eligibility here.
                spans = candidate_spans(prefix, len(prefix))
                eligible.append(row)
                prefixes.append(prefix)
                query_states.append(boundary_state.get(row, len(prefix)))
                for start, end in spans:
                    starts.append(boundary_state.get(row, start))
                    stops.append(boundary_state.get(row, end))
                    bins.append(length_bin(end - start))
                    start_offsets.append(start)
                    stop_offsets.append(end)
                splits.append(len(starts))
            if eligible and mode == "predicted_copy":
                d_model = self.cfg.d_model
                empty = state.new_empty((0, d_model))
                packed = PackedCandidates(
                    query_states=torch.stack(query_states),
                    start_states=torch.stack(starts) if starts else empty,
                    stop_states=torch.stack(stops) if stops else empty,
                    length_bins=torch.tensor(bins, dtype=torch.long, device=device),
                    source_start=torch.tensor(start_offsets, dtype=torch.long, device=device),
                    source_stop=torch.tensor(stop_offsets, dtype=torch.long, device=device),
                    row_splits=torch.tensor(splits, dtype=torch.long, device=device),
                )
                scores = self.score_relation(packed)
                ledger.calls["head_batches"] += 1
                for row, begin, end in zip(eligible, splits[:-1], splits[1:], strict=True):
                    ledger.add(row, "head_queries")
                    ledger.add(row, "candidate_entries", end - begin)
                for query, (row, prefix) in enumerate(zip(eligible, prefixes, strict=True)):
                    from .relation import choose_action, score_snapshot
                    decision = choose_action(scores, query, prefix,
                                             max_len=min(self.cfg.max_len, width))
                    ledger.add(row, "executor_attempts", decision.expansions)
                    ledger.add(row, "predicted_search_attempts", decision.expansions)
                    if not decision.emit:
                        if decision.execution is None:  # defensive: R3 owns this invariant.
                            raise RuntimeError("COPY decision lacks its checked execution")
                        queues.enqueue(row, decision.execution.appended)
                        ledger.add(row, "copy_admissions")
                        ledger.add(row, "copy_admitted_bytes", len(decision.execution.appended))
                    if observer is not None and observation_level != "summary":
                        action = decision.action
                        key = (None if action is None else
                               (action.boundary, action.source_start, action.source_stop,
                                action.step.d4.code, action.step.dx, action.step.dy,
                                action.total_count))
                        ledger.publish(ActionDecision(
                            row, step, "predicted", splits[query + 1] - splits[query], key,
                            decision.score, decision.expansions, decision.faults,
                            decision.exit_reason, queues.pending(row)))
                        if observation_level == "scores":
                            ledger.publish(score_snapshot(scores, query, row, step))
            elif eligible:  # oracle has no neural-head dependency or scoring path.
                for row, prefix in zip(eligible, prefixes, strict=True):
                    action = None if oracle_actions is None else oracle_actions.get((row, len(prefix)))
                    if action is not None:
                        ledger.add(row, "executor_attempts")
                        ledger.add(row, "oracle_executor_calls")
                        execution = execute_copy(prefix, action, policy=ORACLE_SUPPORT,
                                                 max_len=min(self.cfg.max_len, width))
                        queues.enqueue(row, execution.appended)
                        ledger.add(row, "copy_admissions")
                        ledger.add(row, "copy_admitted_bytes", len(execution.appended))
                        if observer is not None and observation_level != "summary":
                            ledger.publish(ActionDecision(
                                row, step, "oracle", 0,
                                (action.boundary, action.source_start, action.source_stop,
                                 action.step.d4.code, action.step.dx, action.step.dy,
                                 action.total_count), None, 1, (), "oracle_executed",
                                len(execution.appended)))
            queued = queues.pop_position()
            token = torch.empty((rows, 1), dtype=torch.long, device=device)
            emit_rows: list[int] = []
            origins = ["pad"] * rows
            for row, byte in enumerate(queued if queued is not None else (None,) * rows):
                if bool(done[row]):
                    token[row, 0] = PAD_TOKEN
                elif byte is None:
                    emit_rows.append(row)
                    origins[row] = "literal"
                else:
                    token[row, 0] = N_SPECIAL + byte
                    origins[row] = "copy"
                    ledger.add(row, "copy_delivered_bytes")
            if emit_rows:
                emit = torch.tensor(emit_rows, dtype=torch.long, device=device)
                # R4's byte path cannot place PAD/BOS into an active causal
                # context.  Mask before the same inverse-CDF arithmetic used by
                # normal generation, so a supplied position-keyed variate stays
                # paired over the *legal* byte distribution.
                logits = raw[emit].clone()
                logits[:, :N_SPECIAL] = float("-inf")
                probs = F.softmax(logits, dim=-1)
                if variates is None:
                    token[emit, 0] = torch.multinomial(probs, 1).squeeze(1)
                else:
                    cumulative = probs.cumsum(dim=-1).contiguous()
                    positive = probs > 0
                    last = positive.shape[-1] - 1 - positive.flip(-1).to(torch.int64).argmax(-1)
                    positions = torch.arange(probs.shape[-1], device=device)[None, :]
                    cumulative = torch.where(positions >= last[:, None], torch.ones_like(cumulative), cumulative)
                    draw = variates[emit, step, None].to(device=device, dtype=cumulative.dtype)
                    token[emit, 0] = torch.searchsorted(cumulative, draw, right=True).squeeze(1)
            ledger.sampled(emit_rows, ~stop.done)
            out[:, step:step + 1] = token
            for row, symbol in enumerate(token[:, 0].tolist()):
                if not bool(done[row]):
                    generated[row].append(symbol - N_SPECIAL)
                    ledger.add(row, "generated_bytes")
            done = torch.from_numpy(stop.step(token.t().cpu().numpy())).to(device)
            for row in range(rows):
                if bool(done[row]):
                    discarded = queues.discard(row)
                    ledger.add(row, "copy_discarded_bytes", discarded)
                    ledger.stopped(stop, row, discarded)
                ledger.rows[row]["copy_pending_bytes"] = queues.pending(row)
            if step + 1 == width or bool(done.all()):
                ledger.complete(step + 1)
                return out[:, :step + 1]
            # Feed exactly the preceding byte, including copied bytes, before
            # the next decision.  No multi-byte fill can enter this loop.
            x = self._embed(token, start=pos)
            cos, sin = self._rope(pos, 1, device, x.dtype)
            for block, cache in zip(self.blocks, caches):
                x = block(x, cos, sin, cache)
            pos += 1
            state = self.norm(x)
            raw = self.head(state)[:, -1]
            ledger.forwarded(False, tuple(origins), stop.done)
            for row, buffer in enumerate(generated):
                if bool(done[row]):
                    continue
                progress = prefix_progress(bytes(buffer))
                if progress.complete:
                    boundary_state.put(row, len(buffer), state[row, 0])
                    ledger.add(row, "boundary_slots_used")
        return out

    @torch.no_grad()
    def generate(
        self,
        n: int,
        max_new: int,
        bos: int = 1,
        temperature: float = 1.0,
        top_k: int | None = None,
        device: str | torch.device = "cpu",
        monitor: HaltProtocol | None = None,
        prompt: Tensor | None = None,
        prime_monitor: bool = True,
        forbid: tuple[int, ...] = (),
        classes: Tensor | None = None,
        variates: Tensor | None = None,
        on_logits: Callable[[int, Tensor], None] | None = None,
        mode: str = "standard",
    ) -> Tensor:
        """Incremental decode with a KV cache, optionally continuing a prefix.

        Without the cache this is quadratic in sequence length, which the byte
        arm survives and the bit arm does not -- its sequences are 8x longer, so
        sampling evaluation dominates the whole run.

        `monitor` decides when a row has terminated. It is the codec's parse
        state rather than a symbol match, because in an untyped alphabet the
        HALT opcode and the operand value zero are the same symbol and only the
        instruction boundary separates them; see `dm.isa.codec.HaltMonitor`.
        The model stays free of any ISA dependency -- all it needs is something
        with a `stride` and a `step`.

        `prompt` is `(n, P)` symbols, *without* BOS, prepended to every row and
        returned verbatim in the output. It is what makes paired comparison
        possible at all for an AR model, which reconstructs nothing: condition
        on the first k% of a held-out program, let the model finish, and compare
        the geometry against the truth (`docs/scale-separation.md`). It is fed
        through the monitor as well as the model, so the parse state starts
        where the prompt left it rather than at an instruction boundary --
        without that, an operand `0x00` at the end of a prompt reads as HALT.
        **`P` must be a whole number of bytecode bytes** (a multiple of
        `monitor.stride`) or the monitor desynchronises; it is checked.

        `prime_monitor=False` says the prompt is *conditioning* rather than
        bytecode, and stops it being fed through the parse state. Claim 3's
        stroke decoder is prompted with a six-byte stroke summary
        (`dm/models/planner.py`), which is not a program: primed, a summary
        whose first field is 0 reads as HALT and the row terminates before it
        has emitted anything.

        Prefill and decode share one path: BOS and the prompt go through as a
        single block, which is one attention call rather than P, and
        `Attention.forward` already distinguishes the two cases by comparing the
        query length against the cache depth.

        Rows that have stopped keep emitting PAD, which every codec drops on
        decode, so the returned block is rectangular without being misread.

        A monitor that also answers `allowed()` (`SupportProtocol`,
        `dm.isa.state.StateMonitor`) turns this into a constrained decode: the
        per-row legal set is applied to the logits **before** `forbid` and
        `top_k`, so truncation sees the surviving distribution rather than
        competing with the mask for slots.

        `variates` is an optional `(n, given + max_new)` block of pre-generated
        uniforms, consumed one per row per step by inverse CDF instead of by
        `torch.multinomial`. It is what makes two decoder cells *paired*: the
        mask changes the distribution, so it also changes how a sampler consumes
        its RNG stream, and two cells seeded identically would still diverge
        after the first masked position. A variate is consumed at every step for
        every row, including rows that have stopped, so the two cells stay
        aligned by absolute step index even though one of them stops earlier.

        `on_logits(step, logits)` sees the **raw** logits -- before temperature,
        before the mask, before `forbid` and before `top_k`. That is the only
        distribution under which the illegal mass `q` of
        `docs/state-freeze.md` §5 is defined, and computing it anywhere else
        would measure the sampler instead of the model.

        `mode` selects Direction 3's decode-time behaviour, and is an argument
        rather than checkpoint state so that two cells can differ *only* in it:

        - `standard` never reads the fusion weights or any latent. It is the
          pre-Direction-3 decode, unchanged, whatever schema the checkpoint has.
        - `soft` prefills the prompt in standard mode and then fuses each
          generated token's embedding with the state that produced it. Prompt KV
          and the first carried state come from that one standard prefill, so the
          first generated token's distribution is *identical* to `standard`'s by
          construction -- a structural zero the evaluator reports rather than
          averages away.
        - `fused` runs a standard prefill, throws its keys and values away, and
          rebuilds the cache from a second prefill whose prompt positions are
          themselves fused. Diagnostic, because it doubles prompt compute.
          Reusing the standard cache here would attend from fused queries to keys
          computed off plain embeddings -- a decode against a prefix the model
          never saw, which produces entirely plausible output.

        The latent is request-local and row-aligned: it lives in this call, one
        row of it per row of the batch, and it is gone when the call returns.
        Stopped rows keep their own state and cannot reach an active row.
        """
        self._check_mode(mode)
        if not self.cfg.causal:
            # A bidirectional model has no next-token distribution to sample
            # from -- every position was trained conditioned on the ones after
            # it. Sampling anyway returns plausible-looking rows, which is the
            # worst possible failure mode for a generation metric.
            raise ValueError(
                "generate() needs a causal model; this one is bidirectional. "
                "Claim 3's composition level is sampled by iterative unmasking "
                "(dm/models/planner.py), not by autoregressive decoding."
            )
        self.eval()
        given = 0 if prompt is None else prompt.shape[1]
        if prompt is not None:
            if prompt.shape[0] != n:
                raise ValueError(f"prompt has {prompt.shape[0]} rows, expected {n}")
            if monitor is not None and prime_monitor and given % monitor.stride:
                raise ValueError(
                    f"prompt of {given} symbols is not a whole number of bytecode bytes "
                    f"at stride {monitor.stride}; the halt monitor would desynchronise"
                )
        width = given + max_new
        if variates is not None and tuple(variates.shape) != (n, width):
            raise ValueError(
                f"variates are {tuple(variates.shape)}, expected {(n, width)}: one "
                "uniform per row per step, so that two runs of this decode consume "
                "the identical stream whatever the mask does to the distribution"
            )
        if variates is not None:
            if not variates.is_floating_point():
                raise ValueError("variates must be floating-point uniforms in [0, 1)")
            if not bool(torch.isfinite(variates).all()):
                raise ValueError("variates must all be finite uniforms in [0, 1)")
            if bool(((variates < 0) | (variates >= 1)).any()):
                raise ValueError("variates must all lie in the half-open interval [0, 1)")
        caches = [
            # One slot per position the model actually consumes: BOS, the
            # prompt, and every generated symbol that is fed back. Sizing this
            # from `max_new` alone silently overruns a prompted decode, and the
            # buffer is preallocated precisely so nothing reallocates mid-run.
            LayerCache(n, self.cfg.n_heads, width, self.cfg.d_head,
                       device, self.embed.weight.dtype)
            for _ in self.blocks
        ]
        out = torch.full((n, width), PAD_TOKEN, dtype=torch.long, device=device)
        cur = torch.full((n, 1), bos, dtype=torch.long, device=device)
        if prompt is not None:
            prompt = prompt.to(device)
            out[:, :given] = prompt
            cur = torch.cat([cur, prompt], dim=1)
        done = torch.zeros(n, dtype=torch.bool, device=device)
        stride = monitor.stride if monitor is not None else 0
        if monitor is not None and given and prime_monitor:
            for end in range(stride, given + 1, stride):
                done = torch.from_numpy(
                    monitor.step(out[:, end - stride : end].t().cpu().numpy())
                ).to(device)

        legality = getattr(monitor, "allowed", None) if monitor is not None else None
        # The prefill's feedback. Empty for `standard` and `soft`, which prefill
        # plainly; for `fused` it is the standard prefill's own states shifted
        # right by one, so every prompt position after BOS is fused with the
        # state of the position before it. That first pass is run *without* a
        # cache and discarded: its keys and values describe plain inputs, and the
        # fused prefill below overwrites the same positions from scratch.
        prefill_feedback: Tensor | None = None
        if mode == "fused":
            _, prior = self.forward(cur, classes=classes, return_state=True)
            prefill_feedback = torch.zeros_like(prior)
            prefill_feedback[:, 1:] = prior[:, :-1]
        #: Request-local, one row per batch row, cleared when this call returns.
        latent: Tensor | None = None
        pos = 0
        for step in range(given, width):
            if step == given:
                feedback, plain = prefill_feedback, 1
            else:
                # Every generated token fuses with the state that produced it.
                # `plain=None` rather than 0: this call's position 0 *is* a
                # generated token, and the sequence's BOS was held plain by the
                # prefill above.
                feedback, plain = (None if mode == "standard" else latent), None
            x = self._embed(cur, start=pos, classes=classes, feedback=feedback,
                            plain=plain)
            cos, sin = self._rope(pos, x.shape[1], device, x.dtype)
            for block, cache in zip(self.blocks, caches):
                x = block(x, cos, sin, cache)
            pos += x.shape[1]
            state = self.norm(x)
            latent = state[:, -1:]
            raw = self.head(state)[:, -1]
            if on_logits is not None:
                on_logits(step, raw)
            logits = raw / max(temperature, 1e-6)
            if legality is not None:
                # Per row, immediately before sampling, and before `top_k` --
                # otherwise truncation spends slots on symbols the mask is about
                # to remove, and the two cells differ in more than the mask.
                allowed = torch.from_numpy(legality()).to(device)
                logits = logits.masked_fill(~allowed, float("-inf"))
            # Ids the output cannot contain, masked before top-k so they cannot
            # occupy a slot in it. Empty for every AR arm -- their alphabet is
            # the codec's and every symbol in it is legal -- and used by claim
            # 3's composition level, whose grid has three ids kept only so
            # token numbering matches the codec's. Masking here rather than
            # trusting training to have learned it keeps a sampled grid
            # decodable by construction, which is the rule the denoiser's
            # sampler already follows.
            if forbid:
                logits[..., list(forbid)] = float("-inf")
            if top_k:
                kth = logits.topk(min(top_k, logits.shape[-1]), dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            if variates is None:
                nxt = torch.multinomial(probs, 1)
            else:
                # Inverse CDF of the same distribution `multinomial` would draw
                # from, but reading a variate this caller pre-generated: one per
                # row per step, so a mask cannot desynchronise two cells.
                cumulative = probs.cumsum(dim=-1).contiguous()
                # Float32 cumulative sums can finish just below one. If the
                # final vocabulary slot is masked, clamping a search result past
                # that total to ``vocab - 1`` would emit a zero-probability,
                # masked symbol. Put the exact endpoint on the last positive
                # bucket of each row instead.
                positive = probs > 0
                if not bool(positive.any(dim=-1).all()):
                    raise ValueError(
                        "inverse-CDF sampling received a row with no positive "
                        "probability after legality/forbid/top_k"
                    )
                if not bool(positive.all(dim=-1).all()):
                    last = positive.shape[-1] - 1 - positive.flip(-1).to(torch.int64).argmax(-1)
                else:
                    last = torch.full(
                        (probs.shape[0],), probs.shape[-1] - 1,
                        dtype=torch.long, device=probs.device,
                    )
                positions = torch.arange(
                    probs.shape[-1], device=probs.device
                )[None, :]
                # The tail after the last positive bucket is a run of masked
                # zeros. Set the whole tail to one, not only the last positive
                # entry: otherwise a float32 roundoff can leave a later zero
                # bucket below the repaired endpoint and make the CDF
                # non-monotone.
                cumulative = torch.where(
                    positions >= last[:, None],
                    torch.ones_like(cumulative), cumulative,
                )
                draw = variates[:, step, None].to(
                    device=device, dtype=cumulative.dtype
                ).contiguous()
                nxt = torch.searchsorted(
                    cumulative, draw, right=True
                ).clamp_(max=probs.shape[-1] - 1)
            if monitor is not None:
                nxt = nxt.masked_fill(done[:, None], PAD_TOKEN)
            out[:, step : step + 1] = nxt
            cur = nxt

            end = step + 1
            if monitor is None or end % stride:
                continue
            # One transfer per bytecode byte rather than per symbol: the monitor
            # cannot decide inside a byte anyway, and the sync is what a decode
            # step actually costs.
            chunk = out[:, end - stride : end].t().cpu().numpy()
            done = torch.from_numpy(monitor.step(chunk)).to(device)
            if bool(done.all()):
                return out[:, :end]

        return out
