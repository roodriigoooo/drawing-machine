"""Training loop for the autoregressive arm.

One run = one (representation, architecture, dataset) cell of the sweep. Every
run writes a JSON record so the sweep is assembled from files rather than from
whatever was on screen at the time.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Protocol

import torch
import torch.nn.functional as F

from .data import augment, composed, quickdraw, synthetic, tabler
from .data import feedback as feedback_data
from .data.dataset import PAD, ProgramDataset, loader
from .data.fingerprint import fingerprint
from .eval.feedback_contract import (
    DEFAULT_GAIN_CALIBRATION,
    DEFAULT_PASS_SCHEDULE,
    GAIN_CALIBRATIONS,
    PASS_SCHEDULES,
    seed_for,
)
from .eval.feedback_contract import (
    feedback_phase_transition as _feedback_phase_transition,
)
from .eval.feedback_contract import gain_calibration as _gain_calibration
from .eval.feedback_contract import pass_schedule as _pass_schedule
from .eval.metrics import bits_per_drawing, sample_quality
from .isa.codec import CODECS
from .isa.spec import Tier
from .models.transformer import (
    SUPPORTED_FEEDBACK_SCHEMAS,
    Config,
    DrawingLM,
)
from .train_feedback import (
    Accounting,
    accumulate,
    calibrate_gain,
    input_token_histogram,
    peak_memory,
    plan_step,
    validate_schema,
)

RUNS = Path("runs")

#: Bumped when a change makes new run records incomparable with old ones, so a
#: resumed sweep cannot silently average across two different training regimes.
#: 2 = bucketed batching, accumulation on an attention budget, real token counts.
#: 3 = deduplicated disjoint splits and a higher-entropy `random_grid` (a
#:     different Tier A distribution), a data seed independent of the model seed
#:     (so all arms share one split and can be compared pairwise), and halting
#:     decided by parse state rather than by matching the HALT symbol.
#: 4 = a training split large enough that the token-matched budget no longer runs
#:     past the val optimum. At 20k programs the 43-token arms saw 27.7 epochs and
#:     were reported 0.2-1.2 bits past their minimum -- the size of the very
#:     effects under test -- while the bit arm saw 3.6, so the two budget regimes
#:     differed in *fit* as well as compute and bracketed nothing. Same generator
#:     and same distribution as schema 3; incomparable because the fit regime is
#:     different, which is exactly what a schema bump is for.
SCHEMA = 4

#: Depth-vs-width at roughly matched parameter count. Structured tasks usually
#: want depth; this is the axis that tests whether drawing is one of them.
SHAPES = {
    "wide": {"d_model": 192, "n_layers": 2, "n_heads": 4},
    "square": {"d_model": 128, "n_layers": 4, "n_heads": 4},
    "deep": {"d_model": 96, "n_layers": 8, "n_heads": 4},
}

#: Deliberately over budget. Not part of any sweep -- these exist to estimate how
#: much of `bits_per_drawing` is the data's own entropy, so that a difference
#: between two arms can be read against the headroom rather than against the
#: total. Without one, "145.3 vs 145.8" has no denominator.
REFERENCE_SHAPES = {
    "reference": {"d_model": 256, "n_layers": 6, "n_heads": 8},
}
ALL_SHAPES = {**SHAPES, **REFERENCE_SHAPES}


@dataclass
class TrainConfig:
    codec: str = "byte"
    shape: str = "square"
    data: str = "synthetic"           # "synthetic" | "quickdraw"
    categories: tuple[str, ...] = ("cat",)
    tier: Tier = Tier.L1
    #: Sized against the token budget, not against how much data seemed like
    #: enough. The token-matched arms process ~23.4M tokens at ~42 tokens per
    #: program, so 20k programs is 27.7 epochs and every one of them was reported
    #: past its val minimum; 100k is 5.5 and the minimum falls outside the budget.
    #: `synthetic.split(100_000, 1_000)` returns unique, val-disjoint programs in
    #: ~3.6s, so this costs nothing but the sweep re-run.
    n_train: int = 100_000
    n_val: int = 1_000
    max_len: int = 2048
    batch_size: int = 64
    steps: int = 4_000
    #: When set, `steps` is derived so every codec processes the same number of
    #: tokens. This is the honest comparison: the bit arm's sequences are ~8x
    #: longer, so matching optimiser steps would silently hand it 8x the compute.
    token_budget: int | None = None
    #: Peak-memory cap, in units of rows x positions^2. Attention memory is
    #: quadratic in sequence length, so a fixed row count means an envelope that
    #: swings ~8x between the byte and bit arms; a batch over budget is split
    #: into accumulation steps instead. 24M measured at ~7.5 GiB on MPS.
    attn_budget: int = 24_000_000
    lr: float = 3e-3
    warmup: int = 200
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    seed: int = 0
    #: Integer-affine expansion of the *training* split only, as kwargs for
    #: `augment.policies` (e.g. `{"shifts": (-8, 8), "mirror": True}`). Empty
    #: disables it. The group is restricted to integer translation, mirroring and
    #: quarter turns because rounding commutes with those and with nothing else:
    #: a non-integer scale destroys 77% of Tier C's repeat structure, which is
    #: the corpus property claim 2 exists to measure.
    augment: dict = field(default_factory=dict)
    #: Common random numbers across codecs: every arm gets the same non-embedding
    #: initialisation, so a paired difference measures the codec and not the
    #: draw. Off by default because it makes runs incomparable with every record
    #: written before it -- a shared-init row must never be paired against an
    #: independent-init one, which the regime component of the tag enforces.
    share_init: bool = False
    #: Condition the model on each drawing's category (`--conditional`).
    #:
    #: QuickDraw only, and the label ceiling is arithmetic: the most a class label
    #: can buy in `bits/drawing` is the mutual information between drawing and
    #: class, which is bounded by H(class) = log2(5) = 2.32 bits on the balanced
    #: five-category corpus -- *below* that corpus's ~2.5-bit resolution floor. So
    #: the likelihood effect is unresolvable before the run starts, by
    #: construction, and the axis has to be read as controllability instead
    #: (`dm/eval/conditioning.py`). Recorded in the config so a conditional record
    #: can never be differenced against an unconditional one by accident.
    conditional: bool = False
    #: Data is held fixed across model seeds on purpose. Deriving the split from
    #: `seed` scored every arm on a *different* val set, and val sets differ from
    #: each other by ~3.5 bits/drawing -- which was the entire reported seed
    #: spread, an order of magnitude above the codec effects it was meant to
    #: bracket. Fixed here, `seed` varies initialisation and batch order only,
    #: and every arm is scored on identical programs.
    data_seed: int = 0
    eval_every: int = 500
    gen_samples: int = 128
    #: Multiple of val p99 to allow a sample before calling it truncated. The cap
    #: is a compute bound, not a measurement: at 1x, ~1% of *real* programs would
    #: not fit and get scored as the model failing to terminate.
    gen_cap: float = 2.0
    device: str = "mps"
    tag: str = ""
    extra: dict = field(default_factory=dict)
    #: Direction 3's latent-feedback architecture, or `"none"` for the model this
    #: project trained everything else with.
    #:
    #: **Last field and defaulted, so no existing record changes.** Every run in
    #: `runs/` predates it, `config_from_record` rebuilds a `TrainConfig` from the
    #: stored dictionary, and an old dictionary simply omits the key. `"none"`
    #: takes the standard step body unchanged, so a schema-4 record written today
    #: is byte-identical to one written before this field existed.
    #:
    #: `SCHEMA` is deliberately **not** bumped. A feedback-trained record is
    #: indeed incomparable with a standard one, but that is carried per record by
    #: this field, which is explicit and readable; a global bump would also mark
    #: every unchanged standard run as a new regime and retire a closed sweep for
    #: nothing.
    feedback_schema: str = "none"
    #: Which named pass schedule the feedback arm draws its plan from.
    #:
    #: Defaulted to the project adaptation v0 froze, so a run that does not name
    #: one trains exactly what it trained before. F5b compares three candidates
    #: and the record has to say which one produced it: the realized pass
    #: histogram alone cannot separate a schedule from an unlucky draw, and a
    #: freeze that inferred the schedule from the histogram would be reading its
    #: own output.
    #:
    #: Ignored entirely when `feedback_schema="none"`, which never draws a plan.
    pass_schedule: str = DEFAULT_PASS_SCHEDULE
    #: Which named calibration the shared input norm's gain gets at the feedback
    #: phase transition, or `"none"` for the initialisation every artifact so far
    #: trained under.
    #:
    #: **A training-protocol condition, not an architecture arm.** The forward
    #: equation is `glu_source_v2`'s either way and the checkpoint's tensor set is
    #: unchanged, so a fourth `feedback_schema` would be a name that distinguishes
    #: nothing a reader could check -- while joining a set of arms whose
    #: checkpoints already load `strict=True` into one another. What differs is
    #: what the gain is calibrated *to*, and that is a fact about the run, so it
    #: lives in the run's configuration and in its record.
    #:
    #: Defaulted and ignored when `feedback_schema="none"`, which has no gain.
    gain_calibration: str = DEFAULT_GAIN_CALIBRATION
    #: Optional artifact marker used by the engineering smoke.  It is kept out
    #: of ordinary records so the historical standard schema stays unchanged;
    #: when present it is copied into both the record and checkpoint payload.
    artifact_provenance: str | None = None


def config_from_record(record: dict, **overrides) -> TrainConfig:
    """The `TrainConfig` a run record was written from.

    The inverse of what `train` serialises, and the only supported way to
    rebuild a run's corpus after the fact: `split()` filters val against train,
    so regenerating with default arguments gives a *different* val set, and two
    val sets differ by ~3.5 bits/drawing -- several times any effect under test.
    `extra` and `augment` come back as dicts and `tier` as an int, so both are
    restored here rather than at each call site.
    """
    config = {**record["config"], **overrides}
    config["tier"] = Tier(config["tier"])
    config["categories"] = tuple(config["categories"])
    if config.get("extra"):
        # Same JSON-has-no-tuple problem as `augment` below, and the same fix.
        # `composed`'s `orbit_sizes` is the only sequence-valued `extra` today,
        # and a list works everywhere it is read -- so this is about the config
        # round-tripping to an equal `TrainConfig` rather than about the corpus,
        # which is why it is a normalisation and not a guard.
        config["extra"] = {
            k: tuple(v) if isinstance(v, list) else v
            for k, v in config["extra"].items()
        }
    if config.get("augment"):
        # JSON has no tuple, so `shifts` and `quarter_turns` come back as lists
        # and `augment.policies` does `shifts + (0,)` -- a TypeError, raised
        # only for the three augmented Tier C records and only when something
        # rebuilds them. Restored here rather than defended against there,
        # because `policies` taking either type would let a list reach `Affine`
        # and make the *corpus* depend on how the config was stored.
        config["augment"] = {
            k: tuple(v) if isinstance(v, list) else v
            for k, v in config["augment"].items()
        }
    return TrainConfig(**config)


def build_labels(cfg: TrainConfig) -> tuple[list[int], list[int]] | None:
    """Each program's class index, for the split `build_data` returns, or None.

    **Only QuickDraw carries labels, and only because they were always there.**
    The other three corpora have no class structure to condition on: Tier A is
    generated from one grammar, Tier C is one icon set, and the composed corpus's
    class would be the *motif's* category, which is not what the scene is of.
    Returning None rather than a zeros array is deliberate -- a single-class
    conditioning signal is a constant, and a constant added to every position is a
    bias the model already has.

    Read through `load_labelled`, which interleaves labels in the same traversal
    that orders the programs, so a label cannot end up on the wrong drawing.
    """
    if cfg.data != "quickdraw":
        return None
    _, train = quickdraw.load_labelled(cfg.categories, "train", limit=cfg.n_train,
                                      **cfg.extra)
    _, val = quickdraw.load_labelled(cfg.categories, "valid", limit=cfg.n_val,
                                     **cfg.extra)
    return train, val


def build_data(cfg: TrainConfig) -> tuple[list[bytes], list[bytes]]:
    if cfg.data == "synthetic":
        return synthetic.split(
            cfg.n_train, cfg.n_val, seed=cfg.data_seed, tier=cfg.tier, **cfg.extra
        )
    if cfg.data == "tabler":
        # Tier C. `split` is grouped by icon family and by exact program
        # identity, because `battery-1..4` are one drawing four times and 125
        # icons share a program outright; either across the split scores
        # memorisation as generalisation (section 8).
        train, val = tabler.split(**cfg.extra)
        if cfg.augment:
            # Train only. Augmenting val would change the denominator between
            # arms and destroy the pairing that every codec difference relies on.
            train = augment.expand(train, augment.policies(**cfg.augment))
        return train, val
    if cfg.data == "quickdraw":
        # `extra` carries `rdp_eps` and `margin`. They belong in the config
        # rather than in the call because `rdp_eps` *is* the corpus: it sets
        # sequence length, and at 8x the byte count the bit arm is the one that
        # decides whether Tier B fits under `max_len` at all. `asdict(cfg)`
        # lands it in the run record, so a Tier B row states the corpus it was
        # measured on instead of inheriting a default that can move.
        train = quickdraw.load(cfg.categories, "train", limit=cfg.n_train, **cfg.extra)
        val = quickdraw.load(cfg.categories, "valid", limit=cfg.n_val, **cfg.extra)
        return train, val
    if cfg.data == "composed":
        # The constructed compositional corpus. `extra` carries the composition
        # policy -- `control`, `distractor_p`, `orbit_sizes` -- and the policy
        # *is* the corpus here in the way `rdp_eps` is for Tier B: a control
        # corpus and a transformed one differ in nothing else and would otherwise
        # be indistinguishable in a record. The flat form is what trains; the
        # structure lives in `dm.data.composed.build`, which the metrics rebuild
        # deterministically from the same seed when they need the provenance.
        return composed.split(cfg.n_train, cfg.n_val, seed=cfg.data_seed,
                              categories=cfg.categories, **cfg.extra)
    if cfg.data == "feedback":
        # Direction 3's paired arms. `structure` *is* the corpus here, in the way
        # `control` is for `composed`: the relational and relation-destroyed
        # splits are the same programs in the same order differing only in which
        # continuation follows which prefix, and nothing but this key
        # distinguishes them in a config.
        #
        # Both arms are built even though one is returned. The derangement is a
        # permutation over the whole block table, so building one arm in
        # isolation is not defined -- and the pairing is the entire experiment.
        extra = dict(cfg.extra)
        structure = extra.pop("structure", "relational")
        if structure not in ("relational", "relation_destroyed"):
            raise ValueError(
                f"unknown feedback training structure {structure!r}; expected "
                "'relational' or 'relation_destroyed'"
            )
        paired = feedback_data.build(
            cfg.n_train, cfg.n_val, data_seed=cfg.data_seed,
            # Frozen, not a knob: the donor map is a function of the corpus and
            # one seed named in the protocol, and a tunable seed is a search over
            # controls (`docs/directions.md` §7 invariant 1).
            # No `flatten` here: this source *is* the flat trace, because the
            # step relation only exists once REPEAT has been expanded. The
            # builder refuses `flatten=False` if a config carries it anyway.
            seed=seed_for("derangement"), tier=cfg.tier, **extra,
        )
        train = (paired.relational if structure == "relational"
                 else paired.destroyed)
        return train, paired.val
    raise ValueError(f"unknown data source {cfg.data!r}")


class Schedule(Protocol):
    """What `lr_at` needs, so the planner's own config can use it unchanged.

    A shared schedule and not a copied one: an AR arm and a planner arm are
    compared on bits/drawing, and two cosine schedules that had drifted apart
    would put part of that difference in the optimiser.
    """

    lr: float
    warmup: int
    steps: int


def lr_at(step: int, cfg: Schedule) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    progress = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def pick_device(requested: str) -> torch.device:
    if requested == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def release(device: torch.device) -> None:
    """Return cached blocks to the driver between runs.

    A sweep trains 32 models in one process. Caching allocators hold every block
    they ever handed out, so without this the arms accumulate each other's peaks
    and the last cell pays for all the earlier ones.
    """
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def evaluation_variates(*, seed: int, step: int, rows: int,
                        width: int) -> torch.Tensor:
    """Pre-generate the uniforms used by a training-time quality diagnostic.

    The diagnostic is deliberately a consumer of its own stream.  Sampling
    through ``torch.multinomial`` would consume the global generator, which is
    also where the legacy batch sampler drew its permutations.  A change in a
    model's stopping behaviour could then change the next epoch's training
    rows.  Deriving this block from the named Direction 3 variate stream and the
    evaluation step makes the diagnostic reproducible and irrelevant to the
    optimisation stream.
    """
    eval_seed = (seed_for("variates") + int(seed) * 1_000_003
                 + int(step) * 10_000_019) % (2**63 - 1)
    return torch.rand(
        rows, width,
        generator=torch.Generator().manual_seed(eval_seed),
        dtype=torch.float32,
    )


def checkpoint(name: str, result: dict, weights: dict) -> None:
    """Write one run's weights and record to `runs/`, atomically, in that order.

    Called at **every eval**, not once at the end. A trainer that stores nothing
    until it returns stores nothing at all when it is killed, and that is not
    hypothetical: the schema-2 planner re-run reached step 11,500 of 12,000 over
    2.6 hours, took a SIGKILL, and left no file at all. The one number it was
    launched for survives only as terminal scrollback
    (`docs/history/lost-run.md`), with no `val_bits` and so no paired interval,
    which is the form every claim-3 difference is quoted in.

    **Atomic, because a truncated record is worse than a missing one.**
    `scripts/sweep.py` globs `runs/*.json` and every reader assumes a whole
    record, so a kill landing inside a plain `write_text` would leave half of one
    where a whole one is expected -- a crash at best and a silently short
    `val_bits` at worst. The temporary files sit in `runs/` rather than in the
    system temporary directory so `os.replace` stays within one filesystem,
    which is the condition that makes it atomic in the first place.

    **Weights first, then the record**, so the only reachable inconsistency is
    the harmless one. The record is the commit marker: a kill between the two
    writes leaves weights one eval ahead of a record that does not claim them,
    and a record that understates what exists costs nothing. The other order
    publishes a record whose `.pt` does not hold the model it describes, and
    `scripts/resample.py` would resample the wrong checkpoint without noticing.
    """
    RUNS.mkdir(parents=True, exist_ok=True)
    staged = RUNS / f".{name}.pt.tmp"
    published_weights = dict(weights)
    if result.get("provenance") is not None:
        # A checkpoint must carry the same engineering/scientific provenance as
        # its record.  A filename is not an audit field: it is too easy for a
        # later smoke report to make an old checkpoint look like a pilot cell.
        published_weights["provenance"] = result["provenance"]
        published_weights["record_name"] = name
    torch.save(published_weights, staged)
    os.replace(staged, RUNS / f"{name}.pt")
    checkpoint_path = RUNS / f"{name}.pt"
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    result["checkpoint_sha256"] = digest.hexdigest()
    staged = RUNS / f".{name}.json.tmp"
    staged.write_text(json.dumps(result, indent=2))
    os.replace(staged, RUNS / f"{name}.json")


def micro_batches(
    inputs: torch.Tensor, targets: torch.Tensor, budget: int,
    classes: torch.Tensor | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]]:
    """Split a batch into accumulation steps that fit `budget` rows x T^2.

    The gradient is unchanged: the caller normalises the summed NLL by the token
    count of the whole batch, so this trades throughput for peak memory and
    nothing else. Optimisation is identical whether or not it splits.

    `classes` is sliced with the rows it belongs to. It is a parameter rather than
    something the caller re-slices because the row offsets live here: a
    conditioning tensor sliced against a different offset trains every drawing on
    a neighbour's class, and the loss would look entirely healthy.
    """
    rows, width = inputs.shape
    per = max(1, min(rows, budget // max(1, width * width)))
    for i in range(0, rows, per):
        yield (inputs[i : i + per], targets[i : i + per],
               None if classes is None else classes[i : i + per])


def train(cfg: TrainConfig, verbose: bool = True) -> dict:
    torch.manual_seed(cfg.seed)
    device = pick_device(cfg.device)
    codec = CODECS[cfg.codec]

    train_programs, val_programs = build_data(cfg)
    labels = build_labels(cfg) if cfg.conditional else None
    if cfg.conditional and labels is None:
        raise ValueError(
            f"--conditional needs class labels and {cfg.data!r} has none; only "
            "QuickDraw carries categories (dm.train.build_labels)"
        )
    n_classes = len(cfg.categories) if labels else 0
    train_set = ProgramDataset(train_programs, codec, cfg.max_len,
                               labels=labels[0] if labels else None)
    val_set = ProgramDataset(val_programs, codec, cfg.max_len,
                             labels=labels[1] if labels else None)
    train_loader = loader(train_set, cfg.batch_size, seed=cfg.seed)
    val_loader = loader(val_set, cfg.batch_size, shuffle=False,
                        seed=cfg.data_seed)
    lengths = val_set.length_stats()
    train_lengths = train_set.length_stats()
    if lengths["truncated"] or train_lengths["truncated"]:
        # Loud, and not an exception: the run is still *interpretable* as long as
        # the reader knows the tail was cut. It is bits/drawing that stops being
        # comparable, and that is a reporting decision, not a training fault.
        print(
            f"  WARNING: max_len={cfg.max_len} truncates "
            f"{100 * train_lengths['truncated']:.2f}% of train and "
            f"{100 * lengths['truncated']:.2f}% of val on the {cfg.codec} codec "
            f"-- bits/drawing is NOT comparable across codecs for this run",
            flush=True,
        )

    if cfg.token_budget:
        train_len = train_set.length_stats()["mean"]
        cfg.steps = max(50, int(cfg.token_budget / (cfg.batch_size * train_len)))
        cfg.warmup = min(cfg.warmup, cfg.steps // 10)
        cfg.eval_every = max(1, cfg.steps // 6)

    model_cfg = Config(vocab_size=codec.vocab_size, max_len=cfg.max_len,
                       **ALL_SHAPES[cfg.shape])
    if n_classes:
        # Applied after construction rather than as a keyword, so the
        # unconditional path builds the identical `Config` it always did -- the
        # comparison this axis rests on is that the two arms differ by an addition.
        model_cfg = replace(model_cfg, n_classes=n_classes)
    if cfg.feedback_schema != "none":
        # Same reasoning one axis over: a standard run builds the `Config` it
        # always built, and the feedback arm differs from it by one field.
        model_cfg = replace(model_cfg,
                            feedback_schema=validate_schema(cfg.feedback_schema))
    model = DrawingLM(model_cfg)
    if cfg.share_init:
        model.share_non_embedding_init(cfg.seed)
    model = model.to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95)
    )

    name = cfg.tag or f"{cfg.data}_{cfg.codec}_{cfg.shape}_s{cfg.seed}"
    if verbose:
        # Flushed because the only way a multi-hour sweep is ever run is with
        # stdout redirected to a log, and a redirected stdout is block-buffered:
        # without this the driver's own `flush=True` lines appear and the
        # per-step progress under them does not, so a live run looks hung.
        print(
            f"[{name}] params={model.n_params():,} vocab={codec.vocab_size} "
            f"len(mean={lengths['mean']:.0f} p99={lengths['p99']:.0f} max={lengths['max']:.0f}) "
            f"steps={cfg.steps} device={device}",
            flush=True,
        )

    history: list[dict] = []
    val_bits: list[float] = []
    best: dict = {}
    best_val_bits: list[float] = []
    # Hoisted out of the record because the record is now written at every eval.
    # `fingerprint` digests every program in both splits and the corpus does not
    # change while a run is training; `n_params` is likewise fixed, and reading
    # it here keeps `snapshot` from closing over a `model` that the teardown
    # below deletes.
    corpus = fingerprint(train_programs, val_programs)
    params = model.n_params()
    # Only when the feedback arm is running, so a standard record keeps exactly
    # the keys it has always had.
    feedback = (
        None if cfg.feedback_schema == "none"
        else Accounting(device=str(device), feedback_schema=cfg.feedback_schema,
                        stride=codec.stride, pass_schedule=cfg.pass_schedule,
                        gain_calibration=cfg.gain_calibration)
    )
    calibration_step: int | None = None
    input_counts = None
    if feedback is not None:
        # Refused here, before the first step, rather than at the first draw: an
        # unknown schedule that surfaced 12,000 steps in would have trained a
        # cell under the default and recorded the name that was asked for.
        _pass_schedule(cfg.pass_schedule)
        _gain_calibration(cfg.gain_calibration)
        if cfg.gain_calibration != DEFAULT_GAIN_CALIBRATION:
            # Derived from the schedule, so a calibration cannot fire at a step
            # the channel does not actually switch on at, and refused *before the
            # run* when the budget never reaches that step -- a cell whose
            # declared lever never moved would otherwise be indistinguishable
            # from one where it moved and changed nothing.
            calibration_step = _feedback_phase_transition(cfg.steps,
                                                          cfg.pass_schedule)
            if calibration_step >= cfg.steps:
                raise ValueError(
                    f"{cfg.pass_schedule!r} first draws more than one pass at "
                    f"step {calibration_step}, which is outside a {cfg.steps}-step "
                    f"budget, so gain_calibration={cfg.gain_calibration!r} would "
                    "never fire: incomplete"
                )
            # Once, before the loop: a function of the corpus and the codec
            # alone, so it neither changes with training nor depends on how the
            # batches happened to be composed.
            input_counts = input_token_histogram(train_set.seqs,
                                                 model_cfg.vocab_size)

    def snapshot(complete: bool) -> dict:
        """The record as of the last eval. Whole at every eval, not only at exit.

        `complete` is the field that keeps a killed run from being read as a
        finished one. Everything else here is exactly what this function used to
        build once, after the loop.
        """
        config = {**asdict(cfg), "tier": int(cfg.tier)}
        if cfg.artifact_provenance is None:
            # Keep ordinary records byte/schema-compatible with runs written
            # before the engineering provenance field existed.
            config.pop("artifact_provenance", None)
        record = {
            "name": name,
            "schema": SCHEMA,
            # Whether the loop ran to `cfg.steps`, and how far it actually got.
            # A partial record is a real measurement of a shorter run and is
            # kept as one -- what it is not is a rung of the budget it asked
            # for, and `config["steps"]` alone cannot tell the two apart.
            # `scripts/sweep.py` keys its ladders on `steps_done` for exactly
            # this reason.
            "complete": complete,
            "steps_done": history[-1]["step"] if history else 0,
            "config": config,
            # The data, not the config that named it. Every table that
            # differences two rows checks this first (`dm/data/fingerprint.py`):
            # a config-derived key cannot see a defaulted `rdp_eps` or an
            # `n_train`, and run 4 was compared against a baseline differing in
            # both plus its categories.
            "corpus": corpus,
            "model": {"params": params, **asdict(model_cfg)},
            "val_lengths": lengths,
            "train_lengths": train_lengths,
            # Length of each val program in *bytecode bytes*, which is the one
            # unit all four codecs share -- `val_lengths` is in the codec's own
            # symbols and so is 8x larger on the bit arm. Recorded because a
            # representation cost is a per-symbol rate: the converged fusion
            # penalty is +0.041 bits per bytecode byte, and stating it as +1.22
            # bits/drawing hides both that it scales with drawing complexity
            # (+0.2 bits on the shortest val quintile, +2.7 on the longest) and
            # that the number is a property of this corpus's length
            # distribution. Tier B's programs are an order of magnitude longer,
            # so bits/drawing will not carry across datasets and this does.
            "val_bytes": [len(p) for p in val_programs],
            # Bits for each val program under the final model. Every arm is
            # trained on the same `data_seed`, so these are aligned across runs
            # and support a paired comparison; see
            # `dm.eval.metrics.paired_delta`.
            "val_bits": val_bits,
            # The same array at the val argmin. `final` is what gets reported --
            # a checkpoint picked by val and then scored on val is selection
            # bias -- but `final` minus `best` is the drift, and under schema 3
            # that drift was the same size as the typing and fusion effects.
            # Recorded so the question "was this arm past its optimum?" is
            # answerable from the file.
            "best_val_bits": best_val_bits,
            "history": history,
            "final": history[-1] if history else {},
            "best": best,
        }
        if cfg.artifact_provenance is not None:
            record["provenance"] = cfg.artifact_provenance
        if feedback is not None:
            # Added only on the feedback arm, so a standard record keeps exactly
            # the key set it has always had and old readers see no new field.
            record["feedback"] = feedback.as_dict()
        return record

    step, started = 0, time.time()
    batches = iter(train_loader)
    # Accumulated on-device so the step loop never blocks on a sync; read only
    # at eval time. `content` is real tokens, `padded` is positions actually
    # pushed through the model -- the gap is what bucketing recovers.
    content = torch.zeros((), dtype=torch.long, device=device)
    padded = 0
    model.train()
    while step < cfg.steps:
        try:
            index, inputs, targets = next(batches)
        except StopIteration:
            batches = iter(train_loader)
            continue

        for group in opt.param_groups:
            group["lr"] = lr_at(step, cfg)
        # Read off the CPU tensors, before the move, and only on the feedback
        # arm. The pass plan must not depend on where the batch happens to live:
        # a device reduction would make the draw a function of the accelerator's
        # own dispatch, and the same experiment on two machines would stop being
        # the same experiment (`docs/directions.md` §7 invariant 5).
        row_lengths = None if feedback is None else (inputs != PAD).sum(dim=1)
        content_symbols = 0 if feedback is None else int((targets != PAD).sum())
        inputs, targets = inputs.to(device), targets.to(device)
        # The conditioning signal is a property of the *program*, so it is looked
        # up by the batch's own index rather than threaded through `collate` --
        # one convention, stated in `dm.data.dataset.ProgramDataset`.
        row_classes = (None if train_set.labels is None
                       else train_set.labels[index].to(device))
        n_tokens = (targets != PAD).sum().clamp(min=1)
        opt.zero_grad(set_to_none=True)
        total_nll = torch.zeros((), device=device)
        if feedback is not None:
            if calibration_step is not None and step == calibration_step:
                # The F5c correction package's entire intervention, applied once
                # at the predeclared transition and recorded in full. It reads
                # the embedding table and the corpus histogram and nothing else:
                # no loss, no validation score, no sample, no case
                # (`docs/directions.md` §7 invariant 1).
                feedback.gain_calibration_event = calibrate_gain(
                    model, input_counts, rule=cfg.gain_calibration, step=step)
                calibration_step = None
            # One loop, two step bodies. The standard body below is left exactly
            # as it was rather than routed through the feedback one: a schema-4
            # record has to stay reproducible byte for byte, and "K=1 is the same
            # arithmetic" is an argument, not a guarantee.
            plan = plan_step(step=step, steps=cfg.steps, lengths=row_lengths,
                             width=inputs.shape[1], d_model=model_cfg.d_model,
                             schedule=cfg.pass_schedule)
            per_pass = accumulate(model, inputs, targets, row_classes, plan=plan,
                                  n_tokens=n_tokens, budget=cfg.attn_budget)
            # Pass 1 is ordinary next-token loss and the only number comparable
            # to a run that never had a channel; the objective it was trained on
            # is reconstructible from `feedback.pass_histogram` and this.
            total_nll = per_pass[0]
            feedback.observe(plan, rows=inputs.shape[0], width=inputs.shape[1],
                             content_symbols=content_symbols)
        else:
            for chunk_in, chunk_tgt, chunk_cls in micro_batches(inputs, targets,
                                                                cfg.attn_budget,
                                                                row_classes):
                logits = (model(chunk_in, chunk_cls) if chunk_cls is not None
                          else model(chunk_in))
                nll = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), chunk_tgt.reshape(-1),
                    ignore_index=PAD, reduction="sum",
                )
                (nll / n_tokens).backward()
                total_nll += nll.detach()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        content += n_tokens
        padded += inputs.numel()
        step += 1

        if step % cfg.eval_every == 0 or step == cfg.steps:
            if feedback is not None:
                # Read at eval time rather than per step: both are host-side
                # queries and the step loop is deliberately free of syncs.
                feedback.wall_clock_s = round(time.time() - started, 1)
                feedback.peak_memory_bytes = peak_memory(device)
            record = {"step": step, "train_loss": float(total_nll / n_tokens),
                      "tokens_seen": int(content), "padded_positions": padded}
            record |= bits_per_drawing(model, val_loader, device)
            # A zero-width decode is not an evaluation; keep the cap positive
            # even for a tiny synthetic split whose p99 rounds down to zero.
            max_new = max(1, min(cfg.max_len, int(cfg.gen_cap * lengths["p99"])))
            record |= sample_quality(
                model, codec, n=cfg.gen_samples,
                max_new=max_new,
                device=device, top_k=40, reference=val_programs,
                variates=evaluation_variates(
                    seed=cfg.seed, step=step, rows=cfg.gen_samples,
                    width=max(1, max_new),
                ),
                # Round-robin over the classes rather than one class or a random
                # draw: the reference is the whole val split, which is balanced by
                # `_interleave`, so an unbalanced sample would report a
                # distribution mismatch that is the sampler's choice of classes.
                classes=(None if not n_classes else
                         torch.arange(cfg.gen_samples, device=device) % n_classes),
            )
            record["elapsed_s"] = round(time.time() - started, 1)
            # Two copies are kept, and no more: the final model's and the val
            # argmin's. Every eval's copy would be six 1,000-float arrays per
            # record, but keeping only the final one made the schema-3 overfit
            # regime invisible from the records -- the paired table could not be
            # recomputed at the optimum, so a run reported 0.2-1.2 bits past its
            # minimum looked exactly like one that was not.
            val_bits = record.pop("val_bits")
            if not best or record["bits_per_drawing"] < best["bits_per_drawing"]:
                best, best_val_bits = record, val_bits
            history.append(record)
            checkpoint(name, snapshot(complete=False),
                       {"cfg": asdict(model_cfg), "state": model.state_dict()})
            model.train()
            if verbose:
                print(
                    f"  step {step:>6}  loss {record['train_loss']:.4f}  "
                    f"bits/drawing {record['bits_per_drawing']:8.1f}  "
                    f"gen_valid {record['gen_validity']:.3f}  "
                    f"{record['elapsed_s']:.0f}s",
                    flush=True,
                )

    # `snapshot` again rather than a second copy of the same dict. Every eval
    # wrote a whole record with `complete` false; this rewrites the last one
    # with it true, so the flag means exactly "the trainer returned" and never
    # "the loop reached its last step", which are different claims when a kill
    # lands between them.
    result = snapshot(complete=True)
    checkpoint(name, result, {"cfg": asdict(model_cfg), "state": model.state_dict()})

    del model, opt, train_loader, val_loader, content
    release(device)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--codec", default="byte", choices=sorted(CODECS))
    ap.add_argument("--shape", default="square", choices=sorted(ALL_SHAPES))
    ap.add_argument("--data", default="synthetic",
                    choices=["synthetic", "quickdraw", "tabler", "composed",
                             "feedback"])
    ap.add_argument("--structure", default=None,
                    choices=["relational", "relation_destroyed"],
                    help="feedback: which arm of Direction 3's paired corpus")
    ap.add_argument("--categories", nargs="+", default=["cat"])
    # Defaults come from TrainConfig so the split size is stated once. A second
    # hardcoded copy here is how the sweep kept running at 20k after the
    # dataclass moved to 100k.
    ap.add_argument("--n-train", type=int, default=TrainConfig.n_train)
    ap.add_argument("--steps", type=int, default=4_000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--tag", default="")
    # Tier B only. Ignored by the synthetic path, which has no RDP stage.
    ap.add_argument("--rdp-eps", type=float, default=None)
    ap.add_argument("--margin", type=int, default=None)
    # Tier A only, and this is claim 2's corpus switch: `--flatten` trains on
    # the L0 *trace* of each program, so translational structure is present in
    # the geometry and absent from the alphabet. Without it the repeat motif is
    # either missing (Tier.L0) or handed over as an opcode (Tier.L1).
    ap.add_argument("--flatten", action="store_true",
                    help="train on the L0 expansion of each program (claim 2)")
    ap.add_argument("--max-repeat", type=int, default=None)
    ap.add_argument("--min-repeat", type=int, default=None)
    # `--data composed` only, and these four *are* the corpus: a control scene
    # and a transformed one differ in nothing else, and the flat and structured
    # spellings of one scene are two different experiments (`dm/data/composed.py`).
    # They travel in `extra` so `asdict(cfg)` lands them in the record and
    # `dm/data/fingerprint.py` gives each policy its own corpus key.
    ap.add_argument("--control", action="store_true",
                    help="composed: translation-only orbits, the matched control")
    ap.add_argument("--structured", action="store_true",
                    help="composed: train on the REPEATX form, not the flat trace "
                         "(the only spelling on which the fusion axis exists)")
    ap.add_argument("--orbit-sizes", type=int, nargs="+", default=None,
                    help="composed: copies per orbit to allow, e.g. 2. Pass 2 to "
                         "match the control, which the canvas caps at two")
    ap.add_argument("--distractor-p", type=float, default=None,
                    help="composed: per-free-cell chance of a non-copy motif")
    ap.add_argument("--share-init", action="store_true",
                    help="common random numbers: identical non-embedding init across codecs")
    ap.add_argument("--feedback-schema", default="none",
                    choices=list(SUPPORTED_FEEDBACK_SCHEMAS),
                    help="Direction 3: train with latent feedback. 'none' is the "
                         "model everything else in this project was trained with, "
                         "and takes the unchanged step body")
    ap.add_argument("--pass-schedule", default=DEFAULT_PASS_SCHEDULE,
                    choices=sorted(PASS_SCHEDULES),
                    help="Direction 3: which named pass schedule the feedback arm "
                         "draws from. Ignored when --feedback-schema is 'none'")
    ap.add_argument("--gain-calibration", default=DEFAULT_GAIN_CALIBRATION,
                    choices=sorted(GAIN_CALIBRATIONS),
                    help="Direction 3: how the shared input norm's gain is set at "
                         "the feedback phase transition. A training-protocol "
                         "condition, not an architecture arm. Ignored when "
                         "--feedback-schema is 'none'")
    ap.add_argument("--conditional", action="store_true",
                    help="quickdraw: condition on each drawing's category. The label "
                         "ceiling is arithmetic (log2 of the category count) and sits "
                         "under the corpus's resolution floor, so read this axis with "
                         "scripts/conditioning.py, not with bits/drawing")
    ap.add_argument("--augment-shifts", type=int, nargs="*", default=None,
                    help="integer translations to expand the TRAIN split by, e.g. -8 8")
    ap.add_argument("--augment-mirror", action="store_true", help="add the x mirror")
    ap.add_argument("--augment-turns", type=int, nargs="*", default=None,
                    help="quarter turns to include, e.g. 0 1 2 3")
    args = ap.parse_args()
    extra = {
        k: v
        for k, v in (
            ("rdp_eps", args.rdp_eps), ("margin", args.margin),
            ("max_repeat", args.max_repeat), ("min_repeat", args.min_repeat),
            ("flatten", args.flatten or None),
            ("control", args.control or None),
            ("structured", args.structured or None),
            ("distractor_p", args.distractor_p),
            ("orbit_sizes",
             tuple(args.orbit_sizes) if args.orbit_sizes is not None else None),
            ("structure", args.structure),
        )
        if v is not None
    }
    aug: dict = {}
    if args.augment_shifts is not None:
        aug["shifts"] = tuple(args.augment_shifts)
    if args.augment_mirror:
        aug["mirror"] = True
    if args.augment_turns is not None:
        aug["quarter_turns"] = tuple(args.augment_turns)
    train(
        TrainConfig(
            codec=args.codec, shape=args.shape, data=args.data,
            categories=tuple(args.categories), n_train=args.n_train, steps=args.steps,
            batch_size=args.batch_size, lr=args.lr, max_len=args.max_len, seed=args.seed,
            data_seed=args.data_seed, eval_every=args.eval_every, device=args.device,
            tag=args.tag, extra=extra, share_init=args.share_init, augment=aug,
            conditional=args.conditional, feedback_schema=args.feedback_schema,
            pass_schedule=args.pass_schedule,
            gain_calibration=args.gain_calibration,
        )
    )


if __name__ == "__main__":
    main()
