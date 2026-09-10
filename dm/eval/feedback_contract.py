"""Direction 3's F0 contract: everything fixed *before* feedback code exists.

`docs/directions.md` §§2-6 is the prose contract and this Module is its
executable half.  Every number a later stage is allowed to read -- schemas, seed
namespaces, the pass schedule, the practical floors, the stability band, the
promotion floors, the parameter arithmetic -- is declared here once, and
`docs/feedback-protocol-v0.json` is a serialisation of exactly this Module.  A
test compares the two, so prose, code and the frozen artifact cannot drift apart
silently (`tests/test_feedback_contract.py`).

**Why a contract Module rather than constants scattered across F1-F8.** Direction
2 paid this cost once already: its C4 driver was edited after the protocol froze
its digest, so for a day no file matching the freeze existed and the run was, on
its face, unreproducible (`docs/context-audit.md`).  The repair was to make the
freeze name content rather than intent.  Here the same rule is applied one stage
earlier: the thresholds exist, in one place, with a hash, before there is any
feedback output that could suggest what they should be.

**What v0 freezes and what it deliberately cannot.** v0 freezes the *interface*:
fixture bytes, schemas, formulas, seed namespaces, pass/fail rules and expected
parameter counts.  It does not freeze source digests, the training corpus or
checkpoint hashes -- F1 changes `dm/models/transformer.py` by design, the paired
corpora do not exist until F4, and checkpoints do not exist until F6.  Protocol
v1 adds the training freeze after the F5 smoke; protocol v2 adds checkpoint and
record bytes before any outcome score.  Two freezes, because a single one would
have to invent future hashes (`docs/directions.md` §7 invariant 10).

Nothing here reads a model, a checkpoint or a report.  That is the point: no
output may choose a corpus, donor, case, threshold or checkpoint
(`docs/directions.md` §7 invariant 1).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

from ..isa.codec import CODECS
from ..models.transformer import Config
from .reports import staged_path

DIRECTION = 3

#: Protocol identity and version chain.  v0 is the F0 contract; v1 will add the
#: training freeze at F5 and v2 the checkpoint/record freeze at F7.  A stage that
#: reads the wrong one fails closed rather than running under a freeze that never
#: covered it.
PROTOCOL = "feedback-v0"
PROTOCOL_SCHEMA = 1
PROTOCOL_PATH = Path("docs/feedback-protocol-v0.json")

#: Where the baseline fixture lives, as a string, so the serialised payload is
#: byte-identical on any platform; `dm.eval.feedback_fixture.FIXTURE_PATH` is the
#: `Path` callers use.
FIXTURE_PATH_STR = "tests/fixtures/feedback_baseline_v1.json"

#: Bumped whenever a change makes new feedback artifacts incomparable with old
#: ones.  Separate numbers because the three artifacts freeze at three different
#: stages and a shared counter would force a spurious bump on two of them.
FIXTURE_SCHEMA = 1
RECORD_SCHEMA = 1
REPORT_SCHEMA = 1

# ---------------------------------------------------------------------------
# architecture

#: `"none"` is the default and *is* the current implementation: an old config
#: dictionary omits the field, loads as `none`, creates no feedback parameters
#: and reproduces the F0 fixtures bit for bit.  `"glu_v1"` is the first opt-in
#: value.  Anything else is refused rather than silently treated as `none` --
#: a typo that disabled feedback would look exactly like a null result.
FEEDBACK_SCHEMAS: tuple[str, ...] = ("none", "glu_v1")
DEFAULT_FEEDBACK_SCHEMA = "none"

# v0 remains the compatibility contract for the first implementation.  The
# complete source recipe normalizes the token embedding before `W_G` (Listing 3,
# line 9), so the next training freeze must name a distinct arm rather than
# silently relabeling old engineering checkpoints as source-faithful.
SOURCE_FEEDBACK_SCHEMA = "glu_source_v1"
SOURCE_FUSION_EQUATION = (
    "z_t = RMSNorm(W_U h_(t-1) * sigmoid(W_G RMSNorm(e_t)))"
)

#: The Listing-3-faithful arm.  Lines 9 and 11 of the appendix listing name the
#: *same* module, `input_rmsnorm_1`: one learned norm reads the gate input and
#: the post-mixin stack input, and the prefix mixin sits between the gated
#: product and that norm.  `glu_source_v1` used two distinct normalization
#: operations -- affine-free at the gate, a separate learned gain on the output
#: -- and let a plain-prefix position past the second one entirely.
#:
#: A third name rather than a correction in place.  The three arms own
#: byte-identical tensor sets, so an artifact written under one loads strict
#: under any of them and computes a different function without complaint; this
#: repository holds no `runs/` artifact under `glu_source_v1`, but an external
#: one cannot be disproved.
SOURCE_FEEDBACK_SCHEMA_V2 = "glu_source_v2"
SOURCE_V2_FUSION_EQUATION = (
    "r_t = W_U h_(t-1) * sigmoid(W_G N(e_t)); "
    "m_t = e_t at a plain position else r_t; "
    "z_t = N(m_t), N = the single learned fuse.norm"
)

#: The arm F5b qualifies, F6 trains and protocol v1 freezes.  A pointer rather
#: than a literal at each call site: two source arms exist and every training,
#: smoke and freeze check has to agree on which one is current without deciding
#: for itself.
QUALIFICATION_FEEDBACK_SCHEMA = SOURCE_FEEDBACK_SCHEMA_V2

#: `z_t = RMSNorm(W_U h_(t-1) * sigmoid(W_G e_t))`, bias-free `W_U, W_G`.
#: Hidden state is the value and the token embedding is the gate.  No additive
#: embedding shortcut: with one, recurrence-trained weights can ignore the wide
#: channel and recover ordinary pretraining loss, which would make a null result
#: uninterpretable (`docs/directions.md` §3.1).
FUSION_EQUATION = "z_t = RMSNorm(W_U h_(t-1) * sigmoid(W_G e_t))"

#: The carried tensor is `self.norm(x)` -- the exact input to the tied LM head,
#: not the pre-norm residual.  Named here because "previous hidden state" has
#: three plausible readings in a pre-norm stack and only one of them is the state
#: the head consumes.
CARRIED_STATE = "self.norm(x), the normalized top-layer state consumed by the tied head"

#: The config field and the runtime argument, spelled once.  Frozen because both
#: are serialised: the field lands in every run record through `asdict`, and the
#: mode name lands in every report row -- so a rename after F6 would make old
#: records unreadable and, worse, readable as something else.
CONFIG_FIELD = "feedback_schema"
RUNTIME_MODE_ARGUMENT = "mode"

#: State-dictionary keys `glu_v1` adds, in `W_U`, `W_G`, gain order.  Parameter
#: names are checkpoint contract, not style: eight checkpoints get written at F6
#: and reloaded at F7/F8 under `strict=True`, and a rename between those stages
#: would fail the load of the pilot's own weights.
FEEDBACK_PARAMETERS: tuple[str, ...] = (
    "fuse.up.weight",     # W_U: D x D, applied to the carried normalized state
    "fuse.gate.weight",   # W_G: D x D, applied to the token embedding
    "fuse.norm.weight",   # the dedicated RMSNorm gain, initialised to 0.02
)

#: The model seam, frozen because training and evaluation are two real callers of
#: it and a seam that moves between F2 and F3 makes their results incomparable.
#: `plain` is a count and not a mask over `feedback`: fusing a *zero* state is
#: `RMSNorm(0) = 0`, which deletes the position's input entirely, and "no
#: feedback here" has to mean the raw embedding instead.
MODEL_INTERFACE: dict[str, str] = {
    "forward": "forward(idx, classes=None, *, feedback=None, plain=0, "
               "return_state=False)",
    "feedback": "(B, T, D) carried normalized states already aligned to idx, so "
                "feedback[:, t] is the state fused into position t",
    "plain": "int, or (B,) per row: how many leading positions keep the raw "
             "embedding -- BOS and the sampled prefix. Position 0 is always "
             "plain, because BOS has no predecessor state",
    "return_state": "also return the (B, T, D) normalized top-layer states, "
                    "which is what the next pass and the next decode step carry",
    "generate": "generate(..., mode='standard'|'soft'|'fused')",
    "fuse": "fuse(hidden, embed) -> the fused stack input z_t",
}

#: Initial gain of the *dedicated* fused RMSNorm, matching the 0.02 embedding
#: initialisation.  A default gain of 1 would hand the stack an input whose RMS
#: is ~50x the embedding's at step 0, so the first optimiser steps would be spent
#: undoing the initialisation rather than learning the channel.
FUSED_NORM_GAIN = 0.02

#: Where each normalization sits in the faithful arm, spelled out because
#: "normalized gate input" was true of `glu_source_v1` as well and is not what
#: distinguishes the two.  Recorded in protocol v1 so an auditor reads the
#: placement from the artifact rather than from prose.
SOURCE_V2_NORMALIZATION: dict[str, object] = {
    "shared_learned_norm": True,
    "norm_parameter": "fuse.norm.weight",
    "gate_input": "N(e_t)",
    "prefix_mixin_placement": "before the shared norm",
    "plain_prefix_stack_input": "N(e_t)",
    "raw_embedding_paths": ["training pass 1", "standard prefill"],
    "initial_gain": FUSED_NORM_GAIN,
    "listing": "Listing 3 (appendix), lines 9-11",
}

#: Runtime modes.  Mode is an argument, never checkpoint architecture: the
#: primary contrast is standard-versus-soft *in the same weights*, which is the
#: only comparison with exact parameter equality (`docs/directions.md` §3.3).
EVALUATION_MODES: tuple[str, ...] = ("standard", "soft")
DIAGNOSTIC_MODES: tuple[str, ...] = ("fused",)
RUNTIME_MODES: tuple[str, ...] = EVALUATION_MODES + DIAGNOSTIC_MODES

#: `SHAPES["square"]` from `dm.train`, restated rather than imported: the pilot
#: is defined by these three numbers, and a later edit to the sweep's shape table
#: must not be able to redefine what Direction 3 trained.
PILOT_SHAPE = "square"
PILOT_D_MODEL = 128
PILOT_N_LAYERS = 4
PILOT_N_HEADS = 4


def feedback_overhead(d_model: int) -> int:
    """Trainable parameters `glu_v1` adds: `2D^2 + D`.

    `W_U` and `W_G` are `D x D` and the dedicated RMSNorm contributes its own
    `D` gains.  The gain term is why this is `2D^2 + D` and not the `2D^2 =
    32,768` an earlier draft recorded: 128 parameters is arithmetically trivial
    and a parameter count that does not equal reality is not, because
    `Config.n_params()` is asserted against the live module.
    """
    return 2 * d_model**2 + d_model


def standard_params(representation: str, d_model: int = PILOT_D_MODEL) -> int:
    """Analytic parameter count of the no-feedback checkpoint for a codec.

    The trunk is held fixed across representations, so the two arms differ in the
    embedding table and in nothing else -- which is what makes the overhead a
    property of `d_model` alone and the same 32,896 in both.
    """
    return Config(vocab_size=CODECS[representation].vocab_size, d_model=d_model,
                  n_layers=PILOT_N_LAYERS, n_heads=PILOT_N_HEADS).n_params()


#: Codecs whose counts are worth stating, including the two the pilot excludes.
#: `token_typed` is over budget *with or without* feedback, and recording that
#: here is what stops a later stage from proposing it as a third arm and
#: discovering the ceiling after training.
COUNTED_REPRESENTATIONS: tuple[str, ...] = ("bit", "byte", "token", "token_typed")
PARAMETER_BUDGET = 1_000_000


def parameter_table(d_model: int = PILOT_D_MODEL) -> dict[str, dict]:
    """Standard, feedback-capable and overhead counts per representation."""
    out: dict[str, dict] = {}
    for name in COUNTED_REPRESENTATIONS:
        standard = standard_params(name, d_model)
        feedback = standard + feedback_overhead(d_model)
        out[name] = {
            "vocab_size": CODECS[name].vocab_size,
            "standard": standard,
            "feedback_capable": feedback,
            "overhead": feedback_overhead(d_model),
            "overhead_fraction": feedback / standard - 1.0,
            "under_budget": feedback < PARAMETER_BUDGET,
        }
    return out


# ---------------------------------------------------------------------------
# seeds

#: Every stream Direction 3 draws from, and what it governs.  Named streams
#: rather than one incremented seed: `prefix` and `jitter` are drawn per row per
#: pass, so sharing a generator would make the jitter values depend on how many
#: prefix draws happened first -- and then a change to the prefix rule would
#: silently move the jitter, which is the kind of coupling that makes two runs
#: incomparable for a reason nobody can find afterwards.
SEED_NAMESPACES: dict[str, str] = {
    "pass_plan": "how many passes each training batch gets",
    "prefix": "plain-prefix length per row per pass",
    "jitter": "carried-state jitter, Uniform[-0.02, 0.02]",
    "derangement": "model-blind donor derangement for the relation-destroyed corpus",
    "variates": "pre-generated uniforms for free-running promotion",
    "bootstrap": "connected-component resamples for every paired interval",
    "stability": "the fixed validation subset the recurrent stability gate uses",
}

#: Direction 2's `BUILDER_SEED`.  `bootstrap` is pinned to it rather than derived
#: so Direction 3's component resamples are *the same clusters in the same order*
#: Direction 2 published its `Delta` intervals over: `G` is a difference of two
#: `Delta`s on the same cases, and a different resample would make the two
#: stages' intervals incomparable for no reason at all.
INHERITED_BOOTSTRAP_SEED = 20260815

_PINNED_SEEDS: dict[str, int] = {"bootstrap": INHERITED_BOOTSTRAP_SEED}

#: Salted with the direction so a namespace name reused by a later direction
#: cannot land on the same stream.
SEED_ROOT = "direction3-feedback"


def seed_for(namespace: str) -> int:
    """The frozen 32-bit seed of one named stream.

    Derived from the name rather than chosen, so adding a stream cannot shift an
    existing one -- and the derived values are written into the protocol, so a
    *rename* is caught by the contract test instead of quietly producing a
    different corpus or a different set of uniforms.
    """
    if namespace not in SEED_NAMESPACES:
        raise KeyError(
            f"unknown seed namespace {namespace!r}; declare it in "
            f"SEED_NAMESPACES before drawing from it, so the protocol records it"
        )
    if namespace in _PINNED_SEEDS:
        return _PINNED_SEEDS[namespace]
    digest = hashlib.sha256(f"{SEED_ROOT}:{namespace}".encode()).hexdigest()
    return int(digest[:8], 16)


def seed_table() -> dict[str, int]:
    return {name: seed_for(name) for name in sorted(SEED_NAMESPACES)}


# ---------------------------------------------------------------------------
# training objective and schedule

#: `L_K = L_1 + (1/(K-1)) * sum(L_k, k=2..K)`, one weight for the whole feedback
#: term.  The mean over later passes rather than a sum: summing three
#: unnormalised losses makes the objective's scale a function of the schedule, so
#: a batch that happened to draw three passes would also get three times the
#: gradient magnitude.
OBJECTIVE = "L_K = L_1 + (1/(K-1)) * sum(L_k, k=2..K) for K>1; L_1 when K=1"
FEEDBACK_WEIGHT = 1.0
MAX_PASSES = 3

#: Carried states are perturbed by `Uniform[-h, h]` before fusion, drawn
#: independently per element.  Matches the fused-norm gain by construction: the
#: perturbation is the size of the signal's own initial scale.
JITTER_HALF_WIDTH = 0.02

#: No detach anywhere.  Later-pass losses backpropagate into the states earlier
#: passes produced, which is the only version of this objective that trains the
#: channel rather than a read-out of a frozen one.
DETACH_CARRIED_STATE = False

#: Memory guard, in rows x positions^2 x passes.  `K` multiplies the guard
#: because a two-pass batch runs the stack twice over the same width; reusing the
#: existing single-pass budget would silently triple peak memory at the end of
#: training, where three-pass batches live.
ATTENTION_BUDGET_RULE = "rows * T^2 * K <= attn_budget"


@dataclass(frozen=True)
class PassPhase:
    """One segment of the frozen pass schedule.

    `weights[k]` is the probability that a batch in this segment gets `k+1`
    passes.  Fractions of *training*, not step counts, so the schedule is
    identical under any budget -- and the pilot's budget is fixed at 24,000
    anyway, which is what makes the two readings agree here and would not
    elsewhere.
    """

    start: float
    stop: float
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        if not 0.0 <= self.start < self.stop <= 1.0:
            raise ValueError(f"phase bounds out of order: [{self.start}, {self.stop})")
        if not self.weights or len(self.weights) > MAX_PASSES:
            raise ValueError(
                f"a phase must weight 1..{MAX_PASSES} passes, got {len(self.weights)}"
            )
        if any(weight < 0.0 for weight in self.weights):
            raise ValueError("pass weights must be non-negative")
        if abs(sum(self.weights) - 1.0) > 1e-12:
            raise ValueError(f"pass weights sum to {sum(self.weights)}, not 1")

    @property
    def mean_passes(self) -> float:
        return sum((index + 1) * weight for index, weight in enumerate(self.weights))


#: The pilot schedule is a project adaptation, not a literal paper copy.  The
#: paper reports a 75/22/3 steady-state mixture and a progressive introduction,
#: but does not specify these exact 50%/25% phase boundaries.  Keeping the
#: project schedule frozen is still necessary; calling its provenance honestly
#: is what makes a later schedule comparison interpretable.
PASS_SCHEDULE: tuple[PassPhase, ...] = (
    PassPhase(0.00, 0.50, (1.0,)),
    PassPhase(0.50, 0.75, (0.75, 0.25)),
    PassPhase(0.75, 1.00, (0.75, 0.22, 0.03)),
)

#: Named pass schedules.  `project_progressive_v1` *is* v0's frozen
#: `PASS_SCHEDULE` under a name, not a copy of it: two objects that had to stay
#: equal by inspection would not.
PASS_SCHEDULES: dict[str, tuple[PassPhase, ...]] = {
    # The source's terminal mixture, introduced at the halfway point.
    "terminal_mix_v1": (
        PassPhase(0.00, 0.50, (1.0,)),
        PassPhase(0.50, 1.00, (0.75, 0.22, 0.03)),
    ),
    "project_progressive_v1": PASS_SCHEDULE,
    # The source's own diagnostic control: a 75/25 one-/two-pass mixture with no
    # three-pass batches, which the paper reports failing to extrapolate past the
    # trained depth. Diagnostic only -- it exists to show the probe can fail.
    "two_pass_control_v1": (
        PassPhase(0.00, 0.50, (1.0,)),
        PassPhase(0.50, 1.00, (0.75, 0.25)),
    ),
}
DEFAULT_PASS_SCHEDULE = "project_progressive_v1"

#: Eligible to freeze protocol v1, **in the order they are tried**.  Predeclared
#: here rather than chosen later: a schedule picked by comparing metric
#: magnitudes would make the qualification a selection over six cells and the
#: pilot its own selection data (`docs/directions.md` §7 invariant 13).
ELIGIBLE_SCHEDULES: tuple[str, ...] = ("terminal_mix_v1", "project_progressive_v1")
DIAGNOSTIC_SCHEDULES: tuple[str, ...] = ("two_pass_control_v1",)

#: What the source actually supplies for the pass mixture.  These values are
#: metadata for the next training freeze, not a replacement for the project's
#: executable schedule above.  In particular, ``phase_boundaries_specified`` is
#: false: sweeping the final three-pass fraction cannot turn the adaptation into
#: source-condition replication.
SOURCE_SCHEDULE: dict[str, object] = {
    "steady_state_pass_weights": (0.75, 0.22, 0.03),
    "progressive_introduction": True,
    "phase_boundaries_specified": False,
    "provenance": "paper_recipe_summary",
}

PROJECT_SCHEDULE_PROVENANCE = "project_adaptation"

#: Machine-readable source-condition ledger for the future training freeze. A
#: narrative caveat is easy to omit when a protocol is regenerated; these rows
#: make every material difference part of the artifact that carries the source
#: hashes. This is descriptive metadata, not a claim that the paper supplied a
#: complete tiny-scale recipe.
SOURCE_CONDITION_LEDGER: tuple[dict[str, str], ...] = (
    {
        "dimension": "pass_schedule",
        "source": "progressive introduction with terminal 75/22/3 mix",
        "project": "50%/25%/25% phases with terminal 75/22/3 mix",
        "status": "project_adaptation",
    },
    {
        "dimension": "optimizer_and_parameter_groups",
        "source": "paper optimizer and parameter-group recipe",
        "project": "project AdamW optimizer and named parameter groups",
        "status": "material_difference",
    },
    {
        "dimension": "lr_cooldown_z_loss",
        "source": "paper LR/cooldown/z-loss package",
        "project": "TrainConfig LR schedule; no source-equivalent cooldown/z-loss package",
        "status": "material_difference",
    },
    {
        "dimension": "scale_and_context",
        "source": "paper 1B-parameter, 200B-token scale and context",
        "project": "D=128, four layers, max_len=2048, finite pilot budget",
        "status": "material_difference",
    },
    {
        "dimension": "batching_unit",
        "source": "token-batched training",
        "project": "program-batched rows with a program-level pass plan",
        "status": "material_difference",
    },
    {
        "dimension": "gate_input_normalization",
        "source": "Listing 3 line 9 applies input_rmsnorm_1 to e before W_G",
        "project": "glu_source_v2 applies the same learned fuse.norm at the gate",
        "status": "listing_3_faithful",
    },
    {
        "dimension": "normalization_placement_and_affinity",
        "source": ("Listing 3 lines 9 and 11 name one module, input_rmsnorm_1, "
                   "applied to the gate input and to the post-mixin stack input; "
                   "line 10's prefix mixin sits between them"),
        "project": ("glu_source_v2 shares the learned fuse.norm across both "
                    "positions and mixes the plain prefix in before it, so a "
                    "plain position reaches the stack as N(e)"),
        "status": "listing_3_faithful",
    },
    {
        "dimension": "raw_embedding_paths",
        "source": ("Listing 3 line 5 runs pass 1 as model(e) and Listing 2 line 1 "
                   "prefills the same way"),
        "project": ("feedback=None is the only condition returning the raw "
                    "embedding: training pass 1 and the standard prefill"),
        "status": "listing_3_faithful",
    },
    {
        "dimension": "shared_input_norm_gain_calibration",
        "source": ("input_rmsnorm_1 is the model's own input norm, trained from "
                   "step 0 alongside the embedding it normalizes"),
        "project": ("a dedicated norm initialised at the embedding's step-0 RMS "
                    "whose channel switches on at the phase transition; under "
                    "gain_calibration='embedding_rms_at_switch_on_v1' it is set "
                    "once, at that transition, to the frequency-weighted RMS of "
                    "the embeddings it then meets"),
        "status": "project_adaptation",
    },
    {
        "dimension": "carried_state_jitter_scale",
        "source": "a fixed half-width, stated as 0.02 and not state-relative",
        "project": ("JITTER_HALF_WIDTH = 0.02, unchanged. v0's payload documents "
                    "it as matching 'the size of the signal's own initial scale', "
                    "and that rationale is wrong: the perturbed signal is the "
                    "carried state at RMS ~1.7, so the perturbation is ~1.1% "
                    "relative. The number is the source's and stays; only the "
                    "rationale is retracted"),
        "status": "source_value_project_rationale_retracted",
    },
    {
        "dimension": "superseded_source_arm",
        "source": "n/a",
        "project": ("glu_source_v1 stays loadable and is not qualified: "
                    "affine-free gate norm, a separate learned output gain and a "
                    "plain prefix that bypassed it"),
        "status": "retained_not_qualified",
    },
)


def phase_for(step: int, steps: int,
              schedule: str = DEFAULT_PASS_SCHEDULE) -> PassPhase:
    """The segment of a named schedule a step falls in.

    The final segment is closed at the top so the last step of training has a
    phase; every other boundary is half-open.  Off-by-one here would move 1 batch
    in 24,000, which is invisible in the accounting and visible in nothing --
    exactly the class of bug a frozen schedule is supposed to make impossible to
    introduce later.
    """
    if steps <= 0:
        raise ValueError("steps must be positive")
    if not 0 <= step <= steps:
        raise ValueError(f"step {step} outside [0, {steps}]")
    phases = pass_schedule(schedule)
    fraction = step / steps
    for phase in phases:
        if phase.start <= fraction < phase.stop:
            return phase
    return phases[-1]


def expected_passes_per_batch(schedule: str = DEFAULT_PASS_SCHEDULE) -> float:
    """Schedule-weighted mean passes per batch: `1.1325` for the pilot schedule.

    Reported because "same training tokens" is not "same compute": the feedback
    arms push ~13% more content-symbol forward passes through the stack than the
    step count suggests, and that belongs in the record rather than in a
    footnote.
    """
    return sum((phase.stop - phase.start) * phase.mean_passes
               for phase in pass_schedule(schedule))


# ---------------------------------------------------------------------------
# design matrix

REPRESENTATIONS: tuple[str, ...] = ("bit", "byte")
TRAINING_STRUCTURES: tuple[str, ...] = ("relational", "relation_destroyed")
MODEL_SEEDS: tuple[int, ...] = (0, 1)
FINAL_STEPS = 24_000

#: The eight checkpoints, in a fixed order so a report's rows cannot be permuted
#: between runs.  Eight and no more: the estimation pilot has no standard-trained
#: control arm, because the within-checkpoint mode contrast is both cleaner and
#: cheaper than a second training run whose weights differ (§5.2).
CELLS: tuple[tuple[str, str, int], ...] = tuple(
    (representation, structure, seed)
    for representation in REPRESENTATIONS
    for structure in TRAINING_STRUCTURES
    for seed in MODEL_SEEDS
)


def cell_name(representation: str, structure: str, seed: int) -> str:
    return f"feedback_{representation}_{structure}_s{seed}"


#: Direction 2's immutable step venue, and only that one.  The composed venue and
#: the axis diagnostics are outside the pilot: 21 components cap the composed
#: precision below what an interaction needs, and a diagnostic venue's null is
#: about the shift rather than the relation.
VENUE = "synthetic_flat_step"
VENUE_MANIFEST_SHA256 = (
    "dcc93caf0003f42ecb5b0ebcb23f29caf5a213942c754f1e7b4480ab0cd79b50"
)
VENUE_CASES = 64

#: The four-way blocks Direction 2's scorer already emits, and the count the
#: sequential scorer must reproduce request for request: 4 primary, 4
#: unrelated-target control, 4 unrelated-block control.
REQUESTS_PER_CASE = 12

# ---------------------------------------------------------------------------
# estimands and inference

ESTIMANDS: dict[str, str] = {
    "Delta": "Delta_i(mode) = D_primary_i(mode) - D_target_control_i(mode)",
    "G": "g_i(r,s,m) = Delta_i(soft) - Delta_i(standard)",
    "I_structure": "i_i(r,m) = g_i(r,relational,m) - g_i(r,relation_destroyed,m)",
    "I_representation": "b_i(m) = i_i(bit,m) - i_i(byte,m)",
}
ESTIMAND_UNIT = "bits per target byte"

#: Contrasts are formed *per case* and only then averaged.  Averaging first and
#: differencing after is the same number only when every cell has the same cases
#: with the same weights, and a single dropped case silently breaks that -- while
#: still producing a plausible table.
CONTRAST_ORDER = "per case, then component bootstrap"

ALPHA = 0.05
MULTIPLICITY = "holm"

#: The primary family: two `I_structure` tests, one per representation.  `G` is
#: a precondition rather than a family member, and `I_representation` is
#: exploratory, so neither enters the correction.
PRIMARY_FAMILY: tuple[str, ...] = tuple(f"I_structure[{r}]" for r in REPRESENTATIONS)

BOOTSTRAP_REPS = 2000
BOOTSTRAP_UNIT = "connected_source_donor_component"

#: Inherited from Direction 2 unchanged, which is the whole reason it can be
#: quoted here at all: same estimand, same unit, same venue, and the threshold
#: predates every Direction 3 number.  At the step venue's 15-byte target it is
#: 0.30 bits per drawing.
SESOI_BITS_PER_TARGET_BYTE = 0.02

#: Generic guards, all as (soft - standard) differences on the model's own
#: validation split.  They exist because a feedback gain bought by wrecking
#: ordinary likelihood or termination is not an accessibility result.
GENERIC_GUARDS: dict[str, float] = {
    "max_validation_cost_bits_per_drawing": 1.0,
    "min_valid_halt_delta": -0.05,
    "max_truncation_rate_increase": 0.0,
}

# ---------------------------------------------------------------------------
# stability (F7) and promotion (F8)

#: Repeated fused prefill depths.  Powers of two through 32 rather than 0..32:
#: a divergence that needs 32 iterations to show is a different failure from one
#: that shows at 2, and the two are told apart by the *shape* of the curve.
STABILITY_PASSES: tuple[int, ...] = (0, 1, 2, 4, 8, 16, 32)

#: Both fused-input and fused-hidden p99 RMS, each against its own reference: the
#: standard input RMS and the pass-0 hidden RMS.  A band rather than a bound,
#: because collapse toward zero is as much a broken channel as blow-up, and only
#: one of the two looks alarming in a loss curve.
STABILITY_RMS_BAND: tuple[float, float] = (0.25, 4.0)

STABILITY_GATE: dict[str, object] = {
    "all_finite": True,
    "rms_p99_band": list(STABILITY_RMS_BAND),
    "max_abs_logit": 100.0,
    "max_pass32_validation_increase_bits_per_drawing": 1.0,
    # Relative state update at q95, and additionally required not to exceed the
    # pass-8 value: a fixed point that is still moving at 32 has not converged
    # however small the step is.
    "max_update_q95": 0.25,
    "update_q95_monotone_from_pass": 8,
    "max_valid_halt_loss": 0.05,
}

# ---------------------------------------------------------------------------
# F5b qualification layer
#
# v0 froze a *contract*, and nothing below rewrites it.  Two of the clauses v0
# named were implemented against the wrong positions, one was named in units the
# bit arm does not have, and one turned out to be structurally unreachable
# (`docs/feedback-stability.md` §1).  Repairing an implementation of a frozen
# criterion is not moving a threshold; naming the repaired criteria in their own
# layer, with their own schema counter, is what keeps the two statements
# separable when an auditor reads the artifacts.

#: The qualification report's own shape.  Separate from `REPORT_SCHEMA`, which
#: v0 froze at 1 and which therefore cannot move to describe a report v0 did not
#: define.
QUALIFICATION_SCHEMA = 1

#: The stability report's shape.  Bumped from `REPORT_SCHEMA` because the F5b
#: repairs change what the fields *mean*: `finite` and `max_abs_logit` are now
#: scored-position quantities, the per-byte costs are semantic bytes rather than
#: symbols, and the rows carry the wavefront quantile the gate reads.  An old
#: report and a new one under one number would be silently incomparable.
STABILITY_REPORT_SCHEMA = 2

#: How many validation programs the recurrent probe runs on, and the length a
#: program must exceed to be eligible.
#:
#: Eligibility is `scored positions > deepest pass`: at pass 32 the wavefront is
#: the positions that can still move, and a row shorter than that contributes an
#: empty wavefront and a converged prefix that reads as convergence when it is
#: only the row's length.  The old smoke took the *first sixteen* validation
#: rows, twelve of which were fully converged at pass 32.
#:
#: Thirty-two rows: enough that no single program dominates a q95 taken over
#: positions, small enough that thirty-two fused prefills stay cheap next to a
#: 24,000-step cell.  The identities are persisted, so the choice is auditable
#: rather than merely deterministic.
STABILITY_SUBSET_SIZE = 32

#: Development seeds, data seed and scale.  Separate from the pilot's `0/1` model
#: seeds and `data_seed=0` so nothing the qualification sees can become
#: estimation data.
DEVELOPMENT_SEEDS: tuple[int, ...] = (100, 101)
DEVELOPMENT_DATA_SEED = 100
DEVELOPMENT_N_TRAIN = 100_000
DEVELOPMENT_N_VAL = 1_000

#: The repaired gate, over v0's thresholds.  Every number here is v0's or the
#: project's existing reporting guard; what F5b adds is *which positions* and
#: *which units* they are measured on, and which clause is withdrawn.
QUALIFICATION_GATE: dict[str, object] = {
    # completeness and resources
    "max_parameters": PARAMETER_BUDGET,
    "max_input_truncation": 0,
    "strict_reload_recomputed": True,
    "hashes_recomputed_at_freeze": True,
    # convergence: the first is the project's existing reporting guard, the
    # second is the frozen generic degradation budget read as a drift bound.
    "max_validation_tail_bits_per_drawing_per_1k": 0.5,
    "max_final_minus_best_bits_per_drawing": 1.0,
    # generic guards, quoted from v0 rather than restated
    **{name: value for name, value in GENERIC_GUARDS.items()},
    # repaired stability
    "stability_subset_size": STABILITY_SUBSET_SIZE,
    "stability_subset_seed_namespace": "stability",
    "scored_positions_only": True,
    "cost_unit": "bits_per_semantic_byte",
    "max_abs_logit": STABILITY_GATE["max_abs_logit"],
    "rms_p99_band": list(STABILITY_RMS_BAND),
    "max_pass32_validation_increase_bits_per_drawing":
        STABILITY_GATE["max_pass32_validation_increase_bits_per_drawing"],
    "max_update_q95_wavefront": STABILITY_GATE["max_update_q95"],
    "update_q95_wavefront_monotone_from_pass":
        STABILITY_GATE["update_q95_monotone_from_pass"],
    "max_valid_halt_loss": STABILITY_GATE["max_valid_halt_loss"],
    # what F5b withdraws and what it declines to gate on
    "withdrawn_clauses": ["update_q95_not_settling"],
    "diagnostic_only_clauses": ["fused_input_rms_band", "channel_scales"],
    "both_seeds_required": True,
    "retry_allowed": False,
    "selection_rule": "first predeclared eligible schedule passing both seeds",
}

#: Why `update_q95_not_settling` is withdrawn rather than merely unmet.  Position
#: 0 is plain, so `state_k[t]` is final for every `t < k` and the recurrence
#: reaches its fixed point in exactly `T` passes for *any* weights.  "Still
#: receding at the deepest pass" therefore cannot separate a bad channel from a
#: long sequence, and the clause has no reading under which it is informative.
WITHDRAWN_CLAUSE_REASONS: dict[str, str] = {
    "update_q95_not_settling": (
        "the fused prefill is triangular: state_k[t] is final for every t < k, "
        "so a deepest-pass update quantile reads sequence length rather than "
        "convergence. Replaced by the wavefront quantile, which is restricted "
        "to the positions that can still move"
    ),
}


def pass_schedule(name: str = DEFAULT_PASS_SCHEDULE) -> tuple[PassPhase, ...]:
    """One named schedule, refused rather than defaulted if it is unknown."""
    if name not in PASS_SCHEDULES:
        raise KeyError(
            f"unknown pass schedule {name!r}; expected one of "
            f"{sorted(PASS_SCHEDULES)}. Refused rather than defaulted: a typo "
            "would train a cell under the wrong schedule and record the right one"
        )
    return PASS_SCHEDULES[name]


#: Free-running promotion, run only after the teacher-forced gate passes under
#: the frozen rule.  Both floors are required: a preference gain with no
#: exact-hit gain is the Direction 2 outcome one stage on, and calling it
#: capability is the error Direction 2's C4 labels exist to prevent.
PROMOTION_GATE: dict[str, object] = {
    "draws_per_world_initial": 8,
    "draws_per_world_expanded": 64,
    "min_delta_gen_soft_minus_standard": 0.01,
    "min_hit_own_soft_minus_standard": 0.02,
    "interval_excludes_zero": True,
    "both_seeds_required": True,
    "structural_mask": "off",
    "shared_uniform_variates": True,
}

# ---------------------------------------------------------------------------
# labels

CLAIM_LABELS: dict[str, str] = {
    "incomplete": "a required artifact, freeze, convergence guard or report is missing "
                  "or malformed; never a null and never a pass",
    "unstable_feedback": "any recurrent stability criterion fails, so no outcome score "
                         "is computed at all",
    "no_feedback_gain": "complete and stable, and the primary gain does not pass",
    "teacher_forced_feedback_only": "the teacher-forced gate passes and the free-running "
                                    "promotion gate does not",
    "structure_specific_feedback": "H3-feedback and H3-structure pass in both named seeds "
                                   "with the generic guards intact",
    "exploratory_representation_interaction": "the preceding label, plus I_representation "
                                              "clearing its exploratory interval and floor "
                                              "in both seeds",
}

#: The rule, in order, and all of it required.  Written as data so the gate
#: Module can enumerate it and a report can state which clause failed rather
#: than returning a bare label.
DECISION_RULE: tuple[str, ...] = (
    "engineering, provenance, convergence and stability gates pass",
    "G_relational > 0 with a component interval excluding zero in both seeds",
    ("I_structure >= SESOI and Holm-corrected evidence passes in both seeds for "
     "at least one representation"),
    ("donor-target and unrelated-block controls pass, and the generic likelihood, "
     "validity and truncation guards hold"),
    ("promoted free generation clears both the preference and the exact-hit floor "
     "in both seeds"),
)

# ---------------------------------------------------------------------------
# F5c correction layer: the one post-hoc package, predeclared
#
# F5b stopped at `unstable_feedback`.  Its own development cells pointed at one
# mechanism -- the shared input norm's gain is calibrated to the embedding's RMS
# at *initialisation* while the channel switches on at the phase transition, by
# which time the embedding has grown about 12x -- and this layer is the single
# correction package that observation is allowed to buy.
#
# Three rules make it a package rather than a search, and all three live here
# rather than in prose.  **One lever moves**: the gain's calibration reference.
# The arm stays `glu_source_v2`, the jitter stays at the source's `0.02`, every
# threshold stays at the value v0 froze, and the schedule is a single predeclared
# one.  **Fresh data and fresh seeds**: the six F5b cells may motivate the design
# and may not qualify it, so nothing this package trains shares a byte or a
# stream with them.  **Three cells and no retry**: two unique seeds plus one
# repeat of the first, and the repeat exists to answer a question F5b could not
# (`docs/feedback-stability.md` §3d: one cell's stability verdict flipped
# between two runs of the identical configuration).


#: The correction package's identity, carried in every artifact it writes so a
#: reader never has to infer which package a cell belongs to from its seeds.
CORRECTION_PACKAGE = "f5c_gain_calibration_v1"

#: Fresh development seeds and a fresh development corpus.  Disjoint from the
#: pilot's `0/1` and from F5b's `100/101` on both axes: a package that reused
#: F5b's data would be selecting a calibration on the trajectories that motivated
#: it, which is the fault `docs/directions.md` §7 invariant 13 names.
CORRECTION_SEEDS: tuple[int, ...] = (200, 201)
CORRECTION_DATA_SEED = 200

#: The seed that runs twice, and how many times each seed runs.  The repeat is a
#: *reproducibility* control and not a second sample: two runs of one
#: configuration either produce the same checkpoint bytes or the two-seed rule is
#: partly measuring nondeterminism rather than seed sensitivity.
#:
#: **Historical defect, found after F5c closed.** Whole checkpoint bytes are not
#: a valid cross-run state comparator: ``torch.save`` ZIP members depend on the
#: output filename and development checkpoints embed a replicate-specific
#: ``record_name``. The frozen rule below is retained to reproduce the immutable
#: decision artifact and must not be reused. Compare
#: ``feedback_evidence.state_dict_sha256`` instead. The closure itself survives
#: because every F5c cell independently failed the generic and stability gates.
CORRECTION_REPEATED_SEED = 200
CORRECTION_REPLICATES: dict[int, int] = {200: 2, 201: 1}
CORRECTION_MAX_CELLS = sum(CORRECTION_REPLICATES.values())

#: One schedule, predeclared, with no comparison.  `project_progressive_v1` is
#: v0's own `PASS_SCHEDULE`; F5b's schedule ordering is a diagnostic direction
#: that explicitly cannot select anything (`docs/feedback-stability.md` §6.1), so
#: the choice here is the project's standing default rather than F5b's ranking.
CORRECTION_SCHEDULE = DEFAULT_PASS_SCHEDULE


# ---------------------------------------------------------------------------
# the one lever: how the shared input norm's gain is calibrated

_CALIBRATION_FORMULA = (
    "g = sqrt(sum_v p_v * mean_d E[v, d]^2), where E is the token embedding "
    "table at the feedback phase transition and p is the token frequency of the "
    "training split's *input* positions (BOS included, PAD excluded); the shared "
    "input norm's gain is then set to g in every component"
)

#: Named training-protocol conditions for the shared input norm's gain.  A
#: protocol condition rather than a fourth architecture arm: the forward equation
#: does not change, so a `glu_source_v3` sharing `glu_source_v2`'s tensor names
#: *and* its arithmetic would be a schema branch distinguishing nothing a reader
#: could check -- while adding a fourth name to the set of arms whose checkpoints
#: already load `strict=True` into one another and compute different functions.
#:
#: `none` is what every existing artifact trained under and stays the default, so
#: no record and no checkpoint changes its meaning.
GAIN_CALIBRATIONS: dict[str, str] = {
    "none": (
        "the shared input norm keeps its FUSED_NORM_GAIN initialisation for the "
        "whole run, so the channel switches on against an embedding that has "
        "grown since step 0"
    ),
    "embedding_rms_at_switch_on_v1": _CALIBRATION_FORMULA,
}
DEFAULT_GAIN_CALIBRATION = "none"
CORRECTION_GAIN_CALIBRATION = "embedding_rms_at_switch_on_v1"

#: Why this statistic and not another, in the artifact rather than in prose.
#: `Fusion.stack_input` returns `N(m)`, so a gain vector filled with `g` hands
#: the trunk rows at RMS exactly `g`; setting `g` to the frequency-weighted RMS
#: of the raw embeddings makes a fused pass's *plain prefix* the size of the
#: raw-embedding prefill it exists to imitate.  That is the stated rationale for
#: the `0.02` initialisation, evaluated at the moment the channel actually
#: switches on rather than at step 0.
#:
#: Frequency-weighted rather than a plain row mean because the trunk meets tokens
#: at their corpus frequency and not uniformly over the vocabulary.  PAD is
#: excluded: it is the tied head's negative class and never a content input.  BOS
#: is included: it is a real input position, and position 0 is always plain.
GAIN_CALIBRATION_RATIONALE: dict[str, object] = {
    "rule": CORRECTION_GAIN_CALIBRATION,
    "formula": _CALIBRATION_FORMULA,
    "applied_at": "the first step whose phase can draw more than one pass",
    "applied_once": True,
    "reads": ["the token embedding table",
              "the training split's input-position histogram"],
    "does_not_read": [
        "any loss, validation score, sample or checkpoint selection",
        "any Direction 2 case, Delta, G or relation outcome",
        "any threshold -- and no threshold moves with it",
    ],
    "pad_excluded": True,
    "bos_included": True,
    "initial_gain": FUSED_NORM_GAIN,
    "jitter_half_width_unchanged": JITTER_HALF_WIDTH,
}


def gain_calibration(name: str = DEFAULT_GAIN_CALIBRATION) -> str:
    """One named calibration rule, refused rather than defaulted if unknown.

    Refused for the reason `pass_schedule` is: a typo would train a cell under
    the default and record the name that was asked for, and this package is one
    lever whose position has to be legible from the artifact alone.
    """
    if name not in GAIN_CALIBRATIONS:
        raise KeyError(
            f"unknown gain calibration {name!r}; expected one of "
            f"{sorted(GAIN_CALIBRATIONS)}"
        )
    return GAIN_CALIBRATIONS[name]


def feedback_phase_transition(steps: int,
                              schedule: str = DEFAULT_PASS_SCHEDULE) -> int:
    """The first step at which a schedule admits a batch with more than one pass.

    Derived from the schedule rather than written down, so a package cannot
    calibrate at a step its schedule does not actually switch on at, and so the
    rule still reads correctly for a schedule that is multi-pass from step 0.
    The search runs over the same integers the training loop passes to
    `phase_for`, so the answer is the step the loop will really see.
    """
    if steps <= 0:
        raise ValueError("steps must be positive")
    for step in range(steps + 1):
        if len(phase_for(step, steps, schedule).weights) > 1:
            return step
    raise ValueError(
        f"{schedule!r} never draws more than one pass over {steps} steps, so it "
        "has no feedback phase transition to calibrate at"
    )


# ---------------------------------------------------------------------------
# what a package is, so a clause never has to know which one it is judging


@dataclass(frozen=True)
class Package:
    """The seeds, data and protocol conditions one qualification package fixes.

    `qualify_cell` applies the same clauses to an F5b development cell and to an
    F5c correction cell; what differs between them is *which* seeds, corpus and
    training-protocol condition are the right ones. Passing that as one frozen
    object keeps the difference in the contract, where it is declared before
    anything runs, rather than in the gate, where it would be an argument list
    nobody counts.
    """

    name: str
    seeds: tuple[int, ...]
    data_seed: int
    n_train: int
    n_val: int
    schedules: tuple[str, ...]
    gain_calibration: str
    replicates: int = 1
    #: Whether every cell of this package must have asked torch for
    #: deterministic kernels.  False for F5b, which did not and could not have
    #: known to; True for F5c, whose reproducibility clause is a statement about
    #: exactly that state.
    deterministic: bool = False

    def as_dict(self) -> dict:
        return {
            "package": self.name,
            "seeds": list(self.seeds),
            "data_seed": self.data_seed,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "schedules": list(self.schedules),
            "gain_calibration": self.gain_calibration,
            "replicates_per_seed": self.replicates,
            "deterministic_execution_required": self.deterministic,
        }


#: F5b, restated as a package so the instrument that judged it and the instrument
#: that judges F5c are one function.  It ran and stopped; nothing here re-opens
#: it, and the six recorded reports are never rewritten.
F5B_PACKAGE = Package(
    name="f5b_schedule_qualification",
    seeds=DEVELOPMENT_SEEDS,
    data_seed=DEVELOPMENT_DATA_SEED,
    n_train=DEVELOPMENT_N_TRAIN,
    n_val=DEVELOPMENT_N_VAL,
    schedules=ELIGIBLE_SCHEDULES + DIAGNOSTIC_SCHEDULES,
    gain_calibration=DEFAULT_GAIN_CALIBRATION,
)

F5C_PACKAGE = Package(
    name=CORRECTION_PACKAGE,
    seeds=CORRECTION_SEEDS,
    data_seed=CORRECTION_DATA_SEED,
    n_train=DEVELOPMENT_N_TRAIN,
    n_val=DEVELOPMENT_N_VAL,
    schedules=(CORRECTION_SCHEDULE,),
    gain_calibration=CORRECTION_GAIN_CALIBRATION,
    replicates=max(CORRECTION_REPLICATES.values()),
    deterministic=True,
)

PACKAGES: dict[str, Package] = {"f5b": F5B_PACKAGE, "f5c": F5C_PACKAGE}


def package(name: str) -> Package:
    """One named package, refused rather than defaulted if it is unknown."""
    if name not in PACKAGES:
        raise KeyError(
            f"unknown qualification package {name!r}; expected one of "
            f"{sorted(PACKAGES)}"
        )
    return PACKAGES[name]


# ---------------------------------------------------------------------------
# execution environment, recorded because a reproducibility claim is about one


#: What a cell must record about the machine that produced it.  F5b recorded none
#: of it, so `docs/feedback-stability.md` §3d's nondeterminism observation cannot
#: be attributed to a device, a threading choice or a determinism setting -- and
#: F5c's reproducibility clause is a statement about exactly those.  A field
#: present and null is a value; an absent field is `incomplete`.
REQUIRED_ENVIRONMENT_FIELDS: tuple[str, ...] = (
    "python",
    "python_implementation",
    "torch",
    "platform",
    "machine",
    "device",
    "deterministic_algorithms",
    "deterministic_algorithms_warn_only",
    "float32_matmul_precision",
    "cudnn_deterministic",
    "cudnn_benchmark",
    "cuda_available",
    "mps_available",
    "torch_num_threads",
    "torch_num_interop_threads",
    "environment_variables",
)

#: The environment variables that can change a result without changing a line of
#: code.  Recorded by name with a null when unset, so "unset" and "never looked
#: at" stay distinguishable.
RECORDED_ENVIRONMENT_VARIABLES: tuple[str, ...] = (
    "PYTHONHASHSEED",
    "CUBLAS_WORKSPACE_CONFIG",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "PYTORCH_ENABLE_MPS_FALLBACK",
    "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
)


# ---------------------------------------------------------------------------
# the correction gate


#: What the historical correction package required on top of every F5b clause.
#: Every threshold below is v0's or the project's existing reporting guard,
#: quoted rather than restated. The whole-file reproducibility entry is preserved
#: as frozen history and is invalid for future use; see the historical-defect
#: note on ``CORRECTION_REPEATED_SEED`` above.
CORRECTION_GATE: dict[str, object] = {
    "package": CORRECTION_PACKAGE,
    "schedule": CORRECTION_SCHEDULE,
    "schedule_comparison": False,
    "seeds": list(CORRECTION_SEEDS),
    "data_seed": CORRECTION_DATA_SEED,
    "repeated_seed": CORRECTION_REPEATED_SEED,
    "replicates": {str(seed): count
                   for seed, count in sorted(CORRECTION_REPLICATES.items())},
    "max_cells": CORRECTION_MAX_CELLS,
    "feedback_schema": QUALIFICATION_FEEDBACK_SCHEMA,
    "gain_calibration": CORRECTION_GAIN_CALIBRATION,
    "jitter_half_width": JITTER_HALF_WIDTH,
    "thresholds_moved": [],
    "reproducibility": "the repeated seed's two cells share one checkpoint SHA-256",
    "deterministic_execution_required": True,
    "retry_allowed": False,
    "selection_rule": (
        "one predeclared schedule; both unique seeds must pass every generic and "
        "stability clause and the repeated seed must reproduce checkpoint bytes"
    ),
    "terminal_rule": (
        "any reproducibility failure or any gate failure closes Direction 3 "
        "permanently at this scale; all three cells passing freezes a new "
        "training protocol and only then authorizes F6; no further gain choices, "
        "jitter variants, schedules or retries"
    ),
}

#: The labels `qualify_correction` may return, and only these.  `incomplete` and
#: the two stop labels are quoted from v0; `incomplete_nondeterministic` is new
#: because "the same configuration produced two different checkpoints" is not a
#: statement about the mechanism and must not be filed as one.
CORRECTION_LABELS: dict[str, str] = {
    "qualified": (
        "every cell passed every clause and the repeated seed reproduced its "
        "checkpoint bytes; a new training protocol may be frozen"
    ),
    "incomplete": CLAIM_LABELS["incomplete"],
    "incomplete_nondeterministic": (
        "the repeated seed did not reproduce its checkpoint bytes under "
        "deterministic execution, so the package measured nondeterminism rather "
        "than the mechanism: move to a deterministic device, or stop"
    ),
    "unstable_feedback": CLAIM_LABELS["unstable_feedback"],
    "no_viable_feedback_implementation_at_scale": (
        "no stability clause failed and not every cell passed; bounded to this "
        "scale and this package"
    ),
}


def correction_cell_name(seed: int, replicate: int,
                         schedule: str = CORRECTION_SCHEDULE) -> str:
    """The tag one correction cell writes its record and checkpoint under.

    The replicate is in the name because the repeated seed's two cells differ in
    nothing else, and two artifacts differing in nothing else would otherwise
    overwrite each other -- which is how the first `terminal_mix_v1 / seed 100`
    artifact stopped existing (`docs/feedback-stability.md` §3d).
    """
    return f"feedback_f5c_{schedule}_s{seed}_r{replicate}"


def training_config_correction(seed: int, replicate: int, *,
                               codec: str = "byte") -> dict:
    """The exact `TrainConfig` keyword arguments for one F5c correction cell.

    Everything F5b's development configuration froze, with two changes and no
    others: the fresh seed and data seed, and the named gain calibration. The
    schedule is not a parameter -- the package declares one and compares none.
    """
    if seed not in CORRECTION_SEEDS:
        raise ValueError(
            f"{seed} is not a correction seed; expected one of "
            f"{list(CORRECTION_SEEDS)}. F5b's development seeds motivated this "
            "package and cannot qualify it"
        )
    allowed = CORRECTION_REPLICATES[seed]
    if not 1 <= replicate <= allowed:
        raise ValueError(
            f"seed {seed} runs {allowed} time(s) in this package, so replicate "
            f"{replicate} is outside it: three cells and no retry"
        )
    if codec not in REPRESENTATIONS:
        raise ValueError(f"unknown representation {codec!r}")
    return {
        **TRAINING_DEFAULTS,
        "codec": codec,
        "seed": seed,
        "data_seed": CORRECTION_DATA_SEED,
        "n_train": DEVELOPMENT_N_TRAIN,
        "n_val": DEVELOPMENT_N_VAL,
        "pass_schedule": CORRECTION_SCHEDULE,
        "gain_calibration": CORRECTION_GAIN_CALIBRATION,
        "extra": {"structure": "relational"},
        "tag": correction_cell_name(seed, replicate),
    }


# ---------------------------------------------------------------------------
# record and report schemas

#: Accounting a feedback training record must carry, each separately.  "Same
#: training tokens" is not "same compute" and not "same data": the bit arm sees
#: eight symbols per semantic byte, the schedule adds ~13% forward passes, and
#: padding is neither. A record that collapses these cannot support any
#: efficiency statement, so the fields are required rather than encouraged.
REQUIRED_RECORD_FIELDS: tuple[str, ...] = (
    "schema",
    "complete",
    "steps_done",
    "programs_seen",
    "semantic_bytes_seen",
    "content_symbols_seen",
    "padded_positions",
    "content_symbol_forward_passes",
    "padded_forward_positions",
    "pass_histogram",
    "expected_passes_per_batch",
    "observed_passes_per_batch",
    "peak_memory_bytes",
    "wall_clock_s",
    "device",
    "feedback_schema",
    "corpus",
    "protocol",
)

#: What a teacher-forced scoring report must publish.  Raw NLLs first: a saved
#: contrast cannot be re-derived into its parts, and every failure mode this
#: pilot can have is diagnosed from the parts (`dm/eval/context.py`).
REQUIRED_SCORE_FIELDS: tuple[str, ...] = (
    "schema",
    "status",
    "protocol",
    "mode",
    "representation",
    "training_structure",
    "model_seed",
    "raw_symbol_nll_bits",
    "nll_bits",
    "D",
    "D_target_control",
    "D_block_control",
    "Delta",
    "first_target_symbol_identical",
    "fields",
    "prefix_nll_bits",
    "sequential_standard_matches_full_forward",
)

#: What the promoted free-running report must publish.  `reach_rate` and the
#: termination categories are required beside every accuracy number for the same
#: reason Direction 2 required them: a completion that never reached the target
#: has to stay in the denominator, visibly.
REQUIRED_GENERATION_FIELDS: tuple[str, ...] = (
    "schema",
    "status",
    "protocol",
    "mode",
    "reach_rate",
    "termination",
    "hit_own_rate",
    "hit_other_rate",
    "longest_compatible_prefix_frac",
    "relation_consistent_rate",
    "D_gen",
    "D_gen_target_control",
    "D_gen_block_control",
    "Delta_gen",
    "draws_per_world",
    "variate_seed",
)

# ---------------------------------------------------------------------------
# protocol payload

QUESTION = (
    "Does returning the previous normalized top-layer state through gated fusion "
    "improve controlled use of separated relational context beyond standard "
    "decoding in the same weights, and is the gain specific to relational "
    "training rather than to generic recurrence?"
)

#: Stated in the protocol because it is the claim most likely to be overstated
#: from a positive result.  Previous top state is a deterministic function of
#: visible history; the mechanism can only make an available relation cheaper to
#: use.
MECHANISM_LIMIT = (
    "accessibility, not information: the carried state adds nothing the visible "
    "history does not already determine, so a gain is never new knowledge, "
    "reasoning, a mutable register or greater asymptotic depth"
)

FREEZES: tuple[str, ...] = (
    "baseline CPU fixture bytes",
    "feedback schema names and the default",
    "fusion equation, carried state and fused-norm gain",
    "seed namespaces and their derived values",
    "training objective, pass schedule and jitter support",
    "design matrix, venue manifest and case count",
    "estimands, contrast order, SESOI, alpha and multiplicity",
    "generic guards, stability gate and promotion floors",
    "claim labels, decision rule and required record/report fields",
    "expected parameter counts",
)

DOES_NOT_FREEZE: tuple[str, ...] = (
    ("source digests -- F1 changes the model by design; protocol v1 freezes them "
     "after the F5 smoke"),
    "training corpus fingerprints -- the paired corpora do not exist until F4",
    ("checkpoint and record hashes -- no checkpoint exists until F6; protocol v2 "
     "freezes them before any outcome score"),
)


def protocol_dict(*, fixture_sha256: str) -> dict:
    """The canonical v0 payload, content-addressed like a Direction 2 manifest."""
    body = {
        "protocol": PROTOCOL,
        "schema": PROTOCOL_SCHEMA,
        "status": "frozen",
        "scope": "contract",
        "direction": DIRECTION,
        "question": QUESTION,
        "mechanism_limit": MECHANISM_LIMIT,
        "freezes": list(FREEZES),
        "does_not_freeze": list(DOES_NOT_FREEZE),
        "supersedes": None,
        "superseded_by": {
            "feedback-v1": "training freeze, after the F5 smoke",
            "feedback-v2": "checkpoint and record freeze, before any outcome score",
        },
        "architecture": {
            "feedback_schemas": list(FEEDBACK_SCHEMAS),
            "default_feedback_schema": DEFAULT_FEEDBACK_SCHEMA,
            "config_field": CONFIG_FIELD,
            "runtime_mode_argument": RUNTIME_MODE_ARGUMENT,
            "feedback_parameters": list(FEEDBACK_PARAMETERS),
            "interface": dict(MODEL_INTERFACE),
            "fusion": FUSION_EQUATION,
            "carried_state": CARRIED_STATE,
            "fused_norm_gain": FUSED_NORM_GAIN,
            "additive_embedding_shortcut": False,
            "evaluation_modes": list(EVALUATION_MODES),
            "diagnostic_modes": list(DIAGNOSTIC_MODES),
            "mode_is_runtime_argument": True,
            "latent_scope": "request-local, row-aligned, cleared at completion",
        },
        "parameters": {
            "shape": PILOT_SHAPE,
            "d_model": PILOT_D_MODEL,
            "trunk": {"n_layers": PILOT_N_LAYERS, "n_heads": PILOT_N_HEADS},
            "overhead_formula": "2 * d_model**2 + d_model",
            "overhead": feedback_overhead(PILOT_D_MODEL),
            "budget": PARAMETER_BUDGET,
            "table": parameter_table(),
        },
        "seeds": {
            "root": SEED_ROOT,
            "namespaces": dict(sorted(SEED_NAMESPACES.items())),
            "values": seed_table(),
            "pinned": dict(sorted(_PINNED_SEEDS.items())),
            "micro_batch_invariant": True,
        },
        "training": {
            "objective": OBJECTIVE,
            "feedback_weight": FEEDBACK_WEIGHT,
            "max_passes": MAX_PASSES,
            "detach_carried_state": DETACH_CARRIED_STATE,
            "jitter_half_width": JITTER_HALF_WIDTH,
            "attention_budget_rule": ATTENTION_BUDGET_RULE,
            "schedule": [
                {"start": phase.start, "stop": phase.stop,
                 "weights": list(phase.weights)}
                for phase in PASS_SCHEDULE
            ],
            "expected_passes_per_batch": expected_passes_per_batch(),
            "final_steps": FINAL_STEPS,
        },
        "matrix": {
            "representations": list(REPRESENTATIONS),
            "training_structures": list(TRAINING_STRUCTURES),
            "model_seeds": list(MODEL_SEEDS),
            "cells": [cell_name(*cell) for cell in CELLS],
            "venue": VENUE,
            "venue_manifest_sha256": VENUE_MANIFEST_SHA256,
            "venue_cases": VENUE_CASES,
            "requests_per_case": REQUESTS_PER_CASE,
        },
        "inference": {
            "estimands": dict(ESTIMANDS),
            "unit": ESTIMAND_UNIT,
            "contrast_order": CONTRAST_ORDER,
            "alpha": ALPHA,
            "multiplicity": MULTIPLICITY,
            "primary_family": list(PRIMARY_FAMILY),
            "bootstrap": {"replicates": BOOTSTRAP_REPS, "unit": BOOTSTRAP_UNIT,
                          "seed": seed_for("bootstrap"),
                          "shared_draws_across_cells": True},
            "sesoi_bits_per_target_byte": SESOI_BITS_PER_TARGET_BYTE,
            "seeds_pooled": False,
            "generic_guards": dict(GENERIC_GUARDS),
            "representation_interaction_status": "exploratory",
        },
        "stability": {"passes": list(STABILITY_PASSES), **STABILITY_GATE},
        "promotion": dict(PROMOTION_GATE),
        "labels": dict(CLAIM_LABELS),
        "decision_rule": list(DECISION_RULE),
        "schemas": {
            "fixture": FIXTURE_SCHEMA,
            "record": RECORD_SCHEMA,
            "report": REPORT_SCHEMA,
            "required_record_fields": list(REQUIRED_RECORD_FIELDS),
            "required_score_fields": list(REQUIRED_SCORE_FIELDS),
            "required_generation_fields": list(REQUIRED_GENERATION_FIELDS),
        },
        "baseline_fixture": {
            "path": FIXTURE_PATH_STR,
            "sha256": fixture_sha256,
            "schema": FIXTURE_SCHEMA,
        },
        "next_stage": {
            "stage": "F1",
            "command": "PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_feedback.py",
            "expectation": "every test in that file is a strict xfail until the "
                           "feedback path exists; F1 is done when they pass with "
                           "the markers removed",
        },
    }
    body["protocol_sha256"] = digest_of(body)
    return body


# ---------------------------------------------------------------------------
# protocol v1: the training freeze


PROTOCOL_V1 = "feedback-v1"
PROTOCOL_V1_PATH = Path("docs/feedback-protocol-v1.json")

#: Files whose bytes determine what a training run *is*. Frozen at F5, after the
#: smoke has exercised them, and checked before F6 launches. v0 could not freeze
#: these because F1 through F4 change them by design; that is the whole reason
#: the training freeze is a separate protocol.
TRAINING_SOURCES: tuple[str, ...] = (
    "dm/models/transformer.py",
    "dm/train.py",
    "dm/train_feedback.py",
    "dm/data/feedback.py",
    "dm/data/synthetic.py",
    "dm/data/dataset.py",
    "dm/isa/codec.py",
    "dm/isa/unroll.py",
    "dm/eval/feedback_contract.py",
)

#: Everything the *qualification* additionally depends on.  A separate tuple
#: because the two answer different questions -- "what trained the cell" and
#: "what judged it" -- and a freeze that hashed only the first could apply a
#: repaired gate under a record claiming the unrepaired one.
QUALIFICATION_SOURCES: tuple[str, ...] = (
    *TRAINING_SOURCES,
    "dm/data/feedback_audit.py",
    "dm/eval/feedback.py",
    "dm/eval/feedback_qualification.py",
)

#: Optimiser settings, taken unchanged from `dm.train.TrainConfig`'s defaults.
#:
#: **Restated rather than inherited**, so a later edit to those defaults cannot
#: retroactively change what the pilot trained, and so the frozen protocol names
#: numbers rather than a reference to numbers. They are the project's existing
#: defaults and not a search: Direction 3 shares Direction 2's venue and corpus
#: family, and picking new hyperparameters here would add a factor nobody
#: measured.
TRAINING_DEFAULTS: dict[str, object] = {
    "shape": PILOT_SHAPE,
    "data": "feedback",
    "n_train": 100_000,
    "n_val": 1_000,
    "max_len": 2048,
    "batch_size": 64,
    "steps": FINAL_STEPS,
    "attn_budget": 24_000_000,
    "lr": 3e-3,
    "warmup": 200,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "data_seed": 0,
    # Twelve evals over the budget. Every eval writes a whole record, which is
    # the property that made a killed schema-2 run recoverable and its absence
    # cost 2.6 hours once (`docs/history/lost-run.md`).
    "eval_every": 2_000,
    "gen_samples": 128,
    "gen_cap": 2.0,
    # Common random numbers within a seed, across representation and structure.
    # Without it the arms of a paired comparison start from unrelated networks,
    # which on Tier B was worth ~2 bits/drawing -- larger than any axis here.
    "share_init": True,
    "feedback_schema": QUALIFICATION_FEEDBACK_SCHEMA,
    "gain_calibration": DEFAULT_GAIN_CALIBRATION,
}


def training_config(representation: str, structure: str, seed: int,
                    schedule: str = DEFAULT_PASS_SCHEDULE,
                    calibration: str = DEFAULT_GAIN_CALIBRATION) -> dict:
    """The exact `TrainConfig` keyword arguments for one of the eight cells.

    F6 is a loop over `CELLS` and this function; there is no free choice left in
    it. A cell that could be launched with different settings from its paired
    twin would make the pair a comparison of two experiments.

    `schedule` is the one F5b qualified, and it is a parameter rather than a
    constant because which schedule that is cannot be known until six
    development cells have run. `protocol_v1_dict` supplies it from the
    qualification result, so the eight estimation cells train under the schedule
    that was actually qualified rather than under whichever one happened to be
    the default.
    """
    if representation not in REPRESENTATIONS:
        raise ValueError(f"unknown representation {representation!r}")
    if structure not in TRAINING_STRUCTURES:
        raise ValueError(f"unknown training structure {structure!r}")
    if seed not in MODEL_SEEDS:
        raise ValueError(f"unknown model seed {seed!r}")
    pass_schedule(schedule)
    gain_calibration(calibration)
    return {
        **TRAINING_DEFAULTS,
        "codec": representation,
        "seed": seed,
        "pass_schedule": schedule,
        "gain_calibration": calibration,
        "extra": {"structure": structure},
        "tag": cell_name(representation, structure, seed),
    }


def training_config_development(schedule: str, seed: int, *,
                                codec: str = "byte") -> dict:
    """The exact `TrainConfig` keyword arguments for one F5b qualification cell.

    Full budget and full scale, because that is the question: a compressed ladder
    qualifies a compressed trajectory and the F5 ladder already showed how little
    a short run says about the final one. What is *not* the pilot's is everything
    that could leak: the model seed, the data seed and therefore the corpus
    itself. Development cells and estimation cells must not share bytes, or the
    schedule was chosen on data the pilot is later scored on
    (`docs/directions.md` §7 invariant 13).

    Relational only. The relation-destroyed arm exists to answer `I_structure`,
    which qualification is not allowed to look at; training it here would produce
    a checkpoint whose only use is the one this stage is forbidden.
    """
    pass_schedule(schedule)
    if seed not in DEVELOPMENT_SEEDS:
        raise ValueError(
            f"{seed} is not a development seed; expected one of "
            f"{list(DEVELOPMENT_SEEDS)}. The pilot's own seeds cannot qualify "
            "the schedule they will be trained under"
        )
    if codec not in REPRESENTATIONS:
        raise ValueError(f"unknown representation {codec!r}")
    return {
        **TRAINING_DEFAULTS,
        "codec": codec,
        "seed": seed,
        "data_seed": DEVELOPMENT_DATA_SEED,
        "n_train": DEVELOPMENT_N_TRAIN,
        "n_val": DEVELOPMENT_N_VAL,
        "pass_schedule": schedule,
        # `structure` and nothing else. `dm.train.build_data` pops that key and
        # forwards the rest of `extra` to the corpus builder and from there to
        # the sampler, so `extra` is a corpus-parameter channel and a
        # descriptive key put here for a reader is a `TypeError` at the first
        # cell. The stage is already legible from `artifact_provenance`,
        # `pass_schedule` and the tag.
        "extra": {"structure": "relational"},
        "tag": development_cell_name(schedule, seed),
    }


def development_cell_name(schedule: str, seed: int) -> str:
    """The tag one qualification cell writes its record and checkpoint under."""
    return f"feedback_qual_{schedule}_s{seed}"


def protocol_v1_dict(*, base: dict, source_sha256: dict[str, str],
                     corpus: dict, smoke: dict,
                     qualification: dict | None = None) -> dict:
    """The training freeze, layered on v0 by digest rather than by copy.

    v0 stays the authority on everything it froze and v1 adds only what could not
    exist before code did: the source digests, the corpus fingerprints and the
    per-cell training configuration. Recording v0's digest rather than restating
    its contents means the two cannot drift apart and an auditor reads one chain
    rather than two overlapping documents.
    """
    if base.get("protocol") != PROTOCOL:
        raise ValueError(f"v1 must extend {PROTOCOL}, got {base.get('protocol')!r}")
    if qualification is not None and not qualification.get("frozen_schedule"):
        # A qualification that named no schedule is a stop, and a stop cannot
        # freeze a training protocol (`docs/directions.md` §7 invariant 10).
        raise ValueError(
            "the qualification froze no schedule, so there is nothing to train "
            f"under: label={qualification.get('label')!r}: incomplete"
        )
    if qualification is not None and (
            qualification.get("frozen_schedule") not in ELIGIBLE_SCHEDULES):
        raise ValueError(
            f"{qualification.get('frozen_schedule')!r} is not an eligible "
            f"schedule; the diagnostic control cannot authorize v1"
        )
    qualified = (DEFAULT_PASS_SCHEDULE if qualification is None
                 else qualification["frozen_schedule"])
    # The calibration is read off the qualification's own package rather than
    # chosen here: v1 must freeze the condition the qualifying cells actually
    # trained under, and the freeze has no standing to pick a different one.
    qualified_calibration = ((qualification or {}).get("package") or {}).get(
        "gain_calibration", DEFAULT_GAIN_CALIBRATION)
    gain_calibration(qualified_calibration)
    body = {
        "protocol": PROTOCOL_V1,
        "schema": PROTOCOL_SCHEMA,
        "status": "frozen",
        "scope": "training",
        "direction": DIRECTION,
        "supersedes": PROTOCOL,
        "base_protocol_sha256": base["protocol_sha256"],
        "freezes": [
            "source digests of every file that determines a training run",
            "source digests of every file that determines the qualification",
            "relational and relation-destroyed corpus fingerprints",
            "the corpus manifest and its donor map, under both hash names",
            "the qualified pass schedule and the cells that qualified it",
            "the repaired qualification gate and the clause it withdraws",
            "per-cell training configuration for all eight cells",
        ],
        "does_not_freeze": [
            ("pilot checkpoint and record hashes -- engineering smoke hashes do "
             "not identify pilot cells; protocol v2 freezes pilot artifacts "
             "before any outcome score"),
        ],
        "architecture": {
            "feedback_schema": QUALIFICATION_FEEDBACK_SCHEMA,
            "fusion": SOURCE_V2_FUSION_EQUATION,
            "normalization": dict(SOURCE_V2_NORMALIZATION),
            "seam": "listing_3_faithful",
            "superseded_arms": {
                "glu_v1": "legacy compatibility; raw gate input",
                SOURCE_FEEDBACK_SCHEMA: (
                    "source-aligned; affine-free gate norm, separate learned "
                    "output gain, raw plain prefix"
                ),
            },
        },
        "schedule_provenance": {
            "kind": PROJECT_SCHEDULE_PROVENANCE,
            "source": {
                **SOURCE_SCHEDULE,
                "steady_state_pass_weights": list(
                    SOURCE_SCHEDULE["steady_state_pass_weights"]
                ),
            },
            "qualified_schedule": (None if qualification is None
                                   else qualification["frozen_schedule"]),
            "project_phases": [
                {"start": phase.start, "stop": phase.stop,
                 "weights": list(phase.weights)}
                for phase in pass_schedule(qualified)
            ],
            "source_conditions_complete": False,
            "note": (
                "The paper reports a steady-state mixture and progressive "
                "introduction, not these exact project phase boundaries."
            ),
        },
        "qualification": qualification,
        "qualification_schema": QUALIFICATION_SCHEMA,
        "qualification_gate": dict(QUALIFICATION_GATE),
        # The training-protocol condition the qualified cells trained under, and
        # the package they belong to. Both, because the arm alone no longer says
        # what a checkpoint is: `glu_source_v2` under `gain_calibration="none"`
        # and under `embedding_rms_at_switch_on_v1` compute the same forward
        # equation from differently calibrated weights, and only the record says
        # which.
        "gain_calibration": {
            "rule": qualified_calibration,
            "meaning": GAIN_CALIBRATIONS[qualified_calibration],
            "rationale": (dict(GAIN_CALIBRATION_RATIONALE)
                          if qualified_calibration != DEFAULT_GAIN_CALIBRATION
                          else None),
            "available": dict(GAIN_CALIBRATIONS),
        },
        "correction_gate": (dict(CORRECTION_GATE)
                            if qualified_calibration == CORRECTION_GAIN_CALIBRATION
                            else None),
        "withdrawn_clauses": dict(WITHDRAWN_CLAUSE_REASONS),
        "stability_report_schema": STABILITY_REPORT_SCHEMA,
        "source_condition_ledger": [dict(row) for row in SOURCE_CONDITION_LEDGER],
        "source_conditions_complete": False,
        "source_sha256": dict(sorted(source_sha256.items())),
        "corpus": corpus,
        "cells": {cell_name(*cell): training_config(
                      *cell, schedule=qualified, calibration=qualified_calibration)
                  for cell in CELLS},
        "training_defaults": dict(TRAINING_DEFAULTS),
        # The smoke run's identity, recorded so it is traceable, and its status,
        # recorded so it can never be mistaken for evidence. One engineering cell
        # has zero scientific decision value.
        "smoke": {**smoke, "provenance": "engineering",
                  "decision_value": "none"},
    }
    body["protocol_sha256"] = digest_of(body)
    return body


def digest_of(body: dict) -> str:
    """SHA-256 over the canonical JSON body, excluding any recorded digest."""
    payload = {key: value for key, value in body.items() if key != "protocol_sha256"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_protocol(path: Path, protocol: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = staged_path(path)
    staged.write_text(json.dumps(protocol, indent=1, sort_keys=True) + "\n")
    staged.replace(path)


def load_protocol(path: Path = PROTOCOL_PATH, *, expect: str = PROTOCOL) -> dict:
    """Read a protocol, refusing a payload whose digest, schema or status disagrees.

    Fail closed on all three.  A protocol that can be edited without its digest
    moving is not a freeze, and a stage that reads an unfrozen protocol has no
    provenance to report -- which `docs/directions.md` §7 invariant 11 resolves
    as `incomplete`, never as a result.
    """
    protocol = json.loads(path.read_text())
    recorded = protocol.get("protocol_sha256")
    actual = digest_of(protocol)
    if recorded != actual:
        raise ValueError(
            f"feedback protocol hash mismatch for {path}: recorded {recorded!r}, "
            f"computed {actual!r}"
        )
    if protocol.get("protocol") != expect:
        raise ValueError(
            f"{path} is protocol {protocol.get('protocol')!r}, expected {expect!r}"
        )
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError(
            f"unsupported feedback protocol schema {protocol.get('schema')!r}; "
            f"expected {PROTOCOL_SCHEMA}"
        )
    if protocol.get("status") != "frozen":
        raise ValueError(f"feedback protocol is not frozen: {path}")
    return protocol


def missing_record_fields(record: dict) -> list[str]:
    """Which required accounting fields a training record does not carry.

    Looks in the record and in its `feedback` block, because the two hold
    different halves of the same statement: `complete`, `steps_done` and `corpus`
    belong to any run, and the pass and symbol counters belong to a feedback run.

    **A field present and null is not missing.** `peak_memory_bytes` is `None` on
    CPU because there is no allocator high-water mark to read, and "not
    measurable here" has to be distinguishable from "nobody recorded it" -- the
    fail-closed rule treats them very differently (`docs/directions.md` §7
    invariant 11).
    """
    present = set(record) | set(record.get("feedback") or {})
    return [field for field in REQUIRED_RECORD_FIELDS if field not in present]


def check_consistency() -> None:
    """Assert the invariants that make the contract self-consistent.

    Called by the writer *and* by the test, so a hand edit to the JSON cannot
    survive and a change here cannot ship without the JSON being rewritten.
    """
    if not DEFAULT_FEEDBACK_SCHEMA == FEEDBACK_SCHEMAS[0] == "none":
        raise ValueError("'none' must be the first and default feedback schema")
    if not math.isclose(expected_passes_per_batch(), 1.1325, abs_tol=1e-12):
        raise ValueError(
            f"pass schedule expects {expected_passes_per_batch()!r} passes/batch, "
            "not the frozen 1.1325"
        )
    covered = 0.0
    for phase in PASS_SCHEDULE:
        if phase.start != covered:
            raise ValueError(f"pass schedule has a gap or overlap at {phase.start}")
        covered = phase.stop
    if covered != 1.0:
        raise ValueError(f"pass schedule covers {covered} of training, not all of it")
    if tuple(PASS_SCHEDULE[-1].weights) != tuple(
            SOURCE_SCHEDULE["steady_state_pass_weights"]):
        raise ValueError(
            "the final project mix must be recorded against the source 75/22/3 "
            "mixture before it can be called an adaptation"
        )
    if SOURCE_SCHEDULE["phase_boundaries_specified"]:
        raise ValueError(
            "source phase boundaries are not specified; do not promote the "
            "project boundaries to source conditions"
        )
    if feedback_overhead(PILOT_D_MODEL) != 32_896:
        raise ValueError("feedback overhead at D=128 must be 2D^2 + D = 32,896")
    if len(FEEDBACK_PARAMETERS) != 3 or len(set(FEEDBACK_PARAMETERS)) != 3:
        raise ValueError(
            "glu_v1 owns exactly three named tensors -- W_U, W_G and the "
            "dedicated RMSNorm gain -- and the 2D^2 + D count assumes it"
        )
    if len(CELLS) != 8:
        raise ValueError(f"the bounded matrix is eight cells, not {len(CELLS)}")
    if len(set(seed_table().values())) != len(SEED_NAMESPACES):
        raise ValueError("two seed namespaces collide on one value")
    if FUSED_NORM_GAIN != JITTER_HALF_WIDTH:
        raise ValueError(
            "the fused-norm gain and the jitter half-width are both the embedding's "
            "initial scale; decoupling them needs its own protocol entry"
        )
    for name, row in parameter_table().items():
        if row["standard"] + row["overhead"] != row["feedback_capable"]:
            raise ValueError(f"{name}: parameter table does not add up")
        if name in REPRESENTATIONS and not row["under_budget"]:
            raise ValueError(f"{name} is in the pilot and busts the parameter budget")


__all__ = [
    "ALPHA",
    "BOOTSTRAP_REPS",
    "BOOTSTRAP_UNIT",
    "CARRIED_STATE",
    "CELLS",
    "CLAIM_LABELS",
    "CONFIG_FIELD",
    "CONTRAST_ORDER",
    "CORRECTION_DATA_SEED",
    "CORRECTION_GAIN_CALIBRATION",
    "CORRECTION_GATE",
    "CORRECTION_LABELS",
    "CORRECTION_MAX_CELLS",
    "CORRECTION_PACKAGE",
    "CORRECTION_REPEATED_SEED",
    "CORRECTION_REPLICATES",
    "CORRECTION_SCHEDULE",
    "CORRECTION_SEEDS",
    "COUNTED_REPRESENTATIONS",
    "DECISION_RULE",
    "DEFAULT_FEEDBACK_SCHEMA",
    "DEFAULT_GAIN_CALIBRATION",
    "DETACH_CARRIED_STATE",
    "DEVELOPMENT_DATA_SEED",
    "DEVELOPMENT_N_TRAIN",
    "DEVELOPMENT_N_VAL",
    "DEVELOPMENT_SEEDS",
    "DIAGNOSTIC_MODES",
    "DIAGNOSTIC_SCHEDULES",
    "DIRECTION",
    "DOES_NOT_FREEZE",
    "ELIGIBLE_SCHEDULES",
    "ESTIMANDS",
    "ESTIMAND_UNIT",
    "EVALUATION_MODES",
    "F5B_PACKAGE",
    "F5C_PACKAGE",
    "FEEDBACK_PARAMETERS",
    "FEEDBACK_SCHEMAS",
    "FEEDBACK_WEIGHT",
    "FINAL_STEPS",
    "FIXTURE_PATH_STR",
    "FIXTURE_SCHEMA",
    "FREEZES",
    "FUSED_NORM_GAIN",
    "FUSION_EQUATION",
    "GAIN_CALIBRATIONS",
    "GAIN_CALIBRATION_RATIONALE",
    "GENERIC_GUARDS",
    "INHERITED_BOOTSTRAP_SEED",
    "JITTER_HALF_WIDTH",
    "MAX_PASSES",
    "MECHANISM_LIMIT",
    "MODEL_INTERFACE",
    "MODEL_SEEDS",
    "MULTIPLICITY",
    "OBJECTIVE",
    "PACKAGES",
    "PARAMETER_BUDGET",
    "PASS_SCHEDULE",
    "PASS_SCHEDULES",
    "PILOT_D_MODEL",
    "PILOT_N_HEADS",
    "PILOT_N_LAYERS",
    "PILOT_SHAPE",
    "PRIMARY_FAMILY",
    "PROMOTION_GATE",
    "PROTOCOL",
    "PROTOCOL_PATH",
    "PROTOCOL_SCHEMA",
    "PROTOCOL_V1",
    "PROTOCOL_V1_PATH",
    "QUALIFICATION_FEEDBACK_SCHEMA",
    "QUALIFICATION_GATE",
    "QUALIFICATION_SCHEMA",
    "QUALIFICATION_SOURCES",
    "QUESTION",
    "RECORDED_ENVIRONMENT_VARIABLES",
    "RECORD_SCHEMA",
    "REPORT_SCHEMA",
    "REPRESENTATIONS",
    "REQUESTS_PER_CASE",
    "REQUIRED_ENVIRONMENT_FIELDS",
    "REQUIRED_GENERATION_FIELDS",
    "REQUIRED_RECORD_FIELDS",
    "REQUIRED_SCORE_FIELDS",
    "RUNTIME_MODES",
    "RUNTIME_MODE_ARGUMENT",
    "SEED_NAMESPACES",
    "SEED_ROOT",
    "SESOI_BITS_PER_TARGET_BYTE",
    "SOURCE_CONDITION_LEDGER",
    "SOURCE_FEEDBACK_SCHEMA",
    "SOURCE_FEEDBACK_SCHEMA_V2",
    "SOURCE_FUSION_EQUATION",
    "SOURCE_SCHEDULE",
    "SOURCE_V2_FUSION_EQUATION",
    "SOURCE_V2_NORMALIZATION",
    "STABILITY_GATE",
    "STABILITY_PASSES",
    "STABILITY_REPORT_SCHEMA",
    "STABILITY_RMS_BAND",
    "STABILITY_SUBSET_SIZE",
    "TRAINING_DEFAULTS",
    "TRAINING_SOURCES",
    "TRAINING_STRUCTURES",
    "VENUE",
    "VENUE_CASES",
    "VENUE_MANIFEST_SHA256",
    "WITHDRAWN_CLAUSE_REASONS",
    "Package",
    "PassPhase",
    "cell_name",
    "check_consistency",
    "correction_cell_name",
    "development_cell_name",
    "digest_of",
    "expected_passes_per_batch",
    "feedback_overhead",
    "feedback_phase_transition",
    "gain_calibration",
    "load_protocol",
    "missing_record_fields",
    "package",
    "parameter_table",
    "pass_schedule",
    "phase_for",
    "protocol_dict",
    "protocol_v1_dict",
    "seed_for",
    "seed_table",
    "standard_params",
    "training_config",
    "training_config_correction",
    "training_config_development",
    "write_protocol",
]
