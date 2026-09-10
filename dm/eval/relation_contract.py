"""Direction 4's R1 contract: everything fixed *before* relation code exists.

`docs/copy-relation.md` is the prose contract and this Module is its executable
half. Every number a later stage is allowed to read -- schemas, seed namespaces,
runtime modes, the effect floors, the guard bounds, the label order, the
parameter arithmetic, the statistical plan -- is declared here once, and
`docs/copy-relation-protocol-v0.json` is a serialisation of exactly this Module.
A test compares the two, so prose, code and the frozen artifact cannot drift
apart silently (`tests/test_relation_contract.py`).

**Why a contract Module rather than constants scattered across R2-R8.** Direction
3 paid this cost twice. Its C4 driver was edited after a protocol froze its
digest, so for a day no file matching the freeze existed. Its F5b corpus clause
compared a manifest to itself and a fully green suite did not notice. Both
repairs were the same shape: make the freeze name *content*, and give an
invariant exactly one owner. Here the same rule is applied at R1, before there is
any relation output that could suggest what a threshold should be.

**What v0 freezes and what it deliberately cannot.** v0 freezes the *interface*:
schemas, runtime modes, seed namespaces, corpus caps and supports, effect floors,
guard bounds, label order and expected parameter counts, plus the digest of the
traced-corpus manifest R1 builds. It does not freeze source digests, checkpoints
or records -- R3 changes `dm/models/transformer.py` by design, and no checkpoint
exists until R6. Protocol v1 adds the training freeze after the R5 smoke;
protocol v2 adds checkpoint and record bytes before any outcome score. Three
freezes, because a single one would have to invent future hashes.

**Direction 4's namespace is disjoint from Direction 3's**
(`docs/copy-relation.md` §7 invariant 2). No threshold, seed, cell name or
artifact path here is derived from a feedback constant, and the seed root is
salted with the direction so a namespace name reused across directions cannot
land on the same stream.

Nothing here reads a model, a checkpoint or a report. That is the point: no
output may choose a corpus, case, threshold or checkpoint
(`docs/copy-relation.md` §7 invariant 1).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from ..data import relation as corpus
from ..isa.codec import CODECS
from ..models.transformer import Config
from .reports import staged_path

#: Read from the corpus Module so the number has exactly one owner.
DIRECTION = corpus.DIRECTION

#: Protocol identity and version chain. v0 is the R1 contract; v1 adds the
#: training freeze at R6 and v2 the checkpoint/record freeze at R7. A stage that
#: reads the wrong one fails closed rather than running under a freeze that never
#: covered it.
PROTOCOL = "relation-v0"
PROTOCOL_SCHEMA = 1
PROTOCOL_PATH = Path("docs/copy-relation-protocol-v0.json")

#: Where the frozen traced-corpus manifest lives, as a string, so the serialised
#: payload is byte-identical on any platform.
CORPUS_MANIFEST_PATH_STR = "runs/relation_corpus_v1.json"
CORPUS_MANIFEST_PATH = Path(CORPUS_MANIFEST_PATH_STR)

#: Bumped whenever a change makes new relation artifacts incomparable with old
#: ones. Separate numbers because the artifacts freeze at different stages and a
#: shared counter would force a spurious bump on the others.
RECORD_SCHEMA = 1
REPORT_SCHEMA = 1

# ---------------------------------------------------------------------------
# architecture

#: `"none"` is the default and *is* the current implementation: an old config
#: dictionary omits the field, loads as `none`, creates no relation parameters
#: and reproduces every existing checkpoint bit for bit. `"span_affine_v1"` is
#: the first opt-in value. Anything else is refused rather than silently treated
#: as `none` -- a typo that disabled the head would look exactly like a null
#: result.
RELATION_SCHEMAS: tuple[str, ...] = ("none", "span_affine_v1")
DEFAULT_RELATION_SCHEMA = "none"

#: Direction 4 v1 refuses to coexist with Direction 3's latent feedback. Two
#: optional parameter blocks in one checkpoint would make "the same weights in
#: two runtime modes" ambiguous about *which* mechanism moved, and Direction 3 is
#: closed at this scale (`PLAN.md` §2).
FORBIDDEN_FEEDBACK_SCHEMA = "none"

#: The three runtime modes of one relation-capable checkpoint. Runtime modes do
#: not create checkpoints: the whole point is that `predicted_copy` and
#: `standard` are the *same weights*, so the contrast has exact parameter,
#: optimizer, corpus and training equality (`docs/copy-relation.md` §2.1).
RUNTIME_MODES: tuple[str, ...] = ("standard", "predicted_copy", "oracle_copy")
PRIMARY_CONTRAST = ("predicted_copy", "standard")
AUXILIARY_CONTRAST = ("standard", "none")
CEILING_CONTRAST = ("oracle_copy", "predicted_copy")

#: The two training arms. Paired on corpus, schedule and non-relation
#: initialisation; they differ in the objective and in owning a head.
TRAINING_ARMS: tuple[str, ...] = ("none", "span_affine_v1")

#: Rank of the low-rank span scorer. Frozen here rather than chosen at R3
#: because it is a free number in `relation_overhead`, and a parameter budget
#: whose terms are decided later is not a budget.
RELATION_RANK = 32

#: Upper edges of the span-length bins the key embedding indexes, in bytes.
#:
#: Owned by `dm.data.relation`, because the last edge is the source byte cap the
#: builder enforces and one number may not have two owners. Restated here because
#: the head's parameter count depends on it and the protocol has to record it.
#: Chosen from the model-blind source-length distribution of the frozen corpus
#: and audited on every build (`dm.data.relation.length_bin_report`): every bin
#: must be reachable and every accepted source must fall inside the last edge.
RELATION_LENGTH_BIN_EDGES: tuple[int, ...] = corpus.LENGTH_BIN_EDGES
RELATION_LENGTH_BINS = len(RELATION_LENGTH_BIN_EDGES)


def length_bin(span_bytes: int) -> int:
    """Which bin a source span of `span_bytes` falls in. Frozen, not learned."""
    for index, edge in enumerate(RELATION_LENGTH_BIN_EDGES):
        if span_bytes <= edge:
            return index
    raise ValueError(
        f"a {span_bytes}-byte source span is past the last length-bin edge "
        f"{RELATION_LENGTH_BIN_EDGES[-1]}; the caps should have refused it")


#: Caps on the R3 predicted-decoding enumerator, frozen before it exists.
#:
#: `docs/copy-relation.md` §3.3 declares that "its candidate and expansion caps
#: are frozen in R1 and recorded" -- so they are recorded here rather than left
#: to the module that will read them. `RELATION_MAX_CANDIDATE_SPANS` bounds how
#: many `(start, stop)` keys a boundary may score and is audited against the
#: widest candidate set the frozen corpus actually presents;
#: `RELATION_KBEST_ACTIONS` bounds how many joint tuples the k-best enumerator
#: expands before the `EMIT`/best-valid-`COPY` decision.
RELATION_MAX_CANDIDATE_SPANS = corpus.MAX_CANDIDATE_SPANS
RELATION_KBEST_ACTIONS = 32

#: The tensors `span_affine_v1` owns, in the order `relation_overhead` counts
#: them. Named rather than summarised so `n_params()` can be checked against the
#: module tensor by tensor at R3, as `test_analytic_param_count_matches_reality`
#: does for `glu_v1`.
RELATION_PARAMETERS: tuple[str, ...] = (
    "norm.weight",      # D
    "query.weight",     # D x rank
    "key_start.weight", # D x rank
    "key_stop.weight",  # D x rank
    "length.weight",    # bins x rank
    "gate.weight",      # D x 2
    "d4.weight",        # D x |D4|
    "dx.weight",        # D x |dx|
    "dy.weight",        # D x |dy|
    "count.weight",     # D x |count|
)

#: The pilot width every Direction 4 cell runs at.
PILOT_D_MODEL = 128

#: The current byte model, and the ceiling a relation-capable one may not pass.
#: `824,704` is `Config(vocab_size=258, d_model=128, n_layers=4).n_params()` and
#: is asserted rather than quoted (`check_consistency`).
BYTE_MODEL_PARAMETERS = 824_704
RELATION_PARAMETER_BUDGET = 32_768
PARAMETER_CAP = BYTE_MODEL_PARAMETERS + RELATION_PARAMETER_BUDGET


def relation_overhead(d_model: int = PILOT_D_MODEL) -> int:
    """Exact trainable-parameter cost of `span_affine_v1`, tensor by tensor.

    Bias-free throughout, so every term is a product of two declared numbers and
    the arithmetic can be checked by hand. The RMSNorm gain is included: an
    earlier Direction 3 estimate left the equivalent term out, and 128 parameters
    is trivial while a count that disagrees with the module is not.

    The translation heads are the term that forced `TRANSLATION_SUPPORT` to be
    three values rather than 256. At `D=128` a factor head over every signed byte
    would cost `2 * 128 * 256 = 65,536` on its own -- twice this whole budget.
    """
    supports = (2, corpus.D4_SUPPORT, corpus.TRANSLATION_SUPPORT,
                corpus.TRANSLATION_SUPPORT, corpus.COUNT_SUPPORT)
    factor_heads = sum(d_model * (width if isinstance(width, int) else len(width))
                       for width in supports)
    span = 3 * d_model * RELATION_RANK + RELATION_LENGTH_BINS * RELATION_RANK
    return d_model + span + factor_heads


def standard_params(representation: str = "byte", d_model: int = PILOT_D_MODEL) -> int:
    """The baseline model's analytic parameter count, from `Config` itself.

    Read from the model rather than restated, so a change to the trunk moves this
    number instead of leaving the contract quoting a count no checkpoint has.
    """
    return Config(vocab_size=CODECS[representation].vocab_size,
                  d_model=d_model).n_params()


def parameter_table(d_model: int = PILOT_D_MODEL) -> dict[str, dict]:
    """Both arms' expected counts and the headroom left under the cap."""
    baseline = standard_params(d_model=d_model)
    overhead = relation_overhead(d_model)
    return {
        "none": {"total": baseline, "relation": 0},
        "span_affine_v1": {"total": baseline + overhead, "relation": overhead},
        "budget": {"cap": PARAMETER_CAP, "allowance": RELATION_PARAMETER_BUDGET,
                   "headroom": RELATION_PARAMETER_BUDGET - overhead},
    }


# ---------------------------------------------------------------------------
# seeds

#: Every stream Direction 4 draws from, and what it governs.
#:
#: Named streams rather than one incremented seed, and the reason is sharper here
#: than in Direction 3: `token` and `action` uniforms are consumed at *different
#: rates* by the two runtime modes, because a copy action replaces several
#: sampling decisions with one. Sharing a sequential generator would shift every
#: later draw exactly where the modes first differ, which is the one place a
#: paired comparison must not move (`docs/copy-relation.md` §7 invariant 24).
SEED_NAMESPACES: dict[str, str] = {
    "corpus": "the scientific traced corpus build, and every split inside it",
    "development_corpus": "the development traced corpus; disjoint from the above",
    "model_a": "first scientific model initialisation",
    "model_b": "second scientific model initialisation",
    "development_a": "first development-only initialisation",
    "development_b": "second development-only initialisation",
    "token": "counter-based uniforms for ordinary symbol sampling",
    "action": "counter-based uniforms for EMIT/COPY promotion",
    "bootstrap": "connected-component resamples for every paired interval",
    "randomization": "component-level sign flips for the paired permutation test",
    "sentinel": "the R0 repeatability sentinel's data and initialisation",
}

#: Salted with the direction so a namespace name reused by a later direction
#: cannot land on Direction 3's stream.
SEED_ROOT = "direction4-relation"


def seed_for(namespace: str) -> int:
    """The frozen 32-bit seed of one named stream.

    Derived from the name rather than chosen, so adding a stream cannot shift an
    existing one -- and the derived values are written into the protocol, so a
    *rename* is caught by the contract test instead of quietly producing a
    different corpus or a different set of uniforms.
    """
    if namespace not in SEED_NAMESPACES:
        raise KeyError(
            f"unknown seed namespace {namespace!r}; declare it in SEED_NAMESPACES "
            "before drawing from it, so the protocol records it")
    digest = hashlib.sha256(f"{SEED_ROOT}:{namespace}".encode()).hexdigest()
    return int(digest[:8], 16)


def seed_table() -> dict[str, int]:
    return {name: seed_for(name) for name in sorted(SEED_NAMESPACES)}


#: The two scientific initialisations, by name. Reported separately and both must
#: pass; pooling them to rescue a failed seed is forbidden
#: (`docs/copy-relation.md` §2.3).
SCIENTIFIC_SEEDS: tuple[str, ...] = ("model_a", "model_b")
DEVELOPMENT_SEEDS: tuple[str, ...] = ("development_a", "development_b")

#: Which corpus seed a provenance draws. Two of them, because
#: `docs/copy-relation.md` §4.2 requires development and scientific manifests,
#: source identities and case IDs to be disjoint -- and a single corpus seed
#: makes "development sees no scientific cases" unenforceable by construction.
CORPUS_SEEDS: dict[str, str] = {"scientific": "corpus",
                                "development": "development_corpus"}


def corpus_seed(provenance: str = "scientific") -> int:
    """The `data_seed` a traced-corpus build of this provenance must use.

    A build that used some other value produced a different corpus, and the
    manifest records the value it used so the two can be compared rather than
    assumed equal (`corpus_seed_faults`).
    """
    if provenance not in CORPUS_SEEDS:
        raise ValueError(
            f"unknown corpus provenance {provenance!r}; expected one of "
            f"{list(CORPUS_SEEDS)}")
    return seed_for(CORPUS_SEEDS[provenance])


def cell_name(arm: str, seed_namespace: str) -> str:
    """The name of one training cell. Four exist and no fifth may be created."""
    if arm not in TRAINING_ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {list(TRAINING_ARMS)}")
    if seed_namespace not in SCIENTIFIC_SEEDS:
        raise ValueError(
            f"{seed_namespace!r} is not a scientific seed; expected one of "
            f"{list(SCIENTIFIC_SEEDS)}")
    return f"relation_{arm}_{seed_namespace}"


CELLS: tuple[str, ...] = tuple(
    cell_name(arm, seed) for arm in TRAINING_ARMS for seed in SCIENTIFIC_SEEDS)

# ---------------------------------------------------------------------------
# training objective

#: `L_total = L_byte + ACTION_WEIGHT * L_action`, one fixed coefficient and not
#: the winner of a sweep. R4 records component losses and shared-trunk gradient
#: norms; a non-finite or zero-gradient component fails qualification, and the
#: coefficient is never tuned on an outcome (`docs/copy-relation.md` §4.4).
#:
#: **The action term marginalises over the joint valid-action set, not over each
#: factor separately.** A per-factor sum -- one `L_span`, one `L_d4`, one `L_dx`
#: and so on, each over its own marginal mass -- is minimised by putting mass on
#: any span, any D4 element and any translation that appear *somewhere* in the
#: valid set, including combinations that appear nowhere in it. On a corpus whose
#: whole point is that only some tuples are valid, that lets probability collect
#: on invalid cross-products and still score well. So the trained quantity is
#: `-log sum_{a in valid(b)} p(a)` with `p(a)` the product of its factors, and
#: the per-factor numbers survive only as reported diagnostics.
OBJECTIVE = ("L_total = L_byte + 0.1 * (L_gate + L_action_joint), "
             "L_action_joint = -log sum_{a in valid(b)} p_span(a) p_d4(a) "
             "p_dx(a) p_dy(a) p_count(a)")
ACTION_WEIGHT = 0.1
ACTION_MARGINALIZATION = "joint_valid_action_set"
ACTION_LOSS_COMPONENTS: tuple[str, ...] = ("gate", "action_joint")
#: Reported per factor, never summed into the objective. Kept so a failure can be
#: localised without changing what was optimised.
ACTION_DIAGNOSTIC_FACTORS: tuple[str, ...] = ("span", "d4", "dx", "dy", "count")

#: Which boundaries carry a gate target.
#:
#: **Reachable ones only, and this is an R1 decision the prose did not settle.**
#: After a `COPY` the decoder appends the whole target block deterministically
#: and resumes at its end, so it never stands at a boundary *inside* that block
#: and never makes a decision there. Averaging `L_gate` over unreachable
#: boundaries would train the head on positions no decode visits and would let a
#: nested corpus's boundary count depend on how deeply its copies nest. Those
#: bytes keep ordinary byte supervision; only the gate skips them
#: (`dm.data.relation.covered_boundaries`).
GATE_POSITIONS = "reachable_boundaries"

#: The ordinary byte head is trained and scored at every non-PAD byte, copied
#: bytes included, so `bits_per_drawing` stays comparable to every existing byte
#: checkpoint. Action NLL is reported as `relation_bits` and is **never** added
#: to it: a deterministic multi-byte action is a semi-Markov decoding policy, not
#: a normalised per-byte distribution (`docs/copy-relation.md` §3.5, §7
#: invariant 13).
BYTE_NLL_COVERS_COPIED_BYTES = True
ACTION_NLL_ENTERS_BITS_PER_DRAWING = False

# ---------------------------------------------------------------------------
# effect floors and guards

#: Paired exact-block improvement `D_copy = H(predicted_copy) - H(standard)`,
#: required in the flat affine stratum and both seeds for any positive label, and
#: additionally in the nested stratum for the compositional label.
MIN_EXACT_BLOCK_GAIN = 0.10

#: Recovery of the same-checkpoint oracle gap,
#: `R_copy = D_copy / (1 - H(standard))`.
MIN_ORACLE_GAP_RECOVERY = 0.50

#: Accepted connected source components a co-primary stratum must carry. Owned by
#: the corpus builder, restated here because it is also a gate.
MIN_CONFIRMATORY_COMPONENTS = corpus.MIN_CONFIRMATORY_COMPONENTS

#: Guard bounds. Every one is an absolute ceiling on a *loss*, so a guard that
#: fails cannot be rescued by a larger primary effect
#: (`docs/copy-relation.md` §2.3 label order).
MAX_STANDARD_LIKELIHOOD_COST = 1.0      # bits/drawing, span_affine_v1 vs paired none
MAX_STANDARD_VALID_HALT_LOSS = 0.05     # span_affine_v1 standard vs paired none
MAX_PREDICTED_VALID_HALT_LOSS = 0.01    # predicted_copy vs the same checkpoint
MAX_FALSE_COPY_RATE = 0.01              # COPY invocations on `generic` programs
MAX_RESOURCE_RATIO = 1.50               # peak memory and wall time vs baseline

#: Oracle execution is a software property, not a result. Anything below exact is
#: a transducer defect and makes the run `incomplete`.
REQUIRED_ORACLE_EXACT = 1.0

GUARDS: dict[str, str] = {
    "generic": "standard likelihood cost and valid-halt loss versus paired none",
    "false_copy": "COPY invoked on programs with no derivable relation",
    "destroyed_relation": "COPY invoked on length- and census-matched negatives",
    "validity": "predicted_copy valid-halt loss versus the same checkpoint",
    "resource": "peak memory and wall time versus the baseline smoke shape",
    "parameters": "total trainable parameters versus the frozen cap",
    "oracle": "oracle_copy reproduces the frozen target exactly",
}

# ---------------------------------------------------------------------------
# statistics

BOOTSTRAP_DRAWS = 2_000
ALPHA = 0.05
MULTIPLICITY_CORRECTION = "holm"
RESAMPLING_UNIT = "connected co-occurrence motif component"
PAIRING = "frozen case, common random numbers within checkpoint and draw"

#: Two seeds cannot support a population-of-models claim. The valid statement is
#: checkpoint-conditional replication under two named initialisations, and it is
#: written into the protocol so a later reader cannot upgrade it.
INFERENCE_SCOPE = (
    "checkpoint-conditional replication under two named initialisations; not a "
    "population-of-models claim")

# ---------------------------------------------------------------------------
# labels

#: Applied in order; the first that matches wins. There is no "promising" label
#: and no pooling of seeds to rescue a failed one
#: (`docs/copy-relation.md` §2.3).
LABELS: tuple[str, ...] = (
    "incomplete",
    "unsafe_copy_channel",
    "no_explicit_relation_gain",
    "explicit_relation_gain",
    "compositional_relation_gain",
)
LABEL_REASONS: dict[str, str] = {
    "incomplete": "missing cell, artifact or hash; truncation; failed reload; "
                  "unqualified device; corpus defect; malformed case",
    "unsafe_copy_channel": "a generic, destroyed-relation, resource, parameter or "
                           "validity guard failed",
    "no_explicit_relation_gain": "the flat affine effect missed its floor in either seed",
    "explicit_relation_gain": "the flat affine stratum passed both seeds and the "
                              "nested stratum did not",
    "compositional_relation_gain": "both co-primary strata passed in both seeds and "
                                   "every guard passed",
}


def label_for(*, complete: bool, guards_pass: bool, flat_pass: bool,
              nested_pass: bool) -> str:
    """The decision label, applied once, in the frozen order.

    A function rather than a table so the ordering is executable: incompleteness
    precedes every safety guard, safety precedes every effect, and a nested
    failure narrows the claim by name instead of erasing a valid flat result.
    """
    if not complete:
        return "incomplete"
    if not guards_pass:
        return "unsafe_copy_channel"
    if not flat_pass:
        return "no_explicit_relation_gain"
    return "compositional_relation_gain" if nested_pass else "explicit_relation_gain"


# ---------------------------------------------------------------------------
# R0: what the instrument has to prove before a scientific cell

#: The digests R0 compares, and the one it explicitly refuses to compare.
#:
#: A serialised checkpoint's SHA-256 identifies an *artifact*: `torch.save`
#: writes a ZIP whose member names depend on the output filename, and run
#: metadata travels inside the same bytes. Two byte-identical state dictionaries
#: can therefore produce different files. Direction 3's correction report made
#: exactly that comparison and it had to be retracted (`PLAN.md` §2), so the
#: rule is frozen here rather than left to a driver.
R0_DIGESTS: tuple[str, ...] = ("model_state", "optimizer_state", "rng_state")
R0_FORBIDDEN_DIGEST = "checkpoint_file_sha256"

#: Repeats of the fixed short sentinel a qualification needs. All of them must
#: agree on all three digests: v1 has exactly one route.
R0_MIN_REPEATS = 5

#: The complete R0 sentinel shape.  This is deliberately data rather than a
#: boolean reported by the runner: a caller cannot turn a smaller or otherwise
#: different experiment into the frozen sentinel by writing
#: ``is_frozen_shape: true``.  `dm.eval.relation_evidence.SentinelConfig` reads
#: these values, and `qualify` compares the raw report to them.
R0_SENTINEL_CONFIG: dict[str, int | float] = {
    "vocab_size": 258,
    "d_model": 64,
    "n_layers": 2,
    "n_heads": 4,
    "max_len": 128,
    "steps": 12,
    "batch": 8,
    "length": 48,
    "lr": 1e-3,
}
R0_RESUME_STEP = 6
R0_READING_FIELDS: tuple[str, ...] = (
    "model_state", "optimizer_state", "rng_state", "final_loss", "seconds")
R0_RESUME_DIGEST_FIELDS: tuple[str, ...] = R0_DIGESTS
R0_RESUME_FIELDS: tuple[str, ...] = (
    "schema", "device", "seed", "sentinel", "environment", "resume_step",
    "serialised", "rebuilt_model_and_optimizer", "process_boundary",
    "uninterrupted", "resumed", "differing_digests", "equivalent",
    "loss_difference")
R0_ENVIRONMENT_FIELDS: tuple[str, ...] = (
    "python", "torch", "numpy", "platform", "machine", "processor", "device",
    "threads", "environment")

#: The metric-level fallback `docs/copy-relation.md` §6 clause 3 allows is
#: **withdrawn in v1**, and the reason is that the version that existed was
#: degenerate rather than merely unused.
#:
#: Its metric was the sentinel's exact-block continuation rate. With a batch of
#: eight the rate moves in steps of `0.125`, coarser than the `0.10` floor it was
#: supposed to resolve; and because the continuations were scored against random
#: reference bytes it read exactly zero in every repeat. A statistic that is
#: always zero has an observed range of zero, so an arbitrarily nondeterministic
#: backend would have "resolved" the floor by missing every continuation.
#:
#: Restoring the route needs a non-degenerate *paired* measurement on the
#: relation task itself, which cannot exist before R3. Until then a backend that
#: cannot reproduce state exactly fails R0, and the floor does not move.
R0_FALLBACK_STATUS = (
    "withdrawn in v1: the implemented metric was degenerate (granularity 1/batch "
    "= 0.125 against a 0.10 floor, and identically zero across repeats). A "
    "replacement must be a paired relation-task measurement and cannot exist "
    "before R3.")

#: `torch.use_deterministic_algorithms(True)` records requested policy. It is
#: never a device certificate, and the protocol says so in the artifact rather
#: than only in prose (`docs/copy-relation.md` §6).
DETERMINISM_IS_POLICY_NOT_CERTIFICATE = True

# ---------------------------------------------------------------------------
# stages

STAGES: dict[str, str] = {
    "R0": "qualify model/optimizer/RNG reproducibility on the chosen backend",
    "R1": "freeze the traced corpus and this executable contract, tests first",
    "R2": "build the deterministic transducer, independent of any neural head",
    "R3": "add the bounded span head and prove default checkpoint compatibility",
    "R4": "add the auxiliary objective and its accounting",
    "R5": "end-to-end smoke and development qualification",
    "R6": "freeze protocol v1 and train the four cells",
    "R7": "freeze protocol v2, score once and apply the label",
    "R8": "optional pre-authorized promotion only",
}
AUTHORIZED_STAGES: tuple[str, ...] = ("R0", "R1")


# ---------------------------------------------------------------------------
# the protocol artifact


def protocol_dict(*, corpus_sha256: str | None = None) -> dict:
    """The v0 payload: this Module, serialised.

    `corpus_sha256` is the traced-corpus manifest's canonical payload digest. It
    is optional only so the protocol can be written in the same command that
    builds the corpus; a protocol without it names no corpus and `load_protocol`
    refuses it for any stage that trains.
    """
    check_consistency()
    body = {
        "protocol": PROTOCOL,
        "schema": PROTOCOL_SCHEMA,
        "direction": DIRECTION,
        "status": "frozen",
        "authorized_stages": list(AUTHORIZED_STAGES),
        "stages": dict(sorted(STAGES.items())),
        "architecture": {
            "relation_schemas": list(RELATION_SCHEMAS),
            "default_relation_schema": DEFAULT_RELATION_SCHEMA,
            "forbidden_feedback_schema": FORBIDDEN_FEEDBACK_SCHEMA,
            "runtime_modes": list(RUNTIME_MODES),
            "primary_contrast": list(PRIMARY_CONTRAST),
            "auxiliary_contrast": list(AUXILIARY_CONTRAST),
            "ceiling_contrast": list(CEILING_CONTRAST),
            "relation_rank": RELATION_RANK,
            "relation_length_bins": RELATION_LENGTH_BINS,
            "relation_length_bin_edges": list(RELATION_LENGTH_BIN_EDGES),
            "max_candidate_spans": RELATION_MAX_CANDIDATE_SPANS,
            "kbest_actions": RELATION_KBEST_ACTIONS,
            "relation_parameters": list(RELATION_PARAMETERS),
            "d_model": PILOT_D_MODEL,
        },
        "parameters": parameter_table(),
        "corpus": {
            "manifest_path": CORPUS_MANIFEST_PATH_STR,
            "canonical_payload_sha256": corpus_sha256,
            "data_seed": corpus_seed("scientific"),
            "development_data_seed": corpus_seed("development"),
            "schema": corpus.CORPUS_SCHEMA,
            "manifest_schema": corpus.MANIFEST_SCHEMA,
            "venues": list(corpus.VENUES),
            "strata": list(corpus.ALL_STRATA),
            "provenances": list(corpus.CORPUS_PROVENANCES),
            "source_partition": {
                "count": corpus.composed.SOURCE_PARTITION_COUNT,
                "names": dict(sorted(corpus.composed.SOURCE_PARTITIONS.items())),
                "marker": corpus.composed.SOURCE_PARTITION_MARKER.decode(),
            },
            "confirmatory_strata": list(corpus.CONFIRMATORY_STRATA),
            "motif_disjoint_strata": list(corpus.MOTIF_DISJOINT_STRATA),
            "caps": {
                "max_source_gap_instructions": corpus.MAX_SOURCE_GAP_INSTRUCTIONS,
                "max_source_instructions": corpus.MAX_SOURCE_INSTRUCTIONS,
                "max_source_bytes": corpus.MAX_SOURCE_BYTES,
                "min_source_instructions": corpus.MIN_SOURCE_INSTRUCTIONS,
                "min_source_bytes": corpus.MIN_SOURCE_BYTES,
                "min_source_coords": corpus.MIN_SOURCE_COORDS,
            },
            "supports": {
                "d4": list(corpus.D4_SUPPORT),
                "translation": list(corpus.TRANSLATION_SUPPORT),
                "count": list(corpus.COUNT_SUPPORT),
                "audit_translation": [min(corpus.AUDIT_TRANSLATIONS),
                                      max(corpus.AUDIT_TRANSLATIONS)],
            },
            "min_confirmatory_components": MIN_CONFIRMATORY_COMPONENTS,
        },
        "objective": {
            "formula": OBJECTIVE,
            "action_weight": ACTION_WEIGHT,
            "components": list(ACTION_LOSS_COMPONENTS),
            "diagnostic_factors": list(ACTION_DIAGNOSTIC_FACTORS),
            "marginalization": ACTION_MARGINALIZATION,
            "gate_positions": GATE_POSITIONS,
            "byte_nll_covers_copied_bytes": BYTE_NLL_COVERS_COPIED_BYTES,
            "action_nll_enters_bits_per_drawing": ACTION_NLL_ENTERS_BITS_PER_DRAWING,
        },
        "floors": {
            "min_exact_block_gain": MIN_EXACT_BLOCK_GAIN,
            "min_oracle_gap_recovery": MIN_ORACLE_GAP_RECOVERY,
            "max_standard_likelihood_cost": MAX_STANDARD_LIKELIHOOD_COST,
            "max_standard_valid_halt_loss": MAX_STANDARD_VALID_HALT_LOSS,
            "max_predicted_valid_halt_loss": MAX_PREDICTED_VALID_HALT_LOSS,
            "max_false_copy_rate": MAX_FALSE_COPY_RATE,
            "max_resource_ratio": MAX_RESOURCE_RATIO,
            "required_oracle_exact": REQUIRED_ORACLE_EXACT,
            "parameter_cap": PARAMETER_CAP,
        },
        "guards": dict(sorted(GUARDS.items())),
        "statistics": {
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "alpha": ALPHA,
            "multiplicity_correction": MULTIPLICITY_CORRECTION,
            "resampling_unit": RESAMPLING_UNIT,
            "pairing": PAIRING,
            "inference_scope": INFERENCE_SCOPE,
        },
        "labels": {"order": list(LABELS), "reasons": dict(sorted(LABEL_REASONS.items()))},
        "cells": list(CELLS),
        "seeds": {
            "root": SEED_ROOT,
            "namespaces": dict(sorted(SEED_NAMESPACES.items())),
            "values": seed_table(),
            "scientific": list(SCIENTIFIC_SEEDS),
            "development": list(DEVELOPMENT_SEEDS),
        },
        "r0": {
            "digests": list(R0_DIGESTS),
            "forbidden_digest": R0_FORBIDDEN_DIGEST,
            "min_repeats": R0_MIN_REPEATS,
            "sentinel_config": dict(R0_SENTINEL_CONFIG),
            "resume": {
                "resume_step": R0_RESUME_STEP,
                "serialised": True,
                "rebuilt_model_and_optimizer": True,
                "process_boundary": False,
                "digest_fields": list(R0_RESUME_DIGEST_FIELDS),
            },
            "reading_fields": list(R0_READING_FIELDS),
            "resume_fields": list(R0_RESUME_FIELDS),
            "environment_fields": list(R0_ENVIRONMENT_FIELDS),
            "fallback_status": R0_FALLBACK_STATUS,
            "routes": ["exact_state_reproduction"],
            "determinism_is_policy_not_certificate":
                DETERMINISM_IS_POLICY_NOT_CERTIFICATE,
        },
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


def corpus_seed_faults(manifest: Mapping, *,
                       provenance: str = "scientific") -> list[str]:
    """Whether a manifest was built under the seed the contract names.

    The contract derives `corpus` from its own salted root; the builder takes a
    `data_seed` argument. Nothing connected the two, so a manifest built at the
    dataclass default of zero satisfied every clause while the protocol recorded
    a seed no build had ever used. This compares them, in that direction, and
    also refuses a scientific manifest that carries the development seed.
    """
    want = corpus_seed(provenance)
    config = (manifest.get("acceptance") or {}).get("config") or {}
    found = config.get("data_seed")
    found_provenance = config.get("provenance")
    faults: list[str] = []
    if found != want:
        faults.append(
            f"the manifest was built at data_seed={found!r}; the contract's "
            f"{provenance} corpus seed is {want}")
    if found_provenance != provenance:
        faults.append(
            f"the manifest carries provenance={found_provenance!r}; the contract "
            f"expects {provenance!r}")
    other = {name: corpus_seed(name) for name in CORPUS_SEEDS if name != provenance}
    for name, value in other.items():
        if found == value:
            faults.append(
                f"the manifest carries the {name} corpus seed and is filed as "
                f"{provenance}")
    return faults


def load_protocol(path: Path = PROTOCOL_PATH, *, expect: str = PROTOCOL,
                  require_corpus: bool = False,
                  corpus_root: Path = Path()) -> dict:
    """Read a protocol, refusing a payload whose digest, schema or status disagrees.

    Fail closed on all four. A protocol that can be edited without its digest
    moving is not a freeze, and a stage that reads an unfrozen protocol has no
    provenance to report -- which `docs/copy-relation.md` §2.3 resolves as
    `incomplete`, never as a result.

    `require_corpus` is what a training stage passes, and it is a check on the
    *bytes* rather than on a string. v0 may legitimately exist before the corpus
    is built, so the digest field may be absent; but a stage that trains must
    have the manifest on disk, must be able to load it -- which validates its own
    digest, its schema and its acceptance verdict -- and must find its canonical
    payload digest equal to the one the protocol froze. An earlier version
    checked only that some digest string was present, so a fresh clone with no
    `runs/` directory passed the training precondition while possessing no
    corpus at all.
    """
    protocol = json.loads(path.read_text())
    recorded = protocol.get("protocol_sha256")
    actual = digest_of(protocol)
    if recorded != actual:
        raise ValueError(
            f"relation protocol hash mismatch for {path}: recorded {recorded!r}, "
            f"computed {actual!r}")
    if protocol.get("protocol") != expect:
        raise ValueError(
            f"{path} is protocol {protocol.get('protocol')!r}, expected {expect!r}")
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError(
            f"unsupported relation protocol schema {protocol.get('schema')!r}; "
            f"expected {PROTOCOL_SCHEMA}")
    if protocol.get("status") != "frozen":
        raise ValueError(f"relation protocol is not frozen: {path}")
    if require_corpus:
        from ..data import relation as corpus_module

        block = protocol.get("corpus") or {}
        recorded = block.get("canonical_payload_sha256")
        if not recorded:
            raise ValueError(
                f"{path} names no traced corpus; a training stage cannot run "
                "under a protocol that does not identify the bytes it trains "
                "on: incomplete")
        manifest_path = corpus_root / block.get("manifest_path", "")
        if not manifest_path.is_file():
            raise ValueError(
                f"{path} names corpus {manifest_path}, which does not exist; a "
                "digest in a protocol is not a corpus: incomplete")
        manifest = corpus_module.load_manifest(manifest_path)
        found = corpus_module.manifest_digests(manifest_path)[
            "canonical_payload_sha256"]
        if found != recorded:
            raise ValueError(
                f"{manifest_path} has payload digest {found}, but {path} froze "
                f"{recorded}: this is a different corpus")
        faults = corpus_seed_faults(manifest)
        if faults:
            raise ValueError("; ".join(faults))
    return protocol


def check_consistency() -> None:
    """Assert the invariants that make the contract self-consistent.

    Called by the writer *and* by the test, so a hand edit to the JSON cannot
    survive and a change here cannot ship without the JSON being rewritten.
    """
    if not DEFAULT_RELATION_SCHEMA == RELATION_SCHEMAS[0] == "none":
        raise ValueError("'none' must be the first and default relation schema")
    if RUNTIME_MODES[0] != "standard":
        raise ValueError(
            "'standard' must be the first runtime mode: it is the pre-Direction-4 "
            "decode and the default a checkpoint without a head can offer")
    if TRAINING_ARMS != RELATION_SCHEMAS:
        raise ValueError(
            "the training arms are exactly the relation schemas; an arm that is "
            "not a schema would be an architecture nothing can serialise")
    if standard_params() != BYTE_MODEL_PARAMETERS:
        raise ValueError(
            f"the byte model is {standard_params()} parameters, not the frozen "
            f"{BYTE_MODEL_PARAMETERS}; the cap was derived from that number")
    if PARAMETER_CAP != 857_472:
        raise ValueError(f"the frozen cap is 857,472, not {PARAMETER_CAP}")
    if relation_overhead() > RELATION_PARAMETER_BUDGET:
        raise ValueError(
            f"span_affine_v1 costs {relation_overhead()} parameters at D="
            f"{PILOT_D_MODEL}, over the {RELATION_PARAMETER_BUDGET} allowance")
    if len(RELATION_PARAMETERS) != len(set(RELATION_PARAMETERS)):
        raise ValueError("span_affine_v1's tensor names must be distinct")
    if len(CELLS) != 4 or len(set(CELLS)) != 4:
        raise ValueError(f"the bounded matrix is four cells, not {len(CELLS)}")
    if len(SCIENTIFIC_SEEDS) != 2:
        raise ValueError("two scientific seeds, reported separately, never pooled")
    if set(SCIENTIFIC_SEEDS) & set(DEVELOPMENT_SEEDS):
        raise ValueError(
            "development and scientific seed namespaces must be disjoint; a "
            "development artifact may never enter a scientific threshold")
    if corpus_seed("scientific") == corpus_seed("development"):
        raise ValueError(
            "the development corpus must be built under its own seed, or "
            "'development sees no scientific cases' is unenforceable")
    if length_bin(1) != 0 or length_bin(RELATION_LENGTH_BIN_EDGES[-1]) != \
            RELATION_LENGTH_BINS - 1:
        raise ValueError("the length-bin edges do not span their own range")
    if list(RELATION_LENGTH_BIN_EDGES) != sorted(set(RELATION_LENGTH_BIN_EDGES)):
        raise ValueError("the length-bin edges must be distinct and increasing")
    if RELATION_LENGTH_BIN_EDGES[-1] != corpus.MAX_SOURCE_BYTES:
        raise ValueError(
            "the last length-bin edge must be the source byte cap, or an "
            "accepted span could fall outside every bin")
    if ACTION_MARGINALIZATION != "joint_valid_action_set":
        raise ValueError(
            "the action term must marginalise the joint valid set; per-factor "
            "marginals let probability collect on invalid cross-products")
    if set(ACTION_LOSS_COMPONENTS) & set(ACTION_DIAGNOSTIC_FACTORS):
        raise ValueError(
            "a per-factor diagnostic may not also be an optimised component")
    if len(set(seed_table().values())) != len(SEED_NAMESPACES):
        raise ValueError("two seed namespaces collide on one value")
    if LABELS[0] != "incomplete" or LABELS[-1] != "compositional_relation_gain":
        raise ValueError("the label order runs from incomplete to the widest claim")
    if set(LABELS) != set(LABEL_REASONS):
        raise ValueError("every label needs a recorded reason")
    if ACTION_NLL_ENTERS_BITS_PER_DRAWING:
        raise ValueError(
            "action NLL may never be added to byte NLL and called comparable "
            "bits/drawing (docs/copy-relation.md §7 invariant 13)")
    if FORBIDDEN_FEEDBACK_SCHEMA != "none":
        raise ValueError(
            "relation parameters and latent feedback may not coexist in v1")
    if set(R0_SENTINEL_CONFIG) != {
            "vocab_size", "d_model", "n_layers", "n_heads", "max_len", "steps",
            "batch", "length", "lr"}:
        raise ValueError("the frozen R0 sentinel config is incomplete")
    if not 0 < R0_RESUME_STEP < int(R0_SENTINEL_CONFIG["steps"]):
        raise ValueError("the frozen R0 resume point is outside the sentinel")
    if tuple(R0_RESUME_DIGEST_FIELDS) != tuple(R0_DIGESTS):
        raise ValueError("R0 resume and repeatability digest fields disagree")
    if len(set(R0_READING_FIELDS)) != len(R0_READING_FIELDS) \
            or len(set(R0_RESUME_FIELDS)) != len(R0_RESUME_FIELDS):
        raise ValueError("R0 report fields must be unique")
    if MIN_CONFIRMATORY_COMPONENTS != corpus.MIN_CONFIRMATORY_COMPONENTS:
        raise ValueError(
            "the component floor is owned by the corpus builder and restated "
            "here; the two have drifted")
    if corpus.TRANSLATION_SUPPORT == corpus.AUDIT_TRANSLATIONS:
        raise ValueError(
            "the audit's translation range must be wider than the head's support, "
            "or the destroyed control can only refuse relations the head could "
            "have named")


__all__ = [
    "CELLS", "LABELS", "PROTOCOL", "PROTOCOL_PATH", "RELATION_SCHEMAS",
    "RUNTIME_MODES", "SEED_NAMESPACES", "check_consistency", "digest_of",
    "label_for", "load_protocol", "parameter_table", "protocol_dict",
    "relation_overhead", "seed_for", "seed_table", "standard_params",
    "write_protocol",
]
