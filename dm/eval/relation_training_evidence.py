"""Strict evidence boundary for temporary R4 training bundles.

This module owns the constants a bundle writer and its validator must agree
on -- schema version, schedule identity, diagnostic cadence policy, component
names/weights and the factor roundoff allowance -- so neither can drift from
the other.  Validation reconstructs continuation and diagnostic facts from raw
history rather than trusting summary fields.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import SimpleNamespace

import torch

from ..data.dataset import BUCKET_EPOCH_ALGORITHM, bucket_epoch_plan
from ..models.relation import FACTOR_NAMES
from ..train import lr_at
from .relation_contract import DEVELOPMENT_SEEDS, SCIENTIFIC_SEEDS, seed_for


class RelationTrainingEvidenceRefused(ValueError):
    """An expected malformed or non-authoritative R4 evidence refusal."""


#: 3 additionally binds the versioned corpus supervision identity, actual
#: process runtime and the complete AdamW recipe. Schema-1/2 bundles lack facts
#: that cannot be reconstructed truthfully and are not migratable.
RELATION_CHECKPOINT_SCHEMA = 3
SCHEDULE_IDENTITY = "historical_cosine_v1"
SEED_NAMESPACE_ENGINEERING = "engineering"
PROVENANCE_ENGINEERING = "engineering"
PROVENANCE_DEVELOPMENT = "development"
PROVENANCE_SCIENTIFIC = "scientific"
PERMITTED_PROVENANCES = frozenset({
    PROVENANCE_ENGINEERING,
    PROVENANCE_DEVELOPMENT,
    PROVENANCE_SCIENTIFIC,
})
DIAGNOSTIC_POLICY_SCHEDULE = "schedule_boundaries_v1"
DIAGNOSTIC_POLICY_EXPLICIT = "explicit_override"
#: The three normalized objective components and their fixed weights in
#: `L_byte + 0.1 * (L_gate + L_action_joint)`.  Weighted norms are derived by
#: the reader; their sum is not the total gradient norm.
COMPONENT_NAMES: tuple[str, ...] = ("byte", "gate", "action_joint")
COMPONENT_WEIGHTS: dict[str, float] = {"byte": 1.0, "gate": 0.1, "action_joint": 0.1}
#: Float32 log-mass of a projected valid set can round to a tiny negative NLL.
#: Per positive query, this much signed roundoff is retained as read; anything
#: more negative or nonfinite is refused.  It never touches the optimized loss.
FACTOR_ALLOWANCE_PER_QUERY = 1e-6

REQUIRED_BUNDLE_KEYS = frozenset({
    "schema", "namespace", "config", "completed_step", "model", "optimizer", "sampler",
    "accounting", "history", "diagnostics", "corpus", "source_hashes", "source_digest",
    "environment", "rng", "content_digests",
})
CONFIG_KEYS = frozenset({
    "arm", "model", "seed", "batch_size", "max_len", "steps", "warmup", "lr", "weight_decay",
    "grad_clip", "adam_betas", "adam_eps", "adam_amsgrad", "adam_foreach", "adam_fused",
    "adam_capturable", "adam_differentiable", "adam_maximize", "attention_budget",
    "device", "deterministic", "artifact_output_policy",
    "corpus_provenance", "seed_namespace", "schedule", "max_cached_cases", "max_cached_spans",
    "max_cached_bytes", "component_gradient_events",
})
SAMPLER_KEYS = frozenset({"algorithm", "lengths", "batch_size", "base_seed", "epoch", "cursor",
                          "batches", "pool_batches", "generator"})
HISTORY_KEYS = frozenset({"step", "batch_ids", "lr", "byte_sum", "gate_sum", "joint_sum",
                          "content_bytes", "reachable_boundaries", "positive_boundaries",
                          "factors"})
FACTOR_KEYS = frozenset({"status", "unit", "denominator", "sums"})
DIAGNOSTIC_KEYS = frozenset({"component_policy", "component_events", "trunk", "component_records"})
TRUNK_KEYS = frozenset({"names", "shapes", "dtype", "parameters", "digest"})
COMPONENT_RECORD_KEYS = frozenset({"step", "batch_ids", "trunk_digest", "components",
                                   "extra_backward_calls", "buffer_bytes"})
COMPONENT_KEYS = frozenset({"status", "norm", "denominator", "weight"})
STATUS_MEASURED, STATUS_NO_POSITIVES, STATUS_NOT_APPLICABLE = (
    "measured", "no_positive_boundaries", "not_applicable")


def validate_provenance_and_seed(corpus_provenance: object, seed_namespace: object, seed: object) -> None:
    """Shared compatibility rule between corpus provenance, seed namespace and seed.

    | corpus/config provenance | permitted model namespace |
    |---|---|
    | engineering | engineering, numeric seed in [0, 2**63 - 1) |
    | development | DEVELOPMENT_SEEDS, matching seed_for(namespace) |
    | scientific  | SCIENTIFIC_SEEDS, matching seed_for(namespace) |
    """
    if type(corpus_provenance) is not str or type(seed_namespace) is not str:
        raise RelationTrainingEvidenceRefused("corpus provenance and seed namespace must be strings")
    if corpus_provenance not in PERMITTED_PROVENANCES:
        raise RelationTrainingEvidenceRefused(f"unknown corpus provenance {corpus_provenance!r}")

    if corpus_provenance == PROVENANCE_ENGINEERING:
        if seed_namespace != SEED_NAMESPACE_ENGINEERING:
            raise RelationTrainingEvidenceRefused(
                f"engineering provenance requires seed namespace {SEED_NAMESPACE_ENGINEERING!r}, got {seed_namespace!r}"
            )
        if type(seed) is not int or isinstance(seed, bool) or not (0 <= seed < 2**63 - 1):
            raise RelationTrainingEvidenceRefused(
                f"engineering seed must be an integer in [0, 2**63 - 1), got {seed!r}"
            )
    elif corpus_provenance == PROVENANCE_DEVELOPMENT:
        if seed_namespace not in DEVELOPMENT_SEEDS:
            raise RelationTrainingEvidenceRefused(
                f"development provenance requires namespace in {list(DEVELOPMENT_SEEDS)}, got {seed_namespace!r}"
            )
        if type(seed) is not int or isinstance(seed, bool) or seed != seed_for(seed_namespace):
            raise RelationTrainingEvidenceRefused(
                f"development seed {seed!r} does not match frozen seed_for({seed_namespace!r})"
            )
    elif corpus_provenance == PROVENANCE_SCIENTIFIC:
        if seed_namespace not in SCIENTIFIC_SEEDS:
            raise RelationTrainingEvidenceRefused(
                f"scientific provenance requires namespace in {list(SCIENTIFIC_SEEDS)}, got {seed_namespace!r}"
            )
        if type(seed) is not int or isinstance(seed, bool) or seed != seed_for(seed_namespace):
            raise RelationTrainingEvidenceRefused(
                f"scientific seed {seed!r} does not match frozen seed_for({seed_namespace!r})"
            )


def default_component_events(steps: int, warmup: int) -> tuple[int, ...]:
    """`schedule_boundaries_v1`: `{1, warmup, warmup+1, steps}` within `[1, steps]`."""
    return tuple(sorted({step for step in (1, warmup, warmup + 1, steps) if 1 <= step <= steps}))


def validate_bundle_shape(bundle: Mapping) -> None:
    """Refuse missing/extra keys and irreconcilable history before a trainer restores state."""
    if not isinstance(bundle, Mapping) or set(bundle) != REQUIRED_BUNDLE_KEYS:
        raise RelationTrainingEvidenceRefused("R4 training bundle key set is not exact")
    if type(bundle.get("schema")) is not int or bundle["schema"] != RELATION_CHECKPOINT_SCHEMA:
        raise RelationTrainingEvidenceRefused(
            f"R4 training bundle schema is not {RELATION_CHECKPOINT_SCHEMA}")
    if bundle.get("namespace") != "direction4-relation-r4-test-only":
        raise RelationTrainingEvidenceRefused("R4 training bundle namespace is not accepted")
    if not isinstance(bundle.get("source_hashes"), dict) or not bundle["source_hashes"]:
        raise RelationTrainingEvidenceRefused("R4 bundle lacks authoritative source hashes")
    if not all(isinstance(path, str) and _sha256(digest)
               for path, digest in bundle["source_hashes"].items()) or \
            not _sha256(bundle.get("source_digest")):
        raise RelationTrainingEvidenceRefused("R4 bundle source identity is malformed")
    if not isinstance(bundle.get("environment"), dict) or not bundle["environment"]:
        raise RelationTrainingEvidenceRefused("R4 bundle lacks an execution environment")
    if type(bundle.get("completed_step")) is not int or bundle["completed_step"] < 0:
        raise RelationTrainingEvidenceRefused("R4 bundle completed step is invalid")
    _validate_config(bundle.get("config"))
    config = bundle["config"]
    if bundle["completed_step"] > config["steps"]:
        raise RelationTrainingEvidenceRefused("R4 bundle completed more steps than its declared horizon")
    _validate_sampler(bundle.get("sampler"))
    sampler = bundle["sampler"]
    _exact_mapping(bundle.get("accounting"), {
        "optimizer_steps", "programs", "semantic_bytes", "content_symbols", "padded_positions",
        "byte_sum", "gate_sum", "joint_sum", "byte_denominator", "gate_denominator",
        "positive_denominator", "reachable_queries", "candidate_spans", "unclipped_grad_norm",
        "clipped_grad_norm",
    }, "accounting")
    accounting = bundle["accounting"]
    if not all(_number(value) for value in accounting.values()) or \
            not all(type(accounting[name]) is int and accounting[name] >= 0 for name in
                    ("optimizer_steps", "programs", "semantic_bytes", "content_symbols",
                     "padded_positions", "byte_denominator", "gate_denominator",
                     "positive_denominator", "reachable_queries", "candidate_spans")):
        raise RelationTrainingEvidenceRefused("R4 bundle accounting values are malformed")
    if not isinstance(bundle.get("history"), list) or len(bundle["history"]) != bundle["completed_step"]:
        raise RelationTrainingEvidenceRefused("R4 bundle history is malformed")
    for expected_step, record in enumerate(bundle["history"], start=1):
        _exact_mapping(record, HISTORY_KEYS, "history record")
        if record["step"] != expected_step or type(record["step"]) is not int or isinstance(record["step"], bool) or \
                not isinstance(record["batch_ids"], list) or not record["batch_ids"] or \
                any(type(index) is not int or isinstance(index, bool) or not (0 <= index < len(sampler["lengths"])) for index in record["batch_ids"]) or \
                len(set(record["batch_ids"])) != len(record["batch_ids"]) or \
                not all(_number(record[name]) for name in
                        HISTORY_KEYS - {"step", "batch_ids", "factors"}):
            raise RelationTrainingEvidenceRefused("R4 bundle history values are malformed")
        counts = ("content_bytes", "reachable_boundaries", "positive_boundaries")
        if any(type(record[name]) is not int or isinstance(record[name], bool) for name in counts):
            raise RelationTrainingEvidenceRefused("R4 history counts must be integers")
        if record["content_bytes"] < 1 or record["reachable_boundaries"] < 1:
            raise RelationTrainingEvidenceRefused("R4 history content_bytes and reachable_boundaries must be positive")
        if not (0 <= record["positive_boundaries"] <= record["reachable_boundaries"] <= record["content_bytes"]):
            raise RelationTrainingEvidenceRefused("R4 history denominator ordering is invalid")
        max_content = sum(sampler["lengths"][c] - 1 for c in record["batch_ids"])
        if record["content_bytes"] != max_content:
            raise RelationTrainingEvidenceRefused("R4 sampler batch lengths do not match history content_bytes")
        _validate_factor_record(record["factors"], config["arm"], record["positive_boundaries"])
    if accounting["optimizer_steps"] != bundle["completed_step"]:
        raise RelationTrainingEvidenceRefused("R4 bundle accounting step disagrees with history")
    _exact_mapping(bundle.get("corpus"), {
        "manifest_path", "canonical_payload_sha256", "file_sha256", "rebuilt_payload_sha256",
        "training_program_fingerprint", "supervision_identity_version", "supervision_sha256",
        "provenance",
    }, "corpus")
    corpus = bundle["corpus"]
    if corpus["manifest_path"] is not None and not isinstance(corpus["manifest_path"], str) or \
            not all(value is None or isinstance(value, str) for value in corpus.values()) or \
            not all(_sha256(corpus[name]) for name in
            ("canonical_payload_sha256", "file_sha256", "rebuilt_payload_sha256",
             "training_program_fingerprint", "supervision_sha256") if corpus[name] is not None) or \
            corpus["supervision_identity_version"] != "r4_training_supervision_v1" or \
            not isinstance(corpus["provenance"], str):
        raise RelationTrainingEvidenceRefused("R4 bundle corpus identity is malformed")
    _exact_mapping(bundle.get("rng"), {"python", "numpy", "torch_cpu", "data_generator"}, "rng",
                   allow_extra={"torch_cuda"})
    rng = bundle["rng"]
    if not isinstance(rng["python"], tuple) or not isinstance(rng["numpy"], tuple) or \
            not _byte_tensor(rng["torch_cpu"]) or \
            (rng["data_generator"] is not None and not _byte_tensor(rng["data_generator"])) or \
            ("torch_cuda" in rng and (not isinstance(rng["torch_cuda"], list) or
                                       not all(_byte_tensor(state) for state in rng["torch_cuda"]))):
        raise RelationTrainingEvidenceRefused("R4 bundle RNG state is malformed")
    if not _byte_tensor(rng["data_generator"]) or not torch.equal(
            sampler["generator"], rng["data_generator"]):
        raise RelationTrainingEvidenceRefused("R4 sampler/private RNG continuation disagrees")
    _exact_mapping(bundle.get("content_digests"), {"model", "optimizer", "rng"}, "content_digests")
    if not all(_sha256(digest) for digest in bundle["content_digests"].values()):
        raise RelationTrainingEvidenceRefused("R4 bundle content digests are malformed")
    _validate_environment(bundle["environment"])
    _validate_checkpoint_state_shape(bundle)
    _validate_diagnostics_shape(bundle.get("diagnostics"), config)
    _reconcile_continuation(bundle)
    _reconcile_diagnostics(bundle)


def _validate_config(config: object) -> None:
    _exact_mapping(config, set(CONFIG_KEYS), "config")
    assert isinstance(config, Mapping)
    if not isinstance(config["model"], dict) or not config["model"] or \
            not all(type(config[name]) is int and not isinstance(config[name], bool) for name in
                    ("seed", "batch_size", "max_len", "attention_budget", "steps", "warmup",
                     "max_cached_cases", "max_cached_spans", "max_cached_bytes")) or \
            not all(_number(config[name]) for name in
                    ("lr", "weight_decay", "grad_clip", "adam_eps")) or \
            not all(isinstance(config[name], str) for name in
                    ("arm", "device", "artifact_output_policy", "corpus_provenance",
                     "seed_namespace", "schedule")):
        raise RelationTrainingEvidenceRefused("R4 bundle config values are malformed")
    if not (0 <= config["seed"] < 2**63 - 1):
        raise RelationTrainingEvidenceRefused("R4 bundle seed is outside the generator domain")
    if not (1 <= config["steps"] < 2**63 - 1) or not (0 <= config["warmup"] < config["steps"]):
        raise RelationTrainingEvidenceRefused("R4 bundle schedule horizon/warmup is invalid")
    if not (1 <= config["batch_size"] < 2**63 - 1) or not (2 <= config["max_len"] < 2**63 - 1) or not (1 <= config["attention_budget"] < 2**63 - 1):
        raise RelationTrainingEvidenceRefused("R4 bundle batch/max-length/budget shape is invalid")
    if any(not (0 <= config[name] < 2**63 - 1) for name in ("max_cached_cases", "max_cached_spans", "max_cached_bytes")):
        raise RelationTrainingEvidenceRefused("R4 config cache limits must be in [0, 2**63 - 1)")
    if type(config["deterministic"]) is not bool:
        raise RelationTrainingEvidenceRefused("R4 bundle deterministic policy is malformed")
    if config["arm"] not in ("none", "span_affine_v1"):
        raise RelationTrainingEvidenceRefused("R4 bundle arm is unknown")
    if config["schedule"] != SCHEDULE_IDENTITY:
        raise RelationTrainingEvidenceRefused("R4 bundle schedule identity is not the declared rule")
    for name in ("lr", "weight_decay", "grad_clip", "adam_eps"):
        if not _finite_nonnegative(config[name]):
            raise RelationTrainingEvidenceRefused(f"R4 config {name} is not finite/nonnegative")
    if config["lr"] == 0 or config["grad_clip"] == 0 or config["adam_eps"] == 0:
        raise RelationTrainingEvidenceRefused("R4 config lr/grad_clip/adam_eps must be positive")
    betas = config["adam_betas"]
    if type(betas) is not tuple or len(betas) != 2 or betas != (0.9, 0.95) or \
            any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value < 1
                for value in betas) or \
            any(config[name] is not False for name in (
                "adam_amsgrad", "adam_capturable", "adam_differentiable", "adam_maximize")) or \
            config["adam_foreach"] is not None or config["adam_fused"] is not None:
        raise RelationTrainingEvidenceRefused("R4 bundle AdamW recipe is not the frozen configuration")
    if min(config["max_cached_cases"], config["max_cached_spans"], config["max_cached_bytes"]) < 0:
        raise RelationTrainingEvidenceRefused("R4 config cache limits must be nonnegative")
    validate_provenance_and_seed(config["corpus_provenance"], config["seed_namespace"], config["seed"])
    events = config["component_gradient_events"]
    if events is not None and (type(events) is not tuple or
                               any(type(step) is not int for step in events) or
                               list(events) != sorted(set(events)) or
                               any(not 1 <= step <= config["steps"] for step in events)):
        raise RelationTrainingEvidenceRefused("R4 bundle explicit component events are malformed")


def _validate_sampler(sampler: object) -> None:
    _exact_mapping(sampler, set(SAMPLER_KEYS), "sampler")
    assert isinstance(sampler, Mapping)
    if not isinstance(sampler["lengths"], list) or not isinstance(sampler["batches"], list):
        raise RelationTrainingEvidenceRefused("R4 sampler lengths/batches are malformed")
    if not sampler["lengths"] or not all(type(length) is int and not isinstance(length, bool) and length > 0 for length in sampler["lengths"]):
        raise RelationTrainingEvidenceRefused("R4 sampler lengths must be a nonempty list of positive integers")
    if any(type(sampler[name]) is not int or isinstance(sampler[name], bool) for name in
           ("batch_size", "base_seed", "epoch", "cursor", "pool_batches")):
        raise RelationTrainingEvidenceRefused("R4 sampler values are malformed")
    if not _byte_tensor(sampler["generator"]) or sampler["algorithm"] != BUCKET_EPOCH_ALGORITHM:
        raise RelationTrainingEvidenceRefused("R4 sampler values are malformed")
    if sampler["batch_size"] < 1 or sampler["epoch"] < 0 or \
            not 0 <= sampler["cursor"] <= len(sampler["batches"]) or \
            not all(isinstance(batch, list) and all(type(index) is int and not isinstance(index, bool) for index in batch)
                    for batch in sampler["batches"]):
        raise RelationTrainingEvidenceRefused("R4 sampler cursor state is malformed")
    if not (0 <= sampler["base_seed"] < 2**63 - 1):
        raise RelationTrainingEvidenceRefused("R4 sampler base_seed is outside the generator domain")
    if sampler["pool_batches"] != 50:
        raise RelationTrainingEvidenceRefused("R4 sampler pool_batches must be 50")


def _validate_factor_record(record: object, arm: str, positives: int) -> None:
    _exact_mapping(record, set(FACTOR_KEYS), "factor record")
    assert isinstance(record, Mapping)
    if record["unit"] != "nats":
        raise RelationTrainingEvidenceRefused("R4 factor diagnostics must be in nats")
    if arm == "none":
        if record != {"status": STATUS_NOT_APPLICABLE, "unit": "nats", "denominator": None,
                      "sums": None}:
            raise RelationTrainingEvidenceRefused("R4 none-arm factor record must be not_applicable")
        return
    if type(positives) is not int or isinstance(positives, bool) or not (0 <= positives <= 2**31 - 1):
        raise RelationTrainingEvidenceRefused("R4 factor positives must be an integer in [0, 2**31 - 1)")
    sums = record["sums"]
    if not isinstance(sums, dict) or set(sums) != set(FACTOR_NAMES) or \
            not all(type(value) is float for value in sums.values()) or \
            type(record["denominator"]) is not int or isinstance(record["denominator"], bool) or \
            record["denominator"] != positives:
        raise RelationTrainingEvidenceRefused("R4 factor record shape/denominator is invalid")
    if positives == 0:
        if record["status"] != STATUS_NO_POSITIVES or any(value != 0.0 for value in sums.values()):
            raise RelationTrainingEvidenceRefused(
                "R4 factor record without positives must be exact zeros with no_positive_boundaries")
        return
    if record["status"] != STATUS_MEASURED:
        raise RelationTrainingEvidenceRefused("R4 factor record with positives must be measured")
    floor = -FACTOR_ALLOWANCE_PER_QUERY * positives
    if any(not math.isfinite(value) or value < floor for value in sums.values()):
        raise RelationTrainingEvidenceRefused(
            "R4 factor sum is nonfinite or below the declared roundoff allowance")


def _validate_diagnostics_shape(diagnostics: object, config: Mapping) -> None:
    _exact_mapping(diagnostics, set(DIAGNOSTIC_KEYS), "diagnostics")
    assert isinstance(diagnostics, Mapping)
    explicit = config["component_gradient_events"]
    expected_policy = DIAGNOSTIC_POLICY_SCHEDULE if explicit is None else DIAGNOSTIC_POLICY_EXPLICIT
    expected_events = (list(default_component_events(config["steps"], config["warmup"]))
                       if explicit is None else list(explicit))
    if diagnostics["component_policy"] != expected_policy or \
            diagnostics["component_events"] != expected_events:
        raise RelationTrainingEvidenceRefused("R4 diagnostic cadence disagrees with its declared policy")
    trunk = diagnostics["trunk"]
    _exact_mapping(trunk, set(TRUNK_KEYS), "trunk identity")
    if not isinstance(trunk["names"], list) or not trunk["names"] or \
            not all(isinstance(name, str) for name in trunk["names"]) or \
            len(set(trunk["names"])) != len(trunk["names"]) or \
            not isinstance(trunk["shapes"], list) or len(trunk["shapes"]) != len(trunk["names"]) or \
            not all(isinstance(shape, list) and all(type(dim) is int and dim > 0 for dim in shape)
                    for shape in trunk["shapes"]) or \
            not isinstance(trunk["dtype"], str) or type(trunk["parameters"]) is not int or \
            not _sha256(trunk["digest"]):
        raise RelationTrainingEvidenceRefused("R4 trunk identity is malformed")
    if any(name.startswith("relation.") for name in trunk["names"]) or \
            trunk["parameters"] != sum(math.prod(shape) for shape in trunk["shapes"]):
        raise RelationTrainingEvidenceRefused("R4 trunk identity includes relation tensors or a wrong count")
    if not isinstance(diagnostics["component_records"], list):
        raise RelationTrainingEvidenceRefused("R4 component records must be a list")
    for record in diagnostics["component_records"]:
        _exact_mapping(record, set(COMPONENT_RECORD_KEYS), "component record")
        if type(record["step"]) is not int or not isinstance(record["batch_ids"], list) or \
                record["trunk_digest"] != trunk["digest"] or \
                type(record["extra_backward_calls"]) is not int or record["extra_backward_calls"] < 0 or \
                type(record["buffer_bytes"]) is not int or record["buffer_bytes"] < 0:
            raise RelationTrainingEvidenceRefused("R4 component record header is malformed")
        _exact_mapping(record["components"], set(COMPONENT_NAMES), "component set")
        for name in COMPONENT_NAMES:
            item = record["components"][name]
            _exact_mapping(item, set(COMPONENT_KEYS), f"component {name}")
            if item["weight"] != COMPONENT_WEIGHTS[name] or type(item["weight"]) is not float:
                raise RelationTrainingEvidenceRefused(f"R4 component {name} weight is not its fixed value")
            if item["status"] == STATUS_MEASURED:
                if type(item["norm"]) is not float or not math.isfinite(item["norm"]) or \
                        item["norm"] < 0 or type(item["denominator"]) is not int or item["denominator"] < 1:
                    raise RelationTrainingEvidenceRefused(
                        f"R4 measured component {name} needs a finite norm and positive denominator")
            elif item["status"] == STATUS_NO_POSITIVES:
                if item["norm"] is not None or item["denominator"] != 0 or name != "action_joint":
                    raise RelationTrainingEvidenceRefused(
                        "R4 no_positive_boundaries applies to action_joint with null norm only")
            elif item["status"] == STATUS_NOT_APPLICABLE:
                if item["norm"] is not None or item["denominator"] is not None or name == "byte":
                    raise RelationTrainingEvidenceRefused(
                        "R4 not_applicable components carry null norm/denominator and exclude byte")
            else:
                raise RelationTrainingEvidenceRefused(f"R4 component {name} status is unknown")


def _reconcile_continuation(bundle: Mapping) -> None:
    """Rebuild accumulated readings, schedule and batch-order identities from raw history."""
    accounting, history, sampler = bundle["accounting"], bundle["history"], bundle["sampler"]
    config = bundle["config"]
    if not all(_finite_nonnegative(value) for value in accounting.values()):
        raise RelationTrainingEvidenceRefused("R4 accounting must be finite and nonnegative")
    counts = {"content_bytes", "reachable_boundaries", "positive_boundaries"}
    schedule = SimpleNamespace(lr=config["lr"], warmup=config["warmup"], steps=config["steps"])
    for record in history:
        if any(type(record[name]) is not int or isinstance(record[name], bool) or record[name] < 0 for name in counts) or any(
                not _finite_nonnegative(record[name]) for name in ("byte_sum", "gate_sum", "joint_sum")):
            raise RelationTrainingEvidenceRefused("R4 history contains invalid numeric readings")
        if not record["batch_ids"] or len(set(record["batch_ids"])) != len(record["batch_ids"]):
            raise RelationTrainingEvidenceRefused("R4 history contains an empty/duplicate batch")
        if record["content_bytes"] < 1 or record["reachable_boundaries"] < 1:
            raise RelationTrainingEvidenceRefused("R4 history content_bytes and reachable_boundaries must be positive")
        if not (0 <= record["positive_boundaries"] <= record["reachable_boundaries"] <= record["content_bytes"]):
            raise RelationTrainingEvidenceRefused("R4 history denominator ordering is invalid")
        if type(record["lr"]) is not float or record["lr"] != lr_at(record["step"] - 1, schedule):
            raise RelationTrainingEvidenceRefused("R4 history learning rate is not the declared schedule value")
    identities = {
        "programs": sum(len(record["batch_ids"]) for record in history),
        "semantic_bytes": sum(record["content_bytes"] for record in history),
        "content_symbols": sum(record["content_bytes"] for record in history),
        "byte_denominator": sum(record["content_bytes"] for record in history),
        "gate_denominator": sum(record["reachable_boundaries"] for record in history),
        "reachable_queries": sum(record["reachable_boundaries"] for record in history),
        "positive_denominator": sum(record["positive_boundaries"] for record in history),
        **{name: sum(record[name] for record in history)
           for name in ("byte_sum", "gate_sum", "joint_sum")},
    }
    if any(accounting[name] != value for name, value in identities.items()):
        raise RelationTrainingEvidenceRefused("R4 accounting disagrees with raw history")
    expected_padded = sum(
        len(record["batch_ids"]) * max(sampler["lengths"][c] - 1 for c in record["batch_ids"])
        for record in history
    )
    if accounting["padded_positions"] != expected_padded:
        raise RelationTrainingEvidenceRefused("R4 accounting padded_positions disagrees with batch geometry")
    # Optimizer groups carry the last used LR: lr_at(0) before any update, else lr_at(k-1).
    groups = bundle["optimizer"].get("param_groups") if isinstance(bundle["optimizer"], Mapping) else None
    last_used = lr_at(max(0, len(history) - 1), schedule)
    if not isinstance(groups, list) or not groups or \
            any(not isinstance(group, Mapping) or group.get("lr") != last_used for group in groups):
        raise RelationTrainingEvidenceRefused("R4 optimizer group learning rate disagrees with the schedule")
    # Sampler position and every consumed batch are replayed through the single
    # historical epoch algorithm, one epoch plan at a time.
    cases, batch_size = len(sampler["lengths"]), sampler["batch_size"]
    per_epoch = (cases + batch_size - 1) // batch_size
    if sampler["batch_size"] != config["batch_size"] or sampler["base_seed"] != config["seed"]:
        raise RelationTrainingEvidenceRefused("R4 sampler shape/seed disagrees with the configuration")
    epoch, cursor = sampler["epoch"], sampler["cursor"]
    if epoch == 0:
        if cursor or sampler["batches"] or history:
            raise RelationTrainingEvidenceRefused("R4 epoch-zero sampler cannot have consumed anything")
        expected_state = torch.Generator().manual_seed(sampler["base_seed"]).get_state()
    else:
        if len(sampler["batches"]) != per_epoch or (epoch - 1) * per_epoch + cursor != len(history):
            raise RelationTrainingEvidenceRefused("R4 sampler cursor disagrees with completed history")
        plan, expected_state = bucket_epoch_plan(sampler["lengths"], batch_size, epoch=epoch - 1,
                                                 seed=sampler["base_seed"],
                                                 pool_batches=sampler["pool_batches"])
        if sampler["batches"] != plan:
            raise RelationTrainingEvidenceRefused("R4 sampler batches are not the historical epoch order")
    if not torch.equal(sampler["generator"], expected_state):
        raise RelationTrainingEvidenceRefused("R4 sampler private RNG record disagrees with its epoch")
    for start in range(0, len(history), per_epoch):
        plan, _ = bucket_epoch_plan(sampler["lengths"], batch_size, epoch=start // per_epoch,
                                    seed=sampler["base_seed"], pool_batches=sampler["pool_batches"])
        consumed = [record["batch_ids"] for record in history[start:start + per_epoch]]
        if consumed != plan[:len(consumed)]:
            raise RelationTrainingEvidenceRefused(
                "R4 history batches disagree with their canonical epoch positions")


def _reconcile_diagnostics(bundle: Mapping) -> None:
    """Expected events are exactly the declared steps not exceeding completion."""
    diagnostics, history, arm = bundle["diagnostics"], bundle["history"], bundle["config"]["arm"]
    expected = [step for step in diagnostics["component_events"] if step <= bundle["completed_step"]]
    records = diagnostics["component_records"]
    if [record["step"] for record in records] != expected:
        raise RelationTrainingEvidenceRefused("R4 component records do not match the declared events")
    trunk_params = diagnostics["trunk"]["parameters"]
    lengths = bundle["sampler"]["lengths"]
    budget = bundle["config"]["attention_budget"]

    for record in records:
        row = history[record["step"] - 1]
        if record["batch_ids"] != row["batch_ids"]:
            raise RelationTrainingEvidenceRefused("R4 component record does not own its step's batch")
        components = record["components"]
        if components["byte"]["status"] != STATUS_MEASURED or \
                components["byte"]["denominator"] != row["content_bytes"]:
            raise RelationTrainingEvidenceRefused("R4 byte component must be measured over content bytes")

        batch_ids = record["batch_ids"]
        t_width = max(lengths[c] - 1 for c in batch_ids)
        rows_per_chunk = max(1, min(len(batch_ids), budget // max(1, t_width ** 2)))
        m_chunks = (len(batch_ids) + rows_per_chunk - 1) // rows_per_chunk

        if arm == "none":
            if any(components[name]["status"] != STATUS_NOT_APPLICABLE
                   for name in ("gate", "action_joint")):
                raise RelationTrainingEvidenceRefused("R4 none-arm action components must be not_applicable")
            if record["extra_backward_calls"] != m_chunks:
                raise RelationTrainingEvidenceRefused(
                    f"R4 none-arm event needs {m_chunks} backward calls, got {record['extra_backward_calls']}")
            max_buffer_bytes = 4 * trunk_params * 1
        else:
            if components["gate"]["status"] != STATUS_MEASURED or \
                    components["gate"]["denominator"] != row["reachable_boundaries"]:
                raise RelationTrainingEvidenceRefused("R4 gate component must be measured over reachable boundaries")
            joint = components["action_joint"]
            if row["positive_boundaries"] == 0:
                if joint["status"] != STATUS_NO_POSITIVES:
                    raise RelationTrainingEvidenceRefused("R4 joint component without positives must be null")
                if record["extra_backward_calls"] != 2 * m_chunks:
                    raise RelationTrainingEvidenceRefused(
                        f"R4 negative-only relation event needs {2 * m_chunks} backward calls, got {record['extra_backward_calls']}")
                max_buffer_bytes = 4 * trunk_params * 2
            else:
                if joint["status"] != STATUS_MEASURED or joint["denominator"] != row["positive_boundaries"]:
                    raise RelationTrainingEvidenceRefused("R4 joint component must be measured over positives")
                if not (2 * m_chunks + 1 <= record["extra_backward_calls"] <= 3 * m_chunks):
                    raise RelationTrainingEvidenceRefused(
                        f"R4 positive relation event needs between {2 * m_chunks + 1} and {3 * m_chunks} backward calls, got {record['extra_backward_calls']}")
                max_buffer_bytes = 4 * trunk_params * 3

        if record["buffer_bytes"] < 0 or record["buffer_bytes"] > max_buffer_bytes:
            raise RelationTrainingEvidenceRefused(
                f"R4 component record buffer_bytes {record['buffer_bytes']} exceeds bound {max_buffer_bytes}")


def _validate_environment(environment: object) -> None:
    """Validate requested and observed process facts without substituting either."""
    _exact_mapping(environment, {"identity", "requested", "observed"}, "execution environment")
    assert isinstance(environment, Mapping)
    if environment["identity"] != "r4_process_runtime_v1":
        raise RelationTrainingEvidenceRefused("R4 execution environment version is unknown")
    _exact_mapping(environment["requested"], {"deterministic_algorithms"},
                   "execution environment requested policy")
    requested = environment["requested"]
    assert isinstance(requested, Mapping)
    if requested["deterministic_algorithms"] is not True:
        raise RelationTrainingEvidenceRefused("R4 requested deterministic policy is not enabled")
    _exact_mapping(environment["observed"], {
        "python", "numpy", "torch", "torch_git_version", "platform", "device",
        "deterministic_algorithms_enabled", "deterministic_algorithms_warn_only",
        "torch_num_threads", "torch_num_interop_threads", "torch_default_dtype",
        "mkldnn_enabled", "environment",
    }, "execution environment observed state")
    observed = environment["observed"]
    assert isinstance(observed, Mapping)
    if not all(isinstance(observed[name], str) for name in (
            "python", "numpy", "torch", "platform", "device", "torch_default_dtype")) or \
            observed["torch_git_version"] is not None and not isinstance(
                observed["torch_git_version"], str) or \
            observed["deterministic_algorithms_enabled"] is not True or \
            type(observed["deterministic_algorithms_warn_only"]) is not bool or \
            type(observed["torch_num_threads"]) is not int or \
            type(observed["torch_num_interop_threads"]) is not int or \
            observed["torch_num_threads"] < 1 or observed["torch_num_interop_threads"] < 1 or \
            type(observed["mkldnn_enabled"]) is not bool or \
            not isinstance(observed["environment"], Mapping) or \
            not all(type(name) is str and (value is None or type(value) is str)
                    for name, value in observed["environment"].items()):
        raise RelationTrainingEvidenceRefused("R4 observed execution environment is malformed")


def _finite_tensor(tensor: torch.Tensor) -> bool:
    if tensor.layout is not torch.strided:
        return False
    return not (tensor.is_floating_point() or tensor.is_complex()) or bool(torch.isfinite(tensor).all())


def _finite_state_tree(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return _finite_tensor(value)
    if isinstance(value, Mapping):
        return all(_finite_state_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_state_tree(item) for item in value)
    return type(value) not in (float,) or math.isfinite(value)


def _validate_checkpoint_state_shape(bundle: Mapping) -> None:
    """Reject rehashed NaN/Inf state before any model or optimizer mutation."""
    model = bundle.get("model")
    if not isinstance(model, Mapping) or not model or \
            not all(type(name) is str and isinstance(value, torch.Tensor)
                    for name, value in model.items()) or not _finite_state_tree(model):
        raise RelationTrainingEvidenceRefused("R4 checkpoint model tensors are malformed or nonfinite")
    optimizer = bundle.get("optimizer")
    if not isinstance(optimizer, Mapping) or set(optimizer) != {"state", "param_groups"} or \
            not isinstance(optimizer["state"], Mapping) or \
            not isinstance(optimizer["param_groups"], list) or not optimizer["param_groups"] or \
            not _finite_state_tree(optimizer):
        raise RelationTrainingEvidenceRefused(
            "R4 checkpoint optimizer state is malformed or nonfinite")


def _finite_nonnegative(value: object) -> bool:
    if type(value) not in (int, float) or value < 0:
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _number(value: object) -> bool:
    return type(value) in (int, float)


def _exact_mapping(value: object, keys: set[str], name: str, *,
                   allow_extra: set[str] = frozenset()) -> None:
    if not isinstance(value, Mapping) or set(value) - allow_extra != keys:
        raise RelationTrainingEvidenceRefused(f"R4 bundle {name} key set is not exact")


def _byte_tensor(value: object) -> bool:
    return isinstance(value, torch.Tensor) and value.dtype is torch.uint8 and value.ndim == 1


def _sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)
