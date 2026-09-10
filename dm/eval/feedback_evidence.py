"""What a qualification cell *is*: corpus, source, environment, reconstruction.

**Why this Module exists at all.** F5b's six cells passed a corpus clause that
compared a manifest to itself. The Adapter hashed the manifest named on the
command line, wrote the two digests into `hashes["corpus"]` under `recorded` and
`recomputed`, and the gate found them equal -- as it always would, because
nothing ever compared either of them to the corpus the run actually built. The
cells trained on `data_seed=100` and filed a `data_seed=0` manifest, and a fully
green suite did not notice.

That was not an oversight in one function. It was a *seam*: corpus identity,
source identity, execution environment, checkpoint reconstruction and report
validation were spread across the Adapter and the contract, each half-owned, and
an invariant that lives in two places is an invariant nobody owns. This Module
takes all five. The Adapter reads arguments, trains, measures and writes files;
the gate applies clauses to what it is handed; and what a cell *is* -- which
bytes it names and how those names are checked -- is here, once.

**Every identity here is a comparison, never a claim.** A digest a program
computes and immediately compares against itself proves nothing
(`docs/directions.md` §7 invariant 11), so each function returns both sides and
the difference between them, and leaves the verdict to
`dm.eval.feedback_qualification`. Missing means `incomplete`; it never means
absent-and-therefore-fine.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from collections.abc import Mapping, Sequence
from dataclasses import fields
from pathlib import Path

import torch

from ..data import feedback as corpus_module
from ..models.transformer import Config, DrawingLM
from ..train import TrainConfig
from .feedback import DEVELOPMENT
from .feedback_contract import (
    QUALIFICATION_FEEDBACK_SCHEMA,
    QUALIFICATION_SOURCES,
    RECORDED_ENVIRONMENT_VARIABLES,
    REQUIRED_ENVIRONMENT_FIELDS,
    TRAINING_STRUCTURES,
)
from .provenance import canonical_digest, combined_source_digest, sha256_file, source_hashes

# ---------------------------------------------------------------------------
# corpus identity


def corpus_identity(manifest_path: Path, record: Mapping, *,
                    structure: str = "relational") -> dict:
    """Reconcile a frozen corpus manifest against the corpus a run really built.

    Two different things carry the word "corpus" in this project and F5b
    conflated them. A manifest's digests identify *the manifest file*: the
    canonical payload digest over its sorted JSON body, and the SHA-256 of the
    bytes on disk. A record's `corpus` is `dm.data.fingerprint.fingerprint` over
    the programs the run actually trained and validated on -- the corpus itself,
    at 8 bytes a half. Equal manifest digests say a manifest was not edited. They
    say nothing whatever about which programs a checkpoint saw.

    So the reconciliation that matters is the *fingerprint* one, and it is exact
    rather than toleranced: both sides are `blake2b` over length-prefixed
    programs in order, computed by one function, so any difference at all means
    the run and the manifest describe different corpora. The audited manifest
    carries a fingerprint per arm, and the arm is named rather than assumed --
    the relational and relation-destroyed splits share a validation half and
    differ only in train, so reading the wrong one would compare a control's
    identity to a relational run's and find the val halves agreeing.

    `data_seed` and the split sizes are reported beside it. They cannot
    contradict a matching fingerprint, and when the fingerprint *does* differ
    they are what says why.
    """
    if structure not in TRAINING_STRUCTURES:
        raise ValueError(
            f"unknown training structure {structure!r}; expected one of "
            f"{list(TRAINING_STRUCTURES)}"
        )
    manifest = corpus_module.load_manifest(manifest_path)
    digests = corpus_module.manifest_digests(manifest_path)
    balance = manifest.get("balance") or {}
    declared = (balance.get("corpus") or {}).get(structure)
    observed = record.get("corpus")
    config = record.get("config") or {}

    differences: list[str] = []
    if not isinstance(declared, dict):
        differences.append(f"manifest carries no {structure!r} fingerprint")
    if not isinstance(observed, dict):
        differences.append("record carries no corpus fingerprint")
    if isinstance(declared, dict) and isinstance(observed, dict):
        for key in sorted(set(declared) | set(observed)):
            if declared.get(key) != observed.get(key):
                differences.append(
                    f"{key}: manifest {declared.get(key)!r} != record "
                    f"{observed.get(key)!r}"
                )
    if balance.get("data_seed") != config.get("data_seed"):
        differences.append(
            f"data_seed: manifest {balance.get('data_seed')!r} != record "
            f"{config.get('data_seed')!r}"
        )
    return {
        "manifest": str(manifest_path),
        "structure": structure,
        "canonical_payload_sha256": digests["canonical_payload_sha256"],
        "recorded_canonical_sha256": digests["recorded_canonical_sha256"],
        "file_sha256": digests["file_sha256"],
        "manifest_fingerprint": declared,
        "record_fingerprint": observed,
        "manifest_data_seed": balance.get("data_seed"),
        "record_data_seed": config.get("data_seed"),
        "manifest_derangement_seed": balance.get("seed"),
        "accepted": corpus_module.accepts(balance),
        "census_accepted": corpus_module.audit_accepts(manifest.get("vm_census") or {}),
        "matches": not differences,
        "differences": differences,
    }


# ---------------------------------------------------------------------------
# source identity


def source_identity() -> dict:
    """Hash every file that trains *or* judges a cell, from the one declared set.

    `dm.eval.feedback_contract.QUALIFICATION_SOURCES` is that set and this is its
    only reader. F5b kept a second, shorter list in the Adapter -- eight files
    against the contract's twelve -- so `dm/data/dataset.py`, `dm/data/synthetic.py`,
    `dm/isa/codec.py` and `dm/isa/unroll.py` determined the batching, the corpus
    bytes and the encoding of every cell without appearing in any cell's source
    digest. A duplicated list is not provenance; it is two provenances, and the
    shorter one wins silently.
    """
    hashes = source_hashes([Path(path) for path in QUALIFICATION_SOURCES])
    return {"files": hashes, "combined": combined_source_digest(hashes)}


# ---------------------------------------------------------------------------
# execution environment


def execution_environment(*, device: str | torch.device) -> dict:
    """The machine state a reproducibility claim is actually about.

    F5b recorded none of this, which is why `docs/feedback-stability.md` §3d's
    two disagreeing runs of one configuration cannot be attributed to anything:
    the device, the determinism flags, the thread counts and the handful of
    environment variables that change numerics without changing code were all
    unrecorded, so "identical training and measurement code" is as far as that
    observation can go.

    Every declared field is always present. A variable that is unset is recorded
    as `None` rather than omitted, because "unset" and "never looked at" are
    different facts and the fail-closed rule treats them differently.
    """
    body = {
        "python": sys.version,
        "python_implementation": platform.python_implementation(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "device": str(device),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "deterministic_algorithms_warn_only": bool(
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        "environment_variables": {
            name: os.environ.get(name) for name in RECORDED_ENVIRONMENT_VARIABLES
        },
    }
    missing = [name for name in REQUIRED_ENVIRONMENT_FIELDS if name not in body]
    if missing:
        raise ValueError(
            f"the execution environment is missing {missing}, which the contract "
            "requires: incomplete"
        )
    return body


def deterministic_execution() -> dict:
    """Ask torch for deterministic kernels, and report what it granted.

    Requested rather than assumed, and reported rather than asserted. Some
    backends have no deterministic implementation of some operator, and on those
    this call raises at the *operator* rather than here -- so the honest artifact
    is "determinism was requested, here is the state that resulted", checked
    afterwards against whether the two runs of the repeated seed agreed.

    `warn_only=False`: a run that quietly fell back to a nondeterministic kernel
    is exactly the run whose repeat would disagree for a reason nobody could
    find afterwards.
    """
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return {
        "requested": True,
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }


# ---------------------------------------------------------------------------
# reconstruction


def state_dict_sha256(checkpoint_path: Path) -> str:
    """Hash model-state content, excluding container and run metadata.

    This is deliberately *not* the checkpoint file SHA-256. ``torch.save``
    writes a ZIP container whose member names depend on the output filename,
    and this project's development checkpoints additionally carry a
    replicate-specific ``record_name``. Two byte-identical state dictionaries
    can therefore produce different checkpoint files. A whole-file digest is
    artifact identity; it is not a valid reproducibility comparison.

    The digest frames the schema marker, sorted tensor name, dtype, shape and raw
    contiguous CPU bytes for every state entry. Config and environment equality
    remain separate clauses: this function answers only whether the learned
    model state is identical.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state")
    if not isinstance(state, Mapping):
        raise TypeError(f"{checkpoint_path} carries no state dictionary")

    invalid = [name for name, tensor in state.items()
               if not isinstance(name, str) or not isinstance(tensor, torch.Tensor)]
    if invalid:
        raise TypeError(
            f"{checkpoint_path} state entry {invalid[0]!r} is not a named tensor"
        )

    digest = hashlib.sha256()

    def frame(payload: bytes) -> None:
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    frame(b"drawing-machine-state-dict-v1")
    for name in sorted(state):
        tensor = state[name]
        if tensor.layout != torch.strided:
            raise ValueError(
                f"{checkpoint_path} state entry {name!r} has unsupported "
                f"layout {tensor.layout}"
            )
        value = tensor.detach().cpu().contiguous()
        frame(name.encode("utf-8"))
        frame(str(value.dtype).encode("ascii"))
        frame(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        frame(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def reconstruct(checkpoint_path: Path, *, device: str | torch.device = "cpu",
                feedback_schema: str = QUALIFICATION_FEEDBACK_SCHEMA) -> tuple[DrawingLM, dict]:
    """Rebuild a checkpoint from its own config and load it `strict=True`.

    The load that matters is today's `Config` accepting yesterday's state
    dictionary, and only an actual load can perform it: a boolean an artifact
    wrote about itself is a claim (`PLAN.md` invariant 10). The model comes back
    with the evidence, because the caller needs the same object the check ran on
    -- reloading twice would leave the guards measuring a model no clause saw.

    The arm is checked here rather than trusted. All three fusion arms own
    byte-identical tensor sets, so a checkpoint written under one loads strict
    under any of them and computes a different function without complaint; the
    `strict` load is therefore necessary and nowhere near sufficient.
    """
    weights = torch.load(checkpoint_path, weights_only=False)
    model = DrawingLM(Config(**weights["cfg"]))
    model.load_state_dict(weights["state"], strict=True)
    model = model.to(device).eval()
    return model, {
        "strict": True,
        "feedback_schema": weights["cfg"].get("feedback_schema"),
        "params_match_config": model.n_params() == model.cfg.n_params(),
        "provenance": weights.get("provenance"),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "expected_feedback_schema": feedback_schema,
    }


# ---------------------------------------------------------------------------
# hashes


def hash_pair(recorded: str | None, recomputed: str | None,
              path: Path | None = None) -> dict:
    """A recorded digest beside the one this process computed. Never one alone.

    `path` travels with the pair so `decide` can recompute the file hash *in a
    later process*, which is the only version of this check with any content.
    """
    pair = {"recorded": recorded, "recomputed": recomputed}
    if path is not None:
        pair["path"] = str(path)
    return pair


def hash_evidence(*, record_path: Path, checkpoint_path: Path,
                  record: Mapping, protocol_body: Mapping,
                  protocol_digest: str, corpus: Mapping,
                  source: Mapping) -> dict:
    """Every artifact digest a cell carries, each as a recorded/recomputed pair.

    The corpus entry is the one that changed. It used to be the manifest's own
    digest compared against a recomputation of the same manifest, which is a
    tautology. It is now the corpus *fingerprint* the manifest declares against
    the one the record carries -- two independently produced statements about
    which programs exist -- and the manifest's file digests ride alongside under
    names that cannot be swapped.
    """
    return {
        "record": hash_pair(sha256_file(record_path), sha256_file(record_path),
                            record_path),
        "checkpoint": hash_pair(record.get("checkpoint_sha256"),
                                sha256_file(checkpoint_path), checkpoint_path),
        "protocol": hash_pair(protocol_body.get("protocol_sha256"), protocol_digest),
        "corpus": hash_pair(
            json.dumps(corpus.get("manifest_fingerprint"), sort_keys=True),
            json.dumps(corpus.get("record_fingerprint"), sort_keys=True),
        ),
        "source": hash_pair(source["combined"], source["combined"]),
    }


# ---------------------------------------------------------------------------
# what a cell may contain, at every level


def _config_fields() -> tuple[str, ...]:
    """`TrainConfig`'s own field names, derived rather than declared.

    `record["config"]` is `asdict(cfg)`, so deriving the allowed keys from the
    dataclass means the declaration cannot drift from the thing it describes --
    and a config key added for Direction 4 shows up here the moment it exists
    instead of the first time somebody reads a refused cell.
    """
    return tuple(sorted(field.name for field in fields(TrainConfig)))


#: What a training record may carry at its top level.  Declared rather than
#: derived because `dm.train.train` builds it literally, and pinned against a
#: real one-step feedback record by `tests/test_feedback_qualification.py` -- so
#: a new record field is a test failure here rather than a silent read there.
RECORD_FIELDS: tuple[str, ...] = (
    "best", "best_val_bits", "checkpoint_sha256", "complete", "config", "corpus",
    "feedback", "final", "history", "model", "name", "provenance", "schema",
    "steps_done", "train_lengths", "val_bits", "val_bytes", "val_lengths",
)

#: What the feedback accounting block may carry.
FEEDBACK_FIELDS: tuple[str, ...] = (
    "content_symbol_forward_passes", "content_symbols_seen", "device",
    "expected_passes_per_batch", "feedback_schema", "gain_calibration",
    "gain_calibration_event", "observed_passes_per_batch",
    "padded_forward_positions", "padded_positions", "pass_histogram",
    "pass_schedule", "peak_memory_bytes", "programs_seen", "protocol", "schema",
    "semantic_bytes_seen", "stride", "wall_clock_s",
)

#: The nested shape a qualification cell is allowed to have.  A mapping from a
#: dotted path to the closed key set at that path; a path that is not named is
#: not descended into.
#:
#: **Why the shape and not only the top level.** F5b's gate refused an unexpected
#: *top-level* key and accepted arbitrary nesting underneath, so "structurally
#: blind to relation outcomes" was true of the artifacts that happened to exist
#: and was not a guarantee. A `Delta` inside `stability.report` would have been
#: read as data. The claim now costs what it is worth: every level a cell is
#: assembled from is declared, and an unnamed key at any of them is a refusal.
#:
#: `history`, `final`, `best` and the guard sub-blocks are deliberately *not*
#: enumerated -- they are eval payloads whose keys legitimately follow the corpus
#: and the sampler -- so what protects them is that nothing in this project ever
#: writes an outcome into `dm.train`'s eval dictionary, and the enclosing record
#: is closed against gaining a new one.
CELL_SHAPE: dict[str, tuple[str, ...]] = {
    "record": RECORD_FIELDS,
    "record.config": _config_fields(),
    "record.feedback": FEEDBACK_FIELDS,
    "reload": ("strict", "feedback_schema", "params_match_config", "provenance",
               "checkpoint_sha256", "expected_feedback_schema"),
    "generic_guards": ("passed", "failures", "limits", "values", "validation",
                       "generation", "generation_n", "generation_max_new",
                       "variates_shape"),
    "stability": ("report",),
    "stability.report": (
        "baseline_bits_per_drawing", "baseline_hidden_rms_p99", "cost_unit",
        "padded_positions", "passes", "programs", "schema", "scored_positions",
        "scored_positions_only", "standard_input_rms_p99", "subset",
        "symbols_per_byte", "valid_halt_loss", "channel_scales",
    ),
    "stability.report.subset": (
        "seed", "deepest_pass", "size", "eligible", "corpus_programs", "indices",
        "lengths", "digest",
    ),
    "hashes": ("record", "checkpoint", "protocol", "corpus", "source"),
    "hashes.record": ("recorded", "recomputed", "path"),
    "hashes.checkpoint": ("recorded", "recomputed", "path"),
    "hashes.protocol": ("recorded", "recomputed", "path"),
    "hashes.corpus": ("recorded", "recomputed", "path"),
    "hashes.source": ("recorded", "recomputed", "path"),
    "environment": REQUIRED_ENVIRONMENT_FIELDS + ("determinism",),
    "environment.environment_variables": RECORDED_ENVIRONMENT_VARIABLES,
    "environment.determinism": ("requested", "deterministic_algorithms",
                                "cudnn_deterministic", "cudnn_benchmark"),
    "corpus_identity": (
        "manifest", "structure", "canonical_payload_sha256",
        "recorded_canonical_sha256", "file_sha256", "manifest_fingerprint",
        "record_fingerprint", "manifest_data_seed", "record_data_seed",
        "manifest_derangement_seed", "accepted", "census_accepted", "matches",
        "differences",
    ),
    "source": ("files", "combined"),
}


def shape_faults(cell: Mapping, top_level: Sequence[str],
                 shape: Mapping[str, Sequence[str]] = CELL_SHAPE) -> dict[str, list[str]]:
    """Every key a cell carries that its declared shape does not name.

    Returns paths to unexpected keys rather than raising, so the caller decides
    whether an unexpected key is an error or a diagnosis -- and so one refusal
    can name all of them instead of the first.
    """
    faults: dict[str, list[str]] = {}

    def walk(node, path: str, allowed: Sequence[str]) -> None:
        unexpected = sorted(set(node) - set(allowed))
        if unexpected:
            faults[path or "cell"] = unexpected
        for key, value in node.items():
            child = f"{path}.{key}" if path else key
            if isinstance(value, Mapping) and child in shape:
                walk(value, child, shape[child])

    walk(cell, "", top_level)
    return faults


# ---------------------------------------------------------------------------
# assembly


def assemble_cell(*, package_name: str, schedule: str, seed: int, replicate: int,
                  record: Mapping, record_path: Path, checkpoint_path: Path,
                  manifest_path: Path, protocol_body: Mapping,
                  protocol_digest: str, reload: Mapping, generic_guards: Mapping,
                  stability_report: Mapping, environment: Mapping,
                  structure: str = "relational",
                  provenance: str = DEVELOPMENT) -> dict:
    """One cell, with every identity computed here rather than at the call site.

    The Adapter supplies what only it can -- the trained record, the measured
    guards, the stability report and the paths -- and this function supplies
    everything that is a *comparison*. That is the split `docs/directions.md` §7
    invariant 15 asks for, applied one level deeper than F5b applied it: the
    Adapter no longer decides what a corpus hash means.
    """
    corpus = corpus_identity(manifest_path, record, structure=structure)
    source = source_identity()
    return {
        "package": package_name,
        "schedule": schedule,
        "seed": seed,
        "replicate": replicate,
        "provenance": provenance,
        "record": record,
        "reload": dict(reload),
        "generic_guards": generic_guards,
        "stability": {"report": stability_report},
        "environment": environment,
        "corpus_identity": corpus,
        "source": source,
        "hashes": hash_evidence(
            record_path=record_path, checkpoint_path=checkpoint_path,
            record=record, protocol_body=protocol_body,
            protocol_digest=protocol_digest, corpus=corpus, source=source,
        ),
    }


# ---------------------------------------------------------------------------
# report validation


def verify_report(path: Path, *, provenance: str = DEVELOPMENT) -> dict:
    """Read one cell report and re-derive every identity it can no longer assert.

    Three checks, in the order that makes each one meaningful. The report must be
    the kind of artifact this stage may read at all; it must match its own
    canonical payload digest, recomputed here rather than trusted; and every hash
    whose artifact is still on disk is recomputed from the bytes, in *this*
    process, replacing what the report said about it. A digest a program computed
    and immediately compared against itself proves nothing, and this is the later
    process that makes the comparison real.

    Raises rather than returning a verdict: a report that does not match its own
    bytes is not evidence about a checkpoint, so there is nothing for a clause to
    weigh.
    """
    body = json.loads(path.read_text())
    if body.get("provenance") != provenance:
        raise ValueError(
            f"{path} is provenance {body.get('provenance')!r}, not {provenance!r}: "
            "refused"
        )
    recorded = body.get("report_sha256")
    recomputed = canonical_digest(body, "report_sha256")
    if not isinstance(recorded, str) or recorded != recomputed:
        raise ValueError(
            f"{path} does not match its own payload digest (recorded {recorded!r}, "
            f"recomputed {recomputed!r}): incomplete"
        )
    cell = body["cell"]
    for name, pair in cell["hashes"].items():
        location = pair.get("path")
        if location is None:
            continue
        artifact = Path(location)
        pair["recomputed"] = (sha256_file(artifact) if artifact.exists()
                              else f"missing:{name}")
    return {
        "body": body,
        "cell": cell,
        "envelope": {
            "package": cell.get("package"),
            "schedule": cell.get("schedule"),
            "seed": cell.get("seed"),
            "replicate": cell.get("replicate"),
            "report": str(path),
            "recorded_report_sha256": recorded,
            "recomputed_report_sha256": recomputed,
            "file_sha256": sha256_file(path),
            "verified": True,
        },
    }


__all__ = [
    "CELL_SHAPE",
    "FEEDBACK_FIELDS",
    "RECORD_FIELDS",
    "assemble_cell",
    "corpus_identity",
    "deterministic_execution",
    "execution_environment",
    "hash_evidence",
    "hash_pair",
    "reconstruct",
    "shape_faults",
    "source_identity",
    "state_dict_sha256",
    "verify_report",
]
