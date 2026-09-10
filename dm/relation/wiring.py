"""Fixed R5 engineering wiring route. Preparation is not launch approval.

R4 temporary bundles are an internal continuation codec only. Persistent R5
checkpoints have their own envelope and cannot be loaded by the R4 public API.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import torch

from dm.eval.relation_evidence import (
    execution_environment,
    optimizer_digest,
    qualify,
    rng_digest,
    state_digest,
)
from dm.isa.codec import N_SPECIAL
from dm.relation.decoding import CONTENT_POLICY, STOPPING_POLICY
from dm.relation.wiring_spec import fixture as _fixture
from dm.relation.wiring_spec import training_config
from dm.train_relation import (
    AUTHORITATIVE_R4_SOURCES,
    SUPERVISION_IDENTITY_VERSION,
    RelationTrainer,
    TrainingCorpus,
    capture_rng_state,
    restore_rng_state,
)

ROOT = Path(__file__).resolve().parents[2]
NAMESPACE = "direction4-r5-wiring-smoke-v1"
ARMS = ("none", "span_affine_v1")
MODES = {"none": ("standard", "oracle_copy"),
         "span_affine_v1": ("standard", "predicted_copy", "oracle_copy")}
EVALUATIONS = (0, 28, 56)
UNIFORM_SEED = 59
R0_PATH = ROOT / "artifacts/relation/r4-r0-cpu-20260906.json"
R0_SHA256 = "3186c7c76d6b47bb149fb551f0b81e75e86028eeee74dd70f7a83fc6d08bd59a"


class WiringRefused(ValueError):
    """The fixed engineering route or its evidence failed preflight."""


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def sources():
    paths = {*AUTHORITATIVE_R4_SOURCES, Path(__file__).resolve(),
             ROOT / "scripts/relation_wiring.py",
             ROOT / "dm/relation/wiring_spec.py"}
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


def validate_r0():
    try:
        raw = R0_PATH.read_bytes()
        report = json.loads(raw)
        decision = qualify(report["repeatability"], report["resume"])
        if decision.get("qualified") is not True or encoded(decision) != encoded(report["decision"]):
            raise WiringRefused("R0 raw evidence does not qualify")
        if hashlib.sha256(raw).hexdigest() != R0_SHA256:
            raise WiringRefused("frozen R0 artifact identity differs")
        if execution_environment(device="cpu") != report["repeatability"]["environment"]:
            raise WiringRefused("R0 environment differs; sentinel requalification required")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, WiringRefused):
            raise
        raise WiringRefused("invalid R0 qualification evidence") from exc
    return {"sha256": R0_SHA256, "route": decision["route"]}


def config(arm):
    return training_config(arm, 24_000_000, steps=56, warmup=7)


def uniforms(case):
    # Exact float32 values; key is case identity + absolute emitted-byte position.
    values = []
    for position in range(case.target_stop):
        key = f"{NAMESPACE}:{UNIFORM_SEED}:{case.case_id}:{position}".encode()
        integer = int.from_bytes(hashlib.sha256(key).digest()[:3], "big")
        values.append(integer / 2**24)
    return values


def run_card(run_id):
    if type(run_id) is not str or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_id) is None:
        raise WiringRefused("run id must be 1..64 ASCII letters/digits/underscore/hyphen")
    training = _fixture()
    rows = []
    for index, case in enumerate(training.cases):
        if not 0 < case.target_start < case.target_stop <= 256:
            raise WiringRefused("fixed fixture target does not fit declared decode capacity")
        values = uniforms(case)
        rows.append({"index": index, "case_id": case.case_id,
                     "prompt_bytes": case.target_start, "target_stop": case.target_stop,
                     "horizon_bytes": case.target_stop - case.target_start,
                     "target_hex": case.target.hex(), "action_targets": len(case.actions),
                     "uniform_sha256": hashlib.sha256(struct.pack(f"<{len(values)}f", *values)).hexdigest()})
    paths = ["run-card.json", "prepared.json", "launch.json", "updates.jsonl", "decodes.jsonl",
             "checkpoint-io.jsonl", "primary-summary.json", "terminal.json"]
    for arm in ARMS:
        paths.extend([f"{arm}-restart-28.pt", f"{arm}-final-56.pt",
                      f"{arm}-restart-updates.jsonl", f"{arm}-restart-result.json",
                      f"{arm}-restart-process.json", f"{arm}-restart-attempt.json",
                      f"{arm}-restart-28.pt.incomplete", f"{arm}-final-56.pt.incomplete"])
    return {"schema": 1, "namespace": NAMESPACE, "run_id": run_id,
            "purpose": "engineering_wiring_only_not_resource_or_learning_qualification",
            "directory": f"artifacts/relation/r5-wiring-smoke-{run_id}",
            "paths": paths, "configs": {arm: asdict(config(arm)) for arm in ARMS},
            "artifact_output_policy": "exclusive_r5_engineering_v1",
            "execution_policy": {"device": "cpu", "dtype": "torch.float32",
                                 "deterministic_algorithms": True, "warn_only": False,
                                 "autocast": False, "telemetry": "off"},
            "arm_order": list(ARMS), "mode_order": MODES, "evaluation_steps": EVALUATIONS,
            "case_order": rows, "program_fingerprint": training.program_fingerprint,
            "supervision_identity": training.supervision_identity,
            "negative_row": {"index": 1, "byte_target": "original_declared_block", "oracle_actions": []},
            "truncation": "none: refuse fixture targets beyond capacity",
            "decode": {"literal_policy": CONTENT_POLICY, "stopping_policy": STOPPING_POLICY,
                       "observation_level": "actions", "uniform_seed": UNIFORM_SEED,
                       "uniform_recipe": "sha256(namespace:seed:case_id:absolute_byte_position) first24bits / 2**24",
                       "uniform_storage": "little_endian_float32", "temperature": 1.0},
            "budget": {"primary_updates": 112, "restart_updates": 56, "optimizer_calls": 168},
            "restart": {"cut": 28, "end": 56, "process": "fresh_python_child_per_arm"},
            "units": {"time": "seconds_perf_counter", "bytes": "hex_exact", "updates": "successful_optimizer_calls"},
            "failure_policy": "terminal_fail_stop_no_retry_no_extension",
            "source_hashes": sources(), "environment": execution_environment(device="cpu")}


def write_json(path, value):
    with Path(path).open("x") as handle:
        handle.write(encoded(value).decode() + "\n")


def append(handle, value):
    handle.write(encoded(value).decode() + "\n")
    handle.flush()


def prepare(run_id):
    destination = ROOT / "artifacts/relation" / f"r5-wiring-smoke-{run_id}"
    # Validate names before filesystem use, and refuse existing output before fixture work.
    if type(run_id) is not str or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_id) is None:
        raise WiringRefused("invalid run id")
    destination.mkdir(exist_ok=False)
    card = run_card(run_id)
    write_json(destination / "run-card.json", card)
    write_json(destination / "prepared.json", {"status": "prepared_not_launched", "card_sha256": digest(card)})
    return destination, digest(card)


def validate_card(card):
    """Recompute the fixed card; a caller-supplied digest is not a recipe."""
    try:
        if type(card) is not dict or encoded(card) != encoded(run_card(card["run_id"])):
            raise WiringRefused("run card, source, runtime, fixture or output path drift")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, WiringRefused):
            raise
        raise WiringRefused("invalid R5 run card") from exc


def validate_binding(card, training, config, role, step, *, bundle_corpus=None):
    """Bind both codec entry routes to one canonical engineering specification.

    R4 still owns complete tensor/optimizer/history validation. This preflight
    closes the additional R5 link between that state and the reviewed card.
    """
    validate_card(card)
    try:
        arm = config["arm"]
        if arm not in ARMS or encoded(config) != encoded(card["configs"][arm]):
            raise WiringRefused("R5 checkpoint config mismatch")
        if type(step) is not int or not 0 <= step <= card["configs"][arm]["steps"] or \
                role not in ("continuation", "restart", "final") or \
                (role == "restart" and step != card["restart"]["cut"]) or \
                (role == "final" and step != card["restart"]["end"]):
            raise WiringRefused("R5 checkpoint role/step mismatch")
        if type(training) is not TrainingCorpus or training.provenance != "engineering" or \
                training.program_fingerprint != card["program_fingerprint"] or \
                training.supervision_identity != card["supervision_identity"] or \
                len(training.cases) != len(card["case_order"]):
            raise WiringRefused("R5 checkpoint corpus mismatch")
        for case, row in zip(training.cases, card["case_order"], strict=True):
            if (case.case_id, case.target_start, case.target_stop, case.target.hex(), len(case.actions)) != \
                    (row["case_id"], row["prompt_bytes"], row["target_stop"], row["target_hex"], row["action_targets"]):
                raise WiringRefused("R5 checkpoint corpus row mismatch")
        if bundle_corpus is not None:
            expected = {
                "manifest_path": None, "canonical_payload_sha256": None,
                "file_sha256": None, "rebuilt_payload_sha256": None,
                "training_program_fingerprint": card["program_fingerprint"],
                "supervision_identity_version": SUPERVISION_IDENTITY_VERSION,
                "supervision_sha256": card["supervision_identity"],
                "provenance": "engineering",
            }
            if encoded(bundle_corpus) != encoded(expected):
                raise WiringRefused("R5 checkpoint bundle corpus mismatch")
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, WiringRefused):
            raise
        raise WiringRefused("invalid R5 checkpoint binding") from exc


def validate(directory):
    directory = Path(directory).resolve()
    try:
        card = json.loads((directory / "run-card.json").read_text())
        validate_card(card)
        prepared = json.loads((directory / "prepared.json").read_text())
        if directory != ROOT / card["directory"]:
            raise WiringRefused("run card, source, runtime, fixture or output path drift")
        if encoded(prepared) != encoded({"status": "prepared_not_launched", "card_sha256": digest(card)}):
            raise WiringRefused("preparation record differs from run card")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, WiringRefused):
            raise
        raise WiringRefused("invalid or incomplete preparation evidence") from exc
    return card


def identity(trainer):
    return {"model": state_digest(trainer.model.state_dict()),
            "optimizer": optimizer_digest(trainer.optimizer, trainer.model),
            "rng": rng_digest("cpu", trainer.cursor.generator),
            "sampler": digest({key: value.tolist() if isinstance(value, torch.Tensor) else value
                               for key, value in trainer.cursor.state_dict().items()}),
            "history": digest(trainer.history), "accounting": asdict(trainer.accounting),
            "diagnostics": digest(trainer.diagnostics())}


def save_checkpoint(path, trainer, card, *, role="continuation"):
    validate_binding(card, trainer.training, asdict(trainer.config), role, trainer.completed_step)
    # Exercise the accepted publication validator in its temporary-only domain.
    with tempfile.TemporaryDirectory(prefix="r5-codec-") as temp:
        state = trainer.save(Path(temp) / "state.pt", sources=AUTHORITATIVE_R4_SOURCES)
        state = {key: value for key, value in state.items() if key not in ("namespace", "schema")}
        envelope = {"schema": 1, "namespace": NAMESPACE, "card_sha256": digest(card),
                    "execution_state": state, "identity": identity(trainer),
                    "arm": trainer.config.arm, "role": role, "completed_step": trainer.completed_step}
        path = Path(path)
        # Retained exclusive stage is also the publication-attempt record.
        # Hard-link commit cannot replace an existing destination, even in a race.
        stage = path.with_name(path.name + ".incomplete")
        if path.exists():
            raise FileExistsError(path)
        with stage.open("xb") as handle:
            torch.save(envelope, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(stage, path)
        stage.unlink()


def load_checkpoint(path, card, training, *, arm=None, role=None, step=None):
    caller_rng = capture_rng_state()
    policy = (torch.are_deterministic_algorithms_enabled(),
              torch.is_deterministic_algorithms_warn_only_enabled())
    try:
        return _load_checkpoint(path, card, training, arm=arm, role=role, step=step)
    except BaseException as exc:
        restore_rng_state(caller_rng)
        torch.use_deterministic_algorithms(policy[0], warn_only=policy[1])
        if isinstance(exc, (OSError, KeyError, TypeError, ValueError)) and not isinstance(exc, WiringRefused):
            raise WiringRefused("invalid R5 checkpoint evidence") from exc
        raise


def _load_checkpoint(path, card, training, *, arm, role, step):
    # Trusted local, model-bearing artifacts only; never load third-party pickle.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if type(payload) is not dict or set(payload) != {"schema", "namespace", "card_sha256", "execution_state", "identity", "arm", "role", "completed_step"} or \
            type(payload["schema"]) is not int or payload["schema"] != 1 or payload["namespace"] != NAMESPACE or payload["card_sha256"] != digest(card):
        raise WiringRefused("R5 checkpoint envelope mismatch")
    if type(payload["completed_step"]) is not int or not 0 <= payload["completed_step"] <= 56 or \
            payload["arm"] not in ARMS or payload["role"] not in ("continuation", "restart", "final") or \
            (payload["role"] == "restart" and payload["completed_step"] != 28) or \
            (payload["role"] == "final" and payload["completed_step"] != 56) or \
            (arm is not None and payload["arm"] != arm) or \
            (role is not None and payload["role"] != role) or \
            (step is not None and payload["completed_step"] != step):
        raise WiringRefused("R5 checkpoint role/arm/step mismatch")
    state = payload["execution_state"]
    arm = state["config"]["arm"]
    if arm != payload["arm"]:
        raise WiringRefused("R5 checkpoint arm mismatch")
    if type(state["completed_step"]) is not int or state["completed_step"] != payload["completed_step"]:
        raise WiringRefused("R5 checkpoint step mismatch")
    if type(state["corpus"]) is not dict:
        raise WiringRefused("R5 checkpoint bundle corpus mismatch")
    validate_binding(card, training, state["config"], payload["role"], payload["completed_step"],
                     bundle_corpus=state["corpus"])
    from dm.eval.relation_training_evidence import RELATION_CHECKPOINT_SCHEMA
    with tempfile.TemporaryDirectory(prefix="r5-codec-") as temp:
        path = Path(temp) / "state.pt"
        torch.save({**state, "schema": RELATION_CHECKPOINT_SCHEMA,
                    "namespace": "direction4-relation-r4-test-only"}, path)
        trainer = RelationTrainer.load(path, training=training, sources=AUTHORITATIVE_R4_SOURCES)
    if trainer.completed_step != payload["completed_step"]:
        raise WiringRefused("R5 checkpoint step mismatch")
    if identity(trainer) != payload["identity"]:
        raise WiringRefused("R5 checkpoint identity mismatch")
    return trainer


def evaluate(trainer, card, handle):
    before = identity(trainer)
    rng = capture_rng_state(trainer.cursor.generator)
    was_training = trainer.model.training
    try:
        trainer.model.eval()
        for mode in MODES[trainer.config.arm]:
            for index, case in enumerate(trainer.training.cases):
                events = []
                kwargs = {}
                if mode == "oracle_copy":
                    kwargs["oracle_actions"] = {(0, action.boundary): action.runtime_key() for action in case.actions}
                start = time.perf_counter()
                try:
                    output = trainer.model.generate_relation(
                        [case.prompt], case.target_stop - case.target_start, mode=mode,
                        literal_policy=CONTENT_POLICY, observer=events.append, observation_level="actions",
                        variates=torch.tensor([uniforms(case)], dtype=torch.float32), **kwargs)
                except BaseException as exc:
                    append(handle, {"arm": trainer.config.arm, "step": trainer.completed_step,
                                    "mode": mode, "row": index, "status": "failed",
                                    "error_type": type(exc).__name__, "error": str(exc),
                                    "seconds": time.perf_counter() - start,
                                    "events": [asdict(event) for event in events]})
                    raise
                seconds = time.perf_counter() - start
                tokens = output[0].tolist()
                if any(token < N_SPECIAL or token >= N_SPECIAL + 256 for token in tokens):
                    raise WiringRefused("common-policy output contains non-byte symbols")
                emitted = bytes(token - N_SPECIAL for token in tokens)[case.target_start:]
                append(handle, {"arm": trainer.config.arm, "step": trainer.completed_step,
                                "mode": mode, "row": index, "case_id": case.case_id, "status": "recorded",
                                "target_bytes": len(case.target), "action_targets": len(case.actions),
                                "emitted_hex": emitted.hex(), "exact_target": emitted == case.target,
                                "seconds": seconds, "events": [asdict(event) for event in events]})
        if rng_digest("cpu", trainer.cursor.generator) != before["rng"]:
            raise WiringRefused("decode consumed training RNG")
    finally:
        trainer.model.train(was_training)
        restore_rng_state(rng, trainer.cursor.generator)
    if identity(trainer) != before:
        raise WiringRefused("decode changed training state")


def updates(trainer, handle, stop):
    while trainer.completed_step < stop:
        start = time.perf_counter()
        trainer.step()
        append(handle, {"arm": trainer.config.arm, "step": trainer.completed_step,
                        "seconds": time.perf_counter() - start, "history": trainer.history[-1]})


def restart_worker(directory, arm):
    directory = Path(directory).resolve()
    card = validate(directory)
    if (directory / "terminal.json").exists():
        raise WiringRefused("terminal route cannot restart; retry refused")
    if arm not in ARMS or not (directory / "launch.json").is_file():
        raise WiringRefused("restart requires declared arm and launched route")
    launch = json.loads((directory / "launch.json").read_text())
    if launch.get("card_sha256") != digest(card) or launch.get("status") != "attempted":
        raise WiringRefused("restart launch binding differs")
    if any((directory / f"{arm}-restart-{suffix}").exists()
           for suffix in ("updates.jsonl", "result.json", "process.json", "attempt.json")):
        raise WiringRefused("restart already attempted; retry refused")
    write_json(directory / f"{arm}-restart-attempt.json", {
        "status": "attempted", "card_sha256": digest(card), "arm": arm})
    start = time.perf_counter()
    trainer = load_checkpoint(directory / f"{arm}-restart-28.pt", card, _fixture(),
                              arm=arm, role="restart", step=28)
    load_seconds = time.perf_counter() - start
    if trainer.completed_step != 28 or trainer.config.arm != arm:
        raise WiringRefused("restart requires exact arm and step 28")
    with (directory / f"{arm}-restart-updates.jsonl").open("x") as handle:
        updates(trainer, handle, 56)
    write_json(directory / f"{arm}-restart-result.json", {
        "identity": identity(trainer), "checkpoint_load_seconds": load_seconds,
        "replayed_updates": 28})


def execute(directory, approved_digest):
    directory = Path(directory).resolve()
    if (directory / "launch.json").exists() or (directory / "terminal.json").exists():
        raise WiringRefused("route already attempted; retry refused")
    card = validate(directory)
    if approved_digest != digest(card):
        raise WiringRefused("explicit approved run-card digest required")
    if torch.get_default_dtype() != torch.float32 or torch.is_autocast_enabled("cpu"):
        raise WiringRefused("R5 requires float32 without autocast")
    validate_r0()
    for name in card["paths"]:
        if name not in ("run-card.json", "prepared.json") and (directory / name).exists():
            raise WiringRefused(f"reserved output exists: {name}")
    write_json(directory / "launch.json", {"card_sha256": digest(card), "status": "attempted"})
    try:
        summaries = {}
        training = _fixture()
        with (directory / "updates.jsonl").open("x") as work, \
                (directory / "decodes.jsonl").open("x") as decode, \
                (directory / "checkpoint-io.jsonl").open("x") as io:
            for arm in ARMS:
                trainer = RelationTrainer(config(arm), training)
                evaluate(trainer, card, decode)
                for stop in (28, 56):
                    updates(trainer, work, stop)
                    evaluate(trainer, card, decode)
                    validate(directory)
                    start = time.perf_counter()
                    name = f"{arm}-{'restart-28' if stop == 28 else 'final-56'}.pt"
                    save_checkpoint(directory / name, trainer, card, role="restart" if stop == 28 else "final")
                    append(io, {"arm": arm, "step": stop, "operation": "save",
                                "seconds": time.perf_counter() - start, "path": name})
                expected = identity(trainer)
                start = time.perf_counter()
                child = subprocess.run([sys.executable, str(ROOT / "scripts/relation_wiring.py"),
                                        "restart-worker", str(directory), "--arm", arm],
                                       check=False, capture_output=True, text=True, cwd=ROOT)
                write_json(directory / f"{arm}-restart-process.json", {
                    "seconds": time.perf_counter() - start, "exit_code": child.returncode,
                    "stdout": child.stdout, "stderr": child.stderr})
                if child.returncode:
                    raise WiringRefused(f"restart child failed for {arm}: {child.stderr}")
                actual = json.loads((directory / f"{arm}-restart-result.json").read_text())
                if actual["identity"] != expected or actual["replayed_updates"] != 28:
                    raise WiringRefused("fresh-process restart differs from uninterrupted trajectory")
                summaries[arm] = {"identity": expected, "restart_exact": True}
        write_json(directory / "primary-summary.json", summaries)
        write_json(directory / "terminal.json", {"status": "complete", "budget": card["budget"],
                                                 "card_sha256": digest(card)})
    except BaseException as exc:
        try:
            write_json(directory / "terminal.json", {"status": "failed", "error_type": type(exc).__name__,
                                                     "error": str(exc), "card_sha256": digest(card)})
        except BaseException as publication_error:  # noqa: BLE001 -- preserve initiating failure
            exc.add_note(f"terminal publication also failed: {publication_error!r}")
        raise
