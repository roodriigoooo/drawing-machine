"""Exclusive bounded demonstrated-affine execution, with its own checkpoint namespace."""
from __future__ import annotations

import hashlib
import json
import os
import resource
import signal
import tempfile
import time
from pathlib import Path

import torch

from dm.eval.relation_evidence import execution_environment, state_digest
from dm.eval.relation_training_evidence import RELATION_CHECKPOINT_SCHEMA
from dm.relation.demonstrated_card import recipe
from dm.relation.demonstrated_fixture import build_fit_fixture
from dm.relation.demonstrated_pilot import build_pilot_fixture
from dm.relation.demonstrated_readout import evaluate, fit_result
from dm.relation.demonstrated_transfer import summarize
from dm.relation.wiring import digest, encoded, identity, validate_r0
from dm.train_relation import (
    AUTHORITATIVE_R4_SOURCES,
    RelationTrainConfig,
    RelationTrainer,
    TrainingCorpus,
)

ROOT = Path(__file__).resolve().parents[2]
NAMESPACE = "direction4-demonstrated-affine-checkpoint-v1"
# Accepted, model-blind artifacts for this one task package, not arbitrary JSON.
AUDIT_IDENTITIES = {
    "fit-audit.json": "a58e81abd0735347f681115373eb6028795db79f9866296027e584ea51e07411",
    "pilot-audit.json": "458dcdbd22a90e66d986277c24506eab6a3a443a21c4b601d13be3fa784d1671",
    "scientific-exclusion.json": "065524f14e78fa61a770e02a04d25d21f3664bcba063c30e503dad865bb397a3",
}


def write(path, value):
    with Path(path).open("x") as stream:
        stream.write(encoded(value).decode() + "\n")


def sources():
    # Includes reference/evaluation code and all transitive local implementation.
    paths = {*AUTHORITATIVE_R4_SOURCES,
             *ROOT.glob("dm/relation/demonstrated*.py"),
             ROOT / "dm/eval/relation_decision_diagnostic.py",
             ROOT / "dm/relation/wiring.py", ROOT / "dm/relation/wiring_spec.py",
             *ROOT.glob("scripts/relation_demonstrated*.py")}
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


def make_card(audit_directory, *, recover=False):
    audits, reports = {}, {}
    bound_sources = sources()
    for name, expected_digest in AUDIT_IDENTITIES.items():
        path = Path(audit_directory) / name
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected_digest:
            raise ValueError(f"unrecognized audit identity: {name}")
        value = json.loads(data)
        reports[name] = value
        for source, expected in value["source_sha256"].items():
            if hashlib.sha256((ROOT / source).read_bytes()).hexdigest() != expected:
                raise ValueError(f"audit source drift: {source}")
        bound_sources.update(value["source_sha256"])
        audits[name] = {"path": str(path.resolve()), "sha256": expected_digest}
        if value.get("rejection_ledger") or value.get("overlaps"):
            raise ValueError("task-invalid audit refuses card freeze")
        if name == "scientific-exclusion.json":
            if value["status"] != "scientific_source_exclusion_pass":
                raise ValueError("scientific exclusion did not pass")
            if hashlib.sha256((ROOT / "runs/relation_corpus_v1.json").read_bytes()).hexdigest() != value["scientific_manifest_file_sha256"]:
                raise ValueError("scientific manifest drift")
    card = {"schema": 1, "namespace": NAMESPACE, "recipe": recipe(), "audits": audits,
            "task_rows": {"fit": reports["fit-audit.json"]["rows"],
                          "pilot": reports["pilot-audit.json"]["panels"]},
            "source_sha256": bound_sources, "r0": validate_r0(),
            "environment": execution_environment(device="cpu"),
            "status": "frozen_pretraining", "optimizer_updates_at_freeze": 0}
    if recover:
        from dm.relation.demonstrated_recovery import specification
        card["recovery"] = specification(card)
    return card


def validate_card(card):
    directory = Path(card["audits"]["fit-audit.json"]["path"]).parent
    rebuilt = make_card(directory, recover=True) if "recovery" in card else make_card(directory)
    if digest(card) != digest(rebuilt):
        raise ValueError("card source/environment/recipe/audit drift")


def cell_config(cell):
    # JSON represents tuples as arrays; restore the trainer's strict field types.
    fields = dict(cell["config"])
    fields["adam_betas"] = tuple(fields["adam_betas"])
    if fields["component_gradient_events"] is not None:
        fields["component_gradient_events"] = tuple(fields["component_gradient_events"])
    return RelationTrainConfig(**fields)


def save_checkpoint(path, trainer, card, cell):
    with tempfile.TemporaryDirectory(prefix="demonstrated-codec-") as temp:
        state = trainer.save(Path(temp) / "state.pt", sources=AUTHORITATIVE_R4_SOURCES)
        state = {k: v for k, v in state.items() if k not in ("schema", "namespace")}
        envelope = {"namespace": NAMESPACE, "schema": 1, "card_sha256": digest(card),
                    "cell_id": cell["cell_id"], "step": trainer.completed_step,
                    "identity": identity(trainer), "state": state}
        stage = Path(str(path) + ".incomplete")
        with stage.open("xb") as stream:
            torch.save(envelope, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(stage, path)
        stage.unlink()
    return envelope["identity"]


def load_checkpoint(path, card, cell, training, step):
    envelope = torch.load(path, map_location="cpu", weights_only=False)
    if envelope["namespace"] != NAMESPACE or envelope["schema"] != 1 or envelope["card_sha256"] != digest(card) or envelope["cell_id"] != cell["cell_id"] or envelope["step"] != step:
        raise ValueError("demonstrated checkpoint binding mismatch")
    state = envelope["state"]
    if digest(state["config"]) != digest(cell["config"]) or state["completed_step"] != step:
        raise ValueError("demonstrated checkpoint config/step mismatch")
    with tempfile.TemporaryDirectory(prefix="demonstrated-codec-") as temp:
        path = Path(temp) / "state.pt"
        torch.save({**state, "schema": RELATION_CHECKPOINT_SCHEMA,
                    "namespace": "direction4-relation-r4-test-only"}, path)
        trainer = RelationTrainer.load(path, training=training, sources=AUTHORITATIVE_R4_SOURCES)
    if identity(trainer) != envelope["identity"]:
        raise ValueError("demonstrated checkpoint content identity mismatch")
    return trainer


def execute(directory, expected_digest, *, authorize_recovery_replay=False):
    directory = Path(directory)
    card = json.loads((directory / "run-card.json").read_text())
    validate_card(card)
    if digest(card) != expected_digest:
        raise ValueError("requested card digest differs")
    write(directory / "launch.json", {"card_sha256": expected_digest, "time": time.time(),
                                       "authorize_recovery_replay": authorize_recovery_replay})
    started = time.perf_counter()
    previous_seconds = card.get("recovery", {}).get("previous_execution_seconds", 0)
    execution_seconds = 14400 - previous_seconds
    fit_results, pilot_rows = [], []
    completed_updates, initial_trunks = {}, {}
    replayed_updates = 0
    trainer, cell = None, None
    previous_handler = signal.getsignal(signal.SIGALRM)

    def timeout(signum, frame):
        raise TimeoutError("frozen execution wall-time cap reached")

    signal.signal(signal.SIGALRM, timeout)
    try:
        if execution_seconds <= 0:
            raise TimeoutError("aggregate cap reached before recovery")
        signal.setitimer(signal.ITIMER_REAL, execution_seconds)
        fit = build_fit_fixture().cases
        panels = dict(build_pilot_fixture().panels)
        with (directory / "updates.jsonl").open("x") as updates, (directory / "readouts.jsonl").open("x") as readouts:
            def emit(row):
                readouts.write(encoded(row).decode() + "\n")
                readouts.flush()

            for cell in card["recipe"]["cells"]:
                if cell["stage"] == "pilot" and not all(r["pass"] for r in fit_results):
                    break
                if cell["stage"] == "pilot" and "recovery" in card and not authorize_recovery_replay:
                    # Fit can finish inside the original total-spend limit, but
                    # all four pilot cells cannot fit after the defect replay.
                    # Do not silently spend the recovery allowance or start an
                    # incomplete paired pilot while permission is pending.
                    validate_card(card)
                    write(directory / "terminal.json", {
                        "status": "fit_complete_pilot_budget_pending", "fit": fit_results,
                        "completed_updates": completed_updates, "replayed_updates": replayed_updates,
                        "total_executed_updates_across_attempts": sum(completed_updates.values()) + replayed_updates,
                        "seconds": time.perf_counter() - started,
                        "total_execution_seconds_across_attempts": previous_seconds + time.perf_counter() - started})
                    return
                remaining = execution_seconds - (time.perf_counter() - started)
                if remaining <= 0:
                    raise TimeoutError("aggregate cap reached")
                cell_seconds = 5400 - (previous_seconds if "recovery" in card and cell["cell_id"] == "fit-131" else 0)
                if cell_seconds <= 0:
                    raise TimeoutError("cell cap reached before recovery")
                signal.setitimer(signal.ITIMER_REAL, min(cell_seconds, remaining))
                cell_start = time.perf_counter()
                cases = fit if cell["stage"] == "fit" else panels["train"]
                training = TrainingCorpus.engineering_cases(cases)
                recovered_history = None
                if "recovery" in card and cell["cell_id"] == card["recovery"]["cell_id"]:
                    from dm.relation.demonstrated_recovery import PREVIOUS, restore
                    trainer, recovered_history = restore(card, cell, training)
                    completed_updates[cell["cell_id"]] = trainer.completed_step
                    for record in recovered_history[:trainer.completed_step]:
                        updates.write(encoded(record).decode() + "\n")
                    for line in (PREVIOUS / "readouts.jsonl").read_text().splitlines():
                        emit(json.loads(line))
                    for old_step in (0, 64, 256):
                        name = f"fit-131-{old_step}-summary.json"
                        write(directory / name, json.loads((PREVIOUS / name).read_text()))
                    save_checkpoint(directory / "fit-131-256.pt", trainer, card, cell)
                    write(directory / "recovery.json", {**card["recovery"],
                                                       "restored_identity": identity(trainer)})
                else:
                    trainer = RelationTrainer(cell_config(cell), training)
                if training.program_fingerprint != cell["program_fingerprint"] or training.supervision_identity != cell["supervision_identity"]:
                    raise ValueError("cell training data differs from card")
                if trainer.model.n_params() != cell["parameter_count"]:
                    raise ValueError("actual model parameter count differs")
                trunk = state_digest({key: value for key, value in trainer.model.state_dict().items()
                                      if not key.startswith("relation.")})
                if cell["stage"] == "pilot":
                    previous = initial_trunks.setdefault(trainer.config.seed, trunk)
                    if trunk != previous:
                        raise ValueError("paired shared initialization differs")
                write(directory / f"{cell['cell_id']}-initial.json", {
                    "identity": identity(trainer), "shared_trunk_sha256": trunk,
                    "completed_updates": trainer.completed_step})
                for step in cell["evaluation_steps"]:
                    if recovered_history is not None and step <= 256:
                        continue
                    while trainer.completed_step < step:
                        tick = time.perf_counter()
                        trainer.step()
                        if recovered_history is not None:
                            if trainer.history[-1] != recovered_history[trainer.completed_step - 1]["history"]:
                                raise ValueError("recovery replay differs from original raw history")
                            replayed_updates += 1
                        completed_updates[cell["cell_id"]] = trainer.completed_step
                        if trainer.history[-1]["batch_ids"] != cell["batch_order"][trainer.completed_step - 1]:
                            raise ValueError("sampler order differs from frozen card")
                        updates.write(encoded({"cell_id": cell["cell_id"], "step": trainer.completed_step,
                                               "seconds": time.perf_counter() - tick,
                                               "history": trainer.history[-1]}).decode() + "\n")
                        updates.flush()
                    path = directory / f"{cell['cell_id']}-{step}.pt"
                    saved_identity = save_checkpoint(path, trainer, card, cell)
                    if step == 1024:
                        trainer = load_checkpoint(path, card, cell, training, step)
                    if cell["stage"] == "fit":
                        rows = evaluate(trainer, fit, "fit", emit, digest(saved_identity))
                        result = {"cell_id": cell["cell_id"], "step": step, **fit_result(rows)}
                        write(directory / f"{cell['cell_id']}-{step}-summary.json", result)
                        if step == 1024:
                            fit_results.append(result)
                    else:
                        selected = set(card["recipe"]["pilot_intermediate_case_ids"])
                        training_rows = evaluate(trainer, tuple(c for c in cases if c.case_id in selected), "train", emit, digest(saved_identity))
                        if step == 1024:
                            pilot_rows.extend(training_rows)
                            for split in ("iid", "affine", "family", "negative"):
                                pilot_rows.extend(evaluate(trainer, panels[split], split, emit, digest(saved_identity)))
                remaining = execution_seconds - (time.perf_counter() - started)
                if remaining <= 0:
                    raise TimeoutError("aggregate cap reached")
                signal.setitimer(signal.ITIMER_REAL, remaining)
                write(directory / f"{cell['cell_id']}-complete.json", {
                    "steps": trainer.completed_step, "seconds": time.perf_counter() - cell_start,
                    "peak_process_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    "peak_rss_unit": "bytes_on_Darwin_kibibytes_on_Linux"})
            if len(fit_results) != 2:
                raise ValueError("missing fit replication")
            if not all(r["pass"] for r in fit_results):
                diagnosis = {"diagnosis": "bounded_fit_failure", "fit": fit_results}
            else:
                diagnosis = summarize(pilot_rows, fit_results, card)
            validate_card(card)
            write(directory / "terminal.json", {"status": "complete", **diagnosis,
                                                 "completed_updates": completed_updates,
                                                 "replayed_updates": replayed_updates,
                                                 "total_executed_updates_across_attempts": sum(completed_updates.values()) + replayed_updates,
                                                 "total_execution_seconds_across_attempts": previous_seconds + time.perf_counter() - started,
                                                 "seconds": time.perf_counter() - started})
    except BaseException as exc:
        write(directory / "terminal.json", {"status": "budget_exhausted" if isinstance(exc, TimeoutError) else "implementation_defective",
                                             "error": str(exc), "error_type": type(exc).__name__,
                                             "completed_updates": completed_updates,
                                             "replayed_updates": replayed_updates,
                                             "active_cell": None if cell is None else cell["cell_id"],
                                             "active_step": None if trainer is None else trainer.completed_step,
                                             "seconds": time.perf_counter() - started})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
