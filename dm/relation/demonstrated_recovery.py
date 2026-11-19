"""One explicitly bound accounting-validator repair, not general checkpoint migration."""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import torch

from dm.eval.relation_training_evidence import RELATION_CHECKPOINT_SCHEMA, validate_bundle_shape
from dm.train_relation import AUTHORITATIVE_R4_SOURCES, RelationTrainer, source_hashes

ROOT = Path(__file__).resolve().parents[2]
PREVIOUS = ROOT / "artifacts/relation/demonstrated-affine-run-02"
OLD_CARD = "7c7fe6b5d4fd43d44df996c735d614011125022f67e0f99fcf0ba033b268e303"
INPUTS = {
    "../demonstrated-affine-run-01/terminal.json": "8e34069b7805fbe0f472a10aff6cb0dde402c7351833a1ea45497f0f68226aaa",
    "run-card.json": "fc6f8589ffbb2a6aa3959f8750ec81625caafb913d4245867e1e73081004d035",
    "fit-131-256.pt": "fb33ccb47df26d48379f359674a83d09e684ceda2b86e22c13aed48298b7bef7",
    "updates.jsonl": "9e800ceaacbeae4c20cf95cd0e704fc8be17f1facddf63b71f3f65cff9e78f6f",
    "readouts.jsonl": "1cf5b6a3c8bf85985ba2fbd69bb301eb7d83746501df72ebfdbc0a9b0097ea21",
    "terminal.json": "720114c1e088de91d303bf13a6cdfca91792755bec7e40742f990e65ce8eb4c9",
    "fit-131-initial.json": "38dbf2da15a6cf639ba9e6ed526689c8b38030919aa3ab29e59c6ad3b3acf44f",
    "fit-131-0-summary.json": "41306f9da4be5274c0a51cc616404179c58afb497d53af7dc0f9c8500a3f5a70",
    "fit-131-64-summary.json": "81f884798fb9681987ae1f5a54dde5721f00f28d44db46e47fddc3c2a1914836",
    "fit-131-256-summary.json": "52ca4b6b46d2928f811e21a8a1588925399e857b198b72dee67d1f3e5c11600f",
}
VALIDATOR = "dm/eval/relation_training_evidence.py"
ALLOWED_CHANGES = {VALIDATOR, "dm/relation/demonstrated_run.py",
                   "scripts/relation_demonstrated_run.py"}


def specification(card):
    from dm.relation.wiring import digest

    for name, expected in INPUTS.items():
        if hashlib.sha256((PREVIOUS / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"recovery input changed: {name}")
    old = json.loads((PREVIOUS / "run-card.json").read_text())
    if digest(old) != OLD_CARD or digest(old["recipe"]) != digest(card["recipe"]):
        raise ValueError("recovery recipe changed")
    if old["environment"] != card["environment"] or old["task_rows"] != card["task_rows"]:
        raise ValueError("recovery environment/task changed")
    changes = {name: {"before": value, "after": card["source_sha256"].get(name)}
               for name, value in old["source_sha256"].items()
               if card["source_sha256"].get(name) != value}
    if set(changes) != ALLOWED_CHANGES:
        raise ValueError("unreviewed recovery source change")
    previous_seconds = sum(json.loads((PREVIOUS / name).read_text())["seconds"]
                           for name in ("terminal.json", "../demonstrated-affine-run-01/terminal.json"))
    return {"kind": "chronological_accounting_validator_repair_v1",
            "previous_directory": str(PREVIOUS), "input_sha256": INPUTS,
            "previous_card_sha256": OLD_CARD, "source_changes": changes,
            "cell_id": "fit-131", "checkpoint_step": 256,
            "previous_executed_updates": 1024, "replayed_updates": 768,
            "previous_execution_seconds": previous_seconds,
            "comparison": "all replayed raw history records must match exactly",
            "state_migration": "only validator source hash/digest in temporary codec; all trainer identities unchanged"}


def restore(card, cell, training):
    from dm.relation.wiring import digest, identity

    if specification(card) != card["recovery"] or cell["cell_id"] != "fit-131":
        raise ValueError("recovery binding mismatch")
    envelope = torch.load(PREVIOUS / "fit-131-256.pt", map_location="cpu", weights_only=False)
    if (envelope["card_sha256"], envelope["cell_id"], envelope["step"], envelope["namespace"]) != (
            OLD_CARD, "fit-131", 256, "direction4-demonstrated-affine-checkpoint-v1"):
        raise ValueError("recovery checkpoint binding mismatch")
    state = envelope["state"]
    if digest(state["config"]) != digest(cell["config"]):
        raise ValueError("recovery checkpoint config changed")
    payload = {**state, "schema": RELATION_CHECKPOINT_SCHEMA,
               "namespace": "direction4-relation-r4-test-only"}
    validate_bundle_shape(payload)
    if digest(state["source_hashes"]) != state["source_digest"]:
        raise ValueError("recovery checkpoint source digest mismatch")
    current = source_hashes()
    changed = {str(Path(name).relative_to(ROOT)) for name in current
               if current[name] != state["source_hashes"].get(name)}
    if set(current) != set(state["source_hashes"]) or changed != {VALIDATOR}:
        raise ValueError("recovery would change training runtime sources")
    old_card = json.loads((PREVIOUS / "run-card.json").read_text())
    if any(value != old_card["source_sha256"][str(Path(name).relative_to(ROOT))]
           for name, value in state["source_hashes"].items()):
        raise ValueError("recovery checkpoint disagrees with historical sources")
    # This explicit, recorded migration is confined to an ephemeral codec file.
    # The immutable old checkpoint and all model/Adam/RNG/accounting data remain intact.
    payload = {**payload, "source_hashes": current, "source_digest": digest(current)}
    with tempfile.TemporaryDirectory(prefix="demonstrated-recovery-") as directory:
        path = Path(directory) / "state.pt"
        torch.save(payload, path)
        trainer = RelationTrainer.load(path, training=training, sources=AUTHORITATIVE_R4_SOURCES)
    if identity(trainer) != envelope["identity"]:
        raise ValueError("recovery altered trainer content identity")
    histories = [json.loads(line) for line in (PREVIOUS / "updates.jsonl").read_text().splitlines()]
    if len(histories) != 1024 or trainer.history != [r["history"] for r in histories[:256]]:
        raise ValueError("recovery history prefix mismatch")
    return trainer, histories
