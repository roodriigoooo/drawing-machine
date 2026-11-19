"""Rebuild the frozen scientific corpus and exclude every matching source orbit."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

from dm.data import relation
from dm.relation.demonstrated import orbit_key
from dm.relation.demonstrated_fixture import build_fit_fixture
from dm.relation.demonstrated_pilot import build_pilot_fixture
from dm.relation.transducer import prefix_index


def first_difference(recorded, rebuilt, path="$",):
    """Return one exact JSON-field witness, without truncating its values."""
    if type(recorded) is not type(rebuilt):
        return {"path": path, "recorded": recorded, "rebuilt": rebuilt}
    if isinstance(recorded, dict):
        for key in sorted(recorded.keys() | rebuilt.keys()):
            if key not in recorded or key not in rebuilt:
                return {"path": f"{path}.{key}", "recorded_present": key in recorded,
                        "rebuilt_present": key in rebuilt}
            difference = first_difference(recorded[key], rebuilt[key], f"{path}.{key}")
            if difference:
                return difference
    elif isinstance(recorded, list):
        if len(recorded) != len(rebuilt):
            return {"path": f"{path}.length", "recorded": len(recorded), "rebuilt": len(rebuilt)}
        for index, (left, right) in enumerate(zip(recorded, rebuilt, strict=True)):
            difference = first_difference(left, right, f"{path}[{index}]")
            if difference:
                return difference
    elif recorded != rebuilt:
        return {"path": path, "recorded": recorded, "rebuilt": rebuilt}
    return None


def audit_scientific_sources(path: Path) -> dict:
    body = relation.load_manifest(path)
    rebuilt = relation.build(relation.manifest_config(body))
    rebuilt_body = json.loads(json.dumps(relation.manifest(rebuilt)))
    provenance = {
        "schema": 1,
        "scientific_manifest_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "scientific_payload_sha256": body["corpus_sha256"],
        "rebuilt_payload_sha256": rebuilt_body["corpus_sha256"],
        "config": body["acceptance"]["config"],
        "environment": {"python": sys.version, "platform": platform.platform()},
        "model_calls": 0, "optimizer_updates": 0,
    }
    if rebuilt_body != body:
        return {**provenance, "status": "scientific_manifest_rebuild_mismatch",
                "differing_sections": sorted(k for k in body.keys() | rebuilt_body.keys()
                                             if body.get(k) != rebuilt_body.get(k)),
                "first_case_difference": first_difference(body.get("cases"), rebuilt_body.get("cases")),
                "first_acceptance_difference": first_difference(body.get("acceptance"), rebuilt_body.get("acceptance")),
                "source_scan_performed": False, "overlaps": []}
    spans = set()
    # The candidate task uses exactly MOVE, LINE, LINE. Scan every such span
    # in every scientific program, including unannotated/distractor geometry.
    # A different opcode skeleton cannot be an affine equivalent of this motif.
    for case in rebuilt.cases:
        marks, _, ops = prefix_index(case.flat)
        for i in range(len(ops) - 2):
            if ops[i:i + 3] == (1, 2, 2):
                spans.add(case.flat[marks[i]:marks[i + 3]])
    keys = {orbit_key(s) for s in spans}
    sources = build_fit_fixture().sources + build_pilot_fixture().sources
    overlaps = [s.hex() for s in sources if orbit_key(s) in keys]
    return {**provenance, "source_scan_performed": True,
            "status": "task_invalid" if overlaps else "scientific_source_exclusion_pass",
            "scientific_manifest_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "scientific_payload_sha256": body["corpus_sha256"],
            "scientific_cases": len(rebuilt.cases),
            "unique_three_instruction_source_spans": len(spans),
            "unique_source_orbits": len(keys), "overlaps": overlaps,
            "task_sources_sha256": hashlib.sha256(b"".join(sources)).hexdigest(),
            "model_calls": 0, "optimizer_updates": 0}
