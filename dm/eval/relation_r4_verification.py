"""Model-blind R4 extraction proof. No checkpoint or scientific scoring path."""

from __future__ import annotations

import hashlib
import json
import platform
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from ..data import relation as corpus
from ..relation.candidates import candidate_spans
from .relation_contract import corpus_seed_faults, load_protocol


def verify_candidates(cases: Sequence[corpus.RelationCase], *,
                      progress: Callable[[str], None] | None = None) -> dict[str, int]:
    """Compare the retained pre-extraction algorithm at every case/boundary.

    Include covered interiors and post-HALT boundaries: the extraction contract
    concerns enumeration, not just the subset queried by supervised training.
    Pass only the prefix to runtime so future bytes cannot influence its result.
    """
    queries = spans = widest = 0
    for index, case in enumerate(cases):
        marks = corpus.boundaries(case.flat)
        halts = corpus._halt_offsets(case.flat)
        for boundary in marks:
            expected = tuple(corpus._eligible_spans(case.flat, marks, boundary, halts))
            actual = candidate_spans(case.flat[:boundary], boundary)
            if actual != expected or corpus.candidate_spans(case.flat, boundary) != expected:
                raise corpus.ManifestRefused(
                    f"candidate extraction mismatch: case={case.case_id}, boundary={boundary}")
            queries += 1
            spans += len(actual)
            widest = max(widest, len(actual))
        if progress is not None and (index + 1) % 5000 == 0:
            progress(f"candidate parity: {index + 1}/{len(cases)} cases, {queries} boundaries")
    return {"cases": len(cases), "boundaries": queries, "candidate_spans": spans,
            "widest_query": widest, "mismatches": 0}


def verify_frozen_corpus(path: Path, *, progress: Callable[[str], None] | None = None) -> dict:
    started = time.perf_counter()
    protocol = load_protocol(require_corpus=True)
    body = corpus.load_manifest(path)
    faults = corpus_seed_faults(body, provenance="scientific")
    if faults:
        raise corpus.ManifestRefused("; ".join(faults))
    digests = corpus.manifest_digests(path)
    if progress:
        progress("rebuilding the full frozen corpus (no model is constructed)")
    rebuilt = corpus.build(corpus.manifest_config(body))
    rebuilt_body = corpus.manifest(rebuilt)
    if json.dumps(rebuilt_body, sort_keys=True) != json.dumps(body, sort_keys=True):
        raise corpus.ManifestRefused("R4 rebuild is not canonical-equal to the frozen manifest")
    rebuild_seconds = time.perf_counter() - started
    if progress:
        progress(f"rebuild exact in {rebuild_seconds:.2f}s; comparing every candidate boundary")
    readings = verify_candidates(rebuilt.cases, progress=progress)
    if corpus.manifest_digests(path) != digests:
        raise corpus.ManifestRefused("manifest changed during R4 verification")
    root = Path(__file__).resolve().parents[2]
    names = ("dm/eval/relation_r4_verification.py", "dm/data/relation.py",
             "dm/relation/candidates.py", "dm/relation/transducer.py", "dm/relation/spec.py")
    return {
        "schema": 1, "kind": "r4_model_blind_extraction_verification", "model_constructed": False,
        "protocol_sha256": protocol["protocol_sha256"],
        "manifest": str(path), **digests,
        "rebuilt_payload_sha256": rebuilt_body["corpus_sha256"],
        "training_cases": len(rebuilt.of(corpus.STRATUM_TRAIN)),
        "candidate_parity": readings,
        "timing": {"rebuild_seconds": rebuild_seconds,
                   "candidate_seconds": time.perf_counter() - started - rebuild_seconds},
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "source_hashes": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                          for name in names},
    }
