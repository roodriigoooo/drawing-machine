"""Verified inputs and immutable identities for state-study measurements.

The state drivers used to rebuild a corpus and then copy the checkpoint's
fingerprint into their output.  That made a changed generator look like the
old experiment.  This Module owns the check once, and the drivers remain thin
Adapters around it.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import scipy
import torch

from ..data.fingerprint import fingerprint
from ..eval.records import corpus_config
from ..isa.spec import Op, spec_for
from ..isa.state import LanguagePolicy
from ..train import build_data


@dataclass(frozen=True)
class VerifiedCorpus:
    """The exact input and policy used by a state measurement."""

    config: Any
    train: list[bytes]
    val: list[bytes]
    fingerprint: dict
    policy: LanguagePolicy
    train_opcodes: tuple[str, ...]
    val_opcodes: tuple[str, ...]


def opcode_names(programs: list[bytes]) -> tuple[str, ...]:
    """Return the parsed opcode set, refusing malformed programs."""
    used: set[Op] = set()
    for index, program in enumerate(programs):
        pc = 0
        while pc < len(program):
            try:
                spec = spec_for(program[pc])
            except Exception as exc:  # ISAError subclasses vary by version.
                raise ValueError(
                    f"program {index} has an unparsable opcode at byte {pc}"
                ) from exc
            if pc + spec.size > len(program):
                raise ValueError(
                    f"program {index} is truncated at byte {pc} ({spec.mnemonic})"
                )
            used.add(spec.op)
            pc += spec.size
    return tuple(op.name for op in sorted(used))


def verify_corpus(record: dict) -> VerifiedCorpus:
    """Rebuild and verify the train/validation corpus named by ``record``.

    Both split fingerprints and both split opcode policies are checked.  A
    report cannot proceed with a copied label, a changed generator default, or
    a train/validation policy mismatch.
    """
    config = corpus_config(record)
    train, val = build_data(config)
    actual = fingerprint(train, val)
    expected = record.get("corpus")
    if not isinstance(expected, dict) or expected != actual:
        raise ValueError(
            f"rebuilt corpus for {record.get('name', '?')} differs from its "
            f"record fingerprint: expected {expected!r}, got {actual!r}"
        )
    train_opcodes = opcode_names(train)
    val_opcodes = opcode_names(val)
    if train_opcodes != val_opcodes:
        raise ValueError(
            f"train/validation opcode policies differ for {record.get('name', '?')}: "
            f"train={train_opcodes}, val={val_opcodes}"
        )
    policy = LanguagePolicy.from_programs(
        train + val, label=f"{config.data}:{record.get('name', '?')}"
    )
    return VerifiedCorpus(
        config=config,
        train=train,
        val=val,
        fingerprint=actual,
        policy=policy,
        train_opcodes=train_opcodes,
        val_opcodes=val_opcodes,
    )


def sha256_file(path: Path) -> str:
    """Content hash an artifact, not its timestamp or filename."""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_hashes(paths: list[Path]) -> dict[str, str]:
    return {str(path): sha256_file(path) for path in paths}


def canonical_digest(body: Mapping[str, Any], exclude: str) -> str:
    """SHA-256 over a report's canonical payload, with its own digest field left out.

    The same shape `dm.eval.feedback_contract.digest_of` and the corpus manifest
    already use, and for the same reason: a document cannot contain a hash of
    itself, so the hash is taken over everything else. It is *not* the file
    hash — reformat the file and this is unchanged — and the two are reported
    under separate names wherever both appear.
    """
    payload = {key: value for key, value in body.items() if key != exclude}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def combined_source_digest(hashes: Mapping[str, str]) -> str:
    """One digest over a whole set of source hashes.

    A per-file map answers "which file changed"; a single digest answers "is this
    the same code", which is what a freeze compares and what a report can quote
    in one line. Canonical JSON with sorted keys, so the digest is a function of
    the mapping and not of the order a caller happened to build it in.
    """
    payload = json.dumps(dict(sorted(hashes.items())), separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def verify_frozen_sources(protocol: dict, paths: tuple[Path, ...]) -> None:
    """Refuse a measurement when selected freeze does not name current code.

    Historical protocols remain immutable records. Reproducing one after code
    changes requires checking out its source revision; silently running today's
    implementation under yesterday's digest recreates C4's freeze breach.
    """
    frozen = protocol.get("source_sha256")
    if not isinstance(frozen, dict):
        raise TypeError("protocol does not freeze source SHA-256 digests")
    missing = [str(path) for path in paths if str(path) not in frozen]
    drifted = [str(path) for path in paths
               if str(path) in frozen and sha256_file(path) != frozen[str(path)]]
    if missing or drifted:
        details = []
        if missing:
            details.append(f"missing source digests: {missing}")
        if drifted:
            details.append(f"source digest mismatch: {drifted}")
        raise ValueError("; ".join(details))


def environment(*, argv: list[str], device: str,
                source_paths: list[Path]) -> dict:
    """Portable execution provenance for a report."""
    return {
        "argv": list(argv),
        "cwd": str(Path.cwd()),
        "python": sys.version,
        "torch": torch.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "device": str(device),
        "source_hashes": source_hashes(source_paths),
    }


def identity(*, record_path: Path, checkpoint_path: Path,
             corpus_fingerprint: dict, policy_digest: str, cells: list[str],
             seeds: list[int], n: int, cap: int, sampler: dict,
             protocol_digest: str, source_paths: list[Path]) -> dict:
    """Return the canonical payload and its short content-addressed key."""
    payload = {
        "record_sha256": sha256_file(record_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "corpus": corpus_fingerprint,
        "policy_digest": policy_digest,
        "cells": sorted(cells),
        "draw_seeds": list(seeds),
        "n": n,
        "cap": cap,
        "sampler": sampler,
        "protocol_digest": protocol_digest,
        "source_hashes": source_hashes(source_paths),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["identity"] = hashlib.sha256(encoded).hexdigest()[:20]
    return payload


__all__ = [
    "VerifiedCorpus",
    "environment",
    "identity",
    "opcode_names",
    "sha256_file",
    "source_hashes",
    "verify_corpus",
    "verify_frozen_sources",
]
