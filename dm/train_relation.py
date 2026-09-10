"""R4 training-side adaptation and exact separately-normalized objective.

The runtime relation package remains target-free.  This module is the only
place where frozen corpus actions become supervision, and it keeps every join
key explicit so length bucketing cannot associate a case with another case's
copy metadata.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import platform
import random
import sys
import tempfile
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import InitVar, asdict, dataclass, field
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .data import relation as corpus
from .data.dataset import BUCKET_EPOCH_ALGORITHM, bucket_epoch_plan
from .eval.relation_contract import (
    corpus_seed_faults,
)
from .eval.relation_evidence import optimizer_digest, rng_digest, state_digest
from .eval.relation_training_evidence import (
    COMPONENT_NAMES,
    COMPONENT_WEIGHTS,
    DIAGNOSTIC_POLICY_EXPLICIT,
    DIAGNOSTIC_POLICY_SCHEDULE,
    RELATION_CHECKPOINT_SCHEMA,
    REQUIRED_BUNDLE_KEYS,
    SCHEDULE_IDENTITY,
    SEED_NAMESPACE_ENGINEERING,
    RelationTrainingEvidenceRefused,
    _finite_nonnegative,
    default_component_events,
    validate_bundle_shape,
    validate_provenance_and_seed,
)
from .isa.codec import BOS, N_SPECIAL, PAD, ByteCodec
from .isa.transform import D4, Transform
from .models.relation import (
    COPY,
    EMIT,
    FACTOR_NAMES,
    RELATION_MAX_CANDIDATE_SPANS,
    EquivalentAction,
    PackedCandidates,
    RelationScores,
    factor_marginal_nll,
    joint_valid_nll,
    length_bin,
)
from .models.transformer import Config, DrawingLM
from .relation.candidates import candidate_spans
from .relation.resources import (
    ResourceObserver,
    resource_phase,
    validate_sample_interval,
)
from .relation.spec import COUNT_SUPPORT, D4_SUPPORT, TRANSLATION_SUPPORT
from .relation.transducer import TransducerFault, prefix_index
from .train import lr_at


class RelationTrainingRefused(ValueError):
    """An expected R4 corpus/batch/checkpoint boundary refusal."""


class RelationTrainingComplete(RelationTrainingRefused):
    """The declared step horizon is complete; no further update is permitted."""


SUPERVISION_IDENTITY_VERSION = "r4_training_supervision_v1"


_VERIFIED_MANIFEST = object()


def _validate_immutable_case(case: corpus.RelationCase) -> None:
    """Validate the nested graph once, before identities or cached targets exist."""
    valid = all(type(getattr(case, name)) is str for name in (
        "case_id", "venue", "stratum", "source_case_id", "donor_case_id"))
    valid &= type(case.flat) is bytes and type(case.structured) is bytes
    valid &= type(case.novel_pair) is bool
    valid &= type(case.target_start) is int and type(case.target_stop) is int
    for name in ("groups", "source_ids", "donor_groups", "donor_source_ids"):
        values = getattr(case, name)
        valid &= type(values) is tuple and all(type(value) is str for value in values)
    valid &= type(case.plan) is tuple and all(
        type(row) is tuple and len(row) == 4 and all(type(value) is int for value in row)
        for row in case.plan)
    valid &= type(case.actions) is tuple
    if not valid:
        raise RelationTrainingRefused("training case requires immutable canonical fields")
    for action in case.actions:
        if type(action) is not corpus.CopyAction or not all(
                type(getattr(action, name)) is int for name in (
                    "boundary", "source_start", "source_stop", "total_count",
                    "target_start", "target_stop")):
            raise RelationTrainingRefused("training action requires immutable canonical fields")
        step = action.step
        if type(step) is not Transform or type(step.d4) is not D4 or \
                type(step.d4.turns) is not int or type(step.d4.mirror) is not bool or \
                type(step.dx) is not int or type(step.dy) is not int:
            raise RelationTrainingRefused("training transform requires immutable canonical fields")


@dataclass(frozen=True)
class TrainingCorpus:
    """A rebuilt, manifest-bound training-only corpus."""

    cases: tuple[corpus.RelationCase, ...]
    manifest_path: Path | None
    canonical_payload_sha256: str | None
    file_sha256: str | None
    rebuilt_payload_sha256: str | None
    provenance: str
    _verification: InitVar[object] = None

    def __post_init__(self, _verification: object) -> None:
        if type(self.cases) is not tuple or not self.cases or \
                not all(type(case) is corpus.RelationCase for case in self.cases):
            raise RelationTrainingRefused("training corpus needs a nonempty tuple of relation cases")
        for case in self.cases:
            _validate_immutable_case(case)
        if self.provenance not in ("engineering", "development", "scientific"):
            raise RelationTrainingRefused(f"unknown training corpus provenance {self.provenance!r}")
        manifest_values = (
            self.canonical_payload_sha256, self.file_sha256, self.rebuilt_payload_sha256,
        )
        if self.provenance == "engineering":
            if self.manifest_path is not None or any(value is not None for value in manifest_values):
                raise RelationTrainingRefused(
                    "explicit engineering training corpus cannot claim a manifest identity")
        elif _verification is not _VERIFIED_MANIFEST or not isinstance(self.manifest_path, Path) or not all(
                isinstance(value, str) and len(value) == 64 for value in manifest_values):
            raise RelationTrainingRefused(
                "manifest-backed training corpus requires its path and all manifest identities")

    @property
    def program_fingerprint(self) -> str:
        digest = hashlib.sha256(b"drawing-machine-r4-training-programs-v1")
        for case in self.cases:
            digest.update(case.case_id.encode("ascii"))
            digest.update(len(case.flat).to_bytes(8, "big"))
            digest.update(case.flat)
        return digest.hexdigest()

    @property
    def supervision_identity(self) -> str:
        """Versioned ordered identity for bytes, action supervision and provenance.

        ``program_fingerprint`` intentionally remains the named byte-only
        measure. This identity additionally binds every fact the adapter uses
        to construct future target-bearing plans, plus corpus provenance.
        """
        digest = hashlib.sha256(SUPERVISION_IDENTITY_VERSION.encode())
        provenance = {
            "provenance": self.provenance,
            "manifest": None if self.manifest_path is None else str(self.manifest_path),
            "canonical_payload_sha256": self.canonical_payload_sha256,
            "file_sha256": self.file_sha256,
            "rebuilt_payload_sha256": self.rebuilt_payload_sha256,
        }
        digest.update(json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode())
        for index, case in enumerate(self.cases):
            digest.update(index.to_bytes(8, "big"))
            digest.update(case.case_id.encode("utf-8"))
            digest.update(len(case.flat).to_bytes(8, "big"))
            digest.update(case.flat)
            actions = [
                (*action.key(), action.target_start, action.target_stop)
                for action in case.actions
            ]
            planning_provenance = {
                "venue": case.venue,
                "stratum": case.stratum,
                "groups": case.groups,
                "plan": case.plan,
                "source_ids": case.source_ids,
                "novel_pair": case.novel_pair,
                "source_case_id": case.source_case_id,
                "donor_case_id": case.donor_case_id,
                "donor_groups": case.donor_groups,
                "donor_source_ids": case.donor_source_ids,
                "target": (case.target_start, case.target_stop),
                "actions": actions,
            }
            digest.update(json.dumps(planning_provenance, sort_keys=True,
                                     separators=(",", ":")).encode())
        return digest.hexdigest()

    @classmethod
    def engineering_cases(cls, cases: Sequence[corpus.RelationCase]) -> TrainingCorpus:
        """Explicit fixture-only constructor; never accepts a manifest digest."""
        frozen = tuple(cases)
        if not frozen:
            raise RelationTrainingRefused("an engineering fixture needs at least one case")
        return cls(frozen, None, None, None, None, "engineering")


def load_training_corpus(path: Path, *, provenance: str = "scientific") -> TrainingCorpus:
    """Strictly load, seed-bind, rebuild and select frozen ``train`` rows.

    A manifest contains no program bytes.  Thus a validated manifest digest is
    necessary but insufficient: the complete deterministic rebuild is compared
    before a training case can escape this function.
    """
    try:
        snapshot = corpus.read_manifest_snapshot(path)
    except corpus.ManifestRefused as exc:
        raise RelationTrainingRefused(str(exc)) from exc
    try:
        body = corpus._validated_manifest_body(snapshot)
    except corpus.ManifestRefused as exc:
        raise RelationTrainingRefused(str(exc)) from exc
    faults = corpus_seed_faults(body, provenance=provenance)
    if faults:
        raise RelationTrainingRefused("; ".join(faults))
    rebuilt = corpus.build(corpus.manifest_config(body))
    rebuilt_body = corpus.manifest(rebuilt)
    if rebuilt_body.get("corpus_sha256") != body.get("corpus_sha256"):
        raise RelationTrainingRefused(
            "rebuilt relation corpus payload does not equal the loaded frozen manifest")
    # Canonical JSON equality catches a case order or non-digested shape change
    # just as explicitly as the payload hash does.
    if json.dumps(rebuilt_body, sort_keys=True, separators=(",", ":")) != json.dumps(
            body, sort_keys=True, separators=(",", ":")):
        raise RelationTrainingRefused("rebuilt relation manifest body is not canonical-equal")
    cases = rebuilt.of(corpus.STRATUM_TRAIN)
    if not cases:
        raise RelationTrainingRefused("the frozen corpus contains no train cases")
    return TrainingCorpus(
        cases=cases,
        manifest_path=Path(snapshot.path),
        canonical_payload_sha256=snapshot.canonical_payload_sha256,
        file_sha256=snapshot.file_sha256,
        rebuilt_payload_sha256=rebuilt_body["corpus_sha256"],
        provenance=provenance,
        _verification=_VERIFIED_MANIFEST,
    )


@dataclass(frozen=True)
class ActionTarget:
    """CPU-only supervision metadata; never accepted by predicted decoding."""

    source_start: int
    source_stop: int
    d4: int
    dx: int
    dy: int
    total_count: int

    @classmethod
    def from_action(cls, action: corpus.CopyAction) -> ActionTarget:
        return cls(action.source_start, action.source_stop, action.step.d4.code,
                   action.step.dx, action.step.dy, action.total_count)


@dataclass(frozen=True)
class QueryPlan:
    row: int
    case_index: int
    boundary: int
    gate_target: int
    spans: tuple[tuple[int, int], ...]
    equivalents: tuple[ActionTarget, ...]


@dataclass(frozen=True)
class RelationBatchPlan:
    """One immutable CPU metadata plan joined to a padded byte batch by index."""

    case_indices: Tensor
    inputs: Tensor
    targets: Tensor
    queries: tuple[QueryPlan, ...]

    def validate(self) -> None:
        if self.case_indices.dtype != torch.long or self.case_indices.ndim != 1:
            raise RelationTrainingRefused("case_indices must be CPU int64 [rows]")
        if self.case_indices.device.type != "cpu":
            raise RelationTrainingRefused("case_indices audit metadata must remain on CPU")
        if self.inputs.ndim != 2 or self.targets.shape != self.inputs.shape:
            raise RelationTrainingRefused("inputs and targets must be equally shaped [rows, T]")
        if self.inputs.dtype != torch.long or self.targets.dtype != torch.long:
            raise RelationTrainingRefused("byte tensors must be int64")
        if self.inputs.device.type != "cpu" or self.targets.device.type != "cpu":
            raise RelationTrainingRefused("batch plans and audit metadata must remain on CPU")
        if self.inputs.shape[0] != self.case_indices.numel():
            raise RelationTrainingRefused("batch rows and case indices disagree")
        if len(set(self.case_indices.tolist())) != self.case_indices.numel():
            raise RelationTrainingRefused("one batch may not repeat a dataset case index")
        if bool((self.case_indices < 0).any()) or not self.inputs.numel():
            raise RelationTrainingRefused("batch indices/shape must be nonempty and nonnegative")
        row_marks: list[set[int]] = []
        for inputs, targets in zip(self.inputs.tolist(), self.targets.tolist(), strict=True):
            length = next((i for i, token in enumerate(targets) if token == PAD), len(targets))
            if not length or any(token != PAD for token in targets[length:]) or any(
                    token != PAD for token in inputs[length:]):
                raise RelationTrainingRefused("byte padding must be a nonempty row's suffix")
            if any(not N_SPECIAL <= token < ByteCodec.vocab_size for token in targets[:length]):
                raise RelationTrainingRefused("targets must be ByteCodec content symbols")
            if inputs[:length] != [BOS, *targets[:length - 1]]:
                raise RelationTrainingRefused("inputs are not the exact BOS-shifted target bytes")
            try:
                marks, _, _ = prefix_index(bytes(token - N_SPECIAL for token in targets[:length]))
            except TransducerFault as exc:
                raise RelationTrainingRefused(f"malformed training row: {exc}") from exc
            row_marks.append(set(marks[:-1]))  # no post-program query
        seen_queries: set[tuple[int, int]] = set()
        for query in self.queries:
            if not isinstance(query, QueryPlan) or any(type(value) is not int for value in
                    (query.row, query.case_index, query.boundary, query.gate_target)):
                raise RelationTrainingRefused("query coordinates and gate must be integers")
            if not 0 <= query.row < self.inputs.shape[0]:
                raise RelationTrainingRefused("query row is outside the padded batch")
            if int(self.case_indices[query.row]) != query.case_index:
                raise RelationTrainingRefused("query case index does not own its batch row")
            if query.boundary < 0 or query.boundary >= self.inputs.shape[1]:
                raise RelationTrainingRefused("query boundary is outside BOS/byte state positions")
            if query.boundary not in row_marks[query.row]:
                raise RelationTrainingRefused("query is not a reachable content instruction boundary")
            key = query.row, query.boundary
            if key in seen_queries:
                raise RelationTrainingRefused("duplicate query boundary")
            seen_queries.add(key)
            if query.gate_target not in (EMIT, COPY):
                raise RelationTrainingRefused("gate target is not EMIT/COPY")
            if len(query.spans) > RELATION_MAX_CANDIDATE_SPANS:
                raise RelationTrainingRefused("query exceeds the frozen candidate-span cap")
            span_set = set()
            for span in query.spans:
                if not isinstance(span, tuple) or len(span) != 2 or any(
                        type(value) is not int for value in span):
                    raise RelationTrainingRefused("candidate endpoints must be integer pairs")
                start, stop = span
                if not 0 <= start < stop <= query.boundary or any(
                        value not in row_marks[query.row] for value in span):
                    raise RelationTrainingRefused("candidate span is not wholly in the canonical prefix")
                if span in span_set:
                    raise RelationTrainingRefused("duplicate candidate span")
                span_set.add(span)
            if query.gate_target == COPY and not query.equivalents:
                raise RelationTrainingRefused("COPY query has no joint equivalence set")
            if query.gate_target == EMIT and query.equivalents:
                raise RelationTrainingRefused("EMIT query carries target action metadata")
            seen_actions = set()
            for action in query.equivalents:
                if not isinstance(action, ActionTarget) or any(type(value) is not int for value in
                        (action.source_start, action.source_stop, action.d4, action.dx,
                         action.dy, action.total_count)):
                    raise RelationTrainingRefused("joint action fields must be integers")
                if (action.source_start, action.source_stop) not in span_set or \
                        action.d4 not in D4_SUPPORT or action.dx not in TRANSLATION_SUPPORT or \
                        action.dy not in TRANSLATION_SUPPORT or action.total_count not in COUNT_SUPPORT:
                    raise RelationTrainingRefused("joint action is not query-local or in support")
                if action in seen_actions:
                    raise RelationTrainingRefused("duplicate joint action key")
                seen_actions.add(action)

    @property
    def content_bytes(self) -> int:
        return int((self.targets != PAD).sum())

    @property
    def reachable_boundaries(self) -> int:
        return len(self.queries)

    @property
    def positive_boundaries(self) -> int:
        return sum(query.gate_target == COPY for query in self.queries)


def build_batch_plan(training: TrainingCorpus, indices: Sequence[int], *,
                     max_len: int, codec: ByteCodec | None = None) -> RelationBatchPlan:
    """Encode selected frozen rows and precompute all target-bearing metadata.

    A fresh planner with retention disabled: this is the reference build that
    every cache policy must reproduce exactly.
    """
    return RelationBatchPlanner(training, max_len=max_len, codec=codec, max_cached_cases=0,
                                max_cached_spans=0, max_cached_bytes=0).batch(indices)


@dataclass(frozen=True)
class CaseQuery:
    """Row-free, case-local query metadata; the unit of cached semantic identity."""

    boundary: int
    gate_target: int
    spans: tuple[tuple[int, int], ...]
    equivalents: tuple[ActionTarget, ...]


@dataclass(frozen=True)
class CaseMetadata:
    """Everything derived once from one immutable corpus case."""

    sequence: tuple[int, ...]
    queries: tuple[CaseQuery, ...]

    @property
    def spans(self) -> int:
        return sum(len(query.spans) for query in self.queries)


#: Charge method published beside every retained-charge reading.  It is a
#: conservative deep Python-object size, deduplicated within one entry; it is
#: not a bound on process RSS, allocator arenas, the verified corpus, the
#: current batch or model/optimizer storage.
RETENTION_CHARGE_METHOD = "python_deep_getsizeof_v1"
_CHARGEABLE_DATACLASSES = (CaseMetadata, CaseQuery, ActionTarget)


def retained_charge(value: object) -> int:
    """Closed-type deep byte charge of one immutable metadata object graph.

    Ints, tuples and the three metadata dataclasses are the only admissible
    types; dataclass instances are charged with their ``__dict__`` storage.
    Aliases within the graph are counted once.  Anything else -- a tensor, a
    list, a model reference -- is an internal invariant failure, not a refusal:
    the trainer, not a caller, decides what enters the cache.
    """
    seen: set[int] = set()

    def visit(obj: object) -> int:
        if id(obj) in seen:
            return 0
        seen.add(id(obj))
        if type(obj) is int:
            return sys.getsizeof(obj)
        if type(obj) is tuple:
            return sys.getsizeof(obj) + sum(visit(item) for item in obj)
        if type(obj) in _CHARGEABLE_DATACLASSES:
            fields = obj.__dict__
            return sys.getsizeof(obj) + sys.getsizeof(fields) + sum(
                visit(item) for item in fields.values())
        raise RuntimeError(
            f"retained metadata contains an unchargeable {type(obj).__name__}; the cache "
            "may hold immutable integer metadata only")

    return visit(value)


@dataclass(frozen=True)
class RelationCacheSnapshot:
    """Read-only accounting of one planner's retention; never the LRU map itself."""

    charge_method: str
    max_cached_cases: int
    max_cached_spans: int
    max_cached_bytes: int
    enabled: bool
    requested_cases: int
    hits: int
    misses: int
    admissions: int
    evictions: int
    disabled_bypasses: int
    oversize_bypasses: int
    resident_cases: int
    resident_spans: int
    retained_charge_bytes: int
    index_charge_bytes: int
    high_water_cases: int
    high_water_spans: int
    high_water_charge_bytes: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class RelationBatchPlanner:
    """Trainer-owned LRU of immutable metadata with three explicit retention limits.

    Only integer tuples and byte encodings survive a batch. Differentiable states
    are always gathered from the current trunk forward. The full corpus contains
    millions of spans: retaining every row would turn a compute optimization
    into a host-memory regression. Eviction changes work, never supervision.

    Any zero limit disables retention.  A disabled planner builds the identical
    batch; it is the reference mode rather than a separate implementation.
    """

    _INDEX_BASELINE = sys.getsizeof(OrderedDict())
    _canonical_index_bytes: ClassVar[dict[int, int]] = {}

    def __init__(self, training: TrainingCorpus, *, max_len: int,
                 codec: ByteCodec | None = None, max_cached_spans: int = 65_536,
                 max_cached_cases: int = 256, max_cached_bytes: int = 16 * 2**20) -> None:
        if type(max_len) is not int or max_len < 2:
            raise RelationTrainingRefused("max_len must leave BOS plus one target byte")
        if type(training) is not TrainingCorpus:
            raise RelationTrainingRefused("a planner binds exactly one TrainingCorpus")
        self.training, self.max_len = training, max_len
        self.codec = ByteCodec() if codec is None else codec
        if type(self.codec) is not ByteCodec:
            raise RelationTrainingRefused("R4 relation training is byte-codec only")
        for value in (max_cached_spans, max_cached_cases, max_cached_bytes):
            if type(value) is not int or value < 0:
                raise RelationTrainingRefused(
                    "metadata cache limits must be nonnegative integers (bool/float refused)")
        self.max_cached_spans, self.max_cached_cases = max_cached_spans, max_cached_cases
        self.max_cached_bytes = max_cached_bytes
        self.enabled = bool(max_cached_spans and max_cached_cases and max_cached_bytes)
        self._cases: OrderedDict[int, tuple[CaseMetadata, int]] = OrderedDict()
        self._resident_spans = 0
        self._entry_charge = 0
        self._counters = {name: 0 for name in (
            "requested_cases", "hits", "misses", "admissions", "evictions",
            "disabled_bypasses", "oversize_bypasses", "high_water_cases", "high_water_spans",
            "high_water_charge_bytes")}

    # -- accounting ---------------------------------------------------------

    @classmethod
    def _index_bytes_for(cls, entries: int) -> int:
        """Container bytes of a compact LRU index holding ``entries`` keys."""
        if entries not in cls._canonical_index_bytes:
            cls._canonical_index_bytes[entries] = (
                sys.getsizeof(OrderedDict.fromkeys(range(entries))) - cls._INDEX_BASELINE)
        return cls._canonical_index_bytes[entries]

    @property
    def index_charge_bytes(self) -> int:
        return sys.getsizeof(self._cases) - self._INDEX_BASELINE

    @property
    def retained_charge_bytes(self) -> int:
        return self._entry_charge + self.index_charge_bytes

    @property
    def cached_spans(self) -> int:
        return self._resident_spans

    def snapshot(self) -> RelationCacheSnapshot:
        self._assert_bounded()
        return RelationCacheSnapshot(
            charge_method=RETENTION_CHARGE_METHOD, max_cached_cases=self.max_cached_cases,
            max_cached_spans=self.max_cached_spans, max_cached_bytes=self.max_cached_bytes,
            enabled=self.enabled, resident_cases=len(self._cases),
            resident_spans=self._resident_spans,
            retained_charge_bytes=self.retained_charge_bytes,
            index_charge_bytes=self.index_charge_bytes, **self._counters)

    def _assert_bounded(self) -> None:
        if len(self._cases) > self.max_cached_cases or \
                self._resident_spans > self.max_cached_spans or \
                self.retained_charge_bytes > self.max_cached_bytes:
            raise RuntimeError("metadata cache exceeded a declared retention limit")
        if not self.enabled and self._cases:
            raise RuntimeError("a disabled metadata cache retained an entry")

    def _mark_high_water(self) -> None:
        counters = self._counters
        counters["high_water_cases"] = max(counters["high_water_cases"], len(self._cases))
        counters["high_water_spans"] = max(counters["high_water_spans"], self._resident_spans)
        counters["high_water_charge_bytes"] = max(counters["high_water_charge_bytes"],
                                                  self.retained_charge_bytes)

    # -- lifecycle ----------------------------------------------------------

    def _charge_for(self, index: int, metadata: CaseMetadata) -> int:
        partial = retained_charge(metadata) + sys.getsizeof(index) + sys.getsizeof((None, None))
        charge = partial + sys.getsizeof(partial)
        if sys.getsizeof(charge) != sys.getsizeof(partial):  # pragma: no cover -- width step
            charge += sys.getsizeof(charge) - sys.getsizeof(partial)
        return charge

    def _lookup(self, index: int) -> CaseMetadata:
        """Serve one case's immutable metadata, updating recency and counters."""
        entry = self._cases.get(index)
        if entry is not None:
            self._cases.move_to_end(index)
            self._counters["requested_cases"] += 1
            self._counters["hits"] += 1
            return entry[0]
        # Derive row-free metadata, validate complete semantics, and calculate
        # admission charge BEFORE any admission or counter increment.
        metadata = _derive_case_metadata(self, index)
        _validate_case_metadata(metadata, self.max_len)
        charge = self._charge_for(index, metadata)

        self._counters["requested_cases"] += 1
        self._counters["misses"] += 1
        if not self.enabled:
            self._counters["disabled_bypasses"] += 1
            return metadata
        if metadata.spans > self.max_cached_spans or charge + self._index_bytes_for(1) > self.max_cached_bytes:
            self._counters["oversize_bypasses"] += 1
            return metadata
        self._admit(index, metadata, charge)
        return metadata

    def _admit(self, index: int, metadata: CaseMetadata, charge: int | None = None) -> None:
        spans = metadata.spans
        if charge is None:
            charge = self._charge_for(index, metadata)
            if spans > self.max_cached_spans or charge + self._index_bytes_for(1) > self.max_cached_bytes:
                self._counters["oversize_bypasses"] += 1
                return
        while self._cases and (
                len(self._cases) + 1 > self.max_cached_cases or
                self._resident_spans + spans > self.max_cached_spans or
                self._entry_charge + charge + self._index_bytes_for(len(self._cases) + 1)
                > self.max_cached_bytes):
            self._evict_oldest()
        self._cases[index] = (metadata, charge)
        self._resident_spans += spans
        self._entry_charge += charge
        self._counters["admissions"] += 1
        self._compact_index()
        self._mark_high_water()
        self._assert_bounded()

    def _evict_oldest(self) -> None:
        _, (evicted, charge) = self._cases.popitem(last=False)
        self._resident_spans -= evicted.spans
        self._entry_charge -= charge
        self._counters["evictions"] += 1

    def _compact_index(self) -> None:
        """Dict storage never shrinks on deletion; rebuild when it exceeds compact size."""
        if self.index_charge_bytes > self._index_bytes_for(len(self._cases)):
            self._cases = OrderedDict(self._cases.items())

    def batch(self, indices: Sequence[int]) -> RelationBatchPlan:
        return _build_batch_plan(self, indices)


def _validate_case_metadata(metadata: object, max_len: int) -> None:
    """Validate complete encoding, query, span, and action semantics of one row-free metadata entry."""
    if not isinstance(metadata, CaseMetadata):
        raise RelationTrainingRefused("metadata must be a CaseMetadata instance")
    sequence = metadata.sequence
    if not isinstance(sequence, tuple) or len(sequence) < 2 or len(sequence) > max_len:
        raise RelationTrainingRefused(
            f"case sequence needs {len(sequence) if isinstance(sequence, tuple) else type(sequence)} tokens, above max_len={max_len}")
    if sequence[0] != BOS:
        raise RelationTrainingRefused("case sequence must start with BOS")
    targets = sequence[1:]
    if any(type(token) is not int or not N_SPECIAL <= token < ByteCodec.vocab_size for token in targets):
        raise RelationTrainingRefused("targets must be ByteCodec content symbols")
    try:
        marks, _, _ = prefix_index(bytes(token - N_SPECIAL for token in targets))
    except TransducerFault as exc:
        raise RelationTrainingRefused(f"malformed training row: {exc}") from exc
    row_marks = set(marks[:-1])

    if not isinstance(metadata.queries, tuple) or not metadata.queries:
        raise RelationTrainingRefused("case metadata has an empty query sequence")

    seen_queries: set[int] = set()
    for query in metadata.queries:
        if not isinstance(query, CaseQuery) or any(type(value) is not int for value in
                (query.boundary, query.gate_target)):
            raise RelationTrainingRefused("query boundary and gate must be integers")
        if query.boundary < 0 or query.boundary >= len(targets):
            raise RelationTrainingRefused("query boundary is outside byte state positions")
        if query.boundary not in row_marks:
            raise RelationTrainingRefused("query is not a reachable content instruction boundary")
        if query.boundary in seen_queries:
            raise RelationTrainingRefused("duplicate query boundary")
        seen_queries.add(query.boundary)
        if query.gate_target not in (EMIT, COPY):
            raise RelationTrainingRefused("gate target is not EMIT/COPY")
        if not isinstance(query.spans, tuple) or len(query.spans) > RELATION_MAX_CANDIDATE_SPANS:
            raise RelationTrainingRefused("query exceeds the frozen candidate-span cap")
        span_set = set()
        for span in query.spans:
            if not isinstance(span, tuple) or len(span) != 2 or any(
                    type(value) is not int for value in span):
                raise RelationTrainingRefused("candidate endpoints must be integer pairs")
            start, stop = span
            if not 0 <= start < stop <= query.boundary or start not in row_marks or stop not in row_marks:
                raise RelationTrainingRefused("candidate span is not wholly in the canonical prefix")
            if span in span_set:
                raise RelationTrainingRefused("duplicate candidate span")
            span_set.add(span)
        if not isinstance(query.equivalents, tuple):
            raise RelationTrainingRefused("query equivalents must be a tuple")
        if query.gate_target == COPY and not query.equivalents:
            raise RelationTrainingRefused("COPY query has no joint equivalence set")
        if query.gate_target == EMIT and query.equivalents:
            raise RelationTrainingRefused("EMIT query carries target action metadata")
        seen_actions = set()
        for action in query.equivalents:
            if not isinstance(action, ActionTarget) or any(type(value) is not int for value in
                    (action.source_start, action.source_stop, action.d4, action.dx,
                     action.dy, action.total_count)):
                raise RelationTrainingRefused("joint action fields must be integers")
            if (action.source_start, action.source_stop) not in span_set or \
                    action.d4 not in D4_SUPPORT or action.dx not in TRANSLATION_SUPPORT or \
                    action.dy not in TRANSLATION_SUPPORT or action.total_count not in COUNT_SUPPORT:
                raise RelationTrainingRefused("joint action is not query-local or in support")
            if action in seen_actions:
                raise RelationTrainingRefused("duplicate joint action key")
            seen_actions.add(action)


def _derive_case_metadata(planner: RelationBatchPlanner, index: int) -> CaseMetadata:
    """Derive one case's complete immutable supervision metadata; no caching here."""
    case = planner.training.cases[index]
    sequence = tuple(planner.codec.with_bos(case.flat))
    if len(sequence) > planner.max_len:
        raise RelationTrainingRefused(
            f"case {case.case_id} needs {len(sequence)} tokens, above max_len={planner.max_len}")
    actions = {action.boundary: action for action in case.actions}
    if len(actions) != len(case.actions):
        raise RelationTrainingRefused(f"case {case.case_id} schedules two actions at one boundary")
    covered = corpus.covered_boundaries(case.actions)
    queries: list[CaseQuery] = []
    for boundary in corpus.boundaries(case.flat):
        if boundary >= len(case.flat) or boundary in covered:
            continue
        action = actions.get(boundary)
        spans = candidate_spans(case.flat[:boundary], boundary)
        equivalents: tuple[ActionTarget, ...] = ()
        gate = EMIT
        if action is not None:
            gate = COPY
            derived = corpus.derivations(case.flat, boundary, action.target_start,
                                         action.target_stop)
            equivalents = tuple(ActionTarget.from_action(item) for item in derived)
            if not equivalents:
                raise RelationTrainingRefused(
                    f"case {case.case_id} action at {boundary} has no derivation")
            span_set = set(spans)
            if any((item.source_start, item.source_stop) not in span_set
                   for item in equivalents):
                raise RelationTrainingRefused(
                    f"case {case.case_id} derivation is outside its query candidate set")
            if len({(item.source_start, item.source_stop, item.d4, item.dx,
                    item.dy, item.total_count) for item in equivalents}) != len(equivalents):
                raise RelationTrainingRefused("duplicate joint action key before aggregation")
        queries.append(CaseQuery(boundary, gate, spans, equivalents))
    return CaseMetadata(sequence, tuple(queries))


def _build_batch_plan(planner: RelationBatchPlanner, indices: Sequence[int]) -> RelationBatchPlan:
    if not indices:
        raise RelationTrainingRefused("cannot plan an empty batch")
    if len(set(indices)) != len(indices):
        raise RelationTrainingRefused("a batch index sequence has a duplicate")
    for index in indices:
        if type(index) is not int or not 0 <= index < len(planner.training.cases):
            raise RelationTrainingRefused("batch index is outside the training corpus")
    # One lookup per case: a hit or a miss, never two independent lookups for
    # the encoding and the queries of the same request.
    metadata = [planner._lookup(index) for index in indices]
    width = max(len(item.sequence) - 1 for item in metadata)
    inputs = torch.full((len(metadata), width), PAD, dtype=torch.long)
    targets = torch.full_like(inputs, PAD)
    plans: list[QueryPlan] = []
    for row, (case_index, item) in enumerate(zip(indices, metadata, strict=True)):
        sequence = item.sequence
        inputs[row, :len(sequence) - 1] = torch.tensor(sequence[:-1], dtype=torch.long)
        targets[row, :len(sequence) - 1] = torch.tensor(sequence[1:], dtype=torch.long)
        # Rows are bound here and only here; cached identity carries no row.
        plans.extend(QueryPlan(row, case_index, query.boundary, query.gate_target,
                               query.spans, query.equivalents) for query in item.queries)
    result = RelationBatchPlan(torch.tensor(indices, dtype=torch.long), inputs, targets,
                               tuple(plans))
    result.validate()
    if not result.content_bytes or not result.reachable_boundaries:
        raise RelationTrainingRefused("training batch has an empty objective denominator")
    return result


def pack_candidates(plan: RelationBatchPlan, states: Tensor) -> tuple[PackedCandidates, tuple[tuple[EquivalentAction, ...], ...], Tensor]:
    """Gather differentiable states for one immutable CPU plan without padding spans."""
    plan.validate()
    if states.ndim != 3 or states.shape[:2] != plan.inputs.shape or not states.is_floating_point():
        raise RelationTrainingRefused("top states must align exactly with the plan's input rectangle")
    device = states.device
    q_rows = torch.tensor([query.row for query in plan.queries], dtype=torch.long, device=device)
    q_offsets = torch.tensor([query.boundary for query in plan.queries], dtype=torch.long,
                             device=device)
    if bool((plan.inputs[q_rows.cpu(), q_offsets.cpu()] == PAD).any()):
        raise RelationTrainingRefused("a query resolves to padding")
    query_states = states[q_rows, q_offsets]
    span_rows: list[int] = []
    lengths: list[int] = []
    source_start: list[int] = []
    source_stop: list[int] = []
    splits = [0]
    equivalents: list[tuple[EquivalentAction, ...]] = []
    gate_targets: list[int] = []
    for query_index, query in enumerate(plan.queries):
        by_span: dict[tuple[int, int], int] = {}
        for start, stop in query.spans:
            by_span[(start, stop)] = len(span_rows)
            span_rows.append(query.row)
            lengths.append(length_bin(stop - start))
            source_start.append(start)
            source_stop.append(stop)
        splits.append(len(span_rows))
        gate_targets.append(query.gate_target)
        if query.gate_target == COPY:
            alternatives = tuple(
                EquivalentAction(query_index, by_span[(item.source_start, item.source_stop)],
                                 item.d4, item.dx, item.dy, item.total_count)
                for item in query.equivalents
            )
            equivalents.append(alternatives)
    rows = torch.tensor(span_rows, dtype=torch.long, device=device)
    starts = torch.tensor(source_start, dtype=torch.long, device=device)
    stops = torch.tensor(source_stop, dtype=torch.long, device=device)
    packed = PackedCandidates(
        query_states=query_states,
        start_states=states[rows, starts],
        stop_states=states[rows, stops],
        length_bins=torch.tensor(lengths, dtype=torch.long, device=device),
        source_start=starts,
        source_stop=stops,
        row_splits=torch.tensor(splits, dtype=torch.long, device=device),
    )
    return packed, tuple(equivalents), torch.tensor(gate_targets, dtype=torch.long,
                                                      device=device)


FACTOR_STATUS_MEASURED = "measured"
FACTOR_STATUS_NO_POSITIVES = "no_positive_boundaries"
FACTOR_STATUS_NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class FactorDiagnostics:
    """Detached per-factor valid-value NLL sums for one logical batch.

    ``sums`` are raw natural-log sums over positive boundaries, accumulated in
    fixed chunk order; ``denominator`` is the whole batch's positive count.  A
    mean is derived by the reader.  ``not_applicable`` (no relation head) carries
    null sums; ``no_positive_boundaries`` carries exact zeros.  Neither is a
    zero-loss success.  Nothing here enters the objective or backward.
    """

    status: str
    denominator: int | None
    sums: dict[str, float] | None
    unit: str = "nats"

    def as_dict(self) -> dict[str, object]:
        return {"status": self.status, "unit": self.unit, "denominator": self.denominator,
                "sums": None if self.sums is None else dict(self.sums)}

    @classmethod
    def not_applicable(cls) -> FactorDiagnostics:
        return cls(FACTOR_STATUS_NOT_APPLICABLE, None, None)


@dataclass(frozen=True)
class ComponentGradient:
    """Aggregate shared-trunk L2 norm of one normalized, unweighted objective term."""

    status: str
    norm: float | None
    denominator: int | None
    weight: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ComponentGradients:
    """One event's component measurements plus their deterministic logical cost."""

    components: dict[str, ComponentGradient]
    extra_backward_calls: int
    buffer_bytes: int

    def as_dict(self) -> dict[str, object]:
        return {"components": {name: item.as_dict() for name, item in self.components.items()},
                "extra_backward_calls": self.extra_backward_calls,
                "buffer_bytes": self.buffer_bytes}


@dataclass(frozen=True)
class TrunkIdentity:
    """The checked shared-trunk parameter set, fixed once at trainer construction.

    For the frozen RoPE model this is ``embed.weight`` (tied head counted once),
    every ``blocks.*`` parameter and ``norm.weight``.  ``relation.*`` is excluded;
    any other name is an unexpected architecture and is refused rather than
    silently filtered.
    """

    names: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    dtype: str
    parameters: int
    digest: str

    @classmethod
    def of(cls, model: torch.nn.Module) -> TrunkIdentity:
        cfg = getattr(model, "cfg", None)
        if not isinstance(cfg, Config):
            raise RelationTrainingRefused("model lacks DrawingLM Config")
        d_model, d_ff = cfg.d_model, cfg.d_ff
        expected: list[tuple[str, tuple[int, ...]]] = [("embed.weight", (ByteCodec.vocab_size, d_model))]
        for i in range(cfg.n_layers):
            expected.append((f"blocks.{i}.norm_attn.weight", (d_model,)))
            expected.append((f"blocks.{i}.attn.qkv.weight", (3 * d_model, d_model)))
            expected.append((f"blocks.{i}.attn.proj.weight", (d_model, d_model)))
            expected.append((f"blocks.{i}.norm_ff.weight", (d_model,)))
            expected.append((f"blocks.{i}.ff.gate.weight", (d_ff, d_model)))
            expected.append((f"blocks.{i}.ff.up.weight", (d_ff, d_model)))
            expected.append((f"blocks.{i}.ff.down.weight", (d_model, d_ff)))
        expected.append(("norm.weight", (d_model,)))

        expected_names = [name for name, _ in expected]
        expected_shapes = dict(expected)

        head = getattr(model, "head", None)
        embed = getattr(model, "embed", None)
        if head is None or embed is None or head.weight is not embed.weight:
            raise RelationTrainingRefused("the R4 trunk expects a tied embedding/byte head")

        all_registered = list(model.named_parameters(remove_duplicate=False))
        unexpected_registered = [
            name for name, _ in all_registered
            if not name.startswith("relation.") and name != "head.weight" and name not in expected_shapes
        ]
        if unexpected_registered:
            raise RelationTrainingRefused(
                f"unexpected shared-trunk parameter {min(unexpected_registered)!r}; the R4 trunk identity is fixed")

        trunk_params = [
            (name, p) for name, p in all_registered
            if not name.startswith("relation.") and name != "head.weight"
        ]
        actual_names = [name for name, _ in trunk_params]
        if actual_names != expected_names:
            extra = set(actual_names) - set(expected_names)
            missing = set(expected_names) - set(actual_names)
            if extra:
                raise RelationTrainingRefused(
                    f"unexpected shared-trunk parameter {min(extra)!r}; the R4 trunk identity is fixed")
            if missing:
                raise RelationTrainingRefused(
                    f"missing shared-trunk parameter {min(missing)!r}; the R4 trunk identity is fixed")
            raise RelationTrainingRefused("shared-trunk parameters are out of canonical order")

        allocated_ranges: list[tuple[str, int, int, int]] = []
        relation_ranges: list[tuple[str, int, int, int]] = [
            (name, p.untyped_storage().data_ptr(), p.data_ptr(),
             p.data_ptr() + p.numel() * p.element_size())
            for name, p in all_registered if name.startswith("relation.")
        ]

        for name, parameter in trunk_params:
            if parameter.dtype != torch.float32:
                raise RelationTrainingRefused(
                    f"trunk parameter {name!r} has dtype {parameter.dtype}, expected torch.float32")
            if tuple(parameter.shape) != expected_shapes[name]:
                raise RelationTrainingRefused(
                    f"trunk parameter {name!r} has shape {tuple(parameter.shape)}, expected {expected_shapes[name]}")
            s_ptr = parameter.untyped_storage().data_ptr()
            start_byte = parameter.data_ptr()
            end_byte = start_byte + parameter.numel() * parameter.element_size()
            for other_name, other_s_ptr, other_start, other_end in allocated_ranges:
                if s_ptr == other_s_ptr or max(start_byte, other_start) < min(end_byte, other_end):
                    raise RelationTrainingRefused(f"trunk parameter {name!r} aliases another tensor")
            for r_name, r_s_ptr, r_start, r_end in relation_ranges:
                if s_ptr == r_s_ptr or max(start_byte, r_start) < min(end_byte, r_end):
                    raise RelationTrainingRefused(f"trunk parameter {name!r} aliases relation parameter {r_name!r}")
            allocated_ranges.append((name, s_ptr, start_byte, end_byte))

        shapes = tuple(shape for _, shape in expected)
        parameters = sum(math.prod(shape) for shape in shapes)
        dtype = "torch.float32"
        digest = hashlib.sha256(json.dumps(
            [expected_names, [list(shape) for shape in shapes], dtype, parameters],
            separators=(",", ":")).encode()).hexdigest()
        return cls(tuple(expected_names), shapes, dtype, parameters, digest)

    def parameters_of(self, model: torch.nn.Module) -> list[Tensor]:
        current = TrunkIdentity.of(model)
        if current != self:
            raise RelationTrainingRefused("model trunk identity disagrees with the recorded identity")
        named = dict(model.named_parameters())
        return [named[name] for name in self.names]

    def as_dict(self) -> dict[str, object]:
        return {"names": list(self.names), "shapes": [list(shape) for shape in self.shapes],
                "dtype": self.dtype, "parameters": self.parameters, "digest": self.digest}


@dataclass(frozen=True)
class RelationLoss:
    byte_sum: Tensor
    gate_sum: Tensor
    joint_sum: Tensor
    content_bytes: int
    reachable_boundaries: int
    positive_boundaries: int
    unclipped_grad_norm: Tensor | None = field(default=None, compare=False)
    clipped_grad_norm: Tensor | None = field(default=None, compare=False)
    factors: FactorDiagnostics | None = field(default=None, compare=False)
    components: ComponentGradients | None = field(default=None, compare=False)

    @property
    def byte(self) -> Tensor:
        return self.byte_sum / self.content_bytes

    @property
    def gate(self) -> Tensor:
        return self.gate_sum / self.reachable_boundaries

    @property
    def action_joint(self) -> Tensor:
        return self.joint_sum / self.positive_boundaries if self.positive_boundaries else self.joint_sum

    @property
    def total(self) -> Tensor:
        return self.byte + 0.1 * (self.gate + self.action_joint)

    @property
    def relation_bits(self) -> Tensor:
        return (self.gate + self.action_joint) / torch.log(self.total.new_tensor(2.0))


def relation_loss(logits: Tensor, plan: RelationBatchPlan, *,
                  scores: RelationScores | None = None,
                  equivalents: Sequence[Sequence[EquivalentAction]] = (),
                  gate_targets: Tensor | None = None) -> RelationLoss:
    """Compute raw sums and the R4 objective without ever combining denominators."""
    plan.validate()
    if logits.shape[:2] != plan.targets.shape or logits.ndim != 3:
        raise RelationTrainingRefused("byte logits do not align with the batch plan")
    byte_sum = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), plan.targets.to(logits.device).reshape(-1),
                               ignore_index=PAD, reduction="sum")
    zero = byte_sum.new_zeros((), dtype=torch.float32)
    if scores is None:
        return RelationLoss(byte_sum, zero, zero, plan.content_bytes,
                            plan.reachable_boundaries, plan.positive_boundaries)
    if gate_targets is None or gate_targets.shape != (plan.reachable_boundaries,):
        raise RelationTrainingRefused("relation scores need one gate target per reachable query")
    expected_gates = torch.tensor([query.gate_target for query in plan.queries],
                                  dtype=torch.long, device=gate_targets.device)
    if gate_targets.dtype != torch.long or not torch.equal(gate_targets, expected_gates):
        raise RelationTrainingRefused("gate targets disagree with the batch plan")
    if len(equivalents) != plan.positive_boundaries:
        raise RelationTrainingRefused("relation scores need one equivalence set per positive query")
    if scores.packed.queries != plan.reachable_boundaries:
        raise RelationTrainingRefused("packed scores disagree with the batch query count")
    spans = [span for query in plan.queries for span in query.spans]
    splits = [0]
    for query in plan.queries:
        splits.append(splits[-1] + len(query.spans))
    if scores.packed.row_splits.tolist() != splits or \
            scores.packed.source_start.tolist() != [start for start, _ in spans] or \
            scores.packed.source_stop.tolist() != [stop for _, stop in spans]:
        raise RelationTrainingRefused("packed score candidates disagree with the batch plan")
    positives = [index for index, query in enumerate(plan.queries) if query.gate_target == COPY]
    for query, alternatives in zip(positives, equivalents, strict=True):
        if not alternatives or any(not isinstance(action, EquivalentAction) or action.query != query
                                   or any(type(value) is not int for value in action.canonical_key())
                                   for action in alternatives):
            raise RelationTrainingRefused("joint targets disagree with positive query ownership")
        begin = int(scores.packed.row_splits[query])
        by_span = {span: begin + index for index, span in enumerate(plan.queries[query].spans)}
        expected = tuple(EquivalentAction(query, by_span[(item.source_start, item.source_stop)],
                                          item.d4, item.dx, item.dy, item.total_count)
                         for item in plan.queries[query].equivalents)
        if len(alternatives) != len(expected) or set(alternatives) != set(expected):
            raise RelationTrainingRefused("joint targets disagree with the complete batch equivalences")
    gate_sum = F.cross_entropy(scores.gate_logits.float(), gate_targets, reduction="sum")
    joint = joint_valid_nll(scores, equivalents)
    if joint.shape != (plan.positive_boundaries,):
        raise RelationTrainingRefused("joint NLL did not return one loss per positive boundary")
    return RelationLoss(byte_sum, gate_sum, joint.sum(), plan.content_bytes,
                        plan.reachable_boundaries, plan.positive_boundaries)


@dataclass(frozen=True)
class RelationTrainConfig:
    """R4-only training configuration; historical ``TrainConfig`` stays untouched."""

    arm: str
    model: dict[str, object]
    seed: int
    batch_size: int
    max_len: int
    #: Declared horizon and warmup of the shared historical cosine schedule.
    #: Required, so a tiny fixture states its budget; the 24,000-step scientific
    #: budget lives in the scientific contract and is not a default here.
    steps: int
    warmup: int
    lr: float = 3e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    #: Complete AdamW recipe is serialized: a changed optimizer group is a
    #: different continuation even if its tensor digest is self-consistent.
    adam_betas: tuple[float, float] = (0.9, 0.95)
    adam_eps: float = 1e-8
    adam_amsgrad: bool = False
    adam_foreach: bool | None = None
    adam_fused: bool | None = None
    adam_capturable: bool = False
    adam_differentiable: bool = False
    adam_maximize: bool = False
    attention_budget: int = 24_000_000
    device: str = "cpu"
    deterministic: bool = True
    artifact_output_policy: str = "test_only"
    corpus_provenance: str = "scientific"
    #: Provenance of `seed`: a named model seed from the frozen table, whose
    #: value must equal `seed_for(name)`, or `engineering` for a numeric fixture.
    seed_namespace: str = SEED_NAMESPACE_ENGINEERING
    #: Provenance label of the LR rule (`dm.train.lr_at`), not a menu.
    schedule: str = SCHEDULE_IDENTITY
    max_cached_cases: int = 256
    max_cached_spans: int = 65_536
    max_cached_bytes: int = 16 * 2**20
    #: Completed-update numbers at which component gradients are measured.
    #: `None` derives the fixed `schedule_boundaries_v1` cadence; an explicit
    #: tuple (possibly empty) is recorded as an engineering override.
    component_gradient_events: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.arm not in ("none", "span_affine_v1"):
            raise RelationTrainingRefused(f"unknown relation training arm {self.arm!r}")
        if self.device != "cpu":
            raise RelationTrainingRefused("R4 training is only qualified for CPU")
        if self.deterministic is not True:
            raise RelationTrainingRefused("R4 requires the qualified deterministic policy")
        if any(type(value) is not int or isinstance(value, bool) for value in
               (self.seed, self.batch_size, self.max_len, self.attention_budget, self.steps,
                self.warmup, self.max_cached_cases, self.max_cached_spans, self.max_cached_bytes)):
            raise RelationTrainingRefused("R4 seed, schedule, cache and batch fields must be integers")
        if not (0 <= self.seed < 2**63 - 1):
            raise RelationTrainingRefused("R4 seed is outside the generator domain")
        if not (1 <= self.steps < 2**63 - 1) or not (0 <= self.warmup < self.steps):
            raise RelationTrainingRefused("R4 needs 1 <= steps < 2**63 - 1 and 0 <= warmup < steps; no clamping")
        if self.schedule != SCHEDULE_IDENTITY:
            raise RelationTrainingRefused(f"R4 schedule identity must be {SCHEDULE_IDENTITY!r}")
        if any(not _finite_nonnegative(value) for value in
               (self.lr, self.weight_decay, self.grad_clip, self.adam_eps)) or \
                self.lr == 0 or self.grad_clip == 0 or self.adam_eps == 0:
            raise RelationTrainingRefused("R4 optimizer values must be finite and in their domains")
        if type(self.adam_betas) is not tuple or len(self.adam_betas) != 2 or \
                any(type(value) not in (int, float) or not math.isfinite(value) or
                    not 0 <= value < 1 for value in self.adam_betas) or \
                self.adam_betas != (0.9, 0.95):
            raise RelationTrainingRefused("R4 AdamW betas must be the frozen (0.9, 0.95) tuple")
        if self.adam_eps != 1e-8 or any(value is not False for value in (
                self.adam_amsgrad, self.adam_capturable, self.adam_differentiable,
                self.adam_maximize)) or self.adam_foreach is not None or self.adam_fused is not None:
            raise RelationTrainingRefused("R4 AdamW execution recipe is fixed")
        if self.artifact_output_policy != "test_only":
            raise RelationTrainingRefused("R4 permits temporary test bundles only")
        if not (1 <= self.batch_size < 2**63 - 1) or not (2 <= self.max_len < 2**63 - 1) or not (1 <= self.attention_budget < 2**63 - 1):
            raise RelationTrainingRefused("invalid relation trainer batch/max-length/budget shape")
        if any(not (0 <= value < 2**63 - 1) for value in (self.max_cached_cases, self.max_cached_spans, self.max_cached_bytes)):
            raise RelationTrainingRefused("metadata cache limits must be in [0, 2**63 - 1)")
        try:
            validate_provenance_and_seed(self.corpus_provenance, self.seed_namespace, self.seed)
        except RelationTrainingEvidenceRefused as exc:
            raise RelationTrainingRefused(str(exc)) from exc
        if self.component_gradient_events is not None:
            events = self.component_gradient_events
            if type(events) is not tuple or any(type(step) is not int for step in events) or \
                    list(events) != sorted(set(events)) or \
                    any(not 1 <= step <= self.steps for step in events):
                raise RelationTrainingRefused(
                    "component gradient events must be a sorted tuple of unique steps in [1, steps]")
        if self.model.get("relation_schema") != self.arm:
            raise RelationTrainingRefused("trainer arm and model relation_schema disagree")
        if self.model.get("feedback_schema", "none") != "none":
            raise RelationTrainingRefused("relation training may not enable latent feedback")
        try:
            model = Config(**self.model)
        except (TypeError, ValueError) as exc:
            raise RelationTrainingRefused(f"relation model config is invalid: {exc}") from exc
        if model.vocab_size != ByteCodec.vocab_size or not model.causal or model.dropout != 0.0:
            raise RelationTrainingRefused(
                "R4 freezes a causal, dropout-free ByteCodec model for exact accumulation")
        if model.max_len != self.max_len:
            raise RelationTrainingRefused("trainer max_len and model max_len must agree exactly")

    @property
    def component_policy(self) -> str:
        return (DIAGNOSTIC_POLICY_SCHEDULE if self.component_gradient_events is None
                else DIAGNOSTIC_POLICY_EXPLICIT)

    @property
    def component_events(self) -> tuple[int, ...]:
        """The complete, precomputed event set; never selected during training."""
        if self.component_gradient_events is None:
            return default_component_events(self.steps, self.warmup)
        return self.component_gradient_events


@dataclass
class RelationAccounting:
    """Raw R4 counters.  Means are deliberately reconstructed by consumers."""

    optimizer_steps: int = 0
    programs: int = 0
    semantic_bytes: int = 0
    content_symbols: int = 0
    padded_positions: int = 0
    byte_sum: float = 0.0
    gate_sum: float = 0.0
    joint_sum: float = 0.0
    byte_denominator: int = 0
    gate_denominator: int = 0
    positive_denominator: int = 0
    reachable_queries: int = 0
    candidate_spans: int = 0
    unclipped_grad_norm: float = 0.0
    clipped_grad_norm: float = 0.0

    def observe(self, plan: RelationBatchPlan, loss: RelationLoss, *,
                unclipped_norm: Tensor, clipped_norm: Tensor) -> None:
        self.optimizer_steps += 1
        self.programs += plan.inputs.shape[0]
        self.semantic_bytes += plan.content_bytes
        self.content_symbols += plan.content_bytes
        self.padded_positions += plan.inputs.numel()
        self.byte_sum += float(loss.byte_sum.detach())
        self.gate_sum += float(loss.gate_sum.detach())
        self.joint_sum += float(loss.joint_sum.detach())
        self.byte_denominator += loss.content_bytes
        self.gate_denominator += loss.reachable_boundaries
        self.positive_denominator += loss.positive_boundaries
        self.reachable_queries += loss.reachable_boundaries
        self.candidate_spans += sum(len(query.spans) for query in plan.queries)
        self.unclipped_grad_norm += float(unclipped_norm.detach())
        self.clipped_grad_norm += float(clipped_norm.detach())


def _grad_norm(model: torch.nn.Module) -> Tensor:
    squared = [parameter.grad.detach().float().square().sum()
               for parameter in model.parameters() if parameter.grad is not None]
    return torch.stack(squared).sum().sqrt() if squared else torch.zeros(())


def _slice_plan(plan: RelationBatchPlan, start: int, stop: int) -> RelationBatchPlan:
    """Rebase immutable row metadata for one contiguous microbatch."""
    if not 0 <= start < stop <= plan.inputs.shape[0]:
        raise RelationTrainingRefused("microbatch range is outside its parent plan")
    queries = tuple(
        QueryPlan(query.row - start, query.case_index, query.boundary, query.gate_target,
                  query.spans, query.equivalents)
        for query in plan.queries if start <= query.row < stop
    )
    result = RelationBatchPlan(plan.case_indices[start:stop].clone(),
                               plan.inputs[start:stop], plan.targets[start:stop], queries)
    result.validate()
    return result


def _microbatch_rows(plan: RelationBatchPlan, attention_budget: int | None) -> int:
    if attention_budget is None:
        return plan.inputs.shape[0]
    if attention_budget < 1:
        raise RelationTrainingRefused("attention budget must be positive")
    return max(1, min(plan.inputs.shape[0], attention_budget // max(1, plan.inputs.shape[1] ** 2)))


class _ComponentAccumulator:
    """Per-parameter float32 gradient sums for one event; freed after the norm.

    The diagnostic is ``||sum_m g_m||`` over microbatches ``m``: gradient
    *vectors* are accumulated and the norm is taken once, in float64, in the
    declared parameter order.  A sum of per-chunk norms is a different number
    and is deliberately not computable from this object.
    """

    def __init__(self, parameters: Sequence[Tensor]) -> None:
        self.parameters = parameters
        self.buffers: dict[str, list[Tensor | None]] = {}
        self.extra_backward_calls = 0

    def observe(self, name: str, term: Tensor) -> None:
        grads = torch.autograd.grad(term, self.parameters, retain_graph=True,
                                    create_graph=False, allow_unused=True)
        self.extra_backward_calls += 1
        buffers = self.buffers.setdefault(name, [None] * len(self.parameters))
        for index, grad in enumerate(grads):
            if grad is None:
                continue
            value = grad.detach().to(torch.float32)
            if buffers[index] is None:
                buffers[index] = value.clone()
            else:
                buffers[index].add_(value)

    def norm(self, name: str) -> float:
        total = torch.zeros((), dtype=torch.float64)
        for buffer in self.buffers[name]:
            if buffer is not None:
                total += buffer.to(torch.float64).square().sum().cpu()
        return float(total.sqrt())

    @property
    def buffer_bytes(self) -> int:
        return sum(buffer.numel() * buffer.element_size()
                   for buffers in self.buffers.values() for buffer in buffers
                   if buffer is not None)


def relation_train_step(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                        plan: RelationBatchPlan, *, grad_clip: float,
                        attention_budget: int | None = None,
                        trunk: TrunkIdentity | None = None) -> RelationLoss:
    """One fixed-denominator, attention-budgeted R4 update.

    Each chunk contributes its raw sums divided by the *parent* plan's three
    denominators before backward, so splitting changes peak memory only.

    Factor marginals are always reported, detached, from the same scores the
    joint loss used.  When ``trunk`` is given this is a declared diagnostic
    event: each present normalized component is differentiated against the
    trunk with ``autograd.grad(retain_graph=True)`` before the chunk's ordinary
    combined backward.  Those calls populate no ``.grad``, touch no optimizer
    state and consume no RNG; the ordinary backward alone drives the update.
    """
    device = next(model.parameters()).device
    optimizer.zero_grad(set_to_none=True)
    byte_sum = torch.zeros((), device=device)
    gate_sum = torch.zeros((), device=device, dtype=torch.float32)
    joint_sum = torch.zeros((), device=device, dtype=torch.float32)
    has_head = getattr(model, "relation", None) is not None
    factor_sums = {name: torch.zeros((), dtype=torch.float32) for name in FACTOR_NAMES}
    accumulator = None if trunk is None else _ComponentAccumulator(trunk.parameters_of(model))
    rows = _microbatch_rows(plan, attention_budget)
    for start in range(0, plan.inputs.shape[0], rows):
        chunk = _slice_plan(plan, start, min(start + rows, plan.inputs.shape[0]))
        logits, states = model(chunk.inputs.to(device), return_state=True)
        if not has_head:
            loss = relation_loss(logits, chunk)
        else:
            packed, equivalents, gates = pack_candidates(chunk, states)
            scores = model.score_relation(packed)
            loss = relation_loss(logits, chunk, scores=scores, equivalents=equivalents,
                                 gate_targets=gates)
            if equivalents:
                for name, column in factor_marginal_nll(scores, equivalents).items():
                    factor_sums[name] += column.sum().cpu()
        terms = {"byte": loss.byte_sum / plan.content_bytes}
        if has_head:
            terms["gate"] = loss.gate_sum / plan.reachable_boundaries
            if chunk.positive_boundaries:
                terms["action_joint"] = loss.joint_sum / plan.positive_boundaries
        if accumulator is not None:
            for name, term in terms.items():
                accumulator.observe(name, term)
        objective = terms["byte"] + 0.1 * (
            terms.get("gate", loss.gate_sum) +
            terms.get("action_joint", loss.joint_sum / plan.positive_boundaries
                      if plan.positive_boundaries else loss.joint_sum)
        )
        if not torch.isfinite(objective):
            raise RelationTrainingRefused("nonfinite training objective")
        objective.backward()
        byte_sum += loss.byte_sum.detach()
        gate_sum += loss.gate_sum.detach()
        joint_sum += loss.joint_sum.detach()
    unclipped = _grad_norm(model)
    if not torch.isfinite(unclipped):
        raise RelationTrainingRefused("nonfinite training gradients")
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    clipped = _grad_norm(model)
    optimizer.step()
    _validate_updated_numerics(model, optimizer)
    if not has_head:
        factors = FactorDiagnostics.not_applicable()
    elif plan.positive_boundaries:
        factors = FactorDiagnostics(FACTOR_STATUS_MEASURED, plan.positive_boundaries,
                                    {name: float(factor_sums[name]) for name in FACTOR_NAMES})
    else:
        factors = FactorDiagnostics(FACTOR_STATUS_NO_POSITIVES, 0,
                                    {name: 0.0 for name in FACTOR_NAMES})
    components = None
    if accumulator is not None:
        measured: dict[str, ComponentGradient] = {}
        for name in COMPONENT_NAMES:
            weight = COMPONENT_WEIGHTS[name]
            if name == "byte" or (has_head and (name == "gate" or plan.positive_boundaries)):
                denominator = {"byte": plan.content_bytes, "gate": plan.reachable_boundaries,
                               "action_joint": plan.positive_boundaries}[name]
                measured[name] = ComponentGradient(FACTOR_STATUS_MEASURED,
                                                   accumulator.norm(name), denominator, weight)
            elif has_head:
                measured[name] = ComponentGradient(FACTOR_STATUS_NO_POSITIVES, None, 0, weight)
            else:
                measured[name] = ComponentGradient(FACTOR_STATUS_NOT_APPLICABLE, None, None, weight)
        components = ComponentGradients(measured, accumulator.extra_backward_calls,
                                        accumulator.buffer_bytes)
        accumulator.buffers.clear()
    return RelationLoss(byte_sum, gate_sum, joint_sum, plan.content_bytes,
                        plan.reachable_boundaries, plan.positive_boundaries,
                        unclipped.detach(), clipped.detach(), factors, components)


def _validate_updated_numerics(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    """One post-update scan; a possibly committed bad update is never retried."""
    for tensor in (*model.state_dict().values(),
                   *(value for state in optimizer.state.values() for value in state.values()
                     if isinstance(value, Tensor))):
        if not torch.isfinite(tensor).all():
            raise RelationTrainingRefused("nonfinite updated model/optimizer state")


@dataclass
class RelationBucketCursor:
    """Resumable cursor over the historical `BucketedBatchSampler` epoch order.

    Epoch ``e`` (zero-based) is generated by `dm.data.dataset.bucket_epoch_plan`
    under ``(base_seed, e)``, exactly as the historical sampler seeded with the
    model seed.  ``epoch`` counts epochs generated so far (one-based, ``0`` =
    none yet), so state ``epoch=0, cursor=0`` is the canonical initial state and
    ``cursor == len(batches)`` is an exhausted epoch awaiting on-demand rollover.
    ``generator`` records the private stream: the initial base seed before any
    epoch, then the post-generation state of the latest epoch.  It is evidence,
    not a continuously advanced sequence.
    """

    lengths: tuple[int, ...]
    batch_size: int
    base_seed: int
    generator: torch.Generator
    epoch: int = 0
    cursor: int = 0
    batches: list[list[int]] = field(default_factory=list)
    pool_batches: int = 50
    algorithm: str = BUCKET_EPOCH_ALGORITHM

    @classmethod
    def seeded(cls, lengths: Sequence[int], batch_size: int, base_seed: int) -> RelationBucketCursor:
        if not lengths or type(batch_size) is not int or batch_size < 1:
            raise RelationTrainingRefused("bucket cursor needs lengths and a positive batch size")
        if type(base_seed) is not int or not 0 <= base_seed < 2**63 - 1:
            raise RelationTrainingRefused("bucket cursor base seed is outside the generator domain")
        return cls(tuple(lengths), batch_size, base_seed,
                   torch.Generator().manual_seed(base_seed))

    @property
    def batches_per_epoch(self) -> int:
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size

    @property
    def consumed_batches(self) -> int:
        return max(0, self.epoch - 1) * self.batches_per_epoch + self.cursor

    def next(self) -> list[int]:
        if self.epoch == 0 or self.cursor >= len(self.batches):
            self.batches, state = bucket_epoch_plan(
                self.lengths, self.batch_size, epoch=self.epoch, seed=self.base_seed,
                pool_batches=self.pool_batches)
            self.generator.set_state(state)
            self.cursor = 0
            self.epoch += 1
        result = list(self.batches[self.cursor])
        self.cursor += 1
        return result

    def state_dict(self) -> dict[str, object]:
        return {"algorithm": self.algorithm, "lengths": list(self.lengths),
                "batch_size": self.batch_size, "base_seed": self.base_seed,
                "epoch": self.epoch, "cursor": self.cursor, "batches": self.batches,
                "pool_batches": self.pool_batches, "generator": self.generator.get_state()}

    @classmethod
    def from_state(cls, state: dict[str, object], *, lengths: Sequence[int], batch_size: int,
                   base_seed: int, pool_batches: int = 50) -> RelationBucketCursor:
        expected = {"algorithm", "lengths", "batch_size", "base_seed", "epoch", "cursor",
                    "batches", "pool_batches", "generator"}
        if not isinstance(state, dict) or set(state) != expected or \
                state.get("lengths") != list(lengths) or state.get("batch_size") != batch_size or \
                state.get("pool_batches") != pool_batches or state.get("base_seed") != base_seed or \
                state.get("algorithm") != BUCKET_EPOCH_ALGORITHM:
            raise RelationTrainingRefused("bucket sampler disagrees with the frozen trainer/corpus shape")
        if not isinstance(state["lengths"], list) or \
                not all(type(item) is int and item > 0 for item in state["lengths"]):
            raise RelationTrainingRefused("bucket sampler state is malformed")
        if any(type(state[name]) is not int for name in
               ("batch_size", "base_seed", "epoch", "cursor", "pool_batches")) or \
                not isinstance(state["batches"], list) or state["epoch"] < 0 or \
                not 0 <= state["cursor"] <= len(state["batches"]):
            raise RelationTrainingRefused("bucket sampler cursor state is malformed")
        if not isinstance(state["generator"], torch.Tensor):
            raise RelationTrainingRefused("bucket sampler private RNG record is malformed")
        if state["epoch"] == 0:
            if state["batches"] or state["cursor"]:
                raise RelationTrainingRefused("an epoch-zero sampler cannot own batches or a cursor")
            expected_state = torch.Generator().manual_seed(base_seed).get_state()
        else:
            expected_batches, expected_state = bucket_epoch_plan(
                tuple(lengths), batch_size, epoch=state["epoch"] - 1, seed=base_seed,
                pool_batches=pool_batches)
            if state["batches"] != expected_batches:
                raise RelationTrainingRefused(
                    "bucket sampler batches are not the historical epoch order for this seed")
        if not torch.equal(state["generator"], expected_state):
            raise RelationTrainingRefused("bucket sampler private RNG record disagrees with its epoch")
        generator = torch.Generator()
        generator.set_state(state["generator"])
        return cls(tuple(state["lengths"]), state["batch_size"], state["base_seed"], generator,
                   state["epoch"], state["cursor"], [list(batch) for batch in state["batches"]],
                   state["pool_batches"])


class RelationTrainer:
    """Small R4-only paired trainer; no evaluation or persistent artifact route.

    ``completed_step == k`` means exactly ``k`` successful optimizer updates.
    Update ``k`` (for ``0 <= k < steps``) uses ``lr_at(k, config)``; at
    ``k == steps`` a further update is refused before any state is touched.
    """

    def __init__(self, config: RelationTrainConfig, training: TrainingCorpus, *,
                 resource_observer: ResourceObserver | None = None,
                 resource_sample_interval: float | None = None) -> None:
        if config.corpus_provenance != training.provenance:
            raise RelationTrainingRefused("trainer/corpus provenance disagrees")
        torch.use_deterministic_algorithms(config.deterministic)
        torch.manual_seed(config.seed)
        self.config, self.training = config, training
        self._bound_training = training
        self._training_identity = (training.program_fingerprint, training.supervision_identity)
        # Capture process identity before model/trainer work. The writer must
        # later prove disk sources and active numerical flags still agree with
        # this executing process, not merely with save-time configuration.
        self._process_environment = _environment(config)
        self._process_source_hashes = source_hashes()
        # Process-local telemetry is deliberately absent from config/bundles.
        self.resource_observer = resource_observer
        # Refuse a sampling period nobody collects rather than sampling silently.
        validate_sample_interval(resource_sample_interval)
        if resource_observer is None and resource_sample_interval is not None:
            raise RelationTrainingRefused("resource sampling requires a collector")
        self.resource_sample_interval = resource_sample_interval
        self.model = DrawingLM(Config(**config.model)).to(config.device)
        self.trunk = TrunkIdentity.of(self.model)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr_at(0, config),
            weight_decay=config.weight_decay, betas=config.adam_betas,
            eps=config.adam_eps, amsgrad=config.adam_amsgrad,
            foreach=config.adam_foreach, fused=config.adam_fused,
            capturable=config.adam_capturable,
            differentiable=config.adam_differentiable,
            maximize=config.adam_maximize,
        )
        # Batch ordering shares the *model* seed with the historical sampler;
        # the DataLoader's +2_000_003 offset was iterator bookkeeping, not order.
        self.cursor = RelationBucketCursor.seeded(
            [len(case.flat) + 1 for case in training.cases], config.batch_size, config.seed)
        self.accounting = RelationAccounting()
        self.history: list[dict[str, object]] = []
        self.component_records: list[dict[str, object]] = []
        self.planner = RelationBatchPlanner(
            training, max_len=config.max_len, max_cached_cases=config.max_cached_cases,
            max_cached_spans=config.max_cached_spans, max_cached_bytes=config.max_cached_bytes)
        self._failed: bool = False

    @property
    def completed_step(self) -> int:
        return self.accounting.optimizer_steps

    def step(self) -> RelationLoss:
        if self._failed:
            raise RelationTrainingRefused("trainer failed an earlier update and is unusable")
        completed = self.completed_step
        if completed >= self.config.steps:
            raise RelationTrainingComplete(
                f"training completed {completed} of {self.config.steps} declared updates")
        try:
            self._assert_process_binding(check_sources=False)
            with self._phase("step"):
                loss = self._update(completed)
            return loss
        except BaseException:
            self._failed = True
            raise

    def _phase(self, name: str):
        return resource_phase(self.resource_observer, name,
                              sample_interval_seconds=self.resource_sample_interval)

    def _update(self, completed: int) -> RelationLoss:
        """The measured body of one update; refusals above it are not work."""
        with self._phase("metadata"):
            indices = self.cursor.next()
            plan = self.planner.batch(indices)
        lr = lr_at(completed, self.config)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        event = (completed + 1) in self.config.component_events
        self.model.train()
        with self._phase("train_update"):
            loss = relation_train_step(self.model, self.optimizer, plan,
                                       grad_clip=self.config.grad_clip,
                                       attention_budget=self.config.attention_budget,
                                       trunk=self.trunk if event else None)
        if loss.unclipped_grad_norm is None or loss.clipped_grad_norm is None or \
                loss.factors is None or (event and loss.components is None):
            raise RuntimeError("relation train step did not record its diagnostics")
        self.accounting.observe(plan, loss, unclipped_norm=loss.unclipped_grad_norm,
                                clipped_norm=loss.clipped_grad_norm)
        self.history.append({
            "step": self.accounting.optimizer_steps,
            "batch_ids": indices,
            "lr": lr,
            "byte_sum": float(loss.byte_sum.detach()),
            "gate_sum": float(loss.gate_sum.detach()),
            "joint_sum": float(loss.joint_sum.detach()),
            "content_bytes": loss.content_bytes,
            "reachable_boundaries": loss.reachable_boundaries,
            "positive_boundaries": loss.positive_boundaries,
            "factors": loss.factors.as_dict(),
        })
        if event:
            assert loss.components is not None
            self.component_records.append({
                "step": self.accounting.optimizer_steps, "batch_ids": indices,
                "trunk_digest": self.trunk.digest, **loss.components.as_dict(),
            })
        return loss

    def diagnostics(self) -> dict[str, object]:
        return {"component_policy": self.config.component_policy,
                "component_events": list(self.config.component_events),
                "trunk": self.trunk.as_dict(),
                "component_records": self.component_records}

    def save(self, path: Path, *, sources: Sequence[Path]) -> dict[str, object]:
        if self._failed:
            raise RelationTrainingRefused("cannot save from a failed trainer")
        self._assert_process_binding(check_sources=True)
        return save_training_bundle(
            path, config=self.config, completed_step=self.completed_step,
            model=self.model, optimizer=self.optimizer, sampler=self.cursor.state_dict(),
            accounting=self.accounting, history=self.history, training=self.training,
            data_generator=self.cursor.generator, sources=sources,
            diagnostics=self.diagnostics(),
            process_environment=self._process_environment,
            process_source_hashes=self._process_source_hashes,
        )

    def _assert_process_binding(self, *, check_sources: bool) -> None:
        if self.training is not self._bound_training or self.planner.training is not self._bound_training:
            raise RelationTrainingRefused("trainer/planner corpus binding changed")
        if check_sources and self._training_identity != (
                self.training.program_fingerprint, self.training.supervision_identity):
            raise RelationTrainingRefused("trainer corpus binding changed since construction")
        if _environment(self.config) != self._process_environment:
            raise RelationTrainingRefused(
                "active runtime differs from the process identity captured before training work")
        if check_sources and source_hashes() != self._process_source_hashes:
            raise RelationTrainingRefused(
                "authoritative source contents changed since this trainer captured its process identity")

    @classmethod
    def load(cls, path: Path, *, training: TrainingCorpus,
             sources: Sequence[Path]) -> RelationTrainer:
        caller_rng = capture_rng_state()
        caller_policy = (torch.are_deterministic_algorithms_enabled(),
                         torch.is_deterministic_algorithms_warn_only_enabled())
        try:
            raw = _read_training_bundle(path)
            trainer = cls(RelationTrainConfig(**raw["config"]), training)
            payload = _restore_training_bundle(raw, model=trainer.model, optimizer=trainer.optimizer,
                                               data_generator=trainer.cursor.generator,
                                               training=training, sources=sources,
                                               trunk=trainer.trunk)
            trainer.cursor = RelationBucketCursor.from_state(
                payload["sampler"], lengths=[len(case.flat) + 1 for case in training.cases],
                batch_size=trainer.config.batch_size, base_seed=trainer.config.seed,
            )
            trainer.accounting = RelationAccounting(**payload["accounting"])
            trainer.history = list(payload["history"])
            trainer.component_records = list(payload["diagnostics"]["component_records"])
            if trainer.accounting.optimizer_steps != payload["completed_step"] or \
                    trainer.cursor.consumed_batches != payload["completed_step"]:
                raise RelationTrainingRefused("resumed accounting, cursor and completed step disagree")
            return trainer
        except BaseException:
            try:
                restore_rng_state(caller_rng)
                torch.use_deterministic_algorithms(caller_policy[0], warn_only=caller_policy[1])
            except BaseException as rollback_error:
                raise RuntimeError("R4 failed to rollback caller runtime state") from rollback_error
            raise


def capture_rng_state(data_generator: torch.Generator | None = None) -> dict[str, object]:
    """All R4 continuation streams, in serialisable non-digest form."""
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
        "data_generator": None if data_generator is None else data_generator.get_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, object], data_generator: torch.Generator | None = None) -> None:
    """Restore exactly the streams captured by :func:`capture_rng_state`."""
    required = {"python", "numpy", "torch_cpu", "data_generator"}
    if set(state) - {"python", "numpy", "torch_cpu", "data_generator", "torch_cuda"} or \
            required - set(state):
        raise RelationTrainingRefused("checkpoint RNG state has an unexpected key set")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch_cpu"])
    if data_generator is None and state["data_generator"] is not None:
        raise RelationTrainingRefused("checkpoint has a private data RNG but no generator was supplied")
    if data_generator is not None and state["data_generator"] is None:
        raise RelationTrainingRefused("checkpoint omitted its private data RNG")
    if data_generator is not None:
        data_generator.set_state(state["data_generator"])
    if "torch_cuda" in state:
        if not torch.cuda.is_available():
            raise RelationTrainingRefused("checkpoint requires CUDA RNG state on a non-CUDA host")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


_BUNDLE_KEYS = REQUIRED_BUNDLE_KEYS

_ROOT = Path(__file__).resolve().parent
AUTHORITATIVE_R4_SOURCES: tuple[Path, ...] = (
    _ROOT / "train_relation.py",
    _ROOT / "train.py",
    _ROOT / "data" / "dataset.py",
    _ROOT / "models" / "transformer.py",
    _ROOT / "models" / "relation.py",
    _ROOT / "relation" / "candidates.py",
    _ROOT / "relation" / "decoding.py",
    _ROOT / "relation" / "resources.py",
    _ROOT / "relation" / "spec.py",
    _ROOT / "relation" / "transducer.py",
    _ROOT / "relation" / "queue.py",
    _ROOT / "data" / "relation.py",
    _ROOT / "data" / "augment.py",
    _ROOT / "data" / "composed.py",
    _ROOT / "data" / "quickdraw.py",
    _ROOT / "data" / "synthetic.py",
    _ROOT / "data" / "fingerprint.py",
    _ROOT / "data" / "feedback_audit.py",
    _ROOT / "isa" / "codec.py",
    _ROOT / "isa" / "spec.py",
    _ROOT / "isa" / "asm.py",
    _ROOT / "isa" / "transform.py",
    _ROOT / "isa" / "unroll.py",
    _ROOT / "isa" / "relative.py",
    _ROOT / "vm" / "interp.py",
    _ROOT / "vm" / "render.py",
    _ROOT / "eval" / "relation_contract.py",
    _ROOT / "eval" / "relation_evidence.py",
    _ROOT / "eval" / "reports.py",
    _ROOT / "eval" / "relation_training_evidence.py",
)


def source_hashes(paths: Sequence[Path] = AUTHORITATIVE_R4_SOURCES) -> dict[str, str]:
    """One authoritative source list, hashed as contents rather than filenames."""
    out: dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            raise RelationTrainingRefused(f"R4 source file is absent: {path}")
        out[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def _source_digest(hashes: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


_RUNTIME_ENVIRONMENT_KEYS = (
    "PYTHONHASHSEED", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "CUBLAS_WORKSPACE_CONFIG", "CUDA_VISIBLE_DEVICES",
)


def _environment(config: RelationTrainConfig) -> dict[str, object]:
    """Requested policy and observed process state captured before R4 work."""
    warn_only = getattr(torch, "is_deterministic_algorithms_warn_only_enabled", lambda: False)
    return {
        "identity": "r4_process_runtime_v1",
        "requested": {"deterministic_algorithms": config.deterministic},
        "observed": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": str(torch.__version__),
            "torch_git_version": torch.version.git_version,
            "platform": platform.platform(),
            "device": config.device,
            "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "deterministic_algorithms_warn_only": bool(warn_only()),
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "torch_default_dtype": str(torch.get_default_dtype()),
            "mkldnn_enabled": bool(torch.backends.mkldnn.enabled),
            "environment": {name: os.environ.get(name) for name in _RUNTIME_ENVIRONMENT_KEYS},
        },
    }


_ADAMW_GROUP_KEYS = frozenset({
    "params", "lr", "betas", "eps", "weight_decay", "amsgrad", "maximize", "foreach",
    "capturable", "differentiable", "fused", "decoupled_weight_decay",
})


def _expected_adamw_group(config: RelationTrainConfig, completed_step: int) -> dict[str, object]:
    return {
        "lr": lr_at(max(0, completed_step - 1), config),
        "betas": config.adam_betas,
        "eps": config.adam_eps,
        "weight_decay": config.weight_decay,
        "amsgrad": config.adam_amsgrad,
        "maximize": config.adam_maximize,
        "foreach": config.adam_foreach,
        "capturable": config.adam_capturable,
        "differentiable": config.adam_differentiable,
        "fused": config.adam_fused,
        "decoupled_weight_decay": True,
    }


def _finite_tensor(value: object, *, name: str) -> Tensor:
    if not isinstance(value, Tensor) or value.layout is not torch.strided:
        raise RelationTrainingRefused(f"{name} must be a strided tensor")
    if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all()):
        raise RelationTrainingRefused(f"{name} contains NaN or infinity")
    return value


def _validate_model_state_dict(state: object, expected: Mapping[str, Tensor]) -> None:
    if not isinstance(state, Mapping) or set(state) != set(expected):
        raise RelationTrainingRefused("checkpoint model state keys disagree with the frozen model")
    for name, reference in expected.items():
        value = _finite_tensor(state[name], name=f"checkpoint model tensor {name!r}")
        if value.device.type != "cpu" or value.dtype != reference.dtype or \
                tuple(value.shape) != tuple(reference.shape):
            raise RelationTrainingRefused(
                f"checkpoint model tensor {name!r} shape/dtype/device disagrees with the fresh model")


def _validate_optimizer_state_dict(state: object, parameters: Sequence[Tensor],
                                   config: RelationTrainConfig, completed_step: int,
                                   parameter_steps: Sequence[int]) -> None:
    if not isinstance(state, Mapping) or set(state) != {"state", "param_groups"} or \
            not isinstance(state["state"], Mapping) or not isinstance(state["param_groups"], list) or \
            len(state["param_groups"]) != 1:
        raise RelationTrainingRefused("checkpoint optimizer state has an unexpected structure")
    group = state["param_groups"][0]
    if not isinstance(group, Mapping) or set(group) != _ADAMW_GROUP_KEYS or \
            not isinstance(group["params"], list) or group["params"] != list(range(len(parameters))):
        raise RelationTrainingRefused(
            "checkpoint optimizer group membership/order disagrees with the frozen model")
    expected = _expected_adamw_group(config, completed_step)
    if any(group[name] != value for name, value in expected.items()):
        raise RelationTrainingRefused("checkpoint optimizer recipe disagrees with the configuration")
    optimizer_state = state["state"]
    expected_ids = {index for index, steps in enumerate(parameter_steps) if steps}
    if set(optimizer_state) != expected_ids:
        raise RelationTrainingRefused(
            "checkpoint optimizer moment membership disagrees with the completed step")
    for index, parameter in enumerate(parameters):
        if index not in optimizer_state:
            continue
        item = optimizer_state[index]
        if not isinstance(item, Mapping) or set(item) != {"step", "exp_avg", "exp_avg_sq"}:
            raise RelationTrainingRefused(
                f"checkpoint optimizer state for parameter {index} has an unexpected key set")
        step = _finite_tensor(item["step"], name=f"optimizer step {index}")
        if step.ndim != 0 or step.dtype != torch.float32 or step.device.type != "cpu" or \
                float(step) != parameter_steps[index]:
            raise RelationTrainingRefused(
                f"checkpoint optimizer step for parameter {index} disagrees with completion")
        for name in ("exp_avg", "exp_avg_sq"):
            value = _finite_tensor(item[name], name=f"optimizer {name} {index}")
            if name == "exp_avg_sq" and bool((value < 0).any()):
                raise RelationTrainingRefused("checkpoint optimizer second moment must be nonnegative")
            if value.device.type != "cpu" or value.dtype != parameter.dtype or \
                    tuple(value.shape) != tuple(parameter.shape):
                raise RelationTrainingRefused(
                    f"checkpoint optimizer {name} {index} shape/dtype/device disagrees with its parameter")


def _validate_runtime_state(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                            config: RelationTrainConfig, completed_step: int,
                            history: Sequence[Mapping[str, object]]) -> None:
    """Validate finite model state and full AdamW recipe before publication."""
    _validate_optimizer_ownership(model, optimizer)
    _validate_frozen_model(model, config)
    _validate_optimizer_state_dict(optimizer.state_dict(), list(model.parameters()), config,
                                   completed_step, _parameter_update_steps(model, history))


def _validate_frozen_model(model: torch.nn.Module, config: RelationTrainConfig) -> None:
    with torch.device("meta"):
        reference = DrawingLM(Config(**config.model))
    if model.cfg != reference.cfg or tuple(model.state_dict()) != tuple(reference.state_dict()) or \
            tuple(name for name, _ in model.named_parameters()) != tuple(
                name for name, _ in reference.named_parameters()):
        raise RelationTrainingRefused("model registration disagrees with frozen architecture")
    _validate_model_state_dict(model.state_dict(), reference.state_dict())


def _parameter_update_steps(model: torch.nn.Module,
                            history: Sequence[Mapping[str, object]]) -> list[int]:
    # Only norm/gate participate when the batch has no positive action targets.
    positive_steps = sum(record["positive_boundaries"] > 0 for record in history)
    return [positive_steps if name.startswith("relation.") and name not in
            ("relation.norm.weight", "relation.gate.weight") else len(history)
            for name, _ in model.named_parameters()]


def _validate_optimizer_ownership(model: torch.nn.Module,
                                  optimizer: torch.optim.Optimizer) -> None:
    if type(optimizer) is not torch.optim.AdamW:
        raise RelationTrainingRefused("R4 continuation requires the fixed AdamW optimizer")
    parameters = list(model.parameters())
    if len(optimizer.param_groups) != 1 or \
            len(optimizer.param_groups[0]["params"]) != len(parameters) or \
            any(left is not right for left, right in
                zip(optimizer.param_groups[0]["params"], parameters, strict=True)):
        raise RelationTrainingRefused(
            "R4 optimizer group membership/order disagrees with the model parameter order")


def _audit_batch_row_facts(cases: Sequence[corpus.RelationCase], batch_ids: Sequence[int],
                           attention_budget: int, arm: str,
                           trunk_parameters: int) -> tuple[int, int, int, int, int]:
    """Derive true content, reachable, positive counts, exact extra backward calls, and buffer bound."""
    row_contents: list[int] = []
    row_reachables: list[int] = []
    row_positives: list[int] = []
    row_widths: list[int] = []

    for idx in batch_ids:
        if type(idx) is not int or isinstance(idx, bool) or not (0 <= idx < len(cases)):
            raise RelationTrainingRefused(f"batch index {idx!r} is outside the verified training corpus")
        case = cases[idx]
        flat_len = len(case.flat)
        row_widths.append(flat_len)
        row_contents.append(flat_len)
        actions = {action.boundary: action for action in case.actions}
        covered = corpus.covered_boundaries(case.actions)
        reachable = 0
        positive = 0
        for boundary in corpus.boundaries(case.flat):
            if boundary >= flat_len or boundary in covered:
                continue
            reachable += 1
            if boundary in actions:
                positive += 1
        row_reachables.append(reachable)
        row_positives.append(positive)

    content_bytes = sum(row_contents)
    reachable_boundaries = sum(row_reachables)
    positive_boundaries = sum(row_positives)

    t_width = max(row_widths)
    rows_per_chunk = max(1, min(len(batch_ids), attention_budget // max(1, t_width ** 2)))
    m_chunks = (len(batch_ids) + rows_per_chunk - 1) // rows_per_chunk

    if arm == "none":
        extra_backwards = m_chunks
        k_factor = 1
    else:
        positive_chunks = 0
        for m in range(m_chunks):
            start = m * rows_per_chunk
            stop = min(start + rows_per_chunk, len(batch_ids))
            if sum(row_positives[start:stop]) > 0:
                positive_chunks += 1
        extra_backwards = 2 * m_chunks + positive_chunks
        k_factor = 2 if positive_boundaries == 0 else 3

    max_buffer_bytes = 4 * trunk_parameters * k_factor
    return content_bytes, reachable_boundaries, positive_boundaries, extra_backwards, max_buffer_bytes


def _audit_history_and_diagnostics(
    history: Sequence[Mapping[str, object]],
    diagnostics: Mapping[str, object],
    training: TrainingCorpus,
    config: RelationTrainConfig,
    trunk_parameters: int,
) -> None:
    for expected_step, record in enumerate(history, start=1):
        c_bytes, r_bounds, p_bounds, extra_calls, max_buffer = _audit_batch_row_facts(
            training.cases, record["batch_ids"], config.attention_budget, config.arm, trunk_parameters)
        if record["content_bytes"] != c_bytes:
            raise RelationTrainingRefused(
                f"R4 history content_bytes {record['content_bytes']} disagrees with verified corpus {c_bytes}")
        if record["reachable_boundaries"] != r_bounds:
            raise RelationTrainingRefused(
                f"R4 history reachable_boundaries {record['reachable_boundaries']} disagrees with verified corpus {r_bounds}")
        if record["positive_boundaries"] != p_bounds:
            raise RelationTrainingRefused(
                f"R4 history positive_boundaries {record['positive_boundaries']} disagrees with verified corpus {p_bounds}")
        if expected_step in config.component_events:
            comp_rec = next(r for r in diagnostics["component_records"] if r["step"] == expected_step)
            if comp_rec["extra_backward_calls"] != extra_calls:
                raise RelationTrainingRefused(
                    f"R4 component record extra_backward_calls {comp_rec['extra_backward_calls']} disagrees with expected {extra_calls}")
            if comp_rec["buffer_bytes"] > max_buffer:
                raise RelationTrainingRefused(
                    f"R4 component record buffer_bytes {comp_rec['buffer_bytes']} exceeds bound {max_buffer}")


def save_training_bundle(path: Path, *, config: RelationTrainConfig, completed_step: int,
                         model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                         sampler: dict[str, object], accounting: RelationAccounting,
                         history: list[dict[str, object]], training: TrainingCorpus,
                         data_generator: torch.Generator | None,
                         sources: Sequence[Path],
                         diagnostics: dict[str, object],
                         process_environment: dict[str, object],
                         process_source_hashes: dict[str, str]) -> dict[str, object]:
    """Atomically replace a temporary R4 test bundle after complete validation.

    A successful publication intentionally replaces an existing destination.
    A validation, serialization or replacement failure removes only its staging
    file and preserves an existing destination.
    """
    if config.artifact_output_policy != "test_only":
        raise RelationTrainingRefused("R4 refuses persistent model-bearing artifacts")
    if completed_step != accounting.optimizer_steps:
        raise RelationTrainingRefused("bundle completed step and accounting step disagree")
    if tuple(sources) != AUTHORITATIVE_R4_SOURCES:
        raise RelationTrainingRefused("R4 source provenance is fixed by the authoritative source list")
    hashes = source_hashes()
    if hashes != process_source_hashes:
        raise RelationTrainingRefused(
            "authoritative source contents drifted from the captured process snapshot")
    environment = _environment(config)
    if environment != process_environment:
        raise RelationTrainingRefused(
            "active runtime drifted from the captured process snapshot before publication")
    _validate_runtime_state(model, optimizer, config, completed_step, history)
    payload: dict[str, object] = {
        "schema": RELATION_CHECKPOINT_SCHEMA,
        "namespace": "direction4-relation-r4-test-only",
        "config": asdict(config),
        "completed_step": completed_step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "sampler": sampler,
        "accounting": asdict(accounting),
        "history": history,
        "diagnostics": diagnostics,
        "corpus": {
            "manifest_path": None if training.manifest_path is None else str(training.manifest_path),
            "canonical_payload_sha256": training.canonical_payload_sha256,
            "file_sha256": training.file_sha256,
            "rebuilt_payload_sha256": training.rebuilt_payload_sha256,
            "training_program_fingerprint": training.program_fingerprint,
            "supervision_identity_version": SUPERVISION_IDENTITY_VERSION,
            "supervision_sha256": training.supervision_identity,
            "provenance": training.provenance,
        },
        "source_hashes": hashes,
        "source_digest": _source_digest(hashes),
        "environment": environment,
        "rng": capture_rng_state(data_generator),
        "content_digests": {"model": state_digest(model.state_dict()),
                            "optimizer": optimizer_digest(optimizer, model),
                            "rng": rng_digest(config.device, data_generator)},
    }
    try:
        validate_bundle_shape(payload)
    except RelationTrainingEvidenceRefused as exc:
        raise RelationTrainingRefused(str(exc)) from exc
    expected_lengths = [len(case.flat) + 1 for case in training.cases]
    if sampler["lengths"] != expected_lengths:
        raise RelationTrainingRefused("R4 sampler lengths disagree with verified training corpus")
    expected_trunk = TrunkIdentity.of(model)
    if diagnostics.get("trunk") != expected_trunk.as_dict():
        raise RelationTrainingRefused("R4 diagnostics trunk identity differs from the model")
    _audit_history_and_diagnostics(
        history, diagnostics, training, config, expected_trunk.parameters
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staged_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    staged = Path(staged_name)
    try:
        torch.save(payload, staged)
        os.replace(staged, path)
    except BaseException:
        try:
            staged.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return payload


def load_training_bundle(path: Path, *, model: torch.nn.Module,
                         optimizer: torch.optim.Optimizer,
                         data_generator: torch.Generator | None,
                         training: TrainingCorpus, sources: Sequence[Path],
                         trunk: TrunkIdentity | None = None) -> dict[str, object]:
    """Strict-load a bundle and prove all content identities before another batch."""
    return _restore_training_bundle(
        _read_training_bundle(path), model=model, optimizer=optimizer,
        data_generator=data_generator, training=training, sources=sources,
        trunk=TrunkIdentity.of(model) if trunk is None else trunk)


def _read_training_bundle(path: Path) -> dict[str, object]:
    """Read and validate once, before model construction can change global RNG."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError) as exc:
        raise RelationTrainingRefused(f"R4 bundle is unreadable: {exc}") from exc
    if isinstance(payload, dict) and payload.get("schema") in (1, 2):
        # Deliberate, non-migratable breaks: schema-1 predates the declared
        # schedule; schema-2 lacks immutable supervision, actual runtime and
        # full optimizer-recipe identities.
        raise RelationTrainingRefused(
            f"R4 bundle schema {payload['schema']} predates required corpus/process binding; "
            f"it cannot be resumed under schema {RELATION_CHECKPOINT_SCHEMA}")
    if not isinstance(payload, dict) or set(payload) != _BUNDLE_KEYS:
        raise RelationTrainingRefused("R4 bundle has an unexpected schema/key set")
    try:
        validate_bundle_shape(payload)
    except RelationTrainingEvidenceRefused as exc:
        raise RelationTrainingRefused(str(exc)) from exc
    return payload


def _restore_training_bundle(payload: dict[str, object], *, model: torch.nn.Module,
                             optimizer: torch.optim.Optimizer,
                             data_generator: torch.Generator | None,
                             training: TrainingCorpus, sources: Sequence[Path],
                             trunk: TrunkIdentity) -> dict[str, object]:
    if payload["schema"] != RELATION_CHECKPOINT_SCHEMA or \
            payload["namespace"] != "direction4-relation-r4-test-only":
        raise RelationTrainingRefused("R4 bundle schema/namespace is not accepted")
    if not isinstance(payload["config"], dict):
        raise RelationTrainingRefused("R4 bundle config is malformed")
    config = RelationTrainConfig(**payload["config"])
    if model.cfg != Config(**config.model):
        raise RelationTrainingRefused("fresh model configuration disagrees with bundle config")
    if payload["diagnostics"]["trunk"] != trunk.as_dict():
        raise RelationTrainingRefused("R4 bundle trunk identity differs from the fresh model")
    if payload["source_digest"] != _source_digest(payload["source_hashes"]):
        raise RelationTrainingRefused("R4 bundle source digest does not match source hashes")
    if payload["environment"] != _environment(config):
        raise RelationTrainingRefused("R4 execution environment differs from the bundle")
    if tuple(sources) != AUTHORITATIVE_R4_SOURCES or source_hashes() != payload["source_hashes"]:
        raise RelationTrainingRefused("R4 source contents changed since the bundle was written")
    corpus_record = payload["corpus"]
    expected_corpus = {
        "manifest_path": None if training.manifest_path is None else str(training.manifest_path),
        "canonical_payload_sha256": training.canonical_payload_sha256,
        "file_sha256": training.file_sha256,
        "rebuilt_payload_sha256": training.rebuilt_payload_sha256,
        "training_program_fingerprint": training.program_fingerprint,
        "supervision_identity_version": SUPERVISION_IDENTITY_VERSION,
        "supervision_sha256": training.supervision_identity,
        "provenance": training.provenance,
    }
    if corpus_record != expected_corpus:
        raise RelationTrainingRefused("R4 corpus identity differs from the resumed training corpus")
    expected_lengths = [len(case.flat) + 1 for case in training.cases]
    if payload["sampler"]["lengths"] != expected_lengths:
        raise RelationTrainingRefused("R4 sampler lengths disagree with verified training corpus")
    # Audit consumed rows against verified corpus facts
    _audit_history_and_diagnostics(
        payload["history"], payload["diagnostics"], training, config, trunk.parameters
    )
    # Reconcile all private ordering state before restoring any stateful object:
    # a syntactically valid cursor still cannot be allowed to replay a different
    # corpus, batch size, or length-bucket regime.
    cursor = RelationBucketCursor.from_state(
        payload["sampler"], lengths=[len(case.flat) + 1 for case in training.cases],
        batch_size=config.batch_size, base_seed=config.seed,
    )
    if cursor.consumed_batches != payload["completed_step"]:
        raise RelationTrainingRefused("R4 sampler position disagrees with the completed step")
    _validate_optimizer_ownership(model, optimizer)
    _validate_frozen_model(model, config)
    _validate_model_state_dict(payload["model"], model.state_dict())
    _validate_optimizer_state_dict(payload["optimizer"], list(model.parameters()),
                                   config, payload["completed_step"],
                                   _parameter_update_steps(model, payload["history"]))
    # The public low-level API receives caller-owned objects. Preserve all of
    # them, including global/private RNG streams, if an unexpected load defect
    # survives structural validation.
    before_model = {name: value.detach().clone() for name, value in model.state_dict().items()}
    before_optimizer = deepcopy(optimizer.state_dict())
    before_rng = capture_rng_state(data_generator)
    try:
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        restore_rng_state(payload["rng"], data_generator)
        # Final verification is part of restoration, not a post-commit check.
        last_used = lr_at(max(0, payload["completed_step"] - 1), config)
        if any(group["lr"] != last_used for group in optimizer.param_groups):
            raise RelationTrainingRefused("R4 optimizer learning rate disagrees with the declared schedule")
        actual = {"model": state_digest(model.state_dict()),
                  "optimizer": optimizer_digest(optimizer, model),
                  "rng": rng_digest(config.device, data_generator)}
        if payload["content_digests"] != actual:
            raise RelationTrainingRefused("R4 bundle content digest disagrees after restore")
    except BaseException:
        try:
            model.load_state_dict(before_model, strict=True)
            optimizer.load_state_dict(before_optimizer)
            restore_rng_state(before_rng, data_generator)
        except BaseException as rollback_error:
            raise RuntimeError("R4 failed to restore checkpoint state and rollback caller objects") from rollback_error
        raise
    return payload
