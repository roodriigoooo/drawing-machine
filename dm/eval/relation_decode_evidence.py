"""Standalone non-gating Point 4 engineering fixtures; never loads a checkpoint.

Runtime owns no dependency on this module. Fixed scripts exercise decoding,
not learned action accuracy. Reports carry no model/optimizer state.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields, replace
from itertools import product
from pathlib import Path

import numpy as np
import torch

from ..isa.codec import BOS, N_SPECIAL
from ..isa.spec import ISAError, Op, Tier, spec_for
from ..isa.transform import D4, Transform
from ..models.relation import RELATION_KBEST_ACTIONS
from ..models.transformer import Config, DrawingLM
from ..relation import (
    COUNT_SUPPORT,
    D4_SUPPORT,
    ORACLE_SUPPORT,
    PREDICTED_SUPPORT,
    TRANSLATION_SUPPORT,
    CopyActionKey,
    FaultCode,
    TransducerFault,
    candidate_spans,
    execute_copy,
    prefix_index,
    prefix_progress,
)
from ..relation.decoding import (
    BATCH_COUNTERS,
    CONTENT_POLICY,
    ROW_COUNTERS,
    STOPPING_POLICY,
    ActionDecision,
    RequestCompleted,
    RequestStarted,
    RowStopped,
    ScoreDetail,
    reconcile,
)
from .relation_evidence import execution_environment, rng_digest, state_digest

ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATHS = (
    "dm/models/transformer.py", "dm/models/relation.py", "dm/relation/__init__.py",
    "dm/relation/decoding.py", "dm/relation/queue.py", "dm/relation/candidates.py",
    "dm/relation/spec.py", "dm/relation/transducer.py", "dm/isa/spec.py", "dm/isa/asm.py",
    "dm/isa/codec.py", "dm/isa/transform.py", "dm/data/augment.py", "pyproject.toml",
    "dm/eval/relation_decode_evidence.py",
    "dm/eval/relation_evidence.py", "scripts/relation_decode_fixture.py",
    "tests/test_relation_decode_policy.py", "tests/test_relation_decode_observation.py",
    "tests/test_relation_decode_evidence.py", "tests/test_relation_decode.py",
    "tests/test_relation_model.py", "tests/test_relation_r4_correction.py",
    "tests/fixtures/relation_generate_sha256.txt",
)
MOTIF = bytes((1, 112, 120, 2, 128, 136))
CONFIG = Config(vocab_size=258, d_model=16, n_layers=1, n_heads=2, max_len=128,
                relation_schema="span_affine_v1")
EVENT_TYPES = {cls.__dataclass_fields__["kind"].default: cls for cls in
               (RequestStarted, ActionDecision, ScoreDetail, RowStopped, RequestCompleted)}


class DecodeEvidenceRefused(ValueError):
    """Malformed, inconsistent or out-of-scope engineering evidence."""


def _json(value):
    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


def _digest(value):
    return hashlib.sha256(json.dumps(value, allow_nan=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class _FixtureSpec:
    name: str
    mode: str
    prompts: tuple[bytes, ...]
    continuations: tuple[bytes, ...]
    counts: tuple[int, ...]
    stop_at: int | None = None
    uniforms: tuple[tuple[float, ...], ...] | None = None
    profile: str = "counts"

    @property
    def horizon(self):
        return max(map(len, self.continuations))


def _fixtures():
    fixtures = []
    for mode in ("standard", "predicted_copy", "oracle_copy"):
        for name, continuation in (("invalid", b"\xff"), ("non_flat", b"\x07"),
                                   ("partial", b"\x01\x00"),
                                   ("operand_zero", b"\x01\x00\x00\x00")):
            fixtures.append(_FixtureSpec(
                f"{mode}_{name}", mode, (b"",), (continuation,), (0,),
            ))
        fixtures.append(_FixtureSpec(
            f"{mode}_literal", mode, (MOTIF,), (MOTIF + b"\x00",), (0,),
        ))
        predecessor = torch.nextafter(torch.tensor(1.0), torch.tensor(0.0)).item()
        fixtures.append(_FixtureSpec(
            f"{mode}_cdf_edges", mode, (b"", b"", b""),
            (b"\x0a", b"\x14", b"\xff"), (0, 0, 0),
            uniforms=((0.0,), (0.5,), (predecessor,)), profile="cdf_edges",
        ))
    for mode in ("predicted_copy", "oracle_copy"):
        for count in (2, 3):
            fixtures.append(_FixtureSpec(
                f"{mode}_count{count}", mode, (MOTIF,),
                (MOTIF * (count - 1) + b"\x00",), (count,),
            ))
        # Equal-length terminal prompt uses five FILLs then HALT, not PAD.
        fixtures.append(_FixtureSpec(
            f"{mode}_heterogeneous", mode,
            (MOTIF, MOTIF, MOTIF, b"\x06" * 5 + b"\x00"),
            (MOTIF * 2 + b"\x00", MOTIF + b"\x00", b"\x00", b""),
            (3, 2, 0, 0),
        ))
        fixtures.append(_FixtureSpec(
            f"{mode}_mid_queue", mode, (MOTIF, MOTIF),
            (MOTIF + b"\x00", MOTIF * 2 + b"\x00"), (2, 3), 8,
        ))
    edge = bytes((1, 0, 0, 2, 1, 1))
    fixtures.extend((
        _FixtureSpec("predicted_copy_fault_then_success", "predicted_copy", (edge,),
                     (edge + b"\x00",), (2,), profile="fault_then_success"),
        _FixtureSpec("predicted_copy_attempt_cap", "predicted_copy", (MOTIF,),
                     (b"\x00",), (0,), profile="attempt_cap"),
    ))
    return tuple(fixtures)


class _FixtureLM(DrawingLM):
    """Finite scripted factor logits on real packed causal states."""

    def score_relation(self, packed):
        value = super().score_relation(packed)
        gates = torch.full_like(value.gate_logits, -1000)
        spans = torch.full_like(value.span_logits, -1000)
        counts = torch.full_like(value.count_logits, -1000)
        for query, row in enumerate(self.query_rows):
            count = self.counts[row] if self.offset == 6 else 0
            if self.profile == "attempt_cap" and self.offset == 6:
                count = 2
            gates[query, int(bool(count))] = 0
            counts[query, max(count - 2, 0)] = 0
            begin, end = packed.row_splits[query:query + 2].tolist()
            for candidate in range(begin, end):
                if (int(packed.source_start[candidate]), int(packed.source_stop[candidate])) == (0, 6):
                    spans[candidate] = 0
        factors = {}
        for name, index in (("d4_logits", 0), ("dx_logits", 1), ("dy_logits", 1)):
            logits = torch.full_like(getattr(value, name), -1000)
            logits[:, index] = 0
            factors[name] = logits
        if self.profile == "fault_then_success" and self.offset == 6:
            factors["dx_logits"][:, 0] = 0
            factors["dx_logits"][:, 1] = -1
        if self.profile == "attempt_cap" and self.offset == 6:
            for logits in factors.values():
                logits.zero_()
            counts.zero_()
        return replace(value, gate_logits=gates, span_logits=spans,
                       count_logits=counts, **factors)


class _Stop:
    stride = 1

    def __init__(self, rows, at):
        self.rows, self.at, self.position = rows, at, 0

    def step(self, chunk):
        self.position += 1
        return np.array([self.position >= self.at, *([False] * (self.rows - 1))])


def _run_fixture(spec):
    name, mode = spec.name, spec.mode
    prompts, continuations = spec.prompts, spec.continuations
    counts, stop_at, horizon = spec.counts, spec.stop_at, spec.horizon
    given = len(prompts[0])
    expected = []
    for row, (prompt, continuation) in enumerate(zip(prompts, continuations, strict=True)):
        program = prompt + continuation
        if stop_at is not None and row == 0:
            program = program[:stop_at]
        expected.append([2 + b for b in program])
    returned_width = max(map(len, expected))
    rectangle = [row + [0] * (returned_width - len(row)) for row in expected]
    torch.manual_seed(51)
    net = _FixtureLM(CONFIG).eval()
    net.counts = counts
    net.profile = spec.profile
    uniforms = (torch.tensor(spec.uniforms, dtype=torch.float32) if spec.uniforms is not None
                else torch.full((len(prompts), given + horizon), 0.5))
    actions = {(row, given): CopyActionKey(given, 0, 6, Transform(), count)
               for row, count in enumerate(counts) if count}
    discrepancies = []

    def scripted(position, raw):
        net.offset = position
        # Eligibility follows fixed scripts, not an observer callback.
        net.query_rows = []
        for row, symbols in enumerate(expected):
            if position >= len(symbols):
                continue
            full = torch.tensor([[BOS, *symbols[:position]]])
            with torch.no_grad():
                actual = net(full)[:, -1]
            discrepancies.append(float((actual[0] - raw[row]).abs().max()))
            if position == given or (position >= given + 6 * max(counts[row] - 1, 0)
                                     and (position - given) % 3 == 0):
                net.query_rows.append(row)
        raw.fill_(-1000)
        if spec.profile == "cdf_edges":
            raw[:, :N_SPECIAL] = 1000
            raw[0:2, N_SPECIAL + 10] = 0
            raw[0:2, N_SPECIAL + 20] = 0
            raw[2, N_SPECIAL + 255] = 0
        else:
            for row, symbols in enumerate(expected):
                raw[row, symbols[position] if position < len(symbols) else N_SPECIAL] = 0

    kwargs = {"mode": mode, "literal_policy": CONTENT_POLICY, "variates": uniforms,
              "on_logits": scripted}
    if mode == "oracle_copy":
        kwargs["oracle_actions"] = actions
    before = state_digest(net.state_dict())
    rng = rng_digest()
    results, snapshots = [], []
    action_streams = []
    for level in (None, "summary", "actions", "scores"):
        events = []
        result = net.generate_relation(
            prompts, horizon, **kwargs,
            monitor=None if stop_at is None else _Stop(len(prompts), stop_at),
            **({} if level is None else {"observer": events.append, "observation_level": level}))
        results.append(result.tolist())
        if level in ("actions", "scores"):
            action_streams.append([_json(asdict(event)) for event in events
                                   if isinstance(event, ActionDecision)])
        if level == "scores":
            snapshots = [_json(asdict(event)) for event in events]
        if state_digest(net.state_dict()) != before or rng_digest() != rng:
            raise RuntimeError("fixture observer changed model or RNG")
    if any(result != rectangle for result in results):
        raise RuntimeError(f"{name}: scripted output mismatch")
    maximum = max(discrepancies, default=0.0)
    if maximum > 2e-6:
        raise RuntimeError(f"{name}: cached/full-prefix discrepancy {maximum}")
    if action_streams[0] != action_streams[1]:
        raise RuntimeError(f"{name}: action observation changed decisions")
    programs = [prompt + continuation
                for prompt, continuation in zip(prompts, continuations, strict=True)]
    if stop_at is not None:
        programs[0] = programs[0][:stop_at]
    lengths = list(map(len, programs))
    copy_positions, action_events = _action_facts(
        snapshots, mode, programs, lengths, given, horizon,
    )
    expected_rows, expected_calls = _expected_counters(
        mode, programs, lengths, action_events, copy_positions, given, returned_width,
    )
    expected_totals = {field: sum(row[field] for row in expected_rows) for field in ROW_COUNTERS}
    expected_totals.update(expected_calls)
    observed = snapshots[-1]
    expected_statuses = []
    for row, program in enumerate(programs):
        status = _stop_fact(program, given, horizon, row, stop_at)
        discarded = sum(event["admission_bytes"] for event in action_events
                        if event["row"] == row) - len(copy_positions[row])
        status.update(pending_before_discard=discarded, discarded_bytes=discarded)
        expected_statuses.append(status)
    return {
        "name": name,
        "mode": mode,
        "prompts": [list(prompt) for prompt in prompts],
        "horizon": horizon,
        "uniforms": uniforms.tolist(),
        "uniform_digest": _digest(uniforms.tolist()),
        "expected_tokens": rectangle,
        "observed_tokens": results[-1],
        "expected_statuses": expected_statuses,
        "observed_statuses": observed["statuses"],
        "expected_work": {"per_row": expected_rows, "totals": expected_totals},
        "observed_work": {"per_row": [dict(pairs) for pairs in observed["per_row"]],
                          "totals": dict(observed["totals"])},
        "events": snapshots,
        "cache_max_abs": maximum,
        "cache_atol": 2e-6,
        "observer_comparison": {
            "levels": ["off", "summary", "actions", "scores"],
            "outputs_equal": True,
            "decisions_equal": True,
            "model_state_equal": True,
            "rng_equal": True,
        },
        "external_monitor": None if stop_at is None else "fixed_row0_byte8_stop",
    }


def build_report():
    """Run fixed CPU fixtures. No publication on any failed check."""
    source_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                     for name in SOURCE_PATHS}
    report = {
        "schema": 1,
        "kind": "r4_point4_decode_work",
        "provenance": "engineering",
        "seed": 51,
        "model_config": asdict(CONFIG),
        "literal_policy": CONTENT_POLICY,
        "stopping_policy": STOPPING_POLICY,
        "determinism": {
            "fixture_selection": "fixed_without_checkpoint_v1",
            "model_weights": "torch_seeded_random_initialization_v1",
            "observer_levels": ["off", "summary", "actions", "scores"],
            "position_keyed_uniforms": True,
        },
        "source_hashes": source_hashes,
        "source_digest": _digest(source_hashes),
        "environment": execution_environment(device="cpu"),
        "fixtures": [_run_fixture(spec) for spec in _fixtures()],
    }
    validate_report(_json(report))
    return report


def _require(condition, detail):
    if not condition:
        raise DecodeEvidenceRefused(detail)


def _keys(value, expected, name):
    _require(type(value) is dict and set(value) == set(expected), f"{name}: key set")


def _finite(value):
    if isinstance(value, dict):
        return all(type(k) is str and _finite(v) for k, v in value.items())
    if isinstance(value, list):
        return all(_finite(v) for v in value)
    return value is None or type(value) in (str, bool, int) or (
        type(value) is float and math.isfinite(value))


def validate_report(report):
    """Reconstruct byte and action counts; reject extra fields and bad coordinates."""
    _require(_finite(report), "nonfinite or non-JSON report")
    _keys(report, ("schema", "kind", "provenance", "seed", "model_config", "literal_policy",
                   "stopping_policy", "determinism", "source_hashes", "source_digest",
                   "environment", "fixtures"), "report")
    _require(type(report["schema"]) is int and report["schema"] == 1
             and report["kind"] == "r4_point4_decode_work"
             and report["provenance"] == "engineering"
             and type(report["seed"]) is int and report["seed"] == 51,
             "report identity")
    _require(_same(report["model_config"], _json(asdict(CONFIG))), "model config")
    _require(report["literal_policy"] == CONTENT_POLICY
             and report["stopping_policy"] == STOPPING_POLICY, "policies")
    _require(_same(report["determinism"], {
        "fixture_selection": "fixed_without_checkpoint_v1",
        "model_weights": "torch_seeded_random_initialization_v1",
        "observer_levels": ["off", "summary", "actions", "scores"],
        "position_keyed_uniforms": True,
    }), "determinism settings")
    _keys(report["source_hashes"], SOURCE_PATHS, "source inventory")
    expected_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                       for name in SOURCE_PATHS}
    _require(report["source_hashes"] == expected_hashes
             and report["source_digest"] == _digest(expected_hashes), "source identity")
    _keys(report["environment"], execution_environment(device="cpu"), "environment")
    _require(_same(report["environment"], _json(execution_environment(device="cpu"))),
             "CPU environment identity")
    specs = _fixtures()
    _require(type(report["fixtures"]) is list and len(report["fixtures"]) == len(specs),
             "fixture inventory")
    for fixture, spec in zip(report["fixtures"], specs, strict=True):
        _validate_fixture(fixture, spec)


_CANDIDATE_FAULTS = {
    FaultCode.BOUNDARY.value,
    FaultCode.SOURCE_RANGE.value,
    FaultCode.SOURCE_EMPTY.value,
    FaultCode.SOURCE_BYTES.value,
    FaultCode.SOURCE_INSTRUCTIONS.value,
    FaultCode.SOURCE_GAP.value,
    FaultCode.SOURCE_COORDINATES.value,
    FaultCode.HALT_SOURCE.value,
    FaultCode.CANVAS.value,
    FaultCode.MAX_LENGTH.value,
}
_SEARCH_EXITS = {
    "no_candidates", "emit_dominates", "accepted_copy", "attempt_cap", "support_exhausted",
}
_SCORE_SUPPORTS = [["gate", [0, 1]], ["d4", list(D4_SUPPORT)],
                   ["dx", list(TRANSLATION_SUPPORT)],
                   ["dy", list(TRANSLATION_SUPPORT)], ["count", list(COUNT_SUPPORT)]]


def _same(actual, expected):
    """JSON equality with exact scalar domains (bool is never an integer)."""
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return actual.keys() == expected.keys() and all(
            _same(actual[key], value) for key, value in expected.items())
    if type(expected) is list:
        return len(actual) == len(expected) and all(
            _same(a, b) for a, b in zip(actual, expected, strict=True))
    return actual == expected


def _judge_query(spec, row, offset, prefix):
    """Independent tiny-fixture exhaustive judge; never calls runtime search/head.

    Enumerate only fixed fixture spans, not arbitrary report endpoints. Float32
    addition order and canonical tie indices are part of the frozen contract.
    """
    spans = candidate_spans(prefix, offset)
    count = spec.counts[row] if offset == 6 else 0
    if spec.profile == "attempt_cap" and offset == 6:
        count = 2
    raw = {"gate": [-1000.0, -1000.0],
           "span": [0.0 if span == (0, 6) else -1000.0 for span in spans],
           "d4": [0.0] + [-1000.0] * 7,
           "dx": [-1000.0, 0.0, -1000.0],
           "dy": [-1000.0, 0.0, -1000.0],
           "count": [-1000.0] * 3}
    raw["gate"][int(bool(count))] = 0.0
    raw["count"][max(count - 2, 0)] = 0.0
    if spec.profile == "fault_then_success" and offset == 6:
        raw["dx"] = [0.0, -1.0, -1000.0]
    if spec.profile == "attempt_cap" and offset == 6:
        for name in ("d4", "dx", "dy", "count"):
            raw[name] = [0.0] * len(raw[name])
    detail = {"kind": "score_detail", "row": row, "byte_offset": offset,
              "endpoints": [list(span) for span in spans], **raw,
              "supports": _SCORE_SUPPORTS,
              "normalization": "raw_logits;normalization=float32_log_softmax_per_factor_real_spans"}
    lp = {name: torch.tensor(values, dtype=torch.float32).log_softmax(0)
          for name, values in raw.items()}
    emit_score = float(lp["gate"][0])
    action = {"kind": "action_decision", "row": row, "byte_offset": offset,
              "action_source": "predicted", "candidate_count": len(spans),
              "action_key": None, "decision_score": emit_score,
              "executor_attempts": 0, "faults": [],
              "exit_reason": "no_candidates", "admission_bytes": 0}
    ranked = []
    for indices in product(range(len(spans)), range(8), range(3), range(3), range(3)):
        s, d, x, y, c = indices
        score = float(lp["gate"][1] + lp["span"][s] + lp["d4"][d]
                      + lp["dx"][x] + lp["dy"][y] + lp["count"][c])
        ranked.append((score, indices))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    faults = {}
    if ranked:
        action["exit_reason"] = "support_exhausted"
    for score, (s, d, x, y, c) in ranked:
        if score < emit_score:
            action["exit_reason"] = "emit_dominates"
            break
        start, stop = spans[s]
        key = CopyActionKey(offset, start, stop,
                            Transform(D4.of(D4_SUPPORT[d]), TRANSLATION_SUPPORT[x],
                                      TRANSLATION_SUPPORT[y]), COUNT_SUPPORT[c])
        action["executor_attempts"] += 1
        try:
            execution = execute_copy(prefix, key, policy=PREDICTED_SUPPORT,
                                     max_len=len(spec.prompts[0]) + spec.horizon)
        except TransducerFault as fault:
            if fault.code.value not in _CANDIDATE_FAULTS:
                raise
            faults[fault.code.value] = faults.get(fault.code.value, 0) + 1
        else:
            action.update(action_key=[offset, start, stop, D4_SUPPORT[d],
                                      TRANSLATION_SUPPORT[x], TRANSLATION_SUPPORT[y],
                                      COUNT_SUPPORT[c]], decision_score=score,
                          admission_bytes=len(execution.appended), exit_reason="accepted_copy")
            break
        if action["executor_attempts"] == RELATION_KBEST_ACTIONS:
            action["exit_reason"] = "attempt_cap"
            break
    action["faults"] = [list(pair) for pair in sorted(faults.items())]
    return action, detail


def _judge_trace(spec, programs):
    """Reconstruct causal query/stop order and queue occupancy from fixture recipe."""
    given = len(spec.prompts[0])
    lengths = list(map(len, programs))
    pending_until = [given] * len(programs)
    trace = []

    def stop(row):
        status = _stop_fact(programs[row], given, spec.horizon, row, spec.stop_at)
        discarded = max(0, pending_until[row] - lengths[row])
        status.update(pending_before_discard=discarded, discarded_bytes=discarded)
        trace.append(status)

    for row, length in enumerate(lengths):
        if length == given:
            stop(row)
    for offset in range(given, max(lengths)):
        for row, program in enumerate(programs):
            if offset >= lengths[row] or offset < pending_until[row]:
                continue
            if spec.mode == "predicted_copy" and prefix_progress(program[:offset]).complete:
                action, detail = _judge_query(spec, row, offset, program[:offset])
                trace.extend((action, detail))
                pending_until[row] = offset + action["admission_bytes"]
            elif spec.mode == "oracle_copy" and offset == given and spec.counts[row]:
                count = spec.counts[row]
                key = CopyActionKey(6, 0, 6, Transform(), count)
                execution = execute_copy(program[:offset], key, policy=ORACLE_SUPPORT,
                                         max_len=given + spec.horizon)
                admission = len(execution.appended)
                trace.append({"kind": "action_decision", "row": row,
                              "byte_offset": offset, "action_source": "oracle",
                              "candidate_count": 0, "action_key": [6, 0, 6, 0, 0, 0, count],
                              "decision_score": None, "executor_attempts": 1, "faults": [],
                              "exit_reason": "oracle_executed", "admission_bytes": admission})
                pending_until[row] = offset + admission
        for row, length in enumerate(lengths):
            if length == offset + 1:
                stop(row)
    return trace


def _pairs(value, names, detail):
    _require(type(value) is list and len(value) == len(names), detail)
    _require(all(type(pair) is list and len(pair) == 2 and type(pair[0]) is str
                 and type(pair[1]) is int and pair[1] >= 0 for pair in value), detail)
    _require([pair[0] for pair in value] == list(names), detail)
    return dict(value)


def _stop_fact(program, given, horizon, row, stop_at):
    remaining = 0
    prompt_halt = False
    width = given + horizon
    for offset, byte in enumerate(program):
        cause = None
        if remaining:
            remaining -= 1
        else:
            try:
                instruction = spec_for(byte)
            except ISAError:
                cause = "invalid_opcode"
            else:
                if instruction.op is Op.HALT:
                    cause = "prompt_halt" if offset < given else "halt"
                    prompt_halt |= offset < given
                elif instruction.tier is not Tier.L0:
                    cause = "non_flat_opcode"
                else:
                    remaining = instruction.size - 1
        external = stop_at is not None and row == 0 and offset + 1 >= stop_at
        at_horizon = offset >= given and offset + 1 == width
        if cause is None and external:
            cause = "external_monitor_stop"
        if cause is None and at_horizon:
            cause = "horizon_partial" if remaining else "horizon_boundary"
        if cause is not None:
            return {
                "row": row,
                "useful_length": len(program),
                "generated_bytes": len(program) - given,
                "cause": cause,
                "byte_offset": offset,
                "pending_before_discard": 0,
                "discarded_bytes": 0,
                "external_monitor_stop": external,
                "at_horizon": at_horizon,
                "prompt_halt": prompt_halt,
                "kind": "row_stopped",
            }
    raise DecodeEvidenceRefused("output has no terminal classification")


def _action_facts(events, mode, programs, lengths, given, horizon):
    actions = [event for event in events if event["kind"] == "action_decision"]
    scores = [event for event in events if event["kind"] == "score_detail"]
    if mode == "standard":
        _require(not actions and not scores, "standard action events")
        return [set() for _ in programs], actions
    if mode == "oracle_copy":
        _require(not scores, "oracle score events")
    stopped = set()
    score_by_query = {}
    for index, event in enumerate(events[1:-1], 1):
        kind = event["kind"]
        if kind == "row_stopped":
            row = event["row"]
            _require(type(row) is int and 0 <= row < len(programs) and row not in stopped,
                     "stop event domain")
            stopped.add(row)
            continue
        if kind not in ("action_decision", "score_detail"):
            _require(False, "event lifecycle")
        row, offset = event["row"], event["byte_offset"]
        _require(type(row) is int and 0 <= row < len(programs) and row not in stopped,
                 "query row domain")
        _require(type(offset) is int and given <= offset < lengths[row], "query offset domain")
        query = (row, offset)
        if kind == "score_detail":
            _require(query not in score_by_query, "duplicate score query")
            score_by_query[query] = event
            continue
        _require(mode == "predicted_copy" or event["action_source"] == "oracle",
                 "action source")
        if mode == "predicted_copy":
            _require(event["action_source"] == "predicted"
                     and index + 1 < len(events)
                     and events[index + 1]["kind"] == "score_detail"
                     and (events[index + 1]["row"], events[index + 1]["byte_offset"]) == query,
                     "predicted action/score lifecycle")
    _require(stopped == set(range(len(programs))), "event stop lifecycle")

    copy_positions = [set() for _ in programs]
    seen_queries = set()
    for event in actions:
        row, offset = event["row"], event["byte_offset"]
        query = (row, offset)
        _require(query not in seen_queries, "duplicate action query")
        seen_queries.add(query)
        prefix = programs[row][:offset]
        try:
            spans = candidate_spans(prefix, len(prefix))
        except TransducerFault as exc:
            raise DecodeEvidenceRefused("action on invalid prefix") from exc
        _require(type(event["candidate_count"]) is int and event["candidate_count"] >= 0,
                 "candidate count domain")
        expected_candidates = len(spans) if mode == "predicted_copy" else 0
        _require(event["candidate_count"] == expected_candidates, "candidate count")
        _require(type(event["executor_attempts"]) is int and event["executor_attempts"] >= 0,
                 "attempt domain")
        faults = event["faults"]
        _require(type(faults) is list and all(
            type(pair) is list and len(pair) == 2 and type(pair[0]) is str
            and pair[0] in _CANDIDATE_FAULTS
            and type(pair[1]) is int and pair[1] > 0 for pair in faults
        ) and [pair[0] for pair in faults] == sorted({pair[0] for pair in faults}),
                 "fault histogram")
        fault_count = sum(pair[1] for pair in faults)
        key_values = event["action_key"]
        admission = event["admission_bytes"]
        _require(type(admission) is int and admission >= 0, "admission domain")
        if mode == "oracle_copy":
            _require(event["decision_score"] is None and event["exit_reason"] == "oracle_executed"
                     and event["executor_attempts"] == 1 and not faults and key_values is not None,
                     "oracle action semantics")
        else:
            _require(type(event["decision_score"]) is float
                     and math.isfinite(event["decision_score"])
                     and type(event["exit_reason"]) is str
                     and event["exit_reason"] in _SEARCH_EXITS
                     and event["executor_attempts"] <= RELATION_KBEST_ACTIONS,
                     "predicted decision domain")
            if key_values is None:
                _require(admission == 0 and event["executor_attempts"] == fault_count,
                         "EMIT decision semantics")
                if event["exit_reason"] == "no_candidates":
                    _require(not spans and event["executor_attempts"] == 0,
                             "empty-candidate exit")
                elif event["exit_reason"] == "emit_dominates":
                    _require(bool(spans) and event["executor_attempts"] < RELATION_KBEST_ACTIONS,
                             "EMIT-dominance exit")
                elif event["exit_reason"] == "attempt_cap":
                    _require(event["executor_attempts"] == RELATION_KBEST_ACTIONS,
                             "attempt-cap exit")
                elif event["exit_reason"] == "support_exhausted":
                    _require(0 < event["executor_attempts"] < RELATION_KBEST_ACTIONS,
                             "support-exhaustion exit")
                else:
                    _require(False, "COPY exit without action")
            else:
                _require(event["exit_reason"] == "accepted_copy"
                         and event["executor_attempts"] == fault_count + 1,
                         "accepted-COPY semantics")
        if key_values is None:
            continue
        _require(type(key_values) is list and len(key_values) == 7
                 and all(type(value) is int for value in key_values), "action key domain")
        boundary, start, stop, d4, dx, dy, count = key_values
        _require(boundary == offset and (start, stop) in spans
                 and d4 in D4_SUPPORT and dx in TRANSLATION_SUPPORT
                 and dy in TRANSLATION_SUPPORT, "action support")
        policy = ORACLE_SUPPORT if mode == "oracle_copy" else PREDICTED_SUPPORT
        _require(count in policy.total_counts, "count support")
        try:
            execution = execute_copy(
                prefix, CopyActionKey(boundary, start, stop, Transform(D4.of(d4), dx, dy), count),
                policy=policy, max_len=given + horizon,
            )
        except TransducerFault as exc:
            raise DecodeEvidenceRefused("committed action does not execute") from exc
        _require(admission == len(execution.appended) > 0, "admission bytes")
        delivered_stop = min(offset + admission, lengths[row])
        expected = execution.appended[:delivered_stop - offset]
        _require(programs[row][offset:delivered_stop] == expected, "delivered COPY payload")
        positions = set(range(offset, delivered_stop))
        _require(not copy_positions[row] & positions, "overlapping COPY admissions")
        copy_positions[row].update(positions)
    if mode == "predicted_copy":
        expected_queries = set()
        for row, program in enumerate(programs):
            starts = {event["byte_offset"] for event in actions
                      if event["row"] == row and event["action_key"] is not None}
            for offset in range(given, lengths[row]):
                if offset in copy_positions[row] and offset not in starts:
                    continue
                try:
                    complete = prefix_progress(program[:offset]).complete
                except TransducerFault as exc:
                    raise DecodeEvidenceRefused("query prefix became invalid") from exc
                if complete:
                    expected_queries.add((row, offset))
        _require(seen_queries == expected_queries, "predicted query lifecycle")
        _require(set(score_by_query) == expected_queries, "score query lifecycle")
    for event in scores:
        row, offset = event["row"], event["byte_offset"]
        spans = candidate_spans(programs[row][:offset], offset)
        _require(event["endpoints"] == [list(span) for span in spans], "score endpoints")
        for name, size in (("gate", 2), ("span", len(spans)), ("d4", len(D4_SUPPORT)),
                           ("dx", len(TRANSLATION_SUPPORT)),
                           ("dy", len(TRANSLATION_SUPPORT)), ("count", len(COUNT_SUPPORT))):
            values = event[name]
            _require(type(values) is list and len(values) == size
                     and all(type(value) is float and math.isfinite(value) for value in values),
                     f"score {name}")
        _require(event["supports"] == _SCORE_SUPPORTS
                 and event["normalization"]
                 == "raw_logits;normalization=float32_log_softmax_per_factor_real_spans",
                 "score support identity")
    return copy_positions, actions


def _expected_counters(mode, programs, lengths, actions, copy_positions, given, width):
    rows = len(programs)
    generated_columns = width - given
    counters = [{name: 0 for name in ROW_COUNTERS} for _ in programs]
    for row, program in enumerate(programs):
        c = counters[row]
        generated = lengths[row] - given
        admitted = sum(event["admission_bytes"] for event in actions if event["row"] == row)
        delivered = len(copy_positions[row])
        row_actions = [event for event in actions if event["row"] == row]
        c.update({
            "generated_bytes": generated,
            "literal_decisions": generated - delivered,
            "sampler_rows": generated - delivered,
            "uniform_coordinates_read": generated - delivered,
            "head_queries": len(row_actions) if mode == "predicted_copy" else 0,
            "candidate_entries": (sum(event["candidate_count"] for event in row_actions)
                                  if mode == "predicted_copy" else 0),
            "executor_attempts": sum(event["executor_attempts"] for event in row_actions),
            "predicted_search_attempts": (sum(event["executor_attempts"] for event in row_actions)
                                          if mode == "predicted_copy" else 0),
            "oracle_executor_calls": (sum(event["executor_attempts"] for event in row_actions)
                                      if mode == "oracle_copy" else 0),
            "copy_admissions": sum(event["action_key"] is not None for event in row_actions),
            "copy_admitted_bytes": admitted,
            "copy_delivered_bytes": delivered,
            "copy_discarded_bytes": admitted - delivered,
            "prefill_row_positions": given + 1,
            "incremental_row_positions": max(0, generated_columns - 1),
        })
        if mode != "standard":
            c["boundary_slots_used"] = len(prefix_index(program[:given])[0])
        stop_offset = lengths[row] - 1
        for offset in range(given, width - 1):
            copied = offset in copy_positions[row]
            emitted = offset < lengths[row]
            if emitted:
                c["fed_copy_bytes" if copied else "fed_literal_bytes"] += 1
            else:
                c["fed_pad_symbols"] += 1
            stopped = lengths[row] == given or stop_offset <= offset
            c["fed_stopped_row_positions" if stopped else "fed_live_row_positions"] += 1
            if mode != "standard" and emitted and not stopped \
                    and prefix_progress(program[:offset + 1]).complete:
                c["boundary_slots_used"] += 1
    if mode == "standard":
        for row, c in enumerate(counters):
            generated = lengths[row] - given
            c["sampler_rows"] = generated_columns
            c["uniform_coordinates_read"] = generated_columns
            c["discarded_sample_rows"] = generated_columns - generated
            c["literal_decisions"] = generated
    calls = {
        "prefill_calls": 1,
        "incremental_calls": max(0, generated_columns - 1),
        "head_batches": (len({event["byte_offset"] for event in actions})
                         if mode == "predicted_copy" else 0),
    }
    if mode == "standard":
        calls["sampler_calls"] = generated_columns
    else:
        calls["sampler_calls"] = sum(
            any(offset < lengths[row] and offset not in copy_positions[row]
                for row in range(rows))
            for offset in range(given, width)
        )
    return counters, calls


def _validate_fixture(f, spec):
    _keys(f, ("name", "mode", "prompts", "horizon", "uniforms", "uniform_digest",
              "expected_tokens", "observed_tokens", "expected_statuses", "observed_statuses",
              "expected_work", "observed_work", "events", "cache_max_abs", "cache_atol",
              "observer_comparison", "external_monitor"), "fixture")
    name, mode = spec.name, spec.mode
    prompts, continuations = spec.prompts, spec.continuations
    stop_at, horizon = spec.stop_at, spec.horizon
    rows, given = len(prompts), len(prompts[0])
    _require(_same([f["name"], f["mode"], f["prompts"], f["horizon"]],
                   [name, mode, [list(p) for p in prompts], horizon]), "fixture coordinates")
    expected_uniforms = ([list(row) for row in spec.uniforms] if spec.uniforms is not None
                         else [[0.5] * (given + horizon) for _ in prompts])
    _require(_same(f["uniforms"], expected_uniforms)
             and f["uniform_digest"] == _digest(f["uniforms"]), "uniform field")
    raw_programs = [p + c for p, c in zip(prompts, continuations, strict=True)]
    if stop_at is not None:
        raw_programs[0] = raw_programs[0][:stop_at]
    lengths = list(map(len, raw_programs))
    width = max(lengths)
    expected = [[N_SPECIAL + byte for byte in program] + [0] * (width - len(program))
                for program in raw_programs]
    _require(_same(f["expected_tokens"], expected)
             and _same(f["observed_tokens"], expected), "raw output")
    _require(_same(f["observer_comparison"], {
        "levels": ["off", "summary", "actions", "scores"],
        "outputs_equal": True,
        "decisions_equal": True,
        "model_state_equal": True,
        "rng_equal": True,
    }) and type(f["cache_atol"]) is float and f["cache_atol"] == 2e-6
       and type(f["cache_max_abs"]) is float
       and 0 <= f["cache_max_abs"] <= f["cache_atol"], "cache/transparency witness")
    _require(f["external_monitor"] == (None if stop_at is None else "fixed_row0_byte8_stop"),
             "external monitor identity")
    events = f["events"]
    _require(type(events) is list and len(events) >= rows + 2, "event stream")
    for event in events:
        _require(type(event) is dict and type(event.get("kind")) is str
                 and event["kind"] in EVENT_TYPES, "event kind")
        _keys(event, (field.name for field in fields(EVENT_TYPES[event["kind"]])), "event")
    _require(events[0]["kind"] == "request_started"
             and events[-1]["kind"] == "request_completed", "event lifecycle")
    _require(sum(event["kind"] == "request_started" for event in events) == 1
             and sum(event["kind"] == "request_completed" for event in events) == 1,
             "duplicate lifecycle")
    _require(_same(events[0], _json(asdict(RequestStarted(mode, CONTENT_POLICY, STOPPING_POLICY,
              "inverse_cdf", rows, given, horizon, 128, "scores")))), "request identity")
    completed = events[-1]
    _require(type(completed["returned_width"]) is int and completed["returned_width"] == width
             and type(completed["per_row"]) is list and len(completed["per_row"]) == rows,
             "completion shape")
    stops = [event for event in events if event["kind"] == "row_stopped"]
    _require(len(stops) == rows and all(type(event["row"]) is int for event in stops),
             "stop count/domain")
    _require({event["row"] for event in stops} == set(range(rows)), "stop rows")
    _require(completed["statuses"] == sorted(stops, key=lambda event: event["row"]),
             "stop status copies")

    judged_trace = _judge_trace(spec, raw_programs)
    _require(_same(events[1:-1], judged_trace), "fixture action/score/stop trace")
    copy_positions, actions = _action_facts(
        [events[0], *judged_trace, events[-1]], mode, raw_programs, lengths, given, horizon,
    )
    expected_statuses = []
    for row, program in enumerate(raw_programs):
        status = _stop_fact(program, given, horizon, row, stop_at)
        discarded = sum(event["admission_bytes"] for event in actions if event["row"] == row) \
            - len(copy_positions[row])
        status["pending_before_discard"] = discarded
        status["discarded_bytes"] = discarded
        expected_statuses.append(status)
    _require(_same(completed["statuses"], expected_statuses)
             and _same(f["expected_statuses"], expected_statuses)
             and _same(f["observed_statuses"], completed["statuses"]),
             "reconstructed stop facts")

    expected_rows, expected_calls = _expected_counters(
        mode, raw_programs, lengths, actions, copy_positions, given, width,
    )
    totals = _pairs(completed["totals"], ROW_COUNTERS + BATCH_COUNTERS, "total counters")
    observed_rows = []
    for row, pairs in enumerate(completed["per_row"]):
        observed = _pairs(pairs, ROW_COUNTERS, "row counters")
        try:
            reconcile(observed, True)
        except RuntimeError as exc:
            raise DecodeEvidenceRefused(str(exc)) from exc
        _require(observed == expected_rows[row], "reconstructed row work")
        observed_rows.append(observed)
    expected_totals = {name: sum(row[name] for row in expected_rows) for name in ROW_COUNTERS}
    expected_totals.update(expected_calls)
    _require(totals == expected_totals
             and _same(f["expected_work"], {"per_row": expected_rows, "totals": expected_totals})
             and _same(f["observed_work"], {"per_row": observed_rows, "totals": totals}),
             "reconstructed batch work")
    try:
        reconcile(totals, True)
    except RuntimeError as exc:
        raise DecodeEvidenceRefused(str(exc)) from exc
    _require(type(completed["kv_allocated_tensor_bytes"]) is int
             and completed["kv_allocated_tensor_bytes"]
             == 2 * rows * (given + horizon) * 16 * 4
             and type(completed["boundary_value_storage_bytes"]) is int
             and completed["boundary_value_storage_bytes"] == (
                 0 if mode == "standard" else rows * (given + horizon + 1) * 16 * 4)
             and type(completed["boundary_index_entries"]) is int
             and completed["boundary_index_entries"] == totals["boundary_slots_used"],
             "storage units")


def publish_report(path: Path):
    """Exclusive publication after validation; an existing path is never replaced."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    report = build_report()
    encoded = json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
    return report
