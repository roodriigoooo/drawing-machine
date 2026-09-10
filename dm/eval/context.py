"""Direction 2's frozen natural twins, their matched controls and the C3 scorer.

A `ContextCase` is a *model-blind* object.  Everything in it -- which prefix,
which target, which counterfactual step, which donor, which unrelated block --
is decided by the ISA, the frozen corpus counts and a seeded RNG, before any
checkpoint exists.  `docs/directions.md` §4.3 is the contract this Module
implements and `tests/test_context.py` is where it is pinned.

**Three estimands, not one.**  Each case carries three four-way blocks, all
teacher-forced under their own target histories:

```text
f(p)  = L(y_B | p) - L(y_A | p)            preference for the B-world target
D                = 1/2 * [f(x_A) - f(x_B)]        relation-bearing prefix change
D_target_control = 1/2 * [g(x_A) - g(x_B)]        g uses unrelated donor targets
D_block_control  = 1/2 * [f(u_A) - f(u_B)]        unrelated prefix block changed

Delta       = D - D_target_control
Delta_block = D - D_block_control
```

`x_A`/`x_B` differ only in the block that carries the relation, and `y_A`/`y_B`
are the continuations that relation implies.  `u_A`/`u_B` differ only in an
equally sized, equally shaped block that carries no relation to the target, and
they are scored against the *same* two targets -- so the compatible target is
`y_A` in both worlds and `D_block_control` is algebraically zero for any model
whose target preference does not move with an irrelevant prefix edit.  That is
the control the retracted schema-1 report did not have: its `D_control` was
zero because both control targets were the same string
([`docs/state.md`](../../docs/state.md)), which tests nothing at all.

**Raw NLLs are persisted, never only the contrast.**  A saved contrast cannot
be re-derived into its parts, and every failure mode in §4.8 is diagnosed from
the parts.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch

from ..data.fingerprint import digest
from ..isa.asm import parse
from ..isa.codec import Codec
from ..isa.spec import SPECS, Kind, Op
from ..isa.state import (
    CANONICAL,
    GrammarState,
    LanguagePolicy,
    SupportTable,
    SymbolCursor,
)
from .recovery import _symbol_bits
from .reports import staged_path

BUILDER_SEED = 20260815
DEFAULT_CASES = 64

#: Bumped from 2 with the completed control set.  A schema-2 manifest has no
#: unrelated-block control, no support columns and no matching features, so it
#: cannot be read as a schema-3 one; the old reports stay readable as history
#: and are never mixed with new output.
CASE_SCHEMA = 3
MANIFEST_SCHEMA = 3

#: The venues Direction 2 scores, and what each one identifies.  Keeping them
#: named here rather than spelled in three scripts is what stops the
#: x-anisotropic primary and its exact rotation from being pooled by accident.
VENUES: dict[str, str] = {
    "synthetic_flat_step": "primary in-support x-axis step twins",
    "synthetic_flat_step_d4r1": "exact quarter-turn rotation of the step venue",
    "synthetic_flat_step_yaxis": "step twins re-laid on the y axis in place",
    "composed_shape_copy2": "co-primary composed copy-2 shape twins",
}

#: Venues that answer the axis-transfer question, and the primary each one is
#: read against.  A diagnostic whose prefixes cost far more than its primary's
#: is out of the model's support and its null says nothing -- `prefix_cost_ratio`
#: is what a protocol conditions on.
AXIS_DIAGNOSTICS: dict[str, str] = {
    "synthetic_flat_step_d4r1": "synthetic_flat_step",
    "synthetic_flat_step_yaxis": "synthetic_flat_step",
}

#: Venues whose result the protocol may treat as confirmatory.  Everything else
#: is a mechanism diagnostic unless a freeze promotes it first (§4.7).
CO_PRIMARY_VENUES: tuple[str, ...] = ("synthetic_flat_step", "composed_shape_copy2")

#: How world B's bytes were obtained.  A self-derived counterfactual has no
#: second identity to hold out; a substituted block does, and it becomes an edge
#: in the dependence graph the bootstrap resamples.
DONOR_KINDS: frozenset[str] = frozenset({"self_translation", "substituted_block"})

Signature = tuple[tuple[str, tuple[str, ...]], ...]

#: The four-way order every block of NLLs uses, so a reader never has to guess
#: which of two crossed terms is which.
FOUR_WAY: tuple[str, ...] = ("a_given_a", "b_given_a", "a_given_b", "b_given_b")


def signature(program: bytes) -> Signature:
    """The opcode/operand-kind skeleton, which is what "same shape" means here.

    Equal byte length is not enough: `MOVE x y` and `CIRCLE r` followed by
    `FILL` occupy the same three bytes and are not the same target.
    """
    return tuple(
        (instr.mnemonic,
         tuple(kind.name for kind in SPECS[Op[instr.mnemonic]].operands))
        for instr in parse(program)
    )


def field_map(program: bytes) -> tuple[str, ...]:
    """One field label per byte: `opcode`, `coord_x`, `delta_y`, `count`, ...

    Paired operand kinds are split into their `x`/`y` halves because the whole
    point of the axis diagnostic is that the corpus is anisotropic, and a
    contribution table that summed the two could not show it.
    """
    fields: list[str] = []
    for instr in parse(program):
        spec = SPECS[Op[instr.mnemonic]]
        fields.append("opcode")
        index = 0
        while index < len(spec.operands):
            kind = spec.operands[index]
            paired = (kind in (Kind.COORD, Kind.DELTA)
                      and index + 1 < len(spec.operands)
                      and spec.operands[index + 1] is kind)
            if paired:
                fields += [f"{kind.name.lower()}_x", f"{kind.name.lower()}_y"]
                index += 2
            else:
                fields.append(kind.name.lower())
                index += 1
    return tuple(fields)


def first_instruction_bytes(program: bytes) -> int:
    """Byte length of the target's first instruction, for its own column."""
    for instr in parse(program):
        return SPECS[Op[instr.mnemonic]].size
    return 0


def byte_distance(left: bytes, right: bytes) -> int:
    """How many byte positions two equal-length strings disagree on."""
    if len(left) != len(right):
        raise ValueError(
            f"byte distance is only defined for equal lengths, got "
            f"{len(left)} and {len(right)}"
        )
    return sum(a != b for a, b in zip(left, right))


def prefix_is_live(prefix: bytes, policy: LanguagePolicy) -> bool:
    """True when a target may still be appended at a clean instruction boundary.

    "Live" is stricter than "not yet halted": the parse must sit on an opcode
    boundary with no open scope, or the target's first byte would land inside an
    operand and the four target strings would not be the objects the schema
    says they are.
    """
    state = GrammarState(policy)
    for byte in prefix:
        if state.done:
            return False
        state.step(byte)
    return (not state.violations and not state.done and
            state.phase.name == "OPCODE" and not state.scope_stack)


def target_is_legal(prefix: bytes, target: bytes, policy: LanguagePolicy) -> bool:
    """True when `prefix + target` stays inside the frozen canonical language."""
    state = GrammarState(policy)
    for byte in prefix + target:
        if state.done:
            return False
        state.step(byte)
    return not state.violations


def legal_first_bytes(prefix: bytes, policy: LanguagePolicy, codec: Codec) -> int:
    """How many symbols the frozen policy admits at the target boundary.

    A model-blind support descriptor.  It is deliberately computed from the
    policy and never from logits: §4.4 forbids a checkpoint from deciding which
    case is difficult enough to keep.
    """
    state = GrammarState(policy)
    for byte in prefix:
        state.step(byte)
    table = SupportTable(codec, policy, CANONICAL)
    return len(table.of(state, SymbolCursor(codec.stride)).legal)


# ---------------------------------------------------------------------------
# schema


@dataclass(frozen=True)
class Intervention:
    """One whole-instruction edit to a prefix, and what makes it matched.

    Spans are byte offsets into the prefix, so `validate_cases` can assert that
    the two worlds differ *here and nowhere else* -- the check that catches a
    builder which silently shifted a later field.
    """

    kind: str
    start: int
    stop: int
    signature: Signature
    dx: int
    dy: int
    byte_distance: int
    #: Prefix bytes between the end of the edited block and the target.  Zero
    #: for a relation-bearing block that abuts its continuation; whatever the
    #: corpus allows for the unrelated control, recorded rather than assumed.
    distance_to_target: int

    @property
    def length(self) -> int:
        return self.stop - self.start

    def to_dict(self) -> dict:
        out = asdict(self)
        out["signature"] = [[name, list(kinds)] for name, kinds in self.signature]
        return out

    @classmethod
    def from_dict(cls, value: dict) -> Intervention:
        value = dict(value)
        value["signature"] = tuple(
            (name, tuple(kinds)) for name, kinds in value["signature"]
        )
        return cls(**value)

    def rotated(self, dx: int, dy: int, distance: int) -> Intervention:
        return replace(self, dx=dx, dy=dy, byte_distance=distance)


@dataclass(frozen=True)
class MatchFeatures:
    """Model-blind matching columns and every relaxation, named.

    §4.4 allows a match to be relaxed and forbids it being relaxed silently, so
    each departure lands in `strata` and travels with the row into the balance
    table and the report.
    """

    target_ordinal_gap: int
    prefix_distance_gap: int
    byte_distance_primary: int
    byte_distance_control: int
    byte_distance_gap: int
    coordinate_marginal_distance: float
    frequency_a: float
    frequency_b: float
    frequency_control_a: float
    frequency_control_b: float
    legal_first_bytes: int
    strata: tuple[str, ...]

    def to_dict(self) -> dict:
        out = asdict(self)
        out["strata"] = list(self.strata)
        return out

    @classmethod
    def from_dict(cls, value: dict) -> MatchFeatures:
        value = dict(value)
        value["strata"] = tuple(value["strata"])
        return cls(**value)


#: Fields serialised as hex rather than as a JSON list of integers.
_BYTE_FIELDS: tuple[str, ...] = (
    "prefix_a", "prefix_b", "target_a", "target_b",
    "block_prefix_a", "block_prefix_b",
    "control_target_a", "control_target_b",
)


@dataclass(frozen=True)
class ContextCase:
    """One immutable case: two worlds, two matched controls, full provenance."""

    case_id: str
    venue: str
    schema: int

    # -- identity, split membership and the dependence graph ----------------
    source_index: int
    source_digest: str
    source_group: str
    donor_index: int
    donor_digest: str
    donor_group: str
    #: How world B's content was obtained.  `self_translation` means the
    #: counterfactual is derived from the source by the recorded relation, so
    #: there is no second identity to hold out; `substituted_block` means a real
    #: donor supplied the bytes and its identity is a dependence edge.
    donor_kind: str
    control_donor_index: int
    control_donor_digest: str
    corpus_split: str
    held_out_from_train: bool

    # -- the two worlds -----------------------------------------------------
    prefix_a: bytes
    prefix_b: bytes
    target_a: bytes
    target_b: bytes

    # -- unrelated-block control: same targets, an irrelevant prefix edit ----
    block_prefix_a: bytes
    block_prefix_b: bytes

    # -- unrelated-target control: same prefixes, donor targets -------------
    control_target_a: bytes
    control_target_b: bytes

    relevant: Intervention
    unrelated: Intervention

    target_signature: Signature
    target_ordinal: int
    target_byte_start: int
    target_instruction_ordinal: int
    control_target_byte_start: int
    #: `(dx, dy, mirror, quarter_turns)` of the relation the target realises.
    transform: tuple[int, int, int, int]
    features: MatchFeatures

    def __post_init__(self) -> None:
        if self.schema != CASE_SCHEMA:
            raise ValueError(
                f"{self.case_id}: case schema {self.schema} != {CASE_SCHEMA}"
            )
        if self.venue not in VENUES:
            raise ValueError(f"{self.case_id}: unknown venue {self.venue!r}")
        if self.donor_kind not in DONOR_KINDS:
            raise ValueError(
                f"{self.case_id}: donor_kind {self.donor_kind!r} is not one of "
                f"{sorted(DONOR_KINDS)}"
            )
        if (self.donor_kind == "self_translation") != (
                self.donor_index == self.source_index):
            raise ValueError(
                f"{self.case_id}: donor_kind {self.donor_kind!r} disagrees with "
                "the recorded donor identity"
            )
        if self.control_donor_index in (self.source_index, self.donor_index):
            raise ValueError(
                f"{self.case_id}: the control donor must be a third identity; "
                "reusing the source or the substitute donor makes the "
                "unrelated-target control partly related"
            )
        if self.target_a == self.target_b:
            raise ValueError(f"{self.case_id}: primary targets must differ")
        if self.control_target_a == self.control_target_b:
            raise ValueError(
                f"{self.case_id}: control targets must differ; an identical "
                "pair makes D_control zero by construction"
            )
        if self.prefix_a == self.prefix_b:
            raise ValueError(f"{self.case_id}: the two worlds share a prefix")
        if self.block_prefix_a == self.block_prefix_b:
            raise ValueError(
                f"{self.case_id}: the unrelated-block control worlds are equal, "
                "so D_block_control is zero by construction rather than by test"
            )
        lengths = {
            len(self.target_a), len(self.target_b),
            len(self.control_target_a), len(self.control_target_b),
        }
        if len(lengths) != 1:
            raise ValueError(f"{self.case_id}: all target spans must have one length")
        if len(self.prefix_a) != len(self.prefix_b):
            raise ValueError(f"{self.case_id}: the two prefixes must be equal length")
        if {len(self.block_prefix_a), len(self.block_prefix_b)} != {len(self.prefix_a)}:
            raise ValueError(
                f"{self.case_id}: the block-control prefixes must match the "
                "primary prefix length, or the target sits at another position"
            )

    @property
    def target_bytes(self) -> int:
        return len(self.target_a)

    @property
    def prefix_bytes(self) -> int:
        return len(self.prefix_a)

    def to_dict(self) -> dict:
        out = asdict(self)
        for key in _BYTE_FIELDS:
            out[key] = getattr(self, key).hex()
        out["relevant"] = self.relevant.to_dict()
        out["unrelated"] = self.unrelated.to_dict()
        out["features"] = self.features.to_dict()
        out["target_signature"] = [
            [name, list(kinds)] for name, kinds in self.target_signature
        ]
        out["transform"] = list(self.transform)
        return out

    @classmethod
    def from_dict(cls, value: dict) -> ContextCase:
        value = dict(value)
        for key in _BYTE_FIELDS:
            value[key] = bytes.fromhex(value[key])
        value["relevant"] = Intervention.from_dict(value["relevant"])
        value["unrelated"] = Intervention.from_dict(value["unrelated"])
        value["features"] = MatchFeatures.from_dict(value["features"])
        value["target_signature"] = tuple(
            (name, tuple(kinds)) for name, kinds in value["target_signature"]
        )
        value["transform"] = tuple(value["transform"])
        return cls(**value)


# ---------------------------------------------------------------------------
# manifest


def manifest_dict(*, cases: list[ContextCase], census: dict, corpus: dict,
                  policy: LanguagePolicy, corpus_stats: dict,
                  seed: int = BUILDER_SEED) -> dict:
    body = {
        "schema": MANIFEST_SCHEMA,
        "status": "frozen",
        "builder_seed": seed,
        "venues": sorted({case.venue for case in cases}),
        "corpus": corpus,
        "corpus_stats": corpus_stats,
        "policy": policy.as_dict(),
        "policy_digest": policy.digest,
        "census": census,
        "cases": [case.to_dict() for case in cases],
    }
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    body["manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
    return body


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = staged_path(path)
    staged.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    staged.replace(path)


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    expected = manifest.get("manifest_sha256")
    body = {key: value for key, value in manifest.items()
            if key != "manifest_sha256"}
    actual = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if expected != actual:
        raise ValueError(f"context manifest hash mismatch for {path}")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(
            f"unsupported context manifest schema {manifest.get('schema')!r}; "
            f"expected {MANIFEST_SCHEMA} with the complete control set"
        )
    if manifest.get("status") != "frozen":
        raise ValueError(f"context manifest is not frozen: {path}")
    cases = [ContextCase.from_dict(value) for value in manifest.get("cases", [])]
    included = (manifest.get("census") or {}).get("included_cases")
    if included != len(cases):
        raise ValueError(
            f"context manifest census says {included!r} cases but contains "
            f"{len(cases)}"
        )
    return manifest


def cases_from_manifest(manifest: dict, venue: str | None = None) -> list[ContextCase]:
    cases = [ContextCase.from_dict(value) for value in manifest["cases"]]
    return cases if venue is None else [c for c in cases if c.venue == venue]


def _assert_single_edit(case: ContextCase, left: bytes, right: bytes,
                        edit: Intervention, role: str) -> None:
    if left[:edit.start] != right[:edit.start] or left[edit.stop:] != right[edit.stop:]:
        raise ValueError(
            f"{case.case_id}: the {role} worlds differ outside the recorded "
            f"span [{edit.start}, {edit.stop})"
        )
    if left[edit.start:edit.stop] == right[edit.start:edit.stop]:
        raise ValueError(
            f"{case.case_id}: the {role} span does not actually change"
        )
    if signature(left[edit.start:edit.stop]) != edit.signature:
        raise ValueError(f"{case.case_id}: {role} block signature drift")


def validate_cases(cases: list[ContextCase], programs: list[bytes],
                   policy: LanguagePolicy) -> None:
    """Revalidate a frozen case table against the rebuilt held-out corpus.

    Every rule of §4.3 is re-checked here rather than trusted to the builder,
    because the builder and the scorer are different runs and a manifest is the
    only thing that travels between them.
    """
    for case in cases:
        for role, index, expected in (
            ("source", case.source_index, case.source_digest),
            ("donor", case.donor_index, case.donor_digest),
            ("control donor", case.control_donor_index, case.control_donor_digest),
        ):
            if index < 0 or index >= len(programs):
                raise ValueError(f"{case.case_id}: {role} index {index} out of range")
            actual = digest([programs[index]])
            if actual != expected:
                raise ValueError(
                    f"{case.case_id}: {role} digest {expected} != rebuilt {actual}"
                )
        for name, prefix in (("prefix_a", case.prefix_a), ("prefix_b", case.prefix_b),
                             ("block_prefix_a", case.block_prefix_a),
                             ("block_prefix_b", case.block_prefix_b)):
            if not prefix_is_live(prefix, policy):
                raise ValueError(f"{case.case_id}: {name} is not a live prefix")
        for target in (case.target_a, case.target_b,
                       case.control_target_a, case.control_target_b):
            if signature(target) != case.target_signature:
                raise ValueError(f"{case.case_id}: target signature drift")
            for prefix in (case.prefix_a, case.prefix_b,
                           case.block_prefix_a, case.block_prefix_b):
                if not target_is_legal(prefix, target, policy):
                    raise ValueError(
                        f"{case.case_id}: a target is not legal in every world"
                    )
        _assert_single_edit(case, case.prefix_a, case.prefix_b,
                            case.relevant, "primary")
        _assert_single_edit(case, case.block_prefix_a, case.block_prefix_b,
                            case.unrelated, "unrelated-block")
        if case.block_prefix_a != case.prefix_a:
            raise ValueError(
                f"{case.case_id}: the unrelated-block control must start from "
                "world A's prefix, or its contrast is not the primary's null"
            )


# ---------------------------------------------------------------------------
# scorer


@torch.no_grad()
def score_spans(model, requests: list[tuple[bytes, bytes]], codec: Codec,
                *, device: str | torch.device = "cpu",
                max_len: int = 2048, batch_size: int = 32) -> list[dict]:
    """Per-byte target bits and total prefix bits for `(prefix, target)` pairs.

    Each target is teacher-forced under **its own** prefix, which is the whole
    reason a four-way block exists: comparing later-token distributions across
    two different histories would compare two different conditionals and call
    the difference an effect (`docs/traps.md`).
    """
    stride = codec.stride
    programs = [prefix + target for prefix, target in requests]
    out: list[dict | None] = [None] * len(requests)
    for index, bits in _symbol_bits(model, programs, codec, device, max_len,
                                    batch_size):
        prefix, target = requests[index]
        lo, hi = len(prefix) * stride, (len(prefix) + len(target)) * stride
        if hi > len(bits):
            raise ValueError(
                f"request {index} is truncated at max_len={max_len}: a target "
                "scored on fewer symbols than it costs is not the estimand"
            )
        per_byte = bits[lo:hi].reshape(len(target), stride).sum(axis=1)
        out[index] = {
            "prefix_bits": float(bits[:lo].sum()),
            "bits": float(per_byte.sum()),
            "per_byte": [float(value) for value in per_byte],
        }
    return out  # type: ignore[return-value]


def _contrast(values: list[float], target_bytes: int) -> dict:
    """`1/2 * [(crossed_A - matched_A) + (crossed_B - matched_B)]`.

    `values` is in `FOUR_WAY` order.  The symmetry is what cancels an
    unconditional preference for either target string: a model that simply
    likes `y_B` pays the same extra bits under both prefixes and contributes
    zero here (§4.7 test 4).
    """
    value = 0.5 * ((values[1] - values[0]) + (values[2] - values[3]))
    return {"bits": value, "bits_per_byte": value / max(1, target_bytes)}


def _difference(left: dict, right: dict) -> dict:
    return {"bits": left["bits"] - right["bits"],
            "bits_per_byte": left["bits_per_byte"] - right["bits_per_byte"]}


def _block(scores: list[dict], target_bytes: int) -> dict:
    return {
        "nll_bits": {name: score["bits"] for name, score in zip(FOUR_WAY, scores)},
        "prefix_bits": {name: score["prefix_bits"]
                        for name, score in zip(FOUR_WAY, scores)},
        "contrast": _contrast([score["bits"] for score in scores], target_bytes),
    }


def _field_contributions(scores: list[dict], target: bytes) -> dict:
    """The same contrast, resolved by ISA field and by first instruction.

    An effect that lives entirely in `coord_x` and nothing else is a different
    finding from one spread over opcodes, and §4.6 asks for both columns.
    """
    fields = field_map(target)
    per_byte = [score["per_byte"] for score in scores]
    out: dict[str, float] = {}
    for name in sorted(set(fields)):
        mask = [index for index, label in enumerate(fields) if label == name]
        totals = [float(sum(row[index] for index in mask)) for row in per_byte]
        out[name] = _contrast(totals, 1)["bits"]
    head = first_instruction_bytes(target)
    totals = [float(sum(row[:head])) for row in per_byte]
    out["first_instruction"] = _contrast(totals, 1)["bits"]
    return out


@torch.no_grad()
def first_token_divergence(model, cases: list[ContextCase], codec: Codec,
                           *, device: str | torch.device = "cpu") -> list[dict]:
    """KL and TV between the two worlds at the first target symbol.

    The only distributional distance whose history is unambiguous: both worlds
    have emitted exactly their own prefix and nothing else.  Later positions
    need a named teacher-forced history and are reported as such or not at all.
    """
    model.eval()
    out = []
    for case in cases:
        rows = []
        for prefix in (case.prefix_a, case.prefix_b):
            encoded = torch.tensor([codec.with_bos(prefix)], dtype=torch.long,
                                   device=device)
            logits = model(encoded)[0, -1].float()
            rows.append(torch.log_softmax(logits, dim=-1))
        p, q = rows[0].exp(), rows[1].exp()
        kl = float((p * (rows[0] - rows[1])).sum() / math.log(2))
        out.append({
            "case_id": case.case_id,
            "kl_bits_a_to_b": kl,
            "total_variation": float(0.5 * (p - q).abs().sum()),
        })
    return out


def score_cases(model, cases: list[ContextCase], codec: Codec,
                *, device: str | torch.device = "cpu",
                max_len: int = 2048, batch_size: int = 32) -> list[dict]:
    """Score the primary block and both matched controls for every case."""
    requests: list[tuple[bytes, bytes]] = []
    for case in cases:
        requests += [
            # primary: relation-bearing prefix change, compatible targets
            (case.prefix_a, case.target_a), (case.prefix_a, case.target_b),
            (case.prefix_b, case.target_a), (case.prefix_b, case.target_b),
            # unrelated-target control: same prefixes, donor targets
            (case.prefix_a, case.control_target_a),
            (case.prefix_a, case.control_target_b),
            (case.prefix_b, case.control_target_a),
            (case.prefix_b, case.control_target_b),
            # unrelated-block control: same targets, irrelevant prefix edit
            (case.block_prefix_a, case.target_a),
            (case.block_prefix_a, case.target_b),
            (case.block_prefix_b, case.target_a),
            (case.block_prefix_b, case.target_b),
        ]
    scores = score_spans(model, requests, codec, device=device,
                         max_len=max_len, batch_size=batch_size)
    rows = []
    for index, case in enumerate(cases):
        start = 12 * index
        primary = scores[start:start + 4]
        target_control = scores[start + 4:start + 8]
        block_control = scores[start + 8:start + 12]
        primary_block = _block(primary, case.target_bytes)
        target_block = _block(target_control, case.target_bytes)
        block_block = _block(block_control, case.target_bytes)
        rows.append({
            "case_id": case.case_id,
            "venue": case.venue,
            "source_index": case.source_index,
            "donor_index": case.donor_index,
            "control_donor_index": case.control_donor_index,
            "control_targets_distinct": (
                case.control_target_a != case.control_target_b
            ),
            "target_bytes": case.target_bytes,
            "prefix_bytes": case.prefix_bytes,
            "strata": list(case.features.strata),
            "primary": primary_block,
            "target_control": target_block,
            "block_control": block_block,
            "D": primary_block["contrast"],
            "D_target_control": target_block["contrast"],
            "D_block_control": block_block["contrast"],
            "D_minus_control": _difference(primary_block["contrast"],
                                           target_block["contrast"]),
            "D_minus_block_control": _difference(primary_block["contrast"],
                                                 block_block["contrast"]),
            "fields": _field_contributions(primary, case.target_a),
            # §4.6: the unchanged target under both prefixes, so a reader can
            # see whether the contrast came from the target that did not move.
            "unchanged_target_nll_bits": {
                "a_given_a": primary[0]["bits"], "a_given_b": primary[2]["bits"],
            },
            "prefix_nll_bits": {
                "a": primary[0]["prefix_bits"], "b": primary[2]["prefix_bits"],
                "block_b": block_control[2]["prefix_bits"],
            },
            "support": {
                "legal_first_bytes": case.features.legal_first_bytes,
                "coordinate_marginal_distance":
                    case.features.coordinate_marginal_distance,
                "frequency_a": case.features.frequency_a,
                "frequency_b": case.features.frequency_b,
            },
        })
    return rows


# ---------------------------------------------------------------------------
# inference


def components(rows: list[dict]) -> dict[int, list[int]]:
    """Connected components of the source/donor/control-donor graph.

    A donor reused by two otherwise distinct primary sources connects both
    cases.  Resampling `source_index` alone would count dependent observations
    as independent clusters and shrink every interval (§4.7 rule 6).
    """
    parent: dict[int, int] = {}

    def find(node: int) -> int:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for row in rows:
        for key in ("donor_index", "control_donor_index"):
            if key not in row:
                raise ValueError(f"context row lacks {key} dependency identity")
            union(int(row["source_index"]), int(row[key]))
    groups: dict[int, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(find(int(row["source_index"])), []).append(index)
    return groups


def _bootstrap(rows: list[dict], field: str, *, seed: int = BUILDER_SEED,
               reps: int = 2000, unit: str = "bits_per_byte") -> dict:
    """Component bootstrap over any per-case contrast.

    `unit` exists because C4's estimand is a normalised distance rather than a
    rate in bits -- a free-running completion has no likelihood of its own. The
    resampling unit is identical, which is the point: the same dependence graph
    governs both stages.
    """
    groups = components(rows)
    if not groups or not rows:
        return {"mean": float("nan"), "ci95": [float("nan"), float("nan")],
                "clusters": 0, "reps": 0, "p_value": float("nan"),
                "cluster_unit": "connected_source_donor_component"}
    keys = sorted(groups)
    values = [float(row[field][unit]) for row in rows]
    rng = random.Random(seed)
    draws = []
    for _ in range(reps):
        sampled = [keys[rng.randrange(len(keys))] for _ in keys]
        picked = [values[index] for key in sampled for index in groups[key]]
        draws.append(sum(picked) / len(picked))
    draws.sort()
    # Two-sided percentile bootstrap p-value, floored at one replicate so a
    # finite resample is never reported as an exact zero.
    below = sum(value <= 0.0 for value in draws) / reps
    above = sum(value >= 0.0 for value in draws) / reps
    p = min(1.0, max(1.0 / reps, 2.0 * min(below, above)))
    return {
        "mean": sum(values) / len(values),
        "ci95": [draws[int(0.025 * (reps - 1))], draws[int(0.975 * (reps - 1))]],
        "p_value": p,
        "clusters": len(keys),
        "largest_cluster": max(len(members) for members in groups.values()),
        "cluster_unit": "connected_source_donor_component",
        "reps": reps,
    }


def summarise_scores(rows: list[dict], *, seed: int = BUILDER_SEED,
                     reps: int = 2000) -> dict:
    def mean(field: str, unit: str) -> float:
        return (float(np.mean([row[field][unit] for row in rows])) if rows
                else float("nan"))

    field_names = sorted({name for row in rows for name in row["fields"]})
    return {
        "n_cases": len(rows),
        "venues": sorted({row["venue"] for row in rows}),
        "D_bits_mean": mean("D", "bits"),
        "D_bits_per_byte_mean": mean("D", "bits_per_byte"),
        "D_control_bits_per_byte_mean": mean("D_target_control", "bits_per_byte"),
        "D_block_control_bits_per_byte_mean": mean("D_block_control",
                                                   "bits_per_byte"),
        "D_minus_control_bits_per_byte_mean": mean("D_minus_control",
                                                   "bits_per_byte"),
        "D_minus_block_control_bits_per_byte_mean": mean(
            "D_minus_block_control", "bits_per_byte"),
        "nondegenerate_control_rate": (
            float(np.mean([row["control_targets_distinct"] for row in rows]))
            if rows else float("nan")
        ),
        "field_contributions_bits_mean": {
            name: float(np.mean([row["fields"].get(name, 0.0) for row in rows]))
            if rows else float("nan")
            for name in field_names
        },
        "prefix_nll_bits_mean": {
            key: (float(np.mean([row["prefix_nll_bits"][key] for row in rows]))
                  if rows else float("nan"))
            for key in ("a", "b", "block_b")
        },
        # Rate, not total, because that is the only form comparable between a
        # venue and its diagnostic: a null on prefixes the model cannot predict
        # is not evidence about the relation (§4.1).
        "prefix_bits_per_byte_mean": (
            float(np.mean([row["prefix_nll_bits"]["a"] / max(1, row["prefix_bytes"])
                           for row in rows])) if rows else float("nan")
        ),
        "matched_target_bits_per_byte_mean": (
            float(np.mean([row["primary"]["nll_bits"]["a_given_a"]
                           / max(1, row["target_bytes"]) for row in rows]))
            if rows else float("nan")
        ),
        "strata": _strata_counts(rows),
        "bootstrap_D": _bootstrap(rows, "D", seed=seed, reps=reps),
        "bootstrap_D_minus_control": _bootstrap(rows, "D_minus_control",
                                                seed=seed, reps=reps),
        "bootstrap_D_minus_block_control": _bootstrap(
            rows, "D_minus_block_control", seed=seed, reps=reps),
    }


def _strata_counts(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for name in row.get("strata", ()):
            counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def summarise_by_venue(rows: list[dict], *, seed: int = BUILDER_SEED,
                       reps: int = 2000) -> dict[str, dict]:
    """One summary per venue.  Venues are never pooled (§4.2)."""
    venues = sorted({row["venue"] for row in rows})
    return {
        venue: summarise_scores([row for row in rows if row["venue"] == venue],
                                seed=seed, reps=reps)
        for venue in venues
    }


__all__ = [
    "AXIS_DIAGNOSTICS",
    "BUILDER_SEED",
    "CASE_SCHEMA",
    "CO_PRIMARY_VENUES",
    "DEFAULT_CASES",
    "DONOR_KINDS",
    "FOUR_WAY",
    "MANIFEST_SCHEMA",
    "VENUES",
    "ContextCase",
    "Intervention",
    "MatchFeatures",
    "byte_distance",
    "cases_from_manifest",
    "components",
    "field_map",
    "first_instruction_bytes",
    "first_token_divergence",
    "legal_first_bytes",
    "load_manifest",
    "manifest_dict",
    "prefix_is_live",
    "score_cases",
    "score_spans",
    "signature",
    "summarise_by_venue",
    "summarise_scores",
    "target_is_legal",
    "validate_cases",
    "write_manifest",
]
