"""Transactional, exact execution of a target-free ``COPY`` action."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..data.augment import Affine, apply
from ..isa.spec import ISAError, Kind, Op, Tier, spec_for
from ..isa.transform import D4, Transform
from .spec import PREDICTED_SUPPORT, CopyActionKey, SupportPolicy


class FaultCode(str, Enum):
    """Stable, caller-actionable reasons an action was refused."""

    MALFORMED_PREFIX = "malformed_prefix"
    ACTION_TYPE = "action_type"
    NON_FLAT_PREFIX = "non_flat_prefix"
    BOUNDARY = "boundary"
    SOURCE_RANGE = "source_range"
    SOURCE_EMPTY = "source_empty"
    SOURCE_BYTES = "source_bytes"
    SOURCE_INSTRUCTIONS = "source_instructions"
    SOURCE_GAP = "source_gap"
    SOURCE_COORDINATES = "source_coordinates"
    HALT_SOURCE = "halt_source"
    UNSUPPORTED_D4 = "unsupported_d4"
    UNSUPPORTED_TRANSLATION = "unsupported_translation"
    UNSUPPORTED_COUNT = "unsupported_count"
    CANVAS = "canvas"
    MAX_LENGTH = "max_length"


class TransducerFault(ValueError):
    """A typed refusal.  Callers must surface it; they may not fall back to EMIT."""

    def __init__(self, code: FaultCode, detail: str) -> None:
        super().__init__(f"{code.value}: {detail}")
        self.code = code


@dataclass(frozen=True)
class CopyExecution:
    """The complete block derived from one validated key and its prefix."""

    action: CopyActionKey
    appended: bytes

    @property
    def copied_images(self) -> int:
        return self.action.total_count - 1

    @property
    def source_bytes(self) -> int:
        return self.action.source_stop - self.action.source_start


@dataclass(frozen=True)
class PrefixProgress:
    """One shared L0 parser reading, including a legitimate partial tail."""

    boundaries: tuple[int, ...]
    coordinates: tuple[int, ...]
    opcodes: tuple[Op, ...]
    complete: bool
    incomplete_at: int | None = None
    incomplete_mnemonic: str | None = None


def prefix_progress(prefix: bytes) -> PrefixProgress:
    """Parse a flat prefix without misclassifying an incomplete final opcode.

    Cached generation observes byte prefixes between operands.  A partial final
    instruction is not a candidate-query boundary, but nor is it a structural
    error.  Unknown/non-L0 instructions remain typed faults immediately.
    """
    if not isinstance(prefix, bytes):
        raise TransducerFault(FaultCode.MALFORMED_PREFIX, "prefix is not immutable bytes")
    marks = [0]
    coords: list[int] = []
    ops: list[Op] = []
    pc = 0
    try:
        while pc < len(prefix):
            spec = spec_for(prefix[pc])
            end = pc + spec.size
            if end > len(prefix):
                return PrefixProgress(tuple(marks), tuple(coords), tuple(ops), False,
                                      pc, spec.mnemonic)
            if spec.tier is not Tier.L0:
                raise TransducerFault(FaultCode.NON_FLAT_PREFIX,
                                      f"{spec.mnemonic} at {pc}")
            index = 0
            while index < len(spec.operands):
                if spec.operands[index] is Kind.COORD:
                    coords.append(pc + 1 + index)
                    index += 2
                else:
                    index += 1
            ops.append(spec.op)
            pc = end
            marks.append(pc)
    except ISAError as exc:
        raise TransducerFault(FaultCode.MALFORMED_PREFIX, str(exc)) from exc
    return PrefixProgress(tuple(marks), tuple(coords), tuple(ops), True)


def prefix_index(prefix: bytes) -> tuple[tuple[int, ...], tuple[int, ...], tuple[Op, ...]]:
    """Canonical boundaries, coordinate-pair starts and opcodes of a complete prefix."""
    progress = prefix_progress(prefix)
    if not progress.complete:
        raise TransducerFault(FaultCode.MALFORMED_PREFIX,
                              f"truncated {progress.incomplete_mnemonic} at {progress.incomplete_at}")
    return progress.boundaries, progress.coordinates, progress.opcodes


def validate_prefix(prefix: bytes) -> None:
    """Fail closed on malformed or non-L0 prefix structure before search.

    This is deliberately separate from action validation: a decoder may decide
    EMIT or may have no candidates, but neither path licenses it to overlook an
    invalid existing byte prefix.  Candidate-dependent checks remain in
    :func:`execute_copy`.
    """
    if not isinstance(prefix, bytes):
        raise TransducerFault(FaultCode.MALFORMED_PREFIX, "prefix is not immutable bytes")
    prefix_index(prefix)


def _validate_action(action: CopyActionKey) -> None:
    """Refuse malformed public action values before accessing their fields."""
    if not isinstance(action, CopyActionKey):
        raise TransducerFault(FaultCode.ACTION_TYPE, "action is not a CopyActionKey")
    for name in ("boundary", "source_start", "source_stop", "total_count"):
        value = getattr(action, name)
        if type(value) is not int:  # bool is intentionally not an integer here.
            raise TransducerFault(FaultCode.ACTION_TYPE, f"{name} is not an integer")
    if not isinstance(action.step, Transform):
        raise TransducerFault(FaultCode.ACTION_TYPE, "step is not a Transform")
    if not isinstance(action.step.d4, D4):
        raise TransducerFault(FaultCode.ACTION_TYPE, "step.d4 is not a D4")
    if type(action.step.d4.turns) is not int or type(action.step.d4.mirror) is not bool:
        raise TransducerFault(FaultCode.ACTION_TYPE, "step.d4 has invalid fields")
    for name in ("dx", "dy"):
        if type(getattr(action.step, name)) is not int:
            raise TransducerFault(FaultCode.ACTION_TYPE, f"step.{name} is not an integer")


def _validate(prefix: bytes, action: CopyActionKey, policy: SupportPolicy,
              max_len: int) -> tuple[int, ...]:
    validate_prefix(prefix)
    _validate_action(action)
    if max_len < 0:
        raise ValueError("max_len must be non-negative")
    if action.boundary != len(prefix):
        raise TransducerFault(FaultCode.BOUNDARY,
                              f"action boundary {action.boundary} is not prefix end {len(prefix)}")
    marks, coords, ops = prefix_index(prefix)
    if action.boundary not in marks:
        raise TransducerFault(FaultCode.BOUNDARY, f"{action.boundary} is not an instruction boundary")
    if not (0 <= action.source_start < action.source_stop <= action.boundary):
        raise TransducerFault(FaultCode.SOURCE_RANGE, "source is not a non-empty earlier prefix span")
    if action.source_start not in marks or action.source_stop not in marks:
        raise TransducerFault(FaultCode.BOUNDARY, "source endpoint is not an instruction boundary")

    bounds = policy.bounds
    start_index, stop_index, boundary_index = (marks.index(action.source_start),
                                                marks.index(action.source_stop),
                                                marks.index(action.boundary))
    source_bytes = action.source_stop - action.source_start
    source_instructions = stop_index - start_index
    gap = boundary_index - stop_index
    if source_instructions < bounds.min_source_instructions:
        raise TransducerFault(FaultCode.SOURCE_INSTRUCTIONS, "source has too few instructions")
    if source_instructions > bounds.max_source_instructions:
        raise TransducerFault(FaultCode.SOURCE_INSTRUCTIONS, "source exceeds instruction cap")
    if source_bytes < bounds.min_source_bytes or source_bytes > bounds.max_source_bytes:
        raise TransducerFault(FaultCode.SOURCE_BYTES, "source violates byte-size bounds")
    if gap > bounds.max_source_gap_instructions:
        raise TransducerFault(FaultCode.SOURCE_GAP, "source stop is too far before boundary")
    if sum(action.source_start <= at < action.source_stop for at in coords) < bounds.min_source_coords:
        raise TransducerFault(FaultCode.SOURCE_COORDINATES, "source has no coordinate pair")
    if Op.HALT in ops[start_index:stop_index]:
        raise TransducerFault(FaultCode.HALT_SOURCE, "source contains HALT")
    if action.step.d4.code not in policy.d4_codes:
        raise TransducerFault(FaultCode.UNSUPPORTED_D4, str(action.step.d4.code))
    if action.step.dx not in policy.translations or action.step.dy not in policy.translations:
        raise TransducerFault(FaultCode.UNSUPPORTED_TRANSLATION,
                              f"({action.step.dx}, {action.step.dy})")
    if action.total_count not in policy.total_counts:
        raise TransducerFault(FaultCode.UNSUPPORTED_COUNT, str(action.total_count))
    if len(prefix) + (action.total_count - 1) * source_bytes > max_len:
        raise TransducerFault(
            FaultCode.MAX_LENGTH,
            f"{len(prefix)} + {(action.total_count - 1) * source_bytes} exceeds {max_len}",
        )
    return marks


def execute_copy(prefix: bytes, action: CopyActionKey, *,
                 policy: SupportPolicy = PREDICTED_SUPPORT,
                 max_len: int) -> CopyExecution:
    """Validate then derive every image of ``action`` from ``prefix``.

    Validation and the arithmetic full max-length preflight happen before any
    affine rewrite. Since the input is immutable and the output is local, a
    refusal has no partial-output side effect.
    """
    _validate(prefix, action, policy, max_len)
    source = prefix[action.source_start:action.source_stop]
    images: list[bytes] = []
    for power in range(1, action.total_count):
        image = apply(source, Affine.of(action.step.power(power)))
        if image is None:
            raise TransducerFault(FaultCode.CANVAS,
                                  f"image {power} leaves the canvas")
        images.append(image)
    appended = b"".join(images)
    return CopyExecution(action=action, appended=appended)
