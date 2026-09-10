"""The no-feedback model's behaviour, frozen before Direction 3 touches it.

**Why this exists at all.** F1 has to prove that adding a feedback path changed
nothing about the path that was already there.  A test written *after* the change
cannot prove that: it compares the new code against the new code's own idea of
what the old code did.  The only version of the claim with any content is a
comparison against bytes captured beforehand, which is what this Module captures
and what `docs/feedback-protocol-v0.json` hashes.  Direction 2 paid the same
provenance cost one stage later and had to repair a freeze afterwards
(`docs/context-audit.md`); Direction 3 pays it first.

**Weights are stored, not a recipe.** `torch.manual_seed(0)` plus a constructor
is not a fixture: it re-derives the weights from an RNG whose stream is a torch
implementation detail, so a torch upgrade would silently change what "the old
model" means and every equivalence test would keep passing.  The captured
`state` is the authority; the seed is recorded as provenance only.

**Serialisation is version-independent on purpose.** Every tensor is stored as
its raw little-endian buffer in base64 with an explicit dtype and shape, inside
one canonical JSON body with a SHA-256 over it.  `torch.save` would have been
one line and would have made the frozen artifact a pickle whose bytes depend on
the torch version that wrote it -- so the hash would move without the *values*
moving, which is the opposite of what a freeze is for.

**What the cells cover.** Five, chosen so that each F1 compatibility obligation
in `docs/directions.md` §3.2 has something to fail against:

| cell | what it pins |
|---|---|
| `byte_standard` | stride-1 logits, prompted decode, monitor, variates, stopped rows |
| `bit_standard` | stride-8 halting and the 8x longer symbol stream |
| `byte_conditional_abspos` | the two optional additive tables, which §3.1 puts *after* fusion |
| `byte_shared_init` / `bit_shared_init` | common random numbers across vocabularies |

The two shared-init cells store a digest of their weights rather than the
weights: what they have to prove is that F1's change to
`share_non_embedding_init` -- which must start drawing the feedback matrices too
-- does not perturb a single existing parameter, and a digest says that exactly.
The logits are kept in full because that is where a diagnosis would start.
"""

from __future__ import annotations

import base64
import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..isa.asm import assemble
from ..isa.codec import CODECS
from ..models.transformer import Config, DrawingLM
from .feedback_contract import (
    CONFIG_FIELD,
    DEFAULT_FEEDBACK_SCHEMA,
    FIXTURE_PATH_STR,
    FIXTURE_SCHEMA,
    PROTOCOL,
)
from .provenance import sha256_file
from .reports import staged_path

FIXTURE_PATH = Path(FIXTURE_PATH_STR)

#: Source files whose behaviour the fixture is a photograph of.  Recorded as
#: provenance, *not* as a runtime gate: F1 changes `transformer.py` by design, so
#: a check that refused to run under a changed digest would refuse to run at
#: exactly the moment the fixture becomes useful.
BASELINE_SOURCES: tuple[str, ...] = (
    "dm/models/transformer.py",
    "dm/isa/codec.py",
)

#: Deliberately tiny, and deliberately not the pilot shape.  The fixture's job is
#: bit-exactness of a code path, not statistical realism, and a 128-wide state
#: dictionary would put ~3 MB of base64 in the repository to prove the same
#: thing.  `d_model=16, n_heads=2` still gives RoPE four frequency pairs, two
#: real blocks, a tied head and a KV cache with a prefill boundary in it.
TINY: dict[str, int] = {"d_model": 16, "n_layers": 2, "n_heads": 2, "max_len": 64}

#: Fixed in-language inputs rather than random ids: a stream that parses is what
#: exercises the halt monitor, and `assemble` states the program in one line
#: instead of hiding it in a corpus generator whose defaults can move.
PROGRAMS: tuple[str, ...] = (
    "MOVE 3 4\nLINE 9 9\nHALT",
    "MOVE 1 2\nLINE 5 6\nHALT",
)

#: Bytecode bytes of the prompt handed to the prompted-decode cells.  Two whole
#: instructions, so the monitor is primed mid-program with nothing owed, and the
#: bit arm's prompt stays a whole number of bytecode bytes.
PROMPT_BYTES = 6
MAX_NEW_BYTES = 6


@dataclass(frozen=True)
class CellSpec:
    """One frozen recipe. Everything a capture needs and nothing a model chooses."""

    name: str
    codec: str
    seed: int = 0
    share_init: int | None = None
    n_classes: int = 0
    abs_pos: bool = False
    #: Store the full state dictionary, or only its digest. Digest-only cells
    #: exist to prove that a parameter set did not move, which needs no diff.
    store_weights: bool = True
    generate: bool = True
    classes: tuple[int, ...] = ()


CELL_SPECS: tuple[CellSpec, ...] = (
    CellSpec("byte_standard", codec="byte"),
    CellSpec("bit_standard", codec="bit"),
    CellSpec("byte_conditional_abspos", codec="byte", n_classes=3, abs_pos=True,
             classes=(0, 2)),
    CellSpec("byte_shared_init", codec="byte", share_init=0, store_weights=False,
             generate=False),
    CellSpec("bit_shared_init", codec="bit", share_init=0, store_weights=False,
             generate=False),
)


# ---------------------------------------------------------------------------
# tensor serialisation


def encode_tensor(tensor: torch.Tensor) -> dict:
    """A tensor as dtype, shape and its raw little-endian buffer in base64."""
    array = tensor.detach().cpu().contiguous().numpy()
    if array.dtype.byteorder not in ("=", "|", "<"):  # pragma: no cover - no BE host
        array = array.astype(array.dtype.newbyteorder("<"))
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "b64": base64.b64encode(array.tobytes()).decode("ascii"),
    }


def decode_tensor(value: dict) -> torch.Tensor:
    """Inverse of `encode_tensor`, refusing a payload of the wrong length."""
    dtype = np.dtype(value["dtype"]).newbyteorder("<")
    raw = base64.b64decode(value["b64"])
    expected = int(np.prod(value["shape"])) if value["shape"] else 1
    array = np.frombuffer(raw, dtype=dtype)
    if array.size != expected:
        raise ValueError(
            f"fixture tensor holds {array.size} elements, shape {value['shape']} "
            f"needs {expected}"
        )
    return torch.from_numpy(array.reshape(value["shape"]).copy())


def _state_digest(state: dict[str, torch.Tensor]) -> str:
    """Content hash of a state dictionary, in sorted name order.

    Names and dtypes go into the hash as well as the buffers, so a renamed or
    retyped parameter is a mismatch rather than a silent pass.
    """
    hasher = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        hasher.update(name.encode())
        hasher.update(str(tensor.dtype).encode())
        hasher.update(str(tuple(tensor.shape)).encode())
        hasher.update(tensor.numpy().tobytes())
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# capture


def build_model(spec: CellSpec) -> tuple[DrawingLM, Config]:
    """The cell's model, built exactly the way `dm.train.train` builds one."""
    codec = CODECS[spec.codec]
    cfg = Config(
        vocab_size=codec.vocab_size, n_classes=spec.n_classes,
        abs_pos=spec.abs_pos, d_model=TINY["d_model"], n_layers=TINY["n_layers"],
        n_heads=TINY["n_heads"], max_len=TINY["max_len"],
    )
    torch.manual_seed(spec.seed)
    model = DrawingLM(cfg)
    if spec.share_init is not None:
        model.share_non_embedding_init(spec.share_init)
    return model.eval(), cfg


def _inputs(spec: CellSpec) -> torch.Tensor:
    codec = CODECS[spec.codec]
    rows = [codec.with_bos(assemble(text)) for text in PROGRAMS]
    width = min(len(row) for row in rows)
    return torch.tensor([row[:width] for row in rows], dtype=torch.long)


def _prompt(spec: CellSpec) -> torch.Tensor:
    codec = CODECS[spec.codec]
    rows = [codec.encode(assemble(text)[:PROMPT_BYTES]) for text in PROGRAMS]
    return torch.tensor(rows, dtype=torch.long)


def _variates(rows: int, width: int, seed: int) -> torch.Tensor:
    """Pre-generated uniforms, so the decode consumes a stream this file owns.

    `torch.multinomial` would make the fixture depend on the sampler's RNG
    consumption pattern, which is the thing Direction 2 discovered changes when a
    mask changes -- and a fixture that moves when the sampler is refactored
    cannot certify anything about the model.
    """
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(rows, width, generator=generator, dtype=torch.float32)


@torch.no_grad()
def capture_cell(spec: CellSpec) -> dict:
    model, cfg = build_model(spec)
    state = model.state_dict()
    idx = _inputs(spec)
    classes = (torch.tensor(spec.classes, dtype=torch.long)
               if spec.n_classes else None)
    cell: dict = {
        "name": spec.name,
        "spec": asdict(spec),
        # The config dictionary an old checkpoint carries: no `feedback_schema`
        # key at all, which is precisely the shape F1 must load as `none`.
        "config": asdict(cfg),
        "n_params": model.n_params(),
        "analytic_n_params": cfg.n_params(),
        "state_sha256": _state_digest(state),
        "inputs": encode_tensor(idx),
        "logits": encode_tensor(model(idx, classes) if classes is not None
                                else model(idx)),
    }
    if classes is not None:
        cell["classes"] = encode_tensor(classes)
    if spec.store_weights:
        cell["state"] = {name: encode_tensor(tensor)
                         for name, tensor in sorted(state.items())}
    if spec.generate:
        codec = CODECS[spec.codec]
        prompt = _prompt(spec)
        max_new = MAX_NEW_BYTES * codec.stride
        variates = _variates(prompt.shape[0], prompt.shape[1] + max_new,
                             seed=spec.seed + 1)
        cell["generation"] = {
            "prompt": encode_tensor(prompt),
            "max_new": max_new,
            "variates": encode_tensor(variates),
            "top_k": 40,
            "temperature": 1.0,
            # Monitored: halting is what makes rows stop at different steps, and
            # stopped-row semantics are an F1 obligation.
            "monitored": encode_tensor(model.generate(
                prompt.shape[0], max_new=max_new, prompt=prompt,
                monitor=codec.halt_monitor(prompt.shape[0]),
                variates=variates, top_k=40,
                classes=classes,
            )),
            # Unmonitored and greedy: a fixed-width block with no early return,
            # so a change in halting cannot hide a change in the logits.
            "greedy": encode_tensor(model.generate(
                prompt.shape[0], max_new=max_new, prompt=prompt, monitor=None,
                top_k=1, classes=classes,
            )),
        }
    return cell


def environment() -> dict:
    """Enough to tell a real regression from a toolchain change.

    Recorded rather than checked.  A fixture mismatch under a moved torch version
    is still a failure -- it means the frozen behaviour is not reproducible here
    -- but the report has to say which of the two happened, because the repairs
    are completely different.
    """
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "device": "cpu",
    }


def fixture_dict(specs: tuple[CellSpec, ...] = CELL_SPECS) -> dict:
    body = {
        "schema": FIXTURE_SCHEMA,
        "status": "frozen",
        "protocol": PROTOCOL,
        "feedback_schema": None,
        "note": "captured from the no-feedback model before any Direction 3 code "
                "existed; `feedback_schema` is null because the field does not "
                "exist yet, which is what an old checkpoint looks like",
        "environment": environment(),
        "baseline_source_sha256": {
            path: sha256_file(Path(path)) for path in BASELINE_SOURCES
        },
        "programs": list(PROGRAMS),
        "tiny": dict(TINY),
        "cells": [capture_cell(spec) for spec in specs],
    }
    body["fixture_sha256"] = _digest_of(body)
    return body


def _digest_of(body: dict) -> str:
    payload = {key: value for key, value in body.items() if key != "fixture_sha256"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_fixture(path: Path, fixture: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = staged_path(path)
    staged.write_text(json.dumps(fixture, indent=1, sort_keys=True) + "\n")
    staged.replace(path)


def load_fixture(path: Path = FIXTURE_PATH) -> dict:
    """Read the fixture, refusing a payload whose digest or schema disagrees."""
    fixture = json.loads(path.read_text())
    recorded = fixture.get("fixture_sha256")
    actual = _digest_of(fixture)
    if recorded != actual:
        raise ValueError(
            f"baseline fixture hash mismatch for {path}: recorded {recorded!r}, "
            f"computed {actual!r}"
        )
    if fixture.get("schema") != FIXTURE_SCHEMA:
        raise ValueError(
            f"unsupported baseline fixture schema {fixture.get('schema')!r}; "
            f"expected {FIXTURE_SCHEMA}"
        )
    if fixture.get("status") != "frozen":
        raise ValueError(f"baseline fixture is not frozen: {path}")
    if not fixture.get("cells"):
        raise ValueError(f"baseline fixture has no cells: {path}")
    return fixture


# ---------------------------------------------------------------------------
# verification


@dataclass
class CellVerdict:
    """Per-cell result of replaying a fixture against the current code."""

    name: str
    ok: bool
    mismatches: list[str] = field(default_factory=list)


def _compare(name: str, expected: torch.Tensor, actual: torch.Tensor,
             out: list[str]) -> None:
    """Exact equality, with the worst deviation reported when it fails.

    Bit-for-bit, not `allclose`.  §3.2 requirement 3 is that the `none` path is
    *unchanged*, and a tolerance turns that into "changed by less than my
    tolerance", which is a different and much weaker claim -- and one that
    accumulates silently across F1 through F8.
    """
    if expected.shape != actual.shape:
        out.append(f"{name}: shape {tuple(actual.shape)} != {tuple(expected.shape)}")
        return
    if torch.equal(expected, actual):
        return
    if expected.is_floating_point():
        worst = float((expected - actual).abs().max())
        count = int((expected != actual).sum())
        out.append(f"{name}: {count} of {expected.numel()} values differ, "
                   f"max |delta| {worst:.3e}")
    else:
        count = int((expected != actual).sum())
        out.append(f"{name}: {count} of {expected.numel()} values differ")


#: Config fields that did not exist when the baseline was captured, with the
#: value a rebuilt config is allowed to have for them.
#:
#: **A frozen config is a record of the fields that existed, not a promise that
#: none will ever be added.** `Config` grows: `feedback_schema` arrived at F1 and
#: `asdict` has carried it since. Comparing the two dictionaries for equality
#: would make the fixture fail for a schema change rather than for a behaviour
#: change, which is the opposite of what it is for.
#:
#: So the tolerance is narrow and explicit. An extra field is accepted only if it
#: is named here *and* holds the declared value -- which is the same statement as
#: §3.2 requirement 1 read from the other side: an old config dictionary omits
#: the field and rebuilds as `none`. An extra field that is not named here is a
#: mismatch, so a future addition has to be declared before a fixture will
#: tolerate it. The value comes from `feedback_contract`, whose payload is
#: hashed, so moving the default breaks the protocol digest too.
BASELINE_COMPATIBLE_DEFAULTS: dict[str, object] = {
    CONFIG_FIELD: DEFAULT_FEEDBACK_SCHEMA,
    # R3 appends an optional architecture field.  A pre-R3 state owns no head,
    # so only the parameter-free default reconstructs it strictly.
    "relation_schema": "none",
}


def _config_mismatches(actual: dict, frozen: dict) -> list[str]:
    out = []
    for key in sorted(frozen):
        if key not in actual:
            out.append(f"config lost field {key!r}, frozen at {frozen[key]!r}")
        elif actual[key] != frozen[key]:
            out.append(f"config {key}={actual[key]!r} != frozen {frozen[key]!r}")
    for key in sorted(set(actual) - set(frozen)):
        if key not in BASELINE_COMPATIBLE_DEFAULTS:
            out.append(
                f"config gained undeclared field {key!r}={actual[key]!r}; add it "
                "to BASELINE_COMPATIBLE_DEFAULTS with the value an old checkpoint "
                "must rebuild as"
            )
        elif actual[key] != BASELINE_COMPATIBLE_DEFAULTS[key]:
            out.append(
                f"config {key}={actual[key]!r}, but an old checkpoint must rebuild "
                f"as {BASELINE_COMPATIBLE_DEFAULTS[key]!r}"
            )
    return out


@torch.no_grad()
def verify_cell(cell: dict) -> CellVerdict:
    """Replay one cell against today's code and report every disagreement.

    Every mismatch is collected rather than raised at the first one: when a real
    regression lands, "the logits and the greedy decode moved but the state
    dictionary did not" is a diagnosis and "the logits moved" is a starting
    point.
    """
    spec = CellSpec(**cell["spec"])
    mismatches: list[str] = []
    model, cfg = build_model(spec)

    mismatches += _config_mismatches(asdict(cfg), cell["config"])
    if model.n_params() != cell["n_params"]:
        mismatches.append(
            f"n_params {model.n_params()} != frozen {cell['n_params']}"
        )
    if cfg.n_params() != cell["analytic_n_params"]:
        mismatches.append(
            f"analytic n_params {cfg.n_params()} != frozen "
            f"{cell['analytic_n_params']}"
        )

    state = model.state_dict()
    if "state" in cell:
        # Load the frozen weights rather than trusting the rebuilt ones: the
        # fixture's authority is its bytes, and `load_state_dict(strict=True)` is
        # itself one of the things F1 must not break.
        frozen = {name: decode_tensor(value)
                  for name, value in cell["state"].items()}
        model.load_state_dict(frozen, strict=True)
        state = model.state_dict()
    if _state_digest(state) != cell["state_sha256"]:
        mismatches.append(
            f"state digest {_state_digest(state)} != frozen {cell['state_sha256']}"
        )

    idx = decode_tensor(cell["inputs"])
    classes = decode_tensor(cell["classes"]) if "classes" in cell else None
    logits = model(idx, classes) if classes is not None else model(idx)
    _compare("logits", decode_tensor(cell["logits"]), logits, mismatches)

    generation = cell.get("generation")
    if generation is not None:
        codec = CODECS[spec.codec]
        prompt = decode_tensor(generation["prompt"])
        variates = decode_tensor(generation["variates"])
        _compare("generation.monitored", decode_tensor(generation["monitored"]),
                 model.generate(prompt.shape[0], max_new=generation["max_new"],
                                prompt=prompt,
                                monitor=codec.halt_monitor(prompt.shape[0]),
                                variates=variates, top_k=generation["top_k"],
                                classes=classes),
                 mismatches)
        _compare("generation.greedy", decode_tensor(generation["greedy"]),
                 model.generate(prompt.shape[0], max_new=generation["max_new"],
                                prompt=prompt, monitor=None, top_k=1,
                                classes=classes),
                 mismatches)
    return CellVerdict(name=cell["name"], ok=not mismatches, mismatches=mismatches)


def verify_fixture(fixture: dict) -> list[CellVerdict]:
    return [verify_cell(cell) for cell in fixture["cells"]]


def describe_failure(fixture: dict, verdicts: list[CellVerdict]) -> str:
    """A failure message that names the likely cause instead of only the symptom."""
    frozen_env = fixture.get("environment", {})
    drift = [
        f"{key}: frozen {frozen_env.get(key)!r}, now {value!r}"
        for key, value in environment().items()
        if key in ("torch", "numpy") and frozen_env.get(key) != value
    ]
    lines = ["the no-feedback path no longer reproduces its frozen baseline:"]
    for verdict in verdicts:
        if verdict.ok:
            continue
        lines.append(f"  {verdict.name}:")
        lines += [f"    {item}" for item in verdict.mismatches]
    if drift:
        lines.append("  toolchain moved since the freeze, which is a different "
                     "repair from a code regression:")
        lines += [f"    {item}" for item in drift]
    return "\n".join(lines)


__all__ = [
    "BASELINE_SOURCES",
    "CELL_SPECS",
    "FIXTURE_PATH",
    "MAX_NEW_BYTES",
    "PROGRAMS",
    "PROMPT_BYTES",
    "TINY",
    "CellSpec",
    "CellVerdict",
    "build_model",
    "capture_cell",
    "decode_tensor",
    "describe_failure",
    "encode_tensor",
    "environment",
    "fixture_dict",
    "load_fixture",
    "verify_cell",
    "verify_fixture",
    "write_fixture",
]
