"""R0: what the measurement platform has to prove before a scientific cell runs.

Direction 3 ended with a reproducibility verdict that had to be retracted. The
comparator was the SHA-256 of two `torch.save` files, and `torch.save` writes a
ZIP whose member names depend on the output filename; the two checkpoints also
carried a replicate-specific `record_name` inside the same bytes. So the digest
identified an *artifact* and was quoted as a statement about learned state
(`PLAN.md` §2). A post-closure tensor-content audit did find real divergence, but
the published comparator never could have shown it either way.

Direction 4 inherits the lesson and not the comparator. This Module owns three
canonical content digests -- learned state, optimizer state and RNG state --
computed from the tensors themselves, plus the fixed sentinel that produces them
and the fail-closed rule that reads them. Every function returns both sides of a
comparison and the difference between them, and leaves the verdict to `qualify`:
a digest a program computes and immediately compares against itself proves
nothing.

**What R0 can and cannot certify.** It can show that repeating one fixed short
training run on one backend reproduces state exactly, or measure how far it does
not. It cannot certify a backend in general, and
`torch.use_deterministic_algorithms(True)` records *requested policy* rather than
granting it -- some operators have no deterministic kernel and raise at the
operator instead of here. So the artifact is "determinism was requested, here is
the state that resulted, here is what repeating it produced", and the rule reads
the third of those.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from ..models.transformer import Config, DrawingLM
from .relation_contract import (
    R0_DIGESTS,
    R0_ENVIRONMENT_FIELDS,
    R0_FALLBACK_STATUS,
    R0_FORBIDDEN_DIGEST,
    R0_MIN_REPEATS,
    R0_READING_FIELDS,
    R0_RESUME_DIGEST_FIELDS,
    R0_RESUME_FIELDS,
    R0_RESUME_STEP,
    R0_SENTINEL_CONFIG,
    seed_for,
)

#: Framed into every digest so a payload from one kind of state can never be
#: mistaken for another, and so a future change to the framing is a visible
#: schema break rather than a silent hash move.
STATE_MARKER = b"drawing-machine-relation-state-v1"
OPTIMIZER_MARKER = b"drawing-machine-relation-optimizer-v1"
RNG_MARKER = b"drawing-machine-relation-rng-v1"

# The raw-reading and resume binding fields are part of the report interface.
# Bump this when their shape changes so an old artifact cannot be qualified by a
# new validator through a permissive default.
EVIDENCE_SCHEMA = 2


def _framed(marker: bytes) -> tuple[object, Callable[[bytes], None]]:
    digest = hashlib.sha256()

    def frame(payload: bytes) -> None:
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    frame(marker)
    return digest, frame


def _frame_tensor(frame: Callable[[bytes], None], name: str,
                  tensor: Tensor) -> None:
    """Name, dtype, shape and raw contiguous CPU bytes -- in that order.

    Raw bytes rather than a float repr: two tensors that print identically at six
    significant figures are not the same tensor, and the whole question R0 asks
    is whether two runs produced the same numbers. `view(torch.uint8)` is exact
    for every floating and integer dtype torch stores strided.
    """
    if tensor.layout != torch.strided:
        raise ValueError(f"state entry {name!r} has unsupported layout {tensor.layout}")
    value = tensor.detach().cpu().contiguous()
    frame(name.encode("utf-8"))
    frame(str(value.dtype).encode("ascii"))
    frame(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    # Flattened before the byte view: AdamW's `step` is a *scalar* tensor, and
    # `view(torch.uint8)` refuses a zero-dimensional reinterpretation. The shape
    # is framed separately, so flattening loses nothing.
    flat = value.reshape(-1)
    frame(flat.view(torch.uint8).numpy().tobytes() if flat.numel() else b"")


def state_digest(state: Mapping[str, Tensor]) -> str:
    """Content digest of a model's learned state, container and metadata excluded.

    This is deliberately **not** the checkpoint file's SHA-256, which the contract
    names `R0_FORBIDDEN_DIGEST` for a reason (`docs/copy-relation.md` §6). A file
    hash answers "are these the same bytes on disk"; this answers "did the two
    runs learn the same numbers", and only the second is a reproducibility claim.
    """
    digest, frame = _framed(STATE_MARKER)
    for name in sorted(state):
        _frame_tensor(frame, name, state[name])
    return digest.hexdigest()


def optimizer_digest(optimizer: torch.optim.Optimizer,
                     model: torch.nn.Module) -> str:
    """Content digest of optimizer state, keyed by **parameter name**.

    `torch.optim` keys its state by parameter *object*, and `state_dict()` keys it
    by position in `param_groups`. Both are stable within a process and neither is
    a name, so a digest built from either would move whenever a module's
    registration order changed -- which is exactly what Direction 4 does when it
    appends a relation block. Keying by name makes the digest a statement about
    the optimizer's contents and not about the module's layout.

    **Group membership is framed, not just group settings.** Two optimizers whose
    moments and hyperparameters agree while a parameter sits in the other group
    train differently: the parameter takes the other group's learning rate, weight
    decay and schedule. Framing the settings alone made those two optimizers
    digest identically, which is the same class of fault as a corpus clause that
    compares a manifest to itself.
    """
    named = {parameter: name for name, parameter in model.named_parameters()}
    digest, frame = _framed(OPTIMIZER_MARKER)
    for index, group in enumerate(optimizer.param_groups):
        settings = {key: value for key, value in group.items() if key != "params"}
        frame(f"group{index}".encode())
        frame(json.dumps(settings, sort_keys=True, separators=(",", ":"),
                         default=str).encode())
        members = sorted(named.get(parameter, "<foreign>")
                         for parameter in group["params"])
        frame(json.dumps(members, separators=(",", ":")).encode())
    entries: list[tuple[str, dict]] = []
    for parameter, state in optimizer.state.items():
        name = named.get(parameter)
        if name is None:  # pragma: no cover -- an optimizer over a foreign tensor
            raise ValueError("optimizer holds state for a tensor the model does not own")
        entries.append((name, state))
    for name, state in sorted(entries):
        frame(name.encode("utf-8"))
        for key in sorted(state):
            value = state[key]
            if isinstance(value, Tensor):
                _frame_tensor(frame, f"{name}.{key}", value)
            else:
                frame(f"{key}={value!r}".encode())
    return digest.hexdigest()


def rng_digest(device: str | torch.device = "cpu",
               generator: torch.Generator | None = None) -> str:
    """Content digest of every generator a training step can consume.

    Five streams when there are five: python, numpy, torch's global CPU stream,
    the accelerator's stream when the device has one, and **the private
    `torch.Generator` the caller draws its data from**. That last one is the one
    a first version omitted, and it is the one that matters most here: every
    sentinel batch comes out of it, so a resume that restored the other four and
    forgot it would train on a different corpus order and still report four
    matching digests.

    The accelerator stream is framed only when the device actually has one. An
    earlier version framed the CPU label a second time for `device="cpu"`, which
    made a CPU digest look like it covered two streams when it covered one.
    """
    digest, frame = _framed(RNG_MARKER)
    frame(b"python")
    version, keys, gauss = random.getstate()
    frame(json.dumps([version, list(keys), gauss], separators=(",", ":")).encode())
    # Framed field by field rather than through JSON: `np.random.get_state()`
    # returns a numpy array beside numpy scalars, and a JSON encoder that has to
    # guess how to render those is one numpy release away from moving the digest
    # of an unchanged state.
    frame(b"numpy")
    name_, state_keys, position, has_gauss, cached = np.random.get_state()
    frame(str(name_).encode())
    frame(np.asarray(state_keys, dtype="<u4").tobytes())
    frame(json.dumps([int(position), int(has_gauss), float(cached)],
                     separators=(",", ":")).encode())
    frame(b"torch.cpu")
    frame(torch.random.get_rng_state().numpy().tobytes())
    name = torch.device(device).type
    if name == "cuda" and torch.cuda.is_available():
        frame(b"torch.cuda")
        frame(torch.cuda.get_rng_state(device).numpy().tobytes())
    elif name == "mps" and torch.backends.mps.is_available():
        frame(b"torch.mps")
        frame(torch.mps.get_rng_state().numpy().tobytes())
    else:
        frame(b"torch.accelerator.none")
    frame(b"data_generator")
    frame(generator.get_state().numpy().tobytes() if generator is not None
          else b"none")
    return digest.hexdigest()


def deterministic_execution() -> dict:
    """Ask torch for deterministic kernels, and report what it granted.

    Requested rather than assumed, and reported rather than asserted. The honest
    artifact is "determinism was requested, here is the state that resulted",
    checked afterwards against whether the repeats agreed.

    `warn_only=True` here, unlike Direction 3's `False`: R0's job is to *measure*
    a backend, and a backend that raises at the first non-deterministic operator
    returns no measurement at all. The granted flags travel with the reading, and
    `qualify` never treats them as a certificate.
    """
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return {
        "requested": True,
        "warn_only": True,
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "is_policy_not_certificate": True,
    }


#: Environment variables that change kernel selection or thread counts and are
#: therefore part of what a repeatability reading is a reading *of*.
RECORDED_ENVIRONMENT_VARIABLES: tuple[str, ...] = (
    "CUBLAS_WORKSPACE_CONFIG", "PYTORCH_ENABLE_MPS_FALLBACK",
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONHASHSEED",
)


def execution_environment(*, device: str | torch.device) -> dict:
    """Portable provenance for one repeatability reading."""
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "device": str(device),
        "threads": torch.get_num_threads(),
        "environment": {name: os.environ.get(name)
                        for name in RECORDED_ENVIRONMENT_VARIABLES},
    }


# ---------------------------------------------------------------------------
# the fixed sentinel


@dataclass(frozen=True)
class SentinelConfig:
    """The one short training run R0 repeats, and nothing else.

    Deliberately tiny and deliberately *fixed*. A sentinel that a caller can
    reshape is not a repeatability instrument: two readings taken at different
    shapes measure different kernels, and the whole reading is a comparison
    between repeats. The defaults are the frozen shape; the fields exist so a
    test can run something smaller, and every field travels into the report so a
    non-default reading cannot be quoted as the frozen one.

    There is **no task metric here**, and its absence is deliberate. An earlier
    version scored greedy continuations against random reference bytes; the rate
    was identically zero in every repeat, moved in steps of 1/batch, and would
    therefore have let an arbitrarily nondeterministic backend "resolve" a 0.10
    floor because every continuation missed. R0 now qualifies on state identity
    alone (`qualify`), and the loss is reported as a diagnostic that gates
    nothing.
    """

    vocab_size: int = int(R0_SENTINEL_CONFIG["vocab_size"])
    d_model: int = int(R0_SENTINEL_CONFIG["d_model"])
    n_layers: int = int(R0_SENTINEL_CONFIG["n_layers"])
    n_heads: int = int(R0_SENTINEL_CONFIG["n_heads"])
    max_len: int = int(R0_SENTINEL_CONFIG["max_len"])
    steps: int = int(R0_SENTINEL_CONFIG["steps"])
    batch: int = int(R0_SENTINEL_CONFIG["batch"])
    length: int = int(R0_SENTINEL_CONFIG["length"])
    lr: float = float(R0_SENTINEL_CONFIG["lr"])

    @property
    def is_frozen_shape(self) -> bool:
        return self == SentinelConfig()


@dataclass(frozen=True)
class SentinelReading:
    """One run of the sentinel: its digests, its diagnostic loss and what it cost."""

    model_state: str
    optimizer_state: str
    rng_state: str
    final_loss: float
    seconds: float

    def digests(self) -> dict[str, str]:
        return {"model_state": self.model_state,
                "optimizer_state": self.optimizer_state,
                "rng_state": self.rng_state}


def _build(config: SentinelConfig,
           device: str | torch.device) -> tuple[DrawingLM, torch.optim.Optimizer]:
    model = DrawingLM(Config(vocab_size=config.vocab_size, d_model=config.d_model,
                             n_layers=config.n_layers, n_heads=config.n_heads,
                             max_len=config.max_len)).to(device)
    return model, torch.optim.AdamW(model.parameters(), lr=config.lr)


def _seed_everything(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    return torch.Generator().manual_seed(seed)


def _train(model: DrawingLM, optimizer: torch.optim.Optimizer,
           generator: torch.Generator, config: SentinelConfig,
           device: str | torch.device, steps: int) -> float:
    """`steps` steps, each drawing its batch **when it runs**.

    Lazily, not from a pre-generated list, because a resume that replays a list
    built before the interruption never exercises the data generator's own state
    -- and that generator is exactly what a real restart has to carry.
    """
    model.train()
    loss = torch.tensor(float("nan"))
    for _ in range(steps):
        rows = torch.randint(0, config.vocab_size, (config.batch, config.length),
                             generator=generator).to(device)
        logits = model(rows[:, :-1])
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, config.vocab_size), rows[:, 1:].reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return float(loss.item())


def _reading(model: DrawingLM, optimizer: torch.optim.Optimizer,
             generator: torch.Generator, device: str | torch.device,
             loss: float, seconds: float) -> SentinelReading:
    return SentinelReading(
        model_state=state_digest(model.state_dict()),
        optimizer_state=optimizer_digest(optimizer, model),
        rng_state=rng_digest(device, generator),
        final_loss=loss,
        seconds=seconds,
    )


def run_sentinel(config: SentinelConfig | None = None, *,
                 device: str | torch.device = "cpu",
                 seed: int | None = None) -> SentinelReading:
    """Train the sentinel once from a fixed seed and return its canonical state.

    **Deterministic kernels are requested before anything runs.** An earlier
    version enabled the policy while assembling the report, which is after every
    repeat had finished -- so the readings were taken with the policy off and the
    artifact said it had been requested. The call is idempotent, so making it
    here means a standalone `run_sentinel` is measured under the same policy a
    full `repeatability` sweep is.

    Every generator is re-seeded here rather than inherited, so two calls in one
    process are two *repeats* and not two points along one stream.
    """
    config = config or SentinelConfig()
    deterministic_execution()
    seed = seed_for("sentinel") if seed is None else seed
    generator = _seed_everything(seed)
    started = time.perf_counter()
    model, optimizer = _build(config, device)
    loss = _train(model, optimizer, generator, config, device, config.steps)
    model.eval()
    return _reading(model, optimizer, generator, device, loss,
                    time.perf_counter() - started)


def resume_equivalence(config: SentinelConfig | None = None, *,
                       device: str | torch.device = "cpu",
                       seed: int | None = None,
                       at: int | None = None,
                       directory: Path | None = None) -> dict:
    """Whether a run interrupted, **written to disk** and restarted reaches the
    same state as one uninterrupted pass.

    **Checked separately from repeatability, because final model equality alone
    does not verify it** (`docs/copy-relation.md` §6 clause 4). A resume that
    restores weights and forgets the optimizer moments trains a different
    trajectory that can still converge to a similar model; a resume that forgets
    a generator draws a different corpus order.

    An earlier version handed an in-memory dictionary back to the *same* model
    and optimizer objects and called that a resume. It exercised nothing: no
    `torch.save`, no `torch.load`, no fresh module, no fresh optimizer, and
    neither the accelerator stream nor the private data generator was restored.
    This one serialises the whole carried state to a real file, constructs a new
    `DrawingLM` and a new `AdamW`, and restores all five generators before
    continuing. A process boundary is the one thing still missing, and it is
    named rather than implied.

    The comparison is against a single uninterrupted run of the *same* sentinel,
    computed here, so the two sides are produced by one function under one seed.
    """
    config = config or SentinelConfig()
    deterministic_execution()
    seed = seed_for("sentinel") if seed is None else seed
    at = (R0_RESUME_STEP if config.is_frozen_shape else config.steps // 2) \
        if at is None else at
    if not isinstance(at, int) or isinstance(at, bool) or not 0 < at < config.steps:
        raise ValueError(
            f"resume point {at!r} must be an integer strictly inside 0..{config.steps}")
    whole = run_sentinel(config, device=device, seed=seed)

    generator = _seed_everything(seed)
    model, optimizer = _build(config, device)
    _train(model, optimizer, generator, config, device, at)

    with tempfile.TemporaryDirectory(dir=directory) as scratch:
        checkpoint = Path(scratch) / "sentinel_resume.pt"
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.random.get_rng_state(),
                "accelerator": _accelerator_rng_state(device),
                "generator": generator.get_state(),
            },
        }, checkpoint)
        # Deleted before the reload so nothing downstream can be reading the
        # objects that produced the file rather than the file.
        del model, optimizer, generator
        carried = torch.load(checkpoint, map_location=device, weights_only=False)

    model, optimizer = _build(config, device)
    model.load_state_dict(carried["model"], strict=True)
    optimizer.load_state_dict(carried["optimizer"])
    random.setstate(carried["rng"]["python"])
    np.random.set_state(carried["rng"]["numpy"])
    torch.random.set_rng_state(carried["rng"]["torch_cpu"].cpu())
    _set_accelerator_rng_state(device, carried["rng"]["accelerator"])
    generator = torch.Generator()
    generator.set_state(carried["rng"]["generator"].cpu())

    loss = _train(model, optimizer, generator, config, device, config.steps - at)
    model.eval()
    resumed = _reading(model, optimizer, generator, device, loss, 0.0)
    differing = sorted(name for name, value in whole.digests().items()
                       if resumed.digests()[name] != value)
    return {
        "schema": EVIDENCE_SCHEMA,
        "device": str(device),
        "seed": seed,
        "sentinel": dict(vars(config)),
        "environment": execution_environment(device=device),
        "resume_step": at,
        "serialised": True,
        "rebuilt_model_and_optimizer": True,
        "process_boundary": False,
        "uninterrupted": whole.digests(),
        "resumed": resumed.digests(),
        "differing_digests": differing,
        "equivalent": not differing,
        "loss_difference": resumed.final_loss - whole.final_loss,
    }


def _accelerator_rng_state(device: str | torch.device) -> Tensor | None:
    name = torch.device(device).type
    if name == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_rng_state(device)
    if name == "mps" and torch.backends.mps.is_available():
        return torch.mps.get_rng_state()
    return None


def _set_accelerator_rng_state(device: str | torch.device,
                               state: Tensor | None) -> None:
    if state is None:
        return
    name = torch.device(device).type
    if name == "cuda" and torch.cuda.is_available():
        torch.cuda.set_rng_state(state.cpu(), device)
    elif name == "mps" and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state.cpu())


# ---------------------------------------------------------------------------
# the R0 decision


def repeatability(config: SentinelConfig | None = None, *,
                  device: str | torch.device = "cpu",
                  repeats: int = R0_MIN_REPEATS,
                  seed: int | None = None) -> dict:
    """Run the sentinel `repeats` times and describe how far the readings moved.

    Findings, not a verdict. Three digests give three exact-agreement counts;
    `qualify` decides what that means. The determinism policy is requested
    *before* the first reading and the granted flags travel with the report, so
    the artifact describes the conditions the readings were actually taken under.

    The loss spread is reported and gates nothing. It is a diagnostic: a backend
    whose digests differ has already failed, and one whose digests agree has a
    spread of exactly zero.
    """
    config = config or SentinelConfig()
    if repeats < R0_MIN_REPEATS:
        raise ValueError(
            f"R0 freezes at least {R0_MIN_REPEATS} repeats; {repeats} cannot "
            "support a repeatability statement")
    determinism = deterministic_execution()
    readings = [run_sentinel(config, device=device, seed=seed)
                for _ in range(repeats)]
    digests = {name: sorted({reading.digests()[name] for reading in readings})
               for name in R0_DIGESTS}
    losses = [reading.final_loss for reading in readings]
    return {
        "schema": EVIDENCE_SCHEMA,
        "device": str(device),
        "sentinel": dict(vars(config)),
        "repeats": repeats,
        "seed": seed_for("sentinel") if seed is None else seed,
        "readings": [vars(reading) for reading in readings],
        "distinct_digests": {name: len(values) for name, values in digests.items()},
        "digests": digests,
        "exact_state_reproduction": all(
            len(values) == 1 for values in digests.values()),
        "observed_loss_range": max(losses) - min(losses),
        "mean_seconds": sum(reading.seconds for reading in readings) / repeats,
        "determinism": determinism,
        "environment": execution_environment(device=device),
        "note": (
            "an observed range over repeats is not a variance, a bound or a "
            "device noise floor; it is what these repeats did"),
    }


def qualify(repeatability_report: dict, resume_report: dict) -> dict:
    """The fail-closed R0 rule: may a scientific cell launch on this platform?

    **One route, and it is exact state reproduction.** At least `R0_MIN_REPEATS`
    repeats of the frozen sentinel must agree on all three content digests, and
    an interrupted-and-restarted run must reach the same three.

    The metric-level fallback `docs/copy-relation.md` §6 clause 3 allows is
    **withdrawn in v1** and is not implemented here. As built it was degenerate:
    the sentinel's exact-block rate moved in steps of `1/batch = 0.125`, coarser
    than the `0.10` floor it was meant to resolve, and it read exactly zero in
    every repeat because the continuations were scored against random bytes. A
    metric that is always zero is satisfied by an arbitrarily nondeterministic
    backend. Restoring the route needs a non-degenerate paired measurement on the
    *relation* task, which cannot exist before R3; until then a backend that
    cannot reproduce state exactly fails R0.

    Fails closed on structure as well as on content: a report that claims exact
    reproduction while carrying no repeat count, no readings and no digests is
    refused, because a missing field is not a satisfied one.
    """
    problems: list[str] = []
    if not isinstance(repeatability_report, dict) or \
            repeatability_report.get("schema") != EVIDENCE_SCHEMA:
        return {
            "schema": EVIDENCE_SCHEMA, "qualified": False, "route": None,
            "problems": [
                "the repeatability report is missing or of an unknown schema"],
        }

    def required(body: Mapping, name: str, kind: type | tuple[type, ...],
                 label: str) -> object:
        if name not in body:
            problems.append(f"the {label} report is missing {name}")
            return None
        value = body[name]
        int_kind = kind is int or (isinstance(kind, tuple) and int in kind)
        if not isinstance(value, kind) or (int_kind and isinstance(value, bool)):
            problems.append(f"the {label} report field {name!r} has the wrong type")
            return None
        return value

    def exact_mapping(body: object, expected: tuple[str, ...], label: str) -> dict | None:
        if not isinstance(body, dict) or set(body) != set(expected):
            problems.append(
                f"the {label} map must carry exactly {list(expected)}")
            return None
        if any(not isinstance(value, str) or not value for value in body.values()):
            problems.append(f"the {label} map contains a missing or non-string digest")
            return None
        return body

    sentinel = required(repeatability_report, "sentinel", dict, "repeatability")
    if isinstance(sentinel, dict) and sentinel != R0_SENTINEL_CONFIG:
        problems.append(
            "the repeatability reading was taken at a sentinel shape other than "
            "the frozen one and cannot qualify a platform")
    device = required(repeatability_report, "device", str, "repeatability")
    seed = required(repeatability_report, "seed", int, "repeatability")
    if isinstance(seed, int) and seed != seed_for("sentinel"):
        problems.append(
            f"the repeatability seed is {seed}, not the frozen sentinel seed "
            f"{seed_for('sentinel')}")
    environment = required(repeatability_report, "environment", dict, "repeatability")
    if isinstance(environment, dict):
        if set(environment) != set(R0_ENVIRONMENT_FIELDS):
            problems.append("the repeatability environment schema is incomplete")
        elif environment.get("device") != device:
            problems.append("the repeatability device does not match its environment")

    repeats = required(repeatability_report, "repeats", int, "repeatability")
    if isinstance(repeats, int) and repeats < R0_MIN_REPEATS:
        problems.append(
            f"a qualification needs at least {R0_MIN_REPEATS} repeats; this reading "
            f"declares {repeats!r}")
    readings = required(repeatability_report, "readings", list, "repeatability")
    if isinstance(readings, list) and isinstance(repeats, int) \
            and len(readings) != repeats:
        problems.append(
            f"the report carries {len(readings)} readings for {repeats!r} declared repeats")

    derived: dict[str, list[str]] = {name: [] for name in R0_DIGESTS}
    valid_readings = isinstance(readings, list)
    if isinstance(readings, list):
        for index, reading in enumerate(readings):
            if not isinstance(reading, dict) or set(reading) != set(R0_READING_FIELDS):
                problems.append(
                    f"repeatability reading {index} must carry exactly "
                    f"{list(R0_READING_FIELDS)}")
                valid_readings = False
                continue
            for name in R0_DIGESTS:
                value = reading[name]
                if not isinstance(value, str) or not value:
                    problems.append(f"repeatability reading {index} has no {name} digest")
                    valid_readings = False
                else:
                    derived[name].append(value)
            for name in ("final_loss", "seconds"):
                value = reading[name]
                if not isinstance(value, (int, float)) or isinstance(value, bool) \
                        or not math.isfinite(float(value)):
                    problems.append(f"repeatability reading {index} has invalid {name}")
                    valid_readings = False
    derived = {name: sorted(set(values)) for name, values in derived.items()}
    if not valid_readings:
        # Keep the derived summaries empty rather than accidentally accepting a
        # partially parsed list.
        derived = {name: [] for name in R0_DIGESTS}

    digest_sets = repeatability_report.get("digests")
    if not isinstance(digest_sets, dict) or set(digest_sets) != set(R0_DIGESTS):
        problems.append(
            f"the report must carry all of {list(R0_DIGESTS)}; it carries "
            f"{sorted(digest_sets) if isinstance(digest_sets, dict) else None}")
    elif digest_sets != derived:
        problems.append("the reported digest sets do not match the raw readings")
    distinct = repeatability_report.get("distinct_digests")
    expected_distinct = {name: len(values) for name, values in derived.items()}
    if not isinstance(distinct, dict) or set(distinct) != set(R0_DIGESTS) \
            or distinct != expected_distinct:
        problems.append("the reported distinct digest counts do not match raw readings")
    expected_exact = valid_readings and all(
        len(values) == 1 for values in derived.values())
    claimed_exact = repeatability_report.get("exact_state_reproduction")
    if not isinstance(claimed_exact, bool) or claimed_exact != expected_exact:
        problems.append(
            "exact_state_reproduction does not match the independently derived digest sets")
    if claimed_exact is not True or expected_exact is not True:
        problems.append(
            "the platform does not reproduce state exactly, and v1 has no metric-level "
            "fallback route: R0 fails")

    determinism = repeatability_report.get("determinism")
    if not isinstance(determinism, dict) or \
            determinism.get("requested") is not True or \
            determinism.get("deterministic_algorithms") is not True or \
            determinism.get("is_policy_not_certificate") is not True:
        problems.append(
            "the reading does not record the determinism policy as requested and enabled")

    if not isinstance(resume_report, dict) or resume_report.get("schema") != \
            EVIDENCE_SCHEMA:
        problems.append("the resume report is missing or of an unknown schema")
    else:
        if set(resume_report) != set(R0_RESUME_FIELDS):
            problems.append(
                f"the resume report must carry exactly {list(R0_RESUME_FIELDS)}")
        resume_device = required(resume_report, "device", str, "resume")
        resume_seed = required(resume_report, "seed", int, "resume")
        resume_sentinel = required(resume_report, "sentinel", dict, "resume")
        resume_environment = required(resume_report, "environment", dict, "resume")
        if isinstance(resume_device, str) and resume_device != device:
            problems.append("repeatability and resume reports use different devices")
        if isinstance(resume_seed, int) and resume_seed != seed:
            problems.append("repeatability and resume reports use different seeds")
        if isinstance(resume_sentinel, dict) and resume_sentinel != sentinel:
            problems.append("repeatability and resume reports use different sentinel configs")
        if isinstance(resume_environment, dict) and resume_environment != environment:
            problems.append("repeatability and resume reports use different environments")
        if isinstance(resume_environment, dict) and \
                set(resume_environment) != set(R0_ENVIRONMENT_FIELDS):
            problems.append("the resume environment schema is incomplete")
        resume_step = required(resume_report, "resume_step", int, "resume")
        if isinstance(resume_step, int) and resume_step != R0_RESUME_STEP:
            problems.append(
                f"the resume point is {resume_step}, not the frozen {R0_RESUME_STEP}")
        for name, expected in (("serialised", True),
                               ("rebuilt_model_and_optimizer", True),
                               ("process_boundary", False)):
            value = resume_report.get(name)
            if value is not expected:
                problems.append(f"the resume report does not prove {name}={expected}")
        uninterrupted = exact_mapping(
            resume_report.get("uninterrupted"), R0_RESUME_DIGEST_FIELDS,
            "uninterrupted")
        resumed = exact_mapping(
            resume_report.get("resumed"), R0_RESUME_DIGEST_FIELDS, "resumed")
        expected_differing = []
        if uninterrupted is not None and resumed is not None:
            expected_differing = sorted(
                name for name in R0_RESUME_DIGEST_FIELDS
                if uninterrupted[name] != resumed[name])
        differing = resume_report.get("differing_digests")
        if not isinstance(differing, list) or differing != expected_differing:
            problems.append(
                "the claimed differing-digest list does not match uninterrupted and "
                "resumed maps")
        equivalent = resume_report.get("equivalent")
        expected_equivalent = uninterrupted is not None and resumed is not None \
            and not expected_differing
        if not isinstance(equivalent, bool) or equivalent != expected_equivalent:
            problems.append(
                "the resume equivalent verdict does not match the compared digest maps")
        loss_difference = resume_report.get("loss_difference")
        if not isinstance(loss_difference, (int, float)) or \
                isinstance(loss_difference, bool) or not math.isfinite(float(loss_difference)):
            problems.append("the resume report has no finite loss difference")
        if equivalent is not True or expected_equivalent is not True:
            problems.append("an interrupted run does not resume to the same state")

    return {
        "schema": EVIDENCE_SCHEMA,
        "qualified": not problems,
        "route": "exact_state_reproduction",
        "fallback_route": R0_FALLBACK_STATUS,
        "problems": problems,
        "forbidden_comparator": R0_FORBIDDEN_DIGEST,
        "determinism_is_policy_not_certificate": True,
    }


__all__ = [
    "EVIDENCE_SCHEMA", "SentinelConfig", "SentinelReading",
    "deterministic_execution", "execution_environment", "optimizer_digest",
    "qualify", "repeatability", "resume_equivalence", "rng_digest",
    "run_sentinel", "state_digest",
]
