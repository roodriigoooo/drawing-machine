"""One drawing, sampled from a class-conditional checkpoint, as bytecode.

Everything downstream of this file -- the device, the wire, the page -- handles
bytecode. This is the only place a neural network appears in the demo path, and
it produces `bytes`, not pixels: that is the whole point the demo exists to
make legible.

The sampler settings are arguments and are carried into the record, because a
sampler is part of a generation result (`dm/eval/sampling.py`). The defaults
here are `top_k=80, T=1.0` rather than the `top_k=40` most reports on file were
drawn at, and that is a deliberate, sourced choice: the no-training sweep in
`docs/conditioning.md` §7 moved `coverage` 0.406 -> 0.487 and `nna` 0.639 ->
0.549 toward their floors at `k=80` without losing fidelity, and named it the
best tested conditional setting. A demo that quietly used the *published* k
would be showing a sampler this project has already measured as more
mode-seeking than necessary.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch

from ..eval.metrics import sample_programs
from ..eval.records import load
from ..isa.asm import AsmError, disassemble, parse
from ..isa.codec import CODECS
from ..vm.interp import VM, Trace

#: The sweep's best tested conditional setting, not the published default.
DEFAULT_TOP_K = 80
DEFAULT_TEMPERATURE = 1.0

#: Bytecode-byte cap for one drawing. The live corpus averages 112.7 bytes and
#: the longest val programs are a few hundred, so this bounds a runaway decode
#: without truncating anything the corpus contains.
DEFAULT_MAX_BYTES = 512


@dataclass(frozen=True)
class Checkpoint:
    """A loaded conditional model plus the facts the demo needs to state."""

    model: object
    record: dict
    codec: object
    classes: tuple[str, ...]
    path: Path

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def parameters(self) -> int:
        return int(self.record["model"]["params"])

    def as_dict(self) -> dict:
        return {
            "name": self.record["name"],
            "path": str(self.path),
            "params": self.parameters,
            "codec": self.record["config"]["codec"],
            "shape": self.record["config"]["shape"],
            "steps": self.record["config"]["steps"],
            "classes": list(self.classes),
            "rdp_eps": (self.record["config"].get("extra") or {}).get("rdp_eps"),
            "val_bits_per_drawing": self.record.get("best_val_bits"),
        }


def load_checkpoint(path: Path) -> Checkpoint:
    """Load, and refuse anything that cannot answer "draw me an X".

    The refusal is not defensive coding. An unconditional checkpoint samples
    perfectly happily when handed a `classes` tensor it ignores, so the demo
    would show five buttons, five different drawings, and no conditioning at
    all -- a class-controlled UI over a model with no class input is a claim
    the run record can disprove and the screen cannot.
    """
    model, record = load(path)
    if not record["model"].get("n_classes"):
        raise ValueError(
            f"{record['name']} has no class input; a word-to-drawing demo needs a "
            "checkpoint trained with --conditional"
        )
    categories = record["config"].get("categories")
    if not categories:
        raise ValueError(f"{record['name']} records no category list to route words against")
    if len(categories) != record["model"]["n_classes"]:
        raise ValueError(
            f"{record['name']} conditions on {record['model']['n_classes']} classes but "
            f"names {len(categories)}; the index a word routes to would be a guess"
        )
    codec = CODECS[record["config"]["codec"]]
    return Checkpoint(model, record, codec, tuple(categories), path)


@dataclass(frozen=True)
class Sampled:
    """One sampled program and everything said about it before it left the host."""

    program: bytes
    class_index: int
    seed: int
    top_k: int | None
    temperature: float
    truncated: bool
    seconds: float
    #: The host reference VM's trace. **Assertion material only**: the demo page
    #: renders the device's geometry and compares it against this. It is carried
    #: because the comparison has to happen somewhere, and never rendered.
    reference: Trace
    disassembly: str
    instructions: int

    @property
    def valid(self) -> bool:
        return bool(self.program) and self.reference.valid

    def as_dict(self) -> dict:
        return {
            "bytecode_hex": self.program.hex(),
            "bytes": len(self.program),
            "instructions": self.instructions,
            "disassembly": self.disassembly,
            "class_index": self.class_index,
            "seed": self.seed,
            "top_k": self.top_k,
            "temperature": self.temperature,
            "truncated": self.truncated,
            "sample_seconds": round(self.seconds, 3),
            "reference_valid": self.reference.valid,
        }


def sample(checkpoint: Checkpoint, class_index: int, seed: int,
           top_k: int | None = DEFAULT_TOP_K,
           temperature: float = DEFAULT_TEMPERATURE,
           max_bytes: int = DEFAULT_MAX_BYTES,
           device: str = "cpu",
           prompt: bytes | None = None) -> Sampled:
    """Draw one program under `class_index`, reproducibly.

    `seed` is the whole reproducibility story: the same checkpoint, class, seed
    and sampler settings return the identical bytes, which is what lets a
    captured record be replayed and re-verified after the boards are gone. It
    is stored in the record rather than left to the global stream.

    `prompt` is a bytecode prefix the model must continue -- the held-out-human
    prefix panel. It is fed through the codec's halt monitor as bytecode, so an
    operand `0x00` at the end of a prefix is not misread as `HALT`
    (`DrawingLM.generate`).
    """
    if not 0 <= class_index < checkpoint.n_classes:
        raise ValueError(f"class {class_index} is outside 0..{checkpoint.n_classes - 1}")
    stride = checkpoint.codec.stride
    max_new = max_bytes * stride

    classes = torch.tensor([class_index], dtype=torch.long)
    encoded = None
    if prompt:
        encoded = torch.tensor(
            [checkpoint.codec.encode(prompt)], dtype=torch.long
        )

    torch.manual_seed(seed)
    started = time.monotonic()
    programs, truncated = sample_programs(
        checkpoint.model, checkpoint.codec, n=1, max_new=max_new, device=device,
        temperature=temperature, top_k=top_k, classes=classes,
        forbid_specials=True, prompt=encoded,
    )
    seconds = time.monotonic() - started

    program = programs[0]
    reference = VM().run(program) if program else VM().run(b"")
    try:
        instructions = len(parse(program))
    except AsmError:
        # A sampled program can be truncated mid-instruction. The VM reports
        # that as a fault and the demo shows the fault; the disassembler raises,
        # so the count is reported as unknown rather than as a smaller number.
        instructions = -1
    return Sampled(
        program=program,
        class_index=class_index,
        seed=seed,
        top_k=top_k,
        temperature=temperature,
        truncated=bool(truncated[0]),
        seconds=seconds,
        reference=reference,
        disassembly=_safe_disassembly(program),
        instructions=instructions,
    )


def _safe_disassembly(program: bytes) -> str:
    """Text for the page, including for a program that does not fully decode.

    A truncated tail is shown as raw bytes rather than dropped: the page's
    instruction panel is a view of what the device is about to execute, and
    hiding the bytes that cause the fault would make a refused run look like a
    clean one.
    """
    if not program:
        return ""
    try:
        return disassemble(program)
    except AsmError:
        head = program
        while head:
            try:
                text = disassemble(head)
            except AsmError:
                head = head[:-1]
                continue
            tail = program[len(head):]
            return text + f"\n; undecodable tail: {tail.hex(' ')}"
        return f"; undecodable: {program.hex(' ')}"
