"""Text assembly <-> ISA bytecode.

The text form exists for tests, dataset inspection and paper figures. Nothing in
the training path depends on it.

    MOVE 10 20
    REPEAT 4 8 0      ; four copies, stepping +8 in x
      LINE 40 40
    ENDREP
    HALT
"""

from __future__ import annotations

from dataclasses import dataclass

from .spec import BY_MNEMONIC, ISAError, decode_operand, encode_operand, spec_for


@dataclass(frozen=True)
class Instr:
    mnemonic: str
    args: tuple[int, ...]

    def __str__(self) -> str:
        return " ".join((self.mnemonic, *(str(a) for a in self.args)))


class AsmError(ISAError):
    pass


def _strip_comment(line: str) -> str:
    for marker in (";", "#"):
        idx = line.find(marker)
        if idx >= 0:
            line = line[:idx]
    return line.strip()


def assemble(text: str) -> bytes:
    out = bytearray()
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = _strip_comment(raw)
        if not line:
            continue
        mnemonic, *rest = line.split()
        spec = BY_MNEMONIC.get(mnemonic.upper())
        if spec is None:
            raise AsmError(f"line {lineno}: unknown mnemonic {mnemonic!r}")
        if len(rest) != len(spec.operands):
            raise AsmError(
                f"line {lineno}: {spec.mnemonic} takes {len(spec.operands)} "
                f"operand(s), got {len(rest)}"
            )
        out.append(int(spec.op))
        for kind, tok in zip(spec.operands, rest):
            try:
                value = int(tok, 0)
            except ValueError as exc:
                raise AsmError(f"line {lineno}: bad operand {tok!r}") from exc
            try:
                out.append(encode_operand(kind, value))
            except ISAError as exc:
                raise AsmError(f"line {lineno}: {exc}") from exc
    return bytes(out)


def parse(program: bytes) -> list[Instr]:
    """Decode bytecode into instructions. Raises on malformed input.

    Use this only where a hard failure is wanted; generated programs should go
    through the VM instead, which reports faults rather than raising.
    """
    instrs: list[Instr] = []
    pc = 0
    while pc < len(program):
        spec = spec_for(program[pc])
        end = pc + spec.size
        if end > len(program):
            raise AsmError(f"truncated {spec.mnemonic} at byte {pc}")
        args = tuple(
            decode_operand(kind, byte)
            for kind, byte in zip(spec.operands, program[pc + 1 : end])
        )
        instrs.append(Instr(spec.mnemonic, args))
        pc = end
    return instrs


def disassemble(program: bytes, indent: bool = True) -> str:
    lines: list[str] = []
    depth = 0
    for instr in parse(program):
        if instr.mnemonic == "ENDREP":
            depth = max(0, depth - 1)
        lines.append(("  " * depth if indent else "") + str(instr))
        if instr.mnemonic == "REPEAT":
            depth += 1
    return "\n".join(lines)
