"""The three representations under test: instruction tokens, bytes, bits.

All three encode the *same* bytecode with the *same* 8-bit coordinate
resolution, so they carry identical information. What differs is how much
structure the alphabet gives away for free:

    TokenCodec   opcodes live in a disjoint region of the vocabulary, so
                 "is this position an opcode?" is one free bit per position.
    ByteCodec    untyped. Byte 0x02 is OP_LINE or the value 2 depending on a
                 context the model has to infer.
    BitCodec     two symbols. Byte boundaries, field widths and opcode
                 identity all have to be discovered.

TokenCodec(typed_operands=True) is a secondary axis: it also splits the operand
alphabet by Kind, testing whether *more* typing helps or merely inflates the
embedding table.

RelativeCodec is a third, and it is orthogonal to both: it wraps any of the
above and shows the model coordinates as deltas from the pen instead of as
canvas positions. Same alphabet, same vocabulary, same sequence length, same
information -- so `CODECS` is a 4 x 2 grid and every axis can be read against
either view.

Decoding is deliberately permissive. A structurally impossible stream decodes to
whatever bytes it names, and the resulting fault is reported by the VM. That
keeps validity measured at one place for all three codecs instead of some
failures surfacing as codec exceptions and others as VM faults.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .relative import to_absolute, to_relative
from .spec import (
    OPCODE_SLOTS,
    SPECS,
    ISAError,
    Kind,
    Op,
    UnknownOpcode,
    spec_for,
)

PAD = 0
BOS = 1
N_SPECIAL = 2

_OPCODE_BYTES = sorted(int(op) for op in SPECS)
_OPCODE_INDEX = {byte: i for i, byte in enumerate(_OPCODE_BYTES)}
#: Padded to the reserved width, with -1 in the slots no opcode occupies yet.
#: A reserved slot is **not a bytecode byte**, which is the same answer `decode`
#: gives it and the same one `HaltMonitor` gives `PAD` -- so a model that emits
#: one has emitted nothing rather than an aliased instruction.
_OPCODE_TABLE = np.full(max(OPCODE_SLOTS, len(_OPCODE_BYTES)), -1, dtype=np.int16)
_OPCODE_TABLE[: len(_OPCODE_BYTES)] = _OPCODE_BYTES

#: Encoded length of the instruction each byte introduces, -1 where the byte is
#: not an opcode. Indexed by bytecode byte, so the halt monitor can take the
#: same branch the VM takes without importing it.
_INSTR_SIZE = np.full(256, -1, dtype=np.int16)
for _spec in SPECS.values():
    _INSTR_SIZE[int(_spec.op)] = _spec.size


def opcode_mask(program: bytes) -> list[bool]:
    """True at every byte position that is an opcode.

    Walks the stream the way the VM does. On an unknown opcode or a truncated
    instruction it marks the rest of the stream as operands, which is arbitrary
    but only ever applies to programs that are already invalid.
    """
    mask = [False] * len(program)
    pc = 0
    while pc < len(program):
        try:
            spec = spec_for(program[pc])
        except UnknownOpcode:
            break
        if pc + spec.size > len(program):
            break
        mask[pc] = True
        pc += spec.size
    return mask


class HaltMonitor:
    """Per-row parse state, so sampling stops exactly where the VM stops.

    Matching HALT as a *symbol* is wrong for every untyped alphabet. Under the
    byte codec `0x00` is the HALT opcode and the operand value zero, and the two
    are told apart only by whether the position is an instruction boundary.
    Matching the symbol therefore truncates programs mid-instruction, and the
    truncation is scored as the representation's fault: measured at 0.66 halted
    for a byte checkpoint that reaches 0.98 when the VM is left to find HALT
    itself. Instructions are variable-length, so a fixed stride cannot locate
    the boundaries either -- only a parse can.

    Streams also stop at an unknown opcode, because `VM.run` breaks there and
    every later byte is dead. That keeps the invariant this class exists for:
    the trace of the truncated stream equals the trace of the full one.
    """

    __slots__ = ("_buf", "codec", "done", "pending", "stride")

    def __init__(self, codec: Codec, rows: int) -> None:
        self.codec, self.stride = codec, codec.stride
        self.pending = np.zeros(rows, dtype=np.int16)  # operand bytes still owed
        self.done = np.zeros(rows, dtype=bool)
        self._buf: list[np.ndarray] = []

    def step(self, chunk: np.ndarray) -> np.ndarray:
        """Consume `stride` symbols per row; return the rows that have stopped.

        `chunk` is (stride, rows) of symbol ids, one bytecode byte's worth.
        """
        byte = self.codec.symbols_to_bytes(chunk)
        live = ~self.done & (byte >= 0)  # < 0 is PAD/BOS, i.e. no bytecode byte
        boundary = live & (self.pending == 0)
        self.pending[live & (self.pending > 0)] -= 1

        size = _INSTR_SIZE[np.where(byte >= 0, byte, 0)]
        stops = boundary & ((byte == int(Op.HALT)) | (size < 0))
        opens = boundary & ~stops
        self.pending[opens] = size[opens] - 1
        self.done |= stops
        return self.done


class Codec(ABC):
    name: str
    vocab_size: int

    @abstractmethod
    def encode(self, program: bytes) -> list[int]: ...

    @abstractmethod
    def decode(self, tokens: list[int]) -> bytes: ...

    @abstractmethod
    def symbols_to_bytes(self, chunk: np.ndarray) -> np.ndarray:
        """(stride, rows) symbol ids -> (rows,) bytecode bytes, -1 for PAD/BOS.

        The vectorised counterpart of `decode`, for the sampler's inner loop.
        `test_symbols_to_bytes_agrees_with_decode` pins the two together.
        """

    @abstractmethod
    def encode_values(self, values: bytes) -> list[int]:
        """Raw bytes as operand-value symbols, with no instruction parse.

        Claim 3's stroke decoder is conditioned on a six-byte stroke summary
        (`dm.isa.strokes`), and a summary is *not* a program -- parsing it would
        read byte 1 as `MOVE`. Spelling it in the codec's own operand alphabet
        instead means the conditioning prefix costs no vocabulary at all, so the
        planner's parameter count stays comparable with the AR baseline's and
        the granularity axis applies to the prefix exactly as it does to the
        stroke. Abstract rather than defaulted: it is `encode` for three of the
        four alphabets and silently wrong for the fourth.
        """

    def opcode_symbol(self, byte: int) -> int | None:
        """The canonical one-symbol spelling of an opcode byte, if available.

        The prefix-state Adapter uses this public seam instead of reaching into
        a codec's alphabet tables. Bit codecs intentionally have no single
        opcode symbol and return ``None``.
        """
        symbols = self.encode(bytes([byte & 0xFF]))
        return symbols[0] if len(symbols) == 1 else None

    def value_symbol(self, kind: Kind, byte: int) -> int | None:
        """The canonical value spelling for one operand byte, if available."""
        symbols = self.encode_values(bytes([byte & 0xFF]))
        return symbols[0] if len(symbols) == 1 else None

    def symbol_to_byte(self, symbol: int) -> int | None:
        """Decode one complete stride-1 symbol at the codec seam."""
        if self.stride != 1:
            return None
        byte = int(self.symbols_to_bytes(
            np.array([[symbol]], dtype=np.int64)
        )[0])
        return byte if byte >= 0 else None

    #: Symbols per bytecode byte. One byte is the smallest unit at which the
    #: stream can be parsed, so it is also how often the halt monitor decides.
    stride: int = 1

    def stream_bytes(self, program: bytes) -> bytes:
        """The bytecode bytes this codec's symbols actually spell.

        The identity for every absolute alphabet, which is why the distinction
        went unnamed until `RelativeCodec` broke it: those symbols spell the
        delta view, so `symbols_to_bytes` -- and with it the halt monitor --
        works in that domain from end to end. Opcodes and instruction sizes are
        the same in both domains, which is what makes that safe; naming the
        domain is what keeps it checkable.
        """
        return program

    def halt_monitor(self, rows: int) -> HaltMonitor:
        return HaltMonitor(self, rows)

    def with_bos(self, program: bytes) -> list[int]:
        return [BOS, *self.encode(program)]

    def __repr__(self) -> str:
        return f"{type(self).__name__}(vocab={self.vocab_size})"


class ByteCodec(Codec):
    name = "byte"
    vocab_size = N_SPECIAL + 256

    def encode(self, program: bytes) -> list[int]:
        return [N_SPECIAL + b for b in program]

    def decode(self, tokens: list[int]) -> bytes:
        return bytes(
            (t - N_SPECIAL) & 0xFF for t in tokens if t >= N_SPECIAL
        )

    def symbols_to_bytes(self, chunk: np.ndarray) -> np.ndarray:
        t = chunk[0].astype(np.int16)
        return np.where(t >= N_SPECIAL, (t - N_SPECIAL) & 0xFF, -1)

    def encode_values(self, values: bytes) -> list[int]:
        return self.encode(values)  # untyped: every position is already a value


class TokenCodec(Codec):
    """Opcodes in a disjoint region; operand values in one shared 256-way
    alphabet, or split per Kind when `typed_operands` is set."""

    name = "token"

    def __init__(self, typed_operands: bool = False,
                 opcode_slots: int = OPCODE_SLOTS,
                 n_kinds: int = len(Kind)) -> None:
        """`opcode_slots` is a **wire-format** parameter, not a tuning knob.

        The layout is `[specials][opcode slots][values]`, so the region's width
        decides where every value token lives. Before it was reserved it was
        `N_OPCODES`, which meant adding an opcode moved the value region and a
        checkpoint trained under one table decoded its own generated symbols to
        *different opcodes* under the next -- silently, because the model is
        rebuilt from the vocabulary in its blob while the codec is rebuilt from
        today's table.

        It defaults to `OPCODE_SLOTS`, which is fixed and larger than
        `N_OPCODES` so that every future opcode is additive. Pass 11 to
        reconstruct a pre-2026-08-11 checkpoint exactly -- a narrower codec is
        legal and simply cannot spell the opcodes added since, which it says at
        `encode` rather than by aliasing two opcodes onto one symbol.

        `n_kinds` is the same idea for the typed alphabet and is **deliberately
        not reserved**. A typed codec gives every `Kind` its own 256 values, so
        adding `Kind.XF` moved `token_typed` from 1,293 symbols to 1,554 -- and
        reserving spare kinds would inflate the embedding of every typed arm
        against an 825k budget that the fusion axis is measured under. An opcode
        shift was gratuitous; a kind shift *is* the typed axis changing, which
        is the thing being measured. So pre-v2 typed checkpoints are
        reconstructed with `TokenCodec(True, opcode_slots=11, n_kinds=5)`.
        """
        if opcode_slots < 1:
            raise ISAError(f"{opcode_slots} opcode slots is not an alphabet")
        self.typed_operands = typed_operands
        self.opcode_slots = opcode_slots
        self.n_kinds = n_kinds
        self._op_base = N_SPECIAL
        self._val_base = self._op_base + opcode_slots
        n_alphabets = n_kinds if typed_operands else 1
        self.vocab_size = self._val_base + 256 * n_alphabets
        self.name = "token_typed" if typed_operands else "token"

    def _value_token(self, kind: Kind, byte: int) -> int:
        offset = 256 * int(kind) if self.typed_operands else 0
        return self._val_base + offset + byte

    # Public codec seam for the state Adapter. The implementation keeps its
    # layout tables private; callers ask the codec how a byte is spelled.
    def opcode_symbol(self, byte: int) -> int | None:
        index = _OPCODE_INDEX.get(int(byte))
        if index is None or index >= self.opcode_slots:
            return None
        return self._op_base + index

    def value_symbol(self, kind: Kind, byte: int) -> int | None:
        if not 0 <= int(kind) < self.n_kinds:
            return None
        return self._value_token(kind, int(byte))

    def symbol_to_byte(self, symbol: int) -> int | None:
        symbol = int(symbol)
        if symbol < self._op_base:
            return None
        if symbol < self._val_base:
            index = symbol - self._op_base
            return _OPCODE_BYTES[index] if index < len(_OPCODE_BYTES) else None
        return (symbol - self._val_base) % 256

    def encode(self, program: bytes) -> list[int]:
        mask = opcode_mask(program)
        tokens: list[int] = []
        pending: list[Kind] = []
        for byte, is_op in zip(program, mask):
            if is_op:
                index = _OPCODE_INDEX[byte]
                if index >= self.opcode_slots:
                    # A legacy-width codec meeting an opcode added after it.
                    # Refused rather than wrapped: the alternative is a symbol
                    # that decodes to a different instruction, which is the
                    # exact failure the reserved region exists to prevent.
                    raise ISAError(
                        f"{spec_for(byte).mnemonic} is opcode index {index}, "
                        f"outside this codec's {self.opcode_slots} slots"
                    )
                tokens.append(self._op_base + index)
                pending = list(spec_for(byte).operands)
            else:
                kind = pending.pop(0) if pending else Kind.SCALAR
                tokens.append(self._value_token(kind, byte))
        return tokens

    def encode_values(self, values: bytes) -> list[int]:
        # `Kind.COORD` because a summary is a position and an extent. It also
        # keeps `token` and `token_typed` spelling the prefix identically,
        # which is what stops the fusion axis from picking up a difference that
        # is really about the conditioning.
        return [self._value_token(Kind.COORD, b) for b in values]

    def decode(self, tokens: list[int]) -> bytes:
        out = bytearray()
        for t in tokens:
            if t < self._op_base:
                continue
            if t < self._val_base:
                idx = t - self._op_base
                if idx < len(_OPCODE_BYTES):
                    out.append(_OPCODE_BYTES[idx])
            else:
                out.append((t - self._val_base) % 256)
        return bytes(out)

    def symbols_to_bytes(self, chunk: np.ndarray) -> np.ndarray:
        t = chunk[0].astype(np.int32)
        out = np.full(t.shape, -1, dtype=np.int16)
        is_op = (t >= self._op_base) & (t < self._val_base)
        out[is_op] = _OPCODE_TABLE[t[is_op] - self._op_base]
        is_val = t >= self._val_base
        out[is_val] = (t[is_val] - self._val_base) % 256
        return out


class BitCodec(Codec):
    """MSB-first bit expansion of the byte stream."""

    name = "bit"
    vocab_size = N_SPECIAL + 2
    stride = 8

    def encode(self, program: bytes) -> list[int]:
        return [
            N_SPECIAL + ((byte >> shift) & 1)
            for byte in program
            for shift in range(7, -1, -1)
        ]

    def decode(self, tokens: list[int]) -> bytes:
        bits = [t - N_SPECIAL for t in tokens if t >= N_SPECIAL]
        del bits[len(bits) - len(bits) % 8 :]  # drop trailing partial byte
        out = bytearray()
        for i in range(0, len(bits), 8):
            byte = 0
            for bit in bits[i : i + 8]:
                byte = (byte << 1) | bit
            out.append(byte)
        return bytes(out)

    def encode_values(self, values: bytes) -> list[int]:
        return self.encode(values)  # two symbols, and neither of them is typed

    def symbols_to_bytes(self, chunk: np.ndarray) -> np.ndarray:
        bits = chunk.astype(np.int16) - N_SPECIAL
        packed = np.zeros(chunk.shape[1], dtype=np.int16)
        for bit in bits:  # MSB first, mirroring `encode`
            packed = (packed << 1) | np.maximum(bit, 0)
        # A byte is only a byte if all eight symbols were bits.
        return np.where((bits >= 0).all(axis=0), packed, -1)


class RelativeCodec(Codec):
    """Any alphabet, over the relative view of the same bytecode.

    Composition rather than four more codec classes, because relativity is an
    *axis*: it has to be readable against typing, granularity and fusion alike,
    and a hand-written `RelativeByteCodec` would be a second place for the byte
    alphabet to drift. `vocab_size` and `stride` are the wrapped codec's, and
    the encoded length is identical symbol for symbol -- `dm.isa.relative`
    rewrites operand *values* and never the instruction stream's shape. So a
    delta arm and its absolute partner differ in exactly one thing, which is
    what makes their difference a measurement.

    `symbols_to_bytes` is delegated *unconverted*, and that is deliberate. Its
    caller is `HaltMonitor`, which needs opcodes and instruction boundaries; the
    rewrite leaves every opcode byte alone, so the parse state is identical in
    both views and a monitor built on the relative view stops exactly where the
    VM stops on the absolute one. Converting there instead would need the
    monitor to carry the pen, and would buy nothing -- it never reads an operand
    value.
    """

    def __init__(self, inner: Codec) -> None:
        self.inner = inner
        self.name = f"{inner.name}_delta"
        self.vocab_size = inner.vocab_size
        self.stride = inner.stride

    def encode(self, program: bytes) -> list[int]:
        return self.inner.encode(to_relative(program))

    def decode(self, tokens: list[int]) -> bytes:
        return to_absolute(self.inner.decode(tokens))

    def stream_bytes(self, program: bytes) -> bytes:
        return to_relative(program)

    def encode_values(self, values: bytes) -> list[int]:
        # Delegated *without* the rewrite. A stroke summary is a layout fact --
        # where on the canvas this stroke sits -- and the composition level's
        # whole job is to place strokes absolutely. Relativising it would make
        # the prefix mean something different from what the denoiser predicted.
        return self.inner.encode_values(values)

    def symbols_to_bytes(self, chunk: np.ndarray) -> np.ndarray:
        return self.inner.symbols_to_bytes(chunk)

    def opcode_symbol(self, byte: int) -> int | None:
        return self.inner.opcode_symbol(byte)

    def value_symbol(self, kind: Kind, byte: int) -> int | None:
        return self.inner.value_symbol(kind, byte)

    def symbol_to_byte(self, symbol: int) -> int | None:
        return self.inner.symbol_to_byte(symbol)


_ABSOLUTE: tuple[Codec, ...] = (
    ByteCodec(), TokenCodec(), TokenCodec(typed_operands=True), BitCodec(),
)

#: The 4 x 2 grid: four alphabets, each over the absolute and the relative view.
CODECS: dict[str, Codec] = {
    c.name: c for c in (*_ABSOLUTE, *(RelativeCodec(c) for c in _ABSOLUTE))
}
