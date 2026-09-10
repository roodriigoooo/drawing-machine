"""Per-field attribution: the partition, the alphabets, and the reconciliation.

Three things have to hold or none of the numbers mean anything.

**Every cut partitions every byte.** A gap loses bits silently and only the
reconciliation would notice, later and as a mystery. A byte counted twice
inflates whichever field it landed in.

**The legal-symbol sets are the codec's, not a guess.** They are what the waste
column measures against, so a set that excluded a symbol actually present would
report an unbounded saving -- the exact failure `docs/traps.md` records for
`dm/eval/redundancy.py`'s feasibility masks, checked there at every position of
a real corpus and checked here the same way.

**The decomposition sums to the arm's own cost.** Tested against
`dm.eval.metrics.per_program_bits`, which computes the same quantity through an
entirely different path.
"""

import numpy as np
import pytest
import torch

from dm.data.dataset import ProgramDataset, collate
from dm.eval.attribution import (
    BITPLANE,
    CUTS,
    FIELDS,
    OPCODES,
    POSITION_CLASSES,
    SLOTS,
    VOCABULARY,
    attribute,
    labels,
    legal_symbols,
    reconcile,
)
from dm.eval.metrics import per_program_bits
from dm.isa.asm import assemble
from dm.isa.codec import BOS, CODECS, PAD, TokenCodec
from dm.isa.spec import SPECS, Kind, Op
from dm.models.transformer import Config, DrawingLM

#: One program reaching every operand `Kind` the ISA defines, so the field cut is
#: exercised on `SCALAR`, `COUNT`, `DELTA` and `XF` and not only on coordinates.
_RICH_SOURCE = """
MOVE 10 20
LINE 30 40
CURVE 1 2 3 4 5 6
CIRCLE 7
WIDTH 3
REPEAT 4 8 0
LINE 40 40
ENDREP
XFORM 3 10 -10
LINE 50 50
ENDX
REPEATX 3 5 -2 4
LINE 9 9
ENDREP
HALT
"""

RICH = assemble(_RICH_SOURCE)

CORPUS = [RICH, assemble("MOVE 5 5\nLINE 200 200\nHALT"),
          assemble("MOVE 1 1\nCURVE 9 9 8 8 7 7\nFILL\nHALT")]


def _model(codec, max_len: int = 256) -> DrawingLM:
    torch.manual_seed(0)
    return DrawingLM(Config(vocab_size=codec.vocab_size, d_model=32, n_layers=2,
                            n_heads=2, max_len=max_len)).eval()


@pytest.mark.parametrize("cut", CUTS)
@pytest.mark.parametrize("program", CORPUS)
def test_every_cut_labels_every_byte_exactly_once(cut, program):
    got = labels(program)[cut]
    assert len(got) == len(program)
    assert got.min() >= 0 and got.max() < len(VOCABULARY[cut])


def test_operands_are_labelled_by_kind_and_never_by_position():
    """A `CIRCLE` radius is a length and a `REPEAT` step is a vector; reading
    either as a coordinate corrupts the field it lands in and the one it left."""
    field = [FIELDS[i] for i in labels(RICH)["field"]]
    names = [SPECS[Op(b)].mnemonic if ok else None
             for b, ok in zip(RICH, _opcode_positions(RICH))]
    for position, mnemonic in enumerate(names):
        if mnemonic != "CIRCLE":
            continue
        assert field[position + 1] == "scalar"
    assert "delta_x" in field and "delta_y" in field       # REPEAT / REPEATX
    assert "xf" in field and "count" in field              # REPEATX
    assert field.count("coord_x") == field.count("coord_y")


def _opcode_positions(program: bytes) -> list[bool]:
    from dm.isa.codec import opcode_mask
    return opcode_mask(program)


def test_the_opcode_cut_agrees_with_the_codec_s_own_walk():
    """Two implementations of "where do instructions start" would eventually
    disagree; `opcode_mask` is the one the token codec encodes with."""
    field = labels(RICH)["field"]
    assert [f == FIELDS.index("opcode") for f in field] == _opcode_positions(RICH)


def test_a_truncated_tail_is_named_unparsed_rather_than_dropped():
    broken = RICH[:-1] + bytes((int(Op.CURVE), 1, 2))     # CURVE wants six
    field = labels(broken)["field"]
    assert (field[-3:] == FIELDS.index("unparsed")).all()
    assert len(field) == len(broken)


def test_an_unknown_opcode_stops_the_walk_where_the_vm_stops():
    bad = bytes((int(Op.MOVE), 1, 2, 0xEE, 9, 9, int(Op.HALT)))
    field = [FIELDS[i] for i in labels(bad)["field"]]
    assert field[:3] == ["opcode", "coord_x", "coord_y"]
    assert set(field[3:]) == {"unparsed"}


@pytest.mark.parametrize("name", ["byte", "token", "token_typed", "bit"])
@pytest.mark.parametrize("program", CORPUS)
def test_the_legal_set_never_excludes_a_symbol_that_is_actually_there(name, program):
    """The failure mode that would make the waste column lie.

    `docs/traps.md`: *a feasibility mask that excludes the byte actually present
    reports an unbounded saving, and it looks exactly like a finding.* So the
    sets are checked at every position of a real program in every alphabet,
    which is how the same rule caught a live fault in `redundancy.py`.
    """
    codec = CODECS[name]
    sets = {k: set(v.tolist()) for k, v in legal_symbols(codec).items()}
    symbols = codec.encode(program)
    from dm.eval.attribution import _CLASS_OF_FIELD
    classes = np.repeat(
        [_CLASS_OF_FIELD[FIELDS[i]] for i in labels(program)["field"]], codec.stride
    )
    for position, (symbol, klass) in enumerate(zip(symbols, classes)):
        assert symbol in sets[klass], (
            f"{name}: symbol {symbol} at {position} is outside its own {klass} set"
        )


@pytest.mark.parametrize("name", ["byte", "token", "token_typed", "bit"])
def test_pad_and_bos_are_illegal_in_every_alphabet(name):
    """They are model controls and never bytecode, so mass on them is waste of
    the same kind and has to be counted with it."""
    for ids in legal_symbols(CODECS[name]).values():
        assert PAD not in ids and BOS not in ids


def test_the_typed_alphabet_gives_each_kind_a_disjoint_region():
    typed = CODECS["token_typed"]
    sets = legal_symbols(typed)
    coord, delta = set(sets["coord"].tolist()), set(sets["delta"].tolist())
    assert not (coord & delta)
    assert set(sets["opcode"].tolist()).isdisjoint(coord)


def test_the_untyped_alphabet_does_not():
    plain = legal_symbols(CODECS["token"])
    assert set(plain["coord"].tolist()) == set(plain["delta"].tolist())


def test_the_xf_field_is_the_narrowest_in_every_alphabet_that_has_widths():
    """Eight elements of D4 against 256 byte values -- the field with the most
    for an alphabet to give away, and the one §7.2's ceiling is read on."""
    for name in ("byte", "token", "token_typed"):
        sets = legal_symbols(CODECS[name])
        assert len(sets["xf"]) == 8
        assert len(sets["coord"]) == 256


def test_a_relative_codec_is_attributed_through_its_inner_alphabet():
    assert (legal_symbols(CODECS["byte_delta"])["opcode"]
            == legal_symbols(CODECS["byte"])["opcode"]).all()


def test_a_legacy_width_token_codec_still_states_its_own_opcode_region():
    """A pre-v2 checkpoint is 11 slots wide, and only the slots an opcode
    actually occupies are legal: a reserved slot is not a bytecode byte."""
    legacy = TokenCodec(opcode_slots=11)
    assert len(legal_symbols(legacy)["opcode"]) == min(len(SPECS), 11)


@pytest.mark.parametrize("name", ["byte", "token", "token_typed", "bit"])
def test_the_cuts_reconcile_with_each_other_and_with_the_model_s_own_cost(name):
    codec = CODECS[name]
    model = _model(codec)
    reading = attribute(model, CORPUS, codec, max_len=256, batch_size=2)

    checked = reconcile(reading)
    assert checked["cut_spread"] < 1e-6

    loader = [collate([ProgramDataset(CORPUS, codec, 256)[i]
                       for i in range(len(CORPUS))])]
    loader = _Loader(loader, ProgramDataset(CORPUS, codec, 256))
    reference = float(per_program_bits(model, loader).mean())
    assert reading["bits_per_drawing"] == pytest.approx(reference, rel=1e-6)


class _Loader:
    """The smallest thing `per_program_bits` accepts: batches and a dataset."""

    def __init__(self, batches, dataset):
        self.batches, self.dataset = batches, dataset

    def __iter__(self):
        return iter(self.batches)


@pytest.mark.parametrize("name", ["byte", "token", "bit"])
def test_the_field_totals_sum_to_the_reported_total(name):
    """At 1e-7, not exactly.

    `bits_per_drawing` accumulates per program so that two arms can be paired,
    and the cuts accumulate through `bincount`; the two add the same float32
    values in different orders. The *cuts* agree with each other to 1e-12
    because they share the summation, and that is the check that would catch a
    lost byte. This one would only ever catch float associativity.
    """
    codec = CODECS[name]
    reading = attribute(_model(codec), CORPUS, codec, max_len=256, batch_size=2)
    for cut in CUTS:
        total = sum(row["bits_per_drawing"] for row in reading["cuts"][cut].values())
        assert total == pytest.approx(reading["bits_per_drawing"], rel=1e-7)


def test_the_bitplane_cut_appears_only_where_a_byte_is_several_symbols():
    assert BITPLANE not in attribute(_model(CODECS["byte"]), CORPUS,
                                     CODECS["byte"], max_len=256,
                                     batch_size=2)["cuts"]
    wide = attribute(_model(CODECS["bit"]), CORPUS, CODECS["bit"], max_len=256,
                     batch_size=2)["cuts"]
    assert BITPLANE in wide
    assert all(name.endswith("]") for name in wide[BITPLANE])
    assert sum(row["symbols"] for row in wide[BITPLANE].values()) == \
        sum(row["symbols"] for row in wide["field"].values())


def test_waste_is_never_negative_and_never_exceeds_the_bits_spent_there():
    """`-log2 P(legal)` is at most `-log2 P(the symbol that was there)`, because
    the symbol that was there is in the legal set. A waste above the spend would
    mean the set excluded the truth."""
    for name in ("byte", "token", "token_typed", "bit"):
        codec = CODECS[name]
        reading = attribute(_model(codec), CORPUS, codec, max_len=256, batch_size=2)
        for field, row in reading["cuts"]["field"].items():
            assert row["waste_per_drawing"] >= -1e-9, field
            assert row["waste_per_drawing"] <= row["bits_per_drawing"] + 1e-9, field


def test_the_vocabularies_are_closed_and_named():
    assert set(_CLASS_KEYS()) == set(POSITION_CLASSES)
    assert len(set(FIELDS)) == len(FIELDS)
    assert len(set(SLOTS)) == len(SLOTS)
    assert len(set(OPCODES)) == len(OPCODES)
    assert set(OPCODES) - {"unparsed"} == {s.mnemonic for s in SPECS.values()}
    assert {k.name.lower() for k in Kind} <= set(FIELDS) | {"coord", "delta"}


def _CLASS_KEYS():
    from dm.eval.attribution import _CLASS_OF_FIELD
    return set(_CLASS_OF_FIELD.values()) | set(POSITION_CLASSES)
