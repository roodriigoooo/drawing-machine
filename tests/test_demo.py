"""The demo's evidence boundary, checked without a board attached.

A demo page is the one artifact in this project whose failure mode is not a
wrong number: it is a *right-looking picture with the wrong provenance*. Every
test below exists to make one such substitution impossible.

`docs/claim4-bringup.md` §8 wrote the rule these enforce — *a visual demo must
render the parsed device trace; rendering a host VM trace and placing a "Pico"
badge beside it would demonstrate the renderer, not Claim 4* — and §11.3 item 4
asks specifically for a test that makes host-VM rendering fail if it is ever
used as the visual source.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from dm.demo import capture, page
from dm.demo.align import align, verify
from dm.demo.device import DeviceRefusal, NativeRehearsal, _canvas, events
from dm.demo.record import SCHEMA, DemoRecord
from dm.demo.router import SYNONYMS, route
from scripts.conformance import ONE

CLASSES = ("cat", "bus", "flower", "sailboat", "bicycle")


# --- the router says what it did -------------------------------------------


def test_a_trained_category_routes_to_its_own_index():
    for index, name in enumerate(CLASSES):
        routing = route(name, CLASSES)
        assert (routing.rule, routing.klass, routing.index) == ("exact", name, index)


@pytest.mark.parametrize("word,expected", [("kitten", "cat"), ("bike", "bicycle"),
                                           ("BOAT", "sailboat"), ("roses", "flower"),
                                           ("  coach ", "bus")])
def test_a_synonym_routes_and_says_so(word, expected):
    routing = route(word, CLASSES)
    assert routing.klass == expected
    assert routing.rule == "synonym"
    # The substitution has to be visible: a page that silently drew a cat for
    # "kitten" would be claiming a vocabulary the checkpoint does not have.
    assert expected in routing.reason and "not a trained category" in routing.reason


@pytest.mark.parametrize("word", ["tiger", "dog", "helicopter", "", "   ", "!!"])
def test_an_unknown_word_is_refused_and_the_refusal_names_every_category(word):
    routing = route(word, CLASSES)
    assert not routing.routed and routing.index is None
    for name in CLASSES:
        assert name in routing.reason


def test_a_synonym_for_an_untrained_category_does_not_resolve():
    """The checkpoint's class list is the authority, not this file's table."""
    routing = route("kitten", ("bus", "flower"))
    assert not routing.routed


def test_the_synonym_table_is_disjoint():
    seen: dict[str, str] = {}
    for klass, words in SYNONYMS.items():
        for word in (klass, *words):
            assert seen.setdefault(word, klass) == klass


# --- the badge cannot be earned by the host --------------------------------


def _record(source: str, matches: bool) -> DemoRecord:
    return DemoRecord(
        word="cat", routing=route("cat", CLASSES).as_dict(),
        checkpoint={"name": "test", "params": 825344, "codec": "byte",
                    "shape": "square", "steps": 24000, "classes": list(CLASSES)},
        sampled={"bytecode_hex": "0102030400", "bytes": 5, "instructions": 2,
                 "disassembly": "MOVE 3 4\nHALT", "class_index": 0, "seed": 0,
                 "top_k": 80, "temperature": 1.0, "truncated": False,
                 "sample_seconds": 0.1, "reference_valid": True},
        device={"matches_reference": matches, "steps": 2, "halted": True,
                "strokes": 0, "regions": 0, "discs": 0, "faults": [],
                "geometry": {"strokes": [], "regions": [], "discs": []},
                "trace_text": "", "alignment": None},
        source=source,
    )


@pytest.mark.parametrize("source,matches,expected", [
    ("rp2040", True, True),
    ("rp2040", False, False),
    ("native", True, False),
    ("native", False, False),
])
def test_the_silicon_badge_requires_both_a_board_and_an_exact_match(source, matches, expected):
    assert _record(source, matches).on_silicon is expected


def test_the_page_reads_the_badge_from_the_record_and_computes_nothing():
    """There must be exactly one source of `on_silicon` and it is Python.

    If the page ever recomputed the badge -- from the source string, from a
    fault list, from anything -- a change on one side could put a silicon badge
    on a host drawing. The template is allowed to *read* the flag and nothing
    else.
    """
    template = page._TEMPLATE
    assert "rec.on_silicon" in template
    # No JavaScript may reconstruct it from the parts.
    assert not re.search(r'source\s*===\s*"rp2040"\s*&&', template)


def test_a_native_record_never_reaches_the_gallery_with_a_badge():
    blob = capture.view(_record("native", True))
    assert blob["on_silicon"] is False
    html = page.render("gallery", checkpoint=blob["checkpoint"], records=[blob])
    payload = json.loads(re.search(r"const DATA = (\{.*\});", html).group(1))
    assert payload["records"][0]["on_silicon"] is False


def test_the_gallery_is_self_contained():
    """No network dependency: it has to work after the boards are in a drawer."""
    blob = capture.view(_record("rp2040", True))
    html = page.render("gallery", checkpoint=blob["checkpoint"], records=[blob])
    # The SVG namespace is an identifier, not a fetch; nothing else may name a host.
    stripped = html.replace("http://www.w3.org/2000/svg", "")
    assert "http://" not in stripped and "https://" not in stripped
    assert "<script src=" not in html and "<link " not in html
    assert "fetch(" not in html and "XMLHttpRequest" not in html


# --- a device frame that disagrees with the reference is refused ------------


class _Wire:
    """A scripted device whose trace can be perturbed one line at a time."""

    source = "rp2040"
    device_path = None

    def __init__(self, program: bytes, corrupt: bool = False):
        self.program, self.corrupt = program, corrupt

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def draw(self, program: bytes):
        with NativeRehearsal() as native:
            lines = []
            for event in native.draw(program):
                blob = event.as_dict()
                if blob["t"] == "done":
                    lines = blob["trace_text"].splitlines()
        if self.corrupt:
            for i, line in enumerate(lines):
                if line.startswith("p "):
                    x, y = line.split()[1:]
                    lines[i] = f"p {int(x) + 4096} {y}"
                    break
        yield from events(_tagged(lines), program, 0.0)


def _tagged(lines):
    for line in lines:
        tag, _, rest = line.partition(" ")
        yield tag, rest.split(), line


def test_a_device_frame_equal_to_the_reference_produces_a_record():
    program = bytes([1, 10, 10, 2, 60, 60, 2, 90, 30, 0])
    outcome = list(capture.draw(_Checkpoint(), _Wire(program), "cat", 0))[-1]
    assert outcome["t"] == "record"
    assert outcome["record"]["on_silicon"] is True


def test_one_corrupted_coordinate_refuses_the_page():
    """A single pixel of divergence must lose the drawing, not caveat it."""
    program = bytes([1, 10, 10, 2, 60, 60, 2, 90, 30, 0])
    outcome = list(capture.draw(_Checkpoint(), _Wire(program, corrupt=True), "cat", 0))[-1]
    assert outcome["t"] == "error"
    assert "refuses" in outcome["message"]


class _Checkpoint:
    """The smallest object `capture.draw` needs, with a fixed program."""

    classes = CLASSES
    n_classes = len(CLASSES)
    record = {"name": "test", "config": {"codec": "byte", "shape": "square",
                                         "steps": 24000, "extra": {"rdp_eps": 4.0}},
              "model": {"params": 825344, "n_classes": 5}}

    def as_dict(self):
        return {"name": "test", "params": 825344, "codec": "byte",
                "shape": "square", "steps": 24000, "classes": list(CLASSES)}


@pytest.fixture(autouse=True)
def _fixed_sample(monkeypatch):
    """`capture.draw` samples; these tests are about the wire, not the model."""
    from dm.demo import sample as sample_mod

    class _Sampled:
        program = bytes([1, 10, 10, 2, 60, 60, 2, 90, 30, 0])
        class_index = 0
        seed = 0

        def as_dict(self):
            from dm.isa.asm import disassemble
            return {"bytecode_hex": self.program.hex(), "bytes": len(self.program),
                    "instructions": 4, "disassembly": disassemble(self.program),
                    "class_index": 0, "seed": 0, "top_k": 80, "temperature": 1.0,
                    "truncated": False, "sample_seconds": 0.01,
                    "reference_valid": True}

    monkeypatch.setattr(capture, "sample", lambda *a, **k: _Sampled())
    monkeypatch.setattr(sample_mod, "sample", lambda *a, **k: _Sampled())


# --- fixed point, and the alignment that is checked rather than trusted -----


def test_the_wire_scale_is_the_curve_denominator():
    assert ONE == 4096
    assert _canvas(ONE) == 1.0
    assert _canvas(ONE * 255) == 255.0


def test_the_alignment_is_discarded_when_the_device_disagrees():
    program = bytes([1, 10, 10, 2, 60, 60, 2, 90, 30, 0])
    predicted = align(program)
    assert predicted is not None
    truthful = [{"points": [[0, 0]] * n} for n in predicted.stroke_points]
    assert verify(predicted, truthful) is predicted
    # One point too many anywhere and the highlight is withdrawn rather than
    # shown one instruction out of step.
    lying = [{"points": [[0, 0]] * (n + 1)} for n in predicted.stroke_points]
    assert verify(predicted, lying) is None


def test_the_transform_tier_is_refused_rather_than_guessed():
    from dm.isa.spec import Op
    assert align(bytes([int(Op.XFORM), 0, 0, 0, int(Op.ENDX), int(Op.HALT)])) is None
    assert align(bytes([int(Op.CALL), 0, int(Op.HALT)])) is None


# --- the wire's framing, on a pipe --------------------------------------


def _stream(text: str, batch: int = 0):
    """`pico.stream_pass` over a pipe, so the framing runs without a board."""
    import os as _os

    from scripts import pico

    read_fd, write_fd = _os.pipe()
    _os.write(write_fd, text.encode())
    _os.close(write_fd)
    try:
        return list(pico.stream_pass(read_fd, timeout_s=2.0, batch=batch))
    finally:
        _os.close(read_fd)


def test_the_stream_ignores_everything_before_its_own_begin():
    """A frame from the previous word can still be draining out of the adapter.

    `scripts/pico.py` already refuses to match another batch's frame for a
    measurement; a live view that took the first bytes it saw would animate the
    previous drawing under the new word.
    """
    text = ("p 1 1\ne stroke 2 1\nEND 1\n"      # the tail of batch 0
            "BEGIN 1\nfrac_bits 12\np 9 9\nEND 1\n")
    tags = [tag for tag, _, _ in _stream(text, batch=1)]
    assert tags == ["frac_bits", "p", "END"]


def test_the_stream_holds_a_partial_line_until_its_newline():
    """Half of `p 12345 67890` is a coordinate that was never sent."""
    import os as _os

    from scripts import pico

    read_fd, write_fd = _os.pipe()
    _os.write(write_fd, b"BEGIN 0\np 40960 8192\np 4096")
    stream = pico.stream_pass(read_fd, timeout_s=2.0, batch=0)
    assert next(stream) == ("p", ["40960", "8192"], "p 40960 8192")
    _os.write(write_fd, b"0 4096\nEND 1\n")
    assert next(stream) == ("p", ["40960", "4096"], "p 40960 4096")
    assert next(stream)[0] == "END"
    _os.close(write_fd)
    _os.close(read_fd)


def test_the_stream_carries_the_raw_line_beside_the_parsed_one():
    """The page shows the actual UART record, not a re-rendering of it."""
    events = _stream("BEGIN 0\nfrac_bits 12\np 4096 8192\nEND 1\n")
    assert ("p", ["4096", "8192"], "p 4096 8192") in events


def test_the_remount_prompt_never_tells_the_operator_to_unplug(monkeypatch, capsys):
    """The prompt for an automatic return must not ask for a manual one.

    `docs/claim4-bringup.md` §8: unplugging in response to the generic BOOTSEL
    prompt during an automatic return turns readiness into `ENOENT`. A run that
    prints that instruction *while* the board is coming back by itself is a run
    telling its operator to break it.
    """
    from scripts import pico

    monkeypatch.setattr(pico, "BOOTSEL_VOLUME", Path("/nonexistent-rpi-rp2"))

    monkeypatch.setattr(pico, "_DELIVERED_A_UF2", False)
    with pytest.raises(pico.PicoError):
        pico.bootsel_volume(timeout_s=0.0)
    first = capsys.readouterr().out
    assert "hold the Pico's BOOTSEL button" in first

    monkeypatch.setattr(pico, "_DELIVERED_A_UF2", True)
    with pytest.raises(pico.PicoError):
        pico.bootsel_volume(timeout_s=0.0)
    later = capsys.readouterr().out
    assert "do not unplug" in later
    assert "hold" not in later.lower() and "plug in" not in later


def test_a_record_from_another_schema_is_refused_rather_than_read():
    blob = _record("rp2040", True).as_dict()
    blob["schema"] = SCHEMA + 1
    with pytest.raises(ValueError):
        DemoRecord.from_dict(blob)


# --- the still figures are drawn from the wire, like every other surface ----


def _wire_record(**overrides) -> DemoRecord:
    """A record with real device geometry, produced without a board."""
    program = bytes([1, 10, 10, 2, 60, 60, 2, 90, 30, 0])
    outcome = list(capture.draw(_Checkpoint(), _Wire(program), "cat", 0))[-1]
    blob = dict(outcome["record"])
    blob.pop("on_silicon")
    blob.update(overrides)
    return DemoRecord.from_dict(blob)


def _dark(image, x0: int = 0, y0: int = 0, x1=None, y1=None) -> int:
    box = image.crop((x0, y0, x1 or image.width, y1 or image.height))
    return sum(1 for pixel in box.convert("L").getdata() if pixel < 200)


def test_a_figure_cannot_be_drawn_from_a_bench_with_no_board():
    """The sheets are captioned "the RP2040 executed"; a rehearsal earns none.

    `docs/demo.md` §1 refuses the silicon badge to a native record on the page.
    A figure is the same claim printed on paper, so it is refused in the same
    place -- before anything is drawn, not with a caveat in the caption.
    """
    from dm.demo import figures

    with pytest.raises(ValueError):
        figures.gallery([_wire_record(source="native")])


def test_a_figure_draws_the_wire_and_never_re_renders_the_program():
    """Strip the device geometry and the cell is empty, program intact.

    This is the host-fallback test in picture form: the bytecode is still there
    and the host reference VM could draw it, and the sheet draws nothing.
    """
    from dm.demo import figures

    drawn = figures.gallery([_wire_record()])
    silent = _wire_record()
    silent.device = {**silent.device,
                     "geometry": {"strokes": [], "regions": [], "discs": []}}
    assert silent.sampled["bytes"] > 0            # the program is untouched
    empty = figures.gallery([silent])

    # Right of the row label, above the caption: only cell ink lives here.
    band = dict(x0=figures.WIDTH // 2, y0=figures.MARGIN,
                y1=figures.MARGIN + int(figures.MAX_CELL))
    assert _dark(drawn, **band) > 0
    assert _dark(empty, **band) == 0


def test_the_prefix_split_falls_only_where_the_instruction_stream_divides():
    """The same rule the live page applies, so the two never disagree."""
    from dm.demo import figures

    points = [(0, 0), (1, 1), (2, 2), (3, 3)]
    runs = figures._runs(points, 0, cut=2, mapping=[0, 1, 2, 3])
    assert [given for _, given in runs] == [True, False]
    # They share the boundary point and nothing more: the pen does not lift
    # where the human stopped, and no segment is drawn twice in two colours.
    assert runs[0][0][-1] == runs[1][0][0]
    assert len(runs[0][0]) + len(runs[1][0]) == len(points) + 1
    # With no alignment there is no split to make, and none is invented.
    assert figures._runs(points, 0, cut=2, mapping=[]) == [(points, False)]


def test_the_prefix_figure_draws_the_human_ink_in_the_colour_the_page_uses():
    from dm.demo import figures

    record = _wire_record(prefix={"source": "quickdraw valid split", "position": 0,
                                  "class_index": 0, "seed": 0, "bytes": 3,
                                  "instructions": 2, "of_instructions": 4,
                                  "fraction": 0.4, "truth_hex": "0102030400"})
    sheet = figures.prefix([record]).convert("RGB")
    pixels = list(sheet.getdata())
    assert any(r > 140 and r - g > 60 and r - b > 60 for r, g, b in pixels)
    assert any(max(p) < 90 for p in pixels)
