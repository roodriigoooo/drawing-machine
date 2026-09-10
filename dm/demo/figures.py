"""The still figures, drawn the way a paper draws them.

Four of the six things `docs/media/README.md` lists are not photographs and not
screenshots: they are figures. A screenshot of an interactive page is a picture
of a browser -- it carries the browser's chrome, the host's theme, the scrollbar
position and whatever the window happened to be sized to on the day. A figure
carries the drawing and its caption, and it is byte-identical every time it is
built.

So the gallery sheet, the novelty sheet, the prefix sheet and the program sheet
are rendered here, straight from the records, as paper: white ground, hairline
ink, a serif caption under each. That removes four capture steps from the bench
runbook and leaves two things a human still has to do -- record the live page
once, and photograph the boards once.

**Provenance is kept visible in the ink, not in a caption alone.** Black is
geometry the device returned. Grey is a training drawing, rendered on the host
from the corpus. Red is the held-out human prefix the model was asked to finish.
Nothing on these sheets is host-rendered *device* geometry: the strokes come out
of `record.device.geometry`, which is the parsed wire and nothing else.
"""

from __future__ import annotations

import statistics
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .record import DemoRecord

CANVAS = 256

#: Everything below is laid out in final pixels and drawn at `SCALE`x, then
#: resampled down. That is what makes a sub-pixel hairline survive: a 1-unit
#: stroke in a 150 px cell is 0.6 px wide, which a direct rasteriser rounds to
#: 1 px of hard black and a downsample renders as the grey a pen leaves.
SCALE = 3
WIDTH = 1180
MARGIN = 30

PAPER = (255, 255, 255)
INK = (22, 22, 24)          # geometry the RP2040 sent back
TRAIN = (156, 156, 164)     # a training drawing, rendered on the host
GIVEN = (196, 58, 46)       # the held-out human prefix
MUTED = (124, 124, 132)
RULE = (208, 208, 212)

_SERIF = ["/System/Library/Fonts/Supplemental/Times New Roman.ttf",
          "/System/Library/Fonts/Supplemental/Georgia.ttf",
          "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"]
_ITALIC = ["/System/Library/Fonts/Supplemental/Times New Roman Italic.ttf",
           "/System/Library/Fonts/Supplemental/Georgia Italic.ttf",
           "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf"]
_MONO = ["/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Monaco.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"]

#: The figure numbers, in the order `docs/demo.md` §6 lists the six shots. The
#: two a human captures are 1 and 2; these four follow them.
NUMBERS = {"gallery": 3, "novelty": 4, "prefix": 5, "program": 6}

_cache: dict = {}


def _font(family: list[str], size: float):
    key = (id(family), size)
    if key not in _cache:
        for path in family:
            try:
                _cache[key] = ImageFont.truetype(path, int(round(size * SCALE)))
                break
            except OSError:
                continue
        else:                                  # no serif on this machine
            _cache[key] = ImageFont.load_default(int(round(size * SCALE)))
    return _cache[key]


# --- ink --------------------------------------------------------------------


def _place(box, pad=0.05):
    """Canvas -> box, one uniform scale, centred.

    The frame is the whole 256-unit canvas rather than each drawing's own
    bounding box: a per-drawing rescale would make a short wide bus and a tall
    flower the same size on the sheet and quietly misreport the geometry.
    """
    x, y, w, h = box
    side = min(w, h) * (1 - 2 * pad)
    s = side / CANVAS
    #: In supersampled units, like every other coordinate handed to `ImageDraw`.
    return s * SCALE, (x + (w - CANVAS * s) / 2) * SCALE, (y + (h - CANVAS * s) / 2) * SCALE


def _runs(points, base, cut, mapping):
    """One stroke split where the human's prefix stops and the model starts.

    Same rule the live page applies (`dm/demo/page.py`): the split can only fall
    where the instruction stream divides it, and the two runs share **exactly**
    the boundary point -- enough that the pen does not lift there, and not one
    point more, because a shared *segment* would be drawn twice and the second
    colour would take the credit for it.
    """
    if cut < 0 or not mapping:
        return [(points, False)]
    out, current, was = [], None, None
    for i, point in enumerate(points):
        index = mapping[base + i] if base + i < len(mapping) else None
        given = index is not None and index < cut
        if current is None or given != was:
            if current:
                current.append(point)
            current, was = [], given
            out.append((current, given))
        current.append(point)
    return out


def _paint(draw, record: DemoRecord, box, *, ink=INK, cut=-1, only_given=False,
           pad=0.05):
    """One record's device geometry into one cell."""
    geometry = record.device.get("geometry") or {}
    alignment = record.device.get("alignment") or {}
    mapping = alignment.get("instruction_of_point") or []
    s, ox, oy = _place(box, pad)

    def xy(points):
        return [(ox + x * s, oy + y * s) for x, y in points]

    for region in geometry.get("regions") or []:
        if not only_given:
            draw.polygon(xy(region["points"]), fill=ink)
    base = 0
    for stroke in geometry.get("strokes") or []:
        points = stroke["points"]
        width = max(2, int(round(max(1.0, stroke.get("width", 1)) * s)))
        for run, given in _runs(points, base, cut, mapping):
            if only_given and not given:
                continue
            if len(run) < 2:
                continue
            draw.line(xy(run), fill=GIVEN if given else ink, width=width,
                      joint="curve")
        base += len(points)
    for disc in geometry.get("discs") or []:
        if only_given:
            continue
        cx, cy, r = disc["cx"] * s, disc["cy"] * s, disc["r"] * s
        draw.ellipse([ox + cx - r, oy + cy - r, ox + cx + r, oy + cy + r],
                     outline=ink,
                     width=max(2, int(round(disc.get("width", 1) * s))))


def _polys(draw, strokes, box, colour, pad=0.05):
    """A training drawing, which is corpus geometry and is drawn in grey."""
    s, ox, oy = _place(box, pad)
    for stroke in strokes:
        points = [(ox + x * s, oy + y * s) for x, y in stroke["points"]]
        if len(points) > 1:
            draw.line(points, fill=colour,
                      width=max(2, int(round(1.6 * s))), joint="curve")


# --- type -------------------------------------------------------------------


def _text(draw, xy, string, font, fill=INK, anchor="la"):
    draw.text((xy[0] * SCALE, xy[1] * SCALE), string, font=font, fill=fill,
              anchor=anchor)


def _rule(draw, x0, y0, x1, y1, fill=RULE):
    draw.line([(x0 * SCALE, y0 * SCALE), (x1 * SCALE, y1 * SCALE)], fill=fill,
              width=max(1, SCALE // 2))


_MEASURE = ImageDraw.Draw(Image.new("RGB", (1, 1)))


def _wrap(string, font, width):
    lines, line = [], ""
    for word in string.split():
        probe = f"{line} {word}".strip()
        if _MEASURE.textlength(probe, font=font) / SCALE <= width and line:
            line = probe
        elif not line:
            line = word
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


CAPTION_SIZE, CAPTION_LEAD = 13.5, 19.0


def _caption_lines(text, width=WIDTH - 2 * MARGIN - 80):
    return _wrap(text, _font(_SERIF, CAPTION_SIZE), width)


def _caption(draw, y, text):
    """Centred serif, the way the figures it is imitating set theirs."""
    font = _font(_SERIF, CAPTION_SIZE)
    for line in _caption_lines(text):
        _text(draw, (WIDTH / 2, y), line, font, INK, anchor="ma")
        y += CAPTION_LEAD
    return y


def _sheet(height):
    image = Image.new("RGB", (WIDTH * SCALE, int(round(height * SCALE))), PAPER)
    return image, ImageDraw.Draw(image)


def _finish(image):
    return image.resize((WIDTH, image.height // SCALE), Image.Resampling.LANCZOS)


# --- selection --------------------------------------------------------------


def _classes(records):
    return list(records[0].checkpoint["classes"]) if records else []


def _grouped(records, classes, *, prefix: bool):
    """Silicon records per class, oldest first, prefix runs kept apart."""
    out = {name: [] for name in classes}
    for record in sorted(records, key=lambda r: r.created):
        if not record.on_silicon or bool(record.prefix) != prefix:
            continue
        name = record.routing.get("class")
        if name in out:
            out[name].append(record)
    return out


def _ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def _median_by(records, key):
    """The middle record, so a sheet shows a typical case and not the best one."""
    ranked = sorted(records, key=key)
    return ranked[len(ranked) // 2]


# --- the four sheets --------------------------------------------------------

#: A cell wider than this stops reading as a figure and starts reading as a
#: slideshow. When there are few enough drawings to exceed it the grid is capped
#: and centred instead of stretched.
MAX_CELL = 158


def _centre(block: float) -> float:
    return max(MARGIN, (WIDTH - block) / 2)


def gallery(records, columns: int = 6):
    classes = _classes(records)
    grouped = _grouped(records, classes, prefix=False)
    columns = min(columns, max((len(v) for v in grouped.values()), default=0))
    if not columns:
        raise ValueError("no silicon records to draw")

    label_w = 62
    cell = min(MAX_CELL, (WIDTH - 2 * MARGIN - label_w) / columns)
    left = _centre(label_w + columns * cell)
    rows = [name for name in classes if grouped[name]]
    mean_bytes = round(statistics.mean(r.sampled["bytes"] for v in grouped.values()
                                       for r in v))
    caption = (
        f"Figure {NUMBERS['gallery']}: Drawings the RP2040 executed, one row per "
        f"category the model was trained on. Each is a program of about {mean_bytes} "
        "bytes that the 825k-parameter model wrote and a $4 microcontroller ran out of "
        "SRAM; every stroke is geometry that came back over the wire and equalled the "
        "reference VM's trace exactly.")
    lines = _caption_lines(caption)
    height = MARGIN + len(rows) * cell + 26 + len(lines) * CAPTION_LEAD + MARGIN
    image, draw = _sheet(height)

    y = MARGIN
    for name in rows:
        _text(draw, (left + label_w - 12, y + cell / 2), name, _font(_ITALIC, 13),
              INK, anchor="rm")
        for column, record in enumerate(grouped[name][:columns]):
            _paint(draw, record, (left + label_w + column * cell, y, cell, cell))
        y += cell
    _caption(draw, y + 26, caption)
    return _finish(image)


def novelty(records):
    classes = _classes(records)
    grouped = _grouped(records, classes, prefix=False)
    picked = {}
    for name, group in grouped.items():
        measured = [r for r in group if r.novelty and not r.novelty.get("empty")]
        if measured:
            picked[name] = _median_by(measured, lambda r: r.novelty["floor_percentile"])
    if not picked:
        raise ValueError("no records carry a novelty measurement — run `demo.py novelty`")

    neighbours = min(len(r.novelty["neighbours"]) for r in picked.values())
    label_w, gap, gutter = 62, 10, 30
    cell = min(MAX_CELL, (WIDTH - 2 * MARGIN - label_w - gutter - neighbours * gap)
               / (1 + neighbours))
    block = label_w + (1 + neighbours) * cell + neighbours * gap + gutter
    left = _centre(block) + label_w
    right = left + cell + gutter
    row = cell + 22

    measured = [r for r in records if r.novelty and not r.novelty.get("empty")]
    verbatim = sum(bool(r.novelty.get("verbatim")) for r in measured)
    bank = max(r.novelty["bank_n"] for r in measured)
    copies = "None" if not verbatim else str(verbatim)
    caption = (
        f"Figure {NUMBERS['novelty']}: Each device drawing beside the training "
        "drawings it is closest to (grey, rendered on the host from the corpus), by "
        f"Chamfer distance over {bank:,} training drawings of its category. The "
        "percentile compares that distance with a held-out drawing baseline. "
        f"{copies} of the {len(measured)} captured programs "
        f"{'is' if verbatim == 1 else 'are'} a byte-for-byte copy of a training program. "
        "These checks do not establish absence of memorisation.")
    lines = _caption_lines(caption)
    height = MARGIN + 20 + len(picked) * row + 18 + len(lines) * CAPTION_LEAD + MARGIN
    image, draw = _sheet(height)

    _text(draw, (left + cell / 2, MARGIN), "on the device", _font(_ITALIC, 12), MUTED,
          anchor="ma")
    _text(draw, (right + (neighbours * (cell + gap) - gap) / 2, MARGIN),
          "nearest training drawings", _font(_ITALIC, 12), MUTED, anchor="ma")

    y = MARGIN + 20
    x = left + cell + gutter / 2
    _rule(draw, x, y + 2, x, y + len(picked) * row - 14)
    for name in classes:
        record = picked.get(name)
        if record is None:
            continue
        _text(draw, (left - 12, y + cell / 2), name, _font(_ITALIC, 13), INK, anchor="rm")
        _paint(draw, record, (left, y, cell, cell))
        _text(draw, (left + cell / 2, y + cell + 2),
              _ordinal(round(record.novelty["floor_percentile"])) + " pct",
              _font(_MONO, 9), MUTED, anchor="ma")
        for i, neighbour in enumerate(record.novelty["neighbours"][:neighbours]):
            box = (right + i * (cell + gap), y, cell, cell)
            _polys(draw, neighbour["strokes"], box, TRAIN)
            _text(draw, (box[0] + cell / 2, y + cell + 2),
                  f"{neighbour['distance']:.1f} px", _font(_MONO, 9), MUTED, anchor="ma")
        y += row
    _caption(draw, y + 18, caption)
    return _finish(image)


def prefix(records):
    classes = _classes(records)
    grouped = _grouped(records, classes, prefix=True)
    rows = [name for name in classes if grouped[name]]
    if not rows:
        raise ValueError("no held-out-prefix records — capture with --prefix-panel")
    pairs = min(2, max(len(grouped[name]) for name in rows))

    label_w, inner, between = 62, 12, 46
    cell = min(MAX_CELL,
               (WIDTH - 2 * MARGIN - label_w - pairs * inner - (pairs - 1) * between)
               / (2 * pairs))
    group_w = 2 * cell + inner
    block = label_w + pairs * group_w + (pairs - 1) * between
    left = _centre(block) + label_w
    row = cell + 10

    fraction = min(r.prefix["fraction"] for name in rows for r in grouped[name])
    caption = (
        f"Figure {NUMBERS['prefix']}: Finishing a drawing it has never seen. Red is a "
        f"prefix of a drawing from the validation split — its first {fraction:.0%} of "
        "instructions, cut at an instruction boundary and handed to the model; black is "
        "what the model wrote to finish it, executed on the device like every other "
        "program here. Selected continuations illustrate behaviour; they do not "
        "establish broad generalisation or exclude memorised fragments.")
    lines = _caption_lines(caption)
    height = MARGIN + 20 + len(rows) * row + 18 + len(lines) * CAPTION_LEAD + MARGIN
    image, draw = _sheet(height)

    def group_x(index):
        return left + index * (group_w + between)

    for index in range(pairs):
        _text(draw, (group_x(index) + cell / 2, MARGIN), "given", _font(_ITALIC, 12),
              MUTED, anchor="ma")
        _text(draw, (group_x(index) + cell + inner + cell / 2, MARGIN),
              "finished by the model", _font(_ITALIC, 12), MUTED, anchor="ma")

    y = MARGIN + 20
    for index in range(pairs):
        x = group_x(index) + cell + inner / 2
        _rule(draw, x, y + 2, x, y + len(rows) * row - 12)
    for name in rows:
        _text(draw, (left - 12, y + cell / 2), name, _font(_ITALIC, 13), INK, anchor="rm")
        for index in range(pairs):
            if index >= len(grouped[name]):
                continue
            record = grouped[name][index]
            cut = record.prefix["instructions"]
            _paint(draw, record, (group_x(index), y, cell, cell), cut=cut, only_given=True)
            _paint(draw, record, (group_x(index) + cell + inner, y, cell, cell), cut=cut)
        y += row
    _caption(draw, y + 18, caption)
    return _finish(image)


def program(records, record: DemoRecord | None = None):
    """One drawing beside the bytes that produced it.

    The record is the middle-length silicon program rather than the shortest or
    the most impressive: the sheet's claim is about what a *typical* output of
    this model is, and the whole listing has to fit beside the picture for the
    claim to be checkable by eye.
    """
    silicon = [r for r in records if r.on_silicon and not r.prefix]
    if not silicon:
        raise ValueError("no silicon records to draw")
    record = record or _median_by(silicon, lambda r: r.sampled["instructions"])
    listing = [line for line in record.sampled["disassembly"].split("\n") if line]
    hex_bytes = record.sampled["bytecode_hex"]

    drawing = 300
    left = MARGIN + 6
    right = left + drawing + 40
    text_w = WIDTH - MARGIN - right
    mono, small, label = _font(_MONO, 10.5), _font(_MONO, 9.5), _font(_ITALIC, 12)

    grouped = " ".join(hex_bytes[i:i + 2] for i in range(0, len(hex_bytes), 2))
    hex_lines = _wrap(grouped, mono, text_w)
    columns = 3 if len(listing) > 34 else 2
    per_column = -(-len(listing) // columns)

    lead = 14.5
    body = 18 + len(hex_lines) * lead + 30 + 18 + per_column * lead
    caption = (
        f"Figure {NUMBERS['program']}: The model's output is a program, not a picture. "
        f"{record.sampled['bytes']} bytes of bytecode (top right) disassemble to "
        f"{record.sampled['instructions']} instructions (bottom right); the RP2040 "
        "executed them out of SRAM and returned the trace the drawing at left is "
        f"reassembled from — {record.device['steps']:,} VM steps, one HALT, no fault.")
    lines = _caption_lines(caption)
    height = MARGIN + max(drawing + 24, body) + 26 + len(lines) * CAPTION_LEAD + MARGIN
    image, draw = _sheet(height)

    # The picture is centred against the listing rather than pinned to the top:
    # the two are one statement, and a drawing floating above a block of bytes
    # reads as two figures that happen to share a sheet.
    y = MARGIN + max(0, (body - drawing - 24) / 2)
    _paint(draw, record, (left, y, drawing, drawing), pad=0.02)
    _text(draw, (left + drawing / 2, y + drawing + 6),
          f"\u201c{record.word}\u201d — seed {record.sampled['seed']}", label, MUTED,
          anchor="ma")

    ty = MARGIN
    _text(draw, (right, ty), f"{record.sampled['bytes']} bytes written to the board",
          label, MUTED)
    ty += 18
    for line in hex_lines:
        _text(draw, (right, ty), line, mono, INK)
        ty += lead
    ty += 30
    _rule(draw, right, ty - 15, WIDTH - MARGIN, ty - 15)
    _text(draw, (right, ty), f"{len(listing)} instructions, as the VM read them",
          label, MUTED)
    ty += 18
    column_w = text_w / columns
    for index, line in enumerate(listing):
        column, offset = divmod(index, per_column)
        _text(draw, (right + column * column_w, ty + offset * lead), line, small, INK)

    _caption(draw, MARGIN + max(drawing + 24, body) + 26, caption)
    return _finish(image)


# --- driver -----------------------------------------------------------------

BUILDERS = {"gallery": gallery, "novelty": novelty, "prefix": prefix,
            "program": program}


def build(records, out_dir: Path, only=None, columns: int = 6) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, builder in BUILDERS.items():
        if only and name not in only:
            continue
        image = builder(records, columns) if name == "gallery" else builder(records)
        path = out_dir / f"{name}.png"
        image.save(path, optimize=True)
        written.append(path)
    return written
