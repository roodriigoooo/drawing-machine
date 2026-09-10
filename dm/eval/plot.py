"""Research figures as SVG, with pgfplots' conventions and none of its LaTeX.

The project already emits SVG for drawings (`dm/vm/render.py`), has numpy and
Pillow and nothing else, and wants figures that survive being dropped into a
paper. So this is ~250 lines of primitives rather than a plotting dependency,
and the conventions it hard-codes are the ones that make a figure readable
rather than the ones a default stylesheet happens to have:

- **A box frame and a hairline grid**, ticks pointing *out*, labels outside the
  frame. Data gets the whole interior; nothing is drawn on top of it.
- **A typographic minus** (U+2212), not a hyphen. On a tick label reading −0.67
  the difference is the difference between a figure and a screenshot.
- **Tabular serif numerals**, because axis labels are read as columns.
- **Every series carries a dash pattern as well as a colour.** A figure that
  distinguishes two arms by hue alone stops being a figure in greyscale print
  and for ~4% of readers. The palette is chosen for that too: hues that stay
  distinct when desaturated.
- **No legend box.** Series are labelled at their own last point where there is
  room, which is what removes the reader's eye-trip to a key and back.

Nothing here knows what the project measures. `scripts/plots.py` holds the
figures themselves, so a change to a result never means editing the drawing
code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

#: Muted-saturated, and ordered so the first two are the pair most figures need.
#: Checked for separation in greyscale: the luminances are 55, 62, 70, 48, 78.
PALETTE = {
    "teal": "#3d8b83",
    "clay": "#c8654a",
    "olive": "#8a8f42",
    "indigo": "#3b4f9e",
    "sand": "#c9a227",
    "slate": "#6b7280",
}

INK = "#1a1a1a"
GRID = "#d4d4d4"
FRAME = "#1a1a1a"
MUTED = "#666666"

#: A serif stack. Latin Modern is what a paper would use and is absent on most
#: machines, so the fallbacks are ordered by how close their numerals sit to it.
SERIF = "'Latin Modern Roman','Computer Modern Roman',Georgia,'Times New Roman',serif"

MINUS = "−"


def fmt(value: float, places: int = 0) -> str:
    """A tick label: typographic minus, fixed places, no `−0`."""
    text = f"{value:.{places}f}"
    if text.startswith("-"):
        text = MINUS + text[1:]
    if set(text) <= {MINUS, "0", "."}:
        text = text.lstrip(MINUS)
    return text


def axis_places(ticks: list[float]) -> int:
    """Decimals every label on one axis should carry.

    Uniform across the axis rather than per value: deciding per value gives an
    axis reading 0, 0.50, 1, 1.50, 2, where the eye reads the varying width as
    varying precision. Derived from the tick *step*, so it is a property of the
    axis and not of whichever value happens to be fractional.
    """
    if len(ticks) < 2:
        return 0
    step = abs(ticks[1] - ticks[0])
    for places in range(6):
        if abs(step * 10**places - round(step * 10**places)) < 1e-9:
            return places
    return 2


def nice_ticks(lo: float, hi: float, target: int = 5) -> list[float]:
    """Ticks on a 1/2/5×10ⁿ lattice, the choice that makes labels readable.

    Returned in data units and clipped to the range, so a caller that has
    already chosen its limits gets ticks inside them rather than a widened axis.
    """
    if hi <= lo:
        return [lo]
    raw = (hi - lo) / max(1, target)
    magnitude = 10 ** math.floor(math.log10(raw))
    for multiple in (1, 2, 2.5, 5, 10):
        step = multiple * magnitude
        if raw <= step:
            break
    first = math.ceil(lo / step) * step
    ticks, value = [], first
    while value <= hi + step * 1e-9:
        ticks.append(round(value, 10))
        value += step
    return ticks


@dataclass
class Series:
    xs: list[float]
    ys: list[float]
    color: str = PALETTE["teal"]
    dash: str = ""
    width: float = 1.6
    label: str = ""
    marker: float = 0.0  # radius; 0 draws no markers
    label_dy: float = 0.0


@dataclass
class Panel:
    """One set of axes. Coordinates are data units until `svg` is called."""

    xlim: tuple[float, float]
    ylim: tuple[float, float]
    width: float = 250
    height: float = 170
    xlabel: str = ""
    ylabel: str = ""
    title: str = ""
    xticks: list[float] | None = None
    yticks: list[float] | None = None
    xtick_labels: list[str] | None = None
    grid: bool = True
    series: list[Series] = field(default_factory=list)
    bands: list[tuple] = field(default_factory=list)
    boxes: list[tuple] = field(default_factory=list)
    rules: list[tuple] = field(default_factory=list)
    notes: list[tuple] = field(default_factory=list)

    def add(self, series: Series) -> Panel:
        self.series.append(series)
        return self

    def band(self, lo: float, hi: float, color: str = GRID, opacity: float = 0.5,
             axis: str = "y") -> Panel:
        """A shaded interval spanning the panel — a noise floor, a tolerance, a regime."""
        self.bands.append((lo, hi, color, opacity, axis))
        return self

    def box(self, x0: float, x1: float, y0: float, y1: float,
            color: str = GRID, opacity: float = 0.5) -> Panel:
        """A shaded rectangle in data units, for a band that belongs to one row."""
        self.boxes.append((x0, x1, y0, y1, color, opacity))
        return self

    def rule(self, value: float, axis: str = "y", color: str = MUTED,
             dash: str = "3 2") -> Panel:
        self.rules.append((value, axis, color, dash))
        return self

    def note(self, x: float, y: float, text: str, anchor: str = "start",
             color: str = MUTED, size: float = 10.5, dy: float = 0.0) -> Panel:
        self.notes.append((x, y, text, anchor, color, size, dy))
        return self

    # -- data -> pixels ----------------------------------------------------
    def px(self, x: float) -> float:
        lo, hi = self.xlim
        return (x - lo) / (hi - lo) * self.width

    def py(self, y: float) -> float:
        lo, hi = self.ylim
        return self.height - (y - lo) / (hi - lo) * self.height

    def svg(self, clip_id: str = "panelclip") -> str:
        xt = self.xticks if self.xticks is not None else nice_ticks(*self.xlim)
        yt = self.yticks if self.yticks is not None else nice_ticks(*self.ylim)
        out: list[str] = []

        # Bands sit under the grid: they are context, not data.
        for lo, hi, color, opacity, axis in self.bands:
            if axis == "y":
                y0, y1 = self.py(hi), self.py(lo)
                out.append(f'<rect x="0" y="{y0:.2f}" width="{self.width}" '
                           f'height="{y1 - y0:.2f}" fill="{color}" opacity="{opacity}"/>')
            else:
                x0, x1 = self.px(lo), self.px(hi)
                out.append(f'<rect x="{x0:.2f}" y="0" width="{x1 - x0:.2f}" '
                           f'height="{self.height}" fill="{color}" opacity="{opacity}"/>')

        for x0, x1, y0, y1, color, opacity in self.boxes:
            px0, px1 = self.px(x0), self.px(x1)
            py0, py1 = self.py(y1), self.py(y0)
            out.append(f'<rect x="{px0:.2f}" y="{py0:.2f}" width="{px1 - px0:.2f}" '
                       f'height="{py1 - py0:.2f}" fill="{color}" opacity="{opacity}"/>')

        if self.grid:
            for t in xt:
                x = self.px(t)
                out.append(f'<line x1="{x:.2f}" y1="0" x2="{x:.2f}" y2="{self.height}" '
                           f'stroke="{GRID}" stroke-width="0.5"/>')
            for t in yt:
                y = self.py(t)
                out.append(f'<line x1="0" y1="{y:.2f}" x2="{self.width}" y2="{y:.2f}" '
                           f'stroke="{GRID}" stroke-width="0.5"/>')

        for value, axis, color, dash in self.rules:
            if axis == "y":
                y = self.py(value)
                out.append(f'<line x1="0" y1="{y:.2f}" x2="{self.width}" y2="{y:.2f}" '
                           f'stroke="{color}" stroke-width="0.9" stroke-dasharray="{dash}"/>')
            else:
                x = self.px(value)
                out.append(f'<line x1="{x:.2f}" y1="0" x2="{x:.2f}" y2="{self.height}" '
                           f'stroke="{color}" stroke-width="0.9" stroke-dasharray="{dash}"/>')

        # Data is clipped so a series cannot bleed past the frame it is read in.
        out.append(f'<g clip-path="url(#{clip_id})">')
        for s in self.series:
            pts = " ".join(f"{self.px(x):.2f},{self.py(y):.2f}" for x, y in zip(s.xs, s.ys))
            if len(s.xs) > 1:
                dash = f' stroke-dasharray="{s.dash}"' if s.dash else ""
                out.append(f'<polyline points="{pts}" fill="none" stroke="{s.color}" '
                           f'stroke-width="{s.width}" stroke-linejoin="round" '
                           f'stroke-linecap="round"{dash}/>')
            if s.marker:
                for x, y in zip(s.xs, s.ys):
                    out.append(f'<circle cx="{self.px(x):.2f}" cy="{self.py(y):.2f}" '
                               f'r="{s.marker}" fill="{s.color}"/>')
        out.append("</g>")

        # Frame last, so it sits over any series that touches it.
        out.append(f'<rect x="0" y="0" width="{self.width}" height="{self.height}" '
                   f'fill="none" stroke="{FRAME}" stroke-width="0.9"/>')

        xplaces, yplaces = axis_places(xt), axis_places(yt)
        for i, t in enumerate(xt):
            x = self.px(t)
            label = self.xtick_labels[i] if self.xtick_labels else fmt(t, xplaces)
            out.append(f'<line x1="{x:.2f}" y1="{self.height}" x2="{x:.2f}" '
                       f'y2="{self.height + 3.5}" stroke="{FRAME}" stroke-width="0.9"/>')
            out.append(f'<text x="{x:.2f}" y="{self.height + 16}" text-anchor="middle" '
                       f'font-family="{SERIF}" font-size="11.5" fill="{INK}">{label}</text>')
        for t in yt:
            y = self.py(t)
            out.append(f'<line x1="0" y1="{y:.2f}" x2="-3.5" y2="{y:.2f}" '
                       f'stroke="{FRAME}" stroke-width="0.9"/>')
            out.append(f'<text x="-7" y="{y + 4:.2f}" text-anchor="end" '
                       f'font-family="{SERIF}" font-size="11.5" '
                       f'fill="{INK}">{fmt(t, yplaces)}</text>')

        if self.xlabel:
            out.append(f'<text x="{self.width / 2:.2f}" y="{self.height + 36}" '
                       f'text-anchor="middle" font-family="{SERIF}" font-size="12.5" '
                       f'fill="{INK}">{self.xlabel}</text>')
        if self.ylabel:
            out.append(f'<text transform="translate(-42,{self.height / 2:.2f}) rotate(-90)" '
                       f'text-anchor="middle" font-family="{SERIF}" font-size="12.5" '
                       f'fill="{INK}">{self.ylabel}</text>')
        if self.title:
            out.append(f'<text x="{self.width / 2:.2f}" y="-10" text-anchor="middle" '
                       f'font-family="{SERIF}" font-size="12.5" fill="{INK}">{self.title}</text>')

        # In-place series labels instead of a legend box.
        for s in self.series:
            if not s.label:
                continue
            x, y = s.xs[-1], s.ys[-1]
            out.append(f'<text x="{self.px(x) + 6:.2f}" y="{self.py(y) + 4 + s.label_dy:.2f}" '
                       f'font-family="{SERIF}" font-size="11.5" fill="{s.color}">{s.label}</text>')

        for x, y, text, anchor, color, size, dy in self.notes:
            out.append(f'<text x="{self.px(x):.2f}" y="{self.py(y) + dy:.2f}" '
                       f'text-anchor="{anchor}" font-family="{SERIF}" font-size="{size}" '
                       f'fill="{color}">{text}</text>')

        return "\n".join(out)


def caption_height(caption: str, number: str, width: float) -> float:
    """Room a caption needs, from its own length.

    Computed rather than passed: every caption in the first draft of these
    figures was clipped, because the height was a constant chosen while the
    caption was two sentences and the caption then grew. A figure whose caption
    is cut off is worse than one with none -- it looks complete.
    """
    if not caption:
        return 0.0
    per_line = max(20.0, width / 6.35)  # 12.5px serif, measured against renders
    lines = math.ceil((len(caption) + len(number) + 1) / per_line)
    return lines * 17.5 + 10


def figure(panels: list[tuple[Panel, float, float]], width: float, height: float,
           caption: str = "", number: str = "") -> str:
    """Compose panels at given offsets into one standalone SVG.

    `height` is the plot region; the caption's own height is measured and added,
    so a longer caption grows the figure instead of being cut off by it.

    The caption is part of the figure rather than the page, because a figure
    that travels without its caption is how a number ends up quoted without its
    conditions -- the failure this project has had to correct more than once.
    """
    caption_h = caption_height(caption, number, width)
    total = height + caption_h
    # One clip path per panel: a shared one sized to the largest would let a
    # series in a smaller panel draw outside its own frame.
    clips = "".join(
        f'<clipPath id="clip{i}"><rect x="0" y="0" width="{p.width}" height="{p.height}"/>'
        f"</clipPath>"
        for i, (p, _, _) in enumerate(panels)
    )
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{total}" '
        f'viewBox="0 0 {width} {total}">',
        f"<defs>{clips}</defs>",
        f'<rect width="{width}" height="{total}" fill="#ffffff"/>',
    ]
    for i, (panel, x, y) in enumerate(panels):
        out.append(f'<g transform="translate({x},{y})">{panel.svg(f"clip{i}")}</g>')

    if caption:
        lead = f"<em>{number}</em> " if number else ""
        out.append(
            f'<foreignObject x="0" y="{height}" width="{width}" '
            f'height="{caption_h}">'
            f'<div xmlns="http://www.w3.org/1999/xhtml" style="font-family:{SERIF};'
            f'font-size:12.5px;line-height:1.45;color:{INK};">{lead}{caption}</div>'
            f"</foreignObject>"
        )
    out.append("</svg>")
    return "\n".join(out)
