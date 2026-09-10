"""Trace -> pixels / SVG.

The raster path is the one the metrics use, so it is supersampled and
deterministic. The SVG path is for figures and for eyeballing datasets.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from ..isa.spec import CANVAS
from .interp import Trace

SUPERSAMPLE = 4


def rasterize(trace: Trace, size: int = CANVAS, supersample: int = SUPERSAMPLE) -> np.ndarray:
    """Ink coverage in [0, 1], shape (size, size). 0 = paper, 1 = ink."""
    hi = size * supersample
    scale = hi / CANVAS
    img = Image.new("L", (hi, hi), 0)
    draw = ImageDraw.Draw(img)

    for region in trace.regions:
        draw.polygon([(x * scale, y * scale) for x, y in region.points], fill=255)

    for stroke in trace.strokes:
        pts = [(x * scale, y * scale) for x, y in stroke.points]
        w = max(1, int(round(stroke.width * scale)))
        draw.line(pts, fill=255, width=w, joint="curve")

    for disc in trace.discs:
        cx, cy, r = disc.cx * scale, disc.cy * scale, disc.r * scale
        draw.ellipse(
            [cx - r, cy - r, cx + r, cy + r],
            outline=255,
            width=max(1, int(round(disc.width * scale))),
        )

    if supersample > 1:
        img = img.resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def to_image(trace: Trace, size: int = CANVAS) -> Image.Image:
    """Black ink on white, for saving as PNG."""
    ink = rasterize(trace, size)
    return Image.fromarray(((1.0 - ink) * 255).astype(np.uint8), mode="L")


def to_svg(trace: Trace, size: int = CANVAS, ink: str = "black",
           background: str | None = "white") -> str:
    """One trace as a standalone SVG.

    `ink` and `background` exist so a figure can **overlay** several traces of
    one program in one frame — render the spans that are copies of each other in
    one colour and the rest in another, with `background=None` on every layer but
    the first. That is the only way to show which bytes of a constructed scene are
    the planted orbit, and it is done here rather than by rewriting this
    function's output in a script, because a caller editing `stroke="black"` out
    of an SVG string is a second renderer that will drift from this one. `ink`
    colours the fill as well as the stroke: a filled region is ink too.
    """
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {CANVAS} {CANVAS}">',
        *([f'<rect width="{CANVAS}" height="{CANVAS}" fill="{background}"/>']
          if background else []),
        f'<g fill="none" stroke="{ink}" stroke-linecap="round" stroke-linejoin="round">',
    ]
    for region in trace.regions:
        pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in region.points)
        parts.append(f'<polygon points="{pts}" fill="{ink}" stroke="none"/>')
    for stroke in trace.strokes:
        pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in stroke.points)
        parts.append(f'<polyline points="{pts}" stroke-width="{stroke.width}"/>')
    for disc in trace.discs:
        parts.append(
            f'<circle cx="{disc.cx:.2f}" cy="{disc.cy:.2f}" r="{disc.r}" '
            f'stroke-width="{disc.width}"/>'
        )
    parts.append("</g></svg>")
    return "\n".join(parts)
