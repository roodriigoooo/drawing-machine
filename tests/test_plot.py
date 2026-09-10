"""The figure primitives, and the two conventions that silently break figures.

`dm/eval/plot.py` draws; `scripts/plots.py` decides what. These cover the
drawing, because both defects they pin were real and both are invisible in
source: an axis whose labels carry different precisions, and a caption cut off
by a height chosen when the caption was shorter.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

from dm.eval.plot import (
    MINUS,
    Panel,
    Series,
    axis_places,
    caption_height,
    figure,
    fmt,
    nice_ticks,
)


def test_minus_is_typographic_not_a_hyphen():
    assert fmt(-0.67, 2) == f"{MINUS}0.67"
    assert "-" not in fmt(-4)


def test_negative_zero_never_renders():
    assert fmt(-0.0, 2) == "0.00"
    assert fmt(-0.0) == "0"


def test_axis_places_is_uniform_across_the_axis():
    """0, 0.50, 1, 1.50, 2 reads as varying precision. It is one axis."""
    ticks = nice_ticks(0, 2)
    places = axis_places(ticks)
    labels = [fmt(t, places) for t in ticks]
    assert labels == ["0.0", "0.5", "1.0", "1.5", "2.0"]
    assert len({len(x.split(".")[-1]) for x in labels}) == 1


def test_integer_axis_gets_no_decimals():
    ticks = nice_ticks(0, 400)
    assert [fmt(t, axis_places(ticks)) for t in ticks][:2] == ["0", "100"]


def test_ticks_stay_inside_the_requested_limits():
    for lo, hi in ((0, 1), (-4, 14), (550, 700), (0.6, 7.5)):
        assert all(lo <= t <= hi for t in nice_ticks(lo, hi))


def test_caption_grows_the_figure_instead_of_being_clipped():
    short, long = "One line.", "A much longer caption. " * 12
    panel = Panel(xlim=(0, 1), ylim=(0, 1))
    a = figure([(panel, 0, 0)], width=400, height=200, caption=short, number="Figure 1.")
    b = figure([(panel, 0, 0)], width=400, height=200, caption=long, number="Figure 1.")
    height = lambda svg: float(re.search(r'height="([\d.]+)"', svg).group(1))
    assert height(b) > height(a) > 200
    assert caption_height("", "", 400) == 0.0


def test_each_panel_clips_to_its_own_frame():
    """One shared clip sized to the largest panel lets a small panel's series
    draw outside its own frame, which reads as data that is not there."""
    small, large = Panel(xlim=(0, 1), ylim=(0, 1), width=100, height=80), Panel(
        xlim=(0, 1), ylim=(0, 1), width=300, height=200)
    svg = figure([(small, 0, 0), (large, 120, 0)], width=460, height=220)
    assert svg.count("<clipPath") == 2
    assert 'url(#clip0)' in svg and 'url(#clip1)' in svg


def test_series_out_of_range_does_not_escape_the_frame():
    panel = Panel(xlim=(0, 1), ylim=(0, 1), width=100, height=100)
    panel.add(Series([0, 5], [0.5, 0.5]))
    assert 'clip-path="url(#clip0)"' in figure([(panel, 0, 0)], width=120, height=120)


def test_no_panel_draws_outside_the_figure_it_is_placed_in():
    """A third defect of the same family as the two above, and the one that
    prompted this test: a figure wide enough to need three panels was laid out
    by arithmetic, and the only way to see whether the arithmetic was right was
    to render it and squint. A preview tool that pads and rescales made two
    correct figures look clipped and would as happily make a clipped one look
    fine.

    So the geometry is checked instead of eyeballed. Every panel is placed by a
    `translate`, so the rightmost coordinate any panel draws, plus its offset,
    must land inside the canvas the figure declares.
    """
    left = Panel(xlim=(0, 1), ylim=(0, 1), width=100, height=80)
    right = Panel(xlim=(0, 1), ylim=(0, 1), width=100, height=80)
    left.add(Series([0, 1], [0, 1]))
    right.add(Series([0, 1], [1, 0]))

    svg = figure([(left, 10, 10), (right, 130, 10)], width=250, height=110)
    assert rightmost(svg) <= 250

    # And it fails when the arithmetic is wrong, which is what makes it a test.
    cramped = figure([(left, 10, 10), (right, 130, 10)], width=180, height=110)
    assert rightmost(cramped) > 180


def rightmost(svg: str) -> float:
    """The largest x any panel draws, in the figure's own coordinates."""
    worst = 0.0
    for tx, _, body in re.findall(
        r'<g transform="translate\(([\d.]+),([\d.]+)\)">(.*?)</g>\s*'
        r"(?=<g transform|<foreignObject|</svg>)", svg, re.DOTALL
    ):
        xs = [float(v) for v in re.findall(r'\b(?:x|x1|x2|cx)="(-?[\d.]+)"', body)]
        for points in re.findall(r'points="([^"]+)"', body):
            xs += [float(v) for v in re.findall(r"(-?[\d.]+)[, ]", points)][::2]
        if xs:
            worst = max(worst, float(tx) + max(xs))
    return worst


def test_every_shipped_figure_fits_its_canvas():
    """The same check against what is actually in `docs/figs`, so a figure that
    grows past its canvas fails here rather than in a reader's browser."""
    figs = sorted((Path(__file__).resolve().parent.parent / "docs" / "figs").glob("fig*.svg"))
    assert figs, "no figures to check"
    for path in figs:
        svg = path.read_text()
        width = float(re.match(r'<svg[^>]*width="([\d.]+)"', svg).group(1))
        assert rightmost(svg) <= width, f"{path.name} draws past its canvas"


def text_extent(svg: str) -> tuple[float, float]:
    """Leftmost and rightmost figure coordinate any label reaches.

    `rightmost` reads geometry attributes and a `<text>` has only its anchor
    point, so a centred axis label twice the width of its panel passes that
    check and still runs off the page. Widths are estimated at 0.52 em of
    average serif advance, which is coarse -- but the failure this guards
    against is an axis label overhanging by tens of pixels, not by one.
    """
    attr = lambda blob, name: (re.search(rf'{name}="([^"]*)"', blob) or [None, None])[1]
    lo, hi = 1e9, -1e9
    for tx, _, body in re.findall(
        r'<g transform="translate\(([-\d.]+),([-\d.]+)\)">(.*?)</g>\s*'
        r"(?=<g transform|<foreignObject|</svg>)", svg, re.DOTALL
    ):
        for blob, text in re.findall(r"<text\b([^>]*)>([^<]*)</text>", body):
            if not text.strip():
                continue
            x = float(attr(blob, "x") or 0)
            size = float(attr(blob, "font-size") or 12)
            anchor = attr(blob, "text-anchor") or "start"
            width = len(text) * size * 0.52
            left = x - (width if anchor == "end"
                        else width / 2 if anchor == "middle" else 0)
            lo, hi = min(lo, float(tx) + left), max(hi, float(tx) + left + width)
    return lo, hi


def test_no_shipped_figure_has_a_label_hanging_off_the_page():
    """The defect this was written for: a three-panel figure whose rightmost
    panel had an axis label longer than the panel, centred on it, reaching 18px
    past the canvas. Every shape was inside the frame and the figure still
    rendered with its label cut off."""
    figs = sorted((Path(__file__).resolve().parent.parent / "docs" / "figs").glob("fig*.svg"))
    assert figs, "no figures to check"
    for path in figs:
        svg = path.read_text()
        width = float(re.match(r'<svg[^>]*width="([\d.]+)"', svg).group(1))
        lo, hi = text_extent(svg)
        assert lo >= 0, f"{path.name} has a label off the left edge ({lo:.1f})"
        assert hi <= width, (
            f"{path.name} has a label {hi - width:.1f}px past its {width:.0f}px canvas")


def test_plot_loader_refuses_failed_schema2_report(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parent.parent / "scripts" / "plots.py"
    spec = importlib.util.spec_from_file_location("plots_script", script)
    assert spec and spec.loader
    plots = importlib.util.module_from_spec(spec)
    sys.modules["plots_script"] = plots
    spec.loader.exec_module(plots)
    monkeypatch.setattr(plots, "RUNS", tmp_path)
    (tmp_path / "failed.json").write_text(json.dumps({
        "report_schema": 2, "status": "failed",
        "failure": {"kind": "empty_geometry"},
    }))

    with pytest.raises(SystemExit, match="geometry is unavailable"):
        plots.record("failed")
