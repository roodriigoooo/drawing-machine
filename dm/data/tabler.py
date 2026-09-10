"""Tier C: Tabler Icons -> L0 bytecode.

Chosen after SVG-Icons8 failed its acceptance test. DeepSVG's release is
preprocessed with `normalize -> zoom(0.9) -> canonicalize`, an arbitrary float
scale followed by rounding, which rounds authored repeats apart: 0.62% of bytes
compressible exactly against 3.17% at +-1 px (`docs/tier-c.md`). The raw icons
are behind an icons8 paid plan.

Tabler is 5,130 outline icons, MIT, **authored on a 24x24 grid**, and that grid
is the whole reason it is here:

    CANVAS is 256, and 24 * 10 = 240 <= 255, so SCALE = 10 is exact.

Exactness survives because **rounding commutes with integer translation**:
`round(x + k) == round(x) + k` for integral `k`. Source coordinates carry up to
three decimals and individual points do round, but two elements separated by an
integral offset in canvas units stay separated by exactly that offset, so a
`REPEAT n dx dy` over them is still lossless. Any pipeline that normalises to a
float bounding box first has already lost this, whatever it does afterwards.

Only `<path>` elements appear in the corpus (20,706 of them) and none uses
`Z`/`z`, so the parser covers exactly the command set Tabler uses.
"""

from __future__ import annotations

import math
import random
import re
from pathlib import Path

from ..isa.spec import CANVAS, Op

#: Source viewBox is 0 0 24 24. The largest integer scale that fits 8 bits.
GRID = 24
SCALE = 10
assert GRID * SCALE <= CANVAS - 1

#: Cubic flattening for arcs only; straight segments stay exact.
ARC_SEGMENTS = 8

_TOKEN = re.compile(r"[MmLlHhVvCcSsQqTtAaZz]|-?\d*\.?\d+(?:[eE][-+]?\d+)?")
_PATH_D = re.compile(r'<path[^>]*\sd="([^"]*)"', re.DOTALL)


class UnsupportedPath(ValueError):
    """A path command outside the Tabler subset."""


def _numbers(tokens: list[str], index: int, count: int) -> tuple[list[float], int]:
    values = [float(t) for t in tokens[index : index + count]]
    if len(values) != count:
        raise UnsupportedPath("truncated path data")
    return values, index + count


def _arc_to_cubics(p0, rx, ry, phi_deg, large, sweep, p1) -> list[tuple]:
    """SVG endpoint-parameterised arc -> cubic Beziers (F.6.5 of the spec)."""
    if rx == 0 or ry == 0 or p0 == p1:
        return [(p0, p0, p1, p1)]
    phi = math.radians(phi_deg)
    cos_p, sin_p = math.cos(phi), math.sin(phi)
    dx2, dy2 = (p0[0] - p1[0]) / 2.0, (p0[1] - p1[1]) / 2.0
    x1p, y1p = cos_p * dx2 + sin_p * dy2, -sin_p * dx2 + cos_p * dy2
    rx, ry = abs(rx), abs(ry)
    lam = (x1p / rx) ** 2 + (y1p / ry) ** 2
    if lam > 1:
        rx, ry = rx * math.sqrt(lam), ry * math.sqrt(lam)
    denom = (rx * y1p) ** 2 + (ry * x1p) ** 2
    num = max(0.0, (rx * ry) ** 2 - denom)
    coef = (1 if large != sweep else -1) * math.sqrt(num / denom) if denom else 0.0
    cxp, cyp = coef * rx * y1p / ry, -coef * ry * x1p / rx
    cx = cos_p * cxp - sin_p * cyp + (p0[0] + p1[0]) / 2.0
    cy = sin_p * cxp + cos_p * cyp + (p0[1] + p1[1]) / 2.0

    theta1 = math.atan2((y1p - cyp) / ry, (x1p - cxp) / rx)
    theta2 = math.atan2((-y1p - cyp) / ry, (-x1p - cxp) / rx)
    delta = theta2 - theta1
    if not sweep and delta > 0:
        delta -= 2 * math.pi
    elif sweep and delta < 0:
        delta += 2 * math.pi

    n = max(1, min(ARC_SEGMENTS, math.ceil(abs(delta) / (math.pi / 2))))
    step = delta / n
    alpha = 4.0 / 3.0 * math.tan(step / 4.0)
    out, theta = [], theta1
    start = p0
    for _ in range(n):
        nxt = theta + step
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        cos_n, sin_n = math.cos(nxt), math.sin(nxt)

        def point(ct, st):
            return (cx + rx * ct * cos_p - ry * st * sin_p,
                    cy + rx * ct * sin_p + ry * st * cos_p)

        end = point(cos_n, sin_n)
        d1 = (-rx * sin_t * cos_p - ry * cos_t * sin_p,
              -rx * sin_t * sin_p + ry * cos_t * cos_p)
        d2 = (-rx * sin_n * cos_p - ry * cos_n * sin_p,
              -rx * sin_n * sin_p + ry * cos_n * cos_p)
        c1 = (start[0] + alpha * d1[0], start[1] + alpha * d1[1])
        c2 = (end[0] - alpha * d2[0], end[1] - alpha * d2[1])
        out.append((start, c1, c2, end))
        start, theta = end, nxt
    return out


def parse_path(d: str) -> list[tuple[tuple[float, float], list[tuple]]]:
    """SVG path data -> `(start_point, segments)` per subpath, absolute units.

    A segment is `("L", end)` or `("C", c1, c2, end)`. The start point is kept
    because the pen has to be placed there before anything is drawn, and a
    cubic's own coordinates cannot recover it.
    """
    tokens = _TOKEN.findall(d)
    subpaths: list[tuple[tuple[float, float], list[tuple]]] = []
    segments: list[tuple] = []
    point = start = (0.0, 0.0)
    command = ""
    i = 0

    def flush() -> None:
        if segments:
            subpaths.append((start, list(segments)))
            segments.clear()

    while i < len(tokens):
        if tokens[i].isalpha():
            command = tokens[i]
            i += 1
            if command in "Zz":
                if segments:
                    segments.append(("L", start))
                continue
        elif command in ("M", "m"):
            command = "L" if command == "M" else "l"   # implicit lineto
        upper = command.upper()
        rel = command.islower()

        if upper == "M":
            (x, y), i = _numbers(tokens, i, 2)
            point = (point[0] + x, point[1] + y) if rel else (x, y)
            flush()
            start = point
        elif upper == "L":
            (x, y), i = _numbers(tokens, i, 2)
            point = (point[0] + x, point[1] + y) if rel else (x, y)
            segments.append(("L", point))
        elif upper == "H":
            (x,), i = _numbers(tokens, i, 1)
            point = (point[0] + x if rel else x, point[1])
            segments.append(("L", point))
        elif upper == "V":
            (y,), i = _numbers(tokens, i, 1)
            point = (point[0], point[1] + y if rel else y)
            segments.append(("L", point))
        elif upper in ("C", "S", "Q", "T"):
            need = {"C": 6, "S": 4, "Q": 4, "T": 2}[upper]
            values, i = _numbers(tokens, i, need)
            points = [(values[k], values[k + 1]) for k in range(0, need, 2)]
            if rel:
                points = [(point[0] + px, point[1] + py) for px, py in points]
            previous = segments[-1] if segments and segments[-1][0] == "C" else None
            if upper == "C":
                c1, c2, end = points
            elif upper == "S":
                # Reflect the previous second control point through the current
                # point; with no previous cubic the reflection is the point.
                c1 = ((2 * point[0] - previous[2][0], 2 * point[1] - previous[2][1])
                      if previous else point)
                c2, end = points
            else:
                if upper == "Q":
                    control, end = points
                else:
                    control = ((2 * point[0] - previous[1][0], 2 * point[1] - previous[1][1])
                               if previous else point)
                    end = points[0]
                c1 = (point[0] + 2.0 / 3 * (control[0] - point[0]),
                      point[1] + 2.0 / 3 * (control[1] - point[1]))
                c2 = (end[0] + 2.0 / 3 * (control[0] - end[0]),
                      end[1] + 2.0 / 3 * (control[1] - end[1]))
            segments.append(("C", c1, c2, end))
            point = end
        elif upper == "A":
            values, i = _numbers(tokens, i, 7)
            rx, ry, rotation, large, sweep, x, y = values
            end = (point[0] + x, point[1] + y) if rel else (x, y)
            for _, c1, c2, tip in _arc_to_cubics(
                point, rx, ry, rotation, int(large), int(sweep), end
            ):
                segments.append(("C", c1, c2, tip))
            point = end
        else:
            raise UnsupportedPath(f"command {command!r}")
    flush()
    return subpaths


def _q(value: float) -> int:
    return max(0, min(CANVAS - 1, round(value * SCALE)))


def svg_to_program(svg: str) -> bytes:
    """One Tabler SVG -> one L0 program. Returns b'' if nothing survives.

    Zero-length segments are dropped: they cost three bytes and draw nothing,
    and after quantisation a 0.05-unit flourish becomes exactly that.
    """
    out = bytearray()
    for d in _PATH_D.findall(svg):
        for start, segments in parse_path(d):
            pen = (_q(start[0]), _q(start[1]))
            body = bytearray()
            cursor = pen
            for segment in segments:
                if segment[0] == "L":
                    end = (_q(segment[1][0]), _q(segment[1][1]))
                    if end == cursor:
                        continue
                    body += bytes((int(Op.LINE), *end))
                else:
                    _, c1, c2, tip = segment
                    end = (_q(tip[0]), _q(tip[1]))
                    control = (_q(c1[0]), _q(c1[1]), _q(c2[0]), _q(c2[1]))
                    if end == cursor and control == (*cursor, *cursor):
                        continue
                    body += bytes((int(Op.CURVE), *control, *end))
                cursor = end
            if body:
                out += bytes((int(Op.MOVE), *pen)) + body
    if not out:
        return b""
    out.append(int(Op.HALT))
    return bytes(out)


def load(icons_dir: Path, style: str = "outline", limit: int | None = None) -> list[bytes]:
    """Every icon in `icons_dir/<style>` as an L0 program, sorted by filename."""
    files = sorted((Path(icons_dir) / style).glob("*.svg"))
    programs = []
    for path in files[:limit] if limit else files:
        try:
            program = svg_to_program(path.read_text())
        except (UnsupportedPath, ValueError):
            continue
        if program:
            programs.append(program)
    return programs


#: Variant suffixes that make two icons near-duplicates of one another:
#: `battery-1..4`, `arrow-up`/`arrow-down`, `bell`/`bell-off`, `star`/`star-filled`.
_VARIANT = re.compile(r"-(\d+|up|down|left|right|off|filled|x)$")


def family(name: str) -> str:
    """The stem shared by an icon and its variants."""
    while _VARIANT.search(name):
        name = _VARIANT.sub("", name)
    return name


def _groups(named: list[tuple[str, bytes]]) -> list[list[int]]:
    """Indices grouped so that near-duplicates never straddle a split.

    Two mechanisms merge here. **Name families** — `battery-1` through
    `battery-4` are the same drawing four times — and **exact program
    identity**, of which this corpus has 125 icons in 53 groups. Either one
    across a split scores memorisation as generalisation, which is the fault
    section 8 of PLAN.md records costing a whole schema.
    """
    parent = list(range(len(named)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    first_of: dict[str, int] = {}
    for index, (name, program) in enumerate(named):
        for key in (f"fam:{family(name)}", f"prog:{program.hex()}"):
            if key in first_of:
                union(first_of[key], index)
            else:
                first_of[key] = index

    buckets: dict[int, list[int]] = {}
    for index in range(len(named)):
        buckets.setdefault(find(index), []).append(index)
    return list(buckets.values())


def split(
    icons_dir: Path | str = "data/tabler/icons",
    style: str = "outline",
    val_frac: float = 0.1,
    seed: int = 0,
) -> tuple[list[bytes], list[bytes]]:
    """Train and val, disjoint at the level of icon *families*, not icons.

    Deterministic: files are sorted, groups are formed from names and program
    bytes, and only the group order is shuffled. The same `seed` always yields
    the same corpus, so a record naming it names the data.
    """
    files = sorted((Path(icons_dir) / style).glob("*.svg"))
    named: list[tuple[str, bytes]] = []
    for path in files:
        try:
            program = svg_to_program(path.read_text())
        except (UnsupportedPath, ValueError):
            continue
        if program:
            named.append((path.stem, program))

    groups = _groups(named)
    random.Random(seed).shuffle(groups)
    target = round(val_frac * len(named))
    val_index: set[int] = set()
    for group in groups:
        if len(val_index) >= target:
            break
        val_index.update(group)

    train = [p for i, (_, p) in enumerate(named) if i not in val_index]
    val = [p for i, (_, p) in enumerate(named) if i in val_index]
    return train, val
