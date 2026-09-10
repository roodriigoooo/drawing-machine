"""Tier B: QuickDraw -> L0 bytecode.

Uses Google's official sketch-rnn `.npz` release (stroke-3, already
RDP-simplified, already split 70k/2.5k/2.5k per class), which is the same
substrate sketch-rnn trained on -- so the baseline comparison is like-for-like.
This loader selects named categories from QuickDraw's 50-million-drawing,
345-category archive; it does not mirror the whole archive. Five selected
categories therefore expose 375,000 source rows before callers choose limits.

A second RDP pass runs *after* scaling to canvas pixels, so `rdp_eps` is
expressed in pixels and is the knob that controls sequence length. That matters:
the bit codec's sequence length is 8x the byte count, so this is the dial that
decides whether the bit arm of the ablation is trainable at all.
"""

from __future__ import annotations

import os
import urllib.request
from itertools import zip_longest
from pathlib import Path
from typing import TypeVar

import numpy as np

from ..isa.spec import CANVAS, Op

T = TypeVar("T")

URL = "https://storage.googleapis.com/quickdraw_dataset/sketchrnn/{category}.npz"
CACHE = Path(os.environ.get("DM_DATA", "data"))
SPLITS = ("train", "valid", "test")


def download(category: str, cache: Path = CACHE) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{category}.npz"
    if not path.exists():
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(URL.format(category=category), tmp)
        tmp.rename(path)
    return path


def _rdp(points: np.ndarray, eps: float) -> np.ndarray:
    """Ramer-Douglas-Peucker, iterative to keep the stack flat on long strokes."""
    n = len(points)
    if n < 3 or eps <= 0:
        return points
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        lo, hi = stack.pop()
        if hi <= lo + 1:
            continue
        a, b = points[lo], points[hi]
        seg = b - a
        length = float(np.hypot(*seg))
        rel = points[lo + 1 : hi] - a
        if length == 0.0:
            dist = np.hypot(rel[:, 0], rel[:, 1])
        else:
            dist = np.abs(seg[0] * rel[:, 1] - seg[1] * rel[:, 0]) / length
        idx = int(np.argmax(dist))
        if dist[idx] > eps:
            split = lo + 1 + idx
            keep[split] = True
            stack.append((lo, split))
            stack.append((split, hi))
    return points[keep]


def _strokes_from_stroke3(s3: np.ndarray) -> list[np.ndarray]:
    """Cumulative deltas, split at pen lifts."""
    pts = np.cumsum(s3[:, :2].astype(np.float64), axis=0)
    lifts = np.flatnonzero(s3[:, 2] == 1)
    bounds = [0, *(lifts + 1).tolist()]
    if bounds[-1] < len(pts):
        bounds.append(len(pts))
    return [pts[a:b] for a, b in zip(bounds, bounds[1:]) if b - a >= 2]


def _fit_to_canvas(strokes: list[np.ndarray], margin: int) -> list[np.ndarray]:
    allpts = np.concatenate(strokes)
    lo, hi = allpts.min(axis=0), allpts.max(axis=0)
    span = float(max(hi[0] - lo[0], hi[1] - lo[1]))
    usable = CANVAS - 1 - 2 * margin
    scale = usable / span if span > 0 else 1.0
    centre = (lo + hi) / 2.0
    shift = np.array([(CANVAS - 1) / 2.0, (CANVAS - 1) / 2.0])
    return [(s - centre) * scale + shift for s in strokes]


def stroke3_to_program(s3: np.ndarray, margin: int = 8, rdp_eps: float = 2.0) -> bytes:
    """One sketch -> one L0 program. Returns b'' for degenerate sketches."""
    strokes = _strokes_from_stroke3(np.asarray(s3))
    if not strokes:
        return b""
    out = bytearray()
    for stroke in _fit_to_canvas(strokes, margin):
        q = np.clip(np.rint(_rdp(stroke, rdp_eps)), 0, CANVAS - 1).astype(np.uint8)
        # Drop points that quantised onto their predecessor: zero-length LINEs
        # are pure sequence-length tax.
        dedup = q[np.insert(np.any(np.diff(q.astype(np.int16), axis=0) != 0, axis=1), 0, True)]
        if len(dedup) < 2:
            continue
        out += bytes((int(Op.MOVE), int(dedup[0][0]), int(dedup[0][1])))
        for x, y in dedup[1:]:
            out += bytes((int(Op.LINE), int(x), int(y)))
    if not out:
        return b""
    out.append(int(Op.HALT))
    return bytes(out)


def _interleave(groups: list[list[T]]) -> list[T]:
    """Round-robin the categories so that *any prefix is balanced*.

    `load(..., limit=n)` slices from the front. Concatenated by category, that
    silently makes a multi-category corpus single-category: `n_val=1000` over
    five categories drew all 1,000 val programs from the first one, and
    `n_train=100_000` over 70k-program categories drew one whole category plus
    part of a second. Train and val would then be different distributions and
    bits/drawing would be measuring the mismatch. Interleaving is deterministic,
    so the cache is the corpus and no seed is involved.

    Generic in the element type because the class labels go through this same
    call: interleaving programs and their labels in one traversal is what keeps
    the two aligned by construction, where deriving labels afterwards from the
    group sizes would be a second implementation of the round-robin that agreed
    until a category ran short.
    """
    out: list[T] = []
    for row in zip_longest(*groups):
        out.extend(p for p in row if p is not None)
    return out


def _cache_key(categories: tuple[str, ...], split: str, rdp_eps: float, margin: int) -> str:
    return f"L0_{'-'.join(categories)}_{split}_eps{rdp_eps}_m{margin}"


def _save(path: Path, programs: list[bytes], labels: list[int] | None = None) -> None:
    lengths = np.array([len(p) for p in programs], dtype=np.int32)
    blob = np.frombuffer(b"".join(programs), dtype=np.uint8)
    if labels is None:
        np.savez_compressed(path, blob=blob, lengths=lengths)
    else:
        np.savez_compressed(path, blob=blob, lengths=lengths,
                            labels=np.array(labels, dtype=np.int16))


def _load_cached(path: Path) -> tuple[list[bytes], list[int] | None]:
    """The cached programs, and their category indices when the cache has them.

    Caches written before class conditioning existed carry no `labels` array, and
    they are *not* stale: the programs in them are the corpus every Tier B result
    was measured on. So a missing label array is a missing label array, never a
    reason to rebuild the programs -- `load_labelled` handles it, and it verifies
    byte-for-byte that its rebuild agrees before it writes anything.
    """
    with np.load(path) as z:
        blob, lengths = z["blob"].tobytes(), z["lengths"]
        labels = z["labels"].tolist() if "labels" in z.files else None
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    return [blob[a:b] for a, b in zip(offsets, offsets[1:])], labels


def _convert(categories: tuple[str, ...], split: str, rdp_eps: float, margin: int,
             cache: Path) -> tuple[list[bytes], list[int]]:
    """Every category converted and interleaved, with each program's class index.

    The labels come from the same `_interleave` that fixes the order, so they
    cannot drift from it: interleaving a per-category list of indices alongside
    the programs is one traversal of one structure. Deriving them afterwards from
    the group sizes would be a second implementation of the round-robin, and the
    two would agree until a category ran short.
    """
    groups: list[list[bytes]] = []
    tags: list[list[int]] = []
    for index, category in enumerate(categories):
        group: list[bytes] = []
        with np.load(download(category, cache), encoding="latin1",
                     allow_pickle=True) as z:
            for s3 in z[split]:
                program = stroke3_to_program(s3, margin=margin, rdp_eps=rdp_eps)
                if program:
                    group.append(program)
        groups.append(group)
        tags.append([index] * len(group))
    return _interleave(groups), _interleave(tags)


def load(
    categories: str | tuple[str, ...] = ("cat",),
    split: str = "train",
    limit: int | None = None,
    rdp_eps: float = 2.0,
    margin: int = 8,
    cache: Path = CACHE,
) -> list[bytes]:
    """L0 programs for the given categories. Converted once, then cached."""
    return load_labelled(categories, split, limit, rdp_eps, margin, cache)[0]


def load_labelled(
    categories: str | tuple[str, ...] = ("cat",),
    split: str = "train",
    limit: int | None = None,
    rdp_eps: float = 2.0,
    margin: int = 8,
    cache: Path = CACHE,
) -> tuple[list[bytes], list[int]]:
    """The same programs, plus each one's index into `categories`.

    QuickDraw's labels are free and this project discarded them for five months.
    They are the input to class conditioning and the prerequisite for any
    "draw an X" claim (`docs/direction.md` §4.4).

    **A cache written before labels existed is completed in place, and only after
    the rebuild is checked against it byte for byte.** Re-converting is
    deterministic, so the programs must come back identical -- and if they do not,
    the conversion code has changed since the cache was written and every Tier B
    result belongs to the *cached* corpus. Overwriting silently there would
    re-fingerprint eleven runs' worth of results, so it raises instead.
    """
    if isinstance(categories, str):
        categories = (categories,)
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{_cache_key(tuple(categories), split, rdp_eps, margin)}.npz"
    if path.exists():
        programs, labels = _load_cached(path)
        if labels is None:
            rebuilt, labels = _convert(categories, split, rdp_eps, margin, cache)
            if rebuilt != programs:
                raise ValueError(
                    f"{path.name} holds {len(programs)} programs that a fresh "
                    f"conversion no longer reproduces ({len(rebuilt)} programs). The "
                    "cache is the corpus every Tier B number was measured on -- move "
                    "it aside deliberately rather than letting a label backfill "
                    "re-fingerprint it"
                )
            _save(path, programs, labels)
    else:
        programs, labels = _convert(categories, split, rdp_eps, margin, cache)
        _save(path, programs, labels)
    if limit:
        return programs[:limit], labels[:limit]
    return programs, labels
