"""Is the drawing new, or is it a training drawing coming back?

A generative demo that cannot answer this is a demo of a lookup table, and the
answer has to be a *measurement* rather than an assurance. Three of them, in
increasing cost, each read against something rather than against zero:

1. **Verbatim bytecode.** Is the sampled program byte-identical to one the model
   was trained on? A set lookup over the whole training split. The expected
   answer is no, and reporting the count over a whole capture -- `0 of 40` -- is
   worth more than any single claim, because it is the one test with no
   parameters to choose.

2. **Nearest training drawing, in geometry, against the floor a real held-out
   drawing achieves.** Chamfer distance from the sample to its nearest training
   drawing of the same class. Alone that number means nothing: a cat *should*
   look like other cats, and how much is a property of the corpus. So the same
   computation runs with **held-out val drawings** in the sample's place. A val
   drawing is novel by construction -- the model never saw it -- so its distance
   distribution is exactly the scale on which the sample's distance is read. A
   sample sitting at the median of that distribution is as far from the training
   set as a genuinely unseen human drawing is.

   This is the move `dm/eval/quality.py` makes for its three set metrics and
   `dm/eval/recovery.py` makes against its constructed ceiling, applied to one
   drawing instead of a set.

3. **Longest copied byte run** (`longest_shared_run`, offline). The largest `k`
   such that some `k` consecutive bytes of the sample occur somewhere in the
   training corpus for its class. It separates "novel drawing" from "novel
   arrangement of memorised fragments", which the geometric distance cannot,
   and it is read against the same quantity measured on held-out val programs.

**What none of them establishes.** These say the sample is not a copy. They do
not say it is *good*; that is what `docs/conditioning.md` §6 measures with set
metrics against real drawings, and it found the sampler sits in each class's
typical middle. Novelty and quality fail in different directions and this file
only sees one of them.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..data import quickdraw
from ..eval.metrics import trace_points
from ..vm.interp import VM

#: Training drawings held per class for the geometric comparison. The full split
#: is 70,000 per class and the nearest-neighbour scan is quadratic in points, so
#: the index subsamples -- deterministically, from the front, which is the same
#: prefix `build_data` gives the trainer.
DEFAULT_PER_CLASS = 1500

#: Held-out val drawings per class used to measure the floor. The floor is a
#: distribution the sample's distance is read as a percentile of, so a few
#: hundred draws resolve it far past what the reading needs, and the full
#: 2,500-per-class split costs twenty minutes to say the same thing.
DEFAULT_FLOOR_PER_CLASS = 300

#: Points resampled along each drawing for the Chamfer comparison. `dm/eval/
#: metrics.py` uses 256 for set metrics; 128 halves a live query's cost and the
#: floor is measured at the identical setting, so the comparison stays paired.
DEFAULT_POINTS = 128


@dataclass(frozen=True)
class Neighbour:
    """One training drawing near the sample, and how near."""

    index: int
    distance: float
    #: The neighbour's strokes, in the same `{"points": [[x, y], ...]}` shape the
    #: device geometry uses, so the page renders a training drawing and a
    #: sampled drawing through one code path.
    strokes: list[dict]

    def as_dict(self) -> dict:
        return {"index": self.index, "distance": round(self.distance, 3),
                "strokes": self.strokes}


def _cloud(program: bytes, points: int) -> np.ndarray | None:
    """A program's geometry as a point cloud of at most `points`, or None if empty."""
    trace = VM().run(program)
    if trace.is_empty:
        return None
    cloud = trace_points(trace, points)
    return cloud.astype(np.float64) if len(cloud) else None


def _pad(clouds: list[np.ndarray], points: int) -> tuple[np.ndarray, np.ndarray]:
    """A ragged list of clouds as one `(N, P, 2)` array plus its true lengths.

    Padding is **not** repetition. `dm/eval/metrics.py` resamples down to a cap
    and leaves shorter drawings short, so a bank built by repeating a drawing's
    last point up to the cap would weight that point once per copy inside the
    Chamfer mean -- a short drawing would score as if most of its ink sat in one
    place. The pad value is NaN and the lengths are carried, so every reduction
    below can exclude it explicitly.
    """
    bank = np.full((len(clouds), points, 2), np.nan)
    lengths = np.zeros(len(clouds), dtype=np.int64)
    for row, cloud in enumerate(clouds):
        bank[row, :len(cloud)] = cloud
        lengths[row] = len(cloud)
    return bank, lengths


def _chamfer_matrix(query: np.ndarray, bank: np.ndarray, lengths: np.ndarray,
                    chunk: int = 256) -> np.ndarray:
    """Symmetric mean nearest-neighbour distance from `query` to every bank row.

    The same quantity `dm.eval.metrics.chamfer` computes for one pair, over a
    whole bank at once, and chunked because the intermediate is `(N, P, Q, 2)`
    -- 400 MB at N=1500, P=Q=128, which is a swap storm rather than a
    measurement.
    """
    if len(bank) == 0:
        return np.zeros(0)
    out = np.empty(len(bank))
    valid_q = query.shape[0]
    for start in range(0, len(bank), chunk):
        stop = min(start + chunk, len(bank))
        block = bank[start:stop]                              # (B, P, 2)
        diff = block[:, :, None, :] - query[None, None, :, :]  # (B, P, Q, 2)
        dist = np.linalg.norm(diff, axis=-1)                   # (B, P, Q)
        # Padded bank points are NaN, so their distances are NaN. `nanmin` over
        # the bank axis drops them from the query's nearest-neighbour search,
        # and `nanmean` over the bank axis averages only the real points.
        to_query = np.nanmin(dist, axis=2)                     # (B, P)
        to_bank = np.nanmin(dist, axis=1)                      # (B, Q)
        out[start:stop] = (np.nanmean(to_query, axis=1)
                           + to_bank[:, :valid_q].mean(axis=1)) / 2.0
    return out


class NoveltyIndex:
    """The training split, in the two forms the questions above need.

    Built once per category list and cached: the geometric bank costs a VM run
    and a resample per drawing, which is minutes at 2,000 × 5, and a live page
    cannot pay that per word.
    """

    def __init__(self, categories: tuple[str, ...], per_class: int = DEFAULT_PER_CLASS,
                 points: int = DEFAULT_POINTS, rdp_eps: float = 4.0,
                 margin: int = 8,
                 floor_per_class: int = DEFAULT_FLOOR_PER_CLASS) -> None:
        self.categories = tuple(categories)
        self.per_class = per_class
        self.floor_per_class = floor_per_class
        self.points = points
        self.rdp_eps = rdp_eps
        self.margin = margin
        self.banks: dict[int, np.ndarray] = {}
        self.lengths: dict[int, np.ndarray] = {}
        self.bank_index: dict[int, list[int]] = {}
        #: The neighbours' own bytecode, kept so the page can draw the *drawing*
        #: rather than the 128-point cloud the distance was computed on. Both are
        #: the same training program; a cloud plotted as dots is unreadable and a
        #: cloud joined into a polyline invents segments between strokes.
        self.bank_programs: dict[int, list[bytes]] = {}
        self.floor: dict[int, np.ndarray] = {}
        self.train_hashes: set[bytes] = set()
        self.n_train = 0

    # -- identity -----------------------------------------------------------

    @property
    def key(self) -> str:
        """A cache name that changes whenever any input to the numbers changes."""
        raw = json.dumps({
            "categories": list(self.categories), "per_class": self.per_class,
            "points": self.points, "rdp_eps": self.rdp_eps, "margin": self.margin,
            "floor_per_class": self.floor_per_class,
        }, sort_keys=True)
        return "novelty_" + hashlib.sha256(raw.encode()).hexdigest()[:16]

    # -- construction -------------------------------------------------------

    def build(self, progress=None) -> "NoveltyIndex":
        extra = {"rdp_eps": self.rdp_eps, "margin": self.margin}
        train, train_labels = quickdraw.load_labelled(self.categories, "train", **extra)
        valid, valid_labels = quickdraw.load_labelled(self.categories, "valid", **extra)
        self.n_train = len(train)
        # Every training program, not the subsample: the verbatim test is the one
        # with no parameters and it must not inherit the bank's.
        self.train_hashes = set(train)

        train_by_class: dict[int, list[int]] = {i: [] for i in range(len(self.categories))}
        for position, label in enumerate(train_labels):
            if len(train_by_class[label]) < self.per_class:
                train_by_class[label].append(position)

        for klass, positions in train_by_class.items():
            clouds, kept, programs = [], [], []
            for position in positions:
                cloud = _cloud(train[position], self.points)
                if cloud is not None:
                    clouds.append(cloud)
                    kept.append(position)
                    programs.append(train[position])
            self.banks[klass], self.lengths[klass] = _pad(clouds, self.points)
            self.bank_index[klass] = kept
            self.bank_programs[klass] = programs
            if progress:
                progress(f"bank {self.categories[klass]}: {len(kept)} drawings")

        # The floor: held-out val drawings answering the identical question.
        val_by_class: dict[int, list[bytes]] = {i: [] for i in range(len(self.categories))}
        for program, label in zip(valid, valid_labels):
            val_by_class[label].append(program)
        for klass, programs in val_by_class.items():
            distances = []
            for program in programs[:self.floor_per_class]:
                cloud = _cloud(program, self.points)
                if cloud is None:
                    continue
                distances.append(float(_chamfer_matrix(
                    cloud, self.banks[klass], self.lengths[klass]).min()))
            self.floor[klass] = np.array(sorted(distances))
            if progress:
                progress(f"floor {self.categories[klass]}: {len(distances)} val drawings")
        return self

    # -- persistence --------------------------------------------------------

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.key}.npz"
        payload = {
            "meta": np.frombuffer(json.dumps({
                "categories": list(self.categories), "per_class": self.per_class,
                "points": self.points, "rdp_eps": self.rdp_eps,
                "margin": self.margin, "n_train": self.n_train,
                "floor_per_class": self.floor_per_class,
            }).encode(), dtype=np.uint8),
            "hashes": np.frombuffer(b"".join(
                hashlib.sha256(p).digest() for p in sorted(self.train_hashes)
            ), dtype=np.uint8),
        }
        for klass in self.banks:
            payload[f"bank_{klass}"] = self.banks[klass].astype(np.float32)
            payload[f"lengths_{klass}"] = self.lengths[klass]
            payload[f"index_{klass}"] = np.array(self.bank_index[klass], dtype=np.int64)
            payload[f"floor_{klass}"] = self.floor[klass].astype(np.float32)
            blob = b"".join(self.bank_programs[klass])
            payload[f"programs_{klass}"] = np.frombuffer(blob, dtype=np.uint8)
            payload[f"proglen_{klass}"] = np.array(
                [len(p) for p in self.bank_programs[klass]], dtype=np.int64)
        np.savez_compressed(path, **payload)
        return path

    @classmethod
    def load(cls, path: Path) -> "NoveltyIndex":
        blob = np.load(path, allow_pickle=False)
        meta = json.loads(bytes(blob["meta"]).decode())
        index = cls(tuple(meta["categories"]), meta["per_class"], meta["points"],
                    meta["rdp_eps"], meta["margin"],
                    meta.get("floor_per_class", DEFAULT_FLOOR_PER_CLASS))
        index.n_train = meta["n_train"]
        digests = bytes(blob["hashes"])
        index._digests = {digests[i:i + 32] for i in range(0, len(digests), 32)}
        for klass in range(len(index.categories)):
            index.banks[klass] = blob[f"bank_{klass}"].astype(np.float64)
            index.lengths[klass] = blob[f"lengths_{klass}"]
            index.bank_index[klass] = blob[f"index_{klass}"].tolist()
            index.floor[klass] = blob[f"floor_{klass}"].astype(np.float64)
            raw = bytes(blob[f"programs_{klass}"])
            programs, offset = [], 0
            for length in blob[f"proglen_{klass}"].tolist():
                programs.append(raw[offset:offset + length])
                offset += length
            index.bank_programs[klass] = programs
        return index

    # -- queries ------------------------------------------------------------

    def is_verbatim(self, program: bytes) -> bool:
        """Byte-identical to some training program.

        Compares SHA-256 digests when the index was loaded from cache and the
        programs themselves when it was just built. Both answer the same
        question; a 256-bit collision is not the uncertainty in this demo.
        """
        digests = getattr(self, "_digests", None)
        if digests is not None:
            return hashlib.sha256(program).digest() in digests
        return program in self.train_hashes

    def nearest(self, program: bytes, class_index: int, k: int = 3) -> dict:
        """The `k` nearest training drawings, and where the sample sits on the floor."""
        cloud = _cloud(program, self.points)
        if cloud is None:
            return {"empty": True}
        distances = _chamfer_matrix(cloud, self.banks[class_index],
                                    self.lengths[class_index])
        order = np.argsort(distances)[:k]
        floor = self.floor[class_index]
        nearest = float(distances[order[0]])
        percentile = float(np.searchsorted(floor, nearest) / len(floor) * 100) if len(floor) else float("nan")
        return {
            "empty": False,
            "verbatim": self.is_verbatim(program),
            "distance": round(nearest, 3),
            "floor_median": round(float(np.median(floor)), 3) if len(floor) else None,
            "floor_percentile": round(percentile, 1),
            "floor_n": int(len(floor)),
            "bank_n": int(len(self.banks[class_index])),
            "train_n": self.n_train,
            "neighbours": [
                Neighbour(int(self.bank_index[class_index][i]), float(distances[i]),
                          _neighbour_strokes(self, class_index, int(i))).as_dict()
                for i in order
            ],
        }


def _neighbour_strokes(index: NoveltyIndex, class_index: int, row: int) -> list[dict]:
    """The neighbour as strokes, executed through the same reference VM.

    The distance above was computed on a 128-point resampling of exactly this
    program, and the page says so. Drawing the resampled cloud instead would be
    more literal and less true: joined into a polyline it invents segments
    between strokes the drawing never connected, and plotted as dots it is
    unreadable, so neither picture is the training drawing the number found.
    """
    trace = VM().run(index.bank_programs[class_index][row])
    return [{"points": [[round(x, 2), round(y, 2)] for x, y in stroke.points],
             "width": stroke.width} for stroke in trace.strokes]
