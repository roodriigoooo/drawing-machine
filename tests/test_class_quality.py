"""`class_quality`, pinned against sets whose answer is arithmetic.

The instrument's whole claim is that the off-diagonal cells make the diagonal
readable, so the tests are built from classes whose geometric separation is
constructed rather than assumed: class 0 draws in one corner of the canvas and
class 1 in the opposite one, which makes "the diagonal is the row minimum" a
fact about the fixture and any other reading a bug here.

The rest is the contract the caller relies on: the matrix diagonal must be the
per-class reading and not a recomputation that could drift from it, unbalanced
or mismatched sets must be refused rather than scored, and an empty decode must
leave the set *and* leave a count behind — the same rule `quality_of` is tested
for, re-checked here because `class_quality` routes around it.
"""

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from dm.eval.quality import class_quality, clouds_of, distribution_metrics
from dm.eval.sampling import SamplingConfig
from dm.isa.asm import assemble
from dm.vm.interp import VM

POINTS = 64

# The driver is a script rather than a package, and the two readings below --
# the per-draw margin and the floor matrix -- are the arithmetic in it that a
# caller can get wrong silently. Same idiom as `tests/test_sweep.py`.
_PATH = Path(__file__).resolve().parent.parent / "scripts" / "class_quality.py"
_spec = importlib.util.spec_from_file_location("class_quality_script", _PATH)
assert _spec and _spec.loader, f"no driver at {_PATH}"
driver = importlib.util.module_from_spec(_spec)
sys.modules["class_quality_script"] = driver
_spec.loader.exec_module(driver)


def corner_program(x: int, y: int, size: int) -> bytes:
    return assemble(f"MOVE {x} {y}\nLINE {x + size} {y}\nLINE {x + size} {y + size}\nHALT")


def corner_class(x: int, y: int, n: int = 12) -> list[bytes]:
    """`n` distinct drawings confined to one corner: same class, real spread."""
    return [corner_program(x + 2 * i, y + i, 20 + i) for i in range(n)]


def two_class_fixture(n: int = 12):
    """Samples and references for two classes at opposite canvas corners.

    Samples and references are *different* drawings of the same corner, so the
    diagonal is a good-but-imperfect match — the realistic case — while the
    off-diagonal is separated by ~150 px of canvas by construction.
    """
    samples = {0: corner_class(10, 10, n), 1: corner_class(180, 180, n)}
    references = {c: clouds_of(corner_class(x, y, n), POINTS)[0]
                  for c, (x, y) in {0: (16, 12), 1: (186, 182)}.items()}
    return samples, references


def test_the_diagonal_is_the_row_minimum_when_the_classes_are_separated():
    """The reading the instrument exists for: asked-for geometry lands nearer
    its own class's reals than any other class's, on a fixture where that is
    true by construction. A matrix that fails this on separated corners would
    fail it on real classes silently."""
    samples, references = two_class_fixture()
    got = class_quality(samples, references, POINTS)

    for asked in got["classes"]:
        row = got["matrix"][asked]
        assert row[asked] == min(row)
        # And the separation is not marginal: the wrong class is a canvas away.
        assert min(v for real, v in enumerate(row) if real != asked) > 3 * row[asked]


def test_the_matrix_diagonal_is_the_per_class_reading_not_a_recomputation():
    """The cell a reader compares against the table must *be* the table's
    number. A drifted recomputation would let the matrix and the diagonal
    disagree about the same sets in one report."""
    samples, references = two_class_fixture()
    got = class_quality(samples, references, POINTS)

    for c in got["classes"]:
        assert got["matrix"][c][c] == got["diagonal"][c]["mmd"]
        alone = distribution_metrics(clouds_of(samples[c], POINTS)[0], references[c])
        assert got["diagonal"][c]["mmd"] == alone["mmd"]
        assert got["diagonal"][c]["nna"] == alone["nna"]


def test_samples_identical_to_the_reference_score_the_memorisation_signature():
    """Returning the reference verbatim must read coverage 1, mmd ~0 and nna
    ~0 on the diagonal — the same degenerate case `distribution_metrics` is
    tested for, surviving the per-class routing."""
    _, references = two_class_fixture()
    reference_programs = {0: corner_class(16, 12), 1: corner_class(186, 182)}
    got = class_quality(reference_programs, references, POINTS)

    for c in got["classes"]:
        assert got["diagonal"][c]["coverage"] == 1.0
        assert got["diagonal"][c]["mmd"] < 0.05
        assert got["diagonal"][c]["nna"] < 0.05


def test_unbalanced_and_mismatched_sets_are_refused_not_scored():
    """Two of the three metrics are functions of the set-size ratio, so an
    unbalanced cell is a different measurement wearing the same name."""
    samples, references = two_class_fixture()

    with pytest.raises(ValueError, match="unbalanced"):
        class_quality({0: samples[0][:8], 1: samples[1]}, references, POINTS)
    smaller_references = {c: clouds_of(corner_class(x, y, 8), POINTS)[0]
                          for c, (x, y) in {0: (16, 12), 1: (186, 182)}.items()}
    with pytest.raises(ValueError, match="unbalanced"):
        class_quality(samples, smaller_references, POINTS)
    with pytest.raises(ValueError, match="classes"):
        class_quality({0: samples[0]}, references, POINTS)


def test_the_margin_is_per_draw_and_not_read_off_the_averaged_matrix():
    """`min off-diagonal − diagonal`, taken in each draw and then averaged.

    Averaging the matrix first and taking the margin after is a different
    number whenever the nearest confuser changes between draws, and it is the
    per-draw one that supports "the diagonal won every time". The fixture makes
    them differ by construction: the confuser swaps between the two draws.
    """
    draws = [{"matrix": [[1.0, 2.0, 9.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]},
             {"matrix": [[1.0, 9.0, 3.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]}]

    got = driver.margins(draws, 3)

    assert got[0] == [1.0, 2.0]           # per draw: 2−1 then 3−1
    # Off the averaged row [1.0, 5.5, 6.0] the margin would read 4.5, which is
    # more than twice either draw's and belongs to a confuser neither draw had.
    assert sum(got[0]) / 2 == 1.5


def test_a_perfect_generator_takes_every_bit_of_the_corpus_margin():
    """The floor matrix's contract: real drawings in the samples' place must
    land on the diagonal *and* define the margin a model is read against.

    Without it, "the diagonal is the row minimum" is unreadable — how far it
    should win by is a property of the classes' shapes, not of the model."""
    train = {0: corner_class(12, 11), 1: corner_class(182, 181)}
    references = {c: clouds_of(corner_class(x, y), POINTS)[0]
                  for c, (x, y) in {0: (16, 12), 1: (186, 182)}.items()}

    floor = driver.floor_reading(train, references, 2, len(train[0]), 1, POINTS)

    assert len(floor) == 1
    for c in (0, 1):
        row = floor[0]["matrix"][c]
        assert row[c] == min(row)
        assert floor[0]["diagonal"][c]["mmd"] == row[c]
    # And the margin it defines is positive, so a ratio against it is finite —
    # the column the report prints divides by exactly this.
    assert all(m > 0 for m in driver.margins(floor, 2)[0])


def test_report_identity_sees_every_sampling_input_and_the_control():
    """No smoke run or neighbouring sampler setting may replace a reading."""
    record, control = {"name": "conditional"}, {"name": "unconditional"}
    baseline = driver.report_path(record, control, 100, 5, 128, 418, "mps",
                                  SamplingConfig(40, 1.0))
    neighbours = {
        driver.report_path(record, control, 50, 5, 128, 418, "mps",
                           SamplingConfig(40, 1.0)),
        driver.report_path(record, control, 100, 2, 128, 418, "mps",
                           SamplingConfig(40, 1.0)),
        driver.report_path(record, control, 100, 5, 64, 418, "mps",
                           SamplingConfig(40, 1.0)),
        driver.report_path(record, control, 100, 5, 128, 512, "mps",
                           SamplingConfig(40, 1.0)),
        driver.report_path(record, control, 100, 5, 128, 418, "cpu",
                           SamplingConfig(40, 1.0)),
        driver.report_path(record, control, 100, 5, 128, 418, "mps",
                           SamplingConfig(80, 1.0)),
        driver.report_path(record, control, 100, 5, 128, 418, "mps",
                           SamplingConfig(40, 1.2)),
        driver.report_path(record, None, 100, 5, 128, 418, "mps",
                           SamplingConfig(40, 1.0)),
    }

    assert baseline not in neighbours
    assert len(neighbours) == 8


def test_failed_report_keeps_partial_draws_and_failure_reason(tmp_path):
    args = SimpleNamespace(n=100, seeds=5, cloud_points=128, device="mps")
    draw = {
        "diagonal": {0: {"validity": 0.99, "empty": 0.01,
                          "truncated": 0.0, "faults": {}}},
        "matrix": [[math.nan]],
    }
    got = driver.make_report(
        {"name": "conditional"}, {"name": "control"}, ["cat"], args,
        SamplingConfig(None, 1.0), 418, [draw], {0: []}, [], [],
        status="failed",
        failure={"kind": "empty_geometry", "side": "conditional", "seed": 0},
    )

    assert got["status"] == "failed"
    assert len(got["draws"]) == 1
    assert got["control_draws"] == []
    assert got["sampling_guards"]["0"][0]["empty"] == 0.01
    assert got["failure"]["kind"] == "empty_geometry"
    path = tmp_path / "failed.json"
    driver.write_report(path, got)
    persisted = json.loads(path.read_text())
    assert persisted["status"] == "failed"
    assert persisted["failure"]["seed"] == 0
    assert persisted["draws"][0]["matrix"] == [[None]]


def test_an_empty_decode_leaves_the_set_and_leaves_a_count():
    """A blank canvas has no geometry and cannot enter a Chamfer distance, but
    dropping it silently would let a class that half-fails publish the score of
    its surviving half as if nothing happened."""
    samples, references = two_class_fixture()
    blank = assemble("HALT")
    assert not len(VM().run(blank))
    samples[0] = samples[0][:6] + [blank] * 6

    got = class_quality(samples, references, POINTS)

    assert got["diagonal"][0]["empty"] == 0.5
    assert got["diagonal"][0]["n_gen"] == 6
    assert got["diagonal"][1]["empty"] == 0.0
    # Dropping blanks would change 12 v 12 into 6 v 12. Preserve exclusion as
    # the finding and refuse to manufacture a geometry number at another ratio.
    assert all(math.isnan(v) for v in got["matrix"][0])
    with pytest.raises(SystemExit, match="do not score an unbalanced set"):
        driver.require_balanced_geometry([got], ["zero", "one"], 12)
