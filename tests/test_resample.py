"""Resample reports keep every expensive reading and block empty-set bias."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from dm.eval.sampling import SamplingConfig

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "resample.py"
_spec = importlib.util.spec_from_file_location("resample_script", _PATH)
assert _spec and _spec.loader, f"no driver at {_PATH}"
driver = importlib.util.module_from_spec(_spec)
sys.modules["resample_script"] = driver
_spec.loader.exec_module(driver)


def test_report_identity_sees_draws_size_quality_and_sampler():
    record = {"name": "arm", "kind": "ar"}
    baseline = driver.report_path(record, 256, 5, True, 128, "random", "mps",
                                  SamplingConfig(40, 1.0))
    neighbours = {
        driver.report_path(record, 128, 5, True, 128, "random", "mps",
                           SamplingConfig(40, 1.0)),
        driver.report_path(record, 256, 2, True, 128, "random", "mps",
                           SamplingConfig(40, 1.0)),
        driver.report_path(record, 256, 5, True, 64, "random", "mps",
                           SamplingConfig(40, 1.0)),
        driver.report_path(record, 256, 5, False, 128, "random", "mps",
                           SamplingConfig(40, 1.0)),
        driver.report_path(record, 256, 5, True, 128, "random", "cpu",
                           SamplingConfig(40, 1.0)),
        driver.report_path(record, 256, 5, True, 128, "random", "mps",
                           SamplingConfig(80, 1.0)),
        driver.report_path(record, 256, 5, True, 128, "random", "mps",
                           SamplingConfig(40, 1.2)),
    }

    assert baseline not in neighbours
    assert len(neighbours) == 7


def test_planner_order_is_keyed_but_irrelevant_to_ar_reports():
    planner = {"name": "planner", "kind": "planner"}
    ar = {"name": "ar", "kind": "ar"}
    sampler = SamplingConfig()

    assert (
        driver.report_path(planner, 256, 5, True, 128, "random", "mps", sampler)
        != driver.report_path(planner, 256, 5, True, 128, "confidence", "mps", sampler)
    )
    assert (
        driver.report_path(ar, 256, 5, True, 128, "random", "mps", sampler)
        == driver.report_path(ar, 256, 5, True, 128, "confidence", "mps", sampler)
    )


def test_report_driver_refuses_empty_exclusion():
    message = driver.empty_failure({"empty": 0.25}, "arm", 100, 3)
    assert message and "25/100" in message
    with pytest.raises(SystemExit, match="unequal set sizes"):
        driver.require_no_empty({"empty": 0.25}, "arm", 100, 3)


def test_nonempty_draw_has_no_failure():
    assert driver.empty_failure({"empty": 0.0}, "arm", 100, 3) is None


def test_failure_summary_can_be_persisted_atomically(tmp_path):
    path = tmp_path / "failed.json"
    partial = {"empty": 0.25, "coverage": None}
    report = {
        "report_schema": 2, "status": "failed", "draws": [partial],
        "failure": {"kind": "empty_geometry", "seed": 3},
    }
    driver.write_report(path, report)
    assert json.loads(path.read_text()) == report
