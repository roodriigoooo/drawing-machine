"""Sampler summary never promotes failed survivor geometry."""

import importlib.util
import json
import sys
from pathlib import Path

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "sampler_attribution.py"
_spec = importlib.util.spec_from_file_location("sampler_attribution_script", _PATH)
assert _spec and _spec.loader
script = importlib.util.module_from_spec(_spec)
sys.modules["sampler_attribution_script"] = script
_spec.loader.exec_module(script)


def test_failed_report_without_failure_record_is_refused(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"report_schema": 2, "status": "failed"}))

    try:
        script.read_report(path)
    except ValueError as exc:
        assert "carries no failure record" in str(exc)
    else:
        raise AssertionError("malformed failed report was accepted")


def test_failed_report_has_no_geometry_draw_means():
    report = {
        "status": "failed",
        "categories": ["cat"],
        "sampling_guards": {"0": [{"validity": 1.0, "empty": 0.01,
                                      "truncated": 0.0, "faults": {}}]},
        "draws": [{"diagonal": {"0": {"coverage": 0.99}}}],
        "failure": {"kind": "empty_geometry"},
    }

    assert script.draw_means(report, "coverage") == []


def test_class_guard_counts_use_per_class_sample_size():
    report = {
        "n": 100, "categories": ["cat", "bus"],
        "sampling_guards": {
            "0": [{"validity": 1.0, "empty": 0.01, "truncated": 0.0, "faults": {}}],
            "1": [{"validity": 1.0, "empty": 0.0, "truncated": 0.0, "faults": {}}],
        },
    }

    assert script.guard_summary(report)[2] == 1


def test_complete_class_report_averages_classes_per_draw():
    report = {
        "status": "complete", "categories": ["cat", "bus"],
        "sampling_guards": {"0": [], "1": []},
        "draws": [{"diagonal": {
            "0": {"coverage": 0.4}, "1": {"coverage": 0.6},
        }}],
    }

    assert script.draw_means(report, "coverage") == [0.5]
