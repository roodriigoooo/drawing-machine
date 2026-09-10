"""Side reports use strict JSON-compatible values."""

import math

from dm.eval.reports import json_safe


def test_json_safe_replaces_nonfinite_readings_recursively():
    report = {
        "draws": [{"coverage": math.nan, "mmd": math.inf}],
        "matrix": (1.0, -math.inf),
    }

    assert json_safe(report) == {
        "draws": [{"coverage": None, "mmd": None}],
        "matrix": [1.0, None],
    }
