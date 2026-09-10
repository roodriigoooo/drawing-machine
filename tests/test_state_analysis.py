from __future__ import annotations

import math

from dm.eval.state_analysis import holm, paired_summary, sign_flip_pvalue, t_critical_975


def test_five_draw_interval_uses_student_t_not_normal():
    assert math.isclose(t_critical_975(5), 2.7764451052)
    result = paired_summary([1.0, 1.0, 1.0, 1.0, 1.0])
    assert result["ci95_half_width"] == 0.0
    assert result["p_value"] == 0.0
    assert result["p_exact_sign_flip"] == 0.0625


def test_sign_flip_and_holm_are_deterministic():
    assert sign_flip_pvalue([1.0, -1.0]) == 1.0
    result = holm({"b": 0.02, "a": 0.001, "c": 0.2, "d": 0.04})
    assert result["a"]["reject"]
    assert not result["b"]["reject"]
    assert not result["c"]["reject"]
    assert result["a"]["rank"] == 1
