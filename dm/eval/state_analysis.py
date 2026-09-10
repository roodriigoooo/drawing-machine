"""Small-sample paired inference for the frozen state-study contract.

This Module keeps the inferential choices out of the report driver.  The draw
seeds are paired by construction, so the primary comparison is a paired delta;
five draws are too few for a normal critical value to be a defensible default.
The interval and primary p-value are two-sided Student-t quantities; the exact
paired sign-flip p-value is retained as a robustness diagnostic. Holm correction
is applied to the four checkpoint
families within each model seed, never across seeds or secondary endpoints.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Mapping

from scipy.stats import t as student_t

# Two-sided .975 quantiles for df 1..30, followed by the normal limit.  The
# state study's frozen draw count is five (df=4), but keeping the small table
# makes the helper useful for contract tests and future smoke runs.
T975 = (
    float("nan"), 12.7062047364, 4.30265272975, 3.18244630528,
    2.77644510520, 2.57058183564, 2.446911846, 2.364624252, 2.306004135,
    2.262157163, 2.228138852, 2.200985160, 2.178812830, 2.160368656,
    2.144786688, 2.131449546, 2.119905299, 2.109815578, 2.100922040,
    2.093024054, 2.085963447, 2.079613845, 2.073873068, 2.068657610,
    2.063898562, 2.059538553, 2.055529439, 2.051830516, 2.048407142,
    2.045229642, 1.96,
)


def t_critical_975(n: int) -> float:
    """Two-sided 95% critical value for ``n`` paired observations."""
    if n < 2:
        return float("nan")
    df = n - 1
    return T975[df] if df < len(T975) else T975[-1]


def sign_flip_pvalue(values: Iterable[float]) -> float:
    """Exact two-sided paired sign-flip p-value for a mean delta."""
    deltas = tuple(float(value) for value in values)
    n = len(deltas)
    if not n:
        return float("nan")
    observed = abs(sum(deltas) / n)
    means = []
    for signs in itertools.product((-1.0, 1.0), repeat=n):
        means.append(abs(sum(sign * value for sign, value in zip(signs, deltas)) / n))
    # Include the observed boundary, as required by the exact randomisation
    # test; the tiny tolerance only absorbs floating point summation order.
    return sum(value + 1e-12 >= observed for value in means) / len(means)


def student_t_pvalue(values: Iterable[float]) -> float:
    """Two-sided paired Student-t p-value for small-sample deltas."""
    deltas = [float(value) for value in values]
    n = len(deltas)
    if n < 2:
        return float("nan")
    mean = sum(deltas) / n
    variance = sum((value - mean) ** 2 for value in deltas) / (n - 1)
    if variance == 0.0:
        return 0.0 if mean else 1.0
    statistic = abs(mean) / math.sqrt(variance / n)
    return float(2.0 * student_t.sf(statistic, n - 1))


def paired_summary(values: Iterable[float]) -> dict:
    """Mean, small-sample t interval/p-value, and raw paired deltas."""
    deltas = [float(value) for value in values]
    n = len(deltas)
    mean = sum(deltas) / n if n else float("nan")
    if n > 1:
        variance = sum((value - mean) ** 2 for value in deltas) / (n - 1)
        stderr = math.sqrt(variance / n)
        half = t_critical_975(n) * stderr
        ci = [mean - half, mean + half]
    else:
        stderr = float("nan")
        ci = [float("nan"), float("nan")]
    p_student = student_t_pvalue(deltas)
    return {
        "n": n,
        "mean": mean,
        "sd": (math.sqrt(sum((value - mean) ** 2 for value in deltas) / (n - 1))
               if n > 1 else float("nan")),
        "stderr": stderr,
        "ci95": ci,
        "ci95_half_width": (ci[1] - mean if n > 1 else float("nan")),
        "p_value": p_student,
        "p_student_t": p_student,
        "p_exact_sign_flip": sign_flip_pvalue(deltas),
        "deltas": deltas,
    }


def holm(pvalues: Mapping[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm step-down decisions and adjusted p-values.

    Sorting ties by family name makes the result deterministic and auditable.
    NaN p-values are never rejected and are sorted last.
    """
    items = sorted(
        pvalues.items(),
        key=lambda item: (math.isnan(float(item[1])), float(item[1]), item[0]),
    )
    m = len(items)
    adjusted: dict[str, dict] = {}
    running = 0.0
    stopped = False
    for rank, (name, raw) in enumerate(items):
        p = float(raw)
        if math.isnan(p):
            adj = float("nan")
            reject = False
            stopped = True
        else:
            adj = min(1.0, max(running, (m - rank) * p))
            running = adj
            reject = not stopped and p <= alpha / (m - rank)
            if not reject:
                stopped = True
        adjusted[name] = {
            "p_raw": p,
            "p_holm": adj,
            "rank": rank + 1,
            "critical": alpha / (m - rank) if m and not math.isnan(p) else float("nan"),
            "reject": reject,
        }
    return adjusted


def family_holm(rows: Mapping[str, Mapping], alpha: float = 0.05) -> dict:
    """Apply Holm to rows containing the frozen Student-t p-value."""
    return holm({name: float(row["p_value"])
                 for name, row in rows.items()}, alpha=alpha)


__all__ = [
    "family_holm",
    "holm",
    "paired_summary",
    "sign_flip_pvalue",
    "student_t_pvalue",
    "t_critical_975",
]
