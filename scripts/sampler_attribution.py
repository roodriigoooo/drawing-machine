#!/usr/bin/env python3
"""Summarise schema-2 sampler reports without treating failures as geometry.

Run after `scripts/class_quality.py` or `scripts/resample.py` sweeps. Complete
reports produce draw means. Failed reports produce only structural failure and
partial-guard provenance; their survivor geometry is never aggregated.

    python3 scripts/sampler_attribution.py runs/*_legal.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from collections import Counter
from pathlib import Path

QUALITY = ("coverage", "mmd", "nna")


def is_class_report(report: dict) -> bool:
    return "categories" in report and "sampling_guards" in report


def read_report(path: Path) -> dict:
    report = json.loads(path.read_text())
    if report.get("report_schema") != 2:
        raise ValueError(f"{path} is not a schema-2 report")
    if report.get("status", "complete") not in {"complete", "failed"}:
        raise ValueError(f"{path} has unknown status {report.get('status')!r}")
    if report.get("status") == "failed" and "failure" not in report:
        raise ValueError(f"{path} is failed but carries no failure record")
    return report


def guard_summary(report: dict) -> tuple[int, float, float, float, Counter]:
    if is_class_report(report):
        guards = [guard for rows in report["sampling_guards"].values() for guard in rows]
        requested = report["n"]
    else:
        draws = report.get("draws", [])
        guards = [
            {"validity": draw["validity"], "empty": draw.get("empty", 0.0),
             "truncated": draw["truncated"], "faults": draw["faults"]}
            for draw in draws
        ]
        requested = report["n"]
    faults: Counter = Counter()
    for guard in guards:
        faults.update(guard["faults"])
    if not guards:
        return 0, math.nan, 0.0, math.nan, faults
    return (
        len(guards), st.mean(g["validity"] for g in guards),
        sum(g["empty"] * requested for g in guards),
        st.mean(g["truncated"] for g in guards), faults,
    )


def draw_means(report: dict, metric: str) -> list[float]:
    if report.get("status", "complete") != "complete":
        return []
    if is_class_report(report):
        return [
            st.mean(draw["diagonal"][str(c)][metric]
                    for c in range(len(report["categories"])))
            for draw in report["draws"]
        ]
    return [draw[metric] for draw in report["draws"]]


def mean_sd(values: list[float]) -> str:
    if not values:
        return "blocked"
    spread = st.stdev(values) if len(values) > 1 else math.nan
    return f"{st.mean(values):.6f} ± {spread:.6f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    args = parser.parse_args()

    for path in args.reports:
        report = read_report(path)
        status = report.get("status", "complete")
        sampler = report["sampler"]
        guards, validity, empties, truncated, faults = guard_summary(report)
        print(f"{path.name}\n  status={status} top_k={sampler['top_k']} "
              f"temperature={sampler['temperature']:g}")
        print(f"  guards={guards} validity={validity:.6f} empties={empties:g} "
              f"truncated={truncated:.6f} faults={dict(faults)}")
        if status != "complete":
            failure = report["failure"]
            print(f"  failure={failure['kind']} seed={failure['seed']}: "
                  f"{failure['message']}\n")
            continue
        for metric in QUALITY:
            print(f"  {metric}={mean_sd(draw_means(report, metric))}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
