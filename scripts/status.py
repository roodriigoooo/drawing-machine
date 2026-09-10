#!/usr/bin/env python3
"""What is training, what has landed, and what is still queued.

A sweep is hours of detached compute across several shells, and the failure this
guards against is real: a run was once orphaned and a second launch raced it
into the same log, overwriting the first's records. So this answers three
questions in one place, from the ground truth rather than from memory.

  1. Which trainers are alive *right now* (there must be exactly one).
  2. Which run records exist, grouped by regime, newest last -- `runs/*.json` is
     the durable log of every command that ever completed, because each record
     carries the full `TrainConfig` that produced it.
  3. What each detached log says on its last line.

    python3 scripts/status.py            # everything
    python3 scripts/status.py --since 60 # only records written in the last hour
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path

RUNS = Path(__file__).resolve().parent.parent / "runs"
#: Anything that trains. Matched against the full command line.
TRAINERS = r"dm\.train|sweep\.py|scale_separation\.py|rerun\.py"


def alive() -> list[str]:
    # `check=False`: grep exits 1 when nothing matches, and "no trainer is
    # running" is the answer this exists to give, not an error.
    found = subprocess.run(
        ["bash", "-lc", f'ps -eo pid,etime,command | grep -E "{TRAINERS}" | grep -v grep'],
        capture_output=True, text=True, check=False,
    ).stdout.strip().splitlines()
    return [" ".join(line.split()[:2] + line.split()[-3:]) for line in found]


def records(since_minutes: float | None) -> dict[str, list[tuple[float, str, dict]]]:
    """Completed runs, keyed by regime, as (mtime, name, headline numbers)."""
    cutoff = time.time() - since_minutes * 60 if since_minutes else 0.0
    out: dict[str, list] = defaultdict(list)
    for path in RUNS.glob("*.json"):
        if path.stat().st_mtime < cutoff or path.name.startswith("summary"):
            continue
        try:
            record = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue  # a record still being written
        if not isinstance(record, dict):
            continue  # not a training record -- eval drivers write their own shapes here
        final = record.get("final") or {}
        if not final:
            continue
        parts = record.get("name", path.stem).split("_")
        regime = parts[1] if len(parts) > 2 else "other"
        out[regime].append((
            path.stat().st_mtime,
            record["name"],
            {
                "steps": record.get("config", {}).get("steps"),
                "bits": final.get("bits_per_drawing"),
                "mins": round(final.get("elapsed_s", 0) / 60, 1),
            },
        ))
    return out


def progress() -> list[str]:
    """How far the running cell is, from the last eval line of the freshest log.

    A run that is silent for longer than its eval interval is the signature of a
    wedge, not of slowness (see PLAN.md section 10), so the age of the last line
    is reported next to the step count -- that is the number worth looking at.
    """
    # The freshest log that has actually reported an eval. A newly queued job
    # creates its log the moment it is launched, so "newest" alone points at an
    # empty file while the real run is still stepping in the previous one.
    logs = sorted(RUNS.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    logs = [p for p in logs if "bits/drawing" in p.read_text()] or logs
    if not logs:
        return []
    log = logs[0]
    text = log.read_text()
    steps = re.findall(r"steps=(\d+)", text)
    evals = re.findall(r"step\s+(\d+)\s+loss.*?(\d+)s\s*$", text, re.MULTILINE)
    name = re.findall(r"^\[([a-z0-9_]+)\] params=", text, re.MULTILINE)
    if not evals or not steps:
        return [f"{log.name}: no eval line yet"]
    done, secs = int(evals[-1][0]), int(evals[-1][1])
    total = int(steps[-1])
    eta = (total - done) * secs / max(1, done) / 60
    age = (time.time() - log.stat().st_mtime) / 60
    if done >= total:
        flag = "  -- finished"
    elif age > 5:
        # Silence past an eval interval is the signature of the Metal wedge in
        # PLAN.md section 10, where a live process wrote nothing for 78 minutes.
        flag = "  <-- SILENT, check with `sample <pid> 4`"
    else:
        flag = ""
    detail = (
        f"  step {done:,}/{total:,} ({100 * done / total:.0f}%)  "
        f"{secs / 60:.0f} min in, ~{eta:.0f} min left  "
        f"(last line {age:.1f} min ago){flag}"
    )
    return [name[-1] if name else log.stem, detail]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", type=float, default=None, metavar="MINUTES")
    args = ap.parse_args()

    running = alive()
    print(f"== training now ({len(running)}; expect 0 or 1) ==")
    for line in running or ["  nothing"]:
        print(f"  {line}")

    print("\n== completed runs ==")
    grouped = records(args.since)
    for regime in sorted(grouped):
        rows = sorted(grouped[regime])
        print(f"  {regime} ({len(rows)})")
        for mtime, name, info in rows:
            when = time.strftime("%H:%M", time.localtime(mtime))
            bits = f"{info['bits']:.2f}" if info["bits"] is not None else "  -  "
            print(f"    {when}  {name:48s} {info['steps']!s:>6} steps  "
                  f"{bits} bits  {info['mins']:>5.1f} min")

    print("\n== in progress ==")
    for line in progress() or ["  nothing training"]:
        print(f"  {line}")

    print("\n== detached logs ==")
    for log in sorted(RUNS.glob("*.log"), key=lambda p: p.stat().st_mtime):
        lines = [ln.rstrip() for ln in log.read_text().splitlines() if ln.strip()]
        stale = "  (stale)" if (time.time() - log.stat().st_mtime) > 3600 else ""
        print(f"  {log.name:22s}{stale:9s} {lines[-1][:74] if lines else '(empty)'}")
    print("\nEvery completed command is a record in runs/*.json carrying the "
          "TrainConfig that produced it; that is the durable log, not the terminal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
