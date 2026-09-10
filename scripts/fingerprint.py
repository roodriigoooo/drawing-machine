"""Backfill the `corpus` fingerprint into run records written before it existed.

Records are the project's only durable output, and a fingerprint that only new
runs carry would split every table in two: a config-derived key for the old
rows and a digest for the new ones, which cannot be compared and so cannot be
grouped. Rebuilding the corpus from `config_from_record` closes that.

The rebuild is exact by construction, and that is the whole reason this is
allowed to exist. `config_from_record` is the supported inverse of what `train`
serialises, and `build_data` is the single code path every corpus goes through
(`PLAN.md` section 8): a val split regenerated with default arguments is a
*different* val split, worth ~3.5 bits/drawing, so a fingerprint computed any
other way would be worse than none at all.

    python3 scripts/fingerprint.py            # report what is missing
    python3 scripts/fingerprint.py --write    # compute and write it in

Verification, not trust: `--write` refuses any record whose rebuilt val split
does not match the `val_bytes` the run itself recorded. That check is what makes
the backfill evidence rather than an assumption -- if a corpus has moved under a
record, the record is named and left alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dm.data.dataset import ProgramDataset  # noqa: E402
from dm.data.fingerprint import fingerprint  # noqa: E402
from dm.isa.codec import CODECS  # noqa: E402
from dm.isa.spec import Tier  # noqa: E402
from dm.train import RUNS, build_data, config_from_record  # noqa: E402
from dm.train_planner import PlannerTrainConfig  # noqa: E402


def rebuild(record: dict, cache: dict) -> tuple[list[bytes], list[bytes]]:
    """The record's own corpus, memoised on the config that names it.

    Tier B rebuilds read 350k programs from an npz cache; a sweep has dozens of
    records over a handful of corpora, so without this the backfill re-reads the
    same corpus once per record.

    A planner record's config is a `PlannerTrainConfig`, which carries
    `max_strokes` and no `max_len`, so it is not a `TrainConfig` and must go
    through `PlannerTrainConfig.corpus()` -- the same indirection the trainer
    uses, and the reason both models are guaranteed to name one corpus the same
    way.
    """
    config = record["config"]
    key = json.dumps(
        {k: config.get(k) for k in
         ("data", "categories", "tier", "n_train", "n_val", "data_seed", "extra", "augment")},
        sort_keys=True, default=str,
    )
    if key not in cache:
        if record.get("kind") == "planner":
            planner = PlannerTrainConfig(**{
                **config, "tier": Tier(config["tier"]),
                "categories": tuple(config["categories"]),
            })
            cache[key] = build_data(planner.corpus())
        else:
            cache[key] = build_data(config_from_record(record))
    return cache[key]


def check(record: dict, val: list[bytes]) -> str | None:
    """None if the rebuilt val split is the one this record was scored on.

    Two levels, because `val_bytes` postdates most of the sweep and refusing
    every record older than a column would leave the tables split exactly the
    way the fingerprint exists to prevent.

    `val_bytes` is the per-program bytecode length the run wrote down, so it
    pins the split program by program. Where it is absent, `val_lengths` is the
    same distribution in the *codec's* symbols and is reproduced exactly by
    re-encoding the rebuild through the record's own codec and `max_len` --
    mean, median, p99, max and the truncated fraction, five statistics over
    1,000 programs. That is not proof of program-for-program identity, but a
    different val split does not reproduce five moments of its own length
    distribution, and the alternative is no check at all.
    """
    recorded = record.get("val_bytes")
    if recorded:
        rebuilt = [len(p) for p in val]
        if rebuilt != recorded:
            return (f"val split differs: rebuilt {len(rebuilt)} programs, "
                    f"recorded {len(recorded)}")
        return None

    lengths = record.get("val_lengths")
    if not lengths:
        return "neither val_bytes nor val_lengths to verify against"
    config = record["config"]
    got = ProgramDataset(val, CODECS[config["codec"]], config["max_len"]).length_stats()
    for stat, want in lengths.items():
        if abs(got[stat] - want) > 1e-9:
            return f"val_lengths[{stat}] rebuilt {got[stat]} against recorded {want}"
    if len(val) != len(record.get("val_bits") or val):
        return f"rebuilt {len(val)} val programs, recorded {len(record['val_bits'])} scores"
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="write the fingerprint in")
    ap.add_argument("--runs", type=Path, default=RUNS)
    args = ap.parse_args()

    cache: dict = {}
    written = skipped = refused = 0
    for path in sorted(args.runs.glob("*.json")):
        record = json.loads(path.read_text())
        if "config" not in record or "final" not in record:
            continue  # not a run record -- quickdraw_check.py writes here too
        if record.get("corpus", {}).get("val"):
            skipped += 1
            continue
        try:
            train, val = rebuild(record, cache)
        except Exception as exc:  # noqa: BLE001 -- a corpus that no longer builds
            print(f"REFUSED {path.name}: cannot rebuild -- {exc}")
            refused += 1
            continue
        problem = check(record, val)
        if problem:
            print(f"REFUSED {path.name}: {problem}")
            refused += 1
            continue
        marks = fingerprint(train, val)
        print(f"{'write' if args.write else 'would write'} {path.name}: "
              f"val {marks['val']} n_train {marks['n_train']:,}")
        if args.write:
            # `corpus` after `config`, so a record reads spec-then-identity.
            ordered = {}
            for key, value in record.items():
                ordered[key] = value
                if key == "config":
                    ordered["corpus"] = marks
            path.write_text(json.dumps(ordered, indent=2))
        written += 1

    verb = "written" if args.write else "pending"
    print(f"\n{written} {verb}, {skipped} already fingerprinted, {refused} refused")
    if refused and args.write:
        print("A refused record is a record whose corpus has moved under it. "
              "It keeps the config-derived key and stays out of any fingerprinted "
              "comparison, which is the safe direction.")


if __name__ == "__main__":
    main()
