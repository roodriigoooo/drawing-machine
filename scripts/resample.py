#!/usr/bin/env python3
"""Generation columns off a checkpoint, over `k` seeds, with their spread.

Three faults this exists for. The first two are the same fault at two levels;
the third is that there was no sample-quality half at all.

**A generation column is a draw, and this project has been reading it as a
number.** `gen_length_emd` is quoted from the *final* eval of a run. Measured
here: the flat AR baseline's final eval reads **2.72** while the same
checkpoint over five seeds reads **10.40 +- 4.48**, and its own last 25 evals
span 2.72 to 42.45. `PLAN.md` section 7 concluded from the 2.72 that "section
9.5a's termination failure barely exists here, so claim 3's termination
prediction had almost no room to win on this venue" -- a conclusion drawn from
the minimum of 25 draws. Every rule this project has about resolution floors
was written for `bits_per_drawing` and none of them was ever applied to the
sampling columns.

**A sampler is a hypothesis and a record cannot re-run it.** When
`CompositionDenoiser.sample`'s unmasking order turned out to be wrong, three
planner records' generation columns described the sampler rather than the
model, and a record's columns must come from the run that wrote them
(`PLAN.md` section 10). So the correction goes here, in a side report keyed by
checkpoint *and* sampler, exactly as `scripts/recovery.py` is keyed by
evaluation corpus -- and for the same reason, which is that the reading is a
*comparison* between two reports and a collision loses the result rather than
a file.

**And `bits_per_drawing` cannot see sample quality, so every column above is a
marginal.** `validity`, `length_emd` and `strokes` each compare one summary
statistic of the samples against the corpus's, and a distribution can match
every marginal anyone thought to check while matching nothing else.
`dm/eval/quality.py` compares the two *sets* of drawings in the geometry the VM
emits, and this script reports its three columns beside the marginals with a
**floor measured at the same set sizes on real drawings** -- without which none
of the three has a readable value. `--no-quality` skips it; it is the slow half.

    # the corrected columns for a planner run, both orderings
    python3 scripts/resample.py runs/quickdraw_plannerdiff24000eps2_*.pt --seeds 5
    python3 scripts/resample.py runs/quickdraw_plannerdiff24000eps2_*.pt \
        --order confidence

    # an AR arm, for the reference the planner has to be read against
    python3 scripts/resample.py runs/quickdraw_planbase24000eps2_*.pt --seeds 5

    # sampler attribution: settings are flags and part of the report identity
    python3 scripts/resample.py runs/quickdraw_planbase12000eps2_byte_square_s0.pt \\
        --seeds 5 --n 256 --top-k none --temperature 1.0 --device mps

CPU by default and no training, so it can run while a sweep does.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dm.data.dataset import ProgramDataset
from dm.eval.metrics import corpus_stats, length_stats, sample_programs
from dm.eval.quality import CLOUD_POINTS, clouds_of, quality_of, subsample
from dm.eval.records import corpus_config, is_planner, load
from dm.eval.reports import json_safe, staged_path
from dm.eval.sampling import SamplingConfig, parse_temperature, parse_top_k
from dm.isa.codec import CODECS
from dm.isa.strokes import stroke_count
from dm.models.planner import generate
from dm.train import RUNS, build_data, config_from_record

#: Columns worth a spread. `validity` and `truncated` are fractions of the
#: sample and move far less than the length columns, but they are draws too and
#: a report that gave one of them a bare number would be repeating the fault.
COLUMNS = ("length_emd", "len_p50", "validity", "truncated", "strokes", "planned")


def write_report(path: Path, report: dict) -> None:
    """Publish strict JSON atomically; consumers must reject failed geometry."""
    staged = staged_path(path)
    staged.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False))
    staged.replace(path)


#: The sample-quality half (`dm/eval/quality.py`). Separate from `COLUMNS`
#: because these are the only ones with a floor: the marginals above are read
#: against the corpus's own value for the same statistic, and these are read
#: against the same computation run on real drawings at the same set sizes.
QUALITY_COLUMNS = ("coverage", "mmd", "nna", "empty")

#: Which real drawings stand as the reference. Fixed and separate from the draw
#: seeds, because a reference that moved between arms would make two reports of
#: one corpus incomparable -- the run-4 fault -- and one that moved between
#: seeds would put the reference's own variance into the model's spread.
REFERENCE_SEED = 1_000


#: `is_planner`, `corpus_config` and `load` moved to `dm/eval/records.py` on
#: 2026-08-11, unchanged, when `scripts/redundancy.py` needed the same three.
#: The `corpus_config` rule is the one that must have exactly one copy.


def report_path(record: dict, n: int, seeds: int, quality: bool, points: int,
                order: str, device: str, sampler: SamplingConfig) -> Path:
    """Every input that can change a resample report appears in its identity."""
    quality_key = f"p{points}_r{REFERENCE_SEED}" if quality else "noquality"
    sampler_kind = order if is_planner(record) else "ar"
    device_key = str(device).replace(":", "-").replace("/", "-")
    slug = (
        f"{record['name']}_gen_{sampler_kind}_n{n}x{seeds}_"
        f"{quality_key}_d{device_key}_{sampler.slug}"
    )
    return RUNS / f"{slug}.json"


def empty_failure(stats: dict, name: str, n: int, seed: int) -> str | None:
    """Why geometry cannot publish, while leaving persistence to the caller."""
    if not stats.get("empty", 0):
        return None
    return (
        f"{name} produced {stats['empty'] * n:g}/{n} empty drawings at seed "
        f"{seed}; geometry would use unequal set sizes. Report the failure; "
        "do not score survivors under another estimator."
    )


def require_no_empty(stats: dict, name: str, n: int, seed: int) -> None:
    """Do not publish geometry after empty exclusion changes set sizes."""
    if message := empty_failure(stats, name, n, seed):
        raise SystemExit(message)


def draw(model, record: dict, codec, programs: list[bytes], seed: int,
         args) -> tuple[dict, list[bytes]]:
    """One seed's generation columns, plus the drawings behind them.

    The programs come back because the sample-quality half is computed on
    *these* samples rather than on a fresh draw. Two draws of one checkpoint
    differ by more than the effects under test -- that is what this whole script
    exists to establish -- so a report whose marginals and whose set comparison
    described different samples would be two reports stapled together.
    """
    torch.manual_seed(seed)
    if is_planner(record):
        out = generate(
            model, codec, n=args.n, steps=record["config"]["gen_steps"],
            device=args.device, temperature=args.temperature, top_k=args.top_k,
            order=args.order, forbid_specials=True,
        )
        stats = corpus_stats(out.programs)
        stats |= length_stats(out.programs, programs, out.cap_hit)
        stats["strokes"] = sum(map(stroke_count, out.programs)) / max(1, len(out.programs))
        stats["planned"] = sum(out.planned) / max(1, len(out.planned))
        return stats, out.programs
    # The AR arm caps a whole program rather than each stroke, and the cap is
    # derived from the val split the way `dm.train` derives it -- a different
    # cap makes `truncated` and `length_emd` describe a different experiment.
    cfg = config_from_record(record)
    p99 = ProgramDataset(programs, codec, cfg.max_len).length_stats()["p99"]
    # A class-conditional arm has to be sampled under *some* class, and the honest
    # default is the trainer's own: round-robin, so the sampled set is balanced the
    # way the reference split is and `length_emd` is not reporting the sampler's
    # choice of classes. `scripts/conditioning.py` is what asks per class.
    n_classes = record["model"].get("n_classes", 0)
    classes = (torch.arange(args.n, device=args.device) % n_classes
               if n_classes else None)
    sampled, cap_hit = sample_programs(
        model, codec, n=args.n, max_new=min(cfg.max_len, int(cfg.gen_cap * p99)),
        device=args.device, temperature=args.temperature, top_k=args.top_k,
        classes=classes, forbid_specials=True,
    )
    stats = corpus_stats(sampled)
    stats |= length_stats(sampled, programs, cap_hit)
    return stats, sampled


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", type=Path, nargs="+")
    ap.add_argument("--seeds", type=int, default=5,
                    help="draws per checkpoint; the point of the script is >1")
    ap.add_argument("--n", type=int, default=256, help="samples per draw")
    ap.add_argument("--order", default="random", choices=("random", "confidence"),
                    help="composition unmasking order; planner arms only")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--top-k", type=parse_top_k, default=40, metavar="K|none",
                    help="AR truncation; 'none' keeps every output byte symbol")
    ap.add_argument("--temperature", type=parse_temperature, default=1.0)
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing report for this exact setting")
    ap.add_argument("--no-quality", dest="quality", action="store_false",
                    help="skip the sample-quality half (`dm/eval/quality.py`), "
                         "which is the slow one -- one Chamfer matrix per seed "
                         "for the samples and one more for the floor")
    ap.add_argument("--cloud-points", type=int, default=CLOUD_POINTS,
                    help="points each drawing is resampled to; a Chamfer number "
                         "is comparable only with another at the same value")
    args = ap.parse_args()
    if args.seeds < 1 or args.n < 1 or args.cloud_points < 1:
        ap.error("--seeds, --n and --cloud-points must all be positive")
    sampler = SamplingConfig(args.top_k, args.temperature)

    # Refuse collisions before loading a model or rebuilding a corpus. With
    # several checkpoints, discovering the second collision after finishing the
    # first would still lose wall time and leave a half-completed invocation.
    outputs = []
    for path in args.checkpoints:
        record = json.loads(path.with_suffix(".json").read_text())
        outputs.append(report_path(record, args.n, args.seeds, args.quality,
                                   args.cloud_points, args.order, args.device, sampler))
    if len(set(outputs)) != len(outputs):
        raise SystemExit("the checkpoint list maps to duplicate report paths")
    if not args.overwrite:
        existing = [path for path in outputs if path.exists()]
        if existing:
            raise SystemExit(
                f"{existing[0]} already exists; pass --overwrite to replace this exact reading"
            )

    # Keyed by the val split's own fingerprint, because the floor is a property
    # of the corpus and the sample size and not of any checkpoint: measuring it
    # again for the second arm on one corpus would spend the same minutes to
    # print the same number, and worse, print a *different* one and invite the
    # reader to difference two floors that are the same quantity.
    floors: dict[tuple, list[dict]] = {}
    failures: list[str] = []
    for path in args.checkpoints:
        model, record = load(path)
        codec = CODECS[record["config"]["codec"]]
        train_programs, programs = build_data(corpus_config(record))
        model.to(args.device)
        planner = is_planner(record)
        out = report_path(record, args.n, args.seeds, args.quality,
                          args.cloud_points, args.order, args.device, sampler)

        reference = None
        if args.quality:
            # Subsampled to the sample count, and with a seed of its own so that
            # every arm on one corpus is scored against the identical reference
            # and no draw of the model can move it. `subsample` says why the two
            # sets have to be the same size.
            reference, _ = clouds_of(subsample(programs, args.n, REFERENCE_SEED),
                                     args.cloud_points)
        draws = []
        failure: dict | None = None
        for seed in range(args.seeds):
            print(f"{record['name']}: draw {seed + 1}/{args.seeds}", flush=True)
            stats, sampled = draw(model, record, codec, programs, seed, args)
            if reference is not None:
                stats |= quality_of(sampled, reference, args.cloud_points)
                if message := empty_failure(stats, record["name"], args.n, seed):
                    failure = {"kind": "empty_geometry", "seed": seed,
                               "message": message}
            draws.append(stats)
            if failure:
                break

        floor: list[dict] = []
        if reference is not None:
            # Real drawings from the *training* split, disjoint from the
            # reference by construction, at the sample count the model was given.
            key = ((record.get("corpus") or {}).get("val"), args.n, args.cloud_points)
            cached = floors.get(key) if key[0] else None
            floor = cached if cached is not None else [
                quality_of(subsample(train_programs, args.n, seed), reference,
                           args.cloud_points)
                for seed in range(args.seeds)
            ]
            # Only when the record could name its corpus. A record predating
            # `dm/data/fingerprint.py` has no key, and caching one under `None`
            # would hand the second checkpoint a floor measured on a different
            # val split -- the run-4 fault, in a cache.
            if key[0]:
                floors[key] = floor

        # The sampler, draw count, set size and quality resolution are all part
        # of the identity. Omitting any one lets a smoke test overwrite a result.
        summary = {
            "report_schema": 2, "status": "failed" if failure else "complete",
            "name": record["name"], "kind": record.get("kind", "ar"),
            "order": args.order if planner else None,
            "sampler": sampler.as_dict(), "device": args.device,
            "seeds": args.seeds, "n": args.n,
            "cloud_points": args.cloud_points if args.quality else None,
            "recorded": {c: record["final"].get(f"gen_{c}") for c in COLUMNS},
            "draws": draws,
            "floor": floor,
        }
        if failure:
            summary["failure"] = failure
            write_report(out, summary)
            failures.append(failure["message"])
            print(f"\n  FAILED -> {out}", flush=True)
            continue
        print(f"\n## {record['name']}  ({codec.name}, "
              f"{record['model']['params']:,} params"
              + (f", order={args.order}" if planner else "") + ")")
        print(f"{args.seeds} seeds x {args.n} samples, against the run's final eval")
        print(f"sampler: top_k={sampler.top_k}, temperature={sampler.temperature:g}, "
              "PAD/BOS forbidden\n")
        print(f"  {'column':<12}{'mean':>9}{'sd':>8}{'min':>9}{'max':>9}"
              f"{'recorded':>11}{'floor':>16}")
        for column in (*COLUMNS, *QUALITY_COLUMNS):
            values = [d[column] for d in draws if column in d]
            if not values:
                continue
            was = record["final"].get(f"gen_{column}")
            sd = st.stdev(values) if len(values) > 1 else float("nan")
            summary[f"{column}_mean"], summary[f"{column}_sd"] = st.mean(values), sd
            real = [f[column] for f in floor if column in f]
            print(f"  {column:<12}{st.mean(values):9.3f}{sd:8.3f}{min(values):9.3f}"
                  f"{max(values):9.3f}"
                  + (f"{was:11.3f}" if isinstance(was, (int, float)) else f"{'--':>11}")
                  + (f"{st.mean(real):10.3f} ±{st.stdev(real):.3f}"
                     if len(real) > 1 else f"{'--':>16}"))
        if floor:
            print("\n  `floor` is the same computation on real drawings held out of the "
                  "reference, at\n  the same two set sizes. Nothing above it is a number "
                  "on its own: `coverage` is\n  bounded by n/m before a model does "
                  "anything, `mmd` is in this corpus's pixels,\n  and `nna`'s textbook "
                  "0.5 holds only when the two sets are the same size.")
        write_report(out, summary)
        print(f"\n  -> {out}")
    if failures:
        raise SystemExit(
            f"{len(failures)} checkpoint(s) failed structural guards; "
            "failure reports were preserved"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
