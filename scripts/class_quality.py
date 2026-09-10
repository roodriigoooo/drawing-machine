#!/usr/bin/env python3
"""Does asking for a cat produce cat-shaped geometry?

`controllability` (`scripts/conditioning.py`) reads 512/512 — and it is the
model grading its own homework: the classifier that checks each sample is the
model that drew it. This is the external check, and the first measurement in
this project that could honestly be called a generation result: samples drawn
under each class, scored as *sets* against **real drawings** of each class with
`dm/eval/quality.py`'s three metrics, at matched set sizes, against a per-class
floor measured the same way.

The reading is the 5×5 `mmd` matrix (rows = asked, cols = real) plus the
diagonal's three metrics against their floors:

- diagonal beats off-diagonal, near floor  → conditioning controls geometry;
- diagonal beats off-diagonal, far above   → it steers, and sample quality is
  the binding problem — claim 3's likelihood/quality split again;
- diagonal ties off-diagonal               → the 512/512 read-back is carried
  by cues geometry cannot see: the XFORM discriminate-but-not-generate shape,
  one level up.

The branches, the floor convention and the traps are pre-registered in
`PLAN.md` §5 of "What is owed".

> **The floor is a matrix, not a column, and that was the correction the first
> reading needed.** "The diagonal is the row minimum" is only half a result:
> how far it *should* win by is a property of the corpus, because Chamfer's
> separation of two classes is a fact about their shapes and not about the
> model. So the same `class_quality` call runs with real train-split drawings
> in the samples' place, giving the margin a perfect generator achieves — and a
> model margin is read against *that*, never against zero. It is the same move
> as reading `recovery` against its constructed ceiling. It costs ~40 s beside
> ~30 min of sampling, so it is computed unconditionally.

    # one checkpoint, against its matched unconditional control
    python3 scripts/class_quality.py runs/quickdraw_cond24000eps4_byte_square_s0.pt \\
        --control runs/quickdraw_m5si24000eps4_byte_square_s0.pt --device mps \\
        --top-k 40 --temperature 1.0

    # the floor matrix alone, merged into that exact report identity
    python3 scripts/class_quality.py runs/quickdraw_cond24000eps4_byte_square_s0.pt \\
        --control runs/quickdraw_m5si24000eps4_byte_square_s0.pt --device mps \\
        --top-k 40 --temperature 1.0 --floor-only

No training. ~25–40 min per checkpoint on mps; the sampling is the slow half.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dm.eval.metrics import corpus_stats, sample_programs
from dm.eval.quality import (
    CLOUD_POINTS,
    class_quality,
    clouds_of,
    distribution_metrics,
    subsample,
)
from dm.eval.records import load
from dm.eval.reports import json_safe, staged_path
from dm.eval.sampling import SamplingConfig, parse_temperature, parse_top_k
from dm.isa.codec import CODECS
from dm.train import RUNS, build_data, build_labels, config_from_record

#: Which real drawings stand as each class's reference. The same constant as
#: `scripts/resample.py`'s, for the same reason: the reference must be identical
#: across every arm and every draw on one corpus, and separate from the draw
#: seeds so no draw of the model can move it.
REFERENCE_SEED = 1_000


def by_class(programs: list[bytes], labels: list[int]) -> dict[int, list[bytes]]:
    out: dict[int, list[bytes]] = {}
    for program, label in zip(programs, labels):
        out.setdefault(label, []).append(program)
    return out


def split_with_labels(record: dict):
    """Both splits, both label lists, and the config — or a refusal.

    The labels ride on `build_labels`'s traversal, which is the same one that
    orders `build_data`'s programs, so a label cannot land on the wrong drawing
    — the invariant `quickdraw.load_labelled` exists to keep.
    """
    cfg = config_from_record(record)
    train, val = build_data(cfg)
    labels = build_labels(cfg)
    if labels is None:
        raise SystemExit(f"{record['name']} is on {cfg.data!r}, which has no classes")
    return cfg, train, val, labels[0], labels[1]


def sample_class(model, codec, n: int, max_new: int, c: int, device,
                 sampler: SamplingConfig) -> tuple[list[bytes], list[bool]]:
    """One class's samples for one draw. The class tensor is the whole ask."""
    return sample_programs(
        model, codec, n=n, max_new=max_new, device=device,
        temperature=sampler.temperature, top_k=sampler.top_k,
        classes=torch.full((n,), c, device=device), forbid_specials=True,
    )


def sample_guards(programs: list[bytes], cap_hit: list[bool]) -> dict:
    """Failure columns that must move beside any apparent quality gain."""
    stats = corpus_stats(programs)
    return {
        "validity": stats["validity"],
        "truncated": sum(cap_hit) / max(1, len(cap_hit)),
        "faults": stats["faults"],
    }


def mean_sd(values: list[float]) -> tuple[float, float]:
    clean = [v for v in values if not math.isnan(v)]  # a NaN cell prints as --
    if not clean:
        return float("nan"), float("nan")
    return st.mean(clean), (st.stdev(clean) if len(clean) > 1 else float("nan"))


def require_balanced_geometry(draws: list[dict], names: list[str], n: int,
                              seed_offset: int = 0) -> None:
    """Stop a setting whose empty exclusions would change metric set sizes."""
    for seed, draw in enumerate(draws):
        for c, name in enumerate(names):
            got = draw["diagonal"][c]
            if got["n_gen"] != n:
                raise SystemExit(
                    f"{name} sampler produced {n - got['n_gen']}/{n} empty drawings "
                    f"at draw seed {seed + seed_offset}; geometry would be "
                    f"{got['n_gen']} v {n}. "
                    "The failure is recorded above the quality axis; do not score an "
                    "unbalanced set."
                )


def margins(draws: list[dict], n_classes: int) -> dict[int, list[float]]:
    """Per class, `min off-diagonal − diagonal` in each draw.

    The row's own reading, kept per draw rather than taken off the averaged
    matrix: the two are not the same number when the nearest confuser changes
    between draws, and it is the per-draw one that says whether the diagonal
    won *every* time.
    """
    return {c: [min(v for j, v in enumerate(d["matrix"][c]) if j != c)
                - d["matrix"][c][c]
                for d in draws]
            for c in range(n_classes)}


def floor_reading(train_by: dict[int, list[bytes]], references: dict,
                  n_classes: int, n: int, seeds: int, points: int) -> list[dict]:
    """What a perfect class-conditional generator scores — the whole matrix.

    Real class-`c` drawings from the **training split** stand in for the
    samples, so the reading is the identical computation with the model
    removed: disjoint from the reference by `build_data`'s dedup, at the
    identical two set sizes, once per seed.

    **It is not a resolution floor.** It bounds quality, not run-to-run spread,
    and the off-diagonal cells bound *separation*: a model whose margin matches
    this one has taken all the margin the corpus offers.
    """
    return [class_quality({c: subsample(train_by[c], n, seed)
                           for c in range(n_classes)}, references, points)
            for seed in range(seeds)]


def print_matrix(matrix: list[list[float]], names: list[str], title: str) -> None:
    print(f"\n  {title}")
    print(f"  {'':<10}" + "".join(f"{names[c][:8]:>10}" for c in range(len(names))))
    for asked, row in enumerate(matrix):
        marker = " ←" if row[asked] == min(row) else " ✗"
        print(f"  {names[asked]:<10}" + "".join(f"{v:10.2f}" for v in row) + marker)


def validate_control(model, record: dict, control_model, control: dict) -> None:
    """Refuse a class-blind checkpoint that is not the matched training arm."""
    if control["model"].get("n_classes"):
        raise SystemExit(f"{control['name']} is itself conditional; the control must be class-blind")
    if control.get("schema") != record.get("schema"):
        raise SystemExit(
            f"{control['name']} has schema {control.get('schema')}, not "
            f"{record.get('schema')}; cross-schema training is not a matched control"
        )
    if not record.get("complete", True) or not control.get("complete", True):
        raise SystemExit("class quality needs two complete training records")
    primary_steps = record.get("steps_done", record["config"]["steps"])
    control_steps = control.get("steps_done", control["config"]["steps"])
    if control_steps != primary_steps:
        raise SystemExit(
            f"{control['name']} reached {control_steps} steps, not {primary_steps}"
        )
    if control.get("corpus") != record.get("corpus"):
        raise SystemExit(
            f"{control['name']} is not on the identical train and val corpus; "
            "a cross-split comparison measures the split"
        )

    primary_cfg, control_cfg = asdict(config_from_record(record)), asdict(
        config_from_record(control)
    )
    for key in ("conditional", "device", "tag"):
        primary_cfg.pop(key, None)
        control_cfg.pop(key, None)
    changed = sorted(k for k in primary_cfg.keys() | control_cfg.keys()
                     if primary_cfg.get(k) != control_cfg.get(k))
    if changed:
        raise SystemExit(
            f"{control['name']} is not the matched training arm; differing config "
            f"fields: {', '.join(changed)}"
        )

    primary_model, control_model_cfg = asdict(model.cfg), asdict(control_model.cfg)
    primary_model["n_classes"] = control_model_cfg["n_classes"] = 0
    changed = sorted(k for k in primary_model.keys() | control_model_cfg.keys()
                     if primary_model.get(k) != control_model_cfg.get(k))
    if changed:
        raise SystemExit(
            f"{control['name']} has a different model apart from the class table: "
            f"{', '.join(changed)}"
        )


def report_path(record: dict, control: dict | None, n: int, seeds: int,
                points: int, max_new: int, device: str,
                sampler: SamplingConfig) -> Path:
    """Every input that can change a report appears in its identity."""
    against = f"vs_{control['name']}" if control else "nocontrol"
    device_key = str(device).replace(":", "-").replace("/", "-")
    slug = (
        f"class_quality_{record['name']}_{against}_n{n}x{seeds}_p{points}_"
        f"r{REFERENCE_SEED}_m{max_new}_d{device_key}_{sampler.slug}"
    )
    return RUNS / f"{slug}.json"


def write_report(path: Path, report: dict) -> None:
    """Publish strict JSON atomically; consumers must reject failed geometry."""
    staged = staged_path(path)
    staged.write_text(json.dumps(json_safe(report), indent=2, default=float,
                                 allow_nan=False))
    staged.replace(path)


def make_report(record: dict, control_record: dict | None, names: list[str],
                args, sampler: SamplingConfig, max_new: int, draws: list[dict],
                floor: dict, floor_draws: list[dict], control_draws: list[dict],
                status: str = "complete", failure: dict | None = None) -> dict:
    """Complete reading or durable failure under the same exact identity."""
    report = {
        "report_schema": 2, "status": status,
        "name": record["name"],
        "control": control_record["name"] if control_record else None,
        "categories": names, "n": args.n, "seeds": args.seeds,
        "cloud_points": args.cloud_points, "reference_seed": REFERENCE_SEED,
        "max_new": max_new, "device": args.device, "sampler": sampler.as_dict(),
        "sampling_guards": {
            str(c): [{k: d["diagonal"][c][k]
                      for k in ("validity", "empty", "truncated", "faults")}
                     for d in draws]
            for c in range(len(names))
        },
        "control_sampling_guards": [d["sampling"] for d in control_draws],
        "draws": draws, "floor": floor,
        "floor_matrix": [d["matrix"] for d in floor_draws],
        "control_draws": control_draws,
    }
    if failure is not None:
        report["failure"] = failure
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path, help="the conditional arm")
    ap.add_argument("--control", type=Path, default=None,
                    help="matched unconditional checkpoint; its class-blind "
                         "samples are what 'just draw something' scores")
    ap.add_argument("--seeds", type=int, default=5, help="draws per class")
    ap.add_argument("--n", type=int, default=100,
                    help="samples per class per draw, and every set's size")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cloud-points", type=int, default=CLOUD_POINTS)
    ap.add_argument("--top-k", type=parse_top_k, default=40, metavar="K|none",
                    help="AR truncation; 'none' keeps every output byte symbol")
    ap.add_argument("--temperature", type=parse_temperature, default=1.0)
    ap.add_argument("--floor-only", action="store_true",
                    help="recompute the floor matrix and merge it into an "
                         "existing report, without sampling the model")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing report for this exact setting")
    args = ap.parse_args()
    if args.seeds < 1 or args.n < 1 or args.cloud_points < 1:
        ap.error("--seeds, --n and --cloud-points must all be positive")
    sampler = SamplingConfig(args.top_k, args.temperature)

    model, record = load(args.checkpoint)
    n_classes = record["model"].get("n_classes", 0)
    if not n_classes:
        raise SystemExit(f"{record['name']} was trained without --conditional; "
                         "there is no class to ask for")
    codec = CODECS[record["config"]["codec"]]
    cfg = config_from_record(record)
    names = list(cfg.categories)
    max_new = min(cfg.max_len,
                  int(record["config"]["gen_cap"] * record["val_lengths"]["p99"]))

    control_model = control_record = None
    if args.control:
        control_model, control_record = load(args.control)
        validate_control(model, record, control_model, control_record)
    out = report_path(record, control_record, args.n, args.seeds,
                      args.cloud_points, max_new, args.device, sampler)
    if out.exists() and not args.floor_only and not args.overwrite:
        raise SystemExit(f"{out} already exists; pass --overwrite to replace this exact reading")

    _, train, val, train_labels, val_labels = split_with_labels(record)
    train_by, val_by = by_class(train, train_labels), by_class(val, val_labels)
    for c in range(n_classes):
        if len(val_by.get(c, ())) < args.n or len(train_by.get(c, ())) < args.n:
            raise SystemExit(
                f"class {c} ({names[c]}) has {len(val_by.get(c, ()))} val / "
                f"{len(train_by.get(c, ()))} train programs against n={args.n}; "
                "a set it cannot fill is a different measurement, not a smaller one"
            )

    # One fixed, seeded reference per class. Every comparison below — samples,
    # floor and control — is against these clouds and no others.
    references = {c: clouds_of(subsample(val_by[c], args.n, REFERENCE_SEED),
                               args.cloud_points)[0]
                  for c in range(n_classes)}

    # The floor, as the whole matrix: what a perfect class-conditional
    # generator scores, and — off the diagonal — how much margin the corpus
    # offers a perfect one. A model margin read against zero says only that the
    # diagonal won; read against this it says whether it won by enough.
    floor_draws = floor_reading(train_by, references, n_classes, args.n,
                                args.seeds, args.cloud_points)
    floor = {c: [d["diagonal"][c] for d in floor_draws] for c in range(n_classes)}

    if args.floor_only:
        if not out.exists():
            raise SystemExit(f"{out} does not exist; --floor-only merges into "
                             "a report, it does not start one")
        report = json.loads(out.read_text())
        if report.get("report_schema") != 2 or report.get("status", "complete") != "complete":
            raise SystemExit(f"{out} is not a complete schema-2 sampler report")
        if (report["reference_seed"] != REFERENCE_SEED or report["n"] != args.n
                or report.get("sampler") != sampler.as_dict()):
            raise SystemExit(f"{out} was written for a different reference, set size "
                             "or sampler; its floor is not this report's floor")
        report["floor"] = floor
        report["floor_matrix"] = [d["matrix"] for d in floor_draws]
        write_report(out, report)
        print_matrix([[st.mean([d["matrix"][a][r] for d in floor_draws])
                       for r in range(n_classes)] for a in range(n_classes)],
                     names, "floor matrix — real drawings in the samples' place")
        print(f"\n  -> {out}")
        return 0

    model.to(args.device)
    draws: list[dict] = []
    control_draws: list[dict] = []
    for seed in range(args.seeds):
        print(f"conditional draw {seed + 1}/{args.seeds}", flush=True)
        torch.manual_seed(seed)
        samples, caps = {}, {}
        for c in range(n_classes):
            samples[c], caps[c] = sample_class(
                model, codec, args.n, max_new, c, args.device, sampler
            )
        reading = class_quality(samples, references, args.cloud_points)
        for c in range(n_classes):
            reading["diagonal"][c].update(sample_guards(samples[c], caps[c]))
        draws.append(reading)
        try:
            require_balanced_geometry([reading], names, args.n, seed_offset=seed)
        except SystemExit as exc:
            write_report(out, make_report(
                record, control_record, names, args, sampler, max_new, draws,
                floor, floor_draws, control_draws, status="failed",
                failure={"kind": "empty_geometry", "side": "conditional",
                         "seed": seed, "message": str(exc)},
            ))
            raise

    if control_model is not None and control_record is not None:
        control_codec = CODECS[control_record["config"]["codec"]]
        control_model.to(args.device)
        for seed in range(args.seeds):
            print(f"control draw {seed + 1}/{args.seeds}", flush=True)
            torch.manual_seed(seed)
            sampled, cap_hit = sample_programs(
                control_model, control_codec, n=args.n, max_new=max_new,
                device=args.device, temperature=sampler.temperature,
                top_k=sampler.top_k, forbid_specials=True,
            )
            clouds, empty = clouds_of(sampled, args.cloud_points)
            guard = sample_guards(sampled, cap_hit)
            guard["empty"] = empty / max(1, len(sampled))
            if empty:
                message = (
                    f"control sampler produced {empty}/{len(sampled)} empty drawings "
                    f"at draw seed {seed}; geometry would be {len(clouds)} v {args.n}. "
                    "The failure is recorded above the quality axis; do not score an "
                    "unbalanced set."
                )
                control_draws.append({"sampling": guard})
                write_report(out, make_report(
                    record, control_record, names, args, sampler, max_new, draws,
                    floor, floor_draws, control_draws, status="failed",
                    failure={"kind": "empty_geometry", "side": "control",
                             "seed": seed, "message": message},
                ))
                raise SystemExit(message)
            control_draws.append({
                **{c: distribution_metrics(clouds, references[c])
                   for c in range(n_classes)},
                "sampling": guard,
            })

    # ---- the report ------------------------------------------------------
    print(f"\n## {record['name']}  ({codec.name}, "
          f"{record['model']['params']:,} params, {n_classes} classes)")
    print(f"{args.seeds} draws x {args.n} samples per class, sets {args.n} v "
          f"{args.n}, {args.cloud_points} cloud points")
    print(f"sampler: top_k={sampler.top_k}, temperature={sampler.temperature:g}, "
          "PAD/BOS forbidden\n")
    print(f"  {'class':<10}{'metric':<10}{'samples':>16}{'floor':>16}"
          f"{'control':>16}")
    for c in range(n_classes):
        for metric in ("coverage", "mmd", "nna"):
            got = mean_sd([d["diagonal"][c][metric] for d in draws])
            base = mean_sd([f[metric] for f in floor[c]])
            ctrl = (mean_sd([d[c][metric] for d in control_draws])
                    if control_draws else (float("nan"),) * 2)
            label = names[c] if metric == "coverage" else ""
            ctrl_txt = (f"{ctrl[0]:9.3f} ±{ctrl[1]:.3f}"
                        if not math.isnan(ctrl[0]) else f"{'--':>16}")
            print(f"  {label:<10}{metric:<10}{got[0]:9.3f} ±{got[1]:.3f}"
                  f"{base[0]:9.3f} ±{base[1]:.3f}{ctrl_txt}")

    print(f"\n  {'class':<10}{'validity':>12}{'empty':>12}{'truncated':>12}")
    for c in range(n_classes):
        diagonal = [d["diagonal"][c] for d in draws]
        print(f"  {names[c]:<10}"
              f"{st.mean(d['validity'] for d in diagonal):12.3f}"
              f"{st.mean(d['empty'] for d in diagonal):12.3f}"
              f"{st.mean(d['truncated'] for d in diagonal):12.3f}")
    if control_draws:
        print(f"  {'control':<10}"
              f"{st.mean(d['sampling']['validity'] for d in control_draws):12.3f}"
              f"{st.mean(d['sampling']['empty'] for d in control_draws):12.3f}"
              f"{st.mean(d['sampling']['truncated'] for d in control_draws):12.3f}")

    def averaged(source: list[dict]) -> list[list[float]]:
        return [[st.mean([d["matrix"][a][r] for d in source])
                 for r in range(n_classes)] for a in range(n_classes)]

    print_matrix(averaged(draws), names,
                 "mmd matrix, mean over draws (rows = asked, cols = real; "
                 "the diagonal should be the row minimum):")
    print_matrix(averaged(floor_draws), names,
                 "the same matrix with real drawings in the samples' place — "
                 "the margin a perfect generator gets:")

    # The reading the matrix exists for, made a column: a diagonal that is the
    # row minimum has won, and only this says whether it won by enough.
    model_margin, floor_margin = margins(draws, n_classes), margins(floor_draws, n_classes)
    print("\n  margin (min off-diagonal − diagonal), px — the model's against "
          "the corpus's own:")
    print(f"  {'class':<10}{'model':>18}{'perfect':>18}{'ratio':>10}{'won':>8}")
    for c in range(n_classes):
        got, base = mean_sd(model_margin[c]), mean_sd(floor_margin[c])
        wins = sum(m > 0 for m in model_margin[c])
        print(f"  {names[c]:<10}{got[0]:+9.2f} ±{got[1]:5.2f}{base[0]:+9.2f} "
              f"±{base[1]:5.2f}{got[0] / base[0]:9.2f}×{wins:5d}/{args.seeds}")

    print("\n  floor is real train-split drawings of the same class against the "
          "same reference,\n  at the same two set sizes — what a perfect "
          "class-conditional generator scores.\n  It is not a resolution floor. "
          "Quote the ordering across the two seeds, never a level.")

    write_report(out, make_report(
        record, control_record, names, args, sampler, max_new, draws, floor,
        floor_draws, control_draws,
    ))
    print(f"\n  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
