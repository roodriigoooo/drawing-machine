#!/usr/bin/env python3
"""Can *scale* join the transform group if the geometry primitive is smaller?

The decisive side measurement for `docs/direction.md` §7.3, pre-registered in
full in `docs/bezier.md` §§1-4 before it was run. In one paragraph:

A repeat is lossless or it is nothing, so it survives a non-integer map only if
**every** operand of its body rounds the same way. Model that as an independent
per-operand divergence probability `p` over an `n`-operand body. Fitting `p` to
the polyline arm's measured retention predicts the control-point arm's at its
own measured `n`, and the prediction on record is that a ×1.07 scale leaves ~86%
of the ceiling where a polyline leaves 23%. If it holds, "a cat twice the size"
is expressible in the ISA. If it fails, scale stays out of the group and Béziers
are a compression result.

**Three arms over one source corpus.** `native` is Tabler as authored -- the
corpus the 23% was measured on, carried as calibration rather than as a control,
because its source is already cubics and it is therefore the mixture rather than
either arm. `polyline` and `bezier` are the same geometry re-spelled through the
same recursion with one boolean changed, so they differ in the primitive and in
nothing else and can be read at matched *measured* fidelity.

**Every number is paired on the source drawing.** The scale refuses a drawing
that would leave the canvas (clamping is a deformation and a deformation of a
repeated body is not a repeat), and the arms refuse different ones, so the
oracles run over the drawings that survive in every arm.

    python3 scripts/bezier_oracle.py                    # Tabler, tol 1/2/4
    python3 scripts/bezier_oracle.py --corpus quickdraw # compression half only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from dm.data import quickdraw, tabler
from dm.data.fingerprint import digest
from dm.data.refit import jitter_program, respell_corpus, scale_program
from dm.eval.repeats import body_numbers, corpus_stats, symmetry_stats

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"

CATEGORIES = ("cat", "dog", "bus", "car", "tree")

#: The scale on record. Chosen there because it is small enough to be a
#: plausible "draw it slightly bigger" and irrational enough on this grid to
#: round every coordinate independently.
SCALE = 1.07

#: Below this the unscaled ceiling is too small for its retention to mean
#: anything -- branch 6 of the pre-registration.
CEILING_FLOOR = 0.005

#: Generators of the R2 low-discrepancy sequence, used to spread the sub-pixel
#: rounding phases. A grid would be worse than useless here: Tabler's
#: coordinates are mostly multiples of ten, so phases at k/10 resonate with the
#: lattice and a grid of them would sample the resonance rather than average it.
_R2 = (0.7548776662466927, 0.5698402909980532)


def phases(draws: int) -> list[tuple[float, float]]:
    """Sub-pixel offsets, one per draw. Deterministic, so a run is repeatable."""
    return [((i * _R2[0]) % 1.0, (i * _R2[1]) % 1.0) for i in range(draws)]


def build(args) -> list[bytes]:
    if args.corpus == "tabler":
        return tabler.split(args.icons, args.style)[0][: args.limit]
    return quickdraw.load(tuple(args.categories), "valid", limit=args.limit,
                          rdp_eps=args.rdp_eps)


def _ceiling(programs: list[bytes], max_body: int) -> dict:
    flat = corpus_stats(programs, max_body=max_body)
    orbit = symmetry_stats(programs, max_body=max_body)
    total = max(1, orbit["bytes"])
    return {
        "bytes": orbit["bytes"],
        "repeat_saved": flat["saved"],
        "repeat": flat["saved"] / total,
        "repeatx_saved": orbit["saved"],
        "repeatx": orbit["fraction"],
    }


def _retention(before: dict, after: dict, key: str) -> float:
    """Fraction of the oracle's saving that survives the operation."""
    saved = before[f"{key}_saved"]
    return after[f"{key}_saved"] / saved if saved else float("nan")


def arms(programs: list[bytes],
         tol: float) -> tuple[dict[str, dict[int, bytes]], dict[str, dict]]:
    """Every arm as `{source index: program}`, so the pairing has a key.

    `native` is the source itself: it *is* the geometry the other two are fitted
    to, so its reconstruction error is zero by definition rather than by
    measurement, and its curve fraction is a property of Tabler's authoring
    rather than of a fitter.
    """
    built: dict[str, dict[int, bytes]] = {"native": dict(enumerate(programs))}
    meta: dict[str, dict] = {"native": {"tol": 0.0, "primitive": "native",
                                        "error_mean": 0.0, "error_max": 0.0,
                                        "curve_fraction": float("nan"),
                                        "numbers_per_segment": float("nan")}}
    for name, allow in (("polyline", False), ("bezier", True)):
        respelled, stats = respell_corpus(programs, tol, allow, "none")
        built[name] = dict(zip(stats.pop("keep"), respelled))
        meta[name] = stats
    return built, meta


ARMS = ("native", "polyline", "bezier")


def measure(programs: list[bytes], tol: float, max_body: int, jitter: int,
            seed: int, draws: int, about: str, factor: float) -> list[dict]:
    """One tolerance: every arm, both operations, over `draws` paired draws.

    **A draw is a sub-pixel rounding phase**, shared by every arm so the
    comparison is paired, and the jitter's RNG is seeded by the draw index for
    the same reason. Retention is reported as a mean over draws with the
    draw-to-draw SD beside it, and the endpoint as a paired difference and a
    sign count -- the reading `docs/traps.md` requires of anything whose value
    depends on a draw.
    """
    built, meta = arms(programs, tol)
    offsets = phases(draws)

    scaled: list[dict[str, dict[int, bytes]]] = []
    jittered: list[dict[str, dict[int, bytes]]] = []
    for draw, phase in enumerate(offsets):
        after_scale: dict[str, dict[int, bytes]] = {}
        after_jitter: dict[str, dict[int, bytes]] = {}
        for name, by_index in built.items():
            rng = np.random.default_rng(seed + draw)
            after_scale[name] = {}
            after_jitter[name] = {}
            for index in sorted(by_index):
                program = by_index[index]
                grown = scale_program(program, factor, phase, about)
                if grown is not None:
                    after_scale[name][index] = grown
                # Two arms draw from generators at the same seed, so they see the
                # same *sequence*. They cannot see the same perturbation per
                # coordinate -- they have different coordinate counts -- so the
                # jitter row is a second reading of the mechanism and never a
                # paired difference at the coordinate level.
                shaken = jitter_program(program, jitter, rng)
                if shaken is not None:
                    after_jitter[name][index] = shaken
        scaled.append(after_scale)
        jittered.append(after_jitter)

    # **Two subsets, because there are two questions.** The compression half
    # (bytes at matched error) is about the primitive and must be read over
    # every drawing the arms could spell: restricting it to the drawings that
    # also survive a ×1.07 would bias it toward small drawings, and on QuickDraw
    # -- whose sketches fill the canvas to the margin -- it would empty the
    # comparison entirely. The retention half is about an *operation* and has to
    # be paired over the drawings that survive it in every arm.
    fitted = sorted(set.intersection(*(set(built[name]) for name in built)))
    order = sorted(set.intersection(*(
        set(built[name]) & set(step[name])
        for name in built for step in (*scaled, *jittered)
    )))

    rows = []
    for name in ARMS:
        base = [built[name][i] for i in order]
        spelled = [built[name][i] for i in fitted]
        unscaled = _ceiling(base, max_body)
        row = {
            "arm": name, "tol": tol, "draws": draws, "about": about,
            "factor": factor, "n_paired": len(order), "n_fitted": len(fitted),
            "n_source": len(programs),
            "paired_fraction": len(order) / max(1, len(programs)),
            "fitted_fraction": len(fitted) / max(1, len(programs)),
            "digest": digest(spelled),
            "bytes_per_drawing": sum(map(len, spelled)) / max(1, len(spelled)),
            "error_mean": meta[name]["error_mean"],
            "error_max": meta[name]["error_max"],
            "curve_fraction": meta[name]["curve_fraction"],
            "numbers_per_segment": meta[name]["numbers_per_segment"],
            "unscaled": unscaled,
            "body_repeat": body_numbers(base, max_body=max_body),
            "body_repeatx": body_numbers(base, max_body=max_body, symmetry=True),
        }
        for label, steps in (("scale", scaled), ("jitter", jittered)):
            per_draw = [_ceiling([step[name][i] for i in order], max_body)
                        for step in steps]
            row[f"{label}_draws"] = per_draw
            for key in ("repeat", "repeatx"):
                values = [_retention(unscaled, after, key) for after in per_draw]
                row[f"{label}_retention_{key}"] = float(np.mean(values))
                row[f"{label}_retention_{key}_sd"] = float(np.std(values, ddof=1)) \
                    if len(values) > 1 else 0.0
                row[f"{label}_retention_{key}_per_draw"] = values
                # Absolute bytes, because a retention is a *ratio* and the arms
                # do not share a denominator: an arm can retain a larger share
                # of a smaller ceiling and end up in the same place. The ISA
                # cares about the bytes, so both are reported.
                row[f"{label}_saved_{key}_per_drawing"] = float(np.mean(
                    [after[f"{key}_saved"] for after in per_draw])) / max(1, len(order))
            row[f"{label}_bytes_per_drawing"] = float(np.mean(
                [after["bytes"] for after in per_draw])) / max(1, len(order))
        rows.append(row)
    return rows


def divergence_model(control: dict, arm: dict, key: str = "repeat") -> dict:
    """Fit `p` on the control arm, then predict the other at its own `n`.

    The exponent is the **measured** operand count of the bodies the oracle
    folded, weighted by what each body saves. `docs/direction.md` §7.3 wrote it
    as an estimate; a model quoted at a guessed exponent cannot be falsified in
    the direction that matters, because any miss is blamed on the guess.
    """
    n_control = control[f"body_{key}"]["weighted_mean"]
    n_arm = arm[f"body_{key}"]["weighted_mean"]
    retention = control[f"scale_retention_{key}"]
    if not np.isfinite([n_control, n_arm, retention]).all() or n_control <= 0:
        return {"p": float("nan"), "predicted": float("nan"),
                "on_record_n4": float("nan"),
                "observed": arm.get(f"scale_retention_{key}", float("nan")),
                "n_control": n_control, "n_arm": n_arm}
    p = 1.0 - retention ** (1.0 / n_control)
    return {
        "p": float(p),
        "n_control": float(n_control),
        "n_arm": float(n_arm),
        "predicted": float((1.0 - p) ** n_arm),
        "observed": float(arm[f"scale_retention_{key}"]),
        "on_record_n4": float((1.0 - p) ** 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="tabler", choices=("tabler", "quickdraw"))
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--tol", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    ap.add_argument("--primary-tol", type=float, default=2.0)
    ap.add_argument("--max-body", type=int, default=64)
    ap.add_argument("--jitter", type=int, default=1)
    ap.add_argument("--factors", type=float, nargs="*", default=[0.5, 0.75, 0.93, 1.07, 1.25, 1.5, 2.0],
                    help="scale ladder, run at the primary tolerance")
    ap.add_argument("--about", default="bbox", choices=("bbox", "origin"),
                    help="scale centre. `origin` puts every drawing at one "
                         "rounding phase, which is what a corpus-wide "
                         "augmentation policy does")
    ap.add_argument("--draws", type=int, default=8,
                    help="sub-pixel rounding phases; a retention read at one is "
                         "one draw, not a number")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--icons", default="data/tabler/icons")
    ap.add_argument("--style", default="outline")
    ap.add_argument("--categories", nargs="+", default=list(CATEGORIES))
    ap.add_argument("--rdp-eps", type=float, default=4.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    programs = build(args)
    print(f"{args.corpus}: {len(programs)} source programs, "
          f"{sum(map(len, programs)) / len(programs):.0f} B each, "
          f"scale x{SCALE}, jitter +-{args.jitter}\n")

    rows: list[dict] = []
    for tol in args.tol:
        rows += measure(programs, tol, args.max_body, args.jitter, args.seed,
                        args.draws, args.about, SCALE)
    # The ladder. A scale that is exact on the integer grid needs no divergence
    # model at all -- `round(k*x) = k*x` -- so the question "can scale join the
    # transform group" has an answer per factor rather than one answer, and the
    # curve is what says where the boundary is. Run at the primary tolerance
    # only: it is a property of the operation, not of the fitter.
    ladder: list[dict] = []
    for factor in args.factors:
        if factor == SCALE:
            continue
        ladder += measure(programs, args.primary_tol, args.max_body, args.jitter,
                          args.seed, args.draws, args.about, factor)

    print(f"{'tol':>5}  {'arm':<9}{'B/draw':>8}{'err':>6}{'curve':>7}"
          f"{'REPEAT':>8}{'n':>5}{'scale retention':>19}"
          f"{'saved B/draw, before -> after':>32}")
    for r in rows:
        before = r["unscaled"]["repeat_saved"] / max(1, r["n_paired"])
        print(f"{r['tol']:>5.1f}  {r['arm']:<9}{r['bytes_per_drawing']:>8.1f}"
              f"{r['error_mean']:>6.2f}{r['curve_fraction']:>7.2f}"
              f"{r['unscaled']['repeat']:>7.2%}"
              f"{r['body_repeat']['weighted_mean']:>5.0f}"
              f"{r['scale_retention_repeat']:>12.1%}"
              f" ±{r['scale_retention_repeat_sd']:>5.1%}"
              f"{before:>24.2f} ->{r['scale_saved_repeat_per_drawing']:>5.2f}")

    primary = {r["arm"]: r for r in rows if r["tol"] == args.primary_tol}
    verdict: dict = {}
    if len(primary) == 3:
        control, test = primary["polyline"], primary["bezier"]
        model = divergence_model(control, test)
        paired = [b - a for a, b in zip(control["scale_retention_repeat_per_draw"],
                                        test["scale_retention_repeat_per_draw"])]
        verdict = {
            "primary_tol": args.primary_tol,
            "draws": args.draws,
            "retention_polyline": control["scale_retention_repeat"],
            "retention_polyline_sd": control["scale_retention_repeat_sd"],
            "retention_bezier": test["scale_retention_repeat"],
            "retention_bezier_sd": test["scale_retention_repeat_sd"],
            "retention_native": primary["native"]["scale_retention_repeat"],
            "retention_native_sd": primary["native"]["scale_retention_repeat_sd"],
            "delta": float(np.mean(paired)),
            "delta_sd": float(np.std(paired, ddof=1)) if len(paired) > 1 else 0.0,
            "draws_favouring_bezier": int(sum(d > 0 for d in paired)),
            "model": model,
            "bytes_polyline": control["bytes_per_drawing"],
            "bytes_bezier": test["bytes_per_drawing"],
            "error_polyline": control["error_mean"],
            "error_bezier": test["error_mean"],
            "paired_fraction": control["paired_fraction"],
            "ceiling_readable": min(
                control["unscaled"]["repeat"], test["unscaled"]["repeat"]
            ) >= CEILING_FLOOR,
        }
        print(f"\n## Primary endpoint at tol {args.primary_tol} px, "
              f"{args.draws} rounding phases (pre-registered, `docs/bezier.md` §4)\n")
        print(f"  native calibration      {verdict['retention_native']:>7.1%}"
              f" ±{verdict['retention_native_sd']:.1%}   (23% on record, one phase)")
        print(f"  polyline retention      {verdict['retention_polyline']:>7.1%}"
              f" ±{verdict['retention_polyline_sd']:.1%}"
              f"   at n = {model['n_control']:.0f} operands/body")
        print(f"  bezier retention        {verdict['retention_bezier']:>7.1%}"
              f" ±{verdict['retention_bezier_sd']:.1%}"
              f"   at n = {model['n_arm']:.0f}")
        print(f"  endpoint (bezier-poly)  {verdict['delta']:>+7.1%}"
              f" ±{verdict['delta_sd']:.1%}"
              f"   in {verdict['draws_favouring_bezier']}/{args.draws} paired draws")
        print(f"\n  divergence model: p = {model['p']:.4f} fitted on the control;"
              f" predicts {model['predicted']:.1%} at n = {model['n_arm']:.0f}"
              f"\n  (and {model['on_record_n4']:.1%} at the n = 4 on record)."
              f" Observed {model['observed']:.1%},"
              f" miss {abs(model['observed'] - model['predicted']):.1%}.")
        print(f"\n  bytes/drawing {verdict['bytes_polyline']:.1f} -> "
              f"{verdict['bytes_bezier']:.1f} "
              f"({1 - verdict['bytes_bezier'] / verdict['bytes_polyline']:+.1%}) "
              f"at mean error {verdict['error_polyline']:.2f} -> "
              f"{verdict['error_bezier']:.2f} px")
        # The ratio and the bytes are two different claims and only one of them
        # is about the ISA. An arm can retain a larger share of a smaller
        # ceiling and arrive at the same absolute saving, in which case the
        # endpoint is real and buys nothing.
        verdict["foldable_after_polyline"] = control["scale_saved_repeat_per_drawing"]
        verdict["foldable_after_bezier"] = test["scale_saved_repeat_per_drawing"]
        print(f"  foldable bytes/drawing after the scale: polyline "
              f"{verdict['foldable_after_polyline']:.2f}, bezier "
              f"{verdict['foldable_after_bezier']:.2f} "
              f"(of {verdict['bytes_polyline']:.0f} and "
              f"{verdict['bytes_bezier']:.0f} total)")
        if not verdict["ceiling_readable"]:
            print(f"\n  WARNING: an unscaled ceiling is below {CEILING_FLOOR:.1%} of "
                  "bytes, so its\n  retention is a ratio of two small numbers "
                  "(branch 6). Diagnostic only.")
        if verdict["paired_fraction"] < 0.8:
            print(f"\n  WARNING: only {verdict['paired_fraction']:.0%} of the corpus "
                  "survives every arm and\n  both operations, so the paired subset "
                  "is not the corpus (branch 7).")

    for r in rows:
        if r["error_max"] > r["tol"] + 1e-9:
            print(f"\n  BROKEN: {r['arm']} at tol {r['tol']} has max error "
                  f"{r['error_max']:.3f}. The fitter splits until this cannot "
                  "happen (branch 5); nothing in this table is readable.")

    if ladder:
        print(f"\n## The scale ladder at tol {args.primary_tol} px "
              f"-- retention of the REPEAT ceiling by factor\n")
        print(f"{'factor':>8}" + "".join(f"{a:>18}" for a in ARMS)
              + f"{'drawings':>10}")
        by_factor: dict[float, dict[str, dict]] = {}
        for r in (*[x for x in rows if x['tol'] == args.primary_tol], *ladder):
            by_factor.setdefault(r["factor"], {})[r["arm"]] = r
        for factor in sorted(by_factor):
            cells = by_factor[factor]
            line = f"{factor:>8.2f}"
            survivors = 0
            for arm in ARMS:
                r = cells.get(arm)
                if r is None:
                    line += "".rjust(18)
                    continue
                survivors = r["n_paired"]
                line += ("        --        " if not r["n_paired"] else
                         f"{r['scale_retention_repeat']:>11.1%}"
                         f" ±{r['scale_retention_repeat_sd']:>4.1%}")
            print(line + f"{survivors:>7}/{len(programs)}")
        print("\n  `drawings` is how many survive the factor in every arm. A scale"
              "\n  is refused, never clamped, when it leaves the canvas -- so the"
              "\n  ladder runs out of corpus before it runs out of exactness, and"
              "\n  the two ends of it fail for unrelated reasons. An integer factor"
              "\n  is exact by arithmetic (`round(k*x) = k*x`) and needs no model;"
              "\n  what it needs is canvas headroom the corpus does not have.")
    rows += ladder

    out = args.out or RUNS / f"bezier_oracle_{args.corpus}_n{len(programs)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()},
                               "scale": SCALE, "verdict": verdict,
                               "rows": rows}, indent=2))
    print(f"\n  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
