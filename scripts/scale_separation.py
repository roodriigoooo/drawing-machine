#!/usr/bin/env python3
"""Scale separation: does local structure transfer up the ladder and global not?

The experiment that can falsify the hierarchical design (claim 3) before any of
it is built, using data that already exists. `dm/data/synthetic.py:LADDER` varies
nesting depth independently of stroke count, so:

    train on `simple`, complete held-out prefixes from every rung, and measure
    local (stroke shape) and global (layout) fidelity separately.

**Prediction: the local gap stays small as the rung gets harder and the global
gap opens up.** Equal degradation falsifies the premise that the two levels are
separable, and that is worth knowing before a planner is written rather than
after. See `docs/scale-separation.md`.

An AR model reconstructs nothing, so the paired traces come from
prefix-conditioned completion: prompt with the first `--prefix` of a program's
*bytes*, let the model finish, and compare the geometry against the full truth.
Prompts are cut in bytecode bytes and sliced out of the full encoding, so they
are exactly the symbols the model would have seen and the halt monitor's parse
state starts correct.

Raw distances are not comparable across rungs -- `elaborate` has ~3x the strokes
of `simple`, which moves both metrics for a perfect model -- so every number is
reported as a skill score against a chance reference on the same rung: the same
metric between the truth and an unrelated program from that rung.

    python3 scripts/scale_separation.py --steps 12000
    python3 scripts/scale_separation.py --checkpoint runs/scale_simple_byte.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dm.data import synthetic
from dm.eval.scale import scale_fidelity
from dm.isa.codec import CODECS, Codec
from dm.models.transformer import Config, DrawingLM
from dm.train import RUNS, TrainConfig, train
from dm.vm.interp import VM


def complete(
    model: DrawingLM,
    codec: Codec,
    programs: list[bytes],
    prefix: float,
    device: str,
    cap: int,
    batch_size: int = 32,
) -> tuple[list[bytes], list[bytes]]:
    """Prompt each program with its own first `prefix` of bytes and finish it.

    Returns the completions *and* the prompts, because the prompt is part of
    every completion and therefore part of every score: a model that emits
    nothing at all still gets credit for the geometry it was handed. Scoring the
    prompt alone gives the floor that credit sits on, and a result is only about
    the model to the extent it clears it.

    Batched by prompt width, because `generate` takes a rectangular prompt and
    padding one would feed the model symbols the program does not contain.
    """
    out: list[bytes] = [b""] * len(programs)
    prompts: list[bytes] = [b""] * len(programs)
    buckets: dict[int, list[int]] = defaultdict(list)
    for i, program in enumerate(programs):
        buckets[max(1, int(len(program) * prefix))].append(i)

    for given_bytes, ids in buckets.items():
        for start in range(0, len(ids), batch_size):
            rows = ids[start : start + batch_size]
            # Sliced out of the *full* encoding: `encode(program[:b])` is not the
            # same thing when the cut lands mid-instruction, because the token
            # codec types operands by walking the stream and a truncated walk
            # types them differently.
            prompt = torch.tensor(
                [codec.encode(programs[i])[: given_bytes * codec.stride] for i in rows],
                dtype=torch.long,
            )
            ids_out = model.generate(
                len(rows), cap * codec.stride, device=device, top_k=40,
                monitor=codec.halt_monitor(len(rows)), prompt=prompt,
            )
            for row, i in enumerate(rows):
                out[i] = codec.decode(ids_out[row].tolist())
                prompts[i] = programs[i][:given_bytes]
    return out, prompts


def coverage(scores: list[dict]) -> float:
    """Matched strokes over the larger of the two stroke counts.

    `scale_fidelity` scores matched pairs only and reports the leftovers
    separately, on the principle that no single number preserves the
    distinction. That is right for the metric and wrong for a headline: a
    prediction holding one stroke that happens to land well scores a better
    *matched* shape distance than one that attempts all forty, so the
    emptiest output wins. The first run of this experiment showed exactly
    that -- local skill came in below the prompt-only floor at every rung,
    because the floor emits fewest strokes of all.

    Dividing by `max(n_truth, n_prediction)` charges both failures: strokes
    the model omitted and strokes it invented.
    """
    ratios = [
        s["matched"] / m
        for s in scores
        if (m := max(s["n_truth"], s["n_prediction"]))
    ]
    return float(np.mean(ratios)) if ratios else 0.0


def evaluate(
    model: DrawingLM, codec: Codec, programs: list[bytes], prefix: float,
    device: str, cap: int, rng: np.random.Generator,
) -> dict:
    """Skill on both scales, plus the chance reference each is scored against."""
    vm = VM()
    truths = [vm.run(p) for p in programs]
    completions, prompts = complete(model, codec, programs, prefix, device, cap)
    predictions = [vm.run(p) for p in completions]
    # Chance is a *different* drawing from the same rung: it has the rung's
    # stroke count and the rung's spatial statistics, so it isolates what the
    # model knows from what the rung makes easy.
    partners = [truths[j] for j in rng.permutation(len(truths))]

    model_scores = [scale_fidelity(t, p) for t, p in zip(truths, predictions)]
    chance_scores = [scale_fidelity(t, p) for t, p in zip(truths, partners)]
    # The floor: what the prompt alone scores. Anything at or below this is the
    # prompt, not the model.
    floor_scores = [scale_fidelity(t, vm.run(p)) for t, p in zip(truths, prompts)]

    def mean(scores: list[dict], key: str) -> float:
        values = [s[key] for s in scores if np.isfinite(s[key])]
        return float(np.mean(values)) if values else float("nan")

    report = {"n": len(programs)}
    covers = {"model": coverage(model_scores), "floor": coverage(floor_scores),
              "chance": coverage(chance_scores)}
    for key in ("global", "local"):
        model_d, chance_d = mean(model_scores, key), mean(chance_scores, key)
        floor_d = mean(floor_scores, key)
        report[f"{key}_model"] = model_d
        report[f"{key}_chance"] = chance_d
        # Skill on the strokes it did match: 1 at truth, 0 at chance.
        matched_skill = 1.0 - model_d / chance_d if chance_d else float("nan")
        floor_matched = 1.0 - floor_d / chance_d if chance_d else float("nan")
        report[f"{key}_matched_skill"] = matched_skill
        report[f"{key}_floor_matched_skill"] = floor_matched
        # ... scaled by how much of the drawing it accounted for. A model that
        # matches half the strokes perfectly scores 0.5; one that emits nothing
        # scores 0, which is what the floor should be worth.
        report[f"{key}_skill"] = matched_skill * covers["model"] / max(1e-9, covers["chance"])
        report[f"{key}_floor_skill"] = floor_matched * covers["floor"] / max(1e-9, covers["chance"])
    report["coverage"] = covers["model"]
    report["coverage_floor"] = covers["floor"]
    report["coverage_chance"] = covers["chance"]
    report["strokes_truth"] = mean(model_scores, "n_truth")
    report["strokes_pred"] = mean(model_scores, "n_prediction")
    report["unmatched_truth"] = mean(model_scores, "unmatched_truth")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-rung", default="simple", choices=sorted(synthetic.LADDER))
    ap.add_argument("--rungs", nargs="+", default=["simple", "busy", "nested", "elaborate"])
    ap.add_argument("--codec", default="byte", choices=sorted(CODECS))
    ap.add_argument("--shape", default="square")
    ap.add_argument("--steps", type=int, default=12_000)
    ap.add_argument("--n-train", type=int, default=100_000)
    ap.add_argument("--n-eval", type=int, default=128)
    ap.add_argument("--prefix", type=float, default=0.25, help="fraction of bytes prompted")
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    codec = CODECS[args.codec]
    tag = f"scale_{args.train_rung}_{args.codec}_{args.shape}_s{args.seed}"
    if args.checkpoint is None:
        # `extra` is forwarded to `synthetic.split`, which is what makes a rung a
        # rung; the run record then carries it, so the checkpoint says what it
        # was trained on.
        train(TrainConfig(
            codec=args.codec, shape=args.shape, steps=args.steps, token_budget=None,
            n_train=args.n_train, seed=args.seed, device=args.device, tag=tag,
            extra=dict(synthetic.LADDER[args.train_rung]),
        ))
        args.checkpoint = RUNS / f"{tag}.pt"

    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = DrawingLM(Config(**blob["cfg"]))
    model.load_state_dict(blob["state"])
    model = model.to(args.device)

    rows = []
    for rung in args.rungs:
        # Seeded by the rung's *position* in the ladder, never by `hash(rung)`:
        # string hashing is randomised per process, so that would hand a
        # different eval set to every invocation and make two runs of this
        # script incomparable -- the same fault as scoring each model seed on a
        # different val set, which cost this project a sweep (§6.1-6.2).
        rung_index = list(synthetic.LADDER).index(rung)
        programs = synthetic.dataset(
            args.n_eval, seed=9_000 + rung_index, **synthetic.LADDER[rung]
        )
        cap = int(2 * np.percentile([len(p) for p in programs], 99))
        report = evaluate(
            model, codec, programs, args.prefix, args.device, cap,
            np.random.default_rng(args.seed),
        )
        rows.append({"rung": rung, **report})
        print(f"[{rung}] {json.dumps({k: round(v, 3) for k, v in report.items()})}",
              flush=True)

    print(f"\n## Scale separation — trained on `{args.train_rung}`, "
          f"{args.prefix:.0%} of each program prompted\n")
    print("| rung | strokes truth/pred | coverage | global skill (floor) | "
          "local skill (floor) | local matched-only |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['rung']} | {r['strokes_truth']:.1f}/{r['strokes_pred']:.1f} | "
              f"{r['coverage']:.2f} ({r['coverage_floor']:.2f}) | "
              f"**{r['global_skill']:+.3f}** ({r['global_floor_skill']:+.3f}) | "
              f"**{r['local_skill']:+.3f}** ({r['local_floor_skill']:+.3f}) | "
              f"{r['local_matched_skill']:+.3f} ({r['local_floor_matched_skill']:+.3f}) |")
    print("\nSkill is `1 - d(model, truth) / d(chance, truth)` on the strokes the "
          "assignment matched, scaled by `coverage` = matched / max(strokes truth, "
          "strokes predicted). Without that scaling the emptiest output wins, "
          "because a single well-placed stroke has a better *matched* distance "
          "than an honest attempt at forty; the matched-only column is kept beside "
          "it so the two failures stay separable. 1.0 is perfect, 0.0 is no better "
          "than an unrelated drawing from the same rung. The bracketed number is "
          "the same score for the *prompt alone*, which every completion contains "
          "-- only the gap above it is the model's. **The prediction under test is "
          "that `local skill` holds up across rungs while `global skill` falls.** "
          "Equal decay falsifies the "
          "premise that the levels are separable, which is what "
          "`docs/scale-separation.md` exists to decide.")
    (RUNS / f"{tag}_scale.json").write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
