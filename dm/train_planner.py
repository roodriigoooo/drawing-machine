"""Training loop for claim 3's stroke planner.

Deliberately a sibling of `dm/train.py` rather than a branch inside it. The two
share a corpus (`build_data`), a device policy, a schedule and a record format,
and they differ in the only place that matters: the AR run reports an exact NLL
and this one reports an **upper bound** (`dm/models/planner.py`). Folding the
second into the first would put two different kinds of number in one column
named `bits_per_drawing`, which is the fault this project has now found four
times at four different levels.

The record is comparable with an AR record in the ways that count:

- `bits_per_drawing` is in the same unit and over the same val split, and the
  split comes from the same `TrainConfig`, so `data_seed` still pins it and the
  comparison is **paired per program** (`val_bits`).
- `params` counts *both* levels. A hierarchical model quoted at one level is the
  StrokeNUWA objection this project raises against others (`PLAN.md` 4).
- `drift` and `tail` are derived from `history` by `scripts/sweep.py` exactly as
  they are for an AR run, so the convergence guards apply unchanged.

    python3 -m dm.train_planner --data quickdraw --codec byte --steps 12000 \
        --tag quickdraw_planner_byte_square_s0
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass, field

import torch
import torch.nn.functional as F

from .data.fingerprint import fingerprint
from .eval.metrics import corpus_stats, length_stats
from .isa.codec import CODECS
from .isa.spec import Tier
from .isa.strokes import stroke_count
from .models.planner import (
    COMPOSITION,
    PlannerConfig,
    StrokePlanner,
    bits_per_drawing,
    generate,
    stroke_batch,
    summary_grid,
)
from .train import TrainConfig, build_data, checkpoint, lr_at, pick_device, release

#: Bumped when a change makes new planner records incomparable with old ones.
#: 1 = the first build: absorbing-state composition diffusion over six-byte
#:     stroke summaries, an AR stroke decoder conditioned on one summary, and a
#:     bits/drawing that is the sum of a diffusion NELBO and an exact
#:     conditional NLL. Not comparable with `dm.train.SCHEMA` by construction --
#:     these are different models -- which is why the counter is separate.
PLANNER_SCHEMA = 2

#: How the budget is split between the two levels. Every entry totals 825k on
#: the byte alphabet -- the AR `square` arm's count to within 0.1% -- so a
#: planner row and an AR row are a like-for-like comparison and claim 3's "at
#: equal parameters" is a fact about the table rather than an intention.
#:
#: They vary with the codec's vocabulary exactly as the AR arms do, and for the
#: same reason: 801k on bit (vocab 4), 925k on `token_typed` (vocab 1293). That
#: spread *is* the IconShop objection, and it is the axis claim 1 measured.
#:
#: `balanced` is the default. The other two exist because "which level is
#: starved?" is a sweep and not a guess, and because the answer is claim 3's
#: most informative negative result if the planner loses: a planner that
#: improves monotonically towards `layout` is short of composition capacity,
#: and one that does not is failing at the factorisation itself.
PLANNER_SHAPES: dict[str, dict] = {
    "balanced": dict(comp_d_model=80, comp_layers=5, stroke_d_model=88, stroke_layers=4),
    "layout": dict(comp_d_model=104, comp_layers=4, stroke_d_model=80, stroke_layers=3),
    "shape": dict(comp_d_model=48, comp_layers=6, stroke_d_model=112, stroke_layers=4),
}


@dataclass
class PlannerTrainConfig:
    """The AR config's fields that still apply, plus the planner's own.

    Not a subclass: `TrainConfig` carries `shape`, which names an entry in
    `SHAPES` and means something different here, and inheriting a field whose
    meaning changed is how a record starts lying about what produced it.
    """

    codec: str = "byte"
    shape: str = "balanced"
    data: str = "synthetic"
    categories: tuple[str, ...] = ("cat",)
    tier: Tier = Tier.L1
    n_train: int = 100_000
    n_val: int = 1_000
    max_strokes: int = 32
    max_stroke_len: int = 256
    batch_size: int = 64
    steps: int = 4_000
    lr: float = 3e-3
    warmup: int = 200
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    seed: int = 0
    data_seed: int = 0
    eval_every: int = 500
    eval_noise_samples: int = 32
    #: `"diffusion"` (claim 3's factorisation) or `"ar"` (the control it has to
    #: beat). `PlannerConfig.comp_objective` carries the argument.
    comp_objective: str = "diffusion"
    gen_samples: int = 128
    gen_steps: int = 16
    #: The composition sampler's unmasking order, recorded so a generation
    #: column can name what produced it.
    #:
    #: `"random"` is the reverse process of the absorbing kernel `nelbo_terms`
    #: trains against and the only defensible default; `"confidence"` is
    #: MaskGIT's heuristic, kept reachable because three records used it and a
    #: retraction that cannot be reproduced is an assertion.
    #:
    #: It is a *config* field rather than a schema bump because the likelihood
    #: half does not depend on it -- the sampler has no gradient and no path
    #: into the bound -- so a sampler change makes the generation columns
    #: incomparable and leaves `bits_per_drawing` comparable, and those are
    #: different statements. Three of this project's six faults were in this
    #: sampler and none of the three records could say which one it ran.
    gen_order: str = "random"
    device: str = "mps"
    tag: str = ""
    extra: dict = field(default_factory=dict)

    def planner(self, vocab_size: int) -> PlannerConfig:
        return PlannerConfig(
            vocab_size=vocab_size, max_strokes=self.max_strokes,
            max_stroke_len=self.max_stroke_len,
            eval_noise_samples=self.eval_noise_samples,
            comp_objective=self.comp_objective,
            **PLANNER_SHAPES[self.shape],
        )

    def corpus(self) -> TrainConfig:
        """The AR config that names the same corpus.

        One code path builds every corpus in this project, and a planner run
        that rebuilt its own split would be scored on different programs from
        the baseline it is meant to beat -- two val sets differ by ~3.5
        bits/drawing (`PLAN.md` 10).
        """
        return TrainConfig(
            codec=self.codec, data=self.data, categories=self.categories,
            tier=self.tier, n_train=self.n_train, n_val=self.n_val,
            seed=self.seed, data_seed=self.data_seed, extra=self.extra,
        )


def stroke_stats(programs: list[bytes], max_strokes: int) -> dict:
    """What the composition level is being asked to model.

    `over_cap` is the fraction of programs with more strokes than there are
    slots. Those are *merged*, not dropped (`dm.isa.strokes.split`), so every
    byte stays in the likelihood -- but the last slot's summary stops describing
    one stroke, so a corpus with a large number here is being modelled at the
    wrong `max_strokes` and the column has to say so.
    """
    counts = sorted(stroke_count(p) for p in programs)
    n = max(1, len(counts))
    return {
        "mean": sum(counts) / n,
        "p50": float(counts[n // 2]),
        "p99": float(counts[int(0.99 * (n - 1))]),
        "max": float(counts[-1]),
        "over_cap": sum(c > max_strokes for c in counts) / n,
    }


def train(cfg: PlannerTrainConfig, verbose: bool = True) -> dict:
    torch.manual_seed(cfg.seed)
    device = pick_device(cfg.device)
    codec = CODECS[cfg.codec]

    train_programs, val_programs = build_data(cfg.corpus())
    planner_cfg = cfg.planner(codec.vocab_size)
    planner = StrokePlanner(planner_cfg).to(device)
    opt = torch.optim.AdamW(
        planner.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95)
    )

    strokes = stroke_stats(val_programs, cfg.max_strokes)
    name = cfg.tag or f"{cfg.data}_planner_{cfg.codec}_{cfg.shape}_s{cfg.seed}"
    counts = planner.level_params()
    if verbose:
        print(
            f"[{name}] params={counts['total']:,} "
            f"(composition {counts['composition']:,} + stroke {counts['stroke']:,}) "
            f"vocab={codec.vocab_size} strokes(mean={strokes['mean']:.1f} "
            f"p99={strokes['p99']:.0f} over_cap={strokes['over_cap']:.3f}) "
            f"steps={cfg.steps} device={device}",
            flush=True,
        )

    history: list[dict] = []
    best: dict = {}
    best_val_bits: list[float] = []
    val_bits: list[float] = []
    # Hoisted out of the record, which is now written at every eval: this
    # digests every program in both splits and the corpus does not change while
    # a run is training.
    corpus = fingerprint(train_programs, val_programs)

    def snapshot(complete: bool) -> dict:
        """The record as of the last eval. Whole at every eval, not only at exit.

        `dm.train.checkpoint` says why it is written that often, and this run is
        the reason: its predecessor at this exact config was killed at step
        11,500 of 12,000 and left nothing behind.
        """
        return {
            "name": name,
            # Read by `scripts/sweep.py:summarise`, which globs `{data}_*.json`
            # and would otherwise report every planner run as an AR run written
            # before the current schema. "This is a different model" and "this
            # is an out-of-date record" are opposite statements and only one of
            # them means delete it.
            "kind": "planner",
            "schema": PLANNER_SCHEMA,
            # Whether `train` returned, and how far the loop actually got. A
            # partial record measures a shorter run honestly; what it must not
            # do is enter a table as a rung of the budget it asked for, and
            # `config["steps"]` alone cannot tell the two apart.
            "complete": complete,
            "steps_done": history[-1]["step"] if history else 0,
            "config": {**asdict(cfg), "tier": int(cfg.tier)},
            # Identical field to the AR record's, and the reason it exists: run
            # 4 was a planner scored on `cat dog bus car tree` at the default
            # `rdp_eps` against an AR baseline on `cat bus flower sailboat
            # bicycle` at 4.0, and nothing in either record could say so.
            "corpus": corpus,
            "model": {"params": counts["total"], **counts, **asdict(planner_cfg)},
            "val_strokes": strokes,
            "val_bytes": [len(p) for p in val_programs],
            "val_bits": val_bits,
            "best_val_bits": best_val_bits,
            "history": history,
            "final": history[-1] if history else {},
            "best": best,
        }

    order = torch.randperm(len(train_programs)).tolist()
    cursor = 0
    started = time.time()
    planner.train()
    for step in range(1, cfg.steps + 1):
        if cursor + cfg.batch_size > len(order):
            order, cursor = torch.randperm(len(train_programs)).tolist(), 0
        chunk = [train_programs[i] for i in order[cursor : cursor + cfg.batch_size]]
        cursor += cfg.batch_size

        for group in opt.param_groups:
            group["lr"] = lr_at(step - 1, cfg)  # type: ignore[arg-type]

        grid = torch.tensor(
            [summary_grid(p, cfg.max_strokes) for p in chunk],
            dtype=torch.long, device=device,
        )
        batch = stroke_batch(chunk, codec, planner_cfg)
        inputs, targets = batch.inputs.to(device), batch.targets.to(device)

        # One optimiser over disjoint parameters, so summing the losses is two
        # independent trainings that happen to share a step counter. They are
        # each normalised to a per-item mean first: the NELBO is per drawing and
        # the decoder's loss is per token, and leaving them on their native
        # scales would make the balance between the levels an accident of how
        # many strokes a corpus happens to have.
        opt.zero_grad(set_to_none=True)
        # One draw: the gradient only needs an unbiased estimate, and for the AR
        # objective there is nothing to draw. `cost_terms` is the one method both
        # levels answer, so the objective is a substitution here rather than a
        # branch -- the schedule, the batch and the record stay identical, which
        # is what makes the two rows a controlled comparison.
        comp_loss = planner.composition.cost_terms(grid, 1).mean() / planner_cfg.slots
        stroke_loss = F.cross_entropy(
            planner.decoder(inputs).reshape(-1, codec.vocab_size),
            targets.reshape(-1), ignore_index=0,
        )
        (comp_loss + stroke_loss).backward()
        torch.nn.utils.clip_grad_norm_(planner.parameters(), cfg.grad_clip)
        opt.step()

        if step % cfg.eval_every and step != cfg.steps:
            continue

        record = {
            "step": step,
            "train_loss": float(comp_loss.detach() + stroke_loss.detach()),
            "composition_loss": float(comp_loss.detach()),
            "stroke_loss": float(stroke_loss.detach()),
        }
        # `seed` fixed across evals so the bound's Monte-Carlo draw is the same
        # every time: without it the curve wanders by the estimator's variance
        # and `tail` reads that as training progress.
        scored = bits_per_drawing(planner, val_programs, codec, device=device,
                                  batch_size=cfg.batch_size, seed=cfg.data_seed)
        val_bits = scored.pop("val_bits")
        record |= scored
        sampled = generate(planner, codec, n=cfg.gen_samples, steps=cfg.gen_steps,
                           device=device, top_k=40, order=cfg.gen_order)
        stats = corpus_stats(sampled.programs)
        # `length_emd` is the column `dm/models/planner.py` states claim 3's
        # falsifiable prediction in -- termination is structural here and
        # sampled in the AR baseline, so this is where the scale argument is
        # decided. It was absent from every planner record until now, while
        # `scripts/sweep.py` read it and printed nan.
        stats |= length_stats(sampled.programs, val_programs, sampled.cap_hit)
        record |= {f"gen_{k}": v for k, v in stats.items()}
        record["gen_strokes"] = (
            sum(map(stroke_count, sampled.programs)) / max(1, len(sampled.programs))
        )
        # What the composition level *planned*, beside what came back. The AR
        # arm has no equivalent: its length is sampled one symbol at a time,
        # and the whole of claim 3's termination argument is that this one is
        # decided at the coarse scale before a single stroke byte exists.
        record["gen_planned_strokes"] = (
            sum(sampled.planned) / max(1, len(sampled.planned))
        )
        record["elapsed_s"] = round(time.time() - started, 1)
        if not best or record["bits_per_drawing"] < best["bits_per_drawing"]:
            best, best_val_bits = record, val_bits
        history.append(record)
        checkpoint(name, snapshot(complete=False),
                   {"cfg": asdict(planner_cfg), "state": planner.state_dict()})
        planner.train()
        if verbose:
            print(
                f"  step {step:>6}  loss {record['train_loss']:.4f}  "
                f"bits/drawing {record['bits_per_drawing']:8.1f} "
                f"(layout {record['composition_bits']:.1f} + "
                f"shape {record['stroke_bits']:.1f})  "
                f"gen_valid {record['gen_validity']:.3f}  "
                f"strokes {record['gen_planned_strokes']:.1f} planned/"
                f"{record['gen_strokes']:.1f} drawn (val {strokes['mean']:.1f})  "
                f"len EMD {record['gen_length_emd']:.1f}  "
                f"{record['elapsed_s']:.0f}s",
                flush=True,
            )

    # Rewrites the last eval's record with `complete` set, so the flag means
    # "the trainer returned" rather than "the loop reached its last step".
    result = snapshot(complete=True)
    checkpoint(name, result,
               {"cfg": asdict(planner_cfg), "state": planner.state_dict()})

    del planner, opt
    release(device)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--codec", default="byte", choices=sorted(CODECS))
    ap.add_argument("--shape", default="balanced", choices=sorted(PLANNER_SHAPES))
    ap.add_argument("--comp-objective", default="diffusion", choices=sorted(COMPOSITION),
                    help="'diffusion' is claim 3's factorisation; 'ar' is the "
                         "equal-parameter control it has to beat")
    ap.add_argument("--data", default="synthetic",
                    choices=["synthetic", "quickdraw", "tabler"])
    ap.add_argument("--categories", nargs="+", default=["cat"])
    ap.add_argument("--n-train", type=int, default=PlannerTrainConfig.n_train)
    ap.add_argument("--n-val", type=int, default=PlannerTrainConfig.n_val)
    ap.add_argument("--max-strokes", type=int, default=PlannerTrainConfig.max_strokes)
    ap.add_argument("--max-stroke-len", type=int,
                    default=PlannerTrainConfig.max_stroke_len)
    ap.add_argument("--steps", type=int, default=4_000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--gen-samples", type=int, default=128)
    ap.add_argument("--gen-order", default="random", choices=("random", "confidence"),
                    help="composition unmasking order; 'random' is the reverse "
                         "process of the kernel the objective trains against")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--tag", default="")
    ap.add_argument("--rdp-eps", type=float, default=None)
    ap.add_argument("--flatten", action="store_true")
    args = ap.parse_args()
    extra = {
        k: v for k, v in (("rdp_eps", args.rdp_eps), ("flatten", args.flatten or None))
        if v is not None
    }
    train(
        PlannerTrainConfig(
            codec=args.codec, shape=args.shape, data=args.data,
            comp_objective=args.comp_objective,
            categories=tuple(args.categories), n_train=args.n_train,
            n_val=args.n_val, max_strokes=args.max_strokes,
            max_stroke_len=args.max_stroke_len, steps=args.steps,
            batch_size=args.batch_size, lr=args.lr, seed=args.seed,
            data_seed=args.data_seed, eval_every=args.eval_every,
            gen_samples=args.gen_samples, gen_order=args.gen_order,
            device=args.device, tag=args.tag, extra=extra,
        )
    )


if __name__ == "__main__":
    main()
