#!/usr/bin/env python3
"""What a class label bought, read on the axis where it can be seen.

`bits/drawing` **cannot** resolve class conditioning on this corpus, and that is
arithmetic rather than a limitation of the runs: a label carries at most
`H(class) = log2(5) = 2.32` bits, and the corpus's run-to-run floor is ~2.5. So
this reports four things instead, and two of them check each other:

1. **the ceiling** — `H(C)` on the actual val split, which bounds any gain;
2. **the gap** — conditional minus unconditional `bits/drawing`, paired per
   program, which *is* `I(X; C)` and so measures how much a drawing says about its
   own category;
3. **the free classifier** — `p(c | x) ∝ p(x | c)·p(c)` on the same model, whose
   `H(C | X)` must equal `H(C)` minus the gap, and whose accuracy must respect the
   Fano bound the gap implies;
4. **controllability** — sample under each class, classify the samples, and read
   the diagonal against chance. This is the axis's actual claim.

    # everything, against the matched unconditional arm
    python3 scripts/conditioning.py runs/quickdraw_cond24000eps4_byte_square_s0.pt \\
        --against runs/quickdraw_budget24000eps4_byte_square_s0.pt

    # the conditional half alone, no control
    python3 scripts/conditioning.py runs/quickdraw_cond24000eps4_byte_square_s0.pt

Five forward passes over the val split plus one sampling pass. No training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dm.data.dataset import ProgramDataset, loader
from dm.eval.conditioning import (
    class_log2_likelihood,
    class_posterior,
    conditional_class_bits,
    controllability,
    fano_error_bound,
    label_entropy,
)
from dm.eval.metrics import paired_delta, per_program_bits
from dm.eval.records import load
from dm.isa.codec import CODECS
from dm.train import RUNS, build_data, build_labels, config_from_record


def val_split(record: dict) -> tuple[list[bytes], list[int], int]:
    """The val programs, their classes, and the class count this run used."""
    cfg = config_from_record(record)
    _, programs = build_data(cfg)
    labels = build_labels(cfg)
    if labels is None:
        raise SystemExit(f"{record['name']} is on {cfg.data!r}, which has no classes")
    # `.get`, because every record written before conditioning existed has no
    # such field and is a legitimate unconditional arm rather than a broken one.
    return programs, labels[1], record["model"].get("n_classes", 0)


def control_bits(path: Path, programs: list[bytes]) -> tuple[dict, np.ndarray]:
    """The unconditional arm's per-program bits on the identical programs.

    Refused unless the control really is unconditional and really is on the same
    corpus: differencing a conditional arm against another conditional one measures
    nothing, and differencing across val splits measures the split. The corpus
    digest is what settles the second question (`dm/data/fingerprint.py`).
    """
    model, record = load(path)
    if record["model"].get("n_classes"):
        raise SystemExit(f"{record['name']} is itself conditional; the control must not be")
    codec = CODECS[record["config"]["codec"]]
    batches = loader(ProgramDataset(programs, codec, record["config"]["max_len"]),
                     32, shuffle=False)
    return record, per_program_bits(model, batches)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path, help="the conditional arm")
    ap.add_argument("--against", type=Path, default=None,
                    help="matched unconditional checkpoint, for the paired gap")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--samples", type=int, default=256,
                    help="samples for the controllability half, round-robin over classes")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    model, record = load(args.checkpoint)
    programs, labels, n_classes = val_split(record)
    if not n_classes:
        raise SystemExit(
            f"{record['name']} was trained without --conditional; there is no "
            "conditioning to read"
        )
    codec = CODECS[record["config"]["codec"]]
    max_len = record["config"]["max_len"]
    ceiling = label_entropy(labels, n_classes)

    print(f"\n## {record['name']}  ({codec.name}, {record['model']['params']:,} params, "
          f"{n_classes} classes)")
    print(f"corpus: {ceiling['n']} val programs, "
          f"H(class) = {ceiling['entropy_bits']:.4f} bits "
          f"(uniform {ceiling['uniform_bits']:.4f}), chance accuracy "
          f"{ceiling['chance_accuracy']:.3f}")
    print(f"ceiling: conditioning can buy at most {ceiling['entropy_bits']:.2f} "
          "bits/drawing, against this corpus's ~2.5-bit resolution floor "
          "-- unresolvable by construction")

    nll = class_log2_likelihood(model.to(args.device), programs, codec, n_classes,
                                device=args.device, max_len=max_len,
                                batch_size=args.batch_size)
    posterior = class_posterior(nll)["posterior"]
    read = conditional_class_bits(posterior, labels)
    # The conditional arm's own bits/drawing is the true-class column.
    true_bits = nll[np.arange(len(labels)), np.asarray(labels)]

    print(f"\nthe free classifier: accuracy {read['accuracy']:.4f} against chance "
          f"{ceiling['chance_accuracy']:.3f}, H(class | drawing) = "
          f"{read['class_bits']:.4f} ±{read['class_bits_stderr']:.4f} bits")
    print(f"  implied mutual information: {ceiling['entropy_bits'] - read['class_bits']:+.4f} "
          "bits/drawing, which is what the paired gap below must equal")
    print("  confusion (rows = asked, cols = read):")
    for i, row in enumerate(read["confusion"]):
        print(f"    {i}  {row}")

    report = {
        "name": record["name"], "codec": codec.name, "n_classes": n_classes,
        "params": record["model"]["params"], "ceiling": ceiling,
        "classifier": {k: v for k, v in read.items() if k != "posterior"},
        "conditional_bits_per_drawing": float(true_bits.mean()),
    }

    if args.against:
        control, uncond = control_bits(args.against, programs)
        if control["corpus"]["val"] != record["corpus"]["val"]:
            raise SystemExit(
                f"{control['name']} was scored on a different val split; a "
                "cross-split difference measures the split"
            )
        # `paired_delta(a, b)` is mean(a - b), so unconditional first: the
        # difference *is* I(X;C) and is positive when conditioning helps.
        gap = paired_delta(list(uncond), [float(b) for b in true_bits])
        implied = ceiling["entropy_bits"] - read["class_bits"]
        print(f"\nthe paired gap against {control['name']}:")
        print(f"  unconditional {uncond.mean():.2f} -> conditional "
              f"{true_bits.mean():.2f} bits/drawing")
        print(f"  Δ {gap['delta']:+.4f} ±{gap['ci95']:.4f} bits saved by the label "
              f"(95% CI over {len(programs)} paired programs)")
        print(f"  the classifier implies I(X;C) = {implied:+.4f}; the paired gap "
              f"says {gap['delta']:+.4f}. Two instruments, one identity.")
        max_accuracy = 1.0 - fano_error_bound(read["class_bits"], n_classes)
        print(f"  Fano bound at this H(class|drawing): no reader of these drawings "
              f"can exceed {max_accuracy:.4f} accuracy; this one reads "
              f"{read['accuracy']:.4f}")
        report["control"] = control["name"]
        report["gap"] = gap
        report["unconditional_bits_per_drawing"] = float(uncond.mean())
        report["implied_mutual_information"] = implied
        report["fano_max_accuracy"] = max_accuracy

    if args.samples:
        control_report = controllability(
            model, codec, n_classes, n=args.samples,
            max_new=min(max_len, int(record["config"]["gen_cap"]
                                     * record["val_lengths"]["p99"])),
            device=args.device, max_len=max_len, batch_size=args.batch_size,
        )
        print(f"\ncontrollability: asked for each class {args.samples // n_classes}x, "
              f"{control_report['n_nonempty']}/{args.samples} decoded non-empty")
        print(f"  the model reads back its own class {control_report['accuracy']:.4f} "
              f"of the time, against chance {control_report['chance']:.3f}")
        print("  confusion (rows = asked, cols = read):")
        for i, row in enumerate(control_report["confusion"]):
            print(f"    {i}  {row}")
        report["controllability"] = control_report

    out = args.out or RUNS / f"conditioning_{record['name']}.json"
    out.write_text(json.dumps(report, indent=2, default=float))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
