"""Metrics.

`bits_per_drawing` is the one number comparable across all four codecs: every
codec encodes the *same* bytecode, so the model's total negative log-likelihood
in bits is the cost of transmitting one drawing under that representation,
regardless of how many symbols it took. Per-token loss is not comparable and
must not be reported across arms.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..isa.codec import BOS, PAD
from ..vm.interp import VM, Trace
from ..vm.render import rasterize


def corpus_stats(programs: list[bytes], vm: VM | None = None) -> dict:
    vm = vm or VM()
    traces = [vm.run(p) for p in programs]
    faults = Counter(f.kind.value for t in traces for f in t.faults)
    n = max(1, len(programs))
    return {
        "n": len(programs),
        "validity": sum(t.valid for t in traces) / n,
        "nonempty": sum(not t.is_empty for t in traces) / n,
        "mean_bytes": sum(len(p) for p in programs) / n,
        "mean_prims": sum(len(t) for t in traces) / n,
        "faults": dict(faults.most_common()),
    }


def trace_points(trace: Trace, n: int = 256) -> np.ndarray:
    """Resample all geometry to a fixed point cloud for Chamfer."""
    pts: list[tuple[float, float]] = []
    for stroke in trace.strokes:
        pts.extend(stroke.points)
    for region in trace.regions:
        pts.extend(region.points)
    for disc in trace.discs:
        theta = np.linspace(0, 2 * math.pi, 16, endpoint=False)
        pts.extend(zip(disc.cx + disc.r * np.cos(theta), disc.cy + disc.r * np.sin(theta)))
    if not pts:
        return np.zeros((0, 2), dtype=np.float32)
    arr = np.asarray(pts, dtype=np.float32)
    if len(arr) > n:
        arr = arr[np.linspace(0, len(arr) - 1, n).astype(int)]
    return arr


def chamfer(a: Trace, b: Trace, n: int = 256) -> float:
    """Symmetric mean nearest-neighbour distance in canvas pixels."""
    pa, pb = trace_points(a, n), trace_points(b, n)
    if len(pa) == 0 or len(pb) == 0:
        return float("inf")
    d = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=-1)
    return float(d.min(axis=1).mean() + d.min(axis=0).mean()) / 2.0


def iou(a: Trace, b: Trace, size: int = 128, thresh: float = 0.25) -> float:
    ia = rasterize(a, size) > thresh
    ib = rasterize(b, size) > thresh
    union = np.logical_or(ia, ib).sum()
    return float(np.logical_and(ia, ib).sum() / union) if union else 1.0


@torch.no_grad()
def per_program_bits(model, loader, device: str | torch.device = "cpu") -> np.ndarray:
    """Bits to transmit each val program, indexed by its position in the split.

    Kept per-program rather than reduced on the spot because every codec encodes
    the *same* programs: holding the split fixed turns the codec comparison into
    a paired one, and the pairing is worth roughly an order of magnitude in
    resolution. Val sets differ from each other by ~3.5 bits/drawing while the
    codec effects under test are ~1 bit, so an unpaired mean cannot see them.
    """
    model.eval()
    bits = np.zeros(len(loader.dataset), dtype=np.float64)
    # A class-conditional model needs the label of each row, and the label is a
    # property of the *dataset*: the batch already carries `index` because
    # bucketing reorders programs. Nothing here changes for an unconditional arm.
    labels = getattr(loader.dataset, "labels", None)
    for index, inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        classes = None if labels is None else labels[index].to(device)
        logits = model(inputs, classes) if classes is not None else model(inputs)
        nll = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=PAD,
            reduction="none",
        ).view(targets.shape)
        # float64 only once off-device: MPS has no float64.
        bits[index.numpy()] = (nll.sum(dim=1) / math.log(2)).cpu().numpy()
    return bits


@torch.no_grad()
def bits_per_drawing(model, loader, device: str | torch.device = "cpu") -> dict:
    """Total NLL in bits per program, plus per-token loss for training curves.

    Only the first number is comparable across representations. `stderr` is over
    programs and describes this val set; it is *not* the uncertainty on a
    difference between two arms, which is what `paired_delta` reports.
    """
    bits = per_program_bits(model, loader, device)
    tokens = np.array([len(s) - 1 for s in loader.dataset.seqs], dtype=np.float64)
    total_bits, total_tokens, n = bits.sum(), tokens.sum(), max(1, len(bits))
    return {
        "bits_per_drawing": total_bits / n,
        "bits_per_drawing_stderr": float(bits.std(ddof=1) / math.sqrt(n)),
        "bits_per_token": total_bits / max(1.0, total_tokens),
        "nats_per_token": total_bits * math.log(2) / max(1.0, total_tokens),
        "tokens_per_drawing": total_tokens / n,
        "val_bits": [round(b, 4) for b in bits],
    }


def paired_delta(a: list[float], b: list[float]) -> dict:
    """Mean of `a - b` over the same programs, with a 95% interval.

    Valid only when both arms were scored on an identical, identically-ordered
    split -- which `TrainConfig.data_seed` is what guarantees.

    `stderr` describes *these two runs* and nothing else. Pairing removes the
    val-set variance and leaves the seed variance untouched, and on this project
    the second is the larger of the two: the same axis measured on two seeds
    moves 5-15x further than this interval allows. Anything combining several
    seeds must fold in the between-seed term as well -- see
    `scripts/sweep.py:paired_table`.
    """
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    stderr = d.std(ddof=1) / math.sqrt(len(d))
    return {
        "delta": float(d.mean()),
        "stderr": float(stderr),
        "ci95": float(1.96 * stderr),
        "n": len(d),
    }


def length_stats(programs: list[bytes], reference: list[bytes] | None = None,
                 cap_hit: Sequence[bool] | None = None) -> dict:
    """Termination, described three ways, because the obvious one is a trap.

    A program that stopped without halting is `no_halt`, but so is one the
    sampler cut short, and the cap is a compute knob rather than a property of
    the model. `truncated` isolates the rows that actually hit the cap, and
    `length_emd` -- earth-mover distance in bytes between the generated and
    real length distributions -- describes termination with no reference to any
    cap at all.

    Split out of `sample_quality` so the planner reports the same three
    columns. It is not a tidy-up: `dm/models/planner.py` states claim 3's
    falsifiable prediction as *"if `gen_length_emd` does not improve, the scale
    argument is weaker than PLAN.md section 5 claims"*, and until this existed
    the planner recorded `corpus_stats` alone -- so the one column the design
    named as its own falsifier was absent from every planner record while
    `scripts/sweep.py` read it and printed nan.

    `cap_hit` is the caller's, because the two arms cap different things: the
    AR arm caps a whole program at `max_new`, the planner caps each stroke at
    the decoder's context. Inferring one rule for both would report a number
    that is right for neither.
    """
    lengths = np.array([len(p) for p in programs], dtype=np.float64)
    if not len(lengths):
        return {}
    stats = {"len_p50": float(np.median(lengths))}
    if cap_hit is not None:
        stats["truncated"] = float(np.asarray(cap_hit, dtype=bool).mean())
    if reference:
        ref = np.array([len(p) for p in reference], dtype=np.float64)
        grid = np.linspace(0, max(lengths.max(), ref.max()), 512)
        cdf = lambda x: np.searchsorted(np.sort(x), grid, side="right") / len(x)
        stats["length_emd"] = float(np.trapezoid(np.abs(cdf(lengths) - cdf(ref)), grid))
    return stats


@torch.no_grad()
def sample_programs(model, codec, n: int = 128, max_new: int = 512,
                    device: str | torch.device = "cpu", temperature: float = 1.0,
                    top_k: int | None = None,
                    classes: torch.Tensor | None = None,
                    forbid_specials: bool = False,
                    variates: torch.Tensor | None = None,
                    mode: str = "standard",
                    prompt: torch.Tensor | None = None,
                    ) -> tuple[list[bytes], list[bool]]:
    """Generate and decode. Returns the programs and which rows hit the cap.

    Split out of `sample_quality` so a caller that wants the drawings themselves
    -- `scripts/resample.py`, to compare the sampled *set* against the corpus --
    does not have to re-derive the cap rule. That rule is the fragile part: a row
    hit the cap when its bytecode is as long as the block it was given, in
    *bytecode* bytes rather than symbols, so the stride has to be divided out and
    the bit codec is 8x off if it is not. `truncated` and `length_emd` both read
    it, and a second copy that drifted would make two reports of one checkpoint
    quietly incomparable.

    `forbid_specials` removes PAD/BOS from output support. They are controls,
    not bytecode targets, but this defaults false for training compatibility:
    changing eval support changes RNG consumption and can move later batches.

    A row hit the cap iff its halt monitor is still live after generation.
    Never infer this from returned tensor width: cached generation returns as
    soon as the final row stops, so that width is usually the last valid program
    length rather than the allocated budget.

    `variates`, when supplied, is a pre-generated `(n, max_new)` uniform block.
    It is the preferred path for a diagnostic that runs during training: the
    block keeps sampling off the global Torch stream and makes standard/soft
    comparisons consume exactly the same draws.

    `mode` is forwarded to the model's runtime decode seam. It defaults to
    standard for compatibility with every existing quality report.

    `prompt` is an optional `(n, P)` bytecode prefix every row continues, and it
    is forwarded rather than reimplemented so the cap rule below stays the only
    copy: a prompted decode returns the prefix verbatim, so a caller comparing
    decoded length against the returned width would call every prompted row
    truncated. Appended last, and defaulting to None, so every existing
    positional call is byte-identical in behaviour.

    `forbid_specials` is explicit because changing output support changes how
    many RNG draws an eval consumes, which can move later training batches. New
    side reports set it and name it; training-time eval keeps its recorded raw
    support until that regime is changed deliberately.
    """
    # PAD and BOS are model-control symbols, never bytecode. They are absent
    # from every target distribution but still occupy logits; full-support
    # sampling can otherwise emit them, spend a decode step on no byte, and
    # make cap detection depend on how often impossible output symbols won.
    monitor = codec.halt_monitor(n)
    ids = model.generate(
        n, max_new, temperature=temperature, top_k=top_k, device=device,
        monitor=monitor, classes=classes,
        forbid=(PAD, BOS) if forbid_specials else (), variates=variates,
        mode=mode, prompt=prompt,
    )
    programs = [codec.decode(row.tolist()) for row in ids.cpu()]
    # `generate` returns early once every row stops, so its returned width is
    # not the cap. Comparing decoded length with that width marks the last row
    # to halt as truncated on every draw (exactly 1/n in most old reports).
    # Monitor state is the source of truth: false means the allocated decode
    # budget ended while that row was still live.
    return programs, [not bool(done) for done in monitor.done]


def sample_quality(model, codec, n: int = 128, max_new: int = 512,
                   device: str | torch.device = "cpu", temperature: float = 1.0,
                   top_k: int | None = None,
                   reference: list[bytes] | None = None,
                   classes: torch.Tensor | None = None,
                   forbid_specials: bool = False,
                   variates: torch.Tensor | None = None,
                   mode: str = "standard") -> dict:
    """Generate, decode, execute. Validity is measured on what the VM says.

    Termination is reported three ways; `length_stats` says why.
    """
    programs, cap_hit = sample_programs(model, codec, n, max_new, device, temperature,
                                        top_k, classes, forbid_specials, variates,
                                        mode)
    stats = corpus_stats(programs)
    stats |= length_stats(programs, reference, cap_hit)
    stats["cap_rule"] = "halt_monitor"
    return {f"gen_{k}": v for k, v in stats.items()}
