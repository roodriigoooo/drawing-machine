"""What a class label buys, on an axis where `bits/drawing` provably cannot see it.

**The ceiling is arithmetic, and it is below the floor.** Conditioning a model on
the class can improve `bits/drawing` by at most the mutual information between
drawing and class,

    I(X; C) = H(C) − H(C | X) ≤ H(C),

which on the balanced five-category QuickDraw corpus is `log2(5) = 2.32` bits. That
corpus's run-to-run resolution floor is **~2.5 bits**. So the likelihood effect of
class conditioning is **unresolvable at k = 2 before any run starts**, by
construction — the same argument the constructed corpora are built on, applied to
an axis rather than to a dataset, and it is the reason this module exists instead
of a row in the sweep table.

**But the bound is an identity, not a nuisance**, and that is what makes the
number worth measuring anyway. At optimality the gap between the unconditional and
conditional arms *is* `I(X; C)`, so it measures **how much a drawing tells you
about its own category**, in bits, and the residual `H(C | X) = H(C) − Δ` is the
class uncertainty a perfect reader would be left with. Two things follow:

1. It can be cross-checked. `class_posterior` turns the conditional model into a
   classifier for free — `p(c | x) ∝ p(x | c)·p(c)`, no second model and no new
   parameters — and its own empirical `H(C | X)` must agree with `H(C) − Δ`.
2. Fano's inequality bounds the achievable error rate from `H(C | X)` alone, so a
   measured Δ predicts a classification accuracy *before* the classifier is run.
   Two instruments, one identity, which is the arrangement that caught three
   faults elsewhere in this project.

> **And the two instruments are not equally good, which is the sharper form of the
> floor argument.** The paired gap is a difference between *two models*, so it
> carries `I(X; C)` **plus** whatever the two runs differ by on their own — and on
> this project that second term is the resolution floor, 2.5 bits, larger than the
> whole 2.32-bit ceiling. Measured on two 30-step smoke arms where the true
> conditioning effect is ~0.01 bits, the paired gap read **+0.34** and the
> classifier read **+0.009**: the gap was ~37x the effect, and all of the excess
> was the arms differing from each other.
>
> `H(C | X)` from `class_posterior` has **no cross-model term** — it is five
> forward passes of one checkpoint — so it escapes the floor entirely. **The label's
> value cannot be measured by differencing two runs and can be measured inside
> one.** Report both, and read the single-model one.

> **Fano is a bound on the expected error of the true conditional, and an
> empirical accuracy can sit just above it on a finite sample.** The smoke reading
> was 0.3889 against a bound of 0.3870 at n = 90, where the accuracy's own
> standard error is ~0.05. A violation of that size is sampling noise; a large one
> would mean `H(C | X)` and the posterior disagree, which cannot happen if both
> come from the same array, so it would mean the array is not what it claims.

**And the axis's real claim is controllability**: asking for a cat has to produce
something that reads as a cat. `controllability` samples under each class and
classifies the samples with the same generative classifier, so the diagonal of its
confusion matrix is "did the conditioning do anything at generation time" —
measured on the model's own terms, with chance and ceiling both known.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import torch

from ..data.dataset import ProgramDataset, loader
from ..isa.codec import Codec
from .metrics import per_program_bits, sample_programs


def label_entropy(labels: list[int], n_classes: int | None = None) -> dict:
    """`H(C)` in bits, and the ceiling it puts on any conditioning gain.

    Empirical rather than `log2(n_classes)`: the corpus is balanced by
    `quickdraw._interleave`, but a `limit=` that lands mid-round is not, and a
    ceiling quoted from the category count would then be slightly too high — in
    the direction that makes a conditioning result look *worse* than it is.
    """
    counts = Counter(labels)
    n = max(1, len(labels))
    entropy = -sum((c / n) * math.log2(c / n) for c in counts.values() if c)
    classes = n_classes or (max(counts) + 1 if counts else 0)
    return {
        "n": len(labels),
        "n_classes": classes,
        "counts": {int(k): int(v) for k, v in sorted(counts.items())},
        "entropy_bits": entropy,
        "uniform_bits": math.log2(classes) if classes else 0.0,
        "chance_accuracy": max(counts.values()) / n if counts else 0.0,
    }


@torch.no_grad()
def class_log2_likelihood(
    model,
    programs: list[bytes],
    codec: Codec,
    n_classes: int,
    device: str | torch.device = "cpu",
    max_len: int = 2048,
    batch_size: int = 32,
) -> np.ndarray:
    """`(n_programs, n_classes)` of −log2 p(program | class), one pass per class.

    The whole conditional-as-classifier reading rests on this being the *same*
    scorer the likelihood table uses, so it calls `per_program_bits` rather than
    re-deriving a forward pass: a second implementation could disagree with
    `bits/drawing` and the cross-check in `class_posterior` would then be
    comparing two instruments instead of one identity.
    """
    if not n_classes:
        raise ValueError("class_log2_likelihood needs a conditional model")
    out = np.zeros((len(programs), n_classes), dtype=np.float64)
    for k in range(n_classes):
        dataset = ProgramDataset(programs, codec, max_len, labels=[k] * len(programs))
        batches = loader(dataset, batch_size, shuffle=False)
        out[:, k] = per_program_bits(model, batches, device)
    return out


def class_posterior(nll_bits: np.ndarray, prior: np.ndarray | None = None) -> dict:
    """Bayes over the class axis: `p(c | x) ∝ p(x | c)·p(c)`.

    A conditional generative model is a classifier and it costs nothing to ask —
    which matters here because the alternative is training a discriminative model,
    whose parameters this project counts and whose result would be about *that*
    model rather than about the conditioning.

    `H(C | X)` is returned in bits and is the quantity that must agree with
    `H(C) − Δbits/drawing`. It is the mean of `−log2 p(c* | x)` over the *true*
    class, not the mean posterior entropy: the identity is about the cost of
    transmitting the true label given the drawing.
    """
    n, k = nll_bits.shape
    log_prior = (np.log2(prior) if prior is not None
                 else np.full(k, -math.log2(k), dtype=np.float64))
    # −log2 p(x|c) + log2 p(c), then normalise over c. In log2 throughout so the
    # units never change hands.
    joint = -nll_bits + log_prior
    shifted = joint - joint.max(axis=1, keepdims=True)
    weights = np.exp2(shifted)
    posterior = weights / weights.sum(axis=1, keepdims=True)
    return {"n": n, "n_classes": k, "posterior": posterior,
            "predicted": posterior.argmax(axis=1)}


def conditional_class_bits(posterior: np.ndarray, labels: list[int]) -> dict:
    """`H(C | X)` and accuracy, from a posterior and the true classes."""
    truth = np.asarray(labels)
    picked = posterior[np.arange(len(truth)), truth]
    # Clipped because a posterior can underflow to exactly zero for a class the
    # model finds impossible, and an infinite mean would hide every other row.
    bits = -np.log2(np.clip(picked, 1e-300, None))
    predicted = posterior.argmax(axis=1)
    return {
        "n": len(truth),
        "accuracy": float((predicted == truth).mean()),
        "class_bits": float(bits.mean()),
        "class_bits_stderr": float(bits.std(ddof=1) / math.sqrt(max(1, len(bits)))),
        "confusion": [[int(((truth == t) & (predicted == p)).sum())
                       for p in range(posterior.shape[1])]
                      for t in range(posterior.shape[1])],
    }


def fano_error_bound(class_bits: float, n_classes: int) -> float:
    """The smallest error rate any classifier can achieve at this `H(C | X)`.

    Fano: `H(C | X) ≤ H(Pe) + Pe·log2(k − 1)`. Inverted numerically because the
    right-hand side is monotone in `Pe` on `[0, 1 − 1/k]` and a closed form would
    be a transcription risk for no gain. **This is what turns a likelihood gap
    into a prediction about a classifier**: measure Δbits, get `H(C | X)`, and the
    accuracy of *any* reader of these drawings is bounded before one is built.
    """
    if n_classes < 2:
        return 0.0

    def rhs(pe: float) -> float:
        if pe <= 0.0:
            return 0.0
        h = -pe * math.log2(pe) - (1 - pe) * math.log2(1 - pe) if pe < 1 else 0.0
        return h + pe * math.log2(n_classes - 1) if n_classes > 2 else h

    lo, hi = 0.0, 1.0 - 1.0 / n_classes
    if class_bits >= rhs(hi):
        return hi
    for _ in range(60):
        mid = (lo + hi) / 2
        if rhs(mid) < class_bits:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


@torch.no_grad()
def controllability(
    model,
    codec: Codec,
    n_classes: int,
    n: int = 128,
    max_new: int = 512,
    device: str | torch.device = "cpu",
    top_k: int | None = 40,
    max_len: int = 2048,
    batch_size: int = 32,
) -> dict:
    """Ask for each class, then ask the model what it drew.

    Round-robin over classes so the request distribution is uniform and the
    diagonal is directly comparable with `chance`. **The classifier is the same
    model**, which is the point and also the limitation: this measures whether the
    conditioning changed what the sampler produces, in the model's own terms. It
    is not evidence that a *human* would call the sample a cat, and nothing here
    should be quoted as if it were — that needs the class-conditional set metrics
    in `dm/eval/quality.py`, against real drawings of one class.

    Samples that decode to nothing are excluded and counted: an empty program has
    no class, and scoring one would credit whichever class assigns the shortest
    sequence the most mass.
    """
    asked = torch.arange(n, device=device) % n_classes
    programs, _ = sample_programs(model, codec, n=n, max_new=max_new, device=device,
                                  top_k=top_k, classes=asked)
    wanted = asked.cpu().numpy()
    keep = [i for i, p in enumerate(programs) if p]
    if not keep:
        return {"n": n, "n_nonempty": 0, "accuracy": float("nan")}
    nll = class_log2_likelihood(model, [programs[i] for i in keep], codec, n_classes,
                               device=device, max_len=max_len, batch_size=batch_size)
    report = conditional_class_bits(class_posterior(nll)["posterior"],
                                    [int(wanted[i]) for i in keep])
    return {
        "n": n,
        "n_nonempty": len(keep),
        "mean_bytes": sum(len(programs[i]) for i in keep) / len(keep),
        "chance": 1.0 / n_classes,
        **report,
    }
