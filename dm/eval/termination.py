"""Termination diagnostics: is the model's halting hazard calibrated?

Every arm generates short. Median generated length is 24-33 bytecode bytes
against a real 41, on arms whose `len EMD` is as low as 4.4, and three sweeps
have reported it without explaining it. Two mechanisms produce it and they call
for opposite fixes:

* **Calibration.** The model over-assigns HALT at instruction boundaries. Then
  free-running generation stops early *by construction*, no sampler is at fault,
  and the fix is in the model or the data.
* **The sampler.** The hazard is right and something in decoding shortens the
  stream anyway -- `top_k` renormalising tail mass onto a mode that includes
  HALT is the obvious suspect, and it is a decode setting, not a property of the
  representation.

This module measures the first, so that what is left over is the second. It is
teacher-forced: one forward pass over the val split, no generation, no sampling
noise, and the same number for every codec whose alphabet has one symbol per
bytecode byte.

`bits_per_drawing` cannot see any of this. The halting hazard is a handful of
bits out of ~161, so a model can be first on likelihood and still generate a
length distribution that is visibly wrong -- which is exactly what the bit arm
does at the converged point.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..data.dataset import ProgramDataset, loader
from ..isa.codec import Codec, opcode_mask
from ..isa.spec import Op


class StrideUnsupported(NotImplementedError):
    """Raised for codecs that spend several symbols on one bytecode byte.

    Under the bit codec, P(the next byte is HALT) is a product over eight
    conditional bit probabilities, and only the first factor lies on the
    teacher-forced path: the rest condition on a prefix of zeros the true
    program does not contain. Reading it off one forward pass would silently
    measure P(the first bit of the next byte is 0), which is not the hazard and
    is not comparable to the stride-1 arms. Eight chained passes per boundary
    would give the true value; until something needs it, refusing is the honest
    behaviour.
    """


def halt_symbol(codec: Codec) -> int:
    """The symbol that opens a HALT instruction, without reaching into privates.

    Encoding a one-instruction program is the codec's own answer to the
    question, so this stays correct for any codec added later.
    """
    return codec.encode(bytes([int(Op.HALT)]))[0]


@torch.no_grad()
def halt_hazard(
    model,
    codec: Codec,
    programs: list[bytes],
    device: str | torch.device = "cpu",
    batch_size: int = 64,
    max_len: int = 2048,
) -> dict:
    """Modelled vs empirical probability of halting at each instruction boundary.

    Both are conditioned on reaching the boundary, so they are hazards and the
    comparison is per-boundary rather than per-program. `excess_halts` is the
    at-risk-weighted sum of (modelled - empirical): the expected number of extra
    halts the model would fire across one pass through the corpus, and the sign
    that says which way generation will be wrong.

    `implied_mean_instructions` builds the survival function from the pooled
    modelled hazards and sums it, which is the mean length the model's own
    halting behaviour implies. Comparing it to the truth turns a hazard curve
    into the one number the generation metrics are already reported in.
    """
    if codec.stride != 1:
        raise StrideUnsupported(
            f"{codec.name} spends {codec.stride} symbols per bytecode byte; "
            "the halting hazard is not on the teacher-forced path"
        )
    target = halt_symbol(codec)
    # A program the encoder truncated has boundaries the model never scored, so
    # it would raise `at_risk` without ever being able to raise `halted_here`
    # and would bias every hazard downward. Dropped, and counted, rather than
    # silently included.
    fits = [p for p in programs if len(codec.with_bos(p)) <= max_len]
    dropped = len(programs) - len(fits)
    dataset = ProgramDataset(fits, codec, max_len)
    boundaries = [[i for i, is_op in enumerate(opcode_mask(p)) if is_op] for p in fits]
    width = max((len(b) for b in boundaries), default=0)

    modelled = np.zeros(width)
    at_risk = np.zeros(width)
    halted_here = np.zeros(width)
    model.eval()
    for index, inputs, _targets in loader(dataset, batch_size, shuffle=False):
        probs = F.softmax(model(inputs.to(device)), dim=-1)[..., target].cpu().numpy()
        for row, program_id in enumerate(index.tolist()):
            positions = boundaries[program_id]
            # Position b of the logits predicts symbol b, i.e. the byte that
            # opens the instruction at byte offset b -- the BOS at sequence
            # position 0 is what shifts the two into alignment.
            usable = [b for b in positions if b < probs.shape[1]]
            for order, b in enumerate(usable):
                modelled[order] += probs[row, b]
                at_risk[order] += 1
            if len(usable) == len(positions) and positions:
                halted_here[len(positions) - 1] += 1

    live = at_risk > 0
    model_hazard = np.divide(modelled, at_risk, out=np.zeros(width), where=live)
    empirical = np.divide(halted_here, at_risk, out=np.zeros(width), where=live)

    def implied_mean(hazard: np.ndarray) -> float:
        """Sum of the survival function, which is the mean of a discrete length.

        `S_i = prod_{j<i} (1 - h_j) = P(len > i)` and `E[len] = sum_i P(len > i)`,
        so feeding this the *empirical* hazard has to return the corpus mean
        exactly. `implied_mean_from_empirical` does that on every call and is
        the instrument's own check: if it does not equal
        `true_mean_instructions`, the pooling or the alignment is wrong and no
        conclusion about the model is admissible.
        """
        survival, total = 1.0, 0.0
        for i in range(width):
            total += survival
            survival *= 1.0 - hazard[i]
        return total

    true_lengths = np.array([len(b) for b in boundaries], dtype=np.float64)
    return {
        "boundary": np.arange(width),
        "at_risk": at_risk,
        "model_hazard": model_hazard,
        "empirical_hazard": empirical,
        "excess_halts": float(((model_hazard - empirical) * at_risk).sum() / max(1, len(fits))),
        "implied_mean_instructions": implied_mean(model_hazard),
        "implied_mean_from_empirical": implied_mean(empirical),
        "true_mean_instructions": float(true_lengths.mean()),
        "n_programs": len(fits),
        "n_dropped": dropped,
    }


def summarise(report: dict, rows: int = 12) -> str:
    """One block per arm, short enough to sit next to the sweep table."""
    check = abs(report["implied_mean_from_empirical"] - report["true_mean_instructions"])
    headline = (
        f"implied mean {report['implied_mean_instructions']:.2f} instructions vs true "
        f"{report['true_mean_instructions']:.2f}  "
        f"(excess halts/program {report['excess_halts']:+.3f}; "
        f"n={report['n_programs']}, dropped={report['n_dropped']}; "
        f"self-check {'ok' if check < 1e-9 else f'FAILED by {check:.3e}'})"
    )
    lines = [
        headline,
        "| boundary | at risk | model P(HALT) | empirical | ratio |",
        "|---|---|---|---|---|",
    ]
    for i in report["boundary"][:rows]:
        empirical = report["empirical_hazard"][i]
        ratio = report["model_hazard"][i] / empirical if empirical else float("nan")
        lines.append(
            f"| {i} | {report['at_risk'][i]:.0f} | {report['model_hazard'][i]:.4f} | "
            f"{empirical:.4f} | {ratio:.2f} |"
        )
    return "\n".join(lines)


__all__ = ["StrideUnsupported", "halt_hazard", "halt_symbol", "summarise"]
