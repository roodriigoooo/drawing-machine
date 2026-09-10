"""C4: does the teacher-forced selectivity survive the model's own outputs?

C3 asks whether a controlled prefix change moves the *conditional distribution*.
It is answered with the truth in hand at every position.  A model can carry an
effect that large under teacher forcing and lose all of it the moment it has to
condition on symbols it emitted itself -- that failure has a name in
`docs/directions.md` §4.1, **exposure-limited context use**, and C4 is the only
instrument here that can assign it.

The estimand is deliberately the free-running mirror of C3's.  Let `c_w` be a
completion decoded from world `w`'s prompt and `d(c, y)` the normalised byte
distance between the completion's first target-length block and a target `y`:

```text
f(w)  = d(c_w, y_other) - d(c_w, y_own)      positive: this world leaned its own way
D_gen = 1/2 * [f(A) + f(B)]
```

Same symmetric shape as `D`, same cancellation of an unconditional preference
for either target string, and it reads in normalised distance rather than bits
because a sampled completion has no likelihood of its own.

**Both of C3's matched controls have a generation twin, and they are here.**
A raw `D_gen` says a completion moved when the prefix moved; it does not say the
*relation* is what moved it, and that gap is the one Direction 2 exists to close
(`docs/directions.md` §2.2).  So the same three scored blocks C3 uses appear
again, in exactly C3's pairing -- `x_A` with `y_A`, `bp_A` with `y_A`:

```text
D_gen_target_control = 1/2 * [(d(c_A,ct_B) - d(c_A,ct_A)) + (d(c_B,ct_A) - d(c_B,ct_B))]
D_gen_block_control  = 1/2 * [(d(cb_A,y_B) - d(cb_A,y_A)) + (d(cb_B,y_A) - d(cb_B,y_B))]
Delta_gen            = D_gen - D_gen_block_control
```

The donor-target control costs **no extra decoding at all**: it scores the same
two completions against the control targets.  The block control needs one more
decode, not two, because `block_prefix_a == prefix_a` in both venues -- the same
identity C3's block control has, which is why `D_minus_block_control` there sits
so close to raw `D`.  `cb_A` is therefore `c_A` byte for byte, and `Delta_gen`
reduces to `1/2 * [f(B) - f_block(B)]`: the relevant edit's effect on world B
minus an equally sized irrelevant edit's.  That is a smaller statistic than it
looks, and saying so here is cheaper than a reader rediscovering it.

`D_gen` stays the frozen primary endpoint, because that is what
[`docs/context-protocol-v4.json`](../../docs/context-protocol-v4.json) declared
before the first run.  The controls are reported beside it, and the rule that
reads them is declared in v5 *before* the run that carries them.

**Three rules that make it a measurement rather than a demonstration.**

- Both worlds consume the **same pre-generated uniforms**, one per row per step.
  Two decodes seeded identically would still diverge, because the prompts differ
  and the distribution therefore differs from the first step; a shared variate
  block is what makes the pair a pair (`dm/models/transformer.py::generate`).
- The structural mask stays **off**.  A mask makes legality differences look
  like context use, which is the Direction 1 result this project already paid
  for (`docs/state.md`).
- **A row that never reaches the target stays in the denominator.**  Its two
  distances are both defined as 1, so it contributes exactly zero to `f` -- no
  evidence either way -- and `reach_rate` is published beside every accuracy
  number.  Conditioning accuracy on reach without publishing reach is the
  survivorship error §4.9 forbids.
"""

from __future__ import annotations

import zlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from ..data.augment import Affine
from ..isa.codec import BOS, PAD, Codec, HaltMonitor
from ..isa.spec import ISAError, Op
from ..isa.state import LanguagePolicy
from .context import BUILDER_SEED, ContextCase, _bootstrap, signature, target_is_legal
from .context_cases import recover_transform, translation

#: Venues whose relation is a translation of the prompt's own last block, and
#: venues whose relation is a D4 image of it.  A completion is checked against
#: the relation its *own* prompt displays, never against the other world's.
STEP_VENUES = ("synthetic_flat_step", "synthetic_flat_step_d4r1",
               "synthetic_flat_step_yaxis")
SHAPE_VENUES = ("composed_shape_copy2",)
VENUE_DRAW_KEYS = STEP_VENUES + SHAPE_VENUES

#: The two decode arms.  `primary` carries the relation-bearing edit and
#: `block` the matched irrelevant one; both worlds of both arms share one
#: variate block, so all four completions of a case are paired.
ARMS = ("primary", "block")


@dataclass(frozen=True)
class CompletionConfig:
    """Everything about a C4 decode that is part of its identity."""

    draws: int = 8
    cap: int = 512
    top_k: int | None = 40
    temperature: float = 1.0
    seed: int = BUILDER_SEED
    #: Per-venue overrides for `draws`, as `((venue, draws), ...)` so the config
    #: stays hashable and orderable.  The frozen promotion rule expands the
    #: *co-primary* venues; spending 64 draws on a diagnostic whose interval
    #: already excludes zero buys nothing, and the column expansion does buy --
    #: `hit_own_rate` -- is a co-primary question.
    draws_by_venue: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        if self.draws < 1 or self.cap < 1:
            raise ValueError("draws and cap must be positive")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError("top_k must be positive or None")
        if not (self.temperature > 0):
            raise ValueError("temperature must be positive")
        for venue, draws in self.draws_by_venue:
            if venue not in VENUE_DRAW_KEYS:
                raise ValueError(f"unknown venue in draws_by_venue: {venue!r}")
            if draws < 1:
                raise ValueError(f"{venue}: draws must be positive")

    def draws_for(self, venue: str) -> int:
        return dict(self.draws_by_venue).get(venue, self.draws)

    def as_dict(self) -> dict:
        draws: int | dict = self.draws
        if self.draws_by_venue:
            draws = {"default": self.draws, **dict(self.draws_by_venue)}
        return {
            "draws_per_world": draws,
            "cap_symbols": self.cap,
            "top_k": self.top_k,
            "temperature": self.temperature,
            "variate_seed": self.seed,
            "structural_mask": "off",
            "shared_uniform_variates": True,
        }


def _safe(fn: Callable, *args):
    """Recovery oracles parse; a freely decoded block need not be parsable.

    Narrow on purpose: an unknown opcode or a truncated instruction is an
    expected outcome of raw decoding and means "no relation recovered", while
    anything else is a fault in the instrument and must not be swallowed.
    """
    try:
        return fn(*args)
    except (ISAError, ValueError, IndexError, KeyError):
        return None


def _distance(block: bytes, target: bytes) -> float:
    """Normalised byte distance, with a short block counted as maximally far."""
    if len(block) < len(target):
        return 1.0
    return sum(a != b for a, b in zip(block, target)) / max(1, len(target))


def _common_prefix(block: bytes, target: bytes) -> int:
    count = 0
    for a, b in zip(block, target):
        if a != b:
            break
        count += 1
    return count


def _expected_relation(case: ContextCase, world: str):
    """The relation this world's prompt displays, in the venue's own terms.

    For a step venue the prompt's last block is copy 2 and the target is copy 3,
    so the relation is one step `d_w` -- `d_A` read off the recorded transform
    and `d_B` off it plus the recorded intervention. For a shape venue the
    relation is the scene's orbit element, which is a property of the scene and
    identical in both worlds; there it is the *content* that separates them.
    """
    if case.venue in SHAPE_VENUES:
        return Affine(dx=case.transform[0], dy=case.transform[1],
                      mirror=bool(case.transform[2]),
                      quarter_turns=case.transform[3])
    step_a = (case.transform[0] // 2, case.transform[1] // 2)
    if world == "a":
        return step_a
    return (step_a[0] + case.relevant.dx, step_a[1] + case.relevant.dy)


def _prompt_for(case: ContextCase, arm: str, world: str) -> bytes:
    if arm == "primary":
        return case.prefix_a if world == "a" else case.prefix_b
    return case.block_prefix_a if world == "a" else case.block_prefix_b


def _measure(case: ContextCase, arm: str, world: str, emitted: bytes,
             halted: bool, policy: LanguagePolicy) -> dict:
    """One completion, scored against the targets its arm is paired with.

    Both arms use C3's pairing -- world A with `target_a`, world B with
    `target_b` -- because that is what makes `D_gen` and `D_gen_block_control`
    the same statistic on two different prefix edits.  The *behavioural*
    columns are computed for the primary arm only: "did it apply the relation"
    is a question about the relation-bearing prompt, and answering it for an
    irrelevant edit would invite the two to be read as one rate.
    """
    prefix = _prompt_for(case, arm, world)
    own = case.target_a if world == "a" else case.target_b
    other = case.target_b if world == "a" else case.target_a
    width = case.target_bytes
    block = emitted[:width]
    reached = len(emitted) >= width
    canonical = target_is_legal(prefix, emitted, policy)
    # Three categories, not two. `HaltMonitor.done` fires on HALT *and* on an
    # unknown opcode, because `VM.run` breaks at both -- reporting them together
    # would score a dead byte as a clean termination.
    if not halted:
        termination = "cap"
    elif canonical and emitted and emitted[-1] == int(Op.HALT):
        termination = "halt"
    else:
        termination = "fault"
    row = {
        "case_id": case.case_id,
        "venue": case.venue,
        "arm": arm,
        "world": world,
        "emitted_bytes": len(emitted),
        "target_bytes": width,
        "reached": reached,
        "termination": termination,
        "distance_own": _distance(block, own),
        "distance_other": _distance(block, other),
        "completion_canonical": bool(canonical),
    }
    if arm != "primary":
        return row
    control_own = (case.control_target_a if world == "a"
                   else case.control_target_b)
    control_other = (case.control_target_b if world == "a"
                     else case.control_target_a)
    source = prefix[case.relevant.start:case.relevant.stop]
    expected = _expected_relation(case, world)
    if case.venue in SHAPE_VENUES:
        realised = _safe(recover_transform, source, block) if reached else None
    else:
        realised = _safe(translation, source, block) if reached else None
    row.update({
        # The donor-target control, read off the completion already decoded.
        "control_distance_own": _distance(block, control_own),
        "control_distance_other": _distance(block, control_other),
        "hit_own": reached and block == own,
        "hit_other": reached and block == other,
        "longest_compatible_prefix": _common_prefix(block, own),
        "skeleton_match": bool(reached
                               and _safe(signature, block) == case.target_signature),
        # Did it apply the relation its own prompt shows, whatever content it
        # applied that relation to? Separating this from `hit_own` is what tells
        # a wrong-offset copy apart from an unrelated block.
        "relation_consistent": bool(reached and realised == expected),
        # `relation_recovered` without `relation_consistent` is a copy at the
        # wrong offset, which is a different failure from an unrelated block.
        "relation_recovered": realised is not None,
    })
    return row


@torch.no_grad()
def complete_cases(model, cases: list[ContextCase], codec: Codec,
                   policy: LanguagePolicy, *, config: CompletionConfig,
                   device: str | torch.device = "cpu") -> list[dict]:
    """Decode all four prompts of every case under one shared block of uniforms.

    Cases are grouped by `(venue, draws, prompt length)` because `generate`
    prefills one prompt tensor for the whole batch; within a group every arm is
    decoded with identical row order, identical batch shape and the identical
    variate block, which is what `docs/directions.md` §4.9 means by paired.

    The venue is in the key because the promotion rule sets `draws` per venue,
    and a group has to have one row count; the seed carries all three parts so
    two groups can never silently share a stream.

    **Identical prompts are decoded once.** `block_prefix_a == prefix_a` in both
    venues, so the block arm's world A is the primary arm's world A byte for
    byte -- decoding it again would spend a third of the run reproducing a
    tensor already in hand. The aliasing is by full-group comparison rather than
    by assumption, so a venue that ever breaks the identity decodes properly.
    """
    model.eval()
    groups: dict[tuple[str, int, int], list[ContextCase]] = {}
    for case in cases:
        key = (case.venue, config.draws_for(case.venue), case.prefix_bytes)
        groups.setdefault(key, []).append(case)
    out: list[dict] = []
    for (venue, draws, prefix_bytes), members in sorted(groups.items()):
        members = sorted(members, key=lambda case: case.case_id)
        rows = len(members) * draws
        # One generator per group, seeded from the group's own identity, so a
        # rerun reproduces the stream whatever order the groups are visited in.
        generator = torch.Generator().manual_seed(
            (config.seed * 1_000_003 + zlib.crc32(venue.encode()) * 31
             + prefix_bytes * 7 + draws) % (2 ** 31)
        )
        variates = torch.rand(rows, prefix_bytes * codec.stride + config.cap,
                              generator=generator)
        decoded: dict[tuple[str, str], tuple[list[bytes], np.ndarray]] = {}
        for arm in ARMS:
            for world in ("a", "b"):
                prompts = [_prompt_for(case, arm, world) for case in members]
                alias = next(
                    (key for key, seen in _seen_prompts(decoded, members, ARMS)
                     if seen == prompts), None,
                )
                if alias is not None:
                    decoded[arm, world] = decoded[alias]
                    continue
                encoded = torch.tensor(
                    [codec.encode(prompt) for prompt in prompts
                     for _ in range(draws)],
                    dtype=torch.long, device=device,
                )
                monitor = HaltMonitor(codec, rows)
                ids = model.generate(
                    rows, max_new=config.cap, temperature=config.temperature,
                    top_k=config.top_k, device=device, monitor=monitor,
                    prompt=encoded, forbid=(PAD, BOS), variates=variates,
                )
                decoded[arm, world] = (
                    [codec.decode(row.tolist()) for row in ids.cpu()],
                    monitor.done.copy(),
                )
        for index, case in enumerate(members):
            for arm in ARMS:
                for world in ("a", "b"):
                    programs, done = decoded[arm, world]
                    prefix = _prompt_for(case, arm, world)
                    for draw in range(draws):
                        row = index * draws + draw
                        program = programs[row]
                        if not program.startswith(prefix):
                            raise ValueError(
                                f"{case.case_id}: the decoded row does not begin "
                                "with its own prompt; prompt and monitor have "
                                "desynchronised"
                            )
                        record = _measure(case, arm, world,
                                          program[len(prefix):],
                                          bool(done[row]), policy)
                        record["draw"] = draw
                        out.append(record)
    return out


def _seen_prompts(decoded: dict, members: list[ContextCase], arms):
    """Already-decoded `(arm, world)` keys with the prompt list each one used."""
    for arm in arms:
        for world in ("a", "b"):
            if (arm, world) in decoded:
                yield (arm, world), [_prompt_for(case, arm, world)
                                     for case in members]


def per_case(rows: list[dict], cases: list[ContextCase]) -> list[dict]:
    """Collapse draws into one paired preference per case.

    The draw is not the unit of inference. Draws inside a case share a prompt
    and a checkpoint, so resampling them would count one case many times --
    the same rule that makes the source/donor component, and never the token,
    C3's resampling unit.
    """
    index = {case.case_id: case for case in cases}
    grouped: dict[str, dict[tuple[str, str], list[dict]]] = {}
    for row in rows:
        key = (row["arm"], row["world"])
        grouped.setdefault(row["case_id"], {}).setdefault(key, []).append(row)
    out = []
    for case_id, arms in grouped.items():
        case = index[case_id]

        def mean(arm: str, world: str, field: str, arms=arms) -> float:
            values = [row[field] for row in arms.get((arm, world), [])]
            return float(np.mean(values)) if values else float("nan")

        def contrast(arm: str, own: str, other: str) -> dict:
            """`1/2 * [f(A) + f(B)]` for one arm and one pair of targets."""
            lean_a = mean(arm, "a", other) - mean(arm, "a", own)
            lean_b = mean(arm, "b", other) - mean(arm, "b", own)
            return {"value": 0.5 * (lean_a + lean_b),
                    "lean_a": lean_a, "lean_b": lean_b}

        # f(w) = d(c_w, y_other) - d(c_w, y_own); positive means this world's
        # completions leaned toward its own compatible target.
        primary = contrast("primary", "distance_own", "distance_other")
        target_control = contrast("primary", "control_distance_own",
                                  "control_distance_other")
        block_control = contrast("block", "distance_own", "distance_other")
        every = arms.get(("primary", "a"), []) + arms.get(("primary", "b"), [])
        out.append({
            "case_id": case_id,
            "venue": case.venue,
            "source_index": case.source_index,
            "donor_index": case.donor_index,
            "control_donor_index": case.control_donor_index,
            "draws": len(every),
            "D_gen": primary,
            "D_gen_target_control": target_control,
            "D_gen_block_control": block_control,
            # The two controlled contrasts, in C3's own arithmetic.
            "D_gen_minus_control": {
                "value": primary["value"] - target_control["value"]},
            "D_gen_minus_block_control": {
                "value": primary["value"] - block_control["value"]},
            "reach_rate": float(np.mean([row["reached"] for row in every])),
            "hit_own_rate": float(np.mean([row["hit_own"] for row in every])),
            "hit_other_rate": float(np.mean([row["hit_other"] for row in every])),
            "relation_consistent_rate": float(
                np.mean([row["relation_consistent"] for row in every])),
            "skeleton_match_rate": float(
                np.mean([row["skeleton_match"] for row in every])),
            "canonical_rate": float(
                np.mean([row["completion_canonical"] for row in every])),
            "halt_rate": float(
                np.mean([row["termination"] == "halt" for row in every])),
            "fault_rate": float(
                np.mean([row["termination"] == "fault" for row in every])),
            "cap_rate": float(
                np.mean([row["termination"] == "cap" for row in every])),
            "relation_recovered_rate": float(
                np.mean([row["relation_recovered"] for row in every])),
            "longest_compatible_prefix_frac": float(np.mean(
                [row["longest_compatible_prefix"] / max(1, row["target_bytes"])
                 for row in every])),
        })
    return out


def summarise_completions(rows: list[dict], *, seed: int = BUILDER_SEED,
                          reps: int = 2000) -> dict:
    """Per-venue C4 summary, with the promotion rule's half-width."""

    def mean(field: str) -> float:
        return float(np.mean([row[field] for row in rows])) if rows else float("nan")

    def contrast_mean(field: str, part: str = "value") -> float:
        return (float(np.mean([row[field][part] for row in rows])) if rows
                else float("nan"))

    #: Every contrast gets the same component bootstrap. A control reported as a
    #: point estimate with no interval cannot be said to be "measured ~0", which
    #: is the exact claim `docs/context.md` has to make about it.
    contrasts = ("D_gen", "D_gen_target_control", "D_gen_block_control",
                 "D_gen_minus_control", "D_gen_minus_block_control")
    bootstraps = {name: _bootstrap(rows, name, seed=seed, reps=reps,
                                   unit="value")
                  for name in contrasts}
    bootstrap = bootstraps["D_gen"]
    low, high = bootstrap["ci95"]
    return {
        "n_cases": len(rows),
        "draws": int(sum(row["draws"] for row in rows)),
        "D_gen_mean": contrast_mean("D_gen"),
        "D_gen_target_control_mean": contrast_mean("D_gen_target_control"),
        "D_gen_block_control_mean": contrast_mean("D_gen_block_control"),
        "D_gen_minus_control_mean": contrast_mean("D_gen_minus_control"),
        "D_gen_minus_block_control_mean": contrast_mean(
            "D_gen_minus_block_control"),
        "lean_a_mean": contrast_mean("D_gen", "lean_a"),
        "lean_b_mean": contrast_mean("D_gen", "lean_b"),
        **{f"bootstrap_{name}": value for name, value in bootstraps.items()},
        # The frozen promotion metric. Reported whether or not it passes, so the
        # decision to expand is never taken on a number chosen afterwards.
        "half_width": (float(high - low) / 2.0 if np.isfinite(low) and
                       np.isfinite(high) else float("nan")),
        "reach_rate": mean("reach_rate"),
        "hit_own_rate": mean("hit_own_rate"),
        "hit_other_rate": mean("hit_other_rate"),
        "relation_consistent_rate": mean("relation_consistent_rate"),
        "skeleton_match_rate": mean("skeleton_match_rate"),
        "canonical_rate": mean("canonical_rate"),
        "halt_rate": mean("halt_rate"),
        "fault_rate": mean("fault_rate"),
        "cap_rate": mean("cap_rate"),
        "relation_recovered_rate": mean("relation_recovered_rate"),
        "longest_compatible_prefix_frac": mean("longest_compatible_prefix_frac"),
    }


def summarise_by_venue(rows: list[dict], *, seed: int = BUILDER_SEED,
                       reps: int = 2000) -> dict[str, dict]:
    return {
        venue: summarise_completions([row for row in rows
                                      if row["venue"] == venue],
                                     seed=seed, reps=reps)
        for venue in sorted({row["venue"] for row in rows})
    }


__all__ = [
    "ARMS",
    "SHAPE_VENUES",
    "STEP_VENUES",
    "CompletionConfig",
    "complete_cases",
    "per_case",
    "summarise_by_venue",
    "summarise_completions",
]
