"""Sampling settings with a stable report identity.

A sampler is part of a generation result. Keeping its parser and key in one
place prevents two report drivers from spelling the same setting differently —
or, worse, overwriting one setting with another under one filename.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass

MASK_MODES = frozenset({"raw", "vm_safe", "canonical"})
POLICY_DIGEST_RE = re.compile(r"[0-9a-f]{12}")


@dataclass(frozen=True)
class SamplingConfig:
    """The AR decode controls that are part of a generation result's identity.

    `mask_mode` and `policy_digest` were added for the state branch
    (`docs/state.md`). They default to `None` and then contribute
    nothing to `slug`, so every existing report key is byte-identical to what it
    was -- a structural mask has to *extend* the sampler identity, because an
    unkeyed side channel is how one setting silently replaces another
    (`docs/traps.md`).
    """

    top_k: int | None = 40
    temperature: float = 1.0
    #: Which legality mask the decoder applied: `raw`, `vm_safe` or `canonical`.
    #: `None` means the question was not asked -- the sampler predates the mask --
    #: and is *not* the same claim as `raw`, which asserts a mask was available
    #: and deliberately not applied.
    mask_mode: str | None = None
    #: `LanguagePolicy.digest`. A mask over a different allowlist is a different
    #: experiment, and the allowlist cannot be recovered from `mask_mode` alone.
    policy_digest: str | None = None

    def __post_init__(self) -> None:
        if self.top_k is not None and self.top_k < 1:
            raise ValueError(f"top_k must be positive or None, got {self.top_k}")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError(
                f"temperature must be finite and positive, got {self.temperature}"
            )
        if self.mask_mode is None:
            if self.policy_digest is not None:
                raise ValueError(
                    "policy_digest requires mask_mode; an unmasked historical "
                    "sampler has no language policy in its identity"
                )
            return
        if self.mask_mode not in MASK_MODES:
            raise ValueError(
                f"mask_mode must be one of {sorted(MASK_MODES)}, got "
                f"{self.mask_mode!r}"
            )
        if self.policy_digest is None:
            raise ValueError(
                f"mask_mode={self.mask_mode!r} requires a policy_digest; the same "
                "mask name over two allowlists is two different experiments"
            )
        if POLICY_DIGEST_RE.fullmatch(self.policy_digest) is None:
            raise ValueError(
                "policy_digest must be the 12-character lowercase hexadecimal "
                f"LanguagePolicy digest, got {self.policy_digest!r}"
            )

    @property
    def slug(self) -> str:
        """Filesystem-safe identity, including fixed legal-output support."""
        top_k = "all" if self.top_k is None else str(self.top_k)
        temperature = repr(float(self.temperature)).replace("-", "m").replace(".", "p")
        slug = f"k{top_k}_t{temperature}_legal"
        if self.mask_mode is not None:
            slug += f"_{self.mask_mode}"
            slug += f"_{self.policy_digest}"
        return slug

    def as_dict(self) -> dict[str, int | float | bool | str | None]:
        return {
            "top_k": self.top_k,
            "temperature": self.temperature,
            "forbid_specials": True,
            "mask_mode": self.mask_mode,
            "policy_digest": self.policy_digest,
        }


def parse_top_k(value: str) -> int | None:
    """Argparse type: a positive integer, or ``none``/``all`` for full support."""
    if value.lower() in {"none", "all"}:
        return None
    try:
        top_k = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("top-k must be a positive integer or 'none'") from exc
    if top_k < 1:
        raise argparse.ArgumentTypeError("top-k must be a positive integer or 'none'")
    return top_k


def parse_temperature(value: str) -> float:
    """Argparse type that refuses silent near-greedy coercion at zero or below."""
    try:
        temperature = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("temperature must be finite and positive") from exc
    if not math.isfinite(temperature) or temperature <= 0:
        raise argparse.ArgumentTypeError("temperature must be finite and positive")
    return temperature
