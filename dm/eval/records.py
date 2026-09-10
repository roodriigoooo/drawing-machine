"""Getting from a run record back to the model and the split it was scored on.

Four functions, and they are here rather than in the script that first needed
them because the second script to need them must not re-derive the rule. The
rule that matters is `corpus_config`: a planner record does **not** serialise a
`TrainConfig`, and going around `PlannerTrainConfig.corpus()` is how a
re-measurement ends up on a split the run was never scored on -- which is the
run-4 fault, and it cost this project a whole comparison.

`composed_val_scenes` is the same rule for the one corpus that carries
*provenance*: the constructed compositional corpus knows which of its own bytes
are copies, and a metric reading it has to rebuild the identical scenes or every
number it reports is attributed to the wrong bytes.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from ..data import composed
from ..data.fingerprint import digest
from ..isa.codec import CODECS, N_SPECIAL, RelativeCodec, TokenCodec
from ..isa.spec import OPCODE_SLOTS, Kind, Tier
from ..models.planner import PlannerConfig, StrokePlanner
from ..models.transformer import Config, DrawingLM
from ..train import config_from_record
from ..train_planner import PlannerTrainConfig


def is_planner(record: dict) -> bool:
    return record.get("kind") == "planner"


def corpus_config(record: dict):
    """The `TrainConfig` naming this record's val split, either kind of record.

    A planner record serialises a `PlannerTrainConfig`, which is deliberately
    not a `TrainConfig` subclass -- `shape` means something different at each
    level. `PlannerTrainConfig.corpus()` is the one supported way to get from
    one to the other.
    """
    if not is_planner(record):
        return config_from_record(record)
    config = dict(record["config"])
    config["tier"] = Tier(config["tier"])
    config["categories"] = tuple(config["categories"])
    fields = PlannerTrainConfig.__dataclass_fields__
    return PlannerTrainConfig(**{k: v for k, v in config.items() if k in fields}).corpus()


def composed_val_scenes(record: dict,
                        programs: list[bytes] | None = None) -> list[composed.Scene]:
    """The val `Scene`s a `--data composed` record was scored on, with their
    copy spans -- rebuilt, because `runs/` stores weights and never a corpus.

    Two refusals rather than a best effort, because both failure modes are
    silent and both produce a plausible number attributed to the wrong bytes:

    - **A record from another corpus.** Nothing else in the project carries
      per-byte provenance, so there is no degraded reading to fall back to.
    - **A record trained on the structured spelling.** `REPEATX` has already
      folded the copies away, so the copies are not in the program the model saw
      and there is nothing on which to ask whether later ones came cheaper.
      `recovery` would happily score those spans against a different, longer
      program and report a number.

    **The rebuild is always verified, and there are two ways to do it because
    there are two kinds of caller.** `composed.val_seed` owns the offset, so the
    scenes agree by construction today; the check is what keeps them agreeing
    after someone moves a default.

    - `programs` given -- the caller already holds the val split from
      `build_data` and the comparison costs a list comparison.
    - `programs` omitted -- verified against the record's own `corpus.val`
      digest, which is the same check for a caller that has no reason to build
      200,000 training scenes to look at 1,000 val ones. The digest *is* the
      corpus (`dm/data/fingerprint.py`), so this is not the weaker reading; it
      is the cheaper spelling of the identical one.
    """
    config = corpus_config(record)
    name = record.get("name", "?")
    if config.data != "composed":
        raise ValueError(
            f"{name} was trained on {config.data!r}; copy provenance exists "
            "only for the constructed corpus"
        )
    extra = dict(config.extra)
    if extra.pop("structured", False):
        raise ValueError(
            f"{name} was trained on the structured spelling, where REPEATX has "
            "already folded the copies away. `recovery` needs the flat trace -- "
            "train the arm without --structured"
        )
    scenes = composed.build(
        config.n_val, "valid", seed=composed.val_seed(config.data_seed),
        categories=config.categories, **extra,
    )
    flat = [s.flat for s in scenes]
    drifted = (flat != programs if programs is not None
               else digest(flat) != record.get("corpus", {}).get("val"))
    if drifted:
        raise ValueError(
            f"the rebuilt scenes for {name} are not the val split it was scored "
            "on; the corpus has drifted from the record"
        )
    return scenes


def codec_for(record: dict):
    """The codec whose alphabet this record's model was actually trained on.

    **`CODECS[name]` is today's table and a checkpoint carries yesterday's.**
    `TokenCodec` lays out `[specials][opcode slots][values]`, so the ISA v2
    migration moved `token` from 269 symbols to 274 and `token_typed` from 1,293
    to 1,554 (`dm/isa/spec.py`). A pre-v2 checkpoint scored through today's codec
    reads its own symbols as *different opcodes* -- and every instrument in
    `dm/eval/` reshapes logits against `codec.vocab_size`, a reshape that
    `docs/traps.md` records as one that "does not raise when the width is wrong;
    it reinterprets the tensor".

    So the width comes from the record and the codec is rebuilt to match, or the
    call fails. `byte` and `bit` are width-stable at any opcode table size, which
    is why no claim-1, claim-2 or claim-4 number moved in that migration; only
    the two token alphabets need reconstructing.

    Ambiguity is refused rather than guessed: a typed vocabulary is
    `2 + slots + 256 * kinds` and two unknowns can in principle admit two
    readings, so a width that does is an error and not a coin toss.
    """
    name = record["config"]["codec"]
    width = (record.get("model") or {}).get("vocab_size")
    base, _, view = name.partition("_delta")
    inner = CODECS[base]
    if width is not None and inner.vocab_size != width:
        inner = _rebuild(base, width)
    return RelativeCodec(inner) if view == "" and name.endswith("_delta") else inner


def _rebuild(name: str, width: int):
    if name not in ("token", "token_typed"):
        raise ValueError(
            f"{name} is {CODECS[name].vocab_size} symbols wide and the record "
            f"says {width}; only the token alphabets are reconstructible"
        )
    typed = name == "token_typed"
    solutions = []
    for kinds in range(1, len(Kind) + 1) if typed else (1,):
        slots = width - N_SPECIAL - 256 * kinds
        # The reserved opcode region has only ever *grown* -- `dm/isa/spec.py`
        # appends rows and never inserts, and the 11 -> 16 migration is the one
        # deliberate break -- so a historical width cannot exceed today's. That
        # bound is what makes the two unknowns solvable: without it a 1,293
        # symbol vocabulary admits five readings, four of which reserve hundreds
        # of slots for fourteen opcodes.
        if 1 <= slots <= OPCODE_SLOTS:
            candidate = TokenCodec(typed, opcode_slots=slots, n_kinds=kinds)
            if candidate.vocab_size == width:
                solutions.append(candidate)
    if len(solutions) != 1:
        raise ValueError(
            f"{len(solutions)} layouts give {name} a {width}-symbol vocabulary; "
            "the record does not determine which the checkpoint used"
        )
    return solutions[0]


def load(path: Path):
    """The model in eval mode and its record, for either kind of checkpoint."""
    record = json.loads(path.with_suffix(".json").read_text())
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if is_planner(record):
        model = StrokePlanner(PlannerConfig(**blob["cfg"]))
    else:
        model = DrawingLM(Config(**blob["cfg"]))
    model.load_state_dict(blob["state"])
    return model.eval(), record
