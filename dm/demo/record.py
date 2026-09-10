"""One drawing, completely described, in one file.

Every surface the demo has -- the live page, the static gallery, the figures,
the videos -- is a view of this record and of nothing else. That is the design
decision the whole demo rests on, and it exists for a reason the rest of this
project keeps paying to relearn: **hardware is available for hours and a
portfolio is read for years.** A record captured once on a bench replays
byte-identically after the boards are in a drawer, and a screen recording made
from the replay shows the same geometry the board produced rather than a
re-creation of it.

It also makes the honesty contract checkable instead of stated. The record
carries `source` (`rp2040` or `native`), the raw UART text, the device geometry
in the fixed-point units the wire used, and the reference-equality verdict.
A page can therefore be *tested* for the property `docs/claim4-bringup.md` §11.1
demands -- that host geometry never reaches the canvas -- because a record whose
source is not `rp2040` is machine-distinguishable from one that is.

Records are written under a tracked artifact directory rather than `runs/`,
which the repository ignores wholesale. `docs/claim4-bringup.md` §11.3 already
owed that move for the cycle measurement; a demo whose evidence is untracked
would repeat the same fault the day it was built.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

#: Bumped when a field's meaning changes rather than when one is added, so a
#: gallery built from mixed captures either works or refuses, and never
#: averages two regimes -- the rule `dm/train.py` records for run records.
SCHEMA = 1


@dataclass
class DemoRecord:
    word: str
    routing: dict
    checkpoint: dict
    sampled: dict
    device: dict
    #: `rp2040` for silicon, `native` for the hardware-free rehearsal.
    source: str
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    schema: int = SCHEMA
    host: str = field(default_factory=lambda: f"{platform.system()} {platform.machine()}")
    #: Filled by the offline novelty pass; absent means not yet measured, which
    #: the page states rather than rendering as a pass.
    novelty: dict | None = None
    #: Present only for the held-out-prefix panel.
    prefix: dict | None = None
    note: str = ""

    # -- identity -----------------------------------------------------------

    @property
    def bytecode(self) -> bytes:
        return bytes.fromhex(self.sampled["bytecode_hex"])

    @property
    def on_silicon(self) -> bool:
        """The one predicate the exact-match badge is allowed to depend on.

        Both halves are required. A device frame that diverged from the
        reference is a finding, not a picture to badge, and a native rehearsal
        is a picture with no device behind it at all.
        """
        return self.source == "rp2040" and bool(self.device.get("matches_reference"))

    @property
    def stem(self) -> str:
        """A filename that is stable under recapture and unique under variation."""
        digest = hashlib.sha256(
            f"{self.checkpoint['name']}|{self.sampled['class_index']}|"
            f"{self.sampled['seed']}|{self.sampled['top_k']}|"
            f"{self.sampled['temperature']}|{self.sampled['bytecode_hex']}|"
            f"{self.prefix['source'] if self.prefix else ''}".encode()
        ).hexdigest()[:10]
        klass = self.routing.get("class") or "unrouted"
        return f"{self.source}_{klass}_{digest}"

    # -- persistence --------------------------------------------------------

    def as_dict(self) -> dict:
        return {
            "schema": self.schema,
            "created": self.created,
            "host": self.host,
            "source": self.source,
            "word": self.word,
            "routing": self.routing,
            "checkpoint": self.checkpoint,
            "sampled": self.sampled,
            "device": self.device,
            "novelty": self.novelty,
            "prefix": self.prefix,
            "note": self.note,
        }

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.stem}.json"
        path.write_text(json.dumps(self.as_dict(), indent=1) + "\n")
        return path

    @classmethod
    def from_dict(cls, blob: dict) -> "DemoRecord":
        if blob.get("schema") != SCHEMA:
            raise ValueError(
                f"record schema {blob.get('schema')} is not {SCHEMA}; recapture it "
                "rather than reading two regimes into one gallery"
            )
        return cls(
            word=blob["word"], routing=blob["routing"], checkpoint=blob["checkpoint"],
            sampled=blob["sampled"], device=blob["device"], source=blob["source"],
            created=blob["created"], schema=blob["schema"], host=blob["host"],
            novelty=blob.get("novelty"), prefix=blob.get("prefix"),
            note=blob.get("note", ""),
        )

    @classmethod
    def load(cls, path: Path) -> "DemoRecord":
        return cls.from_dict(json.loads(path.read_text()))


def load_all(directory: Path) -> list[DemoRecord]:
    """Every record in `directory`, oldest first.

    Sorted by capture time rather than filename so a gallery reads in the order
    the bench produced it, which is the order a viewer watching the recording
    saw.
    """
    records = [DemoRecord.load(p) for p in sorted(directory.glob("*.json"))]
    return sorted(records, key=lambda r: r.created)
