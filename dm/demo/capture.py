"""Word in, record out — the one procedure both the CLI and the server run.

Two entry points would be two chances for the live page and the captured
gallery to disagree about what happened, and the whole design rests on them
being views of one artifact. So this yields the events a live page consumes and
*returns* the record a gallery replays, from the same pass over the same wire.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from .align import align, verify
from .device import DeviceRefusal
from .record import DemoRecord
from .router import route
from .sample import Checkpoint, sample


def view(record: DemoRecord) -> dict:
    """The record as the page consumes it.

    `on_silicon` is computed here, in Python, and the page reads it as a
    boolean. There is deliberately no path by which the page can derive the
    badge from anything else -- see `dm/demo/page.py`.
    """
    blob = record.as_dict()
    blob["on_silicon"] = record.on_silicon
    return blob


def draw(checkpoint: Checkpoint, device, word: str, seed: int,
         top_k: int | None = None, temperature: float | None = None,
         novelty=None, prompt: bytes | None = None,
         prefix_meta: dict | None = None,
         note: str = "") -> Iterator[dict]:
    """Sample, execute, parse, describe. Yields page events; last one is the record.

    The final event is `{"t": "record", "record": ...}` on success or
    `{"t": "error", "message": ...}` on a refusal. A refusal is a first-class
    outcome: `docs/claim4-bringup.md` §11.1 requires that a mismatch, fault,
    incomplete frame or failed BOOTSEL return refuses the page rather than
    falling back to host geometry, so the caller gets no record at all rather
    than a record with a caveat.
    """
    routing = route(word, checkpoint.classes)
    yield {"t": "routing", **routing.as_dict()}
    if not routing.routed:
        yield {"t": "error", "message": routing.reason}
        return

    kwargs = {}
    if top_k is not None:
        kwargs["top_k"] = top_k
    if temperature is not None:
        kwargs["temperature"] = temperature
    sampled = sample(checkpoint, routing.index, seed, prompt=prompt, **kwargs)
    yield {"t": "sampled", **sampled.as_dict()}

    if not sampled.program:
        yield {"t": "error", "message": "the model emitted an empty program"}
        return

    device_blob: dict = {}
    try:
        for event in device.draw(sampled.program):
            blob = event.as_dict()
            if blob["t"] == "done":
                device_blob = {k: v for k, v in blob.items() if k != "t"}
            yield blob
    except DeviceRefusal as failure:
        yield {"t": "error", "message": str(failure)}
        return

    if not device_blob:
        yield {"t": "error", "message": "the device produced no complete trace record"}
        return
    if not device_blob.get("matches_reference"):
        yield {"t": "error",
               "message": ("the device trace does not equal the reference VM's. "
                           "This page refuses to draw a frame it cannot verify.")}
        return

    checked = verify(align(sampled.program), device_blob["geometry"]["strokes"])
    device_blob["alignment"] = checked.as_dict() if checked else None

    record = DemoRecord(
        word=word,
        routing=routing.as_dict(),
        checkpoint=checkpoint.as_dict(),
        sampled=sampled.as_dict(),
        device=device_blob,
        source=device.source,
        prefix=prefix_meta,
        note=note,
    )
    if novelty is not None:
        record.novelty = novelty.nearest(sampled.program, routing.index)
    yield {"t": "record", "record": view(record)}


def save(record_blob: dict, directory: Path) -> Path:
    """Persist a record the page has already been given."""
    blob = dict(record_blob)
    blob.pop("on_silicon", None)
    return DemoRecord.from_dict(blob).save(directory)
