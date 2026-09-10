"""A local page that talks to the board, on the standard library alone.

No framework, no build step, no `npm`. The page is one file this repository
generates, the transport is server-sent events, and the whole server is below.
That is not minimalism for its own sake: a demo whose dependency tree has to be
resolved again in six months is a demo that stops working in six months, and
this one has to survive being opened by someone reading an application.

**One drawing at a time.** The device is a single serial port and a single
mountable drive, so the draw path is serialised behind a lock. A second request
is refused rather than queued, because a queued request would flash the board
with a word whose asker has already navigated away.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import capture, page


class DemoServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, checkpoint, device, records: Path,
                 novelty=None, top_k=None, temperature=None):
        self.checkpoint = checkpoint
        self.device = device
        self.records = records
        self.novelty = novelty
        self.top_k = top_k
        self.temperature = temperature
        self.lock = threading.Lock()
        self.seeds: dict[str, int] = {}
        super().__init__(address, Handler)

    def next_seed(self, word: str, resample: bool) -> int:
        """A fresh seed per resample, a stable one per first ask.

        Asking for `cat` twice in a row should give the same cat -- the demo is
        deterministic and says so -- while "New sample" should not. The counter
        is per word so one class's resamples do not renumber another's.
        """
        key = word.strip().lower()
        if resample or key not in self.seeds:
            self.seeds[key] = self.seeds.get(key, -1) + 1
        return self.seeds[key]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):   # noqa: A003 - quieter than the default
        return

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            return self._html()
        if parsed.path == "/api/draw":
            return self._draw(parse_qs(parsed.query))
        self.send_error(404)

    def _html(self):
        html = page.render("live", checkpoint=self.server.checkpoint.as_dict(),
                           records=[]).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)

    def _event(self, blob: dict) -> None:
        self.wfile.write(b"data: " + json.dumps(blob).encode() + b"\n\n")
        self.wfile.flush()

    def _draw(self, query: dict):
        word = (query.get("word") or [""])[0]
        resample = bool(query.get("resample"))
        server = self.server

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        if not server.lock.acquire(blocking=False):
            self._event({"t": "error",
                         "message": "the board is already drawing; one at a time"})
            return
        try:
            seed = server.next_seed(word, resample)
            for event in capture.draw(server.checkpoint, server.device, word, seed,
                                      top_k=server.top_k,
                                      temperature=server.temperature,
                                      novelty=server.novelty):
                self._event(event)
                if event["t"] == "record":
                    path = capture.save(event["record"], server.records)
                    self._event({"t": "saved", "path": str(path)})
        except BrokenPipeError:
            return
        except Exception as failure:            # noqa: BLE001
            try:
                self._event({"t": "error", "message": f"{type(failure).__name__}: {failure}"})
            except BrokenPipeError:
                return
        finally:
            server.lock.release()
