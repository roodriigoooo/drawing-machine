"""Archive Adapter: a committed hash manifest for artifacts `runs/` cannot carry.

`runs/` is ignored, so every Direction 3 checkpoint, training record and cell
report lives outside the repository. Content hashes inside those files help, but
they are hashes an ignored artifact wrote about itself: nothing in any committed
revision says which bytes the F5b stop was taken over, and a deleted artifact
leaves no trace at all -- which already happened once, to the first
`terminal_mix_v1 / seed 100` cell (`docs/feedback-stability.md` §3d).

This writes the missing half: one committed file naming every Direction 3
artifact by size and SHA-256, at the revision that produced it. It does not copy,
rewrite or re-verify any report. An auditor holding the repository and the
artifacts can then check that the two describe each other; an auditor holding
only the repository can at least see exactly what was claimed to exist.

Thin by contract (`docs/directions.md` §7 invariant 15): it hashes files and
sorts them. Every rule about what those files mean lives in the Modules.

    PYTHONPATH=. .venv/bin/python scripts/feedback_archive.py \\
        --out docs/feedback-artifacts.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dm.eval.provenance import canonical_digest, sha256_file

#: What an archived filename means, matched by suffix on the stem.  Descriptive
#: only: the manifest records bytes, and a label that guessed wrong would be
#: visible next to the hash rather than able to change it.
KINDS: tuple[tuple[str, str], ...] = (
    ("_cell.json", "qualification cell report"),
    ("_report.json", "engineering smoke report"),
    ("_audited.json", "audited paired corpus manifest"),
    ("qualification.json", "qualification decision"),
    ("correction.json", "correction-package decision"),
    (".pt", "checkpoint"),
    (".json", "training record"),
    (".log", "console log"),
)


def kind_of(path: Path) -> str:
    for suffix, label in KINDS:
        if path.name.endswith(suffix):
            return label
    return "artifact"


def source_reconciliation(report: Path) -> dict:
    """Which of a cell report's recorded source digests this tree still produces.

    A cell report names the exact bytes that trained and judged it. When those
    bytes were never committed -- which is the F5b situation this file exists to
    partly repair -- the honest statement is not "the code is archived" but
    "these files still reproduce their recorded digest and these do not". Stated
    per file, computed rather than asserted, so nobody has to take it on trust.
    """
    recorded = json.loads(report.read_text()).get("sources") or {}
    rows = []
    for name in sorted(recorded):
        path = Path(name)
        current = sha256_file(path) if path.is_file() else None
        rows.append({
            "path": name,
            "recorded_sha256": recorded[name],
            "tree_sha256": current,
            "reproduced": current == recorded[name],
        })
    return {
        "report": str(report),
        "note": "a file marked reproduced:false has changed since that cell ran; "
                "its recorded digest names bytes no committed revision holds, "
                "because the code was uncommitted when the cell was produced",
        "files": rows,
        "reproduced": sum(1 for row in rows if row["reproduced"]),
        "total": len(rows),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", type=Path,
                    help="artifacts to archive; defaults to runs/feedback_*")
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--out", type=Path,
                    default=Path("docs/feedback-artifacts.json"))
    ap.add_argument("--against", type=Path, default=None,
                    help="a cell report whose recorded source digests are "
                         "compared against this tree, file by file")
    args = ap.parse_args()

    paths = sorted(args.paths or args.runs.glob("feedback_*"))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        print(f"cannot archive {missing}: incomplete", file=sys.stderr)
        return 1
    if not paths:
        print(f"no artifacts under {args.runs}: incomplete", file=sys.stderr)
        return 1

    body = {
        "what": "SHA-256 of every Direction 3 artifact under the ignored runs/ "
                "directory, recorded in the repository because the artifacts "
                "themselves are not",
        "not": "a copy, a re-verification, or a substitute for the reports; a "
               "cell report still verifies against its own canonical payload "
               "digest and this file says nothing about whether it does",
        "artifacts": [
            {
                "path": str(path),
                "kind": kind_of(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in paths
        ],
    }
    if args.against is not None:
        body["source_reconciliation"] = source_reconciliation(args.against)
    body["archive_sha256"] = canonical_digest(body, "archive_sha256")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    staged = args.out.with_name(f".{args.out.name}.tmp")
    staged.write_text(json.dumps(body, indent=1, sort_keys=True) + "\n")
    staged.replace(args.out)
    print(f"archived {len(paths)} artifacts to {args.out}")
    print(f"archive sha256 {body['archive_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
