"""R1 Adapter: write or verify Direction 4's frozen contract artifact.

Thin by contract (`docs/copy-relation.md` §7 invariant 1): every number,
threshold, formula and seed lives in `dm.eval.relation_contract`, every corpus
rule in `dm.data.relation`, and this file only chooses which of the two things to
do and what to print.

    # verify the tree against the freeze (what CI and the test suite do)
    PYTHONPATH=. .venv/bin/python scripts/relation_contract.py --verify

    # write the protocol -- refuses to replace an existing one without
    # --refreeze, because rewriting a freeze after a later stage has started is
    # how a protocol comes to describe code nobody ran
    PYTHONPATH=. .venv/bin/python scripts/relation_contract.py --write

    # name the traced corpus the protocol freezes over
    PYTHONPATH=. .venv/bin/python scripts/relation_contract.py --write --refreeze \\
        --corpus runs/relation_corpus_v1.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dm.data import relation as corpus_module
from dm.eval import relation_contract as contract


def _corpus_digest(path: Path | None) -> str | None:
    """The manifest's canonical payload digest, or None when none is named.

    The *payload* digest and not the file digest, and the two are never
    interchangeable: reformat the JSON and the file hash moves while the payload
    hash does not. `PLAN.md` lists quoting one as the other as a trap.
    """
    if path is None:
        return None
    corpus_module.load_manifest(path)          # refuses an unfrozen or failed build
    return corpus_module.manifest_digests(path)["canonical_payload_sha256"]


def _write(path: Path, corpus: Path | None, refreeze: bool) -> int:
    contract.check_consistency()
    if path.exists() and not refreeze:
        print(
            f"{path} already exists. Rewriting a freeze after a later stage has "
            "started makes the protocol describe code that never ran. Pass "
            "--refreeze if it genuinely has to move, and bump PROTOCOL_SCHEMA "
            "when the payload shape changes.",
            file=sys.stderr,
        )
        return 1
    protocol = contract.protocol_dict(corpus_sha256=_corpus_digest(corpus))
    contract.write_protocol(path, protocol)
    table = protocol["parameters"]
    print(f"protocol   {path}  sha256 {protocol['protocol_sha256']}")
    print(f"corpus     {protocol['corpus']['canonical_payload_sha256']}")
    print(f"parameters none={table['none']['total']}  "
          f"span_affine_v1={table['span_affine_v1']['total']}  "
          f"cap={table['budget']['cap']}  headroom={table['budget']['headroom']}")
    print(f"cells      {', '.join(protocol['cells'])}")
    return 0


def _verify(path: Path) -> int:
    """Recompute the payload from today's code and compare it to the artifact.

    Not a digest check alone. `load_protocol` already refuses a body whose
    recorded hash does not match its own contents, which catches a hand edit; it
    does *not* catch the file staying self-consistent while the Module moved
    underneath it. So the payload is rebuilt and compared field by field, and the
    fields that differ are named.
    """
    contract.check_consistency()
    frozen = contract.load_protocol(path)
    rebuilt = contract.protocol_dict(
        corpus_sha256=(frozen.get("corpus") or {}).get("canonical_payload_sha256"))
    if rebuilt["protocol_sha256"] == frozen["protocol_sha256"]:
        print(f"protocol  {path}  sha256 {frozen['protocol_sha256']}  matches the tree")
        return 0
    moved = sorted(key for key in set(rebuilt) | set(frozen)
                   if key != "protocol_sha256" and rebuilt.get(key) != frozen.get(key))
    print(f"{path} no longer describes the tree; sections that moved: {moved}",
          file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true",
                        help="serialise the contract to the protocol path")
    action.add_argument("--verify", action="store_true",
                        help="check the frozen protocol against today's code")
    parser.add_argument("--protocol", type=Path, default=contract.PROTOCOL_PATH)
    parser.add_argument("--corpus", type=Path, default=None,
                        help="traced-corpus manifest the protocol freezes over")
    parser.add_argument("--refreeze", action="store_true",
                        help="replace an existing protocol; never routine")
    args = parser.parse_args()
    if args.write:
        return _write(args.protocol, args.corpus, args.refreeze)
    return _verify(args.protocol)


if __name__ == "__main__":
    raise SystemExit(main())
