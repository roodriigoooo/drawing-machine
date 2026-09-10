"""F4 Adapter: build Direction 3's paired corpora and publish their census.

The corpus that trains the control arm is the one artifact a person has to read
before it is accepted. §6 F4 asks for that explicitly: a fingerprint is accepted
by someone looking at real donors, not by a checker confirming its own
arithmetic. So this prints the balance table and a hex sample of real splices,
and exits non-zero when any frozen invariant fails.

Thin by contract (`docs/directions.md` §7 invariant 13): the derangement, the
strata, the census and the acceptance rule all live in `dm.data.feedback`. This
file chooses a path and formats output.

    PYTHONPATH=. .venv/bin/python scripts/feedback_corpus.py --n-train 100000 \
        --out runs/feedback_corpus_v1_audited.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dm.data import feedback as corpus_module
from dm.eval.feedback_contract import seed_for


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-train", type=int, default=100_000)
    ap.add_argument("--n-val", type=int, default=1_000)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None,
                    help="write the frozen manifest here (runs/, so it is "
                         "identified by content hash rather than by filename)")
    ap.add_argument("--examples", type=int, default=8)
    args = ap.parse_args()

    paired = corpus_module.build(args.n_train, args.n_val,
                                 data_seed=args.data_seed,
                                 seed=seed_for("derangement"), flatten=True)
    report = corpus_module.balance_report(paired)
    body = corpus_module.manifest(paired)

    print(json.dumps({key: report[key] for key in
                      ("n_train", "n_val", "blocks", "strata", "census",
                       "exact", "intervention", "ordinals", "corpus")},
                     indent=1))
    print("\ndonor examples (review these before accepting the fingerprint):")
    for row in corpus_module.donor_examples(paired, limit=args.examples):
        print(f"  program {row['program']} ordinal {row['ordinal']} "
              f"@{row['span'][0]}:{row['span'][1]}")
        print(f"    own      {row['own_content']}")
        print(f"    implied  {row['implied_by_relation']}")
        print(f"    donor    {row['donor_content']}  (from program "
              f"{row['donor_program']})")

    print("\nfull-corpus VM fault/stroke census:")
    print(json.dumps(body["vm_census"], indent=1))

    if args.out is not None:
        corpus_module.write_manifest(args.out, body)
        # Both hashes, labelled. They identify different byte strings and
        # `PLAN.md` names calling one the other as a trap; printing one alone is
        # how that trap gets sprung.
        digests = corpus_module.manifest_digests(args.out)
        print(f"\nmanifest {args.out}")
        print(f"  canonical payload sha256  {digests['canonical_payload_sha256']}")
        print(f"  file sha256               {digests['file_sha256']}")

    if (not corpus_module.accepts(report)
            or not corpus_module.audit_accepts(body["vm_census"])):
        failed = [name for name, ok in report["exact"].items() if not ok]
        audit = ("VM census incomplete, faulted, or paired marginals differ"
                 if not corpus_module.audit_accepts(body["vm_census"])
                 else "intervention did not bite or its prevalence does not "
                      "reconcile")
        print(f"\nREFUSED: failing invariants {failed or audit}",
              file=sys.stderr)
        return 1
    print("\nevery frozen invariant holds and the intervention bit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
