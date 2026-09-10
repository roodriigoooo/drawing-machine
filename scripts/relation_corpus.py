"""R1 Adapter: build, audit and freeze Direction 4's traced corpus.

Thin by contract. `dm.data.relation` owns the trace, the splits, the candidate
enumeration and every acceptance clause; this file reads arguments, calls it, and
prints what it found.

    # a small build, audited and printed, nothing written
    PYTHONPATH=. .venv/bin/python scripts/relation_corpus.py --smoke

    # the frozen build
    PYTHONPATH=. .venv/bin/python scripts/relation_corpus.py \\
        --out runs/relation_corpus_v1.json

    # fast, strict structural and cross-field re-validation, no rebuild
    PYTHONPATH=. .venv/bin/python scripts/relation_corpus.py \\
        --audit runs/relation_corpus_v1.json

    # exhaustive: the audit, then a deterministic rebuild from the manifest's
    # own configuration, requiring canonical payload equality
    PYTHONPATH=. .venv/bin/python scripts/relation_corpus.py \\
        --rebuild-verify runs/relation_corpus_v1.json

**A failed acceptance is a non-zero exit, always.** The manifest is still written
so the failure can be read, and `load_manifest` refuses it for anything that
trains -- a corpus that did not pass its own clauses must not be reachable by
accident (`docs/copy-relation.md` §7 invariant 1).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dm.data import relation as corpus_module
from dm.eval import relation_contract as contract

#: A build small enough to run in a minute and large enough to exercise every
#: stratum, venue and control. It cannot pass the component floors, and that is
#: correct: a smoke build is not a corpus and must never be freezable.
#:
#: It also carries the **development** corpus seed, so its manifest cannot be
#: mistaken for the scientific one even if someone points a training stage at it:
#: `contract.corpus_seed_faults` compares the two by name.
SMOKE = corpus_module.BuildConfig(
    n_train_synthetic=400, n_train_composed=800, n_eval=80, n_generic=80,
    train_motifs=200, eval_motifs=120, composed_limit=512,
    data_seed=contract.corpus_seed("development"), provenance="development")

#: The frozen build, at the seed the contract derives. Not the dataclass default:
#: a builder whose seed is a literal and a protocol whose seed is derived are two
#: numbers that were never compared, and the frozen manifest recorded one while
#: the protocol recorded the other.
FROZEN = corpus_module.BuildConfig(
    data_seed=contract.corpus_seed("scientific"), provenance="scientific")


def _print_report(report: dict) -> None:
    coverage = report["coverage"]
    print(f"coverage   {coverage['covered']}/{coverage['positives']} positives "
          f"({coverage['coverage']:.4f}); widest source "
          f"{coverage['widest_source_bytes']}B / "
          f"{coverage['widest_source_instructions']} instructions "
          f"(caps {coverage['max_source_bytes']}B / "
          f"{coverage['max_source_instructions']})")
    print(f"derivations {coverage['single_derivation']} single, "
          f"{coverage['multiple_derivations']} with an equivalent schedule")
    for stratum, found in sorted(report["strata"].items()):
        print(f"  {stratum:<28} {found['cases']:>7} cases  "
              f"{found['components']:>6} components  "
              f"{found['positives']:>7} positives")
    destroyed = report["destroyed"]
    print(f"destroyed  {destroyed['accepted']}/{destroyed['offered']} spliced; "
          f"byte census matches source: {destroyed['byte_census_matches_source']}; "
          f"residual relations: {destroyed['residual_relations']}")
    print(f"pairs      {report['split']['held_out_pairs']} withheld motif pairings")


def _build(config: corpus_module.BuildConfig, out: Path | None) -> int:
    contract.check_consistency()
    print(f"data_seed   {config.data_seed}")
    corpus = corpus_module.build(config)
    body = corpus_module.manifest(corpus)
    _print_report(body["acceptance"])
    if out is not None:
        corpus_module.write_manifest(out, body)
        digests = corpus_module.manifest_digests(out)
        print(f"manifest   {out}")
        print(f"  payload  {digests['canonical_payload_sha256']}")
        print(f"  file     {digests['file_sha256']}")
    if not body["accepted"]:
        for problem in body["problems"]:
            print(f"REFUSED: {problem}", file=sys.stderr)
        return 1
    print("accepted")
    return 0


def _summarise(body: object) -> None:
    """Print what can be printed, without deciding anything.

    Deliberately tolerant: a manifest is summarised so its failure can be *read*,
    and a body malformed enough to crash the printer is exactly the body a
    reviewer most wants a summary of. The verdict is `_audit`'s and is taken from
    `load_manifest`, never from what printed successfully here.
    """
    try:
        _print_report(body["acceptance"])
    except (KeyError, TypeError, IndexError, ValueError, OverflowError):
        print("the acceptance report is malformed and cannot be summarised",
              file=sys.stderr)


def _audit(path: Path, *, provenance: str, rebuild: bool) -> int:
    """Re-validate an existing manifest. Non-zero on **every** failure.

    Two modes, because they cost three orders of magnitude apart and conflating
    them would make the cheap one never run. `--audit` is structural and
    cross-field: it reads the file, checks both digests, and hands the body to
    `load_manifest`, which validates the envelope, the case index, every
    acceptance section, and every acceptance fact the index can independently
    produce. `--rebuild-verify` adds the exhaustive route -- rebuild the corpus
    from the manifest's own configuration, regenerate the manifest and require
    canonical payload equality -- which is the only thing that covers the
    evidence the index cannot reconstruct (coverage, residual relations, the
    island map, fingerprints and every VM figure).

    An earlier version of this function read the JSON itself and compared the two
    digests. That accepted any rehashed body: the acceptance report was believed
    on its own authority, and a rewritten report with a matching digest passed.
    """
    try:
        body = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        print(f"REFUSED: {path} is not readable JSON: {error}", file=sys.stderr)
        return 1
    _summarise(body)
    digests = corpus_module.manifest_digests(path)
    print(f"payload    {digests['canonical_payload_sha256']}")
    print(f"file       {digests['file_sha256']}")

    failures: list[str] = []
    try:
        corpus_module.load_manifest(path)
    except corpus_module.ManifestRefused as error:
        # **`ManifestRefused` only.** Narrowing to builtin `ValueError` was not
        # enough: validation code raises that class too, so a five-thousand-digit
        # histogram key that tripped CPython's integer-conversion limit was
        # printed here as an ordinary refusal while an `OverflowError` two fields
        # away crashed. Classification has to follow the contract boundary, not
        # whichever operation happened to fail, so anything that is not a refusal
        # now propagates and stays loud.
        failures.append(str(error))
    if isinstance(body, dict):
        failures.extend(contract.corpus_seed_faults(body, provenance=provenance))
    if rebuild:
        if failures:
            failures.append(
                "the rebuild was not attempted: a manifest that fails its "
                "structural audit describes no corpus to rebuild")
        else:
            print(f"rebuilding at data_seed="
                  f"{body['acceptance']['config']['data_seed']} ...")
            failures.extend(corpus_module.rebuild_verify(body))
    for failure in failures:
        print(f"REFUSED: {failure}", file=sys.stderr)
    if failures:
        return 1
    print("verified" if rebuild else "accepted")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None,
                        help="write the manifest here")
    parser.add_argument("--audit", type=Path, default=None,
                        help="strictly re-validate an existing manifest")
    parser.add_argument("--rebuild-verify", type=Path, default=None,
                        help="audit a manifest and then rebuild its corpus from "
                             "its own configuration, requiring canonical payload "
                             "equality; exhaustive and slow")
    parser.add_argument("--provenance", default="scientific",
                        choices=sorted(corpus_module.CORPUS_PROVENANCES),
                        help="which corpus seed an audited manifest must carry")
    parser.add_argument("--smoke", action="store_true",
                        help="build the small shape instead of the frozen one")
    parser.add_argument("--data-seed", type=int, default=None)
    args = parser.parse_args()

    if args.audit is not None and args.rebuild_verify is not None:
        parser.error("pass --audit or --rebuild-verify, not both")
    target = args.audit or args.rebuild_verify
    if target is not None:
        return _audit(target, provenance=args.provenance,
                      rebuild=args.rebuild_verify is not None)
    config = SMOKE if args.smoke else FROZEN
    if args.data_seed is not None:
        config = corpus_module.BuildConfig(
            **{**vars(config), "data_seed": args.data_seed})
    return _build(config, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
