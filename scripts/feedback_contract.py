"""F0 Adapter: write or verify Direction 3's frozen contract artifacts.

Two artifacts, one freeze: the baseline CPU fixture
(`tests/fixtures/feedback_baseline_v1.json`) and the protocol that hashes it
(`docs/feedback-protocol-v0.json`).  They are written together because the
protocol records the fixture's digest, so writing one without the other leaves a
protocol pointing at bytes that no longer exist.

Thin by contract (`docs/directions.md` §7 invariant 13): every number, threshold
and formula lives in `dm.eval.feedback_contract`, every tensor in
`dm.eval.feedback_fixture`, and this file only chooses which of the two things to
do and what to print.

    # verify the tree against the freeze (what CI and the test suite do)
    PYTHONPATH=. .venv/bin/python scripts/feedback_contract.py --verify

    # write both artifacts -- refuses to replace an existing fixture without
    # --refreeze, because recapturing after F1 has started would photograph the
    # changed model and make the equivalence claim circular
    PYTHONPATH=. .venv/bin/python scripts/feedback_contract.py --write

    # F5, after the smoke passes: layer the training freeze over v0
    PYTHONPATH=. .venv/bin/python scripts/feedback_contract.py --freeze-training \\
        --corpus runs/feedback_corpus_v1_audited.json \
        --smoke runs/feedback_smoke_report.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import torch

from dm.data import feedback as corpus_module
from dm.eval import feedback as evaluator
from dm.eval import feedback_contract as contract
from dm.eval import feedback_fixture as fixture
from dm.eval.feedback import ENGINEERING
from dm.eval.provenance import sha256_file, source_hashes


def _write(fixture_path: Path, protocol_path: Path, refreeze: bool) -> int:
    contract.check_consistency()
    if fixture_path.exists() and not refreeze:
        # Recapturing after F1 has started would photograph the *changed* model
        # and hand the equivalence claim its own conclusion. The freeze is one
        # command away from being destroyed by a reflex, so it takes a flag.
        print(
            f"{fixture_path} already exists. Rewriting it after F1 has started "
            "recaptures the baseline from changed code and makes the equivalence "
            "claim circular. Pass --refreeze if the fixture genuinely has to move, "
            "and bump FIXTURE_SCHEMA when it does.",
            file=sys.stderr,
        )
        return 1
    body = fixture.fixture_dict()
    fixture.write_fixture(fixture_path, body)
    protocol = contract.protocol_dict(fixture_sha256=body["fixture_sha256"])
    contract.write_protocol(protocol_path, protocol)
    print(f"fixture   {fixture_path}  sha256 {body['fixture_sha256']}")
    print(f"protocol  {protocol_path}  sha256 {protocol['protocol_sha256']}")
    print(f"cells     {', '.join(cell['name'] for cell in body['cells'])}")
    return 0


def _verify(fixture_path: Path, protocol_path: Path) -> int:
    contract.check_consistency()
    body = fixture.load_fixture(fixture_path)
    protocol = contract.load_protocol(protocol_path)

    recorded = protocol["baseline_fixture"]["sha256"]
    if recorded != body["fixture_sha256"]:
        print(f"protocol names fixture {recorded}, file is {body['fixture_sha256']}",
              file=sys.stderr)
        return 1

    expected = contract.protocol_dict(fixture_sha256=body["fixture_sha256"])
    if expected["protocol_sha256"] != protocol["protocol_sha256"]:
        print("the frozen protocol no longer serialises dm.eval.feedback_contract; "
              "one of the two changed without the other", file=sys.stderr)
        return 1

    verdicts = fixture.verify_fixture(body)
    if any(not verdict.ok for verdict in verdicts):
        print(fixture.describe_failure(body, verdicts), file=sys.stderr)
        return 1

    print(f"protocol  {protocol['protocol']}  sha256 {protocol['protocol_sha256']}")
    print(f"fixture   {len(verdicts)} cells reproduce bit-for-bit")
    for verdict in verdicts:
        print(f"  ok  {verdict.name}")
    return 0


def _artifact_path(value: object, *, what: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{what} has no artifact path: incomplete")
    return Path(value)


def _smoke_problems(smoke: dict, smoke_path: Path) -> list[str]:
    """Return every reason an engineering smoke cannot freeze training.

    The report is an index, not an authority.  Hashes and completeness are
    checked against the trainer's separate files so a report cannot claim a
    record or checkpoint that it has overwritten or that no longer exists.
    """
    problems: list[str] = []
    if smoke.get("provenance") != ENGINEERING:
        problems.append("provenance is not engineering")
    if smoke.get("decision_value") != "none":
        problems.append("engineering smoke must have decision_value='none'")
    if smoke.get("status") != "complete":
        problems.append(f"status={smoke.get('status')!r}, not complete")
    if smoke.get("smoke_passed") is not True:
        problems.append("smoke_passed is not true")
    if smoke.get("failures"):
        problems.append(f"smoke failures={smoke['failures']!r}")
    if ((smoke.get("config") or {}).get("feedback_schema")
            != contract.QUALIFICATION_FEEDBACK_SCHEMA):
        problems.append("smoke did not exercise the source-faithful feedback arm")

    stages = smoke.get("stages")
    if not isinstance(stages, dict):
        return problems + ["stages is missing: incomplete"]
    training = stages.get("training")
    reload = stages.get("reload")
    scoring = stages.get("scoring")
    guards = stages.get("generic_guards")
    stability = stages.get("stability")
    if not isinstance(training, dict):
        return problems + ["training stage is missing: incomplete"]
    if not isinstance(reload, dict):
        problems.append("reload stage is missing")
    if not isinstance(scoring, dict):
        problems.append("scoring stage is missing")
    if not isinstance(guards, dict):
        problems.append("generic guard stage is missing")
    if not isinstance(stability, dict):
        problems.append("stability stage is missing")

    if training.get("complete") is not True:
        problems.append("training record is not complete")
    if training.get("missing_record_fields"):
        problems.append("training record has missing fields")
    if training.get("record_history_length", 0) <= 0:
        problems.append("training record has no history")
    if training.get("record_val_bits_length", 0) <= 0:
        problems.append("training record has no final val_bits")
    if training.get("record_best_val_bits_length", 0) <= 0:
        problems.append("training record has no best_val_bits")
    if training.get("feedback_schema") != contract.QUALIFICATION_FEEDBACK_SCHEMA:
        problems.append("training record used the wrong feedback architecture arm")

    try:
        record_path = _artifact_path(training.get("record_path"),
                                     what="training record")
        checkpoint_path = _artifact_path(training.get("checkpoint_path"),
                                         what="checkpoint")
    except ValueError as exc:
        problems.append(str(exc))
        return problems
    if record_path.resolve() == smoke_path.resolve():
        problems.append("smoke report overwrote the training record")
    for path, expected, label in (
        (record_path, training.get("record_sha256"), "training record"),
        (checkpoint_path, training.get("checkpoint_sha256"), "checkpoint"),
    ):
        if not path.exists():
            problems.append(f"{label} does not exist")
            continue
        if not isinstance(expected, str) or sha256_file(path) != expected:
            problems.append(f"{label} hash does not match the smoke report")

    if record_path.exists():
        try:
            record = json.loads(record_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"training record cannot be read: {exc}")
        else:
            if record.get("provenance") != ENGINEERING:
                problems.append("training record is missing engineering provenance")
            if ((record.get("config") or {}).get("feedback_schema")
                    != contract.QUALIFICATION_FEEDBACK_SCHEMA):
                problems.append("training record config used the wrong feedback arm")
            if record.get("complete") is not True:
                problems.append("training record file is not complete")
            if contract.missing_record_fields(record):
                problems.append("training record file is incomplete")
            if not record.get("history") or not record.get("val_bits"):
                problems.append("training record file lost trajectory or val_bits")
            if not record.get("best_val_bits") or not record.get("best"):
                problems.append("training record file lost best trajectory")
            if (record.get("checkpoint_sha256")
                    != training.get("checkpoint_sha256")):
                problems.append("training record does not link this checkpoint")

    if checkpoint_path.exists():
        try:
            weights = torch.load(checkpoint_path, weights_only=False)
        except (EOFError, IndexError, KeyError, OSError, TypeError, ValueError,
                pickle.UnpicklingError, RuntimeError) as exc:
            # Corrupt or non-checkpoint artifacts are incomplete. Keep this
            # boundary explicit rather than allowing a malformed file to look
            # like a missing provenance field.
            problems.append(f"checkpoint cannot be read: {exc}")
        else:
            if weights.get("provenance") != ENGINEERING:
                problems.append("checkpoint is missing engineering provenance")
            if ((weights.get("cfg") or {}).get("feedback_schema")
                    != contract.QUALIFICATION_FEEDBACK_SCHEMA):
                problems.append("checkpoint used the wrong feedback architecture arm")

    if not isinstance(reload, dict) or reload.get("strict") is not True:
        problems.append("strict reload was not recorded")
    if not isinstance(reload, dict) or reload.get("params_match_config") is not True:
        problems.append("reloaded parameters do not match config")
    if not isinstance(reload, dict) or reload.get("provenance") != ENGINEERING:
        problems.append("reloaded checkpoint provenance is not engineering")
    if (not isinstance(reload, dict)
            or reload.get("feedback_schema") != contract.QUALIFICATION_FEEDBACK_SCHEMA):
        problems.append("strict reload used the wrong feedback architecture arm")
    if (not isinstance(scoring, dict)
            or not scoring.get("equivalence", {}).get("matches", False)):
        problems.append("sequential equivalence did not pass")
    if not isinstance(scoring, dict) or scoring.get("first_symbol_identical") is not True:
        problems.append("first-symbol identity did not pass")
    if not isinstance(guards, dict) or guards.get("passed") is not True:
        problems.append("generic guards did not pass")
    if (not isinstance(stability, dict)
            or not stability.get("smoke", {}).get("passed", False)):
        problems.append("smoke stability clauses did not pass")
    if isinstance(guards, dict) and isinstance(stability, dict):
        guard_loss = (guards.get("values") or {}).get("valid_halt_loss")
        stability_report = stability.get("report")
        if not isinstance(stability_report, dict):
            problems.append("stability report is missing: incomplete")
        elif stability_report.get("valid_halt_loss") != guard_loss:
            problems.append("stability report did not apply the valid-halt guard")
    return problems


def _reconstruct_strict(checkpoint_path: Path) -> dict:
    """Rebuild the model from the checkpoint's own config and load it strict.

    Freeze-side, in this process, from the bytes on disk. A qualification report
    carries a `reload.strict` boolean, and a boolean an artifact wrote about
    itself is a claim: `docs/directions.md` §7 invariant 11 says a recorded
    value never replaces the recomputation. The check that matters is precisely
    that today's `Config` and `DrawingLM` still accept yesterday's state
    dictionary under `strict=True` -- which is the load F7 and F8 will do, and
    the one a renamed tensor breaks.
    """
    from dm.models.transformer import Config, DrawingLM

    weights = torch.load(checkpoint_path, weights_only=False)
    model = DrawingLM(Config(**weights["cfg"]))
    model.load_state_dict(weights["state"], strict=True)
    return {
        "path": str(checkpoint_path),
        "strict": True,
        "feedback_schema": weights["cfg"]["feedback_schema"],
        "params": model.n_params(),
        "params_match_config": model.n_params() == model.cfg.n_params(),
        "provenance": weights.get("provenance"),
        "state_keys": sorted(weights["state"]),
    }


def _qualification_problems(result: dict, path: Path) -> list[str]:
    """Why a qualification result cannot authorize protocol v1.

    Every clause here is a refusal, not a judgement. The gate itself already ran
    in `dm.eval.feedback_qualification`; what this adds is that the artifact in
    front of the freeze is the *kind* of artifact the freeze is allowed to read.
    """
    problems: list[str] = []
    if result.get("schema") != contract.QUALIFICATION_SCHEMA:
        problems.append(
            f"qualification schema {result.get('schema')!r}, expected "
            f"{contract.QUALIFICATION_SCHEMA}"
        )
    # Which package the result belongs to decides which seeds and which order are
    # the right ones. Read from the artifact rather than assumed: a freeze that
    # checked F5b's two seeds against an F5c result would pass a package it had
    # never inspected, and a freeze that checked F5c's order against F5b's would
    # refuse one it had.
    declared = result.get("package")
    name = declared.get("package") if isinstance(declared, dict) else None
    known = {entry.name: entry for entry in contract.PACKAGES.values()}
    if name is not None and name not in known:
        problems.append(f"qualification names unknown package {name!r}")
        return problems
    qualified = known.get(name, contract.F5B_PACKAGE)
    correction = qualified.name == contract.CORRECTION_PACKAGE
    expected_order = ([contract.CORRECTION_SCHEDULE] if correction
                      else list(contract.ELIGIBLE_SCHEDULES))
    if result.get("order") != expected_order:
        problems.append("qualification did not use the predeclared eligible order")
    if correction:
        # The clause the correction package exists to add. A qualified label
        # without a reproduced checkpoint is not a result about the mechanism,
        # so the freeze refuses it rather than reading the label alone.
        reproducibility = result.get("reproducibility")
        if not isinstance(reproducibility, dict):
            problems.append(
                f"{path} is a correction result with no reproducibility verdict: "
                "incomplete"
            )
        elif reproducibility.get("reproduced") is not True:
            problems.append(
                f"the repeated seed did not reproduce its checkpoint bytes: "
                f"{reproducibility.get('checkpoint_sha256')}"
            )
    frozen = result.get("frozen_schedule")
    if not frozen:
        problems.append(
            f"no eligible schedule passed both seeds: label={result.get('label')!r}"
        )
        return problems
    if frozen not in contract.ELIGIBLE_SCHEDULES:
        problems.append(f"{frozen!r} is a diagnostic schedule and cannot freeze v1")
    entry = (result.get("schedules") or {}).get(frozen)
    if not isinstance(entry, dict):
        problems.append(f"{frozen!r} has no verdict in {path}: incomplete")
        return problems
    if entry.get("passed") is not True or entry.get("failures"):
        problems.append(f"{frozen!r} is named frozen but did not pass")
    cells = entry.get("cells") or []
    seeds = sorted(cell.get("seed") for cell in cells)
    required_seeds = (sorted(seed for seed, count
                             in contract.CORRECTION_REPLICATES.items()
                             for _ in range(count)) if correction
                      else sorted(qualified.seeds))
    if seeds != required_seeds:
        problems.append(
            f"{frozen!r} carries seeds {seeds}, not {required_seeds}: incomplete"
        )
    for cell in cells:
        if cell.get("passed") is not True:
            problems.append(f"{frozen!r} seed {cell.get('seed')} did not pass")
        if cell.get("eligible") is not True:
            problems.append(f"{frozen!r} seed {cell.get('seed')} is not eligible")
    # The envelope half of the hash chain. `qualify_cell` checks the record,
    # checkpoint, protocol, corpus and source digests; a cell report cannot carry
    # a hash of itself, so `decide` verifies that separately and the freeze
    # requires the verification to be present for the schedule it is freezing.
    verified = {
        (row.get("schedule"), row.get("seed"), row.get("replicate"))
        for row in (result.get("cell_reports") or [])
        if row.get("verified") is True
        and row.get("recorded_report_sha256") == row.get("recomputed_report_sha256")
    }
    for cell in cells:
        key = (frozen, cell.get("seed"), cell.get("replicate"))
        if key not in verified:
            problems.append(
                f"{frozen!r} seed {cell.get('seed')} replicate "
                f"{cell.get('replicate')} has no verified cell-report digest: "
                f"incomplete"
            )
    return problems


def _freeze_training(protocol_path: Path, v1_path: Path, corpus_path: Path,
                     smoke_path: Path, qualification_path: Path) -> int:
    """Layer protocol v1 over v0: source digests, corpus and per-cell configs.

    Runs only after the F5 smoke has passed, because that is the first moment the
    source files it freezes are known to work together. Refuses a smoke report
    that is not marked `engineering`: if the run that exercised the code claimed
    to be evidence, the freeze would be recording the wrong kind of artifact.
    """
    missing = [str(path) for path in (corpus_path, smoke_path, qualification_path)
               if not path.exists()]
    if missing:
        # Missing provenance is `incomplete`, and it has to read as `incomplete`:
        # a traceback is a bug report, and this is a refusal
        # (`docs/directions.md` §7 invariant 11).
        print(f"cannot freeze without {missing}: incomplete", file=sys.stderr)
        return 1
    base = contract.load_protocol(protocol_path)
    corpus = corpus_module.load_manifest(corpus_path)
    digests = corpus_module.manifest_digests(corpus_path)
    try:
        smoke = json.loads(smoke_path.read_text())
        qualification = json.loads(qualification_path.read_text())
    except json.JSONDecodeError as exc:
        print(f"a freeze input is not readable JSON: {exc}: incomplete",
              file=sys.stderr)
        return 1
    problems = _smoke_problems(smoke, smoke_path)
    problems += _qualification_problems(qualification, qualification_path)
    if problems:
        print(f"{smoke_path} / {qualification_path} are not freeze-ready: "
              f"incomplete", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    frozen = qualification["frozen_schedule"]
    # Freeze-side reconstruction: every qualifying checkpoint is rebuilt from its
    # own config and loaded strict, here, rather than trusted from the boolean
    # the cell report wrote about itself.
    reconstructed = []
    declared = qualification.get("package") or {}
    correction = declared.get("package") == contract.CORRECTION_PACKAGE
    for cell in qualification["schedules"][frozen]["cells"]:
        tag = (contract.correction_cell_name(cell["seed"], cell["replicate"])
               if correction
               else contract.development_cell_name(frozen, cell["seed"]))
        checkpoint = Path(f"runs/{tag}.pt")
        if not checkpoint.exists():
            print(f"qualifying checkpoint {checkpoint} is missing: incomplete",
                  file=sys.stderr)
            return 1
        try:
            reconstructed.append(_reconstruct_strict(checkpoint))
        except (EOFError, KeyError, OSError, RuntimeError, TypeError,
                ValueError, pickle.UnpicklingError) as exc:
            print(f"qualifying checkpoint {checkpoint} does not reload strict "
                  f"under the current model: {exc}", file=sys.stderr)
            return 1
    if any(entry["provenance"] != evaluator.DEVELOPMENT
           for entry in reconstructed):
        print("a qualifying checkpoint is not marked as a development artifact",
              file=sys.stderr)
        return 1
    if any(not entry["params_match_config"] for entry in reconstructed):
        print("a qualifying checkpoint's parameter count disagrees with its "
              "config", file=sys.stderr)
        return 1

    body = contract.protocol_v1_dict(
        base=base,
        source_sha256=source_hashes(
            [Path(p) for p in contract.QUALIFICATION_SOURCES]),
        corpus={
            "manifest": str(corpus_path),
            # Two hashes over two different byte strings, under names that say
            # which is which. `PLAN.md` names calling one the other as a trap.
            "canonical_payload_sha256": digests["canonical_payload_sha256"],
            "recorded_canonical_sha256": digests["recorded_canonical_sha256"],
            "file_sha256": digests["file_sha256"],
            "fingerprints": corpus["balance"]["corpus"],
            "blocks": corpus["balance"]["blocks"],
            "program_treatment": corpus["balance"]["intervention"][
                "program_treatment"
            ],
            "vm_census": corpus["vm_census"],
            "accepted": corpus_module.accepts(corpus["balance"]),
            "census_accepted": corpus_module.audit_accepts(corpus["vm_census"]),
        },
        smoke={"report": str(smoke_path),
               "report_sha256": sha256_file(smoke_path),
               "elapsed_s": smoke.get("elapsed_s")},
        qualification={
            **qualification,
            "report": str(qualification_path),
            "report_sha256": sha256_file(qualification_path),
            "freeze_side_reconstruction": reconstructed,
        },
    )
    if not body["corpus"]["accepted"] or not body["corpus"]["census_accepted"]:
        print("the corpus manifest does not clear its own balance report or its "
              "zero-fault VM census", file=sys.stderr)
        return 1
    contract.write_protocol(v1_path, body)
    print(f"protocol  {v1_path}  sha256 {body['protocol_sha256']}")
    print(f"extends   {base['protocol']}  sha256 {base['protocol_sha256']}")
    print(f"schedule  {frozen} (first predeclared eligible schedule passing "
          f"both development seeds)")
    print(f"reloaded  {len(reconstructed)} qualifying checkpoints, strict")
    print(f"corpus    canonical {body['corpus']['canonical_payload_sha256']}")
    print(f"          file      {body['corpus']['file_sha256']}")
    print(f"cells     {len(body['cells'])} frozen training configurations")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true",
                      help="capture the baseline and freeze the protocol over it")
    mode.add_argument("--verify", action="store_true",
                      help="replay the freeze against the current tree")
    mode.add_argument("--freeze-training", action="store_true",
                      help="F5: layer protocol v1 over v0 once the smoke passes")
    ap.add_argument("--corpus", type=Path,
                    default=Path("runs/feedback_corpus_v1_audited.json"))
    ap.add_argument("--smoke", type=Path,
                    default=Path("runs/feedback_smoke_report.json"))
    ap.add_argument("--qualification", type=Path,
                    default=Path("runs/feedback_qualification.json"),
                    help="F5b: the `feedback_qualify.py decide` result that "
                         "names the schedule v1 freezes")
    ap.add_argument("--protocol-v1", type=Path, default=contract.PROTOCOL_V1_PATH)
    ap.add_argument("--refreeze", action="store_true",
                    help="allow --write to replace an existing baseline fixture")
    ap.add_argument("--fixture", type=Path, default=fixture.FIXTURE_PATH)
    ap.add_argument("--protocol", type=Path, default=contract.PROTOCOL_PATH)
    args = ap.parse_args()
    if args.write:
        return _write(args.fixture, args.protocol, args.refreeze)
    if args.freeze_training:
        return _freeze_training(args.protocol, args.protocol_v1, args.corpus,
                                args.smoke, args.qualification)
    return _verify(args.fixture, args.protocol)


if __name__ == "__main__":
    raise SystemExit(main())
