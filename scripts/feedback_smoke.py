"""F5 Adapter: one throwaway run through every Direction 3 code path.

The smoke exists to find the integration faults that unit tests structurally
cannot — a checkpoint that will not reload, a scorer that will not accept a real
checkpoint's config, a stability pass that runs out of memory at the real width.
It trains a short byte/seed-0 relational job and then walks the whole downstream
chain on the checkpoint it produced:

    train -> checkpoint -> reload strict -> standard generation -> soft
    generation -> sequential score in both modes -> equivalence check ->
    stability -> gate

**Its output is worth nothing as evidence and says so.** Every artifact carries
`provenance="engineering"`, and `dm.eval.feedback.refuse_engineering` rejects it
by schema wherever a scientific decision is made. One cell, one seed, and a
budget chosen to be short cannot inform a threshold, a cell selection or a
result. It exists to make F6 boring.

That cuts both ways, and the stability stage is where it shows. The full F7
verdict is computed and recorded, but the smoke *gates* only on
`dm.eval.feedback.SMOKE_STABILITY_CLAUSES` -- the clauses that say something
broke. F7's quality clauses are asked of eight converged final-step checkpoints;
a 400-step cell cannot meet them, and letting it block the stage would be the
same artifact deciding something it has no standing to decide. Every deferred
clause is printed with its value and stays in the report.

    PYTHONPATH=. .venv/bin/python scripts/feedback_smoke.py --steps 200 \\
        --n-train 4000 --device mps --out runs/feedback_smoke_v1_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

import dm.train
from dm.data.dataset import ProgramDataset, loader
from dm.eval import feedback as evaluator
from dm.eval import feedback_contract as contract
from dm.eval.context_cases import CorpusStats, build_step_cases
from dm.eval.feedback import (
    ENGINEERING,
    bootstrap,
    case_requests,
    channel_scales,
    component_draws,
    generic_guards,
    mode_gain,
    score_cases_sequential,
    sequential_matches_full_forward,
    smoke_stability,
    stability,
    stability_verdict,
)
from dm.eval.provenance import sha256_file
from dm.eval.reports import json_safe, staged_path
from dm.eval.sampling import SamplingConfig
from dm.isa.codec import CODECS
from dm.isa.state import LanguagePolicy
from dm.models.transformer import Config, DrawingLM
from dm.train import TrainConfig, train


def _cases(codec, train_programs: list[bytes], val: list[bytes], max_cases: int):
    """A small case table built the way Direction 2 builds one, model-blind."""
    policy = LanguagePolicy.from_programs(train_programs + val)
    stats = CorpusStats.from_programs(train_programs, label="smoke", split="train")
    cases, _ = build_step_cases(val, policy, codec, stats, max_cases=max_cases)
    return cases


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--n-train", type=int, default=4_000)
    ap.add_argument("--n-val", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--cases", type=int, default=8)
    ap.add_argument("--stability-subset", type=int,
                    default=contract.STABILITY_SUBSET_SIZE,
                    help="rows the recurrent probe runs on. A smoke may run "
                         "fewer; the freeze refuses anything but the frozen size")
    ap.add_argument("--gain-calibration",
                    default=contract.DEFAULT_GAIN_CALIBRATION,
                    choices=sorted(contract.GAIN_CALIBRATIONS),
                    help="exercise a named gain calibration end to end. The "
                         "F5c package's one lever fires at the feedback phase "
                         "transition, which on a 24,000-step cell is 12,000 "
                         "steps in -- the most expensive place to discover it "
                         "raises")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tag", default="feedback_smoke")
    ap.add_argument("--out", type=Path,
                    default=Path("runs/feedback_smoke_report.json"))
    ap.add_argument("--training-tag", default=None,
                    help="checkpoint/record stem; defaults to <tag>_training")
    args = ap.parse_args()

    codec = CODECS["byte"]
    started = time.time()
    stage: dict[str, object] = {}
    training_tag = args.training_tag or f"{args.tag}_training"
    training_record_path = dm.train.RUNS / f"{training_tag}.json"
    if args.out.resolve() == training_record_path.resolve():
        raise ValueError(
            "smoke report path aliases the training record; use a separate "
            "--out or --training-tag so the full training history is preserved"
        )

    # --- train -------------------------------------------------------------
    cfg = TrainConfig(
        codec="byte", shape=contract.PILOT_SHAPE, data="feedback",
        n_train=args.n_train, n_val=args.n_val, steps=args.steps,
        eval_every=max(1, args.steps // 2), batch_size=args.batch_size,
        max_len=args.max_len, warmup=min(50, max(1, args.steps // 10)),
        seed=0, data_seed=0, device=args.device, share_init=True,
        feedback_schema=contract.QUALIFICATION_FEEDBACK_SCHEMA,
        gain_calibration=args.gain_calibration,
        extra={"structure": "relational"},
        gen_samples=16, tag=training_tag, artifact_provenance=ENGINEERING,
    )
    train_programs, val_programs = dm.train.build_data(cfg)
    record = train(cfg, verbose=True)
    record_path = dm.train.RUNS / f"{training_tag}.json"
    checkpoint_path = dm.train.RUNS / f"{training_tag}.pt"
    if not record_path.exists() or not checkpoint_path.exists():
        raise RuntimeError(
            "training completed without both its record and checkpoint: incomplete"
        )
    stage["training"] = {
        "complete": record["complete"],
        "steps_done": record["steps_done"],
        "params": record["model"]["params"],
        "feedback_schema": record["config"]["feedback_schema"],
        "missing_record_fields": contract.missing_record_fields(record),
        "feedback": record["feedback"],
        "final_bits_per_drawing": record["final"].get("bits_per_drawing"),
        "record_path": str(record_path),
        "record_sha256": sha256_file(record_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "record_history_length": len(record.get("history", [])),
        "record_val_bits_length": len(record.get("val_bits", [])),
        "record_best_val_bits_length": len(record.get("best_val_bits", [])),
    }

    # --- reload ------------------------------------------------------------
    weights = torch.load(checkpoint_path, weights_only=False)
    model = DrawingLM(Config(**weights["cfg"]))
    model.load_state_dict(weights["state"], strict=True)
    model = model.to(args.device).eval()
    stage["reload"] = {
        "strict": True,
        "feedback_schema": weights["cfg"]["feedback_schema"],
        "params_match_config": model.n_params() == model.cfg.n_params(),
        "provenance": weights.get("provenance"),
    }
    # Diagnostic only. Both sides of the fused-input band move during training
    # and for unrelated reasons; a budget ladder reads this to see whether the
    # gain tracks the embedding it was initialised to match.
    stage["channel"] = channel_scales(model)

    # --- generation in both modes -----------------------------------------
    cases = _cases(codec, train_programs, val_programs, args.cases)
    if not cases:
        print("no step cases in the smoke corpus; raise --n-val", file=sys.stderr)
        return 1
    prompt = torch.tensor([codec.encode(cases[0].prefix_a)] * 4, dtype=torch.long)
    variates = torch.rand(
        4, prompt.shape[1] + 32,
        generator=torch.Generator().manual_seed(contract.seed_for("variates")),
    )
    sampler = SamplingConfig(top_k=40, temperature=1.0)
    decoded = {}
    for mode in contract.RUNTIME_MODES:
        out = model.generate(
            4, max_new=32, prompt=prompt, monitor=codec.halt_monitor(4),
            variates=variates, top_k=sampler.top_k,
            temperature=sampler.temperature, device=args.device, mode=mode,
        )
        decoded[mode] = out.shape[1]
    stage["generation"] = {"widths": decoded, "sampler": sampler.as_dict()}

    # --- generic guards ----------------------------------------------------
    encoded_lengths = [len(codec.encode(program)) for program in val_programs]
    p99_index = int(0.99 * (len(encoded_lengths) - 1))
    generation_max_new = max(1, min(
        args.max_len, int(cfg.gen_cap * sorted(encoded_lengths)[p99_index])
    ))
    guard_variates = torch.rand(
        len(val_programs), generation_max_new,
        generator=torch.Generator().manual_seed(contract.seed_for("variates")),
    )
    stage["generic_guards"] = generic_guards(
        model, val_programs, codec, device=args.device, max_len=args.max_len,
        batch_size=args.batch_size, generation_n=len(val_programs),
        generation_max_new=generation_max_new, variates=guard_variates,
    )

    # --- sequential scoring ------------------------------------------------
    equivalence = sequential_matches_full_forward(
        model, [pair for case in cases for pair in case_requests(case)], codec,
        device=args.device, max_len=args.max_len,
    )
    scored = {
        mode: score_cases_sequential(model, cases, codec, mode=mode,
                                     device=args.device, max_len=args.max_len)
        for mode in contract.EVALUATION_MODES
    }
    gains = mode_gain(scored["standard"], scored["soft"])
    resample = component_draws(gains, reps=200)
    stage["scoring"] = {
        "equivalence": equivalence,
        "cases": len(cases),
        "first_symbol_identical": all(
            left["first_symbol_bits"] == right["first_symbol_bits"]
            for left, right in zip(scored["standard"], scored["soft"])
        ),
        "G": bootstrap(gains, "G", resample),
        "Delta_standard": bootstrap(scored["standard"], "Delta",
                                    component_draws(scored["standard"], reps=200)),
    }

    # --- stability ---------------------------------------------------------
    # Frozen, model-blind and long: rows are drawn from `seed_for("stability")`
    # over the programs that outlive the deepest pass, never from the head of the
    # split. The old first-sixteen subset was mostly converged at pass 32, so the
    # clause read the subset's lengths rather than the channel.
    subset = evaluator.stability_subset(
        val_programs, codec, size=args.stability_subset,
        deepest_pass=max(contract.STABILITY_PASSES),
    )
    held_out = [val_programs[index] for index in subset["indices"]]
    batches = loader(ProgramDataset(held_out, codec, args.max_len),
                     batch_size=len(held_out), shuffle=False)
    _, val_inputs, val_targets = next(iter(batches))
    report = stability(
        model, val_inputs, val_targets, device=args.device,
        valid_halt_loss=stage["generic_guards"]["values"]["valid_halt_loss"],
        symbols_per_byte=codec.stride, subset=subset,
    )
    stability_verdict_result = stability_verdict(report)
    # The F7 verdict is recorded whole; `smoke_stability` only says which of its
    # clauses a 400-step engineering cell is entitled to gate on.
    stage["stability"] = {"report": report,
                          "verdict": stability_verdict_result,
                          "smoke": smoke_stability(stability_verdict_result)}

    arm = contract.QUALIFICATION_FEEDBACK_SCHEMA
    failures = [
        name for name, ok in (
            ("training_complete", record["complete"]),
            ("record_fields", not stage["training"]["missing_record_fields"]),
            ("record_history", stage["training"]["record_history_length"] > 0),
            ("record_val_bits", stage["training"]["record_val_bits_length"] == args.n_val),
            ("record_best_val_bits",
             stage["training"]["record_best_val_bits_length"] == args.n_val),
            ("reload_params", stage["reload"]["params_match_config"]),
            ("reload_provenance", stage["reload"]["provenance"] == ENGINEERING),
            ("source_architecture",
             stage["training"]["feedback_schema"] == arm
             and stage["reload"]["feedback_schema"] == arm),
            ("sequential_equivalence", equivalence["matches"]),
            ("first_symbol_zero", stage["scoring"]["first_symbol_identical"]),
            ("generic_guards", stage["generic_guards"]["passed"]),
            ("stability", stage["stability"]["smoke"]["passed"]),
        ) if not ok
    ]
    smoke_passed = not failures

    body = json_safe({
        "schema": contract.REPORT_SCHEMA,
        "status": "complete" if smoke_passed else "failed",
        "smoke_passed": smoke_passed,
        "failures": failures,
        # The whole point of the file. Read by `refuse_engineering` wherever a
        # scientific decision is made.
        "provenance": ENGINEERING,
        "decision_value": "none",
        "protocol": contract.PROTOCOL,
        "config": {**cfg.__dict__, "tier": int(cfg.tier)},
        "elapsed_s": round(time.time() - started, 1),
        "stages": stage,
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    staged = staged_path(args.out)
    staged.write_text(json.dumps(body, indent=1, sort_keys=True) + "\n")
    staged.replace(args.out)

    print(json.dumps({
        "equivalence": equivalence,
        "first_symbol_identical": stage["scoring"]["first_symbol_identical"],
        "stability": stage["stability"]["verdict"],
        "generic_guards": stage["generic_guards"],
        "missing_record_fields": stage["training"]["missing_record_fields"],
        "smoke_passed": smoke_passed,
        "failures": failures,
        "report": str(args.out),
    }, indent=1))

    deferred = stage["stability"]["smoke"]["deferred_clauses"]
    if deferred:
        print("\nF7 stability clauses this engineering cell does NOT gate on, "
              "and their values here:", file=sys.stderr)
        for name, value in deferred.items():
            print(f"  {name}: {value}", file=sys.stderr)
        print("  These are F7's, on eight converged final-step checkpoints. A "
              "short cell cannot meet them and has no standing to move them; "
              "settle them before F6 launches, not after.", file=sys.stderr)

    try:
        evaluator.refuse_engineering(body, "the smoke report")
    except ValueError:
        pass
    else:  # pragma: no cover - would mean the marker went missing
        print("the smoke report was NOT rejected by the scientific gate",
              file=sys.stderr)
        return 1

    if failures:
        print(f"\nSMOKE FAILED: {failures}", file=sys.stderr)
        return 1
    print("\nevery stage ran; the report is engineering-only and carries no "
          "decision value")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
