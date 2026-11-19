"""Task-parameterized joined boundary/decode records for the bounded package."""
from __future__ import annotations

import math
import time
from dataclasses import asdict

import torch
import torch.nn.functional as F

from dm.eval.relation_decision_diagnostic import (
    correct_set,
    decision_record,
    factor_values,
    offline_search,
    prefix_scores,
)
from dm.isa.codec import BOS, N_SPECIAL
from dm.models.relation import COPY, EMIT, choose_action
from dm.relation.decoding import CONTENT_POLICY
from dm.relation.demonstrated_card import uniforms
from dm.relation.wiring import identity
from dm.train_relation import (
    RelationBatchPlanner,
    TrainingCorpus,
    capture_rng_state,
    restore_rng_state,
)
from dm.vm.interp import VM


def evaluate(trainer, cases, split, emit, checkpoint_digest):
    before = identity(trainer)
    rng = capture_rng_state(trainer.cursor.generator)
    was_training = trainer.model.training
    planner = RelationBatchPlanner(TrainingCorpus.engineering_cases(cases), max_len=256)
    results = []
    try:
        trainer.model.eval()
        with torch.no_grad():
            for index, case in enumerate(cases):
                started = time.perf_counter()
                tokens = torch.tensor([[BOS, *(b + N_SPECIAL for b in case.flat[:-1])]])
                logits = trainer.model(tokens)
                losses = F.cross_entropy(logits[0].float(), torch.tensor(
                    [b + N_SPECIAL for b in case.flat]), reduction="none")
                row = {"cell_seed": trainer.config.seed, "arm": trainer.config.arm,
                       "step": trainer.completed_step, "checkpoint_digest": checkpoint_digest,
                       "case_id": case.case_id, "split": split, "class": case.stratum,
                       "source_ids": case.source_ids, "donor_source_ids": case.donor_source_ids,
                       "all_byte_nll": {"sum": float(losses.sum()), "count": len(losses)},
                       "target_byte_nll": {"sum": float(losses[case.target_start:case.target_stop].sum()),
                                           "count": len(case.target)}, "queries": [], "decodes": {}}
                if trainer.config.arm != "none":
                    for q in planner.batch([index]).queries:
                        prefix = case.flat[:q.boundary]
                        scores, raw, _, _ = prefix_scores(trainer.model, prefix)
                        gate = scores.gate_log_probs(0).tolist()
                        correct = correct_set(scores, q.equivalents)
                        query = {"boundary": q.boundary, "gate_target": q.gate_target,
                                 "gate_nll": -gate[q.gate_target], "correct_set": correct,
                                 "prompt": q.boundary == case.target_start, "raw": raw}
                        if query["prompt"]:
                            offline = offline_search(scores, prefix, 256)
                            valid = next((r for r in reversed(offline["records"]) if r["fault"] is None), None)
                            query.update({"production": decision_record(choose_action(scores, 0, prefix, max_len=256), gate),
                                          "offline": offline,
                                          "highest_valid_full_target_correct": None if valid is None else valid["emitted_hex"] == case.target.hex(),
                                          "margin": None if valid is None else gate[COPY] - gate[EMIT] + sum(factor_values(scores, valid["key"]))})
                        row["queries"].append(query)
                modes = ["standard"]
                if trainer.config.arm != "none":
                    modes.append("predicted_copy")
                if case.actions:
                    modes.append("oracle_copy")
                for mode in modes:
                    events = []
                    kwargs = {} if mode != "oracle_copy" else {"oracle_actions": {
                        (0, a.boundary): a.runtime_key() for a in case.actions}}
                    start = time.perf_counter()
                    output = trainer.model.generate_relation(
                        [case.prompt], 256 - len(case.prompt), mode=mode, literal_policy=CONTENT_POLICY,
                        observer=events.append, observation_level="actions",
                        variates=torch.tensor([uniforms(split, case.case_id)], dtype=torch.float32), **kwargs)
                    tokens_out = output[0].tolist()
                    stop = next(e for e in events if e.kind == "row_stopped")
                    content = tokens_out[:stop.useful_length]
                    if any(not N_SPECIAL <= token < N_SPECIAL + 256 for token in content):
                        raise ValueError("production output contains non-byte content")
                    flat = bytes(token - N_SPECIAL for token in content)
                    trace = VM().run(flat)
                    decisions = [e for e in events if e.kind == "action_decision"]
                    admissions = [e for e in decisions if e.admission_bytes > 0]
                    exact = flat[case.target_start:case.target_stop] == case.target
                    row["decodes"][mode] = {"flat_hex": flat.hex(), "target_exact": exact,
                        "exact_with_copy": exact and any(e.byte_offset == case.target_start for e in admissions),
                        "copy_invocations": len(admissions),
                        "executor_attempts": sum(e.executor_attempts for e in decisions),
                        "refused_attempts": sum(sum(count for _, count in e.faults) for e in decisions),
                        "copied_bytes": sum(e.admission_bytes for e in admissions),
                        "target_reached": len(flat) > case.target_start,
                        "valid_halt": trace.valid and trace.halted,
                        "stop_cause": stop.cause, "truncated": stop.at_horizon and not trace.halted,
                        "seconds": time.perf_counter() - start, "events": [asdict(e) for e in events]}
                row["seconds"] = time.perf_counter() - started
                emit(row)
                results.append(row)
    finally:
        trainer.model.train(was_training)
        restore_rng_state(rng, trainer.cursor.generator)
    if before != identity(trainer):
        raise ValueError("evaluation mutated trainer identity")
    return results


def fit_result(rows):
    positives = [r for r in rows if r["class"] == "positive"]
    negatives = [r for r in rows if r["class"] != "positive"]
    if len(positives) != 16 or len(negatives) != 16 or len({r["case_id"] for r in rows}) != 32:
        raise ValueError("fit readout denominator mismatch")
    exact = sum(r["decodes"]["predicted_copy"]["exact_with_copy"] for r in positives)
    false_copy = sum(r["decodes"]["predicted_copy"]["copy_invocations"] > 0 for r in negatives)
    halt = sum(r["decodes"]["predicted_copy"]["valid_halt"] for r in rows)
    return {"positive_exact_copy": exact, "positive_denominator": 16,
            "negative_copy_cases": false_copy, "negative_denominator": 16,
            "valid_halt": halt, "full_program_denominator": 32,
            "pass": exact >= 15 and false_copy == 0,
            "weak_halt": halt < 32,
            "positive_gate_nll": mean_prompt(positives, "gate_nll"),
            "negative_gate_nll": mean_prompt(negatives, "gate_nll"),
            "positive_margin": mean_prompt(positives, "margin"),
            "highest_valid_correct": sum(next(q for q in r["queries"] if q["prompt"])[
                "highest_valid_full_target_correct"] is True for r in positives)}


def mean_prompt(rows, key):
    values = [next(q for q in r["queries"] if q["prompt"])[key] for r in rows]
    finite = [v for v in values if v is not None and math.isfinite(v)]
    return {"sum": sum(finite), "count": len(finite),
            "mean": sum(finite) / len(finite) if finite else None}
