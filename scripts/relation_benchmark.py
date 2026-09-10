"""Non-gating R4 metadata/gather benchmark; no model or optimizer is constructed.

Measures, on fixed engineering cases:

* metadata construction under identical request traces for the disabled,
  fitting-hot, deliberately thrashing and oversize-bypass cache policies, with
  the planner's own counters and high-water charge beside raw timings;
* packed endpoint gather/backward against the pre-correction per-span path,
  rechecking exact endpoint gradients before any timing is reported; and
* fresh-process RSS high-water readings for the same traces, including the
  empty/corpus-only baseline, in the platform's own units.

Nine repeats after one discarded warmup, comparator order alternated per
repeat, raw samples published with median and range.  This is engineering
evidence about the trainer's host work.  It is not the R5 memory/wall-time
qualification and no timing threshold is invented here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

from dm.data import relation
from dm.train_relation import (
    RETENTION_CHARGE_METHOD,
    RelationBatchPlanner,
    TrainingCorpus,
    build_batch_plan,
    pack_candidates,
)

TRACE = ([0, 1, 2], [3, 4, 5], [6, 7], [7, 3, 1], [0, 6, 2], [4, 5], [1, 2, 3], [5, 6, 7], [0, 4])


def _training() -> TrainingCorpus:
    built = relation.build_venue1(
        8, seed=41, motifs=relation.motif_pool(16, seed=40), tuples=relation.venue1_tuples())
    return TrainingCorpus.engineering_cases(built.cases)


def _policies(training: TrainingCorpus) -> dict[str, dict[str, int]]:
    spans = [sum(len(q.spans) for q in build_batch_plan(training, [i], max_len=256).queries)
             for i in range(len(training.cases))]
    # The oversize case is identified by its span/charge metadata, not by
    # changing corpus semantics: the widest case simply exceeds the span limit.
    return {
        "disabled": {"max_cached_cases": 0},
        "fitting_hot": {},
        "thrashing": {"max_cached_cases": 1},
        "oversize_bypass": {"max_cached_spans": sorted(spans)[-2]},
    }


def _legacy_endpoints(plan, states):
    # Pre-correction indexing algorithm, retained solely as an engineering
    # comparator. It produces one SelectBackward chain per span endpoint.
    starts, stops = [], []
    for query in plan.queries:
        for start, stop in query.spans:
            starts.append(states[query.row, start])
            stops.append(states[query.row, stop])
    return torch.stack(starts), torch.stack(stops)


def _nodes(root):
    pending, visited = [root], set()
    while pending:
        node = pending.pop()
        if node is not None and node not in visited:
            visited.add(node)
            pending.extend(child for child, _ in node.next_functions)
    return len(visited)


def _summary(values: list[float]) -> dict[str, float]:
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


_RSS_SCRIPT = """
import json, resource, sys
from dm.data import relation
from dm.train_relation import RelationBatchPlanner, TrainingCorpus
policy = json.loads(sys.argv[1]); trace = json.loads(sys.argv[2])
built = relation.build_venue1(8, seed=41, motifs=relation.motif_pool(16, seed=40), tuples=relation.venue1_tuples())
training = TrainingCorpus.engineering_cases(built.cases)
corpus_only = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
if policy is not None:
    planner = RelationBatchPlanner(training, max_len=256, **policy)
    for _ in range(3):
        for indices in trace:
            planner.batch(indices)
    snapshot = planner.snapshot().as_dict()
else:
    snapshot = None
print(json.dumps({"corpus_only_maxrss": corpus_only,
                  "final_maxrss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  "snapshot": snapshot}))
"""


def _rss(policy: dict[str, int] | None, root: Path) -> dict[str, object]:
    result = subprocess.run(
        [sys.executable, "-c", _RSS_SCRIPT, json.dumps(policy), json.dumps(TRACE)],
        text=True, capture_output=True, check=True, cwd=root,
        env={**os.environ, "PYTHONPATH": str(root), "PYTHONDONTWRITEBYTECODE": "1"})
    return json.loads(result.stdout)


def benchmark(repeats: int = 9) -> dict[str, object]:
    training = _training()
    policies = _policies(training)
    reference = {tuple(indices): build_batch_plan(training, indices, max_len=256) for indices in TRACE}
    metadata_raw: dict[str, list[float]] = {name: [] for name in policies}
    snapshots: dict[str, dict[str, object]] = {}
    for repeat in range(repeats + 1):  # one discarded warmup; alternate comparator order
        order = list(policies) if repeat % 2 else list(reversed(policies))
        for name in order:
            planner = RelationBatchPlanner(training, max_len=256, **policies[name])
            started = time.perf_counter()
            for _ in range(3):  # three passes over the trace: cold, then hot where policy allows
                for indices in TRACE:
                    plan = planner.batch(indices)
                    expected = reference[tuple(indices)]
                    assert plan.queries == expected.queries and torch.equal(plan.inputs, expected.inputs)
            elapsed = time.perf_counter() - started
            if repeat:
                metadata_raw[name].append(elapsed)
                snapshots[name] = planner.snapshot().as_dict()
    plan = build_batch_plan(training, [0, 1, 2, 3], max_len=256)
    gather_raw = {"legacy_gather_backward_seconds": [], "packed_gather_backward_seconds": []}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(713)
        states = torch.randn(*plan.inputs.shape, 128, requires_grad=True)
    expected_grad = None
    nodes = {}
    for repeat in range(repeats + 1):
        for packed in ((False, True) if repeat % 2 else (True, False)):
            states.grad = None
            started = time.perf_counter()
            if packed:
                result, _, _ = pack_candidates(plan, states)
                starts, stops = result.start_states, result.stop_states
            else:
                starts, stops = _legacy_endpoints(plan, states)
            objective = 2 * starts.sum() + 3 * stops.sum()
            objective.backward(retain_graph=True)
            elapsed = time.perf_counter() - started
            name = "packed" if packed else "legacy"
            nodes[name] = _nodes(objective.grad_fn)
            if expected_grad is None:
                expected_grad = states.grad.clone()
            assert torch.equal(states.grad, expected_grad)
            if repeat:
                gather_raw[f"{name}_gather_backward_seconds"].append(elapsed)
    root = Path(__file__).resolve().parents[1]
    rss = {"corpus_only": _rss(None, root)}
    rss.update({name: _rss(policy, root) for name, policy in policies.items()})
    paths = ("dm/train_relation.py", "dm/models/relation.py", "dm/data/dataset.py",
             "scripts/relation_benchmark.py")
    return {
        "schema": 2, "kind": "r4_engineering_microbenchmark", "model_constructed": False,
        "optimizer_steps": 0, "resource_qualification": False,
        "scope": ("metadata construction under cache policies and endpoint gather/backward only; "
                  "excludes trunk/head training; RSS readings are process-level and include the "
                  "interpreter, torch and the verified corpus"),
        "trace": {"batches": [list(b) for b in TRACE], "passes": 3, "cases": len(training.cases),
                  "distinct_requests": 3 * sum(len(b) for b in TRACE)},
        "gather_shape": {"rows": plan.inputs.shape[0], "positions": plan.inputs.shape[1],
                         "d_model": 128, "queries": len(plan.queries),
                         "spans": sum(len(q.spans) for q in plan.queries)},
        "repeats": repeats, "charge_method": RETENTION_CHARGE_METHOD,
        "policies": policies,
        "metadata_trace_seconds": {"raw": metadata_raw,
                                   "summary": {name: _summary(v) for name, v in metadata_raw.items()}},
        "cache_snapshots": snapshots,
        "gather": {"raw": gather_raw, "summary": {name: _summary(v) for name, v in gather_raw.items()},
                   "autograd_nodes": nodes, "endpoint_gradients_exact": True},
        "rss": {"unit": "bytes" if sys.platform == "darwin" else "kibibytes",
                "note": "ru_maxrss high-water of a fresh process per policy; not retained-charge bytes",
                "readings": rss},
        "training_program_fingerprint": training.program_fingerprint,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "torch": torch.__version__, "threads": torch.get_num_threads(),
                        "page_size": resource.getpagesize()},
        "source_hashes": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in paths},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="optional engineering JSON; refuses overwrite")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error(f"output already exists: {args.output}")
    report = benchmark()
    encoded = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as output:
            output.write(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
