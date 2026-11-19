"""Fixed, zero-update R5 decision inspection; never imported by training runtime."""
from __future__ import annotations

import hashlib
import heapq
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from dm.isa.codec import BOS, N_SPECIAL
from dm.isa.spec import SPECS, Op, Tier
from dm.isa.transform import D4, Transform
from dm.models.relation import (
    _CANDIDATE_LOCAL_FAULTS,
    COPY,
    EMIT,
    RELATION_KBEST_ACTIONS,
    RELATION_MAX_CANDIDATE_SPANS,
    PackedCandidates,
    choose_action,
    length_bin,
    score_snapshot,
)
from dm.relation import (
    COUNT_SUPPORT,
    D4_SUPPORT,
    PREDICTED_SUPPORT,
    TRANSLATION_SUPPORT,
    CopyActionKey,
    TransducerFault,
    candidate_spans,
    execute_copy,
)
from dm.relation import wiring as w
from dm.train_relation import build_batch_plan, capture_rng_state, restore_rng_state

SCHEMA = 1
# Fixed before checkpoint scoring; CPU parity uses the existing 1e-5 scale.
ATOL, RTOL = 1e-5, 1e-5
RUN = Path('artifacts/relation/r5-wiring-smoke-diagnostic-20260910-rb12')
CARD = '88db8c99a1e7e0fd9444dd235d7ef6223106585f3cd85a5fe7b1edd1a692ee48'
PROBE_SOURCES = (
    'dm/eval/relation_decision_diagnostic.py',
    'scripts/relation_decision_diagnostic.py',
    'dm/eval/relation_decision_figures.py',
    'dm/eval/plot.py',
    'tests/test_relation_decision_diagnostic.py',
)


def file_hash(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def metric(total, denominator, *, applicable=True):
    return {'status': 'measured' if applicable else 'not_applicable',
            'sum': total if applicable else None,
            'denominator': denominator if applicable else None,
            'mean': total / denominator if applicable and denominator else None,
            'unit': 'nats', 'uncertainty': None}


def prefix_scores(model, prefix):
    """No supervision, future bytes, case identity or target horizon enters head."""
    tokens = torch.tensor([[BOS, *(b + N_SPECIAL for b in prefix)]])
    logits, states = model(tokens, return_state=True)
    spans = candidate_spans(prefix, len(prefix))
    starts = torch.tensor([s for s, _ in spans], dtype=torch.long)
    stops = torch.tensor([s for _, s in spans], dtype=torch.long)
    packed = PackedCandidates(
        states[0, -1:], states[0, starts], states[0, stops],
        torch.tensor([length_bin(e-s) for s, e in spans], dtype=torch.long),
        starts, stops, torch.tensor([0, len(spans)]))
    scores = model.score_relation(packed)
    raw = asdict(score_snapshot(scores, 0, 0, len(prefix)))
    return scores, raw, logits[0, -1], states[0]


def key_values(key):
    return [key.source_start, key.source_stop, key.step.d4.code,
            key.step.dx, key.step.dy, key.total_count]


def action_key(values, boundary):
    start, stop, d4, dx, dy, count = values
    return CopyActionKey(boundary, start, stop, Transform(D4.of(d4), dx, dy), count)


def factor_values(scores, key):
    spans = list(zip(scores.packed.source_start.tolist(), scores.packed.source_stop.tolist()))
    start, stop, d4, dx, dy, count = key
    indices = [spans.index((start, stop)), D4_SUPPORT.index(d4),
               TRANSLATION_SUPPORT.index(dx), TRANSLATION_SUPPORT.index(dy),
               COUNT_SUPPORT.index(count)]
    return [float(lp[i]) for lp, i in zip(
        (scores.span_log_probs(0), *scores.factor_log_probs(0)), indices, strict=True)]


def ranked_tuples(scores):
    """Bounded best-first lattice; canonical indices break exact score ties.

    Ranking uses production float32 addition including gate. Gate comparison is
    omitted, not replaced with an artificial gate that could erase score gaps.
    """
    lps = (scores.span_log_probs(0), *scores.factor_log_probs(0))
    if not len(lps[0]):
        return
    orders = [sorted(range(len(lp)), key=lambda i: (-float(lp[i]), i)) for lp in lps]

    def item(ranks):
        indices = tuple(order[r] for order, r in zip(orders, ranks, strict=True))
        value = scores.gate_log_probs(0)[COPY]
        for lp, i in zip(lps, indices, strict=True):
            value = value + lp[i]
        return -float(value), indices, ranks

    first = (0, 0, 0, 0, 0)
    heap, seen = [item(first)], {first}
    for _ in range(RELATION_KBEST_ACTIONS):
        if not heap:
            break
        _, indices, ranks = heapq.heappop(heap)
        span, d4, dx, dy, count = indices
        key = [int(scores.packed.source_start[span]), int(scores.packed.source_stop[span]),
               D4_SUPPORT[d4], TRANSLATION_SUPPORT[dx], TRANSLATION_SUPPORT[dy], COUNT_SUPPORT[count]]
        yield key
        for dim in range(5):
            adjacent = list(ranks)
            adjacent[dim] += 1
            adjacent = tuple(adjacent)
            if adjacent[dim] < len(orders[dim]) and adjacent not in seen:
                seen.add(adjacent)
                heapq.heappush(heap, item(adjacent))


def decision_record(decision, gate):
    return {'emit': decision.emit, 'score': decision.score,
            'key': None if decision.action is None else key_values(decision.action),
            'margin': None if decision.emit else decision.score - gate[EMIT],
            'attempts': decision.expansions, 'faults': dict(decision.faults),
            'exit': decision.exit_reason}


def offline_search(scores, prefix, capacity):
    records = []
    for key in ranked_tuples(scores):
        try:
            execution = execute_copy(prefix, action_key(key, len(prefix)),
                                     policy=PREDICTED_SUPPORT, max_len=capacity)
        except TransducerFault as exc:
            if exc.code not in _CANDIDATE_LOCAL_FAULTS:
                raise
            records.append({'key': key, 'fault': exc.code.value})
        else:
            records.append({'key': key, 'fault': None, 'emitted_hex': execution.appended.hex()})
            return {'status': 'highest_ranked_valid_found', 'attempts': len(records), 'records': records}
    return {'status': 'none_found_in_first_32' if len(records) == RELATION_KBEST_ACTIONS
            else 'support_exhausted', 'attempts': len(records), 'records': records}


def join_key(step, checkpoint, case, boundary, prefix, spans):
    return {'step': step, 'checkpoint_sha256': checkpoint, 'case_id': case.case_id,
            'boundary': boundary, 'prefix_sha256': hashlib.sha256(prefix).hexdigest(),
            'layout_sha256': w.digest(spans)}


def validate_join(actual, expected):
    if actual != expected:
        raise ValueError('snapshot/supervision join mismatch')


def correct_set(scores, alternatives):
    keys = [[a.source_start, a.source_stop, a.d4, a.dx, a.dy, a.total_count] for a in alternatives]
    if len({tuple(k) for k in keys}) != len(keys):
        raise ValueError('duplicate complete action key')
    if not keys:
        return {'status': 'not_applicable', 'size': 0, 'log_mass': None, 'best': None, 'keys': []}
    values = [factor_values(scores, key) for key in keys]
    # Match training's float32 sum/logsumexp, without projected cross-products.
    totals = [sum((torch.tensor(v) for v in factors), torch.tensor(0.)) for factors in values]
    best = min(range(len(keys)), key=lambda i: (-float(totals[i]), tuple(keys[i])))
    return {'status': 'measured', 'size': len(keys), 'keys': keys,
            'log_mass': float(torch.logsumexp(torch.stack(totals), 0)),
            'best': {'key': keys[best], 'factor_log_probs': values[best],
                     'log_q': float(totals[best])}}


def correctness(key, prefix, target, correct, capacity):
    if key is None or correct['status'] != 'measured':
        return {'status': 'not_applicable'}
    execution = execute_copy(prefix, action_key(key, len(prefix)), policy=PREDICTED_SUPPORT, max_len=capacity)
    emitted = execution.appended
    return {'status': 'measured', 'correct_first_step': key in correct['keys'],
            'emitted_bytes': len(emitted), 'correct_prefix': target.startswith(emitted),
            'one_action_full_block_exact': emitted == target}


def byte_head(logits, target):
    lp = F.log_softmax(logits.float(), -1)
    content = F.log_softmax(logits[N_SPECIAL:].float(), -1)
    valid = [int(op) for op, spec in SPECS.items() if spec.tier == Tier.L0]
    return {'reference_byte': target, 'reference_log_probability': float(lp[target+N_SPECIAL]),
            'content_reference_log_probability': float(content[target]),
            'valid_L0_opcode_mass': float(content[valid].exp().sum()),
            'halt_probability': float(content[int(Op.HALT)].exp()),
            'instruction_boundary': True, 'normalization': 'content_bytes_v1'}


def populations(fixture):
    plan = build_batch_plan(fixture, range(8), max_len=256)
    groups = {}
    for q in plan.queries:
        prefix = fixture.cases[q.case_index].flat[:q.boundary]
        groups.setdefault(hashlib.sha256(prefix).hexdigest(), []).append(q)
    assert len(plan.queries) == 72 and len(groups) == 57
    assert sum(q.gate_target for q in plan.queries) == 7
    return plan, groups


def freeze(output):
    """Write immutable probe specification before any model is loaded/scored."""
    if torch.get_default_dtype() != torch.float32 or torch.is_autocast_enabled('cpu'):
        raise ValueError('probe requires CPU float32 without autocast')
    card = w.validate(RUN)
    assert w.digest(card) == CARD
    fixture = w._fixture()
    plan, _groups = populations(fixture)
    inputs = {str(p): file_hash(p) for p in sorted(RUN.iterdir()) if p.is_file()}
    spec = {'schema': SCHEMA, 'card_sha256': CARD, 'input_files': inputs,
            'runtime_sources': w.sources(), 'probe_sources': {p: file_hash(p) for p in PROBE_SOURCES},
            'environment': card['environment'] if 'environment' in card else card.get('runtime'),
            'steps': [28, 56], 'arms': list(w.ARMS), 'query_rows': 72,
            'prefix_groups': 57, 'byte_denominator': plan.content_bytes,
            'action_query_records': 144, 'positive_records': 14, 'emit_records': 130,
            'prompt_subset_records': 16, 'all_query_max_len': 256,
            'prompt_max_len': 'case.target_stop (executor byte capacity)',
            'supports': {'d4': D4_SUPPORT, 'translation': TRANSLATION_SUPPORT, 'count': COUNT_SUPPORT},
            'span_cap': RELATION_MAX_CANDIDATE_SPANS, 'search_cap': RELATION_KBEST_ACTIONS,
            'atol': ATOL, 'rtol': RTOL, 'scoring': 'prefix_only_one_query',
            'parity': 'every declared query versus full causal forward',
            'fixture': {k: card[k] for k in ('program_fingerprint', 'supervision_identity', 'case_order')},
            'runtime_policy': 'CPU float32; deterministic; no autocast; telemetry off',
            'uncertainty': None, 'optimizer_updates': 0,
            'population': [{'case': q.case_index, 'boundary': q.boundary, 'gate': q.gate_target}
                           for q in plan.queries]}
    output.mkdir(parents=True, exist_ok=False)
    save(output/'spec.json', spec)
    return spec


def verify_inputs(spec):
    for path, expected in {**spec['input_files'], **spec['runtime_sources'], **spec['probe_sources']}.items():
        if file_hash(path) != expected:
            raise ValueError(f'input/source drift: {path}')
    w.validate(RUN)


def summarize(records, byte_records):
    result = {'byte': byte_records, 'relation': {}, 'prefix_groups': {}}
    for step in (28, 56):
        rows = [r for r in records if r['step'] == step]
        positive = [r for r in rows if r['gate_target'] == COPY]
        negative = [r for r in rows if r['gate_target'] == EMIT]
        result['relation'][str(step)] = {
            'gate': metric(sum(r['gate_nll'] for r in rows), len(rows)),
            'positive_gate': metric(sum(r['gate_nll'] for r in positive), len(positive)),
            'emit_gate': metric(sum(r['gate_nll'] for r in negative), len(negative)),
            'joint': metric(-sum(r['correct']['log_mass'] for r in positive), len(positive)),
            'positive_copy_probabilities': [math.exp(r['gate_log_probs'][COPY]) for r in positive],
            'emit_copy_probabilities': [math.exp(r['gate_log_probs'][COPY]) for r in negative]}
        groups = {}
        for r in rows:
            groups.setdefault(r['prefix_sha256'], []).append(r)
        grouped = {'gate': metric(sum(sum(r['gate_nll'] for r in group)/len(group)
                                      for group in groups.values()), len(groups)),
                   'weighting': 'each eligible prefix group weight 1; labels equally weighted within group'}
        for label, name in ((COPY, 'positive_gate'), (EMIT, 'emit_gate')):
            selected = [[r for r in group if r['gate_target'] == label] for group in groups.values()]
            selected = [group for group in selected if group]
            grouped[name] = metric(sum(sum(r['gate_nll'] for r in group)/len(group)
                                       for group in selected), len(selected))
            grouped[name + '_copy_probabilities'] = [math.exp(group[0]['gate_log_probs'][COPY])
                                                     for group in selected]
            if label == COPY:
                grouped['joint'] = metric(sum(-sum(r['correct']['log_mass'] for r in group)/len(group)
                                              for group in selected), len(selected))
        result['prefix_groups'][str(step)] = grouped
    return result


def execute(output):
    output = Path(output)
    spec = freeze(output)
    caller_rng = capture_rng_state()
    policy = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())
    try:
        fixture = w._fixture()
        plan, groups = populations(fixture)
        records, byte_records, identities, byte_boundaries = [], [], [], []
        traces = [json.loads(line) for line in (RUN/'decodes.jsonl').read_text().splitlines()]
        for arm in w.ARMS:
            for step in (28, 56):
                name = f'{arm}-' + ('restart-28.pt' if step == 28 else 'final-56.pt')
                checkpoint = spec['input_files'][str(RUN/name)]
                trainer = w.load_checkpoint(RUN/name, json.loads((RUN/'run-card.json').read_text()), fixture,
                                            arm=arm, role='restart' if step == 28 else 'final', step=step)
                before = w.identity(trainer)
                mode = trainer.model.training
                gradients = [p.grad for p in trainer.model.parameters()]
                trainer.model.eval()
                totals = {'all': [0., 0], 'target': [0., 0]}
                per_case = []
                try:
                    with torch.no_grad():
                        for index, case in enumerate(fixture.cases):
                            tokens = torch.tensor([[BOS, *(b+N_SPECIAL for b in case.flat[:-1])]])
                            logits, states = trainer.model(tokens, return_state=True)
                            targets = torch.tensor([b+N_SPECIAL for b in case.flat])
                            losses = F.cross_entropy(logits[0].float(), targets, reduction='none')
                            for scope, loss in [('all', losses), ('target', losses[case.target_start:case.target_stop])]:
                                totals[scope][0] += float(loss.sum())
                                totals[scope][1] += len(loss)
                            per_case.append({'case_index': index, 'case_id': case.case_id,
                                             'all': metric(float(losses.sum()), len(losses)),
                                             'target': metric(float(losses[case.target_start:case.target_stop].sum()), len(case.target))})
                            for q in [q for q in plan.queries if q.case_index == index]:
                                byte_boundaries.append({'arm': arm, 'step': step, 'case_index': index,
                                    'case_id': case.case_id, 'boundary': q.boundary,
                                    'checkpoint_sha256': checkpoint, 'role': 'teacher_forced_instruction_boundary',
                                    'action_status': 'not_applicable' if arm == 'none' else 'see queries',
                                    **byte_head(logits[0, q.boundary], case.flat[q.boundary])})
                            if arm == 'none':
                                continue
                            for q in [q for q in plan.queries if q.case_index == index]:
                                prefix = case.flat[:q.boundary]
                                scores, raw, next_logits, prefix_states = prefix_scores(trainer.model, prefix)
                                torch.testing.assert_close(prefix_states, states[0, :q.boundary+1], atol=ATOL, rtol=RTOL)
                                torch.testing.assert_close(next_logits, logits[0, q.boundary], atol=ATOL, rtol=RTOL)
                                spans = list(zip(scores.packed.source_start.tolist(), scores.packed.source_stop.tolist()))
                                key = join_key(step, checkpoint, case, q.boundary, prefix, spans)
                                # Freeze detached raw record before annotation-dependent reductions.
                                snapshot = {'join': key, 'raw': raw}
                                validate_join(snapshot['join'], join_key(step, checkpoint, case, q.boundary, prefix, list(q.spans)))
                                gate = scores.gate_log_probs(0).tolist()
                                production = choose_action(scores, 0, prefix, max_len=256)
                                offline = offline_search(scores, prefix, 256)
                                correct = correct_set(scores, q.equivalents)
                                best_raw = next(ranked_tuples(scores), None)
                                raw_factors = None if best_raw is None else factor_values(scores, best_raw)
                                log_q = None if raw_factors is None else sum(raw_factors)
                                odds = gate[COPY] - gate[EMIT]
                                row = {**key, 'arm': arm, 'case_index': index,
                                       'snapshot_sha256': w.digest(snapshot), 'raw': raw,
                                       'prefix_hex': prefix.hex(), 'prefix_group': key['prefix_sha256'],
                                       'role': 'training_query', 'max_len': 256,
                                       'prompt_subset': q.boundary == case.target_start,
                                       'span_count': len(spans), 'normalization': 'query_local_float32_log_softmax_v1',
                                       'gate_log_probs': gate, 'gate_log_odds': odds,
                                       'gate_target': q.gate_target, 'gate_nll': -gate[q.gate_target],
                                       'unit': 'nats', 'evidence_role': 'derived_fixed_checkpoint', 'uncertainty': None,
                                       'raw_max': None if best_raw is None else {
                                           'key': best_raw, 'factor_log_probs': raw_factors, 'log_q': log_q,
                                           'margin': odds+log_q, 'g_required': 1/(1+math.exp(log_q))},
                                       'correct': correct, 'production': decision_record(production, gate),
                                       'offline': offline, 'byte_head': byte_head(next_logits, case.flat[q.boundary])}
                                selected = None if production.action is None else key_values(production.action)
                                target = case.flat[q.boundary:case.target_stop]
                                row['selected_correctness'] = correctness(selected, prefix, target, correct, 256)
                                valid = offline['records'][-1] if offline['records'] and offline['records'][-1]['fault'] is None else None
                                row['offline_correctness'] = correctness(None if valid is None else valid['key'], prefix, target, correct, 256)
                                row['blockers'] = {'gate_alone': odds < 0,
                                    'raw_tuple_loses': None if log_q is None else odds+log_q < 0,
                                    'best_valid_examined_incorrect': None if valid is None or not q.gate_target else valid['key'] not in correct['keys'],
                                    'search_incomplete': offline['status'] == 'none_found_in_first_32',
                                    'correct_mass_spread': None if not q.gate_target else correct['log_mass']-correct['best']['log_q']}
                                if correct['best']:
                                    correct['best']['margin'] = odds + correct['best']['log_q']
                                    row['best_correct_execution'] = correctness(correct['best']['key'], prefix, target, correct, 256)
                                if row['prompt_subset']:
                                    prompt = choose_action(scores, 0, prefix, max_len=case.target_stop)
                                    row['prompt_production'] = {**decision_record(prompt, gate), 'max_len': case.target_stop}
                                    row['prompt_offline'] = offline_search(scores, prefix, case.target_stop)
                                    observed = next(r for r in traces if (r['arm'],r['step'],r['mode'],r['row']) == (arm,step,'predicted_copy',index))
                                    event = next(e for e in observed['events'] if e['kind'] == 'action_decision')
                                    row['observed_prompt_event'] = event
                                    assert prompt.emit and prompt.exit_reason == event['exit_reason'] == 'emit_dominates'
                                    assert prompt.expansions == event['executor_attempts'] == 0
                                    assert len(spans) == event['candidate_count']
                                    assert event['action_key'] is None
                                    assert math.isclose(prompt.score, event['decision_score'], abs_tol=ATOL, rel_tol=RTOL)
                                records.append(row)
                finally:
                    trainer.model.train(mode)
                assert all(p.grad is grad for p, grad in zip(trainer.model.parameters(), gradients, strict=True))
                after = w.identity(trainer)
                assert before == after and trainer.completed_step == step
                identities.append({'arm': arm, 'step': step, 'before': before, 'after': after, 'zero_updates': True})
                byte_records.append({'arm': arm, 'step': step, 'checkpoint_sha256': checkpoint,
                                     **{scope: metric(*v) for scope, v in totals.items()},
                                     'cases': per_case,
                                     'gate': metric(0, 0, applicable=False) if arm == 'none' else None,
                                     'joint': metric(0, 0, applicable=False) if arm == 'none' else None,
                                     'action_components': 'not_applicable' if arm == 'none' else 'see relation records'})
        validate_population(records, spec)
        ambiguity = []
        histories = [json.loads(line) for line in (RUN/'updates.jsonl').read_text().splitlines()]
        for prefix, group in groups.items():
            if len(group) < 2:
                continue
            first = next(r for r in records if r['prefix_sha256'] == prefix)
            for step in (28,56):
                same = [r for r in records if r['prefix_sha256'] == prefix and r['step'] == step]
                assert all(r['raw'] == same[0]['raw'] for r in same)
            labels = [q.gate_target for q in group]
            weights = [sum(q.case_index in h['history']['batch_ids'] for h in histories if h['arm']=='span_affine_v1') for q in group]
            sets = [{tuple(k) for k in r['correct']['keys']} for r in records if r['step']==56 and r['prefix_sha256']==prefix and r['gate_target']]
            ambiguity.append({'prefix_sha256':prefix, 'prefix_hex':first['prefix_hex'],
                              'members':[{'case':q.case_index,'boundary':q.boundary,'gate':q.gate_target,'presentations':n} for q,n in zip(group,weights,strict=True)],
                              'conflicting_gate':len(set(labels))>1,
                              'common_correct_keys':sorted(set.intersection(*sets)) if sets else None,
                              'case_weighted_gate_error_floor': min(labels.count(0),labels.count(1))/len(labels),
                              'presentation_weighted_gate_error_floor': min(sum(n for n,y in zip(weights,labels,strict=True) if y==0),sum(n for n,y in zip(weights,labels,strict=True) if y==1))/sum(weights)})
        verify_inputs(spec)
        save(output/'queries.json', records)
        save(output/'byte-boundaries.json', byte_boundaries)
        save(output/'summary.json', summarize(records, byte_records))
        save(output/'identities.json', identities)
        save(output/'ambiguity.json', ambiguity)
        save(output/'figure-data.json', {'schema':1, 'uncertainty':None,
             'scope':'in-sample engineering; no qualification',
             'evidence_roles':{'queries':'derived checkpoint scores','history':'observed training','traces':'observed decode','schematic':'schematic'},
             'sources':{str(output/n):file_hash(output/n) for n in ('spec.json','queries.json','summary.json','ambiguity.json','identities.json','byte-boundaries.json')},
             'run_sources':spec['input_files'], 'queries':records,
             'summary':summarize(records,byte_records), 'ambiguity':ambiguity,
             'history':histories, 'traces':traces,
             'row0_geometry': {'prefix_hex':fixture.cases[0].prompt.hex(), 'full_hex':fixture.cases[0].flat.hex()}})
        save(output/'terminal.json', {'status':'complete','optimizer_updates':0,'query_records':len(records),
             'files':{p.name:file_hash(p) for p in output.iterdir() if p.is_file()}})
    except BaseException as exc:
        save(output/'terminal.json', {'status':'failed','type':type(exc).__name__,'error':str(exc)})
        raise
    finally:
        restore_rng_state(caller_rng)
        torch.use_deterministic_algorithms(policy[0], warn_only=policy[1])
    return output


def validate_population(records, spec):
    expected = {(s,q['case'],q['boundary']) for s in spec['steps'] for q in spec['population']}
    actual = [(r['step'],r['case_index'],r['boundary']) for r in records]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('missing, duplicate or foreign query record')
    assert sum(r['gate_target'] for r in records) == 14
    assert sum(r['prompt_subset'] for r in records) == 16
