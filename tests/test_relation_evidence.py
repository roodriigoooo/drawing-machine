"""R0: the canonical digests, the sentinel, and the fail-closed platform rule.

Direction 3's reproducibility verdict had to be retracted because its comparator
was a `torch.save` file hash. The tests here pin the replacement: three digests
over *content*, each one sensitive to the thing it names and blind to the things
it does not, and a rule that refuses a reading rather than reading a missing
field as a pass.

Everything runs at a tiny sentinel shape, which is also the point of one of the
tests: a reading taken at a shape other than the frozen one measures different
kernels and may not qualify a platform.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from dm.eval import relation_contract as contract
from dm.eval import relation_evidence as evidence
from dm.models.transformer import Config, DrawingLM

TINY = evidence.SentinelConfig(vocab_size=64, d_model=32, n_layers=1, n_heads=2,
                               max_len=64, steps=4, batch=4, length=24)


@pytest.fixture()
def model_and_optimizer():
    torch.manual_seed(0)
    model = DrawingLM(Config(vocab_size=32, d_model=16, n_layers=1, n_heads=2,
                             max_len=32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    rows = torch.randint(0, 32, (4, 16))
    logits = model(rows[:, :-1])
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 32),
                                             rows[:, 1:].reshape(-1))
    loss.backward()
    optimizer.step()
    return model, optimizer


# ---------------------------------------------------------------------------
# the digests


def test_the_state_digest_is_a_function_of_the_tensors(model_and_optimizer):
    model, _ = model_and_optimizer
    before = evidence.state_digest(model.state_dict())
    assert before == evidence.state_digest(model.state_dict())
    with torch.no_grad():
        model.norm.weight[0] += 1e-6
    assert evidence.state_digest(model.state_dict()) != before


def test_the_state_digest_sees_a_change_no_float_repr_would_show(model_and_optimizer):
    """Raw bytes, so the smallest representable move is a different digest."""
    model, _ = model_and_optimizer
    before = evidence.state_digest(model.state_dict())
    with torch.no_grad():
        weight = model.norm.weight
        weight[0] = torch.nextafter(weight[0], torch.tensor(float("inf")))
    assert evidence.state_digest(model.state_dict()) != before


def test_the_state_digest_frames_names_so_a_swap_is_visible():
    left = {"a": torch.zeros(2), "b": torch.ones(2)}
    right = {"a": torch.ones(2), "b": torch.zeros(2)}
    assert evidence.state_digest(left) != evidence.state_digest(right)


def test_the_state_digest_distinguishes_shape_from_content():
    assert evidence.state_digest({"a": torch.zeros(4)}) != \
        evidence.state_digest({"a": torch.zeros(2, 2)})


def test_the_optimizer_digest_is_keyed_by_parameter_name(model_and_optimizer):
    """Not by object identity or position: Direction 4 appends a module, and a
    positional key would move the digest of an unchanged optimizer."""
    model, optimizer = model_and_optimizer
    before = evidence.optimizer_digest(optimizer, model)
    assert before == evidence.optimizer_digest(optimizer, model)
    for state in optimizer.state.values():
        state["exp_avg"] += 1e-6
        break
    assert evidence.optimizer_digest(optimizer, model) != before


def test_the_optimizer_digest_sees_a_changed_hyperparameter(model_and_optimizer):
    model, optimizer = model_and_optimizer
    before = evidence.optimizer_digest(optimizer, model)
    optimizer.param_groups[0]["lr"] *= 2
    assert evidence.optimizer_digest(optimizer, model) != before


def test_the_optimizer_digest_sees_a_parameter_moved_between_groups():
    """Same moments, same settings, different group: a different training run."""
    torch.manual_seed(3)
    model = DrawingLM(Config(vocab_size=32, d_model=16, n_layers=1, n_heads=2,
                             max_len=32))
    named = sorted(name for name, _ in model.named_parameters())
    first, second = named[0], named[1]
    lookup = dict(model.named_parameters())

    def digest(left: str, right: str) -> str:
        optimizer = torch.optim.AdamW([
            {"params": [lookup[left]], "lr": 1e-3},
            {"params": [lookup[right]], "lr": 1e-2},
        ])
        return evidence.optimizer_digest(optimizer, model)

    assert digest(first, second) != digest(second, first)


def test_the_optimizer_digest_handles_a_scalar_step_tensor(model_and_optimizer):
    """AdamW's `step` is zero-dimensional and a naive byte view refuses it."""
    model, optimizer = model_and_optimizer
    steps = [state["step"] for state in optimizer.state.values() if "step" in state]
    assert steps and steps[0].dim() == 0
    assert evidence.optimizer_digest(optimizer, model)


def test_the_rng_digest_covers_the_private_data_generator():
    generator = torch.Generator().manual_seed(5)
    before = evidence.rng_digest("cpu", generator)
    torch.rand(4, generator=generator)
    assert evidence.rng_digest("cpu", generator) != before


def test_the_rng_digest_moves_when_any_stream_is_consumed():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    before = evidence.rng_digest()

    random.random()
    after_python = evidence.rng_digest()
    assert after_python != before

    np.random.random()
    after_numpy = evidence.rng_digest()
    assert after_numpy != after_python

    torch.rand(1)
    assert evidence.rng_digest() != after_numpy


def test_the_rng_digest_returns_to_a_restored_state():
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    captured = (random.getstate(), np.random.get_state(),
                torch.random.get_rng_state())
    before = evidence.rng_digest()
    random.random(); np.random.random(); torch.rand(1)
    random.setstate(captured[0])
    np.random.set_state(captured[1])
    torch.random.set_rng_state(captured[2])
    assert evidence.rng_digest() == before


def test_the_three_digests_cannot_collide():
    """Distinct markers, so an optimizer payload can never read as model state."""
    assert len({evidence.STATE_MARKER, evidence.OPTIMIZER_MARKER,
                evidence.RNG_MARKER}) == 3


# ---------------------------------------------------------------------------
# the sentinel


def test_the_sentinel_is_a_function_of_its_seed():
    first = evidence.run_sentinel(TINY, seed=17)
    second = evidence.run_sentinel(TINY, seed=17)
    assert first.digests() == second.digests()
    assert first.final_loss == second.final_loss


def test_the_sentinel_requests_determinism_before_it_measures():
    """The policy has to be on while the readings are taken, not while they are
    written up: an earlier version enabled it after every repeat had finished."""
    torch.use_deterministic_algorithms(False)
    evidence.run_sentinel(TINY, seed=17)
    assert torch.are_deterministic_algorithms_enabled()


def test_the_repeatability_report_records_the_policy_it_ran_under():
    report = evidence.repeatability(TINY, repeats=contract.R0_MIN_REPEATS)
    assert report["determinism"]["requested"]
    assert report["determinism"]["deterministic_algorithms"]


def test_two_seeds_are_two_runs():
    assert evidence.run_sentinel(TINY, seed=17).digests() != \
        evidence.run_sentinel(TINY, seed=18).digests()


def test_the_sentinel_reseeds_rather_than_continuing_a_stream():
    """Two calls in one process are two repeats, not two points on one stream."""
    torch.rand(1000)
    first = evidence.run_sentinel(TINY, seed=17)
    torch.rand(1000)
    assert evidence.run_sentinel(TINY, seed=17).digests() == first.digests()


def test_only_the_frozen_shape_claims_to_be_the_frozen_shape():
    assert not TINY.is_frozen_shape
    assert evidence.SentinelConfig().is_frozen_shape


def test_repeatability_refuses_too_few_repeats():
    with pytest.raises(ValueError, match="at least"):
        evidence.repeatability(TINY, repeats=2)


def test_repeatability_reports_exact_reproduction_on_cpu():
    report = evidence.repeatability(TINY, repeats=contract.R0_MIN_REPEATS)
    assert report["exact_state_reproduction"]
    assert report["distinct_digests"] == {"model_state": 1, "optimizer_state": 1,
                                          "rng_state": 1}
    assert report["environment"]["device"] == "cpu"


def test_an_interrupted_run_resumes_to_the_same_state():
    report = evidence.resume_equivalence(TINY, at=2)
    assert report["equivalent"], report["differing_digests"]
    assert report["resume_step"] == 2


def test_the_resume_check_really_serialises_and_rebuilds():
    """An in-memory handback into the same objects exercises nothing."""
    report = evidence.resume_equivalence(TINY, at=2)
    assert report["serialised"] and report["rebuilt_model_and_optimizer"]
    assert report["process_boundary"] is False, (
        "still missing, and named rather than implied")


def test_resume_equivalence_names_which_digest_moved():
    """All three are compared, because final model equality alone hides two faults."""
    report = evidence.resume_equivalence(TINY, at=2)
    assert set(report["uninterrupted"]) == {"model_state", "optimizer_state",
                                            "rng_state"}


def test_a_resume_that_forgets_the_data_generator_is_caught():
    """The private generator produces every batch; four matching digests without
    it would still be a different corpus order."""
    seed = 21
    generator = evidence._seed_everything(seed)
    model, optimizer = evidence._build(TINY, "cpu")
    evidence._train(model, optimizer, generator, TINY, "cpu", 2)
    with_generator = evidence.rng_digest("cpu", generator)
    torch.rand(1, generator=generator)
    assert evidence.rng_digest("cpu", generator) != with_generator
    assert evidence.rng_digest("cpu") != with_generator


# ---------------------------------------------------------------------------
# the rule


def _reading(**overrides) -> dict:
    sentinel = dict(vars(evidence.SentinelConfig()))
    environment = evidence.execution_environment(device="cpu")
    readings = [{
        "model_state": "a", "optimizer_state": "a", "rng_state": "a",
        "final_loss": 0.0, "seconds": 0.0,
    }] * contract.R0_MIN_REPEATS
    body = {
        "schema": evidence.EVIDENCE_SCHEMA,
        "device": "cpu",
        "sentinel": sentinel,
        "repeats": contract.R0_MIN_REPEATS,
        "seed": contract.seed_for("sentinel"),
        "readings": readings,
        "exact_state_reproduction": True,
        "distinct_digests": {"model_state": 1, "optimizer_state": 1,
                             "rng_state": 1},
        "digests": {name: ["a"] for name in contract.R0_DIGESTS},
        "determinism": {"requested": True, "warn_only": True,
                        "deterministic_algorithms": True,
                        "is_policy_not_certificate": True},
        "environment": environment,
    }
    return {**body, **overrides}


def _resume(**overrides) -> dict:
    environment = evidence.execution_environment(device="cpu")
    digests = {name: "a" for name in contract.R0_DIGESTS}
    body = {
        "schema": evidence.EVIDENCE_SCHEMA, "device": "cpu",
        "seed": contract.seed_for("sentinel"),
        "sentinel": dict(vars(evidence.SentinelConfig())),
        "environment": environment,
        "resume_step": contract.R0_RESUME_STEP,
        "serialised": True, "rebuilt_model_and_optimizer": True,
        "process_boundary": False, "uninterrupted": digests,
        "resumed": dict(digests), "differing_digests": [],
        "equivalent": True, "loss_difference": 0.0,
    }
    return {**body, **overrides}


def test_exact_reproduction_qualifies_outright():
    decision = evidence.qualify(_reading(), _resume())
    assert decision["qualified"], decision["problems"]
    assert decision["route"] == "exact_state_reproduction"


def test_a_reading_at_the_wrong_shape_cannot_qualify_a_platform():
    decision = evidence.qualify(
        _reading(sentinel={"is_frozen_shape": False}), _resume())
    assert not decision["qualified"]
    assert any("sentinel shape" in problem for problem in decision["problems"])


def test_there_is_no_metric_level_fallback_route():
    """The implemented one was degenerate, so a backend that cannot reproduce
    state exactly now fails outright."""
    decision = evidence.qualify(
        _reading(exact_state_reproduction=False,
                 digests={"model_state": ["a", "b"], "optimizer_state": ["a"],
                          "rng_state": ["a"]}),
        _resume())
    assert not decision["qualified"]
    assert decision["route"] == "exact_state_reproduction"
    assert any("no metric-level fallback" in problem
               for problem in decision["problems"])
    assert "withdrawn in v1" in decision["fallback_route"]


def test_a_qualification_needs_the_frozen_repeat_count():
    decision = evidence.qualify(_reading(repeats=3, readings=[{}] * 3), _resume())
    assert not decision["qualified"]
    assert any("repeats" in problem for problem in decision["problems"])


def test_a_report_claiming_exactness_without_evidence_is_refused():
    """The fault the previous rule had: absent is not the same as satisfied."""
    decision = evidence.qualify(
        {"schema": evidence.EVIDENCE_SCHEMA, "exact_state_reproduction": True},
        _resume())
    assert not decision["qualified"]
    problems = " ".join(decision["problems"])
    assert "repeats" in problems, "a missing repeat count must be named"
    assert "must carry all of" in problems, "missing digest sets must be named"
    assert "determinism" in problems


def test_a_reading_that_did_not_record_its_determinism_policy_is_refused():
    decision = evidence.qualify(_reading(determinism={}), _resume())
    assert not decision["qualified"]
    assert any("determinism" in problem for problem in decision["problems"])


def test_a_reading_whose_readings_do_not_match_its_repeat_count_is_refused():
    decision = evidence.qualify(_reading(readings=[{}]), _resume())
    assert not decision["qualified"]
    assert any("readings" in problem for problem in decision["problems"])


def test_a_run_that_cannot_resume_fails_even_when_the_repeats_agreed():
    decision = evidence.qualify(
        _reading(), _resume(equivalent=False,
                            differing_digests=["optimizer_state"]))
    assert not decision["qualified"]
    assert any("resume" in problem for problem in decision["problems"])


def test_raw_readings_override_self_reported_digest_summaries():
    reading = {"model_state": "different", "optimizer_state": "a",
               "rng_state": "a", "final_loss": 0.0, "seconds": 0.0}
    decision = evidence.qualify(
        _reading(readings=[reading] * contract.R0_MIN_REPEATS), _resume())
    assert not decision["qualified"]
    assert any("raw readings" in problem for problem in decision["problems"])


def test_a_resume_map_cannot_claim_equivalence_when_it_differs():
    resumed = {"model_state": "different", "optimizer_state": "a",
               "rng_state": "a"}
    decision = evidence.qualify(
        _reading(), _resume(resumed=resumed,
                            differing_digests=["model_state"], equivalent=False))
    assert not decision["qualified"]
    assert any("resume" in problem for problem in decision["problems"])


def test_both_reports_must_use_the_frozen_seed():
    decision = evidence.qualify(_reading(seed=123), _resume(seed=123))
    assert not decision["qualified"]
    assert any("frozen sentinel seed" in problem
               for problem in decision["problems"])


def test_a_resume_that_did_not_serialise_cannot_qualify():
    decision = evidence.qualify(_reading(), _resume(serialised=False))
    assert not decision["qualified"]
    assert any("serialise" in problem for problem in decision["problems"])


def test_a_malformed_reading_fails_closed():
    decision = evidence.qualify({}, _resume())
    assert not decision["qualified"]
    assert decision["route"] is None


def test_the_decision_names_the_comparator_it_refuses_to_use():
    decision = evidence.qualify(_reading(), _resume())
    assert decision["forbidden_comparator"] == contract.R0_FORBIDDEN_DIGEST
    assert decision["determinism_is_policy_not_certificate"]


def test_determinism_is_reported_as_policy_and_never_as_a_certificate():
    granted = evidence.deterministic_execution()
    assert granted["requested"] and granted["is_policy_not_certificate"]
