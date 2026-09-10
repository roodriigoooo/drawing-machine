"""Harness substitution restores safety even on interrupted measurements."""
import pytest

import dm.train_relation as training_module
from scripts import relation_numerical_guard_benchmark as benchmark


def test_interrupted_trajectory_restores_original_guard(monkeypatch):
    original = training_module._validate_updated_numerics
    failure = KeyboardInterrupt("measurement interrupted")

    def interrupt(self):
        raise failure

    monkeypatch.setattr(training_module.RelationTrainer, "step", interrupt)
    with pytest.raises(KeyboardInterrupt) as caught:
        benchmark.trajectory(benchmark._training_config("none", 1), benchmark._fixture(),
                             training_module.capture_rng_state(), scan=False)
    assert caught.value is failure
    assert training_module._validate_updated_numerics is original


@pytest.mark.parametrize("arm", ["none", "span_affine_v1"])
@pytest.mark.parametrize("budget", [1, 24_000_000])
def test_scan_exact_transparency(arm, budget):
    config, training = benchmark._training_config(arm, budget), benchmark._fixture()
    rng = training_module.capture_rng_state()
    on, on_evidence = benchmark.trajectory(config, training, rng, scan=True)
    off, off_evidence = benchmark.trajectory(config, training, rng, scan=False)
    assert on_evidence == off_evidence
    assert len(on["scan_seconds"]) == 7
    assert off["scan_seconds"] == []
