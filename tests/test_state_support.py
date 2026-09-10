from __future__ import annotations

import math

import numpy as np
import torch

from dm.eval.state_support import GenerationMass, _Totals, mass_by_rule
from dm.isa.codec import ByteCodec
from dm.isa.state import (
    CANONICAL,
    VM_SAFE,
    GrammarState,
    LanguagePolicy,
    LegacyStateMonitor,
    SupportTable,
    SymbolCursor,
)


def _policy():
    return LanguagePolicy.of(["HALT", "MOVE", "LINE", "REPEAT", "ENDREP"])


def test_exclusive_mass_and_stage_costs_reconcile():
    codec = ByteCodec()
    policy = _policy()
    state = GrammarState(policy)
    key = (state.key(), (0, 0))
    probs = np.linspace(1.0, 2.0, codec.vocab_size, dtype=np.float64)
    probs /= probs.sum()
    table = SupportTable(codec, policy, CANONICAL)
    reading = mass_by_rule(probs[None, :], table, [key])
    union = sum(float(values[0]) for values in reading["by_rule"].values())
    assert math.isclose(union, float(reading["q"][0]), rel_tol=0, abs_tol=1e-12)
    stage_total = sum(float(values[0])
                      for values in reading["stage_renorm_bits"].values())
    assert math.isclose(stage_total, float(reading["renorm_bits"][0]),
                        rel_tol=0, abs_tol=1e-12)

    totals = _Totals(1)
    totals.add(0, reading, [key], [state.key().stratum])
    report = totals.as_dict(1, CANONICAL)
    assert "q_std" in report and "renorm_bits_p95" in report
    assert math.isclose(
        report["renorm_bits_per_drawing"],
        sum(report["stage_renorm_bits_per_drawing"].values()),
        rel_tol=0, abs_tol=1e-12,
    )


def test_generation_mass_reports_both_support_levels():
    codec = ByteCodec()
    policy = _policy()

    class Monitor:
        def __init__(self):
            self.codec = codec
            self.policy = policy
            self.states = [GrammarState(policy)]
            self.cursors = [SymbolCursor(codec.stride)]
            self.done = np.array([False])

        @property
        def diagnostic_done(self):
            return self.done

    monitor = Monitor()
    mass = GenerationMass(monitor)
    mass(0, torch.zeros(1, codec.vocab_size))
    result = mass.as_dict()
    assert set(result["levels"]) == {VM_SAFE, CANONICAL}
    for level in result["levels"].values():
        assert level["q_std"] >= 0
        assert "renorm_bits_p99" in level


def test_reported_spread_uses_the_sample_denominator():
    totals = _Totals(2)
    totals.q = [0.0, 2.0]
    totals.renorm_values = [0.0, 2.0]
    result = totals.as_dict(2, CANONICAL)
    assert math.isclose(result["q_std"], math.sqrt(2.0))
    assert math.isclose(result["renorm_bits_std"], math.sqrt(2.0))


def test_generated_mass_keeps_a_sub_float32_illegal_tail():
    codec = ByteCodec()
    policy = LanguagePolicy.of(["HALT"])

    class Monitor:
        def __init__(self):
            self.codec = codec
            self.policy = policy
            self.states = [GrammarState(policy)]
            self.cursors = [SymbolCursor(codec.stride)]
            self.done = np.array([False])

    logits = torch.full((1, codec.vocab_size), -25.0)
    logits[0, 2] = 0.0  # ByteCodec symbol 2 is byte 0 / HALT.
    mass = GenerationMass(Monitor(), level=CANONICAL)
    mass(0, logits)
    assert mass.as_dict()["q_mean"] > 0.0


def test_generated_mass_stops_at_the_diagnostic_terminal_state_only():
    codec = ByteCodec()
    policy = _policy()
    monitor = LegacyStateMonitor(codec, 1, policy)
    mass = GenerationMass(monitor, level=CANONICAL)
    logits = torch.zeros(1, codec.vocab_size)
    mass(0, logits)
    endrep_symbol = codec.encode(bytes([0x08]))[0]
    monitor.step(np.array([[endrep_symbol]], dtype=np.int64))
    assert not monitor.done[0]
    assert monitor.diagnostic_done[0]
    mass(1, logits)
    assert mass.as_dict()["decode_steps"] == 1
