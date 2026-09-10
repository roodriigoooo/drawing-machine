"""The frozen Direction 2 decision rule, and every way it must fail closed.

The gate is written and tested *before* C3 runs, which is the point: a decision
procedure chosen after seeing an effect is not a decision procedure. Fixtures
here are synthetic reports, so the labels are checked against arithmetic rather
than against a checkpoint.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import context_gate

SESOI = 0.02
STEP = "synthetic_flat_step"
SHAPE = "composed_shape_copy2"
ROTATED = "synthetic_flat_step_d4r1"
YAXIS = "synthetic_flat_step_yaxis"


def _summary(delta: float, low: float, high: float, p: float,
             *, raw: float | None = None, block_low: float = 0.05,
             prefix: float = 2.16, matched: float = 0.20) -> dict:
    return {
        "n_cases": 64,
        "prefix_bits_per_byte_mean": prefix,
        "matched_target_bits_per_byte_mean": matched,
        "D_bits_per_byte_mean": delta if raw is None else raw,
        "D_minus_control_bits_per_byte_mean": delta,
        "D_minus_block_control_bits_per_byte_mean": delta,
        "bootstrap_D_minus_control": {"ci95": [low, high], "p_value": p,
                                      "clusters": 64},
        "bootstrap_D_minus_block_control": {"ci95": [block_low, block_low + 0.1],
                                            "p_value": p, "clusters": 64},
    }


def _report(tmp_path, name: str, venues: dict, *, checks: dict | None = None,
            status: str = "complete", manifest_sha: str = "sha-step") -> str:
    body = {
        "report_schema": 3,
        "status": status,
        "manifest": f"runs/{name}.json",
        "manifest_sha256": manifest_sha,
        "gate_input": {"instrument_checks": (
            {"all_present": True} if checks is None else checks)},
        "reports": [
            {"name": f"{name}-s{seed}", "model_seed": seed,
             "checkpoint": f"runs/{name}_s{seed}.pt",
             "checkpoint_sha256": str(seed) * 64,
             "record_sha256": str(seed + 2) * 64,
             "venues": {venue: summary(seed)
                        for venue, summary in venues.items()}}
            for seed in (0, 1)
        ],
    }
    path = tmp_path / f"{name}_report.json"
    path.write_text(json.dumps(body))
    return str(path)


def _protocol(tmp_path, *, family=(STEP, SHAPE)) -> str:
    body = {
        "status": "frozen",
        "context_c3": {
            "co_primary_venues": list(family),
            "axis_diagnostic": {STEP: [ROTATED, YAXIS]},
            "axis_support": {"max_prefix_cost_ratio": 1.5,
                             "max_matched_target_cost_ratio": 5.0},
            "sesoi_bits_per_target_byte": SESOI,
            "alpha": 0.05,
            "required_instruments": ["all_present"],
            "manifests": {
                "runs/step.json": {
                    "manifest_sha256": "sha-step",
                    "checkpoints": ["runs/step_s0.pt", "runs/step_s1.pt"],
                    "venues": [STEP],
                },
                "runs/shape.json": {
                    "manifest_sha256": "sha-shape",
                    "checkpoints": ["runs/shape_s0.pt", "runs/shape_s1.pt"],
                    "venues": [SHAPE],
                },
            },
        }
    }
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(body))
    return str(path)


def _run(tmp_path, monkeypatch, reports: list[str], protocol: str,
         c4: list[str] | None = None) -> dict:
    out = tmp_path / "gate.json"
    monkeypatch.setattr(
        "sys.argv",
        ["context_gate.py", *reports, "--protocol", protocol, "--out", str(out),
         "--overwrite", *(["--c4", *c4] if c4 else [])],
    )
    assert context_gate.main() == 0
    return json.loads(out.read_text())


def _c4_summary(d_gen: float, low: float, high: float, *,
                delta: tuple[float, float, float] | None = None) -> dict:
    """One venue's C4 summary. `delta` adds the schema-5 control columns."""
    body = {
        "n_cases": 64,
        "D_gen_mean": d_gen,
        "bootstrap_D_gen": {"ci95": [low, high], "p_value": 0.001,
                            "clusters": 64},
        "half_width": (high - low) / 2,
        "reach_rate": 0.84,
        "hit_own_rate": 0.14,
        "hit_other_rate": 0.0,
        "relation_consistent_rate": 0.14,
        "canonical_rate": 0.99,
    }
    if delta is not None:
        value, dlow, dhigh = delta
        body |= {
            "D_gen_target_control_mean": 0.0,
            "D_gen_block_control_mean": d_gen - value,
            "D_gen_minus_block_control_mean": value,
            "bootstrap_D_gen_minus_block_control": {
                "ci95": [dlow, dhigh], "p_value": 0.001, "clusters": 64},
        }
    return body


def _c4_report(tmp_path, name: str, venues: dict, *, status: str = "complete",
               manifest_sha: str = "sha-step") -> str:
    body = {
        "report_schema": 4,
        "status": status,
        "manifest": f"runs/{name}.json",
        "manifest_sha256": manifest_sha,
        "reports": [
            {"name": f"{name}-s{seed}", "model_seed": seed,
             "checkpoint": f"runs/{name}_s{seed}.pt",
             "checkpoint_sha256": str(seed) * 64,
             "record_sha256": str(seed + 2) * 64,
             "venues": venues}
            for seed in (0, 1)
        ],
    }
    path = tmp_path / f"{name}_c4.json"
    path.write_text(json.dumps(body))
    return str(path)


def _positive(delta: float, p: float = 0.001, **kwargs):
    return lambda _seed: _summary(delta, delta / 2, delta * 2, p, **kwargs)


def test_a_controlled_effect_in_both_seeds_is_relational_context_use(
        tmp_path, monkeypatch):
    step = _report(tmp_path, "step",
                   {STEP: _positive(0.06), ROTATED: _positive(0.05)})
    shape = _report(tmp_path, "shape", {SHAPE: _positive(0.07)},
                    manifest_sha="sha-shape")
    result = _run(tmp_path, monkeypatch, [step, shape], _protocol(tmp_path))
    assert result["status"] == "complete"
    assert result["labels"] == {STEP: "relational_context_use",
                                SHAPE: "relational_context_use"}
    assert result["holm"][STEP]["reject"] and result["holm"][SHAPE]["reject"]


def test_an_x_only_effect_is_named_anisotropic(tmp_path, monkeypatch):
    step = _report(
        tmp_path, "step",
        {STEP: _positive(0.06),
         # The diagnostic's interval spans zero *and* its prefixes cost what the
         # primary's do, so the model could see and the relation did not
         # transfer. That is anisotropy.
         YAXIS: lambda _seed: _summary(0.001, -0.02, 0.03, 0.6,
                                       prefix=2.4, matched=0.5)},
    )
    shape = _report(tmp_path, "shape", {SHAPE: _positive(0.07)},
                    manifest_sha="sha-shape")
    result = _run(tmp_path, monkeypatch, [step, shape], _protocol(tmp_path))
    assert result["labels"][STEP] == "anisotropic_relation_use"
    assert result["labels"][SHAPE] == "relational_context_use"


def test_an_effect_below_the_sesoi_is_only_prefix_sensitivity(tmp_path,
                                                              monkeypatch):
    step = _report(tmp_path, "step",
                   {STEP: _positive(0.004), ROTATED: _positive(0.004)})
    shape = _report(tmp_path, "shape", {SHAPE: _positive(0.07)},
                    manifest_sha="sha-shape")
    result = _run(tmp_path, monkeypatch, [step, shape], _protocol(tmp_path))
    assert result["labels"][STEP] == "causal_prefix_sensitivity"


def test_a_raw_effect_that_dies_under_the_control_is_lower_order(tmp_path,
                                                                monkeypatch):
    step = _report(
        tmp_path, "step",
        {STEP: lambda _seed: _summary(0.001, -0.05, 0.05, 0.9, raw=1.23)},
    )
    shape = _report(tmp_path, "shape", {SHAPE: _positive(0.07)},
                    manifest_sha="sha-shape")
    result = _run(tmp_path, monkeypatch, [step, shape], _protocol(tmp_path))
    # This is the retracted schema-1 finding in one word: a large raw
    # compatibility preference with no surviving controlled effect.
    assert result["labels"][STEP] == "lower_order_context_use"


def test_a_null_in_both_seeds_is_no_controlled_evidence(tmp_path, monkeypatch):
    step = _report(tmp_path, "step",
                   {STEP: lambda _seed: _summary(0.0005, -0.03, 0.03, 0.8)})
    shape = _report(tmp_path, "shape",
                    {SHAPE: lambda _seed: _summary(0.0004, -0.02, 0.02, 0.7)},
                    manifest_sha="sha-shape")
    result = _run(tmp_path, monkeypatch, [step, shape], _protocol(tmp_path))
    assert set(result["labels"].values()) == {"no_controlled_evidence"}


def test_a_missing_co_primary_venue_is_incomplete(tmp_path, monkeypatch):
    step = _report(tmp_path, "step", {STEP: _positive(0.06)})
    result = _run(tmp_path, monkeypatch, [step], _protocol(tmp_path))
    assert result["status"] == "incomplete"
    assert result["labels"][SHAPE] == "incomplete"
    assert any("missing co-primary venues" in problem
               for problem in result["problems"])


def test_a_failed_instrument_or_unfrozen_manifest_is_incomplete(tmp_path,
                                                                monkeypatch):
    step = _report(tmp_path, "step", {STEP: _positive(0.06)},
                   checks={"all_present": False})
    shape = _report(tmp_path, "shape", {SHAPE: _positive(0.07)},
                    manifest_sha="not-the-frozen-one")
    result = _run(tmp_path, monkeypatch, [step, shape], _protocol(tmp_path))
    assert result["status"] == "incomplete"
    assert set(result["labels"].values()) == {"incomplete"}
    assert any("all_present" in problem for problem in result["problems"])
    assert any("not the frozen one" in problem for problem in result["problems"])


def test_missing_instrument_map_and_checkpoint_are_incomplete(tmp_path, monkeypatch):
    step = Path(_report(tmp_path, "step", {STEP: _positive(0.06)}, checks={}))
    body = json.loads(step.read_text())
    body["reports"] = body["reports"][:1]
    step.write_text(json.dumps(body))
    shape = _report(tmp_path, "shape", {SHAPE: _positive(0.07)},
                    manifest_sha="sha-shape")
    result = _run(tmp_path, monkeypatch, [str(step), shape], _protocol(tmp_path))
    assert result["status"] == "incomplete"
    assert set(result["labels"].values()) == {"incomplete"}
    assert any("all_present missing or failed" in problem
               for problem in result["problems"])
    assert any("checkpoints" in problem for problem in result["problems"])


def test_duplicate_seed_and_malformed_rows_fail_closed(tmp_path, monkeypatch):
    step = Path(_report(tmp_path, "step", {STEP: _positive(0.06)}))
    body = json.loads(step.read_text())
    body["reports"][1]["model_seed"] = 0
    step.write_text(json.dumps(body))
    shape = Path(_report(tmp_path, "shape", {SHAPE: _positive(0.07)},
                         manifest_sha="sha-shape"))
    malformed = json.loads(shape.read_text())
    del malformed["reports"]
    shape.write_text(json.dumps(malformed))
    result = _run(tmp_path, monkeypatch, [str(step), str(shape)],
                  _protocol(tmp_path))
    assert result["status"] == "incomplete"
    assert any("model seeds are missing or duplicated" in problem
               for problem in result["problems"])
    assert any("malformed C3 result columns" in problem
               for problem in result["problems"])


def test_the_protocol_cannot_promote_a_diagnostic_venue(tmp_path, monkeypatch):
    step = _report(tmp_path, "step", {ROTATED: _positive(0.06)})
    protocol = _protocol(tmp_path, family=(ROTATED,))
    result = _run(tmp_path, monkeypatch, [step], protocol)
    assert result["status"] == "incomplete"
    assert any("non-co-primary" in problem for problem in result["problems"])


# ---------------------------------------------------------------------------
# C4: the generation verdict


def _c3_pair(tmp_path):
    return [_report(tmp_path, "step", {STEP: _positive(0.06)}),
            _report(tmp_path, "shape", {SHAPE: _positive(0.07)},
                    manifest_sha="sha-shape")]


def test_no_c4_report_is_not_run_rather_than_a_null(tmp_path, monkeypatch):
    """A diagnostic that has not been run is not a diagnostic that failed."""
    result = _run(tmp_path, monkeypatch, _c3_pair(tmp_path), _protocol(tmp_path))
    assert result["generation_labels"] == {STEP: "not_run", SHAPE: "not_run"}
    assert result["controlled_generation_labels"] == {STEP: "not_run",
                                                      SHAPE: "not_run"}


def test_an_interval_excluding_zero_in_both_seeds_is_generation_robust(
        tmp_path, monkeypatch):
    c4 = [_c4_report(tmp_path, "step", {STEP: _c4_summary(0.084, 0.074, 0.094)}),
          _c4_report(tmp_path, "shape", {SHAPE: _c4_summary(0.035, 0.019, 0.049)},
                     manifest_sha="sha-shape")]
    result = _run(tmp_path, monkeypatch, _c3_pair(tmp_path), _protocol(tmp_path),
                  c4=c4)
    assert result["generation_labels"][STEP] == "generation_robust_context_use"
    assert result["generation_labels"][SHAPE] == "generation_robust_context_use"
    # No control column in a v4-schema report, and the gate says exactly that
    # rather than treating the absence as a pass.
    assert result["controlled_generation_labels"][STEP] == "not_measured"


def test_an_interval_spanning_zero_is_exposure_limited(tmp_path, monkeypatch):
    """The label §4.1 exists to assign: teacher-forced, then lost in generation."""
    c4 = [_c4_report(tmp_path, "step", {STEP: _c4_summary(0.001, -0.01, 0.02)}),
          _c4_report(tmp_path, "shape", {SHAPE: _c4_summary(0.035, 0.019, 0.049)},
                     manifest_sha="sha-shape")]
    result = _run(tmp_path, monkeypatch, _c3_pair(tmp_path), _protocol(tmp_path),
                  c4=c4)
    assert result["generation_labels"][STEP] == "exposure_limited_context_use"
    assert result["generation_labels"][SHAPE] == "generation_robust_context_use"


def test_one_seed_short_of_the_interval_is_not_robust(tmp_path, monkeypatch):
    """Both seeds are required, exactly as they are for the C3 label."""
    path = tmp_path / "step_c4.json"
    body = {
        "status": "complete", "manifest": "runs/step.json",
        "manifest_sha256": "sha-step",
        "report_schema": 4,
        "reports": [
            {"name": "step-s0", "model_seed": 0, "checkpoint": "runs/step_s0.pt",
             "checkpoint_sha256": "0" * 64, "record_sha256": "2" * 64,
             "venues": {STEP: _c4_summary(0.084, 0.074, 0.094)}},
            {"name": "step-s1", "model_seed": 1, "checkpoint": "runs/step_s1.pt",
             "checkpoint_sha256": "1" * 64, "record_sha256": "3" * 64,
             "venues": {STEP: _c4_summary(0.004, -0.01, 0.02)}},
        ],
    }
    path.write_text(json.dumps(body))
    result = _run(tmp_path, monkeypatch, _c3_pair(tmp_path), _protocol(tmp_path),
                  c4=[str(path)])
    assert result["generation_labels"][STEP] == "exposure_limited_context_use"


def test_a_reversed_generation_preference_has_its_own_name(tmp_path, monkeypatch):
    c4 = [_c4_report(tmp_path, "step", {STEP: _c4_summary(-0.05, -0.08, -0.02)})]
    result = _run(tmp_path, monkeypatch, _c3_pair(tmp_path), _protocol(tmp_path),
                  c4=c4)
    assert result["generation_labels"][STEP] == "generation_reversed"


def test_the_block_control_can_take_back_a_positive_d_gen(tmp_path, monkeypatch):
    """The reason C4 grew a control at all.

    A model that lurches at any prefix edit produces the same `D_gen` as one
    that follows the relation. Only `Delta_gen` separates them, and the gate has
    to be able to report the separation as a *negative*.
    """
    c4 = [
        _c4_report(tmp_path, "step",
                   {STEP: _c4_summary(0.084, 0.074, 0.094,
                                      delta=(0.001, -0.01, 0.02))}),
        _c4_report(tmp_path, "shape",
                   {SHAPE: _c4_summary(0.035, 0.019, 0.049,
                                       delta=(0.030, 0.014, 0.046))},
                   manifest_sha="sha-shape"),
    ]
    result = _run(tmp_path, monkeypatch, _c3_pair(tmp_path), _protocol(tmp_path),
                  c4=c4)
    # Both are "robust" on the raw contrast ...
    assert set(result["generation_labels"].values()) == {
        "generation_robust_context_use"}
    # ... and only one of them survives the matched irrelevant edit.
    assert result["controlled_generation_labels"][STEP] == (
        "no_controlled_generation_evidence")
    assert result["controlled_generation_labels"][SHAPE] == (
        "controlled_generation_context_use")


def test_a_c4_report_from_another_manifest_is_incomplete(tmp_path, monkeypatch):
    c4 = [_c4_report(tmp_path, "step", {STEP: _c4_summary(0.084, 0.074, 0.094)},
                     manifest_sha="not-the-frozen-one")]
    result = _run(tmp_path, monkeypatch, _c3_pair(tmp_path), _protocol(tmp_path),
                  c4=c4)
    assert result["status"] == "incomplete"
    assert set(result["generation_labels"].values()) == {"incomplete"}
    assert any("C4 manifest is not the frozen one" in problem
               for problem in result["problems"])


def test_an_unfinished_c4_run_cannot_report_a_verdict(tmp_path, monkeypatch):
    c4 = [_c4_report(tmp_path, "step", {STEP: _c4_summary(0.084, 0.074, 0.094)},
                     status="running")]
    result = _run(tmp_path, monkeypatch, _c3_pair(tmp_path), _protocol(tmp_path),
                  c4=c4)
    assert result["status"] == "incomplete"
    assert any("C4 status=" in problem for problem in result["problems"])


@pytest.mark.parametrize("name", ["context-protocol-v1.json",
                                  "context-protocol-v2.json",
                                  "context-protocol-v3.json",
                                  "context-protocol-v4.json",
                                  "context-protocol-v5.json"])
def test_every_committed_protocol_stays_readable(name):
    """Old freezes are records. They keep parsing, and they keep their hashes.

    A superseded protocol is what says which manifest authorised which run, so
    it is never edited and never deleted -- `docs/state-protocol-v2.json` is the
    same rule one direction back.
    """
    protocol = json.loads(Path("docs") .joinpath(name).read_text())
    frozen = protocol["context_c3"]
    assert protocol["status"] == "frozen"
    assert frozen["co_primary_venues"] == [STEP, SHAPE]
    assert frozen["sesoi_bits_per_target_byte"] == SESOI
    for path, entry in frozen["manifests"].items():
        assert len(entry["manifest_sha256"]) == 64, path
        assert len(entry["checkpoints"]) == 2, path
        for venue in entry["venues"]:
            assert entry["venue_case_counts"][venue] >= (
                frozen["min_venue_yield"][venue]), (path, venue)
    assert frozen["builder"]["manifest_schema"] == 3


def test_a_driver_refuses_source_drift_at_runtime(tmp_path):
    """Freeze enforcement belongs in driver, not permanent repository state.

    Historical protocols must keep naming historical code after repository
    evolves. Selected protocol still cannot authorise current code unless every
    required source digest matches.
    """
    from dm.eval.provenance import verify_frozen_sources

    source = tmp_path / "instrument.py"
    source.write_text("frozen\n")
    protocol = {"source_sha256": {
        str(source): hashlib.sha256(source.read_bytes()).hexdigest(),
    }}
    verify_frozen_sources(protocol, (source,))
    source.write_text("drifted\n")
    with pytest.raises(ValueError, match="source digest mismatch"):
        verify_frozen_sources(protocol, (source,))


def test_the_axis_precondition_was_corrected_and_says_so():
    """v3 drops a bound that conditioned on the outcome, and labels itself.

    The v2 bound made the y-axis diagnostic unreadable exactly when the model
    failed to transfer -- the one case it exists to detect. Correcting that
    after seeing the data is legitimate; quoting the corrected reading as
    confirmatory is not, and the protocol has to carry that distinction rather
    than leave it to a reader.
    """
    v2 = json.loads(Path("docs/context-protocol-v2.json").read_text())
    v3 = json.loads(Path("docs/context-protocol-v3.json").read_text())
    assert "max_matched_target_cost_ratio" in v2["context_c3"]["axis_support"]
    assert "max_matched_target_cost_ratio" not in v3["context_c3"]["axis_support"]
    assert v3["context_c3"]["axis_support"]["max_prefix_cost_ratio"] == (
        v2["context_c3"]["axis_support"]["max_prefix_cost_ratio"])
    assert v3["supersedes"] == "context-v2"
    assert "exploratory" in v3["context_c3"]["axis_reading_status"]
    assert "EXPLORATORY" in v3["note"]


@pytest.mark.parametrize("field", ["co_primary_venues", "manifests", "alpha"])
def test_the_gate_refuses_a_protocol_missing_a_frozen_field(tmp_path, monkeypatch,
                                                            field):
    body = json.loads(Path(_protocol(tmp_path)).read_text())
    del body["context_c3"][field]
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(body))
    step = _report(tmp_path, "step", {STEP: _positive(0.06)})
    monkeypatch.setattr(
        "sys.argv",
        ["context_gate.py", step, "--protocol", str(path), "--out",
         str(tmp_path / "gate.json"), "--overwrite"],
    )
    with pytest.raises(KeyError):
        context_gate.main()


def test_the_gate_refuses_an_unfrozen_protocol(tmp_path, monkeypatch):
    body = json.loads(Path(_protocol(tmp_path)).read_text())
    body["status"] = "draft"
    path = tmp_path / "draft.json"
    path.write_text(json.dumps(body))
    step = _report(tmp_path, "step", {STEP: _positive(0.06)})
    monkeypatch.setattr(
        "sys.argv",
        ["context_gate.py", step, "--protocol", str(path), "--out",
         str(tmp_path / "gate.json"), "--overwrite"],
    )
    with pytest.raises(ValueError, match="not frozen"):
        context_gate.main()


def test_an_out_of_support_axis_diagnostic_cannot_report_anisotropy(tmp_path,
                                                                    monkeypatch):
    """A null on prefixes the model cannot predict is not a failure to transfer.

    The exact quarter turn moves every byte of the program; measured, it costs
    the model 2.3x its own prefix rate and 27x its matched-target rate. Reading
    that null as anisotropy is the same category error as reading a missing
    control as a pass.
    """
    step = _report(
        tmp_path, "step",
        {STEP: _positive(0.06),
         ROTATED: lambda _seed: _summary(0.001, -0.02, 0.03, 0.6,
                                         prefix=4.98, matched=5.51)},
    )
    shape = _report(tmp_path, "shape", {SHAPE: _positive(0.07)},
                    manifest_sha="sha-shape")
    result = _run(tmp_path, monkeypatch, [step, shape], _protocol(tmp_path))
    verdict = result["axis_support"][ROTATED]
    assert verdict["in_support"] is False
    assert verdict["seeds"][0]["prefix_cost_ratio"] == pytest.approx(2.3055, abs=1e-3)
    assert result["labels"][STEP] == "relational_context_use"


def test_a_readable_axis_diagnostic_that_transfers_leaves_the_label_alone(
        tmp_path, monkeypatch):
    step = _report(
        tmp_path, "step",
        {STEP: _positive(0.06),
         YAXIS: _positive(0.05, prefix=2.5, matched=0.6)},
    )
    shape = _report(tmp_path, "shape", {SHAPE: _positive(0.07)},
                    manifest_sha="sha-shape")
    result = _run(tmp_path, monkeypatch, [step, shape], _protocol(tmp_path))
    assert result["axis_support"][YAXIS]["in_support"] is True
    assert result["labels"][STEP] == "relational_context_use"
