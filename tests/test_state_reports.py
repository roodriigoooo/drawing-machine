from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from dm.data.fingerprint import fingerprint
from dm.eval import provenance
from dm.eval.sampling import SamplingConfig
from dm.isa.codec import ByteCodec
from dm.isa.state import LanguagePolicy, Reason, trace
from dm.models.transformer import Config, DrawingLM
from scripts import state_gate, state_sampler


def test_provenance_rejects_a_rebuilt_corpus_label_drift(monkeypatch):
    train = [b"\x00"]
    val = [b"\x00"]
    config = SimpleNamespace(data="synthetic")
    monkeypatch.setattr(provenance, "corpus_config", lambda record: config)
    monkeypatch.setattr(provenance, "build_data", lambda _: (train, val))
    record = {"name": "fixture", "corpus": fingerprint(train, val)}
    verified = provenance.verify_corpus(record)
    assert verified.fingerprint == record["corpus"]

    record["corpus"] = dict(record["corpus"], val="drifted")
    with pytest.raises(ValueError, match="differs from its record fingerprint"):
        provenance.verify_corpus(record)


def test_first_exclusion_uses_opcode_policy_before_scope():
    policy = LanguagePolicy.of(["HALT", "MOVE", "LINE"])
    result = trace(bytes([0x08, 0x00]), policy)
    assert result.first_exclusions[0] is Reason.OPCODE_POLICY


def test_sampler_identity_separates_cell_sets():
    policy = LanguagePolicy.of(["HALT"])
    sampler = SamplingConfig(top_k=1, temperature=1.0)
    raw = state_sampler.report_path(
        "fixture", 2, 1, False, 128, "cpu", sampler, policy, ["raw"]
    )
    full = state_sampler.report_path(
        "fixture", 2, 1, False, 128, "cpu", sampler, policy,
        ["raw", "vm_safe", "canonical"],
    )
    assert raw != full
    assert "cells-raw" in raw.name
    assert "cells-canonical-raw-vmsafe" in full.name
    protocol_a = state_sampler.report_path(
        "fixture", 2, 1, False, 128, "cpu", sampler, policy, ["raw"], "a" * 64
    )
    protocol_b = state_sampler.report_path(
        "fixture", 2, 1, False, 128, "cpu", sampler, policy, ["raw"], "b" * 64
    )
    assert protocol_a != protocol_b
    assert "_praaaaaaaaaaaa" in protocol_a.name


def test_sampler_marks_geometry_with_empty_decodes_incomplete(monkeypatch):
    codec = ByteCodec()
    policy = LanguagePolicy.of(["HALT"])
    model = DrawingLM(Config(vocab_size=codec.vocab_size, d_model=16,
                             n_layers=1, n_heads=1, max_len=32)).eval()
    monkeypatch.setattr(
        state_sampler,
        "quality_of",
        lambda *args, **kwargs: {
            "empty": 1.0, "coverage": 0.0, "mmd": 0.0, "nna": 0.0,
        },
    )
    result = state_sampler.one_cell(
        model, codec, policy, "raw", rows=2, cap=4,
        variates=torch.full((2, 4), 0.5),
        sampler=SamplingConfig(top_k=1, temperature=1.0), device="cpu",
        reference=[b"\x00"], classes=None, cloud_reference=object(),
        cloud_points=8,
    )
    assert result["geometry_status"] == "unavailable_empty"
    assert result["geometry_complete"] is False


def test_sampler_reports_empty_rate_when_geometry_is_disabled():
    codec = ByteCodec()
    policy = LanguagePolicy.of(["HALT"])
    model = DrawingLM(Config(vocab_size=codec.vocab_size, d_model=16,
                             n_layers=1, n_heads=1, max_len=32)).eval()
    result = state_sampler.one_cell(
        model, codec, policy, "raw", rows=2, cap=4,
        variates=torch.full((2, 4), 0.5),
        sampler=SamplingConfig(top_k=1, temperature=1.0), device="cpu",
        reference=[b"\x00"], classes=None, cloud_reference=None,
        cloud_points=8,
    )
    assert 0.0 <= result["empty"] <= 1.0
    assert result["nonempty"] == pytest.approx(1.0 - result["empty"])
    assert "geometry_status" not in result


def test_geometry_empty_policy_is_explicit_for_new_protocols():
    assert state_sampler.geometry_empty_policy({
        "protocol_id": "direction1-state-v2"
    }) == "incomplete"
    assert state_sampler.geometry_empty_policy({
        "sampler": {"empty_geometry_policy": "floor_failure"}
    }) == "floor_failure"
    with pytest.raises(ValueError, match="empty_geometry_policy"):
        state_sampler.geometry_empty_policy({"protocol_id": "future"})


def test_gate_manifest_check_requires_frozen_checkpoint_hashes(tmp_path, monkeypatch):
    entries = []
    reports = []
    readings = []
    context_reports = []
    for index in range(2):
        checkpoint = tmp_path / f"s{index}.pt"
        record = tmp_path / f"s{index}.json"
        checkpoint.write_bytes(f"checkpoint-{index}".encode())
        record.write_text(f"record-{index}")
        checkpoint_sha = state_gate.sha256_file(checkpoint)
        record_sha = state_gate.sha256_file(record)
        entries.append({
            "checkpoint": str(checkpoint), "record": str(record),
            "checkpoint_sha256": checkpoint_sha, "record_sha256": record_sha,
        })
        reports.append({
            "checkpoint": str(checkpoint),
            "identity": {
                "checkpoint_sha256": checkpoint_sha,
                "record_sha256": record_sha,
            },
        })
        readings.append({
            "checkpoint": str(checkpoint),
            "identity": {
                "checkpoint_sha256": checkpoint_sha,
                "record_sha256": record_sha,
            },
        })
        context_reports.append({
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "record_sha256": record_sha,
        })
    manifest = tmp_path / "context.json"
    manifest.write_text("manifest")
    monkeypatch.setattr(
        state_gate, "load_manifest", lambda path: {"manifest_sha256": "manifest-hash"}
    )
    protocol = {
        "checkpoint_manifest": entries,
        "context_c3": {
            "manifest": str(manifest), "manifest_sha256": "manifest-hash",
        },
    }
    c3 = {"manifest": str(manifest), "reports": context_reports}
    assert state_gate.manifest_check(protocol, reports, {"readings": readings}, c3)["all"]

    reports[0]["identity"]["checkpoint_sha256"] = "drifted"
    assert not state_gate.manifest_check(
        protocol, reports, {"readings": readings}, c3
    )["s3"]


def test_gate_requires_dynamic_language_and_a_complete_c3_instrument():
    flat = {"policy": {"opcodes": ["HALT", "MOVE", "LINE"]}}
    structured = {
        "policy": {"opcodes": ["HALT", "MOVE", "REPEAT", "ENDREP"]}
    }
    assert not state_gate._state_bearing(flat)
    assert state_gate._state_bearing(structured)

    # A positive point estimate cannot self-certify controls the protocol says
    # are part of the decision.
    assert not state_gate._c3_contract_complete({"c3_positive": True})
    assert state_gate._c3_contract_complete({
        "instrument_complete": True,
        "instrument_checks": {"nondegenerate": True, "frequency": True},
    })
    assert state_gate._valid_numeric_gate(state_gate.HISTORICAL_THRESHOLDS)
    assert not state_gate._valid_numeric_gate({
        **state_gate.HISTORICAL_THRESHOLDS,
        "decoder_delta_min": float("nan"),
    })
    role_protocol = {
        "checkpoint_manifest": [
            {"checkpoint": "flat.pt", "family": "flat"},
            {"checkpoint": "structured.pt", "family": "structured"},
        ],
        "family_roles": {
            "flat": "flat_control",
            "structured": "state_bearing",
        },
    }
    role_values = [
        {"checkpoint": "flat.pt", **flat},
        {"checkpoint": "structured.pt", **structured},
    ]
    assert state_gate._family_role_contract(role_protocol, role_values)
    role_protocol["family_roles"]["flat"] = "state_bearing"
    assert not state_gate._family_role_contract(role_protocol, role_values)


def test_gate_checks_sampler_and_full_support_contracts():
    protocol = {
        "sampler": {
            "n": 2,
            "draws": 1,
            "top_k": 40,
            "temperature": 1.0,
            "forbid": [0, 1],
            "mask_levels": ["raw", "vm_safe", "canonical"],
        },
        "seed_roles": {"draw_seeds": [0]},
    }
    base_sampler = {
        "top_k": 40,
        "temperature": 1.0,
        "forbid_specials": True,
        "mask_mode": None,
        "policy_digest": None,
    }
    policy_digest = "cf20e79b115e"
    report = {
        "n": 2,
        "seeds": 1,
        "draw_seeds": [0],
        "sampler": base_sampler,
        "policy_digest": policy_digest,
        "identity": {"draw_seeds": [0]},
        "cell_samplers": {
            level: {
                **base_sampler,
                "mask_mode": level,
                "policy_digest": policy_digest,
            }
            for level in ("raw", "vm_safe", "canonical")
        },
        "cells": {
            level: {"draws": [{"n": 2}]}
            for level in ("raw", "vm_safe", "canonical")
        },
    }
    assert state_gate._sampler_contract(protocol, [report])
    assert not state_gate._sampler_contract(
        protocol, [{**report, "n": 3}]
    )
    assert state_gate._support_contract({
        "readings": [{
            "limit": None,
            "n": 10,
            "corpus": {"n_val": 10},
            "levels": {"vm_safe": {}, "canonical": {}},
        }]
    })
