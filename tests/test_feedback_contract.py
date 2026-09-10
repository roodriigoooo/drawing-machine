"""F0: the frozen Direction 3 contract, and the baseline it was frozen over.

Nothing here trains, scores or touches a checkpoint. What it checks is that the
three things F0 exists to produce agree with each other and with the tree:

1. the no-feedback model still produces the bytes the fixture recorded;
2. `docs/feedback-protocol-v0.json` is a serialisation of
   `dm.eval.feedback_contract` and of that fixture's digest;
3. the arithmetic in the contract is the arithmetic `docs/directions.md` states.

The last one matters more than it looks. Every threshold Direction 3 will be
judged against is written down here *before* any feedback output exists, which is
the only ordering under which a threshold is a prediction rather than a
description (`docs/directions.md` §7 invariant 1).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path

import pytest
import torch

from dm.eval import feedback_contract as contract
from dm.eval import feedback_fixture as fixture

PROTOCOL = contract.load_protocol()
FIXTURE = fixture.load_fixture()


# ---------------------------------------------------------------------------
# the baseline


def test_the_no_feedback_path_reproduces_its_frozen_baseline():
    """The claim F1 will rest on, made once while it is still trivially true.

    After F1 this test is the only thing standing between "the feedback path is
    opt-in" and "the feedback path is opt-in as far as anyone checked". It
    compares bit for bit rather than with a tolerance: `allclose` would turn
    §3.2's *unchanged* into *changed by less than my tolerance*, and drift under
    a tolerance accumulates across seven more stages.
    """
    verdicts = fixture.verify_fixture(FIXTURE)
    assert all(verdict.ok for verdict in verdicts), fixture.describe_failure(
        FIXTURE, verdicts
    )


def test_the_fixture_covers_both_pilot_representations_and_both_optional_tables():
    """A baseline that only pinned the byte arm would leave the bit arm's
    stride-8 halting and 8x symbol stream unfrozen, and those are exactly where a
    per-token feedback path is most likely to change behaviour by accident."""
    names = {cell["name"] for cell in FIXTURE["cells"]}
    assert {"byte_standard", "bit_standard"} <= names
    conditional = next(cell for cell in FIXTURE["cells"]
                       if cell["name"] == "byte_conditional_abspos")
    assert conditional["config"]["n_classes"] == 3
    assert conditional["config"]["abs_pos"] is True


def test_a_baseline_config_dictionary_carries_no_feedback_field():
    """What an old checkpoint looks like, which is the input F1 must load.

    §3.2 requirement 1 is that an old config dictionary *omits* the field and
    loads as `none`. A fixture whose configs already carried it would make that
    requirement untestable, so the absence is asserted rather than assumed.
    """
    for cell in FIXTURE["cells"]:
        assert contract.CONFIG_FIELD not in cell["config"], cell["name"]


def test_the_shared_init_cells_pin_one_set_of_non_embedding_weights():
    """Common random numbers across vocabularies, frozen at the value level.

    F1 has to make `share_non_embedding_init` draw the feedback matrices too
    (§3.2 requirement 6). The failure that change can cause is invisible in a
    logit check on one arm: it consumes generator draws, so *every later*
    parameter shifts, and the two codecs stop matching each other.
    """
    byte_cell = next(c for c in FIXTURE["cells"] if c["name"] == "byte_shared_init")
    bit_cell = next(c for c in FIXTURE["cells"] if c["name"] == "bit_shared_init")
    assert byte_cell["state_sha256"] != bit_cell["state_sha256"]  # embeddings differ

    shared = {}
    for spec_dict in (byte_cell["spec"], bit_cell["spec"]):
        model, _ = fixture.build_model(fixture.CellSpec(**spec_dict))
        for name, tensor in model.state_dict().items():
            # `head.weight` is the tied embedding under a second name, and the
            # embedding is exactly what the two codecs differ in.
            if name in ("embed.weight", "head.weight"):
                continue
            if name in shared:
                assert torch.equal(shared[name], tensor), name
            shared[name] = tensor
    assert shared, "nothing was shared"


def test_the_fixture_refuses_a_payload_that_was_edited_after_freezing(tmp_path):
    """A freeze that can be edited without its digest moving is not a freeze."""
    tampered = json.loads(fixture.FIXTURE_PATH.read_text())
    tampered["cells"][0]["n_params"] += 1
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="fixture hash mismatch"):
        fixture.load_fixture(path)


def test_a_fixture_that_is_not_frozen_is_refused(tmp_path):
    body = json.loads(fixture.FIXTURE_PATH.read_text())
    body["status"] = "draft"
    body["fixture_sha256"] = fixture._digest_of(body)
    path = tmp_path / "draft.json"
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="not frozen"):
        fixture.load_fixture(path)


def test_tensors_round_trip_through_the_version_independent_encoding():
    """`torch.save` would have made the frozen bytes a pickle whose contents
    depend on the torch version that wrote them, so the hash would move without
    any value moving."""
    for tensor in (torch.randn(3, 5), torch.arange(7, dtype=torch.long),
                   torch.zeros(0, 4)):
        assert torch.equal(
            fixture.decode_tensor(fixture.encode_tensor(tensor)), tensor
        )


def test_a_truncated_tensor_payload_is_refused_rather_than_reshaped():
    encoded = fixture.encode_tensor(torch.randn(4, 4))
    encoded["shape"] = [4, 5]
    with pytest.raises(ValueError, match="needs 20"):
        fixture.decode_tensor(encoded)


def test_a_failure_report_separates_a_regression_from_a_toolchain_move():
    """The two have completely different repairs, so the message has to say
    which one it is looking at rather than only that the bytes differ."""
    moved = dict(FIXTURE)
    moved["environment"] = {**FIXTURE["environment"], "torch": "0.0.0-not-real"}
    verdict = fixture.CellVerdict(name="byte_standard", ok=False,
                                  mismatches=["logits: 1 of 2 values differ"])
    message = fixture.describe_failure(moved, [verdict])
    assert "byte_standard" in message and "logits" in message
    assert "toolchain moved" in message and "0.0.0-not-real" in message


# ---------------------------------------------------------------------------
# the protocol


def test_the_frozen_protocol_is_a_serialisation_of_the_contract_module():
    """Prose, code and artifact cannot drift apart silently.

    Direction 2 lost a day to the opposite arrangement: a driver was edited after
    the protocol froze its digest, so for a while no file matching the freeze
    existed (`docs/context-audit.md`). Generating the payload from the Module and
    comparing digests makes that failure impossible rather than detectable.
    """
    contract.check_consistency()
    expected = contract.protocol_dict(
        fixture_sha256=FIXTURE["fixture_sha256"]
    )
    assert expected["protocol_sha256"] == PROTOCOL["protocol_sha256"]
    assert expected == PROTOCOL


def test_the_protocol_names_the_fixture_that_is_on_disk():
    assert PROTOCOL["baseline_fixture"]["sha256"] == FIXTURE["fixture_sha256"]
    assert PROTOCOL["baseline_fixture"]["path"] == str(fixture.FIXTURE_PATH)
    assert (fixture.FIXTURE_PATH.exists()
            and Path(PROTOCOL["baseline_fixture"]["path"]).exists())


def test_the_protocol_refuses_a_payload_that_was_edited_after_freezing(tmp_path):
    tampered = json.loads(contract.PROTOCOL_PATH.read_text())
    tampered["inference"]["sesoi_bits_per_target_byte"] = 0.001
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="protocol hash mismatch"):
        contract.load_protocol(path)


def test_v0_claims_no_source_corpus_or_checkpoint_freeze():
    """The three things that cannot exist yet, stated as not frozen.

    F1 changes `dm/models/transformer.py` by design, the paired corpora do not
    exist until F4 and no checkpoint exists until F6. A protocol that quietly
    implied otherwise would let a later stage cite v0 as provenance it never had
    (`docs/directions.md` §7 invariant 10).
    """
    assert "source_sha256" not in PROTOCOL
    assert PROTOCOL["scope"] == "contract"
    text = " ".join(PROTOCOL["does_not_freeze"])
    assert "source digests" in text and "corpus" in text and "checkpoint" in text
    assert set(PROTOCOL["superseded_by"]) == {"feedback-v1", "feedback-v2"}


def test_rewriting_an_existing_freeze_takes_an_explicit_flag(tmp_path, capsys):
    """The freeze is one reflex away from being destroyed.

    Recapturing after F1 has started photographs the *changed* model and hands
    the equivalence claim its own conclusion, so the Adapter refuses rather than
    obliging. Checked here because it is the guard most likely to be needed on a
    day when nobody is thinking about provenance.
    """
    import scripts.feedback_contract as adapter

    fixture_path, protocol_path = tmp_path / "fx.json", tmp_path / "pr.json"
    assert adapter._write(fixture_path, protocol_path, refreeze=False) == 0
    before = fixture_path.read_bytes()

    assert adapter._write(fixture_path, protocol_path, refreeze=False) == 1
    assert "--refreeze" in capsys.readouterr().err
    assert fixture_path.read_bytes() == before

    assert adapter._write(fixture_path, protocol_path, refreeze=True) == 0
    assert adapter._verify(fixture_path, protocol_path) == 0


def test_protocol_v1_extends_v0_by_digest_rather_than_by_copy():
    """Two documents that both restated the same thresholds could disagree. v1
    records v0's digest, so an auditor reads one chain and the pair cannot
    drift."""
    body = contract.protocol_v1_dict(
        base=PROTOCOL, source_sha256={"dm/train.py": "0" * 64},
        corpus={"manifest_sha256": "a" * 64}, smoke={"report": "runs/x.json"},
    )
    assert body["protocol"] == contract.PROTOCOL_V1
    assert body["supersedes"] == contract.PROTOCOL
    assert body["base_protocol_sha256"] == PROTOCOL["protocol_sha256"]
    assert body["protocol_sha256"] == contract.digest_of(body)
    assert body["smoke"]["provenance"] == "engineering"
    assert body["smoke"]["decision_value"] == "none"
    # It freezes what v0 could not, and still not the checkpoints.
    assert set(body["source_sha256"]) == {"dm/train.py"}
    assert "checkpoint" in " ".join(body["does_not_freeze"])


def test_protocol_v1_labels_the_schedule_as_a_project_adaptation():
    body = contract.protocol_v1_dict(base=PROTOCOL, source_sha256={},
                                     corpus={}, smoke={})
    provenance = body["schedule_provenance"]
    assert provenance["kind"] == "project_adaptation"
    assert provenance["source"]["steady_state_pass_weights"] == [0.75, 0.22, 0.03]
    assert provenance["source"]["phase_boundaries_specified"] is False
    assert provenance["source_conditions_complete"] is False
    assert [(row["start"], row["stop"]) for row in provenance["project_phases"]] == [
        (0.0, 0.5), (0.5, 0.75), (0.75, 1.0)
    ]


def test_protocol_v1_carries_the_complete_source_condition_ledger():
    body = contract.protocol_v1_dict(base=PROTOCOL, source_sha256={},
                                     corpus={}, smoke={})
    ledger = body["source_condition_ledger"]
    assert {row["dimension"] for row in ledger} == {
        "pass_schedule", "optimizer_and_parameter_groups", "lr_cooldown_z_loss",
        "scale_and_context", "batching_unit", "gate_input_normalization",
        "normalization_placement_and_affinity", "raw_embedding_paths",
        "superseded_source_arm", "shared_input_norm_gain_calibration",
        "carried_state_jitter_scale",
    }
    assert body["source_conditions_complete"] is False
    assert all({"source", "project", "status"} <= set(row) for row in ledger)
    # The seam is Listing-3-faithful; the *protocol* is still an adaptation,
    # because scale, optimizer and phase boundaries differ. Both statements have
    # to survive together or the ledger is decoration.
    faithful = {row["dimension"] for row in ledger
                if row["status"] == "listing_3_faithful"}
    assert faithful == {"gate_input_normalization",
                        "normalization_placement_and_affinity",
                        "raw_embedding_paths"}
    assert any(row["status"] == "project_adaptation" for row in ledger)
    assert any(row["status"] == "material_difference" for row in ledger)
    # The jitter keeps the source's own 0.02 and only its *rationale* is
    # retracted: v0's payload calls it "the size of the signal's own initial
    # scale", and the signal it perturbs is the carried state at RMS ~1.7, so
    # the perturbation is ~1.1% relative. Changing the number would be a second
    # post-hoc intervention and a source departure; the ledger row carries the
    # correction without one.
    jitter = next(row for row in ledger
                  if row["dimension"] == "carried_state_jitter_scale")
    assert jitter["status"] == "source_value_project_rationale_retracted"
    assert "0.02" in jitter["source"]
    assert contract.JITTER_HALF_WIDTH == 0.02


def test_protocol_v1_declares_the_listing_3_faithful_seam():
    """The freeze has to say *which* normalization, where, and shared with what.

    "Normalizes the gate input" was true of `glu_source_v1` too and is not the
    distinction; an auditor reading the artifact needs the placement and the
    sharing, or the two arms are indistinguishable in the record.
    """
    body = contract.protocol_v1_dict(base=PROTOCOL, source_sha256={},
                                     corpus={}, smoke={})
    architecture = body["architecture"]
    assert architecture["feedback_schema"] == contract.SOURCE_FEEDBACK_SCHEMA_V2
    assert architecture["feedback_schema"] == contract.QUALIFICATION_FEEDBACK_SCHEMA
    assert architecture["fusion"] == contract.SOURCE_V2_FUSION_EQUATION
    assert architecture["seam"] == "listing_3_faithful"
    normalization = architecture["normalization"]
    assert normalization["shared_learned_norm"] is True
    assert normalization["norm_parameter"] == "fuse.norm.weight"
    assert normalization["gate_input"] == "N(e_t)"
    assert normalization["prefix_mixin_placement"] == "before the shared norm"
    assert normalization["plain_prefix_stack_input"] == "N(e_t)"
    assert normalization["initial_gain"] == contract.FUSED_NORM_GAIN
    assert set(normalization["raw_embedding_paths"]) == {
        "training pass 1", "standard prefill"
    }
    # The two earlier arms are named in the freeze rather than deleted from it:
    # they share a tensor set with the qualified one, so a checkpoint written
    # under either loads strict and computes something else.
    assert set(architecture["superseded_arms"]) == {
        "glu_v1", contract.SOURCE_FEEDBACK_SCHEMA
    }


def test_the_v0_freeze_does_not_move_when_a_new_arm_is_added():
    """v0 froze the compatibility contract, not the source arm. Adding
    `glu_source_v2` must leave its payload -- and so its digest -- untouched."""
    assert contract.FEEDBACK_SCHEMAS == ("none", "glu_v1")
    architecture = PROTOCOL["architecture"]
    assert architecture["feedback_schemas"] == ["none", "glu_v1"]
    assert architecture["fusion"] == contract.FUSION_EQUATION
    assert architecture["feedback_parameters"] == list(contract.FEEDBACK_PARAMETERS)
    assert architecture["fused_norm_gain"] == contract.FUSED_NORM_GAIN


def test_protocol_v1_refuses_a_base_that_is_not_v0():
    with pytest.raises(ValueError, match="must extend feedback-v0"):
        contract.protocol_v1_dict(base={"protocol": "something-else"},
                                  source_sha256={}, corpus={}, smoke={})


def test_freeze_adapter_rejects_a_failed_smoke_marked_complete():
    """A completion marker is not a smoke pass and cannot authorize v1."""
    from scripts.feedback_contract import _smoke_problems

    report = {
        "status": "complete",
        "smoke_passed": False,
        "failures": ["generic_guards"],
        "provenance": "engineering",
        "decision_value": "none",
        "config": {"feedback_schema": contract.QUALIFICATION_FEEDBACK_SCHEMA},
        "stages": {},
    }
    problems = _smoke_problems(report, Path("/tmp/smoke-report.json"))
    assert any("smoke_passed" in problem for problem in problems)
    assert any("training stage" in problem for problem in problems)


def test_v1_freezes_a_training_configuration_for_every_cell():
    """F6 is a loop over the cells and this table. A cell that could be launched
    with different settings from its paired twin would make the pair a comparison
    of two experiments."""
    body = contract.protocol_v1_dict(base=PROTOCOL, source_sha256={},
                                     corpus={}, smoke={})
    assert set(body["cells"]) == {contract.cell_name(*cell)
                                  for cell in contract.CELLS}
    for name, config in body["cells"].items():
        assert config["steps"] == contract.FINAL_STEPS
        assert config["feedback_schema"] == contract.QUALIFICATION_FEEDBACK_SCHEMA
        assert config["share_init"] is True
        assert config["data"] == "feedback"
        assert config["tag"] == name
        assert config["extra"]["structure"] in contract.TRAINING_STRUCTURES


def _qualification(frozen: str = "terminal_mix_v1",
                   schedules: dict | None = None) -> dict:
    """A `feedback_qualify.py decide` result, reduced to what the freeze reads."""
    cells = [{"seed": seed, "passed": True, "eligible": True}
             for seed in contract.DEVELOPMENT_SEEDS]
    return {
        "schema": contract.QUALIFICATION_SCHEMA,
        "order": list(contract.ELIGIBLE_SCHEDULES),
        "selection_rule": contract.QUALIFICATION_GATE["selection_rule"],
        "schedules": schedules if schedules is not None else {
            frozen: {"eligible": True, "passed": True, "both_seeds": True,
                     "cells": cells, "failures": {}},
        },
        "frozen_schedule": frozen,
        "label": "qualified",
        "cell_reports": [
            {"schedule": frozen, "seed": seed, "report": f"{frozen}_s{seed}.json",
             "recorded_report_sha256": "e" * 64,
             "recomputed_report_sha256": "e" * 64,
             "file_sha256": "f" * 64, "verified": True}
            for seed in contract.DEVELOPMENT_SEEDS
        ],
    }


def test_every_frozen_training_configuration_actually_builds_its_corpus():
    """`extra` is a corpus-parameter channel, not free-form metadata.

    `dm.train.build_data` pops `structure` and forwards **everything else** in
    `extra` to the corpus builder and from there to the sampler, so a descriptive
    key put there for a reader is a `TypeError` at the first cell. A config table
    that is only checked for its own keys cannot see that; this builds the corpus
    at small scale and lets the real call signature answer.
    """
    import dm.train
    from dm.data.fingerprint import fingerprint
    from dm.train import TrainConfig

    settings = [contract.training_config_development(schedule, seed)
                for schedule in contract.PASS_SCHEDULES
                for seed in contract.DEVELOPMENT_SEEDS]
    settings += [contract.training_config_correction(seed, replicate)
                 for seed, count in contract.CORRECTION_REPLICATES.items()
                 for replicate in range(1, count + 1)]
    settings += [contract.training_config(*cell) for cell in contract.CELLS]
    for config in settings:
        small = {**config, "n_train": 200, "n_val": 40}
        train, val = dm.train.build_data(TrainConfig(**small))
        assert len(train) == 200 and len(val) == 40, config["tag"]
        assert set(config["extra"]) <= {"structure"}, config["extra"]
        # And the corpus a config builds is the corpus its manifest describes.
        # That reconciliation is the one F5b did not have: all six of its cells
        # trained on `data_seed=100` and filed the audited `data_seed=0`
        # manifest, and every clause passed because the corpus clause compared a
        # manifest against itself.
        from dm.data import feedback as corpus_module

        paired = corpus_module.build(200, 40, data_seed=config["data_seed"],
                                     seed=contract.seed_for("derangement"))
        balance = corpus_module.balance_report(paired)
        arm = ("relational" if config["extra"]["structure"] == "relational"
               else "destroyed")
        assert fingerprint(train, val) == balance["corpus"][arm], config["tag"]


def test_v1_freezes_the_qualified_schedule_into_every_estimation_cell():
    """F6 trains under the schedule F5b qualified, not under whichever one is the
    module default. The two were the same until F5b named three of them, which is
    exactly when a default stops being a decision anyone made."""
    body = contract.protocol_v1_dict(
        base=PROTOCOL, source_sha256={}, corpus={}, smoke={},
        qualification=_qualification("terminal_mix_v1"))
    assert body["schedule_provenance"]["qualified_schedule"] == "terminal_mix_v1"
    assert all(config["pass_schedule"] == "terminal_mix_v1"
               for config in body["cells"].values())
    assert body["schedule_provenance"]["project_phases"] == [
        {"start": phase.start, "stop": phase.stop, "weights": list(phase.weights)}
        for phase in contract.pass_schedule("terminal_mix_v1")
    ]
    # The seam is faithful and the protocol is still an adaptation.
    assert body["source_conditions_complete"] is False
    assert body["schedule_provenance"]["kind"] == contract.PROJECT_SCHEDULE_PROVENANCE


def test_v1_carries_the_repaired_gate_and_the_clause_it_withdraws():
    body = contract.protocol_v1_dict(
        base=PROTOCOL, source_sha256={}, corpus={}, smoke={},
        qualification=_qualification())
    assert body["qualification_schema"] == contract.QUALIFICATION_SCHEMA
    assert body["stability_report_schema"] == contract.STABILITY_REPORT_SCHEMA
    gate = body["qualification_gate"]
    assert gate["scored_positions_only"] is True
    assert gate["cost_unit"] == "bits_per_semantic_byte"
    assert gate["stability_subset_size"] == contract.STABILITY_SUBSET_SIZE
    assert gate["max_update_q95_wavefront"] == contract.STABILITY_GATE[
        "max_update_q95"]
    assert gate["selection_rule"] == (
        "first predeclared eligible schedule passing both seeds")
    # Every threshold is v0's or the project's; F5b moves none of them.
    assert gate["max_abs_logit"] == contract.STABILITY_GATE["max_abs_logit"]
    assert gate["max_validation_cost_bits_per_drawing"] == contract.GENERIC_GUARDS[
        "max_validation_cost_bits_per_drawing"]
    assert "update_q95_not_settling" in body["withdrawn_clauses"]


def test_a_stop_cannot_freeze_a_training_protocol():
    for label, frozen in (("unstable_feedback", None),
                          ("no_viable_feedback_implementation_at_scale", None)):
        stopped = {**_qualification(), "frozen_schedule": frozen, "label": label}
        with pytest.raises(ValueError, match="incomplete"):
            contract.protocol_v1_dict(base=PROTOCOL, source_sha256={}, corpus={},
                                      smoke={}, qualification=stopped)


def test_the_diagnostic_control_cannot_freeze_a_training_protocol():
    """It reproduces the source's own missing-three-pass control. A control that
    could authorize the thing it controls for would not be one."""
    control = _qualification("two_pass_control_v1", schedules={
        "two_pass_control_v1": {"eligible": False, "passed": False,
                                "both_seeds": True, "cells": [], "failures": {}},
    })
    with pytest.raises(ValueError, match="not an eligible schedule"):
        contract.protocol_v1_dict(base=PROTOCOL, source_sha256={}, corpus={},
                                  smoke={}, qualification=control)


def test_the_freeze_refuses_a_qualification_it_did_not_like():
    from scripts.feedback_contract import _qualification_problems

    assert not _qualification_problems(_qualification(), Path("q.json"))

    stopped = {**_qualification(), "frozen_schedule": None,
               "label": "unstable_feedback"}
    assert any("no eligible schedule" in problem
               for problem in _qualification_problems(stopped, Path("q.json")))

    one_seed = _qualification()
    one_seed["schedules"]["terminal_mix_v1"]["cells"] = [
        {"seed": contract.DEVELOPMENT_SEEDS[0], "passed": True, "eligible": True}
    ]
    assert any("not [100, 101]" in problem
               for problem in _qualification_problems(one_seed, Path("q.json")))

    unverified = _qualification()
    unverified["cell_reports"] = unverified["cell_reports"][:1]
    assert any("verified cell-report digest" in problem
               for problem in _qualification_problems(unverified, Path("q.json")))

    tampered = _qualification()
    tampered["cell_reports"][0]["recomputed_report_sha256"] = "0" * 64
    assert any("verified cell-report digest" in problem
               for problem in _qualification_problems(tampered, Path("q.json")))

    reordered = {**_qualification(), "order": ["project_progressive_v1"]}
    assert any("predeclared eligible order" in problem
               for problem in _qualification_problems(reordered, Path("q.json")))

    claimed = _qualification()
    claimed["schedules"]["terminal_mix_v1"]["failures"] = {"cells": {}}
    assert any("did not pass" in problem
               for problem in _qualification_problems(claimed, Path("q.json")))


def test_the_freeze_reloads_the_checkpoint_bytes_rather_than_a_boolean(tmp_path):
    """Invariant 11 in its most literal form. A report's `reload.strict` is a
    claim the report makes about itself; the freeze rebuilds the model from the
    checkpoint's own config and loads it strict here, which is the load F7 and
    F8 will do and the one a renamed tensor breaks."""
    from dm.eval.feedback import DEVELOPMENT
    from dm.models.transformer import Config, DrawingLM
    from scripts.feedback_contract import _reconstruct_strict

    cfg = Config(vocab_size=16, d_model=8, n_layers=1, n_heads=2, max_len=64,
                 feedback_schema=contract.QUALIFICATION_FEEDBACK_SCHEMA)
    model = DrawingLM(cfg)
    path = tmp_path / "cell.pt"
    torch.save({"cfg": asdict(cfg), "state": model.state_dict(),
                "provenance": DEVELOPMENT}, path)

    reconstructed = _reconstruct_strict(path)
    assert reconstructed["strict"] is True
    assert reconstructed["params_match_config"] is True
    assert reconstructed["provenance"] == DEVELOPMENT
    assert reconstructed["feedback_schema"] == contract.QUALIFICATION_FEEDBACK_SCHEMA
    assert set(contract.FEEDBACK_PARAMETERS) <= set(reconstructed["state_keys"])

    # A renamed fusion tensor is exactly what strict loading exists to catch, and
    # exactly what a recorded boolean would have missed.
    payload = torch.load(path, weights_only=False)
    payload["state"]["fuse.gain"] = payload["state"].pop("fuse.norm.weight")
    torch.save(payload, path)
    with pytest.raises(RuntimeError):
        _reconstruct_strict(path)


def test_the_training_configuration_names_numbers_rather_than_defaults():
    """Restated rather than inherited from `TrainConfig`, so a later edit to
    those defaults cannot retroactively change what the pilot trained."""
    from dm.train import TrainConfig

    config = contract.training_config("byte", "relational", 0)
    rebuilt = TrainConfig(**config)
    assert rebuilt.steps == contract.FINAL_STEPS
    assert rebuilt.codec == "byte" and rebuilt.seed == 0
    assert rebuilt.extra == {"structure": "relational"}
    assert contract.TRAINING_DEFAULTS["batch_size"] == 64


@pytest.mark.parametrize("bad", [("token", "relational", 0),
                                 ("byte", "shuffled", 0), ("byte", "relational", 7)])
def test_a_cell_outside_the_bounded_matrix_is_refused(bad):
    with pytest.raises(ValueError):
        contract.training_config(*bad)


def test_the_protocol_ends_with_one_executable_next_command():
    """F0's exit condition: no feedback code, and exactly one command that runs
    the stage after it."""
    command = PROTOCOL["next_stage"]["command"]
    assert PROTOCOL["next_stage"]["stage"] == "F1"
    target = Path(command.split()[-1])
    assert target.exists(), f"{target} is named by the protocol and does not exist"


# ---------------------------------------------------------------------------
# the arithmetic


def test_the_feedback_overhead_counts_the_norm_gain():
    """`2D^2 + D`, not `2D^2`. The 128 parameters are arithmetically trivial and
    the discrepancy is not: `Config.n_params()` is asserted against the live
    module, so a count that is 128 short fails the model's own invariant."""
    assert contract.feedback_overhead(128) == 32_896
    assert contract.feedback_overhead(128) == 2 * 128**2 + 128
    assert contract.feedback_overhead(128) != 2 * 128**2


@pytest.mark.parametrize(
    "representation,standard,feedback",
    [("bit", 792_192, 825_088), ("byte", 824_704, 857_600),
     ("token", 826_752, 859_648), ("token_typed", 990_592, 1_023_488)],
)
def test_parameter_counts_are_the_ones_the_pilot_was_sized_against(
        representation, standard, feedback):
    """Pinned as literals, and checked against `Config` rather than restated from
    it: the point of the table is that the two agree."""
    row = contract.parameter_table()[representation]
    assert row["standard"] == standard == contract.standard_params(representation)
    assert row["feedback_capable"] == feedback
    assert row["under_budget"] is (feedback < contract.PARAMETER_BUDGET)


def test_the_pilot_representations_fit_the_budget_and_the_excluded_one_does_not():
    table = contract.parameter_table()
    assert all(table[name]["under_budget"] for name in contract.REPRESENTATIONS)
    assert not table["token_typed"]["under_budget"]


def test_the_pass_schedule_costs_the_frozen_average():
    """`1.1325` passes/batch, so "same training tokens" can be separated from
    "same compute" in the record rather than in a footnote."""
    assert contract.expected_passes_per_batch() == pytest.approx(1.1325, abs=1e-12)


def test_the_pass_schedule_partitions_training_with_no_gap():
    covered = [(phase.start, phase.stop) for phase in contract.PASS_SCHEDULE]
    assert covered[0][0] == 0.0 and covered[-1][1] == 1.0
    assert all(left[1] == right[0] for left, right in pairwise(covered))


@pytest.mark.parametrize(
    "step,weights",
    [(0, (1.0,)), (11_999, (1.0,)), (12_000, (0.75, 0.25)),
     (17_999, (0.75, 0.25)), (18_000, (0.75, 0.22, 0.03)),
     (24_000, (0.75, 0.22, 0.03))],
)
def test_every_step_of_the_pilot_budget_lands_in_the_phase_the_contract_says(
        step, weights):
    """Including the last one. The final segment is closed at the top so step
    24,000 has a phase at all; every other boundary is half-open."""
    assert contract.phase_for(step, contract.FINAL_STEPS).weights == weights


def test_a_step_outside_the_budget_is_refused():
    with pytest.raises(ValueError, match="outside"):
        contract.phase_for(contract.FINAL_STEPS + 1, contract.FINAL_STEPS)


def test_a_phase_whose_weights_do_not_form_a_distribution_is_refused():
    with pytest.raises(ValueError, match="sum to"):
        contract.PassPhase(0.0, 1.0, (0.5, 0.4))
    with pytest.raises(ValueError, match="1..3 passes"):
        contract.PassPhase(0.0, 1.0, (0.25, 0.25, 0.25, 0.25))


# ---------------------------------------------------------------------------
# seeds and inherited instruments


def test_seed_namespaces_are_distinct_and_frozen_at_their_recorded_values():
    """Derived from the name, so adding a stream cannot move an existing one --
    and recorded in the protocol, so a *rename* fails here instead of quietly
    producing a different corpus."""
    table = contract.seed_table()
    assert table == PROTOCOL["seeds"]["values"]
    assert len(set(table.values())) == len(table)
    assert set(table) == set(contract.SEED_NAMESPACES)


def test_the_bootstrap_stream_is_direction_2s_so_the_intervals_are_comparable():
    """`G` is a difference of two `Delta`s on Direction 2's own cases. Resampling
    different clusters would make the two stages' intervals incomparable for no
    reason at all."""
    assert contract.seed_for("bootstrap") == contract.INHERITED_BOOTSTRAP_SEED
    from dm.eval.context import BUILDER_SEED

    assert contract.INHERITED_BOOTSTRAP_SEED == BUILDER_SEED


def test_an_undeclared_seed_namespace_cannot_be_drawn_from():
    with pytest.raises(KeyError, match="unknown seed namespace"):
        contract.seed_for("whatever")


def test_the_venue_is_direction_2s_audited_synthetic_step_manifest():
    """The instrument is reused, its protocol is not. Checking the hash against
    Direction 2's own frozen protocol is what stops a rebuilt-but-not-identical
    manifest from being read as the audited one."""
    v5 = json.loads(Path("docs/context-protocol-v5.json").read_text())
    manifests = v5["context_c3"]["manifests"]
    synthetic = manifests["runs/context_c3_synthetic_flat_v4.json"]
    assert contract.VENUE_MANIFEST_SHA256 == synthetic["manifest_sha256"]
    assert contract.VENUE_CASES == synthetic["venue_case_counts"][contract.VENUE]


def test_the_practical_floor_is_direction_2s_and_predates_every_direction_3_number():
    v5 = json.loads(Path("docs/context-protocol-v5.json").read_text())
    assert (contract.SESOI_BITS_PER_TARGET_BYTE
            == v5["context_c3"]["sesoi_bits_per_target_byte"])
    assert contract.BOOTSTRAP_REPS == v5["context_c3"]["bootstrap"]["replicates"]
    assert contract.BOOTSTRAP_UNIT == v5["context_c3"]["bootstrap"]["unit"]


# ---------------------------------------------------------------------------
# the decision surface


def test_the_bounded_matrix_is_eight_named_cells():
    assert len(contract.CELLS) == 8
    names = [contract.cell_name(*cell) for cell in contract.CELLS]
    assert len(set(names)) == 8
    assert names == PROTOCOL["matrix"]["cells"]


def test_every_stopping_label_has_a_definition_and_none_of_them_is_a_null():
    """A missing artifact resolves to `incomplete`, never to zero and never to a
    pass. Direction 2's retracted schema-1 report is the reason that is a rule."""
    assert set(contract.CLAIM_LABELS) == {
        "incomplete", "unstable_feedback", "no_feedback_gain",
        "teacher_forced_feedback_only", "structure_specific_feedback",
        "exploratory_representation_interaction",
    }
    assert "never a null" in contract.CLAIM_LABELS["incomplete"]


def test_the_decision_rule_requires_every_clause_including_free_generation():
    """Teacher-forced gain alone is `teacher_forced_feedback_only`. Reading it as
    the result is the Direction 2 error one stage on."""
    assert len(contract.DECISION_RULE) == 5
    joined = " ".join(contract.DECISION_RULE)
    assert "both seeds" in joined and "stability" in joined
    assert "free generation" in joined


def test_the_primary_family_is_two_structure_tests_and_excludes_the_exploratory_one():
    assert contract.PRIMARY_FAMILY == ("I_structure[bit]", "I_structure[byte]")
    assert PROTOCOL["inference"]["representation_interaction_status"] == "exploratory"
    assert PROTOCOL["inference"]["seeds_pooled"] is False


def test_the_generic_guards_keep_a_gain_from_being_bought_with_likelihood():
    guards = contract.GENERIC_GUARDS
    assert guards["max_validation_cost_bits_per_drawing"] == 1.0
    assert guards["min_valid_halt_delta"] == -0.05
    assert guards["max_truncation_rate_increase"] == 0.0


def test_the_stability_gate_bounds_collapse_as_well_as_blow_up():
    """A carried state that decays to nothing is as broken a channel as one that
    diverges, and only a band catches both."""
    assert contract.STABILITY_RMS_BAND == (0.25, 4.0)
    assert contract.STABILITY_GATE["rms_p99_band"] == [0.25, 4.0]
    assert contract.STABILITY_PASSES == (0, 1, 2, 4, 8, 16, 32)
    assert contract.STABILITY_GATE["update_q95_monotone_from_pass"] == 8


def test_promotion_requires_preference_and_exact_hits_in_both_seeds():
    gate = contract.PROMOTION_GATE
    assert gate["min_delta_gen_soft_minus_standard"] == 0.01
    assert gate["min_hit_own_soft_minus_standard"] == 0.02
    assert gate["both_seeds_required"] is True
    assert gate["structural_mask"] == "off"


def test_the_fused_norm_gain_and_the_jitter_share_the_embedding_scale():
    """Gain 1 would hand the stack an input whose RMS is ~50x the embedding's at
    step 0, so the first optimiser steps would undo the initialisation instead of
    learning the channel."""
    assert contract.FUSED_NORM_GAIN == 0.02
    assert contract.JITTER_HALF_WIDTH == contract.FUSED_NORM_GAIN
    contract.check_consistency()


def test_glu_v1_owns_exactly_three_named_tensors():
    """Parameter names are checkpoint contract: eight checkpoints are written at
    F6 and reloaded strict at F7 and F8."""
    assert contract.FEEDBACK_PARAMETERS == (
        "fuse.up.weight", "fuse.gate.weight", "fuse.norm.weight"
    )
    assert contract.FEEDBACK_SCHEMAS == ("none", "glu_v1")
    assert contract.DEFAULT_FEEDBACK_SCHEMA == "none"


def test_the_required_record_fields_keep_programs_symbols_and_passes_apart():
    """"Same training tokens" is not "same compute" and not "same data". A record
    that collapses the three cannot support any efficiency statement."""
    required = set(contract.REQUIRED_RECORD_FIELDS)
    assert {"programs_seen", "semantic_bytes_seen", "content_symbols_seen",
            "padded_positions", "content_symbol_forward_passes",
            "pass_histogram", "wall_clock_s"} <= required


def test_the_required_report_fields_persist_raw_nlls_and_complete_denominators():
    assert "raw_symbol_nll_bits" in contract.REQUIRED_SCORE_FIELDS
    assert "sequential_standard_matches_full_forward" in contract.REQUIRED_SCORE_FIELDS
    assert "reach_rate" in contract.REQUIRED_GENERATION_FIELDS
    assert "hit_own_rate" in contract.REQUIRED_GENERATION_FIELDS
