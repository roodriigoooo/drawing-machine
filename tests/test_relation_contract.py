"""R1: the executable contract, and the frozen artifact that has to equal it.

Two things this file exists to make impossible. First, prose, code and the
serialised protocol drifting apart: `docs/copy-relation-protocol-v0.json` is a
serialisation of `dm.eval.relation_contract`, and if the Module moves without the
artifact moving, the freeze names code nobody ran -- Direction 2's C4 fault,
reproduced. Second, Direction 4 inheriting a Direction 3 number by accident: the
two directions must share primitives and share no thresholds, seeds or names.
"""

from __future__ import annotations

import json

import pytest

from dm.data import relation as corpus
from dm.eval import feedback_contract as feedback
from dm.eval import relation_contract as contract
from dm.models.transformer import Config

# ---------------------------------------------------------------------------
# self-consistency


def test_the_contract_is_self_consistent():
    contract.check_consistency()


def test_none_is_the_first_and_default_schema():
    """An old config omits the field, loads as `none` and owns no parameters."""
    assert contract.RELATION_SCHEMAS[0] == contract.DEFAULT_RELATION_SCHEMA == "none"
    assert contract.parameter_table()["none"]["relation"] == 0


def test_standard_is_the_first_runtime_mode():
    assert contract.RUNTIME_MODES[0] == "standard"
    assert contract.PRIMARY_CONTRAST == ("predicted_copy", "standard")


def test_the_matrix_is_four_cells_and_two_seeds():
    assert len(contract.CELLS) == len(set(contract.CELLS)) == 4
    assert len(contract.SCIENTIFIC_SEEDS) == 2
    for arm in contract.TRAINING_ARMS:
        for seed in contract.SCIENTIFIC_SEEDS:
            assert contract.cell_name(arm, seed) in contract.CELLS


def test_a_development_seed_cannot_name_a_scientific_cell():
    """Development artifacts never enter a scientific threshold."""
    with pytest.raises(ValueError, match="not a scientific seed"):
        contract.cell_name("span_affine_v1", "development_a")


def test_an_unknown_arm_is_refused():
    with pytest.raises(ValueError, match="unknown arm"):
        contract.cell_name("glu_v1", "model_a")


# ---------------------------------------------------------------------------
# parameters


def test_the_baseline_count_is_read_from_the_model_not_quoted():
    assert contract.standard_params() == Config(vocab_size=258,
                                                d_model=128).n_params()
    assert contract.standard_params() == contract.BYTE_MODEL_PARAMETERS


def test_the_relation_head_fits_inside_the_frozen_cap():
    table = contract.parameter_table()
    assert table["span_affine_v1"]["total"] <= contract.PARAMETER_CAP
    assert table["budget"]["cap"] == 857_472
    assert table["budget"]["headroom"] >= 0


def test_the_corpus_seed_the_contract_names_is_the_one_a_build_must_use():
    """Nothing connected the two before: the builder took a literal `data_seed`
    while the protocol recorded a derived one, and no clause compared them."""
    want = contract.corpus_seed("scientific")
    assert want == contract.seed_for("corpus")
    assert contract.corpus_seed_faults(
        {"acceptance": {"config": {
            "data_seed": want, "provenance": "scientific"}}}) == []
    faults = contract.corpus_seed_faults(
        {"acceptance": {"config": {"data_seed": 0}}})
    assert faults and "412653006" not in str(0)


def test_the_development_corpus_has_its_own_seed():
    assert contract.corpus_seed("development") != contract.corpus_seed("scientific")
    faults = contract.corpus_seed_faults(
        {"acceptance": {"config": {
            "data_seed": contract.corpus_seed("development")}}})
    assert any("development" in fault for fault in faults)


def test_an_unknown_corpus_provenance_is_refused():
    with pytest.raises(ValueError, match="unknown corpus provenance"):
        contract.corpus_seed("scratch")


def test_the_length_bins_have_edges_and_the_last_is_the_byte_cap():
    """A bin count without boundaries fixes a parameter total and nothing else."""
    assert contract.RELATION_LENGTH_BIN_EDGES[-1] == corpus.MAX_SOURCE_BYTES
    assert contract.length_bin(1) == 0
    assert contract.length_bin(corpus.MAX_SOURCE_BYTES) == \
        contract.RELATION_LENGTH_BINS - 1
    with pytest.raises(ValueError, match="past the last length-bin edge"):
        contract.length_bin(corpus.MAX_SOURCE_BYTES + 1)


def test_the_enumerator_caps_are_frozen_and_cover_the_corpus_bound():
    assert contract.RELATION_MAX_CANDIDATE_SPANS >= corpus.CANDIDATE_SPAN_BOUND
    assert contract.RELATION_KBEST_ACTIONS > 0


def test_the_action_term_marginalises_the_joint_valid_set():
    """Per-factor marginals let probability collect on invalid cross-products."""
    assert contract.ACTION_MARGINALIZATION == "joint_valid_action_set"
    assert "action_joint" in contract.ACTION_LOSS_COMPONENTS
    assert not set(contract.ACTION_LOSS_COMPONENTS) & \
        set(contract.ACTION_DIAGNOSTIC_FACTORS)


def test_the_metric_level_fallback_is_recorded_as_withdrawn():
    assert "withdrawn in v1" in contract.R0_FALLBACK_STATUS
    assert not hasattr(contract, "R0_RESOLUTION_MARGIN")


def test_the_overhead_is_the_sum_of_its_declared_tensors():
    """Hand-checkable arithmetic, so a change to one term cannot hide in a total."""
    d = contract.PILOT_D_MODEL
    rank, bins = contract.RELATION_RANK, contract.RELATION_LENGTH_BINS
    expected = (
        d                                   # RMSNorm gain
        + 3 * d * rank + bins * rank        # query, two span keys, length embedding
        + d * 2                             # EMIT/COPY
        + d * len(corpus.D4_SUPPORT)
        + 2 * d * len(corpus.TRANSLATION_SUPPORT)
        + d * len(corpus.COUNT_SUPPORT)
    )
    assert contract.relation_overhead(d) == expected


def test_a_translation_head_over_every_signed_byte_would_break_the_budget():
    """The reason `TRANSLATION_SUPPORT` is three values and not 256."""
    assert 2 * contract.PILOT_D_MODEL * 256 > contract.RELATION_PARAMETER_BUDGET


# ---------------------------------------------------------------------------
# seeds


def test_a_seed_is_derived_from_its_name_and_is_stable():
    assert contract.seed_for("model_a") == contract.seed_for("model_a")
    assert contract.seed_for("model_a") != contract.seed_for("model_b")


def test_an_undeclared_namespace_cannot_be_drawn_from():
    with pytest.raises(KeyError, match="unknown seed namespace"):
        contract.seed_for("jitter")


def test_no_two_namespaces_collide():
    table = contract.seed_table()
    assert len(set(table.values())) == len(table)


def test_direction_four_shares_no_seed_with_direction_three():
    """Salted roots, so a namespace name reused across directions is a new stream."""
    assert contract.SEED_ROOT != feedback.SEED_ROOT
    shared = set(contract.SEED_NAMESPACES) & set(feedback.SEED_NAMESPACES)
    assert shared, "the test is only meaningful while some name is reused"
    for name in shared:
        assert contract.seed_for(name) != feedback.seed_for(name)


def test_direction_four_shares_no_threshold_with_direction_three():
    assert contract.PROTOCOL != feedback.PROTOCOL
    assert contract.PROTOCOL_PATH != feedback.PROTOCOL_PATH
    assert not set(contract.CELLS) & set(feedback.CELLS)


# ---------------------------------------------------------------------------
# labels


def test_the_label_order_puts_completeness_and_safety_before_any_effect():
    assert contract.label_for(complete=False, guards_pass=True, flat_pass=True,
                              nested_pass=True) == "incomplete"
    assert contract.label_for(complete=True, guards_pass=False, flat_pass=True,
                              nested_pass=True) == "unsafe_copy_channel"


def test_a_nested_failure_narrows_the_claim_rather_than_erasing_it():
    assert contract.label_for(complete=True, guards_pass=True, flat_pass=True,
                              nested_pass=False) == "explicit_relation_gain"
    assert contract.label_for(complete=True, guards_pass=True, flat_pass=True,
                              nested_pass=True) == "compositional_relation_gain"


def test_a_flat_failure_is_a_negative_whatever_nesting_did():
    assert contract.label_for(complete=True, guards_pass=True, flat_pass=False,
                              nested_pass=True) == "no_explicit_relation_gain"


def test_every_label_carries_a_recorded_reason():
    assert set(contract.LABELS) == set(contract.LABEL_REASONS)


# ---------------------------------------------------------------------------
# the accounting boundary


def test_action_nll_never_enters_bits_per_drawing():
    """§7 invariant 13, made structural rather than remembered."""
    assert contract.ACTION_NLL_ENTERS_BITS_PER_DRAWING is False
    assert contract.BYTE_NLL_COVERS_COPIED_BYTES is True


def test_the_gate_skips_boundaries_no_decode_reaches():
    assert contract.GATE_POSITIONS == "reachable_boundaries"


def test_relation_and_latent_feedback_may_not_coexist():
    assert contract.FORBIDDEN_FEEDBACK_SCHEMA == "none"


def test_the_audit_sees_wider_translations_than_the_head_can_name():
    """Otherwise the destroyed control could only refuse namable relations."""
    assert set(corpus.TRANSLATION_SUPPORT) < set(corpus.AUDIT_TRANSLATIONS)


# ---------------------------------------------------------------------------
# the artifact


def test_the_protocol_digest_excludes_itself_and_moves_with_the_body():
    body = contract.protocol_dict()
    assert body["protocol_sha256"] == contract.digest_of(body)
    assert contract.digest_of({**body, "direction": 5}) != body["protocol_sha256"]


def test_the_protocol_round_trips(tmp_path):
    path = tmp_path / "protocol.json"
    contract.write_protocol(path, contract.protocol_dict())
    assert contract.load_protocol(path)["protocol"] == contract.PROTOCOL


def test_an_edited_protocol_is_refused(tmp_path):
    path = tmp_path / "protocol.json"
    body = contract.protocol_dict()
    contract.write_protocol(path, {**body, "direction": 3})
    with pytest.raises(ValueError, match="hash mismatch"):
        contract.load_protocol(path)


def test_an_unfrozen_protocol_is_refused(tmp_path):
    path = tmp_path / "protocol.json"
    body = {**contract.protocol_dict(), "status": "draft"}
    body["protocol_sha256"] = contract.digest_of(body)
    contract.write_protocol(path, body)
    with pytest.raises(ValueError, match="not frozen"):
        contract.load_protocol(path)


def test_a_protocol_naming_no_corpus_cannot_authorize_training(tmp_path):
    path = tmp_path / "protocol.json"
    contract.write_protocol(path, contract.protocol_dict(corpus_sha256=None))
    contract.load_protocol(path)                       # reading is fine
    with pytest.raises(ValueError, match="names no traced corpus"):
        contract.load_protocol(path, require_corpus=True)


def test_a_digest_string_alone_does_not_authorize_training(tmp_path):
    """A fresh clone has no `runs/`, and a digest in a protocol is not a corpus."""
    path = tmp_path / "protocol.json"
    contract.write_protocol(path, contract.protocol_dict(corpus_sha256="0" * 64))
    with pytest.raises(ValueError, match="does not exist"):
        contract.load_protocol(path, require_corpus=True, corpus_root=tmp_path)


def test_a_corpus_whose_digest_disagrees_is_refused(tmp_path):
    from dm.data import relation as corpus_module

    built = corpus_module.build(corpus_module.BuildConfig(
        n_train_synthetic=30, n_train_composed=40, n_eval=12, n_generic=12,
        train_motifs=40, eval_motifs=30, composed_limit=64,
        data_seed=contract.corpus_seed("scientific")))
    body = corpus_module.manifest(built)
    body = {**body, "accepted": True, "problems": []}
    body.pop("corpus_sha256")
    import hashlib
    import json as _json
    body["corpus_sha256"] = hashlib.sha256(
        _json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    manifest_path = tmp_path / contract.CORPUS_MANIFEST_PATH_STR
    corpus_module.write_manifest(manifest_path, body)
    path = tmp_path / "protocol.json"
    contract.write_protocol(path, contract.protocol_dict(corpus_sha256="1" * 64))
    with pytest.raises(ValueError, match="acceptance verdict"):
        contract.load_protocol(path, require_corpus=True, corpus_root=tmp_path)


def test_the_frozen_artifact_describes_todays_module():
    """The freeze must name the code in the tree, not code that once existed."""
    frozen = contract.load_protocol()
    rebuilt = contract.protocol_dict(
        corpus_sha256=(frozen.get("corpus") or {}).get("canonical_payload_sha256"))
    moved = sorted(key for key in set(rebuilt) | set(frozen)
                   if key != "protocol_sha256" and rebuilt.get(key) != frozen.get(key))
    assert moved == [], f"the protocol no longer describes the tree: {moved}"
    assert rebuilt["protocol_sha256"] == frozen["protocol_sha256"]


def test_the_artifact_serialises_the_corpus_caps_the_builder_enforces():
    frozen = json.loads(contract.PROTOCOL_PATH.read_text())
    caps = frozen["corpus"]["caps"]
    assert caps["max_source_bytes"] == corpus.MAX_SOURCE_BYTES
    assert caps["max_source_instructions"] == corpus.MAX_SOURCE_INSTRUCTIONS
    assert caps["max_source_gap_instructions"] == corpus.MAX_SOURCE_GAP_INSTRUCTIONS
    supports = frozen["corpus"]["supports"]
    assert supports["translation"] == list(corpus.TRANSLATION_SUPPORT)
    assert supports["count"] == list(corpus.COUNT_SUPPORT)


def test_only_r0_and_r1_are_authorized():
    assert contract.AUTHORIZED_STAGES == ("R0", "R1")
    assert set(contract.AUTHORIZED_STAGES) <= set(contract.STAGES)
