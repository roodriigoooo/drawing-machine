"""The qualification's evidence layer: identity, environment, reconstruction.

These are the checks F5b's six cells did not have, written against the failure
that revealed the gap. All six trained on `data_seed=100` and filed the audited
`data_seed=0` manifest, and every clause passed -- because the corpus clause
compared a manifest's digest to a recomputation of the same manifest, which is a
tautology, and because the Adapter kept its own shorter list of the files that
determine a cell.

So the tests below are mostly about *comparison*: a digest computed against
itself proves nothing, one authoritative source list, an environment recorded
rather than assumed, and a shape closed at every level rather than at the top.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from dm.data import feedback as corpus_module
from dm.eval import feedback_contract as contract
from dm.eval.feedback import DEVELOPMENT, SCIENTIFIC
from dm.eval.feedback_evidence import (
    CELL_SHAPE,
    FEEDBACK_FIELDS,
    RECORD_FIELDS,
    corpus_identity,
    execution_environment,
    reconstruct,
    shape_faults,
    source_identity,
    state_dict_sha256,
    verify_report,
)
from dm.eval.provenance import canonical_digest


@pytest.fixture(scope="module")
def paired(tmp_path_factory):
    """One small paired corpus and its frozen manifest, built once."""
    directory = tmp_path_factory.mktemp("corpus")
    corpus = corpus_module.build(200, 40, data_seed=0,
                                 seed=contract.seed_for("derangement"))
    path = directory / "corpus.json"
    corpus_module.write_manifest(path, corpus_module.manifest(corpus))
    return corpus, path


# ---------------------------------------------------------------------------
# corpus identity


def test_the_corpus_check_compares_programs_and_not_a_manifest_with_itself(paired):
    """The F5b defect, reproduced and then repaired.

    A manifest's canonical digest and its file hash both identify *the manifest*.
    Comparing one against a recomputation of the other says the file was not
    edited; it says nothing about which programs a checkpoint saw. The check that
    carries content is the fingerprint: `blake2b` over length-prefixed programs
    in order, produced independently by the builder and by the training loop.
    """
    corpus, path = paired
    balance = corpus_module.balance_report(corpus)
    record = {"corpus": balance["corpus"]["relational"], "config": {"data_seed": 0}}
    identity = corpus_identity(path, record)
    assert identity["matches"] is True
    assert identity["differences"] == []
    assert identity["record_fingerprint"] == balance["corpus"]["relational"]

    # The two manifest digests are different byte strings and are named as such.
    assert identity["file_sha256"] != identity["canonical_payload_sha256"]
    assert identity["canonical_payload_sha256"] == identity["recorded_canonical_sha256"]


def test_a_run_on_another_data_seed_is_caught_however_valid_its_manifest(paired):
    """Exactly the artifact F5b produced six times: an internally consistent
    manifest describing a corpus the run never saw."""
    _, path = paired
    record = {
        "corpus": {"train": "c4d24e9dc75fecd3", "val": "6ebe4c3b5c1bdfae",
                   "n_train": 100_000, "n_val": 1_000,
                   "bytes_train": 5_414_908, "bytes_val": 53_963},
        "config": {"data_seed": contract.DEVELOPMENT_DATA_SEED},
    }
    identity = corpus_identity(path, record)
    assert identity["matches"] is False
    assert any("data_seed" in line for line in identity["differences"])
    assert any(line.startswith("train:") for line in identity["differences"])


def test_the_arm_is_named_rather_than_assumed(paired):
    """The two arms share a validation half and differ only in train, so reading
    the wrong one compares a control's identity to a relational run's and finds
    the val halves agreeing."""
    corpus, path = paired
    balance = corpus_module.balance_report(corpus)
    record = {"corpus": balance["corpus"]["relational"], "config": {"data_seed": 0}}
    assert corpus_identity(path, record, structure="relational")["matches"]
    crossed = corpus_identity(path, record, structure="relation_destroyed")
    assert crossed["matches"] is False
    assert all(not line.startswith("val:") for line in crossed["differences"])

    with pytest.raises(ValueError, match="unknown training structure"):
        corpus_identity(path, record, structure="whatever")


def test_a_record_with_no_fingerprint_is_incomplete_and_never_a_match(paired):
    _, path = paired
    assert corpus_identity(path, {"config": {"data_seed": 0}})["matches"] is False


# ---------------------------------------------------------------------------
# source identity


def test_there_is_exactly_one_qualification_source_list():
    """F5b kept a second, shorter list in the Adapter: eight files against the
    contract's twelve, so `dm/data/dataset.py`, `dm/data/synthetic.py`,
    `dm/isa/codec.py` and `dm/isa/unroll.py` determined the batching, the corpus
    bytes and the encoding of every cell without entering any cell's digest.

    A duplicated list is not provenance; it is two provenances, and the shorter
    one wins silently. This pins that the Adapter has no list of its own.
    """
    import scripts.feedback_qualify as adapter

    assert not hasattr(adapter, "QUALIFICATION_SOURCES")
    identity = source_identity()
    assert set(identity["files"]) == set(contract.QUALIFICATION_SOURCES)
    for name in ("dm/data/dataset.py", "dm/data/synthetic.py", "dm/isa/codec.py",
                 "dm/isa/unroll.py"):
        assert name in identity["files"], name
    assert all(Path(name).exists() for name in identity["files"])
    assert len(identity["combined"]) == 64


def test_the_qualification_hashes_everything_that_trains_or_judges_a_cell():
    """Two questions, one digest set: what trained the cell, and what judged it.
    A freeze that hashed only the first could apply a repaired gate under a
    record claiming the unrepaired one."""
    assert set(contract.TRAINING_SOURCES) <= set(contract.QUALIFICATION_SOURCES)
    for name in ("dm/eval/feedback.py", "dm/eval/feedback_qualification.py"):
        assert name in contract.QUALIFICATION_SOURCES, name


# ---------------------------------------------------------------------------
# environment


def test_every_declared_environment_field_is_recorded():
    body = execution_environment(device="cpu")
    assert set(body) == set(contract.REQUIRED_ENVIRONMENT_FIELDS)
    # Present-and-null is a value; absent is `incomplete`. An unset variable has
    # to be distinguishable from one nobody looked at.
    assert set(body["environment_variables"]) == set(
        contract.RECORDED_ENVIRONMENT_VARIABLES)
    assert isinstance(body["deterministic_algorithms"], bool)
    assert body["device"] == "cpu"
    # Whatever else moves, these three are what a reproducibility claim is about.
    assert body["torch"] == torch.__version__
    assert body["torch_num_threads"] == torch.get_num_threads()


# ---------------------------------------------------------------------------
# reconstruction


def test_reconstruction_is_a_load_and_not_a_recorded_boolean(tmp_path):
    """The load that matters is today's `Config` accepting yesterday's state
    dictionary, and only an actual load can perform it."""
    from dm.models.transformer import Config, DrawingLM

    cfg = Config(vocab_size=258, d_model=16, n_layers=2, n_heads=2, max_len=64,
                 feedback_schema=contract.QUALIFICATION_FEEDBACK_SCHEMA)
    model = DrawingLM(cfg)
    path = tmp_path / "cell.pt"
    torch.save({"cfg": {**cfg.__dict__}, "state": model.state_dict(),
                "provenance": DEVELOPMENT}, path)

    rebuilt, evidence = reconstruct(path)
    assert evidence["strict"] is True
    assert evidence["params_match_config"] is True
    assert evidence["provenance"] == DEVELOPMENT
    assert evidence["feedback_schema"] == contract.QUALIFICATION_FEEDBACK_SCHEMA
    assert len(evidence["checkpoint_sha256"]) == 64
    assert rebuilt.n_params() == model.n_params()

    # A renamed tensor is what `strict=True` exists to catch, and it is the load
    # F7 and F8 will do on the pilot's own weights.
    state = model.state_dict()
    state["fuse.renamed"] = state.pop("fuse.norm.weight")
    torch.save({"cfg": {**cfg.__dict__}, "state": state}, path)
    with pytest.raises(RuntimeError):
        reconstruct(path)


def test_model_state_digest_ignores_torch_container_identity(tmp_path):
    """Whole-file hashes identify artifacts, not reproducible model states.

    PyTorch's ZIP member names depend on the output filename. F5c also stored a
    different ``record_name`` in each replicate. The historical correction gate
    compared whole-file bytes, so it could fail even when every tensor agreed.
    The content digest must ignore both and must still move with a tensor.
    """
    state = {"weight": torch.arange(8, dtype=torch.float32).reshape(2, 4)}
    first = tmp_path / "replicate_1.pt"
    second = tmp_path / "replicate_2.pt"
    torch.save({"state": state, "record_name": "r1"}, first)
    torch.save({"state": state, "record_name": "r2"}, second)

    assert first.read_bytes() != second.read_bytes()
    assert state_dict_sha256(first) == state_dict_sha256(second)

    changed = {"weight": state["weight"].clone()}
    changed["weight"][0, 0] += 1
    torch.save({"state": changed, "record_name": "r2"}, second)
    assert state_dict_sha256(first) != state_dict_sha256(second)


# ---------------------------------------------------------------------------
# the declared shape


def test_the_declared_record_shape_is_the_shape_a_run_actually_writes():
    """The declaration is only worth what it costs to keep true. A one-step
    feedback run writes a real record here, so a new field in `dm.train` is a
    failure in this file rather than a silent read in the gate."""
    from dm.train import TrainConfig, train

    cfg = TrainConfig(codec="byte", data="synthetic", n_train=32, n_val=8, steps=2,
                      batch_size=4, max_len=128, eval_every=2, device="cpu",
                      tag="_evidence_shape_probe", gen_samples=4,
                      feedback_schema=contract.QUALIFICATION_FEEDBACK_SCHEMA,
                      artifact_provenance=DEVELOPMENT)
    try:
        record = train(cfg, verbose=False)
    finally:
        for suffix in (".json", ".pt"):
            Path(f"runs/_evidence_shape_probe{suffix}").unlink(missing_ok=True)

    assert set(record) == set(RECORD_FIELDS)
    assert set(record["feedback"]) == set(FEEDBACK_FIELDS)
    # Derived from the dataclass rather than declared, so it cannot drift.
    assert set(record["config"]) == set(CELL_SHAPE["record.config"])


def test_an_unexpected_key_is_reported_with_its_path():
    cell = {"record": {"config": {"seed": 0, "Delta": 1.0}, "surprise": 2},
            "hashes": {"record": {"recorded": "a", "recomputed": "a"}}}
    faults = shape_faults(cell, ("record", "hashes"))
    assert faults["record"] == ["surprise"]
    assert faults["record.config"] == ["Delta"]
    assert "hashes.record" not in faults

    # A level the shape does not name is not descended into, and says so by
    # being absent rather than by silently passing something it never looked at.
    assert "record.config.seed" not in CELL_SHAPE


# ---------------------------------------------------------------------------
# report validation


def _report(tmp_path, **overrides) -> Path:
    body = {
        "schema": contract.QUALIFICATION_SCHEMA,
        "provenance": DEVELOPMENT,
        "package": contract.F5B_PACKAGE.as_dict(),
        "cell": {"package": contract.F5B_PACKAGE.name, "schedule": "terminal_mix_v1",
                 "seed": 100, "replicate": 1,
                 "hashes": {"record": {"recorded": "x", "recomputed": "x",
                                       "path": str(tmp_path / "gone.bin")}}},
    }
    body.update(overrides)
    body["report_sha256"] = canonical_digest(body, "report_sha256")
    path = tmp_path / "cell.json"
    path.write_text(json.dumps(body, indent=1, sort_keys=True))
    return path


def test_a_report_is_checked_against_its_own_bytes_in_a_later_process(tmp_path):
    path = _report(tmp_path)
    verified = verify_report(path)
    assert verified["envelope"]["verified"] is True
    assert verified["envelope"]["replicate"] == 1
    # The artifact behind the hash is gone, and the recomputation says so rather
    # than leaving the report's own claim standing.
    assert verified["cell"]["hashes"]["record"]["recomputed"] == "missing:record"


def test_an_edited_report_no_longer_matches_its_payload_digest(tmp_path):
    path = _report(tmp_path)
    body = json.loads(path.read_text())
    body["cell"]["seed"] = 101
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="incomplete"):
        verify_report(path)


def test_a_scientific_artifact_is_refused_before_anything_else(tmp_path):
    """Both directions fail closed: an engineering smoke cannot qualify anything,
    and a scientific cell is estimation data whose use here would let the pilot
    select its own configuration."""
    path = _report(tmp_path, provenance=SCIENTIFIC)
    with pytest.raises(ValueError, match="refused"):
        verify_report(path)
