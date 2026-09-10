"""F4: the relation-destroyed training control.

`I_structure` is only "the gain is specific to relational structure" if the two
arms differ in the relation and in nothing else. So most of this file is a list
of things that must be **exactly** equal between them -- program lengths,
skeletons, the corpus byte multiset, the coordinate marginals -- and one thing
that must have changed, namely which continuation follows which prefix.

The equalities are exact rather than toleranced on purpose. A permutation of
whole blocks cannot move any of them, so a drift is a defect and never noise, and
a tolerance would only decide how large a defect to accept.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import replace

import pytest

from dm.data import feedback as fb
from dm.data import synthetic
from dm.isa.asm import parse
from dm.vm.interp import VM

SEED = 20260817


def _corpus(n_train: int = 200, n_val: int = 40, seed: int = SEED):
    return fb.build(n_train, n_val, data_seed=0, seed=seed)


CORPUS = _corpus()
REPORT = fb.balance_report(CORPUS)


# ---------------------------------------------------------------------------
# provenance


def test_the_provenance_stream_is_the_ordinary_corpus():
    """The builder keeps the L1 source, and must otherwise be the same loop.

    A second sampler that merely looked similar would give the destroyed arm a
    different corpus from every other Direction 3 run, and the difference would
    be invisible in every number that follows.
    """
    ordinary = synthetic.dataset(200, seed=0, tier=1, flatten=True)
    with_source = fb.generate_with_provenance(200, seed=0)
    assert [source.flat for source in with_source] == ordinary


def test_every_claimed_copy_span_is_the_first_copy_translated():
    """The whole control rests on cutting at the right byte offsets, and the
    offsets come from arithmetic over the L1 stream. Checked against the flat
    bytes it claims to describe, so an off-by-one fails the build rather than
    deranging the wrong region."""
    for source in fb.generate_with_provenance(60, seed=0):
        fb.verify_scopes(source.flat, list(source.scopes))
        for scope in source.scopes:
            assert scope.span(scope.count)[1] <= len(source.flat)


def test_a_wrong_step_is_caught_rather_than_trusted():
    """A displacement that does not describe the bytes it is attached to. The
    check is against the flat program, so a claim the corpus does not support
    fails the build instead of pointing the derangement somewhere plausible."""
    source = next(s for s in fb.generate_with_provenance(60, seed=0) if s.scopes)
    wrong = [replace(scope, dx=scope.dx + 1, dy=scope.dy + 1)
             for scope in source.scopes]
    with pytest.raises(ValueError, match="is not copy 1 translated"):
        fb.verify_scopes(source.flat, wrong)


def test_only_continuation_copies_are_taken():
    """Copies 1 and 2 establish the step; copy 3 onward is what it implies.
    Leaving ordinal 4 in place would leave a longer-range copy of the relation
    the control exists to remove."""
    assert fb.FIRST_CONTINUATION_ORDINAL == 3
    assert min(REPORT["ordinals"]) == 3
    assert set(REPORT["ordinals"]) <= {3, 4}
    for block in CORPUS.blocks:
        assert block.ordinal >= 3


# ---------------------------------------------------------------------------
# the four exact equalities


def test_the_two_arms_have_the_same_programs_of_the_same_lengths():
    assert len(CORPUS.relational) == len(CORPUS.destroyed) == REPORT["n_train"]
    assert REPORT["exact"]["program_lengths_preserved"]
    assert (REPORT["program_bytes"]["relational_total"]
            == REPORT["program_bytes"]["destroyed_total"])


def test_the_two_arms_have_the_same_skeleton_program_for_program():
    """Equal byte length is not enough: `MOVE x y` and `CIRCLE r` plus `FILL`
    occupy the same three bytes and are not interchangeable."""
    assert REPORT["exact"]["skeletons_preserved"]
    for left, right in zip(CORPUS.relational, CORPUS.destroyed):
        assert fb.skeleton(left) == fb.skeleton(right)


def test_the_corpus_byte_multiset_is_identical():
    """Blocks move whole, so every byte survives somewhere. A difference here
    means a donor of the wrong length or shape got through."""
    assert REPORT["exact"]["byte_multiset_preserved"]
    assert (Counter(b for p in CORPUS.relational for b in p)
            == Counter(b for p in CORPUS.destroyed for b in p))


def test_the_coordinate_marginals_are_identical():
    """The confound most likely to be mistaken for the effect: an arm whose
    coordinates were distributed differently would differ in how predictable its
    bytes are, before any relation entered it."""
    assert REPORT["exact"]["coordinate_marginals_preserved"]


def test_the_validation_split_is_the_same_bytes_for_both_arms():
    """One held-out set, always relational. A control arm validated on its own
    intervention would be measuring how well it fits its own control."""
    other = _corpus(seed=SEED + 1)
    assert CORPUS.val == other.val
    assert REPORT["exact"]["relational_train_val_disjoint"]
    assert REPORT["exact"]["destroyed_train_val_disjoint"]


# ---------------------------------------------------------------------------
# the derangement


def test_no_block_keeps_its_own_donor_slot():
    assert REPORT["exact"]["no_identity_donor"]
    assert all(CORPUS.donors[i] != i for i in range(len(CORPUS.blocks)))


def test_no_donor_reproduces_the_relation_it_is_meant_to_destroy():
    """A stranger's block that happens to equal the step's own continuation would
    leave the relation intact under a different name."""
    assert REPORT["exact"]["no_relation_preserving_donor"]
    for index, block in enumerate(CORPUS.blocks):
        assert CORPUS.blocks[CORPUS.donors[index]].content != block.implied


def test_every_block_donates_exactly_once():
    """A permutation, so the corpus-wide block multiset is preserved exactly and
    donor reuse cannot concentrate on a few popular blocks."""
    assert REPORT["exact"]["donor_reuse_is_uniform"]
    assert sorted(CORPUS.donors.values()) == list(range(len(CORPUS.blocks)))


def test_donors_stay_inside_their_stratum():
    """Byte length, skeleton and target ordinal. A donor from another stratum
    would change a program's length or its skeleton, which is the confound the
    whole construction exists to avoid."""
    for index, block in enumerate(CORPUS.blocks):
        assert CORPUS.blocks[CORPUS.donors[index]].stratum == block.stratum


def test_the_derangement_is_a_function_of_the_corpus_and_the_seed():
    """Model-blind and reproducible. Nothing here loads a checkpoint, and no seed
    is ever retried because a balance number looks inconvenient."""
    again = fb.derange(CORPUS.blocks, seed=SEED)[0]
    assert again == CORPUS.donors
    other = fb.derange(CORPUS.blocks, seed=SEED + 1)[0]
    assert other != CORPUS.donors


def test_a_stratum_that_cannot_be_deranged_fails_the_build():
    """Never dropped. Dropping would shorten the destroyed arm's programs, and
    then the arms would differ in length as well as in the relation."""
    lonely = [block for block in CORPUS.blocks
              if block.stratum == CORPUS.blocks[0].stratum][:1]
    with pytest.raises(fb.Unbalanced, match="cannot be deranged"):
        fb.derange(lonely, seed=SEED)


def test_a_stratum_with_no_legal_donor_at_all_fails_rather_than_relaxes():
    """Two blocks that each imply the other's content: every candidate is either
    itself or reproduces its own relation, and there is no repair."""
    stratum = CORPUS.blocks[0].stratum
    pair = [block for block in CORPUS.blocks if block.stratum == stratum][:2]
    left, right = pair
    impossible = [replace(left, implied=right.content),
                  replace(right, implied=left.content)]
    with pytest.raises(fb.Unbalanced, match="no legal donor"):
        fb.derange(impossible, seed=SEED)


# ---------------------------------------------------------------------------
# the intervention has to bite


def test_the_splice_actually_changes_every_continuation():
    assert REPORT["intervention"]["content_change_rate"] > 0.5
    assert REPORT["intervention"]["seam_bigram_total_variation"] > 0.0
    assert fb.accepts(REPORT)


def test_the_report_publishes_program_level_treatment_prevalence():
    """Block prevalence is not program prevalence and must be reported apart."""
    treatment = REPORT["intervention"]["program_treatment"]
    touched = {block.program for block in CORPUS.blocks}
    assert treatment["treated_programs"] == len(touched)
    assert treatment["untreated_programs"] == REPORT["n_train"] - len(touched)
    assert treatment["treated_fraction"] == pytest.approx(
        len(touched) / REPORT["n_train"]
    )


def test_the_destroyed_arm_differs_from_the_relational_one_program_by_program():
    changed = sum(1 for a, b in zip(CORPUS.relational, CORPUS.destroyed) if a != b)
    touched = len({block.program for block in CORPUS.blocks})
    assert changed == touched > 0


def test_a_report_that_fails_an_exact_invariant_is_not_accepted():
    broken = json.loads(json.dumps(REPORT))
    broken["exact"]["byte_multiset_preserved"] = False
    assert not fb.accepts(broken)
    unbitten = json.loads(json.dumps(REPORT))
    unbitten["intervention"]["seam_bigram_total_variation"] = 0.0
    assert not fb.accepts(unbitten)


# ---------------------------------------------------------------------------
# the destroyed programs are still programs


def test_every_destroyed_program_parses_and_runs():
    """A donor is real bytecode of the same skeleton, so the result has to remain
    executable. A control corpus that faulted where the relational one did not
    would differ in validity as well as in the relation."""
    vm = VM()
    for left, right in zip(CORPUS.relational, CORPUS.destroyed):
        assert len(list(parse(right))) == len(list(parse(left)))
        before, after = vm.run(left), vm.run(right)
        assert [f.kind for f in after.faults] == [f.kind for f in before.faults]
        assert len(after.strokes) == len(before.strokes)


# ---------------------------------------------------------------------------
# the frozen record


def test_the_manifest_carries_the_donor_map_and_hashes_itself():
    """A census without the map cannot be re-checked, and the map is what an
    auditor replays."""
    body = fb.manifest(CORPUS)
    assert body["donor_map"] == [CORPUS.donors[i]
                                 for i in range(len(CORPUS.blocks))]
    assert len(body["blocks"]) == len(CORPUS.blocks)
    assert body["balance"]["corpus"]["relational"] != body["balance"]["corpus"]["destroyed"]
    assert body["balance"]["intervention"]["program_treatment"]
    assert set(body["vm_census"]["arms"]) == {
        "relational", "destroyed", "validation"
    }
    assert body["vm_census"]["paired_relational_destroyed"] == {
        "fault_census_equal": True,
        "stroke_census_equal": True,
    }


def test_a_manifest_edited_after_freezing_is_refused(tmp_path):
    path = tmp_path / "corpus.json"
    fb.write_manifest(path, fb.manifest(CORPUS))
    assert fb.load_manifest(path)["schema"] == fb.MANIFEST_SCHEMA

    tampered = json.loads(path.read_text())
    tampered["donor_map"][0] = tampered["donor_map"][1]
    path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="hash mismatch"):
        fb.load_manifest(path)


def test_a_manifest_without_the_audit_or_program_prevalence_is_incomplete(tmp_path):
    body = fb.manifest(CORPUS)
    body.pop("vm_census")
    path = tmp_path / "old-corpus.json"
    # Rehash the edited body so this test reaches the completeness check rather
    # than the independent content-address check.
    import hashlib

    body["corpus_sha256"] = hashlib.sha256(
        json.dumps({k: v for k, v in body.items() if k != "corpus_sha256"},
                   sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    fb.write_manifest(path, body)
    with pytest.raises(ValueError, match="VM census.*incomplete"):
        fb.load_manifest(path)


# ---------------------------------------------------------------------------
# F5b: acceptance fails closed


def test_a_faulted_census_is_refused_however_symmetric_it_is():
    """Equal faults in both arms is not a passing corpus.

    The paired equality check asks whether the intervention preserved the VM
    marginals; it says nothing about whether those marginals were acceptable.
    A control corpus in which every program faults identically would satisfy it,
    and the two arms would differ in the relation *and* in being broken.
    """
    census = fb.vm_census({
        "relational": CORPUS.relational,
        "destroyed": CORPUS.destroyed,
        "validation": CORPUS.val,
    })
    assert fb.audit_accepts(census)

    for arm in census["arms"].values():
        assert arm["faulted_programs"] == 0
        assert arm["valid_programs"] == arm["programs"] == arm["halted_programs"]

    faulted = json.loads(json.dumps(census))
    for arm in faulted["arms"].values():
        arm["faults"] = {"bad_opcode": 3}
        arm["fault_events"] = 3
        arm["faulted_programs"] = 3
        arm["valid_programs"] = arm["programs"] - 3
        arm["valid_halted_programs"] = min(arm["valid_halted_programs"],
                                           arm["valid_programs"])
    assert not fb.audit_accepts(faulted)

    unhalted = json.loads(json.dumps(census))
    for arm in unhalted["arms"].values():
        arm["halted_programs"] = arm["programs"] - 1
        arm["valid_halted_programs"] = arm["programs"] - 1
    assert not fb.audit_accepts(unhalted)


def test_a_treatment_fraction_that_does_not_reconcile_is_refused():
    """Three numbers that agree pairwise but not jointly. `treated + untreated ==
    n_train` was checked and the *fraction* was not, so a report could name a
    prevalence that no pair of its own counts produces."""
    inconsistent = json.loads(json.dumps(REPORT))
    treatment = inconsistent["intervention"]["program_treatment"]
    treatment["treated_fraction"] = treatment["treated_fraction"] / 2
    assert not fb.accepts(inconsistent)


def test_the_two_manifest_hashes_are_separate_and_both_verify(tmp_path):
    """The canonical payload digest and the file SHA-256 identify different byte
    strings. `PLAN.md` names calling one the other as a trap, so the loader
    returns both under names that cannot be swapped by accident."""
    path = tmp_path / "corpus.json"
    fb.write_manifest(path, fb.manifest(CORPUS))
    digests = fb.manifest_digests(path)
    assert digests["canonical_payload_sha256"] == digests["recorded_canonical_sha256"]
    assert digests["file_sha256"] != digests["canonical_payload_sha256"]
    assert len(digests["file_sha256"]) == len(digests["canonical_payload_sha256"]) == 64
    # Rewriting the same payload with different whitespace keeps the canonical
    # digest and changes the file hash: that is the whole distinction.
    body = json.loads(path.read_text())
    path.write_text(json.dumps(body, sort_keys=True, indent=4) + "\n")
    again = fb.manifest_digests(path)
    assert again["canonical_payload_sha256"] == digests["canonical_payload_sha256"]
    assert again["file_sha256"] != digests["file_sha256"]


def test_a_manifest_whose_census_is_faulted_cannot_be_loaded(tmp_path):
    body = fb.manifest(CORPUS)
    for arm in body["vm_census"]["arms"].values():
        arm["faults"] = {"bad_opcode": 1}
        arm["fault_events"] = 1
        arm["faulted_programs"] = 1
        arm["valid_programs"] = arm["programs"] - 1
    body["corpus_sha256"] = hashlib.sha256(
        json.dumps({k: v for k, v in body.items() if k != "corpus_sha256"},
                   sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path = tmp_path / "faulted-corpus.json"
    fb.write_manifest(path, body)
    with pytest.raises(ValueError, match="incomplete"):
        fb.load_manifest(path)


def test_the_manifest_shows_real_donors_for_human_review():
    """§6 F4 asks a person to look at donors before the fingerprint is accepted.
    Hex, so the reviewer sees bytes rather than a rendering that might be hiding
    the defect."""
    examples = fb.donor_examples(CORPUS, limit=4)
    assert len(examples) == 4
    for row in examples:
        assert row["own_content"] != row["donor_content"]
        assert row["own_content"] == row["implied_by_relation"]
        assert row["changed"] is True


def test_the_builder_loads_no_checkpoint():
    """Model-blind is structural, not a promise: the module imports nothing from
    `dm.models`, nothing from `dm.eval` and no tensor library, so there is no
    path by which a logit could reach a donor decision.

    Read off the import statements rather than the file text, so prose that
    merely *mentions* another module does not fail the check and an import
    hidden inside a function does not pass it.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(fb))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported.append("." * node.level + (node.module or ""))
    assert imported, "nothing imported, so the check is not looking at anything"
    for name in imported:
        assert "models" not in name, name
        assert "eval" not in name, name
        assert name.split(".")[0] not in ("torch", "numpy"), name


# ---------------------------------------------------------------------------
# the trainer sees the arm it asked for


def _config(**extra):
    from dm.train import TrainConfig

    return TrainConfig(data="feedback", n_train=200, n_val=40, data_seed=0,
                       extra={"flatten": True, **extra})


def test_the_trainer_gets_the_arm_its_config_names():
    """`structure` *is* the corpus here, the way `control` is for `composed`:
    two splits of the same programs in the same order, differing only in which
    continuation follows which prefix. Nothing else in a config tells them
    apart, so the record's corpus fingerprint has to."""
    from dm.data.fingerprint import fingerprint
    from dm.train import build_data

    relational, val_a = build_data(_config(structure="relational"))
    destroyed, val_b = build_data(_config(structure="relation_destroyed"))
    assert val_a == val_b
    assert relational != destroyed
    assert [len(p) for p in relational] == [len(p) for p in destroyed]
    assert (fingerprint(relational, val_a)["train"]
            != fingerprint(destroyed, val_b)["train"])


def test_the_default_arm_is_the_relational_one():
    from dm.train import build_data

    assert build_data(_config())[0] == build_data(_config(structure="relational"))[0]


def test_an_unknown_arm_is_refused_rather_than_defaulted():
    """A typo that silently trained the relational corpus under the control's
    name would make `I_structure` exactly zero for a reason no table could
    show."""
    from dm.train import build_data

    with pytest.raises(ValueError, match="unknown feedback training structure"):
        build_data(_config(structure="destroyed"))


def test_the_derangement_seed_is_not_a_configurable_knob():
    """A tunable control seed is a search over controls. The builder takes the
    seed the protocol names and the training path passes nothing else."""
    import inspect

    from dm.train import build_data

    source = inspect.getsource(build_data)
    assert 'seed=seed_for("derangement")' in source


def test_a_corpus_with_no_three_copy_repeat_has_nothing_to_destroy():
    with pytest.raises(fb.Unbalanced, match="no continuation blocks"):
        fb.build(4, 2, data_seed=0, seed=SEED, max_repeat=2, min_repeat=2)
