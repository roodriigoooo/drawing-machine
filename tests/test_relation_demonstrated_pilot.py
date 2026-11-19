"""Pilot split leakage and non-derivable-control failure modes."""
from dataclasses import replace

import pytest

from dm.relation.demonstrated import TaskInvalid, audit_negative
from dm.relation.demonstrated_pilot import audit_allocation, build_pilot_fixture


@pytest.fixture(scope="module")
def pilot():
    return build_pilot_fixture()


def changed(pilot, panel, row):
    return replace(pilot, panels=tuple((name, (row,) + cases[1:] if name == panel else cases)
                                       for name, cases in pilot.panels))


def test_allocation_has_declared_sizes_atoms_and_connected_components(pilot):
    result = audit_allocation(pilot)
    assert result["training_tuple_count"] == 64
    assert result["heldout_tuple_count"] == 8
    assert min(result["positive_panel_components"].values()) >= 32
    assert pilot == build_pilot_fixture()


def test_affine_tuple_leak_refused(pilot):
    train = dict(pilot.panels)["train"][0]
    with pytest.raises(TaskInvalid, match="affine tuple leaked"):
        audit_allocation(changed(pilot, "affine", train))


def test_source_family_leak_refused(pilot):
    train = dict(pilot.panels)["train"][0]
    with pytest.raises(TaskInvalid, match="source family leaked"):
        audit_allocation(changed(pilot, "family", train))


def test_training_prefix_duplicate_refused(pilot):
    train = dict(pilot.panels)["train"][0]
    with pytest.raises(TaskInvalid, match="training prefix leaked"):
        audit_allocation(changed(pilot, "iid", train))


def test_fixed_donor_splice_is_matched_and_preserves_membership(pilot):
    index = {case.case_id: case for case in pilot.cases}
    for case in pilot.cases:
        if case.stratum != "broken":
            continue
        recipient, donor = index[case.source_case_id], index[case.donor_case_id]
        assert recipient.case_id != donor.case_id
        assert case.prompt == recipient.prompt[:9] + donor.prompt[9:18] + recipient.prompt[18:]
        assert case.target == donor.target
        assert case.donor_source_ids == donor.source_ids
        assert len(case.prompt) == len(recipient.prompt) == len(donor.prompt)


def test_derivable_donor_target_cannot_be_accepted_as_negative(pilot):
    case = next(c for c in pilot.cases if c.stratum == "broken")
    # The query itself is always a production-supported identity COPY.
    with pytest.raises(TaskInvalid, match="negative COPY witness"):
        audit_negative(case.prompt, case.prompt[18:27])


def test_donor_links_count_towards_component_population(pilot):
    result = audit_allocation(pilot)
    assert result["positive_panel_components"] == {"iid": 32, "affine": 32, "family": 32}


def test_paired_recipe_freezes_sampler_variates_and_effective_defaults():
    from dm.relation.demonstrated_card import recipe, uniforms
    result = recipe()
    cells = result["cells"]
    assert cells[2]["batch_order"] == cells[3]["batch_order"]
    assert cells[4]["batch_order"] == cells[5]["batch_order"]
    assert cells[0]["batch_order"] != cells[1]["batch_order"]
    assert len(result["pilot_intermediate_case_ids"]) == 32
    assert {c["parameter_count"] for c in cells} == {824704, 839808}
    assert len(uniforms("fit", "one")) == 256
    assert uniforms("fit", "one") != uniforms("iid", "one")
    for cell in cells:
        assert len(cell["batch_order"]) == 1024
        assert cell["config"]["model"]["feedback_schema"] == "none"
        assert cell["config"]["model"]["dropout"] == 0
