"""CR-4 bounded sequential scope protocol and interval witnesses."""
import pytest
from test_relation_resources import _record

from dm.relation.resources import PhaseLedger, PhysicalObservationRefused


def _step(ledger, index):
    parent = 3 * index + 1
    ledger(_record("metadata", 0.25, scope_id=parent + 1, parent_scope_id=parent, start=index * 2))
    ledger(_record("train_update", 1.0, scope_id=parent + 2, parent_scope_id=parent, start=index * 2 + 0.25))
    ledger(_record("step", 1.5, scope_id=parent, start=index * 2))


def test_long_trace_retires_completed_scope_ids():
    ledger = PhaseLedger({"step": ("metadata", "train_update")})
    for index in range(10000):
        _step(ledger, index)
        assert not ledger._children_by_parent
        assert not hasattr(ledger, "_seen_scope_ids")
        assert len(ledger._totals) == 3
        assert len(ledger._residuals) == 1
    assert ledger.unattributed_seconds("step", ("metadata", "train_update")) == 2500
    before = ledger.totals()
    with pytest.raises(PhysicalObservationRefused, match="replayed"):
        _step(ledger, 0)
    assert before == ledger.totals()


@pytest.mark.parametrize("bad", ["overlap", "outside", "foreign", "duplicate"])
def test_invalid_topology_does_not_commit_totals(bad):
    ledger = PhaseLedger({"step": ("metadata", "train_update")})
    ledger(_record("metadata", 0.5, scope_id=2, parent_scope_id=1, start=1.0))
    with pytest.raises(PhysicalObservationRefused):
        ledger(_record("train_update", 0.5, scope_id=2 if bad == "duplicate" else 3,
                       parent_scope_id=4 if bad == "foreign" else 1,
                       start=1.25 if bad == "overlap" else 1.5))
        ledger(_record("step", 1.0, scope_id=1, start=0.0))
    assert not ledger.totals()
    with pytest.raises(PhysicalObservationRefused, match="failed"):
        _step(ledger, 2)
