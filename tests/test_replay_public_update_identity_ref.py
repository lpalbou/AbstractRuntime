"""A public update names the identity its run started with (gateway lane ask, ADR 0052): the replay
view folds and caches public updates by version, so the host cannot stamp them at serve time. The view
copies the run's `vars._runtime.identity_ref` (`{of, rev}`, set by the gateway on visit, summon, meet and
session runs) onto that run's public updates; a run without one gets no field, never an invented one."""
import pytest

from abstractruntime.identity.digital_identity import identity_ref, stamp_run_identity_ref
from abstractruntime.replay import ReplayView, replay

pytestmark = pytest.mark.basic


def event(run, seq):
    return {"run_id": run, "seq": seq, "step_id": f"s{seq}", "node_id": "parse", "status": "completed",
            "started_at": f"2027-01-01T12:00:{seq:02d}+00:00",
            "effect": {"type": "emit_event", "payload": {"name": "public_update", "payload": {
                "id": f"{run}:public_update:{seq}", "run_id": run, "text": "The shorter route has stairs."}}},
            "result": {"emitted": True}}


def snapshot(run, ref=None, **runtime):
    run_vars = {"_runtime": dict(runtime)}
    if ref is not None:
        stamp_run_identity_ref(run_vars, ref)
    return {"run_id": run, "workflow_id": "visit", "status": "running", "vars": run_vars}


def updates(view):
    return [m for m in view.to_json()["messages"] if m["kind"] == "public_update"]


def test_the_run_s_identity_ref_is_copied_onto_its_public_updates():
    view = replay({"R": [event("R", 1), event("R", 2)]}, {"R": snapshot("R", identity_ref("entity:castor", 3))},
                  root_run_id="R")
    assert [m["identity_ref"] for m in updates(view)] == [{"of": "entity:castor", "rev": 3}] * 2


def test_a_child_run_s_update_names_the_child_s_ref_not_the_parent_s():
    parent = {"run_id": "R", "seq": 1, "step_id": "spawn", "node_id": "agent", "status": "waiting",
              "effect": {"type": "start_subworkflow", "payload": {"workflow_id": "agent"}},
              "result": {"wait": {"reason": "subworkflow", "wait_key": "subworkflow:C"}}}
    view = replay({"R": [parent], "C": [event("C", 1)]},
                  {"R": snapshot("R", identity_ref("entity:castor", 3)),
                   "C": dict(snapshot("C", identity_ref("session:s1", 1)), parent_run_id="R")}, root_run_id="R")
    assert [m["identity_ref"] for m in updates(view)] == [{"of": "session:s1", "rev": 1}]


def test_a_snapshot_read_after_the_records_stamps_the_updates_already_folded():
    view = ReplayView("R")
    view.apply([event("R", 1)])
    assert "identity_ref" not in updates(view)[0]
    delta = view.apply(runs_by_id={"R": snapshot("R", identity_ref("entity:castor", 3))})
    assert updates(view)[0]["identity_ref"] == {"of": "entity:castor", "rev": 3}
    assert "R:public_update:1" in str(delta)  # the stamped message travels in the delta
    fresh = replay({"R": [event("R", 1)]}, {"R": snapshot("R", identity_ref("entity:castor", 3))}, root_run_id="R")
    assert view.to_json()["messages"] == fresh.to_json()["messages"]


@pytest.mark.parametrize("run_snapshot", [
    None,                                                      # no snapshot read at all
    snapshot("R"),                                             # a run without a ref
    snapshot("R", identity_ref={"of": "entity:castor"}),       # malformed: no rev
    snapshot("R", identity_ref={"of": "castor", "rev": 3}),    # malformed: subject without a namespace
    snapshot("R", identity_ref="entity:castor#3"),             # malformed: text, not {of, rev}
])
def test_no_ref_on_the_run_means_no_field_never_a_fake(run_snapshot):
    view = replay({"R": [event("R", 1)]}, {"R": run_snapshot} if run_snapshot else None, root_run_id="R")
    assert "identity_ref" not in updates(view)[0]


def test_the_cursor_changes_with_the_ref_and_not_for_a_run_without_one():
    records = {"R": [event("R", 1)]}
    without = replay(records, {"R": snapshot("R")}, root_run_id="R").cursor["token"]
    with_ref = replay(records, {"R": snapshot("R", identity_ref("entity:castor", 3))}, root_run_id="R").cursor["token"]
    next_rev = replay(records, {"R": snapshot("R", identity_ref("entity:castor", 4))}, root_run_id="R").cursor["token"]
    assert len({without, with_ref, next_rev}) == 3
    # A run without a ref folds exactly the snapshot it folded before refs existed: its token (an ETag
    # a client holds) does not change with this release.
    from abstractruntime.replay.view import _snapshot
    assert "identity_ref" not in _snapshot(snapshot("R"))
    # A run whose ref is withdrawn loses it from its updates (the view never keeps a stale one).
    view = replay(records, {"R": snapshot("R", identity_ref("entity:castor", 3))}, root_run_id="R")
    view.apply(runs_by_id={"R": snapshot("R")})
    assert "identity_ref" not in updates(view)[0] and view.cursor["token"] == without
