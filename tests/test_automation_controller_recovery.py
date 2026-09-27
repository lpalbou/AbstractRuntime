"""Crash around every controller commit -> exactly one child per attempt (contract A/B, C1, C8).

A process "crash" is an exception raised by the store at one commit point of
one controller decision (before its intent save, before its ledger append,
after the append but before the state save) or at the controller's
parent-wait save after the child was created. The host then restarts: fresh
store objects over the same files, and the controller is driven again.
"""

from __future__ import annotations

import pytest

from automation_harness import Clock, automation_state, children, create, drive, make_runtime, make_stores
from abstractruntime.automations import list_occurrences, occurrence_run_id
from abstractruntime.automations.ledger import automation_records
from abstractruntime.core.models import RunStatus, WaitReason

STORES = ["json", "sqlite"]
DECISIONS = ["automation:admitted:", "automation:dispatched:", "automation:retry_scheduled:", "automation:completed:"]


class Crash(Exception):
    pass


class CrashingStores:
    def __init__(self, run_store, ledger_store, *, key_prefix, point):
        self.fired = False
        self.appended = False
        outer = self

        def intent_key(run):
            intent = ((run.vars.get("_runtime") or {}).get("automation") or {}).get("intent") if isinstance(run.vars, dict) else None
            return intent.get("key") if isinstance(intent, dict) else None

        class Runs:
            def __getattr__(self, name):
                return getattr(run_store, name)

            def save(self, run):
                if not outer.fired:
                    key = intent_key(run)
                    if point == "intent_save" and key and key.startswith(key_prefix):
                        outer.fired = True
                        raise Crash(f"crash before saving the intent of {key}")
                    if point == "state_save" and outer.appended and not key:
                        outer.fired = True
                        raise Crash("crash after the decision append, before the state save")
                    if (
                        point == "parent_wait_save"
                        and run.status == RunStatus.WAITING
                        and run.waiting is not None
                        and run.waiting.reason == WaitReason.SUBWORKFLOW
                    ):
                        outer.fired = True
                        raise Crash("crash after the child was created, before the parent wait is saved")
                return run_store.save(run)

        class Ledger:
            def __getattr__(self, name):
                return getattr(ledger_store, name)

            def append(self, record):
                key = record.idempotency_key or ""
                if not outer.fired and point == "append" and key.startswith(key_prefix):
                    outer.fired = True
                    raise Crash(f"crash before appending {key}")
                ledger_store.append(record)
                if key.startswith(key_prefix):
                    outer.appended = True

        self.run_store = Runs()
        self.ledger_store = Ledger()


def _counts(runtime, aid):
    out = {}
    for rec in automation_records(runtime.ledger_store, aid):
        out[rec["key"]] = out.get(rec["key"], 0) + 1
    return out


def _run_with_crash(tmp_path, monkeypatch, kind, *, key_prefix, point, workflow_id="flaky", fail_until=2):
    clock = Clock(monkeypatch)
    runtime = make_runtime(*make_stores(kind, tmp_path))
    aid = create(runtime, clock, workflow_id=workflow_id, input_data={"prompt": "p", "fail_until": fail_until},
                 trigger={"source_id": "schedule", "source_version": 1, "config": {"start_at": "2026-01-01T00:00:00Z", "every": "1h"}})
    crashing = CrashingStores(*make_stores(kind, tmp_path), key_prefix=key_prefix, point=point)
    crashed = make_runtime(crashing.run_store, crashing.ledger_store)
    crashed_once = False
    for now in ("2026-01-01T00:00:00+00:00", "2026-01-01T00:00:30+00:00"):
        clock.set(now)
        try:
            _wake_and_drive(crashed, aid)
        except Crash:
            crashed_once = True
            break
    # Restart and let the automation finish its first occurrence.
    runtime = make_runtime(*make_stores(kind, tmp_path))
    for now in ("2026-01-01T00:00:30+00:00", "2026-01-01T00:01:00+00:00"):
        clock.set(now)
        _wake_and_drive(runtime, aid)
    return runtime, aid, crashed_once, crashing


def _wake_and_drive(runtime, aid):
    from automation_harness import wake

    wake(runtime, aid)
    drive(runtime, aid)


@pytest.mark.parametrize("kind", STORES)
@pytest.mark.parametrize("key_prefix", DECISIONS)
@pytest.mark.parametrize("point", ["intent_save", "append", "state_save"])
def test_crash_around_each_controller_decision(tmp_path, monkeypatch, kind, key_prefix, point):
    runtime, aid, crashed, _ = _run_with_crash(tmp_path, monkeypatch, kind, key_prefix=key_prefix, point=point)
    assert crashed, "the injected crash point was never reached"
    kids = children(runtime, aid)
    assert [k.run_id for k in kids] == [
        occurrence_run_id(aid, revision=1, index=1),
        occurrence_run_id(aid, revision=1, index=1, attempt=2),
    ]
    assert all(k.status == RunStatus.COMPLETED for k in kids)
    counts = _counts(runtime, aid)
    assert all(n == 1 for n in counts.values()), counts
    st = automation_state(runtime, aid)
    assert st["intent"] is None and st["pending_occurrence"] is None and st["next_index"] == 2
    row = list_occurrences(runtime, aid)["items"][0]
    assert (row["status"], row["attempts"]) == ("completed", 2)


@pytest.mark.parametrize("kind", STORES)
def test_crash_before_the_parent_wait_is_saved_reattaches_to_the_same_child(tmp_path, monkeypatch, kind):
    runtime, aid, crashed, _ = _run_with_crash(
        tmp_path, monkeypatch, kind, key_prefix="-", point="parent_wait_save", workflow_id="echo", fail_until=0
    )
    assert crashed
    kids = children(runtime, aid)
    assert [k.run_id for k in kids] == [occurrence_run_id(aid, revision=1, index=1)]
    done = [r["payload"] for r in automation_records(runtime.ledger_store, aid, "automation.completed")]
    assert [(d["status"], d["attempts"]) for d in done] == [("completed", 1)]
