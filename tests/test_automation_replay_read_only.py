"""Replaying an automation's history is a pure read (runtime 0847 acceptance).

After real occurrences ran, every history surface (controller summary,
occurrence list, attention, the exported history bundle of the controller tree,
each occurrence session's turns) is read through stores that REFUSE every
write, on a runtime whose provider and tool handlers REFUSE to run.
"""

from __future__ import annotations

import pytest

from automation_harness import Clock, at, create, drive, make_runtime, make_stores
from abstractruntime import EffectType, Runtime
from abstractruntime.automations import get_automation, list_attention, list_occurrences, pending_waits
from abstractruntime.history_bundle import export_run_history_bundle
from abstractruntime.session_turns import select_session_turns


class Forbidden(AssertionError):
    pass


class ReadOnlyStores:
    def __init__(self, run_store, ledger_store):
        class Runs:
            def __getattr__(self, name):
                if name in ("save", "create_if_absent", "delete"):
                    raise Forbidden(f"replay attempted run_store.{name}")
                return getattr(run_store, name)

        class Ledger:
            def __getattr__(self, name):
                if name in ("append", "append_chained", "delete"):
                    raise Forbidden(f"replay attempted ledger_store.{name}")
                return getattr(ledger_store, name)

        self.run_store = Runs()
        self.ledger_store = Ledger()


def _refuse(run, effect, default_next_node):
    raise Forbidden(f"replay attempted a {effect.type.value} effect")


@pytest.mark.parametrize("kind", ["json", "sqlite"])
def test_history_replay_makes_no_writes_and_no_provider_or_tool_calls(tmp_path, monkeypatch, kind):
    clock = Clock(monkeypatch)
    runtime = make_runtime(*make_stores(kind, tmp_path))
    aid = create(runtime, clock, input_data={"prompt": "price of ACME?", "notify": True})
    drive(runtime, aid)
    at(runtime, clock, aid, "2026-01-01T00:02:00+00:00")

    ro = ReadOnlyStores(*make_stores(kind, tmp_path))
    replay = Runtime(
        run_store=ro.run_store,
        ledger_store=ro.ledger_store,
        workflow_registry=runtime.workflow_registry,
        effect_handlers={EffectType.LLM_CALL: _refuse, EffectType.TOOL_CALLS: _refuse, EffectType.TOOL_INVOKE: _refuse},
    )
    summary = get_automation(replay.run_store, aid)
    assert summary["state"]["next_index"] == 3
    rows = list_occurrences(replay, aid)["items"]
    assert [r["index"] for r in rows] == [2, 1]
    assert len(list_attention(replay.ledger_store, aid)["items"]) == 2
    assert pending_waits(replay.run_store, aid) == []
    bundle = export_run_history_bundle(
        run_id=aid, run_store=replay.run_store, ledger_store=replay.ledger_store, include_subruns=True
    )
    assert bundle["root_run_id"] == aid
    assert {row["run_id"] for row in rows} <= set(bundle["ledgers"])  # the occurrence subtree was exported
    for row in rows:
        occurrence = replay.get_state(row["run_id"])
        turns = select_session_turns(replay.run_store, occurrence.session_id)
        assert [t.run_id for t in turns] == [row["run_id"]]
    # And the stores the automation actually used are unchanged by all of it.
    assert replay.get_ledger(aid) == runtime.get_ledger(aid)
