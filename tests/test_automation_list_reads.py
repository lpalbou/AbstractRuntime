"""Listing reads that stay O(1) in an automation's history (2026-10-26 gateway load).

- `latest_occurrence` reads the controller ledger from its END and stops at the newest
  admission: the same row as `list_occurrences(limit=1)`, without `JsonlLedgerStore.list`;
- `pending_waits` / `pending_waits_by_automation` come from the WAITING run-index rows
  attributed to the automation, never a walk of the occurrence trees (`list_children`);
- `list_attention` reads back only to the seen floor;
- `normalize_occurrence_output(lazy=True)` resolves only the values it reads;
- the backward line reader yields exactly the file's lines, last first.
"""

from __future__ import annotations

import random
import types

import pytest

from abstractruntime.automations.attention import (
    list_attention,
    normalize_occurrence_output,
    pending_waits,
    pending_waits_by_automation,
)
from abstractruntime.automations.ledger import _reverse_lines, _step_record, automation_records_newest_first
from abstractruntime.automations.service import latest_occurrence, list_occurrences
from abstractruntime.core.models import RunState, RunStatus, WaitReason, WaitState
from abstractruntime.storage.artifacts import FileArtifactStore, artifact_ref
from abstractruntime.storage.json_files import JsonFileRunStore, JsonlLedgerStore
from abstractruntime.storage.observable import ObservableLedgerStore
from abstractruntime.storage.offloading import OffloadingLedgerStore

AUTO = "auto-1"
AT = "2026-10-26T09:00:00+00:00"


def _runtime(tmp_path):
    artifacts = FileArtifactStore(tmp_path / "artifacts")
    ledger = OffloadingLedgerStore(ObservableLedgerStore(JsonlLedgerStore(tmp_path / "ledger")), artifact_store=artifacts)
    return types.SimpleNamespace(run_store=JsonFileRunStore(tmp_path / "runs"), ledger_store=ledger, artifact_store=artifacts)


def _controller(rt, aid=AUTO, *, attention_seq=0):
    run = RunState(
        run_id=aid, workflow_id="controller", status=RunStatus.WAITING, current_node="wait",
        vars={"_meta": {"automation": {"title": "T", "revision": 1, "trigger": {"source_id": "manual", "source_version": 1}}},
              "_runtime": {"automation": {"next_index": 1, "attention_seq": attention_seq}}},
        created_at=AT, updated_at=AT, session_id=f"automation:{aid}",
        waiting=WaitState(reason=WaitReason.EVENT, wait_key=f"automation:{aid}:wake"),
    )
    rt.run_store.save(run)
    return run


def _record(rt, run, name, disc, payload):
    rt.ledger_store.append(_step_record(run, name=name, key=f"automation:{name}:{run.run_id}:{disc}", payload=payload, node_id="n", at=AT))


def _occurrences(rt, run, n, *, prompt_ref_at=None, attention_every=0):
    seq = 0
    for i in range(1, n + 1):
        prompt = f"prompt {i}"
        if prompt_ref_at == i:
            prompt = artifact_ref(rt.artifact_store.store_text(prompt, run_id=run.run_id).artifact_id)
        _record(rt, run, "automation.admitted", i, {"run_id": f"occ-{i}", "index": i, "revision": 1, "event_id": f"e{i}",
                                                     "trigger_envelope": {"fired_at": f"t{i}", "source_id": "manual", "source_version": 1},
                                                     "prepared": {"input_data": {"prompt": prompt}}})
        _record(rt, run, "automation.dispatched", f"d{i}", {"run_id": f"occ-{i}-a1", "index": i, "attempt": 1})
        # Controller records that are not occurrence records sit between them.
        rt.ledger_store.append(_step_record(run, name="automation.coalesced", key=f"c{i}", payload={"index": i}, node_id="n", at=AT))
        attention = None
        if attention_every and i % attention_every == 0:
            seq += 1
            attention = {"kind": "notify", "seq": seq, "title": f"n{seq}", "body": "b"}
        _record(rt, run, "automation.completed", i, {"run_id": f"occ-{i}-a1", "index": i, "status": "completed", "attempts": 1,
                                                      "finished_at": f"f{i}", "notify": None, "attention": attention})
    return seq


def _no_full_ledger_read(monkeypatch):
    def refuse(self, run_id):  # pragma: no cover - the assertion is that it is never called
        raise AssertionError(f"full ledger read of {run_id}")

    monkeypatch.setattr(JsonlLedgerStore, "list", refuse)


def test_latest_occurrence_is_list_occurrences_first_row_from_the_tail(tmp_path, monkeypatch) -> None:
    rt = _runtime(tmp_path)
    run = _controller(rt)
    _occurrences(rt, run, 30, prompt_ref_at=30)
    # An admitted occurrence still in flight is the latest one.
    _record(rt, run, "automation.admitted", 31, {"run_id": "occ-31", "index": 31, "revision": 1, "event_id": "e31",
                                                  "trigger_envelope": {"fired_at": "t31"}, "prepared": {"input_data": {"prompt": "p31"}}})
    expected = list_occurrences(rt, AUTO, limit=1)["items"][0]
    assert expected["index"] == 31 and expected["status"] == "admitted"
    _no_full_ledger_read(monkeypatch)
    assert latest_occurrence(rt, AUTO) == expected
    assert latest_occurrence(rt, AUTO, user_turn=False) == {**expected, "user_turn": None}


def test_latest_occurrence_completed_row_and_artifact_prompt(tmp_path, monkeypatch) -> None:
    rt = _runtime(tmp_path)
    run = _controller(rt)
    _occurrences(rt, run, 5, prompt_ref_at=5)
    expected = list_occurrences(rt, AUTO, limit=1)["items"][0]
    assert expected["user_turn"] == "prompt 5" and expected["status"] == "completed" and expected["run_ids"] == ["occ-5-a1"]
    _no_full_ledger_read(monkeypatch)
    assert latest_occurrence(rt, AUTO) == expected


def test_latest_occurrence_none_before_the_first_fire(tmp_path) -> None:
    rt = _runtime(tmp_path)
    _controller(rt)
    assert latest_occurrence(rt, AUTO) is None


def test_list_occurrences_resolves_prompt_artifacts_of_the_page_only(tmp_path, monkeypatch) -> None:
    rt = _runtime(tmp_path)
    run = _controller(rt)
    _occurrences(rt, run, 3, prompt_ref_at=1)
    loads = []
    real = FileArtifactStore.load
    monkeypatch.setattr(FileArtifactStore, "load", lambda self, aid: loads.append(aid) or real(self, aid))
    assert [r["index"] for r in list_occurrences(rt, AUTO, limit=2)["items"]] == [3, 2]
    assert loads == []
    assert list_occurrences(rt, AUTO, limit=3)["items"][-1]["user_turn"] == "prompt 1"
    assert len(loads) == 1


def test_list_attention_reads_back_to_the_seen_floor_only(tmp_path, monkeypatch) -> None:
    rt = _runtime(tmp_path)
    run = _controller(rt)
    latest = _occurrences(rt, run, 40, attention_every=2)
    assert latest == 20
    full = list_attention(rt.ledger_store, AUTO, after_seq=0, limit=500)
    assert [i["seq"] for i in full["items"]] == list(range(1, 21))
    seen_records = []
    real = automation_records_newest_first

    def counting(*a, **kw):
        for rec in real(*a, **kw):
            seen_records.append(rec)
            yield rec

    import abstractruntime.automations.attention as attention

    monkeypatch.setattr(attention, "automation_records_newest_first", counting)
    page = list_attention(rt.ledger_store, AUTO, after_seq=17, limit=2)
    assert [i["seq"] for i in page["items"]] == [18, 19] and page["next_cursor"] == "att1:19"
    # Occurrences 40..34 only: three unseen items (seq 20, 19, 18), the one at the floor (17)
    # and the quiet occurrences between them; the 33 older completions are never read.
    assert [r["payload"]["index"] for r in seen_records] == [40, 39, 38, 37, 36, 35, 34]
    assert list_attention(rt.ledger_store, AUTO, after_seq=20)["items"] == []


def _wait_run(rt, run_id, *, aid=AUTO, role="occurrence", index=1, parent=AUTO, reason=WaitReason.USER, prompt="Proceed?"):
    run = RunState(
        run_id=run_id, workflow_id="wf", status=RunStatus.WAITING, current_node="ask",
        vars={"_meta": {"occurrence": {"automation_id": aid, "occurrence_index": index, "attempt": 1, "role": role, "session_kind": "occurrence"}}},
        created_at=f"2026-10-26T09:0{index}:00+00:00", updated_at=AT, session_id="s", parent_run_id=parent,
        waiting=WaitState(reason=reason, wait_key=f"ask:{run_id}", prompt=prompt),
    )
    rt.run_store.save(run)
    return run


def test_pending_waits_come_from_the_waiting_index_not_the_trees(tmp_path, monkeypatch) -> None:
    rt = _runtime(tmp_path)
    _controller(rt)
    _controller(rt, "auto-2")
    _wait_run(rt, "occ-1", index=1)
    _wait_run(rt, "occ-2-desc", role="descendant", index=2, parent="occ-2")
    _wait_run(rt, "occ-3-timer", index=3, reason=WaitReason.UNTIL, prompt=None)  # not a wait on a person
    _wait_run(rt, "other-occ", aid="auto-2", index=1, parent="auto-2")
    # A discussion of the automation waiting on its person is not an occurrence wait.
    rt.run_store.save(RunState(run_id="disc", workflow_id="wf", status=RunStatus.WAITING, current_node="n",
                               vars={"_meta": {"discussion": {"automation_id": AUTO, "occurrence_index": 1}}},
                               created_at=AT, updated_at=AT, session_id="d",
                               waiting=WaitState(reason=WaitReason.USER, wait_key="ask:disc", prompt="?")))
    monkeypatch.setattr(JsonFileRunStore, "list_children", lambda *a, **k: (_ for _ in ()).throw(AssertionError("tree walk")))
    waits = pending_waits(rt.run_store, AUTO)
    assert [(w["run_id"], w["kind"], w["index"]) for w in waits] == [("occ-1", "ask_user", 1), ("occ-2-desc", "ask_user", 2)]
    grouped = pending_waits_by_automation(rt.run_store)
    assert {k: [w["run_id"] for w in v] for k, v in grouped.items()} == {AUTO: ["occ-1", "occ-2-desc"], "auto-2": ["other-occ"]}
    assert len(pending_waits(rt.run_store, AUTO, limit=1)) == 1


def test_lazy_output_resolves_only_what_it_reads(tmp_path) -> None:
    rt = _runtime(tmp_path)
    answer = artifact_ref(rt.artifact_store.store_text("the answer", run_id="r").artifact_id)
    missing = {"$artifact": "0" * 32}
    child = RunState(run_id="c", workflow_id="wf", status=RunStatus.COMPLETED, current_node="end",
                     vars={}, output={"response": answer, "success": True, "scratchpad": [missing] * 50}, created_at=AT, updated_at=AT)
    lazy = normalize_occurrence_output(child, artifact_store=rt.artifact_store, lazy=True)
    assert lazy == {"answer": "the answer", "success": True, "notify": None, "error": None}
    with pytest.raises(LookupError):
        normalize_occurrence_output(child, artifact_store=rt.artifact_store)
    # A result object behind a ref: read through.
    inner = artifact_ref(rt.artifact_store.store_json({"answer": "deep", "notify": True, "big": missing}, run_id="r").artifact_id)
    child2 = RunState(run_id="c2", workflow_id="wf", status=RunStatus.COMPLETED, current_node="end",
                      vars={}, output={"success": True, "result": inner}, created_at=AT, updated_at=AT)
    assert normalize_occurrence_output(child2, artifact_store=rt.artifact_store, lazy=True) == {
        "answer": "deep", "success": True, "notify": True, "error": None}


def test_reverse_lines_is_the_file_backwards(tmp_path) -> None:
    rng = random.Random(7)
    path = tmp_path / "f"
    for _ in range(500):
        lines = [bytes(rng.randrange(32, 127) for _ in range(rng.randrange(0, 90))) for _ in range(rng.randrange(1, 25))]
        data = b"\n".join(lines) + (b"\n" if rng.random() < 0.7 else b"")
        path.write_bytes(data)
        assert list(_reverse_lines(path, block=rng.randrange(1, 70))) == data.split(b"\n")[::-1]
