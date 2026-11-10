"""Attributed visit corrections re-recall at the actor boundary; raw steering stays closed."""
from types import SimpleNamespace

import pytest

from abstractruntime.core.models import EffectType, StepPlan
from abstractruntime.core.runtime import EffectOutcome, Runtime
from abstractruntime.core.spec import WorkflowSpec
from abstractruntime.identity.visit_workflow import ReactMiddle, build_visit_workflow
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore
from abstractruntime.storage.steer_sidecar import InMemorySteerSidecar


def setup_visit():
    recalls = []

    def actor(run, ctx):
        return StepPlan(node_id="reason", complete_output={"messages": run.vars["context"]["messages"]})

    middle = ReactMiddle(nodes={"reason": actor})
    full = build_visit_workflow(SimpleNamespace(entity_id="entity:veya", name="Veya"),
                                participants=["person:operator"], budget_profile={"token_budget": 1000},
                                react_middle=middle)
    wf = WorkflowSpec(workflow_id=full.workflow_id, entry_node="reason", nodes=full.nodes)

    def recall(run, effect, next_node):
        recalls.append(effect.payload)
        return EffectOutcome.completed({"handles": [], "as_of_seq": 12})

    rt = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore(),
                 steer_store=InMemorySteerSidecar(), effect_handlers={EffectType.MEMORY_RECALL: recall})
    rid = rt.start(workflow=wf, vars={
        "_runtime": {"entity": {"entity_id": "entity:veya", "visit_id": "v1", "participants": ["person:operator", "entity:veya"]}},
        "_visit": {"participants": ["person:operator", "entity:veya"], "budget": {"token_budget": 1000}},
        "_turn": {"turn_id": "t1", "text": "Find a route.", "rendered_user": "Find a route."},
        "context": {"messages": [{"role": "user", "content": "Find a route."}]},
    })
    return rt, wf, rid, recalls


def test_correction_is_attributed_recalled_once_and_survives_resume():
    rt, wf, rid, recalls = setup_visit()
    seq = rt.queue_visit_guidance(rid, text="I cannot use stairs.", speaker="person:operator")
    assert seq == 1
    # Crash/rebuild after recall, before rendering. Queued words survive in
    # the durable stage and are not recalled/applied a second time.
    rt.tick(workflow=wf, run_id=rid, max_steps=1)
    rt = Runtime(run_store=rt.run_store, ledger_store=rt.ledger_store,
                 steer_store=rt._steer_store,
                 effect_handlers={EffectType.MEMORY_RECALL: lambda *args: pytest.fail("duplicate recall")})
    out = rt.tick(workflow=wf, run_id=rid)
    assert len(recalls) == 1
    assert recalls[0]["cue_text"] == "I cannot use stairs." and recalls[0]["channel"] == "workplace"
    # The correction's recall names each memory's topic too (the live activity panel reads every recall record).
    assert recalls[0]["with_topics"] is True
    messages = out.output["messages"]
    assert messages[0] == {"role": "user", "content": "Find a route."}
    assert len(messages) == 2
    assert "person:operator (operator-visit, correction):\nI cannot use stairs." in messages[1]["content"]
    assert messages[1]["metadata"]["seqs"] == [1]
    assert "I cannot use stairs." in out.vars["_turn"]["text"]


def test_empty_queue_has_no_recall_or_extra_model_work_and_raw_steer_stays_refused():
    rt, wf, rid, recalls = setup_visit()
    with pytest.raises(PermissionError):
        rt.steer(rid, "a raw bypass")
    with pytest.raises(PermissionError):
        rt.queue_visit_guidance(rid, text="forged voice", speaker="person:someone-else")
    out = rt.tick(workflow=wf, run_id=rid)
    assert recalls == [] and len(out.output["messages"]) == 1


def test_previous_turns_queued_words_are_retained_without_becoming_a_new_correction():
    rt, wf, rid, recalls = setup_visit()
    rt.queue_visit_guidance(rid, text="An old correction.", speaker="person:operator")
    run = rt.get_state(rid)
    run.vars["_turn"]["turn_id"] = "t2"
    rt.run_store.save(run)
    out = rt.tick(workflow=wf, run_id=rid)
    assert recalls == [] and len(out.output["messages"]) == 1
    retained = [r for r in rt.get_ledger(rid) if (r.get("effect") or {}).get("payload", {}).get("name") == "abstract.visit_guidance_retained"]
    assert retained and retained[0]["effect"]["payload"]["payload"]["messages"][0]["content"] == "An old correction."
