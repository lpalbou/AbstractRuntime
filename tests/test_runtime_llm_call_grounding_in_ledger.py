from __future__ import annotations

import re

import pytest

from abstractruntime.core.models import Effect, EffectType, StepPlan
from abstractruntime.core.runtime import EffectOutcome, Runtime
from abstractruntime.core.spec import WorkflowSpec
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore


pytestmark = pytest.mark.basic


def test_llm_call_grounding_is_visible_in_ledger_without_affecting_idempotency_key(monkeypatch) -> None:
    import abstractruntime.integrations.abstractcore.llm_client as llm_client

    monkeypatch.setattr(llm_client, "_system_context_header", lambda: "[2000-01-01 00:00:00 FR]")

    run_store = InMemoryRunStore()
    ledger_store = InMemoryLedgerStore()

    def llm_handler(run, effect, default_next_node):  # noqa: ARG001
        return EffectOutcome.completed({"content": "ok", "metadata": {}})

    runtime = Runtime(
        run_store=run_store,
        ledger_store=ledger_store,
        effect_handlers={EffectType.LLM_CALL: llm_handler},
    )

    def node(run, ctx):  # noqa: ARG001
        return StepPlan(
            node_id="n1",
            effect=Effect(type=EffectType.LLM_CALL, payload={"prompt": "hello"}),
            next_node=None,
        )

    wf = WorkflowSpec(workflow_id="wf", entry_node="n1", nodes={"n1": node})
    run_id = runtime.start(workflow=wf)

    # The idempotency key must be computed from the *original* effect payload (no grounding),
    # even though the ledger should include the injected grounding prefix for transparency.
    run0 = runtime.get_state(run_id)
    expected_key = runtime.effect_policy.idempotency_key(
        run=run0,
        node_id="n1",
        effect=Effect(type=EffectType.LLM_CALL, payload={"prompt": "hello"}),
    )

    runtime.tick(workflow=wf, run_id=run_id)

    records = ledger_store.list(run_id)
    assert records, "expected ledger entries"

    first = records[0]
    assert first.get("idempotency_key") == expected_key

    payload = (first.get("effect") or {}).get("payload") or {}
    prompt_sent = str(payload.get("prompt") or "")
    assert prompt_sent.startswith("[2000-01-01 00:00:00 FR] ")
    assert re.match(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} [A-Z]{2}\]", prompt_sent)
