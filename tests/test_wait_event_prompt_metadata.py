from __future__ import annotations

from typing import Any


def test_wait_event_can_carry_prompt_choices_and_allow_free_text() -> None:
    """Regression test: EVENT waits can be used for durable ask+wait UX.

    This is important for thin clients: the runtime must persist enough UI metadata
    (prompt/choices) for a remote host to render the question and resume later.
    """
    from abstractruntime.core.models import Effect, EffectType, RunStatus, StepPlan, WaitReason
    from abstractruntime.core.runtime import Runtime
    from abstractruntime.core.spec import WorkflowSpec
    from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore

    def node(run: Any, ctx: Any) -> StepPlan:
        del run, ctx
        return StepPlan(
            node_id="n1",
            effect=Effect(
                type=EffectType.WAIT_EVENT,
                payload={
                    "wait_key": "wk_ask",
                    "prompt": "Pick one:",
                    "choices": ["a", "b"],
                    "allow_free_text": False,
                },
                result_key="_temp.answer",
            ),
            next_node=None,
        )

    wf = WorkflowSpec(workflow_id="wf_wait_event_prompt", entry_node="n1", nodes={"n1": node})
    runtime = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())

    run_id = runtime.start(workflow=wf, vars={})
    state = runtime.tick(workflow=wf, run_id=run_id, max_steps=10)

    assert state.status == RunStatus.WAITING
    assert state.waiting is not None
    assert state.waiting.reason == WaitReason.EVENT
    assert state.waiting.wait_key == "wk_ask"
    assert state.waiting.prompt == "Pick one:"
    assert state.waiting.choices == ["a", "b"]
    assert state.waiting.allow_free_text is False

    # The durable ledger must include the same metadata for replay across reconnects.
    ledger = runtime.get_ledger(run_id)
    assert isinstance(ledger, list) and ledger
    last = ledger[-1]
    assert isinstance(last, dict)
    wait = (last.get("result") or {}).get("wait") if isinstance(last.get("result"), dict) else None
    assert isinstance(wait, dict)
    assert wait.get("reason") == "event"
    assert wait.get("wait_key") == "wk_ask"
    assert wait.get("prompt") == "Pick one:"
    assert wait.get("choices") == ["a", "b"]
    assert wait.get("allow_free_text") is False


