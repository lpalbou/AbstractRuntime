from __future__ import annotations

from typing import Any, Dict, List, Optional

from abstractruntime import InMemoryLedgerStore, InMemoryRunStore, RunState, RunStatus, Runtime, WorkflowSpec
from abstractruntime.core.models import Effect, EffectType, StepPlan
from abstractruntime.integrations.abstractcore.effect_handlers import make_llm_call_handler
from abstractruntime.integrations.abstractcore.llm_client import HttpResponse, RemoteAbstractCoreLLMClient, _normalize_local_response


class _StubLLM:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def generate(
        self,
        *,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.calls.append(
            {
                "prompt": prompt,
                "messages": messages,
                "system_prompt": system_prompt,
                "tools": tools,
                "params": dict(params or {}),
            }
        )
        return {"content": "ok", "trace_id": "tr_test", "metadata": {"trace_id": "tr_test"}}


def test_llm_call_injects_trace_metadata() -> None:
    stub = _StubLLM()
    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.LLM_CALL: make_llm_call_handler(llm=stub)},
    )

    def ask(run: RunState, ctx) -> StepPlan:
        return StepPlan(
            node_id="ask",
            effect=Effect(
                type=EffectType.LLM_CALL,
                payload={"prompt": "hello", "params": {"max_tokens": 5}},
                result_key="llm",
            ),
            next_node="done",
        )

    def done(run: RunState, ctx) -> StepPlan:
        return StepPlan(node_id="done", complete_output={"llm": run.vars.get("llm")})

    workflow = WorkflowSpec(workflow_id="trace_inject", entry_node="ask", nodes={"ask": ask, "done": done})

    run_id = runtime.start(workflow=workflow, actor_id="ar_test", session_id="sess_test")
    state = runtime.tick(workflow=workflow, run_id=run_id)

    assert state.status == RunStatus.COMPLETED
    assert stub.calls, "Expected stub LLM to be called"
    trace = (stub.calls[-1]["params"].get("trace_metadata") or {})
    assert trace.get("run_id") == run_id
    assert trace.get("actor_id") == "ar_test"
    assert trace.get("session_id") == "sess_test"


def test_normalize_local_response_extracts_trace_id() -> None:
    from abstractcore.core.types import GenerateResponse

    resp = GenerateResponse(content="hi", metadata={"trace_id": "tr_123"})
    out = _normalize_local_response(resp)
    assert out.get("trace_id") == "tr_123"
    assert isinstance(out.get("metadata"), dict)
    assert out["metadata"].get("trace_id") == "tr_123"


def test_remote_client_sends_trace_headers_and_reads_trace_id() -> None:
    captured: Dict[str, Any] = {}

    class _Sender:
        def post(self, url: str, *, headers: Dict[str, str], json: Dict[str, Any], timeout: float) -> HttpResponse:
            captured["url"] = url
            captured["headers"] = dict(headers)
            captured["json"] = dict(json)
            return HttpResponse(
                body={
                    "model": "m",
                    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                },
                headers={"X-AbstractCore-Trace-Id": "tr_remote"},
            )

    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://example.test",
        model="m",
        headers={"User-Agent": "test"},
        request_sender=_Sender(),
    )

    out = client.generate(prompt="hi", params={"trace_metadata": {"run_id": "r1", "actor_id": "a1"}})
    assert out.get("trace_id") == "tr_remote"
    assert captured["headers"].get("X-AbstractCore-Trace-Metadata")
    assert captured["headers"].get("X-AbstractCore-Run-Id") == "r1"
    assert captured["headers"].get("X-AbstractCore-Actor-Id") == "a1"
