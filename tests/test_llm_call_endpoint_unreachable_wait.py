"""Root 1168 (operator 2026-11-08), runtime half: a model endpoint that cannot be reached (the inference
machine is off, asleep or unreachable — AbstractCore's `EndpointUnreachableError`) parks the run on a durable
EVENT wait instead of failing it. `resume` with no payload re-plans the node that issued the call and re-issues
the identical LLM_CALL; `cancel` ends the run as for any wait; any other provider error still fails as before.

The remote lane: the AbstractCore server answers such a condition with a 503 envelope carrying every field of the
exception; the runtime's remote client rebuilds the typed error from it (a generic 500 stays what it was). The
pickle round-trip pins core's `__reduce__` contract (core main 6c1e7a0e) because the error crosses process channels.
"""
from __future__ import annotations

import json
import pickle
from typing import Any, Dict, List

import pytest

from abstractcore.exceptions import EndpointUnreachableError, InvalidRequestError

from abstractruntime import Effect, EffectType, InMemoryLedgerStore, InMemoryRunStore, RunStatus, StepPlan, WorkflowSpec
from abstractruntime.core.models import WaitReason
from abstractruntime.core.runtime import Runtime
from abstractruntime.integrations.abstractcore.effect_handlers import make_llm_call_handler
from abstractruntime.replay.view import PERSON_FACING_KINDS, wait_kind
from abstractruntime.storage.artifacts import InMemoryArtifactStore

BASE_URL = "http://192.168.1.7:1234/v1"


def _unreachable(**over: Any) -> EndpointUnreachableError:
    fields: Dict[str, Any] = dict(
        provider="lmstudio", base_url=BASE_URL, cause="httpx.ConnectError",
        detail="[Errno 61] Connection refused", errno=61, elapsed_s=0.4, retryable=True,
    )
    fields.update(over)
    return EndpointUnreachableError(**fields)


class _LLM:
    """Raises the queued errors first, then answers."""

    def __init__(self, errors: List[Exception]) -> None:
        self.errors = list(errors)
        self.calls: List[Dict[str, Any]] = []

    def generate(self, *, prompt, messages=None, system_prompt=None, tools=None, media=None, params=None):
        self.calls.append({
            "prompt": prompt, "messages": messages, "system_prompt": system_prompt,
            "tools": tools, "media": media, "params": dict(params or {}),
        })
        if self.errors:
            raise self.errors.pop(0)
        return {"content": "the answer", "metadata": {}}


def _ask_node(run, ctx):
    del ctx
    return StepPlan(
        node_id="ask",
        effect=Effect(type=EffectType.LLM_CALL, payload={"prompt": "hello", "provider": "lmstudio", "model": "qwen"}, result_key="llm"),
        next_node="done",
    )


def _done_node(run, ctx):
    del ctx
    return StepPlan(node_id="done", complete_output={"content": (run.vars.get("llm") or {}).get("content")})


WORKFLOW = WorkflowSpec(workflow_id="unreachable-test", entry_node="ask", nodes={"ask": _ask_node, "done": _done_node})


def _runtime(llm: _LLM) -> Runtime:
    return Runtime(
        run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.LLM_CALL: make_llm_call_handler(llm=llm, artifact_store=InMemoryArtifactStore())},
    )


def _llm_rows(rt: Runtime, run_id: str) -> List[Dict[str, Any]]:
    return [r for r in rt.get_ledger(run_id) if (r.get("effect") or {}).get("type") == "llm_call"]


def _semantic(params: Dict[str, Any]) -> Dict[str, Any]:
    """The request as the provider sees it, minus the runtime's per-attempt handles (callbacks, events) and its
    per-issuance trace metadata (the new issuance has a new idempotency key by design)."""
    return {
        k: v for k, v in params.items()
        if k != "trace_metadata" and isinstance(v, (str, int, float, bool, list, dict, type(None)))
    }


# ------------------------------------------------------------------ the handler: park, never fail


def test_unreachable_endpoint_parks_the_run_as_a_signal_wait() -> None:
    llm = _LLM([_unreachable()])
    rt = _runtime(llm)
    run_id = rt.start(workflow=WORKFLOW)
    state = rt.tick(workflow=WORKFLOW, run_id=run_id)
    assert state.status == RunStatus.WAITING and not state.error
    w = state.waiting
    assert w.wait_key == f"endpoint_unreachable:{BASE_URL}" and w.reason == WaitReason.EVENT
    assert w.resume_to_node == "ask" and w.result_key is None and not w.prompt and not w.choices
    d = w.details
    assert d["kind"] == "endpoint_unreachable" and d["base_url"] == BASE_URL and d["provider"] == "lmstudio"
    assert d["cause"] == "httpx.ConnectError" and d["detail"] == "[Errno 61] Connection refused" and d["errno"] == 61
    assert d["elapsed_s"] == 0.4 and d["retryable"] is True and d["attempt"] == 1 and d["since"] and d["node_id"] == "ask"
    assert d["message"] == (
        f"the model endpoint {BASE_URL} did not answer "
        "(httpx.ConnectError: [Errno 61] Connection refused, after 0.4 s); waiting for it"
    )
    # A signal wait, not a question: a client shows "Waiting for ...".
    kind = wait_kind({"reason": "event", "wait_key": w.wait_key, "details": d}, "llm_call", run_id)
    assert kind == "signal" and kind not in PERSON_FACING_KINDS
    # The ledger carries the attempt, and no failure anywhere.
    rows = _llm_rows(rt, run_id)
    waiting_rows = [r for r in rows if r.get("status") == "waiting"]
    assert len(waiting_rows) == 1
    recorded = ((waiting_rows[0].get("result") or {}).get("wait") or {})
    assert recorded.get("wait_key") == w.wait_key and recorded.get("details", {}).get("attempt") == 1
    assert recorded["details"]["base_url"] == BASE_URL and recorded["details"]["errno"] == 61
    assert not [r for r in rt.get_ledger(run_id) if r.get("status") == "failed"]
    # Core's contract: an unreachable endpoint is not an unhealthy model.
    assert EndpointUnreachableError.counts_toward_breaker is False


def test_resume_with_no_payload_reissues_the_identical_call() -> None:
    llm = _LLM([_unreachable()])
    rt = _runtime(llm)
    run_id = rt.start(workflow=WORKFLOW)
    state = rt.tick(workflow=WORKFLOW, run_id=run_id)
    assert state.status == RunStatus.WAITING
    resumed = rt.resume(workflow=WORKFLOW, run_id=run_id, wait_key=state.waiting.wait_key, payload={}, max_steps=10)
    assert resumed.status == RunStatus.COMPLETED and resumed.output == {"content": "the answer"}
    # The identical call is re-issued: the node's effect payload is byte-identical in both ledger rows, and the
    # provider saw the same semantic request (the runtime's per-attempt cancel event / progress callback and
    # its wall-clock metadata block are the only differences).
    assert len(llm.calls) == 2
    first, second = llm.calls
    for key in ("messages", "system_prompt", "tools", "media"):
        assert first[key] == second[key]
    assert first["prompt"].endswith("hello") and second["prompt"].endswith("hello")
    assert _semantic(first["params"]) == _semantic(second["params"])
    rows = _llm_rows(rt, run_id)
    payloads = [r["effect"]["payload"] for r in rows if r.get("status") in ("waiting", "completed")]
    assert len(payloads) == 2 and payloads[0] == payloads[1] == {"prompt": "hello", "provider": "lmstudio", "model": "qwen"}
    assert "endpoint_waits" not in (resumed.vars.get("_runtime") or {}), "the outage counter ends with the success"
    terminal = [r.get("status") for r in rows if r.get("status") in ("waiting", "completed", "failed")]
    assert terminal == ["waiting", "completed"], "one row per attempt, no failure"
    assert len({r.get("idempotency_key") for r in rows}) == 2, "the re-issue is a new effect issuance"


def test_a_second_outage_counts_the_attempt_and_keeps_since() -> None:
    llm = _LLM([
        _unreachable(),
        _unreachable(detail="[Errno 60] Operation timed out", errno=60, elapsed_s=10.0, retryable=False),
    ])
    rt = _runtime(llm)
    run_id = rt.start(workflow=WORKFLOW)
    first = rt.tick(workflow=WORKFLOW, run_id=run_id)
    second = rt.resume(workflow=WORKFLOW, run_id=run_id, wait_key=first.waiting.wait_key, payload={}, max_steps=10)
    assert second.status == RunStatus.WAITING
    d1, d2 = first.waiting.details, second.waiting.details
    assert d2["attempt"] == 2 and d2["since"] == d1["since"]
    assert d2["errno"] == 60 and d2["retryable"] is False and "after 10.0 s" in d2["message"]
    assert len([r for r in _llm_rows(rt, run_id) if r.get("status") == "waiting"]) == 2


def test_cancel_ends_a_parked_run_as_today() -> None:
    llm = _LLM([_unreachable()])
    rt = _runtime(llm)
    run_id = rt.start(workflow=WORKFLOW)
    assert rt.tick(workflow=WORKFLOW, run_id=run_id).status == RunStatus.WAITING
    cancelled = rt.cancel_run(run_id, reason="operator")
    assert cancelled.status == RunStatus.CANCELLED
    assert len(llm.calls) == 1


def test_another_provider_error_still_fails_the_run() -> None:
    err = InvalidRequestError("the request is malformed", status_code=400)
    llm = _LLM([err] * 5)
    rt = _runtime(llm)
    run_id = rt.start(workflow=WORKFLOW)
    state = rt.tick(workflow=WORKFLOW, run_id=run_id)
    assert state.status == RunStatus.FAILED and "malformed" in str(state.error)


# ------------------------------------------------------------------ the error crosses processes whole


def test_endpoint_unreachable_error_survives_pickle_with_its_fields() -> None:
    e = _unreachable()
    back = pickle.loads(pickle.dumps(e))
    assert isinstance(back, EndpointUnreachableError)
    assert (back.provider, back.base_url, back.cause, back.detail, back.errno, back.elapsed_s, back.retryable) == (
        "lmstudio", BASE_URL, "httpx.ConnectError", "[Errno 61] Connection refused", 61, 0.4, True,
    )
    assert str(back) == str(e)


# ------------------------------------------------------------------ the remote lane: the server's 503 envelope


class _Resp:
    def __init__(self, status_code: int, body: Dict[str, Any]) -> None:
        self.status_code = status_code
        self.body = body
        self.text = json.dumps(body)

    def json(self) -> Dict[str, Any]:
        return self.body


class _HttpError(Exception):
    def __init__(self, response: _Resp) -> None:
        self.response = response
        super().__init__(f"HTTP {response.status_code}")


class _Sender:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.posts: List[Dict[str, Any]] = []

    def get(self, url: str, *, headers: Dict[str, str], timeout: float) -> Any:
        raise AssertionError(f"unexpected GET {url}")

    def post(self, url: str, *, headers: Dict[str, str], json: Dict[str, Any], timeout: float) -> Any:
        self.posts.append({"url": url, "json": json})
        raise self.error


ENVELOPE = {"error": {
    "message": f"lmstudio API error: could not connect to {BASE_URL} (httpx.ConnectError: [Errno 61] Connection refused) after 0.4s",
    "type": "endpoint_unreachable", "code": "endpoint_unreachable", "provider": "lmstudio", "base_url": BASE_URL,
    "cause": "httpx.ConnectError", "errno": 61, "elapsed_s": 0.4, "retryable": True,
}}


def _remote(sender: _Sender):
    from abstractruntime.integrations.abstractcore.llm_client import RemoteAbstractCoreLLMClient

    return RemoteAbstractCoreLLMClient(server_base_url="http://core.test", model="lmstudio/qwen", request_sender=sender)


def test_remote_client_rebuilds_the_error_from_the_503_envelope() -> None:
    sender = _Sender(_HttpError(_Resp(503, ENVELOPE)))
    with pytest.raises(EndpointUnreachableError) as info:
        _remote(sender).generate(prompt="hello", params={})
    e = info.value
    assert (e.provider, e.base_url, e.cause, e.errno, e.elapsed_s, e.retryable) == (
        "lmstudio", BASE_URL, "httpx.ConnectError", 61, 0.4, True,
    )
    assert e.detail == "[Errno 61] Connection refused", "recovered from core's composed message (no wire field yet)"
    assert str(e) == ENVELOPE["error"]["message"]
    assert len(sender.posts) == 1


def test_remote_client_leaves_a_generic_500_as_it_was() -> None:
    err = _HttpError(_Resp(500, {"error": {"message": "internal", "type": "server_error"}}))
    sender = _Sender(err)
    with pytest.raises(_HttpError) as info:
        _remote(sender).generate(prompt="hello", params={})
    assert info.value is err
