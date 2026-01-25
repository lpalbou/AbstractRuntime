from __future__ import annotations

from typing import Any, Dict, List, Optional

from abstractruntime.core.models import Effect, EffectType, RunState, RunStatus
from abstractruntime.integrations.abstractcore.effect_handlers import make_llm_call_handler


class _FakeLLM:
    def __init__(self, *, always_truncate: bool = False):
        self.calls: List[Dict[str, Any]] = []
        self._always_truncate = always_truncate

    def generate(
        self,
        *,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        media: Optional[List[Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        del prompt, messages, system_prompt, media, tools
        params2 = dict(params or {})
        self.calls.append(params2)

        if self._always_truncate or len(self.calls) == 1:
            return {"content": "partial", "finish_reason": "length", "metadata": {}}
        return {"content": "ok", "finish_reason": "stop", "metadata": {}}


def _run_state() -> RunState:
    return RunState(
        run_id="run-test",
        workflow_id="wf-test",
        status=RunStatus.RUNNING,
        current_node="node-llm",
        vars={"_limits": {"max_output_tokens": None, "max_input_tokens": None, "max_tokens": 65536}},
    )


def test_llm_call_retries_on_truncation_and_returns_untruncated() -> None:
    llm = _FakeLLM(always_truncate=False)
    handler = make_llm_call_handler(llm=llm, artifact_store=None)
    run = _run_state()

    effect = Effect(
        type=EffectType.LLM_CALL,
        payload={
            "prompt": "hello",
            "provider": "lmstudio",
            "model": "unit-test-model",
            "max_out_tokens": 10,
            "allow_truncation": False,
            "retry_on_truncation": True,
            "max_truncation_attempts": 3,
        },
    )

    out = handler(run, effect, None)
    assert out.status == "completed"
    assert isinstance(out.result, dict)
    assert out.result.get("content") == "ok"
    assert len(llm.calls) == 2
    assert llm.calls[0].get("max_output_tokens") == 10
    assert isinstance(llm.calls[1].get("max_output_tokens"), int)
    assert int(llm.calls[1]["max_output_tokens"]) > 10


def test_llm_call_fails_loudly_if_still_truncated() -> None:
    llm = _FakeLLM(always_truncate=True)
    handler = make_llm_call_handler(llm=llm, artifact_store=None)
    run = _run_state()

    effect = Effect(
        type=EffectType.LLM_CALL,
        payload={
            "prompt": "hello",
            "provider": "lmstudio",
            "model": "unit-test-model",
            "max_out_tokens": 10,
            "allow_truncation": False,
            "retry_on_truncation": True,
            "max_truncation_attempts": 2,
        },
    )

    out = handler(run, effect, None)
    assert out.status == "failed"
    assert isinstance(out.error, str)
    assert "truncated" in out.error.lower()
    assert len(llm.calls) == 2
