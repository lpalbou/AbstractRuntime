"""Round 4 (operator 2026-11-08, "fallback to MTP=off if MTP was selected by mistake"): six Assistant runs
failed instantly because a stored native_mtp default reached a remote openai-compatible model. Core now
degrades such a default-sourced request itself (generates without speculation and reports
`metadata.speculation.applied: false`); the runtime's part is the ledger notice on the LLM_CALL result so the
replay view can say why the call ran without acceleration. A response without the flag adds nothing; an
exception still propagates as before.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from abstractruntime.core.models import Effect, EffectType, RunState
from abstractruntime.integrations.abstractcore.effect_handlers import make_llm_call_handler
from abstractruntime.storage.artifacts import InMemoryArtifactStore


class _LLM:
    def __init__(self, metadata: Dict[str, Any]) -> None:
        self.metadata = metadata
        self.calls: List[Dict[str, Any]] = []

    def generate(
        self,
        *,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        media: Optional[List[Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.calls.append(dict(params or {}))
        return {"content": "generated", "metadata": dict(self.metadata)}


def _run() -> RunState:
    run = RunState.new(
        workflow_id="wf", entry_node="n", vars={"_runtime": {"speculation": {"mode": "native_mtp"}}},
    )
    run.current_node = "n"
    return run


def _effect() -> Effect:
    return Effect(
        type=EffectType.LLM_CALL,
        payload={"prompt": "hello", "provider": "openai-compatible", "model": "qwen/qwen3.8-27b"},
        result_key="llm",
    )


def _call(llm: _LLM):
    return make_llm_call_handler(llm=llm, artifact_store=InMemoryArtifactStore())(_run(), _effect(), None)


def test_degraded_speculation_leaves_one_notice_on_the_result() -> None:
    llm = _LLM({"speculation": {"applied": False, "reason": "no_native_adapter", "mode": "native_mtp"}})
    out = _call(llm)
    assert out.status == "completed"
    assert out.result["content"] == "generated"
    assert out.result["notices"] == [
        "speculation requested (native_mtp) but the openai-compatible provider has no native MTP adapter; "
        "generated without it (reason: no_native_adapter)"
    ]
    # No retry in the runtime: one provider call, the request as asked (Core degraded it).
    assert len(llm.calls) == 1 and llm.calls[0].get("speculation") == {"mode": "native_mtp"}


def test_a_response_without_the_flag_adds_no_notice() -> None:
    applied = _call(_LLM({"speculation": {"applied": True, "mode": "native_mtp"}}))
    assert applied.status == "completed" and "notices" not in applied.result
    silent = _call(_LLM({}))
    assert silent.status == "completed" and "notices" not in silent.result
