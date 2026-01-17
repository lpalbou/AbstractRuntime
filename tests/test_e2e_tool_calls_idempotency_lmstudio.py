from __future__ import annotations

import os
from pathlib import Path

import pytest


def _lmstudio_available(*, base_url: str) -> bool:
    try:
        import httpx

        url = base_url.rstrip("/") + "/models"
        resp = httpx.get(url, timeout=2.0)
        return resp.status_code == 200
    except Exception:
        return False


@pytest.mark.e2e
def test_tool_calls_idempotency_keys_ignore_model_call_id_lmstudio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Level C: LMStudio tool-calling + runtime restart-safe TOOL_CALLS dedupe.

    Enable with:
      ABSTRACT_E2E_LMSTUDIO=1
    """
    if os.environ.get("ABSTRACT_E2E_LMSTUDIO") != "1":
        pytest.skip("Set ABSTRACT_E2E_LMSTUDIO=1 to run this test.")

    base_url = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
    model = os.environ.get("LMSTUDIO_MODEL", "qwen/qwen3-next-80b")
    if not _lmstudio_available(base_url=base_url):
        pytest.skip(f"LMStudio not reachable at {base_url!r}")

    # Ensure AbstractCore picks up the intended endpoint for the local provider.
    monkeypatch.setenv("LMSTUDIO_BASE_URL", base_url)

    from abstractcore.tools import tool
    from abstractruntime import Effect, EffectType, RunStatus, StepPlan, WorkflowSpec
    from abstractruntime.integrations.abstractcore import MappingToolExecutor, create_local_runtime
    from abstractruntime.storage.json_files import JsonFileRunStore, JsonlLedgerStore

    executed = {"count": 0}

    @tool(name="echo")
    def echo(*, text: str) -> dict:
        executed["count"] += 1
        return {"text": text}

    tool_specs = [echo._tool_definition.to_dict()]
    tools = MappingToolExecutor.from_tools([echo])

    base = tmp_path / "stores"
    run_store = JsonFileRunStore(base)
    ledger_store = JsonlLedgerStore(base)

    rt1 = create_local_runtime(
        provider="lmstudio",
        model=model,
        llm_kwargs={"base_url": base_url},
        run_store=run_store,
        ledger_store=ledger_store,
        tool_executor=tools,
    )

    prompt = (
        "Call the tool `echo` exactly once with arguments {\"text\":\"OK\"}.\n"
        "Do not write any other text."
    )

    def llm_node(run, ctx) -> StepPlan:
        del ctx
        return StepPlan(
            node_id="LLM",
            effect=Effect(
                type=EffectType.LLM_CALL,
                payload={
                    "prompt": prompt,
                    "tools": tool_specs,
                    "provider": "lmstudio",
                    "model": model,
                    "params": {"temperature": 0.0},
                },
                result_key="llm_response",
            ),
            next_node="TOOLS",
        )

    def tools_node(run, ctx) -> StepPlan:
        del ctx
        resp = run.vars.get("llm_response")
        tool_calls = resp.get("tool_calls") if isinstance(resp, dict) else None
        if not isinstance(tool_calls, list):
            tool_calls = []
        return StepPlan(
            node_id="TOOLS",
            effect=Effect(
                type=EffectType.TOOL_CALLS,
                payload={"tool_calls": tool_calls, "allowed_tools": ["echo"]},
                result_key="tool_results",
            ),
            next_node="DONE",
        )

    def done_node(run, ctx) -> StepPlan:
        del ctx
        return StepPlan(node_id="DONE", complete_output={"tool_results": run.vars.get("tool_results")})

    wf = WorkflowSpec(workflow_id="e2e_tool_call_idempotency_lmstudio", entry_node="LLM", nodes={"LLM": llm_node, "TOOLS": tools_node, "DONE": done_node})

    run_id = rt1.start(workflow=wf)
    state1 = rt1.tick(workflow=wf, run_id=run_id, max_steps=12)
    assert state1.status == RunStatus.COMPLETED
    assert executed["count"] == 1

    # Simulate a restart where the model call-id differs, but the tool call semantic is the same.
    run = run_store.load(run_id)
    run.status = RunStatus.RUNNING
    run.current_node = "TOOLS"
    run.output = None
    run.vars.pop("tool_results", None)
    resp = run.vars.get("llm_response")
    if isinstance(resp, dict) and isinstance(resp.get("tool_calls"), list) and resp["tool_calls"]:
        tc0 = resp["tool_calls"][0]
        if isinstance(tc0, dict):
            tc0["call_id"] = "call_drifted"
    run_store.save(run)

    rt2 = create_local_runtime(
        provider="lmstudio",
        model=model,
        llm_kwargs={"base_url": base_url},
        run_store=JsonFileRunStore(base),
        ledger_store=JsonlLedgerStore(base),
        tool_executor=tools,
    )
    state2 = rt2.tick(workflow=wf, run_id=run_id, max_steps=6)
    assert state2.status == RunStatus.COMPLETED
    assert executed["count"] == 1  # tool was not re-executed

