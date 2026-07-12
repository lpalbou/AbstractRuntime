"""Per-agent agora identity via alias indirection (hooks plan H8).

Pins the fleet identity contract: N resident runs in ONE process post as N
DISTINCT agora agents. The run carries only a NON-secret alias
(`_runtime.agora_agent`); the key lives in host env (`AGORA_API_KEY__<ALIAS>`)
and is resolved at call time. The alias is force-stamped by the tool-calls
handler (trust boundary — model-supplied values are always overridden), and a
configured alias with a missing key fails loudly rather than falling back to
the global key (posting as the WRONG agent is the bug this exists to fix).
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Tuple

import pytest

from abstractruntime import Effect, EffectType, InMemoryLedgerStore, InMemoryRunStore, RunStatus, StepPlan, WorkflowSpec
from abstractruntime.core.models import RunState
from abstractruntime.core.runtime import Runtime
from abstractruntime.integrations.abstractcore.effect_handlers import make_tool_calls_handler
from abstractruntime.integrations.abstractcore.tool_executor import MappingToolExecutor


class _HubStub(BaseHTTPRequestHandler):
    requests: List[Dict[str, Any]] = []
    responses: Dict[Tuple[str, str], Any] = {}

    def _record_and_reply(self, method: str) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        raw_body = self.rfile.read(length).decode("utf-8") if length else ""
        path, _, query = self.path.partition("?")
        type(self).requests.append(
            {
                "method": method,
                "path": path,
                "query": query,
                "auth": self.headers.get("Authorization"),
                "body": json.loads(raw_body) if raw_body else None,
            }
        )
        payload = type(self).responses.get((method, path), {})
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        self._record_and_reply("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._record_and_reply("POST")

    def log_message(self, *args: Any) -> None:
        del args


@pytest.fixture()
def hub(monkeypatch: pytest.MonkeyPatch):
    _HubStub.requests = []
    _HubStub.responses = {}
    server = HTTPServer(("127.0.0.1", 0), _HubStub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("AGORA_URL", f"http://127.0.0.1:{server.server_address[1]}")
    monkeypatch.setenv("AGORA_API_KEY", "global-key")
    monkeypatch.setenv("AGORA_API_KEY__RESIDENT_A", "secret-key-a")
    monkeypatch.setenv("AGORA_API_KEY__RESIDENT_B", "secret-key-b")
    try:
        yield _HubStub
    finally:
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------------------
# Alias -> key resolution (toolset layer)
# ---------------------------------------------------------------------------

def test_two_aliases_authenticate_as_two_agents(hub) -> None:
    from abstractruntime.integrations.abstractcore.agora_tools import agora_whoami

    agora_whoami(_agora_agent="resident-a")
    agora_whoami(_agora_agent="resident-b")
    agora_whoami()  # no alias -> the process-global identity, unchanged behavior

    auths = [r["auth"] for r in hub.requests]
    assert auths == ["Bearer secret-key-a", "Bearer secret-key-b", "Bearer global-key"]


def test_alias_env_suffix_is_injective_over_the_legal_domain() -> None:
    """Aliases are lowercase slugs; the env-suffix fold is 1:1 on that domain.
    Anything the fold would CONFLATE (underscores, case, dots, spaces) is
    rejected loudly — two differently-named residents can never silently read
    the same key (adversary P1)."""
    from abstractruntime.integrations.abstractcore.agora_tools import _alias_env_suffix

    assert _alias_env_suffix("resident-a") == "RESIDENT_A"
    assert _alias_env_suffix("r2-d2") == "R2_D2"
    for bad in ("resident_a", "Resident-A", "resident.a", "resident a", "-lead", "a--b", " resident-a"):
        with pytest.raises(ValueError):
            _alias_env_suffix(bad)


def test_blank_configured_alias_fails_loud_not_global(hub) -> None:
    """`_runtime.agora_agent` PRESENT but blank = invalid config — it must
    never silently post as the global agent (adversary P2)."""
    handler = make_tool_calls_handler(tools=_agora_executor())
    run = RunState.new(
        workflow_id="wf", entry_node="n", session_id="s1", vars={"_runtime": {"agora_agent": "  "}}
    )
    outcome = handler(run, _whoami_effect(), None)
    assert str(outcome.status) == "completed"
    result = outcome.result["results"][0]
    assert result["success"] is False
    assert "blank" in str(result["error"])
    assert hub.requests == []  # nothing left the process under any identity


def test_missing_alias_key_fails_loud_and_never_falls_back(hub) -> None:
    """A configured alias with no key must NOT silently post as the global agent."""
    from abstractruntime.integrations.abstractcore.agora_tools import agora_whoami

    with pytest.raises(RuntimeError) as e:
        agora_whoami(_agora_agent="ghost")
    assert "AGORA_API_KEY__GHOST" in str(e.value)
    assert hub.requests == []  # refused before any HTTP left the process


def test_per_alias_url_override(hub, monkeypatch) -> None:
    """AGORA_URL__<ALIAS> routes one agent to its own hub; others keep the shared URL."""
    from abstractruntime.integrations.abstractcore.agora_tools import agora_base_url

    monkeypatch.setenv("AGORA_URL__RESIDENT_B", "http://10.0.0.9:9999/")
    assert agora_base_url("resident-b") == "http://10.0.0.9:9999"
    assert agora_base_url("resident-a") == agora_base_url()


# ---------------------------------------------------------------------------
# Handler stamping (trust boundary — mirrors the shell _registry_namespace pins)
# ---------------------------------------------------------------------------

def _agora_executor() -> MappingToolExecutor:
    from abstractruntime.integrations.abstractcore.agora_tools import AGORA_TOOLS

    return MappingToolExecutor.from_tools(list(AGORA_TOOLS))


def _whoami_effect(extra_args: Dict[str, Any] | None = None) -> Effect:
    return Effect(
        type=EffectType.TOOL_CALLS,
        payload={"tool_calls": [{"name": "agora_whoami", "arguments": dict(extra_args or {}), "call_id": "c1"}]},
    )


def test_model_supplied_alias_is_overwritten(hub) -> None:
    """A tool call claiming another agent's alias is re-stamped from run vars."""
    handler = make_tool_calls_handler(tools=_agora_executor())
    run = RunState.new(
        workflow_id="wf", entry_node="n", session_id="s1", vars={"_runtime": {"agora_agent": "resident-a"}}
    )

    outcome = handler(run, _whoami_effect({"_agora_agent": "resident-b"}), None)
    assert str(outcome.status) == "completed"
    assert outcome.result["results"][0]["success"] is True
    assert [r["auth"] for r in hub.requests] == ["Bearer secret-key-a"]


def test_model_supplied_alias_is_stripped_when_run_has_none(hub) -> None:
    """No alias on the run -> a model-supplied alias is removed, not honored."""
    handler = make_tool_calls_handler(tools=_agora_executor())
    run = RunState.new(workflow_id="wf", entry_node="n", session_id="s1", vars={})

    outcome = handler(run, _whoami_effect({"_agora_agent": "resident-b"}), None)
    assert str(outcome.status) == "completed"
    assert [r["auth"] for r in hub.requests] == ["Bearer global-key"]


def test_key_never_rests_in_run_vars_or_ledger(hub) -> None:
    """End-to-end through a real Runtime: the SECRET key appears nowhere durable —
    run vars and every ledger record carry at most the non-secret alias."""
    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.TOOL_CALLS: make_tool_calls_handler(tools=_agora_executor())},
    )

    def tools_node(run, ctx) -> StepPlan:
        del ctx
        return StepPlan(node_id="tools", effect=_whoami_effect(), next_node="done")

    def done_node(run, ctx) -> StepPlan:
        del ctx
        return StepPlan(node_id="done", complete_output={"ok": True})

    workflow = WorkflowSpec(
        workflow_id="agora_alias_test", entry_node="tools", nodes={"tools": tools_node, "done": done_node}
    )
    run_id = runtime.start(workflow=workflow, vars={"_runtime": {"agora_agent": "resident-a"}})
    state = runtime.tick(workflow=workflow, run_id=run_id, max_steps=5)
    assert state.status == RunStatus.COMPLETED
    assert [r["auth"] for r in hub.requests] == ["Bearer secret-key-a"]

    dumped_vars = json.dumps(state.vars, default=str)
    assert "secret-key-a" not in dumped_vars
    records = runtime._ledger_store.list(run_id)
    assert records  # the tool step was ledgered
    for rec in records:
        assert "secret-key-a" not in json.dumps(rec, default=str)


def test_approval_resume_executes_with_the_stamped_alias(hub) -> None:
    """Mirror of the shell namespace-resume pin (adversary P2 coverage gap):
    when an operator policy forces approval on an agora tool, the stamped
    alias rides the stored wait and the approved-resume authenticates as the
    run's identity — never re-planned, never model-controllable."""
    from abstractruntime.integrations.abstractcore.tool_executor import (
        ApprovalToolExecutor,
        ToolApprovalPolicy,
    )

    tools = ApprovalToolExecutor(
        delegate=_agora_executor(),
        policy=ToolApprovalPolicy(require_approval_tools={"agora_whoami"}),
    )
    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.TOOL_CALLS: make_tool_calls_handler(tools=tools)},
    )
    runtime.set_tool_executor_for_resume(tools)

    def tools_node(run, ctx) -> StepPlan:
        del ctx
        return StepPlan(
            node_id="tools",
            effect=Effect(
                type=EffectType.TOOL_CALLS,
                payload={"tool_calls": [{"name": "agora_whoami", "arguments": {"_agora_agent": "resident-b"}, "call_id": "c1"}]},
                result_key="tool_results",
            ),
            next_node="done",
        )

    def done_node(run, ctx) -> StepPlan:
        del ctx
        return StepPlan(node_id="done", complete_output={"ok": True})

    workflow = WorkflowSpec(
        workflow_id="agora_approval_test", entry_node="tools", nodes={"tools": tools_node, "done": done_node}
    )
    run_id = runtime.start(workflow=workflow, vars={"_runtime": {"agora_agent": "resident-a"}})
    state = runtime.tick(workflow=workflow, run_id=run_id, max_steps=5)
    assert state.status == RunStatus.WAITING
    assert state.waiting is not None and state.waiting.details.get("mode") == "approval_required"
    stored = state.waiting.details.get("tool_calls")
    assert stored and stored[0]["arguments"]["_agora_agent"] == "resident-a"  # spoof overridden at plan time

    resumed = runtime.resume(
        workflow=workflow, run_id=run_id, wait_key=state.waiting.wait_key,
        payload={"approved": True}, max_steps=5,
    )
    assert resumed.status == RunStatus.COMPLETED
    assert [r["auth"] for r in hub.requests] == ["Bearer secret-key-a"]


# ---------------------------------------------------------------------------
# Exposure pins
# ---------------------------------------------------------------------------

def test_alias_arg_hidden_from_every_tool_schema() -> None:
    from abstractruntime.integrations.abstractcore.agora_tools import AGORA_TOOLS

    for t in AGORA_TOOLS:
        assert "_agora_agent" not in t._tool_definition.parameters, t._tool_definition.name


def test_alias_keys_alone_enable_the_toolset(monkeypatch) -> None:
    """A fleet host may configure ONLY per-alias keys (no global AGORA_API_KEY)."""
    monkeypatch.delenv("ABSTRACT_ENABLE_AGORA_TOOLS", raising=False)
    monkeypatch.delenv("AGORA_API_KEY", raising=False)
    monkeypatch.setenv("AGORA_API_KEY__RESIDENT_A", "k")
    from abstractruntime.integrations.abstractcore.default_tools import agora_tools_enabled

    assert agora_tools_enabled() is True
    monkeypatch.delenv("AGORA_API_KEY__RESIDENT_A", raising=False)
    assert agora_tools_enabled() is False
