"""Endpoint-profile resolver context around tool execution (Case-1 seam).

Converged design (commons c5747-c5783, 2026-07-26): gateway-registered
endpoint:* profiles are per-principal and invisible to core's in-process
create_llm. The fix chain: gateway installs a principal-bound resolver on
the runtime's llm_client (already shipped); core consults an ambient
contextvar resolver when local resolution fails (their seam,
use_provider_endpoint_profile_resolver — context-manager-only, so a
process-global install is inexpressible); RUNTIME (pinned here) binds that
context around tool execution.

The wrap lives on MappingToolExecutor.execute — ONE site that every path
funnels through by construction: the policy path, approval-resume
(ApprovalToolExecutor.execute_approved delegates into it — the
approve-then-blind trap), and tool_invoke's pre-approved view. Parallel
read groups run under copy_context so pool threads see the same ambient
context (core's same-thread rule).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from abstractruntime.integrations.abstractcore.tool_executor import (
    ApprovalToolExecutor,
    MappingToolExecutor,
    ToolApprovalPolicy,
    attach_endpoint_profile_resolver_getter,
)

try:
    from abstractcore.providers import current_provider_endpoint_profile_resolver
    _SEAM = True
except Exception:  # pragma: no cover - older core
    _SEAM = False

pytestmark = pytest.mark.skipif(not _SEAM, reason="abstractcore seam not installed")


def _resolver(spec: str) -> Optional[Dict[str, Any]]:
    if spec == "endpoint:airelay":
        return {"provider_family": "openai_compatible", "base_url": "http://x", "id": "airelay"}
    return None


def _probe_tool() -> str:
    """Reports whether the ambient resolver context is visible (the exact
    read core's session-route path performs)."""
    r = current_provider_endpoint_profile_resolver()
    if r is None:
        return "no-resolver"
    hit = r("endpoint:airelay")
    return f"resolved:{(hit or {}).get('id')}"


_probe_tool.__name__ = "probe_route"


def _mk_executor() -> MappingToolExecutor:
    ex = MappingToolExecutor({"probe_route": _probe_tool, "read_file": lambda path: f"read:{path}"})
    ex.set_endpoint_profile_resolver_getter(lambda: _resolver)
    return ex


def test_execute_binds_the_resolver_context() -> None:
    ex = _mk_executor()
    out = ex.execute(tool_calls=[{"call_id": "1", "name": "probe_route", "arguments": {}}])
    assert out["results"][0]["output"] == "resolved:airelay"


def test_approval_resume_path_carries_the_same_context() -> None:
    """The approve-then-blind trap: execute_approved delegates into the
    wrapped execute, so an approved tool sees the resolver too."""
    approval = ApprovalToolExecutor(
        delegate=_mk_executor(),
        policy=ToolApprovalPolicy(require_approval_tools={"probe_route"}),
    )
    gated = approval.execute(tool_calls=[{"call_id": "1", "name": "probe_route", "arguments": {}}])
    assert gated["mode"] == "approval_required"
    resumed = approval.execute_approved(tool_calls=[{"call_id": "1", "name": "probe_route", "arguments": {}}])
    assert resumed["results"][0]["output"] == "resolved:airelay"


def test_no_getter_means_no_context_byte_identical() -> None:
    ex = MappingToolExecutor({"probe_route": _probe_tool})
    out = ex.execute(tool_calls=[{"call_id": "1", "name": "probe_route", "arguments": {}}])
    assert out["results"][0]["output"] == "no-resolver"


def test_getter_is_late_bound_not_frozen() -> None:
    """The host may install the resolver AFTER executor construction —
    def-time capture would freeze None forever (the frozen-default lesson)."""
    holder: Dict[str, Any] = {"resolver": None}
    ex = MappingToolExecutor({"probe_route": _probe_tool})
    ex.set_endpoint_profile_resolver_getter(lambda: holder["resolver"])

    out1 = ex.execute(tool_calls=[{"call_id": "1", "name": "probe_route", "arguments": {}}])
    assert out1["results"][0]["output"] == "no-resolver"

    holder["resolver"] = _resolver
    out2 = ex.execute(tool_calls=[{"call_id": "2", "name": "probe_route", "arguments": {}}])
    assert out2["results"][0]["output"] == "resolved:airelay"


def test_parallel_pool_threads_see_the_context() -> None:
    """Parallel-safe groups run in a ThreadPool; copy_context per submission
    keeps the ambient resolver visible (core's same-thread rule)."""
    seen: List[str] = []

    def read_file(path: str) -> str:
        r = current_provider_endpoint_profile_resolver()
        seen.append("ctx" if r is not None else "none")
        return path

    ex = MappingToolExecutor({"read_file": read_file})
    ex.set_endpoint_profile_resolver_getter(lambda: _resolver)
    out = ex.execute(
        tool_calls=[
            {"call_id": "1", "name": "read_file", "arguments": {"path": "a"}},
            {"call_id": "2", "name": "read_file", "arguments": {"path": "b"}},
            {"call_id": "3", "name": "read_file", "arguments": {"path": "c"}},
        ]
    )
    assert all(r["success"] for r in out["results"])
    assert seen == ["ctx", "ctx", "ctx"]


def test_attach_walks_delegate_chains() -> None:
    inner = MappingToolExecutor({"probe_route": _probe_tool})
    wrapper = ApprovalToolExecutor(delegate=inner, policy=ToolApprovalPolicy())
    assert attach_endpoint_profile_resolver_getter(wrapper, lambda: _resolver) is True
    out = wrapper.execute_approved(tool_calls=[{"call_id": "1", "name": "probe_route", "arguments": {}}])
    assert out["results"][0]["output"] == "resolved:airelay"

    class _NoExecutor:
        pass

    assert attach_endpoint_profile_resolver_getter(_NoExecutor(), lambda: _resolver) is False


def test_build_effect_handlers_attaches_from_the_llm_client() -> None:
    """The composition seam: build_effect_handlers threads the llm_client's
    gateway-installed resolver into the executor, late-bound."""
    from abstractruntime.integrations.abstractcore.effect_handlers import build_effect_handlers

    class _FakeLLM:
        _provider_endpoint_profile_resolver = None

        def set_provider_endpoint_profile_resolver(self, r: Any) -> None:
            self._provider_endpoint_profile_resolver = r

    llm = _FakeLLM()
    ex = MappingToolExecutor({"probe_route": _probe_tool})
    build_effect_handlers(llm=llm, tools=ex)  # attach happens here

    out1 = ex.execute(tool_calls=[{"call_id": "1", "name": "probe_route", "arguments": {}}])
    assert out1["results"][0]["output"] == "no-resolver"

    llm.set_provider_endpoint_profile_resolver(_resolver)  # gateway's later install
    out2 = ex.execute(tool_calls=[{"call_id": "2", "name": "probe_route", "arguments": {}}])
    assert out2["results"][0]["output"] == "resolved:airelay"


def test_timeout_thread_carries_the_context() -> None:
    """Adversary P1-1: a configured tool timeout runs the tool in a bare
    Thread whose context starts EMPTY on CPython 3.12 — the seam silently
    died on every lane with a timeout (gateway local mode sets 7200s).
    Pinned: the timeout runner executes under a copied context."""
    ex = MappingToolExecutor({"probe_route": _probe_tool}, timeout_s=30.0)
    ex.set_endpoint_profile_resolver_getter(lambda: _resolver)
    out = ex.execute(tool_calls=[{"call_id": "1", "name": "probe_route", "arguments": {}}])
    assert out["results"][0]["output"] == "resolved:airelay"


def test_pool_plus_timeout_carries_the_context() -> None:
    """The sharp edge: pool submission copies the context, then the nested
    timeout thread must copy it AGAIN one layer down."""
    seen: List[str] = []

    def read_file(path: str) -> str:
        r = current_provider_endpoint_profile_resolver()
        seen.append("ctx" if r is not None else "none")
        return path

    ex = MappingToolExecutor({"read_file": read_file}, timeout_s=30.0)
    ex.set_endpoint_profile_resolver_getter(lambda: _resolver)
    ex.execute(
        tool_calls=[
            {"call_id": "1", "name": "read_file", "arguments": {"path": "a"}},
            {"call_id": "2", "name": "read_file", "arguments": {"path": "b"}},
        ]
    )
    assert seen == ["ctx", "ctx"]


def test_remote_client_public_resolver_attribute_reaches_the_seam() -> None:
    """Adversary P1-2: the remote llm client carries only the PUBLIC
    resolve_provider_endpoint_profile attribute — the getter must fall back
    to it or the whole remote lane stays dark."""
    from abstractruntime.integrations.abstractcore.effect_handlers import build_effect_handlers

    class _RemoteLLM:
        # No private attribute at all — the remote client's real shape.
        resolve_provider_endpoint_profile = staticmethod(_resolver)

    ex = MappingToolExecutor({"probe_route": _probe_tool})
    build_effect_handlers(llm=_RemoteLLM(), tools=ex)
    out = ex.execute(tool_calls=[{"call_id": "1", "name": "probe_route", "arguments": {}}])
    assert out["results"][0]["output"] == "resolved:airelay"
