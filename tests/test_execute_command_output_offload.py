"""execute_command large-output artifact offload (backlog 0215).

Symmetric with read_file: when a command's full stdout exceeds the inline byte budget, it is stored
as a session attachment and the durable result carries a preview + an open_attachment handle
instead of the full blob.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from abstractruntime.core.models import Effect, EffectType, RunState
from abstractruntime.integrations.abstractcore.effect_handlers import make_tool_calls_handler
from abstractruntime.storage.artifacts import InMemoryArtifactStore


class _Tools:
    """Stub executor that returns an execute_command-shaped result with a large stdout."""

    def __init__(self, stdout: str):
        self._stdout = stdout

    def execute(self, *, tool_calls):
        results = []
        for tc in tool_calls:
            if tc.get("name") == "execute_command":
                results.append(
                    {
                        "call_id": tc.get("call_id"),
                        "name": "execute_command",
                        "success": True,
                        "output": {
                            "success": True,
                            "command": tc.get("arguments", {}).get("command"),
                            "return_code": 0,
                            "stdout": self._stdout,
                            "stderr": "",
                            "stdout_preview": self._stdout[:100],
                            "rendered": f"✅ ran\n📤 STDOUT:\n{self._stdout[:100]}... (truncated)",
                        },
                        "error": None,
                    }
                )
            else:
                results.append({"call_id": tc.get("call_id"), "name": tc.get("name"), "success": True, "output": "", "error": None})
        return {"mode": "executed", "results": results}


def _run(handler, stdout: str, env_max: Optional[str] = None, monkeypatch=None):
    if env_max is not None and monkeypatch is not None:
        monkeypatch.setenv("ABSTRACTRUNTIME_MAX_INLINE_BYTES", env_max)
    run = RunState.new(workflow_id="wf", entry_node="n", session_id="s1", vars={})
    effect = Effect(
        type=EffectType.TOOL_CALLS,
        payload={"tool_calls": [{"name": "execute_command", "arguments": {"command": "yes | head -n big"}, "call_id": "c1"}]},
    )
    outcome = handler(run, effect, None)
    assert outcome.status == "completed"
    return outcome.result["results"][0]


def test_large_command_output_is_offloaded(monkeypatch):
    store = InMemoryArtifactStore()
    handler = make_tool_calls_handler(tools=_Tools("X" * 5000), artifact_store=store)
    # Small inline budget so 5000 bytes counts as "large".
    res = _run(handler, "X" * 5000, env_max="1024", monkeypatch=monkeypatch)

    out = res["output"]
    assert isinstance(out, dict)
    # Full blob moved out of the durable result...
    assert out["stdout"] == ""
    aid = out.get("stdout_offloaded_artifact_id")
    assert isinstance(aid, str) and aid
    # ...preview + open_attachment hint preserved for the model.
    assert "open_attachment" in out["rendered"]
    assert aid in out["rendered"]
    # The full output is retrievable from the artifact store.
    art = store.load(aid)
    assert art is not None
    assert art.content == b"X" * 5000


def test_small_command_output_is_not_offloaded(monkeypatch):
    store = InMemoryArtifactStore()
    handler = make_tool_calls_handler(tools=_Tools("hi"), artifact_store=store)
    res = _run(handler, "hi", env_max="1024", monkeypatch=monkeypatch)
    out = res["output"]
    assert out["stdout"] == "hi"
    assert "stdout_offloaded_artifact_id" not in out


class _ToolsFull:
    """Executor returning an execute_command result with configurable success/stdout/stderr, plus
    a generic string-output tool."""

    def __init__(self, *, success=True, stdout="", stderr="", generic=None):
        self._s, self._out, self._err, self._generic = success, stdout, stderr, generic

    def execute(self, *, tool_calls):
        results = []
        for tc in tool_calls:
            if tc.get("name") == "execute_command":
                results.append({
                    "call_id": tc.get("call_id"), "name": "execute_command", "success": self._s,
                    "output": {"success": self._s, "return_code": 0 if self._s else 1,
                               "stdout": self._out, "stderr": self._err, "rendered": "ran"},
                    "error": None if self._s else "nonzero",
                })
            else:
                results.append({"call_id": tc.get("call_id"), "name": tc.get("name"),
                                "success": True, "output": self._generic, "error": None})
        return {"mode": "executed", "results": results}


def _run_full(tools, tool_name="execute_command", env_max="1024", monkeypatch=None):
    monkeypatch.setenv("ABSTRACTRUNTIME_MAX_INLINE_BYTES", env_max)
    monkeypatch.setenv("ABSTRACTRUNTIME_MAX_ATTACHMENT_BYTES", str(50 * 1024 * 1024))
    store = InMemoryArtifactStore()
    handler = make_tool_calls_handler(tools=tools, artifact_store=store)
    run = RunState.new(workflow_id="wf", entry_node="n", session_id="s1", vars={})
    effect = Effect(type=EffectType.TOOL_CALLS,
                    payload={"tool_calls": [{"name": tool_name, "arguments": {}, "call_id": "c1"}]})
    outcome = handler(run, effect, None)
    return outcome.result["results"][0], store


def test_failed_command_large_stdout_is_offloaded(monkeypatch):
    # A noisy FAILURE (exit 1) must still offload — the exact case that matters.
    res, store = _run_full(_ToolsFull(success=False, stdout="E" * 5000, stderr="warn"), monkeypatch=monkeypatch)
    out = res["output"]
    assert out["stdout"] == ""
    aid = out.get("stdout_offloaded_artifact_id")
    assert aid and store.load(aid).content == b"E" * 5000


def test_large_stderr_is_offloaded(monkeypatch):
    res, store = _run_full(_ToolsFull(success=True, stdout="ok", stderr="W" * 5000), monkeypatch=monkeypatch)
    out = res["output"]
    assert out["stderr"] == ""
    aid = out.get("stderr_offloaded_artifact_id")
    assert aid and store.load(aid).content == b"W" * 5000


def test_generic_tool_large_string_output_is_offloaded(monkeypatch):
    res, store = _run_full(_ToolsFull(generic="G" * 5000), tool_name="analyze_code", monkeypatch=monkeypatch)
    aid = res.get("output_offloaded_artifact_id")
    assert aid, res
    # output replaced with a handle string, not the full blob; full text retrievable.
    assert "open_attachment" in str(res["output"]) and "GGGG" not in str(res["output"])
    assert store.load(aid).content == b"G" * 5000


def test_output_over_cap_is_pushed_back_not_silently_kept(monkeypatch):
    # Cap the attachment store tiny so a modest output exceeds it -> explicit narrow-the-command notice.
    monkeypatch.setenv("ABSTRACTRUNTIME_MAX_INLINE_BYTES", "100")
    monkeypatch.setenv("ABSTRACTRUNTIME_MAX_ATTACHMENT_BYTES", "1000")
    store = InMemoryArtifactStore()
    handler = make_tool_calls_handler(tools=_ToolsFull(success=True, stdout="Z" * 5000), artifact_store=store)
    run = RunState.new(workflow_id="wf", entry_node="n", session_id="s1", vars={})
    effect = Effect(type=EffectType.TOOL_CALLS,
                    payload={"tool_calls": [{"name": "execute_command", "arguments": {}, "call_id": "c1"}]})
    out = handler(run, effect, None).result["results"][0]["output"]
    assert out["stdout"] == ""  # not kept inline
    assert "stdout_offloaded_artifact_id" not in out  # not stored (too large)
    assert "retention limit" in out["rendered"] and "narrow" in out["rendered"].lower()
