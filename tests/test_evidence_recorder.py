from __future__ import annotations

from abstractruntime import RunState
from abstractruntime.evidence import EvidenceRecorder
from abstractruntime.storage.artifacts import InMemoryArtifactStore, get_artifact_id


def test_evidence_recorder_fetch_url_stores_raw_and_normalized_as_artifacts() -> None:
    store = InMemoryArtifactStore()
    run = RunState.new(workflow_id="wf", entry_node="n1", vars={})

    tool_calls = [{"name": "fetch_url", "arguments": {"url": "https://example.com"}, "call_id": "c1"}]
    tool_results = {
        "mode": "executed",
        "results": [
            {
                "call_id": "c1",
                "name": "fetch_url",
                "success": True,
                "output": {
                    "url": "https://example.com",
                    "final_url": "https://example.com",
                    "content_type": "text/html",
                    "size_bytes": 12,
                    "raw_text": "<html>hi</html>",
                    "normalized_text": "hi",
                    "rendered": "rendered",
                },
                "error": None,
            }
        ],
    }

    recorder = EvidenceRecorder(artifact_store=store)
    stats = recorder.record_tool_calls(run=run, node_id="node-x", tool_calls=tool_calls, tool_results=tool_results)
    assert stats.recorded == 1

    out = tool_results["results"][0]["output"]
    assert isinstance(out, dict)
    assert "raw_text" not in out
    assert "normalized_text" not in out
    assert isinstance(out.get("raw_artifact"), dict)
    assert isinstance(out.get("normalized_artifact"), dict)

    raw_id = get_artifact_id(out["raw_artifact"])
    norm_id = get_artifact_id(out["normalized_artifact"])
    assert store.load_text(raw_id) == "<html>hi</html>"
    assert store.load_text(norm_id) == "hi"

    spans = run.vars.get("_runtime", {}).get("memory_spans", [])
    assert isinstance(spans, list)
    assert any(s.get("kind") == "evidence" and s.get("tool_name") == "fetch_url" for s in spans if isinstance(s, dict))

    meta = tool_results["results"][0].get("meta")
    assert isinstance(meta, dict)
    evidence_id = meta.get("evidence_id")
    assert isinstance(evidence_id, str) and evidence_id
    payload = store.load_json(evidence_id)
    assert isinstance(payload, dict)
    assert payload.get("tool_name") == "fetch_url"
    assert isinstance(payload.get("artifacts"), dict)


def test_evidence_recorder_execute_command_stores_stdout_stderr_as_artifacts() -> None:
    store = InMemoryArtifactStore()
    run = RunState.new(workflow_id="wf", entry_node="n1", vars={})

    tool_calls = [{"name": "execute_command", "arguments": {"command": "echo hi"}, "call_id": "c1"}]
    tool_results = {
        "mode": "executed",
        "results": [
            {
                "call_id": "c1",
                "name": "execute_command",
                "success": False,
                "output": {
                    "success": False,
                    "error": "Command completed with non-zero exit code: 1",
                    "command": "echo hi",
                    "return_code": 1,
                    "stdout": "out",
                    "stderr": "err",
                    "rendered": "rendered",
                },
                "error": "Command completed with non-zero exit code: 1",
            }
        ],
    }

    recorder = EvidenceRecorder(artifact_store=store)
    stats = recorder.record_tool_calls(run=run, node_id="node-x", tool_calls=tool_calls, tool_results=tool_results)
    assert stats.recorded == 1

    out = tool_results["results"][0]["output"]
    assert isinstance(out, dict)
    assert "stdout" not in out
    assert "stderr" not in out
    assert isinstance(out.get("stdout_artifact"), dict)
    assert isinstance(out.get("stderr_artifact"), dict)
    assert store.load_text(get_artifact_id(out["stdout_artifact"])) == "out"
    assert store.load_text(get_artifact_id(out["stderr_artifact"])) == "err"


def test_evidence_recorder_web_search_stores_results_as_artifact() -> None:
    store = InMemoryArtifactStore()
    run = RunState.new(workflow_id="wf", entry_node="n1", vars={})

    tool_calls = [{"name": "web_search", "arguments": {"query": "q"}, "call_id": "c1"}]
    tool_results = {
        "mode": "executed",
        "results": [
            {
                "call_id": "c1",
                "name": "web_search",
                "success": True,
                "output": '{"engine":"duckduckgo","query":"q","results":[{"title":"t","url":"u"}]}',
                "error": None,
            }
        ],
    }

    recorder = EvidenceRecorder(artifact_store=store)
    stats = recorder.record_tool_calls(run=run, node_id="node-x", tool_calls=tool_calls, tool_results=tool_results)
    assert stats.recorded == 1

    meta = tool_results["results"][0].get("meta")
    assert isinstance(meta, dict)
    evidence_id = meta.get("evidence_id")
    assert isinstance(evidence_id, str) and evidence_id
    payload = store.load_json(evidence_id)
    assert isinstance(payload, dict)
    assert payload.get("tool_name") == "web_search"
    artifacts = payload.get("artifacts")
    assert isinstance(artifacts, dict)
    assert isinstance(artifacts.get("results"), dict)

