"""The recorder must capture evidence from fetch_url's CURRENT envelope shape.

`tests/test_evidence_recorder.py` hand-builds a tool result with `raw_text` and
`normalized_text` inline. That was fetch_url's shape until it adopted a
single-canonical-payload policy: above a size cap it now returns those two keys
as None plus a `*_withheld` descriptor, and ships only `content`. The old test
stays green against its hand-built dict while the real system records nothing —
so it cannot see this regression. These tests use envelopes in the shape the
tool actually returns today.
"""
from __future__ import annotations

from typing import Any, Dict

from abstractruntime import RunState
from abstractruntime.evidence import EvidenceRecorder
from abstractruntime.storage.artifacts import InMemoryArtifactStore, get_artifact_id


def _post_policy_envelope(content: str) -> Dict[str, Any]:
    """A fetch_url result as the tool returns it once the payload policy fires."""
    return {
        "success": True,
        "url": "https://example.com/article",
        "final_url": "https://example.com/article",
        "content_type": "text/html; charset=utf-8",
        "detected_as": "html",
        "size_bytes": 480_000,
        "title": "An Article",
        "content": content,
        "content_chars": len(content),
        # withheld by policy — the keys survive, the payload does not
        "raw_text": None,
        "normalized_text": None,
        "raw_text_withheld": {
            "chars": 479_000,
            "bytes": 480_000,
            "sha256": "0" * 64,
            "content_type": "text/html; charset=utf-8",
            "reason": "raw source is evidence, not payload; over the inline cap",
        },
        "normalized_text_withheld": {
            "chars": 60_000,
            "bytes": 60_000,
            "sha256": "1" * 64,
            "content_type": "text/plain",
            "reason": "redundant with content",
        },
        "rendered": "compact header",
    }


def _record(envelope: Dict[str, Any]) -> tuple[InMemoryArtifactStore, Dict[str, Any], Dict[str, Any]]:
    store = InMemoryArtifactStore()
    run = RunState.new(workflow_id="wf", entry_node="n1", vars={})
    tool_calls = [{"name": "fetch_url", "arguments": {"url": envelope["url"]}, "call_id": "c1"}]
    tool_results = {
        "mode": "executed",
        "results": [
            {"call_id": "c1", "name": "fetch_url", "success": True, "output": envelope, "error": None}
        ],
    }
    recorder = EvidenceRecorder(artifact_store=store)
    stats = recorder.record_tool_calls(
        run=run, node_id="node-x", tool_calls=tool_calls, tool_results=tool_results
    )
    assert stats.recorded == 1
    result = tool_results["results"][0]
    out = result["output"]
    payload = store.load_json(result["meta"]["evidence_id"])
    return store, out, payload


def test_content_is_stored_as_an_artifact_when_raw_text_is_withheld() -> None:
    """The regression: a large fetch used to record NOTHING under the new envelope."""
    content = "# Heading\n\n" + ("Real extracted prose. " * 4_000)
    store, out, payload = _record(_post_policy_envelope(content))

    artifacts = payload.get("artifacts")
    assert isinstance(artifacts, dict) and artifacts, (
        "no artifact recorded for a successful fetch — the evidence chain is broken"
    )
    assert "content" in artifacts, f"content not captured; got {sorted(artifacts)}"

    ref = out.get("content_artifact")
    assert isinstance(ref, dict)
    assert store.load_text(get_artifact_id(ref)) == content


def test_withheld_descriptors_reach_the_evidence_record() -> None:
    """An auditor must be able to tell 'withheld by policy' from 'never fetched'."""
    _, _, payload = _record(_post_policy_envelope("x" * 3_000))

    withheld = payload.get("withheld")
    assert isinstance(withheld, dict), "the tool's own account of what it dropped was discarded"
    assert set(withheld) == {"raw_text_withheld", "normalized_text_withheld"}
    assert withheld["raw_text_withheld"]["chars"] == 479_000
    assert withheld["raw_text_withheld"]["sha256"] == "0" * 64


def test_old_envelope_shape_still_records_raw_and_normalized() -> None:
    """Backward compatibility: a small fetch still inlines both, and both are stored."""
    envelope = {
        "url": "https://example.com",
        "final_url": "https://example.com",
        "content_type": "text/html",
        "size_bytes": 12,
        "content": "hi",
        "raw_text": "<html>hi</html>",
        "normalized_text": "hi",
        "rendered": "rendered",
    }
    store, out, payload = _record(envelope)

    artifacts = payload["artifacts"]
    assert {"raw", "normalized_text", "content"} <= set(artifacts)
    assert store.load_text(get_artifact_id(out["raw_artifact"])) == "<html>hi</html>"
    assert "withheld" not in payload


def test_a_failed_fetch_still_records_why_it_failed() -> None:
    """A failure that stores nothing makes "blocked" and "network down" identical."""
    envelope = {
        "success": False,
        "error": "No readable content extracted (empty_content)",
        "error_class": "empty_content",
        "url": "https://example.com/blocked",
        "final_url": "https://example.com/blocked",
        "status_code": 203,
        "content_type": "text/html; charset=utf-8",
        "rendered_with_browser": False,
        "render_note": "not escalated: HTTP 203 with a stub body is a request-level block",
        "rendered": "no readable content",
    }
    _, _, payload = _record(envelope)
    assert payload.get("error_class") == "empty_content"
    assert payload.get("status_code") == 203


def test_a_rendered_dom_is_stored_separately_and_labelled_html() -> None:
    """The browser-built document is not the bytes the server sent."""
    envelope = _post_policy_envelope("x" * 3_000)
    envelope["rendered_dom"] = "<html><body>post-javascript DOM</body></html>"
    envelope["rendered_with_browser"] = True
    envelope["render_note"] = "content extracted from the rendered DOM after JavaScript ran"
    store, out, payload = _record(envelope)

    assert "rendered_dom" in payload["artifacts"], "the rendered DOM was not recorded"
    ref = out["rendered_dom_artifact"]
    assert store.load_text(get_artifact_id(ref)) == envelope["rendered_dom"]
    assert payload.get("rendered_with_browser") is True
    assert "JavaScript" in str(payload.get("render_note") or "")
