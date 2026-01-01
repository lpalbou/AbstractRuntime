"""abstractruntime.evidence.recorder

Evidence is "provenance-first": a durable record of what the system actually observed at
external boundaries (web + process execution), stored as artifacts with a small JSON index
in run state.

Design goals:
- Always-on capture for a small default set of tools (web_search/fetch_url/execute_command).
- Keep RunState.vars JSON-safe and bounded: store large payloads in ArtifactStore and keep refs.
- Make later indexing/storage upgrades possible (Elastic/vector/etc) without changing semantics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from ..core.models import RunState
from ..storage.artifacts import ArtifactStore, artifact_ref, is_artifact_ref


DEFAULT_EVIDENCE_TOOL_NAMES: tuple[str, ...] = ("web_search", "fetch_url", "execute_command")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_memory_spans(run: RunState) -> list[dict[str, Any]]:
    runtime_ns = run.vars.get("_runtime")
    if not isinstance(runtime_ns, dict):
        runtime_ns = {}
        run.vars["_runtime"] = runtime_ns
    spans = runtime_ns.get("memory_spans")
    if not isinstance(spans, list):
        spans = []
        runtime_ns["memory_spans"] = spans
    return spans


def _preview(text: str, *, limit: int = 160) -> str:
    s = str(text or "").strip()
    if len(s) <= limit:
        return s
    return s[: max(0, limit - 1)] + "…"


def _json_loads_maybe(text: str) -> Optional[Any]:
    if not isinstance(text, str):
        return None
    t = text.strip()
    if not t:
        return None
    if not (t.startswith("{") or t.startswith("[")):
        return None
    try:
        return json.loads(t)
    except Exception:
        return None


def _store_text(
    store: ArtifactStore,
    *,
    text: str,
    run_id: str,
    tags: Dict[str, str],
    content_type: str = "text/plain",
) -> Optional[Dict[str, str]]:
    s = str(text or "")
    if not s:
        return None
    meta = store.store_text(s, content_type=content_type, run_id=run_id, tags=tags)
    return artifact_ref(meta.artifact_id)


def _store_json(
    store: ArtifactStore,
    *,
    data: Any,
    run_id: str,
    tags: Dict[str, str],
) -> Optional[Dict[str, str]]:
    if data is None:
        return None
    meta = store.store_json(data, run_id=run_id, tags=tags)
    return artifact_ref(meta.artifact_id)


@dataclass(frozen=True)
class EvidenceCaptureStats:
    recorded: int = 0


class EvidenceRecorder:
    """Runtime-side recorder for always-on evidence."""

    def __init__(
        self,
        *,
        artifact_store: ArtifactStore,
        tool_names: Sequence[str] = DEFAULT_EVIDENCE_TOOL_NAMES,
    ):
        self._store = artifact_store
        self._tool_names = {str(n).strip() for n in tool_names if isinstance(n, str) and n.strip()}

    def record_tool_calls(
        self,
        *,
        run: RunState,
        node_id: str,
        tool_calls: list[Any],
        tool_results: Dict[str, Any],
    ) -> EvidenceCaptureStats:
        if not isinstance(tool_results, dict):
            return EvidenceCaptureStats(recorded=0)
        results = tool_results.get("results", [])
        if not isinstance(results, list) or not results:
            return EvidenceCaptureStats(recorded=0)
        if not isinstance(tool_calls, list):
            tool_calls = []

        spans = _ensure_memory_spans(run)
        recorded = 0

        for idx, r in enumerate(results):
            if not isinstance(r, dict):
                continue
            call = tool_calls[idx] if idx < len(tool_calls) and isinstance(tool_calls[idx], dict) else {}
            tool_name = str(r.get("name") or call.get("name") or "").strip()
            if not tool_name or tool_name not in self._tool_names:
                continue

            ok = bool(r.get("success") is True)
            call_id = str(r.get("call_id") or call.get("call_id") or "")
            error = r.get("error")
            error_text = str(error).strip() if isinstance(error, str) and error.strip() else None
            args = call.get("arguments") if isinstance(call, dict) else None
            args_dict = dict(args) if isinstance(args, dict) else {}

            output = r.get("output")
            # Tool executors vary: output may be str/dict/None.
            output_dict = dict(output) if isinstance(output, dict) else None
            output_text = str(output or "") if isinstance(output, str) else None

            created_at = utc_now_iso()
            tags: Dict[str, str] = {"kind": "evidence", "tool": tool_name}

            evidence_payload: Dict[str, Any] = {
                "tool_name": tool_name,
                "call_id": call_id,
                "success": ok,
                "error": error_text,
                "created_at": created_at,
                "run_id": run.run_id,
                "workflow_id": run.workflow_id,
                "node_id": node_id,
                "arguments": args_dict,
            }
            if run.actor_id:
                evidence_payload["actor_id"] = str(run.actor_id)
            if getattr(run, "session_id", None):
                evidence_payload["session_id"] = str(run.session_id)

            artifacts: Dict[str, Any] = {}

            if tool_name == "fetch_url":
                url = str(args_dict.get("url") or "")
                if url:
                    tags["url"] = url[:200]

                if isinstance(output_dict, dict):
                    # Store and strip large text fields from the tool output dict.
                    raw_text = output_dict.pop("raw_text", None)
                    norm_text = output_dict.pop("normalized_text", None)
                    content_type = output_dict.get("content_type")
                    content_type_str = str(content_type) if isinstance(content_type, str) else ""

                    raw_ref = None
                    if isinstance(raw_text, str) and raw_text:
                        raw_ref = _store_text(
                            self._store,
                            text=raw_text,
                            run_id=run.run_id,
                            tags={**tags, "part": "raw"},
                            content_type=content_type_str or "text/plain",
                        )
                        output_dict["raw_artifact"] = raw_ref
                        artifacts["raw"] = raw_ref

                    norm_ref = None
                    if isinstance(norm_text, str) and norm_text:
                        norm_ref = _store_text(
                            self._store,
                            text=norm_text,
                            run_id=run.run_id,
                            tags={**tags, "part": "normalized"},
                            content_type="text/plain",
                        )
                        output_dict["normalized_artifact"] = norm_ref
                        artifacts["normalized_text"] = norm_ref

                    evidence_payload["url"] = str(output_dict.get("url") or url)
                    evidence_payload["final_url"] = str(output_dict.get("final_url") or "")
                    evidence_payload["content_type"] = content_type_str
                    evidence_payload["size_bytes"] = output_dict.get("size_bytes")
                    if artifacts:
                        evidence_payload["artifacts"] = artifacts

                    # Write back the stripped/augmented dict into the tool result so run state stays small.
                    r["output"] = output_dict

            elif tool_name == "execute_command":
                cmd = str(args_dict.get("command") or "")
                if cmd:
                    tags["command"] = _preview(cmd, limit=200)

                if isinstance(output_dict, dict):
                    stdout = output_dict.pop("stdout", None)
                    stderr = output_dict.pop("stderr", None)

                    stdout_ref = None
                    if isinstance(stdout, str) and stdout:
                        stdout_ref = _store_text(
                            self._store,
                            text=stdout,
                            run_id=run.run_id,
                            tags={**tags, "part": "stdout"},
                        )
                        output_dict["stdout_artifact"] = stdout_ref
                        artifacts["stdout"] = stdout_ref

                    stderr_ref = None
                    if isinstance(stderr, str) and stderr:
                        stderr_ref = _store_text(
                            self._store,
                            text=stderr,
                            run_id=run.run_id,
                            tags={**tags, "part": "stderr"},
                        )
                        output_dict["stderr_artifact"] = stderr_ref
                        artifacts["stderr"] = stderr_ref

                    evidence_payload["command"] = str(output_dict.get("command") or cmd)
                    evidence_payload["return_code"] = output_dict.get("return_code")
                    evidence_payload["duration_s"] = output_dict.get("duration_s")
                    evidence_payload["working_directory"] = output_dict.get("working_directory")
                    evidence_payload["platform"] = output_dict.get("platform")
                    if artifacts:
                        evidence_payload["artifacts"] = artifacts

                    r["output"] = output_dict

                elif isinstance(output_text, str) and output_text:
                    out_ref = _store_text(
                        self._store,
                        text=output_text,
                        run_id=run.run_id,
                        tags={**tags, "part": "output"},
                    )
                    if out_ref is not None:
                        artifacts["output"] = out_ref
                        evidence_payload["artifacts"] = artifacts

            elif tool_name == "web_search":
                query = str(args_dict.get("query") or "")
                if query:
                    tags["query"] = _preview(query, limit=200)
                    evidence_payload["query"] = query

                if isinstance(output_text, str) and output_text:
                    parsed = _json_loads_maybe(output_text)
                    if parsed is not None:
                        out_ref = _store_json(self._store, data=parsed, run_id=run.run_id, tags={**tags, "part": "results"})
                    else:
                        out_ref = _store_text(self._store, text=output_text, run_id=run.run_id, tags={**tags, "part": "results"})
                    if out_ref is not None:
                        artifacts["results"] = out_ref
                        evidence_payload["artifacts"] = artifacts
                elif isinstance(output_dict, dict):
                    out_ref = _store_json(self._store, data=output_dict, run_id=run.run_id, tags={**tags, "part": "results"})
                    if out_ref is not None:
                        artifacts["results"] = out_ref
                        evidence_payload["artifacts"] = artifacts

            # Store the evidence record itself (small JSON with artifact refs).
            record_ref = _store_json(self._store, data=evidence_payload, run_id=run.run_id, tags=tags)
            if not (isinstance(record_ref, dict) and is_artifact_ref(record_ref)):
                continue
            evidence_id = record_ref["$artifact"]

            # Append to span-like index for fast listing.
            span_record: Dict[str, Any] = {
                "kind": "evidence",
                "artifact_id": evidence_id,
                "created_at": created_at,
                "from_timestamp": created_at,
                "to_timestamp": created_at,
                "message_count": 0,
                "tool_name": tool_name,
                "call_id": call_id,
                "success": ok,
            }
            if tool_name == "fetch_url":
                span_record["url"] = evidence_payload.get("url") or str(args_dict.get("url") or "")
            elif tool_name == "web_search":
                span_record["query"] = str(args_dict.get("query") or "")
            elif tool_name == "execute_command":
                span_record["command_preview"] = _preview(str(args_dict.get("command") or ""))

            # Attach span id back to the tool result entry for easy linking in traces/UIs.
            meta = r.get("meta")
            if not isinstance(meta, dict):
                meta = {}
                r["meta"] = meta
            meta["evidence_id"] = evidence_id

            spans.append(span_record)
            recorded += 1

        return EvidenceCaptureStats(recorded=recorded)


