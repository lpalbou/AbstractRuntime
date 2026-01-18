"""Session attachment registry + on-demand open tool helpers.

This module implements two framework primitives:
- a session-scoped attachment index (metadata-only, LLM-visible via injection)
- a runtime-owned `open_attachment` tool (bounded artifact reads)

These are intentionally integration-scoped (AbstractCore) and are executed inside
the runtime's effect handlers (not via a host ToolExecutor).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ...storage.artifacts import ArtifactStore

_DEFAULT_SESSION_MEMORY_RUN_PREFIX = "session_memory_"
_SAFE_RUN_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")

_TOOL_PREFIX_RE = re.compile(r"^\[(?P<name>[^\]]+)\]:\s*(?P<body>.*)$", re.DOTALL)
_READ_FILE_HEADER_RE = re.compile(r"^File:\s*(?P<path>.+?)\s*\((?P<count>\d+)\s+lines\)\s*$")
_OPEN_ATTACHMENT_HEADER_RE = re.compile(
    r"^Attachment:\s*(?P<handle>.+?)\s*\(id=(?P<artifact_id>[a-zA-Z0-9_-]+)"
    r"(?:,\s*sha=(?P<sha256>[0-9a-fA-F]{8,64}))?"
    r"(?:,\s*lines\s+(?P<start_line>\d+)-(?P<end_line>\d+))?"
    r".*\)\s*$"
)
_LINE_NUMBER_RE = re.compile(r"^\s*(?P<line>\d+):\s")


def session_memory_owner_run_id(session_id: str) -> str:
    """Return the stable session memory owner run id for a session id.

    This mirrors gateway/runtime behavior (`session_memory_<sid>` with a hash fallback)
    so durability works across restarts and across services.
    """
    sid = str(session_id or "").strip()
    if not sid:
        raise ValueError("session_id is required")
    if _SAFE_RUN_ID_PATTERN.match(sid):
        rid = f"{_DEFAULT_SESSION_MEMORY_RUN_PREFIX}{sid}"
        if _SAFE_RUN_ID_PATTERN.match(rid):
            return rid
    digest = hashlib.sha256(sid.encode("utf-8")).hexdigest()[:32]
    return f"{_DEFAULT_SESSION_MEMORY_RUN_PREFIX}sha_{digest}"


def _normalize_handle(raw: Any) -> str:
    s = str(raw or "").strip()
    if not s:
        return ""
    if s.startswith("@"):
        s = s[1:].strip()
    if s.startswith("./"):
        s = s[2:]
    return s


def _safe_tag_subset(tags: Dict[str, str], *, limit: int = 8) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k in sorted(tags.keys()):
        if len(out) >= limit:
            break
        v = tags.get(k)
        if not isinstance(k, str) or not k.strip():
            continue
        if not isinstance(v, str) or not v.strip():
            continue
        if k in {"session_id"}:
            continue
        out[k] = v
    return out


def list_session_attachments(
    *,
    artifact_store: ArtifactStore,
    session_id: str,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """Return the session attachment index (metadata-only, JSON-safe)."""
    rid = session_memory_owner_run_id(session_id)
    metas = artifact_store.list_by_run(rid)
    items = [m for m in metas if isinstance(getattr(m, "tags", None), dict) and (m.tags or {}).get("kind") == "attachment"]
    items.sort(key=lambda m: str(getattr(m, "created_at", "") or ""), reverse=True)

    out: list[Dict[str, Any]] = []
    for m in items[: max(0, int(limit))]:
        tags = dict(getattr(m, "tags", {}) or {})
        handle = _normalize_handle(tags.get("path") or tags.get("source_path") or tags.get("filename") or "")
        filename = str(tags.get("filename") or "").strip() or (handle.rsplit("/", 1)[-1] if handle else "")
        sha256 = str(tags.get("sha256") or "").strip().lower() or None
        if sha256 and not re.fullmatch(r"[0-9a-f]{8,64}", sha256):
            sha256 = None

        out.append(
            {
                "handle": handle,
                "artifact_id": str(getattr(m, "artifact_id", "") or ""),
                "filename": filename,
                "sha256": sha256,
                "content_type": str(getattr(m, "content_type", "") or ""),
                "size_bytes": int(getattr(m, "size_bytes", 0) or 0),
                "created_at": str(getattr(m, "created_at", "") or ""),
                "tags": _safe_tag_subset(tags),
            }
        )
    return out


def render_session_attachments_system_message(
    entries: Iterable[Dict[str, Any]],
    *,
    max_entries: int = 20,
    max_chars: int = 4000,
) -> str:
    """Render a bounded system message suitable for injection into LLM messages."""
    max_e = max(0, int(max_entries))
    max_c = max(0, int(max_chars))
    if max_e <= 0 or max_c <= 0:
        return ""

    lines: list[str] = ["Session attachments (most recent first):"]
    used = len(lines[0]) + 1

    for i, e in enumerate(list(entries)[:max_e]):
        if not isinstance(e, dict):
            continue
        handle = _normalize_handle(e.get("handle") or e.get("source_path") or e.get("filename") or "")
        if not handle:
            handle = str(e.get("filename") or "").strip() or "attachment"
        handle_disp = f"@{handle}"
        artifact_id = str(e.get("artifact_id") or "").strip()
        sha256 = str(e.get("sha256") or "").strip()
        ct = str(e.get("content_type") or "").strip()
        size = e.get("size_bytes")
        created_at = str(e.get("created_at") or "").strip()

        bits: list[str] = []
        if artifact_id:
            bits.append(f"id={artifact_id}")
        if sha256:
            bits.append(f"sha={sha256[:8]}…")
        if ct:
            bits.append(ct)
        if isinstance(size, int) and size > 0:
            bits.append(f"{size:,} bytes")
        if created_at:
            bits.append(f"added {created_at}")
        meta = ", ".join(bits) if bits else ""
        line = f"- {handle_disp}" + (f" ({meta})" if meta else "")

        if used + len(line) + 1 > max_c:
            # Always include an explicit truncation marker if we had at least one entry.
            if i > 0 and used + 18 <= max_c:
                lines.append("- … (truncated)")
            break

        lines.append(line)
        used += len(line) + 1

    rendered = "\n".join(lines)
    return rendered[:max_c]


@dataclass(frozen=True)
class ParsedToolMessage:
    tool_name: str
    body: str


def parse_tool_message(text: str) -> Optional[ParsedToolMessage]:
    m = _TOOL_PREFIX_RE.match(str(text or ""))
    if not m:
        return None
    name = str(m.group("name") or "").strip()
    body = str(m.group("body") or "")
    if not name:
        return None
    return ParsedToolMessage(tool_name=name, body=body)


def _parse_read_file_identity(body: str) -> Optional[Tuple[str, str, int, int]]:
    """Return (path, sha256_of_body, start_line, end_line) when parseable."""
    raw = str(body or "")
    if not raw.strip():
        return None
    lines = raw.splitlines()
    if not lines:
        return None
    m = _READ_FILE_HEADER_RE.match(lines[0].strip())
    if not m:
        return None
    path = _normalize_handle(m.group("path"))
    if not path:
        return None

    start_line = -1
    end_line = -1
    for ln in lines[1:]:
        mm = _LINE_NUMBER_RE.match(ln)
        if not mm:
            continue
        try:
            num = int(mm.group("line"))
        except Exception:
            continue
        if start_line < 0:
            start_line = num
        end_line = num

    if start_line < 0 or end_line < 0:
        start_line = 1
        end_line = 1

    sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return (path, sha, start_line, end_line)


def _parse_open_attachment_identity(body: str) -> Optional[Tuple[str, Optional[str], int, int]]:
    """Return (handle, sha256, start_line, end_line) when parseable."""
    raw = str(body or "")
    if not raw.strip():
        return None
    lines = raw.splitlines()
    if not lines:
        return None
    m = _OPEN_ATTACHMENT_HEADER_RE.match(lines[0].strip())
    if not m:
        return None
    handle = _normalize_handle(m.group("handle"))
    sha256 = m.group("sha256")
    sha = str(sha256 or "").strip().lower() or None
    if sha and not re.fullmatch(r"[0-9a-f]{8,64}", sha):
        sha = None

    start_line = 1
    end_line = 1
    try:
        if m.group("start_line") and m.group("end_line"):
            start_line = int(m.group("start_line"))
            end_line = int(m.group("end_line"))
    except Exception:
        start_line = 1
        end_line = 1

    return (handle, sha, start_line, end_line)


def dedup_messages_view(
    messages: List[Dict[str, Any]],
    *,
    session_attachments: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Return a derived LLM-visible messages list with duplicate doc reads stubbed."""
    if not isinstance(messages, list) or not messages:
        return [] if messages is None else list(messages)

    by_handle: Dict[str, list[Dict[str, Any]]] = {}
    for e in session_attachments or []:
        if not isinstance(e, dict):
            continue
        h = _normalize_handle(e.get("handle"))
        if not h:
            continue
        by_handle.setdefault(h, []).append(e)

    out: list[Dict[str, Any]] = []
    seen: Dict[Tuple[str, str, str, int, int], int] = {}

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "").strip()
        content = msg.get("content")
        content_str = "" if content is None else str(content)

        if role != "tool" or not content_str.strip():
            out.append(dict(msg))
            continue

        parsed = parse_tool_message(content_str)
        if parsed is None:
            out.append(dict(msg))
            continue

        tool = parsed.tool_name
        body = parsed.body

        identity: Optional[Tuple[str, str, str, int, int]] = None
        stub: Optional[str] = None

        if tool == "read_file":
            ident = _parse_read_file_identity(body)
            if ident is not None:
                path, sha, start_line, end_line = ident
                identity = ("read_file", path, sha, start_line, end_line)

                candidates = by_handle.get(path) or []
                attachment_hint = ""
                if len(candidates) == 1:
                    a = candidates[0]
                    aid = str(a.get("artifact_id") or "").strip()
                    sha_a = str(a.get("sha256") or "").strip()
                    if aid:
                        attachment_hint = f" Attached artifact: id={aid}" + (f", sha={sha_a[:8]}…" if sha_a else "")
                elif len(candidates) > 1:
                    bits: list[str] = []
                    for a in candidates[:3]:
                        aid = str(a.get("artifact_id") or "").strip()
                        sha_a = str(a.get("sha256") or "").strip()
                        if aid:
                            bits.append(f"{aid}:{sha_a[:8]}…" if sha_a else aid)
                    if bits:
                        attachment_hint = " Attached candidates: " + ", ".join(bits) + " (specify expected_sha256)"

                stub = (
                    f"[read_file]: (duplicate) File already shown above: {path} lines {start_line}-{end_line}.\n"
                    f"Use open_attachment(handle='@{path}', start_line={start_line}, end_line={end_line}) to re-open.{attachment_hint}"
                )

        elif tool == "open_attachment":
            ident2 = _parse_open_attachment_identity(body)
            if ident2 is not None:
                handle, sha, start_line, end_line = ident2
                key_sha = sha or "unknown"
                identity = ("open_attachment", handle, key_sha, start_line, end_line)
                stub = (
                    f"[open_attachment]: (duplicate) Attachment already shown above: {handle} lines {start_line}-{end_line}.\n"
                    f"Re-open with open_attachment(handle='@{handle}', start_line={start_line}, end_line={end_line})."
                )

        if identity is None:
            out.append(dict(msg))
            continue

        if identity in seen:
            out.append(dict(msg, content=stub or content_str))
            continue

        seen[identity] = len(out)
        out.append(dict(msg))

    return out


def execute_open_attachment(
    *,
    artifact_store: ArtifactStore,
    session_id: str,
    artifact_id: Optional[str],
    handle: Optional[str],
    expected_sha256: Optional[str],
    start_line: int,
    end_line: Optional[int],
    max_chars: int,
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """Runtime-owned tool execution for `open_attachment`."""
    sid = str(session_id or "").strip()
    if not sid:
        return False, {"rendered": "Error: session_id is required to open attachments."}, "session_id is required"

    rid = session_memory_owner_run_id(sid)
    handle_norm = _normalize_handle(handle)
    artifact_id_norm = str(artifact_id or "").strip() or None

    expected = str(expected_sha256 or "").strip().lower() or None
    if expected and expected.startswith("sha256:"):
        expected = expected.split(":", 1)[-1].strip() or None
    if expected and not re.fullmatch(r"[0-9a-f]{8,64}", expected):
        expected = None

    # Clamp numeric args defensively.
    try:
        start = int(start_line)
    except Exception:
        start = 1
    if start < 1:
        start = 1
    try:
        end = int(end_line) if end_line is not None else None
    except Exception:
        end = None
    if end is not None and end < start:
        end = start
    try:
        mc = int(max_chars)
    except Exception:
        mc = 8000
    if mc <= 0:
        mc = 8000
    mc = min(mc, 50_000)

    # Resolve artifact metadata.
    metas = artifact_store.list_by_run(rid)
    candidates = [m for m in metas if isinstance(getattr(m, "tags", None), dict) and (m.tags or {}).get("kind") == "attachment"]

    selected_meta = None
    if artifact_id_norm:
        for m in candidates:
            if str(getattr(m, "artifact_id", "") or "") == artifact_id_norm:
                selected_meta = m
                break
        if selected_meta is None:
            return (
                False,
                {"rendered": f"Error: attachment artifact_id '{artifact_id_norm}' not found in this session."},
                "attachment not found",
            )
    else:
        if not handle_norm:
            return False, {"rendered": "Error: provide artifact_id or handle."}, "missing artifact_id/handle"

        matches: list[Any] = []
        for m in candidates:
            tags = getattr(m, "tags", {}) or {}
            p = _normalize_handle(tags.get("path") or tags.get("source_path"))
            fn = _normalize_handle(tags.get("filename"))
            if p == handle_norm or fn == handle_norm:
                matches.append(m)

        if expected:
            matches2: list[Any] = []
            for m in matches:
                tags = getattr(m, "tags", {}) or {}
                sha = str(tags.get("sha256") or "").strip().lower()
                if sha and sha == expected:
                    matches2.append(m)
            matches = matches2

        if not matches:
            return (
                False,
                {"rendered": f"Error: no attachment matches handle '@{handle_norm}' in this session."},
                "attachment not found",
            )

        if len(matches) > 1:
            # List a few candidates to help the model disambiguate.
            cand: list[dict[str, Any]] = []
            for m in matches[:5]:
                tags = getattr(m, "tags", {}) or {}
                sha = str(tags.get("sha256") or "").strip()
                cand.append({"artifact_id": str(getattr(m, "artifact_id", "") or ""), "sha256": sha or None})
            return (
                False,
                {
                    "rendered": f"Error: multiple attachments match '@{handle_norm}'. Provide expected_sha256 or artifact_id.",
                    "candidates": cand,
                },
                "multiple matches",
            )

        selected_meta = matches[0]

    aid = str(getattr(selected_meta, "artifact_id", "") or "")
    tags = dict(getattr(selected_meta, "tags", {}) or {})
    ct = str(getattr(selected_meta, "content_type", "") or "")
    size_bytes = int(getattr(selected_meta, "size_bytes", 0) or 0)
    sha_tag = str(tags.get("sha256") or "").strip().lower() or None
    handle_final = _normalize_handle(tags.get("path") or tags.get("source_path") or tags.get("filename") or handle_norm or "")
    if not handle_final:
        handle_final = aid

    # Only support text-like content in v0.
    ct_low = ct.lower().strip()
    text_like = ct_low.startswith("text/") or ct_low in {
        "application/json",
        "application/yaml",
        "application/x-yaml",
        "application/xml",
        "application/javascript",
        "application/typescript",
    }
    if not text_like:
        return (
            False,
            {"rendered": f"Error: content_type '{ct}' is not supported by open_attachment(v0)."},
            "unsupported content_type",
        )

    artifact = artifact_store.load(aid)
    if artifact is None:
        return False, {"rendered": f"Error: failed to load artifact '{aid}'."}, "artifact not found"

    try:
        text = artifact.content.decode("utf-8")
    except Exception:
        return False, {"rendered": "Error: attachment is not valid UTF-8 text (binary?)"}, "binary content"

    lines = text.splitlines()
    if not lines:
        header = f"Attachment: @{handle_final} (id={aid}" + (f", sha={sha_tag}" if sha_tag else "") + ", lines 0-0)"
        return True, {"rendered": header, "artifact_id": aid, "handle": handle_final, "sha256": sha_tag, "content_type": ct}, None

    start_idx = min(max(start - 1, 0), len(lines) - 1)
    end_idx = len(lines) - 1 if end is None else min(max(end - 1, start_idx), len(lines) - 1)
    selected = lines[start_idx : end_idx + 1]

    shown_start = start_idx + 1
    shown_end = end_idx + 1
    num_width = max(1, len(str(shown_end)))

    # Build bounded, line-numbered excerpt.
    header = (
        f"Attachment: @{handle_final} (id={aid}"
        + (f", sha={sha_tag}" if sha_tag else "")
        + f", lines {shown_start}-{shown_end})"
    )

    # Allocate budget for excerpt lines.
    remaining = max(0, mc - len(header) - 2)
    rendered_lines: list[str] = []
    used = 0
    truncated = False
    for i, ln in enumerate(selected):
        line_no = shown_start + i
        row = f"{line_no:>{num_width}}: {ln}"
        add_len = len(row) + (1 if rendered_lines else 0)
        if used + add_len > remaining and rendered_lines:
            truncated = True
            break
        if used + add_len > remaining and not rendered_lines:
            # Always show at least one line, even if it truncates.
            row = row[: max(0, remaining - 1)] + "…" if remaining > 1 else "…"
            rendered_lines.append(row)
            truncated = True
            break
        rendered_lines.append(row)
        used += add_len

    rendered = header + "\n\n" + "\n".join(rendered_lines)
    if truncated and len(rendered) + 18 <= mc:
        rendered += "\n\n… (truncated)"

    out: Dict[str, Any] = {
        "rendered": rendered,
        "artifact_id": aid,
        "handle": handle_final,
        "sha256": sha_tag,
        "content_type": ct,
        "size_bytes": size_bytes,
        "start_line": shown_start,
        "end_line": shown_end,
        "truncated": bool(truncated),
    }
    return True, out, None


__all__ = [
    "session_memory_owner_run_id",
    "list_session_attachments",
    "render_session_attachments_system_message",
    "dedup_messages_view",
    "execute_open_attachment",
]
