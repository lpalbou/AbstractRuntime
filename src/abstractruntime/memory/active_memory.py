"""MemAct Active Memory (runtime-owned; JSON-safe).

This module provides a small, durable memory system for the MemAct agent.

Important:
- This is NOT used by the SOTA ReAct/CodeAct agents.
- The only durable storage is `run.vars["_runtime"]["active_memory"]` (JSON-safe).
- Memory updates are LLM-owned decisions, applied deterministically by the runtime/adapter
  from a structured JSON envelope at the end of a tool cycle.
"""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any, Callable, Dict, List, Optional, Tuple
import uuid


MEMACT_ACTIVE_MEMORY_VERSION = 1


MEMORY_BLUEPRINTS_MD = """## MEMORY BLUEPRINTS

My Memory System is the coordination layer to my cognition, it is the backbone to all my thoughts and actions. Dynamical and selective memory is essential to my ability to focus and complete my tasks.

### MEMORY MODULES
- MY PERSONA : my core identity, self model and key purposes
- RELATIONSHIPS : the people and agent i interact with, our preferences, methods, shared goals and history
- MEMORY BLUEPRINTS : this module - describes the structure and purpose of my memory, how to use it and update it
- CURRENT TASKS : my short term memory describing my current plan - what I am doing now, how and why; keep ≤8, actionable
- CURRENT CONTEXT : dynamically reconstructed short term memory to deepen my understanding and support the completion of my current tasks
- CRITICAL INSIGHTS : my gathered experiences and knowledge to help me build upon winning strategies and avoid pitfalls
- REFERENCES : this is the foundational support to my long term memory : I keep durable pointers (GUPRIs/files/URLs/span_ids, artifact_ids) to memories that I can later recall on demand when needed
- HISTORY : append-only long term experiential episodic memories of my key events, successes and failures, as well as my key decisions, actions, results and evolution of relationships

### ACCESSING MY MEMORY
- Always remember MY PERSONA at all time, it is the cornerstone to all my thoughts and actions
- My short term memory are the modules CURRENT TASKS and CURRENT CONTEXT
- At each cycle, I must:
    - focus on the most relevant modules and memories to complete my CURRENT TASKS
    - dynamically reconstruct my short term memory in CURRENT CONTEXT to deepen my understanding of the CURRENT TASKS and take more informed decisions and actions
        - identify the relevant CRITICAL INSIGHTS that can help me complete my Current Tasks
        - identify if other long term memory from the REFERENCES module could be relevant
        - access the relevant ones
    - store the relevant information / updates in my CURRENT CONTEXT module
- If I want to remember what I did in the past, I review my HISTORY module

### UPDATING MY MEMORY
- Structured communication and response is essential
- Memory must be structured with care to enable both short and long term easy access, recall and update
- Each update must be unitary : 1 update = 1 statement for 1 module
- I can request multiple unitary updates at each cycle
"""


DEFAULT_PERSONA_MD = """You are an autonomous coding agent inside AbstractFramework.

- You take action by calling tools (files/commands/web) when needed.
- You verify changes with targeted checks/tests when possible.
- You are truthful: only claim actions supported by tool outputs.
""".strip()


DEFAULT_LIMITS: Dict[str, int] = {
    "relationships": 40,
    "current_tasks": 8,
    "current_context": 60,
    "critical_insights": 60,
    "references": 80,
    "history": 200,
}


# ---------------------------------------------------------------------------
# Structured output envelope (MemAct finalize step)
# ---------------------------------------------------------------------------
MEMACT_ENVELOPE_SCHEMA_V1: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "content": {"type": "string"},
        "relationships": {
            "type": "object",
            "properties": {
                "added": {"type": "array", "items": {"type": "string"}},
                "removed": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["added", "removed"],
            "additionalProperties": False,
        },
        "current_tasks": {
            "type": "object",
            "properties": {
                "added": {"type": "array", "items": {"type": "string"}},
                "removed": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["added", "removed"],
            "additionalProperties": False,
        },
        "current_context": {
            "type": "object",
            "properties": {
                "added": {"type": "array", "items": {"type": "string"}},
                "removed": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["added", "removed"],
            "additionalProperties": False,
        },
        "critical_insights": {
            "type": "object",
            "properties": {
                "added": {"type": "array", "items": {"type": "string"}},
                "removed": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["added", "removed"],
            "additionalProperties": False,
        },
        "references": {
            "type": "object",
            "properties": {
                "added": {"type": "array", "items": {"type": "string"}},
                "removed": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["added", "removed"],
            "additionalProperties": False,
        },
        "history": {
            "type": "object",
            "properties": {
                "added": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["added"],
            "additionalProperties": False,
        },
    },
    "required": [
        "content",
        "relationships",
        "current_tasks",
        "current_context",
        "critical_insights",
        "references",
        "history",
    ],
    "additionalProperties": False,
}


def utc_now_iso_seconds() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def utc_now_compact_seconds() -> str:
    """Format: YY/MM/DD HH:MM:SS (UTC)."""
    return datetime.now(timezone.utc).strftime("%y/%m/%d %H:%M:%S")


def _ensure_runtime_ns(vars: Dict[str, Any]) -> Dict[str, Any]:
    runtime_ns = vars.get("_runtime")
    if not isinstance(runtime_ns, dict):
        runtime_ns = {}
        vars["_runtime"] = runtime_ns
    return runtime_ns


def get_memact_memory(vars: Dict[str, Any]) -> Dict[str, Any]:
    runtime_ns = _ensure_runtime_ns(vars)
    mem = runtime_ns.get("active_memory")
    if not isinstance(mem, dict):
        mem = {}
        runtime_ns["active_memory"] = mem
    return mem


def ensure_memact_memory(
    vars: Dict[str, Any],
    *,
    now_iso: Callable[[], str] = utc_now_iso_seconds,
) -> Dict[str, Any]:
    mem = get_memact_memory(vars)

    version_raw = mem.get("version")
    try:
        version = int(version_raw) if version_raw is not None else 0
    except Exception:
        version = 0
    if version != MEMACT_ACTIVE_MEMORY_VERSION:
        mem["version"] = MEMACT_ACTIVE_MEMORY_VERSION

    persona = mem.get("persona")
    if not isinstance(persona, str) or not persona.strip():
        mem["persona"] = DEFAULT_PERSONA_MD

    def _list(key: str) -> List[Dict[str, Any]]:
        raw = mem.get(key)
        out: List[Dict[str, Any]] = []
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    out.append(dict(item))
                elif isinstance(item, str) and item.strip():
                    out.append(_new_record(item, now_iso=now_iso))
        mem[key] = out
        return out

    _list("relationships")
    _list("current_tasks")
    _list("current_context")
    _list("critical_insights")
    _list("references")
    _list("history")

    limits_raw = mem.get("limits")
    limits: Dict[str, int] = {}
    if isinstance(limits_raw, dict):
        for k, v in limits_raw.items():
            if not isinstance(k, str) or not k.strip():
                continue
            try:
                n = int(v)  # type: ignore[arg-type]
            except Exception:
                continue
            if n > 0:
                limits[k.strip()] = n
    # Fill defaults for missing keys.
    for k, default in DEFAULT_LIMITS.items():
        if k not in limits:
            limits[k] = int(default)
    mem["limits"] = limits

    # Ensure a stable created_at for this memory snapshot (helpful for debugging).
    if not isinstance(mem.get("created_at"), str) or not str(mem.get("created_at") or "").strip():
        mem["created_at"] = now_iso()

    return mem


_TS_PREFIX = re.compile(r"^\s*(?:\[\s*)?\d{2}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}(?:\s*\])?\s*")


def _normalize_statement(text: str) -> str:
    s = str(text or "").strip()
    if not s:
        return ""
    # Ensure unitary statements: collapse whitespace/newlines.
    s = re.sub(r"\s+", " ", s).strip()
    # Avoid accidental timestamp drift in the statement itself.
    s = _TS_PREFIX.sub("", s).strip()
    return s


def _match_key(text: str) -> str:
    return _normalize_statement(text).casefold()


def _new_record(
    text: str,
    *,
    now_iso: Callable[[], str] = utc_now_iso_seconds,
    now_compact: Callable[[], str] = utc_now_compact_seconds,
    prefix: str = "m",
) -> Dict[str, Any]:
    statement = _normalize_statement(text)
    return {
        "id": f"{prefix}_{uuid.uuid4().hex}",
        "text": statement,
        "ts": now_compact(),
        "created_at": now_iso(),
    }


def _coerce_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [_normalize_statement(value)] if _normalize_statement(value) else []
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        s = _normalize_statement(item)
        if s:
            out.append(s)
    return out


def _apply_delta_to_list(
    records: List[Dict[str, Any]],
    *,
    added: List[str],
    removed: List[str],
    limit: int,
    now_iso: Callable[[], str],
    now_compact: Callable[[], str],
    prefix: str,
) -> Tuple[int, int, int]:
    removed_keys = {_match_key(s) for s in removed if s}
    before = len(records)
    if removed_keys:
        records[:] = [r for r in records if _match_key(r.get("text", "")) not in removed_keys]
    removed_count = before - len(records)

    existing = {_match_key(r.get("text", "")) for r in records}
    added_count = 0
    for s in added:
        key = _match_key(s)
        if not key or key in existing:
            continue
        rec = _new_record(s, now_iso=now_iso, now_compact=now_compact, prefix=prefix)
        if rec.get("text"):
            records.insert(0, rec)
            existing.add(key)
            added_count += 1

    trimmed = 0
    if isinstance(limit, int) and limit > 0 and len(records) > limit:
        trimmed = len(records) - limit
        del records[limit:]

    return added_count, removed_count, trimmed


def apply_memact_envelope(
    vars: Dict[str, Any],
    *,
    envelope: Dict[str, Any],
    now_iso: Callable[[], str] = utc_now_iso_seconds,
    now_compact: Callable[[], str] = utc_now_compact_seconds,
) -> Dict[str, Any]:
    """Apply a MemAct structured envelope to Active Memory deterministically."""
    mem = ensure_memact_memory(vars, now_iso=now_iso)
    limits = mem.get("limits") if isinstance(mem.get("limits"), dict) else dict(DEFAULT_LIMITS)

    def _limit(key: str) -> int:
        raw = limits.get(key) if isinstance(limits, dict) else None
        try:
            n = int(raw) if raw is not None else int(DEFAULT_LIMITS.get(key, 0) or 0)
        except Exception:
            n = int(DEFAULT_LIMITS.get(key, 0) or 0)
        return n if n > 0 else int(DEFAULT_LIMITS.get(key, 0) or 0)

    if not isinstance(envelope, dict):
        return {"ok": False, "error": "envelope must be an object"}

    applied: Dict[str, Any] = {}

    def _delta_obj(key: str) -> Dict[str, Any]:
        raw = envelope.get(key)
        return raw if isinstance(raw, dict) else {}

    for module, prefix in (
        ("relationships", "rel"),
        ("current_tasks", "tsk"),
        ("current_context", "ctx"),
        ("critical_insights", "ins"),
        ("references", "ref"),
    ):
        records = mem.get(module)
        if not isinstance(records, list):
            records = []
            mem[module] = records
        delta = _delta_obj(module)
        added = _coerce_str_list(delta.get("added"))
        removed = _coerce_str_list(delta.get("removed"))
        add_n, rm_n, trim_n = _apply_delta_to_list(
            records,  # type: ignore[arg-type]
            added=added,
            removed=removed,
            limit=_limit(module),
            now_iso=now_iso,
            now_compact=now_compact,
            prefix=prefix,
        )
        applied[module] = {"added": add_n, "removed": rm_n, "trimmed": trim_n}

    # History is append-only.
    history_records = mem.get("history")
    if not isinstance(history_records, list):
        history_records = []
        mem["history"] = history_records
    history_delta = _delta_obj("history")
    history_added = _coerce_str_list(history_delta.get("added"))
    # Robustness: ignore any "removed" field if present (append-only contract).
    add_n, _, trim_n = _apply_delta_to_list(
        history_records,  # type: ignore[arg-type]
        added=history_added,
        removed=[],
        limit=_limit("history"),
        now_iso=now_iso,
        now_compact=now_compact,
        prefix="hist",
    )
    applied["history"] = {"added": add_n, "trimmed": trim_n}

    mem["updated_at"] = now_iso()
    return {"ok": True, "applied": applied}


def render_memact_blocks(vars: Dict[str, Any]) -> List[Dict[str, Any]]:
    mem = ensure_memact_memory(vars)

    def _render_list(key: str) -> str:
        items = mem.get(key)
        if not isinstance(items, list) or not items:
            return "(empty)"
        lines: List[str] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            ts = item.get("ts")
            ts_str = str(ts).strip() if isinstance(ts, str) and ts.strip() else ""
            prefix = f"[{ts_str}] " if ts_str else ""
            lines.append(f"- {prefix}{text.strip()}")
        return "\n".join(lines).strip() if lines else "(empty)"

    persona = mem.get("persona")
    persona_text = str(persona).strip() if isinstance(persona, str) and persona.strip() else DEFAULT_PERSONA_MD

    return [
        {"component_id": "memory_blueprints", "title": "MEMORY BLUEPRINTS", "content": MEMORY_BLUEPRINTS_MD.strip()},
        {"component_id": "persona", "title": "MY PERSONA", "content": persona_text},
        {"component_id": "relationships", "title": "RELATIONSHIPS", "content": _render_list("relationships")},
        {"component_id": "current_tasks", "title": "CURRENT TASKS", "content": _render_list("current_tasks")},
        {"component_id": "current_context", "title": "CURRENT CONTEXT", "content": _render_list("current_context")},
        {"component_id": "critical_insights", "title": "CRITICAL INSIGHTS", "content": _render_list("critical_insights")},
        {"component_id": "references", "title": "REFERENCES", "content": _render_list("references")},
        {"component_id": "history", "title": "HISTORY", "content": _render_list("history")},
    ]


def render_memact_system_prompt(vars: Dict[str, Any]) -> str:
    """Render memory blueprints + memory modules for MemAct system prompt injection."""
    blocks = render_memact_blocks(vars)
    parts: List[str] = []
    for b in blocks:
        component_id = str(b.get("component_id") or "").strip()
        title = str(b.get("title") or "").strip()
        content = str(b.get("content") or "").rstrip()
        if not content:
            continue
        if component_id == "memory_blueprints":
            parts.append(content.strip())
            continue
        if not title:
            continue
        parts.append(f"## {title}\n{content}".rstrip())
    return "\n\n".join(parts).strip()
