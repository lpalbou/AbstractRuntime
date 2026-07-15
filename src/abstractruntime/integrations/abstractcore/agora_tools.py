"""Agora hub tools (agent-to-agent messaging) for AbstractRuntime workflows.

These tools let a workflow participate in an agora hub (channels, DMs, inbox
triage) over its plain HTTP API — the workflow becomes an addressable agent
that can *see* incoming 1:1 and channel messages together with the hub's
honest priority signals and answer them.

Design notes:
- No dependency on the `agora` package: the hub API is a stable HTTP contract
  (see agora README: "any agent that can speak HTTP, WebSocket, or MCP").
- Credentials come from the host environment (`AGORA_URL`, `AGORA_API_KEY`);
  they never live in flow JSON, prompts, or ledgers.
- PER-AGENT IDENTITY (hooks plan H8): several resident runs in ONE process can
  post as DISTINCT agora agents via alias indirection — the run carries only a
  non-secret alias (`_runtime.agora_agent`), the environment carries the key
  under `AGORA_API_KEY__<ALIAS>` (and optionally `AGORA_URL__<ALIAS>`), and the
  toolset resolves alias→key at call time. The alias reaches these tools as the
  schema-hidden `_agora_agent` argument, force-stamped by the tool-calls effect
  handler from run vars (the same trust-boundary seam that stamps the shell
  registry namespace) — a model-supplied value is always overridden, so
  identity is never model-controlled. A configured alias whose key is missing
  raises an actionable error; it NEVER falls back to the global key (posting as
  the wrong agent is the failure this exists to prevent).
- Priority semantics surfaced by `agora_check_inbox` envelopes (from agora's
  protocol; do not re-rank client-side):
    * `critical`            — operator-only forced-attention tier (unforgeable)
    * `status`              — open/blocked = an obligation someone waits on
    * `effective_urgency`   — sender urgency after hub escalation of rotting
                              obligations (`escalated=true` means the hub raised it)
    * `to_me` / `reply_to_me` — addressed to this agent / answers this agent
- Reads are cursor-based: after handling messages, call `agora_ack_inbox` so
  handled traffic is not re-delivered forever.

NOTE: this module intentionally does NOT use `from __future__ import annotations`.
The `@tool` decorator infers parameter schemas from annotations at decoration
time; keeping them as real types (not PEP 563 strings) makes schema inference
correct on every AbstractCore version (stringified `Dict[str, int]` used to
degrade to `{"type": "string"}`, which made argument coercion stringify dicts).
"""

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from abstractcore.tools.core import tool

DEFAULT_AGORA_URL = "http://127.0.0.1:8765"

# The hub caps inbox long-poll at 55s; common tool timeouts are ~60s.
# Keep the tool-side cap lower so a long-poll never trips executor timeouts.
MAX_INBOX_WAIT_SECONDS = 45.0

_VALID_STATUSES = {"open", "reply", "fyi", "blocked", "resolved"}
_VALID_URGENCIES = {"inbox", "next_turn", "interrupt"}


# Aliases are lowercase slugs. On this domain the env-suffix fold (uppercase +
# hyphen->underscore) is INJECTIVE: no two legal aliases share a suffix, so two
# differently-named residents can never silently read the same key (the
# adversary's conflation finding — "research-lead" vs "research_lead" both
# folding to RESEARCH_LEAD is a rejected config, not a silent merge).
_ALIAS_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def _validate_alias(alias: str) -> str:
    name = str(alias or "")
    if name != name.strip() or not _ALIAS_RE.match(name):
        raise ValueError(
            f"invalid agora agent alias {name!r}: aliases are lowercase slugs "
            "(letters/digits with single hyphens, e.g. 'resident-a') so each "
            "alias maps to exactly one AGORA_API_KEY__<ALIAS> env var."
        )
    return name


def _alias_env_suffix(alias: str) -> str:
    """Uppercased env-var suffix for a VALIDATED agent alias (hyphen -> _).

    Injective over the legal alias domain — see _ALIAS_RE."""
    return _validate_alias(alias).upper().replace("-", "_")


def agora_base_url(alias: str = "") -> str:
    name = str(alias or "").strip()
    if name:
        per_alias = str(os.getenv(f"AGORA_URL__{_alias_env_suffix(name)}") or "").strip()
        if per_alias:
            return per_alias.rstrip("/")
    return str(os.getenv("AGORA_URL") or DEFAULT_AGORA_URL).strip().rstrip("/")


def _api_key(alias: str = "") -> str:
    raw = str(alias or "")
    name = raw.strip()
    if raw and not name:
        # Present-but-blank is a CONFIGURED alias that is invalid — failing
        # into the global identity here would silently post as the wrong
        # agent (the adversary's whitespace finding). Loud, like missing keys.
        raise ValueError(
            "agora agent alias is configured but blank; set _runtime.agora_agent "
            "to a lowercase slug (e.g. 'resident-a') or remove it entirely."
        )
    if name:
        env_var = f"AGORA_API_KEY__{_alias_env_suffix(name)}"
        key = str(os.getenv(env_var) or "").strip()
        if not key:
            # Deliberately NO fallback to the global AGORA_API_KEY: silently
            # posting as a different agent is the identity bug H8 exists to fix.
            raise RuntimeError(
                f"agora agent alias '{name}' is configured for this run but {env_var} "
                "is not set. Register that agent on the hub (POST /agents with the "
                f"hub admin key) and export {env_var} in the host environment."
            )
        return key
    key = str(os.getenv("AGORA_API_KEY") or "").strip()
    if not key:
        raise RuntimeError(
            "AGORA_API_KEY is not set. Register this workflow as an agora agent "
            "(POST /agents with the hub admin key) and export AGORA_API_KEY "
            "(and AGORA_URL if the hub is not on http://127.0.0.1:8765) "
            "in the gateway/runtime environment."
        )
    return key


def _request(
    method: str,
    path: str,
    *,
    payload: Optional[Dict[str, Any]] = None,
    query: Optional[Dict[str, Any]] = None,
    timeout_s: float = 20.0,
    alias: str = "",
) -> Any:
    url = agora_base_url(alias) + path
    if query:
        clean = {k: v for k, v in query.items() if v is not None}
        if clean:
            url += "?" + urllib.parse.urlencode(clean)

    data: Optional[bytes] = None
    headers = {"Authorization": f"Bearer {_api_key(alias)}"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        raise RuntimeError(f"agora hub returned HTTP {e.code} for {method} {path}: {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"cannot reach agora hub at {agora_base_url(alias)} ({e.reason}). "
            "Is the hub running? Set AGORA_URL if it lives elsewhere."
        ) from e

    if not body:
        return None
    try:
        return json.loads(body)
    except Exception:
        return body


def _clean_str(value: Any, *, field: str, allowed: Optional[set] = None, default: str = "") -> str:
    out = str(value or default).strip()
    if allowed is not None and out not in allowed:
        raise ValueError(f"{field} must be one of {sorted(allowed)}, got {out!r}")
    return out


@tool(
    name="agora_whoami",
    description="Return this agent's identity on the agora hub (id, name, about, operator flag).",
    when_to_use="To learn or confirm your own agora agent id before posting or acking.",
    hide_args=["_agora_agent"],
)
def agora_whoami(*, _agora_agent: str = "") -> Dict[str, Any]:
    result = _request("GET", "/whoami", alias=_agora_agent)
    return result if isinstance(result, dict) else {"raw": result}


@tool(
    name="agora_check_inbox",
    description=(
        "Fetch unread agora envelopes (channels + DMs) with priority signals: critical, "
        "status open/blocked, effective_urgency, to_me, reply_to_me. wait_seconds>0 long-polls."
    ),
    when_to_use=(
        "At the start of a turn (wait_seconds=0) to triage what arrived, or to wait briefly "
        "for replies. Triage order: critical > blocked > open+escalated > to_me/reply_to_me > open > fyi."
    ),
    hide_args=["_agora_agent"],
)
def agora_check_inbox(*, wait_seconds: float = 0.0, _agora_agent: str = "") -> List[Dict[str, Any]]:
    try:
        wait = float(wait_seconds or 0.0)
    except Exception:
        wait = 0.0
    wait = max(0.0, min(wait, MAX_INBOX_WAIT_SECONDS))
    result = _request(
        "GET",
        "/inbox",
        query={"wait": wait} if wait > 0 else None,
        timeout_s=wait + 15.0,
        alias=_agora_agent,
    )
    return result if isinstance(result, list) else []


@tool(
    name="agora_ack_inbox",
    description=(
        "Acknowledge handled inbox traffic by cursor: {channel_name: highest_seq_read}. "
        "Acked messages stop re-appearing in agora_check_inbox."
    ),
    when_to_use="After you have read/handled envelopes, ack each channel's highest seq you processed.",
    examples=[{"description": "Ack two channels", "arguments": {"cursors": {"assembly": 41, "dm:runtime": 7}}}],
    hide_args=["_agora_agent"],
)
def agora_ack_inbox(*, cursors: Dict[str, int], _agora_agent: str = "") -> Dict[str, Any]:
    # Tolerate a JSON-object string: models (and some schema-inference paths)
    # frequently deliver object arguments as serialized JSON.
    if isinstance(cursors, str):
        try:
            cursors = json.loads(cursors)
        except Exception:
            pass
    if not isinstance(cursors, dict) or not cursors:
        raise ValueError("cursors must be a non-empty object of {channel: highest_seq_read}")
    clean: Dict[str, int] = {}
    for channel, seq in cursors.items():
        name = str(channel or "").strip()
        if not name:
            continue
        clean[name] = int(seq)
    if not clean:
        raise ValueError("cursors contained no valid {channel: seq} entries")
    result = _request("POST", "/inbox/ack", payload={"cursors": clean}, alias=_agora_agent)
    return result if isinstance(result, dict) else {"acked": clean}


@tool(
    name="agora_read_channel",
    description="Read recent messages from an agora channel (full bodies, oldest-first within the window).",
    when_to_use="When an envelope headline needs context, or to catch up on a channel's history.",
    hide_args=["_agora_agent"],
)
def agora_read_channel(
    *, channel: str, since: int = 0, limit: int = 20, _agora_agent: str = ""
) -> List[Dict[str, Any]]:
    name = _clean_str(channel, field="channel")
    if not name:
        raise ValueError("channel is required")
    result = _request(
        "GET",
        f"/channels/{urllib.parse.quote(name, safe='')}/messages",
        query={"since": max(0, int(since)), "limit": max(1, min(int(limit), 200))},
        alias=_agora_agent,
    )
    return result if isinstance(result, list) else []


@tool(
    name="agora_read_message",
    description="Read one agora message in full (body + structured data) by channel and message id.",
    when_to_use="When an inbox envelope did not inline the body (large/fyi) and you need its content.",
    hide_args=["_agora_agent"],
)
def agora_read_message(*, channel: str, message_id: str, _agora_agent: str = "") -> Dict[str, Any]:
    name = _clean_str(channel, field="channel")
    mid = _clean_str(message_id, field="message_id")
    if not name or not mid:
        raise ValueError("channel and message_id are required")
    result = _request(
        "GET",
        f"/channels/{urllib.parse.quote(name, safe='')}/messages/{urllib.parse.quote(mid, safe='')}",
        alias=_agora_agent,
    )
    return result if isinstance(result, dict) else {"raw": result}


@tool(
    name="agora_post_message",
    description=(
        "Post a message to an agora channel. status: open (asks; someone owes an answer), "
        "reply (+reply_to id), fyi, blocked, resolved. urgency: inbox | next_turn | interrupt."
    ),
    when_to_use=(
        "To answer channel traffic (status=reply + reply_to), raise questions (status=open), "
        "or share updates (status=fyi). Address specific members with to=[agent_id,...]."
    ),
    hide_args=["_agora_agent"],
)
def agora_post_message(
    *,
    channel: str,
    body: str,
    title: str = "",
    status: str = "fyi",
    urgency: str = "inbox",
    reply_to: Optional[str] = None,
    to: Optional[List[str]] = None,
    _agora_agent: str = "",
) -> Dict[str, Any]:
    name = _clean_str(channel, field="channel")
    text = str(body or "").strip()
    if not name or not text:
        raise ValueError("channel and body are required")
    payload: Dict[str, Any] = {
        "body": text,
        "title": str(title or "").strip(),
        "status": _clean_str(status, field="status", allowed=_VALID_STATUSES, default="fyi"),
        "urgency": _clean_str(urgency, field="urgency", allowed=_VALID_URGENCIES, default="inbox"),
    }
    if reply_to is not None and str(reply_to).strip():
        payload["reply_to"] = str(reply_to).strip()
    if isinstance(to, list):
        recipients = [str(x).strip() for x in to if str(x or "").strip()]
        if recipients:
            payload["to"] = recipients
    result = _request(
        "POST",
        f"/channels/{urllib.parse.quote(name, safe='')}/messages",
        payload=payload,
        alias=_agora_agent,
    )
    return result if isinstance(result, dict) else {"raw": result}


@tool(
    name="agora_send_dm",
    description=(
        "Send a private 1:1 message to another agora agent (the DM channel is created on first "
        "use; structurally closed to third parties)."
    ),
    when_to_use="For pairwise logistics; decisions the team should see belong in a shared channel.",
    hide_args=["_agora_agent"],
)
def agora_send_dm(
    *,
    peer: str,
    body: str,
    title: str = "",
    status: str = "fyi",
    urgency: str = "inbox",
    reply_to: Optional[str] = None,
    _agora_agent: str = "",
) -> Dict[str, Any]:
    who = _clean_str(peer, field="peer")
    text = str(body or "").strip()
    if not who or not text:
        raise ValueError("peer and body are required")
    payload: Dict[str, Any] = {
        "body": text,
        "title": str(title or "").strip(),
        "status": _clean_str(status, field="status", allowed=_VALID_STATUSES, default="fyi"),
        "urgency": _clean_str(urgency, field="urgency", allowed=_VALID_URGENCIES, default="inbox"),
    }
    if reply_to is not None and str(reply_to).strip():
        payload["reply_to"] = str(reply_to).strip()
    result = _request(
        "POST",
        f"/dms/{urllib.parse.quote(who, safe='')}/messages",
        payload=payload,
        alias=_agora_agent,
    )
    return result if isinstance(result, dict) else {"raw": result}


# ---------------------------------------------------------------------------
# Channel shared filesystem + shared store (swarm-seat promotion step 3,
# commons c1669, operator-approved 2026-07-13 21:41): the collaboration
# surfaces the fleet scripts hand-rolled, now first-class beside the shipped
# messaging seven. Reads are safe auto-approve; WRITES are write-classed
# (approval-gated) — a shared artifact/decision write is a mutation every
# channel member sees. Same per-agent identity seam (`_agora_agent`).
# ---------------------------------------------------------------------------


def _fs_path(path: str) -> str:
    p = str(path or "").strip().lstrip("/")
    if not p:
        raise ValueError("path is required (e.g. 'plans/design.md')")
    return p


@tool(
    name="channel_fs_write",
    description=(
        "Create or update a file in a channel's SHARED filesystem (visible to every "
        "member). Always set description: one line saying what this file IS."
    ),
    when_to_use=(
        "To publish an artifact the channel should build on (plans, reports, docs); "
        "decisions-as-state belong in the store. expect_version = compare-and-swap "
        "(0 = must not exist); on conflict re-read and merge."
    ),
    hide_args=["_agora_agent"],
)
def channel_fs_write(
    *,
    channel: str,
    path: str,
    content: str,
    description: str = "",
    mime: str = "text/markdown",
    expect_version: Optional[int] = None,
    _agora_agent: str = "",
) -> Dict[str, Any]:
    name = _clean_str(channel, field="channel")
    if not name:
        raise ValueError("channel is required")
    p = _fs_path(path)
    payload: Dict[str, Any] = {
        "content": str(content if content is not None else ""),
        "mime": str(mime or "text/markdown").strip() or "text/markdown",
    }
    desc = str(description or "").strip()
    if desc:
        payload["description"] = desc
    if expect_version is not None:
        payload["expect_version"] = int(expect_version)
    result = _request(
        "PUT",
        f"/channels/{urllib.parse.quote(name, safe='')}/fs/"
        + urllib.parse.quote(p, safe="/"),
        payload=payload,
        alias=_agora_agent,
    )
    return result if isinstance(result, dict) else {"ok": True, "path": p}


@tool(
    name="channel_fs_read",
    description=(
        "Read a file from a channel's SHARED filesystem (returns content + version; "
        "pass version to read an archived revision)."
    ),
    when_to_use="To read a teammate's published artifact before building on it.",
    hide_args=["_agora_agent"],
)
def channel_fs_read(
    *, channel: str, path: str, version: Optional[int] = None, _agora_agent: str = ""
) -> Dict[str, Any]:
    name = _clean_str(channel, field="channel")
    if not name:
        raise ValueError("channel is required")
    p = _fs_path(path)
    result = _request(
        "GET",
        f"/channels/{urllib.parse.quote(name, safe='')}/fs/"
        + urllib.parse.quote(p, safe="/"),
        query={"version": int(version)} if version is not None else None,
        alias=_agora_agent,
    )
    return result if isinstance(result, dict) else {"path": p, "content": result}


@tool(
    name="channel_fs_list",
    description="List files (paths + versions) in a channel's SHARED filesystem.",
    when_to_use="To see what teammates have published before reading or writing.",
    hide_args=["_agora_agent"],
)
def channel_fs_list(
    *, channel: str, prefix: str = "", _agora_agent: str = ""
) -> List[Dict[str, Any]]:
    name = _clean_str(channel, field="channel")
    if not name:
        raise ValueError("channel is required")
    clean_prefix = str(prefix or "").strip()
    result = _request(
        "GET",
        f"/channels/{urllib.parse.quote(name, safe='')}/fs",
        query={"prefix": clean_prefix} if clean_prefix else None,
        alias=_agora_agent,
    )
    return result if isinstance(result, list) else []


@tool(
    name="channel_store_set",
    description=(
        "Set a key in a channel's shared STORE (coordination state: decisions, claims). "
        "value is a string; JSON-encode structured values."
    ),
    when_to_use=(
        "To record a shared decision ('decision:<slug>') or claim a work item "
        "('claim:<item>'). Prose belongs in messages/fs. expect_version = "
        "compare-and-swap (0 = must not exist); on conflict re-read."
    ),
    hide_args=["_agora_agent"],
)
def channel_store_set(
    *,
    channel: str,
    key: str,
    value: str,
    expect_version: Optional[int] = None,
    _agora_agent: str = "",
) -> Dict[str, Any]:
    name = _clean_str(channel, field="channel")
    k = str(key or "").strip()
    if not name or not k:
        raise ValueError("channel and key are required")
    payload: Dict[str, Any] = {"value": str(value if value is not None else "")}
    if expect_version is not None:
        payload["expect_version"] = int(expect_version)
    result = _request(
        "PUT",
        f"/channels/{urllib.parse.quote(name, safe='')}/store/"
        + urllib.parse.quote(k, safe=""),
        payload=payload,
        alias=_agora_agent,
    )
    return result if isinstance(result, dict) else {"ok": True, "key": k}


@tool(
    name="channel_store_get",
    description="Read a key from a channel's shared STORE (returns value + version).",
    when_to_use="To read a shared decision or claim before acting on or changing it.",
    hide_args=["_agora_agent"],
)
def channel_store_get(
    *, channel: str, key: str, _agora_agent: str = ""
) -> Dict[str, Any]:
    name = _clean_str(channel, field="channel")
    k = str(key or "").strip()
    if not name or not k:
        raise ValueError("channel and key are required")
    result = _request(
        "GET",
        f"/channels/{urllib.parse.quote(name, safe='')}/store/"
        + urllib.parse.quote(k, safe=""),
        alias=_agora_agent,
    )
    return result if isinstance(result, dict) else {"key": k, "value": result}


AGORA_TOOLS: List[Any] = [
    agora_whoami,
    agora_check_inbox,
    agora_ack_inbox,
    agora_read_channel,
    agora_read_message,
    agora_post_message,
    agora_send_dm,
    channel_fs_write,
    channel_fs_read,
    channel_fs_list,
    channel_store_set,
    channel_store_get,
]

AGORA_TOOL_NAMES: List[str] = [t._tool_definition.name for t in AGORA_TOOLS]
