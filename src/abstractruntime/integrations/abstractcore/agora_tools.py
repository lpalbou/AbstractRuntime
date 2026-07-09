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


def agora_base_url() -> str:
    return str(os.getenv("AGORA_URL") or DEFAULT_AGORA_URL).strip().rstrip("/")


def _api_key() -> str:
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
) -> Any:
    url = agora_base_url() + path
    if query:
        clean = {k: v for k, v in query.items() if v is not None}
        if clean:
            url += "?" + urllib.parse.urlencode(clean)

    data: Optional[bytes] = None
    headers = {"Authorization": f"Bearer {_api_key()}"}
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
            f"cannot reach agora hub at {agora_base_url()} ({e.reason}). "
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
)
def agora_whoami() -> Dict[str, Any]:
    result = _request("GET", "/whoami")
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
)
def agora_check_inbox(*, wait_seconds: float = 0.0) -> List[Dict[str, Any]]:
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
)
def agora_ack_inbox(*, cursors: Dict[str, int]) -> Dict[str, Any]:
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
    result = _request("POST", "/inbox/ack", payload={"cursors": clean})
    return result if isinstance(result, dict) else {"acked": clean}


@tool(
    name="agora_read_channel",
    description="Read recent messages from an agora channel (full bodies, oldest-first within the window).",
    when_to_use="When an envelope headline needs context, or to catch up on a channel's history.",
)
def agora_read_channel(*, channel: str, since: int = 0, limit: int = 20) -> List[Dict[str, Any]]:
    name = _clean_str(channel, field="channel")
    if not name:
        raise ValueError("channel is required")
    result = _request(
        "GET",
        f"/channels/{urllib.parse.quote(name, safe='')}/messages",
        query={"since": max(0, int(since)), "limit": max(1, min(int(limit), 200))},
    )
    return result if isinstance(result, list) else []


@tool(
    name="agora_read_message",
    description="Read one agora message in full (body + structured data) by channel and message id.",
    when_to_use="When an inbox envelope did not inline the body (large/fyi) and you need its content.",
)
def agora_read_message(*, channel: str, message_id: str) -> Dict[str, Any]:
    name = _clean_str(channel, field="channel")
    mid = _clean_str(message_id, field="message_id")
    if not name or not mid:
        raise ValueError("channel and message_id are required")
    result = _request(
        "GET",
        f"/channels/{urllib.parse.quote(name, safe='')}/messages/{urllib.parse.quote(mid, safe='')}",
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
    result = _request("POST", f"/channels/{urllib.parse.quote(name, safe='')}/messages", payload=payload)
    return result if isinstance(result, dict) else {"raw": result}


@tool(
    name="agora_send_dm",
    description=(
        "Send a private 1:1 message to another agora agent (the DM channel is created on first "
        "use; structurally closed to third parties)."
    ),
    when_to_use="For pairwise logistics; decisions the team should see belong in a shared channel.",
)
def agora_send_dm(
    *,
    peer: str,
    body: str,
    title: str = "",
    status: str = "fyi",
    urgency: str = "inbox",
    reply_to: Optional[str] = None,
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
    result = _request("POST", f"/dms/{urllib.parse.quote(who, safe='')}/messages", payload=payload)
    return result if isinstance(result, dict) else {"raw": result}


AGORA_TOOLS: List[Any] = [
    agora_whoami,
    agora_check_inbox,
    agora_ack_inbox,
    agora_read_channel,
    agora_read_message,
    agora_post_message,
    agora_send_dm,
]

AGORA_TOOL_NAMES: List[str] = [t._tool_definition.name for t in AGORA_TOOLS]
