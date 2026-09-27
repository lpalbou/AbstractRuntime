"""Occurrence outcomes, the `notify` convention and attention (contract A/D, C15).

Automations are QUIET by default. An occurrence needs attention only when:

- its output carries `notify: true` (-> `{title: <automation title>, body:
  <answer, 280 chars>}`) or `notify: {title, body}` (title <= 120, body <=
  2000 chars); anything else (absent, false, empty) is quiet;
- it FAILED after its retries were exhausted (a failure that a retry fixed is
  quiet);
- it waits on a human (an interactive USER wait that is not a pause, or an
  EVENT wait carrying a prompt or choices). Those are live facts, computed
  from run state by `pending_waits`, never ledger items.

Notify and final failure allocate exactly one attention item per logical
occurrence: the `automation.completed` record carries it (`attention: {kind,
seq, title, body}`), with a per-automation sequence number. `list_attention`
pages them oldest first.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from ..core.models import RunState, RunStatus, WaitReason
from ..core.vars import is_paused_vars
from ..storage.artifacts import get_artifact_id, is_artifact_ref
from .ledger import automation_records
from .models import NOTIFY_BODY_MAX, NOTIFY_DEFAULT_BODY_MAX, NOTIFY_TITLE_MAX

ATTENTION_CURSOR_PREFIX = "att1:"


class UnresolvableReference(LookupError):
    """An offloaded value (`{"$artifact": id}`) that cannot be loaded."""


def resolve_strict(value: Any, *, artifact_store: Any, _depth: int = 0) -> Any:
    """Replace every artifact ref in `value` by its JSON content; raise if one is missing."""
    if _depth > 16:
        return value
    if is_artifact_ref(value):
        artifact_id = get_artifact_id(value)
        if artifact_store is None:
            raise UnresolvableReference(f"offloaded value {artifact_id} needs an artifact store")
        artifact = artifact_store.load(artifact_id)
        if artifact is None:
            raise UnresolvableReference(f"offloaded value {artifact_id} is missing from the artifact store")
        content_type = str(getattr(getattr(artifact, "metadata", None), "content_type", "") or "")
        if content_type.startswith("text/"):
            return artifact.as_text()
        return resolve_strict(artifact.as_json(), artifact_store=artifact_store, _depth=_depth + 1)
    if isinstance(value, dict):
        return {k: resolve_strict(v, artifact_store=artifact_store, _depth=_depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_strict(v, artifact_store=artifact_store, _depth=_depth + 1) for v in value]
    return value


def normalize_occurrence_output(child: RunState, *, artifact_store: Any = None) -> Dict[str, Any]:
    """`{answer, success, notify, error}` from an occurrence root's terminal state.

    Structure only (no text heuristics): an agent-interface target ends with
    `response`/`success`/`meta` pins; a plain flow ends with its end-node pins
    or `{success, result}`. `notify` is read from the same object as the answer.
    """
    output = resolve_strict(child.output, artifact_store=artifact_store) if child.output is not None else None
    out: Dict[str, Any] = output if isinstance(output, dict) else {}
    payload: Dict[str, Any] = out
    result = out.get("result")
    if isinstance(result, dict) and not any(k in out for k in ("response", "answer", "notify")):
        payload = result

    answer = ""
    for key in ("response", "answer"):
        value = payload.get(key)
        if isinstance(value, str):
            answer = value
            break
    else:
        if isinstance(result, str):
            answer = result

    success = (
        child.status == RunStatus.COMPLETED
        and out.get("success") is not False
        and payload.get("success") is not False
    )
    error = None
    if not success:
        error = str(child.error or payload.get("error") or out.get("error") or child.status.value)
    return {"answer": answer, "success": success, "notify": payload.get("notify", out.get("notify")), "error": error}


def notify_payload(raw: Any, *, title: str, answer: str) -> Optional[Dict[str, str]]:
    """The normalized `{title, body}` for a notify value, or None (quiet)."""
    if raw is True:
        return {"title": title[:NOTIFY_TITLE_MAX], "body": (answer or "")[:NOTIFY_DEFAULT_BODY_MAX]}
    if isinstance(raw, Mapping):
        t = raw.get("title")
        b = raw.get("body")
        t = t.strip() if isinstance(t, str) else ""
        b = b if isinstance(b, str) else ""
        if not t and not b.strip():
            return None
        return {"title": (t or title)[:NOTIFY_TITLE_MAX], "body": b[:NOTIFY_BODY_MAX]}
    return None


def is_interactive_wait(run: RunState) -> bool:
    """A run waiting on a human: USER (not a pause) or EVENT with prompt/choices.

    Controller waits (`automation:<id>:wake`) and pause waits never count.
    """
    if run.status != RunStatus.WAITING or run.waiting is None:
        return False
    waiting = run.waiting
    key = str(waiting.wait_key or "")
    if key.startswith("automation:") and key.endswith(":wake"):
        return False
    if is_paused_vars(run.vars):
        return False
    if waiting.reason == WaitReason.USER:
        details = waiting.details if isinstance(waiting.details, dict) else {}
        return key != f"pause:{run.run_id}" and details.get("kind") != "pause"
    if waiting.reason == WaitReason.EVENT:
        return bool(waiting.prompt) or bool(waiting.choices)
    return False


WAIT_KINDS = ("ask_user", "tool_approval", "event")
# The resume payload each kind accepts (`Runtime.resume(..., payload=...)`).
ANSWER_PAYLOADS = {"ask_user": "{response}", "tool_approval": "{approved: true|false}", "event": "{payload}"}


def wait_kind(run: RunState) -> Optional[str]:
    """The kind of an interactive wait, from the wait record's structure only.

    - `tool_approval`: a tool batch awaiting approval (`details.mode ==
      "approval_required"`, the TOOL_CALLS approval wait);
    - `ask_user`: any other USER wait (a question to a person);
    - `event`: an EVENT wait carrying a prompt or choices.
    None when the run is not waiting on a person.
    """
    if not is_interactive_wait(run):
        return None
    details = run.waiting.details if isinstance(run.waiting.details, dict) else {}
    if details.get("mode") == "approval_required":
        return "tool_approval"
    if run.waiting.reason == WaitReason.USER:
        return "ask_user"
    return "event"


def typed_wait(run: RunState) -> Optional[Dict[str, Any]]:
    """`{run_id, wait_key, kind, reason, prompt?, choices?, details?}` for a run waiting on a person."""
    kind = wait_kind(run)
    if kind is None:
        return None
    waiting = run.waiting
    item: Dict[str, Any] = {
        "run_id": run.run_id,
        "wait_key": waiting.wait_key,
        "kind": kind,
        "reason": waiting.reason.value,
    }
    if waiting.prompt:
        item["prompt"] = waiting.prompt
    if waiting.choices:
        item["choices"] = list(waiting.choices)
    if kind == "tool_approval":
        details = waiting.details or {}
        item["details"] = {"tool_calls": list(details.get("tool_calls") or [])}
    return item


def pending_waits(run_store: Any, automation_id: str, *, limit: int = 20) -> List[Dict[str, Any]]:
    """Waits on a person in the automation's occurrence trees, typed by `kind`.

    Item: `{run_id, wait_key, kind: "ask_user"|"tool_approval"|"event", reason,
    index, prompt?, choices?, details?}`; `details.tool_calls` lists the calls
    a `tool_approval` wait would run. Answer with the kind's payload
    (`ANSWER_PAYLOADS`).
    """
    out: List[Dict[str, Any]] = []
    stack = [str(automation_id)]
    seen = set()
    while stack and len(out) < limit:
        parent = stack.pop()
        for child in run_store.list_children(parent_run_id=parent):
            if child.run_id in seen:
                continue
            seen.add(child.run_id)
            stack.append(child.run_id)
            item = typed_wait(child)
            if item is not None:
                meta = (child.vars.get("_meta") or {}).get("occurrence") or {}
                item["index"] = meta.get("occurrence_index")
                out.append(item)
    return out


def _cursor_seq(cursor: Optional[str]) -> int:
    if cursor is None:
        return 0
    if not isinstance(cursor, str) or not cursor.startswith(ATTENTION_CURSOR_PREFIX):
        raise ValueError(f"invalid attention cursor {cursor!r}")
    try:
        return int(cursor[len(ATTENTION_CURSOR_PREFIX):])
    except ValueError as exc:
        raise ValueError(f"invalid attention cursor {cursor!r}") from exc


def attention_cursor(seq: int) -> str:
    return f"{ATTENTION_CURSOR_PREFIX}{int(seq)}"


def list_attention(
    ledger_store: Any,
    automation_id: str,
    *,
    after_seq: int = 0,
    cursor: Optional[str] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    """Attention items with `seq > max(after_seq, cursor)`, OLDEST first.

    `Page = {items, next_cursor}`; `next_cursor` is the cursor of the last item
    returned when more items follow, else None. A client acknowledges only the
    cursor of the last item it displayed, so items it never showed stay unseen.
    """
    floor = max(int(after_seq or 0), _cursor_seq(cursor))
    limit = max(1, min(int(limit), 500))
    items: List[Dict[str, Any]] = []
    for rec in automation_records(ledger_store, automation_id, "automation.completed"):
        p = rec["payload"]
        att = p.get("attention")
        if not isinstance(att, dict) or int(att.get("seq") or 0) <= floor:
            continue
        item = {
            "kind": att.get("kind"),
            "automation_id": automation_id,
            "run_id": p.get("run_id"),
            "index": p.get("index"),
            "at": p.get("finished_at") or p.get("at"),
            "title": att.get("title"),
            "seq": int(att["seq"]),
            "cursor": attention_cursor(int(att["seq"])),
        }
        if att.get("body"):
            item["body"] = att["body"]
        items.append(item)
    items.sort(key=lambda it: it["seq"])
    page = items[:limit]
    return {"items": page, "next_cursor": page[-1]["cursor"] if len(items) > limit else None}


__all__ = [
    "ANSWER_PAYLOADS",
    "ATTENTION_CURSOR_PREFIX",
    "WAIT_KINDS",
    "UnresolvableReference",
    "attention_cursor",
    "is_interactive_wait",
    "list_attention",
    "normalize_occurrence_output",
    "notify_payload",
    "pending_waits",
    "resolve_strict",
    "typed_wait",
    "wait_kind",
]
