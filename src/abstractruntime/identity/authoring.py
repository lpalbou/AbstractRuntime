"""Every memory has an authored title and summary (operator ruling 2026-11-09: "the title + summary must ALWAYS exist
for any memory. those are mandatory metadata of all memories" - no fallback; failures are logged).

Who authors:

- an ELECTED record (interest, lesson, realization, a diary entry's gist): the entity, in its own block (its
  `title:` line, its words as the summary). What the block lacks, the `summaries` capability model authors from the
  body - authoring, never a cut;
- every other record the runtime forms (an episode, a session's look-back summary): the `summaries` capability
  model, in one structured call right after the record forms.

When authoring cannot happen (no capability client, an unreachable endpoint, a reply without the fields), the record
is formed PENDING through AbstractMemory's API (`MemoryRecordInput(pending=("title",) | ("summary",) | both)`: the
missing title is None, a provisional summary stands in for the missing one), which queues its `title.author` /
`digest.author` task in the same call, and one line goes to the authoring-failures file (`title_failures`). The
drainer (`drain_authoring`) runs those two task kinds ONLY - at the own-time loop's tick boundary, at a session's
close and inside a sleep's maintenance - and completes them through `retitle` / `rewrite_digest` (which clear the
marker). A capability-model author names its source: the authoring call's record is kept as a home artifact.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "AUTHORING_CAPABILITY",
    "AUTHORING_TASK_KINDS",
    "AUTHORING_CAPABILITY_KEY",
    "AUTHORING_RESPONSE_SCHEMA",
    "AuthoringFailed",
    "AuthoringUnavailable",
    "CapabilityUnrouted",
    "authoring_llm_call_payload",
    "parse_authored_result",
    "author_metadata",
    "authored_write",
    "declare_missing_metadata",
    "drain_authoring",
]

AUTHORING_CAPABILITY = "summaries"
AUTHORING_TASK_KINDS = ("title.author", "digest.author")
AUTHORING_TEMPLATE_ID = "authoring.metadata"
AUTHORING_TEMPLATE_VERSION = "2026-11-09"
# An unreachable endpoint defers its task this long (no attempt spent; memory's `release(outcome="defer")`).
DEFER_SECONDS = 600


class AuthoringUnavailable(Exception):
    """The capability model cannot be reached or built now (the task waits, no attempt spent)."""


class AuthoringFailed(Exception):
    """The capability model answered without a usable title or summary (an attempt spent)."""


_SYSTEM = (
    "You write the metadata of one memory: a TITLE (a short, complete, readable name for it - one line) and a "
    "SUMMARY (one to three plain sentences saying what it holds). Use only what the memory says; invent nothing."
)


def _prompt(kind: str, body: str) -> str:
    return f"The memory (kind: {kind}):\n\n{body}\n\nWrite its title and its summary."


# THE DURABLE LANE (a visit workflow cannot call a model inside a node): the authoring call is an LLM_CALL effect
# tagged with the capability. Payload shape (the gateway routes `payload.capability` to its capability client):
#   {"capability": "summaries", "system_prompt": <_SYSTEM>, "messages": [{"role": "user", "content": <prompt>}],
#    "response_schema": AUTHORING_RESPONSE_SCHEMA, "response_schema_name": "AuthoredMetadata",
#    "params": {"thinking": "off"}, "_absorb_failure": true}
# The handler that serves it must ECHO the tag in its result (`result.capability == "summaries"`): an answer
# without the echo came from a handler that does not serve the capability (the mind's), and the formation treats it
# as a wiring failure, loudly (logged error + an authoring-failure line, reason capability_unrouted), never as an
# authored title. The fields come from `result.data` (structured output) or, strictly, from `result.content` as JSON.
AUTHORING_CAPABILITY_KEY = "capability"
AUTHORING_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {"title": {"type": "string"}, "summary": {"type": "string"}},
    "required": ["title", "summary"],
    "additionalProperties": False,
}


def authoring_llm_call_payload(kind: str, body: str) -> Dict[str, Any]:
    """The LLM_CALL payload of one authoring call (see the shape above)."""
    return {
        AUTHORING_CAPABILITY_KEY: AUTHORING_CAPABILITY,
        "system_prompt": _SYSTEM + " Answer with exactly one JSON object: {\"title\": \"...\", \"summary\": \"...\"}.",
        "messages": [{"role": "user", "content": _prompt(kind, body)}],
        "response_schema": dict(AUTHORING_RESPONSE_SCHEMA),
        "response_schema_name": "AuthoredMetadata",
        "params": {"thinking": "off"},
        "_absorb_failure": True,
    }


class CapabilityUnrouted(AuthoringFailed):
    """The tagged call was answered without the capability echo: no handler serves the capability."""


def parse_authored_result(result: Any) -> Dict[str, Any]:
    """The {title, summary} an authoring LLM_CALL returned (raises AuthoringUnavailable for a failed call,
    CapabilityUnrouted without the echo, AuthoringFailed for missing or unusable fields)."""
    from .title_failures import TITLE_MAX_CHARS

    if not isinstance(result, dict):
        raise AuthoringUnavailable("the authoring call returned nothing")
    if result.get("capability_unrouted"):
        raise CapabilityUnrouted(
            f"no handler serves LLM_CALL {AUTHORING_CAPABILITY_KEY}={result['capability_unrouted']!r} on this host"
        )
    if result.get("absorbed_failure") or result.get("ok") is False:
        raise AuthoringUnavailable(str(result.get("absorbed_failure") or "the authoring call failed"))
    if result.get(AUTHORING_CAPABILITY_KEY) != AUTHORING_CAPABILITY:
        raise CapabilityUnrouted(
            f"the LLM_CALL tagged {AUTHORING_CAPABILITY_KEY}={AUTHORING_CAPABILITY!r} was answered without the "
            "capability echo - no handler serves that capability on this host"
        )
    data = result.get("data")
    if not isinstance(data, dict):
        try:
            data = json.loads(str(result.get("content") or ""))
        except ValueError as e:
            raise AuthoringFailed("the reply was not the structured title and summary") from e
    title = data.get("title") if isinstance(data, dict) else None
    summary = data.get("summary") if isinstance(data, dict) else None
    if not isinstance(title, str) or not title.strip() or "\n" in title.strip() or len(title.strip()) > TITLE_MAX_CHARS:
        raise AuthoringFailed("the reply carried no usable title")
    if not isinstance(summary, str) or not summary.strip():
        raise AuthoringFailed("the reply carried no summary")
    return {"title": title.strip(), "summary": summary.strip(), "model": result.get("model")}


def _model_class():
    from pydantic import BaseModel

    class AuthoredMetadata(BaseModel):
        title: str
        summary: str

    return AuthoredMetadata


def _unreachable(exc: BaseException) -> bool:
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    name = type(exc).__name__.lower()
    return any(word in name for word in ("connect", "timeout", "unavailable", "substrateunset"))


def author_metadata(llm: Any, *, kind: str, body: str) -> Dict[str, Any]:
    """One structured call: `{"title": str, "summary": str, "raw": <what the model returned, as JSON text>}`.
    Raises AuthoringUnavailable (unreachable endpoint) or AuthoringFailed (no usable fields). Nothing is repaired:
    a title past TITLE_MAX_CHARS, a multi-line title or an empty field fails."""
    from .title_failures import TITLE_MAX_CHARS

    if llm is None:
        raise AuthoringUnavailable("no summaries capability model in this process")
    model_cls = _model_class()
    try:
        resp = llm.generate(prompt=_prompt(kind, body), system_prompt=_SYSTEM, response_model=model_cls)
    except Exception as e:  # noqa: BLE001 - classified below, never swallowed
        if _unreachable(e):
            raise AuthoringUnavailable(f"{type(e).__name__}: {e}") from e
        raise AuthoringFailed(f"the model call failed: {type(e).__name__}: {e}") from e
    if hasattr(resp, "model_dump"):
        data = resp.model_dump()
    elif isinstance(resp, dict):
        data = dict(resp)
    else:
        content = getattr(resp, "content", None)
        try:
            data = json.loads(str(content or ""))
        except ValueError as e:
            raise AuthoringFailed("the reply was not the structured title and summary") from e
    title = data.get("title") if isinstance(data, dict) else None
    summary = data.get("summary") if isinstance(data, dict) else None
    if not isinstance(title, str) or not title.strip() or "\n" in title.strip() or len(title.strip()) > TITLE_MAX_CHARS:
        raise AuthoringFailed("the reply carried no usable title")
    if not isinstance(summary, str) or not summary.strip():
        raise AuthoringFailed("the reply carried no summary")
    return {"title": title.strip(), "summary": summary.strip(), "raw": json.dumps(data, ensure_ascii=False)}


def _reader_of(llm: Any) -> Dict[str, Any]:
    info = getattr(llm, "model_info", None)
    if isinstance(info, dict) and info.get("provider") and info.get("model"):
        return {"provider": str(info["provider"]), "model": str(info["model"]), "thinking": info.get("thinking")}
    return {
        "provider": str(getattr(llm, "provider", "") or getattr(llm, "provider_name", "") or "capability"),
        "model": str(getattr(llm, "model", "") or "summaries"),
        "thinking": None,
    }


def _source_artifact(artifacts: Any, *, record_id: str, kind: str, body: str, authored: Dict[str, Any],
                     reader: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The authoring call kept as a home artifact: the capability-model author's source (ADR 0003)."""
    if artifacts is None or not callable(getattr(artifacts, "store_text", None)):
        raise AuthoringFailed("no artifact store to keep the authoring call (a model author must name its source)")
    meta = artifacts.store_text(
        json.dumps({"template": AUTHORING_TEMPLATE_ID, "version": AUTHORING_TEMPLATE_VERSION, "record_id": record_id,
                    "kind": kind, "prompt": _prompt(kind, body), "reply": authored.get("raw"), "reader": reader},
                   ensure_ascii=False),
        content_type="application/json",
        tags={"kind": "memory_authoring", "record_id": record_id},
    )
    artifact_id = getattr(meta, "artifact_id", None) or (meta.get("artifact_id") if isinstance(meta, dict) else None)
    if not artifact_id:
        raise AuthoringFailed("the authoring call could not be kept as an artifact")
    return [{"kind": "artifact", "artifact_id": str(artifact_id)}]


def _live_task(ms: Any, record_id: str, kind: str) -> Optional[Dict[str, Any]]:
    tasks = getattr(ms, "tasks", None)
    if tasks is None:
        return None
    for t in tasks.tasks(kind=(kind,), target_id=record_id):
        if t.get("status") in ("pending", "deferred", "leased"):
            return t
    return None


def authored_write(ms: Any, artifacts: Any, *, record_id: str, kind: str, body: str, authored: Dict[str, Any],
                   reader: Dict[str, Any], need: Sequence[str], task_ids: Optional[Dict[str, str]] = None,
                   expected: Optional[Dict[str, int]] = None, cause: str = "formation") -> Dict[str, Any]:
    """Write what the capability model authored onto a pending record: `retitle` for the title, `rewrite_digest` for
    the summary, each completing its live task (named, or found by target) and clearing the marker."""
    sources = _source_artifact(artifacts, record_id=record_id, kind=kind, body=body, authored=authored, reader=reader)
    author = {"kind": "capability-model", "id": AUTHORING_CAPABILITY}
    out: Dict[str, Any] = {}
    if "title" in need:
        tid = (task_ids or {}).get("title") or (_live_task(ms, record_id, "title.author") or {}).get("task_id")
        version = int((expected or {}).get("title") or 0)
        out["title"] = ms.retitle(record_id, title=authored["title"], author=author,
                                  cause={"cause": cause, "pending": "title"}, expected_version=version,
                                  reader=reader, sources=sources, task_id=tid)
    if "summary" in need:
        tid = (task_ids or {}).get("summary") or (_live_task(ms, record_id, "digest.author") or {}).get("task_id")
        version = int((expected or {}).get("summary") or 0)
        out["summary"] = ms.rewrite_digest(record_id, text=authored["summary"], author=author,
                                           cause={"cause": cause, "pending": "summary"}, expected_version=version,
                                           reader=reader, sources=sources, task_id=tid)
    for field, res in out.items():
        if not isinstance(res, dict) or res.get("outcome") not in ("applied", "unchanged"):
            raise AuthoringFailed(f"{field} write refused: {(res or {}).get('reason')}")
    return out


def drain_authoring(
    ms: Any,
    *,
    llm_for_capability: Optional[Callable[[str], Any]],
    artifacts: Any,
    home_dir: Any,
    entity: str,
    holder: str,
    cap: int,
    window: str = "turn",
    diary_text: Optional[Callable[[str], Optional[str]]] = None,
    out: Callable[[str], None] = lambda _s: None,
) -> Dict[str, Any]:
    """Run up to `cap` pending `title.author` / `digest.author` tasks (and no other kind): select, claim, read the
    target, one structured call, write through retitle / rewrite_digest (which settle the task). An unreachable
    endpoint (or no client) defers without spending an attempt; a reply without the fields releases for retry (an
    attempt spent) and logs one authoring failure. Run it under the home's writer lease. Returns
    {authored, deferred, failed, considered}."""
    from .title_failures import record_authoring_failure

    receipt = {"authored": 0, "deferred": 0, "failed": 0, "considered": 0}
    tasks = getattr(ms, "tasks", None)
    if tasks is None or int(cap) <= 0:
        return receipt
    selection = tasks.select_for_window(10_000.0, window=window, kinds=list(AUTHORING_TASK_KINDS),
                                        hard_cap=int(cap))
    picked = list(selection.get("tasks") or [])
    if not picked:
        return receipt
    claimed = tasks.claim([t["task_id"] for t in picked], holder=holder, window=window).get("claimed") or []
    llm = None
    for t in claimed:
        receipt["considered"] += 1
        need = ["title"] if t["kind"] == "title.author" else ["summary"]
        try:
            packet = ms.task_packet(t["task_id"])
        except Exception as e:  # noqa: BLE001
            tasks.release(t["task_id"], holder=holder, outcome="retry", reason=f"packet unreadable: {e}")
            receipt["failed"] += 1
            continue
        if packet.get("target_state") != "live":
            tasks.release(t["task_id"], holder=holder, outcome="skip", reason=f"target {packet.get('target_state')}")
            continue
        record = packet.get("record") or {}
        kind = str(record.get("kind") or "memory")
        body = str(record.get("digest") or "")
        if kind == "diary":
            body = str((diary_text(str(record.get("entry_id") or "")) if diary_text else "") or "")
        if not body.strip():
            tasks.release(t["task_id"], holder=holder, outcome="retry", reason="no words to author from")
            receipt["failed"] += 1
            continue
        try:
            if llm is None:
                if llm_for_capability is None:
                    raise AuthoringUnavailable("no summaries capability model in this process")
                try:
                    llm = llm_for_capability(AUTHORING_CAPABILITY)
                except Exception as e:  # noqa: BLE001 - no model resolves / builds: the task waits
                    raise AuthoringUnavailable(f"{type(e).__name__}: {e}") from e
            authored = author_metadata(llm, kind=kind, body=body)
            authored_write(ms, artifacts, record_id=str(t["target_id"]), kind=kind, body=body, authored=authored,
                           reader=_reader_of(llm), need=need, task_ids={need[0]: t["task_id"]},
                           expected={need[0]: int(packet.get("expected_version") or 0)},
                           cause="repair" if str((t.get("cause") or {}).get("cause") if isinstance(t.get("cause"), dict)
                                                 else t.get("cause") or "") == "legacy_missing" else "formation")
            receipt["authored"] += 1
        except AuthoringUnavailable as e:
            until = (datetime.now(timezone.utc) + timedelta(seconds=DEFER_SECONDS)).isoformat()
            tasks.release(t["task_id"], holder=holder, outcome="defer", reason="endpoint unavailable",
                          until=until, detail={"error": str(e)})
            receipt["deferred"] += 1
        except AuthoringFailed as e:
            tasks.release(t["task_id"], holder=holder, outcome="retry", reason=str(e))
            receipt["failed"] += 1
            record_authoring_failure(
                home_dir, entity=entity, record_id=str(t["target_id"]), kind=kind, raw_block=body,
                template_id=AUTHORING_TEMPLATE_ID, provider=None, model=None, reason="malformed",
                missing=need[0], extra={"task_id": t["task_id"], "error": str(e)}, out=out,
            )
    return receipt


def declare_missing_metadata(ms: Any, *, scopes: Sequence[Tuple[str, str]], out: Callable[[str], None] = lambda _s: None
                             ) -> Dict[str, Any]:
    """The sleep's scan (operator 2026-11-09): a record that LACKS a title or a summary (a legacy record formed
    before titles were mandatory, or one stored with the runtime's old placeholder) and carries no pending marker
    gets its authoring task through memory's queue (`tasks.enqueue`, the basis memory's own formation uses, so the
    task is idempotent). Nothing is authored here and nothing is cut. Returns {declared, scanned}."""
    from abstractmemory import TripleQuery
    from abstractmemory.memory_tasks import FORMATION_PRIORITY

    from .title_failures import is_placeholder_title

    receipt = {"declared": 0, "scanned": 0}
    tasks = getattr(ms, "tasks", None)
    if tasks is None:
        return receipt
    for scope, owner in scopes:
        for a in ms.store.query(TripleQuery(scope=scope, owner_id=owner or None, predicate="dcterms:abstract",
                                            limit=0)):
            attrs = a.attributes if isinstance(a.attributes, dict) else {}
            if not attrs.get("record_kind") or isinstance(attrs.get("pending"), dict):
                continue
            receipt["scanned"] += 1
            title = attrs.get("title")
            lacks = []
            if not (isinstance(title, str) and title.strip()) or is_placeholder_title(title) or attrs.get("title_missing"):
                lacks.append(("title.author", "title"))
            if not str(a.object or "").strip():
                lacks.append(("digest.author", "summary"))
            for task_kind, fld in lacks:
                res = tasks.enqueue(
                    task_kind, str(a.subject), scope=scope, owner_id=owner, assertion_id=str(a.assertion_id or ""),
                    basis=f"{a.assertion_id}@pending-{fld}", basis_version=0, strength=1.0,
                    cause={"cause": "legacy_missing", "pending": fld}, priority=FORMATION_PRIORITY,
                )
                if res.get("enqueued"):
                    receipt["declared"] += 1
    if receipt["declared"]:
        out(f"(sleep: {receipt['declared']} record(s) lacking a title or summary were given their authoring task)")
    return receipt
