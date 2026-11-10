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
    "AuthoringCallFailed",
    "AuthoringFailed",
    "AuthoringRefused",
    "AuthoringUnavailable",
    "CapabilityUnrouted",
    "authoring_llm_call_payload",
    "parse_authored_result",
    "author_metadata",
    "authoring_failure_fields",
    "authored_write",
    "capability_formation_author",
    "handler_formation_author",
    "log_pass_authoring_failures",
    "declare_missing_metadata",
    "diary_authoring_payload",
    "drain_authoring",
    "rearm_exhausted_authoring",
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


class AuthoringCallFailed(AuthoringFailed):
    """The capability client RAISED (and the endpoint is not merely unreachable): the call failed before any reply
    existed - never a malformed reply. Its class and message go to the failure line (an attempt spent)."""


class AuthoringRefused(AuthoringFailed):
    """Memory refused the authored write (retitle / rewrite_digest answered neither applied nor unchanged)."""


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


def diary_authoring_payload(result: Any, *, need_title: bool, artifacts: Any, body: str) -> Dict[str, Any]:
    """The DIARY_WRITE payload keys for a public entry's metadata from its authoring LLM_CALL result (Codex review 2
    #10, review 3 #9): `authored_metadata` {summary, title (when he elected no gist), reader, sources (the call kept
    as a home artifact)}, or `authoring_failure` (reason, class, message, the capability's model when its result named
    it) - the projection then forms the missing fields pending and writes the failure line."""
    from .chat import TurnCancelled

    res = result if isinstance(result, dict) else {}
    try:
        parsed = parse_authored_result(result)
        reader = {"provider": str(res.get("provider") or "capability"),
                  "model": str(parsed.get("model") or res.get("model") or AUTHORING_CAPABILITY), "thinking": None}
        authored = {**parsed, "raw": json.dumps({"title": parsed["title"], "summary": parsed["summary"]})}
        sources = _source_artifact(artifacts, record_id="diary", kind="diary", body=body, authored=authored,
                                   reader=reader)
    except TurnCancelled:
        raise  # the operator's Stop (the fenced artifact store refused the call's record): no failure, nothing kept
    except Exception as e:  # noqa: BLE001
        why = authoring_failure_fields(e)
        return {"authoring_failure": {**why, "provider": res.get("provider"), "model": res.get("model"),
                                      "error": f"{type(e).__name__}: {e}"}}
    meta: Dict[str, Any] = {"summary": parsed["summary"], "reader": reader, "sources": sources}
    if need_title:
        meta["title"] = parsed["title"]
    return {"authored_metadata": meta}


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

    from .chat import TurnCancelled

    if llm is None:
        raise AuthoringUnavailable("no summaries capability model in this process")
    model_cls = _model_class()
    try:
        resp = llm.generate(prompt=_prompt(kind, body), system_prompt=_SYSTEM, response_model=model_cls)
    except TurnCancelled:
        raise  # the operator's Stop (a fenced client refused the call): never an authoring failure
    except Exception as e:  # noqa: BLE001 - classified below, never swallowed
        if _unreachable(e):
            raise AuthoringUnavailable(f"{type(e).__name__}: {e}") from e
        from pydantic import ValidationError

        if isinstance(e, ValidationError):  # the reply came back and did not fit {title, summary}: malformed
            raise AuthoringFailed(f"the reply did not fit the title and summary: {type(e).__name__}: {e}") from e
        raise AuthoringCallFailed(f"the model call failed: {type(e).__name__}: {e}") from e
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


def authoring_failure_fields(exc: BaseException, llm: Any = None) -> Dict[str, Any]:
    """What a failure line says about a failed authoring step: the reason word, the provider/model of the capability
    client used (None when none could be built), and the class and message of what actually failed - the client's
    own exception when it raised, never the wrapper. reason: unavailable (unreachable endpoint / no client, no
    attempt spent) | client_error (the client raised) | capability_unrouted | memory_refused | malformed (a reply
    that truly failed to parse or lacked a field) | error (anything else)."""
    root = exc.__cause__ if exc.__cause__ is not None else exc
    if isinstance(exc, AuthoringUnavailable):
        reason = "unavailable"
    elif isinstance(exc, AuthoringCallFailed):
        reason = "client_error"
    elif isinstance(exc, CapabilityUnrouted):
        reason = "capability_unrouted"
    elif isinstance(exc, AuthoringRefused):
        reason = "memory_refused"
    elif isinstance(exc, AuthoringFailed):
        reason = "malformed"
    else:
        reason = "error"
    reader = _reader_of(llm) if llm is not None else {}
    return {"reason": reason, "provider": reader.get("provider"), "model": reader.get("model"),
            "error_class": type(root).__name__, "error_message": str(root)}


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
            raise AuthoringRefused(f"{field} write refused: {(res or {}).get('reason')}")
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
    should_stop: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """Run up to `cap` pending `title.author` / `digest.author` tasks (and no other kind): select, claim, read the
    target, one structured call, write through retitle / rewrite_digest (which settle the task). An unreachable
    endpoint (or no client) defers without spending an attempt; a reply without the fields releases for retry (an
    attempt spent) and logs one authoring failure. Run it under the home's writer lease. Returns
    {authored, deferred, failed, considered}."""
    from .title_failures import record_authoring_failure

    receipt = {"authored": 0, "deferred": 0, "failed": 0, "considered": 0, "rearmed": 0}
    tasks = getattr(ms, "tasks", None)
    if tasks is None or int(cap) <= 0:
        return receipt
    if should_stop is not None and should_stop():  # cancelled before anything: no re-arm, no claim, no call
        receipt["cancelled"] = 0
        return receipt
    # A pending record whose every authoring task ran out (failed after its attempts, skipped, superseded) would
    # stay pending forever: memory re-arms it first, so this pass can author it.
    receipt["rearmed"] = int(rearm_exhausted_authoring(ms, out=out).get("rearmed") or 0)
    selection = tasks.select_for_window(10_000.0, window=window, kinds=list(AUTHORING_TASK_KINDS),
                                        hard_cap=int(cap))
    picked = list(selection.get("tasks") or [])
    if not picked:
        return receipt
    claimed = tasks.claim([t["task_id"] for t in picked], holder=holder, window=window).get("claimed") or []
    llm = None

    def _cancelled(left: List[Dict[str, Any]]) -> bool:
        # CANCELLATION (Codex review 2 #2): the host's Stop - no further model call or write; the tasks still held go
        # back without spending an attempt (defer to now).
        if should_stop is None or not should_stop():
            return False
        now = datetime.now(timezone.utc).isoformat()
        for held in left:
            tasks.release(held["task_id"], holder=holder, outcome="defer", reason="cancelled", until=now)
        receipt["cancelled"] = len(left)
        out(f"(authoring: cancelled - {len(left)} task(s) handed back unspent)")
        return True

    def _failed(t: Dict[str, Any], *, kind: str, body: str, need: List[str], outcome: str, reason: str,
                exc: Optional[BaseException], message: str = "") -> None:
        """EVERY failed attempt leaves its failure line (Codex review 3 #10): retry and defer alike, with the
        record id, the outcome, the original exception's class and message, and the capability model when one was
        built."""
        why = authoring_failure_fields(exc, llm) if exc is not None else {
            "reason": reason, "provider": None, "model": None, "error_class": None, "error_message": message}
        reader = _reader_of(llm) if llm is not None else {}
        record_authoring_failure(
            home_dir, entity=entity, record_id=str(t["target_id"]), kind=kind, raw_block=body,
            template_id=AUTHORING_TEMPLATE_ID, provider=why["provider"] or reader.get("provider"),
            model=why["model"] or reader.get("model"), reason=reason or why["reason"], missing=need[0], out=out,
            extra={"task_id": t["task_id"], "outcome": outcome, "error": str(exc) if exc is not None else message,
                   "error_class": why["error_class"], "error_message": why["error_message"] or message,
                   "capability": AUTHORING_CAPABILITY},
        )

    for n, t in enumerate(claimed):
        if _cancelled(claimed[n:]):
            break
        receipt["considered"] += 1
        need = ["title"] if t["kind"] == "title.author" else ["summary"]
        try:
            packet = ms.task_packet(t["task_id"])
        except Exception as e:  # noqa: BLE001
            tasks.release(t["task_id"], holder=holder, outcome="retry", reason=f"packet unreadable: {e}")
            receipt["failed"] += 1
            _failed(t, kind="memory", body="", need=need, outcome="retry", reason="packet_unreadable", exc=e)
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
            _failed(t, kind=kind, body=body, need=need, outcome="retry", reason="no_source", exc=None,
                    message="no words to author from (the record's body / the diary entry's text is empty)")
            continue
        try:
            if llm is None:
                if llm_for_capability is None:
                    raise AuthoringUnavailable("no summaries capability model in this process")
                try:
                    llm = llm_for_capability(AUTHORING_CAPABILITY)
                except Exception as e:  # noqa: BLE001 - no model resolves / builds: the task waits
                    from .chat import TurnCancelled

                    if isinstance(e, TurnCancelled):
                        raise  # the operator's Stop while the client was resolved: never a deferral
                    raise AuthoringUnavailable(f"{type(e).__name__}: {e}") from e
            if _cancelled(claimed[n:]):  # immediately before the model call (Codex review 4 #3)
                break
            authored = author_metadata(llm, kind=kind, body=body)
            if _cancelled(claimed[n:]):  # the model answered after Stop: nothing is written
                break
            authored_write(ms, artifacts, record_id=str(t["target_id"]), kind=kind, body=body, authored=authored,
                           reader=_reader_of(llm), need=need, task_ids={need[0]: t["task_id"]},
                           expected={need[0]: int(packet.get("expected_version") or 0)},
                           cause=_write_cause(t))
            receipt["authored"] += 1
        except AuthoringUnavailable as e:
            until = (datetime.now(timezone.utc) + timedelta(seconds=DEFER_SECONDS)).isoformat()
            tasks.release(t["task_id"], holder=holder, outcome="defer", reason="endpoint unavailable",
                          until=until, detail={"error": str(e)})
            receipt["deferred"] += 1
            _failed(t, kind=kind, body=body, need=need, outcome="defer", reason="unavailable", exc=e)
        except AuthoringFailed as e:
            tasks.release(t["task_id"], holder=holder, outcome="retry", reason=str(e))
            receipt["failed"] += 1
            _failed(t, kind=kind, body=body, need=need, outcome="retry", reason="", exc=e)
    return receipt


def _write_cause(task: Dict[str, Any]) -> str:
    """The cause the authored write records: "repair" for a task the sleep's repair declared (memory's
    declare_pending stamps its tasks {cause: "repair", pending, declared: True}; a task 7d2419e enqueued directly
    carries cause legacy_missing), "formation" for a record formed pending."""
    cause = task.get("cause")
    word = str(cause.get("cause") or "") if isinstance(cause, dict) else str(cause or "")
    declared = isinstance(cause, dict) and bool(cause.get("declared"))
    return "repair" if declared or word in ("repair", "legacy_missing") else "formation"


def _public_diary_projection(kind: str, a: Any) -> bool:
    """A PUBLIC book projection (AbstractMemory 71de051): its summary is authored like any record's - a legacy one's
    declared pending too (AbstractMemory 8235daf) - while a private one (act-only) and an owner-direct one are not."""
    attrs = a.attributes if isinstance(a.attributes, dict) else {}
    prov = a.provenance if isinstance(getattr(a, "provenance", None), dict) else {}
    return kind == "diary" and not attrs.get("private") and str(prov.get("source") or "") == "diary-projection"


def _legacy_diary_title(kind: str, title: Any, a: Any) -> bool:
    """A public diary projection formed before 7dd32e1 carries the runtime's own act-frame template as its title
    ("Diary entry (<kind>) — <date>[: ...]"): the runtime's label format, never his words."""
    return _public_diary_projection(kind, a) and str(title or "").startswith("Diary entry (")


def _legacy_diary_summary(kind: str, a: Any) -> bool:
    """A public diary projection whose summary was never authored (formed before Codex review 3 #9: his gist or his
    first line stood in, stamped by memory as the owner's own reading): its digest's reading names no capability
    model or operator author."""
    attrs = a.attributes if isinstance(a.attributes, dict) else {}
    stamp = attrs.get("understanding") if isinstance(attrs.get("understanding"), dict) else {}
    author = stamp.get("author") if isinstance(stamp.get("author"), dict) else {}
    return _public_diary_projection(kind, a) and str(author.get("kind") or "") not in ("capability-model", "operator")


#: The runtime's own session look-back title template before 4eefddb formed it pending: f"session reflection:
#: {session_id}" (ChatSession's reflection and the visit lane's close reflection, kind summary). A label format the
#: runtime wrote, never words anyone authored; matched by its prefix only.
RUNTIME_REFLECTION_TITLE_PREFIX = "session reflection: "


def _legacy_mechanical_title(kind: str, title: Any, attrs: Dict[str, Any]) -> bool:
    """A card or dream stored before memory formed them pending (4f3a77e) carries memory's own mechanical label:
    "World model: <subject>" with no authored title provenance, or "Dream: ..."; a session summary stored before the
    runtime formed it pending (4eefddb) carries the runtime's own template "session reflection: <session id>" with no
    authored title provenance. Label formats of the code that wrote them, never text read for meaning."""
    text = str(title or "")
    if kind == "world_model":
        return text.startswith("World model: ") and not attrs.get("title_provenance")
    if kind == "dream":
        return text.startswith("Dream: ")
    if kind == "summary":
        return text.startswith(RUNTIME_REFLECTION_TITLE_PREFIX) and not _authored_title_stamp(attrs)
    return False


#: The kinds the runtime and memory FORM from their own bookkeeping and now author at formation (an exchange's
#: episode; a session's look-back summary, a floored reflection, memory's day / dedup summary). A world-model card and
#: a dream keep their own label rules (above); an elected kind (interest, lesson, realization, diary) is his words.
FORMED_KINDS = frozenset({"episode", "summary"})
#: Who wrote a reading that nobody authored, as memory derives it (`reading_provenance`): a machine (a `mechanical*`
#: digest_method or a mechanical formation stamp) or nobody recorded (no label at all: the Entity app's "who wrote it
#: was not recorded").
UNAUTHORED_READING_KINDS = frozenset({"mechanical", "unrecorded"})


def _authored_title_stamp(attrs: Dict[str, Any]) -> bool:
    """The title carries an authored stamp: the formation step authored it (`metadata_authored.fields` names it) or
    memory recorded who wrote it (`title_provenance` whose author is not mechanical - memory's own reading,
    world_model._authored_title; a declaration's stamp is mechanical)."""
    meta = attrs.get("metadata_authored")
    if isinstance(meta, dict) and "title" in [str(f) for f in (meta.get("fields") or ())]:
        return True
    prov = attrs.get("title_provenance")
    if not isinstance(prov, dict):
        return False
    author = prov.get("author") if isinstance(prov.get("author"), dict) else {}
    return str(author.get("kind") or "") != "mechanical"


def _formed_never_authored(kind: str, a: Any, attrs: Dict[str, Any]) -> bool:
    """A record the runtime or memory formed mechanically before formation authored it (operator 2026-11-09), read BY
    STAMP, never by its text: its kind is formed (FORMED_KINDS), it carries NO authored provenance (no
    `metadata_authored`, no authored `title_provenance`, its current reading not written by a capability model, the
    operator or the entity: memory's `reading_provenance` author is mechanical - a mechanical-v1 / -v2 / -floor-v1 /
    -flow-v1 / -dedup-v1 / -close-v1 digest_method or stamp - or unrecorded). Both its title and its summary are then
    declared pending. A ripple's consolidation record is never taken (memory refuses consolidation_record)."""
    from abstractmemory.understanding import reading_provenance

    if kind not in FORMED_KINDS or attrs.get("summary_role") == "re-understanding":
        return False
    if attrs.get("metadata_authored") or _authored_title_stamp(attrs):
        return False
    author = reading_provenance(a).get("author") or {}
    return str(author.get("kind") or "") in UNAUTHORED_READING_KINDS


def _repair_fields(a: Any) -> List[str]:
    """The fields the sleep's repair declares pending on one stored, unclosed record (pure read;
    `declare_missing_metadata`'s selection, the legacy public diary summary aside): [] for a non-record or a record
    already pending."""
    from .title_failures import is_placeholder_title

    attrs = a.attributes if isinstance(a.attributes, dict) else {}
    if not attrs.get("record_kind") or isinstance(attrs.get("pending"), dict):
        return []
    title = attrs.get("title")
    kind = str(attrs.get("record_kind") or "")
    method = str(attrs.get("digest_method") or "")
    fields: List[str] = []
    if (not (isinstance(title, str) and title.strip()) or is_placeholder_title(title)
            or attrs.get("title_missing") or _legacy_mechanical_title(kind, title, attrs)
            or _legacy_diary_title(kind, title, a)):
        fields.append("title")
    if not str(a.object or "").strip() or _legacy_mechanical_summary(kind, title, method):
        fields.append("summary")
    if _formed_never_authored(kind, a, attrs):
        fields.extend(f for f in ("title", "summary") if f not in fields)
    return [f for f in ("title", "summary") if f in fields]


def _legacy_mechanical_summary(kind: str, title: Any, method: str) -> bool:
    """A legacy card whose body is memory's mechanical card (or carries a mechanical delta), or a legacy dream (its
    body is the night's mechanical composition)."""
    if kind == "world_model":
        try:
            from abstractmemory.understanding import MECHANICAL_CARD_METHODS
        except ImportError:  # pragma: no cover - engine skew
            MECHANICAL_CARD_METHODS = frozenset({"mechanical-card-v2"})
        return method in MECHANICAL_CARD_METHODS or method.endswith("+delta")
    if kind == "dream":
        return str(title or "").startswith("Dream: ")
    return False


def declare_missing_metadata(ms: Any, *, scopes: Sequence[Tuple[str, str]], out: Callable[[str], None] = lambda _s: None
                             ) -> Dict[str, Any]:
    """The sleep's scan (operator 2026-11-09): a record that LACKS a title or a summary (a legacy record formed
    before titles were mandatory, one stored with the runtime's old placeholder, a legacy card "World model:
    <subject>" / mechanical-card-v2 or dream "Dream: ...", a session summary titled with the runtime's template
    "session reflection: <id>") and carries no pending marker is DECLARED PENDING through memory's `declare_pending`
    (AbstractMemory 11687df): the same marker and title.author / digest.author tasks a record formed pending gets, so
    the drainer authors its title and its summary through retitle / rewrite_digest - cards and dreams included. So is
    an episode or a summary the runtime or memory formed MECHANICALLY before formation authored it (selected by
    stamp, `_formed_never_authored`: its reading mechanical or unrecorded, no authored provenance): both fields. A
    declared title is stored None until authored (the old one stays in its history); a declared summary keeps the
    stored text as its provisional version 0. Nothing is authored here and nothing is cut: the drainer authors under
    its per-pass budget, so a home with many such records is repaired over several nights. A CLOSED record (memory
    refuses `closed`; its replacement is the live one) is skipped, counted in `closed`. A refusal (e.g.
    protected_kind: the summary of his own act) is logged loudly and counted. Memory may also answer outcome
    "rearmed" (or "applied" beside new fields) with the fields whose task it REPAIRED (AbstractMemory 39e479e): a
    success, counted in `repaired`. Returns {declared (fields), repaired (fields), scanned, refused, closed}."""
    from abstractmemory import TripleQuery

    receipt = {"declared": 0, "repaired": 0, "scanned": 0, "refused": 0, "closed": 0}
    if getattr(ms, "tasks", None) is None:
        return receipt
    closed = frozenset(str(c.assertion_id) for c in ms.journal.closures(limit=0))
    diary_summary_refusals: Dict[Tuple[str, str], int] = {}
    for scope, owner in scopes:
        for a in ms.store.query(TripleQuery(scope=scope, owner_id=owner or None, predicate="dcterms:abstract",
                                            limit=0)):
            attrs = a.attributes if isinstance(a.attributes, dict) else {}
            if not attrs.get("record_kind") or isinstance(attrs.get("pending"), dict):
                continue
            if str(a.assertion_id or "") in closed:
                receipt["closed"] += 1
                continue
            receipt["scanned"] += 1
            kind = str(attrs.get("record_kind") or "")
            fields = _repair_fields(a)
            if _legacy_diary_summary(kind, a):
                # A legacy public diary summary (his gist / first line stood in): declared pending like a title
                # (AbstractMemory 8235daf, formation's rule diary_summary_refusal: marker + digest.author task, the
                # stand-in kept as provisional version 0), so the drainer authors it. Declared on its own, so a
                # summary memory refuses (a projection claiming a chain position) still lets the title be repaired;
                # refusals are said once per pass and per rule memory names (below).
                res = ms.declare_pending(str(a.subject), fields=["summary"], cause="repair", reason="legacy_missing")
                if res.get("outcome") == "refused":
                    receipt["refused"] += 1
                    why = (str(res.get("reason") or "refused"), str(res.get("detail") or ""))
                    diary_summary_refusals[why] = diary_summary_refusals.get(why, 0) + 1
                else:
                    receipt["declared"] += len(res.get("declared") or [])
                    receipt["repaired"] += len(res.get("repaired") or [])
            if not fields:
                continue
            res = ms.declare_pending(str(a.subject), fields=fields, cause="repair", reason="legacy_missing")
            outcome = res.get("outcome")
            if outcome == "refused":
                receipt["refused"] += 1
                out(f"#REPAIR_REFUSED memory refused to declare {kind} {a.subject} pending ({', '.join(fields)}): "
                    f"{res.get('reason')}{' - ' + str(res.get('detail')) if res.get('detail') else ''}")
                continue
            if outcome not in ("applied", "rearmed", "unchanged"):
                receipt["refused"] += 1
                out(f"#REPAIR_REFUSED memory answered an unknown outcome {outcome!r} declaring {kind} {a.subject}")
                continue
            receipt["declared"] += len(res.get("declared") or [])
            receipt["repaired"] += len(res.get("repaired") or [])
    for (why, detail), count in sorted(diary_summary_refusals.items()):
        out(f"#REPAIR_REFUSED memory refused to declare {count} legacy public diary summary(ies) pending "
            f"({why}{' - ' + detail if detail else ''}): their stored summaries stay as they are")
    if receipt["declared"]:
        out(f"(sleep: {receipt['declared']} missing title(s)/summary(ies) declared pending for their authoring)")
    return receipt


# Logged once per process when memory lacks the re-arm verbs (a loud line, not one per tick).
_REARM_VERB_MISSING_LOGGED = False


def rearm_exhausted_authoring(ms: Any, *, scopes: Optional[Sequence[Tuple[str, str]]] = None,
                              out: Callable[[str], None] = lambda _s: None) -> Dict[str, Any]:
    """Re-arm the pending records whose title.author / digest.author tasks all ended (operator 2026-11-09: a title
    and a summary must ALWAYS exist; a capability failure that exhausted the attempts must not leave a record pending
    with no task). Through memory's own verbs only: `pending_records(ms, scopes=, live_task=False)` finds the
    records with a pending field and no live task, `rearm_pending(ms, record_id, cause="repair", reason=)` queues
    their fresh task (the marker stays, its `rearmed` count grows). `scopes` None = every scope. A refusal is logged
    (#REARM_REFUSED) and counted. Memory without the verbs: one loud #MISSING_VERB line per process, nothing
    re-armed (never a write around memory). Returns {rearmed, refused, unchanged, missing_verb?}."""
    global _REARM_VERB_MISSING_LOGGED

    receipt: Dict[str, Any] = {"rearmed": 0, "refused": 0, "unchanged": 0}
    if getattr(ms, "tasks", None) is None:
        return receipt
    try:
        from abstractmemory import pending_records, rearm_pending
    except ImportError as e:
        receipt["missing_verb"] = "abstractmemory.rearm_pending / pending_records"
        if not _REARM_VERB_MISSING_LOGGED:
            _REARM_VERB_MISSING_LOGGED = True
            out(f"#MISSING_VERB abstractmemory.rearm_pending / pending_records ({e}): a pending record whose "
                "authoring tasks all ended keeps no task until memory ships the re-arm")
        return receipt
    for rec in pending_records(ms, scopes=scopes, live_task=False):
        res = rearm_pending(ms, rec["record_id"], cause="repair",
                            reason="every authoring task of a pending field ended")
        outcome = res.get("outcome")
        if outcome == "rearmed":
            receipt["rearmed"] += 1
        elif outcome == "refused":
            receipt["refused"] += 1
            out(f"#REARM_REFUSED memory refused to re-arm {rec.get('record_kind')} {rec['record_id']}: "
                f"{res.get('reason')}{' - ' + str(res.get('detail')) if res.get('detail') else ''}")
        else:
            receipt["unchanged"] += 1
    if receipt["rearmed"]:
        out(f"(authoring: {receipt['rearmed']} pending record(s) whose tasks had all ended were re-armed)")
    return receipt


# ------------------------------------------------- memory's formation author (AbstractMemory c46b4cd; Codex review 2 #10)
# Memory's own passes (world_model_pass / world_model_update, dream_pass, mine_candidates_pass, consolidation_pass,
# wake_cue_dedup_pass, and sleep_pass running the first four) take a host `author(kind, body, context)` and author a
# record's title and summary WITH it; only a recorded failure forms it pending (`attributes.pending.failure`, and the
# pass receipt's `authoring.failures`, from which the host writes its authoring-failure lines).


def _formation_result(authored: Dict[str, Any], fields: Sequence[str], reader: Dict[str, Any],
                      sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"author": {"kind": "capability-model", "id": AUTHORING_CAPABILITY}, "reader": reader,
                           "sources": sources}
    for f in ("title", "summary"):
        if f in fields:
            out[f] = authored[f]
    return out


def capability_formation_author(llm_for_capability: Optional[Callable[[str], Any]], artifacts: Any,
                                seen: Optional[Dict[str, Any]] = None, *,
                                should_stop: Callable[[], bool]) -> Callable[[str, str, Any], Dict[str, Any]]:
    """Memory's `author` from the summaries capability client (the loop's sleep, a session's card updates): one
    structured call per record, the call kept as a home artifact (its `sources`); raises on any failure, which memory
    records as `author_raised` with the class and message. `seen` (optional) gets the client's reader, so the host's
    failure lines can name the model. `should_stop` (REQUIRED, the admission's fence: Codex review 4 #3) is checked
    immediately before each model call and after it returns - the operator's Stop raises StoppedByOperator, never a
    call."""
    from .lifecycle_fence import StoppedByOperator

    def author(kind: str, body: str, context: Any) -> Dict[str, Any]:
        fields = list((context or {}).get("fields") or ("title", "summary"))
        if llm_for_capability is None:
            raise AuthoringUnavailable("no summaries capability model in this process")
        llm = llm_for_capability(AUTHORING_CAPABILITY)
        reader = _reader_of(llm)
        if seen is not None:
            seen["reader"] = reader
        if should_stop():
            raise StoppedByOperator("before the authoring model was called")
        try:
            authored = author_metadata(llm, kind=str(kind), body=str(body or ""))
        except (AuthoringFailed, AuthoringUnavailable) as e:
            # Memory records the class and message of what the author raised: the client's own exception when it
            # raised (never the wrapper), else the authoring failure itself (a reply that did not fit).
            if e.__cause__ is not None:
                raise e.__cause__ from None
            raise
        if should_stop():
            raise StoppedByOperator("after the authoring model answered (the reply is abandoned)")
        record = f"{(context or {}).get('pass') or 'memory'}:{(context or {}).get('record_kind') or kind}"
        sources = _source_artifact(artifacts, record_id=record, kind=str(kind), body=str(body or ""),
                                   authored=authored, reader=reader)
        return _formation_result(authored, fields, reader, sources)

    return author


def handler_formation_author(handler: Any, artifacts: Any, *, run: Any = None,
                             should_stop: Callable[[], bool]) -> Callable[[str, str, Any], Dict[str, Any]]:
    """Memory's `author` from a host's capability LLM_CALL handler (the durable lanes: open_entity_runtime's
    `capability_handlers["summaries"]`): the authoring payload goes to the handler directly, its result is read as
    the formation step reads one (`parse_authored_result`, the capability echo stamped), the call kept as an
    artifact; raises on any failure. `should_stop` (REQUIRED: Codex review 4 #3) is checked immediately before the
    handler is called and after it returns."""
    from ..core.models import Effect, EffectType
    from .lifecycle_fence import StoppedByOperator

    def author(kind: str, body: str, context: Any) -> Dict[str, Any]:
        fields = list((context or {}).get("fields") or ("title", "summary"))
        if should_stop():
            raise StoppedByOperator("before the authoring model was called")
        out = handler(run, Effect(type=EffectType.LLM_CALL, payload=authoring_llm_call_payload(str(kind), str(body or ""))),
                      None)
        if should_stop():
            raise StoppedByOperator("after the authoring model answered (the reply is abandoned)")
        if getattr(out, "status", None) != "completed" or not isinstance(getattr(out, "result", None), dict):
            raise AuthoringUnavailable(str(getattr(out, "error", "") or "the authoring call failed"))
        result = {**out.result, AUTHORING_CAPABILITY_KEY: AUTHORING_CAPABILITY}
        parsed = parse_authored_result(result)
        reader = {"provider": str(result.get("provider") or "capability"),
                  "model": str(parsed.get("model") or result.get("model") or AUTHORING_CAPABILITY), "thinking": None}
        authored = {**parsed, "raw": json.dumps({"title": parsed["title"], "summary": parsed["summary"]})}
        record = f"{(context or {}).get('pass') or 'memory'}:{(context or {}).get('record_kind') or kind}"
        sources = _source_artifact(artifacts, record_id=record, kind=str(kind), body=str(body or ""),
                                   authored=authored, reader=reader)
        return _formation_result(authored, fields, reader, sources)

    return author


def log_pass_authoring_failures(result: Any, *, home_dir: Any, entity: Optional[str],
                                reader: Optional[Dict[str, Any]] = None,
                                out: Callable[[str], None] = lambda _s: None) -> int:
    """One authoring-failure line per failure a memory pass reports (`result["authoring"]["failures"]`: record_id,
    record_kind, pass, fields, reason no_author_supplied | author_raised | invalid_result, error_class, message, at),
    naming the capability model used when one was built. Returns how many were written."""
    from .title_failures import record_authoring_failure

    failures = list(((result or {}).get("authoring") or {}).get("failures") or []) if isinstance(result, dict) else []
    for f in failures:
        fields = [str(x) for x in (f.get("fields") or [])]
        record_authoring_failure(
            home_dir, entity=entity, record_id=f.get("record_id"), kind=str(f.get("record_kind") or "memory"),
            raw_block="", template_id=f"memory.{f.get('pass') or 'pass'}",
            provider=(reader or {}).get("provider"), model=(reader or {}).get("model"),
            reason=str(f.get("reason") or "unknown"),
            missing="both" if len(fields) == 2 else (fields[0] if fields else "title"),
            extra={"error_class": f.get("error_class"), "error_message": f.get("message"), "pass": f.get("pass"),
                   "capability": AUTHORING_CAPABILITY, "failed_at": f.get("at")},
            out=out,
        )
    return len(failures)
