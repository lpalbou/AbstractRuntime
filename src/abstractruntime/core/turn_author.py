"""Who speaks in a turn: the `author` of a session turn's prompt and of its answer.

A session turn is one root run: a prompt (who asked) and an answer (who replied). Clients label
the bubbles from these two authors, and the history replayed to the next agent run names the
speaker of an entity's reply (an `@entity` mention recorded as a turn of a chat session).

One shape everywhere (every key present, nulls included, never omitted):

    {"kind": "entity" | "agent" | "person",
     "name": str | None,                 # display name (never a model name)
     "entity_id": str | None,            # "entity:<slug>" for kind entity
     "identity_ref": str | None,         # "entity:castor#3" / "session:<id>#2" (digital identity)
     "avatar": {"style": str | None, "color": str | None} | None}

Where the authors come from, first match wins (structured fields only, no text is parsed):

- the ANSWER: `vars._meta.author` when the host stamped one (the gateway's mention door stamps
  the entity); else an entity run (the signed entity stamp `vars._runtime.entity.entity_id`, a
  summon) is kind `entity`; else kind `agent` named by the run's `vars.workflow_selection.name`
  (the entrypoint name the host resolved at start), then the workflow id's bundle id, flow id,
  or the id itself. `identity_ref` is the run's `vars._runtime.identity_ref` when it carries one.
- the PROMPT: `vars._meta.prompt_author` when the host stamped one (the account's display name);
  else the run's origin (`vars._meta.origin`, backlog 1038): a person origin is kind `person`
  named by the actor's user part (`default:alice` -> `alice`); a trigger, agent or system origin
  is kind `agent` named by the origin's client or kind; no origin (a run older than origins) is
  kind `person` with name None.

`speaker_line(author, text)` is the plain rendering a model sees for an entity's reply in
replayed history (`nerin (entity): ...`); every other author leaves the text as it is.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

AUTHOR_KEY = "author"
PROMPT_AUTHOR_KEY = "prompt_author"
# `vars._meta.repeat_of`: this turn's prompt REPEATS another turn's (the run id of the person's own
# turn whose words an `@entity` mention answers again). Clients fold the repeated bubble into that
# turn; replay keeps the prompt so every turn stays a whole user/assistant pair.
REPEAT_OF_KEY = "repeat_of"
AUTHOR_KINDS = ("entity", "agent", "person")
AUTHOR_FIELDS = ("kind", "name", "entity_id", "identity_ref", "avatar")


def _opt_text(value: Any, field: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"author.{field} must be a string or None, not {type(value).__name__}")
    text = value.strip()
    return text or None


def author_for(
    kind: str,
    name: Optional[str],
    *,
    entity_id: Optional[str] = None,
    identity_ref: Optional[str] = None,
    avatar: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """A validated author (see the module doc). Raises ValueError."""
    if kind not in AUTHOR_KINDS:
        raise ValueError(f"author.kind must be one of {list(AUTHOR_KINDS)}, not {kind!r}")
    av: Optional[Dict[str, Any]] = None
    if avatar is not None:
        if not isinstance(avatar, Mapping):
            raise ValueError("author.avatar must be an object {style, color} or None")
        av = {"style": _opt_text(avatar.get("style"), "avatar.style"), "color": _opt_text(avatar.get("color"), "avatar.color")}
    return {
        "kind": kind,
        "name": _opt_text(name, "name"),
        "entity_id": _opt_text(entity_id, "entity_id"),
        "identity_ref": _opt_text(identity_ref, "identity_ref"),
        "avatar": av,
    }


def read_author(value: Any) -> Optional[Dict[str, Any]]:
    """A stored author in the full shape, or None when absent or malformed."""
    if not isinstance(value, Mapping):
        return None
    try:
        return author_for(
            str(value.get("kind") or ""),
            value.get("name"),
            entity_id=value.get("entity_id"),
            identity_ref=value.get("identity_ref"),
            avatar=value.get("avatar"),
        )
    except ValueError:
        return None


def stamp_turn_authors(
    vars_obj: Dict[str, Any], *, author: Optional[Mapping[str, Any]] = None, prompt_author: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    """Write `vars._meta.author` / `vars._meta.prompt_author` (validated; None leaves a key as is)."""
    meta = vars_obj.get("_meta") if isinstance(vars_obj.get("_meta"), dict) else {}
    meta = dict(meta)
    for key, value in ((AUTHOR_KEY, author), (PROMPT_AUTHOR_KEY, prompt_author)):
        if value is None:
            continue
        checked = read_author(value)
        if checked is None:
            raise ValueError(f"_meta.{key} is not a valid author: {value!r}")
        meta[key] = checked
    vars_obj["_meta"] = meta
    return vars_obj


def _meta(vars_obj: Any) -> Mapping[str, Any]:
    meta = vars_obj.get("_meta") if isinstance(vars_obj, Mapping) else None
    return meta if isinstance(meta, Mapping) else {}


def _identity_ref_text(vars_obj: Any) -> Optional[str]:
    from ..identity.digital_identity import read_run_identity_ref

    if not isinstance(vars_obj, Mapping):
        return None
    try:
        ref = read_run_identity_ref(vars_obj)
    except (TypeError, ValueError):
        return None
    return str(ref) if ref is not None else None


def _agent_name(vars_obj: Any, workflow_id: Any) -> Optional[str]:
    sel = vars_obj.get("workflow_selection") if isinstance(vars_obj, Mapping) else None
    name = sel.get("name") if isinstance(sel, Mapping) else None
    if isinstance(name, str) and name.strip():
        return name.strip()
    wid = str(sel.get("workflow_id") or "") if isinstance(sel, Mapping) else ""
    wid = wid.strip() or (str(workflow_id).strip() if isinstance(workflow_id, str) else "")
    if not wid:
        return None
    left, sep, right = wid.partition(":")
    if sep and left.strip() and right.strip() and "/" not in left:
        bundle = left.partition("@")[0].strip()
        return bundle or right.strip()
    return wid


def answer_author(vars_obj: Any, *, workflow_id: Any = None) -> Dict[str, Any]:
    """Who answered this turn (see the module doc)."""
    stored = read_author(_meta(vars_obj).get(AUTHOR_KEY))
    if stored is not None:
        return stored
    runtime_ns = vars_obj.get("_runtime") if isinstance(vars_obj, Mapping) else None
    stamp = runtime_ns.get("entity") if isinstance(runtime_ns, Mapping) else None
    entity_id = str(stamp.get("entity_id") or "").strip() if isinstance(stamp, Mapping) else ""
    if entity_id:
        name = entity_id.split(":", 1)[1] if ":" in entity_id else entity_id
        return author_for("entity", name, entity_id=entity_id, identity_ref=_identity_ref_text(vars_obj))
    return author_for("agent", _agent_name(vars_obj, workflow_id), identity_ref=_identity_ref_text(vars_obj))


def prompt_author(vars_obj: Any) -> Dict[str, Any]:
    """Who wrote this turn's prompt (see the module doc)."""
    meta = _meta(vars_obj)
    stored = read_author(meta.get(PROMPT_AUTHOR_KEY))
    if stored is not None:
        return stored
    origin = meta.get("origin")
    if isinstance(origin, Mapping) and origin.get("kind") in ("trigger", "agent", "system"):
        client = origin.get("client")
        name = client if isinstance(client, str) and client.strip() else str(origin.get("kind"))
        return author_for("agent", name)
    actor = origin.get("actor") if isinstance(origin, Mapping) else None
    name = None
    if isinstance(actor, str) and actor.strip():
        name = actor.strip().rsplit(":", 1)[-1] or None
    return author_for("person", name)


def prompt_repeat_of(vars_obj: Any) -> Optional[str]:
    """The run id this turn's prompt repeats (`vars._meta.repeat_of`), or None."""
    value = _meta(vars_obj).get(REPEAT_OF_KEY)
    return value.strip() if isinstance(value, str) and value.strip() else None


def turn_authors(vars_obj: Any, *, workflow_id: Any = None) -> Dict[str, Dict[str, Any]]:
    """`{"prompt": <author>, "answer": <author>}` of one turn run."""
    return {"prompt": prompt_author(vars_obj), "answer": answer_author(vars_obj, workflow_id=workflow_id)}


def speaker_line(author: Any, text: str) -> str:
    """The text a model sees for a replayed message: `<name> (entity): <text>` for an entity's
    reply, the text unchanged for anyone else (an agent's own reply, a person)."""
    a = read_author(author)
    if a is None or a["kind"] != "entity":
        return text
    name = a["name"] or a["entity_id"] or "entity"
    return f"{name} (entity): {text}"


__all__ = [
    "AUTHOR_FIELDS",
    "AUTHOR_KEY",
    "AUTHOR_KINDS",
    "PROMPT_AUTHOR_KEY",
    "REPEAT_OF_KEY",
    "answer_author",
    "author_for",
    "prompt_author",
    "prompt_repeat_of",
    "read_author",
    "speaker_line",
    "stamp_turn_authors",
    "turn_authors",
]
