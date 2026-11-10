"""THE SELF-KNOWLEDGE BRIEF: what the entity knows about, at a glance, written while it sleeps.

The operator's decision (2026-11-10): "that's one of the thing that can truly help the entity actively explore its
memory when relevant ... accessing a memory first go through discoverability, like in FAIR principles.. it can't search
what it doesn't know exist. so we have to help the entities know what it knows, at least at a very high level. then it
can deep dive actively (consciously) into its memory. i would however be careful with those summaries, so that it
doesn't invalidate the prefix (kv cache prompt). probably one more reason why the title/summaries/trails of thoughts
should be better computed at sleep, so it doesn't bother the ongoing work."

So the brief is:

- WRITTEN IN THE SLEEP, at the end of the night's topics stage (`sleep_self_brief`, run by `life.build_consolidator`
  right after `topics.sleep_communities`): ONE call of the summaries author (the capability that titles every record,
  its briefing naming the entity, `self_brief_system_prompt`) over the night's topics as memory serves them
  (`topics.entity_communities`: each topic's title, summary, size, newest memory, activity, and its `facts.paths`,
  the trails of thought - two memories recalled together, n times - rendered by `brief_body`). 200-400 tokens: the
  hard cap `SELF_BRIEF_MAX_TOKENS` is measured by the runtime's budget estimator (`digest.token_estimate`, memory's
  own), and a longer reply is refused, never cut. On any failure the previous brief stays and one authoring-failure
  line says why. A sleep whose topics did not change (memory wrote no new version) keeps the brief written for the
  current version: no call, no new prefix.
- FENCED: no model call starts after the operator's Stop, a reply that returns after it is abandoned, and the write is
  one fenced write section (the home's write barrier held shared, the fence checked inside it, the file replaced
  atomically inside it). A Stop at any point writes nothing.
- A PROJECTION, NOT A MEMORY: `<home>/self_brief.md` (a header `version_seq`, `written_at`, `tokens`, then the text),
  next to the home's state and phases files, never in the memory graph - a brief stored as a memory would be recalled,
  grouped and titled like the memories it describes.
- IN EVERY LANE'S SYSTEM PREFIX: `chat.compose_system_base(..., self_brief=)` places it under `SELF_BRIEF_HEADING`
  right after the identity block and before everything that changes per turn (see that function for the cache
  reasoning). `self_brief_for_prompt` is the one read every lane makes, at summon; a home without a brief gets no
  section at all.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "SELF_BRIEF_FILENAME",
    "SELF_BRIEF_HEADING",
    "SELF_BRIEF_MAX_TOKENS",
    "SELF_BRIEF_TEMPLATE_ID",
    "SELF_BRIEF_TEMPLATE_VERSION",
    "author_self_brief",
    "brief_body",
    "brief_prompt",
    "read_self_brief",
    "self_brief_for_prompt",
    "self_brief_section",
    "self_brief_system_prompt",
    "sleep_self_brief",
    "write_self_brief",
]

SELF_BRIEF_FILENAME = "self_brief.md"
#: The section's heading in the system prefix (operator 2026-11-10).
SELF_BRIEF_HEADING = "What I know about, at a glance (updated in my sleep):"
#: The hard cap, by the runtime's budget estimator (`digest.token_estimate`): a longer brief is refused, never cut.
SELF_BRIEF_MAX_TOKENS = 400
#: What the prompt asks for: 170 words is about 300 tokens by that estimator (a castor-like brief of 11 topics: 202
#: words, 348 tokens), so a reply a third longer than asked still fits under the cap.
SELF_BRIEF_WORDS = 170
SELF_BRIEF_TEMPLATE_ID = "runtime.self_brief"
SELF_BRIEF_TEMPLATE_VERSION = "2026-11-10"
#: The body's bounds: the topics with their own line (the `topics` tool's page), the trails shown.
BODY_TOPICS = 40
BODY_TRAILS = 8
_STOPPED = "the operator's Stop"
_HEADER_KEYS = ("version_seq", "written_at", "tokens")


# ------------------------------------------------------------------------------------------------ the stored brief


def _path(home_dir: Any) -> Path:
    return Path(home_dir) / SELF_BRIEF_FILENAME


def read_self_brief(home_dir: Any) -> Optional[Dict[str, Any]]:
    """The home's brief: `{"text", "version_seq", "written_at", "tokens"}`, or None when the home has none. Raises
    ValueError for a file that is not a brief (no header, a header without its keys, no text)."""
    import yaml

    path = _path(home_dir)
    if not path.exists():
        return None
    raw = path.read_text(encoding="utf-8")
    head, sep, text = raw[4:].partition("\n---\n") if raw.startswith("---\n") else ("", "", "")
    if not sep:
        raise ValueError(f"{path} has no header (a '---' block naming {', '.join(_HEADER_KEYS)})")
    header = yaml.safe_load(head)
    if not isinstance(header, dict) or any(k not in header for k in _HEADER_KEYS):
        raise ValueError(f"{path}'s header must name {', '.join(_HEADER_KEYS)} (it holds {header!r})")
    if not text.strip():
        raise ValueError(f"{path} holds no brief under its header")
    return {"text": text.strip(), "version_seq": header["version_seq"], "written_at": str(header["written_at"]),
            "tokens": int(header["tokens"])}


def self_brief_for_prompt(home_dir: Any, *, warn: Callable[[str], None]) -> str:
    """THE ONE READ every lane makes for its prefix, at summon: the brief's text, or "" when the home has none. An
    unreadable file is said once through `warn` and gives no section - a projection never blocks a summon."""
    try:
        brief = read_self_brief(home_dir)
    except (OSError, ValueError) as e:
        warn(f"#FALLBACK the self brief is unreadable ({type(e).__name__}: {e}); this session has no brief")
        return ""
    return "" if brief is None else str(brief["text"])


def self_brief_section(text: Optional[str]) -> str:
    """The prefix section: the heading, then the brief. "" (no section, no placeholder) without a brief."""
    body = str(text or "").strip()
    return f"{SELF_BRIEF_HEADING}\n{body}" if body else ""


def write_self_brief(home_dir: Any, text: str, *, version_seq: Optional[int], tokens: int, fence: Any) -> Path:
    """Replace the home's brief: ONE fenced write section (the home's write barrier held shared, so a racing Stop
    waits for it; `fence.check` inside it, so a Stop that landed first refuses it - StoppedByOperator, nothing
    written), the file replaced atomically. `fence` is REQUIRED (the night's LifecycleFence)."""
    import yaml

    from ..utils.atomic_files import atomic_write_text
    from .lifecycle_fence import write_barrier

    header = {"version_seq": None if version_seq is None else int(version_seq),
              "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tokens": int(tokens)}
    content = "---\n" + yaml.safe_dump(header, sort_keys=False) + "---\n" + text.strip() + "\n"
    path = _path(home_dir)
    with write_barrier(home_dir, exclusive=False):
        fence.check("writing the self brief")
        atomic_write_text(path, content)
    return path


# ------------------------------------------------------------------------------------------------ the author


def self_brief_system_prompt(entity_name: str) -> str:
    """The summaries author's briefing for the brief: the task, the entity's naming clause (the same as every
    authored record's, `authoring.entity_naming_clause`) and the topic names' style. An empty name raises."""
    from .authoring import entity_naming_clause

    return ("You write the at-a-glance brief of an entity's memory: a short overview of the topics its memories are "
            "grouped into and of the trails of thought that run through them. The entity reads it at the start of "
            "every conversation, to know what its memory holds and what it can look up there. Use only what the "
            f"topics say; invent nothing. {entity_naming_clause(entity_name)} Write the topic names in sentence "
            "case, with no quotes.")


def brief_prompt(entity_name: str, body: str) -> str:
    """The author's prompt: the topics (`brief_body`), then what to write."""
    name = " ".join(str(entity_name or "").split())
    return (f"{body}\n\nWrite {name}'s at-a-glance brief from these topics: plain text, no headings, no markdown, "
            f"at most {SELF_BRIEF_WORDS} words.\n"
            "- One line per topic, largest first: a short name for the topic, a colon, and what it holds in a few "
            f"words. When the topics do not all fit in {SELF_BRIEF_WORDS} words, give the largest their own line and "
            "name the others together on one last line.\n"
            "- One line beginning \"Trails of thought:\" naming the most frequent trails, each written "
            "\"A → B (n times)\"; leave it out when there are none.\n"
            "- One last line beginning \"Since the last sleep:\" saying in a few words what changed.")


def _model_class():
    from pydantic import BaseModel

    class SelfBrief(BaseModel):
        brief: str

    return SelfBrief


def author_self_brief(llm: Any, *, body: str, entity_name: str) -> Dict[str, Any]:
    """ONE structured call of the summaries capability client: `{"text", "tokens", "raw"}`. Raises (authoring's
    classes, so the failure line reads like every other) AuthoringUnavailable (no client, an unreachable endpoint),
    AuthoringCallFailed (the client raised), AuthoringFailed (a reply without a brief, or over
    SELF_BRIEF_MAX_TOKENS: refused, never cut); TurnCancelled (the operator's Stop through a fenced client) passes."""
    from .authoring import AuthoringCallFailed, AuthoringFailed, AuthoringUnavailable, _unreachable
    from .chat import TurnCancelled
    from .digest import token_estimate

    if llm is None:
        raise AuthoringUnavailable("no summaries capability model in this process")
    system_prompt = self_brief_system_prompt(entity_name)
    try:
        resp = llm.generate(prompt=brief_prompt(entity_name, body), system_prompt=system_prompt,
                            response_model=_model_class())
    except TurnCancelled:
        raise
    except Exception as e:  # noqa: BLE001 - classified, never swallowed
        if _unreachable(e):
            raise AuthoringUnavailable(f"{type(e).__name__}: {e}") from e
        from pydantic import ValidationError

        if isinstance(e, ValidationError):
            raise AuthoringFailed(f"the reply did not fit the brief: {type(e).__name__}: {e}") from e
        raise AuthoringCallFailed(f"the model call failed: {type(e).__name__}: {e}") from e
    if hasattr(resp, "model_dump"):
        data = resp.model_dump()
    elif isinstance(resp, dict):
        data = dict(resp)
    else:
        try:
            data = json.loads(str(getattr(resp, "content", None) or ""))
        except ValueError as e:
            raise AuthoringFailed("the reply was not the structured brief") from e
    raw = json.dumps(data, ensure_ascii=False) if isinstance(data, dict) else str(data)
    text = data.get("brief") if isinstance(data, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise AuthoringFailed("the reply carried no brief")
    # Serving hygiene only (the topics tool's rule for remembered words): a fence in the prefix is defanged.
    text = text.strip().replace("```", "'''")
    tokens = token_estimate(text)
    if tokens > SELF_BRIEF_MAX_TOKENS:
        failed = AuthoringFailed(f"the brief is {tokens} tokens, over the {SELF_BRIEF_MAX_TOKENS}-token cap")
        failed.raw = raw  # type: ignore[attr-defined]
        raise failed
    return {"text": text, "tokens": tokens, "raw": raw}


# ------------------------------------------------------------------------------------------------ the body


def _titled(c: Dict[str, Any]) -> bool:
    return bool(c["title"]) and "title" not in (c["pending"] or ())


def _topic_title(c: Dict[str, Any]) -> str:
    from .tools import sanitize_tool_surface

    return sanitize_tool_surface(str(c["title"]), 120) if _titled(c) else "(untitled yet)"


def brief_body(entity_name: str, payload: Dict[str, Any], receipt: Dict[str, Any],
               titles: Dict[str, Tuple[str, str]]) -> Tuple[str, Dict[str, int]]:
    """The author's body: the topics of memory's payload (version 2) by size then this week's activity, the trails of
    thought across them (their `facts.paths`, most often first, each memory named by its title), and what tonight's
    pass changed (`receipt`: memory's, from the topics stage). `titles`: memory graph id -> (title, formed at) for the
    trails' memories. Returns (body, {"topics": n shown with their line, "trails": n})."""
    from abstractmemory.community_index import FACTS_WINDOW_DAYS

    from .tools import sanitize_tool_surface

    name = " ".join(str(entity_name or "").split())
    topics = sorted(payload["communities"],
                    key=lambda c: (-int(c["size"]), -int(c["facts"]["activity"]), str(c["key"])))
    by_key = {str(c["key"]): c for c in topics}
    titled = [c for c in topics if _titled(c)]
    untitled = [c for c in topics if not _titled(c)]
    fresh = receipt["version_seq"] is not None  # a version formed tonight: its changes are tonight's
    week = f"in the last {FACTS_WINDOW_DAYS} days"
    lines = [f"The topics of {name}'s memory ({len(topics)} topics over {sum(int(c['size']) for c in topics)} "
             f"memories), largest first:"]
    shown = titled[:BODY_TOPICS]
    for i, c in enumerate(shown, start=1):
        facts = c["facts"]
        newest = str(facts["span"]["last"] or "")[:10] or "unknown"
        added = int(facts["changed_since_last_sleep"]["added"]) if fresh and str(c["key"]) not in \
            set(receipt["ids"]["new"]) else 0
        grew = f", {added} added since the last sleep" if added else ""
        summary = (sanitize_tool_surface(str(c["summary"]), 0) if c["summary"] and "summary" not in (c["pending"] or ())
                   else "(its summary is not written yet)")
        lines.append(f"{i}. {_topic_title(c)} - {int(c['size'])} memories, newest {newest}, recalled "
                     f"{int(facts['activity'])} times {week}{grew}. {summary}")
    rest = titled[BODY_TOPICS:]
    if rest:
        lines.append(f"... and {len(rest)} smaller topics ({sum(int(c['size']) for c in rest)} memories): "
                     + "; ".join(_topic_title(c) for c in rest))
    if untitled:
        lines.append(f"{len(untitled)} more topics are not titled yet "
                     f"({sum(int(c['size']) for c in untitled)} memories).")

    # A trail runs from the older memory to the newer. The trails of one memory with the same count (one context
    # recalled them together, n times) are one line: "A → B; C; D (n times each)".
    grouped: Dict[Tuple[int, str, str], Dict[str, Any]] = {}
    for c in topics:
        for p in c["facts"]["paths"]:
            first, then = sorted((str(p["from"]), str(p["to"])), key=lambda g: (titles.get(g, ("", ""))[1], g))
            group = grouped.setdefault((int(p["count"]), str(c["key"]), first), {"topic": c, "to": []})
            group["to"].append(then)
    trails = sorted(grouped.items(), key=lambda kv: (-kv[0][0], -int(kv[1]["topic"]["size"]), kv[0][1],
                                                     kv[0][2]))[:BODY_TRAILS]
    lines.append("")
    if trails:
        lines.append(f"Trails of thought (memories recalled together {week}, most often first):")
        for (count, _key, first), group in trails:
            ends = "; ".join(_memory_title(titles, g) for g in group["to"])
            each = " each" if len(group["to"]) > 1 else ""
            lines.append(f"- {_memory_title(titles, first)} → {ends} ({count} times{each}), in the topic "
                         f"\"{_topic_title(group['topic'])}\"")
    else:
        lines.append(f"Trails of thought: none {week} (no two memories of a topic were recalled together).")

    lines.append("")
    if not fresh:
        lines.append("What changed since the last sleep: no topic changed.")
    else:
        ids = receipt["ids"]
        parts: List[str] = []
        new = [by_key[k] for k in ids["new"] if k in by_key]
        if new:
            parts.append(f"{len(new)} new topics (" + "; ".join(_topic_title(c) for c in new) + ")")
        if int(receipt["dissolved"]):
            parts.append(f"{int(receipt['dissolved'])} topics dissolved")
        renamed = [by_key[k] for k in ids["renamed"] if k in by_key]
        if renamed:
            parts.append(f"{len(renamed)} retitled (" + "; ".join(_topic_title(c) for c in renamed) + ")")
        if int(receipt["moved_out"]):
            parts.append(f"{int(receipt['moved_out'])} memories moved between topics")
        grown = sorted(((int(c["facts"]["changed_since_last_sleep"]["added"]), c) for c in topics
                        if str(c["key"]) not in set(ids["new"])), key=lambda t: (-t[0], str(t[1]["key"])))
        grown = [(n, c) for n, c in grown if n > 0][:3]
        if grown:
            parts.append("grew most: " + "; ".join(f"{_topic_title(c)} (+{n})" for n, c in grown))
        lines.append("What changed since the last sleep: " + ("; ".join(parts) if parts else
                                                               "the topics were regrouped, none visibly changed")
                     + ".")
    return "\n".join(lines), {"topics": len(shown), "trails": len(trails)}


def _memory_title(titles: Dict[str, Tuple[str, str]], gid: str) -> str:
    from .tools import sanitize_tool_surface

    title = titles.get(gid, ("", ""))[0]
    return sanitize_tool_surface(title, 120) if title else "(an untitled memory)"


def _memory_titles(home: Any, ladder: Sequence[Tuple[str, str]], wanted: set) -> Dict[str, Tuple[str, str]]:
    """graph id -> (title, formed at) for the trails' memories: one read of his digests (HomeMemoryReader's)."""
    from .memory_reader import HomeMemoryReader
    from .title_failures import is_placeholder_title

    out: Dict[str, Tuple[str, str]] = {}
    if not wanted:
        return out
    reader = HomeMemoryReader(home, ladder=[[str(s), str(o)] for s, o in ladder])
    for a in reader.digest_assertions_all():
        gid = str(a.subject or "")
        if gid in wanted:
            attrs = a.attributes if isinstance(a.attributes, dict) else {}
            title = str(attrs.get("title") or "")
            if attrs.get("title_missing") or is_placeholder_title(title):
                title = ""
            out[gid] = (title, str(a.observed_at or ""))
    return out


# ------------------------------------------------------------------------------------------------ the night's step


def sleep_self_brief(home: Any, *, entity_id: str, scopes: Sequence[Tuple[str, str]], receipt: Dict[str, Any],
                     llm_for_capability: Optional[Callable[[str], Any]], fence: Any,
                     out: Callable[[str], None]) -> Dict[str, Any]:
    """THE NIGHT'S BRIEF, at the end of the topics stage (`receipt`: what `topics.sleep_communities` returned).
    Never raises; returns the receipt the night's dict carries under `self_brief`:

    - `{"skipped": ...}`: the topics stage did not run to a version this runtime read (the night ended before it, a
      Stop, a memory without the verbs, an error) or the home has no topics yet; nothing written, nothing said
      (the topics stage already said why);
    - `{"kept": ..., "version_seq"}`: memory wrote no new version tonight and the brief on file was written for the
      current one - no call, no new prefix;
    - `{"written": true, "version_seq", "tokens", "topics", "trails"}`: one author call, the file replaced, one
      own_time.log line;
    - `{"failed": <reason>, "error", "kept_previous"}`: no update (the previous brief stays), one authoring-failure
      line (`template_id` runtime.self_brief) and one own_time.log line;
    - `{"stopped": ...}`: the operator's Stop before, during or after the call: nothing written, nothing said."""
    from .authoring import AUTHORING_CAPABILITY, authoring_failure_fields
    from .chat import TurnCancelled
    from .title_failures import record_authoring_failure
    from .topics import entity_communities

    if not isinstance(receipt, dict) or "version_seq" not in receipt or receipt.get("stopped") \
            or receipt.get("error"):
        return {"skipped": "the topics step did not run to a version"}
    if fence.moved():
        return {"stopped": _STOPPED}
    home_dir = home.home_dir
    llm: Any = None
    raw = ""
    try:
        payload = entity_communities(home.ms, scopes=list(scopes), owner_id=str(entity_id))
        if not payload["communities"]:
            return {"skipped": "no topics yet"}
        version_seq = payload["version_seq"]
        if receipt["version_seq"] is None:
            try:
                previous = read_self_brief(home_dir)
            except (OSError, ValueError):
                previous = None  # an unreadable brief is rewritten
            if previous is not None and previous["version_seq"] == version_seq:
                out("(self brief: kept - the topics did not change since it was written)")
                return {"kept": "the topics did not change since it was written", "version_seq": version_seq}
        wanted = {str(p[k]) for c in payload["communities"] for p in c["facts"]["paths"] for k in ("from", "to")}
        body, counts = brief_body(home.name, payload, receipt, _memory_titles(home, scopes, wanted))
        fence.check("before the self brief's model call")
        if llm_for_capability is not None:
            llm = llm_for_capability(AUTHORING_CAPABILITY)
        authored = author_self_brief(llm, body=body, entity_name=home.name)
        raw = authored["raw"]
        fence.check("after the self brief's model answered (the reply is abandoned)")
        write_self_brief(home_dir, authored["text"], version_seq=version_seq, tokens=authored["tokens"], fence=fence)
    except TurnCancelled as stopped:  # the operator's Stop: nothing written, nothing said
        return {"stopped": f"{_STOPPED} ({getattr(stopped, 'where', '')})"}
    except Exception as e:  # noqa: BLE001 - a failed brief keeps the previous one, and says why
        if fence.moved():
            return {"stopped": _STOPPED}
        why = authoring_failure_fields(e, llm)
        raw = raw or str(getattr(e, "raw", "") or "")
        try:
            kept_previous = read_self_brief(home_dir) is not None
        except (OSError, ValueError):
            kept_previous = False
        record_authoring_failure(
            home_dir, entity=str(entity_id), record_id=None, kind="self_brief", raw_block=raw,
            template_id=SELF_BRIEF_TEMPLATE_ID, template_version=SELF_BRIEF_TEMPLATE_VERSION,
            provider=why["provider"], model=why["model"], reason=why["reason"], missing="summary",
            extra={"error_class": why["error_class"], "error_message": why["error_message"],
                   "capability": AUTHORING_CAPABILITY, "kept_previous": kept_previous},
            out=out,
        )
        out(f"(self brief: not rewritten - {why['reason']} ({why['error_class']}: {why['error_message']}); "
            + ("the previous brief stays)" if kept_previous else "there is no brief yet)"))
        return {"failed": why["reason"], "error": f"{why['error_class']}: {why['error_message']}",
                "kept_previous": kept_previous}
    out(f"(self brief: written - {authored['tokens']} tokens over {counts['topics']} topics and {counts['trails']} "
        "trails of thought)")
    return {"written": True, "version_seq": version_seq, "tokens": authored["tokens"], **counts}
