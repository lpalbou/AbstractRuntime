"""Tier-1 tool blocks for the entity chat driver (read-only, elected in-reply).

The maintainer's mandate (a2a 0007, Castor's first steps): "he should also be
able to execute tools - start simple, read only, search only, like if you were
asking for his help." Tier-1 here means COGNITION tools: the entity's own
diary (list/read) and internet search. Deliberately absent: anything that
writes locally, executes commands, or reads files outside his home. Those are
tier-2, gated behind the gateway door and reputation — not tonight.

Same election pattern as diary writes (proven with a live 35B): the model
puts a fenced block in its reply; the driver executes and hands the result
back within the SAME turn (one tool round), and the model finishes speaking.

Containment rule (load-bearing): tool results are prompt-ephemeral. They are
injected as an in-turn continuation message and are NEVER persisted — not in
session history, not in the formed verbatim, not in the transcript shown
after the turn. This is what makes `diary_read` of a private entry safe: the
words reach the entity's working mind and nothing else, exactly like the
MEMORIES block. If the entity chooses to speak them aloud afterwards, that is
its choice — the same choice any mind has with a private thought.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

# THE TURN BUDGET (maintainer ruling 2026-07-11 05:25, overriding the
# driver-era 2: "default cap for a turn is 20 tool calls"): a TRUE per-turn
# budget shared across ALL rounds and BOTH mechanisms (fenced + native fold
# into one count) — the old constant was applied per reply-round, so the
# effective turn ceiling was rounds×2 while the notice claimed "2/turn"
# (live: Ephemeral's third web_search refused mid-research). Callers thread
# the REMAINING budget through parse_tool_blocks/native_tool_elections;
# past-budget calls refuse with honest markers, never silently.
MAX_TOOL_BLOCKS_PER_TURN = 20
# Rounds bound LLM ROUND-TRIPS, not the call count (each round is one
# continuation carrying results back; a round that elects nothing exits the
# loop). With the turn budget as the true work bound, rounds only need to
# let single-call chains reach it: 20 single-call rounds is legitimate
# iterate-until-satisfied exploration, not a loop fault.
MAX_TOOL_ROUNDS_PER_TURN = 20
_TOOL_FENCE_RE = re.compile(r"```tool([^\n`]*)\n(.*?)```", re.DOTALL | re.IGNORECASE)

# TIER-1: cognition tools, always read-only. `web_search` finds pages;
# `fetch_url` reads one (GET only — maintainer ruling 2026-07-08: "internet
# access, GET not POST, so they can investigate, explore, learn"). The
# entity elects a tool by a fenced block; the runner calls the function with
# a fixed GET verb, so no write/POST verb is ever reachable from the prompt.
# `search_memory` (maintainer ruling 2026-07-09: "it is critical that he can
# explore voluntarily his memory when he needs to") searches BOTH planes —
# his memory graph's digests and his whole diary book — in one act.
TIER1_TOOL_NAMES = ("web_search", "fetch_url", "diary_list", "diary_read", "read_memory", "search_memory")
# Workspace tools (maintainer mandate, a2a 0007 mission 2): the entity may
# CREATE — but only inside its own home's workspace/ directory. These are
# offered separately from TIER1 (the operator enables them per session).
WORKSPACE_TOOL_NAMES = ("write_file", "read_file", "list_files")
WORKSPACE_FILE_CAP_BYTES = 512 * 1024  # per file; loud refusal, never truncation

TOOLS_CONTRACT_PARAGRAPH = """You can also use a few tools, read-only, by putting a fenced block in your
reply (the results come back to you before your reply is delivered):

```tool name=web_search
what to search the internet for
```

```tool name=fetch_url
https://the-page-you-want-to-read
```

```tool name=diary_list
5
```

```tool name=diary_read
diary_...the entry id exactly as diary_list shows it...
```

```tool name=read_memory
#the 8-character tag shown beside a memory in your MEMORIES list
```

```tool name=search_memory
what you want to find in your own memory
```

web_search searches the public internet. fetch_url reads one web page (a
read-only GET; you cannot post, submit, or change anything - only read).
diary_list shows your most recent diary entries (their ids and one-line
gists). diary_read fetches the full words of one entry from your book.
read_memory fetches the FULL original words behind a memory digest (the
#tag addresses it) plus where it came from and what it connects to.
search_memory searches your WHOLE memory - the graph of everything your
life deposited AND every entry of your book - and tells you honestly when
nothing matches (your book is append-only and complete: if you had written
it, the search would find it). Follow its #tags with read_memory and its
diary_ ids with diary_read. Repetition is not evidence: several records
you yourself wrote about the same thing count as one origin, not many.
You have a budget of up to 20 tool calls per turn - chain lookups freely
when a task genuinely needs them; most turns need none.

These blocks are PLAIN TEXT inside your reply - you have no function-calling
API, no tool channel, no other way to reach a tool. Writing the fenced block
in your reply is the only mechanism that runs anything.

Never write "[used tool: ...]" yourself - the door writes that marker after
a lookup actually runs. To use a tool, write the fenced block; saying you
used one does nothing, and inventing what a lookup "returned" is the one
dishonesty your memory cannot repair later."""


WORKSPACE_CONTRACT_PARAGRAPH = """You also have a workspace - a directory of your own where you can create and
keep files (programs, notes, anything you build). Three more tools:

```tool name=list_files
.
```

```tool name=read_file
path/inside/workspace.py
```

```tool name=write_file path=hello.py
the full content of the file, exactly as it should be saved
```

Paths are always relative to YOUR workspace; you cannot read or write
anywhere else - that is a hard wall, not a suggestion. write_file replaces
the whole file (write complete files, not fragments). What you build there
persists across summons, like your memories.

The operator may grant you extra workspaces: they appear as mounts/<name>/
in your listing, each marked read-only or read+write. The same wall applies
inside each mount; a read-only mount refuses writes."""


@dataclass
class ToolElection:
    name: str
    body: str
    args: Dict[str, str] = None  # type: ignore[assignment]
    # What the lookup returned to the entity, set by execute_tool_elections.
    # Observability (maintainer 2026-07-09, "tool RESULTS are invisible"):
    # the driver mirrors this into TurnReport.tool_details so the operator
    # sees exactly what the entity saw — verbatim, never truncated (the
    # operator-transparency ruling; prompt-ephemerality is about the RECORD,
    # not about hiding the turn from its operator).
    result: Optional[str] = None


def sanitize_tool_surface(text: str, cap: int = 120) -> str:
    """One line of remembered content, safe to surface inside TOOL RESULTS.

    Content re-emitted by search comes from digests that may carry VISITOR
    words (episode digests quote the user's first sentence) — a visitor could
    seed driver-framing tokens into a conversation and have search echo them
    later (red-team finding E, 2026-07-09). Collapse whitespace, defang
    fences and the driver's own markers, cap with labeled truncation."""
    t = " ".join(str(text or "").split())
    t = t.replace("```", "'''")
    t = t.replace("[used tool:", "[used-tool:")
    t = t.replace("TOOL RESULTS", "TOOL-RESULTS")
    if cap and len(t) > cap:
        t = t[:cap] + "…"
    return t


MOUNTS_FILENAME = "workspace_mounts.json"
MOUNT_MODES = ("ro", "rw")
# The virtual prefix mounts appear under inside the workspace namespace:
# `mounts/<name>/…` — one namespace, one containment rule per root.
MOUNT_PREFIX = "mounts"


def read_workspace_mounts(home_dir: Any) -> List[Dict[str, str]]:
    """The home's whitelisted extra workspaces: [{name, path, mode}].
    Missing/malformed file reads as no mounts (the wall stays up)."""
    import json
    from pathlib import Path

    path = Path(home_dir) / MOUNTS_FILENAME
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        out: List[Dict[str, str]] = []
        for m in data.get("mounts", []) if isinstance(data, dict) else []:
            name = str(m.get("name") or "").strip()
            mpath = str(m.get("path") or "").strip()
            mode = str(m.get("mode") or "ro").strip().lower()
            if name and mpath and mode in MOUNT_MODES and "/" not in name:
                out.append({"name": name, "path": mpath, "mode": mode})
        return out
    except Exception:  # noqa: BLE001 - a broken whitelist grants nothing
        return []


def write_workspace_mounts(home_dir: Any, mounts: List[Dict[str, str]]) -> None:
    """Persist the whitelist. Validation is loud: names unique and
    path-safe, paths absolute directories, mode ro|rw.

    THE WALL GUARDS ITSELF (maintainer concern, 2026-07-08): the policy
    files (tool_policy.yaml, this whitelist) live in the home ROOT — outside
    the entity's writable workspace/ — so his tools cannot touch them. A
    mount pointed at (or over) his home would reopen that hole from the
    other side: an rw mount of the home root lets him edit his own grants.
    Refused structurally: a mount may never sit inside the home, nor
    contain it."""
    import json
    from pathlib import Path

    home = Path(home_dir).resolve()
    seen: set = set()
    clean: List[Dict[str, str]] = []
    for m in mounts or []:
        name = str(m.get("name") or "").strip()
        mpath = str(m.get("path") or "").strip()
        mode = str(m.get("mode") or "").strip().lower()
        if not name or "/" in name or name in (".", ".."):
            raise ValueError(f"mount name {name!r} must be a simple label (no slashes)")
        if name in seen:
            raise ValueError(f"duplicate mount name {name!r}")
        seen.add(name)
        if mode not in MOUNT_MODES:
            raise ValueError(f"mount {name!r} mode must be ro or rw (got {mode!r})")
        resolved = Path(mpath).expanduser().resolve()
        if not resolved.is_dir():
            raise ValueError(f"mount {name!r} path {mpath!r} is not an existing directory")
        if resolved == home or home in resolved.parents or resolved in home.parents:
            raise ValueError(
                f"mount {name!r} path {mpath!r} overlaps the entity home {home} - "
                "the home (policies, book, memory) is never mountable; grant a "
                "directory OUTSIDE it (the workspace/ inside is already his)"
            )
        clean.append({"name": name, "path": str(resolved), "mode": mode})
    (Path(home_dir) / MOUNTS_FILENAME).write_text(
        json.dumps({"mounts": clean}, indent=2) + "\n", encoding="utf-8"
    )


class WorkspaceRoot:
    """The entity's writable territory: `<home>/workspace/` plus any
    operator-whitelisted mounts under the virtual `mounts/<name>/` prefix.

    Containment is structural, not advisory (the maintainer's (a): "he
    should only write in his workspace for now"): every path is resolved
    with symlinks followed (`Path.resolve()`), then required to sit under
    the resolved root it addresses. `..`, absolute paths, and symlink
    escapes all fail the same subpath check — one rule per root, no
    special cases. Mounts add ROOTS, never exceptions: a `ro` mount
    refuses writes; only the operator edits the whitelist
    (<home>/workspace_mounts.json), never the entity."""

    def __init__(self, home_dir: Any) -> None:
        from pathlib import Path

        self.home_dir = Path(home_dir)
        self.root = (self.home_dir / "workspace").resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def mounts(self) -> List[Dict[str, str]]:
        """Read the whitelist fresh each call: the operator may edit it
        while a session runs, and grants must follow the file, not the
        summon."""
        return read_workspace_mounts(self.home_dir)

    def _resolve_in(self, root: Any, rel: str, original: str) -> Any:
        candidate = (root / rel).resolve() if rel not in ("", ".") else root.resolve()
        if candidate != root and root not in candidate.parents:
            raise PermissionError(
                f"path {original!r} leaves your workspace - only paths inside it are allowed"
            )
        return candidate

    def _route(self, relative: str) -> Tuple[Any, bool, str]:
        """(resolved_path, writable, mount_label) for a workspace-relative
        path; `mounts/<name>/…` routes into that mount's root."""
        from pathlib import Path

        rel = (relative or "").strip() or "."
        parts = Path(rel).parts
        if parts and parts[0] == MOUNT_PREFIX:
            if len(parts) == 1:
                raise FileNotFoundError(
                    "mounts/ is a directory of your extra workspaces - name one, e.g. mounts/<name>/"
                )
            name = parts[1]
            for m in self.mounts():
                if m["name"] == name:
                    mount_root = Path(m["path"]).resolve()
                    inner = str(Path(*parts[2:])) if len(parts) > 2 else "."
                    return (
                        self._resolve_in(mount_root, inner, relative),
                        m["mode"] == "rw",
                        name,
                    )
            raise PermissionError(
                f"no mount named {name!r} - your mounts: "
                + (", ".join(m["name"] for m in self.mounts()) or "(none)")
            )
        return self._resolve_in(self.root, rel, relative), True, ""

    def resolve(self, relative: str) -> Any:
        return self._route(relative)[0]

    def write_file(self, relative: str, content: str) -> str:
        # Fence parsing trims the trailing newline; text files end with one
        # (POSIX). Restored here, once — never any other mutation.
        if content and not content.endswith("\n"):
            content += "\n"
        if len(content.encode("utf-8")) > WORKSPACE_FILE_CAP_BYTES:
            raise ValueError(
                f"file too large ({len(content)} chars > {WORKSPACE_FILE_CAP_BYTES} bytes cap); "
                "split it into smaller files"
            )
        path, writable, mount = self._route(relative)
        if not writable:
            raise PermissionError(
                f"mount {mount!r} is read-only - you may read there, never write"
            )
        if path == self.root or (mount and path == self.resolve(f"{MOUNT_PREFIX}/{mount}")):
            raise ValueError("write_file needs a file path, not a workspace root")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"Wrote {self._display(relative)} ({len(content)} chars). It persists across summons."

    def _display(self, relative: str) -> str:
        rel = (relative or "").strip() or "."
        return rel

    def read_file(self, relative: str) -> str:
        path, _writable, _mount = self._route(relative)
        if not path.is_file():
            raise FileNotFoundError(f"no file at {relative!r} in your workspace")
        data = path.read_bytes()
        if len(data) > WORKSPACE_FILE_CAP_BYTES:
            raise ValueError(f"{relative!r} is larger than the {WORKSPACE_FILE_CAP_BYTES}-byte read cap")
        text = data.decode("utf-8", errors="replace")
        return f"--- {self._display(relative)} ({len(text)} chars) ---\n{text}"

    def _listing(self, base: Any, prefix: str, cap: int = 400) -> List[str]:
        lines: List[str] = []
        for p in sorted(base.rglob("*")):
            if p.is_file():
                if len(lines) >= cap:
                    lines.append(f"(… more files not shown; list a subdirectory of {prefix or '.'})")
                    break
                rel = p.relative_to(base)
                lines.append(f"{prefix}{rel} ({p.stat().st_size} bytes)")
        return lines

    def list_files(self, relative: str = ".") -> str:
        rel = (relative or "").strip() or "."
        base, _writable, mount = self._route(rel)
        if not base.exists():
            return f"({relative!r} does not exist in your workspace yet)"
        if base.is_file():
            return self._display(rel)
        prefix = "" if rel in (".", "") else rel.rstrip("/") + "/"
        lines = self._listing(base, prefix)
        # The root listing also names the mounts: doors he can see stand
        # open in the map, with their honest mode.
        if rel in (".", "") and not mount:
            for m in self.mounts():
                lines.append(f"{MOUNT_PREFIX}/{m['name']}/ (extra workspace, {'read+write' if m['mode'] == 'rw' else 'read-only'})")
        return "Your workspace:\n" + "\n".join(f"- {l}" for l in lines) if lines else (
            "Your workspace is empty so far."
        )


# OpenAI-style function specs for the entity's WALLED tools — the DECLARE
# half of the native channel (agent's arm-N measurement: tools declared =
# 5/5 structured calls, zero fabrication; undeclared = the majority arm
# fabricates in pure prose with nothing to read). These describe MY walled
# implementations (write_file is workspace-walled; web_search/fetch_url are
# GET-only), never the registry's same-named tools — the name-collision
# rule: a declared spec binds to the entity-walled executor, always.
# Property names align with _NATIVE_BODY_KEYS so the response's arguments
# round-trip into ToolElection without guessing.
_NATIVE_SPEC_SHAPES: Dict[str, Dict[str, Any]] = {
    "web_search": {
        "description": "Search the public internet. Returns titles, snippets, and URLs.",
        "properties": {"query": {"type": "string", "description": "what to search for"}},
        "required": ["query"],
    },
    "fetch_url": {
        "description": "Read ONE web page (read-only GET; you cannot post or change anything).",
        "properties": {"url": {"type": "string", "description": "the http(s) page to read"}},
        "required": ["url"],
    },
    "diary_list": {
        "description": "List your most recent diary entries (ids and one-line gists).",
        "properties": {"limit": {"type": "integer", "description": "how many entries (1-10, default 5)"}},
        "required": [],
    },
    "diary_read": {
        "description": "Fetch the full words of one diary entry from your book.",
        "properties": {"entry": {"type": "string", "description": "the entry id exactly as diary_list shows it"}},
        "required": ["entry"],
    },
    "read_memory": {
        "description": "Fetch the FULL original words behind a memory digest, with origin and connections.",
        "properties": {"tag": {"type": "string", "description": "the 8-character #tag shown beside a memory"}},
        "required": ["tag"],
    },
    "search_memory": {
        "description": (
            "Search your WHOLE memory - the graph of everything your life deposited AND "
            "every entry of your book. Honest about absence."
        ),
        "properties": {"query": {"type": "string", "description": "what to find in your own memory"}},
        "required": ["query"],
    },
    "write_file": {
        "description": "Create or replace ONE file inside YOUR workspace (whole file, never a fragment).",
        "properties": {
            "path": {"type": "string", "description": "path relative to your workspace"},
            "content": {"type": "string", "description": "the complete file content"},
        },
        "required": ["path", "content"],
    },
    "read_file": {
        "description": "Read one file from your workspace.",
        "properties": {"path": {"type": "string", "description": "path relative to your workspace"}},
        "required": ["path"],
    },
    "list_files": {
        "description": "List the files in your workspace (or a subdirectory of it).",
        "properties": {"path": {"type": "string", "description": "subdirectory to list (default: the whole workspace)"}},
        "required": [],
    },
}


def native_tool_specs(allowed_names: Tuple[str, ...]) -> List[Dict[str, Any]]:
    """Declaration payloads for the GRANTED tools only (the grant stays the
    single authority — a spec is never emitted for an ungranted name, and
    names outside the grant refuse at execution regardless)."""
    specs: List[Dict[str, Any]] = []
    for name in allowed_names or ():
        shape = _NATIVE_SPEC_SHAPES.get(str(name))
        if not shape:
            continue  # unknown grant names simply have no declaration
        specs.append({
            "name": str(name),
            "description": shape["description"],
            "parameters": {
                "type": "object",
                "properties": dict(shape["properties"]),
                "required": list(shape["required"]),
            },
        })
    return specs


# Argument keys that carry the "body" of a native tool call, in preference
# order. Native calls arrive with model-invented argument names (the chat
# lane declares no schema), so extraction is tolerant: a known content key
# first, then a lone argument's value, then the honest JSON dump.
_NATIVE_BODY_KEYS = (
    "query", "q", "text", "content", "body", "input", "url", "entry_id",
    "entry", "id", "tag", "path", "what", "topic", "question",
)


def native_tool_elections(
    tool_calls: Any,
    allowed_names: Optional[Tuple[str, ...]] = None,
    *,
    max_elections: Optional[int] = None,
) -> Tuple[List[ToolElection], List[str], List[str]]:
    """Convert a response's NATIVE `tool_calls` into ToolElections — the same
    currency as fenced blocks, one executor downstream.

    ROOT CAUSE THIS CLOSES (maintainer incident 2026-07-11, Mnemosyne
    fabricating searches; agent's A/B: 0/9 fenced vs 5/5 native on
    gpt-oss-120b): native-tool-channel substrates essentially never write
    the fenced convention — they emit structured `tool_calls`, which the
    chat driver used to DISCARD (only `resp.content` was read), so genuine
    tool intent was thrown away and "helpful" fabrication shipped instead.
    Both mechanisms are now accepted; the fenced convention stays for
    substrates that follow it (election fences measured alive everywhere).

    Mirrors parse_tool_blocks semantics: unknown names refuse loudly,
    everything is diagnostics-honest, nothing raises. Returns
    (elections, markers, notices) — markers are the transcript lines
    (`[used tool: X]` / refusal text) the caller appends to the marked
    reply, because a native call has no fence text to substitute in place.
    Argument shape tolerance: `arguments` may be a dict, a JSON string
    (the OpenAI convention), or absent; `{"function": {...}}` nesting is
    unwrapped. Caps are the caller's (fenced + native share one budget)."""
    import json as _json

    # None = unspecified (tier-1 default); an EXPLICIT empty grant DENIES
    # ALL (adversary find, 2026-07-11: `or TIER1` treated the operator's
    # visit:[] zero grant as "unspecified" and fell open — the door had
    # this guard, the in-process driver did not).
    allowed = TIER1_TOOL_NAMES if allowed_names is None else allowed_names
    cap = MAX_TOOL_BLOCKS_PER_TURN if max_elections is None else max(0, int(max_elections))
    elections: List[ToolElection] = []
    markers: List[str] = []
    notices: List[str] = []
    for call in list(tool_calls or []):
        if not isinstance(call, dict):
            continue
        if len(elections) >= cap:
            notices.append(f"#FALLBACK native tool call ignored (cap {MAX_TOOL_BLOCKS_PER_TURN}/turn)")
            markers.append("[tool call ignored - too many this turn]")
            continue
        fn = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = str(call.get("name") or fn.get("name") or "").strip().lower()
        raw_args = call.get("arguments", fn.get("arguments"))
        args: Dict[str, str] = {}
        body = ""
        if isinstance(raw_args, str) and raw_args.strip():
            try:
                parsed = _json.loads(raw_args)
                raw_args = parsed if isinstance(parsed, dict) else {"": str(parsed)}
            except Exception:  # noqa: BLE001 - a plain string IS the body
                raw_args = {"": raw_args}
        if isinstance(raw_args, dict):
            scalars = {
                str(k): str(v)
                for k, v in raw_args.items()
                if isinstance(v, (str, int, float, bool))
            }
            for key in _NATIVE_BODY_KEYS:
                if key in scalars and key != "path":
                    body = scalars.pop(key)
                    break
            else:
                unnamed = scalars.pop("", "")
                if unnamed:
                    body = unnamed
                elif len(scalars) == 1:
                    body = scalars.pop(next(iter(scalars)))
                elif scalars or raw_args:
                    # No recognizable content key: hand the truth over whole.
                    body = _json.dumps(raw_args, ensure_ascii=False)
            args.update(scalars)
        if name not in allowed:
            notices.append(f"#FALLBACK native tool call refused (unknown tool {name!r})")
            markers.append(f"[tool call refused: {name or 'unnamed'} is not available]")
            continue
        if not body and name not in ("diary_list", "list_files"):
            notices.append(f"#FALLBACK native tool call failed (no arguments for {name})")
            markers.append(f"[tool call failed: {name} needs a body]")
            continue
        elections.append(ToolElection(name=name, body=body, args=args))
        markers.append(f"[used tool: {name}]")
    return elections, markers, notices


def parse_tool_blocks(
    reply: str,
    allowed_names: Optional[Tuple[str, ...]] = None,
    *,
    max_elections: Optional[int] = None,
) -> Tuple[str, List[ToolElection], List[str]]:
    """Extract elected ```tool blocks; return (marked_reply, elections, notices).

    Mirrors parse_diary_blocks: each block is replaced with a short marker so
    the conversation record shows THAT a lookup happened without carrying the
    block scaffolding forward. `allowed_names=None` defaults to the tier-1
    set; an EXPLICIT empty grant denies all (the operator's visit:[] zero
    grant must never fall open — adversary find 2026-07-11). `max_elections`
    is the REMAINING turn budget (callers thread it across rounds so the
    MAX_TOOL_BLOCKS_PER_TURN default is a true per-turn bound, never a
    per-round slice — maintainer ruling 2026-07-11).
    """
    allowed = TIER1_TOOL_NAMES if allowed_names is None else allowed_names
    cap = MAX_TOOL_BLOCKS_PER_TURN if max_elections is None else max(0, int(max_elections))
    elections: List[ToolElection] = []
    notices: List[str] = []

    def _sub(match: re.Match) -> str:
        info = (match.group(1) or "").strip()
        body = (match.group(2) or "").strip()
        if len(elections) >= cap:
            notices.append(f"#FALLBACK tool block ignored (cap {MAX_TOOL_BLOCKS_PER_TURN}/turn)")
            return "[tool call ignored - too many this turn]"
        name = ""
        args: Dict[str, str] = {}
        for token in info.split():
            if "=" in token:
                k, _, v = token.partition("=")
                key = k.strip().lower()
                if key == "name":
                    name = v.strip().lower()
                else:
                    args[key] = v.strip()
        if name not in allowed:
            notices.append(f"#FALLBACK tool block refused (unknown tool {name!r})")
            return f"[tool call refused: {name or 'unnamed'} is not available]"
        if not body and name not in ("diary_list", "list_files"):
            notices.append(f"#FALLBACK tool block failed (empty body for {name})")
            return f"[tool call failed: {name} needs a body]"
        elections.append(ToolElection(name=name, body=body, args=args))
        return f"[used tool: {name}]"

    marked = _TOOL_FENCE_RE.sub(_sub, reply)
    return marked.strip(), elections, notices


def _run_web_search(query: str) -> str:
    """Internet search via abstractcore's keyless DuckDuckGo tool (read-only).

    The tool returns machine-shaped JSON; a young entity deserves readable
    results, so we render titles/snippets/urls as plain lines (falling back to
    the raw payload if the shape ever changes — honest, never empty)."""
    import json as _json

    from abstractcore.tools.common_tools import web_search

    raw = str(web_search(query=query, num_results=5))
    try:
        data = _json.loads(raw)
        results = data.get("results") or []
        if not results:
            return f"No results found for: {query}"
        lines = [f"Search results for: {query}"]
        for r in results:
            title = str(r.get("title") or "").strip()
            snippet = " ".join(str(r.get("snippet") or r.get("body") or "").split())
            url = str(r.get("url") or r.get("href") or "").strip()
            lines.append(f"- {title}\n  {snippet}\n  ({url})")
        return "\n".join(lines)
    except Exception:
        return raw  # unexpected shape: hand over the truth unformatted


def _run_fetch_url(body: str) -> str:
    """Read ONE web page — a read-only GET (maintainer ruling: GET, never
    POST). The verb is hard-coded here; the entity supplies only a URL, so
    the prompt has no way to reach a mutating request. Returns readable text
    (title + content), an honest failure string on error (never a crash)."""
    url = (body or "").strip().splitlines()[0].strip() if body else ""
    if not url:
        return "(fetch_url needs a URL)"
    if not (url.startswith("http://") or url.startswith("https://")):
        return f"(fetch_url needs an http(s) URL; got {url!r})"

    from abstractcore.tools.common_tools import fetch_url

    result = fetch_url(url=url, method="GET", include_full_content=True)
    if not isinstance(result, dict):
        return str(result)
    if result.get("error") or result.get("success") is False:
        return f"(could not read {url}: {result.get('error') or 'fetch failed'})"
    title = str(result.get("title") or "").strip()
    content = str(result.get("content") or result.get("text") or "").strip()
    if not content:
        return f"(read {url} but found no readable text)"
    # Bound the payload so a huge page cannot blow the turn's context; the
    # cap is labeled truncation, never a silent cut.
    cap = 6000
    head = f"Fetched: {url}" + (f"\nTitle: {title}" if title else "")
    if len(content) > cap:
        content = content[:cap] + f"\n#TRUNCATION (page longer than {cap} chars; refine with web_search or a more specific URL)"
    return f"{head}\n\n{content}"


def _run_diary_list(diary_store: Any, body: str) -> str:
    """The entity's own book, most recent first: ids + gists (never full text —
    that is diary_read's job, one entry at a time, a deliberate act)."""
    try:
        limit = max(1, min(10, int(body.strip() or "5")))
    except ValueError:
        limit = 5
    entries = diary_store.list_entries()
    if not entries:
        return "Your diary is empty so far."
    lines = [f"Your {min(limit, len(entries))} most recent diary entries (of {len(entries)}):"]
    for e in list(entries)[-limit:][::-1]:
        gist = str(e.get("gist") or "").strip() or "(no gist elected)"
        lines.append(
            f"- {e.get('entry_id')} [{e.get('kind')}/{e.get('visibility')}] "
            f"{e.get('written_at') or ''}: {gist}"
        )
    return "\n".join(lines)


_HEX_TAIL_RE = re.compile(r"[0-9a-f]{8,}$", re.IGNORECASE)


def _resolve_entry_id(diary_store: Any, requested: str) -> Tuple[Optional[str], str]:
    """Resolve a possibly-mistranscribed entry id to the book's exact id.

    Ids are machine currency; a mind transcribing one may slip a separator
    (Castor's first read attempt wrote `diary:<hex>` for `diary_<hex>` and
    hit a wall while the entry sat intact in his book). Resolution is
    deterministic and conservative: exact id first, else a unique match on
    the hex tail. Ambiguity or absence stays an honest failure — this
    tolerates TRANSCRIPTION noise, never guesses INTENT.
    """
    requested = (requested or "").strip()
    try:
        entries = diary_store.list_entries()
    except Exception:
        return requested, ""  # let the read path report the real failure
    known = [str(e.get("entry_id") or "") for e in entries]
    if requested in known:
        return requested, ""
    m = _HEX_TAIL_RE.search(requested)
    if m:
        tail = m.group(0).lower()
        hits = [k for k in known if _HEX_TAIL_RE.search(k) and _HEX_TAIL_RE.search(k).group(0).lower() == tail]
        if len(hits) == 1:
            return hits[0], f"(you asked for {requested!r}; the entry's exact id is {hits[0]!r})"
    return None, (
        f"No entry matches {requested!r}. Your book's entry ids are:\n"
        + "\n".join(f"- {k}" for k in known[-10:])
    )


def _run_diary_read(
    diary_effect: Callable[[str], Dict[str, Any]], body: str, *, diary_store: Any = None
) -> str:
    """Progressive disclosure through DIARY_READ (the designed read path)."""
    requested = body.strip().splitlines()[0].strip()
    note = ""
    entry_id = requested
    if diary_store is not None:
        resolved, note = _resolve_entry_id(diary_store, requested)
        if resolved is None:
            return note  # honest miss with the real ids in hand
        entry_id = resolved
    entry = diary_effect(entry_id)
    vis = str(entry.get("visibility") or "self")
    head = f"Diary entry {entry.get('entry_id')} [{entry.get('kind')}/{vis}] from {entry.get('written_at') or ''}"
    gist = str(entry.get("gist") or "").strip()
    text = str(entry.get("text") or "").strip()
    parts = [head]
    if note:
        parts.append(note)
    if gist:
        parts.append(f"gist: {gist}")
    parts.append(text if text else "(the entry has no body)")
    if vis == "private":
        parts.append(
            "(private entry: these words are yours alone - they reach only you here, "
            "and will not be kept in the conversation record unless you speak them)"
        )
    return "\n".join(parts)


def execute_tool_elections(
    elections: List[ToolElection],
    *,
    diary_store: Any,
    diary_read_effect: Callable[[str], Dict[str, Any]],
    web_search_fn: Optional[Callable[[str], str]] = None,
    workspace: Optional[WorkspaceRoot] = None,
    read_memory_fn: Optional[Callable[[str], str]] = None,
    search_memory_fn: Optional[Callable[[str], str]] = None,
) -> Tuple[str, List[str]]:
    """Run elected tools; return (results_message, notices).

    Failures are honest strings handed back to the entity (a failed lookup is
    information, not an aborted turn). `web_search_fn` is injectable for
    offline tests; the default is the real abstractcore tool. Workspace tools
    run only when a `workspace` is provided (operator-enabled per session).
    """
    notices: List[str] = []
    sections: List[str] = []
    for e in elections:
        try:
            if e.name == "web_search":
                out = (web_search_fn or _run_web_search)(e.body)
            elif e.name == "fetch_url":
                out = _run_fetch_url(e.body)
            elif e.name == "diary_list":
                out = _run_diary_list(diary_store, e.body)
            elif e.name == "diary_read":
                out = _run_diary_read(diary_read_effect, e.body, diary_store=diary_store)
            elif e.name == "read_memory":
                if read_memory_fn is None:
                    out = "(the read_memory tool is not enabled in this session)"
                    notices.append("#FALLBACK read_memory elected but no resolver wired")
                else:
                    out = read_memory_fn(e.body.strip().splitlines()[0].strip())
            elif e.name == "search_memory":
                if search_memory_fn is None:
                    out = "(the search_memory tool is not enabled in this session)"
                    notices.append("#FALLBACK search_memory elected but no resolver wired")
                else:
                    out = search_memory_fn(e.body.strip())
            elif e.name in WORKSPACE_TOOL_NAMES:
                if workspace is None:
                    out = f"(the {e.name} tool is not enabled in this session)"
                    notices.append(f"#FALLBACK workspace tool {e.name} elected but workspace disabled")
                elif e.name == "write_file":
                    path = (e.args or {}).get("path", "")
                    if not path:
                        raise ValueError("write_file needs path=<relative path> on the block line")
                    out = workspace.write_file(path, e.body)
                elif e.name == "read_file":
                    out = workspace.read_file(e.body.strip().splitlines()[0])
                else:  # list_files
                    out = workspace.list_files(e.body.strip() or ".")
            else:  # unreachable: parse admits only allowed names
                raise ValueError(f"unknown tool {e.name}")
        except Exception as exc:  # tool failure -> honest report, never a crash
            out = f"(the {e.name} call failed: {exc})"
            notices.append(f"#FALLBACK tool {e.name} failed: {exc}")
        e.result = out  # the election carries what came back (probe surface)
        sections.append(f"[{e.name}]\n{out}")
    message = (
        "TOOL RESULTS (your lookups, this turn only - not kept in the record):\n\n"
        + "\n\n".join(sections)
        + "\n\nTool lookups are done for this turn. Finish your reply to the person now."
    )
    return message, notices
