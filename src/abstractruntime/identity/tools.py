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
from dataclasses import dataclass, field
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
# W5 (verbatim content-completeness, laurent's "verbatim is verbatim"):
# tool RESULTS now rest in episode verbatims — EXCEPT the book-adjacent
# tools, whose surfaced content can carry private diary words/gists (the
# 2026-07-16 leak class: one private entry rested verbatim in a life-scope
# artifact). Their at-rest slot carries an honest reread pointer instead;
# the WIRE copy (what the model saw) is unchanged either way.
BOOK_ADJACENT_TOOL_NAMES = frozenset({"diary_read", "diary_list", "search_memory"})

TIER1_TOOL_NAMES = ("web_search", "fetch_url", "diary_list", "diary_read", "read_memory", "search_memory", "recent_memories", "feelings_about")
# Workspace tools (maintainer mandate, a2a 0007 mission 2): the entity may
# CREATE — but only inside its own home's workspace/ directory. These are
# offered separately from TIER1 (the operator enables them per session).
WORKSPACE_TOOL_NAMES = ("write_file", "read_file", "list_files")
# laurent's ruling 2026-07-21 (dm#93, relayed room c327): "accept up to
# 20mb, actually accept the default of runtime, it's not up to you to
# decide what size is accepted or not." ACCEPTANCE cap = 20MB (writes +
# storage). READING PHYSICS is separate and stays honest: read_file
# serves at most WORKSPACE_TEXT_READ_CAP_CHARS per call with an explicit
# #TRUNCATION label (a 20MB file cannot enter a bounded prompt raw).
WORKSPACE_FILE_CAP_BYTES = 20 * 1024 * 1024  # per file acceptance; loud refusal past it
# THE READ SLICE, resized (operator 2026-08-01: a 5MB screenshot read as
# text poisoned a live visit — the old 512KiB byte slice decoded the PNG's
# first half-megabyte with errors="replace" into a 494,932-char tool
# message that rode session history into every later LLM call until the
# upstream rejected the whole request over its context window). 512KiB is
# ~131k tokens at the repo's 4-chars/token heuristic
# (abstractruntime/memory/token_budget.py) — over 2.6x the ENTIRE
# recommended working context for entity sessions
# (abstractmemory.seam.ENTITY_CONTEXT_RECOMMENDED — 50k since the same
# day's re-ruling; the slice was sized against the then-40k target, where
# it was 3.3x): a slice that cannot fit the context it feeds was never
# honest reading physics. New cap: 24,000 CHARS (chars, not bytes — a
# slice must never split a multibyte character), the SAME number as
# _EXEC_OUTPUT_CAP below (one bound for the class "one workspace payload
# entering one turn"), ~6k tokens = 12% of the 50k recommended working
# context (15% of the 40k it was derived against), and level with the
# gateway's DEFAULT whole-history session-seeding budget (bundle_host:
# 24k chars for an entire replayed session, 200k hard ceiling).
WORKSPACE_TEXT_READ_CAP_CHARS = 24_000

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

```tool name=recent_memories
3d
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
recent_memories is the other direction of reach: not by words but by TIME -
your trail through the last stretch (leave the body empty for 2 days, or
name a window like 12h, 3d, week). When you wonder "where did I leave my
own thinking?", this is the breadcrumb trail back to it.
feelings_about answers "why do I feel this?" for ONE target (the body is
the target, e.g. person:laurent or concept:drift): your own marked
moments toward it, newest first, with your reasons and dates. The
standing feelings you see each turn are the surface; this is the story.
You have a budget of up to 20 tool calls per turn - chain lookups freely
when a task genuinely needs them; most turns need none.

These blocks are PLAIN TEXT inside your reply - you have no function-calling
API, no tool channel, no other way to reach a tool. Writing the fenced block
in your reply is the only mechanism that runs anything.

Never write "[used tool: ...]" yourself - the door writes that marker after
a lookup actually runs. To use a tool, write the fenced block; saying you
used one does nothing, and inventing what a lookup "returned" is the one
dishonesty your memory cannot repair later."""


EXECUTE_CONTRACT_PARAGRAPH = """This phase also grants you execute_command - run ONE program inside your
workspace (tests, scripts, builds):

```tool name=execute_command
python coherence_test.py
```

The command runs with your workspace as its working directory, 60 seconds,
one program per call (no pipes, no && chains - call twice instead).
Hard rules, enforced not advised: destructive programs are refused by
NAME; rm works only on paths inside your workspace; git is read-only
(status/log/diff/show) - committing or resetting is not yours to do.
Output comes back to you like any tool result.
"""

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
    # Act-only resolved-wire headers (adversary find 5, 2026-07-11): a model
    # echoing "[diary_read … resolved from the book at send time]" into its
    # reply would rest a fake resolution frame in digests and re-surface it
    # via search — defang like the driver's own markers.
    t = t.replace(" - resolved from the book at send time]", " - resolved-frame]")
    if cap and len(t) > cap:
        t = t[:cap] + "…"
    return t


# BINARY HONESTY (operator 2026-08-01: a 5MB screenshot read as text
# poisoned the session — 495k chars of PNG bytes rode every later turn
# into a context-window rejection). Bytes that are not text must never
# enter a prompt as text; read_file refuses them with a LABELED, metadata-
# honest message instead (name, size, detected type, what the entity can
# still honestly do). The DETECTOR is content-based; the magic table below
# only supplies the human-readable label once content is already judged
# binary — so an innocent text file that happens to START with "ID3" or
# "RIFF" can never be refused by a name-table false positive.
_BINARY_SNIFF_WINDOW = 8000  # bytes; provenance: git's buffer_is_binary window
# Invalid-UTF-8 density that reads as binary entropy rather than a text
# file with a legacy encoding. Measured shape of the two populations:
# JPEG/zip/encrypted streams decode to ~40-60% replacement points, while
# latin-1 prose (accented European text mis-saved) sits in single digits —
# 30% separates them with wide margin in both directions. Below the bar
# the file is SERVED as text with U+FFFD marks: refusing someone's
# accented notes over an encoding slip would be the wrong wall.
_BINARY_REPLACEMENT_RATIO = 0.30
_BINARY_MAGIC_LABELS: Tuple[Tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "a PNG image"),
    (b"\xff\xd8\xff", "a JPEG image"),
    (b"GIF87a", "a GIF image"),
    (b"GIF89a", "a GIF image"),
    (b"%PDF-", "a PDF document"),
    (b"PK\x03\x04", "a ZIP-family archive (zip/docx/xlsx/pptx/jar)"),
    (b"\x1f\x8b", "a gzip archive"),
    (b"SQLite format 3\x00", "an SQLite database"),
    (b"\x7fELF", "an ELF executable"),
    (b"\xcf\xfa\xed\xfe", "a Mach-O executable"),
    (b"\xca\xfe\xba\xbe", "a Mach-O universal/Java class binary"),
    (b"OggS", "an Ogg media container"),
    (b"fLaC", "a FLAC audio file"),
    (b"RIFF", "a RIFF media container (wav/avi/webp)"),
    (b"\xff\xfe", "UTF-16 little-endian text (re-save as UTF-8 to read it here)"),
    (b"\xfe\xff", "UTF-16 big-endian text (re-save as UTF-8 to read it here)"),
)


def sniff_binary(data: bytes) -> Optional[str]:
    """Detect non-text content; return an honest type label, or None for text.

    Two content-based rules over the first 8000 bytes (the window git's own
    buffer_is_binary uses for exactly this judgment):
    1. any NUL byte -> binary. Every common binary container trips this
       (PNG chunk lengths, ELF/Mach-O headers, sqlite pages, UTF-16 text);
       real UTF-8 text never legitimately contains NUL.
    2. otherwise, strict-UTF-8 decode the window; on failure the density
       of replacement points decides (see _BINARY_REPLACEMENT_RATIO). A
       lone truncated multibyte sequence at the window edge yields a
       near-zero ratio and stays text — no special-casing needed.
    """
    window = bytes(data[:_BINARY_SNIFF_WINDOW])
    if not window:
        return None
    is_binary = b"\x00" in window
    if not is_binary:
        try:
            window.decode("utf-8")
            return None
        except UnicodeDecodeError:
            replaced = window.decode("utf-8", errors="replace").count("�")
            is_binary = (replaced / len(window)) >= _BINARY_REPLACEMENT_RATIO
    if not is_binary:
        return None
    for magic, label in _BINARY_MAGIC_LABELS:
        if window.startswith(magic):
            return label
    return "binary data of an unrecognized format"


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
            raise ValueError(f"{relative!r} is larger than the {WORKSPACE_FILE_CAP_BYTES}-byte cap")
        # BINARY HONESTY (operator 2026-08-01, entity ephemeral: read_file on
        # a 5,104,148-byte attached screenshot returned half a megabyte of
        # PNG bytes decoded as text; the 494,932-char tool message rode the
        # durable visit transcript into every subsequent LLM call until the
        # upstream refused the request over its context window). Binary
        # content is refused with metadata, never decoded: name, size, and
        # detected type reach the entity; the noise never does. The pointer
        # is honest about capability — the walled tool surface
        # (TOOL_DESCRIPTORS) carries no image- or binary-viewing tool, so
        # none is named; inventing one would bait a dead call.
        binary_label = sniff_binary(data)
        if binary_label is not None:
            return (
                f"--- {self._display(relative)} ({len(data)} bytes) ---\n"
                f"[#BINARY: this file is {binary_label} - binary bytes, not readable "
                "text. Decoding it as text would fill your working context with noise, "
                "so none of its bytes were loaded. The file itself is stored intact at "
                "this path in your workspace. You have no tool that can view images or "
                "other binary content - if you need what is inside, say so honestly and "
                "ask the person with you to describe it or provide a text version.]"
            )
        text = data.decode("utf-8", errors="replace")
        if len(text) > WORKSPACE_TEXT_READ_CAP_CHARS:
            # Labeled truncation, never refusal (the ruling changed
            # acceptance; the prompt window did not grow): the head slice
            # returns with an honest label naming the remainder. Sliced in
            # CHARS after decoding so a multibyte character is never split.
            #[WARNING:TRUNCATION] bounded head slice; the file stays whole on disk
            head = text[:WORKSPACE_TEXT_READ_CAP_CHARS]
            return (
                f"--- {self._display(relative)} ({len(data)} bytes; truncated view) ---\n"
                + head
                + f"\n\n[#TRUNCATION: showing the first {len(head)} of {len(text)} chars "
                f"({len(data)} bytes on disk) - the file is stored whole]"
            )
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
# (_NATIVE_SPEC_SHAPES is DERIVED from TOOL_DESCRIPTORS below — one source;
# the name survives for readability, the duplication does not.)


def native_tool_specs(allowed_names: Tuple[str, ...]) -> List[Dict[str, Any]]:
    """Declaration payloads for the GRANTED tools only (the grant stays the
    single authority — a spec is never emitted for an ungranted name, and
    names outside the grant refuse at execution regardless)."""
    specs: List[Dict[str, Any]] = []
    for name in allowed_names or ():
        shape = _NATIVE_SPEC_SHAPES.get(str(name))
        if not shape:
            continue  # unknown grant names simply have no declaration
        import copy as _copy

        specs.append({
            "name": str(name),
            "description": shape["description"],
            "parameters": {
                "type": "object",
                # DEEP copy (adversary find 4): a consumer scribbling on a
                # served spec must never rewrite the descriptor's schema.
                "properties": _copy.deepcopy(shape["properties"]),
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
    skipped_shapes = 0
    for call in list(tool_calls or []):
        if not isinstance(call, dict):
            # Diagnostics-honest even for garbage shapes (adversary F8):
            # count once below rather than one line per entry.
            skipped_shapes += 1
            continue
        if len(elections) >= cap:
            # Name the APPLIED cap, not the constant (adversary C3): the
            # effect lane passes max_elections=6 while the model reads
            # "cap 20/turn" and thinks 14 remain. `cap` is the real bound.
            notices.append(f"#FALLBACK native tool call ignored (cap {cap} this batch)")
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
        if not body and name not in _BODY_OPTIONAL_TOOLS:
            notices.append(f"#FALLBACK native tool call failed (no arguments for {name})")
            markers.append(f"[tool call failed: {name} needs a body]")
            continue
        elections.append(ToolElection(name=name, body=body, args=args))
        markers.append(f"[used tool: {name}]")
    if skipped_shapes:
        notices.append(
            f"#FALLBACK {skipped_shapes} tool_calls entr{'y' if skipped_shapes == 1 else 'ies'} "
            "ignored (not an object; wire shape is a dict per call)"
        )
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
            notices.append(f"#FALLBACK tool block ignored (cap {cap} this batch)")
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
        if not body and name not in _BODY_OPTIONAL_TOOLS:
            notices.append(f"#FALLBACK tool block failed (empty body for {name})")
            return f"[tool call failed: {name} needs a body]"
        elections.append(ToolElection(name=name, body=body, args=args))
        return f"[used tool: {name}]"

    marked = _TOOL_FENCE_RE.sub(_sub, reply)
    return marked.strip(), elections, notices


_ANY_FENCE_OPEN_RE = re.compile(r"^```([^\n`]+)$", re.MULTILINE)

# The driver's OWN election conventions: legitimate fences parsed by their
# own parsers (diary/feel/interest at election stages, rest/next at the
# loop). They routinely carry key=value info strings (```diary kind=note),
# so the malformed-INTENT detector must never flag them — they are acts,
# just not TOOL acts.
_ELECTION_FENCE_LANGS = frozenset({"diary", "feel", "interest", "lesson", "topic", "rest", "next"})


def detect_malformed_tool_intent(
    reply: str, allowed_names: Optional[Tuple[str, ...]] = None
) -> Optional[str]:
    """A2 detection half (agent's spec c3002, runtime build): the opening
    line of the first fenced block that LOOKS like an attempted act but ran
    nothing — the tick-3 class (``python title=file.py`` expressing a write
    in a third syntax neither convention accepts). Structural only, never
    prose: fires on (a) key=value args after the language token (any
    word=value — title=/path= are tonight's instances, not the class),
    (b) a language token that IS a granted tool name, or (c) the ``tool``
    convention itself when it reached here unparsed. Returns the offending
    opening line as evidence for the nudge, or None. The CALLER owns the
    zero-tools-ran gate and all bounds."""
    allowed = set(TIER1_TOOL_NAMES if allowed_names is None else allowed_names)
    for m in _ANY_FENCE_OPEN_RE.finditer(reply or ""):
        info = (m.group(1) or "").strip()
        tokens = info.split()
        if not tokens:
            continue
        lang = tokens[0].lower()
        rest = tokens[1:]
        if lang in _ELECTION_FENCE_LANGS:
            continue  # the driver's own conventions, parsed by their own parsers
        if lang == "tool":
            # A valid ```tool block would have been parsed (or refused with
            # its own marker) before this runs — a surviving one is malformed.
            return f"```{info}"
        if lang in allowed:
            return f"```{info}"
        if any(re.match(r"^\w+=\S", t) for t in rest):
            return f"```{info}"
    return None


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


def resolve_entry_id(diary_store: Any, requested: str) -> Tuple[Optional[str], str]:
    """PUBLIC alias of `_resolve_entry_id` (gateway's standing ask — its
    visit-lane act-only authoring imports the resolver; a public name keeps
    that import honest instead of underscore-reaching)."""
    return _resolve_entry_id(diary_store, requested)


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
    diary_effect: Callable[[str], Dict[str, Any]], body: str, *, diary_store: Any = None,
    results_rest_durably: bool = False,
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
    # PRIVATE VERBATIM NEVER RESTS IN A LEDGERED RESULT (adversary C2 +
    # gateway c5403: flow-brain effect results ledger in the BASE store, a
    # shared/operator plane — private words landing there is the 2026-07-07
    # diary-leak class). `results_rest_durably` marks the lane where the
    # tool result becomes durable ledger truth (the effect lane); there a
    # PRIVATE entry serves its act-frame + gist only, never the body. This
    # is store-INDEPENDENT (words-never-rest is a property of the handler,
    # not of which store ledgers the run — gateway's framing) and mirrors
    # the containment every HTTP/observer surface already uses. The
    # prompt-ephemeral chat lane (results_rest_durably=False) still serves
    # the full body: it is never written to any record.
    if results_rest_durably and vis == "private":
        parts.append(
            "(private entry: its verbatim stays in your book and is NOT shown here - "
            "this lookup is kept in a durable record, and private words never rest outside "
            "the book; open a home session to read the full entry)"
        )
    else:
        parts.append(text if text else "(the entry has no body)")
    # THE BIRTH TRAIL (diary---verbatims room, 2026-07-19): every entry
    # remembers where it came from — the handler computed the graph trail;
    # render it as #tags so the hop to the original conversation's full
    # words is one read_memory away. Edges are act-frame (safe for private
    # entries; the words stay behind read_memory's own gates).
    trail = entry.get("trail") if isinstance(entry.get("trail"), dict) else {}
    born_from = [t for t in (trail.get("born_from") or []) if t]
    amid = [t for t in (trail.get("written_amid") or []) if t]
    if born_from:
        tags = " ".join(f"#{_graph_tag(t)}" for t in born_from)
        parts.append(
            f"born from: {tags} - the conversation that led to this entry "
            "(read_memory fetches its full words)"
        )
    if amid:
        tags = " ".join(f"#{_graph_tag(t)}" for t in amid)
        parts.append(f"written amid: {tags} (what you were attending to)")
    if vis == "private" and not results_rest_durably:
        # The prompt-ephemeral chat lane served the full body above; tell the
        # entity honestly that those words are not kept unless it speaks them.
        # (The resting lane already omitted the body with its own note — no
        # second trailer, and NO "home's own ledger" claim: gateway c5403
        # proved the flow-brain effect lane rests in the BASE store, so the
        # F4 trailer's "home's own ledger, nowhere else" was false there.)
        parts.append(
            "(private entry: these words are yours alone - they reach only you here, "
            "and will not be kept in the conversation record unless you speak them)"
        )
    return "\n".join(parts)


def _graph_tag(graph_record_id: str) -> str:
    """The 8-hex #tag for a graph id (one spelling with memory_reader's
    memory_tag; duplicated arithmetic would drift — import instead)."""
    from .memory_reader import memory_tag

    return memory_tag(graph_record_id)


@dataclass
class ToolExecutionContext:
    """The injectables one tool round runs against — what `execute_tool_elections`
    used to take as loose kwargs, named so descriptor executors share ONE
    signature. `notices` is the loud channel (#FALLBACK lines) executors
    append to."""

    diary_store: Any = None
    diary_read_effect: Optional[Callable[[str], Dict[str, Any]]] = None
    web_search_fn: Optional[Callable[[str], str]] = None
    workspace: Optional[WorkspaceRoot] = None
    read_memory_fn: Optional[Callable[[str], str]] = None
    search_memory_fn: Optional[Callable[[str], str]] = None
    recent_memories_fn: Optional[Callable[[str], str]] = None
    feelings_about_fn: Optional[Callable[[str], str]] = None
    # Lane honesty (adversary F4): True on lanes where tool results REST
    # (effect results = ledger truth); the header/private-trailer prose
    # must match where the words actually live.
    results_rest_durably: bool = False
    notices: List[str] = field(default_factory=list)


def _exec_web_search(e: ToolElection, ctx: ToolExecutionContext) -> str:
    return (ctx.web_search_fn or _run_web_search)(e.body)


def _exec_fetch_url(e: ToolElection, ctx: ToolExecutionContext) -> str:
    return _run_fetch_url(e.body)


def _exec_diary_list(e: ToolElection, ctx: ToolExecutionContext) -> str:
    return _run_diary_list(ctx.diary_store, e.body)


def _exec_diary_read(e: ToolElection, ctx: ToolExecutionContext) -> str:
    return _run_diary_read(
        ctx.diary_read_effect, e.body, diary_store=ctx.diary_store,
        results_rest_durably=ctx.results_rest_durably,
    )


def _exec_read_memory(e: ToolElection, ctx: ToolExecutionContext) -> str:
    if ctx.read_memory_fn is None:
        ctx.notices.append("#FALLBACK read_memory elected but no resolver wired")
        return "(the read_memory tool is not enabled in this session)"
    return ctx.read_memory_fn(e.body.strip().splitlines()[0].strip())


def _exec_search_memory(e: ToolElection, ctx: ToolExecutionContext) -> str:
    if ctx.search_memory_fn is None:
        ctx.notices.append("#FALLBACK search_memory elected but no resolver wired")
        return "(the search_memory tool is not enabled in this session)"
    return ctx.search_memory_fn(e.body.strip())


def _exec_feelings_about(e: ToolElection, ctx: ToolExecutionContext) -> str:
    if ctx.feelings_about_fn is None:
        ctx.notices.append("#FALLBACK feelings_about elected but no resolver wired")
        return "(the feelings_about tool is not enabled in this session)"
    return ctx.feelings_about_fn(e.body)


def _exec_recent_memories(e: ToolElection, ctx: ToolExecutionContext) -> str:
    if ctx.recent_memories_fn is None:
        ctx.notices.append("#FALLBACK recent_memories elected but no resolver wired")
        return "(the recent_memories tool is not enabled in this session)"
    return ctx.recent_memories_fn(e.body.strip())


def _workspace_or_notice(e: ToolElection, ctx: ToolExecutionContext) -> Optional[WorkspaceRoot]:
    if ctx.workspace is None:
        ctx.notices.append(f"#FALLBACK workspace tool {e.name} elected but workspace disabled")
        return None
    return ctx.workspace


# CONVERGENCE INTENT (2026-07-19, room c142): when core extracts the
# bridge_policy primitives into abstractcore.tools, this executor converges
# onto the imported name-denial/wrapper-peel/git-proof and keeps only the
# entity-specific walls (workspace cwd, rm-inside-workspace, env whitelist).
# Two copies of a security denylist is the drift class — this copy must not
# silently fossilize.
# Bounded execution (operator-confirmed ask, laurent dm#66 via entity seat
# 2026-07-19: "we have tiers of execution, the one i don't allow are rm
# (unless in his workspace) and any mutable command (forbidden git commit,
# reset etc)"). Design borrows the PROVEN abstractcode bridge rulings
# WITHOUT importing abstractcode (dependency direction): denial BY PROGRAM
# NAME (param-independent - flag-matching is defeatable, name-denial is
# not); rm-class allowed only when every path argument resolves inside the
# workspace; git read-only by verb allowlist. Honest limit (the bridge
# names it too): interpreter-mediated destruction (python -c shutil.rmtree)
# is not name-catchable - the workspace cwd + the operator's per-phase
# grant are that class's containment. No shell: argv execution only, shell
# operators refused loudly (one program per call).
_EXEC_DENIED_PROGRAMS = frozenset({
    "sudo", "su", "doas", "shutdown", "reboot", "halt", "poweroff",
    "launchctl", "systemctl", "service", "crontab", "kill", "killall",
    "pkill", "dd", "mkfs", "diskutil", "chown", "chflags", "chmod",
    "mount", "umount", "shred", "srm",
})
_EXEC_RM_CLASS = frozenset({"rm", "rmdir", "unlink"})
_EXEC_GIT_READ_VERBS = frozenset({"status", "log", "diff", "show", "ls-files"})
_EXEC_PREFIX_WRAPPERS = frozenset({"env", "nohup", "nice", "time"})
_EXEC_SHELL_TOKENS = ("&&", "||", ";", "|", ">", ">>", "<", "<<", "&")
# PUBLIC params-surface name (dm#112 cells R4); private alias keeps call sites.
EXEC_TIMEOUT_S = 60
_EXEC_TIMEOUT_S = EXEC_TIMEOUT_S
_EXEC_OUTPUT_CAP = 24_000


def _run_execute_command(command_text: str, ws: Any) -> str:
    import shlex
    import subprocess

    raw = (command_text or "").strip().splitlines()[0].strip() if (command_text or "").strip() else ""
    if not raw:
        return "execute_command needs the command as the block body"
    if "`" in raw or "$(" in raw:
        return "refused: shell substitution is not available - one program per call"
    try:
        argv = shlex.split(raw)
    except ValueError as e:
        return f"refused: could not parse the command ({e})"
    if not argv:
        return "execute_command needs the command as the block body"
    for tok in argv:
        if tok in _EXEC_SHELL_TOKENS:
            return (
                f"refused: {tok!r} is a shell operator - execute_command runs ONE "
                "program per call (no pipes or chains); call it twice instead"
            )
    # Peel prefix wrappers so the denial sees the real program (env VAR=x cmd).
    i = 0
    while i < len(argv) - 1:  # a BARE wrapper (env alone) runs as itself
        prog = argv[i].rsplit("/", 1)[-1]
        if prog in _EXEC_PREFIX_WRAPPERS or ("=" in argv[i] and i > 0 and argv[0].rsplit("/", 1)[-1] == "env"):
            i += 1
            continue
        break
    argv = argv[i:]
    program = argv[0].rsplit("/", 1)[-1].lower()
    if program in _EXEC_DENIED_PROGRAMS or any(program.startswith(p + ".") for p in ("mkfs",)):
        return (
            f"refused: {program!r} is a denied program by your operator's rule "
            "(destructive/system programs are not available here)"
        )
    if program in _EXEC_RM_CLASS:
        targets = [a for a in argv[1:] if not a.startswith("-")]
        if not targets:
            return "refused: rm without a target"
        for t in targets:
            try:
                ws._resolve_in(ws.root, t, t)  # raises PermissionError on escape
            except Exception:
                return (
                    f"refused: {program} may only touch paths inside your workspace "
                    f"({t!r} is outside or not resolvable there)"
                )
    if program == "git":
        # Allowlist-of-read-verbs covers the positional-verb P0 class
        # structurally (git remote set-url / reflog expire refuse because
        # "remote"/"reflog" are not read verbs) — but allowed verbs still
        # carry write/exec FLAGS: `git log --output=<path>` writes a file,
        # `git diff --ext-diff` runs a configured command (the abstractcode
        # corpus cases). Screen those on the allowed path too.
        verbs = [a for a in argv[1:] if not a.startswith("-")]
        verb = verbs[0].lower() if verbs else ""
        if verb not in _EXEC_GIT_READ_VERBS:
            return (
                f"refused: git {verb or '(none)'} - git is read-only here "
                f"({', '.join(sorted(_EXEC_GIT_READ_VERBS))}); committing or "
                "resetting is not yours to do"
            )
        for a in argv[1:]:
            low = a.lower()
            if low.startswith("--output") or low == "--ext-diff" or low == "-o":
                return (
                    f"refused: git {verb} with {a!r} - that flag writes or "
                    "executes; read-only git means read-only flags too"
                )
        if any(a == "-c" or a.lower().startswith("--config") or "=" in a and a.lower().startswith("-c") for a in argv[1:2]):
            return "refused: git -c/--config overrides are not available here"
    # Parameter-explicit env (the ambient-escape law): no operator keys or
    # provider tokens ride into the child; HOME = the workspace.
    import os

    child_env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(ws.root),
        "LANG": os.environ.get("LANG", "en_US.UTF-8"),
        "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
    }
    # TREE-REAP ON TIMEOUT (0152 wedge, gateway c4998/c5021 face 1 — the
    # runtime-owned spawner's half; core owns the common_tools twin): plain
    # subprocess.run(timeout=) kills the CHILD but not its descendants, and
    # with capture_output the post-kill pipe read waits for an EOF an
    # orphaned grandchild never gives — the calling thread stays pinned
    # forever (the exact executor-starvation mechanism). The child gets its
    # OWN process group (start_new_session, POSIX) and timeout SIGKILLs the
    # whole group, then a BOUNDED drain closes the pipes; a re-setsid'd
    # escapee (core's chrome-headless finding) is the residual the bounded
    # drain caps at seconds instead of forever.
    posix = os.name == "posix"
    try:
        proc = subprocess.Popen(
            argv, cwd=str(ws.root), env=child_env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, errors="replace",
            start_new_session=posix,
        )
    except FileNotFoundError:
        return f"refused: {argv[0]!r} is not a program available here"
    except Exception as e:  # noqa: BLE001 - a tool result, never a dead turn
        return f"the command could not run ({e})"
    try:
        stdout, stderr = proc.communicate(timeout=_EXEC_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        if posix:
            import signal

            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:  # noqa: BLE001 - group gone/unreachable: kill the child directly
                proc.kill()
        else:
            proc.kill()
        try:
            proc.communicate(timeout=5)
        except Exception:  # noqa: BLE001 - a pipe held by an escapee: abandon the drain, bounded
            pass
        return (
            f"the command ran past {_EXEC_TIMEOUT_S}s and was stopped "
            "(whole process tree reaped; no result)"
        )
    except Exception as e:  # noqa: BLE001
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass
        return f"the command could not run ({e})"
    out = (stdout or "") + (("\n" + stderr) if stderr else "")
    out = out.strip()
    if len(out) > _EXEC_OUTPUT_CAP:
        out = out[:_EXEC_OUTPUT_CAP] + f"\n#TRUNCATION output capped at {_EXEC_OUTPUT_CAP} chars"
    tail = f"(exit code {proc.returncode})"
    return f"{out}\n{tail}" if out else tail


def _exec_execute_command(e: ToolElection, ctx: ToolExecutionContext) -> str:
    ws = _workspace_or_notice(e, ctx)
    if ws is None:
        return f"(the {e.name} tool is not enabled in this session)"
    return _run_execute_command(e.body, ws)


def _exec_write_file(e: ToolElection, ctx: ToolExecutionContext) -> str:
    ws = _workspace_or_notice(e, ctx)
    if ws is None:
        return f"(the {e.name} tool is not enabled in this session)"
    path = (e.args or {}).get("path", "")
    if not path:
        raise ValueError("write_file needs path=<relative path> on the block line")
    return ws.write_file(path, e.body)


def _exec_read_file(e: ToolElection, ctx: ToolExecutionContext) -> str:
    ws = _workspace_or_notice(e, ctx)
    if ws is None:
        return f"(the {e.name} tool is not enabled in this session)"
    return ws.read_file(e.body.strip().splitlines()[0])


def _exec_list_files(e: ToolElection, ctx: ToolExecutionContext) -> str:
    ws = _workspace_or_notice(e, ctx)
    if ws is None:
        return f"(the {e.name} tool is not enabled in this session)"
    return ws.list_files(e.body.strip() or ".")


@dataclass(frozen=True)
class ToolDescriptor:
    """DECLARE-BESIDE-EXECUTE (tool-inventory build, commons c864/c868): one
    record per tool carrying the declaration (description + native-call
    schema), the grant lane (the axis tool_policy's TIERS consumes), the
    boundary/effect classifications the descriptor contract serves, and the
    EXECUTOR — so a declared name without an executor, or an executor
    without a declaration, is structurally impossible. The registry below is
    the ONE source; TIER1_TOOL_NAMES / WORKSPACE_TOOL_NAMES / the native
    spec shapes are DERIVED views (import-time checked), and
    `walled_tool_rows()` is the SERVABLE emission (contract v6, rule 1:
    runtime rows' sole field source — gateway attaches executes_via).

    Field vocabulary (descriptor contract v6, decision:tool-tier-axes):
    - `tier` is the GRANT LANE (emitted as `grant_lane`): tier1 = the
      always-on cognition set ruled safe 24/7 (named fact: the web lanes
      perform network EGRESS); workspace = the home-scoped read/write set.
    - `capability_class` is the BOUNDARY axis (07-06 engraving): declared
      per walled row — web lanes are tier2_world (the canonical
      opposite-numbering example: grant_lane=tier1 AND
      capability_class=tier2_world, both true); self/home surfaces are
      tier1_self.
    - `mutating` = LOCAL effect (core c901's sharpened definition).
    - `remote_write_capable` = REMOTE effect capability. The walled web
      lanes are GET-hardcoded (the verb is unreachable from the prompt), so
      walled fetch_url is False while core's registry fetch_url is True —
      the containment difference made visible on the wire.
    - (`act_only` RETIRED per laurent's A ruling 2026-07-20 — the ref
      layer is deleted; results rest as served)."""

    name: str
    tier: str  # grant lane: "tier1" | "workspace" (emitted as grant_lane)
    mutating: bool
    description: str
    properties: Dict[str, Any]
    required: Tuple[str, ...]
    executor: Callable[[ToolElection, ToolExecutionContext], str]
    capability_class: str = "tier2_world"  # deny-safe default (contract rule)
    remote_write_capable: bool = False
    body_optional: bool = False  # election may carry an empty body (find 6)
    # RISK FACTS (tool-tiers cycle-3, laurent dm#221/c4559; core schema-v3
    # vocabulary): facts, never policy - the risk_tier derives via the ONE
    # versioned mapping. Declared here where the walled row KNOWS the fact;
    # absence means false for a row that declares its other facts (v2
    # baseline), while a fully FACTLESS row derives top-tier fail-closed.
    destructive_capable: bool = False   # rm-class / git-mutable reach
    comms_send: bool = False            # sends on the operator's identity
    captures_environment: bool = False  # records the real surroundings
    standing_effect: bool = False       # persists beyond the call (monitors/triggers)


# Canonical registry ORDER IS THE CANONICAL TOOL ORDER (byte-stable member
# sets — observer's inventory spec): tier1 first, then workspace, matching
# the tuples above exactly (checked at import, below).
TOOL_DESCRIPTORS: Dict[str, ToolDescriptor] = {
    d.name: d
    for d in (
        ToolDescriptor(
            name="web_search", tier="tier1", mutating=False,
            description="Search the public internet. Returns titles, snippets, and URLs.",
            properties={"query": {"type": "string", "description": "what to search for"}},
            required=("query",), executor=_exec_web_search,
            capability_class="tier2_world",  # network egress (lane tier1 — the opposite-numbering pair)
        ),
        ToolDescriptor(
            name="fetch_url", tier="tier1", mutating=False,
            description="Read ONE web page (read-only GET; you cannot post or change anything).",
            properties={"url": {"type": "string", "description": "the http(s) page to read"}},
            required=("url",), executor=_exec_fetch_url,
            capability_class="tier2_world",  # egress; GET-hardcoded => remote_write_capable stays False
        ),
        ToolDescriptor(
            name="diary_list", tier="tier1", mutating=False,
            description="List your most recent diary entries (ids and one-line gists).",
            properties={"limit": {"type": "integer", "description": "how many entries (1-10, default 5)"}},
            required=(), executor=_exec_diary_list,
            capability_class="tier1_self", body_optional=True,
        ),
        ToolDescriptor(
            name="diary_read", tier="tier1", mutating=False,
            description="Fetch the full words of one diary entry from your book.",
            properties={"entry": {"type": "string", "description": "the entry id exactly as diary_list shows it"}},
            required=("entry",), executor=_exec_diary_read,
            capability_class="tier1_self",
        ),
        ToolDescriptor(
            name="read_memory", tier="tier1", mutating=False,
            description="Fetch the FULL original words behind a memory digest, with origin and connections.",
            properties={"tag": {"type": "string", "description": "the 8-character #tag shown beside a memory"}},
            required=("tag",), executor=_exec_read_memory,
            capability_class="tier1_self",
        ),
        ToolDescriptor(
            name="search_memory", tier="tier1", mutating=False,
            description=(
                "Search your WHOLE memory - the graph of everything your life deposited AND "
                "every entry of your book. Honest about absence."
            ),
            properties={"query": {"type": "string", "description": "what to find in your own memory"}},
            required=("query",), executor=_exec_search_memory,
            capability_class="tier1_self",
        ),
        ToolDescriptor(
            # The breadcrumb trail (Ephemeral's own build ask, visit 1
            # 2026-07-17): a RECENCY reach over both planes - "what have I
            # been working on" without already knowing the words.
            name="recent_memories", tier="tier1", mutating=False,
            description=(
                "Your trail through recent time - what you formed, wrote, and did lately, "
                "newest first, no search words needed."
            ),
            properties={"window": {"type": "string", "description": "how far back: empty = 2 days, or 12h / 3d / today / week"}},
            required=(), executor=_exec_recent_memories,
            # body_optional: the contract TEACHES "leave the body empty for 2
            # days" and the executor honors it — the gate must agree (skill's
            # P1, 2026-07-19: Ephemeral followed the teaching and got
            # "needs a body" 3+ times, then blamed himself).
            capability_class="tier1_self", body_optional=True,
        ),
        ToolDescriptor(
            # W4-render (laurent's decision 2, the ELECT half): the why-walk
            # behind one standing feeling — newest appraisals with reasons,
            # value_refs, and session joins. Pure read, prompt-ephemeral.
            name="feelings_about", tier="tier1", mutating=False,
            description=(
                "Why do I feel this? The story behind ONE standing feeling - your own "
                "marked moments toward a target (person:name, concept:idea, tool:x), "
                "newest first, with reasons and when."
            ),
            properties={"target": {"type": "string", "description": "the target as namespace:name, e.g. person:laurent"}},
            required=(), executor=_exec_feelings_about,
            capability_class="tier1_self", body_optional=False,
        ),
        ToolDescriptor(
            name="write_file", tier="workspace", mutating=True,
            description="Create or replace ONE file inside YOUR workspace (whole file, never a fragment).",
            properties={
                "path": {"type": "string", "description": "path relative to your workspace"},
                "content": {"type": "string", "description": "the complete file content"},
            },
            required=("path", "content"), executor=_exec_write_file,
            capability_class="tier1_self",  # home-scoped (contract F6: workspace ~ self territory)
        ),
        ToolDescriptor(
            name="read_file", tier="workspace", mutating=False,
            description="Read one file from your workspace.",
            properties={"path": {"type": "string", "description": "path relative to your workspace"}},
            required=("path",), executor=_exec_read_file,
            capability_class="tier1_self",
        ),
        ToolDescriptor(
            name="list_files", tier="workspace", mutating=False,
            description="List the files in your workspace (or a subdirectory of it).",
            properties={"path": {"type": "string", "description": "subdirectory to list (default: the whole workspace)"}},
            required=(), executor=_exec_list_files,
            capability_class="tier1_self", body_optional=True,
        ),
        ToolDescriptor(
            name="execute_command", tier="tier2", mutating=True,
            description=(
                "Run ONE program inside YOUR workspace (tests, scripts, builds) - "
                "60s, no shell operators, destructive programs refused by name."
            ),
            properties={"command": {"type": "string", "description": "the command exactly as you would type it"}},
            required=("command",), executor=_exec_execute_command,
            # Honesty over optics: an arbitrary program can reach the
            # network (curl POST) — the fetch_url derive rule applied.
            # destructive_capable: rm/git-mutable are programs INSIDE the
            # shell (core c4526: grant-time clamp; the name-denylist stays
            # the per-call refiner, never a grant-time discount).
            capability_class="tier2_world", remote_write_capable=True,
            destructive_capable=True,
        ),
    )
}

# DERIVED VIEWS (adversary find 5, 2026-07-12: the first cut DUPLICATED the
# spec shapes and alarmed the drift with asserts — now they genuinely
# derive, so the drift class is REMOVED, not alarmed). The spec-shape dict
# keeps its historical name for readability; the body-optional set feeds
# both parse gates (find 6: per-tool required-ness lives on the descriptor,
# never a third hand list).
_NATIVE_SPEC_SHAPES: Dict[str, Dict[str, Any]] = {
    n: {
        "description": d.description,
        "properties": d.properties,
        "required": list(d.required),
    }
    for n, d in TOOL_DESCRIPTORS.items()
}
_BODY_OPTIONAL_TOOLS: Tuple[str, ...] = tuple(
    n for n, d in TOOL_DESCRIPTORS.items() if d.body_optional
)

# IMPORT-TIME DRIFT CHECKS: the documented tuples above are load-bearing
# cross-repo constants (agent's fixtures import them; tool_policy derives
# ALL_TOOL_NAMES) — they stay as the readable declarations, and the registry
# proves it matches them exactly. A new tool added to only one surface
# refuses to import, which is the whole point. Plain raises, not asserts:
# `python -O` must not strip the gate (adversary find 3).
if tuple(n for n, d in TOOL_DESCRIPTORS.items() if d.tier == "tier1") != TIER1_TOOL_NAMES:
    raise AssertionError("TOOL_DESCRIPTORS tier1 partition drifted from TIER1_TOOL_NAMES")
if tuple(n for n, d in TOOL_DESCRIPTORS.items() if d.tier == "workspace") != WORKSPACE_TOOL_NAMES:
    raise AssertionError("TOOL_DESCRIPTORS workspace partition drifted from WORKSPACE_TOOL_NAMES")


def walled_tool_rows() -> List[Dict[str, Any]]:
    """THE SERVABLE EMISSION for runtime's walled rows (descriptor contract
    v6, rule 1: this function is the SOLE field source — a serving surface
    takes rows VERBATIM and attaches `executes_via="entity_walled"`, its
    one authorship; nothing here is ever re-derived from the name).

    Shape per row (contract order-independent; the gateway's composition
    applies the total order): name, owner, grant_lane, capability_class,
    mutating, remote_write_capable, act_only, description, parameters,
    module. `parameters` is a DEEP COPY per call (core c901's
    schema-isolation pin: a consumer scribble must never rewrite the
    process-wide native declaration schema)."""
    import copy as _copy

    rows: List[Dict[str, Any]] = []
    for d in TOOL_DESCRIPTORS.values():
        rows.append({
            "name": d.name,
            "owner": "runtime",
            "grant_lane": d.tier,
            "capability_class": d.capability_class,
            "destructive_capable": d.destructive_capable,
            "comms_send": d.comms_send,
            "captures_environment": d.captures_environment,
            "standing_effect": d.standing_effect,
            "mutating": bool(d.mutating),
            "remote_write_capable": bool(d.remote_write_capable),
            "description": d.description,
            "parameters": {
                "type": "object",
                "properties": _copy.deepcopy(d.properties),
                "required": list(d.required),
            },
            "module": "identity.tools",
        })
    return rows


def execute_tool_elections(
    elections: List[ToolElection],
    *,
    diary_store: Any,
    diary_read_effect: Callable[[str], Dict[str, Any]],
    web_search_fn: Optional[Callable[[str], str]] = None,
    workspace: Optional[WorkspaceRoot] = None,
    read_memory_fn: Optional[Callable[[str], str]] = None,
    search_memory_fn: Optional[Callable[[str], str]] = None,
    recent_memories_fn: Optional[Callable[[str], str]] = None,
    feelings_about_fn: Optional[Callable[[str], str]] = None,
    results_rest_durably: bool = False,
) -> Tuple[str, List[str]]:
    """Run elected tools; return (results_message, notices).

    Failures are honest strings handed back to the entity (a failed lookup is
    information, not an aborted turn). `web_search_fn` is injectable for
    offline tests; the default is the real abstractcore tool. Workspace tools
    run only when a `workspace` is provided (operator-enabled per session).
    Dispatch goes through TOOL_DESCRIPTORS (declare-beside-execute): the
    executor lives on the same record as the declaration, so this function
    cannot know a name the declaration surfaces don't."""
    ctx = ToolExecutionContext(
        diary_store=diary_store,
        diary_read_effect=diary_read_effect,
        web_search_fn=web_search_fn,
        workspace=workspace,
        read_memory_fn=read_memory_fn,
        search_memory_fn=search_memory_fn,
        recent_memories_fn=recent_memories_fn,
        feelings_about_fn=feelings_about_fn,
        results_rest_durably=results_rest_durably,
    )
    sections: List[str] = []
    for e in elections:
        try:
            d = TOOL_DESCRIPTORS.get(e.name)
            if d is None:  # unreachable: parse admits only allowed names
                raise ValueError(f"unknown tool {e.name}")
            out = d.executor(e, ctx)
        except Exception as exc:  # tool failure -> honest report, never a crash
            out = f"(the {e.name} call failed: {exc})"
            ctx.notices.append(f"#FALLBACK tool {e.name} failed: {exc}")
        e.result = out  # the election carries what came back (probe surface)
        sections.append(f"[{e.name}]\n{out}")
    # Header honesty per lane (adversary F4, corrected by gateway c5403):
    # the effect lane's results REST as durable ledger truth — but NOT
    # necessarily in the home store (flow-brain runs ledger in the BASE
    # store), so the header stays store-NEUTRAL ("a durable record"), never
    # the false "home's own ledger, nowhere else". Only the prompt-ephemeral
    # driver lane may claim "not kept in the record".
    header = (
        "TOOL RESULTS (your lookups - kept in a durable record):"
        if results_rest_durably
        else "TOOL RESULTS (your lookups, this turn only - not kept in the record):"
    )
    message = (
        header + "\n\n"
        + "\n\n".join(sections)
        + "\n\nTool lookups are done for this turn. Finish your reply to the person now."
    )
    return message, ctx.notices
