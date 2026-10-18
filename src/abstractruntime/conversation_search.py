"""Keyword search over a runtime's conversations: what was asked, what was answered, and what each
turn made or was given (its artifacts).

One index per runtime plane, in its own SQLite file next to the plane's stores (so it works with
the file and the SQLite run stores alike, and a principal's search can only ever read the plane
it was opened on). Each turn root (a question and its answer, as `_best_effort_session_turns`
builds them for the session replay) is one row; an FTS5 table indexes the question, the answer and
the turn's media, lowercased and without accents, with the `trigram` tokenizer. A turn's media are
the artifacts it made (generated images, voices, videos, music, by their generation prompt) and the
ones it was sent with (attachments, by their file name), each with its kind; an artifact with no
such text (read-aloud audio, media an older producer saved without its prompt, internal artifacts)
is not indexed.

- **exact**: every query word (3+ characters) must appear in the turn, anywhere (a substring:
  "survival" finds "overall survival", "surviv" too); ranked by bm25, the question weighing more
  than the answer;
- **fuzzy** (when exact finds too little): the query's trigrams are OR-ed and a turn is kept when
  it holds at least `FUZZY_MIN` of them, so "survivl" or "assaay" still find their turn;
- words of 1–2 characters are matched with LIKE on the same normalized text.

Results are grouped by session: its best turn, a snippet around the first match, where it matched
(question or answer). The index syncs incrementally from the run store's index: turn roots whose
`updated_at` changed are re-read (by session), within a time budget per call; `sync` reports what
is left, so a first search on a large plane says it is still indexing instead of pretending the
results are complete.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

FUZZY_MIN = 0.6
_SCHEMA_VERSION = "2"
_LOCKS: Dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _fold(c: str) -> str:
    """One character as the index holds it: case folded, a Latin/Greek/Cyrillic accent removed
    ("é" → "e", "ß" → "ss"); the marks other scripts write with (Devanagari vowel signs) are kept."""
    out = []
    for d in unicodedata.normalize("NFKD", c.casefold()):
        if unicodedata.combining(d) and out and ord(out[-1][-1]) < 0x0530:
            continue
        out.append(d)
    return "".join(out)


def normalize(text: str) -> str:
    """Lowercase, accents removed, whitespace folded (the index and the query alike)."""
    return re.sub(r"\s+", " ", "".join(_fold(c) for c in str(text or ""))).strip()


def _normalized_with_map(text: str) -> Tuple[str, List[int]]:
    """`normalize(text)` without folding whitespace, and for each of its characters the index of
    the character of `text` it came from (so a match found in the index's form marks the source)."""
    out: List[str] = []
    origin: List[int] = []
    for i, c in enumerate(text):
        f = _fold(c)
        out.append(f)
        origin.extend([i] * len(f))
    return "".join(out), origin


def _is_word_char(c: str) -> bool:
    return c == "_" or unicodedata.category(c)[0] in ("L", "N", "M")


def _words(query: str) -> List[str]:
    norm = normalize(query)
    words, cur = [], []
    for c in norm:
        if _is_word_char(c):
            cur.append(c)
        elif cur:
            words.append("".join(cur)); cur = []
    if cur:
        words.append("".join(cur))
    return words


def _unspaced(word: str) -> bool:
    """Written without spaces between words (CJK, kana, Thai, Lao, Khmer…): a substring, not a word."""
    return any(0x0E00 <= ord(c) <= 0x0EFF or 0x1780 <= ord(c) <= 0x17FF or 0x3040 <= ord(c) <= 0x30FF
               or 0x3400 <= ord(c) <= 0x9FFF or 0xAC00 <= ord(c) <= 0xD7AF or 0xF900 <= ord(c) <= 0xFAFF for c in word)


def _whole_word(text: str, word: str) -> bool:
    if _unspaced(word):
        return word in text
    i = text.find(word)
    while i >= 0:
        before = text[i - 1] if i > 0 else " "
        after = text[i + len(word)] if i + len(word) < len(text) else " "
        if not _is_word_char(before) and not _is_word_char(after):
            return True
        i = text.find(word, i + 1)
    return False


def _trigrams(word: str) -> List[str]:
    return [word[i : i + 3] for i in range(len(word) - 2)] if len(word) >= 3 else []


def _fts_phrase(s: str) -> str:
    return '"' + s.replace('"', '""') + '"'


@dataclass
class SearchHit:
    session_id: str
    run_id: str
    prompt: str
    snippet: str
    #: Where the query matched: "question", "answer" or "artifact".
    matched_in: str
    #: (start, length) of each match in `snippet`.
    highlights: List[Tuple[int, int]]
    score: float
    created_at: Optional[str]
    updated_at: Optional[str]
    session_kind: Optional[str]
    fuzzy: bool = False
    #: The artifact that matched (matched_in "artifact"): `{artifact_id, run_id, kind, content_type,
    #: name, prompt}`; `run_id` owns it (the gateway's artifact content route is scoped to it).
    artifact: Optional[Dict[str, Any]] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "run_id": self.run_id,
            "prompt": self.prompt,
            "snippet": self.snippet,
            "matched_in": self.matched_in,
            "highlights": [list(h) for h in self.highlights],
            "score": round(self.score, 4),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "session_kind": self.session_kind,
            "fuzzy": self.fuzzy,
            "artifact": self.artifact,
        }


@dataclass
class SyncReport:
    indexed_turns: int = 0
    reindexed_sessions: int = 0
    pending_sessions: int = 0
    total_turns: int = 0
    seconds: float = 0.0
    complete: bool = True
    errors: List[str] = field(default_factory=list)


class ConversationSearch:
    """The search index of one runtime plane (`db_path` lives in that plane's data folder)."""

    #: The session kinds indexed: conversations and discussions (automation sessions, with their
    #: thousands of occurrences, are not what a person searches, and would be re-read on each run).
    INDEXED_KINDS = frozenset({"chat", "discussion"})

    def __init__(self, db_path: Path | str, *, run_store: Any, ledger_store: Any, artifact_store: Any = None,
                 kinds: Optional[Iterable[str]] = None):
        self.kinds = frozenset(kinds) if kinds is not None else self.INDEXED_KINDS
        self.db_path = Path(db_path)
        self.run_store = run_store
        self.ledger_store = ledger_store
        self.artifact_store = artifact_store
        with _LOCKS_GUARD:
            self._lock = _LOCKS.setdefault(str(self.db_path.resolve()), threading.Lock())
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.trigram = True
        with self._connect() as conn:
            self._ensure_schema(conn)

    # -- storage ---------------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.create_function("whole_word", 2, lambda t, w: 1 if _whole_word(t or "", w or "") else 0, deterministic=True)
        return conn

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        row = conn.execute("SELECT value FROM meta WHERE key='schema'").fetchone()
        tokenizer_row = conn.execute("SELECT value FROM meta WHERE key='tokenizer'").fetchone()
        if row is not None and row["value"] == _SCHEMA_VERSION and tokenizer_row is not None:
            self.trigram = tokenizer_row["value"] == "trigram"
            return
        conn.executescript(
            """
            DROP TABLE IF EXISTS turns_fts;
            DROP TABLE IF EXISTS turns;
            DROP TABLE IF EXISTS runs_seen;
            CREATE TABLE turns (
                rowid INTEGER PRIMARY KEY,
                run_id TEXT UNIQUE NOT NULL,
                session_id TEXT NOT NULL,
                session_kind TEXT,
                created_at TEXT,
                updated_at TEXT,
                prompt TEXT NOT NULL DEFAULT '',
                answer TEXT NOT NULL DEFAULT '',
                q TEXT NOT NULL DEFAULT '',
                a TEXT NOT NULL DEFAULT '',
                m TEXT NOT NULL DEFAULT '',
                media TEXT NOT NULL DEFAULT '[]'
            );
            CREATE INDEX turns_session ON turns(session_id);
            CREATE TABLE runs_seen (run_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, updated_at TEXT);
            """
        )
        try:
            conn.execute("CREATE VIRTUAL TABLE turns_fts USING fts5(q, a, m, content='turns', content_rowid='rowid', tokenize='trigram')")
            self.trigram = True
        except sqlite3.OperationalError:
            # An SQLite older than 3.34 has no trigram tokenizer: whole words, with prefixes.
            conn.execute("CREATE VIRTUAL TABLE turns_fts USING fts5(q, a, m, content='turns', content_rowid='rowid', tokenize='unicode61 remove_diacritics 2')")
            self.trigram = False
        conn.executescript(
            """
            CREATE TRIGGER turns_ai AFTER INSERT ON turns BEGIN
                INSERT INTO turns_fts(rowid, q, a, m) VALUES (new.rowid, new.q, new.a, new.m);
            END;
            CREATE TRIGGER turns_ad AFTER DELETE ON turns BEGIN
                INSERT INTO turns_fts(turns_fts, rowid, q, a, m) VALUES ('delete', old.rowid, old.q, old.a, old.m);
            END;
            CREATE TRIGGER turns_au AFTER UPDATE ON turns BEGIN
                INSERT INTO turns_fts(turns_fts, rowid, q, a, m) VALUES ('delete', old.rowid, old.q, old.a, old.m);
                INSERT INTO turns_fts(rowid, q, a, m) VALUES (new.rowid, new.q, new.a, new.m);
            END;
            """
        )
        conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('schema', ?)", (_SCHEMA_VERSION,))
        conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('tokenizer', ?)", ("trigram" if self.trigram else "unicode61",))
        conn.commit()

    # -- sync ------------------------------------------------------------------------------

    def _root_rows(self) -> List[Dict[str, Any]]:
        """Every turn root of the plane, newest first (the run store's index when it has one)."""
        lister = getattr(self.run_store, "list_run_index", None)
        if callable(lister):
            rows = lister(root_only=True, limit=1_000_000)
            return [r for r in rows if isinstance(r, dict)]
        runs = self.run_store.list_runs(limit=1_000_000)
        out = []
        for r in runs:
            if getattr(r, "parent_run_id", None):
                continue
            out.append({"run_id": r.run_id, "session_id": getattr(r, "session_id", None), "updated_at": getattr(r, "updated_at", None),
                        "session_kind": getattr(r, "session_kind", None)})
        return out

    def sync(self, *, budget_s: float = 3.0) -> SyncReport:
        """Index what changed since the last call (newest sessions first), within `budget_s`."""
        from .history_bundle import _best_effort_session_turns

        t0 = time.monotonic()
        report = SyncReport()
        with self._lock, self._connect() as conn:
            seen = {r["run_id"]: r["updated_at"] for r in conn.execute("SELECT run_id, updated_at FROM runs_seen")}
            stale_sessions: List[str] = []
            kinds: Dict[str, Optional[str]] = {}
            changed: Dict[str, List[Tuple[str, Optional[str]]]] = {}
            roots: set = set()
            for row in self._root_rows():
                rid = str(row.get("run_id") or "")
                sid = str(row.get("session_id") or "") or rid
                if not rid or (row.get("session_kind") or "chat") not in self.kinds:
                    continue
                roots.add(rid)
                kinds.setdefault(sid, row.get("session_kind"))
                at = row.get("updated_at")
                at = str(at) if at is not None else None
                if seen.get(rid, "\0") != at:
                    if sid not in changed:
                        stale_sessions.append(sid)
                    changed.setdefault(sid, []).append((rid, at))
            # Runs gone from the store (deleted, or no longer a turn) leave the index.
            gone = [rid for rid in seen if rid not in roots]
            for chunk in range(0, len(gone), 500):
                part = gone[chunk : chunk + 500]
                marks = ",".join("?" * len(part))
                conn.execute(f"DELETE FROM turns WHERE run_id IN ({marks})", part)
                conn.execute(f"DELETE FROM runs_seen WHERE run_id IN ({marks})", part)
            if gone:
                conn.commit()
            for i, sid in enumerate(stale_sessions):
                if time.monotonic() - t0 > budget_s:
                    report.pending_sessions = len(stale_sessions) - i
                    report.complete = False
                    break
                try:
                    turns = _best_effort_session_turns(
                        run_store=self.run_store, ledger_store=self.ledger_store, artifact_store=self.artifact_store,
                        session_id=sid, limit=100_000, include_stats=False, include_artifacts=False,
                    )
                except Exception as exc:  # noqa: BLE001 - one unreadable session never stops the others
                    # Reported, and marked read: it is read again when it changes, not on every search.
                    report.errors.append(f"{sid}: {exc}")
                    for rid, at in changed[sid]:
                        conn.execute("INSERT OR REPLACE INTO runs_seen(run_id, session_id, updated_at) VALUES (?,?,?)", (rid, sid, at))
                    conn.commit()
                    continue
                media = self._session_media(sid, turns)
                kept = [str(t.get("run_id") or "") for t in turns if t.get("run_id")]
                marks = ",".join("?" * len(kept)) or "''"
                conn.execute(f"DELETE FROM turns WHERE session_id=? AND run_id NOT IN ({marks})", [sid, *kept])
                for t in turns:
                    rid = str(t.get("run_id") or "")
                    if not rid:
                        continue
                    prompt = str(t.get("prompt") or "")
                    answer = str(t.get("answer") or "")
                    items = media.get(rid, [])
                    conn.execute(
                        """INSERT INTO turns(run_id, session_id, session_kind, created_at, updated_at, prompt, answer, q, a, m, media)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?)
                           ON CONFLICT(run_id) DO UPDATE SET session_id=excluded.session_id, session_kind=excluded.session_kind,
                             created_at=excluded.created_at, updated_at=excluded.updated_at, prompt=excluded.prompt,
                             answer=excluded.answer, q=excluded.q, a=excluded.a, m=excluded.m, media=excluded.media""",
                        (rid, sid, kinds.get(sid), t.get("created_at"), t.get("updated_at"), prompt, answer,
                         normalize(prompt), normalize(answer), " | ".join(_media_text(i) for i in items), json.dumps(items)),
                    )
                    report.indexed_turns += 1
                for rid, at in changed[sid]:
                    conn.execute("INSERT OR REPLACE INTO runs_seen(run_id, session_id, updated_at) VALUES (?,?,?)", (rid, sid, at))
                report.reindexed_sessions += 1
                conn.commit()
            report.total_turns = int(conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0])
        report.seconds = round(time.monotonic() - t0, 3)
        return report

    def _session_media(self, sid: str, turns: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Each turn's media: the attachments it was sent with, and what it generated (by the run
        that made it, else by when it was made: during the latest turn started before it)."""
        store = self.artifact_store
        if store is None or not turns:
            return {}
        out: Dict[str, List[Dict[str, Any]]] = {}
        order = sorted((str(t.get("created_at") or ""), str(t.get("run_id") or "")) for t in turns if t.get("run_id"))
        ids = {rid for _, rid in order}
        seen: set = set()
        for t in turns:
            for ref in t.get("attachments") or []:
                aid = str((ref or {}).get("$artifact") or (ref or {}).get("artifact_id") or "") if isinstance(ref, dict) else ""
                meta = _metadata(store, aid) if aid else None
                if meta is None or aid in seen:
                    continue
                seen.add(aid)
                entry = _media_entry(meta, kind="attachment", name=(ref or {}).get("filename"))
                if entry["name"]:                                   # what is searched: its file name
                    out.setdefault(str(t["run_id"]), []).append(entry)
        try:
            generated = store.search(session_id=sid, limit=0)
        except Exception:  # noqa: BLE001 - an unreadable catalog leaves the media out, never the turns
            generated = []
        for meta in generated:
            tags = dict(getattr(meta, "tags", None) or {})
            aid = str(getattr(meta, "artifact_id", "") or "")
            if tags.get("kind") != "generated_media" or tags.get("purpose") == "voice_playback" or aid in seen:
                continue
            seen.add(aid)
            entry = _media_entry(meta, kind=str(tags.get("modality") or "") or None)
            if not entry["prompt"]:
                continue        # nothing to find but its kind (older producers kept no prompt; read-aloud audio)
            turn = self._turn_of(str(tags.get("run_id") or getattr(meta, "run_id", "") or ""), ids)
            if turn is None:
                made = str(getattr(meta, "created_at", "") or "")
                before = [rid for at, rid in order if at <= made]
                turn = before[-1] if before else order[-1][1]
            out.setdefault(turn, []).append(entry)
        return out

    def _turn_of(self, run_id: str, turn_ids: set) -> Optional[str]:
        """The turn a run belongs to: itself, or the nearest ancestor that is a turn (4 levels)."""
        rid = run_id
        for _ in range(5):
            if not rid:
                return None
            if rid in turn_ids:
                return rid
            try:
                run = self.run_store.load(rid)
            except Exception:  # noqa: BLE001
                return None
            rid = str(getattr(run, "parent_run_id", "") or "") if run is not None else ""
        return None

    # -- search ----------------------------------------------------------------------------

    def search(self, query: str, *, limit: int = 20, sessions: Optional[Iterable[str]] = None,
               exclude: Optional[Iterable[str]] = None, session_kinds: Optional[Iterable[str]] = None) -> List[SearchHit]:
        """The sessions whose questions or answers match `query`, best first (one hit each): exact
        matches first, then (when they are fewer than `limit`) fuzzy ones.

        `sessions` (only these), `exclude` (never these) and `session_kinds` filter in the query
        itself, so a filter never empties a page that had matches further down."""
        words = _words(query)
        if not words:
            return []
        limit = max(1, min(int(limit), 100))
        scope = _Scope(sessions, exclude, session_kinds)
        with self._connect() as conn:
            exact = self._exact(conn, words, scope, limit)
            hits = sorted(exact.values(), key=lambda h: h.updated_at or "", reverse=True)
            hits = sorted(hits, key=lambda h: -h.score)                    # stable: ties stay newest first
            # A typo: only when nothing matches exactly (a few exact hits are never padded with near ones).
            if not hits and self.trigram:
                fuzzy = list(self._fuzzy(conn, words, scope, limit))
                fuzzy = sorted(sorted(fuzzy, key=lambda h: h.updated_at or "", reverse=True), key=lambda h: -h.score)
                hits += fuzzy[: limit - len(hits)]
        return hits[:limit]

    def _pages(self, conn: sqlite3.Connection, match: Optional[str], shorts: Sequence[str], scope: "_Scope",
               keep: Any, words: List[str], fuzzy: bool, limit: int) -> Dict[str, SearchHit]:
        """Rows in rank order, page after page, until `limit` sessions are found (or none are left)."""
        best: Dict[str, SearchHit] = {}
        page, offset = 400, 0
        while True:
            rows = self._rows(conn, match, shorts, scope, page, offset)
            for r in rows:
                if keep(r):
                    self._take(best, r, words, fuzzy)
            if len(best) >= limit or len(rows) < page or offset > 20_000:
                return best
            offset += page

    def _rows(self, conn: sqlite3.Connection, match: Optional[str], shorts: Sequence[str], scope: "_Scope",
              cap: int, offset: int) -> List[sqlite3.Row]:
        args: List[Any] = []
        if match is not None:
            sql = "SELECT t.*, bm25(turns_fts, 3.0, 1.0, 2.0) AS rank FROM turns t JOIN turns_fts f ON f.rowid = t.rowid WHERE turns_fts MATCH ?"
            args.append(match)
            order = "rank"
        else:
            sql = "SELECT t.*, 0 AS rank FROM turns t WHERE 1=1"
            order = "t.updated_at DESC"
        for w in shorts:
            sql += " AND (whole_word(t.q, ?) OR whole_word(t.a, ?) OR whole_word(t.m, ?))"
            args += [w, w, w]
        where, more = scope.sql()
        sql += where + f" ORDER BY {order} LIMIT ? OFFSET ?"
        return list(conn.execute(sql, [*args, *more, cap, offset]))

    def _exact(self, conn: sqlite3.Connection, words: List[str], scope: "_Scope", limit: int) -> Dict[str, SearchHit]:
        long = [w for w in words if len(w) >= 3]
        short = [w for w in words if len(w) < 3]
        match = " AND ".join(_fts_phrase(w) if self.trigram else _fts_phrase(w) + "*" for w in long) if long else None
        return self._pages(conn, match, short, scope, lambda r: True, words, False, limit)

    def _fuzzy(self, conn: sqlite3.Connection, words: List[str], scope: "_Scope", limit: int) -> List[SearchHit]:
        long = [w for w in words if len(w) >= 3]
        grams = sorted({g for w in long for g in _trigrams(w)})[:60]           # a long query stays a bounded OR
        if not grams:
            return []

        def keep(r: sqlite3.Row) -> bool:
            text = r["q"] + " " + r["a"] + " " + r["m"]
            return all(_near_word(w, text) for w in long)

        return list(self._pages(conn, " OR ".join(_fts_phrase(g) for g in grams), [], scope, keep, words, True, limit).values())

    def _take(self, best: Dict[str, SearchHit], r: sqlite3.Row, words: List[str], fuzzy: bool) -> None:
        sid = r["session_id"]
        long = [w for w in words if len(w) >= 3] or words

        def holds(text: str) -> bool:
            if fuzzy:
                return all(_near_word(w, text) for w in long)
            return all(_whole_word(text, w) if len(w) < 3 else w in text for w in words)

        # Where it matched, in this order: the question, an artifact of the turn, the answer.
        artifact = None
        if holds(r["q"]):
            place, source = "question", r["prompt"]
        else:
            artifact = next((i for i in json.loads(r["media"] or "[]") if holds(_media_text(i))), None)
            if artifact is not None:
                place, source = "artifact", artifact.get("prompt") or artifact.get("name") or artifact.get("kind") or ""
            else:
                place, source = ("answer", r["answer"]) if r["answer"] else ("question", r["prompt"])
        snippet, highlights = _snippet(source, words, fuzzy=fuzzy)
        # bm25 is negative (lower is better); a question match weighs in.
        score = -float(r["rank"] or 0) + (1.0 if place == "question" else 0.0)
        hit = SearchHit(session_id=sid, run_id=r["run_id"], prompt=_first_line(r["prompt"]), snippet=snippet,
                        matched_in=place, highlights=highlights, score=score, created_at=r["created_at"],
                        updated_at=r["updated_at"], session_kind=r["session_kind"], fuzzy=fuzzy, artifact=artifact)
        old = best.get(sid)
        if old is None or hit.score > old.score or (hit.score == old.score and (hit.updated_at or "") > (old.updated_at or "")):
            best[sid] = hit


class _Scope:
    """Which sessions a search may return, as SQL on the `turns` table."""

    def __init__(self, sessions: Optional[Iterable[str]], exclude: Optional[Iterable[str]], kinds: Optional[Iterable[str]]):
        self.only = sorted(set(sessions)) if sessions is not None else None
        self.exclude = sorted(set(exclude or ()))
        self.kinds = sorted(set(kinds)) if kinds is not None else None

    def sql(self) -> Tuple[str, List[Any]]:
        where, args = "", []
        if self.only is not None:
            where += f" AND t.session_id IN ({','.join('?' * len(self.only)) or chr(39) * 2})"
            args += self.only
        if self.exclude:
            where += f" AND t.session_id NOT IN ({','.join('?' * len(self.exclude))})"
            args += self.exclude
        if self.kinds is not None:
            where += f" AND COALESCE(t.session_kind, 'chat') IN ({','.join('?' * len(self.kinds)) or chr(39) * 2})"
            args += self.kinds
        return where, args


def _metadata(store: Any, artifact_id: str) -> Any:
    try:
        return store.get_metadata(artifact_id)
    except Exception:  # noqa: BLE001 - a missing artifact is no media, never an error
        return None


def _media_entry(meta: Any, *, kind: Optional[str] = None, name: Any = None) -> Dict[str, Any]:
    """What the index keeps of an artifact: its kind, name and generation prompt, and where to fetch it."""
    tags = dict(getattr(meta, "tags", None) or {})
    extra = dict(getattr(meta, "metadata", None) or {})
    descriptor = getattr(meta, "descriptor", None)
    generation = dict(getattr(descriptor, "generation", None) or {}) if descriptor is not None else {}
    semantic = str(getattr(descriptor, "semantic_kind", "") or "") if descriptor is not None else ""
    filename = name or extra.get("filename") or tags.get("filename")
    prompt = generation.get("prompt")
    return {
        "artifact_id": str(getattr(meta, "artifact_id", "") or ""),
        "run_id": getattr(meta, "run_id", None),
        "kind": kind or semantic or None,
        "content_type": getattr(meta, "content_type", None),
        "name": str(filename) if filename else None,
        "prompt": str(prompt)[:2000] if isinstance(prompt, str) and prompt.strip() else None,
    }


def _media_text(entry: Dict[str, Any]) -> str:
    return normalize(" ".join(str(entry.get(k) or "") for k in ("kind", "name", "prompt")))


def _near_word(word: str, text: str) -> bool:
    """`text` holds a word close to `word`: one sharing most of its trigrams and not much longer
    ("survival" for "survivl"), so trigrams scattered over a long text never pass for a word."""
    if word in text:
        return True
    grams = _trigrams(word)
    if not grams or not any(g in text for g in grams):
        return False
    for token in re.findall(r"[\w\u0300-\u036f]+", text):
        if abs(len(token) - len(word)) <= max(2, len(word) // 3) and _cover(word, token) >= FUZZY_MIN:
            return True
    return False


def _cover(word: str, text: str) -> float:
    grams = _trigrams(word)
    return sum(g in text for g in grams) / max(1, len(grams)) if grams else (1.0 if word in text else 0.0)


def _first_line(text: str, limit: int = 120) -> str:
    line = next((l.strip() for l in str(text or "").splitlines() if l.strip()), "")
    return line if len(line) <= limit else line[: limit - 1].rstrip() + "…"


def _snippet(source: str, words: List[str], *, fuzzy: bool = False, width: int = 140) -> Tuple[str, List[Tuple[int, int]]]:
    """About `width` characters of `source` around its first match, and each match's place in it
    (start, length) in Unicode scalars, mapped from the index's form back to the source text."""
    flat = re.sub(r"\s+", " ", str(source or "")).strip()
    norm, origin = _normalized_with_map(flat)
    found: List[Tuple[int, int]] = []          # (start, end) in `flat`
    targets = list(words)
    if fuzzy:
        # A word with a typo: mark the source's words that hold most of its trigrams.
        targets = []
        for m in re.finditer(r"\S+", norm):
            token = "".join(c for c in m.group(0) if _is_word_char(c))
            if token and any(len(w) >= 3 and _cover(w, token) >= FUZZY_MIN and _cover(token, w) >= 0.4 for w in words):
                targets.append(token)
    for w in dict.fromkeys(targets):
        i = norm.find(w)
        while i >= 0 and w:
            if len(w) >= 3 or _whole_word(norm[max(0, i - 1) : i + len(w) + 1], w):
                found.append((origin[i], origin[i + len(w) - 1] + 1))
            i = norm.find(w, i + max(1, len(w)))
    first = min((a for a, _ in found), default=0)
    start = max(0, first - width // 3)
    # Start and end on a space when there is one near.
    if start > 0:
        sp = flat.rfind(" ", max(0, start - 20), start)
        start = sp + 1 if sp >= 0 else start
    end = min(len(flat), start + width)
    if end < len(flat):
        sp = flat.find(" ", end - 20, end)
        end = sp if sp > start else end
    lead = "…" if start > 0 else ""
    text = lead + flat[start:end] + ("…" if end < len(flat) else "")
    marks = sorted({(a - start + len(lead), b - a) for a, b in found if a >= start and b <= end})
    return text, marks
