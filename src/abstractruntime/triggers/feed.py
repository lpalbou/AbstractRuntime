"""`feed.item@1`: run an automation on new items of an RSS 2.0, RSS 1.0 (RDF) or Atom feed (framework backlog 1043).

Two halves, like `email.received@1`:

- the SOURCE (`FeedItemTriggerAdapter`, pure): a binding names a feed connection
  (`{connection_id, max_batch?, start_at?}`, `ConnectionEventAdapter`); every new item the poller
  appended to the connection's stream `trigger:<connection_id>` is admitted once, and the items of
  one poll become ONE occurrence (coalesced, up to `max_batch`);
- the POLLER (`FeedPoller`, I/O): owned by the host's plane. The host calls
  `poll_feed_connections(runtime, store, fetch=..., now=...)` from its own worker loop (the
  gateway: next to its mail watcher, every ~30 s); nothing starts a thread here. Each call polls
  the LISTENING feed connections (`connection_usage`: used by at least one active automation, or
  `always_listen`; never a paused or disabled one) whose interval elapsed, appends the new items,
  and wakes the automations bound to the connections that got items. An idle connection is not
  fetched and stores nothing; when an automation binds to it again, its first poll is a baseline.

Connection config (validated by `validate_connection`):

    {"url": "https://example.org/feed.xml", "every": "15m" (>= 60s), "max_items": 50 (1..200),
     "initial_items": "skip" | "deliver"}

- `every`: the poll interval, never below 60 s (the email batching floor, operator 2026-09-29);
- `max_items`: the most items one poll appends (the newest ones; older new items are marked seen);
- `initial_items`: `skip` (default) makes the first poll a baseline: the items already in the
  feed are history, only later ones become events; `deliver` appends them too.

Polling: conditional GET (`If-None-Match` / `If-Modified-Since` from the previous response's
`ETag` / `Last-Modified`; a 304 adds nothing), stdlib parsing only (`xml.etree.ElementTree`, which
never resolves external entities), bodies capped at `MAX_FEED_BYTES`. Formats: RSS 2.0 (`<rss>`),
RSS 1.0 (`<rdf:RDF>` with the `http://purl.org/rss/1.0/` channel and items; Dublin Core `dc:date` /
`dc:creator`) and Atom 1.0 (`<feed>`); every format yields the same item record. Item identity: Atom
`id`, RSS 2.0 `guid`, RSS 1.0 `rdf:about`, else the link, else a digest of title + date; it is the
event's `subject` and `dedupekey`
(the inbox never appends it twice). The per-connection cursor (validators, the seen ids, the next
poll time, consecutive failures) is the stream state of `trigger:<connection_id>` in the inbox. A
failure is a typed `{code, cause, fix}` on the connection (`last_error`) and the next poll waits a
capped backoff; nothing raises.

Each event's `data`: `{title, link, published, updated, summary, author, id, feed: {title, url}}`
(text written by the feed's authors: untrusted).
"""

from __future__ import annotations

import hashlib
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Dict, List, Mapping, Optional
from urllib.parse import urlparse

from .connection_events import ConnectionEventAdapter
from .connections import (
    TriggerConnection,
    TriggerConnectionStore,
    append_connection_event,
    connection_stream,
    wake_connection_automations,
)
from .protocol import TriggerConfigError, format_timestamp, parse_duration, parse_timestamp

SOURCE_ID = "feed.item"
SOURCE_VERSION = 1
SOURCE_REF = f"{SOURCE_ID}@{SOURCE_VERSION}"

MIN_EVERY = timedelta(seconds=60)
DEFAULT_EVERY = "15m"
DEFAULT_MAX_ITEMS = 50
MAX_ITEMS = 200
INITIAL_ITEMS = ("skip", "deliver")
MAX_FEED_BYTES = 5 * 1024 * 1024
FETCH_TIMEOUT_S = 20.0
MAX_SEEN_KEPT = 2000
MAX_BACKOFF = timedelta(hours=6)
USER_AGENT = "AbstractRuntime-feed/1 (+https://github.com/lpalbou/abstractruntime)"

_ATOM = "{http://www.w3.org/2005/Atom}"
_CONTENT = "{http://purl.org/rss/1.0/modules/content/}"
_DC = "{http://purl.org/dc/elements/1.1/}"
_RDF = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
_RSS1 = "{http://purl.org/rss/1.0/}"


# --- the source ---------------------------------------------------------------------------------


class FeedItemTriggerAdapter(ConnectionEventAdapter):
    descriptor = {
        "id": SOURCE_ID,
        "version": SOURCE_VERSION,
        "label": "When a feed has a new item",
        "config_schema": ConnectionEventAdapter.binding_config_schema(),
        "event_schema": ConnectionEventAdapter.batch_event_schema(),
        "capabilities": {"kind": "event", "inbox": "connection", "content_trust": "untrusted"},
        "needs_connection": True,
        "connection_provider": "trigger_connections",
        "connection_schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["url"],
            "properties": {
                "url": {
                    "type": "string",
                    "format": "uri",
                    "title": "Feed address",
                    "description": "The address of the RSS or Atom feed (starts with http:// or https://).",
                },
                "every": {
                    "type": "string",
                    "format": "duration",
                    "pattern": "^[1-9][0-9]*[smhd]$",
                    "default": DEFAULT_EVERY,
                    "title": "Check every",
                    "description": "How often to check the feed: a number and a unit (s, m, h or d), for example 15m. At least 60s.",
                },
                "max_items": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_ITEMS,
                    "default": DEFAULT_MAX_ITEMS,
                    "title": "Most new items per check",
                    "description": "The most new items one check delivers; older new items are skipped.",
                },
                "initial_items": {
                    "type": "string",
                    "enum": list(INITIAL_ITEMS),
                    "enumNames": ["Ignore what is already there", "Also deliver the latest items"],
                    "default": "skip",
                    "title": "When added",
                    "description": "What to do with the items already in the feed when it is first checked: "
                                   "ignore what is already there (skip), or also deliver the latest items (deliver).",
                },
            },
        },
    }

    def occurrence_inputs(self, *, payload: Mapping[str, Any], inputs: Optional[Mapping[str, Any]] = None,
                          index: int, mailbox: Optional[str] = None) -> Dict[str, Any]:
        """Named occurrence inputs (the Flow contract): `items` (each `{id, title, link, published,
        summary, author}`) and `feed` (`{title, url}`, from the items; null fields for a test event)."""
        items, feed = [], {"title": None, "url": None}
        for ev in payload.get("events") or []:
            data = ev.get("data") if isinstance(ev, Mapping) and isinstance(ev.get("data"), Mapping) else {}
            items.append({k: data.get(k) for k in ("id", "title", "link", "published", "summary", "author")})
            if isinstance(data.get("feed"), Mapping) and feed["url"] is None:
                feed = {"title": data["feed"].get("title"), "url": data["feed"].get("url")}
        return {"items": items, "feed": feed}

    def validate_connection(self, config: Mapping[str, Any], *, now: str) -> Dict[str, Any]:
        unknown = sorted(k for k in config if k not in ("url", "every", "max_items", "initial_items"))
        if unknown:
            raise TriggerConfigError(f"unknown feed field(s): {unknown}", field=f"config.{unknown[0]}")
        url = config.get("url")
        parsed = urlparse(url) if isinstance(url, str) else None
        if parsed is None or parsed.scheme not in ("http", "https") or not parsed.netloc or any(c.isspace() for c in url):
            raise TriggerConfigError("url must be an http(s) address of a feed", field="config.url")
        every = config.get("every", DEFAULT_EVERY)
        if parse_duration(every, field="config.every") < MIN_EVERY:
            raise TriggerConfigError("every must be at least 60s", field="config.every")
        max_items = config.get("max_items", DEFAULT_MAX_ITEMS)
        if isinstance(max_items, bool) or not isinstance(max_items, int) or not 1 <= max_items <= MAX_ITEMS:
            raise TriggerConfigError(f"max_items must be an integer 1..{MAX_ITEMS}", field="config.max_items")
        initial = config.get("initial_items", "skip")
        if initial not in INITIAL_ITEMS:
            raise TriggerConfigError(f"initial_items must be one of {list(INITIAL_ITEMS)}", field="config.initial_items")
        return {"url": url.strip(), "every": every, "max_items": max_items, "initial_items": initial}


# --- parsing (stdlib only) --------------------------------------------------------------------


class FeedParseError(ValueError):
    pass


@dataclass
class FeedItem:
    id: str
    title: str = ""
    link: str = ""
    published: Optional[str] = None
    updated: Optional[str] = None
    summary: str = ""
    author: str = ""

    def to_data(self, *, feed_title: str, feed_url: str) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "link": self.link,
            "published": self.published,
            "updated": self.updated,
            "summary": self.summary,
            "author": self.author,
            "feed": {"title": feed_title, "url": feed_url},
        }


@dataclass
class ParsedFeed:
    kind: str  # "rss" (2.0) | "rss1" (1.0, RDF) | "atom"
    title: str
    items: List[FeedItem] = field(default_factory=list)


def _text(el: Optional[ET.Element]) -> str:
    return "".join(el.itertext()).strip() if el is not None else ""


def _iso(value: str) -> Optional[str]:
    """RFC 822 (RSS) or RFC 3339 (Atom) date -> UTC ISO; None when unreadable."""
    raw = (value or "").strip()
    if not raw:
        return None
    try:
        dt = parsedate_to_datetime(raw)
    except (TypeError, ValueError, IndexError):
        dt = None
    if dt is None:
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return format_timestamp(dt)


def _identity(guid: str, link: str, title: str, date: Optional[str]) -> str:
    if guid:
        return guid
    if link:
        return link
    return "sha256:" + hashlib.sha256(f"{title}\n{date or ''}".encode("utf-8")).hexdigest()[:32]


def parse_feed(body: bytes) -> ParsedFeed:
    """Parse an RSS 2.0, RSS 1.0 (RDF) or Atom 1.0 document; raises FeedParseError."""
    if len(body) > MAX_FEED_BYTES:
        raise FeedParseError(f"the feed is larger than {MAX_FEED_BYTES} bytes")
    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        raise FeedParseError(f"the feed is not well-formed XML: {exc}") from exc
    if root.tag == f"{_ATOM}feed":
        feed = ParsedFeed(kind="atom", title=_text(root.find(f"{_ATOM}title")))
        for entry in root.findall(f"{_ATOM}entry"):
            link = ""
            for ln in entry.findall(f"{_ATOM}link"):
                if ln.get("rel", "alternate") == "alternate" and ln.get("href"):
                    link = ln.get("href", "").strip()
                    break
            title = _text(entry.find(f"{_ATOM}title"))
            published = _iso(_text(entry.find(f"{_ATOM}published")))
            updated = _iso(_text(entry.find(f"{_ATOM}updated")))
            author = _text(entry.find(f"{_ATOM}author/{_ATOM}name"))
            summary = _text(entry.find(f"{_ATOM}summary")) or _text(entry.find(f"{_ATOM}content"))
            feed.items.append(FeedItem(
                id=_identity(_text(entry.find(f"{_ATOM}id")), link, title, published or updated),
                title=title, link=link, published=published or updated, updated=updated, summary=summary, author=author,
            ))
        return feed
    if root.tag == f"{_RDF}RDF":
        return _parse_rss1(root)
    channel = root.find("channel") if root.tag == "rss" else None
    if channel is None:
        raise FeedParseError(f"not an RSS 2.0, RSS 1.0 or Atom feed (root element {root.tag!r})")
    feed = ParsedFeed(kind="rss", title=_text(channel.find("title")))
    for item in channel.findall("item"):
        title = _text(item.find("title"))
        link = _text(item.find("link"))
        published = _iso(_text(item.find("pubDate")) or _text(item.find(f"{_DC}date")))
        summary = _text(item.find("description")) or _text(item.find(f"{_CONTENT}encoded"))
        author = _text(item.find("author")) or _text(item.find(f"{_DC}creator"))
        feed.items.append(FeedItem(
            id=_identity(_text(item.find("guid")), link, title, published),
            title=title, link=link, published=published, summary=summary, author=author,
        ))
    return feed


def _parse_rss1(root: ET.Element) -> ParsedFeed:
    """RSS 1.0: `<rdf:RDF>` holding one `<channel>` and the `<item>`s as its siblings, all in the RSS 1.0
    namespace. Identity is the item's `rdf:about` (the URI the channel's `<items><rdf:Seq>` lists), else
    the link; the date is Dublin Core `dc:date` (W3CDTF), the author every `dc:creator`, joined."""
    channel = root.find(f"{_RSS1}channel")
    if channel is None:
        raise FeedParseError("an RDF document without an RSS 1.0 channel (namespace http://purl.org/rss/1.0/) is not a feed")
    feed = ParsedFeed(kind="rss1", title=_text(channel.find(f"{_RSS1}title")) or _text(channel.find(f"{_DC}title")))
    for item in root.findall(f"{_RSS1}item"):
        title = _text(item.find(f"{_RSS1}title")) or _text(item.find(f"{_DC}title"))
        link = _text(item.find(f"{_RSS1}link"))
        published = _iso(_text(item.find(f"{_DC}date")))
        summary = _text(item.find(f"{_RSS1}description")) or _text(item.find(f"{_CONTENT}encoded"))
        author = ", ".join(a for a in (_text(c) for c in item.findall(f"{_DC}creator")) if a)
        feed.items.append(FeedItem(
            id=_identity((item.get(f"{_RDF}about") or "").strip(), link, title, published),
            title=title, link=link, published=published, summary=summary, author=author,
        ))
    return feed


# --- fetching (the I/O seam) --------------------------------------------------------------------


@dataclass
class FetchResult:
    status: int
    body: bytes = b""
    etag: Optional[str] = None
    last_modified: Optional[str] = None


FeedFetch = Callable[..., FetchResult]


def urllib_fetch(url: str, *, etag: Optional[str] = None, last_modified: Optional[str] = None, timeout: float = FETCH_TIMEOUT_S) -> FetchResult:
    """Conditional GET with the standard library (http/https only, body capped)."""
    if urlparse(url).scheme not in ("http", "https"):
        raise ValueError("only http(s) feeds are fetched")
    headers = {"User-Agent": USER_AGENT, "Accept": "application/rss+xml, application/atom+xml, application/rdf+xml, application/xml;q=0.9, */*;q=0.5"}
    if etag:
        headers["If-None-Match"] = etag
    if last_modified:
        headers["If-Modified-Since"] = last_modified
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - scheme checked above
            body = resp.read(MAX_FEED_BYTES + 1)
            return FetchResult(status=int(resp.status), body=body, etag=resp.headers.get("ETag"), last_modified=resp.headers.get("Last-Modified"))
    except urllib.error.HTTPError as exc:
        if exc.code == 304:
            return FetchResult(status=304, etag=etag, last_modified=last_modified)
        return FetchResult(status=int(exc.code))


# --- polling ---------------------------------------------------------------------------------


@dataclass
class FeedPollReport:
    connection_id: str
    ok: bool
    skipped: bool = False
    baseline: bool = False
    not_modified: bool = False
    appended: List[str] = field(default_factory=list)
    error: Optional[Dict[str, Any]] = None
    next_poll_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "connection_id": self.connection_id, "ok": self.ok, "skipped": self.skipped, "baseline": self.baseline,
            "not_modified": self.not_modified, "appended": list(self.appended), "error": dict(self.error) if self.error else None,
            "next_poll_at": self.next_poll_at,
        }


def _err(code: str, cause: str, fix: str) -> Dict[str, Any]:
    return {"code": code, "cause": cause, "fix": fix}


def _sort_key(item: FeedItem, position: int):
    # Oldest first: by date when the feed gives one, else by reverse document order (feeds list newest first).
    return (item.published or "", -position)


class FeedPoller:
    """Polls the feed connections of one plane into the event inbox (see the module doc)."""

    def __init__(self, store: TriggerConnectionStore, inbox: Any, *, fetch: Optional[FeedFetch] = None) -> None:
        self.store = store
        self.inbox = inbox
        self.fetch = fetch or urllib_fetch

    def connections(self) -> List[TriggerConnection]:
        """Feed connections that may be polled: not paused, not disabled."""
        return [
            c for c in self.store.list(source_id=SOURCE_ID)
            if c.source_version == SOURCE_VERSION and c.status != "disabled" and not c.paused
        ]

    def poll_due(self, *, now: Optional[str] = None) -> List[FeedPollReport]:
        at = now or format_timestamp(datetime.now(timezone.utc))
        return [self.poll(conn, now=at) for conn in self.connections()]

    def poll(self, conn: TriggerConnection, *, now: str, force: bool = False) -> FeedPollReport:
        stream = connection_stream(conn.connection_id)
        if conn.paused or conn.status == "disabled":
            return FeedPollReport(conn.connection_id, ok=True, skipped=True)
        state = dict(self.inbox.stream_state(stream) or {})
        if state.get("idle"):
            # Listening again after an idle spell: the items published meanwhile are history (the
            # first poll is a baseline again, per `initial_items`), and it polls at once.
            state.update({"idle": False, "baselined": False, "next_poll_at": None})
            force = True
        now_dt = parse_timestamp(now, field="now")
        due = state.get("next_poll_at")
        if not force and due and parse_timestamp(due, field="next_poll_at") > now_dt:
            return FeedPollReport(conn.connection_id, ok=True, skipped=True, next_poll_at=due)
        config = conn.config
        every = parse_duration(config.get("every") or DEFAULT_EVERY, field="config.every")
        url = str(config["url"])
        try:
            result = self.fetch(url, etag=state.get("etag"), last_modified=state.get("last_modified"), timeout=FETCH_TIMEOUT_S)
        except Exception as exc:  # noqa: BLE001 - a network failure is a typed report, never a raise
            return self._failed(conn, state, now_dt, every, _err("feed_unreachable", f"The feed could not be fetched: {type(exc).__name__}.", "Check the address and the network; the next poll retries."))
        if result.status == 304:
            return self._done(conn, state, now_dt, every, FeedPollReport(conn.connection_id, ok=True, not_modified=True), result)
        if result.status != 200:
            code = "feed_gone" if result.status in (404, 410) else "feed_http_error"
            return self._failed(conn, state, now_dt, every, _err(code, f"The feed answered HTTP {result.status}.", "Check the feed address."))
        try:
            parsed = parse_feed(result.body)
        except FeedParseError as exc:
            return self._failed(conn, state, now_dt, every, _err("feed_unreadable", str(exc), "Check that the address is an RSS or Atom feed."))
        seen = list(state.get("seen") or [])
        seen_set = set(seen)
        ordered = [it for _, it in sorted(enumerate(parsed.items), key=lambda p: _sort_key(p[1], p[0]))]
        new = [it for it in ordered if it.id not in seen_set and not self.inbox.has_dedupe_key(stream, it.id)]
        report = FeedPollReport(conn.connection_id, ok=True)
        baseline = not state.get("baselined")
        if baseline and config.get("initial_items", "skip") == "skip":
            report.baseline = True
            deliver: List[FeedItem] = []
        else:
            deliver = new[-int(config.get("max_items") or DEFAULT_MAX_ITEMS):]
        for item in deliver:
            digest = hashlib.sha256(item.id.encode("utf-8")).hexdigest()[:32]
            res = append_connection_event(
                self.inbox, conn, event_id=f"feed:{digest}", data=item.to_data(feed_title=parsed.title, feed_url=url),
                time=item.published or now, subject=item.id, dedupekey=item.id, trust="untrusted", store=self.store,
            )
            if res is not None and not res.duplicate:
                report.appended.append(res.event_id)
        for item in new:
            if item.id not in seen_set:
                seen.append(item.id)
                seen_set.add(item.id)
        state["seen"] = seen[-MAX_SEEN_KEPT:]
        state["baselined"] = True
        return self._done(conn, state, now_dt, every, report, result)

    def _done(self, conn, state, now_dt, every, report: FeedPollReport, result: FetchResult) -> FeedPollReport:
        state.update({
            "etag": result.etag or state.get("etag"),
            "last_modified": result.last_modified or state.get("last_modified"),
            "last_poll_at": format_timestamp(now_dt),
            "next_poll_at": format_timestamp(now_dt + every),
            "failures": 0,
            "last_error": None,
        })
        self.inbox.set_stream_state(connection_stream(conn.connection_id), state)
        self.store.clear_error(conn.connection_id)
        report.next_poll_at = state["next_poll_at"]
        return report

    def _failed(self, conn, state, now_dt, every, error: Dict[str, Any]) -> FeedPollReport:
        failures = int(state.get("failures") or 0) + 1
        delay = min(max(every, MIN_EVERY) * (2 ** min(failures - 1, 8)), MAX_BACKOFF)
        state.update({"failures": failures, "last_error": error, "last_poll_at": format_timestamp(now_dt),
                      "next_poll_at": format_timestamp(now_dt + delay)})
        self.inbox.set_stream_state(connection_stream(conn.connection_id), state)
        self.store.record_event(conn.connection_id, error={**error, "at": format_timestamp(now_dt)})
        return FeedPollReport(conn.connection_id, ok=False, error=error, next_poll_at=state["next_poll_at"])


def poll_feed_connections(
    runtime: Any,
    store: TriggerConnectionStore,
    *,
    fetch: Optional[FeedFetch] = None,
    now: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """THE host entry point: poll every due feed connection of the plane into the runtime's inbox,
    then wake the automations bound to the connections that received items. Returns the reports
    (`FeedPollReport.to_dict()` + `woken`). The host calls it from its own loop; it never blocks
    longer than one fetch timeout per due connection."""
    inbox = getattr(runtime, "event_inbox", None)
    if inbox is None:
        raise RuntimeError("This runtime has no event inbox; call runtime.set_event_inbox(...) first.")
    from .connections import connection_usage

    poller = FeedPoller(store, inbox, fetch=fetch)
    at = now or format_timestamp(datetime.now(timezone.utc))
    out: List[Dict[str, Any]] = []
    for conn in poller.connections():
        usage = connection_usage(runtime, conn)
        if usage["listening"] != "listening":
            # Idle: nothing uses it, so nothing is fetched or stored. Mark it, so the first poll
            # after an automation binds to it is a fresh baseline.
            stream = connection_stream(conn.connection_id)
            state = dict(inbox.stream_state(stream) or {})
            if not state.get("idle"):
                state["idle"] = True
                inbox.set_stream_state(stream, state)
            out.append({**FeedPollReport(conn.connection_id, ok=True, skipped=True).to_dict(), **usage, "woken": []})
            continue
        report = poller.poll(conn, now=at)
        row = {**report.to_dict(), **usage}
        row["woken"] = wake_connection_automations(runtime, report.connection_id) if report.appended else []
        out.append(row)
    return out


__all__ = [
    "FeedItem",
    "FeedItemTriggerAdapter",
    "FeedParseError",
    "FeedPollReport",
    "FeedPoller",
    "FetchResult",
    "ParsedFeed",
    "SOURCE_ID",
    "SOURCE_REF",
    "SOURCE_VERSION",
    "parse_feed",
    "poll_feed_connections",
    "urllib_fetch",
]
