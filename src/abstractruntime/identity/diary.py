"""Entity diary: the voluntary, entity-elected episodic record (phase 0).

Design (a2a thread 0003, dual-plane resolution):

- The diary is DISTINCT from the involuntary memory graph. The graph is what
  happens TO the entity (host formation policy fires MEMORY_FORM per turn);
  the diary is what the entity ELECTS to remember — inner thoughts, research
  ideas, goals, notes-to-self. Voluntariness is a property of the CALLER
  CHANNEL, never the payload: `DIARY_WRITE` has a handler ONLY on the entity's
  home runtime (workplaces fail loudly with "no effect handler registered"),
  and the handler factory is BOUND to one entity at construction time — the
  author is never read from the payload.

- Dual-plane residence:
  * The CHAIN (here) is the truth plane: a per-entity hash-chained append-only
    ledger (`HashChainedLedgerStore`) keyed by the synthetic run id
    `diary:<entity_id>` — one entity, one chain, one head. Full first-person
    prose lives here, never purged: `DiaryStore` simply does not expose a
    delete surface (025's deletability is capability-detected, so absence IS
    the guarantee).
  * The PROJECTION (optional, via the abstractmemory seam) is the findability
    plane: a `kind="diary"` digest record so entries index into channels and
    conduct spreading (serendipitous re-encounter). Chain-append is
    truth-first; a failed projection degrades with a labeled #FALLBACK and is
    repairable — never a lost entry.

- Words-as-re-entry: every entry captures its re-entry key AT WRITE TIME
  (`as_of_seq` — the memory journal high-water mark of the writing moment;
  `anchor_record_ids` — what the entity was attending to; `receipts` —
  workplace ledger span refs). Entries written without the key are permanently
  un-reconnectable, so the schema carries it from day one even though the
  re-entry tool ships later.

- Replay safety: the entry id derives from (run_id, turn_id, sha256(text)) —
  the same content-aware pattern as MEMORY_FORM — and `append_entry` dedups on
  it, so the runtime's at-least-once effect execution cannot duplicate chain
  entries.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..core.models import Effect, EffectType, RunState, StepRecord, StepStatus
from ..core.runtime import EffectHandler, EffectOutcome, normalize_utc_iso, utc_now_iso
from ..storage.base import LedgerStore
from ..storage.ledger_chain import HashChainedLedgerStore

_DIARY_NODE_ID = "DIARY"
_VISIBILITIES = ("self", "private")


def diary_chain_id(entity_id: str) -> str:
    """The synthetic run id keying an entity's diary chain."""
    return f"diary:{entity_id}"


def derive_entry_id(*, run_id: str, turn_id: str, text: str) -> str:
    """Content-aware entry id: at-least-once replays of the same volitional act
    dedup, while distinct entries in the same turn write independently."""
    text_fp = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    basis = f"{run_id}|{turn_id}|{text_fp}"
    return "diary_" + hashlib.sha256(basis.encode("utf-8")).hexdigest()[:24]


@dataclass
class DiaryEntry:
    """One elected diary entry (the JSON-safe chain payload)."""

    entry_id: str
    author: str  # entity id — bound by the handler factory, never payload-supplied
    text: str  # full first-person prose, verbatim, never truncated
    gist: Optional[str] = None  # entity-authored digest for the projection (no machine truncation)
    kind: str = "note"  # note | idea | reflection | commitment | ... (open vocabulary)
    visibility: str = "self"  # self = chain + projection; private = chain only

    # Home receipt time vs the writer's claimed time (claim vs stamp).
    written_at: str = ""
    written_at_claim: Optional[str] = None

    # Re-entry key (captured at write time; see module docstring).
    as_of_seq: Optional[int] = None
    anchor_record_ids: List[str] = field(default_factory=list)
    # GRAPH ids of the same anchors (edge currency; anchor_record_ids are
    # digest-row ids — the two-namespace gotcha). The projection writes
    # written_amid edges from these so the diary is CONNECTED (maintainer:
    # "the diary is connected to none other memory and that is not ok").
    anchor_graph_ids: List[str] = field(default_factory=list)
    receipts: List[Dict[str, Any]] = field(default_factory=list)

    # Message-to-future-self: normalized UTC deadline the home scheduler can
    # honor (wired to the heartbeat later; captured from day one).
    remind_at: Optional[str] = None

    # Entity-elected resolution (a2a 0009 identity-card convention): the
    # entry_id of an open question this entry answers. The card's "resolved
    # questions" read joins on this.
    resolves: Optional[str] = None

    # Provenance of the volitional act (which run/turn elected this).
    origin: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "entry_id": self.entry_id,
            "author": self.author,
            "text": self.text,
            "kind": self.kind,
            "visibility": self.visibility,
            "written_at": self.written_at,
        }
        if self.gist:
            out["gist"] = self.gist
        if self.written_at_claim:
            out["written_at_claim"] = self.written_at_claim
        if self.as_of_seq is not None:
            out["as_of_seq"] = self.as_of_seq
        if self.anchor_record_ids:
            out["anchor_record_ids"] = list(self.anchor_record_ids)
        if self.anchor_graph_ids:
            out["anchor_graph_ids"] = list(self.anchor_graph_ids)
        if self.receipts:
            out["receipts"] = list(self.receipts)
        if self.remind_at:
            out["remind_at"] = self.remind_at
        if self.resolves:
            out["resolves"] = self.resolves
        if self.origin:
            out["origin"] = dict(self.origin)
        return out


class DiaryStore:
    """Append-only, hash-chained, structurally non-deletable diary of ONE entity.

    Entries ride `StepRecord` (step_id = entry_id, result = entry dict) so the
    existing ledger substrate provides persistence, ordering, and — through
    `HashChainedLedgerStore` — tamper evidence (`prev_hash`/`record_hash`;
    `signature` stays reserved for the 008 key work).

    There is deliberately NO delete/purge surface here: never-purge is a
    property of the type, not a policy check.
    """

    def __init__(self, *, entity_id: str, ledger_store: LedgerStore):
        eid = str(entity_id or "").strip()
        if not eid:
            raise ValueError("DiaryStore requires a non-empty entity_id")
        self._entity_id = eid
        self._chain_id = diary_chain_id(eid)
        if isinstance(ledger_store, HashChainedLedgerStore):
            self._ledger = ledger_store
        else:
            self._ledger = HashChainedLedgerStore(ledger_store)

    @property
    def entity_id(self) -> str:
        return self._entity_id

    @property
    def chain_id(self) -> str:
        return self._chain_id

    def get_entry(self, entry_id: str) -> Optional[Dict[str, Any]]:
        rec = self.get_record(entry_id)
        return dict(rec.get("result") or {}) if rec else None

    def get_record(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """Full chain record (entry + prev_hash/record_hash) for one entry."""
        for rec in self._ledger.list(self._chain_id):
            if rec.get("step_id") == entry_id:
                return dict(rec)
        return None

    def append_entry(self, entry: DiaryEntry) -> Dict[str, Any]:
        """Append one entry; idempotent by entry_id (replays return the stored
        entry unchanged). Linear pre-scan is acceptable at diary cadence
        (entries are elected, not per-turn); index when a real diary outgrows it.
        """
        existing = self.get_entry(entry.entry_id)
        if existing is not None:
            return {"entry": existing, "replayed": True}

        record = StepRecord(
            run_id=self._chain_id,
            step_id=entry.entry_id,
            node_id=_DIARY_NODE_ID,
            status=StepStatus.COMPLETED,
            result=entry.to_dict(),
            started_at=entry.written_at,
            ended_at=entry.written_at,
            actor_id=self._entity_id,
            session_id=str(entry.origin.get("session_id")) if entry.origin.get("session_id") else None,
            idempotency_key=entry.entry_id,
        )
        self._ledger.append(record)
        return {"entry": entry.to_dict(), "replayed": False, "record_hash": record.record_hash}

    def list_entries(
        self,
        *,
        kind: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Entries in chain order (oldest first), with optional filters.
        `since`/`until` compare against the home-stamped `written_at` (UTC)."""
        out: List[Dict[str, Any]] = []
        for rec in self._ledger.list(self._chain_id):
            entry = rec.get("result")
            if not isinstance(entry, dict):
                continue
            if kind is not None and entry.get("kind") != kind:
                continue
            written = str(entry.get("written_at") or "")
            if since is not None and written < since:
                continue
            if until is not None and written > until:
                continue
            out.append(dict(entry))
        if limit is not None and limit >= 0:
            out = out[-limit:]
        return out

    def head(self) -> Optional[str]:
        records = self._ledger.list(self._chain_id)
        return records[-1].get("record_hash") if records else None


def _coerce_receipts(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    out: List[Dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            out.append({str(k): v for k, v in item.items()})
    return out


def build_diary_effect_handlers(
    *,
    entity_id: str,
    diary_store: DiaryStore,
    memory_system: Any = None,
    now_iso: Callable[[], str] = utc_now_iso,
) -> Dict[EffectType, EffectHandler]:
    """Handlers for the entity diary. Register ONLY on the entity's home runtime.

    The factory binds the author: `entity_id` is stamped on every entry and the
    payload cannot override it — combined with home-only registration this makes
    only-entity-writes structural (a workplace runtime has no handler at all and
    the effect fails loudly there).

    `memory_system` (optional, duck-typed `remember_many`) enables the graph
    projection plane; entries with `visibility="private"` never project.
    """
    eid = str(entity_id or "").strip()
    if not eid:
        raise ValueError("build_diary_effect_handlers requires a non-empty entity_id")
    if diary_store.entity_id != eid:
        raise ValueError(
            f"diary_store is bound to entity {diary_store.entity_id!r}, not {eid!r} — "
            "one diary, one entity, one chain"
        )

    def _project(entry: DiaryEntry) -> tuple[Optional[str], List[str]]:
        """Project the MEMORY OF THE ACT into the graph (findability plane).

        The graph involuntarily records that the act happened — "I wrote
        about X in my diary" — never a second copy of the prose; private
        entries project act-only, content-free. The mechanics live in the
        abstractmemory integration (`identity_support.project_diary_entry`),
        imported lazily so the identity KERNEL carries no optional-stack
        imports (install-boundary contract); chain-only diaries
        (memory_system=None) never touch that path.

        Returns (record_id, warnings); failure degrades, never blocks the
        chain."""
        if memory_system is None:
            return None, []
        from ..integrations.abstractmemory.identity_support import project_diary_entry

        return project_diary_entry(
            memory_system,
            entity_id=eid,
            entry_id=entry.entry_id,
            kind=entry.kind,
            visibility=entry.visibility,
            gist=entry.gist,
            written_at=entry.written_at,
            turn_id=str(entry.origin.get("turn_id") or ""),
            origin=dict(entry.origin),
            anchor_graph_ids=list(entry.anchor_graph_ids),
            resolves=entry.resolves,
        )

    def _handle_diary_write(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del default_next_node
        payload = dict(effect.payload or {})

        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            return EffectOutcome.failed("DIARY_WRITE requires payload.text (the entry, verbatim)")
        text = text.strip()

        turn_id = str(payload.get("turn_id") or "").strip()
        if not turn_id:
            return EffectOutcome.failed(
                "DIARY_WRITE requires payload.turn_id (it derives the entry id that makes "
                "at-least-once replays safe against the append-only chain)"
            )

        visibility = str(payload.get("visibility") or "self").strip().lower()
        if visibility not in _VISIBILITIES:
            return EffectOutcome.failed(f"DIARY_WRITE visibility must be one of {_VISIBILITIES}, got {visibility!r}")

        kind = str(payload.get("kind") or "note").strip().lower() or "note"

        remind_at_raw = payload.get("remind_at")
        remind_at: Optional[str] = None
        if remind_at_raw is not None:
            remind_at = normalize_utc_iso(remind_at_raw)
            if remind_at is None:
                # A silently dropped reminder is a broken promise to a future
                # self; a malformed one mis-orders scheduler due-ness. Loud.
                return EffectOutcome.failed(
                    f"DIARY_WRITE remind_at is not a valid ISO timestamp: {remind_at_raw!r}"
                )

        as_of_seq_raw = payload.get("as_of_seq")
        as_of_seq: Optional[int] = None
        if as_of_seq_raw is not None and not isinstance(as_of_seq_raw, bool):
            try:
                as_of_seq = int(as_of_seq_raw)
            except (TypeError, ValueError):
                return EffectOutcome.failed(f"DIARY_WRITE as_of_seq must be an integer, got {as_of_seq_raw!r}")

        anchors = payload.get("anchor_record_ids")
        anchor_ids = [str(a).strip() for a in anchors if str(a).strip()] if isinstance(anchors, (list, tuple)) else []
        graph_anchors = payload.get("anchor_graph_ids")
        anchor_gids = (
            [str(a).strip() for a in graph_anchors if str(a).strip()]
            if isinstance(graph_anchors, (list, tuple))
            else []
        )

        gist_raw = payload.get("gist")
        gist = gist_raw.strip() if isinstance(gist_raw, str) and gist_raw.strip() else None

        run_id = str(getattr(run, "run_id", "") or "")
        entry = DiaryEntry(
            entry_id=derive_entry_id(run_id=run_id, turn_id=turn_id, text=text),
            author=eid,
            text=text,
            gist=gist,
            kind=kind,
            visibility=visibility,
            written_at=now_iso(),
            written_at_claim=str(payload.get("written_at")) if payload.get("written_at") else None,
            as_of_seq=as_of_seq,
            anchor_record_ids=anchor_ids,
            anchor_graph_ids=anchor_gids,
            receipts=_coerce_receipts(payload.get("receipts")),
            remind_at=remind_at,
            resolves=(str(payload.get("resolves")).strip() or None) if payload.get("resolves") else None,
            origin={
                "run_id": run_id or None,
                "turn_id": turn_id,
                "session_id": getattr(run, "session_id", None),
                "actor_id": getattr(run, "actor_id", None),
            },
        )

        try:
            appended = diary_store.append_entry(entry)
        except Exception as e:
            return EffectOutcome.failed(f"DIARY_WRITE chain append failed: {e}")

        replayed = bool(appended.get("replayed"))
        stored = appended.get("entry") or entry.to_dict()

        # The act-memory projects UNCONDITIONALLY (involuntary — the entity
        # cannot opt out of the trace; private strips content, not the record).
        # Projection is idempotent memory-side (supplied-id dedup), so
        # re-projecting on a replayed chain entry repairs an earlier
        # projection failure instead of duplicating.
        projected_record_id, warnings = _project(entry)

        result: Dict[str, Any] = {
            "entry_id": str(stored.get("entry_id") or entry.entry_id),
            "chain_id": diary_store.chain_id,
            "written_at": stored.get("written_at"),
            "kind": stored.get("kind"),
            "visibility": stored.get("visibility"),
            "replayed": replayed,
        }
        if projected_record_id:
            result["projected_record_id"] = projected_record_id
        if stored.get("remind_at"):
            result["remind_at"] = stored.get("remind_at")
        if warnings:
            result["warnings"] = warnings
        return EffectOutcome.completed(result)

    def _handle_diary_read(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        """Progressive disclosure: fetch the verbatim entry from the book.

        "I remember I wrote X... what was it again?" — recall surfaced the
        memory of the act (gist + entry_id); this fetches the words. The
        result carries a pre-shaped `re_entry` block (the entry's re-entry
        key as a MEMORY_RECALL payload) so the optional third disclosure step
        — re-lighting the past trajectory — is one effect away.

        Reading is pure consumption: it deposits NOTHING graph-side
        (commit_selection stays the only strengthening path).
        """
        del run, default_next_node
        payload = dict(effect.payload or {})
        entry_id = str(payload.get("entry_id") or "").strip()
        if not entry_id:
            return EffectOutcome.failed("DIARY_READ requires payload.entry_id (the book's chain entry id)")

        entry = diary_store.get_entry(entry_id)
        if entry is None:
            return EffectOutcome.failed(
                f"DIARY_READ: no entry {entry_id!r} in {diary_store.chain_id} — "
                "the memory of the act references a book entry that does not exist (chain integrity issue?)"
            )

        result: Dict[str, Any] = dict(entry)
        result["chain_id"] = diary_store.chain_id
        re_entry: Dict[str, Any] = {"cue_text": entry.get("text") or ""}
        if entry.get("as_of_seq") is not None:
            re_entry["as_of"] = entry.get("as_of_seq")
        if entry.get("anchor_record_ids"):
            re_entry["anchor_record_ids"] = list(entry.get("anchor_record_ids") or [])
        result["re_entry"] = re_entry
        return EffectOutcome.completed(result)

    return {
        EffectType.DIARY_WRITE: _handle_diary_write,
        EffectType.DIARY_READ: _handle_diary_read,
    }


def verify_diary_chain(diary_store: DiaryStore) -> Dict[str, Any]:
    """Verify the diary's hash chain (tamper evidence). Delegates to the shared
    ledger-chain verifier; the diary adds no bespoke integrity rules."""
    from ..storage.ledger_chain import verify_ledger_chain

    records = diary_store._ledger.list(diary_store.chain_id)  # noqa: SLF001 - integrity check is a diary concern
    return verify_ledger_chain(records)
