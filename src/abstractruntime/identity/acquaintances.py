"""Acquaintances: whom an entity (or a session) has interacted with — DERIVED, never stored.

Two lists, computed from records the runtime and the home already keep (nothing is written; nothing goes into
`identity.json`): the other ENTITIES and the HUMANS it interacted with, each with when it first and last met
them, how many interactions, and on what kind of task (`track_record.task_kind_of`).

Where an interaction is read (identifiers only — no text of any run, message or task is ever read):

- a run's door-stamped participants, `vars._runtime.entity.participants` (`person:<user>`, `entity:<slug>`,
  …): the gateway stamps them from the authenticated principal at visit open, at summon, and on both legs
  of an entity-to-entity meet (the other entity and the convener). One run = one interaction with each
  participant other than the subject itself;
- a run a person started, `vars._meta.origin` with kind `person`: its `actor` (`<tenant>:<user>`);
- a task left in `<home>/task_inbox.jsonl`: its `added` event's `by` (`person:<user>`), kind
  `work/task_inbox`.

Runs read: turn roots that are not reserved runtime runs (`track_record.is_counted_candidate`), whatever
their origin — a visit is an interaction even when no origin was recorded for it. Sub-runs are never read
(they carry their root's stamp; counting them would count one visit many times).

Not counted, and counted in `not_counted` so the answer says so: a participant of another namespace
(`agent:<name>`, a shared-room agent: neither an entity nor a human), a participant with no namespace, and a
person-started run whose origin names no actor. A work order records no author, so work orders name no one.
The two human id forms are kept as recorded: the door stamp's `person:<user>` carries no tenant and the
origin's `<tenant>:<user>` does, so the same person can appear once in each form (they are never merged by
guessing).

Read cost: one run-index listing (the home store, or the session's runs), one run load per turn root, one
read of the task inbox. O(index rows + turn roots + task events).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ..core.run_origin import origin_of
from .digital_identity import IdentityRef, identity_ref, read_identity
from .track_record import (
    TASK_INBOX_KIND,
    TrackRecordError,
    _index_rows,
    _load,
    _note_ref,
    is_counted_candidate,
    read_manifest,
    read_task_events,
    task_kind_of,
)

ENTITY_NAMESPACE = "entity"
PERSON_NAMESPACE = "person"
NOT_COUNTED_OTHER_NAMESPACE = "participant of another kind (not an entity or a human)"
NOT_COUNTED_NO_NAMESPACE = "participant without a namespace"
NOT_COUNTED_NO_ACTOR = "person-started run without an actor"


@dataclass(frozen=True)
class Acquaintance:
    """One entity or human met: `id` as recorded (`entity:pollux`, `person:laurent`, `default:laurent`)."""

    id: str
    first_seen: str
    last_seen: str
    interactions: int
    kinds: Mapping[str, int]  # task kind -> interactions, sorted by kind

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "interactions": self.interactions,
            "kinds": dict(self.kinds),
        }


@dataclass(frozen=True)
class Acquaintances:
    subject: str  # entity:<slug> | session:<session_id>
    identity_ref: Optional[IdentityRef]
    computed_at: str
    as_of: Optional[str]
    entities: Tuple[Acquaintance, ...]  # first met first
    humans: Tuple[Acquaintance, ...]
    not_counted: Mapping[str, int]  # reason -> occurrences (interactions with no usable identifier)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "identity_ref": self.identity_ref.to_dict() if self.identity_ref is not None else None,
            "computed_at": self.computed_at,
            "as_of": self.as_of,
            "entities": [a.to_dict() for a in self.entities],
            "humans": [a.to_dict() for a in self.humans],
            "not_counted": dict(self.not_counted),
        }


class _Fold:
    def __init__(self, self_id: Optional[str]) -> None:
        self.self_id = self_id
        self.met: Dict[str, Dict[str, Dict[str, Any]]] = {ENTITY_NAMESPACE: {}, PERSON_NAMESPACE: {}}
        self.not_counted: Dict[str, int] = {}
        self.stamps: List[str] = []

    def skip(self, reason: str) -> None:
        self.not_counted[reason] = self.not_counted.get(reason, 0) + 1

    def meet(self, who_list: List[Tuple[str, str]], *, kind: str, first: str, last: str) -> None:
        """One interaction with each distinct `(list, id)` in `who_list`."""
        for namespace, who in sorted(set(who_list)):
            entry = self.met[namespace].setdefault(who, {"first": first, "last": last, "n": 0, "kinds": {}})
            entry["n"] += 1
            entry["kinds"][kind] = entry["kinds"].get(kind, 0) + 1
            entry["first"] = min(entry["first"], first)
            entry["last"] = max(entry["last"], last)

    def classify_participant(self, participant: Any) -> Optional[Tuple[str, str]]:
        text = str(participant or "").strip()
        if not text or text == self.self_id:
            return None
        namespace, sep, name = text.partition(":")
        if not sep or not namespace or not name:
            self.skip(NOT_COUNTED_NO_NAMESPACE)
            return None
        if namespace in (ENTITY_NAMESPACE, PERSON_NAMESPACE):
            return namespace, text
        self.skip(NOT_COUNTED_OTHER_NAMESPACE)
        return None

    def result(self, *, subject: str, ref: Optional[IdentityRef]) -> Acquaintances:
        def listed(namespace: str) -> Tuple[Acquaintance, ...]:
            items = [
                Acquaintance(
                    id=who,
                    first_seen=e["first"],
                    last_seen=e["last"],
                    interactions=e["n"],
                    kinds={k: e["kinds"][k] for k in sorted(e["kinds"])},
                )
                for who, e in self.met[namespace].items()
            ]
            return tuple(sorted(items, key=lambda a: (a.first_seen, a.id)))

        stamps = [s for s in self.stamps if s]
        return Acquaintances(
            subject=subject,
            identity_ref=ref,
            computed_at=datetime.now(timezone.utc).isoformat(),
            as_of=max(stamps) if stamps else None,
            entities=listed(ENTITY_NAMESPACE),
            humans=listed(PERSON_NAMESPACE),
            not_counted={k: self.not_counted[k] for k in sorted(self.not_counted)},
        )


def _fold_runs(fold: _Fold, run_store: Any, rows: List[Dict[str, Any]], refs: List[Tuple[str, str, IdentityRef]]) -> None:
    for row in sorted(rows, key=lambda r: (str(r.get("created_at") or ""), str(r.get("run_id") or ""))):
        fold.stamps.append(str(row.get("updated_at") or ""))
        if not is_counted_candidate(row):
            continue
        run = _load(run_store, str(row["run_id"]))
        _note_ref(refs, row, run)
        vars_obj = run.vars if isinstance(run.vars, Mapping) else {}
        who: List[Tuple[str, str]] = []
        runtime_ns = vars_obj.get("_runtime")
        stamp = runtime_ns.get("entity") if isinstance(runtime_ns, Mapping) else None
        if isinstance(stamp, Mapping):
            participants = stamp.get("participants")
            if participants is not None and not isinstance(participants, list):
                raise TrackRecordError(f"run {run.run_id}: entity stamp participants is not a list")
            for p in participants or []:
                classified = fold.classify_participant(p)
                if classified is not None:
                    who.append(classified)
        origin = origin_of(vars_obj)
        if origin is not None and origin["kind"] == "person":
            actor = str(origin.get("actor") or "").strip()
            if actor:
                who.append((PERSON_NAMESPACE, actor))
            else:
                fold.skip(NOT_COUNTED_NO_ACTOR)
        if who:
            fold.meet(
                who,
                kind=task_kind_of(run, role=row.get("role")),
                first=str(row.get("created_at") or ""),
                last=str(row.get("updated_at") or ""),
            )


def _fold_tasks(fold: _Fold, home_dir: Path) -> None:
    for ev in read_task_events(home_dir):
        if ev.get("event") != "added":
            continue
        at = str(ev.get("at") or "")
        fold.stamps.append(at)
        classified = fold.classify_participant(ev.get("by"))
        if classified is not None:
            fold.meet([classified], kind=TASK_INBOX_KIND, first=at, last=at)


def entity_acquaintances(home_dir: Any, run_store: Any) -> Acquaintances:
    """The entities and humans an entity interacted with: its home's run store (`identity.entity_runtime`) and
    its task inbox. The entity itself (its manifest `entity_id`) is never its own acquaintance. Subject
    `entity:<slug>`; `identity_ref` from the home's `identity.json` (None while it has none)."""
    home = Path(home_dir)
    manifest = read_manifest(home)
    slug = str(manifest.get("slug") or "").strip()
    self_id = str(manifest.get("entity_id") or "").strip()
    if not slug or not self_id:
        raise TrackRecordError(f"{home / 'manifest.json'} lacks slug or entity_id")
    subject = f"entity:{slug}"
    recorded = read_identity(home)
    fold = _Fold(self_id)
    _fold_runs(fold, run_store, _index_rows(run_store), [])
    _fold_tasks(fold, home)
    return fold.result(subject=subject, ref=identity_ref(subject, recorded.rev) if recorded is not None else None)


def session_acquaintances(session_id: str, run_store: Any) -> Acquaintances:
    """The same over one session's runs (a regular agent's session on the host store): the people who
    started its turns, and the participants stamped on its runs. Subject `session:<session_id>`;
    `identity_ref` = the ref its newest turn root carries (`_runtime.identity_ref`), None when none does."""
    sid = str(session_id or "").strip()
    if not sid:
        raise TrackRecordError("session_id is required")
    subject = f"session:{sid}"
    fold = _Fold(None)
    refs: List[Tuple[str, str, IdentityRef]] = []
    _fold_runs(fold, run_store, _index_rows(run_store, session_id=sid), refs)
    ref = max(refs, key=lambda r: (r[0], r[1]))[2] if refs else None
    if ref is not None and ref.of != subject:
        raise TrackRecordError(f"session {sid}: its newest run carries the identity ref {ref} of another subject")
    return fold.result(subject=subject, ref=ref)


__all__ = [
    "Acquaintance",
    "Acquaintances",
    "NOT_COUNTED_NO_ACTOR",
    "NOT_COUNTED_NO_NAMESPACE",
    "NOT_COUNTED_OTHER_NAMESPACE",
    "entity_acquaintances",
    "session_acquaintances",
]
