"""One-shot backfill: written_amid edges for historical diary projections.

Maintainer ruling (2026-07-07): "the diary is connected to none other memory
and that is not ok." New diary writes carry written_amid edges; this script
repairs the HISTORICAL projections (written before the ruling) by re-running
each projection with the anchors read from the book.

Mechanism (policy-subagent verified, memory-lane cleared after the
COMPONENT/CONTEXT predicate policy landed): remember_many with the same
`diary:<entry_id>` idempotency key re-derives the SAME record id — the digest
assertion is skipped as existing (append-only), while the edge assertions
(new identities) are written. Anchors in the book are digest ROW ids; edges
need GRAPH ids, so each anchor is resolved row->subject before writing.
Running the script twice adds nothing (edge assertions dedup on their
derived identities).

Usage:
    python scripts/backfill_diary_edges.py <home_dir> [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("home", help="entity home directory")
    parser.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = parser.parse_args()

    home = Path(args.home).expanduser().resolve()
    manifest = json.loads((home / "manifest.json").read_text(encoding="utf-8"))
    entity_id = str(manifest["entity_id"])

    from abstractmemory import MemorySystem, SQLiteJournal, SQLiteTripleStore, TripleQuery
    from abstractmemory.records import MemoryRecordInput

    from abstractruntime.identity import DiaryStore
    from abstractruntime.storage.sqlite import SqliteDatabase, SqliteLedgerStore

    store = SQLiteTripleStore(home / "memory.sqlite3")
    journal = SQLiteJournal(home / "memory.sqlite3")
    ms = MemorySystem(store=store, journal=journal)
    diary = DiaryStore(
        entity_id=entity_id,
        ledger_store=SqliteLedgerStore(SqliteDatabase(str(home / "home.sqlite3"))),
    )

    def row_to_graph(row_id: str) -> str | None:
        """Anchors in the book are digest ROW ids; the edge currency is the
        assertion's SUBJECT (graph id). Unresolvable anchors are skipped
        loudly (they may predate vectors or belong to pruned smoke data)."""
        rows = store.query(TripleQuery(assertion_ids=(str(row_id),), limit=1))
        if not rows:
            return None
        return str(rows[0].subject)

    # Existing written_amid edges (skip already-repaired projections).
    existing: set[tuple[str, str]] = set()
    for scope in ("diary",):
        for a in ms.query(TripleQuery(scope=scope, owner_id=entity_id, limit=0)):
            if isinstance(a.attributes, dict) and a.attributes.get("record_edge") \
                    and str(a.predicate) == "written_amid":
                existing.add((str(a.subject), str(a.object)))

    entries = diary.list_entries()
    repaired = skipped = anchorless = 0
    for e in entries:
        entry_id = str(e.get("entry_id"))
        anchors_raw = list(e.get("anchor_record_ids") or [])
        graph_anchors = list(e.get("anchor_graph_ids") or [])
        if not graph_anchors:
            resolved = [row_to_graph(a) for a in anchors_raw]
            graph_anchors = [g for g in resolved if g and g.startswith("ex:")]
            dropped = len(anchors_raw) - len(graph_anchors)
            if dropped:
                print(f"  #FALLBACK {entry_id}: {dropped}/{len(anchors_raw)} anchors unresolvable, skipped")
        if not graph_anchors:
            anchorless += 1
            continue

        # The projection's own graph id (deterministic from the entry id).
        proj_rows = ms.query(TripleQuery(scope="diary", owner_id=entity_id, limit=0))
        proj_id = next(
            (str(a.subject) for a in proj_rows
             if isinstance(a.attributes, dict) and a.attributes.get("entry_id") == entry_id),
            None,
        )
        if proj_id is None:
            print(f"  #FALLBACK {entry_id}: no projection record found, skipped")
            continue

        new_edges = [g for g in graph_anchors if (proj_id, g) not in existing and g != proj_id]
        if not new_edges:
            skipped += 1
            continue

        print(f"  {entry_id} ({proj_id}): +{len(new_edges)} written_amid edge(s)")
        if not args.dry_run:
            # Re-projection with the SAME idempotency key: digest skipped as
            # existing; edge assertions written (append-only edge repair).
            visibility = str(e.get("visibility") or "self")
            kind = str(e.get("kind") or "note")
            private = visibility == "private"
            written_at = str(e.get("written_at") or "")
            date = written_at[:10]
            title = f"Diary entry ({'private' if private else kind}) — {date}"
            digest = ("Wrote a private diary entry." if private
                      else (str(e.get("gist") or "").strip()
                            or f"Wrote a diary entry ({kind}) at {written_at}; no gist elected."))
            ms.remember_many(
                [MemoryRecordInput(
                    kind="diary", title=title, digest=digest, keywords=(),
                    payload_ref=None,
                    attributes={"entry_id": entry_id, "written_at": written_at,
                                **({"private": True} if private else
                                   {"diary_type": kind if kind in ("note", "idea", "commitment",
                                    "reflection", "question", "problem") else "note",
                                    "diary_kind": kind})},
                    # provenance.source must stay in the declared diary-writer
                    # vocabulary (D4 form-gate: one writer per plane); the
                    # backfill IS the projection channel, re-run — the ruling
                    # rides in a separate provenance key.
                    provenance={"source": "diary-projection",
                                "entry_id": entry_id,
                                "backfill": "2026-07-07 diary connectivity ruling"},
                    edges=tuple(("written_amid", g) for g in new_edges),
                )],
                scope="diary", owner_id=entity_id,
                idempotency_key=f"diary:{entry_id}",
                turn_id="backfill-written-amid",
            )
        repaired += 1

    print(f"\n{'DRY RUN: would repair' if args.dry_run else 'repaired'} {repaired} projection(s); "
          f"{skipped} already-edged; {anchorless} anchorless (pre-anchor entries)")
    store.close()
    journal.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
