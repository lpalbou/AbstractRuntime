### Fixed

- **The sleep's repair now reaches episodes and session summaries formed before formation authored them**
  (`identity/authoring.py`; operator 2026-11-09: every memory has an authored title and summary, the sleep repairs
  leftovers). `declare_missing_metadata` selects them BY STAMP, never by their text: a record of kind `episode` or
  `summary` that carries no authored provenance (no `metadata_authored`, no authored `title_provenance`, its current
  reading not written by a capability model, the operator or the entity) and whose reading memory derives
  (`reading_provenance`) as `mechanical` (a `mechanical-v1` / `-v2` / `-floor-v1` / `-flow-v1` / `-dedup-v1` /
  `-close-v1` digest_method or stamp) or `unrecorded` (no label at all: the Entity app's "who wrote it was not
  recorded") is declared pending for BOTH its title and its summary through memory's `declare_pending`. The stored
  title and text are kept as version 0; nothing is cut. A summary titled with the runtime's own pre-4eefddb template
  `session reflection: <session id>` (matched by that prefix only) has its title declared even when its summary was
  already authored. A ripple's consolidation record is never taken.
- **A closed record is no longer scanned by the repair**: memory refuses `closed` (its replacement is the live
  record), so a superseded episode used to cost one `#REPAIR_REFUSED` line every night. The receipt counts them in
  `closed`.

The drainer authors the declared fields with the summaries capability model and stamps the capability model's
provenance on each (`retitle` -> `title_provenance`, `rewrite_digest` -> `understanding`, `digest_method
authored-v1`: author `{kind: capability-model, id: summaries}`, the client's reader, the call kept as a home
artifact). BUDGET: one task per field, at most 64 per sleep pass (`build_consolidator(authoring_cap=64)`, about 32
records with both fields), plus `rebuild.cap_per_turn` (default 3) at each own-time tick boundary and at a session's
close. A home with thousands of such records is repaired over many nights, its declared titles reading None until
authored. Castor's live home, counted read-only on a copy on 2026-11-10: 119 records newly selected (105 session
reflections, 9 day summaries, 5 visit episodes: 238 fields), 126 closed mechanical episodes skipped.
