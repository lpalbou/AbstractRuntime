### Fixed

- **What the identity runtime stores is whole; what a screen shows is the app's fold** (root ADR 0054; operator
  2026-11-08: "the text stored should never be truncated, otherwise it's not a verbatim; the text shown is
  UI/app dependent"). Six write paths cut text to a fixed length on the way to the record; none does now:
  - `identity/digest.py`: when no whole sentence of a side fits the digest budget (one long run-on sentence,
    prose without terminal punctuation), the stored digest carries that side whole, not its first 200 characters.
  - `identity/reflection.py`: the `[kept interest: "…"]`, `[learned: "…"]` and `[topic: "…"]` markers written
    into the stored reply carry the elected words whole (they were cut at 80 characters, silently).
  - `identity/act_only.py`: a captured diary entry's `gist` (run vars, the visit sheet) is his elected gist whole,
    else the entry's first non-empty line whole (120 characters was a cut).
  - `automations/wakes.py`, `automations/models.py`, `automations/attention.py`: a note wake's `notify.title` and
    `notify.body`, the definition's `notify` and every attention item's title and body are stored whole (they were
    cut at 120 / 2,000, and the default body at 280). The automation `title` keeps its validated 120-character
    bound; `NOTIFY_TITLE_MAX`, `NOTIFY_BODY_MAX` and `NOTIFY_DEFAULT_BODY_MAX` are gone. `docs/automations.md` says so.
  - `identity/life.py`: the day-open cue's quote of an open question or problem (a prompt, so it may fold) folds
    visibly with the desk's `drive_gist` rule (first line, 60 words, " …", reread command beside it) instead of a
    silent 120-character cut mid-word.
  Tests: `tests/test_stored_text_whole_adr0054.py` (one per site, each red against the old cut) and
  `tests/test_schedule_run.py` (a 2,000+ character note reaches the attention item whole).
