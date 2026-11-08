### Fixed

- **No length cap anywhere on stored text** (root ADR 0026/0054; operator 2026-11-08: stored text is never cut or
  capped, what is shown is each app's fold). The remaining bounds are gone:
  - the automation `title` (`automations/models.py` `validate_title`, `automations/wakes.py`): any non-empty text,
    whitespace trimmed; the 120-character refusal is what made the Assistant web cut titles before saving
    (root/assistant 1154); a note wake's derived "Reminder: <note>" title is whole. `TITLE_MAX` is gone;
    `docs/automations.md` says so;
  - a published event's `title`, `body` and `subject` (`core/publish.py`): non-empty / text, no 120 / 2,000 bound;
  - the mechanical record title (`identity/digest.py`): the chosen sentence whole (it kept 14 words and a "…"), and
    the words whole when no sentence qualified (was 8 words);
  - the `[felt: …]` and `[realized: …]` markers in the stored reply (`identity/reflection.py`) carry the record's
    words, his reason and the realization whole (they were marked cuts at 60 / 80); `_receipt_gist` is gone, and the
    two visit-workflow refusal notices that used it quote the realization whole;
  - a failed occurrence's attention body is the whole error (`automations/controller.py`, was 2,000);
  - `#FALLBACK` notices (`identity/reflection.py`) quote the skipped line whole — notices rest in run vars and in the
    close report, so they are stored (were 80 / 60);
  - `identity/life.py`: the loop status `day_cause.detail` (was 200), the entity's `next:` cue (was 400) and the rest
    reason written to the sleep state (was 400) are whole; the standing-interest offer in the day cue folds visibly
    with `drive_gist` instead of a silent 110-character cut;
  - `identity/chat.py`: the turn report's tool `arg` (was 120), the resolution gist handed to the reflection prompt
    (was 100) and the sheet line's first line (was 120) are whole.
  Left as log-only (console `out()` echoes): `chat.py` lesson/interest/realization echoes at 70, `life.py` tick
  `reply_head` at 160 and work verdict at 120. Left as prompt currency under a token budget: the recall handles
  (`chat.py` ~1960, title 120 / digest 280) and the recall path labels. Still bounded and MARKED, not changed here:
  the reflection sheet line (`chat.py` `_SHEET_LINE_MAX = 160`, `#TRUNCATION`) and the world-model card text
  (`chat.py` `_CARD_TEXT_MAX = 1600`, `#TRUNCATION`), both stored; a ruling on them is pending.
