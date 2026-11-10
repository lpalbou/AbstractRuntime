# 0870 — A visit without a word is taken as the previous summon for the identity cue

> Package: AbstractGateway (`entity_visits.py` `previous_summon_at`, `_visit.identity_since` at visit open), AbstractRuntime (`identity/visit_workflow.py` `visit_without_words`, `identity/prelude.py` identity cue)
> Type: bug
> Status: proposed
> Created: 2026-11-11
> Priority: P3
> Labels: entities, visits, identity, prelude

## Summary

Found by the visit-without-words fix (runtime 0.19.4, commit d5c55d9; framework backlog 1185, operator ruling
2026-11-11 00:40: a visit opened and closed without a word "should not even be remembered").

The prelude's identity cue (ADR 0052 §7) tells the entity, in one sentence, about the change that made its current
look and voice, when that change is newer than `identity_since`. The gateway sets `_visit.identity_since` at visit
open from `previous_summon_at(entities_dir, slug)`: the time of the newest `summon` host marker. A visit opened and
closed without a word still writes its `summon` marker, so it becomes "the previous summon" for the next visit.

Case: the operator changes the entity's look or voice, opens a visit, closes it without a word (no turn, so no
model call ever read that visit's prelude), then opens another visit. The second visit's `identity_since` is the
wordless visit's open time, which is after the change, so the cue is not told. The entity is never told about a
change it never saw.

## Scope

1. `previous_summon_at` skips a summon whose visit had no word. Structural only: the summon marker's `run_id` names
   the visit run, and the runtime's `visit_without_words(run)` (or `track_record.is_wordless_visit`) answers from
   `_visit.turn_n`; or the matching `session_closed` marker's turn count. Never from any reason or text field.
2. The operator-visit chat lane (`entity_chat.py`) writes `summon` without a `run_id`; its `session_closed` marker
   carries `turns`. Decide whether that lane follows the same rule, and name the field it reads.
3. A visit still open (no close marker yet) and without a word: not a previous summon either (it counts from its
   first line, as in the track record).
4. Cost: one run load (or one marker match) per candidate summon, newest first, stopping at the first visit with a
   word; no extra per-turn work.

## Validation

Change the look; open and close a visit without a word; open a second visit: its prelude carries the identity cue.
Same sequence with one line in the first visit: the second visit's prelude does not repeat the cue.
