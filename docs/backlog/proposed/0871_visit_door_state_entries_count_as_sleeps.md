# 0871 — A visit's sleep and wake entries count as sleeps in the life statistics

> Package: AbstractRuntime (`identity/life.py` `life_sleep_stats`), AbstractGateway (`entity_phase.py` `admit_visit_posture` / `continue_sleep`, `entity_visits.py` visit close and `_restore_prior_state`, the entity tile and card that show `sleeps`)
> Type: bug
> Status: proposed
> Created: 2026-11-11
> Priority: P3
> Labels: entities, visits, state-history, life-stats, honest-status

## Summary

Found by the visit-without-words fix (runtime 0.19.4, commit d5c55d9; framework backlog 1185, operator ruling
2026-11-11 00:40: a visit opened and closed without a word "should not even be remembered").

Every visit writes two state transitions to `<home>/state_history.jsonl`: at open, the visit door's posture
(`asleep`, `mode: visiting`, `written_by: "visit-door"`, `entity_phase.admit_visit_posture`), and at close the wake
(or the interrupted sleep written back). `life_sleep_stats` counts every `asleep` row in `sleeps` (the visit-door
rows go to its `visit_yields` bucket, but stay in the total) and every row in `transitions` and `sleep_share`. The
entity tile (`routes/entities.py`, `sleeps`) and the console's identity card ("Sleeps") show that total. So each
visit, a wordless one included, adds one "sleep" to the entity's life, though nothing slept.

## Scope

1. Runtime half: `life_sleep_stats` leaves out every row whose `written_by` is `"visit-door"` from `sleeps`,
   `wakes`, `transitions` and `sleep_share`. Structural only: read `written_by` (and decide whether the legacy
   `mode: "visiting"` fold of rows written before the stamp stays); never parse the reason text. Whether
   `visit_yields` stays as its own count is part of the decision.
2. Gateway half: every write the visit door makes is stamped `written_by="visit-door"`. Today the open is; the
   close's wake (`entity_visits.py`, "visitor session ended (...)") and the aborted-open restore
   (`_restore_prior_state`) write with the default `operator` stamp, so they read as operator wakes.
3. The interrupted sleep written back at close (`continue_sleep`) keeps its original `written_by` by contract
   (contract-phases v2 §4.4), so it reads as a second sleep by its original author. It needs a structural mark
   (a field on the row, never the reason) so the count sees one sleep, not two.
4. The runtime's CLI visit (`chat.py`, `--pause-loop`) already stamps both its yield and its restore `visit-door`;
   keep it covered by the same test.

## Validation

A home with one operator sleep and one wake, then a visit opened and closed without a word (and one with a line):
`life_sleep_stats` reports 1 sleep and 1 wake before and after; a visit during that sleep leaves it 1 sleep; the
tile and the card show the same number.
