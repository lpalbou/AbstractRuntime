### Fixed

- **Own time withdrawn ends on the turn in flight** (`identity/life.py`, `identity/chat.py`; operator 2026-11-10
  23:09: "if I remove the personal time, it's non negotiable; but it's ok to let the entity finish its turn").
  When the operator turned own time off (the gateway writes the grant `disabled`, then the stop command), the loop
  honored the stop at the next tick boundary but kept working past the revoke: the tick's look-back already under
  way still formed its summary, interest and diary entry, and the day's close then ran a second model look-back
  (on the live home: 22:03 revoke, look-back records at 22:07, a day summary and diary at 22:11-22:12, exit 22:12).
  Now the turn in flight finishes whole and nothing after it runs:
  - a revoke seen when the turn ends skips the tick's look-back and the authoring drain;
  - a look-back whose model call was in flight keeps nothing of its reply (`ChatSession.withdrawn_check`, set by
    the loop; `_reflect_over(abandon_if_withdrawn=True)` for the tick and day look-backs, never a salvage);
  - the day closes without its look-back.
  The day's look-back is deferred, never lost: `pending_reflection.json` carries the whole day to the next open
  (a visit, the next own-time day). "Withdrawn" is the operator's stop (the stop command, the STOP file, the kill
  switch) or, on a personal day, the grant revoked or expired; a work day answers to the stop only. A consumed stop
  command is now remembered by the loop (`_should_stop` treats `stop_command` as final, like the kill switch).
  A normal close (rest, the day's last tick) still runs its look-back.
