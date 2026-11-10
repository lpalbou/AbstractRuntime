# 0868 — Turning own time off also stops a loop that is doing work

> Package: AbstractRuntime (identity/life.py own-time loop: `_should_stop`, `_own_time_withdrawn`), AbstractGateway (entity_phase.py personal toggle)
> Type: bug
> Status: proposed
> Created: 2026-11-11
> Priority: P2
> Labels: entities, own-time, work, operator-control

## Summary

Found while fixing "own time withdrawn ends on the turn in flight" (runtime 0.19.3, commit 7239fa0). The operator's
personal-time toggle is the only switch (there is no work switch, CONTRACT v3 D1). When it turns own time off, the
gateway revokes the grant and, whenever the loop is running, enqueues the durable stop command
(`entity_phase.py`, the toggle: `stop_loop(home_dir, reason="personal turned off by ...")`). The loop honours that
stop command on any day, a work day included (`LifeLoop._should_stop` reads the command whatever the phase;
`_own_time_withdrawn` says "a work day answers to his stop only", and the toggle sends exactly that stop). So
turning personal time off while the entity is working on a job also ends the work: the turn in flight finishes,
then the loop exits. Jobs still queued are not lost but nothing runs them until the next trigger restarts a loop.

The operator's intent when turning personal time off is "no more own time", not "stop the work".

## Scope

1. Decide the contract with the operator: does turning personal time off stop a loop on a work day? (Today: yes.)
2. If not: the toggle sends its stop only when the open day is personal (the gateway reads `day_kind` from
   `loop_status`), or the stop command carries its cause (`personal_off`) and a work-day loop ignores that cause;
   either way the work day ends normally and the loop does not open a personal day after it (the grant is off).
3. If yes: say it on the toggle (the Entity app) and make queued jobs restart a loop on their own, not wait for an
   unrelated trigger.

## Validation

A loop on a work day with a second job queued: turning personal time off lets the current job's turn and the next
queued job run (or, under the "yes" contract, the app says the work stops and the queued job starts on its own);
no personal day opens after the work.
