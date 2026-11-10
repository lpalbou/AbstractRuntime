# 0869 — A loop waking from its own sleep after a revoke still writes "own time resumes"

> Package: AbstractRuntime (identity/life.py `LifeLoop._wake_from_self_sleep`)
> Type: bug
> Status: proposed
> Created: 2026-11-11
> Priority: P3
> Labels: entities, own-time, state-history, honest-status

## Summary

Found while fixing "own time withdrawn ends on the turn in flight" (runtime 0.19.3, commit 7239fa0). When an entity
wakes from a sleep it elected itself, `_wake_from_self_sleep` writes the state back to `awake` with the reason
"woke from self-elected sleep - own time resumes" (and "maintenance cycle complete - personal time resumes" after
the maintenance window). It writes that reason without reading the personal grant: if the operator turned own time
off during the sleep, the state history still says own time resumes, while the loop then stops at its next
boundary. The reason row is false; the app or a reader of `state_history.jsonl` sees own time resuming after the
operator withdrew it.

## Scope

1. `_wake_from_self_sleep` reads the withdrawal (`_own_time_withdrawn`, or the grant and the stop) before writing,
   and writes a reason that says what happens: awake, own time withdrawn during the sleep (no "resumes").
2. Same for the maintenance-cycle wake ("personal time resumes").
3. No other behaviour change: the loop still exits at the boundary as 0.19.3 does.

## Validation

A self-elected sleep, the grant revoked during it: the wake row's reason does not say "resumes" and names the
withdrawal; with the grant still on, the reason is unchanged.
