# 0850 — Growing automations: keep the replayed history out of the controller ledger

- **Status**: PROPOSED 2026-10-19
- **Owner**: runtime (`automations/controller.py` `build_prepared`, the admission / dispatch /
  retry records); read paths in `automations/service.py` (`list_occurrences`) and the gateway's
  automation routes benefit without change
- **Origin**: operator, 2026-10-19: an automation's page on the Apple Watch takes seconds to load
  on a small public gateway ("why is it so slow? do you see problems in the code?")

## Problem

A growing automation hands each occurrence the conversation so far: `context.messages`, windowed
to `HISTORY_REPLAY_MAX_TOKENS` (50k tokens). `build_prepared` freezes that history inline in the
occurrence's prepared inputs, and the controller's ledger then carries it about five times per
occurrence:

- `automation.admitted`: `prepared.input_data.context` and again in `delta`;
- `start_subworkflow` (the dispatch effect's `vars`), `automation.dispatched`, the `resume`
  record that brings the child's result back, and `automation.retry_scheduled` on each retry.

Measured on the public gateway (2026-10-19), automation `0f83b6d5` ("investigate today's news.",
44 occurrences, growing):

| Record type | Count | Total | Largest |
|---|---|---|---|
| `start_subworkflow` | 140 | 7.46 MB | 179 KB |
| `automation.dispatched` | 70 | 7.40 MB | 180 KB |
| `resume` | 72 | 6.35 MB | 263 KB |
| `automation.admitted` | 44 | 5.47 MB | 263 KB |
| `automation.retry_scheduled` | 26 | 3.30 MB | 161 KB |
| everything else | 187 | 0.17 MB | 3 KB |

The controller ledger is 29.5 MB for 539 records. `context` alone is 127 KB of a 130 KB prepared
input. The size grows with every occurrence until the window is full, then by about 0.6 MB per
occurrence.

Every read that walks the controller's records pays for all of it: `list_occurrences` and
`automation_records` (`ledger_store.list` parses the whole ledger) run for the occurrences page,
the automation's detail, its summary row and the gateway's `_triggers_by_revision`. Measured:

| Request | Growing automation, 44 runs | Same shape, 1 run |
|---|---|---|
| `GET /automations/{id}/occurrences?limit=1` | 2.57 s | 0.08 s |
| `GET /automations/{id}` | 1.0–6.6 s | 0.06 s |
| `GET /automations` (5 automations) | 9–22 s | — |
| `GET /runs/{id}` (plain run read, baseline) | 0.09 s | |

The cost is in the size of the ledger, not in what is asked: `limit=1` and `limit=6` take the
same 2.3 s.

## Proposed fix

1. **Freeze the replayed history once, as an artifact, and keep a reference inline**, exactly as
   `_freeze_email_inputs` already does for email bodies: `build_prepared` stores
   `context.messages` in the artifact store and puts the `$artifact` ref in `prepared`. Admission,
   dispatch, retry and resume records then carry a ~200-byte ref instead of up to 263 KB. The
   history the occurrence was given stays exact and auditable (the artifact is immutable and
   content-addressed), so provenance is unchanged.
2. **Resolve the ref when the child run starts**, strictly: an unreadable history fails the
   admission (today's rule for a missing seed), never a run without context.
3. **Existing ledgers stay readable as they are**: inline histories in old records are still
   valid. A one-time compaction is not needed for correctness. If it is wanted for the speed of
   old automations, it is a separate, opt-in maintenance command.
4. **Optional, read side**: `list_occurrences` could read only the records it needs through a
   records-of-these-types path on the ledger store (the conversation search item 0849 wants the
   same). Once the records are small this matters much less; measure after step 1 first.

## Acceptance

- After 50 growing occurrences at a full 50k-token window, the controller ledger stays under
  1 MB (today: about 30 MB).
- `GET /automations/{id}/occurrences?limit=1` and `GET /automations/{id}` answer in under 0.3 s on
  the public gateway for such an automation, and `GET /automations` in under 2 s for five of them.
- The occurrence's child run receives exactly the history it receives today (a test compares the
  messages handed to the child before and after the change).
- An occurrence whose history artifact cannot be read fails its admission with a clear reason.

## Notes

- Clients cannot fix this: the Apple Watch now keeps an automation's definition and results once
  read and re-reads the list at most once a minute, which hides the cost after the first read
  but not on it.
- Related: 0849 (a records-of-one-type ledger read).
