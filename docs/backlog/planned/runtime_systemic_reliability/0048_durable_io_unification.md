# Planned: utils/durable_io — one fsync-hardened writer + one recovering JSONL reader

## Metadata
- Created: 2026-07-12
- Status: Planned
- Completed: N/A
- Priority: P0 (the RunState checkpoint is the least-protected file in the system)

## ADR status
- Governing ADRs: None
- ADR impact: fsync policy (what gets fsynced when) lands as one paragraph
  in `docs/architecture.md` §Reliability.

## Context
The repo has ONE hardened file writer — `utils/atomic_files.atomic_write_text`
(tmp + fsync + os.replace, atomic_files.py:25-42) — used only by
`prompt_overlay` and `tool_policy` (two operator config files).

## Current code reality (line-verified 2026-07-12)
- FIVE hand-rolled tmp+replace copies with NO fsync:
  - `json_files.py:145-156` — the RunState checkpoint itself;
  - `artifacts.py:1484` and `:1518`;
  - `commands.py:209` and `:388`.
- `JsonlLedgerStore.append` has neither lock nor fsync (json_files.py:448-453)
  → run-save vs ledger-append ordering after power loss is undefined;
  idempotency keys have been silently absorbing this.
- Corrupt-line recovery differs per store: the ledger recovers concatenated
  objects with #FALLBACK (json_files.py:474-499); the run store silently
  returns None on a corrupt checkpoint (json_files.py:204-205); the command
  store silently skips bad lines with no log (commands.py:253-256).

## Problem
Crash semantics are five folklores instead of one sentence. The most
important file on disk (the run checkpoint) has the weakest write path, and
two of the three recovery behaviors are silent.

## What we want to do
One `utils/durable_io.py`:
- `atomic_write_text/bytes(path, data, fsync=True)` (absorb atomic_files);
- `append_jsonl_line(path, obj, fsync=...)`;
- `iter_jsonl_recovering(path, on_recover=warn_counter)` — the ledger's
  concatenated-object recovery, shared.
Point all five hand-rolled writers and all three readers at it. Decide fsync
policy ONCE: fsync run checkpoints on STATUS TRANSITIONS at minimum;
measure the JSONL-append fsync cost before defaulting it on (SQLite paths
already ride WAL).

## Why
Uniform, stateable crash semantics; cheap now, expensive to retrofit after
1.0 durability promises.

## Requirements
- Behavior-preserving except added fsyncs and un-silenced recovery (silent
  skips become warn-once + counter, 0054).
- `utils/atomic_files.py` stays as a re-export shim (two consumers +
  possible external imports).
- Recovery events surface identically across stores (same counter names).

## Suggested implementation
Mechanical; the only judgment call is fsync-on-append default for JSONL —
benchmark and record the decision.

## Scope
The module, the five writer swaps, the three reader swaps, the policy
paragraph, pins.

## Non-goals
- No change to SQLite paths (WAL owns their durability).
- No new file formats.

## Dependencies and related tasks
- 0045 (windows interact with fsync points), 0054 (counters), 0055 (JSONL
  count/list uses the shared reader).

## Expected outcomes
One sentence describes every file write's crash contract; corrupt-input
recovery is visible in counters everywhere.

## Validation
- Pins: torn-write simulation (truncate mid-file) per store → recovery
  behavior identical + counted; fsync-policy benchmark recorded.
- Full suite green.

## Progress checklist
- [ ] durable_io module
- [ ] Five writers swapped
- [ ] Three readers swapped (silence removed)
- [ ] fsync policy decided + documented
- [ ] Fable5 adversary folded
