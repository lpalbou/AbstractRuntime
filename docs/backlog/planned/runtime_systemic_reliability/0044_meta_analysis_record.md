# 0044 — Runtime meta-analysis record (2026-07-12): from artisanal to systemic reliability

> TRACK SOURCE DOCUMENT — the analysis and ranked R-plan below were split
> into executable backlog items in this folder (planned 0045-0055) and
> `../../proposed/runtime_systemic_reliability/` (0056-0063). Mapping:
> R1→0046, R2→0045, R3→0047, R4→0048, R5→0049, R6→0050, R7→0051, R8→0053,
> R9→0052, R10→0054, R11→0055, R12→0056, R13→0057, R14→0058,
> R15→0059/0060/0061/0062/0063. This file stays as the evidence record;
> execute from the items.

Maintainer ask (Laurent, 2026-07-12 18:18): "can you do a deep meta analysis
with 2 adversarial sub agents and see if there is anything we could do to
improve the runtime as a whole? you have a huge responsibility, as everything
goes through you."

Method: two fable5 adversaries over the whole package — one attacking
structural integrity (module cohesion, layering, concurrency ownership,
storage surface), one attacking operational reliability (crash-replay
completeness, 24/7 scale, sync-store contract, observability) — synthesized
with the runtime seat's own session observations. All findings line-verified
against the tree at commit 5f81ab7.

## Combined verdict

The package is structurally sound where it matters: durable-core semantics
(idempotency keys, recheck-before-save, watermark-deduped sidecars,
hash-chained appends, read-side recovery) are consistently designed and
unusually well pinned (~1,145 tests). BUT its reliability today is the
product of ARTISANAL per-feature rigor (the steer drain, the chain-fork fix,
criterion-7), not SYSTEMIC guarantees — and artisanal rigor does not survive
the next ten features or the next contributor. No redesign is needed; the
debt is concentrated and specific.

## The two correctness-grade findings (both adversaries converged)

1. **No per-run driver exclusion.** Zero locks in core/runtime.py; tick()
   and resume() can drive the SAME run concurrently (scheduler poll thread,
   emit_event fan-out, gateway runner, MultiStoreScheduler), and
   JsonFileRunStore.load() returns ALIASED objects — so it is a data race,
   not just a save clobber. B8 as filed is UNDER-SCOPED: five non-tick-thread
   RunState writers exist (pause/cancel; cross-run memory-owner saves in
   _handle_memory_tag/note/compact/rehydrate; emit_event inline resume;
   history_bundle.persist_workflow_snapshot; active_context.rehydrate).
2. **O(entire ledger) idempotency lookup per effect step.**
   _find_prior_completed_result lists the FULL ledger on every effectful
   step; resume() does two more full lists. A 24/7 resident at 100k records
   pays a full-file parse per step — the deterministic scale cliff for the
   fleet, charged to the tick thread.

## Also found (new, serious)

3. **Crash-ordering is four different orderings, two silent-on-failure**:
   completion saves state THEN appends the terminal record (crash between →
   terminal run with a never-terminating ledger — the poller-busy-loop shape,
   produced runtime-side); resume() appends records BEFORE saving, inside
   `except Exception: pass` (ledger claims a step the durable state denies).
4. **The RunState checkpoint is the least-protected file in the system**:
   five hand-rolled tmp+replace writers with NO fsync (run checkpoint,
   artifacts, commands); the one hardened fsync writer (utils/atomic_files)
   is used only by two config files. Corrupt-line recovery differs per store
   (ledger recovers+warns; run store silently Nones; command store silently
   skips).
5. **No schema versioning on run files/records**: a v-next enum in one run
   file RAISES OUT of every list_runs/scheduler scan on the JSON store
   (blast radius = the whole directory); SQLite silently returns None (a
   zombie run). Two stores, two failure modes, both wrong.

## Ranked plan

### This week (P0)
- **R1. Per-run tick guard + concurrent-driver detection** — per-run_id
  in-process lock taken by tick()/resume(); second entrant refuses loudly;
  save() warns when persisted updated_at is newer than the loaded snapshot
  (cross-process detector). Closes the two-drivers-one-run class at the
  kernel.
- **R2. ONE written crash-ordering invariant + fix the two inversions** —
  rule: state save is the commit point; transition records append BEFORE the
  save that makes them true; terminal-path append failures are never silent.
  One page in docs/architecture.md (which currently has zero occurrences of
  "thread"/"concurrency"/"single-writer").
- **R3. Indexed idempotency lookup** — SQLite: idempotency_key column +
  index (ledger_heads-style backfill precedent); JSONL: bounded reverse-tail
  scan + per-run key cache invalidated on append.
- **R4. utils/durable_io.py** — one atomic_write (fsync) + one
  append_jsonl + one recovering-iterator; point all five hand-rolled copies
  at it; fsync run checkpoints on status transitions at minimum.
- **R5. Mechanical llm_client split** — 11,825 lines → package
  integrations/abstractcore/llm/ (grounding / residency / blocs /
  prompt_cache / media_artifacts / normalize / protocol / local / multilocal
  / remote), llm_client.py stays as a pure re-export facade (private-name
  imports keep working; core/runtime.py:562's four grounding imports move
  DOWN to core, killing the core→integrations inversion).

### Before the fleet runs 24/7 (P1)
- **R6. B8 executed as designed, scoped to ALL FIVE writers** — signal-only
  control sidecar (the proven steer pattern) for pause/cancel; memory-owner
  cross-run writes through a per-run-id lock or owner-mutation queue;
  emit_event signal-only delivery as a host-selectable mode (inline resume
  stays for single-process hosts).
- **R7. Runtime-owned durable event mailbox** — take the gateway's
  events_inbox copy home (durable append, per-run monotonic seq, 500-cap,
  event_id dedup, receiver counts — gateway explicitly asked "carry both");
  generalize the steer sidecar store into a named-mailbox primitive so
  steer/events share ONE crash-ordering. Also kills the
  list_runs(limit=10_000) scan per emit.
- **R8. Bounded run-vars growth** — byte-cap node-trace entries (digest +
  artifact ref above threshold), cap _runtime.inbox and warning lists,
  vars-size gauge with threshold warning.
- **R9. Sync-store event-loop tripwire** — get_running_loop() check in hot
  store methods → warn-once + counter ("sync store called on a running event
  loop"); optional AsyncRunStoreFacade. Two incidents already; prevents the
  third structurally.
- **R10. RuntimeHealth counters** — one monotonic-counter dict + last-error
  ring on Runtime (tick durations, drains/failures, replay collapses,
  recovered lines, guard trips), incremented at existing warning sites,
  one getter the gateway serves. Counters, not log spam.
- **R11. JSONL honesty** — count() through the same recovery decode as
  list() (the divergence class is structural today); document JSONL as
  dev-tier, SQLite as the 24/7 default.

### Before 1.0 (P2)
- **R12. Schema-version + quarantine loading** — schema: run.v1 on save;
  unknown status/reason → per-run quarantine (skip + loud counter), never
  scan-crash, never silent zombie; split-runner rule documented.
- **R13. Declarative trust-boundary stamp registry** — {tool → (arg,
  source, on_missing)} table consumed by the planner loop + a test that
  every schema-hidden arg has a stamp entry (shell namespace + agora alias
  are two hand-rolled copies today; make the third impossible to forget).
- **R14. Terminal-run archival helper** — archive_terminal_runs(before=...)
  moving run+ledger to an archive subdir (host-invoked, marker-recorded);
  prevents the ~10k-file directory cliff and operator rm-under-live-sidecar.
- **R15. Remaining moves**: B3 (chat-shape envelope → the trailing-message
  lane that tool-loop shape already uses); memory-span handlers out of the
  Runtime class (~1,700 lines → memory/span_effects.py); visual_to_flow
  mega-function decomposition (band-by-band, strongest-tested area);
  root-API __getattr__ deprecation machinery + public/internal split;
  dependency honesty (docs re-scope or [documents]/[media] extras);
  _jsonable ×3 and _set_nested ×2 dedup.

### The ONE highest-leverage investment
A generalized KILL-AND-REPLAY HARNESS promoted from criterion-7 into shared
test infrastructure: SIGKILL at injected persistence point N → restart from
stores → assert terminal-output/ledger convergence + no doubled side effects
+ run under two concurrent drivers. It converts every crash-window and
concurrency claim from folklore into a checkable property, is the first
consumer of R1-R3, and keeps R6-R15 honest as they land.

### Explicitly DO NOT do (unanimous)
- No async kernel rewrite (the sync one-writer tick is why crash windows are
  analyzable; facades + tripwires only).
- No tick/resume/wait state-machine decomposition (one state machine; after
  R15's memory extraction, ~3,200 lines of coherent kernel).
- No new eventing abstraction (the hooks thin-layer ruling paid off all
  week; named mailboxes over the existing primitive, R7).
- No deep-copy/frozen stores (hot-path tax; R1's guard addresses the risk).
- No auto-pruning/compacting ledgers in place (append-only is the audit and
  identity posture; archival is explicit and marker-recorded).
- No merging the three LLM client classes (genuinely different transports).

## Session observations (the seat's own, for the record)
- Same-hour reactive shipping works BECAUSE the adversary pass is mandatory:
  the seat's own P0 this session (steer ack-before-save) was caught there.
- Accretion lands where asks point, not where architecture wants: llm_client
  absorbed four unrelated features this session alone.
- Drift pins are alarms, not fixes: visit-id literal, alias tables, and
  gateway pattern-copies each carry one; R5/R7/R12 retire the causes.
