# Runtime systemic reliability backlog track

## Status
Mixed: planned (this track, items 0045-0055) + proposed
(`../../proposed/runtime_systemic_reliability/`, items 0056-0061).

## Purpose
Convert the runtime's reliability from ARTISANAL (per-feature rigor: the
steer drain, the chain-fork fix, criterion-7 — each excellent, each
hand-built) into SYSTEMIC guarantees, before the 24/7 resident fleet and
before 1.0. Source: the maintainer-requested two-adversary meta-analysis of
2026-07-12 (`0044_meta_analysis_record.md`) — one adversary attacked
structural integrity, one attacked operational reliability; both converged
with the runtime seat's session observations.

The track's organizing idea: the same-hour shipping model works because the
adversary pass is mandatory; these items convert that per-feature vigilance
into structural guarantees so the vigilance gets cheaper, not lost.

## Items (planned, this folder)
- `0044_meta_analysis_record.md`: the full two-adversary analysis + ranked
  R1-R15 plan (the track's source document; not itself executable work).
- `0045_crash_replay_harness_and_ordering_invariant.md`: THE keystone —
  generalized kill-and-replay test harness + ONE written crash-ordering
  invariant + fix the two ordering inversions. First consumer of everything
  below; do first.
- `0046_per_run_driver_exclusion.md`: per-run tick/resume mutual exclusion +
  concurrent-driver detection (correctness-grade; two drivers can race one
  run over ALIASED state today).
- `0047_indexed_idempotency_lookup.md`: kill the O(entire-ledger) scan per
  effect step (the deterministic 24/7 scale cliff).
- `0048_durable_io_unification.md`: one fsync-hardened atomic-write +
  recovering-JSONL module; the RunState checkpoint is currently the least
  protected file in the system.
- `0049_llm_client_mechanical_split.md`: 11.8k-line module → subpackage with
  a re-export facade; also kills the core→integrations grounding-import
  inversion.
- `0050_control_sidecar_all_writers.md`: B8 executed as designed and
  RE-SCOPED to all five non-tick-thread RunState writers.
- `0051_runtime_owned_durable_event_mailbox.md`: unify the three message
  lanes — bring the gateway's events_inbox durable-append/event_id-dedup
  home over the steer sidecar's store shape.
- `0052_sync_store_event_loop_tripwire.md`: three incidents of sync stores
  on serving loops; make the contract self-enforcing.
- `0053_bounded_run_vars_growth.md`: caps + gauges for node traces, inboxes,
  warning lists before residents run 24/7.
- `0054_runtime_health_counters.md`: counters-not-folklore observability the
  gateway can serve.
- `0055_jsonl_store_honesty.md`: count()/list() convergence (the structural
  divergence class that busy-looped a poller) + tier documentation.
- `0064_prompt_cache_fingerprint_stability.md`: per-turn KV-cache rebuild
  fix (core's MLX c1127 layer 3) — tool_calls in the fingerprint +
  msg-0 grounding stability (B3-coupled with proposed 0063). Added
  2026-07-12 from core's bench evidence.

## Reading order
0044 first (the evidence), then 0045 (the keystone harness), then 0046-0048
(kernel correctness), then 0049 (structure), then 0050-0055 (fleet
readiness) in any order — 0050 and 0051 pair well since both reuse the
sidecar pattern.

## Governing ADRs
None — this repo records durable policy in `docs/architecture.md` and the
workspace `AGENTS.md`, not an ADR directory. Items that create durable rules
(0045's crash-ordering invariant, 0046's ownership rule) land those rules in
`docs/architecture.md` as part of their scope and say so explicitly.

## Scope
Runtime-tree work only, with one cross-repo coordination item (0051 takes
the gateway's events_inbox copy home; gateway pre-agreed: "carry receiver
counts + event_id dedup if it moves").

## Non-goals (unanimous across both adversaries + the seat — do NOT build)
- No async kernel rewrite: the sync one-writer tick is why crash windows are
  analyzable at all. Facades and tripwires (0052) only.
- No tick/resume/wait state-machine decomposition: it is ONE state machine;
  splitting scatters the invariants that contain the races.
- No new eventing abstraction: the hooks thin-layer ruling ("compile down to
  existing primitives") paid off and stays the law; 0051 is named mailboxes
  over an existing primitive, not a bus.
- No deep-copy/frozen store views: hot-path tax proportional to vars size;
  0046's guard addresses the actual risk.
- No auto-pruning/compacting ledgers in place: append-only is the audit and
  identity posture; archival stays explicit and marker-recorded (proposed
  0058).
- No merging the three LLM client classes: genuinely different transports.

## Notes for future agents
- Every item here must go through the house discipline: build → fable5
  adversary on the built code → fold findings → full suite → commit with the
  no-vendor-trailer rule.
- 0045 exists so the OTHER items' crash/concurrency claims become checkable
  properties; if you land 0046-0051 without 0045, you are re-creating the
  artisanal pattern this track exists to end.
- The B-series debt referenced throughout (B3/B8/B11) was filed from the
  code seat's prompt-cache adversary (commons c971, 2026-07-12); B8 is
  superseded by 0050's wider scope.
