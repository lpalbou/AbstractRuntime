# 0845 — Session prompt cache: resume on re-attach, purge explicitly, never recompute silently

**Status**: planned · **Priority**: P1 · **Created**: 2026-09-18
**Package**: abstractruntime (contract owner) · **Slices**: abstractgateway (surface), abstractcore (store)

## Why

Operator ruling, 2026-09-17, verbatim:

> "when i re-attach to a session, the session normally still exists somewhere. if it was
> active and especially, then it still has a cache. if i ask a question to a session that
> was stopped, it should resume with the cache instead of recomputing everything, except
> if the cache doesn't exist anymore (eg was purged)"

and, on why the runtime owns this:

> "the gateway is what activates or pause the runtime... but nothing ever gets canceled if
> the gateway is paused of killed since... it's on the runtime and the runtime is durable
> by design.. so yes everything should resume with the same parameters settings etc."

A durable run store plus a volatile, invisible, unnamed cache is a half-durable session:
the transcript resumes and the compute does not. The operator experienced this as a
follow-up question taking 113 s where the previous turn took 7 s.

## What already landed (2026-09-17, uncommitted at time of writing)

These removed the two *accidental* ways a live session lost its cache. They are not the
contract this item asks for.

- **The cache key no longer moves when the workflow is republished.**
  `_derive_prompt_cache_key` hashed `workflow_id`, which embeds the bundle version
  (`…@0.0.3:c53b1579`, or `…_0_0_3_c53b1579_assistant_agent` for the agent sub-workflow).
  A republish bumped the version, so the same session asked for a different key on its next
  turn and its warm cache was never looked up again — observed live as
  `session:75e5050b…` becoming `session:cc044849…` mid-conversation.
  Fixed by `_workflow_identity_for_prompt_cache`
  (`src/abstractruntime/integrations/abstractcore/effect_handlers.py`).
- **A rebuilt host no longer orphans the cache.** AbstractCore now shares one loaded model
  *and its prompt-cache store* process-wide per model, so a new provider instance adopts
  the caches of the model it adopts (see abstractcore item 0847).

## The gap

1. **No resume across a process restart.** Every cache is in-process. A gateway restart
   (or a crash, or `SUP_HANG_KILL`) silently drops every session's KV state, and the next
   turn does a full prefill with no explanation. AbstractCore already has the primitives
   (`prompt_cache_export` / durable blocs / `prompt_cache_exports` root dir) and the
   runtime already passes `prompt_cache_export_root_dir` — nothing uses them for sessions.
2. **No visibility.** A caller cannot ask "does this session still have a cache, how big,
   and on which model?". The only signal is `metadata.prompt_cache` on a completed
   `llm_call` — after the cost has been paid. There is no session-level surface.
3. **No purge, and no eviction honesty.** `PromptCacheStore` is an LRU bounded at 32
   entries by default. Now that the store is shared per model, those 32 entries are shared
   by *every session and every node* of that model: a handful of concurrent sessions can
   silently evict each other, which reads to the operator exactly like the bug that was
   just fixed. There is no explicit purge, no per-session reservation, and no record that
   an eviction happened.
4. **No stated lifetime.** Nothing says how long a stopped session keeps its cache, or
   what a "purged" cache means for the next turn.

## Scope

### In scope

- A named session-cache lifecycle in the runtime: `resume` (default), `purge`, and the
  rule that a purged/evicted/absent cache is REPORTED, never silently recomputed
  (`prompt_cache.outcome` already distinguishes `cold` / `rebuilt` / `hit_restore`; lift it
  to the session surface and to the run's first ledger record).
- Session-cache introspection: given a `session_id`, report the keys it owns, their model,
  token counts, last use, and whether they are resident — resolvable without running a turn.
- Explicit purge (per session, per model) that the gateway can expose, and that is the ONLY
  thing that may drop a live session's cache besides its own eviction policy.
- Eviction honesty: when a session's key is evicted by the shared LRU/TTL, record it so the
  next turn can say "recomputed because the cache was evicted at HH:MM" instead of being
  indistinguishable from a bug.
- Per-session fairness in the shared store: either a reservation (N keys per session) or a
  bound expressed in bytes with a documented victim-selection rule.
- Decide and document the restart story: either (a) durable export/import of the session
  prefix on a bounded schedule, or (b) an explicit "caches do not survive a restart"
  contract, surfaced to clients so they stop implying otherwise. Do not leave it implicit.

### Out of scope

- Rewriting the gateway's host-rebuild path (abstractgateway 0846).
- Cross-process/shared-memory KV caches, or a cache server.
- Any change to prefix identity (`prompt_cache_prepare_modules` + fork) — that lane is
  correct and was fixed separately.

## Acceptance criteria

- [ ] Re-attaching to a stopped session and asking a question reuses its cache; the ledger
      shows `hit_restore` with `cached_tokens` ≈ the prior turn's context, and the wall time
      is within ~2× of a warm turn (measured on the real assistant session shape: ~7 s vs the
      113 s regression that prompted this item).
- [ ] A purged session reports `outcome=cold` **with a stated reason** (`purged` /
      `evicted` / `restart` / `model_unloaded`), visible without reading provider metadata.
- [ ] Concurrent sessions on one model cannot silently evict each other under the default
      bound; the policy and its limits are documented and tested with N sessions > bound.
- [ ] Session-cache introspection returns the truth for a live, a stopped and a purged
      session, and is cheap (no model touch, no generation).
- [ ] The restart story is documented and tested for whichever option is chosen.

## Evidence

- Live session `sess_02fc09701ab24853a92960852427e351` (2026-09-17): key
  `session:75e5050b…` for eight agent calls 21:25–21:31, then `session:cc044849…` at 21:55
  after two workflow republishes; the 21:54 `route_call` took 92 s against 7.5 s warm.
- `PromptCacheStore(max_entries=32)` — `abstractcore/providers/base.py`.
- Cache/weight sharing measurements: `abstractcore` item 0847.
