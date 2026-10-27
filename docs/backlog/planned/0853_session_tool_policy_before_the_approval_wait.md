# 0853-abstractruntime: [FEATURE] Session tool policy: a conversation's approve/deny applied before the approval wait

> Created: 2026-10-27
> Status: Planned (implemented on branch `feat/session-tool-policy`; ships with the next release)
> Type: feature
> Priority: P1
> Labels: tool-approval, ledger, replay, sessions
> Root: framework 1061 (cross-repo); related framework 0993 (one ledger replay, many views)

## Summary

"Approve all" / "approve this kind" becomes a durable, server-side session policy that the
TOOL_CALLS handler consults before deciding whether a batch waits. Covered calls run with no wait
and no interaction; denied calls are refused with no wait. Each decision is an
`abstract.tool_policy_decision` ledger record.

## Why

Operator, 2026-10-27, verbatim: "when a user gives permission to either all tools of a kind (e.g.
fetch url) OR all tools (all available/authorized for the user and authorized by the gateway), I
still see very quickly the gate approval showing up in apps and disappearing. Before sending the
gate approval, check whether the tool is already approved for the current session, and if so, do
not send the gate approval at all."

Before: the runtime knew only the start-time `_runtime.tool_policy`
(`effect_handlers.py` `_execute_with_run_policy_unscoped`), so it emitted `approval_required` for
every gated call; clients answered it after it arrived (panel-chat `autoApproveCurrentWait`,
desktop `should_auto_approve_tool_batch`, iPhone `applyConversationTrust`). The gate showed in
every client and in the replayable ledger.

## What ships (branch `feat/session-tool-policy`)

- `src/abstractruntime/storage/session_tool_policy.py`: `SessionToolPolicyStore` protocol,
  `InMemorySessionToolPolicyStore`, `SqliteSessionToolPolicyStore` (revision, `if_revision`
  conflicts, append-only history), `session_policy_names`.
- `Runtime(session_tool_policy_store=...)`, `Runtime.set_session_tool_policy_store()`; the tick
  attaches it to the run (`_runtime_session_tool_policy_store`).
- `effect_handlers.py`: `_session_tool_policy_for_run` (scope rules), deny in the pre-pass,
  approve inside `_execute_with_run_policy` (both the run-policy and the static branch),
  `_append_tool_policy_record` (no record, no auto-approval).
- Docs: `docs/tool-approval.md#session-tool-policy-a-conversations-standing-word`.
- Tests: `tests/test_session_tool_policy.py` (no wait + audit record, delegated child, mid-run
  change with the one-off path intact, deny, mixed batch still waits, start-time require list
  overridden, out-of-scope runs keep asking, reset, restart via sqlite, no record means no
  auto-approval, store revisions/history). Mutation-checked.

## Remaining

- Release (minor bump: new public API) and the gateway's `ABSTRACTRUNTIME_FLOOR` raised to it.
- Covering tools enabled after an `all` grant (today they ask until the person grants again).
- Answering waits that already exist when a policy is written (today the client answers the shown
  one).

## Acceptance criteria

- [x] No `approval_required` wait for a session-approved call, in the root run or a delegated
      child; an audit record instead.
- [x] Deny refuses with no wait; out-of-scope runs unchanged.
- [ ] Released, and the gateway floor raised.
