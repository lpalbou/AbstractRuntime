# 0857 — Is Empty accepts lists (nothing or no elements = empty)

**Status**: completed 2026-10-28 on branch `fix/email-triage-qa`, UNRELEASED (PR held for the release lead: a runtime
release cascades into the gateway and root floors) · **Priority**: P2 · **Created**: 2026-10-28
**Cross-package**: AbstractFlow backlog 0160 (operator QA 2026-10-28 item (e)); the Flow editor lets Is Empty take a
list only after this is released (branch `fix/is-empty-lists` there).

## Why
Operator QA 2026-10-28 (e): "Is Empty must also accept arrays (undefined or 0 elements = empty)". The email-triage
flow checked its `triage` list variable with Array Length + Compare, and a Compare left on its default `==` inverted
the check. One node that says "the list is empty" is the readable way.

## Current code reality (before)
`visualflow_compiler/visual/builtins.py` `string_is_empty`: `"" if text is None else str(text)` then `== ""`, so
`[]` became `"[]"` and was never empty.

## Change
`string_is_empty`: `None` → true; `list`, `tuple`, `set`, `frozenset`, `dict` → `len == 0`; anything else → `str(value)
== ""` (unchanged for text; whitespace, `0`, `False` stay not empty). The node type and pin id (`text`) are unchanged,
so every saved flow compiles as before and a text input behaves as before.

## Evidence
- `tests/test_visualflow_string_contains_replace.py::test_is_empty_accepts_lists_and_objects` and `::test_is_empty_on_a_list_variable_in_a_compiled_flow` (Get Variable → Is Empty → If; RED before: `[]`
  returned False).
- The operator's email-triage flow with Array Length + Compare replaced by Is Empty(Get triage) → If false → Notify
  me, run through the executor with 5 emails / 2 important: reaches Notify me and publishes the alert
  (`untracked/release-closing/w6-flow/repro_email_triage.py` at the workspace root).
- Full suite: 3863 passed; the only failures are environment ones present on `main` (a real abstractvision plugin
  test, three modules importing the old Python `abstractflow.FlowRunner`).
