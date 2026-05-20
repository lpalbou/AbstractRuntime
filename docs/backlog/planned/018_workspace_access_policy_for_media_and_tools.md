## 018_workspace_access_policy_for_media_and_tools (planned)

**Status**: Planned
**Priority**: High
**Depends on**: 005_abstractcore_integration (completed), 009_artifact_store (completed), 023_truthful_local_media_residency_boundaries (completed), 024_runtime_owned_run_scoped_media_execution (completed)

---

## Goal

Close the remaining filesystem-policy gap so raw local media inputs follow the same runtime-visible path rules as tool calls, while documenting the tool/workspace policy Runtime already ships today.

---

## Current code reality

- `WorkspaceScope` already defines `workspace_root`, access modes, allowlists, and denylists.
- Tool-call handling already rewrites and blocks filesystem-style tool arguments under that policy.
- Subworkflows already inherit workspace policy durably.
- There are already tests for restart persistence, allowlist mode, and mount-style allowed roots.
- The remaining gap is that raw `LLM_CALL` media strings are still treated as local file paths, and remote media docs still describe flows where local paths can be forwarded.
- `execute_command` is still explicitly not an OS sandbox, so policy must stay honest about what Runtime can and cannot guarantee.

This means the old item framing is stale: Runtime does not need a greenfield workspace policy design for tools, but it still needs to close the raw-media path hole and document the shipped policy truthfully.

---

## Problem

Tool calls and some attachment flows already honor a durable workspace policy. Raw media paths in `LLM_CALL` do not yet clearly do the same. That creates a boundary mismatch:

- a tool read may be blocked by workspace policy
- the same file may still be read and forwarded as media if passed as a bare string path

That inconsistency is the real high-priority gap.

---

## Planned scope

1. Reuse or extend the shared workspace policy helpers for raw media-path authorization.
2. Decide and document the preferred boundary:
   - artifact refs for externally supplied media
   - bare local paths only after workspace-policy validation
3. Ensure remote media forwarding cannot read files outside authorized roots.
4. Document the already-shipped tool/workspace behavior honestly, including risky modes and `execute_command` limitations.
5. Add conformance tests that cover tool paths and media paths under the same policy vocabulary.

---

## Acceptance Criteria

- [ ] Document the effective default policy Runtime already enforces for tool paths.
- [ ] Document how users consciously add allowlist and denylist paths.
- [ ] Document the semantics and warnings for risky broader-access modes.
- [ ] Raw local media paths are either validated under the same policy or explicitly rejected in favor of artifact refs.
- [ ] Remote media forwarding cannot bypass the chosen path policy.
- [ ] Add tests or conformance fixtures for allowed workspace paths, denied external paths, explicit allowlist paths, explicit denylist overrides, and media-path validation.

---

## Notes

- Policy should be expressed in terms of canonical resolved paths where possible.
- Denylist entries should override broad allowlist entries.
- The active workspace should be explicit in runtime-visible configuration, not inferred from process current working directory alone.
- `execute_command` remains a risky capability even under path rewriting; this item is about boundary truthfulness, not full sandboxing.
