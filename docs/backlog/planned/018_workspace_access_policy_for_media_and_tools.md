## 018_workspace_access_policy_for_media_and_tools (planned)

**Status**: Planned
**Priority**: High
**Depends on**: 005_abstractcore_integration (completed), 009_artifact_store (completed)

---

## Goal

Define and verify the runtime policy for local filesystem access used by LLMs,
agents, media inputs, generated outputs, and tool execution.

This item should determine which checks belong in AbstractRuntime, which checks
belong in host applications, and how the effective policy is represented in
durable run state and effect payloads.

---

## Context / Problem

AbstractRuntime can pass local media paths to AbstractCore and can execute tools
that read and write files. That is expected for local agency, but it must be
controlled consistently.

The default posture should be deny by default:
- all filesystem paths are blacklisted unless explicitly allowed
- the active local workspace folder is whitelisted by default
- callers may consciously add whitelist or blacklist entries
- a separate risky mode may allow access to the whole machine, limited only by
  the OS permissions of the user/group running the runtime process

This matters especially for remote AbstractCore execution, where a local media
path may be read by the runtime and sent to a remote server as request content.

---

## Proposed Investigation

1. Inventory every runtime path where filesystem content can be read or written:
   LLM media inputs, artifact materialization, generated media storage, tool
   calls, workspace-scoped tools, and attachment/session helpers.
2. Identify the current source of truth for workspace roots, allowlists,
   denylists, and risky full-machine access.
3. Decide whether raw path authorization should be enforced before effect
   handlers call providers, inside shared path helpers, or entirely by hosts
   before they submit workflows.
4. Define durable policy metadata for a run so replay, resume, and audit can
   explain why a path was allowed.
5. Make artifact refs the preferred boundary for externally supplied media;
   raw local paths should be trusted only after policy validation.

---

## Acceptance Criteria

- [ ] Document the effective default policy: deny all paths except the active workspace.
- [ ] Document how users consciously add whitelist and blacklist paths.
- [ ] Document the semantics and warning requirements for risky full-machine access.
- [ ] Determine the enforcement layer for raw path media passed to AbstractCore.
- [ ] Determine the enforcement layer for tool read/write access.
- [ ] Add tests or conformance fixtures for allowed workspace paths, denied external paths, explicit allowlist paths, explicit blacklist overrides, and full-machine mode.
- [ ] Ensure remote media forwarding cannot bypass the chosen path policy.

---

## Notes

- Policy should be expressed in terms of canonical resolved paths where possible.
- Blacklist entries should override broad whitelist entries.
- The active workspace should be explicit in runtime-visible configuration, not
  inferred from process current working directory alone.
- Full-machine access is intentionally risky and should require an explicit mode
  or consent signal.
