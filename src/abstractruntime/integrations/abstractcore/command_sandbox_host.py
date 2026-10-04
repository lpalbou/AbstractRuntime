"""Runtime-owned facade for AbstractCore's command-sandbox HOST API (round 12).

Hosts (AbstractGateway's `serve`) configure the command sandbox through this module instead of
importing `abstractcore.tools.sandbox` directly, keeping the host -> Runtime -> AbstractCore
boundary. Pure re-exports, no logic, so nothing can drift from Core:

- `configure_host(*, env, unsandboxed_commands_allowed=False)`: set once at boot (same values
  again = no-op; different values = RuntimeError).
- `host_policy()`: {configured, unsandboxed_commands_allowed, env_keys, kind} (names only).
- `host_sandbox_kind(posture="allowed_only")`: "macos-sandbox-exec" | "linux-bwrap" |
  "linux-landlock" | "none".
- `KIND_LABELS`, `KIND_NONE`.
- `reset_host_for_tests()`: tests only, forget the host policy.

Needs AbstractCore >= 2.25.0 (this package's floor); an older Core fails loudly on import.
"""

from __future__ import annotations

from abstractcore.tools.sandbox import (
    KIND_LABELS,
    KIND_NONE,
    _reset_host_for_tests as reset_host_for_tests,
    configure_host,
    host_policy,
    host_sandbox_kind,
)

__all__ = ["KIND_LABELS", "KIND_NONE", "configure_host", "host_policy", "host_sandbox_kind", "reset_host_for_tests"]
