"""Runtime-owned facade for AbstractCore's command-sandbox HOST API (round 12).

Hosts (AbstractGateway's `serve`) configure the command sandbox through this module instead of
importing `abstractcore.tools.sandbox` directly, keeping the host -> Runtime -> AbstractCore
boundary. Re-exports, no logic of its own, so nothing can drift from Core:

- `configure_host(*, env, unsandboxed_commands_allowed=False)`: set once at boot (same values
  again = no-op; different values = RuntimeError).
- `host_policy()`: {configured, unsandboxed_commands_allowed, env_keys, kind} (names only).
- `host_sandbox_kind(posture="allowed_only")`: "macos-sandbox-exec" | "linux-bwrap" | "windows-restricted-token" |
  "linux-landlock" | "none".
- `host_unprotected_volumes()`: {drive: sentence} for the drives the Windows sandbox cannot protect
  (no access lists, as exFAT, or a network share); `{}` elsewhere and with an AbstractCore that does
  not report them (before 2.39.0).
- `KIND_LABELS`, `KIND_NONE`.
- `reset_host_for_tests()`: tests only, forget the host policy.

Needs AbstractCore >= 2.25.0 (this package's floor); an older Core fails loudly on import.
"""

from __future__ import annotations

from typing import Dict

from abstractcore.tools import sandbox as _core_sandbox
from abstractcore.tools.sandbox import (
    KIND_LABELS,
    KIND_NONE,
    _reset_host_for_tests as reset_host_for_tests,
    configure_host,
    host_policy,
    host_sandbox_kind,
)


def host_unprotected_volumes() -> Dict[str, str]:
    """AbstractCore's `windows_unprotected_volumes()` (read at call time), or `{}` when this Core
    has none."""
    probe = getattr(_core_sandbox, "windows_unprotected_volumes", None)
    return dict(probe()) if callable(probe) else {}


__all__ = ["KIND_LABELS", "KIND_NONE", "configure_host", "host_policy", "host_sandbox_kind", "host_unprotected_volumes", "reset_host_for_tests"]
