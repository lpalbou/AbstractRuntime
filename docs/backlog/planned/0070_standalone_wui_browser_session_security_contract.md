## 0070_standalone_wui_browser_session_security_contract (planned)

**Status**: Planned
**Priority**: High
**Depends on**: ADR-0002 (accepted), 020_runtime_gateway_install_boundary (completed), 021_runtime_gateway_env_namespace_cleanup (completed)

---

## Goal

Make the Runtime-owned contract for the standalone `agora-wui` browser/session lane explicit: Runtime must remain a durable workflow kernel behind a same-origin WUI server shell, not quietly become the browser auth/session authority.

---

## Current code reality

- Runtime already documents the explicit host handoff boundary for remote and hybrid execution.
- Runtime already refuses to read `ABSTRACTGATEWAY_*` configuration directly or reinterpret Gateway bearer tokens as Core server/provider credentials.
- Artifact descriptor metadata already redacts obvious secret fields such as bearer, CSRF, refresh, id, and session tokens.
- Runtime observability traces already redact provider API keys and the same family of token-shaped parameters at the top level.
- Agora tool identity already uses a non-secret runtime alias (`_runtime.agora_agent`) while the actual hub key stays in host environment configuration.
- What does **not** exist yet is a Runtime-local artifact that says, in one place, which parts of the browser/session story belong to the same-origin WUI shell and which parts Runtime explicitly refuses to own.

---

## Problem

`commons/plan/agora-ui.md@6` now makes the standalone Agora contract primary and the Framework-native embedding path secondary. That still leaves one Runtime-specific risk:

- a host or future web package may assume the browser can hand Runtime bearer-style browser authority directly;
- a future adapter may blur browser session lifecycle, host-origin policy, and runtime/provider credentials into one bucket;
- the package may keep the right behavior in code today but fail to publish a conformance gate that proves the boundary stays intact during the `agora-wui` extraction.

Without a Runtime-owned contract, the migration can drift into "the browser session just is the runtime credential," which is exactly the boundary Runtime has already rejected elsewhere.

---

## Planned scope

1. Write the Runtime-side contract for the standalone WUI/browser-session/server-shell lane.
2. Make the primary invariants explicit:
   - browser auth/authorship stays behind a same-origin server shell;
   - Runtime receives explicit host-constructed call context, not ambient browser cookies or bearer authority;
   - Runtime does not own login/logout/CSRF/origin/host policy for the web product itself;
   - persisted runtime observability and artifact metadata must redact browser-session secret material.
3. Define principal/runtime isolation expectations for the server-shell handoff:
   - one browser principal must not silently inherit another principal's runtime/session authority;
   - runtime/session routing has to be explicit in the host handoff;
   - logout/revocation must stop new server-shell exchanges even if Runtime remains alive underneath.
4. Keep any Continuum/Gateway/App-server embedding rules as a clearly secondary adapter lane over the same standalone contract.
5. Add focused Runtime regression coverage for the redaction and explicit-boundary pieces Runtime already owns in code.

---

## Acceptance Criteria

- [ ] Runtime docs name the standalone WUI same-origin server shell as the owner of browser login/logout/cookie/CSRF/origin behavior.
- [ ] Runtime docs state that browser/session credentials are not Runtime provider credentials and are not a substitute for explicit host-to-Runtime auth/context handoff.
- [ ] Runtime docs state that the Framework-native embedding path is an optional secondary adapter lane, not the definition of the standalone package contract.
- [ ] Runtime-visible observability traces redact bearer, CSRF, refresh, id, and session token fields.
- [ ] Artifact descriptor metadata redacts the same family of browser-session secret fields.
- [ ] Tests cover the redaction behavior the contract relies on.

---

## Non-goals

- Do not make `abstractruntime` the product owner of `agora-wui`.
- Do not make Runtime the cookie/session store or the browser-origin enforcement point.
- Do not force standalone `agora` / `agora-tui` / `agora-wui` interoperability to depend on Framework-native embedding.
- Do not redefine the general Agora package-to-package contract inside Runtime.

---

## Current code pointers

- `docs/adr/0002_execution_modes_local_remote_hybrid.md`
- `docs/integrations/abstractcore.md`
- `src/abstractruntime/integrations/abstractcore/effect_handlers.py`
- `src/abstractruntime/storage/artifacts.py`
- `src/abstractruntime/integrations/abstractcore/agora_tools.py`
- `tests/test_provider_endpoint_profile_resolution.py`
- `tests/test_abstractcore_run_facade.py`

---

## Notes

- The same-origin WUI server shell may be a standalone package artifact outside this repository; this item only captures the Runtime contract that such a shell must honor.
- The safest wording is "standalone contract first, Framework adapter second." Runtime should keep mirroring that split in both code and documentation.
