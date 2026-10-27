# ADR 0008: Runtime integration facades are the complete host surface for AbstractCore

Status: Accepted.
Date: 2026-10-27

## Context

Hosts (AbstractGateway first) depend on AbstractRuntime, and AbstractRuntime owns the
AbstractCore integration (ADR 0001, 0004-0007). The gateway enforces its side with a test that
fails on any direct `abstractcore` import. That rule only holds if every AbstractCore name a host
legitimately needs is reachable through a runtime facade. On 2026-10-27 the gateway's mailbox
code needed AbstractCore's SRV lookup (`resolve_srv_default`), the email facade did not re-export
it, and the host imported AbstractCore directly: the boundary test went red and stayed red
across a release line. The hole was not the import; it was a facade that lagged behind what the
host needed, with no rule saying whose job it was to close the gap.

## Decision

1. The modules under `abstractruntime.integrations.abstractcore` whose name ends in `_facade`
   (discovery, host, config, run, email, structured, voice, telegram, comms, mcp, tool inventory,
   server, data registry) are the ONLY surface through which a host reaches AbstractCore. They
   are complete by contract: a name a host needs and cannot import from a facade is a runtime
   defect, fixed by adding the re-export or method in the runtime and cutting a patch release.
2. A host never works around a missing name by importing `abstractcore` directly, vendoring the
   code, or guessing a replacement. It raises the gap to the runtime (backlog item or PR) and
   pins the runtime floor that carries the fix.
3. A re-export is unchanged in identity (`facade.name is abstractcore.module.name`) and listed in
   the facade's `__all__`; the facade documents which AbstractCore version introduced the name.
4. The facades are additive within a minor line: a name, once exported, is not removed without
   a deprecation note and a major or minor bump.

## Consequences

### Positive
- The host's boundary test is a true gate: green means every AbstractCore access is a runtime
  facade, so a Core change breaks at most one package.
- Version floors stay layered: a host pins the runtime, the runtime pins Core.

### Negative
- A host feature that needs one new Core name waits for a runtime patch release (about 20
  minutes with the release workflow), and the host's runtime floor moves with it.
- The runtime carries thin re-exports it does not use itself.

### Neutral
- Patch releases of the runtime for facade additions are normal, not exceptional.

## Enforcement
- Code review: a runtime PR that adds a host-facing Core name adds it to the facade's `__all__`
  (and to a names list such as `email_facade.GATEWAY_NAMES` where one exists) with a one-line
  comment naming the Core version and the host need.
- Hosts cite this ADR in their own boundary rule (AbstractGateway ADR 0001) and never add an
  exemption to their import-boundary test for a name a facade could carry.
- A facade gap found during host work becomes a runtime backlog item or PR before the host
  ships; the host's changelog names the runtime version that closes it.

## Validation
- `tests/test_email_review_fixes.py`: the email facade's export set equals Core's plus the
  runtime's additions, every `GATEWAY_NAMES` entry imports, and re-exports keep identity.
- AbstractGateway `tests/test_gateway_import_boundary.py` stays green against the released
  runtime floor the gateway pins.
- Release check: the runtime's docs list each facade's host-facing names
  (`docs/integrations/abstractcore.md`, `docs/api.md`).

## Backlog links
- Originating: AbstractGateway backlog 1044 (mailbox Read and write access) needed
  `resolve_srv_default`; closed by runtime 0.13.1 (`email_facade.resolve_srv_default`).
- Adoption: AbstractGateway `tests/test_gateway_import_boundary.py` (backlog 0059).

## Related
- ADR 0001 (layered coupling), 0004 (media execution), 0005 (discovery queries), 0006 (bloc
  prompt cache): the per-area facades this ADR generalises.
- `src/abstractruntime/integrations/abstractcore/email_facade.py`, `voice_facade.py`,
  `discovery_facade.py`.
- AbstractGateway ADR 0001: AbstractCore access goes through AbstractRuntime facades.
