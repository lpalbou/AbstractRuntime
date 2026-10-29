# 0865 — A `model_loaded` ledger event with the reserved context

> Package: AbstractRuntime (with AbstractCore residency claims)
> Type: task
> Created: 2026-10-29
> Priority: normal
> Labels: residency, ledger, observability

## Summary

Operator ruling (2026-10-29): when a model loads it reserves the maximum context that fits (AbstractCore ADR 0010
point 8), and "never silently: the choice is written to the ledger (a model_loaded event with model, advertised context,
reserved context, free memory, buffer, reason) and to the log". AbstractCore already logs it and carries it on the
provider's residency claim (`context_reservation`: advertised, reserved, free_bytes, buffer, reason). The ledger event
is not written yet: a load happens inside a client build, outside any run's effect, so there is no step record to
attach it to.

## Scope

- When a pooled client is built for a local model (`MultiLocalAbstractCoreLLMClient._get_client`), read the provider's
  `context_reservation` and append one `model_loaded` record to the ledger of the run that triggered the build (or a
  host-level ledger when no run did), with model, advertised context, reserved context, free memory, buffer, reason.
- The replay view and Observer show it as one readable line.

## Acceptance criteria

- A chat that loads the default model leaves one `model_loaded` record with the numbers; a test is red without it.
