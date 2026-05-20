## ADR 0004: Runtime owns run-scoped media execution truth

### Status
Accepted (2026-05-20)

### Context
Hosts sometimes need to trigger image generation, TTS, or STT for an existing run after the main workflow was already created.

If a host executes that media work directly and only appends ledger-like records afterward, Runtime is no longer the
source of truth for:

- what was requested;
- what provider/model path actually executed;
- which artifacts were produced;
- how replay should represent the step.

We also observed that local media residency can be misunderstood as successful warmup even when the local execution mode
is a one-shot subprocess boundary that cannot reuse the previous warm state.

### Decision
AbstractRuntime owns run-scoped media execution truth.

Specifically:

- Hosts must route run-scoped image/TTS/STT work through Runtime, not execute it out-of-band.
- The integration provides a public durable run facade that creates Runtime-owned child runs for these operations.
- Operator/control-plane prompt-cache and model-residency controls remain separate host-oriented helpers and do not
  create durable run history by themselves.
- Local unsupported media residency must be reported explicitly with machine-readable fields such as
  `execution_mode`, `requires_long_lived_server`, and `config_hint`.
- Media-only normalized results must separate orchestration identity from actual media backend identity via
  `runtime_provider` / `runtime_model` and `media_provider` / `media_model`.

### Consequences
- Gateway/other hosts have a clean migration target for run-scoped media routes without synthesizing history after the
  fact.
- Ledger and artifact truth stay Runtime-authored and replay-friendly.
- Optional local media warmup can remain non-fatal while still rendering honestly as a warning/no-op.
- Consumers can distinguish the runtime orchestration model from the media backend that produced the output.

### See Also
- Implementation: [`../backlog/completed/023_truthful_local_media_residency_boundaries.md`](../backlog/completed/023_truthful_local_media_residency_boundaries.md)
- Implementation: [`../backlog/completed/024_runtime_owned_run_scoped_media_execution.md`](../backlog/completed/024_runtime_owned_run_scoped_media_execution.md)
- Integration guide: [`../integrations/abstractcore.md`](../integrations/abstractcore.md)
