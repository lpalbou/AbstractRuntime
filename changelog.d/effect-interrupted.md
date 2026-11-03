### Added

- **`EffectInterrupted`: a host stop interrupts an effect, it never fails it** (`core/effect_cancellation.py`,
  exported as `abstractruntime.EffectInterrupted`; gateway backlog 1103). A host that stops on purpose (a gateway stop
  or restart, a model-host restart for an update) raises it into the effect in flight. The runtime writes no terminal
  record and re-raises: the ledger tail stays STARTED, the run stays RUNNING at its last durable step, and the next
  tick (after the host is back) sends the effect again under the same idempotency key as attempt 1, so no retry is
  consumed and nothing reaches an automation's outcome. A BaseException like `EffectKilled`, so no `except Exception`
  on the way turns it into a failure. The interrupted call's live text ends `cancelled` with `detail: "interrupted"`.

### Fixed

- **An automation occurrence whose agent completed with `success: false` names the agent's reason**
  (`automations/attention.py`). The attention item and the `automation_failed` email said "completed" (the run's status
  word); the reason is now read from the Agent node's `meta.error`, and when there is none it says so plainly.
