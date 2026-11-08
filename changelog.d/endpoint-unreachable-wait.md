### Added

- **An unreachable model endpoint parks the run; it no longer fails it** (`integrations/abstractcore/
  effect_handlers.py`, `integrations/abstractcore/llm_client.py`; root 1168, operator 2026-11-08). When the provider
  raises AbstractCore's `EndpointUnreachableError` (no TCP connection to the inference endpoint: the machine is off,
  asleep or unreachable), the LLM_CALL handler returns a durable EVENT wait — `wait_key = endpoint_unreachable:
  <base_url>`, `resume_to_node` = the node that issued the call, no prompt (a signal wait, presented as "Waiting
  for …", never as a question), `details {kind, message, base_url, provider, cause, detail, errno, elapsed_s,
  retryable, attempt, since, node_id}`, the reason text `the model endpoint <base_url> did not answer (<cause>:
  <detail>, after <t> s); waiting for it`. The ledger keeps one WAITING record per attempt; `_runtime.endpoint_waits`
  counts the attempts per node and is cleared when the call goes through. A `resume` with no payload re-plans the
  node and re-issues the identical call as a new effect issuance (the parked attempt is never replayed as a result);
  `cancel` ends the run as today; any other provider error fails as before. Core never counts the error in its
  breaker and the runtime records no failure. The remote client (`RemoteAbstractCoreLLMClient`) rebuilds the typed
  error from the AbstractCore server's 503 envelope (`error.code == "endpoint_unreachable"` with every field); a
  generic 500 stays the error it was. `detail` is recovered from core's composed message because the envelope does
  not carry it yet. The gateway's endpoint prober issues the resume; the kit's wording is its own.
  Imports `EndpointUnreachableError` by name: this needs AbstractCore at or after core main 6c1e7a0e (the class, the
  503 envelope and its pickle `__reduce__`), unpublished at the time of writing — the runtime stays local until core
  publishes and the floor in `pyproject.toml` is raised then. `docs/architecture.md` (runtime loop) describes the wait.
