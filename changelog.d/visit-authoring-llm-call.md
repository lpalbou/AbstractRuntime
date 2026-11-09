### Changed

- **A visit's episode is formed with its authored title and summary** (`identity/visit_workflow.py`,
  `identity/entity_runtime.py`, `identity/authoring.py`; operator 2026-11-09: metadata generated with the memory).
  The durable visit workflow cannot call a model inside a node, so its formation step issues the authoring call as an
  LLM_CALL effect tagged `payload.capability = "summaries"` (`authoring_llm_call_payload`: system prompt, one user
  message with the exchange, `response_schema` {title, summary}, `params.thinking = "off"`, `_absorb_failure`)
  in a new `VISIT_AUTHOR` node between COMMIT and FORM; FORM forms the episode with the authored title and summary
  (`attributes.metadata_authored`), and only a failed call forms it pending (memory's task) with a failure line
  written once the record has its id. `open_entity_runtime(capability_handlers={cap: handler})` routes a tagged
  LLM_CALL to its capability handler and stamps the echo `result.capability`; a capability nobody serves is answered
  `{ok: False, capability_unrouted: <cap>}` with an error in the log (reason `capability_unrouted`), never by the
  mind. The fields are read from `result.data` (structured output) or strictly from `result.content` as JSON.
