### Fixed

- **An authoring failure line says what failed and on which model** (`identity/authoring.py`, `identity/chat.py`,
  `identity/visit_workflow.py`). When the summaries capability client RAISES, the line's reason is `client_error`
  (was `malformed`) and it carries `error_class` / `error_message` (the client's own exception, not the wrapper),
  `provider` / `model` of the capability client used (the drainer wrote None) and `capability: summaries`.
  `malformed` now means only a reply that came back and did not fit the title and summary (missing fields, a pydantic
  `ValidationError`); `memory_refused` is a refused authored write (`AuthoringRefused`). The visit lane names the
  capability call's own provider/model when its result carries them, never the visit's mind in their place.
  `authoring_failure_fields(exc, llm)` is the one classifier.
