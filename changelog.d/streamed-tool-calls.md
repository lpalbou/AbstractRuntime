### Fixed

- **A streamed tool call keeps its arguments** (`integrations/abstractcore/llm_client.py`). An LLM call made
  with `stream: true` could record a tool call with the right name and empty arguments when the provider
  streamed OpenAI-style deltas (a named first piece, then fragments of the arguments) or re-sent a growing
  call under the same id. The streaming fold now joins argument fragments onto their call by id or index,
  never drops a fragment that repeats an earlier one, and keeps the latest version of a call that is re-sent,
  so the recorded tool calls match the non-streamed result. Complete calls on separate chunks and re-sent
  full lists are handled as before.
