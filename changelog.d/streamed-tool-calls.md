### Fixed

- **A streamed tool call that is sent again keeps its latest version** (`integrations/abstractcore/llm_client.py`).
  When a provider re-sent a call under the same id, the streaming fold kept the first version it saw, so a call
  that was streamed as it grew was recorded with incomplete arguments. The fold now keeps the latest version, and
  the recorded tool calls match the non-streamed result. Calls delivered on separate chunks and re-sent full
  lists are recorded as before. With AbstractCore 2.37.0 or newer, the providers deliver each streamed native
  tool call once and complete; with an older AbstractCore, a call streamed by an OpenAI-compatible server can
  still be recorded with its name and no arguments.
