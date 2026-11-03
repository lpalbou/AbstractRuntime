### Fixed

- **A tool that reports its failure as a JSON object keeps its explanation** (`integrations/abstractcore/tool_executor.py`,
  framework backlog 1005). A failed result's `output` used to be emptied when the tool returned its failure as a JSON
  string, so the agent and the run ledger saw only the error line (for a web search: "No results found", never the
  backends tried, the bot-challenge status or the hint). The parsed object is now the result's `output`, as for a tool
  that returns a dict; a plain text error (`Error: ...`) still keeps an empty output and its message in `error`.
- **The visual-flow Agent node shows the model that explanation** (`visualflow_compiler/compiler.py`). A failed call
  whose `output` is an object is rendered as its error line followed by the object (`[web_search]: Error: No results
  found` then the JSON), under the same `agent_observation_max_chars` limit; a plain text failure still shows its error
  line alone.
- Failure payloads are now recorded in the run ledger and the history bundle like success payloads: whatever a tool puts
  in its failure JSON (its parameters included) is stored with the run.
