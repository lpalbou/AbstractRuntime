### Fixed

- **A tool that reports its failure as a JSON object keeps its explanation** (`integrations/abstractcore/tool_executor.py`,
  framework backlog 1005). A failed result's `output` used to be emptied when the tool returned its failure as a JSON
  string, so the agent and the run ledger saw only the error line (for a web search: "No results found", never the
  backends tried, the bot-challenge status or the hint). The parsed object is now the result's `output`, as for a tool
  that returns a dict; a plain text error (`Error: ...`) still keeps an empty output and its message in `error`.
