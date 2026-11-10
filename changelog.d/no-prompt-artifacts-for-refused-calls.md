### Fixed

- **A call refused before it ran saves no prompt artifacts** (castor live E2E 2026-11-10, e2e-restart3 C1_07/C1_11).
  After a Stop landed mid-turn, the refused model call's node trace still carried its whole payload, and the trace's
  bounding offloaded ~48 KB of the prompt (messages, system prompt, tools) into artifact files after the Stop. The
  fence's refusal at dispatch now marks its cancelled outcome `ran: False`, and the Runtime's node trace records such
  an effect without its payload (type, result key and the cancellation kept; nothing offloaded). The ledger's own
  records of the cancelled act are unchanged.
