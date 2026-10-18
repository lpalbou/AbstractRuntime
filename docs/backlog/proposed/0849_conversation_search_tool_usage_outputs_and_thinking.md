# 0849 — Conversation search: tool usage, tool outputs and intermediate thinking

- **Status**: PROPOSED 2026-10-18
- **Owner**: runtime (`conversation_search.py`), surfaced by the gateway (`GET /api/gateway/sessions/search`)
- **Origin**: operator, while building conversation search for the Apple Watch (2026-10-18): the
  index holds each turn's question, answer and media (generated artifacts' kind and prompt,
  attachments' kind and file name). These three sources were weighed and left out on purpose.

## What is indexed today

`ConversationSearch` keeps one row per turn root: the question and the answer as the session replay
builds them, and the turn's media. An FTS5 `trigram` index over text lowercased and without accents;
exact matches first, then a fuzzy pass; filters (archived, session purpose, kinds) inside the query.

## Proposed sources

1. **Tool usage per turn** — the tools a turn called and their key arguments: search terms, URLs,
   file names, commands ("the conversation where it read the Apple support page"). Source: the
   turn's agent run(s) `tool_calls` ledger records (`effect.payload.tool_calls[{name, arguments}]`),
   read once when the turn is indexed (a turn's ledger does not change after it ends). Cost: one
   ledger read per new turn; agent ledgers are 0.2–3 MB, mostly thinking records, so the read needs
   a records-of-one-type path (or the compact ledger mode the watch's live activity also wants).
   Index the arguments only (name + the argument values a person would recognise, cut at ~200
   characters each), never the results.
2. **Tool outputs** — what tools returned (pages read, command output, file contents). Large (30–90
   KB per call), noisy (a web page matches most words), and sensitive (file contents, command
   output, tokens printed by a command would become searchable from a search box). If ever
   indexed: opt-in per account, outputs cut short, and a redaction pass for secrets.
3. **Intermediate thinking** — the agent's own words between tool calls (`result.content` of its
   thinking steps; the last one is the answer, already indexed). Small and often useful ("I'll
   check the logs first"), but it is the model's working text, not what the person was told; mixed
   into results it would surface claims the answer later corrected. If indexed: its own field,
   weighted below the answer, shown as "while working".

## Open questions

- One ledger read per turn at index time vs. a runtime-maintained per-turn digest written when the
  turn ends (cheaper reads, one more write path).
- Whether the gateway's compact ledger mode (proposed for the watch's live activity) should be the
  common source for both.
