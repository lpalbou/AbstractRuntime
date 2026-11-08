### Fixed

- **The session sheet and the authored world-model card are stored whole** (`identity/chat.py`,
  `identity/visit_workflow.py`; root ADR 0054, memory owner 2026-11-08: abstractmemory has no length caps on its
  columns and `author_world_model` stores text whole). The 160-character `_sheet_line` cut on each sheet moment
  (run vars, `pending_reflection.json`) and the 1,600-character `_bounded_card_text` cut on an authored briefing are
  gone; the sheet line is the record's digest byte for byte and the card is what the entity wrote. A PROMPT that
  lists the sheet (the look-back, the card-authoring prompt) folds each moment visibly at read time
  (`sheet_prompt_line`, `SHEET_PROMPT_LINE_MAX = 160`, "… [folded for this prompt; the record is whole]"): the
  prompt sees the whole conversation beside the sheet, so the sheet is an index of the moments and an unfolded one
  would add ~5x tokens per moment and no new words. The feel-marker renderer keeps reading the whole lines, so what
  rests in the reply is the record's words, never the fold.
