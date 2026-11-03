### Changed

- **Compare reads text next to a number exactly, and refuses what is not a number instead of answering False**
  (`visual/builtins.py` `control_compare`, ADR 0049 as amended 2026-11-03). When one side is a number or true/false
  and the other is text, the text is read as that type by the Flow editor's literal grammar (`_exact_from_text`: a
  decimal number, ASCII, no `_`, finite; or `true`/`false` in any case), so `5 == "5"` and `"9" < 10` are true. This
  covers flows saved by AbstractFlow 0.13.0 (a typed value stored as text next to a wired number) and text that
  arrives at run time from a form, an API input or a model, with no re-save. Before, these answered False silently.
  Text that is not such a value (`5 == "five"`, `"1_000"`, `"0x10"`, `"inf"`, `"nan"`, empty text), and
  `<`/`<=`/`>`/`>=` between values Python cannot order (`None > 3`, a list and a number), now fail the step with an
  error naming the node: "b is text ('five'), not a number; a is a number (5)." Text against text is never converted
  (`"10" > "9"` compares text). Equality of same-type values and Python's equality across other types (`None == 5`)
  are unchanged; true/false compares with numbers as Python's booleans do (`True == 1`). The conversion table is
  `tests/fixtures/visualflow/compare_text_operands.json`.
