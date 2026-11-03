### Changed

- **Compare reads text next to a number or true/false exactly** (`visual/builtins.py` `_exact_from_text`, ADR 0049 as
  amended 2026-11-03: "we must have automatic conversion"). When one side is a number or true/false and the other is
  text, the text is read as that type by the Flow editor's literal grammar (a decimal number, ASCII, no `_`, finite,
  a whole number kept exact beyond 2**53; or `true`/`false` in any case), so `5 == "5"` and `"9" < 10` are true. This
  covers flows saved by AbstractFlow 0.13.0 (a typed value stored as text next to a wired number) and text arriving at
  run time from a form, an API input or a model, with no re-save. `"1_000"`, `"0x10"`, `"inf"`, `"nan"`, full-width
  digits and empty text are refused. Text against text is never converted (`"10" > "9"` compares text). The table is
  `tests/fixtures/visualflow/compare_text_operands.json`.
