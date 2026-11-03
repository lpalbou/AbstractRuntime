### Changed

- **Compare refuses values it cannot compare instead of answering False** (`visual/builtins.py` `control_compare`,
  ADR 0049). Text that is not a number next to a number, or not `true`/`false` next to true/false (`5 == "five"`,
  every operator; see the next entry for text that is one), and `<`/`<=`/`>`/`>=` between values Python cannot order
  (`None > 3`, a list and a number), now fail the step with an error naming the node and pins: "Compare 'Is five'
  (node same) ==: b is text ('five'), not a number; a is a number (5)." Before, these answered False silently.
  Equality of same-type values and Python's equality across other types (`None == 5`) are unchanged; true/false
  compares with numbers as Python's booleans do (`True == 1`).
