### Changed

- **Compare refuses a number against text instead of answering False** (`visual/builtins.py` `control_compare`,
  ADR 0049). A number or true/false on one side and text on the other (`5 == "5"`, every operator), or `<`/`<=`/`>`/`>=`
  between values Python cannot order (`None > 3`, a list and a number), now fails the step with an error naming the
  node and pins: "One side is a number and the other is text. Open the flow in the editor and save it to store typed
  values." Before, these answered False silently. Equality of same-type values and Python's equality across other types
  (`None == 5`) are unchanged; true/false compares with numbers as Python's booleans do (`True == 1`). Flows saved by
  AbstractFlow 0.13.0 with a typed value stored as text next to a wired number hit this error until they are opened
  and saved in AbstractFlow, which stores the typed value.
