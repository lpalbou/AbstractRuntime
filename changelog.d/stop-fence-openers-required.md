### Changed

- **`open_home(..., fence=)` and `ChatSession(..., fence=)` take the fence by signature** (Fable review F4, I6):
  keyword-only, no default. An unfenced home or session was a silent path (the gateway's hosted chat is exactly that
  path); a reader that acts on nothing now writes `fence=None` itself, explicit and grep-able, and a ChatSession over
  a fenced home refuses `fence=None`. Every runtime caller passes its fence (the loop's day factory, the CLI visit,
  the salvage door) and the static scan covers `ChatSession` too. **Breaking for hosts**: the gateway's `open_home`
  and `ChatSession` constructions must pass the fence.
