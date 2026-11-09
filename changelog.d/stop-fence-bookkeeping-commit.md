### Fixed

- **The loop's bookkeeping and the chat lane's pending look-back marker never land after a racing Stop** (Fable
  review F8). The loop's spend, personal-time usage, task status (taken / done / parked) and work-order archive, and
  `ChatSession`'s write-ahead marker (`pending_reflection.json`), checked the fence and then wrote outside any lock:
  a Stop in that window landed a file after the paused row. Each is one `fence.commit` step now (the home state lock
  and the generation re-read there): a Stop arriving meanwhile waits for the write, and a write decided before a
  Stop never lands after it (about 0.4 ms once per tick or day close, never per effect).
