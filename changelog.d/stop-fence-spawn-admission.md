### Changed

- **`spawn_loop_process(..., fence=)` takes the start's admission (required)** (Codex review 4 #7, P1). The spawn
  checked eligibility, cleared a stale STOP and fast-forwarded the command inbox, and only then admitted itself - a
  Stop in between became the spawn's own generation, so the reservation, the log and the child appeared after the
  Stop. The host passes the fence it decided the start at: the spawn refuses at once when it moved, clears the
  control files only inside `fence.commit`, and reserves at that admission. **Breaking for hosts**: the gateway's
  spawn calls must pass the fence (an unfenced call raises `TypeError`).
