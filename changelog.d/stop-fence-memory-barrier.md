### Fixed

- **A memory commit whose guard passed lands before the Stop's bump, never after it** (Codex review 4 #2, memory's
  half; AbstractMemory 85ecf75). Memory's commit guard runs under its database locks and the bump under the home
  state lock, so "guard passes -> another process bumps -> COMMIT lands" was possible. `bump_lifecycle_generation`
  now holds, in this order, the home state lock -> memory's write barrier (`MemorySystem.write_barrier()` over a
  handle opened for the bump: BEGIN IMMEDIATE on the journal, graph and task-queue files) -> the runtime's own write
  barrier (exclusive) -> the generation write, released in reverse. The barrier waits at most
  `MEMORY_BARRIER_TIMEOUT_S` (10 s) for a stuck writer and then bumps anyway: the Stop never hangs, and its receipt
  says `barrier_note: "barrier timed out: <the held lock>"` (`bump_lifecycle_generation(receipt=)`,
  `hard_stop_loop`'s result). Accepted residue, documented: a LanceDB store takes no lock (one write per concurrent
  LanceDB writer whose guard passed may land after the bump; its next write is refused), and memory's two SQLite
  autocommit bookkeeping writes (`_write_pin` without a cursor, the off-file `mark_journal_applied`) stay outside a
  barrier taken through another handle (memory 0.5.1 items, not records). The runtime's AbstractMemory floor is
  85ecf75 (the version is set at the memory cut).
