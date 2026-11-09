### Changed

- **The loop heartbeat checks the fence, not phases.yaml** (Fable review F10c). `touch_loop_heartbeat` runs before
  and after every model call of the loop; it took the home lock and parsed phases.yaml each time (~0.7 ms per call).
  It checks the loop's fence under the same lock now (one stat): no yaml parse, no subprocess; the lock (under which
  the Stop writes `stopped`) and the durable, fsynced status write stay.
- The Stop-fence tests compare a home's SQLite files by the hash of their logical dump, not by row counts (an UPDATE
  after the Stop was invisible to them), and the formation-authoring test records the write through the production
  handler table instead of re-wrapping it (Fable review F10a/b).
