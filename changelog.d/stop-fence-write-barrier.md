### Fixed

- **A diary or artifact write whose fence check passed always lands before a racing Stop, never after it** (Codex
  review 4 #2, the runtime's stores). The diary book and the artifact store checked the fence before their write and
  outside any lock the bump takes, so "check passes -> another process bumps -> the write lands" left a write after
  the paused row. The home now has a write barrier (`lifecycle_fence.write_barrier`, a cross-process flock on
  `<home>/.write_barrier.lock`): a store write is one `FenceHolder.writing(where)` section - the barrier held shared,
  the fence checked and the write made inside it - and `bump_lifecycle_generation` holds it exclusive around the
  generation write (inside the home state lock: the home lock outer, the barrier inner). `DiaryStore(guard=)` and
  `FileArtifactStore(guard=)` take the home's fence holder (a `.writing` section), or a host's callable guard
  together with `barrier_home=` (the home whose barrier the write holds); a bare callable is refused. **Breaking for
  the gateway's shared home**: its `DiaryStore(guard=self._commit_guard)` adds `barrier_home=self.home_dir`. Memory's own
  commit points join through memory's `write_barrier()` (its owner's half).
