### Fixed

- **The fence fails closed on every signal failure** (Codex review 4 #9, P2). A home that was never stopped had no
  signal file, and "absent then absent" read as unchanged in every process, so another process's Stop whose first
  signal write failed was hidden; a failed removal of a stale signal hid it too; a corrupted phases.yaml left
  `moved()` False; and a failed truth read kept the cached generation standing. Now: admission (`LifecycleFence.admit`,
  inside the home state lock) establishes a valid signal - written when missing or behind, refusing admission when it
  is ahead of phases.yaml; a missing signal reads as moved; the check also observes phases.yaml's own signature (two
  stats; the file is parsed only when it changed), so a bump whose signal failed or a corruption is seen at the next
  check; and a failed truth read drops the cached generation.
