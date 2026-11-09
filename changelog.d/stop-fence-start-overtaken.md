### Fixed

- **A start the operator's Stop overtook is never reported started** (Fable review F6). A Stop landing between the
  spawn reservation and `Popen`'s return, with the child still alive at the 0.6 s check, let `spawn_loop_process`
  return `{pid, generation}` while `loop_status` already said `stopped / kill_switch`. It now raises
  `StoppedByOperator` (the child refuses itself at its boot, or the Stop kills it).
