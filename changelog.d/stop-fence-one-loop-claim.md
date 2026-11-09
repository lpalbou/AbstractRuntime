### Fixed

- **A loop never claims a home another live loop holds; a hand start reserves like a spawn** (Fable review F7). The
  boot claim merged its `between` over whatever stood, so a reservation naming another live child, or another live
  loop's registration, was replaced (two loops over one home). The claim is one locked, fenced read-check-write now
  (`LoopAlreadyRunning`, writing nothing, when the record is another live process's - the pid alive and the same
  incarnation). The hand start (`python -m abstractruntime.identity.life` without `--generation`) used to check
  status outside any lock and never reserve: it reserves under the spawn lock now, naming itself as the child.
