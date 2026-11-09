### Fixed

- **Every failed authoring attempt of the drainer leaves its failure line** (`identity/authoring.py`; Codex review 3
  #10). Only a malformed reply was recorded; an unreadable task packet (`packet_unreadable`), a record with no words
  to author from (`no_source`) and an unreachable capability (`unavailable`, deferred without spending an attempt) now
  write one line too, each with the record id, the `outcome` (retry / defer), the original exception's class and
  message and the capability model when one was built.
