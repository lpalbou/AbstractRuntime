### Changed

- **He wakes when his sleep's maintenance is done; a visit interrupts his sleep and hands it back**
  (`identity/life.py`; contract-phases v2 §4.4, §10.6, §10.8; spec sleep->X#maintenance_done). His own sleep no
  longer naps `rest_minutes` after the pass: he wakes as soon as the pass returns (a stop that arrived during it ends
  the loop right then). No deadline is stamped on his sleep and the night's `should_continue` reads no clock: a pass
  ends at its next stage only for a visitor, a wake, a pause or the STOP brake. A visitor who knocks during the pass
  takes the posture (the pass yields at its next stage, and the night voice, a model call, is skipped); when the
  door's close restores the interrupted sleep (`written_by=self`), the loop resumes its maintenance pass - the SAME
  sleep, counted once, no sleep->sleep - and wakes him into what the toggles allow.
