### Fixed

- **A readmission serves new work only; a write still in flight from before the Stop stays refused** (Codex review 4
  #5, P1). `readmit` swapped the one fence every guard read, so a memory write of an effect admitted before the Stop,
  suspended inside its handler, passed its commit guard at the new generation once Resume had readmitted the
  runtime. An operation now keeps the admission it started under: the fenced handler table fixes each effect's
  admission for its whole duration (`lifecycle_fence.admitted`, per thread and home; a child run inside a
  subworkflow inherits it), a `ChatSession` runs its entry points under its own fence, and the holder's checks - the
  memory commit guard, the diary and artifact guards, the authoring callbacks - resolve to it. A host that stamps
  `lifecycle_fence.RUN_GENERATION_VAR` into a run's vars keeps every effect of that run judged against the
  generation it was admitted at.
- **A cached runtime authors public diary entries again after Resume** (Codex review 4 #4, P1): the visit lane's
  diary authoring checked the fence the runtime was OPENED with, so after Stop -> Resume -> readmit every new turn
  carrying a public diary entry was cancelled. It checks the operation's admission now.
