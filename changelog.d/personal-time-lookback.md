### Changed

- **Personal time says what it is for, and every look-back offers interests, lessons, questions and problems
  again** (`identity/life.py`, `identity/chat.py`, `identity/reflection.py`; operator 2026-11-08).
  - The own-time contract now reads "This time is yours to explore the tensions, the questions and the centres of
    interest that are alive in you, with the tools you have: a time to explore yourself and discover more about
    yourself and the world around you." It no longer says "there is nothing you are supposed to do with it, and no
    way to spend it wrong"; rest is "honoured when you are genuinely done" (no longer "always a valid choice").
    "Nothing owed" left the desk line and the wake cues.
  - The cue after a rest no longer repeats the rest reason ("you rested and your own time resumes."); the reason
    stays in the records (the sleep state, `own_time.log`, the tick record).
  - The desk ("Alive on your desk today") shows each offered drive's gist and its next step when the record carries
    one. The gist rule (no regex on his prose): the record's stored title; else the first non-empty line of its
    stored text; past 60 words (`DRIVE_GIST_MAX_WORDS`) it shows the first 60 words, cut at a word boundary, followed
    by a visible " …" (read_memory opens the whole record). The offered ids still feed the first-turn commit
    exclusion, and the own-time log gains one day-open line naming the offered ids and their stored strength.
    The standing-interests note in the same cue no longer says "Yours if it pulls, never owed."
  - The model look-back runs at every visit close, every own-time day close (the zero-LLM own-time close is gone)
    and the end of every personal-time tick (not on the tick that elects rest: the day close follows). It offers
    feelings, interests, lessons, questions and problems through the same parsers as before. Two interests and two
    lessons per day of own time or per visit, counted across the turns and every look-back.
  - A look-back sends the same system prompt as the last turn's own call, so a provider's prefix cache can reuse
    it. A personal tick's look-back sees that tick's own exchange plus a recap of the day's earlier moments (the
    last 12); `TICK_LOOKBACK_CONTEXT = "day"` in `identity/chat.py` gives it the whole day instead. Visit and
    own-time day closes always see the whole discussion. Cost: one more model call per personal tick, about 2-4k
    new tokens beyond the tick's own system prompt.
  - A tick's look-back stores its summary record only when it kept something (a feeling, interest, lesson, diary
    entry, realization or topic); a look-back that keeps nothing stores nothing.
  - The own-time loop reads `cognition.yaml` `recall.shelf_size` again at each day-open, so a new width takes
    effect at the next own-time day without restarting the loop.

### Fixed

- **Reflection summaries store their digest whole** (`identity/chat.py`, ADR 0054). The look-back summary's digest
  was cut at 280 characters with a marker; it is now stored as written, for visits, own-time days and ticks.
