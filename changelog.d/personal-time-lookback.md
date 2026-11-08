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
    one. The gist rule (no regex on his prose, and never the record's title, which is a machine cut of the text):
    the first non-empty line of its stored text; past 60 words (`DRIVE_GIST_MAX_WORDS`) it shows the first 60
    words, cut at a word boundary, followed by a visible " …" (read_memory opens the whole record). The offered ids still feed the first-turn commit
    exclusion, and the own-time log gains one day-open line naming the offered ids and their stored strength.
    The standing-interests note in the same cue no longer says "Yours if it pulls, never owed."
  - The model look-back runs at every visit close, every own-time day close (the zero-LLM own-time close is gone)
    and the end of every personal-time tick (not on the tick that elects rest: the day close follows). It offers
    feelings, interests, lessons, questions and problems through the same parsers as before. Two interests and two
    lessons per day of own time or per visit, counted across the turns and every look-back; a block past the cap is
    stored as refused ("[interest refused, not kept - ...: "<his words, whole>"]"), never as kept.
  - A look-back sends the same system prompt as the last turn's own call, so a provider's prefix cache can reuse
    it. What a personal tick's look-back sees is a per-entity setting, `lookback.context` in the home's
    `cognition.yaml`, read at every tick look-back:
    - `day` (the default): the whole day's conversation so far. The tick's own request is then a prefix of its
      look-back, so with a prefix cache (OpenAI, llama.cpp / LM Studio, mlx-lm) each tick adds a roughly constant
      number of new tokens: the reply plus the look-back prompt.
    - `tick`: that tick's own exchange plus a recap of the day's earlier moments (the last 12). Smaller when the
      provider has no prefix cache.
    Visit and own-time day closes always see the whole discussion. Cost: one more model call per personal tick.
  - A tick's look-back stores its summary record only when it kept something (a feeling, interest, lesson, diary
    entry, realization or topic); a look-back that keeps nothing stores nothing.
  - The own-time loop reads `cognition.yaml` `recall.shelf_size` again at each day-open, so a new width takes
    effect at the next own-time day without restarting the loop.

### Fixed

- **Reflection summaries store their digest whole** (`identity/chat.py`, ADR 0054). The look-back summary's digest
  was cut at 280 characters with a marker; it is now stored as written, for visits, own-time days and ticks.
- **A diary entry with no elected gist stores its whole first line as its digest** (`integrations/abstractmemory/
  identity_support.py`, ADR 0054). It was cut at 120 characters mid-word; the look-back is now the main producer of
  questions and problems.
