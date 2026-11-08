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
  - The desk ("Alive on your desk today") shows each offered drive's gist, the whole first sentence of its stored
    text, and its next step when the record carries one. The offered ids still feed the first-turn commit
    exclusion, and the own-time log gains one day-open line naming the offered ids and their stored strength.
  - The model look-back runs at every visit close, every own-time day close (the zero-LLM own-time close is gone)
    and the end of every personal-time tick (not on the tick that elects rest: the day close follows). It offers
    feelings, interests, lessons, questions and problems through the same parsers as before. Cost: one more model
    call per personal tick that formed records, and one at each own-time day close.
  - The own-time loop reads `cognition.yaml` `recall.shelf_size` again at each day-open, so a new width takes
    effect at the next own-time day without restarting the loop.
