### Fixed

- **A tick look-back's summary record has an authored title, never "tick reflection: <session> #<n>"**
  (`identity/chat.py`; Codex review 3 #8). The tick look-back passed that template as the title and only the summary
  was authored. Both are authored now in the formation step; the tick's label is kept as structure
  (`attributes.lookback_label = "<session> #<n>"`).
