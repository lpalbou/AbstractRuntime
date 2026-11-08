### Changed

- **Interest, lesson, realization and diary titles are explicit and whole** (operator 2026-11-08: "the title of any
  lesson, question, interest, world model must be explicit and easy to read, for a human and for an agent"; root
  ADR 0054). The look-back offers and the entity contract now ask for a first line `title: <a short, complete
  name>` in each ```interest, ```lesson and (optionally) ```realize block, and a `gist:` line for a question. The
  title is that line, verbatim; without one it is the body's first complete sentence (`digest.split_sentences`),
  whole. Before, these titles were the body's first 8 words, and diary titles a 60-character slug ending in "…".
  The record's digest is the words without the title line. New helper `identity.reflection.elected_title_and_words`;
  `RealizeElection.title`. Already-stored titles are not rewritten.
