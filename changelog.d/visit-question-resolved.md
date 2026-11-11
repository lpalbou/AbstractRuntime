### Fixed

- **A question the visit answered is marked resolved** (`identity/visit_workflow.py`, `identity/reflection.py`,
  `identity/act_only.py`, `identity/diary.py`, `identity/chat.py`; root backlog 1206). Operator 2026-11-11 02:00: an
  entity asked him a question in a visit, he answered, nothing marked it resolved, no badge appeared, and the
  question stayed under OPEN QUESTIONS ("it's important as if unresolved, this is probably looked back into its
  memory"; "investigate and fix"). Memory already resolves a question when a later diary entry carries
  `resolves=<its entry id>` and the book confirms the claim (`resolves_status`), but the visit lane never asked for
  it. Now:
  - the visit's look-back lists the entity's open questions that this visit asked or touched (a question it kept
    during the visit, or one a turn's recall showed it), each by entry id and title as formed, read from memory's
    own open-question set (`abstractmemory.diary.open_questions`, scope `diary`, once per close), and offers a
    ```` ```resolve ```` block: `resolves=<the question's id>`, `evidence=<the numbered moment that answered it>`,
    then the answer in the entity's words. The block is captured at the result boundary like a ```` ```diary ````
    block and written through the same `DIARY_WRITE` as a note carrying `resolves=`; its `evidence=` numbers link
    the entry to those moments. A block naming no question or giving no answer is refused, said, with the words
    kept in its marker. The model declares the resolution, the book confirms it; nothing reads the entity's words.
  - the look-back names a resolution a turn already made (the resolution-to-lesson cue the chat lane had), and the
    look-back's own capture warnings (a refused claim) join the close's `reflection_notices`.
  - the act-only capture's metadata of an entry that claimed `resolves=` carries `resolves` and `resolves_status`;
    when the book confirms an open question or problem, the `DIARY_WRITE` result and that metadata also carry
    `resolved_record_id` (the question's graph id) and `resolved_title` (its title as formed; `""` and
    `resolved_visibility: "private"` for a private question).
  - a visit turn whose diary entry resolves an open question serves a second notice beside the entry's own:
    `{"kind": "question_resolved", "record_id": <the question's graph id>, "title": <its title>}`
    (`chat.question_resolved_notice`, which a host applies to the look-back's captured `diary_entries` for the
    close). The chat lane's prompt and notices are unchanged.
