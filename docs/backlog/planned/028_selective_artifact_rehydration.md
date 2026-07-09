# 028 — Selective artifact rehydration: fetch the relevant section, not the file

**Status**: planned
**Origin**: maintainer ask (2026-07-07, round 9). Reproduces the working
agent's observed technique for consuming large artifacts without paying
their full context cost.

## The observed mechanism

Faced with a 40-70KB report/transcript, the working agent does NOT read it
whole. It (a) greps/indexes first (find where the relevant content lives),
(b) extracts the single most relevant block (e.g. `jq` for the longest
message; `sed -n` for a line range; head/tail for boundaries), (c) reads
only that, and (d) keeps the artifact path so deeper sections stay one
fetch away. Channels-before-scan, tiered access, pointer retained — the
memory system's own doctrine, applied to file I/O.

## The gap in the framework

Payload tiers today are whole-object: `digest` (small) or `raw`
(everything). `MEMORY_REHYDRATE` and `DIARY_READ` return complete payloads.
There is no middle tier: "the section of this artifact that answers the
current need."

## Proposed shape (investigate)

- **Range/section tier on artifact reads**: `payload(ref, tier="raw",
  range=(start,end))` or an anchored section fetch (`section="§3"`,
  `around="<match>", window=N`) on the runtime artifact store — cheap,
  because artifacts are files.
- **Recall-integrated**: a handle's payload fetch takes the CURRENT cue and
  returns the best-matching slice of the raw payload (keyword-scan inside
  the artifact — the existing in-Python keyword channel, reused at the
  payload level) with the offsets, so the full text stays one explicit
  fetch away.
- **Honesty**: a slice is a VIEW, never a replacement — label
  `{slice_of: ref, range: ..., total_size: ...}` so consumers know they
  hold a window (the seq-gap honesty rule, applied to bytes).

## Evidence to gather

Token cost + answer quality on the keystone diary verbatims and the 0002
experiment artifacts: whole-payload vs cue-matched slice.
