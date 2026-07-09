# 027 — Quote provenance: quoted claims carry their source, structurally

**Status**: planned
**Origin**: maintainer ask (2026-07-07, round 9) + a live incident the same
night: the working agent attributed a phrase to another agent's message when
it was in fact the agent's own gloss — caught by the maintainer, corrected on
the channel. In a long context, QUOTES DRIFT FROM THEIR SOURCES.

## The failure class

When an agent (or entity) restates another party's words from deep context,
nothing binds the restatement to the original. The words survive; the
attribution decays. In the entity world this becomes a memory-integrity
issue: a formed record saying "X said Y" with no link to where Y actually
lives is unverifiable forever.

## The observed countermeasure (what the working agent does when careful)

Re-read the source file immediately before quoting; keep the quote and its
file/message id adjacent in the output. The discipline is cheap; forgetting
it once is what produced the incident.

## Proposed shape (investigate)

- **Formation-time rule**: records whose digest quotes/attributes another
  party carry a `derived_from` edge (or receipt) to the source record /
  ledger span / a2a message id. The diary receipts pattern already does
  this for ledger spans — generalize the convention to attributed claims.
- **Turn-loop rule (entity chat)**: when the working set injects a memory
  whose digest contains an attribution, the rendered form keeps the record
  id adjacent (the observer's "why did this enter context" pattern applied
  to "who actually said this").
- **a2a discipline (already partially codified)**: replies enumerate the
  ask numbers they answer; extend: verbatim quotes of another agent name
  the message id inline.

## Guardrail

Never machine-verify attribution by string matching alone (paraphrase is
legitimate); the rule is that the LINK must exist, not that the words must
be identical.
