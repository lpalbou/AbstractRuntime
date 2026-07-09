# 026 — Context checkpoint: agent-elected self-compression with source pointers

**Status**: planned
**Origin**: maintainer ask (2026-07-07, identity round 9): "if you know of
mechanisms to better handle memory (you seem to do it quite well) and it
doesn't yet exist in the abstractframework, create backlog items with the
necessary details." This item reproduces the working agent's own observed
long-session technique.

## The observed mechanism (what the working agent actually does)

Across a session of millions of tokens, the agent periodically **writes its
load-bearing state OUT of the live context into durable files** (workspace
`AGENTS.md` dated notes; extracted reports under `/tmp`; a2a thread messages)
and afterwards **trusts the written digest over its own recollection** —
re-reading the file when the state matters rather than asserting from deep
context. The write is *elected* (the agent decides "this is load-bearing"),
*compressed* (a dated bullet, not a transcript), and *pointer-preserving*
(the note names the files/ids where the full truth lives).

This differs from the shipped `MEMORY_COMPACT` in three ways: it is
**agent-elected** (not budget-triggered), **pointer-preserving by contract**
(every compressed claim carries where to verify it), and the digest is
**authored by the agent in its own words** (not a mechanical summarizer).

## Proposed shape (investigate, not prescribe)

- A `CONTEXT_CHECKPOINT` composition (host-side; likely zero new effects):
  the agent (or entity) elects to write a "state digest" — current goals,
  open obligations, decisions made, WHERE the full record lives — via the
  existing `DIARY_WRITE` (kind=`note` or a new `checkpoint` diary_type) or
  `MEMORY_FORM` with `derived_from`/receipt pointers.
- On context pressure (or session resume), the digest is the FIRST thing
  rehydrated — cheaper and more faithful than replaying raw history; raw
  stays reachable through the pointers (payload tiers / DIARY_READ).
- Relation to the entity turn loop: a long conversation periodically
  checkpoints instead of letting the working set fight raw history for
  context.

## Guardrails (from the lossless doctrine)

- A checkpoint NEVER replaces the record — it is a reading aid over an
  intact archive (ADR-0026: no silent truncation; pointers keep the full
  truth one fetch away).
- Checkpoint cadence must be bounded (the reflection-guard family: ≤N per
  session) and elected, never forced per-turn.

## Evidence to gather

- Measure on the keystone-style harness: context cost of digest-first
  resume vs raw-history resume at equal answer quality.
