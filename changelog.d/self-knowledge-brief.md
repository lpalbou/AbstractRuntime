### Added

- **The entity's self-knowledge brief: what it knows about, at a glance** (`identity/self_brief.py`,
  `life.build_consolidator`, `chat.compose_system_base`). An entity cannot search for what it does not know it has, so
  at the end of every sleep's topics stage the night makes one call to the summaries capability over the entity's
  topics (title, summary, size, newest memory, recalls of the last 7 days, largest first), its trails of thought
  (AbstractMemory's `facts.paths`: memories of one topic recalled together, written `A → B (n times)`) and what the
  pass changed, with the briefing that names the entity. The reply is stored in `<home>/self_brief.md` under a header
  (`version_seq`, `written_at`, `tokens`), never in the memory graph. Hard cap 400 tokens by the runtime's budget
  estimator (`digest.token_estimate`); a longer reply is refused, never cut. A sleep whose topics did not change keeps
  the brief without a model call. A failure keeps the previous brief and writes one authoring-failure line
  (`template_id` `runtime.self_brief`). Fenced: no call after the operator's Stop, a reply returning after it is
  abandoned, the write is a fenced write section. The night's result carries `self_brief`; `own_time.log` gets one
  line per sleep.
- **The brief sits in every lane's system prefix** (`compose_system_base(..., self_brief=)`, read at summon by
  `self_brief_for_prompt`): `ChatSession` (hosted chat, own time and work through the life factory's re-compose) and
  the durable visit's OPEN node place the section "What I know about, at a glance (updated in my sleep):" right after
  the identity block, before everything that changes per turn, so the provider's cached prefix stays byte-identical
  across turns and changes at most once per sleep. A home without a brief gets no section. The `topics` tool's
  teaching (descriptor and tools paragraph) names the brief: "Your at-a-glance brief lists your topics; use topics to
  dig in."

### Changed

- `authoring.entity_naming_clause(entity_name)`: the naming clause of the summaries author's briefing, shared by
  `authoring_system_prompt` (unchanged output) and the brief's `self_brief_system_prompt`.
