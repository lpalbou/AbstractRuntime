### Fixed

- **A visual Agent node's agent can stay out of a message addressed to entities** (abstractgateway
  c26d38a2, AbstractAgent ff1c4d1). The gateway writes `_runtime.mentions`
  (`{entities: [{entity_id, name}], line}`) on the run when the person's message mentions entities,
  but an Agent node runs its loop in a child run whose `_runtime` is built fresh, so the child never
  saw it and asked again after an empty reply instead of ending the turn silently. The Agent node now
  carries `mentions` into its child (a copy; an explicit child value wins). The gateway's context line
  already reaches the child's messages the way the conversation does (the node's `context` pin, or
  `use_context`).
