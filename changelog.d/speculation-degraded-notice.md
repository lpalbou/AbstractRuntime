### Added

- **The ledger says when a call ran without the speculation it asked for** (`integrations/abstractcore/
  effect_handlers.py`; round 4, operator 2026-11-08: "fallback to MTP=off if MTP was selected by mistake"). When a
  provider response's `metadata.speculation.applied` is `false` (AbstractCore degraded a default-sourced
  `native_mtp` request for a provider without a native MTP adapter and generated without it; core main
  44897db3…904272da), the LLM_CALL result carries one notice worded from that metadata — `speculation requested
  (<mode>, N draft tokens) was not applied: <message> (reason: <reason>)` — so the replay view can show why. No retry
  in the runtime; a response without the key adds nothing; an exception still propagates as before. Six Assistant
  runs on 2026-11-08 failed instantly on a stored native_mtp default applied to a remote endpoint model; Core's
  degrade plus this notice is the fix. Needs the AbstractCore that carries the metadata contract (unpublished at
  the time of writing; the runtime floor is raised when it publishes).
