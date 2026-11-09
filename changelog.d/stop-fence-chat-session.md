### Changed

- **`ChatSession.fence` replaces the host-set `cancel_check`** (STOP-LANE-DESIGN §2.2; Codex review 3 #1). The
  session takes its home's fence (`open_home(..., fence=)`) or the host's (`session.fence = <LifecycleFence>`; the
  own-time loop sets its own). **Breaking for hosts**: assigning `session.cancel_check` now raises `AttributeError`
  naming the fence, so a host that still sets it fails at once instead of running unfenced turns.
  `salvage_pending_lookback(..., fence=)` takes the door's fence (required).

### Fixed

- **No model call after a Stop, retries included, and a late reply lands nowhere** (Codex review 3 #1; design class
  A): `_generate` checks the fence before EVERY attempt (a harmony-400 retry after the Stop used to call the model
  again), maps a call that failed after the Stop to the Stop, and abandons a reply that returned after it (not
  counted in spend, nothing formed). A client that honours `cancel_event=` (abstractcore's providers) gets the
  fence's event: the Stop in the bumping process stops the decode within a token.
- **Every chat effect is fenced around its authoring** (design class B): `_effect` checks before the authoring call
  and again before the handler, and `_author_records` / `_author_diary_entry` abandon an authoring reply that
  returned after the Stop (no artifact kept, no write).
- **The CLI visit writes nothing after a Stop** (Codex review 3 #5): it is admitted at the home's generation; its
  auto-yield posture, its timeout restore and its close restore are fenced, and the restores check the posture and
  write in one locked step (the timeout restore used to write awake unconditionally); a Stop mid-visit ends it with
  one plain line, without a look-back.
- **The visit lane's durable diary capture after a Stop** (Codex review 3 #1): a fenced book refusal is the Stop - no
  rescue copy of the reply is written into the home - and the public entry's authoring call to the host's summaries
  handler never runs after it.
