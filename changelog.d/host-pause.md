### Added

- **An own-time loop freezes while its host is paused** (`identity.life.host_pause`). A host that pauses (the
  gateway's kill switch) names its pause file in `ABSTRACTRUNTIME_HOST_PAUSE_FILE` for the processes it starts.
  The loop reads it at every boundary and treats a paused host as the entity's own `paused` state: no summon, no
  tick, no sleep window while it stands, and the same day continues when it lifts. The entity's state file is not
  written for it.
