### Added

- **A public update names the identity its run started with** (`replay/view.py`, ADR 0052, framework backlog 1119).
  The replay view copies the run's `vars._runtime.identity_ref` (`{"of", "rev"}`, stamped by the gateway on visit,
  summon, meet and session runs) onto each of that run's `public_update` messages as `identity_ref`, so a host serves
  cached views with the ref already in place. A run without a ref, or with a malformed one, gives no field, never an
  invented one; a snapshot read after the updates were folded stamps them then; the ref is part of the snapshot
  digest, so the cursor token (an ETag) changes with it, and a run without a ref keeps the token it had.
