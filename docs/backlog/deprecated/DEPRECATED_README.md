# Deprecated Backlog Items

This folder contains legacy and superseded backlog items preserved for historical reference.

## Why These Are Deprecated

These files represent earlier iterations of the backlog that have been:
- Superseded by the reorganized `completed/` and `planned/` structure
- Implemented and documented in the new structure
- Abandoned or merged into other items

## Contents

### From Original Backlog (docs/backlog/)
- `001_runtime_kernel.md` - Superseded by `completed/001_runtime_kernel.md`
- `002_persistence_and_ledger.md` - Superseded by `completed/002_persistence_and_ledger.md`
- `003_wait_resume_and_scheduler.md` - Split into `completed/003_wait_primitives.md` and `planned/004_scheduler_driver.md`
- `004_effect_handlers_and_integrations.md` - Superseded by `completed/005_abstractcore_integration.md`
- `005_examples_and_composition.md` - Superseded by `planned/010_examples_and_composition.md`
- `006_ai_fingerprint_and_provenance.md` - Split into `completed/007_provenance_hash_chain.md` and `planned/008_signatures_and_keys.md`

### From Old Backlog (docs/backlog-old/)
These were temporary implementation specs created in the abstractcore workspace before abstractruntime was fully accessible:
- `001_integrations_abstractcore.md` - Implemented
- `002_snapshots_bookmarks.md` - Implemented
- `003_provenance_ledger_chain.md` - Implemented
- `004_tests.md` - Implemented
- `005_docs_updates.md` - Implemented
- `README.md` - Original context document (preserved as-is)

## Note

Do not add new items here. Use `completed/` for finished work and `planned/` for future work.
