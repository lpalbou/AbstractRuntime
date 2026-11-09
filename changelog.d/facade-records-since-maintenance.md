### Added

- **`memory_facade.records_since_maintenance`** (`integrations/abstractmemory/memory_facade.py`, ADR 0011). The
  memory facade re-exports AbstractMemory's `records_since_maintenance` unchanged (the same object), so the gateway's
  idle-sleep sweep reads an entity's records since its last completed sleep through the runtime seam instead of
  importing AbstractMemory directly. Needs an AbstractMemory that has `sleep_cadence.records_since_maintenance`.
