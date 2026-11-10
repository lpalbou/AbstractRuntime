### Added

- **The memory facade exports the topic verbs** a host serves and runs (COMMUNITIES-DESIGN §3; AbstractMemory
  3082695 / d550551): `communities_pass` and `embed_missing` (from `abstractmemory.communities`), and `communities` and
  `community_of` (from `abstractmemory.community_index`). They are memory's own functions, unchanged, from their
  modules: memory exports none of them on its package root, where `abstractmemory.communities` is the module. The
  gateway's `/communities` route serves `memory_facade.communities(system, scopes=, owner_id=)` (payload version 2).

### Changed

- The `topics` tool reads `abstractmemory.community_index.communities` (no longer the package root). The list is that
  one read: each topic's newest date comes from memory's `facts.span`, with no scan of the entity's digests. A
  drill-down reads the digests once, for the members' titles.
