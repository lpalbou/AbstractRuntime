### Added

- **The entity's `topics` tool** (tier 1, read-only, body optional; operator 2026-11-10: topics help the entity "to
  know what it knows, and let it delve deeper into subparts of its mind/memory"). It reads the records AbstractMemory
  serves (`communities()`, payload version 2), never a cache of its own (`identity/topics.topics_text`, through
  `HomeMemoryReader.topics`).
  - Empty body: the list, with each topic's title, one-line summary, size and newest memory, by size.
  - A topic's `#tag`: that topic's memories by title, newest first; memories placed since the last sleep are
    marked.
  - A memory's `#tag`: the topic that memory belongs to.
  - `#tag words`: that topic's memories ranked by meaning.
  - Words: the topic closest to them by meaning, its closest memories first.

  Meaning ranks through the store's vector query (memory's VectorMatrix), one embedding per call. Without an
  embedder the tool says so and lists newest first. A topic's tag is the tail of its stable key, so it still names
  the same topic after a sleep. Every memory shown carries its `#tag`, registered for `read_memory`. An untitled topic
  shows as "(untitled yet - N memories)", never a title cut from its members.
- The tool is taught in the tools contract paragraph, declared natively, and wired on the chat lane (ChatSession)
  and the effect lane (`ENTITY_TOOLS_EXECUTE`). A lane that does not wire it says so (`#FALLBACK topics elected but
  no resolver wired`). A memory without `communities()`, or a payload of another version, fails the call loudly.

### Fixed

- `search_memory` and `read_memory` never serve a topic as if it were a memory. `HomeMemoryReader`'s
  direct abstracts reads (`digest_assertions_all`, `find_tag_in_home`) skip memory's `community` records; memory keeps
  them off recall, probes, vectors and FTS itself (AbstractMemory aac9476).
