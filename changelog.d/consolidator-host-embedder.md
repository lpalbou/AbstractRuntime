### Added

- **`build_consolidator(..., embedder=)`: the night uses the host's own embedder.** A host whose embedder is its own
  resolved AbstractCore client (the gateway's, possibly in its model-host child) - not a URL - hands it to the night,
  which passes it unchanged to `open_home` and builds nothing from `embedding_model` / `embedding_base_url`. Without it
  the night's repair refused every vectored record (`embedder_unavailable`) and the missing-embedding step embedded
  nothing.
