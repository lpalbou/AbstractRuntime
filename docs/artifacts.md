# Runtime artifacts

AbstractRuntime stores large payloads as artifacts so run state, ledger records,
and workflow outputs remain JSON-safe. Artifacts are the durable record for
files, generated media, tool evidence, exported history bundles, and other
payloads that should be referenced by id rather than embedded inline.

The implementation lives in `src/abstractruntime/storage/artifacts.py`.

## File-like vocabulary boundary

Runtime artifacts are not the same thing as live filesystem paths:

- `Artifact`: a Runtime-owned durable payload safe to persist, search, reuse,
  and pass between runs by reference.
- `Workspace File` / `Workspace Folder`: a server-side path capability under
  Gateway/runtime workspace policy. These are path values, not durable payloads.
- `Local File` / `Local Folder`: a client-side intake source. In hosted/browser
  mode they should be uploaded and normalized into artifacts before durable
  execution.

Gateway/Flow product copy may say `Server File` / `Server Folder` when users
choose an origin, but Runtime stays anchored on artifact refs versus
workspace-scoped paths.

## Artifact identity

An artifact has a stable `artifact_id`, a `blob_id` for content deduplication
when the backend supports it, and a `run_id` scope. JSON state and Gateway APIs
pass artifacts with refs such as:

```json
{
  "$artifact": "a7050ebc5c8330...",
  "artifact_id": "a7050ebc5c8330...",
  "run_id": "9e19bd6a-ba07-4c2e-86c6-94ec7ca0a373",
  "content_type": "audio/wav",
  "size_bytes": 5293012
}
```

The ref is a pointer, not authorization. Hosts such as AbstractGateway are
responsible for deciding which principal may list or read an artifact.

## Stored metadata

Runtime stores two metadata layers:

- `tags`: string fields for compatibility and simple lookups.
- `metadata`: structured JSON for producer-specific details.
- `descriptor`: a Runtime-owned `ArtifactDescriptor` that normalizes the fields
  Gateway and Observer should use.

The descriptor separates display format from semantic meaning:

- `render_kind`: how the artifact should be rendered, such as `image`, `audio`,
  `video`, `markdown`, `html`, `json`, `text`, or `document`.
- `semantic_kind`: what the artifact represents, such as `voice`, `music`,
  `sound`, `transcript`, `evidence`, `workflow_snapshot`, or `image`.
- `classification_source`: whether the classification came from the producer,
  runtime tags, MIME inference, or legacy fallback.

The descriptor can also carry `session_id`, `workflow_id`, `node_id`, `turn_id`,
`ledger_cursor`, `producer`, `generation`, `media`, `source_refs`, `links`,
`security`, and action metadata. Producer metadata may be sparse; consumers
should show missing fields as unavailable rather than guessing.

## Generated media provenance

When generated media is produced through the Runtime AbstractCore integration
with output selectors, Runtime stores descriptor and metadata alongside the
bytes. Current generated outputs include image, video, voice/TTS, music, sound
or audio outputs, and transcription-style text outputs where the host route
stores a transcript artifact.

Generated-media descriptors record available producer facts:

- package/capability route, provider, model, backend, and runtime provider/model
  when they differ;
- prompt or TTS text, requested format, output index, negative prompt, and
  redacted generation parameters;
- source artifact refs for edit, image-to-video, cloned/reference voice, or
  other source-media flows when provided;
- measured media facts such as duration, sample rate, dimensions, channels, or
  frame counts when the store can inspect the bytes.

Runtime redacts obvious secret fields and bounds large metadata values. It does
not store raw provider requests as indexed descriptor fields.

Gateway or package producers that store artifacts outside the main Runtime
generated-media path should use `build_artifact_descriptor_payload(...)` from
`abstractruntime.storage.artifacts`. That helper applies the Runtime descriptor
schema, bounded secret-key redaction, and prompt/text sensitivity labels without
making Gateway invent a parallel descriptor contract.

## Catalog and search

`InMemoryArtifactStore` and `FileArtifactStore` support:

- `store(...)`, `load(...)`, and `get_metadata(...)`;
- `update_metadata(...)` for descriptor or structured metadata enrichment;
- `search(...)` for bounded pages;
- `count(...)`, `facet_counts(...)`, and `stats(...)` for exact totals and
  filter chips;
- `record_access(...)` for explicit metadata/content/preview/download/export
  counters.

`FileArtifactStore` maintains a repairable SQLite catalog for descriptor fields,
time filters, exact counts, byte totals, and facets. The catalog is an index of
Runtime-owned artifact metadata; the payload bytes and metadata files remain the
source of truth.

Plain `load(...)` and `get_metadata(...)` are side-effect free. UI and HTTP
layers that want access statistics must call `record_access(...)` or use Gateway
content routes that label the action.

## Gateway and Observer

AbstractGateway exposes Runtime artifacts through bounded HTTP APIs. Search
responses include `artifact_envelope_v1`, which projects Runtime descriptors,
media facts, access stats, and action links while preserving legacy row fields.
Gateway only forwards descriptor action links that are relative Gateway/UI
links; arbitrary external provider URLs should be represented as trace
availability or Gateway-owned trace records instead.

AbstractObserver renders Gateway envelopes. It should not infer canonical
artifact meaning from filenames or raw content except as a visible legacy
fallback. Use the Runtime tab for artifact inventory and the Observe tab for the
workflow/ledger narrative.

## Retrieval boundaries

Artifact search answers questions about stored files and media by metadata,
scope, type, time, producer, and links back to runs. It is not a semantic memory
search system.

- Use the ledger for "what happened in this workflow?"
- Use artifact search for "what files/media exist and how were they produced?"
- Use AbstractMemory/KG retrieval for "what knowledge or relationships were
  learned?"
- Use Gateway audit/provider links for system-level request traces when the
  envelope reports them.

## Limits

Legacy artifacts may only have MIME type and tags. Runtime projects them with
fallback descriptor fields so clients can still list and preview them, but
producer-level prompt/model/source provenance is only available for artifacts
created through descriptor-aware paths.
