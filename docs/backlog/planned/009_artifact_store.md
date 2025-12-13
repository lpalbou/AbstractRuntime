## 009_artifact_store (planned)

### Goal
Store large payloads (documents, images, tool outputs) by reference instead of embedding into `RunState.vars`.

### Rationale
Durable state must stay JSON-serializable and reasonably sized.

### MVP design
- `ArtifactStore` interface
- file-based implementation
- store `artifact_id` references inside `RunState.vars`

### Future
- S3 / object storage backend
- content-addressed storage

