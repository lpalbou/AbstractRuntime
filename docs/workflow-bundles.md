# WorkflowBundles (`.flow`)

A **WorkflowBundle** is a portable distribution unit for VisualFlow JSON workflows:
- bundle format: zip file with `manifest.json`, `flows/*.json`, optional `assets/*`
- portability comes from shipping **VisualFlow JSON**, not `WorkflowSpec` (which contains Python callables)

Implementation pointers:
- manifest model: `src/abstractruntime/workflow_bundle/models.py`
- pack/unpack helpers: `src/abstractruntime/workflow_bundle/packer.py`, `src/abstractruntime/workflow_bundle/reader.py`
- on-disk registry: `src/abstractruntime/workflow_bundle/registry.py`
- compiler: `src/abstractruntime/visualflow_compiler/*`

## Bundle layout

Minimal bundle:

```
manifest.json
flows/<flow_id>.json
```

Optional:

```
assets/<name>
```

## Packing a bundle

```python
from abstractruntime.workflow_bundle import pack_workflow_bundle

pack_workflow_bundle(
    root_flow_json="flows/root.json",
    out_path="out/my_bundle.flow",
    bundle_id="my_bundle",
    bundle_version="0.1.0",
)
```

`pack_workflow_bundle(...)` is stdlib-only and validates that referenced subflows exist in `flows_dir` (defaults to the root file’s directory).

## Reading a bundle

```python
from abstractruntime.workflow_bundle import open_workflow_bundle

b = open_workflow_bundle("out/my_bundle.flow")
print(b.manifest.bundle_id, b.manifest.bundle_version)
print([ep.flow_id for ep in b.manifest.entrypoints])
```

## Registry (installed bundles)

`WorkflowBundleRegistry` is a host-side convenience layer for storing and resolving `.flow` bundles from a directory:
- default directory resolution: `default_workflow_bundles_dir()` (`src/abstractruntime/workflow_bundle/registry.py`)
- resolve `bundle_id[@version]` and entrypoints (`resolve_bundle`, `resolve_entrypoint`)

## See also

- `architecture.md` — VisualFlow → WorkflowSpec compilation path

