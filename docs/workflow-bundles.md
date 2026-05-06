# WorkflowBundles (`.flow`)

A **WorkflowBundle** is a portable distribution unit for VisualFlow JSON workflows:
- bundle format: zip file with `manifest.json`, `flows/*.json`, optional `assets/*`
- portability comes from shipping **VisualFlow JSON**, not `WorkflowSpec` (which contains Python callables)

Implementation pointers:
- manifest model: `src/abstractruntime/workflow_bundle/models.py`
- pack/unpack helpers: `src/abstractruntime/workflow_bundle/packer.py`, `src/abstractruntime/workflow_bundle/reader.py`
- on-disk registry: `src/abstractruntime/workflow_bundle/registry.py`
- compiler: `src/abstractruntime/visualflow_compiler/*`

The compiler also handles current VisualFlow authoring conveniences such as multi-entry execution fan-in. When a node has multiple incoming `exec-in` routes and per-route input overrides, the compiler lowers them into internal `join_exec` and `path_mux` nodes so the bundle remains portable and the runtime behavior stays explicit.

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

## VisualFlow multi-entry fan-in

Visual authoring tools may connect more than one execution edge into the same target `exec-in` pin. For example, a first prompt can enter a node from `on_flow_start`, while a later loop can re-enter the same node with a different prompt produced by the previous turn.

Store two metadata fields on the target node:
- `entryRoutes`: ordered execution entries. Each route has a stable `key`, `sourceNodeId`, and `sourceHandle`.
- `inputRouteOverrides`: per-input route overrides. Shape: `pinId -> routeKey -> {sourceNodeId, sourceHandle}`.

Minimal target-node fragment:

```json
{
  "id": "ask",
  "type": "ask_user",
  "data": {
    "pinDefaults": {"prompt": "start"},
    "entryRoutes": [
      {"key": "start::exec-out", "sourceNodeId": "start", "sourceHandle": "exec-out"},
      {"key": "ask::exec-out", "sourceNodeId": "ask", "sourceHandle": "exec-out"}
    ],
    "inputRouteOverrides": {
      "prompt": {
        "ask::exec-out": {"sourceNodeId": "ask", "sourceHandle": "response"}
      }
    }
  }
}
```

Compiler behavior:
- incoming exec edges are rerouted through an internal `join_exec` node
- overridden pins are routed through internal `path_mux` nodes
- the selected route is persisted in run state, so pause/resume and file-store restarts keep the same input selection
- stale metadata is rejected when `entryRoutes` no longer matches the incoming exec edges

Authoring guidance:
- use the default route key `${sourceNodeId}::${sourceHandle}` unless your editor needs a custom stable key
- keep route keys unique per target node
- use one normal data edge or `pinDefaults` for the fallback value, then add `inputRouteOverrides` only for routes that need a different value

## See also

- `architecture.md` — VisualFlow → WorkflowSpec compilation path
