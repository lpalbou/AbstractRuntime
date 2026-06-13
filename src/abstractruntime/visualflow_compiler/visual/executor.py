"""VisualFlow lowering utilities (VisualFlow JSON → Flow IR).

This is a stdlib-only subset of AbstractFlow's visual executor, extracted into
AbstractRuntime so the VisualFlow compiler can run without importing AbstractFlow.

Scope:
- VisualFlow → Flow lowering (no runtime wiring / no web/editor host concerns)
"""

from __future__ import annotations

import json
import keyword
import re
import textwrap
from typing import Any, Dict, List, Optional

from ..flow import Flow

from .agent_ids import visual_react_workflow_id
from .builtins import get_builtin_handler
from .code_executor import create_code_handler, ensure_code_permissions_allowed, normalize_code_permissions
from .execution_metrics import capture_execution_start, finish_execution_metrics
from .models import NodeType, VisualEdge, VisualFlow


# Type alias for data edge mapping
# Maps target_node_id -> { target_pin -> (source_node_id, source_pin) }
DataEdgeMap = Dict[str, Dict[str, tuple[str, str]]]


def _build_data_edge_map(edges: List[VisualEdge]) -> DataEdgeMap:
    """Build a mapping of data edges for input resolution."""
    data_edges: DataEdgeMap = {}

    for edge in edges:
        # Skip execution edges
        if edge.sourceHandle == "exec-out" or edge.targetHandle == "exec-in":
            continue

        if edge.target not in data_edges:
            data_edges[edge.target] = {}

        target_handle = str(edge.targetHandle or "")
        if target_handle in data_edges[edge.target]:
            prev_source, prev_handle = data_edges[edge.target][target_handle]
            raise ValueError(
                "Multiple data edges target the same input pin. "
                f"Target '{edge.target}.{target_handle}' is already wired from "
                f"'{prev_source}.{prev_handle}' and cannot also be wired from "
                f"'{edge.source}.{edge.sourceHandle}'. Use multi-entry route overrides "
                "for per-execution-path values instead of wiring duplicate data inputs."
            )

        data_edges[edge.target][target_handle] = (edge.source, edge.sourceHandle)

    return data_edges


def visual_to_flow(visual: VisualFlow) -> Flow:
    """Convert a visual flow definition to an AbstractFlow `Flow`."""
    import datetime

    # Authoring-time lowering:
    # - Multi-entry exec fan-in + per-entry input overrides are lowered into
    #   internal join_exec/path_mux junction nodes.
    #
    # This keeps the persisted authoring graph clean while keeping runtime
    # semantics centralized in AbstractRuntime (framework ADR-0012).
    from .multi_entry_lowering import lower_authoring_multi_entry

    visual = lower_authoring_multi_entry(visual)

    flow = Flow(visual.id)

    data_edge_map = _build_data_edge_map(visual.edges)

    # Store node outputs during execution (visual data-edge evaluation cache)
    flow._node_outputs = {}  # type: ignore[attr-defined]
    flow._data_edge_map = data_edge_map  # type: ignore[attr-defined]
    flow._pure_node_ids = set()  # type: ignore[attr-defined]
    flow._volatile_pure_node_ids = set()  # type: ignore[attr-defined]
    # Snapshot of "static" node outputs (literals, schemas, etc.). This is used to
    # reset the in-memory cache when the same compiled VisualFlow is executed by
    # multiple runs (e.g. recursive/mutual subflows). See compiler._sync_effect_results_to_node_outputs.
    flow._static_node_outputs = {}  # type: ignore[attr-defined]
    flow._active_run_id = None  # type: ignore[attr-defined]

    def _normalize_pin_defaults(raw: Any) -> Dict[str, Any]:
        if not isinstance(raw, dict):
            return {}
        out: Dict[str, Any] = {}
        for k, v in raw.items():
            if not isinstance(k, str) or not k:
                continue
            # Allow JSON-serializable values (including arrays/objects) for defaults.
            # These are cloned at use-sites to avoid cross-run mutation.
            if v is None or isinstance(v, (str, int, float, bool, dict, list)):
                out[k] = v
        return out

    def _clone_default(value: Any) -> Any:
        # Prevent accidental shared-mutation of dict/list defaults across runs.
        if isinstance(value, (dict, list)):
            try:
                import copy

                return copy.deepcopy(value)
            except Exception:
                return value
        return value

    pin_defaults_by_node_id: Dict[str, Dict[str, Any]] = {}
    for node in visual.nodes:
        raw_defaults = node.data.get("pinDefaults") if isinstance(node.data, dict) else None
        normalized = _normalize_pin_defaults(raw_defaults)
        if normalized:
            pin_defaults_by_node_id[node.id] = normalized

    LITERAL_NODE_TYPES = {
        "literal_string",
        "literal_number",
        "literal_boolean",
        "literal_json",
        "json_schema",
        "literal_array",
    }

    pure_base_handlers: Dict[str, Any] = {}
    pure_node_ids: set[str] = set()

    def _has_execution_pins(type_str: str, node_data: Dict[str, Any]) -> bool:
        # Internal junction nodes always have execution semantics, even if a host
        # built the VisualFlow programmatically without template pin metadata.
        if type_str == "join_exec":
            return True
        pins: list[Any] = []
        inputs = node_data.get("inputs")
        outputs = node_data.get("outputs")
        if isinstance(inputs, list):
            pins.extend(inputs)
        if isinstance(outputs, list):
            pins.extend(outputs)

        if pins:
            for p in pins:
                if isinstance(p, dict) and p.get("type") == "execution":
                    return True
            return False

        if type_str in LITERAL_NODE_TYPES:
            return False
        # These nodes are pure (data-only) even if the JSON document omitted template pins.
        # This keeps programmatic tests and host-built VisualFlows portable.
        if type_str in {"get_var", "get_context", "bool_var", "var_decl"}:
            return False
        if type_str == "break_object":
            return False
        if type_str == "tool_parameters":
            return False
        if get_builtin_handler(type_str) is not None:
            return False
        return True

    def _input_pin_ids(node_data: Dict[str, Any]) -> set[str]:
        pins = node_data.get("inputs") if isinstance(node_data, dict) else None
        if not isinstance(pins, list):
            return set()
        out: set[str] = set()
        for pin in pins:
            if not isinstance(pin, dict):
                continue
            if pin.get("type") == "execution":
                continue
            pin_id = pin.get("id")
            if isinstance(pin_id, str) and pin_id:
                out.add(pin_id)
        return out

    evaluating: set[str] = set()
    volatile_pure_node_ids: set[str] = getattr(flow, "_volatile_pure_node_ids", set())  # type: ignore[attr-defined]

    def _ensure_node_output(node_id: str) -> None:
        if node_id in flow._node_outputs and node_id not in volatile_pure_node_ids:  # type: ignore[attr-defined]
            return

        handler = pure_base_handlers.get(node_id)
        if handler is None:
            return

        if node_id in evaluating:
            raise ValueError(f"Data edge cycle detected at '{node_id}'")

        evaluating.add(node_id)
        try:
            resolved_input: Dict[str, Any] = {}

            for target_pin, (source_node, source_pin) in data_edge_map.get(node_id, {}).items():
                _ensure_node_output(source_node)
                if source_node not in flow._node_outputs:  # type: ignore[attr-defined]
                    continue
                source_output = flow._node_outputs[source_node]  # type: ignore[attr-defined]
                if isinstance(source_output, dict) and source_pin in source_output:
                    resolved_input[target_pin] = source_output[source_pin]
                elif source_pin in ("result", "output"):
                    resolved_input[target_pin] = source_output

            defaults = pin_defaults_by_node_id.get(node_id)
            if defaults:
                for pin_id, value in defaults.items():
                    if pin_id in data_edge_map.get(node_id, {}):
                        continue
                    if pin_id not in resolved_input:
                        resolved_input[pin_id] = _clone_default(value)

            result = handler(resolved_input if resolved_input else {})
            flow._node_outputs[node_id] = result  # type: ignore[attr-defined]
        finally:
            # IMPORTANT: even if an upstream pure node raises (bad input / parse_json failure),
            # we must not leave `node_id` in `evaluating`, otherwise later evaluations can
            # surface as a misleading "data edge cycle" at this node.
            try:
                evaluating.remove(node_id)
            except KeyError:
                pass

    EFFECT_NODE_TYPES = {
        "ask_user",
        "answer_user",
        "model_residency",
        "llm_call",
        "generate_image",
        "edit_image",
        "image_to_image",
        "upscale_image",
        "image_upscale",
        "generate_video",
        "text_to_video",
        "image_to_video",
        "generate_voice",
        "generate_music",
        "transcribe_audio",
        "listen_voice",
        "tool_calls",
        "call_tool",
        "wait_until",
        "wait_event",
        "emit_event",
        "memory_note",
        "memory_query",
        "memory_tag",
        "memory_compact",
        "memory_rehydrate",
        "memory_kg_assert",
        "memory_kg_query",
        "memory_kg_resolve",
    }

    literal_node_ids: set[str] = set()
    # Pre-evaluate literal nodes and store their values
    for node in visual.nodes:
        type_str = node.type.value if hasattr(node.type, "value") else str(node.type)
        if type_str in LITERAL_NODE_TYPES:
            literal_value = node.data.get("literalValue")
            flow._node_outputs[node.id] = {"value": literal_value}  # type: ignore[attr-defined]
            literal_node_ids.add(node.id)
    # Capture baseline outputs (typically only literal nodes). This baseline must
    # remain stable across runs so we can safely reset `_node_outputs` when switching
    # between different `RunState.run_id` contexts (self-recursive subflows).
    try:
        flow._static_node_outputs = dict(flow._node_outputs)  # type: ignore[attr-defined]
    except Exception:
        flow._static_node_outputs = {}  # type: ignore[attr-defined]

    # Compute execution reachability and ignore disconnected execution nodes.
    #
    # Visual editors often contain experimentation / orphan nodes. These should not
    # prevent execution of the reachable pipeline.
    exec_node_ids: set[str] = set()
    for node in visual.nodes:
        type_str = node.type.value if hasattr(node.type, "value") else str(node.type)
        if type_str in LITERAL_NODE_TYPES:
            continue
        if _has_execution_pins(type_str, node.data):
            exec_node_ids.add(node.id)

    def _pick_entry() -> Optional[str]:
        # Prefer explicit entryNode if it is an execution node.
        if isinstance(getattr(visual, "entryNode", None), str) and visual.entryNode in exec_node_ids:
            return visual.entryNode
        # Otherwise, infer entry as a node with no incoming execution edges.
        targets = {e.target for e in visual.edges if getattr(e, "targetHandle", None) == "exec-in"}
        for node in visual.nodes:
            if node.id in exec_node_ids and node.id not in targets:
                return node.id
        # Fallback: first exec node in document order
        for node in visual.nodes:
            if node.id in exec_node_ids:
                return node.id
        return None

    entry_exec = _pick_entry()
    reachable_exec: set[str] = set()
    if entry_exec:
        adj: Dict[str, list[str]] = {}
        for e in visual.edges:
            if getattr(e, "targetHandle", None) != "exec-in":
                continue
            if e.source not in exec_node_ids or e.target not in exec_node_ids:
                continue
            adj.setdefault(e.source, []).append(e.target)
        stack = [entry_exec]
        while stack:
            cur = stack.pop()
            if cur in reachable_exec:
                continue
            reachable_exec.add(cur)
            for nxt in adj.get(cur, []):
                if nxt not in reachable_exec:
                    stack.append(nxt)

    ignored_exec = sorted([nid for nid in exec_node_ids if nid not in reachable_exec])
    if ignored_exec:
        # Runtime-local metadata for hosts/UIs that want to show warnings.
        flow._ignored_exec_nodes = ignored_exec  # type: ignore[attr-defined]

    def _decode_separator(value: str) -> str:
        return value.replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")

    def _resolve_user_file_path(payload: Dict[str, Any], file_path: str, *, operation: str):
        from pathlib import Path

        try:
            from abstractruntime.integrations.abstractcore.workspace_scoped_tools import (
                WorkspaceScope,
                resolve_user_path,
                resolve_user_workspace_path,
            )

            scope = WorkspaceScope.from_input_data(payload)
            if scope is not None:
                resolved = resolve_user_workspace_path(scope=scope, user_path=file_path)
                return resolved.resolved_path, resolved.virtual_path
        except Exception as e:
            raise ValueError(f"{operation} path rejected by workspace policy: {e}") from e

        path = Path(file_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        return path, str(path)

    def _artifact_id_from_value(value: Any) -> str:
        if isinstance(value, dict):
            raw = value.get("$artifact") or value.get("artifact_id")
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
        if isinstance(value, str) and value.strip():
            return value.strip()
        return ""

    def _artifact_store_from_payload(payload: Dict[str, Any]) -> Any:
        store = payload.get("_runtime_artifact_store")
        return store if store is not None else None

    def _runtime_artifact_ref(meta: Any) -> Dict[str, Any]:
        tags = dict(getattr(meta, "tags", None) or {})
        ref: Dict[str, Any] = {
            "$artifact": str(getattr(meta, "artifact_id", "") or ""),
            "artifact_id": str(getattr(meta, "artifact_id", "") or ""),
            "run_id": str(getattr(meta, "run_id", "") or "") or None,
            "content_type": str(getattr(meta, "content_type", "") or "") or None,
            "modality": str(tags.get("modality") or "") or None,
        }
        filename = str(tags.get("filename") or "").strip()
        if filename:
            ref["filename"] = filename
        source_path = str(tags.get("path") or tags.get("source_path") or "").strip()
        if source_path:
            ref["source_path"] = source_path
        size_bytes = getattr(meta, "size_bytes", None)
        if size_bytes is not None:
            try:
                ref["size_bytes"] = int(size_bytes)
            except Exception:
                pass
        return {k: v for k, v in ref.items() if v is not None}

    def _workspace_child_virtual_path(base_virtual_path: str, child_rel: str) -> str:
        base0 = str(base_virtual_path or "").strip().strip("/")
        rel0 = str(child_rel or "").strip().strip("/")
        if not base0:
            return rel0
        if not rel0:
            return base0
        return f"{base0}/{rel0}"

    def _parse_extensions(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            items = value
        elif isinstance(value, tuple):
            items = list(value)
        else:
            items = str(value).replace("\n", ",").split(",")
        out: list[str] = []
        seen: set[str] = set()
        for item in items:
            text = str(item or "").strip().lower()
            if not text:
                continue
            if text.startswith("."):
                text = text[1:]
            if text and text not in seen:
                out.append(text)
                seen.add(text)
        return out

    def _create_read_file_handler(_data: Dict[str, Any]):
        import json

        def handler(input_data: Any) -> Dict[str, Any]:
            payload = input_data if isinstance(input_data, dict) else {}
            raw_path = payload.get("file_path")
            if not isinstance(raw_path, str) or not raw_path.strip():
                raise ValueError("read_file requires a non-empty 'file_path' input.")

            file_path = raw_path.strip()
            path, virtual_path = _resolve_user_file_path(payload, file_path, operation="read_file")

            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            if not path.is_file():
                raise ValueError(f"Not a file: {file_path}")

            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError as e:
                raise ValueError(f"Cannot read '{file_path}' as UTF-8: {e}") from e

            # Detect JSON primarily from file extension; also opportunistically parse
            # when the content looks like JSON. Markdown and text are returned as-is.
            lower_name = path.name.lower()
            content_stripped = text.lstrip()
            looks_like_json = bool(content_stripped) and content_stripped[0] in "{["

            if lower_name.endswith(".json"):
                try:
                    return {"content": json.loads(text), "file_path": virtual_path}
                except Exception as e:
                    raise ValueError(f"Invalid JSON in '{file_path}': {e}") from e

            if looks_like_json:
                try:
                    return {"content": json.loads(text), "file_path": virtual_path}
                except Exception:
                    pass

            return {"content": text, "file_path": virtual_path}

        return handler

    def _create_write_file_handler(_data: Dict[str, Any]):
        import json

        def handler(input_data: Any) -> Dict[str, Any]:
            payload = input_data if isinstance(input_data, dict) else {}
            raw_path = payload.get("file_path")
            if not isinstance(raw_path, str) or not raw_path.strip():
                raise ValueError("write_file requires a non-empty 'file_path' input.")

            file_path = raw_path.strip()
            path, virtual_path = _resolve_user_file_path(payload, file_path, operation="write_file")

            raw_content = payload.get("content")

            if path.name.lower().endswith(".json"):
                if isinstance(raw_content, str):
                    try:
                        raw_content = json.loads(raw_content)
                    except Exception as e:
                        raise ValueError(f"write_file JSON content must be valid JSON: {e}") from e
                text = json.dumps(raw_content, indent=2, ensure_ascii=False)
                if not text.endswith("\n"):
                    text += "\n"
            else:
                if raw_content is None:
                    text = ""
                elif isinstance(raw_content, str):
                    text = raw_content
                elif isinstance(raw_content, (dict, list)):
                    text = json.dumps(raw_content, indent=2, ensure_ascii=False)
                else:
                    text = str(raw_content)

            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")

            return {"bytes": len(text.encode("utf-8")), "file_path": virtual_path}

        return handler

    def _create_read_pdf_handler(_data: Dict[str, Any]):
        def handler(input_data: Any) -> Dict[str, Any]:
            payload = input_data if isinstance(input_data, dict) else {}
            raw_path = payload.get("file_path")
            if not isinstance(raw_path, str) or not raw_path.strip():
                raise ValueError("read_pdf requires a non-empty 'file_path' input.")

            file_path = raw_path.strip()
            path, virtual_path = _resolve_user_file_path(payload, file_path, operation="read_pdf")

            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            if not path.is_file():
                raise ValueError(f"Not a file: {file_path}")
            if path.suffix.lower() != ".pdf":
                raise ValueError("read_pdf requires a .pdf file path.")

            from abstractruntime.documents import extract_pdf_text

            result = extract_pdf_text(
                path,
                page_start=payload.get("page_start"),
                page_end=payload.get("page_end"),
                max_chars=payload.get("max_chars"),
            )
            return {
                "content": result.content,
                "pages": result.pages,
                "processed_pages": result.processed_pages,
                "metadata": result.metadata,
                "warnings": result.warnings,
                "truncated": result.truncated,
                "content_type": "application/pdf",
                "file_path": virtual_path,
            }

        return handler

    def _create_write_pdf_handler(_data: Dict[str, Any]):
        def handler(input_data: Any) -> Dict[str, Any]:
            payload = input_data if isinstance(input_data, dict) else {}
            raw_path = payload.get("file_path")
            if not isinstance(raw_path, str) or not raw_path.strip():
                raise ValueError("write_pdf requires a non-empty 'file_path' input.")

            file_path = raw_path.strip()
            path, virtual_path = _resolve_user_file_path(payload, file_path, operation="write_pdf")
            if path.suffix.lower() != ".pdf":
                raise ValueError("write_pdf requires a .pdf file path.")

            from abstractruntime.documents import render_pdf_bytes

            title = payload.get("title")
            pdf_title = title.strip() if isinstance(title, str) and title.strip() else None
            pdf_bytes, metadata = render_pdf_bytes(payload.get("content"), title=pdf_title)

            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(pdf_bytes)

            return {
                **metadata,
                "file_path": virtual_path,
            }

        return handler

    def _create_list_folder_files_handler(_data: Dict[str, Any]):
        from abstractcore.tools.abstractignore import AbstractIgnore
        from abstractcore.utils.file_filters import file_matches_filters, guess_file_family
        from pathlib import Path
        import os

        def handler(input_data: Any) -> Dict[str, Any]:
            payload = input_data if isinstance(input_data, dict) else {}
            raw_path = payload.get("folder_path")
            if not isinstance(raw_path, str) or not raw_path.strip():
                raise ValueError("list_folder_files requires a non-empty 'folder_path' input.")

            folder_path = raw_path.strip()
            folder, virtual_folder = _resolve_user_file_path(payload, folder_path, operation="list_folder_files")
            if not folder.exists():
                raise FileNotFoundError(f"Folder not found: {folder_path}")
            if not folder.is_dir():
                raise ValueError(f"Not a folder: {folder_path}")

            recursive = bool(payload.get("recursive", False))
            include_directories = bool(payload.get("include_directories", False))
            family = str(payload.get("family") or "any").strip().lower() or "any"
            extensions = _parse_extensions(payload.get("extensions"))
            limit_raw = payload.get("limit", 500)
            try:
                limit = max(1, min(int(limit_raw), 5000))
            except Exception:
                limit = 500
            max_depth_raw = payload.get("max_depth", 0)
            try:
                max_depth = max(0, int(max_depth_raw))
            except Exception:
                max_depth = 0

            ignore = AbstractIgnore.for_path(folder)
            files: list[str] = []
            entries: list[Dict[str, Any]] = []
            truncated = False

            for dirpath, dirnames, filenames in os.walk(folder):
                current = Path(dirpath)
                rel_dir = current.relative_to(folder).as_posix() if current != folder else ""
                depth = 0 if not rel_dir else len([part for part in rel_dir.split("/") if part])

                kept_dirs: list[str] = []
                for dirname in dirnames:
                    candidate = current / dirname
                    if ignore is not None and ignore.is_ignored(candidate, is_dir=True):
                        continue
                    if max_depth > 0 and depth + 1 > max_depth:
                        continue
                    kept_dirs.append(dirname)
                    if include_directories:
                        rel_child = candidate.relative_to(folder).as_posix()
                        entries.append(
                            {
                                "path": _workspace_child_virtual_path(virtual_folder, rel_child),
                                "name": candidate.name,
                                "kind": "folder",
                                "family": "folder",
                            }
                        )
                        if len(entries) >= limit:
                            truncated = True
                            break
                dirnames[:] = kept_dirs if recursive else []
                if truncated:
                    break

                for filename in filenames:
                    candidate = current / filename
                    if ignore is not None and ignore.is_ignored(candidate, is_dir=False):
                        continue
                    if not file_matches_filters(candidate, family=family, extensions=extensions):
                        continue
                    rel_child = candidate.relative_to(folder).as_posix()
                    virtual_child = _workspace_child_virtual_path(virtual_folder, rel_child)
                    files.append(virtual_child)
                    entry: Dict[str, Any] = {
                        "path": virtual_child,
                        "name": candidate.name,
                        "kind": "file",
                        "family": guess_file_family(candidate),
                    }
                    try:
                        entry["size_bytes"] = int(candidate.stat().st_size)
                    except Exception:
                        pass
                    entries.append(entry)
                    if len(entries) >= limit:
                        truncated = True
                        break
                if truncated or not recursive:
                    break

            return {
                "files": files,
                "entries": entries,
                "count": len(files),
                "truncated": truncated,
                "folder_path": virtual_folder,
            }

        return handler

    def _create_import_workspace_file_handler(_data: Dict[str, Any]):
        import hashlib
        import mimetypes

        def handler(input_data: Any) -> Dict[str, Any]:
            payload = input_data if isinstance(input_data, dict) else {}
            store = _artifact_store_from_payload(payload)
            if store is None:
                raise ValueError("import_workspace_file requires a runtime artifact store.")
            raw_path = payload.get("file_path")
            if not isinstance(raw_path, str) or not raw_path.strip():
                raise ValueError("import_workspace_file requires a non-empty 'file_path' input.")
            path, virtual_path = _resolve_user_file_path(payload, raw_path.strip(), operation="import_workspace_file")
            if not path.exists():
                raise FileNotFoundError(f"File not found: {raw_path}")
            if not path.is_file():
                raise ValueError(f"Not a file: {raw_path}")
            content = path.read_bytes()
            filename = path.name
            content_type = str(payload.get("content_type") or "").strip().lower()
            if not content_type:
                guessed, _enc = mimetypes.guess_type(filename)
                content_type = str(guessed or "application/octet-stream")
            session_id = str(payload.get("_runtime_session_id") or "").strip()
            run_id = str(payload.get("_runtime_run_id") or "").strip() or None
            tags: Dict[str, str] = {
                "kind": "run_input",
                "target": "server",
                "source": "workspace",
                "path": virtual_path,
                "filename": filename,
                "modality": str(payload.get("modality") or ""),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            if session_id:
                tags["session_id"] = session_id
            meta = store.store(bytes(content), content_type=content_type, run_id=run_id, tags=tags)
            ref = _runtime_artifact_ref(meta)
            return {
                "artifact": ref,
                "artifact_ref": ref,
                "artifact_id": str(getattr(meta, "artifact_id", "") or ""),
                "content_type": str(getattr(meta, "content_type", "") or "") or content_type,
                "size_bytes": int(getattr(meta, "size_bytes", 0) or 0),
                "source_path": virtual_path,
            }

        return handler

    def _create_export_artifact_handler(_data: Dict[str, Any]):
        import hashlib

        def handler(input_data: Any) -> Dict[str, Any]:
            payload = input_data if isinstance(input_data, dict) else {}
            store = _artifact_store_from_payload(payload)
            if store is None:
                raise ValueError("export_artifact requires a runtime artifact store.")
            artifact_id = _artifact_id_from_value(payload.get("artifact"))
            if not artifact_id:
                raise ValueError("export_artifact requires an artifact input.")
            destination = payload.get("file_path")
            if not isinstance(destination, str) or not destination.strip():
                raise ValueError("export_artifact requires a non-empty 'file_path' input.")
            artifact = store.load(artifact_id)
            if artifact is None:
                raise FileNotFoundError(f"Artifact not found: {artifact_id}")
            path, virtual_path = _resolve_user_file_path(payload, destination.strip(), operation="export_artifact")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(bytes(artifact.content or b""))
            return {
                "artifact_id": artifact_id,
                "file_path": virtual_path,
                "bytes": len(artifact.content or b""),
                "sha256": hashlib.sha256(bytes(artifact.content or b"")).hexdigest(),
                "content_type": str(getattr(artifact.metadata, "content_type", "") or ""),
            }

        return handler

    def _create_read_artifact_handler(_data: Dict[str, Any]):
        import base64

        def handler(input_data: Any) -> Dict[str, Any]:
            from abstractcore.utils.file_filters import guess_file_family

            payload = input_data if isinstance(input_data, dict) else {}
            store = _artifact_store_from_payload(payload)
            if store is None:
                raise ValueError("read_artifact requires a runtime artifact store.")
            artifact_id = _artifact_id_from_value(payload.get("artifact"))
            if not artifact_id:
                raise ValueError("read_artifact requires an artifact input.")
            artifact = store.load(artifact_id)
            if artifact is None:
                raise FileNotFoundError(f"Artifact not found: {artifact_id}")

            meta = artifact.metadata
            tags = dict(getattr(meta, "tags", None) or {})
            filename = str(tags.get("filename") or "").strip()
            source_path = str(tags.get("path") or tags.get("source_path") or "").strip()
            content_type = str(getattr(meta, "content_type", "") or "").strip().lower()
            content_family = guess_file_family(filename or source_path or artifact_id)
            warnings: list[str] = []
            truncated = False
            text_value = ""
            json_value: Any = None
            base64_value = ""
            content_value: Any = None
            is_binary = False
            raw = bytes(artifact.content or b"")

            max_text_bytes = 262144
            max_binary_bytes = 131072
            if content_type == "application/json" or content_family == "json":
                try:
                    text_candidate = raw.decode("utf-8")
                    json_value = json.loads(text_candidate)
                    text_value = text_candidate
                    content_value = json_value
                except Exception as exc:
                    warnings.append(f"JSON projection failed: {exc}")
            if content_value is None and (content_type.startswith("text/") or content_family in {"text", "code"}):
                try:
                    text_value = raw.decode("utf-8")
                    content_value = text_value
                except Exception as exc:
                    warnings.append(f"UTF-8 text projection failed: {exc}")
                    is_binary = True
            if content_value is None and not is_binary:
                try:
                    text_value = raw.decode("utf-8")
                    if len(raw) <= max_text_bytes:
                        content_value = text_value
                    else:
                        content_value = text_value[:max_text_bytes]
                        truncated = True
                        warnings.append("#TRUNCATION: text projection truncated to 262144 bytes.")
                except Exception:
                    is_binary = True
            if is_binary or content_value is None:
                is_binary = True
                if len(raw) > max_binary_bytes:
                    base64_value = base64.b64encode(raw[:max_binary_bytes]).decode("ascii")
                    truncated = True
                    warnings.append("#TRUNCATION: binary projection truncated to 131072 bytes before base64 encoding.")
                else:
                    base64_value = base64.b64encode(raw).decode("ascii")
                content_value = {"encoding": "base64", "data": base64_value}

            ref = _runtime_artifact_ref(meta)
            return {
                "artifact": ref,
                "artifact_ref": ref,
                "artifact_id": artifact_id,
                "content_type": content_type or "application/octet-stream",
                "content_family": content_family,
                "size_bytes": int(getattr(meta, "size_bytes", len(raw)) or len(raw)),
                "filename": filename,
                "source_path": source_path,
                "content": content_value,
                "text": text_value,
                "json": json_value,
                "binary_base64": base64_value,
                "is_binary": is_binary,
                "warnings": warnings,
                "truncated": truncated,
            }

        return handler

    def _create_concat_handler(data: Dict[str, Any]):
        config = data.get("concatConfig", {}) if isinstance(data, dict) else {}
        separator = " "
        if isinstance(config, dict):
            sep_raw = config.get("separator")
            if isinstance(sep_raw, str):
                separator = sep_raw
        separator = _decode_separator(separator)

        pin_order: list[str] = []
        pins = data.get("inputs") if isinstance(data, dict) else None
        if isinstance(pins, list):
            for p in pins:
                if not isinstance(p, dict):
                    continue
                if p.get("type") == "execution":
                    continue
                pid = p.get("id")
                if isinstance(pid, str) and pid:
                    pin_order.append(pid)

        if not pin_order:
            # Backward-compat: programmatic/test-created VisualNodes may omit template pins.
            # In that case, infer a stable pin order from the provided input keys at runtime
            # (prefer a..z single-letter pins), so `a`, `b`, ... behave as expected.
            pin_order = []

        def handler(input_data: Any) -> str:
            if not isinstance(input_data, dict):
                return str(input_data or "")

            parts: list[str] = []
            if pin_order:
                order = pin_order
            else:
                # Stable inference for missing pin metadata.
                keys = [k for k in input_data.keys() if isinstance(k, str)]
                letter = sorted([k for k in keys if len(k) == 1 and "a" <= k <= "z"])
                other = sorted([k for k in keys if k not in set(letter)])
                order = letter + other

            for pid in order:
                if pid in input_data:
                    v = input_data.get(pid)
                    parts.append("" if v is None else str(v))
            return separator.join(parts)

        return handler

    def _create_make_array_handler(data: Dict[str, Any]):
        """Build an array from 1+ inputs in pin order.

        Design:
        - We treat missing/unset pins as absent (skip None) to avoid surprising `null`
          elements when a pin is present but unconnected.
        - We do NOT flatten arrays/tuples; if you want flattening/concatenation,
          use `array_concat`.
        """
        pin_order: list[str] = []
        pins = data.get("inputs") if isinstance(data, dict) else None
        if isinstance(pins, list):
            for p in pins:
                if not isinstance(p, dict):
                    continue
                if p.get("type") == "execution":
                    continue
                pid = p.get("id")
                if isinstance(pid, str) and pid:
                    pin_order.append(pid)

        if not pin_order:
            pin_order = ["a", "b"]

        def handler(input_data: Any) -> list[Any]:
            if not isinstance(input_data, dict):
                if input_data is None:
                    return []
                if isinstance(input_data, list):
                    return list(input_data)
                if isinstance(input_data, tuple):
                    return list(input_data)
                return [input_data]

            out: list[Any] = []
            for pid in pin_order:
                if pid not in input_data:
                    continue
                v = input_data.get(pid)
                if v is None:
                    continue
                out.append(v)
            return out

        return handler

    def _create_array_concat_handler(data: Dict[str, Any]):
        pin_order: list[str] = []
        pins = data.get("inputs") if isinstance(data, dict) else None
        if isinstance(pins, list):
            for p in pins:
                if not isinstance(p, dict):
                    continue
                if p.get("type") == "execution":
                    continue
                pid = p.get("id")
                if isinstance(pid, str) and pid:
                    pin_order.append(pid)

        if not pin_order:
            pin_order = ["a", "b"]

        def handler(input_data: Any) -> list[Any]:
            if not isinstance(input_data, dict):
                if input_data is None:
                    return []
                if isinstance(input_data, list):
                    return list(input_data)
                if isinstance(input_data, tuple):
                    return list(input_data)
                return [input_data]

            out: list[Any] = []
            for pid in pin_order:
                if pid not in input_data:
                    continue
                v = input_data.get(pid)
                if v is None:
                    continue
                if isinstance(v, list):
                    out.extend(v)
                    continue
                if isinstance(v, tuple):
                    out.extend(list(v))
                    continue
                out.append(v)
            return out

        return handler

    def _create_tool_parameters_handler(data: Dict[str, Any]):
        cfg = data.get("toolParametersConfig", {}) if isinstance(data, dict) else {}
        tool_name = ""
        if isinstance(cfg, dict):
            raw = cfg.get("tool") or cfg.get("name")
            if isinstance(raw, str):
                tool_name = raw.strip()

        param_ids: list[str] = []
        pins = data.get("inputs") if isinstance(data, dict) else None
        if isinstance(pins, list):
            for p in pins:
                if not isinstance(p, dict):
                    continue
                if p.get("type") == "execution":
                    continue
                pid = p.get("id")
                if isinstance(pid, str) and pid.strip():
                    param_ids.append(pid.strip())

        def handler(input_data: Any) -> Dict[str, Any]:
            payload = input_data if isinstance(input_data, dict) else {}

            args: Dict[str, Any] = {}
            out: Dict[str, Any] = {}

            for pid in param_ids:
                value = payload.get(pid) if isinstance(payload, dict) else None
                out[pid] = value
                if value is not None:
                    args[pid] = value

            out["tool_call"] = {"name": tool_name, "arguments": args}
            return out

        return handler

    def _create_break_object_handler(data: Dict[str, Any]):
        config = data.get("breakConfig", {}) if isinstance(data, dict) else {}
        selected = config.get("selectedPaths", []) if isinstance(config, dict) else []
        selected_paths = [p.strip() for p in selected if isinstance(p, str) and p.strip()]

        def _get_path(value: Any, path: str) -> Any:
            current = value
            for part in path.split("."):
                if current is None:
                    return None
                if isinstance(current, dict):
                    current = current.get(part)
                    continue
                if isinstance(current, list) and part.isdigit():
                    idx = int(part)
                    if idx < 0 or idx >= len(current):
                        return None
                    current = current[idx]
                    continue
                return None
            return current

        def handler(input_data):
            src_obj = None
            if isinstance(input_data, dict):
                src_obj = input_data.get("object")

            # Best-effort: tolerate JSON-ish strings (common when breaking LLM outputs).
            if isinstance(src_obj, str) and src_obj.strip():
                try:
                    parser = get_builtin_handler("parse_json")
                    if parser is not None:
                        src_obj = parser({"text": src_obj, "wrap_scalar": True})
                except Exception:
                    pass

            out: Dict[str, Any] = {}
            for path in selected_paths:
                out[path] = _get_path(src_obj, path)
            return out

        return handler

    def _get_by_path(value: Any, path: str) -> Any:
        """Best-effort dotted-path lookup supporting dicts and numeric list indices."""
        current = value
        for part in path.split("."):
            if current is None:
                return None
            if isinstance(current, dict):
                current = current.get(part)
                continue
            if isinstance(current, list) and part.isdigit():
                idx = int(part)
                if idx < 0 or idx >= len(current):
                    return None
                current = current[idx]
                continue
            return None
        return current

    def _create_get_var_handler(_data: Dict[str, Any]):
        # Pure node: reads from the current run vars (attached onto the Flow by the compiler).
        # Mark as volatile so it is recomputed whenever requested (avoids stale cached reads).
        def handler(input_data: Any) -> Dict[str, Any]:
            payload = input_data if isinstance(input_data, dict) else {}
            raw_name = payload.get("name")
            name = (raw_name if isinstance(raw_name, str) else str(raw_name or "")).strip()
            run_vars = getattr(flow, "_run_vars", None)  # type: ignore[attr-defined]
            if not isinstance(run_vars, dict) or not name:
                return {"value": None}
            return {"value": _get_by_path(run_vars, name)}

        return handler

    def _create_get_context_handler(_data: Dict[str, Any]):
        # Pure node: reads from the current run vars (attached onto the Flow by the compiler).
        # Mark as volatile so it is recomputed whenever requested (avoids stale cached reads).
        def handler(_input_data: Any) -> Dict[str, Any]:
            del _input_data
            run_vars = getattr(flow, "_run_vars", None)  # type: ignore[attr-defined]
            if not isinstance(run_vars, dict):
                return {"context": {}, "task": "", "messages": []}

            ctx = run_vars.get("context")
            ctx_dict = ctx if isinstance(ctx, dict) else {}

            task = ctx_dict.get("task")
            task_str = task if isinstance(task, str) else str(task or "")

            msgs = ctx_dict.get("messages")
            if isinstance(msgs, list):
                messages = msgs
            elif isinstance(msgs, tuple):
                messages = list(msgs)
            elif msgs is None:
                messages = []
            else:
                messages = [msgs]

            return {"context": ctx_dict, "task": task_str, "messages": messages}

        return handler

    def _create_bool_var_handler(data: Dict[str, Any]):
        """Pure node: reads a workflow-level boolean variable from run.vars with a default.

        Config is stored in the visual node's `literalValue` as either:
        - a string: variable name
        - an object: { "name": "...", "default": true|false }
        """
        raw_cfg = data.get("literalValue")
        name_cfg = ""
        default_cfg = False
        if isinstance(raw_cfg, str):
            name_cfg = raw_cfg.strip()
        elif isinstance(raw_cfg, dict):
            n = raw_cfg.get("name")
            if isinstance(n, str):
                name_cfg = n.strip()
            d = raw_cfg.get("default")
            if isinstance(d, bool):
                default_cfg = d

        def handler(input_data: Any) -> Dict[str, Any]:
            del input_data
            run_vars = getattr(flow, "_run_vars", None)  # type: ignore[attr-defined]
            if not isinstance(run_vars, dict) or not name_cfg:
                return {"name": name_cfg, "value": bool(default_cfg)}

            raw = _get_by_path(run_vars, name_cfg)
            if isinstance(raw, bool):
                return {"name": name_cfg, "value": raw}
            return {"name": name_cfg, "value": bool(default_cfg)}

        return handler

    def _create_var_decl_handler(data: Dict[str, Any]):
        """Pure node: typed workflow variable declaration (name + type + default).

        Config is stored in `literalValue`:
          { "name": "...", "type": "boolean|number|string|object|array|any", "default": ... }

        Runtime semantics:
        - Read `run.vars[name]` (via `flow._run_vars`), and return it if it matches the declared type.
        - Otherwise fall back to the declared default.
        """
        raw_cfg = data.get("literalValue")
        name_cfg = ""
        type_cfg = "any"
        default_cfg: Any = None
        if isinstance(raw_cfg, dict):
            n = raw_cfg.get("name")
            if isinstance(n, str):
                name_cfg = n.strip()
            t = raw_cfg.get("type")
            if isinstance(t, str) and t.strip():
                type_cfg = t.strip()
            default_cfg = raw_cfg.get("default")

        allowed_types = {"boolean", "number", "string", "object", "array", "any"}
        if type_cfg not in allowed_types:
            type_cfg = "any"

        def _matches(v: Any) -> bool:
            if type_cfg == "any":
                return True
            if type_cfg == "boolean":
                return isinstance(v, bool)
            if type_cfg == "number":
                return isinstance(v, (int, float)) and not isinstance(v, bool)
            if type_cfg == "string":
                return isinstance(v, str)
            if type_cfg == "array":
                return isinstance(v, list)
            if type_cfg == "object":
                return isinstance(v, dict)
            return True

        def _default_for_type() -> Any:
            if type_cfg == "boolean":
                return False
            if type_cfg == "number":
                return 0
            if type_cfg == "string":
                return ""
            if type_cfg == "array":
                return []
            if type_cfg == "object":
                return {}
            return None

        def handler(input_data: Any) -> Dict[str, Any]:
            del input_data
            run_vars = getattr(flow, "_run_vars", None)  # type: ignore[attr-defined]
            if not isinstance(run_vars, dict) or not name_cfg:
                v = default_cfg if _matches(default_cfg) else _default_for_type()
                return {"name": name_cfg, "value": v}

            raw = _get_by_path(run_vars, name_cfg)
            if _matches(raw):
                return {"name": name_cfg, "value": raw}

            v = default_cfg if _matches(default_cfg) else _default_for_type()
            return {"name": name_cfg, "value": v}

        return handler

    def _create_set_var_handler(_data: Dict[str, Any]):
        # Execution node: does not mutate run.vars here (handled by compiler adapter).
        # This handler exists to participate in data-edge resolution and expose outputs.
        #
        # Important UX contract:
        # - In the visual editor, primitive pins (boolean/number/string) show default UI controls
        #   even when the user hasn't explicitly edited them.
        # - If we treat "missing" as None here, `Set Variable` would write None and this can
        #   cause typed `Variable` (`var_decl`) to fall back to its default (e.g. staying True).
        # - Therefore we default missing primitive values to their natural defaults.
        pins = _data.get("inputs") if isinstance(_data, dict) else None
        value_pin_type: Optional[str] = None
        if isinstance(pins, list):
            for p in pins:
                if not isinstance(p, dict):
                    continue
                if p.get("id") != "value":
                    continue
                t = p.get("type")
                if isinstance(t, str) and t:
                    value_pin_type = t
                break

        def handler(input_data: Any) -> Dict[str, Any]:
            payload = input_data if isinstance(input_data, dict) else {}
            value_specified = isinstance(payload, dict) and "value" in payload
            value = payload.get("value")

            if not value_specified:
                if value_pin_type == "boolean":
                    value = False
                elif value_pin_type == "number":
                    value = 0
                elif value_pin_type == "string":
                    value = ""

            return {"name": payload.get("name"), "value": value}

        return handler

    def _wrap_builtin(handler, data: Dict[str, Any]):
        literal_value = data.get("literalValue")
        # Preserve pin order for builtins that need deterministic input selection (e.g. coalesce).
        pin_order: list[str] = []
        pins = data.get("inputs") if isinstance(data, dict) else None
        if isinstance(pins, list):
            for p in pins:
                if not isinstance(p, dict):
                    continue
                if p.get("type") == "execution":
                    continue
                pid = p.get("id")
                if isinstance(pid, str) and pid:
                    pin_order.append(pid)

        def wrapped(input_data):
            if isinstance(input_data, dict):
                inputs = input_data.copy()
            else:
                inputs = {"value": input_data, "a": input_data, "text": input_data}

            if literal_value is not None:
                inputs["_literalValue"] = literal_value
            if pin_order:
                inputs["_pin_order"] = list(pin_order)

            return handler(inputs)

        return wrapped

    def _create_agent_input_handler(data: Dict[str, Any]):
        cfg = data.get("agentConfig", {}) if isinstance(data, dict) else {}
        cfg = cfg if isinstance(cfg, dict) else {}

        def _normalize_response_schema(raw: Any) -> Optional[Dict[str, Any]]:
            """Normalize a structured-output schema input into a JSON Schema dict.

            Supported inputs (best-effort):
            - JSON Schema dict: {"type":"object","properties":{...}, ...}
            - LMStudio/OpenAI-style wrapper: {"type":"json_schema","json_schema": {"schema": {...}}}
            """
            if raw is None:
                return None

            schema: Optional[Dict[str, Any]] = None
            if isinstance(raw, dict):
                if raw.get("type") == "json_schema" and isinstance(raw.get("json_schema"), dict):
                    inner = raw.get("json_schema")
                    if isinstance(inner, dict) and isinstance(inner.get("schema"), dict):
                        schema = dict(inner.get("schema") or {})
                else:
                    schema = dict(raw)

            if not isinstance(schema, dict) or not schema:
                return None

            # Resolve stable schema refs (no silent fallback).
            ref = schema.get("$ref")
            if isinstance(ref, str) and ref.strip().startswith("abstractsemantics:"):
                try:
                    from abstractsemantics import resolve_schema_ref  # type: ignore
                except Exception as e:
                    import sys

                    raise RuntimeError(
                        "Structured-output schema ref requires `abstractsemantics` to be installed in the same "
                        f"Python environment as the runtime/gateway (cannot resolve $ref={ref!r}). "
                        f"Current python: {sys.executable}. Install with: `pip install abstractsemantics`."
                    ) from e
                resolved = resolve_schema_ref(schema)
                if isinstance(resolved, dict) and resolved:
                    return resolved
                raise RuntimeError(f"Unknown structured-output schema ref: {ref}")

            return schema

        def _normalize_tool_names(raw: Any) -> list[str]:
            if raw is None:
                return []
            items: list[Any]
            if isinstance(raw, list):
                items = raw
            elif isinstance(raw, tuple):
                items = list(raw)
            else:
                items = [raw]
            out: list[str] = []
            for t in items:
                if isinstance(t, str) and t.strip():
                    out.append(t.strip())
            # preserve order, remove duplicates
            seen: set[str] = set()
            uniq: list[str] = []
            for t in out:
                if t in seen:
                    continue
                seen.add(t)
                uniq.append(t)
            return uniq

        def handler(input_data):
            task = ""
            if isinstance(input_data, dict):
                raw_task = input_data.get("prompt")
                task = "" if raw_task is None else str(raw_task)
            else:
                task = str(input_data)

            context_raw = input_data.get("context", {}) if isinstance(input_data, dict) else {}
            context = context_raw if isinstance(context_raw, dict) else {}
            provider = input_data.get("provider") if isinstance(input_data, dict) else None
            model = input_data.get("model") if isinstance(input_data, dict) else None

            system_raw = input_data.get("system") if isinstance(input_data, dict) else ""
            system = system_raw if isinstance(system_raw, str) else str(system_raw or "")

            tools_specified = isinstance(input_data, dict) and "tools" in input_data
            tools_raw = input_data.get("tools") if isinstance(input_data, dict) else None
            tools = _normalize_tool_names(tools_raw) if tools_specified else []
            if not tools_specified:
                tools = _normalize_tool_names(cfg.get("tools"))

            out: Dict[str, Any] = {
                "task": task,
                "context": context,
                "provider": provider if isinstance(provider, str) else None,
                "model": model if isinstance(model, str) else None,
                "system": system,
                "tools": tools,
            }

            # Optional pin overrides (passed through for compiler/runtime consumption).
            if isinstance(input_data, dict) and "max_iterations" in input_data:
                out["max_iterations"] = input_data.get("max_iterations")
            if isinstance(input_data, dict) and (
                "max_in_tokens" in input_data or "max_input_tokens" in input_data or "maxInputTokens" in input_data
            ):
                if "max_in_tokens" in input_data:
                    out["max_input_tokens"] = input_data.get("max_in_tokens")
                elif "max_input_tokens" in input_data:
                    out["max_input_tokens"] = input_data.get("max_input_tokens")
                else:
                    out["max_input_tokens"] = input_data.get("maxInputTokens")
            if isinstance(input_data, dict) and (
                "max_out_tokens" in input_data or "max_output_tokens" in input_data or "maxOutputTokens" in input_data
            ):
                if "max_out_tokens" in input_data:
                    out["max_output_tokens"] = input_data.get("max_out_tokens")
                elif "max_output_tokens" in input_data:
                    out["max_output_tokens"] = input_data.get("max_output_tokens")
                else:
                    out["max_output_tokens"] = input_data.get("maxOutputTokens")
            if isinstance(input_data, dict) and "temperature" in input_data:
                out["temperature"] = input_data.get("temperature")
            if isinstance(input_data, dict) and "seed" in input_data:
                out["seed"] = input_data.get("seed")
            if isinstance(input_data, dict) and "prompt_cache_binding" in input_data:
                binding = input_data.get("prompt_cache_binding")
                if isinstance(binding, dict) and binding:
                    out["prompt_cache_binding"] = dict(binding)
                elif isinstance(binding, str) and binding.strip():
                    out["prompt_cache_binding"] = binding.strip()

            if isinstance(input_data, dict) and ("resp_schema" in input_data or "response_schema" in input_data):
                schema = _normalize_response_schema(
                    input_data.get("resp_schema")
                    if "resp_schema" in input_data
                    else input_data.get("response_schema")
                )
                if isinstance(schema, dict) and schema:
                    out["response_schema"] = schema

            include_context_specified = isinstance(input_data, dict) and (
                "include_context" in input_data or "use_context" in input_data
            )
            if include_context_specified:
                raw_inc = (
                    input_data.get("include_context")
                    if isinstance(input_data, dict) and "include_context" in input_data
                    else input_data.get("use_context") if isinstance(input_data, dict) else None
                )
                out["include_context"] = _coerce_bool(raw_inc)

            # Memory-source access pins (467): pass through for compiler/runtime pre-call scheduling.
            if isinstance(input_data, dict):
                # v0: memory object pin (preferred). Expand into legacy per-pin keys so
                # the compiler/runtime logic remains backward compatible.
                mem_obj = input_data.get("memory")
                if isinstance(mem_obj, dict):
                    for bool_key in (
                        "use_session_attachments",
                        "use_span_memory",
                        "use_semantic_search",
                        "use_kg_memory",
                    ):
                        if bool_key in input_data:
                            continue
                        if bool_key in mem_obj and mem_obj.get(bool_key) is not None:
                            out[bool_key] = _coerce_bool(mem_obj.get(bool_key))

                    for raw_key in (
                        "memory_query",
                        "memory_scope",
                        "recall_level",
                        "max_span_messages",
                        "kg_max_input_tokens",
                        "kg_limit",
                        "kg_min_score",
                    ):
                        if raw_key in input_data:
                            continue
                        if raw_key in mem_obj and mem_obj.get(raw_key) is not None:
                            out[raw_key] = mem_obj.get(raw_key)

                for bool_key in (
                    "use_session_attachments",
                    "use_span_memory",
                    "use_semantic_search",
                    "use_kg_memory",
                ):
                    if bool_key in input_data:
                        out[bool_key] = _coerce_bool(input_data.get(bool_key))

                for raw_key in (
                    "memory_query",
                    "memory_scope",
                    "recall_level",
                    "max_span_messages",
                    "kg_max_input_tokens",
                    "kg_limit",
                    "kg_min_score",
                ):
                    if raw_key in input_data:
                        out[raw_key] = input_data.get(raw_key)

            return out

        return handler

    def _create_subflow_effect_builder(data: Dict[str, Any]):
        input_pin_ids: list[str] = []
        pins = data.get("inputs") if isinstance(data, dict) else None
        if isinstance(pins, list):
            for p in pins:
                if not isinstance(p, dict):
                    continue
                if p.get("type") == "execution":
                    continue
                pid = p.get("id")
                if isinstance(pid, str) and pid:
                    # Control pin (not forwarded into child vars).
                    if pid in {"inherit_context", "inheritContext"}:
                        continue
                    input_pin_ids.append(pid)

        inherit_cfg = None
        if isinstance(data, dict):
            cfg = data.get("effectConfig")
            if isinstance(cfg, dict):
                inherit_cfg = cfg.get("inherit_context")
                if inherit_cfg is None:
                    inherit_cfg = cfg.get("inheritContext")
        inherit_context_default = bool(inherit_cfg) if inherit_cfg is not None else False

        def handler(input_data):
            subflow_id = (
                data.get("subflowId")
                or data.get("flowId")  # legacy
                or data.get("workflowId")
                or data.get("workflow_id")
            )

            sub_vars_dict: Dict[str, Any] = {}
            if isinstance(input_data, dict):
                base: Dict[str, Any] = {}
                if isinstance(input_data.get("vars"), dict):
                    base.update(dict(input_data["vars"]))
                elif isinstance(input_data.get("input"), dict):
                    base.update(dict(input_data["input"]))

                if input_pin_ids:
                    for pid in input_pin_ids:
                        if pid in ("vars", "input") and isinstance(input_data.get(pid), dict):
                            continue
                        if pid in input_data:
                            base[pid] = input_data.get(pid)
                    sub_vars_dict = base
                else:
                    if base:
                        sub_vars_dict = base
                    else:
                        sub_vars_dict = dict(input_data)
            else:
                if input_pin_ids and len(input_pin_ids) == 1:
                    sub_vars_dict = {input_pin_ids[0]: input_data}
                else:
                    sub_vars_dict = {"input": input_data}

            # Never forward control pins into the child run vars.
            sub_vars_dict.pop("inherit_context", None)
            sub_vars_dict.pop("inheritContext", None)
            sub_vars_dict.pop("child_session_id", None)
            sub_vars_dict.pop("childSessionId", None)
            sub_vars_dict.pop("session_id", None)
            sub_vars_dict.pop("sessionId", None)

            inherit_context_specified = isinstance(input_data, dict) and (
                "inherit_context" in input_data or "inheritContext" in input_data
            )
            if inherit_context_specified:
                raw_inherit = (
                    input_data.get("inherit_context")
                    if isinstance(input_data, dict) and "inherit_context" in input_data
                    else input_data.get("inheritContext") if isinstance(input_data, dict) else None
                )
                inherit_context_value = _coerce_bool(raw_inherit)
            else:
                inherit_context_value = inherit_context_default

            pending: Dict[str, Any] = {
                "output": None,
                "_pending_effect": (
                    {
                        "type": "start_subworkflow",
                        "workflow_id": subflow_id,
                        "vars": sub_vars_dict,
                        # Start subworkflows in async+wait mode so hosts (notably AbstractFlow Web)
                        # can tick child runs incrementally and stream their node_start/node_complete
                        # events for better observability (nested/recursive subflows).
                        #
                        # Non-interactive hosts (tests/CLI) still complete synchronously because
                        # FlowRunner.run() auto-drives WAITING(SUBWORKFLOW) children and resumes
                        # parents until completion.
                        "async": True,
                        "wait": True,
                        **({"inherit_context": True} if inherit_context_value else {}),
                    }
                ),
            }

            # Optional: allow overriding the child session_id (useful for scheduling runs where
            # each execution should be isolated from others). This is a control field and is
            # not forwarded into the child vars.
            if isinstance(input_data, dict):
                raw_sid = None
                for k in ("child_session_id", "childSessionId", "session_id", "sessionId"):
                    if k in input_data:
                        raw_sid = input_data.get(k)
                        break
                if isinstance(raw_sid, str) and raw_sid.strip():
                    # Keep the key name stable for the runtime handler.
                    eff = pending.get("_pending_effect")
                    if isinstance(eff, dict):
                        eff["session_id"] = raw_sid.strip()
                        pending["_pending_effect"] = eff

            return pending

        return handler

    def _create_event_handler(event_type: str, data: Dict[str, Any]):
        # Event nodes are special: they bridge external inputs / runtime vars into the graph.
        #
        # Critical constraint: RunState.vars must remain JSON-serializable for durable execution.
        # The runtime persists per-node outputs in `vars["_temp"]["node_outputs"]`. If an event node
        # returns the full `run.vars` dict (which contains `_temp`), we create a self-referential
        # cycle: `_temp -> node_outputs -> <start_output>['_temp'] -> _temp`, which explodes during
        # persistence (e.g. JsonFileRunStore uses dataclasses.asdict()).
        #
        # Therefore, `on_flow_start` must *not* leak internal namespaces like `_temp` into outputs.
        start_pin_ids: list[str] = []
        pins = data.get("outputs") if isinstance(data, dict) else None
        if isinstance(pins, list):
            for p in pins:
                if not isinstance(p, dict):
                    continue
                if p.get("type") == "execution":
                    continue
                pid = p.get("id")
                if isinstance(pid, str) and pid:
                    start_pin_ids.append(pid)

        def handler(input_data):
            if event_type == "on_flow_start":
                # Prefer explicit pins: the visual editor treats non-exec output pins as
                # "Flow Start Parameters" (initial vars). Only expose those by default.
                if isinstance(input_data, dict):
                    defaults_raw = data.get("pinDefaults") if isinstance(data, dict) else None
                    defaults = defaults_raw if isinstance(defaults_raw, dict) else {}
                    if start_pin_ids:
                        out: Dict[str, Any] = {}
                        for pid in start_pin_ids:
                            if pid in input_data:
                                out[pid] = input_data.get(pid)
                                continue
                            if isinstance(pid, str) and pid in defaults:
                                dv = defaults.get(pid)
                                out[pid] = _clone_default(dv)
                                # Also seed run.vars for downstream Get Variable / debugging.
                                if not pid.startswith("_") and pid not in input_data:
                                    input_data[pid] = _clone_default(dv)
                                continue
                            out[pid] = None
                        return out
                    # Backward-compat: older/test-created flows may omit pin metadata.
                    # In that case, expose non-internal keys only (avoid `_temp`, `_limits`, ...).
                    out2 = {k: v for k, v in input_data.items() if isinstance(k, str) and not k.startswith("_")}
                    # If pinDefaults exist, apply them for missing non-internal keys.
                    for k, dv in defaults.items():
                        if not isinstance(k, str) or not k or k.startswith("_"):
                            continue
                        if k in out2 or k in input_data:
                            continue
                        out2[k] = _clone_default(dv)
                        input_data[k] = _clone_default(dv)
                    return out2

                # Non-dict input: if there is a single declared pin, map into it; otherwise
                # keep a generic `input` key.
                if start_pin_ids and len(start_pin_ids) == 1:
                    return {start_pin_ids[0]: input_data}
                return {"input": input_data}
            if event_type == "on_user_request":
                message = input_data.get("message", "") if isinstance(input_data, dict) else str(input_data)
                context = input_data.get("context", {}) if isinstance(input_data, dict) else {}
                return {"message": message, "context": context}
            if event_type == "on_agent_message":
                sender = input_data.get("sender", "unknown") if isinstance(input_data, dict) else "unknown"
                message = input_data.get("message", "") if isinstance(input_data, dict) else str(input_data)
                channel = data.get("eventConfig", {}).get("channel", "")
                return {"sender": sender, "message": message, "channel": channel}
            return input_data

        return handler

    def _create_flow_end_handler(data: Dict[str, Any]):
        pin_ids: list[str] = []
        pins = data.get("inputs") if isinstance(data, dict) else None
        if isinstance(pins, list):
            for p in pins:
                if not isinstance(p, dict):
                    continue
                if p.get("type") == "execution":
                    continue
                pid = p.get("id")
                if isinstance(pid, str) and pid:
                    pin_ids.append(pid)

        def handler(input_data: Any):
            if not pin_ids:
                if isinstance(input_data, dict):
                    return dict(input_data)
                return input_data

            if not isinstance(input_data, dict):
                if len(pin_ids) == 1:
                    return {pin_ids[0]: input_data}
                return {"response": input_data}

            return {pid: input_data.get(pid) for pid in pin_ids}

        return handler

    def _create_expression_handler(expression: str):
        def handler(input_data):
            namespace = {"x": input_data, "input": input_data}
            if isinstance(input_data, dict):
                namespace.update(input_data)
            try:
                return eval(expression, {"__builtins__": {}}, namespace)
            except Exception as e:
                return {"error": str(e)}

        return handler

    def _create_if_handler(data: Dict[str, Any]):
        def handler(input_data):
            condition = input_data.get("condition") if isinstance(input_data, dict) else bool(input_data)
            return {"branch": "true" if condition else "false", "condition": condition}

        return handler

    def _create_switch_handler(data: Dict[str, Any]):
        def handler(input_data):
            value = input_data.get("value") if isinstance(input_data, dict) else input_data

            config = data.get("switchConfig", {}) if isinstance(data, dict) else {}
            raw_cases = config.get("cases", []) if isinstance(config, dict) else []

            value_str = "" if value is None else str(value)
            if isinstance(raw_cases, list):
                for case in raw_cases:
                    if not isinstance(case, dict):
                        continue
                    case_id = case.get("id")
                    case_value = case.get("value")
                    if not isinstance(case_id, str) or not case_id:
                        continue
                    if case_value is None:
                        continue
                    if value_str == str(case_value):
                        return {"branch": f"case:{case_id}", "value": value, "matched": str(case_value)}

            return {"branch": "default", "value": value}

        return handler

    def _create_while_handler(data: Dict[str, Any]):
        def handler(input_data):
            condition = input_data.get("condition") if isinstance(input_data, dict) else bool(input_data)
            return {"condition": bool(condition)}

        return handler

    def _create_for_handler(data: Dict[str, Any]):
        def handler(input_data):
            payload = input_data if isinstance(input_data, dict) else {}
            start = payload.get("start")
            end = payload.get("end")
            step = payload.get("step")
            return {"start": start, "end": end, "step": step}

        return handler

    def _create_loop_handler(data: Dict[str, Any]):
        def handler(input_data):
            items = input_data.get("items") if isinstance(input_data, dict) else input_data
            if items is None:
                items = []
            if not isinstance(items, (list, tuple)):
                items = [items]
            items_list = list(items) if isinstance(items, tuple) else list(items)  # type: ignore[arg-type]
            return {"items": items_list, "count": len(items_list)}

        return handler

    def _coerce_bool(value: Any) -> bool:
        """Best-effort boolean parsing (handles common string forms)."""
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            try:
                return float(value) != 0.0
            except Exception:
                return False
        if isinstance(value, str):
            s = value.strip().lower()
            if not s:
                return False
            if s in {"false", "0", "no", "off"}:
                return False
            if s in {"true", "1", "yes", "on"}:
                return True
        return False

    def _create_effect_handler(effect_type: str, data: Dict[str, Any]):
        effect_config = data.get("effectConfig", {})

        if effect_type == "ask_user":
            return _create_ask_user_handler(data, effect_config)
        if effect_type == "answer_user":
            return _create_answer_user_handler(data, effect_config)
        if effect_type == "model_residency":
            return _create_model_residency_handler(data, effect_config)
        if effect_type == "llm_call":
            return _create_llm_call_handler(data, effect_config)
        if effect_type == "generate_image":
            return _create_generate_image_handler(data, effect_config)
        if effect_type in {"edit_image", "image_to_image"}:
            return _create_edit_image_handler(data, effect_config)
        if effect_type in {"upscale_image", "image_upscale"}:
            return _create_upscale_image_handler(data, effect_config)
        if effect_type in {"generate_video", "text_to_video"}:
            return _create_generate_video_handler(data, effect_config)
        if effect_type == "image_to_video":
            return _create_image_to_video_handler(data, effect_config)
        if effect_type == "generate_voice":
            return _create_generate_voice_handler(data, effect_config)
        if effect_type == "generate_music":
            return _create_generate_music_handler(data, effect_config)
        if effect_type == "transcribe_audio":
            return _create_transcribe_audio_handler(data, effect_config)
        if effect_type == "listen_voice":
            return _create_listen_voice_handler(data, effect_config)
        if effect_type == "tool_calls":
            return _create_tool_calls_handler(data, effect_config)
        if effect_type == "call_tool":
            return _create_call_tool_handler(data, effect_config)
        if effect_type == "wait_until":
            return _create_wait_until_handler(data, effect_config)
        if effect_type == "wait_event":
            return _create_wait_event_handler(data, effect_config)
        if effect_type == "memory_note":
            return _create_memory_note_handler(data, effect_config)
        if effect_type == "memory_query":
            return _create_memory_query_handler(data, effect_config)
        if effect_type == "memory_tag":
            return _create_memory_tag_handler(data, effect_config)
        if effect_type == "memory_compact":
            return _create_memory_compact_handler(data, effect_config)
        if effect_type == "memory_rehydrate":
            return _create_memory_rehydrate_handler(data, effect_config)
        if effect_type == "memory_kg_assert":
            return _create_memory_kg_assert_handler(data, effect_config)
        if effect_type == "memory_kg_query":
            return _create_memory_kg_query_handler(data, effect_config)
        if effect_type == "memory_kg_resolve":
            return _create_memory_kg_resolve_handler(data, effect_config)

        return lambda x: x

    def _create_tool_calls_handler(data: Dict[str, Any], config: Dict[str, Any]):
        import json

        allowed_default = None
        if isinstance(config, dict):
            raw = config.get("allowed_tools")
            if raw is None:
                raw = config.get("allowedTools")
            allowed_default = raw

        def _normalize_str_list(raw: Any) -> list[str]:
            if not isinstance(raw, list):
                return []
            out: list[str] = []
            for x in raw:
                if isinstance(x, str) and x.strip():
                    out.append(x.strip())
            return out

        def _normalize_tool_calls(raw: Any) -> list[Dict[str, Any]]:
            if raw is None:
                return []
            if isinstance(raw, dict):
                return [dict(raw)]
            if isinstance(raw, list):
                out: list[Dict[str, Any]] = []
                for x in raw:
                    if isinstance(x, dict):
                        out.append(dict(x))
                return out
            if isinstance(raw, str) and raw.strip():
                # Best-effort: tolerate JSON strings coming from parse_json/text nodes.
                try:
                    parsed = json.loads(raw)
                except Exception:
                    return []
                return _normalize_tool_calls(parsed)
            return []

        def handler(input_data: Any):
            payload = input_data if isinstance(input_data, dict) else {}

            tool_calls_raw = payload.get("tool_calls")
            tool_calls = _normalize_tool_calls(tool_calls_raw)

            allow_specified = "allowed_tools" in payload or "allowedTools" in payload
            allowed_raw = payload.get("allowed_tools")
            if allowed_raw is None:
                allowed_raw = payload.get("allowedTools")
            allowed_tools = _normalize_str_list(allowed_raw) if allow_specified else []
            if not allow_specified:
                allowed_tools = _normalize_str_list(allowed_default)

            pending: Dict[str, Any] = {"type": "tool_calls", "tool_calls": tool_calls}
            # Only include allowlist when explicitly provided (empty list means "allow none").
            if allow_specified or isinstance(allowed_default, list):
                pending["allowed_tools"] = allowed_tools

            return {
                "results": None,
                "success": None,
                "_pending_effect": pending,
            }

        return handler

    def _create_call_tool_handler(data: Dict[str, Any], config: Dict[str, Any]):
        import json

        allowed_default = None
        if isinstance(config, dict):
            raw = config.get("allowed_tools")
            if raw is None:
                raw = config.get("allowedTools")
            allowed_default = raw

        def _normalize_str_list(raw: Any) -> list[str]:
            if not isinstance(raw, list):
                return []
            out: list[str] = []
            for x in raw:
                if isinstance(x, str) and x.strip():
                    out.append(x.strip())
            return out

        def _normalize_tool_call(raw: Any) -> Optional[Dict[str, Any]]:
            if raw is None:
                return None
            if isinstance(raw, dict):
                return dict(raw)
            if isinstance(raw, str) and raw.strip():
                # Best-effort: tolerate JSON strings coming from stringify/parse nodes.
                try:
                    parsed = json.loads(raw)
                except Exception:
                    return None
                if isinstance(parsed, dict):
                    return dict(parsed)
            return None

        def handler(input_data: Any):
            payload = input_data if isinstance(input_data, dict) else {}

            raw_call = payload.get("tool_call")
            if raw_call is None:
                raw_call = payload.get("toolCall")
            tool_call = _normalize_tool_call(raw_call)
            tool_calls = [tool_call] if isinstance(tool_call, dict) else []

            allow_specified = "allowed_tools" in payload or "allowedTools" in payload
            allowed_raw = payload.get("allowed_tools")
            if allowed_raw is None:
                allowed_raw = payload.get("allowedTools")
            allowed_tools = _normalize_str_list(allowed_raw) if allow_specified else []
            if not allow_specified:
                allowed_tools = _normalize_str_list(allowed_default)

            pending: Dict[str, Any] = {"type": "tool_calls", "tool_calls": tool_calls}
            # Only include allowlist when explicitly provided (empty list means "allow none").
            if allow_specified or isinstance(allowed_default, list):
                pending["allowed_tools"] = allowed_tools

            return {
                "result": None,
                "success": None,
                "_pending_effect": pending,
            }

        return handler

    def _nonempty_str(value: Any) -> str:
        return value.strip() if isinstance(value, str) and value.strip() else ""

    def _coerce_int(value: Any) -> Optional[int]:
        if value is None or isinstance(value, bool):
            return None
        try:
            return int(float(value))
        except Exception:
            return None

    def _coerce_upscale_resolution(value: Any) -> Optional[Any]:
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            lowered = text.lower()
            if lowered.endswith("x"):
                try:
                    scale = float(lowered[:-1])
                except Exception:
                    return None
                return text if scale > 0 else None
            try:
                parsed = int(float(text))
            except Exception:
                return None
            return parsed if parsed > 0 else None
        try:
            parsed = int(float(value))
        except Exception:
            return None
        return parsed if parsed > 0 else None

    def _coerce_float(value: Any) -> Optional[float]:
        if value is None or isinstance(value, bool):
            return None
        try:
            out = float(value)
        except Exception:
            return None
        return out if out == out else None

    def _dict_input(input_data: Any) -> Dict[str, Any]:
        return input_data if isinstance(input_data, dict) else {}

    def _input_or_config(input_data: Any, config: Dict[str, Any], *keys: str, default: Any = None) -> Any:
        payload = _dict_input(input_data)
        for key in keys:
            if key in payload and payload.get(key) not in (None, ""):
                return payload.get(key)
        for key in keys:
            if isinstance(config, dict) and key in config and config.get(key) not in (None, ""):
                return config.get(key)
        return default

    def _normalize_generation_count(value: Any) -> Optional[int]:
        count = _coerce_int(value)
        if count is None or count < 1:
            return None
        return count

    def _normalize_seed_list(value: Any) -> Optional[list[int]]:
        if value is None:
            return None
        parsed = value
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = [part.strip() for part in text.split(",")]
        if not isinstance(parsed, list):
            return None
        seeds: list[int] = []
        for item in parsed:
            seed = _coerce_int(item)
            if seed is None:
                return None
            seeds.append(seed)
        return seeds or None

    def _normalize_lora_adapters(value: Any) -> Optional[list[Dict[str, Any]]]:
        if value is None:
            return None
        parsed = value
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                parsed = json.loads(text)
            except Exception:
                return None
        if isinstance(parsed, dict):
            parsed = [parsed]
        if not isinstance(parsed, list):
            return None
        adapters: list[Dict[str, Any]] = []
        for item in parsed:
            if isinstance(item, dict) and item:
                adapters.append(dict(item))
        return adapters or None

    def _image_media_item(value: Any, *, role: str) -> Any:
        if isinstance(value, dict):
            item = dict(value)
            item.setdefault("type", "image")
            if role and not any(key in item for key in ("role", "purpose", "kind")):
                item["role"] = role
            return item
        if isinstance(value, str) and value.strip():
            ref = value.strip()
            item: Dict[str, Any] = {"type": "image", "role": role}
            if ref.lower().startswith("data:") or ref.startswith(("http://", "https://")):
                item["url"] = ref
            elif ref.startswith(("/", "~")):
                item["file_path"] = ref
            else:
                item["$artifact"] = ref
            return item
        return None

    def _create_generate_image_handler(data: Dict[str, Any], config: Dict[str, Any]):
        def handler(input_data: Any):
            payload = _dict_input(input_data)
            prompt = str(_input_or_config(payload, config, "prompt", default="") or "")
            fmt = str(_input_or_config(payload, config, "format", "response_format", default="png") or "png").strip().lower() or "png"
            output_spec: Dict[str, Any] = {"modality": "image", "task": "image_generation", "format": fmt}
            for key in ("size", "negative_prompt", "quality", "style"):
                value = _input_or_config(payload, config, key)
                if isinstance(value, str) and value.strip():
                    output_spec[key] = value.strip()
            for key in ("width", "height", "seed", "steps"):
                value = _coerce_int(_input_or_config(payload, config, key))
                if value is not None:
                    output_spec[key] = value
            guidance = _coerce_float(_input_or_config(payload, config, "guidance_scale", "guidanceScale"))
            if guidance is not None:
                output_spec["guidance_scale"] = guidance
            guidance_2 = _coerce_float(_input_or_config(payload, config, "guidance_2", "guidance2", "guidanceTwo"))
            if guidance_2 is not None:
                output_spec["guidance_2"] = guidance_2
            count = _normalize_generation_count(_input_or_config(payload, config, "count", "n"))
            if count is not None:
                output_spec["count"] = count
            seeds = _normalize_seed_list(_input_or_config(payload, config, "seeds"))
            if seeds is not None:
                output_spec["seeds"] = seeds
            lora_adapters = _normalize_lora_adapters(
                _input_or_config(payload, config, "lora_adapters", "loraAdapters")
            )
            if lora_adapters is not None:
                output_spec["lora_adapters"] = lora_adapters
            extra = _input_or_config(payload, config, "extra")
            if isinstance(extra, dict) and extra:
                output_spec["extra"] = dict(extra)
            image_provider = _nonempty_str(
                _input_or_config(payload, config, "image_provider", "imageProvider", "provider_image")
            )
            image_model = _nonempty_str(
                _input_or_config(payload, config, "image_model", "imageModel", "model_image")
            )
            if image_provider:
                output_spec["provider"] = image_provider
            if image_model:
                output_spec["model"] = image_model
            pending: Dict[str, Any] = {
                "type": "llm_call",
                "prompt": prompt,
                "system_prompt": "",
                "tools": [],
                "params": {},
                "output": output_spec,
            }
            return {"image_artifact": None, "artifact_ref": None, "artifact_id": "", "success": None, "_pending_effect": pending}

        return handler

    def _create_edit_image_handler(data: Dict[str, Any], config: Dict[str, Any]):
        def handler(input_data: Any):
            payload = _dict_input(input_data)
            prompt = str(_input_or_config(payload, config, "prompt", "text", default="") or "")
            fmt = str(_input_or_config(payload, config, "format", "response_format", default="png") or "png").strip().lower() or "png"
            output_spec: Dict[str, Any] = {"modality": "image", "task": "image_edit", "format": fmt}
            for key in ("size", "negative_prompt", "quality", "style"):
                value = _input_or_config(payload, config, key)
                if isinstance(value, str) and value.strip():
                    output_spec[key] = value.strip()
            for key in ("width", "height", "seed", "steps"):
                value = _coerce_int(_input_or_config(payload, config, key))
                if value is not None:
                    output_spec[key] = value
            guidance = _coerce_float(_input_or_config(payload, config, "guidance_scale", "guidanceScale"))
            if guidance is not None:
                output_spec["guidance_scale"] = guidance
            guidance_2 = _coerce_float(_input_or_config(payload, config, "guidance_2", "guidance2", "guidanceTwo"))
            if guidance_2 is not None:
                output_spec["guidance_2"] = guidance_2
            strength = _coerce_float(_input_or_config(payload, config, "strength"))
            if strength is not None:
                output_spec["strength"] = strength
            count = _normalize_generation_count(_input_or_config(payload, config, "count", "n"))
            if count is not None:
                output_spec["count"] = count
            seeds = _normalize_seed_list(_input_or_config(payload, config, "seeds"))
            if seeds is not None:
                output_spec["seeds"] = seeds
            lora_adapters = _normalize_lora_adapters(
                _input_or_config(payload, config, "lora_adapters", "loraAdapters")
            )
            if lora_adapters is not None:
                output_spec["lora_adapters"] = lora_adapters
            extra = _input_or_config(payload, config, "extra")
            if isinstance(extra, dict) and extra:
                output_spec["extra"] = dict(extra)
            image_provider = _nonempty_str(
                _input_or_config(payload, config, "image_provider", "imageProvider", "provider_image")
            )
            image_model = _nonempty_str(
                _input_or_config(payload, config, "image_model", "imageModel", "model_image")
            )
            if image_provider:
                output_spec["provider"] = image_provider
            if image_model:
                output_spec["model"] = image_model

            source_ref = _input_or_config(
                payload,
                config,
                "image_artifact",
                "source_image",
                "sourceImage",
                "input_image",
                "inputImage",
                "image",
                "artifact",
                "file_path",
                "filePath",
                "source",
            )
            mask_ref = _input_or_config(payload, config, "mask_artifact", "mask_image", "maskImage", "image_mask", "mask")
            media = []
            source_item = _image_media_item(source_ref, role="source")
            if source_item:
                media.append(source_item)
            mask_item = _image_media_item(mask_ref, role="mask")
            if mask_item:
                media.append(mask_item)

            pending: Dict[str, Any] = {
                "type": "llm_call",
                "prompt": prompt,
                "system_prompt": "",
                "tools": [],
                "params": {},
                "media": media,
                "output": output_spec,
            }
            return {"image_artifact": None, "artifact_ref": None, "artifact_id": "", "success": None, "_pending_effect": pending}

        return handler

    def _create_upscale_image_handler(data: Dict[str, Any], config: Dict[str, Any]):
        def handler(input_data: Any):
            payload = _dict_input(input_data)
            fmt = str(_input_or_config(payload, config, "format", "response_format", default="png") or "png").strip().lower() or "png"
            output_spec: Dict[str, Any] = {"modality": "image", "task": "image_upscale", "format": fmt}
            for key in ("scale", "output_format"):
                value = _input_or_config(payload, config, key)
                if isinstance(value, str) and value.strip():
                    output_spec[key] = value.strip()
            resolution = _coerce_upscale_resolution(_input_or_config(payload, config, "resolution"))
            if resolution is not None:
                output_spec["resolution"] = resolution
            for key in ("seed", "quantize"):
                value = _coerce_int(_input_or_config(payload, config, key))
                if value is not None:
                    output_spec[key] = value
            softness = _coerce_float(_input_or_config(payload, config, "softness"))
            if softness is not None:
                output_spec["softness"] = softness
            if "vae_tiling" in payload or "vaeTiling" in payload or (isinstance(config, dict) and ("vae_tiling" in config or "vaeTiling" in config)):
                output_spec["vae_tiling"] = _coerce_bool(_input_or_config(payload, config, "vae_tiling", "vaeTiling"))
            extra = _input_or_config(payload, config, "extra")
            if isinstance(extra, dict) and extra:
                output_spec["extra"] = dict(extra)
            image_provider = _nonempty_str(
                _input_or_config(payload, config, "image_provider", "imageProvider", "provider_image")
            )
            image_model = _nonempty_str(
                _input_or_config(payload, config, "image_model", "imageModel", "model_image")
            )
            if image_provider:
                output_spec["provider"] = image_provider
            if image_model:
                output_spec["model"] = image_model

            source_ref = _input_or_config(
                payload,
                config,
                "image_artifact",
                "source_image",
                "sourceImage",
                "input_image",
                "inputImage",
                "image",
                "artifact",
                "file_path",
                "filePath",
                "source",
            )
            media = []
            source_item = _image_media_item(source_ref, role="source")
            if source_item:
                media.append(source_item)
            pending: Dict[str, Any] = {
                "type": "llm_call",
                "prompt": str(_input_or_config(payload, config, "prompt", default="") or ""),
                "system_prompt": "",
                "tools": [],
                "params": {},
                "media": media,
                "output": output_spec,
            }
            return {"image_artifact": None, "artifact_ref": None, "artifact_id": "", "success": None, "_pending_effect": pending}

        return handler

    def _create_generate_video_handler(data: Dict[str, Any], config: Dict[str, Any]):
        def handler(input_data: Any):
            payload = _dict_input(input_data)
            prompt = str(_input_or_config(payload, config, "prompt", "text", default="") or "")
            fmt = str(_input_or_config(payload, config, "format", "response_format", default="mp4") or "mp4").strip().lower() or "mp4"
            output_spec: Dict[str, Any] = {"modality": "video", "task": "text_to_video", "format": fmt}
            for key in ("size", "negative_prompt"):
                value = _input_or_config(payload, config, key)
                if isinstance(value, str) and value.strip():
                    output_spec[key] = value.strip()
            for key in ("width", "height", "seed", "steps", "fps"):
                value = _coerce_int(_input_or_config(payload, config, key))
                if value is not None:
                    output_spec[key] = value
            frames = _coerce_int(_input_or_config(payload, config, "num_frames", "numFrames", "frames"))
            if frames is not None:
                output_spec["num_frames"] = frames
            guidance = _coerce_float(_input_or_config(payload, config, "guidance_scale", "guidanceScale"))
            if guidance is not None:
                output_spec["guidance_scale"] = guidance
            guidance_2 = _coerce_float(_input_or_config(payload, config, "guidance_2", "guidance2", "guidanceTwo"))
            if guidance_2 is not None:
                output_spec["guidance_2"] = guidance_2
            count = _normalize_generation_count(_input_or_config(payload, config, "count", "n"))
            if count is not None:
                output_spec["count"] = count
            seeds = _normalize_seed_list(_input_or_config(payload, config, "seeds"))
            if seeds is not None:
                output_spec["seeds"] = seeds
            flow_shift = _coerce_float(_input_or_config(payload, config, "flow_shift", "flowShift"))
            if flow_shift is not None:
                output_spec["flow_shift"] = flow_shift
            lora_adapters = _normalize_lora_adapters(
                _input_or_config(payload, config, "lora_adapters", "loraAdapters")
            )
            if lora_adapters is not None:
                output_spec["lora_adapters"] = lora_adapters
            extra = _input_or_config(payload, config, "extra")
            if isinstance(extra, dict) and extra:
                output_spec["extra"] = dict(extra)
            video_provider = _nonempty_str(
                _input_or_config(payload, config, "video_provider", "videoProvider", "provider_video")
            )
            video_model = _nonempty_str(_input_or_config(payload, config, "video_model", "videoModel", "model_video"))
            if video_provider:
                output_spec["provider"] = video_provider
            if video_model:
                output_spec["model"] = video_model
            pending: Dict[str, Any] = {
                "type": "llm_call",
                "prompt": prompt,
                "system_prompt": "",
                "tools": [],
                "params": {},
                "output": output_spec,
            }
            return {"video_artifact": None, "artifact_ref": None, "artifact_id": "", "success": None, "_pending_effect": pending}

        return handler

    def _create_image_to_video_handler(data: Dict[str, Any], config: Dict[str, Any]):
        def handler(input_data: Any):
            payload = _dict_input(input_data)
            prompt = str(_input_or_config(payload, config, "prompt", "text", default="") or "")
            fmt = str(_input_or_config(payload, config, "format", "response_format", default="mp4") or "mp4").strip().lower() or "mp4"
            output_spec: Dict[str, Any] = {"modality": "video", "task": "image_to_video", "format": fmt}
            for key in ("size", "negative_prompt"):
                value = _input_or_config(payload, config, key)
                if isinstance(value, str) and value.strip():
                    output_spec[key] = value.strip()
            for key in ("width", "height", "seed", "steps", "fps"):
                value = _coerce_int(_input_or_config(payload, config, key))
                if value is not None:
                    output_spec[key] = value
            frames = _coerce_int(_input_or_config(payload, config, "num_frames", "numFrames", "frames"))
            if frames is not None:
                output_spec["num_frames"] = frames
            guidance = _coerce_float(_input_or_config(payload, config, "guidance_scale", "guidanceScale"))
            if guidance is not None:
                output_spec["guidance_scale"] = guidance
            guidance_2 = _coerce_float(_input_or_config(payload, config, "guidance_2", "guidance2", "guidanceTwo"))
            if guidance_2 is not None:
                output_spec["guidance_2"] = guidance_2
            count = _normalize_generation_count(_input_or_config(payload, config, "count", "n"))
            if count is not None:
                output_spec["count"] = count
            seeds = _normalize_seed_list(_input_or_config(payload, config, "seeds"))
            if seeds is not None:
                output_spec["seeds"] = seeds
            flow_shift = _coerce_float(_input_or_config(payload, config, "flow_shift", "flowShift"))
            if flow_shift is not None:
                output_spec["flow_shift"] = flow_shift
            lora_adapters = _normalize_lora_adapters(
                _input_or_config(payload, config, "lora_adapters", "loraAdapters")
            )
            if lora_adapters is not None:
                output_spec["lora_adapters"] = lora_adapters
            extra = _input_or_config(payload, config, "extra")
            if isinstance(extra, dict) and extra:
                output_spec["extra"] = dict(extra)
            video_provider = _nonempty_str(
                _input_or_config(payload, config, "video_provider", "videoProvider", "provider_video")
            )
            video_model = _nonempty_str(_input_or_config(payload, config, "video_model", "videoModel", "model_video"))
            if video_provider:
                output_spec["provider"] = video_provider
            if video_model:
                output_spec["model"] = video_model

            source_ref = _input_or_config(
                payload,
                config,
                "image_artifact",
                "source_image",
                "sourceImage",
                "input_image",
                "inputImage",
                "image",
                "artifact",
                "file_path",
                "filePath",
                "source",
            )
            media = []
            source_item = _image_media_item(source_ref, role="source")
            if source_item:
                media.append(source_item)
            pending: Dict[str, Any] = {
                "type": "llm_call",
                "prompt": prompt,
                "system_prompt": "",
                "tools": [],
                "params": {},
                "media": media,
                "output": output_spec,
            }
            return {"video_artifact": None, "artifact_ref": None, "artifact_id": "", "success": None, "_pending_effect": pending}

        return handler

    def _create_model_residency_handler(data: Dict[str, Any], config: Dict[str, Any]):
        def handler(input_data: Any):
            payload = _dict_input(input_data)
            operation = str(_input_or_config(payload, config, "operation", default="list_loaded") or "list_loaded").strip()
            task = _nonempty_str(_input_or_config(payload, config, "task"))
            provider = _nonempty_str(_input_or_config(payload, config, "provider"))
            model = _nonempty_str(_input_or_config(payload, config, "model"))
            runtime_id = _nonempty_str(_input_or_config(payload, config, "runtime_id", "runtimeId"))
            base_url = _nonempty_str(_input_or_config(payload, config, "base_url", "baseUrl"))
            provider_api_key = _nonempty_str(_input_or_config(payload, config, "provider_api_key", "api_key", "apiKey"))
            timeout_s = _coerce_float(_input_or_config(payload, config, "timeout_s", "timeout", "timeoutS"))
            options = _input_or_config(payload, config, "options")
            cache_policy = _nonempty_str(_input_or_config(payload, config, "cache_policy", "cachePolicy"))

            pending: Dict[str, Any] = {
                "type": "model_residency",
                "operation": operation or "list_loaded",
            }
            if task:
                pending["task"] = task
            if provider:
                pending["provider"] = provider
            if model:
                pending["model"] = model
            if runtime_id:
                pending["runtime_id"] = runtime_id
            if base_url:
                pending["base_url"] = base_url
            if provider_api_key:
                pending["provider_api_key"] = provider_api_key
            if timeout_s is not None:
                pending["timeout_s"] = timeout_s
            if isinstance(options, dict):
                pending["options"] = dict(options)
            if "pin" in payload or (isinstance(config, dict) and "pin" in config):
                pending["pin"] = _coerce_bool(_input_or_config(payload, config, "pin", default=True))
            if "required" in payload or (isinstance(config, dict) and "required" in config):
                pending["required"] = _coerce_bool(_input_or_config(payload, config, "required", default=False))
            if cache_policy:
                pending["cache_policy"] = cache_policy

            return {
                "ok": None,
                "success": None,
                "affected_models": [],
                "models": [],
                "error": "",
                "warnings": [],
                "result": None,
                "_pending_effect": pending,
            }

        return handler

    def _create_generate_voice_handler(data: Dict[str, Any], config: Dict[str, Any]):
        def handler(input_data: Any):
            payload = _dict_input(input_data)
            text = str(_input_or_config(payload, config, "text", "prompt", default="") or "")
            fmt = str(_input_or_config(payload, config, "format", "response_format", default="wav") or "wav").strip().lower() or "wav"
            output_spec: Dict[str, Any] = {"modality": "voice", "task": "tts", "format": fmt}
            for key in ("voice", "profile", "instructions"):
                value = _input_or_config(payload, config, key)
                if isinstance(value, str) and value.strip():
                    output_spec[key] = value.strip()
            quality_preset = _nonempty_str(_input_or_config(payload, config, "quality_preset", "qualityPreset", "quality"))
            if quality_preset:
                output_spec["quality_preset"] = quality_preset
            tts_provider = _nonempty_str(
                _input_or_config(payload, config, "tts_provider", "ttsProvider", "provider_voice")
            )
            tts_model = _nonempty_str(
                _input_or_config(payload, config, "tts_model", "ttsModel", "model_voice")
            )
            if tts_provider:
                output_spec["provider"] = tts_provider
            if tts_model:
                output_spec["model"] = tts_model
            speed = _coerce_float(_input_or_config(payload, config, "speed"))
            if speed is not None:
                output_spec["speed"] = speed
            extra = _input_or_config(payload, config, "extra")
            if isinstance(extra, dict) and extra:
                output_spec["extra"] = dict(extra)
            pending: Dict[str, Any] = {
                "type": "llm_call",
                "prompt": text,
                "system_prompt": "",
                "tools": [],
                "params": {},
                "output": output_spec,
            }
            return {"audio_artifact": None, "artifact_ref": None, "artifact_id": "", "success": None, "_pending_effect": pending}

        return handler

    def _create_generate_music_handler(data: Dict[str, Any], config: Dict[str, Any]):
        def handler(input_data: Any):
            payload = _dict_input(input_data)
            prompt = str(_input_or_config(payload, config, "prompt", "text", default="") or "")
            fmt = (
                str(_input_or_config(payload, config, "format", "response_format", default="wav") or "wav")
                .strip()
                .lower()
                or "wav"
            )
            output_spec: Dict[str, Any] = {"modality": "music", "task": "music_generation", "format": fmt}

            for key in (
                "lyrics",
                "text_planner_mode",
                "vocal_language",
                "negative_prompt",
                "keyscale",
                "timesignature",
            ):
                value = _input_or_config(payload, config, key)
                if isinstance(value, str) and value.strip():
                    output_spec[key] = value.strip()

            duration_s = _coerce_float(_input_or_config(payload, config, "duration_s", "duration", "durationS"))
            if duration_s is not None:
                output_spec["duration_s"] = duration_s

            bpm = _coerce_float(_input_or_config(payload, config, "bpm"))
            if bpm is not None:
                output_spec["bpm"] = bpm

            seed = _coerce_int(_input_or_config(payload, config, "seed"))
            if seed is not None:
                output_spec["seed"] = seed

            sample_rate = _coerce_int(_input_or_config(payload, config, "sample_rate", "sampleRate"))
            if sample_rate is not None:
                output_spec["sample_rate"] = sample_rate

            num_steps = _coerce_int(_input_or_config(payload, config, "num_inference_steps", "numInferenceSteps"))
            if num_steps is not None:
                output_spec["num_inference_steps"] = num_steps

            guidance = _coerce_float(_input_or_config(payload, config, "guidance_scale", "guidanceScale"))
            if guidance is not None:
                output_spec["guidance_scale"] = guidance

            for key in ("instrumental", "enhance_prompt", "structure_prompt", "auto_lyrics", "planning"):
                raw_value = _input_or_config(payload, config, key, default=None)
                if raw_value is not None:
                    output_spec[key] = _coerce_bool(raw_value)

            music_provider = _nonempty_str(
                _input_or_config(payload, config, "music_provider", "musicProvider", "provider_music")
            )
            music_model = _nonempty_str(
                _input_or_config(payload, config, "music_model", "musicModel", "model_music")
            )
            legacy_music_backend = _nonempty_str(
                _input_or_config(payload, config, "music_backend", "musicBackend", "backend_music")
            )
            if music_provider:
                output_spec["provider"] = music_provider
            if music_model:
                output_spec["model"] = music_model
            if legacy_music_backend:
                raise ValueError(
                    "generate_music uses `music_provider` as the backend selector; "
                    "`music_backend` is not supported."
                )

            extra = _input_or_config(payload, config, "extra")
            if isinstance(extra, dict) and extra:
                output_spec["extra"] = dict(extra)
            for key in ("composition_plan", "positive_styles", "negative_styles"):
                value = _input_or_config(payload, config, key)
                if isinstance(value, (dict, list)) and value:
                    output_spec[key] = value

            pending: Dict[str, Any] = {
                "type": "llm_call",
                "prompt": prompt,
                "system_prompt": "",
                "tools": [],
                "params": {},
                "output": output_spec,
            }
            return {
                "music_artifact": None,
                "audio_artifact": None,
                "artifact_ref": None,
                "artifact_id": "",
                "success": None,
                "_pending_effect": pending,
            }

        return handler

    def _create_transcribe_audio_handler(data: Dict[str, Any], config: Dict[str, Any]):
        def handler(input_data: Any):
            payload = _dict_input(input_data)
            audio_ref = _input_or_config(payload, config, "audio_artifact", "artifact", "audio")
            output_spec: Dict[str, Any] = {"modality": "text", "task": "transcription"}
            for key in ("language", "prompt", "response_format", "format"):
                value = _input_or_config(payload, config, key)
                if isinstance(value, str) and value.strip():
                    output_spec[key] = value.strip()
            stt_provider = _nonempty_str(
                _input_or_config(payload, config, "stt_provider", "sttProvider", "provider_voice")
            )
            stt_model = _nonempty_str(
                _input_or_config(payload, config, "stt_model", "sttModel", "model_voice")
            )
            if stt_provider:
                output_spec["provider"] = stt_provider
            if stt_model:
                output_spec["model"] = stt_model
            temp = _coerce_float(_input_or_config(payload, config, "temperature"))
            if temp is not None:
                output_spec["temperature"] = temp
            media_item: Any = audio_ref
            if isinstance(audio_ref, str) and audio_ref.strip() and not audio_ref.startswith(("/", "~")):
                media_item = {"$artifact": audio_ref.strip(), "type": "audio"}
            pending: Dict[str, Any] = {
                "type": "llm_call",
                "prompt": str(output_spec.get("prompt") or "Transcribe this audio."),
                "system_prompt": "",
                "tools": [],
                "params": {},
                "media": [media_item] if media_item else [],
                "output": output_spec,
            }
            return {"text": "", "transcript_artifact": None, "success": None, "_pending_effect": pending}

        return handler

    def _create_listen_voice_handler(data: Dict[str, Any], config: Dict[str, Any]):
        def handler(input_data: Any):
            payload = _dict_input(input_data)
            prompt = str(_input_or_config(payload, config, "prompt", default="Speak now.") or "Speak now.")
            wait_key = str(_input_or_config(payload, config, "wait_key", "event_key", default="voice_input") or "voice_input")
            details: Dict[str, Any] = {"input_mode": "voice", "content_types": ["audio/wav", "audio/mpeg", "audio/webm"]}
            language = _input_or_config(payload, config, "language")
            if isinstance(language, str) and language.strip():
                details["language"] = language.strip()
            stt_provider = _nonempty_str(
                _input_or_config(payload, config, "stt_provider", "sttProvider", "provider_voice")
            )
            if stt_provider:
                details["provider"] = stt_provider
            stt_model = _nonempty_str(
                _input_or_config(payload, config, "stt_model", "sttModel", "model_voice")
            )
            if stt_model:
                details["model"] = stt_model
            max_duration_s = _coerce_float(_input_or_config(payload, config, "max_duration_s", "maxDurationS"))
            if max_duration_s is not None:
                details["max_duration_s"] = max_duration_s
            return {
                "audio_artifact": None,
                "text": "",
                "_pending_effect": {
                    "type": "wait_event",
                    "wait_key": wait_key,
                    "prompt": prompt,
                    "allow_free_text": True,
                    "details": details,
                },
            }

        return handler

    def _create_ask_user_handler(data: Dict[str, Any], config: Dict[str, Any]):
        def handler(input_data):
            prompt = input_data.get("prompt", "Please respond:") if isinstance(input_data, dict) else str(input_data)
            choices = input_data.get("choices", []) if isinstance(input_data, dict) else []
            allow_free_text = config.get("allowFreeText", True)

            return {
                "response": f"[User prompt: {prompt}]",
                "prompt": prompt,
                "choices": choices,
                "allow_free_text": allow_free_text,
                "_pending_effect": {
                    "type": "ask_user",
                    "prompt": prompt,
                    "choices": choices,
                    "allow_free_text": allow_free_text,
                },
            }

        return handler

    def _create_answer_user_handler(data: Dict[str, Any], config: Dict[str, Any]):
        def handler(input_data):
            raw_message = input_data.get("message", "") if isinstance(input_data, dict) else input_data
            message = "" if raw_message is None else str(raw_message)
            raw_level = input_data.get("level") if isinstance(input_data, dict) else None
            level = str(raw_level).strip().lower() if isinstance(raw_level, str) else ""
            if level == "warn":
                level = "warning"
            if level == "info":
                level = "message"
            if level not in {"message", "warning", "error"}:
                level = "message"
            return {"message": message, "level": level, "_pending_effect": {"type": "answer_user", "message": message, "level": level}}

        return handler

    def _create_llm_call_handler(data: Dict[str, Any], config: Dict[str, Any]):
        provider_default = config.get("provider", "")
        model_default = config.get("model", "")
        temperature = config.get("temperature", 0.7)
        seed_default = config.get("seed", -1)
        thinking_default = config.get("thinking")
        tools_default_raw = config.get("tools")
        include_context_cfg = config.get("include_context")
        if include_context_cfg is None:
            include_context_cfg = config.get("use_context")
        include_context_default = _coerce_bool(include_context_cfg) if include_context_cfg is not None else False

        max_input_tokens_default = config.get("max_input_tokens")
        if max_input_tokens_default is None:
            max_input_tokens_default = config.get("maxInputTokens")

        max_output_tokens_default = config.get("max_output_tokens")
        if max_output_tokens_default is None:
            max_output_tokens_default = config.get("maxOutputTokens")

        output_default_marker = object()
        output_default = config.get("output", output_default_marker)
        if output_default is output_default_marker:
            output_default = config.get("outputs", output_default_marker)

        structured_output_fallback_cfg = config.get("structured_output_fallback")
        structured_output_fallback_default = (
            _coerce_bool(structured_output_fallback_cfg) if structured_output_fallback_cfg is not None else False
        )

        # Tool definitions (ToolSpecs) are required for tool calling. In the visual editor we
        # store tools as a portable `string[]` allowlist; at execution time we translate to
        # strict ToolSpecs `{name, description, parameters}` expected by AbstractCore.
        def _strip_tool_spec(raw: Any) -> Optional[Dict[str, Any]]:
            if not isinstance(raw, dict):
                return None
            name = raw.get("name")
            if not isinstance(name, str) or not name.strip():
                return None
            desc = raw.get("description")
            params = raw.get("parameters")
            out: Dict[str, Any] = {
                "name": name.strip(),
                "description": str(desc or ""),
                "parameters": dict(params) if isinstance(params, dict) else {},
            }
            return out

        def _normalize_tool_names(raw: Any) -> list[str]:
            if not isinstance(raw, list):
                return []
            out: list[str] = []
            for t in raw:
                if isinstance(t, str) and t.strip():
                    out.append(t.strip())
            return out

        # Precompute a best-effort "available ToolSpecs by name" map so we can turn tool names
        # into ToolSpecs without going through the web backend.
        tool_specs_by_name: Dict[str, Dict[str, Any]] = {}
        try:
            from abstractruntime.integrations.abstractcore.default_tools import list_default_tool_specs

            base_specs = list_default_tool_specs()
            if not isinstance(base_specs, list):
                base_specs = []
            for s in base_specs:
                stripped = _strip_tool_spec(s)
                if stripped is not None:
                    tool_specs_by_name[stripped["name"]] = stripped
        except Exception:
            pass

        # Optional schema-only runtime tools (used by AbstractAgent). These are useful for
        # "state machine" autonomy where the graph can route tool-like requests to effect nodes.
        try:
            from abstractagent.logic.builtins import (  # type: ignore
                ASK_USER_TOOL,
                COMPACT_MEMORY_TOOL,
                INSPECT_VARS_TOOL,
                RECALL_MEMORY_TOOL,
                REMEMBER_TOOL,
            )

            builtin_defs = [ASK_USER_TOOL, RECALL_MEMORY_TOOL, INSPECT_VARS_TOOL, REMEMBER_TOOL, COMPACT_MEMORY_TOOL]
            for tool_def in builtin_defs:
                try:
                    d = tool_def.to_dict()
                except Exception:
                    d = None
                stripped = _strip_tool_spec(d)
                if stripped is not None and stripped["name"] not in tool_specs_by_name:
                    tool_specs_by_name[stripped["name"]] = stripped
        except Exception:
            pass

        def _normalize_tools(raw: Any) -> list[Dict[str, Any]]:
            # Already ToolSpecs (from pins): accept and strip UI-only fields.
            if isinstance(raw, list) and raw and all(isinstance(x, dict) for x in raw):
                out: list[Dict[str, Any]] = []
                for x in raw:
                    stripped = _strip_tool_spec(x)
                    if stripped is not None:
                        out.append(stripped)
                return out

            # Tool names (portable representation): resolve against known tool specs.
            names = _normalize_tool_names(raw)
            out: list[Dict[str, Any]] = []
            for name in names:
                spec = tool_specs_by_name.get(name)
                if spec is not None:
                    out.append(spec)
            return out

        def _normalize_response_schema(raw: Any) -> Optional[Dict[str, Any]]:
            """Normalize a structured-output schema input into a JSON Schema dict.

            Supported inputs (best-effort):
            - JSON Schema dict: {"type":"object","properties":{...}, ...}
            - LMStudio/OpenAI-style wrapper: {"type":"json_schema","json_schema": {"schema": {...}}}
            """
            if raw is None:
                return None

            schema: Optional[Dict[str, Any]] = None
            if isinstance(raw, dict):
                # Wrapper form (OpenAI "response_format": {type:"json_schema", json_schema:{schema:{...}}})
                if raw.get("type") == "json_schema" and isinstance(raw.get("json_schema"), dict):
                    inner = raw.get("json_schema")
                    if isinstance(inner, dict) and isinstance(inner.get("schema"), dict):
                        schema = dict(inner.get("schema") or {})
                else:
                    # Plain JSON Schema dict
                    schema = dict(raw)

            if not isinstance(schema, dict) or not schema:
                return None

            # Resolve stable schema refs (no silent fallback).
            ref = schema.get("$ref")
            if isinstance(ref, str) and ref.strip().startswith("abstractsemantics:"):
                try:
                    from abstractsemantics import resolve_schema_ref  # type: ignore
                except Exception as e:
                    import sys

                    raise RuntimeError(
                        "Structured-output schema ref requires `abstractsemantics` to be installed in the same "
                        f"Python environment as the runtime/gateway (cannot resolve $ref={ref!r}). "
                        f"Current python: {sys.executable}. Install with: `pip install abstractsemantics`."
                    ) from e
                resolved = resolve_schema_ref(schema)
                if isinstance(resolved, dict) and resolved:
                    return resolved
                raise RuntimeError(f"Unknown structured-output schema ref: {ref}")

            return schema

        def _normalize_thinking(raw: Any) -> Any:
            if isinstance(raw, bool):
                return raw
            if isinstance(raw, str):
                clean = raw.strip()
                return clean or None
            return None

        def handler(input_data):
            if isinstance(input_data, dict):
                raw_prompt = input_data.get("prompt")
                prompt = "" if raw_prompt is None else str(raw_prompt)
            else:
                prompt = str(input_data)
            system = input_data.get("system", "") if isinstance(input_data, dict) else ""

            tools_specified = isinstance(input_data, dict) and "tools" in input_data
            tools_raw = input_data.get("tools") if isinstance(input_data, dict) else None
            tools = _normalize_tools(tools_raw) if tools_specified else []
            if not tools_specified:
                tools = _normalize_tools(tools_default_raw)

            include_context_specified = isinstance(input_data, dict) and (
                "include_context" in input_data or "use_context" in input_data
            )
            if include_context_specified:
                raw_inc = (
                    input_data.get("include_context")
                    if isinstance(input_data, dict) and "include_context" in input_data
                    else input_data.get("use_context") if isinstance(input_data, dict) else None
                )
                include_context_value = _coerce_bool(raw_inc)
            else:
                include_context_value = include_context_default

            max_input_tokens_value: Optional[int] = None
            raw_max_in: Any = None
            if isinstance(input_data, dict):
                if "max_in_tokens" in input_data:
                    raw_max_in = input_data.get("max_in_tokens")
                elif "max_input_tokens" in input_data:
                    raw_max_in = input_data.get("max_input_tokens")
                elif "maxInputTokens" in input_data:
                    raw_max_in = input_data.get("maxInputTokens")
            if raw_max_in is None:
                raw_max_in = max_input_tokens_default

            try:
                if raw_max_in is not None and not isinstance(raw_max_in, bool):
                    parsed = int(raw_max_in)
                    if parsed > 0:
                        max_input_tokens_value = parsed
            except Exception:
                max_input_tokens_value = None

            max_output_tokens_value: Optional[int] = None
            raw_max_out: Any = None
            if isinstance(input_data, dict):
                if "max_out_tokens" in input_data:
                    raw_max_out = input_data.get("max_out_tokens")
                elif "max_output_tokens" in input_data:
                    raw_max_out = input_data.get("max_output_tokens")
                elif "maxOutputTokens" in input_data:
                    raw_max_out = input_data.get("maxOutputTokens")
            if raw_max_out is None:
                raw_max_out = max_output_tokens_default
            try:
                if raw_max_out is not None and not isinstance(raw_max_out, bool):
                    parsed = int(raw_max_out)
                    if parsed > 0:
                        max_output_tokens_value = parsed
            except Exception:
                max_output_tokens_value = None

            provider = (
                input_data.get("provider")
                if isinstance(input_data, dict) and isinstance(input_data.get("provider"), str)
                else input_data.get("provider_text")
                if isinstance(input_data, dict) and isinstance(input_data.get("provider_text"), str)
                else provider_default
            )
            model = (
                input_data.get("model")
                if isinstance(input_data, dict) and isinstance(input_data.get("model"), str)
                else input_data.get("model_text")
                if isinstance(input_data, dict) and isinstance(input_data.get("model_text"), str)
                else model_default
            )

            # Allow pins to override sampling params.
            temperature_value = temperature
            if isinstance(input_data, dict) and "temperature" in input_data:
                raw_temp = input_data.get("temperature")
                try:
                    if raw_temp is not None and not isinstance(raw_temp, bool):
                        temperature_value = float(raw_temp)
                except Exception:
                    pass

            seed_value_raw: Any = seed_default
            if isinstance(input_data, dict) and "seed" in input_data:
                seed_value_raw = input_data.get("seed")
            seed_value = -1
            try:
                if seed_value_raw is not None and not isinstance(seed_value_raw, bool):
                    seed_value = int(seed_value_raw)
            except Exception:
                seed_value = -1

            thinking_raw: Any = thinking_default
            if isinstance(input_data, dict) and "thinking" in input_data:
                thinking_raw = input_data.get("thinking")
            thinking_value = _normalize_thinking(thinking_raw)

            params: Dict[str, Any] = {"temperature": float(temperature_value)}
            if seed_value >= 0:
                params["seed"] = seed_value
            if thinking_value is not None:
                params["thinking"] = thinking_value
            if isinstance(max_output_tokens_value, int) and max_output_tokens_value > 0:
                params["max_output_tokens"] = int(max_output_tokens_value)
            if isinstance(input_data, dict) and "prompt_cache_binding" in input_data:
                binding = input_data.get("prompt_cache_binding")
                if isinstance(binding, dict) and binding:
                    params["prompt_cache_binding"] = dict(binding)
                elif isinstance(binding, str) and binding.strip():
                    params["prompt_cache_binding"] = binding.strip()

            output_specified = False
            output_request: Any = None
            if isinstance(input_data, dict):
                if "output" in input_data:
                    output_request = input_data.get("output")
                    output_specified = True
                elif "outputs" in input_data:
                    output_request = input_data.get("outputs")
                    output_specified = True
            if not output_specified and output_default is not output_default_marker:
                output_request = output_default
                output_specified = True

            # Memory-source access pins (467): pass through for compiler/runtime pre-call scheduling.
            #
            # NOTE: we intentionally keep values JSON-safe and mostly unmodified here.
            # The compiler layer applies policy defaults/clamping and maps these into
            # MEMORY_* effects or LLM params as needed.
            mem_cfg: Dict[str, Any] = {}
            if isinstance(input_data, dict):
                # v0: memory object pin (preferred). Expand into legacy per-pin keys so
                # the compiler/runtime logic remains backward compatible.
                mem_obj = input_data.get("memory")
                if isinstance(mem_obj, dict):
                    for bool_key in (
                        "use_session_attachments",
                        "use_span_memory",
                        "use_semantic_search",
                        "use_kg_memory",
                    ):
                        if bool_key in input_data:
                            continue
                        if bool_key in mem_obj and mem_obj.get(bool_key) is not None:
                            mem_cfg[bool_key] = _coerce_bool(mem_obj.get(bool_key))

                    for raw_key in (
                        "memory_query",
                        "memory_scope",
                        "recall_level",
                        "max_span_messages",
                        "kg_max_input_tokens",
                        "kg_limit",
                        "kg_min_score",
                    ):
                        if raw_key in input_data:
                            continue
                        if raw_key in mem_obj and mem_obj.get(raw_key) is not None:
                            mem_cfg[raw_key] = mem_obj.get(raw_key)

                for bool_key in (
                    "use_session_attachments",
                    "use_span_memory",
                    "use_semantic_search",
                    "use_kg_memory",
                ):
                    if bool_key in input_data:
                        mem_cfg[bool_key] = _coerce_bool(input_data.get(bool_key))

                for raw_key in (
                    "memory_query",
                    "memory_scope",
                    "recall_level",
                    "max_span_messages",
                    "kg_max_input_tokens",
                    "kg_limit",
                    "kg_min_score",
                ):
                    if raw_key in input_data:
                        mem_cfg[raw_key] = input_data.get(raw_key)

            response_schema = None
            if isinstance(input_data, dict) and ("resp_schema" in input_data or "response_schema" in input_data):
                response_schema = _normalize_response_schema(
                    input_data.get("resp_schema")
                    if "resp_schema" in input_data
                    else input_data.get("response_schema")
                )

            def _attach_response_schema(pending_effect: Dict[str, Any]) -> None:
                if not isinstance(response_schema, dict) or not response_schema:
                    return
                pending_effect["response_schema"] = response_schema
                # Name is optional; AbstractRuntime will fall back to a safe default.
                pending_effect["response_schema_name"] = "LLM_StructuredOutput"
                if structured_output_fallback_default:
                    pending_effect["structured_output_fallback"] = True

            if not provider and not model:
                pending_auto: Dict[str, Any] = {
                    "type": "llm_call",
                    "prompt": prompt,
                    "system_prompt": system,
                    "tools": tools,
                    "params": dict(params),
                    "include_context": include_context_value,
                    **mem_cfg,
                }
                if output_specified:
                    pending_auto["output"] = output_request
                _attach_response_schema(pending_auto)
                return {
                    "response": "",
                    "_pending_effect": pending_auto,
                }

            if not provider or not model:
                pending_missing: Dict[str, Any] = {
                    "type": "llm_call",
                    "prompt": prompt,
                    "system_prompt": system,
                    "tools": tools,
                    "params": dict(params),
                    "include_context": include_context_value,
                    **mem_cfg,
                }
                if output_specified:
                    pending_missing["output"] = output_request
                _attach_response_schema(pending_missing)
                return {
                    "response": "[LLM Call: incomplete provider/model override]",
                    "_pending_effect": pending_missing,
                    "error": "Provider and model must both be set, or both left blank for Gateway/Core defaults",
                }

            pending: Dict[str, Any] = {
                "type": "llm_call",
                "prompt": prompt,
                "system_prompt": system,
                "tools": tools,
                "params": dict(params),
                "provider": provider,
                "model": model,
                "include_context": include_context_value,
                **mem_cfg,
            }
            if output_specified:
                pending["output"] = output_request
            if isinstance(max_input_tokens_value, int) and max_input_tokens_value > 0:
                pending["max_input_tokens"] = int(max_input_tokens_value)
            _attach_response_schema(pending)

            # Optional explicit context pin: if provided, context.messages overrides inherited run context messages.
            context_msgs: list[Dict[str, Any]] = []
            context_media_override: Optional[list[Any]] = None
            if isinstance(input_data, dict):
                context_raw = input_data.get("context")
                context_raw = context_raw if isinstance(context_raw, dict) else {}
                raw_msgs = context_raw.get("messages")
                if isinstance(raw_msgs, list):
                    context_msgs = [dict(m) for m in raw_msgs if isinstance(m, dict)]
                # Attachments are passed as `context.attachments` and mapped into `payload.media`
                # for the underlying LLM_CALL effect. This keeps the durable context object portable
                # while letting LLM execution consume media in a single, flat field.
                if "attachments" in context_raw:
                    raw_attachments = context_raw.get("attachments")
                    if isinstance(raw_attachments, list):
                        cleaned: list[Any] = []
                        for a in raw_attachments:
                            if isinstance(a, dict):
                                cleaned.append(dict(a))
                            elif isinstance(a, str) and a.strip():
                                cleaned.append(a.strip())
                        # Preserve explicit empty list (means "no attachments for this call").
                        context_media_override = cleaned
            if context_media_override is not None:
                pending["media"] = context_media_override
            if context_msgs:
                messages = list(context_msgs)
                sys_text = str(system or "").strip() if isinstance(system, str) else ""
                if sys_text:
                    insert_at = 0
                    while insert_at < len(messages):
                        if messages[insert_at].get("role") != "system":
                            break
                        insert_at += 1
                    messages.insert(insert_at, {"role": "system", "content": sys_text})
                messages.append({"role": "user", "content": str(prompt or "")})

                if isinstance(max_input_tokens_value, int) and max_input_tokens_value > 0:
                    try:
                        from abstractruntime.memory.token_budget import (
                            trim_messages_to_max_input_tokens,
                        )

                        messages = trim_messages_to_max_input_tokens(
                            messages,
                            max_input_tokens=int(max_input_tokens_value),
                            model=model if isinstance(model, str) else None,
                        )
                    except Exception:
                        pass

                pending["messages"] = messages
                pending.pop("prompt", None)
                pending.pop("system_prompt", None)

            return {
                "response": None,
                "_pending_effect": pending,
            }

        return handler

    def _create_model_catalog_handler(data: Dict[str, Any]):
        cfg = data.get("modelCatalogConfig", {}) if isinstance(data, dict) else {}
        cfg = dict(cfg) if isinstance(cfg, dict) else {}

        allowed_providers_default = cfg.get("allowedProviders")
        allowed_models_default = cfg.get("allowedModels")
        index_default = cfg.get("index", 0)

        def _as_str_list(raw: Any) -> list[str]:
            if not isinstance(raw, list):
                return []
            out: list[str] = []
            for x in raw:
                if isinstance(x, str) and x.strip():
                    out.append(x.strip())
            return out

        def handler(input_data: Any):
            # Allow pin-based overrides (data edges) while keeping node config as defaults.
            allowed_providers = _as_str_list(
                input_data.get("allowed_providers") if isinstance(input_data, dict) else None
            ) or _as_str_list(allowed_providers_default)
            allowed_models = _as_str_list(
                input_data.get("allowed_models") if isinstance(input_data, dict) else None
            ) or _as_str_list(allowed_models_default)

            idx_raw = input_data.get("index") if isinstance(input_data, dict) else None
            try:
                idx = int(idx_raw) if idx_raw is not None else int(index_default or 0)
            except Exception:
                idx = 0
            if idx < 0:
                idx = 0

            try:
                from abstractcore.providers.registry import get_all_providers_with_models, get_available_models_for_provider
            except Exception:
                return {"providers": [], "models": [], "pair": None, "provider": "", "model": ""}

            providers_meta = get_all_providers_with_models(include_models=False)
            available_providers: list[str] = []
            for p in providers_meta:
                if not isinstance(p, dict):
                    continue
                if p.get("status") != "available":
                    continue
                name = p.get("name")
                if isinstance(name, str) and name.strip():
                    available_providers.append(name.strip())

            if allowed_providers:
                allow = {x.lower(): x for x in allowed_providers}
                available_providers = [p for p in available_providers if p.lower() in allow]

            pairs: list[dict[str, str]] = []
            model_ids: list[str] = []

            allow_models_norm = {m.strip() for m in allowed_models if isinstance(m, str) and m.strip()}

            for provider in available_providers:
                try:
                    models = get_available_models_for_provider(provider)
                except Exception:
                    models = []
                if not isinstance(models, list):
                    models = []
                for m in models:
                    if not isinstance(m, str) or not m.strip():
                        continue
                    model = m.strip()
                    mid = f"{provider}/{model}"
                    if allow_models_norm:
                        # Accept either full ids or raw model names.
                        if mid not in allow_models_norm and model not in allow_models_norm:
                            continue
                    pairs.append({"provider": provider, "model": model, "id": mid})
                    model_ids.append(mid)

            selected = pairs[idx] if pairs and idx < len(pairs) else (pairs[0] if pairs else None)
            return {
                "providers": available_providers,
                "models": model_ids,
                "pair": selected,
                "provider": selected.get("provider", "") if isinstance(selected, dict) else "",
                "model": selected.get("model", "") if isinstance(selected, dict) else "",
            }

        return handler

    def _create_provider_catalog_handler(data: Dict[str, Any]):
        def _as_str_list(raw: Any) -> list[str]:
            if not isinstance(raw, list):
                return []
            out: list[str] = []
            for x in raw:
                if isinstance(x, str) and x.strip():
                    out.append(x.strip())
            return out

        def handler(input_data: Any):
            allowed_providers = _as_str_list(
                input_data.get("allowed_providers") if isinstance(input_data, dict) else None
            )

            try:
                from abstractcore.providers.registry import get_all_providers_with_models
            except Exception:
                return {"providers": []}

            providers_meta = get_all_providers_with_models(include_models=False)
            available: list[str] = []
            for p in providers_meta:
                if not isinstance(p, dict):
                    continue
                if p.get("status") != "available":
                    continue
                name = p.get("name")
                if isinstance(name, str) and name.strip():
                    available.append(name.strip())

            if allowed_providers:
                allow = {x.lower() for x in allowed_providers}
                available = [p for p in available if p.lower() in allow]

            return {"providers": available}

        return handler

    def _create_provider_models_handler(data: Dict[str, Any]):
        cfg = data.get("providerModelsConfig", {}) if isinstance(data, dict) else {}
        cfg = dict(cfg) if isinstance(cfg, dict) else {}

        def _as_str_list(raw: Any) -> list[str]:
            if not isinstance(raw, list):
                return []
            out: list[str] = []
            for x in raw:
                if isinstance(x, str) and x.strip():
                    out.append(x.strip())
            return out

        def _route_filter_from(raw: Any) -> Any:
            if raw is None:
                return None
            if isinstance(raw, str):
                clean = raw.strip()
                return clean or None
            if isinstance(raw, (list, tuple, set)):
                out = [str(x).strip() for x in raw if str(x).strip()]
                return out or None
            return None

        def handler(input_data: Any):
            provider = None
            if isinstance(input_data, dict) and isinstance(input_data.get("provider"), str):
                provider = input_data.get("provider")
            if not provider and isinstance(cfg.get("provider"), str):
                provider = cfg.get("provider")

            provider = str(provider or "").strip()
            if not provider:
                return {"provider": "", "models": []}

            allowed_models = _as_str_list(
                input_data.get("allowed_models") if isinstance(input_data, dict) else None
            )
            if not allowed_models:
                # Optional allowlist from node config when the pin isn't connected.
                allowed_models = _as_str_list(cfg.get("allowedModels")) or _as_str_list(cfg.get("allowed_models"))
            allow = {m for m in allowed_models if m}

            capability_routes = None
            if isinstance(input_data, dict):
                capability_routes = _route_filter_from(input_data.get("capability_route"))
                if capability_routes is None:
                    capability_routes = _route_filter_from(input_data.get("capability_routes"))
            if capability_routes is None:
                capability_routes = (
                    _route_filter_from(cfg.get("capabilityRoute"))
                    or _route_filter_from(cfg.get("capability_route"))
                    or _route_filter_from(cfg.get("capabilityRoutes"))
                    or _route_filter_from(cfg.get("capability_routes"))
                )

            try:
                from abstractcore.providers.registry import get_available_models_for_provider
            except Exception:
                return {"provider": provider, "models": []}

            try:
                models = get_available_models_for_provider(provider)
            except Exception:
                models = []
            if not isinstance(models, list):
                models = []

            if capability_routes:
                try:
                    from abstractcore.providers.model_capabilities import filter_models_by_capabilities

                    models = filter_models_by_capabilities(models, capability_routes=capability_routes)
                except Exception as e:
                    return {"provider": provider, "models": [], "error": f"Invalid capability_route filter: {e}"}

            out: list[str] = []
            for m in models:
                if not isinstance(m, str) or not m.strip():
                    continue
                name = m.strip()
                mid = f"{provider}/{name}"
                if allow and (name not in allow and mid not in allow):
                    continue
                out.append(name)

            return {"provider": provider, "models": out}

        return handler

    def _create_wait_until_handler(data: Dict[str, Any], config: Dict[str, Any]):
        from datetime import datetime as _dt, timedelta, timezone

        duration_type = config.get("durationType", "seconds")

        def handler(input_data):
            duration = input_data.get("duration", 0) if isinstance(input_data, dict) else 0

            try:
                amount = float(duration)
            except (TypeError, ValueError):
                amount = 0

            now = _dt.now(timezone.utc)
            if duration_type == "timestamp":
                until = str(duration or "")
            elif duration_type == "minutes":
                until = (now + timedelta(minutes=amount)).isoformat()
            elif duration_type == "hours":
                until = (now + timedelta(hours=amount)).isoformat()
            else:
                until = (now + timedelta(seconds=amount)).isoformat()

            return {"_pending_effect": {"type": "wait_until", "until": until}}

        return handler

    def _create_wait_event_handler(data: Dict[str, Any], config: Dict[str, Any]):
        def handler(input_data):
            # `wait_event` is a durable pause that waits for an external signal.
            #
            # Input shape (best-effort):
            # - event_key: str (required; defaults to "default" for backward-compat)
            # - prompt: str (optional; enables human-in-the-loop UX for EVENT waits)
            # - choices: list[str] (optional)
            # - allow_free_text: bool (optional; default True)
            #
            # NOTE: The compiler will wrap `_pending_effect` into an AbstractRuntime Effect payload.
            event_key = input_data.get("event_key", "default") if isinstance(input_data, dict) else str(input_data)
            prompt = None
            choices = None
            allow_free_text = True
            if isinstance(input_data, dict):
                p = input_data.get("prompt")
                if isinstance(p, str) and p.strip():
                    prompt = p
                ch = input_data.get("choices")
                if isinstance(ch, list):
                    # Keep choices JSON-safe and predictable.
                    choices = [str(c) for c in ch if isinstance(c, str) and str(c).strip()]
                aft = input_data.get("allow_free_text")
                if aft is None:
                    aft = input_data.get("allowFreeText")
                if aft is not None:
                    allow_free_text = bool(aft)
 
            pending: Dict[str, Any] = {"type": "wait_event", "wait_key": event_key}
            if prompt is not None:
                pending["prompt"] = prompt
            if isinstance(choices, list):
                pending["choices"] = choices
            # Always include allow_free_text so hosts can render consistent UX.
            pending["allow_free_text"] = allow_free_text
            return {
                "event_data": {},
                "event_key": event_key,
                "_pending_effect": pending,
            }

        return handler

    def _create_memory_note_handler(data: Dict[str, Any], config: Dict[str, Any]):
        def handler(input_data):
            content = input_data.get("content", "") if isinstance(input_data, dict) else str(input_data)
            tags = input_data.get("tags") if isinstance(input_data, dict) else None
            sources = input_data.get("sources") if isinstance(input_data, dict) else None
            location = input_data.get("location") if isinstance(input_data, dict) else None
            scope = input_data.get("scope") if isinstance(input_data, dict) else None

            pending: Dict[str, Any] = {"type": "memory_note", "note": content, "tags": tags if isinstance(tags, dict) else {}}
            if isinstance(sources, dict):
                pending["sources"] = sources
            if isinstance(location, str) and location.strip():
                pending["location"] = location.strip()
            if isinstance(scope, str) and scope.strip():
                pending["scope"] = scope.strip()

            keep_in_context_specified = isinstance(input_data, dict) and (
                "keep_in_context" in input_data or "keepInContext" in input_data
            )
            if keep_in_context_specified:
                raw_keep = (
                    input_data.get("keep_in_context")
                    if isinstance(input_data, dict) and "keep_in_context" in input_data
                    else input_data.get("keepInContext") if isinstance(input_data, dict) else None
                )
                keep_in_context = _coerce_bool(raw_keep)
            else:
                # Visual-editor config (checkbox) default.
                keep_cfg = None
                if isinstance(config, dict):
                    keep_cfg = config.get("keep_in_context")
                    if keep_cfg is None:
                        keep_cfg = config.get("keepInContext")
                keep_in_context = _coerce_bool(keep_cfg)
            if keep_in_context:
                pending["keep_in_context"] = True

            return {"note_id": None, "_pending_effect": pending}

        return handler

    def _create_memory_query_handler(data: Dict[str, Any], config: Dict[str, Any]):
        def handler(input_data):
            query = input_data.get("query", "") if isinstance(input_data, dict) else str(input_data)
            limit = input_data.get("limit", 10) if isinstance(input_data, dict) else 10
            recall_level = input_data.get("recall_level") if isinstance(input_data, dict) else None
            tags = input_data.get("tags") if isinstance(input_data, dict) else None
            tags_mode = input_data.get("tags_mode") if isinstance(input_data, dict) else None
            usernames = input_data.get("usernames") if isinstance(input_data, dict) else None
            locations = input_data.get("locations") if isinstance(input_data, dict) else None
            since = input_data.get("since") if isinstance(input_data, dict) else None
            until = input_data.get("until") if isinstance(input_data, dict) else None
            scope = input_data.get("scope") if isinstance(input_data, dict) else None
            try:
                limit_int = int(limit) if limit is not None else 10
            except Exception:
                limit_int = 10

            pending: Dict[str, Any] = {"type": "memory_query", "query": query, "limit_spans": limit_int, "return": "both"}
            if isinstance(recall_level, str) and recall_level.strip():
                pending["recall_level"] = recall_level.strip()
            if isinstance(tags, dict):
                pending["tags"] = tags
            if isinstance(tags_mode, str) and tags_mode.strip():
                pending["tags_mode"] = tags_mode.strip()
            if isinstance(usernames, list):
                pending["usernames"] = [str(x).strip() for x in usernames if isinstance(x, str) and str(x).strip()]
            if isinstance(locations, list):
                pending["locations"] = [str(x).strip() for x in locations if isinstance(x, str) and str(x).strip()]
            if since is not None:
                pending["since"] = since
            if until is not None:
                pending["until"] = until
            if isinstance(scope, str) and scope.strip():
                pending["scope"] = scope.strip()

            return {"results": [], "rendered": "", "_pending_effect": pending}

        return handler

    def _create_memory_kg_assert_handler(data: Dict[str, Any], config: Dict[str, Any]):
        def _normalize_assertions(raw: Any) -> list[Dict[str, Any]]:
            if raw is None:
                return []
            if isinstance(raw, dict):
                return [dict(raw)]
            if isinstance(raw, list):
                out: list[Dict[str, Any]] = []
                for x in raw:
                    if isinstance(x, dict):
                        out.append(dict(x))
                return out
            return []

        def handler(input_data):
            payload = input_data if isinstance(input_data, dict) else {}
            assertions_raw = payload.get("assertions")
            if assertions_raw is None:
                assertions_raw = payload.get("triples")
            if assertions_raw is None:
                assertions_raw = payload.get("items")

            assertions = _normalize_assertions(assertions_raw)

            pending: Dict[str, Any] = {"type": "memory_kg_assert", "assertions": assertions}
            scope = payload.get("scope")
            if isinstance(scope, str) and scope.strip():
                pending["scope"] = scope.strip()
            owner_id = payload.get("owner_id")
            if isinstance(owner_id, str) and owner_id.strip():
                pending["owner_id"] = owner_id.strip()
            span_id = payload.get("span_id")
            if isinstance(span_id, str) and span_id.strip():
                pending["span_id"] = span_id.strip()
            attributes_defaults = payload.get("attributes_defaults")
            if isinstance(attributes_defaults, dict) and attributes_defaults:
                pending["attributes_defaults"] = dict(attributes_defaults)
            allow_custom = payload.get("allow_custom_predicates")
            if isinstance(allow_custom, bool):
                pending["allow_custom_predicates"] = bool(allow_custom)

            return {"assertion_ids": [], "count": 0, "_pending_effect": pending}

        return handler

    def _create_memory_kg_query_handler(data: Dict[str, Any], config: Dict[str, Any]):
        def handler(input_data):
            payload = input_data if isinstance(input_data, dict) else {}
            pending: Dict[str, Any] = {"type": "memory_kg_query"}

            for k in (
                "subject",
                "predicate",
                "object",
                "recall_level",
                "scope",
                "owner_id",
                "since",
                "until",
                "active_at",
                "query_text",
                "order",
                # Optional packetization / packing controls (Active Memory mapping).
                "model",
            ):
                v = payload.get(k)
                if isinstance(v, str) and v.strip():
                    pending[k] = v.strip()

            max_input_tokens = payload.get("max_input_tokens")
            if max_input_tokens is None:
                max_input_tokens = payload.get("max_in_tokens")
            if max_input_tokens is not None and not isinstance(max_input_tokens, bool):
                try:
                    pending["max_input_tokens"] = int(float(max_input_tokens))
                except Exception:
                    pass

            min_score = payload.get("min_score")
            if min_score is not None and not isinstance(min_score, bool):
                try:
                    pending["min_score"] = float(min_score)
                except Exception:
                    pass

            limit = payload.get("limit")
            if limit is not None and not isinstance(limit, bool):
                try:
                    pending["limit"] = int(limit)
                except Exception:
                    pass

            return {"items": [], "count": 0, "_pending_effect": pending}

        return handler

    def _create_memory_kg_resolve_handler(data: Dict[str, Any], config: Dict[str, Any]):
        def handler(input_data):
            payload = input_data if isinstance(input_data, dict) else {}
            pending: Dict[str, Any] = {"type": "memory_kg_resolve"}

            for k in ("label", "expected_type", "recall_level", "scope", "owner_id"):
                v = payload.get(k)
                if isinstance(v, str) and v.strip():
                    pending[k] = v.strip()

            include_semantic = payload.get("include_semantic")
            if include_semantic is None:
                include_semantic = payload.get("includeSemantic")
            if isinstance(include_semantic, bool):
                pending["include_semantic"] = bool(include_semantic)

            min_score = payload.get("min_score")
            if min_score is None:
                min_score = payload.get("minScore")
            if min_score is not None and not isinstance(min_score, bool):
                try:
                    pending["min_score"] = float(min_score)
                except Exception:
                    pass

            max_candidates = payload.get("max_candidates")
            if max_candidates is None:
                max_candidates = payload.get("maxCandidates")
            if max_candidates is None:
                max_candidates = payload.get("limit")
            if max_candidates is not None and not isinstance(max_candidates, bool):
                try:
                    pending["max_candidates"] = int(float(max_candidates))
                except Exception:
                    pass

            return {"candidates": [], "count": 0, "_pending_effect": pending}

        return handler

    def _create_memory_tag_handler(data: Dict[str, Any], config: Dict[str, Any]):
        def handler(input_data):
            span_id = None
            tags: Dict[str, Any] = {}
            merge = None
            scope = None
            if isinstance(input_data, dict):
                span_id = input_data.get("span_id")
                if span_id is None:
                    span_id = input_data.get("spanId")
                raw_tags = input_data.get("tags")
                if isinstance(raw_tags, dict):
                    tags = raw_tags
                if "merge" in input_data:
                    merge = _coerce_bool(input_data.get("merge"))
                if isinstance(input_data.get("scope"), str):
                    scope = str(input_data.get("scope") or "").strip() or None

            pending: Dict[str, Any] = {"type": "memory_tag", "span_id": span_id, "tags": tags}
            if merge is not None:
                pending["merge"] = bool(merge)
            if scope is not None:
                pending["scope"] = scope
            return {"rendered": "", "success": False, "_pending_effect": pending}

        return handler

    def _create_memory_compact_handler(data: Dict[str, Any], config: Dict[str, Any]):
        def handler(input_data):
            preserve_recent = input_data.get("preserve_recent") if isinstance(input_data, dict) else None
            compression_mode = input_data.get("compression_mode") if isinstance(input_data, dict) else None
            focus = input_data.get("focus") if isinstance(input_data, dict) else None

            pending: Dict[str, Any] = {"type": "memory_compact"}
            if preserve_recent is not None:
                pending["preserve_recent"] = preserve_recent
            if isinstance(compression_mode, str) and compression_mode.strip():
                pending["compression_mode"] = compression_mode.strip()
            if isinstance(focus, str) and focus.strip():
                pending["focus"] = focus.strip()

            return {"span_id": None, "_pending_effect": pending}

        return handler

    def _create_memory_rehydrate_handler(data: Dict[str, Any], config: Dict[str, Any]):
        def handler(input_data):
            raw = input_data.get("span_ids") if isinstance(input_data, dict) else None
            if raw is None and isinstance(input_data, dict):
                raw = input_data.get("span_id")
            span_ids: list[Any] = []
            if isinstance(raw, list):
                span_ids = list(raw)
            elif raw is not None:
                span_ids = [raw]

            placement = input_data.get("placement") if isinstance(input_data, dict) else None
            placement_str = str(placement).strip() if isinstance(placement, str) else "after_summary"
            if placement_str not in {"after_summary", "after_system", "end"}:
                placement_str = "after_summary"

            max_messages = input_data.get("max_messages") if isinstance(input_data, dict) else None
            recall_level = input_data.get("recall_level") if isinstance(input_data, dict) else None

            pending: Dict[str, Any] = {"type": "memory_rehydrate", "span_ids": span_ids, "placement": placement_str}
            if max_messages is not None:
                pending["max_messages"] = max_messages
            if isinstance(recall_level, str) and recall_level.strip():
                pending["recall_level"] = recall_level.strip()
            return {"inserted": 0, "skipped": 0, "_pending_effect": pending}

        return handler

    def _create_handler(node_type: NodeType, data: Dict[str, Any]) -> Any:
        type_str = node_type.value if isinstance(node_type, NodeType) else str(node_type)

        if type_str == "get_var":
            return _create_get_var_handler(data)

        if type_str == "get_context":
            return _create_get_context_handler(data)

        if type_str == "bool_var":
            return _create_bool_var_handler(data)

        if type_str == "var_decl":
            return _create_var_decl_handler(data)

        if type_str == "set_var":
            return _create_set_var_handler(data)

        if type_str == "concat":
            return _create_concat_handler(data)

        if type_str == "make_array":
            return _create_make_array_handler(data)

        if type_str == "array_concat":
            return _create_array_concat_handler(data)

        if type_str == "read_file":
            return _create_read_file_handler(data)

        if type_str == "write_file":
            return _create_write_file_handler(data)

        if type_str == "read_pdf":
            return _create_read_pdf_handler(data)

        if type_str == "write_pdf":
            return _create_write_pdf_handler(data)

        if type_str == "list_folder_files":
            return _create_list_folder_files_handler(data)

        if type_str == "import_workspace_file":
            return _create_import_workspace_file_handler(data)

        if type_str == "export_artifact":
            return _create_export_artifact_handler(data)

        if type_str == "read_artifact":
            return _create_read_artifact_handler(data)

        # Sequence / Parallel are scheduler nodes compiled specially by `compile_flow`.
        # Their runtime semantics are handled in `abstractflow.adapters.control_adapter`.
        if type_str in ("sequence", "parallel"):
            return lambda x: x

        builtin = get_builtin_handler(type_str)
        if builtin:
            return _wrap_builtin(builtin, data)

        if type_str == "code":
            return _create_code_runtime_handler(data)

        if type_str == "agent":
            return _create_agent_input_handler(data)

        if type_str == "model_catalog":
            return _create_model_catalog_handler(data)

        if type_str == "provider_catalog":
            return _create_provider_catalog_handler(data)

        if type_str == "provider_models":
            return _create_provider_models_handler(data)

        if type_str == "subflow":
            return _create_subflow_effect_builder(data)

        if type_str == "join_exec":
            # Internal control node; runtime semantics are implemented in the compiler adapters.
            return lambda x: x

        if type_str == "break_object":
            return _create_break_object_handler(data)

        if type_str == "tool_parameters":
            return _create_tool_parameters_handler(data)

        if type_str == "function":
            if "code" in data:
                return create_code_handler(data["code"], data.get("functionName", "transform"))
            if "expression" in data:
                return _create_expression_handler(data["expression"])
            return lambda x: x

        if type_str == "on_flow_end":
            return _create_flow_end_handler(data)

        if type_str in ("on_flow_start", "on_user_request", "on_agent_message"):
            return _create_event_handler(type_str, data)

        if type_str == "if":
            return _create_if_handler(data)
        if type_str == "switch":
            return _create_switch_handler(data)
        if type_str == "while":
            return _create_while_handler(data)
        if type_str == "for":
            return _create_for_handler(data)
        if type_str == "loop":
            return _create_loop_handler(data)

        if type_str in EFFECT_NODE_TYPES:
            return _create_effect_handler(type_str, data)

        return lambda x: x

    for node in visual.nodes:
        type_str = node.type.value if hasattr(node.type, "value") else str(node.type)

        if type_str in LITERAL_NODE_TYPES:
            continue

        base_handler = _create_handler(node.type, node.data)

        if not _has_execution_pins(type_str, node.data):
            pure_base_handlers[node.id] = base_handler
            pure_node_ids.add(node.id)
            # Pure nodes are deterministic, but their upstream inputs can change on
            # recursive / multi-entry workflow iterations. Treat all pure nodes as
            # volatile so data-edge resolution recomputes from the latest upstream
            # node outputs instead of reusing a stale cached value from a prior pass.
            volatile_pure_node_ids.add(node.id)
            continue

        # Ignore disconnected/unreachable execution nodes.
        if reachable_exec and node.id not in reachable_exec:
            continue

        wrapped_handler = _create_data_aware_handler(
            node_id=node.id,
            node_type=type_str,
            base_handler=base_handler,
            data_edges=data_edge_map.get(node.id, {}),
            pin_defaults=pin_defaults_by_node_id.get(node.id),
            input_pin_ids=_input_pin_ids(node.data),
            node_outputs=flow._node_outputs,  # type: ignore[attr-defined]
            ensure_node_output=_ensure_node_output,
            volatile_node_ids=volatile_pure_node_ids,
        )

        input_key = node.data.get("inputKey")
        output_key = node.data.get("outputKey")

        effect_type: Optional[str] = None
        effect_config: Optional[Dict[str, Any]] = None
        if type_str in EFFECT_NODE_TYPES:
            effect_type = type_str
            effect_config = node.data.get("effectConfig", {})
        elif type_str == "on_schedule":
            # Schedule trigger: compiles into WAIT_UNTIL under the hood.
            effect_type = "on_schedule"
            effect_config = node.data.get("eventConfig", {})
        elif type_str == "on_event":
            # Custom event listener (Blueprint-style "Custom Event").
            # Compiles into WAIT_EVENT under the hood.
            effect_type = "on_event"
            effect_config = node.data.get("eventConfig", {})
        elif type_str == "agent":
            effect_type = "agent"
            raw_cfg = node.data.get("agentConfig", {})
            cfg = dict(raw_cfg) if isinstance(raw_cfg, dict) else {}
            cfg.setdefault(
                "_react_workflow_id",
                visual_react_workflow_id(flow_id=visual.id, node_id=node.id),
            )
            effect_config = cfg
        elif type_str in ("sequence", "parallel"):
            # Control-flow scheduler nodes. Store pin order so compilation can
            # execute branches deterministically (Blueprint-style).
            effect_type = type_str

            pins = node.data.get("outputs") if isinstance(node.data, dict) else None
            exec_ids: list[str] = []
            if isinstance(pins, list):
                for p in pins:
                    if not isinstance(p, dict):
                        continue
                    if p.get("type") != "execution":
                        continue
                    pid = p.get("id")
                    if isinstance(pid, str) and pid:
                        exec_ids.append(pid)

            def _then_key(h: str) -> int:
                try:
                    if h.startswith("then:"):
                        return int(h.split(":", 1)[1])
                except Exception:
                    pass
                return 10**9

            then_handles = sorted([h for h in exec_ids if h.startswith("then:")], key=_then_key)
            cfg = {"then_handles": then_handles}
            if type_str == "parallel":
                cfg["completed_handle"] = "completed"
            effect_config = cfg
        elif type_str == "join_exec":
            # Internal execution fan-in node.
            effect_type = "join_exec"
            internal = False
            try:
                internal = bool(node.data.get("_internal")) if isinstance(node.data, dict) else False
            except Exception:
                internal = False
            effect_config = {"_internal": True} if internal else {}
        elif type_str == "loop":
            # Control-flow scheduler node (Blueprint-style foreach).
            # Runtime semantics are handled in `abstractflow.adapters.control_adapter`.
            effect_type = type_str
            effect_config = {}
        elif type_str == "while":
            # Control-flow scheduler node (Blueprint-style while).
            # Runtime semantics are handled in `abstractflow.adapters.control_adapter`.
            effect_type = type_str
            effect_config = {}
        elif type_str == "for":
            # Control-flow scheduler node (Blueprint-style numeric for).
            # Runtime semantics are handled in `abstractflow.adapters.control_adapter`.
            effect_type = type_str
            effect_config = {}
        elif type_str == "subflow":
            effect_type = "start_subworkflow"
            subflow_id = node.data.get("subflowId") or node.data.get("flowId")
            output_pin_ids: list[str] = []
            outs = node.data.get("outputs")
            if isinstance(outs, list):
                for p in outs:
                    if not isinstance(p, dict):
                        continue
                    if p.get("type") == "execution":
                        continue
                    pid = p.get("id")
                    if isinstance(pid, str) and pid and pid != "output":
                        output_pin_ids.append(pid)
            effect_config = {"workflow_id": subflow_id, "output_pins": output_pin_ids}

        # Always attach minimal visual metadata for downstream compilation/wrapping.
        meta_cfg: Dict[str, Any] = {"_visual_type": type_str}
        if isinstance(effect_config, dict):
            meta_cfg.update(effect_config)
        effect_config = meta_cfg

        flow.add_node(
            node_id=node.id,
            handler=wrapped_handler,
            input_key=input_key,
            output_key=output_key,
            effect_type=effect_type,
            effect_config=effect_config,
        )

    for edge in visual.edges:
        if edge.targetHandle == "exec-in":
            if edge.source in flow.nodes and edge.target in flow.nodes:
                flow.add_edge(edge.source, edge.target, source_handle=edge.sourceHandle)

    if visual.entryNode and visual.entryNode in flow.nodes:
        flow.set_entry(visual.entryNode)
    else:
        targets = {e.target for e in visual.edges if e.targetHandle == "exec-in"}
        for node_id in flow.nodes:
            if node_id not in targets:
                flow.set_entry(node_id)
                break
        if not flow.entry_node and flow.nodes:
            flow.set_entry(next(iter(flow.nodes)))

    # Pure (no-exec) nodes are cached in `flow._node_outputs` for data-edge resolution.
    # Some schedulers (While, On Event, On Schedule) must invalidate these caches between iterations.
    flow._pure_node_ids = pure_node_ids  # type: ignore[attr-defined]

    return flow


def _sanitize_python_identifier(raw: Any) -> str:
    name = str(raw or "").strip()
    if not name:
        return "param"
    out = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not out:
        out = "param"
    if not re.match(r"^[A-Za-z_]", out):
        out = f"p_{out}"
    if keyword.iskeyword(out):
        out = f"{out}_"
    return out


def _generate_code_from_body(data: Dict[str, Any], function_name: str) -> str:
    body = textwrap.dedent(str(data.get("codeBody") or "")).strip()
    lines = [f"def {function_name}(_input):"]

    inputs = data.get("inputs")
    if isinstance(inputs, list):
        for pin in inputs:
            if not isinstance(pin, dict):
                continue
            pin_id = pin.get("id")
            if not isinstance(pin_id, str) or not pin_id:
                continue
            if pin_id == "permissions" or pin.get("type") == "execution":
                continue
            name = _sanitize_python_identifier(pin_id)
            lines.append(f"    {name} = _input.get({json.dumps(pin_id)})")

    if not body:
        lines.append("    return _input")
    else:
        for line in body.replace("\r\n", "\n").split("\n"):
            lines.append(f"    {line}")
    lines.append("")
    return "\n".join(lines)


def _create_code_runtime_handler(data: Dict[str, Any]):
    """Create a Code-node handler that honors the resolved permissions pin."""
    function_name = data.get("functionName", "transform")
    if not isinstance(function_name, str) or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", function_name) or keyword.iskeyword(function_name):
        function_name = "transform"
    if isinstance(data.get("codeBody"), str):
        code = _generate_code_from_body(data, function_name)
    else:
        code = data.get("code", "def transform(_input):\n    return _input")
    pin_defaults = data.get("pinDefaults") if isinstance(data.get("pinDefaults"), dict) else {}
    configured_permissions = pin_defaults.get("permissions") if isinstance(pin_defaults, dict) else None
    handlers: Dict[str, Any] = {}

    def handler(input_data: Any) -> Any:
        setattr(handler, "_last_permissions", None)
        permissions = configured_permissions or "sandbox"
        payload = input_data
        if isinstance(input_data, dict):
            raw_permissions = input_data.get("permissions")
            if raw_permissions is not None:
                permissions = raw_permissions
            payload = {k: v for k, v in input_data.items() if k != "permissions"}

        key = normalize_code_permissions(permissions)
        setattr(handler, "_last_permissions", key)
        ensure_code_permissions_allowed(key)
        if key not in handlers:
            handlers[key] = create_code_handler(code, function_name, permissions=permissions)
        return handlers[key](payload)

    return handler


def _create_data_aware_handler(
    node_id: str,
    node_type: str,
    base_handler,
    data_edges: Dict[str, tuple[str, str]],
    pin_defaults: Optional[Dict[str, Any]],
    input_pin_ids: Optional[set[str]],
    node_outputs: Dict[str, Dict[str, Any]],
    *,
    ensure_node_output=None,
    volatile_node_ids: Optional[set[str]] = None,
):
    """Wrap a handler to resolve data edge inputs before execution."""

    volatile: set[str] = volatile_node_ids if isinstance(volatile_node_ids, set) else set()
    declared_input_pin_ids: set[str] = input_pin_ids if isinstance(input_pin_ids, set) else set()

    def output_record_for_pins(
        result: Any,
        execution: Optional[Dict[str, Any]] = None,
        *,
        success: bool = True,
        error: Optional[str] = None,
    ) -> Any:
        if node_type != "code":
            return result
        record: Dict[str, Any] = {}
        if isinstance(result, dict):
            record.update(result)
        record["success"] = success
        record["output"] = result
        record["result"] = result
        if error:
            record["error"] = error
        if execution is not None:
            record["execution"] = execution
        return record

    def wrapped_handler(input_data):
        resolved_input: Dict[str, Any] = {}

        if isinstance(input_data, dict):
            resolved_input.update(input_data)

        for target_pin, (source_node, source_pin) in data_edges.items():
            if ensure_node_output is not None and (source_node not in node_outputs or source_node in volatile):
                ensure_node_output(source_node)
            if source_node in node_outputs:
                source_output = node_outputs[source_node]
                if isinstance(source_output, dict) and source_pin in source_output:
                    resolved_input[target_pin] = source_output[source_pin]
                elif source_pin in ("result", "output"):
                    resolved_input[target_pin] = source_output

        if pin_defaults:
            for pin_id, value in pin_defaults.items():
                # Connected pins always win (even if the upstream value is None).
                if pin_id in data_edges:
                    continue
                # Declared input-pin defaults are authored node configuration, so
                # they override same-named keys carried by the ambient exec payload.
                if pin_id not in declared_input_pin_ids and pin_id in resolved_input:
                    continue
                # Clone object/array defaults so handlers can't mutate the shared default.
                if isinstance(value, (dict, list)):
                    try:
                        import copy

                        resolved_input[pin_id] = copy.deepcopy(value)
                    except Exception:
                        resolved_input[pin_id] = value
                else:
                    resolved_input[pin_id] = value

        execution: Optional[Dict[str, Any]] = None
        if node_type == "code":
            execution_start = capture_execution_start()
            try:
                result = base_handler(resolved_input if resolved_input else input_data)
                execution = finish_execution_metrics(execution_start)
                permissions = getattr(base_handler, "_last_permissions", None)
                if isinstance(permissions, str) and permissions:
                    execution["permissions"] = permissions
            except Exception as exc:
                execution = finish_execution_metrics(execution_start)
                permissions = getattr(base_handler, "_last_permissions", None)
                if isinstance(permissions, str) and permissions:
                    execution["permissions"] = permissions
                failure_record = output_record_for_pins(None, execution, success=False, error=str(exc))
                node_outputs[node_id] = failure_record
                setattr(wrapped_handler, "_last_node_output", failure_record)
                raise
        else:
            result = base_handler(resolved_input if resolved_input else input_data)
        node_output = output_record_for_pins(result, execution)
        node_outputs[node_id] = node_output
        setattr(wrapped_handler, "_last_node_output", node_output)
        return result

    setattr(wrapped_handler, "_visual_node_type", node_type)
    return wrapped_handler
