"""VisualFlow lowering utilities (VisualFlow JSON → Flow IR).

This is a stdlib-only subset of AbstractFlow's visual executor, extracted into
AbstractRuntime so the VisualFlow compiler can run without importing AbstractFlow.

Scope:
- VisualFlow → Flow lowering (no runtime wiring / no web/editor host concerns)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..flow import Flow

from .agent_ids import visual_react_workflow_id
from .builtins import get_builtin_handler
from .code_executor import create_code_handler
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

        data_edges[edge.target][edge.targetHandle] = (edge.source, edge.sourceHandle)

    return data_edges


def visual_to_flow(visual: VisualFlow) -> Flow:
    """Convert a visual flow definition to an AbstractFlow `Flow`."""
    import datetime

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
        "llm_call",
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

    def _create_read_file_handler(_data: Dict[str, Any]):
        import json
        from pathlib import Path

        def handler(input_data: Any) -> Dict[str, Any]:
            payload = input_data if isinstance(input_data, dict) else {}
            raw_path = payload.get("file_path")
            if not isinstance(raw_path, str) or not raw_path.strip():
                raise ValueError("read_file requires a non-empty 'file_path' input.")

            file_path = raw_path.strip()
            path = Path(file_path).expanduser()
            if not path.is_absolute():
                path = Path.cwd() / path

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
                    return {"content": json.loads(text)}
                except Exception as e:
                    raise ValueError(f"Invalid JSON in '{file_path}': {e}") from e

            if looks_like_json:
                try:
                    return {"content": json.loads(text)}
                except Exception:
                    pass

            return {"content": text}

        return handler

    def _create_write_file_handler(_data: Dict[str, Any]):
        import json
        from pathlib import Path

        def handler(input_data: Any) -> Dict[str, Any]:
            payload = input_data if isinstance(input_data, dict) else {}
            raw_path = payload.get("file_path")
            if not isinstance(raw_path, str) or not raw_path.strip():
                raise ValueError("write_file requires a non-empty 'file_path' input.")

            file_path = raw_path.strip()
            path = Path(file_path).expanduser()
            if not path.is_absolute():
                path = Path.cwd() / path

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

            return {"bytes": len(text.encode("utf-8")), "file_path": str(path)}

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
                    raise RuntimeError(f"Structured-output schema ref requires abstractsemantics: {ref}") from e
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
        if effect_type == "llm_call":
            return _create_llm_call_handler(data, effect_config)
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
            message = input_data.get("message", "") if isinstance(input_data, dict) else str(input_data or "")
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
                    raise RuntimeError(f"Structured-output schema ref requires abstractsemantics: {ref}") from e
                resolved = resolve_schema_ref(schema)
                if isinstance(resolved, dict) and resolved:
                    return resolved
                raise RuntimeError(f"Unknown structured-output schema ref: {ref}")

            return schema

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
                else provider_default
            )
            model = (
                input_data.get("model")
                if isinstance(input_data, dict) and isinstance(input_data.get("model"), str)
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

            params: Dict[str, Any] = {"temperature": float(temperature_value)}
            if seed_value >= 0:
                params["seed"] = seed_value
            if isinstance(max_output_tokens_value, int) and max_output_tokens_value > 0:
                params["max_output_tokens"] = int(max_output_tokens_value)

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

            if not provider or not model:
                return {
                    "response": "[LLM Call: missing provider/model]",
                    "_pending_effect": {
                        "type": "llm_call",
                        "prompt": prompt,
                        "system_prompt": system,
                        "tools": tools,
                        "params": dict(params),
                        "include_context": include_context_value,
                        **mem_cfg,
                    },
                    "error": "Missing provider or model configuration",
                }

            response_schema = None
            if isinstance(input_data, dict) and ("resp_schema" in input_data or "response_schema" in input_data):
                response_schema = _normalize_response_schema(
                    input_data.get("resp_schema")
                    if "resp_schema" in input_data
                    else input_data.get("response_schema")
                )

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
            if isinstance(max_input_tokens_value, int) and max_input_tokens_value > 0:
                pending["max_input_tokens"] = int(max_input_tokens_value)
            if isinstance(response_schema, dict) and response_schema:
                pending["response_schema"] = response_schema
                # Name is optional; AbstractRuntime will fall back to a safe default.
                pending["response_schema_name"] = "LLM_StructuredOutput"
                if structured_output_fallback_default:
                    pending["structured_output_fallback"] = True

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

        # Sequence / Parallel are scheduler nodes compiled specially by `compile_flow`.
        # Their runtime semantics are handled in `abstractflow.adapters.control_adapter`.
        if type_str in ("sequence", "parallel"):
            return lambda x: x

        builtin = get_builtin_handler(type_str)
        if builtin:
            return _wrap_builtin(builtin, data)

        if type_str == "code":
            code = data.get("code", "def transform(input):\n    return input")
            function_name = data.get("functionName", "transform")
            return create_code_handler(code, function_name)

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
            if type_str in {"get_var", "get_context", "bool_var", "var_decl"}:
                volatile_pure_node_ids.add(node.id)
            continue

        # Ignore disconnected/unreachable execution nodes.
        if reachable_exec and node.id not in reachable_exec:
            continue

        wrapped_handler = _create_data_aware_handler(
            node_id=node.id,
            base_handler=base_handler,
            data_edges=data_edge_map.get(node.id, {}),
            pin_defaults=pin_defaults_by_node_id.get(node.id),
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


def _create_data_aware_handler(
    node_id: str,
    base_handler,
    data_edges: Dict[str, tuple[str, str]],
    pin_defaults: Optional[Dict[str, Any]],
    node_outputs: Dict[str, Dict[str, Any]],
    *,
    ensure_node_output=None,
    volatile_node_ids: Optional[set[str]] = None,
):
    """Wrap a handler to resolve data edge inputs before execution."""

    volatile: set[str] = volatile_node_ids if isinstance(volatile_node_ids, set) else set()

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
                if pin_id not in resolved_input:
                    # Clone object/array defaults so handlers can't mutate the shared default.
                    if isinstance(value, (dict, list)):
                        try:
                            import copy

                            resolved_input[pin_id] = copy.deepcopy(value)
                        except Exception:
                            resolved_input[pin_id] = value
                    else:
                        resolved_input[pin_id] = value

        result = base_handler(resolved_input if resolved_input else input_data)
        node_outputs[node_id] = result
        return result

    return wrapped_handler
