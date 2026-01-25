"""abstractruntime.integrations.abstractcore.effect_handlers

Effect handlers wiring for AbstractRuntime.

These handlers implement:
- `EffectType.LLM_CALL`
- `EffectType.TOOL_CALLS`

They are designed to keep `RunState.vars` JSON-safe.
"""

from __future__ import annotations

import json
import hashlib
import os
import mimetypes
import re
import tempfile
import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple, Type

from ...core.models import Effect, EffectType, RunState, RunStatus, WaitReason, WaitState
from ...core.runtime import EffectOutcome, EffectHandler
from ...storage.base import RunStore
from ...storage.artifacts import ArtifactStore, is_artifact_ref, get_artifact_id
from .llm_client import AbstractCoreLLMClient
from .tool_executor import ToolExecutor
from .logging import get_logger
from .session_attachments import (
    dedup_messages_view,
    execute_open_attachment,
    list_session_attachments,
    render_active_attachments_system_message,
    render_session_attachments_system_message,
    session_memory_owner_run_id,
)
from .workspace_scoped_tools import WorkspaceScope, rewrite_tool_arguments

logger = get_logger(__name__)

_JSON_SCHEMA_PRIMITIVE_TYPES: Set[str] = {"string", "integer", "number", "boolean", "array", "object", "null"}

_ABS_PATH_RE = re.compile(r"^[a-zA-Z]:[\\\\/]")


def _is_abs_path_like(path: str) -> bool:
    pth = str(path or "").strip()
    if not pth:
        return False
    if pth.startswith("/"):
        return True
    return bool(_ABS_PATH_RE.match(pth))


def _guess_ext_from_content_type(content_type: str) -> str:
    ct = str(content_type or "").strip().lower()
    if not ct:
        return ""
    ext = mimetypes.guess_extension(ct) or ""
    if ext == ".jpe":
        return ".jpg"
    return ext


def _safe_materialized_filename(*, desired: str, artifact_id: str, ext: str) -> str:
    """Return a filesystem-safe filename for a materialized artifact.

    Used for temp-file materialization only; avoids leaking absolute paths and keeps names
    conservative for cross-platform filesystem limits.
    """
    label = str(desired or "").replace("\\", "/").strip()
    if "/" in label or _is_abs_path_like(label):
        label = label.rsplit("/", 1)[-1]
    label = label.strip().strip("/")
    if not label:
        label = str(artifact_id or "").strip() or "attachment"
    if ext and not Path(label).suffix:
        label = f"{label}{ext}"

    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", label).strip("._") or "attachment"
    try:
        stem = Path(safe).stem
        suf = Path(safe).suffix
    except Exception:
        stem, suf = safe, ""
    short = str(artifact_id or "").strip()[:8]
    if short:
        safe = f"{stem}__{short}{suf}"

    max_len = 220
    if len(safe) > max_len:
        try:
            suf = Path(safe).suffix
        except Exception:
            suf = ""
        keep = max_len - len(suf)
        safe = safe[: max(1, keep)] + suf
    return safe


def _jsonable(value: Any) -> Any:
    """Best-effort conversion to JSON-safe objects.

    Runtime traces and effect outcomes are persisted in RunState.vars and must remain JSON-safe.
    """
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)


def _normalize_response_schema(raw: Any) -> Optional[Dict[str, Any]]:
    """Normalize user-provided schema inputs into a JSON Schema dict (object root).

    Supported inputs (best-effort):
    - JSON Schema object:
        {"type":"object","properties":{...}, "required":[...], ...}
      (also accepts missing type when properties exist)
    - OpenAI/LMStudio wrapper shapes:
        {"type":"json_schema","json_schema":{"schema":{...}}}
        {"json_schema":{"schema":{...}}}
        {"schema":{...}}  (inner wrapper copy/pasted from provider docs)
    - "Field map" shortcut (common authoring mistake but unambiguous intent):
        {"choice":"...", "score": 0, "meta": {"foo":"bar"}, ...}
      Coerces into a JSON Schema object with properties inferred from values.
    """

    def _is_schema_type(value: Any) -> bool:
        if isinstance(value, str):
            return value in _JSON_SCHEMA_PRIMITIVE_TYPES
        if isinstance(value, list) and value and all(isinstance(x, str) for x in value):
            return all(x in _JSON_SCHEMA_PRIMITIVE_TYPES for x in value)
        return False

    def _looks_like_json_schema(obj: Dict[str, Any]) -> bool:
        if "$schema" in obj or "$id" in obj or "$ref" in obj or "$defs" in obj or "definitions" in obj:
            return True
        if "oneOf" in obj or "anyOf" in obj or "allOf" in obj:
            return True
        if "enum" in obj or "const" in obj:
            return True
        if "items" in obj:
            return True
        if "required" in obj and isinstance(obj.get("required"), list):
            return True
        props = obj.get("properties")
        if isinstance(props, dict):
            return True
        if "type" in obj and _is_schema_type(obj.get("type")):
            return True
        return False

    def _unwrap_wrapper(obj: Dict[str, Any]) -> Dict[str, Any]:
        current = dict(obj)

        # If someone pasted an enclosing request object, tolerate common keys.
        for wrapper_key in ("response_format", "responseFormat"):
            inner = current.get(wrapper_key)
            if isinstance(inner, dict):
                current = dict(inner)

        # OpenAI/LMStudio wrapper: {type:"json_schema", json_schema:{schema:{...}}}
        if current.get("type") == "json_schema" and isinstance(current.get("json_schema"), dict):
            inner = current.get("json_schema")
            if isinstance(inner, dict) and isinstance(inner.get("schema"), dict):
                return dict(inner.get("schema") or {})

        # Slightly-less-wrapped: {json_schema:{schema:{...}}}
        if isinstance(current.get("json_schema"), dict):
            inner = current.get("json_schema")
            if isinstance(inner, dict) and isinstance(inner.get("schema"), dict):
                return dict(inner.get("schema") or {})

        # Inner wrapper copy/paste: {schema:{...}} or {name,strict,schema:{...}}
        if "schema" in current and isinstance(current.get("schema"), dict) and not _looks_like_json_schema(current):
            return dict(current.get("schema") or {})

        return current

    def _infer_schema_from_value(value: Any) -> Dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, bool):
            return {"type": "boolean"}
        if isinstance(value, int) and not isinstance(value, bool):
            return {"type": "integer"}
        if isinstance(value, float):
            return {"type": "number"}
        if isinstance(value, str):
            return {"type": "string", "description": value}
        if isinstance(value, list):
            # Prefer a simple, safe array schema; do not attempt enum constraints here.
            item_schema: Dict[str, Any] = {}
            for item in value:
                if item is None:
                    continue
                if isinstance(item, bool):
                    item_schema = {"type": "boolean"}
                    break
                if isinstance(item, int) and not isinstance(item, bool):
                    item_schema = {"type": "integer"}
                    break
                if isinstance(item, float):
                    item_schema = {"type": "number"}
                    break
                if isinstance(item, str):
                    item_schema = {"type": "string"}
                    break
                if isinstance(item, dict):
                    item_schema = _coerce_object_schema(item)
                    break
            out: Dict[str, Any] = {"type": "array"}
            if item_schema:
                out["items"] = item_schema
            return out
        if isinstance(value, dict):
            # If it already looks like a schema, keep it as-is (with minor fixes).
            if _looks_like_json_schema(value):
                return _coerce_object_schema(value)
            # Otherwise treat nested dict as another field-map object.
            return _coerce_object_schema(value)
        return {"type": "string", "description": str(value)}

    def _coerce_object_schema(obj: Dict[str, Any]) -> Dict[str, Any]:
        # If it's already a JSON schema, normalize the minimal invariants we need.
        if _looks_like_json_schema(obj):
            out = dict(obj)
            props = out.get("properties")
            if isinstance(props, dict) and out.get("type") is None:
                out["type"] = "object"
            # Nothing else to do here; deeper normalization is handled by the pydantic conversion.
            return out

        # Otherwise, interpret as "properties map" (field → schema/description/example).
        properties: Dict[str, Any] = {}
        required: list[str] = []
        for k, v in obj.items():
            if not isinstance(k, str) or not k.strip():
                continue
            key = k.strip()
            required.append(key)
            properties[key] = _infer_schema_from_value(v)

        schema: Dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            schema["required"] = required
        return schema

    if raw is None:
        return None

    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            raw = parsed
        except Exception:
            # Keep raw string; caller will treat as absent/invalid.
            return None

    if not isinstance(raw, dict) or not raw:
        return None

    candidate = _unwrap_wrapper(raw)
    if not isinstance(candidate, dict) or not candidate:
        return None

    normalized = _coerce_object_schema(candidate)
    return normalized if isinstance(normalized, dict) and normalized else None


def _pydantic_model_from_json_schema(schema: Dict[str, Any], *, name: str) -> Type[Any]:
    """Best-effort conversion from a JSON schema dict to a Pydantic model.

    This exists so structured output requests can remain JSON-safe in durable
    effect payloads (we persist the schema, not the Python class).
    """
    try:
        from pydantic import BaseModel, create_model
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Pydantic is required for structured outputs: {e}")

    from typing import Literal, Union

    NoneType = type(None)

    def _python_type(sub_schema: Any, *, nested_name: str) -> Any:
        if not isinstance(sub_schema, dict):
            return Any
        # Enums: represent as Literal[...] so Pydantic can enforce allowed values.
        enum_raw = sub_schema.get("enum")
        if isinstance(enum_raw, list) and enum_raw:
            try:
                return Literal.__getitem__(tuple(enum_raw))  # type: ignore[attr-defined]
            except Exception:
                return Any

        t = sub_schema.get("type")
        if isinstance(t, list) and t:
            # Union types (e.g. ["string","null"]).
            variants: list[Any] = []
            for tt in t:
                if tt == "null":
                    variants.append(NoneType)
                    continue
                if isinstance(tt, str) and tt:
                    variants.append(_python_type(dict(sub_schema, type=tt), nested_name=nested_name))
            # Drop Any from unions to avoid masking concrete variants.
            variants2 = [v for v in variants if v is not Any]
            variants = variants2 or variants
            if not variants:
                return Any
            if len(variants) == 1:
                return variants[0]
            try:
                return Union.__getitem__(tuple(variants))  # type: ignore[attr-defined]
            except Exception:
                return Any
        if t == "string":
            return str
        if t == "integer":
            return int
        if t == "number":
            return float
        if t == "boolean":
            return bool
        if t == "array":
            items = sub_schema.get("items")
            return list[_python_type(items, nested_name=f"{nested_name}Item")]  # type: ignore[index]
        if t == "object":
            props = sub_schema.get("properties")
            if isinstance(props, dict) and props:
                return _model(sub_schema, name=nested_name)
            return Dict[str, Any]
        return Any

    def _model(obj_schema: Dict[str, Any], *, name: str) -> Type[BaseModel]:
        schema_type = obj_schema.get("type")
        if schema_type is None and isinstance(obj_schema.get("properties"), dict):
            schema_type = "object"
        if isinstance(schema_type, list) and "object" in schema_type:
            schema_type = "object"
        if schema_type != "object":
            raise ValueError("response_schema must be a JSON schema object")
        props = obj_schema.get("properties")
        if not isinstance(props, dict) or not props:
            raise ValueError("response_schema must define properties")
        required_raw = obj_schema.get("required")
        required: Set[str] = set()
        if isinstance(required_raw, list):
            required = {str(x) for x in required_raw if isinstance(x, str)}

        fields: Dict[str, Tuple[Any, Any]] = {}
        for prop_name, prop_schema in props.items():
            if not isinstance(prop_name, str) or not prop_name.strip():
                continue
            # Keep things simple: only support identifier-like names to avoid aliasing issues.
            if not prop_name.isidentifier():
                raise ValueError(
                    f"Invalid property name '{prop_name}'. Use identifier-style names (letters, digits, underscore)."
                )
            t = _python_type(prop_schema, nested_name=f"{name}_{prop_name}")
            if prop_name in required:
                fields[prop_name] = (t, ...)
            else:
                fields[prop_name] = (Optional[t], None)

        return create_model(name, **fields)  # type: ignore[call-arg]

    return _model(schema, name=name)


def _trace_context(run: RunState) -> Dict[str, str]:
    ctx: Dict[str, str] = {
        "run_id": run.run_id,
        "workflow_id": str(run.workflow_id),
        "node_id": str(run.current_node),
    }
    if run.actor_id:
        ctx["actor_id"] = str(run.actor_id)
    session_id = getattr(run, "session_id", None)
    if session_id:
        ctx["session_id"] = str(session_id)
    if run.parent_run_id:
        ctx["parent_run_id"] = str(run.parent_run_id)
    return ctx


def _resolve_llm_call_media(
    media: Any,
    *,
    artifact_store: Optional[ArtifactStore],
    temp_dir: Optional[Path] = None,
) -> tuple[Optional[list[Any]], Optional[str]]:
    """Resolve a JSON-safe media list into inputs suitable for AbstractCore `generate(media=...)`.

    Supported media item shapes (best-effort):
    - str: treated as a local file path (passthrough)
    - {"$artifact": "...", ...}: ArtifactStore-backed attachment (materialized to a temp file)
    - {"artifact_id": "...", ...}: alternate artifact ref form (materialized)

    Returns:
        (resolved_media, error)
    """
    if media is None:
        return None, None
    if isinstance(media, tuple):
        media_items = list(media)
    else:
        media_items = media
    if not isinstance(media_items, list) or not media_items:
        return None, None

    def _artifact_id_from_item(item: Any) -> Optional[str]:
        if isinstance(item, dict):
            if is_artifact_ref(item):
                try:
                    aid = get_artifact_id(item)
                except Exception:
                    aid = None
                if isinstance(aid, str) and aid.strip():
                    return aid.strip()
            raw = item.get("artifact_id")
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
        return None

    out: list[Any] = []
    for item in media_items:
        if isinstance(item, str):
            path = item.strip()
            if path:
                out.append(path)
            continue

        artifact_id = _artifact_id_from_item(item)
        if artifact_id is None:
            return None, f"Unsupported media item (expected path or artifact ref): {type(item).__name__}"
        if artifact_store is None:
            return None, "Artifact-backed media requires an ArtifactStore (missing artifact_store)"
        if temp_dir is None:
            return None, "Internal error: temp_dir is required for artifact-backed media"

        artifact = artifact_store.load(str(artifact_id))
        if artifact is None:
            return None, f"Artifact '{artifact_id}' not found"

        content = getattr(artifact, "content", None)
        if not isinstance(content, (bytes, bytearray)):
            return None, f"Artifact '{artifact_id}' content is not bytes"

        # Preserve best-effort filename extension for downstream media detection and label
        # attachments with a safe, non-absolute path identifier.
        filename = ""
        source_path = ""
        if isinstance(item, dict):
            raw_source = item.get("source_path") or item.get("path")
            if isinstance(raw_source, str) and raw_source.strip():
                source_path = raw_source.strip()
            raw_name = source_path or item.get("filename") or item.get("name")
            if isinstance(raw_name, str) and raw_name.strip():
                filename = raw_name.strip()
        ext = Path(filename).suffix if filename else ""
        if not ext:
            ct = str(getattr(getattr(artifact, "metadata", None), "content_type", "") or "")
            ext = _guess_ext_from_content_type(ct)

        desired = source_path or filename or artifact_id
        safe_name = _safe_materialized_filename(desired=desired, artifact_id=str(artifact_id), ext=str(ext))
        p = temp_dir / safe_name
        try:
            p.write_bytes(bytes(content))
        except Exception as e:
            return None, f"Failed to materialize artifact '{artifact_id}': {e}"
        out.append(str(p))

    return (out or None), None


def _inline_active_text_attachments(
    *,
    messages: Any,
    media: Any,
    artifact_store: Optional[ArtifactStore],
    temp_dir: Optional[Path],
    max_inline_text_bytes: int,
) -> tuple[Any, Any]:
    """Inline small text-like artifact media into the last user message.

    Returns: (updated_messages, remaining_media)

    This is a derived view: it does not mutate durable run context.
    """
    if not isinstance(messages, list) or not messages:
        return messages, media
    if media is None:
        return messages, media
    media_items = list(media) if isinstance(media, (list, tuple)) else None
    if not media_items:
        return messages, media
    if artifact_store is None or temp_dir is None:
        return messages, media

    user_idx: Optional[int] = None
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if not isinstance(m, dict):
            continue
        if m.get("role") != "user":
            continue
        if isinstance(m.get("content"), str):
            user_idx = i
            break
    if user_idx is None:
        return messages, media

    base_text = str(messages[user_idx].get("content") or "")

    def _is_text_like_content_type(ct: str) -> bool:
        ct_low = str(ct or "").lower().strip()
        if not ct_low:
            return False
        if ct_low.startswith("text/"):
            return True
        return ct_low in {
            "application/json",
            "application/yaml",
            "application/x-yaml",
            "application/xml",
            "application/javascript",
            "application/typescript",
        }

    def _filename_for_item(item: Dict[str, Any], *, artifact_id: str) -> str:
        name = str(item.get("filename") or "").strip()
        if name:
            return name
        src = str(item.get("source_path") or item.get("path") or "").strip()
        if src:
            try:
                return Path(src).name or src.rsplit("/", 1)[-1]
            except Exception:
                return src.rsplit("/", 1)[-1]
        return artifact_id

    inline_blocks: list[str] = []
    remaining: list[Any] = []

    # Import lazily to avoid making media processing a hard dependency of the runtime kernel.
    try:
        from abstractcore.media.auto_handler import AutoMediaHandler  # type: ignore
    except Exception:
        AutoMediaHandler = None  # type: ignore[assignment]

    handler = None
    if AutoMediaHandler is not None:
        try:
            handler = AutoMediaHandler(enable_events=False)
        except Exception:
            handler = None

    for item in media_items:
        if not isinstance(item, dict):
            remaining.append(item)
            continue
        aid = item.get("$artifact") or item.get("artifact_id")
        if not isinstance(aid, str) or not aid.strip():
            remaining.append(item)
            continue
        artifact_id = aid.strip()

        meta = artifact_store.get_metadata(artifact_id)
        if meta is None:
            remaining.append(item)
            continue

        ct = str(item.get("content_type") or getattr(meta, "content_type", "") or "")
        if not _is_text_like_content_type(ct):
            remaining.append(item)
            continue
        try:
            size_bytes = int(getattr(meta, "size_bytes", 0) or 0)
        except Exception:
            remaining.append(item)
            continue
        if size_bytes > int(max_inline_text_bytes):
            remaining.append(item)
            continue

        artifact = artifact_store.load(artifact_id)
        if artifact is None:
            remaining.append(item)
            continue
        content = getattr(artifact, "content", None)
        if not isinstance(content, (bytes, bytearray)):
            remaining.append(item)
            continue

        # Materialize into temp_dir (required by AbstractCore media processors).
        name = _filename_for_item(item, artifact_id=artifact_id)
        ext = Path(name).suffix or _guess_ext_from_content_type(ct)
        p = temp_dir / _safe_materialized_filename(desired=name, artifact_id=artifact_id, ext=ext)
        try:
            p.write_bytes(bytes(content))
        except Exception:
            remaining.append(item)
            continue

        processed = ""
        if handler is not None:
            try:
                res = handler.process_file(p, max_inline_tabular_bytes=int(max_inline_text_bytes), format_output="structured")
                if getattr(res, "success", False) and getattr(res, "media_content", None) is not None:
                    processed = str(getattr(res.media_content, "content", "") or "")
            except Exception:
                processed = ""

        if not processed:
            try:
                processed = bytes(content).decode("utf-8")
            except Exception:
                remaining.append(item)
                continue

        label = _filename_for_item(item, artifact_id=artifact_id)
        inline_blocks.append(f"\n\n--- Content from {label} ---\n{processed}\n--- End of {label} ---")

    if not inline_blocks:
        return messages, media

    updated = list(messages)
    updated[user_idx] = dict(updated[user_idx], content=base_text + "".join(inline_blocks))
    return updated, (remaining or None)


def make_llm_call_handler(*, llm: AbstractCoreLLMClient, artifact_store: Optional[ArtifactStore] = None) -> EffectHandler:
    def _handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        payload = dict(effect.payload or {})
        prompt = payload.get("prompt")
        messages = payload.get("messages")
        system_prompt = payload.get("system_prompt")
        media = payload.get("media")
        provider = payload.get("provider")
        model = payload.get("model")
        tools_raw = payload.get("tools")
        tools = tools_raw if isinstance(tools_raw, list) and len(tools_raw) > 0 else None
        response_schema = _normalize_response_schema(payload.get("response_schema"))
        response_schema_name = payload.get("response_schema_name")
        structured_output_fallback = payload.get("structured_output_fallback")
        raw_params = payload.get("params")
        params = dict(raw_params) if isinstance(raw_params, dict) else {}

        # Propagate durable trace context into AbstractCore calls.
        trace_metadata = params.get("trace_metadata")
        if not isinstance(trace_metadata, dict):
            trace_metadata = {}
        trace_metadata.update(_trace_context(run))
        params["trace_metadata"] = trace_metadata

        # Support per-effect routing: allow the payload to override provider/model.
        # These reserved keys are consumed by MultiLocalAbstractCoreLLMClient and
        # ignored by LocalAbstractCoreLLMClient.
        if isinstance(provider, str) and provider.strip():
            params["_provider"] = provider.strip()
        if isinstance(model, str) and model.strip():
            params["_model"] = model.strip()

        def _nonempty_str(value: Any) -> Optional[str]:
            if not isinstance(value, str):
                return None
            text = value.strip()
            return text if text else None

        prompt = _nonempty_str(prompt)

        has_messages = isinstance(messages, list) and len(messages) > 0
        has_prompt = isinstance(prompt, str) and bool(prompt)
        if not has_prompt and not has_messages:
            return EffectOutcome.failed(
                "llm_call requires payload.prompt or payload.messages"
            )

        # Some agent loops (notably ReAct) require a strict "no in-loop truncation" policy for
        # correctness: every iteration must see the full scratchpad/history accumulated so far.
        # These runs can opt out of runtime-level input trimming via `_runtime.disable_input_trimming`.
        runtime_ns = run.vars.get("_runtime") if isinstance(run.vars, dict) else None
        disable_input_trimming = bool(runtime_ns.get("disable_input_trimming")) if isinstance(runtime_ns, dict) else False
        if isinstance(runtime_ns, dict):
            pending_media = runtime_ns.get("pending_media")
            if isinstance(pending_media, list) and pending_media:
                combined: list[Any] = []
                if isinstance(media, tuple):
                    combined.extend(list(media))
                elif isinstance(media, list):
                    combined.extend(list(media))
                combined.extend(pending_media)

                def _media_key(item: Any) -> Optional[Tuple[str, str]]:
                    if isinstance(item, str):
                        s = item.strip()
                        return ("path", s) if s else None
                    if isinstance(item, dict):
                        aid = item.get("$artifact") or item.get("artifact_id")
                        if isinstance(aid, str) and aid.strip():
                            return ("artifact", aid.strip())
                    return None

                merged: list[Any] = []
                seen: set[Tuple[str, str]] = set()
                for it in combined:
                    k = _media_key(it)
                    if k is None or k in seen:
                        continue
                    merged.append(dict(it) if isinstance(it, dict) else it)
                    seen.add(k)

                media = merged
            if isinstance(runtime_ns.get("pending_media"), list):
                runtime_ns["pending_media"] = []

        # Enforce a per-call (or per-run) input-token budget by trimming oldest non-system messages.
        #
        # This is separate from provider limits: it protects reasoning quality and latency by keeping
        # the active context window bounded even when the model supports very large contexts.
        max_input_tokens: Optional[int] = None
        try:
            raw_max_in = payload.get("max_input_tokens")
            if raw_max_in is None:
                limits = run.vars.get("_limits") if isinstance(run.vars, dict) else None
                raw_max_in = limits.get("max_input_tokens") if isinstance(limits, dict) else None
            if raw_max_in is not None and not isinstance(raw_max_in, bool):
                parsed = int(raw_max_in)
                if parsed > 0:
                    max_input_tokens = parsed
        except Exception:
            max_input_tokens = None

        if not disable_input_trimming and isinstance(max_input_tokens, int) and max_input_tokens > 0 and isinstance(messages, list) and messages:
            try:
                from abstractruntime.memory.token_budget import trim_messages_to_max_input_tokens

                model_name = model if isinstance(model, str) and model.strip() else None
                messages = trim_messages_to_max_input_tokens(messages, max_input_tokens=int(max_input_tokens), model=model_name)
            except Exception:
                # Never fail an LLM call due to trimming.
                pass

        # Enforce output token budgets (max_output_tokens) when configured.
        #
        # Priority:
        # 1) explicit params (payload.params.max_output_tokens / max_tokens)
        # 2) explicit payload field (payload.max_output_tokens / max_out_tokens)
        # 3) run-level default limits (run.vars._limits.max_output_tokens)
        max_output_tokens: Optional[int] = None
        try:
            raw_max_out = None
            if "max_output_tokens" in params:
                raw_max_out = params.get("max_output_tokens")
            elif "max_tokens" in params:
                raw_max_out = params.get("max_tokens")
            if raw_max_out is None:
                raw_max_out = payload.get("max_output_tokens")
                if raw_max_out is None:
                    raw_max_out = payload.get("max_out_tokens")
            if raw_max_out is None:
                limits = run.vars.get("_limits") if isinstance(run.vars, dict) else None
                raw_max_out = limits.get("max_output_tokens") if isinstance(limits, dict) else None
            if raw_max_out is not None and not isinstance(raw_max_out, bool):
                parsed = int(raw_max_out)
                if parsed > 0:
                    max_output_tokens = parsed
        except Exception:
            max_output_tokens = None

        if (
            isinstance(max_output_tokens, int)
            and max_output_tokens > 0
            and "max_output_tokens" not in params
            and "max_tokens" not in params
        ):
            params["max_output_tokens"] = int(max_output_tokens)

        def _coerce_boolish(value: Any) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return value != 0
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "y", "on"}
            return False

        # Optional attachment registry injections (active + session index).
        #
        # These are derived views: they do not mutate the durable run context.
        session_attachments: Optional[list[Dict[str, Any]]] = None
        try:
            include_raw = params.get("include_session_attachments_index") if isinstance(params, dict) else None
            if include_raw is None:
                # Run-level override (467):
                # Allow hosts/workflows to control attachment index injection without having to
                # touch every individual LLM_CALL payload (especially inside Agent subworkflows).
                runtime_ns = run.vars.get("_runtime") if isinstance(run.vars, dict) else None
                control = runtime_ns.get("control") if isinstance(runtime_ns, dict) else None
                override = control.get("include_session_attachments_index") if isinstance(control, dict) else None
                if override is not None:
                    include_raw = override

            if include_raw is None:
                # Default heuristic:
                # - Agents (tools present): include.
                # - Raw LLM calls: include only when explicitly using context.
                #
                # Exception: when active attachments are present for this call, skip the stored
                # session attachment index by default (it is redundant and encourages re-opening).
                has_active_media = bool(list(media)) if isinstance(media, (list, tuple)) else bool(media)
                if has_active_media:
                    include_raw = False
                else:
                    inc_ctx = payload.get("include_context")
                    if inc_ctx is None:
                        inc_ctx = payload.get("use_context")
                    include_raw = True if tools is not None else _coerce_boolish(inc_ctx)

            include_index = _coerce_boolish(include_raw)
            sid = getattr(run, "session_id", None)
            sid_str = str(sid or "").strip() if isinstance(sid, str) or sid is not None else ""

            active_msg = ""
            try:
                active_msg = render_active_attachments_system_message(media, max_entries=12, max_chars=2000)
            except Exception:
                active_msg = ""

            session_msg = ""
            if include_index and artifact_store is not None and sid_str:
                session_attachments = list_session_attachments(
                    artifact_store=artifact_store, session_id=sid_str, limit=20
                )
                has_open_attachment_tool = any(
                    isinstance(t, dict) and str(t.get("name") or "").strip() == "open_attachment" for t in (tools or [])
                )
                session_msg = render_session_attachments_system_message(
                    session_attachments,
                    max_entries=20,
                    max_chars=4000,
                    include_open_attachment_hint=has_open_attachment_tool,
                )

            if active_msg or session_msg:
                if not isinstance(messages, list):
                    messages = []
                # Remove any previous injected attachment system messages to avoid staleness/duplication.
                cleaned: list[Dict[str, Any]] = []
                for m in messages:
                    if not isinstance(m, dict):
                        continue
                    if m.get("role") != "system":
                        cleaned.append(m)
                        continue
                    c = m.get("content")
                    c_str = str(c or "")
                    if c_str.strip().startswith("Active attachments") or c_str.strip().startswith("Stored session attachments") or c_str.strip().startswith("Session attachments"):
                        continue
                    cleaned.append(m)

                injected: list[Dict[str, Any]] = []
                if active_msg:
                    injected.append({"role": "system", "content": active_msg})
                if session_msg:
                    injected.append({"role": "system", "content": session_msg})
                messages = injected + cleaned
        except Exception:
            session_attachments = None

        fallback_enabled = _coerce_boolish(structured_output_fallback)
        base_params = dict(params)

        try:
            # View-time dedup of repeated document reads (keeps LLM-visible context lean).
            if isinstance(messages, list) and messages:
                messages = dedup_messages_view(list(messages), session_attachments=session_attachments)

            def _extract_user_text_for_context(*, prompt_value: Any, messages_value: Any) -> str:
                if isinstance(prompt_value, str) and prompt_value.strip():
                    return prompt_value.strip()
                if isinstance(messages_value, list):
                    for m in reversed(messages_value):
                        if not isinstance(m, dict):
                            continue
                        if m.get("role") != "user":
                            continue
                        c = m.get("content")
                        if isinstance(c, str) and c.strip():
                            return c.strip()
                return ""

            # Preserve the "real" user text for `/use_context` without including inlined attachment blocks.
            user_text_for_context = _extract_user_text_for_context(prompt_value=prompt, messages_value=messages)

            structured_requested = isinstance(response_schema, dict) and response_schema
            params_for_call = dict(params)
            if structured_requested:
                model_name = (
                    str(response_schema_name).strip()
                    if isinstance(response_schema_name, str) and response_schema_name.strip()
                    else "StructuredOutput"
                )
                params_for_call["response_model"] = _pydantic_model_from_json_schema(response_schema, name=model_name)

            structured_failed = False
            structured_error: Optional[str] = None

            messages_for_call = messages
            media_for_call = media

            resolved_media: Optional[list[Any]] = None
            tmpdir: Optional[tempfile.TemporaryDirectory] = None
            if media_for_call is not None:
                tmpdir = tempfile.TemporaryDirectory(prefix="abstractruntime_media_")
                try:
                    max_inline_text_bytes = 120_000
                    raw_max_inline = params.get("max_inline_attachment_bytes")
                    if raw_max_inline is None:
                        raw_max_inline = params.get("max_inline_text_attachment_bytes")
                    if raw_max_inline is not None and not isinstance(raw_max_inline, bool):
                        try:
                            max_inline_text_bytes = max(0, int(raw_max_inline))
                        except Exception:
                            max_inline_text_bytes = 120_000

                    messages_for_call, media_for_call = _inline_active_text_attachments(
                        messages=messages_for_call,
                        media=media_for_call,
                        artifact_store=artifact_store,
                        temp_dir=Path(tmpdir.name),
                        max_inline_text_bytes=max_inline_text_bytes,
                    )
                    resolved_media, err = _resolve_llm_call_media(
                        media_for_call,
                        artifact_store=artifact_store,
                        temp_dir=Path(tmpdir.name),
                    )
                    if err:
                        tmpdir.cleanup()
                        return EffectOutcome.failed(err)
                except Exception as e:
                    tmpdir.cleanup()
                    return EffectOutcome.failed(str(e))

            # Framework default: Glyph compression is experimental and opt-in.
            #
            # Avoid noisy warnings and unnecessary decision overhead for non-vision models unless
            # the caller explicitly requests compression via `params.glyph_compression`.
            if isinstance(media_for_call, list) and media_for_call and "glyph_compression" not in params_for_call:
                params_for_call["glyph_compression"] = "never"

            runtime_observability = {
                "llm_generate_kwargs": _jsonable(
                    {
                        "prompt": str(prompt or ""),
                        "messages": messages_for_call,
                        "system_prompt": system_prompt,
                        "media": media_for_call,
                        "tools": tools,
                        "params": params_for_call,
                        "structured_output_fallback": fallback_enabled,
                    }
                ),
            }
            truncation_attempts: list[dict[str, Any]] = []
            had_truncation = False

            def _finish_reason_is_truncation(value: Any) -> bool:
                if not isinstance(value, str):
                    return False
                return value.strip().lower() in {"length", "max_tokens", "max_output_tokens"}

            def _bump_max_output_tokens(current: dict[str, Any]) -> dict[str, Any]:
                updated = dict(current)
                raw = updated.get("max_output_tokens")
                if raw is None:
                    raw = updated.get("max_tokens")
                cur = 0
                if raw is not None and not isinstance(raw, bool):
                    try:
                        cur = int(raw)
                    except Exception:
                        cur = 0
                if cur <= 0:
                    # If the caller didn't specify an output budget, assume we're at least at the
                    # runtime/provider default (often ~2k). Use the run limits as a better hint.
                    hinted = None
                    try:
                        limits = run.vars.get("_limits") if isinstance(run.vars, dict) else None
                        hinted = limits.get("max_output_tokens") if isinstance(limits, dict) else None
                    except Exception:
                        hinted = None
                    if hinted is not None and not isinstance(hinted, bool):
                        try:
                            hinted_i = int(hinted)
                        except Exception:
                            hinted_i = 0
                        if hinted_i > 0:
                            cur = hinted_i
                        else:
                            cur = 2048
                    else:
                        cur = 2048

                bumped = max(cur * 2, cur + 500)
                cap_raw = payload.get("max_output_tokens_cap")
                if cap_raw is None:
                    cap_raw = payload.get("max_truncation_max_output_tokens")
                cap = 8192
                if cap_raw is not None and not isinstance(cap_raw, bool):
                    try:
                        cap = max(256, int(cap_raw))
                    except Exception:
                        cap = 8192
                updated["max_output_tokens"] = min(bumped, cap)
                updated.pop("max_tokens", None)
                return updated

            retry_on_truncation_raw = payload.get("retry_on_truncation")
            if retry_on_truncation_raw is None:
                retry_on_truncation_raw = payload.get("no_truncation")
            retry_on_truncation = True
            if retry_on_truncation_raw is not None:
                retry_on_truncation = _coerce_boolish(retry_on_truncation_raw)

            allow_truncation_raw = payload.get("allow_truncation")
            if allow_truncation_raw is None:
                allow_truncation_raw = payload.get("allow_truncated")
            allow_truncation = _coerce_boolish(allow_truncation_raw) if allow_truncation_raw is not None else False

            max_truncation_attempts = 3
            raw_attempts = payload.get("max_truncation_attempts")
            if raw_attempts is None:
                raw_attempts = payload.get("truncation_max_attempts")
            if raw_attempts is not None and not isinstance(raw_attempts, bool):
                try:
                    max_truncation_attempts = max(1, int(raw_attempts))
                except Exception:
                    max_truncation_attempts = 3

            params_attempt = dict(params_for_call)
            base_params_attempt = dict(base_params)

            try:
                last_finish_reason: Optional[str] = None
                for attempt in range(1, max_truncation_attempts + 1):
                    try:
                        result = llm.generate(
                            prompt=str(prompt or ""),
                            messages=messages_for_call,
                            system_prompt=system_prompt,
                            media=resolved_media,
                            tools=tools,
                            params=params_attempt,
                        )
                    except Exception as e:
                        looks_like_validation = False
                        try:
                            from pydantic import ValidationError as PydanticValidationError  # type: ignore

                            looks_like_validation = isinstance(e, PydanticValidationError)
                        except Exception:
                            looks_like_validation = False

                        msg = str(e)
                        if not looks_like_validation:
                            lowered = msg.lower()
                            if "validation errors for" in lowered or "structured output generation failed" in lowered:
                                looks_like_validation = True

                        if not (fallback_enabled and structured_requested and looks_like_validation):
                            raise

                        logger.warning(
                            "LLM_CALL structured output failed; retrying without schema",
                            error=msg,
                        )

                        result = llm.generate(
                            prompt=str(prompt or ""),
                            messages=messages_for_call,
                            system_prompt=system_prompt,
                            media=resolved_media,
                            tools=tools,
                            params=base_params_attempt,
                        )
                        structured_failed = True
                        structured_error = msg

                    finish_reason = None
                    if isinstance(result, dict):
                        fr = result.get("finish_reason")
                        finish_reason = fr if isinstance(fr, str) else None
                    last_finish_reason = finish_reason

                    truncation_attempts.append(
                        {
                            "attempt": attempt,
                            "finish_reason": finish_reason,
                            "max_output_tokens": params_attempt.get("max_output_tokens"),
                            "structured_fallback": bool(structured_failed),
                        }
                    )

                    if not _finish_reason_is_truncation(finish_reason):
                        break
                    had_truncation = True

                    if allow_truncation:
                        break

                    if not retry_on_truncation or attempt >= max_truncation_attempts:
                        break

                    params_attempt = _bump_max_output_tokens(params_attempt)
                    base_params_attempt = _bump_max_output_tokens(base_params_attempt)

                if _finish_reason_is_truncation(last_finish_reason) and not allow_truncation:
                    budgets = ", ".join(
                        [
                            str(a.get("max_output_tokens"))
                            for a in truncation_attempts
                            if a.get("max_output_tokens") is not None
                        ][:6]
                    )
                    suffix = " …" if len(truncation_attempts) > 6 else ""
                    raise RuntimeError(
                        "LLM_CALL output was truncated (finish_reason=length). "
                        f"Attempted max_output_tokens: {budgets}{suffix}. "
                        "Increase max_output_tokens/max_out_tokens (or set allow_truncation=true)."
                    )

                # Keep observability aligned with the actual params used.
                if had_truncation or len(truncation_attempts) > 1:
                    runtime_observability["llm_generate_kwargs"] = _jsonable(
                        {
                            "prompt": str(prompt or ""),
                            "messages": messages_for_call,
                            "system_prompt": system_prompt,
                            "media": media_for_call,
                            "tools": tools,
                            "params": params_attempt,
                            "structured_output_fallback": fallback_enabled,
                            "truncation_attempts": truncation_attempts,
                        }
                    )
            finally:
                if tmpdir is not None:
                    tmpdir.cleanup()

            if structured_requested and isinstance(result, dict):
                # Best-effort: when structured outputs fail (or providers ignore response_model),
                # try to parse the returned text into `data` so downstream nodes can consume it.
                try:
                    existing_data = result.get("data")
                except Exception:
                    existing_data = None

                content_value = result.get("content") if isinstance(result.get("content"), str) else None

                if existing_data is None and isinstance(content_value, str) and content_value.strip():
                    parsed: Any = None
                    parse_error: Optional[str] = None
                    try:
                        from abstractruntime.visualflow_compiler.visual.builtins import data_parse_json

                        parsed = data_parse_json({"text": content_value, "wrap_scalar": True})
                    except Exception as e:
                        parse_error = str(e)
                        parsed = None

                    # Response schemas in this system are object-only; wrap non-dicts for safety.
                    if parsed is not None and not isinstance(parsed, dict):
                        parsed = {"value": parsed}
                    if parsed is not None:
                        result["data"] = parsed

                    if parse_error is not None:
                        meta = result.get("metadata")
                        if not isinstance(meta, dict):
                            meta = {}
                            result["metadata"] = meta
                        meta["_structured_output_parse_error"] = parse_error

            if isinstance(result, dict):
                meta = result.get("metadata")
                if not isinstance(meta, dict):
                    meta = {}
                    result["metadata"] = meta
                if structured_failed:
                    meta["_structured_output_fallback"] = {"used": True, "error": structured_error or ""}
                if had_truncation or len(truncation_attempts) > 1:
                    meta["_truncation"] = {
                        "attempts": truncation_attempts,
                        "resolved": not _finish_reason_is_truncation(result.get("finish_reason") if isinstance(result.get("finish_reason"), str) else None),
                    }
                existing = meta.get("_runtime_observability")
                if not isinstance(existing, dict):
                    existing = {}
                    meta["_runtime_observability"] = existing
                existing.update(runtime_observability)

            # VisualFlow "Use context" UX: when requested, persist the turn into the run's
            # active context (`vars.context.messages`) so subsequent LLM/Agent/Subflow nodes
            # can see the interaction history without extra wiring.
            #
            # IMPORTANT: This is opt-in via payload.include_context/use_context; AbstractRuntime
            # does not implicitly store all LLM calls in context.
            try:
                inc_raw = payload.get("include_context")
                if inc_raw is None:
                    inc_raw = payload.get("use_context")
                if _coerce_boolish(inc_raw):
                    from abstractruntime.core.vars import get_context

                    ctx_ns = get_context(run.vars)
                    msgs_any = ctx_ns.get("messages")
                    if not isinstance(msgs_any, list):
                        msgs_any = []
                        ctx_ns["messages"] = msgs_any

                    def _extract_assistant_text() -> str:
                        if isinstance(result, dict):
                            c = result.get("content")
                            if isinstance(c, str) and c.strip():
                                return c.strip()
                            d = result.get("data")
                            if isinstance(d, (dict, list)):
                                import json as _json

                                return _json.dumps(d, ensure_ascii=False, indent=2)
                        return ""

                    user_text = user_text_for_context
                    assistant_text = _extract_assistant_text()
                    node_id = str(getattr(run, "current_node", None) or "").strip() or "unknown"

                    if user_text:
                        msgs_any.append(
                            {
                                "role": "user",
                                "content": user_text,
                                "metadata": {"kind": "llm_turn", "node_id": node_id},
                            }
                        )
                    if assistant_text:
                        msgs_any.append(
                            {
                                "role": "assistant",
                                "content": assistant_text,
                                "metadata": {"kind": "llm_turn", "node_id": node_id},
                            }
                        )
                    if isinstance(getattr(run, "output", None), dict):
                        run.output["messages"] = msgs_any
            except Exception:
                pass
            return EffectOutcome.completed(result=result)
        except Exception as e:
            logger.error("LLM_CALL failed", error=str(e))
            return EffectOutcome.failed(str(e))

    return _handler


def make_tool_calls_handler(
    *,
    tools: Optional[ToolExecutor] = None,
    artifact_store: Optional[ArtifactStore] = None,
    run_store: Optional[RunStore] = None,
) -> EffectHandler:
    """Create a TOOL_CALLS effect handler.

    Tool execution is performed exclusively via the host-configured ToolExecutor.
    This keeps `RunState.vars` and ledger payloads JSON-safe (durable execution).
    """
    def _handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        payload = dict(effect.payload or {})
        tool_calls = payload.get("tool_calls")
        if not isinstance(tool_calls, list):
            return EffectOutcome.failed("tool_calls requires payload.tool_calls (list)")
        allowed_tools_raw = payload.get("allowed_tools")
        allowlist_enabled = isinstance(allowed_tools_raw, list)
        allowed_tools: Set[str] = set()
        if allowlist_enabled:
            allowed_tools = {str(t) for t in allowed_tools_raw if isinstance(t, str) and t.strip()}

        if tools is None:
            return EffectOutcome.failed(
                "TOOL_CALLS requires a ToolExecutor; configure Runtime with "
                "MappingToolExecutor/AbstractCoreToolExecutor/PassthroughToolExecutor."
            )

        original_call_count = len(tool_calls)

        # Always block non-dict tool call entries: passthrough hosts expect dicts and may crash otherwise.
        blocked_by_index: Dict[int, Dict[str, Any]] = {}
        pre_results_by_index: Dict[int, Dict[str, Any]] = {}
        planned: list[Dict[str, Any]] = []

        # For evidence and deterministic resume merging, keep a positional tool call list aligned to the
        # *original* tool call order. Blocked entries are represented as empty-args stubs.
        tool_calls_for_evidence: list[Dict[str, Any]] = []

        # Optional workspace policy (run.vars-driven). When configured, this rewrites/blocks
        # filesystem-ish tool arguments before they reach the ToolExecutor.
        scope: Optional[WorkspaceScope] = None
        try:
            vars0 = getattr(run, "vars", None)
            scope = WorkspaceScope.from_input_data(vars0) if isinstance(vars0, dict) else None
        except Exception as e:
            return EffectOutcome.failed(str(e))

        sid_str = str(getattr(run, "session_id", "") or "").strip()
        session_attachments_cache: Optional[list[Dict[str, Any]]] = None

        def _loads_dict_like(value: Any) -> Optional[Dict[str, Any]]:
            if value is None:
                return None
            if isinstance(value, dict):
                return dict(value)
            if not isinstance(value, str):
                return None
            text = value.strip()
            if not text:
                return None
            try:
                parsed = json.loads(text)
            except Exception:
                return None
            return parsed if isinstance(parsed, dict) else None

        def _ensure_session_memory_run_exists(*, session_id: str) -> None:
            if run_store is None:
                return
            sid = str(session_id or "").strip()
            if not sid:
                return
            rid = session_memory_owner_run_id(sid)
            try:
                existing = run_store.load(str(rid))
            except Exception:
                existing = None
            if existing is not None:
                return
            now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
            run0 = RunState(
                run_id=str(rid),
                workflow_id="__session_memory__",
                status=RunStatus.COMPLETED,
                current_node="done",
                vars={
                    "context": {"task": "", "messages": []},
                    "scratchpad": {},
                    "_runtime": {"memory_spans": []},
                    "_temp": {},
                    "_limits": {},
                },
                waiting=None,
                output={"messages": []},
                error=None,
                created_at=now_iso,
                updated_at=now_iso,
                actor_id=None,
                session_id=sid,
                parent_run_id=None,
            )
            try:
                run_store.save(run0)
            except Exception:
                # Best-effort: artifacts can still be stored, but run-scoped APIs may 404.
                pass

        def _max_attachment_bytes() -> int:
            raw = str(os.getenv("ABSTRACTGATEWAY_MAX_ATTACHMENT_BYTES", "") or "").strip()
            if raw:
                try:
                    v = int(raw)
                    if v > 0:
                        return v
                except Exception:
                    pass
            return 25 * 1024 * 1024

        def _enqueue_pending_media(media_items: Any) -> None:
            if not isinstance(media_items, list) or not media_items:
                return
            vars0 = getattr(run, "vars", None)
            if not isinstance(vars0, dict):
                return
            runtime_ns = vars0.get("_runtime")
            if not isinstance(runtime_ns, dict):
                runtime_ns = {}
                vars0["_runtime"] = runtime_ns

            pending = runtime_ns.get("pending_media")
            if not isinstance(pending, list):
                pending = []

            def _key(item: Any) -> Optional[Tuple[str, str]]:
                if isinstance(item, str):
                    s = item.strip()
                    return ("path", s) if s else None
                if isinstance(item, dict):
                    aid = item.get("$artifact") or item.get("artifact_id")
                    if isinstance(aid, str) and aid.strip():
                        return ("artifact", aid.strip())
                return None

            seen: set[Tuple[str, str]] = set()
            for it in pending:
                k = _key(it)
                if k is not None:
                    seen.add(k)

            for it in media_items:
                k = _key(it)
                if k is None or k in seen:
                    continue
                if isinstance(it, str):
                    pending.append(it.strip())
                elif isinstance(it, dict):
                    pending.append(dict(it))
                else:
                    continue
                seen.add(k)

            runtime_ns["pending_media"] = pending

        def _register_read_file_as_attachment(*, session_id: str, file_path: str) -> Optional[Dict[str, Any]]:
            if artifact_store is None:
                return None
            sid = str(session_id or "").strip()
            if not sid:
                return None
            fp_raw = str(file_path or "").strip()
            if not fp_raw:
                return None

            try:
                p = Path(fp_raw).expanduser()
            except Exception:
                return None
            try:
                resolved = p.resolve()
            except Exception:
                resolved = p

            try:
                size = int(resolved.stat().st_size)
            except Exception:
                size = -1
            max_bytes = _max_attachment_bytes()
            if size >= 0 and size > max_bytes:
                return None

            try:
                content = resolved.read_bytes()
            except Exception:
                return None
            if len(content) > max_bytes:
                return None

            sha256 = hashlib.sha256(bytes(content)).hexdigest()

            handle = resolved.as_posix()
            if scope is not None:
                try:
                    handle = resolved.relative_to(scope.root).as_posix()
                except Exception:
                    handle = resolved.as_posix()

            filename = resolved.name or (handle.split("/")[-1] if handle else "")
            guessed, _enc = mimetypes.guess_type(filename)
            content_type = str(guessed or "text/plain")

            rid = session_memory_owner_run_id(sid)
            try:
                existing = artifact_store.list_by_run(str(rid))
            except Exception:
                existing = []
            for m in existing or []:
                tags = getattr(m, "tags", None)
                if not isinstance(tags, dict):
                    continue
                if str(tags.get("kind") or "") != "attachment":
                    continue
                if str(tags.get("path") or "") != str(handle):
                    continue
                if str(tags.get("sha256") or "") == sha256:
                    return {
                        "artifact_id": str(getattr(m, "artifact_id", "") or ""),
                        "handle": str(handle),
                        "sha256": sha256,
                        "content_type": content_type,
                        "size_bytes": len(content),
                    }

            _ensure_session_memory_run_exists(session_id=sid)

            tags: Dict[str, str] = {
                "kind": "attachment",
                "source": "tool.read_file",
                "path": str(handle),
                "filename": str(filename),
                "session_id": sid,
                "sha256": sha256,
            }
            try:
                meta = artifact_store.store(bytes(content), content_type=str(content_type), run_id=str(rid), tags=tags)
            except Exception:
                return None
            return {
                "artifact_id": str(getattr(meta, "artifact_id", "") or ""),
                "handle": str(handle),
                "sha256": sha256,
                "content_type": content_type,
                "size_bytes": len(content),
            }

        def _normalize_attachment_query(raw: Any) -> str:
            text = str(raw or "").strip()
            if not text:
                return ""
            if text.startswith("@"):
                text = text[1:].strip()
            text = text.replace("\\", "/")
            if text.lower().startswith("file://"):
                try:
                    from urllib.parse import unquote, urlparse

                    parsed = urlparse(text)
                    if parsed.scheme == "file":
                        text = unquote(parsed.path)
                except Exception:
                    # Best-effort fallback: strip the prefix.
                    text = text[7:]
                text = str(text or "").strip()
            while text.startswith("./"):
                text = text[2:]
            return text

        def _get_session_attachments() -> list[Dict[str, Any]]:
            nonlocal session_attachments_cache
            if session_attachments_cache is not None:
                return list(session_attachments_cache)
            if artifact_store is None or not sid_str:
                session_attachments_cache = []
                return []
            try:
                session_attachments_cache = list_session_attachments(
                    artifact_store=artifact_store, session_id=sid_str, limit=5000
                )
            except Exception:
                session_attachments_cache = []
            return list(session_attachments_cache)

        def _read_file_output_from_open_attachment(*, file_path: str, opened: Dict[str, Any]) -> Optional[str]:
            rendered = opened.get("rendered")
            if not isinstance(rendered, str) or not rendered.strip():
                return None

            start2 = opened.get("start_line")
            end2 = opened.get("end_line")
            count = 0
            try:
                if isinstance(start2, int) and isinstance(end2, int) and start2 >= 1 and end2 >= start2:
                    count = end2 - start2 + 1
            except Exception:
                count = 0

            if count <= 0:
                try:
                    count = len([ln for ln in rendered.splitlines() if re.match(r"^\\s*\\d+:\\s", ln)])
                except Exception:
                    count = 0

            header = f"File: {str(file_path)} ({max(0, int(count))} lines)"
            aid2 = str(opened.get("artifact_id") or "").strip()
            sha2 = str(opened.get("sha256") or "").strip()
            handle2 = str(opened.get("handle") or "").strip()
            bits2: list[str] = []
            if handle2:
                bits2.append(f"@{handle2}")
            if aid2:
                bits2.append(f"id={aid2}")
            if sha2:
                bits2.append(f"sha={sha2[:8]}…")
            info = f"(from attachment: {', '.join(bits2)})" if bits2 else "(from attachment)"

            body = "\n".join(rendered.splitlines()[1:]).lstrip("\n")
            return header + "\n" + info + ("\n" + body if body else "")

        def _attachment_backed_result_for_scope_error(
            *,
            tool_name: str,
            arguments: Dict[str, Any],
            call_id: str,
            runtime_call_id_out: Optional[str],
        ) -> Optional[Dict[str, Any]]:
            if artifact_store is None or not sid_str:
                return None

            def _as_result(*, success: bool, output: Any, error: Optional[str]) -> Dict[str, Any]:
                return {
                    "call_id": call_id,
                    "runtime_call_id": runtime_call_id_out,
                    "name": str(tool_name or ""),
                    "success": bool(success),
                    "output": output,
                    "error": error if not success else None,
                }

            if tool_name == "read_file":
                fp = (
                    arguments.get("file_path")
                    or arguments.get("path")
                    or arguments.get("filename")
                    or arguments.get("file")
                )
                fp_norm = _normalize_attachment_query(fp)
                if not fp_norm:
                    return None

                start_line = arguments.get("start_line") or arguments.get("startLine") or arguments.get("start_line_one_indexed") or 1
                end_line = (
                    arguments.get("end_line")
                    or arguments.get("endLine")
                    or arguments.get("end_line_one_indexed_inclusive")
                    or arguments.get("end_line_one_indexed")
                )
                try:
                    start_i = int(start_line) if start_line is not None and not isinstance(start_line, bool) else 1
                except Exception:
                    start_i = 1
                end_i: Optional[int] = None
                try:
                    if end_line is not None and not isinstance(end_line, bool):
                        end_i = int(end_line)
                except Exception:
                    end_i = None

                ok, opened, err = execute_open_attachment(
                    artifact_store=artifact_store,
                    session_id=sid_str,
                    artifact_id=None,
                    handle=str(fp_norm),
                    expected_sha256=None,
                    start_line=int(start_i),
                    end_line=int(end_i) if end_i is not None else None,
                    max_chars=8000,
                )
                if err == "attachment not found":
                    return None
                if isinstance(opened, dict):
                    _enqueue_pending_media(opened.get("media"))
                    if ok:
                        output_text = _read_file_output_from_open_attachment(file_path=str(fp), opened=opened)
                        if output_text is not None:
                            return _as_result(success=True, output=output_text, error=None)
                    rendered = opened.get("rendered")
                    if isinstance(rendered, str) and rendered.strip():
                        return _as_result(success=False, output=rendered, error=str(err or "Failed to open attachment"))
                return _as_result(success=False, output=None, error=str(err or "Failed to open attachment"))

            if tool_name == "list_files":
                dir_path = arguments.get("directory_path") or arguments.get("path") or arguments.get("folder")
                prefix = _normalize_attachment_query(dir_path)
                if not prefix:
                    return None

                entries = _get_session_attachments()
                if not entries:
                    return None

                prefix_slash = prefix if prefix.endswith("/") else prefix + "/"
                matches: list[Dict[str, Any]] = []
                for e in entries:
                    h = _normalize_attachment_query(e.get("handle"))
                    if not h:
                        continue
                    if h == prefix or h.startswith(prefix_slash):
                        matches.append(dict(e))

                if not matches:
                    return None

                pattern = str(arguments.get("pattern") or "*").strip() or "*"
                recursive = bool(arguments.get("recursive"))
                include_hidden = bool(arguments.get("include_hidden") or arguments.get("includeHidden"))
                head_limit = arguments.get("head_limit") or arguments.get("headLimit")
                try:
                    head_n = int(head_limit) if head_limit is not None and not isinstance(head_limit, bool) else 10
                except Exception:
                    head_n = 10
                head_n = max(1, head_n)

                import fnmatch

                patterns = [p.strip() for p in str(pattern).split("|") if p.strip()] or ["*"]

                def _matches(name: str) -> bool:
                    low = str(name or "").lower()
                    for pat in patterns:
                        if fnmatch.fnmatch(low, pat.lower()):
                            return True
                    return False

                def _is_hidden(rel: str) -> bool:
                    parts = [p for p in str(rel or "").split("/") if p]
                    return any(p.startswith(".") for p in parts)

                rows: list[tuple[str, int]] = []
                for e in matches:
                    h = _normalize_attachment_query(e.get("handle"))
                    if not h:
                        continue
                    rel = h[len(prefix_slash) :] if h.startswith(prefix_slash) else h
                    rel = rel.lstrip("/")
                    if not rel:
                        continue
                    if not recursive and "/" in rel:
                        continue
                    if not include_hidden and _is_hidden(rel):
                        continue
                    if not _matches(rel):
                        continue
                    try:
                        size_b = int(e.get("size_bytes") or 0)
                    except Exception:
                        size_b = 0
                    rows.append((rel, size_b))

                rows.sort(key=lambda x: x[0].lower())
                shown = rows[:head_n]

                hidden_note = "hidden entries excluded" if not include_hidden else "hidden entries included"
                lines: list[str] = [
                    f"Entries in '{prefix}' matching '{pattern}' ({hidden_note}; attachments only; filesystem access blocked):"
                ]
                if not shown:
                    lines.append("  (no attached entries)")
                else:
                    for rel, size_b in shown:
                        size_disp = f" ({size_b:,} bytes)" if size_b > 0 else ""
                        lines.append(f"  {rel}{size_disp}")
                    if len(rows) > head_n:
                        lines.append(f"  ... ({len(rows) - head_n} more)")

                return _as_result(success=True, output="\n".join(lines).rstrip(), error=None)

            if tool_name == "skim_folders":
                raw_paths = arguments.get("paths") or arguments.get("path") or arguments.get("folder")
                paths_list: list[str] = []
                if isinstance(raw_paths, list):
                    paths_list = [str(p).strip() for p in raw_paths if isinstance(p, str) and p.strip()]
                elif isinstance(raw_paths, str) and raw_paths.strip():
                    paths_list = [raw_paths.strip()]
                if not paths_list:
                    return None

                entries = _get_session_attachments()
                if not entries:
                    return None

                include_hidden = bool(arguments.get("include_hidden") or arguments.get("includeHidden"))
                blocks: list[str] = []
                matched_any = False

                for folder in paths_list:
                    prefix = _normalize_attachment_query(folder)
                    if not prefix:
                        continue
                    prefix_slash = prefix if prefix.endswith("/") else prefix + "/"
                    rows: list[tuple[str, int]] = []
                    for e in entries:
                        h = _normalize_attachment_query(e.get("handle"))
                        if not h:
                            continue
                        if h == prefix or h.startswith(prefix_slash):
                            rel = h[len(prefix_slash) :] if h.startswith(prefix_slash) else h
                            rel = rel.lstrip("/")
                            if not rel:
                                continue
                            if not include_hidden and any(seg.startswith(".") for seg in rel.split("/") if seg):
                                continue
                            try:
                                size_b = int(e.get("size_bytes") or 0)
                            except Exception:
                                size_b = 0
                            rows.append((rel, size_b))

                    if not rows:
                        continue
                    matched_any = True
                    rows.sort(key=lambda x: x[0].lower())
                    hidden_note = "hidden entries excluded" if not include_hidden else "hidden entries included"
                    lines: list[str] = [
                        f"Folder map for '{prefix}' ({hidden_note}; attachments only; filesystem access blocked):"
                    ]
                    for rel, size_b in rows[:200]:
                        size_disp = f" ({size_b:,} bytes)" if size_b > 0 else ""
                        lines.append(f"  {rel}{size_disp}")
                    if len(rows) > 200:
                        lines.append(f"  ... ({len(rows) - 200} more)")
                    blocks.append("\n".join(lines).rstrip())

                if not matched_any:
                    return None
                return _as_result(success=True, output="\n\n".join(blocks).rstrip(), error=None)

            if tool_name in {"skim_files", "search_files"}:
                # Attachment-backed reads/searches: operate only on session attachments, never the filesystem.
                entries = _get_session_attachments()
                if not entries:
                    return None

                if tool_name == "skim_files":
                    raw_paths = (
                        arguments.get("paths")
                        or arguments.get("path")
                        or arguments.get("file_path")
                        or arguments.get("filename")
                        or arguments.get("file")
                    )
                    paths_list: list[str] = []
                    if isinstance(raw_paths, list):
                        paths_list = [str(p).strip() for p in raw_paths if isinstance(p, str) and p.strip()]
                    elif isinstance(raw_paths, str) and raw_paths.strip():
                        paths_list = [raw_paths.strip()]
                    if not paths_list:
                        return None

                    head_lines = arguments.get("head_lines") or arguments.get("headLines") or 25
                    try:
                        head_n = int(head_lines) if head_lines is not None and not isinstance(head_lines, bool) else 25
                    except Exception:
                        head_n = 25
                    head_n = min(max(1, head_n), 400)

                    rendered_blocks: list[str] = []
                    matched_any = False
                    for p in paths_list:
                        p_norm = _normalize_attachment_query(p)
                        if not p_norm:
                            continue
                        ok, opened, err = execute_open_attachment(
                            artifact_store=artifact_store,
                            session_id=sid_str,
                            artifact_id=None,
                            handle=str(p_norm),
                            expected_sha256=None,
                            start_line=1,
                            end_line=int(head_n),
                            max_chars=12000,
                        )
                        if err == "attachment not found":
                            continue
                        matched_any = True
                        if isinstance(opened, dict):
                            _enqueue_pending_media(opened.get("media"))
                            block = opened.get("rendered")
                            if isinstance(block, str) and block.strip():
                                rendered_blocks.append(block.strip())
                            else:
                                rendered_blocks.append(f"Error: failed to skim attachment '{p_norm}'.")
                        else:
                            rendered_blocks.append(f"Error: failed to skim attachment '{p_norm}': {err or 'unknown error'}")

                    if not matched_any:
                        return None

                    out_text = "\n\n".join(rendered_blocks).strip()
                    return _as_result(success=True, output=out_text, error=None)

                # search_files
                pattern = str(arguments.get("pattern") or "").strip()
                if not pattern:
                    return None
                path_raw = arguments.get("path") or arguments.get("file_path") or arguments.get("directory_path") or ""
                path_norm = _normalize_attachment_query(path_raw)
                head_limit = arguments.get("head_limit") or arguments.get("headLimit")
                max_hits = arguments.get("max_hits") or arguments.get("maxHits")
                try:
                    head_n = int(head_limit) if head_limit is not None and not isinstance(head_limit, bool) else 10
                except Exception:
                    head_n = 10
                try:
                    max_files = int(max_hits) if max_hits is not None and not isinstance(max_hits, bool) else 8
                except Exception:
                    max_files = 8
                head_n = max(1, head_n)
                max_files = max(1, max_files)

                try:
                    rx = re.compile(pattern, re.IGNORECASE)
                except Exception as e:
                    return _as_result(success=False, output=None, error=f"Invalid regex pattern '{pattern}': {e}")

                prefix_slash = ""
                candidates: list[Dict[str, Any]] = []
                if path_norm:
                    exact = [e for e in entries if _normalize_attachment_query(e.get("handle")) == path_norm]
                    if exact:
                        candidates = exact
                    else:
                        prefix_slash = path_norm if path_norm.endswith("/") else path_norm + "/"
                        candidates = [
                            e for e in entries if _normalize_attachment_query(e.get("handle")).startswith(prefix_slash)
                        ]
                else:
                    candidates = list(entries)

                if not candidates:
                    return None

                out_lines: list[str] = [
                    f"Search results in session attachments for pattern '{pattern}' (attachments only; filesystem access blocked):"
                ]
                matched_files = 0
                for e in candidates:
                    if matched_files >= max_files:
                        break
                    aid = str(e.get("artifact_id") or "").strip()
                    handle = _normalize_attachment_query(e.get("handle"))
                    if not aid or not handle:
                        continue
                    art = artifact_store.load(aid)
                    if art is None:
                        continue
                    try:
                        text = art.content.decode("utf-8")
                    except Exception:
                        continue
                    hits: list[str] = []
                    for i, ln in enumerate(text.splitlines(), start=1):
                        if rx.search(ln):
                            hits.append(f"{i}: {ln}")
                            if len(hits) >= head_n:
                                break
                    if not hits:
                        continue
                    matched_files += 1
                    out_lines.append(f"\nFile: {handle}")
                    out_lines.extend(["  " + h for h in hits])

                if matched_files == 0:
                    return _as_result(success=True, output="No matches found in session attachments.", error=None)

                if matched_files >= max_files and len(candidates) > max_files:
                    out_lines.append(f"\nNote: stopped after max_hits={max_files}; more attachments may match.")

                return _as_result(success=True, output="\n".join(out_lines).rstrip(), error=None)

            return None

        # Parse + plan tool calls (preserve order; runtime-owned tools must not run ahead of host tools).
        for idx, tc in enumerate(tool_calls):
            if not isinstance(tc, dict):
                blocked_by_index[idx] = {
                    "call_id": "",
                    "runtime_call_id": None,
                    "name": "",
                    "success": False,
                    "output": None,
                    "error": "Invalid tool call (expected an object)",
                }
                tool_calls_for_evidence.append({})
                continue

            name_raw = tc.get("name")
            name = name_raw.strip() if isinstance(name_raw, str) else ""
            call_id = str(tc.get("call_id") or "")
            runtime_call_id = tc.get("runtime_call_id")
            runtime_call_id_str = str(runtime_call_id).strip() if runtime_call_id is not None else ""
            runtime_call_id_out = runtime_call_id_str or None

            if allowlist_enabled:
                if not name:
                    blocked_by_index[idx] = {
                        "call_id": call_id,
                        "runtime_call_id": runtime_call_id_out,
                        "name": "",
                        "success": False,
                        "output": None,
                        "error": "Tool call missing a valid name",
                    }
                    tool_calls_for_evidence.append(
                        {"call_id": call_id, "runtime_call_id": runtime_call_id_out, "name": "", "arguments": {}}
                    )
                    continue
                if name not in allowed_tools:
                    blocked_by_index[idx] = {
                        "call_id": call_id,
                        "runtime_call_id": runtime_call_id_out,
                        "name": name,
                        "success": False,
                        "output": None,
                        "error": f"Tool '{name}' is not allowed for this node",
                    }
                    # Do not leak arguments for disallowed tools into the durable wait payload.
                    tool_calls_for_evidence.append(
                        {"call_id": call_id, "runtime_call_id": runtime_call_id_out, "name": name, "arguments": {}}
                    )
                    continue

            raw_arguments = tc.get("arguments") or {}
            arguments = dict(raw_arguments) if isinstance(raw_arguments, dict) else (_loads_dict_like(raw_arguments) or {})

            if name == "open_attachment":
                tool_calls_for_evidence.append(
                    {
                        "call_id": call_id,
                        "runtime_call_id": runtime_call_id_out,
                        "name": name,
                        "arguments": dict(arguments),
                    }
                )
                planned.append(
                    {
                        "idx": idx,
                        "kind": "runtime",
                        "name": name,
                        "call_id": call_id,
                        "runtime_call_id": runtime_call_id_out,
                        "arguments": dict(arguments),
                    }
                )
                continue

            # Host tools: rewrite under workspace scope (when configured) before execution.
            tc2 = dict(tc)
            if scope is not None:
                try:
                    rewritten_args = rewrite_tool_arguments(tool_name=name, args=arguments, scope=scope)
                    tc2["arguments"] = rewritten_args
                except Exception as e:
                    fixed = _attachment_backed_result_for_scope_error(
                        tool_name=name,
                        arguments=dict(arguments),
                        call_id=call_id,
                        runtime_call_id_out=runtime_call_id_out,
                    )
                    if fixed is not None:
                        blocked_by_index[idx] = fixed
                        tool_calls_for_evidence.append(
                            {
                                "call_id": call_id,
                                "runtime_call_id": runtime_call_id_out,
                                "name": name,
                                "arguments": tc.get("arguments") or {},
                            }
                        )
                        continue
                    blocked_by_index[idx] = {
                        "call_id": call_id,
                        "runtime_call_id": runtime_call_id_out,
                        "name": name,
                        "success": False,
                        "output": None,
                        "error": str(e),
                    }
                    tool_calls_for_evidence.append(
                        {
                            "call_id": call_id,
                            "runtime_call_id": runtime_call_id_out,
                            "name": name,
                            "arguments": tc.get("arguments") or {},
                        }
                    )
                    continue
            else:
                tc2["arguments"] = dict(arguments)

            tool_calls_for_evidence.append(tc2)
            planned.append({"idx": idx, "kind": "host", "name": name, "tc": tc2})

        # Fast path: if nothing is planned (everything blocked), return blocked results.
        if not planned and blocked_by_index:
            merged_results: list[Any] = []
            for idx in range(len(tool_calls)):
                fixed = blocked_by_index.get(idx)
                merged_results.append(
                    fixed
                    if fixed is not None
                    else {
                        "call_id": "",
                        "runtime_call_id": None,
                        "name": "",
                        "success": False,
                        "output": None,
                        "error": "Missing tool result",
                    }
                )
            return EffectOutcome.completed(result={"mode": "executed", "results": merged_results})

        has_host_calls = any(item.get("kind") == "host" for item in planned)
        if not has_host_calls:
            results_by_index: Dict[int, Dict[str, Any]] = dict(blocked_by_index)
            for item in planned:
                if item.get("kind") != "runtime":
                    continue
                args = dict(item.get("arguments") or {})
                call_id = str(item.get("call_id") or "")
                runtime_call_id_out = item.get("runtime_call_id")
                aid = args.get("artifact_id") or args.get("$artifact") or args.get("id")
                handle = args.get("handle") or args.get("path")
                expected_sha256 = args.get("expected_sha256") or args.get("sha256")
                start_line = args.get("start_line") or args.get("startLine") or 1
                end_line = args.get("end_line") or args.get("endLine")
                max_chars = args.get("max_chars") or args.get("maxChars") or 8000

                if artifact_store is None:
                    results_by_index[item["idx"]] = {
                        "call_id": call_id,
                        "runtime_call_id": runtime_call_id_out,
                        "name": "open_attachment",
                        "success": False,
                        "output": {"rendered": "Error: ArtifactStore is not available (cannot open attachments)."},
                        "error": "ArtifactStore is not available",
                    }
                    continue

                success, output, err = execute_open_attachment(
                    artifact_store=artifact_store,
                    session_id=sid_str,
                    artifact_id=str(aid).strip() if aid is not None else None,
                    handle=str(handle).strip() if handle is not None else None,
                    expected_sha256=str(expected_sha256).strip() if expected_sha256 is not None else None,
                    start_line=int(start_line) if not isinstance(start_line, bool) else 1,
                    end_line=int(end_line) if end_line is not None and not isinstance(end_line, bool) else None,
                    max_chars=int(max_chars) if not isinstance(max_chars, bool) else 8000,
                )
                if bool(success) and isinstance(output, dict):
                    _enqueue_pending_media(output.get("media"))
                results_by_index[item["idx"]] = {
                    "call_id": call_id,
                    "runtime_call_id": runtime_call_id_out,
                    "name": "open_attachment",
                    "success": bool(success),
                    "output": _jsonable(output),
                    "error": str(err or "") if not success else None,
                }

            merged_results: list[Any] = []
            for idx in range(len(tool_calls)):
                r = results_by_index.get(idx)
                merged_results.append(
                    r
                    if r is not None
                    else {
                        "call_id": "",
                        "runtime_call_id": None,
                        "name": "",
                        "success": False,
                        "output": None,
                        "error": "Missing tool result",
                    }
                )
            return EffectOutcome.completed(result={"mode": "executed", "results": merged_results})

        # Detect delegating executors (best-effort): passthrough/untrusted modes cannot safely
        # interleave runtime-owned tools with host tools, so we fall back to the legacy wait behavior.
        executor_delegates = False
        try:
            probe = tools.execute(tool_calls=[])
            mode_probe = probe.get("mode")
            executor_delegates = bool(mode_probe and mode_probe != "executed")
        except Exception:
            executor_delegates = False

        if executor_delegates:
            host_tool_calls: list[Dict[str, Any]] = []
            for item in planned:
                if item.get("kind") == "runtime":
                    args = dict(item.get("arguments") or {})
                    aid = args.get("artifact_id") or args.get("$artifact") or args.get("id")
                    handle = args.get("handle") or args.get("path")
                    expected_sha256 = args.get("expected_sha256") or args.get("sha256")
                    start_line = args.get("start_line") or args.get("startLine") or 1
                    end_line = args.get("end_line") or args.get("endLine")
                    max_chars = args.get("max_chars") or args.get("maxChars") or 8000

                    if artifact_store is None:
                        pre_results_by_index[item["idx"]] = {
                            "call_id": item.get("call_id") or "",
                            "runtime_call_id": item.get("runtime_call_id"),
                            "name": "open_attachment",
                            "success": False,
                            "output": {"rendered": "Error: ArtifactStore is not available (cannot open attachments)."},
                            "error": "ArtifactStore is not available",
                        }
                        continue

                    success, output, err = execute_open_attachment(
                        artifact_store=artifact_store,
                        session_id=sid_str,
                        artifact_id=str(aid).strip() if aid is not None else None,
                        handle=str(handle).strip() if handle is not None else None,
                        expected_sha256=str(expected_sha256).strip() if expected_sha256 is not None else None,
                        start_line=int(start_line) if not isinstance(start_line, bool) else 1,
                        end_line=int(end_line) if end_line is not None and not isinstance(end_line, bool) else None,
                        max_chars=int(max_chars) if not isinstance(max_chars, bool) else 8000,
                    )
                    if bool(success) and isinstance(output, dict):
                        _enqueue_pending_media(output.get("media"))
                    pre_results_by_index[item["idx"]] = {
                        "call_id": item.get("call_id") or "",
                        "runtime_call_id": item.get("runtime_call_id"),
                        "name": "open_attachment",
                        "success": bool(success),
                        "output": _jsonable(output),
                        "error": str(err or "") if not success else None,
                    }
                    continue

                if item.get("kind") == "host":
                    tc2 = item.get("tc")
                    if isinstance(tc2, dict):
                        host_tool_calls.append(tc2)

            try:
                result = tools.execute(tool_calls=host_tool_calls)
            except Exception as e:
                logger.error("TOOL_CALLS execution failed", error=str(e))
                return EffectOutcome.failed(str(e))

            mode = result.get("mode")
            if mode and mode != "executed":
                wait_key = payload.get("wait_key") or result.get("wait_key") or f"tool_calls:{run.run_id}:{run.current_node}"
                raw_wait_reason = result.get("wait_reason")
                wait_reason = WaitReason.EVENT
                if isinstance(raw_wait_reason, str) and raw_wait_reason.strip():
                    try:
                        wait_reason = WaitReason(raw_wait_reason.strip())
                    except ValueError:
                        wait_reason = WaitReason.EVENT
                elif str(mode).strip().lower() == "delegated":
                    wait_reason = WaitReason.JOB

                tool_calls_for_wait = result.get("tool_calls")
                if not isinstance(tool_calls_for_wait, list):
                    tool_calls_for_wait = host_tool_calls

                details: Dict[str, Any] = {"mode": mode, "tool_calls": _jsonable(tool_calls_for_wait)}
                executor_details = result.get("details")
                if isinstance(executor_details, dict) and executor_details:
                    details["executor"] = _jsonable(executor_details)
                if blocked_by_index or pre_results_by_index:
                    details["original_call_count"] = original_call_count
                    if blocked_by_index:
                        details["blocked_by_index"] = {str(k): _jsonable(v) for k, v in blocked_by_index.items()}
                    if pre_results_by_index:
                        details["pre_results_by_index"] = {str(k): _jsonable(v) for k, v in pre_results_by_index.items()}
                    details["tool_calls_for_evidence"] = _jsonable(tool_calls_for_evidence)

                wait = WaitState(
                    reason=wait_reason,
                    wait_key=str(wait_key),
                    resume_to_node=payload.get("resume_to_node") or default_next_node,
                    result_key=effect.result_key,
                    details=details,
                )
                return EffectOutcome.waiting(wait)

            # Defensive: if a delegating executor unexpectedly executes, merge like legacy path.
            existing_results = result.get("results")
            merged_results: list[Any] = []
            executed_iter = iter(existing_results if isinstance(existing_results, list) else [])
            for idx in range(len(tool_calls)):
                fixed = pre_results_by_index.get(idx) or blocked_by_index.get(idx)
                if fixed is not None:
                    merged_results.append(fixed)
                    continue
                try:
                    merged_results.append(next(executed_iter))
                except StopIteration:
                    merged_results.append(
                        {
                            "call_id": "",
                            "runtime_call_id": None,
                            "name": "",
                            "success": False,
                            "output": None,
                            "error": "Missing tool result",
                        }
                    )
            return EffectOutcome.completed(result={"mode": "executed", "results": merged_results})

        # Executing mode: preserve ordering by interleaving runtime-owned tools and host tools.
        results_by_index: Dict[int, Dict[str, Any]] = dict(blocked_by_index)

        i = 0
        while i < len(planned):
            item = planned[i]
            kind = item.get("kind")
            if kind == "runtime":
                args = dict(item.get("arguments") or {})
                call_id = str(item.get("call_id") or "")
                runtime_call_id_out = item.get("runtime_call_id")
                aid = args.get("artifact_id") or args.get("$artifact") or args.get("id")
                handle = args.get("handle") or args.get("path")
                expected_sha256 = args.get("expected_sha256") or args.get("sha256")
                start_line = args.get("start_line") or args.get("startLine") or 1
                end_line = args.get("end_line") or args.get("endLine")
                max_chars = args.get("max_chars") or args.get("maxChars") or 8000

                if artifact_store is None:
                    results_by_index[item["idx"]] = {
                        "call_id": call_id,
                        "runtime_call_id": runtime_call_id_out,
                        "name": "open_attachment",
                        "success": False,
                        "output": {"rendered": "Error: ArtifactStore is not available (cannot open attachments)."},
                        "error": "ArtifactStore is not available",
                    }
                    i += 1
                    continue

                success, output, err = execute_open_attachment(
                    artifact_store=artifact_store,
                    session_id=sid_str,
                    artifact_id=str(aid).strip() if aid is not None else None,
                    handle=str(handle).strip() if handle is not None else None,
                    expected_sha256=str(expected_sha256).strip() if expected_sha256 is not None else None,
                    start_line=int(start_line) if not isinstance(start_line, bool) else 1,
                    end_line=int(end_line) if end_line is not None and not isinstance(end_line, bool) else None,
                    max_chars=int(max_chars) if not isinstance(max_chars, bool) else 8000,
                )
                if bool(success) and isinstance(output, dict):
                    _enqueue_pending_media(output.get("media"))
                results_by_index[item["idx"]] = {
                    "call_id": call_id,
                    "runtime_call_id": runtime_call_id_out,
                    "name": "open_attachment",
                    "success": bool(success),
                    "output": _jsonable(output),
                    "error": str(err or "") if not success else None,
                }
                i += 1
                continue

            # Host tool segment.
            seg_items: list[Dict[str, Any]] = []
            seg_calls: list[Dict[str, Any]] = []
            while i < len(planned) and planned[i].get("kind") == "host":
                seg_items.append(planned[i])
                tc2 = planned[i].get("tc")
                if isinstance(tc2, dict):
                    seg_calls.append(tc2)
                i += 1

            if not seg_calls:
                continue

            try:
                seg_result = tools.execute(tool_calls=seg_calls)
            except Exception as e:
                logger.error("TOOL_CALLS execution failed", error=str(e))
                return EffectOutcome.failed(str(e))

            mode = seg_result.get("mode")
            if mode and mode != "executed":
                return EffectOutcome.failed("ToolExecutor returned delegated mode during executed TOOL_CALLS batch")

            seg_results = seg_result.get("results")
            if not isinstance(seg_results, list):
                return EffectOutcome.failed("ToolExecutor returned invalid results")

            # Map results back to original tool call indices and register read_file outputs as attachments.
            max_inline_bytes = 256 * 1024
            try:
                raw_max_inline = str(os.getenv("ABSTRACTRUNTIME_MAX_INLINE_BYTES", "") or "").strip()
                if raw_max_inline:
                    max_inline_bytes = max(1, int(raw_max_inline))
            except Exception:
                max_inline_bytes = 256 * 1024

            for seg_item, r in zip(seg_items, seg_results):
                idx = int(seg_item.get("idx") or 0)
                r_out: Any = r
                if isinstance(r, dict):
                    r_out = dict(r)
                else:
                    tc2 = seg_item.get("tc") if isinstance(seg_item.get("tc"), dict) else {}
                    r_out = {
                        "call_id": str(tc2.get("call_id") or ""),
                        "runtime_call_id": tc2.get("runtime_call_id"),
                        "name": str(tc2.get("name") or ""),
                        "success": False,
                        "output": None,
                        "error": "Invalid tool result",
                    }

                if seg_item.get("name") != "read_file":
                    results_by_index[idx] = _jsonable(r_out)
                    continue
                tc2 = seg_item.get("tc")
                args = tc2.get("arguments") if isinstance(tc2, dict) else None
                if not isinstance(args, dict):
                    args = {}
                fp = args.get("file_path") or args.get("path") or args.get("filename") or args.get("file")
                if fp is None:
                    results_by_index[idx] = _jsonable(r_out)
                    continue

                # Fallback: if filesystem read_file fails, attempt to resolve from the session attachment store.
                #
                # This supports browser uploads (no server-side file path) and intentionally bypasses
                # workspace allow/ignore policies because the user explicitly provided the bytes.
                if isinstance(r, dict) and r.get("success") is not True and artifact_store is not None and sid_str:
                    start_line = args.get("start_line") or args.get("startLine") or args.get("start_line_one_indexed") or 1
                    end_line = (
                        args.get("end_line")
                        or args.get("endLine")
                        or args.get("end_line_one_indexed_inclusive")
                        or args.get("end_line_one_indexed")
                    )
                    try:
                        start_i = int(start_line) if start_line is not None and not isinstance(start_line, bool) else 1
                    except Exception:
                        start_i = 1
                    end_i: Optional[int] = None
                    try:
                        if end_line is not None and not isinstance(end_line, bool):
                            end_i = int(end_line)
                    except Exception:
                        end_i = None

                    success2, out2, _err2 = execute_open_attachment(
                        artifact_store=artifact_store,
                        session_id=sid_str,
                        artifact_id=None,
                        handle=str(fp),
                        expected_sha256=None,
                        start_line=int(start_i),
                        end_line=int(end_i) if end_i is not None else None,
                        max_chars=8000,
                    )
                    if isinstance(out2, dict):
                        _enqueue_pending_media(out2.get("media"))
                    if success2 and isinstance(out2, dict):
                        output_text = _read_file_output_from_open_attachment(file_path=str(fp), opened=out2)
                        if output_text is not None and isinstance(r_out, dict):
                            r_out["success"] = True
                            r_out["output"] = output_text
                            r_out["error"] = None
                            results_by_index[idx] = _jsonable(r_out)
                            continue

                if not isinstance(r, dict) or r.get("success") is not True:
                    results_by_index[idx] = _jsonable(r_out)
                    continue
                out = r.get("output")
                if not isinstance(out, str) or not out.lstrip().startswith("File:"):
                    results_by_index[idx] = _jsonable(r_out)
                    continue
                att = _register_read_file_as_attachment(session_id=sid_str, file_path=str(fp))
                if att and isinstance(r_out, dict):
                    # If the read_file output would be offloaded anyway, keep the durable ledger lean by
                    # returning a stub and rely on the attachment + open_attachment for bounded excerpts.
                    try:
                        n = len(out.encode("utf-8"))
                    except Exception:
                        n = len(out)
                    if n > max_inline_bytes:
                        aid = str(att.get("artifact_id") or "").strip()
                        handle = str(att.get("handle") or "").strip()
                        sha = str(att.get("sha256") or "").strip()
                        sha_disp = (sha[:8] + "…") if sha else ""
                        display = handle.replace("\\", "/")
                        if _is_abs_path_like(display):
                            display = display.rsplit("/", 1)[-1] or display
                        hint = (
                            f"[read_file]: (stored as attachment) @{display} "
                            f"(id={aid}{', sha=' + sha_disp if sha_disp else ''}).\n"
                            f"Use open_attachment(artifact_id='{aid}', start_line=1, end_line=200) for bounded excerpts."
                        )
                        r_out["output"] = hint

                results_by_index[idx] = _jsonable(r_out)

            # Fill missing results when executor returned fewer entries than expected.
            if len(seg_results) < len(seg_items):
                for seg_item in seg_items[len(seg_results) :]:
                    idx = int(seg_item.get("idx") or 0)
                    tc2 = seg_item.get("tc") if isinstance(seg_item.get("tc"), dict) else {}
                    results_by_index[idx] = {
                        "call_id": str(tc2.get("call_id") or ""),
                        "runtime_call_id": tc2.get("runtime_call_id"),
                        "name": str(tc2.get("name") or ""),
                        "success": False,
                        "output": None,
                        "error": "Missing tool result",
                    }

        merged_results: list[Any] = []
        for idx in range(len(tool_calls)):
            r = results_by_index.get(idx)
            if r is None:
                merged_results.append(
                    {
                        "call_id": "",
                        "runtime_call_id": None,
                        "name": "",
                        "success": False,
                        "output": None,
                        "error": "Missing tool result",
                    }
                )
            else:
                merged_results.append(r)

        return EffectOutcome.completed(result={"mode": "executed", "results": merged_results})

    return _handler


def build_effect_handlers(
    *,
    llm: AbstractCoreLLMClient,
    tools: ToolExecutor = None,
    artifact_store: Optional[ArtifactStore] = None,
    run_store: Optional[RunStore] = None,
) -> Dict[EffectType, Any]:
    return {
        EffectType.LLM_CALL: make_llm_call_handler(llm=llm, artifact_store=artifact_store),
        EffectType.TOOL_CALLS: make_tool_calls_handler(tools=tools, artifact_store=artifact_store, run_store=run_store),
    }
