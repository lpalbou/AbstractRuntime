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

from ...core.event_keys import build_tool_approval_wait_key
from ...core.models import Effect, EffectType, RunState, RunStatus, WaitReason, WaitState
from ...core.runtime import EffectOutcome, EffectHandler
from ...storage.base import RunStore
from ...storage.artifacts import ArtifactStore, is_artifact_ref, get_artifact_id
from .llm_client import AbstractCoreLLMClient, _models_agree
from .output_specs import (
    is_abstractcore_output_request,
    output_request_has_generated_media,
    output_request_has_non_text_result,
    normalize_output_specs_for_runtime,
)
from .tool_executor import ToolApprovalPolicy, ToolExecutor
from .logging import get_logger
from .session_attachments import (
    dedup_messages_view,
    execute_open_attachment,
    list_session_attachments,
    materialize_attachment_path,
    render_active_attachments_system_message,
    render_session_attachments_system_message,
    session_memory_owner_run_id,
)
from .workspace_scoped_tools import WorkspaceScope, rewrite_tool_arguments

logger = get_logger(__name__)

_JSON_SCHEMA_PRIMITIVE_TYPES: Set[str] = {"string", "integer", "number", "boolean", "array", "object", "null"}

_AGORA_TOOL_NAMES_CACHE: Optional[frozenset] = None

# Tools that receive the schema-hidden `_session_route` stamp (the run's own
# provider/model, derived from `_runtime.*` — never payload-claimed). This is
# a DELIBERATE manual allowlist (exact names): tighter than deriving from
# tool signatures, which would auto-stamp any host tool that declares the
# param. The cost is manual sync — core announces new consumers on the
# thread (core owns the param's consumption contract) and this set widens
# with a matching pin; a consumer missing from this set degrades to core's
# configured fallback, never to a trust hole.
_SESSION_ROUTE_TOOL_NAMES = frozenset({"analyze_media"})


def _agora_tool_names() -> frozenset:
    """Exact names of the runtime's agora toolset (H8 identity stamping targets).

    Lazily imported so this module never pays the agora import unless tool calls
    actually execute; sourced from the toolset's own registry so the stamp list
    can never drift from the declared tools.
    """
    global _AGORA_TOOL_NAMES_CACHE
    if _AGORA_TOOL_NAMES_CACHE is None:
        try:
            from .agora_tools import AGORA_TOOL_NAMES
        except ImportError:  # pragma: no cover - abstractcore genuinely absent
            # Do NOT cache the empty set: a transient failure must never
            # permanently disable a trust-boundary stamp for the process
            # lifetime (adversary finding). Without abstractcore the agora
            # toolset cannot execute either, so an empty answer here is
            # consistent, not a bypass.
            return frozenset()
        _AGORA_TOOL_NAMES_CACHE = frozenset(AGORA_TOOL_NAMES)
    return _AGORA_TOOL_NAMES_CACHE

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
    explicit = {
        "audio/wav": ".wav",
        "audio/wave": ".wav",
        "audio/x-wav": ".wav",
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
        "audio/ogg": ".ogg",
        "audio/flac": ".flac",
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "video/mp4": ".mp4",
        "video/quicktime": ".mov",
    }
    if ct in explicit:
        return explicit[ct]
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


def _is_sensitive_observability_key(key: str) -> bool:
    key_l = str(key or "").lower()
    if not key_l:
        return False
    if any(fragment in key_l for fragment in ("api_key", "api-key", "apikey", "authorization", "password", "secret")):
        return True
    parts = [p for p in re.split(r"[^a-z0-9]+", key_l) if p]
    if "token" not in parts:
        return False
    return any(part in parts for part in ("access", "auth", "bearer", "bot", "csrf", "id", "refresh", "session")) or key_l in {
        "token",
        "auth_token",
        "access_token",
        "refresh_token",
        "id_token",
        "bearer_token",
        "csrf_token",
        "session_token",
    }


def _observability_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Return params safe for persisted runtime observability traces."""

    callback_keys = {"on_progress", "progress_callback", "progress_event_callback"}
    out: Dict[str, Any] = {}
    for key, value in dict(params or {}).items():
        key_s = str(key)
        if key_s in callback_keys or callable(value):
            continue
        if _is_sensitive_observability_key(key_s):
            out[key_s] = "[redacted]" if value not in (None, "") else value
            continue
        out[key_s] = value
    return out


_MISSING = object()


def _first_output_provider(value: Any) -> Optional[str]:
    if isinstance(value, dict):
        provider = value.get("provider")
        if isinstance(provider, str) and provider.strip():
            return provider.strip()
        specs = value.get("specs")
        if isinstance(specs, list):
            for item in specs:
                found = _first_output_provider(item)
                if found:
                    return found
    elif isinstance(value, list):
        for item in value:
            found = _first_output_provider(item)
            if found:
                return found
    return None


def _replace_output_provider(value: Any, *, old_provider: str, new_provider: str) -> Any:
    old_norm = str(old_provider or "").strip().lower()
    if not old_norm or not isinstance(new_provider, str) or not new_provider.strip():
        return value
    if isinstance(value, dict):
        out = dict(value)
        provider = out.get("provider")
        if isinstance(provider, str) and provider.strip().lower() == old_norm:
            out["provider"] = new_provider.strip()
        if isinstance(out.get("specs"), list):
            out["specs"] = [_replace_output_provider(item, old_provider=old_provider, new_provider=new_provider) for item in out["specs"]]
        return out
    if isinstance(value, list):
        return [_replace_output_provider(item, old_provider=old_provider, new_provider=new_provider) for item in value]
    return value


def _apply_provider_endpoint_profile_resolution(*, llm: AbstractCoreLLMClient, params: Dict[str, Any]) -> None:
    """Resolve a host-provided virtual provider profile into Core routing params.

    Hosts can attach `resolve_provider_endpoint_profile(provider_id)` to the LLM
    client. Runtime keeps this as a generic optional hook so it does not import
    Gateway, and only the transient `params` dict receives the raw provider key.
    """

    resolver = getattr(llm, "resolve_provider_endpoint_profile", None)
    if not callable(resolver):
        return
    provider = params.get("_provider")
    if not (isinstance(provider, str) and provider.strip()):
        provider = _first_output_provider(params.get("output"))
    if not (isinstance(provider, str) and provider.strip()):
        return
    try:
        resolved = resolver(provider.strip())
    except Exception as exc:
        raise RuntimeError(f"Failed to resolve provider endpoint profile {provider!r}: {exc}") from exc
    if not isinstance(resolved, dict):
        return

    provider_family = str(resolved.get("provider") or resolved.get("provider_family") or "").strip()
    if not provider_family:
        raise RuntimeError(f"Provider endpoint profile {provider!r} did not resolve to a provider family.")

    params["_provider"] = provider_family
    if "output" in params:
        params["output"] = _replace_output_provider(params.get("output"), old_provider=provider, new_provider=provider_family)

    base_url = resolved.get("base_url")
    if isinstance(base_url, str) and base_url.strip() and not str(params.get("base_url") or "").strip():
        params["base_url"] = base_url.strip()

    api_key = resolved.get("api_key")
    if isinstance(api_key, str) and api_key.strip():
        existing_key = params.get("provider_api_key") or params.get("api_key")
        if not (isinstance(existing_key, str) and existing_key.strip()):
            params["provider_api_key"] = api_key.strip()

    metadata = {
        "id": str(resolved.get("id") or "").strip(),
        "virtual_provider": str(resolved.get("virtual_provider") or provider).strip(),
        "display_name": str(resolved.get("display_name") or "").strip(),
        "provider_family": provider_family,
        "scope": str(resolved.get("scope") or "").strip(),
        "base_url_configured": bool(str(resolved.get("base_url") or "").strip()),
        "api_key_set": bool(isinstance(api_key, str) and api_key.strip()),
    }
    params["_provider_endpoint_profile"] = {k: v for k, v in metadata.items() if v not in ("", None)}


def _stamp_declared_route(
    result: Any,
    *,
    declared_provider: Any,
    declared_model: Any,
    endpoint_profile: Any = None,
) -> None:
    """Put the DECLARED route on the record next to the served one.

    A ledger record used to declare the request (`provider: openai, model:
    gpt-5.6-sol`) while the response came from a different endpoint entirely, so
    the observability layer reported the request and not the service. The client
    stamps `result["route"]` from the constructed provider instance; here we add
    what the flow asked for and raise `route["mismatch"]` when the two disagree.
    Best-effort and never fatal: an unstamped result still gets a declared-only
    route so the record never silently implies a route it cannot vouch for.
    """

    if not isinstance(result, dict):
        return
    declared_p = str(declared_provider or "").strip() or None
    declared_m = str(declared_model or "").strip() or None
    route = result.get("route")
    if not isinstance(route, dict):
        route = {"source": "declared", "provider": None, "base_url": None, "served_model": result.get("model")}
        result["route"] = route
    route["declared_provider"] = declared_p
    route["declared_model"] = declared_m
    if isinstance(endpoint_profile, dict):
        vp = str(endpoint_profile.get("virtual_provider") or "").strip()
        if vp:
            route["endpoint_profile"] = vp
    served = route.get("served_model") or result.get("model")
    if declared_m and served and not _models_agree(declared_m, served):
        route["mismatch"] = True
        route.setdefault(
            "mismatch_reason",
            f"declared model {declared_m!r} was served by {str(served)!r}",
        )
    route.setdefault("mismatch", False)


def _resolved_generate_route_outputs(summary: Dict[str, Any]) -> list[Dict[str, Any]]:
    outputs = summary.get("outputs")
    if not isinstance(outputs, list):
        return []
    return [dict(item) for item in outputs if isinstance(item, dict)]


def _resolved_action_id_from_output(summary: Dict[str, Any]) -> str:
    outputs = _resolved_generate_route_outputs(summary)
    if not outputs:
        return "generate_text"
    first = outputs[0]
    modality = str(first.get("modality") or "").strip().lower()
    task = str(first.get("task") or "").strip().lower()
    if modality == "text":
        if task == "transcription":
            return "transcribe_audio"
        return "generate_text"
    if modality == "image":
        if task in {"image_edit", "image_to_image"}:
            return "edited_image"
        if task in {"image_upscale", "upscale_image"}:
            return "upscaled_image"
        return "generated_image"
    if modality == "video":
        if task in {"image_to_video", "i2v"}:
            return "image_to_video"
        return "generated_video"
    if modality == "voice":
        return "generated_voice"
    if modality == "music":
        if task == "text_to_audio":
            return "generated_sound"
        return "generated_music"
    return "generate_text"


def _runtime_resolved_action_from_generate_metadata(metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    summary = metadata.get("_resolved_generate_route")
    if not isinstance(summary, dict):
        return None
    request_summary = summary.get("request") if isinstance(summary.get("request"), dict) else {}
    outputs = _resolved_generate_route_outputs(summary)
    text_route = summary.get("text_route") if isinstance(summary.get("text_route"), dict) else None
    input_routes = summary.get("input_routes") if isinstance(summary.get("input_routes"), list) else []
    output_routes = summary.get("output_routes") if isinstance(summary.get("output_routes"), list) else []
    effective_route = {
        "text_route": text_route,
        "input_routes": [dict(item) for item in input_routes if isinstance(item, dict)],
        "output_routes": [dict(item) for item in output_routes if isinstance(item, dict)],
        "reasoning": summary.get("reasoning"),
        "reasoning_source": summary.get("reasoning_source"),
    }
    first_output = outputs[0] if outputs else {}
    family = str(first_output.get("modality") or "text").strip().lower() or "text"
    task = str(first_output.get("task") or "text_generation").strip().lower() or "text_generation"
    return {
        "kind": "generate",
        "action_id": _resolved_action_id_from_output(summary),
        "family": family,
        "modality": family,
        "task": task,
        "normalized_request": dict(request_summary) if isinstance(request_summary, dict) else {},
        "normalized_output": outputs,
        "effective_route": effective_route,
        "requested_override": {
            "text_route": text_route.get("field_sources") if isinstance(text_route, dict) else None,
            "output_routes": [item.get("field_sources") for item in output_routes if isinstance(item, dict)],
            "reasoning_source": summary.get("reasoning_source"),
        },
        "override_disposition": "resolved",
        "policy_class": "runtime_call",
    }


def _coerce_media_input(media: Any) -> Any:
    """Accept AbstractCore's single-media convenience shapes while keeping payloads JSON-safe."""
    if media is None:
        return None
    if isinstance(media, tuple):
        return list(media)
    if isinstance(media, list):
        return media
    if isinstance(media, (str, dict)):
        return [media]
    return media


def _payload_output_request(payload: Dict[str, Any], params: Dict[str, Any]) -> Any:
    if "output" in params:
        return params.get("output")
    if "output" in payload:
        return payload.get("output")
    if "outputs" in payload:
        # Runtime convenience alias for AbstractCore's `output=...`.
        return payload.get("outputs")
    return _MISSING


def _runtime_output_tags(run: RunState, *, generated_media: bool) -> Dict[str, str]:
    tags: Dict[str, str] = {
        "kind": "generated_media" if generated_media else "llm_output",
        "source": "llm_call",
        "run_id": str(run.run_id),
        "workflow_id": str(run.workflow_id),
        "node_id": str(run.current_node),
    }
    if run.actor_id:
        tags["actor_id"] = str(run.actor_id)
    if getattr(run, "session_id", None):
        tags["session_id"] = str(run.session_id)
    if run.parent_run_id:
        tags["parent_run_id"] = str(run.parent_run_id)
    return tags


def _augment_output_request_for_runtime(output: Any, *, run: RunState) -> Any:
    """Attach run-scoped artifact metadata to AbstractCore output specs."""

    if not is_abstractcore_output_request(output):
        return output

    specs = normalize_output_specs_for_runtime(output)
    out_specs: list[Dict[str, Any]] = []
    for spec in specs:
        s = dict(spec)
        if not isinstance(s.get("run_id"), str) or not str(s.get("run_id") or "").strip():
            s["run_id"] = str(run.run_id)

        modality = str(s.get("modality") or "").strip().lower()
        task = str(s.get("task") or "").strip()
        tags = _runtime_output_tags(run, generated_media=modality != "text")
        existing_tags = s.get("tags")
        if isinstance(existing_tags, dict):
            tags.update({str(k): str(v) for k, v in existing_tags.items() if v is not None})
        if modality:
            tags.setdefault("modality", modality)
        if task:
            tags.setdefault("task", task)
        s["tags"] = tags
        out_specs.append(s)

    if isinstance(output, (list, tuple)):
        return out_specs
    return out_specs[0] if out_specs else output


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
        from pydantic import BaseModel, Field, create_model
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
            description = None
            default_value = None
            has_default = False
            if isinstance(prop_schema, dict):
                raw_description = prop_schema.get("description")
                if isinstance(raw_description, str) and raw_description.strip():
                    description = raw_description.strip()
                if "default" in prop_schema:
                    default_value = prop_schema.get("default")
                    has_default = True

            if prop_name in required:
                fields[prop_name] = (t, Field(..., description=description))
            else:
                field_default = default_value if has_default else None
                fields[prop_name] = (Optional[t], Field(field_default, description=description))

        return create_model(name, **fields)  # type: ignore[call-arg]

    return _model(schema, name=name)


def _validate_structured_output_candidate(candidate: Any, *, response_model: Type[Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        validated = response_model.model_validate(candidate)
    except Exception as e:
        return None, str(e)

    try:
        dumped = validated.model_dump(mode="json")
    except Exception:
        try:
            dumped = validated.model_dump()
        except Exception:
            dumped = candidate

    if isinstance(dumped, dict):
        return dict(dumped), None
    return None, "validated structured output did not serialize to an object"


def _stringify_structured_output_candidate(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    try:
        rendered = json.dumps(_jsonable(value), ensure_ascii=True, sort_keys=True, indent=2)
    except Exception:
        rendered = str(value)
    return rendered.strip()


def _truncate_prompt_block(value: Any, *, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return f"{text[: max_chars - 3]}..."


def _build_structured_output_repair_prompt(
    *,
    original_task: str,
    schema: Dict[str, Any],
    invalid_output: str,
    error: str,
) -> str:
    schema_text = json.dumps(_jsonable(schema), ensure_ascii=True, sort_keys=True, indent=2)
    task_text = _truncate_prompt_block(original_task, max_chars=4000)
    invalid_text = _truncate_prompt_block(invalid_output, max_chars=4000)
    error_text = _truncate_prompt_block(error, max_chars=2000)
    return (
        "You returned structured output that does not satisfy the required JSON schema.\n"
        "Return corrected JSON only.\n\n"
        f"Original task:\n{task_text}\n\n"
        f"Required JSON schema:\n{schema_text}\n\n"
        f"Previous invalid output:\n{invalid_text}\n\n"
        f"Validation error:\n{error_text}\n\n"
        "Return exactly one JSON object that satisfies the schema. Do not include markdown or explanation."
    )


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


def _env_flag(name: str) -> Optional[bool]:
    """Tri-state env flag: True/False when explicitly set, None when unset/unparseable."""
    raw = os.getenv(name)
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _coerce_positive_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(str(value).strip() if isinstance(value, str) else value)
    except Exception:
        return None
    return parsed if parsed > 0 else None


def _derive_prompt_cache_key(
    *,
    namespace: str,
    session_id: str,
    provider: str,
    model: str,
    workflow_id: str,
    node_id: str,
    version: int = 1,
) -> str:
    ns = str(namespace or "").strip() or "session"
    raw = f"v{int(version)}|{session_id}|{provider}|{model}|{workflow_id}|{node_id}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    return f"{ns}:{digest}"


def _llm_prompt_cache_identity(llm: Any) -> tuple[Optional[str], Optional[str]]:
    getter = getattr(llm, "default_prompt_cache_identity", None)
    if callable(getter):
        try:
            raw = getter()
        except Exception:
            raw = None
        if isinstance(raw, dict):
            provider = raw.get("provider")
            model = raw.get("model")
            return (
                provider.strip().lower() if isinstance(provider, str) and provider.strip() else None,
                model.strip() if isinstance(model, str) and model.strip() else None,
            )
        if isinstance(raw, (list, tuple)) and len(raw) >= 2:
            provider, model = raw[0], raw[1]
            return (
                provider.strip().lower() if isinstance(provider, str) and provider.strip() else None,
                model.strip() if isinstance(model, str) and model.strip() else None,
            )

    provider = getattr(llm, "_default_provider", None) or getattr(llm, "_provider", None)
    model = getattr(llm, "_default_model", None) or getattr(llm, "_model", None)
    return (
        provider.strip().lower() if isinstance(provider, str) and provider.strip() else None,
        model.strip() if isinstance(model, str) and model.strip() else None,
    )


def _normalize_prompt_cache_binding_request(params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    binding = params.pop("expected_prompt_cache_binding", None)
    second = params.get("prompt_cache_binding")
    if binding is None:
        binding = second
    elif second is not None and second != binding:
        raise ValueError("expected_prompt_cache_binding and prompt_cache_binding must match when both are supplied.")

    if binding is None:
        return None
    if isinstance(binding, str):
        # ONE-MEANING-PER-NAME (agent seat c1670 convention; bloc-seam
        # adversary A-3, 2026-07-13): a bare STRING means cache-key intent
        # and routes to prompt_cache_key; the DICT shape is reserved for
        # durable-bloc strict verification. Coercing strings into
        # {"binding_id": ...} manufactured exactly the shape core refuses
        # (prompt_cache_binding_bare_string) one hop before its refusal —
        # the 2026-07-11 live-visit collision class.
        key_s = binding.strip()
        params.pop("prompt_cache_binding", None)
        if key_s:
            existing = params.get("prompt_cache_key")
            if existing is not None and str(existing).strip() and str(existing).strip() != key_s:
                raise ValueError("prompt_cache_key and a string prompt_cache_binding must match.")
            params["prompt_cache_key"] = key_s
        return None
    if not isinstance(binding, dict):
        raise ValueError("prompt_cache_binding must be an object (or a string cache key).")
    params["prompt_cache_binding"] = dict(binding)
    return dict(binding)


def _maybe_inject_runtime_thinking(*, run: RunState, params: Dict[str, Any]) -> None:
    """Fold the run's `_runtime.thinking` into LLM_CALL params when the call
    carries no explicit value.

    WHY (wire witness 2026-08-04, probe session acode-74ac46b3b5b2): run-level
    reasoning is declared once in `_runtime.thinking` and consumed per call.
    Agent react loops read it via abstractagent's `runtime_llm_params`, but
    every OTHER LLM_CALL emitter — the visual Agent node's structured-output
    post-pass (compiler params carry only the node pin/config), plain visual
    `llm_call` nodes (config/pins only) — sent params without `thinking`, so
    a run that asked for medium still emitted ABSENT-effort formatting calls.
    This is the one seam every LLM_CALL crosses, exactly like the
    audio_policy/stt_language riders below it.

    Precedence: an explicit `params.thinking` wins — INCLUDING False ("off"
    is a decision, so the gate is key-presence, never truthiness). Then the
    executing run's `_runtime.thinking` (bool or non-empty str, the consumer
    vocabulary). A run with neither stays absent and the capability-route
    default in llm_client applies, unchanged."""
    if "thinking" in params:
        return
    try:
        runtime_ns = run.vars.get("_runtime") if isinstance(run.vars, dict) else None
    except Exception:
        return
    thinking = runtime_ns.get("thinking") if isinstance(runtime_ns, dict) else None
    if isinstance(thinking, bool) or (isinstance(thinking, str) and thinking.strip()):
        params["thinking"] = thinking


def _maybe_inject_prompt_cache_key(
    *,
    run: RunState,
    params: Dict[str, Any],
    default_provider: Optional[str] = None,
    default_model: Optional[str] = None,
) -> None:
    binding = _normalize_prompt_cache_binding_request(params)
    if isinstance(binding, dict):
        binding_key = binding.get("key")
        if isinstance(binding_key, str) and binding_key.strip():
            binding_key_s = binding_key.strip()
            current_key = params.get("prompt_cache_key")
            if isinstance(current_key, str) and current_key.strip() and current_key.strip() != binding_key_s:
                raise ValueError("prompt_cache_key and prompt_cache_binding.key must match.")
            params["prompt_cache_key"] = binding_key_s
        if "prompt_cache_key" in params:
            return
        # Binding is present but does not carry a usable key. Do not inject a derived key.
        return

    # Explicit caller override wins (including None/empty for "disable").
    if "prompt_cache_key" in params:
        return

    runtime_ns = run.vars.get("_runtime") if isinstance(run.vars, dict) else None
    cfg = runtime_ns.get("prompt_cache") if isinstance(runtime_ns, dict) else None

    enabled: bool
    namespace: str = "session"
    if isinstance(cfg, bool):
        enabled = cfg
    elif isinstance(cfg, dict):
        enabled_raw = cfg.get("enabled")
        enabled = bool(enabled_raw) if enabled_raw is not None else True
        ns = cfg.get("namespace")
        if isinstance(ns, str) and ns.strip():
            namespace = ns.strip()
        key_override = cfg.get("key")
        if isinstance(key_override, str) and key_override.strip():
            params["prompt_cache_key"] = key_override.strip()
            return
    else:
        # Default ON (backlog 0212): the derived key below is session-scoped (it requires a
        # session_id and hashes it together with provider/model/workflow/node), so reuse can
        # never cross sessions. Operators can opt out process-wide via
        # ABSTRACTRUNTIME_PROMPT_CACHE=0; hosts with their own env namespace should translate
        # that config into `_runtime.prompt_cache` explicitly.
        env_pref = _env_flag("ABSTRACTRUNTIME_PROMPT_CACHE")
        enabled = True if env_pref is None else env_pref

    if not enabled:
        return

    # Require a session_id to avoid cross-run accidental reuse.
    session_id = str(getattr(run, "session_id", "") or "").strip()
    if not session_id:
        trace_md = params.get("trace_metadata")
        if isinstance(trace_md, dict):
            sid = trace_md.get("session_id")
            if isinstance(sid, str) and sid.strip():
                session_id = sid.strip()
    if not session_id:
        return

    trace_md = params.get("trace_metadata") if isinstance(params.get("trace_metadata"), dict) else {}
    workflow_id = str(trace_md.get("workflow_id") or run.workflow_id or "").strip()
    node_id = str(trace_md.get("node_id") or run.current_node or "").strip()

    provider = str(params.get("_provider") or default_provider or "").strip().lower()
    model = str(params.get("_model") or default_model or "").strip()
    if not provider or not model:
        return

    params["prompt_cache_key"] = _derive_prompt_cache_key(
        namespace=namespace,
        session_id=session_id,
        provider=provider,
        model=model,
        workflow_id=workflow_id,
        node_id=node_id,
    )
    # Session-attribution rider for the derived (session-scoped) key: the
    # client stamps it onto the provider cache entry AFTER the generate that
    # creates it (missing keys reject meta updates). Explicit/binding keys are
    # caller-owned and may be shared across sessions, so they never get one.
    attribution = {
        "session_id": session_id,
        "run_id": str(getattr(run, "run_id", "") or "").strip() or None,
        "workflow_id": workflow_id or None,
        "node_id": node_id or None,
        "namespace": namespace,
    }
    params["_prompt_cache_attribution"] = {k: v for k, v in attribution.items() if v is not None}


def _normalize_explicit_prompt_cache_binding_without_deriving(params: Dict[str, Any]) -> None:
    binding = _normalize_prompt_cache_binding_request(params)
    if not isinstance(binding, dict):
        return
    binding_key = binding.get("key")
    if isinstance(binding_key, str) and binding_key.strip():
        binding_key_s = binding_key.strip()
        current_key = params.get("prompt_cache_key")
        if isinstance(current_key, str) and current_key.strip() and current_key.strip() != binding_key_s:
            raise ValueError("prompt_cache_key and prompt_cache_binding.key must match.")
        params["prompt_cache_key"] = binding_key_s


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
        content_type = ""
        if isinstance(item, dict):
            for key in ("content_type", "mime_type", "mimeType", "mime"):
                raw_ct = item.get(key)
                if isinstance(raw_ct, str) and raw_ct.strip():
                    content_type = raw_ct.strip()
                    break
        if not content_type:
            raw_artifact_ct = getattr(artifact, "content_type", None)
            if isinstance(raw_artifact_ct, str) and raw_artifact_ct.strip():
                content_type = raw_artifact_ct.strip()

        ext = Path(filename).suffix if filename else ""
        if not ext:
            ct = content_type or str(getattr(getattr(artifact, "metadata", None), "content_type", "") or "")
            ext = _guess_ext_from_content_type(ct)

        desired = source_path or filename or artifact_id
        safe_name = _safe_materialized_filename(desired=desired, artifact_id=str(artifact_id), ext=str(ext))
        p = temp_dir / safe_name
        try:
            p.write_bytes(bytes(content))
        except Exception as e:
            return None, f"Failed to materialize artifact '{artifact_id}': {e}"
        resolved: Dict[str, Any] = {"file_path": str(p), "$artifact": str(artifact_id), "artifact_id": str(artifact_id)}
        if isinstance(item, dict):
            for key in ("role", "purpose", "kind"):
                raw_role = item.get(key)
                if isinstance(raw_role, str) and raw_role.strip():
                    resolved[key] = raw_role.strip()
        if filename:
            resolved["filename"] = filename
        if content_type:
            resolved["content_type"] = content_type
            base_type = content_type.split(";", 1)[0].strip().lower()
            if base_type.startswith("audio/"):
                resolved["type"] = "audio"
            elif base_type.startswith("image/"):
                resolved["type"] = "image"
            elif base_type.startswith("video/"):
                resolved["type"] = "video"
            elif base_type.startswith("text/"):
                resolved["type"] = "text"
        out.append(resolved)

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


_ABORT_USAGE_KEYS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "prompt_tokens",
    "completion_tokens",
)


def _looks_like_aborted_generation(result: Any) -> bool:
    """True when an LLM_CALL result is the corpse of an aborted generation.

    Operator evidence 2026-08-02 (LM Studio 0.3.x + qwen3.6-35b-a3b): a
    generation cut mid-tool-call returns HTTP 200 with the tool-call PREFACE as
    content, `tool_calls: []`, `finish_reason: "stop"` and an all-zero usage
    block. The dropped call appears only in the provider's server log; the wire
    body has no error field. So `finish_reason` cannot be the detector — the
    usage block is: text cannot have been produced by a completion that
    consumed zero prompt tokens and produced zero completion tokens.

    Absent or empty usage is UNKNOWN, never "aborted": we do not manufacture a
    verdict out of missing evidence. A result carrying tool calls is never lost
    work. Kept in sync with abstractcore's provider-side detector; either
    signal alone is enough downstream.
    """
    if not isinstance(result, dict):
        return False
    if result.get("tool_calls"):
        return False
    usage = result.get("usage")
    if not isinstance(usage, dict) or not usage:
        return False
    counters = [usage.get(k) for k in _ABORT_USAGE_KEYS if usage.get(k) is not None]
    if not counters or any(bool(c) for c in counters):
        return False
    for key in ("content", "reasoning"):
        val = result.get(key)
        if isinstance(val, str) and val.strip():
            return True
    return False


def make_llm_call_handler(*, llm: AbstractCoreLLMClient, artifact_store: Optional[ArtifactStore] = None) -> EffectHandler:
    def _handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        payload = dict(effect.payload or {})
        prompt = payload.get("prompt")
        text_input = payload.get("text")
        messages = payload.get("messages")
        system_prompt = payload.get("system_prompt")
        media = _coerce_media_input(payload.get("media"))
        provider = payload.get("provider")
        model = payload.get("model")
        tools_raw = payload.get("tools")
        tools = tools_raw if isinstance(tools_raw, list) and len(tools_raw) > 0 else None
        response_schema = _normalize_response_schema(payload.get("response_schema"))
        response_schema_name = payload.get("response_schema_name")
        structured_output_fallback = payload.get("structured_output_fallback")
        raw_params = payload.get("params")
        params = dict(raw_params) if isinstance(raw_params, dict) else {}

        output_request = _payload_output_request(payload, params)
        if output_request is not _MISSING:
            params["output"] = output_request
        if text_input is not None and not (isinstance(prompt, str) and prompt.strip()):
            prompt = str(text_input)

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
        _apply_provider_endpoint_profile_resolution(llm=llm, params=params)

        default_provider, default_model = _llm_prompt_cache_identity(llm)
        if output_request is not _MISSING and output_request_has_non_text_result(output_request):
            _normalize_explicit_prompt_cache_binding_without_deriving(params)
        else:
            _maybe_inject_prompt_cache_key(
                run=run,
                params=params,
                default_provider=default_provider,
                default_model=default_model,
            )
        if "output" in params:
            params["output"] = _augment_output_request_for_runtime(params.get("output"), run=run)

        if artifact_store is None and output_request_has_generated_media(params.get("output")):
            return EffectOutcome.failed(
                "llm_call generated media outputs require an ArtifactStore", retryable=False
            )

        def _nonempty_str(value: Any) -> Optional[str]:
            if not isinstance(value, str):
                return None
            text = value.strip()
            return text if text else None

        prompt = _nonempty_str(prompt)

        has_messages = isinstance(messages, list) and len(messages) > 0
        has_prompt = isinstance(prompt, str) and bool(prompt)
        has_text_input = isinstance(text_input, str) and bool(text_input.strip())
        has_media_input = isinstance(media, list) and len(media) > 0
        has_output_request = "output" in params and is_abstractcore_output_request(params.get("output"))
        if not has_prompt and not has_messages and not has_text_input and not (has_media_input and has_output_request):
            # A PAYLOAD-SHAPE refusal is deterministic: the identical payload
            # meets the identical refusal on every attempt, so retrying only
            # multiplies the wait before the caller reads it. Name the node and
            # the usual cause -- an unset flow input on the run -- because the
            # caller who hits this is usually a person who left the Run dialog's
            # prompt field empty, not a flow author.
            node_hint = str(getattr(run, "current_node", "") or "").strip()
            where = f" (node {node_hint})" if node_hint else ""
            fix = (
                " Supply the flow's prompt input when starting the run "
                "(input_data.<prompt pin>), or connect/pin it in the flow."
            )
            if has_media_input or "output" in params:
                return EffectOutcome.failed(
                    f"llm_call{where} requires payload.prompt, payload.messages, payload.text, "
                    f"or media with payload.output; all were empty.{fix}",
                    retryable=False,
                )
            return EffectOutcome.failed(
                f"llm_call{where} requires payload.prompt or payload.messages; both were empty.{fix}",
                retryable=False,
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
            # Treat 0/negative as "unset" so we still fall back to run-level limits.
            parsed_max_out: Optional[int] = None
            if raw_max_out is not None and not isinstance(raw_max_out, bool):
                try:
                    parsed_max_out = int(raw_max_out)
                except Exception:
                    parsed_max_out = None
            if parsed_max_out is None or parsed_max_out <= 0:
                raw_max_out = None

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

                def _is_attachment_index_message(m: Dict[str, Any]) -> bool:
                    # Prefer a robust metadata marker; fall back to the legacy content-prefix
                    # heuristic on system messages (older transcripts injected these as system).
                    meta = m.get("metadata")
                    if isinstance(meta, dict) and meta.get("kind") == "attachment_index":
                        return True
                    if m.get("role") == "system":
                        c_str = str(m.get("content") or "").strip()
                        return (
                            c_str.startswith("Active attachments")
                            or c_str.startswith("Stored session attachments")
                            or c_str.startswith("Session attachments")
                        )
                    return False

                # Remove any previously injected attachment index to avoid staleness/duplication.
                cleaned: list[Dict[str, Any]] = [
                    m for m in messages if isinstance(m, dict) and not _is_attachment_index_message(m)
                ]

                # The attachment index rides the TAIL as a SYSTEM message (backlog 0212 cache
                # stability). It MUST stay role=system: it is context, not user speech, and both the
                # grounding-envelope placement and durable-context extraction key off "the last user
                # message" — a user-role index would be mistaken for user input and polluted into
                # durable context.messages.
                # PROVIDER DELIVERY CONTRACT (updated after the 2026-07-09 production incident):
                # non-leading system messages are a TRANSPORT concern each provider must handle —
                # native OpenAI passes them through (accepted anywhere), Anthropic converts them to
                # <system_instruction>-wrapped user turns, and OpenAICompatibleProvider normalizes
                # them the same way because strict vLLM-class servers (e.g. OVH Qwen templates)
                # HARD-REJECT them with HTTP 400 "System message must be at the beginning." — which
                # failed a production assistant's first message when this index was tail-injected.
                # Template strictness is a per-model-endpoint property: never assume a family is
                # tolerant because one model on it is. Pinned by
                # abstractcore tests/providers/test_openai_compatible_strict_system_messages.py.
                injected: list[Dict[str, Any]] = []
                if active_msg:
                    injected.append({"role": "system", "content": active_msg, "metadata": {"kind": "attachment_index"}})
                if session_msg:
                    injected.append({"role": "system", "content": session_msg, "metadata": {"kind": "attachment_index"}})
                messages = cleaned + injected
        except Exception:
            session_attachments = None

        fallback_enabled = _coerce_boolish(structured_output_fallback)
        base_params = dict(params)
        # The structured-output FALLBACK and REPAIR lanes below re-issue
        # generation from THIS snapshot, which is taken before the per-call
        # rider block mutates `params_for_call` — without its own injection
        # the run-level reasoning silently vanished exactly when the model
        # was struggling (adversary defect A, 2026-08-04). Same presence
        # gate: an explicit pin (including False) survives untouched.
        _maybe_inject_runtime_thinking(run=run, params=base_params)

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
            structured_model_name = "StructuredOutput"
            structured_response_model: Optional[Type[Any]] = None

            # Runtime-owned defaults (run-scoped): allow workflows/clients to set once at
            # run start via `run.vars["_runtime"]` and have all LLM calls inherit them.
            #
            # This is intentionally narrow: only copy explicit policy keys we support.
            try:
                runtime_ns = run.vars.get("_runtime") if isinstance(run.vars, dict) else None
            except Exception:
                runtime_ns = None
            if isinstance(runtime_ns, dict):
                audio_policy = runtime_ns.get("audio_policy")
                if "audio_policy" not in params_for_call and isinstance(audio_policy, str) and audio_policy.strip():
                    params_for_call["audio_policy"] = audio_policy.strip()

                stt_language = runtime_ns.get("stt_language")
                if stt_language is None:
                    stt_language = runtime_ns.get("audio_language")
                if "stt_language" not in params_for_call and isinstance(stt_language, str) and stt_language.strip():
                    params_for_call["stt_language"] = stt_language.strip()

            # Run-level reasoning rides the same seam (wire witness 2026-08-04:
            # the Agent structured post-pass and plain llm_call nodes emitted
            # ABSENT-effort calls inside medium runs). Explicit params win,
            # including False.
            _maybe_inject_runtime_thinking(run=run, params=params_for_call)

            if structured_requested:
                structured_model_name = (
                    str(response_schema_name).strip()
                    if isinstance(response_schema_name, str) and response_schema_name.strip()
                    else "StructuredOutput"
                )
                structured_response_model = _pydantic_model_from_json_schema(
                    response_schema, name=structured_model_name
                )
                params_for_call["response_model"] = structured_response_model

            structured_failed = False
            structured_error: Optional[str] = None
            structured_repair_used = False
            structured_repair_error: Optional[str] = None
            structured_validation_error: Optional[str] = None

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
                        "params": _observability_params(params_for_call),
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
                """DEPRECATED (2026-08-09) — retained for reference, no longer called.

                This raised `max_output_tokens` on `finish_reason=length` so the call
                could be retried with a bigger budget. Both of its cases are wrong:

                * DEFAULT POLICY. Callers send the model's maximum output budget unless
                  they say otherwise, and the bump is clamped by `min(bumped, cap)` where
                  `cap` falls back to the model's own capabilities. At the maximum that is
                  `min(2*max, max) = max` — a no-op. The "escalating retry" is therefore a
                  byte-identical re-run of a request that already failed, up to
                  `max_truncation_attempts` times.
                * EXPLICIT CALLER BUDGET. If an operator DID name a smaller budget, doubling
                  it silently overrides their number. The framework budget law is explicit
                  that what the caller named is never touched, high or low.

                A budget that is already maximal cannot be increased, so truncation is a
                fact to report, not a condition to retry. The call site now records the
                truncation, warns once, and lets the terminal block surface
                `finish_reason=length` to the caller. `retry_on_truncation` and
                `max_truncation_attempts` are deprecated with it and no longer change
                behaviour; `allow_truncation` is unaffected and still returns the partial.

                Kept (not deleted) so the rationale stays attached to the code, and because
                a provider that reports `length` for a reason OTHER than an exhausted output
                budget would need this discussion re-opened rather than re-invented.
                """
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

                cap: int | None = None
                if cap_raw is not None and not isinstance(cap_raw, bool):
                    try:
                        cap = max(256, int(cap_raw))
                    except Exception:
                        cap = None

                if cap is None:
                    hinted_cap = None
                    try:
                        limits = run.vars.get("_limits") if isinstance(run.vars, dict) else None
                        hinted_cap = limits.get("max_output_tokens") if isinstance(limits, dict) else None
                    except Exception:
                        hinted_cap = None
                    if hinted_cap is not None and not isinstance(hinted_cap, bool):
                        try:
                            hinted_i = int(hinted_cap)
                        except Exception:
                            hinted_i = 0
                        if hinted_i > 0:
                            cap = hinted_i

                if cap is None:
                    # Prefer model capabilities over arbitrary "unbounded" caps.
                    try:
                        caps = getattr(llm, "get_model_capabilities", None)
                        model_caps = caps() if callable(caps) else None
                        raw = model_caps.get("max_output_tokens") if isinstance(model_caps, dict) else None
                        if raw is None and isinstance(model_caps, dict):
                            raw = model_caps.get("max_tokens")
                        cap_i = int(raw) if raw is not None and not isinstance(raw, bool) else 0
                        if cap_i > 0:
                            cap = cap_i
                    except Exception:
                        pass

                # If the model capabilities are unknown, keep the cap effectively unbounded.
                if cap is None:
                    cap = 1_000_000

                updated["max_output_tokens"] = min(bumped, cap)
                updated.pop("max_tokens", None)
                return updated

            # DEPRECATED (2026-08-09): `retry_on_truncation` / `no_truncation` and
            # `max_truncation_attempts` / `truncation_max_attempts` no longer change
            # behaviour — truncation is reported, never retried with a raised budget
            # (see `_bump_max_output_tokens` for the full rationale). Still parsed so
            # existing payloads keep working, and a caller who explicitly sets one is
            # told it is inert rather than left to assume it took effect.
            retry_on_truncation_raw = payload.get("retry_on_truncation")
            if retry_on_truncation_raw is None:
                retry_on_truncation_raw = payload.get("no_truncation")
            retry_on_truncation = True
            if retry_on_truncation_raw is not None:
                retry_on_truncation = _coerce_boolish(retry_on_truncation_raw)
                logger.warning(
                    "LLM_CALL retry_on_truncation is DEPRECATED and ignored; truncation is "
                    "reported once and never retried with a raised budget "
                    "(a maximal budget cannot be raised; an operator budget must not be "
                    "overridden). Use allow_truncation=true to accept a partial answer.",
                    requested=retry_on_truncation,
                )

            allow_truncation_raw = payload.get("allow_truncation")
            if allow_truncation_raw is None:
                allow_truncation_raw = payload.get("allow_truncated")
            allow_truncation = _coerce_boolish(allow_truncation_raw) if allow_truncation_raw is not None else False

            # DEPRECATED with `retry_on_truncation` above: the loop now always makes
            # exactly one attempt, so this bound is inert. Parsed and warned-about only.
            max_truncation_attempts = 3
            raw_attempts = payload.get("max_truncation_attempts")
            if raw_attempts is None:
                raw_attempts = payload.get("truncation_max_attempts")
            if raw_attempts is not None and not isinstance(raw_attempts, bool):
                try:
                    max_truncation_attempts = max(1, int(raw_attempts))
                except Exception:
                    max_truncation_attempts = 3
                logger.warning(
                    "LLM_CALL max_truncation_attempts is DEPRECATED and ignored; the call is "
                    "attempted once and truncation is reported, never retried.",
                    requested=max_truncation_attempts,
                )

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

                    # DEPRECATED PATH, DELIBERATELY NOT TAKEN (see
                    # `_bump_max_output_tokens`). Truncation is reported once, loudly,
                    # and the call is NOT retried with a raised budget. Retrying could
                    # only ever do one of two wrong things:
                    #
                    #   * default policy — the budget already IS the model's max, so the
                    #     bump resolves to `min(2*max, max) = max` and the "retry" re-runs
                    #     a byte-identical request. Pure burn, up to 3x.
                    #   * explicit caller budget — raising it overrides a number the
                    #     operator NAMED, which the framework budget law forbids: what the
                    #     caller named is never touched.
                    #
                    # Either way the honest move is to surface `finish_reason=length` to
                    # the caller, which the terminal block below already does.
                    logger.warning(
                        "LLM_CALL output truncated (finish_reason=length); not retrying — "
                        "the budget cannot be raised above the model's maximum, and an "
                        "operator-set budget must not be overridden",
                        finish_reason=finish_reason,
                        attempt=attempt,
                        max_output_tokens=params_attempt.get("max_output_tokens"),
                    )
                    break

                if _finish_reason_is_truncation(last_finish_reason) and not allow_truncation:
                    budgets = ", ".join(
                        [
                            str(a.get("max_output_tokens"))
                            for a in truncation_attempts
                            if a.get("max_output_tokens") is not None
                        ][:6]
                    )
                    suffix = " …" if len(truncation_attempts) > 6 else ""
                    # NON-RETRYABLE: the handler already retried truncation internally with
                    # escalating budgets; the outer policy would replay the identical bump
                    # sequence (params rebuilt from the payload) — a deterministic x9 LLM burn.
                    return EffectOutcome.failed(
                        "LLM_CALL output was truncated (finish_reason=length). "
                        f"Attempted max_output_tokens: {budgets}{suffix}. "
                        "Increase max_output_tokens/max_out_tokens (or set allow_truncation=true).",
                        retryable=False,
                    )

                if had_truncation and not _finish_reason_is_truncation(last_finish_reason):
                    logger.warning(
                        "LLM_CALL output truncation resolved after retries",
                        attempts=len(truncation_attempts),
                        max_output_tokens=params_attempt.get("max_output_tokens"),
                    )

                # ABORTED GENERATION (operator 2026-08-02) — the OTHER way a
                # generation dies. finish_reason says "stop", so the loop above
                # never fires; the tell is a zero-token usage block next to
                # non-empty content and no tool calls. The provider cut the
                # generation mid-flight (client disconnect / server stop) and
                # DROPPED the tool call it was emitting; what comes back is the
                # preface, which downstream read as an ordinary assistant turn.
                # Record it on the step result so the LEDGER carries the fault
                # (`aborted_generation` rides `llm_call`'s durable result, and
                # the agent loops' parse nodes route it to a named retry rather
                # than to an intent guess). Not raised: the handler cannot ask
                # for a smaller unit of work — the loop can, and a blind replay
                # of a multi-minute generation is the cure costing more than
                # the disease.
                if _looks_like_aborted_generation(result):
                    logger.warning(
                        "LLM_CALL generation was aborted mid-flight; any tool call it carried was lost",
                        finish_reason=last_finish_reason,
                        usage=result.get("usage") if isinstance(result, dict) else None,
                    )
                    meta = result.get("metadata") if isinstance(result, dict) else None
                    if not isinstance(meta, dict):
                        meta = {}
                        if isinstance(result, dict):
                            result["metadata"] = meta
                    meta["generation_aborted"] = True
                    meta.setdefault("output_truncated", True)
                    meta.setdefault("truncation_kind", "aborted_generation")
                    runtime_observability["aborted_generation"] = {
                        "finish_reason": last_finish_reason,
                        "usage": result.get("usage") if isinstance(result, dict) else None,
                        "content_preview": (
                            str(result.get("content") or "")[:200] if isinstance(result, dict) else None
                        ),
                    }

                # Keep observability aligned with the actual params used.
                if had_truncation or len(truncation_attempts) > 1:
                    runtime_observability["llm_generate_kwargs"] = _jsonable(
                        {
                            "prompt": str(prompt or ""),
                            "messages": messages_for_call,
                            "system_prompt": system_prompt,
                            "media": media_for_call,
                            "tools": tools,
                            "params": _observability_params(params_attempt),
                            "structured_output_fallback": fallback_enabled,
                            "truncation_attempts": truncation_attempts,
                        }
                    )
            finally:
                if tmpdir is not None:
                    tmpdir.cleanup()

            if structured_requested and isinstance(result, dict):
                # Best-effort: when structured outputs fail (or providers ignore response_model),
                # try to parse the returned text into `data`, then validate it against the
                # requested schema. When `structured_output_fallback` is enabled, ask the model
                # to repair one invalid response before failing.
                parse_error: Optional[str] = None

                def _extract_candidate(current_result: Dict[str, Any]) -> tuple[Any, Optional[str], Optional[str]]:
                    try:
                        existing_data = current_result.get("data")
                    except Exception:
                        existing_data = None

                    content_text = current_result.get("content") if isinstance(current_result.get("content"), str) else None
                    if existing_data is not None:
                        return existing_data, content_text, None

                    if isinstance(content_text, str) and content_text.strip():
                        parsed: Any = None
                        err: Optional[str] = None
                        try:
                            from abstractruntime.visualflow_compiler.visual.builtins import data_parse_json

                            parsed = data_parse_json({"text": content_text, "wrap_scalar": True})
                        except Exception as e:
                            err = str(e)
                            parsed = None

                        if parsed is not None and not isinstance(parsed, dict):
                            parsed = {"value": parsed}
                        if parsed is not None:
                            current_result["data"] = parsed
                        return parsed, content_text, err

                    return None, content_text, None

                candidate, content_value, parse_error = _extract_candidate(result)
                if parse_error is not None:
                    meta = result.get("metadata")
                    if not isinstance(meta, dict):
                        meta = {}
                        result["metadata"] = meta
                    meta["_structured_output_parse_error"] = parse_error

                validated_data: Optional[Dict[str, Any]] = None
                if candidate is not None and structured_response_model is not None:
                    validated_data, structured_validation_error = _validate_structured_output_candidate(
                        candidate, response_model=structured_response_model
                    )
                    if validated_data is not None:
                        result["data"] = validated_data

                needs_repair = validated_data is None and (
                    parse_error is not None or structured_validation_error is not None
                )
                if needs_repair and fallback_enabled and structured_response_model is not None:
                    invalid_output = ""
                    if isinstance(content_value, str) and content_value.strip():
                        invalid_output = content_value.strip()
                    elif candidate is not None:
                        invalid_output = _stringify_structured_output_candidate(candidate)
                    repair_error_text = structured_validation_error or parse_error or "response did not satisfy the schema"
                    repair_prompt = _build_structured_output_repair_prompt(
                        original_task=user_text_for_context or str(prompt or ""),
                        schema=response_schema,
                        invalid_output=invalid_output,
                        error=repair_error_text,
                    )
                    repair_params = dict(base_params_attempt)
                    repair_params.pop("output", None)
                    repair_params.pop("glyph_compression", None)

                    logger.warning(
                        "LLM_CALL structured output invalid after parse; requesting repair",
                        error=repair_error_text,
                        model=structured_model_name,
                    )
                    structured_repair_used = True
                    repaired_result = llm.generate(
                        prompt=repair_prompt,
                        messages=None,
                        system_prompt=system_prompt,
                        media=None,
                        tools=None,
                        params=repair_params,
                    )
                    if not isinstance(repaired_result, dict):
                        return EffectOutcome.failed("LLM_CALL structured output repair returned a non-object result", retryable=False)

                    repaired_candidate, repaired_content, repaired_parse_error = _extract_candidate(repaired_result)
                    repaired_validated: Optional[Dict[str, Any]] = None
                    repaired_validation_error: Optional[str] = None
                    if repaired_candidate is not None:
                        repaired_validated, repaired_validation_error = _validate_structured_output_candidate(
                            repaired_candidate, response_model=structured_response_model
                        )

                    if repaired_validated is None:
                        structured_repair_error = repaired_validation_error or repaired_parse_error or "repair response did not satisfy the schema"
                        if isinstance(repaired_content, str) and repaired_content.strip():
                            structured_repair_error = (
                                f"{structured_repair_error}; repair content={_truncate_prompt_block(repaired_content, max_chars=400)}"
                            )
                        return EffectOutcome.failed(f"LLM_CALL structured output repair failed: {structured_repair_error}", retryable=False)

                    repaired_result["data"] = repaired_validated
                    result = repaired_result
                    structured_validation_error = None
                    parse_error = repaired_parse_error
                elif structured_validation_error is not None:
                    meta = result.get("metadata")
                    if not isinstance(meta, dict):
                        meta = {}
                        result["metadata"] = meta
                    meta["_structured_output_validation_error"] = structured_validation_error

            if isinstance(result, dict):
                meta = result.get("metadata")
                if not isinstance(meta, dict):
                    meta = {}
                    result["metadata"] = meta
                if structured_failed:
                    meta["_structured_output_fallback"] = {"used": True, "error": structured_error or ""}
                if structured_repair_used or structured_repair_error is not None:
                    meta["_structured_output_repair"] = {
                        "used": structured_repair_used,
                        "error": structured_repair_error or "",
                    }
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
                resolved_action = _runtime_resolved_action_from_generate_metadata(meta)
                if resolved_action is not None:
                    meta["_runtime_resolved_action"] = _jsonable(resolved_action)
                # The ledger must report the SERVICE, not the request: pair the
                # declared route with the served one and flag a disagreement.
                try:
                    _stamp_declared_route(
                        result,
                        declared_provider=provider,
                        declared_model=model,
                        endpoint_profile=params.get("_provider_endpoint_profile"),
                    )
                except Exception:  # noqa: BLE001 - observability must never fail a call
                    logger.debug("route disclosure stamping failed", exc_info=True)

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
            return EffectOutcome.failed(str(e), retryable=_llm_error_is_retryable(e))

    return _handler


# 4xx statuses that are legitimately TRANSIENT: request timeout, conflict during model
# load/warm-up (LM Studio JIT, vLLM), too-early, and rate limiting.
_TRANSIENT_4XX = (408, 409, 425, 429)
# Server-side Harmony parse failure of the model's own output (vllm#23567,
# openai/harmony#38/#80): a 400 that is actually a sampling race.
_HARMONY_ARTIFACT_RE = re.compile(
    r"unexpected tokens remaining in message header|HarmonyError|harmony generation artifact",
    re.IGNORECASE,
)


def _llm_error_is_retryable(exc: Exception) -> bool:
    """Classify LLM provider failures for the retry policy.

    Deterministic CLIENT errors — invalid request shape, authentication, unknown model,
    unsupported features, bad configuration — fail identically on every attempt; retrying them
    burns attempts, latency, and (for paid APIs) money before surfacing the same message.
    Classification order (adversarial-review hardened): (1) the HTTP STATUS CODE attribute when
    the raise site attached it (the one unambiguous fact an HTTP error carries — message prose
    is not a contract), (2) abstractcore exception types, (3) a conservative message fallback
    for our own providers' "API error (NNN)" / OpenAI SDK "Error code: NNN" dialects.
    """
    # Harmony generation artifact (gpt-oss on vLLM, maintainer directive
    # 2026-07-09 "maybe it's something our parser could self-correct"): the
    # server's strict openai-harmony parser 400s when the MODEL'S OWN sampled
    # output violates its template (unclosed `to=...` header). The request is
    # valid and a resample usually passes — transient, ALWAYS retryable. This
    # check precedes the status-code rule because these arrive as 400.
    if _HARMONY_ARTIFACT_RE.search(str(exc or "")):
        return True

    # CONFIGURATION refusals are deterministic. "No default provider/model is
    # configured" cannot become true between attempts, so retrying it triples
    # the wait a NEW USER endures before seeing the message that tells them
    # what to configure. Never retryable.
    try:
        from .llm_client import DefaultRouteProviderError, NoDefaultProviderConfigured

        if isinstance(exc, NoDefaultProviderConfigured):
            return False
        if isinstance(exc, DefaultRouteProviderError):
            # A pure attribution wrapper: it adds "and this came from your
            # config" to somebody else's failure, so it must not change that
            # failure's retryability. Classify the cause.
            cause = exc.__cause__
            return _llm_error_is_retryable(cause) if isinstance(cause, Exception) else True
    except ImportError:  # pragma: no cover - the module is a hard dependency here
        pass

    # An unresolvable provider NAME is deterministic for the same reason: the
    # registry does not gain a provider between attempts. Our own registry text
    # (`abstractcore/providers/registry.py`), matched conservatively.
    if str(exc or "").startswith("Unknown provider:"):
        return False

    # Structured prompt-cache failures (2026-07-13 bloc-seam adversary A-2):
    # binding verification errors (missing/mismatch/invalid/bare-string) and
    # capability refusals are DETERMINISTIC — the same params meet the same
    # refusal on every attempt; retrying burns the whole retry budget before
    # surfacing an identical message. Generic operation failures (I/O during
    # load/save) may be transient and keep the retryable default.
    try:
        from abstractcore.providers.base import PromptCacheError  # type: ignore

        if isinstance(exc, PromptCacheError):
            code = str(getattr(exc, "code", "") or "")
            if code.startswith("prompt_cache_binding_") or code == "prompt_cache_unsupported":
                return False
            return True
    except ImportError:
        pass

    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        if 400 <= status < 500 and status not in _TRANSIENT_4XX:
            return False
        return True

    try:
        from abstractcore.exceptions import (  # type: ignore
            AuthenticationError,
            InvalidRequestError,
            ModelNotFoundError,
            UnsupportedFeatureError,
        )

        if isinstance(exc, (InvalidRequestError, AuthenticationError, ModelNotFoundError, UnsupportedFeatureError)):
            return False
    except Exception:
        pass

    text = str(exc or "")
    m = re.search(r"API error \((\d{3})\)", text) or re.search(r"Error code: (\d{3})", text)
    if m:
        code = int(m.group(1))
        if 400 <= code < 500 and code not in _TRANSIENT_4XX:
            return False
    return True


_RISK_ROW_CACHE: Optional[Dict[str, Dict[str, Any]]] = None


def _risk_row_for_tool(name: str) -> Optional[Dict[str, Any]]:
    """The tool's inventory row (walled + core registry), for risk
    derivation. Cached per process (the registries are import-stable);
    unknown names return None and derive top-tier at the caller."""
    global _RISK_ROW_CACHE
    if _RISK_ROW_CACHE is None:
        rows: Dict[str, Dict[str, Any]] = {}
        try:
            from ...identity.tools import walled_tool_rows

            for r in walled_tool_rows():
                rows[str(r.get("name") or "")] = r
        except Exception:  # noqa: BLE001
            pass
        try:
            from .tool_inventory_facade import core_registry_tool_rows

            for r in core_registry_tool_rows():
                rows.setdefault(str(r.get("name") or ""), r)
        except Exception:  # noqa: BLE001
            pass
        _RISK_ROW_CACHE = rows
    return _RISK_ROW_CACHE.get(str(name or "").strip())


def _refiner_operator_email(run: "RunState") -> Optional[str]:
    """The gateway-injected operator-self address (send_email refiner seam,
    gateway c4692): a MODEL-UNWRITABLE `_runtime.operator_email` key that
    the door sets from config at run start (payload values popped first -
    a model-supplied operator_email never survives). Absent/blank/non-str
    -> None (the refiner then holds the ceiling: everything asks)."""
    if not isinstance(getattr(run, "vars", None), dict):
        return None
    rt = run.vars.get("_runtime")
    if not isinstance(rt, dict):
        return None
    val = rt.get("operator_email")
    if not isinstance(val, str) or not val.strip():
        return None
    return _norm_email(val)


def _norm_email(value: str) -> str:
    """Conservative normalization (fable5 c4691 P1-5): strip + NFC +
    lowercase ONLY. NO NFKC, NO confusable folding, NO IDN/punycode
    collapse - a homoglyph domain must NEVER compare equal to the real
    address (loose matching is the hole; strict is the defense)."""
    import unicodedata

    return unicodedata.normalize("NFC", str(value)).strip().lower()


def _send_email_recipient_refiner(call: Dict[str, Any], run: "RunState") -> str:
    """send_email_recipient@v1 (laurent dm#244): recipient == the registered
    operator address -> "auto"; ANY other recipient anywhere -> "ask".

    DENY-SAFE at every gap (fable5 c4691): empty/unresolved recipients ->
    ask (never vacuous-true all()); a WRAPPER-nested args shape -> ask (the
    P0-2 parser differential: the executor might unwrap to different
    recipients, so any wrapper presence forbids auto); display-name/group
    tokens compared VERBATIM (no bracket-address extraction - that is the
    readonly_git positional-spoof class); self-value absent -> ask; any
    exception -> ask. The refiner may only return "auto" (a downgrade);
    the caller adds nothing on "ask"."""
    try:
        self_addr = _refiner_operator_email(run)
        if not self_addr:
            return "ask"  # no self-value: the ceiling stands
        args = (call or {}).get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                return "ask"  # unparseable args: cannot prove self
        if not isinstance(args, dict):
            return "ask"
        # P0-2 differential guard: a wrapper-nested shape ("arguments" key)
        # may be unwrapped differently by the executor - never auto on it.
        if "arguments" in args:
            return "ask"
        from abstractcore.tools.comms_tools import _coerce_str_list

        recipients: List[str] = []
        for field in ("to", "cc", "bcc"):
            recipients.extend(_coerce_str_list(args.get(field)))
        if not recipients:
            return "ask"  # P0-1: no proven recipient -> never vacuous auto
        return "auto" if all(_norm_email(r) == self_addr for r in recipients) else "ask"
    except Exception:  # noqa: BLE001 - any refiner failure holds the ceiling
        return "ask"


# Per-call refiners keyed by the core-declared refiner-id (risk_facts.py
# KNOWN_REFINER_IDS). A refiner may only LOWER a call to auto; absence of
# a registered fn for a declared id is deny-safe (no downgrade -> ask).
_GIT_READ_VERBS = frozenset({"status", "log", "diff", "show", "ls-files"})
_GIT_SHELL_TOKENS = ("&&", "||", ";", "|", ">", ">>", "<", "<<", "&")
_EXEC_KNOWN_ARG_KEYS = frozenset({"command", "working_directory", "timeout", "capture_output"})


def _git_read_only_refiner(call: Dict[str, Any], run: "RunState") -> str:
    """git_read_only@v1 (converged contract c5028 R2; the abstractcode
    read-only-git PROOF ported to the approval point so the client's
    330-line shell twin can die): an execute_command call whose command is
    a PROVEN read-only git invocation -> "auto"; everything else -> "ask".

    Maximally conservative two-stage proof (deny-safe at every gap — the
    send_email refiner's discipline):
    - raw charset: shell substitution (`, $(, ${) or a second line -> ask;
    - shlex tokens: any shell operator -> ask; NO wrapper peeling (env/nohup
      -wrapped git is UNPROVEN -> ask; a wrong 'unproven' costs one prompt,
      never a silent mutation);
    - argv[0] basename must be exactly `git`; any global option BEFORE the
      verb (-C/-c/--git-dir...) -> ask;
    - verb in the read allowlist ({status,log,diff,show,ls-files} — the
      allowlist covers the positional-verb P0 class structurally: `git
      remote set-url`/`reflog expire` refuse because remote/reflog are not
      read verbs);
    - write/exec FLAGS on allowed verbs (--output*, --ext-diff, -o) -> ask
      (the corpus cases: `git log --output=<path>` writes, `git diff
      --ext-diff` executes);
    - unknown argument keys on the call -> ask (the wrapper-differential
      guard: an executor might interpret keys this proof did not see).
    Returns only "auto" (a downgrade) or "ask" (adds nothing)."""
    import shlex

    try:
        args = (call or {}).get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:  # noqa: BLE001
                return "ask"
        if not isinstance(args, dict):
            return "ask"
        if set(str(k) for k in args.keys()) - _EXEC_KNOWN_ARG_KEYS:
            return "ask"
        raw = str(args.get("command") or "").strip()
        if not raw or "\n" in raw or "\r" in raw:
            return "ask"
        if "`" in raw or "$(" in raw or "${" in raw:
            return "ask"
        try:
            argv = shlex.split(raw)
        except ValueError:
            return "ask"
        if not argv or any(tok in _GIT_SHELL_TOKENS for tok in argv):
            return "ask"
        if argv[0].rsplit("/", 1)[-1].lower() != "git":
            return "ask"
        rest = argv[1:]
        if not rest or rest[0].startswith("-"):
            return "ask"  # bare git / global options before the verb
        verb = rest[0].lower()
        if verb not in _GIT_READ_VERBS:
            return "ask"
        for a in rest[1:]:
            low = a.lower()
            if low.startswith("--output") or low == "--ext-diff" or low == "-o":
                return "ask"
        return "auto"
    except Exception:  # noqa: BLE001 - any doubt asks
        return "ask"


_TOOL_REFINERS: Dict[str, Any] = {
    "send_email_recipient@v1": _send_email_recipient_refiner,
    # Registered ahead of core's row declaration (dm#244 architecture: core
    # hosts the refiner-id on the tool row, runtime implements at the
    # approval point) — INERT until execute_command's inventory row carries
    # risk_refiner=git_read_only@v1 (asked of core with the c5028 fold).
    "git_read_only@v1": _git_read_only_refiner,
}


def _run_effect_seq(run: "RunState") -> int:
    """The run's effect-issuance counter (`_runtime.effect_seq`).

    Mirrors `abstractruntime.core.policy._effect_seq`: it advances in the same
    save that lands a step, so a crash-replay reads the SAME value while a
    genuine later issuance reads a higher one. That is precisely the
    unique-but-replay-stable property a tool-approval wait_key needs.
    """
    try:
        rt = run.vars.get("_runtime") if isinstance(getattr(run, "vars", None), dict) else None
        return int(rt.get("effect_seq", 0)) if isinstance(rt, dict) else 0
    except (TypeError, ValueError, AttributeError):
        return 0


def _effect_idempotency_key_from_tool_calls(tool_calls: Any) -> Optional[str]:
    """Recover the effect's idempotency key from the runtime-stamped call ids.

    `Runtime._ensure_tool_calls_have_runtime_ids` writes
    `runtime_call_id = "rtcall_{idempotency_key}_{index}"` onto every TOOL_CALLS
    entry, so the effect's own at-most-once identity is already on the payload
    the handler receives -- no new plumbing, and it is the exact identity the
    runtime uses to decide whether a replay is the same issuance.
    """
    prefix = "rtcall_"
    for tc in tool_calls or []:
        if not isinstance(tc, dict):
            continue
        raw = tc.get("runtime_call_id")
        text = str(raw).strip() if raw is not None else ""
        if not text.startswith(prefix) or "_" not in text[len(prefix) :]:
            continue
        candidate = text[len(prefix) :].rsplit("_", 1)[0].strip()
        if candidate:
            return candidate
    return None


def _execute_with_run_policy(tools: Any, calls: List[Dict[str, Any]], run: "RunState") -> Dict[str, Any]:
    """Per-run tool-policy consumer (restores the 2026-02-21 feature that
    regressed; two independent confirmations it was consumer-less — the
    2026-07-06 audit and flow's live gateway check 2026-07-20).

    `_runtime.tool_policy = {"auto_approve_tools": [...],
    "require_approval_tools": [...]}` (the wire shape thin clients send)
    OVERRIDES the executor's static approval policy for THIS run, both
    directions: a run-auto name skips the static gate (execute_approved),
    a run-require name forces the approval wait even where the static
    policy would auto-run. Applies only to approval-gated executors
    (execute_approved present) — plain executors have no approval concept
    to override. Malformed policy = static behavior (fail toward asking)."""
    pol = None
    if isinstance(run.vars, dict):
        rt = run.vars.get("_runtime")
        if isinstance(rt, dict):
            pol = rt.get("tool_policy")
    if (
        isinstance(pol, dict)
        and (pol.get("auto_approve_tools") or pol.get("require_approval_tools")
             or pol.get("auto_approve_max_risk_rank") is not None)
        and callable(getattr(tools, "execute_approved", None))
    ):
        auto = {str(t).strip() for t in (pol.get("auto_approve_tools") or []) if str(t).strip()}
        req = {str(t).strip() for t in (pol.get("require_approval_tools") or []) if str(t).strip()}
        # TIER CEILING (tool-tiers cycle-3; the c4343/c4352 commitment):
        # auto_approve_max_risk_tier=N auto-approves calls whose tool's
        # DERIVED risk_tier <= N. Names stay the finest grain: an explicit
        # require_approval_tools name forces the ask even under the
        # ceiling (require wins - the executor's standing contract).
        # Resolution is registry-side (derive_risk_tier over the served
        # row facts) - never a name heuristic; unknown tools derive 4
        # (fail-closed) and thus never ride a ceiling below 4.
        # RANK semantics (semantics c4589: tier = the WORD on the wire;
        # the ceiling compares INTEGERS = rank). ONE spelling: the _tier
        # alias was dropped SAME-DAY on code-tui's own release (c4614:
        # "this client consumes NO risk_* key today") - zero consumers
        # ever held it, so the alias died before it could become a
        # migration (the annotate-tier-field lesson, third application).
        ceiling = pol.get("auto_approve_max_risk_rank")
        if ceiling is not None:
            try:
                ceiling_n = int(ceiling)
            except (TypeError, ValueError):
                ceiling_n = None
            if ceiling_n is not None:
                try:
                    from .tool_inventory_facade import annotate_tool_rows, derive_risk_tier
                    from ..abstractcore.default_tools import get_default_toolsets  # type: ignore

                    for call in calls:
                        cname = str((call or {}).get("name") or "").strip()
                        if not cname or cname in req or cname in auto:
                            continue
                        row = _risk_row_for_tool(cname)
                        if row is not None and row.get("model_controlled_destination"):
                            # Band-neutral APPROVAL fact (core c4586 P1's
                            # law): the model chooses where output goes, so
                            # the prompt IS the exfiltration defense - a
                            # tier ceiling never silences it. Explicit
                            # name-list auto (above) remains the operator's
                            # override.
                            continue
                        tier_val = derive_risk_tier(row) if row is not None else 4
                        if tier_val <= ceiling_n:
                            auto.add(cname)
                except Exception:  # noqa: BLE001 - a failed derivation fails toward asking
                    pass
        # PER-CALL REFINERS (laurent dm#244, send_email_recipient@v1): a
        # tool whose served row declares risk_refiner may be LOWERED to
        # auto for THIS call when the refiner proves it safe (recipient ==
        # the operator's own address). Runs AFTER the ceiling and can
        # downgrade even a model_controlled_destination tool - that is the
        # designed per-argument exception (the model is not choosing a
        # dangerous destination when the destination is provably self).
        # require always wins; a missing/failed refiner adds nothing (ask).
        try:
            for call in calls:
                cname = str((call or {}).get("name") or "").strip()
                if not cname or cname in req or cname in auto:
                    continue
                row = _risk_row_for_tool(cname)
                refiner_id = str((row or {}).get("risk_refiner") or "").strip()
                fn = _TOOL_REFINERS.get(refiner_id) if refiner_id else None
                if fn is not None and fn(call, run) == "auto":
                    auto.add(cname)
        except Exception:  # noqa: BLE001 - a refiner-pass failure fails toward asking
            pass
        run_policy = ToolApprovalPolicy(auto_approve_tools=auto, require_approval_tools=req)
        try:
            requires = run_policy.requires_approval(calls)
        except Exception:
            requires = True
        if not requires:
            return tools.execute_approved(tool_calls=calls)
        return {
            "mode": "approval_required",
            "wait_reason": "user",
            "tool_calls": _jsonable(calls),
            "details": {
                "kind": "tool_approval",
                "policy": run_policy.describe(),
                "policy_source": "run",
            },
        }
    return tools.execute(tool_calls=calls)


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
            return EffectOutcome.failed("tool_calls requires payload.tool_calls (list)", retryable=False)
        allowed_tools_raw = payload.get("allowed_tools")
        allowlist_enabled = isinstance(allowed_tools_raw, list)
        allowed_tools: Set[str] = set()
        if allowlist_enabled:
            allowed_tools = {str(t) for t in allowed_tools_raw if isinstance(t, str) and t.strip()}

        if tools is None:
            return EffectOutcome.failed(
                "TOOL_CALLS requires a ToolExecutor; configure Runtime with "
                "MappingToolExecutor/AbstractCoreToolExecutor/PassthroughToolExecutor.",
                retryable=False,
            )

        original_call_count = len(tool_calls)

        # Always block non-dict tool call entries: passthrough hosts expect dicts and may crash otherwise.
        blocked_by_index: Dict[int, Dict[str, Any]] = {}
        pre_results_by_index: Dict[int, Dict[str, Any]] = {}
        planned: list[Dict[str, Any]] = []
        # call_id -> artifact id, for analyze_media calls whose file_path was
        # resolved from a session attachment. Lets the RESULT name the exact
        # `open_attachment(...)` that would put the image in front of the model
        # — advice the tool itself cannot give, because whether these bytes are
        # attachable is a session fact core does not have.
        analyze_media_artifact_by_call: Dict[str, str] = {}

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
            parsed = _coerce_positive_int(payload.get("max_attachment_bytes"))
            if parsed is not None:
                return parsed
            vars0 = getattr(run, "vars", None)
            runtime_ns = vars0.get("_runtime") if isinstance(vars0, dict) else None
            if isinstance(runtime_ns, dict):
                parsed = _coerce_positive_int(runtime_ns.get("max_attachment_bytes"))
                if parsed is not None:
                    return parsed
            parsed = _coerce_positive_int(os.getenv("ABSTRACTRUNTIME_MAX_ATTACHMENT_BYTES"))
            if parsed is not None:
                return parsed
            return 50 * 1024 * 1024

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

        def _register_text_as_attachment(
            *, session_id: str, text: str, filename: str, source: str
        ) -> Optional[Dict[str, Any]]:
            """Store an in-memory STRING (e.g. large command output) as a session attachment.

            Symmetric with `_register_read_file_as_attachment`, but the content is the given text
            rather than a file on disk (command output is ephemeral). Dedups by sha256 within the
            session so re-running the same command doesn't accumulate duplicate artifacts.
            """
            if artifact_store is None:
                return None
            sid = str(session_id or "").strip()
            if not sid:
                return None
            content = str(text or "").encode("utf-8", errors="replace")
            if len(content) > _max_attachment_bytes():
                # Too large even to store as an attachment; keep it out of the store.
                return None
            sha256 = hashlib.sha256(content).hexdigest()
            handle = str(filename or f"command-output-{sha256[:8]}.txt")
            content_type = "text/plain"

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
                if str(tags.get("sha256") or "") == sha256 and str(tags.get("path") or "") == handle:
                    return {
                        "artifact_id": str(getattr(m, "artifact_id", "") or ""),
                        "handle": handle,
                        "sha256": sha256,
                        "content_type": content_type,
                        "size_bytes": len(content),
                    }

            _ensure_session_memory_run_exists(session_id=sid)
            tags2: Dict[str, str] = {
                "kind": "attachment",
                "source": str(source or "tool.output"),
                "path": handle,
                "filename": handle,
                "session_id": sid,
                "sha256": sha256,
            }
            try:
                meta = artifact_store.store(content, content_type=content_type, run_id=str(rid), tags=tags2)
            except Exception:
                return None
            return {
                "artifact_id": str(getattr(meta, "artifact_id", "") or ""),
                "handle": handle,
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

        def _attachment_media_path_for(raw: Any) -> Optional[Tuple[str, str]]:
            """Local path for a session attachment addressed by id or name.

            Accepts the two spellings the runtime shows the model (artifact id
            and display filename) plus the handle, and requires an
            UNAMBIGUOUS single match — two attachments sharing a name is not a
            resolution, and guessing which one the operator meant is exactly
            the silent-wrong-answer class this fix exists to remove.
            """
            query = _normalize_attachment_query(raw)
            if not query or artifact_store is None:
                return None
            needle = query.lower()
            matches = []
            for entry in _get_session_attachments():
                handle = _normalize_attachment_query(entry.get("handle")).lower()
                if (
                    str(entry.get("artifact_id") or "").strip().lower() == needle
                    or handle == needle
                    or str(entry.get("filename") or "").strip().lower() == needle
                    or (handle and handle.endswith("/" + needle))
                ):
                    matches.append(entry)
            if len(matches) != 1:
                return None
            try:
                resolved = materialize_attachment_path(
                    artifact_store=artifact_store, entry=matches[0]
                )
            except Exception:
                return None
            if not resolved:
                return None
            return resolved, str(matches[0].get("artifact_id") or "")

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

            # Persistent shell tools (backlog 0220): the session registry namespace is a TRUST
            # BOUNDARY argument — always stamped with this run's id, overwriting anything the
            # model supplied, so one run can never reach another run's sessions. The stamped
            # value rides the approval wait's stored tool_calls, so the approved-resume path
            # executes with the same namespace.
            if name in (
                "shell_exec",
                "shell_write_stdin",
                "shell_close",
                "local_helper_start",
                "local_helper_status",
                "local_helper_stop",
            ):
                arguments["_registry_namespace"] = str(getattr(run, "run_id", "") or "")

            # Agora tools (hooks plan H8): the agent identity alias is a TRUST BOUNDARY
            # argument — always derived from run vars (`_runtime.agora_agent`), overwriting
            # anything the model supplied, so a model can never post as another agent.
            # The alias is a NON-secret name; the key stays in host env (AGORA_API_KEY__<ALIAS>).
            # Exact-name match against the runtime's own toolset (never a prefix guess:
            # a custom tool that merely starts with "agora_" must not receive the stamp).
            if name in _agora_tool_names():
                rv = run.vars.get("_runtime") if isinstance(run.vars, dict) else None
                raw_alias = rv.get("agora_agent") if isinstance(rv, dict) else None
                if raw_alias is None:
                    # No alias configured: strip anything the model supplied
                    # (a spoof attempt must not survive) — global identity.
                    arguments.pop("_agora_agent", None)
                else:
                    # Configured — stamp the RAW value, even blank/invalid:
                    # the toolset validates and fails LOUD (a blank alias must
                    # never silently fall back to the global identity).
                    arguments["_agora_agent"] = str(raw_alias)

            # Session-route stamp (vision-capability ruling, 2026-07-26): tools
            # that delegate sight (analyze_media) resolve the RUN's own route
            # FIRST — fallback config is solely for vision-less models. The
            # route is a TRUST BOUNDARY argument: always popped (a payload-
            # claimed route must never survive — derive-not-claim, the door
            # rule generalized), then injected from the run's own
            # `_runtime.provider/model` for the declared consumer tools only.
            # Absent route vars stamp NOTHING (core's graceful degradation:
            # unstamped = pre-ruling fallback behavior, byte-identical).
            arguments.pop("_session_route", None)
            if name in _SESSION_ROUTE_TOOL_NAMES:
                rv = run.vars.get("_runtime") if isinstance(run.vars, dict) else None
                provider = str(rv.get("provider") or "").strip() if isinstance(rv, dict) else ""
                model = str(rv.get("model") or "").strip() if isinstance(rv, dict) else ""
                if provider or model:
                    arguments["_session_route"] = {
                        "provider": provider or None,
                        "model": model or None,
                    }

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

            # Delegated sight addresses BOTH namespaces (2026-08-21 review).
            # `analyze_media` stats the path it is given, so a session
            # attachment — addressed by artifact id, or by the display name
            # this runtime itself put in the model's system message — always
            # failed with "does not exist". Resolved HERE, before the wall,
            # because the tool must actually run on real bytes (unlike
            # read_file, whose recovery can synthesize the result text).
            # A miss leaves the argument untouched: the file namespace then
            # answers, so a genuinely absent path refuses exactly as before.
            if name == "analyze_media":
                resolved_media = _attachment_media_path_for(arguments.get("file_path"))
                if resolved_media:
                    media_path, media_artifact_id = resolved_media
                    arguments = dict(arguments)
                    arguments["file_path"] = media_path
                    tc = {**tc, "arguments": arguments}
                    if media_artifact_id:
                        analyze_media_artifact_by_call[str(call_id)] = media_artifact_id

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
                result = _execute_with_run_policy(tools, host_tool_calls, run)
            except Exception as e:
                logger.error("TOOL_CALLS execution failed", error=str(e))
                return EffectOutcome.failed(str(e))

            mode = result.get("mode")
            if mode and mode != "executed":
                # ONE KEY PER APPROVAL INSTANCE. The old fallback was
                # `tool_calls:{run_id}:{node_id}` -- constant for every
                # approval round of an agent node (an agent loops on the same
                # node), so a driver that deduplicates by key answered the
                # first approval and parked the run forever on the second
                # (live 2026-07-31, multiagent/bugfix). An explicit
                # payload/executor key still wins; the derived key is the
                # default for BOTH the run-policy branch (which supplies none)
                # and the plain approval branch.
                wait_key = (
                    payload.get("wait_key")
                    or result.get("wait_key")
                    or build_tool_approval_wait_key(
                        run_id=run.run_id,
                        node_id=run.current_node,
                        effect_idempotency_key=_effect_idempotency_key_from_tool_calls(tool_calls),
                        effect_seq=_run_effect_seq(run),
                        tool_calls=host_tool_calls,
                    )
                )
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
                seg_result = _execute_with_run_policy(tools, seg_calls, run)
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

            def _offload_text(text: str, *, source: str) -> tuple[Optional[str], bool, int]:
                """Offload a large tool-output string to a session artifact (backlog 0215).

                Returns (artifact_id, too_large, n_bytes):
                - small (<= max_inline_bytes): (None, False, n) — caller keeps it inline.
                - offloadable (<= max_attachment_bytes): (artifact_id, False, n) — stored; caller
                  replaces the inline blob with a preview + open_attachment handle.
                - too large (> max_attachment_bytes): (None, True, n) — NOT stored and MUST NOT be
                  kept inline; caller surfaces an explicit, actionable notice so the agent/user
                  decides (narrow the command, redirect to a file). This is a marked decision point,
                  never a silent drop (ADR-0026).
                """
                try:
                    n = len(str(text or "").encode("utf-8"))
                except Exception:
                    n = len(str(text or ""))
                if n <= max_inline_bytes:
                    return (None, False, n)
                if artifact_store is None or not sid_str:
                    return (None, False, n)  # cannot offload without a store/session; keep inline
                if n > _max_attachment_bytes():
                    return (None, True, n)
                att = _register_text_as_attachment(session_id=sid_str, text=str(text), filename="", source=source)
                aid = str(att.get("artifact_id") or "").strip() if att else ""
                return (aid or None, False, n)

            def _too_large_notice(n_bytes: int, *, what: str) -> str:
                cap = _max_attachment_bytes()
                return (
                    f"\n\n[{what} was {n_bytes} bytes, exceeding the {cap}-byte retention limit — "
                    f"it was NOT stored to keep the run record bounded. Re-run narrowing the output "
                    f"(e.g. pipe through head/grep, or redirect to a file and read a bounded range).]"
                )

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

                # Delegated sight returns ONE bounded reading. When the image is a
                # session attachment the model has a strictly better option than
                # re-reading someone else's summary — look at it itself — and this
                # is the only layer that knows the exact call, because whether these
                # bytes are attachable is a session fact the tool does not have.
                if seg_item.get("name") == "analyze_media" and isinstance(r_out, dict):
                    call_key = str(
                        r_out.get("call_id")
                        or (seg_item.get("tc") or {}).get("call_id")
                        or ""
                    )
                    aid_media = analyze_media_artifact_by_call.get(call_key)
                    out_media = r_out.get("output")
                    if (
                        aid_media
                        and isinstance(out_media, str)
                        and out_media
                        and r_out.get("success") is not False
                    ):
                        r_out["output"] = out_media + (
                            f'\n(or see it yourself: open_attachment(artifact_id="{aid_media}"))'
                        )

                # execute_command output offload (backlog 0215): the full stdout/stderr live in the
                # result dict and land in the durable ledger. Symmetric with read_file, offload a
                # large stdout OR stderr to a session artifact and keep the ledger lean + give the
                # model an open_attachment handle. Applies regardless of exit code (a failed-but-
                # verbose command is exactly when offload matters); the bounded `rendered` preview
                # the model sees is preserved. Output beyond the retention cap is surfaced as an
                # explicit narrow-the-command notice, never silently kept inline or dropped.
                if seg_item.get("name") == "execute_command":
                    out_obj = r_out.get("output") if isinstance(r_out, dict) else None
                    if isinstance(out_obj, dict):
                        for field in ("stdout", "stderr"):
                            val = out_obj.get(field)
                            if not isinstance(val, str) or not val:
                                continue
                            aid, too_large, n_bytes = _offload_text(val, source=f"tool.execute_command.{field}")
                            if aid:
                                out_obj[field] = ""
                                out_obj[f"{field}_offloaded_artifact_id"] = aid
                                hint = (
                                    f"\n\n(Full {field} was {n_bytes} bytes; stored as attachment id={aid}. "
                                    f"Use open_attachment(artifact_id='{aid}', start_line=1, end_line=200) for bounded excerpts.)"
                                )
                                rendered = out_obj.get("rendered")
                                if isinstance(rendered, str):
                                    out_obj["rendered"] = rendered + hint
                            elif too_large:
                                out_obj[field] = ""
                                notice = _too_large_notice(n_bytes, what=f"Command {field}")
                                rendered = out_obj.get("rendered")
                                out_obj["rendered"] = (rendered if isinstance(rendered, str) else "") + notice
                    results_by_index[idx] = _jsonable(r_out)
                    continue

                if seg_item.get("name") != "read_file":
                    # Generic offload for ANY other host tool that returns a large string output
                    # (backlog 0215: "any output of any tool execution"). Structured/dict outputs are
                    # left untouched (we can't know which field is the payload); read_file and
                    # execute_command have dedicated branches above.
                    generic_out = r_out.get("output") if isinstance(r_out, dict) else None
                    if isinstance(generic_out, str) and generic_out:
                        aid, too_large, n_bytes = _offload_text(generic_out, source=f"tool.{seg_item.get('name') or 'output'}")
                        if aid:
                            r_out["output"] = (
                                f"(Output was {n_bytes} bytes; stored as attachment id={aid}. "
                                f"Use open_attachment(artifact_id='{aid}', start_line=1, end_line=200) for bounded excerpts.)"
                            )
                            r_out["output_offloaded_artifact_id"] = aid
                        elif too_large:
                            r_out["output"] = _too_large_notice(n_bytes, what="Tool output").strip()
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


def _coerce_boolish(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return float(value) != 0.0
        except Exception:
            return default
    if isinstance(value, str):
        text = value.strip().lower()
        if not text:
            return default
        if text in {"false", "0", "no", "off", "disable", "disabled"}:
            return False
        if text in {"true", "1", "yes", "on", "enable", "enabled"}:
            return True
    return default


def _normalize_model_residency_operation(raw: Any) -> str:
    op = str(raw or "list_loaded").strip().lower().replace("-", "_")
    if op in {"list", "loaded", "list_loaded", "list_models", "loaded_models"}:
        return "list_loaded"
    if op in {"load", "warm", "preload"}:
        return "load"
    if op in {"unload", "release", "evict"}:
        return "unload"
    if op in {"lock", "lock_model", "lock_residency"}:
        return "lock"
    if op in {"unlock", "unlock_model", "unlock_residency"}:
        return "unlock"
    return op


def _model_residency_not_found(payload: Dict[str, Any]) -> bool:
    try:
        if int(payload.get("status_code")) == 404:
            return True
    except Exception:
        pass
    for key in ("code", "type", "reason"):
        raw = payload.get(key)
        if isinstance(raw, str) and "not_found" in raw.lower():
            return True
    error = payload.get("error")
    if isinstance(error, dict):
        return _model_residency_not_found(error)
    if isinstance(error, str):
        text = error.lower().replace("-", "_").replace(" ", "_")
        return "not_found" in text
    return False


def _model_residency_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in {"1", "true", "yes", "on", "loaded", "resident"}:
            return True
        if raw in {"0", "false", "no", "off", "not_loaded", "unloaded", "not_found"}:
            return False
    return None


def _model_residency_load_verified(result: Dict[str, Any]) -> Optional[bool]:
    runtime = result.get("runtime")
    if isinstance(runtime, dict):
        for key in ("loaded", "resident", "provider_resident"):
            parsed = _model_residency_bool(runtime.get(key))
            if parsed is not None:
                return parsed
        state = str(runtime.get("state") or runtime.get("provider_state") or "").strip().lower()
        if state in {"loaded", "resident", "provider_loaded"}:
            return True
        if state in {"not_loaded", "provider_not_loaded", "provider_residency_unknown", "unloaded", "not_found"}:
            return False

    for key in ("loaded", "resident", "unloaded"):
        parsed = _model_residency_bool(result.get(key))
        if parsed is not None:
            return (not parsed) if key == "unloaded" else parsed
    return None


def _append_model_residency_warning(result: Dict[str, Any], warning: str) -> None:
    existing = result.get("warnings")
    if isinstance(existing, list):
        warnings = existing
    else:
        warnings = []
        result["warnings"] = warnings
    if warning not in [str(item) for item in warnings]:
        warnings.append(warning)


def _soft_model_residency_failure(*, operation: str, message: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "ok": False,
        "success": False,
        "operation": operation,
        "error": message,
        "warnings": [message],
        "status_hint": "warning",
        "degraded": True,
        "affected_models": [],
    }
    if payload:
        out["diagnostics"] = _jsonable(payload)
    return out


def make_model_residency_handler(*, control: Any) -> EffectHandler:
    def _handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        _ = (run, default_next_node)
        payload = dict(effect.payload or {})
        operation = _normalize_model_residency_operation(payload.get("operation"))
        required = _coerce_boolish(payload.get("required"), default=False)

        if operation not in {"list_loaded", "load", "unload", "lock", "unlock"}:
            result = _soft_model_residency_failure(
                operation=operation,
                message=f"Unsupported model_residency operation: {operation!r}",
            )
            return EffectOutcome.failed(result["error"]) if required else EffectOutcome.completed(_jsonable(result))

        method_name = {
            "list_loaded": "list_model_residency",
            "load": "load_model_residency",
            "unload": "unload_model_residency",
            "lock": "lock_model_residency",
            "unlock": "unlock_model_residency",
        }[operation]
        method = getattr(control, method_name, None)
        if not callable(method):
            result = _soft_model_residency_failure(
                operation=operation,
                message="Configured AbstractCore client does not expose model residency controls.",
            )
            return EffectOutcome.failed(result["error"]) if required else EffectOutcome.completed(_jsonable(result))

        options = payload.get("options")
        call_kwargs: Dict[str, Any] = {}
        for key in ("task", "provider", "model", "runtime_id", "base_url", "timeout_s", "provider_api_key", "api_key"):
            raw = payload.get(key)
            if raw is not None and raw != "":
                call_kwargs[key] = raw
        if isinstance(options, dict):
            call_kwargs["options"] = dict(options)
        if "pin" in payload:
            call_kwargs["pin"] = _coerce_boolish(payload.get("pin"), default=True)

        try:
            if operation == "list_loaded":
                result = method(
                    task=call_kwargs.pop("task", None),
                    provider=call_kwargs.pop("provider", None),
                    model=call_kwargs.pop("model", None),
                    **call_kwargs,
                )
            elif operation == "load":
                task = call_kwargs.pop("task", None)
                if not isinstance(task, str) or not task.strip():
                    raise ValueError("model_residency load requires payload.task")
                result = method(
                    task=task,
                    provider=call_kwargs.pop("provider", None),
                    model=call_kwargs.pop("model", None),
                    options=call_kwargs.pop("options", None),
                    pin=call_kwargs.pop("pin", True),
                    **call_kwargs,
                )
            elif operation in {"lock", "unlock"}:
                # `lock_model_residency`/`unlock_model_residency` accept a
                # payload mapping and/or kwargs (kwargs win); the selector
                # fields (and `task`, which the clients validate) ride as
                # kwargs here. `options`/`pin` are load/unload concerns, not
                # lock request fields.
                call_kwargs.pop("options", None)
                call_kwargs.pop("pin", None)
                result = method(**call_kwargs)
            else:
                if "force" in payload:
                    call_kwargs["force"] = _coerce_boolish(payload.get("force"), default=False)
                result = method(
                    task=call_kwargs.pop("task", None),
                    runtime_id=call_kwargs.pop("runtime_id", None),
                    provider=call_kwargs.pop("provider", None),
                    model=call_kwargs.pop("model", None),
                    options=call_kwargs.pop("options", None),
                    **call_kwargs,
                )
        except Exception as e:
            result = _soft_model_residency_failure(operation=operation, message=str(e))

        if not isinstance(result, dict):
            result = {"ok": True, "operation": operation, "data": _jsonable(result)}
        else:
            result = _jsonable(result)
            if not isinstance(result, dict):
                result = {"ok": True, "operation": operation, "data": result}

        result.setdefault("operation", operation)
        if operation == "list_loaded":
            result.setdefault("ok", True)
            if "models" not in result and isinstance(result.get("data"), list):
                result["models"] = result.get("data")
            result.setdefault("success", result.get("ok") is not False)
            result.setdefault("affected_models", result.get("models") if isinstance(result.get("models"), list) else [])
        elif operation == "load":
            result.setdefault("ok", True)
            if result.get("ok") is not False:
                loaded = _model_residency_load_verified(result)
                if loaded is not True:
                    warning = (
                        "model_residency load did not verify loaded provider residency"
                        if loaded is None
                        else "model_residency load completed without a loaded model"
                    )
                    result["ok"] = False
                    result.setdefault("error", warning)
                    _append_model_residency_warning(result, warning)
                    result.setdefault("status_hint", "warning")
                    result.setdefault("degraded", True)
            result.setdefault("success", result.get("ok") is not False)
            if "affected_models" not in result:
                result["affected_models"] = [result["runtime"]] if isinstance(result.get("runtime"), dict) else []
        elif operation in {"lock", "unlock"}:
            result.setdefault("ok", True)
            result.setdefault("success", result.get("ok") is not False)
            result.setdefault("affected_models", [])
        elif operation == "unload":
            result.setdefault("unloaded", False)
            if result.get("ok") is False and _model_residency_not_found(result) and not required:
                warning = str(result.get("error") or "Requested runtime was not resident.")
                result = {
                    "ok": True,
                    "success": True,
                    "operation": "unload",
                    "unloaded": False,
                    "warnings": [warning],
                    "affected_models": [],
                    "diagnostics": {"source": "abstractruntime", "not_found": True, "original": result},
                }
            else:
                result.setdefault("ok", True)
                result.setdefault("success", result.get("ok") is not False)
                if "affected_models" not in result:
                    result["affected_models"] = [result["runtime"]] if isinstance(result.get("runtime"), dict) else []

        if result.get("ok") is False and not required:
            result.setdefault("status_hint", "warning")
            result.setdefault("degraded", True)
            result["success"] = False

        if result.get("ok") is False and required:
            return EffectOutcome.failed(str(result.get("error") or "model_residency failed"))
        return EffectOutcome.completed(_jsonable(result))

    return _handler


class _PreApprovedExecutorView:
    """Executor view for AUTHORED deterministic invocations (TOOL_INVOKE).

    Exposes ONLY `execute`, which routes to the inner executor's
    `execute_approved` when present (the post-approval path: same tools,
    same limits, no gate) and plain `execute` otherwise. Deliberately does
    NOT expose `execute_approved` itself, so the shared handler machinery
    sees a plain executor and its approval logic never engages - the trust
    decision rides the EFFECT CLASS (host-constructed from node types),
    never a payload field a model could stamp (commons c4204 ruling)."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def execute(self, *, tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        approved = getattr(self._inner, "execute_approved", None)
        if callable(approved):
            return approved(tool_calls=tool_calls)
        return self._inner.execute(tool_calls=tool_calls)


def make_tool_invoke_handler(
    *,
    tools: Optional[ToolExecutor] = None,
    artifact_store: Optional[ArtifactStore] = None,
    run_store: Optional[RunStore] = None,
) -> EffectHandler:
    """TOOL_INVOKE: one AUTHORED tool call, executed without the approval
    gate (flow's deterministic fixed-verb nodes, laurent dm#49).

    Payload contract (flow c4206): {name, arguments, result_key?} - ONE
    call per effect (a fixed-verb node binds exactly one verb; batches are
    the agent lane's shape). Everything else is the TOOL_CALLS machinery
    verbatim by delegation: workspace walls, argument rewriting, artifact
    offload, idempotency - the ONLY delta is the executor view above, so
    wall/rewrite fixes land on both lanes automatically. The result under
    result_key carries the SAME envelope as TOOL_CALLS (results[0].output
    = the tool's raw output, verbatim) so flow's existing first.output ->
    pin mapping works unchanged."""
    if tools is None:
        delegate = make_tool_calls_handler(
            tools=None, artifact_store=artifact_store, run_store=run_store)
    else:
        delegate = make_tool_calls_handler(
            tools=_PreApprovedExecutorView(tools),
            artifact_store=artifact_store, run_store=run_store)

    def _handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        payload = dict(effect.payload or {})
        name = str(payload.get("name") or "").strip()
        if not name:
            return EffectOutcome.failed("tool_invoke requires payload.name (the fixed verb)", retryable=False)
        arguments = payload.get("arguments")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            return EffectOutcome.failed("tool_invoke payload.arguments must be an object", retryable=False)
        inner_payload: Dict[str, Any] = {
            "tool_calls": [{"name": name, "arguments": arguments}],
        }
        for key in ("result_key", "allowed_tools"):
            if payload.get(key) is not None:
                inner_payload[key] = payload[key]
        inner_effect = Effect(
            type=EffectType.TOOL_CALLS,
            payload=inner_payload,
            result_key=effect.result_key,
        )
        return delegate(run, inner_effect, default_next_node)

    return _handler


def build_effect_handlers(
    *,
    llm: AbstractCoreLLMClient,
    tools: ToolExecutor = None,
    artifact_store: Optional[ArtifactStore] = None,
    run_store: Optional[RunStore] = None,
) -> Dict[EffectType, Any]:
    # Case-1 seam (converged 2026-07-26): thread the llm_client's
    # endpoint-profile resolver into tool execution so core's session-route
    # path can construct gateway-registered endpoint:* providers. The getter
    # is LATE-BOUND over the client (gateway calls
    # set_provider_endpoint_profile_resolver after construction); the attach
    # walks delegate chains so wrapped executors (approval views, MCP
    # delegation) reach the inner MappingToolExecutor.
    if tools is not None:
        from .tool_executor import attach_endpoint_profile_resolver_getter

        def _resolver_from_llm() -> Any:
            # Private attribute on Local/MultiLocal clients; the REMOTE
            # client carries only the PUBLIC `resolve_provider_endpoint_profile`
            # (gateway's fallback attach sets it, bundle_host.py:90-100) —
            # reading only the private name left the whole remote lane dark
            # (route-context adversary P1-2, 2026-07-26).
            r = getattr(llm, "_provider_endpoint_profile_resolver", None)
            if r is None:
                r = getattr(llm, "resolve_provider_endpoint_profile", None)
            return r

        attach_endpoint_profile_resolver_getter(tools, _resolver_from_llm)
    return {
        EffectType.LLM_CALL: make_llm_call_handler(llm=llm, artifact_store=artifact_store),
        EffectType.MODEL_RESIDENCY: make_model_residency_handler(control=llm),
        EffectType.TOOL_CALLS: make_tool_calls_handler(tools=tools, artifact_store=artifact_store, run_store=run_store),
        EffectType.TOOL_INVOKE: make_tool_invoke_handler(tools=tools, artifact_store=artifact_store, run_store=run_store),
    }
