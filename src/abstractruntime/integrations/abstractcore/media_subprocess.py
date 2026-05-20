"""Subprocess worker for local AbstractCore media generation.

Native Apple/Metal backends can abort the interpreter on driver-level failures. Running
local image generation in a worker process keeps the gateway/runtime parent alive and
turns those failures into ordinary step errors.
"""

from __future__ import annotations

import base64
import json
import sys
import traceback
from dataclasses import asdict, is_dataclass
from types import SimpleNamespace
from typing import Any, Dict


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (bytes, bytearray)):
        return {"data_b64": base64.b64encode(bytes(value)).decode("ascii")}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if is_dataclass(value):
        return _jsonable(asdict(value))
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _jsonable(model_dump())
    to_dict = getattr(value, "dict", None)
    if callable(to_dict):
        return _jsonable(to_dict())
    return str(value)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _serialize_generated_item(item: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "modality": _field(item, "modality", None),
        "task": _field(item, "task", None),
        "artifact_ref": _field(item, "artifact_ref", None),
        "content_type": _field(item, "content_type", None),
        "format": _field(item, "format", None),
        "backend_id": _field(item, "backend_id", None),
        "provider": _field(item, "provider", None),
        "model": _field(item, "model", None),
        "metadata": _field(item, "metadata", {}) or {},
    }
    data = _field(item, "data", None)
    if isinstance(data, (bytes, bytearray)):
        out["data_b64"] = base64.b64encode(bytes(data)).decode("ascii")
    elif data is not None:
        out["data"] = _jsonable(data)
    return {k: _jsonable(v) for k, v in out.items() if v is not None}


def _serialize_generation_issue(issue: Any) -> Dict[str, Any]:
    return {
        "modality": str(_field(issue, "modality", "") or ""),
        "task": str(_field(issue, "task", "") or ""),
        "message": str(_field(issue, "message", "") or ""),
        "type": str(_field(issue, "type", "error") or "error"),
        "metadata": _jsonable(_field(issue, "metadata", {}) or {}),
    }


def _artifact_ref_from_raw(raw: Dict[str, Any]) -> Dict[str, Any] | None:
    artifact_id = raw.get("artifact_id") or raw.get("$artifact")
    if not isinstance(artifact_id, str) or not artifact_id.strip():
        ref = raw.get("artifact_ref")
        if isinstance(ref, dict):
            artifact_id = ref.get("artifact_id") or ref.get("$artifact")
            if isinstance(artifact_id, str) and artifact_id.strip():
                return {
                    "$artifact": artifact_id.strip(),
                    "artifact_id": artifact_id.strip(),
                    "content_type": str(ref.get("content_type") or raw.get("content_type") or "application/octet-stream"),
                    "size_bytes": ref.get("size_bytes") or raw.get("size_bytes"),
                }
        return None
    return {
        "$artifact": artifact_id.strip(),
        "artifact_id": artifact_id.strip(),
        "content_type": str(raw.get("content_type") or raw.get("mime_type") or "application/octet-stream"),
        "size_bytes": raw.get("size_bytes"),
    }


def _bytes_from_raw(raw: Any) -> tuple[bytes | None, Dict[str, Any] | None, Dict[str, Any]]:
    if isinstance(raw, (bytes, bytearray)):
        return bytes(raw), None, {}
    if not isinstance(raw, dict):
        return None, None, {"raw": _jsonable(raw)}

    artifact_ref = _artifact_ref_from_raw(raw)
    for key in ("data", "content", "bytes"):
        value = raw.get(key)
        if isinstance(value, (bytes, bytearray)):
            return bytes(value), artifact_ref, {k: _jsonable(v) for k, v in raw.items() if k not in {key}}
    for key in ("data_base64", "b64_json", "base64", "data_b64"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            try:
                return base64.b64decode(value), artifact_ref, {k: _jsonable(v) for k, v in raw.items() if k not in {key}}
            except Exception:
                continue
    return None, artifact_ref, _jsonable(raw) if isinstance(_jsonable(raw), dict) else {}


def _image_kwargs_from_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    ignored = {
        "modality",
        "task",
        "format",
        "content_type",
        "mime_type",
        "response_format",
        "run_id",
        "tags",
    }
    return {str(k): v for k, v in spec.items() if str(k) not in ignored and v is not None}


def _run_image_spec(registry: Any, *, prompt: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    fmt = str(spec.get("format") or "png")
    content_type = str(spec.get("content_type") or spec.get("mime_type") or f"image/{fmt}")
    kwargs = _image_kwargs_from_spec(spec)
    raw = registry.vision.t2i(prompt, **kwargs)
    data, artifact_ref, metadata = _bytes_from_raw(raw)
    return {
        "modality": "image",
        "task": "image_generation",
        "data": data,
        "artifact_ref": artifact_ref,
        "content_type": content_type,
        "format": fmt,
        "backend_id": getattr(registry.vision, "backend_id", None),
        "provider": str(spec.get("provider") or getattr(registry.vision, "backend_id", None) or "abstractvision"),
        "model": str(spec.get("model") or ""),
        "metadata": metadata,
    }


def _serialize_response(resp: Any) -> Dict[str, Any]:
    outputs_raw = _field(resp, "outputs", {}) or {}
    outputs: Dict[str, Any] = {}
    if isinstance(outputs_raw, dict):
        for modality, items in outputs_raw.items():
            if isinstance(items, list):
                outputs[str(modality)] = [_serialize_generated_item(item) for item in items]

    resources_raw = _field(resp, "resources", {}) or {}
    resources: Dict[str, Any] = {}
    if isinstance(resources_raw, dict):
        for modality, items in resources_raw.items():
            if isinstance(items, list):
                resources[str(modality)] = [_jsonable(item) for item in items]

    return {
        "content": _field(resp, "content", None),
        "outputs": outputs,
        "resources": resources,
        "warnings": [_serialize_generation_issue(item) for item in (_field(resp, "warnings", []) or [])],
        "errors": [_serialize_generation_issue(item) for item in (_field(resp, "errors", []) or [])],
        "metadata": _jsonable(_field(resp, "metadata", {}) or {}),
    }


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object request.")

        provider = str(payload.get("provider") or "").strip()
        model = str(payload.get("model") or "").strip()
        llm_kwargs = payload.get("llm_kwargs") if isinstance(payload.get("llm_kwargs"), dict) else {}
        specs = payload.get("specs") if isinstance(payload.get("specs"), list) else []
        prompt = str(payload.get("prompt") or "")
        if not specs:
            raise ValueError("Missing output specs for local media subprocess.")

        from abstractcore.core.multimodal_generation import MultimodalGenerateResponse  # type: ignore
        from abstractcore.capabilities.registry import CapabilityRegistry  # type: ignore

        owner = SimpleNamespace(config=dict(llm_kwargs))
        registry = CapabilityRegistry(owner)
        response = MultimodalGenerateResponse(
            metadata={
                "media_only": True,
                "subprocess": True,
                "runtime_provider": provider or None,
                "runtime_model": model or None,
                "execution_mode": "local_one_shot_subprocess",
            }
        )
        for spec in specs:
            if not isinstance(spec, dict):
                raise ValueError("Output spec must be an object.")
            modality = str(spec.get("modality") or "").strip().lower()
            task = str(spec.get("task") or "").strip().lower()
            if modality != "image" or (task and task not in {"image_generation", "t2i", "text_to_image"}):
                raise ValueError(f"Unsupported subprocess media spec: modality={modality!r} task={task!r}")
            response.add_output("image", _run_image_spec(registry, prompt=prompt, spec=dict(spec)))

        print(json.dumps({"ok": True, "response": _serialize_response(response)}, ensure_ascii=False), flush=True)
        return 0
    except BaseException as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
