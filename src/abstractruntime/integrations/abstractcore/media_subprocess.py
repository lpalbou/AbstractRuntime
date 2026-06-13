"""Subprocess worker for local AbstractCore media generation.

Native Apple/Metal backends can abort the interpreter on driver-level failures. Running
local media generation in a worker process keeps the gateway/runtime parent alive and
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
        "prompt",
        "count",
        "n",
        "seeds",
    }
    return {str(k): v for k, v in spec.items() if str(k) not in ignored and v is not None}


def _video_kwargs_from_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    ignored = {
        "modality",
        "task",
        "format",
        "content_type",
        "mime_type",
        "response_format",
        "run_id",
        "tags",
        "count",
        "n",
        "seeds",
    }
    return {str(k): v for k, v in spec.items() if str(k) not in ignored and v is not None}


def _spec_generation_count_seeds(spec: Dict[str, Any]) -> tuple[int, list[int] | None]:
    raw_seeds = spec.get("seeds")
    default_count = len(raw_seeds) if isinstance(raw_seeds, list) and raw_seeds else 1
    raw_count = spec.get("count", spec.get("n", default_count))
    try:
        count = int(raw_count) if raw_count is not None else 1
    except Exception as exc:
        raise ValueError("Output count must be an integer >= 1.") from exc
    if count < 1:
        raise ValueError("Output count must be >= 1.")

    if raw_seeds is None:
        return count, None
    if isinstance(raw_seeds, (str, bytes, bytearray)) or not isinstance(raw_seeds, list):
        raise ValueError("Output seeds must be a list of integers.")
    seeds: list[int] = []
    for value in raw_seeds:
        try:
            seeds.append(int(value))
        except Exception as exc:
            raise ValueError("Output seeds must be integers.") from exc
    if not seeds:
        raise ValueError("Output seeds cannot be empty.")
    return count, seeds


def _media_path_from_item(item: Any) -> str:
    if isinstance(item, str) and item.strip():
        return item.strip()
    if isinstance(item, dict):
        for key in ("file_path", "filePath", "path"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _media_mime_from_item(item: Any) -> str:
    if isinstance(item, dict):
        for key in ("content_type", "mime_type", "mimeType", "mime"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().lower()
    return ""


def _is_image_media_item(item: Any) -> bool:
    mime = _media_mime_from_item(item)
    if mime.startswith("image/"):
        return True
    if isinstance(item, dict):
        raw_type = item.get("type") or item.get("media_type") or item.get("mediaType")
        if isinstance(raw_type, str) and raw_type.strip().lower() == "image":
            return True
    path = _media_path_from_item(item).lower()
    return path.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"))


def _media_image_source_and_mask(media: Any, *, task: str) -> tuple[str, str | None]:
    image_items = [item for item in list(media or []) if _is_image_media_item(item)]
    if not image_items:
        raise ValueError(f"Local {task} subprocess requires a source image file.")
    source_item = next(
        (
            item
            for item in image_items
            if str(item.get("role") or item.get("purpose") or "").strip().lower() == "source"
        ),
        image_items[0],
    )
    source_path = _media_path_from_item(source_item)
    if not source_path:
        raise ValueError(f"Local {task} subprocess media must include file_path for the source image.")
    mask_item = next(
        (
            item
            for item in image_items
            if str(item.get("role") or item.get("purpose") or "").strip().lower() == "mask"
        ),
        None,
    )
    mask_path = _media_path_from_item(mask_item) if mask_item is not None else ""
    if mask_item is not None and not mask_path:
        raise ValueError(f"Local {task} subprocess media must include file_path for the mask image.")
    return source_path, (mask_path or None)


def _generated_item_from_raw(
    raw: Any,
    *,
    modality: str,
    task: str,
    fmt: str,
    content_type: str,
    provider: str,
    model: str,
    backend_id: Any,
) -> Dict[str, Any]:
    data, artifact_ref, metadata = _bytes_from_raw(raw)
    return {
        "modality": modality,
        "task": task,
        "data": data,
        "artifact_ref": artifact_ref,
        "content_type": content_type,
        "format": fmt,
        "backend_id": backend_id,
        "provider": provider,
        "model": model,
        "metadata": metadata,
    }


def _run_image_spec(registry: Any, *, prompt: str, spec: Dict[str, Any], media: Any) -> list[Dict[str, Any]]:
    fmt = str(spec.get("format") or "png")
    content_type = str(spec.get("content_type") or spec.get("mime_type") or f"image/{fmt}")
    kwargs = _image_kwargs_from_spec(spec)
    extra = kwargs.get("extra")
    extra_dict = dict(extra) if isinstance(extra, dict) else {}
    extra_dict["on_progress"] = _emit_progress_event
    kwargs["extra"] = extra_dict
    count, seeds = _spec_generation_count_seeds(spec)
    task = str(spec.get("task") or "").strip().lower()
    provider = str(spec.get("provider") or getattr(registry.vision, "backend_id", None) or "abstractvision")
    model = str(spec.get("model") or "")
    backend_id = getattr(registry.vision, "backend_id", None)
    outputs: list[Any]

    if task in {"", "image_generation", "t2i", "text_to_image"}:
        if count > 1 or seeds is not None:
            outputs = list(registry.vision.t2i_batch(prompt, count=count, seeds=seeds, **kwargs) or [])
        else:
            outputs = [registry.vision.t2i(prompt, **kwargs)]
        normalized_task = "image_generation"
    elif task in {"image_edit", "image_to_image", "i2i", "edit_image"}:
        source_path, mask_path = _media_image_source_and_mask(media, task="image edit")
        if count > 1 or seeds is not None:
            outputs = list(
                registry.vision.i2i_batch(
                    prompt,
                    source_path,
                    mask=mask_path,
                    count=count,
                    seeds=seeds,
                    **kwargs,
                )
                or []
            )
        else:
            outputs = [registry.vision.i2i(prompt, source_path, mask=mask_path, **kwargs)]
        normalized_task = "image_edit"
    elif task in {"image_upscale", "image_upscaling", "upscale", "upscale_image"}:
        source_path, _mask_path = _media_image_source_and_mask(media, task="image upscale")
        outputs = [registry.vision.upscale_image(source_path, **kwargs)]
        normalized_task = "image_upscale"
    else:
        raise ValueError(f"Unsupported subprocess image task: {task!r}")

    return [
        _generated_item_from_raw(
            raw,
            modality="image",
            task=normalized_task,
            fmt=fmt,
            content_type=content_type,
            provider=provider,
            model=model,
            backend_id=backend_id,
        )
        for raw in outputs
    ]


def _emit_progress_event(event: Any) -> None:
    print(
        json.dumps(
            {
                "type": "progress",
                "event": _jsonable(event),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def _run_video_spec(
    registry: Any,
    *,
    prompt: str,
    spec: Dict[str, Any],
    media: Any,
) -> Dict[str, Any]:
    task = str(spec.get("task") or "").strip().lower()
    fmt = str(spec.get("format") or "mp4").strip().lower() or "mp4"
    content_type = str(spec.get("content_type") or spec.get("mime_type") or f"video/{fmt}")
    kwargs = _video_kwargs_from_spec(spec)
    extra = kwargs.get("extra")
    extra_dict = dict(extra) if isinstance(extra, dict) else {}
    extra_dict["on_progress"] = _emit_progress_event
    kwargs["extra"] = extra_dict
    count, seeds = _spec_generation_count_seeds(spec)
    provider = str(spec.get("provider") or getattr(registry.vision, "backend_id", None) or "abstractvision")
    model = str(spec.get("model") or "")
    backend_id = getattr(registry.vision, "backend_id", None)

    if task in {"image_to_video", "i2v", "video_from_image", "video_edit"}:
        image_items = [item for item in list(media or []) if _is_image_media_item(item)]
        if len(image_items) != 1:
            raise ValueError("Local image-to-video subprocess requires exactly one source image file.")
        image_path = _media_path_from_item(image_items[0])
        if not image_path:
            raise ValueError("Local image-to-video subprocess media must include file_path.")
        kwargs.setdefault("prompt", prompt)
        if count > 1 or seeds is not None:
            raw_items = list(registry.vision.i2v_batch(image_path, prompt=prompt, count=count, seeds=seeds, **kwargs) or [])
        else:
            raw_items = [registry.vision.i2v(image_path, **kwargs)]
        normalized_task = "image_to_video"
    elif task in {"", "video_generation", "text_to_video", "t2v"}:
        if count > 1 or seeds is not None:
            raw_items = list(registry.vision.t2v_batch(prompt, count=count, seeds=seeds, **kwargs) or [])
        else:
            raw_items = [registry.vision.t2v(prompt, **kwargs)]
        normalized_task = "text_to_video"
    else:
        raise ValueError(f"Unsupported subprocess video task: {task!r}")

    return {
        "modality": "video",
        "task": normalized_task,
        "outputs": [
            _generated_item_from_raw(
                raw,
                modality="video",
                task=normalized_task,
                fmt=fmt,
                content_type=content_type,
                provider=provider,
                model=model,
                backend_id=backend_id,
            )
            for raw in raw_items
        ],
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
        media = payload.get("media") if isinstance(payload.get("media"), list) else []
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
            if modality == "image" and (not task or task in {"image_generation", "t2i", "text_to_image"}):
                for item in _run_image_spec(registry, prompt=prompt, spec=dict(spec), media=media):
                    response.add_output("image", item)
                continue
            if modality == "video":
                video_result = _run_video_spec(registry, prompt=prompt, spec=dict(spec), media=media)
                for item in list(video_result.get("outputs") or []):
                    response.add_output("video", item)
                continue
            raise ValueError(f"Unsupported subprocess media spec: modality={modality!r} task={task!r}")

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
