"""Shared snapshot discovery helpers for AbstractCore-backed runtimes.

These helpers keep host-facing discovery/catalog logic inside Runtime instead of
duplicating Core integration code in hosts like Gateway.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .logging import get_logger

logger = get_logger(__name__)

_VISION_TASKS = {"text_to_image", "image_to_image", "text_to_video", "image_to_video"}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _dedupe_strings(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _provider_string_map(value: Any) -> Dict[str, list[str]]:
    out: Dict[str, list[str]] = {}
    if not isinstance(value, dict):
        return out
    for provider, raw_values in value.items():
        provider_id = str(provider or "").strip()
        if not provider_id:
            continue
        values: list[str] = []
        if isinstance(raw_values, str):
            values = [raw_values]
        elif isinstance(raw_values, list):
            for item in raw_values:
                if isinstance(item, str) and item.strip():
                    values.append(item.strip())
                elif isinstance(item, dict):
                    for key in ("model", "model_id", "id", "name"):
                        model_id = item.get(key)
                        if isinstance(model_id, str) and model_id.strip():
                            values.append(model_id.strip())
                            break
        cleaned = _dedupe_strings(values)
        if cleaned:
            out[provider_id] = cleaned
    return out


def _provider_models_from_mapping(mapping: Dict[str, list[str]]) -> list[Dict[str, str]]:
    rows: list[Dict[str, str]] = []
    for provider, models in mapping.items():
        provider_id = str(provider or "").strip()
        if not provider_id:
            continue
        for model in models:
            model_id = str(model or "").strip()
            if model_id:
                rows.append({"provider": provider_id, "model": model_id, "id": f"{provider_id}/{model_id}"})
    return rows


def _music_provider_id(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if not isinstance(value, dict):
        return ""
    for key in ("provider", "provider_id", "backend_id", "id", "name"):
        raw = value.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return ""


def _music_model_provider(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    for key in ("provider", "provider_id", "owned_by", "backend", "backend_id"):
        raw = value.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return ""


def _music_model_id(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if not isinstance(value, dict):
        return ""
    for key in ("model", "model_id", "id", "name"):
        raw = value.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return ""


def _music_models_by_provider(models: list[Dict[str, Any]]) -> Dict[str, list[str]]:
    out: Dict[str, list[str]] = {}
    seen: Dict[str, set[str]] = {}
    for item in models:
        provider = _music_model_provider(item)
        model = _music_model_id(item)
        if not provider or not model:
            continue
        provider_seen = seen.setdefault(provider.lower(), set())
        if model.lower() in provider_seen:
            continue
        provider_seen.add(model.lower())
        out.setdefault(provider, []).append(model)
    return out


def _voice_record_provider(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    for key in ("provider", "engine_id", "engine", "backend"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    params = item.get("params")
    if isinstance(params, dict):
        for key in ("provider", "engine"):
            value = params.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    tags = item.get("tags")
    if isinstance(tags, dict):
        for key in ("provider", "engine_id"):
            value = tags.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _voice_record_model(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    for key in ("model", "model_id", "voice_model", "language"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    params = item.get("params")
    if isinstance(params, dict):
        for key in ("model", "model_id"):
            value = params.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _voice_record_key(item: Any) -> str:
    if not isinstance(item, dict):
        return str(item)
    key_parts = [
        _voice_record_provider(item),
        _voice_record_model(item),
        str(item.get("id") or item.get("profile_id") or item.get("voice_id") or item.get("qualified_id") or "").strip(),
    ]
    return "|".join(part for part in key_parts if part)


def _voice_catalog_controls() -> Dict[str, Any]:
    return {
        "speed": {"supported": True, "min": 0.5, "max": 2.0, "default": 1.0},
        "quality_preset": {"supported": True, "values": ["low", "standard", "high"], "default": "standard"},
        "instructions": {"supported": True},
        "profile": {"supported": True},
        "voice_clone": {"supported": True},
    }


def _vision_models_by_provider(models: list[Dict[str, Any]]) -> Dict[str, list[str]]:
    out: Dict[str, list[str]] = {}
    seen: Dict[str, set[str]] = {}
    for item in models:
        provider = str(item.get("provider") or item.get("owned_by") or item.get("backend") or "").strip()
        model = str(item.get("model") or item.get("routed_model") or item.get("model_id") or item.get("id") or "").strip()
        if not provider or not model:
            continue
        provider_seen = seen.setdefault(provider.lower(), set())
        if model.lower() in provider_seen:
            continue
        provider_seen.add(model.lower())
        out.setdefault(provider, []).append(model)
    return out


def _provider_filtered_values(payload: Dict[str, Any], provider: Optional[str], *map_keys: str) -> list[str]:
    wanted = str(provider or "").strip().lower()
    values: list[str] = []
    for key in map_keys:
        mapping = payload.get(key)
        if not isinstance(mapping, dict):
            continue
        if wanted:
            for map_provider, map_values in mapping.items():
                if str(map_provider or "").strip().lower() != wanted:
                    continue
                values.extend(str(x).strip() for x in (map_values or []) if isinstance(x, str) and str(x).strip())
        else:
            for map_values in mapping.values():
                values.extend(str(x).strip() for x in (map_values or []) if isinstance(x, str) and str(x).strip())
    return _dedupe_strings(values)


def _filter_provider_model_catalog_response(
    response: Dict[str, Any],
    *,
    provider: Optional[str],
    model_keys: tuple[str, ...],
) -> Dict[str, Any]:
    wanted = str(provider or "").strip()
    if not wanted:
        return response
    models = _provider_filtered_values(response, wanted, *model_keys)
    if not models:
        active = str(response.get("active_provider") or response.get("provider") or "").strip().lower()
        if active == wanted.lower():
            raw = response.get("models")
            if isinstance(raw, list):
                models = _dedupe_strings([str(x).strip() for x in raw if isinstance(x, str) and str(x).strip()])
    out = dict(response)
    out["provider"] = wanted
    out["providers"] = [wanted] if models else []
    out["models"] = models
    out["available"] = bool(models)
    return out


def _filter_voice_catalog_response(
    response: Dict[str, Any],
    *,
    provider: Optional[str],
    model: Optional[str],
    providers_only: bool,
) -> Dict[str, Any]:
    wanted_provider = str(provider or "").strip().lower()
    wanted_model = str(model or "").strip().lower()
    out = dict(response)
    record_values: list[Any] = []
    for key in ("profiles", "voices", "cloned_voices"):
        values = response.get(key)
        if isinstance(values, list):
            record_values.extend(values)
    derived_providers = [
        _voice_record_provider(item)
        for item in record_values
        if isinstance(item, dict) and _voice_record_provider(item)
    ]
    map_providers: list[str] = []
    for key in ("tts_models_by_provider", "tts_voices_by_provider", "tts_profiles_by_provider"):
        mapping = response.get(key)
        if isinstance(mapping, dict):
            map_providers.extend(str(k).strip() for k in mapping.keys() if str(k).strip())
    providers = _dedupe_strings(
        [
            str(x).strip()
            for x in list(response.get("tts_providers") or response.get("providers") or []) + derived_providers + map_providers
            if isinstance(x, str) and str(x).strip()
        ]
    )
    stt_providers = _dedupe_strings(
        [str(x).strip() for x in (response.get("stt_providers") or []) if isinstance(x, str) and str(x).strip()]
    )
    if wanted_provider:
        providers = [p for p in providers if p.lower() == wanted_provider]
        stt_providers = [p for p in stt_providers if p.lower() == wanted_provider]

    def keep_record(item: Any) -> bool:
        if not isinstance(item, dict):
            return not wanted_provider and not wanted_model
        item_provider = _voice_record_provider(item).lower()
        item_model = _voice_record_model(item).lower()
        if wanted_provider and item_provider and item_provider != wanted_provider:
            return False
        if wanted_model and item_model and item_model != wanted_model:
            return False
        return True

    if providers_only:
        out["providers"] = providers
        out["tts_providers"] = providers
        out["stt_providers"] = stt_providers
        out["profiles"] = []
        out["voices"] = []
        out["cloned_voices"] = []
        out["available"] = bool(providers or stt_providers)
        return out

    for key in ("profiles", "voices", "cloned_voices"):
        values = response.get(key)
        if isinstance(values, list):
            out[key] = [item for item in values if keep_record(item)]
    for key in ("tts_models_by_provider", "stt_models_by_provider", "tts_voices_by_provider", "tts_profiles_by_provider"):
        mapping = response.get(key)
        if isinstance(mapping, dict) and wanted_provider:
            out[key] = {k: v for k, v in mapping.items() if str(k).strip().lower() == wanted_provider}
    if wanted_provider:
        tts_models = _provider_filtered_values(response, wanted_provider, "tts_models_by_provider", "models_by_provider")
        stt_models = _provider_filtered_values(response, wanted_provider, "stt_models_by_provider")
        if not tts_models and str(response.get("active_provider") or response.get("provider") or "").strip().lower() == wanted_provider:
            raw = response.get("tts_models") or response.get("models")
            if isinstance(raw, list):
                tts_models = _dedupe_strings([str(x).strip() for x in raw if isinstance(x, str) and str(x).strip()])
        if not stt_models and str(response.get("active_stt_provider") or "").strip().lower() == wanted_provider:
            raw = response.get("stt_models")
            if isinstance(raw, list):
                stt_models = _dedupe_strings([str(x).strip() for x in raw if isinstance(x, str) and str(x).strip()])
        out["models"] = tts_models
        out["tts_models"] = tts_models
        if "stt_models" in out or stt_models:
            out["stt_models"] = stt_models
    out["providers"] = providers or out.get("providers") or []
    out["tts_providers"] = providers or out.get("tts_providers") or []
    out["stt_providers"] = stt_providers or out.get("stt_providers") or []
    return out


def _filter_vision_provider_models_response(
    response: Dict[str, Any],
    *,
    provider: Optional[str],
    providers_only: bool,
) -> Dict[str, Any]:
    wanted = str(provider or "").strip().lower()
    items = response.get("models")
    models = [dict(x) for x in items if isinstance(x, dict)] if isinstance(items, list) else []
    models_by_provider = _provider_string_map(response.get("models_by_provider"))
    if not models_by_provider:
        models_by_provider = _vision_models_by_provider(models)
    if wanted:
        models = [
            item
            for item in models
            if str(item.get("provider") or item.get("owned_by") or "").strip().lower() == wanted
        ]
        models_by_provider = {
            key: values for key, values in models_by_provider.items() if str(key or "").strip().lower() == wanted
        }
    providers = _dedupe_strings(
        [str(x) for x in response.get("providers") or [] if isinstance(x, str)]
        + [str(x) for x in response.get("available_providers") or [] if isinstance(x, str)]
        + list(models_by_provider.keys())
        + [
            str(item.get("provider") or item.get("owned_by") or "").strip()
            for item in models
            if str(item.get("provider") or item.get("owned_by") or "").strip()
        ]
    )
    if wanted:
        providers = [p for p in providers if p.lower() == wanted]
    out = dict(response)
    out["providers"] = providers
    out["available_providers"] = [
        p
        for p in _dedupe_strings([str(x) for x in response.get("available_providers") or [] if isinstance(x, str)] or providers)
        if not wanted or p.lower() == wanted
    ] or providers
    out["models_by_provider"] = models_by_provider
    out["provider_models"] = _provider_models_from_mapping(models_by_provider)
    if providers_only:
        out["models"] = []
        out["available"] = bool(providers or out.get("available_providers"))
    else:
        out["models"] = models
        out["available"] = bool(models or models_by_provider)
    return out


def _vision_provider_model_item(*, provider: str, model_id: str, raw: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    provider_s = str(provider or "").strip() or "unknown"
    model_s = str(model_id or "").strip()
    item = {
        "id": model_s,
        "model": model_s,
        "provider": provider_s,
        "backend": provider_s,
        "routed_model": model_s,
        "object": "model",
        "owned_by": provider_s,
    }
    if isinstance(raw, dict):
        item["raw"] = dict(raw)
    return item


def _provider_models_from_cached_vision_response(cached: Optional[Dict[str, Any]], *, task: Optional[str]) -> Dict[str, Any]:
    models: list[Dict[str, Any]] = []
    values = cached.get("models") if isinstance(cached, dict) else None
    if isinstance(values, list):
        for item in values:
            if not isinstance(item, dict):
                continue
            model_id = str(item.get("id") or item.get("model") or "").strip()
            if not model_id:
                continue
            tasks = item.get("tasks")
            if task and isinstance(tasks, list) and task not in [str(t) for t in tasks]:
                continue
            provider = str(item.get("provider") or "huggingface").strip() or "huggingface"
            models.append(_vision_provider_model_item(provider=provider, model_id=model_id, raw=item))
    models_by_provider = _vision_models_by_provider(models)
    return {
        "available": bool(models),
        "route_available": True,
        "models": models,
        "models_by_provider": models_by_provider,
        "provider_models": _provider_models_from_mapping(models_by_provider),
        "providers": list(models_by_provider.keys()),
        "available_providers": list(models_by_provider.keys()),
        "task": task,
        "source": "abstractvision_local_cache",
        "stale": False,
        "refreshed_at": _utc_now_iso(),
        "error": None,
    }


def _merge_vision_provider_model_responses(*responses: Optional[Dict[str, Any]], task: Optional[str]) -> Dict[str, Any]:
    models: list[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    errors: list[str] = []
    providers: list[str] = []
    available_providers: list[str] = []
    for response in responses:
        if not isinstance(response, dict):
            continue
        err = response.get("error")
        if isinstance(err, str) and err.strip():
            errors.append(err.strip())
        providers.extend(str(x) for x in response.get("providers") or [] if isinstance(x, str))
        available_providers.extend(str(x) for x in response.get("available_providers") or [] if isinstance(x, str))
        items = response.get("models")
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            provider = str(item.get("provider") or item.get("owned_by") or "").strip()
            model = str(item.get("model") or item.get("routed_model") or item.get("id") or "").strip()
            if not model:
                continue
            key = (provider.lower(), model.lower())
            if key in seen:
                continue
            seen.add(key)
            models.append(dict(item))
    models_by_provider = _vision_models_by_provider(models)
    providers = _dedupe_strings(providers + list(models_by_provider.keys()))
    available_providers = _dedupe_strings(available_providers) or providers
    return {
        "available": bool(models or providers or available_providers),
        "route_available": True,
        "models": models,
        "models_by_provider": models_by_provider,
        "provider_models": _provider_models_from_mapping(models_by_provider),
        "providers": providers or available_providers,
        "available_providers": available_providers or providers,
        "task": task,
        "source": "abstractruntime.discovery",
        "stale": False,
        "refreshed_at": _utc_now_iso(),
        "error": "; ".join(errors) if errors and not models else None,
    }


def _with_status(
    payload: Dict[str, Any],
    *,
    source: str,
    available: Optional[bool] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    out = dict(payload)
    out["route_available"] = True
    out["source"] = source
    out.setdefault("stale", False)
    out["refreshed_at"] = _utc_now_iso()
    out["error"] = error
    if available is not None:
        out["available"] = bool(available)
    else:
        out.setdefault("available", True)
    return out


def _env_first(*names: str, default: Optional[str] = None) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default


def _runtime_capability_owner_config(
    *,
    voice_base_url: Optional[str] = None,
    voice_api_key: Optional[str] = None,
    vision_base_url: Optional[str] = None,
    vision_api_key: Optional[str] = None,
    music_base_url: Optional[str] = None,
    music_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    if isinstance(voice_base_url, str) and voice_base_url.strip():
        cfg["voice_remote_base_url"] = voice_base_url.strip()
    if isinstance(voice_api_key, str) and voice_api_key.strip():
        cfg["voice_remote_api_key"] = voice_api_key.strip()
    if isinstance(vision_base_url, str) and vision_base_url.strip():
        cfg["vision_base_url"] = vision_base_url.strip()
        cfg.setdefault("vision_backend", "openai-compatible")
    if isinstance(vision_api_key, str) and vision_api_key.strip():
        cfg["vision_api_key"] = vision_api_key.strip()
    if isinstance(music_base_url, str) and music_base_url.strip():
        cfg["music_acemusic_base_url"] = music_base_url.strip()
    if isinstance(music_api_key, str) and music_api_key.strip():
        cfg["music_acemusic_api_key"] = music_api_key.strip()
    for env_key, cfg_key in {
        "ABSTRACTVOICE_TTS_ENGINE": "voice_tts_engine",
        "ABSTRACTVOICE_STT_ENGINE": "voice_stt_engine",
        "ABSTRACTVOICE_TTS_MODEL": "voice_tts_model",
        "ABSTRACTVOICE_STT_MODEL": "voice_stt_model",
        "ABSTRACTVOICE_REMOTE_BASE_URL": "voice_remote_base_url",
        "ABSTRACTVOICE_REMOTE_API_KEY": "voice_remote_api_key",
        "ABSTRACTVISION_BACKEND": "vision_backend",
        "ABSTRACTVISION_BASE_URL": "vision_base_url",
        "ABSTRACTVISION_API_KEY": "vision_api_key",
        "ABSTRACTVISION_MODEL_ID": "vision_model_id",
        "ABSTRACTVISION_DIFFUSERS_MODEL_ID": "vision_model_id",
        "ABSTRACTVISION_MFLUX_MODEL": "vision_mflux_model",
        "ABSTRACTVISION_MFLUX_BASE_MODEL": "vision_mflux_base_model",
        "ABSTRACTVISION_MODEL_DIR": "vision_model_dir",
        "ABSTRACTVISION_MFLUX_QUANTIZE": "vision_mflux_quantize",
        "ABSTRACTVISION_MFLUX_ALLOW_DOWNLOAD": "vision_mflux_allow_download",
        "ABSTRACTVISION_DIFFUSERS_DEVICE": "vision_device",
        "ABSTRACTVISION_DIFFUSERS_TORCH_DTYPE": "vision_torch_dtype",
        "ABSTRACTVISION_SDCPP_MODEL": "vision_sdcpp_model",
        "ABSTRACTVISION_SDCPP_DIFFUSION_MODEL": "vision_sdcpp_diffusion_model",
        "ABSTRACTVISION_SDCPP_BIN": "vision_sdcpp_bin",
        "ABSTRACTMUSIC_BACKEND": "music_backend",
        "ABSTRACTMUSIC_MODEL_ID": "music_model_id",
        "ABSTRACTMUSIC_BASE_URL": "music_acemusic_base_url",
        "ACEMUSIC_BASE_URL": "music_acemusic_base_url",
        "ABSTRACTMUSIC_API_KEY": "music_acemusic_api_key",
        "ACEMUSIC_API_KEY": "music_acemusic_api_key",
    }.items():
        value = os.getenv(env_key)
        if isinstance(value, str) and value.strip() and cfg_key not in cfg:
            cfg[cfg_key] = value.strip()
    return cfg


def _runtime_capability_registry(
    *,
    voice_base_url: Optional[str] = None,
    voice_api_key: Optional[str] = None,
    vision_base_url: Optional[str] = None,
    vision_api_key: Optional[str] = None,
    music_base_url: Optional[str] = None,
    music_api_key: Optional[str] = None,
) -> Any:
    from abstractcore.capabilities import CapabilityRegistry

    preferred_backends: Dict[str, str] = {}
    for capability in ("voice", "audio", "vision", "music"):
        value = _env_first(f"ABSTRACTCORE_{capability.upper()}_BACKEND")
        if value:
            preferred_backends[capability] = value
    owner = type(
        "_RuntimeCapabilityOwner",
        (),
        {
            "config": _runtime_capability_owner_config(
                voice_base_url=voice_base_url,
                voice_api_key=voice_api_key,
                vision_base_url=vision_base_url,
                vision_api_key=vision_api_key,
                music_base_url=music_base_url,
                music_api_key=music_api_key,
            )
        },
    )()
    return CapabilityRegistry(owner, preferred_backends=preferred_backends)


def local_list_providers(
    *,
    include_models: bool = False,
    default_provider: Optional[str] = None,
    default_model: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        from abstractcore.providers.registry import get_all_providers_with_models

        providers = get_all_providers_with_models(include_models=bool(include_models))
    except Exception as exc:
        return _with_status(
            {
                "items": [],
                "default_provider": default_provider,
                "default_model": default_model,
            },
            source="abstractcore.local",
            available=False,
            error=str(exc),
        )

    items: list[Dict[str, Any]] = []
    if isinstance(providers, list):
        for provider in providers:
            if isinstance(provider, dict):
                name = provider.get("name")
                if isinstance(name, str) and name.strip():
                    items.append(dict(provider))
    items.sort(key=lambda item: str(item.get("name") or ""))
    return _with_status(
        {
            "items": items,
            "default_provider": default_provider,
            "default_model": default_model,
        },
        source="abstractcore.local",
        available=bool(items),
        error=None,
    )


def local_list_provider_models(
    provider_name: str,
    *,
    base_url: Optional[str] = None,
    provider_api_key: Optional[str] = None,
    timeout_s: Optional[float] = None,
) -> Dict[str, Any]:
    provider_text = str(provider_name or "").strip()
    if not provider_text:
        return _with_status(
            {"provider": None, "models": []},
            source="abstractcore.local",
            available=False,
            error="provider_name is required",
        )
    kwargs: Dict[str, Any] = {}
    if isinstance(base_url, str) and base_url.strip():
        kwargs["base_url"] = base_url.strip()
    if isinstance(provider_api_key, str) and provider_api_key.strip():
        kwargs["api_key"] = provider_api_key.strip()
    if timeout_s is not None:
        kwargs["timeout"] = float(timeout_s)
    try:
        from abstractcore.providers.registry import get_available_models_for_provider

        models = get_available_models_for_provider(provider_text, **kwargs)
    except Exception as exc:
        return _with_status(
            {"provider": provider_text, "models": []},
            source="abstractcore.local",
            available=False,
            error=str(exc),
        )
    out = [str(model).strip() for model in list(models or []) if isinstance(model, str) and str(model).strip()]
    out.sort()
    return _with_status(
        {"provider": provider_text, "models": out},
        source="abstractcore.local",
        available=bool(out),
        error=None,
    )


def local_get_model_capabilities(model_name: str) -> Dict[str, Any]:
    model_text = str(model_name or "").strip()
    if not model_text:
        return _with_status(
            {"model": None, "capabilities": {}},
            source="abstractcore.local",
            available=False,
            error="model_name is required",
        )
    try:
        from abstractcore.architectures.detection import get_model_capabilities

        capabilities = get_model_capabilities(model_text)
    except Exception as exc:
        return _with_status(
            {"model": model_text, "capabilities": {}},
            source="abstractcore.local",
            available=False,
            error=str(exc),
        )
    if not isinstance(capabilities, dict):
        capabilities = {}
    return _with_status(
        {"model": model_text, "capabilities": dict(capabilities)},
        source="abstractcore.local",
        available=bool(capabilities),
        error=None,
    )


def local_get_voice_catalog(
    *,
    base_url: Optional[str] = None,
    provider_api_key: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    providers_only: bool = False,
) -> Dict[str, Any]:
    try:
        voice = _runtime_capability_registry(
            voice_base_url=base_url,
            voice_api_key=provider_api_key,
        ).voice
        catalog = voice.voice_catalog() if hasattr(voice, "voice_catalog") else {}
    except Exception as exc:
        payload = _with_status(
            {
                "kind": "tts",
                "profiles": [],
                "voices": [],
                "cloned_voices": [],
                "models": [],
                "tts_models": [],
                "stt_models": [],
                "providers": [],
                "tts_providers": [],
                "stt_providers": [],
                "controls": _voice_catalog_controls(),
            },
            source="abstractvoice.local",
            available=False,
            error=str(exc),
        )
        return _filter_voice_catalog_response(payload, provider=provider, model=model, providers_only=providers_only)

    out = dict(catalog) if isinstance(catalog, dict) else {}
    out.setdefault("kind", "tts")
    out.setdefault("profiles", [])
    out.setdefault("voices", out.get("profiles") if isinstance(out.get("profiles"), list) else [])
    out.setdefault("cloned_voices", [])
    out.setdefault("models", out.get("tts_models") if isinstance(out.get("tts_models"), list) else [])
    out.setdefault("tts_models", out.get("models") if isinstance(out.get("models"), list) else [])
    out.setdefault("stt_models", [])
    out.setdefault("controls", _voice_catalog_controls())
    payload = _with_status(
        out,
        source="abstractvoice.local",
        available=bool(
            out.get("profiles")
            or out.get("voices")
            or out.get("cloned_voices")
            or out.get("tts_models")
            or out.get("stt_models")
            or out.get("tts_providers")
            or out.get("stt_providers")
        ),
        error=None,
    )
    return _filter_voice_catalog_response(payload, provider=provider, model=model, providers_only=providers_only)


def local_list_tts_models(
    *,
    base_url: Optional[str] = None,
    provider_api_key: Optional[str] = None,
    provider: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        voice = _runtime_capability_registry(
            voice_base_url=base_url,
            voice_api_key=provider_api_key,
        ).voice
        catalog = voice.voice_catalog() if hasattr(voice, "voice_catalog") else {}
        models = voice.list_tts_models()
    except Exception as exc:
        return _filter_provider_model_catalog_response(
            _with_status(
                {
                    "models": [],
                    "providers": [],
                    "available_providers": [],
                    "active_provider": None,
                    "models_by_provider": {},
                    "tts_models_by_provider": {},
                    "provider_models": [],
                    "controls": _voice_catalog_controls(),
                },
                source="abstractvoice.local",
                available=False,
                error=str(exc),
            ),
            provider=provider,
            model_keys=("models_by_provider", "tts_models_by_provider"),
        )

    if not isinstance(catalog, dict):
        catalog = {}
    models_by_provider = _provider_string_map(catalog.get("tts_models_by_provider"))
    cleaned = _dedupe_strings(
        [str(item) for values in models_by_provider.values() for item in values]
        + [str(item) for item in list(models or [])]
    )
    providers = _dedupe_strings(
        [str(item) for item in list(catalog.get("tts_providers") or []) if isinstance(item, str)]
        + list(models_by_provider.keys())
    )
    active_provider = str(catalog.get("active_tts_provider") or catalog.get("engine_id") or "").strip()
    if cleaned and not models_by_provider and (active_provider or providers):
        models_by_provider[active_provider or providers[0]] = cleaned
    payload = _with_status(
        {
            "models": cleaned,
            "active_model": cleaned[0] if cleaned else None,
            "providers": providers,
            "available_providers": list(catalog.get("available_tts_providers") or providers)
            if isinstance(catalog.get("available_tts_providers"), list)
            else providers,
            "active_provider": active_provider or (providers[0] if providers else None),
            "models_by_provider": models_by_provider,
            "tts_models_by_provider": models_by_provider,
            "provider_models": _provider_models_from_mapping(models_by_provider),
            "controls": _voice_catalog_controls(),
        },
        source="abstractvoice.local",
        available=bool(cleaned or providers),
        error=None,
    )
    return _filter_provider_model_catalog_response(
        payload,
        provider=provider,
        model_keys=("models_by_provider", "tts_models_by_provider"),
    )


def local_list_stt_models(
    *,
    base_url: Optional[str] = None,
    provider_api_key: Optional[str] = None,
    provider: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        voice = _runtime_capability_registry(
            voice_base_url=base_url,
            voice_api_key=provider_api_key,
        ).voice
        catalog = voice.voice_catalog() if hasattr(voice, "voice_catalog") else {}
        models = voice.list_stt_models()
    except Exception as exc:
        return _filter_provider_model_catalog_response(
            _with_status(
                {
                    "models": [],
                    "providers": [],
                    "available_providers": [],
                    "active_provider": None,
                    "models_by_provider": {},
                    "stt_models_by_provider": {},
                    "provider_models": [],
                },
                source="abstractvoice.local",
                available=False,
                error=str(exc),
            ),
            provider=provider,
            model_keys=("models_by_provider", "stt_models_by_provider"),
        )

    if not isinstance(catalog, dict):
        catalog = {}
    models_by_provider = _provider_string_map(catalog.get("stt_models_by_provider"))
    cleaned = _dedupe_strings(
        [str(item) for values in models_by_provider.values() for item in values]
        + [str(item) for item in list(models or [])]
    )
    providers = _dedupe_strings(
        [str(item) for item in list(catalog.get("stt_providers") or []) if isinstance(item, str)]
        + list(models_by_provider.keys())
    )
    active_provider = str(catalog.get("active_stt_provider") or "").strip()
    if cleaned and not models_by_provider and (active_provider or providers):
        models_by_provider[active_provider or providers[0]] = cleaned
    payload = _with_status(
        {
            "models": cleaned,
            "active_model": cleaned[0] if cleaned else None,
            "providers": providers,
            "available_providers": list(catalog.get("available_stt_providers") or providers)
            if isinstance(catalog.get("available_stt_providers"), list)
            else providers,
            "active_provider": active_provider or (providers[0] if providers else None),
            "models_by_provider": models_by_provider,
            "stt_models_by_provider": models_by_provider,
            "provider_models": _provider_models_from_mapping(models_by_provider),
        },
        source="abstractvoice.local",
        available=bool(cleaned or providers),
        error=None,
    )
    return _filter_provider_model_catalog_response(
        payload,
        provider=provider,
        model_keys=("models_by_provider", "stt_models_by_provider"),
    )


def local_list_music_providers(
    *,
    task: Optional[str] = None,
    base_url: Optional[str] = None,
    provider_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    task_value = str(task or "").strip() or "text_to_music"
    try:
        music = _runtime_capability_registry(
            music_base_url=base_url,
            music_api_key=provider_api_key,
        ).music
        raw = music.available_providers(task=task_value) if hasattr(music, "available_providers") else []
    except Exception as exc:
        return _with_status(
            {
                "task": task_value,
                "providers": [],
                "available_providers": [],
                "provider_details": [],
            },
            source="abstractmusic.local",
            available=False,
            error=str(exc),
        )

    details = [dict(item) for item in list(raw or []) if isinstance(item, dict)]
    providers = _dedupe_strings([_music_provider_id(item) for item in details if _music_provider_id(item)])
    return _with_status(
        {
            "task": task_value,
            "providers": providers,
            "available_providers": providers,
            "provider_details": details,
        },
        source="abstractmusic.local",
        available=bool(providers or details),
        error=None,
    )


def local_list_music_models(
    *,
    task: Optional[str] = None,
    base_url: Optional[str] = None,
    provider_api_key: Optional[str] = None,
    provider: Optional[str] = None,
) -> Dict[str, Any]:
    task_value = str(task or "").strip() or "text_to_music"
    provider_value = str(provider or "").strip()
    try:
        music = _runtime_capability_registry(
            music_base_url=base_url,
            music_api_key=provider_api_key,
        ).music
        raw = music.list_models(task=task_value, provider=provider_value or None) if hasattr(music, "list_models") else []
    except Exception as exc:
        return _with_status(
            {
                "task": task_value,
                "provider": provider_value or None,
                "models": [],
                "providers": [],
                "available_providers": [],
                "models_by_provider": {},
                "provider_models": [],
            },
            source="abstractmusic.local",
            available=False,
            error=str(exc),
        )

    models = [dict(item) for item in list(raw or []) if isinstance(item, dict)]
    if provider_value:
        models = [
            item for item in models if _music_model_provider(item).lower() == provider_value.lower()
        ]
    models_by_provider = _music_models_by_provider(models)
    providers = _dedupe_strings(
        [_music_model_provider(item) for item in models if _music_model_provider(item)]
    )
    return _with_status(
        {
            "task": task_value,
            "provider": provider_value or None,
            "models": models,
            "providers": providers,
            "available_providers": providers,
            "models_by_provider": models_by_provider,
            "provider_models": _provider_models_from_mapping(models_by_provider),
        },
        source="abstractmusic.local",
        available=bool(models or providers),
        error=None,
    )


def local_list_vision_provider_models(
    *,
    task: Optional[str] = None,
    base_url: Optional[str] = None,
    provider_api_key: Optional[str] = None,
    provider: Optional[str] = None,
    providers_only: bool = False,
) -> Dict[str, Any]:
    task_value = str(task or "").strip() or None
    if task_value and task_value not in _VISION_TASKS:
        return _filter_vision_provider_models_response(
            _with_status(
                {
                    "models": [],
                    "providers": [],
                    "available_providers": [],
                    "models_by_provider": {},
                    "provider_models": [],
                    "task": task_value,
                },
                source="abstractvision.local",
                available=False,
                error="task must be one of: text_to_image, image_to_image, text_to_video, image_to_video",
            ),
            provider=provider,
            providers_only=providers_only,
        )

    catalog_error: Optional[str] = None
    availability: Dict[str, Any] = {}
    items: list[Dict[str, Any]] = []
    backend_id: Optional[str] = None
    try:
        vision = _runtime_capability_registry(
            vision_base_url=base_url,
            vision_api_key=provider_api_key,
        ).vision
        backend_id = getattr(vision, "backend_id", None)
        if hasattr(vision, "available_providers"):
            raw_availability = vision.available_providers(task=task_value)
            if isinstance(raw_availability, dict):
                availability = dict(raw_availability)
        if not providers_only:
            items = [dict(item) for item in list(vision.list_provider_models(task=task_value) or []) if isinstance(item, dict)]
    except Exception as exc:
        catalog_error = str(exc)

    models_by_provider = _vision_models_by_provider(items)
    providers = _dedupe_strings(
        [str(item) for item in list(availability.get("providers") or []) if isinstance(item, str)]
        + list(models_by_provider.keys())
    )
    available_providers = _dedupe_strings(
        [str(item) for item in list(availability.get("available_providers") or []) if isinstance(item, str)]
    ) or list(providers)
    payload = _with_status(
        {
            "models": items,
            "task": task_value,
            "providers": providers or available_providers,
            "available_providers": available_providers or providers,
            "models_by_provider": models_by_provider,
            "provider_models": _provider_models_from_mapping(models_by_provider),
            "details": availability.get("details") if isinstance(availability.get("details"), dict) else {},
            "backend_id": backend_id,
        },
        source="abstractvision.local",
        available=bool(items or providers or available_providers),
        error=catalog_error,
    )
    if not items:
        cached = local_list_cached_vision_models(task=task_value)
        if isinstance(cached, dict) and cached.get("models"):
            payload = _merge_vision_provider_model_responses(
                payload,
                _provider_models_from_cached_vision_response(cached, task=task_value),
                task=task_value,
            )
    return _filter_vision_provider_models_response(payload, provider=provider, providers_only=providers_only)


def local_list_cached_vision_models(
    *,
    task: Optional[str] = None,
    provider: Optional[str] = None,
) -> Dict[str, Any]:
    task_value = str(task or "").strip() or None
    if task_value and task_value not in _VISION_TASKS:
        return _with_status(
            {"models": [], "task": task_value},
            source="abstractvision.local_cache",
            available=False,
            error="task must be one of: text_to_image, image_to_image, text_to_video, image_to_video",
        )
    try:
        from abstractcore.capabilities.vision_catalog import get_local_vision_cache_catalog

        raw = get_local_vision_cache_catalog()
    except Exception as exc:
        return _with_status(
            {"models": [], "task": task_value},
            source="abstractvision.local_cache",
            available=False,
            error=str(exc),
        )

    out = dict(raw) if isinstance(raw, dict) else {}
    models = [dict(item) for item in list(out.get("models") or []) if isinstance(item, dict)]
    if task_value:
        models = [
            item
            for item in models
            if not isinstance(item.get("tasks"), list)
            or task_value in [str(task_item) for task_item in item.get("tasks") or []]
        ]
    provider_value = str(provider or "").strip().lower()
    if provider_value:
        models = [
            item
            for item in models
            if str(item.get("provider") or "").strip().lower() == provider_value
        ]
    out["models"] = models
    out["task"] = task_value
    error_text = out.get("error")
    return _with_status(
        out,
        source="abstractvision.local_cache",
        available=bool(models),
        error=str(error_text).strip() or None if error_text is not None else None,
    )


__all__ = [
    "local_get_model_capabilities",
    "local_list_music_models",
    "local_list_music_providers",
    "local_get_voice_catalog",
    "local_list_cached_vision_models",
    "local_list_providers",
    "local_list_provider_models",
    "local_list_stt_models",
    "local_list_tts_models",
    "local_list_vision_provider_models",
]
