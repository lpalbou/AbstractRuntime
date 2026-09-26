"""abstractruntime.integrations.abstractcore.llm_client

AbstractCore-backed LLM clients for AbstractRuntime.

Design intent:
- Keep `RunState.vars` JSON-safe: normalize outputs into dicts.
- Support both execution topologies:
  - local/in-process: call AbstractCore's `create_llm(...).generate(...)`
  - remote: call AbstractCore server `/v1/chat/completions`

Remote mode is the preferred way to support per-request dynamic routing (e.g. `base_url`).
"""

from __future__ import annotations

import ast
import base64
from copy import deepcopy
import hashlib
import io
import itertools
import json
import locale
import mimetypes
import os
from pathlib import Path
import re
import selectors
import subprocess
import sys
import tempfile
import threading
import time
import weakref
import uuid
import wave
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from contextlib import contextmanager
from typing import Any, Dict, List, Mapping, Optional, Protocol, Tuple
from urllib.parse import quote, urlencode

from .logging import get_logger
from .output_specs import (
    capability_default_reasoning_for_text as _capability_default_reasoning_for_text,
    capability_default_route_keys_for_spec as _capability_default_route_keys_for_spec,
    is_abstractcore_output_request as _is_abstractcore_output_request,
    normalize_output_specs_for_runtime as _normalize_output_specs_for_runtime,
    output_request_has_generated_media as _output_request_has_generated_media,
    output_request_has_non_text_result as _output_request_has_non_text_result,
    output_runtime_metadata as _output_runtime_metadata,
    strip_runtime_output_metadata_for_core as _strip_runtime_output_metadata_for_core,
)

logger = get_logger(__name__)

_ABSTRACTCORE_PROVIDER_API_KEY_HEADER = "X-AbstractCore-Provider-API-Key"
_LOCAL_GENERATE_LOCKS: Dict[Tuple[str, str], threading.Lock] = {}
_LOCAL_GENERATE_LOCKS_LOCK = threading.Lock()
_LOCAL_GENERATE_LOCKS_WARNED: set[Tuple[str, str]] = set()
_LOCAL_GENERATE_LOCKS_WARNED_LOCK = threading.Lock()
_LOCAL_IMAGE_SUBPROCESS_LOCK = threading.Lock()


def _capability_default_route_target(row: Any) -> Optional[Dict[str, Any]]:
    """A route row that names somewhere to SEND a call, or `None`.

    This is a narrower question than the grid's `configured` flag, which asks
    whether an operator has set anything at all on a route and counts the
    reasoning effort. A row carrying only a reasoning effort is configured and
    is honoured -- by `_with_capability_default_reasoning`, which reads it
    directly -- but it names no provider, model, base URL or plugin option, so
    it contributes nothing to the routing merge and is not a target.
    """
    if not isinstance(row, dict):
        return None
    if row.get("source") == "not_configured":
        return None
    if not (row.get("provider") or row.get("model") or row.get("base_url") or row.get("options")):
        return None
    return dict(row)


def _output_default_route_keys(
    spec: Dict[str, Any],
    *,
    has_source_image: bool = False,
) -> Tuple[Optional[str], Optional[str]]:
    """(exact, broad) capability-default route keys for a media output spec.

    Voice/music/sound joined this merge 2026-07-17 (tracing the offline-TTS
    outage: a bare TTS spec never received the gateway's configured
    `output.voice` route at THIS layer, so abstractcore's facade resolved from
    ITS OWN config -- two different truths, and the ledgered spec showed no
    merge). One resolution layer: the runtime merge is what the ledger records
    and what executes; the facade stays the fallback only when no route is
    configured.

    The mapping itself now comes from THE ONE TABLE in AbstractCore -- see
    `capability_default_route_keys_for_spec`. The copy that used to live here
    had drifted from core's and minted store-impossible keys.
    """

    return _capability_default_route_keys_for_spec(spec, has_source_image=has_source_image)


def _with_capability_default_route(
    spec: Dict[str, Any],
    capability_defaults: Optional[Dict[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    routed = dict(spec)
    if (
        not isinstance(capability_defaults, dict)
        or str(routed.get("provider") or "").strip()
        or str(routed.get("model") or "").strip()
        or str(routed.get("base_url") or "").strip()
    ):
        return routed
    primary_key, fallback_key = _output_default_route_keys(routed)
    if not primary_key:
        return routed
    route = _capability_default_route_target(capability_defaults.get(primary_key))
    if route is None and fallback_key:
        route = _capability_default_route_target(capability_defaults.get(fallback_key))
    if route is None:
        return routed
    for key in ("provider", "model", "base_url"):
        value = route.get(key)
        if isinstance(value, str) and value.strip():
            routed[key] = value.strip()
    options = route.get("options")
    if isinstance(options, dict):
        for key, value in options.items():
            if isinstance(key, str) and key.strip() and key not in routed and value is not None:
                routed[key.strip()] = value
    return routed


def _with_capability_default_reasoning(
    params: Dict[str, Any],
    capability_defaults: Optional[Dict[str, Dict[str, Any]]],
) -> Any:
    """Resolve `thinking` for one call and return the effective value.

    THE REASONING DIAL FOLLOWS THE SAME CASCADE AS PROVIDER/MODEL. AbstractCore
    stores a reasoning effort on the text-generation capability route; when a
    call names none, that stored effort is what the execution host applies.

    Precedence, highest first:

      1. EXPLICIT PIN -- any `thinking` the caller set, INCLUDING ``False``.
         ``False`` means "reasoning off for this call" and is a decision, not an
         absence, so it outranks the default exactly as a pinned provider does.
      2. HOST DEFAULT -- the configured reasoning on the text route.
      3. NOTHING -- `thinking` stays absent and the model behaves as it does
         without the parameter.

    `params` is mutated in place because it is the per-call kwargs dict that is
    about to be forwarded to the provider.
    """

    pinned = params.get("thinking")
    if pinned is not None and not (isinstance(pinned, str) and not pinned.strip()):
        return pinned
    configured = _capability_default_reasoning_for_text(capability_defaults)
    if configured:
        params["thinking"] = configured
        return configured
    params.pop("thinking", None)
    return None


def _with_output_progress_callback(spec: Dict[str, Any], progress_callback: Optional[Any]) -> Dict[str, Any]:
    if not callable(progress_callback):
        return dict(spec)
    routed = dict(spec)
    extra = routed.get("extra")
    extra_dict = dict(extra) if isinstance(extra, dict) else {}
    extra_dict["on_progress"] = progress_callback
    routed["extra"] = extra_dict
    return routed


def _progress_number(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value)
        except Exception:
            return None
    return None


def _progress_looks_complete(progress: Dict[str, Any]) -> bool:
    direct = _progress_number(progress.get("progress"))
    if direct is None:
        direct = _progress_number(progress.get("percent"))
        if direct is not None and direct > 1:
            direct = direct / 100.0
    if direct is not None and direct >= 1.0:
        return True
    for current_key, total_key in (("step", "total_steps"), ("frame", "total_frames"), ("current", "total")):
        current = _progress_number(progress.get(current_key))
        total = _progress_number(progress.get(total_key))
        if current is not None and total is not None and total > 0 and current >= total:
            return True
    return False


def _progress_is_reported(progress: Dict[str, Any]) -> bool:
    mode = str(progress.get("progress_mode") or "").strip().lower()
    if progress.get("reported") is False:
        return False
    if mode == "unreported":
        return False
    return True


def _spec_generation_count(spec: Dict[str, Any]) -> Optional[int]:
    if not isinstance(spec, dict):
        return None
    raw = spec.get("count", spec.get("n"))
    if raw is None:
        seeds = _spec_generation_seeds(spec)
        return len(seeds) if seeds else None
    try:
        count = int(raw)
    except Exception as exc:
        raise ValueError("Vision output count must be an integer >= 1.") from exc
    if count < 1:
        raise ValueError("Vision output count must be >= 1.")
    return count


def _spec_generation_seeds(spec: Dict[str, Any]) -> Optional[List[int]]:
    if not isinstance(spec, dict):
        return None
    raw = spec.get("seeds")
    if raw is None:
        return None
    if isinstance(raw, (str, bytes, bytearray)) or not isinstance(raw, list):
        raise ValueError("Vision output seeds must be a list of integers.")
    seeds: List[int] = []
    for value in raw:
        try:
            seeds.append(int(value))
        except Exception as exc:
            raise ValueError("Vision output seeds must be integers.") from exc
    if not seeds:
        raise ValueError("Vision output seeds cannot be empty.")
    return seeds


def _spec_lora_adapters(spec: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    raw = spec.get("lora_adapters")
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError("Vision output lora_adapters must be a list of adapter objects.")
    adapters: List[Dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Vision output lora_adapters[{index}] must be an object.")
        adapters.append({str(key): value for key, value in item.items() if value is not None})
    return adapters


def _register_subprocess_stream(selector: Any, fileobj: Any, label: str) -> None:
    try:
        selector.register(fileobj, selectors.EVENT_READ, data=label)
    except TypeError:
        selector.register(fileobj, selectors.EVENT_READ)


def _subprocess_stream_label(key: Any, *, stdout: Any, stderr: Any) -> str:
    label = getattr(key, "data", None)
    if isinstance(label, str) and label:
        return label
    fileobj = getattr(key, "fileobj", None)
    if fileobj is stderr:
        return "stderr"
    if fileobj is stdout:
        return "stdout"
    return "stdout"


def _progress_for_remote_job_state(progress: Dict[str, Any], *, job_id: str, state: str) -> Dict[str, Any]:
    payload = dict(progress)
    payload.setdefault("job_id", job_id)
    payload["job_state"] = state
    terminal = state in {"succeeded", "failed"}
    if terminal:
        payload["terminal"] = True
        payload["progress_mode"] = "terminal"
        payload.setdefault("progress_source", "core")
        payload.setdefault("status", state)
        payload.setdefault("phase", state)
        if state == "succeeded":
            payload["progress"] = 1.0
        return payload
    payload.setdefault("terminal", False)
    if _progress_looks_complete(payload):
        raw_progress = payload.get("progress")
        if raw_progress is not None:
            payload.setdefault("raw_progress", raw_progress)
        payload["progress"] = min(float(_progress_number(payload.get("progress")) or 1.0), 0.999)
        payload["phase"] = "finalizing"
        payload["status"] = "running"
        payload["message"] = "Finalizing output"
        payload["progress_mode"] = "finalizing"
        payload.setdefault("progress_source", "runtime")
        payload["terminal"] = False
    return payload


@dataclass
class _PromptCacheSessionState:
    system_module_hash: str
    tools_module_hash: str
    prefix_cache_key: str
    message_hashes: List[str]


def _fingerprint_projection(value: Any) -> Any:
    """Deterministic JSON-safe projection for fingerprint hashing (0064
    adversary P2-1): default=str leaked `id()` through default reprs
    (`<X object at 0x...>`), so two equal objects built on different turns
    hashed differently - a silent full-rebuild class. JSON-safe leaves pass
    through; exotic leaves project to a type-qualified str() with any
    memory address scrubbed; unsortable dict keys stringify."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_fingerprint_projection(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _fingerprint_projection(v) for k, v in value.items()}
    text = str(value)
    text = re.sub(r" at 0x[0-9a-fA-F]+", " at 0x", text)
    return f"{type(value).__name__}:{text}"


def _has_local_prompt_cache_control_plane(provider: Any) -> bool:
    """True for providers that keep an IN-PROCESS prompt cache the host can plan into
    (MLX, HuggingFace) — the same two questions `_maybe_prepare_prompt_cache` asks."""
    if provider is None:
        return False
    try:
        supports = getattr(provider, "supports_prompt_cache", None)
        if not callable(supports) or not bool(supports()):
            return False
        supports_op = getattr(provider, "prompt_cache_supports_operation", None)
        return bool(callable(supports_op) and supports_op("prepare_modules"))
    except Exception:
        return False


def _uses_local_full_context_prompt_cache(provider: Any, *, provider_name: str) -> bool:
    """Whether a runtime-derived key needs explicit full-history call shape.

    Native MLX has keyed APC without the legacy prepare/fork control plane.
    It still requires messages=[] for a standalone, full-context prompt.
    Keep this bridge narrow: changing None to [] on unrelated transports can
    change their endpoint/template (notably Ollama's generate versus chat).
    """
    if _has_local_prompt_cache_control_plane(provider):
        return True
    if str(provider_name or "").strip().lower() != "mlx":
        return False
    try:
        # Do not use the legacy helper's inferred keyed mode: only an
        # explicitly advertised profile identifies this native APC contract.
        getter = getattr(provider, "get_prompt_cache_capabilities", None)
        if not callable(getter):
            return False
        capabilities = getter()
        if callable(getattr(capabilities, "to_dict", None)):
            capabilities = capabilities.to_dict()
        return (
            isinstance(capabilities, dict)
            and capabilities.get("supported") is True
            and capabilities.get("mode") == "keyed"
        )
    except Exception:
        return False


def _callable_accepts_kwarg(fn: Any, name: str) -> bool:
    """True when `fn` can be called with keyword `name` (explicit parameter or **kwargs)."""
    import inspect

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    if name in params:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


def _prompt_cache_message_fingerprint(message: Any) -> str:
    """ROLE + CONTENT + (canonical) TOOL_CALLS fingerprint (0064 fix 1).

    tool_calls were EXCLUDED before, so two assistant messages with the same
    content head but different tool_calls fingerprinted the SAME — a false
    prefix MATCH could serve stale KV (correctness, not just perf). The
    serialization is CANONICAL (sort_keys + fixed separators, default=str),
    so dict key-order drift across JSON round-trips (run-store save/load,
    ledger replay) can never mint a false MISMATCH. Messages WITHOUT
    tool_calls keep the pre-fix payload shape byte-for-byte, so the common
    case pays zero fingerprint churn on upgrade (one-time rebuild only for
    sessions whose history carries tool calls)."""
    if not isinstance(message, dict):
        payload = {"role": "", "content": str(message)}
    else:
        role = str(message.get("role") or "")
        content = message.get("content")
        if isinstance(content, (dict, list)):
            try:
                content_norm = json.dumps(content, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            except Exception:
                content_norm = str(content)
        elif content is None:
            content_norm = ""
        else:
            content_norm = str(content)
        payload = {"role": role, "content": content_norm}
        tool_calls = message.get("tool_calls")
        if tool_calls:
            payload["tool_calls"] = json.dumps(
                _fingerprint_projection(tool_calls), sort_keys=True,
                ensure_ascii=False, separators=(",", ":"),
            )

    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _pop_provider_api_key(values: Dict[str, Any]) -> Optional[str]:
    """Return and remove a per-request provider key from common compatibility names."""

    for key in ("provider_api_key", "api_key"):
        raw = values.pop(key, None)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return None


# Construction kwargs that describe WHERE a provider is reached and WITH WHAT
# credential. They are bound to the exact provider identity they were configured
# for and MUST NOT be inherited by a client built for a different provider.
#
# Routing defect 2026-07-31 (36-run benchmark wave): the pool's construction
# kwargs carry the gateway default endpoint profile's base_url + api_key. Every
# per-call provider override (`lmstudio`, `ollama`, `openai`, ...) inherited them,
# so `create_llm("lmstudio", base_url="<airelay>/v1", api_key="<airelay key>")`
# built an LM Studio client that actually talked to the default relay. Pins were
# silently served by the default endpoint's models — the exact symptom measured.
_CONNECTION_SCOPED_LLM_KWARGS: Tuple[str, ...] = ("base_url", "api_key", "api_base", "organization", "project")


class _Unset:
    """Sentinel distinguishing "not supplied" from an explicit None."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<unset>"


_UNSET = _Unset()


def _speculation_construction_request(params: Mapping[str, Any]) -> Any:
    """The speculation request that must reach `create_llm`, or None.

    Speculation is a LOAD-TIME property: MLX binds the MTP drafter to the
    target while the weights are loaded, so AbstractCore refuses a per-call
    request on a provider whose lane was never prepared -- "speculation must
    be requested when the provider is created -- it selects the runtime that
    loads the weights".

    The runtime carried the request as a generate PARAM only (see
    `visual/executor.py` and `compiler.py`, which set `params["speculation"]`),
    and pooled clients are keyed by provider/model alone. So a run that asked
    for a depth could never get one: the pooled provider had been built
    without speculation, and a brand-new run built one the same way
    microseconds before failing with that message -- which reads as a caller
    error when the caller never had a way to ask.

    "Off" and absent return None on purpose: constructing a separate provider
    for "no speculation" would split the pool for nothing.
    """
    value = params.get("speculation")
    if value is None or value is False:
        return None
    if value is True:
        return True
    if isinstance(value, Mapping):
        mode = str(value.get("mode") or "").strip().lower()
        if mode in {"", "off", "none", "disabled"}:
            return None
        if value.get("enabled") is False:
            return None
        return dict(value)
    return None


def _speculation_fingerprint(value: Any) -> str:
    """Stable short digest of a speculation request, for the client cache key.

    Two depths are two different LOADS of the model -- one pooled instance
    cannot serve both, so the request has to be part of the key that selects
    the instance.
    """
    if value is None:
        return ""
    try:
        canonical = json.dumps(value, sort_keys=True, default=str)
    except Exception:
        canonical = repr(value)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _split_connection_scoped_llm_kwargs(
    llm_kwargs: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Split construction kwargs into (shared, connection-scoped).

    `shared` are provider-agnostic knobs (enable_tracing, max_traces, timeouts…)
    that any pooled client may inherit. `connection` names the endpoint and the
    credential, and only travels with the identity it was configured for.
    """

    kwargs = dict(llm_kwargs or {})
    connection: Dict[str, Any] = {}
    for key in _CONNECTION_SCOPED_LLM_KWARGS:
        if key in kwargs:
            value = kwargs.pop(key)
            if value is not None and str(value).strip() != "":
                connection[key] = value
    return kwargs, connection


def _models_agree(requested: Any, served: Any) -> bool:
    """True when a served model id is the requested one.

    Providers legitimately answer with a pinned snapshot of what was asked
    (`gpt-5.4-mini` -> `gpt-5.4-mini-2026-03-17`) or a namespaced form
    (`qwen/qwen3.6-27b` -> `qwen3.6-27b`). Those are the SAME model and must not
    be flagged. A different model id is a routing lie and must be.
    """

    req = str(requested or "").strip().lower()
    srv = str(served or "").strip().lower()
    if not req or not srv:
        return True
    if req == srv:
        return True
    req_tail = req.rsplit("/", 1)[-1]
    srv_tail = srv.rsplit("/", 1)[-1]
    if req_tail == srv_tail:
        return True
    return srv_tail.startswith(req_tail) or req_tail.startswith(srv_tail)


def _stamp_effective_route(
    result: Any,
    *,
    requested_provider: Any,
    requested_model: Any,
    client: Any,
) -> Any:
    """Record WHERE the call was actually served, next to what it asked for.

    Observability defect 2026-07-31: a ledger record declared the REQUEST
    (`provider: openai, model: gpt-5.6-sol`) while the response came from an
    entirely different endpoint. Reporting the request as if it were the service
    is what let a silent misroute survive a 36-run benchmark wave. `result.route`
    is sourced from the constructed provider instance, so it cannot repeat the
    request back; `route.mismatch` is set when the served model disagrees.
    """

    if not isinstance(result, dict):
        return result
    llm = getattr(client, "_llm", None)
    effective_provider = str(getattr(llm, "provider", "") or getattr(client, "_provider", "") or "").strip()
    effective_model = str(getattr(llm, "model", "") or getattr(client, "_model", "") or "").strip()
    served_model = result.get("model")
    if not (isinstance(served_model, str) and served_model.strip()):
        raw = result.get("raw_response")
        served_model = raw.get("model") if isinstance(raw, dict) else None
    route: Dict[str, Any] = {
        "requested_provider": str(requested_provider or "").strip() or None,
        "requested_model": str(requested_model or "").strip() or None,
        "provider": effective_provider or None,
        "model": effective_model or None,
        "base_url": str(getattr(llm, "base_url", "") or "").strip() or None,
        "served_model": str(served_model or "").strip() or None,
    }
    route["mismatch"] = not _models_agree(route["requested_model"] or route["model"], route["served_model"])
    result["route"] = route
    return result


def _core_server_root_url(server_base_url: str) -> str:
    base = str(server_base_url or "").strip().rstrip("/")
    if base.lower().endswith("/v1"):
        base = base[:-3].rstrip("/")
    return base


def _join_core_control_url(server_base_url: str, path: str) -> str:
    root = _core_server_root_url(server_base_url)
    suffix = str(path or "").strip()
    if not suffix.startswith("/"):
        suffix = f"/{suffix}"
    return f"{root}{suffix}"


def _join_core_v1_url(server_base_url: str, path: str) -> str:
    root = _core_server_root_url(server_base_url)
    suffix = str(path or "").strip()
    if not suffix.startswith("/"):
        suffix = f"/{suffix}"
    if suffix == "/v1" or suffix.startswith("/v1/"):
        return f"{root}{suffix}"
    return f"{root}/v1{suffix}"


def _join_core_provider_v1_url(server_base_url: str, provider: str, path: str) -> str:
    root = _core_server_root_url(server_base_url)
    provider_s = str(provider or "").strip().lower().replace("_", "-")
    suffix = str(path or "").strip()
    if not suffix.startswith("/"):
        suffix = f"/{suffix}"
    return f"{root}/{quote(provider_s, safe='')}/v1{suffix}"


def _set_header_case_insensitive(headers: Dict[str, str], name: str, value: str) -> None:
    for existing in list(headers.keys()):
        if str(existing).lower() == name.lower():
            headers[existing] = value
            return
    headers[name] = value


def _local_generate_lock(*, provider: str, model: str) -> Optional[threading.Lock]:
    """Return a process-wide generation lock for providers that are not thread-safe.

    Ordinary MLX/Metal can crash when generations run on competing threads.
    Keep a per-model fallback lock; only an instance advertising its own safe
    scheduler may bypass it at call time.
    """

    prov = str(provider or "").strip().lower()
    if prov != "mlx":
        return None
    key = (prov, str(model or "").strip())
    with _LOCAL_GENERATE_LOCKS_LOCK:
        lock = _LOCAL_GENERATE_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _LOCAL_GENERATE_LOCKS[key] = lock
        return lock


def _local_instance_schedules_generation(provider: Any) -> bool:
    """Only an explicit instance guarantee permits bypassing the safety lock."""
    try:
        capability = getattr(provider, "supports_concurrent_generation", None)
        return callable(capability) and capability() is True
    except Exception as exc:
        logger.warning(
            "#FALLBACK concurrency capability probe failed; retaining the local "
            f"generation safety lock: {exc}"
        )
        return False


def _warn_local_generate_lock_once(*, provider: str, model: str) -> None:
    prov = str(provider or "").strip().lower()
    key = (prov, str(model or "").strip())
    with _LOCAL_GENERATE_LOCKS_WARNED_LOCK:
        if key in _LOCAL_GENERATE_LOCKS_WARNED:
            return
        _LOCAL_GENERATE_LOCKS_WARNED.add(key)
    logger.warning(
        "Local provider generation is serialized for safety (prevents MLX/Metal crashes under concurrency).",
        provider=prov,
        model=key[1],
    )

_SYSTEM_CONTEXT_HEADER_RE = re.compile(
    # ChatML-style user-turn grounding prefix, matching `chat-mlx.py` / `chat-hf.py`:
    #   "[YYYY-MM-DD HH:MM:SS CC]" (optionally followed by whitespace + user text).
    # Backward compatible with the historical "[YYYY/MM/DD HH:MM CC]" form.
    r"^\[\d{4}[-/]\d{2}[-/]\d{2}\s+\d{2}:\d{2}(?::\d{2})?\s+[A-Z]{2}\](?:\s|$)",
    re.IGNORECASE,
)

_LEGACY_SYSTEM_CONTEXT_HEADER_RE = re.compile(
    r"^Grounding:\s*\d{4}/\d{2}/\d{2}\|\d{2}:\d{2}\|[A-Z]{2}$",
    re.IGNORECASE,
)

_LEGACY_SYSTEM_CONTEXT_HEADER_PARSE_RE = re.compile(
    r"^Grounding:\s*(\d{4}/\d{2}/\d{2})\|(\d{2}:\d{2})\|([A-Z]{2})$",
    re.IGNORECASE,
)

_RUNTIME_METADATA_ENVELOPE_RE = re.compile(
    r"^\s*<runtime_metadata>\s*.*?\s*</runtime_metadata>\s*",
    re.IGNORECASE | re.DOTALL,
)

# A user turn whose ENTIRE content is a runtime metadata envelope. Such turns are
# runtime-owned injection artifacts (see `_normalize_turn_grounding`): they are
# dropped and re-appended fresh on every call, so injection stays idempotent when
# both the runtime ledger pass and the LLM client pass normalize the same payload.
_RUNTIME_METADATA_ONLY_RE = re.compile(
    r"^\s*<runtime_metadata>\s*.*?\s*</runtime_metadata>\s*$",
    re.IGNORECASE | re.DOTALL,
)


def _is_runtime_grounding_only_user_message(message: Any) -> bool:
    """Return True for runtime-owned, envelope-only trailing user turns.

    These are machine-generated grounding carriers (never user-authored text):
    the runtime appends them for tool-loop shaped conversations so the per-call
    timestamp entropy stays out of the cacheable message prefix.
    """
    if not isinstance(message, dict):
        return False
    if str(message.get("role") or "").strip().lower() != "user":
        return False
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        return False
    return bool(_RUNTIME_METADATA_ONLY_RE.match(content))


def _is_volatile_message(message: Any) -> bool:
    """True for messages STRUCTURALLY marked per-call-ephemeral (`volatile: true`).

    B1 fix, runtime half (code seat's prompt-cache adversary, commons c971):
    adapters mark per-call tail messages (e.g. the react loop's
    "[loop] iteration N of M.") with a top-level `volatile` flag instead of
    relying on content regexes. Flagged messages are EXCLUDED from the durable
    prompt-cache fingerprint sequence (they change every call, so hashing them
    forces a full re-prefill each cycle) and the flag itself is STRIPPED before
    the provider call (an unknown field reaching strict provider SDKs is a
    guaranteed-400 risk).
    """
    return isinstance(message, dict) and bool(message.get("volatile"))


def _strip_volatile_markers(messages: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    """Remove the `volatile` marker field before the provider boundary.

    The MESSAGE still rides the payload (it carries real per-call information);
    only the marker key is dropped. Copies are shallow per flagged message so
    callers' durable structures are never mutated."""
    if not isinstance(messages, list):
        return messages
    if not any(_is_volatile_message(m) for m in messages):
        return messages
    out: List[Dict[str, Any]] = []
    for m in messages:
        if _is_volatile_message(m):
            clean = dict(m)
            clean.pop("volatile", None)
            out.append(clean)
        else:
            out.append(m)
    return out


# ---------------------------------------------------------------------------
# SYNTHESIZED CARRIERS (mission A3, 2026-09-22) — the same STRUCTURAL-marker
# discipline as `volatile` above, for a different failure.
#
# A payload-boundary repair may have to invent a `user`-role message: abstractagent's
# `sanitize_transcript_messages` folds a tool result whose call id no announced call
# owns into `[unpaired tool result]: …`, because strict providers 400 on an orphan
# `tool` message. That carrier is NOT a turn — it is re-derived from the durable
# transcript every time a payload is built. On the operator's live run 081d8daa it
# was also the LAST message of a tool-loop payload, so `_normalize_turn_grounding`
# read the payload as "chat shape" and injected the grounding envelope into it; the
# next iteration rebuilt it without the envelope and stamped the NEW carrier. The
# two consecutive prompts were identical for messages 0..12 and differed by exactly
# 118 chars (one envelope) at index 13 of 21 — thousands of tokens before the end of
# a 13k-token prompt, outside every checkpoint window — so iterations 3, 4 and the
# whole next run went COLD.
#
# The marker is never pattern-matched from the prose: the producer declares it and
# it is STRIPPED before the provider call, exactly like `volatile`.
# ---------------------------------------------------------------------------

SYNTHETIC_MESSAGE_KEY = "_af_synthetic"
SYNTHETIC_TOOL_RESULT = "tool_result"


def _message_is_synthetic_carrier(message: Any) -> bool:
    """True when a payload-boundary repair synthesized this message."""
    if not isinstance(message, dict):
        return False
    return bool(str(message.get(SYNTHETIC_MESSAGE_KEY) or "").strip())


def _strip_synthetic_message_markers(
    messages: Optional[List[Dict[str, Any]]]
) -> Optional[List[Dict[str, Any]]]:
    """Remove the carrier marker before the provider boundary (unknown field = 400 risk)."""
    if not isinstance(messages, list):
        return messages
    if not any(_message_is_synthetic_carrier(m) for m in messages):
        return messages
    out: List[Dict[str, Any]] = []
    for m in messages:
        if _message_is_synthetic_carrier(m):
            clean = dict(m)
            clean.pop(SYNTHETIC_MESSAGE_KEY, None)
            out.append(clean)
        else:
            out.append(m)
    return out


def _transcript_carries_grounding_envelope(messages: Any) -> bool:
    """True when ANY non-synthetic user message of the transcript is already stamped.

    "Is this turn grounded?" is a question about the transcript, not about its last
    message: once a carrier or an operator interjection sits after the task, the
    stamped message is several positions back. Checking only the last one appended a
    fresh envelope on every iteration, which is the entropy the cache cannot absorb.
    """
    if not isinstance(messages, list):
        return False
    for m in messages:
        if not isinstance(m, dict):
            continue
        if str(m.get("role") or "").strip().lower() != "user":
            continue
        if _message_is_synthetic_carrier(m) or _is_volatile_message(m):
            continue
        if _content_carries_grounding_envelope(m.get("content")):
            return True
        # A stamped durable message the payload MERGED behind another user
        # message (alternation-strict adjacency repair: task + operator guidance
        # drained before the first reply) carries its envelope at a paragraph
        # boundary, not at the head. It is still the turn's grounding.
        if _content_has_merged_grounding_envelope(m.get("content")):
            return True
    return False


_MERGED_RUNTIME_METADATA_ENVELOPE_RE = re.compile(
    r"\n\n<runtime_metadata>\s*\{.*?\}\s*</runtime_metadata>",
    re.DOTALL,
)


def _content_has_merged_grounding_envelope(content: Any) -> bool:
    if isinstance(content, str):
        return bool(_MERGED_RUNTIME_METADATA_ENVELOPE_RE.search(content))
    if isinstance(content, list):
        return any(
            isinstance(item, dict)
            and str(item.get("type") or "").strip().lower() == "text"
            and bool(_MERGED_RUNTIME_METADATA_ENVELOPE_RE.search(str(item.get("text") or "")))
            for item in content
        )
    return False


_ZONEINFO_TAB_CANDIDATES = [
    "/usr/share/zoneinfo/zone.tab",
    "/usr/share/zoneinfo/zone1970.tab",
    "/var/db/timezone/zoneinfo/zone.tab",
    "/var/db/timezone/zoneinfo/zone1970.tab",
]


def _detect_timezone_name() -> Optional[str]:
    """Best-effort IANA timezone name (e.g. 'Europe/Paris')."""

    tz_env = os.environ.get("TZ")
    if isinstance(tz_env, str):
        tz = tz_env.strip().lstrip(":")
        if tz and "/" in tz:
            return tz

    # Common on Debian/Ubuntu.
    try:
        with open("/etc/timezone", "r", encoding="utf-8", errors="ignore") as f:
            line = f.readline().strip()
        if line and "/" in line:
            return line
    except Exception:
        pass

    # Common on macOS + many Linux distros (symlink or copied file).
    try:
        real = os.path.realpath("/etc/localtime")
    except Exception:
        real = ""
    if real:
        match = re.search(r"/zoneinfo/(.+)$", real)
        if match:
            tz = match.group(1).strip()
            if tz and "/" in tz:
                return tz

    return None


def _country_from_zone_tab(*, zone_name: str, tab_paths: Optional[List[str]] = None) -> Optional[str]:
    """Resolve ISO2 country code from zone.tab / zone1970.tab."""
    zone = str(zone_name or "").strip()
    if not zone:
        return None

    paths = list(tab_paths) if isinstance(tab_paths, list) and tab_paths else list(_ZONEINFO_TAB_CANDIDATES)
    for tab_path in paths:
        try:
            with open(tab_path, "r", encoding="utf-8", errors="ignore") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split("\t")
                    if len(parts) < 3:
                        continue
                    cc_field = parts[0].strip()
                    tz_field = parts[2].strip()
                    if tz_field != zone:
                        continue
                    cc = cc_field.split(",", 1)[0].strip()
                    if len(cc) == 2 and cc.isalpha():
                        return cc.upper()
        except Exception:
            continue
    return None


def _detect_country() -> str:
    """Best-effort 2-letter country code detection.

    Order:
    1) Explicit env override: ABSTRACT_COUNTRY / ABSTRACTFRAMEWORK_COUNTRY
    2) Timezone (IANA name) via zone.tab mapping
    3) Locale region from `locale.getlocale()` or locale env vars (LANG/LC_ALL/LC_CTYPE)

    Notes:
    - Avoid parsing encoding-only strings like `UTF-8` as a country (a common locale env pitfall).
    - Prefer timezone over locale: hosted shells frequently inherit an `en_US` locale even when
      the runtime is actually operating in another local timezone.
    - If no reliable region is found, return `XX` (unknown).
    """

    def _normalize_country_code(value: Optional[str]) -> Optional[str]:
        if not isinstance(value, str):
            return None
        raw = value.strip()
        if not raw:
            return None

        base = raw.split(".", 1)[0].split("@", 1)[0].strip()
        if len(base) == 2 and base.isalpha():
            return base.upper()

        parts = [p.strip() for p in re.split(r"[_-]", base) if p.strip()]
        for part in parts[1:]:
            if len(part) == 2 and part.isalpha():
                return part.upper()
        return None

    # Explicit override (preferred).
    for key in ("ABSTRACT_COUNTRY", "ABSTRACTFRAMEWORK_COUNTRY"):
        cc = _normalize_country_code(os.environ.get(key))
        if cc is not None:
            return cc

    tz_name = _detect_timezone_name()
    if tz_name:
        cc = _country_from_zone_tab(zone_name=tz_name)
        if cc is not None:
            return cc

    candidates: List[str] = []
    try:
        loc = locale.getlocale()[0]
        if isinstance(loc, str) and loc.strip():
            candidates.append(loc)
    except Exception:
        pass

    for key in ("LC_ALL", "LANG", "LC_CTYPE"):
        v = os.environ.get(key)
        if isinstance(v, str) and v.strip():
            candidates.append(v)

    for cand in candidates:
        cc = _normalize_country_code(cand)
        if cc is not None:
            return cc

    return "XX"


def _normalize_country_code(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None
    base = raw.split(".", 1)[0].split("@", 1)[0].strip()
    if len(base) == 2 and base.isalpha():
        return base.upper()
    parts = [p.strip() for p in re.split(r"[_-]", base) if p.strip()]
    for part in parts[1:]:
        if len(part) == 2 and part.isalpha():
            return part.upper()
    return None


def _system_context_header() -> str:
    # Use local datetime (timezone-aware) to match the user's environment.
    # Format: "[YYYY-MM-DD HH:MM:SS CC]"
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    return f"[{stamp} {_detect_country()}]"

def _strip_system_context_header(system_prompt: Optional[str]) -> Optional[str]:
    """Remove a runtime-injected system-context header from the system prompt (best-effort).

    Why:
    - Historically AbstractRuntime injected a "Grounding: ..." line into the *system prompt*.
    - Prompt/KV caching works best when stable prefixes (system/tools/history) do not contain per-turn entropy.
    - We still want date/time/country per turn, but we inject it into the *current user turn* instead.
    """
    if not isinstance(system_prompt, str):
        return system_prompt
    raw = system_prompt
    lines = raw.splitlines()
    if not lines:
        return None
    first = lines[0].strip()
    if not (_LEGACY_SYSTEM_CONTEXT_HEADER_RE.match(first) or _SYSTEM_CONTEXT_HEADER_RE.match(first)):
        return raw
    rest = "\n".join(lines[1:]).lstrip()
    return rest if rest else None


def _strip_internal_system_messages(messages: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    """Remove internal system messages that should never leak into model outputs.

    Today this is intentionally narrow and only strips the synthetic tool-activity
    summaries that can be injected by some agent hosts:
      "Recent tool activity (auto): ..."

    Why:
    - Some local/open models will echo system-message content verbatim.
    - These tool-trace summaries are *operator/debug* context, not user-facing content.
    """
    if not isinstance(messages, list) or not messages:
        return messages

    out: List[Dict[str, Any]] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "").strip().lower()
        if role == "system":
            c = m.get("content")
            if isinstance(c, str) and c.lstrip().startswith("Recent tool activity"):
                continue
        out.append(dict(m))

    return out or None


def _coalesce_leading_system_messages(
    *,
    system_prompt: Optional[str],
    messages: Optional[List[Dict[str, Any]]],
) -> tuple[Optional[str], Optional[List[Dict[str, Any]]]]:
    """Merge consecutive leading system messages into a single system prompt.

    Many local/chat-template based servers only accept a single leading system
    message. AbstractRuntime may synthesize extra leading system messages
    (attachments, memory notes, host hints), so normalize them before dispatch.
    """
    if not isinstance(messages, list) or not messages:
        return system_prompt, messages

    leading_parts: List[str] = []
    if isinstance(system_prompt, str) and system_prompt.strip():
        leading_parts.append(system_prompt)

    remaining: List[Dict[str, Any]] = []
    collecting = True
    for item in messages:
        if not isinstance(item, dict):
            collecting = False
            remaining.append({"role": "user", "content": str(item or "")})
            continue
        msg = dict(item)
        role = str(msg.get("role") or "").strip().lower()
        if collecting and role == "system":
            content = msg.get("content")
            content_str = content if isinstance(content, str) else str(content or "")
            if content_str.strip():
                leading_parts.append(content_str)
            continue
        collecting = False
        remaining.append(msg)

    merged_system = "\n\n".join(part.rstrip() for part in leading_parts if isinstance(part, str) and part.strip())
    return (merged_system or None), (remaining or None)


def _detect_runtime_user(trace_metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
    if isinstance(trace_metadata, dict):
        for key in ("user", "user_id", "username", "owner_id", "actor_id"):
            value = trace_metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if value is not None and str(value).strip():
                return str(value).strip()

    for key in ("ABSTRACT_USER", "ABSTRACTFRAMEWORK_USER", "USER", "LOGNAME"):
        value = os.environ.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _client_grounding_metadata(trace_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return browser-provided prompt-grounding metadata, if present.

    Browser context is useful for user-facing date/time/location grounding in hosted apps, but it
    is not trusted for authorization, routing, or policy. Keep it explicitly provenance-labeled.
    """
    if not isinstance(trace_metadata, dict):
        return {}
    raw = trace_metadata.get("client_context")
    if not isinstance(raw, dict):
        return {}

    def _text(key: str, *, max_len: int = 160) -> str:
        value = raw.get(key)
        if not isinstance(value, str):
            return ""
        return value.strip()[:max_len]

    out: Dict[str, Any] = {"source": "browser_untrusted"}
    local_datetime = _text("local_datetime", max_len=80)
    if local_datetime and re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:\d{2})?$", local_datetime):
        out["local_datetime"] = local_datetime
    timezone_name = _text("timezone", max_len=120)
    if timezone_name and re.match(r"^[A-Za-z0-9_+./-]+$", timezone_name):
        out["timezone"] = timezone_name
    locale_name = _text("locale", max_len=80)
    if locale_name and re.match(r"^[A-Za-z0-9_@.+-]+$", locale_name):
        out["locale"] = locale_name
    locale_country = _normalize_country_code(_text("locale_country", max_len=8))
    if locale_country:
        out["locale_country"] = locale_country
    country = _country_from_zone_tab(zone_name=timezone_name) if timezone_name else None
    if country is None:
        country = _normalize_country_code(_text("country", max_len=8))
    if country is None and locale_country:
        country = locale_country
    if country is None and locale_name:
        country = _normalize_country_code(locale_name)
    if country:
        out["country"] = country
    offset = raw.get("timezone_offset_minutes")
    if isinstance(offset, (int, float)) and -14 * 60 <= float(offset) <= 14 * 60:
        out["timezone_offset_minutes"] = int(offset)

    if "local_datetime" in out:
        display_dt = str(out["local_datetime"]).replace("T", " ")
        display_dt = display_dt.replace("Z", "")
        display_dt = re.split(r"[+-]\d{2}:\d{2}$", display_dt)[0]
        out["display"] = f"[{display_dt[:19]} {out.get('country') or 'XX'}]"

    return out if len(out) > 1 else {}


def _runtime_grounding_metadata(trace_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    now = datetime.now().astimezone()
    country = _detect_country()
    timezone_name = _detect_timezone_name()
    user = _detect_runtime_user(trace_metadata)
    metadata: Dict[str, Any] = {
        "local_datetime": now.isoformat(timespec="seconds"),
        "country": country,
        "display": f"[{now.strftime('%Y-%m-%d %H:%M:%S')} {country}]",
        "source": "abstractruntime",
        "prompt_injected": False,
    }
    if timezone_name:
        metadata["timezone"] = timezone_name
    if user:
        metadata["user"] = user
    client_metadata = _client_grounding_metadata(trace_metadata)
    if client_metadata:
        server_context = {
            key: value
            for key, value in metadata.items()
            if key in {"local_datetime", "timezone", "country", "display", "source"}
        }
        merged = dict(metadata)
        for key in ("local_datetime", "timezone", "country", "display", "locale", "timezone_offset_minutes"):
            value = client_metadata.get(key)
            if value is not None and str(value).strip():
                merged[key] = value
        merged["source"] = "browser_untrusted"
        merged["server_context"] = server_context
        if user:
            merged["user"] = user
        return merged
    return metadata


# Contract line appended to the system prompt whenever a <runtime_metadata>
# envelope is injected into the user turn. Without it, models read the raw
# machine context as user-authored tokens and can infer a reply language or
# locale from it, even when the user's request is written in another language.
# The contract is stable text (no per-turn entropy) so prompt/KV caching is
# unaffected.
_RUNTIME_GROUNDING_CONTRACT = (
    "RUNTIME GROUNDING: a user message may begin with a machine-generated "
    "<runtime_metadata>{...}</runtime_metadata> envelope carrying the local "
    "date/time that message was sent (and optionally timezone, country, or OS user) "
    "for grounding; the newest one is the current time. It is not "
    "written by the user and it is not a language or locale preference. Always respond "
    "in the language of the user's request itself unless the user explicitly asks "
    "otherwise."
)

# Fields surfaced to the model inside the per-turn prompt envelope. The default
# is strictly temporal: A/B tests against a live 397B model (2026-06-10) showed
# that locale-correlated fields (country "FR", timezone "Europe/Paris", a
# French-looking OS user) flip replies into the host country's language for
# borderline requests, in 3/3 samples, even when the user prompt is entirely
# English, carries explicit language rules, and the system prompt explains the
# envelope. With the temporal-only envelope the same payload answered in the
# request language. Full grounding (timezone/country/user) remains available in
# result metadata (`runtime_grounding`) for hosts, and operators can opt fields
# back into the prompt via ABSTRACTRUNTIME_GROUNDING_PROMPT_FIELDS.
_DEFAULT_PROMPT_GROUNDING_FIELDS: tuple[str, ...] = ("local_datetime", "display")
_ALLOWED_PROMPT_GROUNDING_FIELDS = {"local_datetime", "timezone", "country", "user", "display"}
_DISPLAY_COUNTRY_SUFFIX_RE = re.compile(r"\s+[A-Z]{2}\]$")


def _prompt_grounding_fields() -> tuple[str, ...]:
    raw = os.getenv("ABSTRACTRUNTIME_GROUNDING_PROMPT_FIELDS")
    if raw is None or not raw.strip():
        return _DEFAULT_PROMPT_GROUNDING_FIELDS
    fields = tuple(field.strip() for field in raw.split(",") if field.strip())
    filtered = tuple(field for field in fields if field in _ALLOWED_PROMPT_GROUNDING_FIELDS)
    return filtered or _DEFAULT_PROMPT_GROUNDING_FIELDS


def _append_runtime_grounding_contract(system_prompt: Optional[str], *, injected: bool) -> Optional[str]:
    """Append the grounding contract to the system prompt when the envelope is injected.

    Idempotent: never duplicates the contract (e.g. composed prompts or replayed
    system text that already carries it).
    """
    if not injected:
        return system_prompt
    base = system_prompt if isinstance(system_prompt, str) else ""
    if _RUNTIME_GROUNDING_CONTRACT in base:
        return system_prompt
    if not base.strip():
        return _RUNTIME_GROUNDING_CONTRACT
    return f"{base.rstrip()}\n\n{_RUNTIME_GROUNDING_CONTRACT}"


def _runtime_grounding_prompt_envelope(grounding: Dict[str, Any]) -> str:
    fields = _prompt_grounding_fields()
    prompt_payload: Dict[str, Any] = {}
    for key in fields:
        value = grounding.get(key)
        if value is not None and str(value).strip():
            prompt_payload[key] = value
    # The display string embeds the ISO country code ("[... FR]"); strip it
    # when country is not an opted-in prompt field so ambient locale does not
    # leak into the model's language choice through the back door.
    if "display" in prompt_payload and "country" not in fields:
        prompt_payload["display"] = _DISPLAY_COUNTRY_SUFFIX_RE.sub("]", str(prompt_payload["display"]))
    return "<runtime_metadata>" + json.dumps(
        prompt_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "</runtime_metadata>"


def _strip_runtime_grounding_prefix(text: str) -> str:
    """Remove runtime-owned grounding prefixes while preserving user text."""
    if not isinstance(text, str) or not text.strip():
        return ""
    raw = str(text)
    while raw.strip():
        stripped = raw.lstrip()
        meta_match = _RUNTIME_METADATA_ENVELOPE_RE.match(stripped)
        if meta_match:
            raw = stripped[meta_match.end() :].lstrip()
            continue

        header_match = _SYSTEM_CONTEXT_HEADER_RE.match(stripped)
        if header_match:
            raw = stripped[header_match.end() :].lstrip()
            continue

        first_line = stripped.splitlines()[0].strip()
        if _LEGACY_SYSTEM_CONTEXT_HEADER_PARSE_RE.match(first_line):
            raw = "\n".join(stripped.splitlines()[1:]).lstrip()
            continue

        return stripped
    return ""


def _head_runtime_grounding_envelope(text: Any) -> Optional[str]:
    """Return the message's own head `<runtime_metadata>` envelope, or None.

    "Own" means exactly one well-formed envelope at the head, with no second
    envelope or legacy `Grounding:` header stacked behind it (that shape is a
    pre-0212 artifact and must still be normalized away).
    """
    if not isinstance(text, str) or not text.strip():
        return None
    match = _RUNTIME_METADATA_ENVELOPE_RE.match(text)
    if not match:
        return None
    rest = text[match.end() :]
    if _strip_runtime_grounding_prefix(rest) != rest.strip():
        return None  # stacked artifacts behind the head envelope
    return text[: match.end()]


def _content_carries_grounding_envelope(content: Any) -> bool:
    """True when this message content already carries its own head envelope.

    Handles both content shapes the payload boundary accepts: a plain string and
    an OpenAI-style content-part list (the first text part is the head).
    """
    if isinstance(content, str):
        return _head_runtime_grounding_envelope(content) is not None
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            if str(item.get("type") or "").strip().lower() != "text":
                continue
            return _head_runtime_grounding_envelope(item.get("text")) is not None
    return False


def _inject_runtime_grounding_into_text(text: str, grounding: Dict[str, Any]) -> str:
    """Stamp the grounding envelope ONCE, then keep it byte-for-byte.

    BYTE-STABILITY CONTRACT (mission A, 2026-09-22). The envelope carries a
    per-second timestamp. Stripping an envelope that is already there and
    re-injecting a fresh one made the SAME user turn render differently on every
    pass (runtime ledger pass, then client pass) and — far worse — differently
    from the bytes that turn was durably stored with, so the next conversational
    turn's prompt diverged at the START of the previous user message and no
    prefix cache could restore past it. Measured on the MLX native lane (4B pair,
    8-turn chat, untracked/missionA): 185-388 tokens re-prefilled EVERY turn, and
    the divergence grows with the conversation.

    An envelope already present is therefore the FINAL word for that message: its
    timestamp means "when this turn was sent", which is what it should have meant
    all along. Only a message that carries none gets a fresh stamp. Legacy
    `Grounding:` headers and stacked artifacts are still normalized away.
    """
    existing = _head_runtime_grounding_envelope(text)
    if existing is not None:
        return str(text)
    cleaned = _strip_runtime_grounding_prefix(text)
    envelope = _runtime_grounding_prompt_envelope(grounding)
    return f"{envelope}\n{cleaned}" if cleaned else envelope


def _strip_runtime_grounding_echo(text: Any) -> Any:
    if not isinstance(text, str) or not text.strip():
        return text
    return _strip_runtime_grounding_prefix(text)


def _sanitize_runtime_grounding_echoes(result: Dict[str, Any]) -> Dict[str, Any]:
    """Remove only runtime-owned metadata envelopes from user-facing response text."""
    if not isinstance(result, dict):
        return result
    for key in ("content", "response"):
        value = result.get(key)
        if isinstance(value, str):
            result[key] = _strip_runtime_grounding_echo(value)
    text_value = result.get("text")
    if isinstance(text_value, str):
        result["text"] = _strip_runtime_grounding_echo(text_value)
    elif isinstance(text_value, dict):
        text_dict = dict(text_value)
        for key in ("content", "response"):
            value = text_dict.get(key)
            if isinstance(value, str):
                text_dict[key] = _strip_runtime_grounding_echo(value)
        result["text"] = text_dict
    return result


def _normalize_turn_grounding(
    *,
    prompt: str,
    messages: Optional[List[Dict[str, Any]]],
    grounding: Optional[Dict[str, Any]] = None,
) -> tuple[str, Optional[List[Dict[str, Any]]]]:
    """Inject runtime context into the current turn for LLM calls only.

    The envelope is deliberately tagged and machine-owned:
      <runtime_metadata>{...}</runtime_metadata>

    That keeps date/location/user context visible to the model without mutating
    the durable human text field into a natural-language prefix that downstream
    TTS may speak. Media-only requests call this function with `grounding=None`,
    which strips legacy prefixes/artifacts but does not inject new prompt text.

    Placement (prompt-prefix cache stability, backlog 0212):
    - Chat shape (the last `user` message is the FINAL message): the envelope is
      injected at the head of that final user turn.
    - Tool-loop shape (messages continue past the last `user` message, e.g.
      `user task, assistant tool_calls, tool, ...`): rewriting the last user
      message would mutate message[0] with per-second timestamp entropy on every
      iteration and defeat provider prompt caching. Instead, the envelope rides
      a TRAILING, envelope-only `user` message appended after the transcript.
      Stale envelope-only messages from a previous pass are dropped first, so
      double normalization (runtime ledger pass + client pass) stays idempotent.

    STAMP ONCE (mission A, 2026-09-22). Both rules above now apply only to a user
    turn that carries NO envelope yet. A turn that already carries one keeps it
    byte-for-byte, and the tool-loop shape then appends no trailer at all. The
    envelope's timestamp therefore means "when this turn was sent", and the bytes
    a turn is sent with are the bytes it can be replayed with — which is the whole
    precondition for a conversational prompt cache: turn N's prompt must be an
    exact byte-prefix of turn N+1's. Adapters that own a durable transcript stamp
    it there once (`abstractruntime.turn_grounding.stamp_user_turn_grounding`,
    called by abstractagent's react/codeact/memact `reason` boundary) and the
    session replay path returns that stamped content verbatim
    (`abstractruntime.session_history`). Hosts that stamp nothing keep the old
    per-call behaviour, one pass later.
    """

    def _clean_or_inject_text(value: str) -> str:
        if grounding:
            return _inject_runtime_grounding_into_text(value, grounding)
        return _strip_runtime_grounding_prefix(value)

    def _strip_text(value: str) -> str:
        return _strip_runtime_grounding_prefix(value)

    def _apply_to_content(content: Any, *, text_fn) -> Any:
        if isinstance(content, str):
            return text_fn(content)
        if isinstance(content, list):
            items: List[Any] = [dict(item) if isinstance(item, dict) else item for item in content]
            for idx, item in enumerate(items):
                if not isinstance(item, dict):
                    continue
                if str(item.get("type") or "").strip().lower() != "text":
                    continue
                text_value = item.get("text")
                item["text"] = text_fn(text_value if isinstance(text_value, str) else str(text_value or ""))
                items[idx] = item
                return items
            if grounding and text_fn is _clean_or_inject_text:
                items.insert(0, {"type": "text", "text": _runtime_grounding_prompt_envelope(grounding)})
            return items
        return text_fn(str(content or ""))

    prompt_str = str(prompt or "")
    if prompt_str.strip():
        return _clean_or_inject_text(prompt_str), messages

    if isinstance(messages, list) and messages:
        out: List[Dict[str, Any]] = []
        for m in messages:
            entry = dict(m) if isinstance(m, dict) else {"role": "user", "content": str(m)}
            # Runtime-owned injection artifact from a previous normalization pass:
            # drop it here and (when grounding is active) re-append a fresh one below.
            if _is_runtime_grounding_only_user_message(entry):
                continue
            out.append(entry)

        # SYNTHESIZED CARRIERS ARE NOT THE TURN (mission A3, 2026-09-22).
        # abstractagent's payload-boundary repair folds a tool result whose call id
        # nothing announced into a `user` message (`[unpaired tool result]: …`),
        # marked `_af_synthetic`. In a tool loop that carrier is the LAST message,
        # so the scan below used to classify the payload as "chat shape" and inject
        # the envelope at its head — into a message the NEXT iteration rebuilds
        # from the durable transcript without it, stamping the newer carrier
        # instead. Measured on live run 081d8daa: consecutive prompts identical for
        # messages 0..12 and 118 chars apart (exactly one envelope) at index 13 of
        # 21 — thousands of tokens before the prompt's end, so nothing restored.
        # Skipping carriers puts `last_user_idx` back on the durable turn, which
        # the adapter already stamped, and the tool-loop branch then leaves the
        # whole transcript byte-for-byte alone.
        last_user_idx: Optional[int] = None
        for i in range(len(out) - 1, -1, -1):
            role = str(out[i].get("role") or "").strip().lower()
            if role != "user":
                continue
            if _message_is_synthetic_carrier(out[i]):
                continue
            # A `volatile` message is per-call BY DECLARATION (the loops' old
            # trailing position line; hosts may still emit one). It is gone from
            # the next call's payload, so an envelope written into it is written
            # into bytes the next call will not have — measured on the hermetic
            # gateway (mission A3): `[loop] iteration N of 8.` was the last user
            # message, was read as "chat shape" and stamped on every iteration.
            if _is_volatile_message(out[i]):
                continue
            last_user_idx = i
            break

        if last_user_idx is None:
            if grounding:
                out.append({"role": "user", "content": _runtime_grounding_prompt_envelope(grounding)})
            return prompt_str, out

        if last_user_idx == len(out) - 1:
            # Chat shape: the final message is the current user turn.
            out[last_user_idx]["content"] = _apply_to_content(
                out[last_user_idx].get("content"), text_fn=_clean_or_inject_text
            )
            return prompt_str, out

        # Tool-loop shape.
        #
        # When the durable user turn ALREADY carries its own envelope (stamped once,
        # when the turn was created — see `_inject_runtime_grounding_into_text`), it
        # is the grounding for this turn: leave it exactly as stored and append
        # nothing. That is what makes iteration N's prompt an exact byte-prefix of
        # iteration N+1's for hosts with no volatile tail, and it removes the ~50
        # trailing tokens of per-iteration entropy for hosts that have one. The cost
        # is that "now" inside a long tool loop is the time the TURN was sent, not
        # the time this iteration started — seconds to minutes, on a field whose
        # resolution is the second and whose purpose is "what day/time is it".
        #
        # When it carries none (a host that never stamped one, or a legacy
        # `Grounding:` header), the pre-0212 behaviour stands: strip the artifact
        # from the earlier user turn — rewriting it with per-second entropy on every
        # iteration would mutate message[0] and defeat prompt caching — and carry
        # fresh grounding in a TRAILING envelope-only user message instead.
        # `grounding=None` (media-only calls) still means CLEAN: a stamped turn is
        # kept only while grounding is active — the strip path below is the one
        # that removes envelope artifacts when the caller asked for none.
        #
        # ANY stamped user message grounds the turn (mission A3). The old test
        # looked only at `out[last_user_idx]`; after an operator interjection is
        # drained mid-loop the newest durable user message is that interjection,
        # which `stamp_user_turn_grounding` stamps, while the ORIGINAL task
        # several messages back carries the turn's envelope. Either way the model
        # has the local time, so a trailer would only add per-iteration entropy at
        # the one place the cache cannot afford it — the end of the prompt.
        if grounding and _transcript_carries_grounding_envelope(out):
            return prompt_str, out
        out[last_user_idx]["content"] = _apply_to_content(out[last_user_idx].get("content"), text_fn=_strip_text)
        if grounding:
            if str(out[-1].get("role") or "").strip().lower() == "user":
                # The payload already ends with an adapter-authored user message (a
                # loop tail or a tool-result carrier, mission A3). A second trailing
                # user message would be a user,user pair, which alternation-strict
                # templates reject; ride the head of that message instead. Per-call
                # bytes either way — this legacy path is for hosts that never stamp.
                last = dict(out[-1])
                last["content"] = _apply_to_content(last.get("content"), text_fn=_clean_or_inject_text)
                out[-1] = last
            else:
                out.append({"role": "user", "content": _runtime_grounding_prompt_envelope(grounding)})
        return prompt_str, out

    return prompt_str, messages


def _mark_grounding_prompt_injected(grounding: Dict[str, Any], injected: bool) -> Dict[str, Any]:
    out = dict(grounding)
    out["prompt_injected"] = bool(injected)
    return {
        **out,
    }


def _attach_runtime_grounding(result: Dict[str, Any], grounding: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not grounding:
        return result
    meta = result.get("metadata")
    if not isinstance(meta, dict):
        meta = {}
        result["metadata"] = meta
    meta["runtime_grounding"] = dict(grounding)
    return result


def _maybe_parse_tool_calls_from_text(
    *,
    content: Optional[str],
    allowed_tool_names: Optional[set[str]] = None,
    model_name: Optional[str] = None,
    tool_handler: Any = None,
) -> tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """Deprecated: tool-call parsing belongs to AbstractCore.

    AbstractCore now normalizes non-streaming responses by populating structured `tool_calls`
    and returning cleaned `content`. This helper remains only for backward compatibility with
    older AbstractCore versions and will be removed in the next major release.
    """
    # Keep behavior for external callers/tests that still import it.
    if not isinstance(content, str) or not content.strip():
        return None, None
    if tool_handler is None:
        from abstractcore.tools.handler import UniversalToolHandler

        tool_handler = UniversalToolHandler(str(model_name or ""))

    try:
        parsed = tool_handler.parse_response(content, mode="prompted")
    except Exception:
        return None, None

    calls = getattr(parsed, "tool_calls", None)
    cleaned = getattr(parsed, "content", None)
    if not isinstance(calls, list) or not calls:
        return None, None

    out_calls: List[Dict[str, Any]] = []
    for tc in calls:
        name = getattr(tc, "name", None)
        arguments = getattr(tc, "arguments", None)
        call_id = getattr(tc, "call_id", None)
        if not isinstance(name, str) or not name.strip():
            continue
        if isinstance(allowed_tool_names, set) and allowed_tool_names and name not in allowed_tool_names:
            continue
        out_calls.append(
            {
                "name": name.strip(),
                "arguments": _jsonable(arguments) if arguments is not None else {},
                "call_id": str(call_id) if call_id is not None else None,
            }
        )

    if not out_calls:
        return None, None
    return out_calls, (str(cleaned) if isinstance(cleaned, str) else "")


@dataclass(frozen=True)
class HttpResponse:
    body: Dict[str, Any]
    headers: Dict[str, str]


@dataclass(frozen=True)
class HttpBinaryResponse:
    content: bytes
    headers: Dict[str, str]


class RequestSender(Protocol):
    def get(
        self,
        url: str,
        *,
        headers: Dict[str, str],
        timeout: float,
    ) -> Any: ...

    def post(
        self,
        url: str,
        *,
        headers: Dict[str, str],
        json: Dict[str, Any],
        timeout: float,
    ) -> Any: ...

    def post_jsonl_stream(
        self,
        url: str,
        *,
        headers: Dict[str, str],
        json: Dict[str, Any],
        timeout: float,
    ) -> Any: ...


class AbstractCoreLLMClient(Protocol):
    def default_prompt_cache_identity(self) -> Tuple[Optional[str], Optional[str]]:
        """Return the default provider/model identity used to partition derived prompt-cache keys."""

    def generate(
        self,
        *,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        media: Optional[List[Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return a JSON-safe dict with at least: content/tool_calls/usage/model."""

    def stream_tts(
        self,
        *,
        text: str,
        output: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Yield JSON-safe TTS stream events and finalize a durable artifact on successful completion."""

    def get_model_capabilities(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Return model capability metadata for a specific model or the default client model."""

    def get_execution_capabilities(
        self, model_name: Optional[str] = None, *, provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return execution-host capabilities without constructing/loading a provider."""

    def get_prompt_cache_capabilities(self, **kwargs: Any) -> Dict[str, Any]:
        """Return a JSON-safe prompt-cache capability payload."""

    def get_prompt_cache_stats(self, **kwargs: Any) -> Dict[str, Any]:
        """Return a JSON-safe prompt-cache stats payload."""

    def prompt_cache_set(
        self,
        *,
        key: str,
        make_default: bool = True,
        ttl_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Set or select a prompt cache key."""

    def prompt_cache_update(
        self,
        *,
        key: str,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        add_generation_prompt: bool = False,
        ttl_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Append content into a prompt cache key."""

    def prompt_cache_fork(
        self,
        *,
        from_key: str,
        to_key: str,
        make_default: bool = False,
        ttl_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Fork one prompt cache key into another."""

    def prompt_cache_clear(
        self,
        *,
        key: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Clear one prompt cache key or the whole in-process cache."""

    def prompt_cache_prepare_modules(
        self,
        *,
        namespace: str,
        modules: List[Dict[str, Any]],
        make_default: bool = False,
        ttl_s: Optional[float] = None,
        version: int = 1,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Prepare hierarchical prompt-cache modules."""

    def list_prompt_cache_exports(
        self,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """List host-local exported prompt-cache artifacts for one provider/model runtime target."""

    def prompt_cache_export(
        self,
        *,
        name: str,
        key: str,
        q8: bool = False,
        meta: Optional[Dict[str, Any]] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Export a live local provider prompt cache to a durable host-local artifact."""

    def prompt_cache_import(
        self,
        *,
        name: str,
        key: Optional[str] = None,
        make_default: bool = True,
        clear_existing: bool = False,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Import one previously exported host-local prompt-cache artifact."""

    def upsert_text_bloc(
        self,
        *,
        path: str,
        content: str,
        sha256: Optional[str] = None,
        content_sha256: Optional[str] = None,
        media_type: str = "text",
        size_bytes: Optional[int] = None,
        mtime_ns: Optional[int] = None,
        format: Optional[str] = None,
        estimated_tokens: Optional[int] = None,
        relpath_base: Optional[str] = None,
        summary: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Persist or update one durable text bloc."""

    def get_bloc_record(
        self,
        *,
        sha256: Optional[str] = None,
        bloc_id: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Read one durable bloc record by sha256 or bloc_id."""

    def list_blocs(
        self,
        *,
        sha256: Optional[str] = None,
        bloc_id: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """List durable bloc records, optionally filtered by sha256 or bloc_id."""

    def get_bloc_kv_manifest(
        self,
        *,
        sha256: Optional[str] = None,
        bloc_id: Optional[int] = None,
        artifact_path: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Inspect one durable bloc KV manifest."""

    def ensure_bloc_kv_artifact(
        self,
        *,
        sha256: Optional[str] = None,
        bloc_id: Optional[int] = None,
        artifact_path: Optional[str] = None,
        force_rebuild: bool = False,
        debug: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Compile or validate one durable bloc KV artifact."""

    def load_bloc_kv_artifact(
        self,
        *,
        sha256: Optional[str] = None,
        bloc_id: Optional[int] = None,
        artifact_path: Optional[str] = None,
        stable_cache_key: Optional[str] = None,
        key: Optional[str] = None,
        make_default: bool = False,
        force_rebuild: bool = False,
        debug: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Load or fork one durable bloc KV artifact into a prompt-cache key."""

    def list_bloc_kv_artifacts(
        self,
        *,
        sha256: Optional[str] = None,
        bloc_id: Optional[int] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """List durable bloc KV artifacts, optionally filtered by bloc/provider/model."""

    def delete_bloc_kv_artifact(
        self,
        *,
        sha256: Optional[str] = None,
        bloc_id: Optional[int] = None,
        artifact_path: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        clear_loaded: bool = False,
        force: bool = False,
        dry_run: bool = False,
        debug: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Delete one durable bloc KV artifact with optional live-binding safety."""

    def prune_bloc_kv_artifacts(
        self,
        *,
        sha256: Optional[str] = None,
        bloc_id: Optional[int] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        clear_loaded: bool = False,
        force: bool = False,
        dry_run: bool = False,
        debug: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Delete matching durable bloc KV artifacts by filter."""

    def delete_bloc(
        self,
        *,
        sha256: Optional[str] = None,
        bloc_id: Optional[int] = None,
        delete_kv: bool = True,
        clear_loaded: bool = False,
        force: bool = False,
        dry_run: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Delete one durable bloc and optionally its derived KV artifacts."""


class AbstractCoreControlClient(Protocol):
    """Runtime/provider control-plane calls exposed by AbstractCore-capable clients."""

    def get_model_residency_capabilities(self, **kwargs: Any) -> Dict[str, Any]:
        """Return task-level model residency support truth for this client."""

    def list_model_residency(
        self,
        *,
        task: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Return a JSON-safe snapshot of currently resident models."""

    def load_model_residency(
        self,
        *,
        task: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        pin: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Request idempotent model residency."""

    def unload_model_residency(
        self,
        *,
        task: Optional[str] = None,
        runtime_id: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        force: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Request best-effort model unload (locked runtimes refuse without force)."""

    def lock_model_residency(
        self,
        payload: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Lock a warm model runtime against unloading (payload mapping and/or kwargs; kwargs win)."""

    def unlock_model_residency(
        self,
        payload: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Clear a model runtime lock (payload mapping and/or kwargs; kwargs win)."""

    def get_context_estimate(
        self,
        payload: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Relay core's analytical context-fit estimate (payload mapping and/or kwargs; kwargs win)."""


def _jsonable(value: Any) -> Any:
    """Best-effort conversion to JSON-safe objects.

    This is intentionally conservative: if a value isn't naturally JSON-serializable,
    we fall back to `str(value)`.
    """

    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if is_dataclass(value):
        return _jsonable(asdict(value))

    # Pydantic v2
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _jsonable(model_dump())

    # Pydantic v1
    to_dict = getattr(value, "dict", None)
    if callable(to_dict):
        return _jsonable(to_dict())

    return str(value)


def _env_flag_enabled(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"0", "false", "no", "off", "disable", "disabled"}


def _loads_last_json_line(text: str) -> Dict[str, Any]:
    """Parse the last JSON object line from a subprocess stream.

    Some native model stacks write progress to stdout. The worker prints a single JSON
    line as its final record, so parsing from the bottom keeps the transport robust.
    """

    for line in reversed(str(text or "").splitlines()):
        candidate = line.strip()
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Subprocess did not return a JSON response.")


def _decode_subprocess_media_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = _jsonable(payload)
    if not isinstance(out, dict):
        raise ValueError("Subprocess returned an invalid media response.")

    outputs = out.get("outputs")
    if isinstance(outputs, dict):
        for items in outputs.values():
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                data_b64 = item.pop("data_b64", None)
                if isinstance(data_b64, str) and data_b64:
                    item["data"] = base64.b64decode(data_b64.encode("ascii"))
    return out


def _is_subprocess_safe_image_specs(specs: List[Dict[str, Any]], media: Optional[List[Any]]) -> bool:
    if not _env_flag_enabled("ABSTRACTRUNTIME_LOCAL_IMAGE_SUBPROCESS", default=True):
        return False
    if not specs:
        return False
    normalized_media = list(media or [])
    for spec in specs:
        if not isinstance(spec, dict):
            return False
        modality = str(spec.get("modality") or "").strip().lower()
        task = str(spec.get("task") or "").strip().lower()
        if modality != "image":
            return False
        if task in {"", "image_generation", "t2i", "text_to_image"}:
            if normalized_media:
                return False
            continue
        if task in {"image_edit", "image_to_image", "i2i", "edit_image"}:
            image_items = [item for item in normalized_media if _is_image_media_item(item)]
            if not image_items:
                return False
            source_items = [
                item
                for item in image_items
                if str(item.get("role") or item.get("purpose") or "").strip().lower() == "source"
            ]
            mask_items = [
                item
                for item in image_items
                if str(item.get("role") or item.get("purpose") or "").strip().lower() == "mask"
            ]
            source_item = source_items[0] if source_items else image_items[0]
            source_path = _media_path_from_item(source_item)
            if not source_path or source_path.lower().startswith("data:") or source_path.startswith(("http://", "https://")):
                return False
            if len(mask_items) > 1:
                return False
            if mask_items:
                mask_path = _media_path_from_item(mask_items[0])
                if not mask_path or mask_path.lower().startswith("data:") or mask_path.startswith(("http://", "https://")):
                    return False
            continue
        if task in {"image_upscale", "image_upscaling", "upscale", "upscale_image"}:
            image_items = [item for item in normalized_media if _is_image_media_item(item)]
            if len(image_items) != 1:
                return False
            path = _media_path_from_item(image_items[0])
            if not path or path.lower().startswith("data:") or path.startswith(("http://", "https://")):
                return False
            continue
        if task:
            return False
    return True


def _is_subprocess_safe_video_specs(specs: List[Dict[str, Any]], media: Optional[List[Any]]) -> bool:
    if not _env_flag_enabled("ABSTRACTRUNTIME_LOCAL_VIDEO_SUBPROCESS", default=True):
        return False
    if len(specs) != 1 or not isinstance(specs[0], dict):
        return False
    spec = specs[0]
    modality = str(spec.get("modality") or "").strip().lower()
    task = str(spec.get("task") or "").strip().lower()
    if modality != "video":
        return False
    if task in {"", "video_generation", "text_to_video", "t2v"}:
        return not media
    if task not in {"image_to_video", "i2v", "video_from_image", "video_edit"}:
        return False

    image_items = [item for item in list(media or []) if _is_image_media_item(item)]
    if len(image_items) != 1 or len(list(media or [])) != 1:
        return False
    path = _media_path_from_item(image_items[0])
    return bool(path and not path.lower().startswith("data:") and not path.startswith(("http://", "https://")))


def _run_local_image_subprocess(
    *,
    provider: str,
    model: str,
    llm_kwargs: Dict[str, Any],
    prompt: str,
    specs: List[Dict[str, Any]],
    media: Optional[List[Any]] = None,
    progress_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    request = {
        "provider": str(provider or ""),
        "model": str(model or ""),
        "llm_kwargs": _jsonable(llm_kwargs or {}),
        "prompt": str(prompt or ""),
        "specs": _jsonable(specs),
        "media": _jsonable(media or []),
    }
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")

    def _invoke() -> subprocess.Popen[str]:
        return subprocess.Popen(
            [sys.executable, "-m", "abstractruntime.integrations.abstractcore.media_subprocess"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

    if _env_flag_enabled("ABSTRACTRUNTIME_LOCAL_IMAGE_SUBPROCESS_SERIALIZE", default=True):
        lock = _LOCAL_IMAGE_SUBPROCESS_LOCK
    else:
        lock = threading.Lock()

    with lock:
        proc = _invoke()
        assert proc.stdin is not None
        proc.stdin.write(json.dumps(request, ensure_ascii=False))
        proc.stdin.close()

        parsed: Optional[Dict[str, Any]] = None
        output_tail: List[str] = []
        selector = selectors.DefaultSelector()
        if proc.stdout is not None:
            _register_subprocess_stream(selector, proc.stdout, "stdout")
        proc_stderr = getattr(proc, "stderr", None)
        if proc_stderr is not None:
            _register_subprocess_stream(selector, proc_stderr, "stderr")

        def _consume_stdout_line(line: str) -> None:
            nonlocal parsed
            stripped = str(line).strip()
            if stripped:
                output_tail.append(stripped)
                del output_tail[:-40]
            try:
                item = json.loads(stripped)
            except Exception:
                return
            if not isinstance(item, dict):
                return
            if item.get("type") == "progress":
                event = item.get("event")
                if callable(progress_callback) and isinstance(event, dict):
                    progress_callback(event)
                return
            parsed = item

        def _consume_diagnostic_line(line: str) -> None:
            stripped = str(line).strip()
            if stripped:
                output_tail.append(stripped)
                del output_tail[:-40]

        try:
            while True:
                ready = selector.select(timeout=0.25)
                if not ready:
                    if proc.poll() is not None:
                        break
                    continue
                for key, _mask in ready:
                    line = key.fileobj.readline()
                    if not line:
                        if proc.poll() is not None:
                            break
                        continue
                    if _subprocess_stream_label(key, stdout=proc.stdout, stderr=proc_stderr) == "stdout":
                        _consume_stdout_line(line)
                    else:
                        _consume_diagnostic_line(line)
                if proc.poll() is not None:
                    break
            if proc.stdout is not None:
                try:
                    remaining = proc.stdout.read()
                except Exception:
                    remaining = ""
                for line in str(remaining or "").splitlines():
                    _consume_stdout_line(line)
            if proc_stderr is not None:
                try:
                    remaining_err = proc_stderr.read()
                except Exception:
                    remaining_err = ""
                for line in str(remaining_err or "").splitlines():
                    _consume_diagnostic_line(line)
        finally:
            try:
                selector.close()
            except Exception:
                pass

        returncode = proc.wait()

    if returncode != 0:
        if parsed and parsed.get("ok") is False:
            detail = str(parsed.get("error") or "")
        else:
            detail = "\n".join(output_tail).strip() or "local image generation failed"
        raise RuntimeError(
            f"Local image generation subprocess exited with code {returncode}: {detail[-2000:]}"
        )
    if not isinstance(parsed, dict):
        raise RuntimeError("Local image generation subprocess returned no parseable response.")
    if parsed.get("ok") is False:
        raise RuntimeError(str(parsed.get("error") or "local image generation failed"))
    response = parsed.get("response")
    if not isinstance(response, dict):
        raise RuntimeError("Local image generation subprocess returned an invalid response.")
    return _decode_subprocess_media_payload(response)


def _run_local_video_subprocess(
    *,
    provider: str,
    model: str,
    llm_kwargs: Dict[str, Any],
    prompt: str,
    specs: List[Dict[str, Any]],
    media: Optional[List[Any]],
    progress_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    request = {
        "provider": str(provider or ""),
        "model": str(model or ""),
        "llm_kwargs": _jsonable(llm_kwargs or {}),
        "prompt": str(prompt or ""),
        "specs": _jsonable(specs),
        "media": _jsonable(media or []),
    }
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")

    def _invoke() -> subprocess.Popen[str]:
        return subprocess.Popen(
            [sys.executable, "-m", "abstractruntime.integrations.abstractcore.media_subprocess"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

    if _env_flag_enabled("ABSTRACTRUNTIME_LOCAL_VIDEO_SUBPROCESS_SERIALIZE", default=True):
        lock = _LOCAL_IMAGE_SUBPROCESS_LOCK
    else:
        lock = threading.Lock()

    with lock:
        proc = _invoke()
        assert proc.stdin is not None
        proc.stdin.write(json.dumps(request, ensure_ascii=False))
        proc.stdin.close()

        parsed: Optional[Dict[str, Any]] = None
        output_tail: List[str] = []
        selector = selectors.DefaultSelector()
        if proc.stdout is not None:
            _register_subprocess_stream(selector, proc.stdout, "stdout")
        proc_stderr = getattr(proc, "stderr", None)
        if proc_stderr is not None:
            _register_subprocess_stream(selector, proc_stderr, "stderr")

        def _consume_stdout_line(line: str) -> None:
            nonlocal parsed
            stripped = str(line).strip()
            if stripped:
                output_tail.append(stripped)
                del output_tail[:-40]
            try:
                item = json.loads(stripped)
            except Exception:
                return
            if not isinstance(item, dict):
                return
            if item.get("type") == "progress":
                event = item.get("event")
                if callable(progress_callback) and isinstance(event, dict):
                    progress_callback(event)
                return
            parsed = item

        def _consume_diagnostic_line(line: str) -> None:
            stripped = str(line).strip()
            if stripped:
                output_tail.append(stripped)
                del output_tail[:-40]

        try:
            while True:
                ready = selector.select(timeout=0.25)
                if not ready:
                    if proc.poll() is not None:
                        break
                    continue
                for key, _mask in ready:
                    line = key.fileobj.readline()
                    if not line:
                        if proc.poll() is not None:
                            break
                        continue
                    if _subprocess_stream_label(key, stdout=proc.stdout, stderr=proc_stderr) == "stdout":
                        _consume_stdout_line(line)
                    else:
                        _consume_diagnostic_line(line)
                if proc.poll() is not None:
                    break
            if proc.stdout is not None:
                try:
                    remaining = proc.stdout.read()
                except Exception:
                    remaining = ""
                for line in str(remaining or "").splitlines():
                    _consume_stdout_line(line)
            if proc_stderr is not None:
                try:
                    remaining_err = proc_stderr.read()
                except Exception:
                    remaining_err = ""
                for line in str(remaining_err or "").splitlines():
                    _consume_diagnostic_line(line)
        finally:
            try:
                selector.close()
            except Exception:
                pass

        returncode = proc.wait()

    if returncode != 0:
        detail = ""
        if parsed and parsed.get("ok") is False:
            detail = str(parsed.get("error") or "")
        if not detail:
            detail = "\n".join(output_tail).strip() or "local video generation failed"
        raise RuntimeError(
            f"Local video generation subprocess exited with code {returncode}: {detail[-2000:]}"
        )
    if not isinstance(parsed, dict):
        raise RuntimeError("Local video generation subprocess returned no parseable response.")
    if parsed.get("ok") is False:
        raise RuntimeError(str(parsed.get("error") or "local video generation failed"))
    response = parsed.get("response")
    if not isinstance(response, dict):
        raise RuntimeError("Local video generation subprocess returned an invalid response.")
    return _decode_subprocess_media_payload(response)


def _prompt_cache_capabilities_payload(provider: Any) -> Dict[str, Any]:
    if provider is None:
        return {"supported": False, "capabilities": {"supported": False, "mode": "none"}}

    getter = getattr(provider, "get_prompt_cache_capabilities", None)
    if callable(getter):
        try:
            caps = getter()
            to_dict = getattr(caps, "to_dict", None)
            if callable(to_dict):
                return {"supported": bool(getattr(caps, "supported", False)), "capabilities": to_dict()}
            if isinstance(caps, dict):
                return {"supported": bool(caps.get("supported")), "capabilities": dict(caps)}
        except Exception as e:
            return {"supported": False, "error": str(e), "capabilities": {"supported": False, "mode": "none"}}

    try:
        supported = bool(getattr(provider, "supports_prompt_cache", lambda: False)())
    except Exception:
        supported = False
    mode = "keyed" if supported else "none"
    return {
        "supported": supported,
        "capabilities": {
            "supported": supported,
            "mode": mode,
        },
    }


def _prompt_cache_capabilities_dict(provider: Any) -> Dict[str, Any]:
    info = _prompt_cache_capabilities_payload(provider)
    caps = info.get("capabilities") if isinstance(info, dict) else None
    if isinstance(caps, dict):
        return dict(caps)
    return {"supported": False, "mode": "none"}


def _prompt_cache_supports(provider: Any, operation: str) -> bool:
    try:
        fn = getattr(provider, "prompt_cache_supports_operation", None)
        if callable(fn):
            return bool(fn(operation))
    except Exception:
        return False

    caps = _prompt_cache_capabilities_dict(provider)
    op = str(operation or "").strip().lower()
    if op == "stats":
        return bool(caps.get("supports_stats"))
    if op == "set":
        return bool(caps.get("supports_set"))
    if op == "clear":
        return bool(caps.get("supports_clear"))
    if op == "update":
        return bool(caps.get("supports_update"))
    if op == "fork":
        return bool(caps.get("supports_fork"))
    if op in {"prepare", "prepare_modules", "modules"}:
        return bool(caps.get("supports_prepare_modules"))
    if op == "save":
        return bool(caps.get("supports_save"))
    if op == "load":
        return bool(caps.get("supports_load"))
    return False


def _prompt_cache_error_payload(provider: Any, *, operation: str, error: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "supported": False,
        "operation": str(operation or "").strip(),
        "capabilities": _prompt_cache_capabilities_dict(provider),
    }

    to_dict = getattr(error, "to_dict", None)
    if callable(to_dict):
        try:
            data = to_dict()
        except Exception:
            data = {}
        if isinstance(data, dict):
            payload["code"] = str(data.get("code") or "prompt_cache_error")
            payload["error"] = str(data.get("message") or error)
            if isinstance(data.get("capabilities"), dict):
                payload["capabilities"] = dict(data["capabilities"])
            return payload

    payload["code"] = "prompt_cache_error"
    payload["error"] = str(error)
    return payload


def _prompt_cache_unsupported_payload(provider: Any, *, operation: str, error: str) -> Dict[str, Any]:
    return {
        "supported": False,
        "operation": str(operation or "").strip(),
        "code": "prompt_cache_unsupported",
        "error": str(error),
        "capabilities": _prompt_cache_capabilities_dict(provider),
    }


def _runtime_default_blocs_root_dir() -> Path:
    return Path.home() / ".abstractruntime" / "blocs"


def _coerce_bloc_root_dir(root_dir: Any) -> Path:
    if isinstance(root_dir, Path):
        raw = str(root_dir).strip()
    elif isinstance(root_dir, str):
        raw = root_dir.strip()
    else:
        raw = ""
    if not raw:
        return _runtime_default_blocs_root_dir()
    return Path(raw).expanduser()


_RUNTIME_PROMPT_CACHE_EXPORT_SCHEMA = "abstractruntime-prompt-cache-export/v1"
_PROMPT_CACHE_EXPORT_META_SUFFIX = ".meta.json"


def _runtime_default_prompt_cache_export_root_dir() -> Path:
    return Path.home() / ".abstractruntime" / "prompt_cache_exports"


def _coerce_prompt_cache_export_root_dir(root_dir: Any) -> Path:
    if isinstance(root_dir, Path):
        raw = str(root_dir).strip()
    elif isinstance(root_dir, str):
        raw = root_dir.strip()
    else:
        raw = ""
    if not raw:
        return _runtime_default_prompt_cache_export_root_dir()
    return Path(raw).expanduser()


def _prompt_cache_export_slug(value: Any, *, fallback: str) -> str:
    text = str(value or "").strip()
    if not text:
        text = fallback
    text = text.replace("/", "-").replace("\\", "-")
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text)
    text = text.strip("._-")
    return text or fallback


def _prompt_cache_export_partition_component(value: Any, *, fallback: str) -> str:
    text = str(value or "").strip()
    if not text:
        text = fallback
    return quote(text, safe="") or fallback


def _prompt_cache_export_name(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("name is required")
    return _prompt_cache_export_slug(text, fallback="prompt-cache-export")


def _prompt_cache_export_token_count(value: Any) -> Optional[int]:
    try:
        parsed = int(value)
    except Exception:
        return None
    if parsed < 0:
        return None
    return parsed


def _prompt_cache_artifact_extension(provider: Any) -> str:
    getter = getattr(provider, "prompt_cache_artifact_extension", None)
    if callable(getter):
        try:
            value = str(getter() or "").strip()
        except Exception:
            value = ""
        if value:
            return value if value.startswith(".") else f".{value}"
    return ".bin"


def _prompt_cache_artifact_format(provider: Any) -> Optional[str]:
    getter = getattr(provider, "prompt_cache_artifact_format", None)
    if callable(getter):
        try:
            value = str(getter() or "").strip()
        except Exception:
            value = ""
        if value:
            return value
    return None


def _prompt_cache_export_dir(*, root_dir: Path, provider: str, model: str) -> Path:
    return root_dir / _prompt_cache_export_partition_component(
        provider, fallback="unknown-provider"
    ) / _prompt_cache_export_partition_component(model, fallback="unknown-model")


def _prompt_cache_export_paths(
    *,
    root_dir: Path,
    provider: str,
    model: str,
    name: str,
    extension: str,
) -> Tuple[str, Path, Path]:
    normalized_name = _prompt_cache_export_name(name)
    artifact_extension = extension if str(extension or "").startswith(".") else f".{extension}"
    directory = _prompt_cache_export_dir(root_dir=root_dir, provider=provider, model=model)
    artifact_filename = f"{normalized_name}{artifact_extension}"
    artifact_path = directory / artifact_filename
    meta_path = directory / f"{artifact_filename}{_PROMPT_CACHE_EXPORT_META_SUFFIX}"
    return normalized_name, artifact_path, meta_path


def _prompt_cache_export_item_from_record(meta: Dict[str, Any], *, artifact_path: Path, meta_path: Path) -> Dict[str, Any]:
    item = dict(meta)
    artifact_filename = str(item.get("artifact_filename") or artifact_path.name).strip() or artifact_path.name
    name = str(item.get("name") or artifact_path.stem).strip() or artifact_path.stem
    token_count = _prompt_cache_export_token_count(item.get("token_count"))
    if token_count is not None:
        item["token_count"] = token_count
    item.update(
        {
            "name": name,
            "provider": str(item.get("provider") or "").strip() or None,
            "model": str(item.get("model") or "").strip() or None,
            "artifact_filename": artifact_filename,
            "artifact_path": str(artifact_path),
            "artifact_exists": artifact_path.exists(),
            "artifact_extension": str(item.get("artifact_extension") or artifact_path.suffix or "").strip() or None,
            "artifact_format": str(item.get("artifact_format") or "").strip() or None,
            "meta_path": str(meta_path),
        }
    )
    return {
        "name": item["name"],
        "provider": item["provider"],
        "model": item["model"],
        "saved_at": item.get("saved_at"),
        "token_count": item.get("token_count"),
        "key": item.get("key"),
        "artifact_filename": item["artifact_filename"],
        "artifact_path": item["artifact_path"],
        "artifact_exists": item["artifact_exists"],
        "artifact_extension": item["artifact_extension"],
        "artifact_format": item["artifact_format"],
        "meta_path": item["meta_path"],
        "meta": item,
    }


def _read_prompt_cache_export_record(meta_path: Path) -> Optional[Dict[str, Any]]:
    try:
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None
    artifact_filename = str(raw.get("artifact_filename") or "").strip()
    if not artifact_filename:
        if not meta_path.name.endswith(_PROMPT_CACHE_EXPORT_META_SUFFIX):
            return None
        artifact_filename = meta_path.name[: -len(_PROMPT_CACHE_EXPORT_META_SUFFIX)]
    artifact_path = meta_path.parent / artifact_filename
    return _prompt_cache_export_item_from_record(raw, artifact_path=artifact_path, meta_path=meta_path)


def _list_prompt_cache_exports_local(
    *,
    root_dir: Path,
    provider: Any,
    provider_name: str,
    model: str,
) -> Dict[str, Any]:
    export_dir = _prompt_cache_export_dir(root_dir=root_dir, provider=provider_name, model=model)
    items: List[Dict[str, Any]] = []
    if export_dir.exists():
        for meta_path in sorted(export_dir.glob(f"*{_PROMPT_CACHE_EXPORT_META_SUFFIX}")):
            item = _read_prompt_cache_export_record(meta_path)
            if isinstance(item, dict):
                items.append(item)
    items.sort(key=lambda item: str(item.get("saved_at") or item.get("name") or ""), reverse=True)
    return {
        "supported": True,
        "ok": True,
        "operation": "list_exports",
        "local_only": True,
        "provider": provider_name,
        "model": model,
        "root_dir": str(root_dir),
        "items": items,
        "capabilities": _prompt_cache_capabilities_dict(provider),
    }


def _prompt_cache_export_local_only_payload(*, operation: str) -> Dict[str, Any]:
    return {
        "supported": False,
        "operation": str(operation or "").strip(),
        "code": "prompt_cache_local_only",
        "error": (
            "Prompt cache export/import admin is local-only. "
            "Remote and hybrid runtimes do not expose a host-local prompt-cache export root."
        ),
        "capabilities": {"supported": False, "mode": "none"},
    }


def _load_abstractcore_bloc_api() -> Dict[str, Any]:
    try:
        import abstractcore as abstractcore_module  # type: ignore
    except Exception:  # pragma: no cover
        abstractcore_module = None  # type: ignore[assignment]
    try:
        from abstractcore.core import bloc_kv as bloc_kv_module  # type: ignore
    except Exception:  # pragma: no cover
        bloc_kv_module = None  # type: ignore[assignment]
    from abstractcore.core.file_blocs import FileBlocStore  # type: ignore

    def _method(name: str) -> Any:
        for module in (abstractcore_module, bloc_kv_module):
            if module is None:
                continue
            value = getattr(module, name, None)
            if callable(value):
                return value
        return None

    return {
        "FileBlocStore": FileBlocStore,
        "ensure_bloc_kv_artifact": _method("ensure_bloc_kv_artifact"),
        "load_bloc_kv_artifact": _method("load_bloc_kv_artifact"),
        "read_bloc_kv_manifest": _method("read_bloc_kv_manifest"),
        "list_bloc_kv_artifacts": _method("list_bloc_kv_artifacts"),
        "find_bloc_kv_live_bindings": _method("find_bloc_kv_live_bindings"),
        "delete_bloc_kv_artifact": _method("delete_bloc_kv_artifact"),
        "prune_bloc_kv_artifacts": _method("prune_bloc_kv_artifacts"),
        "delete_bloc": _method("delete_bloc"),
    }


def _bloc_error_payload(provider: Any, *, operation: str, error: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"ok": False, "operation": str(operation or "").strip()}
    to_dict = getattr(error, "to_dict", None)
    if callable(to_dict):
        try:
            data = to_dict()
        except Exception:
            data = {}
        if isinstance(data, dict):
            payload["code"] = str(data.get("code") or "bloc_error")
            payload["error"] = str(data.get("message") or error)
            if isinstance(data.get("capabilities"), dict):
                payload["capabilities"] = dict(data["capabilities"])
            return payload
    payload["code"] = "bloc_error"
    payload["error"] = str(error)
    if provider is not None:
        payload["capabilities"] = _prompt_cache_capabilities_dict(provider)
    return payload


def _bloc_not_found_payload(*, operation: str, selector: str) -> Dict[str, Any]:
    return {
        "ok": False,
        "operation": str(operation or "").strip(),
        "code": "not_found",
        "error": f"bloc not found for {selector}",
    }


def _bloc_selector_error_payload(*, operation: str) -> Dict[str, Any]:
    return {
        "ok": False,
        "operation": str(operation or "").strip(),
        "code": "invalid_request",
        "error": "provide sha256 or bloc_id",
    }


def _bloc_dependency_missing_payload(*, operation: str, helper: str) -> Dict[str, Any]:
    return {
        "ok": False,
        "operation": str(operation or "").strip(),
        "code": "dependency_missing",
        "error": (
            "Installed AbstractCore does not expose the required durable bloc lifecycle helper "
            f"`{helper}`. Upgrade to a matching AbstractCore build that includes bloc delete/list/prune support."
        ),
    }


def _bloc_in_use_payload(*, operation: str, error: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "ok": False,
        "operation": str(operation or "").strip(),
        "code": "artifact_in_use",
        "error": str(error),
    }
    live = getattr(error, "live_bindings", None)
    if isinstance(live, list):
        out["live_bindings"] = [dict(item) for item in live if isinstance(item, dict)]
    return out


def _resolve_local_bloc_store(*, root_dir: Any) -> Any:
    api = _load_abstractcore_bloc_api()
    store_cls = api["FileBlocStore"]
    return store_cls(root_dir=_coerce_bloc_root_dir(root_dir))


def _refresh_bloc_record(store: Any, record: Any) -> Any:
    if record is None:
        return None
    ensure_ids = getattr(store, "ensure_bloc_ids", None)
    if callable(ensure_ids):
        try:
            ensure_ids()
        except Exception:
            pass
    sha256 = getattr(record, "sha256", None)
    if isinstance(sha256, str) and sha256.strip():
        getter = getattr(store, "get", None)
        if callable(getter):
            try:
                fresh = getter(sha256.strip().lower())
            except Exception:
                fresh = None
            if fresh is not None:
                return fresh
    return record


def _resolve_local_bloc_record(
    *,
    store: Any,
    sha256: Optional[str],
    bloc_id: Optional[int],
) -> tuple[Any, Optional[Dict[str, Any]]]:
    getter = getattr(store, "get", None)
    get_by_bloc_id = getattr(store, "get_by_bloc_id", None)
    sha_value = str(sha256 or "").strip().lower()
    if sha_value:
        if not callable(getter):
            return None, _bloc_error_payload(None, operation="record", error="bloc store does not implement get()")
        try:
            record = getter(sha_value)
        except Exception as exc:
            return None, _bloc_error_payload(None, operation="record", error=exc)
        if record is None:
            return None, _bloc_not_found_payload(operation="record", selector=f"sha256={sha_value}")
        return _refresh_bloc_record(store, record), None
    if bloc_id is not None:
        if not callable(get_by_bloc_id):
            return None, _bloc_error_payload(
                None,
                operation="record",
                error="bloc store does not implement get_by_bloc_id()",
            )
        try:
            record = get_by_bloc_id(int(bloc_id))
        except Exception as exc:
            return None, _bloc_error_payload(None, operation="record", error=exc)
        if record is None:
            return None, _bloc_not_found_payload(operation="record", selector=f"bloc_id={bloc_id}")
        return _refresh_bloc_record(store, record), None
    return None, _bloc_selector_error_payload(operation="record")


def _upsert_text_bloc_local(
    *,
    root_dir: Any,
    path: str,
    content: str,
    sha256: Optional[str] = None,
    content_sha256: Optional[str] = None,
    media_type: str = "text",
    size_bytes: Optional[int] = None,
    mtime_ns: Optional[int] = None,
    format: Optional[str] = None,
    estimated_tokens: Optional[int] = None,
    relpath_base: Optional[str] = None,
    summary: Optional[str] = None,
    keywords: Optional[List[str]] = None,
) -> Dict[str, Any]:
    content_text = str(content or "")
    if not content_text:
        return {
            "ok": False,
            "operation": "upsert_text",
            "code": "invalid_request",
            "error": "content is required",
        }
    path_text = str(path or "").strip()
    if not path_text:
        return {
            "ok": False,
            "operation": "upsert_text",
            "code": "invalid_request",
            "error": "path is required",
        }
    root_path = _coerce_bloc_root_dir(root_dir)
    store = _resolve_local_bloc_store(root_dir=root_path)
    content_sha = (
        str(content_sha256).strip().lower()
        if isinstance(content_sha256, str) and content_sha256.strip()
        else hashlib.sha256(content_text.encode("utf-8")).hexdigest()
    )
    bloc_sha = (
        str(sha256).strip().lower()
        if isinstance(sha256, str) and sha256.strip()
        else hashlib.sha256(content_text.encode("utf-8")).hexdigest()
    )
    relpath_value = str(relpath_base).strip() if isinstance(relpath_base, str) and relpath_base.strip() else None
    relpath_base_path = Path(relpath_value).expanduser() if relpath_value else None
    try:
        record = store.upsert(
            file_meta={
                "path": path_text,
                "media_type": str(media_type or "text"),
                "size_bytes": int(size_bytes) if size_bytes is not None else len(content_text.encode("utf-8")),
                "mtime_ns": int(mtime_ns) if mtime_ns is not None else time.time_ns(),
                "sha256": bloc_sha,
                "content_sha256": content_sha,
                "format": format,
                "content_length": len(content_text),
                "estimated_tokens": int(estimated_tokens) if estimated_tokens is not None else None,
            },
            content=content_text,
            relpath_base=relpath_base_path,
            summary=summary,
            keywords=keywords,
        )
    except Exception as exc:
        return _bloc_error_payload(None, operation="upsert_text", error=exc)
    record = _refresh_bloc_record(store, record)
    return {"ok": True, "operation": "upsert_text", "record": record.to_dict()}


def _get_bloc_record_local(
    *,
    root_dir: Any,
    sha256: Optional[str] = None,
    bloc_id: Optional[int] = None,
) -> Dict[str, Any]:
    store = _resolve_local_bloc_store(root_dir=root_dir)
    record, error = _resolve_local_bloc_record(store=store, sha256=sha256, bloc_id=bloc_id)
    if error is not None:
        error["operation"] = "record"
        return error
    return {"ok": True, "operation": "record", "record": record.to_dict()}


def _list_blocs_local(
    *,
    root_dir: Any,
    sha256: Optional[str] = None,
    bloc_id: Optional[int] = None,
) -> Dict[str, Any]:
    store = _resolve_local_bloc_store(root_dir=root_dir)
    if (isinstance(sha256, str) and sha256.strip()) or bloc_id is not None:
        record, error = _resolve_local_bloc_record(store=store, sha256=sha256, bloc_id=bloc_id)
        if error is not None:
            if error.get("code") == "not_found":
                return {"ok": True, "operation": "list", "records": []}
            error["operation"] = "list"
            return error
        return {"ok": True, "operation": "list", "records": [record.to_dict()]}

    list_fn = getattr(store, "list", None)
    if not callable(list_fn):
        return _bloc_error_payload(None, operation="list", error="bloc store does not implement list()")

    ensure_ids = getattr(store, "ensure_bloc_ids", None)
    if callable(ensure_ids):
        try:
            ensure_ids()
        except Exception:
            pass

    try:
        records = []
        for record in list(list_fn() or []):
            refreshed = _refresh_bloc_record(store, record)
            if refreshed is not None:
                records.append(refreshed.to_dict())
        return {"ok": True, "operation": "list", "records": records}
    except Exception as exc:
        return _bloc_error_payload(None, operation="list", error=exc)


def _get_bloc_kv_manifest_local(
    *,
    provider: Any,
    model: str,
    root_dir: Any,
    sha256: Optional[str] = None,
    bloc_id: Optional[int] = None,
    artifact_path: Optional[str] = None,
) -> Dict[str, Any]:
    api = _load_abstractcore_bloc_api()
    store = _resolve_local_bloc_store(root_dir=root_dir)
    record, error = _resolve_local_bloc_record(store=store, sha256=sha256, bloc_id=bloc_id)
    if error is not None:
        error["operation"] = "kv_manifest"
        return error
    try:
        manifest = api["read_bloc_kv_manifest"](
            provider=provider,
            store=store,
            model=str(model or "").strip(),
            record=record,
            artifact_path=artifact_path,
        )
    except Exception as exc:
        return _bloc_error_payload(provider, operation="kv_manifest", error=exc)
    if manifest is None:
        selector = f"sha256={record.sha256}" if getattr(record, "sha256", None) else f"bloc_id={bloc_id}"
        return _bloc_not_found_payload(operation="kv_manifest", selector=selector)
    return {"ok": True, "operation": "kv_manifest", "manifest": manifest.to_dict()}


def _ensure_bloc_kv_artifact_local(
    *,
    provider: Any,
    model: str,
    root_dir: Any,
    sha256: Optional[str] = None,
    bloc_id: Optional[int] = None,
    artifact_path: Optional[str] = None,
    force_rebuild: bool = False,
    debug: bool = False,
) -> Dict[str, Any]:
    api = _load_abstractcore_bloc_api()
    store = _resolve_local_bloc_store(root_dir=root_dir)
    record, error = _resolve_local_bloc_record(store=store, sha256=sha256, bloc_id=bloc_id)
    if error is not None:
        error["operation"] = "kv_ensure"
        return error
    try:
        result = api["ensure_bloc_kv_artifact"](
            provider=provider,
            store=store,
            model=str(model or "").strip(),
            record=record,
            artifact_path=artifact_path,
            force_rebuild=bool(force_rebuild),
            debug=bool(debug),
        )
    except Exception as exc:
        return _bloc_error_payload(provider, operation="kv_ensure", error=exc)
    artifact: Dict[str, Any] = {
        "artifact_path": str(result.artifact_path),
        "manifest_path": str(result.manifest_path),
        "compiled": bool(result.compiled),
        "rebuilt": bool(result.rebuilt),
        "source_cache_key": result.source_cache_key,
        "binding_id": result.binding_id,
        "prompt_cache_binding": result.prompt_cache_binding,
        "manifest": result.manifest.to_dict(),
    }
    if result.debug is not None:
        artifact["debug"] = result.debug
    return {"ok": True, "operation": "kv_ensure", "artifact": artifact}


def _load_bloc_kv_artifact_local(
    *,
    provider: Any,
    model: str,
    root_dir: Any,
    sha256: Optional[str] = None,
    bloc_id: Optional[int] = None,
    artifact_path: Optional[str] = None,
    stable_cache_key: Optional[str] = None,
    key: Optional[str] = None,
    make_default: bool = False,
    force_rebuild: bool = False,
    debug: bool = False,
) -> Dict[str, Any]:
    api = _load_abstractcore_bloc_api()
    store = _resolve_local_bloc_store(root_dir=root_dir)
    record, error = _resolve_local_bloc_record(store=store, sha256=sha256, bloc_id=bloc_id)
    if error is not None:
        error["operation"] = "kv_load"
        return error
    try:
        result = api["load_bloc_kv_artifact"](
            provider=provider,
            store=store,
            model=str(model or "").strip(),
            record=record,
            artifact_path=artifact_path,
            stable_cache_key=stable_cache_key,
            key=key,
            make_default=bool(make_default),
            force_rebuild=bool(force_rebuild),
            debug=bool(debug),
        )
    except Exception as exc:
        return _bloc_error_payload(provider, operation="kv_load", error=exc)
    artifact: Dict[str, Any] = {
        "artifact_path": str(result.artifact_path),
        "manifest_path": str(result.manifest_path),
        "compiled": bool(result.compiled),
        "loaded": bool(result.loaded),
        "reloaded_stable_key": bool(result.reloaded_stable_key),
        "key": result.key,
        "stable_cache_key": result.stable_cache_key,
        "forked_from": result.forked_from,
        "binding_id": result.binding_id,
        "prompt_cache_binding": result.prompt_cache_binding,
        "manifest": result.manifest.to_dict(),
    }
    if result.debug is not None:
        artifact["debug"] = result.debug
    return {"ok": True, "operation": "kv_load", "artifact": artifact}


def _list_bloc_kv_artifacts_local(
    *,
    root_dir: Any,
    sha256: Optional[str] = None,
    bloc_id: Optional[int] = None,
    provider_name: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    api = _load_abstractcore_bloc_api()
    list_fn = api.get("list_bloc_kv_artifacts")
    if not callable(list_fn):
        return _bloc_dependency_missing_payload(operation="kv_list", helper="list_bloc_kv_artifacts")
    store = _resolve_local_bloc_store(root_dir=root_dir)
    try:
        artifacts = list_fn(
            store=store,
            sha256=sha256,
            bloc_id=bloc_id,
            provider=provider_name,
            model=model,
        )
    except Exception as exc:
        return _bloc_error_payload(None, operation="kv_list", error=exc)
    out = [dict(item) for item in list(artifacts or []) if isinstance(item, dict)]
    return {"ok": True, "operation": "kv_list", "artifacts": out}


def _delete_bloc_kv_artifact_local(
    *,
    provider: Any,
    root_dir: Any,
    sha256: Optional[str] = None,
    bloc_id: Optional[int] = None,
    artifact_path: Optional[str] = None,
    provider_name: Optional[str] = None,
    model: Optional[str] = None,
    clear_loaded: bool = False,
    force: bool = False,
    dry_run: bool = False,
    debug: bool = False,
) -> Dict[str, Any]:
    api = _load_abstractcore_bloc_api()
    delete_fn = api.get("delete_bloc_kv_artifact")
    if not callable(delete_fn):
        return _bloc_dependency_missing_payload(operation="kv_delete", helper="delete_bloc_kv_artifact")
    store = _resolve_local_bloc_store(root_dir=root_dir)
    try:
        result = delete_fn(
            store=store,
            provider=provider,
            sha256=sha256,
            bloc_id=bloc_id,
            provider_name=provider_name,
            model=model,
            artifact_path=artifact_path,
            clear_loaded=bool(clear_loaded),
            force=bool(force),
            dry_run=bool(dry_run),
            debug=bool(debug),
        )
    except Exception as exc:
        if hasattr(exc, "live_bindings"):
            return _bloc_in_use_payload(operation="kv_delete", error=exc)
        return _bloc_error_payload(provider, operation="kv_delete", error=exc)
    return {"ok": True, "operation": "kv_delete", "result": result.to_dict()}


def _prune_bloc_kv_artifacts_local(
    *,
    provider: Any,
    root_dir: Any,
    sha256: Optional[str] = None,
    bloc_id: Optional[int] = None,
    provider_name: Optional[str] = None,
    model: Optional[str] = None,
    clear_loaded: bool = False,
    force: bool = False,
    dry_run: bool = False,
    debug: bool = False,
) -> Dict[str, Any]:
    api = _load_abstractcore_bloc_api()
    prune_fn = api.get("prune_bloc_kv_artifacts")
    if not callable(prune_fn):
        return _bloc_dependency_missing_payload(operation="kv_prune", helper="prune_bloc_kv_artifacts")
    store = _resolve_local_bloc_store(root_dir=root_dir)
    try:
        results = prune_fn(
            store=store,
            provider=provider,
            sha256=sha256,
            bloc_id=bloc_id,
            provider_name=provider_name,
            model=model,
            clear_loaded=bool(clear_loaded),
            force=bool(force),
            dry_run=bool(dry_run),
            debug=bool(debug),
        )
    except Exception as exc:
        if hasattr(exc, "live_bindings"):
            return _bloc_in_use_payload(operation="kv_prune", error=exc)
        return _bloc_error_payload(provider, operation="kv_prune", error=exc)
    return {"ok": True, "operation": "kv_prune", "results": [item.to_dict() for item in list(results or [])]}


def _delete_bloc_local(
    *,
    provider: Any,
    root_dir: Any,
    sha256: Optional[str] = None,
    bloc_id: Optional[int] = None,
    delete_kv: bool = True,
    clear_loaded: bool = False,
    force: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    api = _load_abstractcore_bloc_api()
    delete_fn = api.get("delete_bloc")
    if not callable(delete_fn):
        return _bloc_dependency_missing_payload(operation="delete", helper="delete_bloc")
    store = _resolve_local_bloc_store(root_dir=root_dir)
    try:
        result = delete_fn(
            store=store,
            provider=provider,
            sha256=sha256,
            bloc_id=bloc_id,
            delete_kv=bool(delete_kv),
            clear_loaded=bool(clear_loaded),
            force=bool(force),
            dry_run=bool(dry_run),
        )
    except Exception as exc:
        if hasattr(exc, "live_bindings"):
            return _bloc_in_use_payload(operation="delete", error=exc)
        return _bloc_error_payload(provider, operation="delete", error=exc)
    return {"ok": True, "operation": "delete", "result": result.to_dict()}


def _bloc_kv_entry_provider(entry: Dict[str, Any]) -> Optional[str]:
    if not isinstance(entry, dict):
        return None
    manifest = entry.get("manifest")
    raw = entry.get("provider")
    if not raw and isinstance(manifest, dict):
        raw = manifest.get("provider")
    value = str(raw or "").strip().lower()
    return value or None


def _bloc_kv_entry_model(entry: Dict[str, Any]) -> Optional[str]:
    if not isinstance(entry, dict):
        return None
    manifest = entry.get("manifest")
    raw = entry.get("model")
    if not raw and isinstance(manifest, dict):
        raw = manifest.get("model")
    value = str(raw or "").strip()
    return value or None


def _bloc_kv_entry_artifact_path(entry: Dict[str, Any]) -> Optional[str]:
    if not isinstance(entry, dict):
        return None
    for key in ("artifact_path", "manifest_path"):
        raw = entry.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return None


def _filter_bloc_kv_entries_by_artifact_path(entries: List[Dict[str, Any]], artifact_path: Optional[str]) -> List[Dict[str, Any]]:
    artifact_text = str(artifact_path or "").strip()
    if not artifact_text:
        return list(entries)
    out: List[Dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for key in ("artifact_path", "manifest_path"):
            raw = entry.get(key)
            if isinstance(raw, str) and raw.strip() == artifact_text:
                out.append(entry)
                break
    return out


def _multilocal_loaded_provider(multilocal: Any, provider: Optional[str], model: Optional[str]) -> Any:
    provider_s = str(provider or "").strip().lower()
    model_s = str(model or "").strip()
    if not provider_s or not model_s:
        return None
    client = getattr(multilocal, "_clients", {}).get((provider_s, model_s))
    return getattr(client, "_llm", None) if client is not None else None


def _find_entry_live_bindings_local(*, provider: Any, entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    if provider is None:
        return []
    api = _load_abstractcore_bloc_api()
    finder = api.get("find_bloc_kv_live_bindings")
    if not callable(finder):
        return []
    artifact_path = _bloc_kv_entry_artifact_path(entry)
    if not artifact_path:
        return []
    try:
        live = finder(provider=provider, artifact_path=artifact_path)
    except Exception:
        return []
    return [dict(item) for item in list(live or []) if isinstance(item, dict)]


def _has_prompt_cache_binding(params: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(params, dict):
        return False
    return params.get("prompt_cache_binding") is not None or params.get("expected_prompt_cache_binding") is not None


def _normalize_prompt_cache_binding_params(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = dict(params or {})
    binding = out.pop("expected_prompt_cache_binding", None)
    second = out.get("prompt_cache_binding")
    if binding is None:
        binding = second
    elif second is not None and second != binding:
        raise ValueError("expected_prompt_cache_binding and prompt_cache_binding must match when both are supplied.")
    if binding is None:
        return out
    if isinstance(binding, str):
        # ONE-MEANING-PER-NAME (agent seat c1670; bloc-seam adversary A-3):
        # a bare string is cache-key intent — route to prompt_cache_key and
        # drop the binding param; dict shapes stay strict durable-bloc
        # verification. Mirrors effect_handlers' request normalization.
        key_s = binding.strip()
        out.pop("prompt_cache_binding", None)
        if key_s:
            existing = out.get("prompt_cache_key")
            if existing is not None and str(existing).strip() and str(existing).strip() != key_s:
                raise ValueError("prompt_cache_key and a string prompt_cache_binding must match.")
            out["prompt_cache_key"] = key_s
        return out
    if not isinstance(binding, dict):
        raise ValueError("prompt_cache_binding must be an object (or a string cache key).")
    binding = dict(binding)
    out["prompt_cache_binding"] = binding
    binding_key = binding.get("key")
    if isinstance(binding_key, str) and binding_key.strip():
        binding_key = binding_key.strip()
        existing = out.get("prompt_cache_key")
        if existing is not None and str(existing).strip() and str(existing).strip() != binding_key:
            raise ValueError("prompt_cache_key and prompt_cache_binding.key must match.")
        out["prompt_cache_key"] = binding_key
    return out


def _normalize_residency_task(task: Any) -> str:
    raw = str(task or "").strip().lower().replace("-", "_")
    aliases = {
        "": "text_generation",
        "text": "text_generation",
        "llm": "text_generation",
        "chat": "text_generation",
        "chat_completion": "text_generation",
        "chat_completions": "text_generation",
        "completion": "text_generation",
        "completions": "text_generation",
        "image": "image_generation",
        "images": "image_generation",
        "vision": "image_generation",
        "t2i": "image_generation",
        "text_to_image": "image_generation",
        "i2i": "image_to_image",
        "image_to_image": "image_to_image",
        "image_edit": "image_to_image",
        "edit_image": "image_to_image",
        "upscale": "image_upscale",
        "upscaler": "image_upscale",
        "image_upscale": "image_upscale",
        "image_upscaling": "image_upscale",
        "upscale_image": "image_upscale",
        "video": "video_generation",
        "videos": "video_generation",
        "video_generation": "video_generation",
        "t2v": "text_to_video",
        "text_to_video": "text_to_video",
        "i2v": "image_to_video",
        "image_to_video": "image_to_video",
        "video_from_image": "image_to_video",
        "voice": "tts",
        "speech": "tts",
        "audio_speech": "tts",
        "text_to_speech": "tts",
        "audio": "tts",
        "music": "music_generation",
        "song": "music_generation",
        "t2m": "music_generation",
        "text_to_music": "music_generation",
        "lyrics_to_music": "music_generation",
        "sound": "text_to_audio",
        "sfx": "text_to_audio",
        "sound_generation": "text_to_audio",
        "transcription": "stt",
        "transcriptions": "stt",
        "transcribe": "stt",
        "speech_to_text": "stt",
        "audio_transcription": "stt",
        "audio_transcriptions": "stt",
        "embedding": "embedding",
        "embeddings": "embedding",
        "text_embedding": "embedding",
    }
    return aliases.get(raw, raw)


def _residency_task_filter(task: Any) -> Optional[str]:
    raw = str(task or "").strip()
    if not raw:
        return None
    task_s = _normalize_residency_task(raw)
    if task_s in {"*", "all"}:
        return None
    return task_s


_LOCAL_CAPABILITY_RESIDENCY_LIST_TASKS = (
    "image_generation",
    "image_to_image",
    "image_upscale",
    "video_generation",
    "text_to_video",
    "image_to_video",
    "tts",
    "stt",
    "music_generation",
    "text_to_audio",
    # In-process embedding models (core's EmbeddingManager registry): listed
    # and ejectable like any other local model (mission M2, 2026-09-25).
    "embedding",
)


def _model_residency_unsupported_payload(
    *,
    operation: str,
    task: Any = None,
    provider: Any = None,
    model: Any = None,
    error: str,
) -> Dict[str, Any]:
    task_s = _normalize_residency_task(task)
    payload = {
        "ok": False,
        "success": False,
        "supported": False,
        "operation": str(operation or "").strip(),
        "task": task_s,
        "provider": str(provider or "").strip() or None,
        "model": str(model or "").strip() or None,
        "code": "model_residency_unsupported",
        "error": str(error),
        "warnings": [str(error)],
        "status_hint": "warning",
        "degraded": True,
        "diagnostics": {"source": "abstractruntime"},
        "affected_models": [],
    }
    if task_s in {"image_generation", "image_to_image", "image_upscale", "video_generation", "text_to_video", "image_to_video"}:
        payload["execution_mode"] = "local_one_shot_subprocess"
        payload["local_media_residency_backend"] = "none"
        payload["requires_long_lived_core_backend"] = True
        payload["requires_long_lived_server"] = True
        payload["config_hint"] = (
            "Set ABSTRACTCORE_SERVER_BASE_URL to a long-lived AbstractCore server to enable media warmup."
        )
    elif task_s in {"tts", "stt", "music_generation", "text_to_audio"}:
        payload["local_media_residency_backend"] = "none"
        payload["requires_long_lived_core_backend"] = True
        payload["requires_long_lived_server"] = True
        payload["config_hint"] = (
            "Set ABSTRACTCORE_SERVER_BASE_URL to a long-lived AbstractCore server that exposes this media "
            "residency task, or keep local Runtime media residency unsupported."
        )
    return payload


def _model_residency_capability_task(
    *,
    task: str,
    supported: bool,
    reason: str = "",
    operations: Optional[List[str]] = None,
    truth_source: str = "abstractcore",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    item: Dict[str, Any] = {
        "task": _normalize_residency_task(task),
        "supported": bool(supported),
        "operations": list(operations or ["list_loaded", "load", "unload"]) if supported else [],
        "truth_source": str(truth_source or "abstractcore"),
    }
    if reason:
        item["reason"] = reason
    if isinstance(extra, dict):
        item.update({str(k): _jsonable(v) for k, v in extra.items() if v is not None})
    return item


def _local_model_residency_capabilities(*, mode: str, source: str, text_loads_other_models: bool) -> Dict[str, Any]:
    tasks = {
        "text_generation": _model_residency_capability_task(
            task="text_generation",
            supported=True,
            truth_source="abstractcore.provider.get_model_residency",
            extra={
                "runtime_cache_supported": True,
                "loads_other_models": bool(text_loads_other_models),
                "provider_residency_required_for_loaded_true": True,
            },
        ),
        "image_generation": _model_residency_capability_task(
            task="image_generation",
            supported=True,
            truth_source="abstractcore.capability_plugin",
            extra={
                "local_media_residency_backend": "capability_plugin",
                "requires_installed_capability_plugin": True,
            },
        ),
        "image_to_image": _model_residency_capability_task(
            task="image_to_image",
            supported=True,
            truth_source="abstractcore.capability_plugin",
            extra={
                "local_media_residency_backend": "capability_plugin",
                "requires_installed_capability_plugin": True,
                "shares_backend_cache_with": "image_generation",
            },
        ),
        "image_upscale": _model_residency_capability_task(
            task="image_upscale",
            supported=True,
            truth_source="abstractcore.capability_plugin",
            extra={
                "local_media_residency_backend": "capability_plugin",
                "requires_installed_capability_plugin": True,
                "shares_backend_cache_with": "image_generation",
            },
        ),
        "text_to_video": _model_residency_capability_task(
            task="text_to_video",
            supported=True,
            truth_source="abstractcore.capability_plugin",
            extra={
                "local_media_residency_backend": "capability_plugin",
                "requires_installed_capability_plugin": True,
            },
        ),
        "video_generation": _model_residency_capability_task(
            task="video_generation",
            supported=True,
            truth_source="abstractcore.capability_plugin",
            extra={
                "local_media_residency_backend": "capability_plugin",
                "requires_installed_capability_plugin": True,
                "includes_tasks": ["text_to_video", "image_to_video"],
            },
        ),
        "image_to_video": _model_residency_capability_task(
            task="image_to_video",
            supported=True,
            truth_source="abstractcore.capability_plugin",
            extra={
                "local_media_residency_backend": "capability_plugin",
                "requires_installed_capability_plugin": True,
            },
        ),
        "tts": _model_residency_capability_task(
            task="tts",
            supported=True,
            truth_source="abstractcore.capability_plugin",
            extra={"local_media_residency_backend": "capability_plugin", "requires_installed_capability_plugin": True},
        ),
        "stt": _model_residency_capability_task(
            task="stt",
            supported=True,
            truth_source="abstractcore.capability_plugin",
            extra={"local_media_residency_backend": "capability_plugin", "requires_installed_capability_plugin": True},
        ),
        "music_generation": _model_residency_capability_task(
            task="music_generation",
            supported=True,
            truth_source="abstractcore.capability_plugin",
            extra={"local_media_residency_backend": "capability_plugin", "requires_installed_capability_plugin": True},
        ),
    }
    return {
        "ok": True,
        "supported": True,
        "operation": "capabilities",
        "mode": mode,
        "source": source,
        "tasks": tasks,
        "supported_tasks": [task for task, info in tasks.items() if info.get("supported") is True],
        "unsupported_tasks": [task for task, info in tasks.items() if info.get("supported") is not True],
        "diagnostics": {"source": source},
    }


def _residency_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in {"1", "true", "yes", "on"}:
            return True
        if raw in {"0", "false", "no", "off"}:
            return False
    return None


def _unknown_provider_residency_claim(*, provider: str, model: str, warning: str) -> Dict[str, Any]:
    _ = provider, model
    return {
        "provider_residency_verified": False,
        "provider_resident": None,
        "provider_residency_source": "abstractcore.provider",
        "provider_state": "provider_residency_unknown",
        "warnings": [warning],
    }


def _normalize_residency_size_extras(record: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize ollama-style `size`/`size_vram` extras into `size_bytes`/
    `size_vram_bytes` (originals kept) so downstream consumers see one name."""
    for raw_name, normalized_name in (("size", "size_bytes"), ("size_vram", "size_vram_bytes")):
        if record.get(normalized_name) is not None:
            continue
        raw_value = record.get(raw_name)
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            continue
        record[normalized_name] = int(raw_value)
    return record


def _sweep_host_loaded_models() -> List[Dict[str, Any]]:
    """Core-owned host sweep of local model servers (ADR 0007: relayed truth,
    never synthesized here). Best-effort: unavailable core or a failing sweep
    yields an empty list."""
    try:
        from abstractcore.utils.residency import sweep_loaded_models  # type: ignore
    except Exception:
        return []
    try:
        return [dict(record) for record in sweep_loaded_models() if isinstance(record, dict)]
    except Exception:
        return []


def _sweep_provider_names() -> Tuple[str, ...]:
    """Core's `SWEEP_PROVIDERS` tuple, or empty when core is unavailable."""
    try:
        from abstractcore.utils.residency import SWEEP_PROVIDERS  # type: ignore

        return tuple(str(name).strip().lower() for name in SWEEP_PROVIDERS)
    except Exception:
        return ()


def _sweep_verifies_model_resident(provider: str, model: str) -> bool:
    """Does the host sweep verify (provider, model) resident on its local model
    server? Sweep providers only (the sweep can never answer for anything
    else); best-effort, never raises. The local-lane twin of core's
    `_sweep_verifies_model_resident`, using the SAME canonical alias rules."""
    provider_s = str(provider or "").strip().lower()
    model_s = str(model or "").strip()
    if not provider_s or not model_s or provider_s not in _sweep_provider_names():
        return False
    try:
        from abstractcore.utils.residency import sweep_models_match  # type: ignore
    except Exception:
        return False
    try:
        for record in _sweep_host_loaded_models():
            if str(record.get("provider") or "").strip().lower() != provider_s:
                continue
            if sweep_models_match(provider_s, model_s, record):
                return True
    except Exception:
        return False
    return False


def _merge_host_sweep_into_text_records(
    records: List[Dict[str, Any]],
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Merge the core host sweep into a local text-residency listing, in
    place, using the CANONICAL `abstractcore.utils.residency` helpers (the
    same alias/dedup rules core's `/acore/models/loaded` uses): pool/client
    records win and absorb missing memory fields; sweep-only entries are
    appended tagged `source: "provider_server"` relaying ONLY what the sweep
    reported (no task label is invented — server enumerations cannot
    classify residents)."""
    try:
        from abstractcore.utils.residency import (  # type: ignore
            SWEEP_PROVIDERS,
            normalize_sweep_model,
            sweep_models_match,
        )
    except Exception:
        # Without the core helpers there is no core sweep either.
        return records
    provider_filter = str(provider or "").strip().lower()
    model_filter = str(model or "").strip()
    if provider_filter and provider_filter not in SWEEP_PROVIDERS:
        # The sweep can never answer for this provider: skip the live probes.
        return records
    sweep_records = _sweep_host_loaded_models()
    if not sweep_records:
        return records
    remaining: List[Dict[str, Any]] = list(sweep_records)
    for existing in records:
        existing_provider = str(existing.get("provider") or "").strip().lower()
        match: Optional[Dict[str, Any]] = None
        for candidate in remaining:
            if str(candidate.get("provider") or "").strip().lower() != existing_provider:
                continue
            if sweep_models_match(existing_provider, existing.get("model"), candidate):
                match = candidate
                break
        if match is None:
            continue
        remaining.remove(match)
        # Memory truth the sweep knows and the pool row does not. `cache_bytes`
        # and `est_weights_bytes` ride along with the size pair: they are
        # provider/core-owned figures, so an absent value on the pool row is
        # filled, never overwritten.
        for field_name in ("size_bytes", "size_vram_bytes", "est_weights_bytes", "cache_bytes"):
            if existing.get(field_name) is None and match.get(field_name) is not None:
                existing[field_name] = match[field_name]
    for record in remaining:
        record_provider = str(record.get("provider") or "").strip().lower()
        if provider_filter and record_provider != provider_filter:
            continue
        if model_filter and normalize_sweep_model(model_filter) != normalize_sweep_model(record.get("model")):
            continue
        record.setdefault("loaded", True)
        record.setdefault("resident", True)
        record["source"] = "provider_server"
        # Sweep-only rows on a SWEEP provider ARE lockable: locking such a pair
        # ADOPTS it into this client's pool (client construction only — never a
        # provider-side load) and then enforces the lock like any managed pair
        # (unload refuses without force). Same rule core's server-side sweep
        # merge applies to its own sweep rows. A row from any other source has
        # nothing here to enforce a lock with, so it stays not lockable.
        if record_provider in SWEEP_PROVIDERS:
            record["lockable"] = True
        else:
            record.setdefault("lockable", False)
        # Sweep rows are observed on THIS host too — same core-owned stamps
        # core's server-side sweep merge applies to its own sweep rows.
        _stamp_local_record_modalities(record, model=record.get("model"))
        _stamp_local_record_host_identity(record)
        records.append(_normalize_residency_size_extras(record))
    return records


def _mlx_process_residency_rows() -> List[Dict[str, Any]]:
    """Core's PROCESS-level MLX residency (`abstractcore.providers.mlx_residency`):
    every MLX model whose weights are alive in this process, whoever holds them
    (a pool client, an override client, a chat summarizer built at boot, an old
    runtime after a bundle reload, another principal's service). Best-effort: an
    older core or a failing probe yields []."""
    try:
        from abstractcore.providers.mlx_residency import resident_models  # type: ignore
    except Exception:
        return []
    try:
        return [dict(r) for r in resident_models() if isinstance(r, dict) and r.get("weights_alive")]
    except Exception:
        return []


def _mlx_process_eject(model: str) -> Optional[Dict[str, Any]]:
    """Unload EVERY holder of `model` in this process (core `eject_model`), then
    collect + clear MLX's cache. None when core has no process-level eject."""
    try:
        from abstractcore.providers.mlx_residency import eject_model  # type: ignore
    except Exception:
        return None
    try:
        return eject_model(str(model or "").strip() or None, reason="model_residency_unload")
    except Exception as exc:  # noqa: BLE001 - the eject report must reach the caller
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "model": model}


def _hf_process_residency_rows() -> List[Dict[str, Any]]:
    """Core's PROCESS-level HuggingFace residency (`abstractcore.providers.
    hf_residency`): every transformers / GGUF model whose weights are alive in
    this process, whoever holds them. Unlike MLX, HuggingFace instances do not
    share weights: each holder is a FULL COPY. Best-effort: an older core or a
    failing probe yields []."""
    try:
        from abstractcore.providers.hf_residency import resident_models  # type: ignore
    except Exception:
        return []
    try:
        return [dict(r) for r in resident_models() if isinstance(r, dict) and r.get("weights_alive")]
    except Exception:
        return []


def _hf_process_eject(model: str) -> Optional[Dict[str, Any]]:
    """Unload EVERY HuggingFace holder of `model` in this process (core
    `hf_residency.eject_model`), then collect + return torch's MPS pool. None
    when core has no process-level eject for this backend."""
    try:
        from abstractcore.providers.hf_residency import eject_model  # type: ignore
    except Exception:
        return None
    try:
        return eject_model(str(model or "").strip() or None, reason="model_residency_unload")
    except Exception as exc:  # noqa: BLE001 - the eject report must reach the caller
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "model": model}


_EMBEDDINGS_MANAGER_MODULE = "abstractcore.embeddings.manager"


def _embedding_process_rows() -> List[Dict[str, Any]]:
    """Core's process-level truth for in-process embedding models
    (`EmbeddingManager` registry): one row per model whose weights are alive,
    whoever built the embedder (gateway memory, a flow, a tool). A process that
    never imported the manager holds none, and this never imports it (it pulls
    sentence-transformers and torch). An imported core WITHOUT the registry is
    an incompatible core: that raises, it is not read as "nothing loaded"."""
    mod = sys.modules.get(_EMBEDDINGS_MANAGER_MODULE)
    if mod is None:
        return []
    fn = getattr(mod, "resident_embedding_models", None)
    if not callable(fn):
        raise RuntimeError(
            "this AbstractCore has no embedding residency registry (resident_embedding_models); "
            "upgrade abstractcore to list or eject embedding models"
        )
    return [dict(r) for r in fn() if isinstance(r, dict) and r.get("weights_alive", True)]


def _embedding_process_eject(model: Optional[str]) -> Dict[str, Any]:
    """Unload EVERY in-process embedder holding `model`, collect, return torch's
    MPS pool (core `eject_embedding_models`). The embedders stay usable: the
    next embedding reloads the model."""
    mod = sys.modules.get(_EMBEDDINGS_MANAGER_MODULE)
    if mod is None:
        return {"ok": True, "model": model, "holders_found": 0, "holders_unloaded": [], "holders_refused": [],
                "residual": None, "note": "no embedding model was ever loaded in this process"}
    fn = getattr(mod, "eject_embedding_models", None)
    if not callable(fn):
        raise RuntimeError(
            "this AbstractCore has no embedding eject (eject_embedding_models); upgrade abstractcore"
        )
    return dict(fn(str(model or "").strip() or None, reason="model_residency_unload"))


def _embedding_residency_record(row: Dict[str, Any], name: str, *, source: str) -> Dict[str, Any]:
    runtime_id = f"local:embedding:huggingface:{name}"
    holders = int(row.get("holders") or 0)
    return {
        "task": "embedding",
        "provider": "huggingface",
        "model": name,
        "backend": "embeddings",
        "runtime_id": runtime_id,
        "load_id": runtime_id,
        "loaded": True,
        "resident": True,
        "state": "provider_loaded",
        "provider_state": "resident",
        "provider_residency_verified": True,
        "provider_resident": True,
        "provider_residency_source": "abstractcore.embeddings.process",
        "source": source,
        "isolation": "in_process",
        "runtime_cached": False,
        "lockable": False,
        "locked": False,
        "device": row.get("device"),
        "model_path": row.get("model_path"),
        "process_holders": holders,
        "weights_bytes": row.get("weights_bytes"),
        "est_weights_bytes": row.get("weights_bytes"),
        "held_bytes": row.get("held_bytes"),
        "shared_weights": False,
    }


def _local_embedding_residency_result(
    *,
    operation: str,
    provider: Optional[str],
    model: Optional[str],
    source: str,
) -> Dict[str, Any]:
    provider_s = str(provider or "").strip().lower()
    model_s = str(model or "").strip()
    if provider_s and provider_s != "huggingface" and operation != "list_loaded":
        # Ollama / LM Studio / OpenAI-compatible embedders hold nothing here.
        return _model_residency_unsupported_payload(
            operation=operation, task="embedding", provider=provider_s, model=model_s,
            error=f"{provider_s} serves its embedding models from its own server; nothing is held in this process",
        )
    if operation == "list_loaded":
        records = [
            _embedding_residency_record(row, name, source=source)
            for row in _embedding_process_rows()
            for name in [str(n) for n in (row.get("models") or []) if str(n).strip()]
            if (not model_s or name == model_s) and provider_s in ("", "huggingface")
        ]
        return _with_local_model_residency_summary(
            {"ok": True, "supported": True, "operation": "list_loaded", "task": "embedding", "models": records,
             "diagnostics": {"source": source, "capability": "embeddings", "count": len(records)}},
            operation="list_loaded",
            models=records,
        )
    if operation == "unload":
        if not model_s:
            return {"ok": False, "success": False, "supported": True, "operation": "unload", "task": "embedding",
                    "unloaded": False, "error": "embedding unload requires a model (or its local:embedding: runtime_id)",
                    "warnings": ["embedding unload requires a model"], "affected_models": []}
        held = [r for r in _embedding_process_rows() if model_s in [str(n) for n in (r.get("models") or [])]]
        report = _embedding_process_eject(model_s)
        ok = bool(report.get("ok"))
        runtime = {
            "task": "embedding", "provider": "huggingface", "model": model_s, "backend": "embeddings",
            "runtime_id": f"local:embedding:huggingface:{model_s}", "loaded": not ok, "resident": not ok,
            "state": "unloaded" if ok else "provider_loaded", "isolation": "in_process", "source": source,
        }
        error: Optional[str] = None
        if not ok:
            refused = report.get("holders_refused") or []
            residual = report.get("residual") or {}
            error = (f"{len(refused)} embedder(s) refused to unload {model_s}: {refused[0].get('error')}" if refused
                     else f"{model_s} is still resident in this process after the eject "
                          f"({int(residual.get('holders') or 0)} holder(s), {int(residual.get('held_bytes') or 0)} bytes held)")
        result = {
            "ok": ok,
            "supported": True,
            "operation": "unload",
            "task": "embedding",
            "provider": "huggingface",
            "model": model_s,
            "unloaded": bool(ok and held),
            "runtime": runtime,
            "process_eject": _jsonable({k: v for k, v in report.items() if k not in ("residual",)}),
            "diagnostics": {"source": source, "capability": "embeddings", "was_resident": bool(held)},
        }
        if error:
            result["error"] = error
            result["warnings"] = [error]
        return _with_local_model_residency_summary(
            result, operation="unload", runtime=runtime,
            action="unloaded" if (ok and held) else "not_unloaded", changed=bool(ok and held),
        )
    return _model_residency_unsupported_payload(
        operation=operation, task="embedding", provider=provider_s, model=model_s,
        error="embedding models load on first use (an embedding request); there is no explicit preload",
    )


def _parse_local_residency_runtime_id(runtime_id: Any) -> Optional[Tuple[str, str, str]]:
    """`local:<task>:<provider>:<model>` -> (task, provider, model), the form
    every local residency row carries when its backend gives no id of its own.
    The model may contain ':' (`gemma3:1b`); an `endpoint:<name>` provider
    keeps its colon. None for any other id."""
    raw = str(runtime_id or "").strip()
    if not raw.startswith("local:"):
        return None
    parts = raw.split(":", 3)
    if len(parts) < 4:
        return None
    _, task, provider, rest = parts
    provider = provider.strip().lower()
    if provider == "endpoint" and ":" in rest:
        name, rest = rest.split(":", 1)
        provider = f"endpoint:{name.strip()}"
    task_s = _normalize_residency_task(task)
    if not task_s or not provider or not rest.strip():
        return None
    return task_s, provider, rest.strip()


def _core_process_residency() -> Any:
    """Core's process residency module with the claimant registry, or None on
    an AbstractCore without it (then no switch/failed-load eject runs: a
    process-wide eject without knowing every owner's claims is unsafe)."""
    try:
        import abstractcore.providers.process_residency as pr  # type: ignore
    except Exception:
        return None
    return pr if callable(getattr(pr, "eject_unclaimed", None)) else None


class _NullLock:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _core_residency_lock() -> Any:
    pr = _core_process_residency()
    return pr.residency_lock() if pr is not None else _NullLock()


def _eject_report_summary(label: str, report: Dict[str, Any]) -> Dict[str, Any]:
    provider_s, _, model_s = label.partition("/")
    out: Dict[str, Any] = {"provider": provider_s, "model": model_s}
    for key in ("ok", "skipped", "deferred", "reason", "error", "holders_found", "freed_bytes", "ts"):
        if report.get(key) is not None:
            out[key] = report[key]
    residual = report.get("residual")
    if isinstance(residual, dict):
        out["residual"] = {k: residual.get(k) for k in ("holders", "held_bytes")}
    if report.get("ok") is False and not out.get("error"):
        out["error"] = f"still resident after the eject: {out.get('residual')}"
    return out


_IN_PROCESS_TIMED_OPTIONS = ("ttl_s", "keep_alive")


def _load_option_warnings(
    *,
    provider: str,
    model: str,
    requested: Dict[str, Any],
    provider_load_result: Any,
    provider_called: bool,
) -> Tuple[List[str], List[str]]:
    """(warnings, unsupported_options) for a load response. Provider warnings
    are copied up; `ttl_s` / `keep_alive` are reported as NOT applied when the
    provider cannot apply them (in-process providers have no idle unload) or
    when this call never reached the provider (the model was already loaded)."""
    warnings: List[str] = []
    unsupported: List[str] = []
    if isinstance(provider_load_result, dict):
        warnings.extend(str(w) for w in (provider_load_result.get("warnings") or []) if str(w).strip())
        unsupported.extend(str(o) for o in (provider_load_result.get("unsupported_options") or []))
    timed = [k for k in _IN_PROCESS_TIMED_OPTIONS if requested.get(k) is not None and k not in unsupported]
    if timed:
        names = ", ".join(timed)
        if str(provider).strip().lower() in _PROCESS_RESIDENCY_PROVIDERS:
            warnings.append(
                f"{provider} has no idle/TTL unload: {names} not applied; {model} stays loaded until it is unloaded"
            )
        elif not provider_called:
            warnings.append(f"{model} was already loaded, so this load did not apply {names}")
        else:
            timed = []
        unsupported.extend(timed)
    return warnings, unsupported


def _load_option_report(**kw: Any) -> Dict[str, Any]:
    warnings, unsupported = _load_option_warnings(**kw)
    out: Dict[str, Any] = {}
    if warnings:
        out["warnings"] = warnings
    if unsupported:
        out["unsupported_options"] = unsupported
    return out


def _explicit_eject_guard(
    owner: Any,
    *,
    task: Optional[str],
    runtime_id: Optional[str],
    provider: Optional[str],
    model: Optional[str],
    force: bool,
    source: str,
) -> Dict[str, Any]:
    """Claims of OTHER owners on the model an explicit eject targets.

    Returns {"in_process": bool, "refusal": payload|None, "pooled_elsewhere": n,
    "forced_over_locks": [...]} . A refusal is the structured 409-style payload
    `refused: "model_locked_by_other_client"` naming the holders."""
    out: Dict[str, Any] = {"in_process": False, "refusal": None, "pooled_elsewhere": 0, "forced_over_locks": []}
    task_s, provider_s, model_s, _ = _resolve_unload_selector(
        owner, task=task, runtime_id=runtime_id, provider=provider, model=model,
    )
    if task_s != "text_generation" or str(provider_s) not in _PROCESS_RESIDENCY_PROVIDERS or not model_s:
        return out
    out["in_process"] = True
    pr = _core_process_residency()
    if pr is None:
        return out
    mine = f"runtime client {id(owner):x}"
    others = [c for c in pr.claims_for(provider_s, model_s) if c.get("owner") != mine]
    locks = [c for c in others if c.get("locked")]
    holders = [{k: c.get(k) for k in ("kind", "owner", "runtime_id", "model") if c.get(k) is not None} for c in locks]
    if locks and not force:
        message = (
            f"{provider_s}/{model_s} is locked by another client in this process "
            f"({', '.join(sorted({str(h.get('kind')) + ': ' + str(h.get('owner')) for h in holders}))}); "
            "unload it there, or pass force=true to eject it anyway"
        )
        out["refusal"] = {
            "ok": False,
            "success": False,
            "supported": True,
            "operation": "unload",
            "task": "text_generation",
            "provider": provider_s,
            "model": model_s,
            "unloaded": False,
            "refused": "model_locked_by_other_client",
            "status_code": 409,
            "locked_by": holders,
            "error": message,
            "warnings": [message],
            "affected_models": [],
            "diagnostics": {"source": source, "reason": "model_locked_by_other_client"},
        }
        return out
    out["forced_over_locks"] = holders if locks else []
    owners = {str(c.get("owner")) for c in others if not c.get("locked")}
    out["pooled_elsewhere"] = len(owners)
    return out


def _provider_inflight_count(instance: Any) -> int:
    """Generations currently running on a provider instance (core's
    `InflightGenerations`); 0 when the provider does not track them."""
    try:
        registry = instance._inflight_generations()
        return int(registry.active())
    except Exception:
        return 0


def _resolve_unload_selector(
    holder: Any,
    *,
    task: Optional[str],
    runtime_id: Optional[str],
    provider: Optional[str],
    model: Optional[str],
) -> Tuple[str, str, str, Optional[str]]:
    """(task, provider, model, runtime_id_for_capability) for an unload.

    The console, tray and CLI eject a row by sending only its `runtime_id`.
    Every task is addressed: `local:<task>:<provider>:<model>` is parsed
    generically (a TTS / STT / image / embedding row, not only text), and a
    backend's own id (abstractvision's `diffusers/<model>`...) is looked up in
    the current listing. A synthesized `local:` id is not passed on to the
    capability plugin (it never issued it); a backend's own id is."""
    explicit_task = str(task or "").strip()
    task_s = _normalize_residency_task(task)
    provider_s = str(provider or "").strip().lower()
    model_s = str(model or "").strip()
    rid = str(runtime_id or "").strip() or None
    parsed = _parse_local_residency_runtime_id(rid)
    if parsed is not None:
        p_task, p_provider, p_model = parsed
        if not explicit_task:
            task_s = p_task
        if task_s == p_task and (not provider_s or not model_s):
            provider_s, model_s = provider_s or p_provider, model_s or p_model
        return task_s, provider_s, model_s, None
    if rid and not explicit_task and (not provider_s or not model_s):
        lister = getattr(holder, "list_model_residency", None)
        listing = lister() if callable(lister) else {}
        for row in list((listing or {}).get("models") or []):
            if not isinstance(row, dict):
                continue
            if rid in (str(row.get("runtime_id") or ""), str(row.get("load_id") or "")):
                task_s = _normalize_residency_task(row.get("task"))
                provider_s = provider_s or str(row.get("provider") or "").strip().lower()
                model_s = model_s or str(row.get("model") or "").strip()
                break
    return task_s, provider_s, model_s, rid


# Providers whose weights live IN THIS PROCESS and can therefore be held by
# instances no runtime pool reaches. Both have a core process-level truth.
_PROCESS_RESIDENCY_PROVIDERS = ("mlx", "huggingface")


def _process_residency_rows_for(provider: str) -> List[Dict[str, Any]]:
    provider_s = str(provider or "").strip().lower()
    if provider_s == "mlx":
        return _mlx_process_residency_rows()
    if provider_s == "huggingface":
        return _hf_process_residency_rows()
    return []


def _process_eject_for(provider: str, model: str) -> Optional[Dict[str, Any]]:
    provider_s = str(provider or "").strip().lower()
    if provider_s == "mlx":
        return _mlx_process_eject(model)
    if provider_s == "huggingface":
        return _hf_process_eject(model)
    return None


def _mlx_process_row_extras(row: Dict[str, Any]) -> Dict[str, Any]:
    extras: Dict[str, Any] = {
        "process_holders": int(row.get("holders") or 0),
        "held_bytes": int(row.get("held_bytes") or 0),
        "cache_bytes": int(row.get("cache_bytes") or 0),
        "process_lane": row.get("lane"),
    }
    if isinstance(row.get("weights_bytes"), int) and not isinstance(row.get("weights_bytes"), bool):
        extras["est_weights_bytes"] = int(row["weights_bytes"])
    return extras


def _merge_process_residency_into_text_records(
    records: List[Dict[str, Any]],
    *,
    lane_provider: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Fold the process-level residency of one in-process backend (`mlx` or
    `huggingface`) into a text-residency listing, in place. A pool row whose
    own instance says "not loaded" while the weights are still alive in the
    process becomes `resident: True` with `provider_state:
    "resident_via_other_holders"`; a model no pool row names (held only by
    unreachable holders) is APPENDED, tagged
    `source: "abstractcore.provider.<backend>.process"`. Either way the listing
    can no longer say "nothing loaded" over gigabytes of live buffers."""
    lane = str(lane_provider or "").strip().lower()
    provider_filter = str(provider or "").strip().lower()
    if provider_filter and provider_filter != lane:
        return records
    model_filter = str(model or "").strip()
    rows = _process_residency_rows_for(lane)
    if not rows:
        return records
    copies_note = " (each a full copy of the weights)" if lane == "huggingface" else ""
    for row in rows:
        names = [str(n) for n in (row.get("models") or []) if str(n).strip()]
        if not names:
            names = [str(row.get("model_path") or "").strip()]
        extras = _mlx_process_row_extras(row)
        for name in names:
            if not name or (model_filter and name != model_filter):
                continue
            existing = next(
                (
                    r
                    for r in records
                    if isinstance(r, dict)
                    and str(r.get("provider") or "").strip().lower() == lane
                    and str(r.get("model") or "").strip() == name
                ),
                None,
            )
            if existing is None:
                record = _local_residency_record(
                    provider=lane,
                    model=name,
                    default=False,
                    runtime_cached=False,
                    include_provider_state=False,
                )
                record.update(
                    {
                        "resident": True,
                        "loaded": True,
                        "state": "provider_loaded",
                        "provider_residency_verified": True,
                        "provider_resident": True,
                        "provider_residency_source": f"abstractcore.provider.{lane}.process",
                        "provider_state": "resident_via_other_holders",
                        "source": f"abstractcore.provider.{lane}.process",
                        "lockable": False,
                        "warnings": [
                            f"held in memory by {extras['process_holders']} provider instance(s) outside this "
                            f"runtime's pool{copies_note}; unloading it ejects every holder in the process"
                        ],
                        **extras,
                    }
                )
                records.append(_normalize_residency_size_extras(record))
                continue
            if existing.get("resident") is not True:
                existing.update(
                    {
                        "resident": True,
                        "loaded": True,
                        "state": "provider_loaded",
                        "provider_residency_verified": True,
                        "provider_resident": True,
                        "provider_state": "resident_via_other_holders",
                    }
                )
                existing.setdefault("warnings", [])
                if isinstance(existing["warnings"], list):
                    existing["warnings"].append(
                        f"this runtime's own instance released the weights, but {extras['process_holders']} other "
                        f"provider instance(s) in the process still hold them{copies_note}; unloading ejects every holder"
                    )
            for key, value in extras.items():
                if existing.get(key) is None:
                    existing[key] = value
    return records


def _merge_mlx_process_residency_into_text_records(
    records: List[Dict[str, Any]],
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    return _merge_process_residency_into_text_records(records, lane_provider="mlx", provider=provider, model=model)


def _merge_hf_process_residency_into_text_records(
    records: List[Dict[str, Any]],
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    return _merge_process_residency_into_text_records(records, lane_provider="huggingface", provider=provider, model=model)


def _local_memory_snapshot() -> Dict[str, Any]:
    try:
        from abstractcore.utils.memory import get_memory_snapshot  # type: ignore
    except Exception as exc:  # pragma: no cover - the core sibling ships it
        return {"ok": False, "error": f"AbstractCore memory snapshot is unavailable: {exc}"}
    try:
        snapshot = get_memory_snapshot()
    except Exception as exc:  # noqa: BLE001 - core never raises; belt and braces
        return {"ok": False, "error": str(exc)}
    if not isinstance(snapshot, dict):
        return {"ok": False, "error": "invalid memory snapshot"}
    return dict(snapshot)


def _provider_prompt_cache_stats_raw(provider: Any) -> Optional[Dict[str, Any]]:
    """A provider's raw `get_prompt_cache_stats()` payload, or None when the
    provider does not support stats (or the query fails)."""
    method = getattr(provider, "get_prompt_cache_stats", None)
    if not callable(method) or not _prompt_cache_supports(provider, "stats"):
        return None
    try:
        stats = method()
    except Exception:
        return None
    return stats if isinstance(stats, dict) else None


def _local_record_prompt_cache_bytes(provider_instance: Any) -> Optional[int]:
    """Total prompt-cache store bytes held by a pooled provider instance, or
    None when unknown. Core-owned arithmetic (`prompt_cache_store_bytes` over
    the provider's OWN `get_prompt_cache_stats()` payload) — nothing is
    computed here; this is the local-lane twin of core's
    `_gateway_runtime_prompt_cache_bytes`. Best-effort: never raises."""
    if provider_instance is None:
        return None
    stats = _provider_prompt_cache_stats_raw(provider_instance)
    if stats is None:
        return None
    try:
        from abstractcore.utils.memory import prompt_cache_store_bytes  # type: ignore
    except Exception:
        return None
    try:
        total = prompt_cache_store_bytes(stats)
    except Exception:
        return None
    return total if isinstance(total, int) and not isinstance(total, bool) else None


def _session_prompt_cache_rows_from_stats(
    stats: Any,
    *,
    provider: Optional[str],
    model: Optional[str],
    runtime_id: Optional[str],
    session_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Rows for `list_session_prompt_caches` from one provider's raw
    `get_prompt_cache_stats()` payload. Unknown fields stay None; the
    session_id comes from stamped key meta and filters when given."""
    if not isinstance(stats, dict):
        return []
    keys = stats.get("keys")
    if not isinstance(keys, list):
        return []
    meta_by_key = stats.get("meta_by_key") if isinstance(stats.get("meta_by_key"), dict) else {}
    session_filter = str(session_id or "").strip()
    rows: List[Dict[str, Any]] = []
    for key in keys:
        key_s = str(key or "").strip()
        if not key_s:
            continue
        meta = meta_by_key.get(key_s)
        meta = dict(meta) if isinstance(meta, dict) else {}
        row_session_id = meta.get("session_id")
        row_session_id = (
            row_session_id.strip() if isinstance(row_session_id, str) and row_session_id.strip() else None
        )
        if session_filter and row_session_id != session_filter:
            continue
        token_count = meta.get("token_count")
        cache_bytes = meta.get("bytes")
        rows.append(
            {
                "key": key_s,
                "provider": provider,
                "model": model,
                "runtime_id": runtime_id,
                "session_id": row_session_id,
                "token_count": token_count if isinstance(token_count, int) and not isinstance(token_count, bool) else None,
                "bytes": cache_bytes if isinstance(cache_bytes, int) and not isinstance(cache_bytes, bool) else None,
                "created_at_s": meta.get("created_at_s"),
                "last_used_at_s": meta.get("last_used_at_s"),
                "meta": meta,
            }
        )
    return rows


def _clear_provider_prompt_cache_key(provider: Any, *, key: Any) -> Dict[str, Any]:
    """Per-row clear outcome fields for `clear_session_prompt_caches`."""
    key_s = str(key or "").strip()
    method = getattr(provider, "prompt_cache_clear", None)
    if not key_s or not callable(method):
        return {"cleared": False, "error": "provider does not support prompt_cache_clear"}
    try:
        ok = bool(method(key_s))
    except Exception as exc:  # noqa: BLE001 - partial failures are reported per row
        return {"cleared": False, "error": str(exc)}
    if ok:
        return {"cleared": True}
    return {"cleared": False, "error": "prompt_cache_clear returned False"}


def _stamp_local_record_modalities(
    record: Dict[str, Any],
    *,
    model: Any,
    provider_instance: Any = None,
) -> None:
    """Stamp core's registry-declared modalities onto a LOCALLY-SERVED text
    residency record, in place (ADR 0007: core-owned truth relayed — core's
    server lane stamps its own rows the same way; without this the gateway
    LOCAL lane never carries `modalities`, since provider claims don't).

    Registry miss → field omitted (never the text-only guess). Runtime truth
    beats declared truth: a provider instance that knows its vision lane is
    unusable (`_vision_usable is False`, MLX) has `input.image` removed and
    the divergence noted. Outright assignment on a registry hit (post-claim),
    mirroring core: a hostile claim cannot override registry truth.
    Best-effort: never raises.

    IMPORT ORDER (E2E-verified env quirk): a FRESH import of
    `abstractcore.providers.model_capabilities` can trip the pre-existing
    architectures↔media circular import unless `abstractcore.utils` is
    imported first — hence the guarded two-step lazy import."""
    try:
        import abstractcore.utils  # noqa: F401 - import-order guard (see docstring)
        from abstractcore.providers.model_capabilities import modalities_for_model  # type: ignore
    except Exception:
        return
    try:
        modalities = modalities_for_model(str(model or ""))
        if modalities is None:
            return
        modalities = list(modalities)
        if (
            "input.image" in modalities
            and provider_instance is not None
            and getattr(provider_instance, "_vision_usable", None) is False
        ):
            modalities = [m for m in modalities if m != "input.image"]
            record["modalities_note"] = "vision_unusable"
        record["modalities"] = modalities
    except Exception:
        pass


def _stamp_local_record_host_identity(record: Dict[str, Any]) -> None:
    """Stamp THIS host's identity onto a LOCALLY-SERVED residency record, in
    place (core-owned truth via `abstractcore.utils.hostinfo`; the remote lane
    is deliberately NOT stamped — its rows already carry the SERVER's identity
    and overwriting with the client's would be a lie). `setdefault`, mirroring
    core: an already-attributed record is never overwritten. Best-effort:
    never raises."""
    try:
        from abstractcore.utils.hostinfo import get_host_identity  # type: ignore
    except Exception:
        return
    try:
        identity = get_host_identity()
        record.setdefault("host_id", identity.get("host_id"))
        record.setdefault("host_name", identity.get("host_name"))
    except Exception:
        pass


def _local_provider_residency_claim(
    *,
    provider: str,
    model: str,
    provider_instance: Any = None,
) -> Dict[str, Any]:
    provider_s = str(provider or "").strip().lower()
    model_s = str(model or "").strip()
    method = getattr(provider_instance, "get_model_residency", None)
    if not callable(method):
        return _unknown_provider_residency_claim(
            provider=provider_s,
            model=model_s,
            warning="AbstractCore provider does not expose verified model residency.",
        )

    try:
        raw_claim = method(task="text_generation", model=model_s)
    except Exception as exc:  # noqa: BLE001
        return _unknown_provider_residency_claim(
            provider=provider_s,
            model=model_s,
            warning=f"AbstractCore provider residency query failed: {exc}",
        )
    if not isinstance(raw_claim, dict):
        return _unknown_provider_residency_claim(
            provider=provider_s,
            model=model_s,
            warning="AbstractCore provider residency query returned a non-mapping response.",
        )

    verified = _residency_bool(raw_claim.get("provider_residency_verified")) is True
    provider_resident = _residency_bool(raw_claim.get("provider_resident"))
    if provider_resident is None and verified:
        provider_resident = _residency_bool(raw_claim.get("resident"))
    if provider_resident is None and verified:
        provider_resident = _residency_bool(raw_claim.get("loaded"))

    source = raw_claim.get("provider_residency_source") or raw_claim.get("source") or "abstractcore.provider"
    claim: Dict[str, Any] = {
        "provider_residency_verified": verified,
        "provider_resident": provider_resident,
        "provider_residency_source": str(source),
    }
    provider_state = raw_claim.get("provider_state") or raw_claim.get("state")
    if isinstance(provider_state, str) and provider_state.strip():
        claim["provider_state"] = provider_state.strip()

    blocked = {
        "task",
        "provider",
        "model",
        "runtime_id",
        "resident",
        "loaded",
        "state",
        "source",
        "isolation",
        "default",
        "pinned",
        # Lock state is runtime-owned enforcement truth (client-side pairs set;
        # managed rows are stamped by `_stamp_local_lock_state`): claims cannot
        # override it — the same post-fix blocked set core's gateway uses.
        "locked",
        "locked_at",
        "lockable",
        "cache_state",
        "runtime_cached",
        "provider_residency_verified",
        "provider_resident",
        "provider_residency_source",
        "provider_state",
    }
    for key, value in raw_claim.items():
        if key in blocked or value is None:
            continue
        claim[str(key)] = value
    return _normalize_residency_size_extras(claim)


def _local_residency_record(
    *,
    provider: str,
    model: str,
    default: bool = False,
    runtime_cached: bool = True,
    provider_instance: Any = None,
    include_provider_state: bool = True,
    lock_owner: Any = None,
) -> Dict[str, Any]:
    provider_s = str(provider or "").strip().lower()
    model_s = str(model or "").strip()
    provider_claim = (
        _local_provider_residency_claim(
            provider=provider_s,
            model=model_s,
            provider_instance=provider_instance,
        )
        if include_provider_state and (runtime_cached or provider_instance is not None)
        else {
            "provider_residency_verified": False,
            "provider_resident": None,
            "provider_residency_source": "abstractcore.provider",
            "provider_state": "provider_residency_unknown",
        }
    )

    verified = bool(provider_claim.get("provider_residency_verified"))
    provider_resident_raw = provider_claim.get("provider_resident")
    provider_resident = provider_resident_raw if isinstance(provider_resident_raw, bool) else None

    if verified:
        resident = bool(provider_resident)
        state = "provider_loaded" if resident else "provider_not_loaded"
    else:
        resident = False
        state = "provider_residency_unknown" if runtime_cached else "not_found"

    if not runtime_cached and not verified:
        resident = False
        state = "not_found"

    record = {
        "task": "text_generation",
        "provider": provider_s,
        "model": model_s,
        "runtime_id": f"local:text_generation:{provider_s}:{model_s}",
        "resident": resident,
        "loaded": resident,
        "state": state,
        "runtime_cached": bool(runtime_cached),
        "cache_state": "runtime_client_cached" if runtime_cached else "not_cached",
        # `pinned` is a truthful alias of `locked` (core parity) — NEVER the
        # default-identity flag: presenting every capability-default model as
        # "pinned" was exactly the default-vs-loaded lie. `default` alone
        # carries the default-identity pair; `_stamp_local_lock_state` raises
        # `pinned` alongside `locked` on managed list rows.
        "pinned": False,
        "default": bool(default),
        "source": "abstractruntime.local",
        "isolation": "in_process",
        **provider_claim,
    }
    # Locally-served managed rows carry core-owned modalities + host identity
    # (the gateway LOCAL lane's analog of core's server-side record stamps;
    # provider claims never include these).
    _stamp_local_record_modalities(record, model=model_s, provider_instance=provider_instance)
    _stamp_local_record_host_identity(record)
    # Per-model memory figures. `est_weights_bytes` is pure claim truth (MLX /
    # HuggingFace stamp it on `get_model_residency`) and already rode the claim
    # merge above. `cache_bytes` has no claim slot, so it is read the same way
    # core's server lane reads it — core's own arithmetic over the provider's
    # own prompt-cache stats — and only when the claim did not supply one.
    if record.get("cache_bytes") is None:
        cache_bytes = _local_record_prompt_cache_bytes(provider_instance)
        if cache_bytes is not None:
            record["cache_bytes"] = cache_bytes
    if lock_owner is not None:
        # Load/unload response records carry the same lock truth as list rows
        # (alias parity: a re-load of a LOCKED pair must not answer
        # `pinned: false` beside `lock.locked: true`).
        _stamp_local_lock_state(
            record,
            locked=(provider_s, model_s) in _local_locked_residency_pairs(lock_owner),
        )
    return record


def _local_provider_load_options(
    *,
    options: Optional[Dict[str, Any]] = None,
    pin: Optional[bool] = True,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    load_options = dict(options or {}) if isinstance(options, dict) else {}
    if pin is not None:
        load_options.setdefault("pin", bool(pin))
    if isinstance(extra, dict):
        for key in ("ttl_s", "keep_alive"):
            value = extra.get(key)
            if value is not None:
                load_options.setdefault(key, value)
    return load_options


def _load_local_provider_residency(
    *,
    provider_instance: Any,
    model: str,
    options: Optional[Dict[str, Any]] = None,
    pin: Optional[bool] = True,
    extra: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Optional[str]]:
    method = getattr(provider_instance, "load_model", None)
    if not callable(method):
        return None, "AbstractCore provider does not expose load_model(model_name)."
    try:
        load_options = _local_provider_load_options(options=options, pin=pin, extra=extra)
        return method(str(model or "").strip(), **load_options), None
    except Exception as exc:  # noqa: BLE001
        return None, f"Provider model load failed: {exc}"


def _unload_local_provider_residency(
    *,
    provider_instance: Any,
    model: str,
    options: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Optional[str]]:
    method = getattr(provider_instance, "unload_model", None)
    if not callable(method):
        return None, "AbstractCore provider does not expose unload_model(model_name)."
    # EJECT = STOP FIRST (2026-09-23): the runtime effects running this model
    # are cancelled with an attribution (`cancelled_by="model_eject"`) before
    # the provider frees anything; the provider's own unload then waits for
    # them to unwind (AbstractCore `InflightGenerations`) and refuses to free
    # memory under a call that did not stop.
    try:
        from ...core.effect_cancellation import request_model_effects_cancel

        provider_label = str(getattr(provider_instance, "provider", "") or "").strip()
        stopped = request_model_effects_cancel(
            provider_label,
            str(model or "").strip(),
            cancelled_by="model_eject",
            reason=f"model {provider_label}/{str(model or '').strip()} ejected (model residency unload)",
        )
        if stopped:
            logger.warning(
                f"model eject: cancelled {len(stopped)} in-flight effect(s) using "
                f"{provider_label}/{str(model or '').strip()} before unloading it"
            )
    except Exception as exc:  # noqa: BLE001
        logger.error(f"model eject: could not cancel in-flight effects before unload: {exc}")
    try:
        unload_options = dict(options or {}) if isinstance(options, dict) else {}
        if unload_options:
            return method(str(model or "").strip(), **unload_options), None
        return method(str(model or "").strip()), None
    except TypeError:
        try:
            return method(str(model or "").strip()), None
        except Exception as exc:  # noqa: BLE001
            return None, f"Provider model unload failed: {exc}"
    except Exception as exc:  # noqa: BLE001
        return None, f"Provider model unload failed: {exc}"


def _merge_optional_payload(payload: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge the two supported calling conventions into one request mapping.

    Contract (gateway reviewer, misbinding bugs in convention-guessing):
    - `payload` is an OPTIONAL Mapping and is only ever the payload mapping —
      the first positional argument is never reinterpreted as anything else.
    - `kwargs` merge into a COPY of the payload; kwargs win on key conflicts.
    """
    merged: Dict[str, Any] = {}
    if payload is not None:
        if not isinstance(payload, Mapping):
            raise TypeError(
                "payload must be a mapping of request fields when provided "
                f"(got {type(payload).__name__}); pass fields as keyword arguments otherwise."
            )
        merged.update(payload)
    merged.update(kwargs)
    return merged


def _local_locked_residency_pairs(owner: Any) -> set:
    """Client-side model-lock state for a local pool owner.

    Keyed by normalized (provider, model). Lazily created so test doubles
    built via `object.__new__` participate without running `__init__`."""
    pairs = getattr(owner, "_locked_model_residency", None)
    if not isinstance(pairs, set):
        pairs = set()
        try:
            owner._locked_model_residency = pairs
        except Exception:
            pass
    return pairs


def _stamp_local_lock_state(record: Dict[str, Any], *, locked: bool) -> Dict[str, Any]:
    """Stamp client-side lock truth onto a MANAGED local record (mirrors
    core's `_gateway_text_residency_record`: managed rows are ALWAYS lockable
    and `locked` is the enforcement flag, never a provider claim — the claim
    helper additionally blocks locked/lockable/locked_at, like core).

    `pinned` is a truthful alias of `locked` (same value, core parity): no
    row may carry `pinned: true` unless it is actually locked."""
    record["locked"] = bool(locked)
    record["pinned"] = bool(locked)
    record["lockable"] = True
    return record


# Ollama's server-side default keep-alive; restored on unlock so an unlocked
# model goes back to ordinary server-managed residency (same value core uses).
_OLLAMA_DEFAULT_KEEP_ALIVE = "5m"


def _apply_local_provider_side_lock_knob(
    *,
    provider: str,
    provider_instance: Any,
    model: str,
    lock: bool,
) -> Dict[str, Any]:
    """Best-effort provider-side reinforcement of the client-side lock.

    Mirrors core's `_apply_provider_side_lock_knob`: ollama honors
    `keep_alive` (-1 pins server-side; the 5m default restores normal
    behavior); no other provider exposes a residency-pin knob. The
    client-side flag is the enforcement truth either way — a provider-side
    failure is REPORTED, never raised.

    The knob rides Ollama's native load request, which would LOAD a
    non-resident model. Lock callers verify residency up front (the lock rule
    refuses non-resident pairs); unlock verifies here and skips the restore
    when the model is gone — unlocking a locked-but-since-evicted pair must
    never load it back as a side effect."""
    provider_s = str(provider or "").strip().lower()
    if provider_s != "ollama":
        if provider_s == "lmstudio" and lock:
            # Honesty note (core parity): LM Studio exposes no residency-pin
            # knob, so the lock protects only against THIS stack's unloads —
            # the external server keeps its own eviction policy.
            return {
                "supported": False,
                "applied": False,
                "detail": "lock guards this stack's unloads; the external server may still evict on its own policy",
            }
        return {"supported": False, "applied": False}
    if not lock and provider_instance is not None:
        # Restore only on VERIFIED residency (the knob rides a load request).
        # UNKNOWN is not evidence of eviction: a transient probe failure on a
        # genuinely resident model still skips the restore — the safe act —
        # but the detail must not claim "not resident" it never verified.
        claim = _local_provider_residency_claim(
            provider=str(provider or "").strip().lower(),
            model=str(model or "").strip(),
            provider_instance=provider_instance,
        )
        resident = claim.get("provider_resident")
        if resident is not True:
            detail = (
                "model is not resident server-side; keep_alive restore skipped (no load side effect)"
                if resident is False
                else "model residency unverified; keep_alive restore skipped (no load side effect)"
            )
            return {"supported": True, "applied": False, "detail": detail}
    keep_alive: Any = -1 if lock else _OLLAMA_DEFAULT_KEEP_ALIVE
    method = getattr(provider_instance, "load_model", None)
    if not callable(method):
        return {
            "supported": True,
            "applied": False,
            "detail": "provider instance does not expose load_model",
        }
    try:
        method(str(model or "").strip(), keep_alive=keep_alive)
    except Exception as exc:  # noqa: BLE001 - reported, never raised
        return {
            "supported": True,
            "applied": False,
            "detail": f"ollama keep_alive={keep_alive} update failed: {exc}",
        }
    return {"supported": True, "applied": True}


def _local_lock_pair_from_request(
    merged: Dict[str, Any],
    *,
    default_provider: str,
    default_model: str,
) -> Tuple[str, str, bool]:
    """Resolve the (provider, model) selector for a local lock/unlock request:
    explicit provider/model, else a `local:text_generation:...` runtime_id,
    else — ONLY when no selector was given at all — the client's default
    identity.

    Returns (provider, model, unresolved_runtime_id). A NON-EMPTY runtime_id
    that does not address a local text runtime must never fall back to the
    default pair: `lock(runtime_id="rid-core-42")` would otherwise lock (and
    unlock UNLOCK) a runtime the caller never addressed."""
    provider_s = str(merged.get("provider") or "").strip().lower()
    model_s = str(merged.get("model") or "").strip()
    runtime_id = str(merged.get("runtime_id") or "").strip()
    if (not provider_s or not model_s) and runtime_id:
        prefix = "local:text_generation:"
        if runtime_id.startswith(prefix):
            rest = runtime_id[len(prefix) :]
            if ":" in rest:
                rid_provider, rid_model = rest.split(":", 1)
                provider_s = provider_s or rid_provider.strip().lower()
                model_s = model_s or rid_model.strip()
        if not provider_s or not model_s:
            return provider_s, model_s, True
    if not provider_s and not model_s and not runtime_id:
        provider_s = str(default_provider or "").strip().lower()
        model_s = str(default_model or "").strip()
    return provider_s, model_s, False


def _local_model_residency_lock_result(
    merged: Dict[str, Any],
    *,
    lock: bool,
    default_provider: str,
    default_model: str,
    known_pairs: set,
    locked_pairs: set,
    provider_instance_lookup: Any,
    source: str,
    adopt_pair: Any = None,
    drop_adopted_pair: Any = None,
) -> Dict[str, Any]:
    """Shared local lock/unlock implementation (Local + MultiLocal clients).

    The client-side flag is the enforcement truth (checked by
    `unload_model_residency`); the ollama keep_alive knob is best-effort
    reinforcement reported in `provider_side`, exactly like core's
    `/acore/models/lock` contract.

    LOCK RULE (core parity): a lock requires provider-VERIFIED residency
    (`provider_resident is True`) on top of the warm-pair requirement;
    locking a non-resident pair refuses with `error: "model_not_resident"`
    (load with lock:true instead). Unlock never requires residency — a
    locked-but-since-evicted pair must always be unlockable.

    SWEEP ADOPTION (core parity with `_resolve_or_adopt_text_runtime_for_lock`):
    a lock naming a pair with NO warm pooled client, on a SWEEP provider
    (Ollama / LM Studio) whose server the host sweep verifies holds the model,
    is ADOPTED — `adopt_pair` constructs the pooled client through the ordinary
    `_get_client` path (CLIENT CONSTRUCTION ONLY, never a provider-side model
    load), residency is re-verified with the provider's own probe, and the lock
    is set; the response then carries `adopted: true`. A provider probe that
    disagrees with the sweep drops the just-adopted pooled client
    (`drop_adopted_pair`) and refuses `model_not_resident` — a stray pool entry
    would present a row nobody asked for. Not sweep-resident and not warm keeps
    today's `not_found` refusal."""
    operation = "lock" if lock else "unlock"
    task_raw = merged.get("task")
    if task_raw is not None and str(task_raw).strip():
        task_s = _normalize_residency_task(task_raw)
        if task_s != "text_generation":
            return _model_residency_unsupported_payload(
                operation=operation,
                task=task_s,
                provider=str(merged.get("provider") or "").strip().lower(),
                model=str(merged.get("model") or "").strip(),
                error=f"model_residency {operation} is only supported for text_generation runtimes.",
            )
    provider_s, model_s, unresolved_runtime_id = _local_lock_pair_from_request(
        merged,
        default_provider=default_provider,
        default_model=default_model,
    )
    if unresolved_runtime_id:
        # A foreign/unparseable runtime_id addresses NOTHING here — refuse
        # honestly (never the default pair the caller did not name).
        message = "Requested local runtime was not found in this client."
        return {
            "ok": False,
            "success": False,
            "supported": True,
            "operation": operation,
            "runtime_id": str(merged.get("runtime_id") or "").strip(),
            "error": message,
            "warnings": [message],
            "affected_models": [],
            "diagnostics": {"source": source, "reason": "not_found"},
        }
    if not provider_s or not model_s:
        message = f"model_residency {operation} requires runtime_id or provider/model"
        return {
            "ok": False,
            "success": False,
            "supported": True,
            "operation": operation,
            "error": message,
            "warnings": [message],
            "affected_models": [],
        }
    pair = (provider_s, model_s)
    adopted = False
    # Unlock must stay reachable for a locked pair even after its pooled
    # client was evicted; lock requires a known (warm) local runtime, the
    # analog of core's registry-entry requirement — OR a sweep-resident pair
    # this client can adopt (see the ADOPTION paragraph in the docstring).
    if pair not in known_pairs and not (not lock and pair in locked_pairs):
        if lock and callable(adopt_pair) and _sweep_verifies_model_resident(provider_s, model_s):
            adopted = True
        else:
            message = "Requested local runtime was not found in this client."
            return {
                "ok": False,
                "success": False,
                "supported": True,
                "operation": operation,
                "provider": provider_s,
                "model": model_s,
                "error": message,
                "warnings": [message],
                "affected_models": [],
                "diagnostics": {"source": source, "reason": "not_found"},
            }
    if adopted:
        # CONSTRUCTION ONLY: the ordinary pooled-client path, which builds the
        # provider client without ever asking it to load a model.
        try:
            provider_instance = adopt_pair(provider_s, model_s)
        except Exception as exc:  # noqa: BLE001 - a failed adoption is reported, never raised
            message = f"Adopting sweep-resident {provider_s}/{model_s} failed: {exc}"
            return {
                "ok": False,
                "success": False,
                "supported": True,
                "operation": operation,
                "provider": provider_s,
                "model": model_s,
                "error": message,
                "warnings": [message],
                "affected_models": [],
                "diagnostics": {"source": source, "reason": "adoption_failed"},
            }
    else:
        provider_instance = provider_instance_lookup(provider_s, model_s) if callable(provider_instance_lookup) else None
    if lock:
        # LOCK RULE (core parity): lock requires provider-VERIFIED residency.
        # A warm pool client alone is configuration, not memory — for lmstudio
        # a constructed HTTP client counted as "warm" with nothing resident,
        # and the lock presented a configured model as loaded.
        claim = _local_provider_residency_claim(
            provider=provider_s,
            model=model_s,
            provider_instance=provider_instance,
        )
        if claim.get("provider_resident") is not True:
            if adopted and callable(drop_adopted_pair):
                # The provider's own probe disagreed with the sweep: drop the
                # just-adopted pooled client — adoption did not complete, and a
                # stray pool entry would present a row nobody asked for.
                try:
                    drop_adopted_pair(provider_s, model_s)
                except Exception:
                    pass
            detail = (
                f"Model {provider_s}/{model_s} is not resident in provider memory; "
                "load it first (load with lock:true) before locking."
            )
            return {
                "ok": False,
                "success": False,
                "supported": True,
                "operation": operation,
                "provider": provider_s,
                "model": model_s,
                "runtime_id": f"local:text_generation:{provider_s}:{model_s}",
                "error": "model_not_resident",
                "detail": detail,
                "warnings": [detail],
                "affected_models": [],
                "diagnostics": {"source": source, "reason": "model_not_resident"},
            }
        locked_pairs.add(pair)
    else:
        locked_pairs.discard(pair)
    provider_side = _apply_local_provider_side_lock_knob(
        provider=provider_s,
        provider_instance=provider_instance,
        model=model_s,
        lock=lock,
    )
    return {
        "ok": True,
        "operation": operation,
        "locked": bool(lock),
        "runtime_id": f"local:text_generation:{provider_s}:{model_s}",
        "provider": provider_s,
        "model": model_s,
        "provider_side": provider_side,
        **({"adopted": True} if adopted else {}),
        "diagnostics": {"source": source, **({"adopted": True} if adopted else {})},
    }


def _local_model_locked_refusal(*, provider: str, model: str, source: str) -> Dict[str, Any]:
    """The soft refusal payload for unloading a client-side-locked pair
    without force (the local analog of core's 409 `model_locked` envelope;
    a payload, never an exception)."""
    provider_s = str(provider or "").strip().lower()
    model_s = str(model or "").strip()
    return {
        "ok": False,
        "success": False,
        "supported": True,
        "operation": "unload",
        "task": "text_generation",
        "unloaded": False,
        "error": "model_locked",
        "detail": (
            f"Model residency for {provider_s}/{model_s} is locked; pass force=true to unload."
        ),
        "runtime_id": f"local:text_generation:{provider_s}:{model_s}",
        "provider": provider_s,
        "model": model_s,
        "affected_models": [],
        "diagnostics": {"source": source, "reason": "model_locked"},
    }


def _local_context_estimate(
    *,
    provider: str,
    model: str,
    context_length: Any = None,
    base_url: Any = None,
) -> Dict[str, Any]:
    """Relay core's analytical context-fit estimator (ADR 0007: core-owned
    truth; an absent estimator degrades to the structured unsupported
    envelope instead of raising)."""
    provider_s = str(provider or "").strip().lower()
    model_s = str(model or "").strip()
    if not provider_s or not model_s:
        message = "context_estimate requires provider and model"
        return {
            "ok": False,
            "success": False,
            "supported": True,
            "operation": "context_estimate",
            "error": message,
            "warnings": [message],
        }
    try:
        from abstractcore.utils.context_estimate import estimate_context_fit  # type: ignore
    except Exception as exc:
        return {
            "ok": False,
            "supported": False,
            "operation": "context_estimate",
            "error": f"AbstractCore context estimator is unavailable: {exc}",
        }
    ctx: Optional[int] = None
    if context_length is not None and not isinstance(context_length, bool):
        try:
            ctx = int(context_length)
        except Exception:
            ctx = None
    call_kwargs: Dict[str, Any] = {"context_length": ctx}
    base_s = str(base_url or "").strip()
    if base_s:
        call_kwargs["base_url"] = base_s
    try:
        result = estimate_context_fit(provider_s, model_s, **call_kwargs)
    except Exception as exc:  # noqa: BLE001 - the estimator never raises; belt and braces
        return {"ok": False, "operation": "context_estimate", "error": str(exc)}
    if not isinstance(result, dict):
        return {"ok": False, "operation": "context_estimate", "error": "invalid context estimate response"}
    return dict(result)


def _provider_supports_uncached_text_residency(provider: str) -> bool:
    provider_s = str(provider or "").strip().lower()
    if not provider_s:
        return False
    try:
        from abstractcore.providers.registry import get_provider_registry  # type: ignore

        provider_cls = get_provider_registry().get_provider_class(provider_s)
    except Exception:
        return False
    mode = getattr(provider_cls, "TEXT_MODEL_RESIDENCY_CONTROL_PLANE", None)
    return str(mode or "").strip().lower() == "server"


def _local_model_record_summary(
    runtime: Dict[str, Any],
    *,
    operation: str,
    action: str,
    changed: bool,
) -> Dict[str, Any]:
    keys = (
        "task",
        "provider",
        "model",
        "runtime_id",
        "loaded",
        "resident",
        "state",
        "runtime_cached",
        "cache_state",
        "provider_residency_verified",
        "provider_resident",
        "provider_residency_source",
        "provider_state",
        "provider_instance_ids",
    )
    out = {key: runtime.get(key) for key in keys if key in runtime}
    out["operation"] = operation
    out["action"] = action
    out["changed"] = bool(changed)
    return out


def _local_model_residency_load_failure(
    *,
    operation: str,
    task: str,
    provider: str,
    model: str,
    runtime: Dict[str, Any],
    message: str,
    source: str,
    runtime_cache_loaded_new: Optional[bool] = None,
    provider_load_result: Any = None,
) -> Dict[str, Any]:
    diagnostics: Dict[str, Any] = {"source": source}
    if runtime_cache_loaded_new is not None:
        diagnostics["runtime_cache_loaded_new"] = bool(runtime_cache_loaded_new)
    out: Dict[str, Any] = {
        "ok": False,
        "supported": True,
        "operation": operation,
        "task": task,
        "provider": provider or None,
        "model": model or None,
        "loaded_new": False,
        "runtime": runtime,
        "error": message,
        "warnings": [message],
        "status_hint": "warning",
        "degraded": True,
        "diagnostics": diagnostics,
    }
    if runtime_cache_loaded_new is not None:
        out["runtime_cache_loaded_new"] = bool(runtime_cache_loaded_new)
    if provider_load_result is not None:
        out["provider_load_result"] = _jsonable(provider_load_result)
    out["success"] = False
    out["affected_models"] = [
        _local_model_record_summary(
            runtime,
            operation=operation,
            action="load_failed",
            changed=False,
        )
    ]
    return out


def _local_capability_residency_core(holder: Any) -> Any:
    core = getattr(holder, "_capability_residency_core", None)
    if core is not None:
        return core

    lock = getattr(holder, "_capability_residency_core_lock", None)
    if lock is None:
        lock = threading.Lock()
        setattr(holder, "_capability_residency_core_lock", lock)

    with lock:
        core = getattr(holder, "_capability_residency_core", None)
        if core is not None:
            return core
        try:
            from abstractcore.server.capability_generation import create_capability_generation_core  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"AbstractCore capability residency bridge is unavailable: {exc}") from exc
        core = create_capability_generation_core()
        setattr(holder, "_capability_residency_core", core)
        return core


def _local_capability_residency_target(core: Any, task: str) -> Tuple[Any, str]:
    task_s = _normalize_residency_task(task)
    if task_s == "tts":
        return getattr(core, "voice", None), "voice"
    if task_s == "stt":
        return getattr(core, "audio", None), "audio"
    if task_s in {"music_generation", "text_to_audio"}:
        # abstractmusic serves text_to_audio (sound effects) beside music.
        return getattr(core, "music", None), "music"
    if task_s in {"image_generation", "image_to_image", "image_upscale", "video_generation", "text_to_video",
                  "image_to_video"}:
        # abstractvision's facade owns image, upscale AND video residency.
        return getattr(core, "vision", None), "vision"
    raise ValueError(f"Unsupported local capability residency task: {task!r}")


def _local_capability_residency_record(record: Dict[str, Any], *, task: str, source: str) -> Dict[str, Any]:
    out = dict(record)
    task_s = _normalize_residency_task(out.get("task") or task)
    out["task"] = task_s
    provider_s = str(out.get("provider") or out.get("backend_kind") or out.get("engine") or "").strip().lower()
    model_s = str(out.get("model") or out.get("model_id") or out.get("engine") or "").strip()
    if provider_s:
        out["provider"] = provider_s
    if model_s:
        out["model"] = model_s

    loaded = _residency_bool(out.get("loaded"))
    resident = _residency_bool(out.get("resident"))
    if loaded is None and resident is not None:
        loaded = resident
    if loaded is not None:
        out["loaded"] = bool(loaded)
        out.setdefault("resident", bool(loaded))
        out.setdefault("provider_residency_verified", True)
        out.setdefault("provider_resident", bool(loaded))
        out.setdefault("provider_loaded", bool(loaded))
        out.setdefault("provider_residency_source", "abstractcore.capability_plugin")

    state = str(out.get("state") or "").strip().lower()
    if not state:
        if loaded is True:
            out["state"] = "resident"
        elif out.get("error"):
            out["state"] = "failed"
        else:
            out["state"] = "configured"
    out.setdefault("provider_state", str(out.get("state") or "").strip().lower())

    runtime_id = str(out.get("runtime_id") or out.get("load_id") or "").strip()
    if not runtime_id:
        runtime_id = f"local:{task_s}:{provider_s or 'default'}:{model_s or 'default'}"
    out["runtime_id"] = runtime_id
    out.setdefault("load_id", runtime_id)
    out.setdefault("source", source)
    out.setdefault("isolation", "in_process")
    out.setdefault("runtime_cached", bool(out.get("loaded") is True or out.get("resident") is True))
    return out


def _local_all_model_residency_result(
    holder: Any,
    *,
    source: str,
    text_records: List[Dict[str, Any]],
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    provider_s = str(provider or "").strip().lower()
    model_s = str(model or "").strip()
    records = list(text_records)
    # Sweep-only rows carry no task label (relayed, never inferred) and must
    # not be counted as verified text_generation runtimes.
    task_counts: Dict[str, int] = {
        "text_generation": sum(
            1 for record in text_records if str(record.get("task") or "") == "text_generation"
        )
    }
    task_errors: Dict[str, str] = {}

    for task_s in _LOCAL_CAPABILITY_RESIDENCY_LIST_TASKS:
        result = _local_capability_residency_result(
            holder,
            operation="list_loaded",
            task=task_s,
            provider=provider_s or None,
            model=model_s or None,
            source=source,
        )
        task_records = [
            dict(item)
            for item in list(result.get("models") or [])
            if isinstance(item, dict)
        ] if isinstance(result, dict) else []
        records.extend(task_records)
        task_counts[task_s] = len(task_records)
        if isinstance(result, dict) and result.get("ok") is False and result.get("error"):
            task_errors[task_s] = str(result.get("error"))

    deduped: List[Dict[str, Any]] = []
    seen_record_keys: set[str] = set()
    for record in records:
        record_key = str(record.get("runtime_id") or record.get("load_id") or "").strip()
        if not record_key:
            record_key = "|".join(
                str(record.get(key) or "").strip()
                for key in ("task", "provider", "model")
            )
        if record_key and record_key in seen_record_keys:
            continue
        if record_key:
            seen_record_keys.add(record_key)
        deduped.append(record)

    records = sorted(deduped, key=lambda item: str(item.get("runtime_id") or item.get("load_id") or ""))
    diagnostics: Dict[str, Any] = {"source": source, "count": len(records), "task_counts": task_counts}
    if task_errors:
        diagnostics["task_errors"] = task_errors
    result = {
        "ok": True,
        "supported": True,
        "operation": "list_loaded",
        "models": records,
        "diagnostics": diagnostics,
    }
    return _with_local_model_residency_summary(result, operation="list_loaded", models=records)


def _local_capability_residency_loaded_new(runtime: Dict[str, Any]) -> bool:
    for key in ("loaded_new", "created_new", "created", "warmed_new", "preloaded_new"):
        parsed = _residency_bool(runtime.get(key))
        if parsed is not None:
            return bool(parsed)
    details = runtime.get("details")
    if isinstance(details, dict):
        before = _residency_bool(details.get("engine_cached_before"))
        after = _residency_bool(details.get("engine_cached_after"))
        if before is not None and after is not None:
            return bool(after and not before)
    return False


def _local_capability_residency_result(
    holder: Any,
    *,
    operation: str,
    task: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
    pin: Optional[bool] = True,
    runtime_id: Optional[str] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    source: str,
) -> Dict[str, Any]:
    task_s = _normalize_residency_task(task)
    provider_s = str(provider or "").strip().lower()
    model_s = str(model or "").strip()
    if task_s == "embedding":
        try:
            return _local_embedding_residency_result(
                operation=operation, provider=provider_s, model=model_s, source=source,
            )
        except Exception as exc:  # noqa: BLE001 - an incompatible core is reported, never read as "none"
            return _model_residency_unsupported_payload(
                operation=operation, task=task_s, provider=provider_s, model=model_s, error=str(exc),
            )
    try:
        target, capability = _local_capability_residency_target(_local_capability_residency_core(holder), task_s)
    except Exception as exc:  # noqa: BLE001
        return _model_residency_unsupported_payload(
            operation=operation,
            task=task_s,
            provider=provider_s,
            model=model_s,
            error=str(exc),
        )
    if target is None:
        return _model_residency_unsupported_payload(
            operation=operation,
            task=task_s,
            provider=provider_s,
            model=model_s,
            error=f"Local {task_s} capability facade is unavailable.",
        )

    payload: Dict[str, Any] = {
        "task": task_s,
        "provider": provider_s or None,
        "model": model_s or None,
        "options": dict(options or {}) if isinstance(options, dict) else {},
    }
    if runtime_id:
        payload["runtime_id"] = str(runtime_id)
        payload["load_id"] = str(runtime_id)
    if pin is not None:
        payload["pin"] = bool(pin)
    if isinstance(kwargs, dict):
        for key in ("base_url", "timeout_s", "ttl_s", "provider_api_key"):
            value = kwargs.get(key)
            if value is not None and value != "":
                payload[key] = value

    try:
        if operation == "list_loaded":
            method = getattr(target, "list_loaded_models", None)
            if not callable(method):
                method = getattr(target, "list_resident_models", None)
            if not callable(method):
                raise RuntimeError(f"Local {capability} capability does not expose loaded-model listing.")
            filters = {k: v for k, v in payload.items() if k in {"task", "provider", "model", "runtime_id", "load_id"} and v}
            records = [
                _local_capability_residency_record(dict(item), task=task_s, source=source)
                for item in list(method(filters) or [])
                if isinstance(item, dict)
            ]
            return _with_local_model_residency_summary(
                {
                    "ok": True,
                    "supported": True,
                    "operation": "list_loaded",
                    "task": task_s,
                    "models": records,
                    "diagnostics": {"source": source, "capability": capability, "count": len(records)},
                },
                operation="list_loaded",
                models=records,
            )
        if operation == "load":
            method = getattr(target, "load_resident_model", None)
            if not callable(method):
                raise RuntimeError(f"Local {capability} capability does not expose load_resident_model.")
            runtime = _local_capability_residency_record(dict(method(payload) or {}), task=task_s, source=source)
            if runtime.get("loaded") is not True:
                runtime_error = runtime.get("error")
                if isinstance(runtime_error, dict):
                    message = str(runtime_error.get("message") or runtime_error.get("code") or "").strip()
                else:
                    message = str(runtime_error or "").strip()
                if not message:
                    message = "model_residency load completed without a loaded model"
                return _local_model_residency_load_failure(
                    operation="load",
                    task=task_s,
                    provider=provider_s or runtime.get("provider"),
                    model=model_s or runtime.get("model"),
                    runtime=runtime,
                    message=message,
                    source=source,
                )
            loaded_new = _local_capability_residency_loaded_new(runtime)
            return _with_local_model_residency_summary(
                {
                    "ok": True,
                    "supported": True,
                    "operation": "load",
                    "task": task_s,
                    "provider": provider_s or runtime.get("provider"),
                    "model": model_s or runtime.get("model"),
                    "loaded_new": loaded_new,
                    "runtime": runtime,
                    "diagnostics": {"source": source, "capability": capability, "loaded_new": loaded_new},
                },
                operation="load",
                runtime=runtime,
                action="loaded" if loaded_new else "already_loaded",
                changed=loaded_new,
            )
        if operation == "unload":
            method = getattr(target, "unload_resident_model", None)
            if not callable(method):
                raise RuntimeError(f"Local {capability} capability does not expose unload_resident_model.")
            runtime = _local_capability_residency_record(dict(method(payload) or {}), task=task_s, source=source)
            unloaded = _residency_bool(runtime.get("unloaded"))
            changed = bool(unloaded is not False)
            return _with_local_model_residency_summary(
                {
                    "ok": True,
                    "supported": True,
                    "operation": "unload",
                    "task": task_s,
                    "provider": provider_s or runtime.get("provider"),
                    "model": model_s or runtime.get("model"),
                    "unloaded": changed,
                    "runtime": runtime,
                    "diagnostics": {"source": source, "capability": capability, "unloaded": changed},
                },
                operation="unload",
                runtime=runtime,
                action="unloaded" if changed else "not_unloaded",
                changed=changed,
            )
    except Exception as exc:  # noqa: BLE001
        return _local_model_residency_load_failure(
            operation=operation,
            task=task_s,
            provider=provider_s,
            model=model_s,
            runtime=_local_capability_residency_record(
                {
                    "task": task_s,
                    "provider": provider_s,
                    "model": model_s,
                    "loaded": False,
                    "state": "failed",
                    "error": {"code": "capability_residency_error", "message": str(exc)},
                },
                task=task_s,
                source=source,
            ),
            message=str(exc),
            source=source,
        )

    return _model_residency_unsupported_payload(
        operation=operation,
        task=task_s,
        provider=provider_s,
        model=model_s,
        error=f"Unsupported model_residency operation: {operation!r}",
    )


def _with_local_model_residency_summary(
    result: Dict[str, Any],
    *,
    operation: str,
    runtime: Optional[Dict[str, Any]] = None,
    models: Optional[List[Dict[str, Any]]] = None,
    action: Optional[str] = None,
    changed: bool = False,
) -> Dict[str, Any]:
    out = result
    out["success"] = out.get("ok") is not False
    if models is not None:
        out["affected_models"] = [
            _local_model_record_summary(
                item,
                operation=operation,
                action=action or "listed",
                changed=False,
            )
            for item in models
            if isinstance(item, dict)
        ]
        return out
    if isinstance(runtime, dict):
        out["affected_models"] = [
            _local_model_record_summary(
                runtime,
                operation=operation,
                action=action or operation,
                changed=changed,
            )
        ]
    return out


def _artifact_id_from_media_item(item: Any) -> Optional[str]:
    if not isinstance(item, dict):
        return None
    for key in ("$artifact", "artifact_id"):
        raw = item.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return None


def _media_items_need_artifact_store(media: Optional[List[Any]]) -> bool:
    return any(_artifact_id_from_media_item(item) is not None for item in list(media or []))


def _resolve_media_artifacts(
    media: Optional[List[Any]],
    *,
    artifact_store: Optional[Any],
    temp_dir: Optional[str] = None,
) -> Optional[List[Any]]:
    if not media:
        return media
    if artifact_store is None:
        if _media_items_need_artifact_store(media):
            raise ValueError("Artifact-backed media requires an ArtifactStore.")
        return media

    load_fn = getattr(artifact_store, "load", None)
    meta_fn = getattr(artifact_store, "get_metadata", None)
    content_path_fn = getattr(artifact_store, "_content_path", None)
    out: List[Any] = []

    for item in list(media):
        if not isinstance(item, dict):
            out.append(item)
            continue

        aid = _artifact_id_from_media_item(item)
        if aid is None:
            out.append(item)
            continue

        # If caller already provided a path/content, keep it.
        if isinstance(item.get("file_path"), str) and str(item.get("file_path") or "").strip():
            out.append(item)
            continue
        if item.get("content") is not None:
            out.append(item)
            continue

        meta = None
        if callable(meta_fn):
            try:
                meta = meta_fn(str(aid))
            except Exception:
                meta = None

        content_type = ""
        if isinstance(item.get("content_type"), str):
            content_type = str(item.get("content_type") or "")
        elif isinstance(item.get("mime_type"), str):
            content_type = str(item.get("mime_type") or "")
        if not content_type and meta is not None:
            content_type = str(getattr(meta, "content_type", "") or "")

        filename = ""
        if isinstance(item.get("filename"), str):
            filename = str(item.get("filename") or "")
        elif meta is not None:
            tags = getattr(meta, "tags", None)
            if isinstance(tags, dict):
                filename = str(tags.get("filename") or tags.get("path") or "")

        file_path = ""
        if callable(content_path_fn):
            try:
                p = content_path_fn(str(aid))
                if hasattr(p, "exists") and p.exists():
                    file_path = str(p)
                    guessed_ext = mimetypes.guess_extension(content_type or "") or ""
                    current_ext = os.path.splitext(file_path)[1].lower()
                    if (
                        isinstance(temp_dir, str)
                        and temp_dir.strip()
                        and guessed_ext
                        and current_ext in {"", ".bin"}
                    ):
                        typed_path = os.path.join(temp_dir, f"{str(aid).strip()}{guessed_ext}")
                        try:
                            with open(file_path, "rb") as src, open(typed_path, "wb") as dst:
                                dst.write(src.read())
                            file_path = typed_path
                        except Exception:
                            pass
            except Exception:
                file_path = ""

        if not file_path and callable(load_fn):
            try:
                art = load_fn(str(aid))
            except Exception:
                art = None
            if art is not None and getattr(art, "content", None) is not None:
                raw = bytes(getattr(art, "content") or b"")
                ext = os.path.splitext(filename)[1] if filename else ""
                if not ext:
                    ext = mimetypes.guess_extension(content_type or "") or ""
                if not ext:
                    ext = ".bin"
                try:
                    if isinstance(temp_dir, str) and temp_dir.strip():
                        p = os.path.join(temp_dir, f"{str(aid).strip()}{ext}")
                        with open(p, "wb") as f:
                            f.write(raw)
                        file_path = p
                    else:
                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext, prefix="artifact_")
                        tmp.write(raw)
                        tmp.flush()
                        tmp.close()
                        file_path = tmp.name
                except Exception:
                    file_path = ""

        if file_path:
            # Preserve content type alongside the resolved path. Artifact stores
            # often use extensionless content paths, so a raw path can lose the
            # modality and make downstream transcription reject valid audio.
            resolved: Dict[str, Any] = {"file_path": str(file_path)}
            for key in ("role", "purpose", "kind"):
                raw_role = item.get(key)
                if isinstance(raw_role, str) and raw_role.strip():
                    resolved[key] = raw_role.strip()
            if filename:
                resolved["filename"] = str(filename)
            if content_type:
                resolved["content_type"] = str(content_type)
                resolved["mime_type"] = str(content_type)
            base_type = str(content_type or "").split(";", 1)[0].strip().lower()
            if base_type.startswith("audio/"):
                resolved["type"] = "audio"
            elif base_type.startswith("image/"):
                resolved["type"] = "image"
            elif base_type.startswith("video/"):
                resolved["type"] = "video"
            elif base_type.startswith("text/"):
                resolved["type"] = "text"
            resolved["artifact_id"] = str(aid)
            resolved["$artifact"] = str(aid)
            out.append(resolved)
            continue

        raise ValueError(f"Unable to resolve artifact '{aid}' to provider-ready media content.")

    return out or media


def _loads_dict_like(raw: Any) -> Optional[Dict[str, Any]]:
    """Parse a JSON-ish or Python-literal dict safely."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    candidate = re.sub(r"\btrue\b", "True", text, flags=re.IGNORECASE)
    candidate = re.sub(r"\bfalse\b", "False", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\bnull\b", "None", candidate, flags=re.IGNORECASE)
    try:
        parsed = ast.literal_eval(candidate)
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    return {str(k): v for k, v in parsed.items()}


def _normalize_tool_calls(tool_calls: Any) -> Optional[List[Dict[str, Any]]]:
    """Normalize tool call shapes into AbstractRuntime's standard dict form.

    Standard shape:
        {"name": str, "arguments": dict, "call_id": Optional[str]}
    """
    if tool_calls is None:
        return None
    if not isinstance(tool_calls, list):
        return None

    normalized: List[Dict[str, Any]] = []
    for tc in tool_calls:
        name: Optional[str] = None
        arguments: Any = None
        call_id: Any = None

        if isinstance(tc, dict):
            call_id = tc.get("call_id", None)
            if call_id is None:
                call_id = tc.get("id", None)

            raw_name = tc.get("name")
            raw_args = tc.get("arguments")

            func = tc.get("function") if isinstance(tc.get("function"), dict) else None
            if func and (not isinstance(raw_name, str) or not raw_name.strip()):
                raw_name = func.get("name")
            if func and raw_args is None:
                raw_args = func.get("arguments")

            if isinstance(raw_name, str):
                name = raw_name.strip()
            arguments = raw_args if raw_args is not None else {}
        else:
            raw_name = getattr(tc, "name", None)
            raw_args = getattr(tc, "arguments", None)
            call_id = getattr(tc, "call_id", None)
            if isinstance(raw_name, str):
                name = raw_name.strip()
            arguments = raw_args if raw_args is not None else {}

        if not isinstance(name, str) or not name:
            continue

        if isinstance(arguments, str):
            parsed = _loads_dict_like(arguments)
            arguments = parsed if isinstance(parsed, dict) else {}

        if not isinstance(arguments, dict):
            arguments = {}

        normalized.append(
            {
                "name": name,
                "arguments": _jsonable(arguments),
                "call_id": str(call_id) if call_id is not None else None,
            }
        )

    return normalized or None


def _artifact_ref_payload(ref: Any, *, content_type: Optional[str] = None, size_bytes: Optional[int] = None) -> Optional[Dict[str, Any]]:
    if not isinstance(ref, dict):
        return None
    artifact_id = ref.get("$artifact") or ref.get("artifact_id") or ref.get("id")
    if not isinstance(artifact_id, str) or not artifact_id.strip():
        return None
    out = dict(ref)
    out["$artifact"] = artifact_id.strip()
    out["artifact_id"] = artifact_id.strip()
    if content_type and "content_type" not in out:
        out["content_type"] = str(content_type)
    if size_bytes is not None and "size_bytes" not in out:
        out["size_bytes"] = int(size_bytes)
    return out


def _string_tags(tags: Optional[Dict[str, Any]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not isinstance(tags, dict):
        return out
    for k, v in tags.items():
        if k is None or v is None:
            continue
        out[str(k)] = str(v)
    return out


def _artifact_render_kind(content_type: str) -> str:
    ct = str(content_type or "").strip().lower()
    if ct.startswith("image/"):
        return "image"
    if ct.startswith("audio/"):
        return "audio"
    if ct.startswith("video/"):
        return "video"
    if ct in {
        "application/javascript",
        "application/x-javascript",
        "application/typescript",
        "application/x-python-code",
        "application/x-sh",
        "application/x-shellscript",
        "text/javascript",
        "text/jsx",
        "text/typescript",
        "text/tsx",
        "text/x-python",
        "text/x-shellscript",
        "text/x-script.python",
        "text/x-go",
        "text/x-rust",
        "text/x-java-source",
        "text/x-c",
        "text/x-c++",
        "text/x-csharp",
        "text/x-php",
        "text/x-ruby",
    }:
        return "code"
    if ct in {"application/json"} or ct.endswith("+json"):
        return "json"
    if ct in {"text/markdown", "text/x-markdown"}:
        return "markdown"
    if "html" in ct:
        return "html"
    if ct.startswith("text/"):
        return "text"
    if "pdf" in ct or "document" in ct or "officedocument" in ct:
        return "document"
    return "binary"


def _generated_semantic_kind(*, modality: str, task: str, content_type: str) -> str:
    modality0 = str(modality or "").strip().lower()
    task0 = str(task or "").strip().lower()
    if modality0 in {"voice", "music", "sound", "image", "video", "code"}:
        return modality0
    if task0 in {"code", "coding", "code_generation", "script", "program"}:
        return "code"
    if task0 in {"tts", "speech", "speech_generation"}:
        return "voice"
    if task0 in {"music", "music_generation", "text_to_music", "lyrics_to_music"}:
        return "music"
    if task0 in {"sound", "sound_generation", "text_to_audio", "audio_generation"}:
        return "sound"
    if task0 in {"stt", "transcription", "speech_to_text"}:
        return "transcript"
    render = _artifact_render_kind(content_type)
    return render if render != "binary" else (modality0 or "artifact")


def _safe_artifact_value(value: Any, *, depth: int = 0) -> Any:
    """Return bounded, JSON-safe artifact metadata with obvious secret fields redacted."""

    if depth > 6:
        return "#TRUNCATION: metadata depth limit"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, (bytes, bytearray)):
        return {"bytes": len(value), "redacted": True}
    if isinstance(value, str):
        if len(value) > 12000:
            return value[:12000] + "#TRUNCATION"
        return value
    if isinstance(value, (list, tuple)):
        items = list(value)
        out = [_safe_artifact_value(v, depth=depth + 1) for v in items[:120]]
        if len(items) > 120:
            out.append({"truncated_items": len(items) - 120})
        return out
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key, raw in list(value.items())[:160]:
            key_s = str(key)
            key_l = key_s.lower()
            if key_l in {
                "access_token",
                "api_token",
                "auth_token",
                "bearer_token",
                "bot_token",
                "client_secret",
                "csrf_token",
                "id_token",
                "password",
                "refresh_token",
                "secret",
                "session_token",
                "token",
            } or any(fragment in key_l for fragment in ("api_key", "api-key", "apikey", "authorization", "private_key")):
                out[key_s] = "[redacted]" if raw not in (None, "") else raw
                continue
            token_parts = [p for p in re.split(r"[^a-z0-9]+", key_l) if p]
            if "password" in token_parts or "secret" in token_parts or "token" in token_parts:
                out[key_s] = "[redacted]" if raw not in (None, "") else raw
                continue
            if key_l in {"data", "b64_json", "image", "video", "audio", "content"} and isinstance(raw, str) and len(raw) > 2048:
                out[key_s] = {"chars": len(raw), "redacted": True}
                continue
            if callable(raw):
                continue
            out[key_s] = _safe_artifact_value(raw, depth=depth + 1)
        if len(value) > 160:
            out["truncated_keys"] = len(value) - 160
        return out
    return _jsonable(value)


def _generated_artifact_context(*, prompt: str, media: Optional[List[Any]], output_request: Any) -> Dict[str, Any]:
    specs: List[Dict[str, Any]] = []
    if _is_abstractcore_output_request(output_request):
        try:
            specs = [dict(s) for s in _normalize_output_specs_for_runtime(output_request) if isinstance(s, dict)]
        except Exception:
            specs = []
    elif isinstance(output_request, dict):
        specs = [dict(output_request)]

    source_refs: List[Dict[str, Any]] = []
    for item in list(media or []):
        if not isinstance(item, dict):
            continue
        artifact_id = _artifact_id_from_media_item(item)
        if not artifact_id:
            continue
        ref: Dict[str, Any] = {"kind": "artifact", "artifact_id": artifact_id}
        for src_key, out_key in (
            ("run_id", "run_id"),
            ("content_type", "content_type"),
            ("mime_type", "content_type"),
            ("type", "modality"),
            ("role", "role"),
            ("purpose", "role"),
            ("filename", "filename"),
        ):
            raw = item.get(src_key)
            if raw is not None and str(raw).strip() and out_key not in ref:
                ref[out_key] = str(raw).strip()
        source_refs.append(ref)

    return {"prompt": str(prompt or ""), "specs": specs, "source_refs": source_refs}


def _matching_output_spec(context: Optional[Dict[str, Any]], *, modality: str, task: str) -> Dict[str, Any]:
    specs = context.get("specs") if isinstance(context, dict) else None
    if not isinstance(specs, list):
        return {}
    modality0 = str(modality or "").strip().lower()
    task0 = str(task or "").strip().lower()
    fallback: Dict[str, Any] = {}
    for spec in specs:
        if not isinstance(spec, dict):
            continue
        spec_modality = str(spec.get("modality") or "").strip().lower()
        spec_task = str(spec.get("task") or "").strip().lower()
        if spec_modality == modality0 and (not task0 or spec_task == task0):
            return dict(spec)
        if spec_modality == modality0 and not fallback:
            fallback = dict(spec)
    return fallback


def _generation_params_from_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    excluded = {
        "artifact_id",
        "data",
        "input",
        "media",
        "modality",
        "model",
        "output",
        "prompt",
        "provider",
        "run_id",
        "tags",
        "task",
        "text",
        "type",
    }
    out: Dict[str, Any] = {}
    for key, value in dict(spec or {}).items():
        key_s = str(key)
        if key_s in excluded or callable(value):
            continue
        if value is None:
            continue
        out[key_s] = _safe_artifact_value(value)
    return out


def _generated_artifact_descriptor_and_metadata(
    *,
    item: Any,
    content_type: str,
    tags: Dict[str, str],
    generation_context: Optional[Dict[str, Any]],
    item_index: int,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    modality = str(_field(item, "modality", "") or "").strip().lower()
    task = str(_field(item, "task", "") or "").strip().lower()
    spec = _matching_output_spec(generation_context, modality=modality, task=task)
    semantic_kind = _generated_semantic_kind(modality=modality, task=task, content_type=content_type)
    render_kind = _artifact_render_kind(content_type)

    prompt = str((generation_context or {}).get("prompt") or "").strip()
    provider = (
        str(_field(item, "provider", "") or "").strip()
        or str(spec.get("provider") or "").strip()
        or str(tags.get("provider") or "").strip()
    )
    model = (
        str(_field(item, "model", "") or "").strip()
        or str(spec.get("model") or "").strip()
        or str(tags.get("model") or "").strip()
    )
    backend = str(_field(item, "backend_id", "") or spec.get("backend_id") or spec.get("backend") or "").strip()

    producer: Dict[str, Any] = {
        "package": "abstractruntime.integrations.abstractcore",
        "capability_route": task or modality or "generated_media",
    }
    if provider:
        producer["provider"] = provider
    if model:
        producer["model"] = model
    if backend:
        producer["backend"] = backend
    runtime_provider = str(tags.get("provider") or "").strip()
    runtime_model = str(tags.get("model") or "").strip()
    if runtime_provider and runtime_provider != provider:
        producer["runtime_provider"] = runtime_provider
    if runtime_model and runtime_model != model:
        producer["runtime_model"] = runtime_model

    generation: Dict[str, Any] = {"output_index": int(item_index)}
    if prompt:
        prompt_value = _safe_artifact_value(prompt)
        if task in {"tts", "speech", "speech_generation"} or modality == "voice":
            generation["text"] = prompt_value
        else:
            generation["prompt"] = prompt_value
    fmt = str(_field(item, "format", "") or spec.get("format") or spec.get("output_format") or "").strip()
    if fmt:
        generation["requested_format"] = fmt
    negative_prompt = spec.get("negative_prompt")
    if negative_prompt is not None and str(negative_prompt).strip():
        generation["negative_prompt"] = _safe_artifact_value(negative_prompt)
    params = _generation_params_from_spec(spec)
    if params:
        generation["params"] = params

    source_refs = list((generation_context or {}).get("source_refs") or [])
    provenance: Dict[str, Any] = {"source": tags.get("source") or "llm_call"}
    if tags.get("run_id"):
        provenance["run_id"] = tags["run_id"]
    if tags.get("request_id"):
        provenance["request_id"] = tags["request_id"]
    security: Dict[str, Any] = {"redaction": "bounded_secret_key_redaction_v1"}
    user_content_fields = [key for key in ("prompt", "text", "negative_prompt") if generation.get(key)]
    if user_content_fields:
        security.update(
            {
                "sensitivity": "user_content",
                "recorded_user_content_fields": user_content_fields,
                "prompt_storage": "descriptor_generation_bounded",
            }
        )
    descriptor: Dict[str, Any] = {
        "semantic_kind": semantic_kind,
        "render_kind": render_kind,
        "modality": modality or semantic_kind,
        "task": task,
        "classification_source": "producer",
        "session_id": tags.get("session_id") or None,
        "workflow_id": tags.get("workflow_id") or tags.get("workflow") or None,
        "node_id": tags.get("node_id") or tags.get("node") or None,
        "step_id": tags.get("step_id") or None,
        "effect_id": tags.get("effect_id") or tags.get("effect_idempotency_key") or None,
        "turn_id": tags.get("turn_id") or tags.get("turn") or None,
        "ledger_cursor": tags.get("ledger_cursor") or tags.get("step_cursor") or None,
        "parent_run_id": tags.get("parent_run_id") or None,
        "actor_id": tags.get("actor_id") or None,
        "producer": producer,
        "provenance": provenance,
        "generation": generation,
        "source_refs": source_refs,
        "security": security,
    }

    item_metadata = _field(item, "metadata", {}) or {}
    metadata = {
        "schema": "abstractruntime.generated_media_metadata.v1",
        "producer": producer,
        "generation": generation,
        "source_refs": source_refs,
        "security": security,
        "capability_metadata": _safe_artifact_value(item_metadata),
    }
    return descriptor, metadata


def _update_generated_artifact_metadata(
    *,
    artifact_store: Optional[Any],
    artifact_ref: Dict[str, Any],
    tags: Dict[str, str],
    metadata: Dict[str, Any],
    descriptor: Dict[str, Any],
) -> None:
    if artifact_store is None:
        return
    artifact_id = str(artifact_ref.get("artifact_id") or artifact_ref.get("$artifact") or "").strip()
    if not artifact_id:
        return
    update = getattr(artifact_store, "update_metadata", None)
    if not callable(update):
        return
    try:
        update(artifact_id, tags=tags, metadata=metadata, descriptor=descriptor)
    except Exception:
        return


def _store_generated_bytes(
    data: bytes,
    *,
    artifact_store: Optional[Any],
    run_id: Optional[str],
    content_type: str,
    tags: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    descriptor: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    if artifact_store is None:
        return None
    store = getattr(artifact_store, "store", None)
    if not callable(store):
        return None
    string_tags = _string_tags(tags)
    artifact_id: Optional[str] = None
    step_id = string_tags.get("step_id")
    if isinstance(step_id, str) and step_id.strip():
        try:
            from ...storage.artifacts import compute_artifact_id

            scope = str(run_id).strip() if isinstance(run_id, str) and run_id.strip() else "generated-media"
            artifact_id = compute_artifact_id(bytes(data), run_id=f"{scope}:step:{step_id.strip()}")
        except Exception:
            artifact_id = None
    meta = store(
        bytes(data),
        content_type=str(content_type or "application/octet-stream"),
        run_id=str(run_id).strip() if isinstance(run_id, str) and run_id.strip() else None,
        tags=string_tags,
        metadata=metadata,
        descriptor=descriptor,
        artifact_id=artifact_id,
    )
    artifact_id = getattr(meta, "artifact_id", None)
    if not isinstance(artifact_id, str) or not artifact_id.strip():
        return None
    size_bytes = getattr(meta, "size_bytes", None)
    try:
        size_i = int(size_bytes) if size_bytes is not None else len(data)
    except Exception:
        size_i = len(data)
    return {
        "$artifact": artifact_id.strip(),
        "artifact_id": artifact_id.strip(),
        "content_type": str(content_type or "application/octet-stream"),
        "size_bytes": size_i,
    }


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _normalize_generation_issue(issue: Any) -> Dict[str, Any]:
    return {
        "modality": str(_field(issue, "modality", "") or ""),
        "task": str(_field(issue, "task", "") or ""),
        "message": str(_field(issue, "message", "") or ""),
        "type": str(_field(issue, "type", "error") or "error"),
        "metadata": _jsonable(_field(issue, "metadata", {}) or {}),
    }


def _normalize_generated_resource(resource: Any) -> Dict[str, Any]:
    artifact_ref = _artifact_ref_payload(_field(resource, "artifact_ref", None))
    out: Dict[str, Any] = {
        "modality": str(_field(resource, "modality", "") or ""),
        "task": str(_field(resource, "task", "") or ""),
        "resource_type": str(_field(resource, "resource_type", "") or ""),
        "resource_id": str(_field(resource, "resource_id", "") or ""),
        "name": _field(resource, "name", None),
        "backend_id": _field(resource, "backend_id", None),
        "provider": _field(resource, "provider", None),
        "model": _field(resource, "model", None),
        "artifact_ref": artifact_ref,
        "metadata": _jsonable(_field(resource, "metadata", {}) or {}),
    }
    if artifact_ref is not None:
        out["artifact_id"] = artifact_ref.get("artifact_id")
    return {k: v for k, v in out.items() if v is not None}


def _normalize_generated_item(
    item: Any,
    *,
    artifact_store: Optional[Any],
    run_id: Optional[str],
    default_tags: Optional[Dict[str, Any]],
    generation_context: Optional[Dict[str, Any]] = None,
    item_index: int = 0,
    fallback_modality: Optional[str] = None,
) -> Dict[str, Any]:
    modality = str(_field(item, "modality", fallback_modality or "") or fallback_modality or "").strip().lower()
    task = str(_field(item, "task", "") or "").strip().lower()
    fmt = _field(item, "format", None)
    content_type = _field(item, "content_type", None)
    if not isinstance(content_type, str) or not content_type.strip():
        if modality == "image":
            content_type = f"image/{str(fmt or 'png').strip().lower() or 'png'}"
        elif modality == "video":
            content_type = f"video/{str(fmt or 'mp4').strip().lower() or 'mp4'}"
        elif modality in {"voice", "audio", "music", "sound"}:
            content_type = f"audio/{str(fmt or 'wav').strip().lower() or 'wav'}"
        else:
            content_type = "application/octet-stream"
    content_type = str(content_type).strip() or "application/octet-stream"

    data = _field(item, "data", None)
    data_len: Optional[int] = None
    if isinstance(data, (bytes, bytearray)):
        data_len = len(data)

    artifact_ref = _artifact_ref_payload(
        _field(item, "artifact_ref", None),
        content_type=content_type,
        size_bytes=data_len,
    )

    tags = _string_tags(default_tags)
    tags.update({"kind": "generated_media"})
    if modality:
        tags["modality"] = modality
    if task:
        tags["task"] = task
    descriptor, artifact_metadata = _generated_artifact_descriptor_and_metadata(
        item=item,
        content_type=content_type,
        tags=tags,
        generation_context=generation_context,
        item_index=item_index,
    )

    if artifact_ref is None and isinstance(data, (bytes, bytearray)):
        tags.update({"kind": "generated_media"})
        if modality:
            tags["modality"] = modality
        if task:
            tags["task"] = task
        stored_ref = _store_generated_bytes(
            bytes(data),
            artifact_store=artifact_store,
            run_id=run_id,
            content_type=content_type,
            tags=tags,
            metadata=artifact_metadata,
            descriptor=descriptor,
        )
        artifact_ref = stored_ref
    elif artifact_ref is not None:
        _update_generated_artifact_metadata(
            artifact_store=artifact_store,
            artifact_ref=artifact_ref,
            tags=tags,
            metadata=artifact_metadata,
            descriptor=descriptor,
        )

    out: Dict[str, Any] = {
        "modality": modality,
        "task": task,
        "content_type": content_type,
        "format": fmt,
        "backend_id": _field(item, "backend_id", None),
        "provider": _field(item, "provider", None),
        "model": _field(item, "model", None),
        "artifact_ref": artifact_ref,
        "metadata": _jsonable(_field(item, "metadata", {}) or {}),
    }
    if artifact_ref is not None:
        out["artifact_id"] = artifact_ref.get("artifact_id")
        out["size_bytes"] = artifact_ref.get("size_bytes")
    elif isinstance(data, (bytes, bytearray)):
        raise ValueError("Generated binary media requires an ArtifactStore that can persist artifacts.")
    elif data is not None:
        out["data"] = _jsonable(data)
    return {k: v for k, v in out.items() if v is not None}


def _first_media_identity(
    *,
    outputs: Dict[str, List[Dict[str, Any]]],
    resources: Dict[str, List[Dict[str, Any]]],
) -> tuple[Optional[str], Optional[str]]:
    for bucket in (outputs, resources):
        for items in bucket.values():
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                provider = item.get("provider")
                model = item.get("model")
                provider_s = str(provider).strip() if isinstance(provider, str) and provider.strip() else None
                model_s = str(model).strip() if isinstance(model, str) and model.strip() else None
                if provider_s is not None or model_s is not None:
                    return provider_s, model_s
    return None, None


def _normalize_multimodal_response(
    resp: Any,
    *,
    artifact_store: Optional[Any] = None,
    run_id: Optional[str] = None,
    default_tags: Optional[Dict[str, Any]] = None,
    generation_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    text_resp = _field(resp, "text", None)
    text = _normalize_local_response(text_resp) if text_resp is not None else None

    outputs_raw = _field(resp, "outputs", {}) or {}
    outputs: Dict[str, List[Dict[str, Any]]] = {}
    if isinstance(outputs_raw, dict):
        for modality, items in outputs_raw.items():
            if not isinstance(items, list):
                continue
            normalized_items = [
                _normalize_generated_item(
                    item,
                    artifact_store=artifact_store,
                    run_id=run_id,
                    default_tags=default_tags,
                    generation_context=generation_context,
                    item_index=idx,
                    fallback_modality=str(modality),
                )
                for idx, item in enumerate(items)
            ]
            if normalized_items:
                outputs[str(modality)] = normalized_items

    resources_raw = _field(resp, "resources", {}) or {}
    resources: Dict[str, List[Dict[str, Any]]] = {}
    if isinstance(resources_raw, dict):
        for modality, items in resources_raw.items():
            if not isinstance(items, list):
                continue
            normalized_resources = [_normalize_generated_resource(item) for item in items]
            if normalized_resources:
                resources[str(modality)] = normalized_resources

    warnings = [_normalize_generation_issue(x) for x in (_field(resp, "warnings", []) or []) if x is not None]
    errors = [_normalize_generation_issue(x) for x in (_field(resp, "errors", []) or []) if x is not None]
    metadata = _jsonable(_field(resp, "metadata", {}) or {})
    if not isinstance(metadata, dict):
        metadata = {"value": metadata}

    media_provider, media_model = _first_media_identity(outputs=outputs, resources=resources)
    legacy_runtime_provider = None
    legacy_runtime_model = None
    if metadata.get("subprocess") is True or metadata.get("execution_mode") == "local_one_shot_subprocess":
        raw_provider = metadata.get("provider")
        raw_model = metadata.get("model")
        if isinstance(raw_provider, str) and raw_provider.strip():
            legacy_runtime_provider = raw_provider.strip()
        if isinstance(raw_model, str) and raw_model.strip():
            legacy_runtime_model = raw_model.strip()

    runtime_provider = metadata.get("runtime_provider")
    runtime_model = metadata.get("runtime_model")
    runtime_provider_s = str(runtime_provider).strip() if isinstance(runtime_provider, str) and runtime_provider.strip() else None
    runtime_model_s = str(runtime_model).strip() if isinstance(runtime_model, str) and runtime_model.strip() else None
    if runtime_provider_s is None:
        runtime_provider_s = legacy_runtime_provider
    if runtime_model_s is None:
        runtime_model_s = legacy_runtime_model

    has_media_outputs = any(bool(items) for items in outputs.values()) or any(bool(items) for items in resources.values())
    media_only = bool(has_media_outputs) and text is None

    text_provider = text.get("provider") if isinstance(text, dict) else None
    text_model = text.get("model") if isinstance(text, dict) else None
    top_provider = (
        (str(text_provider).strip() if isinstance(text_provider, str) and text_provider.strip() else None)
        or (media_provider if media_only else None)
    )
    top_model = (
        (str(text_model).strip() if isinstance(text_model, str) and text_model.strip() else None)
        or (media_model if media_only else None)
        or (
            str(metadata.get("model")).strip()
            if (not media_only and isinstance(metadata.get("model"), str) and str(metadata.get("model")).strip())
            else None
        )
    )

    content = text.get("content") if isinstance(text, dict) else _field(resp, "content", None)
    result: Dict[str, Any] = {
        "content": content,
        "reasoning": text.get("reasoning") if isinstance(text, dict) else None,
        "data": text.get("data") if isinstance(text, dict) else None,
        "text": text,
        "outputs": outputs,
        "resources": resources,
        "warnings": warnings,
        "errors": errors,
        "usage": text.get("usage") if isinstance(text, dict) else None,
        "provider": top_provider,
        "model": top_model,
        "runtime_provider": runtime_provider_s,
        "runtime_model": runtime_model_s,
        "media_provider": media_provider,
        "media_model": media_model,
        "finish_reason": text.get("finish_reason") if isinstance(text, dict) else None,
        "metadata": metadata,
        "trace_id": text.get("trace_id") if isinstance(text, dict) else None,
        "gen_time": text.get("gen_time") if isinstance(text, dict) else None,
    }
    return result


def _normalize_local_response(
    resp: Any,
    *,
    artifact_store: Optional[Any] = None,
    run_id: Optional[str] = None,
    default_tags: Optional[Dict[str, Any]] = None,
    generation_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Normalize an AbstractCore local `generate()` result into JSON."""

    def _extract_reasoning_from_openai_like(raw: Any) -> Optional[str]:
        """Best-effort extraction of model reasoning from OpenAI-style payloads.

        LM Studio and some providers store reasoning in `choices[].message.reasoning_content`
        while leaving `content` empty during tool-call turns.
        """

        def _from_message(msg: Any) -> Optional[str]:
            if not isinstance(msg, dict):
                return None
            for key in ("reasoning", "reasoning_content", "thinking", "thinking_content"):
                val = msg.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
            return None

        if isinstance(raw, dict):
            # OpenAI chat completion: choices[].message
            choices = raw.get("choices")
            if isinstance(choices, list):
                for c in choices:
                    if not isinstance(c, dict):
                        continue
                    r = _from_message(c.get("message"))
                    if r:
                        return r
                    # Streaming-style payloads may use `delta`.
                    r = _from_message(c.get("delta"))
                    if r:
                        return r

            # Some variants store a single message at the top level.
            r = _from_message(raw.get("message"))
            if r:
                return r

        return None

    # AbstractCore multimodal output response (`generate(..., output=...)`).
    if hasattr(resp, "outputs") or hasattr(resp, "resources"):
        return _normalize_multimodal_response(
            resp,
            artifact_store=artifact_store,
            run_id=run_id,
            default_tags=default_tags,
            generation_context=generation_context,
        )

    # Dict-like already
    if isinstance(resp, dict):
        out = _jsonable(resp)
        if isinstance(out, dict):
            meta = out.get("metadata")
            if isinstance(meta, dict) and "trace_id" in meta and "trace_id" not in out:
                out["trace_id"] = meta["trace_id"]
            # Some providers place reasoning under metadata (e.g. LM Studio gpt-oss).
            if "reasoning" not in out and isinstance(meta, dict) and isinstance(meta.get("reasoning"), str):
                out["reasoning"] = meta.get("reasoning")
            if (
                (not isinstance(out.get("reasoning"), str) or not str(out.get("reasoning") or "").strip())
                and isinstance(out.get("raw_response"), dict)
            ):
                extracted = _extract_reasoning_from_openai_like(out.get("raw_response"))
                if extracted:
                    out["reasoning"] = extracted
            if (not isinstance(out.get("reasoning"), str) or not str(out.get("reasoning") or "").strip()) and isinstance(out.get("raw"), dict):
                extracted = _extract_reasoning_from_openai_like(out.get("raw"))
                if extracted:
                    out["reasoning"] = extracted
            if (not isinstance(out.get("reasoning"), str) or not str(out.get("reasoning") or "").strip()) and isinstance(out.get("choices"), list):
                extracted = _extract_reasoning_from_openai_like(out)
                if extracted:
                    out["reasoning"] = extracted
        return out

    # Pydantic structured output
    if hasattr(resp, "model_dump") or hasattr(resp, "dict"):
        return {
            "content": None,
            "data": _jsonable(resp),
            "tool_calls": None,
            "usage": None,
            "model": None,
            "finish_reason": None,
            "metadata": None,
            "trace_id": None,
        }

    # AbstractCore GenerateResponse
    content = getattr(resp, "content", None)
    raw_response = getattr(resp, "raw_response", None)
    tool_calls = getattr(resp, "tool_calls", None)
    usage = getattr(resp, "usage", None)
    model = getattr(resp, "model", None)
    finish_reason = getattr(resp, "finish_reason", None)
    metadata = getattr(resp, "metadata", None)
    gen_time = getattr(resp, "gen_time", None)
    trace_id: Optional[str] = None
    reasoning: Optional[str] = None
    if isinstance(metadata, dict):
        raw = metadata.get("trace_id")
        if raw is not None:
            trace_id = str(raw)
        r = metadata.get("reasoning")
        if isinstance(r, str) and r.strip():
            reasoning = r.strip()
    if reasoning is None and raw_response is not None:
        extracted = _extract_reasoning_from_openai_like(_jsonable(raw_response))
        if extracted:
            reasoning = extracted

    return {
        "content": content,
        "reasoning": reasoning,
        "data": None,
        "raw_response": _jsonable(raw_response) if raw_response is not None else None,
        "tool_calls": _jsonable(tool_calls) if tool_calls is not None else None,
        "usage": _jsonable(usage) if usage is not None else None,
        "model": model,
        "finish_reason": finish_reason,
        "metadata": _jsonable(metadata) if metadata is not None else None,
        "trace_id": trace_id,
        "gen_time": float(gen_time) if isinstance(gen_time, (int, float)) else None,
    }


_THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>\s*", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_UNCLOSED_RE = re.compile(r"<think>(.*)\Z", re.DOTALL | re.IGNORECASE)


def _split_think_blocks(text: str) -> Tuple[str, Optional[str]]:
    """Split `<think>...</think>` transport markup out of assembled stream text.

    Streamed-vs-non-streamed parity (code seat c1017): on thinking models the
    NON-streamed path arrives think-free (the provider stack extracts reasoning
    into metadata), but raw stream deltas can carry the think text inline —
    assembling them verbatim put `<think>` blocks into `content`, and the react
    parse node behaved differently per arm (streamed runs re-nudged to
    max_iterations on tasks the non-streamed arm concluded in 3 calls).

    Returns (content_without_think, reasoning_or_None). An UNCLOSED trailing
    block (stream ended mid-thought) is extracted too — thought text is never
    left masquerading as the answer.
    """
    if "<think>" not in text.lower():
        return text, None
    parts: list[str] = []

    def _collect(m: "re.Match[str]") -> str:
        parts.append(m.group(1).strip())
        return ""

    out = _THINK_BLOCK_RE.sub(_collect, text)
    tail = _THINK_OPEN_UNCLOSED_RE.search(out)
    if tail:
        parts.append(tail.group(1).strip())
        out = out[: tail.start()]
    reasoning = "\n\n".join(p for p in parts if p) or None
    if reasoning is None:
        return text, None
    return out.strip(), reasoning


# Providers whose STREAMED lane does not report `metadata["prompt_cache"]`,
# which their non-streamed lane does (the prompt-cache diagnosis reads it first).
# Streaming is refused for these when a prompt-cache key is in play. Empty today:
# MLX was the one entry until AbstractCore 8d59974 ("streamed calls end with the
# sync lane's finish_reason, usage and prompt_cache"), which puts the same
# record on the stream's terminal chunk.
_STREAM_LANES_WITHOUT_PROMPT_CACHE_TELEMETRY: frozenset = frozenset()


def _mark_stream_unavailable(on_delta: Any, detail: str) -> None:
    """Tell the runtime's live emitter why this call does not stream (no-op without one)."""

    from ...core.live_deltas import LiveDeltaEmitter

    if isinstance(on_delta, LiveDeltaEmitter):
        on_delta.mark_unavailable(detail)


class _LiveThinkSplitter:
    """Route live content fragments to the `content` or `reasoning` channel.

    Some providers stream thinking inline as `<think>...</think>` markup inside
    content deltas (the aggregated record splits it after the fact with
    `_split_think_blocks`). Live deltas cannot wait for the end, so this splits
    incrementally: text inside a think block goes to `reasoning`, the rest to
    `content`, and a tag cut across two fragments (`"<th"` + `"ink>"`) is held
    back until it can be decided. The markup itself is never emitted.

    HARMONY (gpt-oss) transcripts are parsed first, with AbstractCore's rules
    (architectures/response_postprocessing.py `split_harmony_response_text`,
    providers/streaming.py `_collect_harmony_tool_content`): a segment is
    `<|channel|>NAME[ to=RECIPIENT]<|message|>BODY` closed by `<|end|>`,
    `<|call|>` or `<|return|>`, and `<|start|>ROLE` opens the next one.
    `final` -> content, `analysis` -> reasoning, a header with `to=` (a tool
    call, e.g. `commentary to=functions.read_file`) -> held back, `commentary`
    without a recipient (a preamble for the user) -> content; any other
    channel -> held back. Framing tokens are never emitted.

    `held_back` / `emitted_content` let the caller report a call whose whole
    answer was held back (`tool_envelope_holdback`), so it is never silent.
    """

    _HARMONY_TOKENS = ("<|channel|>", "<|message|>", "<|end|>", "<|start|>", "<|call|>", "<|return|>")
    _OPEN = "<think>"
    _CLOSE = "</think>"
    # Tool-call envelopes a local model may write as TEXT. AbstractCore's
    # UnifiedStreamProcessor already withholds them from content chunks
    # (providers/streaming.py, IncrementalToolDetector patterns) when the
    # model's format is known; this is the second line: once an opening
    # marker shows up in live content, no more content is sent for the call
    # (the final record replaces the live text anyway). Reasoning still flows.
    _TOOL_MARKERS = (
        "<tool_call>",
        "<|tool_call|>",
        "<|tool_call>",
        "<|tool_call_start|>",
        "<function_call>",
        "```tool_code",
    )

    def __init__(self, emit: Any) -> None:
        self._emit = emit
        self._inside = False
        self._hold = ""
        self._content_hold = ""
        self._tool_envelope_seen = False
        self.held_back = False
        self.emitted_content = False
        # Harmony stage.
        self._h_state = "text"  # text | header | role
        self._h_mode = "content"  # content | reasoning | held
        self._h_buf = ""
        self._h_header = ""

    # -- harmony stage -------------------------------------------------------
    def _h_route(self, text: str) -> None:
        if not text:
            return
        if self._h_state == "header":
            self._h_header += text
        elif self._h_state == "role":
            return
        elif self._h_mode == "content":
            self._think_feed(text)
        elif self._h_mode == "reasoning":
            self._emit(text, "reasoning")
        else:
            self.held_back = True

    def _h_open_body(self) -> None:
        header = self._h_header.strip()
        self._h_header = ""
        name = header.split("<|", 1)[0].split()[0].lower() if header.split("<|", 1)[0].split() else ""
        if "to=" in header:
            self._h_mode = "held"
            self.held_back = True
        elif name == "analysis":
            self._h_mode = "reasoning"
        elif name in ("final", "commentary"):
            self._h_mode = "content"
        else:
            self._h_mode = "held"
            self.held_back = True
        self._h_state = "text"

    def _harmony_feed(self, text: str) -> None:
        buf = self._h_buf + text
        self._h_buf = ""
        while buf:
            if self._h_state == "header":
                idx = buf.find("<|message|>")
                if idx < 0:
                    keep = self._partial_suffix(buf, "<|message|>")
                    self._h_header += buf[: len(buf) - keep] if keep else buf
                    self._h_buf = buf[len(buf) - keep:] if keep else ""
                    return
                self._h_header += buf[:idx]
                buf = buf[idx + len("<|message|>"):]
                self._h_open_body()
                continue
            hits = [(buf.find(t), t) for t in self._HARMONY_TOKENS]
            hits = [(i, t) for i, t in hits if i >= 0]
            if not hits:
                keep = max(self._partial_suffix(buf, t) for t in self._HARMONY_TOKENS)
                self._h_route(buf[: len(buf) - keep] if keep else buf)
                self._h_buf = buf[len(buf) - keep:] if keep else ""
                return
            idx, token = min(hits)
            self._h_route(buf[:idx])
            buf = buf[idx + len(token):]
            if token == "<|channel|>":
                self._h_state = "header"
                self._h_header = ""
            elif token == "<|message|>":
                # `<|start|>assistant<|message|>` (no channel): plain answer.
                self._h_state = "text"
                self._h_mode = "content"
            elif token == "<|start|>":
                self._h_state = "role"
            else:  # <|end|> / <|call|> / <|return|>: segment closed
                self._h_state = "text"
                self._h_mode = "content"

    def _emit_content(self, text: str, *, final: bool = False) -> None:
        """Send content unless a tool envelope began; hold a possible partial marker."""

        if self._tool_envelope_seen or not (text or (final and self._content_hold)):
            return
        buf = self._content_hold + text
        self._content_hold = ""
        lowered = buf.lower()
        cut = min((i for i in (lowered.find(m) for m in self._TOOL_MARKERS) if i >= 0), default=-1)
        if cut >= 0:
            self._tool_envelope_seen = True
            self.held_back = True
            if cut:
                self._emit(buf[:cut], "content")
                self.emitted_content = True
            return
        keep = 0 if final else max(self._partial_suffix(buf, m) for m in self._TOOL_MARKERS)
        ready = buf[: len(buf) - keep] if keep else buf
        if ready:
            self._emit(ready, "content")
            self.emitted_content = True
        self._content_hold = buf[len(buf) - keep:] if keep else ""

    @staticmethod
    def _partial_suffix(text: str, tag: str) -> int:
        lowered = text.lower()
        for n in range(min(len(tag) - 1, len(text)), 0, -1):
            if lowered.endswith(tag[:n]):
                return n
        return 0

    def feed(self, text: str) -> None:
        self._harmony_feed(text)

    def _think_feed(self, text: str) -> None:
        buf = self._hold + text
        self._hold = ""
        while buf:
            tag = self._CLOSE if self._inside else self._OPEN
            channel = "reasoning" if self._inside else "content"
            idx = buf.lower().find(tag)
            if idx >= 0:
                if idx:
                    self._send(buf[:idx], channel)
                buf = buf[idx + len(tag):]
                self._inside = not self._inside
                continue
            keep = self._partial_suffix(buf, tag)
            ready = buf[: len(buf) - keep] if keep else buf
            if ready:
                self._send(ready, channel)
            self._hold = buf[len(buf) - keep:] if keep else ""
            return

    def _send(self, text: str, channel: str) -> None:
        if channel == "content":
            self._emit_content(text)
        else:
            self._emit(text, channel)

    def finish(self) -> None:
        if self._h_buf:
            pending, self._h_buf = self._h_buf, ""
            if self._h_state != "header":
                self._h_route(pending)
        if self._hold:
            self._send(self._hold, "reasoning" if self._inside else "content")
            self._hold = ""
        if self._content_hold:
            self._emit_content("", final=True)


def _normalize_local_streaming_response(
    stream: Any,
    on_token: Optional[Any] = None,
    on_delta: Optional[Any] = None,
) -> Dict[str, Any]:
    """Consume an AbstractCore streaming `generate(..., stream=True)` iterator into a single JSON result.

    AbstractRuntime currently persists a single effect outcome object per LLM call, so even when
    the underlying provider streams we aggregate into one final dict and surface timing fields.

    `on_token` (code seat c990, in-process streaming surface): an optional
    `on_token(delta: str, meta: dict)` callback fired per content chunk,
    BEST-EFFORT and never load-bearing — the durable result is byte-identical
    with or without it, and a raising callback is disabled for the rest of the
    stream (one warning), never failing the call. Same-process hosts (the
    abstractcode CLI) use it for live tail rendering; gateway-hosted surfaces
    need the durable plane instead (deliberately not built here).

    `on_delta` (token streaming, 2026-09-26): the PER-CALL live delta
    callback `on_delta(text: str, channel: str)` the runtime passes as
    `params["_on_delta"]` when a run streams to a host sink. Channel
    `"content"` carries answer text; `"reasoning"` carries thinking — both the
    provider's `metadata["reasoning_delta"]` fragments and inline `<think>`
    markup split out of content (`_LiveThinkSplitter`). Same failure contract
    as `on_token`: best-effort, a raising callback is disabled for the rest of
    the stream with one warning, and the aggregated result is unchanged.
    Unlike `on_token` it belongs to ONE call, so concurrent runs sharing a
    client never see each other's text.
    """
    import time

    start_perf = time.perf_counter()

    token_cb = on_token if callable(on_token) else None
    delta_cb = on_delta if callable(on_delta) else None

    def _fire_delta(text: str, channel: str) -> None:
        nonlocal delta_cb
        if delta_cb is None or not text:
            return
        try:
            delta_cb(text, channel)
        except Exception as e:
            logger.warning(f"live delta callback raised; disabled for this stream: {e}")
            delta_cb = None

    think_splitter = _LiveThinkSplitter(_fire_delta)

    def _fire_content_delta(text: str) -> None:
        if delta_cb is not None and text:
            think_splitter.feed(text)

    def _fire_reasoning_delta(meta: Any) -> None:
        if delta_cb is None or not isinstance(meta, dict):
            return
        # Only the per-chunk DELTA key: `metadata["reasoning"]` on the trailing
        # chunk is the complete aggregate and would repeat everything.
        r_delta = meta.get("reasoning_delta")
        if isinstance(r_delta, str) and r_delta:
            _fire_delta(r_delta, "reasoning")

    def _fire_token(delta: str, model_name: Optional[str], fr: Any) -> None:
        nonlocal token_cb
        if token_cb is None or not delta:
            return
        try:
            token_cb(delta, {"model": model_name, "finish_reason": fr})
        except Exception as e:
            logger.warning(f"on_token callback raised; disabled for this stream: {e}")
            token_cb = None

    chunks: list[str] = []
    tool_calls: Any = None
    tool_call_keys: set = set()
    usage: Any = None
    model: Optional[str] = None
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = {}
    trace_id: Optional[str] = None
    reasoning: Optional[str] = None
    ttft_ms: Optional[float] = None
    # Parity with the non-streamed record: providers attach the raw wire
    # payload per chunk; the LAST one (the terminal chunk: finish reason,
    # usage) is kept, so `raw_response` exists in both modes. It is that
    # terminal chunk, not a reassembled full response.
    raw_response: Any = None

    def _fold_tool_calls(tc: Any) -> None:
        """Accumulate streamed tool calls across chunks (c1017 parity).

        The non-streamed response carries the COMPLETE tool-call list; stream
        processors may emit calls on separate chunks (or re-send the full list
        per chunk). Last-non-None-wins silently DROPPED earlier calls in the
        incremental case — accumulate with id-dedup instead, which is identical
        for re-sent full lists and lossless for incremental ones."""
        nonlocal tool_calls
        if tc is None:
            return
        if not isinstance(tc, list):
            tool_calls = tc
            return
        if not isinstance(tool_calls, list):
            tool_calls = []
        for call in tc:
            if isinstance(call, dict):
                key = str(call.get("id") or call.get("call_id") or "") or json.dumps(
                    _jsonable(call), sort_keys=True, default=str
                )
            else:
                key = str(call)
            if key in tool_call_keys:
                continue
            tool_call_keys.add(key)
            tool_calls.append(call)

    def _maybe_capture_ttft(*, content: Any, tool_calls_value: Any, meta: Any) -> None:
        nonlocal ttft_ms
        if ttft_ms is not None:
            return

        if isinstance(meta, dict):
            timing = meta.get("_timing") if isinstance(meta.get("_timing"), dict) else None
            if isinstance(timing, dict) and isinstance(timing.get("ttft_ms"), (int, float)):
                ttft_ms = float(timing["ttft_ms"])
                return

        has_content = isinstance(content, str) and bool(content)
        has_tools = isinstance(tool_calls_value, list) and bool(tool_calls_value)
        if has_content or has_tools:
            ttft_ms = round((time.perf_counter() - start_perf) * 1000, 1)

    for chunk in stream:
        if chunk is None:
            continue

        if isinstance(chunk, dict):
            _fire_reasoning_delta(chunk.get("metadata"))
            content = chunk.get("content")
            if isinstance(content, str) and content:
                chunks.append(content)
                _fire_token(content, model, chunk.get("finish_reason"))
                _fire_content_delta(content)

            rr = chunk.get("raw_response")
            if rr is not None:
                raw_response = rr

            tc = chunk.get("tool_calls")
            _fold_tool_calls(tc)

            u = chunk.get("usage")
            if u is not None:
                usage = u

            m = chunk.get("model")
            if model is None and isinstance(m, str) and m.strip():
                model = m.strip()

            fr = chunk.get("finish_reason")
            if fr is not None:
                finish_reason = str(fr)

            meta = chunk.get("metadata")
            _maybe_capture_ttft(content=content, tool_calls_value=tc, meta=meta)

            if isinstance(meta, dict):
                meta_json = _jsonable(meta)
                if isinstance(meta_json, dict):
                    metadata.update(meta_json)
                    raw_trace = meta_json.get("trace_id")
                    if trace_id is None and raw_trace is not None:
                        trace_id = str(raw_trace)
                    r = meta_json.get("reasoning")
                    if isinstance(r, str) and r.strip():
                        # LAST non-empty wins (core contract v1, c5769): the
                        # trailing chunk carries the guaranteed complete
                        # aggregate; first-non-empty persisted ONE FRAGMENT.
                        # Display fragments ride `reasoning_delta`, never
                        # read here.
                        reasoning = r.strip()
            continue

        _fire_reasoning_delta(getattr(chunk, "metadata", None))
        content = getattr(chunk, "content", None)
        if isinstance(content, str) and content:
            chunks.append(content)
            _fire_token(content, model, getattr(chunk, "finish_reason", None))
            _fire_content_delta(content)

        rr = getattr(chunk, "raw_response", None)
        if rr is not None:
            raw_response = rr

        tc = getattr(chunk, "tool_calls", None)
        _fold_tool_calls(tc)

        u = getattr(chunk, "usage", None)
        if u is not None:
            usage = u

        m = getattr(chunk, "model", None)
        if model is None and isinstance(m, str) and m.strip():
            model = m.strip()

        fr = getattr(chunk, "finish_reason", None)
        if fr is not None:
            finish_reason = str(fr)

        meta = getattr(chunk, "metadata", None)
        _maybe_capture_ttft(content=content, tool_calls_value=tc, meta=meta)

        if isinstance(meta, dict):
            meta_json = _jsonable(meta)
            if isinstance(meta_json, dict):
                metadata.update(meta_json)
                raw_trace = meta_json.get("trace_id")
                if trace_id is None and raw_trace is not None:
                    trace_id = str(raw_trace)
                r = meta_json.get("reasoning")
                if isinstance(r, str) and r.strip():
                    # LAST non-empty wins (core contract v1, reasoning-1st-
                    # citizen plan 2026-07-26): streamed metadata.reasoning
                    # carries per-chunk snapshots and core GUARANTEES the
                    # trailing chunk is the complete aggregate — first-non-
                    # empty persisted ONE FRAGMENT and silently violated the
                    # operator's keep ruling (core audit, c5769). Per-chunk
                    # display fragments ride `reasoning_delta`, a key this
                    # fold deliberately never reads.
                    reasoning = r.strip()

    if delta_cb is not None:
        think_splitter.finish()
        if think_splitter.held_back and not think_splitter.emitted_content:
            # The whole answer was held back (a tool call, or a channel we do not
            # show): say so rather than leave the live view silent.
            _mark_stream_unavailable(on_delta, "tool_envelope_holdback")

    # `reasoning_delta` is a per-chunk DISPLAY fragment; the merged metadata
    # kept the LAST fragment, which the non-streamed record never has
    # (parity). The complete thought is `reasoning` above.
    metadata.pop("reasoning_delta", None)

    gen_time = round((time.perf_counter() - start_perf) * 1000, 1)

    # Parity with the non-streamed shape (c1017): providers strip `<think>`
    # markup and surface it as reasoning on the non-streamed path; assembled
    # deltas must not differ. Provider-reported reasoning (metadata) wins;
    # the split only fills the gap when deltas carried the markup inline.
    content = "".join(chunks)
    content, think_reasoning = _split_think_blocks(content)
    if reasoning is None and think_reasoning:
        reasoning = think_reasoning

    return {
        "content": content,
        "reasoning": reasoning,
        "data": None,
        "raw_response": _jsonable(raw_response) if raw_response is not None else None,
        "tool_calls": _jsonable(tool_calls) if tool_calls is not None else None,
        "usage": _jsonable(usage) if usage is not None else None,
        "model": model,
        "finish_reason": finish_reason,
        "metadata": metadata or None,
        "trace_id": trace_id,
        "gen_time": gen_time,
        "ttft_ms": ttft_ms,
    }


def _attach_provider_endpoint_profile_resolver_to_client(client: Any, resolver: Any) -> None:
    """Attach a Gateway endpoint-profile resolver to a Runtime/Core client pair."""
    try:
        setattr(client, "resolve_provider_endpoint_profile", resolver)
    except Exception:
        pass
    llm = getattr(client, "_llm", None)
    if llm is not None:
        try:
            setattr(llm, "resolve_provider_endpoint_profile", resolver)
        except Exception:
            pass


def _normalize_core_capability_defaults(value: Any) -> Dict[str, Dict[str, Any]]:
    """Normalize a Core/Gateway capability-default payload to route-keyed rows."""
    if not value:
        return {}
    rows: Any
    if isinstance(value, dict) and isinstance(value.get("routes"), list):
        rows = value.get("routes")
    elif isinstance(value, dict):
        rows = []
        for key, row in value.items():
            if isinstance(row, dict):
                item = dict(row)
                item.setdefault("key", str(key))
                rows.append(item)
    else:
        rows = value
    if not isinstance(rows, list):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = str(row.get("key") or "").strip().lower()
        if not key and row.get("kind") and row.get("modality"):
            key = f"{str(row.get('kind')).strip().lower()}.{str(row.get('modality')).strip().lower()}"
        if "." not in key:
            continue
        out[key] = dict(row)
    return out


def _coerce_core_config_file(path: Optional[str | Path]) -> Optional[str]:
    if path is None:
        return None
    try:
        text = str(Path(path).expanduser())
    except Exception:
        text = str(path)
    text = text.strip()
    return text or None


# THE ROUTE THAT ANSWERS "what model does this host use for text". Named here
# so the refusal below can say it out loud; AbstractCore canonicalizes it to
# the storage key `input.text`.
TEXT_CAPABILITY_ROUTE_KEY = "output.text"


class NoDefaultProviderConfigured(ValueError):
    """No text default is configured and the call named no provider.

    A CONFIGURATION refusal, not a provider failure: it fails identically on
    every attempt, so `_llm_error_is_retryable` classifies it non-retryable and
    the message is the whole UX. It names the exact commands that fix it --
    an error a new user can act on without leaving the terminal.
    """


def no_default_provider_configured_error(
    *,
    core_config_file: Optional[str] = None,
    what: str = "text generation",
) -> NoDefaultProviderConfigured:
    """Build the fresh-install refusal, naming every way out of it.

    THE ERROR IS THE UX. A new user meets this before they meet any document,
    so it states the route by name, the AbstractCore CLI command, the Gateway
    route, the per-call pin, and the store that was actually consulted.
    """

    lines = [
        f"no provider/model is configured for {what} and this call named none.",
        "Fix it with either entry point:",
        "  - AbstractCore CLI:  abstractcore config set-default "
        f"{TEXT_CAPABILITY_ROUTE_KEY} --provider <provider> --model <model>",
        "  - Gateway:           PUT /api/gateway/config/capability-defaults/output/text "
        '{"provider": "<provider>", "model": "<model>"}  (console: Capability defaults)',
        "Or pin this call: provider/model on the node, or "
        "input_data._runtime.provider / _runtime.model on the run.",
        "Inspect what is set with: abstractcore config defaults",
    ]
    # Name the scoped store ONLY when it exists. A host may hand down a scoped
    # path that was never created (the AbstractCore manager then resolves its
    # own store), and printing that path as "the store" would send the operator
    # to edit a file nobody reads -- a worse dead end than saying nothing.
    if core_config_file:
        try:
            if Path(core_config_file).is_file():
                lines.append(f"Store consulted: {core_config_file}")
        except Exception:
            pass
    return NoDefaultProviderConfigured("\n".join(lines))


class DefaultRouteProviderError(ValueError):
    """A client for the host's CONFIGURED DEFAULT provider/model failed to build.

    The bare provider error ("Unknown provider: notaprovider") is true and
    useless: it never says that this provider is not something the caller
    typed, it is what the OPERATOR configured as the host default, nor where
    that setting lives. Wrapping it turns a mystery into an edit.
    """


def _missing_weights_hint(provider: str, model: str) -> str:
    """`abstractcore models download <provider> <artifact>`, when that helps.

    Returns "" unless the weights really are the problem: a provider AbstractCore
    can fetch for, and a probe that says the model is not on this machine. On a
    relay provider, or when the weights ARE present (so the failure is something
    else entirely), a download instruction would be noise pointing the operator
    away from the real cause.

    Best-effort by construction -- this runs while building an error message,
    so a probe that is slow, broken or unavailable simply adds no line.
    """

    try:
        from .config_facade import (
            probe_model_presence,
            recommended_model_downloads,
            split_model_artifact,
        )

        # The route stores the SERVED id; the recommendation names the exact
        # weights, quantization included. Fetching the served id would ask
        # LM Studio for whatever quant it prefers, which is not what the
        # execution host was configured with.
        artifact = model
        for item in recommended_model_downloads():
            if str(item.get("provider", "")).strip().lower() == str(provider).strip().lower():
                candidate = str(item.get("artifact") or "").strip()
                base, _quant = split_model_artifact(candidate)
                if base.lower() == str(model).strip().lower():
                    artifact = candidate
                    break
        presence = probe_model_presence(provider, artifact)
        if presence.get("status") != "absent" or not presence.get("downloadable"):
            return ""
        return f"abstractcore models download {provider} {artifact}"
    except Exception:
        return ""


def default_route_provider_error(
    exc: Exception,
    *,
    provider: str,
    model: str,
    core_config_file: Optional[str] = None,
) -> DefaultRouteProviderError:
    lines = [
        str(exc),
        "",
        f"This provider/model ({provider}/{model}) is the execution host's configured "
        f"default for text generation (capability route {TEXT_CAPABILITY_ROUTE_KEY}), "
        "not a value this call supplied. Change it with:",
        f"  abstractcore config set-default {TEXT_CAPABILITY_ROUTE_KEY} "
        "--provider <provider> --model <model>",
        '  PUT /api/gateway/config/capability-defaults/output/text {"provider": "...", "model": "..."}',
    ]
    # CHANGING THE DEFAULT IS THE WRONG ADVICE WHEN THE DEFAULT IS RIGHT. On a
    # fresh install this pair is the RECOMMENDED default and the only thing
    # wrong with it is that nobody has fetched the weights yet -- so name the
    # command that fetches them, and the exact artifact, which is not the
    # served id whenever a quantization is pinned.
    download_line = _missing_weights_hint(provider, model)
    if download_line:
        lines.append("Or download the weights this default needs:")
        lines.append(f"  {download_line}")
    if core_config_file:
        try:
            if Path(core_config_file).is_file():
                lines.append(f"Store: {core_config_file}")
        except Exception:
            pass
    wrapped = DefaultRouteProviderError("\n".join(lines))
    wrapped.__cause__ = exc
    return wrapped


def _attach_core_execution_context_to_client(
    client: Any,
    *,
    core_config_file: Optional[str] = None,
    capability_defaults: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    """Attach scoped Core execution config to a Runtime/Core client pair.

    Gateway hosts may run many principals in one Python process. This deliberately
    avoids mutating AbstractCore's process-global config singleton or environment.
    """
    targets = [client]
    llm = getattr(client, "_llm", None)
    if llm is not None:
        targets.append(llm)
    for target in targets:
        if core_config_file:
            try:
                setattr(target, "_abstractcore_config_file", core_config_file)
            except Exception:
                pass
        if capability_defaults is not None:
            try:
                setattr(target, "_abstractcore_capability_defaults", deepcopy(capability_defaults))
            except Exception:
                pass


class LocalAbstractCoreLLMClient:
    """In-process LLM client using AbstractCore's provider stack."""

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        artifact_store: Optional[Any] = None,
        bloc_root_dir: Optional[str | Path] = None,
        prompt_cache_export_root_dir: Optional[str | Path] = None,
        core_config_file: Optional[str | Path] = None,
        capability_defaults: Optional[Any] = None,
    ):
        # In this monorepo layout, `import abstractcore` can resolve to a namespace package
        # (the outer project directory) when running from the repo root. In that case, the
        # top-level re-export `from abstractcore import create_llm` is unavailable even though
        # the actual module tree (e.g. `abstractcore.core.factory`) is importable.
        #
        # Prefer the canonical public import, but fall back to the concrete module path so
        # in-repo tooling/tests don't depend on editable-install import ordering.
        try:
            from abstractcore import create_llm  # type: ignore
        except Exception:  # pragma: no cover
            from abstractcore.core.factory import create_llm  # type: ignore
        from abstractcore.tools.handler import UniversalToolHandler

        self._provider = provider
        self._model = model
        self._artifact_store = artifact_store
        self._bloc_root_dir = _coerce_bloc_root_dir(bloc_root_dir)
        self._prompt_cache_export_root_dir = _coerce_prompt_cache_export_root_dir(prompt_cache_export_root_dir)
        self._core_config_file = _coerce_core_config_file(core_config_file)
        self._capability_defaults = _normalize_core_capability_defaults(capability_defaults)
        self._generate_lock = _local_generate_lock(provider=self._provider, model=self._model)
        kwargs = dict(llm_kwargs or {})
        # Native model loading may need the scoped default (e.g. whether to
        # prepare its MTP head). Attaching this only after create_llm is too late.
        if self._core_config_file:
            kwargs["_abstractcore_config_file"] = self._core_config_file
        if capability_defaults is not None:
            kwargs["_abstractcore_capability_defaults"] = deepcopy(self._capability_defaults)
        kwargs.setdefault("enable_tracing", True)
        if kwargs.get("enable_tracing"):
            # Keep a small in-memory ring buffer for exact request/response observability.
            # This enables hosts (AbstractCode/AbstractFlow) to inspect trace payloads by trace_id.
            kwargs.setdefault("max_traces", 50)
        self._llm_kwargs = dict(kwargs)
        self._llm = create_llm(provider, model=model, **kwargs)
        _attach_core_execution_context_to_client(
            self,
            core_config_file=self._core_config_file,
            capability_defaults=self._capability_defaults if capability_defaults is not None else None,
        )
        self._tool_handler = UniversalToolHandler(model)
        self._prompt_cache_state_lock = threading.Lock()
        self._prompt_cache_state: Dict[str, _PromptCacheSessionState] = {}
        self._capability_residency_core = None
        self._capability_residency_core_lock = threading.Lock()
        self._provider_endpoint_profile_resolver = None
        self._locked_model_residency: set = set()
        self._on_token: Optional[Any] = None

    def _stream_parity_refusal(self, params: Dict[str, Any]) -> Optional[str]:
        """Why a streamed call on this provider would record less than a non-streamed one, or None."""

        if getattr(self._llm, "_stream_options_unsupported", False):
            # abstractcore openai_compatible_provider: the server rejected
            # `stream_options`, so streamed calls carry no usage.
            return "usage_unavailable"
        key = params.get("prompt_cache_key")
        if (
            str(self._provider or "").strip().lower() in _STREAM_LANES_WITHOUT_PROMPT_CACHE_TELEMETRY
            and isinstance(key, str)
            and key.strip()
        ):
            return "prompt_cache_unavailable"
        return None

    def set_on_token(self, callback: Optional[Any]) -> None:
        """Register an in-process token callback (code seat c990's streaming ask).

        `callback(delta: str, meta: dict)` fires per content chunk on
        `stream=True` calls. BEST-EFFORT and never load-bearing: the durable
        result is byte-identical with or without it; a raising callback is
        disabled for the remainder of that stream with one warning. Host-side
        registration only — callbacks never ride effect payloads (they are not
        durable data). Pass None to unregister."""
        self._on_token = callback if callable(callback) else None

    def default_prompt_cache_identity(self) -> Tuple[Optional[str], Optional[str]]:
        return self._provider, self._model

    def stream_tts(
        self,
        *,
        text: str,
        output: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        spec = {"modality": "voice", "task": "tts"}
        if isinstance(output, dict):
            spec.update(output)
        # STREAM-LANE capability-defaults merge (continuum dm 2026-07-22,
        # operator incident dm#133): the non-stream lane merges via
        # resolve_generate_route; this lane never traversed it, so a BARE
        # spec fell to the voice plugin's env-or-openai default instead of
        # the operator's configured output.voice engine. Explicit
        # provider/model/base_url still win (the helper is a no-op then).
        spec = _with_capability_default_route(spec, getattr(self, "_capability_defaults", None))
        fmt = str(spec.get("format") or spec.get("response_format") or "wav").strip().lower() or "wav"
        if fmt == "wave":
            fmt = "wav"
        if fmt != "wav":
            raise ValueError("Local TTS streaming currently supports wav only.")
        voice_facade = getattr(self._llm, "voice", None)
        method = getattr(voice_facade, "tts_stream", None)
        if not callable(method):
            method = getattr(self._llm, "tts_stream", None)
        if not callable(method):
            raise ValueError("Local AbstractCore client does not expose streaming TTS.")

        stream_params = dict(params or {})
        run_id, tags = _trace_run_id_and_tags_from_params(
            stream_params,
            task="tts",
            modality="voice",
            model=str(spec.get("model") or "") or None,
        )
        stream = method(
            str(text or ""),
            voice=spec.get("voice") or spec.get("voice_id"),
            format="wav",
            profile=spec.get("profile"),
            speed=spec.get("speed"),
            instructions=spec.get("instructions"),
            quality_preset=spec.get("quality_preset") or spec.get("quality"),
            provider=spec.get("provider"),
            model=spec.get("model"),
            cancel_event=stream_params.get("cancel_event"),
        )
        segments: List[bytes] = []
        terminal_seen = False
        for event in stream:
            payload = dict(event) if isinstance(event, dict) else {"type": "event", "value": _jsonable(event)}
            audio = payload.pop("audio", None)
            if isinstance(audio, (bytes, bytearray)):
                audio_bytes = bytes(audio)
                segments.append(audio_bytes)
                payload["audio_b64"] = base64.b64encode(audio_bytes).decode("ascii")
                payload.setdefault("size_bytes", len(audio_bytes))
            elif payload.get("type") == "audio":
                audio_bytes = _decode_stream_audio_b64(payload)
                if audio_bytes:
                    segments.append(audio_bytes)
            if payload.get("type") in {"done", "cancelled"}:
                terminal_seen = True
                if payload.get("type") == "done" and payload.get("ok") is not False:
                    artifact = _finalize_tts_stream_artifact(
                        segments=segments,
                        artifact_store=self._artifact_store,
                        run_id=run_id,
                        tags=tags,
                        text=text,
                        spec=spec,
                        provider=str(spec.get("provider") or payload.get("provider") or "abstractcore-local"),
                        model=str(spec.get("model") or payload.get("model") or "") or None,
                        metadata={
                            "stream": {
                                "chunks": len(segments),
                                "transport": "in-process",
                                "chunk_format": "wav-segment",
                            },
                            "terminal_event": _jsonable(payload),
                        },
                    )
                    payload = dict(payload)
                    payload["audio_artifact"] = artifact
                yield payload
                continue
            yield payload
        if not terminal_seen:
            yield {
                "type": "error",
                "ok": False,
                "error": "TTS stream ended without a terminal done/cancelled event.",
            }

    def set_provider_endpoint_profile_resolver(self, resolver: Any) -> None:
        self._provider_endpoint_profile_resolver = resolver
        _attach_provider_endpoint_profile_resolver_to_client(self, resolver)

    def get_model_residency_capabilities(self, **kwargs: Any) -> Dict[str, Any]:
        _ = kwargs
        return _local_model_residency_capabilities(
            mode="local_single_client",
            source="abstractruntime.local",
            text_loads_other_models=False,
        )

    def list_model_residency(
        self,
        *,
        task: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        _ = kwargs
        task_s = _residency_task_filter(task)
        if task_s is not None and task_s != "text_generation":
            return _local_capability_residency_result(
                self,
                operation="list_loaded",
                task=task_s,
                provider=provider,
                model=model,
                source="abstractruntime.local",
            )
        record = _local_residency_record(
            provider=self._provider,
            model=self._model,
            default=True,
            provider_instance=getattr(self, "_llm", None),
            lock_owner=self,
        )
        _stamp_local_lock_state(
            record,
            locked=(str(self._provider or "").strip().lower(), str(self._model or "").strip())
            in _local_locked_residency_pairs(self),
        )
        if isinstance(provider, str) and provider.strip() and provider.strip().lower() != self._provider:
            records: List[Dict[str, Any]] = []
        elif isinstance(model, str) and model.strip() and model.strip() != self._model:
            records = []
        else:
            records = [record]
        records = _merge_host_sweep_into_text_records(records, provider=provider, model=model)
        records = _merge_mlx_process_residency_into_text_records(records, provider=provider, model=model)
        records = _merge_hf_process_residency_into_text_records(records, provider=provider, model=model)
        if task_s is None:
            return _local_all_model_residency_result(
                self,
                source="abstractruntime.local",
                text_records=records,
                provider=provider,
                model=model,
            )
        result = {
            "ok": True,
            "supported": True,
            "operation": "list_loaded",
            "task": "text_generation",
            "models": records,
            "diagnostics": {"source": "abstractruntime.local", "count": len(records)},
        }
        return _with_local_model_residency_summary(result, operation="list_loaded", models=records)

    def load_model_residency(
        self,
        *,
        task: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        pin: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        task_s = _normalize_residency_task(task)
        if task_s != "text_generation":
            provider_s = str(provider or "").strip().lower()
            model_s = str(model or "").strip()
            return _local_capability_residency_result(
                self,
                operation="load",
                task=task_s,
                provider=provider_s,
                model=model_s,
                options=options,
                pin=pin,
                kwargs=dict(kwargs or {}),
                source="abstractruntime.local",
            )
        provider_s = str(provider or self._provider or "").strip().lower()
        model_s = str(model or self._model or "").strip()
        _ = kwargs
        if provider_s != self._provider or model_s != self._model:
            return _model_residency_unsupported_payload(
                operation="load",
                task=task_s,
                provider=provider_s,
                model=model_s,
                error=(
                    "This local client can only report its already-active text-generation model. "
                    "Use MultiLocalAbstractCoreLLMClient or remote AbstractCore for loading other models."
                ),
            )
        before_record = _local_residency_record(
            provider=self._provider,
            model=self._model,
            default=True,
            provider_instance=getattr(self, "_llm", None),
            lock_owner=self,
        )
        provider_load_result: Any = None
        if before_record.get("loaded") is not True:
            provider_load_result, load_error = _load_local_provider_residency(
                provider_instance=getattr(self, "_llm", None),
                model=self._model,
                options=options,
                pin=pin,
                extra=dict(kwargs or {}),
            )
            if load_error:
                return _local_model_residency_load_failure(
                    operation="load",
                    task="text_generation",
                    provider=self._provider,
                    model=self._model,
                    runtime=before_record,
                    message=load_error,
                    source="abstractruntime.local",
                    provider_load_result=provider_load_result,
                )

        record = _local_residency_record(
            provider=self._provider,
            model=self._model,
            default=True,
            provider_instance=getattr(self, "_llm", None),
            lock_owner=self,
        )
        provider_loaded_new = bool(before_record.get("loaded") is not True and record.get("loaded") is True)
        if record.get("loaded") is not True:
            return _local_model_residency_load_failure(
                operation="load",
                task="text_generation",
                provider=self._provider,
                model=self._model,
                runtime=record,
                message="model_residency load completed without a loaded model",
                source="abstractruntime.local",
                provider_load_result=provider_load_result,
            )
        result = {
            "ok": True,
            "supported": True,
            "operation": "load",
            "task": "text_generation",
            "loaded_new": provider_loaded_new,
            "provider_loaded_new": provider_loaded_new,
            "runtime_cache_loaded_new": False,
            "runtime": record,
            **({"provider_load_result": _jsonable(provider_load_result)} if provider_load_result is not None else {}),
            **_load_option_report(
                provider=provider_s,
                model=model_s,
                requested=_local_provider_load_options(options=options, pin=pin, extra=dict(kwargs or {})),
                provider_load_result=provider_load_result,
                provider_called=before_record.get("loaded") is not True,
            ),
            "diagnostics": {
                "source": "abstractruntime.local",
                "loaded_new": provider_loaded_new,
                "provider_loaded_new": provider_loaded_new,
                "runtime_cache_loaded_new": False,
            },
        }
        return _with_local_model_residency_summary(
            result,
            operation="load",
            runtime=record,
            action="loaded" if provider_loaded_new else "already_loaded",
            changed=provider_loaded_new,
        )

    def unload_model_residency(
        self,
        *,
        task: Optional[str] = None,
        runtime_id: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        force: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Explicit eject (console / tray / CLI). An in-process model's eject is
        process-wide, so it first consults every OTHER owner's claims (core
        `process_residency`): another client's LOCK refuses the eject unless
        `force=True`; other clients that merely pool the model are ejected as
        well and counted in `ejected_from_other_clients`."""
        guard = _explicit_eject_guard(self, task=task, runtime_id=runtime_id, provider=provider, model=model,
                                      force=force, source='abstractruntime.local')
        if guard.get("refusal") is not None:
            return guard["refusal"]
        result = self._unload_model_residency(task=task, runtime_id=runtime_id, provider=provider, model=model,
                                              options=options, force=force, **kwargs)
        if isinstance(result, dict) and guard.get("in_process"):
            result["ejected_from_other_clients"] = int(guard.get("pooled_elsewhere") or 0)
            if guard.get("forced_over_locks"):
                result["forced_over_other_client_locks"] = guard["forced_over_locks"]
        return result

    def _unload_model_residency(
        self,
        *,
        task: Optional[str] = None,
        runtime_id: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        force: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        _ = kwargs
        task_s, provider_sel, model_sel, capability_runtime_id = _resolve_unload_selector(
            self, task=task, runtime_id=runtime_id, provider=provider, model=model,
        )
        if task_s != "text_generation":
            return _local_capability_residency_result(
                self,
                operation="unload",
                task=task_s,
                provider=provider_sel,
                model=model_sel,
                options=options,
                runtime_id=capability_runtime_id,
                kwargs=dict(kwargs or {}),
                source="abstractruntime.local",
            )
        if capability_runtime_id and not (provider_sel and model_sel):
            # A runtime_id this client does not know must never fall back to
            # the default model (it would eject a model nobody addressed).
            message = f"no local model residency row has runtime_id {capability_runtime_id!r}"
            return {"ok": False, "success": False, "supported": True, "operation": "unload",
                    "task": "text_generation", "unloaded": False, "error": message, "warnings": [message],
                    "affected_models": []}
        provider_s = str(provider_sel or self._provider or "").strip().lower()
        model_s = str(model_sel or self._model or "").strip()
        locked_pairs = _local_locked_residency_pairs(self)
        force_unlock_pending = False
        if (provider_s, model_s) in locked_pairs:
            if not force:
                # Locked pairs refuse plain unloads (payload, never an
                # exception); force=true unlocks first — same choke-point
                # semantics as core's /acore/models/unload 409.
                return _local_model_locked_refusal(
                    provider=provider_s,
                    model=model_s,
                    source="abstractruntime.local",
                )
            # The discard is DEFERRED until the provider unload succeeds: a
            # raising provider must leave the pair resident AND still locked
            # (same ordering fix as core's force path).
            force_unlock_pending = True
        if provider_s != self._provider or model_s != self._model:
            requested = _local_residency_record(
                provider=provider_s,
                model=model_s,
                default=False,
                runtime_cached=False,
                include_provider_state=False,
                lock_owner=self,
            )
            result = {
                "ok": True,
                "supported": True,
                "operation": "unload",
                "task": "text_generation",
                "unloaded": False,
                "runtime_cache_unloaded": False,
                "runtime": requested,
                "warnings": ["Requested local runtime was not resident in this client."],
                "diagnostics": {"source": "abstractruntime.local", "reason": "not_found"},
            }
            return _with_local_model_residency_summary(
                result,
                operation="unload",
                runtime=requested,
                action="not_found",
                changed=False,
            )

        before_record = _local_residency_record(
            provider=self._provider,
            model=self._model,
            default=True,
            provider_instance=getattr(self, "_llm", None),
            lock_owner=self,
        )
        provider_unload_result: Any = None
        unload_error: Optional[str] = None
        should_call_unload = before_record.get("loaded") is not False or before_record.get("provider_residency_verified") is not True
        if should_call_unload:
            provider_unload_result, unload_error = _unload_local_provider_residency(
                provider_instance=getattr(self, "_llm", None),
                model=self._model,
                options=options,
            )
        if should_call_unload and unload_error is None:
            # Core's unload_model already dropped the in-provider prompt-cache
            # stores; drop this client's mirrors of them too.
            self._drop_prompt_cache_client_state()
        if force_unlock_pending and unload_error is None:
            locked_pairs.discard((provider_s, model_s))

        # Process-wide eject of in-process MLX weights (see the multilocal
        # twin): this client's instance released its references; every other
        # holder in the process is unloaded by core's `eject_model`.
        process_eject: Optional[Dict[str, Any]] = None
        if provider_s in _PROCESS_RESIDENCY_PROVIDERS and unload_error is None:
            process_eject = _process_eject_for(provider_s, model_s)
            if process_eject is not None and process_eject.get("ok") is False and process_eject.get("error"):
                unload_error = f"process-wide {provider_s} eject failed: {process_eject['error']}"

        record = _local_residency_record(
            provider=self._provider,
            model=self._model,
            default=True,
            provider_instance=getattr(self, "_llm", None),
            lock_owner=self,
        )
        unloaded = bool(before_record.get("loaded") is True and record.get("loaded") is False)
        warnings: List[str] = []
        error: Optional[str] = None
        if unload_error:
            error = unload_error
            warnings.append(unload_error)
        elif record.get("provider_residency_verified") is not True:
            error = "model_residency unload did not verify unloaded provider residency"
            warnings.append(error)
        elif record.get("loaded") is True:
            residual = (process_eject or {}).get("residual") if isinstance(process_eject, dict) else None
            if isinstance(residual, dict):
                error = (
                    "model_residency unload completed but the weights are still resident in this process: "
                    f"{int(residual.get('holders') or 0)} holder(s) still alive, "
                    f"{int(residual.get('held_bytes') or 0)} bytes held"
                )
            else:
                error = "model_residency unload completed but provider still reports the model loaded"
            warnings.append(error)
        elif isinstance(process_eject, dict) and process_eject.get("holders_refused"):
            error = (
                "model_residency unload: "
                f"{len(process_eject['holders_refused'])} holder(s) refused to unload "
                f"({'; '.join(str(h.get('error')) for h in process_eject['holders_refused'])})"
            )
            warnings.append(error)
        result = {
            "ok": error is None,
            "supported": True,
            "operation": "unload",
            "task": "text_generation",
            "unloaded": unloaded,
            "runtime_cache_unloaded": False,
            "runtime": record,
            **({"error": error} if error else {}),
            **({"warnings": warnings} if warnings else {}),
            **({"provider_unload_result": _jsonable(provider_unload_result)} if provider_unload_result is not None else {}),
            **({"process_eject": _jsonable(process_eject)} if process_eject is not None else {}),
            "diagnostics": {
                "source": "abstractruntime.local",
                "runtime_cache_unloaded": False,
                "provider_unload_attempted": should_call_unload,
            },
        }
        if error:
            result.setdefault("status_hint", "warning")
            result.setdefault("degraded", True)
        return _with_local_model_residency_summary(
            result,
            operation="unload",
            runtime=record,
            action="unload_failed" if error else ("unloaded" if unloaded else "already_unloaded"),
            changed=unloaded,
        )

    def lock_model_residency(
        self,
        payload: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        merged = _merge_optional_payload(payload, kwargs)
        provider_s = str(self._provider or "").strip().lower()
        model_s = str(self._model or "").strip()
        return _local_model_residency_lock_result(
            merged,
            lock=True,
            default_provider=provider_s,
            default_model=model_s,
            known_pairs={(provider_s, model_s)},
            locked_pairs=_local_locked_residency_pairs(self),
            provider_instance_lookup=lambda p, m: (
                getattr(self, "_llm", None) if (p, m) == (provider_s, model_s) else None
            ),
            source="abstractruntime.local",
        )

    def unlock_model_residency(
        self,
        payload: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        merged = _merge_optional_payload(payload, kwargs)
        provider_s = str(self._provider or "").strip().lower()
        model_s = str(self._model or "").strip()
        return _local_model_residency_lock_result(
            merged,
            lock=False,
            default_provider=provider_s,
            default_model=model_s,
            known_pairs={(provider_s, model_s)},
            locked_pairs=_local_locked_residency_pairs(self),
            provider_instance_lookup=lambda p, m: (
                getattr(self, "_llm", None) if (p, m) == (provider_s, model_s) else None
            ),
            source="abstractruntime.local",
        )

    def get_context_estimate(
        self,
        payload: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        merged = _merge_optional_payload(payload, kwargs)
        return _local_context_estimate(
            provider=merged.get("provider") or self._provider,
            model=merged.get("model") or self._model,
            context_length=merged.get("context_length"),
            base_url=merged.get("base_url"),
        )

    def get_memory_snapshot(self, **kwargs: Any) -> Dict[str, Any]:
        _ = kwargs
        return _local_memory_snapshot()

    def list_session_prompt_caches(self, session_id: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        _ = kwargs
        stats = _provider_prompt_cache_stats_raw(getattr(self, "_llm", None))
        rows = _session_prompt_cache_rows_from_stats(
            stats,
            provider=self._provider,
            model=self._model,
            runtime_id=f"local:text_generation:{self._provider}:{self._model}",
            session_id=session_id,
        )
        return {"ok": True, "caches": rows}

    def clear_session_prompt_caches(self, session_id: str, **kwargs: Any) -> Dict[str, Any]:
        _ = kwargs
        session_s = str(session_id or "").strip()
        if not session_s:
            return {
                "ok": False,
                "error": "clear_session_prompt_caches requires a session_id",
                "cleared": [],
                "count": 0,
            }
        cleared: List[Dict[str, Any]] = []
        count = 0
        for row in self.list_session_prompt_caches(session_id=session_s).get("caches") or []:
            out = dict(row)
            out.update(_clear_provider_prompt_cache_key(getattr(self, "_llm", None), key=row.get("key")))
            if out.get("cleared"):
                count += 1
                self._forget_prompt_cache_key(str(row.get("key") or ""))
            cleared.append(out)
        return {"ok": True, "cleared": cleared, "count": count}

    def _forget_prompt_cache_key(self, key: str) -> None:
        key_s = str(key or "").strip()
        if not key_s:
            return
        lock = getattr(self, "_prompt_cache_state_lock", None)
        state = getattr(self, "_prompt_cache_state", None)
        if isinstance(state, dict):
            if lock is not None:
                with lock:
                    state.pop(key_s, None)
            else:
                state.pop(key_s, None)

    def _drop_prompt_cache_client_state(self) -> None:
        """Unload hygiene: the provider's own unload dropped its stores; drop
        the client-side mirrors so a later session cannot see stale
        prepared-prefix bookkeeping."""
        lock = getattr(self, "_prompt_cache_state_lock", None)
        state = getattr(self, "_prompt_cache_state", None)
        if isinstance(state, dict):
            if lock is not None:
                with lock:
                    state.clear()
            else:
                state.clear()

    def _maybe_stamp_prompt_cache_attribution(
        self,
        *,
        key: Optional[str],
        attribution: Optional[Dict[str, Any]],
    ) -> None:
        """Best-effort session attribution on a session-scoped cache key.

        Placed AFTER generate: `BaseProvider.prompt_cache_update_key_meta`
        merges into an EXISTING entry and returns False for a missing key, and
        the entry may only exist once the first generate on the key has
        completed. Stamped after EVERY generate that used a derived key — a
        done-set would go stale against core's LRU (an evicted-then-recreated
        key would silently keep empty meta forever); the merge itself is
        cheap and idempotent. Never fails the call."""
        key_s = str(key or "").strip()
        if not key_s or not isinstance(attribution, dict):
            return
        if not str(attribution.get("session_id") or "").strip():
            return
        meta_setter = getattr(getattr(self, "_llm", None), "prompt_cache_update_key_meta", None)
        if not callable(meta_setter):
            return
        try:
            meta_setter(key_s, **{k: v for k, v in attribution.items() if v is not None})
        except Exception:
            pass

    def _maybe_prepare_prompt_cache(
        self,
        *,
        prompt_cache_key: Optional[str],
        system_prompt: Optional[str],
        tools: Optional[List[Dict[str, Any]]],
        messages: Optional[List[Dict[str, Any]]],
        thinking: Any = None,
    ) -> None:
        key = str(prompt_cache_key or "").strip()
        if not key:
            return

        provider = getattr(self, "_llm", None)
        if provider is None:
            return

        try:
            supports = getattr(provider, "supports_prompt_cache", None)
            if callable(supports) and not bool(supports()):
                return
        except Exception:
            return

        try:
            supports_op = getattr(provider, "prompt_cache_supports_operation", None)
            if callable(supports_op) and not bool(supports_op("prepare_modules")):
                return
        except Exception:
            return

        # Build the immutable prefix cache as an ordered BLOC CHAIN, then fork the chain's
        # final key ONCE into the per-session key. The message lane is deliberately NOT
        # maintained here — see the long note at the state check below.
        #
        # SEPARATE `system` AND `tools` BLOCS (restored 2026-08-03). A bloc is an
        # independently-keyed slice of one rendered conversation, and the separation is the
        # whole point: `system` alone is the bloc that many agents and sessions share, and
        # keeping it separately keyed is what will let it be reused when the tool set
        # differs. Merging the two collapsed that abstraction for a render bug that belonged
        # to the render layer.
        #
        # The bug: chat templates fold the tool instructions INTO the single system turn, so
        # rendering each module as its own standalone conversation emitted two consecutive
        # `<|im_start|>system` blocks — bytes generate() never produces — and the token LCP
        # between the prefix cache and the real prompt stopped at the end of the system text
        # (measured 618 of 2148 prefix tokens reachable). It is fixed where it lives:
        # `BaseProvider.prompt_cache_plan_bloc_chain` renders the CUMULATIVE conversation
        # through the provider's own generate() renderer and cuts it at successor-independent
        # TOKEN boundaries, so the `system` bloc ends mid-turn (before `<|im_end|>`) and the
        # `tools` bloc carries the rest of that same turn. Concatenation is byte-identical to
        # the single-shot render; each bloc keeps its own key.
        #
        # Reachable fraction of generate()'s prompt, measured tokenizer-only on an
        # agent-shaped chain (700-token persona + 14 tool schemas, 2026-08-03):
        #
        #   lane           N=1 old/new     N=2 old/new     N=3 old/new
        #   qwen3 ChatML   99.6% / 99.2%   53.3% / 99.8%   52.8% / 99.8%
        #   gemma-4 turn   99.6% / 99.3%   52.9% / 99.8%   52.4% / 99.8%
        #   llama-3 plain  99.7% / 99.7%   53.7% / 99.9%   53.3% / 99.9%
        #
        # The old shape was ANTI-composable — it degraded with every bloc added. The
        # tax for keeping the blocs separate is ZERO at N>=2 (they cache exactly as
        # many tokens as one merged module would); the ~0.4% at N=1 is the system
        # turn's closing tag, left uncached so a tools bloc can still extend that turn.
        try:
            prep_fn = getattr(provider, "prompt_cache_prepare_modules", None)
            if not callable(prep_fn):
                return
            modules: List[Dict[str, Any]] = [
                {"module_id": "system", "system_prompt": system_prompt, "add_generation_prompt": False}
            ]
            if tools:
                modules.append({"module_id": "tools", "tools": tools, "add_generation_prompt": False})
            # THE CHAIN MUST BE PLANNED UNDER THE SAME `thinking` generate() GETS (2026-09-17).
            # Thinking controls can rewrite the HEAD of the system block — Qwen3.8 renders
            # "Reasoning effort is set to low. …" before the persona — so a chain planned
            # without it shares 3 tokens (`<|im_start|>system\n`) with the real prompt and
            # the whole prefix is unreachable. Measured on the gateway (Qwen3.8-27B-4bit,
            # thinking=minimal): `rebuilt` cached=0 then `hit_restore` cached=3, two full
            # ~5.5k prefills per session. AbstractCore owns the rewrite; this only hands it
            # the request.
            prep_kwargs: Dict[str, Any] = {}
            if thinking is not None:
                if _callable_accepts_kwarg(prep_fn, "thinking"):
                    prep_kwargs["thinking"] = thinking
                else:
                    # One f-string: this module's logger is a StructuredLogger, whose
                    # `warning()` takes the message only — printf-style args raise, and
                    # the blanket `except` below would turn that into a silently missing
                    # prefix cache.
                    logger.warning(
                        f"#FALLBACK prompt-cache prefix planned WITHOUT thinking={thinking!r}: this "
                        f"AbstractCore's prompt_cache_prepare_modules does not accept `thinking`. For "
                        f"models that render the control into the system block the prepared prefix "
                        f"will not match the prompt (no prefill savings). Upgrade abstractcore."
                    )
            prep = prep_fn(
                namespace="abstractcode",
                modules=modules,
                make_default=False,
                **prep_kwargs,
            )
        except Exception:
            return

        # Providers that don't implement in-process prefix caching return supported=False.
        if not isinstance(prep, dict) or prep.get("supported") is not True:
            return

        final_prefix_key = prep.get("final_cache_key")
        if not isinstance(final_prefix_key, str) or not final_prefix_key.strip():
            return
        final_prefix_key = final_prefix_key.strip()

        system_hash = ""
        tools_hash = ""
        for item in prep.get("modules") or []:
            if not isinstance(item, dict):
                continue
            module_id = str(item.get("module_id") or "").strip()
            module_hash = str(item.get("module_hash") or "").strip()
            if module_id == "system" and module_hash:
                system_hash = module_hash
            elif module_id == "tools" and module_hash:
                tools_hash = module_hash
        system_hash = system_hash or "none"
        tools_hash = tools_hash or "none"

        # `messages` is accepted for API stability but no longer drives this method.
        _ = messages

        # ------------------------------------------------------------------
        # The message lane belongs to generate(), not to this method (2026-08-02).
        #
        # What this used to do: fingerprint the message list every call and, on any
        # divergence from the recorded sequence, `prompt_cache_clear(key)` + re-fork +
        # re-append the whole history. Three things made that a net destroyer of the very
        # state it exists to build:
        #
        #  1. `_prompt_cache_state` is PER LLMClient INSTANCE and a fresh client is built
        #     per llm_call effect, so `state is None` on essentially every call — the
        #     "needs_rebuild" clear fired unconditionally and the session cache never held
        #     more than the bare prefix. (MLX's `prompt_cache_clear` additionally drops
        #     `_hybrid_snapshots[key]`, the exact state the untrimmable/hybrid lane needs.)
        #  2. The bytes this lane appended never matched what generate() sends anyway: the
        #     cached lane strips the runtime-grounding envelope and skips volatile messages,
        #     while generate() receives them. Every append was work that the delta feed then
        #     had to trim back off.
        #  3. Since 0819, `mlx_provider._prepare_cache_delta_feed` already does the right
        #     thing for a full-context caller: token-level LCP against the key's fed-token
        #     record, trim the cache to the shared prefix, feed only the suffix. That
        #     handles a rewritten tail (loop counters, edits, truncation) correctly and
        #     WITHOUT throwing the shared prefix away.
        #
        # So the contract is now: this method only guarantees "the session key exists and
        # was forked from the current (system+tools) prefix". Everything downstream of that
        # boundary is the provider's delta feed. Divergence is no longer a rebuild trigger —
        # a diverging transcript (including two sub-runs sharing one derived key) is handled
        # by LCP+trim, which is correct by construction: the cache is trimmed to the true
        # shared token prefix and the rest is fed. Correctness never depended on the clear.
        #
        # The identity check reads the PROVIDER's cache meta (`forked_from`, written by
        # `prompt_cache_fork`) rather than the per-instance state dict, so it survives the
        # fresh-client-per-call shape that defeated the old check.
        # ------------------------------------------------------------------
        live_meta: Dict[str, Any] = {}
        try:
            meta_fn = getattr(provider, "prompt_cache_key_meta", None)
            if callable(meta_fn):
                meta = meta_fn(key)
                if isinstance(meta, dict):
                    live_meta = meta
        except Exception:
            live_meta = {}
        already_forked = bool(live_meta) and str(live_meta.get("forked_from") or "") == final_prefix_key

        with self._prompt_cache_state_lock:
            state = self._prompt_cache_state.get(key)
            if already_forked:
                if state is None or state.prefix_cache_key != final_prefix_key:
                    self._prompt_cache_state[key] = _PromptCacheSessionState(
                        system_module_hash=system_hash,
                        tools_module_hash=tools_hash,
                        prefix_cache_key=final_prefix_key,
                        message_hashes=[],
                    )
                return

            # No usable session cache for this prefix identity (first call, or the
            # system/tools prefix genuinely changed): (re)fork from the prefix once.
            try:
                clearer = getattr(provider, "prompt_cache_clear", None)
                if callable(clearer):
                    clearer(key)
            except Exception:
                pass

            forked = False
            try:
                forker = getattr(provider, "prompt_cache_fork", None)
                if callable(forker):
                    forked = bool(forker(final_prefix_key, key, make_default=False))
            except Exception:
                forked = False

            if not forked:
                try:
                    setter = getattr(provider, "prompt_cache_set", None)
                    updater = getattr(provider, "prompt_cache_update", None)
                    if callable(setter) and callable(updater) and bool(setter(key, make_default=False)):
                        # Fork-less fallback: ONE update carrying system AND tools together,
                        # so the provider renders the single merged system turn in one go.
                        # This lane has no bloc structure by construction (there is nothing
                        # to reuse — the key is being built from empty anyway); it exists
                        # only so providers without `prompt_cache_fork` still get a warm
                        # session key.
                        updater(
                            key,
                            system_prompt=system_prompt,
                            tools=tools,
                            add_generation_prompt=False,
                            **({"thinking": thinking} if thinking is not None else {}),
                        )
                        forked = True
                except Exception:
                    forked = False

            if not forked:
                return

            # Stamp the prefix identity so the NEXT call (a different client instance) can
            # recognize this key as already prepared instead of clearing it.
            try:
                meta_setter = getattr(provider, "prompt_cache_update_key_meta", None)
                if callable(meta_setter):
                    meta_setter(key, forked_from=final_prefix_key)
            except Exception:
                pass

            self._prompt_cache_state[key] = _PromptCacheSessionState(
                system_module_hash=system_hash,
                tools_module_hash=tools_hash,
                prefix_cache_key=final_prefix_key,
                message_hashes=[],
            )

    def generate(
        self,
        *,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        media: Optional[List[Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        tmpdir: Optional[tempfile.TemporaryDirectory] = None
        if isinstance(media, list) and media and self._artifact_store is not None:
            has_artifacts = any(
                isinstance(item, dict)
                and (
                    (isinstance(item.get("$artifact"), str) and str(item.get("$artifact") or "").strip())
                    or (isinstance(item.get("artifact_id"), str) and str(item.get("artifact_id") or "").strip())
                )
                and not (isinstance(item.get("file_path"), str) and str(item.get("file_path") or "").strip())
                and item.get("content") is None
                for item in media
            )
            if has_artifacts:
                tmpdir = tempfile.TemporaryDirectory(prefix="abstractruntime_llm_media_")
                try:
                    media = _resolve_media_artifacts(media, artifact_store=self._artifact_store, temp_dir=tmpdir.name)
                except Exception:
                    tmpdir.cleanup()
                    raise
            else:
                media = _resolve_media_artifacts(media, artifact_store=self._artifact_store)
        else:
            media = _resolve_media_artifacts(media, artifact_store=self._artifact_store)

        try:
            params = _normalize_prompt_cache_binding_params(params)
            prompt_cache_attribution = params.pop("_prompt_cache_attribution", None)
            # The runtime's PER-CALL live delta callback (core.live_deltas),
            # never a provider kwarg and never the per-client `on_token`
            # (which is shared by every run using this client).
            on_delta = params.pop("_on_delta", None)
            prompt = _promote_text_param_to_prompt(prompt, params)
            has_binding = _has_prompt_cache_binding(params)
            output_request = params.get("output")
            acore_output_request = _is_abstractcore_output_request(output_request)
            if _output_request_has_generated_media(output_request) and self._artifact_store is None:
                raise ValueError("Generated media outputs require an ArtifactStore.")
            skip_turn_grounding = _output_request_has_non_text_result(output_request) or bool(media)
            trace_metadata = params.get("trace_metadata") if isinstance(params.get("trace_metadata"), dict) else {}
            run_id = trace_metadata.get("run_id") if isinstance(trace_metadata, dict) else None
            run_id = str(run_id).strip() if isinstance(run_id, str) and run_id.strip() else None
            default_artifact_tags: Dict[str, Any] = {
                "source": "llm_call",
                "provider": self._provider,
                "model": self._model,
            }
            if isinstance(trace_metadata, dict):
                for key in (
                    "workflow_id",
                    "node_id",
                    "step_id",
                    "effect_idempotency_key",
                    "actor_id",
                    "session_id",
                    "parent_run_id",
                    "request_id",
                ):
                    raw = trace_metadata.get(key)
                    if raw is not None and str(raw).strip():
                        default_artifact_tags[key] = str(raw)
            output_run_id, output_tags = _output_runtime_metadata(output_request)
            if run_id is None and output_run_id:
                run_id = output_run_id
            if output_tags:
                default_artifact_tags.update(output_tags)

            system_prompt = _strip_system_context_header(system_prompt)
            runtime_grounding = _mark_grounding_prompt_injected(
                _runtime_grounding_metadata(trace_metadata),
                not skip_turn_grounding,
            )
            prompt, messages = _normalize_turn_grounding(
                prompt=str(prompt or ""),
                messages=messages,
                grounding=runtime_grounding if not skip_turn_grounding else None,
            )
            messages = _strip_internal_system_messages(messages)
            system_prompt, messages = _coalesce_leading_system_messages(
                system_prompt=system_prompt,
                messages=messages,
            )
            system_prompt = _append_runtime_grounding_contract(
                system_prompt,
                injected=not skip_turn_grounding,
            )
            generation_context = _generated_artifact_context(
                prompt=str(prompt or ""),
                media=media,
                output_request=output_request,
            )

            stream_raw = params.pop("stream", None)
            if stream_raw is None:
                stream_raw = params.pop("streaming", None)
            if isinstance(stream_raw, str):
                stream = stream_raw.strip().lower() in {"1", "true", "yes", "y", "on"}
            else:
                stream = bool(stream_raw) if stream_raw is not None else False

            # Structured/artifact output requests never ride the token-stream path
            # (code seat c1009 defect 1): the durable contract for those calls is a
            # VALIDATED object — the streamed normalizer cannot produce `data` or wire
            # artifact-backed outputs, so a review/structured call under
            # `_runtime.stream=true` completed with an empty answer. Streaming is a
            # per-call rendering optimization; correctness wins, on_token stays
            # silent for these calls.
            if stream and (
                acore_output_request
                or output_request is not None
                or params.get("response_model") is not None
                or params.get("response_format") is not None
            ):
                stream = False
                _mark_stream_unavailable(on_delta, "structured_output")

            # STREAMING PARITY GATE (S-2): a streamed call must record what the
            # non-streamed call records. Where the provider cannot, the call
            # runs non-streamed and says why (never a silent downgrade).
            if stream:
                refusal = self._stream_parity_refusal(params)
                if refusal is not None:
                    stream = False
                    _mark_stream_unavailable(on_delta, refusal)

            requested_base_url = params.get("base_url")
            requested_provider = params.get("_provider")
            requested_model = params.get("_model")
            requested_thinking = _with_capability_default_reasoning(
                params, getattr(self, "_capability_defaults", None)
            )

            # `base_url` is a provider construction concern in local mode. We intentionally
            # do not create new providers per call unless the host explicitly chooses to.
            params.pop("base_url", None)
            # Reserved routing keys (used by MultiLocalAbstractCoreLLMClient).
            params.pop("_provider", None)
            params.pop("_model", None)

            if acore_output_request and "output" in params:
                params["output"] = _strip_runtime_output_metadata_for_core(params.get("output"))

            if acore_output_request and not tools:
                from abstractcore.core.generate_contract import normalize_generate_request, resolve_generate_route  # type: ignore

                capability_defaults = getattr(self, "_capability_defaults", {})
                resolved_generate_route = resolve_generate_route(
                    request=normalize_generate_request(
                        prompt=str(prompt or ""),
                        messages=messages,
                        media=media,
                    ),
                    output=params.get("output"),
                    scoped_routes=capability_defaults,
                    explicit_text_route={
                        "provider": requested_provider,
                        "model": requested_model,
                        "base_url": requested_base_url,
                    },
                    explicit_reasoning=requested_thinking,
                )
                resolved_generate_route_summary = resolved_generate_route.to_summary()
                specs = [dict(spec) for spec in resolved_generate_route.output_specs]
                media_only = bool(specs) and all(
                    isinstance(spec, dict)
                    and (
                        str(spec.get("modality") or "").strip().lower() in {"image", "video", "voice"}
                        or (
                            str(spec.get("modality") or "").strip().lower() == "text"
                            and str(spec.get("task") or "").strip().lower() == "transcription"
                        )
                    )
                    for spec in specs
                )
                run_spec = getattr(self._llm, "_run_multimodal_spec", None)
                if media_only and callable(run_spec):
                    if _is_subprocess_safe_image_specs(specs, media):
                        result_obj = _run_local_image_subprocess(
                            provider=self._provider,
                            model=self._model,
                            llm_kwargs=getattr(self, "_llm_kwargs", {}),
                            prompt=str(prompt or ""),
                            specs=[dict(spec) for spec in specs],
                            media=media,
                            progress_callback=params.get("on_progress"),
                        )
                        result = _normalize_multimodal_response(
                            result_obj,
                            artifact_store=self._artifact_store,
                            run_id=run_id,
                            default_tags=default_artifact_tags,
                            generation_context=generation_context,
                        )
                        meta = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
                        meta["_resolved_generate_route"] = resolved_generate_route_summary
                        result["metadata"] = meta
                    elif _is_subprocess_safe_video_specs(specs, media):
                        result_obj = _run_local_video_subprocess(
                            provider=self._provider,
                            model=self._model,
                            llm_kwargs=getattr(self, "_llm_kwargs", {}),
                            prompt=str(prompt or ""),
                            specs=[dict(spec) for spec in specs],
                            media=media,
                            progress_callback=params.get("on_progress"),
                        )
                        result = _normalize_multimodal_response(
                            result_obj,
                            artifact_store=self._artifact_store,
                            run_id=run_id,
                            default_tags=default_artifact_tags,
                            generation_context=generation_context,
                        )
                        meta = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
                        meta["_resolved_generate_route"] = resolved_generate_route_summary
                        result["metadata"] = meta
                    else:
                        from abstractcore.core.multimodal_generation import MultimodalGenerateResponse  # type: ignore

                        progress_callback = params.get("on_progress")
                        result_obj = MultimodalGenerateResponse(
                            metadata={
                                "media_only": True,
                                "runtime_provider": self._provider,
                                "runtime_model": self._model,
                                "_resolved_generate_route": resolved_generate_route_summary,
                            }
                        )
                        for spec in specs:
                            run_spec(
                                result=result_obj,
                                spec=_with_output_progress_callback(dict(spec), progress_callback),
                                prompt=str(prompt or ""),
                                media=media,
                                artifact_store=self._artifact_store,
                            )
                        result = _normalize_local_response(
                            result_obj,
                            artifact_store=self._artifact_store,
                            run_id=run_id,
                            default_tags=default_artifact_tags,
                            generation_context=generation_context,
                        )
                    _sanitize_runtime_grounding_echoes(result)
                    _attach_runtime_grounding(result, runtime_grounding)
                    result["tool_calls"] = []
                    return result

            # A PROMPT-ONLY CALL UNDER A RUNTIME-DERIVED KEY IS STILL FULL-CONTEXT (2026-09-17).
            #
            # AbstractCore reads the caller SHAPE to pick a cache discipline: `messages=None`
            # means "the cache IS my context, this prompt is the next fragment" (KV-mode
            # sessions) and the prompt is APPENDED to whatever the key holds. That is never
            # true here — a runtime llm_call re-sends its whole context every time, and the
            # key was derived by the runtime (the attribution rider is how we know), not
            # chosen by a caller who wanted append semantics. Measured on the assistant's
            # `route_call` node: the full system+tools+prompt render was stacked onto the
            # same key every turn (6790 → 10130 → 13480 cached tokens over three turns),
            # zero reuse, and the router read every earlier routing request as live context.
            #
            # `messages=[]` is AbstractCore's documented spelling of "full context, empty so
            # far": same rendered bytes (the prompt is the final user turn either way), but
            # the key now gets prefix/delta discipline instead of append.
            #
            # Apply to the legacy local append lane AND native MLX's full-history
            # keyed APC lane (which does not offer prepare/fork). Elsewhere the
            # rewrite can change the wire: Ollama picks `/api/chat` over `/api/generate` on
            # `messages is not None`, so a cache-discipline fix would have switched its
            # endpoint and template (adversarial find, 2026-09-17).
            call_messages = _strip_synthetic_message_markers(_strip_volatile_markers(messages))
            if (
                call_messages is None
                and prompt_cache_attribution is not None
                and not has_binding
                and isinstance(params.get("prompt_cache_key"), str)
                and params.get("prompt_cache_key").strip()
                and _uses_local_full_context_prompt_cache(
                    getattr(self, "_llm", None), provider_name=self._provider,
                )
            ):
                call_messages = []

            def _invoke_provider(stream_flag: bool) -> Tuple[Dict[str, Any], bool]:
                """One provider call -> (normalized result, whether it streamed)."""

                options_unsupported_before = bool(getattr(self._llm, "_stream_options_unsupported", False))
                resp = self._llm.generate(
                    prompt=str(prompt or ""),
                    messages=call_messages,
                    system_prompt=system_prompt,
                    tools=tools,
                    media=media,
                    stream=stream_flag,
                    **params,
                )
                if stream_flag and hasattr(resp, "__next__"):
                    # The OpenAI-compatible provider learns that its server
                    # rejects `stream_options` (so no streamed usage) while
                    # opening the stream, before the first token. Look at the
                    # first chunk, then check: if usage just became
                    # unavailable, abandon the stream and answer non-streamed.
                    try:
                        first_chunk = next(resp)
                        head: List[Any] = [first_chunk]
                    except StopIteration:
                        head = []
                    if not options_unsupported_before and getattr(self._llm, "_stream_options_unsupported", False):
                        _close = getattr(resp, "close", None)
                        if callable(_close):
                            try:
                                _close()
                            except Exception:
                                pass
                        _mark_stream_unavailable(on_delta, "usage_unavailable")
                        return _invoke_provider(False)
                    streamed = _normalize_local_streaming_response(
                        itertools.chain(head, resp),
                        on_token=getattr(self, "_on_token", None),
                        on_delta=on_delta,
                    )
                    if streamed.get("usage") is None:
                        # Found out only at the end, for a reason the provider
                        # did not flag: this call is reported, and the next call
                        # tries streaming again (a one-off server hiccup must not
                        # switch streaming off for the model). A server that
                        # REJECTS usage in streams is the provider's own flag,
                        # checked before every call in `_stream_parity_refusal`.
                        _mark_stream_unavailable(on_delta, "usage_unavailable")
                    return streamed, True
                if stream_flag:
                    _mark_stream_unavailable(on_delta, "provider_cannot_stream")
                return (
                    _normalize_local_response(
                        resp,
                        artifact_store=self._artifact_store if acore_output_request else None,
                        run_id=run_id,
                        default_tags=default_artifact_tags,
                        generation_context=generation_context,
                    ),
                    False,
                )

            lock = getattr(self, "_generate_lock", None)
            # Query the loaded instance on each call, not the provider name or
            # constructor options: only its internal scheduler guarantees safe
            # admission, and that guarantee can disappear after an unload.
            if lock is not None and _local_instance_schedules_generation(self._llm):
                lock = None
            if lock is None:
                if not has_binding:
                    self._maybe_prepare_prompt_cache(
                        prompt_cache_key=params.get("prompt_cache_key"),
                        system_prompt=system_prompt,
                        tools=tools,
                        messages=messages,
                        thinking=params.get("thinking"),
                    )
                result, stream = _invoke_provider(stream)
                _sanitize_runtime_grounding_echoes(result)
                _attach_runtime_grounding(result, runtime_grounding)
                result["tool_calls"] = _normalize_tool_calls(result.get("tool_calls"))
            else:
                # Serialize generation for non-thread-safe providers (e.g. MLX).
                _warn_local_generate_lock_once(provider=self._provider, model=self._model)
                with lock:
                    if not has_binding:
                        self._maybe_prepare_prompt_cache(
                            prompt_cache_key=params.get("prompt_cache_key"),
                            system_prompt=system_prompt,
                            tools=tools,
                            messages=messages,
                            thinking=params.get("thinking"),
                        )
                    result, stream = _invoke_provider(stream)
                    _sanitize_runtime_grounding_echoes(result)
                    _attach_runtime_grounding(result, runtime_grounding)
                    result["tool_calls"] = _normalize_tool_calls(result.get("tool_calls"))

            # Durable observability: ensure a provider request payload exists even when the
            # underlying provider does not attach `_provider_request` metadata.
            #
            # AbstractCode's `/llm --verbatim` expects `metadata._provider_request.payload.messages`
            # to be present to display the exact system/user content that was sent.
            try:
                meta = result.get("metadata")
                if not isinstance(meta, dict):
                    meta = {}
                    result["metadata"] = meta

                if "_provider_request" not in meta:
                    out_messages: List[Dict[str, str]] = []
                    if isinstance(system_prompt, str) and system_prompt:
                        out_messages.append({"role": "system", "content": system_prompt})
                    if isinstance(messages, list) and messages:
                        # Copy dict entries defensively (caller-owned objects).
                        out_messages.extend([dict(m) for m in messages if isinstance(m, dict)])

                    # Append the current prompt as the final user message unless it's already present.
                    prompt_str = str(prompt or "")
                    if prompt_str:
                        last = out_messages[-1] if out_messages else None
                        if not (
                            isinstance(last, dict) and last.get("role") == "user" and last.get("content") == prompt_str
                        ):
                            out_messages.append({"role": "user", "content": prompt_str})

                    payload: Dict[str, Any] = {
                        "model": str(self._model),
                        "messages": out_messages,
                        "stream": bool(stream),
                    }
                    if runtime_grounding:
                        payload["runtime_grounding"] = dict(runtime_grounding)
                    if tools is not None:
                        payload["tools"] = tools

                    # Include generation params for debugging; keep JSON-safe (e.g. response_model).
                    # Host callbacks (`on_progress`, injected by the runtime for
                    # every LLM_CALL) are dropped rather than stringified: a
                    # `<bound method ...>` repr in every persisted provider
                    # request is noise no reader can act on.
                    # The runtime's `cancel_event` (a live threading.Event) is the
                    # same class of in-process handle and is dropped the same way.
                    payload["params"] = (
                        _jsonable(
                            {
                                k: v
                                for k, v in params.items()
                                if not callable(v) and not isinstance(v, threading.Event)
                            }
                        )
                        if params
                        else {}
                    )

                    meta["_provider_request"] = {
                        "transport": "local",
                        "provider": str(self._provider),
                        "model": str(self._model),
                        "payload": payload,
                    }
            except Exception:
                # Never fail an LLM call due to observability.
                pass

            self._maybe_stamp_prompt_cache_attribution(
                key=params.get("prompt_cache_key"),
                attribution=prompt_cache_attribution,
            )
            return result
        finally:
            if tmpdir is not None:
                try:
                    tmpdir.cleanup()
                except Exception:
                    pass

    def get_model_capabilities(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get model capabilities including max_tokens, vision_support, etc.

        Uses AbstractCore's architecture detection system to query model limits
        and features. This allows the runtime to be aware of model constraints
        for resource tracking and warnings.

        Returns:
            Dict with model capabilities. Always includes 'max_tokens' (default: DEFAULT_MAX_TOKENS).
        """
        target_model = str(model_name or self._model or "").strip() or self._model
        from .discovery_queries import local_get_model_capabilities

        payload = local_get_model_capabilities(target_model)
        capabilities = payload.get("capabilities") if isinstance(payload, dict) else None
        if isinstance(capabilities, dict):
            return capabilities
        from abstractruntime.core.vars import DEFAULT_MAX_TOKENS

        return {"max_tokens": DEFAULT_MAX_TOKENS}

    def lookup_model_capabilities(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        target_model = str(model_name or self._model or "").strip() or self._model
        from .discovery_queries import local_get_model_capabilities

        return local_get_model_capabilities(target_model)

    def get_execution_capabilities(
        self, model_name: Optional[str] = None, *, provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        from abstractcore.providers.speculation import get_execution_capabilities

        target_model = str(model_name or self._model or "").strip()
        target_provider = str(provider or self._provider or "").strip().lower()
        instance = self._llm if (target_model == self._model and target_provider == self._provider) else None
        return get_execution_capabilities(target_model, provider=target_provider, instance=instance)

    def list_providers(
        self,
        *,
        include_models: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        _ = kwargs
        from .discovery_queries import local_list_providers

        return local_list_providers(
            include_models=include_models,
            default_provider=self._provider,
            default_model=self._model,
        )

    def list_provider_models(
        self,
        provider_name: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        from .discovery_queries import local_list_provider_models

        call_kwargs = dict(kwargs)
        provider_api_key = _pop_provider_api_key(call_kwargs)
        return local_list_provider_models(
            provider_name,
            base_url=call_kwargs.get("base_url"),
            provider_api_key=provider_api_key,
            input_type=call_kwargs.get("input_type"),
            output_type=call_kwargs.get("output_type"),
            capability_route=call_kwargs.get("capability_route", call_kwargs.get("capability_routes")),
            timeout_s=call_kwargs.get("timeout_s"),
        )

    def list_embedding_models(
        self,
        *,
        base_url: Optional[str] = None,
        provider_api_key: Optional[str] = None,
        provider: Optional[str] = None,
        providers_only: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        call_kwargs = dict(kwargs)
        provider_api_key = provider_api_key or _pop_provider_api_key(call_kwargs)
        from .discovery_queries import local_list_embedding_models

        return local_list_embedding_models(
            base_url=base_url,
            provider_api_key=provider_api_key,
            provider=provider,
            providers_only=providers_only,
            timeout_s=call_kwargs.get("timeout_s"),
        )

    def get_voice_catalog(
        self,
        *,
        base_url: Optional[str] = None,
        provider_api_key: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        providers_only: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        call_kwargs = dict(kwargs)
        provider_api_key = provider_api_key or _pop_provider_api_key(call_kwargs)
        from .discovery_queries import local_get_voice_catalog

        return local_get_voice_catalog(
            base_url=base_url,
            provider_api_key=provider_api_key,
            provider=provider,
            model=model,
            providers_only=providers_only,
        )

    def list_tts_models(
        self,
        *,
        base_url: Optional[str] = None,
        provider_api_key: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        call_kwargs = dict(kwargs)
        provider_api_key = provider_api_key or _pop_provider_api_key(call_kwargs)
        from .discovery_queries import local_list_tts_models

        return local_list_tts_models(
            base_url=base_url,
            provider_api_key=provider_api_key,
            provider=provider,
        )

    def list_stt_models(
        self,
        *,
        base_url: Optional[str] = None,
        provider_api_key: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        call_kwargs = dict(kwargs)
        provider_api_key = provider_api_key or _pop_provider_api_key(call_kwargs)
        from .discovery_queries import local_list_stt_models

        return local_list_stt_models(
            base_url=base_url,
            provider_api_key=provider_api_key,
            provider=provider,
        )

    def list_music_providers(
        self,
        *,
        task: Optional[str] = None,
        base_url: Optional[str] = None,
        provider_api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        call_kwargs = dict(kwargs)
        provider_api_key = provider_api_key or _pop_provider_api_key(call_kwargs)
        from .discovery_queries import local_list_music_providers

        return local_list_music_providers(
            task=task,
            base_url=base_url,
            provider_api_key=provider_api_key,
        )

    def list_music_models(
        self,
        *,
        task: Optional[str] = None,
        base_url: Optional[str] = None,
        provider_api_key: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        call_kwargs = dict(kwargs)
        provider_api_key = provider_api_key or _pop_provider_api_key(call_kwargs)
        from .discovery_queries import local_list_music_models

        return local_list_music_models(
            task=task,
            base_url=base_url,
            provider_api_key=provider_api_key,
            provider=provider,
        )

    def list_vision_provider_models(
        self,
        *,
        task: Optional[str] = None,
        base_url: Optional[str] = None,
        provider_api_key: Optional[str] = None,
        provider: Optional[str] = None,
        providers_only: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        call_kwargs = dict(kwargs)
        provider_api_key = provider_api_key or _pop_provider_api_key(call_kwargs)
        from .discovery_queries import local_list_vision_provider_models

        return local_list_vision_provider_models(
            task=task,
            base_url=base_url,
            provider_api_key=provider_api_key,
            provider=provider,
            providers_only=providers_only,
        )

    def list_vision_adapters(
        self,
        *,
        model: Optional[str] = None,
        task: Optional[str] = None,
        base_url: Optional[str] = None,
        provider_api_key: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        call_kwargs = dict(kwargs)
        provider_api_key = provider_api_key or _pop_provider_api_key(call_kwargs)
        from .discovery_queries import local_list_vision_adapters

        return local_list_vision_adapters(
            model=model,
            task=task,
            base_url=base_url,
            provider_api_key=provider_api_key,
            provider=provider,
        )

    def list_cached_vision_models(
        self,
        *,
        task: Optional[str] = None,
        base_url: Optional[str] = None,
        provider_api_key: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        _ = (base_url, provider_api_key, kwargs)
        from .discovery_queries import local_list_cached_vision_models

        return local_list_cached_vision_models(task=task, provider=provider)

    def get_prompt_cache_capabilities(self, **kwargs: Any) -> Dict[str, Any]:
        _ = kwargs
        return _prompt_cache_capabilities_payload(getattr(self, "_llm", None))

    def get_prompt_cache_stats(self, **kwargs: Any) -> Dict[str, Any]:
        _ = kwargs
        provider = getattr(self, "_llm", None)
        if provider is None:
            return _prompt_cache_unsupported_payload(
                provider,
                operation="stats",
                error="Runtime LLM client has no provider instance",
            )
        if not _prompt_cache_supports(provider, "stats") or not hasattr(provider, "get_prompt_cache_stats"):
            return _prompt_cache_unsupported_payload(
                provider,
                operation="stats",
                error="Provider does not support prompt cache stats",
            )
        try:
            return {
                "supported": True,
                "operation": "stats",
                "capabilities": _prompt_cache_capabilities_dict(provider),
                "stats": provider.get_prompt_cache_stats(),
            }
        except Exception as e:
            return _prompt_cache_error_payload(provider, operation="stats", error=e)

    def prompt_cache_set(
        self,
        *,
        key: str,
        make_default: bool = True,
        ttl_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        _ = kwargs
        provider = getattr(self, "_llm", None)
        if provider is None:
            return _prompt_cache_unsupported_payload(
                provider,
                operation="set",
                error="Runtime LLM client has no provider instance",
            )
        if not _prompt_cache_supports(provider, "set") or not hasattr(provider, "prompt_cache_set"):
            return _prompt_cache_unsupported_payload(
                provider,
                operation="set",
                error="Provider does not support prompt cache control plane",
            )
        try:
            ok = provider.prompt_cache_set(key, make_default=bool(make_default), ttl_s=ttl_s)
            return {
                "supported": True,
                "operation": "set",
                "ok": bool(ok),
                "capabilities": _prompt_cache_capabilities_dict(provider),
            }
        except Exception as e:
            return _prompt_cache_error_payload(provider, operation="set", error=e)

    def prompt_cache_update(
        self,
        *,
        key: str,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        add_generation_prompt: bool = False,
        ttl_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        _ = kwargs
        provider = getattr(self, "_llm", None)
        if provider is None:
            return _prompt_cache_unsupported_payload(
                provider,
                operation="update",
                error="Runtime LLM client has no provider instance",
            )
        if not _prompt_cache_supports(provider, "update") or not hasattr(provider, "prompt_cache_update"):
            return _prompt_cache_unsupported_payload(
                provider,
                operation="update",
                error="Provider does not support prompt cache control plane",
            )
        try:
            ok = provider.prompt_cache_update(
                key,
                prompt=str(prompt or ""),
                messages=messages,
                system_prompt=system_prompt,
                tools=tools,
                add_generation_prompt=bool(add_generation_prompt),
                ttl_s=ttl_s,
            )
            return {
                "supported": True,
                "operation": "update",
                "ok": bool(ok),
                "capabilities": _prompt_cache_capabilities_dict(provider),
            }
        except Exception as e:
            return _prompt_cache_error_payload(provider, operation="update", error=e)

    def prompt_cache_fork(
        self,
        *,
        from_key: str,
        to_key: str,
        make_default: bool = False,
        ttl_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        _ = kwargs
        provider = getattr(self, "_llm", None)
        if provider is None:
            return _prompt_cache_unsupported_payload(
                provider,
                operation="fork",
                error="Runtime LLM client has no provider instance",
            )
        if not _prompt_cache_supports(provider, "fork") or not hasattr(provider, "prompt_cache_fork"):
            return _prompt_cache_unsupported_payload(
                provider,
                operation="fork",
                error="Provider does not support prompt cache control plane",
            )
        try:
            ok = provider.prompt_cache_fork(
                from_key,
                to_key,
                make_default=bool(make_default),
                ttl_s=ttl_s,
            )
            return {
                "supported": True,
                "operation": "fork",
                "ok": bool(ok),
                "capabilities": _prompt_cache_capabilities_dict(provider),
            }
        except Exception as e:
            return _prompt_cache_error_payload(provider, operation="fork", error=e)

    def prompt_cache_clear(
        self,
        *,
        key: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        _ = kwargs
        provider = getattr(self, "_llm", None)
        if provider is None:
            return _prompt_cache_unsupported_payload(
                provider,
                operation="clear",
                error="Runtime LLM client has no provider instance",
            )
        if not _prompt_cache_supports(provider, "clear") or not hasattr(provider, "prompt_cache_clear"):
            return _prompt_cache_unsupported_payload(
                provider,
                operation="clear",
                error="Provider does not support prompt cache control plane",
            )
        try:
            ok = provider.prompt_cache_clear(key)
            return {
                "supported": True,
                "operation": "clear",
                "ok": bool(ok),
                "capabilities": _prompt_cache_capabilities_dict(provider),
            }
        except Exception as e:
            return _prompt_cache_error_payload(provider, operation="clear", error=e)

    def prompt_cache_prepare_modules(
        self,
        *,
        namespace: str,
        modules: List[Dict[str, Any]],
        make_default: bool = False,
        ttl_s: Optional[float] = None,
        version: int = 1,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # `thinking` is part of the prefix identity (it can rewrite the head of the
        # system block) — a host that names it must not have it dropped on the floor.
        thinking = kwargs.get("thinking")
        provider = getattr(self, "_llm", None)
        if provider is None:
            return _prompt_cache_unsupported_payload(
                provider,
                operation="prepare_modules",
                error="Runtime LLM client has no provider instance",
            )
        if not _prompt_cache_supports(provider, "prepare_modules") or not hasattr(provider, "prompt_cache_prepare_modules"):
            return _prompt_cache_unsupported_payload(
                provider,
                operation="prepare_modules",
                error="Provider does not support prompt cache module preparation",
            )
        try:
            extra: Dict[str, Any] = {}
            if thinking is not None:
                if not _callable_accepts_kwarg(provider.prompt_cache_prepare_modules, "thinking"):
                    return _prompt_cache_unsupported_payload(
                        provider,
                        operation="prepare_modules",
                        error=(
                            "This AbstractCore's prompt_cache_prepare_modules does not accept "
                            "`thinking`; a prefix planned without it would not match the prompt. "
                            "Upgrade abstractcore."
                        ),
                    )
                extra["thinking"] = thinking
            result = provider.prompt_cache_prepare_modules(
                namespace=namespace,
                modules=modules,
                make_default=bool(make_default),
                ttl_s=ttl_s,
                version=int(version),
                **extra,
            )
            if isinstance(result, dict):
                result.setdefault("operation", "prepare_modules")
                result.setdefault("capabilities", _prompt_cache_capabilities_dict(provider))
                return result
            return {
                "supported": True,
                "operation": "prepare_modules",
                "capabilities": _prompt_cache_capabilities_dict(provider),
                "result": result,
            }
        except Exception as e:
            return _prompt_cache_error_payload(provider, operation="prepare_modules", error=e)

    def list_prompt_cache_exports(
        self,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        root_dir = _coerce_prompt_cache_export_root_dir(
            kwargs.pop("prompt_cache_export_root_dir", self._prompt_cache_export_root_dir)
        )
        if isinstance(provider, str) and provider.strip() and provider.strip().lower() != self._provider:
            return {
                "supported": True,
                "ok": True,
                "operation": "list_exports",
                "local_only": True,
                "provider": provider.strip().lower(),
                "model": model or self._model,
                "root_dir": str(root_dir),
                "items": [],
                "capabilities": _prompt_cache_capabilities_dict(getattr(self, "_llm", None)),
            }
        if isinstance(model, str) and model.strip() and model.strip() != self._model:
            return {
                "supported": True,
                "ok": True,
                "operation": "list_exports",
                "local_only": True,
                "provider": self._provider,
                "model": model.strip(),
                "root_dir": str(root_dir),
                "items": [],
                "capabilities": _prompt_cache_capabilities_dict(getattr(self, "_llm", None)),
            }
        return _list_prompt_cache_exports_local(
            root_dir=root_dir,
            provider=getattr(self, "_llm", None),
            provider_name=self._provider,
            model=self._model,
        )

    def prompt_cache_export(
        self,
        *,
        name: str,
        key: str,
        q8: bool = False,
        meta: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        provider = getattr(self, "_llm", None)
        target_provider = str(kwargs.pop("provider", "") or "").strip().lower()
        target_model = str(kwargs.pop("model", "") or "").strip()
        if provider is None:
            return _prompt_cache_unsupported_payload(
                provider,
                operation="export",
                error="Runtime LLM client has no provider instance",
            )
        if (target_provider and target_provider != self._provider) or (target_model and target_model != self._model):
            return {
                "supported": False,
                "operation": "export",
                "code": "invalid_target",
                "error": (
                    "Local prompt-cache export is bound to the active runtime provider/model "
                    f"{self._provider}/{self._model}; requested "
                    f"{target_provider or self._provider}/{target_model or self._model}."
                ),
                "capabilities": _prompt_cache_capabilities_dict(provider),
            }
        if not _prompt_cache_supports(provider, "save") or not hasattr(provider, "prompt_cache_save"):
            return _prompt_cache_unsupported_payload(
                provider,
                operation="export",
                error="Provider does not support host-local prompt cache export",
            )
        export_root_dir = _coerce_prompt_cache_export_root_dir(
            kwargs.pop("prompt_cache_export_root_dir", self._prompt_cache_export_root_dir)
        )
        try:
            normalized_name, artifact_path, meta_path = _prompt_cache_export_paths(
                root_dir=export_root_dir,
                provider=self._provider,
                model=self._model,
                name=name,
                extension=_prompt_cache_artifact_extension(provider),
            )
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            provider_result = provider.prompt_cache_save(
                str(key or "").strip(),
                str(artifact_path),
                q8=bool(q8),
                meta=dict(meta or {}),
            )
            if isinstance(provider_result, dict) and provider_result.get("supported") is False:
                return provider_result
            provider_meta = (
                dict(provider_result.get("meta") or {})
                if isinstance(provider_result, dict) and isinstance(provider_result.get("meta"), dict)
                else {}
            )
            record: Dict[str, Any] = {
                "schema": _RUNTIME_PROMPT_CACHE_EXPORT_SCHEMA,
                "name": normalized_name,
                "provider": self._provider,
                "model": self._model,
                "saved_at": str(provider_meta.get("saved_at") or datetime.now(timezone.utc).isoformat()),
                "key": str(key or "").strip(),
                "artifact_filename": artifact_path.name,
                "artifact_extension": artifact_path.suffix,
                "artifact_format": _prompt_cache_artifact_format(provider),
                "provider_meta": provider_meta,
            }
            token_count = _prompt_cache_export_token_count(provider_meta.get("token_count"))
            if token_count is not None:
                record["token_count"] = token_count
            quantized = provider_meta.get("quantized")
            if quantized is not None:
                record["quantized"] = quantized
            meta_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            return {
                "supported": True,
                "ok": True,
                "operation": "export",
                "local_only": True,
                "provider": self._provider,
                "model": self._model,
                "name": normalized_name,
                "artifact_filename": artifact_path.name,
                "artifact_path": str(artifact_path),
                "meta_path": str(meta_path),
                "capabilities": _prompt_cache_capabilities_dict(provider),
                "meta": record,
                "provider_response": provider_result if isinstance(provider_result, dict) else {"result": provider_result},
            }
        except Exception as e:
            return _prompt_cache_error_payload(provider, operation="export", error=e)

    def prompt_cache_import(
        self,
        *,
        name: str,
        key: Optional[str] = None,
        make_default: bool = True,
        clear_existing: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        provider = getattr(self, "_llm", None)
        target_provider = str(kwargs.pop("provider", "") or "").strip().lower()
        target_model = str(kwargs.pop("model", "") or "").strip()
        if provider is None:
            return _prompt_cache_unsupported_payload(
                provider,
                operation="import",
                error="Runtime LLM client has no provider instance",
            )
        if (target_provider and target_provider != self._provider) or (target_model and target_model != self._model):
            return {
                "supported": False,
                "operation": "import",
                "code": "invalid_target",
                "error": (
                    "Local prompt-cache import is bound to the active runtime provider/model "
                    f"{self._provider}/{self._model}; requested "
                    f"{target_provider or self._provider}/{target_model or self._model}."
                ),
                "capabilities": _prompt_cache_capabilities_dict(provider),
            }
        if not _prompt_cache_supports(provider, "load") or not hasattr(provider, "prompt_cache_load"):
            return _prompt_cache_unsupported_payload(
                provider,
                operation="import",
                error="Provider does not support host-local prompt cache import",
            )
        export_root_dir = _coerce_prompt_cache_export_root_dir(
            kwargs.pop("prompt_cache_export_root_dir", self._prompt_cache_export_root_dir)
        )
        listed = _list_prompt_cache_exports_local(
            root_dir=export_root_dir,
            provider=provider,
            provider_name=self._provider,
            model=self._model,
        )
        if not listed.get("ok"):
            return listed
        normalized_name = _prompt_cache_export_name(name)
        items = [dict(item) for item in list(listed.get("items") or []) if isinstance(item, dict)]
        matches = [item for item in items if str(item.get("name") or "").strip() == normalized_name]
        if not matches:
            return {
                "supported": False,
                "operation": "import",
                "code": "not_found",
                "error": f"Prompt cache export '{normalized_name}' was not found for {self._provider}/{self._model}.",
                "capabilities": _prompt_cache_capabilities_dict(provider),
            }
        record = matches[0]
        artifact_path = Path(str(record.get("artifact_path") or "")).expanduser()
        if not artifact_path.exists():
            return {
                "supported": False,
                "operation": "import",
                "code": "not_found",
                "error": f"Prompt cache export artifact is missing: {artifact_path}",
                "capabilities": _prompt_cache_capabilities_dict(provider),
            }
        warnings: List[str] = []
        requested_key = str(key).strip() if isinstance(key, str) and key.strip() else None
        try:
            if clear_existing:
                if _prompt_cache_supports(provider, "clear") and hasattr(provider, "prompt_cache_clear"):
                    probe_key = f"import-probe:{uuid.uuid4().hex[:12]}"
                    provider.prompt_cache_load(
                        str(artifact_path),
                        key=probe_key,
                        make_default=False,
                    )
                    try:
                        provider.prompt_cache_clear(None)
                    except Exception as clear_error:
                        warnings.append(f"best-effort clear_existing failed: {clear_error}")
                else:
                    warnings.append("clear_existing requested, but this provider does not support prompt cache clear.")
            provider_result = provider.prompt_cache_load(
                str(artifact_path),
                key=requested_key,
                make_default=bool(make_default),
            )
            if isinstance(provider_result, dict) and provider_result.get("supported") is False:
                return provider_result
            effective_key = requested_key
            if isinstance(provider_result, dict):
                provider_key = provider_result.get("key")
                if isinstance(provider_key, str) and provider_key.strip():
                    effective_key = provider_key.strip()
            out = {
                "supported": True,
                "ok": True,
                "operation": "import",
                "local_only": True,
                "provider": self._provider,
                "model": self._model,
                "name": normalized_name,
                "key": effective_key,
                "make_default": bool(make_default),
                "clear_existing": bool(clear_existing),
                "artifact_filename": artifact_path.name,
                "artifact_path": str(artifact_path),
                "capabilities": _prompt_cache_capabilities_dict(provider),
                "meta": record.get("meta") if isinstance(record.get("meta"), dict) else record,
                "provider_response": provider_result if isinstance(provider_result, dict) else {"result": provider_result},
            }
            if warnings:
                out["warnings"] = warnings
            return out
        except Exception as e:
            payload = _prompt_cache_error_payload(provider, operation="import", error=e)
            if warnings:
                payload["warnings"] = warnings
            return payload

    def upsert_text_bloc(
        self,
        *,
        path: str,
        content: str,
        sha256: Optional[str] = None,
        content_sha256: Optional[str] = None,
        media_type: str = "text",
        size_bytes: Optional[int] = None,
        mtime_ns: Optional[int] = None,
        format: Optional[str] = None,
        estimated_tokens: Optional[int] = None,
        relpath_base: Optional[str] = None,
        summary: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        root_dir = kwargs.pop("bloc_root_dir", self._bloc_root_dir)
        return _upsert_text_bloc_local(
            root_dir=root_dir,
            path=path,
            content=content,
            sha256=sha256,
            content_sha256=content_sha256,
            media_type=media_type,
            size_bytes=size_bytes,
            mtime_ns=mtime_ns,
            format=format,
            estimated_tokens=estimated_tokens,
            relpath_base=relpath_base,
            summary=summary,
            keywords=keywords,
        )

    def get_bloc_record(
        self,
        *,
        sha256: Optional[str] = None,
        bloc_id: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        root_dir = kwargs.pop("bloc_root_dir", self._bloc_root_dir)
        return _get_bloc_record_local(root_dir=root_dir, sha256=sha256, bloc_id=bloc_id)

    def list_blocs(
        self,
        *,
        sha256: Optional[str] = None,
        bloc_id: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        root_dir = kwargs.pop("bloc_root_dir", self._bloc_root_dir)
        return _list_blocs_local(root_dir=root_dir, sha256=sha256, bloc_id=bloc_id)

    def get_bloc_kv_manifest(
        self,
        *,
        sha256: Optional[str] = None,
        bloc_id: Optional[int] = None,
        artifact_path: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        root_dir = kwargs.pop("bloc_root_dir", self._bloc_root_dir)
        return _get_bloc_kv_manifest_local(
            provider=getattr(self, "_llm", None),
            model=self._model,
            root_dir=root_dir,
            sha256=sha256,
            bloc_id=bloc_id,
            artifact_path=artifact_path,
        )

    def ensure_bloc_kv_artifact(
        self,
        *,
        sha256: Optional[str] = None,
        bloc_id: Optional[int] = None,
        artifact_path: Optional[str] = None,
        force_rebuild: bool = False,
        debug: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        root_dir = kwargs.pop("bloc_root_dir", self._bloc_root_dir)
        return _ensure_bloc_kv_artifact_local(
            provider=getattr(self, "_llm", None),
            model=self._model,
            root_dir=root_dir,
            sha256=sha256,
            bloc_id=bloc_id,
            artifact_path=artifact_path,
            force_rebuild=force_rebuild,
            debug=debug,
        )

    def load_bloc_kv_artifact(
        self,
        *,
        sha256: Optional[str] = None,
        bloc_id: Optional[int] = None,
        artifact_path: Optional[str] = None,
        stable_cache_key: Optional[str] = None,
        key: Optional[str] = None,
        make_default: bool = False,
        force_rebuild: bool = False,
        debug: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        root_dir = kwargs.pop("bloc_root_dir", self._bloc_root_dir)
        return _load_bloc_kv_artifact_local(
            provider=getattr(self, "_llm", None),
            model=self._model,
            root_dir=root_dir,
            sha256=sha256,
            bloc_id=bloc_id,
            artifact_path=artifact_path,
            stable_cache_key=stable_cache_key,
            key=key,
            make_default=make_default,
            force_rebuild=force_rebuild,
            debug=debug,
        )

    def list_bloc_kv_artifacts(
        self,
        *,
        sha256: Optional[str] = None,
        bloc_id: Optional[int] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        _ = (provider, model)
        root_dir = kwargs.pop("bloc_root_dir", self._bloc_root_dir)
        return _list_bloc_kv_artifacts_local(root_dir=root_dir, sha256=sha256, bloc_id=bloc_id)

    def delete_bloc_kv_artifact(
        self,
        *,
        sha256: Optional[str] = None,
        bloc_id: Optional[int] = None,
        artifact_path: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        clear_loaded: bool = False,
        force: bool = False,
        dry_run: bool = False,
        debug: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        _ = (provider, model)
        root_dir = kwargs.pop("bloc_root_dir", self._bloc_root_dir)
        return _delete_bloc_kv_artifact_local(
            provider=getattr(self, "_llm", None),
            root_dir=root_dir,
            sha256=sha256,
            bloc_id=bloc_id,
            artifact_path=artifact_path,
            clear_loaded=clear_loaded,
            force=force,
            dry_run=dry_run,
            debug=debug,
        )

    def prune_bloc_kv_artifacts(
        self,
        *,
        sha256: Optional[str] = None,
        bloc_id: Optional[int] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        clear_loaded: bool = False,
        force: bool = False,
        dry_run: bool = False,
        debug: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        _ = (provider, model)
        root_dir = kwargs.pop("bloc_root_dir", self._bloc_root_dir)
        return _prune_bloc_kv_artifacts_local(
            provider=getattr(self, "_llm", None),
            root_dir=root_dir,
            sha256=sha256,
            bloc_id=bloc_id,
            clear_loaded=clear_loaded,
            force=force,
            dry_run=dry_run,
            debug=debug,
        )

    def delete_bloc(
        self,
        *,
        sha256: Optional[str] = None,
        bloc_id: Optional[int] = None,
        delete_kv: bool = True,
        clear_loaded: bool = False,
        force: bool = False,
        dry_run: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        root_dir = kwargs.pop("bloc_root_dir", self._bloc_root_dir)
        return _delete_bloc_local(
            provider=getattr(self, "_llm", None),
            root_dir=root_dir,
            sha256=sha256,
            bloc_id=bloc_id,
            delete_kv=delete_kv,
            clear_loaded=clear_loaded,
            force=force,
            dry_run=dry_run,
        )


class MultiLocalAbstractCoreLLMClient:
    """Local AbstractCore client with per-request provider/model routing.

    This keeps the same `generate(...)` signature as AbstractCoreLLMClient by
    using reserved keys in `params`:
    - `_provider`: override provider for this request
    - `_model`: override model for this request
    """

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        artifact_store: Optional[Any] = None,
        bloc_root_dir: Optional[str | Path] = None,
        prompt_cache_export_root_dir: Optional[str | Path] = None,
        core_config_file: Optional[str | Path] = None,
        capability_defaults: Optional[Any] = None,
    ):
        # `_llm_kwargs` stays the single source of truth (hosts and tests read
        # and even replace it). ROUTING INVARIANT: its connection-scoped half
        # (base_url/api_key) belongs to the DEFAULT provider identity only, and
        # `_create_client` re-splits it per construction so a per-call provider
        # override can never inherit another provider's address or credential.
        self._llm_kwargs = dict(llm_kwargs or {})
        self._default_provider = provider.strip().lower()
        self._default_model = model.strip()
        self._artifact_store = artifact_store
        self._bloc_root_dir = _coerce_bloc_root_dir(bloc_root_dir)
        self._prompt_cache_export_root_dir = _coerce_prompt_cache_export_root_dir(prompt_cache_export_root_dir)
        self._core_config_file = _coerce_core_config_file(core_config_file)
        self._capability_defaults = _normalize_core_capability_defaults(capability_defaults)
        self._capability_defaults_explicit = capability_defaults is not None
        self._clients: Dict[Tuple[str, str], LocalAbstractCoreLLMClient] = {}
        self._override_clients: Dict[Tuple[str, ...], LocalAbstractCoreLLMClient] = {}
        self._capability_residency_core = None
        self._capability_residency_core_lock = threading.Lock()
        self._provider_endpoint_profile_resolver = None
        self._locked_model_residency: set = set()
        # Residency claims (core `process_residency`): what this client pools,
        # locks or is building, so a process-wide eject by ANY owner (another
        # user's service, an entity runtime, the core server) skips it.
        self._building: Dict[Tuple[str, str], int] = {}
        self._pending_ejects: Dict[str, Dict[str, Any]] = {}
        self._last_switch_ejects: Dict[str, Any] = {}
        _pr = _core_process_residency()
        if _pr is not None:
            _pr.register_claimant(self)
        # Fresh-install guard (release gap 1, gateway c5878, 2026-07-27): a
        # brand-new install has NO provider configured anywhere. Eagerly
        # building the default client here crashed the whole runtime at
        # construction ("Unknown provider: "), so the shipped catalog could
        # never even load. With both fields blank we skip the eager build;
        # calls that arrive WITH a provider (payload override, run vars)
        # work immediately, and calls with none fail at call time with a
        # message that says what to configure.
        if self._default_provider or self._default_model:
            try:
                self._default_client = self._get_client(self._default_provider, self._default_model)
            except Exception as exc:
                # SAME GUARD, SECOND SHAPE (2026-08-01). The blank-pair case
                # above stopped being the fresh install the day the recommended
                # seed started WRITING a default: a new machine now boots with
                # `lmstudio/qwen/qwen3.5-9b` configured and those weights not
                # downloaded yet. Eagerly building that client raised
                # ModelNotFoundError at construction and the shipped catalog
                # could never load -- release gap 1 reopened wearing a
                # different error.
                #
                # The eager build is a WARM-UP, never a requirement. A default
                # that cannot be built yet is deferred to call time, where the
                # same failure is raised WITH attribution (which route
                # configured it, how to change it, how to fetch the weights) by
                # `_get_client`. Nothing is swallowed: a run still fails, and it
                # fails legibly. What no longer happens is a whole host refusing
                # to start because one model has not been downloaded.
                self._default_client = None
                logger.warning(
                    f"#FALLBACK: the configured default {self._default_provider}/{self._default_model} "
                    f"could not be prepared at startup ({exc}) - the runtime still serves per-call "
                    "provider choices; a call that uses the default will report this with the route "
                    "that configured it"
                )
        else:
            self._default_client = None
            logger.warning(
                "#FALLBACK: no default provider/model configured - the runtime serves "
                "per-call provider choices only; calls without one will ask for configuration"
            )

        # Provide a stable underlying LLM for components that need one (e.g. summarizer).
        self._llm = getattr(self._default_client, "_llm", None)

    def default_prompt_cache_identity(self) -> Tuple[Optional[str], Optional[str]]:
        return self._default_provider, self._default_model

    def set_default_provider_model(
        self,
        *,
        provider: Optional[str],
        model: Optional[str],
        llm_kwargs: Optional[Dict[str, Any]] = None,
        capability_defaults: Optional[Any] = _UNSET,
    ) -> bool:
        """Re-point the pool's DEFAULT identity in place. Returns True if it changed.

        A default is a DEFAULT: the execution host resolves it once at
        construction, but the operator may change it at any moment from the
        console. Baking it in forever meant a console change was invisible
        until the host reloaded its bundles (defect 2026-07-31, reproduced
        live: the run right after a console default change still used the
        previous provider/model). This is the in-place refresh the host calls
        when its capability defaults are rewritten.

        Only the DEFAULT identity moves. Per-call provider/model overrides are
        resolved per call and never touch this, so an app override can never be
        clobbered by a default change.

        ROUTING INVARIANT (see `_create_client`): connection-scoped kwargs
        (base_url/api_key/...) belong to the default provider identity ONLY.
        A new default therefore REPLACES that half wholesale -- inheriting the
        previous endpoint/credential is exactly the misroute the 2026-07-31
        isolation fix closed. Pooled clients built under the old default are
        evicted for the same reason; per-override clients are keyed by their
        own endpoint+credential and stay valid.
        """

        provider_s = str(provider or "").strip().lower()
        model_s = str(model or "").strip()

        if llm_kwargs is None:
            # Nothing said about the endpoint -> keep the pool exactly as it is
            # (this is the capability-defaults-only refresh path).
            next_kwargs: Dict[str, Any] = dict(getattr(self, "_llm_kwargs", None) or {})
        else:
            shared_kwargs, _old_connection = _split_connection_scoped_llm_kwargs(getattr(self, "_llm_kwargs", None))
            new_shared, new_connection = _split_connection_scoped_llm_kwargs(llm_kwargs)
            # The caller owns the default endpoint (the host's endpoint
            # profile): its connection half REPLACES the old one, its
            # provider-agnostic knobs win, and knobs it does not mention
            # (timeout, read_idle_timeout_s, tracing) are preserved.
            next_kwargs = dict(shared_kwargs)
            next_kwargs.update(new_shared)
            next_kwargs.update(new_connection)

        capability_changed = False
        if capability_defaults is not _UNSET:
            normalized_defaults = _normalize_core_capability_defaults(capability_defaults)
            explicit_defaults = capability_defaults is not None
            capability_changed = (
                normalized_defaults != getattr(self, "_capability_defaults", {})
                or explicit_defaults != getattr(self, "_capability_defaults_explicit", False)
            )
            if capability_changed:
                self._capability_defaults = normalized_defaults
                self._capability_defaults_explicit = explicit_defaults

        identity_changed = (
            provider_s != str(getattr(self, "_default_provider", "") or "")
            or model_s != str(getattr(self, "_default_model", "") or "")
            or next_kwargs != dict(getattr(self, "_llm_kwargs", None) or {})
        )
        if not identity_changed and not capability_changed:
            return False

        self._llm_kwargs = next_kwargs
        self._default_provider = provider_s
        self._default_model = model_s
        # Evict the shared pool: entries built for the OLD default carry its
        # connection kwargs, and an entry for the NEW default provider was
        # built WITHOUT them (it was not the default then). Both are wrong now.
        # Rebuild is lazy, so this costs one construction per identity in use.
        #
        # LOCK EXEMPTION (review fix): a LOCKED pair's pooled client holds the
        # resident weights (in-process providers would be GC'd — a silent lock
        # bypass), so it survives the eviction. The one structural exception:
        # a locked pair that IS the new default identity — keeping its stale
        # entry would serve default traffic with the old connection kwargs
        # (the 2026-07-31 misroute), so it is evicted LOUDLY and its dangling
        # flag cleared so a re-lock binds the rebuilt client.
        locked_pairs = _local_locked_residency_pairs(self)
        kept_clients: Dict[Tuple[str, str], LocalAbstractCoreLLMClient] = {}
        previous_clients = dict(getattr(self, "_clients", {}) or {})
        previous_overrides = dict(getattr(self, "_override_clients", {}) or {}) if capability_changed else {}
        for pool_key, pool_client in previous_clients.items():
            if pool_key not in locked_pairs:
                continue
            if pool_key == (provider_s, model_s):
                locked_pairs.discard(pool_key)
                logger.warning(
                    f"🔒 pool eviction dropped a LOCKED pair {pool_key[0]}/{pool_key[1]}: it is the "
                    "new default identity and must be rebuilt with the new connection kwargs; "
                    "its lock was cleared — re-lock to pin the rebuilt client"
                )
                continue
            kept_clients[pool_key] = pool_client
            logger.info(f"🔒 pool eviction skipped (locked): {pool_key[0]}/{pool_key[1]}")
        self._clients = kept_clients
        if capability_changed:
            kept_overrides: Dict[Tuple[str, ...], LocalAbstractCoreLLMClient] = {}
            for override_key, override_client in dict(getattr(self, "_override_clients", {}) or {}).items():
                if (override_key[0], override_key[1]) not in locked_pairs:
                    continue
                kept_overrides[override_key] = override_client
                logger.info(
                    f"🔒 override eviction skipped (locked): {override_key[0]}/{override_key[1]}"
                )
            self._override_clients = kept_overrides
        if capability_changed:
            # The capability routes changed: the residency core re-reads them.
            # Its resident engines (TTS/STT/image/music) are unloaded first --
            # dropping the core used to leave them in memory, unreachable.
            self._retire_capability_residency_core()
        # CLEAN SWITCH (M1 finding, 2026-09-25): evicting a pool entry only
        # dropped a Python reference. An in-process model (MLX / HuggingFace)
        # the new pool no longer uses stayed resident -- 17.36 GB measured
        # after a console default switch -- until someone ejected it by hand.
        # Ejected BEFORE the new default is built, so a switch between two
        # large models never holds both at once.
        dropped: List[Tuple[Tuple[str, str], Any]] = [
            (key, getattr(client, "_llm", None)) for key, client in previous_clients.items()
            if self._clients.get(key) is not client
        ]
        dropped.extend(
            ((key[0], key[1]), getattr(client, "_llm", None)) for key, client in previous_overrides.items()
            if self._override_clients.get(key) is not client
        )
        del previous_clients, previous_overrides
        self._eject_models_dropped_by_switch(dropped)
        del dropped
        if provider_s or model_s:
            self._default_client = self._get_client(provider_s, model_s)
        else:
            self._default_client = None
        self._llm = getattr(self._default_client, "_llm", None)
        return True

    def _eject_models_dropped_by_switch(self, dropped: List[Tuple[Tuple[str, str], Any]]) -> None:
        """Eject, process-wide, every in-process model a default switch dropped
        from the pool that NO owner in the process still claims (this client,
        other users' clients, entity runtimes, the core server's managed
        runtimes; see core `process_residency.eject_unclaimed`). A model still
        generating on a dropped instance is ejected when that generation ends
        -- a switch never cancels a running call. Reports land in
        `self._last_switch_ejects` and the log."""
        by_pair: Dict[Tuple[str, str], List[Any]] = {}
        for (provider_s, model_s), instance in dropped:
            if str(provider_s) not in _PROCESS_RESIDENCY_PROVIDERS:
                continue
            by_pair.setdefault((provider_s, model_s), []).append(instance)
        reports: Dict[str, Any] = {}
        self._last_switch_ejects = reports
        for (provider_s, model_s), instances in by_pair.items():
            busy = [i for i in instances if i is not None and _provider_inflight_count(i) > 0]
            label = f"{provider_s}/{model_s}"
            if busy:
                reports[label] = {"deferred": True, "reason": "waiting for the in-flight call"}
                if label in self._pending_ejects:
                    continue  # one waiter per model
                self._pending_ejects[label] = {"provider": provider_s, "model": model_s,
                                               "reason": "waiting for the in-flight call", "since": time.time()}
                logger.info(f"default switch: {label} is still generating; it is unloaded when that call ends")
                threading.Thread(
                    target=self._eject_after_inflight,
                    args=(provider_s, model_s, [weakref.ref(i) for i in busy], reports),
                    name=f"switch-eject:{label}",
                    daemon=True,
                ).start()
                continue
            reports[label] = self._eject_dropped_pair(provider_s, model_s)

    def _eject_dropped_pair(self, provider_s: str, model_s: str, *, reason: str = "default_switch") -> Dict[str, Any]:
        """Eject a model this client dropped, unless ANY owner in the process
        (this client included: pool, override, lock, build, default) still
        claims it. The claim check and the eject run under core's process
        residency lock, so a rebuild cannot slip in between."""
        label = f"{provider_s}/{model_s}"
        pr = _core_process_residency()
        if pr is None:
            logger.warning(f"{reason}: {label} not unloaded: this AbstractCore cannot tell which owners still use it")
            return {"ok": True, "skipped": True,
                    "reason": "this AbstractCore has no residency claim registry; upgrade it to unload switched-away models"}
        report = pr.eject_unclaimed(provider_s, model_s, reason=reason) or {"ok": True, "holders_found": 0}
        if report.get("skipped"):
            logger.info(f"{reason}: {label} kept: {report.get('reason')}")
        elif report.get("ok") is False:
            logger.warning(f"{reason}: {label} is still resident after the eject: {report.get('error') or report.get('residual')}")
        else:
            logger.info(f"{reason}: unloaded {label} from {int(report.get('holders_found') or 0)} holder(s) no owner uses any more")
        return report

    def _eject_after_inflight(self, provider_s: str, model_s: str, refs: List[Any], reports: Dict[str, Any]) -> None:
        label = f"{provider_s}/{model_s}"
        try:
            # Ends when the call ends or the instance is collected.
            while any((r() is not None and _provider_inflight_count(r()) > 0) for r in refs):
                time.sleep(0.5)
            reports[label] = self._eject_dropped_pair(provider_s, model_s)
        except Exception as exc:  # noqa: BLE001 - the report must say it failed
            reports[label] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            logger.warning(f"default switch: deferred unload of {label} failed: {exc}")
        finally:
            self._pending_ejects.pop(label, None)

    def _retire_capability_residency_core(self) -> None:
        core = getattr(self, "_capability_residency_core", None)
        self._capability_residency_core = None
        if core is None:
            return
        for facade_name in ("vision", "voice", "audio", "music"):
            try:
                facade = getattr(core, facade_name, None)
                lister = getattr(facade, "list_resident_models", None) or getattr(facade, "list_loaded_models", None)
                unloader = getattr(facade, "unload_resident_model", None)
                if not callable(lister) or not callable(unloader):
                    continue
                for record in list(lister({}) or []):
                    if not isinstance(record, dict) or record.get("resident") is False:
                        continue
                    selector = {k: record.get(k) for k in ("task", "provider", "model", "load_id") if record.get(k)}
                    unloader(selector)
                    logger.info(f"capability routes changed: unloaded {facade_name} model {record.get('model')}")
            except Exception as exc:  # noqa: BLE001 - one facade must not keep the others resident
                logger.warning(f"capability routes changed: could not unload the {facade_name} models: {exc}")

    def set_capability_defaults(self, capability_defaults: Optional[Any]) -> bool:
        """Refresh the non-text capability routes (image/voice/music/...) in place.

        Same reason as `set_default_provider_model`: these routes are the
        operator's console defaults and must not be frozen at host construction.
        """
        return self.set_default_provider_model(
            provider=getattr(self, "_default_provider", None),
            model=getattr(self, "_default_model", None),
            capability_defaults=capability_defaults,
        )

    def _capability_lookup_model(self, model_name: Optional[str]) -> str:
        """The model a capability lookup is about: the one named, else the
        configured default's NAME (metadata needs no weights). With neither,
        refuse with the fresh-install message -- the one case it is true for."""
        target = str(model_name or getattr(self, "_default_model", "") or "").strip()
        if not target:
            raise no_default_provider_configured_error(
                core_config_file=getattr(self, "_core_config_file", None),
                what="model capability lookup",
            )
        return target

    def _create_client(
        self,
        provider: str,
        model: str,
        *,
        llm_kwargs_override: Optional[Dict[str, Any]] = None,
    ) -> LocalAbstractCoreLLMClient:
        key = (provider.strip().lower(), model.strip())
        # ROUTING INVARIANT (defect 2026-07-31): only the DEFAULT provider
        # inherits the pool's connection-scoped kwargs. A client built for any
        # other provider starts from the provider-agnostic half and gets its
        # endpoint/credential from the explicit per-call override, or from that
        # provider's own configuration inside AbstractCore. Inheriting them was
        # what made every `lmstudio`/`ollama`/`openai` pin land on the gateway's
        # default endpoint profile.
        shared_kwargs, default_connection_kwargs = _split_connection_scoped_llm_kwargs(
            getattr(self, "_llm_kwargs", None)
        )
        llm_kwargs = dict(shared_kwargs)
        if key[0] == str(getattr(self, "_default_provider", "") or "").strip().lower():
            llm_kwargs.update(default_connection_kwargs)
        elif default_connection_kwargs:
            logger.info(
                f"provider override {key[0]!r} does not inherit the default provider "
                f"{str(getattr(self, '_default_provider', '') or '')!r} connection settings "
                f"({', '.join(sorted(default_connection_kwargs))}); it resolves its own endpoint"
            )
        if llm_kwargs_override:
            llm_kwargs.update(dict(llm_kwargs_override))
        # POOLED instances serve every run/session of this runtime, so a construction-time
        # instance-default cache key (create_llm's prompt_cache_key convenience for
        # instance-per-session callers) would stamp one session's identity onto all traffic.
        # Session-scoped keys are injected PER CALL by the LLM_CALL effect handler instead.
        if "prompt_cache_key" in llm_kwargs:
            llm_kwargs.pop("prompt_cache_key", None)
            logger.warning(
                "#FALLBACK: dropping construction-time prompt_cache_key from pooled LLM client "
                "kwargs (pooled instances serve many sessions; per-call keys are injected by the "
                "runtime instead)"
            )
        extra_kwargs: Dict[str, Any] = {
            name: value
            for name, value in {
                "bloc_root_dir": self._bloc_root_dir,
                "prompt_cache_export_root_dir": self._prompt_cache_export_root_dir,
                "core_config_file": self._core_config_file,
                "capability_defaults": self._capability_defaults
                if getattr(self, "_capability_defaults_explicit", bool(self._capability_defaults)) else None,
            }.items()
            if value is not None
        }
        optional_names = tuple(extra_kwargs.keys())
        last_exc: Optional[TypeError] = None
        client: Optional[LocalAbstractCoreLLMClient] = None
        tried_variants: set[Tuple[str, ...]] = set()
        while True:
            variant_key = tuple(sorted(extra_kwargs.keys()))
            if variant_key in tried_variants:
                break
            tried_variants.add(variant_key)
            try:
                client = LocalAbstractCoreLLMClient(
                    provider=key[0],
                    model=key[1],
                    llm_kwargs=llm_kwargs,
                    artifact_store=self._artifact_store,
                    **extra_kwargs,
                )
                break
            except TypeError as exc:
                last_exc = exc
                message = str(exc)
                unsupported_name = next((name for name in optional_names if name in message), None)
                if unsupported_name is None or unsupported_name not in extra_kwargs:
                    raise
                extra_kwargs = {name: value for name, value in extra_kwargs.items() if name != unsupported_name}
        if client is None:
            if last_exc is not None:
                raise last_exc
            raise TypeError("Failed to construct LocalAbstractCoreLLMClient")
        if self._core_config_file or getattr(self, "_capability_defaults_explicit", bool(self._capability_defaults)):
            _attach_core_execution_context_to_client(
                client,
                core_config_file=self._core_config_file,
                capability_defaults=self._capability_defaults
                if getattr(self, "_capability_defaults_explicit", bool(self._capability_defaults)) else None,
            )
        resolver = getattr(self, "_provider_endpoint_profile_resolver", None)
        if callable(resolver):
            _attach_provider_endpoint_profile_resolver_to_client(client, resolver)
        pool_on_token = getattr(self, "_pool_on_token", None)
        if callable(pool_on_token):
            client.set_on_token(pool_on_token)
        return client

    def set_on_token(self, callback: Optional[Any]) -> None:
        """Pool-wide token callback (code seat c1009 ask 3): registers on every
        pooled client — existing AND future (per-request provider/model
        overrides create clients lazily; without the fan-out they would
        silently not stream). Same contract as the per-client setter."""
        self._pool_on_token = callback if callable(callback) else None
        for client in list(getattr(self, "_clients", {}).values()):
            client.set_on_token(self._pool_on_token)
        for client in list(getattr(self, "_override_clients", {}).values()):
            client.set_on_token(self._pool_on_token)
        default_client = getattr(self, "_default_client", None)
        if default_client is not None:
            default_client.set_on_token(self._pool_on_token)

    def _get_client(
        self,
        provider: str,
        model: str,
        *,
        llm_kwargs_override: Optional[Dict[str, Any]] = None,
    ) -> LocalAbstractCoreLLMClient:
        if not str(provider or "").strip():
            # Fresh-install path: no default was configured and this call
            # brought no provider of its own. Say what to configure instead
            # of crashing with core's bare "Unknown provider: ".
            raise no_default_provider_configured_error(
                core_config_file=getattr(self, "_core_config_file", None),
            )
        key = (provider.strip().lower(), model.strip())
        if llm_kwargs_override:
            base_url = str(llm_kwargs_override.get("base_url") or "").strip()
            api_key = str(llm_kwargs_override.get("api_key") or "").strip()
            api_key_fp = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16] if api_key else ""
            # Speculation is part of the key because it is part of the LOAD:
            # a provider built for depth 3 cannot serve a depth-5 request, and
            # one built without speculation cannot serve either.
            spec_fp = _speculation_fingerprint(llm_kwargs_override.get("speculation"))
            override_key = (key[0], key[1], base_url, api_key_fp, spec_fp)
            override_clients = getattr(self, "_override_clients", None)
            if override_clients is None:
                override_clients = {}
                self._override_clients = override_clients
            client = override_clients.get(override_key)
            if client is None:
                with self._building_claim(key):
                    client = self._create_client(key[0], key[1], llm_kwargs_override=llm_kwargs_override)
                    with _core_residency_lock():
                        override_clients[override_key] = client
            return client
        client = self._clients.get(key)
        if client is None:
            try:
                with self._building_claim(key):
                    client = self._create_client(key[0], key[1])
                    with _core_residency_lock():
                        self._clients[key] = client
            except DefaultRouteProviderError:
                raise
            except Exception as exc:
                # ATTRIBUTION. When the pair that failed IS the host default,
                # the operator did not type it on this call -- they set it in
                # the config store, possibly weeks ago through the other entry
                # point. Say so, and say where to change it.
                if key == (
                    str(getattr(self, "_default_provider", "") or "").strip().lower(),
                    str(getattr(self, "_default_model", "") or "").strip(),
                ):
                    raise default_route_provider_error(
                        exc,
                        provider=key[0],
                        model=key[1],
                        core_config_file=getattr(self, "_core_config_file", None),
                    ) from exc
                raise
        return client

    @contextmanager
    def _building_claim(self, key: Tuple[str, str]):
        """Claim `key` while its client is being built: an eject that runs
        meanwhile (another client's switch, the core server) must not unload
        the weights this build is loading. Released after the pool insert."""
        with _core_residency_lock():
            building = self.__dict__.setdefault("_building", {})
            building[key] = int(building.get(key, 0)) + 1
        try:
            yield
        finally:
            with _core_residency_lock():
                n = int(self._building.get(key, 1)) - 1
                if n > 0:
                    self._building[key] = n
                else:
                    self._building.pop(key, None)

    def residency_claims(self) -> List[Dict[str, Any]]:
        """Every (provider, model) this client still wants resident: pooled,
        per-override, locked, being built, or its default."""
        owner = f"runtime client {id(self):x}"
        locked = set(_local_locked_residency_pairs(self))
        claims: List[Dict[str, Any]] = []
        for p, m in list(getattr(self, "_clients", {}) or {}):
            claims.append({"provider": p, "model": m, "locked": (p, m) in locked, "kind": "pool", "owner": owner})
        for k in list(getattr(self, "_override_clients", {}) or {}):
            claims.append({"provider": k[0], "model": k[1], "locked": (k[0], k[1]) in locked, "kind": "override",
                           "owner": owner})
        for p, m in locked:
            claims.append({"provider": p, "model": m, "locked": True, "kind": "lock", "owner": owner})
        for p, m in list((getattr(self, "_building", {}) or {}).keys()):
            claims.append({"provider": p, "model": m, "locked": False, "kind": "building", "owner": owner})
        if getattr(self, "_default_provider", None) and getattr(self, "_default_model", None):
            claims.append({"provider": self._default_provider, "model": self._default_model, "locked": False,
                           "kind": "default", "owner": owner})
        return claims

    def _drop_failed_load_client(self, key: Tuple[str, str]) -> None:
        """A load failed: drop the pool entry AND free what the failed load may
        have left in memory (e.g. the main weights loaded, the drafter failed)
        unless another owner claims the model."""
        with _core_residency_lock():
            self._clients.pop(key, None)
        if str(key[0]) in _PROCESS_RESIDENCY_PROVIDERS:
            label = f"{key[0]}/{key[1]}"
            report = self._eject_dropped_pair(key[0], key[1], reason="failed_load")
            self._last_switch_ejects = {**(getattr(self, "_last_switch_ejects", {}) or {}),
                                        label: {**report, "reason": report.get("reason") or "failed_load"}}

    def residency_eject_diagnostics(self) -> Dict[str, Any]:
        """Pending and last switch/failed-load ejects, for listings (the console
        shows "will be unloaded when its call ends" / "unload failed: ...")."""
        return {
            "pending_ejects": [dict(v) for v in list((getattr(self, "_pending_ejects", {}) or {}).values())],
            "last_switch_ejects": [
                _eject_report_summary(label, report)
                for label, report in list((getattr(self, "_last_switch_ejects", {}) or {}).items())
                if isinstance(report, dict)
            ],
        }

    def set_provider_endpoint_profile_resolver(self, resolver: Any) -> None:
        self._provider_endpoint_profile_resolver = resolver
        _attach_provider_endpoint_profile_resolver_to_client(self, resolver)
        for client in list(getattr(self, "_clients", {}).values()):
            _attach_provider_endpoint_profile_resolver_to_client(client, resolver)
        for client in list(getattr(self, "_override_clients", {}).values()):
            _attach_provider_endpoint_profile_resolver_to_client(client, resolver)
        default_client = getattr(self, "_default_client", None)
        if default_client is not None:
            _attach_provider_endpoint_profile_resolver_to_client(default_client, resolver)
        self._llm = getattr(default_client, "_llm", None)

    def get_provider_instance(self, *, provider: str, model: str) -> Any:
        """Return the underlying AbstractCore provider instance for (provider, model)."""
        client = self._get_client(str(provider or ""), str(model or ""))
        return getattr(client, "_llm", None)

    def list_loaded_clients(self) -> List[Tuple[str, str]]:
        """Return (provider, model) pairs loaded in this process (best-effort)."""
        out = list(getattr(self, "_clients", {}).keys())
        # Read the pair positionally: the override key grows an axis whenever a
        # new construction-time property has to select the instance (base_url,
        # api key, now the speculation request). Destructuring the whole tuple
        # made this raise `too many values to unpack` the moment it did.
        for key in getattr(self, "_override_clients", {}).keys():
            if not isinstance(key, tuple) or len(key) < 2:
                continue
            pair = (key[0], key[1])
            if pair not in out:
                out.append(pair)
        return out

    def get_model_residency_capabilities(self, **kwargs: Any) -> Dict[str, Any]:
        _ = kwargs
        return _local_model_residency_capabilities(
            mode="local_multi_client",
            source="abstractruntime.multilocal",
            text_loads_other_models=True,
        )

    def list_model_residency(
        self,
        *,
        task: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        result = self._list_model_residency(task=task, provider=provider, model=model, **kwargs)
        if isinstance(result, dict):
            # Switch / failed-load ejects: pending ("will be unloaded when its
            # call ends") and last outcomes ("unloaded" / "kept: in use by ..." /
            # "failed: ..."), so the console can say so.
            diagnostics = result.setdefault("diagnostics", {})
            if isinstance(diagnostics, dict):
                diagnostics.update(self.residency_eject_diagnostics())
        return result

    def _list_model_residency(
        self,
        *,
        task: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        _ = kwargs
        task_s = _residency_task_filter(task)
        if task_s is not None and task_s != "text_generation":
            return _local_capability_residency_result(
                self,
                operation="list_loaded",
                task=task_s,
                provider=provider,
                model=model,
                source="abstractruntime.multilocal",
            )

        provider_filter = str(provider or "").strip().lower()
        model_filter = str(model or "").strip()
        locked_pairs = _local_locked_residency_pairs(self)
        records: List[Dict[str, Any]] = []
        for provider_s, model_s in self.list_loaded_clients():
            if provider_filter and provider_s != provider_filter:
                continue
            if model_filter and model_s != model_filter:
                continue
            cached_client = self._clients.get((provider_s, model_s))
            records.append(
                _stamp_local_lock_state(
                    _local_residency_record(
                        provider=provider_s,
                        model=model_s,
                        default=(provider_s, model_s) == (self._default_provider, self._default_model),
                        provider_instance=getattr(cached_client, "_llm", None),
                    ),
                    locked=(provider_s, model_s) in locked_pairs,
                )
            )
        records = _merge_host_sweep_into_text_records(records, provider=provider, model=model)
        records = _merge_mlx_process_residency_into_text_records(records, provider=provider, model=model)
        records = _merge_hf_process_residency_into_text_records(records, provider=provider, model=model)
        if task_s is None:
            return _local_all_model_residency_result(
                self,
                source="abstractruntime.multilocal",
                text_records=records,
                provider=provider,
                model=model,
            )
        result = {
            "ok": True,
            "supported": True,
            "operation": "list_loaded",
            "task": "text_generation",
            "models": records,
            "diagnostics": {"source": "abstractruntime.multilocal", "count": len(records)},
        }
        return _with_local_model_residency_summary(result, operation="list_loaded", models=records)

    def load_model_residency(
        self,
        *,
        task: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        pin: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        task_s = _normalize_residency_task(task)
        if task_s != "text_generation":
            provider_s = str(provider or "").strip().lower()
            model_s = str(model or "").strip()
            return _local_capability_residency_result(
                self,
                operation="load",
                task=task_s,
                provider=provider_s,
                model=model_s,
                options=options,
                pin=pin,
                kwargs=dict(kwargs or {}),
                source="abstractruntime.multilocal",
            )
        provider_s = str(provider or self._default_provider or "").strip().lower()
        model_s = str(model or self._default_model or "").strip()
        _ = kwargs
        if not provider_s or not model_s:
            return {
                "ok": False,
                "success": False,
                "supported": True,
                "operation": "load",
                "task": "text_generation",
                "provider": provider_s or None,
                "model": model_s or None,
                "error": "model_residency load requires provider and model",
                "warnings": ["model_residency load requires provider and model"],
                "affected_models": [],
            }
        key = (provider_s, model_s)
        runtime_cache_loaded_new = key not in self._clients
        client = self._get_client(provider_s, model_s)
        before_record = _local_residency_record(
            provider=provider_s,
            model=model_s,
            default=key == (self._default_provider, self._default_model),
            provider_instance=getattr(client, "_llm", None),
            lock_owner=self,
        )
        provider_load_result: Any = None
        if before_record.get("loaded") is not True:
            provider_load_result, load_error = _load_local_provider_residency(
                provider_instance=getattr(client, "_llm", None),
                model=model_s,
                options=options,
                pin=pin,
                extra=dict(kwargs or {}),
            )
            if load_error:
                if runtime_cache_loaded_new and key != (self._default_provider, self._default_model):
                    self._drop_failed_load_client(key)
                    before_record = _local_residency_record(
                        provider=provider_s,
                        model=model_s,
                        default=False,
                        runtime_cached=False,
                        include_provider_state=False,
                        lock_owner=self,
                    )
                return _local_model_residency_load_failure(
                    operation="load",
                    task="text_generation",
                    provider=provider_s,
                    model=model_s,
                    runtime=before_record,
                    message=load_error,
                    source="abstractruntime.multilocal",
                    runtime_cache_loaded_new=runtime_cache_loaded_new,
                    provider_load_result=provider_load_result,
                )

        record = _local_residency_record(
            provider=provider_s,
            model=model_s,
            default=key == (self._default_provider, self._default_model),
            provider_instance=getattr(client, "_llm", None),
            lock_owner=self,
        )
        provider_loaded_new = bool(before_record.get("loaded") is not True and record.get("loaded") is True)
        loaded_new = bool((runtime_cache_loaded_new or provider_loaded_new) and record.get("loaded") is True)
        if record.get("loaded") is not True:
            if runtime_cache_loaded_new and key != (self._default_provider, self._default_model):
                self._drop_failed_load_client(key)
                record = _local_residency_record(
                    provider=provider_s,
                    model=model_s,
                    default=False,
                    runtime_cached=False,
                    include_provider_state=False,
                    lock_owner=self,
                )
            return _local_model_residency_load_failure(
                operation="load",
                task="text_generation",
                provider=provider_s,
                model=model_s,
                runtime=record,
                message="model_residency load completed without a loaded model",
                source="abstractruntime.multilocal",
                runtime_cache_loaded_new=runtime_cache_loaded_new,
                provider_load_result=provider_load_result,
            )
        result = {
            "ok": True,
            "supported": True,
            "operation": "load",
            "task": "text_generation",
            "loaded_new": loaded_new,
            "provider_loaded_new": provider_loaded_new,
            "runtime_cache_loaded_new": runtime_cache_loaded_new,
            "runtime": record,
            **({"provider_load_result": _jsonable(provider_load_result)} if provider_load_result is not None else {}),
            **_load_option_report(
                provider=provider_s,
                model=model_s,
                requested=_local_provider_load_options(options=options, pin=pin, extra=dict(kwargs or {})),
                provider_load_result=provider_load_result,
                provider_called=before_record.get("loaded") is not True,
            ),
            "diagnostics": {
                "source": "abstractruntime.multilocal",
                "loaded_new": loaded_new,
                "provider_loaded_new": provider_loaded_new,
                "runtime_cache_loaded_new": runtime_cache_loaded_new,
            },
        }
        return _with_local_model_residency_summary(
            result,
            operation="load",
            runtime=record,
            action="loaded" if loaded_new else "already_loaded",
            changed=loaded_new,
        )

    def unload_model_residency(
        self,
        *,
        task: Optional[str] = None,
        runtime_id: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        force: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Explicit eject (console / tray / CLI). An in-process model's eject is
        process-wide, so it first consults every OTHER owner's claims (core
        `process_residency`): another client's LOCK refuses the eject unless
        `force=True`; other clients that merely pool the model are ejected as
        well and counted in `ejected_from_other_clients`."""
        guard = _explicit_eject_guard(self, task=task, runtime_id=runtime_id, provider=provider, model=model,
                                      force=force, source='abstractruntime.multilocal')
        if guard.get("refusal") is not None:
            return guard["refusal"]
        result = self._unload_model_residency(task=task, runtime_id=runtime_id, provider=provider, model=model,
                                              options=options, force=force, **kwargs)
        if isinstance(result, dict) and guard.get("in_process"):
            result["ejected_from_other_clients"] = int(guard.get("pooled_elsewhere") or 0)
            if guard.get("forced_over_locks"):
                result["forced_over_other_client_locks"] = guard["forced_over_locks"]
        return result

    def _unload_model_residency(
        self,
        *,
        task: Optional[str] = None,
        runtime_id: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        force: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        _ = kwargs
        task_s, provider_s, model_s, capability_runtime_id = _resolve_unload_selector(
            self, task=task, runtime_id=runtime_id, provider=provider, model=model,
        )
        if task_s != "text_generation":
            return _local_capability_residency_result(
                self,
                operation="unload",
                task=task_s,
                provider=provider_s,
                model=model_s,
                options=options,
                runtime_id=capability_runtime_id,
                kwargs=dict(kwargs or {}),
                source="abstractruntime.multilocal",
            )
        if not provider_s or not model_s:
            return {
                "ok": False,
                "success": False,
                "supported": True,
                "operation": "unload",
                "task": "text_generation",
                "unloaded": False,
                "error": "model_residency unload requires runtime_id or provider/model",
                "warnings": ["model_residency unload requires runtime_id or provider/model"],
                "affected_models": [],
            }
        locked_pairs = _local_locked_residency_pairs(self)
        force_unlock_pending = False
        if (provider_s, model_s) in locked_pairs:
            if not force:
                # Locked pairs refuse plain unloads (payload, never an
                # exception); force=true unlocks first — same choke-point
                # semantics as core's /acore/models/unload 409.
                return _local_model_locked_refusal(
                    provider=provider_s,
                    model=model_s,
                    source="abstractruntime.multilocal",
                )
            # The discard is DEFERRED until the provider unload succeeds: a
            # raising provider must leave the pair resident AND still locked
            # (same ordering fix as core's force path).
            force_unlock_pending = True

        key = (provider_s, model_s)
        default_key = key == (self._default_provider, self._default_model)
        client = self._clients.get(key)
        runtime_cached_before = client is not None
        provider_instance = getattr(client, "_llm", None)
        transient_control_created = False
        transient_error: Optional[str] = None
        if provider_instance is None and _provider_supports_uncached_text_residency(provider_s):
            try:
                transient_client = self._create_client(provider_s, model_s)
                provider_instance = getattr(transient_client, "_llm", None)
                transient_control_created = provider_instance is not None
            except Exception as exc:  # noqa: BLE001
                transient_error = f"Unable to create provider control client for {provider_s}/{model_s}: {exc}"

        record = _local_residency_record(
            provider=provider_s,
            model=model_s,
            default=default_key,
            runtime_cached=runtime_cached_before,
            provider_instance=provider_instance,
            include_provider_state=provider_instance is not None,
            lock_owner=self,
        )
        if transient_error:
            result = {
                "ok": False,
                "supported": True,
                "operation": "unload",
                "task": "text_generation",
                "unloaded": False,
                "runtime_cache_unloaded": False,
                "runtime": record,
                "error": transient_error,
                "warnings": [transient_error],
                "status_hint": "warning",
                "degraded": True,
                "diagnostics": {"source": "abstractruntime.multilocal", "reason": "control_client_create_failed"},
            }
            return _with_local_model_residency_summary(
                result,
                operation="unload",
                runtime=record,
                action="unload_failed",
                changed=False,
            )

        if client is None and provider_instance is None and provider_s in _PROCESS_RESIDENCY_PROVIDERS:
            # No pool client for this pair -- but the WEIGHTS may still be in
            # this process, held by instances this pool cannot reach (an
            # override client, a boot-time summarizer, an old runtime). That is
            # exactly the row the listing reports as
            # `resident_via_other_holders`; the operator's eject must free it.
            held_rows = [
                row for row in _process_residency_rows_for(provider_s)
                if model_s in [str(n) for n in (row.get("models") or [])]
            ]
            if held_rows:
                process_eject = _process_eject_for(provider_s, model_s)
                remaining = [
                    row for row in _process_residency_rows_for(provider_s)
                    if model_s in [str(n) for n in (row.get("models") or [])]
                ]
                record_after = _local_residency_record(
                    provider=provider_s,
                    model=model_s,
                    default=default_key,
                    runtime_cached=False,
                    include_provider_state=False,
                    lock_owner=self,
                )
                _merge_process_residency_into_text_records([record_after], lane_provider=provider_s, provider=provider_s, model=model_s)
                error: Optional[str] = None
                if remaining:
                    error = (
                        "model_residency unload: the weights are still resident in this process after ejecting "
                        f"every reachable holder ({int(remaining[0].get('holders') or 0)} holder(s) still alive, "
                        f"{int(remaining[0].get('held_bytes') or 0)} bytes held)"
                    )
                elif isinstance(process_eject, dict) and process_eject.get("error"):
                    error = f"process-wide {provider_s} eject failed: {process_eject['error']}"
                result = {
                    "ok": error is None,
                    "supported": True,
                    "operation": "unload",
                    "task": "text_generation",
                    "unloaded": error is None,
                    "runtime_cache_unloaded": False,
                    "runtime": record_after,
                    **({"error": error} if error else {}),
                    "warnings": (
                        [error] if error else
                        [f"ejected {len((process_eject or {}).get('holders_unloaded') or [])} provider instance(s) that held "
                         f"{model_s} outside this runtime's pool"]
                    ),
                    **({"process_eject": _jsonable(process_eject)} if process_eject is not None else {}),
                    "diagnostics": {"source": "abstractruntime.multilocal", "reason": "process_holders_ejected"},
                }
                if error:
                    result.setdefault("status_hint", "warning")
                    result.setdefault("degraded", True)
                return _with_local_model_residency_summary(
                    result,
                    operation="unload",
                    runtime=record_after,
                    action="unload_failed" if error else "unloaded",
                    changed=error is None,
                )

        if client is None and provider_instance is None:
            result = {
                "ok": True,
                "supported": True,
                "operation": "unload",
                "task": "text_generation",
                "unloaded": False,
                "runtime_cache_unloaded": False,
                "runtime": record,
                "warnings": ["Requested local runtime was not resident."],
                "diagnostics": {"source": "abstractruntime.multilocal", "reason": "not_found"},
            }
            return _with_local_model_residency_summary(
                result,
                operation="unload",
                runtime=record,
                action="not_found",
                changed=False,
            )

        provider_unload_result: Any = None
        unload_error: Optional[str] = None
        should_call_unload = record.get("loaded") is not False or record.get("provider_residency_verified") is not True
        if should_call_unload:
            provider_unload_result, unload_error = _unload_local_provider_residency(
                provider_instance=provider_instance,
                model=model_s,
                options=options,
            )
        if should_call_unload and unload_error is None and client is not None:
            # Core's unload_model already dropped the in-provider prompt-cache
            # stores; drop the pooled client's mirrors of them too.
            dropper = getattr(client, "_drop_prompt_cache_client_state", None)
            if callable(dropper):
                dropper()
        if force_unlock_pending and unload_error is None:
            locked_pairs.discard(key)

        # PROCESS-WIDE eject for in-process MLX weights. This pool's instance
        # released ITS references above; the weights stay in Metal memory while
        # any other instance holds them (override clients, the chat summarizer
        # built at boot, an old runtime after a bundle reload, another
        # principal's service) -- none of which this pool can reach. Core's
        # `eject_model` unloads every holder, collects, clears MLX's cache, and
        # reports what is still resident. Its report rides in the payload.
        process_eject: Optional[Dict[str, Any]] = None
        if provider_s in _PROCESS_RESIDENCY_PROVIDERS and unload_error is None:
            process_eject = _process_eject_for(provider_s, model_s)
            if process_eject is not None and process_eject.get("ok") is False and process_eject.get("error"):
                unload_error = f"process-wide {provider_s} eject failed: {process_eject['error']}"

        record_after_unload = _local_residency_record(
            provider=provider_s,
            model=model_s,
            default=default_key,
            runtime_cached=runtime_cached_before,
            provider_instance=provider_instance,
            include_provider_state=provider_instance is not None,
            lock_owner=self,
        )
        unloaded = bool(record.get("loaded") is True and record_after_unload.get("loaded") is False)
        runtime_cache_unloaded = False
        warnings: List[str] = []
        error: Optional[str] = None
        if unload_error:
            error = unload_error
            warnings.append(unload_error)
        elif record_after_unload.get("provider_residency_verified") is not True:
            error = "model_residency unload did not verify unloaded provider residency"
            warnings.append(error)
        elif record_after_unload.get("loaded") is True:
            residual = (process_eject or {}).get("residual") if isinstance(process_eject, dict) else None
            if isinstance(residual, dict):
                error = (
                    "model_residency unload completed but the weights are still resident in this process: "
                    f"{int(residual.get('holders') or 0)} holder(s) still alive, "
                    f"{int(residual.get('held_bytes') or 0)} bytes held"
                )
            else:
                error = "model_residency unload completed but provider still reports the model loaded"
            warnings.append(error)
        elif isinstance(process_eject, dict) and process_eject.get("holders_refused"):
            error = (
                "model_residency unload: "
                f"{len(process_eject['holders_refused'])} holder(s) refused to unload "
                f"({'; '.join(str(h.get('error')) for h in process_eject['holders_refused'])})"
            )
            warnings.append(error)

        if error is None and client is not None and not default_key:
            self._clients.pop(key, None)
            runtime_cache_unloaded = True
            record_after_unload = _local_residency_record(
                provider=provider_s,
                model=model_s,
                default=False,
                runtime_cached=False,
                provider_instance=provider_instance,
                include_provider_state=provider_instance is not None,
                lock_owner=self,
            )

        result = {
            "ok": error is None,
            "supported": True,
            "operation": "unload",
            "task": "text_generation",
            "unloaded": unloaded,
            "runtime_cache_unloaded": runtime_cache_unloaded,
            "runtime": record_after_unload,
            **({"error": error} if error else {}),
            **({"warnings": warnings} if warnings else {}),
            **({"provider_unload_result": _jsonable(provider_unload_result)} if provider_unload_result is not None else {}),
            **({"process_eject": _jsonable(process_eject)} if process_eject is not None else {}),
            "diagnostics": {
                "source": "abstractruntime.multilocal",
                "runtime_cache_unloaded": runtime_cache_unloaded,
                "provider_unload_attempted": should_call_unload,
                "provider_control_client_created": transient_control_created,
            },
        }
        if error:
            result.setdefault("status_hint", "warning")
            result.setdefault("degraded", True)
        return _with_local_model_residency_summary(
            result,
            operation="unload",
            runtime=record_after_unload,
            action="unload_failed" if error else ("unloaded" if unloaded else "already_unloaded"),
            changed=bool(unloaded or runtime_cache_unloaded),
        )

    def _lock_known_pairs(self) -> set:
        """Warm (provider, model) pairs a lock can target: the pool plus the
        configured default identity (the analog of core's registry entries)."""
        known = set(self.list_loaded_clients())
        default_provider = str(getattr(self, "_default_provider", "") or "").strip().lower()
        default_model = str(getattr(self, "_default_model", "") or "").strip()
        if default_provider and default_model:
            known.add((default_provider, default_model))
        return known

    def _lock_provider_instance(self, provider_s: str, model_s: str) -> Any:
        clients = getattr(self, "_clients", None)
        client = clients.get((provider_s, model_s)) if isinstance(clients, dict) else None
        if client is None and (provider_s, model_s) == (
            str(getattr(self, "_default_provider", "") or "").strip().lower(),
            str(getattr(self, "_default_model", "") or "").strip(),
        ):
            client = getattr(self, "_default_client", None)
        if client is None:
            # A lock on the plain (provider, model) pair is honored when the
            # warm client is override-keyed — that client IS the one holding
            # (and able to verify) the residency the lock rule requires.
            override_clients = getattr(self, "_override_clients", None)
            if isinstance(override_clients, dict):
                for key, candidate in override_clients.items():
                    if isinstance(key, tuple) and len(key) >= 2 and (key[0], key[1]) == (provider_s, model_s):
                        client = candidate
                        break
        return getattr(client, "_llm", None)

    def _adopt_lock_pair(self, provider_s: str, model_s: str) -> Any:
        """Construct the pooled client for a sweep-resident pair and return its
        provider instance (lock ADOPTION). The ORDINARY `_get_client` path —
        client CONSTRUCTION only. Adoption never pushes a second copy of the
        weights through this stack: no cold load, no generation.

        It is not a no-op on the provider, and the distinction matters. The
        lock step that follows adoption calls `_apply_provider_side_lock_knob`,
        and for ollama that knob rides the native load request —
        `load_model(model, keep_alive=-1)` POSTs `/api/generate` with
        `prompt: ""`, the standard preload idiom. On an already-verified-
        resident model (the only kind the lock rule accepts) that request is a
        keep-alive REFRESH, not a load: the weights are already in the
        server's memory and only the TTL moves. `unlock` re-verifies residency
        first and skips the restore when the model is gone, so neither
        direction can load a model back as a side effect."""
        client = self._get_client(provider_s, model_s)
        return getattr(client, "_llm", None)

    def _drop_adopted_lock_pair(self, provider_s: str, model_s: str) -> None:
        """Undo `_adopt_lock_pair` when adoption did not complete."""
        clients = getattr(self, "_clients", None)
        if isinstance(clients, dict):
            clients.pop((provider_s, model_s), None)

    def lock_model_residency(
        self,
        payload: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        merged = _merge_optional_payload(payload, kwargs)
        return _local_model_residency_lock_result(
            merged,
            lock=True,
            default_provider=str(getattr(self, "_default_provider", "") or ""),
            default_model=str(getattr(self, "_default_model", "") or ""),
            known_pairs=self._lock_known_pairs(),
            locked_pairs=_local_locked_residency_pairs(self),
            provider_instance_lookup=self._lock_provider_instance,
            source="abstractruntime.multilocal",
            adopt_pair=self._adopt_lock_pair,
            drop_adopted_pair=self._drop_adopted_lock_pair,
        )

    def unlock_model_residency(
        self,
        payload: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        merged = _merge_optional_payload(payload, kwargs)
        return _local_model_residency_lock_result(
            merged,
            lock=False,
            default_provider=str(getattr(self, "_default_provider", "") or ""),
            default_model=str(getattr(self, "_default_model", "") or ""),
            known_pairs=self._lock_known_pairs(),
            locked_pairs=_local_locked_residency_pairs(self),
            provider_instance_lookup=self._lock_provider_instance,
            source="abstractruntime.multilocal",
        )

    def get_context_estimate(
        self,
        payload: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        merged = _merge_optional_payload(payload, kwargs)
        return _local_context_estimate(
            provider=merged.get("provider") or getattr(self, "_default_provider", None),
            model=merged.get("model") or getattr(self, "_default_model", None),
            context_length=merged.get("context_length"),
            base_url=merged.get("base_url"),
        )

    def generate(
        self,
        *,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        media: Optional[List[Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        params = dict(params or {})
        provider = params.pop("_provider", None)
        model = params.pop("_model", None)

        provider_str = (
            str(provider).strip().lower() if isinstance(provider, str) and provider.strip() else self._default_provider
        )
        model_str = str(model).strip() if isinstance(model, str) and model.strip() else self._default_model

        llm_kwargs_override: Dict[str, Any] = {}
        base_url = params.pop("base_url", None)
        if isinstance(base_url, str) and base_url.strip():
            llm_kwargs_override["base_url"] = base_url.strip()
        provider_api_key = _pop_provider_api_key(params)
        if provider_api_key:
            llm_kwargs_override["api_key"] = provider_api_key

        # A speculation request has to reach `create_llm`, not just `generate`:
        # it selects the runtime that loads the weights. The param is LEFT in
        # place as well so Core still validates it and reports the outcome --
        # once the lane is loaded, the per-call request matches and is a no-op.
        speculation_request = _speculation_construction_request(params)
        if speculation_request is not None:
            llm_kwargs_override["speculation"] = deepcopy(speculation_request)

        client = self._get_client(provider_str, model_str, llm_kwargs_override=llm_kwargs_override or None)
        result = client.generate(
            prompt=prompt,
            messages=messages,
            system_prompt=system_prompt,
            tools=tools,
            media=media,
            params=params,
        )
        return _stamp_effective_route(
            result,
            requested_provider=provider_str,
            requested_model=model_str,
            client=client,
        )

    def stream_tts(
        self,
        *,
        text: str,
        output: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        stream_params = dict(params or {})
        provider = stream_params.pop("_provider", None)
        model = stream_params.pop("_model", None)

        provider_str = (
            str(provider).strip().lower() if isinstance(provider, str) and provider.strip() else self._default_provider
        )
        model_str = str(model).strip() if isinstance(model, str) and model.strip() else self._default_model

        llm_kwargs_override: Dict[str, Any] = {}
        base_url = stream_params.pop("base_url", None)
        if isinstance(base_url, str) and base_url.strip():
            llm_kwargs_override["base_url"] = base_url.strip()
        provider_api_key = _pop_provider_api_key(stream_params)
        if provider_api_key:
            llm_kwargs_override["api_key"] = provider_api_key

        client = self._get_client(provider_str, model_str, llm_kwargs_override=llm_kwargs_override or None)
        return client.stream_tts(text=text, output=output, params=stream_params)

    # CATALOG QUERIES NEVER GO THROUGH A LOADED CLIENT (defect 2026-09-22).
    # They used to be routed via the DEFAULT pooled client, so the moment the
    # configured default could not be built (its weights not downloaded --
    # the warm-up above soft-fails that on purpose so the host still boots)
    # every provider's model list, every capability lookup and every
    # voice/music/vision catalog raised "no provider/model is configured",
    # which was both wrong (one WAS configured) and total (every picker in
    # every UI was empty, for providers unrelated to the broken default).
    # `LocalAbstractCoreLLMClient` answers these with the stateless
    # `discovery_queries.local_*` helpers; the pool delegates the same way,
    # exactly as `list_providers` below already did.

    def get_model_capabilities(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        from .discovery_queries import local_get_model_capabilities

        payload = local_get_model_capabilities(self._capability_lookup_model(model_name))
        capabilities = payload.get("capabilities") if isinstance(payload, dict) else None
        if isinstance(capabilities, dict):
            return capabilities
        from abstractruntime.core.vars import DEFAULT_MAX_TOKENS

        return {"max_tokens": DEFAULT_MAX_TOKENS}

    def lookup_model_capabilities(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        from .discovery_queries import local_get_model_capabilities

        return local_get_model_capabilities(self._capability_lookup_model(model_name))

    def get_execution_capabilities(
        self, model_name: Optional[str] = None, *, provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        from abstractcore.providers.speculation import get_execution_capabilities

        target_model = str(model_name or self._default_model or "").strip()
        target_provider = str(provider or self._default_provider or "").strip().lower()
        # Discovery must not call _get_client: that could load tens of GB.
        client = self._clients.get((target_provider, target_model))
        instance = getattr(client, "_llm", None) if client is not None else None
        return get_execution_capabilities(target_model, provider=target_provider, instance=instance)

    def list_providers(
        self,
        *,
        include_models: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        _ = kwargs
        from .discovery_queries import local_list_providers

        return local_list_providers(
            include_models=include_models,
            default_provider=self._default_provider,
            default_model=self._default_model,
        )

    def list_provider_models(
        self,
        provider_name: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        from .discovery_queries import local_list_provider_models

        call_kwargs = dict(kwargs)
        provider_api_key = _pop_provider_api_key(call_kwargs)
        return local_list_provider_models(
            provider_name,
            base_url=call_kwargs.get("base_url"),
            provider_api_key=provider_api_key,
            input_type=call_kwargs.get("input_type"),
            output_type=call_kwargs.get("output_type"),
            capability_route=call_kwargs.get("capability_route", call_kwargs.get("capability_routes")),
            timeout_s=call_kwargs.get("timeout_s"),
        )

    def list_embedding_models(
        self,
        *,
        base_url: Optional[str] = None,
        provider_api_key: Optional[str] = None,
        provider: Optional[str] = None,
        providers_only: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        call_kwargs = dict(kwargs)
        provider_api_key = provider_api_key or _pop_provider_api_key(call_kwargs)
        from .discovery_queries import local_list_embedding_models

        return local_list_embedding_models(
            base_url=base_url,
            provider_api_key=provider_api_key,
            provider=provider,
            providers_only=providers_only,
            timeout_s=call_kwargs.get("timeout_s"),
        )

    def get_voice_catalog(
        self,
        *,
        base_url: Optional[str] = None,
        provider_api_key: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        providers_only: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        call_kwargs = dict(kwargs)
        provider_api_key = provider_api_key or _pop_provider_api_key(call_kwargs)
        from .discovery_queries import local_get_voice_catalog

        return local_get_voice_catalog(
            base_url=base_url,
            provider_api_key=provider_api_key,
            provider=provider,
            model=model,
            providers_only=providers_only,
        )

    def list_tts_models(
        self,
        *,
        base_url: Optional[str] = None,
        provider_api_key: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        call_kwargs = dict(kwargs)
        provider_api_key = provider_api_key or _pop_provider_api_key(call_kwargs)
        from .discovery_queries import local_list_tts_models

        return local_list_tts_models(
            base_url=base_url,
            provider_api_key=provider_api_key,
            provider=provider,
        )

    def list_stt_models(
        self,
        *,
        base_url: Optional[str] = None,
        provider_api_key: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        call_kwargs = dict(kwargs)
        provider_api_key = provider_api_key or _pop_provider_api_key(call_kwargs)
        from .discovery_queries import local_list_stt_models

        return local_list_stt_models(
            base_url=base_url,
            provider_api_key=provider_api_key,
            provider=provider,
        )

    def list_music_providers(
        self,
        *,
        task: Optional[str] = None,
        base_url: Optional[str] = None,
        provider_api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        call_kwargs = dict(kwargs)
        provider_api_key = provider_api_key or _pop_provider_api_key(call_kwargs)
        from .discovery_queries import local_list_music_providers

        return local_list_music_providers(
            task=task,
            base_url=base_url,
            provider_api_key=provider_api_key,
        )

    def list_music_models(
        self,
        *,
        task: Optional[str] = None,
        base_url: Optional[str] = None,
        provider_api_key: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        call_kwargs = dict(kwargs)
        provider_api_key = provider_api_key or _pop_provider_api_key(call_kwargs)
        from .discovery_queries import local_list_music_models

        return local_list_music_models(
            task=task,
            base_url=base_url,
            provider_api_key=provider_api_key,
            provider=provider,
        )

    def list_vision_provider_models(
        self,
        *,
        task: Optional[str] = None,
        base_url: Optional[str] = None,
        provider_api_key: Optional[str] = None,
        provider: Optional[str] = None,
        providers_only: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        call_kwargs = dict(kwargs)
        provider_api_key = provider_api_key or _pop_provider_api_key(call_kwargs)
        from .discovery_queries import local_list_vision_provider_models

        return local_list_vision_provider_models(
            task=task,
            base_url=base_url,
            provider_api_key=provider_api_key,
            provider=provider,
            providers_only=providers_only,
        )

    def list_cached_vision_models(
        self,
        *,
        task: Optional[str] = None,
        base_url: Optional[str] = None,
        provider_api_key: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        _ = (base_url, provider_api_key, kwargs)
        from .discovery_queries import local_list_cached_vision_models

        return local_list_cached_vision_models(task=task, provider=provider)

    def list_vision_adapters(
        self,
        *,
        model: Optional[str] = None,
        task: Optional[str] = None,
        base_url: Optional[str] = None,
        provider_api_key: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        call_kwargs = dict(kwargs)
        provider_api_key = provider_api_key or _pop_provider_api_key(call_kwargs)
        from .discovery_queries import local_list_vision_adapters

        return local_list_vision_adapters(
            model=model,
            task=task,
            base_url=base_url,
            provider_api_key=provider_api_key,
            provider=provider,
        )

    def get_prompt_cache_capabilities(
        self,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        _ = kwargs
        provider_str = (
            str(provider).strip().lower() if isinstance(provider, str) and provider.strip() else self._default_provider
        )
        model_str = str(model).strip() if isinstance(model, str) and model.strip() else self._default_model
        client = self._get_client(provider_str, model_str)
        return client.get_prompt_cache_capabilities()

    def get_prompt_cache_stats(
        self,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        provider_str = (
            str(provider).strip().lower() if isinstance(provider, str) and provider.strip() else self._default_provider
        )
        model_str = str(model).strip() if isinstance(model, str) and model.strip() else self._default_model
        client = self._get_client(provider_str, model_str)
        return client.get_prompt_cache_stats(**kwargs)

    def get_memory_snapshot(self, **kwargs: Any) -> Dict[str, Any]:
        _ = kwargs
        return _local_memory_snapshot()

    def _pooled_clients(self) -> List[Any]:
        """Existing pooled clients only (never constructs new ones)."""
        clients: List[Any] = list(getattr(self, "_clients", {}).values())
        clients.extend(list(getattr(self, "_override_clients", {}).values()))
        default_client = getattr(self, "_default_client", None)
        if default_client is not None:
            clients.append(default_client)
        unique: List[Any] = []
        seen: set[int] = set()
        for client in clients:
            if id(client) in seen:
                continue
            seen.add(id(client))
            unique.append(client)
        return unique

    def _pooled_client_error_row(self, client: Any, *, error: str) -> Dict[str, Any]:
        return {
            "provider": str(getattr(client, "_provider", "") or "") or None,
            "model": str(getattr(client, "_model", "") or "") or None,
            "error": error,
        }

    def list_session_prompt_caches(self, session_id: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        _ = kwargs
        rows: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        for client in self._pooled_clients():
            lister = getattr(client, "list_session_prompt_caches", None)
            if not callable(lister):
                errors.append(self._pooled_client_error_row(client, error="client does not implement list_session_prompt_caches"))
                continue
            try:
                listing = lister(session_id=session_id)
            except Exception as exc:  # noqa: BLE001 - reported per client, never raised
                errors.append(self._pooled_client_error_row(client, error=str(exc)))
                continue
            if not isinstance(listing, dict):
                errors.append(self._pooled_client_error_row(client, error="invalid session prompt cache listing"))
                continue
            if listing.get("ok") is False:
                errors.append(self._pooled_client_error_row(client, error=str(listing.get("error") or "listing failed")))
                continue
            rows.extend(row for row in (listing.get("caches") or []) if isinstance(row, dict))
        out: Dict[str, Any] = {"ok": True, "caches": rows}
        if errors:
            out["errors"] = errors
        return out

    def clear_session_prompt_caches(self, session_id: str, **kwargs: Any) -> Dict[str, Any]:
        _ = kwargs
        session_s = str(session_id or "").strip()
        if not session_s:
            return {
                "ok": False,
                "error": "clear_session_prompt_caches requires a session_id",
                "cleared": [],
                "count": 0,
            }
        cleared: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        count = 0
        for client in self._pooled_clients():
            clearer = getattr(client, "clear_session_prompt_caches", None)
            if not callable(clearer):
                errors.append(self._pooled_client_error_row(client, error="client does not implement clear_session_prompt_caches"))
                continue
            try:
                result = clearer(session_id=session_s)
            except Exception as exc:  # noqa: BLE001 - reported per client, never raised
                errors.append(self._pooled_client_error_row(client, error=str(exc)))
                continue
            if not isinstance(result, dict):
                errors.append(self._pooled_client_error_row(client, error="invalid session prompt cache clear result"))
                continue
            if result.get("ok") is False:
                errors.append(self._pooled_client_error_row(client, error=str(result.get("error") or "clear failed")))
                continue
            cleared.extend(row for row in (result.get("cleared") or []) if isinstance(row, dict))
            raw_count = result.get("count")
            if isinstance(raw_count, int) and not isinstance(raw_count, bool):
                count += raw_count
        out: Dict[str, Any] = {"ok": True, "cleared": cleared, "count": count}
        if errors:
            out["errors"] = errors
        return out

    def prompt_cache_set(
        self,
        *,
        key: str,
        make_default: bool = True,
        ttl_s: Optional[float] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        provider_str = (
            str(provider).strip().lower() if isinstance(provider, str) and provider.strip() else self._default_provider
        )
        model_str = str(model).strip() if isinstance(model, str) and model.strip() else self._default_model
        client = self._get_client(provider_str, model_str)
        return client.prompt_cache_set(key=key, make_default=make_default, ttl_s=ttl_s, **kwargs)

    def prompt_cache_update(
        self,
        *,
        key: str,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        add_generation_prompt: bool = False,
        ttl_s: Optional[float] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        provider_str = (
            str(provider).strip().lower() if isinstance(provider, str) and provider.strip() else self._default_provider
        )
        model_str = str(model).strip() if isinstance(model, str) and model.strip() else self._default_model
        client = self._get_client(provider_str, model_str)
        return client.prompt_cache_update(
            key=key,
            prompt=prompt,
            messages=messages,
            system_prompt=system_prompt,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            ttl_s=ttl_s,
            **kwargs,
        )

    def prompt_cache_fork(
        self,
        *,
        from_key: str,
        to_key: str,
        make_default: bool = False,
        ttl_s: Optional[float] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        provider_str = (
            str(provider).strip().lower() if isinstance(provider, str) and provider.strip() else self._default_provider
        )
        model_str = str(model).strip() if isinstance(model, str) and model.strip() else self._default_model
        client = self._get_client(provider_str, model_str)
        return client.prompt_cache_fork(
            from_key=from_key,
            to_key=to_key,
            make_default=make_default,
            ttl_s=ttl_s,
            **kwargs,
        )

    def prompt_cache_clear(
        self,
        *,
        key: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        provider_str = (
            str(provider).strip().lower() if isinstance(provider, str) and provider.strip() else self._default_provider
        )
        model_str = str(model).strip() if isinstance(model, str) and model.strip() else self._default_model
        client = self._get_client(provider_str, model_str)
        return client.prompt_cache_clear(key=key, **kwargs)

    def prompt_cache_prepare_modules(
        self,
        *,
        namespace: str,
        modules: List[Dict[str, Any]],
        make_default: bool = False,
        ttl_s: Optional[float] = None,
        version: int = 1,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        provider_str = (
            str(provider).strip().lower() if isinstance(provider, str) and provider.strip() else self._default_provider
        )
        model_str = str(model).strip() if isinstance(model, str) and model.strip() else self._default_model
        client = self._get_client(provider_str, model_str)
        return client.prompt_cache_prepare_modules(
            namespace=namespace,
            modules=modules,
            make_default=make_default,
            ttl_s=ttl_s,
            version=version,
            **kwargs,
        )

    def list_prompt_cache_exports(
        self,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        provider_str = (
            str(provider).strip().lower() if isinstance(provider, str) and provider.strip() else self._default_provider
        )
        model_str = str(model).strip() if isinstance(model, str) and model.strip() else self._default_model
        loaded_client = self._clients.get((provider_str, model_str))
        provider_obj = getattr(loaded_client, "_llm", None) if loaded_client is not None else None
        root_dir = _coerce_prompt_cache_export_root_dir(
            kwargs.pop("prompt_cache_export_root_dir", self._prompt_cache_export_root_dir)
        )
        return _list_prompt_cache_exports_local(
            root_dir=root_dir,
            provider=provider_obj,
            provider_name=provider_str,
            model=model_str,
        )

    def prompt_cache_export(
        self,
        *,
        name: str,
        key: str,
        q8: bool = False,
        meta: Optional[Dict[str, Any]] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        provider_str = (
            str(provider).strip().lower() if isinstance(provider, str) and provider.strip() else self._default_provider
        )
        model_str = str(model).strip() if isinstance(model, str) and model.strip() else self._default_model
        client = self._get_client(provider_str, model_str)
        return client.prompt_cache_export(
            name=name,
            key=key,
            q8=q8,
            meta=meta,
            provider=provider_str,
            model=model_str,
            **kwargs,
        )

    def prompt_cache_import(
        self,
        *,
        name: str,
        key: Optional[str] = None,
        make_default: bool = True,
        clear_existing: bool = False,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        provider_str = (
            str(provider).strip().lower() if isinstance(provider, str) and provider.strip() else self._default_provider
        )
        model_str = str(model).strip() if isinstance(model, str) and model.strip() else self._default_model
        client = self._get_client(provider_str, model_str)
        return client.prompt_cache_import(
            name=name,
            key=key,
            make_default=make_default,
            clear_existing=clear_existing,
            provider=provider_str,
            model=model_str,
            **kwargs,
        )

    def upsert_text_bloc(
        self,
        *,
        path: str,
        content: str,
        sha256: Optional[str] = None,
        content_sha256: Optional[str] = None,
        media_type: str = "text",
        size_bytes: Optional[int] = None,
        mtime_ns: Optional[int] = None,
        format: Optional[str] = None,
        estimated_tokens: Optional[int] = None,
        relpath_base: Optional[str] = None,
        summary: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        root_dir = kwargs.pop("bloc_root_dir", self._bloc_root_dir)
        return _upsert_text_bloc_local(
            root_dir=root_dir,
            path=path,
            content=content,
            sha256=sha256,
            content_sha256=content_sha256,
            media_type=media_type,
            size_bytes=size_bytes,
            mtime_ns=mtime_ns,
            format=format,
            estimated_tokens=estimated_tokens,
            relpath_base=relpath_base,
            summary=summary,
            keywords=keywords,
        )

    def get_bloc_record(
        self,
        *,
        sha256: Optional[str] = None,
        bloc_id: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        root_dir = kwargs.pop("bloc_root_dir", self._bloc_root_dir)
        return _get_bloc_record_local(root_dir=root_dir, sha256=sha256, bloc_id=bloc_id)

    def list_blocs(
        self,
        *,
        sha256: Optional[str] = None,
        bloc_id: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        root_dir = kwargs.pop("bloc_root_dir", self._bloc_root_dir)
        return _list_blocs_local(root_dir=root_dir, sha256=sha256, bloc_id=bloc_id)

    def get_bloc_kv_manifest(
        self,
        *,
        sha256: Optional[str] = None,
        bloc_id: Optional[int] = None,
        artifact_path: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        root_dir = kwargs.pop("bloc_root_dir", self._bloc_root_dir)
        provider_str = (
            str(provider).strip().lower() if isinstance(provider, str) and provider.strip() else self._default_provider
        )
        model_str = str(model).strip() if isinstance(model, str) and model.strip() else self._default_model
        client = self._get_client(provider_str, model_str)
        return _get_bloc_kv_manifest_local(
            provider=getattr(client, "_llm", None),
            model=model_str,
            root_dir=root_dir,
            sha256=sha256,
            bloc_id=bloc_id,
            artifact_path=artifact_path,
        )

    def ensure_bloc_kv_artifact(
        self,
        *,
        sha256: Optional[str] = None,
        bloc_id: Optional[int] = None,
        artifact_path: Optional[str] = None,
        force_rebuild: bool = False,
        debug: bool = False,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        root_dir = kwargs.pop("bloc_root_dir", self._bloc_root_dir)
        provider_str = (
            str(provider).strip().lower() if isinstance(provider, str) and provider.strip() else self._default_provider
        )
        model_str = str(model).strip() if isinstance(model, str) and model.strip() else self._default_model
        client = self._get_client(provider_str, model_str)
        return _ensure_bloc_kv_artifact_local(
            provider=getattr(client, "_llm", None),
            model=model_str,
            root_dir=root_dir,
            sha256=sha256,
            bloc_id=bloc_id,
            artifact_path=artifact_path,
            force_rebuild=force_rebuild,
            debug=debug,
        )

    def load_bloc_kv_artifact(
        self,
        *,
        sha256: Optional[str] = None,
        bloc_id: Optional[int] = None,
        artifact_path: Optional[str] = None,
        stable_cache_key: Optional[str] = None,
        key: Optional[str] = None,
        make_default: bool = False,
        force_rebuild: bool = False,
        debug: bool = False,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        root_dir = kwargs.pop("bloc_root_dir", self._bloc_root_dir)
        provider_str = (
            str(provider).strip().lower() if isinstance(provider, str) and provider.strip() else self._default_provider
        )
        model_str = str(model).strip() if isinstance(model, str) and model.strip() else self._default_model
        client = self._get_client(provider_str, model_str)
        return _load_bloc_kv_artifact_local(
            provider=getattr(client, "_llm", None),
            model=model_str,
            root_dir=root_dir,
            sha256=sha256,
            bloc_id=bloc_id,
            artifact_path=artifact_path,
            stable_cache_key=stable_cache_key,
            key=key,
            make_default=make_default,
            force_rebuild=force_rebuild,
            debug=debug,
        )

    def list_bloc_kv_artifacts(
        self,
        *,
        sha256: Optional[str] = None,
        bloc_id: Optional[int] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        root_dir = kwargs.pop("bloc_root_dir", self._bloc_root_dir)
        provider_str = str(provider).strip().lower() if isinstance(provider, str) and provider.strip() else None
        model_str = str(model).strip() if isinstance(model, str) and model.strip() else None
        return _list_bloc_kv_artifacts_local(
            root_dir=root_dir,
            sha256=sha256,
            bloc_id=bloc_id,
            provider_name=provider_str,
            model=model_str,
        )

    def delete_bloc_kv_artifact(
        self,
        *,
        sha256: Optional[str] = None,
        bloc_id: Optional[int] = None,
        artifact_path: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        clear_loaded: bool = False,
        force: bool = False,
        dry_run: bool = False,
        debug: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        root_dir = kwargs.pop("bloc_root_dir", self._bloc_root_dir)
        provider_str = str(provider).strip().lower() if isinstance(provider, str) and provider.strip() else None
        model_str = str(model).strip() if isinstance(model, str) and model.strip() else None
        listed = _list_bloc_kv_artifacts_local(
            root_dir=root_dir,
            sha256=sha256,
            bloc_id=bloc_id,
            provider_name=provider_str,
            model=model_str,
        )
        if not listed.get("ok"):
            return listed
        entries = _filter_bloc_kv_entries_by_artifact_path(list(listed.get("artifacts") or []), artifact_path)
        if not entries:
            selector = f"artifact_path={artifact_path}" if artifact_path else f"sha256={sha256}" if sha256 else f"bloc_id={bloc_id}"
            return _bloc_not_found_payload(operation="kv_delete", selector=selector)
        if len(entries) != 1:
            return {
                "ok": False,
                "operation": "kv_delete",
                "code": "invalid_request",
                "error": "delete_bloc_kv_artifact requires a selector that resolves to exactly one artifact",
            }
        entry = entries[0]
        entry_provider = provider_str or _bloc_kv_entry_provider(entry)
        entry_model = model_str or _bloc_kv_entry_model(entry)
        loaded_provider = _multilocal_loaded_provider(self, entry_provider, entry_model)
        return _delete_bloc_kv_artifact_local(
            provider=loaded_provider,
            root_dir=root_dir,
            sha256=sha256,
            bloc_id=bloc_id,
            artifact_path=_bloc_kv_entry_artifact_path(entry) or artifact_path,
            provider_name=entry_provider,
            model=entry_model,
            clear_loaded=clear_loaded,
            force=force,
            dry_run=dry_run,
            debug=debug,
        )

    def prune_bloc_kv_artifacts(
        self,
        *,
        sha256: Optional[str] = None,
        bloc_id: Optional[int] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        clear_loaded: bool = False,
        force: bool = False,
        dry_run: bool = False,
        debug: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        root_dir = kwargs.pop("bloc_root_dir", self._bloc_root_dir)
        provider_str = str(provider).strip().lower() if isinstance(provider, str) and provider.strip() else None
        model_str = str(model).strip() if isinstance(model, str) and model.strip() else None
        listed = _list_bloc_kv_artifacts_local(
            root_dir=root_dir,
            sha256=sha256,
            bloc_id=bloc_id,
            provider_name=provider_str,
            model=model_str,
        )
        if not listed.get("ok"):
            return listed
        entries = [dict(item) for item in list(listed.get("artifacts") or []) if isinstance(item, dict)]
        if not entries:
            return {"ok": True, "operation": "kv_prune", "results": []}

        live_bindings: List[Dict[str, Any]] = []
        if not (clear_loaded or force):
            for entry in entries:
                loaded_provider = _multilocal_loaded_provider(
                    self,
                    provider_str or _bloc_kv_entry_provider(entry),
                    model_str or _bloc_kv_entry_model(entry),
                )
                live_bindings.extend(_find_entry_live_bindings_local(provider=loaded_provider, entry=entry))
            if live_bindings:
                return {
                    "ok": False,
                    "operation": "kv_prune",
                    "code": "artifact_in_use",
                    "error": "matching bloc KV artifacts may be loaded in live prompt-cache keys",
                    "live_bindings": live_bindings,
                }

        results: List[Dict[str, Any]] = []
        for entry in entries:
            entry_provider = provider_str or _bloc_kv_entry_provider(entry)
            entry_model = model_str or _bloc_kv_entry_model(entry)
            loaded_provider = _multilocal_loaded_provider(self, entry_provider, entry_model)
            payload = _delete_bloc_kv_artifact_local(
                provider=loaded_provider,
                root_dir=root_dir,
                sha256=sha256,
                bloc_id=bloc_id,
                artifact_path=_bloc_kv_entry_artifact_path(entry),
                provider_name=entry_provider,
                model=entry_model,
                clear_loaded=clear_loaded,
                force=force,
                dry_run=dry_run,
                debug=debug,
            )
            if not payload.get("ok"):
                return payload
            results.append(dict(payload.get("result") or {}))
        return {"ok": True, "operation": "kv_prune", "results": results}

    def delete_bloc(
        self,
        *,
        sha256: Optional[str] = None,
        bloc_id: Optional[int] = None,
        delete_kv: bool = True,
        clear_loaded: bool = False,
        force: bool = False,
        dry_run: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        root_dir = kwargs.pop("bloc_root_dir", self._bloc_root_dir)
        if not delete_kv:
            return _delete_bloc_local(
                provider=None,
                root_dir=root_dir,
                sha256=sha256,
                bloc_id=bloc_id,
                delete_kv=False,
                clear_loaded=clear_loaded,
                force=force,
                dry_run=dry_run,
            )

        listed = _list_bloc_kv_artifacts_local(root_dir=root_dir, sha256=sha256, bloc_id=bloc_id)
        if not listed.get("ok"):
            return listed
        entries = [dict(item) for item in list(listed.get("artifacts") or []) if isinstance(item, dict)]

        live_bindings: List[Dict[str, Any]] = []
        if not (clear_loaded or force):
            for entry in entries:
                loaded_provider = _multilocal_loaded_provider(
                    self,
                    _bloc_kv_entry_provider(entry),
                    _bloc_kv_entry_model(entry),
                )
                live_bindings.extend(_find_entry_live_bindings_local(provider=loaded_provider, entry=entry))
            if live_bindings:
                return {
                    "ok": False,
                    "operation": "delete",
                    "code": "artifact_in_use",
                    "error": "bloc has loaded KV artifacts in live prompt-cache keys",
                    "live_bindings": live_bindings,
                }

        kv_results: List[Dict[str, Any]] = []
        for entry in entries:
            entry_provider = _bloc_kv_entry_provider(entry)
            entry_model = _bloc_kv_entry_model(entry)
            loaded_provider = _multilocal_loaded_provider(self, entry_provider, entry_model)
            payload = _delete_bloc_kv_artifact_local(
                provider=loaded_provider,
                root_dir=root_dir,
                sha256=sha256,
                bloc_id=bloc_id,
                artifact_path=_bloc_kv_entry_artifact_path(entry),
                provider_name=entry_provider,
                model=entry_model,
                clear_loaded=clear_loaded,
                force=force,
                dry_run=dry_run,
                debug=False,
            )
            if not payload.get("ok"):
                return payload
            kv_results.append(dict(payload.get("result") or {}))

        deleted = _delete_bloc_local(
            provider=None,
            root_dir=root_dir,
            sha256=sha256,
            bloc_id=bloc_id,
            delete_kv=False,
            clear_loaded=clear_loaded,
            force=force,
            dry_run=dry_run,
        )
        if not deleted.get("ok"):
            return deleted
        result = dict(deleted.get("result") or {})
        result["kv_results"] = kv_results
        result["live_bindings"] = live_bindings
        deleted["result"] = result
        return deleted


class RemoteGenerationCancelled(RuntimeError):
    """The effect's cancel event severed an in-flight request to a remote
    AbstractCore server (the server sees the disconnect and cancels the
    generation; see `HttpxRequestSender.post(cancel_event=)`)."""


class HttpxRequestSender:
    """Default request sender based on httpx (sync)."""

    #: `post(..., cancel_event=)` is honoured: the request runs on its own
    #: connection and a set event SEVERS it (socket shutdown), which is the
    #: wire form of a Stop for a remote AbstractCore server — the server's
    #: client-disconnect watcher cancels the generation. Custom senders
    #: without this attribute are never handed the event.
    supports_cancel_event = True

    def __init__(self):
        import httpx

        self._httpx = httpx

    def _post_cancellable(self, url: str, *, headers: Dict[str, str], json: Dict[str, Any], timeout: Any,
                          cancel_event: "threading.Event") -> Any:
        """POST on a dedicated connection that `cancel_event` severs.

        The TCP socket is captured through httpcore's `trace` extension
        (`connection.connect_tcp.complete` fires for a NEW connection, hence
        the un-pooled client); a watcher thread waits on the event and shuts
        the socket down (`shutdown(SHUT_RDWR)` wakes a thread blocked in
        recv; `close()` does not). Same mechanism as AbstractCore's
        `providers/generation_cancel.HttpCancelGuard`, kept self-contained so
        a thin remote runtime does not depend on a newer AbstractCore."""
        import socket as _socket

        streams: List[Any] = []
        lock = threading.Lock()
        done = threading.Event()
        severed = {"n": 0}

        def _sever() -> None:
            with lock:
                targets = list(streams)
            for stream in targets:
                try:
                    sock = stream.get_extra_info("socket")
                    if sock is not None:
                        sock.shutdown(_socket.SHUT_RDWR)
                        severed["n"] += 1
                except OSError:
                    pass
                except Exception:  # noqa: BLE001
                    pass

        def _trace(name: str, info: Any) -> None:
            if name == "connection.connect_tcp.complete" and isinstance(info, dict):
                stream = info.get("return_value")
                if stream is not None:
                    with lock:
                        streams.append(stream)
                    if cancel_event.is_set():
                        _sever()

        def _watch() -> None:
            # 0.05 s is only how long an idle watcher lingers after its
            # request ended; the cancel itself wakes `wait` immediately.
            while not done.is_set():
                if cancel_event.wait(0.05):
                    if not done.is_set():
                        _sever()
                    return

        watcher = threading.Thread(target=_watch, name="abstractruntime-remote-cancel", daemon=True)
        watcher.start()
        try:
            with self._httpx.Client(timeout=timeout) as client:
                try:
                    return client.post(url, headers=headers, json=json, extensions={"trace": _trace})
                except Exception as exc:
                    if cancel_event.is_set():
                        raise RemoteGenerationCancelled(
                            f"remote AbstractCore request to {url} severed by the run's cancel "
                            f"({severed['n']} connection(s) cut; the server cancels on disconnect)"
                        ) from exc
                    raise
        finally:
            done.set()

    def get(
        self,
        url: str,
        *,
        headers: Dict[str, str],
        timeout: float,
    ) -> HttpResponse:
        resp = self._httpx.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return HttpResponse(body=resp.json(), headers=dict(resp.headers))

    def post(
        self,
        url: str,
        *,
        headers: Dict[str, str],
        json: Dict[str, Any],
        timeout: float,
        cancel_event: Optional["threading.Event"] = None,
    ) -> HttpResponse:
        if cancel_event is not None:
            resp = self._post_cancellable(url, headers=headers, json=json, timeout=timeout, cancel_event=cancel_event)
        else:
            resp = self._httpx.post(url, headers=headers, json=json, timeout=timeout)
        resp.raise_for_status()
        return HttpResponse(body=resp.json(), headers=dict(resp.headers))

    def post_bytes(
        self,
        url: str,
        *,
        headers: Dict[str, str],
        json: Dict[str, Any],
        timeout: float,
    ) -> HttpBinaryResponse:
        resp = self._httpx.post(url, headers=headers, json=json, timeout=timeout)
        resp.raise_for_status()
        return HttpBinaryResponse(content=bytes(resp.content or b""), headers=dict(resp.headers))

    def post_jsonl_stream(
        self,
        url: str,
        *,
        headers: Dict[str, str],
        json: Dict[str, Any],
        timeout: float,
    ) -> Any:
        with self._httpx.stream("POST", url, headers=headers, json=json, timeout=timeout) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                yield line

    def post_multipart(
        self,
        url: str,
        *,
        headers: Dict[str, str],
        data: Dict[str, Any],
        files: Dict[str, Any],
        timeout: float,
    ) -> Any:
        resp = self._httpx.post(url, headers=headers, data=data, files=files, timeout=timeout)
        resp.raise_for_status()
        content_type = str(resp.headers.get("content-type", "") or "").split(";", 1)[0].strip().lower()
        if content_type == "application/json" or content_type.endswith("+json"):
            return HttpResponse(body=resp.json(), headers=dict(resp.headers))
        return HttpBinaryResponse(content=bytes(resp.content or b""), headers=dict(resp.headers))


def _http_error_response_body(error: Any) -> Optional[Any]:
    """Best-effort parse of the response body carried by a raised HTTP-status
    error (the raise_for_status pattern: httpx.HTTPStatusError and fakes carry
    `.response`). Returns a dict when the body is JSON, the stripped text
    otherwise, or None when nothing is accessible."""
    response = getattr(error, "response", None)
    if response is None:
        return None
    body = getattr(response, "body", None)
    if isinstance(body, dict):
        return dict(body)
    json_fn = getattr(response, "json", None)
    if callable(json_fn):
        try:
            parsed = json_fn()
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            return parsed
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        try:
            parsed = json.loads(text)
        except Exception:
            return text.strip()
        return parsed if isinstance(parsed, dict) else text.strip()
    return None


def _unwrap_http_response(value: Any) -> Tuple[Dict[str, Any], Dict[str, str]]:
    if isinstance(value, dict):
        return value, {}
    body = getattr(value, "body", None)
    headers = getattr(value, "headers", None)
    if isinstance(body, dict) and isinstance(headers, dict):
        return body, headers
    json_fn = getattr(value, "json", None)
    hdrs = getattr(value, "headers", None)
    if callable(json_fn) and hdrs is not None:
        try:
            payload = json_fn()
        except Exception:
            payload = {}
        return payload if isinstance(payload, dict) else {"data": _jsonable(payload)}, dict(hdrs)
    return {"data": _jsonable(value)}, {}


def _unwrap_binary_response(value: Any) -> Tuple[bytes, Dict[str, str]]:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value), {}
    content = getattr(value, "content", None)
    headers = getattr(value, "headers", None)
    if isinstance(content, (bytes, bytearray)):
        return bytes(content), dict(headers) if isinstance(headers, dict) else {}
    if isinstance(value, dict):
        data = value.get("content") or value.get("bytes") or value.get("data")
        if isinstance(data, (bytes, bytearray)):
            return bytes(data), {}
        if isinstance(data, str):
            raw = data.split(",", 1)[1] if data.startswith("data:") and "," in data else data
            try:
                return base64.b64decode("".join(raw.split()), validate=True), {}
            except Exception as e:
                raise ValueError("Remote binary response string must be base64 or a data URL.") from e
    raise ValueError("Remote binary response did not contain bytes.")


def _decode_stream_audio_b64(event: Dict[str, Any]) -> Optional[bytes]:
    raw = event.get("audio_b64")
    if not isinstance(raw, str) or not raw.strip():
        return None
    return base64.b64decode("".join(raw.strip().split()), validate=True)


def _combine_wav_segments(segments: List[bytes]) -> bytes:
    if not segments:
        raise ValueError("TTS stream completed without audio chunks.")
    channels: Optional[int] = None
    sampwidth: Optional[int] = None
    framerate: Optional[int] = None
    frames: List[bytes] = []
    for index, segment in enumerate(segments):
        with wave.open(io.BytesIO(bytes(segment)), "rb") as wf:
            c = int(wf.getnchannels())
            sw = int(wf.getsampwidth())
            fr = int(wf.getframerate())
            if channels is None:
                channels, sampwidth, framerate = c, sw, fr
            elif (channels, sampwidth, framerate) != (c, sw, fr):
                raise ValueError(f"TTS stream WAV segment {index} does not match the first segment format.")
            frames.append(wf.readframes(wf.getnframes()))
    out = io.BytesIO()
    with wave.open(out, "wb") as wf:
        wf.setnchannels(int(channels or 1))
        wf.setsampwidth(int(sampwidth or 2))
        wf.setframerate(int(framerate or 24000))
        wf.writeframes(b"".join(frames))
    return out.getvalue()


def _finalize_tts_stream_artifact(
    *,
    segments: List[bytes],
    artifact_store: Optional[Any],
    run_id: Optional[str],
    tags: Dict[str, Any],
    text: str,
    spec: Dict[str, Any],
    provider: str,
    model: Optional[str],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    audio_bytes = _combine_wav_segments(segments)
    result = _normalize_multimodal_response(
        {
            "outputs": {
                "voice": [
                    {
                        "modality": "voice",
                        "task": "tts",
                        "data": audio_bytes,
                        "content_type": "audio/wav",
                        "format": "wav",
                        "provider": provider,
                        "model": model,
                        "metadata": metadata if isinstance(metadata, dict) else {},
                    }
                ]
            },
            "metadata": {"model": model, "provider": provider, "streaming": True},
        },
        artifact_store=artifact_store,
        run_id=run_id,
        default_tags=tags,
        generation_context=_generated_artifact_context(prompt=text, media=None, output_request=spec),
    )
    voice_items = result.get("outputs", {}).get("voice") if isinstance(result, dict) else None
    if not isinstance(voice_items, list) or not voice_items:
        raise ValueError("TTS stream finalization did not produce a voice artifact descriptor.")
    return dict(voice_items[0])


def _trace_run_id_and_tags_from_params(
    params: Dict[str, Any],
    *,
    task: str,
    modality: str,
    model: Optional[str] = None,
) -> tuple[Optional[str], Dict[str, Any]]:
    trace_metadata = params.get("trace_metadata") if isinstance(params.get("trace_metadata"), dict) else {}
    run_id = trace_metadata.get("run_id") if isinstance(trace_metadata, dict) else None
    run_id = str(run_id).strip() if isinstance(run_id, str) and run_id.strip() else None
    tags: Dict[str, Any] = {
        "kind": "generated_media",
        "source": "llm_call",
        "modality": modality,
        "task": task,
    }
    if isinstance(model, str) and model.strip():
        tags["model"] = model.strip()
    if isinstance(trace_metadata, dict):
        for key in (
            "workflow_id",
            "node_id",
            "step_id",
            "effect_idempotency_key",
            "actor_id",
            "session_id",
            "parent_run_id",
            "request_id",
        ):
            raw = trace_metadata.get(key)
            if raw is not None and str(raw).strip():
                tags[key] = str(raw)
    output_run_id, output_tags = _output_runtime_metadata(params.get("output"))
    if run_id is None and output_run_id:
        run_id = output_run_id
    if output_tags:
        tags.update(output_tags)
    return run_id, tags


def _mime_type_for_path(path: str, *, fallback: str = "application/octet-stream") -> str:
    guessed, _enc = mimetypes.guess_type(str(path or ""))
    return str(guessed or fallback)


def _promote_text_param_to_prompt(prompt: Any, params: Dict[str, Any]) -> str:
    prompt_s = str(prompt or "")
    if "text" not in params:
        return prompt_s
    text_value = params.pop("text")
    if prompt_s.strip():
        return prompt_s
    return "" if text_value is None else str(text_value)


def _redact_data_urls_for_observability(value: Any) -> Any:
    if isinstance(value, str):
        raw = value.strip()
        if raw.lower().startswith("data:") and ";base64," in raw[:160].lower():
            header, b64 = raw.split(",", 1)
            try:
                size_bytes = len(base64.b64decode("".join(b64.split()), validate=False))
            except Exception:
                size_bytes = None
            size_label = f"{size_bytes} bytes" if isinstance(size_bytes, int) else "unknown size"
            return f"{header},<redacted {size_label}>"
        return value
    if isinstance(value, list):
        return [_redact_data_urls_for_observability(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _redact_data_urls_for_observability(v) for k, v in value.items()}
    return value


def _data_url_for_file(path: str) -> tuple[str, str, int]:
    with open(path, "rb") as f:
        raw = f.read()
    mime = _mime_type_for_path(path)
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}", mime, len(raw)


def _decode_data_url(value: str) -> tuple[bytes, str]:
    raw = str(value or "").strip()
    if not raw.lower().startswith("data:") or "," not in raw:
        raise ValueError("media data URL must start with data: and contain a comma")
    header, data = raw.split(",", 1)
    mime = header[5:].split(";", 1)[0].strip().lower() or "application/octet-stream"
    if ";base64" not in header.lower():
        raise ValueError("media data URL must be base64 encoded")
    return base64.b64decode("".join(data.split()), validate=True), mime


def _data_url_for_media_item(item: Any) -> tuple[str, str, Optional[int]]:
    if isinstance(item, str) and item.strip():
        raw_item = item.strip()
        if raw_item.lower().startswith("data:"):
            raw, mime = _decode_data_url(raw_item)
            return raw_item, mime, len(raw)
        if raw_item.startswith(("http://", "https://")):
            return raw_item, _mime_type_for_path(raw_item, fallback=""), None

    path = _media_path_from_item(item)
    if path:
        return _data_url_for_file(path)

    if isinstance(item, (bytes, bytearray)):
        raw = bytes(item)
        data_url = f"data:application/octet-stream;base64,{base64.b64encode(raw).decode('ascii')}"
        return data_url, "application/octet-stream", len(raw)

    if not isinstance(item, dict):
        raise ValueError("Remote media item must be a file path, URL, data URL, or content bytes.")

    for key in ("url", "uri"):
        raw_url = item.get(key)
        if not isinstance(raw_url, str) or not raw_url.strip():
            continue
        url = raw_url.strip()
        if url.lower().startswith("data:"):
            raw, mime = _decode_data_url(url)
            return url, mime, len(raw)
        if url.startswith(("http://", "https://")):
            return url, _media_mime_from_item(item), None
        raise ValueError("Remote media URL must be http(s) or a data URL.")

    content = None
    for key in ("content", "data", "bytes"):
        if key in item:
            content = item.get(key)
            break
    if isinstance(content, (bytes, bytearray)):
        raw = bytes(content)
    elif isinstance(content, str) and content.strip().lower().startswith("data:"):
        raw, mime = _decode_data_url(content)
        return content.strip(), mime, len(raw)
    elif isinstance(content, str) and str(item.get("content_format") or item.get("contentFormat") or "").strip().lower() == "base64":
        raw = base64.b64decode("".join(content.strip().split()), validate=True)
    elif content is None:
        raise ValueError("Remote media item is missing file path, URL, data URL, or content bytes.")
    else:
        raise ValueError("Remote media content must be bytes, base64, or a data URL.")

    mime = _media_mime_from_item(item) or "application/octet-stream"
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}", mime, len(raw)


def _media_path_from_item(item: Any) -> Optional[str]:
    if isinstance(item, str) and item.strip():
        return item.strip()
    if isinstance(item, dict):
        for key in ("file_path", "filePath", "path"):
            raw = item.get(key)
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
    return None


def _media_mime_from_item(item: Any) -> str:
    if isinstance(item, dict):
        for key in ("content_type", "mime_type", "mimeType", "mime"):
            raw = item.get(key)
            if isinstance(raw, str) and raw.strip():
                return raw.strip().lower()
    path = _media_path_from_item(item)
    if path:
        return _mime_type_for_path(path, fallback="").lower()
    return ""


def _is_audio_media_item(item: Any) -> bool:
    mime = _media_mime_from_item(item)
    if mime.startswith("audio/"):
        return True
    if isinstance(item, dict):
        raw_type = item.get("type") or item.get("media_type") or item.get("mediaType")
        if isinstance(raw_type, str) and raw_type.strip().lower() == "audio":
            return True
    return False


def _is_image_media_item(item: Any) -> bool:
    mime = _media_mime_from_item(item)
    if mime.startswith("image/"):
        return True
    if isinstance(item, dict):
        raw_type = item.get("type") or item.get("media_type") or item.get("mediaType")
        if isinstance(raw_type, str) and raw_type.strip().lower() == "image":
            return True
    path = _media_path_from_item(item)
    if path:
        guessed = _mime_type_for_path(path, fallback="")
        return guessed.startswith("image/")
    return False


def _media_role_from_item(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    for key in ("role", "purpose", "kind"):
        raw = item.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip().lower()
    return ""


def _text_from_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if str(item.get("type") or "").strip().lower() != "text":
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        return "\n".join(parts).strip()
    return ""


def _remote_media_content_items(*, text: str, media: Optional[List[Any]]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if str(text or ""):
        items.append({"type": "text", "text": str(text or "")})
    for media_item in list(media or []):
        data_url, mime, _size = _data_url_for_media_item(media_item)
        if mime.lower().startswith("image/"):
            items.append({"type": "image_url", "image_url": {"url": data_url}})
        else:
            items.append({"type": "file", "file_url": {"url": data_url}})
    return items


def _merge_remote_media_content(existing: Any, *, media: Optional[List[Any]]) -> List[Dict[str, Any]]:
    if isinstance(existing, list):
        items: List[Dict[str, Any]] = []
        for item in existing:
            if isinstance(item, dict):
                items.append(dict(item))
            elif item is not None:
                items.append({"type": "text", "text": str(item)})
        items.extend(_remote_media_content_items(text="", media=media))
        return items

    existing_text = existing if isinstance(existing, str) else ""
    return _remote_media_content_items(text=existing_text, media=media)


class RemoteAbstractCoreLLMClient:
    """Remote LLM client calling an AbstractCore server endpoint."""

    def __init__(
        self,
        *,
        server_base_url: str,
        model: str,
        # Runtime authority default: long-running workflow steps may legitimately take a long time.
        # Keep this aligned with AbstractRuntime's orchestration defaults.
        timeout_s: Optional[float] = None,
        headers: Optional[Dict[str, str]] = None,
        request_sender: Optional[RequestSender] = None,
        artifact_store: Optional[Any] = None,
        core_config_file: Optional[str | Path] = None,
        capability_defaults: Optional[Any] = None,
    ):
        from .constants import DEFAULT_LLM_TIMEOUT_S

        self._server_base_url = _core_server_root_url(server_base_url)
        self._model = model
        self._timeout_s = float(timeout_s) if timeout_s is not None else DEFAULT_LLM_TIMEOUT_S
        self._headers = dict(headers or {})
        self._sender = request_sender or HttpxRequestSender()
        self._artifact_store = artifact_store
        self._core_config_file = _coerce_core_config_file(core_config_file)
        self._capability_defaults = deepcopy(_normalize_core_capability_defaults(capability_defaults))
        self._capability_defaults_explicit = capability_defaults is not None
        self._capability_defaults_lock = threading.RLock()
        # Negative cache: set once the server unambiguously reports the
        # key_meta route missing (older core); transient failures keep retrying.
        self._prompt_cache_key_meta_route_unsupported = False
        _attach_core_execution_context_to_client(
            self,
            core_config_file=self._core_config_file,
            capability_defaults=self._capability_defaults if capability_defaults is not None else None,
        )

    def default_prompt_cache_identity(self) -> Tuple[Optional[str], Optional[str]]:
        return "remote", self._model

    def set_capability_defaults(self, capability_defaults: Optional[Any]) -> bool:
        """Refresh caller-owned policy without changing the remote endpoint/model.

        Replace snapshots instead of editing dictionaries used by in-flight
        requests. None releases the scope; an empty mapping explicitly clears
        its policies and must not inherit the server's configured MTP default.
        """
        routes = deepcopy(_normalize_core_capability_defaults(capability_defaults))
        explicit = capability_defaults is not None
        with self._capability_defaults_lock:
            if routes == self._capability_defaults and explicit == self._capability_defaults_explicit:
                return False
            self._capability_defaults = routes
            self._capability_defaults_explicit = explicit
            self._abstractcore_capability_defaults = deepcopy(routes) if explicit else None
        return True

    def _scoped_speculation_default(self) -> Any:
        """Resolve caller-owned scope; otherwise leave policy at the remote host.

        The existing speculation wire is sufficient: scoped absence is Off,
        whereas absence of a scope must not suppress the remote host's policy.
        No backend support/head decisions are made on this client machine.
        """
        with self._capability_defaults_lock:
            routes = getattr(self, "_abstractcore_capability_defaults", None)
            has_routes = routes is not None or self._capability_defaults_explicit
            if routes is None and has_routes:
                routes = self._capability_defaults
            routes = deepcopy(routes)
            config_file = getattr(self, "_abstractcore_config_file", None) or self._core_config_file
        if not has_routes and not config_file:
            return _UNSET
        from abstractcore.providers.speculation import configured_speculation_default

        policy = configured_speculation_default(
            config_file=config_file, capability_defaults=routes if has_routes else None,
        )
        return False if policy is None else deepcopy(policy)

    def get_model_residency_capabilities(self, **kwargs: Any) -> Dict[str, Any]:
        _ = kwargs
        tasks = {
            "text_generation": _model_residency_capability_task(
                task="text_generation",
                supported=True,
                truth_source="abstractcore.server./acore/models",
            ),
            "image_generation": _model_residency_capability_task(
                task="image_generation",
                supported=True,
                truth_source="abstractcore.server./acore/models",
            ),
            "image_to_image": _model_residency_capability_task(
                task="image_to_image",
                supported=True,
                truth_source="abstractcore.server./acore/models",
                extra={"shares_backend_cache_with": "image_generation"},
            ),
            "image_upscale": _model_residency_capability_task(
                task="image_upscale",
                supported=True,
                truth_source="abstractcore.server./acore/models",
                extra={"shares_backend_cache_with": "image_generation"},
            ),
            "text_to_video": _model_residency_capability_task(
                task="text_to_video",
                supported=True,
                truth_source="abstractcore.server./acore/models",
            ),
            "video_generation": _model_residency_capability_task(
                task="video_generation",
                supported=True,
                truth_source="abstractcore.server./acore/models",
                extra={"includes_tasks": ["text_to_video", "image_to_video"]},
            ),
            "image_to_video": _model_residency_capability_task(
                task="image_to_video",
                supported=True,
                truth_source="abstractcore.server./acore/models",
            ),
            "tts": _model_residency_capability_task(
                task="tts",
                supported=True,
                truth_source="abstractcore.server./acore/models",
            ),
            "stt": _model_residency_capability_task(
                task="stt",
                supported=True,
                truth_source="abstractcore.server./acore/models",
            ),
            "music_generation": _model_residency_capability_task(
                task="music_generation",
                supported=False,
                truth_source="abstractcore.server./acore/models",
                reason="The current AbstractCore server residency control plane does not implement music_generation.",
            ),
        }
        return {
            "ok": True,
            "supported": True,
            "operation": "capabilities",
            "mode": "remote_core_server",
            "source": "abstractruntime.remote",
            "relay_only": True,
            "tasks": tasks,
            "supported_tasks": [task for task, info in tasks.items() if info.get("supported") is True],
            "unsupported_tasks": [task for task, info in tasks.items() if info.get("supported") is not True],
            "diagnostics": {"source": "abstractruntime.remote", "server_base_url": self._server_base_url},
        }

    def _prompt_cache_proxy_fields(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        base_url = kwargs.get("base_url")
        if isinstance(base_url, str) and base_url.strip():
            out["base_url"] = base_url.strip()
        return out

    def _headers_with_provider_api_key(self, api_key: Optional[str]) -> Dict[str, str]:
        headers = dict(self._headers)
        if isinstance(api_key, str) and api_key.strip():
            _set_header_case_insensitive(
                headers,
                _ABSTRACTCORE_PROVIDER_API_KEY_HEADER,
                api_key.strip(),
            )
        return headers

    def _discovery_error_payload(self, *, source: str, error: Any) -> Dict[str, Any]:
        response = getattr(error, "response", None)
        status_code = None
        try:
            status_code = int(getattr(response, "status_code", None))
        except Exception:
            status_code = None
        detail: Any = None
        if response is not None:
            body = getattr(response, "body", None)
            if isinstance(body, dict):
                detail = _jsonable(body)
            else:
                json_fn = getattr(response, "json", None)
                if callable(json_fn):
                    try:
                        detail = _jsonable(json_fn())
                    except Exception:
                        detail = None
                if detail is None:
                    text = getattr(response, "text", None)
                    if isinstance(text, str) and text.strip():
                        detail = text.strip()
        payload = {
            "available": False,
            "route_available": response is not None,
            "source": source,
            "stale": False,
            "refreshed_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "error": str(error),
        }
        if status_code is not None:
            payload["status_code"] = status_code
        if detail is not None:
            payload["upstream_error"] = detail
        return payload

    def _discovery_get(
        self,
        path: str,
        *,
        source: str,
        query: Optional[Dict[str, Any]] = None,
        provider_api_key: Optional[str] = None,
        v1: bool = True,
        timeout_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        encoded: Dict[str, str] = {}
        for key, raw in dict(query or {}).items():
            if raw is None:
                continue
            if isinstance(raw, bool):
                encoded[str(key)] = "true" if raw else "false"
                continue
            text = str(raw).strip()
            if text:
                encoded[str(key)] = text
        url = _join_core_v1_url(self._server_base_url, path) if v1 else _join_core_control_url(self._server_base_url, path)
        if encoded:
            url = f"{url}?{urlencode(encoded)}"
        try:
            raw = self._sender.get(
                url,
                headers=self._headers_with_provider_api_key(provider_api_key),
                timeout=float(timeout_s) if timeout_s is not None else self._timeout_s,
            )
            resp, _resp_headers = _unwrap_http_response(raw)
        except Exception as exc:
            return self._discovery_error_payload(source=source, error=exc)
        if not isinstance(resp, dict):
            return self._discovery_error_payload(source=source, error=f"invalid discovery response for {path}")
        out = dict(resp)
        out.setdefault("available", True)
        out["route_available"] = True
        out["source"] = source
        out.setdefault("stale", False)
        out.setdefault("error", None)
        out["refreshed_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        return out

    def get_model_capabilities(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        target_model = str(model_name or self._model or "").strip() or self._model
        from .discovery_queries import local_get_model_capabilities

        payload = local_get_model_capabilities(target_model)
        capabilities = payload.get("capabilities") if isinstance(payload, dict) else None
        if isinstance(capabilities, dict):
            return capabilities
        from abstractruntime.core.vars import DEFAULT_MAX_TOKENS

        return {"max_tokens": DEFAULT_MAX_TOKENS}

    def lookup_model_capabilities(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        target_model = str(model_name or self._model or "").strip() or self._model
        from .discovery_queries import local_get_model_capabilities

        return local_get_model_capabilities(target_model)

    def get_execution_capabilities(
        self, model_name: Optional[str] = None, *, provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        # The remote host's installed backend and loaded instance are the
        # authority. Never substitute this client's local model registry.
        payload = self._discovery_get(
            "/models/execution-capabilities", source="abstractcore.remote",
            query={"model_name": model_name or self._model, "provider": provider},
        )
        policy = self._scoped_speculation_default()
        if policy is not _UNSET and isinstance(payload.get("speculation"), dict):
            speculation = dict(payload["speculation"])
            speculation["default"] = deepcopy(policy)
            speculation["effective_default"] = deepcopy(policy) if speculation.get("supported") is True else False
            payload["speculation"] = speculation
        return payload

    def list_providers(
        self,
        *,
        include_models: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        call_kwargs = dict(kwargs)
        payload = self._discovery_get(
            "/providers",
            source="abstractcore.remote",
            query={"include_models": bool(include_models)},
            v1=False,
            timeout_s=call_kwargs.get("timeout_s"),
        )
        providers = payload.get("providers")
        items = [dict(item) for item in list(providers or []) if isinstance(item, dict)]
        items.sort(key=lambda item: str(item.get("name") or ""))
        provider_hint = None
        model_hint = str(self._model or "").strip() or None
        if model_hint and "/" in model_hint:
            maybe_provider, maybe_model = model_hint.split("/", 1)
            if maybe_provider.strip() and maybe_model.strip():
                provider_hint = maybe_provider.strip().lower()
                model_hint = maybe_model.strip()
        payload["items"] = items
        payload["default_provider"] = provider_hint
        payload["default_model"] = model_hint
        payload["available"] = bool(items)
        return payload

    def list_provider_models(
        self,
        provider_name: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        provider_text = str(provider_name or "").strip()
        if not provider_text:
            return self._discovery_error_payload(source="abstractcore.remote", error="provider_name is required")
        call_kwargs = dict(kwargs)
        provider_api_key = _pop_provider_api_key(call_kwargs)
        capability_route = call_kwargs.get("capability_route", call_kwargs.get("capability_routes"))
        if isinstance(capability_route, set):
            capability_route = ",".join(str(item).strip() for item in sorted(capability_route, key=str) if str(item).strip())
        elif isinstance(capability_route, (list, tuple)):
            capability_route = ",".join(str(item).strip() for item in capability_route if str(item).strip())
        payload = self._discovery_get(
            "/models",
            source="abstractcore.remote",
            query={
                "provider": provider_text,
                "base_url": call_kwargs.get("base_url"),
                "input_type": call_kwargs.get("input_type"),
                "output_type": call_kwargs.get("output_type"),
                "capability_route": capability_route,
            },
            provider_api_key=provider_api_key,
            timeout_s=call_kwargs.get("timeout_s"),
        )
        data = payload.get("data")
        out: List[str] = []
        if isinstance(data, list):
            prefix = f"{provider_text.lower()}/"
            for item in data:
                model_id = ""
                if isinstance(item, str):
                    model_id = item.strip()
                elif isinstance(item, dict):
                    model_id = str(item.get("id") or item.get("model") or item.get("name") or "").strip()
                if not model_id:
                    continue
                if model_id.lower().startswith(prefix):
                    model_id = model_id[len(prefix) :]
                out.append(model_id)
        out = sorted({model.strip(): model.strip() for model in out if model.strip()}.values(), key=str.lower)
        payload["provider"] = provider_text
        payload["models"] = out
        payload["available"] = bool(out)
        return payload

    def list_embedding_models(
        self,
        *,
        base_url: Optional[str] = None,
        provider_api_key: Optional[str] = None,
        provider: Optional[str] = None,
        providers_only: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        from .discovery_queries import _dedupe_strings, _provider_models_from_mapping

        provider_text = str(provider or "").strip().lower()
        call_kwargs = dict(kwargs)
        provider_api_key = provider_api_key or _pop_provider_api_key(call_kwargs)
        if providers_only:
            payload = self._discovery_get(
                "/embeddings/providers",
                source="abstractcore.remote",
                query={"provider": provider_text or None},
                provider_api_key=provider_api_key,
                timeout_s=call_kwargs.get("timeout_s"),
            )
            try:
                status_code = int(payload.get("status_code") or 0)
            except Exception:
                status_code = 0
            if not payload.get("available") and status_code in {404, 405}:
                payload = self._discovery_get(
                    "/providers",
                    source="abstractcore.remote",
                    query={"include_models": False},
                    provider_api_key=provider_api_key,
                    v1=False,
                    timeout_s=call_kwargs.get("timeout_s"),
                )
                provider_details = self._embedding_provider_details_from_remote_provider_catalog(
                    payload.get("providers"),
                    provider=provider_text or None,
                )
                providers = _dedupe_strings(
                    [
                        str(item.get("provider") or item.get("id") or "").strip()
                        for item in provider_details
                        if isinstance(item, dict)
                    ]
                )
                payload.update(
                    {
                        "kind": "embedding_providers",
                        "scope": "embedding.text",
                        "provider": provider_text or None,
                        "providers": providers,
                        "available_providers": providers,
                        "embedding_providers": providers,
                        "provider_details": provider_details,
                        "models": [],
                        "embedding_models": [],
                        "models_by_provider": {},
                        "embedding_models_by_provider": {},
                        "provider_models": [],
                        "available": bool(providers),
                        "error": None if providers or not provider_text else f"Unsupported embedding provider: {provider_text}",
                    }
                )
            else:
                payload.setdefault("kind", "embedding_providers")
                payload.setdefault("scope", "embedding.text")
                payload.setdefault("provider", provider_text or None)
                payload.setdefault("models", [])
                payload.setdefault("embedding_models", [])
                payload.setdefault("models_by_provider", {})
                payload.setdefault("embedding_models_by_provider", {})
                payload.setdefault("provider_models", [])
            return payload

        payload = self._discovery_get(
            "/models",
            source="abstractcore.remote",
            query={
                "provider": provider_text or None,
                "output_type": "embeddings",
                "base_url": base_url,
            },
            provider_api_key=provider_api_key,
            timeout_s=call_kwargs.get("timeout_s"),
        )
        data = payload.get("data")
        models_by_provider: Dict[str, List[str]] = {}
        if isinstance(data, list):
            for item in data:
                model_id = ""
                provider_id = ""
                if isinstance(item, str):
                    model_id = item.strip()
                elif isinstance(item, dict):
                    model_id = str(item.get("id") or item.get("model") or item.get("name") or "").strip()
                    provider_id = str(item.get("owned_by") or item.get("provider") or "").strip().lower()
                if not model_id:
                    continue
                if not provider_id and "/" in model_id:
                    maybe_provider, maybe_model = model_id.split("/", 1)
                    if maybe_provider.strip() and maybe_model.strip():
                        provider_id = maybe_provider.strip().lower()
                        model_id = maybe_model.strip()
                if provider_text:
                    prefix = f"{provider_text}/"
                    if model_id.lower().startswith(prefix):
                        model_id = model_id[len(prefix) :]
                    provider_id = provider_text
                if not provider_id:
                    provider_id = "unknown"
                current = models_by_provider.setdefault(provider_id, [])
                current.append(model_id)
        models_by_provider = {
            provider_id: _dedupe_strings(values)
            for provider_id, values in models_by_provider.items()
            if _dedupe_strings(values)
        }
        if provider_text:
            models_by_provider = {
                key: value
                for key, value in models_by_provider.items()
                if key.strip().lower() == provider_text
            }
        models = _dedupe_strings([model for values in models_by_provider.values() for model in values])
        providers = _dedupe_strings(list(models_by_provider.keys()))
        payload["kind"] = "embedding_models"
        payload["scope"] = "embedding.text"
        payload["provider"] = provider_text or None
        payload["providers"] = providers
        payload["available_providers"] = providers
        payload["embedding_providers"] = providers
        payload["models"] = models
        payload["embedding_models"] = models
        payload["models_by_provider"] = models_by_provider
        payload["embedding_models_by_provider"] = models_by_provider
        payload["provider_models"] = _provider_models_from_mapping(models_by_provider)
        payload["available"] = bool(models_by_provider)
        return payload

    def _embedding_provider_details_from_remote_provider_catalog(
        self,
        raw_providers: Any,
        *,
        provider: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        provider_text = str(provider or "").strip().lower()
        details: List[Dict[str, Any]] = []
        if not isinstance(raw_providers, list):
            return details
        for item in raw_providers:
            if not isinstance(item, dict):
                continue
            features = item.get("supported_features")
            if isinstance(features, list):
                feature_keys = {str(feature or "").strip().lower() for feature in features}
                if "embeddings" not in feature_keys:
                    continue
            else:
                continue
            provider_id = str(item.get("provider") or item.get("name") or item.get("id") or "").strip()
            if not provider_id:
                continue
            if provider_text and provider_id.lower() != provider_text:
                continue
            row: Dict[str, Any] = {
                "id": provider_id,
                "provider": provider_id,
                "label": str(item.get("display_name") or provider_id).strip() or provider_id,
            }
            details.append(row)
        return details

    def get_voice_catalog(
        self,
        *,
        base_url: Optional[str] = None,
        provider_api_key: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        providers_only: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        from .discovery_queries import _filter_voice_catalog_response

        call_kwargs = dict(kwargs)
        provider_api_key = provider_api_key or _pop_provider_api_key(call_kwargs)
        payload = self._discovery_get(
            "/audio/voices",
            source="abstractcore.remote",
            query={
                "base_url": base_url,
                "provider": provider,
                "model": model,
                "providers_only": providers_only,
            },
            provider_api_key=provider_api_key,
            timeout_s=call_kwargs.get("timeout_s"),
        )
        return _filter_voice_catalog_response(payload, provider=provider, model=model, providers_only=providers_only)

    def list_tts_models(
        self,
        *,
        base_url: Optional[str] = None,
        provider_api_key: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        from .discovery_queries import _filter_provider_model_catalog_response

        call_kwargs = dict(kwargs)
        provider_api_key = provider_api_key or _pop_provider_api_key(call_kwargs)
        payload = self._discovery_get(
            "/audio/speech/models",
            source="abstractcore.remote",
            query={"base_url": base_url, "provider": provider},
            provider_api_key=provider_api_key,
            timeout_s=call_kwargs.get("timeout_s"),
        )
        return _filter_provider_model_catalog_response(
            payload,
            provider=provider,
            model_keys=("models_by_provider", "tts_models_by_provider"),
        )

    def list_stt_models(
        self,
        *,
        base_url: Optional[str] = None,
        provider_api_key: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        from .discovery_queries import _filter_provider_model_catalog_response

        call_kwargs = dict(kwargs)
        provider_api_key = provider_api_key or _pop_provider_api_key(call_kwargs)
        payload = self._discovery_get(
            "/audio/transcriptions/models",
            source="abstractcore.remote",
            query={"base_url": base_url, "provider": provider},
            provider_api_key=provider_api_key,
            timeout_s=call_kwargs.get("timeout_s"),
        )
        return _filter_provider_model_catalog_response(
            payload,
            provider=provider,
            model_keys=("models_by_provider", "stt_models_by_provider"),
        )

    def list_music_providers(
        self,
        *,
        task: Optional[str] = None,
        base_url: Optional[str] = None,
        provider_api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        from .discovery_queries import _dedupe_strings, _music_provider_id

        call_kwargs = dict(kwargs)
        provider_api_key = provider_api_key or _pop_provider_api_key(call_kwargs)
        task_value = str(task or "").strip() or "text_to_music"
        payload = self._discovery_get(
            "/audio/music/providers",
            source="abstractmusic.remote",
            query={"task": task_value, "base_url": base_url},
            provider_api_key=provider_api_key,
            timeout_s=call_kwargs.get("timeout_s"),
        )
        details = [dict(item) for item in list(payload.get("providers") or []) if isinstance(item, dict)]
        providers = _dedupe_strings([_music_provider_id(item) for item in details if _music_provider_id(item)])
        payload = dict(payload)
        payload["task"] = task_value
        payload["providers"] = providers
        payload["available_providers"] = providers
        payload["provider_details"] = details
        return payload

    def list_music_models(
        self,
        *,
        task: Optional[str] = None,
        base_url: Optional[str] = None,
        provider_api_key: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        from .discovery_queries import (
            _dedupe_strings,
            _music_model_provider,
            _music_models_by_provider,
            _provider_models_from_mapping,
        )

        call_kwargs = dict(kwargs)
        provider_api_key = provider_api_key or _pop_provider_api_key(call_kwargs)
        task_value = str(task or "").strip() or "text_to_music"
        payload = self._discovery_get(
            "/audio/music/models",
            source="abstractmusic.remote",
            query={"task": task_value, "provider": provider, "base_url": base_url},
            provider_api_key=provider_api_key,
            timeout_s=call_kwargs.get("timeout_s"),
        )
        models = [dict(item) for item in list(payload.get("models") or []) if isinstance(item, dict)]
        provider_value = str(provider or "").strip()
        if provider_value:
            models = [
                item for item in models if _music_model_provider(item).lower() == provider_value.lower()
            ]
        models_by_provider = _music_models_by_provider(models)
        providers = _dedupe_strings(
            [_music_model_provider(item) for item in models if _music_model_provider(item)]
        )
        payload = dict(payload)
        payload["task"] = task_value
        payload["provider"] = provider_value or None
        payload["models"] = models
        payload["providers"] = providers
        payload["available_providers"] = providers
        payload["models_by_provider"] = models_by_provider
        payload["provider_models"] = _provider_models_from_mapping(models_by_provider)
        return payload

    def list_vision_provider_models(
        self,
        *,
        task: Optional[str] = None,
        base_url: Optional[str] = None,
        provider_api_key: Optional[str] = None,
        provider: Optional[str] = None,
        providers_only: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        from .discovery_queries import _filter_vision_provider_models_response

        call_kwargs = dict(kwargs)
        provider_api_key = provider_api_key or _pop_provider_api_key(call_kwargs)
        payload = self._discovery_get(
            "/vision/provider_models",
            source="abstractcore.remote",
            query={
                "task": task,
                "base_url": base_url,
                "provider": provider,
                "providers_only": providers_only,
            },
            provider_api_key=provider_api_key,
            timeout_s=call_kwargs.get("timeout_s"),
        )
        return _filter_vision_provider_models_response(
            payload,
            provider=provider,
            providers_only=providers_only,
        )

    def list_cached_vision_models(
        self,
        *,
        task: Optional[str] = None,
        base_url: Optional[str] = None,
        provider_api_key: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        call_kwargs = dict(kwargs)
        provider_api_key = provider_api_key or _pop_provider_api_key(call_kwargs)
        payload = self._discovery_get(
            "/vision/models",
            source="abstractcore.remote",
            query={"task": task, "base_url": base_url, "provider": provider},
            provider_api_key=provider_api_key,
            timeout_s=call_kwargs.get("timeout_s"),
        )
        items = payload.get("models")
        models = [dict(item) for item in list(items or []) if isinstance(item, dict)]
        provider_value = str(provider or "").strip().lower()
        if provider_value:
            models = [
                item
                for item in models
                if str(item.get("provider") or "").strip().lower() == provider_value
            ]
        payload["models"] = models
        payload["available"] = bool(models)
        return payload

    def list_vision_adapters(
        self,
        *,
        model: Optional[str] = None,
        task: Optional[str] = None,
        base_url: Optional[str] = None,
        provider_api_key: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        call_kwargs = dict(kwargs)
        provider_api_key = provider_api_key or _pop_provider_api_key(call_kwargs)
        payload = self._discovery_get(
            "/vision/adapters",
            source="abstractcore.remote",
            query={
                "model": model,
                "task": task,
                "base_url": base_url,
                "provider": provider,
            },
            provider_api_key=provider_api_key,
            timeout_s=call_kwargs.get("timeout_s"),
        )
        adapters = [dict(item) for item in list(payload.get("adapters") or []) if isinstance(item, dict)]
        provider_value = str(provider or "").strip().lower()
        if provider_value:
            adapters = [
                item
                for item in adapters
                if str(item.get("provider") or item.get("raw", {}).get("provider") or "").strip().lower() == provider_value
            ]
        payload["model"] = str(model or "").strip() or None
        payload["task"] = str(task or "").strip() or None
        payload["provider"] = str(provider or "").strip() or None
        payload["adapters"] = adapters
        payload["count"] = len(adapters)
        payload["available"] = bool(adapters)
        return payload

    def _prompt_cache_get(self, path: str, *, operation: str, kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        call_kwargs = dict(kwargs or {})
        provider_api_key = _pop_provider_api_key(call_kwargs)
        proxy_fields = self._prompt_cache_proxy_fields(call_kwargs)
        url = _join_core_control_url(self._server_base_url, path)
        if proxy_fields:
            url = f"{url}?{urlencode(proxy_fields)}"
        try:
            raw = self._sender.get(
                url,
                headers=self._headers_with_provider_api_key(provider_api_key),
                timeout=self._timeout_s,
            )
            resp, _resp_headers = _unwrap_http_response(raw)
        except Exception as e:
            return {
                "supported": False,
                "operation": operation,
                "error": str(e),
                "capabilities": {"supported": False, "mode": "none"},
            }

        if isinstance(resp, dict):
            return resp
        return {
            "supported": False,
            "operation": operation,
            "error": f"invalid prompt cache {operation} response",
            "capabilities": {"supported": False, "mode": "none"},
        }

    def _prompt_cache_post(
        self,
        path: str,
        *,
        operation: str,
        body: Dict[str, Any],
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        call_kwargs = dict(kwargs or {})
        provider_api_key = _pop_provider_api_key(call_kwargs)
        url = _join_core_control_url(self._server_base_url, path)
        payload = dict(body)
        payload.update(self._prompt_cache_proxy_fields(call_kwargs))
        try:
            raw = self._sender.post(
                url,
                headers=self._headers_with_provider_api_key(provider_api_key),
                json=payload,
                timeout=self._timeout_s,
            )
            resp, _resp_headers = _unwrap_http_response(raw)
        except Exception as e:
            return {
                "supported": False,
                "operation": operation,
                "error": str(e),
                "capabilities": {"supported": False, "mode": "none"},
            }

        if isinstance(resp, dict):
            return resp
        return {
            "supported": False,
            "operation": operation,
            "error": f"invalid prompt cache {operation} response",
            "capabilities": {"supported": False, "mode": "none"},
        }

    def get_prompt_cache_capabilities(self, **kwargs: Any) -> Dict[str, Any]:
        return self._prompt_cache_get("/acore/prompt_cache/capabilities", operation="capabilities", kwargs=kwargs)

    def get_prompt_cache_stats(self, **kwargs: Any) -> Dict[str, Any]:
        return self._prompt_cache_get("/acore/prompt_cache/stats", operation="stats", kwargs=kwargs)

    def get_memory_snapshot(self, **kwargs: Any) -> Dict[str, Any]:
        _ = kwargs
        url = _join_core_control_url(self._server_base_url, "/acore/memory")
        try:
            raw = self._sender.get(url, headers=dict(self._headers), timeout=self._timeout_s)
            resp, _resp_headers = _unwrap_http_response(raw)
        except Exception as exc:  # noqa: BLE001 - relay surface never raises
            return {"ok": False, "error": str(exc)}
        if not isinstance(resp, dict):
            return {"ok": False, "error": "invalid memory snapshot response"}
        out = dict(resp)
        out.pop("ok", None)
        return out

    def list_session_prompt_caches(self, session_id: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        # No-selector stats: the core server enumerates its loaded runtimes.
        result = self._prompt_cache_get("/acore/prompt_cache/stats", operation="stats", kwargs=kwargs)
        if not isinstance(result, dict):
            return {"ok": False, "error": "invalid prompt cache stats response", "caches": []}
        runtimes = result.get("runtimes")
        if not isinstance(runtimes, list):
            error = result.get("error") or "core server did not return cross-runtime prompt cache stats"
            return {"ok": False, "error": str(error), "caches": []}
        rows: List[Dict[str, Any]] = []
        for entry in runtimes:
            if not isinstance(entry, dict):
                continue
            rows.extend(
                _session_prompt_cache_rows_from_stats(
                    entry.get("stats"),
                    provider=str(entry.get("provider") or "").strip().lower() or None,
                    model=str(entry.get("model") or "").strip() or None,
                    runtime_id=str(entry.get("runtime_id") or "").strip() or None,
                    session_id=session_id,
                )
            )
        return {"ok": True, "caches": rows}

    def clear_session_prompt_caches(self, session_id: str, **kwargs: Any) -> Dict[str, Any]:
        session_s = str(session_id or "").strip()
        if not session_s:
            return {
                "ok": False,
                "error": "clear_session_prompt_caches requires a session_id",
                "cleared": [],
                "count": 0,
            }
        listing = self.list_session_prompt_caches(session_id=session_s, **dict(kwargs))
        if listing.get("ok") is not True:
            return {
                "ok": False,
                "error": str(listing.get("error") or "unable to enumerate session prompt caches"),
                "cleared": [],
                "count": 0,
            }
        cleared: List[Dict[str, Any]] = []
        count = 0
        for row in listing.get("caches") or []:
            out = dict(row)
            body: Dict[str, Any] = {"key": row.get("key")}
            runtime_id = str(row.get("runtime_id") or "").strip()
            if runtime_id:
                body["runtime_id"] = runtime_id
            else:
                if row.get("provider"):
                    body["provider"] = row.get("provider")
                if row.get("model"):
                    body["model"] = row.get("model")
            result = self._prompt_cache_post("/acore/prompt_cache/clear", operation="clear", body=body, kwargs={})
            if isinstance(result, dict) and result.get("ok") is True:
                out["cleared"] = True
                count += 1
            else:
                out["cleared"] = False
                error = result.get("error") if isinstance(result, dict) else None
                out["error"] = str(error or "prompt cache clear failed")
            cleared.append(out)
        return {"ok": True, "cleared": cleared, "count": count}

    def _maybe_stamp_prompt_cache_attribution(
        self,
        *,
        key: Optional[str],
        attribution: Optional[Dict[str, Any]],
        effective_model: Optional[str],
    ) -> None:
        """Best-effort session attribution relay (`POST /acore/prompt_cache/key_meta`).

        AFTER the chat call: the server-side cache entry exists only once the
        generate that created it has completed (a missing key is refused with
        `code: prompt_cache_missing_key`). Stamped after EVERY generate that
        used a derived key — a done-set would go stale against core's cache
        LRU, and the merge is idempotent server-side. Never fails the LLM
        call. A server that unambiguously lacks the route (404/405) is
        negative-cached per client instance; transient transport failures
        keep retrying on the next generate."""
        if getattr(self, "_prompt_cache_key_meta_route_unsupported", False):
            return
        key_s = str(key or "").strip()
        if not key_s or not isinstance(attribution, dict):
            return
        if not str(attribution.get("session_id") or "").strip():
            return
        provider: Optional[str] = None
        model: Optional[str] = None
        model_s = str(effective_model or "").strip()
        if "/" in model_s:
            maybe_provider, maybe_model = model_s.split("/", 1)
            if maybe_provider.strip() and maybe_model.strip():
                provider = maybe_provider.strip().lower()
                model = maybe_model.strip()
        if not provider or not model:
            # The key_meta route needs a runtime selector; a bare model name
            # cannot address one, so the stamp is skipped rather than guessed.
            return
        body = {
            "provider": provider,
            "model": model,
            "key": key_s,
            "meta": {k: v for k, v in attribution.items() if v is not None},
        }
        url = _join_core_control_url(self._server_base_url, "/acore/prompt_cache/key_meta")
        try:
            self._sender.post(url, headers=dict(self._headers), json=body, timeout=self._timeout_s)
        except Exception as exc:  # noqa: BLE001 - best-effort; never fails the call
            status_code = None
            try:
                status_code = int(getattr(getattr(exc, "response", None), "status_code", None))
            except Exception:
                status_code = None
            if status_code in (404, 405):
                self._prompt_cache_key_meta_route_unsupported = True

    def prompt_cache_set(
        self,
        *,
        key: str,
        make_default: bool = True,
        ttl_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {"key": key, "make_default": bool(make_default)}
        if ttl_s is not None:
            body["ttl_s"] = ttl_s
        return self._prompt_cache_post("/acore/prompt_cache/set", operation="set", body=body, kwargs=kwargs)

    def prompt_cache_update(
        self,
        *,
        key: str,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        add_generation_prompt: bool = False,
        ttl_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "key": key,
            "prompt": prompt,
            "messages": messages,
            "system_prompt": system_prompt,
            "tools": tools,
            "add_generation_prompt": bool(add_generation_prompt),
        }
        if ttl_s is not None:
            body["ttl_s"] = ttl_s
        return self._prompt_cache_post(
            "/acore/prompt_cache/update",
            operation="update",
            body={k: v for k, v in body.items() if v is not None},
            kwargs=kwargs,
        )

    def prompt_cache_fork(
        self,
        *,
        from_key: str,
        to_key: str,
        make_default: bool = False,
        ttl_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "from_key": from_key,
            "to_key": to_key,
            "make_default": bool(make_default),
        }
        if ttl_s is not None:
            body["ttl_s"] = ttl_s
        return self._prompt_cache_post("/acore/prompt_cache/fork", operation="fork", body=body, kwargs=kwargs)

    def prompt_cache_clear(
        self,
        *,
        key: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {}
        if key is not None:
            body["key"] = key
        return self._prompt_cache_post("/acore/prompt_cache/clear", operation="clear", body=body, kwargs=kwargs)

    def prompt_cache_prepare_modules(
        self,
        *,
        namespace: str,
        modules: List[Dict[str, Any]],
        make_default: bool = False,
        ttl_s: Optional[float] = None,
        version: int = 1,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "namespace": namespace,
            "modules": modules,
            "make_default": bool(make_default),
            "version": int(version),
        }
        if ttl_s is not None:
            body["ttl_s"] = ttl_s
        thinking = kwargs.pop("thinking", None)
        if thinking is not None:
            body["thinking"] = thinking
        return self._prompt_cache_post(
            "/acore/prompt_cache/prepare_modules",
            operation="prepare_modules",
            body=body,
            kwargs=kwargs,
        )

    def list_prompt_cache_exports(self, **kwargs: Any) -> Dict[str, Any]:
        _ = kwargs
        return _prompt_cache_export_local_only_payload(operation="list_exports")

    def prompt_cache_export(
        self,
        *,
        name: str,
        key: str,
        q8: bool = False,
        meta: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        _ = (name, key, q8, meta, kwargs)
        return _prompt_cache_export_local_only_payload(operation="export")

    def prompt_cache_import(
        self,
        *,
        name: str,
        key: Optional[str] = None,
        make_default: bool = True,
        clear_existing: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        _ = (name, key, make_default, clear_existing, kwargs)
        return _prompt_cache_export_local_only_payload(operation="import")

    def _default_bloc_target_fields(self) -> Dict[str, Any]:
        provider = None
        model = str(self._model or "").strip() or None
        if model and "/" in model:
            maybe_provider, maybe_model = model.split("/", 1)
            if maybe_provider.strip() and maybe_model.strip():
                provider = maybe_provider.strip().lower()
                model = maybe_model.strip()
        if provider is None or model is None:
            return {}
        return {"provider": provider, "model": model}

    def _bloc_proxy_fields(self, kwargs: Dict[str, Any], *, include_default_target: bool = True) -> Dict[str, Any]:
        base_url = kwargs.get("base_url")
        if isinstance(base_url, str) and base_url.strip():
            return {"base_url": base_url.strip()}
        out = self._default_bloc_target_fields() if include_default_target else {}
        for key in ("runtime_id", "provider", "model"):
            raw = kwargs.get(key)
            if isinstance(raw, str) and raw.strip():
                out[key] = raw.strip()
        return {key: value for key, value in out.items() if value is not None}

    def _bloc_get(
        self,
        path: str,
        *,
        operation: str,
        query: Optional[Dict[str, Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        include_default_target: bool = True,
    ) -> Dict[str, Any]:
        call_kwargs = dict(kwargs or {})
        provider_api_key = _pop_provider_api_key(call_kwargs)
        timeout_s = float(call_kwargs.get("timeout_s")) if call_kwargs.get("timeout_s") is not None else self._timeout_s
        payload = dict(self._bloc_proxy_fields(call_kwargs, include_default_target=include_default_target))
        for key, value in dict(query or {}).items():
            if value is not None:
                payload[key] = value
        encoded: Dict[str, str] = {}
        for key, raw in payload.items():
            if raw is None:
                continue
            if isinstance(raw, bool):
                encoded[str(key)] = "true" if raw else "false"
                continue
            text = str(raw).strip()
            if text:
                encoded[str(key)] = text
        url = _join_core_control_url(self._server_base_url, path)
        if encoded:
            url = f"{url}?{urlencode(encoded)}"
        try:
            raw = self._sender.get(
                url,
                headers=self._headers_with_provider_api_key(provider_api_key),
                timeout=timeout_s,
            )
            resp, _resp_headers = _unwrap_http_response(raw)
        except Exception as exc:
            return {"ok": False, "operation": operation, "error": str(exc), "diagnostics": {"source": "abstractcore.remote"}}
        if isinstance(resp, dict):
            return resp
        return {"ok": False, "operation": operation, "error": f"invalid bloc {operation} response"}

    def _bloc_post(
        self,
        path: str,
        *,
        operation: str,
        body: Dict[str, Any],
        kwargs: Optional[Dict[str, Any]] = None,
        include_default_target: bool = True,
    ) -> Dict[str, Any]:
        call_kwargs = dict(kwargs or {})
        provider_api_key = _pop_provider_api_key(call_kwargs)
        timeout_s = float(call_kwargs.get("timeout_s")) if call_kwargs.get("timeout_s") is not None else self._timeout_s
        payload = {k: _jsonable(v) for k, v in dict(body).items() if v is not None}
        payload.update(self._bloc_proxy_fields(call_kwargs, include_default_target=include_default_target))
        url = _join_core_control_url(self._server_base_url, path)
        try:
            raw = self._sender.post(
                url,
                headers=self._headers_with_provider_api_key(provider_api_key),
                json=payload,
                timeout=timeout_s,
            )
            resp, _resp_headers = _unwrap_http_response(raw)
        except Exception as exc:
            return {"ok": False, "operation": operation, "error": str(exc), "diagnostics": {"source": "abstractcore.remote"}}
        if isinstance(resp, dict):
            return resp
        return {"ok": False, "operation": operation, "error": f"invalid bloc {operation} response"}

    def upsert_text_bloc(
        self,
        *,
        path: str,
        content: str,
        sha256: Optional[str] = None,
        content_sha256: Optional[str] = None,
        media_type: str = "text",
        size_bytes: Optional[int] = None,
        mtime_ns: Optional[int] = None,
        format: Optional[str] = None,
        estimated_tokens: Optional[int] = None,
        relpath_base: Optional[str] = None,
        summary: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "path": path,
            "content": content,
            "sha256": sha256,
            "content_sha256": content_sha256,
            "media_type": media_type,
            "size_bytes": size_bytes,
            "mtime_ns": mtime_ns,
            "format": format,
            "estimated_tokens": estimated_tokens,
            "relpath_base": relpath_base,
            "summary": summary,
            "keywords": keywords,
        }
        return self._bloc_post("/acore/blocs/upsert_text", operation="upsert_text", body=body, kwargs=kwargs)

    def get_bloc_record(
        self,
        *,
        sha256: Optional[str] = None,
        bloc_id: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._bloc_get(
            "/acore/blocs/record",
            operation="record",
            query={"sha256": sha256, "bloc_id": bloc_id},
            kwargs=kwargs,
        )

    def list_blocs(
        self,
        *,
        sha256: Optional[str] = None,
        bloc_id: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._bloc_get(
            "/acore/blocs",
            operation="list",
            query={"sha256": sha256, "bloc_id": bloc_id},
            kwargs=kwargs,
            include_default_target=False,
        )

    def get_bloc_kv_manifest(
        self,
        *,
        sha256: Optional[str] = None,
        bloc_id: Optional[int] = None,
        artifact_path: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._bloc_get(
            "/acore/blocs/kv/manifest",
            operation="kv_manifest",
            query={"sha256": sha256, "bloc_id": bloc_id, "artifact_path": artifact_path},
            kwargs=kwargs,
        )

    def ensure_bloc_kv_artifact(
        self,
        *,
        sha256: Optional[str] = None,
        bloc_id: Optional[int] = None,
        artifact_path: Optional[str] = None,
        force_rebuild: bool = False,
        debug: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "sha256": sha256,
            "bloc_id": bloc_id,
            "artifact_path": artifact_path,
            "force_rebuild": bool(force_rebuild),
            "debug": bool(debug),
        }
        return self._bloc_post("/acore/blocs/kv/ensure", operation="kv_ensure", body=body, kwargs=kwargs)

    def load_bloc_kv_artifact(
        self,
        *,
        sha256: Optional[str] = None,
        bloc_id: Optional[int] = None,
        artifact_path: Optional[str] = None,
        stable_cache_key: Optional[str] = None,
        key: Optional[str] = None,
        make_default: bool = False,
        force_rebuild: bool = False,
        debug: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "sha256": sha256,
            "bloc_id": bloc_id,
            "artifact_path": artifact_path,
            "stable_cache_key": stable_cache_key,
            "key": key,
            "make_default": bool(make_default),
            "force_rebuild": bool(force_rebuild),
            "debug": bool(debug),
        }
        return self._bloc_post("/acore/blocs/kv/load", operation="kv_load", body=body, kwargs=kwargs)

    def list_bloc_kv_artifacts(
        self,
        *,
        sha256: Optional[str] = None,
        bloc_id: Optional[int] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._bloc_get(
            "/acore/blocs/kv/list",
            operation="kv_list",
            query={"sha256": sha256, "bloc_id": bloc_id, "provider": provider, "model": model},
            kwargs=kwargs,
            include_default_target=False,
        )

    def delete_bloc_kv_artifact(
        self,
        *,
        sha256: Optional[str] = None,
        bloc_id: Optional[int] = None,
        artifact_path: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        clear_loaded: bool = False,
        force: bool = False,
        dry_run: bool = False,
        debug: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "sha256": sha256,
            "bloc_id": bloc_id,
            "artifact_path": artifact_path,
            "provider": provider,
            "model": model,
            "clear_loaded": bool(clear_loaded),
            "force": bool(force),
            "dry_run": bool(dry_run),
            "debug": bool(debug),
        }
        return self._bloc_post(
            "/acore/blocs/kv/delete",
            operation="kv_delete",
            body=body,
            kwargs=kwargs,
            include_default_target=False,
        )

    def prune_bloc_kv_artifacts(
        self,
        *,
        sha256: Optional[str] = None,
        bloc_id: Optional[int] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        clear_loaded: bool = False,
        force: bool = False,
        dry_run: bool = False,
        debug: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "sha256": sha256,
            "bloc_id": bloc_id,
            "provider": provider,
            "model": model,
            "clear_loaded": bool(clear_loaded),
            "force": bool(force),
            "dry_run": bool(dry_run),
            "debug": bool(debug),
        }
        return self._bloc_post(
            "/acore/blocs/kv/prune",
            operation="kv_prune",
            body=body,
            kwargs=kwargs,
            include_default_target=False,
        )

    def delete_bloc(
        self,
        *,
        sha256: Optional[str] = None,
        bloc_id: Optional[int] = None,
        delete_kv: bool = True,
        clear_loaded: bool = False,
        force: bool = False,
        dry_run: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "sha256": sha256,
            "bloc_id": bloc_id,
            "delete_kv": bool(delete_kv),
            "clear_loaded": bool(clear_loaded),
            "force": bool(force),
            "dry_run": bool(dry_run),
        }
        return self._bloc_post(
            "/acore/blocs/delete",
            operation="delete",
            body=body,
            kwargs=kwargs,
            include_default_target=False,
        )

    def _model_residency_error_payload(self, *, operation: str, error: Any) -> Dict[str, Any]:
        status_code = None
        response = getattr(error, "response", None)
        try:
            status_code = int(getattr(response, "status_code", None))
        except Exception:
            status_code = None
        payload: Dict[str, Any] = {
            "ok": False,
            "success": False,
            # This client DOES implement the op — the transport/server failed.
            # `supported: false` is reserved for the facade's optional-method
            # degradation; stamping it here made a genuine 404 look identical
            # to "not implemented" (review fix).
            "supported": True,
            "operation": operation,
            "error": str(error),
            "warnings": [str(error)],
            "diagnostics": {"source": "abstractcore.remote"},
            "affected_models": [],
        }
        if status_code is not None:
            payload["status_code"] = status_code
        upstream = _http_error_response_body(error)
        if upstream is not None:
            payload["upstream_error"] = _jsonable(upstream)
        return payload

    def _model_residency_get(self, path: str, *, operation: str, kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        call_kwargs = dict(kwargs or {})
        provider_api_key = _pop_provider_api_key(call_kwargs)
        query: Dict[str, Any] = {}
        for key in ("task", "provider", "model", "base_url"):
            raw = call_kwargs.get(key)
            if isinstance(raw, str) and raw.strip():
                query[key] = raw.strip()
        url = _join_core_control_url(self._server_base_url, path)
        if query:
            url = f"{url}?{urlencode(query)}"
        try:
            raw = self._sender.get(
                url,
                headers=self._headers_with_provider_api_key(provider_api_key),
                timeout=self._timeout_s,
            )
            resp, _resp_headers = _unwrap_http_response(raw)
        except Exception as e:
            return self._model_residency_error_payload(operation=operation, error=e)
        return resp if isinstance(resp, dict) else {"ok": False, "operation": operation, "data": _jsonable(resp)}

    def _model_residency_post(
        self,
        path: str,
        *,
        operation: str,
        body: Dict[str, Any],
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        call_kwargs = dict(kwargs or {})
        provider_api_key = _pop_provider_api_key(call_kwargs)
        payload = {k: _jsonable(v) for k, v in dict(body).items() if v is not None}
        for key in ("base_url", "timeout_s"):
            raw = call_kwargs.get(key)
            if raw is not None and raw != "":
                payload[key] = _jsonable(raw)
        url = _join_core_control_url(self._server_base_url, path)
        try:
            raw = self._sender.post(
                url,
                headers=self._headers_with_provider_api_key(provider_api_key),
                json=payload,
                timeout=self._timeout_s,
            )
            resp, _resp_headers = _unwrap_http_response(raw)
        except Exception as e:
            return self._model_residency_error_payload(operation=operation, error=e)
        return resp if isinstance(resp, dict) else {"ok": False, "operation": operation, "data": _jsonable(resp)}

    def list_model_residency(
        self,
        *,
        task: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        call_kwargs = dict(kwargs)
        if task is not None:
            call_kwargs["task"] = _normalize_residency_task(task)
        if provider is not None:
            call_kwargs["provider"] = provider
        if model is not None:
            call_kwargs["model"] = model
        result = self._model_residency_get(
            "/acore/models/loaded",
            operation="list_loaded",
            kwargs=call_kwargs,
        )
        if isinstance(result, dict):
            result.setdefault("operation", "list_loaded")
            if "models" not in result:
                for key in ("loaded", "runtimes", "data"):
                    if isinstance(result.get(key), list):
                        result["models"] = result.get(key)
                        break
            if isinstance(result.get("models"), list):
                for record in result["models"]:
                    if isinstance(record, dict):
                        _normalize_residency_size_extras(record)
            result.setdefault("success", result.get("ok") is not False)
            result.setdefault("affected_models", result.get("models") if isinstance(result.get("models"), list) else [])
        return result

    def load_model_residency(
        self,
        *,
        task: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        pin: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "task": _normalize_residency_task(task),
            "provider": provider,
            "model": model,
            "options": options if isinstance(options, dict) else None,
            "pin": bool(pin),
        }
        result = self._model_residency_post(
            "/acore/models/load",
            operation="load",
            body=body,
            kwargs=kwargs,
        )
        if isinstance(result, dict):
            result.setdefault("operation", "load")
            result.setdefault("success", result.get("ok") is not False)
            if "affected_models" not in result:
                result["affected_models"] = [result["runtime"]] if isinstance(result.get("runtime"), dict) else []
        return result

    def unload_model_residency(
        self,
        *,
        task: Optional[str] = None,
        runtime_id: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        force: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "task": _normalize_residency_task(task) if task is not None else None,
            "runtime_id": runtime_id,
            "provider": provider,
            "model": model,
            "options": options if isinstance(options, dict) else None,
        }
        if force:
            # Only ride the body when explicitly forcing — older cores'
            # UnloadModelRequest predates the field.
            body["force"] = True
        result = self._model_residency_post(
            "/acore/models/unload",
            operation="unload",
            body=body,
            kwargs=kwargs,
        )
        if isinstance(result, dict) and result.get("status_code") == 409:
            # The core server refuses to unload a LOCKED runtime with HTTP 409
            # `{"ok": false, "error": "model_locked", "detail", "runtime_id"}`.
            # The request sender raised on the status (raise_for_status
            # pattern); relay the server's envelope as a payload — the 409
            # must never escape as an exception to the effect handler.
            upstream = result.get("upstream_error")
            if isinstance(upstream, dict):
                converted: Dict[str, Any] = dict(upstream)
            else:
                converted = {"ok": False, "error": "model_locked"}
                if isinstance(upstream, str) and upstream.strip():
                    converted["detail"] = upstream.strip()
            converted["ok"] = False
            converted.setdefault("error", "model_locked")
            converted["status_code"] = 409
            converted.setdefault("unloaded", False)
            result = converted
        if isinstance(result, dict):
            result.setdefault("operation", "unload")
            result.setdefault("success", result.get("ok") is not False)
            if "affected_models" not in result:
                result["affected_models"] = [result["runtime"]] if isinstance(result.get("runtime"), dict) else []
        return result

    def lock_model_residency(
        self,
        payload: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._model_residency_lock_post(payload, kwargs, operation="lock", path="/acore/models/lock")

    def unlock_model_residency(
        self,
        payload: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._model_residency_lock_post(payload, kwargs, operation="unlock", path="/acore/models/unlock")

    def _model_residency_lock_post(
        self,
        payload: Optional[Mapping[str, Any]],
        kwargs: Dict[str, Any],
        *,
        operation: str,
        path: str,
    ) -> Dict[str, Any]:
        merged = _merge_optional_payload(payload, kwargs)
        task_raw = merged.get("task")
        if task_raw is not None and str(task_raw).strip():
            task_s = _normalize_residency_task(task_raw)
            if task_s != "text_generation":
                # Same guard as the local clients: core's lock route addresses
                # TEXT runtimes only — silently dropping a media task would
                # relay the request onto a text runtime the caller never named.
                return _model_residency_unsupported_payload(
                    operation=operation,
                    task=task_s,
                    provider=str(merged.get("provider") or "").strip().lower(),
                    model=str(merged.get("model") or "").strip(),
                    error=f"model_residency {operation} is only supported for text_generation runtimes.",
                )
        # `LockModelRequest` selector: {runtime_id | provider+model}, base_url
        # optional (the base_url/timeout_s proxy fields ride through the
        # shared post helper from the remaining merged fields).
        body = {key: merged.pop(key, None) for key in ("runtime_id", "provider", "model")}
        result = self._model_residency_post(path, operation=operation, body=body, kwargs=merged)
        if isinstance(result, dict) and operation == "lock" and result.get("status_code") == 409:
            # Core's lock route refuses a non-resident model with HTTP 409
            # `{"ok": false, "error": "model_not_resident", "detail",
            # "runtime_id"}` (the lock rule: lock requires provider-verified
            # residency). Relay the server's envelope as a payload — same
            # conversion idiom as the unload 409.
            upstream = result.get("upstream_error")
            if isinstance(upstream, dict):
                converted: Dict[str, Any] = dict(upstream)
            else:
                converted = {"ok": False, "error": "model_not_resident"}
                if isinstance(upstream, str) and upstream.strip():
                    converted["detail"] = upstream.strip()
            converted["ok"] = False
            converted.setdefault("error", "model_not_resident")
            converted["status_code"] = 409
            converted.setdefault("locked", False)
            result = converted
        if isinstance(result, dict):
            result.setdefault("operation", operation)
            result.setdefault("success", result.get("ok") is not False)
        return result

    def get_context_estimate(
        self,
        payload: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        merged = _merge_optional_payload(payload, kwargs)
        provider_api_key = _pop_provider_api_key(merged)
        query: Dict[str, str] = {}
        for key in ("provider", "model"):
            raw = merged.get(key)
            if isinstance(raw, str) and raw.strip():
                query[key] = raw.strip()
        context_length = merged.get("context_length")
        if context_length is not None and not isinstance(context_length, bool):
            try:
                query["context_length"] = str(int(context_length))
            except Exception:
                pass
        url = _join_core_control_url(self._server_base_url, "/acore/models/context_estimate")
        if query:
            url = f"{url}?{urlencode(query)}"
        try:
            raw = self._sender.get(
                url,
                headers=self._headers_with_provider_api_key(provider_api_key),
                timeout=self._timeout_s,
            )
            resp, _resp_headers = _unwrap_http_response(raw)
        except Exception as e:
            return self._model_residency_error_payload(operation="context_estimate", error=e)
        return resp if isinstance(resp, dict) else {"ok": False, "operation": "context_estimate", "data": _jsonable(resp)}

    def _effective_model_from_params(self, params: Dict[str, Any]) -> str:
        provider = params.pop("_provider", None)
        model = params.pop("_model", None)
        provider_s = str(provider).strip() if isinstance(provider, str) and provider.strip() else ""
        model_s = str(model).strip() if isinstance(model, str) and model.strip() else ""
        if provider_s and model_s:
            return f"{provider_s}/{model_s}"
        if model_s:
            return model_s
        return self._model

    def _trace_run_id_and_tags(
        self,
        params: Dict[str, Any],
        *,
        task: str,
        modality: str,
        model: Optional[str] = None,
    ) -> tuple[Optional[str], Dict[str, Any]]:
        trace_metadata = params.get("trace_metadata") if isinstance(params.get("trace_metadata"), dict) else {}
        run_id = trace_metadata.get("run_id") if isinstance(trace_metadata, dict) else None
        run_id = str(run_id).strip() if isinstance(run_id, str) and run_id.strip() else None
        tags: Dict[str, Any] = {
            "kind": "generated_media",
            "source": "remote_llm_call",
            "modality": modality,
            "task": task,
        }
        if isinstance(model, str) and model.strip():
            tags["model"] = model.strip()
        if isinstance(trace_metadata, dict):
            for key in (
                "workflow_id",
                "node_id",
                "step_id",
                "effect_idempotency_key",
                "actor_id",
                "session_id",
                "parent_run_id",
                "request_id",
            ):
                raw = trace_metadata.get(key)
                if raw is not None and str(raw).strip():
                    tags[key] = str(raw)
        output_run_id, output_tags = _output_runtime_metadata(params.get("output"))
        if run_id is None and output_run_id:
            run_id = output_run_id
        if output_tags:
            tags.update(output_tags)
        # The pre-strip stash (see the strip site): when the output spec was
        # already sanitized for core, the original runtime metadata rides here.
        stash = params.get("_runtime_output_metadata")
        if isinstance(stash, dict):
            stash_run_id = stash.get("run_id")
            if run_id is None and isinstance(stash_run_id, str) and stash_run_id.strip():
                run_id = stash_run_id.strip()
            if isinstance(stash.get("tags"), dict):
                tags.update(stash["tags"])
        return run_id, tags

    def _post_bytes(self, url: str, *, headers: Dict[str, str], json_body: Dict[str, Any]) -> tuple[bytes, Dict[str, str]]:
        sender = self._sender
        post_bytes = getattr(sender, "post_bytes", None)
        if callable(post_bytes):
            raw = post_bytes(url, headers=headers, json=json_body, timeout=self._timeout_s)
        else:
            raw = sender.post(url, headers=headers, json=json_body, timeout=self._timeout_s)
        return _unwrap_binary_response(raw)

    def _post_jsonl_stream(self, url: str, *, headers: Dict[str, str], json_body: Dict[str, Any]) -> Any:
        sender = self._sender
        post_jsonl_stream = getattr(sender, "post_jsonl_stream", None)
        if not callable(post_jsonl_stream):
            raise ValueError("Configured request sender does not support JSONL streaming.")
        return post_jsonl_stream(url, headers=headers, json=json_body, timeout=self._timeout_s)

    def _post_multipart(
        self,
        url: str,
        *,
        headers: Dict[str, str],
        data: Dict[str, Any],
        file_path: str,
        file_field: str = "file",
    ) -> tuple[Dict[str, Any], Dict[str, str]]:
        filename = os.path.basename(file_path) or "media.bin"
        mime = _mime_type_for_path(file_path)
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        files = {file_field: (filename, file_bytes, mime)}
        sender = self._sender
        post_multipart = getattr(sender, "post_multipart", None)
        if callable(post_multipart):
            raw = post_multipart(url, headers=headers, data=data, files=files, timeout=self._timeout_s)
        else:
            raise ValueError("Remote multipart media output requires a request sender with post_multipart().")
        return _unwrap_http_response(raw)

    def _post_multipart_files(
        self,
        url: str,
        *,
        headers: Dict[str, str],
        data: Dict[str, Any],
        files: Dict[str, tuple[str, bytes, str]],
        timeout: Optional[float] = None,
    ) -> tuple[Dict[str, Any], Dict[str, str]]:
        sender = self._sender
        post_multipart = getattr(sender, "post_multipart", None)
        if not callable(post_multipart):
            raise ValueError("Remote multipart media output requires a request sender with post_multipart().")
        raw = post_multipart(url, headers=headers, data=data, files=files, timeout=timeout)
        return _unwrap_http_response(raw)

    def _remote_image_generation(
        self,
        *,
        spec: Dict[str, Any],
        prompt: str,
        headers: Dict[str, str],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        endpoint_model = str(spec.get("model") or "").strip()
        endpoint_provider = str(spec.get("provider") or "").strip().lower().replace("_", "-")
        if endpoint_model and "/" in endpoint_model:
            head = endpoint_model.split("/", 1)[0].strip().lower()
        else:
            head = ""
        if (
            endpoint_model
            and not endpoint_provider
            and head == ""
            and endpoint_model.lower().startswith(("gpt-image", "dall-e"))
        ):
            endpoint_model = f"openai-compatible/{endpoint_model}"
        elif endpoint_model.lower().startswith(("openai/openai-compatible/", "openai/openai_compatible/")):
            endpoint_model = endpoint_model.split("/", 1)[1]
        elif endpoint_model and endpoint_provider in {"openai", "openai-compatible"} and head not in {"openai-compatible"}:
            endpoint_model = f"openai-compatible/{endpoint_model}"
        body: Dict[str, Any] = {
            "prompt": prompt,
            "response_format": "b64_json",
        }
        if endpoint_provider:
            body["provider"] = endpoint_provider
        if endpoint_model:
            body["model"] = endpoint_model
        for key in (
            "n",
            "width",
            "height",
            "size",
            "negative_prompt",
            "seed",
            "steps",
            "guidance_scale",
            "guidance_2",
            "quality",
            "style",
            "user",
            "background",
            "output_format",
            "output_compression",
            "moderation",
            "extra",
        ):
            if key in spec and spec.get(key) is not None:
                body[key] = spec.get(key)
        count = _spec_generation_count(spec)
        if count is not None:
            body["n"] = count
        seeds = _spec_generation_seeds(spec)
        if seeds is not None:
            body["seeds"] = list(seeds)
        lora_adapters = _spec_lora_adapters(spec)
        if lora_adapters:
            body["lora_adapters"] = lora_adapters
        endpoint_model_lower = endpoint_model.lower()
        official_openai_gpt_image = (
            endpoint_model_lower.startswith("openai-compatible/gpt-image")
            or endpoint_model_lower.startswith("openai/gpt-image")
            or endpoint_model_lower.startswith("gpt-image")
        )
        if official_openai_gpt_image:
            allowed_sizes = {"1024x1024", "1024x1536", "1536x1024", "auto"}
            requested_size = str(body.get("size") or "").strip().lower()
            if requested_size not in allowed_sizes:
                width = body.get("width")
                height = body.get("height")
                derived = f"{int(width)}x{int(height)}" if isinstance(width, int) and isinstance(height, int) else ""
                body["size"] = derived if derived in allowed_sizes else "auto"
            body.pop("response_format", None)
            for local_only_key in (
                "width",
                "height",
                "seed",
                "steps",
                "guidance_scale",
                "guidance_2",
                "negative_prompt",
            ):
                body.pop(local_only_key, None)
        base_url = params.get("base_url")
        if isinstance(base_url, str) and base_url.strip():
            body["base_url"] = base_url.strip()

        progress_callback = params.get("on_progress")
        if callable(progress_callback):
            url = _join_core_v1_url(self._server_base_url, "/vision/jobs/images/generations")
            raw = self._sender.post(url, headers=headers, json=body, timeout=None)
            start_resp, _resp_headers = _unwrap_http_response(raw)
            job_id = str(start_resp.get("job_id") or "").strip() if isinstance(start_resp, dict) else ""
            if not job_id:
                raise ValueError("Remote image generation job did not return job_id.")
            resp = self._poll_remote_vision_job(job_id=job_id, headers=headers, params=params)
        else:
            url = _join_core_v1_url(self._server_base_url, "/images/generations")
            raw = self._sender.post(url, headers=headers, json=body, timeout=None)
            resp, _resp_headers = _unwrap_http_response(raw)

        data_items = resp.get("data") if isinstance(resp, dict) else None
        if not isinstance(data_items, list) or not data_items:
            raise ValueError("Remote image generation returned no data items.")

        fmt = str(spec.get("format") or spec.get("output_format") or "png").strip().lower() or "png"
        content_type = f"image/{fmt}"
        outputs: List[Dict[str, Any]] = []
        for item in data_items:
            if not isinstance(item, dict):
                continue
            raw_b64 = item.get("b64_json") or item.get("image") or item.get("data")
            if not isinstance(raw_b64, str) or not raw_b64.strip():
                continue
            image_bytes = base64.b64decode("".join(raw_b64.strip().split()), validate=True)
            outputs.append(
                {
                    "modality": "image",
                    "task": "image_generation",
                    "data": image_bytes,
                    "content_type": content_type,
                    "format": fmt,
                    "provider": "abstractcore-server",
                    "model": str(body.get("model") or "") or None,
                    "metadata": {"_provider_request": {"url": url, "payload": body}},
                }
            )
        if not outputs:
            raise ValueError("Remote image generation response did not contain b64_json data.")
        run_id, tags = self._trace_run_id_and_tags(
            params,
            task="image_generation",
            modality="image",
            model=str(body.get("model") or "") or None,
        )
        return _normalize_multimodal_response(
            {"outputs": {"image": outputs}, "metadata": {"model": body.get("model"), "provider": "abstractcore-server"}},
            artifact_store=self._artifact_store,
            run_id=run_id,
            default_tags=tags,
            generation_context=_generated_artifact_context(prompt=prompt, media=None, output_request=spec),
        )

    def _remote_image_edit(
        self,
        *,
        spec: Dict[str, Any],
        prompt: str,
        media: Optional[List[Any]],
        headers: Dict[str, str],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        image_item: Any = None
        mask_item: Any = None
        for item in list(media or []):
            role = _media_role_from_item(item)
            if role == "mask":
                if mask_item is not None:
                    raise ValueError("Remote image edit accepts at most one mask media item.")
                mask_item = item
                continue
            if _is_image_media_item(item):
                if image_item is not None:
                    raise ValueError("Remote image edit requires exactly one source image media item.")
                image_item = item

        if image_item is None:
            raise ValueError("Remote image edit requires exactly one source image media item.")

        def _file_tuple(field: str, item: Any) -> tuple[str, bytes, str]:
            path = _media_path_from_item(item)
            if not path:
                raise ValueError(f"Remote image edit {field} media must resolve to a local file path.")
            if path.lower().startswith("data:") or path.startswith(("http://", "https://")):
                raise ValueError(f"Remote image edit {field} media must be a local file path or artifact-backed file.")
            filename = os.path.basename(path) or f"{field}.bin"
            mime = _media_mime_from_item(item) or _mime_type_for_path(path)
            with open(path, "rb") as f:
                file_bytes = f.read()
            return filename, file_bytes, mime

        endpoint_model = str(spec.get("model") or "").strip()
        endpoint_provider = str(spec.get("provider") or "").strip().lower().replace("_", "-")
        data: Dict[str, Any] = {
            "prompt": prompt,
            "response_format": "b64_json",
        }
        if endpoint_model:
            data["model"] = endpoint_model
        if endpoint_provider:
            data["provider"] = endpoint_provider
        count = _spec_generation_count(spec)
        seeds = _spec_generation_seeds(spec)
        for key in ("size", "negative_prompt", "seed", "steps", "guidance_scale", "guidance_2"):
            if key in spec and spec.get(key) is not None:
                data[key] = spec.get(key)
        if count is not None:
            data["n"] = str(count)
        if seeds is not None:
            data["seeds"] = ",".join(str(seed) for seed in seeds)
        extra = dict(spec.get("extra")) if isinstance(spec.get("extra"), dict) else {}
        for key in (
            "quality",
            "style",
            "strength",
            "background",
            "output_format",
            "output_compression",
            "moderation",
        ):
            if key in spec and spec.get(key) is not None:
                extra.setdefault(key, spec.get(key))
        lora_adapters = _spec_lora_adapters(spec)
        if lora_adapters:
            data["lora_adapters_json"] = json.dumps(lora_adapters, ensure_ascii=False, separators=(",", ":"))
        if extra:
            data["extra_json"] = json.dumps(extra, ensure_ascii=False, separators=(",", ":"))
        base_url = params.get("base_url")
        if isinstance(base_url, str) and base_url.strip():
            data["base_url"] = base_url.strip()

        files: Dict[str, tuple[str, bytes, str]] = {"image": _file_tuple("image", image_item)}
        if mask_item is not None:
            files["mask"] = _file_tuple("mask", mask_item)

        progress_callback = params.get("on_progress")
        if callable(progress_callback):
            url = _join_core_v1_url(self._server_base_url, "/vision/jobs/images/edits")
            start_resp, _resp_headers = self._post_multipart_files(url, headers=headers, data=data, files=files, timeout=None)
            job_id = str(start_resp.get("job_id") or "").strip() if isinstance(start_resp, dict) else ""
            if not job_id:
                raise ValueError("Remote image edit job did not return job_id.")
            resp = self._poll_remote_vision_job(job_id=job_id, headers=headers, params=params)
        elif endpoint_provider:
            data.pop("provider", None)
            url = _join_core_provider_v1_url(self._server_base_url, endpoint_provider, "/images/edits")
            resp, _resp_headers = self._post_multipart_files(url, headers=headers, data=data, files=files, timeout=None)
        else:
            url = _join_core_v1_url(self._server_base_url, "/images/edits")
            resp, _resp_headers = self._post_multipart_files(url, headers=headers, data=data, files=files, timeout=None)

        data_items = resp.get("data") if isinstance(resp, dict) else None
        if not isinstance(data_items, list) or not data_items:
            raise ValueError("Remote image edit returned no data items.")

        fmt = str(spec.get("format") or spec.get("output_format") or "png").strip().lower() or "png"
        content_type = f"image/{fmt}"
        outputs: List[Dict[str, Any]] = []
        for item in data_items:
            if not isinstance(item, dict):
                continue
            raw_b64 = item.get("b64_json") or item.get("image") or item.get("data")
            if not isinstance(raw_b64, str) or not raw_b64.strip():
                continue
            image_bytes = base64.b64decode("".join(raw_b64.strip().split()), validate=True)
            outputs.append(
                {
                    "modality": "image",
                    "task": "image_edit",
                    "data": image_bytes,
                    "content_type": content_type,
                    "format": fmt,
                    "provider": endpoint_provider or "abstractcore-server",
                    "model": str(data.get("model") or "") or None,
                    "metadata": {"_provider_request": {"url": url, "payload": data}},
                }
            )
        if not outputs:
            raise ValueError("Remote image edit response did not contain b64_json data.")
        run_id, tags = self._trace_run_id_and_tags(
            params,
            task="image_edit",
            modality="image",
            model=str(data.get("model") or "") or None,
        )
        return _normalize_multimodal_response(
            {"outputs": {"image": outputs}, "metadata": {"model": data.get("model"), "provider": "abstractcore-server"}},
            artifact_store=self._artifact_store,
            run_id=run_id,
            default_tags=tags,
            generation_context=_generated_artifact_context(prompt=prompt, media=media, output_request=spec),
        )

    def _remote_image_upscale(
        self,
        *,
        spec: Dict[str, Any],
        media: Optional[List[Any]],
        headers: Dict[str, str],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        image_item: Any = None
        for item in list(media or []):
            if _is_image_media_item(item):
                if image_item is not None:
                    raise ValueError("Remote image upscale requires exactly one source image media item.")
                image_item = item
        if image_item is None:
            raise ValueError("Remote image upscale requires exactly one source image media item.")

        path = _media_path_from_item(image_item)
        if not path:
            raise ValueError("Remote image upscale media must resolve to a local file path.")
        if path.lower().startswith("data:") or path.startswith(("http://", "https://")):
            raise ValueError("Remote image upscale media must be a local file path or artifact-backed file.")
        filename = os.path.basename(path) or "image.bin"
        mime = _media_mime_from_item(image_item) or _mime_type_for_path(path)
        with open(path, "rb") as f:
            file_bytes = f.read()

        endpoint_model = str(spec.get("model") or "").strip()
        endpoint_provider = str(spec.get("provider") or "").strip().lower().replace("_", "-")
        data: Dict[str, Any] = {"response_format": "b64_json"}
        if endpoint_model:
            data["model"] = endpoint_model
        if endpoint_provider:
            data["provider"] = endpoint_provider
        for key in (
            "scale",
            "resolution",
            "softness",
            "seed",
            "quantize",
            "vae_tiling",
            "output_format",
            "format",
        ):
            if key in spec and spec.get(key) is not None:
                data[key] = spec.get(key)
        extra = dict(spec.get("extra")) if isinstance(spec.get("extra"), dict) else {}
        if extra:
            data["extra_json"] = json.dumps(extra, ensure_ascii=False, separators=(",", ":"))
        base_url = params.get("base_url")
        if isinstance(base_url, str) and base_url.strip():
            data["base_url"] = base_url.strip()

        files = {"image": (filename, file_bytes, mime)}
        progress_callback = params.get("on_progress")
        if callable(progress_callback):
            url = _join_core_v1_url(self._server_base_url, "/vision/jobs/images/upscale")
            start_resp, _resp_headers = self._post_multipart_files(url, headers=headers, data=data, files=files, timeout=None)
            job_id = str(start_resp.get("job_id") or "").strip() if isinstance(start_resp, dict) else ""
            if not job_id:
                raise ValueError("Remote image upscale job did not return job_id.")
            resp = self._poll_remote_vision_job(job_id=job_id, headers=headers, params=params)
        else:
            if endpoint_provider:
                data.pop("provider", None)
                url = _join_core_provider_v1_url(self._server_base_url, endpoint_provider, "/images/upscale")
            else:
                url = _join_core_v1_url(self._server_base_url, "/images/upscale")
            resp, _resp_headers = self._post_multipart_files(url, headers=headers, data=data, files=files, timeout=None)

        data_items = resp.get("data") if isinstance(resp, dict) else None
        if not isinstance(data_items, list) or not data_items:
            raise ValueError("Remote image upscale returned no data items.")

        fmt = str(spec.get("format") or spec.get("output_format") or "png").strip().lower() or "png"
        content_type = f"image/{fmt}"
        outputs: List[Dict[str, Any]] = []
        for item in data_items:
            if not isinstance(item, dict):
                continue
            raw_b64 = item.get("b64_json") or item.get("image") or item.get("data")
            if not isinstance(raw_b64, str) or not raw_b64.strip():
                continue
            image_bytes = base64.b64decode("".join(raw_b64.strip().split()), validate=True)
            outputs.append(
                {
                    "modality": "image",
                    "task": "image_upscale",
                    "data": image_bytes,
                    "content_type": content_type,
                    "format": fmt,
                    "provider": endpoint_provider or "abstractcore-server",
                    "model": str(data.get("model") or "") or None,
                    "metadata": {"_provider_request": {"url": url, "payload": data}},
                }
            )
        if not outputs:
            raise ValueError("Remote image upscale response did not contain b64_json data.")
        run_id, tags = self._trace_run_id_and_tags(
            params,
            task="image_upscale",
            modality="image",
            model=str(data.get("model") or "") or None,
        )
        return _normalize_multimodal_response(
            {"outputs": {"image": outputs}, "metadata": {"model": data.get("model"), "provider": "abstractcore-server"}},
            artifact_store=self._artifact_store,
            run_id=run_id,
            default_tags=tags,
            generation_context=_generated_artifact_context(prompt="", media=media, output_request=spec),
        )

    def _remote_video_generation(
        self,
        *,
        spec: Dict[str, Any],
        prompt: str,
        headers: Dict[str, str],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        endpoint_model = str(spec.get("model") or "").strip()
        endpoint_provider = str(spec.get("provider") or "").strip().lower().replace("_", "-")
        body: Dict[str, Any] = {"prompt": prompt}
        if endpoint_provider:
            body["provider"] = endpoint_provider
        if endpoint_model:
            body["model"] = endpoint_model
        for key in ("n", "width", "height", "fps", "seed", "steps", "guidance_scale", "guidance_2", "flow_shift", "negative_prompt", "extra"):
            if key in spec and spec.get(key) is not None:
                body[key] = spec.get(key)
        count = _spec_generation_count(spec)
        if count is not None:
            body["n"] = count
        seeds = _spec_generation_seeds(spec)
        if seeds is not None:
            body["seeds"] = list(seeds)
        lora_adapters = _spec_lora_adapters(spec)
        if lora_adapters:
            body["lora_adapters"] = lora_adapters
        num_frames = spec.get("num_frames")
        if num_frames is None:
            num_frames = spec.get("frames")
        if num_frames is not None:
            body["num_frames"] = num_frames
        base_url = params.get("base_url")
        if isinstance(base_url, str) and base_url.strip():
            body["base_url"] = base_url.strip()

        progress_callback = params.get("on_progress")
        if callable(progress_callback):
            url = _join_core_v1_url(self._server_base_url, "/vision/jobs/videos/generations")
            raw = self._sender.post(url, headers=headers, json=body, timeout=None)
            start_resp, _resp_headers = _unwrap_http_response(raw)
            job_id = str(start_resp.get("job_id") or "").strip() if isinstance(start_resp, dict) else ""
            if not job_id:
                raise ValueError("Remote video generation job did not return job_id.")
            resp = self._poll_remote_vision_job(job_id=job_id, headers=headers, params=params)
        else:
            url = _join_core_v1_url(self._server_base_url, "/videos/generations")
            raw = self._sender.post(url, headers=headers, json=body, timeout=None)
            resp, _resp_headers = _unwrap_http_response(raw)

        data_items = resp.get("data") if isinstance(resp, dict) else None
        if not isinstance(data_items, list) or not data_items:
            raise ValueError("Remote video generation returned no data items.")

        fmt = str(spec.get("format") or spec.get("output_format") or "mp4").strip().lower() or "mp4"
        content_type = f"video/{fmt}"
        outputs: List[Dict[str, Any]] = []
        for item in data_items:
            if not isinstance(item, dict):
                continue
            raw_b64 = item.get("b64_json") or item.get("video") or item.get("data")
            if not isinstance(raw_b64, str) or not raw_b64.strip():
                continue
            video_bytes = base64.b64decode("".join(raw_b64.strip().split()), validate=True)
            outputs.append(
                {
                    "modality": "video",
                    "task": "text_to_video",
                    "data": video_bytes,
                    "content_type": content_type,
                    "format": fmt,
                    "provider": endpoint_provider or "abstractcore-server",
                    "model": str(body.get("model") or "") or None,
                    "metadata": {"_provider_request": {"url": url, "payload": body}},
                }
            )
        if not outputs:
            raise ValueError("Remote video generation response did not contain b64_json data.")
        run_id, tags = self._trace_run_id_and_tags(
            params,
            task="text_to_video",
            modality="video",
            model=str(body.get("model") or "") or None,
        )
        return _normalize_multimodal_response(
            {"outputs": {"video": outputs}, "metadata": {"model": body.get("model"), "provider": "abstractcore-server"}},
            artifact_store=self._artifact_store,
            run_id=run_id,
            default_tags=tags,
            generation_context=_generated_artifact_context(prompt=prompt, media=None, output_request=spec),
        )

    def _poll_remote_vision_job(
        self,
        *,
        job_id: str,
        headers: Dict[str, str],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        callback = params.get("on_progress")
        poll_interval_s = params.get("progress_poll_interval_s")
        if poll_interval_s is None:
            poll_interval_s = params.get("poll_interval_s")
        try:
            interval = float(poll_interval_s) if poll_interval_s is not None else 1.0
        except Exception:
            interval = 1.0
        interval = max(0.05, min(interval, 10.0))
        last_progress_key = ""
        while True:
            url = f"{_join_core_v1_url(self._server_base_url, f'/vision/jobs/{quote(job_id)}')}?consume=true"
            raw = self._sender.get(url, headers=headers, timeout=None)
            job, _resp_headers = _unwrap_http_response(raw)
            if not isinstance(job, dict):
                raise ValueError("Remote vision job poll returned a non-object response.")
            state = str(job.get("state") or "").strip().lower()
            progress = job.get("progress")
            if callable(callback) and isinstance(progress, dict):
                progress_payload = _progress_for_remote_job_state(progress, job_id=job_id, state=state)
                if _progress_is_reported(progress_payload) or state in {"succeeded", "failed"}:
                    try:
                        progress_key = json.dumps(progress_payload, sort_keys=True, default=str)
                    except Exception:
                        progress_key = str(progress_payload)
                    if progress_key != last_progress_key:
                        callback(progress_payload)
                        last_progress_key = progress_key
            if state == "succeeded":
                result = job.get("result")
                if not isinstance(result, dict):
                    raise ValueError("Remote vision job succeeded without a result object.")
                return result
            if state == "failed":
                raise ValueError(str(job.get("error") or "Remote vision job failed."))
            time.sleep(interval)

    def _remote_image_to_video(
        self,
        *,
        spec: Dict[str, Any],
        prompt: str,
        media: Optional[List[Any]],
        headers: Dict[str, str],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        image_item: Any = None
        for item in list(media or []):
            if _is_image_media_item(item):
                if image_item is not None:
                    raise ValueError("Remote image-to-video requires exactly one source image media item.")
                image_item = item
        if image_item is None:
            raise ValueError("Remote image-to-video requires exactly one source image media item.")

        path = _media_path_from_item(image_item)
        if not path:
            raise ValueError("Remote image-to-video media must resolve to a local file path.")
        if path.lower().startswith("data:") or path.startswith(("http://", "https://")):
            raise ValueError("Remote image-to-video media must be a local file path or artifact-backed file.")
        filename = os.path.basename(path) or "image.bin"
        mime = _media_mime_from_item(image_item) or _mime_type_for_path(path)
        with open(path, "rb") as f:
            file_bytes = f.read()

        endpoint_model = str(spec.get("model") or "").strip()
        endpoint_provider = str(spec.get("provider") or "").strip().lower().replace("_", "-")
        data: Dict[str, Any] = {"prompt": prompt}
        if endpoint_model:
            data["model"] = endpoint_model
        if endpoint_provider:
            data["provider"] = endpoint_provider
        count = _spec_generation_count(spec)
        seeds = _spec_generation_seeds(spec)
        for key in ("width", "height", "fps", "seed", "steps", "guidance_scale", "guidance_2", "flow_shift", "negative_prompt"):
            if key in spec and spec.get(key) is not None:
                data[key] = spec.get(key)
        if count is not None:
            data["n"] = str(count)
        if seeds is not None:
            data["seeds"] = ",".join(str(seed) for seed in seeds)
        num_frames = spec.get("num_frames")
        if num_frames is None:
            num_frames = spec.get("frames")
        if num_frames is not None:
            data["num_frames"] = num_frames
        extra = dict(spec.get("extra")) if isinstance(spec.get("extra"), dict) else {}
        lora_adapters = _spec_lora_adapters(spec)
        if lora_adapters:
            data["lora_adapters_json"] = json.dumps(lora_adapters, ensure_ascii=False, separators=(",", ":"))
        if extra:
            data["extra_json"] = json.dumps(extra, ensure_ascii=False, separators=(",", ":"))
        base_url = params.get("base_url")
        if isinstance(base_url, str) and base_url.strip():
            data["base_url"] = base_url.strip()

        progress_callback = params.get("on_progress")
        if callable(progress_callback):
            url = _join_core_v1_url(self._server_base_url, "/vision/jobs/videos/edits")
            start_resp, _resp_headers = self._post_multipart_files(
                url,
                headers=headers,
                data=data,
                files={"image": (filename, file_bytes, mime)},
                timeout=None,
            )
            job_id = str(start_resp.get("job_id") or "").strip() if isinstance(start_resp, dict) else ""
            if not job_id:
                raise ValueError("Remote image-to-video job did not return job_id.")
            resp = self._poll_remote_vision_job(job_id=job_id, headers=headers, params=params)
        else:
            if endpoint_provider:
                data.pop("provider", None)
                url = _join_core_provider_v1_url(self._server_base_url, endpoint_provider, "/videos/edits")
            else:
                url = _join_core_v1_url(self._server_base_url, "/videos/edits")
            resp, _resp_headers = self._post_multipart_files(
                url,
                headers=headers,
                data=data,
                files={"image": (filename, file_bytes, mime)},
                timeout=None,
            )

        data_items = resp.get("data") if isinstance(resp, dict) else None
        if not isinstance(data_items, list) or not data_items:
            raise ValueError("Remote image-to-video returned no data items.")

        fmt = str(spec.get("format") or spec.get("output_format") or "mp4").strip().lower() or "mp4"
        content_type = f"video/{fmt}"
        outputs: List[Dict[str, Any]] = []
        for item in data_items:
            if not isinstance(item, dict):
                continue
            raw_b64 = item.get("b64_json") or item.get("video") or item.get("data")
            if not isinstance(raw_b64, str) or not raw_b64.strip():
                continue
            video_bytes = base64.b64decode("".join(raw_b64.strip().split()), validate=True)
            outputs.append(
                {
                    "modality": "video",
                    "task": "image_to_video",
                    "data": video_bytes,
                    "content_type": content_type,
                    "format": fmt,
                    "provider": endpoint_provider or "abstractcore-server",
                    "model": str(data.get("model") or "") or None,
                    "metadata": {"_provider_request": {"url": url, "payload": data}},
                }
            )
        if not outputs:
            raise ValueError("Remote image-to-video response did not contain b64_json data.")
        run_id, tags = self._trace_run_id_and_tags(
            params,
            task="image_to_video",
            modality="video",
            model=str(data.get("model") or "") or None,
        )
        return _normalize_multimodal_response(
            {"outputs": {"video": outputs}, "metadata": {"model": data.get("model"), "provider": "abstractcore-server"}},
            artifact_store=self._artifact_store,
            run_id=run_id,
            default_tags=tags,
            generation_context=_generated_artifact_context(prompt=prompt, media=media, output_request=spec),
        )

    def _remote_tts(
        self,
        *,
        spec: Dict[str, Any],
        text: str,
        headers: Dict[str, str],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        fmt = str(spec.get("format") or spec.get("response_format") or "wav").strip().lower() or "wav"
        endpoint_model = str(spec.get("model") or "").strip()
        voice = spec.get("voice") or spec.get("voice_id")
        body: Dict[str, Any] = {
            "input": str(text or ""),
            "response_format": fmt,
        }
        if voice is not None:
            body["voice"] = voice
        if endpoint_model:
            body["model"] = endpoint_model
        endpoint_provider = str(spec.get("provider") or "").strip().lower().replace("_", "-")
        for key in ("speed", "instructions", "provider", "profile", "quality_preset"):
            if key in spec and spec.get(key) is not None:
                body[key] = spec.get(key)
        if "quality" in spec and "quality_preset" not in body and spec.get("quality") is not None:
            body["quality_preset"] = spec.get("quality")
        base_url = params.get("base_url")
        if isinstance(base_url, str) and base_url.strip():
            body["base_url"] = base_url.strip()

        if endpoint_provider:
            body.pop("provider", None)
            url = _join_core_provider_v1_url(self._server_base_url, endpoint_provider, "/audio/speech")
        else:
            url = _join_core_v1_url(self._server_base_url, "/audio/speech")
        audio_bytes, resp_headers = self._post_bytes(url, headers=headers, json_body=body)
        content_type = str(resp_headers.get("content-type") or f"audio/{fmt}").split(";", 1)[0].strip() or f"audio/{fmt}"
        run_id, tags = self._trace_run_id_and_tags(
            params,
            task="tts",
            modality="voice",
            model=str(body.get("model") or "") or None,
        )
        return _normalize_multimodal_response(
            {
                "outputs": {
                    "voice": [
                        {
                            "modality": "voice",
                            "task": "tts",
                            "data": audio_bytes,
                            "content_type": content_type,
                            "format": fmt,
                            "provider": "abstractcore-server",
                            "model": str(body.get("model") or "") or None,
                            "metadata": {"_provider_request": {"url": url, "payload": body}},
                        }
                    ]
                },
                "metadata": {"model": body.get("model"), "provider": "abstractcore-server"},
            },
            artifact_store=self._artifact_store,
            run_id=run_id,
            default_tags=tags,
            generation_context=_generated_artifact_context(prompt=text, media=None, output_request=spec),
        )

    def _remote_tts_stream(
        self,
        *,
        spec: Dict[str, Any],
        text: str,
        headers: Dict[str, str],
        params: Dict[str, Any],
    ):
        fmt = str(spec.get("format") or spec.get("response_format") or "wav").strip().lower() or "wav"
        if fmt == "wave":
            fmt = "wav"
        if fmt != "wav":
            raise ValueError("Remote TTS streaming currently supports wav only.")
        endpoint_model = str(spec.get("model") or "").strip()
        voice = spec.get("voice") or spec.get("voice_id")
        body: Dict[str, Any] = {
            "input": str(text or ""),
            "response_format": "wav",
        }
        if voice is not None:
            body["voice"] = voice
        if endpoint_model:
            body["model"] = endpoint_model
        endpoint_provider = str(spec.get("provider") or "").strip().lower().replace("_", "-")
        for key in ("speed", "instructions", "provider", "profile", "quality_preset"):
            if key in spec and spec.get(key) is not None:
                body[key] = spec.get(key)
        if "quality" in spec and "quality_preset" not in body and spec.get("quality") is not None:
            body["quality_preset"] = spec.get("quality")
        base_url = params.get("base_url")
        if isinstance(base_url, str) and base_url.strip():
            body["base_url"] = base_url.strip()

        if endpoint_provider:
            body.pop("provider", None)
            url = _join_core_provider_v1_url(self._server_base_url, endpoint_provider, "/audio/speech/stream")
        else:
            url = _join_core_v1_url(self._server_base_url, "/audio/speech/stream")

        run_id, tags = self._trace_run_id_and_tags(
            params,
            task="tts",
            modality="voice",
            model=str(body.get("model") or "") or None,
        )
        segments: List[bytes] = []
        terminal_seen = False
        for raw_line in self._post_jsonl_stream(url, headers=headers, json_body=body):
            if isinstance(raw_line, bytes):
                line = raw_line.decode("utf-8", errors="replace")
            else:
                line = str(raw_line)
            if not line.strip():
                continue
            event = json.loads(line)
            if not isinstance(event, dict):
                event = {"type": "event", "value": _jsonable(event)}
            if event.get("type") == "audio":
                audio_bytes = _decode_stream_audio_b64(event)
                if audio_bytes:
                    segments.append(audio_bytes)
            if event.get("type") in {"done", "cancelled"}:
                terminal_seen = True
                if event.get("type") == "done" and event.get("ok") is not False:
                    artifact = _finalize_tts_stream_artifact(
                        segments=segments,
                        artifact_store=self._artifact_store,
                        run_id=run_id,
                        tags=tags,
                        text=text,
                        spec=spec,
                        provider="abstractcore-server",
                        model=str(body.get("model") or "") or None,
                        metadata={
                            "_provider_request": {"url": url, "payload": body},
                            "stream": {
                                "chunks": len(segments),
                                "transport": "jsonl",
                                "chunk_format": "wav-segment",
                            },
                        },
                    )
                    event = dict(event)
                    event["audio_artifact"] = artifact
                yield event
                continue
            yield event
        if not terminal_seen:
            yield {
                "type": "error",
                "ok": False,
                "error": "TTS stream ended without a terminal done/cancelled event.",
            }

    def _remote_music(
        self,
        *,
        spec: Dict[str, Any],
        prompt: str,
        headers: Dict[str, str],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        fmt = str(spec.get("format") or spec.get("response_format") or "wav").strip().lower() or "wav"
        legacy_backend = str(spec.get("backend") or spec.get("music_backend") or "").strip()
        if legacy_backend:
            raise ValueError(
                "Music output routing uses `provider` as the backend selector; "
                "`backend` and `music_backend` are not supported."
            )
        provider_task = str(spec.get("task") or "music_generation").strip().lower().replace("-", "_") or "music_generation"
        if provider_task in {"music", "song", "t2m", "music_generation"}:
            provider_task = "text_to_music"
        elif provider_task in {"sound", "sfx", "sound_generation", "audio_generation"}:
            provider_task = "text_to_audio"
        result_task = "sound_generation" if provider_task == "text_to_audio" else "music_generation"
        result_modality = "sound" if result_task == "sound_generation" else "music"
        body: Dict[str, Any] = {
            "prompt": str(prompt or ""),
            "task": provider_task,
            "format": fmt,
        }
        for key, value in spec.items():
            if key in {
                "modality",
                "type",
                "output",
                "task",
                "prompt",
                "input",
                "text",
                "format",
                "response_format",
                "backend",
                "music_backend",
                "run_id",
                "tags",
                "artifact_id",
            }:
                continue
            if value is not None:
                body[key] = value
        base_url = params.get("base_url")
        if isinstance(base_url, str) and base_url.strip():
            body["base_url"] = base_url.strip()

        url = _join_core_v1_url(self._server_base_url, "/audio/music")
        audio_bytes, resp_headers = self._post_bytes(url, headers=headers, json_body=body)
        content_type = str(resp_headers.get("content-type") or f"audio/{fmt}").split(";", 1)[0].strip() or f"audio/{fmt}"
        media_provider = str(body.get("provider") or "abstractcore-server").strip() or "abstractcore-server"
        run_id, tags = self._trace_run_id_and_tags(
            params,
            task=result_task,
            modality=result_modality,
            model=str(body.get("model") or "") or None,
        )
        return _normalize_multimodal_response(
            {
                "outputs": {
                    result_modality: [
                        {
                            "modality": result_modality,
                            "task": result_task,
                            "data": audio_bytes,
                            "content_type": content_type,
                            "format": fmt,
                            "provider": media_provider,
                            "model": str(body.get("model") or "") or None,
                            "metadata": {"_provider_request": {"url": url, "payload": body}},
                        }
                    ]
                },
                "metadata": {"model": body.get("model"), "provider": "abstractcore-server"},
            },
            artifact_store=self._artifact_store,
            run_id=run_id,
            default_tags=tags,
            generation_context=_generated_artifact_context(prompt=prompt, media=None, output_request=spec),
        )

    def _remote_transcription(
        self,
        *,
        spec: Dict[str, Any],
        media: Optional[List[Any]],
        headers: Dict[str, str],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        media_paths: List[str] = []
        for item in list(media or []):
            path = _media_path_from_item(item)
            if not path:
                continue
            if path.lower().startswith("data:") or path.startswith(("http://", "https://")):
                raise ValueError("Remote transcription requires a local file path or artifact-backed audio media item.")
            media_paths.append(path)
        if len(media_paths) != 1:
            if any(_is_audio_media_item(item) for item in list(media or [])):
                raise ValueError("Remote transcription audio media must resolve to a local file path.")
            raise ValueError("Remote transcription requires exactly one audio media item.")
        endpoint_model = str(spec.get("model") or "").strip()
        endpoint_provider = str(spec.get("provider") or "").strip().lower().replace("_", "-")
        data: Dict[str, Any] = {}
        if endpoint_model:
            data["model"] = endpoint_model
        for key in ("language", "prompt", "response_format", "temperature", "format", "provider"):
            if key in spec and spec.get(key) is not None:
                data[key] = spec.get(key)
        base_url = params.get("base_url")
        if isinstance(base_url, str) and base_url.strip():
            data["base_url"] = base_url.strip()
        if endpoint_provider:
            data.pop("provider", None)
            url = _join_core_provider_v1_url(self._server_base_url, endpoint_provider, "/audio/transcriptions")
        else:
            url = _join_core_v1_url(self._server_base_url, "/audio/transcriptions")
        resp, _resp_headers = self._post_multipart(url, headers=headers, data=data, file_path=media_paths[0])
        text = resp.get("text") if isinstance(resp, dict) else None
        if text is None and isinstance(resp, dict):
            text = resp.get("content") or resp.get("data")
        text_resp = {
            "content": str(text or "").strip(),
            "model": data.get("model"),
            "metadata": {"task": "transcription", "modality": "text", "_provider_request": {"url": url, "payload": data}},
        }
        return _normalize_multimodal_response(
            {"text": text_resp, "metadata": {"model": data.get("model"), "provider": "abstractcore-server"}},
            artifact_store=self._artifact_store,
        )

    def _generate_remote_multimodal(
        self,
        *,
        prompt: str,
        messages: Optional[List[Dict[str, Any]]],
        media: Optional[List[Any]],
        params: Dict[str, Any],
        headers: Dict[str, str],
    ) -> Dict[str, Any]:
        output_request = params.get("output")
        specs = _normalize_output_specs_for_runtime(output_request)
        if len(specs) != 1:
            raise ValueError("Remote multimodal generation currently supports one output spec per LLM_CALL.")

        spec = specs[0]
        modality = str(spec.get("modality") or "").strip().lower()
        task = str(spec.get("task") or "").strip().lower()
        text = str(params.get("text") or prompt or "").strip()
        if not text and isinstance(messages, list):
            for msg in reversed(messages):
                if not isinstance(msg, dict) or msg.get("role") != "user":
                    continue
                content = msg.get("content")
                content_text = _text_from_message_content(content)
                if content_text:
                    text = content_text
                    break

        if modality == "image":
            image_edit_tasks = {"image_edit", "image_to_image", "i2i", "edit_image"}
            image_upscale_tasks = {"image_upscale", "image_upscaling", "upscale", "upscale_image"}
            image_generation_tasks = {"", "image_generation", "t2i", "text_to_image"}
            if task in image_edit_tasks:
                if not text:
                    raise ValueError("Remote image edit requires prompt or text.")
                return self._remote_image_edit(spec=spec, prompt=text, media=media, headers=headers, params=params)
            if task in image_upscale_tasks:
                return self._remote_image_upscale(spec=spec, media=media, headers=headers, params=params)
            if task not in image_generation_tasks:
                raise ValueError(f"Unsupported remote image task: {task!r}")
            if media:
                raise ValueError("Remote image generation does not accept input media; use task='image_edit' for image edits.")
            if not text:
                raise ValueError("Remote image generation requires prompt or text.")
            return self._remote_image_generation(spec=spec, prompt=text, headers=headers, params=params)

        if modality == "video":
            image_to_video_tasks = {"image_to_video", "i2v", "video_from_image", "video_edit"}
            text_to_video_tasks = {"", "video_generation", "text_to_video", "t2v"}
            if task in image_to_video_tasks:
                if not text:
                    raise ValueError("Remote image-to-video requires prompt or text.")
                return self._remote_image_to_video(spec=spec, prompt=text, media=media, headers=headers, params=params)
            if task not in text_to_video_tasks:
                raise ValueError(f"Unsupported remote video task: {task!r}")
            if media:
                raise ValueError(
                    "Remote text-to-video does not accept input media; use task='image_to_video' for image-to-video."
                )
            if not text:
                raise ValueError("Remote text-to-video requires prompt or text.")
            return self._remote_video_generation(spec=spec, prompt=text, headers=headers, params=params)

        if modality == "voice":
            if task in {"voice_clone", "clone"}:
                raise ValueError("Remote voice clone is not supported through this client yet; use local execution.")
            if media:
                raise ValueError("Remote voice output does not accept input audio media yet; use local execution for cloning or reference-guided TTS.")
            if not text:
                raise ValueError("Remote TTS requires prompt or text.")
            return self._remote_tts(spec=spec, text=text, headers=headers, params=params)

        if modality == "music":
            if media:
                raise ValueError("Remote music output does not accept input audio media yet; use lyrics/text fields instead.")
            if not text:
                raise ValueError("Remote music generation requires prompt or text.")
            return self._remote_music(spec=spec, prompt=text, headers=headers, params=params)

        if modality == "text" and (task == "transcription" or (media and not text)):
            audio_items = [item for item in list(media or []) if _is_audio_media_item(item)]
            if len(audio_items) != 1 or len(list(media or [])) != 1:
                raise ValueError("Remote transcription requires exactly one audio media item.")
            return self._remote_transcription(spec=spec, media=media, headers=headers, params=params)

        raise ValueError(f"Unsupported remote multimodal output: modality={modality!r} task={task!r}")

    def _resolve_media_for_call(
        self,
        media: Optional[List[Any]],
    ) -> tuple[Optional[List[Any]], Optional[tempfile.TemporaryDirectory]]:
        tmpdir: Optional[tempfile.TemporaryDirectory] = None
        if isinstance(media, list) and media and self._artifact_store is not None:
            has_artifacts = any(
                isinstance(item, dict)
                and (
                    (isinstance(item.get("$artifact"), str) and str(item.get("$artifact") or "").strip())
                    or (isinstance(item.get("artifact_id"), str) and str(item.get("artifact_id") or "").strip())
                )
                and not (isinstance(item.get("file_path"), str) and str(item.get("file_path") or "").strip())
                and item.get("content") is None
                for item in media
            )
            if has_artifacts:
                tmpdir = tempfile.TemporaryDirectory(prefix="abstractruntime_remote_llm_media_")
                try:
                    media = _resolve_media_artifacts(media, artifact_store=self._artifact_store, temp_dir=tmpdir.name)
                except Exception:
                    tmpdir.cleanup()
                    raise
            else:
                media = _resolve_media_artifacts(media, artifact_store=self._artifact_store)
        else:
            media = _resolve_media_artifacts(media, artifact_store=self._artifact_store)
        return media, tmpdir

    def generate(
        self,
        *,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        media: Optional[List[Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        resolved_media, tmpdir = self._resolve_media_for_call(media)
        try:
            return self._generate_resolved(
                prompt=prompt,
                messages=messages,
                system_prompt=system_prompt,
                tools=tools,
                media=resolved_media,
                params=params,
            )
        finally:
            if tmpdir is not None:
                tmpdir.cleanup()

    def stream_tts(
        self,
        *,
        text: str,
        output: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        stream_params = _normalize_prompt_cache_binding_params(dict(params or {}))
        provider_api_key = _pop_provider_api_key(stream_params)
        req_headers = self._headers_with_provider_api_key(provider_api_key)
        trace_metadata = stream_params.get("trace_metadata") if isinstance(stream_params.get("trace_metadata"), dict) else None
        if isinstance(trace_metadata, dict) and trace_metadata:
            req_headers["X-AbstractCore-Trace-Metadata"] = json.dumps(
                trace_metadata, ensure_ascii=False, separators=(",", ":")
            )
            header_map = {
                "actor_id": "X-AbstractCore-Actor-Id",
                "session_id": "X-AbstractCore-Session-Id",
                "run_id": "X-AbstractCore-Run-Id",
                "parent_run_id": "X-AbstractCore-Parent-Run-Id",
            }
            for key, header in header_map.items():
                val = trace_metadata.get(key)
                if val is not None and header not in req_headers:
                    req_headers[header] = str(val)

        spec = {"modality": "voice", "task": "tts"}
        if isinstance(output, dict):
            spec.update(output)
        # STREAM-LANE capability-defaults merge (same incident as the local
        # client's site): bare remote stream requests inherit the operator's
        # output.voice default instead of the server's fallback chain.
        spec = _with_capability_default_route(spec, getattr(self, "_capability_defaults", None))
        return self._remote_tts_stream(
            spec=spec,
            text=str(text or ""),
            headers=req_headers,
            params=stream_params,
        )

    def _generate_resolved(
        self,
        *,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        media: Optional[List[Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        params = _normalize_prompt_cache_binding_params(params)
        prompt_cache_attribution = params.pop("_prompt_cache_attribution", None)
        # REMOTE MODE DOES NOT STREAM LIVE DELTAS. The AbstractCore server call
        # below is a single non-streaming request (`"stream": False`), so the
        # runtime's per-call `_on_delta` is dropped here on purpose: a run that
        # asked for `_runtime.stream` still completes with the same durable
        # answer, only without a live preview, and the live view is told why.
        _mark_stream_unavailable(params.pop("_on_delta", None), "remote_core")
        prompt = _promote_text_param_to_prompt(prompt, params)
        provider_api_key = _pop_provider_api_key(params)
        req_headers = self._headers_with_provider_api_key(provider_api_key)
        requested_base_url = params.get("base_url")
        requested_provider = params.get("_provider")
        requested_model = params.get("_model")
        requested_thinking = _with_capability_default_reasoning(
            params, getattr(self, "_capability_defaults", None)
        )
        effective_model = self._effective_model_from_params(params)

        trace_metadata = params.pop("trace_metadata", None)
        system_prompt = _strip_system_context_header(system_prompt)
        output_request = params.get("output")
        acore_output_request = _is_abstractcore_output_request(output_request)
        if _output_request_has_generated_media(output_request) and self._artifact_store is None:
            raise ValueError("Generated media outputs require an ArtifactStore.")
        skip_turn_grounding = _output_request_has_non_text_result(output_request) or bool(media)
        runtime_grounding = _mark_grounding_prompt_injected(
            _runtime_grounding_metadata(trace_metadata if isinstance(trace_metadata, dict) else None),
            not skip_turn_grounding,
        )
        prompt, messages = _normalize_turn_grounding(
            prompt=str(prompt or ""),
            messages=messages,
            grounding=runtime_grounding if not skip_turn_grounding else None,
        )
        messages = _strip_internal_system_messages(messages)
        system_prompt, messages = _coalesce_leading_system_messages(
            system_prompt=system_prompt,
            messages=messages,
        )
        system_prompt = _append_runtime_grounding_contract(
            system_prompt,
            injected=not skip_turn_grounding,
        )

        if isinstance(trace_metadata, dict) and trace_metadata:
            req_headers["X-AbstractCore-Trace-Metadata"] = json.dumps(
                trace_metadata, ensure_ascii=False, separators=(",", ":")
            )
            header_map = {
                "actor_id": "X-AbstractCore-Actor-Id",
                "session_id": "X-AbstractCore-Session-Id",
                "run_id": "X-AbstractCore-Run-Id",
                "parent_run_id": "X-AbstractCore-Parent-Run-Id",
            }
            for key, header in header_map.items():
                val = trace_metadata.get(key)
                if val is not None and header not in req_headers:
                    req_headers[header] = str(val)

        resolved_generate_route_summary = None
        scoped_capability_defaults = (
            self._capability_defaults
            or _normalize_core_capability_defaults(getattr(self, "_abstractcore_capability_defaults", None))
        )
        scoped_core_config_file = str(getattr(self, "_abstractcore_config_file", self._core_config_file) or "").strip() or None
        if acore_output_request and "output" in params:
            # Extract runtime trace metadata BEFORE stripping for core: the
            # server must not see runtime keys, but artifact tracing needs
            # them (regression caught at the landing audit: stripped specs
            # reached _trace_run_id_and_tags and artifacts lost run_id/tags).
            _stash_run_id, _stash_tags = _output_runtime_metadata(params.get("output"))
            if _stash_run_id or _stash_tags:
                params["_runtime_output_metadata"] = {
                    "run_id": _stash_run_id,
                    "tags": _stash_tags,
                }
            params["output"] = _strip_runtime_output_metadata_for_core(params.get("output"))

        if acore_output_request and not tools and (scoped_capability_defaults or scoped_core_config_file):
            from abstractcore.core.generate_contract import normalize_generate_request, resolve_generate_route  # type: ignore

            resolved_generate_route = resolve_generate_route(
                request=normalize_generate_request(
                    prompt=str(prompt or ""),
                    messages=messages,
                    media=media,
                ),
                output=params.get("output"),
                scoped_routes=scoped_capability_defaults,
                config_file=scoped_core_config_file,
                explicit_text_route={
                    "provider": requested_provider,
                    "model": requested_model,
                    "base_url": requested_base_url,
                },
                explicit_reasoning=requested_thinking,
            )
            resolved_generate_route_summary = resolved_generate_route.to_summary()
            maybe_specs = [dict(spec) for spec in resolved_generate_route.output_specs]
            if (
                len(maybe_specs) == 1
                and maybe_specs[0].get("modality") == "text"
                and maybe_specs[0].get("task") not in {"transcription"}
                and str(prompt or "").strip()
            ):
                params.pop("output", None)
                acore_output_request = False
            else:
                params["output"] = maybe_specs[0] if len(maybe_specs) == 1 else maybe_specs

        if acore_output_request and (skip_turn_grounding or (media and not str(prompt or "").strip())):
            params_for_mm = dict(params)
            if isinstance(trace_metadata, dict):
                params_for_mm["trace_metadata"] = trace_metadata
            result = self._generate_remote_multimodal(
                prompt=str(prompt or ""),
                messages=messages,  # type: ignore[arg-type]
                media=media,
                params=params_for_mm,
                headers=req_headers,
            )
            if resolved_generate_route_summary is not None:
                meta = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
                meta["_resolved_generate_route"] = resolved_generate_route_summary
                result["metadata"] = meta
            _sanitize_runtime_grounding_echoes(result)
            _attach_runtime_grounding(result, runtime_grounding)
            return result

        # Build OpenAI-like messages for AbstractCore server.
        messages = _strip_synthetic_message_markers(_strip_volatile_markers(messages))
        out_messages: List[Dict[str, Any]] = []
        if system_prompt:
            out_messages.append({"role": "system", "content": system_prompt})

        if messages:
            out_messages.extend([dict(m) for m in messages if isinstance(m, dict)])
        else:
            out_messages.append({"role": "user", "content": prompt})

        if media:
            # AbstractCore Server chat endpoints accept OpenAI content arrays with
            # image_url/file data URLs. Attach media to the last user turn.
            user_idx = None
            for i in range(len(out_messages) - 1, -1, -1):
                if out_messages[i].get("role") == "user":
                    user_idx = i
                    break
            if user_idx is None:
                out_messages.append({"role": "user", "content": ""})
                user_idx = len(out_messages) - 1
            out_messages[user_idx]["content"] = _merge_remote_media_content(
                out_messages[user_idx].get("content"),
                media=media,
            )

        body: Dict[str, Any] = {
            "model": effective_model,
            "messages": out_messages,
            # Always non-streaming in remote mode: no live token deltas (see the
            # `_on_delta` note at the top of this method).
            "stream": False,
            # Orchestrator policy: ask AbstractCore server to use the same timeout it expects.
            # This keeps runtime authority even when the actual provider call happens server-side.
            "timeout_s": self._timeout_s,
        }

        # Dynamic routing support (AbstractCore server feature).
        base_url = params.pop("base_url", None)
        if base_url:
            body["base_url"] = base_url

        prompt_cache_key = params.get("prompt_cache_key")
        if isinstance(prompt_cache_key, str) and prompt_cache_key.strip():
            body["prompt_cache_key"] = prompt_cache_key.strip()
        prompt_cache_binding = params.get("prompt_cache_binding")
        if prompt_cache_binding is not None:
            body["prompt_cache_binding"] = _jsonable(prompt_cache_binding)

        # Thinking/reasoning control (reasoning-1st-citizen R-A, 2026-07-26):
        # the abstractcore server accepts `thinking` on its chat routes, but
        # this pass-through allowlist silently DROPPED it — any gateway built
        # on the remote runtime lost reasoning config entirely (found
        # independently by two adversaries: runtime's plan cycle-2 and
        # agent's cycle-1 P0). `thinking` was already read above for route
        # resolution; now it rides the POST body too.
        thinking = params.get("thinking")
        if isinstance(thinking, bool):
            body["thinking"] = thinking
        elif isinstance(thinking, str) and thinking.strip():
            body["thinking"] = thinking.strip()
        elif thinking is not None and not isinstance(thinking, str):
            raise ValueError("thinking must be a bool or a reasoning-effort string")

        # Core owns speculation validation/execution. Transport an explicit
        # False unchanged: it is an override, not an absent request/default.
        speculation = params.get("speculation")
        if speculation is None:
            policy = self._scoped_speculation_default()
            if policy is not _UNSET:
                speculation = policy
        if speculation is not None:
            if not isinstance(speculation, (bool, dict)):
                raise ValueError("Remote speculation must be a bool or a dict")
            body["speculation"] = _jsonable(speculation)

        # Pass through common OpenAI-compatible parameters.
        for key in (
            "temperature",
            "max_tokens",
            "stop",
            "seed",
            "frequency_penalty",
            "presence_penalty",
        ):
            if key in params and params[key] is not None:
                if key == "seed":
                    try:
                        seed_i = int(params[key])
                    except Exception:
                        continue
                    if seed_i >= 0:
                        body[key] = seed_i
                    continue
                if key == "temperature":
                    try:
                        body[key] = float(params[key])
                    except Exception:
                        continue
                    continue
                body[key] = params[key]

        if tools is not None:
            body["tools"] = tools

        url = _join_core_v1_url(self._server_base_url, "/chat/completions")
        # Stop (2026-09-23): the effect's cancel event severs this request; the
        # AbstractCore server's client-disconnect watcher then cancels the
        # generation (and severs ITS upstream model request in turn).
        remote_cancel = params.get("cancel_event")
        if isinstance(remote_cancel, threading.Event) and getattr(self._sender, "supports_cancel_event", False):
            raw = self._sender.post(
                url, headers=req_headers, json=body, timeout=self._timeout_s, cancel_event=remote_cancel
            )
        else:
            if isinstance(remote_cancel, threading.Event) and not getattr(self, "_remote_cancel_unsupported_warned", False):
                self._remote_cancel_unsupported_warned = True
                logger.warning(
                    "Remote AbstractCore request sender does not support cancel_event: a Stop cannot "
                    "sever in-flight remote generations (sender=%s)" % type(self._sender).__name__
                )
            raw = self._sender.post(url, headers=req_headers, json=body, timeout=self._timeout_s)
        resp, resp_headers = _unwrap_http_response(raw)
        lower_headers = {str(k).lower(): str(v) for k, v in resp_headers.items()}
        trace_id = lower_headers.get("x-abstractcore-trace-id") or lower_headers.get("x-trace-id")

        self._maybe_stamp_prompt_cache_attribution(
            key=body.get("prompt_cache_key"),
            attribution=prompt_cache_attribution,
            effective_model=effective_model,
        )

        # Normalize OpenAI-like response.
        try:
            choice0 = (resp.get("choices") or [])[0]
            msg = choice0.get("message") or {}
            observable_body = _redact_data_urls_for_observability(body)
            meta: Dict[str, Any] = {
                "_provider_request": {"url": url, "payload": observable_body}
            }
            # Preserve Core's actual execution outcomes just as the local
            # adapter does, without accepting remote trace/request provenance.
            core_metadata = resp.get("abstractcore")
            if isinstance(core_metadata, dict):
                for key in ("execution", "speculation", "performance", "prompt_cache"):
                    if isinstance(core_metadata.get(key), dict):
                        meta[key] = _jsonable(core_metadata[key])
            if runtime_grounding:
                meta["runtime_grounding"] = dict(runtime_grounding)
            if trace_id:
                meta["trace_id"] = trace_id
            reasoning = msg.get("reasoning")
            if not isinstance(reasoning, str) or not reasoning.strip():
                reasoning = msg.get("reasoning_content")
            if not isinstance(reasoning, str) or not reasoning.strip():
                reasoning = msg.get("thinking")
            if not isinstance(reasoning, str) or not reasoning.strip():
                reasoning = msg.get("thinking_content")
            result = {
                "content": msg.get("content"),
                "reasoning": reasoning,
                "data": None,
                "raw_response": _jsonable(resp) if resp is not None else None,
                "tool_calls": _jsonable(msg.get("tool_calls")) if msg.get("tool_calls") is not None else None,
                "usage": _jsonable(resp.get("usage")) if resp.get("usage") is not None else None,
                "model": resp.get("model"),
                "finish_reason": choice0.get("finish_reason"),
                "metadata": meta,
                "trace_id": trace_id,
            }
            _sanitize_runtime_grounding_echoes(result)
            result["tool_calls"] = _normalize_tool_calls(result.get("tool_calls"))

            return result
        except Exception:
            # Fallback: return the raw response in JSON-safe form.
            logger.warning("Remote LLM response normalization failed; returning raw JSON")
            return {
                "content": None,
                "data": _jsonable(resp),
                "tool_calls": None,
                "usage": None,
                "model": resp.get("model") if isinstance(resp, dict) else None,
                "finish_reason": None,
                "metadata": {
                    "_provider_request": {"url": url, "payload": _redact_data_urls_for_observability(body)},
                    "runtime_grounding": dict(runtime_grounding) if runtime_grounding else None,
                    "trace_id": trace_id,
                }
                if trace_id
                else {
                    "_provider_request": {"url": url, "payload": _redact_data_urls_for_observability(body)},
                    "runtime_grounding": dict(runtime_grounding) if runtime_grounding else None,
                },
                "trace_id": trace_id,
                "raw_response": _jsonable(resp) if resp is not None else None,
            }
