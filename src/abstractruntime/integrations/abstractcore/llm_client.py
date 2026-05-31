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
import hashlib
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
import uuid
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol, Tuple
from urllib.parse import quote, urlencode

from .logging import get_logger
from .output_specs import (
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


@dataclass
class _PromptCacheSessionState:
    system_module_hash: str
    tools_module_hash: str
    prefix_cache_key: str
    message_hashes: List[str]


def _prompt_cache_message_fingerprint(message: Any) -> str:
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

    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _pop_provider_api_key(values: Dict[str, Any]) -> Optional[str]:
    """Return and remove a per-request provider key from common compatibility names."""

    for key in ("provider_api_key", "api_key"):
        raw = values.pop(key, None)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return None


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

    MLX/Metal can crash the process when concurrent generations occur from multiple threads
    (e.g. gateway ticking multiple runs concurrently). We serialize MLX generation per model
    as a safety contract.
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
    2) Locale region from `locale.getlocale()` or locale env vars (LANG/LC_ALL/LC_CTYPE)
    3) Timezone (IANA name) via zone.tab mapping

    Notes:
    - Avoid parsing encoding-only strings like `UTF-8` as a country (a common locale env pitfall).
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

    tz_name = _detect_timezone_name()
    if tz_name:
        cc = _country_from_zone_tab(zone_name=tz_name)
        if cc is not None:
            return cc

    return "XX"


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
    return metadata


def _runtime_grounding_prompt_envelope(grounding: Dict[str, Any]) -> str:
    prompt_payload: Dict[str, Any] = {}
    for key in ("local_datetime", "timezone", "country", "user", "display"):
        value = grounding.get(key)
        if value is not None and str(value).strip():
            prompt_payload[key] = value
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


def _inject_runtime_grounding_into_text(text: str, grounding: Dict[str, Any]) -> str:
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
    """Inject runtime context into the current user turn for LLM calls only.

    The envelope is deliberately tagged and machine-owned:
      <runtime_metadata>{...}</runtime_metadata>

    That keeps date/location/user context visible to the model without mutating
    the durable human text field into a natural-language prefix that downstream
    TTS may speak. Media-only requests call this function with `grounding=None`,
    which strips legacy prefixes but does not inject new prompt text.
    """

    def _clean_or_inject_text(value: str) -> str:
        if grounding:
            return _inject_runtime_grounding_into_text(value, grounding)
        return _strip_runtime_grounding_prefix(value)

    def _clean_content(content: Any) -> Any:
        if isinstance(content, str):
            return _clean_or_inject_text(content)
        if isinstance(content, list):
            items: List[Any] = [dict(item) if isinstance(item, dict) else item for item in content]
            for idx, item in enumerate(items):
                if not isinstance(item, dict):
                    continue
                if str(item.get("type") or "").strip().lower() != "text":
                    continue
                text_value = item.get("text")
                item["text"] = _clean_or_inject_text(text_value if isinstance(text_value, str) else str(text_value or ""))
                items[idx] = item
                return items
            if grounding:
                items.insert(0, {"type": "text", "text": _runtime_grounding_prompt_envelope(grounding)})
            return items
        return _clean_or_inject_text(str(content or ""))

    prompt_str = str(prompt or "")
    if prompt_str.strip():
        return _clean_or_inject_text(prompt_str), messages

    if isinstance(messages, list) and messages:
        out: List[Dict[str, Any]] = []
        for m in messages:
            out.append(dict(m) if isinstance(m, dict) else {"role": "user", "content": str(m)})

        for i in range(len(out) - 1, -1, -1):
            role = str(out[i].get("role") or "").strip().lower()
            if role != "user":
                continue
            out[i]["content"] = _clean_content(out[i].get("content"))
            return prompt_str, out

        if grounding:
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

    def get_model_capabilities(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Return model capability metadata for a specific model or the default client model."""

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
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Request best-effort model unload."""


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
    if media:
        # Image edits can carry artifact-backed files and masks. Keep those in-process until
        # the subprocess transport has an explicit media-file contract.
        return False
    if not specs:
        return False
    for spec in specs:
        if not isinstance(spec, dict):
            return False
        modality = str(spec.get("modality") or "").strip().lower()
        task = str(spec.get("task") or "").strip().lower()
        if modality != "image":
            return False
        if task and task not in {"image_generation", "t2i", "text_to_image"}:
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


def _local_image_subprocess_timeout_s(params: Dict[str, Any], llm_kwargs: Dict[str, Any]) -> float:
    for raw in (
        os.environ.get("ABSTRACTRUNTIME_LOCAL_IMAGE_SUBPROCESS_TIMEOUT_S"),
        params.get("timeout_s"),
        params.get("timeout"),
        llm_kwargs.get("timeout_s"),
        llm_kwargs.get("timeout"),
    ):
        if raw is None:
            continue
        try:
            timeout = float(raw)
        except Exception:
            continue
        if timeout > 0:
            return timeout
    return 3600.0


def _local_video_subprocess_timeout_s(params: Dict[str, Any], llm_kwargs: Dict[str, Any]) -> float:
    for raw in (
        os.environ.get("ABSTRACTRUNTIME_LOCAL_VIDEO_SUBPROCESS_TIMEOUT_S"),
        os.environ.get("ABSTRACTRUNTIME_LOCAL_MEDIA_SUBPROCESS_TIMEOUT_S"),
        params.get("timeout_s"),
        params.get("timeout"),
        llm_kwargs.get("timeout_s"),
        llm_kwargs.get("timeout"),
    ):
        if raw is None:
            continue
        try:
            timeout = float(raw)
        except Exception:
            continue
        if timeout > 0:
            return timeout
    return 7200.0


def _run_local_image_subprocess(
    *,
    provider: str,
    model: str,
    llm_kwargs: Dict[str, Any],
    prompt: str,
    specs: List[Dict[str, Any]],
    timeout_s: float,
) -> Dict[str, Any]:
    request = {
        "provider": str(provider or ""),
        "model": str(model or ""),
        "llm_kwargs": _jsonable(llm_kwargs or {}),
        "prompt": str(prompt or ""),
        "specs": _jsonable(specs),
    }
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")

    def _invoke() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "abstractruntime.integrations.abstractcore.media_subprocess"],
            input=json.dumps(request, ensure_ascii=False),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_s,
            env=env,
            check=False,
        )

    if _env_flag_enabled("ABSTRACTRUNTIME_LOCAL_IMAGE_SUBPROCESS_SERIALIZE", default=True):
        with _LOCAL_IMAGE_SUBPROCESS_LOCK:
            proc = _invoke()
    else:
        proc = _invoke()

    parsed: Optional[Dict[str, Any]] = None
    try:
        parsed = _loads_last_json_line(proc.stdout)
    except Exception:
        parsed = None

    if proc.returncode != 0:
        stderr = str(proc.stderr or "").strip()
        stdout = str(proc.stdout or "").strip()
        if parsed and parsed.get("ok") is False:
            detail = str(parsed.get("error") or "local image generation failed")
        else:
            detail = stderr or stdout or "local image generation failed"
        raise RuntimeError(
            f"Local image generation subprocess exited with code {proc.returncode}: {detail[-2000:]}"
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
    timeout_s: float,
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
            stderr=subprocess.STDOUT,
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
        deadline = time.time() + float(timeout_s)
        selector = selectors.DefaultSelector()
        if proc.stdout is not None:
            selector.register(proc.stdout, selectors.EVENT_READ)

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

        try:
            while True:
                if time.time() >= deadline and proc.poll() is None:
                    proc.kill()
                    raise TimeoutError(f"Local video generation subprocess timed out after {timeout_s:.0f}s.")
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
                    _consume_stdout_line(line)
                if proc.poll() is not None:
                    break
            if proc.stdout is not None:
                try:
                    remaining = proc.stdout.read()
                except Exception:
                    remaining = ""
                for line in str(remaining or "").splitlines():
                    _consume_stdout_line(line)
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
        binding = {"binding_id": binding}
    elif not isinstance(binding, dict):
        raise ValueError("prompt_cache_binding must be an object or binding_id string.")
    else:
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
        "transcription": "stt",
        "transcriptions": "stt",
        "transcribe": "stt",
        "speech_to_text": "stt",
        "audio_transcription": "stt",
        "audio_transcriptions": "stt",
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
    "video_generation",
    "text_to_video",
    "image_to_video",
    "tts",
    "stt",
    "music_generation",
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
    if task_s in {"image_generation", "image_to_image", "video_generation", "text_to_video", "image_to_video"}:
        payload["execution_mode"] = "local_one_shot_subprocess"
        payload["local_media_residency_backend"] = "none"
        payload["requires_long_lived_core_backend"] = True
        payload["requires_long_lived_server"] = True
        payload["config_hint"] = (
            "Set ABSTRACTCORE_SERVER_BASE_URL to a long-lived AbstractCore server to enable media warmup."
        )
    elif task_s in {"tts", "stt", "music_generation"}:
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
    return claim


def _local_residency_record(
    *,
    provider: str,
    model: str,
    default: bool = False,
    runtime_cached: bool = True,
    provider_instance: Any = None,
    include_provider_state: bool = True,
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

    return {
        "task": "text_generation",
        "provider": provider_s,
        "model": model_s,
        "runtime_id": f"local:text_generation:{provider_s}:{model_s}",
        "resident": resident,
        "loaded": resident,
        "state": state,
        "runtime_cached": bool(runtime_cached),
        "cache_state": "runtime_client_cached" if runtime_cached else "not_cached",
        "pinned": bool(default),
        "default": bool(default),
        "source": "abstractruntime.local",
        "isolation": "in_process",
        **provider_claim,
    }


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
    if task_s == "music_generation":
        return getattr(core, "music", None), "music"
    if task_s in {"image_generation", "image_to_image"}:
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
    task_counts: Dict[str, int] = {"text_generation": len(text_records)}
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
            if content_type:
                resolved["content_type"] = str(content_type)
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


def _store_generated_bytes(
    data: bytes,
    *,
    artifact_store: Optional[Any],
    run_id: Optional[str],
    content_type: str,
    tags: Optional[Dict[str, Any]] = None,
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
        elif modality in {"voice", "audio", "music"}:
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

    if artifact_ref is None and isinstance(data, (bytes, bytearray)):
        tags = _string_tags(default_tags)
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
        )
        artifact_ref = stored_ref

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
                    fallback_modality=str(modality),
                )
                for item in items
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


def _normalize_local_streaming_response(stream: Any) -> Dict[str, Any]:
    """Consume an AbstractCore streaming `generate(..., stream=True)` iterator into a single JSON result.

    AbstractRuntime currently persists a single effect outcome object per LLM call, so even when
    the underlying provider streams we aggregate into one final dict and surface timing fields.
    """
    import time

    start_perf = time.perf_counter()

    chunks: list[str] = []
    tool_calls: Any = None
    usage: Any = None
    model: Optional[str] = None
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = {}
    trace_id: Optional[str] = None
    reasoning: Optional[str] = None
    ttft_ms: Optional[float] = None

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
            content = chunk.get("content")
            if isinstance(content, str) and content:
                chunks.append(content)

            tc = chunk.get("tool_calls")
            if tc is not None:
                tool_calls = tc

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
                    if reasoning is None and isinstance(r, str) and r.strip():
                        reasoning = r.strip()
            continue

        content = getattr(chunk, "content", None)
        if isinstance(content, str) and content:
            chunks.append(content)

        tc = getattr(chunk, "tool_calls", None)
        if tc is not None:
            tool_calls = tc

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
                if reasoning is None and isinstance(r, str) and r.strip():
                    reasoning = r.strip()

    gen_time = round((time.perf_counter() - start_perf) * 1000, 1)

    return {
        "content": "".join(chunks),
        "reasoning": reasoning,
        "data": None,
        "tool_calls": _jsonable(tool_calls) if tool_calls is not None else None,
        "usage": _jsonable(usage) if usage is not None else None,
        "model": model,
        "finish_reason": finish_reason,
        "metadata": metadata or None,
        "trace_id": trace_id,
        "gen_time": gen_time,
        "ttft_ms": ttft_ms,
    }


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
        self._generate_lock = _local_generate_lock(provider=self._provider, model=self._model)
        if self._generate_lock is not None:
            _warn_local_generate_lock_once(provider=self._provider, model=self._model)
        kwargs = dict(llm_kwargs or {})
        kwargs.setdefault("enable_tracing", True)
        if kwargs.get("enable_tracing"):
            # Keep a small in-memory ring buffer for exact request/response observability.
            # This enables hosts (AbstractCode/AbstractFlow) to inspect trace payloads by trace_id.
            kwargs.setdefault("max_traces", 50)
        self._llm_kwargs = dict(kwargs)
        self._llm = create_llm(provider, model=model, **kwargs)
        self._tool_handler = UniversalToolHandler(model)
        self._prompt_cache_state_lock = threading.Lock()
        self._prompt_cache_state: Dict[str, _PromptCacheSessionState] = {}
        self._capability_residency_core = None
        self._capability_residency_core_lock = threading.Lock()

    def default_prompt_cache_identity(self) -> Tuple[Optional[str], Optional[str]]:
        return self._provider, self._model

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
        )
        if isinstance(provider, str) and provider.strip() and provider.strip().lower() != self._provider:
            records: List[Dict[str, Any]] = []
        elif isinstance(model, str) and model.strip() and model.strip() != self._model:
            records = []
        else:
            records = [record]
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
        **kwargs: Any,
    ) -> Dict[str, Any]:
        _ = kwargs
        task_s = _normalize_residency_task(task)
        if task_s != "text_generation":
            provider_s = str(provider or "").strip().lower()
            model_s = str(model or "").strip()
            return _local_capability_residency_result(
                self,
                operation="unload",
                task=task_s,
                provider=provider_s,
                model=model_s,
                options=options,
                runtime_id=runtime_id,
                kwargs=dict(kwargs or {}),
                source="abstractruntime.local",
            )
        provider_s = str(provider or self._provider or "").strip().lower()
        model_s = str(model or self._model or "").strip()
        if provider_s != self._provider or model_s != self._model:
            requested = _local_residency_record(
                provider=provider_s,
                model=model_s,
                default=False,
                runtime_cached=False,
                include_provider_state=False,
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

        record = _local_residency_record(
            provider=self._provider,
            model=self._model,
            default=True,
            provider_instance=getattr(self, "_llm", None),
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
            error = "model_residency unload completed but provider still reports the model loaded"
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

    def _maybe_prepare_prompt_cache(
        self,
        *,
        prompt_cache_key: Optional[str],
        system_prompt: Optional[str],
        tools: Optional[List[Dict[str, Any]]],
        messages: Optional[List[Dict[str, Any]]],
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

        # Build immutable prefix caches for (system, tools), then maintain a mutable per-session
        # cache (history) under `prompt_cache_key`.
        try:
            prep_fn = getattr(provider, "prompt_cache_prepare_modules", None)
            if not callable(prep_fn):
                return
            modules: List[Dict[str, Any]] = [
                {"module_id": "system", "system_prompt": system_prompt, "add_generation_prompt": False}
            ]
            if tools:
                modules.append({"module_id": "tools", "tools": tools, "add_generation_prompt": False})
            prep = prep_fn(
                namespace="abstractcode",
                modules=modules,
                make_default=False,
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

        msg_list: List[Dict[str, Any]] = list(messages) if isinstance(messages, list) and messages else []
        msg_hashes: List[str] = [_prompt_cache_message_fingerprint(m) for m in msg_list]

        with self._prompt_cache_state_lock:
            state = self._prompt_cache_state.get(key)
            needs_rebuild = (
                state is None
                or state.system_module_hash != system_hash
                or state.tools_module_hash != tools_hash
                or state.prefix_cache_key != final_prefix_key
            )

            if needs_rebuild:
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
                            updater(key, system_prompt=system_prompt, tools=tools, add_generation_prompt=False)
                            forked = True
                    except Exception:
                        forked = False

                if not forked:
                    return

                state = _PromptCacheSessionState(
                    system_module_hash=system_hash,
                    tools_module_hash=tools_hash,
                    prefix_cache_key=final_prefix_key,
                    message_hashes=[],
                )
                self._prompt_cache_state[key] = state

            if not msg_list:
                state.message_hashes = []
                return

            if msg_hashes[: len(state.message_hashes)] == state.message_hashes:
                new_msgs = msg_list[len(state.message_hashes) :]
                if not new_msgs:
                    return
                try:
                    updater = getattr(provider, "prompt_cache_update", None)
                    if callable(updater) and bool(updater(key, messages=new_msgs, add_generation_prompt=False)):
                        state.message_hashes.extend(msg_hashes[len(state.message_hashes) :])
                except Exception:
                    pass
                return

            # History diverged (edits/truncation): rebuild per-session history cache from the prefix.
            try:
                clearer = getattr(provider, "prompt_cache_clear", None)
                if callable(clearer):
                    clearer(key)
            except Exception:
                pass
            try:
                forker = getattr(provider, "prompt_cache_fork", None)
                if not callable(forker) or not bool(forker(final_prefix_key, key, make_default=False)):
                    return
            except Exception:
                return
            try:
                updater = getattr(provider, "prompt_cache_update", None)
                if callable(updater) and bool(updater(key, messages=msg_list, add_generation_prompt=False)):
                    state.message_hashes = list(msg_hashes)
            except Exception:
                pass

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
            prompt = _promote_text_param_to_prompt(prompt, params)
            has_binding = _has_prompt_cache_binding(params)
            output_request = params.get("output")
            acore_output_request = _is_abstractcore_output_request(output_request)
            if _output_request_has_generated_media(output_request) and self._artifact_store is None:
                raise ValueError("Generated media outputs require an ArtifactStore.")
            skip_turn_grounding = _output_request_has_non_text_result(output_request)
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

            stream_raw = params.pop("stream", None)
            if stream_raw is None:
                stream_raw = params.pop("streaming", None)
            if isinstance(stream_raw, str):
                stream = stream_raw.strip().lower() in {"1", "true", "yes", "y", "on"}
            else:
                stream = bool(stream_raw) if stream_raw is not None else False

            # `base_url` is a provider construction concern in local mode. We intentionally
            # do not create new providers per call unless the host explicitly chooses to.
            params.pop("base_url", None)
            # Reserved routing keys (used by MultiLocalAbstractCoreLLMClient).
            params.pop("_provider", None)
            params.pop("_model", None)

            if acore_output_request and "output" in params:
                params["output"] = _strip_runtime_output_metadata_for_core(params.get("output"))

            if acore_output_request and not tools:
                specs = _normalize_output_specs_for_runtime(output_request)
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
                            timeout_s=_local_image_subprocess_timeout_s(params, getattr(self, "_llm_kwargs", {})),
                        )
                        result = _normalize_multimodal_response(
                            result_obj,
                            artifact_store=self._artifact_store,
                            run_id=run_id,
                            default_tags=default_artifact_tags,
                        )
                    elif _is_subprocess_safe_video_specs(specs, media):
                        result_obj = _run_local_video_subprocess(
                            provider=self._provider,
                            model=self._model,
                            llm_kwargs=getattr(self, "_llm_kwargs", {}),
                            prompt=str(prompt or ""),
                            specs=[dict(spec) for spec in specs],
                            media=media,
                            timeout_s=_local_video_subprocess_timeout_s(params, getattr(self, "_llm_kwargs", {})),
                            progress_callback=params.get("on_progress"),
                        )
                        result = _normalize_multimodal_response(
                            result_obj,
                            artifact_store=self._artifact_store,
                            run_id=run_id,
                            default_tags=default_artifact_tags,
                        )
                    else:
                        from abstractcore.core.multimodal_generation import MultimodalGenerateResponse  # type: ignore

                        result_obj = MultimodalGenerateResponse(
                            metadata={
                                "media_only": True,
                                "runtime_provider": self._provider,
                                "runtime_model": self._model,
                            }
                        )
                        for spec in specs:
                            run_spec(
                                result=result_obj,
                                spec=dict(spec),
                                prompt=str(prompt or ""),
                                media=media,
                                artifact_store=self._artifact_store,
                            )
                        result = _normalize_local_response(
                            result_obj,
                            artifact_store=self._artifact_store,
                            run_id=run_id,
                            default_tags=default_artifact_tags,
                        )
                    _sanitize_runtime_grounding_echoes(result)
                    _attach_runtime_grounding(result, runtime_grounding)
                    result["tool_calls"] = []
                    return result

            lock = getattr(self, "_generate_lock", None)
            if lock is None:
                if not has_binding:
                    self._maybe_prepare_prompt_cache(
                        prompt_cache_key=params.get("prompt_cache_key"),
                        system_prompt=system_prompt,
                        tools=tools,
                        messages=messages,
                    )
                resp = self._llm.generate(
                    prompt=str(prompt or ""),
                    messages=messages,
                    system_prompt=system_prompt,
                    tools=tools,
                    media=media,
                    stream=stream,
                    **params,
                )
                if stream and hasattr(resp, "__next__"):
                    result = _normalize_local_streaming_response(resp)
                else:
                    result = _normalize_local_response(
                        resp,
                        artifact_store=self._artifact_store if acore_output_request else None,
                        run_id=run_id,
                        default_tags=default_artifact_tags,
                    )
                _sanitize_runtime_grounding_echoes(result)
                _attach_runtime_grounding(result, runtime_grounding)
                result["tool_calls"] = _normalize_tool_calls(result.get("tool_calls"))
            else:
                # Serialize generation for non-thread-safe providers (e.g. MLX).
                with lock:
                    if not has_binding:
                        self._maybe_prepare_prompt_cache(
                            prompt_cache_key=params.get("prompt_cache_key"),
                            system_prompt=system_prompt,
                            tools=tools,
                            messages=messages,
                        )
                    resp = self._llm.generate(
                        prompt=str(prompt or ""),
                        messages=messages,
                        system_prompt=system_prompt,
                        tools=tools,
                        media=media,
                        stream=stream,
                        **params,
                    )
                    if stream and hasattr(resp, "__next__"):
                        result = _normalize_local_streaming_response(resp)
                    else:
                        result = _normalize_local_response(
                            resp,
                            artifact_store=self._artifact_store if acore_output_request else None,
                            run_id=run_id,
                            default_tags=default_artifact_tags,
                        )
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
                    payload["params"] = _jsonable(params) if params else {}

                    meta["_provider_request"] = {
                        "transport": "local",
                        "provider": str(self._provider),
                        "model": str(self._model),
                        "payload": payload,
                    }
            except Exception:
                # Never fail an LLM call due to observability.
                pass

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
        _ = kwargs
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
            result = provider.prompt_cache_prepare_modules(
                namespace=namespace,
                modules=modules,
                make_default=bool(make_default),
                ttl_s=ttl_s,
                version=int(version),
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
    ):
        self._llm_kwargs = dict(llm_kwargs or {})
        self._default_provider = provider.strip().lower()
        self._default_model = model.strip()
        self._artifact_store = artifact_store
        self._bloc_root_dir = _coerce_bloc_root_dir(bloc_root_dir)
        self._prompt_cache_export_root_dir = _coerce_prompt_cache_export_root_dir(prompt_cache_export_root_dir)
        self._clients: Dict[Tuple[str, str], LocalAbstractCoreLLMClient] = {}
        self._override_clients: Dict[Tuple[str, str, str, str], LocalAbstractCoreLLMClient] = {}
        self._capability_residency_core = None
        self._capability_residency_core_lock = threading.Lock()
        self._default_client = self._get_client(self._default_provider, self._default_model)

        # Provide a stable underlying LLM for components that need one (e.g. summarizer).
        self._llm = getattr(self._default_client, "_llm", None)

    def default_prompt_cache_identity(self) -> Tuple[Optional[str], Optional[str]]:
        return self._default_provider, self._default_model

    def _create_client(
        self,
        provider: str,
        model: str,
        *,
        llm_kwargs_override: Optional[Dict[str, Any]] = None,
    ) -> LocalAbstractCoreLLMClient:
        key = (provider.strip().lower(), model.strip())
        llm_kwargs = dict(self._llm_kwargs)
        if llm_kwargs_override:
            llm_kwargs.update(dict(llm_kwargs_override))
        try:
            return LocalAbstractCoreLLMClient(
                provider=key[0],
                model=key[1],
                llm_kwargs=llm_kwargs,
                artifact_store=self._artifact_store,
                bloc_root_dir=self._bloc_root_dir,
                prompt_cache_export_root_dir=self._prompt_cache_export_root_dir,
            )
        except TypeError as exc:
            message = str(exc)
            if "prompt_cache_export_root_dir" not in message and "bloc_root_dir" not in message:
                raise
            try:
                return LocalAbstractCoreLLMClient(
                    provider=key[0],
                    model=key[1],
                    llm_kwargs=llm_kwargs,
                    artifact_store=self._artifact_store,
                    bloc_root_dir=self._bloc_root_dir,
                )
            except TypeError as fallback_exc:
                if "bloc_root_dir" not in str(fallback_exc):
                    raise
                return LocalAbstractCoreLLMClient(
                    provider=key[0],
                    model=key[1],
                    llm_kwargs=llm_kwargs,
                    artifact_store=self._artifact_store,
                )

    def _get_client(
        self,
        provider: str,
        model: str,
        *,
        llm_kwargs_override: Optional[Dict[str, Any]] = None,
    ) -> LocalAbstractCoreLLMClient:
        key = (provider.strip().lower(), model.strip())
        if llm_kwargs_override:
            base_url = str(llm_kwargs_override.get("base_url") or "").strip()
            api_key = str(llm_kwargs_override.get("api_key") or "").strip()
            api_key_fp = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16] if api_key else ""
            override_key = (key[0], key[1], base_url, api_key_fp)
            override_clients = getattr(self, "_override_clients", None)
            if override_clients is None:
                override_clients = {}
                self._override_clients = override_clients
            client = override_clients.get(override_key)
            if client is None:
                client = self._create_client(key[0], key[1], llm_kwargs_override=llm_kwargs_override)
                override_clients[override_key] = client
            return client
        client = self._clients.get(key)
        if client is None:
            client = self._create_client(key[0], key[1])
            self._clients[key] = client
        return client

    def get_provider_instance(self, *, provider: str, model: str) -> Any:
        """Return the underlying AbstractCore provider instance for (provider, model)."""
        client = self._get_client(str(provider or ""), str(model or ""))
        return getattr(client, "_llm", None)

    def list_loaded_clients(self) -> List[Tuple[str, str]]:
        """Return (provider, model) pairs loaded in this process (best-effort)."""
        out = list(getattr(self, "_clients", {}).keys())
        for provider, model, _base_url, _api_key_fp in getattr(self, "_override_clients", {}).keys():
            pair = (provider, model)
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
        records: List[Dict[str, Any]] = []
        for provider_s, model_s in self.list_loaded_clients():
            if provider_filter and provider_s != provider_filter:
                continue
            if model_filter and model_s != model_filter:
                continue
            cached_client = self._clients.get((provider_s, model_s))
            records.append(
                _local_residency_record(
                    provider=provider_s,
                    model=model_s,
                    default=(provider_s, model_s) == (self._default_provider, self._default_model),
                    provider_instance=getattr(cached_client, "_llm", None),
                )
            )
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
                    self._clients.pop(key, None)
                    before_record = _local_residency_record(
                        provider=provider_s,
                        model=model_s,
                        default=False,
                        runtime_cached=False,
                        include_provider_state=False,
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
        )
        provider_loaded_new = bool(before_record.get("loaded") is not True and record.get("loaded") is True)
        loaded_new = bool((runtime_cache_loaded_new or provider_loaded_new) and record.get("loaded") is True)
        if record.get("loaded") is not True:
            if runtime_cache_loaded_new and key != (self._default_provider, self._default_model):
                self._clients.pop(key, None)
                record = _local_residency_record(
                    provider=provider_s,
                    model=model_s,
                    default=False,
                    runtime_cached=False,
                    include_provider_state=False,
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
        **kwargs: Any,
    ) -> Dict[str, Any]:
        _ = kwargs
        task_s = _normalize_residency_task(task)
        provider_s = str(provider or "").strip().lower()
        model_s = str(model or "").strip()
        if not provider_s or not model_s:
            raw_runtime_id = str(runtime_id or "").strip()
            prefix = "local:text_generation:"
            if raw_runtime_id.startswith(prefix):
                rest = raw_runtime_id[len(prefix) :]
                if ":" in rest:
                    provider_s, model_s = rest.split(":", 1)
                    provider_s = provider_s.strip().lower()
                    model_s = model_s.strip()
        if task_s != "text_generation":
            return _local_capability_residency_result(
                self,
                operation="unload",
                task=task_s,
                provider=provider_s,
                model=model_s,
                options=options,
                runtime_id=runtime_id,
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

        record_after_unload = _local_residency_record(
            provider=provider_s,
            model=model_s,
            default=default_key,
            runtime_cached=runtime_cached_before,
            provider_instance=provider_instance,
            include_provider_state=provider_instance is not None,
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
            error = "model_residency unload completed but provider still reports the model loaded"
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

        client = self._get_client(provider_str, model_str, llm_kwargs_override=llm_kwargs_override or None)
        return client.generate(
            prompt=prompt,
            messages=messages,
            system_prompt=system_prompt,
            tools=tools,
            media=media,
            params=params,
        )

    def get_model_capabilities(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        # Best-effort: use requested model name or the default client model.
        return self._default_client.get_model_capabilities(model_name=model_name)

    def lookup_model_capabilities(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        return self._default_client.lookup_model_capabilities(model_name=model_name)

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
        return self._default_client.list_provider_models(provider_name, **kwargs)

    def list_embedding_models(
        self,
        *,
        base_url: Optional[str] = None,
        provider_api_key: Optional[str] = None,
        provider: Optional[str] = None,
        providers_only: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._default_client.list_embedding_models(
            base_url=base_url,
            provider_api_key=provider_api_key,
            provider=provider,
            providers_only=providers_only,
            **kwargs,
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
        return self._default_client.get_voice_catalog(
            base_url=base_url,
            provider_api_key=provider_api_key,
            provider=provider,
            model=model,
            providers_only=providers_only,
            **kwargs,
        )

    def list_tts_models(
        self,
        *,
        base_url: Optional[str] = None,
        provider_api_key: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._default_client.list_tts_models(
            base_url=base_url,
            provider_api_key=provider_api_key,
            provider=provider,
            **kwargs,
        )

    def list_stt_models(
        self,
        *,
        base_url: Optional[str] = None,
        provider_api_key: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._default_client.list_stt_models(
            base_url=base_url,
            provider_api_key=provider_api_key,
            provider=provider,
            **kwargs,
        )

    def list_music_providers(
        self,
        *,
        task: Optional[str] = None,
        base_url: Optional[str] = None,
        provider_api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._default_client.list_music_providers(
            task=task,
            base_url=base_url,
            provider_api_key=provider_api_key,
            **kwargs,
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
        return self._default_client.list_music_models(
            task=task,
            base_url=base_url,
            provider_api_key=provider_api_key,
            provider=provider,
            **kwargs,
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
        return self._default_client.list_vision_provider_models(
            task=task,
            base_url=base_url,
            provider_api_key=provider_api_key,
            provider=provider,
            providers_only=providers_only,
            **kwargs,
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
        return self._default_client.list_cached_vision_models(
            task=task,
            base_url=base_url,
            provider_api_key=provider_api_key,
            provider=provider,
            **kwargs,
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


class HttpxRequestSender:
    """Default request sender based on httpx (sync)."""

    def __init__(self):
        import httpx

        self._httpx = httpx

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
    ) -> HttpResponse:
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
    ):
        from .constants import DEFAULT_LLM_TIMEOUT_S

        self._server_base_url = _core_server_root_url(server_base_url)
        self._model = model
        self._timeout_s = float(timeout_s) if timeout_s is not None else DEFAULT_LLM_TIMEOUT_S
        self._headers = dict(headers or {})
        self._sender = request_sender or HttpxRequestSender()
        self._artifact_store = artifact_store

    def default_prompt_cache_identity(self) -> Tuple[Optional[str], Optional[str]]:
        return "remote", self._model

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
        payload = self._discovery_get(
            "/models",
            source="abstractcore.remote",
            query={
                "provider": provider_text,
                "base_url": call_kwargs.get("base_url"),
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
            "supported": False,
            "operation": operation,
            "error": str(error),
            "warnings": [str(error)],
            "diagnostics": {"source": "abstractcore.remote"},
            "affected_models": [],
        }
        if status_code is not None:
            payload["status_code"] = status_code
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
        **kwargs: Any,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "task": _normalize_residency_task(task) if task is not None else None,
            "runtime_id": runtime_id,
            "provider": provider,
            "model": model,
            "options": options if isinstance(options, dict) else None,
        }
        result = self._model_residency_post(
            "/acore/models/unload",
            operation="unload",
            body=body,
            kwargs=kwargs,
        )
        if isinstance(result, dict):
            result.setdefault("operation", "unload")
            result.setdefault("success", result.get("ok") is not False)
            if "affected_models" not in result:
                result["affected_models"] = [result["runtime"]] if isinstance(result.get("runtime"), dict) else []
        return result

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

    def _post_bytes(self, url: str, *, headers: Dict[str, str], json_body: Dict[str, Any]) -> tuple[bytes, Dict[str, str]]:
        sender = self._sender
        post_bytes = getattr(sender, "post_bytes", None)
        if callable(post_bytes):
            raw = post_bytes(url, headers=headers, json=json_body, timeout=self._timeout_s)
        else:
            raw = sender.post(url, headers=headers, json=json_body, timeout=self._timeout_s)
        return _unwrap_binary_response(raw)

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
    ) -> tuple[Dict[str, Any], Dict[str, str]]:
        sender = self._sender
        post_multipart = getattr(sender, "post_multipart", None)
        if not callable(post_multipart):
            raise ValueError("Remote multipart media output requires a request sender with post_multipart().")
        raw = post_multipart(url, headers=headers, data=data, files=files, timeout=self._timeout_s)
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
                "negative_prompt",
            ):
                body.pop(local_only_key, None)
        base_url = params.get("base_url")
        if isinstance(base_url, str) and base_url.strip():
            body["base_url"] = base_url.strip()

        url = _join_core_v1_url(self._server_base_url, "/images/generations")
        raw = self._sender.post(url, headers=headers, json=body, timeout=self._timeout_s)
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
        for key in ("size", "negative_prompt", "seed", "steps", "guidance_scale"):
            if key in spec and spec.get(key) is not None:
                data[key] = spec.get(key)
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
        if extra:
            data["extra_json"] = json.dumps(extra, ensure_ascii=False, separators=(",", ":"))
        base_url = params.get("base_url")
        if isinstance(base_url, str) and base_url.strip():
            data["base_url"] = base_url.strip()

        files: Dict[str, tuple[str, bytes, str]] = {"image": _file_tuple("image", image_item)}
        if mask_item is not None:
            files["mask"] = _file_tuple("mask", mask_item)

        if endpoint_provider:
            data.pop("provider", None)
            url = _join_core_provider_v1_url(self._server_base_url, endpoint_provider, "/images/edits")
        else:
            url = _join_core_v1_url(self._server_base_url, "/images/edits")
        resp, _resp_headers = self._post_multipart_files(url, headers=headers, data=data, files=files)

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
        for key in ("n", "width", "height", "fps", "seed", "steps", "guidance_scale", "negative_prompt", "extra"):
            if key in spec and spec.get(key) is not None:
                body[key] = spec.get(key)
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
            raw = self._sender.post(url, headers=headers, json=body, timeout=self._timeout_s)
            start_resp, _resp_headers = _unwrap_http_response(raw)
            job_id = str(start_resp.get("job_id") or "").strip() if isinstance(start_resp, dict) else ""
            if not job_id:
                raise ValueError("Remote video generation job did not return job_id.")
            resp = self._poll_remote_vision_job(job_id=job_id, headers=headers, params=params)
        else:
            url = _join_core_v1_url(self._server_base_url, "/videos/generations")
            raw = self._sender.post(url, headers=headers, json=body, timeout=self._timeout_s)
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
        deadline = time.time() + max(1.0, float(self._timeout_s or 1.0))
        last_progress_key = ""
        while True:
            url = f"{_join_core_v1_url(self._server_base_url, f'/vision/jobs/{quote(job_id)}')}?consume=true"
            raw = self._sender.get(url, headers=headers, timeout=self._timeout_s)
            job, _resp_headers = _unwrap_http_response(raw)
            if not isinstance(job, dict):
                raise ValueError("Remote vision job poll returned a non-object response.")
            progress = job.get("progress")
            if callable(callback) and isinstance(progress, dict):
                progress_payload = dict(progress)
                progress_payload.setdefault("job_id", job_id)
                try:
                    progress_key = json.dumps(progress_payload, sort_keys=True, default=str)
                except Exception:
                    progress_key = str(progress_payload)
                if progress_key != last_progress_key:
                    callback(progress_payload)
                    last_progress_key = progress_key
            state = str(job.get("state") or "").strip().lower()
            if state == "succeeded":
                result = job.get("result")
                if not isinstance(result, dict):
                    raise ValueError("Remote vision job succeeded without a result object.")
                return result
            if state == "failed":
                raise ValueError(str(job.get("error") or "Remote vision job failed."))
            if time.time() >= deadline:
                raise TimeoutError(f"Timed out waiting for remote vision job {job_id}.")
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
        for key in ("width", "height", "fps", "seed", "steps", "guidance_scale", "negative_prompt"):
            if key in spec and spec.get(key) is not None:
                data[key] = spec.get(key)
        num_frames = spec.get("num_frames")
        if num_frames is None:
            num_frames = spec.get("frames")
        if num_frames is not None:
            data["num_frames"] = num_frames
        extra = dict(spec.get("extra")) if isinstance(spec.get("extra"), dict) else {}
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
        )

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
        body: Dict[str, Any] = {
            "prompt": str(prompt or ""),
            "task": str(spec.get("task") or "music_generation"),
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
            task="music_generation",
            modality="music",
            model=str(body.get("model") or "") or None,
        )
        return _normalize_multimodal_response(
            {
                "outputs": {
                    "music": [
                        {
                            "modality": "music",
                            "task": "music_generation",
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
            image_generation_tasks = {"", "image_generation", "t2i", "text_to_image"}
            if task in image_edit_tasks:
                if not text:
                    raise ValueError("Remote image edit requires prompt or text.")
                return self._remote_image_edit(spec=spec, prompt=text, media=media, headers=headers, params=params)
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
        prompt = _promote_text_param_to_prompt(prompt, params)
        provider_api_key = _pop_provider_api_key(params)
        req_headers = self._headers_with_provider_api_key(provider_api_key)
        effective_model = self._effective_model_from_params(params)

        trace_metadata = params.pop("trace_metadata", None)
        system_prompt = _strip_system_context_header(system_prompt)
        output_request = params.get("output")
        acore_output_request = _is_abstractcore_output_request(output_request)
        if _output_request_has_generated_media(output_request) and self._artifact_store is None:
            raise ValueError("Generated media outputs require an ArtifactStore.")
        skip_turn_grounding = _output_request_has_non_text_result(output_request)
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
            _sanitize_runtime_grounding_echoes(result)
            _attach_runtime_grounding(result, runtime_grounding)
            return result

        # Build OpenAI-like messages for AbstractCore server.
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
        raw = self._sender.post(url, headers=req_headers, json=body, timeout=self._timeout_s)
        resp, resp_headers = _unwrap_http_response(raw)
        lower_headers = {str(k).lower(): str(v) for k, v in resp_headers.items()}
        trace_id = lower_headers.get("x-abstractcore-trace-id") or lower_headers.get("x-trace-id")

        # Normalize OpenAI-like response.
        try:
            choice0 = (resp.get("choices") or [])[0]
            msg = choice0.get("message") or {}
            observable_body = _redact_data_urls_for_observability(body)
            meta: Dict[str, Any] = {
                "_provider_request": {"url": url, "payload": observable_body}
            }
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
