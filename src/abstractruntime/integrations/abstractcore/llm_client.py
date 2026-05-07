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
import re
import tempfile
import threading
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, Tuple
from urllib.parse import urlencode

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


def _inject_turn_grounding(
    *,
    prompt: str,
    messages: Optional[List[Dict[str, Any]]],
) -> tuple[str, Optional[List[Dict[str, Any]]]]:
    """Inject date/time/country into the *current user turn* (not the system prompt)."""
    header = _system_context_header()

    def _prefix_with_header(text: str) -> str:
        """Prefix with the current header, or rewrite a legacy `Grounding:` prefix into bracket form."""
        if not isinstance(text, str) or not text.strip():
            return header
        raw = str(text)
        first = raw.lstrip().splitlines()[0].strip()
        if _SYSTEM_CONTEXT_HEADER_RE.match(first):
            return raw
        legacy = _LEGACY_SYSTEM_CONTEXT_HEADER_PARSE_RE.match(first)
        if legacy:
            date_part, time_part, cc = legacy.group(1), legacy.group(2), legacy.group(3).upper()
            date_part = date_part.replace("/", "-")
            time_part = f"{time_part}:00" if len(time_part) == 5 else time_part
            bracket = f"[{date_part} {time_part} {cc}]"
            rest = "\n".join(raw.lstrip().splitlines()[1:]).lstrip()
            return f"{bracket} {rest}" if rest else bracket
        return f"{header} {raw}"

    def _prefix_content(content: Any) -> Any:
        if isinstance(content, str):
            return _prefix_with_header(content)
        if isinstance(content, list):
            items: List[Any] = [dict(item) if isinstance(item, dict) else item for item in content]
            for item in items:
                if not isinstance(item, dict):
                    continue
                if str(item.get("type") or "").strip().lower() != "text":
                    continue
                text_value = item.get("text")
                if isinstance(text_value, str):
                    item["text"] = _prefix_with_header(text_value)
                    return items
            return [{"type": "text", "text": header}, *items]
        return _prefix_with_header(str(content or ""))

    prompt_str = str(prompt or "")
    if prompt_str.strip():
        return _prefix_with_header(prompt_str), messages

    if isinstance(messages, list) and messages:
        out: List[Dict[str, Any]] = []
        for m in messages:
            out.append(dict(m) if isinstance(m, dict) else {"role": "user", "content": str(m)})

        for i in range(len(out) - 1, -1, -1):
            role = str(out[i].get("role") or "").strip().lower()
            if role != "user":
                continue
            content = out[i].get("content")
            out[i]["content"] = _prefix_content(content)
            return prompt_str, out

        # No user message found; append a synthetic user turn.
        out.append({"role": "user", "content": header})
        return prompt_str, out

    # No place to inject; best-effort no-op.
    return prompt_str, messages


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
            # Return a raw file path to trigger AutoMediaHandler processing.
            out.append(str(file_path))
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
    meta = store(
        bytes(data),
        content_type=str(content_type or "application/octet-stream"),
        run_id=str(run_id).strip() if isinstance(run_id, str) and run_id.strip() else None,
        tags=_string_tags(tags),
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
        elif modality in {"voice", "audio"}:
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
        "model": (text.get("model") if isinstance(text, dict) else None) or metadata.get("model"),
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
        self._generate_lock = _local_generate_lock(provider=self._provider, model=self._model)
        if self._generate_lock is not None:
            _warn_local_generate_lock_once(provider=self._provider, model=self._model)
        kwargs = dict(llm_kwargs or {})
        kwargs.setdefault("enable_tracing", True)
        if kwargs.get("enable_tracing"):
            # Keep a small in-memory ring buffer for exact request/response observability.
            # This enables hosts (AbstractCode/AbstractFlow) to inspect trace payloads by trace_id.
            kwargs.setdefault("max_traces", 50)
        self._llm = create_llm(provider, model=model, **kwargs)
        self._tool_handler = UniversalToolHandler(model)
        self._prompt_cache_state_lock = threading.Lock()
        self._prompt_cache_state: Dict[str, _PromptCacheSessionState] = {}

    def default_prompt_cache_identity(self) -> Tuple[Optional[str], Optional[str]]:
        return self._provider, self._model

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
            params = dict(params or {})
            prompt = _promote_text_param_to_prompt(prompt, params)
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
                for key in ("workflow_id", "node_id", "actor_id", "session_id", "parent_run_id"):
                    raw = trace_metadata.get(key)
                    if raw is not None and str(raw).strip():
                        default_artifact_tags[key] = str(raw)
            output_run_id, output_tags = _output_runtime_metadata(output_request)
            if run_id is None and output_run_id:
                run_id = output_run_id
            if output_tags:
                default_artifact_tags.update(output_tags)

            system_prompt = _strip_system_context_header(system_prompt)
            if not skip_turn_grounding:
                prompt, messages = _inject_turn_grounding(prompt=str(prompt or ""), messages=messages)
            else:
                prompt = str(prompt or "")
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

            lock = getattr(self, "_generate_lock", None)
            if lock is None:
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
                result["tool_calls"] = _normalize_tool_calls(result.get("tool_calls"))
            else:
                # Serialize generation for non-thread-safe providers (e.g. MLX).
                with lock:
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

    def get_model_capabilities(self) -> Dict[str, Any]:
        """Get model capabilities including max_tokens, vision_support, etc.

        Uses AbstractCore's architecture detection system to query model limits
        and features. This allows the runtime to be aware of model constraints
        for resource tracking and warnings.

        Returns:
            Dict with model capabilities. Always includes 'max_tokens' (default: DEFAULT_MAX_TOKENS).
        """
        try:
            from abstractcore.architectures.detection import get_model_capabilities
            return get_model_capabilities(self._model)
        except Exception:
            # Safe fallback if detection fails
            from abstractruntime.core.vars import DEFAULT_MAX_TOKENS

            return {"max_tokens": DEFAULT_MAX_TOKENS}

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
    ):
        self._llm_kwargs = dict(llm_kwargs or {})
        self._default_provider = provider.strip().lower()
        self._default_model = model.strip()
        self._artifact_store = artifact_store
        self._clients: Dict[Tuple[str, str], LocalAbstractCoreLLMClient] = {}
        self._default_client = self._get_client(self._default_provider, self._default_model)

        # Provide a stable underlying LLM for components that need one (e.g. summarizer).
        self._llm = getattr(self._default_client, "_llm", None)

    def default_prompt_cache_identity(self) -> Tuple[Optional[str], Optional[str]]:
        return self._default_provider, self._default_model

    def _get_client(self, provider: str, model: str) -> LocalAbstractCoreLLMClient:
        key = (provider.strip().lower(), model.strip())
        client = self._clients.get(key)
        if client is None:
            client = LocalAbstractCoreLLMClient(
                provider=key[0],
                model=key[1],
                llm_kwargs=self._llm_kwargs,
                artifact_store=self._artifact_store,
            )
            self._clients[key] = client
        return client

    def get_provider_instance(self, *, provider: str, model: str) -> Any:
        """Return the underlying AbstractCore provider instance for (provider, model)."""
        client = self._get_client(str(provider or ""), str(model or ""))
        return getattr(client, "_llm", None)

    def list_loaded_clients(self) -> List[Tuple[str, str]]:
        """Return (provider, model) pairs loaded in this process (best-effort)."""
        return list(self._clients.keys())

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

        client = self._get_client(provider_str, model_str)
        return client.generate(
            prompt=prompt,
            messages=messages,
            system_prompt=system_prompt,
            tools=tools,
            media=media,
            params=params,
        )

    def get_model_capabilities(self) -> Dict[str, Any]:
        # Best-effort: use default model capabilities. Per-model limits can be added later.
        return self._default_client.get_model_capabilities()

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

        self._server_base_url = server_base_url.rstrip("/")
        self._model = model
        self._timeout_s = float(timeout_s) if timeout_s is not None else DEFAULT_LLM_TIMEOUT_S
        self._headers = dict(headers or {})
        self._sender = request_sender or HttpxRequestSender()
        self._artifact_store = artifact_store

    def default_prompt_cache_identity(self) -> Tuple[Optional[str], Optional[str]]:
        return "remote", self._model

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

    def _prompt_cache_get(self, path: str, *, operation: str, kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        call_kwargs = dict(kwargs or {})
        provider_api_key = _pop_provider_api_key(call_kwargs)
        proxy_fields = self._prompt_cache_proxy_fields(call_kwargs)
        url = f"{self._server_base_url}{path}"
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
        url = f"{self._server_base_url}{path}"
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
            for key in ("workflow_id", "node_id", "actor_id", "session_id", "parent_run_id"):
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

    def _remote_image_generation(
        self,
        *,
        spec: Dict[str, Any],
        prompt: str,
        headers: Dict[str, str],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        endpoint_model = str(spec.get("model") or "").strip()
        body: Dict[str, Any] = {
            "prompt": prompt,
            "response_format": "b64_json",
        }
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
        base_url = params.get("base_url")
        if isinstance(base_url, str) and base_url.strip():
            body["base_url"] = base_url.strip()

        url = f"{self._server_base_url}/v1/images/generations"
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
        body: Dict[str, Any] = {
            "input": str(text or ""),
            "voice": spec.get("voice") or spec.get("voice_id") or "alloy",
            "response_format": fmt,
        }
        if endpoint_model:
            body["model"] = endpoint_model
        for key in ("speed", "instructions", "provider"):
            if key in spec and spec.get(key) is not None:
                body[key] = spec.get(key)
        base_url = params.get("base_url")
        if isinstance(base_url, str) and base_url.strip():
            body["base_url"] = base_url.strip()

        url = f"{self._server_base_url}/v1/audio/speech"
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
        data: Dict[str, Any] = {}
        if endpoint_model:
            data["model"] = endpoint_model
        for key in ("language", "prompt", "response_format", "temperature", "format"):
            if key in spec and spec.get(key) is not None:
                data[key] = spec.get(key)
        base_url = params.get("base_url")
        if isinstance(base_url, str) and base_url.strip():
            data["base_url"] = base_url.strip()
        url = f"{self._server_base_url}/v1/audio/transcriptions"
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
            if task and task not in {"image_generation", "t2i"}:
                raise ValueError("Remote image edits are not supported through this client yet; use local execution.")
            if media:
                raise ValueError("Remote image output does not accept input media yet; use local execution for image edits.")
            if not text:
                raise ValueError("Remote image generation requires prompt or text.")
            return self._remote_image_generation(spec=spec, prompt=text, headers=headers, params=params)

        if modality == "voice":
            if task in {"voice_clone", "clone"}:
                raise ValueError("Remote voice clone is not supported through this client yet; use local execution.")
            if media:
                raise ValueError("Remote voice output does not accept input audio media yet; use local execution for cloning or reference-guided TTS.")
            if not text:
                raise ValueError("Remote TTS requires prompt or text.")
            return self._remote_tts(spec=spec, text=text, headers=headers, params=params)

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
        params = dict(params or {})
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
        if not skip_turn_grounding:
            prompt, messages = _inject_turn_grounding(prompt=str(prompt or ""), messages=messages)
        else:
            prompt = str(prompt or "")
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
            return self._generate_remote_multimodal(
                prompt=str(prompt or ""),
                messages=messages,  # type: ignore[arg-type]
                media=media,
                params=params_for_mm,
                headers=req_headers,
            )

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

        url = f"{self._server_base_url}/v1/chat/completions"
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
                    "trace_id": trace_id,
                }
                if trace_id
                else {"_provider_request": {"url": url, "payload": _redact_data_urls_for_observability(body)}},
                "trace_id": trace_id,
                "raw_response": _jsonable(resp) if resp is not None else None,
            }
