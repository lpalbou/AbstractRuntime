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
import json
import re
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple

from .logging import get_logger

logger = get_logger(__name__)


def _extract_prompted_tool_calls(
    *,
    content: Optional[str],
    tool_handler: Any,
    allowed_tool_names: Optional[set[str]] = None,
) -> tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """Extract tool calls from *content* using AbstractCore's UniversalToolHandler.

    This keeps tool-call parsing/cleaning logic centralized in AbstractCore.

    Returns:
        (tool_calls, cleaned_content)
        - tool_calls: canonical JSON-safe list when detected, else None
        - cleaned_content: content with tool-call markup stripped, else None
    """
    if not isinstance(content, str) or not content.strip():
        return None, None
    if tool_handler is None:
        return None, None

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


def _maybe_parse_tool_calls_from_text(
    *,
    content: Optional[str],
    allowed_tool_names: Optional[set[str]] = None,
    model_name: Optional[str] = None,
    tool_handler: Any = None,
) -> tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """Best-effort tool-call extraction from assistant text.

    This is a durability/robustness fallback used when a provider returns no structured
    tool calls but tool syntax is present in the assistant `content`.

    Parsing/cleaning is delegated to AbstractCore's UniversalToolHandler so the runtime
    does not grow its own parsing stack.
    """
    if tool_handler is None:
        from abstractcore.tools.handler import UniversalToolHandler

        tool_handler = UniversalToolHandler(str(model_name or ""))
    return _extract_prompted_tool_calls(
        content=content,
        tool_handler=tool_handler,
        allowed_tool_names=allowed_tool_names,
    )


@dataclass(frozen=True)
class HttpResponse:
    body: Dict[str, Any]
    headers: Dict[str, str]


class RequestSender(Protocol):
    def post(
        self,
        url: str,
        *,
        headers: Dict[str, str],
        json: Dict[str, Any],
        timeout: float,
    ) -> Any: ...


class AbstractCoreLLMClient(Protocol):
    def generate(
        self,
        *,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return a JSON-safe dict with at least: content/tool_calls/usage/model."""


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


def _normalize_local_response(resp: Any) -> Dict[str, Any]:
    """Normalize an AbstractCore local `generate()` result into JSON."""

    # Dict-like already
    if isinstance(resp, dict):
        out = _jsonable(resp)
        if isinstance(out, dict):
            meta = out.get("metadata")
            if isinstance(meta, dict) and "trace_id" in meta and "trace_id" not in out:
                out["trace_id"] = meta["trace_id"]
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
    tool_calls = getattr(resp, "tool_calls", None)
    usage = getattr(resp, "usage", None)
    model = getattr(resp, "model", None)
    finish_reason = getattr(resp, "finish_reason", None)
    metadata = getattr(resp, "metadata", None)
    trace_id: Optional[str] = None
    if isinstance(metadata, dict):
        raw = metadata.get("trace_id")
        if raw is not None:
            trace_id = str(raw)

    return {
        "content": content,
        "data": None,
        "tool_calls": _jsonable(tool_calls) if tool_calls is not None else None,
        "usage": _jsonable(usage) if usage is not None else None,
        "model": model,
        "finish_reason": finish_reason,
        "metadata": _jsonable(metadata) if metadata is not None else None,
        "trace_id": trace_id,
    }


class LocalAbstractCoreLLMClient:
    """In-process LLM client using AbstractCore's provider stack."""

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        llm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        from abstractcore import create_llm
        from abstractcore.tools.handler import UniversalToolHandler

        self._provider = provider
        self._model = model
        kwargs = dict(llm_kwargs or {})
        kwargs.setdefault("enable_tracing", True)
        if kwargs.get("enable_tracing"):
            kwargs.setdefault("max_traces", 0)
        self._llm = create_llm(provider, model=model, **kwargs)
        self._tool_handler = UniversalToolHandler(model)

    def generate(
        self,
        *,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        params = dict(params or {})

        # `base_url` is a provider construction concern in local mode. We intentionally
        # do not create new providers per call unless the host explicitly chooses to.
        params.pop("base_url", None)
        # Reserved routing keys (used by MultiLocalAbstractCoreLLMClient).
        params.pop("_provider", None)
        params.pop("_model", None)

        capabilities: List[str] = []
        get_capabilities = getattr(self._llm, "get_capabilities", None)
        if callable(get_capabilities):
            try:
                capabilities = list(get_capabilities())
            except Exception:
                capabilities = []
        provider_supports_tools = "tools" in set(c.lower() for c in capabilities)

        # Decide tool-calling mode using AbstractCore's model capability detection.
        #
        # Many local providers (notably LMStudio) advertise "tools" at the provider level,
        # but the underlying OSS model may require *prompted* tool calling (tool calls in
        # content, not in API fields). AbstractCore tracks this via model capabilities.
        #
        # Rule:
        # - Use native tools only when BOTH:
        #   - provider supports passing tools, AND
        #   - model is marked as native tool-support
        # - Otherwise prefer prompted tool calling (more robust across OSS models).
        use_native_tools = bool(tools and provider_supports_tools and getattr(self._tool_handler, "supports_native", False))

        if tools and not use_native_tools:
            # Fallback tool calling via prompting for providers/models without native tool support.
            from abstractcore.tools import ToolDefinition

            tool_defs = [
                ToolDefinition(
                    name=t.get("name", ""),
                    description=t.get("description", ""),
                    parameters=t.get("parameters", {}),
                )
                for t in tools
            ]
            tools_prompt = self._tool_handler.format_tools_prompt(tool_defs)
            combined_system_prompt = (str(system_prompt or "")).strip()
            tools_prompt = (str(tools_prompt or "")).strip()
            if tools_prompt:
                combined_system_prompt = (
                    f"{combined_system_prompt}\n\n{tools_prompt}".strip() if combined_system_prompt else tools_prompt
                )

            resp = self._llm.generate(
                prompt=str(prompt or ""),
                messages=messages,
                system_prompt=combined_system_prompt or None,
                stream=False,
                **params,
            )
            result = _normalize_local_response(resp)

            allowed_names = {str(t.name) for t in tool_defs if getattr(t, "name", None)}
            tool_calls, cleaned = _maybe_parse_tool_calls_from_text(
                content=result.get("content") if isinstance(result.get("content"), str) else None,
                allowed_tool_names=allowed_names or None,
                model_name=self._model,
                tool_handler=self._tool_handler,
            )
            if tool_calls:
                result["tool_calls"] = tool_calls
                if cleaned is not None:
                    result["content"] = cleaned
            return result

        resp = self._llm.generate(
            prompt=str(prompt or ""),
            messages=messages,
            system_prompt=system_prompt,
            tools=tools,
            stream=False,
            **params,
        )
        result = _normalize_local_response(resp)
        result["tool_calls"] = _normalize_tool_calls(result.get("tool_calls"))

        # Some providers/models include tool-call transcript markup in `content` even when
        # structured tool_calls are present; strip it so history/UI stays clean.
        if tools and result.get("tool_calls"):
            content = result.get("content")
            if isinstance(content, str) and content.strip():
                try:
                    from abstractcore.tools.core import ToolCall as CoreToolCall
                    from abstractcore.tools.parser import clean_tool_syntax

                    core_calls: list[CoreToolCall] = []
                    for tc in result.get("tool_calls") or []:
                        if not isinstance(tc, dict):
                            continue
                        name = tc.get("name")
                        if not isinstance(name, str) or not name.strip():
                            continue
                        args = tc.get("arguments")
                        args_dict = dict(args) if isinstance(args, dict) else {}
                        core_calls.append(CoreToolCall(name=name.strip(), arguments=args_dict, call_id=tc.get("call_id")))
                    if core_calls:
                        result["content"] = clean_tool_syntax(content, core_calls)
                except Exception:
                    pass

        # Fallback: some providers return tool call tags in content without structured tool_calls.
        if tools and not result.get("tool_calls"):
            content = result.get("content")
            allowed_names: set[str] = set()
            for t in tools:
                if not isinstance(t, dict):
                    continue
                name = t.get("name")
                if isinstance(name, str) and name.strip():
                    allowed_names.add(name.strip())
                    continue
                func = t.get("function") if isinstance(t.get("function"), dict) else None
                fname = func.get("name") if isinstance(func, dict) else None
                if isinstance(fname, str) and fname.strip():
                    allowed_names.add(fname.strip())
            tool_calls, cleaned = _maybe_parse_tool_calls_from_text(
                content=content if isinstance(content, str) else None,
                allowed_tool_names=allowed_names or None,
                model_name=self._model,
                tool_handler=self._tool_handler,
            )
            if tool_calls:
                result["tool_calls"] = tool_calls
                if isinstance(cleaned, str):
                    result["content"] = cleaned

        return result

    def get_model_capabilities(self) -> Dict[str, Any]:
        """Get model capabilities including max_tokens, vision_support, etc.

        Uses AbstractCore's architecture detection system to query model limits
        and features. This allows the runtime to be aware of model constraints
        for resource tracking and warnings.

        Returns:
            Dict with model capabilities. Always includes 'max_tokens' (default 32768).
        """
        try:
            from abstractcore.architectures.detection import get_model_capabilities
            return get_model_capabilities(self._model)
        except Exception:
            # Safe fallback if detection fails
            return {"max_tokens": 32768}


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
    ):
        self._llm_kwargs = dict(llm_kwargs or {})
        self._default_provider = provider.strip().lower()
        self._default_model = model.strip()
        self._clients: Dict[Tuple[str, str], LocalAbstractCoreLLMClient] = {}
        self._default_client = self._get_client(self._default_provider, self._default_model)

        # Provide a stable underlying LLM for components that need one (e.g. summarizer).
        self._llm = getattr(self._default_client, "_llm", None)

    def _get_client(self, provider: str, model: str) -> LocalAbstractCoreLLMClient:
        key = (provider.strip().lower(), model.strip())
        client = self._clients.get(key)
        if client is None:
            client = LocalAbstractCoreLLMClient(provider=key[0], model=key[1], llm_kwargs=self._llm_kwargs)
            self._clients[key] = client
        return client

    def generate(
        self,
        *,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
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
            params=params,
        )

    def get_model_capabilities(self) -> Dict[str, Any]:
        # Best-effort: use default model capabilities. Per-model limits can be added later.
        return self._default_client.get_model_capabilities()


class HttpxRequestSender:
    """Default request sender based on httpx (sync)."""

    def __init__(self):
        import httpx

        self._httpx = httpx

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


class RemoteAbstractCoreLLMClient:
    """Remote LLM client calling an AbstractCore server endpoint."""

    def __init__(
        self,
        *,
        server_base_url: str,
        model: str,
        timeout_s: float = 60.0,
        headers: Optional[Dict[str, str]] = None,
        request_sender: Optional[RequestSender] = None,
    ):
        self._server_base_url = server_base_url.rstrip("/")
        self._model = model
        self._timeout_s = timeout_s
        self._headers = dict(headers or {})
        self._sender = request_sender or HttpxRequestSender()

    def generate(
        self,
        *,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        params = dict(params or {})
        req_headers = dict(self._headers)

        trace_metadata = params.pop("trace_metadata", None)
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

        # Build OpenAI-like messages for AbstractCore server.
        out_messages: List[Dict[str, str]] = []
        if system_prompt:
            out_messages.append({"role": "system", "content": system_prompt})

        if messages:
            out_messages.extend(messages)
        else:
            out_messages.append({"role": "user", "content": prompt})

        body: Dict[str, Any] = {
            "model": self._model,
            "messages": out_messages,
            "stream": False,
        }

        # Dynamic routing support (AbstractCore server feature).
        base_url = params.pop("base_url", None)
        if base_url:
            body["base_url"] = base_url

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
            result = {
                "content": msg.get("content"),
                "data": None,
                "tool_calls": _jsonable(msg.get("tool_calls")) if msg.get("tool_calls") is not None else None,
                "usage": _jsonable(resp.get("usage")) if resp.get("usage") is not None else None,
                "model": resp.get("model"),
                "finish_reason": choice0.get("finish_reason"),
                "metadata": {"trace_id": trace_id} if trace_id else None,
                "trace_id": trace_id,
            }
            result["tool_calls"] = _normalize_tool_calls(result.get("tool_calls"))

            # If tool calls are already structured, ensure any echoed tool markup is removed
            # from the assistant content so downstream history/UI stays clean.
            if tools and result.get("tool_calls"):
                content = result.get("content")
                if isinstance(content, str) and content.strip():
                    try:
                        from abstractcore.tools.core import ToolCall as CoreToolCall
                        from abstractcore.tools.parser import clean_tool_syntax

                        core_calls: list[CoreToolCall] = []
                        for tc in result.get("tool_calls") or []:
                            if not isinstance(tc, dict):
                                continue
                            name = tc.get("name")
                            if not isinstance(name, str) or not name.strip():
                                continue
                            args = tc.get("arguments")
                            args_dict = dict(args) if isinstance(args, dict) else {}
                            core_calls.append(
                                CoreToolCall(name=name.strip(), arguments=args_dict, call_id=tc.get("call_id"))
                            )
                        if core_calls:
                            result["content"] = clean_tool_syntax(content, core_calls)
                    except Exception:
                        pass

            if tools and not result.get("tool_calls"):
                allowed_names: set[str] = set()
                for t in tools:
                    if not isinstance(t, dict):
                        continue
                    name = t.get("name")
                    if isinstance(name, str) and name.strip():
                        allowed_names.add(name.strip())
                        continue
                    func = t.get("function") if isinstance(t.get("function"), dict) else None
                    fname = func.get("name") if isinstance(func, dict) else None
                    if isinstance(fname, str) and fname.strip():
                        allowed_names.add(fname.strip())

                tool_calls, cleaned = _maybe_parse_tool_calls_from_text(
                    content=result.get("content") if isinstance(result.get("content"), str) else None,
                    allowed_tool_names=allowed_names or None,
                    model_name=self._model,
                )
                if tool_calls:
                    result["tool_calls"] = tool_calls
                    if isinstance(cleaned, str):
                        result["content"] = cleaned

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
                "metadata": {"trace_id": trace_id} if trace_id else None,
                "trace_id": trace_id,
            }
