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
            # Some providers place reasoning under metadata (e.g. LM Studio gpt-oss).
            if "reasoning" not in out and isinstance(meta, dict) and isinstance(meta.get("reasoning"), str):
                out["reasoning"] = meta.get("reasoning")
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
        kwargs = dict(llm_kwargs or {})
        kwargs.setdefault("enable_tracing", True)
        if kwargs.get("enable_tracing"):
            # Keep a small in-memory ring buffer for exact request/response observability.
            # This enables hosts (AbstractCode/AbstractFlow) to inspect trace payloads by trace_id.
            kwargs.setdefault("max_traces", 50)
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

        resp = self._llm.generate(
            prompt=str(prompt or ""),
            messages=messages,
            system_prompt=system_prompt,
            tools=tools,
            stream=stream,
            **params,
        )
        if stream and hasattr(resp, "__next__"):
            result = _normalize_local_streaming_response(resp)
        else:
            result = _normalize_local_response(resp)
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
                    if not (isinstance(last, dict) and last.get("role") == "user" and last.get("content") == prompt_str):
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
        # Runtime authority default: long-running workflow steps may legitimately take a long time.
        # Keep this aligned with AbstractRuntime's orchestration defaults.
        timeout_s: Optional[float] = None,
        headers: Optional[Dict[str, str]] = None,
        request_sender: Optional[RequestSender] = None,
    ):
        from .constants import DEFAULT_LLM_TIMEOUT_S

        self._server_base_url = server_base_url.rstrip("/")
        self._model = model
        self._timeout_s = float(timeout_s) if timeout_s is not None else DEFAULT_LLM_TIMEOUT_S
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
            # Orchestrator policy: ask AbstractCore server to use the same timeout it expects.
            # This keeps runtime authority even when the actual provider call happens server-side.
            "timeout_s": self._timeout_s,
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
            meta: Dict[str, Any] = {
                "_provider_request": {"url": url, "payload": body}
            }
            if trace_id:
                meta["trace_id"] = trace_id
            result = {
                "content": msg.get("content"),
                "reasoning": msg.get("reasoning"),
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
                    "_provider_request": {"url": url, "payload": body},
                    "trace_id": trace_id,
                }
                if trace_id
                else {"_provider_request": {"url": url, "payload": body}},
                "trace_id": trace_id,
                "raw_response": _jsonable(resp) if resp is not None else None,
            }
