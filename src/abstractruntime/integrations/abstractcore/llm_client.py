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

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Protocol

from .logging import get_logger

logger = get_logger(__name__)


class RequestSender(Protocol):
    def post(
        self,
        url: str,
        *,
        headers: Dict[str, str],
        json: Dict[str, Any],
        timeout: float,
    ) -> Dict[str, Any]: ...


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


def _normalize_local_response(resp: Any) -> Dict[str, Any]:
    """Normalize an AbstractCore local `generate()` result into JSON."""

    # Dict-like already
    if isinstance(resp, dict):
        return _jsonable(resp)

    # Pydantic structured output
    if hasattr(resp, "model_dump") or hasattr(resp, "dict"):
        return {
            "content": None,
            "data": _jsonable(resp),
            "tool_calls": None,
            "usage": None,
            "model": None,
            "finish_reason": None,
        }

    # AbstractCore GenerateResponse
    content = getattr(resp, "content", None)
    tool_calls = getattr(resp, "tool_calls", None)
    usage = getattr(resp, "usage", None)
    model = getattr(resp, "model", None)
    finish_reason = getattr(resp, "finish_reason", None)

    return {
        "content": content,
        "data": None,
        "tool_calls": _jsonable(tool_calls) if tool_calls is not None else None,
        "usage": _jsonable(usage) if usage is not None else None,
        "model": model,
        "finish_reason": finish_reason,
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
        self._llm = create_llm(provider, model=model, **(llm_kwargs or {}))
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

        # If tools provided, use UniversalToolHandler to format them into prompt
        # This works for models without native tool support
        effective_prompt = prompt
        if tools:
            from abstractcore.tools import ToolDefinition
            tool_defs = []
            for t in tools:
                tool_defs.append(ToolDefinition(
                    name=t.get("name", ""),
                    description=t.get("description", ""),
                    parameters=t.get("parameters", {}),
                ))
            tools_prompt = self._tool_handler.format_tools_prompt(tool_defs)
            effective_prompt = f"{tools_prompt}\n\nUser request: {prompt}"

        resp = self._llm.generate(
            prompt=effective_prompt,
            messages=messages,
            system_prompt=system_prompt,
            stream=False,
            **params,
        )
        
        result = _normalize_local_response(resp)
        
        # Parse tool calls from response if tools were provided
        if tools and result.get("content"):
            parsed = self._tool_handler.parse_response(result["content"], mode="prompted")
            if parsed.tool_calls:
                result["tool_calls"] = [
                    {"name": tc.name, "arguments": tc.arguments, "call_id": tc.call_id}
                    for tc in parsed.tool_calls
                ]
        
        return result


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
    ) -> Dict[str, Any]:
        resp = self._httpx.post(url, headers=headers, json=json, timeout=timeout)
        resp.raise_for_status()
        return resp.json()


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
        resp = self._sender.post(url, headers=self._headers, json=body, timeout=self._timeout_s)

        # Normalize OpenAI-like response.
        try:
            choice0 = (resp.get("choices") or [])[0]
            msg = choice0.get("message") or {}
            return {
                "content": msg.get("content"),
                "data": None,
                "tool_calls": _jsonable(msg.get("tool_calls")) if msg.get("tool_calls") is not None else None,
                "usage": _jsonable(resp.get("usage")) if resp.get("usage") is not None else None,
                "model": resp.get("model"),
                "finish_reason": choice0.get("finish_reason"),
            }
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
            }

