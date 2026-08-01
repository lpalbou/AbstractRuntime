"""abstractruntime.integrations.abstractcore.tool_executor

Tool execution adapters.

- `AbstractCoreToolExecutor`: executes tool calls in-process using AbstractCore's
  global tool registry.
- `PassthroughToolExecutor`: does not execute; returns tool calls to the host.

The runtime can use passthrough mode for untrusted environments (server/edge) and
pause until the host resumes with the tool results.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import inspect
import json
import re
import threading
import uuid
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Set

from .logging import get_logger

logger = get_logger(__name__)


# Backlog 0214 — parallel read-only tool execution.
# A batch may declare several INDEPENDENT read-only calls (the ReAct prompt tells the model to
# batch reads/searches). We may execute those concurrently, but ONLY tools that are known to be
# side-effect-free may run in parallel; anything else — side-effecting tools, MCP/remote tools, or
# any name we do not recognize — runs strictly sequentially (fail-safe: never parallelize an effect
# we cannot prove is read-only, and never reorder a side effect relative to its neighbors).
_PARALLEL_SAFE_TOOL_NAMES: frozenset[str] = frozenset(
    {
        # Local filesystem reads / search / static analysis (no writes).
        "list_files",
        "skim_folders",
        "search_files",
        "analyze_code",
        "skim_files",
        "read_file",
        "open_attachment",
        # Web reads (network I/O, no local side effects).
        "skim_websearch",
        "skim_url",
        "web_search",
        "fetch_url",
    }
)

# Max concurrent read-only tool invocations within one batch. Reads are I/O-bound (file/network),
# so a small pool captures most of the latency win without oversubscribing threads.
_PARALLEL_TOOL_MAX_WORKERS = 8


def _is_parallel_safe_tool(name: str) -> bool:
    return str(name or "").strip() in _PARALLEL_SAFE_TOOL_NAMES


def _endpoint_profile_route_context(getter: Optional[Callable[[], Any]]):
    """Context manager installing the host's endpoint-profile resolver for
    the duration of one tool batch (Case-1 seam, converged 2026-07-26).

    The resolver lets core's session-route path construct gateway-registered
    `endpoint:*` providers that are invisible to ~/.abstractcore config
    (per-principal profiles). Degradations are structural no-ops: no getter,
    getter returns None, or an abstractcore too old to ship the seam — all
    yield a null context and behavior is byte-identical to pre-seam.
    """
    import contextlib

    resolver = None
    if callable(getter):
        try:
            resolver = getter()
        except Exception:
            resolver = None
    if resolver is None:
        return contextlib.nullcontext()
    try:
        from abstractcore.providers import use_provider_endpoint_profile_resolver
    except Exception:
        return contextlib.nullcontext()
    return use_provider_endpoint_profile_resolver(resolver)


def attach_endpoint_profile_resolver_getter(executor: Any, getter: Optional[Callable[[], Any]]) -> bool:
    """Attach a resolver getter to an executor, walking delegate chains.

    Hosts compose executors in layers (ApprovalToolExecutor -> Mapping,
    delegating MCP views, pre-approved views); the wrap lives on the inner
    MappingToolExecutor, so the attach must reach it through whatever
    wrapper the host built. Returns True when at least one executor in the
    chain accepted the getter (a False return means the composition has no
    in-process executor — attach elsewhere or the seam stays dark).
    """
    attached = False
    seen: set[int] = set()
    stack = [executor]
    while stack:
        obj = stack.pop()
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        setter = getattr(obj, "set_endpoint_profile_resolver_getter", None)
        if callable(setter):
            try:
                setter(getter)
                attached = True
            except Exception:
                pass
        for attr in ("_delegate", "_inner", "_executor", "_fallback"):
            child = getattr(obj, attr, None)
            if child is not None:
                stack.append(child)
    return attached


class ToolExecutor(Protocol):
    def execute(self, *, tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]: ...


def _normalize_timeout_s(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
    except Exception:
        return None
    # Contract: non-positive values are treated as "unlimited".
    return None if f <= 0 else f


def _call_with_timeout(func: Callable[[], Any], *, timeout_s: Optional[float]) -> tuple[bool, Any, Optional[str]]:
    """Execute a callable with a best-effort timeout.

    Important limitation (Python semantics): we cannot forcibly stop a running function
    without process isolation. On timeout we return an error, but the underlying callable
    may still finish later (daemon thread).
    """
    timeout_s = _normalize_timeout_s(timeout_s)
    if timeout_s is None:
        try:
            return True, func(), None
        except Exception as e:
            return False, None, str(e)

    result: Dict[str, Any] = {"done": False, "ok": False, "value": None, "error": None}

    # A bare Thread starts with an EMPTY contextvars context on CPython 3.12
    # — the ambient endpoint-profile resolver (and any other ContextVar) was
    # silently dropped on every lane that configures a tool timeout (route-
    # context adversary P1-1, 2026-07-26: gateway's local tool mode sets a
    # 7200s timeout on the bare executor, killing the seam). Capture the
    # calling thread's context and run the tool under it.
    import contextvars

    ctx = contextvars.copy_context()

    def _runner() -> None:
        try:
            result["value"] = ctx.run(func)
            result["ok"] = True
        except Exception as e:
            result["error"] = str(e)
            result["ok"] = False
        finally:
            result["done"] = True

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join(timeout_s)

    if not result.get("done", False):
        return False, None, f"Tool execution timed out after {timeout_s}s"
    if result.get("ok", False):
        return True, result.get("value"), None
    return False, None, str(result.get("error") or "Tool execution failed")


class MappingToolExecutor:
    """Executes tool calls using an explicit {tool_name -> callable} mapping.

    This is the recommended durable execution path: the mapping is held by the
    host/runtime process and is never persisted inside RunState.
    """

    def __init__(self, tool_map: Dict[str, Callable[..., Any]], *, timeout_s: Optional[float] = None):
        self._tool_map = dict(tool_map)
        self._timeout_s = _normalize_timeout_s(timeout_s)
        # Endpoint-profile resolver getter (Case-1 seam, 2026-07-26): a
        # LATE-BOUND callable returning the per-principal resolver the host
        # installed on this runtime's llm_client (gateway sets it per built
        # runtime — identity rides the closure, so this executor never sees
        # principals). Late-bound because the host may set the resolver
        # AFTER construction (def-time capture would freeze None forever).
        self._endpoint_profile_resolver_getter: Optional[Callable[[], Any]] = None

    def set_endpoint_profile_resolver_getter(self, getter: Optional[Callable[[], Any]]) -> None:
        self._endpoint_profile_resolver_getter = getter if callable(getter) else None

    @classmethod
    def from_tools(cls, tools: Sequence[Callable[..., Any]], *, timeout_s: Optional[float] = None) -> "MappingToolExecutor":
        tool_map: Dict[str, Callable[..., Any]] = {}
        for t in tools:
            tool_def = getattr(t, "_tool_definition", None)
            if tool_def is not None:
                name = str(getattr(tool_def, "name", "") or "")
                func = getattr(tool_def, "function", None) or t
            else:
                name = str(getattr(t, "__name__", "") or "")
                func = t

            if not name:
                raise ValueError("Tool is missing a name")
            if not callable(func):
                raise ValueError(f"Tool '{name}' is not callable")
            if name in tool_map:
                raise ValueError(f"Duplicate tool name '{name}'")

            tool_map[name] = func

        return cls(tool_map, timeout_s=timeout_s)

    def set_timeout_s(self, timeout_s: Optional[float]) -> None:
        self._timeout_s = _normalize_timeout_s(timeout_s)

    def execute(self, *, tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        # ONE wrap site covers every execution path by construction: the
        # policy path, approval-resume (ApprovalToolExecutor.execute_approved
        # delegates here), and tool_invoke's pre-approved view all funnel
        # into this method. Core's context manager is the ONLY install path
        # for the resolver (identity-by-closure is structural, c5759); an
        # absent resolver or an older core without the seam degrades to a
        # no-op context — byte-identical behavior.
        with _endpoint_profile_route_context(self._endpoint_profile_resolver_getter):
            return self._execute_inner(tool_calls=tool_calls)

    def _execute_inner(self, *, tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        results: List[Dict[str, Any]] = []

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

        def _unwrap_wrapper_args(kwargs: Dict[str, Any]) -> Dict[str, Any]:
            """Unwrap common wrapper shapes like {"name":..., "arguments":{...}}.

            Some models emit tool kwargs wrapped inside an "arguments" object and may
            mistakenly place real kwargs alongside wrapper fields. We unwrap and merge
            (inner args take precedence).
            """
            current: Dict[str, Any] = dict(kwargs or {})
            wrapper_keys = {"name", "arguments", "call_id", "id"}
            for _ in range(4):
                inner = current.get("arguments")
                inner_dict = _loads_dict_like(inner)
                if not isinstance(inner_dict, dict):
                    break
                extras = {k: v for k, v in current.items() if k not in wrapper_keys}
                merged = dict(inner_dict)
                for k, v in extras.items():
                    merged.setdefault(k, v)
                current = merged
            return current

        def _filter_kwargs(func: Callable[..., Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
            """Best-effort filtering of unexpected kwargs for callables without **kwargs."""
            try:
                sig = inspect.signature(func)
            except Exception:
                return kwargs

            params = list(sig.parameters.values())
            if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
                return kwargs

            allowed = {
                p.name
                for p in params
                if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            }
            return {k: v for k, v in kwargs.items() if k in allowed}

        def _normalize_key(key: str) -> str:
            # Lowercase and remove common separators so `file_path`, `filePath`,
            # `file-path`, `file path` all normalize to the same token.
            return re.sub(r"[\s_\-]+", "", str(key or "").strip().lower())

        _SYNONYM_ALIASES: Dict[str, List[str]] = {
            # Common semantic drift across many tools
            "path": ["file_path", "directory_path", "path"],
            # Common CLI/media naming drift
            "filename": ["file_path"],
            "filepath": ["file_path"],
            "dir": ["directory_path", "path"],
            "directory": ["directory_path", "path"],
            "folder": ["directory_path", "path"],
            "query": ["pattern", "query"],
            "regex": ["pattern", "regex"],
            # Range drift (used by multiple tools)
            "start": ["start_line", "start"],
            "end": ["end_line", "end"],
            "startlineoneindexed": ["start_line"],
            "endlineoneindexedinclusive": ["end_line"],
        }

        def _canonicalize_kwargs(func: Callable[..., Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
            """Best-effort canonicalization of kwarg names.

            Strategy:
            - Unwrap common wrapper shapes (nested `arguments`)
            - Map keys by normalized form (case + separators)
            - Apply a small, tool-agnostic synonym table (path/query/start/end)
            - Finally, filter unexpected kwargs for callables without **kwargs
            """
            if not isinstance(kwargs, dict) or not kwargs:
                return {}

            # 1) Unwrap wrapper shapes early.
            current = _unwrap_wrapper_args(kwargs)

            try:
                sig = inspect.signature(func)
            except Exception:
                return current

            params = list(sig.parameters.values())
            allowed_names = {
                p.name
                for p in params
                if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            }
            norm_to_param = { _normalize_key(n): n for n in allowed_names }

            out: Dict[str, Any] = dict(current)

            # 2) Normalized (morphological) key mapping.
            for k in list(out.keys()):
                if k in allowed_names:
                    continue
                nk = _normalize_key(k)
                target = norm_to_param.get(nk)
                if target and target not in out:
                    out[target] = out.pop(k)

            # 3) Synonym mapping (semantic).
            for k in list(out.keys()):
                if k in allowed_names:
                    continue
                nk = _normalize_key(k)
                candidates = _SYNONYM_ALIASES.get(nk, [])
                for cand in candidates:
                    if cand in allowed_names and cand not in out:
                        out[cand] = out.pop(k)
                        break

            # 4) Filter unexpected kwargs when callable doesn't accept **kwargs.
            return _filter_kwargs(func, out)

        def _error_from_output(value: Any) -> Optional[str]:
            """Detect tool failures reported as string outputs (instead of exceptions)."""
            # Structured tool outputs may explicitly report failure without raising.
            # Only treat as error when the tool declares failure.
            def _from_mapping(mapping: Dict[str, Any]) -> Optional[str]:
                success = mapping.get("success")
                ok = mapping.get("ok")
                status_hint = str(mapping.get("status_hint") or "").strip().lower()
                err = mapping.get("error") or mapping.get("message")
                if success is False or ok is False or status_hint == "error":
                    text = str(err or "Tool reported failure").strip()
                    return text or "Tool reported failure"
                if err not in {None, ""}:
                    text = str(err).strip()
                    return text or "Tool reported failure"
                return None

            if isinstance(value, dict):
                return _from_mapping(value)
            if not isinstance(value, str):
                return None
            text = value.strip()
            if not text:
                return None
            if text.startswith("{") and text.endswith("}"):
                try:
                    parsed = json.loads(text)
                except Exception:
                    parsed = None
                if isinstance(parsed, dict):
                    return _from_mapping(parsed)
            if text.startswith("Error:"):
                cleaned = text[len("Error:") :].strip()
                return cleaned or text
            if text.startswith(("❌", "🚫", "⏰")):
                cleaned = text.lstrip("❌🚫⏰").strip()
                if cleaned.startswith("Error:"):
                    cleaned = cleaned[len("Error:") :].strip()
                return cleaned or text
            return None

        def _make_result(*, call_id: str, runtime_call_id: Optional[str], name: str, output: Any) -> Dict[str, Any]:
            error = _error_from_output(output)
            if error is not None:
                # Preserve structured outputs for provenance/evidence. For string-only error outputs
                # (the historical convention), keep output empty and store the message in `error`.
                output_json = None if isinstance(output, str) else _jsonable(output)
                return {
                    "call_id": call_id,
                    "runtime_call_id": runtime_call_id,
                    "name": name,
                    "success": False,
                    "output": output_json,
                    "error": error,
                }

            return {
                "call_id": call_id,
                "runtime_call_id": runtime_call_id,
                "name": name,
                "success": True,
                "output": _jsonable(output),
                "error": None,
            }

        def _build_result(tc: Dict[str, Any]) -> Dict[str, Any]:
            """Execute a single tool call and return its result dict (never raises)."""
            if not isinstance(tc, dict):
                # Malformed batch entry (upstream always passes dicts, but honor the never-raises
                # contract so one bad entry cannot abort the whole batch on the sequential path).
                return {
                    "call_id": "",
                    "runtime_call_id": None,
                    "name": "",
                    "success": False,
                    "output": None,
                    "error": f"Invalid tool call entry (expected object, got {type(tc).__name__})",
                }
            name = str(tc.get("name", "") or "")
            raw_arguments = tc.get("arguments") or {}
            arguments = dict(raw_arguments) if isinstance(raw_arguments, dict) else (_loads_dict_like(raw_arguments) or {})
            call_id = str(tc.get("call_id") or "")
            runtime_call_id = tc.get("runtime_call_id")
            runtime_call_id_str = str(runtime_call_id).strip() if runtime_call_id is not None else ""
            runtime_call_id_out = runtime_call_id_str or None

            func = self._tool_map.get(name)
            if func is None:
                return {
                    "call_id": call_id,
                    "runtime_call_id": runtime_call_id_out,
                    "name": name,
                    "success": False,
                    "output": None,
                    "error": f"Tool '{name}' not found",
                }

            arguments = _canonicalize_kwargs(func, arguments)

            # Schema-aware type coercion (backlog 039): share the SAME coercion the AbstractCore
            # registry applies, so the runtime mapping-executor path and the registry path behave
            # identically. String flags like use_regex="false" / allow_dangerous="false" are coerced
            # to their declared types; an un-coercible typed value fails loudly (no silent default).
            try:
                from abstractcore.tools.arg_coercion import (
                    ArgumentCoercionError,
                    coerce_arguments_for_callable,
                )

                try:
                    arguments, _coercion_warnings = coerce_arguments_for_callable(func, arguments)
                except ArgumentCoercionError as coercion_error:
                    return {
                        "call_id": call_id,
                        "runtime_call_id": runtime_call_id_out,
                        "name": name,
                        "success": False,
                        "output": None,
                        "error": f"Invalid argument type for tool '{name}': {coercion_error}",
                    }
                for _warning in _coercion_warnings:
                    # StructuredLogger.warning(message, **kwargs) does not support
                    # %-style lazy args; format eagerly.
                    logger.warning(f"{_warning} (tool={name})")
            except ImportError:
                # AbstractCore coercion unavailable; proceed with canonicalized args (backstop:
                # high-risk tools keep their own per-tool coercion).
                pass

            def _invoke() -> Any:
                try:
                    return func(**arguments)
                except TypeError:
                    # Retry once with sanitized kwargs for common wrapper/extra-arg failures.
                    filtered = _canonicalize_kwargs(func, arguments)
                    if filtered != arguments:
                        return func(**filtered)
                    raise

            ok, output, err = _call_with_timeout(_invoke, timeout_s=self._timeout_s)
            if ok:
                return _make_result(call_id=call_id, runtime_call_id=runtime_call_id_out, name=name, output=output)
            return {
                "call_id": call_id,
                "runtime_call_id": runtime_call_id_out,
                "name": name,
                "success": False,
                "output": None,
                "error": str(err or "Tool execution failed"),
            }

        # Execution ordering (backlog 0214): walk the batch in order, grouping CONSECUTIVE
        # read-only calls into a parallel batch and running everything else strictly sequentially.
        # This preserves the exact observable ordering of side effects (a side-effecting call runs
        # after all reads before it and before all reads after it) while collapsing the latency of
        # independent read batches from sum(latency) to ~max(latency). Results are placed by original
        # index so the returned order is identical to the serial path.
        n = len(tool_calls)
        results = [None] * n  # type: ignore[assignment]
        i = 0
        while i < n:
            name_i = str((tool_calls[i] or {}).get("name", "") or "")
            if _is_parallel_safe_tool(name_i):
                # Extend the group over consecutive parallel-safe calls.
                j = i
                group: List[tuple[int, Dict[str, Any]]] = []
                while j < n and _is_parallel_safe_tool(str((tool_calls[j] or {}).get("name", "") or "")):
                    group.append((j, tool_calls[j]))
                    j += 1
                if len(group) == 1:
                    idx, tc = group[0]
                    results[idx] = _build_result(tc)
                else:
                    import contextvars
                    from concurrent.futures import ThreadPoolExecutor

                    workers = min(len(group), _PARALLEL_TOOL_MAX_WORKERS)
                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        # copy_context per submission: ContextVars (the
                        # endpoint-profile resolver context, trace state)
                        # do not cross thread boundaries on their own —
                        # a parallel-safe tool must see the same ambient
                        # context the sequential path sees (core's
                        # same-thread rule, c5783).
                        futures = {
                            pool.submit(contextvars.copy_context().run, _build_result, tc): idx
                            for idx, tc in group
                        }
                        for fut, idx in futures.items():
                            try:
                                results[idx] = fut.result()
                            except Exception as e:  # pragma: no cover - _build_result never raises
                                tc = tool_calls[idx]
                                results[idx] = {
                                    "call_id": str((tc or {}).get("call_id") or ""),
                                    "runtime_call_id": None,
                                    "name": str((tc or {}).get("name", "") or ""),
                                    "success": False,
                                    "output": None,
                                    "error": f"Tool execution failed: {e}",
                                }
                i = j
            else:
                # Side-effecting / unknown / MCP: run alone, in order.
                results[i] = _build_result(tool_calls[i])
                i += 1

        return {"mode": "executed", "results": list(results)}


def _jsonable(value: Any) -> Any:
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

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _jsonable(model_dump())

    to_dict = getattr(value, "dict", None)
    if callable(to_dict):
        return _jsonable(to_dict())

    return str(value)


class AbstractCoreToolExecutor:
    """Executes tool calls using AbstractCore's global tool registry."""

    def __init__(self, *, timeout_s: Optional[float] = None):
        self._timeout_s = _normalize_timeout_s(timeout_s)

    def set_timeout_s(self, timeout_s: Optional[float]) -> None:
        self._timeout_s = _normalize_timeout_s(timeout_s)

    def execute(self, *, tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        from abstractcore.tools.core import ToolCall
        from abstractcore.tools.registry import execute_tool

        calls: list[ToolCall] = []
        runtime_call_ids: list[Optional[str]] = []
        for tc in tool_calls:
            calls.append(
                ToolCall(
                    name=str(tc.get("name")),
                    arguments=dict(tc.get("arguments") or {}),
                    call_id=tc.get("call_id"),
                )
            )
            runtime_call_id = tc.get("runtime_call_id")
            runtime_call_id_str = str(runtime_call_id).strip() if runtime_call_id is not None else ""
            runtime_call_ids.append(runtime_call_id_str or None)

        normalized = []
        for call, runtime_call_id in zip(calls, runtime_call_ids):
            ok, out, err = _call_with_timeout(lambda c=call: execute_tool(c), timeout_s=self._timeout_s)
            if ok:
                r = out
                normalized.append(
                    {
                        "call_id": getattr(r, "call_id", "") if r is not None else "",
                        "runtime_call_id": runtime_call_id,
                        "name": getattr(call, "name", ""),
                        "success": bool(getattr(r, "success", False)) if r is not None else True,
                        "output": _jsonable(getattr(r, "output", None)) if r is not None else None,
                        "error": getattr(r, "error", None) if r is not None else None,
                    }
                )
                continue

            normalized.append(
                {
                    "call_id": str(getattr(call, "call_id", "") or ""),
                    "runtime_call_id": runtime_call_id,
                    "name": getattr(call, "name", ""),
                    "success": False,
                    "output": None,
                    "error": str(err or "Tool execution failed"),
                }
            )

        return {"mode": "executed", "results": normalized}


class PassthroughToolExecutor:
    """Returns tool calls unchanged without executing them."""

    def __init__(self, *, mode: str = "passthrough"):
        self._mode = mode

    def execute(self, *, tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"mode": self._mode, "tool_calls": _jsonable(tool_calls)}


def _mcp_result_to_output(result: Any) -> Any:
    if not isinstance(result, dict):
        return _jsonable(result)

    content = result.get("content")
    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "text":
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
        if texts:
            joined = "\n".join(texts).strip()
            if joined:
                try:
                    return _jsonable(json.loads(joined))
                except Exception:
                    return joined

    return _jsonable(result)


def _mcp_result_to_error(result: Any) -> Optional[str]:
    if not isinstance(result, dict):
        return None
    output = _mcp_result_to_output(result)

    # MCP-native error flag.
    if result.get("isError") is True:
        if isinstance(output, str) and output.strip():
            return output.strip()
        return "MCP tool call reported error"

    # Some real MCP servers return error strings inside content while leaving `isError=false`.
    # Match the local executor's convention for string error outputs.
    if isinstance(output, str):
        text = output.strip()
        if not text:
            return None
        if text.startswith("Error:"):
            cleaned = text[len("Error:") :].strip()
            return cleaned or text
        if text.startswith(("❌", "🚫", "⏰")):
            cleaned = text.lstrip("❌🚫⏰").strip()
            if cleaned.startswith("Error:"):
                cleaned = cleaned[len("Error:") :].strip()
            return cleaned or text
        if text.lower().startswith("traceback"):
            return text
    return None


class McpToolExecutor:
    """Executes tool calls remotely via an MCP server (Streamable HTTP / JSON-RPC)."""

    def __init__(
        self,
        *,
        server_id: str,
        mcp_url: str,
        timeout_s: Optional[float] = 30.0,
        mcp_client: Optional[Any] = None,
    ):
        self._server_id = str(server_id or "").strip()
        if not self._server_id:
            raise ValueError("McpToolExecutor requires a non-empty server_id")
        self._mcp_url = str(mcp_url or "").strip()
        if not self._mcp_url:
            raise ValueError("McpToolExecutor requires a non-empty mcp_url")
        self._timeout_s = _normalize_timeout_s(timeout_s)
        self._mcp_client = mcp_client

    def execute(self, *, tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        from abstractcore.mcp import McpClient, parse_namespaced_tool_name

        results: List[Dict[str, Any]] = []
        client = self._mcp_client or McpClient(url=self._mcp_url, timeout_s=self._timeout_s)
        close_client = self._mcp_client is None
        try:
            for tc in tool_calls:
                name = str(tc.get("name", "") or "")
                call_id = str(tc.get("call_id") or "")
                runtime_call_id = tc.get("runtime_call_id")
                runtime_call_id_str = str(runtime_call_id).strip() if runtime_call_id is not None else ""
                runtime_call_id_out = runtime_call_id_str or None
                raw_arguments = tc.get("arguments") or {}
                arguments = dict(raw_arguments) if isinstance(raw_arguments, dict) else {}

                remote_name = name
                parsed = parse_namespaced_tool_name(name)
                if parsed is not None:
                    server_id, tool_name = parsed
                    if server_id != self._server_id:
                        results.append(
                            {
                                "call_id": call_id,
                                "runtime_call_id": runtime_call_id_out,
                                "name": name,
                                "success": False,
                                "output": None,
                                "error": f"MCP tool '{name}' targets server '{server_id}', expected '{self._server_id}'",
                            }
                        )
                        continue
                    remote_name = tool_name

                try:
                    mcp_result = client.call_tool(name=remote_name, arguments=arguments)
                    err = _mcp_result_to_error(mcp_result)
                    if err is not None:
                        results.append(
                            {
                                "call_id": call_id,
                                "runtime_call_id": runtime_call_id_out,
                                "name": name,
                                "success": False,
                                "output": None,
                                "error": err,
                            }
                        )
                        continue
                    results.append(
                        {
                            "call_id": call_id,
                            "runtime_call_id": runtime_call_id_out,
                            "name": name,
                            "success": True,
                            "output": _mcp_result_to_output(mcp_result),
                            "error": None,
                        }
                    )
                except Exception as e:
                    results.append(
                        {
                            "call_id": call_id,
                            "runtime_call_id": runtime_call_id_out,
                            "name": name,
                            "success": False,
                            "output": None,
                            "error": str(e),
                        }
                    )

        finally:
            if close_client:
                try:
                    client.close()
                except Exception:
                    pass

        return {"mode": "executed", "results": results}


class DelegatingMcpToolExecutor:
    """Delegates tool calls to an MCP server by returning a durable JOB wait payload.

    This executor does not execute tools directly; it packages the tool calls plus
    MCP endpoint metadata into a `WAITING` state so an external worker can execute
    them and resume the run with results.
    """

    def __init__(
        self,
        *,
        server_id: str,
        mcp_url: str,
        transport: str = "streamable_http",
        wait_key_factory: Optional[Callable[[], str]] = None,
    ):
        self._server_id = str(server_id or "").strip()
        if not self._server_id:
            raise ValueError("DelegatingMcpToolExecutor requires a non-empty server_id")
        self._mcp_url = str(mcp_url or "").strip()
        if not self._mcp_url:
            raise ValueError("DelegatingMcpToolExecutor requires a non-empty mcp_url")
        self._transport = str(transport or "").strip() or "streamable_http"
        self._wait_key_factory = wait_key_factory or (lambda: f"mcp_job:{uuid.uuid4().hex}")

    def execute(self, *, tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "mode": "delegated",
            "wait_reason": "job",
            "wait_key": self._wait_key_factory(),
            "tool_calls": _jsonable(tool_calls),
            "details": {
                "protocol": "mcp",
                "transport": self._transport,
                "url": self._mcp_url,
                "server_id": self._server_id,
                "tool_name_prefix": f"mcp::{self._server_id}::",
            },
        }


_DEFAULT_SAFE_AUTO_APPROVE: Set[str] = {
    # Read-only filesystem
    "list_files",
    "skim_folders",
    "analyze_code",
    "read_file",
    "skim_files",
    "search_files",
    # Network read-only (fixed destinations: search engines / the model's
    # query text rides to a FIXED endpoint - egress accepted visibly per
    # laurent's tier-1 placement)
    "web_search",
    "skim_websearch",
    "skim_url",
    # fetch_url REMOVED from auto (core adversary P1, c4586): its
    # destination is MODEL-CHOSEN and core's row declares
    # remote_write_capable=True - auto-approving it made the exfiltration
    # chain silent (read a secret via observe-auto, POST it to a
    # model-chosen webhook via this entry, no outreach grant, no prompt).
    # The 2026-07-12 ruling stands: fetch_url is never read-only-safe.
    # Comms (required for bridge-owned delivery flows like Telegram)
    "send_telegram_message",
    "send_telegram_artifact",
    # Agora agent-to-agent hub (hub-scoped comms; reads are cursor-based, posts
    # go to invite-only channels/DMs on the configured hub)
    "agora_whoami",
    "agora_check_inbox",
    "agora_ack_inbox",
    "agora_read_channel",
    "agora_read_message",
    "agora_post_message",
    "agora_send_dm",
    # Agora channel shared-fs/store READS (c1669 step 3): hub-scoped, read-only.
    "channel_fs_read",
    "channel_fs_list",
    "channel_store_get",
}


_DEFAULT_REQUIRE_APPROVAL: Set[str] = {
    # Model-chosen network destination (core adversary P1, c4586): the
    # model picks URL+method+body - the approval prompt IS the
    # exfiltration defense the band-neutral model_controlled_destination
    # decision assumes exists.
    "fetch_url",
    # fetch_url's peer (operator dm#24, core c5005): renders a MODEL-CHOSEN
    # target in a headless browser AND executes the page's JS (which may
    # issue its own outbound requests) - mcd by declared fact, broker/ask
    # by derivation; this name entry is the belt beside the fact.
    "browser_probe",
    # Side effects
    "write_file",
    "edit_file",
    "execute_command",
    # Persistent shell sessions (backlog 0220): execute_command-level trust with state
    # persistence; opt-in via ABSTRACT_ENABLE_SHELL_TOOLS and still approval-gated per call.
    "shell_exec",
    "shell_write_stdin",
    "shell_close",
    # Comms with higher exfil/spam risk (explicit allow needed; unknown tools also require approval)
    "send_email",
    "send_whatsapp_message",
    # Agora channel shared-fs/store WRITES (c1669 step 3): a shared artifact or
    # decision write is a mutation every channel member sees — write-classed.
    "channel_fs_write",
    "channel_store_set",
}


class ToolApprovalPolicy:
    """Decide whether a batch of tool calls should require user approval.

    Contract:
    - Any tool name not in `auto_approve_tools` requires approval.
    - Any tool name in `require_approval_tools` requires approval (even if also in auto list).
    """

    def __init__(
        self,
        *,
        auto_approve_tools: Optional[Set[str]] = None,
        require_approval_tools: Optional[Set[str]] = None,
    ) -> None:
        # Important: callers may intentionally pass an empty set to disable auto-approval
        # (e.g., "approval required for all tools"). Treat None as "use defaults".
        if auto_approve_tools is None or require_approval_tools is None:
            # Owner-review wiring (c3835 review): defaults flow through the
            # toolset-aware fold point so ENABLED toolsets' approval facts
            # (camera today) ride every default-constructed policy. Lazy
            # import; on any failure the base constants stand (fail toward
            # the stricter base, never toward silence).
            try:
                from .default_tools import default_approval_policy_sets

                fold_auto, fold_req = default_approval_policy_sets()
            except Exception as exc:  # noqa: BLE001
                # LOUD degrade (camera adversary P2-1): the fold's actionable
                # error must not die silently — but never RAISE here:
                # bundle_host catches around this constructor and falls back
                # to the UNGATED MappingToolExecutor (a raise here = real
                # fail-open at the gateway). Base constants are the stricter
                # default (default-deny: unlisted names ask).
                # %s-format, never kwargs (gateway door-half P2, c3934): on
                # the stdlib-logger fallback a kwargs call raises TypeError
                # INSIDE this except block, escapes __init__, and engages
                # bundle_host's UNGATED executor fallback - the exact
                # fail-open this catch exists to prevent.
                try:
                    logger.warning(
                        "default approval fold failed; using base constants: %s", exc,
                    )
                except Exception:  # noqa: BLE001 - logging must never fail the gate
                    pass
                fold_auto, fold_req = set(_DEFAULT_SAFE_AUTO_APPROVE), set(_DEFAULT_REQUIRE_APPROVAL)
        else:
            fold_auto, fold_req = set(), set()
        auto = fold_auto if auto_approve_tools is None else set(auto_approve_tools)
        req = fold_req if require_approval_tools is None else set(require_approval_tools)
        self.auto_approve_tools = set(auto)
        self.require_approval_tools = set(req)

    def requires_approval(self, tool_calls: Sequence[Dict[str, object]]) -> bool:
        for tc in tool_calls or []:
            name = str((tc or {}).get("name") or "").strip()
            if not name:
                return True
            if name in self.require_approval_tools:
                return True
            if name not in self.auto_approve_tools:
                return True
        return False

    def describe(self) -> Dict[str, List[str]]:
        return {
            "auto_approve_tools": sorted(self.auto_approve_tools),
            "require_approval_tools": sorted(self.require_approval_tools),
        }


class ApprovalToolExecutor:
    """Execute tool calls with a safe auto-approve policy + durable approval waits.

    This executor wraps another executor (typically `MappingToolExecutor`) and:
    - executes safe tool batches immediately
    - returns `mode="approval_required"` for any batch that requires approval

    Note: `make_tool_calls_handler` detects "delegating" executors by probing `execute([])`.
    To ensure tool waits are handled via a durable wait/resume path, we return a non-executed
    mode for empty batches.
    """

    def __init__(
        self,
        *,
        delegate: ToolExecutor,
        policy: Optional[ToolApprovalPolicy] = None,
        wait_key_factory: Optional[Callable[[], str]] = None,
    ) -> None:
        self._delegate = delegate
        self._policy = policy or ToolApprovalPolicy()
        # NO DEFAULT FACTORY. The old default minted `tool_approval:{uuid4}`
        # per call: distinct, but re-randomised on every crash-replay of the
        # SAME approval, so the key a host handed out could never be
        # recomputed. Emitting nothing lets the runtime derive the durable key
        # (`build_tool_approval_wait_key`: run + node + the effect's own
        # idempotency identity) for this branch AND for the run-policy branch,
        # which never supplied one. A host that wants to own the key still
        # passes an explicit factory and still wins.
        self._wait_key_factory = wait_key_factory

    @property
    def policy(self) -> ToolApprovalPolicy:
        return self._policy

    def _approval_wait(self, calls: List[Dict[str, Any]], details: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "mode": "approval_required",
            "wait_reason": "user",
            "tool_calls": _jsonable(calls),
            "details": details,
        }
        factory = self._wait_key_factory
        if callable(factory):
            key = factory()
            if isinstance(key, str) and key.strip():
                out["wait_key"] = key.strip()
        return out

    def execute(self, *, tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        calls = list(tool_calls or [])
        if not calls:
            return self._approval_wait(calls, {"kind": "tool_approval"})

        try:
            requires = self._policy.requires_approval(calls)
        except Exception:
            requires = True

        if not requires:
            return self._delegate.execute(tool_calls=calls)

        return self._approval_wait(calls, {"kind": "tool_approval", "policy": self._policy.describe()})

    def execute_approved(self, *, tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute a previously-approved tool batch, bypassing the approval policy."""
        return self._delegate.execute(tool_calls=list(tool_calls or []))
