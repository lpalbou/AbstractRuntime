"""abstractruntime.integrations.abstractcore.effect_handlers

Effect handlers wiring for AbstractRuntime.

These handlers implement:
- `EffectType.LLM_CALL`
- `EffectType.TOOL_CALLS`

They are designed to keep `RunState.vars` JSON-safe.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Set, Tuple, Type

from ...core.models import Effect, EffectType, RunState, WaitReason, WaitState
from ...core.runtime import EffectOutcome, EffectHandler
from .llm_client import AbstractCoreLLMClient
from .tool_executor import ToolExecutor
from .logging import get_logger

logger = get_logger(__name__)

_JSON_SCHEMA_PRIMITIVE_TYPES: Set[str] = {"string", "integer", "number", "boolean", "array", "object", "null"}


def _jsonable(value: Any) -> Any:
    """Best-effort conversion to JSON-safe objects.

    Runtime traces and effect outcomes are persisted in RunState.vars and must remain JSON-safe.
    """
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)


def _normalize_response_schema(raw: Any) -> Optional[Dict[str, Any]]:
    """Normalize user-provided schema inputs into a JSON Schema dict (object root).

    Supported inputs (best-effort):
    - JSON Schema object:
        {"type":"object","properties":{...}, "required":[...], ...}
      (also accepts missing type when properties exist)
    - OpenAI/LMStudio wrapper shapes:
        {"type":"json_schema","json_schema":{"schema":{...}}}
        {"json_schema":{"schema":{...}}}
        {"schema":{...}}  (inner wrapper copy/pasted from provider docs)
    - "Field map" shortcut (common authoring mistake but unambiguous intent):
        {"choice":"...", "score": 0, "meta": {"foo":"bar"}, ...}
      Coerces into a JSON Schema object with properties inferred from values.
    """

    def _is_schema_type(value: Any) -> bool:
        if isinstance(value, str):
            return value in _JSON_SCHEMA_PRIMITIVE_TYPES
        if isinstance(value, list) and value and all(isinstance(x, str) for x in value):
            return all(x in _JSON_SCHEMA_PRIMITIVE_TYPES for x in value)
        return False

    def _looks_like_json_schema(obj: Dict[str, Any]) -> bool:
        if "$schema" in obj or "$id" in obj or "$ref" in obj or "$defs" in obj or "definitions" in obj:
            return True
        if "oneOf" in obj or "anyOf" in obj or "allOf" in obj:
            return True
        if "enum" in obj or "const" in obj:
            return True
        if "items" in obj:
            return True
        if "required" in obj and isinstance(obj.get("required"), list):
            return True
        props = obj.get("properties")
        if isinstance(props, dict):
            return True
        if "type" in obj and _is_schema_type(obj.get("type")):
            return True
        return False

    def _unwrap_wrapper(obj: Dict[str, Any]) -> Dict[str, Any]:
        current = dict(obj)

        # If someone pasted an enclosing request object, tolerate common keys.
        for wrapper_key in ("response_format", "responseFormat"):
            inner = current.get(wrapper_key)
            if isinstance(inner, dict):
                current = dict(inner)

        # OpenAI/LMStudio wrapper: {type:"json_schema", json_schema:{schema:{...}}}
        if current.get("type") == "json_schema" and isinstance(current.get("json_schema"), dict):
            inner = current.get("json_schema")
            if isinstance(inner, dict) and isinstance(inner.get("schema"), dict):
                return dict(inner.get("schema") or {})

        # Slightly-less-wrapped: {json_schema:{schema:{...}}}
        if isinstance(current.get("json_schema"), dict):
            inner = current.get("json_schema")
            if isinstance(inner, dict) and isinstance(inner.get("schema"), dict):
                return dict(inner.get("schema") or {})

        # Inner wrapper copy/paste: {schema:{...}} or {name,strict,schema:{...}}
        if "schema" in current and isinstance(current.get("schema"), dict) and not _looks_like_json_schema(current):
            return dict(current.get("schema") or {})

        return current

    def _infer_schema_from_value(value: Any) -> Dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, bool):
            return {"type": "boolean"}
        if isinstance(value, int) and not isinstance(value, bool):
            return {"type": "integer"}
        if isinstance(value, float):
            return {"type": "number"}
        if isinstance(value, str):
            return {"type": "string", "description": value}
        if isinstance(value, list):
            # Prefer a simple, safe array schema; do not attempt enum constraints here.
            item_schema: Dict[str, Any] = {}
            for item in value:
                if item is None:
                    continue
                if isinstance(item, bool):
                    item_schema = {"type": "boolean"}
                    break
                if isinstance(item, int) and not isinstance(item, bool):
                    item_schema = {"type": "integer"}
                    break
                if isinstance(item, float):
                    item_schema = {"type": "number"}
                    break
                if isinstance(item, str):
                    item_schema = {"type": "string"}
                    break
                if isinstance(item, dict):
                    item_schema = _coerce_object_schema(item)
                    break
            out: Dict[str, Any] = {"type": "array"}
            if item_schema:
                out["items"] = item_schema
            return out
        if isinstance(value, dict):
            # If it already looks like a schema, keep it as-is (with minor fixes).
            if _looks_like_json_schema(value):
                return _coerce_object_schema(value)
            # Otherwise treat nested dict as another field-map object.
            return _coerce_object_schema(value)
        return {"type": "string", "description": str(value)}

    def _coerce_object_schema(obj: Dict[str, Any]) -> Dict[str, Any]:
        # If it's already a JSON schema, normalize the minimal invariants we need.
        if _looks_like_json_schema(obj):
            out = dict(obj)
            props = out.get("properties")
            if isinstance(props, dict) and out.get("type") is None:
                out["type"] = "object"
            # Nothing else to do here; deeper normalization is handled by the pydantic conversion.
            return out

        # Otherwise, interpret as "properties map" (field → schema/description/example).
        properties: Dict[str, Any] = {}
        required: list[str] = []
        for k, v in obj.items():
            if not isinstance(k, str) or not k.strip():
                continue
            key = k.strip()
            required.append(key)
            properties[key] = _infer_schema_from_value(v)

        schema: Dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            schema["required"] = required
        return schema

    if raw is None:
        return None

    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            raw = parsed
        except Exception:
            # Keep raw string; caller will treat as absent/invalid.
            return None

    if not isinstance(raw, dict) or not raw:
        return None

    candidate = _unwrap_wrapper(raw)
    if not isinstance(candidate, dict) or not candidate:
        return None

    normalized = _coerce_object_schema(candidate)
    return normalized if isinstance(normalized, dict) and normalized else None


def _pydantic_model_from_json_schema(schema: Dict[str, Any], *, name: str) -> Type[Any]:
    """Best-effort conversion from a JSON schema dict to a Pydantic model.

    This exists so structured output requests can remain JSON-safe in durable
    effect payloads (we persist the schema, not the Python class).
    """
    try:
        from pydantic import BaseModel, create_model
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Pydantic is required for structured outputs: {e}")

    def _python_type(sub_schema: Any, *, nested_name: str) -> Any:
        if not isinstance(sub_schema, dict):
            return Any
        t = sub_schema.get("type")
        if t == "string":
            return str
        if t == "integer":
            return int
        if t == "number":
            return float
        if t == "boolean":
            return bool
        if t == "array":
            items = sub_schema.get("items")
            return list[_python_type(items, nested_name=f"{nested_name}Item")]  # type: ignore[index]
        if t == "object":
            props = sub_schema.get("properties")
            if isinstance(props, dict) and props:
                return _model(sub_schema, name=nested_name)
            return Dict[str, Any]
        return Any

    def _model(obj_schema: Dict[str, Any], *, name: str) -> Type[BaseModel]:
        schema_type = obj_schema.get("type")
        if schema_type is None and isinstance(obj_schema.get("properties"), dict):
            schema_type = "object"
        if isinstance(schema_type, list) and "object" in schema_type:
            schema_type = "object"
        if schema_type != "object":
            raise ValueError("response_schema must be a JSON schema object")
        props = obj_schema.get("properties")
        if not isinstance(props, dict) or not props:
            raise ValueError("response_schema must define properties")
        required_raw = obj_schema.get("required")
        required: Set[str] = set()
        if isinstance(required_raw, list):
            required = {str(x) for x in required_raw if isinstance(x, str)}

        fields: Dict[str, Tuple[Any, Any]] = {}
        for prop_name, prop_schema in props.items():
            if not isinstance(prop_name, str) or not prop_name.strip():
                continue
            # Keep things simple: only support identifier-like names to avoid aliasing issues.
            if not prop_name.isidentifier():
                raise ValueError(
                    f"Invalid property name '{prop_name}'. Use identifier-style names (letters, digits, underscore)."
                )
            t = _python_type(prop_schema, nested_name=f"{name}_{prop_name}")
            if prop_name in required:
                fields[prop_name] = (t, ...)
            else:
                fields[prop_name] = (Optional[t], None)

        return create_model(name, **fields)  # type: ignore[call-arg]

    return _model(schema, name=name)


def _trace_context(run: RunState) -> Dict[str, str]:
    ctx: Dict[str, str] = {
        "run_id": run.run_id,
        "workflow_id": str(run.workflow_id),
        "node_id": str(run.current_node),
    }
    if run.actor_id:
        ctx["actor_id"] = str(run.actor_id)
    session_id = getattr(run, "session_id", None)
    if session_id:
        ctx["session_id"] = str(session_id)
    if run.parent_run_id:
        ctx["parent_run_id"] = str(run.parent_run_id)
    return ctx


def make_llm_call_handler(*, llm: AbstractCoreLLMClient) -> EffectHandler:
    def _handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        payload = dict(effect.payload or {})
        prompt = payload.get("prompt")
        messages = payload.get("messages")
        system_prompt = payload.get("system_prompt")
        provider = payload.get("provider")
        model = payload.get("model")
        tools_raw = payload.get("tools")
        tools = tools_raw if isinstance(tools_raw, list) and len(tools_raw) > 0 else None
        response_schema = _normalize_response_schema(payload.get("response_schema"))
        response_schema_name = payload.get("response_schema_name")
        structured_output_fallback = payload.get("structured_output_fallback")
        raw_params = payload.get("params")
        params = dict(raw_params) if isinstance(raw_params, dict) else {}

        # Propagate durable trace context into AbstractCore calls.
        trace_metadata = params.get("trace_metadata")
        if not isinstance(trace_metadata, dict):
            trace_metadata = {}
        trace_metadata.update(_trace_context(run))
        params["trace_metadata"] = trace_metadata

        # Support per-effect routing: allow the payload to override provider/model.
        # These reserved keys are consumed by MultiLocalAbstractCoreLLMClient and
        # ignored by LocalAbstractCoreLLMClient.
        if isinstance(provider, str) and provider.strip():
            params["_provider"] = provider.strip()
        if isinstance(model, str) and model.strip():
            params["_model"] = model.strip()

        if not prompt and not messages:
            return EffectOutcome.failed("llm_call requires payload.prompt or payload.messages")

        # Enforce a per-call (or per-run) input-token budget by trimming oldest non-system messages.
        #
        # This is separate from provider limits: it protects reasoning quality and latency by keeping
        # the active context window bounded even when the model supports very large contexts.
        max_input_tokens: Optional[int] = None
        try:
            raw_max_in = payload.get("max_input_tokens")
            if raw_max_in is None:
                limits = run.vars.get("_limits") if isinstance(run.vars, dict) else None
                raw_max_in = limits.get("max_input_tokens") if isinstance(limits, dict) else None
            if raw_max_in is not None and not isinstance(raw_max_in, bool):
                parsed = int(raw_max_in)
                if parsed > 0:
                    max_input_tokens = parsed
        except Exception:
            max_input_tokens = None

        if isinstance(max_input_tokens, int) and max_input_tokens > 0 and isinstance(messages, list) and messages:
            try:
                from abstractruntime.memory.token_budget import trim_messages_to_max_input_tokens

                model_name = model if isinstance(model, str) and model.strip() else None
                messages = trim_messages_to_max_input_tokens(messages, max_input_tokens=int(max_input_tokens), model=model_name)
            except Exception:
                # Never fail an LLM call due to trimming.
                pass

        def _coerce_boolish(value: Any) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return value != 0
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "y", "on"}
            return False

        fallback_enabled = _coerce_boolish(structured_output_fallback)
        base_params = dict(params)

        try:
            structured_requested = isinstance(response_schema, dict) and response_schema
            params_for_call = dict(params)
            if structured_requested:
                model_name = (
                    str(response_schema_name).strip()
                    if isinstance(response_schema_name, str) and response_schema_name.strip()
                    else "StructuredOutput"
                )
                params_for_call["response_model"] = _pydantic_model_from_json_schema(response_schema, name=model_name)

            runtime_observability = {
                "llm_generate_kwargs": _jsonable(
                    {
                        "prompt": str(prompt or ""),
                        "messages": messages,
                        "system_prompt": system_prompt,
                        "tools": tools,
                        "params": params_for_call,
                        "structured_output_fallback": fallback_enabled,
                    }
                ),
            }

            structured_failed = False
            structured_error: Optional[str] = None
            try:
                result = llm.generate(
                    prompt=str(prompt or ""),
                    messages=messages,
                    system_prompt=system_prompt,
                    tools=tools,
                    params=params_for_call,
                )
            except Exception as e:
                looks_like_validation = False
                try:
                    from pydantic import ValidationError as PydanticValidationError  # type: ignore

                    looks_like_validation = isinstance(e, PydanticValidationError)
                except Exception:
                    looks_like_validation = False

                msg = str(e)
                if not looks_like_validation:
                    lowered = msg.lower()
                    if "validation errors for" in lowered or "structured output generation failed" in lowered:
                        looks_like_validation = True

                if not (fallback_enabled and structured_requested and looks_like_validation):
                    raise

                logger.warning(
                    "LLM_CALL structured output failed; retrying without schema",
                    error=msg,
                )

                result = llm.generate(
                    prompt=str(prompt or ""),
                    messages=messages,
                    system_prompt=system_prompt,
                    tools=tools,
                    params=dict(base_params),
                )
                structured_failed = True
                structured_error = msg

            if structured_requested and isinstance(result, dict):
                # Best-effort: when structured outputs fail (or providers ignore response_model),
                # try to parse the returned text into `data` so downstream nodes can consume it.
                try:
                    existing_data = result.get("data")
                except Exception:
                    existing_data = None

                content_value = result.get("content") if isinstance(result.get("content"), str) else None

                if existing_data is None and isinstance(content_value, str) and content_value.strip():
                    parsed: Any = None
                    parse_error: Optional[str] = None
                    try:
                        from abstractruntime.visualflow_compiler.visual.builtins import data_parse_json

                        parsed = data_parse_json({"text": content_value, "wrap_scalar": True})
                    except Exception as e:
                        parse_error = str(e)
                        parsed = None

                    # Response schemas in this system are object-only; wrap non-dicts for safety.
                    if parsed is not None and not isinstance(parsed, dict):
                        parsed = {"value": parsed}
                    if parsed is not None:
                        result["data"] = parsed

                    if parse_error is not None:
                        meta = result.get("metadata")
                        if not isinstance(meta, dict):
                            meta = {}
                            result["metadata"] = meta
                        meta["_structured_output_parse_error"] = parse_error

            if isinstance(result, dict):
                meta = result.get("metadata")
                if not isinstance(meta, dict):
                    meta = {}
                    result["metadata"] = meta
                if structured_failed:
                    meta["_structured_output_fallback"] = {"used": True, "error": structured_error or ""}
                existing = meta.get("_runtime_observability")
                if not isinstance(existing, dict):
                    existing = {}
                    meta["_runtime_observability"] = existing
                existing.update(runtime_observability)

            # VisualFlow "Use context" UX: when requested, persist the turn into the run's
            # active context (`vars.context.messages`) so subsequent LLM/Agent/Subflow nodes
            # can see the interaction history without extra wiring.
            #
            # IMPORTANT: This is opt-in via payload.include_context/use_context; AbstractRuntime
            # does not implicitly store all LLM calls in context.
            try:
                inc_raw = payload.get("include_context")
                if inc_raw is None:
                    inc_raw = payload.get("use_context")
                if _coerce_boolish(inc_raw):
                    from abstractruntime.core.vars import get_context

                    ctx_ns = get_context(run.vars)
                    msgs_any = ctx_ns.get("messages")
                    if not isinstance(msgs_any, list):
                        msgs_any = []
                        ctx_ns["messages"] = msgs_any

                    def _extract_last_user_text() -> str:
                        if isinstance(prompt, str) and prompt.strip():
                            return prompt.strip()
                        if isinstance(messages, list):
                            for m in reversed(messages):
                                if not isinstance(m, dict):
                                    continue
                                if m.get("role") != "user":
                                    continue
                                c = m.get("content")
                                if isinstance(c, str) and c.strip():
                                    return c.strip()
                        return ""

                    def _extract_assistant_text() -> str:
                        if isinstance(result, dict):
                            c = result.get("content")
                            if isinstance(c, str) and c.strip():
                                return c.strip()
                            d = result.get("data")
                            if isinstance(d, (dict, list)):
                                import json as _json

                                return _json.dumps(d, ensure_ascii=False, indent=2)
                        return ""

                    user_text = _extract_last_user_text()
                    assistant_text = _extract_assistant_text()
                    node_id = str(getattr(run, "current_node", None) or "").strip() or "unknown"

                    if user_text:
                        msgs_any.append(
                            {
                                "role": "user",
                                "content": user_text,
                                "metadata": {"kind": "llm_turn", "node_id": node_id},
                            }
                        )
                    if assistant_text:
                        msgs_any.append(
                            {
                                "role": "assistant",
                                "content": assistant_text,
                                "metadata": {"kind": "llm_turn", "node_id": node_id},
                            }
                        )
                    if isinstance(getattr(run, "output", None), dict):
                        run.output["messages"] = msgs_any
            except Exception:
                pass
            return EffectOutcome.completed(result=result)
        except Exception as e:
            logger.error("LLM_CALL failed", error=str(e))
            return EffectOutcome.failed(str(e))

    return _handler


def make_tool_calls_handler(*, tools: Optional[ToolExecutor] = None) -> EffectHandler:
    """Create a TOOL_CALLS effect handler.

    Tool execution is performed exclusively via the host-configured ToolExecutor.
    This keeps `RunState.vars` and ledger payloads JSON-safe (durable execution).
    """
    def _handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        payload = dict(effect.payload or {})
        tool_calls = payload.get("tool_calls")
        if not isinstance(tool_calls, list):
            return EffectOutcome.failed("tool_calls requires payload.tool_calls (list)")
        allowed_tools_raw = payload.get("allowed_tools")
        allowlist_enabled = isinstance(allowed_tools_raw, list)
        allowed_tools: Set[str] = set()
        if allowlist_enabled:
            allowed_tools = {str(t) for t in allowed_tools_raw if isinstance(t, str) and t.strip()}

        if tools is None:
            return EffectOutcome.failed(
                "TOOL_CALLS requires a ToolExecutor; configure Runtime with "
                "MappingToolExecutor/AbstractCoreToolExecutor/PassthroughToolExecutor."
            )

        original_call_count = len(tool_calls)

        # Always block non-dict tool call entries: passthrough hosts expect dicts and may crash otherwise.
        blocked_by_index: Dict[int, Dict[str, Any]] = {}
        filtered_tool_calls: list[Dict[str, Any]] = []

        # For evidence and deterministic resume merging, keep a positional tool call list aligned to the
        # *original* tool call order. Blocked entries are represented as empty-args stubs.
        tool_calls_for_evidence: list[Dict[str, Any]] = []

        for idx, tc in enumerate(tool_calls):
            if not isinstance(tc, dict):
                blocked_by_index[idx] = {
                    "call_id": "",
                    "name": "",
                    "success": False,
                    "output": None,
                    "error": "Invalid tool call (expected an object)",
                }
                tool_calls_for_evidence.append({})
                continue

            name_raw = tc.get("name")
            name = name_raw.strip() if isinstance(name_raw, str) else ""
            call_id = str(tc.get("call_id") or "")

            if allowlist_enabled:
                if not name:
                    blocked_by_index[idx] = {
                        "call_id": call_id,
                        "name": "",
                        "success": False,
                        "output": None,
                        "error": "Tool call missing a valid name",
                    }
                    tool_calls_for_evidence.append({"call_id": call_id, "name": "", "arguments": {}})
                    continue
                if name not in allowed_tools:
                    blocked_by_index[idx] = {
                        "call_id": call_id,
                        "name": name,
                        "success": False,
                        "output": None,
                        "error": f"Tool '{name}' is not allowed for this node",
                    }
                    # Do not leak arguments for disallowed tools into the durable wait payload.
                    tool_calls_for_evidence.append({"call_id": call_id, "name": name, "arguments": {}})
                    continue

            # Allowed (or allowlist disabled): include for execution and keep full args for evidence.
            filtered_tool_calls.append(tc)
            tool_calls_for_evidence.append(tc)

        # If everything was blocked, complete immediately with blocked results (no waiting/execution).
        if not filtered_tool_calls and blocked_by_index:
            return EffectOutcome.completed(
                result={
                    "mode": "executed",
                    "results": [blocked_by_index[i] for i in sorted(blocked_by_index.keys())],
                }
            )

        try:
            result = tools.execute(tool_calls=filtered_tool_calls)
        except Exception as e:
            logger.error("TOOL_CALLS execution failed", error=str(e))
            return EffectOutcome.failed(str(e))

        mode = result.get("mode")
        if mode and mode != "executed":
            # Passthrough/untrusted mode: pause until an external host resumes with tool results.
            #
            # Correctness/security: persist only allowlist-safe tool calls in the wait payload.
            wait_key = payload.get("wait_key") or result.get("wait_key") or f"tool_calls:{run.run_id}:{run.current_node}"
            raw_wait_reason = result.get("wait_reason")
            wait_reason = WaitReason.EVENT
            if isinstance(raw_wait_reason, str) and raw_wait_reason.strip():
                try:
                    wait_reason = WaitReason(raw_wait_reason.strip())
                except ValueError:
                    wait_reason = WaitReason.EVENT
            elif str(mode).strip().lower() == "delegated":
                wait_reason = WaitReason.JOB

            tool_calls_for_wait = result.get("tool_calls")
            if not isinstance(tool_calls_for_wait, list):
                tool_calls_for_wait = filtered_tool_calls

            details: Dict[str, Any] = {"mode": mode, "tool_calls": _jsonable(tool_calls_for_wait)}
            executor_details = result.get("details")
            if isinstance(executor_details, dict) and executor_details:
                # Avoid collisions with our reserved keys.
                details["executor"] = _jsonable(executor_details)
            if blocked_by_index:
                details["original_call_count"] = original_call_count
                details["blocked_by_index"] = {str(k): _jsonable(v) for k, v in blocked_by_index.items()}
                details["tool_calls_for_evidence"] = _jsonable(tool_calls_for_evidence)

            wait = WaitState(
                reason=wait_reason,
                wait_key=str(wait_key),
                resume_to_node=payload.get("resume_to_node") or default_next_node,
                result_key=effect.result_key,
                details=details,
            )
            return EffectOutcome.waiting(wait)

        if blocked_by_index:
            existing_results = result.get("results")
            if isinstance(existing_results, list):
                merged_results: list[Any] = []
                executed_iter = iter(existing_results)
                for idx in range(len(tool_calls)):
                    blocked = blocked_by_index.get(idx)
                    if blocked is not None:
                        merged_results.append(blocked)
                        continue
                    try:
                        merged_results.append(next(executed_iter))
                    except StopIteration:
                        merged_results.append(
                            {
                                "call_id": "",
                                "name": "",
                                "success": False,
                                "output": None,
                                "error": "Missing tool result",
                            }
                        )
                result = dict(result)
                result["results"] = merged_results

        return EffectOutcome.completed(result=result)

    return _handler


def build_effect_handlers(*, llm: AbstractCoreLLMClient, tools: ToolExecutor = None) -> Dict[EffectType, Any]:
    return {
        EffectType.LLM_CALL: make_llm_call_handler(llm=llm),
        EffectType.TOOL_CALLS: make_tool_calls_handler(tools=tools),
    }
