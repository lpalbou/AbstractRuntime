"""abstractruntime.integrations.abstractcore.effect_handlers

Effect handlers wiring for AbstractRuntime.

These handlers implement:
- `EffectType.LLM_CALL`
- `EffectType.TOOL_CALLS`

They are designed to keep `RunState.vars` JSON-safe.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Set, Tuple, Type

from ...core.models import Effect, EffectType, RunState, WaitReason, WaitState
from ...core.runtime import EffectOutcome, EffectHandler
from .llm_client import AbstractCoreLLMClient
from .tool_executor import ToolExecutor
from .logging import get_logger

logger = get_logger(__name__)


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
        if obj_schema.get("type") != "object":
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
        tools = payload.get("tools")
        response_schema = payload.get("response_schema")
        response_schema_name = payload.get("response_schema_name")
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

        try:
            if isinstance(response_schema, dict) and response_schema:
                model_name = (
                    str(response_schema_name).strip()
                    if isinstance(response_schema_name, str) and response_schema_name.strip()
                    else "StructuredOutput"
                )
                params["response_model"] = _pydantic_model_from_json_schema(response_schema, name=model_name)

            result = llm.generate(
                prompt=str(prompt or ""),
                messages=messages,
                system_prompt=system_prompt,
                tools=tools,
                params=params,
            )
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
        allowed_tools: Optional[Set[str]] = None
        if isinstance(allowed_tools_raw, list):
            allowed_tools = {str(t) for t in allowed_tools_raw if isinstance(t, str) and t.strip()}

        if tools is None:
            return EffectOutcome.failed(
                "TOOL_CALLS requires a ToolExecutor; configure Runtime with "
                "MappingToolExecutor/AbstractCoreToolExecutor/PassthroughToolExecutor."
            )

        filtered_tool_calls = tool_calls
        blocked_by_index: Dict[int, Dict[str, Any]] = {}
        if allowed_tools:
            filtered_tool_calls = []
            for idx, tc in enumerate(tool_calls):
                if not isinstance(tc, dict):
                    filtered_tool_calls.append(tc)
                    continue
                name = tc.get("name")
                name_str = str(name) if isinstance(name, str) else ""
                if name_str and name_str not in allowed_tools:
                    blocked_by_index[idx] = {
                        "call_id": str(tc.get("call_id") or ""),
                        "name": name_str,
                        "success": False,
                        "output": None,
                        "error": f"Tool '{name_str}' is not allowed for this node",
                    }
                    continue
                filtered_tool_calls.append(tc)

            if not filtered_tool_calls:
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
            wait_key = payload.get("wait_key") or f"tool_calls:{run.run_id}:{run.current_node}"
            wait = WaitState(
                reason=WaitReason.EVENT,
                wait_key=str(wait_key),
                resume_to_node=payload.get("resume_to_node") or default_next_node,
                result_key=effect.result_key,
                details={"mode": mode, "tool_calls": tool_calls},
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
