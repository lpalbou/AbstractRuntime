"""abstractruntime.integrations.abstractcore.effect_handlers

Effect handlers wiring for AbstractRuntime.

These handlers implement:
- `EffectType.LLM_CALL`
- `EffectType.TOOL_CALLS`

They are designed to keep `RunState.vars` JSON-safe.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ...core.models import Effect, EffectType, RunState, WaitReason, WaitState
from ...core.runtime import EffectOutcome, EffectHandler
from .llm_client import AbstractCoreLLMClient
from .tool_executor import ToolExecutor
from .logging import get_logger

logger = get_logger(__name__)


def make_llm_call_handler(*, llm: AbstractCoreLLMClient) -> EffectHandler:
    def _handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        payload = dict(effect.payload or {})
        prompt = payload.get("prompt")
        messages = payload.get("messages")
        system_prompt = payload.get("system_prompt")
        tools = payload.get("tools")
        params = payload.get("params")

        if not prompt and not messages:
            return EffectOutcome.failed("llm_call requires payload.prompt or payload.messages")

        try:
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


def make_tool_calls_handler(*, tools: ToolExecutor = None) -> EffectHandler:
    """Create a TOOL_CALLS effect handler.
    
    Tool execution priority:
    1. Tools from effect payload (payload.tools)
    2. Tools from run.vars["_tools"] (set at workflow start)
    3. Fallback to provided ToolExecutor
    
    This allows agents to pass tools directly without needing a registry.
    """
    def _execute_tools_directly(tool_calls: list, tool_funcs: list) -> dict:
        """Execute tools directly from function list."""
        # Build lookup from tool name to function
        tool_lookup = {}
        for t in tool_funcs:
            if hasattr(t, '_tool_definition'):
                td = t._tool_definition
                tool_lookup[td.name] = td.function
            elif callable(t):
                # Fallback: use function name
                tool_lookup[t.__name__] = t
        
        results = []
        for tc in tool_calls:
            name = tc.get("name", "")
            args = tc.get("arguments", {})
            call_id = tc.get("call_id", "")
            
            func = tool_lookup.get(name)
            if func:
                try:
                    output = func(**args)
                    results.append({
                        "call_id": call_id,
                        "name": name,
                        "success": True,
                        "output": str(output) if output is not None else None,
                        "error": None,
                    })
                except Exception as e:
                    results.append({
                        "call_id": call_id,
                        "name": name,
                        "success": False,
                        "output": None,
                        "error": str(e),
                    })
            else:
                results.append({
                    "call_id": call_id,
                    "name": name,
                    "success": False,
                    "output": None,
                    "error": f"Tool '{name}' not found",
                })
        
        return {"mode": "executed", "results": results}
    
    def _handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        payload = dict(effect.payload or {})
        tool_calls = payload.get("tool_calls")
        if not isinstance(tool_calls, list):
            return EffectOutcome.failed("tool_calls requires payload.tool_calls (list)")

        try:
            # Priority 1: Tools from effect payload
            tool_funcs = payload.get("tools")
            
            # Priority 2: Tools from run.vars
            if not tool_funcs:
                tool_funcs = run.vars.get("_tools")
            
            # If we have tool functions, execute directly
            if tool_funcs:
                result = _execute_tools_directly(tool_calls, tool_funcs)
            # Priority 3: Fallback to ToolExecutor
            elif tools:
                result = tools.execute(tool_calls=tool_calls)
            else:
                return EffectOutcome.failed("No tools available for execution")
                
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

        return EffectOutcome.completed(result=result)

    return _handler


def build_effect_handlers(*, llm: AbstractCoreLLMClient, tools: ToolExecutor = None) -> Dict[EffectType, Any]:
    return {
        EffectType.LLM_CALL: make_llm_call_handler(llm=llm),
        EffectType.TOOL_CALLS: make_tool_calls_handler(tools=tools),
    }

