"""Public run-scoped AbstractCore facade for durable Runtime execution.

This helper is for host-triggered work that should still be authored by the
Runtime ledger instead of executing in host route/controller code.

Current focus:
- durable run-scoped `LLM_CALL` child runs
- convenience wrappers for image generation, music generation, TTS, and STT/transcription
"""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, Optional, Protocol

from ...core.models import Effect, EffectType, RunState, StepPlan
from ...core.spec import WorkflowSpec

_RESULT_KEY = "result"
_WORKFLOW_PREFIX = "wf_abstractcore_run_facade"


class AbstractCoreRuntimeLike(Protocol):
    """Duck-typed runtime contract needed for durable child-run helpers."""

    def start(
        self,
        *,
        workflow: WorkflowSpec,
        vars: Optional[Dict[str, Any]] = None,
        actor_id: Optional[str] = None,
        session_id: Optional[str] = None,
        parent_run_id: Optional[str] = None,
    ) -> str:
        ...

    def tick(self, *, workflow: WorkflowSpec, run_id: str, max_steps: int = 100) -> RunState:
        ...

    def get_state(self, run_id: str) -> RunState:
        ...


def _coerce_runtime(runtime: Any) -> AbstractCoreRuntimeLike:
    missing = [name for name in ("start", "tick", "get_state") if not callable(getattr(runtime, name, None))]
    if missing:
        methods = ", ".join(missing)
        raise TypeError(
            "Runtime does not implement the AbstractCore durable run facade contract. "
            f"Missing methods: {methods}."
        )
    return runtime


def _workflow_id_for_output(output: Any) -> str:
    suffix = "llm_call"
    if isinstance(output, dict):
        task = str(output.get("task") or "").strip().lower()
        modality = str(output.get("modality") or "").strip().lower()
        suffix = task or modality or suffix
    safe = re.sub(r"[^a-z0-9_]+", "_", suffix).strip("_") or "llm_call"
    return f"{_WORKFLOW_PREFIX}_{safe}"


def _build_child_vars(*, parent: RunState, child_vars: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    runtime_ns = parent.vars.get("_runtime") if isinstance(parent.vars, dict) else None
    if isinstance(runtime_ns, dict):
        out["_runtime"] = copy.deepcopy(runtime_ns)

    if not isinstance(child_vars, dict):
        return out

    for key, value in child_vars.items():
        if key == "_runtime" and isinstance(value, dict):
            merged = out.get("_runtime")
            if not isinstance(merged, dict):
                merged = {}
                out["_runtime"] = merged
            merged.update(copy.deepcopy(value))
            continue
        out[str(key)] = copy.deepcopy(value)
    return out


class AbstractCoreRunFacade:
    """Public host-facing helper for durable run-scoped AbstractCore work.

    Methods on this facade create child runs under an existing parent run. The
    Runtime executes the actual `LLM_CALL`, writes the ledger, and stores the
    durable result in the child run output.
    """

    __slots__ = ("_runtime",)

    def __init__(self, runtime: Any):
        self._runtime = _coerce_runtime(runtime)

    @classmethod
    def from_runtime(cls, runtime: Any) -> "AbstractCoreRunFacade":
        """Bind the durable run facade to a runtime."""

        return cls(runtime)

    def execute_llm_call(
        self,
        run_id: str,
        *,
        prompt: Optional[str] = None,
        text: Optional[str] = None,
        messages: Optional[list[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        media: Any = None,
        output: Any = None,
        params: Optional[Dict[str, Any]] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        response_schema_name: Optional[str] = None,
        structured_output_fallback: Any = None,
        child_vars: Optional[Dict[str, Any]] = None,
        workflow_id: Optional[str] = None,
    ) -> RunState:
        """Execute a durable child `LLM_CALL` under an existing run."""

        parent = self._runtime.get_state(run_id)
        payload: Dict[str, Any] = {}
        if prompt is not None:
            payload["prompt"] = str(prompt)
        if text is not None:
            payload["text"] = str(text)
        if isinstance(messages, list):
            payload["messages"] = copy.deepcopy(messages)
        if system_prompt is not None:
            payload["system_prompt"] = str(system_prompt)
        if media is not None:
            payload["media"] = copy.deepcopy(media)
        if output is not None:
            payload["output"] = copy.deepcopy(output)
        if isinstance(params, dict) and params:
            payload["params"] = copy.deepcopy(params)
        if isinstance(response_schema, dict) and response_schema:
            payload["response_schema"] = copy.deepcopy(response_schema)
        if isinstance(response_schema_name, str) and response_schema_name.strip():
            payload["response_schema_name"] = response_schema_name.strip()
        if structured_output_fallback is not None:
            payload["structured_output_fallback"] = structured_output_fallback

        result_key = _RESULT_KEY

        def call(run: RunState, ctx: Any) -> StepPlan:
            _ = ctx
            return StepPlan(
                node_id="call",
                effect=Effect(
                    type=EffectType.LLM_CALL,
                    payload=copy.deepcopy(payload),
                    result_key=result_key,
                ),
                next_node="done",
            )

        def done(run: RunState, ctx: Any) -> StepPlan:
            _ = ctx
            return StepPlan(
                node_id="done",
                complete_output={"result": copy.deepcopy(run.vars.get(result_key))},
            )

        workflow = WorkflowSpec(
            workflow_id=workflow_id or _workflow_id_for_output(output),
            entry_node="call",
            nodes={"call": call, "done": done},
        )
        child_run_id = self._runtime.start(
            workflow=workflow,
            vars=_build_child_vars(parent=parent, child_vars=child_vars),
            actor_id=parent.actor_id,
            session_id=parent.session_id,
            parent_run_id=parent.run_id,
        )
        return self._runtime.tick(workflow=workflow, run_id=child_run_id)

    def generate_image(
        self,
        run_id: str,
        *,
        prompt: str,
        output: Optional[Dict[str, Any]] = None,
        media: Any = None,
        params: Optional[Dict[str, Any]] = None,
        child_vars: Optional[Dict[str, Any]] = None,
    ) -> RunState:
        """Create a durable child run for image generation."""

        spec = {"modality": "image", "task": "image_generation"}
        if isinstance(output, dict):
            spec.update(copy.deepcopy(output))
        return self.execute_llm_call(
            run_id,
            prompt=prompt,
            media=media,
            output=spec,
            params=params,
            child_vars=child_vars,
        )

    def generate_voice(
        self,
        run_id: str,
        *,
        text: str,
        output: Optional[Dict[str, Any]] = None,
        media: Any = None,
        params: Optional[Dict[str, Any]] = None,
        child_vars: Optional[Dict[str, Any]] = None,
    ) -> RunState:
        """Create a durable child run for TTS/voice generation."""

        spec = {"modality": "voice", "task": "tts"}
        if isinstance(output, dict):
            spec.update(copy.deepcopy(output))
        return self.execute_llm_call(
            run_id,
            text=text,
            media=media,
            output=spec,
            params=params,
            child_vars=child_vars,
        )

    def generate_music(
        self,
        run_id: str,
        *,
        prompt: str,
        output: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        child_vars: Optional[Dict[str, Any]] = None,
    ) -> RunState:
        """Create a durable child run for music generation."""

        spec = {"modality": "music", "task": "music_generation"}
        if isinstance(output, dict):
            spec.update(copy.deepcopy(output))
        return self.execute_llm_call(
            run_id,
            prompt=prompt,
            output=spec,
            params=params,
            child_vars=child_vars,
        )

    def transcribe_audio(
        self,
        run_id: str,
        *,
        media: Any,
        prompt: Optional[str] = None,
        output: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        child_vars: Optional[Dict[str, Any]] = None,
    ) -> RunState:
        """Create a durable child run for STT/transcription."""

        spec = {"modality": "text", "task": "transcription"}
        if isinstance(output, dict):
            spec.update(copy.deepcopy(output))
        return self.execute_llm_call(
            run_id,
            prompt=prompt,
            media=media,
            output=spec,
            params=params,
            child_vars=child_vars,
        )


def get_abstractcore_run_facade(runtime: Any) -> AbstractCoreRunFacade:
    """Return the public AbstractCore durable run facade bound to a runtime."""

    return AbstractCoreRunFacade.from_runtime(runtime)


__all__ = [
    "AbstractCoreRunFacade",
    "get_abstractcore_run_facade",
]
