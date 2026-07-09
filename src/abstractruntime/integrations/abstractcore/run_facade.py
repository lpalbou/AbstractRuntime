"""Public run-scoped AbstractCore facade for durable Runtime execution.

This helper is for host-triggered work that should still be authored by the
Runtime ledger instead of executing in host route/controller code.

Current focus:
- durable run-scoped `LLM_CALL` child runs
- durable run-scoped `TOOL_CALLS` child runs
- convenience wrappers for image/video generation, music generation, TTS, and STT/transcription
- convenience wrappers for durable outbound comms sends
"""

from __future__ import annotations

import copy
import re
import threading
import uuid
from typing import Any, Dict, Optional, Protocol

from ...core.models import Effect, EffectType, RunState, RunStatus, StepPlan
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

    def resume(
        self,
        *,
        workflow: WorkflowSpec,
        run_id: str,
        wait_key: Optional[str],
        payload: Dict[str, Any],
        max_steps: int = 100,
    ) -> RunState:
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


def _validate_music_output_spec(output: Any) -> None:
    if not isinstance(output, dict):
        return
    legacy_backend = str(output.get("backend") or output.get("music_backend") or "").strip()
    if legacy_backend:
        raise ValueError(
            "Music output uses `provider` as the backend selector; "
            "`backend` and `music_backend` are not supported."
        )


def _workflow_id_for_tool_calls(tool_calls: list[Dict[str, Any]]) -> str:
    suffix = "tool_calls"
    if isinstance(tool_calls, list) and tool_calls:
        first = tool_calls[0]
        if isinstance(first, dict):
            name = str(first.get("name") or "").strip().lower()
            if name:
                suffix = f"tool_{name}"
    safe = re.sub(r"[^a-z0-9_]+", "_", suffix).strip("_") or "tool_calls"
    return f"{_WORKFLOW_PREFIX}_{safe}"


def _build_tool_calls_workflow(
    *,
    workflow_id: str,
    payload: Dict[str, Any],
    result_key: str,
) -> WorkflowSpec:
    def call(run: RunState, ctx: Any) -> StepPlan:
        _ = ctx
        return StepPlan(
            node_id="call",
            effect=Effect(
                type=EffectType.TOOL_CALLS,
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

    return WorkflowSpec(
        workflow_id=workflow_id,
        entry_node="call",
        nodes={"call": call, "done": done},
    )


def _build_tool_calls_resume_workflow(
    *,
    workflow_id: str,
    result_key: str,
    resume_to_node: Optional[str],
) -> WorkflowSpec:
    node_id = str(resume_to_node or "done").strip() or "done"

    def done(run: RunState, ctx: Any) -> StepPlan:
        _ = ctx
        return StepPlan(
            node_id=node_id,
            complete_output={"result": copy.deepcopy(run.vars.get(result_key))},
        )

    return WorkflowSpec(
        workflow_id=workflow_id,
        entry_node=node_id,
        nodes={node_id: done},
    )


def _build_stream_wait_workflow(
    *,
    workflow_id: str,
    wait_key: str,
    payload: Dict[str, Any],
    result_key: str,
) -> WorkflowSpec:
    def wait(run: RunState, ctx: Any) -> StepPlan:
        _ = run, ctx
        return StepPlan(
            node_id="wait",
            effect=Effect(
                type=EffectType.WAIT_EVENT,
                payload={
                    "wait_key": wait_key,
                    "resume_to_node": "done",
                    "prompt": "Streaming voice synthesis is running.",
                    "allow_free_text": False,
                    "details": copy.deepcopy(payload),
                },
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

    return WorkflowSpec(
        workflow_id=workflow_id,
        entry_node="wait",
        nodes={"wait": wait, "done": done},
    )


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

    def execute_tool_calls(
        self,
        run_id: str,
        *,
        tool_calls: list[Dict[str, Any]],
        allowed_tools: Optional[list[str]] = None,
        child_vars: Optional[Dict[str, Any]] = None,
        workflow_id: Optional[str] = None,
    ) -> RunState:
        """Execute durable child `TOOL_CALLS` under an existing run."""

        parent = self._runtime.get_state(run_id)
        payload: Dict[str, Any] = {"tool_calls": copy.deepcopy(tool_calls or [])}
        if isinstance(allowed_tools, list) and allowed_tools:
            payload["allowed_tools"] = [str(name) for name in allowed_tools if isinstance(name, str) and name.strip()]

        result_key = _RESULT_KEY
        workflow = _build_tool_calls_workflow(
            workflow_id=workflow_id or _workflow_id_for_tool_calls(payload["tool_calls"]),
            payload=payload,
            result_key=result_key,
        )
        child_run_id = self._runtime.start(
            workflow=workflow,
            vars=_build_child_vars(parent=parent, child_vars=child_vars),
            actor_id=parent.actor_id,
            session_id=parent.session_id,
            parent_run_id=parent.run_id,
        )
        return self._runtime.tick(workflow=workflow, run_id=child_run_id)

    def resume_tool_calls(
        self,
        child_run_id: str,
        *,
        payload: Dict[str, Any],
        wait_key: Optional[str] = None,
        max_steps: int = 100,
    ) -> RunState:
        """Resume a waiting durable `TOOL_CALLS` child run.

        Typical payloads are:
        - `{"approved": true}` for approval-gated execution
        - `{"mode": "executed", "results": [...]}` for passthrough/delegated execution
        """

        child = self._runtime.get_state(child_run_id)
        waiting = child.waiting
        if child.status != RunStatus.WAITING or waiting is None:
            raise ValueError(f"Run '{child_run_id}' is not waiting")

        result_key = str(waiting.result_key or _RESULT_KEY).strip() or _RESULT_KEY
        workflow = _build_tool_calls_resume_workflow(
            workflow_id=str(child.workflow_id or _workflow_id_for_tool_calls([])),
            result_key=result_key,
            resume_to_node=waiting.resume_to_node,
        )
        return self._runtime.resume(
            workflow=workflow,
            run_id=child.run_id,
            wait_key=wait_key if wait_key is not None else waiting.wait_key,
            payload=copy.deepcopy(payload),
            max_steps=max_steps,
        )

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

    def edit_image(
        self,
        run_id: str,
        *,
        prompt: str,
        media: Any,
        output: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        child_vars: Optional[Dict[str, Any]] = None,
    ) -> RunState:
        """Create a durable child run for image-to-image editing."""

        spec = {"modality": "image", "task": "image_edit"}
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

    def upscale_image(
        self,
        run_id: str,
        *,
        media: Any,
        output: Optional[Dict[str, Any]] = None,
        prompt: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        child_vars: Optional[Dict[str, Any]] = None,
    ) -> RunState:
        """Create a durable child run for image upscaling."""

        spec = {"modality": "image", "task": "image_upscale"}
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

    def generate_video(
        self,
        run_id: str,
        *,
        prompt: str,
        output: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        child_vars: Optional[Dict[str, Any]] = None,
    ) -> RunState:
        """Create a durable child run for text-to-video generation."""

        spec = {"modality": "video", "task": "text_to_video"}
        if isinstance(output, dict):
            spec.update(copy.deepcopy(output))
        return self.execute_llm_call(
            run_id,
            prompt=prompt,
            output=spec,
            params=params,
            child_vars=child_vars,
        )

    def image_to_video(
        self,
        run_id: str,
        *,
        prompt: str,
        media: Any,
        output: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        child_vars: Optional[Dict[str, Any]] = None,
    ) -> RunState:
        """Create a durable child run for image-to-video generation."""

        spec = {"modality": "video", "task": "image_to_video"}
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

    def stream_voice(
        self,
        run_id: str,
        *,
        text: str,
        output: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        child_vars: Optional[Dict[str, Any]] = None,
    ):
        """Create a durable child run and yield Runtime-owned TTS stream events."""

        client = getattr(self._runtime, "_abstractcore_llm_client", None)
        stream_fn = getattr(client, "stream_tts", None)
        if not callable(stream_fn):
            raise ValueError("Runtime AbstractCore client does not expose streaming TTS.")

        parent = self._runtime.get_state(run_id)
        spec = {"modality": "voice", "task": "tts"}
        if isinstance(output, dict):
            spec.update(copy.deepcopy(output))
        stream_params = copy.deepcopy(params) if isinstance(params, dict) else {}
        wait_key = f"abstractcore.voice.tts.stream:{uuid.uuid4()}"
        result_key = _RESULT_KEY
        payload = {
            "mode": "abstractcore_voice_stream",
            "text": str(text or ""),
            "output": copy.deepcopy(spec),
            "params": copy.deepcopy(stream_params),
        }
        workflow = _build_stream_wait_workflow(
            workflow_id=f"{_workflow_id_for_output(spec)}_stream",
            wait_key=wait_key,
            payload=payload,
            result_key=result_key,
        )
        child_run_id = self._runtime.start(
            workflow=workflow,
            vars=_build_child_vars(parent=parent, child_vars=child_vars),
            actor_id=parent.actor_id,
            session_id=parent.session_id,
            parent_run_id=parent.run_id,
        )
        child = self._runtime.tick(workflow=workflow, run_id=child_run_id)
        if child.status != RunStatus.WAITING or child.waiting is None:
            raise ValueError("Runtime failed to initialize streaming voice child run.")

        cancel_event = threading.Event()
        trace_metadata = dict(stream_params.get("trace_metadata") or {}) if isinstance(stream_params.get("trace_metadata"), dict) else {}
        trace_metadata.setdefault("run_id", child.run_id)
        trace_metadata.setdefault("parent_run_id", parent.run_id)
        if parent.session_id is not None:
            trace_metadata.setdefault("session_id", parent.session_id)
        if parent.actor_id is not None:
            trace_metadata.setdefault("actor_id", parent.actor_id)
        trace_metadata.setdefault("workflow_id", child.workflow_id)
        stream_params["trace_metadata"] = trace_metadata
        stream_params["output"] = copy.deepcopy(spec)
        stream_params["cancel_event"] = cancel_event

        stream = stream_fn(text=str(text or ""), output=copy.deepcopy(spec), params=stream_params)

        def _final_result_for_event(event: Dict[str, Any]) -> Dict[str, Any]:
            artifact = event.get("audio_artifact") if isinstance(event, dict) else None
            if event.get("type") == "done" and isinstance(artifact, dict):
                return {
                    "content": None,
                    "outputs": {"voice": [copy.deepcopy(artifact)]},
                    "metadata": {
                        "streaming": True,
                        "terminal_event": copy.deepcopy(event),
                    },
                }
            if event.get("type") == "cancelled":
                return {
                    "content": None,
                    "outputs": {},
                    "errors": [{"message": "TTS stream was cancelled.", "code": "cancelled"}],
                    "metadata": {"streaming": True, "cancelled": True, "terminal_event": copy.deepcopy(event)},
                }
            return {
                "content": None,
                "outputs": {},
                "errors": [{"message": str(event.get("error") or "TTS stream failed"), "code": "stream_error"}],
                "metadata": {"streaming": True, "terminal_event": copy.deepcopy(event)},
            }

        def _resume_cancelled(reason: str) -> None:
            event = {
                "type": "cancelled",
                "ok": False,
                "cancelled": True,
                "error": str(reason or "TTS stream was cancelled."),
                "child_run_id": child.run_id,
                "run_id": parent.run_id,
            }
            try:
                self._runtime.resume(
                    workflow=workflow,
                    run_id=child.run_id,
                    wait_key=wait_key,
                    payload=_final_result_for_event(event),
                    max_steps=100,
                )
                return
            except Exception:
                pass
            cancel_run = getattr(self._runtime, "cancel_run", None)
            if callable(cancel_run):
                try:
                    cancel_run(child.run_id, reason=str(reason or "TTS stream was cancelled."))
                except Exception:
                    pass

        def _events():
            terminal_seen = False
            try:
                yield {
                    "type": "runtime_start",
                    "schema": "abstractruntime.tts_stream.start.v1",
                    "ok": True,
                    "run_id": parent.run_id,
                    "child_run_id": child.run_id,
                    "wait_key": wait_key,
                    "durability": "runtime_child_run",
                    "transport": "jsonl",
                    "chunk_format": "wav-segment",
                }
                for raw_event in stream:
                    event = dict(raw_event) if isinstance(raw_event, dict) else {"type": "event", "value": str(raw_event)}
                    event.setdefault("child_run_id", child.run_id)
                    event.setdefault("run_id", parent.run_id)
                    event_type = str(event.get("type") or "").strip().lower()
                    if event_type in {"done", "cancelled", "error"}:
                        terminal_seen = True
                        final_result = _final_result_for_event(event)
                        final_state = self._runtime.resume(
                            workflow=workflow,
                            run_id=child.run_id,
                            wait_key=wait_key,
                            payload=final_result,
                            max_steps=100,
                        )
                        event["child_run_status"] = str(final_state.status.value if hasattr(final_state.status, "value") else final_state.status)
                        yield event
                        return
                    yield event
                if not terminal_seen:
                    event = {
                        "type": "error",
                        "ok": False,
                        "error": "TTS stream ended without a terminal event.",
                        "child_run_id": child.run_id,
                        "run_id": parent.run_id,
                    }
                    final_state = self._runtime.resume(
                        workflow=workflow,
                        run_id=child.run_id,
                        wait_key=wait_key,
                        payload=_final_result_for_event(event),
                        max_steps=100,
                    )
                    event["child_run_status"] = str(final_state.status.value if hasattr(final_state.status, "value") else final_state.status)
                    yield event
            except GeneratorExit:
                cancel_event.set()
                close = getattr(stream, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass
                if not terminal_seen:
                    _resume_cancelled("TTS stream disconnected before completion")
                raise
            except Exception as exc:
                event = {
                    "type": "error",
                    "ok": False,
                    "error": str(exc),
                    "child_run_id": child.run_id,
                    "run_id": parent.run_id,
                }
                try:
                    final_state = self._runtime.resume(
                        workflow=workflow,
                        run_id=child.run_id,
                        wait_key=wait_key,
                        payload=_final_result_for_event(event),
                        max_steps=100,
                    )
                    event["child_run_status"] = str(final_state.status.value if hasattr(final_state.status, "value") else final_state.status)
                except Exception:
                    cancel_run = getattr(self._runtime, "cancel_run", None)
                    if callable(cancel_run):
                        try:
                            cancel_run(child.run_id, reason=str(exc))
                        except Exception:
                            pass
                yield event

        return _events()

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
            _validate_music_output_spec(output)
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

    def send_email(
        self,
        run_id: str,
        *,
        to: Any,
        subject: str,
        account: Optional[str] = None,
        body_text: Optional[str] = None,
        body_html: Optional[str] = None,
        cc: Any = None,
        bcc: Any = None,
        timeout_s: float = 30.0,
        headers: Optional[Dict[str, str]] = None,
        child_vars: Optional[Dict[str, Any]] = None,
        workflow_id: Optional[str] = None,
    ) -> RunState:
        """Create a durable child run for one outbound email send."""

        tool_call = {
            "name": "send_email",
            "call_id": "send_email",
            "arguments": {
                "to": copy.deepcopy(to),
                "subject": str(subject),
                "account": account,
                "body_text": body_text,
                "body_html": body_html,
                "cc": copy.deepcopy(cc),
                "bcc": copy.deepcopy(bcc),
                "timeout_s": float(timeout_s),
                "headers": copy.deepcopy(headers) if isinstance(headers, dict) else headers,
            },
        }
        return self.execute_tool_calls(
            run_id,
            tool_calls=[tool_call],
            allowed_tools=["send_email"],
            child_vars=child_vars,
            workflow_id=workflow_id,
        )

    def send_telegram_message(
        self,
        run_id: str,
        *,
        chat_id: int,
        text: str,
        parse_mode: str = "",
        disable_web_page_preview: bool = False,
        timeout_s: float = 20.0,
        bot_token_env_var: str = "ABSTRACT_TELEGRAM_BOT_TOKEN",
        child_vars: Optional[Dict[str, Any]] = None,
        workflow_id: Optional[str] = None,
    ) -> RunState:
        """Create a durable child run for one outbound Telegram message."""

        tool_call = {
            "name": "send_telegram_message",
            "call_id": "send_telegram_message",
            "arguments": {
                "chat_id": int(chat_id),
                "text": str(text),
                "parse_mode": str(parse_mode),
                "disable_web_page_preview": bool(disable_web_page_preview),
                "timeout_s": float(timeout_s),
                "bot_token_env_var": str(bot_token_env_var),
            },
        }
        return self.execute_tool_calls(
            run_id,
            tool_calls=[tool_call],
            allowed_tools=["send_telegram_message"],
            child_vars=child_vars,
            workflow_id=workflow_id,
        )


def get_abstractcore_run_facade(runtime: Any) -> AbstractCoreRunFacade:
    """Return the public AbstractCore durable run facade bound to a runtime."""

    return AbstractCoreRunFacade.from_runtime(runtime)


__all__ = [
    "AbstractCoreRunFacade",
    "get_abstractcore_run_facade",
]
