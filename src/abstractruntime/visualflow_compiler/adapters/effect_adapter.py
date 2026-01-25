"""Adapter for creating effect nodes in visual flows.

This adapter creates node handlers that produce AbstractRuntime Effects,
enabling visual flows to pause and wait for external input (user prompts,
events, delays, etc.).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from abstractruntime.core.models import RunState, StepPlan


def create_ask_user_handler(
    node_id: str,
    next_node: Optional[str],
    input_key: Optional[str] = None,
    output_key: Optional[str] = None,
    allow_free_text: bool = True,
) -> Callable:
    """Create a node handler that asks the user for input.

    This handler produces an ASK_USER effect that pauses the flow
    until the user provides a response.

    Args:
        node_id: Unique identifier for this node
        next_node: ID of the next node to transition to after response
        input_key: Key in run.vars to read prompt/choices from
        output_key: Key in run.vars to write the response to
        allow_free_text: Whether to allow free text response

    Returns:
        A node handler that produces ASK_USER effect
    """
    from abstractruntime.core.models import StepPlan, Effect, EffectType

    def handler(run: "RunState", ctx: Any) -> "StepPlan":
        """Ask user and wait for response."""
        # Get input from vars
        if input_key:
            input_data = run.vars.get(input_key, {})
        else:
            input_data = run.vars

        # Extract prompt and choices
        if isinstance(input_data, dict):
            prompt = input_data.get("prompt", "Please respond:")
            choices = input_data.get("choices", [])
        else:
            prompt = str(input_data) if input_data else "Please respond:"
            choices = []

        # Ensure choices is a list
        if not isinstance(choices, list):
            choices = []

        # Create the effect
        effect = Effect(
            type=EffectType.ASK_USER,
            payload={
                "prompt": prompt,
                "choices": choices,
                "allow_free_text": allow_free_text,
            },
            result_key=output_key or "_temp.user_response",
        )

        return StepPlan(
            node_id=node_id,
            effect=effect,
            next_node=next_node,
        )

    return handler


def create_answer_user_handler(
    node_id: str,
    next_node: Optional[str],
    input_key: Optional[str] = None,
    output_key: Optional[str] = None,
) -> Callable:
    """Create a node handler that requests the host UI to display a message.

    This handler produces an ANSWER_USER effect that completes immediately.
    """
    from abstractruntime.core.models import StepPlan, Effect, EffectType

    def handler(run: "RunState", ctx: Any) -> "StepPlan":
        if input_key:
            input_data = run.vars.get(input_key, {})
        else:
            input_data = run.vars

        if isinstance(input_data, dict):
            message = input_data.get("message") or input_data.get("text") or ""
            level_raw = input_data.get("level")
        else:
            message = str(input_data) if input_data is not None else ""
            level_raw = None

        level = str(level_raw).strip().lower() if isinstance(level_raw, str) else ""
        if level == "warn":
            level = "warning"
        if level == "info":
            level = "message"
        if level not in {"message", "warning", "error"}:
            level = "message"

        effect = Effect(
            type=EffectType.ANSWER_USER,
            payload={"message": str(message), "level": level},
            result_key=output_key or "_temp.answer_user",
        )

        return StepPlan(
            node_id=node_id,
            effect=effect,
            next_node=next_node,
        )

    return handler


def create_wait_until_handler(
    node_id: str,
    next_node: Optional[str],
    input_key: Optional[str] = None,
    output_key: Optional[str] = None,
    duration_type: str = "seconds",
) -> Callable:
    """Create a node handler that waits until a specified time.

    Args:
        node_id: Unique identifier for this node
        next_node: ID of the next node to transition to after waiting
        input_key: Key in run.vars to read duration from
        output_key: Key in run.vars to write the completion info to
        duration_type: How to interpret duration (seconds/minutes/hours/timestamp)

    Returns:
        A node handler that produces WAIT_UNTIL effect
    """
    from datetime import datetime, timedelta, timezone
    from abstractruntime.core.models import StepPlan, Effect, EffectType

    def handler(run: "RunState", ctx: Any) -> "StepPlan":
        """Wait until time and then continue."""
        # Get input from vars
        if input_key:
            input_data = run.vars.get(input_key, {})
        else:
            input_data = run.vars

        # Extract duration
        if isinstance(input_data, dict):
            duration = input_data.get("duration", 0)
        else:
            duration = input_data

        # Convert to seconds
        try:
            amount = float(duration) if duration else 0
        except (TypeError, ValueError):
            amount = 0

        # Calculate target time
        now = datetime.now(timezone.utc)

        if duration_type == "timestamp":
            # Already an ISO timestamp
            until = str(duration)
        elif duration_type == "minutes":
            until = (now + timedelta(minutes=amount)).isoformat()
        elif duration_type == "hours":
            until = (now + timedelta(hours=amount)).isoformat()
        else:  # seconds
            until = (now + timedelta(seconds=amount)).isoformat()

        # Create the effect
        effect = Effect(
            type=EffectType.WAIT_UNTIL,
            payload={"until": until},
            result_key=output_key or "_temp.wait_result",
        )

        return StepPlan(
            node_id=node_id,
            effect=effect,
            next_node=next_node,
        )

    return handler


def create_wait_event_handler(
    node_id: str,
    next_node: Optional[str],
    input_key: Optional[str] = None,
    output_key: Optional[str] = None,
) -> Callable:
    """Create a node handler that waits for an external event.

    Args:
        node_id: Unique identifier for this node
        next_node: ID of the next node to transition to after event
        input_key: Key in run.vars to read event_key from
        output_key: Key in run.vars to write the event data to

    Returns:
        A node handler that produces WAIT_EVENT effect
    """
    from abstractruntime.core.models import StepPlan, Effect, EffectType

    def handler(run: "RunState", ctx: Any) -> "StepPlan":
        """Wait for event and then continue."""
        # Get input from vars
        if input_key:
            input_data = run.vars.get(input_key, {})
        else:
            input_data = run.vars

        # Extract event key + optional host UX fields (prompt/choices).
        if isinstance(input_data, dict):
            event_key = input_data.get("event_key")
            if event_key is None:
                event_key = input_data.get("wait_key")
            if not event_key:
                event_key = "default"
            prompt = input_data.get("prompt")
            choices = input_data.get("choices")
            allow_free_text = input_data.get("allow_free_text")
            if allow_free_text is None:
                allow_free_text = input_data.get("allowFreeText")
        else:
            event_key = str(input_data) if input_data else "default"
            prompt = None
            choices = None
            allow_free_text = None

        # Create the effect
        effect = Effect(
            type=EffectType.WAIT_EVENT,
            payload={
                "wait_key": str(event_key),
                **({"prompt": prompt} if isinstance(prompt, str) and prompt.strip() else {}),
                **({"choices": choices} if isinstance(choices, list) else {}),
                **({"allow_free_text": bool(allow_free_text)} if allow_free_text is not None else {}),
            },
            result_key=output_key or "_temp.event_data",
        )

        return StepPlan(
            node_id=node_id,
            effect=effect,
            next_node=next_node,
        )

    return handler


def create_memory_note_handler(
    node_id: str,
    next_node: Optional[str],
    input_key: Optional[str] = None,
    output_key: Optional[str] = None,
) -> Callable:
    """Create a node handler that stores a memory note.

    Args:
        node_id: Unique identifier for this node
        next_node: ID of the next node to transition to after storing
        input_key: Key in run.vars to read note content from
        output_key: Key in run.vars to write the note_id to

    Returns:
        A node handler that produces MEMORY_NOTE effect
    """
    from abstractruntime.core.models import StepPlan, Effect, EffectType

    def handler(run: "RunState", ctx: Any) -> "StepPlan":
        """Store memory note and continue."""
        # Get input from vars
        if input_key:
            input_data = run.vars.get(input_key, {})
        else:
            input_data = run.vars

        # Extract content
        if isinstance(input_data, dict):
            content = input_data.get("content", "")
            tags = input_data.get("tags") if isinstance(input_data.get("tags"), dict) else {}
            sources = input_data.get("sources") if isinstance(input_data.get("sources"), dict) else None
            scope = input_data.get("scope") if isinstance(input_data.get("scope"), str) else None
            location = input_data.get("location") if isinstance(input_data.get("location"), str) else None
            keep_in_context = input_data.get("keep_in_context")
            if keep_in_context is None:
                keep_in_context = input_data.get("keepInContext")
        else:
            content = str(input_data) if input_data else ""
            tags = {}
            sources = None
            scope = None
            location = None
            keep_in_context = None

        # Create the effect
        payload: Dict[str, Any] = {"note": content, "tags": tags}
        if sources is not None:
            payload["sources"] = sources
        if scope:
            payload["scope"] = scope
        if isinstance(location, str) and location.strip():
            payload["location"] = location.strip()
        if keep_in_context is not None:
            payload["keep_in_context"] = keep_in_context

        effect = Effect(
            type=EffectType.MEMORY_NOTE,
            payload=payload,
            result_key=output_key or "_temp.note_id",
        )

        return StepPlan(
            node_id=node_id,
            effect=effect,
            next_node=next_node,
        )

    return handler


def create_memory_query_handler(
    node_id: str,
    next_node: Optional[str],
    input_key: Optional[str] = None,
    output_key: Optional[str] = None,
) -> Callable:
    """Create a node handler that queries memory.

    Args:
        node_id: Unique identifier for this node
        next_node: ID of the next node to transition to after query
        input_key: Key in run.vars to read query from
        output_key: Key in run.vars to write results to

    Returns:
        A node handler that produces MEMORY_QUERY effect
    """
    from abstractruntime.core.models import StepPlan, Effect, EffectType

    def handler(run: "RunState", ctx: Any) -> "StepPlan":
        """Query memory and continue."""
        # Get input from vars
        if input_key:
            input_data = run.vars.get(input_key, {})
        else:
            input_data = run.vars

        # Extract query params
        if isinstance(input_data, dict):
            query = input_data.get("query", "")
            limit = input_data.get("limit", 10)
            tags = input_data.get("tags") if isinstance(input_data.get("tags"), dict) else None
            tags_mode = input_data.get("tags_mode")
            if tags_mode is None:
                tags_mode = input_data.get("tagsMode")
            usernames = input_data.get("usernames")
            locations = input_data.get("locations")
            since = input_data.get("since")
            until = input_data.get("until")
            scope = input_data.get("scope") if isinstance(input_data.get("scope"), str) else None
        else:
            query = str(input_data) if input_data else ""
            limit = 10
            tags = None
            tags_mode = None
            usernames = None
            locations = None
            since = None
            until = None
            scope = None

        def _normalize_str_list(raw: Any) -> Optional[List[str]]:
            if raw is None:
                return None
            if isinstance(raw, str):
                s = raw.strip()
                return [s] if s else None
            if not isinstance(raw, list):
                return None
            out: List[str] = []
            for x in raw:
                if isinstance(x, str) and x.strip():
                    out.append(x.strip())
            return out or None

        # Create the effect
        payload: Dict[str, Any] = {"query": query, "limit_spans": limit, "return": "both"}
        if tags is not None:
            payload["tags"] = tags
        if isinstance(tags_mode, str) and tags_mode.strip():
            payload["tags_mode"] = tags_mode.strip()
        usernames_list = _normalize_str_list(usernames)
        if usernames_list is not None:
            payload["usernames"] = usernames_list
        locations_list = _normalize_str_list(locations)
        if locations_list is not None:
            payload["locations"] = locations_list
        if since is not None:
            payload["since"] = since
        if until is not None:
            payload["until"] = until
        if scope:
            payload["scope"] = scope

        effect = Effect(
            type=EffectType.MEMORY_QUERY,
            payload=payload,
            result_key=output_key or "_temp.memory_results",
        )

        return StepPlan(
            node_id=node_id,
            effect=effect,
            next_node=next_node,
        )

    return handler


def create_memory_kg_assert_handler(
    node_id: str,
    next_node: Optional[str],
    input_key: Optional[str] = None,
    output_key: Optional[str] = None,
) -> Callable:
    """Create a node handler that asserts triples into AbstractMemory (host-provided handler)."""
    from abstractruntime.core.models import StepPlan, Effect, EffectType

    def _normalize_assertions(raw: Any) -> list[Dict[str, Any]]:
        if raw is None:
            return []
        if isinstance(raw, dict):
            return [dict(raw)]
        if isinstance(raw, list):
            out: list[Dict[str, Any]] = []
            for x in raw:
                if isinstance(x, dict):
                    out.append(dict(x))
            return out
        return []

    def handler(run: "RunState", ctx: Any) -> "StepPlan":
        del ctx
        if input_key:
            input_data = run.vars.get(input_key, {})
        else:
            input_data = run.vars

        assertions_raw: Any = None
        scope: Optional[str] = None
        owner_id: Optional[str] = None
        span_id: Optional[str] = None
        attributes_defaults: Optional[Dict[str, Any]] = None
        allow_custom_predicates: Optional[bool] = None
        if isinstance(input_data, dict):
            assertions_raw = input_data.get("assertions")
            if assertions_raw is None:
                assertions_raw = input_data.get("triples")
            if assertions_raw is None:
                assertions_raw = input_data.get("items")
            scope = input_data.get("scope") if isinstance(input_data.get("scope"), str) else None
            owner_id = input_data.get("owner_id") if isinstance(input_data.get("owner_id"), str) else None
            span_id = input_data.get("span_id") if isinstance(input_data.get("span_id"), str) else None
            attributes_defaults = input_data.get("attributes_defaults") if isinstance(input_data.get("attributes_defaults"), dict) else None
            allow_custom_predicates = (
                input_data.get("allow_custom_predicates")
                if isinstance(input_data.get("allow_custom_predicates"), bool)
                else input_data.get("allow_custom")
                if isinstance(input_data.get("allow_custom"), bool)
                else None
            )

        assertions = _normalize_assertions(assertions_raw)
        payload: Dict[str, Any] = {"assertions": assertions}
        if scope:
            payload["scope"] = scope
        if owner_id:
            payload["owner_id"] = owner_id
        if span_id:
            payload["span_id"] = span_id
        if attributes_defaults:
            payload["attributes_defaults"] = dict(attributes_defaults)
        if allow_custom_predicates is not None:
            payload["allow_custom_predicates"] = bool(allow_custom_predicates)

        return StepPlan(
            node_id=node_id,
            effect=Effect(type=EffectType.MEMORY_KG_ASSERT, payload=payload, result_key=output_key or "_temp.memory_kg_assert"),
            next_node=next_node,
        )

    return handler


def create_memory_kg_query_handler(
    node_id: str,
    next_node: Optional[str],
    input_key: Optional[str] = None,
    output_key: Optional[str] = None,
) -> Callable:
    """Create a node handler that queries AbstractMemory triples (host-provided handler)."""
    from abstractruntime.core.models import StepPlan, Effect, EffectType

    def handler(run: "RunState", ctx: Any) -> "StepPlan":
        del ctx
        if input_key:
            input_data = run.vars.get(input_key, {})
        else:
            input_data = run.vars

        payload: Dict[str, Any] = {}
        if isinstance(input_data, dict):
            for k in (
                "subject",
                "predicate",
                "object",
                "scope",
                "owner_id",
                "since",
                "until",
                "active_at",
                "query_text",
                "order",
            ):
                v = input_data.get(k)
                if isinstance(v, str) and v.strip():
                    payload[k] = v.strip()
            min_score = input_data.get("min_score")
            if min_score is not None and not isinstance(min_score, bool):
                try:
                    payload["min_score"] = float(min_score)
                except Exception:
                    pass
            limit = input_data.get("limit")
            if limit is None:
                limit = input_data.get("limit_spans")
            if limit is not None and not isinstance(limit, bool):
                try:
                    payload["limit"] = int(limit)
                except Exception:
                    pass

        return StepPlan(
            node_id=node_id,
            effect=Effect(type=EffectType.MEMORY_KG_QUERY, payload=payload, result_key=output_key or "_temp.memory_kg_query"),
            next_node=next_node,
        )

    return handler


def create_memory_kg_resolve_handler(
    node_id: str,
    next_node: Optional[str],
    input_key: Optional[str] = None,
    output_key: Optional[str] = None,
) -> Callable:
    """Create a node handler that resolves entity candidates from AbstractMemory triples."""
    from abstractruntime.core.models import StepPlan, Effect, EffectType

    def handler(run: "RunState", ctx: Any) -> "StepPlan":
        del ctx
        if input_key:
            input_data = run.vars.get(input_key, {})
        else:
            input_data = run.vars

        payload: Dict[str, Any] = {}
        if isinstance(input_data, dict):
            for k in ("label", "expected_type", "scope", "owner_id", "recall_level"):
                v = input_data.get(k)
                if isinstance(v, str) and v.strip():
                    payload[k] = v.strip()

            min_score = input_data.get("min_score")
            if min_score is not None and not isinstance(min_score, bool):
                try:
                    payload["min_score"] = float(min_score)
                except Exception:
                    pass

            max_candidates = input_data.get("max_candidates")
            if max_candidates is None:
                max_candidates = input_data.get("limit")
            if max_candidates is not None and not isinstance(max_candidates, bool):
                try:
                    payload["max_candidates"] = int(max_candidates)
                except Exception:
                    pass

        return StepPlan(
            node_id=node_id,
            effect=Effect(type=EffectType.MEMORY_KG_RESOLVE, payload=payload, result_key=output_key or "_temp.memory_kg_resolve"),
            next_node=next_node,
        )

    return handler


def create_memory_tag_handler(
    node_id: str,
    next_node: Optional[str],
    input_key: Optional[str] = None,
    output_key: Optional[str] = None,
) -> Callable:
    """Create a node handler that applies tags to an existing memory span record."""
    from abstractruntime.core.models import StepPlan, Effect, EffectType

    def handler(run: "RunState", ctx: Any) -> "StepPlan":
        del ctx
        if input_key:
            input_data = run.vars.get(input_key, {})
        else:
            input_data = run.vars

        span_id: Any = None
        tags: Dict[str, Any] = {}
        merge: Optional[bool] = None
        scope: Optional[str] = None
        if isinstance(input_data, dict):
            span_id = input_data.get("span_id")
            if span_id is None:
                span_id = input_data.get("spanId")
            raw_tags = input_data.get("tags")
            tags = raw_tags if isinstance(raw_tags, dict) else {}
            if "merge" in input_data:
                merge = bool(input_data.get("merge"))
            if isinstance(input_data.get("scope"), str) and str(input_data.get("scope") or "").strip():
                scope = str(input_data.get("scope") or "").strip()

        payload: Dict[str, Any] = {"span_id": span_id, "tags": tags}
        if merge is not None:
            payload["merge"] = merge
        if scope is not None:
            payload["scope"] = scope

        return StepPlan(
            node_id=node_id,
            effect=Effect(type=EffectType.MEMORY_TAG, payload=payload, result_key=output_key or "_temp.memory_tag"),
            next_node=next_node,
        )

    return handler


def create_memory_compact_handler(
    node_id: str,
    next_node: Optional[str],
    input_key: Optional[str] = None,
    output_key: Optional[str] = None,
) -> Callable:
    """Create a node handler that requests runtime-owned memory compaction."""
    from abstractruntime.core.models import StepPlan, Effect, EffectType

    def handler(run: "RunState", ctx: Any) -> "StepPlan":
        del ctx
        if input_key:
            input_data = run.vars.get(input_key, {})
        else:
            input_data = run.vars

        preserve_recent: Optional[int] = None
        compression_mode: Optional[str] = None
        focus: Optional[str] = None
        if isinstance(input_data, dict):
            if input_data.get("preserve_recent") is not None:
                try:
                    preserve_recent = int(input_data.get("preserve_recent"))
                except Exception:
                    preserve_recent = None
            compression_mode = input_data.get("compression_mode") if isinstance(input_data.get("compression_mode"), str) else None
            focus = input_data.get("focus") if isinstance(input_data.get("focus"), str) else None

        payload: Dict[str, Any] = {}
        if preserve_recent is not None:
            payload["preserve_recent"] = preserve_recent
        if isinstance(compression_mode, str) and compression_mode.strip():
            payload["compression_mode"] = compression_mode.strip()
        if isinstance(focus, str) and focus.strip():
            payload["focus"] = focus.strip()

        return StepPlan(
            node_id=node_id,
            effect=Effect(type=EffectType.MEMORY_COMPACT, payload=payload, result_key=output_key or "_temp.memory_compact"),
            next_node=next_node,
        )

    return handler


def create_memory_rehydrate_handler(
    node_id: str,
    next_node: Optional[str],
    input_key: Optional[str] = None,
    output_key: Optional[str] = None,
) -> Callable:
    """Create a node handler that rehydrates recalled spans into context.messages.

    This produces a runtime-owned `EffectType.MEMORY_REHYDRATE` so rehydration is durable and host-agnostic.
    """
    from abstractruntime.core.models import StepPlan, Effect, EffectType

    def handler(run: "RunState", ctx: Any) -> "StepPlan":
        del ctx
        if input_key:
            input_data = run.vars.get(input_key, {})
        else:
            input_data = run.vars

        span_ids = []
        placement = "after_summary"
        max_messages = None
        if isinstance(input_data, dict):
            raw = input_data.get("span_ids")
            if raw is None:
                raw = input_data.get("span_id")
            if isinstance(raw, list):
                span_ids = list(raw)
            elif raw is not None:
                span_ids = [raw]
            if isinstance(input_data.get("placement"), str):
                placement = str(input_data.get("placement") or "").strip() or placement
            if input_data.get("max_messages") is not None:
                max_messages = input_data.get("max_messages")

        payload: Dict[str, Any] = {"span_ids": span_ids, "placement": placement}
        if max_messages is not None:
            payload["max_messages"] = max_messages

        return StepPlan(
            node_id=node_id,
            effect=Effect(
                type=EffectType.MEMORY_REHYDRATE,
                payload=payload,
                result_key=output_key or "_temp.memory_rehydrate",
            ),
            next_node=next_node,
        )

    return handler


def create_llm_call_handler(
    node_id: str,
    next_node: Optional[str],
    input_key: Optional[str] = None,
    output_key: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.7,
    seed: int = -1,
) -> Callable:
    """Create a node handler that makes an LLM call.

    Args:
        node_id: Unique identifier for this node
        next_node: ID of the next node to transition to after LLM response
        input_key: Key in run.vars to read prompt/system from
        output_key: Key in run.vars to write response to
        provider: LLM provider to use
        model: Model name to use
        temperature: Temperature parameter
        seed: Seed parameter (-1 means random/unset)

    Returns:
        A node handler that produces LLM_CALL effect
    """
    from abstractruntime.core.models import StepPlan, Effect, EffectType

    def handler(run: "RunState", ctx: Any) -> "StepPlan":
        """Make LLM call and continue."""
        # Get input from vars
        if input_key:
            input_data = run.vars.get(input_key, {})
        else:
            input_data = run.vars

        # Extract prompt and system
        if isinstance(input_data, dict):
            prompt = input_data.get("prompt", "")
            system = input_data.get("system", "")
        else:
            prompt = str(input_data) if input_data else ""
            system = ""

        # Build messages for LLM
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Create the effect
        effect = Effect(
            type=EffectType.LLM_CALL,
            payload={
                "messages": messages,
                "provider": provider,
                "model": model,
                "params": {
                    "temperature": float(temperature),
                },
            },
            result_key=output_key or "_temp.llm_response",
        )
        try:
            seed_i = int(seed)
        except Exception:
            seed_i = -1
        if seed_i >= 0:
            effect.payload["params"]["seed"] = seed_i

        return StepPlan(
            node_id=node_id,
            effect=effect,
            next_node=next_node,
        )

    return handler


def create_tool_calls_handler(
    node_id: str,
    next_node: Optional[str],
    input_key: Optional[str] = None,
    output_key: Optional[str] = None,
    allowed_tools: Optional[List[str]] = None,
) -> Callable:
    """Create a node handler that executes tool calls via AbstractRuntime.

    This produces a durable `EffectType.TOOL_CALLS` so tool execution stays runtime-owned.

    Inputs:
    - `tool_calls`: list[dict] (or a single dict) in the common shape
      `{name, arguments, call_id?}`.
    - Optional `allowed_tools`: list[str] allowlist. If provided as a list, the
      runtime effect handler enforces it (empty list => allow none).
    """
    from abstractruntime.core.models import StepPlan, Effect, EffectType

    def _normalize_tool_calls(raw: Any) -> list[Dict[str, Any]]:
        if raw is None:
            return []
        if isinstance(raw, dict):
            return [dict(raw)]
        if isinstance(raw, list):
            out: list[Dict[str, Any]] = []
            for x in raw:
                if isinstance(x, dict):
                    out.append(dict(x))
            return out
        return []

    def _normalize_str_list(raw: Any) -> list[str]:
        if not isinstance(raw, list):
            return []
        out: list[str] = []
        for x in raw:
            if isinstance(x, str) and x.strip():
                out.append(x.strip())
        return out

    def handler(run: "RunState", ctx: Any) -> "StepPlan":
        del ctx
        if input_key:
            input_data = run.vars.get(input_key, {})
        else:
            input_data = run.vars

        tool_calls: list[Dict[str, Any]] = []
        allowlist: Optional[list[str]] = list(allowed_tools) if isinstance(allowed_tools, list) else None

        if isinstance(input_data, dict):
            raw_calls = input_data.get("tool_calls")
            if raw_calls is None:
                raw_calls = input_data.get("toolCalls")
            tool_calls = _normalize_tool_calls(raw_calls)

            # Optional override when the input explicitly provides an allowlist.
            if "allowed_tools" in input_data or "allowedTools" in input_data:
                raw_allowed = input_data.get("allowed_tools")
                if raw_allowed is None:
                    raw_allowed = input_data.get("allowedTools")
                allowlist = _normalize_str_list(raw_allowed)
        else:
            tool_calls = _normalize_tool_calls(input_data)

        payload: Dict[str, Any] = {"tool_calls": tool_calls}
        if isinstance(allowlist, list):
            payload["allowed_tools"] = _normalize_str_list(allowlist)

        return StepPlan(
            node_id=node_id,
            effect=Effect(
                type=EffectType.TOOL_CALLS,
                payload=payload,
                result_key=output_key or "_temp.tool_calls",
            ),
            next_node=next_node,
        )

    return handler


def create_call_tool_handler(
    node_id: str,
    next_node: Optional[str],
    input_key: Optional[str] = None,
    output_key: Optional[str] = None,
    allowed_tools: Optional[List[str]] = None,
) -> Callable:
    """Create a node handler that executes a single tool call via AbstractRuntime.

    This is a convenience wrapper over `EffectType.TOOL_CALLS` that accepts a single
    tool call dict (instead of an array) and returns a 1-element tool_calls list.

    Inputs:
    - `tool_call`: dict in the common shape `{name, arguments, call_id?}`.
    - Optional `allowed_tools`: list[str] allowlist. If provided as a list, the
      runtime effect handler enforces it (empty list => allow none).
    """
    from abstractruntime.core.models import StepPlan, Effect, EffectType

    def _normalize_tool_call(raw: Any) -> Optional[Dict[str, Any]]:
        if raw is None:
            return None
        if isinstance(raw, dict):
            return dict(raw)
        return None

    def _normalize_str_list(raw: Any) -> list[str]:
        if not isinstance(raw, list):
            return []
        out: list[str] = []
        for x in raw:
            if isinstance(x, str) and x.strip():
                out.append(x.strip())
        return out

    def handler(run: "RunState", ctx: Any) -> "StepPlan":
        del ctx
        if input_key:
            input_data = run.vars.get(input_key, {})
        else:
            input_data = run.vars

        tool_call: Optional[Dict[str, Any]] = None
        allowlist: Optional[list[str]] = list(allowed_tools) if isinstance(allowed_tools, list) else None

        if isinstance(input_data, dict):
            raw_call = input_data.get("tool_call")
            if raw_call is None:
                raw_call = input_data.get("toolCall")
            tool_call = _normalize_tool_call(raw_call)

            # Optional override when the input explicitly provides an allowlist.
            if "allowed_tools" in input_data or "allowedTools" in input_data:
                raw_allowed = input_data.get("allowed_tools")
                if raw_allowed is None:
                    raw_allowed = input_data.get("allowedTools")
                allowlist = _normalize_str_list(raw_allowed)
        else:
            tool_call = _normalize_tool_call(input_data)

        payload: Dict[str, Any] = {"tool_calls": [tool_call] if isinstance(tool_call, dict) else []}
        if isinstance(allowlist, list):
            payload["allowed_tools"] = _normalize_str_list(allowlist)

        return StepPlan(
            node_id=node_id,
            effect=Effect(
                type=EffectType.TOOL_CALLS,
                payload=payload,
                result_key=output_key or "_temp.call_tool",
            ),
            next_node=next_node,
        )

    return handler


def create_start_subworkflow_handler(
    node_id: str,
    next_node: Optional[str],
    input_key: Optional[str] = None,
    output_key: Optional[str] = None,
    workflow_id: Optional[str] = None,
) -> Callable:
    """Create a node handler that starts a subworkflow by workflow id.

    This is the effect-level equivalent of `create_subflow_node_handler`, but it
    defers lookup/execution to the runtime's workflow registry.
    """
    from abstractruntime.core.models import StepPlan, Effect, EffectType

    def handler(run: "RunState", ctx: Any) -> "StepPlan":
        if not workflow_id:
            return StepPlan(
                node_id=node_id,
                complete_output={
                    "success": False,
                    "error": "start_subworkflow requires workflow_id (node config missing)",
                },
            )

        if input_key:
            input_data = run.vars.get(input_key, {})
        else:
            input_data = run.vars

        sub_vars: Dict[str, Any] = {}
        if isinstance(input_data, dict):
            # Prefer explicit "vars" field, else pass through common "input" field.
            if isinstance(input_data.get("vars"), dict):
                sub_vars = dict(input_data["vars"])
            elif isinstance(input_data.get("input"), dict):
                sub_vars = dict(input_data["input"])
            else:
                sub_vars = dict(input_data)
        else:
            sub_vars = {"input": input_data}

        return StepPlan(
            node_id=node_id,
            effect=Effect(
                type=EffectType.START_SUBWORKFLOW,
                payload={
                    "workflow_id": workflow_id,
                    "vars": sub_vars,
                    "async": False,
                },
                result_key=output_key or f"_temp.effects.{node_id}",
            ),
            next_node=next_node,
        )

    return handler
