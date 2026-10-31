"""Opt-in conversational questions, beside unchanged legacy ASK_USER waits.

The runtime owns identity and absolute deadlines.  A fallback is a host
continuation instruction, never an answer spoken by the person.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, Optional
import uuid

from .models import WaitReason, WaitState


QUESTION_HISTORY_LIMIT = 64


def question_of(wait: Optional[WaitState]) -> Optional[Dict[str, Any]]:
    if wait is None or wait.reason != WaitReason.USER or not isinstance(wait.details, dict):
        return None
    value = wait.details.get("question")
    return value if isinstance(value, dict) and value.get("id") else None


def make_question(policy: Any, *, published_at: str, issuance_id: Optional[str] = None) -> Dict[str, Any]:
    if not isinstance(policy, dict) or set(policy) - {"importance", "timeout_s", "fallback"}:
        raise ValueError("question must contain importance and optional timeout_s/fallback")
    importance = policy.get("importance")
    if importance not in ("essential", "clarification"):
        raise ValueError("question.importance must be essential or clarification")
    timeout = policy.get("timeout_s")
    fallback = policy.get("fallback")
    deadline = None
    if importance == "essential":
        if timeout is not None or fallback is not None:
            raise ValueError("essential questions cannot have a timeout or fallback")
    else:
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 120 <= timeout <= 300:
            raise ValueError("clarification question.timeout_s must be between 120 and 300 seconds")
        if not isinstance(fallback, str) or not fallback.strip():
            raise ValueError("clarification question requires a non-empty declared fallback")
        deadline = (datetime.fromisoformat(published_at) + timedelta(seconds=timeout)).isoformat()
        fallback = fallback.strip()
    return {
        "id": issuance_id or str(uuid.uuid4()), "purpose": "conversation", "importance": importance,
        "published_at": published_at, "deadline": deadline, "fallback": fallback,
    }


def resolution_payload(question: Dict[str, Any], outcome: str) -> Dict[str, Any]:
    value: Dict[str, Any] = {"question": {"id": question["id"], "outcome": outcome}}
    if outcome == "expired":
        value.update(timed_out=True, fallback=question["fallback"])
    return value
