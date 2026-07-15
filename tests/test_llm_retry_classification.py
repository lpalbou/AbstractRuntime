"""Deterministic LLM client errors must not be retried (2026-07-09 production incident:
a permanent OVH 400 burned 3 attempts — latency + cost — before surfacing identically).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from abstractruntime import Effect, EffectType, InMemoryLedgerStore, InMemoryRunStore, StepPlan, WorkflowSpec
from abstractruntime.core.policy import RetryPolicy
from abstractruntime.core.runtime import EffectOutcome, Runtime
from abstractruntime.integrations.abstractcore.effect_handlers import _llm_error_is_retryable


def test_classifier_types_and_message_fallback():
    from abstractcore.exceptions import AuthenticationError, InvalidRequestError, ModelNotFoundError

    assert _llm_error_is_retryable(InvalidRequestError("Invalid request: Error code: 400 - {...}")) is False
    assert _llm_error_is_retryable(AuthenticationError("bad key")) is False
    assert _llm_error_is_retryable(ModelNotFoundError("no such model")) is False
    # Our own providers' message format (type lost, e.g. across layers): 4xx non-retryable...
    assert _llm_error_is_retryable(RuntimeError(
        'OpenAI-compatible server API error (400): {"error":{"message":"System message must be at the beginning."}}'
    )) is False
    # ...while rate limits and server errors stay retryable.
    assert _llm_error_is_retryable(RuntimeError("OpenAI-compatible server API error (429): slow down")) is True
    assert _llm_error_is_retryable(RuntimeError("OpenAI-compatible server API error (500): boom")) is True
    assert _llm_error_is_retryable(RuntimeError("connection reset by peer")) is True


def test_prompt_cache_binding_failures_are_not_retried():
    """Bloc-seam adversary A-2 (2026-07-13): binding verification failures
    (missing/mismatch/invalid) and capability refusals are deterministic —
    the same params meet the same refusal every attempt. Generic operation
    failures (I/O during load/save) may be transient and stay retryable."""
    from abstractcore.providers.base import PromptCacheError, PromptCacheUnsupportedError

    for code in (
        "prompt_cache_binding_missing",
        "prompt_cache_binding_mismatch",
        "prompt_cache_binding_invalid",
        "prompt_cache_binding_invalid_key",
        "prompt_cache_binding_bare_string",
    ):
        exc = PromptCacheError("refused", operation="generate", code=code)
        assert _llm_error_is_retryable(exc) is False, code

    assert _llm_error_is_retryable(PromptCacheUnsupportedError(operation="load")) is False
    # Operation failures keep the retryable default (may be transient I/O).
    assert (
        _llm_error_is_retryable(
            PromptCacheError("disk hiccup", operation="save", code="prompt_cache_operation_failed")
        )
        is True
    )


def test_non_retryable_outcome_stops_the_retry_loop():
    attempts: List[int] = []

    def llm_handler(run, effect, default_next_node) -> EffectOutcome:
        del run, effect, default_next_node
        attempts.append(1)
        return EffectOutcome.failed(
            'OpenAI-compatible server API error (400): {"error":{"message":"System message must be at the beginning."}}',
            retryable=False,
        )

    def node(run, ctx) -> StepPlan:
        del ctx
        return StepPlan(
            node_id="n",
            effect=Effect(type=EffectType.LLM_CALL, payload={"prompt": "hi"}, result_key="out"),
            next_node="n",
        )

    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.LLM_CALL: llm_handler},
        effect_policy=RetryPolicy(llm_max_attempts=3, tool_max_attempts=1),
    )
    workflow = WorkflowSpec(workflow_id="retry_test", entry_node="n", nodes={"n": node})
    run_id = runtime.start(workflow=workflow)
    state = runtime.tick(workflow=workflow, run_id=run_id, max_steps=3)

    assert len(attempts) == 1, f"deterministic 400 was retried: {len(attempts)} attempts"
    assert str(state.status).endswith("FAILED") or state.error


def test_transient_failure_still_retries():
    attempts: List[int] = []

    def llm_handler(run, effect, default_next_node) -> EffectOutcome:
        del run, effect, default_next_node
        attempts.append(1)
        if len(attempts) < 3:
            return EffectOutcome.failed("connection reset by peer")  # retryable default
        return EffectOutcome.completed({"content": "ok"})

    def node(run, ctx) -> StepPlan:
        del ctx
        if (run.vars or {}).get("out"):
            return StepPlan(node_id="n", complete_output={"done": True})
        return StepPlan(
            node_id="n",
            effect=Effect(type=EffectType.LLM_CALL, payload={"prompt": "hi"}, result_key="out"),
            next_node="n",
        )

    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.LLM_CALL: llm_handler},
        effect_policy=RetryPolicy(llm_max_attempts=3, tool_max_attempts=1),
    )
    workflow = WorkflowSpec(workflow_id="retry_test2", entry_node="n", nodes={"n": node})
    run_id = runtime.start(workflow=workflow)
    runtime.tick(workflow=workflow, run_id=run_id, max_steps=5)

    assert len(attempts) == 3  # two transient failures + one success


def test_status_code_attribute_wins_over_prose():
    from abstractcore.exceptions import ProviderAPIError

    # 4xx via attribute -> non-retryable even when the message has no recognizable dialect.
    err = ProviderAPIError("something opaque from a proxy", status_code=422)
    assert _llm_error_is_retryable(err) is False
    # Transient 4xx stay retryable: model-load conflicts (LM Studio JIT), timeouts, rate limits.
    for code in (408, 409, 425, 429):
        assert _llm_error_is_retryable(ProviderAPIError("busy", status_code=code)) is True
    # 5xx via attribute -> retryable.
    assert _llm_error_is_retryable(ProviderAPIError("boom", status_code=503)) is True


def test_native_openai_sdk_dialect_matches():
    # Native OpenAI wrap sites can lose the type; the SDK message dialect must still classify.
    assert _llm_error_is_retryable(RuntimeError(
        "OpenAI API error: Invalid request: Error code: 400 - {'error': {...}}"
    )) is False


def test_harmony_generation_artifact_is_retryable_despite_400():
    """gpt-oss on vLLM (maintainer directive 2026-07-09): the server's strict
    openai-harmony parser 400s when the MODEL'S OWN sampled output violates
    its template (e.g. an unclosed `to=tool` recipient header). The request
    is valid and a resample usually passes — a sampling race, never an
    invalid request. 21 of these in one night of Castor's own time."""
    from abstractcore.exceptions import InvalidRequestError, ProviderAPIError

    # Even wrapped as InvalidRequestError with a 400 attribute (older
    # abstractcore), the message signature must win.
    err = InvalidRequestError(
        'OpenAI-compatible server API error (400): unexpected tokens '
        'remaining in message header: Some("to=tool")'
    )
    err.status_code = 400
    assert _llm_error_is_retryable(err) is True
    # The new abstractcore mapping (ProviderAPIError with the transient label).
    err2 = ProviderAPIError(
        "OpenAI-compatible server API error (400): unexpected tokens remaining "
        "in message header [transient harmony generation artifact - the model's "
        "sampled output violated its template; a retry resamples]",
        status_code=400,
    )
    assert _llm_error_is_retryable(err2) is True
    # A plain 400 stays non-retryable (the deterministic-error rule holds).
    err3 = InvalidRequestError("OpenAI-compatible server API error (400): missing field 'messages'")
    err3.status_code = 400
    assert _llm_error_is_retryable(err3) is False
