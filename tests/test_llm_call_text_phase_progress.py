"""A plain TEXT llm_call gets the durable progress channel too.

Generated media was the first caller of `abstract.progress`; a local model's
prefill is the same problem in the small. These pin the runtime half of the
contract:

* every LLM_CALL — not only generated-media ones — is offered an `on_progress`
  callback, so a provider that CAN report the prefill/generation boundary has
  somewhere to report it;
* each callback becomes exactly one durable `abstract.progress` EMIT_EVENT
  record scoped to the run that made the call, carrying the provider's payload
  verbatim under the `kind: "llm"` discriminator a client filters on;
* the callable itself never lands in the ledger — not in the llm_call effect's
  params, and not in the persisted `_provider_request` trace.
"""

from __future__ import annotations

from abstractruntime import Effect, EffectType, Runtime, StepPlan, WorkflowSpec
from abstractruntime.integrations.abstractcore.effect_handlers import build_effect_handlers
from abstractruntime.integrations.abstractcore.llm_client import LocalAbstractCoreLLMClient
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore


PHASE_EVENTS = [
    {"kind": "llm", "phase": "prefill", "event_index": 0, "elapsed_s": 0.045, "provider": "mlx",
     "model": "mlx-community/Qwen3.5-4B-4bit", "prompt_tokens": 808, "cached_tokens": 690,
     "fed_tokens": 118, "generated_tokens": 0},
    {"kind": "llm", "phase": "generate", "event_index": 1, "elapsed_s": 0.153, "generated_tokens": 1,
     "first_token": True, "ttft_s": 0.153},
    {"kind": "llm", "phase": "complete", "event_index": 2, "elapsed_s": 3.663, "generated_tokens": 551,
     "tokens_per_second": 157.37, "final": True, "finish_reason": "stop"},
]


class _PhaseLLM:
    """A provider that reports phases, as MLX does."""

    def __init__(self) -> None:
        self.seen_kwargs: dict = {}

    def generate(self, **kwargs):
        self.seen_kwargs = dict(kwargs)
        callback = kwargs.get("on_progress")
        assert callable(callback), "the runtime did not offer a progress callback to a text llm_call"
        for event in PHASE_EVENTS:
            callback(dict(event))

        class _Response:
            content = "A ledger is an append-only record of steps."
            model = "mlx-community/Qwen3.5-4B-4bit"
            finish_reason = "stop"
            usage = {"prompt_tokens": 808, "output_tokens": 551, "total_tokens": 1359}
            metadata: dict = {}
            tool_calls: list = []

        return _Response()


def _client(llm) -> LocalAbstractCoreLLMClient:
    client = object.__new__(LocalAbstractCoreLLMClient)
    client._provider = "mlx"
    client._model = "mlx-community/Qwen3.5-4B-4bit"
    client._artifact_store = None
    client._generate_lock = None
    client._capability_defaults = {}
    client._llm = llm
    client._maybe_prepare_prompt_cache = lambda **_kwargs: None
    return client


def _run_text_call(llm):
    ledger = InMemoryLedgerStore()
    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=ledger,
        effect_handlers=build_effect_handlers(llm=_client(llm)),
    )

    def reason(run, ctx):
        del run, ctx
        return StepPlan(
            node_id="reason",
            effect=Effect(
                type=EffectType.LLM_CALL,
                payload={"prompt": "What is a ledger?", "params": {"max_output_tokens": 600}},
                result_key="result",
            ),
            next_node="done",
        )

    def done(run, ctx):
        del ctx
        return StepPlan(node_id="done", complete_output={"result": run.vars.get("result")})

    workflow = WorkflowSpec(workflow_id="text_phase", entry_node="reason",
                            nodes={"reason": reason, "done": done})
    run_id = runtime.start(workflow=workflow, vars={})
    state = runtime.tick(workflow=workflow, run_id=run_id)
    assert state.status.value == "completed"
    return ledger.list(run_id)


def _progress_payloads(records):
    out = []
    for record in records:
        payload = (record.get("effect") or {}).get("payload") or {}
        if payload.get("name") == "abstract.progress":
            out.append(payload.get("payload") or {})
    return out


def test_a_text_llm_call_writes_one_durable_record_per_phase_event() -> None:
    records = _run_text_call(_PhaseLLM())
    payloads = _progress_payloads(records)

    assert len(payloads) == len(PHASE_EVENTS), "one ledger record per provider event, no more and no fewer"
    assert [p["phase"] for p in payloads] == ["prefill", "generate", "complete"]
    assert all(p["kind"] == "llm" for p in payloads)

    prefill = payloads[0]
    assert prefill["prompt_tokens"] == 808
    assert prefill["cached_tokens"] == 690
    assert prefill["fed_tokens"] == 118
    assert prefill["provider"] == "mlx"
    # Runtime identity is stamped on so a client can attribute the phase to the
    # step it belongs to without joining against anything else.
    for key in ("run_id", "workflow_id", "node_id", "step_id", "attempt"):
        assert key in prefill
    assert prefill["node_id"] == "reason"

    assert payloads[1]["first_token"] is True
    assert payloads[1]["ttft_s"] == 0.153
    assert payloads[2]["final"] is True
    assert payloads[2]["tokens_per_second"] == 157.37


def test_progress_records_are_run_scoped_emit_events() -> None:
    records = _run_text_call(_PhaseLLM())
    progress = [
        r for r in records
        if ((r.get("effect") or {}).get("payload") or {}).get("name") == "abstract.progress"
    ]
    assert progress
    for record in progress:
        assert (record.get("effect") or {}).get("type") == "emit_event"
        assert ((record.get("effect") or {}).get("payload") or {}).get("scope") == "run"
        assert record.get("status") == "completed"
        assert (record.get("result") or {}).get("emitted") is True


def test_the_callback_never_lands_in_the_ledger() -> None:
    llm = _PhaseLLM()
    records = _run_text_call(llm)
    llm_records = [r for r in records if (r.get("effect") or {}).get("type") == "llm_call"]
    assert llm_records
    for record in llm_records:
        params = ((record.get("effect") or {}).get("payload") or {}).get("params") or {}
        assert "on_progress" not in params

    # ... and not in the provider-request trace the runtime stamps for /llm --verbatim.
    assert "on_progress" in llm.seen_kwargs  # the provider really was offered one
    completed = [r for r in llm_records if r.get("status") == "completed"]
    assert completed
    trace = ((completed[-1].get("result") or {}).get("metadata") or {}).get("_provider_request") or {}
    trace_params = ((trace.get("payload") or {}).get("params")) or {}
    assert "on_progress" not in trace_params
    assert not any(callable(value) for value in trace_params.values())


def test_a_provider_with_no_phase_signal_leaves_the_ledger_clean() -> None:
    class _SilentLLM:
        def generate(self, **kwargs):
            assert callable(kwargs.get("on_progress"))  # offered...

            class _Response:  # ...and simply never called
                content = "done"
                model = "gpt-4o-mini"
                finish_reason = "stop"
                usage = {}
                metadata: dict = {}
                tool_calls: list = []

            return _Response()

    records = _run_text_call(_SilentLLM())
    assert _progress_payloads(records) == [], "a silent provider must not produce progress records"
