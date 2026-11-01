"""The `decision` output modality (0.16.0): text in, calibrated typed answers out, no bytes.

A decision is an `LLM_CALL` whose output spec says `modality: "decision"`. The runtime adds no
effect type and no vocabulary of its own: AbstractCore's `output.decision` route resolves the
decision model, core's base provider runs it through its decision pool, and the normalized
result carries the System One body (`decision`) plus the `answers` shortcut. Pinned here, with
fakes only (no model is ever loaded):

- normalization from a `MultimodalGenerateResponse` with `.decision` (and from the body alone),
  JSON-serializable, needing no artifact store;
- the local client passes the RESOLVED spec through `generate(prompt, output=[spec])`, never the
  media-only fast path, never demanding an ArtifactStore;
- the pooled client serves a decision-only request without building the text client, and the
  effect handler completes it with no artifact store, action id `decision`;
- the remote client posts `/v1/decisions` and maps the server's body to the same result;
- the resolved action id, the (not generated-media) tag, and the `output.decision` route key.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from abstractruntime import Effect, EffectType, Runtime, StepPlan, WorkflowSpec
from abstractruntime.integrations.abstractcore.effect_handlers import (
    _augment_output_request_for_runtime,
    _resolved_action_id_from_output,
    build_effect_handlers,
)
from abstractruntime.integrations.abstractcore.llm_client import (
    LocalAbstractCoreLLMClient,
    MultiLocalAbstractCoreLLMClient,
    RemoteAbstractCoreLLMClient,
    _media_only_output_request,
    _normalize_local_response,
    _output_default_route_keys,
)
from abstractruntime.integrations.abstractcore.output_specs import (
    capability_default_route_keys_for_spec,
    output_request_has_generated_media,
    output_request_has_non_text_result,
)
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore

pytestmark = pytest.mark.basic

STATE = "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card."
QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {"returns": "Wrong or damaged items", "shipping": "Delays", "billing": "Charges"},
    },
    "escalate": {"type": "noul", "instructions": "Does this need urgent human attention?"},
    "frustration": {"type": "score", "instructions": "How frustrated is the customer?", "criteria": ["Calm", "Frustrated", "Very angry"]},
}
DECISION_ROUTE = {"output.decision": {"provider": "mlx", "model": "jaredpalmer/kev-4b"}}


def _decision_response(*, model: str = "jaredpalmer/kev-4b", provider: str = "mlx"):
    from abstractcore.decision.types import ChoiceAnswer, DecisionResponse, DecisionUsage, NoulAnswer, ScoreAnswer

    return DecisionResponse(
        model=model,
        answers={
            "department": ChoiceAnswer(choice="shipping", confidence=0.81, probabilities={"returns": 0.1, "shipping": 0.81, "billing": 0.09}),
            "escalate": NoulAnswer(noul=0.12),
            "frustration": ScoreAnswer(score=1.4, confidence=0.6, legend={"0": "Calm", "1": "Frustrated", "2": "Very angry"}, probabilities={"0": 0.1, "1": 0.4, "2": 0.5}),
        },
        usage=DecisionUsage(input_tokens=42, output_tokens=0),
        latency_ms=12.5,
        provider=provider,
        backend="fake",
        precision="8bit",
    )


def _core_shaped_response(response, *, text_provider_class: str = "CapabilityHostProvider"):
    """Exactly what `BaseProvider._run_decision_output` leaves on the response."""
    from abstractcore.core.multimodal_generation import MultimodalGenerateResponse

    wire = response.to_systemone()
    return MultimodalGenerateResponse(
        decision=response,
        metadata={
            "model": response.model,
            "provider": text_provider_class,
            "decision": wire,
            "content": json.dumps(wire["answers"], ensure_ascii=False),
            "decision_backend": response.backend,
            "decision_precision": response.precision,
        },
    )


# --------------------------------------------------------------------------- normalization


def test_normalize_decision_response_without_artifact_store() -> None:
    response = _decision_response()
    out = _normalize_local_response(_core_shaped_response(response))  # no artifact store at all

    body = response.to_systemone()
    assert out["decision"] == body
    assert out["answers"] == body["answers"]
    assert out["answers"]["department"]["choice"] == "shipping"
    assert out["answers"]["escalate"]["noul"] == pytest.approx(0.12)
    assert out["content"] == json.dumps(body["answers"], ensure_ascii=False)
    assert out["outputs"] == {} and out["resources"] == {}
    assert out["warnings"] == [] and out["errors"] == []
    # The DECISION model that answered, not the text provider that carried the call.
    assert out["provider"] == "mlx" and out["model"] == "jaredpalmer/kev-4b"
    assert out["usage"] == {"input_tokens": 42, "output_tokens": 0}
    assert out["metadata"]["decision"] == body
    assert out["metadata"]["provider"] == "CapabilityHostProvider"
    json.dumps(out)


def test_normalize_decision_truncation_rides_the_body() -> None:
    from abstractcore.core.multimodal_generation import MultimodalGenerateResponse

    response = _decision_response()
    response.truncated = True
    response.usage.state_tokens = 70000
    response.usage.state_tokens_used = 65536
    out = _normalize_local_response(MultimodalGenerateResponse(
        decision=response, metadata={"decision": response.to_systemone(), "decision_provider": "mlx"}))

    assert out["decision"]["truncated"] is True
    assert out["decision"]["usage"]["state_tokens_used"] == 65536
    assert out["answers"]["frustration"]["score"] == pytest.approx(1.4)
    assert out["content"] == json.dumps(out["answers"], ensure_ascii=False)
    assert out["provider"] == "mlx" and out["model"] == "jaredpalmer/kev-4b"
    json.dumps(out)


def test_a_response_without_a_decision_gains_no_decision_keys() -> None:
    from abstractcore.core.multimodal_generation import MultimodalGenerateResponse

    out = _normalize_local_response(MultimodalGenerateResponse(metadata={"model": "m"}))
    assert "decision" not in out and "answers" not in out


# --------------------------------------------------------------------------- local client


def test_local_client_passes_the_resolved_decision_spec_to_generate_without_artifact_store() -> None:
    seen = {}

    class _FakeLLM:
        provider = "lmstudio"
        model = "qwen/qwen3.6-27b"

        def generate(self, **kwargs):
            seen.update(kwargs)
            return _core_shaped_response(_decision_response(), text_provider_class="LMStudioProvider")

    client = object.__new__(LocalAbstractCoreLLMClient)
    client._provider = "lmstudio"
    client._model = "qwen/qwen3.6-27b"
    client._artifact_store = None
    client._generate_lock = None
    client._llm = _FakeLLM()
    client._capability_defaults = dict(DECISION_ROUTE)
    client._maybe_prepare_prompt_cache = lambda **_kwargs: None

    out = client.generate(
        prompt=STATE,
        params={
            "output": {"modality": "decision", "questions": QUESTIONS, "truncate": True, "run_id": "run-d", "tags": {"node_id": "n-d"}},
            "trace_metadata": {"run_id": "run-d", "node_id": "n-d"},
        },
    )

    # The spec reached `generate(prompt, output=[spec])` with the `output.decision` route on it,
    # runtime-only keys stripped, and no artifact store was demanded or passed.
    assert seen["prompt"] == STATE
    assert seen["output"] == [
        {"modality": "decision", "task": "decision", "questions": QUESTIONS, "truncate": True, "provider": "mlx", "model": "jaredpalmer/kev-4b"}
    ]
    assert "artifact_store" not in seen
    assert seen["tools"] is None and seen["media"] is None and seen["stream"] is False
    assert out["answers"]["department"]["choice"] == "shipping"
    assert out["decision"]["model"] == "mlx/jaredpalmer/kev-4b"
    assert out["provider"] == "mlx" and out["model"] == "jaredpalmer/kev-4b"
    assert out["outputs"] == {} and not out["tool_calls"]
    json.dumps(out)


def test_an_authored_decision_route_is_never_clobbered_by_the_default() -> None:
    seen = {}

    class _FakeLLM:
        def generate(self, **kwargs):
            seen.update(kwargs)
            return _core_shaped_response(_decision_response(model="ggml-org/Kev-4B-GGUF:Q4_K_M", provider="huggingface"))

    client = object.__new__(LocalAbstractCoreLLMClient)
    client._provider = "lmstudio"
    client._model = "qwen/qwen3.6-27b"
    client._artifact_store = None
    client._generate_lock = None
    client._llm = _FakeLLM()
    client._capability_defaults = dict(DECISION_ROUTE)
    client._maybe_prepare_prompt_cache = lambda **_kwargs: None

    out = client.generate(
        prompt=STATE,
        params={"output": {"modality": "decision", "questions": QUESTIONS, "provider": "huggingface", "model": "ggml-org/Kev-4B-GGUF:Q4_K_M"}},
    )
    assert seen["output"][0]["provider"] == "huggingface"
    assert seen["output"][0]["model"] == "ggml-org/Kev-4B-GGUF:Q4_K_M"
    assert out["provider"] == "huggingface"


# --------------------------------------------------------------------------- pooled client + handler


@pytest.fixture
def no_text_model(monkeypatch):
    """`create_llm` fails like on the light profile (no text model can be built); the decision
    model behind core's pool is a fake. Core's real `_run_decision_output` runs."""
    import abstractcore
    from abstractcore.core import factory
    from abstractcore.providers.base import BaseProvider

    built = []
    seen = {}

    def create_llm(provider, model=None, **kwargs):
        built.append((provider, model))
        raise ImportError('MLX dependencies not installed. Install with: pip install "abstractcore[apple]"')

    monkeypatch.setattr(abstractcore, "create_llm", create_llm)
    monkeypatch.setattr(factory, "create_llm", create_llm)

    class _FakeDecisionModel:
        def decide(self, state, questions, truncate=None):
            seen["state"] = state
            seen["questions"] = questions
            seen["truncate"] = truncate
            return _decision_response()

    class _FakeDecisionProvider:
        def decision_model(self):
            return _FakeDecisionModel()

    def fake_acquire(key, *, hold, factory, options):
        # core's real lane (`_decision_route` + `pool.use`) asked the pool for this route
        seen["spec"] = {"provider": key[0], "model": key[1], **options}
        return _FakeDecisionProvider()

    real_route = BaseProvider._decision_route

    def observed_route(self, spec):
        seen["provider_class"] = type(self).__name__
        return real_route(self, spec)

    from abstractcore.decision import pool

    monkeypatch.setattr(pool, "_acquire", fake_acquire)
    monkeypatch.setattr(BaseProvider, "_decision_route", observed_route)
    return built, seen


def _drive_decision(client, *, artifact_store=None):
    ledger = InMemoryLedgerStore()
    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=ledger,
        artifact_store=artifact_store,
        effect_handlers=build_effect_handlers(llm=client, artifact_store=artifact_store),
    )

    def decide(run, ctx):
        del run, ctx
        return StepPlan(
            node_id="decide",
            effect=Effect(
                type=EffectType.LLM_CALL,
                payload={"prompt": STATE, "output": {"modality": "decision", "questions": QUESTIONS}},
                result_key="verdict",
            ),
            next_node="done",
        )

    def done(run, ctx):
        del ctx
        return StepPlan(node_id="done", complete_output={"verdict": run.vars.get("verdict")})

    workflow = WorkflowSpec(workflow_id="wf_decide", entry_node="decide", nodes={"decide": decide, "done": done})
    run_id = runtime.start(workflow=workflow, vars={})
    state = runtime.tick(workflow=workflow, run_id=run_id)
    return state, ledger.list(run_id)


def test_a_decision_only_request_never_builds_the_text_client_and_needs_no_artifact_store(no_text_model) -> None:
    built, seen = no_text_model
    client = MultiLocalAbstractCoreLLMClient(
        provider="mlx", model="mlx-community/Qwen3.8-Flash-Next-4bit", capability_defaults=dict(DECISION_ROUTE),
    )
    built.clear()  # the start-up warm-up may try (and soft-fail); the decision must not

    assert _media_only_output_request(
        prompt=STATE, messages=None, media=None,
        params={"output": {"modality": "decision", "questions": QUESTIONS}}, capability_defaults=DECISION_ROUTE,
    )

    state, records = _drive_decision(client, artifact_store=None)

    assert state.status.value == "completed", state.error
    assert built == []  # no text provider was built for the decision
    assert seen["provider_class"] == "CapabilityHostProvider"
    assert seen["state"] == STATE and seen["questions"] == QUESTIONS and seen["truncate"] is None
    # The route the runtime resolved (its capability defaults) is what core's pool was asked for.
    assert seen["spec"]["provider"] == "mlx" and seen["spec"]["model"] == "jaredpalmer/kev-4b"

    result = state.output["verdict"]
    assert result["answers"]["department"]["choice"] == "shipping"
    assert result["decision"]["usage"]["input_tokens"] == 42
    assert result["provider"] == "mlx" and result["model"] == "jaredpalmer/kev-4b"
    assert result["outputs"] == {} and result["resources"] == {}
    assert result["metadata"]["_resolved_generate_route"]["outputs"][0]["modality"] == "decision"
    assert result["metadata"]["_runtime_resolved_action"]["action_id"] == "decision"
    json.dumps(records)


# --------------------------------------------------------------------------- remote client


class _RemoteDecisionSender:
    def __init__(self, *, model: str = "mlx/jaredpalmer/kev-4b") -> None:
        self.calls = []
        self.model = model

    def get(self, url, *, headers, timeout):
        raise AssertionError("not used")

    def post(self, url, *, headers, json, timeout):
        self.calls.append({"method": "POST", "url": url, "headers": dict(headers), "json": json, "timeout": timeout})
        body = _decision_response().to_systemone()
        body["model"] = self.model
        return body


def _remote_client(sender, **kwargs):
    return RemoteAbstractCoreLLMClient(
        server_base_url="http://core.test",
        model="openai/gpt-4o-mini",
        request_sender=sender,
        headers={"Authorization": "Bearer core-token"},
        artifact_store=None,
        **kwargs,
    )


def test_remote_decision_posts_v1_decisions_and_maps_the_body() -> None:
    sender = _RemoteDecisionSender()
    client = _remote_client(sender)

    out = client.generate(
        prompt=STATE,
        params={
            "output": {"modality": "decision", "questions": QUESTIONS, "provider": "mlx", "model": "jaredpalmer/kev-4b", "truncate": True},
            "trace_metadata": {"run_id": "run-dec", "node_id": "n-dec"},
        },
    )

    call = sender.calls[0]
    assert call["url"] == "http://core.test/v1/decisions"
    assert call["json"] == {"model": "mlx/jaredpalmer/kev-4b", "state": STATE, "questions": QUESTIONS, "truncate": True}
    assert call["headers"]["Authorization"] == "Bearer core-token"  # same headers as the speech/music routes
    assert call["headers"]["X-AbstractCore-Run-Id"] == "run-dec"
    expected = _decision_response().to_systemone()
    expected["model"] = "mlx/jaredpalmer/kev-4b"
    assert out["decision"] == expected
    assert out["answers"] == expected["answers"]
    assert out["content"] == json.dumps(expected["answers"], ensure_ascii=False)
    assert out["provider"] == "mlx" and out["model"] == "jaredpalmer/kev-4b"
    assert out["outputs"] == {} and out["resources"] == {}
    assert out["usage"] == {"input_tokens": 42, "output_tokens": 0}
    assert out["metadata"]["_provider_request"]["url"] == "http://core.test/v1/decisions"
    json.dumps(out)


def test_remote_decision_takes_the_route_from_the_capability_default() -> None:
    sender = _RemoteDecisionSender()
    client = _remote_client(sender, capability_defaults=dict(DECISION_ROUTE))

    client.generate(prompt=STATE, params={"output": {"modality": "decision", "questions": QUESTIONS}})

    assert sender.calls[0]["json"]["model"] == "mlx/jaredpalmer/kev-4b"
    assert "truncate" not in sender.calls[0]["json"]


def test_remote_decision_without_a_route_lets_the_server_pick_and_a_bare_model_goes_bare() -> None:
    sender = _RemoteDecisionSender(model="huggingface/ggml-org/Kev-4B-GGUF:Q4_K_M")
    client = _remote_client(sender)

    out = client.generate(prompt=STATE, params={"output": {"modality": "decision", "questions": QUESTIONS}})
    assert "model" not in sender.calls[0]["json"]
    # The identity reported is the one the SERVER answered with (`provider/model`).
    assert out["provider"] == "huggingface" and out["model"] == "ggml-org/Kev-4B-GGUF:Q4_K_M"

    client.generate(prompt=STATE, params={"output": {"modality": "decision", "questions": QUESTIONS, "model": "kev-latest"}})
    assert sender.calls[1]["json"]["model"] == "kev-latest"


def test_remote_decision_refuses_what_the_server_would_refuse_before_posting() -> None:
    sender = _RemoteDecisionSender()
    client = _remote_client(sender)

    with pytest.raises(ValueError, match="questions"):
        client.generate(prompt=STATE, params={"output": {"modality": "decision"}})
    with pytest.raises(ValueError, match="state"):
        client.generate(prompt="", params={"output": {"modality": "decision", "questions": QUESTIONS}})
    assert sender.calls == []


def test_remote_decision_status_errors_propagate_like_the_other_remote_lanes() -> None:
    class _Overflow(Exception):
        def __init__(self) -> None:
            super().__init__("422 Unprocessable Entity")
            self.response = SimpleNamespace(
                status_code=422,
                body={"detail": {"error": {"message": "state is 70000 tokens, the model reads 65536", "type": "context_overflow"}}},
            )

    class _RefusingSender(_RemoteDecisionSender):
        def post(self, url, *, headers, json, timeout):
            raise _Overflow()

    client = _remote_client(_RefusingSender())
    with pytest.raises(_Overflow) as exc:
        client.generate(prompt=STATE, params={"output": {"modality": "decision", "questions": QUESTIONS, "model": "kev-latest"}})
    assert exc.value.response.body["detail"]["error"]["type"] == "context_overflow"


# --------------------------------------------------------------------------- handler vocabulary


def test_resolved_action_id_is_decision() -> None:
    assert _resolved_action_id_from_output({"outputs": [{"modality": "decision", "task": "decision"}]}) == "decision"


def test_a_decision_is_not_tagged_generated_media() -> None:
    run = SimpleNamespace(run_id="r1", workflow_id="wf", current_node="decide", actor_id=None, session_id=None, parent_run_id=None)
    spec = _augment_output_request_for_runtime({"modality": "decision", "questions": QUESTIONS}, run=run)
    assert spec["tags"]["kind"] == "llm_output"
    assert spec["tags"]["modality"] == "decision"
    assert spec["run_id"] == "r1"
    voice = _augment_output_request_for_runtime({"modality": "voice"}, run=run)
    assert voice["tags"]["kind"] == "generated_media"


def test_a_decision_is_non_text_but_not_generated_media() -> None:
    spec = {"modality": "decision", "questions": QUESTIONS}
    assert output_request_has_non_text_result(spec) is True
    assert output_request_has_generated_media(spec) is False


def test_the_decision_route_key_is_output_decision() -> None:
    assert _output_default_route_keys({"modality": "decision", "task": ""}) == ("output.decision", None)
    assert capability_default_route_keys_for_spec({"modality": "decision", "task": "decision"}) == ("output.decision", None)
