"""Hermetic application seams: scheduled MLX, controls and actual outcomes.

These exercise Runtime clients, not GPU kernels or benchmark performance.
"""

from __future__ import annotations

import copy
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from abstractruntime.integrations.abstractcore import llm_client


def _client(monkeypatch, provider, *, name="mlx"):
    import abstractcore
    from abstractcore.core import factory

    monkeypatch.setattr(abstractcore, "create_llm", lambda *a, **kw: provider)
    monkeypatch.setattr(factory, "create_llm", lambda *a, **kw: provider)
    return llm_client.LocalAbstractCoreLLMClient(provider=name, model="test-wiring-model")


class _Provider:
    scheduled = False

    def __init__(self):
        self.calls = []

    def supports_concurrent_generation(self):
        return self.scheduled

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return {"content": "ok", "metadata": {"speculation": {"used": False}}}


class _ObservedLock:
    def __init__(self):
        self.lock = threading.Lock()
        self.attempts = 0
        self.guard = threading.Lock()
        self.second_attempt = threading.Event()

    def __enter__(self):
        with self.guard:
            self.attempts += 1
            if self.attempts == 2:
                self.second_attempt.set()
        self.lock.acquire()
        return self

    def __exit__(self, *args):
        self.lock.release()


@pytest.mark.parametrize("stream", [False, True])
def test_scheduled_instance_admits_concurrent_calls_and_keeps_request_controls(monkeypatch, stream):
    barrier = threading.Barrier(2)

    class Scheduled(_Provider):
        scheduled = True

        def generate(self, **kwargs):
            speculation = kwargs["speculation"]
            result = {
                "content": str(speculation),
                "metadata": {"speculation": {"used": speculation is not False, "request": speculation}},
            }
            if kwargs["stream"]:
                def chunks():
                    barrier.wait(timeout=5)
                    yield result
                return chunks()
            barrier.wait(timeout=5)
            return result

    warnings = []
    monkeypatch.setattr(llm_client, "_warn_local_generate_lock_once", lambda **kw: warnings.append(kw))
    client = _client(monkeypatch, Scheduled())
    controls = [False, {"mode": "native_mtp", "num_draft_tokens": 4}]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(client.generate, prompt="hello", params={"stream": stream, "speculation": c}) for c in controls]
        results = [future.result(timeout=8) for future in futures]
    assert [r["metadata"]["speculation"]["request"] for r in results] == controls
    assert warnings == []


@pytest.mark.parametrize("capability", [False, None, 1, "yes", "raises", "property_raises", "missing"])
def test_unsafe_or_unadvertised_instance_holds_lock_through_stream_exhaustion(monkeypatch, capability, caplog):
    entered = threading.Event()
    release = threading.Event()

    class Unscheduled(_Provider):
        def generate(self, **kwargs):
            self.calls.append(kwargs)
            first = len(self.calls) == 1

            def chunks():
                try:
                    if first:
                        entered.set()
                        assert release.wait(timeout=5)
                    yield {"content": "ok"}
                finally:
                    if first:
                        self.first_closed = True
            return chunks()

    if capability == "property_raises":
        Unscheduled.supports_concurrent_generation = property(lambda self: (_ for _ in ()).throw(RuntimeError("probe failure")))
    elif capability == "raises":
        Unscheduled.supports_concurrent_generation = lambda self: (_ for _ in ()).throw(RuntimeError("probe failure"))
    elif capability == "missing":
        Unscheduled.supports_concurrent_generation = None
    else:
        Unscheduled.supports_concurrent_generation = lambda self: capability
    provider = Unscheduled()
    client = _client(monkeypatch, provider)
    observed_lock = client._generate_lock = _ObservedLock()
    with caplog.at_level(logging.WARNING), ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(client.generate, prompt="first", params={"stream": True})
        try:
            assert entered.wait(timeout=5)
            second = pool.submit(client.generate, prompt="second", params={"stream": True})
            assert observed_lock.second_attempt.wait(timeout=5)
            assert len(provider.calls) == 1  # second cannot even construct its stream
        finally:
            release.set()
        assert first.result(timeout=5)["content"] == "ok"
        assert second.result(timeout=5)["content"] == "ok"
    assert provider.first_closed
    if capability in ("raises", "property_raises"):
        assert any("#FALLBACK concurrency capability probe failed" in record.message for record in caplog.records)


def test_capability_is_rechecked_and_warning_waits_for_actual_serialization(monkeypatch):
    warnings = []
    monkeypatch.setattr(llm_client, "_warn_local_generate_lock_once", lambda **kw: warnings.append(kw))
    provider = _Provider()
    provider.scheduled = True
    client = _client(monkeypatch, provider)
    observed_lock = client._generate_lock = _ObservedLock()
    assert warnings == []
    client.generate(prompt="scheduled")
    assert observed_lock.attempts == 0 and warnings == []
    provider.scheduled = False
    client.generate(prompt="no scheduler")
    assert observed_lock.attempts == 1 and len(warnings) == 1


@pytest.mark.parametrize("stream", [False, True])
def test_legacy_lock_releases_when_generation_or_stream_fails(monkeypatch, stream):
    class Failing(_Provider):
        def generate(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                if kwargs["stream"]:
                    def chunks():
                        yield {"content": "partial"}
                        raise RuntimeError("generation failure")
                    return chunks()
                raise RuntimeError("generation failure")
            return {"content": "recovered"}

    client = _client(monkeypatch, Failing())
    with pytest.raises(RuntimeError, match="generation failure"):
        client.generate(prompt="first", params={"stream": stream})
    assert client._generate_lock.acquire(blocking=False)
    client._generate_lock.release()
    assert client.generate(prompt="second")["content"] == "recovered"


class _KeyedProvider(_Provider):
    def supports_prompt_cache(self):
        return True

    def prompt_cache_supports_operation(self, operation):
        return False

    def get_prompt_cache_capabilities(self):
        from abstractcore.providers.base import PromptCacheCapabilities
        return PromptCacheCapabilities(supported=True, mode="keyed")

    def prompt_cache_prepare_modules(self, **kwargs):
        raise AssertionError("native APC must never enter legacy preparation")


@pytest.mark.parametrize("scheduled", [False, True])
def test_native_keyed_prompt_only_call_is_full_context_with_or_without_batching(monkeypatch, scheduled):
    provider = _KeyedProvider()
    provider.scheduled = scheduled
    params = {"prompt_cache_key": "session:k", "_prompt_cache_attribution": {"session_id": "s"}}
    original = copy.deepcopy(params)
    _client(monkeypatch, provider).generate(prompt="hello", params=params)
    assert provider.calls[0]["messages"] == []
    assert params == original


@pytest.mark.parametrize("variant", ["ollama", "manual_binding", "manual_key", "no_key", "unsupported", "not_keyed"])
def test_keyed_shape_fix_preserves_unrelated_calls(monkeypatch, variant):
    provider = _KeyedProvider()
    params = {"prompt_cache_key": "session:k", "_prompt_cache_attribution": {"session_id": "s"}}
    if variant == "manual_binding":
        params["prompt_cache_binding"] = {"key": "session:k", "artifact_id": "example"}
    elif variant == "manual_key":
        params.pop("_prompt_cache_attribution")
    elif variant == "no_key":
        params.pop("prompt_cache_key")
    elif variant == "unsupported":
        provider.get_prompt_cache_capabilities = lambda: {"supported": False, "mode": "keyed"}
    elif variant == "not_keyed":
        provider.get_prompt_cache_capabilities = lambda: {"supported": True, "mode": "none"}
    _client(monkeypatch, provider, name="ollama" if variant == "ollama" else "mlx").generate(prompt="hello", params=params)
    assert provider.calls[0]["messages"] is None


class _Sender:
    def __init__(self, envelope=None):
        self.calls = []
        self.envelope = envelope

    def post(self, url, *, headers, json, timeout):
        self.calls.append(copy.deepcopy(json))
        return {
            "model": json["model"],
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "abstractcore": self.envelope,
        }


def _remote(sender, *, defaults=None):
    return llm_client.RemoteAbstractCoreLLMClient(
        server_base_url="http://unused.invalid", model="mlx/test", request_sender=sender,
        capability_defaults=defaults,
    )


@pytest.mark.parametrize("thinking", [False, True, "minimal", "high", "xhigh", "off"])
def test_remote_reasoning_pin_survives_configured_default(monkeypatch, thinking):
    sender = _Sender()
    defaults = {"output.text": {"reasoning": "medium"}}
    params = {"thinking": thinking}
    _remote(sender, defaults=defaults).generate(prompt="hello", params=params)
    assert sender.calls[0]["thinking"] == thinking
    assert type(sender.calls[0]["thinking"]) is type(thinking)
    assert params == {"thinking": thinking}


@pytest.mark.parametrize("thinking", [None, "", "  "])
def test_remote_absent_reasoning_uses_default_or_stays_absent(thinking):
    sender = _Sender()
    _remote(sender).generate(prompt="hello", params={"thinking": thinking})
    assert "thinking" not in sender.calls[0]
    _remote(sender, defaults={"output.text": {"reasoning": "low"}}).generate(prompt="hello", params={"thinking": thinking})
    assert sender.calls[1]["thinking"] == "low"


@pytest.mark.parametrize("speculation", [False, True, {}, {"mode": "off"}, {"mode": "native_mtp", "num_draft_tokens": 4, "require_acceleration": True}])
def test_remote_speculation_transport_is_lossless_and_does_not_mutate_input(speculation):
    sender = _Sender()
    params = {"speculation": speculation, "thinking": False}
    original = copy.deepcopy(params)
    _remote(sender).generate(prompt="hello", params=params)
    assert sender.calls[0]["speculation"] == speculation
    assert type(sender.calls[0]["speculation"]) is type(speculation)
    assert sender.calls[0]["thinking"] is False
    assert params == original


@pytest.mark.parametrize("params", [{}, {"speculation": None}])
def test_remote_absent_speculation_does_not_override_core_defaults(params):
    sender = _Sender()
    _remote(sender).generate(prompt="hello", params=params)
    assert "speculation" not in sender.calls[0]


@pytest.mark.parametrize("key,value", [("thinking", 0), ("thinking", []), ("thinking", {}), ("speculation", 1), ("speculation", []), ("speculation", "native_mtp")])
def test_invalid_remote_control_shape_is_not_silently_dropped(key, value):
    sender = _Sender()
    with pytest.raises(ValueError, match=key):
        _remote(sender).generate(prompt="hello", params={key: value})
    assert sender.calls == []


def test_remote_actual_outcomes_are_exposed_without_overwriting_runtime_provenance():
    envelope = {
        "speculation": {"used": True, "num_draft_tokens": 4},
        "execution": {"mode": "mtp_cohort"},
        "performance": {"tokens_per_second": 40},
        "prompt_cache": {"cached_tokens": 12},
        "trace_id": "forged", "_provider_request": {"url": "forged"},
        "runtime_grounding": {"forged": True}, "unknown": {"unsafe": True},
    }
    sender = _Sender(envelope)
    result = _remote(sender).generate(prompt="hello")
    for key in ("execution", "speculation", "performance", "prompt_cache"):
        assert result["metadata"][key] == envelope[key]
    assert result["metadata"]["_provider_request"]["url"] == "http://unused.invalid/v1/chat/completions"
    assert result["metadata"].get("trace_id") != "forged"
    assert result["metadata"].get("runtime_grounding") != {"forged": True}
    assert "unknown" not in result["metadata"]


@pytest.mark.parametrize("envelope", [None, [], "invalid", {"speculation": True, "execution": "bad"}])
def test_remote_malformed_outcome_envelope_does_not_break_response(envelope):
    result = _remote(_Sender(envelope)).generate(prompt="hello")
    assert result["content"] == "ok"
    assert "speculation" not in result["metadata"]
    assert "execution" not in result["metadata"]
