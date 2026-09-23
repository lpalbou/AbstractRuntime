"""Execution capability queries must never load models or guess remote support."""
from types import SimpleNamespace
from copy import deepcopy
from urllib.parse import parse_qs, urlparse

import pytest

from abstractruntime.integrations.abstractcore.llm_client import (
    LocalAbstractCoreLLMClient, MultiLocalAbstractCoreLLMClient, RemoteAbstractCoreLLMClient,
)
from abstractruntime.integrations.abstractcore.discovery_facade import AbstractCoreDiscoveryFacade
from abstractruntime.integrations.abstractcore.config_facade import normalize_speculation_control


@pytest.mark.parametrize("value", [None, False, True, {}, {"num_draft_tokens": 4}, {"mode": "off"}])
def test_config_facade_uses_core_validation_without_filling_missing_keys(value):
    from abstractcore.providers.speculation import normalize_speculation_value
    assert normalize_speculation_control(value) == normalize_speculation_value(value)
    if isinstance(value, dict):
        assert set(normalize_speculation_control(value)) == set(value)
        assert normalize_speculation_control(value) is not value


@pytest.mark.parametrize("value", [2, [], {"num_draft_tokens": 0}, {"typo": 3}])
def test_config_facade_rejects_invalid_controls(value):
    with pytest.raises(ValueError):
        normalize_speculation_control(value)


@pytest.mark.parametrize("pooled", [False, True])
def test_local_discovery_uses_only_an_existing_matching_instance(monkeypatch, pooled):
    import abstractcore.providers.speculation as core
    calls = []
    def describe(model_name, *, provider, instance=None):
        calls.append((model_name, provider, instance))
        return {"speculation": {"ready": instance is not None}, "concurrency": {"supported": instance is not None}}
    monkeypatch.setattr(core, "get_execution_capabilities", describe)
    instance = object()
    if pooled:
        client = MultiLocalAbstractCoreLLMClient.__new__(MultiLocalAbstractCoreLLMClient)
        client._default_provider, client._default_model = "mlx", "present"
        client._clients = {("mlx", "present"): SimpleNamespace(_llm=instance)}
        client._get_client = lambda *a, **kw: pytest.fail("discovery must not load a provider")
    else:
        client = LocalAbstractCoreLLMClient.__new__(LocalAbstractCoreLLMClient)
        client._provider, client._model, client._llm = "mlx", "present", instance
    assert client.get_execution_capabilities()["speculation"]["ready"] is True
    assert client.get_execution_capabilities("absent", provider="mlx")["speculation"]["ready"] is False
    assert client.get_execution_capabilities("present", provider="huggingface")["speculation"]["ready"] is False
    assert calls == [("present", "mlx", instance), ("absent", "mlx", None), ("present", "huggingface", None)]


def test_remote_and_facade_preserve_execution_host_authority(monkeypatch):
    import abstractcore.providers.speculation as core
    monkeypatch.setattr(core, "get_execution_capabilities", lambda *a, **kw: pytest.fail("remote must not inspect local registry"))
    class Sender:
        def get(self, url, *, headers, timeout):
            query = parse_qs(urlparse(url).query)
            assert urlparse(url).path == "/v1/models/execution-capabilities"
            assert query == {"model_name": ["repo/model"], "provider": ["mlx"]}
            return {"speculation": {"supported": False, "reason": "backend_not_implemented"}, "concurrency": {"supported": False}}
    client = RemoteAbstractCoreLLMClient(server_base_url="http://unused.invalid", model="other", request_sender=Sender())
    facade = AbstractCoreDiscoveryFacade(SimpleNamespace(_abstractcore_llm_client=client))
    result = facade.get_execution_capabilities("repo/model", provider="mlx")
    assert result["speculation"]["supported"] is False
    assert result["source"] == "abstractcore.remote"


def test_remote_discovery_failure_is_unavailable_not_local_success():
    class Sender:
        def get(self, *args, **kwargs):
            raise ConnectionError("offline")
    client = RemoteAbstractCoreLLMClient(server_base_url="http://unused.invalid", model="repo/model", request_sender=Sender())
    result = client.get_execution_capabilities(provider="mlx")
    assert result["available"] is False
    assert "offline" in result["error"]
    assert "speculation" not in result


@pytest.mark.parametrize("pooled", [False, True])
@pytest.mark.parametrize("routes", [None, {}, {"input.text": {"options": {"speculation": False}}}, {"input.text": {"options": {"speculation": {"num_draft_tokens": 4}}}}])
def test_scoped_policy_is_available_before_provider_load_without_losing_absence(monkeypatch, tmp_path, pooled, routes):
    import abstractcore
    from abstractcore.core import factory
    from abstractcore.providers.speculation import configured_speculation_default
    calls = []
    before = deepcopy(routes)
    config_file = str(tmp_path / "scoped-core.json")

    def create(provider, *, model, **kwargs):
        assert kwargs["_abstractcore_config_file"] == config_file
        assert ("_abstractcore_capability_defaults" in kwargs) == (routes is not None)
        value = configured_speculation_default(
            config_file=kwargs.get("_abstractcore_config_file"),
            capability_defaults=kwargs.get("_abstractcore_capability_defaults"),
        )
        expected = None if routes == {} else (routes["input.text"]["options"]["speculation"] if routes else {"mode": "native_mtp", "num_draft_tokens": 2, "require_acceleration": False})
        assert value == expected
        calls.append(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(abstractcore, "create_llm", create)
    monkeypatch.setattr(factory, "create_llm", create)
    cls = MultiLocalAbstractCoreLLMClient if pooled else LocalAbstractCoreLLMClient
    client = cls(provider="mlx", model="test", core_config_file=config_file, capability_defaults=routes)
    assert len(calls) == 1
    provider = client._llm
    if routes is not None:
        from abstractruntime.integrations.abstractcore.llm_client import _normalize_core_capability_defaults
        assert provider._abstractcore_capability_defaults == _normalize_core_capability_defaults(routes)
        provider._abstractcore_capability_defaults.setdefault("input.text", {}).setdefault("options", {})["speculation"] = True
        assert routes == before
    else:
        assert not hasattr(provider, "_abstractcore_capability_defaults")


def test_pool_refresh_distinguishes_inherit_from_authoritative_empty_scope(monkeypatch):
    import abstractcore
    from abstractcore.core import factory
    captures = []
    def create(*args, **kwargs):
        captures.append(kwargs)
        return SimpleNamespace()
    monkeypatch.setattr(abstractcore, "create_llm", create)
    monkeypatch.setattr(factory, "create_llm", create)
    client = MultiLocalAbstractCoreLLMClient(provider="mlx", model="test")
    assert "_abstractcore_capability_defaults" not in captures[-1]
    assert client.set_capability_defaults({}) is True
    assert captures[-1]["_abstractcore_capability_defaults"] == {}
    assert client.set_capability_defaults(None) is True
    assert "_abstractcore_capability_defaults" not in captures[-1]
