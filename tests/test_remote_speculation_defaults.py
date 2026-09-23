"""Remote policy forwarding is scoped intent, never a local capability claim."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import threading

import pytest

from abstractruntime.integrations.abstractcore.llm_client import RemoteAbstractCoreLLMClient


D3 = {"mode": "native_mtp", "num_draft_tokens": 3, "require_acceleration": False}
D4 = {"mode": "native_mtp", "num_draft_tokens": 4, "require_acceleration": False}
D5 = {"mode": "native_mtp", "num_draft_tokens": 5, "require_acceleration": False}
ABSENT = object()


def routes(policy):
    return {"input.text": {"options": {"speculation": deepcopy(policy)}}}


class Sender:
    def __init__(self, supported=True):
        self.bodies = []
        self.capabilities = {
            "speculation": {
                "supported": supported, "ready": False, "reason": "remote_not_prepared",
                "requires_reload": True, "default": deepcopy(D5),
                "effective_default": deepcopy(D5) if supported else False,
            },
            "concurrency": {"supported": True, "source": "remote_instance"},
        }

    def get(self, url, **kwargs):
        return self.capabilities

    def post(self, url, *, json, **kwargs):
        self.bodies.append(json)
        return {
            "model": json["model"],
            "choices": [{"message": {"content": "ok", "role": "assistant"}}],
            "abstractcore": {"speculation": {
                "used": False, "reason": "remote_not_prepared",
                "details": {"effective_policy": deepcopy(json.get("speculation", D5))},
            }},
        }


def client(sender, **kwargs):
    return RemoteAbstractCoreLLMClient(
        server_base_url="http://unused.invalid", model="mlx/exact-artifact",
        request_sender=sender, **kwargs,
    )


@pytest.mark.parametrize("scope,expected", [
    (None, ABSENT), ({}, False),
    ({"input.text": {"options": {"reasoning": "high"}}}, False),
    (routes(None), False), (routes(False), False), (routes(D4), D4),
    ({"routes": [{"key": "input.text", "options": {"speculation": D4}}]}, D4),
])
@pytest.mark.parametrize("per_call", [ABSENT, None, False, D3, {}])
def test_remote_scope_precedence_preserves_off_inherit_and_partial_controls(scope, expected, per_call):
    sender = Sender()
    runtime_client = client(sender, capability_defaults=scope)
    params = {} if per_call is ABSENT else {"speculation": deepcopy(per_call)}
    before = deepcopy(params)
    result = runtime_client.generate(prompt="hello", params=params)
    assert params == before
    if per_call is not ABSENT and per_call is not None:
        expected = per_call
    body = sender.bodies[-1]
    if expected is ABSENT:
        assert "speculation" not in body
        effective = D5
    else:
        assert body["speculation"] == expected
        effective = expected
    outcome = result["metadata"]["speculation"]
    assert outcome["used"] is False
    assert outcome["reason"] == "remote_not_prepared"
    assert outcome["details"]["effective_policy"] == effective


def test_unscoped_remote_never_reads_client_machine_default(monkeypatch):
    import abstractcore.providers.speculation as core
    monkeypatch.setattr(core, "configured_speculation_default", lambda **kw: pytest.fail("host owns unscoped defaults"))
    sender = Sender()
    runtime_client = client(sender)
    runtime_client.generate(prompt="hello")
    assert "speculation" not in sender.bodies[-1]
    assert runtime_client.get_execution_capabilities()["speculation"]["default"] == D5


def test_explicit_config_file_policy_is_resolved_by_core_and_refreshes(tmp_path):
    path = tmp_path / "core.json"
    path.write_text(json.dumps({"capability_defaults": {"routes": routes(D4)}}))
    sender = Sender()
    runtime_client = client(sender, core_config_file=path)
    runtime_client.generate(prompt="hello")
    assert sender.bodies[-1]["speculation"] == D4
    path.write_text(json.dumps({"capability_defaults": {"routes": routes(False)}}))
    runtime_client.generate(prompt="hello")
    assert sender.bodies[-1]["speculation"] is False
    assert runtime_client.get_execution_capabilities()["speculation"]["default"] is False
    # Explicit empty scope takes precedence over a config file's policy.
    path.write_text(json.dumps({"capability_defaults": {"routes": routes(D4)}}))
    assert runtime_client.set_capability_defaults({}) is True
    runtime_client.generate(prompt="hello")
    assert sender.bodies[-1]["speculation"] is False
    assert runtime_client.set_capability_defaults(None) is True
    runtime_client.generate(prompt="hello")
    assert sender.bodies[-1]["speculation"] == D4


@pytest.mark.parametrize("supported", [False, True])
@pytest.mark.parametrize("scope,expected", [(None, D5), ({}, False), (routes(False), False), (routes(D4), D4)])
def test_remote_discovery_overlays_only_scoped_policy_not_actual_capability(supported, scope, expected):
    sender = Sender(supported=supported)
    original = deepcopy(sender.capabilities)
    runtime_client = client(sender, capability_defaults=scope)
    result = runtime_client.get_execution_capabilities(provider="mlx")
    block = result["speculation"]
    assert block["default"] == expected
    assert block["effective_default"] == (expected if supported else False)
    for field in ("supported", "ready", "reason", "requires_reload"):
        assert block[field] == original["speculation"][field]
    assert result["concurrency"] == original["concurrency"]
    assert sender.capabilities == original


def test_remote_capability_refresh_releases_scope_and_does_not_mutate_old_request():
    entered = threading.Event()
    release = threading.Event()

    class BlockingSender(Sender):
        def post(self, url, *, json, **kwargs):
            result = super().post(url, json=json, **kwargs)
            if len(self.bodies) == 1:
                entered.set()
                assert release.wait(3), "test did not release first request"
            return result

    sender = BlockingSender()
    original_scope = routes(D3)
    runtime_client = client(sender, capability_defaults=original_scope)
    original_scope["input.text"]["options"]["speculation"]["num_draft_tokens"] = 5
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(runtime_client.generate, prompt="first")
        try:
            assert entered.wait(3), "first request never entered sender"
            next_scope = routes(D4)
            assert runtime_client.set_capability_defaults(next_scope) is True
            assert runtime_client.set_capability_defaults(next_scope) is False
            next_scope["input.text"]["options"]["speculation"]["num_draft_tokens"] = 5
            second = pool.submit(runtime_client.generate, prompt="second")
            assert second.result(timeout=3)["content"] == "ok"
            assert sender.bodies[0]["speculation"] == D3
            assert sender.bodies[1]["speculation"] == D4
            assert runtime_client.get_execution_capabilities()["speculation"]["default"] == D4
        finally:
            release.set()
        assert first.result(timeout=3)["content"] == "ok"
    assert runtime_client.set_capability_defaults({}) is True
    runtime_client.generate(prompt="third")
    assert sender.bodies[-1]["speculation"] is False
    assert runtime_client.set_capability_defaults(None) is True
    assert runtime_client.set_capability_defaults(None) is False
    runtime_client.generate(prompt="fourth")
    assert "speculation" not in sender.bodies[-1]
    assert runtime_client._model == "mlx/exact-artifact"
    assert runtime_client._server_base_url == "http://unused.invalid"

