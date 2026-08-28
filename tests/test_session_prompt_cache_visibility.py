"""Host facade relays for memory snapshots and session prompt-cache
visibility/clearing (core-owned truth; runtime relays, ADR 0007).
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Tuple

from abstractruntime.integrations.abstractcore.llm_client import (
    LocalAbstractCoreLLMClient,
    MultiLocalAbstractCoreLLMClient,
    RemoteAbstractCoreLLMClient,
)


class _StatsProvider:
    """Provider fake exposing the raw stats/clear prompt-cache protocol."""

    def __init__(self, *, meta_by_key: Dict[str, Dict[str, Any]], fail_clear: Tuple[str, ...] = ()) -> None:
        self.meta_by_key = {k: dict(v) for k, v in meta_by_key.items()}
        self.fail_clear = tuple(fail_clear)
        self.cleared: List[str] = []

    def supports_prompt_cache(self) -> bool:
        return True

    def prompt_cache_supports_operation(self, operation: str) -> bool:
        return operation in {"stats", "clear"}

    def get_prompt_cache_stats(self) -> Dict[str, Any]:
        return {
            "entries": len(self.meta_by_key),
            "keys": list(self.meta_by_key.keys()),
            "meta_by_key": {k: dict(v) for k, v in self.meta_by_key.items() if v},
        }

    def prompt_cache_clear(self, key: Optional[str] = None) -> bool:
        if key in self.fail_clear:
            raise RuntimeError(f"clear exploded for {key}")
        self.cleared.append(str(key))
        self.meta_by_key.pop(str(key), None)
        return True


def _local_client(provider: Any, *, provider_name: str = "mlx", model: str = "qwen") -> LocalAbstractCoreLLMClient:
    client = LocalAbstractCoreLLMClient.__new__(LocalAbstractCoreLLMClient)
    client._provider = provider_name  # type: ignore[attr-defined]
    client._model = model  # type: ignore[attr-defined]
    client._llm = provider  # type: ignore[attr-defined]
    client._llm_kwargs = {}  # type: ignore[attr-defined]
    client._artifact_store = None  # type: ignore[attr-defined]
    client._prompt_cache_state_lock = threading.Lock()  # type: ignore[attr-defined]
    client._prompt_cache_state = {}  # type: ignore[attr-defined]
    return client


def _multilocal_with_pool(clients: Dict[Tuple[str, str], Any]) -> MultiLocalAbstractCoreLLMClient:
    client = MultiLocalAbstractCoreLLMClient.__new__(MultiLocalAbstractCoreLLMClient)
    client._clients = dict(clients)  # type: ignore[attr-defined]
    client._override_clients = {}  # type: ignore[attr-defined]
    client._default_client = None  # type: ignore[attr-defined]
    return client


_SESSION_META = {
    "session_id": "sess-1",
    "run_id": "run-1",
    "namespace": "session",
    "token_count": 42,
    "bytes": 1024,
    "created_at_s": 111.0,
}


def test_local_list_session_prompt_caches_reads_stamped_meta_and_filters() -> None:
    provider = _StatsProvider(
        meta_by_key={
            "session:abc": dict(_SESSION_META),
            "session:other": {"session_id": "sess-2", "created_at_s": 222.0},
            "unstamped:key": {},
        }
    )
    client = _local_client(provider)

    listed = client.list_session_prompt_caches()
    assert listed["ok"] is True
    by_key = {row["key"]: row for row in listed["caches"]}
    assert set(by_key) == {"session:abc", "session:other", "unstamped:key"}
    row = by_key["session:abc"]
    assert row["provider"] == "mlx"
    assert row["model"] == "qwen"
    assert row["runtime_id"] == "local:text_generation:mlx:qwen"
    assert row["session_id"] == "sess-1"
    assert row["token_count"] == 42
    assert row["bytes"] == 1024
    assert row["created_at_s"] == 111.0
    assert row["last_used_at_s"] is None
    assert row["meta"]["namespace"] == "session"
    assert by_key["unstamped:key"]["session_id"] is None
    assert by_key["unstamped:key"]["token_count"] is None

    filtered = client.list_session_prompt_caches(session_id="sess-1")
    assert [row["key"] for row in filtered["caches"]] == ["session:abc"]


def test_local_list_session_prompt_caches_without_stats_support_is_empty_ok() -> None:
    client = _local_client(object())

    assert client.list_session_prompt_caches() == {"ok": True, "caches": []}


def test_local_clear_session_prompt_caches_reports_partial_failures_per_row() -> None:
    provider = _StatsProvider(
        meta_by_key={
            "session:abc": dict(_SESSION_META),
            "session:def": {"session_id": "sess-1"},
            "session:other": {"session_id": "sess-2"},
        },
        fail_clear=("session:def",),
    )
    client = _local_client(provider)
    client._prompt_cache_state["session:abc"] = object()  # type: ignore[index]
    client._prompt_cache_state["session:def"] = object()  # type: ignore[index]

    result = client.clear_session_prompt_caches("sess-1")

    assert result["ok"] is True
    assert result["count"] == 1
    by_key = {row["key"]: row for row in result["cleared"]}
    assert set(by_key) == {"session:abc", "session:def"}
    assert by_key["session:abc"]["cleared"] is True
    assert by_key["session:def"]["cleared"] is False
    assert "clear exploded" in by_key["session:def"]["error"]
    assert provider.cleared == ["session:abc"]
    # Other sessions' caches stay untouched; cleared key drops client mirrors.
    assert "session:other" in provider.meta_by_key
    assert "session:abc" not in client._prompt_cache_state
    assert "session:def" in client._prompt_cache_state


def test_clear_session_prompt_caches_requires_a_session_id() -> None:
    client = _local_client(_StatsProvider(meta_by_key={}))
    result = client.clear_session_prompt_caches("")
    assert result["ok"] is False
    assert result["cleared"] == []
    assert result["count"] == 0


def test_multilocal_list_and_clear_route_to_the_pooled_clients() -> None:
    provider_a = _StatsProvider(meta_by_key={"session:abc": dict(_SESSION_META)})
    provider_b = _StatsProvider(meta_by_key={"session:zzz": {"session_id": "sess-1"}, "other": {}})
    pool = {
        ("mlx", "qwen"): _local_client(provider_a, provider_name="mlx", model="qwen"),
        ("lmstudio", "gemma"): _local_client(provider_b, provider_name="lmstudio", model="gemma"),
    }
    client = _multilocal_with_pool(pool)

    listed = client.list_session_prompt_caches(session_id="sess-1")
    assert listed["ok"] is True
    assert {(row["provider"], row["key"]) for row in listed["caches"]} == {
        ("mlx", "session:abc"),
        ("lmstudio", "session:zzz"),
    }

    cleared = client.clear_session_prompt_caches("sess-1")
    assert cleared["ok"] is True
    assert cleared["count"] == 2
    assert "errors" not in cleared
    assert provider_a.cleared == ["session:abc"]
    assert provider_b.cleared == ["session:zzz"]


def test_multilocal_list_and_clear_report_failing_pooled_clients() -> None:
    provider_ok = _StatsProvider(meta_by_key={"session:abc": dict(_SESSION_META)})
    good = _local_client(provider_ok, provider_name="mlx", model="qwen")

    class _ExplodingClient:
        _provider = "lmstudio"
        _model = "gemma"

        def list_session_prompt_caches(self, session_id: Optional[str] = None) -> Dict[str, Any]:
            raise RuntimeError("pooled client exploded")

        def clear_session_prompt_caches(self, session_id: str) -> Dict[str, Any]:
            raise RuntimeError("pooled client exploded")

    client = _multilocal_with_pool({("mlx", "qwen"): good, ("lmstudio", "gemma"): _ExplodingClient()})

    listed = client.list_session_prompt_caches(session_id="sess-1")
    assert listed["ok"] is True
    assert [row["key"] for row in listed["caches"]] == ["session:abc"]
    assert listed["errors"] == [
        {"provider": "lmstudio", "model": "gemma", "error": "pooled client exploded"}
    ]

    cleared = client.clear_session_prompt_caches("sess-1")
    assert cleared["ok"] is True
    assert cleared["count"] == 1
    assert cleared["errors"] == [
        {"provider": "lmstudio", "model": "gemma", "error": "pooled client exploded"}
    ]


def test_local_memory_snapshot_relays_the_core_function(monkeypatch) -> None:
    import abstractcore.utils.memory as core_memory

    snapshot = {
        "ts": 1.0,
        "ram": {"total_bytes": 8, "available_bytes": 4, "used_bytes": 4, "percent": 50.0},
        "process": {"rss_bytes": 2},
        "device": {"backend": None, "allocated_bytes": None, "total_bytes": None, "free_bytes": None},
    }
    monkeypatch.setattr(core_memory, "get_memory_snapshot", lambda: dict(snapshot))

    local = LocalAbstractCoreLLMClient.__new__(LocalAbstractCoreLLMClient)
    multi = MultiLocalAbstractCoreLLMClient.__new__(MultiLocalAbstractCoreLLMClient)

    assert local.get_memory_snapshot() == snapshot
    assert multi.get_memory_snapshot() == snapshot


class _RemoteRecordingSender:
    def __init__(self, *, get_responses: List[Any] = None, post_responses: List[Any] = None, fail: bool = False) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._get_responses = list(get_responses or [])
        self._post_responses = list(post_responses or [])
        self._fail = fail

    def get(self, url: str, *, headers: Dict[str, str], timeout: float) -> Any:
        self.calls.append({"method": "GET", "url": url, "headers": dict(headers), "timeout": timeout})
        if self._fail:
            raise RuntimeError("core server unreachable")
        return self._get_responses.pop(0)

    def post(self, url: str, *, headers: Dict[str, str], json: Dict[str, Any], timeout: float) -> Any:
        self.calls.append({"method": "POST", "url": url, "headers": dict(headers), "json": dict(json), "timeout": timeout})
        if self._fail:
            raise RuntimeError("core server unreachable")
        return self._post_responses.pop(0)


def _remote_client(sender: Any) -> RemoteAbstractCoreLLMClient:
    return RemoteAbstractCoreLLMClient(
        server_base_url="http://core.test/v1",
        model="mlx/qwen",
        timeout_s=30,
        request_sender=sender,
    )


def test_remote_memory_snapshot_strips_the_ok_wrapper() -> None:
    sender = _RemoteRecordingSender(get_responses=[{"ok": True, "ts": 1.0, "ram": {"total_bytes": 8}}])
    client = _remote_client(sender)

    snapshot = client.get_memory_snapshot()

    assert sender.calls[0]["url"] == "http://core.test/acore/memory"
    assert snapshot == {"ts": 1.0, "ram": {"total_bytes": 8}}


def test_remote_memory_snapshot_transport_failure_reports_instead_of_raising() -> None:
    client = _remote_client(_RemoteRecordingSender(fail=True))

    snapshot = client.get_memory_snapshot()

    assert snapshot["ok"] is False
    assert "unreachable" in snapshot["error"]


_REMOTE_STATS_PAYLOAD = {
    "ok": True,
    "operation": "stats",
    "runtimes": [
        {
            "runtime_id": "rt-1",
            "provider": "mlx",
            "model": "qwen",
            "stats": {
                "keys": ["session:abc", "other:key"],
                "meta_by_key": {"session:abc": dict(_SESSION_META)},
            },
        },
        {"runtime_id": "rt-2", "provider": "ollama", "model": "granite", "stats": None, "error": "boom"},
    ],
}


def test_remote_list_session_prompt_caches_builds_rows_from_cross_runtime_stats() -> None:
    sender = _RemoteRecordingSender(get_responses=[dict(_REMOTE_STATS_PAYLOAD)])
    client = _remote_client(sender)

    listed = client.list_session_prompt_caches(session_id="sess-1")

    assert sender.calls[0]["url"] == "http://core.test/acore/prompt_cache/stats"
    assert listed["ok"] is True
    assert len(listed["caches"]) == 1
    row = listed["caches"][0]
    assert row["key"] == "session:abc"
    assert row["provider"] == "mlx"
    assert row["model"] == "qwen"
    assert row["runtime_id"] == "rt-1"
    assert row["session_id"] == "sess-1"
    assert row["token_count"] == 42
    assert row["bytes"] == 1024


def test_remote_list_session_prompt_caches_transport_failure_is_ok_false() -> None:
    client = _remote_client(_RemoteRecordingSender(fail=True))

    listed = client.list_session_prompt_caches()

    assert listed["ok"] is False
    assert listed["caches"] == []
    assert listed["error"]


def test_remote_clear_session_prompt_caches_posts_clear_per_runtime_key() -> None:
    sender = _RemoteRecordingSender(
        get_responses=[dict(_REMOTE_STATS_PAYLOAD)],
        post_responses=[{"ok": True, "supported": True, "operation": "clear"}],
    )
    client = _remote_client(sender)

    result = client.clear_session_prompt_caches("sess-1")

    assert result["ok"] is True
    assert result["count"] == 1
    assert result["cleared"][0]["cleared"] is True
    post = sender.calls[-1]
    assert post["method"] == "POST"
    assert post["url"] == "http://core.test/acore/prompt_cache/clear"
    assert post["json"] == {"key": "session:abc", "runtime_id": "rt-1"}


def test_remote_clear_session_prompt_caches_reports_row_failures() -> None:
    sender = _RemoteRecordingSender(
        get_responses=[dict(_REMOTE_STATS_PAYLOAD)],
        post_responses=[{"ok": False, "error": "nope"}],
    )
    client = _remote_client(sender)

    result = client.clear_session_prompt_caches("sess-1")

    assert result["ok"] is True
    assert result["count"] == 0
    assert result["cleared"][0]["cleared"] is False
    assert result["cleared"][0]["error"] == "nope"


class _ResidentProvider:
    def __init__(self, *, provider: str, model: str, loaded: bool = True) -> None:
        self.provider = provider
        self.model = model
        self.loaded = loaded
        self.unload_model_calls: List[str] = []

    def get_model_residency(self, **kwargs: Any) -> Dict[str, Any]:
        return {
            "task": "text_generation",
            "provider": self.provider,
            "model": str(kwargs.get("model") or self.model),
            "provider_residency_verified": True,
            "provider_resident": bool(self.loaded),
            "loaded": bool(self.loaded),
            "state": "loaded" if self.loaded else "not_loaded",
            "source": "abstractcore.provider.test",
        }

    def unload_model(self, model_name: str, **kwargs: Any) -> Dict[str, Any]:
        _ = kwargs
        self.loaded = False
        self.unload_model_calls.append(str(model_name))
        return {"supported": True, "operation": "unload"}


def test_local_unload_drops_prompt_cache_client_mirrors() -> None:
    provider = _ResidentProvider(provider="lmstudio", model="qwen", loaded=True)
    client = _local_client(provider, provider_name="lmstudio", model="qwen")
    client._prompt_cache_state["session:abc"] = object()  # type: ignore[index]

    result = client.unload_model_residency(task="text_generation", provider="lmstudio", model="qwen")

    assert result["ok"] is True
    assert provider.unload_model_calls == ["qwen"]
    assert client._prompt_cache_state == {}


def test_multilocal_unload_drops_the_pooled_client_mirrors(monkeypatch) -> None:
    import abstractruntime.integrations.abstractcore.llm_client as llm_mod

    class _DummyLocal:
        def __init__(self, *, provider: str, model: str, llm_kwargs: Dict[str, Any], artifact_store: Any) -> None:
            _ = llm_kwargs, artifact_store
            self._provider = provider
            self._model = model
            self._llm = _ResidentProvider(provider=provider, model=model, loaded=True)
            self.dropped = 0

        def get_model_capabilities(self) -> Dict[str, Any]:
            return {}

        def _drop_prompt_cache_client_state(self) -> None:
            self.dropped += 1

    monkeypatch.setattr(llm_mod, "LocalAbstractCoreLLMClient", _DummyLocal)
    client = MultiLocalAbstractCoreLLMClient(provider="lmstudio", model="default")
    pooled = client._default_client

    result = client.unload_model_residency(task="text_generation", provider="lmstudio", model="default")

    assert result["ok"] is True
    assert pooled.dropped == 1
