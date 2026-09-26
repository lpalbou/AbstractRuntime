"""Clean eject across every path (mission M2, 2026-09-25).

1. A default-model switch used to drop the old pool clients WITHOUT unloading
   them: with the chat summarizer pinning the boot-time instance, 17.36 GB of
   the previous default stayed resident (hermetic M1 run). The switch now runs
   the process-wide eject on every dropped in-process model nothing in the
   pool still uses, and the summarizer resolves the current default per call.
2. Embedding models are listed (`local:embedding:huggingface:<model>`) and
   ejectable through the same residency API.
3. The console sends only `runtime_id`: every `local:<task>:<provider>:<model>`
   id (TTS/STT/image, not only text) routes to the right facade.
4. Every listed capability task has a residency target: no listing carries a
   permanent `task_errors` entry for video / upscale / text_to_audio.
Fakes only: no model, no MLX, no torch.
"""
from __future__ import annotations

import gc
import sys
import threading
import time
import types
import weakref
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

import abstractruntime.integrations.abstractcore.llm_client as llm_mod
from abstractruntime.integrations.abstractcore.llm_client import MultiLocalAbstractCoreLLMClient


class _FakeInflight:
    def __init__(self):
        self.n = 0

    def active(self):
        return self.n


class _FakeProvider:
    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = model
        self.inflight = _FakeInflight()

    def _inflight_generations(self):
        return self.inflight


import abstractcore.providers.process_residency as core_pr


@pytest.fixture
def pool(monkeypatch):
    gc.collect()  # clients of earlier tests must not claim models here
    ejects: List[tuple] = []
    monkeypatch.setattr(core_pr, "eject",
                        lambda b, m, reason="eject": (ejects.append((b, m)), {"ok": True, "holders_found": 1})[1])
    monkeypatch.setattr(llm_mod, "_mlx_process_residency_rows", lambda: [])
    monkeypatch.setattr(llm_mod, "_hf_process_residency_rows", lambda: [])

    def fake_create(self, provider, model, *, llm_kwargs_override=None):
        return SimpleNamespace(_llm=_FakeProvider(provider, model), _drop_prompt_cache_client_state=lambda: None)

    monkeypatch.setattr(MultiLocalAbstractCoreLLMClient, "_create_client", fake_create)
    client = MultiLocalAbstractCoreLLMClient(provider="mlx", model="vendor/A")
    return client, ejects


def test_switching_the_default_ejects_the_previous_in_process_model(pool):
    client, ejects = pool
    old = client._llm
    assert client.set_default_provider_model(provider="mlx", model="vendor/B") is True
    assert ejects == [("mlx", "vendor/A")], "the dropped default must be ejected process-wide (MUTANT: no eject -> RED)"
    assert client._llm is not old and client._llm.model == "vendor/B"
    assert client._last_switch_ejects["mlx/vendor/A"]["ok"] is True


def test_a_model_the_new_pool_still_uses_or_a_locked_one_is_not_ejected(pool):
    client, ejects = pool
    client._get_client("huggingface", "vendor/H")           # a per-call override in the pool
    client._locked_model_residency.add(("huggingface", "vendor/H"))
    client._get_client("mlx", "vendor/C")
    # Switching TO a model already pooled keeps it; A and C are dropped.
    client.set_default_provider_model(provider="mlx", model="vendor/C")
    assert sorted(ejects) == [("mlx", "vendor/A")], "locked H survives; C is the new default"


def test_remote_providers_are_never_ejected(pool):
    client, ejects = pool
    client._get_client("ollama", "gemma3:1b")
    client.set_default_provider_model(provider="lmstudio", model="qwen")
    assert ("ollama", "gemma3:1b") not in ejects and ("mlx", "vendor/A") in ejects


def test_a_switch_never_cancels_a_running_generation_it_ejects_after(pool):
    client, ejects = pool
    old = client._llm
    old.inflight.n = 1                                       # still generating
    client.set_default_provider_model(provider="mlx", model="vendor/B")
    assert ejects == [] and client._last_switch_ejects["mlx/vendor/A"]["deferred"] is True
    old.inflight.n = 0                                       # the call ends
    deadline = time.time() + 5
    while not ejects and time.time() < deadline:
        time.sleep(0.05)
    assert ejects == [("mlx", "vendor/A")]


def test_capability_route_change_unloads_the_old_capability_core_residents(pool):
    client, _ = pool
    unloaded: List[Dict[str, Any]] = []

    class _Facade:
        def list_resident_models(self, filters):
            return [{"task": "tts", "provider": "abstractvoice", "model": "kokoro", "load_id": "v/kokoro", "resident": True}]

        def unload_resident_model(self, selector):
            unloaded.append(dict(selector))
            return {"unloaded": True}

    client._capability_residency_core = SimpleNamespace(voice=_Facade())
    client.set_capability_defaults({"output": {"audio": {"provider": "abstractvoice", "model": "other"}}})
    assert client._capability_residency_core is None
    assert unloaded == [{"task": "tts", "provider": "abstractvoice", "model": "kokoro", "load_id": "v/kokoro"}]


def test_text_only_switch_keeps_the_capability_core(pool):
    client, _ = pool
    core = SimpleNamespace()
    client._capability_residency_core = core
    client.set_default_provider_model(provider="mlx", model="vendor/B")
    assert client._capability_residency_core is core


# -- summarizer ---------------------------------------------------------------
def test_chat_summarizer_follows_the_current_default_and_pins_nothing(monkeypatch):
    from abstractruntime.integrations.abstractcore.summarizer import AbstractCoreChatSummarizer

    seen: List[Any] = []

    class _Basic:
        def __init__(self, llm, **kwargs):
            self.llm = llm

        def summarize_chat_history(self, **kwargs):
            seen.append(self.llm)
            return SimpleNamespace(summary="s", key_points=[], confidence=1.0, focus_alignment=1.0,
                                   word_count_original=1, word_count_summary=1)

    import abstractcore.processing as processing
    monkeypatch.setattr(processing, "BasicSummarizer", _Basic)

    class _LLM:
        pass

    holder = SimpleNamespace(_llm=_LLM())
    summarizer = AbstractCoreChatSummarizer(llm_resolver=lambda provider, model: holder._llm)
    summarizer.summarize_chat_history([{"role": "user", "content": "x"}])
    first = weakref.ref(holder._llm)
    holder._llm = _LLM()                                     # the default switched
    summarizer.summarize_chat_history([{"role": "user", "content": "y"}])
    assert seen[1] is holder._llm, "summarizes with the CURRENT default"
    seen.clear()
    gc.collect()
    assert first() is None, "the previous default is not retained by the summarizer"

    holder._llm = None
    with pytest.raises(RuntimeError, match="no default model"):
        summarizer.summarize_chat_history([{"role": "user", "content": "z"}])


def test_summarizer_uses_the_runs_own_model_and_never_loads_the_default(pool, monkeypatch):
    """S7. MUTANT: the resolver ignores the run's route -> the default's
    instance summarizes (and would be loaded just for that) -> RED."""
    from abstractruntime.integrations.abstractcore import factory
    from abstractruntime.integrations.abstractcore.summarizer import AbstractCoreChatSummarizer

    client, _ = pool
    seen: List[Any] = []

    class _Basic:
        def __init__(self, llm, **kwargs):
            self.llm = llm

        def summarize_chat_history(self, **kwargs):
            seen.append(self.llm)
            return SimpleNamespace(summary="s", key_points=[], confidence=1.0, focus_alignment=1.0,
                                   word_count_original=1, word_count_summary=1)

    import abstractcore.processing as processing
    monkeypatch.setattr(processing, "BasicSummarizer", _Basic)
    summarizer = AbstractCoreChatSummarizer(
        llm_resolver=lambda provider=None, model=None: factory._summarizer_llm(client, provider, model))
    out = summarizer.summarize_chat_history([{"role": "user", "content": "x"}], provider="mlx", model="vendor/RUN")
    assert seen[-1].model == "vendor/RUN" and out["model"] == "vendor/RUN"
    out = summarizer.summarize_chat_history([{"role": "user", "content": "x"}])
    assert seen[-1] is client._llm and out["model"] == "vendor/A"


def test_compaction_passes_the_runs_route_to_the_summarizer():
    import inspect

    from abstractruntime.core import runtime as rt_mod

    source = inspect.getsource(rt_mod.Runtime)
    assert 'summarize_kwargs = {"provider": route_provider, "model": route_model}' in source
    assert rt_mod._accepts_kwarg(lambda messages, **kw: None, "model")
    assert not rt_mod._accepts_kwarg(lambda messages, preserve_recent=6: None, "model")


def test_factory_wires_the_summarizer_lazily():
    import inspect

    from abstractruntime.integrations.abstractcore import factory

    source = inspect.getsource(factory)
    assert "llm=llm_client._llm" not in source, "a captured boot-time instance pins the old default"
    assert "llm_resolver=" in source


# -- embeddings ---------------------------------------------------------------
@pytest.fixture
def fake_embeddings(monkeypatch):
    state = {"rows": [{"backend": "embeddings", "models": ["st/mini"], "holders": 2, "weights_bytes": 90,
                       "held_bytes": 90, "weights_alive": True, "device": "mps:0"}], "ejects": []}
    mod = types.ModuleType("abstractcore.embeddings.manager")
    mod.resident_embedding_models = lambda: [dict(r) for r in state["rows"]]

    def eject(model, reason="eject"):
        state["ejects"].append(model)
        state["rows"] = [r for r in state["rows"] if model not in r["models"]]
        return {"ok": True, "holders_found": 2, "holders_unloaded": [{"id": 1}, {"id": 2}], "residual": None}

    mod.eject_embedding_models = eject
    monkeypatch.setitem(sys.modules, "abstractcore.embeddings.manager", mod)
    return state


def test_embedding_models_are_listed_and_ejectable_by_runtime_id(pool, fake_embeddings):
    client, _ = pool
    listing = client.list_model_residency()
    rows = [m for m in listing["models"] if m.get("task") == "embedding"]
    assert len(rows) == 1
    row = rows[0]
    assert row["runtime_id"] == "local:embedding:huggingface:st/mini" and row["backend"] == "embeddings"
    assert row["held_bytes"] == 90 and row["process_holders"] == 2 and row["resident"] is True
    assert "embedding" not in (listing["diagnostics"].get("task_errors") or {})

    result = client.unload_model_residency(runtime_id="local:embedding:huggingface:st/mini")
    assert result["ok"] is True and result["unloaded"] is True and fake_embeddings["ejects"] == ["st/mini"]
    assert not [m for m in client.list_model_residency(task="embedding")["models"]]


def test_embedding_listing_without_any_embedder_is_empty_not_an_error(pool, monkeypatch):
    client, _ = pool
    monkeypatch.delitem(sys.modules, "abstractcore.embeddings.manager", raising=False)
    listing = client.list_model_residency(task="embedding")
    assert listing["ok"] is True and listing["models"] == []


def test_remote_embedder_provider_filter_lists_nothing_and_refuses_unload(pool, fake_embeddings):
    client, _ = pool
    assert client.list_model_residency(task="embedding", provider="ollama")["models"] == []
    out = client.unload_model_residency(task="embedding", provider="ollama", model="nomic")
    assert out["ok"] is False and "own server" in out["error"]


# -- non-text eject by runtime_id ------------------------------------------------
def test_console_eject_of_a_tts_row_by_runtime_id_reaches_the_voice_facade(pool):
    client, _ = pool
    calls: List[Dict[str, Any]] = []

    class _Voice:
        def unload_resident_model(self, payload):
            calls.append(dict(payload))
            return {"task": "tts", "provider": payload["provider"], "model": payload["model"], "unloaded": True}

    client._capability_residency_core = SimpleNamespace(voice=_Voice())
    result = client.unload_model_residency(runtime_id="local:tts:abstractvoice:kokoro:v1")
    assert result["ok"] is True and result["task"] == "tts"
    assert calls and calls[0]["provider"] == "abstractvoice" and calls[0]["model"] == "kokoro:v1"
    assert "load_id" not in calls[0], "a synthesized local: id is not the plugin's id"


def test_console_eject_by_a_plugin_own_id_resolves_through_the_listing(pool, monkeypatch):
    client, _ = pool
    calls: List[Dict[str, Any]] = []

    class _Vision:
        def list_loaded_models(self, filters):
            return [{"task": "text_to_image", "provider": "huggingface", "model": "flux", "load_id": "diffusers/flux",
                     "resident": True, "loaded": True}]

        def unload_resident_model(self, payload):
            calls.append(dict(payload))
            return {"task": "text_to_image", "provider": "huggingface", "model": "flux", "unloaded": True}

    client._capability_residency_core = SimpleNamespace(vision=_Vision())
    result = client.unload_model_residency(runtime_id="diffusers/flux")
    assert result["ok"] is True and calls and calls[0]["load_id"] == "diffusers/flux"


def test_parse_local_runtime_id_forms():
    parse = llm_mod._parse_local_residency_runtime_id
    assert parse("local:text_generation:ollama:gemma3:1b") == ("text_generation", "ollama", "gemma3:1b")
    assert parse("local:stt:abstractvoice:whisper") == ("stt", "abstractvoice", "whisper")
    assert parse("local:text_generation:endpoint:lab:qwen") == ("text_generation", "endpoint:lab", "qwen")
    assert parse("local:embedding:huggingface:st/mini") == ("embedding", "huggingface", "st/mini")
    assert parse("diffusers/flux") is None and parse("local:tts:") is None


# -- listed tasks all have a target ----------------------------------------------
def test_every_listed_capability_task_has_a_residency_target():
    core = SimpleNamespace(vision="V", voice="VO", audio="A", music="M")
    for task in llm_mod._LOCAL_CAPABILITY_RESIDENCY_LIST_TASKS:
        if task == "embedding":
            continue
        target, _ = llm_mod._local_capability_residency_target(core, task)   # MUTANT: unmapped task -> ValueError
        assert target in {"V", "VO", "A", "M"}, task


def test_all_listing_has_no_permanent_task_errors_with_working_facades(pool):
    client, _ = pool

    class _Facade:
        def list_loaded_models(self, filters):
            return []

    client._capability_residency_core = SimpleNamespace(vision=_Facade(), voice=_Facade(), audio=_Facade(),
                                                        music=_Facade())
    listing = client.list_model_residency()
    assert listing["ok"] is True
    assert not listing["diagnostics"].get("task_errors"), listing["diagnostics"].get("task_errors")


def test_the_previous_model_is_ejected_before_the_new_default_loads(monkeypatch, pool):
    """Two 17 GB models must never be resident together because of a switch."""
    client, ejects = pool
    order: List[str] = []
    monkeypatch.setattr(core_pr, "eject", lambda b, m, reason="eject": (order.append(f"eject {m}"), {"ok": True})[1])
    original = MultiLocalAbstractCoreLLMClient._create_client

    def recording_create(self, provider, model, *, llm_kwargs_override=None):
        order.append(f"build {model}")
        return original(self, provider, model, llm_kwargs_override=llm_kwargs_override)

    monkeypatch.setattr(MultiLocalAbstractCoreLLMClient, "_create_client", recording_create)
    client.set_default_provider_model(provider="mlx", model="vendor/B")
    assert order == ["eject vendor/A", "build vendor/B"]


# -- review follow-up (2026-09-26) ---------------------------------------------------
def test_a_switch_never_ejects_a_model_another_client_in_the_process_still_holds(pool):
    """S2: two services (users / entities) in one process. MUTANT: judge
    "unused" from this client's pool only -> B's model is ejected -> RED."""
    a, ejects = pool
    b = MultiLocalAbstractCoreLLMClient(provider="mlx", model="vendor/A")      # user B, same model
    a.set_default_provider_model(provider="mlx", model="vendor/B")
    assert ejects == [], "B still has vendor/A as its default"
    assert a._last_switch_ejects["mlx/vendor/A"]["skipped"] is True
    b._locked_model_residency.add(("mlx", "vendor/A"))
    b.set_default_provider_model(provider="mlx", model="vendor/C")
    assert ejects == [], "B's LOCK keeps it"
    b._locked_model_residency.clear()
    # B's capability change evicts its pool: now NOBODY claims vendor/A.
    b.set_capability_defaults({"output": {"audio": {"provider": "x", "model": "y"}}})
    assert ("mlx", "vendor/A") in ejects and ("mlx", "vendor/B") not in ejects and ("mlx", "vendor/C") not in ejects


def test_a_model_being_built_by_another_client_is_not_ejected(pool):
    a, ejects = pool
    b = MultiLocalAbstractCoreLLMClient(provider="lmstudio", model="q")
    b._building[("mlx", "vendor/A")] = 1                                       # B is loading it right now
    a.set_default_provider_model(provider="mlx", model="vendor/B")
    assert ejects == [] and a._last_switch_ejects["mlx/vendor/A"]["claims"][0]["kind"] == "building"
    b._building.clear()


def test_deferred_eject_is_visible_and_rechecks_claims_before_ejecting(pool):
    """S3: pending state in the listing; a rebuild while waiting cancels the eject."""
    a, ejects = pool
    old = a._llm
    old.inflight.n = 1
    a.set_default_provider_model(provider="mlx", model="vendor/B")
    diag = a.list_model_residency(task="text_generation")["diagnostics"]
    assert diag["pending_ejects"] and diag["pending_ejects"][0]["model"] == "vendor/A"
    assert diag["pending_ejects"][0]["reason"] == "waiting for the in-flight call"
    assert diag["last_switch_ejects"][0]["deferred"] is True
    a._get_client("mlx", "vendor/A")                                           # back in use meanwhile
    old.inflight.n = 0
    deadline = time.time() + 5
    while a._pending_ejects and time.time() < deadline:
        time.sleep(0.05)
    assert ejects == [], "re-checked under the lock: now claimed again"
    last = a.list_model_residency(task="text_generation")["diagnostics"]["last_switch_ejects"]
    assert last[0]["skipped"] is True and a.list_model_residency()["diagnostics"]["pending_ejects"] == []


class _LoadableProvider(_FakeProvider):
    def __init__(self, provider, model, *, loaded=False, fail=False, warnings=None):
        super().__init__(provider, model)
        self.loaded, self.fail, self.warnings = loaded, fail, warnings or []

    def get_model_residency(self, **kwargs):
        return {"task": "text_generation", "provider": self.provider, "model": self.model,
                "provider_residency_verified": True, "provider_resident": self.loaded, "loaded": self.loaded,
                "state": "loaded" if self.loaded else "not_loaded", "source": "fake"}

    def load_model(self, name, **kwargs):
        if self.fail:
            raise RuntimeError("drafter failed after the main weights loaded")
        self.loaded = True
        out = {"action": "loaded"}
        if self.warnings:
            out["warnings"] = list(self.warnings)
            out["unsupported_options"] = ["ttl_s"]
        return out


@pytest.fixture
def loadable(monkeypatch, pool):
    client, ejects = pool
    made: Dict[tuple, Any] = {}

    def create(self, provider, model, *, llm_kwargs_override=None):
        spec = made.get((provider, model)) or {}
        return SimpleNamespace(_llm=_LoadableProvider(provider, model, **spec),
                               _drop_prompt_cache_client_state=lambda: None)

    monkeypatch.setattr(MultiLocalAbstractCoreLLMClient, "_create_client", create)
    return client, ejects, made


def test_ttl_on_an_already_loaded_model_is_reported_not_applied(loadable):
    """S6. MUTANT: skip the report when the provider load did not run -> RED."""
    client, _, made = loadable
    made[("mlx", "vendor/M")] = {"loaded": True}
    out = client.load_model_residency(task="text_generation", provider="mlx", model="vendor/M", options={"ttl_s": 20})
    assert out["ok"] is True and out["unsupported_options"] == ["ttl_s"]
    assert any("no idle/TTL unload" in w for w in out["warnings"])
    made[("ollama", "g")] = {"loaded": True}
    out = client.load_model_residency(task="text_generation", provider="ollama", model="g", keep_alive="5m")
    assert out["unsupported_options"] == ["keep_alive"] and "already loaded" in out["warnings"][0]


def test_provider_load_warnings_reach_the_top_level_warnings(loadable):
    client, _, made = loadable
    made[("mlx", "vendor/W")] = {"warnings": ["MLX has no idle/TTL unload: ttl_s not applied"]}
    out = client.load_model_residency(task="text_generation", provider="mlx", model="vendor/W", options={"ttl_s": 5})
    assert out["warnings"] == ["MLX has no idle/TTL unload: ttl_s not applied"]
    assert out["unsupported_options"] == ["ttl_s"]


def test_a_failed_load_ejects_its_partial_weights(loadable):
    """Nit 8. MUTANT: plain pool pop -> partial weights stay -> RED."""
    client, ejects, made = loadable
    made[("mlx", "vendor/F")] = {"fail": True}
    out = client.load_model_residency(task="text_generation", provider="mlx", model="vendor/F")
    assert out["ok"] is False and ("mlx", "vendor/F") not in client._clients
    assert ejects == [("mlx", "vendor/F")]
    assert client.list_model_residency()["diagnostics"]["last_switch_ejects"][-1]["model"] == "vendor/F"
