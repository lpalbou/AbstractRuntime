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


@pytest.fixture
def pool(monkeypatch):
    ejects: List[tuple] = []
    monkeypatch.setattr(llm_mod, "_process_eject_for",
                        lambda p, m: (ejects.append((p, m)), {"ok": True, "holders_found": 1})[1])
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
    summarizer = AbstractCoreChatSummarizer(llm_resolver=lambda: holder._llm)
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
