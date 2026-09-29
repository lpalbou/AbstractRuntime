"""A resident (explicitly loaded) image model serves generation (framework backlog 0991).

Before 2026-09-29 `load_model_residency(task="image_generation")` -- the Gateway's
`/models/load`, the console Load button, a flow's model_residency node -- loaded
the weights into the client's capability residency core, while every image
request ran in a one-shot subprocess that loaded the model AGAIN (54-59 s per
FLUX.2 klein image on CUDA vs 15-16 s warm, and a second copy on the GPU).

Contract pinned here:
- a request whose every media spec names a model RESIDENT in the residency core
  runs in-process on that core (no subprocess, no second load);
- a pooled client (MultiLocal) finds its pool owner's residency core;
- anything not explicitly loaded (other model, active-but-not-resident, no core)
  keeps the isolated subprocess path.
Fakes only: no model, no torch.
"""
from __future__ import annotations

import gc
import threading
import weakref
from typing import Any, Dict, List, Optional

import pytest

import abstractruntime.integrations.abstractcore.llm_client as llm_mod
from abstractruntime.integrations.abstractcore.llm_client import (
    LocalAbstractCoreLLMClient,
    MultiLocalAbstractCoreLLMClient,
)
from abstractruntime.storage.artifacts import InMemoryArtifactStore


class _FakeVisionFacade:
    """The residency + t2i surface of an AbstractCore vision facade."""

    def __init__(self, resident: List[Dict[str, Any]]):
        self.resident = resident
        self.t2i_calls: List[Dict[str, Any]] = []

    def list_loaded_models(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        f = dict(filters or {})
        out = []
        for row in self.resident:
            if f.get("model") and row.get("model") != f["model"]:
                continue
            if f.get("resident") is not None and bool(row.get("resident")) is not bool(f["resident"]):
                continue
            out.append(dict(row))
        return out

    def t2i(self, prompt: str, **kwargs: Any) -> bytes:
        self.t2i_calls.append({"prompt": prompt, **kwargs})
        return b"png-from-resident-pipeline"


def _resident_core(resident: List[Dict[str, Any]]):
    from abstractcore.server.capability_generation import create_capability_generation_core

    facade = _FakeVisionFacade(resident)
    return create_capability_generation_core(vision_facade=facade), facade


def _client(store) -> LocalAbstractCoreLLMClient:
    class _RouteLLM:
        def _run_multimodal_spec(self, **_kwargs):
            raise AssertionError("the route provider's own vision plugin must not load a second copy")

    client = object.__new__(LocalAbstractCoreLLMClient)
    client._provider = "lmstudio"
    client._model = "qwen3.5-9b"
    client._llm_kwargs = {}
    client._artifact_store = store
    client._generate_lock = None
    client._llm = _RouteLLM()
    client._maybe_prepare_prompt_cache = lambda **_kwargs: None
    client._capability_residency_core = None
    client._capability_residency_core_lock = threading.Lock()
    return client


_KLEIN = "black-forest-labs/FLUX.2-klein-4B"
_RESIDENT_ROW = {
    "task": "text_to_image",
    "provider": "huggingface",
    "model": _KLEIN,
    "load_id": f"diffusers/{_KLEIN}",
    "resident": True,
    "loaded": True,
    "state": "resident",
}


def _image_params() -> Dict[str, Any]:
    return {
        "output": {"modality": "image", "provider": "huggingface", "model": _KLEIN, "run_id": "run-img"},
        "trace_metadata": {"run_id": "run-img"},
    }


@pytest.fixture
def subprocess_calls(monkeypatch):
    calls: List[Dict[str, Any]] = []

    def fake_subprocess(**kwargs):
        calls.append(kwargs)
        return {
            "outputs": {"image": [{"modality": "image", "task": "image_generation", "data": b"png-from-subprocess",
                                   "content_type": "image/png", "format": "png",
                                   "provider": "huggingface", "model": kwargs["specs"][0].get("model")}]},
            "metadata": {"media_only": True, "execution_mode": "local_one_shot_subprocess"},
        }

    monkeypatch.setattr(llm_mod, "_run_local_image_subprocess", fake_subprocess)
    return calls


def test_resident_image_model_serves_generation_in_process(subprocess_calls) -> None:
    store = InMemoryArtifactStore()
    client = _client(store)
    core, facade = _resident_core([dict(_RESIDENT_ROW)])
    client._capability_residency_core = core

    out = client.generate(prompt="A red mug.", params=_image_params())

    assert subprocess_calls == [], "a resident model must not be reloaded in a subprocess"
    assert len(facade.t2i_calls) == 1
    assert facade.t2i_calls[0]["model"] == _KLEIN
    assert facade.t2i_calls[0]["provider"] == "huggingface"
    item = out["outputs"]["image"][0]
    assert store.load(item["artifact_id"]).content == b"png-from-resident-pipeline"


def test_pooled_client_uses_the_pool_owners_residency_core(subprocess_calls) -> None:
    store = InMemoryArtifactStore()
    client = _client(store)

    class _Owner:
        pass

    owner = _Owner()
    core, facade = _resident_core([dict(_RESIDENT_ROW)])
    owner._capability_residency_core = core
    client._capability_residency_parent = weakref.ref(owner)

    client.generate(prompt="A red mug.", params=_image_params())

    assert subprocess_calls == []
    assert len(facade.t2i_calls) == 1


@pytest.mark.parametrize(
    "resident",
    [
        [],  # nothing loaded
        [dict(_RESIDENT_ROW, model="other/model", load_id="diffusers/other/model")],  # another model
        [dict(_RESIDENT_ROW, resident=False, state="active")],  # used once, never explicitly loaded
    ],
    ids=["nothing-loaded", "other-model-resident", "active-not-resident"],
)
def test_without_an_explicit_resident_load_generation_stays_isolated(subprocess_calls, resident) -> None:
    store = InMemoryArtifactStore()
    client = _client(store)
    core, facade = _resident_core(resident)
    client._capability_residency_core = core

    out = client.generate(prompt="A red mug.", params=_image_params())

    assert len(subprocess_calls) == 1
    assert facade.t2i_calls == []
    assert store.load(out["outputs"]["image"][0]["artifact_id"]).content == b"png-from-subprocess"


def test_no_residency_core_keeps_the_subprocess_and_creates_no_core(subprocess_calls) -> None:
    store = InMemoryArtifactStore()
    client = _client(store)

    client.generate(prompt="A red mug.", params=_image_params())

    assert len(subprocess_calls) == 1
    assert client._capability_residency_core is None, "a generation must not create a residency core"


def test_multilocal_pool_clients_point_at_their_owner(monkeypatch) -> None:
    gc.collect()
    built: List[Any] = []

    class _FakeLocal:
        def __init__(self, *, provider, model, llm_kwargs=None, artifact_store=None, **_kw):
            self._provider, self._model = provider, model
            self._llm = None
            built.append(self)

        def set_on_token(self, _cb):
            return None

    monkeypatch.setattr(llm_mod, "LocalAbstractCoreLLMClient", _FakeLocal)
    pool = MultiLocalAbstractCoreLLMClient(provider="lmstudio", model="qwen3.5-9b")
    client = pool._get_client("lmstudio", "qwen3.5-9b")

    assert built and client is built[-1]
    parent_ref = getattr(client, "_capability_residency_parent", None)
    assert callable(parent_ref) and parent_ref() is pool
    pool._capability_residency_core = object()
    assert llm_mod._resident_capability_core_for(client) is pool._capability_residency_core


def test_real_abstractvision_plugin_resident_load_is_the_pipeline_generation_uses(monkeypatch, subprocess_calls) -> None:
    """End to end through the real AbstractCore capability core and the real
    AbstractVision plugin: the load's backend is preloaded ONCE and the image
    request runs on that same backend object (provider alias huggingface ==
    diffusers)."""
    plugin_mod = pytest.importorskip("abstractvision.integrations.abstractcore_plugin")
    from abstractvision.types import GeneratedAsset

    backends: List[Any] = []

    class _FakeDiffusersBackend:
        backend_kind = "diffusers"

        def __init__(self, model_id):
            self.model_id = model_id
            self.preloads = 0
            self.generations = 0
            self.unloaded = False
            backends.append(self)

        def preload(self):
            self.preloads += 1

        def unload(self):
            self.unloaded = True

        def get_capabilities(self):
            return None

        def generate_image(self, request):
            self.generations += 1
            return GeneratedAsset(media_type="image", data=b"png-real-plugin", mime_type="image/png")

    monkeypatch.setattr(
        plugin_mod._AbstractVisionCapability,
        "_make_diffusers_backend",
        lambda self, *, model_id=None: _FakeDiffusersBackend(model_id),
    )

    store = InMemoryArtifactStore()
    client = _client(store)
    loaded = llm_mod._local_capability_residency_result(
        client, operation="load", task="image_generation", provider="huggingface", model=_KLEIN, source="test",
    )
    assert loaded.get("ok") is True, loaded
    assert len(backends) == 1 and backends[0].preloads == 1

    out = client.generate(prompt="A red mug.", params=_image_params())
    client.generate(prompt="A blue mug.", params=_image_params())

    assert subprocess_calls == []
    assert len(backends) == 1, "generation must reuse the resident backend, never build a second one"
    assert backends[0].generations == 2
    assert store.load(out["outputs"]["image"][0]["artifact_id"]).content == b"png-real-plugin"

    # Unload returns generation to the isolated path.
    unloaded = llm_mod._local_capability_residency_result(
        client, operation="unload", task="image_generation", provider="huggingface", model=_KLEIN, source="test",
    )
    assert unloaded.get("ok") is True, unloaded
    assert backends[0].unloaded is True
    client.generate(prompt="A green mug.", params=_image_params())
    assert len(subprocess_calls) == 1
