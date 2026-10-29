"""A runtime client releases what it drops; holders are named in words.

Reproduced on the Linux test server (gateway 0.17.x, two accounts, 2026-10-29):
a console change to the EMBEDDING default evicted the admin runtime's whole
text pool; the boot-warmed `Qwen3.5-9B-Kimi-k3-Distilled-GGUF` copy was dropped,
its eject skipped because both accounts' runtimes claimed the gateway default
("still in use by runtime client 73a2242516a0, runtime client 73a46204fef0; not
ejected"), and the copy stayed resident with no pool holding it
(`resident_via_other_holders`).

Fakes only: no model.
"""
from __future__ import annotations

import gc
from types import SimpleNamespace
from typing import Any, List

import pytest

import abstractcore.providers.process_residency as core_pr
import abstractruntime.integrations.abstractcore.llm_client as llm_mod
from abstractruntime.integrations.abstractcore.llm_client import MultiLocalAbstractCoreLLMClient


class _Inflight:
    def __init__(self):
        self.n = 0

    def active(self):
        return self.n


class _Provider:
    def __init__(self, provider: str, model: str, builds: List[Any]):
        self.provider, self.model = provider, model
        self.inflight = _Inflight()
        self.unloaded: List[str] = []
        builds.append(self)

    def _inflight_generations(self):
        return self.inflight

    def unload_model(self, model: str) -> None:
        self.unloaded.append(model)


@pytest.fixture
def env(monkeypatch):
    gc.collect()
    ejects: List[tuple] = []
    builds: List[_Provider] = []
    monkeypatch.setattr(core_pr, "eject",
                        lambda b, m, reason="eject": (ejects.append((b, m)), {"ok": True, "holders_found": 1})[1])
    monkeypatch.setattr(llm_mod, "_mlx_process_residency_rows", lambda: [])
    monkeypatch.setattr(llm_mod, "_hf_process_residency_rows", lambda: [])

    def fake_create(self, provider, model, *, llm_kwargs_override=None):
        return SimpleNamespace(_llm=_Provider(provider, model, builds), _drop_prompt_cache_client_state=lambda: None)

    monkeypatch.setattr(MultiLocalAbstractCoreLLMClient, "_create_client", fake_create)
    return ejects, builds


HF = ("huggingface", "mradermacher/Qwen3.5-9B-Kimi-k3-Distilled-GGUF")
ROUTES_A = {"routes": [{"key": "input.text", "provider": HF[0], "model": HF[1]},
                       {"key": "embedding.text", "provider": "huggingface", "model": "qwen3-embedding"}]}
ROUTES_B = {"routes": [{"key": "input.text", "provider": HF[0], "model": HF[1]},
                       {"key": "embedding.text", "provider": "huggingface", "model": "other-embedding"}]}


def test_a_non_text_route_change_keeps_the_text_model(env):
    """THE SERVER BUG. MUTANT: evict the pool on every capability change -> a
    second build and a dropped, orphaned copy -> RED."""
    ejects, builds = env
    admin = MultiLocalAbstractCoreLLMClient(provider=HF[0], model=HF[1])
    admin.set_capability_defaults(ROUTES_A)      # boot: first push of the config payload
    admin.set_capability_defaults(ROUTES_B)      # the operator changes the embedding model
    assert len(builds) == 1, "the text model was rebuilt for an embedding change"
    assert admin._clients[HF]._llm is builds[0] and ejects == [] and builds[0].unloaded == []
    assert admin._clients[HF]._capability_defaults["embedding.text"]["model"] == "other-embedding"


def test_a_text_route_change_still_rebuilds(env):
    _, builds = env
    c = MultiLocalAbstractCoreLLMClient(provider=HF[0], model=HF[1])
    c.set_capability_defaults(ROUTES_A)
    moved = {"routes": [{"key": "input.text", "provider": HF[0], "model": HF[1], "options": {"speculation": True}}]}
    c.set_capability_defaults(moved)
    assert len(builds) == 2, "a load-time text option change must rebuild the text client"


def test_an_unbuilt_default_does_not_keep_another_holders_copy(env, monkeypatch):
    """bob's runtime names the same default but never built it. MUTANT: the
    "default" claim -> the dropped copy is kept for nobody -> RED."""
    ejects, builds = env
    monkeypatch.setattr(llm_mod, "_WARM_DEFAULT_MODEL", False)
    bob = MultiLocalAbstractCoreLLMClient(provider=HF[0], model=HF[1])     # holds nothing
    monkeypatch.setattr(llm_mod, "_WARM_DEFAULT_MODEL", True)
    admin = MultiLocalAbstractCoreLLMClient(provider=HF[0], model=HF[1])
    admin.set_default_provider_model(provider="huggingface", model="vendor/other")
    assert ejects[0] == HF, f"nobody holds the 9B any more; it must be ejected (got {ejects})"
    assert bob._clients == {}


def test_a_dropped_private_copy_is_released_when_another_runtime_keeps_its_own(env):
    """bob's runtime really holds its own copy: the process-wide eject is
    skipped, and admin's own dropped copy must still be freed. MUTANT: no
    release -> admin's copy stays resident, held by no pool -> RED."""
    ejects, builds = env
    bob = MultiLocalAbstractCoreLLMClient(provider=HF[0], model=HF[1])
    admin = MultiLocalAbstractCoreLLMClient(provider=HF[0], model=HF[1])
    bob_copy, admin_copy = builds[0], builds[1]
    admin.set_default_provider_model(provider="huggingface", model="vendor/other")
    report = admin._last_switch_ejects[f"{HF[0]}/{HF[1]}"]
    assert ejects == [] and report["skipped"] is True
    assert admin_copy.unloaded == [HF[1]], "admin dropped its copy but never freed it"
    assert bob_copy.unloaded == [], "bob's copy is bob's"
    assert report["released_own_copies"] == 1


def test_holders_are_named_in_words_never_by_id(env):
    """ADR 0043. MUTANT: the reason built from `owner` ids -> "runtime client 7f..." -> RED."""
    ejects, builds = env
    bob = MultiLocalAbstractCoreLLMClient(provider=HF[0], model=HF[1])
    bob.residency_owner_label = "the runtime of account default/bob"
    admin = MultiLocalAbstractCoreLLMClient(provider=HF[0], model=HF[1])
    admin.residency_owner_label = "the runtime of account default/admin"
    admin.set_default_provider_model(provider="huggingface", model="vendor/other")
    reason = admin._last_switch_ejects[f"{HF[0]}/{HF[1]}"]["reason"]
    assert "the runtime of account default/bob (loaded)" in reason, reason
    assert "runtime client" not in reason, reason
    rows = llm_mod._stamp_readable_holders([{"provider": HF[0], "model": HF[1], "resident": True}])
    assert rows[0]["held_by"] == ["the runtime of account default/bob (loaded)"]
    orphan = llm_mod._stamp_readable_holders([{"provider": "huggingface", "model": "nobody/holds", "resident": True}])
    assert orphan[0]["held_by"] == ["no runtime uses it (unloading it frees the memory)"]


def test_a_busy_dropped_copy_is_freed_when_its_call_ends_and_never_lent(env):
    """Review: the release must not cancel a running call nor let /v1 borrow
    the dropped copy. MUTANT: release ignores in-flight -> unloaded mid-call -> RED."""
    import time as _t
    ejects, builds = env
    bob = MultiLocalAbstractCoreLLMClient(provider=HF[0], model=HF[1])
    admin = MultiLocalAbstractCoreLLMClient(provider=HF[0], model=HF[1])
    admin_copy = builds[1]
    admin_copy.inflight.n = 1                      # a run is still generating on it
    admin.set_default_provider_model(provider="huggingface", model="vendor/other")
    assert admin_copy.unloaded == [] and getattr(admin_copy, "_abstractcore_request_scoped", False) is True
    admin_copy.inflight.n = 0                      # the call ends
    deadline = _t.time() + 5
    while not admin_copy.unloaded and _t.time() < deadline:
        _t.sleep(0.05)
    assert admin_copy.unloaded == [HF[1]] and builds[0].unloaded == []


def test_a_standalone_local_client_claims_the_model_it_holds(monkeypatch):
    """ROOT backlog 0897. MUTANT: no claim -> another client's switch ejects
    the weights this client still uses -> RED."""
    from abstractruntime.integrations.abstractcore.llm_client import LocalAbstractCoreLLMClient
    gc.collect()
    ejects: List[tuple] = []
    monkeypatch.setattr(core_pr, "eject",
                        lambda b, m, reason="eject": (ejects.append((b, m)), {"ok": True, "holders_found": 1})[1])
    monkeypatch.setattr(llm_mod, "_mlx_process_residency_rows", lambda: [])
    monkeypatch.setattr(llm_mod, "_hf_process_residency_rows", lambda: [])
    local = LocalAbstractCoreLLMClient(provider=HF[0], model=HF[1],
                                       llm_factory=lambda model, **kw: SimpleNamespace(llm=object(), model=model))
    builds: List[Any] = []

    def fake_create(self, provider, model, *, llm_kwargs_override=None):
        return SimpleNamespace(_llm=_Provider(provider, model, builds), _drop_prompt_cache_client_state=lambda: None)

    monkeypatch.setattr(MultiLocalAbstractCoreLLMClient, "_create_client", fake_create)
    pool = MultiLocalAbstractCoreLLMClient(provider=HF[0], model=HF[1])
    pool.set_default_provider_model(provider="huggingface", model="vendor/other")
    assert ejects == [], "the standalone client still holds the model"
    local._llm.llm = None                          # it unloads: the claim goes with the weights
    assert local.residency_claims() == []


def test_release_never_unloads_a_copy_a_call_just_started_on(env):
    ejects, builds = env
    c = MultiLocalAbstractCoreLLMClient(provider="lmstudio", model="q")
    racing = _Provider(HF[0], HF[1], builds)
    racing.inflight.n = 1
    out = c._release_own_copies_if_kept(HF[0], HF[1], [racing], {"ok": True, "skipped": True, "reason": "x"})
    assert racing.unloaded == [] and out["release_deferred"] == 1
    racing.inflight.n = 0
    import time as _t
    deadline = _t.time() + 5
    while c._pending_ejects and _t.time() < deadline:   # let the waiter finish HERE: a leaked
        _t.sleep(0.05)                                 # waiter ejects into the next test's fixture
    assert not c._pending_ejects


def test_a_client_a_pool_adopts_does_not_claim_on_its_own():
    """0897's other half. MUTANT: the pool keeps its clients registered ->
    a model the pool dropped stays "in use" by the dropped client -> RED."""
    from abstractruntime.integrations.abstractcore.llm_client import LocalAbstractCoreLLMClient
    gc.collect()
    local = LocalAbstractCoreLLMClient(provider=HF[0], model="vendor/adopted",
                                       llm_factory=lambda model, **kw: SimpleNamespace(llm=object(), model=model))
    assert core_pr.claims_for(HF[0], "vendor/adopted"), "a standalone client claims"
    pool = MultiLocalAbstractCoreLLMClient(provider="lmstudio", model="q")
    pool._adopt_client(local)
    assert core_pr.claims_for(HF[0], "vendor/adopted") == []
