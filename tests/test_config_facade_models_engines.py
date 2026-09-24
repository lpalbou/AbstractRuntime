"""Models & engines passthroughs in `config_facade` (AbstractCore >= 2.14.0).

The Gateway re-exposes AbstractCore's host profile, engines, catalog,
installed models, deletes and host jobs through these functions. The pins:

- every call reaches the Core function it names, with the host's arguments
  mapped one to one (fake Core modules installed in `sys.modules`);
- Core refusals arrive as `HostActionRefused` with the HTTP status and body
  Core's own server answers with, so a host never imports a Core exception;
- an AbstractCore without these modules raises `AbstractCoreTooOld` (a
  `NotImplementedError`) naming the installed version and the floor;
- one end-to-end pass against the REAL AbstractCore installed in the test
  environment (dry runs only, jobs kept in memory).
"""

from __future__ import annotations

import sys
import types
from typing import Any, Dict, List, Optional

import pytest

from abstractruntime.integrations.abstractcore import config_facade as facade

pytestmark = pytest.mark.basic

NEW_SURFACE = [
    "MODELS_ENGINES_MIN_ABSTRACTCORE",
    "AbstractCoreTooOld",
    "HostActionRefused",
    "models_engines_support",
    "host_profile",
    "engine_inventory",
    "engine_status",
    "engine_install_plan",
    "engine_download_url",
    "engine_install",
    "model_catalog",
    "list_installed_models",
    "model_delete_blockers",
    "delete_model_artifact",
    "start_model_download_job",
    "host_jobs_list",
    "host_job",
    "host_job_cancel",
    "console_fragment",
]


# ---------------------------------------------------------------------------
# Fake AbstractCore modules
# ---------------------------------------------------------------------------


class _Calls(list):
    def last(self) -> Any:
        return self[-1]


class FakeEngineInstallRefused(RuntimeError):
    def __init__(self, message: str, *, reason: str = "refused", plan: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.reason = reason
        self.plan = plan


class FakeJobBusy(RuntimeError):
    def __init__(self, message: str, job: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.job = job


class FakeRegistry:
    def __init__(self, persist_dir: Any = None) -> None:
        self.persist_dir = persist_dir
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.cancelled: List[str] = []
        self.cancel_by: List[Any] = []

    def list(self) -> List[Dict[str, Any]]:
        return list(self.jobs.values())

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self.jobs.get(job_id)

    def cancel(self, job_id: str, *, by: str = "api", user: Optional[str] = None) -> Optional[Dict[str, Any]]:
        self.cancel_by.append((job_id, by, user))
        job = self.jobs.get(job_id)
        if job is None:
            return None
        self.cancelled.append(job_id)
        return dict(job, status="cancelled")


def _job(job_id: str, kind: str = "download", status: str = "completed", started_at: str = "2026-09-23T10:00:00Z") -> Dict[str, Any]:
    return {"schema": "host_job_v1", "job_id": job_id, "kind": kind, "status": status, "started_at": started_at}


@pytest.fixture()
def fake_core(monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    """Install fake `abstractcore.*` modules; returns the call recorders."""

    calls = types.SimpleNamespace(
        host=_Calls(), engines=_Calls(), catalog=_Calls(), installed=_Calls(), blockers=_Calls(),
        delete_job=_Calls(), download_job=_Calls(), install=_Calls(), fragment=_Calls(),
    )
    registry = FakeRegistry()
    persisted: Dict[str, Dict[str, Any]] = {}
    state = types.SimpleNamespace(
        calls=calls,
        registry=registry,
        persisted=persisted,
        blockers={"found": True, "row": {}, "delete_blockers": [], "error": None},
        install_error=None,
        download_error=None,
    )

    host_profile_mod = types.ModuleType("abstractcore.utils.host_profile")

    def host_profile(*, refresh: bool = False) -> Dict[str, Any]:
        calls.host.append({"refresh": refresh})
        return {"schema": "host_profile_v1", "os": "darwin"}

    host_profile_mod.host_profile = host_profile

    engines_mod = types.ModuleType("abstractcore.config.engines")
    engines_mod.EngineInstallRefused = FakeEngineInstallRefused

    def engine_inventory(probe: bool = False) -> Dict[str, Any]:
        calls.engines.append({"probe": probe})
        return {"schema": "engines_status_v1", "engines": [{"id": "ollama"}]}

    def engine_status(engine_id: str, *, probe: bool = False) -> Dict[str, Any]:
        if engine_id != "ollama":
            raise KeyError(f"unknown engine {engine_id!r}; known: ollama")
        return {"id": "ollama", "probe": probe}

    def engine_install_plan(engine_id: str) -> Dict[str, Any]:
        if engine_id != "ollama":
            raise KeyError(f"unknown engine {engine_id!r}; known: ollama")
        return {"available": True, "argv": ["brew", "install", "ollama"]}

    def engine_download_url(engine_id: str) -> str:
        if engine_id != "ollama":
            raise KeyError(f"unknown engine {engine_id!r}")
        return "https://ollama.com/download"

    def engine_install(engine_id: str, *, dry_run: bool, force: bool, allow: Any, run_inline: bool) -> Dict[str, Any]:
        calls.install.append({"engine_id": engine_id, "dry_run": dry_run, "force": force, "allow": allow, "run_inline": run_inline})
        if state.install_error is not None:
            raise state.install_error
        return _job("eng_1", kind="engine_install", status="completed" if run_inline else "queued")

    engines_mod.engine_inventory = engine_inventory
    engines_mod.engine_status = engine_status
    engines_mod.engine_install_plan = engine_install_plan
    engines_mod.engine_download_url = engine_download_url
    engines_mod.engine_install = engine_install

    catalog_mod = types.ModuleType("abstractcore.config.model_catalog")

    def catalog(q: Any = None, *, engine: Any = None, fits: bool = False, hub: bool = False, tags: Any = None) -> Dict[str, Any]:
        calls.catalog.append({"q": q, "engine": engine, "fits": fits, "hub": hub, "tags": tags})
        return {"schema": "model_catalog_v1", "rows": []}

    catalog_mod.catalog = catalog

    materializer_mod = types.ModuleType("abstractcore.config.model_materializer")

    def list_installed(provider: Any = None) -> Dict[str, Any]:
        calls.installed.append({"provider": provider})
        return {"schema": "models_installed_v1", "rows": []}

    def delete_blockers(provider: str, artifact: str) -> Dict[str, Any]:
        calls.blockers.append((provider, artifact))
        return dict(state.blockers)

    materializer_mod.list_installed = list_installed
    materializer_mod.delete_blockers = delete_blockers
    materializer_mod.delete_artifact = lambda *a, **k: {}

    jobs_mod = types.ModuleType("abstractcore.config.host_jobs")
    jobs_mod.JobBusy = FakeJobBusy
    jobs_mod.default_registry = lambda: registry

    def start_delete_job(provider: str, artifact: str, *, dry_run: bool, force: bool, run_inline: bool) -> Dict[str, Any]:
        calls.delete_job.append({"provider": provider, "artifact": artifact, "dry_run": dry_run, "force": force, "run_inline": run_inline})
        return _job("rm_1", kind="delete", status="completed" if run_inline else "queued")

    def start_download_job(provider: str, artifact: str, *, dry_run: bool, expected_bytes: Any, run_inline: bool) -> Dict[str, Any]:
        calls.download_job.append({"provider": provider, "artifact": artifact, "dry_run": dry_run, "expected_bytes": expected_bytes, "run_inline": run_inline})
        if state.download_error is not None:
            raise state.download_error
        return _job("dl_1", status="completed" if run_inline else "queued")

    jobs_mod.start_delete_job = start_delete_job
    jobs_mod.start_download_job = start_download_job
    jobs_mod.read_persisted_jobs = lambda directory: list(persisted.values())
    jobs_mod.read_persisted_job = lambda job_id, directory: persisted.get(job_id)
    jobs_mod.request_cancel = lambda job_id, directory, by="other_process", user=None: (dict(persisted[job_id], cancel_requested=True, cancelled_by=by, cancelled_by_user=user) if job_id in persisted else None)

    web_mod = types.ModuleType("abstractcore.console.web")

    def fragment(kind: str) -> Dict[str, str]:
        calls.fragment.append(kind)
        if kind not in {"models", "engines"}:
            raise ValueError(f"unknown fragment kind {kind!r}")
        return {"html": f"<div data-acc-kind={kind}></div>", "js": "/*js*/", "css": "/*css*/"}

    web_mod.fragment = fragment

    for name, mod in {
        "abstractcore.utils.host_profile": host_profile_mod,
        "abstractcore.config.engines": engines_mod,
        "abstractcore.config.model_catalog": catalog_mod,
        "abstractcore.config.model_materializer": materializer_mod,
        "abstractcore.config.host_jobs": jobs_mod,
        "abstractcore.console.web": web_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return state


# ---------------------------------------------------------------------------
# Surface and passthroughs
# ---------------------------------------------------------------------------


def test_new_surface_is_exported_and_callable() -> None:
    for name in NEW_SURFACE:
        assert name in facade.__all__, name
        assert hasattr(facade, name), name
    assert facade.MODELS_ENGINES_MIN_ABSTRACTCORE == "2.14.0"
    assert issubclass(facade.AbstractCoreTooOld, NotImplementedError)
    assert issubclass(facade.HostActionRefused, RuntimeError)


def test_reads_pass_arguments_through_unchanged(fake_core) -> None:
    assert facade.host_profile(refresh=True)["schema"] == "host_profile_v1"
    assert fake_core.calls.host.last() == {"refresh": True}

    assert facade.engine_inventory(probe=True)["engines"] == [{"id": "ollama"}]
    assert fake_core.calls.engines.last() == {"probe": True}

    facade.model_catalog("qwen", engine="ollama", fits_only=True, hub=True, tags=["chat"])
    assert fake_core.calls.catalog.last() == {"q": "qwen", "engine": "ollama", "fits": True, "hub": True, "tags": ["chat"]}
    facade.model_catalog()
    # Empty strings are "no filter", never a filter on the empty string.
    facade.model_catalog("", engine="")
    assert fake_core.calls.catalog.last() == {"q": None, "engine": None, "fits": False, "hub": False, "tags": None}

    facade.list_installed_models("lmstudio")
    assert fake_core.calls.installed.last() == {"provider": "lmstudio"}
    facade.list_installed_models("")
    assert fake_core.calls.installed.last() == {"provider": None}

    assert facade.engine_install_plan("ollama")["argv"] == ["brew", "install", "ollama"]
    assert facade.engine_download_url("ollama") == "https://ollama.com/download"
    assert facade.engine_status("ollama", probe=True) == {"id": "ollama", "probe": True}
    assert facade.console_fragment("engines")["html"].startswith("<div")


def test_unknown_engine_and_fragment_are_404_refusals(fake_core) -> None:
    for call in (
        lambda: facade.engine_status("nope"),
        lambda: facade.engine_install_plan("nope"),
        lambda: facade.engine_download_url("nope"),
        lambda: facade.console_fragment("runtimes"),
    ):
        with pytest.raises(facade.HostActionRefused) as info:
            call()
        assert info.value.status_code == 404


# ---------------------------------------------------------------------------
# Jobs: install, download, delete
# ---------------------------------------------------------------------------


def test_engine_install_runs_a_dry_run_inline_and_a_real_one_in_the_background(fake_core) -> None:
    job = facade.engine_install("ollama", dry_run=True, allow=False)
    assert job["status"] == "completed"
    assert fake_core.calls.install.last() == {"engine_id": "ollama", "dry_run": True, "force": False, "allow": False, "run_inline": True}

    job = facade.engine_install("ollama", allow=True)
    assert job["status"] == "queued"
    assert fake_core.calls.install.last()["run_inline"] is False


@pytest.mark.parametrize(
    "reason, code",
    [("not_allowed", 403), ("unknown_engine", 404), ("unsupported", 409), ("no_plan", 409)],
)
def test_engine_install_refusals_carry_status_and_plan(fake_core, reason: str, code: int) -> None:
    fake_core.install_error = FakeEngineInstallRefused("nope", reason=reason, plan={"argv": ["x"]})
    with pytest.raises(facade.HostActionRefused) as info:
        facade.engine_install("ollama")
    refused = info.value
    assert refused.status_code == code
    body = refused.payload()
    assert body == {"ok": False, "status": "refused", "message": "nope", "reason": reason, "install": {"argv": ["x"]}}


def test_a_second_engine_install_is_busy_with_the_running_job(fake_core) -> None:
    fake_core.install_error = FakeJobBusy("an engine install is running", job={"job_id": "eng_9"})
    with pytest.raises(facade.HostActionRefused) as info:
        facade.engine_install("ollama")
    assert info.value.status_code == 409
    assert info.value.payload()["status"] == "busy"
    assert info.value.payload()["job"] == {"job_id": "eng_9"}


def test_download_job_passthrough_and_invalid_input(fake_core) -> None:
    job = facade.start_model_download_job("ollama", "qwen3:8b", expected_bytes=123)
    assert job["job_id"] == "dl_1" and job["status"] == "queued"
    assert fake_core.calls.download_job.last() == {
        "provider": "ollama", "artifact": "qwen3:8b", "dry_run": False, "expected_bytes": 123, "run_inline": False,
    }
    facade.start_model_download_job("ollama", "qwen3:8b", dry_run=True)
    assert fake_core.calls.download_job.last()["run_inline"] is True

    fake_core.download_error = ValueError("a provider and an artifact are required")
    with pytest.raises(facade.HostActionRefused) as info:
        facade.start_model_download_job("", "")
    assert info.value.status_code == 400
    assert info.value.status == "invalid"


def test_delete_checks_blockers_before_starting_a_job(fake_core) -> None:
    # Not installed: 404, and no job is started.
    fake_core.blockers = {"found": False, "row": None, "delete_blockers": [], "error": None}
    with pytest.raises(facade.HostActionRefused) as info:
        facade.delete_model_artifact("ollama", "gone:1b")
    assert info.value.status_code == 404
    assert info.value.status == "not_found"
    assert fake_core.calls.delete_job == []

    # A remote engine: not found, with a blocker -> 409.
    fake_core.blockers = {"found": False, "row": None, "delete_blockers": ["remote_engine"], "error": "serves remotely"}
    with pytest.raises(facade.HostActionRefused) as info:
        facade.delete_model_artifact("openai", "gpt")
    assert info.value.status_code == 409
    assert info.value.payload()["delete_blockers"] == ["remote_engine"]
    assert "serves remotely" in info.value.message

    # Loaded: refused without force, started with it.
    fake_core.blockers = {"found": True, "row": {}, "delete_blockers": ["loaded"], "error": None}
    with pytest.raises(facade.HostActionRefused) as info:
        facade.delete_model_artifact("ollama", "qwen3:8b")
    assert info.value.status_code == 409
    assert "force=true" in info.value.message
    assert fake_core.calls.delete_job == []
    job = facade.delete_model_artifact("ollama", "qwen3:8b", force=True)
    assert job["kind"] == "delete"
    assert fake_core.calls.delete_job.last() == {"provider": "ollama", "artifact": "qwen3:8b", "dry_run": False, "force": True, "run_inline": False}

    # A hard blocker is refused even with force.
    fake_core.blockers = {"found": True, "row": {}, "delete_blockers": ["unknown_location"], "error": None}
    with pytest.raises(facade.HostActionRefused) as info:
        facade.delete_model_artifact("lmstudio", "x", force=True)
    assert info.value.status_code == 409
    assert "force" not in info.value.message

    # No blockers, dry run: finished inline.
    fake_core.blockers = {"found": True, "row": {}, "delete_blockers": [], "error": None}
    job = facade.delete_model_artifact("ollama", "qwen3:8b", dry_run=True)
    assert job["status"] == "completed"
    assert fake_core.calls.delete_job.last()["run_inline"] is True


def test_jobs_list_merges_persisted_and_live_jobs_newest_first(fake_core, tmp_path) -> None:
    fake_core.registry.persist_dir = tmp_path
    fake_core.persisted["dl_old"] = _job("dl_old", started_at="2026-09-23T09:00:00Z")
    fake_core.persisted["rm_live"] = _job("rm_live", kind="delete", status="running", started_at="2026-09-23T11:00:00Z")
    # The live copy of the same id wins over its persisted snapshot.
    fake_core.registry.jobs["rm_live"] = _job("rm_live", kind="delete", status="completed", started_at="2026-09-23T11:00:00Z")
    fake_core.registry.jobs["eng_new"] = _job("eng_new", kind="engine_install", status="running", started_at="2026-09-23T12:00:00Z")

    payload = facade.host_jobs_list()
    assert payload["schema"] == "host_jobs_v1"
    assert payload["generated_at"].endswith("Z")
    assert [j["job_id"] for j in payload["jobs"]] == ["eng_new", "rm_live", "dl_old"]
    assert payload["jobs"][1]["status"] == "completed"
    assert [j["job_id"] for j in facade.host_jobs_list(kind="delete")["jobs"]] == ["rm_live"]
    assert [j["job_id"] for j in facade.host_jobs_list(status="running")["jobs"]] == ["eng_new"]

    assert facade.host_job("eng_new")["kind"] == "engine_install"
    assert facade.host_job("dl_old")["job_id"] == "dl_old"  # persisted fallback
    assert facade.host_job("unknown") is None

    assert facade.host_job_cancel("eng_new")["status"] == "cancelled"
    assert fake_core.registry.cancelled == ["eng_new"]
    assert facade.host_job_cancel("dl_old")["cancel_requested"] is True  # another process's job
    assert facade.host_job_cancel("unknown") is None
    # Who asked travels to AbstractCore (mission KK): the default is `api`; a
    # console click says `console` and names the signed-in account.
    assert fake_core.registry.cancel_by[0] == ("eng_new", "api", None)
    moved = facade.host_job_cancel("dl_old", by="console", user="admin")
    assert (moved["cancelled_by"], moved["cancelled_by_user"]) == ("console", "admin")


def test_jobs_without_persistence_read_only_the_live_registry(fake_core) -> None:
    fake_core.persisted["dl_old"] = _job("dl_old")
    assert facade.host_jobs_list()["jobs"] == []
    assert facade.host_job("dl_old") is None
    assert facade.host_job_cancel("dl_old") is None


# ---------------------------------------------------------------------------
# Older AbstractCore
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module, call",
    [
        ("abstractcore.utils.host_profile", lambda: facade.host_profile()),
        ("abstractcore.config.engines", lambda: facade.engine_inventory()),
        ("abstractcore.config.engines", lambda: facade.engine_install("ollama", dry_run=True)),
        ("abstractcore.config.model_catalog", lambda: facade.model_catalog()),
        ("abstractcore.config.host_jobs", lambda: facade.host_jobs_list()),
        ("abstractcore.config.host_jobs", lambda: facade.start_model_download_job("ollama", "x")),
        ("abstractcore.console.web", lambda: facade.console_fragment("models")),
    ],
)
def test_an_older_abstractcore_raises_too_old_with_the_upgrade_command(monkeypatch, module: str, call) -> None:
    # `None` in sys.modules makes the import fail exactly like a missing module.
    monkeypatch.setitem(sys.modules, module, None)
    monkeypatch.setattr(facade, "_installed_abstractcore_version", lambda: "2.13.42")
    with pytest.raises(facade.AbstractCoreTooOld) as info:
        call()
    err = info.value
    assert isinstance(err, NotImplementedError)
    assert err.installed == "2.13.42"
    assert err.required == "2.14.0"
    assert err.missing == module
    assert "abstractcore>=2.14.0" in str(err) and "2.13.42" in str(err)

    support = facade.models_engines_support()
    assert support["available"] is False
    assert module in support["missing"]


def test_a_materializer_without_the_list_and_delete_verbs_is_too_old(fake_core, monkeypatch) -> None:
    old = types.ModuleType("abstractcore.config.model_materializer")
    old.probe = lambda *a, **k: None  # 2.13.x: probe and download only
    monkeypatch.setitem(sys.modules, "abstractcore.config.model_materializer", old)
    with pytest.raises(facade.AbstractCoreTooOld) as info:
        facade.list_installed_models()
    assert info.value.missing == "abstractcore.config.model_materializer.list_installed"
    with pytest.raises(facade.AbstractCoreTooOld):
        facade.delete_model_artifact("ollama", "x")
    assert "abstractcore.config.model_materializer.list_installed" in facade.models_engines_support()["missing"]


def test_missing_abstractcore_is_a_runtime_error_not_too_old(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "abstractcore", None)
    with pytest.raises(RuntimeError) as info:
        facade.host_profile()
    assert not isinstance(info.value, NotImplementedError)
    assert "not installed" in str(info.value)


# ---------------------------------------------------------------------------
# The real AbstractCore in this environment (dry runs, in-memory jobs)
# ---------------------------------------------------------------------------


@pytest.fixture()
def real_core_jobs():
    support = facade.models_engines_support()
    if not support["available"]:
        pytest.skip(f"installed AbstractCore lacks models & engines: {support}")
    from abstractcore.config import host_jobs

    registry = host_jobs.HostJobRegistry(persist_dir=None)
    host_jobs.set_default_registry(registry)
    try:
        yield registry
    finally:
        host_jobs.set_default_registry(None)


def test_real_abstractcore_passthroughs(real_core_jobs) -> None:
    assert facade.host_profile()["schema"] == "host_profile_v1"
    plan = facade.engine_install_plan("ollama")
    assert "available" in plan and "argv" in plan

    job = facade.engine_install("ollama", dry_run=True, allow=False)
    assert job["schema"] == "host_job_v1"
    assert job["kind"] == "engine_install"
    assert job["dry_run"] is True
    assert job["status"] in {"completed", "failed"}
    assert facade.host_job(job["job_id"])["job_id"] == job["job_id"]
    assert [j["job_id"] for j in facade.host_jobs_list(kind="engine_install")["jobs"]] == [job["job_id"]]

    with pytest.raises(facade.HostActionRefused) as info:
        facade.engine_install("ollama", dry_run=False, allow=False)
    assert info.value.status_code == 403
    assert info.value.reason == "not_allowed"

    with pytest.raises(facade.HostActionRefused) as info:
        facade.engine_install("not-an-engine", dry_run=True)
    assert info.value.status_code == 404

    frag = facade.console_fragment("models")
    assert set(frag) == {"html", "js", "css"}
    assert "AbstractCoreConsole" in frag["js"]
