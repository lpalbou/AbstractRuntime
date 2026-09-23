"""Data-registry facade (commons c1771, 2026-07-14): the gateway's Data &
Caches lane imports THIS runtime module instead of abstractcore directly
(boundary 0059, config_facade precedent).

Pins: exact three-callable surface, pass-through fidelity against core's
REAL registry (isolated via ABSTRACTFRAMEWORK_DATA_REGISTRY env), verbatim
refusal propagation (owner + rule), and dry-run accounting.
"""

from __future__ import annotations

import pytest

from abstractruntime.integrations.abstractcore import data_registry_facade as facade


@pytest.fixture()
def registry_env(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ABSTRACTFRAMEWORK_DATA_REGISTRY", str(tmp_path / "registry.json"))
    monkeypatch.delenv("ABSTRACTFRAMEWORK_DATA_REGISTRY_DISABLE", raising=False)
    # ensure_data_home_registered dedupes per process; isolate per test.
    from abstractcore.utils import data_registry as core_registry

    monkeypatch.setattr(core_registry, "_ensured_names", set())
    monkeypatch.setattr(core_registry, "_ensure_warned", set())
    return tmp_path


def test_surface_is_exactly_four_callables() -> None:
    assert facade.__all__ == [
        "ensure_data_home_registered",
        "list_data_homes",
        "purge_data_home",
        "unregister_data_home",
    ]
    for name in facade.__all__:
        assert callable(getattr(facade, name))


def test_register_list_and_sizes_round_trip(registry_env) -> None:
    home = registry_env / "cache_home"
    home.mkdir()
    (home / "blob.bin").write_bytes(b"x" * 128)

    row = facade.ensure_data_home_registered(
        "test-cache",
        path=str(home),
        kind="model-cache",
        owner="runtime-test",
        safe_to_purge=True,
        description="facade round-trip fixture",
    )
    assert row is not None

    rows = facade.list_data_homes(include_sizes=True)
    match = [r for r in rows if r["name"] == "test-cache"]
    assert len(match) == 1
    assert match[0]["owner"] == "runtime-test"
    assert match[0]["safe_to_purge"] is True
    assert match[0]["exists"] is True
    assert match[0]["size_bytes"] >= 128


def test_purge_dry_run_then_real_and_protected_refusal(registry_env) -> None:
    purgeable = registry_env / "purgeable"
    purgeable.mkdir()
    (purgeable / "junk.txt").write_text("bytes")
    protected = registry_env / "protected_home"
    protected.mkdir()
    (protected / "life.txt").write_text("precious")

    facade.ensure_data_home_registered(
        "purgeable", path=str(purgeable), kind="model-cache", owner="runtime-test", safe_to_purge=True
    )
    facade.ensure_data_home_registered(
        "protected", path=str(protected), kind="entity-home", owner="gateway", safe_to_purge=False
    )

    dry = facade.purge_data_home("purgeable", dry_run=True)
    assert dry.get("dry_run") is True
    assert (purgeable / "junk.txt").exists(), "dry run must not delete"

    real = facade.purge_data_home("purgeable")
    assert not (purgeable / "junk.txt").exists()
    assert purgeable.is_dir(), "the home DIRECTORY itself survives a purge"
    assert real.get("dry_run") in (False, None)

    # Refusals propagate VERBATIM (owner + rule), typed as core's error.
    from abstractcore.utils.data_registry import DataRegistryError

    with pytest.raises(DataRegistryError, match="gateway"):
        facade.purge_data_home("protected")
    with pytest.raises(DataRegistryError, match="not in the data registry"):
        facade.purge_data_home("never-registered")
    assert (protected / "life.txt").exists()
