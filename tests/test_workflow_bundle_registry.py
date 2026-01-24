from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest


def _write_bundle(path: Path, *, bundle_id: str, bundle_version: str, flow_id: str, name: str) -> None:
    manifest = {
        "bundle_format_version": "1",
        "bundle_id": bundle_id,
        "bundle_version": bundle_version,
        "created_at": "2026-01-24T00:00:00Z",
        "entrypoints": [
            {
                "flow_id": flow_id,
                "name": name,
                "description": "",
                "interfaces": ["abstractcode.agent.v1"],
            }
        ],
        "default_entrypoint": flow_id,
        "flows": {flow_id: f"flows/{flow_id}.json"},
        "metadata": {"test": True},
    }
    flow = {"id": flow_id, "name": name, "interfaces": ["abstractcode.agent.v1"], "nodes": [], "edges": [], "entryNode": None}

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        zf.writestr(f"flows/{flow_id}.json", json.dumps(flow, ensure_ascii=False, indent=2))


@pytest.mark.basic
def test_workflow_bundle_registry_install_list_resolve_remove(tmp_path) -> None:
    from abstractruntime.workflow_bundle import WorkflowBundleRegistry

    reg_dir = tmp_path / "reg"
    src = tmp_path / "src.flow"
    _write_bundle(src, bundle_id="demo-bundle", bundle_version="0.0.1", flow_id="root", name="Demo Bundle")

    reg = WorkflowBundleRegistry(reg_dir)
    installed = reg.install(src)
    assert installed.bundle_id == "demo-bundle"
    assert installed.bundle_version == "0.0.1"
    assert installed.path.exists()

    eps = reg.list_entrypoints(interface="abstractcode.agent.v1", latest_only=True)
    assert [e.bundle_ref for e in eps] == ["demo-bundle@0.0.1"]

    ep2 = reg.resolve_entrypoint("demo-bundle", interface="abstractcode.agent.v1")
    assert ep2.bundle_ref == "demo-bundle@0.0.1"
    assert ep2.flow_id == "root"

    ep3 = reg.resolve_entrypoint("Demo Bundle", interface="abstractcode.agent.v1")
    assert ep3.bundle_ref == "demo-bundle@0.0.1"

    removed = reg.remove("demo-bundle@0.0.1")
    assert removed == 1
    assert reg.list_entrypoints(interface="abstractcode.agent.v1") == []


@pytest.mark.basic
def test_workflow_bundle_registry_install_bytes(tmp_path) -> None:
    from abstractruntime.workflow_bundle import WorkflowBundleRegistry, WorkflowBundleRegistryError

    reg_dir = tmp_path / "reg"
    src = tmp_path / "src.flow"
    _write_bundle(src, bundle_id="demo-bundle", bundle_version="0.0.1", flow_id="root", name="Demo Bundle")

    reg = WorkflowBundleRegistry(reg_dir)
    installed = reg.install_bytes(src.read_bytes(), filename_hint=src.name)
    assert installed.bundle_ref == "demo-bundle@0.0.1"
    assert installed.path.exists()

    with pytest.raises(WorkflowBundleRegistryError):
        reg.install_bytes(src.read_bytes(), filename_hint=src.name, overwrite=False)

    installed2 = reg.install_bytes(src.read_bytes(), filename_hint=src.name, overwrite=True)
    assert installed2.bundle_ref == "demo-bundle@0.0.1"
