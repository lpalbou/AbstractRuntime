"""The controller bundle ships inside the wheel and resolves to the pinned id (contract D, C5)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from abstractruntime.automations import CONTROLLER_WORKFLOW_ID, controller_bundle_path, controller_workflow_spec
from abstractruntime.automations.adapters import ADAPTER_OUTCOMES
from abstractruntime.workflow_bundle.reader import open_workflow_bundle

REPO = Path(__file__).resolve().parents[1]
BUNDLE_FILES = (
    "abstractruntime/automations/bundles/automation-controller/manifest.json",
    "abstractruntime/automations/bundles/automation-controller/flows/controller.json",
)


def test_bundle_is_the_pinned_version_and_compiles_to_the_pinned_id():
    bundle = open_workflow_bundle(controller_bundle_path())
    assert bundle.manifest.bundle_id == "abstractframework.automation-controller"
    assert bundle.manifest.bundle_version == "1.0.0"
    assert bundle.manifest.metadata.get("internal") is True
    # Hosts namespace bundle flows as "<bundle_id>@<version>:<flow_id>".
    assert f"{bundle.manifest.bundle_id}@{bundle.manifest.bundle_version}:controller" == CONTROLLER_WORKFLOW_ID
    assert controller_workflow_spec().workflow_id == CONTROLLER_WORKFLOW_ID


def test_every_automation_node_has_an_adapter_and_every_outcome_an_edge():
    flow = json.loads((controller_bundle_path() / "flows" / "controller.json").read_text())
    automation_nodes = {n["id"] for n in flow["nodes"] if n["type"] == "automation"}
    assert {f"automation.{n}" for n in automation_nodes} == set(ADAPTER_OUTCOMES)
    for node_id in automation_nodes:
        handles = {e["sourceHandle"] for e in flow["edges"] if e["source"] == node_id}
        outcomes = ADAPTER_OUTCOMES[f"automation.{node_id}"]
        assert handles == ({f"case:{o}" for o in outcomes} if outcomes else {"exec-out"}), node_id


def test_an_unknown_automation_adapter_fails_the_compile():
    from abstractruntime.visualflow_compiler import compile_visualflow

    flow = json.loads((controller_bundle_path() / "flows" / "controller.json").read_text())
    flow["nodes"].append({"id": "teleport", "type": "automation", "data": {"nodeType": "automation"}})
    flow["edges"].append({"source": "next", "target": "teleport", "sourceHandle": "exec-out", "targetHandle": "exec-in"})
    flow["edges"] = [e for e in flow["edges"] if not (e["source"] == "next" and e["target"] == "read_definition")]
    with pytest.raises(ValueError, match="unknown automation adapter 'automation.teleport'"):
        compile_visualflow(flow)


@pytest.mark.integration
def test_wheel_contains_the_bundle(tmp_path):
    hatchling = pytest.importorskip("hatchling.builders.wheel", reason="hatchling is the build backend")
    wheel = hatchling.WheelBuilder(str(REPO)).build(directory=str(tmp_path), versions=["standard"])
    path = next(iter(wheel))
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        for rel in BUNDLE_FILES:
            assert rel in names, rel
        entry_points = next(n for n in names if n.endswith(".dist-info/entry_points.txt"))
        text = zf.read(entry_points).decode()
    assert "[abstractruntime.trigger_sources]" in text
    assert "schedule = abstractruntime.triggers.schedule:ScheduleTriggerAdapter" in text
