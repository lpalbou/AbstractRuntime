"""The shipped controller bundle (contract D, amendment 5).

`abstractframework.automation-controller@1.0.0` is a directory bundle inside
this package (`automations/bundles/automation-controller/`). Every automation
root run persists the workflow id `CONTROLLER_WORKFLOW_ID`
(`abstractframework.automation-controller@1.0.0:controller`), so a restarted
host resolves exactly this pinned version; a host that cannot is expected to
fail the controller loudly, never to substitute another flow.
"""

from __future__ import annotations

import threading
from importlib import resources
from pathlib import Path
from typing import Any, Optional

from ..core.spec import WorkflowSpec
from ..workflow_bundle.reader import open_workflow_bundle
from .models import CONTROLLER_BUNDLE_ID, CONTROLLER_BUNDLE_VERSION, CONTROLLER_FLOW_ID, CONTROLLER_WORKFLOW_ID


class ControllerBundleError(RuntimeError):
    """The packaged controller bundle is missing or is not the pinned version."""


def controller_bundle_path() -> Path:
    """Filesystem path of the packaged controller bundle directory."""
    path = Path(str(resources.files("abstractruntime.automations") / "bundles" / "automation-controller"))
    if not (path / "manifest.json").is_file():
        raise ControllerBundleError(f"the automation controller bundle is missing from this install ({path})")
    return path


_SPEC_LOCK = threading.Lock()
_SPEC: Optional[WorkflowSpec] = None


def controller_workflow_spec() -> WorkflowSpec:
    """The compiled controller flow, with the pinned versioned workflow id."""
    global _SPEC
    with _SPEC_LOCK:
        if _SPEC is not None:
            return _SPEC
        bundle = open_workflow_bundle(controller_bundle_path())
        manifest = bundle.manifest
        if manifest.bundle_id != CONTROLLER_BUNDLE_ID or manifest.bundle_version != CONTROLLER_BUNDLE_VERSION:
            raise ControllerBundleError(
                f"packaged controller bundle is {manifest.bundle_id}@{manifest.bundle_version}, "
                f"expected {CONTROLLER_BUNDLE_ID}@{CONTROLLER_BUNDLE_VERSION}"
            )
        relpath = manifest.flows.get(CONTROLLER_FLOW_ID)
        if not relpath:
            raise ControllerBundleError(f"controller bundle has no flow {CONTROLLER_FLOW_ID!r}")
        raw = bundle.read_json(relpath)
        raw["id"] = CONTROLLER_WORKFLOW_ID

        from ..visualflow_compiler import compile_visualflow

        spec = compile_visualflow(raw)
        if spec.workflow_id != CONTROLLER_WORKFLOW_ID:
            raise ControllerBundleError(f"compiled controller id {spec.workflow_id!r} != {CONTROLLER_WORKFLOW_ID!r}")
        _SPEC = spec
        return spec


def register_controller_bundle(registry: Any) -> WorkflowSpec:
    """Register the controller spec in a workflow registry (anything with `register(spec)`).

    Hosts that load bundles from directories can instead pass
    `controller_bundle_path()` to their bundle loader; both yield the workflow
    id `CONTROLLER_WORKFLOW_ID`.
    """
    spec = controller_workflow_spec()
    registry.register(spec)
    return spec


__all__ = [
    "ControllerBundleError",
    "controller_bundle_path",
    "controller_workflow_spec",
    "register_controller_bundle",
]
