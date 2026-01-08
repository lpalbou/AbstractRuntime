"""
abstractruntime.workflow_bundle

WorkflowBundle (.flow) support (portable distribution unit).

Design intent:
- Stdlib-only (no extra deps) to keep AbstractRuntime minimal.
- Bundles are transport/content; hosts decide trust and execution policy.
- Bundle IDs are used by hosts to namespace workflow ids and avoid collisions.
"""

from .models import (
    WORKFLOW_BUNDLE_FORMAT_VERSION_V1,
    WorkflowBundleEntrypoint,
    WorkflowBundleError,
    WorkflowBundleManifest,
    workflow_bundle_manifest_from_dict,
    workflow_bundle_manifest_to_dict,
)
from .reader import WorkflowBundle, open_workflow_bundle

__all__ = [
    "WORKFLOW_BUNDLE_FORMAT_VERSION_V1",
    "WorkflowBundleError",
    "WorkflowBundleEntrypoint",
    "WorkflowBundleManifest",
    "workflow_bundle_manifest_from_dict",
    "workflow_bundle_manifest_to_dict",
    "WorkflowBundle",
    "open_workflow_bundle",
]


