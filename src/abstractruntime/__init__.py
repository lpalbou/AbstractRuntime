"""
AbstractRuntime

Durable graph runner (interrupt → checkpoint → resume).

This package provides a minimal execution substrate:
- workflow graphs (state machines)
- durable RunState with WAITING / RESUME semantics
- append-only execution journal (ledger)

Higher-level orchestration and UI graph authoring is expected to live in AbstractFlow.
"""

from .core.models import (
    Effect,
    EffectType,
    RunState,
    RunStatus,
    StepPlan,
    WaitReason,
    WaitState,
)
from .core.health import RuntimeHealth
from .core.runtime import Runtime
from .core.spec import WorkflowSpec
from .core.policy import (
    EffectPolicy,
    DefaultEffectPolicy,
    RetryPolicy,
    NoRetryPolicy,
    compute_idempotency_key,
)
from .storage.base import DeletableLedgerStore, DeletableRunStore, QueryableRunStore
from .storage.in_memory import InMemoryLedgerStore, InMemoryRunStore
from .storage.json_files import JsonFileRunStore, JsonlLedgerStore
from .storage.sqlite import (
    SqliteCommandCursorStore,
    SqliteCommandStore,
    SqliteDatabase,
    SqliteLedgerStore,
    SqliteRunStore,
)
from .storage.commands import (
    CommandAppendResult,
    CommandCursorStore,
    CommandRecord,
    CommandStore,
    DeletableCommandStore,
    InMemoryCommandCursorStore,
    InMemoryCommandStore,
    JsonFileCommandCursorStore,
    JsonlCommandStore,
)
from .storage.ledger_chain import HashChainedLedgerStore, verify_ledger_chain
from .storage.steer_sidecar import InMemorySteerSidecar, SqliteSteerSidecar, SteerSidecarStore
from .storage.observable import ObservableLedgerStore, ObservableLedgerStoreProtocol
from .storage.snapshots import Snapshot, SnapshotStore, InMemorySnapshotStore, JsonSnapshotStore
from .storage.offloading import OffloadingLedgerStore, OffloadingRunStore, offload_large_values
from .storage.artifacts import (
    Artifact,
    ArtifactAccessStats,
    ArtifactDescriptor,
    ArtifactMetadata,
    ArtifactStore,
    InMemoryArtifactStore,
    FileArtifactStore,
    artifact_ref,
    is_artifact_ref,
    get_artifact_id,
    resolve_artifact,
    compute_artifact_id,
)
from .identity.fingerprint import ActorFingerprint
from .identity.tools import (
    TIER1_TOOL_NAMES,
    TOOL_DESCRIPTORS,
    WORKSPACE_TOOL_NAMES,
    ToolDescriptor,
    walled_tool_rows,
)
from .identity.tool_policy import (
    LEGACY_PHASE_ALIASES,
    PHASES,
    PHASE_PERSONAL,
    PHASE_SLEEP,
    PHASE_VISIT,
    PHASE_WORK,
    ToolGrant,
    canonical_phase,
    read_policy_file,
    resolve_tool_grant,
    write_policy_file,
)
from .scheduler import (
    WorkflowRegistry,
    Scheduler,
    SchedulerStats,
    ScheduledRuntime,
    create_scheduled_runtime,
)
from .memory import ActiveContextPolicy, TimeRange
from .workflow_bundle import (
    WORKFLOW_BUNDLE_FORMAT_VERSION_V1,
    InstalledWorkflowBundle,
    WorkflowBundle,
    WorkflowBundleEntrypoint,
    WorkflowBundleError,
    WorkflowBundleManifest,
    WorkflowBundleRegistry,
    WorkflowBundleRegistryError,
    WorkflowEntrypointRef,
    default_workflow_bundles_dir,
    open_workflow_bundle,
    sanitize_bundle_id,
    sanitize_bundle_version,
    workflow_bundle_manifest_from_dict,
    workflow_bundle_manifest_to_dict,
)
from .history_bundle import (
    RUN_HISTORY_BUNDLE_VERSION_V1,
    export_run_history_bundle,
    persist_workflow_snapshot,
)
from .session_history import (
    SESSION_TURN_KIND,
    session_chat_messages,
)

__all__ = [
    # Core models
    "Effect",
    "EffectType",
    "RunState",
    "RunStatus",
    "StepPlan",
    "WaitReason",
    "WaitState",
    # Spec + runtime
    "WorkflowSpec",
    "Runtime",
    "RuntimeHealth",
    # Scheduler
    "WorkflowRegistry",
    "Scheduler",
    "SchedulerStats",
    "ScheduledRuntime",
    "create_scheduled_runtime",
    # Storage backends
    "QueryableRunStore",
    "DeletableRunStore",
    "DeletableLedgerStore",
    "InMemoryRunStore",
    "InMemoryLedgerStore",
    "JsonFileRunStore",
    "JsonlLedgerStore",
    "SqliteDatabase",
    "SqliteRunStore",
    "SqliteLedgerStore",
    "CommandRecord",
    "CommandAppendResult",
    "CommandStore",
    "DeletableCommandStore",
    "CommandCursorStore",
    "InMemoryCommandStore",
    "JsonlCommandStore",
    "InMemoryCommandCursorStore",
    "JsonFileCommandCursorStore",
    "SqliteCommandStore",
    "SqliteCommandCursorStore",
    "HashChainedLedgerStore",
    "verify_ledger_chain",
    "SteerSidecarStore",
    "InMemorySteerSidecar",
    "SqliteSteerSidecar",
    "ObservableLedgerStore",
    "ObservableLedgerStoreProtocol",
    "OffloadingRunStore",
    "OffloadingLedgerStore",
    "offload_large_values",
    "Snapshot",
    "SnapshotStore",
    "InMemorySnapshotStore",
    "JsonSnapshotStore",
    # Artifacts
    "Artifact",
    "ArtifactAccessStats",
    "ArtifactDescriptor",
    "ArtifactMetadata",
    "ArtifactStore",
    "InMemoryArtifactStore",
    "FileArtifactStore",
    "artifact_ref",
    "is_artifact_ref",
    "get_artifact_id",
    "resolve_artifact",
    "compute_artifact_id",
    # Identity
    "ActorFingerprint",
    # Entity life phases + per-phase tool grants (config-object consensus,
    # F7/N7: the phase SET is runtime's — the door imports from the root,
    # never a second copy; legacy aliases die before release)
    "LEGACY_PHASE_ALIASES",
    "PHASES",
    "PHASE_PERSONAL",
    "PHASE_SLEEP",
    "PHASE_VISIT",
    "PHASE_WORK",
    "ToolGrant",
    "canonical_phase",
    "read_policy_file",
    "resolve_tool_grant",
    "write_policy_file",
    # Walled tool inventory (descriptor contract v6: the emission is the
    # sole field source for runtime rows; gateway attaches executes_via)
    "TIER1_TOOL_NAMES",
    "TOOL_DESCRIPTORS",
    "WORKSPACE_TOOL_NAMES",
    "ToolDescriptor",
    "walled_tool_rows",
    # Effect policies
    "EffectPolicy",
    "DefaultEffectPolicy",
    "RetryPolicy",
    "NoRetryPolicy",
    "compute_idempotency_key",
    # Memory
    "ActiveContextPolicy",
    "TimeRange",
    # WorkflowBundles (portable distribution unit)
    "WORKFLOW_BUNDLE_FORMAT_VERSION_V1",
    "WorkflowBundleError",
    "WorkflowBundleEntrypoint",
    "WorkflowBundleManifest",
    "WorkflowBundle",
    "InstalledWorkflowBundle",
    "WorkflowBundleRegistry",
    "WorkflowBundleRegistryError",
    "WorkflowEntrypointRef",
    "default_workflow_bundles_dir",
    "sanitize_bundle_id",
    "sanitize_bundle_version",
    "workflow_bundle_manifest_from_dict",
    "workflow_bundle_manifest_to_dict",
    "open_workflow_bundle",
    # Run history bundle (portable replay)
    "RUN_HISTORY_BUNDLE_VERSION_V1",
    "export_run_history_bundle",
    "persist_workflow_snapshot",
    # Durable session conversation replay (read side)
    "SESSION_TURN_KIND",
    "session_chat_messages",
]
