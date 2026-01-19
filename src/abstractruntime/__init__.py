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
from .core.runtime import Runtime
from .core.spec import WorkflowSpec
from .core.policy import (
    EffectPolicy,
    DefaultEffectPolicy,
    RetryPolicy,
    NoRetryPolicy,
    compute_idempotency_key,
)
from .storage.base import QueryableRunStore
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
    InMemoryCommandCursorStore,
    InMemoryCommandStore,
    JsonFileCommandCursorStore,
    JsonlCommandStore,
)
from .storage.ledger_chain import HashChainedLedgerStore, verify_ledger_chain
from .storage.observable import ObservableLedgerStore, ObservableLedgerStoreProtocol
from .storage.snapshots import Snapshot, SnapshotStore, InMemorySnapshotStore, JsonSnapshotStore
from .storage.offloading import OffloadingLedgerStore, OffloadingRunStore, offload_large_values
from .storage.artifacts import (
    Artifact,
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
    WorkflowBundle,
    WorkflowBundleEntrypoint,
    WorkflowBundleError,
    WorkflowBundleManifest,
    open_workflow_bundle,
    workflow_bundle_manifest_from_dict,
    workflow_bundle_manifest_to_dict,
)
from .history_bundle import (
    RUN_HISTORY_BUNDLE_VERSION_V1,
    export_run_history_bundle,
    persist_workflow_snapshot,
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
    # Scheduler
    "WorkflowRegistry",
    "Scheduler",
    "SchedulerStats",
    "ScheduledRuntime",
    "create_scheduled_runtime",
    # Storage backends
    "QueryableRunStore",
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
    "CommandCursorStore",
    "InMemoryCommandStore",
    "JsonlCommandStore",
    "InMemoryCommandCursorStore",
    "JsonFileCommandCursorStore",
    "SqliteCommandStore",
    "SqliteCommandCursorStore",
    "HashChainedLedgerStore",
    "verify_ledger_chain",
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
    "workflow_bundle_manifest_from_dict",
    "workflow_bundle_manifest_to_dict",
    "open_workflow_bundle",
    # Run history bundle (portable replay)
    "RUN_HISTORY_BUNDLE_VERSION_V1",
    "export_run_history_bundle",
    "persist_workflow_snapshot",
]
