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
from .storage.ledger_chain import HashChainedLedgerStore, verify_ledger_chain
from .storage.observable import ObservableLedgerStore, ObservableLedgerStoreProtocol
from .storage.snapshots import Snapshot, SnapshotStore, InMemorySnapshotStore, JsonSnapshotStore
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
    "HashChainedLedgerStore",
    "verify_ledger_chain",
    "ObservableLedgerStore",
    "ObservableLedgerStoreProtocol",
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
]

