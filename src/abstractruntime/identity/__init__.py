"""Identity and provenance primitives."""

from .diary import DiaryEntry, DiaryStore, build_diary_effect_handlers, diary_chain_id, verify_diary_chain
from .fingerprint import ActorFingerprint
from .prelude import render_summon_prelude
from .track_record import (
    OutcomeEvidence,
    Reputation,
    TaskKindRecord,
    TokenCounts,
    TrackRecord,
    TrackRecordError,
    TrustCell,
    compute_reputation,
    entity_fingerprint,
    entity_track_record,
    session_track_record,
    task_kind_of,
)

__all__ = [
    "ActorFingerprint",
    "DiaryEntry",
    "DiaryStore",
    "build_diary_effect_handlers",
    "diary_chain_id",
    "render_summon_prelude",
    "verify_diary_chain",
    # Track record and reputation (ADR 0052 §5-§6)
    "OutcomeEvidence",
    "Reputation",
    "TaskKindRecord",
    "TokenCounts",
    "TrackRecord",
    "TrackRecordError",
    "TrustCell",
    "compute_reputation",
    "entity_fingerprint",
    "entity_track_record",
    "session_track_record",
    "task_kind_of",
]


