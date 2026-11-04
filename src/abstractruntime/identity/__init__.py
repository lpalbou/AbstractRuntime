"""Identity and provenance primitives."""

from .digital_identity import (
    AvatarChoice,
    DigitalIdentity,
    IdentityFingerprint,
    IdentityInvalid,
    IdentityRef,
    IdentityUnreadable,
    VoiceChoice,
    default_identity,
    identity_at_rev,
    identity_change_cue,
    identity_fingerprint,
    identity_ref,
    parse_identity_ref,
    read_identity,
    read_run_identity_ref,
    record_default_identity,
    stamp_run_identity_ref,
    write_identity,
)
from .acquaintances import Acquaintance, Acquaintances, entity_acquaintances, session_acquaintances
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
    entity_track_record,
    session_track_record,
    task_kind_of,
)

__all__ = [
    "ActorFingerprint",
    "AvatarChoice",
    "DiaryEntry",
    "DiaryStore",
    "DigitalIdentity",
    "IdentityFingerprint",
    "IdentityInvalid",
    "IdentityRef",
    "IdentityUnreadable",
    "VoiceChoice",
    "build_diary_effect_handlers",
    "default_identity",
    "diary_chain_id",
    "identity_at_rev",
    "identity_change_cue",
    "identity_fingerprint",
    "identity_ref",
    "parse_identity_ref",
    "read_identity",
    "read_run_identity_ref",
    "record_default_identity",
    "render_summon_prelude",
    "stamp_run_identity_ref",
    "verify_diary_chain",
    "write_identity",
    # Track record and reputation (ADR 0052 §5-§6)
    "OutcomeEvidence",
    "Reputation",
    "TaskKindRecord",
    "TokenCounts",
    "TrackRecord",
    "TrackRecordError",
    "TrustCell",
    "compute_reputation",
    "entity_track_record",
    "session_track_record",
    "task_kind_of",
    # Acquaintances (derived: the entities and humans met)
    "Acquaintance",
    "Acquaintances",
    "entity_acquaintances",
    "session_acquaintances",
]


