"""abstractruntime.storage.artifacts

Artifact storage for large payloads.

Artifacts are stored by reference (artifact_id) instead of embedding
large data directly into RunState.vars. This keeps run state small
and JSON-serializable while supporting large payloads like:
- Documents and files
- Large LLM responses
- Tool outputs (search results, database queries)
- Media content (images, audio, video)

Design:
- Content-addressed: artifact_id is derived from content hash
- Metadata-rich: stores content_type, size, timestamps
- Simple interface: store/load/exists/delete
"""

from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import re
import threading
import uuid
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


# Valid artifact ID pattern: alphanumeric, hyphens, underscores
_ARTIFACT_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")
ARTIFACT_DESCRIPTOR_SCHEMA = "abstractruntime.artifact_descriptor.v1"
ARTIFACT_DESCRIPTOR_SCHEMA_VERSION = 1
ARTIFACT_DESCRIPTOR_SECURITY_REDACTION = "bounded_secret_key_redaction_v1"

_ARTIFACT_SECRET_KEY_EXACT = {
    "access_token",
    "api_token",
    "auth_token",
    "bearer_token",
    "bot_token",
    "client_secret",
    "csrf_token",
    "id_token",
    "password",
    "refresh_token",
    "secret",
    "session_token",
    "token",
}
_ARTIFACT_SECRET_KEY_FRAGMENTS = (
    "api_key",
    "api-key",
    "apikey",
    "authorization",
    "private_key",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for k, v in value.items():
            if k is None:
                continue
            out[str(k)] = _json_safe(v)
        return out
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


def _json_safe_dict(value: Any) -> Dict[str, Any]:
    safe = _json_safe(value)
    return safe if isinstance(safe, dict) else {}


def _artifact_safe_metadata_value(value: Any, *, depth: int = 0) -> Any:
    """Bound descriptor metadata and redact obvious credential fields.

    Artifact descriptors are indexed and rendered by Observer. They may include user
    prompts, model parameters, and provider metadata, but they must not become an
    accidental credential sink.
    """

    if depth > 6:
        return "#TRUNCATION: metadata depth limit"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, (bytes, bytearray)):
        return {"bytes": len(value), "redacted": True}
    if isinstance(value, str):
        return value[:12000] + "#TRUNCATION" if len(value) > 12000 else value
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        out = [_artifact_safe_metadata_value(item, depth=depth + 1) for item in items[:120]]
        if len(items) > 120:
            out.append({"truncated_items": len(items) - 120})
        return out
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key, raw in list(value.items())[:160]:
            if key is None:
                continue
            key_s = str(key)
            key_l = key_s.lower().strip()
            token_parts = [part for part in re.split(r"[^a-z0-9]+", key_l) if part]
            if (
                key_l in _ARTIFACT_SECRET_KEY_EXACT
                or any(fragment in key_l for fragment in _ARTIFACT_SECRET_KEY_FRAGMENTS)
                or "password" in token_parts
                or "secret" in token_parts
                or "token" in token_parts
            ):
                out[key_s] = "[redacted]" if raw not in (None, "") else raw
                continue
            if key_l in {"data", "b64_json", "image", "video", "audio", "content"} and isinstance(raw, str) and len(raw) > 2048:
                out[key_s] = {"chars": len(raw), "redacted": True}
                continue
            if callable(raw):
                continue
            out[key_s] = _artifact_safe_metadata_value(raw, depth=depth + 1)
        if len(value) > 160:
            out["truncated_keys"] = len(value) - 160
        return out
    return str(value)


def _artifact_security_for_generation(generation: Dict[str, Any]) -> Dict[str, Any]:
    fields = [
        key
        for key in ("prompt", "text", "input", "negative_prompt")
        if str((generation or {}).get(key) or "").strip()
    ]
    security: Dict[str, Any] = {"redaction": ARTIFACT_DESCRIPTOR_SECURITY_REDACTION}
    if fields:
        security.update(
            {
                "sensitivity": "user_content",
                "recorded_user_content_fields": fields,
                "prompt_storage": "descriptor_generation_bounded",
            }
        )
    return security


def build_artifact_descriptor_payload(
    *,
    semantic_kind: str,
    render_kind: str,
    modality: str = "",
    task: str = "",
    classification_source: str = "producer",
    content_type: str = "",
    session_id: str = "",
    workflow_id: str = "",
    node_id: str = "",
    step_id: str = "",
    effect_id: str = "",
    turn_id: str = "",
    ledger_cursor: str = "",
    parent_run_id: str = "",
    actor_id: str = "",
    run_id: str = "",
    request_id: str = "",
    source: str = "",
    producer: Optional[Dict[str, Any]] = None,
    generation: Optional[Dict[str, Any]] = None,
    source_refs: Optional[List[Dict[str, Any]]] = None,
    media: Optional[Dict[str, Any]] = None,
    metadata_schema: str = "abstractruntime.artifact_metadata.v1",
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Build the canonical descriptor plus structured metadata for stored artifacts.

    Gateway and provider integrations should use this helper instead of hand-authoring
    descriptor dictionaries. Runtime remains the owner of descriptor semantics; Gateway
    may project or index the payload.
    """

    semantic_kind0 = str(semantic_kind or "").strip().lower()
    render_kind0 = str(render_kind or "").strip().lower() or _render_kind_from_content_type(content_type)
    modality0 = str(modality or "").strip().lower() or (
        semantic_kind0 if semantic_kind0 in {"voice", "music", "sound"} else render_kind0
    )
    generation0 = _artifact_safe_metadata_value(generation or {})
    if not isinstance(generation0, dict):
        generation0 = {}
    producer0 = _artifact_safe_metadata_value(producer or {})
    if not isinstance(producer0, dict):
        producer0 = {}
    refs0 = _artifact_safe_metadata_value(source_refs or [])
    if not isinstance(refs0, list):
        refs0 = []
    media0 = _artifact_safe_metadata_value(media or {})
    if not isinstance(media0, dict):
        media0 = {}

    provenance: Dict[str, Any] = {}
    if source:
        provenance["source"] = str(source)
    if run_id:
        provenance["run_id"] = str(run_id)
    if request_id:
        provenance["request_id"] = str(request_id)
    security = _artifact_security_for_generation(generation0)

    descriptor: Dict[str, Any] = {
        "schema": ARTIFACT_DESCRIPTOR_SCHEMA,
        "schema_version": ARTIFACT_DESCRIPTOR_SCHEMA_VERSION,
        "semantic_kind": semantic_kind0,
        "render_kind": render_kind0,
        "modality": modality0,
        "task": str(task or "").strip().lower(),
        "classification_source": str(classification_source or "producer").strip() or "producer",
        "session_id": _str_or_none(session_id),
        "workflow_id": _str_or_none(workflow_id),
        "node_id": _str_or_none(node_id),
        "step_id": _str_or_none(step_id),
        "effect_id": _str_or_none(effect_id),
        "turn_id": _str_or_none(turn_id),
        "ledger_cursor": _str_or_none(ledger_cursor),
        "parent_run_id": _str_or_none(parent_run_id),
        "actor_id": _str_or_none(actor_id),
        "producer": producer0,
        "provenance": provenance,
        "generation": generation0,
        "media": media0,
        "source_refs": refs0,
        "security": security,
    }
    metadata: Dict[str, Any] = {
        "schema": str(metadata_schema or "abstractruntime.artifact_metadata.v1"),
        "producer": producer0,
        "generation": generation0,
        "source_refs": refs0,
        "security": security,
    }
    return descriptor, metadata


def _str_or_none(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def _str_dict(value: Any) -> Dict[str, str]:
    if not isinstance(value, dict):
        return {}
    out: Dict[str, str] = {}
    for k, v in value.items():
        if k is None or v is None:
            continue
        ks = str(k).strip()
        if not ks:
            continue
        out[ks] = str(v)
    return out


def _render_kind_from_content_type(content_type: str) -> str:
    ct = str(content_type or "").strip().lower()
    if ct.startswith("image/"):
        return "image"
    if ct.startswith("audio/"):
        return "audio"
    if ct.startswith("video/"):
        return "video"
    if ct in {
        "application/javascript",
        "application/x-javascript",
        "application/typescript",
        "application/x-python-code",
        "application/x-sh",
        "application/x-shellscript",
        "text/javascript",
        "text/jsx",
        "text/typescript",
        "text/tsx",
        "text/x-python",
        "text/x-shellscript",
        "text/x-script.python",
        "text/x-go",
        "text/x-rust",
        "text/x-java-source",
        "text/x-c",
        "text/x-c++",
        "text/x-csharp",
        "text/x-php",
        "text/x-ruby",
    }:
        return "code"
    if ct in {"application/json"} or ct.endswith("+json"):
        return "json"
    if ct in {"text/markdown", "text/x-markdown"}:
        return "markdown"
    if "html" in ct:
        return "html"
    if ct.startswith("text/"):
        return "text"
    if "pdf" in ct or "document" in ct or "officedocument" in ct:
        return "document"
    return "binary"


def _semantic_kind_from_tags(
    *,
    content_type: str,
    tags: Dict[str, str],
    render_kind: str,
) -> str:
    explicit = _str_or_none(tags.get("semantic_kind") or tags.get("artifact_type"))
    if explicit:
        return explicit.lower()
    modality = str(tags.get("modality") or "").strip().lower()
    task = str(tags.get("task") or tags.get("provider_task") or "").strip().lower()
    kind = str(tags.get("kind") or "").strip().lower()
    if modality in {"voice", "music", "sound", "image", "video", "text", "document", "code"}:
        return modality
    if task in {"code", "coding", "code_generation", "script", "program"}:
        return "code"
    if kind in {"code", "source_code", "script", "program", "executable"}:
        return "code"
    if task in {"tts", "speech", "speech_generation"}:
        return "voice"
    if task in {"music", "music_generation", "text_to_music"}:
        return "music"
    if task in {"stt", "transcription", "speech_to_text"}:
        return "transcript"
    if kind in {"attachment", "evidence", "workflow_snapshot", "conversation_span", "memory_note"}:
        return kind
    if render_kind in {"image", "video", "audio", "document", "json", "html", "markdown", "code", "text"}:
        return render_kind
    return _render_kind_from_content_type(content_type)


def _inspect_media(content: bytes, *, content_type: str) -> Dict[str, Any]:
    ct = str(content_type or "").strip().lower()
    if ct.startswith("image/"):
        try:
            from PIL import Image  # type: ignore

            with Image.open(io.BytesIO(content)) as img:
                width, height = img.size
                return {
                    "kind": "image",
                    "width": int(width),
                    "height": int(height),
                    "format": str(img.format or "").lower() or None,
                    "inspection_source": "pillow",
                }
        except Exception as e:
            return {
                "kind": "image",
                "inspection_source": "pillow",
                "inspection_error": str(e),
            }
    if ct in {"audio/wav", "audio/x-wav", "audio/wave"} or ct.endswith("/wav"):
        try:
            with wave.open(io.BytesIO(content), "rb") as wf:
                frame_count = int(wf.getnframes())
                sample_rate = int(wf.getframerate())
                channels = int(wf.getnchannels())
                duration_s = float(frame_count / sample_rate) if sample_rate > 0 else 0.0
                return {
                    "kind": "audio",
                    "duration_s": duration_s,
                    "sample_rate": sample_rate,
                    "channels": channels,
                    "frames": frame_count,
                    "inspection_source": "wave",
                }
        except Exception as e:
            return {
                "kind": "audio",
                "inspection_source": "wave",
                "inspection_error": str(e),
            }
    if ct.startswith("audio/"):
        return {
            "kind": "audio",
            "inspection_source": "unsupported",
            "inspection_status": "duration_unavailable",
        }
    if ct.startswith("video/"):
        return {
            "kind": "video",
            "inspection_source": "unsupported",
            "inspection_status": "video_inspector_not_configured",
        }
    return {}


@dataclass
class ArtifactDescriptor:
    """Versioned human/runtime descriptor for an artifact.

    Runtime owns this descriptor. Gateway may index and project it; Observer should render it.
    """

    schema: str = ARTIFACT_DESCRIPTOR_SCHEMA
    schema_version: int = ARTIFACT_DESCRIPTOR_SCHEMA_VERSION
    semantic_kind: str = ""
    render_kind: str = ""
    modality: str = ""
    task: str = ""
    classification_source: str = ""
    session_id: Optional[str] = None
    workflow_id: Optional[str] = None
    node_id: Optional[str] = None
    step_id: Optional[str] = None
    effect_id: Optional[str] = None
    turn_id: Optional[str] = None
    ledger_cursor: Optional[str] = None
    parent_run_id: Optional[str] = None
    actor_id: Optional[str] = None
    producer: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    generation: Dict[str, Any] = field(default_factory=dict)
    media: Dict[str, Any] = field(default_factory=dict)
    source_refs: List[Dict[str, Any]] = field(default_factory=list)
    links: Dict[str, Any] = field(default_factory=dict)
    security: Dict[str, Any] = field(default_factory=dict)
    actions: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": int(self.schema_version),
            "semantic_kind": self.semantic_kind,
            "render_kind": self.render_kind,
            "modality": self.modality,
            "task": self.task,
            "classification_source": self.classification_source,
            "session_id": self.session_id,
            "workflow_id": self.workflow_id,
            "node_id": self.node_id,
            "step_id": self.step_id,
            "effect_id": self.effect_id,
            "turn_id": self.turn_id,
            "ledger_cursor": self.ledger_cursor,
            "parent_run_id": self.parent_run_id,
            "actor_id": self.actor_id,
            "producer": _json_safe_dict(self.producer),
            "provenance": _json_safe_dict(self.provenance),
            "generation": _json_safe_dict(self.generation),
            "media": _json_safe_dict(self.media),
            "source_refs": _json_safe(self.source_refs),
            "links": _json_safe_dict(self.links),
            "security": _json_safe_dict(self.security),
            "actions": _json_safe_dict(self.actions),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "ArtifactDescriptor":
        d = data if isinstance(data, dict) else {}
        try:
            schema_version = int(d.get("schema_version") or ARTIFACT_DESCRIPTOR_SCHEMA_VERSION)
        except Exception:
            schema_version = ARTIFACT_DESCRIPTOR_SCHEMA_VERSION
        source_refs = _json_safe(d.get("source_refs") or [])
        if not isinstance(source_refs, list):
            source_refs = []
        return cls(
            schema=str(d.get("schema") or ARTIFACT_DESCRIPTOR_SCHEMA),
            schema_version=schema_version,
            semantic_kind=str(d.get("semantic_kind") or "").strip().lower(),
            render_kind=str(d.get("render_kind") or "").strip().lower(),
            modality=str(d.get("modality") or "").strip().lower(),
            task=str(d.get("task") or "").strip().lower(),
            classification_source=str(d.get("classification_source") or "").strip(),
            session_id=_str_or_none(d.get("session_id")),
            workflow_id=_str_or_none(d.get("workflow_id")),
            node_id=_str_or_none(d.get("node_id")),
            step_id=_str_or_none(d.get("step_id")),
            effect_id=_str_or_none(d.get("effect_id")),
            turn_id=_str_or_none(d.get("turn_id")),
            ledger_cursor=_str_or_none(d.get("ledger_cursor")),
            parent_run_id=_str_or_none(d.get("parent_run_id")),
            actor_id=_str_or_none(d.get("actor_id")),
            producer=_json_safe_dict(d.get("producer") or {}),
            provenance=_json_safe_dict(d.get("provenance") or {}),
            generation=_json_safe_dict(d.get("generation") or {}),
            media=_json_safe_dict(d.get("media") or {}),
            source_refs=source_refs,
            links=_json_safe_dict(d.get("links") or {}),
            security=_json_safe_dict(d.get("security") or {}),
            actions=_json_safe_dict(d.get("actions") or {}),
        )


@dataclass
class ArtifactAccessStats:
    """Access counters updated through explicit ArtifactStore APIs."""

    access_count: int = 0
    metadata_access_count: int = 0
    content_access_count: int = 0
    preview_count: int = 0
    download_count: int = 0
    last_accessed_at: Optional[str] = None
    last_action: Optional[str] = None
    last_actor_id: Optional[str] = None
    last_session_id: Optional[str] = None
    last_run_id: Optional[str] = None
    actions: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "access_count": int(self.access_count),
            "metadata_access_count": int(self.metadata_access_count),
            "content_access_count": int(self.content_access_count),
            "preview_count": int(self.preview_count),
            "download_count": int(self.download_count),
            "last_accessed_at": self.last_accessed_at,
            "last_action": self.last_action,
            "last_actor_id": self.last_actor_id,
            "last_session_id": self.last_session_id,
            "last_run_id": self.last_run_id,
            "actions": dict(self.actions or {}),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "ArtifactAccessStats":
        d = data if isinstance(data, dict) else {}

        def _int(name: str) -> int:
            try:
                return max(0, int(d.get(name) or 0))
            except Exception:
                return 0

        actions: Dict[str, int] = {}
        if isinstance(d.get("actions"), dict):
            for k, v in d["actions"].items():
                try:
                    actions[str(k)] = max(0, int(v))
                except Exception:
                    continue
        return cls(
            access_count=_int("access_count"),
            metadata_access_count=_int("metadata_access_count"),
            content_access_count=_int("content_access_count"),
            preview_count=_int("preview_count"),
            download_count=_int("download_count"),
            last_accessed_at=_str_or_none(d.get("last_accessed_at")),
            last_action=_str_or_none(d.get("last_action")),
            last_actor_id=_str_or_none(d.get("last_actor_id")),
            last_session_id=_str_or_none(d.get("last_session_id")),
            last_run_id=_str_or_none(d.get("last_run_id")),
            actions=actions,
        )

    def record(
        self,
        *,
        action: str = "access",
        actor_id: Optional[str] = None,
        session_id: Optional[str] = None,
        run_id: Optional[str] = None,
        at: Optional[str] = None,
    ) -> None:
        action0 = str(action or "access").strip().lower() or "access"
        self.access_count += 1
        self.actions[action0] = int(self.actions.get(action0, 0)) + 1
        if action0 == "metadata":
            self.metadata_access_count += 1
        elif action0 in {"content", "load", "read", "open"}:
            self.content_access_count += 1
        elif action0 == "preview":
            self.preview_count += 1
        elif action0 == "download":
            self.download_count += 1
        self.last_accessed_at = str(at or utc_now_iso())
        self.last_action = action0
        self.last_actor_id = _str_or_none(actor_id)
        self.last_session_id = _str_or_none(session_id)
        self.last_run_id = _str_or_none(run_id)


def _normalize_descriptor(
    *,
    descriptor: Optional[Union[ArtifactDescriptor, Dict[str, Any]]],
    content_type: str,
    run_id: Optional[str],
    tags: Dict[str, str],
    media: Optional[Dict[str, Any]] = None,
) -> ArtifactDescriptor:
    if isinstance(descriptor, ArtifactDescriptor):
        d = ArtifactDescriptor.from_dict(descriptor.to_dict())
    else:
        d = ArtifactDescriptor.from_dict(descriptor or {})

    render_kind = d.render_kind or _render_kind_from_content_type(content_type)
    semantic_kind = d.semantic_kind or _semantic_kind_from_tags(
        content_type=content_type,
        tags=tags,
        render_kind=render_kind,
    )
    modality = d.modality or str(tags.get("modality") or "").strip().lower()
    if not modality:
        modality = semantic_kind if semantic_kind in {"voice", "music", "sound"} else render_kind

    d.render_kind = render_kind
    d.semantic_kind = semantic_kind
    d.modality = modality
    d.task = d.task or str(tags.get("task") or tags.get("provider_task") or "").strip().lower()
    if not d.classification_source:
        if descriptor is not None:
            d.classification_source = "runtime_declared"
        elif tags.get("semantic_kind") or tags.get("artifact_type") or tags.get("modality") or tags.get("task"):
            d.classification_source = "runtime_tags"
        else:
            d.classification_source = "runtime_mime_inferred"

    d.session_id = d.session_id or _str_or_none(tags.get("session_id"))
    d.workflow_id = d.workflow_id or _str_or_none(tags.get("workflow_id") or tags.get("workflow"))
    d.node_id = d.node_id or _str_or_none(tags.get("node_id") or tags.get("node"))
    d.step_id = d.step_id or _str_or_none(tags.get("step_id"))
    d.effect_id = d.effect_id or _str_or_none(tags.get("effect_id") or tags.get("effect_idempotency_key"))
    d.turn_id = d.turn_id or _str_or_none(tags.get("turn_id") or tags.get("turn"))
    d.ledger_cursor = d.ledger_cursor or _str_or_none(tags.get("ledger_cursor") or tags.get("step_cursor"))
    d.parent_run_id = d.parent_run_id or _str_or_none(tags.get("parent_run_id"))
    d.actor_id = d.actor_id or _str_or_none(tags.get("actor_id"))
    if run_id and "run_id" not in d.provenance:
        d.provenance["run_id"] = str(run_id)
    if tags.get("source") and "source" not in d.provenance:
        d.provenance["source"] = str(tags.get("source"))
    if media:
        merged = dict(media)
        merged.update({k: v for k, v in d.media.items() if v is not None})
        d.media = _json_safe_dict(merged)
    return d


def _merge_descriptor_payload(
    base: Dict[str, Any],
    descriptor: Union[ArtifactDescriptor, Dict[str, Any]],
) -> Dict[str, Any]:
    """Merge descriptor enrichment without erasing existing inspected facts."""
    merged = dict(base or {})
    incoming = ArtifactDescriptor.from_dict(
        descriptor.to_dict() if isinstance(descriptor, ArtifactDescriptor) else descriptor
    ).to_dict()
    dict_fields = {"producer", "provenance", "generation", "media", "links", "security", "actions"}
    for key, value in incoming.items():
        if key in {"schema", "schema_version"}:
            merged[key] = value
            continue
        if key in dict_fields:
            if isinstance(value, dict) and value:
                previous = _json_safe_dict(merged.get(key) or {})
                previous.update(value)
                merged[key] = previous
            continue
        if key == "source_refs":
            if isinstance(value, list) and value:
                merged[key] = value
            continue
        if value is None:
            continue
        if isinstance(value, str) and not value:
            continue
        if isinstance(value, (list, dict)) and not value:
            continue
        merged[key] = value
    return merged


def _artifact_facet_value(meta: "ArtifactMetadata", field: str) -> str:
    field0 = str(field or "").strip()
    descriptor_fields = {
        "semantic_kind",
        "render_kind",
        "modality",
        "task",
        "classification_source",
        "session_id",
        "workflow_id",
        "node_id",
        "turn_id",
        "actor_id",
    }
    if field0 in descriptor_fields:
        return str(getattr(meta.descriptor, field0, None) or "")
    if field0 in {"content_type", "run_id", "artifact_id", "created_at", "blob_id"}:
        return str(getattr(meta, field0, None) or "")
    if field0.startswith("tag:"):
        return str((meta.tags or {}).get(field0.split(":", 1)[1], "") or "")
    if field0.startswith("metadata:"):
        return str((meta.metadata or {}).get(field0.split(":", 1)[1], "") or "")
    return ""


@dataclass
class ArtifactMetadata:
    """Metadata about a stored artifact."""

    artifact_id: str
    content_type: str  # MIME type or semantic type
    size_bytes: int
    created_at: str
    blob_id: Optional[str] = None  # Global (cross-run) content hash for dedupe
    run_id: Optional[str] = None  # Optional association with a run
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    descriptor: ArtifactDescriptor = field(default_factory=ArtifactDescriptor)
    access: ArtifactAccessStats = field(default_factory=ArtifactAccessStats)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "blob_id": self.blob_id,
            "content_type": self.content_type,
            "size_bytes": int(self.size_bytes),
            "created_at": self.created_at,
            "run_id": self.run_id,
            "tags": _str_dict(self.tags),
            "metadata": _json_safe_dict(self.metadata),
            "descriptor": self.descriptor.to_dict(),
            "access": self.access.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArtifactMetadata":
        tags = _str_dict(data.get("tags") or {})
        content_type = str(data["content_type"])
        run_id = data.get("run_id")
        descriptor = _normalize_descriptor(
            descriptor=data.get("descriptor") if isinstance(data.get("descriptor"), dict) else None,
            content_type=content_type,
            run_id=str(run_id) if run_id is not None else None,
            tags=tags,
        )
        return cls(
            artifact_id=data["artifact_id"],
            blob_id=data.get("blob_id"),
            content_type=content_type,
            size_bytes=data["size_bytes"],
            created_at=data["created_at"],
            run_id=run_id,
            tags=tags,
            metadata=_json_safe_dict(data.get("metadata") or {}),
            descriptor=descriptor,
            access=ArtifactAccessStats.from_dict(data.get("access") or {}),
        )


@dataclass
class Artifact:
    """An artifact with its content and metadata."""

    metadata: ArtifactMetadata
    content: bytes

    @property
    def artifact_id(self) -> str:
        return self.metadata.artifact_id

    @property
    def content_type(self) -> str:
        return self.metadata.content_type

    def as_text(self, encoding: str = "utf-8") -> str:
        """Decode content as text."""
        return self.content.decode(encoding)

    def as_json(self) -> Any:
        """Parse content as JSON."""
        return json.loads(self.content.decode("utf-8"))


def compute_artifact_id(content: bytes, *, run_id: Optional[str] = None) -> str:
    """Compute a deterministic artifact id.

    By default, artifacts are content-addressed (SHA-256, truncated) so the same bytes
    produce the same id.

    If `run_id` is provided, the id is *namespaced to that run* so each run can have a
    distinct artifact_id (while still enabling cross-run blob dedupe via `blob_id`).
    """
    h = hashlib.sha256()
    if run_id is not None:
        rid = str(run_id).strip()
        if rid:
            h.update(rid.encode("utf-8"))
            h.update(b"\0")
    h.update(content)
    return h.hexdigest()[:32]


def compute_blob_id(content: bytes) -> str:
    """Compute a stable, global content hash for artifact blob dedupe."""
    return hashlib.sha256(content).hexdigest()


def validate_artifact_id(artifact_id: str) -> None:
    """Validate artifact ID to prevent path traversal attacks.
    
    Raises:
        ValueError: If artifact_id contains invalid characters.
    """
    if not artifact_id:
        raise ValueError("artifact_id cannot be empty")
    if not _ARTIFACT_ID_PATTERN.match(artifact_id):
        raise ValueError(
            f"Invalid artifact_id '{artifact_id}': must contain only "
            "alphanumeric characters, hyphens, and underscores"
        )


class ArtifactStore(ABC):
    """Abstract base class for artifact storage."""

    @abstractmethod
    def store(
        self,
        content: bytes,
        *,
        content_type: str = "application/octet-stream",
        run_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        descriptor: Optional[Union[ArtifactDescriptor, Dict[str, Any]]] = None,
        artifact_id: Optional[str] = None,
    ) -> ArtifactMetadata:
        """Store an artifact and return its metadata.

        Args:
            content: The artifact content as bytes.
            content_type: MIME type or semantic type.
            run_id: Optional run ID to associate with.
            tags: Optional key-value tags.
            metadata: Optional structured metadata that is stored but not blindly indexed.
            descriptor: Optional canonical ArtifactDescriptorV1 fields.
            artifact_id: Optional explicit ID (defaults to content hash).

        Returns:
            ArtifactMetadata with the artifact_id.
        """
        ...

    def update_metadata(
        self,
        artifact_id: str,
        *,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        descriptor: Optional[Union[ArtifactDescriptor, Dict[str, Any]]] = None,
        replace_tags: bool = False,
        replace_metadata: bool = False,
    ) -> Optional[ArtifactMetadata]:
        """Update descriptor metadata for an artifact when the store supports it."""
        _ = (artifact_id, tags, metadata, descriptor, replace_tags, replace_metadata)
        return None

    def record_access(
        self,
        artifact_id: str,
        *,
        action: str = "access",
        actor_id: Optional[str] = None,
        session_id: Optional[str] = None,
        run_id: Optional[str] = None,
        at: Optional[str] = None,
    ) -> Optional[ArtifactMetadata]:
        """Record an explicit artifact access event.

        Plain ``load()`` and ``get_metadata()`` remain side-effect free. HTTP/UI layers
        should call this method for previews, downloads, content reads, or metadata views.
        """
        _ = (artifact_id, action, actor_id, session_id, run_id, at)
        return None

    @abstractmethod
    def load(self, artifact_id: str) -> Optional[Artifact]:
        """Load an artifact by ID.

        Args:
            artifact_id: The artifact ID.

        Returns:
            Artifact if found, None otherwise.
        """
        ...

    @abstractmethod
    def get_metadata(self, artifact_id: str) -> Optional[ArtifactMetadata]:
        """Get artifact metadata without loading content.

        Args:
            artifact_id: The artifact ID.

        Returns:
            ArtifactMetadata if found, None otherwise.
        """
        ...

    @abstractmethod
    def exists(self, artifact_id: str) -> bool:
        """Check if an artifact exists.

        Args:
            artifact_id: The artifact ID.

        Returns:
            True if artifact exists.
        """
        ...

    @abstractmethod
    def delete(self, artifact_id: str) -> bool:
        """Delete an artifact.

        Args:
            artifact_id: The artifact ID.

        Returns:
            True if deleted, False if not found.
        """
        ...

    @abstractmethod
    def list_by_run(self, run_id: str) -> List[ArtifactMetadata]:
        """List all artifacts associated with a run.

        Args:
            run_id: The run ID.

        Returns:
            List of ArtifactMetadata.
        """
        ...

    @abstractmethod
    def list_all(self, *, limit: int = 1000) -> List[ArtifactMetadata]:
        """List all artifacts.

        Args:
            limit: Maximum number of artifacts to return. Values <= 0 mean unlimited.

        Returns:
            List of ArtifactMetadata.
        """
        ...

    def delete_by_run(self, run_id: str) -> int:
        """Delete all artifacts associated with a run.

        Args:
            run_id: The run ID.

        Returns:
            Number of artifacts deleted.
        """
        artifacts = self.list_by_run(run_id)
        count = 0
        for meta in artifacts:
            if self.delete(meta.artifact_id):
                count += 1
        return count

    def search(
        self,
        *,
        run_id: Optional[str] = None,
        content_type: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        semantic_kind: Optional[str] = None,
        render_kind: Optional[str] = None,
        modality: Optional[str] = None,
        session_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        node_id: Optional[str] = None,
        created_after: Optional[str] = None,
        created_before: Optional[str] = None,
        offset: int = 0,
        order: str = "desc",
        limit: int = 1000,
    ) -> List[ArtifactMetadata]:
        """Filter artifacts by simple metadata fields.

        This is intentionally a *metadata filter*, not semantic search. Semantic/embedding
        retrieval belongs in AbstractMemory or higher-level components.
        """
        if run_id is None:
            candidates = list(self.list_all(limit=0))
        else:
            candidates = list(self.list_by_run(run_id))

        if content_type is not None:
            candidates = [m for m in candidates if m.content_type == content_type]

        if tags:
            candidates = [
                m
                for m in candidates
                if all((m.tags or {}).get(k) == v for k, v in tags.items())
            ]

        def _matches_descriptor(m: ArtifactMetadata) -> bool:
            d = m.descriptor
            if semantic_kind and d.semantic_kind != str(semantic_kind).strip().lower():
                return False
            if render_kind and d.render_kind != str(render_kind).strip().lower():
                return False
            if modality and d.modality != str(modality).strip().lower():
                return False
            if session_id and d.session_id != str(session_id).strip():
                return False
            if workflow_id and d.workflow_id != str(workflow_id).strip():
                return False
            if node_id and d.node_id != str(node_id).strip():
                return False
            if created_after and str(m.created_at or "") < str(created_after):
                return False
            if created_before and str(m.created_at or "") > str(created_before):
                return False
            return True

        candidates = [m for m in candidates if _matches_descriptor(m)]
        reverse = str(order or "desc").strip().lower() != "asc"
        candidates.sort(key=lambda m: (str(m.created_at or ""), str(m.artifact_id or "")), reverse=reverse)
        start = max(0, int(offset or 0))
        sliced = candidates[start:]
        if int(limit) <= 0:
            return sliced
        return sliced[: int(limit)]

    def count(self, **filters: Any) -> int:
        """Return an exact count for the supported metadata filters."""
        filters["limit"] = 0
        return len(self.search(**filters))

    def facet_counts(self, field: str, **filters: Any) -> Dict[str, int]:
        """Return exact counts by a descriptor or metadata facet."""
        filters["limit"] = 0
        counts: Dict[str, int] = {}
        for meta in self.search(**filters):
            value = _artifact_facet_value(meta, field)
            if not value:
                value = "(none)"
            counts[value] = counts.get(value, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    def stats(self, *, facets: Optional[List[str]] = None, **filters: Any) -> Dict[str, Any]:
        """Return exact count, byte total, and optional facets for metadata filters."""
        filters0 = dict(filters or {})
        filters0.pop("limit", None)
        filters0.pop("offset", None)
        filters0.pop("order", None)
        rows = self.search(limit=0, **filters0)
        return {
            "total": len(rows),
            "total_bytes": sum(int(m.size_bytes) for m in rows if isinstance(m.size_bytes, int) and m.size_bytes >= 0),
            "facets": {field: self.facet_counts(field, **filters0) for field in (facets or [])},
            "filters": filters0,
            "exact": True,
            "source": "artifact_store_scan",
        }

    def content_path(self, artifact_id: str) -> Optional[Path]:
        """Return a local content path when this store is file-backed.

        Stores that cannot expose a stable local file path return ``None``; callers
        should fall back to ``load()``.
        """
        return None

    # Convenience methods

    def store_text(
        self,
        text: str,
        *,
        content_type: str = "text/plain",
        encoding: str = "utf-8",
        run_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        descriptor: Optional[Union[ArtifactDescriptor, Dict[str, Any]]] = None,
    ) -> ArtifactMetadata:
        """Store text content."""
        return self.store(
            text.encode(encoding),
            content_type=content_type,
            run_id=run_id,
            tags=tags,
            metadata=metadata,
            descriptor=descriptor,
        )

    def store_json(
        self,
        data: Any,
        *,
        run_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        descriptor: Optional[Union[ArtifactDescriptor, Dict[str, Any]]] = None,
    ) -> ArtifactMetadata:
        """Store JSON-serializable data."""
        content = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return self.store(
            content,
            content_type="application/json",
            run_id=run_id,
            tags=tags,
            metadata=metadata,
            descriptor=descriptor,
        )

    def load_text(self, artifact_id: str, encoding: str = "utf-8") -> Optional[str]:
        """Load artifact as text."""
        artifact = self.load(artifact_id)
        if artifact is None:
            return None
        return artifact.as_text(encoding)

    def load_json(self, artifact_id: str) -> Optional[Any]:
        """Load artifact as JSON."""
        artifact = self.load(artifact_id)
        if artifact is None:
            return None
        return artifact.as_json()


class InMemoryArtifactStore(ArtifactStore):
    """In-memory artifact store for testing and development."""

    def __init__(self) -> None:
        self._artifacts: Dict[str, Artifact] = {}
        self._lock = threading.RLock()

    def store(
        self,
        content: bytes,
        *,
        content_type: str = "application/octet-stream",
        run_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        descriptor: Optional[Union[ArtifactDescriptor, Dict[str, Any]]] = None,
        artifact_id: Optional[str] = None,
    ) -> ArtifactMetadata:
        if artifact_id is None:
            artifact_id = compute_artifact_id(content, run_id=run_id)
        validate_artifact_id(artifact_id)
        tags0 = _str_dict(tags or {})
        metadata0 = _json_safe_dict(metadata or {})
        descriptor0 = _normalize_descriptor(
            descriptor=descriptor,
            content_type=content_type,
            run_id=run_id,
            tags=tags0,
            media=_inspect_media(bytes(content), content_type=content_type),
        )

        metadata = ArtifactMetadata(
            artifact_id=artifact_id,
            blob_id=compute_blob_id(content),
            content_type=content_type,
            size_bytes=len(content),
            created_at=utc_now_iso(),
            run_id=run_id,
            tags=tags0,
            metadata=metadata0,
            descriptor=descriptor0,
        )

        with self._lock:
            self._artifacts[artifact_id] = Artifact(metadata=metadata, content=content)
        return metadata

    def load(self, artifact_id: str) -> Optional[Artifact]:
        with self._lock:
            return self._artifacts.get(artifact_id)

    def get_metadata(self, artifact_id: str) -> Optional[ArtifactMetadata]:
        with self._lock:
            artifact = self._artifacts.get(artifact_id)
        if artifact is None:
            return None
        return artifact.metadata

    def update_metadata(
        self,
        artifact_id: str,
        *,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        descriptor: Optional[Union[ArtifactDescriptor, Dict[str, Any]]] = None,
        replace_tags: bool = False,
        replace_metadata: bool = False,
    ) -> Optional[ArtifactMetadata]:
        validate_artifact_id(artifact_id)
        with self._lock:
            artifact = self._artifacts.get(artifact_id)
            if artifact is None:
                return None
            meta = artifact.metadata
            tags0 = _str_dict(tags or {})
            if tags is not None:
                meta.tags = tags0 if replace_tags else {**(meta.tags or {}), **tags0}
            metadata0 = _json_safe_dict(metadata or {})
            if metadata is not None:
                meta.metadata = metadata0 if replace_metadata else {**(meta.metadata or {}), **metadata0}
            if descriptor is not None or tags is not None:
                base = meta.descriptor.to_dict()
                if descriptor is not None:
                    base = _merge_descriptor_payload(base, descriptor)
                meta.descriptor = _normalize_descriptor(
                    descriptor=base,
                    content_type=meta.content_type,
                    run_id=meta.run_id,
                    tags=meta.tags,
                )
            return meta

    def record_access(
        self,
        artifact_id: str,
        *,
        action: str = "access",
        actor_id: Optional[str] = None,
        session_id: Optional[str] = None,
        run_id: Optional[str] = None,
        at: Optional[str] = None,
    ) -> Optional[ArtifactMetadata]:
        with self._lock:
            artifact = self._artifacts.get(artifact_id)
            if artifact is None:
                return None
            artifact.metadata.access.record(
                action=action,
                actor_id=actor_id,
                session_id=session_id,
                run_id=run_id,
                at=at,
            )
            return artifact.metadata

    def exists(self, artifact_id: str) -> bool:
        with self._lock:
            return artifact_id in self._artifacts

    def delete(self, artifact_id: str) -> bool:
        with self._lock:
            if artifact_id in self._artifacts:
                del self._artifacts[artifact_id]
                return True
        return False

    def list_by_run(self, run_id: str) -> List[ArtifactMetadata]:
        with self._lock:
            return [
                a.metadata
                for a in self._artifacts.values()
                if a.metadata.run_id == run_id
            ]

    def list_all(self, *, limit: int = 1000) -> List[ArtifactMetadata]:
        with self._lock:
            results = [a.metadata for a in self._artifacts.values()]
        # Sort by created_at descending
        results.sort(key=lambda m: m.created_at, reverse=True)
        if int(limit) <= 0:
            return results
        return results[:limit]


class FileArtifactStore(ArtifactStore):
    """File-based artifact store.

    Directory structure (v1, cross-run blob dedupe):
        base_dir/
            artifacts/
                blobs/{blob_id}.bin   # global content-addressed bytes
                refs/{artifact_id}.meta  # per-artifact metadata (points to blob_id)

    Legacy layout (v0) is still supported for reads:
        base_dir/
            artifacts/
                {artifact_id}.bin
                {artifact_id}.meta
    """

    def __init__(self, base_dir: Union[str, Path]) -> None:
        self._base = Path(base_dir)
        self._artifacts_dir = self._base / "artifacts"
        self._blobs_dir = self._artifacts_dir / "blobs"
        self._refs_dir = self._artifacts_dir / "refs"
        self._catalog_path = self._artifacts_dir / "artifact_catalog.sqlite3"
        self._lock = threading.RLock()
        self._blobs_dir.mkdir(parents=True, exist_ok=True)
        self._refs_dir.mkdir(parents=True, exist_ok=True)
        self._init_catalog()
        self._rebuild_catalog_if_empty()

    def _connect_catalog(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._catalog_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_catalog(self) -> None:
        with self._connect_catalog() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS artifact_catalog (
                    artifact_id TEXT PRIMARY KEY,
                    run_id TEXT,
                    blob_id TEXT,
                    content_type TEXT,
                    size_bytes INTEGER,
                    created_at TEXT,
                    semantic_kind TEXT,
                    render_kind TEXT,
                    modality TEXT,
                    task TEXT,
                    session_id TEXT,
                    workflow_id TEXT,
                    node_id TEXT,
                    turn_id TEXT,
                    classification_source TEXT,
                    tags_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    descriptor_json TEXT NOT NULL,
                    access_json TEXT NOT NULL,
                    raw_json TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_artifact_catalog_run ON artifact_catalog(run_id, created_at DESC, artifact_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_artifact_catalog_session ON artifact_catalog(session_id, created_at DESC, artifact_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_artifact_catalog_semantic ON artifact_catalog(semantic_kind, created_at DESC, artifact_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_artifact_catalog_render ON artifact_catalog(render_kind, created_at DESC, artifact_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_artifact_catalog_workflow ON artifact_catalog(workflow_id, created_at DESC, artifact_id)")

    def _catalog_count(self) -> int:
        try:
            with self._connect_catalog() as conn:
                row = conn.execute("SELECT COUNT(*) AS n FROM artifact_catalog").fetchone()
                return int(row["n"] if row is not None else 0)
        except Exception:
            return 0

    def _metadata_paths(self) -> List[Path]:
        return list(self._refs_dir.glob("*.meta")) + list(self._artifacts_dir.glob("*.meta"))

    def _read_metadata_path(self, metadata_path: Path) -> Optional[ArtifactMetadata]:
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata_dict = json.load(f)
            return ArtifactMetadata.from_dict(metadata_dict)
        except (json.JSONDecodeError, IOError, KeyError, TypeError, ValueError):
            return None

    def _scan_metadata_files(self) -> List[ArtifactMetadata]:
        results: List[ArtifactMetadata] = []
        seen: set[str] = set()
        for path in self._metadata_paths():
            meta = self._read_metadata_path(path)
            if meta is None or meta.artifact_id in seen:
                continue
            seen.add(meta.artifact_id)
            results.append(meta)
        return results

    def _index_metadata(self, meta: ArtifactMetadata, *, conn: Optional[sqlite3.Connection] = None) -> None:
        d = meta.descriptor
        raw = meta.to_dict()
        row = (
            meta.artifact_id,
            meta.run_id,
            meta.blob_id,
            meta.content_type,
            int(meta.size_bytes),
            meta.created_at,
            d.semantic_kind,
            d.render_kind,
            d.modality,
            d.task,
            d.session_id,
            d.workflow_id,
            d.node_id,
            d.turn_id,
            d.classification_source,
            json.dumps(meta.tags or {}, ensure_ascii=False, sort_keys=True),
            json.dumps(meta.metadata or {}, ensure_ascii=False, sort_keys=True),
            json.dumps(d.to_dict(), ensure_ascii=False, sort_keys=True),
            json.dumps(meta.access.to_dict(), ensure_ascii=False, sort_keys=True),
            json.dumps(raw, ensure_ascii=False, sort_keys=True),
        )
        sql = """
            INSERT INTO artifact_catalog (
                artifact_id, run_id, blob_id, content_type, size_bytes, created_at,
                semantic_kind, render_kind, modality, task, session_id, workflow_id,
                node_id, turn_id, classification_source, tags_json, metadata_json,
                descriptor_json, access_json, raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(artifact_id) DO UPDATE SET
                run_id=excluded.run_id,
                blob_id=excluded.blob_id,
                content_type=excluded.content_type,
                size_bytes=excluded.size_bytes,
                created_at=excluded.created_at,
                semantic_kind=excluded.semantic_kind,
                render_kind=excluded.render_kind,
                modality=excluded.modality,
                task=excluded.task,
                session_id=excluded.session_id,
                workflow_id=excluded.workflow_id,
                node_id=excluded.node_id,
                turn_id=excluded.turn_id,
                classification_source=excluded.classification_source,
                tags_json=excluded.tags_json,
                metadata_json=excluded.metadata_json,
                descriptor_json=excluded.descriptor_json,
                access_json=excluded.access_json,
                raw_json=excluded.raw_json
        """
        if conn is not None:
            conn.execute(sql, row)
            return
        with self._connect_catalog() as c:
            c.execute(sql, row)

    def rebuild_catalog(self) -> int:
        """Rebuild the file-backed artifact catalog from metadata sidecars."""
        seen: set[str] = set()
        count = 0
        with self._lock:
            with self._connect_catalog() as conn:
                conn.execute("DELETE FROM artifact_catalog")
                for path in self._metadata_paths():
                    meta = self._read_metadata_path(path)
                    if meta is None or meta.artifact_id in seen:
                        continue
                    seen.add(meta.artifact_id)
                    self._index_metadata(meta, conn=conn)
                    count += 1
        return count

    def _rebuild_catalog_if_empty(self) -> None:
        if self._catalog_count() > 0:
            return
        if not self._metadata_paths():
            return
        self.rebuild_catalog()

    def _legacy_content_path(self, artifact_id: str) -> Path:
        validate_artifact_id(artifact_id)
        return self._artifacts_dir / f"{artifact_id}.bin"

    def _legacy_metadata_path(self, artifact_id: str) -> Path:
        validate_artifact_id(artifact_id)
        return self._artifacts_dir / f"{artifact_id}.meta"

    def _ref_metadata_path(self, artifact_id: str) -> Path:
        validate_artifact_id(artifact_id)
        return self._refs_dir / f"{artifact_id}.meta"

    def _blob_path(self, blob_id: str) -> Path:
        validate_artifact_id(blob_id)
        return self._blobs_dir / f"{blob_id}.bin"

    def _write_blob(self, *, blob_id: str, content: bytes) -> Path:
        path = self._blob_path(blob_id)
        if path.exists():
            return path

        tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with open(tmp, "wb") as f:
                f.write(content)
            tmp.replace(path)
        finally:
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
        return path

    def _content_path(self, artifact_id: str) -> Path:
        validate_artifact_id(artifact_id)
        meta = self.get_metadata(artifact_id)
        blob_id = getattr(meta, "blob_id", None) if meta is not None else None
        if isinstance(blob_id, str) and blob_id.strip():
            return self._blob_path(blob_id.strip())
        return self._legacy_content_path(artifact_id)

    def content_path(self, artifact_id: str) -> Optional[Path]:
        path = self._content_path(artifact_id)
        return path if path.exists() else None

    def _metadata_path(self, artifact_id: str) -> Path:
        validate_artifact_id(artifact_id)
        p = self._ref_metadata_path(artifact_id)
        if p.exists():
            return p
        return self._legacy_metadata_path(artifact_id)

    def _write_metadata(self, meta: ArtifactMetadata) -> None:
        metadata_path = self._ref_metadata_path(meta.artifact_id)
        tmp = metadata_path.with_name(f"{metadata_path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(meta.to_dict(), f, ensure_ascii=False, indent=2)
            tmp.replace(metadata_path)
        finally:
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
        self._index_metadata(meta)

    def store(
        self,
        content: bytes,
        *,
        content_type: str = "application/octet-stream",
        run_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        descriptor: Optional[Union[ArtifactDescriptor, Dict[str, Any]]] = None,
        artifact_id: Optional[str] = None,
    ) -> ArtifactMetadata:
        if artifact_id is None:
            artifact_id = compute_artifact_id(content, run_id=run_id)
        validate_artifact_id(artifact_id)
        blob_id = compute_blob_id(content)
        tags0 = _str_dict(tags or {})
        metadata0 = _json_safe_dict(metadata or {})
        descriptor0 = _normalize_descriptor(
            descriptor=descriptor,
            content_type=content_type,
            run_id=run_id,
            tags=tags0,
            media=_inspect_media(bytes(content), content_type=content_type),
        )

        metadata = ArtifactMetadata(
            artifact_id=artifact_id,
            blob_id=blob_id,
            content_type=content_type,
            size_bytes=len(content),
            created_at=utc_now_iso(),
            run_id=run_id,
            tags=tags0,
            metadata=metadata0,
            descriptor=descriptor0,
        )

        with self._lock:
            # Write blob bytes (deduped across runs)
            self._write_blob(blob_id=blob_id, content=content)
            self._write_metadata(metadata)

        return metadata

    def load(self, artifact_id: str) -> Optional[Artifact]:
        metadata_path = self._metadata_path(artifact_id)
        if not metadata_path.exists():
            return None

        metadata = self._read_metadata_path(metadata_path)
        if metadata is None:
            return None
        content_path = self._content_path(artifact_id)
        if not content_path.exists():
            return None

        with open(content_path, "rb") as f:
            content = f.read()
        return Artifact(metadata=metadata, content=content)

    def get_metadata(self, artifact_id: str) -> Optional[ArtifactMetadata]:
        validate_artifact_id(artifact_id)
        metadata_path = self._ref_metadata_path(artifact_id)
        if not metadata_path.exists():
            metadata_path = self._legacy_metadata_path(artifact_id)
            if not metadata_path.exists():
                return None

        return self._read_metadata_path(metadata_path)

    def update_metadata(
        self,
        artifact_id: str,
        *,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        descriptor: Optional[Union[ArtifactDescriptor, Dict[str, Any]]] = None,
        replace_tags: bool = False,
        replace_metadata: bool = False,
    ) -> Optional[ArtifactMetadata]:
        validate_artifact_id(artifact_id)
        with self._lock:
            meta = self.get_metadata(artifact_id)
            if meta is None:
                return None
            tags0 = _str_dict(tags or {})
            if tags is not None:
                meta.tags = tags0 if replace_tags else {**(meta.tags or {}), **tags0}
            metadata0 = _json_safe_dict(metadata or {})
            if metadata is not None:
                meta.metadata = metadata0 if replace_metadata else {**(meta.metadata or {}), **metadata0}
            if descriptor is not None or tags is not None:
                base = meta.descriptor.to_dict()
                if descriptor is not None:
                    base = _merge_descriptor_payload(base, descriptor)
                meta.descriptor = _normalize_descriptor(
                    descriptor=base,
                    content_type=meta.content_type,
                    run_id=meta.run_id,
                    tags=meta.tags,
                )
            self._write_metadata(meta)
            return meta

    def record_access(
        self,
        artifact_id: str,
        *,
        action: str = "access",
        actor_id: Optional[str] = None,
        session_id: Optional[str] = None,
        run_id: Optional[str] = None,
        at: Optional[str] = None,
    ) -> Optional[ArtifactMetadata]:
        validate_artifact_id(artifact_id)
        with self._lock:
            meta = self.get_metadata(artifact_id)
            if meta is None:
                return None
            meta.access.record(
                action=action,
                actor_id=actor_id,
                session_id=session_id,
                run_id=run_id,
                at=at,
            )
            self._write_metadata(meta)
            return meta

    def exists(self, artifact_id: str) -> bool:
        meta = self.get_metadata(artifact_id)
        if meta is None:
            return False
        return self._content_path(artifact_id).exists()

    def delete(self, artifact_id: str) -> bool:
        validate_artifact_id(artifact_id)
        metadata_path = self._ref_metadata_path(artifact_id)
        legacy_meta = self._legacy_metadata_path(artifact_id)
        legacy_content = self._legacy_content_path(artifact_id)

        deleted = False
        if metadata_path.exists():
            metadata_path.unlink()
            deleted = True
        if legacy_meta.exists():
            legacy_meta.unlink()
            deleted = True
        if legacy_content.exists():
            legacy_content.unlink()
            deleted = True

        try:
            with self._connect_catalog() as conn:
                conn.execute("DELETE FROM artifact_catalog WHERE artifact_id = ?", (str(artifact_id),))
        except Exception:
            pass
        return deleted

    def list_by_run(self, run_id: str) -> List[ArtifactMetadata]:
        return self.search(run_id=run_id, limit=0)

    def list_all(self, *, limit: int = 1000) -> List[ArtifactMetadata]:
        return self.search(limit=limit)

    def _catalog_filter_parts(
        self,
        *,
        run_id: Optional[str] = None,
        content_type: Optional[str] = None,
        semantic_kind: Optional[str] = None,
        render_kind: Optional[str] = None,
        modality: Optional[str] = None,
        session_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        node_id: Optional[str] = None,
        created_after: Optional[str] = None,
        created_before: Optional[str] = None,
    ) -> tuple[List[str], List[Any]]:
        clauses: List[str] = []
        params: List[Any] = []

        def _eq(column: str, value: Optional[str], *, lower: bool = False) -> None:
            text = str(value or "").strip()
            if not text:
                return
            clauses.append(f"{column} = ?")
            params.append(text.lower() if lower else text)

        _eq("run_id", run_id)
        _eq("content_type", content_type)
        _eq("semantic_kind", semantic_kind, lower=True)
        _eq("render_kind", render_kind, lower=True)
        _eq("modality", modality, lower=True)
        _eq("session_id", session_id)
        _eq("workflow_id", workflow_id)
        _eq("node_id", node_id)
        if created_after:
            clauses.append("created_at >= ?")
            params.append(str(created_after))
        if created_before:
            clauses.append("created_at <= ?")
            params.append(str(created_before))
        return clauses, params

    def search(
        self,
        *,
        run_id: Optional[str] = None,
        content_type: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        semantic_kind: Optional[str] = None,
        render_kind: Optional[str] = None,
        modality: Optional[str] = None,
        session_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        node_id: Optional[str] = None,
        created_after: Optional[str] = None,
        created_before: Optional[str] = None,
        offset: int = 0,
        order: str = "desc",
        limit: int = 1000,
    ) -> List[ArtifactMetadata]:
        clauses, params = self._catalog_filter_parts(
            run_id=run_id,
            content_type=content_type,
            semantic_kind=semantic_kind,
            render_kind=render_kind,
            modality=modality,
            session_id=session_id,
            workflow_id=workflow_id,
            node_id=node_id,
            created_after=created_after,
            created_before=created_before,
        )
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        direction = "ASC" if str(order or "desc").strip().lower() == "asc" else "DESC"
        sql = f"SELECT raw_json FROM artifact_catalog {where} ORDER BY created_at {direction}, artifact_id {direction}"
        tag_filter = _str_dict(tags or {})
        use_sql_limit = not tag_filter and int(limit) > 0
        if use_sql_limit:
            sql += " LIMIT ? OFFSET ?"
            params.extend([int(limit), max(0, int(offset or 0))])
        rows: List[sqlite3.Row]
        try:
            with self._connect_catalog() as conn:
                rows = list(conn.execute(sql, tuple(params)).fetchall())
        except Exception:
            metas = self._scan_metadata_files()
            if run_id is not None:
                metas = [m for m in metas if m.run_id == run_id]
            if content_type is not None:
                metas = [m for m in metas if m.content_type == content_type]
            if tags:
                tag_filter0 = _str_dict(tags)
                metas = [m for m in metas if all((m.tags or {}).get(k) == v for k, v in tag_filter0.items())]
            if semantic_kind:
                metas = [m for m in metas if m.descriptor.semantic_kind == str(semantic_kind).strip().lower()]
            if render_kind:
                metas = [m for m in metas if m.descriptor.render_kind == str(render_kind).strip().lower()]
            if modality:
                metas = [m for m in metas if m.descriptor.modality == str(modality).strip().lower()]
            if session_id:
                metas = [m for m in metas if m.descriptor.session_id == str(session_id).strip()]
            if workflow_id:
                metas = [m for m in metas if m.descriptor.workflow_id == str(workflow_id).strip()]
            if node_id:
                metas = [m for m in metas if m.descriptor.node_id == str(node_id).strip()]
            if created_after:
                metas = [m for m in metas if str(m.created_at or "") >= str(created_after)]
            if created_before:
                metas = [m for m in metas if str(m.created_at or "") <= str(created_before)]
            reverse = str(order or "desc").strip().lower() != "asc"
            metas.sort(key=lambda m: (str(m.created_at or ""), str(m.artifact_id or "")), reverse=reverse)
            start = max(0, int(offset or 0))
            metas = metas[start:]
            if int(limit) > 0:
                metas = metas[: int(limit)]
            return metas

        metas: List[ArtifactMetadata] = []
        for row in rows:
            try:
                meta = ArtifactMetadata.from_dict(json.loads(str(row["raw_json"])))
            except Exception:
                continue
            if tag_filter and not all((meta.tags or {}).get(k) == v for k, v in tag_filter.items()):
                continue
            metas.append(meta)
        if tag_filter:
            start = max(0, int(offset or 0))
            metas = metas[start:]
            if int(limit) > 0:
                metas = metas[: int(limit)]
        return metas

    def count(self, **filters: Any) -> int:
        if filters.get("tags"):
            return super().count(**filters)
        clauses, params = self._catalog_filter_parts(
            run_id=filters.get("run_id"),
            content_type=filters.get("content_type"),
            semantic_kind=filters.get("semantic_kind"),
            render_kind=filters.get("render_kind"),
            modality=filters.get("modality"),
            session_id=filters.get("session_id"),
            workflow_id=filters.get("workflow_id"),
            node_id=filters.get("node_id"),
            created_after=filters.get("created_after"),
            created_before=filters.get("created_before"),
        )
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        try:
            with self._connect_catalog() as conn:
                row = conn.execute(f"SELECT COUNT(*) AS n FROM artifact_catalog {where}", tuple(params)).fetchone()
            return int(row["n"] if row is not None else 0)
        except Exception:
            return super().count(**filters)

    def facet_counts(self, field: str, **filters: Any) -> Dict[str, int]:
        field0 = str(field or "").strip()
        columns = {
            "artifact_id": "artifact_id",
            "blob_id": "blob_id",
            "content_type": "content_type",
            "run_id": "run_id",
            "created_at": "created_at",
            "semantic_kind": "semantic_kind",
            "render_kind": "render_kind",
            "modality": "modality",
            "task": "task",
            "classification_source": "classification_source",
            "session_id": "session_id",
            "workflow_id": "workflow_id",
            "node_id": "node_id",
            "turn_id": "turn_id",
        }
        column = columns.get(field0)
        if column is None or filters.get("tags"):
            return super().facet_counts(field, **filters)
        clauses, params = self._catalog_filter_parts(
            run_id=filters.get("run_id"),
            content_type=filters.get("content_type"),
            semantic_kind=filters.get("semantic_kind"),
            render_kind=filters.get("render_kind"),
            modality=filters.get("modality"),
            session_id=filters.get("session_id"),
            workflow_id=filters.get("workflow_id"),
            node_id=filters.get("node_id"),
            created_after=filters.get("created_after"),
            created_before=filters.get("created_before"),
        )
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = (
            f"SELECT COALESCE(NULLIF({column}, ''), '(none)') AS value, COUNT(*) AS n "
            f"FROM artifact_catalog {where} GROUP BY value ORDER BY n DESC, value ASC"
        )
        try:
            with self._connect_catalog() as conn:
                rows = conn.execute(sql, tuple(params)).fetchall()
            return {str(row["value"]): int(row["n"]) for row in rows}
        except Exception:
            return super().facet_counts(field, **filters)

    def stats(self, *, facets: Optional[List[str]] = None, **filters: Any) -> Dict[str, Any]:
        if filters.get("tags"):
            return super().stats(facets=facets, **filters)
        filters0 = dict(filters or {})
        filters0.pop("limit", None)
        filters0.pop("offset", None)
        filters0.pop("order", None)
        clauses, params = self._catalog_filter_parts(
            run_id=filters0.get("run_id"),
            content_type=filters0.get("content_type"),
            semantic_kind=filters0.get("semantic_kind"),
            render_kind=filters0.get("render_kind"),
            modality=filters0.get("modality"),
            session_id=filters0.get("session_id"),
            workflow_id=filters0.get("workflow_id"),
            node_id=filters0.get("node_id"),
            created_after=filters0.get("created_after"),
            created_before=filters0.get("created_before"),
        )
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        try:
            with self._connect_catalog() as conn:
                row = conn.execute(
                    f"SELECT COUNT(*) AS n, COALESCE(SUM(size_bytes), 0) AS bytes_total FROM artifact_catalog {where}",
                    tuple(params),
                ).fetchone()
            total = int(row["n"] if row is not None else 0)
            total_bytes = int(row["bytes_total"] if row is not None else 0)
        except Exception:
            return super().stats(facets=facets, **filters0)
        return {
            "total": total,
            "total_bytes": total_bytes,
            "facets": {field: self.facet_counts(field, **filters0) for field in (facets or [])},
            "filters": filters0,
            "exact": True,
            "source": "artifact_catalog",
        }

    def gc(self, *, dry_run: bool = True) -> Dict[str, Any]:
        """Garbage collect unreferenced blobs.

        Notes:
        - This only applies to the v1 `artifacts/blobs` layout.
        - Safe-by-default: `dry_run=True` returns the plan without deleting.
        """

        report: Dict[str, Any] = {
            "dry_run": bool(dry_run),
            "blobs_total": 0,
            "blobs_referenced": 0,
            "blobs_deleted": 0,
            "bytes_reclaimed": 0,
            "errors": [],
        }

        referenced: set[str] = set()
        for meta in self.list_all(limit=1_000_000):
            blob_id = getattr(meta, "blob_id", None)
            if isinstance(blob_id, str) and blob_id.strip():
                referenced.add(blob_id.strip())

        report["blobs_referenced"] = len(referenced)

        blobs = list(self._blobs_dir.glob("*.bin"))
        report["blobs_total"] = len(blobs)

        for p in blobs:
            blob_id = p.stem
            if blob_id in referenced:
                continue
            try:
                size = p.stat().st_size
            except Exception:
                size = 0
            if not dry_run:
                try:
                    p.unlink()
                except Exception as e:
                    report["errors"].append({"blob_id": blob_id, "error": str(e)})
                    continue
            report["blobs_deleted"] += 1
            report["bytes_reclaimed"] += int(size)

        return report


# Artifact reference helpers for use in RunState.vars

def artifact_ref(artifact_id: str) -> Dict[str, str]:
    """Create an artifact reference for storing in vars.

    Usage:
        metadata = artifact_store.store_json(large_data)
        run.vars["result"] = artifact_ref(metadata.artifact_id)
    """
    return {"$artifact": artifact_id}


def is_artifact_ref(value: Any) -> bool:
    """Check if a value is an artifact reference."""
    return isinstance(value, dict) and "$artifact" in value


def get_artifact_id(ref: Dict[str, str]) -> str:
    """Extract artifact ID from a reference."""
    return ref["$artifact"]


def resolve_artifact(ref: Dict[str, str], store: ArtifactStore) -> Optional[Artifact]:
    """Resolve an artifact reference to its content."""
    if not is_artifact_ref(ref):
        return None
    return store.load(get_artifact_id(ref))
