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
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


# Valid artifact ID pattern: alphanumeric, hyphens, underscores
_ARTIFACT_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ArtifactMetadata:
    """Metadata about a stored artifact."""

    artifact_id: str
    content_type: str  # MIME type or semantic type
    size_bytes: int
    created_at: str
    run_id: Optional[str] = None  # Optional association with a run
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArtifactMetadata":
        return cls(
            artifact_id=data["artifact_id"],
            content_type=data["content_type"],
            size_bytes=data["size_bytes"],
            created_at=data["created_at"],
            run_id=data.get("run_id"),
            tags=data.get("tags") or {},
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

    If `run_id` is provided, the id is *namespaced to that run* to avoid cross-run
    collisions when using a shared `FileArtifactStore(base_dir)` and to preserve
    correct `list_by_run(...)` / purge-by-run semantics.
    """
    h = hashlib.sha256()
    if run_id is not None:
        rid = str(run_id).strip()
        if rid:
            h.update(rid.encode("utf-8"))
            h.update(b"\0")
    h.update(content)
    return h.hexdigest()[:32]


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
        artifact_id: Optional[str] = None,
    ) -> ArtifactMetadata:
        """Store an artifact and return its metadata.

        Args:
            content: The artifact content as bytes.
            content_type: MIME type or semantic type.
            run_id: Optional run ID to associate with.
            tags: Optional key-value tags.
            artifact_id: Optional explicit ID (defaults to content hash).

        Returns:
            ArtifactMetadata with the artifact_id.
        """
        ...

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
            limit: Maximum number of artifacts to return.

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
        limit: int = 1000,
    ) -> List[ArtifactMetadata]:
        """Filter artifacts by simple metadata fields.

        This is intentionally a *metadata filter*, not semantic search. Semantic/embedding
        retrieval belongs in AbstractMemory or higher-level components.
        """
        if run_id is None:
            candidates = list(self.list_all(limit=limit))
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

        candidates.sort(key=lambda m: m.created_at, reverse=True)
        return candidates[:limit]

    # Convenience methods

    def store_text(
        self,
        text: str,
        *,
        content_type: str = "text/plain",
        encoding: str = "utf-8",
        run_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> ArtifactMetadata:
        """Store text content."""
        return self.store(
            text.encode(encoding),
            content_type=content_type,
            run_id=run_id,
            tags=tags,
        )

    def store_json(
        self,
        data: Any,
        *,
        run_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> ArtifactMetadata:
        """Store JSON-serializable data."""
        content = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return self.store(
            content,
            content_type="application/json",
            run_id=run_id,
            tags=tags,
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

    def store(
        self,
        content: bytes,
        *,
        content_type: str = "application/octet-stream",
        run_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        artifact_id: Optional[str] = None,
    ) -> ArtifactMetadata:
        if artifact_id is None:
            artifact_id = compute_artifact_id(content, run_id=run_id)

        metadata = ArtifactMetadata(
            artifact_id=artifact_id,
            content_type=content_type,
            size_bytes=len(content),
            created_at=utc_now_iso(),
            run_id=run_id,
            tags=tags or {},
        )

        self._artifacts[artifact_id] = Artifact(metadata=metadata, content=content)
        return metadata

    def load(self, artifact_id: str) -> Optional[Artifact]:
        return self._artifacts.get(artifact_id)

    def get_metadata(self, artifact_id: str) -> Optional[ArtifactMetadata]:
        artifact = self._artifacts.get(artifact_id)
        if artifact is None:
            return None
        return artifact.metadata

    def exists(self, artifact_id: str) -> bool:
        return artifact_id in self._artifacts

    def delete(self, artifact_id: str) -> bool:
        if artifact_id in self._artifacts:
            del self._artifacts[artifact_id]
            return True
        return False

    def list_by_run(self, run_id: str) -> List[ArtifactMetadata]:
        return [
            a.metadata
            for a in self._artifacts.values()
            if a.metadata.run_id == run_id
        ]

    def list_all(self, *, limit: int = 1000) -> List[ArtifactMetadata]:
        results = [a.metadata for a in self._artifacts.values()]
        # Sort by created_at descending
        results.sort(key=lambda m: m.created_at, reverse=True)
        return results[:limit]


class FileArtifactStore(ArtifactStore):
    """File-based artifact store.

    Directory structure:
        base_dir/
            artifacts/
                {artifact_id}.bin     # content
                {artifact_id}.meta    # metadata JSON
    """

    def __init__(self, base_dir: Union[str, Path]) -> None:
        self._base = Path(base_dir)
        self._artifacts_dir = self._base / "artifacts"
        self._artifacts_dir.mkdir(parents=True, exist_ok=True)

    def _content_path(self, artifact_id: str) -> Path:
        validate_artifact_id(artifact_id)
        return self._artifacts_dir / f"{artifact_id}.bin"

    def _metadata_path(self, artifact_id: str) -> Path:
        validate_artifact_id(artifact_id)
        return self._artifacts_dir / f"{artifact_id}.meta"

    def store(
        self,
        content: bytes,
        *,
        content_type: str = "application/octet-stream",
        run_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        artifact_id: Optional[str] = None,
    ) -> ArtifactMetadata:
        if artifact_id is None:
            artifact_id = compute_artifact_id(content, run_id=run_id)

        metadata = ArtifactMetadata(
            artifact_id=artifact_id,
            content_type=content_type,
            size_bytes=len(content),
            created_at=utc_now_iso(),
            run_id=run_id,
            tags=tags or {},
        )

        # Write content
        content_path = self._content_path(artifact_id)
        with open(content_path, "wb") as f:
            f.write(content)

        # Write metadata
        metadata_path = self._metadata_path(artifact_id)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata.to_dict(), f, ensure_ascii=False, indent=2)

        return metadata

    def load(self, artifact_id: str) -> Optional[Artifact]:
        content_path = self._content_path(artifact_id)
        metadata_path = self._metadata_path(artifact_id)

        if not content_path.exists() or not metadata_path.exists():
            return None

        with open(content_path, "rb") as f:
            content = f.read()

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata_dict = json.load(f)

        metadata = ArtifactMetadata.from_dict(metadata_dict)
        return Artifact(metadata=metadata, content=content)

    def get_metadata(self, artifact_id: str) -> Optional[ArtifactMetadata]:
        metadata_path = self._metadata_path(artifact_id)

        if not metadata_path.exists():
            return None

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata_dict = json.load(f)

        return ArtifactMetadata.from_dict(metadata_dict)

    def exists(self, artifact_id: str) -> bool:
        return self._content_path(artifact_id).exists()

    def delete(self, artifact_id: str) -> bool:
        content_path = self._content_path(artifact_id)
        metadata_path = self._metadata_path(artifact_id)

        deleted = False
        if content_path.exists():
            content_path.unlink()
            deleted = True
        if metadata_path.exists():
            metadata_path.unlink()
            deleted = True

        return deleted

    def list_by_run(self, run_id: str) -> List[ArtifactMetadata]:
        results = []
        for metadata_path in self._artifacts_dir.glob("*.meta"):
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata_dict = json.load(f)
                if metadata_dict.get("run_id") == run_id:
                    results.append(ArtifactMetadata.from_dict(metadata_dict))
            except (json.JSONDecodeError, IOError):
                continue
        return results

    def list_all(self, *, limit: int = 1000) -> List[ArtifactMetadata]:
        results = []
        for metadata_path in self._artifacts_dir.glob("*.meta"):
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata_dict = json.load(f)
                results.append(ArtifactMetadata.from_dict(metadata_dict))
            except (json.JSONDecodeError, IOError):
                continue
        # Sort by created_at descending
        results.sort(key=lambda m: m.created_at, reverse=True)
        return results[:limit]


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
