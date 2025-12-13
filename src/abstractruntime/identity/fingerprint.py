"""abstractruntime.identity.fingerprint

AI fingerprint / provenance (v0.1 - data model + hashing only).

Important:
- This does NOT implement cryptographic signatures yet (no non-forgeability).
- It provides a stable, deterministic *identifier* given stable inputs.

Backlog item will define:
- keypair-based identity (public key) and signing of ledger chains.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


def _canonical_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ActorFingerprint:
    """A stable identifier for an actor (agent/service/human).

    This is intentionally minimal. For accountability you typically want:
    - stable actor id
    - metadata about the owner/org
    - (future) signature key to make logs tamper-evident
    """

    actor_id: str
    kind: str  # "agent" | "human" | "service"
    display_name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def from_metadata(cls, *, kind: str, display_name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> "ActorFingerprint":
        payload = {
            "kind": kind,
            "display_name": display_name,
            "metadata": metadata or {},
        }
        actor_id = f"ar_{sha256_hex(_canonical_json(payload))[:24]}"
        return cls(actor_id=actor_id, kind=kind, display_name=display_name, metadata=metadata or {})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


