"""abstractruntime.storage.ledger_chain

Tamper-evident provenance for the execution ledger.

This module provides:
- A `HashChainedLedgerStore` decorator that wraps any `LedgerStore` and injects
  `prev_hash` + `record_hash` into each appended `StepRecord`.
- A `verify_ledger_chain()` utility to validate the chain.

Important scope boundary:
- This is **tamper-evident**, not tamper-proof.
- Cryptographic signatures (non-forgeability) are intentionally out of scope for v0.1.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from .base import LedgerStore
from ..core.models import StepRecord


def _canonical_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_record_hash(*, record: Dict[str, Any], prev_hash: Optional[str]) -> str:
    """Compute record hash from a JSON dict.

    Rules:
    - `prev_hash` is included in the hashed payload.
    - `record_hash` and `signature` fields are excluded (to avoid recursion).
    """

    clean = dict(record)
    clean.pop("record_hash", None)
    clean.pop("signature", None)

    clean["prev_hash"] = prev_hash
    return _sha256_hex(_canonical_json(clean))


class HashChainedLedgerStore(LedgerStore):
    """LedgerStore decorator adding a SHA-256 hash chain."""

    def __init__(self, inner: LedgerStore):
        self._inner = inner
        self._head_by_run: Dict[str, Optional[str]] = {}

    def _get_head(self, run_id: str) -> Optional[str]:
        if run_id in self._head_by_run:
            return self._head_by_run[run_id]

        # Best-effort bootstrap for process restarts.
        records = self._inner.list(run_id)
        if not records:
            self._head_by_run[run_id] = None
            return None

        last = records[-1]
        head = last.get("record_hash")
        self._head_by_run[run_id] = head
        return head

    def append(self, record: StepRecord) -> None:
        prev = self._get_head(record.run_id)

        # Compute hash from record dict + prev hash.
        record.prev_hash = prev
        record_dict = asdict(record)
        record_hash = compute_record_hash(record=record_dict, prev_hash=prev)
        record.record_hash = record_hash

        self._inner.append(record)
        self._head_by_run[record.run_id] = record_hash

    def list(self, run_id: str) -> List[Dict[str, Any]]:
        return self._inner.list(run_id)


def verify_ledger_chain(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Verify a list of stored ledger records.

    Returns a structured report with the first failing index and error details.
    """

    report: Dict[str, Any] = {
        "ok": True,
        "count": len(records),
        "errors": [],
        "first_bad_index": None,
        "head_hash": records[-1].get("record_hash") if records else None,
        "computed_head_hash": None,
    }

    prev: Optional[str] = None
    computed_head: Optional[str] = None

    for i, r in enumerate(records):
        expected_prev = prev
        actual_prev = r.get("prev_hash")

        if actual_prev != expected_prev:
            report["ok"] = False
            report["first_bad_index"] = report["first_bad_index"] or i
            report["errors"].append(
                {
                    "index": i,
                    "type": "prev_hash_mismatch",
                    "expected_prev_hash": expected_prev,
                    "actual_prev_hash": actual_prev,
                }
            )

        stored_hash = r.get("record_hash")
        if not stored_hash:
            report["ok"] = False
            report["first_bad_index"] = report["first_bad_index"] or i
            report["errors"].append(
                {
                    "index": i,
                    "type": "missing_record_hash",
                }
            )
            # Cannot continue computing chain reliably
            break

        computed_hash = compute_record_hash(record=r, prev_hash=actual_prev)
        if computed_hash != stored_hash:
            report["ok"] = False
            report["first_bad_index"] = report["first_bad_index"] or i
            report["errors"].append(
                {
                    "index": i,
                    "type": "record_hash_mismatch",
                    "stored": stored_hash,
                    "computed": computed_hash,
                }
            )

        prev = stored_hash
        computed_head = stored_hash

    report["computed_head_hash"] = computed_head
    return report

