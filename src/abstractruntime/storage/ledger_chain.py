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
import threading
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
    """LedgerStore decorator adding a SHA-256 hash chain.

    FORK SAFETY (2026-07-10 adversarial finding, maintainer-driven lease
    review): the original implementation cached the chain head PER PROCESS,
    so two handles over one persisted ledger (two processes, or two store
    instances in one process) each computed `prev_hash` from their own
    stale head — the inner store serialized both inserts and the chain
    FORKED permanently (`verify` reports prev_hash_mismatch forever; on
    never-purge stores like the diary book there is no repair). The
    per-directory writer lease makes that unreachable in normal topology;
    the chain now refuses to fork even without it:

    - The head is re-read from the PERSISTED tail on every append (cheap
      `last_record()` fast path when the inner store provides it; full
      `list()` tail otherwise). No trust in process-local state.
    - When the inner store offers `append_chained` (SqliteLedgerStore),
      head-read + hash + insert run inside ONE write transaction
      (`BEGIN IMMEDIATE`) — cross-process fork-free BY CONSTRUCTION.
    - A per-instance lock serializes same-instance threaded appenders
      (two gateway threads could previously fork the chain in-process).
    """

    def __init__(self, inner: LedgerStore):
        self._inner = inner
        self._append_lock = threading.Lock()

    def _persisted_head(self, run_id: str) -> Optional[str]:
        """The chain head as PERSISTED — never a process-local cache."""
        last_record = getattr(self._inner, "last_record", None)
        if callable(last_record):
            tail = last_record(run_id)
            return tail.get("record_hash") if isinstance(tail, dict) else None
        records = self._inner.list(run_id)
        if not records:
            return None
        return records[-1].get("record_hash")

    def append(self, record: StepRecord) -> None:
        with self._append_lock:
            append_chained = getattr(self._inner, "append_chained", None)
            if callable(append_chained):
                # Transaction-internal derivation: the store reads the head
                # under its write lock; the hash policy stays ours.
                append_chained(
                    record,
                    lambda record_dict, prev: compute_record_hash(
                        record=record_dict, prev_hash=prev
                    ),
                )
                return
            prev = self._persisted_head(record.run_id)
            record.prev_hash = prev
            record_dict = asdict(record)
            record.record_hash = compute_record_hash(record=record_dict, prev_hash=prev)
            self._inner.append(record)

    def list(self, run_id: str) -> List[Dict[str, Any]]:
        return self._inner.list(run_id)

    def delete(self, run_id: str) -> int:
        fn = getattr(self._inner, "delete", None)
        if not callable(fn):
            raise NotImplementedError("Inner LedgerStore does not support delete")
        rid = str(run_id or "").strip()
        return int(fn(rid))


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
