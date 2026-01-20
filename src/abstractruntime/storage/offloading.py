"""abstractruntime.storage.offloading

Storage decorators for keeping durable JSON payloads small by offloading large values
to the ArtifactStore (store refs inline).

This is intentionally opt-in and layered at the persistence boundary (RunStore/LedgerStore)
so node/tool code does not need to remember to offload manually.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional, Tuple

from .artifacts import ArtifactStore, artifact_ref, is_artifact_ref
from .base import LedgerStore, RunStore
from ..core.models import RunState, RunStatus, StepRecord

DEFAULT_MAX_INLINE_BYTES = 256 * 1024


def _default_max_inline_bytes() -> int:
    raw = str(os.getenv("ABSTRACTRUNTIME_MAX_INLINE_BYTES", "")).strip()
    if not raw:
        return DEFAULT_MAX_INLINE_BYTES
    try:
        return int(raw)
    except Exception:
        return DEFAULT_MAX_INLINE_BYTES


def _json_dumps_bytes(value: Any) -> Optional[bytes]:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    except Exception:
        return None


def offload_large_values(
    value: Any,
    *,
    artifact_store: ArtifactStore,
    run_id: str,
    max_inline_bytes: int,
    base_tags: Optional[Dict[str, str]] = None,
    root_path: str = "",
    allow_offload: Optional[Callable[[str, Any], bool]] = None,
    allow_root_replace: bool = False,
) -> Any:
    """Replace large JSON leaves/subtrees with ArtifactStore-backed refs.

    This is best-effort: values that cannot be serialized are either left untouched
    (for non-bytes) or offloaded (bytes/bytearray) to preserve JSON-safe persistence.
    """

    if max_inline_bytes <= 0:
        return value

    tags0 = dict(base_tags or {})

    def _can_offload(path: str, v: Any) -> bool:
        if allow_offload is None:
            return True
        try:
            return bool(allow_offload(path, v))
        except Exception:
            return False

    stack: set[int] = set()

    def _offload_bytes(*, content: bytes, content_type: str, path: str, kind: str) -> Dict[str, str]:
        tags = dict(tags0)
        tags["path"] = str(path)
        tags["kind"] = str(kind)
        meta = artifact_store.store(content, content_type=content_type, run_id=run_id, tags=tags)
        return artifact_ref(str(getattr(meta, "artifact_id", "") or ""))

    def _walk(cur: Any, *, path: str, root: bool) -> Tuple[Any, bool]:
        if cur is None:
            return None, False
        if is_artifact_ref(cur):
            return cur, False

        if isinstance(cur, (bytes, bytearray)):
            # Bytes are not JSON-serializable; offload unconditionally.
            ref = _offload_bytes(content=bytes(cur), content_type="application/octet-stream", path=path, kind="bytes")
            return ref, True

        if isinstance(cur, str):
            n = len(cur.encode("utf-8"))
            if n > max_inline_bytes and _can_offload(path, cur) and (allow_root_replace or not root):
                ref = _offload_bytes(content=cur.encode("utf-8"), content_type="text/plain", path=path, kind="text")
                return ref, True
            return cur, False

        if isinstance(cur, dict):
            oid = id(cur)
            if oid in stack:
                raise ValueError(f"Cycle detected while offloading at {path}")
            stack.add(oid)
            try:
                changed = False
                out: Dict[str, Any] = {}
                for k, v in cur.items():
                    key = str(k)
                    child_path = f"{path}.{key}" if path else key
                    new_v, ch = _walk(v, path=child_path, root=False)
                    if ch:
                        changed = True
                    out[key] = new_v

                # If still large, optionally offload the whole subtree (shape changes).
                if _can_offload(path, out) and (allow_root_replace or not root):
                    payload = _json_dumps_bytes(out)
                    if payload is not None and len(payload) > max_inline_bytes:
                        ref = _offload_bytes(
                            content=payload,
                            content_type="application/json",
                            path=path,
                            kind="json",
                        )
                        return ref, True

                return (out, True) if changed else (cur, False)
            finally:
                stack.discard(oid)

        if isinstance(cur, list):
            oid = id(cur)
            if oid in stack:
                raise ValueError(f"Cycle detected while offloading at {path}")
            stack.add(oid)
            try:
                changed = False
                out_list: List[Any] = []
                for i, item in enumerate(cur):
                    child_path = f"{path}[{i}]" if path else f"[{i}]"
                    new_item, ch = _walk(item, path=child_path, root=False)
                    if ch:
                        changed = True
                    out_list.append(new_item)

                if _can_offload(path, out_list) and (allow_root_replace or not root):
                    payload = _json_dumps_bytes(out_list)
                    if payload is not None and len(payload) > max_inline_bytes:
                        ref = _offload_bytes(
                            content=payload,
                            content_type="application/json",
                            path=path,
                            kind="json",
                        )
                        return ref, True

                return (out_list, True) if changed else (cur, False)
            finally:
                stack.discard(oid)

        return cur, False

    start_path = str(root_path or "").strip()
    out, _ = _walk(value, path=start_path, root=True)
    return out


def _offload_run_state(
    run: RunState,
    *,
    artifact_store: ArtifactStore,
    max_inline_bytes: int,
) -> RunState:
    """Create a persisted RunState copy with oversized internal payloads offloaded."""

    if max_inline_bytes <= 0:
        return run

    rid = str(run.run_id or "").strip()
    if not rid:
        return run

    # Safety: run.vars contains execution-critical state (e.g. VisualFlow persisted node_outputs).
    # Offloading those values during a non-terminal run can break crash recovery if a downstream
    # node expects the original value (e.g. large strings) after a restart.
    #
    # We therefore only offload run-owned/private vars once the run is terminal. Ledger offloading
    # remains always-on because ledger records are not used for execution.
    terminal = run.status in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED)

    # Only offload runtime-owned/private namespaces in vars to avoid breaking user semantics.
    vars_obj: Dict[str, Any] = dict(run.vars or {})

    def _allow_vars(path: str, v: Any) -> bool:
        # Root namespaces must remain dict-shaped; allow offloading only *within* private keys.
        # e.g. vars._temp.<...>, vars._runtime.<...>, vars._last_output.<...>
        p = str(path or "")
        if p == "vars":
            return False
        if p.startswith("vars._") and p.count(".") >= 2:
            return True
        return False

    # Apply per-private-namespace so we preserve the root dict objects.
    changed = False
    if terminal:
        for k, v in list(vars_obj.items()):
            key = str(k)
            if not key.startswith("_"):
                continue
            root_path = f"vars.{key}"
            try:
                new_v = offload_large_values(
                    v,
                    artifact_store=artifact_store,
                    run_id=rid,
                    max_inline_bytes=max_inline_bytes,
                    base_tags={"source": "run_store_offload"},
                    root_path=root_path,
                    allow_offload=_allow_vars,
                    allow_root_replace=False,
                )
            except Exception:
                new_v = v
            if new_v is not v:
                vars_obj[key] = new_v
                changed = True

    output = run.output
    output_changed = False
    if terminal and output is not None:
        def _allow_output(path: str, v: Any) -> bool:
            # Output is terminal-facing; safe to offload anywhere, including root if needed.
            return True

        try:
            new_out = offload_large_values(
                output,
                artifact_store=artifact_store,
                run_id=rid,
                max_inline_bytes=max_inline_bytes,
                base_tags={"source": "run_output_offload"},
                root_path="output",
                allow_offload=_allow_output,
                allow_root_replace=True,
            )
        except Exception:
            new_out = output
        if new_out is not output:
            output = new_out  # type: ignore[assignment]
            output_changed = True

    if not changed and not output_changed:
        return run

    return replace(run, vars=vars_obj, output=output)


def _offload_step_record(
    record: StepRecord,
    *,
    artifact_store: ArtifactStore,
    max_inline_bytes: int,
) -> StepRecord:
    if max_inline_bytes <= 0:
        return record

    rid = str(record.run_id or "").strip()
    if not rid:
        return record

    changed = False
    eff = record.effect
    res = record.result

    def _allow(path: str, v: Any) -> bool:
        # Ledger records are observability/provenance: ok to offload nested payloads.
        # Preserve effect/result object shape by disallowing root replacement.
        return True

    if eff is not None:
        try:
            new_eff = offload_large_values(
                eff,
                artifact_store=artifact_store,
                run_id=rid,
                max_inline_bytes=max_inline_bytes,
                base_tags={"source": "ledger_effect_offload"},
                root_path="ledger.effect",
                allow_offload=_allow,
                allow_root_replace=False,
            )
        except Exception:
            new_eff = eff
        if new_eff is not eff:
            eff = new_eff
            changed = True

    if res is not None:
        try:
            new_res = offload_large_values(
                res,
                artifact_store=artifact_store,
                run_id=rid,
                max_inline_bytes=max_inline_bytes,
                base_tags={"source": "ledger_result_offload"},
                root_path="ledger.result",
                allow_offload=_allow,
                allow_root_replace=False,
            )
        except Exception:
            new_res = res
        if new_res is not res:
            res = new_res
            changed = True

    return replace(record, effect=eff, result=res) if changed else record


class OffloadingRunStore(RunStore):
    """RunStore decorator that offloads oversized payloads to the ArtifactStore."""

    def __init__(
        self,
        inner: RunStore,
        *,
        artifact_store: ArtifactStore,
        max_inline_bytes: Optional[int] = None,
    ) -> None:
        self._inner = inner
        self._artifact_store = artifact_store
        self._max_inline_bytes = _default_max_inline_bytes() if max_inline_bytes is None else int(max_inline_bytes)

    @property
    def inner(self) -> RunStore:
        return self._inner

    def save(self, run: RunState) -> None:
        persisted = _offload_run_state(run, artifact_store=self._artifact_store, max_inline_bytes=self._max_inline_bytes)
        self._inner.save(persisted)

    def load(self, run_id: str) -> Optional[RunState]:
        return self._inner.load(run_id)

    # --- QueryableRunStore passthrough (best-effort) ---

    def list_runs(self, *, status=None, wait_reason=None, workflow_id=None, limit: int = 100):  # type: ignore[override]
        fn = getattr(self._inner, "list_runs", None)
        if not callable(fn):
            raise NotImplementedError("Inner RunStore does not support list_runs")
        return fn(status=status, wait_reason=wait_reason, workflow_id=workflow_id, limit=limit)

    def list_run_index(self, *, status=None, workflow_id=None, session_id=None, root_only: bool = False, limit: int = 100):  # type: ignore[override]
        fn = getattr(self._inner, "list_run_index", None)
        if not callable(fn):
            raise NotImplementedError("Inner RunStore does not support list_run_index")
        return fn(status=status, workflow_id=workflow_id, session_id=session_id, root_only=root_only, limit=limit)

    def list_due_wait_until(self, *, now_iso: str, limit: int = 100):  # type: ignore[override]
        fn = getattr(self._inner, "list_due_wait_until", None)
        if not callable(fn):
            raise NotImplementedError("Inner RunStore does not support list_due_wait_until")
        return fn(now_iso=now_iso, limit=limit)

    def list_children(self, *, parent_run_id: str, status=None):  # type: ignore[override]
        fn = getattr(self._inner, "list_children", None)
        if not callable(fn):
            raise NotImplementedError("Inner RunStore does not support list_children")
        return fn(parent_run_id=parent_run_id, status=status)


class OffloadingLedgerStore(LedgerStore):
    """LedgerStore decorator that offloads oversized effect/result payloads to the ArtifactStore."""

    def __init__(
        self,
        inner: LedgerStore,
        *,
        artifact_store: ArtifactStore,
        max_inline_bytes: Optional[int] = None,
    ) -> None:
        self._inner = inner
        self._artifact_store = artifact_store
        self._max_inline_bytes = _default_max_inline_bytes() if max_inline_bytes is None else int(max_inline_bytes)

    @property
    def inner(self) -> LedgerStore:
        return self._inner

    def append(self, record: StepRecord) -> None:
        persisted = _offload_step_record(record, artifact_store=self._artifact_store, max_inline_bytes=self._max_inline_bytes)
        self._inner.append(persisted)

    def list(self, run_id: str) -> List[Dict[str, Any]]:
        return self._inner.list(run_id)

    def subscribe(self, callback, *, run_id: Optional[str] = None):  # type: ignore[override]
        fn = getattr(self._inner, "subscribe", None)
        if not callable(fn):
            raise RuntimeError("Inner LedgerStore does not support subscribe()")
        return fn(callback, run_id=run_id)

    def count(self, run_id: str) -> int:
        fn = getattr(self._inner, "count", None)
        if callable(fn):
            try:
                return int(fn(run_id))
            except Exception:
                pass
        try:
            records = self._inner.list(run_id)
            return int(len(records) if isinstance(records, list) else 0)
        except Exception:
            return 0

    def count_many(self, run_ids: List[str]) -> Dict[str, int]:  # type: ignore[override]
        fn = getattr(self._inner, "count_many", None)
        if callable(fn):
            try:
                out = fn(run_ids)
                return out if isinstance(out, dict) else {}
            except Exception:
                return {}
        return {str(r or "").strip(): self.count(str(r or "").strip()) for r in (run_ids or []) if str(r or "").strip()}

    def metrics_many(self, run_ids: List[str]) -> Dict[str, Dict[str, int]]:  # type: ignore[override]
        fn = getattr(self._inner, "metrics_many", None)
        if callable(fn):
            try:
                out = fn(run_ids)
                return out if isinstance(out, dict) else {}
            except Exception:
                return {}
        return {}
