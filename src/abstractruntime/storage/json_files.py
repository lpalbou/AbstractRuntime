"""abstractruntime.storage.json_files

Simple file-based persistence:
- RunState checkpoints as JSON (one file per run)
- Ledger as JSONL (append-only)

This is meant as a straightforward MVP backend.
"""

from __future__ import annotations

import logging
import json
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import IDEMPOTENCY_TAIL_WINDOW, LedgerStore, RunStore
from .serialize import dumps_compact, runstate_to_dict, steprecord_to_dict
from ..core.models import RunState, StepRecord, RunStatus, StepStatus, WaitState, WaitReason
from ..core.run_lifecycle import run_lifecycle_index_fields

logger = logging.getLogger(__name__)


class JsonFileRunStore(RunStore):
    """File-based run store with query support.

    Implements both RunStore (ABC) and QueryableRunStore (Protocol).

    Query operations scan all run_*.json files, which is acceptable for MVP
    but needs lightweight indexing for interactive workloads (e.g. WS tick loops)
    once the run directory grows.

    OWNERSHIP CONTRACT (ruling 2026-07-09, runtime seat; the maintainer's
    instinct "if you get something from a store you should not be able to
    mutate the store" made explicit): `load()` may return the SAME RunState
    object it cached on `save()` — loaded runs are ALIASED, not copies.
    Within a process, only the run's OWNING thread (the tick/host thread
    driving `Runtime.tick`) may mutate a loaded RunState; every other writer
    goes through a durable side channel drained at tick boundaries (the
    directive sidecar / command inbox — agency-parity 0222), never through
    direct mutation of a loaded object. Rationale: a defensive deep-copy per
    load would tax every tick in proportion to vars size (history bundles,
    event inboxes), and a frozen view cannot distinguish the owner from
    bystanders — the single-writer rule is the only contract that keeps the
    hot path allocation-free AND makes the by-reference serialization in
    `save()` safe (no concurrent mutation mid-serialization can exist under
    it; see storage/serialize.py, backlog 0067).
    Cross-PROCESS readers are already safe: the cache is validated by file
    mtime and re-reads on any external write.
    """

    def __init__(self, base_dir: str | Path, *, run_cache_max: int = 512):
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)
        self._index_lock = threading.Lock()
        self._children_index: Optional[Dict[str, set[str]]] = None
        self._run_parent_index: Dict[str, Optional[str]] = {}
        self._run_cache_lock = threading.Lock()
        # run_id -> (mtime_ns, RunState), LRU-BOUNDED (flow's 2026-07-11
        # incident finding: the unbounded cache retained every RunState a
        # full directory scan ever loaded — ~1.5GB RSS on a 3k-run dir with
        # history-bearing vars, paid permanently by long-lived processes).
        # Eviction is safe under the ownership contract above: a re-load
        # after eviction re-reads the last SAVED state from disk, which is
        # exactly what any non-owner reader is entitled to see.
        from collections import OrderedDict

        self._run_cache: "OrderedDict[str, tuple[tuple[int, int, int], RunState]]" = OrderedDict()
        self._run_cache_max = max(1, int(run_cache_max))
        # SCAN MEMO (2026-07-15, entity's measured incident): run_id ->
        # (identity-token, index-fields dict). The RunState LRU above cannot
        # cover a large directory (512 entries vs 3,241 files on the live
        # box), so every runner poll evicted and RE-PARSED ~2.7k multi-MB,
        # mostly TERMINAL files that could never match its RUNNING filter —
        # one worker thread pegged ~98% CPU deep in json.loads, taxing every
        # endpoint through the GIL. The memo holds ONLY the small fields
        # scans filter on (status/wait/ids/timestamps/lifecycle — never
        # vars), so it covers the WHOLE directory in ~300B/file; entries are
        # identity-validated per use (same cross-process guarantee as the
        # RunState cache) and terminal files never change, so scans cost one
        # stat each instead of a parse. Unparseable files memo a tombstone
        # so a torn file is not re-parsed on every poll.
        self._scan_memo_lock = threading.Lock()
        self._scan_memo: Dict[str, tuple[tuple[int, int, int], Dict[str, Any]]] = {}

    @staticmethod
    def _stat_token(st: Any) -> tuple[int, int, int]:
        """File identity token: (mtime_ns, inode, size).

        mtime alone is NOT enough (scan-memo adversary P1-1, 2026-07-15): on
        1s-granularity filesystems (HFS+, some NFS/SMB, FAT) or under
        mtime-preserving tooling (rsync -a, tar), a rewrite can land with an
        identical mtime and the stale filter would then exclude the file
        BEFORE the full load that is the only thing that refreshes the memo
        — permanent staleness at any scale, plus an incoherent shape the
        old code could never produce (a status filter passing on a stale
        memo while the loaded object disagrees). `save()` writes via
        tmp.replace(), which mints a NEW INODE on every save, so the inode
        catches every same-mtime rewrite through the store's own write path
        at zero extra syscall cost; size catches most external in-place
        edits."""
        return (
            int(getattr(st, "st_mtime_ns", 0) or 0),
            int(getattr(st, "st_ino", 0) or 0),
            int(getattr(st, "st_size", 0) or 0),
        )

    def _cache_put(
        self,
        rid: str,
        token: tuple[int, int, int],
        run: RunState,
        *,
        memo_rid: Optional[str] = None,
    ) -> None:
        """Insert/refresh under the lock, evicting least-recently-used.

        `memo_rid` keys the SCAN MEMO (a per-FILE index — always the
        filename-derived id) separately from the RunState LRU (keyed by the
        internal run_id). They differ exactly for glob-matching copies
        (run_<id>_backup.json): memoizing the copy's token under the
        INTERNAL id used to poison the original's memo entry whenever
        directory order parsed the copy last (adversary P2-1's ping-pong,
        second head)."""
        with self._run_cache_lock:
            self._run_cache[rid] = (token, run)
            self._run_cache.move_to_end(rid)
            while len(self._run_cache) > self._run_cache_max:
                self._run_cache.popitem(last=False)
        self._scan_memo_put(memo_rid or rid, token, self._index_fields_of(run))

    @staticmethod
    def _index_fields_of(run: RunState) -> Dict[str, Any]:
        """The small filter/index fields scans need — never vars."""
        waiting = run.waiting
        return {
            "status": str(getattr(run.status, "value", run.status)),
            "workflow_id": str(run.workflow_id or ""),
            "session_id": str(run.session_id) if run.session_id else None,
            "parent_run_id": str(run.parent_run_id) if run.parent_run_id else None,
            "actor_id": str(run.actor_id) if run.actor_id else None,
            "created_at": str(run.created_at) if run.created_at else None,
            "updated_at": str(run.updated_at) if run.updated_at else None,
            "wait_reason": str(getattr(getattr(waiting, "reason", None), "value", waiting.reason)) if waiting is not None else None,
            "wait_until": str(waiting.until) if (waiting is not None and waiting.until) else None,
            **run_lifecycle_index_fields(run.vars),
        }

    def _scan_memo_put(self, rid: str, token: tuple[int, int, int], fields: Dict[str, Any]) -> None:
        if not rid or token[0] <= 0:
            return
        with self._scan_memo_lock:
            self._scan_memo[rid] = (token, fields)

    def _scan_memo_prune(self, seen_rids: set) -> None:
        """Drop memo entries whose files vanished (scan-memo adversary P2-4:
        externally pruned runs — the class's own recommended maintenance —
        held ~1.6KB/entry forever). Callers pass the ids the glob just saw;
        the set difference is cheap on a walk we already paid for."""
        with self._scan_memo_lock:
            for rid in set(self._scan_memo) - seen_rids:
                self._scan_memo.pop(rid, None)

    def _scan_fields(self, p: Path) -> Optional[Dict[str, Any]]:
        """Index fields for a run file: memo hit when the identity token
        matches, else one parse. None = unreadable file (tombstoned by
        token so a torn file is not re-parsed every poll)."""
        rid = self._run_id_from_path(p)
        try:
            token = self._stat_token(p.stat())
        except Exception:
            return None
        if rid and token[0] > 0:
            with self._scan_memo_lock:
                cached = self._scan_memo.get(rid)
            if cached is not None and tuple(cached[0]) == token:
                fields = cached[1]
                return None if fields.get("__unparseable") else fields
        # ANY load failure tombstones for scans (adversary P2-3: a valid-JSON
        # file with a bogus status enum raised ValueError out of every scan
        # API on every poll, and every run ranked below it was lost with the
        # call). Direct load() stays loud — polls must survive, repairs must
        # see the real error.
        try:
            run = self._load_from_path(p)
        except Exception as e:  # noqa: BLE001 - poll lanes survive bad files
            logger.warning("#FALLBACK run file %s unreadable for scans (%s); tombstoned", p.name, e)
            run = None
        if run is None:
            self._scan_memo_put(rid, token, {"__unparseable": True})
            return None
        fields = self._index_fields_of(run)
        # Memoize under the FILENAME-derived id (adversary P2-1): a
        # glob-matching copy (run_<id>_backup.json) has an internal run_id
        # that differs from its filename — _load_from_path's cache put keys
        # on the internal id, so the copy never memoized and every warm scan
        # re-parsed it (and its put poisoned the original's entry pre-token).
        self._scan_memo_put(rid, token, fields)
        return fields

    def _path(self, run_id: str) -> Path:
        return self._base / f"run_{run_id}.json"

    def _run_id_from_path(self, p: Path) -> str:
        name = str(getattr(p, "name", "") or "")
        if not name.startswith("run_") or not name.endswith(".json"):
            return ""
        return name[len("run_") : -len(".json")]

    def _ensure_children_index(self) -> None:
        if self._children_index is not None:
            return
        with self._index_lock:
            if self._children_index is not None:
                return

            children: Dict[str, set[str]] = {}
            run_parent: Dict[str, Optional[str]] = {}

            for run in self._iter_all_runs():
                parent = run.parent_run_id
                run_parent[run.run_id] = parent
                if isinstance(parent, str) and parent:
                    children.setdefault(parent, set()).add(run.run_id)

            self._children_index = children
            self._run_parent_index = run_parent

    def _drop_from_children_index(self, run_id: str) -> None:
        with self._index_lock:
            if self._children_index is None:
                return
            parent = self._run_parent_index.pop(run_id, None)
            if isinstance(parent, str) and parent:
                siblings = self._children_index.get(parent)
                if siblings is not None:
                    siblings.discard(run_id)
                    if not siblings:
                        self._children_index.pop(parent, None)

    def _update_children_index_on_save(self, run: RunState) -> None:
        run_id = run.run_id
        new_parent = run.parent_run_id

        with self._index_lock:
            if self._children_index is None:
                return

            old_parent = self._run_parent_index.get(run_id)
            if isinstance(old_parent, str) and old_parent and old_parent != new_parent:
                siblings = self._children_index.get(old_parent)
                if siblings is not None:
                    siblings.discard(run_id)
                    if not siblings:
                        self._children_index.pop(old_parent, None)

            self._run_parent_index[run_id] = new_parent
            if isinstance(new_parent, str) and new_parent:
                self._children_index.setdefault(new_parent, set()).add(run_id)

    def save(self, run: RunState) -> None:
        p = self._path(run.run_id)
        # Atomic write to prevent corrupted/partial JSON when multiple threads/processes
        # (e.g. WS tick loop + UI pause/cancel) write the same run file concurrently.
        tmp = p.with_name(f"{p.name}.{uuid.uuid4().hex}.tmp")
        try:
            with tmp.open("w", encoding="utf-8") as f:
                # Compact + by-reference serialization (backlog 0067): this is
                # the most frequent durable write in the system (~5 per agent
                # cycle); `indent=2` cost +72% CPU and ~35% bytes, and
                # `asdict` deep-copied the whole vars tree for nothing under
                # the single-writer contract documented on this class.
                f.write(dumps_compact(runstate_to_dict(run)))
            tmp.replace(p)
        finally:
            # Best-effort cleanup if replace() failed.
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
        self._update_children_index_on_save(run)
        try:
            token = self._stat_token(p.stat())
        except Exception:
            token = (0, 0, 0)
        if token[0] > 0:
            self._cache_put(str(run.run_id), token, run)

    def load(self, run_id: str) -> Optional[RunState]:
        p = self._path(run_id)
        if not p.exists():
            return None
        return self._load_from_path(p)

    def delete(self, run_id: str) -> bool:
        rid = str(run_id or "").strip()
        if not rid:
            return False
        p = self._path(rid)
        existed = p.exists()
        try:
            if existed:
                p.unlink()
        finally:
            self._drop_from_children_index(rid)
            with self._run_cache_lock:
                self._run_cache.pop(rid, None)
            with self._scan_memo_lock:
                self._scan_memo.pop(rid, None)
        return bool(existed)

    def _load_from_path(self, p: Path) -> Optional[RunState]:
        """Load a RunState from a file path."""
        rid_hint = self._run_id_from_path(p)
        try:
            token = self._stat_token(p.stat())
        except Exception:
            token = (0, 0, 0)
        if rid_hint and token[0] > 0:
            with self._run_cache_lock:
                cached = self._run_cache.get(rid_hint)
                if cached is not None and tuple(cached[0]) == token:
                    self._run_cache.move_to_end(rid_hint)  # LRU touch
                    return cached[1]
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

        # Reconstruct enums and nested dataclasses
        raw_status = data.get("status")
        status = raw_status if isinstance(raw_status, RunStatus) else RunStatus(str(raw_status))

        waiting: Optional[WaitState] = None
        raw_waiting = data.get("waiting")
        if isinstance(raw_waiting, dict):
            raw_reason = raw_waiting.get("reason")
            if raw_reason is None:
                raise ValueError("Persisted waiting state missing 'reason'")
            reason = raw_reason if isinstance(raw_reason, WaitReason) else WaitReason(str(raw_reason))
            waiting = WaitState(
                reason=reason,
                wait_key=raw_waiting.get("wait_key"),
                until=raw_waiting.get("until"),
                resume_to_node=raw_waiting.get("resume_to_node"),
                result_key=raw_waiting.get("result_key"),
                prompt=raw_waiting.get("prompt"),
                choices=raw_waiting.get("choices"),
                allow_free_text=bool(raw_waiting.get("allow_free_text", True)),
                details=raw_waiting.get("details"),
            )

        run = RunState(
            run_id=data["run_id"],
            workflow_id=data["workflow_id"],
            status=status,
            current_node=data["current_node"],
            vars=data.get("vars") or {},
            waiting=waiting,
            output=data.get("output"),
            error=data.get("error"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            actor_id=data.get("actor_id"),
            session_id=data.get("session_id"),
            parent_run_id=data.get("parent_run_id"),
        )
        rid = str(getattr(run, "run_id", "") or "").strip() or rid_hint
        if rid and token[0] > 0:
            # Memo under the FILENAME id: the memo indexes FILES (scans walk
            # the glob), the LRU indexes RUNS — for a copy the two diverge
            # and the copy's token must never overwrite the original's memo.
            self._cache_put(rid, token, run, memo_rid=rid_hint or rid)
        return run

    def _iter_all_runs(self) -> List[RunState]:
        """Iterate over all stored runs."""
        runs: List[RunState] = []
        for p in self._base.glob("run_*.json"):
            run = self._load_from_path(p)
            if run is not None:
                runs.append(run)
        return runs

    # --- QueryableRunStore methods ---

    def list_runs(
        self,
        *,
        status: Optional[RunStatus] = None,
        wait_reason: Optional[WaitReason] = None,
        workflow_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[RunState]:
        """List runs matching the given filters.

        Performance note:
        - We order by run file mtime (close to updated_at) and stop once we have `limit` matches.
        - This avoids parsing every historical run JSON file on large runtimes.

        Scale verdict (2026-07-11 incident review, measured on a 3,030-run dir):
        - FRESHNESS IS SAFE at/below `limit` concurrent matches: a newly saved
          run has the newest mtime, is inspected first, and is therefore always
          included. With fewer than `limit` matching runs the scan visits every
          file, so nothing can be dropped.
        - STARVATION HOLE above `limit`: mtime refreshes on every save, so with
          more than `limit` concurrently RUNNING runs the actively-ticked ones
          keep re-claiming the newest-mtime slots and a never-yet-ticked run's
          mtime only ages — it can be starved indefinitely. The gateway polls
          this with limit=run_scan_limit (default 200); raise the limit or add
          a status index before hosting >200 concurrent RUNNING runs.
        - COST: every call globs+stats the whole directory (~31 ms warm at 3k
          files; ~2.2 s cold because scarce matches force parsing every file).
          The gateway runner issues 3 such scans per 0.25 s poll. Linear in
          total run files, including terminal ones — archive/prune terminal
          runs or add an index before this directory reaches ~10k files.
        - MEMORY: `_run_cache` is LRU-BOUNDED (default 512 entries,
          `run_cache_max` constructor param) since the 2026-07-11 incident
          review — the unbounded cache retained every RunState a full scan
          ever loaded (~1.5 GB RSS on a 3k-run dir). Eviction is safe under
          the ownership contract: a re-load re-reads the last saved state.
        """
        lim = max(1, int(limit or 100))
        ranked: list[tuple[int, Path]] = []
        for p in self._base.glob("run_*.json"):
            try:
                st = p.stat()
                mtime_ns = int(getattr(st, "st_mtime_ns", 0) or 0)
            except Exception:
                continue
            ranked.append((mtime_ns, p))
        ranked.sort(key=lambda x: x[0], reverse=True)
        self._scan_memo_prune({self._run_id_from_path(p) for _m, p in ranked})

        results: List[RunState] = []
        for _mtime_ns, p in ranked:
            # Memo-first filtering (2026-07-15 scan-memo): non-matching
            # unchanged files cost one stat, never a parse — the runner's
            # RUNNING polls used to re-parse every multi-MB terminal file
            # the 512-entry RunState LRU had just evicted.
            fields = self._scan_fields(p)
            if fields is None:
                continue
            if status is not None and fields.get("status") != str(getattr(status, "value", status)):
                continue
            if workflow_id is not None and fields.get("workflow_id") != str(workflow_id):
                continue
            if wait_reason is not None:
                want = str(getattr(wait_reason, "value", wait_reason))
                if fields.get("wait_reason") != want:
                    continue
            run = self._load_from_path(p)
            if run is None:
                continue
            results.append(run)
            if len(results) >= lim:
                break

        results.sort(key=lambda r: r.updated_at or "", reverse=True)
        return results[:lim]

    def list_run_index(
        self,
        *,
        status: Optional[RunStatus] = None,
        workflow_id: Optional[str] = None,
        session_id: Optional[str] = None,
        root_only: bool = False,
        limit: int = 100,
        oldest_first: bool = False,
    ) -> List[Dict[str, Any]]:
        """List lightweight run index rows without depending on full RunState consumers."""
        lim = max(1, int(limit or 100))
        ranked: list[tuple[int, Path]] = []
        for p in self._base.glob("run_*.json"):
            try:
                st = p.stat()
                mtime_ns = int(getattr(st, "st_mtime_ns", 0) or 0)
            except Exception:
                continue
            ranked.append((mtime_ns, p))
        # mtime ranking approximates updated_at; oldest_first inverts so a
        # stall query's window keeps the OLDEST waits (0054 adversary P1-1).
        ranked.sort(key=lambda x: x[0], reverse=not oldest_first)
        self._scan_memo_prune({self._run_id_from_path(p) for _m, p in ranked})

        out: List[Dict[str, Any]] = []
        sid = str(session_id or "").strip() if session_id is not None else None

        for _mtime_ns, p in ranked:
            # Memo-first (2026-07-15 scan-memo): index rows ARE the memo
            # fields — an index page over unchanged files parses nothing.
            fields = self._scan_fields(p)
            if fields is None:
                continue
            if status is not None and fields.get("status") != str(getattr(status, "value", status)):
                continue
            if workflow_id is not None and fields.get("workflow_id") != str(workflow_id):
                continue
            if sid is not None and str(fields.get("session_id") or "").strip() != sid:
                continue
            if bool(root_only) and str(fields.get("parent_run_id") or "").strip():
                continue

            rid = self._run_id_from_path(p)
            # Copy dict values one level deep (adversary P2-2): rows alias
            # the memo's nested run_lifecycle dict — a consumer mutating
            # row["run_lifecycle"] would poison every future index call
            # until the file's identity changes.
            out.append({
                "run_id": rid,
                **{
                    k: (dict(v) if isinstance(v, dict) else v)
                    for k, v in fields.items()
                    if not k.startswith("__")
                },
            })
            if len(out) >= lim:
                break

        out.sort(key=lambda r: str(r.get("updated_at") or ""), reverse=not oldest_first)
        return out[:lim]

    def list_due_wait_until(
        self,
        *,
        now_iso: str,
        limit: int = 100,
    ) -> List[RunState]:
        """List runs whose wait DEADLINE has passed: UNTIL waits always;
        EVENT waits when they carry `until` (the D3 idle-timeout shape —
        the scheduler must wake a parked visit whose deadline passed)."""
        results: List[RunState] = []

        # Memo-first (2026-07-15 scan-memo): the scheduler's due-scan runs on
        # a poll loop too — filter on stat-validated fields, parse only runs
        # that are actually due.
        paths = list(self._base.glob("run_*.json"))
        self._scan_memo_prune({self._run_id_from_path(p) for p in paths})
        for p in paths:
            fields = self._scan_fields(p)
            if fields is None:
                continue
            if fields.get("status") != RunStatus.WAITING.value:
                continue
            if fields.get("wait_reason") not in (WaitReason.UNTIL.value, WaitReason.EVENT.value):
                continue
            until = fields.get("wait_until")
            if not until:
                continue
            # Check if the wait time has passed (ISO string comparison works for UTC)
            if str(until) <= now_iso:
                run = self._load_from_path(p)
                if run is not None and run.waiting is not None and run.waiting.until:
                    results.append(run)

        # Sort by waiting.until ascending (earliest due first)
        results.sort(key=lambda r: r.waiting.until if r.waiting else "")

        return results[:limit]

    def list_children(
        self,
        *,
        parent_run_id: str,
        status: Optional[RunStatus] = None,
    ) -> List[RunState]:
        """List child runs of a parent."""
        self._ensure_children_index()
        with self._index_lock:
            child_ids = list((self._children_index or {}).get(parent_run_id, set()))

        results: List[RunState] = []
        for run_id in sorted(child_ids):
            run = self.load(run_id)
            if run is None:
                self._drop_from_children_index(run_id)
                continue
            if status is not None and run.status != status:
                continue
            results.append(run)

        return results


class JsonlLedgerStore(LedgerStore):
    def __init__(self, base_dir: str | Path):
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)
        # Write-through idempotency cache (backlog 0047): the tick loop asks
        # "is there a prior COMPLETED result for this key?" before EVERY
        # effect step; a full-file scan per step is O(ledger) and was the
        # measured scale cliff. COMPLETED records are cached here at append
        # time so the crash-replay lookup (same process) is O(1); cold
        # processes fall back to a bounded backward tail read (see
        # find_completed_result).
        from collections import OrderedDict

        self._idem_cache_lock = threading.Lock()
        self._idem_cache: "OrderedDict[tuple[str, str], Dict[str, Any]]" = OrderedDict()
        self._idem_cache_max = 512

    def _path(self, run_id: str) -> Path:
        return self._base / f"ledger_{run_id}.jsonl"

    def append(self, record: StepRecord) -> None:
        p = self._path(record.run_id)
        record_dict = steprecord_to_dict(record)
        line = dumps_compact(record_dict)
        with p.open("a", encoding="utf-8") as f:
            # Write as a single append operation to reduce the chance of
            # concurrent writers producing concatenated JSON objects on one line.
            f.write(line + "\n")
        self._idem_cache_note(record_dict, line)

    # Records above this size are not cached; the bounded tail read serves
    # them. 512 entries x 64KB caps worst-case cache memory at ~32MB.
    _IDEM_CACHE_MAX_RECORD_BYTES = 65536

    def _idem_cache_note(self, record_dict: Dict[str, Any], record_json: str) -> None:
        """Cache a COMPLETED record's serialized LINE under (run_id, key).

        The already-built line string is cached (parsed on hit), never the
        live result object: workflow reducers write results into vars and
        may mutate them in place afterwards, while the legacy lookup parsed
        disk bytes — a by-reference cache would silently diverge from disk
        truth (and skip the tuple->list coercion a JSON round-trip applies).
        Reusing the append's own serialization costs ZERO extra encoding
        (2026-07-13 perf adversary N2: re-serializing the result here cost
        +98% per completed append on result-dominated records); hits are
        rare (crash-replay), so parse-on-hit is the right trade.
        """
        try:
            if record_dict.get("status") != StepStatus.COMPLETED.value:
                return
            key = record_dict.get("idempotency_key")
            run_id = record_dict.get("run_id")
            if not (isinstance(key, str) and key and isinstance(run_id, str) and run_id):
                return
            if len(record_json) > self._IDEM_CACHE_MAX_RECORD_BYTES:
                return
            with self._idem_cache_lock:
                cache_key = (run_id, key)
                # Oldest-completed-wins parity with the legacy full scan:
                # never overwrite an existing entry for the same key.
                if cache_key not in self._idem_cache:
                    self._idem_cache[cache_key] = record_json
                    while len(self._idem_cache) > self._idem_cache_max:
                        self._idem_cache.popitem(last=False)
        except Exception:  # noqa: BLE001 - cache maintenance must never fail an append
            pass

    # Byte ceiling for the backward tail read (2026-07-13 perf adversary N3:
    # a line cap alone let 256 five-MB record lines force a ~1.3GB read and
    # buffer per cold probe on media-shaped ledgers). Beyond the ceiling the
    # probe answers an honest miss — the same documented at-least-once
    # semantics as the line window. Block size is an attribute so the cap
    # is testable at small scale (a 64KB first read swallows small files
    # before the cap can bite).
    _TAIL_MAX_BYTES = 32 * 1024 * 1024
    _TAIL_BLOCK_BYTES = 65536

    def _tail_lines(self, p: Path, max_lines: int) -> List[str]:
        """Read up to the last `max_lines` lines by seeking backwards.

        Reads O(tail bytes) bounded by `_TAIL_MAX_BYTES`, never the whole
        file — the point of the tail window is that fat early records
        (large LLM payloads) cost nothing. A byte-cap stop may leave the
        oldest collected line truncated mid-record; the caller's parse loop
        skips unparseable lines, which keeps that honest.
        """
        block = int(self._TAIL_BLOCK_BYTES)
        chunks: List[bytes] = []
        newlines = 0
        total = 0
        with p.open("rb") as f:
            f.seek(0, 2)
            pos = f.tell()
            while pos > 0 and newlines <= max_lines and total < self._TAIL_MAX_BYTES:
                read_size = min(block, pos)
                pos -= read_size
                f.seek(pos)
                data = f.read(read_size)
                chunks.append(data)
                newlines += data.count(b"\n")
                total += len(data)
        buf = b"".join(reversed(chunks))
        text = buf.decode("utf-8", errors="replace")
        # Split on the writer's ACTUAL line discipline — "\n" only. NEVER
        # str.splitlines() here (replay adversary P0, 2026-07-14): JSON
        # leaves U+2028/U+2029/U+0085 RAW under ensure_ascii=False, and
        # splitlines() splits on all three — a completed record carrying
        # U+2028 in scraped/LLM text fragmented into unparseable pieces, the
        # cold crash-replay probe missed it, and the effect RE-EXECUTED.
        # (Raw "\r" cannot occur inside a line: JSON escapes all controls
        # below U+0020.)
        lines = text.split("\n")
        if lines and lines[-1] == "":
            lines.pop()  # trailing newline artifact, not an empty line
        if pos > 0 and total >= self._TAIL_MAX_BYTES and lines:
            # Byte-cap stop mid-file: the oldest collected line is a
            # truncated fragment of a record — drop it rather than hand a
            # fragment to the parser (it could coincidentally parse).
            lines = lines[1:]
        return lines[-max_lines:] if len(lines) > max_lines else lines

    def find_completed_result(
        self, run_id: str, idempotency_key: str
    ) -> Optional[Dict[str, Any]]:
        """Bounded idempotency lookup (backlog 0047).

        Semantics: exact for issuance-scoped keys (`_runtime.effect_seq` is
        hashed into every key since 2026-07-13), because a prior COMPLETED
        record for the CURRENT issuance can only exist within the current
        step's own records — which are at the ledger tail by construction
        (crash-replay is "effect completed, save didn't land"). The lookup
        is write-through-cache first, then a backward tail read of
        IDEMPOTENCY_TAIL_WINDOW lines. Deliberately NO full-file fallback:
        the common case is a MISS (fresh key per issuance), and a full scan
        on miss would reinstate the O(ledger)-per-step cliff this replaces.
        Beyond the window the runtime re-executes — the documented
        at-least-once default for effects without a completed record.
        """
        key = str(idempotency_key or "")
        rid = str(run_id or "")
        if not key or not rid:
            return None
        p = self._path(rid)
        if not p.exists():
            # Disk truth wins over any cached answer (replay adversary P2-b:
            # a SECOND store instance over the same directory kept serving a
            # run's results from its write-through cache after another
            # instance deleted the ledger). No file = no records.
            with self._idem_cache_lock:
                for cache_key in [k for k in self._idem_cache if k[0] == rid]:
                    del self._idem_cache[cache_key]
            return None
        with self._idem_cache_lock:
            cached = self._idem_cache.get((rid, key))
        if cached is not None:
            try:
                rec = json.loads(cached)
                if isinstance(rec, dict):
                    return rec.get("result")
            except json.JSONDecodeError:  # pragma: no cover - we serialized it
                pass
        try:
            lines = self._tail_lines(p, IDEMPOTENCY_TAIL_WINDOW)
        except OSError:
            return None
        # The quoted-key prefilter is a fast REJECT for lines that cannot
        # contain the key — but it is only sound when the key serializes to
        # itself. Keys containing characters JSON escapes ('"', '\\',
        # controls) appear ESCAPED on disk and would false-negative a
        # genuine hit (replay adversary P2-a; host-pluggable EffectPolicy
        # keys are arbitrary strings). Such keys skip straight to parsing.
        prefilter_safe = '"' not in key and "\\" not in key and key.isprintable()
        # Oldest-first within the window: parity with the legacy full scan
        # (first completed match wins) for pre-issuance-key ledgers.
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if prefilter_safe and f'"{key}"' not in line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue
            if rec.get("idempotency_key") != key:
                continue
            if rec.get("status") != StepStatus.COMPLETED.value:
                continue
            return rec.get("result")
        return None

    def list(self, run_id: str) -> List[Dict[str, Any]]:
        p = self._path(run_id)
        if not p.exists():
            return []
        out: List[Dict[str, Any]] = []
        decoder = json.JSONDecoder()
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if isinstance(record, dict):
                        out.append(record)
                    continue
                except json.JSONDecodeError:
                    pass

                # Recover from concatenated JSON objects on one line.
                # This can happen if a process crashes mid-write or if a file was
                # manually edited/concatenated. We do a best-effort recovery by
                # repeatedly using raw_decode and advancing the index.
                recovered: List[Dict[str, Any]] = []
                i = 0
                while i < len(line):
                    while i < len(line) and line[i].isspace():
                        i += 1
                    if i >= len(line):
                        break
                    try:
                        obj, end = decoder.raw_decode(line, idx=i)
                    except json.JSONDecodeError:
                        break
                    if isinstance(obj, dict):
                        recovered.append(obj)
                    i = end

                if recovered:
                    logger.warning(
                        "JsonlLedgerStore.list #FALLBACK recovered %s JSON objects from one line in %s",
                        len(recovered),
                        str(p),
                    )
                    out.extend(recovered)
        return out

    def count(self, run_id: str) -> int:
        """Return the number of ledger records for run_id (fast path).

        This avoids JSON parsing when only a count is needed (e.g. UI dropdowns).
        """
        p = self._path(run_id)
        if not p.exists():
            return 0
        n = 0
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    n += 1
        return n

    def delete(self, run_id: str) -> int:
        rid = str(run_id or "").strip()
        if not rid:
            return 0
        p = self._path(rid)
        count = len(self.list(rid))
        if p.exists():
            p.unlink()
        with self._idem_cache_lock:
            # Drop the run's cached results: a recreated ledger under the
            # same run_id must never see the deleted run's completions.
            for cache_key in [k for k in self._idem_cache if k[0] == rid]:
                del self._idem_cache[cache_key]
        return count
