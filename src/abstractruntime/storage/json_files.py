"""abstractruntime.storage.json_files

Simple file-based persistence:
- RunState checkpoints as JSON (one file per run)
- Ledger as JSONL (append-only)

This is meant as a straightforward MVP backend.
"""

from __future__ import annotations

import contextlib
import logging
import json
import os
import threading
import time
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
        # EVENT-WAIT INDEX (2026-09-22, mission B2): wait_key -> {run_id} for
        # runs parked in WAITING(EVENT), plus the reverse map used to retract a
        # run's old key on save. `emit_event` used to find its listeners with a
        # `list_runs(WAITING, EVENT, limit=10_000)` over the WHOLE directory —
        # 57ms per call on a 9,326-run store (0.17-0.25s on the operator's
        # 15GB one) and TWICE per chat turn, on a store that has zero event
        # waiters. Built lazily from one scan (so a restart heals itself) and
        # maintained on save/delete, exactly like `_children_index`; a lookup
        # re-loads and re-verifies every candidate against disk, so a stale
        # entry costs one load and self-heals, never a wrong resume.
        # SCOPE, stated plainly: like `_children_index`, this index sees the
        # saves made through THIS store object. A second PROCESS parking a run
        # in WAITING(EVENT) is not seen until this process rebuilds — the same
        # single-writer assumption `save()`'s by-reference serialization and
        # the ownership contract above already rest on (emit_event RESUMES the
        # waiters it finds, i.e. it writes them, so the two processes would be
        # racing the same run file either way).
        self._event_index_lock = threading.Lock()
        self._event_wait_index: Optional[Dict[str, set[str]]] = None
        self._run_event_wait_key: Dict[str, str] = {}
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
        # SIDECAR (2026-08-19, operator's session-reload investigation —
        # measured on the live 7.9k-run / 1.8GB directory): the memo dies
        # with the process, so EVERY gateway restart re-paid a full-JSON
        # parse of the whole store on the first scan (6.3s) and AGAIN on
        # the first list_children (7.3s via _iter_all_runs) — the "first
        # session reload is slow" experience. The sidecar persists the
        # memo's (token, fields) rows; entries are identity-validated
        # per use, so a stale/corrupt/foreign sidecar degrades to
        # re-parses — with the stat token's ONE documented blind spot
        # (adversarial review 2026-08-19, F2): an EXTERNAL in-place
        # rewrite that preserves mtime_ns+size+inode would go unseen
        # across restarts too (pre-sidecar, a restart healed it). No
        # writer in this system does that (save() is tmp+replace = new
        # inode), and the blast radius is filter/index rows — load()
        # always returns disk truth. Cache, not truth: persist/load
        # failures are swallowed.
        self._scan_sidecar_path = self._base / ".runs_scan_cache.json"
        self._scan_sidecar_loaded = False
        # New knowledge since the last persist (new rids, changed fields,
        # tombstones, prunes). Persist policy lives in
        # `_maybe_persist_scan_memo` — checked at scan ends only, never on
        # the save() hot path.
        self._scan_dirty = 0
        self._scan_last_persist = time.monotonic()

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
        """The small filter/index fields scans need — never vars.

        `__run_id` is the INTERNAL run id (dunder = stripped from
        list_run_index rows): the children index needs it because a
        glob-matching copy's filename-derived id differs from the id
        the tree actually references (adversary P2-1's copy class)."""
        waiting = run.waiting
        return {
            "__run_id": str(run.run_id),
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
            # Only NEW rids count as persist-worthy dirt (adversarial
            # review 2026-08-19, F3): an active run's row goes token-
            # stale by the next restart regardless, so re-persisting on
            # every field flip bought nothing and cost a 4.4MB write
            # per 32 saves. New/pruned/deleted rids are what a restart
            # actually wants to remember.
            if rid not in self._scan_memo:
                self._scan_dirty += 1
            self._scan_memo[rid] = (token, fields)

    def _scan_memo_prune(self, seen_rids: set) -> None:
        """Drop memo entries whose files vanished (scan-memo adversary P2-4:
        externally pruned runs — the class's own recommended maintenance —
        held ~1.6KB/entry forever). Callers pass the ids the glob just saw;
        the set difference is cheap on a walk we already paid for."""
        with self._scan_memo_lock:
            for rid in set(self._scan_memo) - seen_rids:
                self._scan_memo.pop(rid, None)
                self._scan_dirty += 1

    def _load_scan_sidecar_once(self) -> None:
        """Seed the scan memo from the persisted sidecar, once per process.

        Entries are (token, fields) rows exactly as the memo holds them;
        every use re-validates the token against the file's current stat,
        so a stale entry costs one re-parse and a corrupt/foreign sidecar
        degrades to the pre-sidecar cold scan. In-memory entries win over
        sidecar rows (they are at least as fresh)."""
        if self._scan_sidecar_loaded:
            return
        with self._scan_memo_lock:
            if self._scan_sidecar_loaded:
                return
            self._scan_sidecar_loaded = True
            try:
                raw = json.loads(self._scan_sidecar_path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                return
            except Exception as e:  # noqa: BLE001 - cache, not truth
                logger.warning("#FALLBACK scan sidecar unreadable (%s); cold scan", e)
                return
            entries = raw.get("entries") if isinstance(raw, dict) else None
            if int((raw or {}).get("version") or 0) != 1 or not isinstance(entries, dict):
                return
            for rid, item in entries.items():
                try:
                    token = (int(item[0][0]), int(item[0][1]), int(item[0][2]))
                    fields = item[1]
                except Exception:
                    continue
                if rid and isinstance(fields, dict) and token[0] > 0 and rid not in self._scan_memo:
                    self._scan_memo[rid] = (token, fields)

    def _maybe_persist_scan_memo(self) -> None:
        """Persist the memo when enough new knowledge accumulated.

        Called at SCAN ends only (list_runs / list_run_index /
        list_due_wait_until / the children-index build) — never on the
        save() hot path. The thresholds keep steady-state writes rare
        (a run's own saves only mark dirt; the next scan flushes it),
        while a cold rebuild (thousands of parses) persists immediately."""
        with self._scan_memo_lock:
            dirty = self._scan_dirty
            elapsed = time.monotonic() - self._scan_last_persist
        if dirty >= 32 or (dirty > 0 and elapsed >= 120.0):
            self._persist_scan_memo()

    def _persist_scan_memo(self) -> None:
        with self._scan_memo_lock:
            snapshot = {rid: [list(tok), fields] for rid, (tok, fields) in self._scan_memo.items()}
            self._scan_dirty = 0
            self._scan_last_persist = time.monotonic()
        # uuid tmp (adversarial review 2026-08-19, F4): pid alone
        # collides for two stores over one dir in one process — an
        # interleaved write_text could publish a torn (= discarded)
        # sidecar. Same discipline as save()'s tmp names.
        tmp = self._scan_sidecar_path.with_name(f".runs_scan_cache.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            # default=str never fires today (_index_fields_of str-coerces
            # everything; lifecycle fields are str/int) — it exists so a
            # future non-JSON-safe field degrades to a stringly row
            # instead of killing every persist. If you ADD a field with
            # non-str semantics, round-trip it explicitly.
            tmp.write_text(
                json.dumps({"version": 1, "entries": snapshot}, separators=(",", ":"), default=str),
                encoding="utf-8",
            )
            os.replace(tmp, self._scan_sidecar_path)
        except Exception as e:  # noqa: BLE001 - a failed persist costs a
            # future cold scan, never correctness.
            logger.warning("#FALLBACK scan sidecar persist failed (%s)", e)
            with contextlib.suppress(Exception):
                tmp.unlink()

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

            # Build from SCAN FIELDS, never full parses (2026-08-19,
            # operator's session-reload investigation): the old
            # `_iter_all_runs` walk json.load-ed every run file — 7.3s
            # on the live 7.9k-run / 1.8GB directory, paid on the FIRST
            # list_children of every process (= the first history_bundle
            # of the first session reload after every gateway restart),
            # and it ignored the memo a scan had just populated. The
            # memo/sidecar fields carry `__run_id` (internal id) and
            # `parent_run_id` — everything this index needs; unchanged
            # files now cost one stat, changed ones one parse.
            self._load_scan_sidecar_once()
            for p in self._base.glob("run_*.json"):
                fields = self._scan_fields(p)
                if fields is None:
                    continue
                rid = str(fields.get("__run_id") or self._run_id_from_path(p) or "")
                if not rid:
                    continue
                parent = fields.get("parent_run_id")
                parent = str(parent) if parent else None
                run_parent[rid] = parent
                if parent:
                    children.setdefault(parent, set()).add(rid)

            self._children_index = children
            self._run_parent_index = run_parent
        self._maybe_persist_scan_memo()

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

    # --- event-wait index (emit_event listener lookup) -------------------

    @staticmethod
    def _event_wait_key_of(run: RunState) -> Optional[str]:
        """The wait_key this run is parked on as an EVENT waiter, else None."""
        if str(getattr(run.status, "value", run.status)) != RunStatus.WAITING.value:
            return None
        waiting = getattr(run, "waiting", None)
        if waiting is None:
            return None
        reason = getattr(waiting, "reason", None)
        if str(getattr(reason, "value", reason)) != WaitReason.EVENT.value:
            return None
        wait_key = getattr(waiting, "wait_key", None)
        if not isinstance(wait_key, str) or not wait_key:
            return None
        return wait_key

    def _ensure_event_wait_index(self) -> None:
        """Build the wait_key -> {run_id} index once, from ONE directory scan.

        Filtering runs off the memo's `wait_reason` field (one stat per
        unchanged file, the same cost as the scan this index replaces), then
        parsing only the EVENT waiters — there are typically none, and the
        operator's 9,326-run store has exactly zero. A restart therefore heals
        any in-memory drift at the cost of one scan on the first emit."""
        if self._event_wait_index is not None:
            return
        with self._event_index_lock:
            if self._event_wait_index is not None:
                return
            index: Dict[str, set[str]] = {}
            by_run: Dict[str, str] = {}
            self._load_scan_sidecar_once()
            for p in self._base.glob("run_*.json"):
                fields = self._scan_fields(p)
                if fields is None:
                    continue
                if fields.get("status") != RunStatus.WAITING.value:
                    continue
                if fields.get("wait_reason") != WaitReason.EVENT.value:
                    continue
                # The memo holds no wait_key (it would change the sidecar row
                # shape and cost every deployment a cold re-parse on upgrade);
                # an EVENT waiter is rare enough to parse.
                run = self._load_from_path(p)
                if run is None:
                    continue
                wait_key = self._event_wait_key_of(run)
                if wait_key is None:
                    continue
                rid = str(run.run_id)
                index.setdefault(wait_key, set()).add(rid)
                by_run[rid] = wait_key
            self._event_wait_index = index
            self._run_event_wait_key = by_run
        self._maybe_persist_scan_memo()

    def _drop_from_event_wait_index(self, run_id: str) -> None:
        with self._event_index_lock:
            if self._event_wait_index is None:
                return
            key = self._run_event_wait_key.pop(run_id, None)
            if not isinstance(key, str) or not key:
                return
            holders = self._event_wait_index.get(key)
            if holders is not None:
                holders.discard(run_id)
                if not holders:
                    self._event_wait_index.pop(key, None)

    def _update_event_wait_index_on_save(self, run: RunState) -> None:
        """Retract the run's old key and record its new one (if any).

        The retraction is what makes a status change safe: a run that RESUMES
        (or completes, or re-parks on a DIFFERENT key) is removed from the key
        it was parked on, so it can never be resumed twice by the same event.
        """
        run_id = str(run.run_id)
        new_key = self._event_wait_key_of(run)
        with self._event_index_lock:
            if self._event_wait_index is None:
                return  # not built yet: the lazy build will read disk truth
            old_key = self._run_event_wait_key.pop(run_id, None)
            if isinstance(old_key, str) and old_key and old_key != new_key:
                holders = self._event_wait_index.get(old_key)
                if holders is not None:
                    holders.discard(run_id)
                    if not holders:
                        self._event_wait_index.pop(old_key, None)
            if new_key is not None:
                self._event_wait_index.setdefault(new_key, set()).add(run_id)
                self._run_event_wait_key[run_id] = new_key

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
        self._update_event_wait_index_on_save(run)
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
            self._drop_from_event_wait_index(rid)
            with self._run_cache_lock:
                self._run_cache.pop(rid, None)
            with self._scan_memo_lock:
                if self._scan_memo.pop(rid, None) is not None:
                    self._scan_dirty += 1
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

    # `_iter_all_runs` was removed 2026-08-19 (operator's session-reload
    # investigation): it full-parsed every run file — 7.3s on the live
    # 7.9k-run / 1.8GB directory — and its one caller (the children-index
    # build) now reads the memoized scan fields instead. A whole-store
    # RunState walk has no sub-O(store) implementation; anything that
    # thinks it needs one should go through `_scan_fields`.

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
        self._load_scan_sidecar_once()
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
        self._maybe_persist_scan_memo()
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
        self._load_scan_sidecar_once()
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
        self._maybe_persist_scan_memo()
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
        self._load_scan_sidecar_once()
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

        self._maybe_persist_scan_memo()
        return results[:limit]

    def list_event_waiters(
        self,
        *,
        wait_keys: List[str],
        limit: int = 100,
    ) -> List[RunState]:
        """Runs parked in WAITING(EVENT) on any of `wait_keys` — O(waiters).

        The set this returns is exactly the subset of
        `list_runs(status=WAITING, wait_reason=EVENT, limit=big)` whose
        `waiting.wait_key` is in `wait_keys`, in the same `updated_at`
        descending order — that equivalence is the correctness bar, and
        `tests/test_event_wait_index.py` pins it against the scan.

        Every candidate the index names is re-loaded and re-verified against
        disk before it is returned, so an index entry that no longer matches
        (the run resumed, completed, or re-parked on another key) is dropped
        rather than resumed.
        """
        keys = [str(k) for k in (wait_keys or []) if isinstance(k, str) and k]
        if not keys:
            return []
        wanted = set(keys)
        self._ensure_event_wait_index()
        with self._event_index_lock:
            index = self._event_wait_index or {}
            candidate_ids: List[str] = []
            seen: set[str] = set()
            for key in keys:
                for rid in index.get(key, set()):
                    if rid not in seen:
                        seen.add(rid)
                        candidate_ids.append(rid)

        results: List[RunState] = []
        for rid in candidate_ids:
            run = self.load(rid)
            if run is None:
                self._drop_from_event_wait_index(rid)
                continue
            actual = self._event_wait_key_of(run)
            if actual is None or actual not in wanted:
                # Disk disagrees with the index: re-file this run under what
                # it actually says (or drop it) and skip it for this event.
                self._update_event_wait_index_on_save(run)
                continue
            results.append(run)

        results.sort(key=lambda r: r.updated_at or "", reverse=True)
        return results[: max(1, int(limit or 100))]

    def list_event_waiters_by_prefix(self, *, prefix: str, limit: int = 100) -> List[RunState]:
        """Runs parked in WAITING(EVENT) on a wait_key starting with `prefix`.

        Same verification and ordering as `list_event_waiters`; feeds
        `emit_event`'s `available_listeners_in_session` diagnostic, which the
        whole-store scan used to produce as a side effect. Callers apply their
        own run-level filters (e.g. paused) to the RunStates, exactly as they
        did to the scan's output.
        """
        pre = str(prefix or "")
        if not pre:
            return []
        self._ensure_event_wait_index()
        with self._event_index_lock:
            index = self._event_wait_index or {}
            candidate_ids: List[str] = []
            seen: set[str] = set()
            for key, rids in index.items():
                if not key.startswith(pre):
                    continue
                for rid in rids:
                    if rid not in seen:
                        seen.add(rid)
                        candidate_ids.append(rid)

        results: List[RunState] = []
        for rid in candidate_ids:
            run = self.load(rid)
            if run is None:
                self._drop_from_event_wait_index(rid)
                continue
            actual = self._event_wait_key_of(run)
            if actual is None or not actual.startswith(pre):
                self._update_event_wait_index_on_save(run)
                continue
            results.append(run)

        results.sort(key=lambda r: r.updated_at or "", reverse=True)
        return results[: max(1, int(limit or 100))]

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
