"""abstractruntime.storage.json_files

Simple file-based persistence:
- RunState checkpoints as JSON (one file per run)
- Ledger as JSONL (append-only)

This is meant as a straightforward MVP backend.
"""

from __future__ import annotations

import bisect
import contextlib
import logging
import sys
from array import array
import json
import os
import threading
from dataclasses import dataclass
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import IDEMPOTENCY_TAIL_WINDOW, LedgerStore, RunStore, coerce_ledger_seq, mailbox_names_of
from .serialize import dumps_compact, runstate_to_dict, steprecord_to_dict
from .change_ring import ChangeRing as _ChangeRing
from ..core.models import RunState, StepRecord, RunStatus, StepStatus, WaitState, WaitReason
from ..core.run_attribution import workspace_index_fields
from ..core.run_attribution import automation_index_fields, is_turn_root, latest_occurrence_of, row_matches
from ..core.run_identity import verify_run_identity
from ..core.run_lifecycle import run_lifecycle_index_fields

try:  # POSIX advisory file locks (the JSONL ledger's cross-process append lock)
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - Windows
    _fcntl = None  # type: ignore[assignment]


def _inode_of(f: Any) -> int:
    try:
        return int(os.fstat(f.fileno()).st_ino)
    except OSError:
        return -1


def _flock(f: Any) -> None:
    if _fcntl is not None:
        _fcntl.flock(f.fileno(), _fcntl.LOCK_EX)


def _funlock(f: Any) -> None:
    if _fcntl is not None:
        _fcntl.flock(f.fileno(), _fcntl.LOCK_UN)

logger = logging.getLogger(__name__)

# Scan-sidecar row shape version. 2 (automations, 2026-09-27): rows gained the
# attribution fields `automation_id`, `role`, `occurrence_index`,
# `session_kind`; 3 (2026-09-27): rows gained `workspace_root`. An older
# sidecar is ignored so every file is re-read once.
_SCAN_SIDECAR_VERSION = 3


def _consumer_fields(run: RunState) -> Dict[str, Any]:
    """`automation_consumer` on a CONTROLLER's row only: `{status, trigger}` (the automation's
    status and trigger binding), so the active-consumers index (`automations.consumers`) is
    built from index rows without loading controller runs. Other rows carry nothing more."""
    meta = run.vars.get("_meta") if isinstance(run.vars, dict) else None
    if not (isinstance(meta, dict) and isinstance(meta.get("automation"), dict)):
        return {}
    from ..automations.consumers import consumer_index_fields

    return {"automation_consumer": consumer_index_fields(run)}


def _lacks_consumer_fields(fields: Dict[str, Any]) -> bool:
    """A controller row memoized before `automation_consumer` existed (an older sidecar): re-parse
    it once instead of bumping the sidecar version (which would re-parse every run file)."""
    return fields.get("role") == "controller" and "automation_consumer" not in fields


@dataclass
class _ActiveEntry:
    """What the ACTIVE index knows about one run id (see `JsonFileRunStore`)."""

    active: bool = False
    # Bumped by every note/drop: a listing that read the file as terminal drops
    # the member only if nothing noted it in between (a per-store sequence).
    gen: int = 0
    # The file's identity token after THIS store's last save of it as active
    # (`None` when the last save was terminal, or never by this store): a save
    # that finds the file unchanged since then is the common per-tick save and
    # announces nothing; any other active save announces `~id`.
    saved_token: Optional[tuple] = None
    # This store's writes of an active file in flight: the id is a candidate
    # before the note lands and is never dropped for a failed stat meanwhile.
    publishing: int = 0


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

    # A run temp file older than this is an orphan of a crashed writer (a
    # save()/create_if_absent() publishes its temp within milliseconds).
    STALE_TEMP_AGE_S = 600.0
    # An index build re-applies the journal records written in this many bytes BEFORE its mark
    # (`_mark_journal_start`; ~400 records): a run whose `+` line landed before the mark while its
    # file was published after the folder scan is otherwise in neither (gateway backlog 0963).
    JOURNAL_REPLAY_TAIL_BYTES = 16 * 1024
    # Background reconciliation of the active index (foreign writers that bypass the journal).
    ACTIVE_INDEX_RESCAN_S = 60.0
    # Trusted run index: with the folder and the creation log unchanged since the last
    # reconciliation, the next full pass waits at most this long (a foreign in-place edit
    # keeps the folder's mtime; nothing in this system writes run files in place).
    RUN_INDEX_FULL_RECONCILE_S = 600.0
    # A reconciliation difference on a file changed less than this long ago may be a write
    # still in flight (its log line or its own memo update not landed yet): repaired, not
    # counted as a miss.
    RUN_INDEX_IN_FLIGHT_S = 5.0
    # Event-wait and mailbox indexes: a lookup re-reads the active runs another process changed
    # when the folder or the creation log moved, and at least this often regardless (a folder
    # mtime that did not move: a coarse-mtime filesystem, an in-place edit).
    EVENT_INDEX_REFRESH_S = 2.0
    _ACTIVE_STATUSES = frozenset({"running", "waiting"})

    def __init__(
        self,
        base_dir: str | Path,
        *,
        run_cache_max: int = 512,
        use_active_index: bool = True,
        trust_index: bool = True,
    ):
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)
        self._sweep_stale_temp_files()
        self._index_lock = threading.Lock()
        self._children_index: Optional[Dict[str, set[str]]] = None
        # SESSION INDEX (automations review 45 H1): session_id -> {file run id:
        # session_kind}. Session-scoped reads (turn selection, the discussion
        # anchor on EVERY root start that names a session) used to glob+stat
        # the whole directory — 277 ms median at 20k runs. Built lazily from
        # one memo scan, maintained on save/create/delete, and every candidate
        # is re-validated by stat (a deleted file drops out). Same single-
        # writer scope as `_children_index`: runs another PROCESS adds to a
        # session are seen after this process rebuilds (one writer process per
        # store, automations contract C16).
        self._session_index_lock = threading.Lock()
        self._session_index: Optional[Dict[str, Dict[str, Optional[str]]]] = None
        self._run_session: Dict[str, str] = {}
        # CREATION JOURNAL (review 51 J51-1): the session and children indexes
        # change only when a run file is CREATED or DELETED (a run's session,
        # parent and session_kind are fixed at creation). Every writer — this
        # object, another store object, another process on the same folder —
        # appends `+<run_id>` / `-<run_id>` to `.runs_created.log` (O_APPEND,
        # one short line). A lookup stats the journal (one stat) and applies
        # only the new lines, so a discussion created through another store
        # object is seen before the next session lookup; standby/multi-worker
        # gateways on one data folder stay consistent. A shrunk (rewritten)
        # journal or an unreadable one drops both indexes for a full rebuild.
        self._journal_path = self._base / ".runs_created.log"
        self._journal_lock = threading.Lock()
        self._journal_offset: Optional[int] = None
        from collections import deque as _deque

        self._journal_own_appends: Any = _deque(maxlen=8192)
        # Entries whose file change is not visible yet (review 53 J53-1): the
        # journal line is written BEFORE the run file is published (or
        # unlinked), so a reader can see `+id` with no file yet. They are
        # re-checked on every lookup until the file appears (or disappears for
        # `-id`), and forgotten after STALE_TEMP_AGE_S (the writer crashed).
        self._journal_pending: Dict[str, tuple[str, float]] = {}
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
        # SCOPE: maintained by the saves made through THIS store object, and
        # refreshed from the store's change signal before every lookup, so a
        # run another PROCESS parked in WAITING(EVENT) is found (the
        # cross-process refresh below, 2026-10-26).
        self._event_index_lock = threading.Lock()
        self._event_wait_index: Optional[Dict[str, set[str]]] = None
        self._run_event_wait_key: Dict[str, str] = {}
        # MAILBOX INDEX (backlog 1046, design round-1 F3): mailbox name ->
        # {run_id} for ACTIVE runs (RUNNING/WAITING) declaring `events_mailbox`.
        # Durable emit_event used to find its mailboxes in the newest
        # `run_scan_limit` (200) active runs only: past that, a parked
        # automation's mailbox silently missed events. Same lifecycle and
        # scope as the event-wait index above: built lazily from one scan of
        # the active ids, maintained on save/delete, refreshed across
        # processes before every lookup, every candidate re-verified on disk.
        self._mailbox_index: Optional[Dict[str, set[str]]] = None
        self._run_mailboxes: Dict[str, frozenset] = {}
        # CROSS-PROCESS REFRESH of the two indexes above (2026-10-26, split mode: `serve
        # --no-runner` + `abstractgateway runner` on one folder). The runner process parks a run
        # in WAITING(EVENT) by re-saving a run that stays active: no creation-log line, so the
        # API process's indexes never heard of it and an event it delivered missed the waiter.
        # Every lookup now reads the store's change signal first: when the creation log or the
        # folder's mtime moved since the last refresh (every save moves the folder: tmp +
        # replace), or EVENT_INDEX_REFRESH_S passed, each ACTIVE run (the active index, kept
        # current by the creation log) is stat()ed and the ones whose file changed since this
        # store last filed them are re-read and re-filed. Unchanged folder: two stats. Changed
        # folder: one stat per active run, one parse per run another process changed (the
        # parse the scan memo makes anyway). This store's own saves record their token, so its
        # own writes are never re-read.
        self._xproc_lock = threading.Lock()
        self._xproc_mark: Optional[tuple] = None
        self._xproc_at: float = 0.0
        self._xproc_tokens: Dict[str, tuple] = {}
        self._xproc_refreshes = 0
        self._xproc_refiled = 0
        # ACTIVE INDEX (2026-10-16, operator's "the observer is extremely slow to
        # load"): the ids of every run whose status is RUNNING or WAITING — the
        # question both listings are asked on every poll by the gateway runner
        # (3 scans per 0.25 s), the Observer board, the tray and the Assistant.
        # Each of those scans globbed + stat()ed the WHOLE directory (10,678
        # files on the operator's store); every stat drops the GIL, so the scan
        # convoys behind any CPU-bound thread: 0.19 s alone, 2.9 s with two
        # concurrent, 67–147 s beside a generating model (measured 2026-10-16).
        # Built lazily from ONE scan, maintained on save()/delete() and through
        # the creation journal (`_index_fields_into_indexes`), like the event
        # index; a listing takes its CANDIDATES from the set and re-verifies
        # every one against disk with `_scan_fields` (stat + memo) — a member
        # that is terminal or gone by then is filtered out AND dropped, so a
        # stale entry costs one stat and self-heals, never a wrong row. The
        # ranking (mtime), filters, limit and final `updated_at` sort are the
        # scan's own, applied to the candidates: with a complete set the rows
        # are identical to the scan's (differential proof: tests + the
        # untracked/active-index harness). What the journal does not carry — a
        # foreign process flipping a file to RUNNING without announcing it — is
        # caught by a background RESCAN at most every ACTIVE_INDEX_RESCAN_S
        # (daemon thread, never on a caller's path). `use_active_index=False`
        # keeps the plain scan (the proof's reference).
        self._use_active_index = bool(use_active_index)
        self._active_lock = threading.Lock()
        # run id -> _ActiveEntry: membership, generation, this store's last
        # active-save token, in-flight writes. One record per run this store
        # has heard of; inactive records with nothing else to say are pruned
        # at every rescan.
        self._active: Dict[str, _ActiveEntry] = {}
        self._active_seq = 0
        # CHANGE SIGNAL (2026-10-26, the gateway's idle core): `change_token()` is this counter
        # plus the creation log's size. Every save, create, verified status change and journal
        # entry (another process's create / delete / "became active") moves one of them, so
        # "did anything change?" is one stat instead of a glob + stat of every run file (0.18 s
        # at 11,500 runs, every 0.5 s, from the gateway runner). `_change_ring` names the runs
        # behind the counter's last bumps (`changed_run_ids`).
        self._change_ring = _ChangeRing()
        self._active_built = False
        self._active_building = False
        self._active_built_at: float = 0.0
        self._active_rescan_inflight = False
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
        # TRUSTED RUN INDEX (2026-10-26, the gateway's idle core, backlog 1054): every
        # listing that is not RUNNING/WAITING or session-scoped globbed + stat()ed the whole
        # folder (11,500 files) to re-check the memo above, and under contention each stat
        # waits for the interpreter lock (0.25 s alone, minutes beside a busy thread). Once
        # ONE complete pass has filled the memo (`_memo_complete`, set by the active-index
        # build, which reads every file), the memo IS the listing: ranked by the memo's
        # mtime, no glob, no stat of a run file. It stays current through this store's own
        # save/create/delete, the creation log (`+` create, `-` delete, `~` became active,
        # `=` a finished run's file changed — written after publication by every save of a
        # run that is not active), a stat of the memo's ACTIVE entries when the folder's
        # mtime moved (another process ticking a run writes no log line), and a background
        # RECONCILIATION (the active-index rescan: a full stat pass in a daemon thread, at
        # most every ACTIVE_INDEX_RESCAN_S when the folder or the log moved, at least every
        # RUN_INDEX_FULL_RECONCILE_S) that repairs and logs any difference. Risk: a stale
        # row, never a lost run (`load()` reads disk). `trust_index=False` ("verify") keeps
        # the directory scan for every listing and compares it with the in-memory answer
        # (SHADOW mode: `run_index_stats()["shadow_mismatches"]`). Needs the active index
        # (`use_active_index=False` is the plain scan, with no shadow).
        self._trust_index = bool(trust_index)
        self._memo_complete = False
        # Bumped by every memo put/pop: the ranked listing is cached on it, and a read that
        # began before this store's own save of a run never overwrites that save's entry.
        self._memo_version = 0
        self._memo_own_seq: Dict[str, int] = {}
        self._memo_active: set[str] = set()
        self._ranked_cache: Optional[tuple[int, list]] = None
        self._memo_active_checked_dir: Optional[int] = None
        self._journal_last_size: Optional[int] = None
        self._reconcile_marks: Optional[tuple] = None
        self._reconcile_last_full = time.monotonic()
        # Writes through this store in flight or done (begin and end both bump it): a shadow
        # comparison that overlapped one is not a verdict.
        self._own_writes = 0
        self._index_stats_lock = threading.Lock()
        self._index_stats: Dict[str, Any] = {
            "trusted_listings": 0,
            "scanned_listings": 0,
            "shadow_compared": 0,
            "shadow_mismatches": 0,
            "shadow_unstable": 0,
            "last_mismatch": None,
            "reconciles": 0,
            "reconcile_repairs": 0,
            "last_reconcile": None,
        }

    def _sweep_stale_temp_files(self) -> None:
        """Delete `run_<id>.json.<hex>.tmp` orphans left by a crash between a
        temp write and its publication (os.link / replace). Fresh temps may
        belong to a live writer in another process and are kept. Best effort:
        a failed unlink only leaves the orphan for the next open."""
        cutoff = time.time() - float(self.STALE_TEMP_AGE_S)
        for tmp in self._base.glob("run_*.json.*.tmp"):
            try:
                if tmp.stat().st_mtime < cutoff:
                    tmp.unlink()
            except OSError:
                continue

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
        seq0: Optional[int] = None,
        own: bool = False,
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
        self._scan_memo_put(memo_rid or rid, token, self._index_fields_of(run), seq0=seq0, own=own)

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
            **automation_index_fields(run.vars, run_id=str(run.run_id)),
            **workspace_index_fields(run.vars),
            **_consumer_fields(run),
        }

    def _scan_memo_put(
        self,
        rid: str,
        token: tuple[int, int, int],
        fields: Dict[str, Any],
        *,
        seq0: Optional[int] = None,
        own: bool = False,
    ) -> None:
        """`seq0`: the memo version when the read behind this entry began; the entry is dropped
        when this store saved the run after that (its save's entry is fresher). `own`: the entry
        of this store's own save."""
        if not rid or token[0] <= 0:
            return
        with self._scan_memo_lock:
            if seq0 is not None and not own and self._memo_own_seq.get(rid, -1) > seq0:
                return
            self._memo_version += 1
            if own:
                self._memo_own_seq[rid] = self._memo_version
            if self._is_active_status(fields.get("status")):
                self._memo_active.add(rid)
            else:
                self._memo_active.discard(rid)
            # Only NEW rids count as persist-worthy dirt (adversarial
            # review 2026-08-19, F3): an active run's row goes token-
            # stale by the next restart regardless, so re-persisting on
            # every field flip bought nothing and cost a 4.4MB write
            # per 32 saves. New/pruned/deleted rids are what a restart
            # actually wants to remember.
            prev = self._scan_memo.get(rid)
            if prev is None:
                self._scan_dirty += 1
            self._scan_memo[rid] = (token, fields)
        consumer = fields.get("automation_consumer")
        if consumer is not None and (prev is None or prev[1].get("automation_consumer") != consumer):
            # A controller appeared, or its automation status / trigger changed, through ANY save
            # this process makes (or a file another process changed, seen by a scan): the
            # active-consumers index's change signal (`automations.consumers`).
            from ..automations.consumers import invalidate_active_consumers

            invalidate_active_consumers(self)

    def _scan_memo_prune(self, seen_rids: set) -> List[str]:
        """Drop memo entries whose files vanished (scan-memo adversary P2-4:
        externally pruned runs — the class's own recommended maintenance —
        held ~1.6KB/entry forever). Callers pass the ids the glob just saw;
        the set difference is cheap on a walk we already paid for. A file
        that exists now (created after the glob listed the folder) keeps its
        entry: the trusted index lists from the memo. Returns the ids dropped."""
        with self._scan_memo_lock:
            gone = [rid for rid in self._scan_memo if rid not in seen_rids]
        dropped: List[str] = []
        for rid in gone:
            if self._path(rid).exists():
                continue
            if self._memo_pop(rid):
                dropped.append(rid)
        return dropped

    def _memo_pop(self, rid: str, *, token: Optional[tuple] = None) -> bool:
        """Forget a run file's memo entry (only when it still holds `token`, if given)."""
        with self._scan_memo_lock:
            cur = self._scan_memo.get(rid)
            if cur is None or (token is not None and tuple(cur[0]) != tuple(token)):
                return False
            self._scan_memo.pop(rid, None)
            self._memo_active.discard(rid)
            self._memo_own_seq.pop(rid, None)
            self._memo_version += 1
            self._scan_dirty += 1
            return True

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
            if int((raw or {}).get("version") or 0) != _SCAN_SIDECAR_VERSION or not isinstance(entries, dict):
                return
            for rid, item in entries.items():
                try:
                    token = (int(item[0][0]), int(item[0][1]), int(item[0][2]))
                    fields = item[1]
                except Exception:
                    continue
                if rid and isinstance(fields, dict) and token[0] > 0 and rid not in self._scan_memo:
                    self._scan_memo[rid] = (token, fields)
                    if self._is_active_status(fields.get("status")):
                        self._memo_active.add(rid)
            self._memo_version += 1

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
                json.dumps({"version": _SCAN_SIDECAR_VERSION, "entries": snapshot}, separators=(",", ":"), default=str),
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
        with self._scan_memo_lock:
            seq0 = self._memo_version
        try:
            token = self._stat_token(p.stat())
        except Exception:
            return None
        if rid and token[0] > 0:
            with self._scan_memo_lock:
                cached = self._scan_memo.get(rid)
            if cached is not None and tuple(cached[0]) == token and not _lacks_consumer_fields(cached[1]):
                fields = cached[1]
                return None if fields.get("__unparseable") else fields
        # ANY load failure tombstones for scans (adversary P2-3: a valid-JSON
        # file with a bogus status enum raised ValueError out of every scan
        # API on every poll, and every run ranked below it was lost with the
        # call). Direct load() stays loud — polls must survive, repairs must
        # see the real error.
        try:
            run = self._load_from_path(p)  # its own read-start version guards the memo put
        except Exception as e:  # noqa: BLE001 - poll lanes survive bad files
            logger.warning("#FALLBACK run file %s unreadable for scans (%s); tombstoned", p.name, e)
            run = None
        if run is None:
            self._scan_memo_put(rid, token, {"__unparseable": True}, seq0=seq0)
            return None
        # `_load_from_path` memoized these fields under the FILENAME-derived
        # id (adversary P2-1: a glob-matching copy run_<id>_backup.json has an
        # internal run_id that differs from its filename) with the token of
        # the file it actually read (fstat of the open file, so a replace
        # between the stat above and the read can never pair new fields with
        # the old token).
        return self._index_fields_of(run)

    def _path(self, run_id: str) -> Path:
        return self._base / f"run_{run_id}.json"

    def _run_id_from_path(self, p: Path) -> str:
        name = str(getattr(p, "name", "") or "")
        if not name.startswith("run_") or not name.endswith(".json"):
            return ""
        return name[len("run_") : -len(".json")]

    def _ensure_children_index(self) -> None:
        if self._children_index is not None:
            self._sync_indexes()
            if self._children_index is not None:
                return
        self._mark_journal_start()
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

    # --- creation journal ------------------------------------------------

    def _journal_append(self, line: str) -> None:
        """Append one entry, BEFORE the file change it announces (a crash
        after publication can then never hide a run from other processes).

        Records are framed `\n<op><run_id>\n`: the leading newline ends any
        torn line a crashed writer left, so a torn tail never merges with the
        next record (review 53 J53-2). This object applies the change to its
        own indexes itself, so when nothing else was appended since its last
        sync it skips its own record (no re-read)."""
        data = ("\n" + line + "\n").encode("utf-8")
        try:
            with self._journal_lock:
                with open(self._journal_path, "ab") as f:
                    f.write(data)
                    f.flush()
                    end = f.tell()
                if self._journal_offset is not None and self._journal_offset == end - len(data):
                    self._journal_offset = end
                # (start, end) of this store's own lines: `changed_run_ids` tells its own
                # appends from another process's.
                self._journal_own_appends.append((end - len(data), end))
        except OSError:
            # Other store objects would miss this run until they rebuild:
            # loud, because a missed discussion run weakens read-only anchoring.
            logger.error("JsonFileRunStore: creation journal append failed for %s", line, exc_info=True)

    def _journal_size(self) -> Optional[int]:
        try:
            return int(self._journal_path.stat().st_size)
        except FileNotFoundError:
            return 0
        except OSError:
            return None

    def _mark_journal_start(self) -> None:
        """Called BEFORE an index's full scan: entries appended from here on
        are re-applied by `_sync_indexes` (idempotent), so nothing created
        during the scan is lost."""
        with self._journal_lock:
            if self._journal_offset is None:
                size = self._journal_size()
                self._journal_offset = self._journal_replay_start(size) if size is not None else 0

    def _journal_replay_start(self, size: int) -> int:
        """Where the first sync after an index build starts reading: the first record boundary
        within JOURNAL_REPLAY_TAIL_BYTES before `size` (the journal's size at the mark).

        `_save` appends a new run's `+id` BEFORE publishing its file (J53-1). A build that marks
        the journal at `size` and then scans the folder misses a run whose line is before the
        mark and whose file lands after the scan; re-applying the tail (idempotent: a `+` whose
        file exists is filed, one whose file is still absent stays pending until it appears)
        closes that window. Cost: one read of at most JOURNAL_REPLAY_TAIL_BYTES and one stat per
        record in it, once per build. A record older than the tail is still caught by the
        periodic reconciliation, which logs what it repairs."""
        start = max(0, int(size) - int(self.JOURNAL_REPLAY_TAIL_BYTES))
        if start == 0:
            return 0
        try:
            with open(self._journal_path, "rb") as f:
                f.seek(start)
                chunk = f.read(int(size) - start)
        except OSError:
            logger.warning("JsonFileRunStore: creation journal tail unreadable; index build replays nothing before its mark",
                           exc_info=True)
            return int(size)
        boundary = chunk.find(b"\n")  # records are framed `\n<op><run_id>\n`: start after a newline
        return start + boundary + 1 if boundary >= 0 else int(size)

    def _sync_indexes(self) -> None:
        """Apply journal lines appended since the last sync (one stat when
        nothing changed), then re-check pending entries."""
        if self._journal_offset is None:
            return
        size = self._journal_size()
        self._journal_last_size = size
        with self._journal_lock:
            offset = self._journal_offset
            if offset is None or size == offset:
                pending = dict(self._journal_pending)
                offset = None
        if offset is None:
            if pending:
                self._apply_journal_entries(pending.items())
            return
        with self._journal_lock:
            offset = self._journal_offset
            if offset is None or size == offset:
                return
            if size is None or size < offset:
                # Unreadable or rewritten journal: rebuild from disk.
                with self._index_lock:
                    self._children_index = None
                with self._session_index_lock:
                    self._session_index = None
                with self._active_lock:
                    self._active_built = False
                    self._memo_complete = False
                self._journal_offset = None
                return
            try:
                with open(self._journal_path, "rb") as f:
                    f.seek(offset)
                    chunk = f.read(size - offset)
            except OSError:
                with self._index_lock:
                    self._children_index = None
                with self._session_index_lock:
                    self._session_index = None
                with self._active_lock:
                    self._active_built = False
                    self._memo_complete = False
                self._journal_offset = None
                return
            complete = chunk.rfind(b"\n") + 1  # a line still being written waits for the next sync
            self._journal_offset = offset + complete
            pending = dict(self._journal_pending)
        entries = list(pending.items())
        now = time.time()
        for raw in chunk[:complete].decode("utf-8", "replace").splitlines():
            op, rid = raw[:1], raw[1:].strip()
            if op in ("+", "-", "~", "=") and rid:
                entries.append((rid, (op, now)))
        self._apply_journal_entries(entries)

    def _apply_journal_entries(self, entries: Any) -> None:
        cutoff = time.time() - float(self.STALE_TEMP_AGE_S)
        for rid, (op, since) in entries:
            p = self._path(rid)
            exists = p.exists()
            if op in ("+", "~", "=") and exists:
                fields = self._scan_fields(p)
                if fields is not None:
                    self._index_fields_into_indexes(rid, fields)
                # `~` announces a flip TO active: until the file says so it is
                # not resolved (the writer's replace may land after its line).
                # `=` is written AFTER its file was published: one read resolves it.
                resolved = op in ("+", "=") or (fields is not None and self._is_active_status(fields.get("status"))) or since < cutoff
            elif op == "-" and not exists:
                self._drop_from_children_index(rid)
                self._drop_from_session_index(rid)
                self._forget_active(rid)
                self._memo_pop(rid)
                resolved = True
            elif op == "=":
                resolved = True  # the file is gone: its `-` line (or the reconciliation) says so
            else:
                resolved = since < cutoff  # the announced change never happened: drop it
            with self._journal_lock:
                if resolved:
                    self._journal_pending.pop(rid, None)
                else:
                    self._journal_pending[rid] = (op, since)

    def _index_fields_into_indexes(self, rid: str, fields: Dict[str, Any]) -> None:
        parent = fields.get("parent_run_id")
        parent = str(parent) if parent else None
        with self._index_lock:
            if self._children_index is not None:
                self._run_parent_index[rid] = parent
                if parent:
                    self._children_index.setdefault(parent, set()).add(rid)
        sid = str(fields.get("session_id") or "").strip()
        with self._session_index_lock:
            if self._session_index is not None and sid:
                self._session_index.setdefault(sid, {})[rid] = fields.get("session_kind")
                self._run_session[rid] = sid
        self._note_active_status(rid, fields.get("status"))

    def warm_session_index(self) -> None:
        """Build the session and children indexes now (host boot), so the
        one-time scan (~0.75 s at 20k runs) is not paid by the first chat."""
        self._ensure_session_index()
        self._ensure_children_index()
        if self._use_active_index:
            self._ensure_active_index()

    def _ensure_session_index(self) -> None:
        if self._session_index is not None:
            self._sync_indexes()
            if self._session_index is not None:
                return
        self._mark_journal_start()
        with self._session_index_lock:
            if self._session_index is not None:
                return
            index: Dict[str, Dict[str, Optional[str]]] = {}
            by_run: Dict[str, str] = {}
            self._load_scan_sidecar_once()
            for p in self._base.glob("run_*.json"):
                fields = self._scan_fields(p)
                if fields is None:
                    continue
                sid = str(fields.get("session_id") or "").strip()
                rid = self._run_id_from_path(p)
                if not sid or not rid:
                    continue
                index.setdefault(sid, {})[rid] = fields.get("session_kind")
                by_run[rid] = sid
            self._session_index = index
            self._run_session = by_run
        self._maybe_persist_scan_memo()

    def _update_session_index_on_save(self, run: RunState) -> None:
        rid = str(run.run_id)
        sid = str(run.session_id or "").strip()
        kind = automation_index_fields(run.vars, run_id=rid)["session_kind"]
        with self._session_index_lock:
            if self._session_index is None:
                return  # not built yet: the lazy build reads disk truth
            old = self._run_session.pop(rid, None)
            if old and old != sid:
                members = self._session_index.get(old)
                if members is not None:
                    members.pop(rid, None)
                    if not members:
                        self._session_index.pop(old, None)
            if sid:
                self._session_index.setdefault(sid, {})[rid] = kind
                self._run_session[rid] = sid

    def _drop_from_session_index(self, run_id: str) -> None:
        with self._session_index_lock:
            if self._session_index is None:
                return
            sid = self._run_session.pop(run_id, None)
            members = self._session_index.get(sid) if sid else None
            if members is not None:
                members.pop(run_id, None)
                if not members:
                    self._session_index.pop(sid, None)

    def session_kinds(self, session_id: str) -> frozenset:
        """The `session_kind` values of the session's runs (empty: no runs).
        From the session index after one stat of the creation journal (runs
        created through other store objects/processes are applied first)."""
        sid = str(session_id or "").strip()
        if not sid:
            return frozenset()
        self._ensure_session_index()
        with self._session_index_lock:
            members = dict((self._session_index or {}).get(sid) or {})
        return frozenset(k for k in members.values() if k)

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
        mark = self._xproc_change_mark()
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
        self._xproc_note_built(mark)
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

    # --- mailbox index (durable emit_event fan-out) ------------------------

    @classmethod
    def _mailboxes_of(cls, run: RunState) -> frozenset:
        """Mailbox names an ACTIVE run declares (`vars.events_mailbox`)."""
        if not cls._is_active_status(run.status):
            return frozenset()
        return mailbox_names_of(getattr(run, "vars", None))

    def _ensure_mailbox_index(self) -> None:
        if self._mailbox_index is not None:
            return
        mark = self._xproc_change_mark()
        with self._event_index_lock:
            if self._mailbox_index is not None:
                return
            ids = self._active_candidate_ids()
            if ids is None:
                ids = [self._run_id_from_path(p) for p in self._base.glob("run_*.json")]
            index: Dict[str, set[str]] = {}
            by_run: Dict[str, frozenset] = {}
            for rid in ids:
                if not rid:
                    continue
                p = self._path(rid)
                fields = self._scan_fields(p)
                if fields is None or not self._is_active_status(fields.get("status")):
                    continue
                run = self._load_from_path(p)
                if run is None:
                    continue
                names = self._mailboxes_of(run)
                if not names:
                    continue
                key = str(run.run_id)
                by_run[key] = names
                for n in names:
                    index.setdefault(n, set()).add(key)
            self._mailbox_index = index
            self._run_mailboxes = by_run
        self._xproc_note_built(mark)

    def _drop_from_mailbox_index(self, run_id: str) -> None:
        with self._event_index_lock:
            if self._mailbox_index is None:
                return
            for n in self._run_mailboxes.pop(run_id, frozenset()):
                holders = self._mailbox_index.get(n)
                if holders is not None:
                    holders.discard(run_id)
                    if not holders:
                        self._mailbox_index.pop(n, None)

    def _update_mailbox_index_on_save(self, run: RunState) -> None:
        run_id = str(run.run_id)
        new = self._mailboxes_of(run)
        with self._event_index_lock:
            if self._mailbox_index is None:
                return  # not built yet: the lazy build reads disk truth
            old = self._run_mailboxes.pop(run_id, frozenset())
            for n in old - new:
                holders = self._mailbox_index.get(n)
                if holders is not None:
                    holders.discard(run_id)
                    if not holders:
                        self._mailbox_index.pop(n, None)
            if new:
                for n in new:
                    self._mailbox_index.setdefault(n, set()).add(run_id)
                self._run_mailboxes[run_id] = new

    # --- cross-process refresh of the event-wait and mailbox indexes ----------

    def _xproc_change_mark(self) -> tuple:
        return (self._journal_size(), self._dir_mtime())

    def _xproc_note_own(self, rid: str, token: Optional[tuple], active: bool) -> None:
        """This store wrote `rid` (its indexes were updated by the write itself)."""
        with self._xproc_lock:
            if active and token is not None and token[0] > 0:
                self._xproc_tokens[rid] = tuple(token)
            else:
                self._xproc_tokens.pop(rid, None)

    def _xproc_note_built(self, mark: tuple) -> None:
        """An index was just built from disk: what it read is filed (the mark is the change
        signal read BEFORE the build, so a write during the build is re-read next time)."""
        with self._scan_memo_lock:
            memo = {rid: tuple(v[0]) for rid, v in self._scan_memo.items() if self._is_active_status(v[1].get("status"))}
        with self._xproc_lock:
            for rid, tok in memo.items():
                self._xproc_tokens.setdefault(rid, tok)
            if self._xproc_mark is None:
                self._xproc_mark = mark
                self._xproc_at = time.monotonic()

    def _refresh_event_indexes(self) -> None:
        """File what OTHER processes changed among the active runs (see `__init__`)."""
        if self._event_wait_index is None and self._mailbox_index is None:
            return
        mark = self._xproc_change_mark()
        now = time.monotonic()
        with self._xproc_lock:
            fresh = (
                mark == self._xproc_mark
                and None not in mark
                and now - self._xproc_at < float(self.EVENT_INDEX_REFRESH_S)
            )
            if fresh:
                return
            self._xproc_mark = mark
            self._xproc_at = now
            self._xproc_refreshes += 1
        ids = self._active_candidate_ids()  # applies other processes' creation-log lines first
        if ids is None:
            ids = [rid for rid in (self._run_id_from_path(p) for p in self._base.glob("run_*.json")) if rid]
        keep: Dict[str, tuple] = {}
        with self._xproc_lock:
            before = dict(self._xproc_tokens)
        refiled = 0
        for rid in ids:
            p = self._path(rid)
            try:
                tok = self._stat_token(p.stat())
            except OSError:
                continue  # gone: the lookup drops its entries when it re-verifies them
            if before.get(rid) == tok:
                keep[rid] = tok
                continue
            fields = self._scan_fields(p)  # memo hit, or the one parse (the run is cached by it)
            if fields is not None:
                event_waiter = (
                    fields.get("status") == RunStatus.WAITING.value
                    and fields.get("wait_reason") == WaitReason.EVENT.value
                )
                need_event = self._event_wait_index is not None and (event_waiter or rid in self._run_event_wait_key)
                need_mailbox = self._mailbox_index is not None and (
                    self._is_active_status(fields.get("status")) or rid in self._run_mailboxes
                )
                if need_event or need_mailbox:
                    run = self._load_from_path(p)
                    if run is not None:
                        if need_event:
                            self._update_event_wait_index_on_save(run)
                        if need_mailbox:
                            self._update_mailbox_index_on_save(run)
                        refiled += 1
            keep[rid] = tok
        with self._xproc_lock:
            # A token noted by this store's own save while the pass ran is newer than the one
            # the pass read: it wins. Runs no longer active are forgotten (bounded table).
            for rid, tok in self._xproc_tokens.items():
                if rid in keep and before.get(rid) != tok:
                    keep[rid] = tok
            self._xproc_tokens = keep
            self._xproc_refiled += refiled

    def event_index_stats(self) -> Dict[str, Any]:
        """Counters of the cross-process refresh (refresh passes run, runs re-filed)."""
        with self._xproc_lock:
            return {
                "refreshes": int(self._xproc_refreshes),
                "refiled": int(self._xproc_refiled),
                "tracked": len(self._xproc_tokens),
            }

    # --- active index --------------------------------------------------------

    @classmethod
    def _is_active_status(cls, status: Any) -> bool:
        return str(getattr(status, "value", status)) in cls._ACTIVE_STATUSES

    @property
    def _active_index(self) -> Optional[set[str]]:
        """The active ids as a set (a snapshot; None before the first build)."""
        with self._active_lock:
            if not self._active_built:
                return None
            return {rid for rid, e in self._active.items() if e.active}

    def _active_entry(self, rid: str) -> _ActiveEntry:
        e = self._active.get(rid)
        if e is None:
            e = self._active[rid] = _ActiveEntry()
        return e

    def _note_active_status(self, rid: str, status: Any) -> None:
        """Record what a save, a journal line or a verified read says about a
        run. Always recorded (before the first build too): a scan in progress
        merges by generation, a build not yet started reads disk truth and
        reconciles these notes with it."""
        if not rid:
            return
        with self._active_lock:
            e = self._active_entry(rid)
            self._active_seq += 1
            e.gen = self._active_seq
            e.active = self._is_active_status(status)
            self._change_ring.note(self._active_seq, rid)

    def _forget_active(self, rid: str) -> None:
        """The run is gone (deleted here, or a `-id` line): drop its record."""
        with self._active_lock:
            self._active.pop(rid, None)
            self._active_seq += 1
            self._change_ring.note(self._active_seq, rid)

    # --- change signal ---------------------------------------------------

    def change_token(self) -> Tuple[int, int]:
        """`(creation-log size, active-index counter)`: equal tokens mean no run of this folder
        was saved, created or deleted in between, by this store (the counter) or by another
        process's create / delete / became-active line (the log). One stat, no file read.

        Not seen: another PROCESS re-saving a run that stays active (no log line: the
        single-writer contract above, one ticking process per folder). A run that ends, or a
        finished run re-saved, writes `=id` (a store of this version)."""
        size = self._journal_size()
        with self._active_lock:
            seq = self._active_seq
        return (int(size if size is not None else -1), int(seq))

    def changed_run_ids(self, since: Any) -> Optional[set]:
        """The run ids this store saved, created, deleted or re-read with a new status after the
        token `since`; None when that is unknown (the creation log moved, which may be another
        process, or the record of changes no longer reaches back to `since`)."""
        try:
            size0, seq0 = int(since[0]), int(since[1])
        except Exception:
            return None
        if self._journal_size() != size0 and not self._journal_moved_only_by_me(size0):
            return None
        with self._active_lock:
            return self._change_ring.since(seq0)

    def _journal_moved_only_by_me(self, size0: int) -> bool:
        """Whether the creation log grew from `size0` only by this store's own appends."""
        with self._journal_lock:
            own = list(self._journal_own_appends)
        size = self._journal_size()
        if size is None:
            return False
        cur = size0
        for start, end in own:
            if start == cur:
                cur = end
        return cur == size

    def _drop_active_if_unchanged(self, rid: str, gen0: int) -> None:
        """A verified read said "not active": drop the member unless something
        noted it since the read began, or this store is publishing it."""
        with self._active_lock:
            e = self._active.get(rid)
            if e is None or e.publishing > 0 or e.gen != gen0:
                return
            self._active_seq += 1
            e.gen = self._active_seq
            e.active = False
            self._change_ring.note(self._active_seq, rid)

    def _scan_active_ids(self) -> set[str]:
        """ONE directory scan: the ids whose memo/scan fields say RUNNING or
        WAITING (memo-first: unchanged files cost one stat, never a parse).

        Once the memo is complete this pass is the trusted index's
        RECONCILIATION: every file whose memo entry it had to change (new,
        changed, or gone without a log line), changed more than
        RUN_INDEX_IN_FLIGHT_S ago, is a difference the index missed: repaired
        here, counted (`run_index_stats()["reconcile_repairs"]`) and logged."""
        self._load_scan_sidecar_once()
        complete = self._memo_complete
        t0 = time.monotonic()
        settled_ns = time.time_ns() - int(float(self.RUN_INDEX_IN_FLIGHT_S) * 1e9)
        repaired: List[str] = []
        found: set[str] = set()
        seen: set[str] = set()
        for p in self._base.glob("run_*.json"):
            rid = self._run_id_from_path(p)
            if rid:
                seen.add(rid)
            if complete:
                with self._scan_memo_lock:
                    prev = self._scan_memo.get(rid)
                prev_tok = tuple(prev[0]) if prev is not None else None
            fields = self._scan_fields(p)
            if complete:
                with self._scan_memo_lock:
                    cur = self._scan_memo.get(rid)
                cur_tok = tuple(cur[0]) if cur is not None else None
                if cur_tok is not None and cur_tok != prev_tok and cur_tok[0] < settled_ns:
                    repaired.append(rid)
            if fields is None:
                continue
            if self._is_active_status(fields.get("status")):
                found.add(rid)
        # Deletions announced by the log while the pass ran are not misses.
        self._sync_indexes()
        dropped = self._scan_memo_prune(seen)
        if complete:
            repaired.extend(dropped)
            self._note_reconcile(repaired, files=len(seen), seconds=time.monotonic() - t0)
        return found

    def _note_reconcile(self, repaired: List[str], *, files: int, seconds: float) -> None:
        with self._index_stats_lock:
            st = self._index_stats
            st["reconciles"] += 1
            st["reconcile_repairs"] += len(repaired)
            st["last_reconcile"] = {
                "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "files": int(files),
                "seconds": round(float(seconds), 3),
                "repairs": len(repaired),
                "sample": sorted(repaired)[:5],
            }
        if repaired:
            logger.warning(
                "JsonFileRunStore: run index reconciliation repaired %d entr%s the index had missed "
                "(a writer that bypassed the creation log?): %s",
                len(repaired),
                "y" if len(repaired) == 1 else "ies",
                ", ".join(sorted(repaired)[:5]),
            )

    def _build_active_index(self) -> bool:
        """Reconcile the records with ONE directory scan, taken OUTSIDE the
        lock so saves never wait behind it. Records noted while the scan ran
        keep their note; every other record takes the scan's verdict (adds
        AND removes); inactive records with nothing else to say are pruned.
        Returns False when another scan was already in flight."""
        with self._active_lock:
            if self._active_building:
                return False
            self._active_building = True
            start = self._active_seq
        # What the folder and the log looked like when this pass began: the next
        # reconciliation is due only when one of them moved (trusted mode).
        marks = (self._journal_size(), self._dir_mtime())
        try:
            found = self._scan_active_ids()
        except Exception:
            with self._active_lock:
                self._active_building = False
            raise
        with self._active_lock:
            for rid in found:
                e = self._active_entry(rid)
                if e.gen <= start:
                    e.active = True
            for rid, e in list(self._active.items()):
                if rid not in found and e.gen <= start:
                    e.active = False
                if not e.active and e.publishing == 0 and e.saved_token is None and e.gen <= start:
                    self._active.pop(rid, None)
            # A log rewritten while the pass ran (offset reset): rebuild on the next call.
            self._active_built = self._journal_offset is not None
            self._memo_complete = self._active_built
            self._active_built_at = time.monotonic()
            self._reconcile_marks = marks
            self._reconcile_last_full = self._active_built_at
            self._active_building = False
        self._maybe_persist_scan_memo()
        return True

    def _ensure_active_index(self) -> None:
        """Build once (the journal keeps it current afterwards) and kick the
        periodic background rescan when due. `warm_session_index()` builds it
        at host boot so the first poll does not pay the scan."""
        if self._active_built:
            self._sync_indexes()
            if self._active_built:
                self._maybe_rescan_active_index()
                return
        self._mark_journal_start()
        if not self._build_active_index():
            # Another thread is building: wait for it rather than serve nothing.
            deadline = time.monotonic() + 300.0
            while time.monotonic() < deadline:
                with self._active_lock:
                    if self._active_built and not self._active_building:
                        return
                time.sleep(0.005)

    def rescan_active_index(self) -> None:
        """Reconcile the active index with the directory NOW, in the caller's
        thread (the proof harness; a host that wants disk truth before a
        decision). Builds the index when it is not built yet."""
        self._mark_journal_start()
        self._build_active_index()

    def _maybe_rescan_active_index(self, *, dir_mtime: Any = False) -> None:
        """Reconcile with disk in a daemon thread at most every
        ACTIVE_INDEX_RESCAN_S: catches a writer that touched a file without a
        journal line. Never blocks a caller.

        Trusted mode: only when the folder's mtime or the creation log moved
        since the last pass began (a create, replace or delete moves the
        folder), and at least every RUN_INDEX_FULL_RECONCILE_S regardless, so
        an idle store stats no run file. `dir_mtime`: the caller's fresh stat
        of the folder (False: stat it here when due)."""
        with self._active_lock:
            if self._active_rescan_inflight or self._active_building:
                return
            now = time.monotonic()
            if now - self._active_built_at < float(self.ACTIVE_INDEX_RESCAN_S):
                return
            marks0 = self._reconcile_marks
            full_due = now - self._reconcile_last_full >= float(self.RUN_INDEX_FULL_RECONCILE_S)
        if self._trust_index and not full_due and marks0 is not None:
            size = self._journal_last_size if dir_mtime is not False else self._journal_size()
            dm = self._dir_mtime() if dir_mtime is False else dir_mtime
            if (size, dm) == tuple(marks0) and size is not None and dm is not None:
                return
        with self._active_lock:
            if self._active_rescan_inflight or self._active_building:
                return
            self._active_rescan_inflight = True

        def _rescan() -> None:
            try:
                self._build_active_index()
            except Exception:  # noqa: BLE001 - a failed rescan is retried next time
                logger.warning("JsonFileRunStore: active-index rescan failed", exc_info=True)
            finally:
                with self._active_lock:
                    self._active_rescan_inflight = False

        threading.Thread(target=_rescan, name="runstore-active-rescan", daemon=True).start()

    def _active_candidate_ids(self, *, status: Any = None) -> Optional[List[str]]:
        """The ids to rank for an active-status listing (members plus the ids
        this store is publishing), or None when the plain scan must run (index
        off, or a non-active status)."""
        if not self._use_active_index:
            return None
        if status is not None and not self._is_active_status(status):
            return None
        self._ensure_active_index()
        with self._active_lock:
            return [rid for rid, e in self._active.items() if e.active or e.publishing > 0]

    def _active_candidates(self, status: Any) -> Optional[List[Path]]:
        if status is None:
            return None
        ids = self._active_candidate_ids(status=status)
        return None if ids is None else [self._base / f"run_{rid}.json" for rid in ids]

    def _verify_active_fields(self, p: Path, status: Any) -> Optional[Dict[str, Any]]:
        """A candidate's current fields; a member whose file is gone or no
        longer has the asked status is dropped from the set (self-heal) —
        unless a save noted it while the file was being read (the generation
        moved) or this store is publishing it: then the read was stale."""
        rid = self._run_id_from_path(p)
        with self._active_lock:
            e = self._active.get(rid)
            gen0 = e.gen if e is not None else 0
        fields = self._scan_fields(p)
        want = str(getattr(status, "value", status))
        if fields is None or fields.get("status") != want:
            if fields is None or not self._is_active_status(fields.get("status")):
                self._drop_active_if_unchanged(rid, gen0)
            return None
        return fields

    def _drop_if_unchanged(self, rid: str) -> None:
        """A candidate's file could not be stat()ed: drop the member unless it
        is there now, this store is publishing it, or something noted it since."""
        with self._active_lock:
            e = self._active.get(rid)
            if e is None or e.publishing > 0:
                return
            gen0 = e.gen
        try:
            self._path(rid).stat()
            return
        except Exception:
            pass
        self._drop_active_if_unchanged(rid, gen0)

    def _active_publish_begin(self, rid: str) -> None:
        with self._active_lock:
            self._active_entry(rid).publishing += 1

    def _active_publish_end(self, rid: str, *, status: Any, token: Optional[tuple]) -> None:
        """The write is over (published or failed): note the status when it
        was published (`token` is the file's identity), and remember the token
        of an active save so the next save can tell "unchanged since"."""
        with self._active_lock:
            e = self._active_entry(rid)
            e.publishing = max(0, e.publishing - 1)
            if token is not None:
                self._active_seq += 1
                e.gen = self._active_seq
                e.active = self._is_active_status(status)
                e.saved_token = token if e.active else None
                self._change_ring.note(self._active_seq, rid)

    def _own_write_mark(self) -> None:
        with self._index_stats_lock:
            self._own_writes += 1

    def save(self, run: RunState) -> None:
        self._own_write_mark()
        try:
            self._save(run)
        finally:
            self._own_write_mark()

    def _save(self, run: RunState) -> None:
        p = self._path(run.run_id)
        rid = str(run.run_id)
        try:
            token_before: Optional[tuple] = self._stat_token(p.stat())
        except FileNotFoundError:
            token_before = None
        except Exception:
            token_before = (0, 0, 0)
        is_new = token_before is None
        now_active = self._is_active_status(run.status)
        if is_new:
            self._journal_append(f"+{run.run_id}")  # before publication (J53-1)
        elif now_active:
            with self._active_lock:
                e = self._active.get(rid)
                unchanged_since_my_active_save = e is not None and e.saved_token is not None and e.saved_token == token_before
            if not unchanged_since_my_active_save:
                # A run that BECOMES active through this store (resumed,
                # re-opened, first save after a restart, or re-saved after
                # another writer touched the file) is announced as `~id`
                # BEFORE publication, like `+id`: a reader keeps the entry
                # pending until the file reads active, so it can neither read
                # stale content behind the line nor miss a file that landed
                # before the line. The common per-tick save of a run this
                # store itself saved as active adds no line.
                self._journal_append(f"~{rid}")
        if now_active:
            self._active_publish_begin(rid)
        # Atomic write to prevent corrupted/partial JSON when multiple threads/processes
        # (e.g. WS tick loop + UI pause/cancel) write the same run file concurrently.
        tmp = p.with_name(f"{p.name}.{uuid.uuid4().hex}.tmp")
        published = False
        try:
            with tmp.open("w", encoding="utf-8") as f:
                # Compact + by-reference serialization (backlog 0067): this is
                # the most frequent durable write in the system (~5 per agent
                # cycle); `indent=2` cost +72% CPU and ~35% bytes, and
                # `asdict` deep-copied the whole vars tree for nothing under
                # the single-writer contract documented on this class.
                f.write(dumps_compact(runstate_to_dict(run)))
            tmp.replace(p)
            published = True
            if not is_new and not now_active:
                # A run that is not active changed its file (it finished, or a finished
                # run was re-saved: archive mark, title): `=id` AFTER publication, so a
                # reader that applies it reads the new file. The trusted index of every
                # other store object learns it from this line (nothing else announces it).
                self._journal_append(f"={rid}")
        finally:
            # Best-effort cleanup if replace() failed.
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            if not published and now_active:
                self._active_publish_end(rid, status=run.status, token=None)
        self._update_children_index_on_save(run)
        self._update_event_wait_index_on_save(run)
        self._update_mailbox_index_on_save(run)
        self._update_session_index_on_save(run)
        try:
            token = self._stat_token(p.stat())
        except Exception:
            token = (0, 0, 0)
        if now_active:
            self._active_publish_end(rid, status=run.status, token=token)
        else:
            self._note_active_status(rid, run.status)
            with self._active_lock:
                e = self._active.get(rid)
                if e is not None:
                    e.saved_token = None
        self._xproc_note_own(rid, token, now_active)
        if token[0] > 0:
            self._cache_put(str(run.run_id), token, run, own=True)

    def create_if_absent(self, run: RunState) -> Tuple[RunState, bool]:
        """Create the run file only if absent: a fully written temp file in the
        SAME directory is published with `os.link` (atomic; fails with
        FileExistsError when the id is taken), then the temp name is unlinked.
        Readers therefore never see a partial file and an existing run is
        never overwritten — there is no replace() fallback. Filesystems
        without hard links raise instead of degrading.

        The in-memory children/event-wait indexes are updated afterwards; they
        are rebuildable (a restart rebuilds them from disk) and are refreshed
        from the existing run when the id was taken, which repairs them after
        a crash between publication and index update. Process-crash recovery
        only: no fsync, so power-loss durability is not claimed.
        """
        self._own_write_mark()
        try:
            return self._create_if_absent(run)
        finally:
            self._own_write_mark()

    def _create_if_absent(self, run: RunState) -> Tuple[RunState, bool]:
        p = self._path(run.run_id)
        tmp = p.with_name(f"{p.name}.{uuid.uuid4().hex}.tmp")
        created = False
        rid = str(run.run_id)
        now_active = self._is_active_status(run.status)
        if now_active:
            self._active_publish_begin(rid)
        try:
            with tmp.open("w", encoding="utf-8") as f:
                f.write(dumps_compact(runstate_to_dict(run)))
            if not p.exists():
                self._journal_append(f"+{run.run_id}")  # before publication (J53-1)
            try:
                os.link(tmp, p)
                created = True
            except FileExistsError:
                created = False
        finally:
            with contextlib.suppress(FileNotFoundError):
                tmp.unlink()
            if not created and now_active:
                self._active_publish_end(rid, status=run.status, token=None)
        if created:
            self._update_children_index_on_save(run)
            self._update_event_wait_index_on_save(run)
            self._update_mailbox_index_on_save(run)
            self._update_session_index_on_save(run)
            try:
                token = self._stat_token(p.stat())
            except Exception:
                token = (0, 0, 0)
            if now_active:
                self._active_publish_end(rid, status=run.status, token=token)
            else:
                self._note_active_status(rid, run.status)
            self._xproc_note_own(rid, token, now_active)
            if token[0] > 0:
                self._cache_put(str(run.run_id), token, run, own=True)
            return run, True
        existing = self._load_from_path(p)
        if existing is None:
            raise ValueError(f"run file {p.name} exists but is unreadable; refusing to overwrite it")
        self._update_children_index_on_save(existing)
        self._update_event_wait_index_on_save(existing)
        self._update_mailbox_index_on_save(existing)
        self._update_session_index_on_save(existing)
        self._note_active_status(str(existing.run_id), existing.status)
        verify_run_identity(existing, run)
        return existing, False

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
        self._own_write_mark()
        try:
            if existed:
                self._journal_append(f"-{rid}")  # before the unlink (J53-1)
                p.unlink()
        finally:
            self._own_write_mark()
            self._drop_from_children_index(rid)
            self._drop_from_event_wait_index(rid)
            self._drop_from_mailbox_index(rid)
            self._drop_from_session_index(rid)
            self._forget_active(rid)
            self._xproc_note_own(rid, None, False)
            with self._run_cache_lock:
                self._run_cache.pop(rid, None)
            self._memo_pop(rid)
        return bool(existed)

    def _load_from_path(self, p: Path, *, seq0: Optional[int] = None) -> Optional[RunState]:
        """Load a RunState from a file path."""
        rid_hint = self._run_id_from_path(p)
        if seq0 is None:
            with self._scan_memo_lock:
                seq0 = self._memo_version
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
                # The token of the file actually read (a replace after the stat
                # above must not pair the new content with the old token).
                with contextlib.suppress(Exception):
                    token = self._stat_token(os.fstat(f.fileno()))
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
            self._cache_put(rid, token, run, memo_rid=rid_hint or rid, seq0=seq0)
        return run

    # `_iter_all_runs` was removed 2026-08-19 (operator's session-reload
    # investigation): it full-parsed every run file — 7.3s on the live
    # 7.9k-run / 1.8GB directory — and its one caller (the children-index
    # build) now reads the memoized scan fields instead. A whole-store
    # RunState walk has no sub-O(store) implementation; anything that
    # thinks it needs one should go through `_scan_fields`.

    # --- trusted run index ---------------------------------------------------

    @property
    def trust_index(self) -> bool:
        """True: listings answer from the in-memory run index ("trusted"); False: from the
        directory scan, compared with the index ("verify", shadow mode)."""
        return self._trust_index

    def _dir_mtime(self) -> Optional[int]:
        try:
            return int(self._base.stat().st_mtime_ns)
        except OSError:
            return None

    def _trusted_ready(self) -> Optional[int]:
        """Prepare the in-memory index for one listing: built (one pass, in the caller's
        thread, only the very first time), the creation log applied, a reconciliation
        kicked when due, the ACTIVE entries re-read when the folder moved. Returns the
        folder's mtime (the listing's stability mark), or None when the index cannot
        answer (not complete yet, being built by another thread, or no active index)."""
        if not self._use_active_index:
            return None
        with self._active_lock:
            built, building = self._active_built, self._active_building
        if not built:
            if building:
                return None  # the boot build is running: this call scans, as before
            self._mark_journal_start()
            if not self._build_active_index():
                return None
        else:
            self._sync_indexes()
        if not (self._active_built and self._memo_complete):
            return None
        dm = self._dir_mtime()
        self._maybe_rescan_active_index(dir_mtime=dm)
        self._refresh_active_memo(dm)
        return dm if dm is not None else 0

    def _refresh_active_memo(self, dir_mtime: Optional[int]) -> None:
        """Re-read the memo's ACTIVE entries (one stat each) when the folder's mtime moved
        since the last check: another process ticking a run re-saves it with no log line,
        and every save (tmp + replace) moves the folder. Finished runs never change without
        a log line (`=id`), so they cost nothing."""
        if dir_mtime is not None:
            with self._scan_memo_lock:
                if self._memo_active_checked_dir == dir_mtime:
                    return
                self._memo_active_checked_dir = dir_mtime
        with self._scan_memo_lock:
            rids = list(self._memo_active)
            tokens = {rid: self._scan_memo[rid][0] for rid in rids if rid in self._scan_memo}
        for rid in rids:
            p = self._path(rid)
            try:
                tok = self._stat_token(p.stat())
            except FileNotFoundError:
                old = tokens.get(rid)
                if old is not None:
                    self._memo_pop(rid, token=old)
                continue
            except OSError:
                continue
            if tokens.get(rid) is not None and tuple(tokens[rid]) == tok:
                continue
            self._scan_fields(p)

    def _trusted_ranked(self) -> list:
        """Every listable memo entry `(mtime_ns, file name, run id, fields, token)`, newest first
        (the scan's ranking: mtime, then the file name), cached on the memo version."""
        with self._scan_memo_lock:
            version = self._memo_version
            cached = self._ranked_cache
            if cached is not None and cached[0] == version:
                return cached[1]
            items = [
                (tok[0], f"run_{rid}.json", rid, fields, tok)
                for rid, (tok, fields) in self._scan_memo.items()
                if not fields.get("__unparseable")
            ]
        items.sort(key=lambda x: (x[0], x[1]), reverse=True)
        with self._scan_memo_lock:
            if self._memo_version == version:
                self._ranked_cache = (version, items)
        return items

    def _trusted_matches(
        self,
        *,
        status: Any = None,
        wait_reason: Any = None,
        workflow_id: Optional[str] = None,
        root_only: bool = False,
        automation_id: Any = None,
        role: Any = None,
        session_kind: Any = None,
        oldest_first: bool = False,
    ):
        """`(token, run id, fields)` of the memo entries passing a listing's filters, in the
        scan's rank order (lazy)."""
        ranked = self._trusted_ranked()
        want_status = str(getattr(status, "value", status)) if status is not None else None
        want_wait = str(getattr(wait_reason, "value", wait_reason)) if wait_reason is not None else None
        want_wf = str(workflow_id) if workflow_id is not None else None
        attributed = automation_id is not None or role is not None or session_kind is not None
        for _mtime_ns, _name, rid, fields, token in (reversed(ranked) if oldest_first else ranked):
            if want_status is not None and fields.get("status") != want_status:
                continue
            if want_wf is not None and fields.get("workflow_id") != want_wf:
                continue
            if want_wait is not None and fields.get("wait_reason") != want_wait:
                continue
            if root_only and not is_turn_root(parent_run_id=fields.get("parent_run_id"), role=fields.get("role")):
                continue
            if attributed and not row_matches(fields, automation_id=automation_id, role=role, session_kind=session_kind):
                continue
            yield token, rid, fields

    @staticmethod
    def _index_row(rid: str, fields: Dict[str, Any]) -> Dict[str, Any]:
        # Copy dict values one level deep (adversary P2-2): rows alias the
        # memo's nested run_lifecycle dict — a consumer mutating
        # row["run_lifecycle"] would poison every future index call until
        # the file's identity changes.
        return {
            "run_id": rid,
            **{k: (dict(v) if isinstance(v, dict) else v) for k, v in fields.items() if not k.startswith("__")},
        }

    def _trusted_index_rows(self, *, limit: int, oldest_first: bool = False, **filters: Any) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for _m, rid, fields in self._trusted_matches(oldest_first=oldest_first, **filters):
            out.append(self._index_row(rid, fields))
            if len(out) >= limit:
                break
        out.sort(key=lambda r: str(r.get("updated_at") or ""), reverse=not oldest_first)
        return out[:limit]

    def _trusted_run_ids(self, *, limit: int, **filters: Any) -> List[str]:
        """What a trusted `list_runs` would return, as ids (for the shadow comparison)."""
        picked: List[tuple[str, str]] = []
        for _m, rid, fields in self._trusted_matches(**filters):
            picked.append((str(fields.get("updated_at") or ""), str(fields.get("__run_id") or rid)))
            if len(picked) >= limit:
                break
        picked.sort(key=lambda x: x[0], reverse=True)
        return [rid for _u, rid in picked[:limit]]

    def _load_trusted(self, rid: str, token: Any) -> Optional[RunState]:
        """A listed run: the cached RunState when it is the file the memo describes (no
        stat), else a load from disk."""
        with self._run_cache_lock:
            cached = self._run_cache.get(rid)
            if cached is not None and token is not None and tuple(cached[0]) == tuple(token):
                self._run_cache.move_to_end(rid)
                return cached[1]
        return self._load_from_path(self._path(rid))

    def _shadow_mark(self) -> tuple:
        with self._index_stats_lock:
            own = self._own_writes
        return (self._journal_size(), self._dir_mtime(), own)

    def _shadow_compare(self, kind: str, mark0: tuple, trusted: Any, scanned: Any) -> None:
        """Verify mode: compare the in-memory answer with the scan's. A comparison that
        overlapped a write (the log, the folder or this store's own writes moved) is
        counted as unstable, never as a mismatch."""
        stable = self._shadow_mark() == mark0
        with self._index_stats_lock:
            st = self._index_stats
            if not stable:
                st["shadow_unstable"] += 1
                return
            st["shadow_compared"] += 1
            if trusted == scanned:
                return
            st["shadow_mismatches"] += 1
            t_ids = [r.get("run_id") if isinstance(r, dict) else r for r in trusted]
            s_ids = [r.get("run_id") if isinstance(r, dict) else r for r in scanned]
            sample = {
                "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "listing": kind,
                "only_in_index": [x for x in t_ids if x not in set(s_ids)][:5],
                "only_in_scan": [x for x in s_ids if x not in set(t_ids)][:5],
                "same_ids": set(t_ids) == set(s_ids),
            }
            st["last_mismatch"] = sample
        logger.warning("JsonFileRunStore: run index (verify mode) differs from the directory scan: %s", sample)

    def _count_listing(self, trusted: bool) -> None:
        with self._index_stats_lock:
            self._index_stats["trusted_listings" if trusted else "scanned_listings"] += 1

    def run_index_stats(self) -> Dict[str, Any]:
        """The trusted run index at a glance (a copy). `mode`: "trusted" or "verify";
        `complete`: one full pass done (the index can answer); `entries` / `active_entries`:
        run files known / of them RUNNING or WAITING; `trusted_listings` / `scanned_listings`:
        listings answered from memory / by the directory scan; verify mode only:
        `shadow_compared` (stable comparisons of the two answers), `shadow_mismatches` (of
        them, different: the number to watch, 0 = the index would have answered the same),
        `shadow_unstable` (overlapped a write, not a verdict), `last_mismatch`; both modes:
        `reconciles`, `reconcile_repairs` (differences the background reconciliation found
        and repaired: 0 = nothing bypassed the creation log) and `last_reconcile`."""
        with self._scan_memo_lock:
            entries = len(self._scan_memo)
            active = len(self._memo_active)
        with self._index_stats_lock:
            st = dict(self._index_stats)
            for k in ("last_mismatch", "last_reconcile"):
                if isinstance(st.get(k), dict):
                    st[k] = dict(st[k])
        return {
            "mode": "trusted" if self._trust_index else "verify",
            "complete": bool(self._memo_complete and self._active_built),
            "entries": entries,
            "active_entries": active,
            **st,
        }

    def reconcile_run_index(self) -> Dict[str, Any]:
        """Run the reconciliation NOW in the caller's thread (tests, a host that wants disk
        truth before a decision) and return `run_index_stats()`."""
        self.rescan_active_index()
        return self.run_index_stats()

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
        - COST: a RUNNING/WAITING listing (no session scope) ranks only the
          ACTIVE index's members, one stat each (see the class notes); any
          other call globs+stats the whole directory (~31 ms warm at 3k
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
        # ACTIVE INDEX: a RUNNING/WAITING listing ranks only the index's
        # candidates (each re-verified below); any other listing is answered
        # by the TRUSTED RUN INDEX (memory, no stat), or scans in verify mode.
        candidates = self._active_candidates(status)
        shadow: Optional[tuple] = None
        if candidates is None:
            mark0 = self._shadow_mark() if not self._trust_index else None
            if self._trusted_ready() is not None:
                filters = dict(status=status, wait_reason=wait_reason, workflow_id=workflow_id)
                if self._trust_index:
                    self._count_listing(True)
                    results0: List[RunState] = []
                    for token, rid, _fields in self._trusted_matches(**filters):
                        run = self._load_trusted(rid, token)
                        if run is None:
                            continue
                        results0.append(run)
                        if len(results0) >= lim:
                            break
                    results0.sort(key=lambda r: r.updated_at or "", reverse=True)
                    self._maybe_persist_scan_memo()
                    return results0[:lim]
                shadow = (mark0, self._trusted_run_ids(limit=lim, **filters))
            self._count_listing(False)
        ranked: list[tuple[int, Path]] = []
        for p in candidates if candidates is not None else self._base.glob("run_*.json"):
            try:
                st = p.stat()
                mtime_ns = int(getattr(st, "st_mtime_ns", 0) or 0)
            except Exception:
                if candidates is not None:
                    self._drop_if_unchanged(self._run_id_from_path(p))
                continue
            ranked.append((mtime_ns, p))
        # mtime, then the file name: the same order whether the candidates came
        # from the index or from the glob (two files saved in the same
        # nanosecond rank alike on both paths).
        ranked.sort(key=lambda x: (x[0], x[1].name), reverse=True)
        if candidates is None:
            self._scan_memo_prune({self._run_id_from_path(p) for _m, p in ranked})

        results: List[RunState] = []
        for _mtime_ns, p in ranked:
            # Memo-first filtering (2026-07-15 scan-memo): non-matching
            # unchanged files cost one stat, never a parse — the runner's
            # RUNNING polls used to re-parse every multi-MB terminal file
            # the 512-entry RunState LRU had just evicted.
            fields = self._verify_active_fields(p, status) if candidates is not None else self._scan_fields(p)
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
        if shadow is not None:
            self._shadow_compare("list_runs", shadow[0], shadow[1], [str(r.run_id) for r in results[:lim]])
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
        automation_id: Any = None,
        role: Any = None,
        session_kind: Any = None,
    ) -> List[Dict[str, Any]]:
        """List lightweight run index rows without depending on full RunState consumers.

        `root_only=True` returns turn roots (see `QueryableRunIndexStore`)."""
        lim = max(1, int(limit or 100))
        self._load_scan_sidecar_once()
        sid = str(session_id or "").strip() if session_id is not None else None
        candidates: Optional[List[Path]] = None
        shadow: Optional[tuple] = None
        if sid is not None:
            # Session-scoped: candidates from the session index, O(session).
            self._ensure_session_index()
            with self._session_index_lock:
                candidates = list(((self._session_index or {}).get(sid) or {}).keys())
            paths = [self._base / f"run_{rid}.json" for rid in candidates]
        else:
            # ACTIVE INDEX: a RUNNING/WAITING page ranks the index's candidates
            # (each re-verified below) instead of the whole directory.
            candidates = self._active_candidates(status)
            if candidates is None:
                # TRUSTED RUN INDEX: any other page comes from memory (no glob, no
                # stat); verify mode scans and compares (shadow).
                mark0 = self._shadow_mark() if not self._trust_index else None
                if self._trusted_ready() is not None:
                    filters = dict(
                        status=status, workflow_id=workflow_id, root_only=bool(root_only),
                        automation_id=automation_id, role=role, session_kind=session_kind,
                    )
                    trusted_rows = self._trusted_index_rows(limit=lim, oldest_first=bool(oldest_first), **filters)
                    if self._trust_index:
                        self._count_listing(True)
                        self._maybe_persist_scan_memo()
                        return trusted_rows
                    shadow = (mark0, trusted_rows)
                self._count_listing(False)
            paths = candidates if candidates is not None else list(self._base.glob("run_*.json"))
        indexed = sid is None and candidates is not None
        ranked: list[tuple[int, Path]] = []
        for p in paths:
            try:
                st = p.stat()
                mtime_ns = int(getattr(st, "st_mtime_ns", 0) or 0)
            except Exception:
                if indexed:
                    self._drop_if_unchanged(self._run_id_from_path(p))
                continue
            ranked.append((mtime_ns, p))
        # mtime ranking approximates updated_at; oldest_first inverts so a
        # stall query's window keeps the OLDEST waits (0054 adversary P1-1).
        # The file name breaks ties so the index path and the glob path agree.
        ranked.sort(key=lambda x: (x[0], x[1].name), reverse=not oldest_first)
        if sid is None and not indexed:
            self._scan_memo_prune({self._run_id_from_path(p) for _m, p in ranked})

        out: List[Dict[str, Any]] = []

        for _mtime_ns, p in ranked:
            # Memo-first (2026-07-15 scan-memo): index rows ARE the memo
            # fields — an index page over unchanged files parses nothing.
            fields = self._verify_active_fields(p, status) if indexed else self._scan_fields(p)
            if fields is None:
                continue
            if status is not None and fields.get("status") != str(getattr(status, "value", status)):
                continue
            if workflow_id is not None and fields.get("workflow_id") != str(workflow_id):
                continue
            if sid is not None and str(fields.get("session_id") or "").strip() != sid:
                continue
            if bool(root_only) and not is_turn_root(parent_run_id=fields.get("parent_run_id"), role=fields.get("role")):
                continue
            if not row_matches(fields, automation_id=automation_id, role=role, session_kind=session_kind):
                continue

            out.append(self._index_row(self._run_id_from_path(p), fields))
            if len(out) >= lim:
                break

        out.sort(key=lambda r: str(r.get("updated_at") or ""), reverse=not oldest_first)
        self._maybe_persist_scan_memo()
        if shadow is not None:
            self._shadow_compare("list_run_index", shadow[0], shadow[1], out[:lim])
        return out[:lim]

    def latest_occurrence_row(self, automation_id: str) -> Optional[Dict[str, Any]]:
        """Index row of the automation's highest-numbered occurrence (newest
        attempt), or None — answered from the scan memo (one stat per file,
        no parse for unchanged files)."""
        return latest_occurrence_of(
            self.list_run_index(automation_id=str(automation_id), role="occurrence", limit=1_000_000)
        )

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
        # The runner asks this on every poll: WAITING members of the active
        # index are the only candidates (each re-verified), else the scan.
        ids = self._active_candidate_ids(status=RunStatus.WAITING)
        if ids is None:
            paths = list(self._base.glob("run_*.json"))
            self._scan_memo_prune({self._run_id_from_path(p) for p in paths})
        else:
            paths = [self._base / f"run_{rid}.json" for rid in ids]
        for p in paths:
            fields = self._verify_active_fields(p, RunStatus.WAITING) if ids is not None else self._scan_fields(p)
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
        self._refresh_event_indexes()
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
        self._refresh_event_indexes()
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

    def list_mailbox_runs(self, *, names: List[str], limit: int = 10_000) -> List[RunState]:
        """ACTIVE runs (RUNNING/WAITING) declaring any of `names` as their
        `events_mailbox` — O(mailbox holders), never capped by a recency window.

        Every candidate is re-loaded and re-verified against disk (a run that
        finished or dropped the mailbox since is re-filed and skipped)."""
        wanted = {str(n).strip() for n in (names or []) if isinstance(n, str) and str(n).strip()}
        if not wanted:
            return []
        self._ensure_mailbox_index()
        self._refresh_event_indexes()
        with self._event_index_lock:
            index = self._mailbox_index or {}
            candidate_ids = sorted({rid for n in wanted for rid in index.get(n, set())})
        results: List[RunState] = []
        for rid in candidate_ids:
            run = self.load(rid)
            if run is None:
                self._drop_from_mailbox_index(rid)
                continue
            if not (self._mailboxes_of(run) & wanted):
                self._update_mailbox_index_on_save(run)
                continue
            results.append(run)
        results.sort(key=lambda r: r.updated_at or "", reverse=True)
        return results[: max(1, int(limit or 10_000))]

    def next_wait_deadline(self) -> Optional[str]:
        """The earliest `wait_until` among WAITING runs (UNTIL, or EVENT with
        a deadline), or None — over EVERY waiting run, from the scan memo (one
        stat per unchanged file, no parse). The scheduler's sleep horizon:
        a recency-limited listing cannot answer it (design round-1 F3)."""
        self._load_scan_sidecar_once()
        ids = self._active_candidate_ids(status=RunStatus.WAITING)
        if ids is None:
            paths = list(self._base.glob("run_*.json"))
        else:
            paths = [self._base / f"run_{rid}.json" for rid in ids]
        best: Optional[str] = None
        for p in paths:
            fields = self._verify_active_fields(p, RunStatus.WAITING) if ids is not None else self._scan_fields(p)
            if fields is None or fields.get("status") != RunStatus.WAITING.value:
                continue
            if fields.get("wait_reason") not in (WaitReason.UNTIL.value, WaitReason.EVENT.value):
                continue
            until = fields.get("wait_until")
            if until and (best is None or str(until) < best):
                best = str(until)
        self._maybe_persist_scan_memo()
        return best

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


def _jsonl_line_records(line: Any) -> List[Dict[str, Any]]:
    """The records one ledger line holds, exactly as `JsonlLedgerStore.list` reads it: one JSON object per line,
    or (crash / manual concatenation) several objects recovered with `raw_decode`; a non-object or unparseable
    line holds none."""
    text = line.decode("utf-8", errors="replace") if isinstance(line, (bytes, bytearray)) else str(line)
    text = text.strip()
    if not text:
        return []
    try:
        record = json.loads(text)
        return [record] if isinstance(record, dict) else []
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    recovered: List[Dict[str, Any]] = []
    i = 0
    while i < len(text):
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text):
            break
        try:
            obj, end = decoder.raw_decode(text, idx=i)
        except json.JSONDecodeError:
            break
        if isinstance(obj, dict):
            recovered.append(obj)
        i = end
    return recovered


class _JsonlRowIndex:
    """Byte offset of the line holding each record of one ledger file (ADR 0011 row address, JSON-file backend).

    `offsets[seq - 1]` is the offset of the line that holds record `seq`; a concatenated line (several records on
    one line, see `_jsonl_line_records`) repeats its offset, so a record's position inside its line is
    `seq - 1 - bisect_left(offsets, offset)`. `size` is the byte length indexed (newline-terminated lines only);
    `ino` the file's inode, so a deleted-and-recreated ledger is re-indexed from scratch."""

    __slots__ = ("ino", "size", "offsets", "persisted_size")

    def __init__(self, ino: int) -> None:
        self.ino = ino
        self.size = 0
        self.offsets = array("q")
        self.persisted_size = 0  # the indexed size the `.idx` file on disk covers


# Persisted row index (`ledger_<run_id>.idx`, beside the ledger): magic, then (inode, indexed size, row count) as
# native int64s, then the offsets. Byte order is in the magic, so a file moved across architectures is rebuilt.
_IDX_MAGIC = b"ARLIDX1" + (b"L" if sys.byteorder == "little" else b"B")


class JsonlLedgerStore(LedgerStore):
    # Row-index cache bound (ADR 0011 / backlog 0856 performance): runs whose line offsets are kept (8 bytes per
    # record), least recently used evicted. A 10k-row ledger costs ~80KB.
    _ROW_INDEX_MAX_RUNS = 128
    # The persisted `.idx` is rewritten (atomically, whole) once the newly indexed ledger bytes reach
    # max(_ROW_INDEX_PERSIST_BYTES, 1/_ROW_INDEX_PERSIST_FRACTION of the ledger): a fresh process then parses at most
    # that tail instead of the whole file, and the rewrites total a small multiple of the final index size.
    _ROW_INDEX_PERSIST_BYTES = 256 * 1024
    _ROW_INDEX_PERSIST_FRACTION = 8
    # First backward window of `find_started_record` (doubles until the STARTED row is found or the file start).
    _STARTED_SCAN_WINDOW_BYTES = 65536

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
        # THE DURABLE CURSOR (ADR-0010, gateway backlog 0082): every appended line carries its
        # per-run `seq` (1, 2, 3, ...), the same cursor SQLite allocates, so a cursor survives a
        # reconnect and a file -> SQLite migration. A line written before seq existed reads as
        # the seq after the previous line (its position). The head cache keeps an append O(1):
        # run_id -> (file size after our last append, last seq); a size we did not write means
        # another writer appended, and the head is re-read from the file's last line.
        self._seq_lock = threading.Lock()
        # run_id -> (file size, last seq, file inode): a file deleted and recreated by another
        # store object (another process) has another inode, so a cached head is never reused for it.
        self._seq_heads: "OrderedDict[str, tuple[int, int, int]]" = OrderedDict()
        self._seq_heads_max = 4096
        # list_after resume points: run_id -> (seq, byte offset just after that record's line).
        # run_id -> (seq, byte offset, file inode); a hint is used only for the same file (inode)
        # and only when it points just after a newline (a line start).
        self._offsets: "OrderedDict[str, tuple[int, int, int]]" = OrderedDict()
        self._offsets_max = 4096
        # Line-offset index per run (`_JsonlRowIndex`), kept current by our own appends (write-through) and
        # extended by reading ONLY the bytes another writer appended since (keyed by inode + indexed size), so a
        # row-address read is a seek + one line parse instead of a full `list()` parse per call.
        self._row_index_lock = threading.Lock()
        self._row_index: "OrderedDict[str, _JsonlRowIndex]" = OrderedDict()

    def _path(self, run_id: str) -> Path:
        return self._base / f"ledger_{run_id}.jsonl"

    def append(self, record: StepRecord) -> Optional[int]:
        """Append one record with its per-run `seq`; return that seq."""
        p = self._path(record.run_id)
        record_dict = steprecord_to_dict(record)
        with self._seq_lock:
            with p.open("a", encoding="utf-8") as f:
                # Another PROCESS appending to the same run (the split API process, a CLI next to
                # a gateway) must not take the same seq: an advisory exclusive lock on the file
                # spans reading the head and writing the line (POSIX; elsewhere the in-process
                # lock alone, the pre-seq posture).
                _flock(f)
                try:
                    # A crash mid-write can leave a torn last line (no newline). Appending onto it
                    # would glue a whole record to the fragment and `list()` would drop both:
                    # end the fragment first, so it stays alone on its (unreadable) line.
                    self._terminate_torn_line(record.run_id, p, f)
                    seq = self._head_seq(record.run_id, p) + 1
                    record_dict["seq"] = seq
                    line = dumps_compact(record_dict)
                    start = f.tell()  # where this record's line begins (after any torn-line newline)
                    # Write as a single append operation to reduce the chance of
                    # concurrent writers producing concatenated JSON objects on one line.
                    f.write(line + "\n")
                    f.flush()
                    size = f.tell()
                    ino = _inode_of(f)
                finally:
                    _funlock(f)
            self._row_index_note_append(record.run_id, ino, start, size)
            self._seq_heads[record.run_id] = (size, seq, ino)
            self._seq_heads.move_to_end(record.run_id)
            while len(self._seq_heads) > self._seq_heads_max:
                self._seq_heads.popitem(last=False)
        self._idem_cache_note(record_dict, line)
        return seq

    def _terminate_torn_line(self, run_id: str, p: Path, f: Any) -> None:
        """Write a newline when the file does not end with one (caller holds the file lock)."""
        try:
            st = os.fstat(f.fileno())
        except OSError:
            return
        size = st.st_size
        cached = self._seq_heads.get(run_id)
        if size == 0 or (cached is not None and cached[0] == size and cached[2] == st.st_ino):
            return  # empty, or our own last append (which ended with a newline)
        with p.open("rb") as rf:
            rf.seek(size - 1)
            last = rf.read(1)
        if last != b"\n":
            logger.warning("JsonlLedgerStore: %s ends with a torn line; it is closed before the next append", p.name)
            f.write("\n")
            f.flush()

    def _head_seq(self, run_id: str, p: Path) -> int:
        """The run's last seq (0 when it has no records). Caller holds `_seq_lock`."""
        try:
            st = p.stat()
        except FileNotFoundError:
            self._seq_heads.pop(run_id, None)
            return 0
        size = st.st_size
        cached = self._seq_heads.get(run_id)
        if cached is not None and cached[0] == size and cached[2] == st.st_ino:
            return cached[1]
        if size == 0:
            return 0
        for line in reversed(self._tail_lines(p, 1)):
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                break
            if isinstance(last, dict) and isinstance(last.get("seq"), int) and not isinstance(last.get("seq"), bool):
                return int(last["seq"])
            break
        # A ledger written before seq (or a torn last line): number by position, once.
        records = self.list(run_id)
        return int(records[-1]["seq"]) if records else 0

    @staticmethod
    def _number(records: List[Dict[str, Any]], start: int = 0) -> List[Dict[str, Any]]:
        """Give each record without a persisted seq the seq after the previous record's."""
        prev = start
        for rec in records:
            seq = rec.get("seq")
            if isinstance(seq, int) and not isinstance(seq, bool) and seq > prev:
                prev = seq
            else:
                prev += 1
                rec["seq"] = prev
        return records

    # -- row address (ADR 0011) -------------------------------------------------------------------------------

    def _row_index_note_append(self, run_id: str, ino: int, line_start: int, size_after: int) -> None:
        """Write-through: extend a CURRENT index by the line we just wrote (no read). An index that is not exactly
        at `line_start` (another writer appended, a torn line was closed, or none exists yet) is left alone: the
        next read extends it from its own size."""
        with self._row_index_lock:
            idx = self._row_index.get(run_id)
            if idx is None and line_start == 0:  # a new ledger: indexed from birth, for free
                idx = _JsonlRowIndex(ino)
                self._row_index[run_id] = idx
                while len(self._row_index) > self._ROW_INDEX_MAX_RUNS:
                    self._row_index.popitem(last=False)
            if idx is None or idx.ino != ino or idx.size != line_start:
                return
            idx.offsets.append(line_start)
            idx.size = size_after
            self._row_index_maybe_persist(run_id, idx)

    def _row_index_current(self, run_id: str) -> Optional[_JsonlRowIndex]:
        """The run's line-offset index, extended to the file's current newline-terminated end; None when the
        ledger file does not exist."""
        p = self._path(run_id)
        try:
            st = p.stat()
        except FileNotFoundError:
            with self._row_index_lock:
                self._row_index.pop(run_id, None)
            return None
        with self._row_index_lock:
            idx = self._row_index.get(run_id)
            if idx is None or idx.ino != st.st_ino or st.st_size < idx.size:
                # A fresh process, an evicted run, or a recreated file: the persisted index, when it still matches
                # this file, saves the full parse (C3); else rebuild from byte 0.
                idx = self._row_index_load(run_id, p, st) or _JsonlRowIndex(st.st_ino)
            if st.st_size > idx.size:
                self._row_index_extend(p, idx)
                self._row_index_maybe_persist(run_id, idx)
            self._row_index[run_id] = idx
            self._row_index.move_to_end(run_id)
            while len(self._row_index) > self._ROW_INDEX_MAX_RUNS:
                self._row_index.popitem(last=False)
            return idx

    def _idx_path(self, run_id: str) -> Path:
        return self._base / f"ledger_{run_id}.idx"

    def _row_index_load(self, run_id: str, p: Path, st: os.stat_result) -> Optional[_JsonlRowIndex]:
        """The persisted index when it describes THIS file (same inode, a size within it ending on a line break);
        None otherwise (it is then rebuilt and rewritten)."""
        try:
            data = self._idx_path(run_id).read_bytes()
        except OSError:
            return None
        head = len(_IDX_MAGIC) + 24
        if len(data) < head or data[: len(_IDX_MAGIC)] != _IDX_MAGIC:
            return None
        meta = array("q")
        meta.frombytes(data[len(_IDX_MAGIC):head])
        ino, size, count = meta
        if ino != st.st_ino or size > st.st_size or size < 0 or len(data) != head + 8 * count:
            return None
        if size:
            with p.open("rb") as f:
                f.seek(size - 1)
                if f.read(1) != b"\n":
                    return None
        idx = _JsonlRowIndex(st.st_ino)
        idx.offsets.frombytes(data[head:])
        if idx.offsets and not (0 <= idx.offsets[-1] < size):
            return None
        idx.size = idx.persisted_size = size
        return idx

    def _row_index_maybe_persist(self, run_id: str, idx: _JsonlRowIndex) -> None:
        due = max(self._ROW_INDEX_PERSIST_BYTES, idx.size // self._ROW_INDEX_PERSIST_FRACTION)
        if idx.size - idx.persisted_size >= due:
            self._row_index_persist(run_id, idx)

    def _row_index_persist(self, run_id: str, idx: _JsonlRowIndex) -> None:
        """Write the index atomically beside the ledger (best effort: a failure only costs a re-parse later)."""
        target = self._idx_path(run_id)
        tmp = target.with_name(f"{target.name}.{uuid.uuid4().hex}.tmp")
        try:
            meta = array("q", [idx.ino, idx.size, len(idx.offsets)])
            tmp.write_bytes(_IDX_MAGIC + meta.tobytes() + idx.offsets.tobytes())
            os.replace(tmp, target)
            idx.persisted_size = idx.size
        except OSError:
            with contextlib.suppress(OSError):
                tmp.unlink()

    @staticmethod
    def _row_index_extend(p: Path, idx: _JsonlRowIndex) -> None:
        """Index the complete lines appended after `idx.size` (parsing each once, as `list()` would)."""
        with p.open("rb") as f:
            f.seek(idx.size)
            data = f.read()
        pos = 0
        while True:
            nl = data.find(b"\n", pos)
            if nl < 0:
                break  # a trailing line without its newline is still being written: indexed next time
            for _ in _jsonl_line_records(data[pos:nl]):
                idx.offsets.append(idx.size + pos)
            pos = nl + 1
        idx.size += pos

    def _row_index_drop(self, run_id: str) -> None:
        with self._row_index_lock:
            self._row_index.pop(run_id, None)

    def get_record_unresolved(self, run_id: str, seq: int) -> Optional[Dict[str, Any]]:
        """Point read through the line-offset index: one seek + one line parse at the record's POSITION, verified
        against the seq persisted on the line (ADR 0010). Positions and seqs agree on a ledger written by this
        store (a line without a persisted seq reads as its position); when they do not (a rewritten file, an
        import with gaps) the read falls back to a scan by persisted seq, never answers another row."""
        rid = str(run_id or "")
        s = coerce_ledger_seq(seq)
        if not rid or s is None:
            return None
        idx = self._row_index_current(rid)
        if idx is None:
            return None
        if s > len(idx.offsets):
            # Beyond the newline-terminated rows: only a trailing line still being written can be there.
            return super().get_record_unresolved(rid, s)
        offset = idx.offsets[s - 1]
        first = bisect.bisect_left(idx.offsets, offset)
        with self._path(rid).open("rb") as f:
            if offset > 0:
                f.seek(offset - 1)
                boundary = f.read(1) == b"\n"
            else:
                boundary = True
            f.seek(offset)
            records = _jsonl_line_records(f.readline()) if boundary else []
        k = (s - 1) - first
        if k >= len(records):  # the file no longer matches the index (rewritten in place): re-read honestly
            self._row_index_drop(rid)
            return super().get_record_unresolved(rid, s)
        out = dict(records[k])
        persisted = out.get("seq")
        if isinstance(persisted, int) and not isinstance(persisted, bool) and persisted != s:
            # The row at this position carries another seq: find the row whose persisted seq is `s`.
            return super().get_record_unresolved(rid, s)
        out["seq"] = s
        return out

    def find_started_record(
        self, run_id: str, step_id: str, *, before_seq: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """The STARTED row of `step_id`, searched BACKWARDS from `before_seq` in doubling byte windows (the
        STARTED row of a terminal row is normally a few rows above it); a line is parsed only when its bytes
        contain the step id. A step has exactly one STARTED row (`StepRecord.start` mints the step id)."""
        rid = str(run_id or "")
        sid = str(step_id or "")
        if not rid or not sid:
            return None
        idx = self._row_index_current(rid)
        if idx is None:
            return None
        offsets = idx.offsets
        limit = coerce_ledger_seq(before_seq)
        upper = len(offsets) if limit is None else min(limit - 1, len(offsets))  # last candidate seq
        if upper < 1:
            return None
        needle = json.dumps(sid, ensure_ascii=False).encode("utf-8")
        # End of the line holding row `upper`: the next distinct line offset, else the indexed end.
        nxt = bisect.bisect_right(offsets, offsets[upper - 1])
        hi = offsets[nxt] if nxt < len(offsets) else idx.size
        window = int(self._STARTED_SCAN_WINDOW_BYTES)
        floor = offsets[0]  # the first record's line (a file may start with blank or corrupt lines)
        with self._path(rid).open("rb") as f:
            while hi > floor:
                row = max(0, bisect.bisect_left(offsets, max(0, hi - window)))
                lo = offsets[row] if row < len(offsets) and offsets[row] < hi else hi
                if lo >= hi:  # the window is smaller than one line: take the whole line ending at hi
                    lo = offsets[max(0, bisect.bisect_left(offsets, hi) - 1)]
                if lo >= hi:  # pragma: no cover - offsets below hi always exist while hi > floor
                    break
                f.seek(lo)
                data = f.read(hi - lo)
                end = len(data)
                while True:
                    pos = data.rfind(needle, 0, end)
                    if pos < 0:
                        break
                    line_off = offsets[bisect.bisect_right(offsets, lo + pos) - 1]
                    first = bisect.bisect_left(offsets, line_off)
                    nl = data.find(b"\n", line_off - lo)
                    line = data[line_off - lo: nl if nl >= 0 else len(data)]
                    for k, rec in reversed(list(enumerate(_jsonl_line_records(line)))):
                        position = first + k + 1
                        persisted = rec.get("seq")
                        seq = persisted if isinstance(persisted, int) and not isinstance(persisted, bool) else position
                        # seq >= position (seqs only grow), so rows past `upper` are never candidates; the
                        # persisted seq decides against `before_seq`.
                        if position > upper or (limit is not None and seq >= limit):
                            continue
                        status = rec.get("status")
                        status = getattr(status, "value", status)
                        if str(rec.get("step_id") or "") == sid and status == StepStatus.STARTED.value:
                            out = dict(rec)
                            out["seq"] = seq
                            return out
                    end = min(line_off - lo, pos)  # strictly decreasing: the search always terminates
                    if end <= 0:
                        break
                hi = lo
                window *= 2
        return None

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
        return self._number(out)

    def list_after(self, *, run_id: str, after: int, limit: int = 1000) -> Tuple[List[Dict[str, Any]], int]:
        """Records with seq > `after` (at most `limit`), and the last seq returned (or `after`).

        Same cursor as `SqliteLedgerStore.list_after`. Byte-offset friendly: the store remembers
        where the last returned record ended, so a follower reading forward (`after` = the seq it
        last got) seeks there instead of re-reading the file from the start."""
        rid = str(run_id or "").strip()
        a = max(0, int(after or 0))
        lim = max(1, int(limit or 1000))
        if not rid:
            return ([], a)
        p = self._path(rid)
        try:
            st = p.stat()
        except FileNotFoundError:
            return ([], a)
        size, ino = st.st_size, st.st_ino
        start_seq, offset = 0, 0
        with self._seq_lock:
            hint = self._offsets.get(rid)
        if hint is not None and hint[0] <= a and hint[1] <= size and hint[2] == ino:
            start_seq, offset = hint[0], hint[1]
        out: List[Dict[str, Any]] = []
        prev = start_seq  # the seq of the last record before `pos`
        pos = offset  # byte offset of the next unread line
        resume = (start_seq, offset)  # where the next call can start: (seq before it, its offset)
        with p.open("rb") as f:
            if offset > 0:
                f.seek(offset - 1)
                if f.read(1) != b"\n":
                    # Not a line start any more (the file was rewritten under the hint): read
                    # from the start.
                    start_seq, offset, prev, pos = 0, 0, 0, 0
                    resume = (0, 0)
            f.seek(offset)
            for raw in f:
                if not raw.endswith(b"\n"):
                    break  # a line still being written: the next read gets it whole
                line_start, line_prev = pos, prev
                pos += len(raw)
                text = raw.decode("utf-8", errors="replace").strip()
                objs: List[Any] = []
                if text:
                    try:
                        objs = [json.loads(text)]
                    except json.JSONDecodeError:
                        decoder = json.JSONDecoder()  # concatenated objects on one line (see list())
                        i = 0
                        while i < len(text):
                            while i < len(text) and text[i].isspace():
                                i += 1
                            if i >= len(text):
                                break
                            try:
                                obj, i = decoder.raw_decode(text, idx=i)
                            except json.JSONDecodeError:
                                break
                            objs.append(obj)
                batch = self._number([o for o in objs if isinstance(o, dict)], prev)
                if batch:
                    prev = int(batch[-1]["seq"])
                wanted = [rec for rec in batch if int(rec["seq"]) > a]
                if len(out) + len(wanted) > lim:
                    out.extend(wanted[: lim - len(out)])
                    resume = (line_prev, line_start)  # this line was only partly returned
                    break
                out.extend(wanted)
                resume = (prev, pos)
                if len(out) >= lim:
                    break
        if resume[1] > offset or hint is None:
            with self._seq_lock:
                self._offsets[rid] = (resume[0], resume[1], ino)
                self._offsets.move_to_end(rid)
                while len(self._offsets) > self._offsets_max:
                    self._offsets.popitem(last=False)
        return (out, int(out[-1]["seq"])) if out else ([], a)

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
        with self._seq_lock:
            self._seq_heads.pop(rid, None)
            self._offsets.pop(rid, None)
        with self._idem_cache_lock:
            # Drop the run's cached results: a recreated ledger under the
            # same run_id must never see the deleted run's completions.
            for cache_key in [k for k in self._idem_cache if k[0] == rid]:
                del self._idem_cache[cache_key]
        self._row_index_drop(rid)
        with contextlib.suppress(OSError):
            self._idx_path(rid).unlink()
        return count
