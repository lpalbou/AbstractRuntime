"""Runtime-owned AbstractMemory facade: the complete host surface (ADR 0009).

Hosts (AbstractGateway, AbstractNote, ...) reach AbstractMemory through THIS module and the effect-handler builders
of `abstractruntime.integrations.abstractmemory`, never by importing `abstractmemory` (ADR 0008 decision 2, ADR 0009
decision 1). A name or capability a host needs and cannot reach here is a runtime defect: add it here and cut a patch
release; never work around it in the host.

What lives here:
- `MemoryFacade`: one mind (`open_mind(path)`: the SQLite triple store, journal and `abstractmemory.graph.MemoryGraph`
  of one home) plus the runtime stores that hold its ledger sources (ledger, runs, artifacts, diary, event inbox).
  `stamp` adds the ledger refs the runtime knows to a provenance dict (never fabricating one); `reconstruct_source`
  turns a `SourceRef` dict back into the verbatim it points at (never raising for a missing source); `history`,
  `sources_of`, `derived_from_source` and `apply_revision` pass through to the graph.
- `is_real_run(run_id, run_store=)`: a synthetic run id (`chat-<sid>`, `diary:<eid>`, `global_memory`,
  `session_memory_<sid>`, or one the run store cannot load) is never a `run`/`step` source.
- The AbstractMemory names hosts use, re-exported UNCHANGED (`facade.X is abstractmemory.X`) and resolved lazily by a
  module `__getattr__`: `GATEWAY_NAMES` lists them (seeded from AbstractGateway's direct imports, audit 2026-10-27).

AbstractMemory is imported lazily, inside functions (`_memory()`, `_graph()`), so importing this module never imports
AbstractMemory (`tests/test_runtime_install_boundary.py`, ADR 0009 enforcement). `get_record`-backed step
reconstruction and artifact/run/diary/event reconstruction need no AbstractMemory at all.
AbstractMemory >= 0.4.0 provides `abstractmemory.graph` (SourceRef, Provenance, Revision, MemoryGraph).
"""

from __future__ import annotations

import base64
import dataclasses
import importlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from ...storage.base import coerce_ledger_seq

# -- lazy AbstractMemory access ----------------------------------------------------------------------------------

_INSTALL_HINT = (
    "AbstractMemory is unavailable. Install a Runtime environment with AbstractMemory >= 0.4.0 "
    "(`pip install \"AbstractMemory>=0.4.0\"`; the graph package `abstractmemory.graph` is required)."
)


def _import(module: str) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise RuntimeError(f"{_INSTALL_HINT} (failed to import {module!r}: {exc})") from exc


def _memory() -> Any:
    """The `abstractmemory` package (lazy)."""
    return _import("abstractmemory")


def _graph() -> Any:
    """The `abstractmemory.graph` package (lazy; AbstractMemory 0.4.0+)."""
    return _import("abstractmemory.graph")


# Re-exported names -> the AbstractMemory module that defines them as public API. Each resolves to AbstractMemory's
# own object (identity), on first access.
_REEXPORTS: Dict[str, str] = {
    # Entity context budgets and the identity floor (AbstractMemory 0.3.x; gateway entity_gate.py).
    "SELF_FRACTION_FLOOR": "abstractmemory",
    "ENTITY_CONTEXT_ACCEPTABLE": "abstractmemory",
    "ENTITY_CONTEXT_FLOOR": "abstractmemory",
    "ENTITY_CONTEXT_RECOMMENDED": "abstractmemory",
    "entity_recall_budget": "abstractmemory",
    # Entity replay stream (gateway entity_replay.py: `abstractmemory.replay`, `abstractmemory.records`).
    "REPLAY_FAMILIES": "abstractmemory.replay",
    "REPLAY_STREAM": "abstractmemory.replay",
    "REPLAY_STREAM_VERSION": "abstractmemory.replay",
    "RESERVED_FAMILIES": "abstractmemory.replay",
    "resolve_digest_assertion": "abstractmemory.records",
    # Spark templates (gateway entities.py, template_store.py).
    "DEFAULT_SPARK_TEMPLATE": "abstractmemory",
    "lint_spark": "abstractmemory",
    "canonical_spark_hash": "abstractmemory",
    # Stores, journal, queries (gateway entities.py, memory_store.py, routes/gateway.py, routes/entity_replay.py).
    "MemorySystem": "abstractmemory",
    "SQLiteJournal": "abstractmemory",
    "SQLiteTripleStore": "abstractmemory",
    "LanceDBTripleStore": "abstractmemory",
    "InMemoryTripleStore": "abstractmemory",
    "TripleQuery": "abstractmemory",
    "TripleAssertion": "abstractmemory",
    "ClosureRecord": "abstractmemory",
    # Entity health, chains, engram, night passes, embeddings, drives, disposal (gateway entities.py, routes).
    "cognition_health": "abstractmemory",
    "verify_diary_chain": "abstractmemory",
    "engram": "abstractmemory",
    "sleep_pass": "abstractmemory",
    "dream_pass": "abstractmemory",
    "reembed_store": "abstractmemory",
    "read_embedding_pin": "abstractmemory",
    "DRIVE_PRESSURE_BOUND": "abstractmemory",
    "drive_pressure": "abstractmemory",
    "promote_candidate": "abstractmemory",
    "reject_candidate": "abstractmemory",
    # Diary reads over the graph plane (gateway entities.py: open questions / problems / ideas).
    "open_questions": "abstractmemory",
    "open_problems": "abstractmemory",
    "open_ideas": "abstractmemory",
    # The graph (AbstractMemory 0.4.0+, ADR 0003): typed provenance, revisions, the one write path.
    "SourceRef": "abstractmemory.graph",
    "Provenance": "abstractmemory.graph",
    "ProvenanceRequired": "abstractmemory.graph",
    "InvalidSourceRef": "abstractmemory.graph",
    "Revision": "abstractmemory.graph",
    "RevisionOp": "abstractmemory.graph",
    "RevisionResult": "abstractmemory.graph",
    "MemoryGraph": "abstractmemory.graph",
}

#: The AbstractMemory names hosts rely on (AbstractGateway's 34 direct import lines, audit 2026-10-27, plus the
#: graph names of AbstractMemory 0.4.0). A test checks each resolves to AbstractMemory's own object.
GATEWAY_NAMES: Tuple[str, ...] = tuple(_REEXPORTS)


def __getattr__(name: str) -> Any:
    module = _REEXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    # No hasattr fallback: a name AbstractMemory dropped fails loudly here (and in the identity test).
    return getattr(_import(module), name)


# -- read bounds (every read is bounded: AbstractMemory ADR 0003 decision 12) ------------------------------------

#: Ledger records a reconstructed `run` source carries by default, and the most one call may ask for.
DEFAULT_RUN_LEDGER_LIMIT = 200
MAX_RUN_LEDGER_LIMIT = 1000


# -- synthetic run ids ------------------------------------------------------------------------------------------

#: Run-id prefixes that never name a loadable run: the home-direct entity driver (`chat-<session_id>`), the diary
#: chain (`diary:<entity_id>`), the session memory owner (`session_memory_<sid>`).
SYNTHETIC_RUN_PREFIXES: Tuple[str, ...] = ("chat-", "diary:", "session_memory_")


def _synthetic_exact_ids() -> Tuple[str, ...]:
    from .effect_handlers import _DEFAULT_GLOBAL_MEMORY_RUN_ID, _global_memory_owner_id

    return tuple(dict.fromkeys((_DEFAULT_GLOBAL_MEMORY_RUN_ID, _global_memory_owner_id())))


def is_real_run(run_id: Any, *, run_store: Any = None) -> bool:
    """True when `run_id` can name a ledger run: a non-empty string that is not a synthetic id (`chat-*`,
    `diary:*`, `session_memory_*`, the global memory owner) and, when `run_store` is given, that the store loads.
    A run id that fails this is never stamped as a `run` or `step` ref (ADR 0009 decision 4)."""
    if not isinstance(run_id, str) or not run_id.strip():
        return False
    rid = run_id.strip()
    if rid.startswith(SYNTHETIC_RUN_PREFIXES) or rid in _synthetic_exact_ids():
        return False
    if run_store is None:
        return True
    try:
        return run_store.load(rid) is not None
    except Exception:
        return False


# -- source ref shapes (validated locally: reconstruction of steps/artifacts needs no AbstractMemory) ------------

_REF_REQUIRED: Mapping[str, Tuple[str, ...]] = {
    "run": ("run_id",),
    "step": ("run_id", "seq"),
    "artifact": ("artifact_id",),
    "message": ("message_id",),
    "event": ("event_id",),
    "diary_entry": ("entry_id",),
}


def _ref_problem(ref: Any) -> Optional[str]:
    if not isinstance(ref, Mapping):
        return f"a SourceRef is a dict (got {type(ref).__name__})"
    kind = str(ref.get("kind") or "").strip().lower()
    if kind not in _REF_REQUIRED:
        return f"SourceRef.kind must be one of {sorted(_REF_REQUIRED)} (got {ref.get('kind')!r})"
    for name in _REF_REQUIRED[kind]:
        value = ref.get(name)
        if name == "seq":
            if coerce_ledger_seq(value) is None or isinstance(value, bool) or not isinstance(value, int):
                return f"a 'step' SourceRef requires an int 'seq' >= 1 (got {value!r})"
        elif not isinstance(value, str) or not value.strip():
            return f"a {kind!r} SourceRef requires {name!r}"
    return None


def _status(value: Any) -> str:
    return str(getattr(value, "value", value) or "")


def _ref_dedupe(refs: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Validate through `SourceRef`, collapse on the canonical key (first wins), and drop a `run` ref whose run is
    already named by a `step` ref (the step is the more precise address of the same source)."""
    SourceRef = _graph().SourceRef
    by_key: Dict[str, Any] = {}
    for item in refs:
        ref = SourceRef.from_dict(item)
        by_key.setdefault(ref.key(), ref)
    step_runs = {r.run_id for r in by_key.values() if r.kind == "step"}
    return [r.to_dict() for r in by_key.values() if not (r.kind == "run" and r.run_id in step_runs)]


class MemoryFacade:
    """The host surface over one mind and the runtime stores that hold its ledger sources (ADR 0009).

    Every store is optional: `stamp` needs none of them (a `run_store` makes `is_real_run` check that a run loads),
    the graph reads need `graph`, and `reconstruct_source` answers `unsupported` for a kind whose store is absent.
    """

    def __init__(
        self,
        *,
        graph: Any = None,
        ledger_store: Any = None,
        run_store: Any = None,
        artifact_store: Any = None,
        diary_store: Any = None,
        event_inbox: Any = None,
        runtime_id: Optional[str] = None,
    ) -> None:
        self._graph = graph
        self.ledger_store = ledger_store
        self.run_store = run_store
        self.artifact_store = artifact_store
        self.diary_store = diary_store
        self.event_inbox = event_inbox
        self.runtime_id = str(runtime_id).strip() if isinstance(runtime_id, str) and runtime_id.strip() else None

    # -- opening a mind --------------------------------------------------------------------------------------------

    @classmethod
    def open_mind(
        cls, path: Any, *, embedder: Any = None, runtime_id: Optional[str] = None, **stores: Any
    ) -> "MemoryFacade":
        """Open one SQLite home: `SQLiteTripleStore` and `SQLiteJournal` on the SAME file (so a revision's journal
        closures share its transaction) and the `MemoryGraph` over them (graph tables are created on open).
        `runtime_id` names the ledger this facade resolves (the plane's or the home's); `stores` are the facade's
        other keyword stores (`ledger_store`, `run_store`, `artifact_store`, ...)."""
        am = _memory()
        db = Path(path).expanduser()
        store = am.SQLiteTripleStore(db, embedder=embedder) if embedder is not None else am.SQLiteTripleStore(db)
        journal = am.SQLiteJournal(db)
        graph = _graph().MemoryGraph(store, journal)
        return cls(graph=graph, runtime_id=runtime_id, **stores)

    @property
    def graph(self) -> Any:
        if self._graph is None:
            raise RuntimeError("this MemoryFacade has no mind: open one with MemoryFacade.open_mind(path)")
        return self._graph

    @property
    def store(self) -> Any:
        return self.graph.store

    @property
    def journal(self) -> Any:
        return self.graph.journal

    # -- stamping ----------------------------------------------------------------------------------------------------

    def is_real_run(self, run_id: Any) -> bool:
        return is_real_run(run_id, run_store=self.run_store)

    def _real(self, run_id: Any, trusted_run_id: Optional[str]) -> bool:
        # A run the runtime is executing right now (its step context names it) is loadable by construction: only the
        # synthetic-id check applies, and the hot path pays no run-store read.
        if trusted_run_id is not None and isinstance(run_id, str) and run_id.strip() == trusted_run_id:
            return is_real_run(run_id)
        return self.is_real_run(run_id)

    def source_refs(
        self,
        *,
        run_id: Any,
        seq: Any = None,
        step_id: Optional[str] = None,
        artifact_ids: Iterable[Any] = (),
        runtime_id: Optional[str] = None,
        _trusted_run_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """The refs the runtime can vouch for: a `step` ref when `run_id` is real and `seq` is a ledger row address,
        else a `run` ref when `run_id` is real; one `artifact` ref per artifact id. Nothing for a synthetic run."""
        rt = runtime_id or self.runtime_id
        real = self._real(run_id, _trusted_run_id)
        rid = str(run_id).strip() if real else None
        refs: List[Dict[str, Any]] = []
        s = coerce_ledger_seq(seq)
        if rid and s is not None:
            ref: Dict[str, Any] = {"kind": "step", "run_id": rid, "seq": s}
            if isinstance(step_id, str) and step_id.strip():
                ref["step_id"] = step_id.strip()
            refs.append(ref)
        elif rid:
            refs.append({"kind": "run", "run_id": rid})
        if isinstance(artifact_ids, str):
            artifact_ids = (artifact_ids,)
        for aid in artifact_ids or ():
            if not isinstance(aid, str) or not aid.strip():
                continue
            ref = {"kind": "artifact", "artifact_id": aid.strip()}
            if rid:
                ref["run_id"] = rid
            refs.append(ref)
        if rt:
            for ref in refs:
                ref["runtime_id"] = rt
        return refs

    def stamp(
        self,
        provenance: Optional[Mapping[str, Any]],
        *,
        run_id: Any,
        seq: Any = None,
        step_id: Optional[str] = None,
        artifact_ids: Iterable[Any] = (),
        runtime_id: Optional[str] = None,
        _trusted_run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return `provenance` with the ledger refs the runtime knows added to its `sources` list.

        - Graph shape (the dict has `sources` or `derived_from` and only graph keys): validated by `Provenance`
          (`actor` defaults to "runtime") and returned in the graph shape with the refs appended.
        - Legacy flat dict (`run_id`, `turn_id`, `span_id`, `payload_ref`, `entry_id`, ...): the legacy keys stay
          (existing readers keep working); `sources` is ADDED, lifted by `Provenance.from_legacy` plus the refs. A
          flat `run_id` that is not a real run is not lifted to a `run` ref. No `sources` key is added when there is
          nothing to name.
        Never fabricates: no step ref without a `seq`, no run/step ref for a synthetic or unloadable run id.
        """
        G = _graph()
        prov = dict(provenance or {})
        new_refs = self.source_refs(
            run_id=run_id, seq=seq, step_id=step_id, artifact_ids=artifact_ids, runtime_id=runtime_id,
            _trusted_run_id=_trusted_run_id,
        )
        graph_shape = bool(prov) and set(prov) <= G.PROVENANCE_KEYS and ("sources" in prov or "derived_from" in prov)
        if graph_shape:
            base = G.Provenance.from_dict({**prov, "actor": prov.get("actor") or "runtime"})
            merged = _ref_dedupe([*(s.to_dict() for s in base.sources), *new_refs])
            return {**base.to_dict(), "sources": merged}

        existing = prov.get("sources")
        flat = {k: v for k, v in prov.items() if k != "sources"}
        lift = dict(flat)
        legacy_run = lift.get("run_id")
        if isinstance(legacy_run, str) and not legacy_run.strip().startswith("diary:") and not self._real(
            legacy_run, _trusted_run_id
        ):
            lift.pop("run_id", None)  # a synthetic run id is never lifted to a run ref
        lifted = [s.to_dict() for s in G.Provenance.from_legacy(lift, actor="runtime").sources]
        rt = runtime_id or self.runtime_id
        if rt:  # refs the runtime lifts are its own: they name the ledger they resolve against
            for ref in lifted:
                ref.setdefault("runtime_id", rt)
        merged = _ref_dedupe([*(existing if isinstance(existing, list) else []), *lifted, *new_refs])
        if not merged:
            return flat
        return {**flat, "sources": merged}

    def stamp_current_step(
        self,
        provenance: Optional[Mapping[str, Any]],
        *,
        run_id: Any,
        artifact_ids: Iterable[Any] = (),
        runtime_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """`stamp` with the step the runtime is executing for `run_id` (`core.step_context`): a `step` ref
        `(run_id, seq, step_id)` when the handler runs inside a runtime step of THAT run; outside any step (a handler
        called directly, the home-direct entity driver) no seq is known and none is guessed."""
        from ...core.step_context import current_step_context

        ctx = current_step_context(run_id) if isinstance(run_id, str) and run_id.strip() else None
        if ctx is None:
            return self.stamp(provenance, run_id=run_id, artifact_ids=artifact_ids, runtime_id=runtime_id)
        return self.stamp(
            provenance, run_id=ctx.run_id, seq=ctx.seq, step_id=ctx.step_id, artifact_ids=artifact_ids,
            runtime_id=runtime_id, _trusted_run_id=ctx.run_id,
        )

    # -- graph pass-through ------------------------------------------------------------------------------------------

    def history(self, memory_id: str, **kwargs: Any) -> List[Dict[str, Any]]:
        return self.graph.history(memory_id, **kwargs)

    def sources_of(self, memory_id: str, **kwargs: Any) -> List[Dict[str, Any]]:
        """The memory's ledger sources as SourceRef dicts (`MemoryGraph.sources_of`)."""
        return [ref.to_dict() for ref in self.graph.sources_of(memory_id, **kwargs)]

    def derived_from_source(self, ref: Any, **kwargs: Any) -> List[str]:
        return self.graph.derived_from_source(ref, **kwargs)

    def apply_revision(self, revision: Any) -> Dict[str, Any]:
        """Apply a `Revision` (or its dict form) through `MemoryGraph.apply_revision`; returns the RevisionResult as
        a dict. Raises `ProvenanceRequired` for an ungrounded revision (stamp first; never fabricate a ref)."""
        G = _graph()
        rev = revision if isinstance(revision, G.Revision) else G.Revision.from_dict(revision)
        return dataclasses.asdict(self.graph.apply_revision(rev))

    # -- reconstruction ----------------------------------------------------------------------------------------------

    def reconstruct_source(self, ref: Any, *, limit: int = DEFAULT_RUN_LEDGER_LIMIT) -> Dict[str, Any]:
        """The verbatim a SourceRef points at (ADR 0009 decision 5). Never raises for a missing source:
        `{"status": "ok", "ref", "kind", "content", "metadata"}`, `{"status": "source_gone", "ref", "reason"}`, or
        `{"status": "unsupported", "ref", "reason"}` (no store for that kind here, an invalid ref, a ref of another
        runtime, or `message`, which has no indexed path today).

        `runtime_id`: a ref naming another runtime than this facade's is `unsupported` (its row address belongs to
        another ledger). `limit` bounds the ledger records of a `run` source (default 200, at most 1000); the
        metadata says how many were left out."""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            return {"status": "unsupported", "ref": ref, "reason": f"limit must be a positive int (got {limit!r})"}
        limit = min(limit, MAX_RUN_LEDGER_LIMIT)
        problem = _ref_problem(ref)
        if problem is not None:
            return {"status": "unsupported", "ref": ref, "reason": f"invalid SourceRef: {problem}"}
        ref = dict(ref)
        kind = str(ref["kind"]).strip().lower()
        ref_rt = ref.get("runtime_id")
        if isinstance(ref_rt, str) and ref_rt.strip() and ref_rt.strip() != self.runtime_id:
            # A facade without a runtime_id cannot tell whether a named runtime is its own: never answer `ok` for
            # what may be another ledger's row.
            return _unsupported(
                ref, f"ref names runtime {ref_rt.strip()}, this facade resolves "
                f"{self.runtime_id or 'an unnamed runtime (build it with runtime_id=)'}"
            )
        try:
            if kind == "run":
                return self._reconstruct_run(ref, limit)
            handler = getattr(self, f"_reconstruct_{kind}")
            return handler(ref)
        except Exception as exc:  # a store failing to read is reported, never raised to the reader
            return {"status": "source_gone", "ref": ref, "reason": f"{kind} source could not be read: {exc}"}

    def _reconstruct_step(self, ref: Dict[str, Any]) -> Dict[str, Any]:
        if self.ledger_store is None:
            return _unsupported(ref, "no ledger store was given to this facade")
        run_id, seq = str(ref["run_id"]).strip(), int(ref["seq"])
        record = self.ledger_store.get_record(run_id, seq)
        if record is None:
            return _gone(ref, f"ledger row {run_id}#{seq} does not exist (the run's ledger was deleted, or no such seq)")
        want = ref.get("step_id")
        if isinstance(want, str) and want.strip() and str(record.get("step_id") or "") != want.strip():
            return _gone(
                ref,
                f"ledger row {run_id}#{seq} holds step {record.get('step_id')!r}, not {want.strip()!r} "
                "(the ledger was rewritten)",
            )
        effect = record.get("effect") if isinstance(record.get("effect"), dict) else {}
        return _ok(ref, "step", record, {
            "run_id": run_id,
            "seq": seq,
            "step_id": record.get("step_id"),
            "node_id": record.get("node_id"),
            "status": _status(record.get("status")),
            "effect_type": _status(effect.get("type")) or None,
            "started_at": record.get("started_at"),
            "ended_at": record.get("ended_at"),
        })

    def _reconstruct_run(self, ref: Dict[str, Any], limit: int) -> Dict[str, Any]:
        if self.run_store is None:
            return _unsupported(ref, "no run store was given to this facade")
        from ...storage.serialize import runstate_to_dict

        run_id = str(ref["run_id"]).strip()
        state = self.run_store.load(run_id)
        if state is None:
            return _gone(ref, f"run {run_id!r} does not exist (deleted, or never a stored run)")
        content: Dict[str, Any] = {"run": runstate_to_dict(state)}
        metadata: Dict[str, Any] = {
            "run_id": run_id,
            "workflow_id": getattr(state, "workflow_id", None),
            "status": _status(getattr(state, "status", None)),
        }
        if self.ledger_store is not None:
            records = self.ledger_store.list(run_id)
            content["ledger"] = records[:limit]  # the oldest first: a run reads from its start
            metadata["ledger_records"] = len(records)
            metadata["ledger_returned"] = len(content["ledger"])
            metadata["ledger_left_out"] = max(0, len(records) - limit)
            metadata["truncated"] = len(records) > limit
        return _ok(ref, "run", content, metadata)

    def _reconstruct_artifact(self, ref: Dict[str, Any]) -> Dict[str, Any]:
        if self.artifact_store is None:
            return _unsupported(ref, "no artifact store was given to this facade")
        aid = str(ref["artifact_id"]).strip()
        artifact = self.artifact_store.load(aid)
        if artifact is None:
            return _gone(ref, f"artifact {aid!r} does not exist (deleted or collected)")
        meta = artifact.metadata.to_dict()
        try:
            content: Any = artifact.content.decode("utf-8")
            meta["encoding"] = "utf-8"
        except UnicodeDecodeError:
            content = base64.b64encode(artifact.content).decode("ascii")
            meta["encoding"] = "base64"
        return _ok(ref, "artifact", content, meta)

    def _reconstruct_diary_entry(self, ref: Dict[str, Any]) -> Dict[str, Any]:
        if self.diary_store is None:
            return _unsupported(ref, "no diary store was given to this facade")
        chain = ref.get("run_id")
        if isinstance(chain, str) and chain.strip() and chain.strip() != self.diary_store.chain_id:
            return _unsupported(ref, f"the ref names diary {chain!r}; this facade reads {self.diary_store.chain_id!r}")
        eid = str(ref["entry_id"]).strip()
        record = self.diary_store.get_record(eid)
        if record is None:
            return _gone(ref, f"diary entry {eid!r} is not in diary {self.diary_store.chain_id!r}")
        return _ok(ref, "diary_entry", dict(record.get("result") or {}), {
            "entry_id": eid,
            "chain_id": self.diary_store.chain_id,
            "record_hash": record.get("record_hash"),
            "prev_hash": record.get("prev_hash"),
        })

    def _reconstruct_event(self, ref: Dict[str, Any]) -> Dict[str, Any]:
        if self.event_inbox is None:
            return _unsupported(ref, "no event inbox was given to this facade")
        event_id = str(ref["event_id"]).strip()
        rec = self.event_inbox.get(event_id)
        if rec is None or rec.get("pruned") or "payload" not in rec:
            return _gone(ref, f"event {event_id!r} is not in the inbox (never appended, or pruned by retention)")
        stream = ref.get("stream")
        if isinstance(stream, str) and stream.strip() and rec.get("stream") != stream.strip():
            return _gone(ref, f"event {event_id!r} is on stream {rec.get('stream')!r}, not {stream.strip()!r}")
        return _ok(ref, "event", rec.get("payload"), {
            "event_id": event_id,
            "stream": rec.get("stream"),
            "seq": rec.get("seq"),
            "appended_at": rec.get("appended_at"),
        })

    def _reconstruct_message(self, ref: Dict[str, Any]) -> Dict[str, Any]:
        return _unsupported(
            ref, "message refs have no indexed path today (a message id is not a ledger key); name its step or run"
        )


def _ok(ref: Dict[str, Any], kind: str, content: Any, metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {"status": "ok", "ref": ref, "kind": kind, "content": content, "metadata": metadata}


def _gone(ref: Dict[str, Any], reason: str) -> Dict[str, Any]:
    return {"status": "source_gone", "ref": ref, "reason": reason}


def _unsupported(ref: Dict[str, Any], reason: str) -> Dict[str, Any]:
    return {"status": "unsupported", "ref": ref, "reason": reason}


__all__ = [
    "DEFAULT_RUN_LEDGER_LIMIT",
    "GATEWAY_NAMES",
    "MAX_RUN_LEDGER_LIMIT",
    "MemoryFacade",
    "SYNTHETIC_RUN_PREFIXES",
    "is_real_run",
    *GATEWAY_NAMES,
]
