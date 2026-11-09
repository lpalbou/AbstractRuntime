"""Per-entity runtime rooted in the home (plan item 8, runtime R2 half).

The signed consensus plan (a2a/fs/plan.md; laurent's consequence (c) +
item 8): each entity gets its OWN runtime — a real run store named
`runtime_<slug>` INSIDE the home directory, one Runtime instance per
entity, bound to that store and the home's own effect handlers. Copying
the home directory then moves pending runs, waits, and commitments with
the life (the same self-containment rule memory/diary/artifacts already
obey).

What this module is: the COMPOSITION — it builds the per-home Runtime
from parts that all already exist (SqliteRunStore/SqliteLedgerStore over
one SQLite file; `open_home`'s seam + diary handlers; the home's own
artifact store). What it is NOT: the door. The handlers here are the RAW
home handlers (stamp-agnostic, exactly what the home-direct driver uses);
the gateway's GW-C half wraps ITS per-entity runtimes with the stamp
verification layer (`install_entity_routing`) so door-served visit runs
join the verified path — crypto stays at the routing layer, never in the
composition (frozen spec, thread 0013: "the visit path JOINS the verified
path").

Honest premise carried from the plan: the `runtime.sqlite3` seen in
Castor's home is a 0-byte orphan nothing ever read. This store is named
`runtime_<slug>.sqlite3` — a NEW file; the orphan is left untouched (an
operator artifact to clean, never silently deleted by code).

LLM_CALL / TOOL_CALLS handlers are deliberately absent here: the host
(door or driver factory) supplies them at composition time with its own
provider wiring — the home contributes memory, diary, and artifacts; the
mind's substrate is resolved per the no-fallback chain elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..core.runtime import Runtime
from ..storage.sqlite import SqliteDatabase, SqliteLedgerStore, SqliteRunStore
from .chat import ChatHome, open_home

RUN_STORE_PREFIX = "runtime_"


def entity_run_store_path(home_dir: Path) -> Path:
    """`<home>/runtime_<slug>.sqlite3` — the slug is the DIRECTORY name.

    The directory IS the registry key (gateway GW-B: slug normalization is
    the one name key; the manifest's random id is a birth marker, never a
    key), so the store name derives from it and from nothing else — never
    from the address (relocation-stable keys invariant)."""
    home_dir = Path(home_dir)
    return home_dir / f"{RUN_STORE_PREFIX}{home_dir.name}.sqlite3"


def _manifest_name(home_dir: Path) -> str:
    """The entity NAME the manifest claims: `entity:<name>` or the legacy
    `entity:<name>@<home_id>` — both generations parse to <name>."""
    import json

    try:
        manifest = json.loads((Path(home_dir) / "manifest.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - open_home gives the missing-manifest refusal
        return ""
    entity_id = str(manifest.get("entity_id") or "")
    tail = entity_id.split(":", 1)[1] if ":" in entity_id else entity_id
    return tail.split("@", 1)[0].strip().lower()


def _refuse_moved_home(home_dir: Path) -> None:
    """GW-B's moved-home refusal, home-direct (gateway extension ask,
    0014/081145Z): a copied/renamed directory whose manifest names a
    DIFFERENT entity must refuse before a stray `runtime_<straydir>.sqlite3`
    is minted beside the true one. The door already refuses at lookup
    (manifest_for); this gives raw-path callers the same wall. The legacy
    in-process driver's open_home stays permissive until the migration
    retires it — this guard covers the NEW surface with zero legacy risk."""
    home_dir = Path(home_dir)
    claimed = _manifest_name(home_dir)
    if claimed and claimed != home_dir.name.strip().lower():
        raise ValueError(
            f"moved-home collision: directory {home_dir.name!r} carries a manifest "
            f"for entity {claimed!r} - the directory name IS the registry key; "
            "a moved or copied home must keep its name (or be adopted through "
            "the door, which refuses occupied names)"
        )


@dataclass
class EntityRuntime:
    """One entity's runtime: the home plus a Runtime bound to its stores.

    One life, one runtime — callers hold ONE of these per entity (the
    gateway keys its instances by slug; the CLI builds one per session).
    `close()` releases the home's engine and checkpoints the run store so
    the directory is copy-clean."""

    home: ChatHome
    runtime: Runtime
    run_store: SqliteRunStore
    ledger_store: SqliteLedgerStore
    db: SqliteDatabase
    store_path: Path

    @property
    def entity_id(self) -> str:
        return self.home.entity_id

    def close(self) -> None:
        self.home.close()
        self.db.close()


def _route_capability_calls(mind_handler: Any, capability_handlers: dict) -> Any:
    """LLM_CALL routing by `payload.capability` (identity/authoring.py AUTHORING_CAPABILITY_KEY): a tagged call goes
    to its capability handler and its completed result is stamped with the echo; an unserved capability is answered
    without the echo (loud in the log), never by the mind."""
    import logging

    from ..core.runtime import EffectOutcome
    from .authoring import AUTHORING_CAPABILITY_KEY

    log = logging.getLogger(__name__)

    def handler(run: Any, effect: Any, default_next_node: Any = None) -> Any:
        cap = (effect.payload or {}).get(AUTHORING_CAPABILITY_KEY) if isinstance(effect.payload, dict) else None
        if not cap:
            if mind_handler is None:
                return EffectOutcome.failed("no LLM_CALL handler on this entity runtime")
            return mind_handler(run, effect, default_next_node)
        target = capability_handlers.get(str(cap))
        if target is None:
            log.error("entity run %s: no handler serves LLM_CALL capability %r (formation stays pending)",
                      getattr(run, "run_id", "?"), cap)
            return EffectOutcome.completed({"ok": False, "capability_unrouted": str(cap)})
        out = target(run, effect, default_next_node)
        if getattr(out, "status", None) == "completed" and isinstance(getattr(out, "result", None), dict):
            out.result[AUTHORING_CAPABILITY_KEY] = str(cap)
        return out

    return handler


def open_entity_runtime(
    home_dir: Path,
    *,
    embedder: Any = None,
    attention_window: Optional[int] = None,
    extra_handlers: Optional[dict] = None,
    workflow_registry: Any = None,
    capability_handlers: Optional[dict] = None,
    fence: Any,
) -> EntityRuntime:
    """Open an existing home and bind its own Runtime to it.

    `fence` (REQUIRED, a LifecycleFence: STOP-LANE-DESIGN §2.2): the generation the run is admitted at. The home opens
    fenced (memory commit guard, diary, artifacts) and the FINAL handler table - the host's LLM/tool handlers and the
    capability routing included - is wrapped once: after the operator's Stop every effect returns cancelled
    (KILL_SWITCH_REASON; the Runtime ends the run CANCELLED, never failed) and a model reply that returned after it is
    abandoned. The Runtime's own run ledger keeps an unguarded artifact store: the records of a Stop are written
    after it.

    - Run store + ledger: `runtime_<slug>.sqlite3` INSIDE the home (one
      file, both tables — pending runs and waits travel on directory copy).
    - Effect handlers: the home's seam + diary handlers (memory writes land
      in the home's memory.sqlite3, diary in home.sqlite3 — strict=True,
      the entity posture). `extra_handlers` lets the host add LLM/tool
      handlers at composition; home handlers win on collision REFUSAL:
      overriding a home's memory/diary handler would re-route identity
      writes, so it raises instead (the routing-refuses-to-shadow rule).
    - Artifacts: the home's own store — visit-run verbatims/artifacts live
      IN the home like every other part of the life.
    - `capability_handlers` ({capability: LLM_CALL handler}, e.g. the host's
      summaries client): an LLM_CALL whose payload names `capability` goes to
      that handler and its result carries the echo `capability`; a capability
      no handler serves is answered `{ok: False, capability_unrouted: <name>}`
      with no echo and an error in the log - never the mind's answer (the
      formation then forms pending and logs the failure; identity/authoring.py).
    """
    _refuse_moved_home(Path(home_dir))
    from .lifecycle_fence import LifecycleFence, fence_handlers

    if not isinstance(fence, LifecycleFence):
        raise TypeError(f"open_entity_runtime: fence must be the run's LifecycleFence, not {fence!r}")
    home = open_home(home_dir, embedder=embedder, attention_window=attention_window, fence=fence)
    store_path = entity_run_store_path(Path(home_dir))
    db = SqliteDatabase(store_path)
    run_store = SqliteRunStore(db)
    ledger_store = SqliteLedgerStore(db)

    handlers = dict(home.handlers)
    summaries_handler = dict(capability_handlers or {}).get("summaries")
    if summaries_handler is not None:
        # The flow-brain sleep pass (MEMORY_CONSOLIDATE) authors its records with the host's summaries capability
        # (AbstractMemory c46b4cd; Codex review 2 #10).
        from ..integrations.abstractmemory.brain_handlers import build_entity_brain_effect_handlers
        from .authoring import handler_formation_author

        handlers.update(build_entity_brain_effect_handlers(
            memory_system=home.ms, entity_id=home.entity_id, home_dir=Path(home_dir),
            author=handler_formation_author(summaries_handler, home.artifacts),
        ))
    if extra_handlers:
        collisions = sorted(
            getattr(k, "value", str(k)) for k in extra_handlers.keys() & handlers.keys()
        )
        if collisions:
            raise ValueError(
                "extra_handlers would shadow the home's own handlers "
                f"({', '.join(collisions)}) - identity effects route through the home, "
                "never a host override; wiring drift must stay loud"
            )
        handlers.update(extra_handlers)

    # G1 IS STRUCTURAL, NOT OPTIONAL (frozen seam spec, thread 0013): every
    # host-supplied LLM handler on an ENTITY runtime is wrapped with BOTH
    # G1 directions — READ: act-only refs rest in the transcript/ledger,
    # words resolve fresh from the book at send time through the run's own
    # DIARY_READ handler; WRITE: the reply's diary fences are captured at
    # the result boundary and written through DIARY_WRITE BEFORE the result
    # persists (the A/B privacy grep found the raw reply resting in the run
    # store otherwise). Raw handlers here; stamp-verified when the door
    # wraps the routing. A host cannot forget the privacy boundary because
    # the composition never offers it unwrapped.
    from ..core.models import EffectType as _ET
    from .act_only import wrap_llm_handler_with_act_only

    # A-ruling (2026-07-20): the ref/dereference layer is DELETED — the
    # wrapper now carries only the WRITE-boundary diary capture (words fly
    # to the book before the result rests). Diary tool results rest AS
    # SERVED in the durable transcript (the home is the privacy boundary).
    def _author_diary(run: Any, text: str, need_title: bool) -> dict:
        """A public diary entry's summary (and its title when he elected no gist), authored through the host's
        summaries capability handler BEFORE the entry is written (Codex review 2 #10, review 3 #9). No handler = the
        authoring failure (capability_unrouted), said."""
        from ..core.models import Effect as _Effect
        from .authoring import AUTHORING_CAPABILITY, AUTHORING_CAPABILITY_KEY, authoring_llm_call_payload
        from .authoring import diary_authoring_payload

        def _payload(result: Any) -> dict:
            return diary_authoring_payload(result, need_title=need_title, artifacts=home.artifacts, body=text)

        target = dict(capability_handlers or {}).get(AUTHORING_CAPABILITY)
        if target is None:
            return _payload({"ok": False, "capability_unrouted": AUTHORING_CAPABILITY})
        fence.check("before the diary entry's authoring call")  # no model call after the operator's Stop
        try:
            out = target(run, _Effect(type=_ET.LLM_CALL, payload=authoring_llm_call_payload("diary", text)), None)
        except Exception as e:  # noqa: BLE001 - the entry is still written; its metadata forms pending, said
            return _payload({"ok": False, "absorbed_failure": f"{type(e).__name__}: {e}"})
        fence.check("after the diary entry's authoring call returned")  # a late reply is abandoned
        if getattr(out, "status", None) == "completed" and isinstance(getattr(out, "result", None), dict):
            return _payload({**out.result, AUTHORING_CAPABILITY_KEY: AUTHORING_CAPABILITY})
        return _payload({"ok": False, "absorbed_failure": str(getattr(out, "error", "") or "failed")})

    if _ET.LLM_CALL in handlers:
        handlers[_ET.LLM_CALL] = wrap_llm_handler_with_act_only(
            handlers[_ET.LLM_CALL],
            diary_write_handler=handlers.get(_ET.DIARY_WRITE),
            # A failed book write saves the reply here before failing the
            # turn (record-everything ruling, 2026-07-26) — the words stay
            # in the life even when the store misbehaves.
            rescue_dir=home_dir,
            author_diary=_author_diary,
        )

    if _ET.LLM_CALL in handlers or capability_handlers:
        handlers[_ET.LLM_CALL] = _route_capability_calls(handlers.get(_ET.LLM_CALL), dict(capability_handlers or {}))

    # Durable-write discipline (0067-M): oversized effect/result leaves in
    # the visit ledger offload into the HOME's artifact store (refs on disk,
    # rehydrated on read) — ledger rows stay bounded while the bytes remain
    # part of the life, inside the home directory like every other artifact.
    # The raw SqliteLedgerStore stays on the EntityRuntime handle (type-stable
    # for direct consumers); the Runtime writes through the offloading wrap.
    from ..storage.artifacts import FileArtifactStore
    from ..storage.offloading import OffloadingLedgerStore

    handlers = fence_handlers(handlers, fence)
    # The run's own records (ledger offload, evidence, outputs) are not the entity's acts: an unguarded twin of the
    # home's store (the same directory; writes are atomic renames, the catalog is SQLite).
    run_artifacts = FileArtifactStore(str(Path(home_dir) / "artifacts"))

    # Optional workflow registry: composed (subflow-calling) workflows — the
    # entity-life master and its brain subflows — need START_SUBWORKFLOW to
    # resolve child specs. Absent registry keeps single-spec behavior.
    _rt_kwargs: dict = {}
    if workflow_registry is not None:
        _rt_kwargs["workflow_registry"] = workflow_registry
    runtime = Runtime(
        run_store=run_store,
        ledger_store=OffloadingLedgerStore(ledger_store, artifact_store=run_artifacts),
        effect_handlers=handlers,
        artifact_store=run_artifacts,
        **_rt_kwargs,
    )
    return EntityRuntime(
        home=home,
        runtime=runtime,
        run_store=run_store,
        ledger_store=ledger_store,
        db=db,
        store_path=store_path,
    )
