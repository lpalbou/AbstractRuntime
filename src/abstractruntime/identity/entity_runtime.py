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


def open_entity_runtime(
    home_dir: Path,
    *,
    embedder: Any = None,
    attention_window: Optional[int] = None,
    extra_handlers: Optional[dict] = None,
) -> EntityRuntime:
    """Open an existing home and bind its own Runtime to it.

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
    """
    home = open_home(home_dir, embedder=embedder, attention_window=attention_window)
    store_path = entity_run_store_path(Path(home_dir))
    db = SqliteDatabase(store_path)
    run_store = SqliteRunStore(db)
    ledger_store = SqliteLedgerStore(db)

    handlers = dict(home.handlers)
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

    if _ET.LLM_CALL in handlers and _ET.DIARY_READ in handlers:
        handlers[_ET.LLM_CALL] = wrap_llm_handler_with_act_only(
            handlers[_ET.LLM_CALL],
            diary_read_handler=handlers[_ET.DIARY_READ],
            diary_write_handler=handlers.get(_ET.DIARY_WRITE),
        )

    runtime = Runtime(
        run_store=run_store,
        ledger_store=ledger_store,
        effect_handlers=handlers,
        artifact_store=home.artifacts,
    )
    return EntityRuntime(
        home=home,
        runtime=runtime,
        run_store=run_store,
        ledger_store=ledger_store,
        db=db,
        store_path=store_path,
    )
