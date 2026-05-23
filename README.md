# AbstractRuntime

**AbstractRuntime** is a durable workflow runtime (interrupt → checkpoint → resume) with an append-only execution ledger.

It is designed for long-running workflows that must survive restarts and explicitly model blocking (human input, timers, external events, subworkflows) without keeping Python stacks alive.

**Version:** 0.4.22 • **Python:** 3.10+

**Status:** pre-1.0 (API may evolve). For production use, pin versions and follow `CHANGELOG.md`.

## AbstractFramework ecosystem

AbstractRuntime is one component of the wider [AbstractFramework](https://github.com/lpalbou/AbstractFramework) ecosystem:
- **AbstractRuntime** (this repo) — durable workflow kernel (`src/abstractruntime/core/*`)
- **AbstractCore** — LLM + tools integration (wired via `src/abstractruntime/integrations/abstractcore/*`)  
  Repo: [lpalbou/abstractcore](https://github.com/lpalbou/abstractcore)

At a high level, hosts define workflow graphs (`WorkflowSpec`) and AbstractRuntime executes them durably. When nodes request LLM/tool work (`EffectType.LLM_CALL`, `EffectType.TOOL_CALLS`), those effects are typically handled via AbstractCore.

```mermaid
flowchart LR
  Host["Host app / orchestrator"] -->|"WorkflowSpec"| RT["AbstractRuntime"]
  RT -->|"LLM_CALL / TOOL_CALLS"| AC["AbstractCore"]
  AC -->|"results / waits"| RT
```

## Install

Core runtime:

```bash
pip install abstractruntime
```

AbstractCore integration (LLM + tools):

```bash
pip install "abstractruntime[abstractcore]"
```

The `abstractcore` extra installs AbstractCore 2.13.27 or newer so the hardened server auth model, provider-key header routing, generated-media contracts, capability catalog, prompt-cache control plane, durable bloc prompt-cache helpers, bindings, lifecycle operations, public output-selector contract, async/sync text-generation output-selector parity, task-aware model residency for text/image/TTS/STT, and the public local vision-cache catalog helper used by Runtime discovery remain aligned. Use `abstractruntime[multimodal]` when you need common media dependencies and capability plugins (installs `abstractcore[remote,vision,voice,audio,music]>=2.13.27`, including `abstractmusic>=0.1.12`).

Hardware profile cascades are available for native Python installs:
`abstractruntime[apple]`, `abstractruntime[gpu]`, `abstractruntime[all-apple]`,
and `abstractruntime[all-gpu]` delegate to the matching AbstractCore profile
without making the runtime kernel itself hardware-specific.

MCP worker entrypoint (default toolsets over stdio):

```bash
pip install "abstractruntime[mcp-worker]"
```

## Quick start (pause + resume)

```python
from abstractruntime import Effect, EffectType, Runtime, StepPlan, WorkflowSpec
from abstractruntime.storage import InMemoryLedgerStore, InMemoryRunStore


def ask(run, ctx):
    return StepPlan(
        node_id="ask",
        effect=Effect(
            type=EffectType.ASK_USER,
            payload={"prompt": "Continue?"},
            result_key="user_answer",
        ),
        next_node="done",
    )


def done(run, ctx):
    answer = run.vars.get("user_answer") or {}
    text = answer.get("text") if isinstance(answer, dict) else None
    return StepPlan(node_id="done", complete_output={"answer": text})


wf = WorkflowSpec(workflow_id="demo", entry_node="ask", nodes={"ask": ask, "done": done})
rt = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())

run_id = rt.start(workflow=wf)
state = rt.tick(workflow=wf, run_id=run_id)
assert state.status.value == "waiting"

state = rt.resume(
    workflow=wf,
    run_id=run_id,
    wait_key=state.waiting.wait_key,
    payload={"text": "yes"},
)
assert state.status.value == "completed"
```

## What’s included (v0.4.22)

Kernel (dependency-light):
- workflow graphs: `WorkflowSpec` (`src/abstractruntime/core/spec.py`)
- durable execution: `Runtime.start/tick/resume` (`src/abstractruntime/core/runtime.py`)
- durable waits/events: `WAIT_EVENT`, `WAIT_UNTIL`, `ASK_USER`, `EMIT_EVENT`
- append-only ledger (`StepRecord`) + node traces (`vars["_runtime"]["node_traces"]`)
- retries/idempotency hooks: `src/abstractruntime/core/policy.py`
- runtime-aware limits (`_limits`) with a default iteration budget of 50 (`docs/limits.md`)

Durability + storage:
- stores: in-memory, JSON/JSONL, SQLite (`src/abstractruntime/storage/*`)
- durable command inbox primitives (idempotent, append-only): `CommandStore`, `CommandCursorStore` (`src/abstractruntime/storage/commands.py`, `src/abstractruntime/storage/sqlite.py`)
- artifacts + offloading (store large payloads by reference)
- snapshots/bookmarks (`docs/snapshots.md`)
- tamper-evident hash-chained ledger (`docs/provenance.md`)

Drivers + distribution:
- scheduler: `create_scheduled_runtime()` (`src/abstractruntime/scheduler/*`)
- VisualFlow compiler + WorkflowBundles (`src/abstractruntime/visualflow_compiler/*`, `src/abstractruntime/workflow_bundle/*`)
- VisualFlow multi-entry execution lowering for fan-in routes and per-entry input overrides (`docs/workflow-bundles.md`)
- run history export: `export_run_history_bundle(...)` (`src/abstractruntime/history_bundle.py`)

Optional integrations:
- AbstractCore (LLM + tools, `MODEL_RESIDENCY`, public discovery/host/run facades, cached sessions, local-only prompt-cache export/import admin, durable bloc prompt-cache controls, bindings, lifecycle operations, generated image/voice/music outputs, host email helpers, Telegram host wrappers, and tool approval waits): `docs/integrations/abstractcore.md`
- For outbound comms, use the durable run facade when the send belongs to a run: `get_abstractcore_run_facade(...).send_email(...)` / `send_telegram_message(...)`. If that child run pauses for approval or passthrough execution, resume it through `resume_tool_calls(...)`. Direct host-facade send helpers and the standalone email comms facade remain host-local and nondurable.
- AbstractMemory TripleStore integration for `MEMORY_KG_*` effects. Runtime
  depends on the light AbstractMemory contract; hosts choose storage backends
  such as LanceDB, SQLite, or in-memory stores.
- comms toolset gating (email/WhatsApp/Telegram): `docs/tools-comms.md`

## Built-in scheduler (zero-config)

```python
from abstractruntime import create_scheduled_runtime

sr = create_scheduled_runtime()
run_id, state = sr.run(my_workflow)

if state.status.value == "waiting":
    state = sr.respond(run_id, {"text": "yes"})

sr.stop()
```

For persistent storage:

```python
from abstractruntime import create_scheduled_runtime, JsonFileRunStore, JsonlLedgerStore

sr = create_scheduled_runtime(
    run_store=JsonFileRunStore("./data"),
    ledger_store=JsonlLedgerStore("./data"),
)
```

## Documentation

| Document | Description |
|----------|-------------|
| [Getting Started](docs/getting-started.md) | Install + first durable workflow |
| [API Reference](docs/api.md) | Public API surface (imports + pointers) |
| [Docs Index](docs/README.md) | Full docs map (guides + reference) |
| [FAQ](docs/faq.md) | Common questions and gotchas |
| [Troubleshooting](docs/troubleshooting.md) | Symptom-oriented setup, runtime, and integration fixes |
| [Architecture](docs/architecture.md) | Component map + diagrams |
| [Overview](docs/proposal.md) | Design goals, core concepts, and scope |
| [Integrations](docs/integrations/) | Integration guides (AbstractCore) |
| [Snapshots](docs/snapshots.md) | Named checkpoints for run state |
| [Provenance](docs/provenance.md) | Tamper-evident ledger documentation |
| [Evidence](docs/evidence.md) | Artifact-backed evidence capture for web/command tools |
| [Limits](docs/limits.md) | `_limits` namespace and RuntimeConfig |
| [WorkflowBundles](docs/workflow-bundles.md) | `.flow` bundle format (VisualFlow distribution) |
| [MCP Worker](docs/mcp-worker.md) | `abstractruntime-mcp-worker` CLI |
| [Changelog](CHANGELOG.md) | Release notes |
| [Contributing](CONTRIBUTING.md) | How to build/test and submit changes |
| [Code of Conduct](CODE_OF_CONDUCT.md) | Contributor conduct expectations |
| [Security](SECURITY.md) | Responsible vulnerability reporting |
| [Acknowledgments](ACKNOWLEDGMENTS.md) | Credits |
| [ROADMAP](ROADMAP.md) | Prioritized next steps |

## Development

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[abstractcore,mcp-worker,test,docs]"
python -m pytest -q
```

See `CONTRIBUTING.md` for contribution guidelines and doc conventions.
