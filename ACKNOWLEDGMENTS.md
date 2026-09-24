# Acknowledgments

AbstractRuntime is designed to pair with the wider Abstract ecosystem and the open-source Python tooling community.

This project depends on (and is shaped by) the following libraries.
The canonical dependency list lives in `pyproject.toml`.

## Runtime dependencies (core install)

- **abstractsemantics** (`>=0.0.5`) — structured schema registry support (declared in `pyproject.toml`, used in `src/abstractruntime/integrations/abstractmemory/effect_handlers.py` and VisualFlow execution wiring).
- **AbstractMemory** (`>=0.3.0`) — TripleStore models and store contract used by Runtime's `MEMORY_KG_*` effects. Durable/vector backend dependencies such as LanceDB remain selected by hosts.

## Runtime integrations
- **abstractcore** (`abstractcore[remote,tools,vision,voice,audio,music]>=2.15.1`) — LLM, tools, media, and capability integration used by the base `abstractruntime` install; the `apple` and `gpu` extras select `abstractcore[all-apple]` / `abstractcore[all-gpu]` at the same floor (declared in `pyproject.toml`, implementation under `src/abstractruntime/integrations/abstractcore/*`, docs: `docs/integrations/abstractcore.md`).
  - The AbstractCore integration uses **httpx** for remote mode (`src/abstractruntime/integrations/abstractcore/llm_client.py`) and **pydantic** for structured validation (`src/abstractruntime/integrations/abstractcore/effect_handlers.py`). These are provided by AbstractCore’s dependency set.
  - The `tools` extra of AbstractCore backs the base Runtime toolset and the `abstractruntime-mcp-worker` entry point.
- **RestrictedPython** (`>=7.0`, core install) — sandbox for VisualFlow “Code” nodes and pin expressions (`src/abstractruntime/visualflow_compiler/visual/code_executor.py`).
- **pypdf** and **reportlab** — VisualFlow `Read PDF` / `Write PDF` nodes in the base install.

## Build & test tooling

- **hatchling** — build backend (`pyproject.toml` `[build-system]`).
- **pytest** — test runner (`pytest.ini`, `tests/`).

And thanks to everyone who reports bugs, discusses design tradeoffs, and contributes improvements.

See also: `LICENSE`, `CONTRIBUTING.md`.
