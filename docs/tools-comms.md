# Communication tools (`comms` toolset)

AbstractRuntime’s AbstractCore integration can expose an optional `comms` toolset (email, WhatsApp, Telegram). These tools are executed as **durable tool calls** via `EffectType.TOOL_CALLS`:
- tool requests/results are recorded in the **ledger** (`src/abstractruntime/core/models.py`)
- execution is controlled by the configured `ToolExecutor` (`src/abstractruntime/integrations/abstractcore/tool_executor.py`)

This document covers what is implemented in this repo: **toolset gating + wiring**. Provider credentials/config are defined by **AbstractCore tools**.

Implementation pointers (this repo):
- toolset gating: `src/abstractruntime/integrations/abstractcore/default_tools.py`
- tool execution: `src/abstractruntime/integrations/abstractcore/tool_executor.py`

## Enable (opt-in)

The `comms` toolset is disabled by default. Enable it via env vars (checked by `default_tools.comms_tools_enabled()`):

- `ABSTRACT_ENABLE_COMMS_TOOLS=1` (enable email + WhatsApp + Telegram)
- `ABSTRACT_ENABLE_EMAIL_TOOLS=1` (email only)
- `ABSTRACT_ENABLE_WHATSAPP_TOOLS=1` (WhatsApp only)
- `ABSTRACT_ENABLE_TELEGRAM_TOOLS=1` (Telegram only)

## Discover what gets enabled

```bash
python - <<'PY'
from abstractruntime.integrations.abstractcore.default_tools import list_default_tool_specs
comms = [s for s in list_default_tool_specs() if s.get("toolset") == "comms"]
print([s.get("name") for s in comms])
PY
```

## Wire into a runtime (local tool execution)

```python
import os

from abstractruntime.integrations.abstractcore import MappingToolExecutor, create_local_runtime
from abstractruntime.integrations.abstractcore.default_tools import get_default_tools

os.environ["ABSTRACT_ENABLE_COMMS_TOOLS"] = "1"

tool_executor = MappingToolExecutor.from_tools(get_default_tools())
rt = create_local_runtime(provider="ollama", model="qwen3:4b", tool_executor=tool_executor)
```

Notes:
- Install extras: `pip install "abstractruntime[abstractcore]"` (or `abstractruntime[mcp-worker]`).
- In untrusted deployments, prefer passthrough tools so a host/worker boundary approves and executes tool calls (`PassthroughToolExecutor` in `src/abstractruntime/integrations/abstractcore/tool_executor.py`).

## Credentials/config (provided by AbstractCore)

The actual comms tools live in AbstractCore:
- email + WhatsApp: `abstractcore.tools.comms_tools`
- Telegram: `abstractcore.tools.telegram_tools`

AbstractRuntime does **not** store secrets in run state. Secrets should be supplied as environment variables in the **process that executes the tool calls**.

Practical starting points (as implemented in AbstractCore v2.11.0):
- Email:
  - `ABSTRACT_EMAIL_ACCOUNTS_CONFIG=/path/to/emails.yaml` (YAML/JSON config), or `ABSTRACT_EMAIL_{IMAP,SMTP}_*` env vars
  - Passwords are resolved indirectly via `*_PASSWORD_ENV_VAR` (default: `EMAIL_PASSWORD`)
  - Repo templates:
    - `emails.config.example.yaml` (static examples: OVH + Gmail)
    - `configs/emails.yaml` (default config that inherits from `ABSTRACT_EMAIL_*` env vars via `${ENV_VAR}` interpolation)
  - YAML/JSON value interpolation:
    - `${ENV_VAR}` (required)
    - `${ENV_VAR:-default}` (optional default; can be empty)
- WhatsApp (Twilio):
  - defaults use `TWILIO_ACCOUNT_SID` and `TWILIO_AUTH_TOKEN`
- Telegram:
  - transport selection via `ABSTRACT_TELEGRAM_TRANSPORT` (`tdlib` default, or `bot_api`)
  - bot token default env var: `ABSTRACT_TELEGRAM_BOT_TOKEN`

## Security and privacy notes

- Tool calls and results are durable: message bodies, recipients, and response metadata may be persisted in the ledger and/or checkpoint vars.
- Keep secrets out of tool arguments; prefer env-var resolution. Even when a tool accepts `*_env_var` parameters, those should be **names**, not secret values.
- Treat run storage and ledgers as sensitive when enabling comms tools.

## See also

- `integrations/abstractcore.md` — AbstractCore wiring (`LLM_CALL`, `TOOL_CALLS`)
- `provenance.md` — tamper-evident ledger
