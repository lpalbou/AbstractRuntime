# Communication tools (`comms` toolset)

AbstractRuntime’s AbstractCore integration can expose an optional `comms` toolset (email, WhatsApp, Telegram). These tools are executed as **durable tool calls** via `EffectType.TOOL_CALLS`:
- tool requests/results are recorded in the **ledger** (`src/abstractruntime/core/models.py`)
- execution is controlled by the configured `ToolExecutor` (`src/abstractruntime/integrations/abstractcore/tool_executor.py`)

This document covers what is implemented in this repo: **toolset gating + wiring**. Provider credentials/config are defined by **AbstractCore tools**.

Implementation pointers (this repo):
- toolset gating: `src/abstractruntime/integrations/abstractcore/default_tools.py`
- tool execution: `src/abstractruntime/integrations/abstractcore/tool_executor.py`

## Email for each user

A host that serves users with their own mailboxes binds each run to its user's account and passes
`email_enabled=True` to the toolset functions for users with a connected account; see [email.md](email.md). The
email tools are `list_email_accounts`, `send_email`, `reply_email`, `list_emails`, `search_emails`, `read_email` and
`get_email_attachment`; `send_email` runs unattended only to the user's own address or to recipients pre-authorised
for the run ([tool-approval.md](tool-approval.md#per-call-refiners)), and file arguments (`attachments`,
`output_dir`) stay inside the run's workspace.

## Enable (opt-in, process-wide)

The `comms` toolset is disabled by default. For a single process-wide setup, enable it via env vars (checked by `default_tools.comms_tools_enabled()`):

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
- Install Runtime with `pip install abstractruntime`; AbstractCore tool integration and the MCP worker entry point are part of the base remote-light install.
- In untrusted deployments, prefer passthrough tools so a host/worker boundary approves and executes tool calls (`PassthroughToolExecutor` in `src/abstractruntime/integrations/abstractcore/tool_executor.py`).
- For local bridge-owned delivery flows, `ApprovalToolExecutor` can auto-run the Telegram send tools while requiring approval for email, WhatsApp, unknown tools, and write/command-style tools by default.
- Separate from the durable `TOOL_CALLS` path, Runtime also exposes **host wrappers** for operator-owned email and Telegram surfaces:
  - email helpers on `get_abstractcore_host_facade(runtime)` and `abstractruntime.integrations.abstractcore.comms_facade`
  - Telegram lifecycle/send wrappers in `abstractruntime.integrations.abstractcore.telegram_facade`
  - read/bootstrap helpers stay host-local and do not create run history by themselves
  - if an outbound send belongs to a run, prefer the durable run facade:
    `get_abstractcore_run_facade(runtime).send_email(...)` /
    `send_telegram_message(...)`
  - if that durable child run pauses for approval or passthrough execution, resume it via
    `get_abstractcore_run_facade(runtime).resume_tool_calls(...)`

## Credentials/config (provided by AbstractCore)

The actual comms tools live in AbstractCore:
- email + WhatsApp: `abstractcore.tools.comms_tools`
- Telegram: `abstractcore.tools.telegram_tools`

AbstractRuntime does **not** store secrets in run state. Secrets should be supplied as environment variables in the **process that executes the tool calls**.

Practical starting points (provided by AbstractCore; see `pyproject.toml` for the minimum supported version):
- Email: the account is configured in AbstractCore's email settings (`abstractcore email connect ...`, or the Email
  page of the AbstractCore consoles), with credentials encrypted at rest; a multi-user host supplies each run's
  account through `Runtime.set_email_context_resolver` ([email.md](email.md)). AbstractCore imports a pre-2.20
  configuration once and then ignores it (see AbstractCore's email documentation).
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
