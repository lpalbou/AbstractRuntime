# Communication Tools (Email + WhatsApp)

This document explains how to **enable** and **use** the comms toolset (email + WhatsApp) when running AbstractRuntime workflows through hosts such as **AbstractFlow** and **AbstractGateway**.

These capabilities are implemented as **durable tools** executed via `EffectType.TOOL_CALLS` (not new runtime EffectTypes). This means:
- tool requests/results are recorded in the **ledger** (observable + replayable),
- execution is controlled by the host’s `ToolExecutor` (local, passthrough/approval, or remote worker).

## Toolset overview

When enabled, the `comms` toolset adds these tools:

**Email (SMTP/IMAP; stdlib)**
- `send_email` (SMTP)
- `list_emails` (IMAP digest: since + all/unread/read)
- `read_email` (IMAP UID → headers + body)

**WhatsApp (provider-backed; v1 = Twilio REST)**
- `send_whatsapp_message`
- `list_whatsapp_messages`
- `read_whatsapp_message`

## Enabling the toolset (opt-in)

The comms tools are **disabled by default** and only appear in AbstractRuntime’s default tool discovery when you explicitly enable them:

- `ABSTRACT_ENABLE_COMMS_TOOLS=1` (enable email + WhatsApp)
- `ABSTRACT_ENABLE_EMAIL_TOOLS=1` (enable email only)
- `ABSTRACT_ENABLE_WHATSAPP_TOOLS=1` (enable WhatsApp only)

This affects:
- `abstractruntime.integrations.abstractcore.default_tools.get_default_tools()`
- `abstractruntime.integrations.abstractcore.default_tools.build_default_tool_map()`
- `abstractruntime.integrations.abstractcore.default_tools.list_default_tool_specs()` (what UIs use to populate allowlists)

Quick sanity check:

```bash
python - <<'PY'
from abstractruntime.integrations.abstractcore.default_tools import list_default_tool_specs
names = [s.get("name") for s in list_default_tool_specs() if s.get("toolset") == "comms"]
print(names)
PY
```

## Credentials and secrets (important)

These tools intentionally avoid passing secrets in tool-call arguments. Instead, they resolve credentials from **environment variables** in the process that executes the tool.

### Email
- Default env var: `EMAIL_PASSWORD`
- Override per call via `password_env_var` (e.g. `"password_env_var": "GMAIL_APP_PASSWORD"`)

### WhatsApp (Twilio v1)
- `TWILIO_ACCOUNT_SID`
- `TWILIO_AUTH_TOKEN`

If the env var is missing at execution time, the tool returns `{ "success": false, "error": "..." }` (JSON-safe).

## Using from AbstractFlow (authoring UI)

AbstractFlow’s backend exposes tool specs to the UI via `GET /api/tools`.
That endpoint is backed by AbstractRuntime’s `list_default_tool_specs()`.

To make comms tools show up in the **Tools Allowlist** selectors:
- run the AbstractFlow backend with `ABSTRACT_ENABLE_COMMS_TOOLS=1`

In the editor:
1. Add a **Tools Allowlist** node and enable the `comms` tools you want.
2. Use one of these patterns:

### Pattern A (LLM decides tool call)
- `LLM Call` node (connect the allowlist to its `tools` pin).
- Connect `LLM Call.tool_calls` → `Tool Calls.tool_calls`.
- The `Tool Calls` node executes those calls via the runtime.

### Pattern B (deterministic tool call)
- Build a literal tool call object `{name, arguments}` (e.g. via **Literal JSON** + **Make Array**).
- Connect it to `Tool Calls.tool_calls`.

Tool call shape (same as any other tool):

```json
[
  {
    "name": "send_email",
    "arguments": {
      "smtp_host": "smtp.gmail.com",
      "username": "xxx@gmail.com",
      "password_env_var": "EMAIL_PASSWORD",
      "to": "you@example.com",
      "subject": "Hello",
      "body_text": "Hi!"
    },
    "call_id": "optional-id"
  }
]
```

## Using from AbstractGateway

### Bundle mode (typical gateway deployment)
In bundle mode, the gateway wires tool execution based on `ABSTRACTGATEWAY_TOOL_MODE`:

- `ABSTRACTGATEWAY_TOOL_MODE=passthrough` (default, safest)
  - `TOOL_CALLS` becomes a durable **WAIT** for external execution/approval.
  - A client/tool-worker must execute the tool call(s) and resume the run with results.
- `ABSTRACTGATEWAY_TOOL_MODE=local` (dev / trusted deployments)
  - the gateway executes the default tool map in-process.

To allow comms tool execution in bundle mode:
1. Enable comms tools in the gateway process:
   - `ABSTRACT_ENABLE_COMMS_TOOLS=1`
2. Ensure credentials exist in that same environment:
   - email: `EMAIL_PASSWORD` (or call-specific `password_env_var`)
   - whatsapp: `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`
3. Choose execution mode:
   - passthrough (default) if you want an approval/worker boundary
   - local if the gateway itself is trusted to send messages

### VisualFlow host mode (gateway running VisualFlows directly)
The VisualFlow host uses AbstractFlow’s workspace-scoped local tool executor. With comms enabled, the tools are executed in the gateway process (similar trust implications as local mode above).

## Ledger / privacy note

Tool calls and results are durable and observable:
- the ledger records the tool call payload (including message bodies and recipients),
- secrets are not recorded if you rely on env-var resolution as intended.

Treat ledger/run storage as sensitive when using comms tools.

## See also
- `docs/misc/communication-tools.md` (practical Gmail + WhatsApp examples and credential setup)
