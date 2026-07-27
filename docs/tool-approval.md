# Tool approval: risk tiers, the run-policy ceiling, and per-call refiners

AbstractRuntime decides, per tool call, whether to **auto-run** a tool or
**ask** the operator first. This is the runtime enforcement half of the
framework-wide *tool tiers* concept (operator ruling, tool-tiers wave 2026-07-23):
the gateway serves defaults and apps override them, but the decision that
actually gates execution runs here, at the effect boundary.

Implementation pointers (this repo):
- approval sets + policy: `src/abstractruntime/integrations/abstractcore/tool_executor.py`
- run-policy consumer + refiner dispatch: `src/abstractruntime/integrations/abstractcore/effect_handlers.py` (`_execute_with_run_policy`)
- risk facts + the served row shape: `src/abstractruntime/identity/tools.py` (`walled_tool_rows`), `src/abstractruntime/integrations/abstractcore/tool_inventory_facade.py` (`annotate_tool_rows`, `derive_risk_assessment`)
- the fact→tier mapping is **hosted by AbstractCore** (`abstractcore/tools/risk_facts.py`); runtime imports it (import-never-copy) and degrades to a byte-identical seed only under version skew.

## Two gates, never one

A tool call passes two independent gates:

1. **Availability** — is the tool present in the run's granted set at all?
   A tool above the run's grant is *absent* (not registered), not merely
   asked. Life-plane entity tools (memory/diary/reflection) are available by
   channel structure and are **not on the consent ladder** (`grantable:false`):
   stripping them is an identity risk the consent surface must be unable to
   express.
2. **Approval** — for an available, mutating-or-risky tool, auto-run or ask?

The gateway's grant model expresses both; this page documents the approval
gate, which is what `_execute_with_run_policy` computes.

## Risk facts and the derived tier

Every served tool row carries declared **facts** (never policy):
`mutating`, `remote_write_capable`, `comms_send`, `captures_environment`,
`standing_effect`, `destructive_capable`, plus the band-neutral
`model_controlled_destination`. A single versioned mapping derives the
operator-facing risk from those facts (max-wins):

| band (`risk_tier`) | `risk_rank` | facts |
| --- | --- | --- |
| `observe` | 1 | every declared fact false (read-only) |
| `act` | 2 | `mutating` or `remote_write_capable` |
| `outreach` | 3 | `comms_send` / `captures_environment` / `standing_effect` |
| `destroy` | 4 | `destructive_capable` (e.g. a shell reaching `rm`/`git reset`) |

Wire shape on every row: `risk_tier` is the **band word** (stable identity),
`risk_rank` is the **integer** (display/compare ordinal), `risk_presentation`
is the render word. A **factless** row (no declared facts — e.g. an
undeclared MCP tool) derives `risk_rank` 4 but `risk_presentation`
`"unvetted"`: gated at the top, never *rendered* "destroy" (deny-safe, but
honest that it is unvetted rather than proven-destructive).

## The run-policy ceiling

A run carries an optional policy under the model-unwritable
`_runtime.tool_policy` key (the gateway injects it; the model cannot set it):

- `auto_approve_tools` / `require_approval_tools` — explicit name lists, the
  finest grain. **`require` always wins** over any tier ceiling.
- `auto_approve_max_risk_rank` — a ceiling: a call whose tool derives
  `risk_rank <= N` auto-runs; above it, asks.

`model_controlled_destination` tools (the model chooses where output goes —
`fetch_url`, and see the refiner below) are **never silenced by the ceiling**:
the approval prompt is the exfiltration defense. Only an explicit
`auto_approve_tools` name — the operator's conscious act — overrides that.

## Per-call refiners (`send_email_recipient@v1`)

Some tools carry a `risk_refiner` id on their row. A refiner may **only lower**
a single call below its band, at approval time, when it can *prove* the call
is safe; it can never raise the band, and any call it cannot prove holds the
ceiling (deny-safe).

`send_email` (band `outreach`) declares `send_email_recipient@v1` (operator
ruling dm#244): a send to **the registered operator's own address**
auto-approves; a send to any other recipient asks.

- The operator address arrives as `_runtime.operator_email` — a
  **model-unwritable** key the gateway injects from the account record at run
  start (one source of truth; payload-supplied values are dropped).
- The refiner unions **all** recipient fields (`to`/`cc`/`bcc`); **every**
  recipient must equal the operator address (normalized strip + NFC +
  lowercase — no confusable/IDN folding, so a homoglyph domain never matches).
- Deny-safe at every gap → **ask**: no operator email configured (the feature
  is simply off — the email is optional), empty/unresolved recipients, a
  wrapper-nested argument shape, a display-name/group token
  (`Operator <op@self.com>` is compared verbatim, never bracket-parsed), any
  parse failure, or a version-skewed runtime with the refiner unregistered.

Until the operator email is configured, `send_email` stays in the
require-approval set: self and others both ask.
