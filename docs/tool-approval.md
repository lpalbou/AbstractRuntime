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

## Per-call refiners

Some tools carry a `risk_refiner` id on their row. A refiner may **only lower**
a single call below its band, at approval time, when it can *prove* the call
is safe; it can never raise the band, and any call it cannot prove holds the
ceiling (deny-safe).

`send_email` and `reply_email` (band `outreach`) declare
`send_email_recipient@v2`: a `send_email` whose every recipient is **the
registered user's own address** or **a recipient pre-authorised for the run**
auto-approves; anything else asks.

- Self is `_runtime.operator_email` (below). Pre-authorised recipients are
  `_runtime.email_allowed_recipients`: exact addresses (and `"self"`) that an
  automation takes from its definition (`policy.email_allowed_recipients`,
  default `["self"]`, see [automations.md](automations.md#tool-approval)).
  The key is model-unwritable: the controller sets it at admission, hosts pop
  client-supplied values (`strip_client_email_keys`), and a child run takes
  its parent's value.
- `reply_email` always asks: its recipients come from the original message,
  not from the arguments.
- A call with an argument key `send_email` does not accept asks.
- The refiner runs with or without a per-run `_runtime.tool_policy`. Without
  one, it lowers the call on top of the executor's static policy; the
  executor's `require_approval_tools` still wins.
- Decisions are per call and applied by name: a batch with one `send_email`
  that needs a person waits as a whole, even beside a send to self.
- Approval is separate from the account's recipient policy (allowlist or
  denylist), which AbstractCore enforces on every send afterwards
  (`guarded_send`, on every email send path). A client that auto-approves
  every tool for its user ("approve all") decides only the approval wait;
  the recipient policy still decides who can receive mail. See
  [email.md](email.md#sending-without-asking).

`send_email_recipient@v1` (self only) stays registered for older tool rows.
The rules below apply to both versions.

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

## Session tool policy: a conversation's standing word

A person can approve tools for a whole conversation ("approve all", "approve all web tools",
"approve `fetch_url`"), deny some, or reset. That decision is a **session tool policy**: one row
per `session_id` in a store outside run vars
(`src/abstractruntime/storage/session_tool_policy.py`: `SqliteSessionToolPolicyStore`,
`InMemorySessionToolPolicyStore`), installed with `Runtime(session_tool_policy_store=...)` or
`Runtime.set_session_tool_policy_store(store)`. Hosts write it (the gateway's
`/api/gateway/sessions/{session_id}/tool-policy` routes, which expand and clamp the request to
the tools the account is authorized for); the TOOL_CALLS handler reads the run's session row once
per batch, **before** it decides whether the batch must wait.

Store API (both stores): `get(session_id)`, `get_with_revision(session_id) -> (revision, policy)`
(the base of a host's guarded read-merge-write), `put(session_id, policy, if_revision=None)`,
`delete(session_id, if_revision=None, actor=None) -> revision`, `revision(session_id)`,
`history(session_id)`. Revisions only grow (a reset is a revision, `get` then answers `None`); an
`if_revision` mismatch raises `SessionToolPolicyConflict` (`.expected`, `.actual`), which the
gateway answers as a 409 carrying the current revision. A store read failure during a tool batch
is logged and keeps the normal decision (ask where the defaults ask), never a wider one.

Stored shape (the runtime reads only these keys):

```json
{"version": 1, "session_id": "s-123", "revision": 5,
 "approve": {"fetch_url": "kind:web"}, "deny": {"execute_command": "tool"}, "ask": ["send_email"],
 "set_by": {"tenant_id": "default", "user_id": "laurent", "client": "panel-chat"},
 "set_at": "2026-10-27T10:00:00+00:00"}
```

Decision order for each call:

1. The run's tool ceiling (`allowed_tools`): a call outside it is blocked first; a session policy
   never widens what a run may call.
2. The row is the one of the run tree's ROOT session (a child started with another `session_id`
   does not borrow that session's grant). No row: nothing changes.
3. `deny` → refused with no wait (`"Denied by this conversation's tool policy"`), in every run of
   the session, automation occurrences and untrusted-input runs included (narrowing is always
   safe). A deny set while an approval is shown also wins over that approval when it is answered.
   `approve` (5) never applies to untrusted-input runs or automation occurrences and their
   descendants.
4. Names the run policy lists in `withheld_tools` are never session-approved.
5. `approve` (minus `ask`, minus `deny`) counts as auto for the batch, over the start-time
   `require_approval_tools`, the gateway's default grant and the executor's default caution list.
6. Everything else as before (run policy, tier ceiling, refiners, static policy).

The approval decision stays per batch: if any call still needs approval, the whole batch waits as
before. A batch that executes only because of the session policy, and every denial, writes one
`emit_event` record named `abstract.tool_policy_decision` to the run's ledger
(`decision: auto_approved | denied`, `source: session_policy`, the calls with the selector that
matched, and the policy's `revision`, `set_by`, `set_at`). There is no wait, no waiting record and
no resume for these calls, so every client renders them as activity, never as a question. If the
record cannot be written, the batch is not auto-approved (it asks).

Every run of the session reads the same row: each turn's root run, delegated child runs (they
inherit `session_id`), runs resumed after a restart. A change applies from the next batch. A wait
that already exists is not answered by the policy: a client that offers "approve all" on a shown
wait writes the policy first, then answers that wait as a one-off approval.
