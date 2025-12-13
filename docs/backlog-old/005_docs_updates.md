## Todo 5 — Docs updates (proposal + README)

### Goal
Update `abstractruntime/docs/proposal.md` and `abstractruntime/README.md` so they:
- reflect the actual kernel that exists (no longer “placeholder only”)
- document the AbstractCore integration module approach
- document snapshots/bookmarks
- document tamper-evident provenance (hash chain)
- clarify local/remote/hybrid execution modes

---

### 1) `abstractruntime/README.md` — required fixes

Current issue:
- The README currently says **“placeholder package (planning stage). No runtime implementation yet.”**
- But the repo already contains a real MVP kernel (`RunState`, `Runtime`, file persistence, pause/resume tests).

Required changes:

#### Replace the “Status” section
- Replace with something like:
  - **Status: MVP kernel implemented (v0.1), integration modules in progress**

#### Add a minimal “Quick start”
Show:
- define workflow
- start run
- tick until waiting
- resume

Example (doc-level pseudo; keep stable):

```python
from abstractruntime import Runtime, WorkflowSpec, StepPlan, Effect, EffectType
from abstractruntime.storage.in_memory import InMemoryRunStore, InMemoryLedgerStore


def n1(run, ctx):
    return StepPlan(
        node_id="n1",
        effect=Effect(type=EffectType.ASK_USER, payload={"prompt": "Continue?"}, result_key="user"),
        next_node="n2",
    )

def n2(run, ctx):
    return StepPlan(node_id="n2", complete_output={"ok": True, "answer": run.vars.get("user")})

workflow = WorkflowSpec(workflow_id="demo", entry_node="n1", nodes={"n1": n1, "n2": n2})
rt = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())
run_id = rt.start(workflow=workflow)

state = rt.tick(workflow=workflow, run_id=run_id)
assert state.status.value == "waiting"

state = rt.resume(workflow=workflow, run_id=run_id, wait_key=state.waiting.wait_key, payload={"text": "yes"})
assert state.status.value == "completed"
```

#### Add “Integration with AbstractCore” section
Document:
- `EffectType.LLM_CALL` and `EffectType.TOOL_CALLS`
- local vs remote vs hybrid
- that integration lives in `abstractruntime.integrations.abstractcore`

#### Add “Provenance” section
Explain:
- actor identity (`ActorFingerprint`, `RunState.actor_id`)
- tamper-evident hash chain (and what it does *not* guarantee)

---

### 2) `abstractruntime/docs/proposal.md` — required additions

`docs/proposal.md` already describes the kernel well.
Update it by adding sections (or extending existing ones) for:

#### AbstractCore integration layer
- Explain layered coupling explicitly:
  - core runtime stays independent
  - integration module imports AbstractCore
- Provide a short diagram of execution modes:
  - local / remote / hybrid

#### Snapshots/bookmarks
- Add a short definition + rationale:
  - operator-facing “save point”
  - UI-friendly named checkpoint
- Clarify restore semantics (workflow spec compatibility is not guaranteed in v0.1)

#### Tamper-evident provenance
- Add the hash chain concept
- Mention signatures as optional extra

---

### 3) Optional: one additional doc page (if you want)

If you want to keep README short, add a single page:
- `docs/integrations/abstractcore.md`

But this is optional; the todo only requires README + proposal.

---

### Deliverable checklist

- [ ] README no longer claims “no runtime implementation”
- [ ] README has a minimal Quick start
- [ ] README documents integration modes + effect types
- [ ] Proposal documents integration module + snapshots + provenance


