from abstractruntime import Effect, EffectType, Runtime, StepPlan, WorkflowSpec
from abstractruntime.storage import InMemoryLedgerStore, InMemoryRunStore

def ask(run, ctx):
    return StepPlan(node_id="ask", effect=Effect(type=EffectType.ASK_USER,
        payload={"prompt": "Continue?"}, result_key="user_answer"), next_node="done")

def done(run, ctx):
    answer = run.vars.get("user_answer") or {}
    text = answer.get("text") if isinstance(answer, dict) else None
    return StepPlan(node_id="done", complete_output={"answer": text})

wf = WorkflowSpec(workflow_id="demo", entry_node="ask", nodes={"ask": ask, "done": done})
rt = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())
run_id = rt.start(workflow=wf)
state = rt.tick(workflow=wf, run_id=run_id)
assert state.status.value == "waiting", state.status
state = rt.resume(workflow=wf, run_id=run_id, wait_key=state.waiting.wait_key, payload={"text": "yes"})
assert state.status.value == "completed", state.status
print("README quick start: PASS")

from abstractruntime import create_scheduled_runtime, JsonFileRunStore, JsonlLedgerStore
print("README scheduler imports: PASS")
