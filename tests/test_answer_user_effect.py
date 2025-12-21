from abstractruntime import Effect, EffectType, Runtime, StepPlan, WorkflowSpec
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore


def test_answer_user_effect_completes_and_stores_message() -> None:
    rt = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())

    def show_node(run, ctx):
        return StepPlan(
            node_id="SHOW",
            effect=Effect(
                type=EffectType.ANSWER_USER,
                payload={"message": "Hello"},
                result_key="user_message",
            ),
            next_node="DONE",
        )

    def done_node(run, ctx):
        return StepPlan(
            node_id="DONE",
            complete_output={"msg": run.vars.get("user_message")},
        )

    wf = WorkflowSpec(
        workflow_id="wf_answer_user",
        entry_node="SHOW",
        nodes={"SHOW": show_node, "DONE": done_node},
    )

    run_id = rt.start(workflow=wf, vars={})
    state = rt.tick(workflow=wf, run_id=run_id)

    assert state.status.value == "completed"
    assert state.output == {"msg": {"message": "Hello"}}


def test_answer_user_terminal_effect_completes_run() -> None:
    rt = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())

    def show_only(run, ctx):
        return StepPlan(
            node_id="SHOW",
            effect=Effect(
                type=EffectType.ANSWER_USER,
                payload={"message": "Done"},
                result_key="_temp.msg",
            ),
            next_node=None,
        )

    wf = WorkflowSpec(
        workflow_id="wf_answer_user_terminal",
        entry_node="SHOW",
        nodes={"SHOW": show_only},
    )

    run_id = rt.start(workflow=wf, vars={})
    state = rt.tick(workflow=wf, run_id=run_id)

    assert state.status.value == "completed"
    assert state.output == {"success": True, "result": {"message": "Done"}}

