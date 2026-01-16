import tempfile
from pathlib import Path


class _StubLLM:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, *, prompt, messages, system_prompt, media, tools, params):
        self.calls += 1
        assert isinstance(media, list) and media, "expected media list"
        p = media[0]
        assert isinstance(p, str) and p, "expected media to be a temp file path"
        with open(p, "rb") as f:
            assert f.read() == b"hello world"
        return {"content": "OK", "metadata": {}}


def test_llm_call_media_artifact_refs_persist_across_restart():
    """Level B: file-backed stores + restart simulation for artifact-backed media."""
    from abstractruntime import Effect, EffectType, Runtime, RunStatus, StepPlan, WorkflowSpec
    from abstractruntime.integrations.abstractcore.effect_handlers import build_effect_handlers
    from abstractruntime.storage.artifacts import FileArtifactStore
    from abstractruntime.storage.json_files import JsonFileRunStore, JsonlLedgerStore

    def wait_node(run, ctx):
        return StepPlan(
            node_id="WAIT",
            effect=Effect(
                type=EffectType.WAIT_EVENT,
                payload={"wait_key": "evt:go", "resume_to_node": "CALL"},
                result_key="evt_payload",
            ),
            next_node="CALL",
        )

    def call_node(run, ctx):
        ctx_ns = run.vars.get("context")
        attachments = ctx_ns.get("attachments") if isinstance(ctx_ns, dict) else None
        return StepPlan(
            node_id="CALL",
            effect=Effect(
                type=EffectType.LLM_CALL,
                payload={"prompt": "Hello", "media": attachments, "params": {}},
                result_key="llm_result",
            ),
            next_node="DONE",
        )

    def done_node(run, ctx):
        res = run.vars.get("llm_result")
        content = res.get("content") if isinstance(res, dict) else None
        return StepPlan(node_id="DONE", complete_output={"content": content})

    wf = WorkflowSpec(workflow_id="wf_restart_media", entry_node="WAIT", nodes={"WAIT": wait_node, "CALL": call_node, "DONE": done_node})

    with tempfile.TemporaryDirectory() as d:
        base = Path(d)

        artifact_store_1 = FileArtifactStore(base)
        meta = artifact_store_1.store(b"hello world", content_type="text/plain", run_id="session_memory_s1")
        attachments = [{"$artifact": meta.artifact_id, "filename": "notes.txt"}]

        rt1 = Runtime(
            run_store=JsonFileRunStore(base),
            ledger_store=JsonlLedgerStore(base),
            artifact_store=artifact_store_1,
            effect_handlers=build_effect_handlers(llm=_StubLLM(), artifact_store=artifact_store_1),
        )

        run_id = rt1.start(workflow=wf, vars={"context": {"attachments": attachments}})
        state1 = rt1.tick(workflow=wf, run_id=run_id)
        assert state1.status == RunStatus.WAITING

        # Restart simulation: new Runtime + new ArtifactStore instance (same base dir).
        artifact_store_2 = FileArtifactStore(base)
        llm2 = _StubLLM()
        rt2 = Runtime(
            run_store=JsonFileRunStore(base),
            ledger_store=JsonlLedgerStore(base),
            artifact_store=artifact_store_2,
            effect_handlers=build_effect_handlers(llm=llm2, artifact_store=artifact_store_2),
        )

        loaded = rt2.get_state(run_id)
        assert loaded.status == RunStatus.WAITING
        ctx_loaded = loaded.vars.get("context")
        assert isinstance(ctx_loaded, dict)
        assert ctx_loaded.get("attachments") == attachments

        final = rt2.resume(
            workflow=wf,
            run_id=run_id,
            wait_key=loaded.waiting.wait_key,  # type: ignore[union-attr]
            payload={"go": True},
            max_steps=10,
        )
        assert final.status == RunStatus.COMPLETED
        assert final.output == {"content": "OK"}
        assert llm2.calls == 1

        # Ensure durable trace contains only the artifact refs (not temp paths).
        llm_result = final.vars.get("llm_result")
        assert isinstance(llm_result, dict)
        meta_out = llm_result.get("metadata")
        assert isinstance(meta_out, dict)
        obs = meta_out.get("_runtime_observability")
        assert isinstance(obs, dict)
        kwargs = obs.get("llm_generate_kwargs")
        assert isinstance(kwargs, dict)
        assert kwargs.get("media") == attachments

