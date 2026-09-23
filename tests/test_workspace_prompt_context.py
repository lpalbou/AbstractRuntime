"""Policy must reach the model, not merely be stored in run variables."""

import pytest

from abstractruntime import Effect, EffectType, RunState
from abstractruntime.integrations.abstractcore.effect_handlers import make_llm_call_handler
from abstractruntime.integrations.abstractcore.workspace_scoped_tools import (
    WorkspaceScope, describe_workspace_scope, resolve_user_path,
)


class CaptureLLM:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return {"content": "ok"}


def scope_vars(tmp_path, mode="workspace_or_allowed"):
    return {
        "workspace_root": str(tmp_path / "chat"),
        "workspace_access_mode": mode,
        "workspace_allowed_paths": [str(tmp_path / ".cache")],
        "workspace_ignored_paths": [str(tmp_path / ".cache" / "private")],
    }


def test_model_receives_effective_scope_stable_across_react_calls(tmp_path):
    llm = CaptureLLM()
    handler = make_llm_call_handler(llm=llm)
    run = RunState.new(workflow_id="agent", entry_node="reason", vars=scope_vars(tmp_path))
    effect = Effect(type=EffectType.LLM_CALL, payload={
        "prompt": "Find the model in .cache", "system_prompt": "You are an assistant.",
        "tools": [{"name": "list_files", "description": "List files", "parameters": {"type": "object"}}],
    })
    for _ in range(2):
        assert handler(run, effect, None).status == "completed"
    prompt = llm.calls[0]["system_prompt"]
    assert prompt == llm.calls[1]["system_prompt"]
    assert prompt.startswith("You are an assistant.")
    assert prompt.count("Workspace access (") == 1
    assert str(tmp_path / "chat") in prompt
    assert f'"cache" -> "{tmp_path / ".cache"}"' in prompt
    assert str(tmp_path / ".cache" / "private") in prompt
    assert "not OS mounts" in prompt
    assert effect.payload["system_prompt"] == "You are an assistant."


def test_denial_names_existing_grants_and_alias_matches_resolver(tmp_path):
    scope = WorkspaceScope.from_input_data(scope_vars(tmp_path))
    assert resolve_user_path(scope=scope, user_path="cache") == tmp_path / ".cache"
    assert resolve_user_path(scope=scope, user_path=".cache") == tmp_path / "chat" / ".cache"
    assert resolve_user_path(scope=scope, user_path=str(tmp_path / ".cache")) == tmp_path / ".cache"
    with pytest.raises(ValueError) as exc:
        resolve_user_path(scope=scope, user_path=str(tmp_path))
    assert '"cache" ->' in str(exc.value)
    assert "absolute paths must stay under it" not in str(exc.value)
    with pytest.raises(ValueError, match="blocked by workspace_ignored_paths"):
        resolve_user_path(scope=scope, user_path="cache/private/file")


@pytest.mark.parametrize("mode", ["workspace_only", "all_except_ignored"])
def test_other_modes_do_not_advertise_inactive_mounts(tmp_path, mode):
    scope = WorkspaceScope.from_input_data(scope_vars(tmp_path, mode))
    prompt = describe_workspace_scope(scope)
    assert mode in prompt
    assert '"cache" ->' not in prompt
    assert "Excluded paths" in prompt


@pytest.mark.parametrize("with_tools,with_scope", [(False, True), (True, False)])
def test_no_workspace_disclosure_for_unscoped_or_tool_free_calls(tmp_path, with_tools, with_scope):
    llm = CaptureLLM()
    run = RunState.new(workflow_id="wf", entry_node="n", vars=scope_vars(tmp_path) if with_scope else {})
    effect = Effect(type=EffectType.LLM_CALL, payload={
        "prompt": "Hello", "system_prompt": "Original",
        "tools": [{"name": "list_files"}] if with_tools else [],
    })
    assert make_llm_call_handler(llm=llm)(run, effect, None).status == "completed"
    assert llm.calls[0]["system_prompt"] == "Original"
