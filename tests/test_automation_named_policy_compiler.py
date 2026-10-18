"""Validation and safeguard compilation for durable named tool snapshots."""
import pytest


@pytest.mark.parametrize('policy', [[], {'source': 'forged'}, {'auto_approve_tools': [False]}, {'require_approval_tools': ['']}])
def test_named_policy_rejects_invalid_shapes(policy):
    from abstractruntime.automations.models import AutomationError, validate_policy
    with pytest.raises(AutomationError):
        validate_policy({'tool_policy': policy})


def test_named_policy_preserves_safeguards_and_require_wins():
    from abstractruntime.automations.controller import apply_named_tool_policy
    from abstractruntime.automations.models import validate_policy
    assert 'tool_policy' not in validate_policy({'tool_policy': None})
    for untrusted in (False, True):
        inputs = {'tools': ['execute_command', 'read_file', 'send_email', 'mcp::host::tool'],
                  '_runtime': {'allowed_tools': ['execute_command', 'read_file', 'send_email', 'mcp::host::tool']}}
        apply_named_tool_policy(inputs, {'auto_approve_tools': ['execute_command', 'read_file', 'send_email', 'mcp::host::tool', 'outside'],
                                         'require_approval_tools': ['read_file']}, untrusted_input=untrusted)
        compiled = inputs['_runtime']['tool_policy']
        assert 'execute_command' in compiled['auto_approve_tools']
        assert not {'send_email', 'read_file', 'outside'} & set(compiled['auto_approve_tools'])
        assert 'read_file' in compiled['require_approval_tools']
        assert 'send_email' in compiled['withheld_tools']
        assert compiled['source']
        if untrusted:
            assert compiled['untrusted_input'] is True
            assert 'mcp::host::tool' not in inputs['tools']
            assert 'mcp::host::tool' not in compiled['auto_approve_tools']


