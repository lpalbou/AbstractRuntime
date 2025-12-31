"""abstractruntime.integrations.abstractcore.constants

Single source of truth for AbstractRuntime orchestration defaults when it
executes AbstractCore-backed effects (LLM calls and tool calls).
"""

# Default *effect* timeouts (seconds).
#
# IMPORTANT: These are NOT a "workflow/run TTL". Workflows can be long-lived
# (hours/days) or even continuous. These limits only apply to a *single*
# runtime-managed operation (e.g. one LLM HTTP request, one tool call).
#
# Rationale:
# - Local inference can be slow for large contexts.
# - In an orchestrator, timeouts are policy and should be explicit + consistent.
DEFAULT_LLM_TIMEOUT_S: float = 7200.0
DEFAULT_TOOL_TIMEOUT_S: float = 7200.0


