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
# The NO-PROGRESS bound (0152 face 2, core c5051 + runtime c5041): httpx's
# READ timeout separated from the absolute total above — a stream that
# delivers nothing for this long dies at the socket instead of pinning a
# tick worker for the 2h backstop. None disables (core's byte-identical
# pre-fix behavior); entity lanes thread tighter numbers.
DEFAULT_LLM_READ_IDLE_TIMEOUT_S: float = 300.0
DEFAULT_TOOL_TIMEOUT_S: float = 7200.0


