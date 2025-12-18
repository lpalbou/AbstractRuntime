## 015_agent_integration_improvements (planned)

> Update (2025-12-15): Much of this item has been addressed by the framework-wide durable tool execution work:
> - AbstractAgent's ReAct workflow now uses `EffectType.TOOL_CALLS` (ledger-recorded, retry/idempotency-capable).
> - Tool execution is via host-configured `ToolExecutor` (durable; no callables in `RunState.vars`).
> - BaseAgent save/load is reference-only and delegates to RunStore.
>
> Remaining work is mostly *ergonomics* (a tiny convenience constructor in the integration layer), not core correctness.

### Goal
Improve the integration experience between AbstractRuntime and agent implementations (like AbstractAgent's ReactAgent).

### Context
Based on building AbstractAgent on top of AbstractRuntime, several friction points were identified:

1. **Verbose agent setup** - Creating an agent requires multiple steps:
   ```python
   from abstractruntime.integrations.abstractcore import MappingToolExecutor, create_local_runtime
   from abstractagent.agents.react import ReactAgent
   from abstractagent.tools import ALL_TOOLS

   tools = list(ALL_TOOLS)
   runtime = create_local_runtime(
       provider="ollama",
       model="...",
       tool_executor=MappingToolExecutor.from_tools(tools),
   )
   agent = ReactAgent(runtime=runtime, tools=tools)
   ```

2. **Tool execution durability** - Tool callables must never be stored in `RunState.vars` (must be JSON-safe across resume).

3. **No run_id persistence at agent level** - Resuming an agent across process restarts requires manually tracking the run_id.

### Proposed Improvements

#### 1. Agent Factory Pattern
Add a convenience factory in the AbstractCore integration:

```python
# Current (still a little verbose, but correct)
from abstractruntime.integrations.abstractcore import MappingToolExecutor, create_local_runtime
from abstractagent.agents.react import ReactAgent

tools = [...]
runtime = create_local_runtime(
    provider="ollama",
    model="...",
    tool_executor=MappingToolExecutor.from_tools(tools),
)
agent = ReactAgent(runtime=runtime, tools=tools)

# Proposed (simple)
from abstractruntime.integrations.abstractcore import create_agent_runtime

runtime = create_agent_runtime(
    provider="ollama",
    model="qwen3:4b-instruct-2507-q4_K_M",
    tools=[list_files, read_file],  # Direct tool functions
)
```

#### 2. TOOL_CALLS Effect for Agent Tool Execution
Modify the ReAct workflow to use TOOL_CALLS effect instead of direct execution (now implemented):

```python
# Previous (pre-effect): direct execution in act_node
# result = tool_registry.execute_tool(tool_call)

# Proposed: Via effect system
return StepPlan(
    node_id="act",
    effect=Effect(
        type=EffectType.TOOL_CALLS,
        payload={"tool_calls": pending_tool_calls},
        result_key="tool_results",
    ),
    next_node="observe",
)
```

Benefits:
- Tool calls recorded in ledger
- Retry/idempotency support
- Consistent architecture

#### 3. Agent State Persistence
Add optional run_id persistence to ReactAgent (now supported via BaseAgent.save_state/load_state):

```python
agent = ReactAgent(runtime=runtime, state_file="agent_state.json")
agent.start("task")  # Saves run_id to file
# ... process restart ...
agent = ReactAgent(runtime=runtime, state_file="agent_state.json")
agent.resume_from_file()  # Loads run_id and continues
```

### Acceptance Criteria
- [ ] Agent creation requires ≤3 lines of code for simple cases
- [x] Tool execution is recorded in the ledger
- [x] Agent can resume across process restarts with minimal code

### Dependencies
- AbstractCore tool registry
- Existing TOOL_CALLS effect handler

### Priority
Medium - Improves developer experience but not blocking

### Effort
Medium - 2-3 days
