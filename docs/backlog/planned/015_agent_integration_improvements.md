## 015_agent_integration_improvements (planned)

### Goal
Improve the integration experience between AbstractRuntime and agent implementations (like AbstractAgent's ReactAgent).

### Context
Based on building AbstractAgent on top of AbstractRuntime, several friction points were identified:

1. **Verbose agent setup** - Creating an agent requires multiple steps:
   ```python
   tool_registry = ToolRegistry()
   for tool in ALL_TOOLS:
       tool_registry.register(tool)
   runtime = create_local_runtime(provider="ollama", model="...")
   agent = ReactAgent(runtime=runtime, tool_registry=tool_registry)
   ```

2. **Tool execution not using TOOL_CALLS effect** - The ReAct workflow executes tools directly in Python, bypassing the effect system. This means:
   - Tool calls aren't recorded in the ledger
   - No retry/idempotency for tool execution
   - Inconsistent with the effect-based architecture

3. **No run_id persistence at agent level** - Resuming an agent across process restarts requires manually tracking the run_id.

### Proposed Improvements

#### 1. Agent Factory Pattern
Add a convenience factory in the AbstractCore integration:

```python
# Current (verbose)
runtime = create_local_runtime(provider="ollama", model="...")
agent = ReactAgent(runtime=runtime, tool_registry=registry)

# Proposed (simple)
from abstractruntime.integrations.abstractcore import create_agent_runtime

runtime = create_agent_runtime(
    provider="ollama",
    model="qwen3:4b-instruct-2507-q4_K_M",
    tools=[list_files, read_file],  # Direct tool functions
)
```

#### 2. TOOL_CALLS Effect for Agent Tool Execution
Modify the ReAct workflow to use TOOL_CALLS effect instead of direct execution:

```python
# Current: Direct execution in act_node
result = tool_registry.execute_tool(tool_call)

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
Add optional run_id persistence to ReactAgent:

```python
agent = ReactAgent(runtime=runtime, state_file="agent_state.json")
agent.start("task")  # Saves run_id to file
# ... process restart ...
agent = ReactAgent(runtime=runtime, state_file="agent_state.json")
agent.resume_from_file()  # Loads run_id and continues
```

### Acceptance Criteria
- [ ] Agent creation requires ≤3 lines of code for simple cases
- [ ] Tool execution is recorded in the ledger
- [ ] Agent can resume across process restarts with minimal code

### Dependencies
- AbstractCore tool registry
- Existing TOOL_CALLS effect handler

### Priority
Medium - Improves developer experience but not blocking

### Effort
Medium - 2-3 days
