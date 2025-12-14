## 010_examples_and_composition (COMPLETED 2025-12-13)

### Goal
Provide concrete, runnable examples demonstrating the simplified API:
- Zero-config setup with `create_scheduled_runtime()`
- `run()` and `respond()` for simple workflows
- `ask_user` interrupt (pause for user input)
- `wait_until` (scheduled resumption)
- Tool passthrough (for AbstractCore integration)
- Workflow-as-node composition (after 011 is complete)

### Deliverables
- `examples/` directory with runnable Python scripts
- Each example should be self-contained and copy-pasteable

### Proposed Examples

1. **01_hello_world.py** — Minimal workflow with zero-config
2. **02_ask_user.py** — Pause for user input, resume with response
3. **03_wait_until.py** — Schedule a task for later
4. **04_multi_step.py** — Multi-node workflow with branching
5. **05_persistence.py** — File-based storage, survive restart
6. **06_llm_integration.py** — AbstractCore LLM call (requires abstractcore)
7. **07_react_agent.py** — Full ReAct agent with tools (requires abstractcore + abstractagent)

### Acceptance criteria
- A developer can copy an example and have it running in < 5 minutes
- Examples use the simplified API (`run()`, `respond()`)
- Each example has clear comments explaining what's happening
- Examples 1-5 work without external dependencies
- Examples 6-7 require abstractcore/abstractagent but are clearly documented

### Priority
**HIGH** - This is the most impactful improvement for adoption. Without examples, developers cannot understand how to use the library.

---

## Completion Notes

**Completed:** 2025-12-13

**Deliverables:**
- Created `examples/` directory with 7 runnable examples
- All examples tested and working

**Examples implemented:**
1. `01_hello_world.py` - Minimal workflow with zero-config ✅
2. `02_ask_user.py` - Pause for user input, resume with response ✅
3. `03_wait_until.py` - Schedule a task for later ✅
4. `04_multi_step.py` - Multi-node workflow with branching ✅
5. `05_persistence.py` - File-based storage, survive restart ✅
6. `06_llm_integration.py` - AbstractCore LLM call ✅
7. `07_react_agent.py` - Full ReAct agent with tools ✅

**Additional improvements made:**
- AbstractAgent now uses TOOL_CALLS effect for ledger recording
- Created RegistryToolExecutor for agent-specific tool execution
- Added observe_node to ReAct workflow for proper effect-based architecture

