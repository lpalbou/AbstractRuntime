## 010_examples_and_composition (planned)

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

### Acceptance criteria
- A developer can copy an example and have it running in < 5 minutes
- Examples use the simplified API (`run()`, `respond()`)
- Each example has clear comments explaining what's happening

