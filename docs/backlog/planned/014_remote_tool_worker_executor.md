## 014_remote_tool_worker_executor (planned)

### Goal
Provide an optional remote tool execution pathway:
- runtime submits `tool_calls` to a remote worker service
- runtime waits for job completion (`WaitReason.JOB`)
- host resumes with results

### Context / problem
Today, remote/untrusted mode uses tool passthrough:
- runtime returns WAITING with tool calls
- the host executes tools and resumes

That is correct, but sometimes you want:
- centralized tool execution on a backend
- thin clients that cannot execute tools

### Non-goals
- No mandatory worker service.
- No cluster leasing semantics.

---

### Proposed design

#### A) Worker API contract (minimal)

- `POST /v1/tool_jobs`
  - request: `{run_id, tool_calls: [...], metadata?}`
  - response: `{job_id}`

- `GET /v1/tool_jobs/{job_id}`
  - response: `{status: pending|running|completed|failed, result?, error?}`

Alternatively, the worker can emit events and the host resumes runs.

#### B) Runtime integration
Add a `RemoteToolExecutor` implementation:
- on execute(tool_calls):
  - POST tool job
  - return `{mode: "job", job_id, tool_calls}`

Update the `TOOL_CALLS` effect handler:
- if mode == job:
  - return `WAITING` with `WaitReason.JOB` and `wait_key=job_id`

#### C) Scheduler/driver integration
A separate driver process can:
- poll job statuses
- resume runs when completed

---

### Files to add / modify
- `src/abstractruntime/integrations/abstractcore/tool_executor.py` (add RemoteToolExecutor)
- new module: `src/abstractruntime/integrations/tool_worker_client.py`
- docs page explaining worker contract
- tests:
  - stub worker client
  - verify WAITING semantics for job mode

---

### Acceptance criteria
- Thin clients can still run workflows with tools by delegating to a worker.
- The runtime remains durable and does not keep in-flight tool execution in RAM.

### Test plan
- Unit tests with stubbed HTTP sender.
