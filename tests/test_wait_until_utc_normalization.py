"""WAIT_UNTIL due-ness is an ISO *string* comparison (tick auto-unblock and
every RunStore.list_due_wait_until implementation), which is only correct when
all deadlines share the aware-UTC representation. These tests pin the kernel's
write-boundary normalization: any ISO-8601 payload.until (offset, 'Z', naive)
is stored as +00:00, so string comparison equals real time comparison.

Regression context: a "+02:00" deadline that was already due compared as NOT
due against utc_now_iso() and fired hours late (found in the 0003 identity
round while designing entity self-timers; the deposit-gate invariant depends
on this kernel guarantee).
"""

from datetime import datetime, timedelta, timezone

from abstractruntime import Effect, EffectType, Runtime, StepPlan, WorkflowSpec
from abstractruntime.core.runtime import normalize_utc_iso
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore


def _wait_workflow(until_value: str) -> WorkflowSpec:
    def wait_node(run, ctx):
        return StepPlan(
            node_id="WAIT",
            effect=Effect(
                type=EffectType.WAIT_UNTIL,
                payload={"until": until_value, "resume_to_node": "DONE"},
            ),
            next_node="DONE",
        )

    def done_node(run, ctx):
        return StepPlan(node_id="DONE", complete_output={"ok": True})

    return WorkflowSpec(workflow_id="wf_wait_until_utc", entry_node="WAIT", nodes={"WAIT": wait_node, "DONE": done_node})


def _runtime() -> Runtime:
    return Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())


class TestNormalizeUtcIso:
    def test_offset_is_converted_to_utc(self):
        assert normalize_utc_iso("2026-07-06T23:46:00+02:00") == "2026-07-06T21:46:00+00:00"

    def test_z_suffix_is_accepted(self):
        assert normalize_utc_iso("2026-07-06T21:46:00Z") == "2026-07-06T21:46:00+00:00"

    def test_naive_is_assumed_utc(self):
        assert normalize_utc_iso("2026-07-06T21:46:00") == "2026-07-06T21:46:00+00:00"

    def test_invalid_returns_none(self):
        assert normalize_utc_iso("next tuesday") is None
        assert normalize_utc_iso("") is None
        assert normalize_utc_iso(None) is None


class TestWaitUntilHandlerNormalization:
    def test_past_deadline_with_offset_completes_immediately(self):
        """The regression: a deadline already past in real time, expressed with
        a non-UTC offset, must be recognized as due (string comparison against
        the raw value says "not due" for ~2h)."""
        past = datetime.now(timezone.utc) - timedelta(minutes=5)
        # Same instant, +02:00 representation — lexicographically "in the future".
        offset_repr = past.astimezone(timezone(timedelta(hours=2))).isoformat()
        assert offset_repr > datetime.now(timezone.utc).isoformat()  # the trap is real

        rt = _runtime()
        wf = _wait_workflow(offset_repr)
        run_id = rt.start(workflow=wf, vars={})
        state = rt.tick(workflow=wf, run_id=run_id, max_steps=10)
        assert state.status.value == "completed"

    def test_future_deadline_with_offset_is_stored_normalized(self):
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        offset_repr = future.astimezone(timezone(timedelta(hours=-7))).isoformat()

        rt = _runtime()
        wf = _wait_workflow(offset_repr)
        run_id = rt.start(workflow=wf, vars={})
        state = rt.tick(workflow=wf, run_id=run_id, max_steps=10)
        assert state.status.value == "waiting"
        assert state.waiting is not None
        assert state.waiting.until == future.isoformat()
        assert state.waiting.until.endswith("+00:00")

    def test_future_naive_deadline_waits(self):
        naive = (datetime.now(timezone.utc) + timedelta(hours=1)).replace(tzinfo=None).isoformat()
        rt = _runtime()
        wf = _wait_workflow(naive)
        run_id = rt.start(workflow=wf, vars={})
        state = rt.tick(workflow=wf, run_id=run_id, max_steps=10)
        assert state.status.value == "waiting"
        assert state.waiting is not None
        assert state.waiting.until.endswith("+00:00")

    def test_unparseable_deadline_fails_loudly(self):
        rt = _runtime()
        wf = _wait_workflow("tomorrow at nine")
        run_id = rt.start(workflow=wf, vars={})
        state = rt.tick(workflow=wf, run_id=run_id, max_steps=10)
        assert state.status.value == "failed"
        assert "ISO timestamp" in (state.error or "")
