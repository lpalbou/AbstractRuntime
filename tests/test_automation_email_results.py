"""Email result is a user delivery choice, independent of model notifications."""

import pytest

from automation_harness import Clock, at, children, drive, make_runtime, make_stores, request
from abstractruntime.automations import apply_automation_command, create_automation, get_automation
from abstractruntime.automations.attention import list_attention
from abstractruntime.automations.models import AutomationError, validate_notify


@pytest.mark.parametrize("store", ["json", "sqlite"])
@pytest.mark.parametrize("model_notify", [None, False, {"title": "Short notice", "body": "Only a summary"}])
def test_email_result_delivers_full_answer_and_revised_recipients(tmp_path, monkeypatch, store, model_notify):
    runtime = make_runtime(*make_stores(store, tmp_path))
    clock = Clock(monkeypatch)
    req = request(input_data={"prompt": "Full result " * 1000, "notify": model_notify})
    req["notify"] = {"channels": ["console", "email"], "recipients": ["self", "FIRST@example.test"]}
    aid, _ = create_automation(runtime, req, now=clock.now)
    drive(runtime, aid)
    first = children(runtime, aid)[0]
    [item] = list_attention(runtime.ledger_store, aid)["items"]
    assert item["email_result"] == first.output["response"]
    assert len(item["email_result"]) > 2000
    assert item["recipients"] == ["self", "first@example.test"]
    # Choosing result recipients never grants model tools permission to email them.
    assert first.vars["_runtime"]["email_allowed_recipients"] == ["self"]
    revised = apply_automation_command(runtime, automation_id=aid, command_id="change-recipient",
        type="automation.revise", now=clock.now,
        payload={"changes": {"notify": {"channels": ["console", "email"], "recipients": ["second@example.test"]}}})
    assert revised["status"] == "applied"
    at(runtime, clock, aid, "2026-01-01T00:02:00+00:00")
    items = list_attention(runtime.ledger_store, aid)["items"]
    assert [i["recipients"] for i in items] == [["self", "first@example.test"], ["second@example.test"]]
    assert get_automation(runtime.run_store, aid)["definition"]["policy"]["email_allowed_recipients"] == ["self"]
    disabled = apply_automation_command(runtime, automation_id=aid, command_id="disable-email-result",
        type="automation.revise", now=clock.now,
        payload={"changes": {"notify": {"channels": ["console"]}}})
    assert disabled["status"] == "applied"
    at(runtime, clock, aid, "2026-01-01T00:04:00+00:00")
    assert len(children(runtime, aid)) == 3
    final_items = list_attention(runtime.ledger_store, aid)["items"]
    # A model-authored notice may remain, but opt-out never carries an email result.
    assert [i["index"] for i in final_items if "email_result" in i] == [1, 2]


@pytest.mark.parametrize("store", ["json", "sqlite"])
def test_console_only_quiet_run_stays_quiet(tmp_path, monkeypatch, store):
    runtime = make_runtime(*make_stores(store, tmp_path))
    clock = Clock(monkeypatch)
    aid, _ = create_automation(runtime, request(input_data={"prompt": "Quiet", "notify": False}), now=clock.now)
    drive(runtime, aid)
    assert list_attention(runtime.ledger_store, aid)["items"] == []


@pytest.mark.parametrize("recipients", [[], "self", ["bad"], ["Name <a@example.test>"], [True]])
def test_result_recipient_validation_reports_notify_field(recipients):
    with pytest.raises(AutomationError) as exc:
        validate_notify({"channels": ["email"], "recipients": recipients})
    assert exc.value.field.startswith("notify.recipients")
