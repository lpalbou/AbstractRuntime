"""A healthy own-time day must never read as a corpse while it is mid-tick
(review of 0.18.4: a tick can make ~22 LLM calls of up to 1800 s each)."""
import inspect
import json
import os
from datetime import datetime, timedelta, timezone

from abstractruntime.identity.life import (
    _beat_before_each_call,
    read_loop_status,
    touch_loop_heartbeat,
)


def _write(home, **fields):
    (home / "loop_status").write_text(json.dumps(fields) + "\n", encoding="utf-8")


def _stale_day(home, pid):
    old = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
    _write(home, phase="day", updated_at=old, pid=pid)
    return old


class _LLM:
    def __init__(self):
        self.calls = 0

    def generate(self, messages, system_prompt=None, thinking=None, **kwargs):
        self.calls += 1
        return "ok"


def test_each_llm_call_refreshes_this_process_day(tmp_path):
    _stale_day(tmp_path, os.getpid())
    assert read_loop_status(tmp_path).get("running") is False  # frozen 3 h: read as a corpse
    llm = _beat_before_each_call(_LLM(), tmp_path)
    assert llm.generate(messages=[]) == "ok" and llm.calls == 1
    assert read_loop_status(tmp_path).get("running") is True


def test_heartbeat_keeps_the_signature_chat_session_inspects(tmp_path):
    params = inspect.signature(_beat_before_each_call(_LLM(), tmp_path).generate).parameters
    assert "thinking" in params and "system_prompt" in params


def test_heartbeat_never_revives_a_stopped_or_foreign_status(tmp_path):
    _write(tmp_path, phase="stopped", updated_at="2026-08-01T11:39:22+00:00", pid=os.getpid())
    touch_loop_heartbeat(tmp_path)
    assert json.loads((tmp_path / "loop_status").read_text())["updated_at"] == "2026-08-01T11:39:22+00:00"
    old = _stale_day(tmp_path, os.getpid() + 999_999)
    touch_loop_heartbeat(tmp_path)
    assert json.loads((tmp_path / "loop_status").read_text())["updated_at"] == old


def test_heartbeat_tolerates_a_missing_or_corrupt_file(tmp_path):
    touch_loop_heartbeat(tmp_path)
    (tmp_path / "loop_status").write_text("{not json", encoding="utf-8")
    touch_loop_heartbeat(tmp_path)
