"""A healthy own-time day must never read as a corpse while it is mid-tick
(review of 0.18.4: a tick can make ~22 LLM calls of up to 1800 s each)."""
from abstractruntime.identity.lifecycle_fence import LifecycleFence  # noqa: E402
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
    llm = _beat_before_each_call(_LLM(), tmp_path, fence=LifecycleFence.admit(tmp_path))
    assert llm.generate(messages=[]) == "ok" and llm.calls == 1
    assert read_loop_status(tmp_path).get("running") is True


def test_heartbeat_keeps_the_signature_chat_session_inspects(tmp_path):
    params = inspect.signature(_beat_before_each_call(_LLM(), tmp_path, fence=LifecycleFence.admit(tmp_path)).generate).parameters
    assert "thinking" in params and "system_prompt" in params


def test_heartbeat_never_revives_a_stopped_or_foreign_status(tmp_path):
    _write(tmp_path, phase="stopped", updated_at="2026-08-01T11:39:22+00:00", pid=os.getpid())
    touch_loop_heartbeat(tmp_path, fence=LifecycleFence.admit(tmp_path))
    assert json.loads((tmp_path / "loop_status").read_text())["updated_at"] == "2026-08-01T11:39:22+00:00"
    old = _stale_day(tmp_path, os.getpid() + 999_999)
    touch_loop_heartbeat(tmp_path, fence=LifecycleFence.admit(tmp_path))
    assert json.loads((tmp_path / "loop_status").read_text())["updated_at"] == old


def test_heartbeat_tolerates_a_missing_or_corrupt_file(tmp_path):
    touch_loop_heartbeat(tmp_path, fence=LifecycleFence.admit(tmp_path))
    (tmp_path / "loop_status").write_text("{not json", encoding="utf-8")
    touch_loop_heartbeat(tmp_path, fence=LifecycleFence.admit(tmp_path))


def test_heartbeat_also_after_the_call(tmp_path):
    """A long call followed by a long tool round must not outlast the bound."""
    seen = []

    class _Slow:
        def generate(self, messages, system_prompt=None, **kwargs):
            seen.append(json.loads((tmp_path / "loop_status").read_text())["updated_at"])
            _stale_day(tmp_path, os.getpid())  # the call itself takes "hours"
            return "ok"

    _stale_day(tmp_path, os.getpid())
    _beat_before_each_call(_Slow(), tmp_path, fence=LifecycleFence.admit(tmp_path)).generate(messages=[])
    assert read_loop_status(tmp_path).get("running") is True


def test_the_own_time_session_factory_installs_the_heartbeat(tmp_path, monkeypatch):
    """Wiring: the LLM the real own-time factory hands to ChatSession beats."""
    import abstractcore
    from test_prompt_overlay import ENTITY_SLUG, _create_home, _FakeResponse

    from abstractruntime.identity.life import build_session_factory

    home = tmp_path / "entities" / ENTITY_SLUG
    _create_home(home)

    class _LLMStub:
        def generate(self, **kwargs):
            return _FakeResponse("ok")

    monkeypatch.setattr(abstractcore, "create_llm", lambda *a, **k: _LLMStub())
    factory = build_session_factory(
        home, provider="lmstudio", model="test", base_url="http://x",
        embedding_model=None, embedding_base_url="http://x",
        context_window=32768, out=lambda s: None, fence=LifecycleFence.admit(home),
    )
    session = factory()
    try:
        _stale_day(home, os.getpid())
        assert read_loop_status(home).get("running") is False
        session.llm.generate(messages=[])
        assert read_loop_status(home).get("running") is True
    finally:
        session.home.close()


def test_the_own_time_factory_rereads_the_recall_width_each_day(tmp_path, monkeypatch):
    """Operator ruling 2026-11-08: cognition.yaml recall.shelf_size (75)
    beats the loop's spawn-time --shelf-size (50) at the next day-open, and
    a change between days lands without a loop restart."""
    import abstractcore
    from test_prompt_overlay import ENTITY_SLUG, _create_home, _FakeResponse

    from abstractruntime.identity.life import build_session_factory

    home = tmp_path / "entities" / ENTITY_SLUG
    _create_home(home)

    class _LLMStub:
        def generate(self, **kwargs):
            return _FakeResponse("ok")

    monkeypatch.setattr(abstractcore, "create_llm", lambda *a, **k: _LLMStub())
    lines = []
    factory = build_session_factory(
        home, provider="lmstudio", model="test", base_url="http://x",
        embedding_model=None, embedding_base_url="http://x",
        context_window=32768, shelf_size=50, out=lines.append, fence=LifecycleFence.admit(home),
    )
    day1 = factory()
    try:
        assert day1.profile["shelf_size"] == 50, "no profile: the spawn width"
    finally:
        day1.home.close()
    (home / "cognition.yaml").write_text("schema_version: 1\nrecall:\n  shelf_size: 75\n", encoding="utf-8")
    day2 = factory()
    try:
        assert day2.profile["shelf_size"] == 75, "the profile's width, re-read at day-open"
    finally:
        day2.home.close()
    assert any("recall width re-read at day-open: 50 -> 75" in ln for ln in lines), lines
