"""Resolution verdicts survive real-home projection and immutable diary replay."""

from types import SimpleNamespace

import pytest

pytest.importorskip("abstractmemory")

from abstractruntime.core.models import Effect, EffectType
from abstractruntime.identity.diary import build_diary_effect_handlers, derive_entry_id
from abstractruntime.identity.entity_runtime import open_entity_runtime
from abstractruntime.identity.lifecycle_fence import LifecycleFence
from test_visit_question_resolved import _projection
from test_visit_workflow import _make_home

pytestmark = pytest.mark.basic


@pytest.fixture
def home(tmp_path):
    path = _make_home(tmp_path, slug="resolution-projection")
    ert = open_entity_runtime(path, fence=LifecycleFence.admit(path))
    handlers = build_diary_effect_handlers(
        entity_id=ert.entity_id, diary_store=ert.home.diary,
        memory_system=ert.home.ms, home_dir=path,
    )
    run = SimpleNamespace(run_id="projection-fixture", session_id="projection-session", actor_id="gateway")

    def write(_expect_projection=True, **payload):
        result = handlers[EffectType.DIARY_WRITE](run, Effect(type=EffectType.DIARY_WRITE, payload=payload), None)
        assert result.status == "completed", result.error
        if _expect_projection:
            assert result.result.get("projected_record_id"), result.result
        return result.result

    try:
        yield ert, write, run
    finally:
        ert.close()


@pytest.mark.parametrize("private_question", [False, True])
@pytest.mark.parametrize("private_answer", [False, True])
def test_projection_keeps_resolution_keys_without_private_titles(home, private_question, private_answer):
    ert, write, _ = home
    question = write(text="The private question words.", gist="What makes a trail useful?", turn_id="q",
                     kind="question", visibility="private" if private_question else "self")
    question_row = _projection(ert, question["entry_id"])
    answer = write(text="The private answer words.", gist="A useful trail", turn_id="a", kind="note",
                   resolves=question["entry_id"], visibility="private" if private_answer else "self")
    attributes = _projection(ert, answer["entry_id"]).attributes
    assert attributes["resolves"] == question["entry_id"]
    assert attributes["resolves_status"] == "resolved_open_question"
    assert attributes["resolved_record_id"] == question_row.subject
    if private_question or private_answer:
        assert "resolved_title" not in attributes
    else:
        assert attributes["resolved_title"] == "What makes a trail useful?"
    if private_answer:
        assert "private answer words" not in str(attributes)


def test_refused_resolution_projects_the_verdict_without_a_confirmed_target(home):
    ert, write, _ = home
    answer = write(text="A claim without a target.", turn_id="missing", resolves="diary_missing")
    attributes = _projection(ert, answer["entry_id"]).attributes
    assert attributes["resolves_status"] == "target_not_found"
    assert "resolved_record_id" not in attributes and "resolved_title" not in attributes


def test_replay_keeps_original_resolution_and_visibility_after_later_entries(home):
    ert, write, _ = home
    first = write(text="First question", gist="First question", turn_id="q1", kind="question")
    second = write(text="Second question", gist="Second question", turn_id="q2", kind="question")
    payload = dict(text="The original answer.", turn_id="a", kind="note", resolves=first["entry_id"])
    original = write(**payload)
    duplicate_payload = dict(text="A later duplicate claim.", turn_id="later", resolves=first["entry_id"])
    duplicate = write(**duplicate_payload)
    assert duplicate["resolves_status"] == "target_already_resolved"
    assert write(**duplicate_payload)["resolves_status"] == "target_already_resolved"
    replay = write(**{**payload, "resolves": second["entry_id"], "visibility": "private"})
    assert replay["replayed"] is True
    for key in ("resolves", "resolves_status", "resolved_record_id", "resolved_title", "visibility"):
        assert replay.get(key) == original.get(key), key
    assert len([entry for entry in ert.home.diary.list_entries() if entry["entry_id"] == original["entry_id"]]) == 1


def test_replay_does_not_validate_a_target_created_after_the_original_act(home):
    ert, write, run = home
    future_id = derive_entry_id(run_id=run.run_id, turn_id="future", text="A future question")
    payload = dict(text="An early claim.", turn_id="early", resolves=future_id)
    original = write(**payload)
    assert original["resolves_status"] == "target_not_found"
    write(text="A future question", gist="Later question", turn_id="future", kind="question")
    replay = write(**payload)
    assert replay["resolves_status"] == "target_not_found"
    assert "resolved_record_id" not in replay


def test_replay_of_a_successful_resolution_does_not_resolve_itself_again(home):
    _, write, _ = home
    question = write(text="Question", gist="A question", turn_id="q", kind="question")
    payload = dict(text="Answer", turn_id="a", resolves=question["entry_id"])
    original = write(**payload)
    replay = write(**payload)
    assert replay["replayed"] is True
    assert original["resolves_status"] == replay["resolves_status"] == "resolved_open_question"
    assert original["resolved_record_id"] == replay["resolved_record_id"]


def test_target_metadata_failure_keeps_chain_and_defers_projection_until_replay(home, monkeypatch):
    from abstractruntime.identity import diary
    ert, write, _ = home
    question = write(text="Question", gist="The question", kind="question", turn_id="q")
    target = _projection(ert, question["entry_id"])
    original = diary._projection_row
    calls = []
    def fail_once(*args):
        calls.append(args)
        if len(calls) == 1:
            raise OSError("temporary target read failure")
        return original(*args)
    monkeypatch.setattr(diary, "_projection_row", fail_once)
    payload = dict(text="Answer", turn_id="a", resolves=question["entry_id"])
    first = write(_expect_projection=False, **payload)
    assert first["resolves_status"] == "resolved_open_question"
    assert not first.get("projected_record_id")
    assert any("#FALLBACK" in warning for warning in first["warnings"])
    assert ert.home.diary.get_record(first["entry_id"])
    replay = write(**payload)
    assert replay["replayed"] is True
    assert replay["resolved_record_id"] == target.subject
    assert _projection(ert, replay["entry_id"]).attributes["resolved_title"] == "The question"


def test_projection_failure_keeps_chain_and_replay_forms_one_projection(home, monkeypatch):
    ert, write, _ = home
    original = ert.home.ms.remember_many
    calls = []
    def fail_once(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise OSError("temporary projection failure")
        return original(*args, **kwargs)
    monkeypatch.setattr(ert.home.ms, "remember_many", fail_once)
    payload = dict(text="Remember this", gist="A durable act", turn_id="repair")
    first = write(_expect_projection=False, **payload)
    assert not first.get("projected_record_id")
    assert any("#FALLBACK" in warning for warning in first["warnings"])
    assert ert.home.diary.get_record(first["entry_id"])
    second = write(**payload)
    third = write(**payload)
    assert second["replayed"] and third["replayed"]
    assert second["projected_record_id"] == third["projected_record_id"]
    assert len([e for e in ert.home.diary.list_entries() if e["entry_id"] == first["entry_id"]]) == 1
