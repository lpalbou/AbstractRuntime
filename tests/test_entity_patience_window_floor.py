"""The entity patience window has an 1800s floor (operator ruling 2026-11-07).

Incident: the entity's own-time loop stopped itself after three turns in a
row hit "timed out after 120.0s (configured timeout)" on a slow but healthy
remote model. Ruling: "if that was a timeout, minimum timeout should be
1800s". Every entity lane (visit/chat, own time, the sleep narrator) reads
its LLM timeout and retry budget from `_entity_patience_window()`.
"""

import pytest

import abstractruntime.identity.chat as chat_mod
import abstractruntime.identity.life as life_mod

TIMEOUT_ENV = "ABSTRACTRUNTIME_ENTITY_LLM_TIMEOUT_S"
BUDGET_ENV = "ABSTRACTRUNTIME_ENTITY_RETRY_BUDGET_S"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(TIMEOUT_ENV, raising=False)
    monkeypatch.delenv(BUDGET_ENV, raising=False)


def test_default_is_1800_per_attempt_and_1800_budget():
    assert chat_mod._entity_patience_window() == (1800.0, 1800.0)


def test_env_below_the_floor_is_raised_to_1800(monkeypatch, capsys):
    monkeypatch.setattr(chat_mod, "_PATIENCE_FLOOR_SAID", False)
    monkeypatch.setenv(TIMEOUT_ENV, "120")
    monkeypatch.setenv(BUDGET_ENV, "180")
    assert chat_mod._entity_patience_window() == (1800.0, 1800.0)
    # The raise is said, once, with the number the operator set.
    err = capsys.readouterr().err
    assert "ABSTRACTRUNTIME_ENTITY_LLM_TIMEOUT_S=120" in err and "1800" in err
    chat_mod._entity_patience_window()
    assert capsys.readouterr().err == ""


def test_zero_still_means_no_client_timeout(monkeypatch):
    monkeypatch.setenv(TIMEOUT_ENV, "0")
    timeout, budget = chat_mod._entity_patience_window()
    assert timeout is None
    assert budget == 1800.0


def test_budget_is_never_shorter_than_one_attempt(monkeypatch):
    monkeypatch.setenv(TIMEOUT_ENV, "3600")
    monkeypatch.setenv(BUDGET_ENV, "600")
    assert chat_mod._entity_patience_window() == (3600.0, 3600.0)
    monkeypatch.setenv(BUDGET_ENV, "7200")
    assert chat_mod._entity_patience_window() == (3600.0, 7200.0)


def test_budget_zero_means_no_budget(monkeypatch):
    monkeypatch.setenv(BUDGET_ENV, "0")
    assert chat_mod._entity_patience_window() == (1800.0, None)


def test_unparseable_values_fall_back_to_the_floor(monkeypatch):
    monkeypatch.setenv(TIMEOUT_ENV, "soon")
    monkeypatch.setenv(BUDGET_ENV, "nan")
    assert chat_mod._entity_patience_window() == (1800.0, 1800.0)


def test_every_entity_lane_reads_the_same_window():
    # Visit/chat (chat.py) and own time + the sleep narrator (life.py) share
    # ONE definition, so the floor cannot drift between lanes.
    assert life_mod._entity_patience_window is chat_mod._entity_patience_window
    assert chat_mod.ENTITY_LLM_TIMEOUT_FLOOR_S == 1800.0
    assert chat_mod.ENTITY_LLM_READ_IDLE_TIMEOUT_S >= 1800.0


def test_a_negative_value_is_a_typo_not_a_switch_off(monkeypatch):
    from abstractruntime.identity.chat import ENTITY_LLM_TIMEOUT_FLOOR_S, _entity_patience_window

    monkeypatch.setenv("ABSTRACTRUNTIME_ENTITY_LLM_TIMEOUT_S", "-1800")
    monkeypatch.setenv("ABSTRACTRUNTIME_ENTITY_RETRY_BUDGET_S", "-5")
    assert _entity_patience_window() == (ENTITY_LLM_TIMEOUT_FLOOR_S, ENTITY_LLM_TIMEOUT_FLOOR_S)
