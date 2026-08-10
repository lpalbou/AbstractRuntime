"""read_idle_timeout_s threading (0152 face 2; core c5051 shipped the base
param, runtime confirmed the name at c5041 and committed same-day threading).

The factory seeds the NO-PROGRESS bound beside the absolute total: 1800s
default for orchestrated lanes (operator ruling 2026-08-02 — a slow local
model may legitimately pause between chunks, while the 7200s total protects
the whole generation); callers override or disable (None = core's
byte-identical pre-fix behavior); entity lanes thread their own tighter 60s.

STREAM-ONLY: abstractcore applies this bound only when streaming=True
(providers/_http.py). On a non-streaming request the body arrives only after
generation finishes, so a read bound there is a hard generation cap, not an
idle gap — that is what cost a coding session ~70% of its compute.
"""

from abstractruntime.integrations.abstractcore.constants import (
    DEFAULT_LLM_READ_IDLE_TIMEOUT_S,
)


def test_factory_seeds_read_idle_beside_total(monkeypatch):
    captured = {}

    class _Stop(Exception):
        pass

    class _FakeLocal:
        def __init__(self, **kwargs):
            captured.update(kwargs.get("llm_kwargs") or {})
            raise _Stop()  # kwargs captured; the rest of the factory is not under test

    import abstractruntime.integrations.abstractcore.factory as fac

    monkeypatch.setattr(fac, "MultiLocalAbstractCoreLLMClient", _FakeLocal)
    try:
        fac.create_local_runtime(provider="lmstudio", model="test-model")
    except Exception:
        pass
    assert captured.get("timeout") is not None
    assert captured.get("read_idle_timeout_s") == DEFAULT_LLM_READ_IDLE_TIMEOUT_S


def test_caller_override_and_disable_respected(monkeypatch):
    captured = {}

    class _Stop(Exception):
        pass

    class _FakeLocal:
        def __init__(self, **kwargs):
            captured.update(kwargs.get("llm_kwargs") or {})
            raise _Stop()  # kwargs captured; the rest of the factory is not under test

    import abstractruntime.integrations.abstractcore.factory as fac

    monkeypatch.setattr(fac, "MultiLocalAbstractCoreLLMClient", _FakeLocal)
    try:
        fac.create_local_runtime(
            provider="lmstudio", model="test-model",
            llm_kwargs={"read_idle_timeout_s": None},
        )
    except Exception:
        pass
    # None rides through untouched: core's None = pre-fix behavior.
    assert captured.get("read_idle_timeout_s") is None


def test_default_value_is_the_committed_number():
    # #[WARNING:TIMEOUT] Operator ruling 2026-08-02: 300 -> 1800. Local models
    # can be legitimately slow BETWEEN chunks, and the old 300 was doubling as
    # a hard generation cap because abstractcore applied it to non-streaming
    # requests too (fixed: providers/_http.py gates it behind streaming=True).
    # This assertion is the guard against a silent re-lowering.
    assert DEFAULT_LLM_READ_IDLE_TIMEOUT_S == 1800.0


def test_entity_lanes_thread_the_tighter_number():
    """The patience-window sites carry read_idle=60 beside timeout=120."""
    from pathlib import Path

    import abstractruntime.identity.chat as chat_mod
    import abstractruntime.identity.life as life_mod

    for mod in (chat_mod, life_mod):
        src = Path(mod.__file__).read_text(encoding="utf-8")
        assert '"read_idle_timeout_s": 60' in src, mod.__name__
        # The skew ladder pops it for older cores (never a crash).
        assert 'pop("read_idle_timeout_s"' in src, mod.__name__
