"""`config_facade.voice_input_hint`: the speech-input row's served hint, through the facade.

AbstractGateway serves `GET /voice/defaults` with AbstractCore's one-line hint for the
configured speech-input route (round 16) and must not import AbstractCore itself
(its import-boundary test). The facade is the only door: it passes the route and
host through unchanged, returns None when AbstractCore has no hint, and fails
loudly when AbstractCore lacks the function (a host serving the row without its
hint would otherwise be a silent regression).
"""

from __future__ import annotations

import builtins
from typing import Any, Dict, Optional

import pytest

from abstractruntime.integrations.abstractcore import config_facade as facade


def test_the_facade_returns_cores_answer_for_the_given_route(monkeypatch: pytest.MonkeyPatch) -> None:
    import abstractcore.config.recommendations as core_rec

    seen: Dict[str, Any] = {}

    def fake(route: Any, host: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        seen["route"] = dict(route)
        seen["host"] = host
        return {"code": "apple_gpu_engine", "sentence": "Runs on the GPU with mlx-whisper.", "route": None}

    monkeypatch.setattr(core_rec, "voice_input_hint", fake)
    answer = facade.voice_input_hint({"provider": "faster-whisper", "model": "large-v3"}, {"accelerator": "metal"})
    assert answer == {"code": "apple_gpu_engine", "sentence": "Runs on the GPU with mlx-whisper.", "route": None}
    assert seen == {"route": {"provider": "faster-whisper", "model": "large-v3"}, "host": {"accelerator": "metal"}}


def test_no_hint_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    import abstractcore.config.recommendations as core_rec

    monkeypatch.setattr(core_rec, "voice_input_hint", lambda route, host=None: None)
    assert facade.voice_input_hint({"provider": "mlx-whisper", "model": "large-v3"}) is None


def test_an_abstractcore_without_the_hint_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def refusing_import(name: str, *args: Any, **kwargs: Any):
        if name == "abstractcore.config.recommendations":
            raise ImportError("no recommendations module (AbstractCore < 2.26.0)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refusing_import)
    with pytest.raises(RuntimeError, match="AbstractCore >= 2.26.0"):
        facade.voice_input_hint({"provider": "faster-whisper", "model": "large-v3"})
