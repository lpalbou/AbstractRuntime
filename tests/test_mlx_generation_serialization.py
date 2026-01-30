from __future__ import annotations

import threading
import time
from types import SimpleNamespace


def test_local_mlx_generation_is_serialized(monkeypatch) -> None:
    """Prevent MLX/Metal process crashes from concurrent generation threads."""
    active = 0
    max_active = 0
    active_lock = threading.Lock()

    class DummyLLM:
        def generate(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal active, max_active
            with active_lock:
                active += 1
                max_active = max(max_active, active)
            # Yield the GIL so concurrent threads can overlap if not locked.
            time.sleep(0.05)
            with active_lock:
                active -= 1
            return SimpleNamespace(
                content="ok",
                raw_response=None,
                tool_calls=None,
                usage=None,
                model="dummy",
                finish_reason="stop",
                metadata={},
                gen_time=None,
            )

    def _create_llm(provider: str, *, model: str, **kwargs):  # type: ignore[no-untyped-def]
        assert provider == "mlx"
        assert model == "mlx-community/gpt-oss-20b-MXFP4-Q8"
        return DummyLLM()

    # LocalAbstractCoreLLMClient imports `create_llm` from `abstractcore` (fallback: abstractcore.core.factory)
    import abstractcore
    from abstractcore.core import factory as ac_factory

    monkeypatch.setattr(abstractcore, "create_llm", _create_llm, raising=True)
    monkeypatch.setattr(ac_factory, "create_llm", _create_llm, raising=True)

    from abstractruntime.integrations.abstractcore.llm_client import LocalAbstractCoreLLMClient

    client = LocalAbstractCoreLLMClient(provider="mlx", model="mlx-community/gpt-oss-20b-MXFP4-Q8")

    barrier = threading.Barrier(2)

    def _call():  # type: ignore[no-untyped-def]
        barrier.wait()
        client.generate(prompt="hi", messages=None, system_prompt=None, tools=None, media=None, params={"stream": False})

    t1 = threading.Thread(target=_call)
    t2 = threading.Thread(target=_call)
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert max_active == 1, "Expected MLX generation to be serialized (no concurrent .generate() entry)"

