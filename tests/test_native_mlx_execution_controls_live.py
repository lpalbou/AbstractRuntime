"""Opt-in real Runtime -> Core -> MLX boundary test; never uses a live server.

Run one model/depth case per process with at least 60 seconds between GPU blocks.
Set ABSTRACTRUNTIME_MLX_LIVE_MODEL to an already downloaded local checkpoint and
ABSTRACTRUNTIME_MLX_LIVE_HEAD for a separate 27B head. No model downloads occur.
"""
from __future__ import annotations

import importlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


@pytest.mark.skipif(not os.getenv("ABSTRACTRUNTIME_MLX_LIVE_MODEL"), reason="explicit local MLX checkpoint required")
def test_local_and_remote_runtime_share_native_scheduler_and_controls(monkeypatch, tmp_path):
    model = os.environ["ABSTRACTRUNTIME_MLX_LIVE_MODEL"]
    assert Path(model).is_dir(), "Pass an existing local target; downloads are not allowed"
    depth = int(os.environ.get("ABSTRACTRUNTIME_MLX_LIVE_DEPTH", "2"))
    assert depth >= 0
    for name, value in {
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "ABSTRACTCORE_SERVER_DISABLE_CENTRALIZED_CONFIG": "1",
        "ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED": "1",
        "ABSTRACTCORE_CONFIG_FILE": str(tmp_path / "core.json"),
        "ABSTRACTFRAMEWORK_DATA_REGISTRY": str(tmp_path / "registry.json"),
    }.items():
        monkeypatch.setenv(name, value)

    from abstractcore.core.retry import RetryConfig
    from abstractruntime.integrations.abstractcore.llm_client import (
        LocalAbstractCoreLLMClient, RemoteAbstractCoreLLMClient,
    )
    from fastapi.testclient import TestClient

    spec = {"mode": "native_mtp", "num_draft_tokens": max(depth, 2), "require_acceleration": True}
    head = os.environ.get("ABSTRACTRUNTIME_MLX_LIVE_HEAD")
    if head:
        assert Path(head).is_dir()
        spec["drafter"] = head
    local = LocalAbstractCoreLLMClient(
        provider="mlx", model=model,
        core_config_file=tmp_path / "core.json",
        llm_kwargs={
            "speculation": spec, "mlx_batching": True, "mlx_max_batch_size": 4,
            "mlx_batch_wait_ms": 200, "max_output_tokens": 32,
            "retry_config": RetryConfig(max_attempts=1), "enable_tracing": False,
            "mlx_cache_scope": "runtime-wiring-live",
        },
    )
    llm = local._llm
    server = importlib.import_module("abstractcore.server.app")
    runtime = server._GatewayLoadedRuntime(
        provider="mlx", model=model, base_url=None, explicit_provider_key_hash=None, llm=llm,
    )
    monkeypatch.setattr(server, "_get_loaded_gateway_runtime", lambda **kwargs: runtime)
    monkeypatch.setattr(server, "create_llm", lambda *args, **kwargs: llm)
    sent = []

    class IsolatedASGISender:
        def post(self, url, *, headers, json, timeout):
            # Exercise the real Core server router without sockets or host discovery.
            assert url == "http://runtime-wiring.test/v1/chat/completions"
            sent.append(json)
            transport = TestClient(server.app)
            try:
                response = transport.post("/v1/chat/completions", headers=headers, json=json)
            finally:
                transport.close()
            assert response.status_code == 200, response.text
            return response.json()

    remote = RemoteAbstractCoreLLMClient(
        server_base_url="http://runtime-wiring.test", model="mlx/" + model,
        request_sender=IsolatedASGISender(), timeout_s=180,
        core_config_file=tmp_path / "core.json",
    )
    control = False if depth == 0 else {"mode": "native_mtp", "num_draft_tokens": depth, "require_acceleration": True}
    gate = threading.Barrier(4, timeout=30)

    def call(index):
        is_remote = index >= 2
        client = remote if is_remote else local
        params = {
            "temperature": 0, "thinking": False, "speculation": control,
            "max_tokens" if is_remote else "max_output_tokens": 32,
        }
        if not is_remote:
            params["stream"] = index == 1
        gate.wait()
        result = client.generate(prompt=f"List the integers from 1 to 40, one per line. Request {index}.", params=params)
        metadata = result["metadata"]
        assert metadata["speculation"]["used"] is (depth > 0), result
        if depth:
            assert metadata["speculation"]["num_draft_tokens"] == depth, result
        assert metadata["execution"]["mode"] in ("continuous", "cohort"), result
        assert result.get("content"), result
        usage = result["usage"]
        assert usage.get("output_tokens", usage.get("completion_tokens", 0)) > 0, result
        return {"path": "remote" if is_remote else "local", "stream": index == 1,
                "usage": result["usage"], "execution": metadata["execution"],
                "speculation": metadata["speculation"], "content": result["content"]}

    try:
        assert llm.supports_concurrent_generation() is True
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(call, index) for index in range(4)]
            rows = [future.result(timeout=180) for future in futures]
        assert len(sent) == 2 and all(body["thinking"] is False and body["speculation"] == control for body in sent)
        # Actual scheduler evidence, not merely concurrent Python thread starts.
        for path in ("local", "remote"):
            assert max(row["execution"]["peak_batch_size"] for row in rows if row["path"] == path) >= 2, rows
        # A separate keyed request verifies the full-history shape repair.
        # Different APC managers are deliberately separate scheduler groups;
        # do not mistake that existing tenant/cache boundary for an outer lock.
        keyed = local.generate(prompt="Write the numbers one through five.", params={
            "temperature": 0, "thinking": False, "speculation": control,
            "max_output_tokens": 16, "prompt_cache_key": "runtime-keyed",
            "_prompt_cache_attribution": {"source": "runtime", "session_id": "keyed"},
        })
        assert keyed["metadata"]["speculation"]["used"] is (depth > 0), keyed
        assert keyed.get("content"), keyed
        evidence = {"model": model, "depth": depth, "rows": rows,
                    "keyed": {"content": keyed["content"], "metadata": keyed["metadata"]}}
        (tmp_path / "native-runtime-wiring.json").write_text(json.dumps(evidence, indent=2, default=str))
        print("NATIVE_RUNTIME_WIRING=" + json.dumps(evidence, default=str))
    finally:
        runtime.provider_executor.shutdown(wait=True, cancel_futures=True)
        llm.unload_model(model)
