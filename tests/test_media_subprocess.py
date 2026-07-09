from __future__ import annotations

import io
import json
import sys
import types
from pathlib import Path

from abstractruntime.integrations.abstractcore import media_subprocess


def _install_fake_abstractcore(monkeypatch, *, vision) -> None:
    class _FakeResponse:
        def __init__(self, metadata=None):
            self.outputs = {}
            self.resources = {}
            self.warnings = []
            self.errors = []
            self.metadata = metadata or {}

        def add_output(self, modality, item):
            self.outputs.setdefault(modality, []).append(item)

    class _FakeRegistry:
        def __init__(self, _owner):
            self.vision = vision

    abstractcore_pkg = types.ModuleType("abstractcore")
    core_pkg = types.ModuleType("abstractcore.core")
    capabilities_pkg = types.ModuleType("abstractcore.capabilities")
    multimodal_module = types.ModuleType("abstractcore.core.multimodal_generation")
    registry_module = types.ModuleType("abstractcore.capabilities.registry")
    multimodal_module.MultimodalGenerateResponse = _FakeResponse
    registry_module.CapabilityRegistry = _FakeRegistry

    monkeypatch.setitem(sys.modules, "abstractcore", abstractcore_pkg)
    monkeypatch.setitem(sys.modules, "abstractcore.core", core_pkg)
    monkeypatch.setitem(sys.modules, "abstractcore.capabilities", capabilities_pkg)
    monkeypatch.setitem(sys.modules, "abstractcore.core.multimodal_generation", multimodal_module)
    monkeypatch.setitem(sys.modules, "abstractcore.capabilities.registry", registry_module)


def test_media_subprocess_main_dispatches_image_edit_specs(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(b"source-png")

    seen = {}

    class _FakeVision:
        backend_id = "fake-vision"

        def i2i(self, prompt, image, mask=None, **kwargs):
            seen["prompt"] = prompt
            seen["image"] = image
            seen["mask"] = mask
            seen["kwargs"] = kwargs
            return {"data_b64": "cG5nLWVkaXQ=", "content_type": "image/png"}

    _install_fake_abstractcore(monkeypatch, vision=_FakeVision())

    payload = {
        "provider": "mlx",
        "model": "chat-model",
        "llm_kwargs": {},
        "prompt": "edit this image",
        "media": [{"file_path": str(source), "type": "image", "role": "source"}],
        "specs": [
            {
                "modality": "image",
                "task": "image_edit",
                "provider": "mlx-gen",
                "model": "AbstractFramework/qwen-image-edit-2511-8bit",
                "format": "png",
                "steps": 4,
                "lora_adapters": [{"source": "demo/adapter.safetensors", "scale": 0.8}],
            }
        ],
    }

    stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
    monkeypatch.setattr(sys, "stdout", stdout)

    rc = media_subprocess.main()

    assert rc == 0
    result = json.loads(stdout.getvalue())
    assert result["ok"] is True
    outputs = result["response"]["outputs"]["image"]
    assert len(outputs) == 1
    assert outputs[0]["task"] == "image_edit"
    assert outputs[0]["provider"] == "mlx-gen"
    assert outputs[0]["model"] == "AbstractFramework/qwen-image-edit-2511-8bit"
    assert seen["prompt"] == "edit this image"
    assert seen["image"] == str(source)
    assert seen["mask"] is None
    assert seen["kwargs"]["steps"] == 4
    assert seen["kwargs"]["lora_adapters"] == [{"source": "demo/adapter.safetensors", "scale": 0.8}]


def test_media_subprocess_main_dispatches_image_upscale_specs(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(b"source-png")

    seen = {}

    class _FakeVision:
        backend_id = "fake-vision"

        def upscale_image(self, image, **kwargs):
            seen["image"] = image
            seen["kwargs"] = kwargs
            return {"data_b64": "cG5nLXVwc2NhbGU=", "content_type": "image/png"}

    _install_fake_abstractcore(monkeypatch, vision=_FakeVision())

    payload = {
        "provider": "mlx",
        "model": "chat-model",
        "llm_kwargs": {},
        "prompt": "",
        "media": [{"file_path": str(source), "type": "image", "role": "source"}],
        "specs": [
            {
                "modality": "image",
                "task": "image_upscale",
                "provider": "mlx-gen",
                "model": "AbstractFramework/seedvr2-3b-8bit",
                "format": "png",
                "resolution": "2x",
                "softness": 0.25,
            }
        ],
    }

    stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
    monkeypatch.setattr(sys, "stdout", stdout)

    rc = media_subprocess.main()

    assert rc == 0
    result = json.loads(stdout.getvalue())
    assert result["ok"] is True
    outputs = result["response"]["outputs"]["image"]
    assert len(outputs) == 1
    assert outputs[0]["task"] == "image_upscale"
    assert outputs[0]["provider"] == "mlx-gen"
    assert outputs[0]["model"] == "AbstractFramework/seedvr2-3b-8bit"
    assert seen["image"] == str(source)
    assert seen["kwargs"]["resolution"] == "2x"
    assert seen["kwargs"]["softness"] == 0.25
