"""Media specs routed to an `endpoint:<id>` profile run in process.

The one-shot media subprocess isolates LOCAL pipelines; it has no host
endpoint-profile resolver, so a gateway profile (e.g. an OpenAI-compatible
relay serving `gpt-image-2`) could not be resolved there. Regression for the
operator report of 2026-10-18.
"""

from __future__ import annotations

from pathlib import Path

from abstractruntime.integrations.abstractcore.llm_client import (
    _is_subprocess_safe_image_specs,
    _is_subprocess_safe_video_specs,
    _spec_routes_to_endpoint_profile,
)


def test_endpoint_spec_detection() -> None:
    assert _spec_routes_to_endpoint_profile({"provider": "endpoint:relay"}) is True
    assert _spec_routes_to_endpoint_profile({"provider": "Endpoint:Relay"}) is True
    assert _spec_routes_to_endpoint_profile({"provider": "huggingface"}) is False
    assert _spec_routes_to_endpoint_profile({}) is False
    assert _spec_routes_to_endpoint_profile(None) is False


def test_endpoint_image_specs_skip_the_subprocess(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(b"png")
    media = [{"file_path": str(source), "type": "image", "role": "source"}]

    assert _is_subprocess_safe_image_specs([{"modality": "image", "task": "text_to_image", "provider": "huggingface"}], None) is True
    assert _is_subprocess_safe_image_specs(
        [{"modality": "image", "task": "text_to_image", "provider": "endpoint:relay", "model": "gpt-image-2"}], None
    ) is False
    assert _is_subprocess_safe_image_specs([{"modality": "image", "task": "image_edit", "provider": "huggingface"}], media) is True
    assert _is_subprocess_safe_image_specs(
        [{"modality": "image", "task": "image_edit", "provider": "endpoint:relay", "model": "gpt-image-2"}], media
    ) is False


def test_endpoint_video_spec_skips_the_subprocess() -> None:
    assert _is_subprocess_safe_video_specs([{"modality": "video", "task": "text_to_video"}], None) is True
    assert _is_subprocess_safe_video_specs([{"modality": "video", "task": "text_to_video", "provider": "endpoint:relay"}], None) is False
