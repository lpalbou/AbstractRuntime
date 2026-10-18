"""Explicit integration probe: run with pytest tests/media_endpoint_profile_probe.py.

Requires optional AbstractVision >=0.3.34. Not in default collection because
Runtime does not depend on the vision plugin. No skips: this command must pass.
Real Core/Vision profile resolution, t2i/i2i serialization and artifact bytes;
only final HTTP transport is replaced. No network or provider subprocess allowed.
"""
import base64

import pytest

from abstractruntime.integrations.abstractcore.llm_client import MultiLocalAbstractCoreLLMClient
from abstractruntime.storage.artifacts import InMemoryArtifactStore

PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l9sAAAAASUVORK5CYII=')


@pytest.fixture(params=['image_generation', 'image_edit'])
def media_client(tmp_path, monkeypatch, request):
    import socket
    import subprocess
    def forbidden(*args, **kwargs):
        raise AssertionError('Media resolver test must not access network or spawn providers')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    monkeypatch.setenv('ABSTRACTRUNTIME_LOCAL_IMAGE_SUBPROCESS', '1')
    from abstractcore.capabilities.registry import CapabilityRegistry
    from abstractcore.config import manager
    config = manager.ConfigurationManager(config_file=str(tmp_path / 'core.json'), apply_env=False)
    monkeypatch.setattr(manager, '_config_manager', config)
    calls = []

    from abstractvision.integrations.abstractcore_plugin import _AbstractVisionCapability
    from abstractvision.backends.openai_compatible import OpenAICompatibleVisionBackend

    def post_json(backend, *, path, payload):
        assert path == '/images/generations'
        calls.append((payload['model'], backend._cfg.base_url, backend._cfg.api_key))
        return {'data': [{'b64_json': base64.b64encode(PNG).decode()}]}

    monkeypatch.setattr(OpenAICompatibleVisionBackend, '_post_json', post_json)

    def post_multipart(backend, *, path, fields, files):
        assert path == '/images/edits'
        assert any(item[1] == PNG for item in files.values())
        calls.append((fields['model'], backend._cfg.base_url, backend._cfg.api_key))
        return {'data': [{'b64_json': base64.b64encode(PNG).decode()}]}

    monkeypatch.setattr(OpenAICompatibleVisionBackend, '_post_multipart', post_multipart)

    def plugins(registry):
        registry.register_vision_backend(backend_id='abstractvision:openai',
            factory=lambda owner: _AbstractVisionCapability(owner), priority=1000)

    monkeypatch.setattr(CapabilityRegistry, '_ensure_plugins_loaded', plugins)
    store = InMemoryArtifactStore()
    client = MultiLocalAbstractCoreLLMClient(provider='', model='', artifact_store=store,
        core_config_file=str(tmp_path / 'core.json'), capability_defaults={
            'output.image': {'provider': 'endpoint:private', 'model': 'image-test'}})
    client._test_task = request.param
    source = tmp_path / 'source.png'
    source.write_bytes(PNG)
    client._test_source = str(source)
    return client, store, calls


def resolver(key):
    return lambda name: {'provider': 'openai-compatible', 'base_url': 'https://private.example/v1', 'api_key': key}


def generate(client):
    media = [{'file_path': client._test_source, 'content_type': 'image/png', 'role': 'source'}] if client._test_task == 'image_edit' else None
    return client.generate(prompt='Draw a dot', media=media, params={
        'output': {'modality': 'image', 'task': client._test_task},
        'trace_metadata': {'run_id': 'media-run'}})


@pytest.mark.parametrize('attach_before_creation', [True, False])
def test_media_host_receives_resolver_before_or_after_creation(media_client, attach_before_creation):
    client, store, calls = media_client
    if not attach_before_creation:
        client._media_host_client()
    client.set_provider_endpoint_profile_resolver(resolver('first-key'))
    result = generate(client)
    assert calls == [('image-test', 'https://private.example/v1', 'first-key')]
    item = result['outputs']['image'][0]
    artifact = store.load(item['artifact_ref']['$artifact'])
    assert artifact is not None and artifact.content == PNG
    assert 'first-key' not in str(result)
    client.set_provider_endpoint_profile_resolver(resolver('rotated-key'))
    generate(client)
    assert calls[-1][2] == 'rotated-key'


def test_rebuilt_media_host_keeps_resolver(media_client):
    client, _, calls = media_client
    client.set_provider_endpoint_profile_resolver(resolver('key'))
    generate(client)
    client.set_capability_defaults({'output.image': {'provider': 'endpoint:private', 'model': 'image-new'}})
    generate(client)
    assert [call[0] for call in calls] == ['image-test', 'image-new']
