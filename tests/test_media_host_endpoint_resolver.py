"""Media-only clients retain the host's private endpoint resolver.

Real pooled client, capability host, route-default merge and artifact storage;
only the vision plugin transport is replaced with an owner-resolver consumer.
"""
import base64

import pytest

from abstractruntime.integrations.abstractcore.llm_client import MultiLocalAbstractCoreLLMClient
from abstractruntime.storage.artifacts import InMemoryArtifactStore

PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l9sAAAAASUVORK5CYII=')


@pytest.fixture
def media_client(tmp_path, monkeypatch):
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

    class Vision:
        backend_id = 'test-owner-resolver'

        def __init__(self, owner):
            self.owner = owner

        def t2i(self, prompt, **kwargs):
            # Same host/plugin contract as AbstractVision endpoint profiles:
            # this identity is absent from local Core config by design.
            resolved = self.owner.resolve_provider_endpoint_profile(kwargs['provider'])
            assert resolved['provider'] == 'openai-compatible'
            calls.append((kwargs['model'], resolved['base_url'], resolved['api_key']))
            return PNG

        def i2i(self, prompt, image, **kwargs):
            return self.t2i(prompt, **kwargs)

    def plugins(registry):
        registry.register_vision_backend(backend_id=Vision.backend_id, factory=Vision, priority=1000)

    monkeypatch.setattr(CapabilityRegistry, '_ensure_plugins_loaded', plugins)
    store = InMemoryArtifactStore()
    client = MultiLocalAbstractCoreLLMClient(provider='', model='', artifact_store=store,
        core_config_file=str(tmp_path / 'core.json'), capability_defaults={
            'output.image': {'provider': 'endpoint:private', 'model': 'image-test'}})
    return client, store, calls


def resolver(key):
    return lambda name: {'provider': 'openai-compatible', 'base_url': 'https://private.example/v1', 'api_key': key}


def generate(client):
    return client.generate(prompt='Draw a dot', params={
        'output': {'modality': 'image', 'task': 'image_generation'},
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
