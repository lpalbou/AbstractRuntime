from __future__ import annotations

from abstractruntime.integrations.abstractcore.effect_handlers import _observability_params
from abstractruntime.storage.artifacts import build_artifact_descriptor_payload


def test_observability_params_redact_browser_session_secrets() -> None:
    observed = _observability_params(
        {
            "bearer_token": "bearer-secret",
            "csrf_token": "csrf-secret",
            "session_token": "session-secret",
            "id_token": "id-secret",
            "refresh_token": "refresh-secret",
            "authorization_header": "Bearer hidden",
            "temperature": 0.2,
            "safe": "kept",
        }
    )

    assert observed["bearer_token"] == "[redacted]"
    assert observed["csrf_token"] == "[redacted]"
    assert observed["session_token"] == "[redacted]"
    assert observed["id_token"] == "[redacted]"
    assert observed["refresh_token"] == "[redacted]"
    assert observed["authorization_header"] == "[redacted]"
    assert observed["temperature"] == 0.2
    assert observed["safe"] == "kept"


def test_artifact_descriptor_redacts_browser_session_secrets() -> None:
    descriptor, metadata = build_artifact_descriptor_payload(
        semantic_kind="text",
        render_kind="text",
        task="text_generation",
        generation={
            "prompt": "hello",
            "bearer_token": "bearer-secret",
            "nested": {
                "csrf_token": "csrf-secret",
                "session_token": "session-secret",
                "authorization_header": "Bearer hidden",
                "safe": "kept",
            },
        },
    )

    assert descriptor["generation"]["bearer_token"] == "[redacted]"
    assert descriptor["generation"]["nested"]["csrf_token"] == "[redacted]"
    assert descriptor["generation"]["nested"]["session_token"] == "[redacted]"
    assert descriptor["generation"]["nested"]["authorization_header"] == "[redacted]"
    assert descriptor["generation"]["nested"]["safe"] == "kept"
    assert descriptor["security"]["redaction"] == "bounded_secret_key_redaction_v1"

    assert metadata["generation"]["bearer_token"] == "[redacted]"
    assert metadata["generation"]["nested"]["csrf_token"] == "[redacted]"
    assert metadata["generation"]["nested"]["session_token"] == "[redacted]"
    assert metadata["generation"]["nested"]["authorization_header"] == "[redacted]"
