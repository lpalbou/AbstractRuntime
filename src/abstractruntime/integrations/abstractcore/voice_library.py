"""Voice library operations behind `AbstractCoreVoiceFacade` (voice_facade.py).

One function per execution side:

- `local_voice_library(op, args)` calls AbstractCore's voice capability in this process
  (`CapabilityRegistry(...).voice`, AbstractCore 2.30.0+ with AbstractVoice 0.18.2+);
- `remote_voice_library(client, op, args)` calls a remote AbstractCore server's `/v1` voice
  routes through a `RemoteAbstractCoreLLMClient`.

Both answer the same shapes and raise `VoiceLibraryError` with an HTTP-like status, so a host
maps a refusal to its own answer without knowing which side ran it. Cloned voices live in
AbstractVoice's store (one per machine user), the same store synthesis reads.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from urllib.parse import quote, urlencode

VOICE_LIBRARY_OPS = (
    "list_languages",
    "list_cloned_voices",
    "get_cloned_voice",
    "update_cloned_voice",
    "delete_cloned_voice",
    "design_voice",
    "clone_voice",
    "compatibility_catalog",
    "deploy_voice_to_library",
)


class VoiceLibraryError(RuntimeError):
    """A voice library refusal: `status_code` (404 unknown voice, 422 refused input, 401 the
    remote server wants its operator token, 501 the capability or engine runtime is missing,
    502 the remote server failed) and `detail` (the sentence to show)."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = int(status_code)
        self.detail = str(detail)


def _require_op(op: str) -> None:
    if op not in VOICE_LIBRARY_OPS:
        raise VoiceLibraryError(422, f"Unknown voice library operation {op!r} (known: {', '.join(VOICE_LIBRARY_OPS)}).")


def _text(value: Any) -> Optional[str]:
    text = str(value).strip() if value is not None else ""
    return text or None


def _check_update(args: Dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    name, reference_text = args.get("name"), args.get("reference_text")
    if name is None and reference_text is None:
        raise VoiceLibraryError(422, "Send `name` and/or `reference_text`.")
    for field, value in (("name", name), ("reference_text", reference_text)):
        if value is not None and not str(value).strip():
            raise VoiceLibraryError(422, f"`{field}` must not be empty.")
    return (str(name) if name is not None else None, str(reference_text) if reference_text is not None else None)


# ---------------------------------------------------------------------------------------
# Local: AbstractCore's voice capability in this process
# ---------------------------------------------------------------------------------------


def _local_error(exc: Exception, *, action: str) -> VoiceLibraryError:
    if isinstance(exc, VoiceLibraryError):
        return exc
    try:
        from abstractcore.capabilities.errors import CapabilityUnavailableError
    except Exception:  # pragma: no cover - AbstractCore is a dependency of this integration
        CapabilityUnavailableError = ()  # type: ignore[assignment]
    if CapabilityUnavailableError and isinstance(exc, CapabilityUnavailableError):
        return VoiceLibraryError(501, str(exc))
    if isinstance(exc, KeyError):
        return VoiceLibraryError(404, f"{action} failed: {exc.args[0] if exc.args else exc}")
    if isinstance(exc, ValueError):
        return VoiceLibraryError(422, f"{action} failed: {exc}")
    message = str(exc)
    if "is not installed" in message and "install it with" in message.lower():
        return VoiceLibraryError(501, f"{action} failed: {message}")
    if "needs its runtime" in message:
        return VoiceLibraryError(501, f"{action} failed: {message}")
    return VoiceLibraryError(500, f"{action} failed: {message}")


def _local_registry(voice_openai_api_key: Optional[str], voice_stores: Optional[Dict[str, Any]] = None) -> Any:
    from .discovery_queries import _runtime_capability_registry

    stores = dict(voice_stores or {})
    return _runtime_capability_registry(
        voice_openai_api_key=voice_openai_api_key,
        voice_store_dir=stores.get("voice_store_dir"),
        voice_library_dirs=stores.get("voice_library_dirs"),
    )


def _local_voice(voice_openai_api_key: Optional[str], voice_stores: Optional[Dict[str, Any]] = None) -> Any:
    return _local_registry(voice_openai_api_key, voice_stores).voice


def _clone_record(voice: Any, out: Any) -> Dict[str, Any]:
    """The stored record of a voice just created (the plugin may answer an id or an object)."""
    if isinstance(out, dict):
        record = dict(out)
        record.setdefault("ok", True)
        return record
    voice_id = str(out if isinstance(out, str) else (getattr(out, "resource_id", "") or getattr(out, "voice_id", "") or "")).strip()
    if not voice_id:
        raise VoiceLibraryError(500, "Voice cloning did not return a voice id.")
    try:
        record = dict(voice.get_cloned_voice(voice_id))
    except Exception:
        record = {}  # the voice was created; only its record could not be read back
    record.setdefault("voice_id", voice_id)
    record.setdefault("ok", True)
    return record


def voice_store_settings(llm_kwargs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The voice-store settings a runtime was built with (`voice_store_dir`, `voice_library_dirs`
    in its AbstractCore client settings): the stores its synthesis uses, so the voice library
    lists and changes the same voices (runtime backlog 0851)."""
    kw = dict(llm_kwargs or {})
    out: Dict[str, Any] = {}
    if isinstance(kw.get("voice_store_dir"), str) and kw["voice_store_dir"].strip():
        out["voice_store_dir"] = kw["voice_store_dir"].strip()
    libs = kw.get("voice_library_dirs")
    if isinstance(libs, str):
        import os

        libs = [x for x in libs.split(os.pathsep) if x.strip()]
    if isinstance(libs, (list, tuple)) and libs:
        out["voice_library_dirs"] = [str(d) for d in libs if str(d or "").strip()]
    return out


def local_voice_library(
    op: str,
    args: Dict[str, Any],
    *,
    registry: Any = None,
    voice_openai_api_key: Optional[str] = None,
    voice_stores: Optional[Dict[str, Any]] = None,
) -> Any:
    """`registry`: the AbstractCore capability registry the runtime's synthesis uses (its
    AbstractCore instance's `capabilities`), so the library acts on the same voice manager and
    settings; without one, a registry is built from the voice settings given."""
    _require_op(op)
    action = op.replace("_", " ").capitalize()
    try:
        if op in ("get_cloned_voice", "update_cloned_voice", "delete_cloned_voice", "deploy_voice_to_library"):
            if not str(args.get("voice_id") or "").strip():
                raise VoiceLibraryError(422, "`voice_id` must not be empty.")
        reg = registry if registry is not None else _local_registry(voice_openai_api_key, voice_stores)
        if op == "deploy_voice_to_library":
            deploy = getattr(reg.get_voice(), "deploy_voice_to_library", None)
            if not callable(deploy):
                raise VoiceLibraryError(501, "Deploying a voice to the library needs AbstractVoice 0.19.0 or later.")
            return dict(deploy(str(args["voice_id"])))
        voice = reg.voice
        if op == "list_languages":
            return {"languages": voice.list_languages(kind=_text(args.get("kind")), provider=_text(args.get("provider")))}
        if op == "list_cloned_voices":
            voices = voice.list_cloned_voices(
                provider=_text(args.get("provider")), model=_text(args.get("model")), language=_text(args.get("language"))
            )
            return {"voices": list(voices), "count": len(voices)}
        if op == "get_cloned_voice":
            return dict(voice.get_cloned_voice(str(args["voice_id"])))
        if op == "update_cloned_voice":
            voice_id = str(args["voice_id"])
            name, reference_text = _check_update(args)
            voice.get_cloned_voice(voice_id)  # unknown id: 404 before any change
            out: Dict[str, Any] = {}
            if name is not None:
                out = voice.rename_cloned_voice(voice_id, name)
            if reference_text is not None:
                out = voice.set_cloned_voice_reference_text(voice_id, reference_text)
            return dict(out)
        if op == "delete_cloned_voice":
            return dict(voice.delete_cloned_voice(str(args["voice_id"])))
        if op == "design_voice":
            description = str(args.get("description") or "")
            if not description.strip():
                raise VoiceLibraryError(422, "`description` must not be empty.")
            return dict(
                voice.design_voice(
                    description,
                    name=_text(args.get("name")),
                    language=_text(args.get("language")),
                    text=_text(args.get("text")),
                )
            )
        if op == "clone_voice":
            audio = args.get("audio")
            if not isinstance(audio, (bytes, bytearray)) or not audio:
                raise VoiceLibraryError(422, "`audio` must be the reference recording's bytes.")
            provider = _text(args.get("provider"))
            language = _text(args.get("language"))
            out = voice.clone(
                bytes(audio),
                name=_text(args.get("name")),
                reference_text=_text(args.get("reference_text")),
                provider=provider,
                model=_text(args.get("model")),
                cloning_engine=provider,
                **({"language": language} if language else {}),
            )
            return _clone_record(voice, out)
        # compatibility_catalog
        return dict(voice.compatibility_catalog())
    except Exception as exc:
        raise _local_error(exc, action=action) from exc


# ---------------------------------------------------------------------------------------
# Remote: a remote AbstractCore server's /v1 voice routes
# ---------------------------------------------------------------------------------------


_PASSED_STATUSES = (400, 401, 403, 404, 409, 413, 422, 429, 501, 503)


def _detail_text(detail: Any) -> str:
    if isinstance(detail, list):  # FastAPI validation errors
        parts = [str(d.get("msg") or d) if isinstance(d, dict) else str(d) for d in detail]
        return "; ".join(p for p in parts if p)
    return str(detail) if detail is not None else ""


def _remote_error(exc: Exception, *, action: str) -> VoiceLibraryError:
    if isinstance(exc, VoiceLibraryError):
        return exc
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if not isinstance(status, int):
        status = getattr(exc, "status_code", None)
    detail: Any = None
    if response is not None:
        body: Any = None
        try:
            body = response.json()
        except Exception:
            body = getattr(response, "body", None)
            if body is None:
                detail = getattr(response, "text", None)
        if isinstance(body, dict):
            # FastAPI's `{"detail": ...}`, or AbstractCore server's `{"error": {"message": ...}}`.
            error = body.get("error")
            detail = body.get("detail") or (error.get("message") if isinstance(error, dict) else error)
        elif body is not None:
            detail = body
    text = _detail_text(detail) or str(exc)
    if isinstance(status, int) and status in _PASSED_STATUSES:
        # The server's own answer: a refusal, a missing route, or "try again later" (429/503).
        return VoiceLibraryError(status, f"{action} failed: {text}")
    if isinstance(status, int):
        return VoiceLibraryError(502, f"{action} failed on the AbstractCore server (HTTP {status}): {text}")
    return VoiceLibraryError(502, f"{action} failed: the AbstractCore server could not be reached ({exc})")


def _remote_request(client: Any, method: str, path: str, *, query: Optional[Dict[str, Any]] = None,
                    json: Optional[Dict[str, Any]] = None, provider_api_key: Optional[str] = None) -> Dict[str, Any]:
    from .llm_client import _join_core_v1_url, _unwrap_http_response

    encoded = {k: str(v).strip() for k, v in dict(query or {}).items() if v is not None and str(v).strip()}
    url = _join_core_v1_url(client._server_base_url, path)
    if encoded:
        url = f"{url}?{urlencode(encoded)}"
    headers = client._headers_with_provider_api_key(provider_api_key)
    sender = client._sender
    if method == "GET":
        raw = sender.get(url, headers=headers, timeout=client._timeout_s)
    elif method == "POST":
        raw = sender.post(url, headers=headers, json=dict(json or {}), timeout=client._timeout_s)
    else:
        request = getattr(sender, "request", None)
        if not callable(request):
            raise VoiceLibraryError(501, f"This request sender cannot send {method}; use HttpxRequestSender or add request().")
        raw = request(method, url, headers=headers, json=json, timeout=client._timeout_s)
    body, _headers = _unwrap_http_response(raw)
    return body


def remote_voice_library(client: Any, op: str, args: Dict[str, Any], *, provider_api_key: Optional[str] = None) -> Any:
    _require_op(op)
    action = op.replace("_", " ").capitalize()
    try:
        if op == "list_languages":
            body = _remote_request(client, "GET", "/audio/languages", query={"kind": args.get("kind"), "provider": args.get("provider")},
                                   provider_api_key=provider_api_key)
            return {"languages": body.get("languages") or {}}
        if op == "list_cloned_voices":
            body = _remote_request(client, "GET", "/voice/clones",
                                   query={k: args.get(k) for k in ("provider", "model", "language")},
                                   provider_api_key=provider_api_key)
            voices = list(body.get("voices") or [])
            return {"voices": voices, "count": len(voices)}
        if op == "deploy_voice_to_library":
            raise VoiceLibraryError(
                501,
                "Deploying a voice to a library is not available on a remote AbstractCore server: its voices live in "
                "the server's own store.",
            )
        if op in ("get_cloned_voice", "update_cloned_voice", "delete_cloned_voice") and not str(args.get("voice_id") or "").strip():
            raise VoiceLibraryError(422, "`voice_id` must not be empty.")
        voice_path = f"/voice/clones/{quote(str(args.get('voice_id') or ''), safe='')}"
        if op == "get_cloned_voice":
            return _remote_request(client, "GET", voice_path, provider_api_key=provider_api_key)
        if op == "update_cloned_voice":
            name, reference_text = _check_update(args)
            payload = {k: v for k, v in (("name", name), ("reference_text", reference_text)) if v is not None}
            return _remote_request(client, "PATCH", voice_path, json=payload, provider_api_key=provider_api_key)
        if op == "delete_cloned_voice":
            return _remote_request(client, "DELETE", voice_path, provider_api_key=provider_api_key)
        if op == "design_voice":
            description = str(args.get("description") or "")
            if not description.strip():
                raise VoiceLibraryError(422, "`description` must not be empty.")
            payload = {"description": description, **{k: _text(args.get(k)) for k in ("name", "language", "text") if _text(args.get(k))}}
            return _remote_request(client, "POST", "/voice/design", json=payload, provider_api_key=provider_api_key)
        if op == "clone_voice":
            audio = args.get("audio")
            if not isinstance(audio, (bytes, bytearray)) or not audio:
                raise VoiceLibraryError(422, "`audio` must be the reference recording's bytes.")
            from .llm_client import _join_core_v1_url

            data = {k: _text(args.get(k)) for k in ("name", "reference_text", "provider", "model", "language") if _text(args.get(k))}
            if not callable(getattr(client._sender, "post_multipart", None)):
                raise VoiceLibraryError(501, "This request sender cannot send multipart uploads; use HttpxRequestSender or add post_multipart().")
            filename = _text(args.get("filename")) or "reference.wav"
            content_type = _text(args.get("content_type")) or "audio/wav"
            body, _headers = client._post_multipart_files(
                _join_core_v1_url(client._server_base_url, "/voice/clone"),
                headers=client._headers_with_provider_api_key(provider_api_key),
                data=data,
                files={"file": (filename, bytes(audio), content_type)},
                timeout=client._timeout_s,
            )
            # One shape on both sides: the stored record, flat, with its voice_id (the server
            # answers `{ok, id, voice_id, voice: {record}}`).
            record = dict(body.get("voice") or {}) if isinstance(body.get("voice"), dict) else {}
            out = {**record, **{k: v for k, v in body.items() if k not in ("voice", "id")}}
            out.setdefault("voice_id", body.get("voice_id") or body.get("id") or record.get("voice_id"))
            out.setdefault("ok", True)
            return out
        # compatibility_catalog: the full voice catalog carries it (Core has no route of its own).
        body = _remote_request(client, "GET", "/audio/voices", provider_api_key=provider_api_key)
        catalog = body.get("compatibility_catalog")
        if not isinstance(catalog, dict) or not catalog:
            raise VoiceLibraryError(501, "The AbstractCore server's voice catalog carries no compatibility catalog (AbstractCore 2.30.0+ with AbstractVoice 0.18.2+).")
        return catalog
    except Exception as exc:
        raise _remote_error(exc, action=action) from exc


__all__ = ["VOICE_LIBRARY_OPS", "VoiceLibraryError", "local_voice_library", "remote_voice_library"]
