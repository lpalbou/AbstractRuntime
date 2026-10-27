# 0852 — A capability's refusal of the request is not retried and reaches the host as a refusal

**Status**: completed (2026-10-27, 0.13.0) · **Priority**: P2 · **Created**: 2026-10-27

## Why
Measured 2026-10-27 through the gateway (`POST /runs/{id}/voice/tts`, Qwen3-TTS, `one_shot: true`
with 2,399 characters): AbstractVoice refuses at once ("one_shot synthesis takes at most 1000
characters in a single generation"), the media effect retries it three times, and the host
answers HTTP 500 "Effect failed after 3 attempts". The refusal is the caller's input, not a
transient failure: retrying wastes three engine calls and the host cannot answer 422.

## Current code reality
- The LLM_CALL media path (`llm_client.py` `_run_multimodal_spec` → AbstractCore
  `providers/base.py` → plugin) raises `ValueError` for refused input; the effect's retry policy
  treats every exception alike.
- The voice library facade already maps `ValueError` to a 422 `VoiceLibraryError`
  (`voice_library.py`), so the two surfaces disagree.

## Scope
- In: classify refused input (`ValueError`, AbstractCore `CapabilityUnavailableError`) as
  non-retryable in the media effects; record it on the child run with a code (`invalid_request`,
  `capability_unavailable`) the host maps to 422 / 501.
- Out: engine limits themselves (AbstractVoice).

## Validation
A one_shot request over the limit makes one engine call and the child run records
`invalid_request`; a transient error is still retried; mutants: drop the classification and the
test sees three calls.

## Report (completed 2026-10-27, 0.13.0)
`effect_handlers._llm_error_is_retryable`: a plain `ValueError`, a `KeyError` and AbstractCore's
`CapabilityUnavailableError` are non-retryable (one engine call). Subclasses of `ValueError` such as
`json.JSONDecodeError` keep the retryable default. Not done here: an error code on the child run for the host to map to
422/501 (the run failure record carries prose only); AbstractGateway backlog 0942 tracks the 422 answer of
`/voice/tts`.

Validation: `tests/test_llm_retry_classification.py::test_a_capability_refusal_is_not_retried` (refusals non-retryable;
a JSON decode error and a connection error still retried).
