# 0852 — A capability's refusal of the request is not retried and reaches the host as a refusal

**Status**: planned · **Priority**: P2 · **Created**: 2026-10-27

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
