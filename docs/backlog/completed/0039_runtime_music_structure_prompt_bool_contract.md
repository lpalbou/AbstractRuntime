# Completed: Runtime Music `structure_prompt` Boolean Contract Alignment

## Metadata
- Created: 2026-05-22
- Completed: 2026-05-22
- Status: Completed
- Origin: moved from `docs/backlog/proposed/0039_runtime_music_structure_prompt_bool_contract.md`

## Problem

Gateway `MusicGenerateRequest` exposes `structure_prompt` as `Optional[bool]`.
AbstractFlow mirrors that boolean contract in its Generate Music node.

Runtime's native `generate_music` VisualFlow handler forwarded
`structure_prompt` only through the string-field path. Boolean values authored
by Flow were therefore dropped before the Gateway/Core music call.

## Implementation

- Moved `structure_prompt` from the music string option group to the boolean
  option group in the native `generate_music` VisualFlow handler.
- Preserved explicit `False` values so callers can distinguish "disabled" from
  "not specified".
- Added compiler coverage proving:
  - `structure_prompt=True` is forwarded as a boolean in the pending music
    output selector.
  - `structure_prompt=False` from node input pins is preserved as an explicit
    boolean override.

## Outcome

Runtime now keeps the same boolean contract used by AbstractFlow, Gateway, and
AbstractCore. A Flow-authored Generate Music node can enable or disable backend
structured prompt planning without Runtime stringifying or dropping the control.

## Validation

- `PYTHONPATH=src python -m pytest -q tests/test_visualflow_media_nodes.py`

## Related Flow work

AbstractFlow completed Gateway `0.2.17` native media alignment in
`abstractflow/docs/backlog/completed/0072_gateway_0_2_17_native_media_contract_alignment.md`.
