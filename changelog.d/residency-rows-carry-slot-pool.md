### Fixed

- **The residency listing carries the model's slot pool** (`integrations/abstractcore/llm_client.py`, core ADR 0011, OVH 8080 pass four). A GGUF row now carries `slots`, `slots_max`, `active`, `queued`, `copies`, `slot_bytes`, `shared_weights` and `slot_refusal_reason`, folded from Core's process row. Before, only the gateway's `/host/state` showed them, so the Resources table could not say that requests were waiting, or why another slot is not built. A row whose holders share one weights instance no longer says "each a full copy of the weights".
