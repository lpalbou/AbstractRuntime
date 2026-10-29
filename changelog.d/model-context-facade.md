### Added

- **A local model's loaded context, through the config facade** (`integrations/abstractcore/config_facade.py`).
  `configured_model_context(provider, model)` and `set_model_context(provider, model, tokens)` read and set the
  per-model context AbstractCore loads a model with (AbstractCore ADR 0010 point 8: a set context is a pin, loaded
  as is or refused, never shrunk). The gateway's Resources page "Set context" calls them instead of importing
  AbstractCore. On a Core without `abstractcore.config.model_contexts` (every release up to 2.31.1) they raise
  `AbstractCoreTooOld` naming what is missing.
