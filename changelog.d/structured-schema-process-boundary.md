### Fixed

- Structured generation carries JSON schemas across execution-client boundaries and creates validators where inference runs, preserving nested validation and defaults without serializing Python classes. Remote Core calls forward their structured response format and explicitly refuse Python validator classes that cannot cross HTTP. Local explicit validator classes remain supported. Request serialization failures are not retried, and execution diagnostics include the actual ledger step identity.

### Added

- Hosts can start Core's independently authenticated HTTP listener through a lazy Runtime lifecycle facade; unused server dependencies stay unloaded.
