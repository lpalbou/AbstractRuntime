### Added

- Remote transcription carries the spoken-language facts: the Core server's `language` (what the engine was told; `null` = detected) and `detected_language` ride the child result's `metadata` (and `text.metadata`), `null` when an older server omits them.
