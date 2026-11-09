### Fixed

- **An authored summary is no longer stamped mechanical** (`identity/chat.py`, `identity/visit_workflow.py`). The
  visit episode formed with `digest_method: mechanical-v2` before its authored title and summary were merged, and
  the chat lane's `_author_records` kept the formation's mechanical label after the summaries capability wrote the
  summary; memory therefore read a model-authored summary as mechanical and queued a `digest.rewrite` for the sleep
  (tokens spent rewriting authored words). When the summary is authored the record now says
  `digest_method: authored:summaries` (`chat.AUTHORED_SUMMARIES_METHOD`, one helper `authored_digest_attributes`
  for both lanes, the visit close summary included); a failed authoring keeps `mechanical-v2` (its provisional
  digest is the mechanical one) and a title-only authoring leaves the digest's label alone.
