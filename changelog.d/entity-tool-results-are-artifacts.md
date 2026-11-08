### Added

- **Every tool result during entity activity rests whole as an artifact of the entity's home** (`identity/tools.py`,
  `identity/tool_effects.py`, `identity/chat.py`; root backlog 1145 step 1; operator 2026-11-08: "there should be no
  activity, either by human or by entities, that does not go through the runtime ... to enable retrospective analysis /
  search, including to rehydrate context/memory"). `execute_tool_elections(..., artifact_store=, artifact_context=)`
  stores what each executed tool returned in the home's `FileArtifactStore` through `keep_tool_result_artifact`:
  content-addressed (one blob per content, an artifact per run), no size cut, with metadata `tool`, `arguments` (the
  election body whole and its block args), `turn_id` / `tick_id`, `record_id` when known, `created_at`, `lane`,
  `run_id`, `session_id`, `entity_id`, `content_type`, `bytes`, `shown_is_view`, `failed`, and tags `kind=tool_result`,
  `tool`, `lane`, `entity_id`, `turn_id`. The result text the model reads ends with one line, `[artifact: <id>]`, so the
  verbatim and the memory records can name it; `ToolElection.artifact_id` carries it, the ENTITY_TOOLS_EXECUTE result's
  `results[].artifact_id` and the turn report's `tool_details[].artifact_id` too. Where the shown result is a bounded view,
  the artifact is the whole material: `fetch_url` keeps the whole page past its 6,000-character view, `read_file` the
  whole file (byte for byte, content type from the bytes; an image's bytes too, `tool_result_media` unchanged),
  `execute_command` the whole output past its cap. Book-adjacent tools (diary_read, diary_list, search_memory) are the
  one exception: their content is the entity's own book/graph, already in the home, and the W5 containment rule keeps
  private words out of life scope. A store failure is a loud `#FALLBACK` notice, never a lost turn. Wired on the
  ENTITY_TOOLS_EXECUTE effect lane (`tool_effects.py`, the real run's id and node) and on the home-direct chat lane
  (`chat.py`: visits driven by `ChatSession` and personal time, where the stand-in `chat-<session>` run id and the turn
  id, which is the tick id, name the origin). A lane that passes no store behaves as before (no artifact, no line).
