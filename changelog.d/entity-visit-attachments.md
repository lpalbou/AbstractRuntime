### Added

- **An image attached in an entity visit reaches the entity's model as an image** (`identity/visit_workflow.py`,
  `identity/tools.py`; operator 2026-11-08: images already work with every other client, so they must work for entities
  too). A visit turn's resume payload may carry `attachments`: a structured list, one record per file the visitor
  attached, with a workspace-relative `path` (and whatever the door recorded: `bytes`, `sha256`, `content_type`).
  Attachments are never parsed from the visitor's words. The runtime keeps the records whole on the turn
  (`_turn.attachments`), adds one honest line per file to the rendered message — for an image
  `(attached: shared/x.png — a PNG image, 1,204,113 bytes; shown to you with this message; read_file re-opens it)`,
  for anything else `(attached: shared/notes.pdf — a PDF document, 88,201 bytes; read_file opens it)`, for a file that
  cannot be opened `(attached: … — could not be opened: <why>)` — while `turn.text` keeps the human's raw words. This
  turn's images (PNG, JPEG, GIF, WEBP by magic, within the workspace cap) are shown to the model on every call of the
  turn: on the react arm as `context.attachments` (absolute paths, cleared at HARVEST, so the next turn does not re-send
  them — a re-look is `read_file`), on the home-direct arm as the LLM_CALL payload's `media`. The appended user message
  carries `metadata.attachments` and the durable visit history entry carries `attachments`, so the transcript keeps the
  reference (path, bytes, sha256, type); the file itself stays in the workspace. A turn with files and no words is a
  turn: `turn.text` stays empty, the rendered message is the decoration alone (presence, memories, the attachment
  lines), the recall cue and the episode's verbatim name what was shown; a turn with neither words nor files still
  parks without a turn.
- **`tool_result_media(election, workspace)`** (`identity/tools.py`): the media one executed tool election shows the
  model — absolute file paths, or None when the result is text only. Declared beside the executor
  (`ToolDescriptor.result_media`); today `read_file` on an image declares its path, re-resolved and re-sniffed under
  the same containment and size cap as the read, never derived from the result text. A host that serves tool results
  as `{"success": True, "output": …}` makes `output` the dict abstractagent's sight lane reads,
  `{"rendered": <the result text>, "media": [<absolute path>]}`, and the image rides the next model call.
- **`sniff_image(data)`** (`identity/tools.py`): `(label, content_type)` for a PNG/JPEG/GIF/WEBP by magic, else None.

### Changed

- **`WorkspaceRoot.read_file` on an image shows it instead of refusing it.** The result is a label,
  `--- shared/x.png (1,204,113 bytes) ---` then `[#IMAGE: a PNG image; it is attached to your next look as an image]`;
  no image byte ever enters the prompt as text (the 2026-08-01 wall stands). Other binaries are still refused with the
  `#BINARY` notice, which no longer claims "You have no tool that can view images or other binary content"; it says
  what is true: images are shown when read, other binary content is not readable here. WEBP (`RIFF….WEBP`) is now
  labelled "a WEBP image" instead of "a RIFF media container".
