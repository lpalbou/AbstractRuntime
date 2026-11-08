### Fixed

- **An activated skill is always readable** (`integrations/abstractcore/skill_roots.py`, `effect_handlers.py`;
  operator 2026-11-08: an agent in Code, with a skill activated for its session, called `read_file` on the skill's
  `SKILL.md` and got `Error: File '…' is ignored by .abstractignore policy`, because AbstractCore's file tool applies
  the workspace `.abstractignore`, which prunes the gateway data folder where the shelf lives). A `read_file` whose
  path lies inside one of the run's own skill folders is now answered by the runtime itself — the whole file, UTF-8,
  in the same line-numbered shape AbstractCore renders, `start_line`/`end_line` honoured — before AbstractCore's tool
  runs, so neither the workspace `.abstractignore` nor the workspace scope can refuse a skill the model was told
  about. The folders come from the run's own record, never from the model's words: `skills` as the workforce payload
  (`shelf`, `imported_shelf`, `roots`, `ABSTRACTCODE_SKILLS_ROOTS`, with the `active` names), `_runtime.skills_resolution`
  (`shelf`/`imported_shelf`/`roots` when the host records them, with its `active` names), and the `<path>` each skill
  carries in `_runtime.skills_block` (AbstractSkill 0.3.11 renders the SKILL.md it loaded); a folder counts as
  `<root>/<name>` or `<root>/skills/<name>` only when it holds a `SKILL.md`. The requested path must lie under a
  folder lexically and, once resolved with `realpath`, still under that folder's real path: a symlink leading out is
  refused with a typed error, never served and never handed to AbstractCore. Any other path reaches AbstractCore
  exactly as before.
