### Security

- **A skill folder is served only under a root the host registered** (`integrations/abstractcore/skill_roots.py`,
  `effect_handlers.py`; review 2026-11-09 finding 1, BLOCKER). The `read_file` interception that serves an activated
  skill's files took its folders from run variables a client writes (`skills` mapping roots, `_runtime.skills_resolution`
  and the `<path>` of `_runtime.skills_block`), so any client could read every file beside any `SKILL.md` on the machine
  (another account's imported shelf, `~/.claude/skills/*`). The folders now come from the host: it registers its skill
  roots once per process with `register_skill_roots([...])` (`registered_skill_roots()`, `clear_skill_roots()`), and a
  folder is served only as `<root>/<name>` or `<root>/skills/<name>` of a registered root, for a skill NAME the run
  selected (`_runtime.skills_resolution.active`, else the `skills` names, plus the `<name>`s of `_runtime.skills_block`).
  With no root registered nothing is served and every path takes the normal `read_file` path (fail closed). A host
  other than AbstractGateway that relied on the interception must now register its roots.
