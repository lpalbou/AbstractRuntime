from __future__ import annotations

import re

from abstractruntime.memory.active_memory import (
    MEMORY_BLUEPRINTS_MD,
    apply_memact_envelope,
    ensure_memact_memory,
    get_memact_memory,
    render_memact_system_prompt,
)


def test_ensure_memact_memory_initializes_schema() -> None:
    vars: dict = {}
    mem = ensure_memact_memory(vars, now_iso=lambda: "2026-01-05T06:00:00+00:00")

    assert mem.get("version") == 1
    assert isinstance(mem.get("persona"), str) and mem["persona"].strip()
    assert isinstance(mem.get("limits"), dict)
    assert mem.get("created_at") == "2026-01-05T06:00:00+00:00"

    for key in (
        "relationships",
        "current_tasks",
        "current_context",
        "critical_insights",
        "references",
        "history",
    ):
        assert isinstance(mem.get(key), list)


def test_ensure_memact_memory_coerces_string_lists_to_records() -> None:
    vars: dict = {
        "_runtime": {
            "active_memory": {
                "version": 1,
                "persona": "p",
                "current_tasks": ["Do the thing"],
            }
        }
    }

    mem = ensure_memact_memory(vars, now_iso=lambda: "2026-01-05T06:00:00+00:00")
    tasks = mem.get("current_tasks")
    assert isinstance(tasks, list) and len(tasks) == 1
    rec = tasks[0]
    assert isinstance(rec, dict)
    assert isinstance(rec.get("id"), str) and rec["id"]
    assert rec.get("text") == "Do the thing"
    assert rec.get("created_at") == "2026-01-05T06:00:00+00:00"
    assert isinstance(rec.get("ts"), str) and re.match(r"^\d{2}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}$", rec["ts"])


def test_apply_memact_envelope_add_remove_and_history_append_only() -> None:
    vars: dict = {}

    def now_iso() -> str:
        return "2026-01-05T06:00:00+00:00"

    def now_compact() -> str:
        return "26/01/05 06:00:00"

    out = apply_memact_envelope(
        vars,
        envelope={
            "content": "ignored by runtime memory apply",
            "relationships": {"added": ["Alice prefers concise summaries"], "removed": []},
            "current_tasks": {"added": ["Fix MemAct memory persistence"], "removed": []},
            "current_context": {"added": ["Repo: /Users/albou/abstractframework"], "removed": []},
            "critical_insights": {"added": ["Avoid duplicating user turns in provider payloads"], "removed": []},
            "references": {"added": ["docs/architecture.md (macro architecture overview)"], "removed": []},
            "history": {"added": ["Implemented MemAct agent scaffolding"], "removed": ["should be ignored"]},
        },
        now_iso=now_iso,
        now_compact=now_compact,
    )
    assert out.get("ok") is True

    mem = get_memact_memory(vars)
    assert isinstance(mem.get("updated_at"), str) and mem["updated_at"]
    assert len(mem.get("relationships") or []) == 1
    assert len(mem.get("current_tasks") or []) == 1
    assert len(mem.get("current_context") or []) == 1
    assert len(mem.get("critical_insights") or []) == 1
    assert len(mem.get("references") or []) == 1
    assert len(mem.get("history") or []) == 1

    # De-dupe: adding the exact same statement again should not create duplicates.
    _ = apply_memact_envelope(
        vars,
        envelope={
            "content": "",
            "relationships": {"added": ["Alice prefers concise summaries"], "removed": []},
            "current_tasks": {"added": ["Fix MemAct memory persistence"], "removed": []},
            "current_context": {"added": [], "removed": []},
            "critical_insights": {"added": [], "removed": []},
            "references": {"added": [], "removed": []},
            "history": {"added": ["Implemented MemAct agent scaffolding"]},
        },
        now_iso=now_iso,
        now_compact=now_compact,
    )
    mem2 = get_memact_memory(vars)
    assert len(mem2.get("relationships") or []) == 1
    assert len(mem2.get("current_tasks") or []) == 1
    assert len(mem2.get("history") or []) == 1

    # Remove by "rendered" statement (timestamp prefix should be ignored).
    rendered = f"[{(mem2.get('current_tasks') or [{}])[0].get('ts')}] Fix MemAct memory persistence"
    _ = apply_memact_envelope(
        vars,
        envelope={
            "content": "",
            "relationships": {"added": [], "removed": []},
            "current_tasks": {"added": [], "removed": [rendered]},
            "current_context": {"added": [], "removed": []},
            "critical_insights": {"added": [], "removed": []},
            "references": {"added": [], "removed": []},
            "history": {"added": []},
        },
        now_iso=now_iso,
        now_compact=now_compact,
    )
    mem3 = get_memact_memory(vars)
    assert len(mem3.get("current_tasks") or []) == 0

    # History is append-only: removals are ignored.
    _ = apply_memact_envelope(
        vars,
        envelope={
            "content": "",
            "relationships": {"added": [], "removed": []},
            "current_tasks": {"added": [], "removed": []},
            "current_context": {"added": [], "removed": []},
            "critical_insights": {"added": [], "removed": []},
            "references": {"added": [], "removed": []},
            "history": {"added": [], "removed": ["Implemented MemAct agent scaffolding"]},
        },
        now_iso=now_iso,
        now_compact=now_compact,
    )
    mem4 = get_memact_memory(vars)
    assert len(mem4.get("history") or []) == 1


def test_render_memact_system_prompt_contains_blueprints_and_modules() -> None:
    vars: dict = {}
    ensure_memact_memory(vars, now_iso=lambda: "2026-01-05T06:00:00+00:00")

    rendered = render_memact_system_prompt(vars)
    assert MEMORY_BLUEPRINTS_MD.strip() in rendered
    assert "## MY PERSONA" in rendered
    assert "## CURRENT TASKS" in rendered
    assert "## HISTORY" in rendered
    assert rendered.find("## MEMORY BLUEPRINTS") < rendered.find("## MY PERSONA")
