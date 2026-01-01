from __future__ import annotations

from abstractruntime.memory.active_memory import (
    ensure_active_memory,
    compute_active_memory_token_breakdown,
    get_active_memory,
    render_active_memory_for_prompt,
    upsert_current_context_item,
    upsert_task,
    add_critical_insight,
    add_key_history_event,
    apply_active_memory_delta,
)


def test_ensure_active_memory_initializes_schema() -> None:
    vars: dict = {}
    mem = ensure_active_memory(vars, now_iso=lambda: "2025-01-01T00:00:00+00:00")
    assert mem.get("version") == 7
    # Token-native budgeting (0 => auto from vars["_limits"].max_tokens when present).
    assert isinstance(mem.get("max_tokens"), int)
    assert isinstance(mem.get("persona_md"), str) and mem["persona_md"].strip()
    assert isinstance(mem.get("memory_organization_md"), str) and mem["memory_organization_md"].strip()
    assert isinstance(mem.get("tasks"), list)
    assert isinstance(mem.get("current_context"), list)
    assert isinstance(mem.get("critical_insights"), list)
    assert isinstance(mem.get("key_history"), list)
    budgets = mem.get("budgets")
    assert isinstance(budgets, dict)
    for key in (
        "persona_pct",
        "memory_organization_pct",
        "current_tasks_pct",
        "current_context_pct",
        "tools_pct",
        "critical_insights_pct",
        "key_history_pct",
    ):
        assert key in budgets

    sys_ids = mem.get("system_component_ids")
    assert isinstance(sys_ids, list)
    assert "persona" in sys_ids and "tools" in sys_ids


def test_ensure_active_memory_migrates_v3_default_budgets() -> None:
    vars: dict = {
        "_runtime": {
            "active_memory": {
                "version": 3,
                "budgets": {
                    "persona_pct": 0.10,
                    "memory_organization_pct": 0.07,
                    "tools_pct": 0.10,
                },
            }
        }
    }
    mem = ensure_active_memory(vars, now_iso=lambda: "2025-01-01T00:00:00+00:00")
    assert mem.get("version") == 7
    budgets = mem.get("budgets")
    assert isinstance(budgets, dict)
    assert budgets.get("persona_pct") == 0.075
    assert budgets.get("memory_organization_pct") == 0.08
    assert budgets.get("tools_pct") == 0.115


def test_upsert_task_preserves_created_at_and_caps_to_five() -> None:
    vars: dict = {}

    times = [
        "2025-01-01T00:00:00+00:00",
        "2025-01-01T00:00:01+00:00",
        "2025-01-01T00:00:02+00:00",
        "2025-01-01T00:00:03+00:00",
        "2025-01-01T00:00:04+00:00",
        "2025-01-01T00:00:05+00:00",
        "2025-01-01T00:00:06+00:00",
        "2025-01-01T00:00:07+00:00",
    ]

    def now_iso() -> str:
        return times.pop(0)

    task_ids = []
    for i in range(6):
        task_ids.append(upsert_task(vars, title=f"t{i}", now_iso=now_iso))

    mem = get_active_memory(vars)
    assert isinstance(mem.get("tasks"), list)
    assert len(mem["tasks"]) == 5  # capped

    # Updating an existing task keeps created_at.
    created_at_before = None
    for t in mem["tasks"]:
        if isinstance(t, dict) and t.get("task_id") == task_ids[-1]:
            created_at_before = t.get("created_at")
            break
    assert isinstance(created_at_before, str) and created_at_before

    upsert_task(vars, task_id=task_ids[-1], title="updated", now_iso=now_iso)
    mem2 = get_active_memory(vars)
    created_at_after = None
    updated_at_after = None
    for t in mem2["tasks"]:
        if isinstance(t, dict) and t.get("task_id") == task_ids[-1]:
            created_at_after = t.get("created_at")
            updated_at_after = t.get("updated_at")
            break
    assert created_at_after == created_at_before
    assert updated_at_after != created_at_before


def test_upsert_current_context_item_preserves_created_at() -> None:
    vars: dict = {}

    times = [
        "2025-01-01T00:00:00+00:00",
        "2025-01-01T00:00:01+00:00",
        "2025-01-01T00:00:02+00:00",
    ]

    def now_iso() -> str:
        return times.pop(0)

    cid = upsert_current_context_item(
        vars,
        title="doc",
        summary="s1",
        refs=[{"kind": "file", "path": "docs/x.md"}],
        now_iso=now_iso,
    )
    mem = get_active_memory(vars)
    assert isinstance(mem.get("current_context"), list) and mem["current_context"]
    created = mem["current_context"][0].get("created_at")

    upsert_current_context_item(
        vars,
        context_id=cid,
        title="doc",
        summary="s2",
        refs=[{"kind": "file", "path": "docs/y.md"}],
        now_iso=now_iso,
    )
    mem2 = get_active_memory(vars)
    item = next(x for x in mem2["current_context"] if isinstance(x, dict) and x.get("context_id") == cid)
    assert item.get("created_at") == created
    assert item.get("updated_at") != created


def test_render_active_memory_for_prompt_is_bounded_and_includes_sections() -> None:
    vars: dict = {"_runtime": {"toolset_id": "ts_test", "tool_specs": [{"name": "read_file", "description": "d", "parameters": {"file_path": {"type": "string"}, "encoding": {"type": "string", "default": "utf-8"}}}]}}
    mem = ensure_active_memory(vars, now_iso=lambda: "2025-01-01T00:00:00+00:00")
    mem["max_tokens"] = 300

    upsert_task(
        vars,
        title="Implement active memory",
        purpose="keep context stable",
        status="doing",
        done=["scaffold"],
        next_steps=["tests"],
        issues=["none"],
        now_iso=lambda: "2025-01-01T00:00:01+00:00",
    )
    upsert_current_context_item(
        vars,
        title="ADR-0007",
        summary="active vs stored memory",
        refs=[{"kind": "file", "path": "docs/adr/0007-active-context-and-memory-provenance.md"}],
        now_iso=lambda: "2025-01-01T00:00:02+00:00",
    )
    add_critical_insight(vars, text="Prefer artifact refs over truncation", now_iso=lambda: "2025-01-01T00:00:03+00:00")
    add_key_history_event(
        vars,
        kind="decision",
        summary="Adopt runtime-owned active memory blocks",
        refs=[{"kind": "backlog", "id": 153}],
        ts="2025-01-01T00:00:04+00:00",
        now_iso=lambda: "2025-01-01T00:00:04+00:00",
    )

    rendered = render_active_memory_for_prompt(vars, include_tools_summary=True, max_tokens=300, token_counter=lambda s: len(str(s).split()))
    assert len(str(rendered).split()) <= 300
    assert "## Persona (persistent)" in rendered
    assert "## Memory Organization (persistent)" in rendered
    assert "## Tools (session)" in rendered
    assert "## Current Tasks (evolving)" in rendered
    assert "## Current Context (dynamic)" in rendered


def test_compute_active_memory_token_breakdown_allocates_total_budget() -> None:
    vars: dict = {"_runtime": {"tool_specs": [{"name": "read_file", "description": "d", "parameters": {"file_path": {"type": "string"}}}]}}
    mem = ensure_active_memory(vars, now_iso=lambda: "2025-01-01T00:00:00+00:00")
    mem["max_tokens"] = 100
    mem["persona_md"] = "persona"
    mem["memory_organization_md"] = "org"

    upsert_task(vars, title="t", now_iso=lambda: "2025-01-01T00:00:01+00:00")

    out = compute_active_memory_token_breakdown(vars, token_counter=lambda s: len(str(s).split()))
    assert out.get("active_memory_max_tokens") == 100
    components = out.get("components")
    assert isinstance(components, dict)
    assert sum(int(c.get("max_tokens") or 0) for c in components.values() if isinstance(c, dict)) == 100
    assert int(components["persona"]["used_tokens"]) > 0


def test_apply_active_memory_delta_updates_evolving_modules() -> None:
    vars: dict = {"_runtime": {}}
    ensure_active_memory(vars, now_iso=lambda: "2025-01-01T00:00:00+00:00")

    delta = {
        "current_tasks": {
            "upsert": [
                {"task_id": "t_1", "title": "Do thing", "status": "doing", "next": ["run tests"]},
            ]
        },
        "current_context": {
            "upsert": [
                {"context_id": "c_1", "title": "Foo", "summary": "Bar", "refs": [{"kind": "file", "path": "x.py"}]},
            ]
        },
        "critical_insights": {"add": [{"insight_id": "i_1", "text": "Beware", "tags": ["pitfall"]}]},
        "key_history": {"add": [{"event_id": "h_1", "kind": "decision", "summary": "Chose X"}]},
    }

    out = apply_active_memory_delta(vars, delta=delta, now_iso=lambda: "2025-01-01T00:00:01+00:00")
    assert out.get("ok") is True

    mem = get_active_memory(vars)
    assert any(isinstance(t, dict) and t.get("task_id") == "t_1" for t in mem.get("tasks") or [])
    assert any(isinstance(c, dict) and c.get("context_id") == "c_1" for c in mem.get("current_context") or [])
    assert any(isinstance(i, dict) and i.get("insight_id") == "i_1" for i in mem.get("critical_insights") or [])
    assert any(isinstance(h, dict) and h.get("event_id") == "h_1" for h in mem.get("key_history") or [])


def test_apply_active_memory_delta_accepts_string_shorthand_for_common_ops() -> None:
    vars: dict = {"_runtime": {}}
    ensure_active_memory(vars, now_iso=lambda: "2025-01-01T00:00:00+00:00")

    delta = {
        "current_tasks": {"upsert": ["Do thing"]},
        "current_context": {"upsert": ["Reviewed work-rtype/main.py"]},
        "key_history": {"add": ["Decision: keep tools prompt architecture-specific"]},
    }

    out = apply_active_memory_delta(vars, delta=delta, now_iso=lambda: "2025-01-01T00:00:01+00:00")
    assert out.get("ok") is True

    mem = get_active_memory(vars)
    assert any(isinstance(t, dict) and t.get("title") == "Do thing" for t in mem.get("tasks") or [])
    assert any(isinstance(c, dict) and c.get("title") == "Reviewed work-rtype/main.py" for c in mem.get("current_context") or [])
    assert any(isinstance(h, dict) and h.get("summary") == "Decision: keep tools prompt architecture-specific" for h in mem.get("key_history") or [])
