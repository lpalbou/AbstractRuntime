"""browser_probe's `target` arg rides the workspace wall (core c4872 note 1).

Core shipped browser_probe with a local-file target arg named `target` — a
new spelling the argument rewriter did not cover, so local-file probes
bypassed `resolve_user_path` entirely. Pinned here: bare paths and file://
URLs wall exactly like read_file's file_path (containment refusals
included); http(s) targets pass untouched (egress is the approval lane's
jurisdiction, not the wall's).
"""

from pathlib import Path

import pytest

from abstractruntime.integrations.abstractcore.workspace_scoped_tools import (
    WorkspaceScope,
    rewrite_tool_arguments,
)


def _scope(tmp_path: Path) -> WorkspaceScope:
    ws = tmp_path / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    scope = WorkspaceScope.from_input_data({"workspace_root": str(ws)})
    assert scope is not None
    return scope


def test_bare_path_targets_are_walled(tmp_path: Path) -> None:
    scope = _scope(tmp_path)
    (tmp_path / "ws" / "page.html").write_text("<p>hi</p>", encoding="utf-8")
    out = rewrite_tool_arguments(
        tool_name="browser_probe", args={"target": "page.html"}, scope=scope
    )
    assert out["target"] == str((tmp_path / "ws" / "page.html").resolve())


def test_outside_path_targets_refuse(tmp_path: Path) -> None:
    scope = _scope(tmp_path)
    outside = tmp_path / "secret.html"
    outside.write_text("<p>secret</p>", encoding="utf-8")
    with pytest.raises(ValueError):
        rewrite_tool_arguments(
            tool_name="browser_probe", args={"target": str(outside)}, scope=scope
        )


def test_file_url_targets_wall_the_inner_path(tmp_path: Path) -> None:
    scope = _scope(tmp_path)
    (tmp_path / "ws" / "app.html").write_text("<p>app</p>", encoding="utf-8")
    inner = str((tmp_path / "ws" / "app.html"))
    out = rewrite_tool_arguments(
        tool_name="browser_probe", args={"target": f"file://{inner}"}, scope=scope
    )
    assert out["target"] == f"file://{(tmp_path / 'ws' / 'app.html').resolve()}"

    outside = tmp_path / "leak.html"
    outside.write_text("<p>leak</p>", encoding="utf-8")
    with pytest.raises(ValueError):
        rewrite_tool_arguments(
            tool_name="browser_probe", args={"target": f"file://{outside}"}, scope=scope
        )


def test_http_targets_pass_untouched(tmp_path: Path) -> None:
    scope = _scope(tmp_path)
    for url in ("http://localhost:3000/", "https://example.org/page"):
        out = rewrite_tool_arguments(
            tool_name="browser_probe", args={"target": url}, scope=scope
        )
        assert out["target"] == url
